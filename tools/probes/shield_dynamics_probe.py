"""Замер динамики метапластического щита на РЕАЛЬНОЙ волне обучения.

Загружает обученного кандидата production-загрузчиком, прогоняет настоящую
волну (реальные предложения → реальная токенизация → реальная
нейромедиаторная награда) и измеряет:

  * состояние трассы сразу после загрузки (миграция legacy-семантики);
  * распределение затухания 1/(1+κ·B) после волны;
  * долю замороженных строк (attenuation < 1e-3) — базовая патология была
    34.5 % на матрице величины;
  * СВЯЗЬ затухания с |w|: старая семантика замораживала крупное
    (медианный |w| замороженных 1.68× выше живых), новая должна замораживать
    по ИСПОЛЬЗОВАНИЮ, а не по величине веса;
  * покрытие: сколько строк реально участвовало в обновлениях.

Запуск (через лимитер памяти!):
    python tools/limited_run.py --ram-free-pct 85 --log D:\\...\\log.txt ^
        -- python tools/probes/shield_dynamics_probe.py <checkpoint.pt> [--pairs 12]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Реальные предложения (не шаблоны): обычные вопросы и ответы из
# повседневной речи. Нужен РЕАЛЬНЫЙ путь обучения, а не синтетика.
SENTENCES = [
    "I am making mayonnaise, it was starting to thicken but now it has become runny and liquid.",
    "How can I fix errors in my grammar when writing an essay?",
    "Hi, how are you doing? I am getting ready to go running to stay in shape.",
    "The museum closes at six in the evening on weekdays.",
    "She poured the coffee into a ceramic mug and sat by the window.",
    "Water freezes at zero degrees Celsius and boils at one hundred.",
    "My brother bought a second-hand bicycle last weekend.",
    "The train to the airport leaves every twenty minutes.",
    "Reading before bed helps me fall asleep faster.",
    "They painted the fence green because the old colour had faded.",
    "A good cup of coffee starts with freshly ground beans.",
    "The library lends books for two weeks at a time.",
]


def _rank_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Спирменова корреляция по рангам (без scipy)."""
    if x.numel() < 3:
        return 0.0
    rx = torch.argsort(torch.argsort(x)).float()
    ry = torch.argsort(torch.argsort(y)).float()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.norm() * ry.norm()
    return float((rx * ry).sum() / denom) if float(denom) > 0 else 0.0


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--runtime-version", type=int, default=19, choices=(19, 20))
    ap.add_argument("--fresh", action="store_true",
                    help="не грузить чекпоинт: свежий организм (пик ~3.7 ГБ) — "
                         "нужно, когда полное восстановление кандидата не влезает "
                         "в лимит commit на этой машине")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from ziwago_assistant import ZiwagoBiologicalSNNAssistant
    from ziwago.app.data import TokenizedDialogPair

    t0 = time.perf_counter()
    assistant = ZiwagoBiologicalSNNAssistant(
        root=ROOT, checkpoint_path=None if args.fresh else Path(args.checkpoint),
        workers=2, cpu_threads=4, seed=42, runtime_version=args.runtime_version)
    organism = assistant.organism
    shield = assistant.metaplasticity_shield
    readout = organism.dendritic_plateau_readout
    load_sec = time.perf_counter() - t0

    report: dict = {
        "checkpoint": None if args.fresh else str(Path(args.checkpoint).resolve()),
        "mode": "fresh" if args.fresh else "checkpoint",
        "runtime_version": args.runtime_version,
        "load_sec": round(load_sec, 2),
        "semantics": shield.u_slow_semantics,
        "trace_layout": list(shield.u_slow.shape),
        "legacy_trace_dropped": getattr(organism, "legacy_shield_trace_dropped", None),
        "after_load": shield.stats(),
        "migration": dict(getattr(shield, "last_migration", {}) or {}),
    }

    # |w| по строкам — по колонкам, где вес реально есть (маску берём заново
    # после волны: срез до обучения — это копия, и переиспользовать её нельзя).
    def _row_absw():
        mask = readout.w_dend_out.abs().sum(dim=0) > 0
        if not bool(mask.any()):
            return torch.zeros(readout.d_dend), 0
        used = readout.w_dend_out[:, mask]
        return used.abs().mean(dim=1), int(mask.sum())

    w_before, n_before = _row_absw()

    pairs = []
    for sentence in SENTENCES[: args.pairs]:
        ids = organism.encode_text(sentence)
        pairs.append(TokenizedDialogPair("probe", ids, ids, quality_reward_hint=2.0))

    t1 = time.perf_counter()
    wave = organism.train_wave_with_neuromodulator_reward(pairs, wave_idx=1)
    wave_sec = time.perf_counter() - t1

    w_after, n_active = _row_absw()
    trace = shield.u_slow.detach().clone()
    attenuation = 1.0 / (1.0 + shield.kappa_shield * trace)

    touched = trace > 0
    frozen = attenuation < 1e-3
    protected = attenuation < 1e-2

    report["wave"] = {
        "pairs": len(pairs),
        "tokens": sum(len(p.stimulus_ids) for p in pairs),
        "wave_sec": round(wave_sec, 2),
        "stats": {k: wave.get(k) for k in (
            "pairs", "tokens_trained", "tokens", "readout_token_updates",
            "protected_core_token_updates", "dendritic_tokens_trained",
            "readout_step_limited", "mean_plasticity_gate_f_mod")},
    }
    report["after_wave"] = shield.stats()
    report["coverage"] = {
        "rows_total": int(trace.numel()),
        "rows_touched": int(touched.sum()),
        "rows_touched_fraction": float(touched.float().mean()),
        "rows_frozen_lt_1e3": int(frozen.sum()),
        "rows_frozen_fraction": float(frozen.float().mean()),
        "rows_protected_lt_1e2": int(protected.sum()),
        "attenuation_min": float(attenuation.min()),
        "attenuation_p01": float(attenuation.kthvalue(max(1, int(0.01 * attenuation.numel()))).values),
        "attenuation_median": float(attenuation.median()),
        "attenuation_max": float(attenuation.max()),
        "trace_max": float(trace.max()),
        "trace_mean_touched": float(trace[touched].mean()) if int(touched.sum()) else 0.0,
    }
    # СВЯЗЬ затухания с величиной веса: старая семантика замораживала крупное.
    report["weight_link"] = {
        "spearman_attenuation_vs_absw": _rank_correlation(attenuation, w_after),
        "mean_absw_touched": float(w_after[touched].mean()) if int(touched.sum()) else None,
        "mean_absw_untouched": float(w_after[~touched].mean()) if int((~touched).sum()) else None,
        "median_absw_touched": float(w_after[touched].median()) if int(touched.sum()) else None,
        "median_absw_untouched": float(w_after[~touched].median()) if int((~touched).sum()) else None,
        "absw_change_mean": float((w_after - w_before).mean()),
    }
    report["memory"] = {
        "trace_bytes": int(trace.numel() * trace.element_size()),
        "legacy_trace_bytes": int(trace.numel() * 4 * max(1, n_active)),
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"записано: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
