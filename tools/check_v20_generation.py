"""Проверка совместимости чекпоинта и языковой генерации на архитектуре v20.

Что проверяем (без обрезок и искусственных лимитов):
  1. Совместимость: грузится ли существующий чекпоинт (v19-артефакт) в рантайм v20,
     что при этом происходит с состоянием химии и весами коры.
  2. Генерация: один и тот же чекпоинт, один и тот же набор вопросов, два рантайма
     (v19 и v20). Полное окно токенов, без арбитража конфликта и без каких-либо
     дополнительных ограничений — говорит система как есть.
  3. Критерий: совпадает ли ответ с выученным (точное совпадение по эталону
     обучающей выборки) и останавливает ли модель сама (выученный EOS).
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\yarma\Desktop\Ziwago18")
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from ziwago_assistant import ZiwagoBiologicalSNNAssistant  # noqa: E402

CKPT = Path(r"D:\Ziwago18-storage\Produce\checkpoints\ziwago_v19_bankfixed_20261109.pt")
CURRICULUM = Path(r"D:\Ziwago18-storage\data\hf\squad\language_v19_v6_union\train.jsonl")
QUESTIONS = 6


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def load_rows() -> list[dict]:
    return [json.loads(line) for line in CURRICULUM.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_version(version: int, rows: list[dict]) -> dict:
    started = time.perf_counter()
    assistant = ZiwagoBiologicalSNNAssistant(root=ROOT, checkpoint_path=CKPT, workers=1,
                                            cpu_threads=6, seed=20261109, runtime_version=version)
    core = assistant.organism
    ro = core.dendritic_plateau_readout
    info = assistant.loaded_checkpoint_info or {}
    chem = core.chemistry
    state = chem.s
    loaded = {
        "runtime_version": version,
        "loaded_ok": bool(info),
        "schema": info.get("schema"),
        "certification": info.get("certification"),
        "checkpoint_sha256": (info.get("checkpoint_sha256") or "")[:16],
        "active_vocab": len(ro.col_to_tok),
        "w_dend_out_abs_max": round(float(ro.w_dend_out.abs().max()), 6),
        "w_dend_out_norm": round(float(ro.w_dend_out.norm()), 4),
        "chemistry_class": type(chem).__name__,
        "chemistry_state": {"da": round(float(state.da), 4), "ne": round(float(state.ne), 4),
                            "ach": round(float(state.ach), 4), "ht": round(float(state.ht), 4),
                            "ado": round(float(state.ado), 6), "ox": round(float(state.ox), 4),
                            "cort": round(float(state.cort), 4)},
        "awake": bool(getattr(chem, "is_awake", True)),
        "integrated_life": hasattr(core, "life"),
        "load_seconds": round(time.perf_counter() - started, 1),
    }
    emit({"stage": "loaded", **loaded})

    window = int(ro.max_pos)
    results = []
    exact = 0
    stopped = 0
    for index, row in enumerate(rows[:QUESTIONS]):
        prompt = str(row["prompt"])
        reference = str(row["response"])
        t0 = time.perf_counter()
        try:
            answer = assistant.chat(prompt, max_tokens=window)
            text = str(answer.get("pure_weight_generated_response", ""))
            kind = str(answer.get("generation_kind", ""))
            finish = str(answer.get("finish_reason", ""))
            action = str(answer.get("basal_ganglia_selected_action", ""))
        except Exception as exc:                    # noqa: BLE001
            text, kind, finish, action = f"<EXCEPTION {type(exc).__name__}: {exc}>", "error", "error", ""
        hit = text.strip() == reference.strip()
        exact += int(hit)
        stopped += int(bool(answer.get("stopped_by_learned_eos")))
        results.append({
            "index": index, "prompt": prompt, "reference": reference,
            "answer": text, "exact_match": hit, "generation_kind": kind,
            "finish_reason": finish, "basal_ganglia": action,
            "seconds": round(time.perf_counter() - t0, 1),
        })
        emit({"stage": "generation", "runtime_version": version, "index": index,
              "answer": text[:220], "exact_match": hit, "kind": kind, "finish": finish,
              "action": action, "seconds": round(time.perf_counter() - t0, 1)})
    summary = {"runtime_version": version, "cases": len(results), "exact_matches": exact,
               "learned_eos_stops": stopped, "window_tokens": window,
               "total_seconds": round(time.perf_counter() - started, 1),
               "answers": results}
    del assistant, core, ro
    gc.collect()
    return summary


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", default="19,20",
                        help="Какие рантаймы проверять: 19,20 или 20")
    args = parser.parse_args()
    wanted = [int(part) for part in args.versions.split(",") if part.strip()]
    rows = load_rows()
    emit({"stage": "start", "checkpoint": str(CKPT), "questions": QUESTIONS,
          "rows": len(rows), "versions": wanted,
          "free_ram_note": "generation без арбитража и без обрезки"})
    summaries = []
    for version in wanted:
        summary = run_version(version, rows)
        emit({"stage": "summary", **summary})
        summaries.append(summary)
    emit({"stage": "verdict",
          **{f"v{summary['runtime_version']}_exact":
             f"{summary['exact_matches']}/{summary['cases']}" for summary in summaries},
          **{f"v{summary['runtime_version']}_eos_stops":
             summary["learned_eos_stops"] for summary in summaries},
          **{f"v{summary['runtime_version']}_loaded": summary["loaded_ok"] for summary in summaries},
          **{f"v{summary['runtime_version']}_schema": summary["schema"] for summary in summaries},
          **{f"v{summary['runtime_version']}_chemistry": summary["chemistry_class"] for summary in summaries},
          **{f"v{summary['runtime_version']}_awake": summary["awake"] for summary in summaries}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
