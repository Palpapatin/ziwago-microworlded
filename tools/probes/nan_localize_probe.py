"""Локализация NaN в весах по ходу обучения (покандитно, по парам).

Прогоняет волну обучения по списку реальных предложений и после КАЖДОЙ пары
сканирует матрицы ридеаута на non-finite. Печатает первую пару, на которой
появились NaN, и агрегирует источник по матрицам.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

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


def _scan(readout) -> dict:
    out = {}
    for name in ("w_dend_out", "w_core_dend_out", "w_swr_dend_out"):
        w = getattr(readout, name, None)
        if w is None:
            continue
        finite = torch.isfinite(w)
        bad = int((~finite).sum())
        out[name] = bad
        if bad:
            idx = (~finite).nonzero()
            out[name + "_first"] = (int(idx[0][0]), int(idx[0][1]))
            out[name + "_cols"] = (int(idx[:, 1].min()), int(idx[:, 1].max()))
    return out


def main() -> int:
    from ziwago_assistant import ZiwagoBiologicalSNNAssistant
    from ziwago.app.data import TokenizedDialogPair

    assistant = ZiwagoBiologicalSNNAssistant(root=ROOT, checkpoint_path=None, workers=2,
                                             cpu_threads=4, seed=42, runtime_version=19)
    organism = assistant.organism
    readout = organism.dendritic_plateau_readout
    print("после конструкции:", _scan(readout), flush=True)

    for index, sentence in enumerate(SENTENCES, start=1):
        ids = organism.encode_text(sentence)
        pair = TokenizedDialogPair("probe", ids, ids, quality_reward_hint=2.0)
        wave = organism.train_wave_with_neuromodulator_reward([pair], wave_idx=index)
        scan = _scan(readout)
        bad = {k: v for k, v in scan.items() if not k.endswith("_first") and not k.endswith("_cols")}
        print(f"пара {index:02d} tokens={len(ids)} updates={wave.get('readout_token_updates')} "
              f"step_limited={wave.get('readout_step_limited')} -> {bad}", flush=True)
        if any(v for v in bad.values()):
            print(f"   ПОДРОБНОСТИ: {scan}", flush=True)
            print(f"   wave stats: { {k: wave.get(k) for k in ('readout_token_updates','readout_step_limited','readout_max_effective_gain','readout_step_factor_max','mean_plasticity_gate_f_mod')} }", flush=True)
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
