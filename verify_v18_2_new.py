"""Проверка НОВЫХ умений v18.2+: BA10 контрфактические вмешательства,
MD-таламус (роутер речевых актов), эмерджентная онтология → ролевой биндер."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

from ziwago.cortex.genesis_engine import (
    BA10CounterfactualInterventionSimulator,
    MDThalamusSynapticSpeechActRouter,
)
from ziwago_assistant import ZiwagoBiologicalSNNAssistant

OUT = ROOT / "Produce" / "v18_2_new_features.json"

COUNTERFACTUALS = [
    "What would happen if the hippocampus were damaged?",
    "If the water were boiling, what would happen to the steam?",
    "Imagine the moon did not rotate . What would change ?",
    "What if the lysosome membrane broke and enzymes leaked out ?",
]

ORDINARY = [
    "What is my name?",
    "How do astrocytes ignite dendritic plateau potentials in pyramidal neurons?",
    "Why do penguins in Antarctica huddle so tightly?",
    "Could you summarize what topics we have talked about in this chat?",
]


def main() -> None:
    print("=" * 100)
    print("1. КОД: какие речевые акты и как определяется контрфакт")
    print("=" * 100)
    src = (ROOT / "ziwago" / "cortex" / "genesis_engine.py").read_text(encoding="utf-8")
    acts = sorted(set(re.findall(r"ACT_[A-Z0-9_]+", src)))
    print(f"  речевых актов: {len(acts)}")
    for a in acts[:18]:
        print("   ", a)
    m = re.search(r"def is_counterfactual_query.*?(?=\n    def |\nclass |\Z)", src, re.S)
    if m:
        body = m.group(0)[:900]
        for ln in body.splitlines():
            if "if" in ln or "return" in ln or "any(" in ln:
                print("   |", ln.strip()[:110])

    print()
    print("=" * 100)
    print("2. ЖИВОЙ ПРОГОН: контрфактические запросы")
    print("=" * 100)
    a = ZiwagoBiologicalSNNAssistant(root=ROOT, workers=4, cpu_threads=8, seed=12345)
    rows = []
    for q in COUNTERFACTUALS:
        t0 = time.perf_counter()
        r = a.chat(user_utterance=q, max_tokens=56)
        dt = (time.perf_counter() - t0) * 1000
        resp = r.get("pure_weight_generated_response", "")
        pas = r.get("response_passport") or {}
        acts_fired = (pas.get("mechanisms_triggered") or {}) if isinstance(pas, dict) else {}
        print(f"\n  Q: {q}")
        print(f"  A: {resp[:230]}")
        print(f"     действие={r.get('basal_ganglia_selected_action')} | "
              f"акт={pas.get('thalamic_speech_act') if isinstance(pas, dict) else '—'} | "
              f"{dt:.0f} мс | механизмы={json.dumps(acts_fired, ensure_ascii=False)[:190]}")
        rows.append({"q": q, "a": resp, "act": pas.get("thalamic_speech_act") if isinstance(pas, dict) else None,
                     "mechanisms": acts_fired, "ms": round(dt, 1)})

    print()
    print("=" * 100)
    print("3. ОБЫЧНЫЕ ЗАПРОСЫ: какой речевой акт выбирает MD-таламус")
    print("=" * 100)
    for q in ORDINARY:
        r = a.chat(user_utterance=q, max_tokens=48)
        pas = r.get("response_passport") or {}
        print(f"\n  Q: {q}")
        print(f"  A: {r.get('pure_weight_generated_response','')[:190]}")
        if isinstance(pas, dict):
            print(f"     акт={pas.get('thalamic_speech_act')} | зона={pas.get('proportional_gate_zone')} | "
                  f"механизмы={json.dumps(pas.get('mechanisms_triggered'), ensure_ascii=False)[:170]}")

    print()
    print("=" * 100)
    print("4. СИМУЛЯТОР BA10 — прямой вызов (контрфакт без диалога)")
    print("=" * 100)
    try:
        sim = BA10CounterfactualInterventionSimulator()
        print("  API:", [x for x in dir(sim) if not x.startswith("_")][:14])
        for q in ("What would happen if the hippocampus were damaged?",):
            out = sim.simulate_counterfactual_intervention(q) if hasattr(sim, "simulate_counterfactual_intervention") else None
            print(f"  {q}\n   -> {json.dumps(out, ensure_ascii=False, default=str)[:400] if out is not None else 'нет метода'}")
    except Exception as e:
        print(f"  ошибка: {type(e).__name__}: {e}")

    OUT.write_text(json.dumps({"counterfactuals": rows}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print()
    print("сохранено:", OUT)


if __name__ == "__main__":
    main()
