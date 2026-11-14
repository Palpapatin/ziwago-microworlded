"""Диагностика вырожденного аттрактора в новом мастере v18.1 (1262 МБ).

Наблюдение: пять разных вопросов получают ОДИН И ТОТ ЖЕ ответ
«I know antagonists do not maintain the ability to as a causal agent…».
Проверяем: (а) сколько разных промптов схлопываются в один ответ,
(б) один ли это полюс CA3, (в) что паспорт говорит про grounding.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

from ziwago_assistant import ZiwagoBiologicalSNNAssistant

OUT = ROOT / "Produce" / "v18_2b_attractor_dominance.json"

DIVERSE = [
    "Why do penguins in Antarctica huddle so tightly?",
    "How do grid cells in the hippocampus support spatial navigation?",
    "Explain in detail how the Moon rotates every 27.3 days.",
    "Tell me about the hippocampus and grid cells.",
    "What is the boiling point of water at sea level?",
    "Who wrote the novel Moby-Dick?",
    "What is my name?",
    "How do astrocytes ignite dendritic plateau potentials in pyramidal neurons?",
]


def main() -> None:
    a = ZiwagoBiologicalSNNAssistant(root=ROOT, workers=4, cpu_threads=8, seed=12345)
    rows = []
    print("=" * 100)
    print("ДОМИНИРОВАНИЕ АТТРАКТОРА: 8 максимально разных вопросов")
    print("=" * 100)
    for q in DIVERSE:
        r = a.chat(user_utterance=q, max_tokens=48)
        resp = r.get("pure_weight_generated_response", "")
        pas = r.get("response_passport") or {}
        mech = pas.get("mechanisms_triggered") or {}
        rows.append({"q": q, "resp": resp, "act": r.get("basal_ganglia_selected_action"),
                     "poles": pas.get("selected_ca3_pole_indices"),
                     "sal": pas.get("selected_ca3_pole_saliences"),
                     "grounding": mech.get("son_vte_grounding"),
                     "zone": pas.get("proportional_gate_zone")})
        print(f"\n  Q: {q}")
        print(f"  A: {resp[:150]}")
        print(f"     act={r.get('basal_ganglia_selected_action')} | полюс={pas.get('selected_ca3_pole_indices')} "
              f"| salience={pas.get('selected_ca3_pole_saliences')} | grounding={mech.get('son_vte_grounding')} "
              f"| зона={pas.get('proportional_gate_zone')}")

    uniq = Counter(r["resp"] for r in rows)
    print()
    print("=" * 100)
    print(f"РАЗЛИЧНЫХ ОТВЕТОВ НА 8 РАЗНЫХ ВОПРОСОВ: {len(uniq)}")
    print("=" * 100)
    for resp, c in uniq.most_common():
        print(f"  x{c}  {resp[:140]}")
    poles = Counter(tuple(r["poles"] or []) for r in rows)
    print(f"\n  выбранные полюса CA3: {dict(poles)}")

    # проверка: сколько раз повторяется один и тот же кластер слов
    print()
    print("=" * 100)
    print("ЧАСТОТНОСТЬ КЛЮЧЕВОГО КЛАСТЕРА «dendritic plateau potentials in pyramidal neurons»")
    print("=" * 100)
    key = "dendritic plateau potentials in pyramidal neurons"
    hits = sum(1 for r in rows if key in r["resp"])
    print(f"  встречается в {hits} из {len(rows)} ответах")

    OUT.write_text(json.dumps({"rows": rows, "unique_answers": len(uniq),
                               "cluster_hits": f"{hits}/{len(rows)}"},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("сохранено:", OUT)


if __name__ == "__main__":
    main()
