"""Критерий апоптоза SWS из предложения: «удалять полюса, дающие grammar_coherent == False».
Проверяем на РЕАЛЬНЫХ мусорных ответах v18.2+, которые я измерил."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ziwago.evaluation.gates import evaluate_grammar_and_artifacts

GARBAGE = [
    ("окрошка (Tell me about the hippocampus...)",
     "After crossing Bel Pre Road f in. overly from the all that aircraft the, a as your foring of clear have,'s."),
    ("Alex-утечка (grid cells в сессии с Priya)",
     "Instead, it is Alex. once your in it?"),
    ("абсурдная подстановка (grid cells)",
     "I know grid as read: to fix grammar errors in your essay, grid each sentence aloud, check subject-verb agreement, and simplify overly long clauses."),
    ("сдвиг слота (working as read)",
     "I know working as read: to fix grammar errors in your essay, working each sentence aloud, check subject-verb agreement, and simplify overly long clauses."),
    ("мусор из v18.1 (Pauls West Side)", "Pauls West Side early Saturday morning."),
    ("окрошка из v18.1 (пирамидальные)",
     "closed knowledge.ches stabil ap nm the hidden wooden chest and shared the golden berries with her friends."),
]

GOOD = [
    "Pyramidal neurons receive bottom-up sensory spikes on basal dendrites and top-down cortical predictions on apical tuft NMDA compartments.",
    "Your name is Priya, and you are building a neural forecasting system for extreme weather.",
    "It is possible to fix runny custard by slowly whisking the mixture into a fresh egg yolk until the emulsion stabilizes.",
]

print("=" * 104)
print("КРИТЕРИЙ АПОПТОЗА SWS: grammar_coherent == False -> полюс удаляется")
print("=" * 104)
print(f"{'образец':<46} {'грамм.':>7} {'uniq':>6} {'загл':>5} {'знак':>5} {'ВЕРДИКТ':>22}")
print("-" * 104)
kept = 0
for name, t in GARBAGE:
    e = evaluate_grammar_and_artifacts(t)
    ok = bool(e["grammar_coherent"])
    kept += int(ok)
    verdict = "ОСТАВЛЕН (не отсечён)" if ok else "удалён"
    print(f"{name[:45]:<46} {str(e['grammar_coherent']):>7} {e['unique_token_ratio']:>6.2f} "
          f"{str(e['starts_with_capital']):>5} {str(e['ends_with_punctuation']):>5} {verdict:>22}")
print("-" * 104)
print(f"  МУСОР, ПРОШЕДШИЙ КРИТЕРИЙ: {kept} из {len(GARBAGE)}")

print()
print("=" * 104)
print("КОНТРОЛЬ: нормальные ответы должны проходить")
print("=" * 104)
for t in GOOD:
    e = evaluate_grammar_and_artifacts(t)
    print(f"  {'ПРОШЁЛ' if e['grammar_coherent'] else 'ОТСЕЯН':>7}  {t[:88]}")
