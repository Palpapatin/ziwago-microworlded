"""Прибор живого графа импортов (задача #20: разрезание import-рёбер).

Импортирует живой путь v14 (moe_cortex-тренер, который используют рантаймы
v19/v20) и IntegratedOrganism, затем проверяет, что genesis-балласт
(neuroweave8/9, unified.pipeline, organism.genesis, train_ziwago_v10)
больше НЕ подтягивается на уровне модуля.

Запуск (только импорт, без инстанцирования организма/чекпоинта):
    python tools/limited_run.py --ram-pct 55 --cpu-pct 50 -- \
        python tools/probe_import_graph.py

Каждое число — прямое измерение sys.modules (AGENTS.md §6/§7).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Балластные префиксы/модули, которых НЕ должно быть в живом графе.
BALLAST_PREFIXES = ("neuroweave8", "neuroweave9")
BALLAST_MODULES = (
    "ziwago.unified.pipeline",
    "ziwago.organism.genesis",
    "train_ziwago_v10",
)
# Живые точки входа, которые реально грузит рантайм v19/v20.
LIVE_ENTRIES = (
    "train_ziwago_v14_pure_synaptic_dialogue",
    "ziwago.organism.integrated",
)


def main() -> int:
    before = set(sys.modules)
    for entry in LIVE_ENTRIES:
        __import__(entry)
    after = set(sys.modules)

    nw8 = sorted(m for m in after if m.startswith("neuroweave8"))
    nw9 = sorted(m for m in after if m.startswith("neuroweave9"))
    ballast = sorted(m for m in after if m in BALLAST_MODULES)
    newly = sorted(after - before)

    print("=== ЖИВОЙ ГРАФ ИМПОРТОВ (задача #20) ===")
    print(f"live_entries        : {list(LIVE_ENTRIES)}")
    print(f"modules_before      : {len(before)}")
    print(f"modules_after       : {len(after)}")
    print(f"modules_newly_loaded: {len(newly)}")
    print(f"neuroweave8_count   : {len(nw8)}")
    print(f"neuroweave9_count   : {len(nw9)}")
    print(f"ballast_present     : {ballast}")
    if nw8:
        print(f"  nw8_modules       : {nw8}")
    if nw9:
        print(f"  nw9_modules       : {nw9}")

    clean = not nw8 and not nw9 and not ballast
    print(f"RESULT              : {'CLEAN (балласт отрезан)' if clean else 'BALLAST STILL PRESENT'}")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())