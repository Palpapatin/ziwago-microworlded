"""Печать таблицы механизмов из собранной карты (для чтения в консоли)."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1
            else r"D:\Ziwago18-storage\Produce\module_map_table_20261113.md")
text = path.read_text(encoding="utf-8")
section = text.split("## Уровень 2")[1]
rows = [line for line in section.splitlines() if line.startswith("| `")]
counts: Counter = Counter()
for line in rows:
    counts[line.split("**")[1]] += 1
print("Итого по механизмам:", dict(counts))
print()
for line in rows:
    if "**SHIPPED**" not in line:
        print(line[:200])
