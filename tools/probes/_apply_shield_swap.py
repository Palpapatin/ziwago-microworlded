"""Точечная замена класса щита в ziwago/cortex/next16_core.py.

Заменяет диапазон строк между маркерами `class FusiZenkeCascadeMetaplasticityShield:`
и следующим разделителем `# ====` на содержимое scratch-файла.
Использование:
    python tools/probes/_apply_shield_swap.py <scratch_class.py> <target.py>
"""
from __future__ import annotations

import sys
from pathlib import Path

START = "class FusiZenkeCascadeMetaplasticityShield:"


def main() -> int:
    scratch = Path(sys.argv[1])
    target = Path(sys.argv[2])
    new_class = scratch.read_text(encoding="utf-8").rstrip("\n")
    lines = target.read_text(encoding="utf-8").splitlines(keepends=True)

    start = next((i for i, l in enumerate(lines) if l.startswith(START)), None)
    if start is None:
        print("класс не найден", file=sys.stderr)
        return 2
    end = None
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("# ====") and j > start + 5:
            end = j
            break
    if end is None:
        print("конец класса не найден", file=sys.stderr)
        return 2
    # Отрезаем хвостовые пустые строки перед разделителем.
    while end - 1 > start and lines[end - 1].strip() == "":
        end -= 1
    backup = target.with_suffix(target.suffix + ".bak_20261113")
    if not backup.exists():
        backup.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    out = "".join(lines[:start]) + new_class + "\n\n\n" + "".join(lines[end:])
    target.write_text(out, encoding="utf-8")
    print(f"заменено строк {start + 1}..{end} в {target} (бэкап: {backup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
