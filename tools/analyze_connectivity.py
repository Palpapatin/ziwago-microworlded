"""Связность проекта: замыкание импортов от рабочих точек входа (полные dotted-пути).

Модули регистрируются по полному dotted-имени (ziwago.evaluation.gates),
относительные импорты разворачиваются от пакета импортирующего файла.
"""
from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

PKGS = ("ziwago", "neuroweave9", "neuroweave8")
# v19_source_backup — откатная копия исходников (миграция на D:), не живой код
SKIP_TOP = {"__pycache__", ".git", ".ipynb_checkpoints", "v19_source_backup"}


def register() -> Dict[str, Path]:
    """dotted-имя модуля -> файл."""
    mods: Dict[str, Path] = {}
    for p in sorted(ROOT.rglob("*.py")):
        rel = p.relative_to(ROOT)
        if any(part in SKIP_TOP for part in rel.parts):
            continue
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            mods[".".join(parts[:-1])] = p
        elif len(parts) == 1:
            mods[parts[0][:-3]] = p
        else:
            mods[".".join(parts[:-1] + [parts[-1][:-3]])] = p
    return mods


def dotted_of(path: Path) -> str:
    rel = path.relative_to(ROOT)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        return ".".join(parts[:-1])
    if len(parts) == 1:
        return parts[0][:-3]
    return ".".join(parts[:-1] + [parts[-1][:-3]])


def resolve(name: str, mods: Dict[str, Path]) -> Optional[str]:
    """Ищет модуль по имени или по любому его суффиксу-паре."""
    if name in mods:
        return name
    cands = [m for m in mods if m.endswith("." + name) or m == name]
    if len(cands) == 1:
        return cands[0]
    if cands:  # предпочитаем самый короткий (ближе к корню)
        return sorted(cands, key=len)[0]
    return None


def deps_of(path: Path, mods: Dict[str, Path]) -> Set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return set()
    own = dotted_of(path)
    pkg = own.rsplit(".", 1)[0] if "." in own else ""
    out: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                r = resolve(a.name, mods)
                if r:
                    out.add(r)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = pkg.split(".") if pkg else []
                up = node.level - 1
                if up:
                    parts = parts[:-up] if up <= len(parts) else []
                base = ".".join(parts + ([base] if base else []))
            if base:
                r = resolve(base, mods)
                if r:
                    out.add(r)
            for a in node.names:
                r = resolve(f"{base}.{a.name}" if base else a.name, mods)
                if r:
                    out.add(r)
    return out


def main() -> None:
    mods = register()
    entries: List[Path] = []
    for pat in ("train_ziwago_v18*.py", "train_ziwago_v20*.py", "run_ziwago_v25*.py",
                "ziwago_assistant.py", "verify_*.py", "tests/*.py", "tools/*.py",
                "run_*.py", "interactive_*.py"):
        entries.extend(sorted(ROOT.glob(pat)))
    entries = sorted(set(entries))

    graph: Dict[str, Set[str]] = {}
    for name, path in mods.items():
        graph[name] = deps_of(path, mods) - {name}

    used_by: Dict[str, Set[str]] = defaultdict(set)
    for m, ds in graph.items():
        for d in ds:
            used_by[d].add(m)

    roots = {dotted_of(p) for p in entries if dotted_of(p) in mods}
    reach: Set[str] = set()
    stack = list(roots)
    while stack:
        m = stack.pop()
        if m in reach:
            continue
        reach.add(m)
        stack.extend(graph.get(m, ()))

    unreachable = sorted(set(mods) - reach)

    print("=" * 100)
    print("СВЯЗНОСТЬ · замыкание импортов от рабочих точек входа")
    print("=" * 100)
    print(f"  точек входа {len(entries)} | модулей {len(mods)} | "
          f"достижимо {len(reach)} | НЕ связано {len(unreachable)}")
    print()
    print("--- НЕ СВЯЗАНЫ (кандидаты на удаление) ---")
    for m in unreachable:
        p = mods[m]
        users = sorted(used_by.get(m, []))
        print(f"  {str(p.relative_to(ROOT)):<58} {p.stat().st_size/1024:>8.1f} КБ"
              f"  [{', '.join(users) if users else 'никто не импортирует'}]")

    print()
    print("--- ТОЧКИ ВХОДА ---")
    for r in sorted(roots):
        print("   ", r)

    (ROOT / "Produce" / "connectivity_report.json").write_text(
        json.dumps({"reachable": sorted(reach), "unreachable": unreachable,
                    "roots": sorted(roots),
                    "used_by": {k: sorted(v) for k, v in used_by.items()}},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("отчёт: Produce/connectivity_report.json")


if __name__ == "__main__":
    main()
