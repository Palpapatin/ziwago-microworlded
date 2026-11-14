"""Статическая карта создания классов Ziwago18: кто кого инстанцирует.

Для каждого класса, определённого в `ziwago/`, ищутся точки конструирования
(`ClassName(...)`) в других файлах пакета. Нет точки — класс не может быть
создан в живом пути (кандидат в DEAD). Точка есть, но только под флагом —
кандидат в OFF (проверяется по флагу рядом с точкой).

Запуск:
    python tools/probes/class_instantiation_map.py [--out <json>]
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]


def _enclosing_guards(node: ast.AST, tree: ast.AST) -> List[str]:
    """Условия `if`, внутри которых создаётся объект (для отсева OFF)."""
    guards: List[str] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: List[str] = []

        def visit_If(self, node: ast.If) -> None:
            try:
                text = ast.unparse(node.test)
            except Exception:  # noqa: BLE001
                text = "<expr>"
            self.stack.append(text)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if node is target:
                guards.extend(self.stack)
            self.generic_visit(node)

    target = node
    Visitor().visit(tree)
    return guards


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    files = sorted((ROOT / "ziwago").rglob("*.py"))
    sources = {path: path.read_text(encoding="utf-8", errors="replace") for path in files}
    trees = {path: ast.parse(src) for path, src in sources.items()}

    defined: Dict[str, List[str]] = {}
    for path, tree in trees.items():
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                defined.setdefault(node.name, []).append(path.relative_to(ROOT).as_posix())

    # точки конструирования + условие внешнего if (для отсева OFF)
    sites: Dict[str, List[Dict[str, object]]] = {}
    for path, tree in trees.items():
        rel = path.relative_to(ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            name = node.func.id
            if name not in defined:
                continue
            guard = _enclosing_guards(node, tree)
            sites.setdefault(name, []).append({
                "file": rel, "line": node.lineno,
                "own_file": rel in defined[name],
                "guards": guard,
            })

    # флаги рядом с точкой: ищем в +-6 строках присваивание *_enabled
    flag_re = re.compile(r"(\w*enabled\w*)\s*=\s*(True|False)")
    out = {}
    for name, where in sorted(defined.items()):
        entries = []
        for site in sites.get(name, []):
            src = sources[ROOT / site["file"]].splitlines()
            lo = max(0, site["line"] - 7)
            window = "\n".join(src[lo:site["line"] + 2])
            flags = flag_re.findall(window)
            entries.append({
                "at": f"{site['file']}:{site['line']}",
                "inside_own_module": site["own_file"],
                "flags_nearby": [f"{f}={v}" for f, v in flags],
                "guards": site.get("guards", []),
            })
        out[name] = {
            "defined_in": where,
            "instantiations": entries,
            "external_instantiations": [e for e in entries if not e["inside_own_module"]],
        }

    report = {
        "classes_total": len(defined),
        "classes_with_no_instantiation_anywhere": sorted(
            name for name in defined
            if not [e for e in out[name]["instantiations"] if not e["inside_own_module"]]),
        "classes_only_instantiated_in_own_module": sorted(
            name for name in defined
            if out[name]["instantiations"] and not out[name]["external_instantiations"]),
        "classes": out,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"записано: {args.out}")
        print("классов всего:", report["classes_total"])
        print("без внешнего конструирования:", len(report["classes_with_no_instantiation_anywhere"]))
        for name in report["classes_with_no_instantiation_anywhere"]:
            print("   -", name, "->", out[name]["defined_in"][0])
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
