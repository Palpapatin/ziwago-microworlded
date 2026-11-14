"""Карта «заявлено vs работает» по живым модулям Ziwago18.

Метод (без внешних пакетов):
  1. СОБЫТИЯ ВЫЗОВОВ: `sys.setprofile` записывает, какие функции реально
     исполнились во время настоящей волны обучения и настоящей генерации.
     Это прямой ответ «модуль в живом пути или нет» — без grep-эвристик.
  2. ГРАФ ОБЪЕКТОВ: обход `vars()` от организма и ядра собирает все
     экземпляры классов → «модуль создан и подвешен» (это и есть
     разница между OFF и DEAD).
  3. СОСТАВ ЧЕКПОИНТА: какие поддеревья состояния сохраняются (то, что
     организм ЗАЯВЛЯЕТ как свою часть) — читается из pickle-метаданных
     zip-контейнера, без подъёма модели.

Классификация:
  SHIPPED — есть в графе объектов И его функции вызывались в живом пути;
  PHANTOM — создан, но ни одна его функция не вызывалась в живом пути;
  OFF     — создан только под флагом, который по умолчанию выключен;
  DEAD    — не создан нигде в живом пути;
  ORPHAN  — состояние сохраняется в чекпоинте, но кода в живом пути нет.

Запуск:
    python tools/probes/module_map_probe.py --checkpoint <pt> [--fresh] [--pairs 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


# ─────────────────────────── 1. события вызовов ───────────────────────────
class CallRecorder:
    """Считает вызовы функций по (файл, функция). Только call-события."""

    def __init__(self, root: Path):
        self.root = str(root).lower() + os.sep
        self.calls: dict[tuple[str, str], int] = defaultdict(int)
        self._active = False

    def _norm(self, filename: str) -> str | None:
        low = filename.lower()
        if low.startswith(self.root):
            rel = filename[len(self.root):].replace("\\", "/")
            return rel
        return None

    def __call__(self, frame, event, arg):
        if event != "call":
            return
        code = frame.f_code
        rel = self._norm(code.co_filename)
        if rel is None:
            return
        self.calls[(rel, code.co_name)] += 1

    def __enter__(self):
        sys.setprofile(self)
        return self

    def __exit__(self, *exc):
        sys.setprofile(None)
        return False


# ─────────────────────────── 2. граф объектов ───────────────────────────
def walk_objects(*roots, max_depth: int = 6) -> dict[str, int]:
    """Классы, реально созданные в организме: {qualified_class: instances}."""
    seen_ids: set[int] = set()
    counts: dict[str, int] = defaultdict(int)
    stack = [(root, 0) for root in roots]
    while stack:
        obj, depth = stack.pop()
        if depth > max_depth:
            continue
        oid = id(obj)
        if oid in seen_ids:
            continue
        seen_ids.add(oid)
        cls = type(obj)
        if cls.__module__.startswith("ziwago"):
            counts[f"{cls.__module__}.{cls.__qualname__}"] += 1
        if isinstance(obj, (str, bytes, int, float, bool, type(None), torch.Tensor)):
            continue
        if isinstance(obj, dict):
            for value in list(obj.values())[:200]:
                stack.append((value, depth + 1))
        elif isinstance(obj, (list, tuple, set)):
            for value in list(obj)[:200]:
                stack.append((value, depth + 1))
        else:
            for value in list(vars(obj).values())[:200] if hasattr(obj, "__dict__") else []:
                stack.append((value, depth + 1))
    return dict(counts)


# ───────────────────── 3. состав чекпоинта (только метаданные) ─────────────
def checkpoint_modules(checkpoint: Path) -> dict[str, object]:
    """Какие поддеревья состояния есть в чекпоинте (без подъёма модели)."""
    import zipfile
    sys.path.insert(0, str(ROOT / "tools" / "probes"))
    from ckpt_layout import _Unpickler, _walk  # noqa: E402

    with zipfile.ZipFile(checkpoint) as zf:
        pkl_name = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        with zf.open(pkl_name) as fh:
            obj = _Unpickler(fh).load()
    tensors: list[tuple[str, dict]] = []
    _walk(obj, "", tensors, set())
    modules: dict[str, int] = defaultdict(int)
    for path, meta in tensors:
        head = path.split(".")[0]
        if head:
            modules[head] += 1
    return {"top_level_trees": dict(modules), "tensor_count": len(tensors)}


def module_files() -> dict[str, dict]:
    """Файлы ziwago/ с размерами и числом функций/классов."""
    import ast
    out: dict[str, dict] = {}
    for path in sorted((ROOT / "ziwago").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        source = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            out[rel] = {"lines": source.count("\n") + 1, "functions": 0, "classes": 0}
            continue
        functions = sum(1 for node in ast.walk(tree)
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
        classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        out[rel] = {"lines": source.count("\n") + 1, "functions": functions, "classes": classes}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--runtime-version", type=int, default=19, choices=(19, 20))
    ap.add_argument("--fresh", action="store_true", help="не грузить чекпоинт (свежий организм)")
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from ziwago_assistant import ZiwagoBiologicalSNNAssistant
    from ziwago.app.data import TokenizedDialogPair

    # Состав чекпоинта читаем ПЕРВЫМ: это только метаданные pickle, а организм
    # в памяти поднимает commit до предела и чтение zip'а может упасть.
    trees = checkpoint_modules(Path(args.checkpoint))

    t0 = time.perf_counter()
    assistant = ZiwagoBiologicalSNNAssistant(
        root=ROOT, checkpoint_path=None if args.fresh else Path(args.checkpoint),
        workers=2, cpu_threads=4, seed=42, runtime_version=args.runtime_version)
    organism = assistant.organism
    core = assistant.genesis
    load_sec = time.perf_counter() - t0

    # --- граф объектов (до обучения: кто создан при сборке) ---
    graph_before = walk_objects(core, organism)

    # --- живой путь: волна обучения + генерация ---
    sentences = [
        "I am making mayonnaise and it has become runny.",
        "How can I fix errors in my grammar when writing an essay?",
        "The museum closes at six in the evening on weekdays.",
        "She poured the coffee into a ceramic mug and sat by the window.",
    ]
    pairs = []
    for sentence in sentences[: args.pairs]:
        ids = organism.encode_text(sentence)
        pairs.append(TokenizedDialogPair("probe", ids, ids, quality_reward_hint=2.0))

    # Профилируем ВСЁ дерево ziwago/ (корень — проект), чтобы ключи файлов
    # совпадали с ключами module_files(). Фазы разделены: иначе нельзя
    # отличить «работает при обучении» от «работает при генерации».
    phase_calls: dict[str, dict[tuple[str, str], int]] = {}
    for phase in ("train", "chat", "generate"):
        with CallRecorder(ROOT) as recorder:
            if phase == "train":
                for index, pair in enumerate(pairs, start=1):
                    # Через ОБЁРТКУ: там живёт накопление ролевых связей.
                    core.train_wave_with_neuromodulator_reward([pair], wave_idx=index)
            elif phase == "chat":
                try:
                    assistant.chat("How can I fix errors in my grammar when writing an essay?")
                except Exception as exc:  # noqa: BLE001 — генерация может отказать
                    print(f"chat raised: {type(exc).__name__}: {exc}")
            else:
                try:
                    core.generate_response("How can I fix errors in my grammar?", max_tokens=32)
                except Exception as exc:  # noqa: BLE001
                    print(f"generate_response raised: {type(exc).__name__}: {exc}")
        phase_calls[phase] = dict(recorder.calls)
        print(f"фаза {phase}: вызовов {sum(recorder.calls.values())}, функций {len(recorder.calls)}",
              flush=True)

    calls: dict[tuple[str, str], int] = {}
    for phase_calls_map in phase_calls.values():
        for key, count in phase_calls_map.items():
            calls[key] = calls.get(key, 0) + count
    graph_after = walk_objects(core, organism)

    # --- что сохраняется в состоянии ---
    persisted: list[str] = []
    for export_name in ("export_checkpoint_payload", "export_state_dict"):
        exporter = getattr(organism, export_name, None)
        if exporter is None:
            continue
        try:
            payload = exporter()
        except Exception as exc:  # noqa: BLE001
            print(f"{export_name} raised: {type(exc).__name__}: {exc}")
            continue
        if isinstance(payload, tuple) and payload:
            payload = payload[0]
        if isinstance(payload, dict):
            persisted = sorted(payload.keys())
            break

    # --- классификация по файлам ---
    files = module_files()
    per_file: dict[str, dict] = {}
    for rel, info in files.items():
        if rel.endswith("__init__.py"):
            continue
        file_calls = {fn: n for (f, fn), n in calls.items() if f == rel}
        per_phase = {}
        for phase, phase_map in phase_calls.items():
            per_phase[phase] = sum(n for (f, _), n in phase_map.items() if f == rel)
        classes_here = sorted({cls for cls in graph_after if cls.startswith(rel[:-3].replace("/", ".") + ".")})
        per_file[rel] = {
            "lines": info["lines"], "functions": info["functions"], "classes": info["classes"],
            "functions_called": len(file_calls),
            "call_total": sum(file_calls.values()),
            "calls_by_phase": per_phase,
            "called_functions": sorted(file_calls)[:40],
            "instances": {cls: graph_after[cls] for cls in classes_here},
        }

    report = {
        "mode": "fresh" if args.fresh else "checkpoint",
        "runtime_version": args.runtime_version,
        "checkpoint": None if args.fresh else str(Path(args.checkpoint).resolve()),
        "load_sec": round(load_sec, 2),
        "pairs": len(pairs),
        "graph_classes_before": len(graph_before),
        "graph_classes_after": len(graph_after),
        "persisted_state_keys": persisted,
        "checkpoint_trees": trees,
        "files": per_file,
        "total_calls_in_ziwago": sum(calls.values()),
        "distinct_functions_called": len(calls),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"записано: {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
