"""Карта памяти чекпоинта без загрузки тензоров.

Читает только `data.pkl` через кастомный Unpickler: storages подменяются
заглушками, поэтому ни один байт весов не читается. Нужен, чтобы понять,
из чего состоит payload (и сколько в нём места у `u_slow`), не поднимая
4+ ГБ в RAM.

Запуск:
    python tools/probes/ckpt_layout.py <checkpoint.pt> [--top 25] [--grep u_slow]
"""
from __future__ import annotations

import argparse
import collections
import pickle
import sys
import zipfile
from typing import Any, Dict, List, Tuple

DTYPE_BYTES = {
    "FloatStorage": 4, "DoubleStorage": 8, "HalfStorage": 2, "BFloat16Storage": 2,
    "LongStorage": 8, "IntStorage": 4, "ShortStorage": 2, "CharStorage": 1,
    "ByteStorage": 1, "BoolStorage": 1, "Float8_e4m3fnStorage": 1,
    "Float8_e5m2Storage": 1, "ComplexDoubleStorage": 16, "ComplexFloatStorage": 8,
    "QInt8Storage": 1, "QUInt8Storage": 1, "QInt32Storage": 4, "QUInt4x2Storage": 1,
    "QUInt2x4Storage": 1, "Int8Storage": 1, "UInt8Storage": 1, "Int16Storage": 2,
    "Int32Storage": 4, "Int64Storage": 8, "Float16Storage": 2, "BoolStorage": 1,
}


class _StorageStub:
    __slots__ = ("key", "dtype_name", "numel")

    def __init__(self, key: str, dtype_name: str, numel: int):
        self.key, self.dtype_name, self.numel = key, dtype_name, numel

    def __repr__(self) -> str:  # pragma: no cover - диагностика
        return f"<storage {self.key} {self.dtype_name} n={self.numel}>"


def _rebuild_tensor_v2(storage, storage_offset, size, stride, *args, **kwargs):
    return {"__tensor__": True, "storage": storage, "offset": storage_offset,
            "size": tuple(int(s) for s in size), "stride": tuple(int(s) for s in stride)}


def _rebuild_tensor(storage, storage_offset, size, stride, *args, **kwargs):
    return _rebuild_tensor_v2(storage, storage_offset, size, stride)


class _Unpickler(pickle.Unpickler):
    """Подменяет storages и сборку тензоров — веса не читаются."""

    def find_class(self, module: str, name: str):
        if module == "torch._utils" and name.startswith("_rebuild_tensor"):
            return _rebuild_tensor_v2 if name.endswith("_v2") else _rebuild_tensor
        if module == "torch" and name.endswith("Storage"):
            return type(name, (), {})
        return super().find_class(module, name)

    def persistent_load(self, pid):
        if isinstance(pid, tuple) and pid and pid[0] == "storage":
            _, dtype, key, location, numel = pid
            return _StorageStub(str(key), str(getattr(dtype, "__name__", dtype)), int(numel))
        return _StorageStub(f"pid:{pid}", "unknown", 0)


def _walk(obj: Any, prefix: str, out: List[Tuple[str, Dict[str, Any]]], seen: set):
    if isinstance(obj, dict) and obj.get("__tensor__"):
        storage = obj["storage"]
        out.append((prefix, {"key": storage.key, "dtype": storage.dtype_name,
                             "numel": storage.numel, "shape": obj["size"],
                             "stride": obj.get("stride", ())}))
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk(v, f"{prefix}.{k}" if prefix else str(k), out, seen)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _walk(v, f"{prefix}[{i}]", out, seen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--grep", default=None, help="показать только пути, содержащие подстроку")
    ap.add_argument("--prefix", default="", help="отфильтровать по префиксу пути")
    args = ap.parse_args()

    with zipfile.ZipFile(args.checkpoint) as zf:
        names = [n for n in zf.namelist() if n.endswith("data.pkl")]
        if not names:
            print("data.pkl не найден", file=sys.stderr)
            return 2
        pkl_name = names[0]
        prefix = pkl_name[: -len("data.pkl")]
        with zf.open(pkl_name) as fh:
            obj = _Unpickler(fh).load()

    tensors: List[Tuple[str, Dict[str, Any]]] = []
    _walk(obj, "", tensors, set())

    # Размер на диске по storage-ключу.
    with zipfile.ZipFile(args.checkpoint) as zf:
        entry_sizes = {}
        for info in zf.infolist():
            if info.filename.startswith(prefix) and "/data/" in info.filename:
                key = info.filename.rsplit("/", 1)[1]
                entry_sizes[key] = info.file_size

    by_storage: Dict[str, List[Tuple[str, Dict[str, Any]]]] = collections.defaultdict(list)
    for path, meta in tensors:
        by_storage[meta["key"]].append((path, meta))

    rows = []
    for key, items in by_storage.items():
        size_bytes = entry_sizes.get(key)
        for path, meta in items:
            rows.append({
                "path": path, "key": key, "dtype": meta["dtype"], "shape": meta["shape"],
                "numel": meta["numel"], "bytes": size_bytes,
                "est_mb": (meta["numel"] * DTYPE_BYTES.get(meta["dtype"], 4)) / 1e6,
            })

    if args.grep:
        rows = [r for r in rows if args.grep.lower() in r["path"].lower()]
    if args.prefix:
        rows = [r for r in rows if r["path"].lower().startswith(args.prefix.lower())]

    rows.sort(key=lambda r: -(r["bytes"] or r["numel"] * 4))
    total_unique = sum((entry_sizes.get(k) or 0) for k in by_storage)
    print(f"checkpoint: {args.checkpoint}")
    print(f"prefix: {prefix}  tensors: {len(tensors)}  unique storages: {len(by_storage)}")
    print(f"unique storage bytes: {total_unique/1e9:.3f} GB")
    print(f"{'MB':>10}  {'shape':<24} {'dtype':<10} {'key':>7} path")
    for r in rows[: args.top]:
        shape = "x".join(str(s) for s in r["shape"])[:24]
        mb = (r["bytes"] / 1e6) if r["bytes"] else r["est_mb"]
        print(f"{mb:10.1f}  {shape:<24} {r['dtype']:<10} {r['key']:>7} {r['path']}")
    # Группы: один storage, разделяемый несколькими тензорами (дедупликация).
    shared = {k: v for k, v in by_storage.items() if len(v) > 1}
    if shared:
        print(f"\nstorages, разделяемые несколькими тензорами: {len(shared)}")
        for k, items in sorted(shared.items(), key=lambda kv: -(entry_sizes.get(kv[0]) or 0))[:10]:
            mb = (entry_sizes.get(k) or 0) / 1e6
            print(f"  key={k} {mb:.1f} MB -> {len(items)} тензоров")
            for path, _ in items[:6]:
                print(f"      {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
