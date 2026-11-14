"""B.1 diag v10: напечатать ВСЕ __object__-теги (не только bad) + пути.

Цель: точный класс из train_ziwago_v11_5_predictive_readout_pfc (модуль удалён)
и train_ziwago_v14 (4 тега). READONLY load, без конструкции.
"""
from __future__ import annotations
import faulthandler
import mmap
import sys
import time
from collections import Counter
from pathlib import Path

faulthandler.enable(all_threads=True)
ROOT = Path(r"C:/Users/yarma/Desktop/Ziwago18")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
torch.set_num_threads(4)
CKPT = Path(r"D:/Ziwago18-storage/Produce/checkpoints/ziwago_v19_apical2_20261109.pt")
_t0 = time.perf_counter()
_KEEPALIVE = []


def _ro(filename, shared, size):
    f = open(filename, "rb")
    mm = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_READ)
    _KEEPALIVE.append((f, mm))
    return torch.frombuffer(mm, dtype=torch.uint8).untyped_storage()


torch.UntypedStorage.from_file = _ro


def log(m):
    print(f"[diag10 +{time.perf_counter()-_t0:6.1f}s] {m}", flush=True)


def main():
    payload = torch.load(str(CKPT), map_location="cpu", weights_only=True, mmap=True)
    tags = Counter()
    first = {}

    def walk(o, path):
        if isinstance(o, dict):
            if "__object__" in o:
                t = o["__object__"]
                tags[t] += 1
                first.setdefault(t, path)
            for k, v in o.items():
                walk(v, f"{path}.{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")

    for top in ("core_runtime_state", "v19_state"):
        if top in payload:
            walk(payload[top], top)
    log("ALL distinct __object__ tags:")
    for t in sorted(tags):
        log(f"  {t}  count={tags[t]}  path={first[t]}")
    log("DIAG10 DONE")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except BaseException as exc:  # noqa: BLE001
        import traceback
        log(f"FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        code = 1
    raise SystemExit(code)