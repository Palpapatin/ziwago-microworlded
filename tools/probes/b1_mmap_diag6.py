"""B.1 diag v6: измерить, КОММИТИТ ли torch Windows mmap весь файл.

diag2: холодный mmap без конструкции -> пик 6.86 ГБ. diag5: mmap + конструкция
-> OOM 8.63 ГБ. Гипотеза: from_file(shared=False) на Windows создаёт
PAGE_WRITECOPY / MEM_PRIVATE -> весь файл 4.5 ГБ идёт в private commit.
Если так — mmap НЕ экономит commit, и load-first не помещается.

Прибор: VirtualQuery по data_ptr тензоров -> Type (MEM_MAPPED=0x40000 file-backed
evictable / MEM_PRIVATE=0x20000 commit), Protect (PAGE_READONLY=0x02 /
PAGE_WRITECOPY=0x08 / PAGE_READWRITE=0x04), State. Плюс RSS/USS до/после load
и после чтения всех данных. Без конструкции организма (дёшево).
"""
from __future__ import annotations
import ctypes
import faulthandler
import sys
import time
from pathlib import Path

faulthandler.enable(all_threads=True)
ROOT = Path(r"C:/Users/yarma/Desktop/Ziwago18")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
torch.set_num_threads(4)
import psutil  # noqa: E402

CKPT = Path(r"D:/Ziwago18-storage/Produce/checkpoints/ziwago_v19_apical2_20261109.pt")
_t0 = time.perf_counter()
_proc = psutil.Process()

MEM_COMMIT = 0x1000
MEM_MAPPED = 0x40000
MEM_PRIVATE = 0x20000
PROT = {0x01: "NOACCESS", 0x02: "READONLY", 0x04: "READWRITE",
        0x08: "WRITECOPY", 0x10: "EXECUTE", 0x20: "EXECUTE_READ",
        0x40: "EXECUTE_READWRITE", 0x80: "EXECUTE_WRITECOPY"}


class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p),
                ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", ctypes.c_ulong),
                ("RegionSize", ctypes.c_size_t),
                ("State", ctypes.c_ulong),
                ("Protect", ctypes.c_ulong),
                ("Type", ctypes.c_ulong)]


def vq(ptr):
    mbi = MEMORY_BASIC_INFORMATION()
    ok = ctypes.windll.kernel32.VirtualQuery(ctypes.c_void_p(ptr),
                                             ctypes.byref(mbi), ctypes.sizeof(mbi))
    if not ok:
        return "VirtualQuery FAILED"
    typ = ("MAPPED" if mbi.Type & MEM_MAPPED else
           "PRIVATE" if mbi.Type & MEM_PRIVATE else f"0x{mbi.Type:x}")
    st = "COMMIT" if mbi.State & MEM_COMMIT else f"0x{mbi.State:x}"
    pr = PROT.get(mbi.Protect, f"0x{mbi.Protect:x}")
    apr = PROT.get(mbi.AllocationProtect, f"0x{mbi.AllocationProtect:x}")
    return (f"Type={typ} State={st} Protect={pr} AllocProtect={apr} "
            f"RegionSize={mbi.RegionSize/1e9:.3f}GB")


def mem(tag):
    mi = _proc.memory_info()
    try:
        uss = _proc.memory_full_info().uss
    except Exception:
        uss = -1
    print(f"[diag6 +{time.perf_counter()-_t0:6.1f}s] {tag}: "
          f"RSS={mi.rss/1e9:.3f}GB USS(private)={uss/1e9:.3f}GB", flush=True)


def log(m):
    print(f"[diag6 +{time.perf_counter()-_t0:6.1f}s] {m}", flush=True)


def main():
    mem("baseline (torch imported)")
    payload = torch.load(str(CKPT), map_location="cpu", weights_only=True, mmap=True)
    mem("after torch.load mmap (no data touched yet)")
    log(f"file size = {CKPT.stat().st_size/1e9:.3f}GB")

    rs = payload["v19_state"]["document_qa"]["ranker"]["weights"]
    for name in ("match_bias", "embedding.weight", "head.bias"):
        t = rs[name]
        log(f"ranker.{name} data_ptr=0x{t.data_ptr():x} -> {vq(t.data_ptr())}")

    # большой тензор коры из core_runtime_state
    core = payload["core_runtime_state"]["core"]
    big = None
    def walk(o):
        nonlocal big
        if isinstance(o, torch.Tensor):
            if big is None or o.numel() > big.numel():
                big = o
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
    walk(core)
    if big is not None:
        log(f"biggest core tensor numel={big.numel()} shape={tuple(big.shape)} "
            f"data_ptr=0x{big.data_ptr():x} -> {vq(big.data_ptr())}")

    # трогаем ВСЕ данные: сумма по всем тензорам payload (форсирует fault страниц)
    total = 0
    nbytes = 0
    def touch(o):
        nonlocal total, nbytes
        if isinstance(o, torch.Tensor):
            total += float(o.detach().reshape(-1)[:1].sum()) if o.numel() else 0.0
            nbytes += o.numel() * o.element_size()
        elif isinstance(o, dict):
            for v in o.values():
                touch(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                touch(v)
    touch(payload)
    mem("after touching first element of every tensor")
    log(f"payload tensor bytes (views) = {nbytes/1e9:.3f}GB")
    log("DIAG6 DONE")
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