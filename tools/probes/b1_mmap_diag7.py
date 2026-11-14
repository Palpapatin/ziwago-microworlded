"""B.1 diag v7: подменить torch Windows WRITECOPY-отображение на READONLY.

diag6 доказал: from_file(shared=False) -> PAGE_WRITECOPY -> коммитит весь файл
4.535 ГБ (Job peak 7.12 ГБ при USS 1 ГБ). Это корень и OOM, и AV.

Гипотеза fixes: Python mmap.ACCESS_READ даёт PAGE_READONLY file-backed (evictable,
НЕ коммитится). Подменяем torch.UntypedStorage.from_file на обёртку над
read-only mmap + torch.frombuffer (без копии). torch.load(mmap=True) тогда
получает read-only overall_storage; тензоры — views; восстановление клонирует.

Прибор (БЕЗ конструкции, дёшево): VirtualQuery -> Protect=READONLY Type=MAPPED;
RSS/USS/Job-peak до/после; data_ptr совпадает с адресом mmap (нет копии);
clone всех 20 весов ranker читается. Если commit stays low — механизм рабочий.
"""
from __future__ import annotations
import ctypes
import faulthandler
import mmap
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
_KEEPALIVE = []

PROT = {0x01: "NOACCESS", 0x02: "READONLY", 0x04: "READWRITE",
        0x08: "WRITECOPY", 0x20: "EXECUTE_READ", 0x40: "EXECUTE_READWRITE"}
MEM_MAPPED = 0x40000
MEM_PRIVATE = 0x20000
MEM_COMMIT = 0x1000


class MBI(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p),
                ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", ctypes.c_ulong),
                ("RegionSize", ctypes.c_size_t),
                ("State", ctypes.c_ulong),
                ("Protect", ctypes.c_ulong),
                ("Type", ctypes.c_ulong)]


def vq(ptr):
    mbi = MBI()
    ok = ctypes.windll.kernel32.VirtualQuery(ctypes.c_void_p(ptr),
                                             ctypes.byref(mbi), ctypes.sizeof(mbi))
    if not ok:
        return "VirtualQuery FAILED"
    typ = ("MAPPED" if mbi.Type & MEM_MAPPED else
           "PRIVATE" if mbi.Type & MEM_PRIVATE else f"0x{mbi.Type:x}")
    st = "COMMIT" if mbi.State & MEM_COMMIT else f"0x{mbi.State:x}"
    return (f"Type={typ} State={st} Protect={PROT.get(mbi.Protect, hex(mbi.Protect))} "
            f"AllocProtect={PROT.get(mbi.AllocationProtect, hex(mbi.AllocationProtect))} "
            f"RegionSize={mbi.RegionSize/1e9:.3f}GB")


def mem(tag):
    mi = _proc.memory_info()
    try:
        uss = _proc.memory_full_info().uss
    except Exception:
        uss = -1
    print(f"[diag7 +{time.perf_counter()-_t0:6.1f}s] {tag}: "
          f"RSS={mi.rss/1e9:.3f}GB USS={uss/1e9:.3f}GB", flush=True)


def log(m):
    print(f"[diag7 +{time.perf_counter()-_t0:6.1f}s] {m}", flush=True)


def _readonly_from_file(filename, shared, size):
    f = open(filename, "rb")
    mm = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_READ)
    _KEEPALIVE.append((f, mm))
    t = torch.frombuffer(mm, dtype=torch.uint8)
    log(f"frombuffer: data_ptr=0x{t.data_ptr():x} numel={t.numel()} "
        f"storage_nbytes={t.untyped_storage().nbytes()}")
    return t.untyped_storage()


def main():
    mem("baseline")
    torch.UntypedStorage.from_file = _readonly_from_file  # monkeypatch
    payload = torch.load(str(CKPT), map_location="cpu", weights_only=True, mmap=True)
    mem("after torch.load with READONLY mmap")

    rs = payload["v19_state"]["document_qa"]["ranker"]["weights"]
    t = rs["match_bias"]
    log(f"match_bias -> {vq(t.data_ptr())}")
    big = rs["embedding.weight"]
    log(f"embedding.weight -> {vq(big.data_ptr())}")

    ok = 0
    for name, w in rs.items():
        c = w.clone()
        assert torch.isfinite(c).all(), name
        ok += 1
    mem(f"after clone+isfinite of all {ok} ranker weights")
    log(f"ALL {ok} RANKER WEIGHTS READ OK via READONLY mapping")
    log("DIAG7 DONE (no construction)")
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