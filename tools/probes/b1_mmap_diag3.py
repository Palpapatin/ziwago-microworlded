"""B.1 diag v3: изоляция триггера AV — конструкция организма ПЕРЕД mmap-загрузкой.

diag2 (без конструкции) читает все веса mmap без AV. Полный probe (с конструкцией)
детерминированно падает на isfinite(original) в document_qa:523. Здесь ровно одно
отличие от diag2 — сначала строим организм (integrated=True), БЕЗ monitor-потока и
БЕЗ count_tensor_bytes. На каждом весе: clone -> isfinite(clone) -> isfinite(original).
Если clone проходит, а isfinite(original) падает → лечение = клон до native-чтения.
Если падает и clone → mmap после конструкции невалиден (нужна другая стратегия).
"""
from __future__ import annotations
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

CKPT = Path(r"D:/Ziwago18-storage/Produce/checkpoints/ziwago_v19_apical2_20261109.pt")
_t0 = time.perf_counter()


def log(m):
    print(f"[diag3 +{time.perf_counter()-_t0:6.1f}s] {m}", flush=True)


def main():
    from ziwago.organism_core import ZiwagoOrganism
    org = ZiwagoOrganism(root=ROOT, workers=2, cpu_threads=4, seed=42,
                         auto_load_checkpoint=False, integrated=True)
    log("organism constructed (integrated=True)")

    payload = torch.load(str(CKPT), map_location="cpu", weights_only=True, mmap=True)
    log("torch.load mmap done")
    weights = payload["v19_state"]["document_qa"]["ranker"]["weights"]
    log(f"weights n={len(weights)}")

    for name, t in weights.items():
        log(f"  -> {name} shape={tuple(t.shape)} numel={t.numel()}")
        c = t.clone()
        log(f"     clone OK")
        ok_c = bool(torch.isfinite(c).all().item())
        log(f"     isfinite(clone)={ok_c}")
        ok_o = bool(torch.isfinite(t).all().item())
        log(f"     isfinite(original mmap)={ok_o}")

    log("ALL DIAG3 PASSED")
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