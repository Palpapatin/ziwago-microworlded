"""B.1 micro-diagnostic v2: перебор ВСЕХ 20 тензоров document ranker.

v1 показал: скаляр match_bias читается из mmap без AV. Полный probe падает на
isfinite в цикле по всем весам. Здесь перебираем каждый вес: clone + isfinite
(клон) + isfinite (оригинал mmap), печатая имя/форму ДО операции — так видно,
на каком именно тензоре native AV. Затем воспроизводим реальный
DocumentEvidenceRanker.load_state_dict (строка 525 candidate.load_state_dict).
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
    print(f"[diag +{time.perf_counter()-_t0:6.1f}s] {m}", flush=True)


def main():
    payload = torch.load(str(CKPT), map_location="cpu", weights_only=True, mmap=True)
    log("torch.load mmap done")
    ranker_state = payload["v19_state"]["document_qa"]["ranker"]
    weights = ranker_state["weights"]
    log(f"weights n={len(weights)}")

    for name, t in weights.items():
        log(f"  -> {name} dtype={t.dtype} shape={tuple(t.shape)} numel={t.numel()} "
            f"offset={t.storage_offset()} contig={t.is_contiguous()}")
        c = t.clone()
        ok_c = bool(torch.isfinite(c).all().item())
        log(f"     clone+isfinite(clone)={ok_c}")
        ok_o = bool(torch.isfinite(t).all().item())
        log(f"     isfinite(original mmap)={ok_o}")

    log("ALL WEIGHTS READ OK; now real DocumentEvidenceRanker.load_state_dict")
    from ziwago.cortex.document_qa import DocumentEvidenceRanker
    r = DocumentEvidenceRanker()
    r.load_state_dict(ranker_state)
    log(f"DocumentEvidenceRanker.load_state_dict OK trained={r.trained}")
    log("ALL DIAG PASSED")
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