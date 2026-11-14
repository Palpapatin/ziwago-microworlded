"""B.1 diag v5: load-first mmap — отображение создаётся ХОЛОДНЫМ, до конструкции.

diag2: холодный mmap читается без AV. diag3/4: mmap ПОСЛЕ конструкции — clone AV
(не память, не кэш). Гипотеза: баг в СОЗДАНИИ отображения при тёплом аллокаторе.
Если создать mmap ПЕРВЫМ (холодный), а потом строить организм — отображение может
остаться валидным. Порядок: load mmap -> clone(до) -> construct -> clone(после)
-> полный document_qa.load_state_dict. Память под контролем limited_run.
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
    print(f"[diag5 +{time.perf_counter()-_t0:6.1f}s] {m}", flush=True)


def main():
    payload = torch.load(str(CKPT), map_location="cpu", weights_only=True, mmap=True)
    log("load-first torch.load mmap done (cold allocator)")
    ranker_state = payload["v19_state"]["document_qa"]["ranker"]
    t = ranker_state["weights"]["match_bias"]
    c = t.clone()
    log(f"clone BEFORE construction OK numel={c.numel()}")

    from ziwago.organism_core import ZiwagoOrganism
    org = ZiwagoOrganism(root=ROOT, workers=2, cpu_threads=4, seed=42,
                         auto_load_checkpoint=False, integrated=True)
    log("organism constructed AFTER mmap load")

    c2 = t.clone()
    log(f"clone AFTER construction OK numel={c2.numel()}")

    from ziwago.cortex.document_qa import DocumentEvidenceRanker
    r = DocumentEvidenceRanker()
    r.load_state_dict(ranker_state)
    log(f"DocumentEvidenceRanker.load_state_dict OK trained={r.trained}")
    log("ALL DIAG5 PASSED")
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