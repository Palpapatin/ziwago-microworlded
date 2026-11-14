"""B.1 chemistry leaf-diff: точное поле химии, которое отличается после restore.

b1_fidelity_diff: 41/42 ключей ядра совпадают побайтно, отличается ТОЛЬКО
chemistry. Рекурсивно спускаемся в chemistry (__object__/attributes/dict),
сравниваем тензоры побайтно и скаляры по значению; печатаем пути листьев
с live vs source значениями. Это покажет, легитимная ли это трансформация
restore (time_ms / neuromod reference) или дефект.
"""
from __future__ import annotations
import faulthandler
import sys
import time
from pathlib import Path

faulthandler.enable(all_threads=True)
ROOT = Path(r"C:/Users/yarma/Desktop/Ziwago18")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, r"D:/Ziwago18-storage/Produce")
import torch  # noqa: E402
torch.set_num_threads(4)
import b1_production_probe as P  # noqa: E402
from ziwago.organism_core import ZiwagoOrganism, _readonly_checkpoint_mapping  # noqa: E402

_t0 = time.perf_counter()


def log(m):
    print(f"[chem +{time.perf_counter()-_t0:6.1f}s] {m}", flush=True)


def leaf_diff(a, b, path, out):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            out.append((path, f"type {type(a).__name__}", f"type {type(b).__name__}"))
            return
        if a.shape != b.shape or a.dtype != b.dtype:
            out.append((path, f"{tuple(a.shape)}/{a.dtype}", f"{tuple(b.shape)}/{b.dtype}"))
            return
        if not torch.equal(a.detach().cpu(), b.detach().cpu()):
            out.append((path, f"tensor numel={a.numel()} first={a.reshape(-1)[:3].tolist()}",
                        f"tensor first={b.reshape(-1)[:3].tolist()}"))
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=repr):
            if k not in a:
                out.append((f"{path}.{k}", "<MISSING live>", repr(b[k])[:80]))
            elif k not in b:
                out.append((f"{path}.{k}", repr(a[k])[:80], "<MISSING src>"))
            else:
                leaf_diff(a[k], b[k], f"{path}.{k}", out)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append((path, f"len={len(a)}", f"len={len(b)}"))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            leaf_diff(x, y, f"{path}[{i}]", out)
        return
    if a != b:
        out.append((path, repr(a)[:80], repr(b)[:80]))


def main():
    org = ZiwagoOrganism(root=ROOT, workers=2, cpu_threads=4, seed=42,
                         auto_load_checkpoint=False, integrated=True)
    org.load_v19_checkpoint(P.CKPT)
    log("loaded")
    excluded = {"root", "dialog_streamer", "moe_cortex", "spiking_language_loop"}
    live_core = {k: v for k, v in vars(org.core_organism).items()
                 if k not in excluded and not k.startswith("external_") and not callable(v)}
    live_chem = P.export_form_nocopy(live_core["chemistry"])
    with _readonly_checkpoint_mapping():
        payload = torch.load(str(P.CKPT), map_location="cpu", weights_only=True, mmap=True)
    src_chem = P.normalize_tags(payload["core_runtime_state"]["core"]["chemistry"])
    out = []
    leaf_diff(live_chem, src_chem, "chemistry", out)
    log(f"chemistry leaf differences = {len(out)}")
    for path, lv, sv in out:
        log(f"  {path}\n     live={lv}\n     src ={sv}")
    log("CHEM DIFF DONE")
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