"""B.1 fidelity diff: КАКОЙ ключ ядра отличается после restore.

b1_fidelity3: NORM_MATCH=False (fp_restored != fp_source_norm).Restore легитимно
трансформирует состояние (apply_checkpoint_overlay достраивает pos/trace до
max_pos; load_runtime_state добавляет prune_threshold). Нужно точно локализовать
расхождение: live-only / source-only ключи + per-key fingerprint общих ключей.

Использует helpers из b1_production_probe (state_fingerprint, export_form_nocopy,
normalize_tags). READONLY load, no-copy. Под limited_run.
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
    print(f"[diff +{time.perf_counter()-_t0:6.1f}s] {m}", flush=True)


def main():
    org = ZiwagoOrganism(root=ROOT, workers=2, cpu_threads=4, seed=42,
                         auto_load_checkpoint=False, integrated=True)
    log("constructed")
    org.load_v19_checkpoint(P.CKPT)
    log("loaded")
    excluded = {"root", "dialog_streamer", "moe_cortex", "spiking_language_loop"}
    live_core = {k: v for k, v in vars(org.core_organism).items()
                 if k not in excluded and not k.startswith("external_") and not callable(v)}
    live_form = P.export_form_nocopy(live_core)
    with _readonly_checkpoint_mapping():
        payload = torch.load(str(P.CKPT), map_location="cpu", weights_only=True, mmap=True)
    src = P.normalize_tags(payload["core_runtime_state"]["core"])
    lk, sk = set(live_form), set(src)
    log(f"live keys={len(lk)} source keys={len(sk)}")
    log(f"LIVE-ONLY keys: {sorted(lk - sk)}")
    log(f"SOURCE-ONLY keys: {sorted(sk - lk)}")
    mism = []
    for k in sorted(lk & sk):
        a = P.state_fingerprint(live_form[k])
        b = P.state_fingerprint(src[k])
        if a != b:
            mism.append(k)
            log(f"  DIFF {k}: live={a[:12]} src={b[:12]}")
    log(f"common keys={len(lk & sk)} MISMATCHED={len(mism)}: {mism}")
    log("DIFF DONE")
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