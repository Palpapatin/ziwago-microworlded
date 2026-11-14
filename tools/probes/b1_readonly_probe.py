"""B.1 decisive: полный реальный путь загрузки с READONLY mmap (construct-first).

diag6: torch Windows from_file -> WRITECOPY -> коммитит весь файл (peak 7.12 ГБ).
diag7: monkeypatch from_file -> Python mmap.ACCESS_READ + frombuffer -> READONLY
file-backed, БЕЗ копии, Job peak 2.32 ГБ, все веса читаются.

Этот probe: конструкция организма ПЕРВОЙ (как в organism_core), затем
torch.load(mmap=True) с READONLY-подменой, полный v19 restore + побайтный
fingerprint. Замер: Job commit peak (limited_run), RSS/USS peak (monitor),
fp_core_restored == fp_core_source. Критерий B.1: peak <= 6 ГБ, fingerprint match.
"""
from __future__ import annotations

import faulthandler
import gc
import hashlib
import json
import mmap
import sys
import threading
import time
from pathlib import Path

faulthandler.enable(all_threads=True)
ROOT = Path(r"C:/Users/yarma/Desktop/Ziwago18")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import psutil  # noqa: E402

torch.set_num_threads(4)
CKPT = Path(r"D:/Ziwago18-storage/Produce/checkpoints/ziwago_v19_apical2_20261109.pt")
_proc = psutil.Process()
_t0 = time.perf_counter()
_PEAK_RSS = 0.0
_PEAK_PRIV = 0.0
_stop = False
_KEEPALIVE = []


def _readonly_from_file(filename, shared, size):
    f = open(filename, "rb")
    mm = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_READ)
    _KEEPALIVE.append((f, mm))
    t = torch.frombuffer(mm, dtype=torch.uint8)
    return t.untyped_storage()


torch.UntypedStorage.from_file = _readonly_from_file  # READONLY вместо WRITECOPY


def _snap():
    rss = _proc.memory_info().rss / 1e9
    try:
        priv = _proc.memory_full_info().uss / 1e9
    except Exception:
        priv = 0.0
    return rss, priv


def _mon():
    global _PEAK_RSS, _PEAK_PRIV
    while not _stop:
        try:
            rss, priv = _snap()
            _PEAK_RSS = max(_PEAK_RSS, rss)
            _PEAK_PRIV = max(_PEAK_PRIV, priv)
        except Exception:
            pass
        time.sleep(0.1)


def log(msg):
    rss, priv = _snap()
    print(f"[ro +{time.perf_counter()-_t0:6.1f}s rss={rss:.3f} priv={priv:.3f} "
          f"peak_rss={_PEAK_RSS:.3f} peak_priv={_PEAK_PRIV:.3f}] {msg}", flush=True)


BULK_CONTAINER_THRESHOLD = 256


def _hash_tensor(digest, value):
    digest.update(str((value.dtype, tuple(value.shape))).encode())
    tensor = value.detach().cpu()
    if tensor.numel() == 0:
        return
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    data = tensor.reshape(-1).view(torch.uint8).numpy()
    digest.update(memoryview(data))


def _is_scalar(value):
    return not isinstance(value, (dict, list, tuple, torch.Tensor))


def _is_flat(value):
    if _is_scalar(value):
        return True
    if isinstance(value, dict):
        return all(_is_scalar(item) for item in value.values())
    return False


def state_fingerprint(state):
    digest = hashlib.sha256()

    def visit(value):
        digest.update(type(value).__name__.encode())
        if isinstance(value, torch.Tensor):
            _hash_tensor(digest, value)
        elif isinstance(value, dict):
            digest.update(str(len(value)).encode())
            if value and all(_is_scalar(item) for item in value.values()):
                digest.update(json.dumps(
                    {f"{type(key).__name__}:{key!r}": item for key, item in value.items()},
                    sort_keys=True, ensure_ascii=False).encode())
                return
            if len(value) > BULK_CONTAINER_THRESHOLD and all(_is_flat(item) for item in value.values()):
                try:
                    digest.update(json.dumps(value, sort_keys=True, ensure_ascii=False, default=repr).encode())
                    return
                except (TypeError, ValueError):
                    pass
            for key in sorted(value, key=repr):
                visit(key)
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(str(len(value)).encode())
            if value and all(_is_scalar(item) for item in value):
                digest.update(json.dumps(list(value), sort_keys=True, ensure_ascii=False).encode())
                return
            if len(value) > BULK_CONTAINER_THRESHOLD and all(_is_flat(item) for item in value):
                try:
                    digest.update(json.dumps(list(value), sort_keys=True, ensure_ascii=False, default=repr).encode())
                    return
                except (TypeError, ValueError):
                    pass
            for item in value:
                visit(item)
        else:
            digest.update(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())

    visit(state)
    return digest.hexdigest()


def count_tensor_bytes(value):
    total = 0
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        for v in value.values():
            total += count_tensor_bytes(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            total += count_tensor_bytes(v)
    return total


def main():
    log(f"start READONLY construct-first free={psutil.virtual_memory().available/1e9:.2f}GB torch={torch.__version__}")
    from ziwago.organism_core import ZiwagoOrganism
    from ziwago.organism.verified import validate_v19_envelope
    log("imports done")

    org = ZiwagoOrganism(root=ROOT, workers=2, cpu_threads=4, seed=42,
                         auto_load_checkpoint=False, integrated=True)
    log("STEP 1: organism constructed (no auto-load)")

    payload = torch.load(str(CKPT), map_location="cpu", weights_only=True, mmap=True)
    log("STEP 2: torch.load READONLY mmap done (AFTER construction)")

    rs = payload["v19_state"]["document_qa"]["ranker"]["weights"]
    c = rs["match_bias"].clone()
    log(f"STEP 2b: clone match_bias AFTER construction OK numel={c.numel()} (mapping survived)")

    nbytes = count_tensor_bytes(payload)
    log(f"payload tensor bytes={nbytes/1e9:.3f}GB")

    validate_v19_envelope(payload)
    log("STEP 3: v19 envelope validated")

    org.core_organism.load_self_contained_checkpoint(CKPT, preloaded_payload=payload)
    log("STEP 4: core restored")

    org.load_v19_state(payload["v19_state"])
    log("STEP 5: v19 genesis state restored")

    core_restored = org.core_organism.export_runtime_state()["core"]
    fp_restored = state_fingerprint(core_restored)
    fp_source = state_fingerprint(payload["core_runtime_state"]["core"])
    log(f"STEP 6: fp_restored={fp_restored[:16]} fp_source={fp_source[:16]} match={fp_restored == fp_source}")

    fp_full = state_fingerprint(org.core_organism.export_runtime_state())
    log(f"STEP 7: fp_full={fp_full}")

    del payload
    gc.collect()
    log(f"final rss={_snap()[0]:.3f}GB peak_rss={_PEAK_RSS:.3f}GB peak_priv={_PEAK_PRIV:.3f}GB")
    print(f"RESULT strategy=readonly_construct_first core_match={fp_restored == fp_source} "
          f"fp_full={fp_full} payload_GB={nbytes/1e9:.3f} "
          f"peak_rss_GB={_PEAK_RSS:.3f} peak_priv_GB={_PEAK_PRIV:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    threading.Thread(target=_mon, daemon=True).start()
    try:
        code = main()
    except BaseException as exc:  # noqa: BLE001
        import traceback
        print(f"[ro] FAILED: {type(exc).__name__}: {exc}; "
              f"peak_rss={_PEAK_RSS:.3f}GB peak_priv={_PEAK_PRIV:.3f}GB", flush=True)
        traceback.print_exc()
        code = 1
    finally:
        _stop = True
    raise SystemExit(code)