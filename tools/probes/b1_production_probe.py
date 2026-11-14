"""B.1 production: полный путь РЕАЛЬНОГО загрузчика + верность + пик.

Тестирует production-код (не monkeypatch probe):
  ZiwagoOrganism.load_v19_checkpoint -> _load_checkpoint_payload (READONLY mmap)
  -> _load_v19_payload -> load_self_contained_checkpoint -> load_runtime_tree
  (миграция neuroweave10->ziwago). Заодно проверяет, что READONLY-отображение
  переживает выход из контекста (keepalive через torch.frombuffer DataPtr).

Верность: fp_restored (ziwago-теги) vs fp_source. Сырой fp_source содержит
neuroweave10-теги -> raw match=False ОЖИДАЕМ (легитимный rename рефакторинга,
diag9 31/31). Нормализуем теги источника neuroweave10->ziwago -> normalized match
должен быть True = побайтная верность данных. Пик: Job commit (limited_run) +
RSS/USS (monitor). Критерий B.1: peak <= 6 ГБ, normalized match=True, без AV.
"""
from __future__ import annotations

import faulthandler
import gc
import hashlib
import json
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
# --v19 -> integrated=False (v19 mode, chemistry stays NeuroChemistry = matches the
# v19 checkpoint owner). Default integrated=True (v20). Isolates B.1 loader fidelity
# from the B.3 chemistry-owner replacement defect.
INTEGRATED = "--v19" not in sys.argv
# --restore-only -> measure the PRODUCTION load footprint (construct + load) without
# the probe-only fidelity reload, which otherwise contaminates the peak.
RESTORE_ONLY = "--restore-only" in sys.argv
_proc = psutil.Process()
_t0 = time.perf_counter()
_PEAK_RSS = 0.0
_PEAK_PRIV = 0.0
_PEAK_COMMIT = 0.0
_stop = False


def _snap():
    rss = _proc.memory_info().rss / 1e9
    try:
        priv = _proc.memory_full_info().uss / 1e9
    except Exception:
        priv = 0.0
    return rss, priv


def _commit():
    try:
        return _proc.memory_info().private / 1e9
    except Exception:
        return 0.0


def _mon():
    global _PEAK_RSS, _PEAK_PRIV, _PEAK_COMMIT
    while not _stop:
        try:
            rss, priv = _snap()
            _PEAK_RSS = max(_PEAK_RSS, rss)
            _PEAK_PRIV = max(_PEAK_PRIV, priv)
            _PEAK_COMMIT = max(_PEAK_COMMIT, _commit())
        except Exception:
            pass
        time.sleep(0.05)


def log(msg):
    rss, priv = _snap()
    print(f"[prod +{time.perf_counter()-_t0:6.1f}s rss={rss:.3f} priv={priv:.3f} "
          f"commit={_commit():.3f} peak_rss={_PEAK_RSS:.3f} peak_priv={_PEAK_PRIV:.3f} "
          f"peak_commit={_PEAK_COMMIT:.3f}] {msg}", flush=True)


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


def normalize_tags(o):
    from ziwago.cortex.moe_cortex import _LEGACY_TAG_MODULE

    def norm_tag(tag):
        if tag in _LEGACY_TAG_MODULE:
            module, name = tag.split(":", 1)
            return f"{_LEGACY_TAG_MODULE[tag]}:{name}"
        if tag.startswith("neuroweave10."):
            return "ziwago." + tag[len("neuroweave10."):]
        return tag

    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if k == "__object__" and isinstance(v, str):
                out[k] = norm_tag(v)
            else:
                out[k] = normalize_tags(v)
        return out
    if isinstance(o, list):
        return [normalize_tags(v) for v in o]
    if isinstance(o, tuple):
        return tuple(normalize_tags(v) for v in o)
    return o


def export_form_nocopy(value):
    """Зеркало export_runtime_tree, но БЕЗ clone тензоров (для верности in-place).

    state_fingerprint хеширует данные тензора (detach/cpu/reshape/view/numpy —
    views, не копия для CPU contiguous). Клон в export_runtime_tree не меняет
    хеш, но даёт +4.5 ГБ пик. Здесь тензоры остаются как есть; структура
    (__object__/__sequence__/...) идентична export -> тот же fingerprint.
    """
    if isinstance(value, torch.Tensor):
        return value
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if isinstance(value, slice):
        return {"__slice__": [value.start, value.stop, value.step]}
    if isinstance(value, torch.Generator):
        return {"__generator__": value.get_state(), "device": str(value.device)}
    if isinstance(value, (torch.dtype, torch.device)):
        return {"__torch_type__": str(value)}
    if isinstance(value, dict):
        return {k: export_form_nocopy(v) for k, v in value.items() if not callable(v)}
    if isinstance(value, list):
        return [export_form_nocopy(v) for v in value]
    if isinstance(value, (tuple, set)):
        values = sorted(value, key=repr) if isinstance(value, set) else value
        return {"__sequence__": type(value).__name__,
                "items": [export_form_nocopy(v) for v in values]}
    if callable(value):
        raise TypeError("Callables are not runtime checkpoint data")
    if hasattr(value, "__dict__"):
        cls = type(value)
        return {"__object__": f"{cls.__module__}:{cls.__name__}",
                "attributes": {k: export_form_nocopy(v) for k, v in vars(value).items()
                               if not callable(v)}}
    raise TypeError(f"Unsupported runtime state: {type(value).__name__}")


def _tensor_census(root):
    """Sum unique tensor storage bytes reachable from root (dedup by data_ptr)."""
    seen_storages = set()
    seen_objs = set()
    total = 0

    def visit(v):
        nonlocal total
        if isinstance(v, torch.Tensor):
            st = v.untyped_storage()
            key = st.data_ptr()
            if key not in seen_storages:
                seen_storages.add(key)
                total += st.nbytes()
            return
        if isinstance(v, dict):
            for x in v.values():
                visit(x)
            return
        if isinstance(v, (list, tuple, set)):
            for x in v:
                visit(x)
            return
        if v is None or isinstance(v, (bool, int, float, str, bytes)):
            return
        if hasattr(v, "__dict__"):
            if id(v) in seen_objs:
                return
            seen_objs.add(id(v))
            for x in vars(v).values():
                visit(x)

    visit(root)
    return total


def main():
    log(f"start PRODUCTION free={psutil.virtual_memory().available/1e9:.2f}GB torch={torch.__version__}")
    from ziwago.organism_core import ZiwagoOrganism, _readonly_checkpoint_mapping
    log("imports done")

    org = ZiwagoOrganism(root=ROOT, workers=2, cpu_threads=4, seed=42,
                         auto_load_checkpoint=False, integrated=INTEGRATED)
    log(f"STEP 1: organism constructed (no auto-load) integrated={INTEGRATED} "
        f"chemistry={type(org.core_organism.chemistry).__name__}")

    if RESTORE_ONLY:
        gc.collect()
        torch._C._host_emptyCache()
        log("STEP 1b: construction after gc+emptyCache (construction cache released)")
        construct_bytes = _tensor_census(org)
        log(f"STEP 1b-census: construction WHOLE-organism tensor bytes = {construct_bytes/1e9:.3f} GB")
        payload = org._load_checkpoint_payload(CKPT)
        log("STEP 2a: payload mapped READONLY")
        org._load_v19_payload(CKPT, payload)
        log("STEP 2b: _load_v19_payload done (model restored, payload still mapped)")
        src_bytes = _tensor_census(payload)
        log(f"STEP 2b-src: payload unique tensor bytes = {src_bytes/1e9:.3f} GB")
        del payload
        gc.collect()
        torch._C._host_emptyCache()
        log("STEP 2c: payload deleted + gc+emptyCache (TRUE model commit footprint)")
        nbytes = _tensor_census(org)
        log(f"STEP 2d: restored WHOLE-organism unique tensor bytes = {nbytes/1e9:.3f} GB "
            f"(checkpoint file = {CKPT.stat().st_size/1e9:.3f} GB)")
        print(f"RESULT restore_only integrated={INTEGRATED} "
              f"peak_rss_GB={_PEAK_RSS:.3f} peak_priv_GB={_PEAK_PRIV:.3f} "
              f"peak_commit_GB={_PEAK_COMMIT:.3f} model_tensor_GB={nbytes/1e9:.3f}", flush=True)
        return 0

    info = org.load_v19_checkpoint(CKPT)
    log(f"STEP 2: load_v19_checkpoint OK info={info}")
    gc.collect()
    torch._C._host_emptyCache()
    log("STEP 2b: gc + host_emptyCache after restore (release allocator cache)")

    excluded = {"root", "dialog_streamer", "moe_cortex", "spiking_language_loop"}
    live_core = {k: v for k, v in vars(org.core_organism).items()
                 if k not in excluded and not k.startswith("external_") and not callable(v)}
    fp_restored = state_fingerprint(export_form_nocopy(live_core))
    log(f"STEP 3: fp_restored={fp_restored[:16]} (no-copy in-place, BEFORE source reload)")
    del live_core
    gc.collect()
    torch._C._host_emptyCache()

    with _readonly_checkpoint_mapping():
        payload = torch.load(str(CKPT), map_location="cpu", weights_only=True, mmap=True)
    log("STEP 4: source payload re-loaded READONLY for fidelity")
    fp_source_raw = state_fingerprint(payload["core_runtime_state"]["core"])
    fp_source_norm = state_fingerprint(normalize_tags(payload["core_runtime_state"]["core"]))
    log(f"STEP 5: fp_source_raw={fp_source_raw[:16]} raw_match={fp_restored == fp_source_raw}")
    log(f"STEP 6: fp_source_norm={fp_source_norm[:16]} NORM_MATCH={fp_restored == fp_source_norm}")

    # Щит (2026-11-13): трасса перешла на семантику reinforcement_v2 (вектор
    # на строку) и legacy-матрица (372.8 МБ) СБРАСЫВАЕТСЯ при миграции.
    # Загрузчик подменяет её нулевым вектором ДО восстановления, поэтому
    # эталон для побайтного сравнения — payload после той же подмены.
    from ziwago.organism.flat import drop_legacy_shield_trace
    migrated_source = normalize_tags(payload["core_runtime_state"]["core"])
    drop_report = drop_legacy_shield_trace(migrated_source)
    fp_source_migrated = state_fingerprint(migrated_source)
    norm_match_migrated = fp_restored == fp_source_migrated
    log(f"STEP 7: legacy_trace_drop={drop_report} "
        f"fp_source_migrated={fp_source_migrated[:16]} NORM_MATCH_MIGRATED={norm_match_migrated}")
    shield = getattr(org.core_organism.dendritic_plateau_readout, "shield", None)
    if shield is not None:
        log(f"STEP 8: shield semantics={shield.u_slow_semantics} "
            f"shape={tuple(shield.u_slow.shape)} "
            f"migration={getattr(shield, 'last_migration', None)}")

    del payload
    gc.collect()
    log(f"final rss={_snap()[0]:.3f}GB peak_rss={_PEAK_RSS:.3f}GB peak_priv={_PEAK_PRIV:.3f}GB")
    print(f"RESULT production integrated={INTEGRATED} norm_match={fp_restored == fp_source_norm} "
          f"raw_match={fp_restored == fp_source_raw} "
          f"peak_rss_GB={_PEAK_RSS:.3f} peak_priv_GB={_PEAK_PRIV:.3f}", flush=True)
    return 0 if fp_restored == fp_source_norm else 3


if __name__ == "__main__":
    threading.Thread(target=_mon, daemon=True).start()
    try:
        code = main()
    except BaseException as exc:  # noqa: BLE001
        import traceback
        print(f"[prod] FAILED: {type(exc).__name__}: {exc}; "
              f"peak_rss={_PEAK_RSS:.3f}GB peak_priv={_PEAK_PRIV:.3f}GB", flush=True)
        traceback.print_exc()
        code = 1
    finally:
        _stop = True
    raise SystemExit(code)