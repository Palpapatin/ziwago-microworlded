"""B.3 production fidelity: v19-чекпоинт в едином владельце OrganismPhysiology.

B.3 меняет владельца химии при рождении (NeuroChemistry -> OrganismPhysiology).
Полный NORM_MATCH (restored == source) больше НЕ ожидается: живая химия легитимно
содержит v20-поля, отсутствующие в v19-источнике. Верность загрузчика B.1 после B.3
доказывается четырьмя замерами:
  (A) NOCHEM_MATCH: всё кроме химии побайтно совпадает с нормализованным источником.
  (B) CHEM_SUBSET: состояние химии-источника сохранено поле-в-поле (0 расхождений);
      v20-поля — только добавления, не потери.
  (C) CHEM_ROUNDTRIP: export -> load -> export химии побайтно идентичен.
  (D) Пик USS <= 6 ГБ, без AV/OOM.
Метод fingerprint идентичен B.1 (импорт из b1_production_probe).
"""
from __future__ import annotations

import faulthandler
import gc
import sys
import threading
import time
from pathlib import Path

faulthandler.enable(all_threads=True)
ROOT = Path(r"C:/Users/yarma/Desktop/Ziwago18")
PRODUCE = Path(r"D:/Ziwago18-storage/Produce")
for p in (str(ROOT), str(PRODUCE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
import psutil  # noqa: E402

torch.set_num_threads(4)
CKPT = PRODUCE / "checkpoints" / "ziwago_v19_apical2_20261109.pt"

from b1_production_probe import (  # noqa: E402
    state_fingerprint, normalize_tags, export_form_nocopy,
)

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
    print(f"[b3 +{time.perf_counter()-_t0:6.1f}s rss={rss:.3f} priv={priv:.3f} "
          f"commit={_commit():.3f} peak_rss={_PEAK_RSS:.3f} peak_priv={_PEAK_PRIV:.3f} "
          f"peak_commit={_PEAK_COMMIT:.3f}] {msg}", flush=True)


def subset_match(source, restored, path="chem"):
    """Source state must be preserved in restored; restored may add fields.

    Игнорирует строку тега __object__ (намеренная смена владельца NeuroChemistry ->
    OrganismPhysiology / ChemState -> V20ChemState), сравнивает ЗНАЧЕНИЯ полей.
    Возвращает (mismatches, additions).
    """
    mismatches = []
    additions = []

    def walk(s, r, p):
        if isinstance(s, torch.Tensor):
            if not isinstance(r, torch.Tensor) or s.dtype != r.dtype or s.shape != r.shape \
                    or not torch.equal(s.detach().cpu(), r.detach().cpu()):
                mismatches.append((p, "tensor", "differs"))
            return
        if isinstance(s, dict):
            if not isinstance(r, dict):
                mismatches.append((p, "dict", type(r).__name__))
                return
            if "__object__" in s:
                sa = s.get("attributes", {})
                ra = r.get("attributes", {}) if isinstance(r, dict) else {}
                for k in ra:
                    if k not in sa:
                        additions.append(f"{p}.{k}")
                for k, v in sa.items():
                    if k not in ra:
                        mismatches.append((f"{p}.{k}", "present", "MISSING"))
                    else:
                        walk(v, ra[k], f"{p}.{k}")
                return
            if "__sequence__" in s:
                si, ri = s.get("items", []), r.get("items", [])
                if len(si) != len(ri):
                    mismatches.append((p, f"len{len(si)}", f"len{len(ri)}"))
                for i, (a, b) in enumerate(zip(si, ri)):
                    walk(a, b, f"{p}[{i}]")
                return
            for k, v in s.items():
                if k == "__object__":
                    continue
                if k not in r:
                    mismatches.append((f"{p}.{k}", "present", "MISSING"))
                else:
                    walk(v, r[k], f"{p}.{k}")
            return
        if isinstance(s, (list, tuple)):
            if not isinstance(r, (list, tuple)) or len(s) != len(r):
                mismatches.append((p, f"seq{len(s)}", f"seq{len(r) if isinstance(r,(list,tuple)) else '?'}"))
                return
            for i, (a, b) in enumerate(zip(s, r)):
                walk(a, b, f"{p}[{i}]")
            return
        if s != r:
            mismatches.append((p, repr(s), repr(r)))

    walk(source, restored, path)
    return mismatches, additions


def main():
    log(f"start B.3 fidelity free={psutil.virtual_memory().available/1e9:.2f}GB torch={torch.__version__}")
    from ziwago.organism_core import ZiwagoOrganism, _readonly_checkpoint_mapping
    from ziwago.organism.physiology import OrganismPhysiology
    from ziwago.cortex.moe_cortex import export_runtime_tree, load_runtime_tree
    log("imports done")

    org = ZiwagoOrganism(root=ROOT, workers=2, cpu_threads=4, seed=42,
                         auto_load_checkpoint=False, integrated=False)
    chem_owner = type(org.core_organism.chemistry).__name__
    log(f"STEP 1: organism constructed (no auto-load) integrated=False chemistry={chem_owner}")

    info = org.load_v19_checkpoint(CKPT)
    log(f"STEP 2: load_v19_checkpoint OK info={info}")
    gc.collect()
    torch._C._host_emptyCache()

    excluded = {"root", "dialog_streamer", "moe_cortex", "spiking_language_loop"}
    live_core = {k: v for k, v in vars(org.core_organism).items()
                 if k not in excluded and not k.startswith("external_") and not callable(v)}
    fp_restored_full = state_fingerprint(export_form_nocopy(live_core))
    log(f"STEP 3: fp_restored_full={fp_restored_full[:16]} (live core incl. OrganismPhysiology)")

    with _readonly_checkpoint_mapping():
        payload = torch.load(str(CKPT), map_location="cpu", weights_only=True, mmap=True)
    log("STEP 4: source payload re-loaded READONLY")
    source_norm = normalize_tags(payload["core_runtime_state"]["core"])
    fp_source_full = state_fingerprint(source_norm)
    log(f"STEP 5: fp_source_full={fp_source_full[:16]} FULL_MATCH={fp_restored_full == fp_source_full} "
        f"(ожидается False: смена владельца химии)")

    # (A) everything except chemistry
    live_nochem = {k: v for k, v in live_core.items() if k != "chemistry"}
    src_nochem = {k: v for k, v in source_norm.items() if k != "chemistry"}
    fp_restored_nochem = state_fingerprint(export_form_nocopy(live_nochem))
    fp_source_nochem = state_fingerprint(src_nochem)
    nochem_match = fp_restored_nochem == fp_source_nochem
    log(f"STEP 6 (A): fp_restored_nochem={fp_restored_nochem[:16]} fp_source_nochem={fp_source_nochem[:16]} "
        f"NOCHEM_MATCH={nochem_match}")

    # (B) chemistry source-state subset preservation
    src_chem = source_norm["chemistry"]
    restored_chem = export_form_nocopy(org.core_organism.chemistry)
    mismatches, additions = subset_match(src_chem, restored_chem)
    log(f"STEP 7 (B): CHEM_SUBSET mismatches={len(mismatches)} additions={len(additions)} "
        f"src_tag={src_chem.get('__object__')} restored_tag={restored_chem.get('__object__')}")
    for m in mismatches[:20]:
        log(f"  MISMATCH {m}")
    log(f"  v20 additions (sample): {additions[:18]}")

    # (C) chemistry roundtrip byte-identical
    chem_src = export_runtime_tree(org.core_organism.chemistry)
    fp_chem_src = state_fingerprint(chem_src)
    fresh = OrganismPhysiology(org.core_organism.moe_cortex.n_columns)
    rt = load_runtime_tree(fresh, chem_src)
    fp_chem_rt = state_fingerprint(export_runtime_tree(rt))
    chem_roundtrip = fp_chem_src == fp_chem_rt
    log(f"STEP 8 (C): fp_chem_src={fp_chem_src[:16]} fp_chem_rt={fp_chem_rt[:16]} "
        f"CHEM_ROUNDTRIP={chem_roundtrip}")

    del payload, live_core, live_nochem, src_nochem, chem_src, rt
    gc.collect()
    torch._C._host_emptyCache()
    log(f"final rss={_snap()[0]:.3f}GB peak_rss={_PEAK_RSS:.3f}GB peak_priv={_PEAK_PRIV:.3f}GB "
        f"peak_commit={_PEAK_COMMIT:.3f}GB")

    ok = nochem_match and len(mismatches) == 0 and chem_roundtrip and _PEAK_PRIV <= 6.0
    print(f"RESULT B.3_fidelity integrated=False owner={chem_owner} "
          f"nochem_match={nochem_match} chem_subset_mismatches={len(mismatches)} "
          f"chem_additions={len(additions)} chem_roundtrip={chem_roundtrip} "
          f"full_match={fp_restored_full == fp_source_full} "
          f"peak_rss_GB={_PEAK_RSS:.3f} peak_priv_GB={_PEAK_PRIV:.3f} peak_commit_GB={_PEAK_COMMIT:.3f} "
          f"ALL_PASS={ok}", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    threading.Thread(target=_mon, daemon=True).start()
    try:
        code = main()
    except BaseException as exc:  # noqa: BLE001
        import traceback
        print(f"[b3] FAILED: {type(exc).__name__}: {exc}; "
              f"peak_rss={_PEAK_RSS:.3f}GB peak_priv={_PEAK_PRIV:.3f}GB", flush=True)
        traceback.print_exc()
        code = 1
    finally:
        _stop = True
    raise SystemExit(code)