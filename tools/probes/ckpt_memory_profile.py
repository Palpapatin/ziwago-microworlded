#!/usr/bin/env python3
"""Profile checkpoint tensor layout without loading, or measure production restore.

The default mode reads only pickle metadata. ``--restore`` uses the production
READONLY-mmap loader and is refused unless at least 10 GB of physical RAM is free.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
import threading
import time
from pathlib import Path
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROBE_DIR = Path(__file__).resolve().parent
if str(PROBE_DIR) not in sys.path:
    sys.path.insert(0, str(PROBE_DIR))

from ckpt_layout import DTYPE_BYTES, _Unpickler, _walk

_REDUNDANT_FIELDS = {
    "col_to_tok", "tok_to_col", "w_ca3_episode_links", "shield_state",
    "bank_router_state", "router_residual_bound", "vsa_encoder_state",
    "moe_cortex_state", "son_motor_state", "chemistry_state", "w_ba44_bridge",
    "son_w_basal", "son_w_apical", "w_stimulus_to_onset", "w_readout",
    "token_embed", "conversational_state", "spiking_language_state",
    "language_memory_records", "w_dend_out", "w_core_dend_out", "w_swr_dend_out",
    "w_ca3_attractors", "w_ca3_sdr", "w_ca3_salience", "sdr_column_usage",
}


def _layout(checkpoint: Path) -> dict[str, Any]:
    with zipfile.ZipFile(checkpoint) as archive:
        data_pickle = next((name for name in archive.namelist() if name.endswith("data.pkl")), None)
        if data_pickle is None:
            raise ValueError("Checkpoint does not contain data.pkl")
        prefix = data_pickle[:-len("data.pkl")]
        with archive.open(data_pickle) as stream:
            payload = _Unpickler(stream).load()
        storage_bytes = {
            entry.filename.rsplit("/", 1)[1]: entry.file_size
            for entry in archive.infolist()
            if entry.filename.startswith(prefix) and "/data/" in entry.filename
        }

    tensors: list[tuple[str, dict[str, Any]]] = []
    _walk(payload, "", tensors, set())
    rows = []
    for path, meta in tensors:
        nbytes = storage_bytes.get(meta["key"])
        if nbytes is None:
            nbytes = int(meta["numel"]) * DTYPE_BYTES.get(meta["dtype"], 4)
        path_parts = path.split(".")
        if path in _REDUNDANT_FIELDS:
            materialization = "no; flat alias is pruned before runtime restore"
        elif path_parts[-1] == "u_slow" and len(meta["shape"]) == 2:
            materialization = "no; legacy matrix is migrated to the current row trace"
        elif path.startswith("core_runtime_state."):
            materialization = "yes; authoritative runtime state"
        else:
            materialization = "overlay/metadata; retained for backward compatibility"
        rows.append({
            "attribute": path,
            "storage_key": meta["key"],
            "shape": list(meta["shape"]),
            "dtype": meta["dtype"],
            "storage_mb": round(nbytes / 1e6, 3),
            "materialization": materialization,
        })
    rows.sort(key=lambda row: row["storage_mb"], reverse=True)
    return {
        "mode": "metadata_only",
        "checkpoint": str(checkpoint),
        "tensor_count": len(rows),
        "unique_storage_count": len(storage_bytes),
        "unique_storage_gb": round(sum(storage_bytes.values()) / 1e9, 3),
        "largest_tensors": rows[:100],
        "notes": [
            "No model was constructed and no tensor storage was materialized.",
            "storage_mb is serialized storage size, not measured private-memory growth.",
            "Run with --restore on a machine with at least 10 GB free for per-attribute USS measurements.",
        ],
    }


def _private_gb(process) -> float:
    return float(process.memory_full_info().uss) / 1e9


def _runtime_restore(checkpoint: Path) -> dict[str, Any]:
    import psutil
    import torch

    available_gb = psutil.virtual_memory().available / 1e9
    if available_gb < 10.0:
        raise RuntimeError(
            f"Restore profiling requires >=10 GB free RAM; currently {available_gb:.2f} GB."
        )

    import ziwago.organism.flat as flat
    from ziwago.organism_core import ZiwagoOrganism

    process = psutil.Process()
    profile_rows: list[dict[str, Any]] = []
    original_tree = flat.load_runtime_tree
    stop_monitor = threading.Event()
    peaks = {"private_gb": 0.0, "uss_gb": 0.0}

    def monitor_memory():
        while not stop_monitor.is_set():
            try:
                info = process.memory_info()
                peaks["private_gb"] = max(
                    peaks["private_gb"], float(getattr(info, "private", info.rss)) / 1e9)
                peaks["uss_gb"] = max(peaks["uss_gb"], _private_gb(process))
            except (psutil.Error, OSError):
                pass
            stop_monitor.wait(0.05)

    def profiled_tree(current, state, _tensor_memo=None, _reuse_current_tensors=False):
        frame = sys._getframe(1)
        key = frame.f_locals.get("key")
        before = _private_gb(process)
        restored = original_tree(current, state, _tensor_memo, _reuse_current_tensors)
        after = _private_gb(process)
        if key is not None and frame.f_code.co_name == "load_runtime_state":
            profile_rows.append({
                "attribute": str(key),
                "private_before_gb": round(before, 4),
                "private_after_gb": round(after, 4),
                "private_delta_mb": round((after - before) * 1000.0, 3),
                "storage_reused": isinstance(current, torch.Tensor) and restored is current,
                "materialization": "required",
            })
        return restored

    flat.load_runtime_tree = profiled_tree
    monitor = threading.Thread(target=monitor_memory, daemon=True)
    monitor.start()
    started = time.perf_counter()
    organism = None
    payload = None
    try:
        organism = ZiwagoOrganism(root=ROOT, workers=2, cpu_threads=4, seed=42,
                                  auto_load_checkpoint=False, integrated=False)
        after_construct = _private_gb(process)
        payload = organism._load_checkpoint_payload(checkpoint)
        after_mapping = _private_gb(process)
        organism._load_v19_payload(checkpoint, payload)
        after_restore = _private_gb(process)
        del payload
        payload = None
        gc.collect()
        torch._C._host_emptyCache()
        after_release = _private_gb(process)
        return {
            "mode": "production_restore",
            "checkpoint": str(checkpoint),
            "free_ram_before_gb": round(available_gb, 3),
            "private_after_construct_gb": round(after_construct, 3),
            "private_after_readonly_mapping_gb": round(after_mapping, 3),
            "private_after_restore_gb": round(after_restore, 3),
            "private_after_payload_release_gb": round(after_release, 3),
            "peak_private_gb": round(peaks["private_gb"], 3),
            "peak_uss_gb": round(peaks["uss_gb"], 3),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "per_attribute": profile_rows,
            "notes": [
                "Private-memory deltas are process USS samples around each top-level runtime attribute.",
                "Use tools/limited_run.py for a hard Job Object memory limit.",
            ],
        }
    finally:
        stop_monitor.set()
        if "monitor" in locals():
            monitor.join(timeout=1.0)
        flat.load_runtime_tree = original_tree
        if payload is not None:
            del payload
        del organism
        gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--restore", action="store_true",
                        help="also construct and restore the production organism")
    parser.add_argument("--out", type=Path, help="write JSON report to this path")
    args = parser.parse_args()
    result = _layout(args.checkpoint)
    if args.restore:
        result["restore"] = _runtime_restore(args.checkpoint)

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
