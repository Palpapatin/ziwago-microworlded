#!/usr/bin/env python3
"""Measure fresh v20 construction memory without loading or saving checkpoints."""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROFILED_CLASSES = {
    "ZiwagoOrganism",
    "ZiwagoPureSynapticDialogueOrganismV14",
    "SparselyGatedCorticalMoE",
    "CorticalColumn",
    "GenesisSynapse",
    "DendriticTree",
    "SONReflectiveColumn",
    "PredictiveCodingCorticalReadout",
    "DendriticPlateauMatrixReadout",
}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    from ziwago_assistant import ZiwagoBiologicalSNNAssistant

    process = psutil.Process()
    rows: list[dict] = []
    starts: dict[int, tuple[str, float]] = {}
    peaks = {"uss_gb": 0.0, "rss_gb": 0.0}
    stop_monitor = threading.Event()

    def private_gb() -> float:
        return float(process.memory_full_info().uss) / 1e9

    def tensor_inventory(instance) -> list[dict]:
        tensors = [
            {"field": name, "shape": list(value.shape),
             "dtype": str(value.dtype), "size_mb": round(
                 value.numel() * value.element_size() / 1_000_000.0, 2)}
            for name, value in vars(instance).items()
            if hasattr(value, "numel") and hasattr(value, "element_size")
        ]
        return sorted(tensors, key=lambda item: item["size_mb"], reverse=True)[:12]

    def monitor_memory() -> None:
        while not stop_monitor.is_set():
            try:
                info = process.memory_info()
                peaks["rss_gb"] = max(peaks["rss_gb"], float(info.rss) / 1e9)
                peaks["uss_gb"] = max(peaks["uss_gb"], private_gb())
            except (psutil.Error, OSError):
                pass
            stop_monitor.wait(0.05)

    def profile(frame, event, _arg) -> None:
        instance = frame.f_locals.get("self")
        if instance is None:
            return
        class_name = type(instance).__name__
        frame_id = id(frame)
        if event == "call" and frame.f_code.co_name == "__init__" and class_name in PROFILED_CLASSES:
            starts[frame_id] = (class_name, private_gb())
        elif event == "return" and frame_id in starts:
            name, before = starts.pop(frame_id)
            after = private_gb()
            row = {"class": name, "uss_before_gb": round(before, 4),
                   "uss_after_gb": round(after, 4),
                   "uss_delta_mb": round((after - before) * 1000.0, 2)}
            if name in {"CorticalColumn", "GenesisSynapse"}:
                row["largest_tensor_fields"] = tensor_inventory(instance)
            rows.append(row)

    monitor = threading.Thread(target=monitor_memory, daemon=True)
    assistant = None
    started = time.perf_counter()
    monitor.start()
    sys.setprofile(profile)
    try:
        assistant = ZiwagoBiologicalSNNAssistant(
            root=ROOT, checkpoint_path=None, workers=1, cpu_threads=4,
            seed=20261115, auto_load_checkpoint=False, runtime_version=20,
        )
    except Exception as exc:
        sys.setprofile(None)
        stop_monitor.set()
        monitor.join(timeout=1.0)
        print(json.dumps({
            "stage": "construction_failed",
            "exception": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "peak_uss_gb": round(peaks["uss_gb"], 3),
            "peak_rss_gb": round(peaks["rss_gb"], 3),
            "constructors_in_progress": [
                {"class": name, "uss_at_start_gb": round(before, 4)}
                for name, before in starts.values()
            ],
            "largest_constructor_deltas": sorted(
                rows, key=lambda row: row["uss_delta_mb"], reverse=True)[:30],
        }, ensure_ascii=False), flush=True)
        return 1
    finally:
        sys.setprofile(None)
        stop_monitor.set()
        monitor.join(timeout=1.0)

    print(json.dumps({
        "stage": "construction_complete",
        "loaded_checkpoint": bool(assistant.loaded_checkpoint_info),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "peak_uss_gb": round(peaks["uss_gb"], 3),
        "peak_rss_gb": round(peaks["rss_gb"], 3),
        "largest_constructor_deltas": sorted(
            rows, key=lambda row: row["uss_delta_mb"], reverse=True)[:30],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
