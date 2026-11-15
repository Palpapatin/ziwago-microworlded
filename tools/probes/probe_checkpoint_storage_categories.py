#!/usr/bin/env python3
"""Read-only inventory of unique checkpoint storage by candidate E-7 category."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CHECKPOINT = Path(
    r"D:\Ziwago18-storage\Produce\checkpoints\ziwago_v19_bankfixed_20261109.pt"
)


def _category(path: tuple[object, ...]) -> str:
    names = [str(part).lower() for part in path]
    key = names[-1] if names else ""
    if (key == "w_epoch0_snapshot" or key == "branch_kernel"
            or key in {"eligibility", "het_trace", "u_slow"}
            or key.startswith(("si_", "pnn_", "spine_"))):
        return "plasticity_candidate"
    if (key.startswith(("last_", "_last_", "_loc_", "cum_"))
            or "telemetry" in key or "raster" in key or "history" in key):
        return "telemetry_candidate"
    return "core_candidate"


def _walk(value, path, seen, totals) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        storage = value.untyped_storage()
        pointer = storage.data_ptr()
        if pointer not in seen:
            category = _category(path)
            size = storage.nbytes()
            seen[pointer] = (category, size, ".".join(map(str, path)))
            totals[category] += size
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _walk(child, path + (key,), seen, totals)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk(child, path + (index,), seen, totals)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    from ziwago.organism_core import _readonly_checkpoint_mapping
    import torch

    with _readonly_checkpoint_mapping():
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)

    totals = {
        "core_candidate": 0,
        "plasticity_candidate": 0,
        "telemetry_candidate": 0,
    }
    seen = {}
    preferred_roots = ("core_runtime_state", "v19_state", "v20_state", "genesis_v18_state")
    for key in preferred_roots:
        if key in payload:
            _walk(payload[key], (key,), seen, totals)
    for key, value in payload.items():
        if key not in preferred_roots:
            _walk(value, (key,), seen, totals)

    total = sum(totals.values())
    largest = sorted(
        (
            {"category": category, "bytes": size, "path": path}
            for category, size, path in seen.values()
        ),
        key=lambda item: (-item["bytes"], item["path"]),
    )[:20]
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "unique_storage_count": len(seen),
        "unique_storage_bytes": total,
        "categories": {
            key: {
                "bytes": size,
                "gib": round(size / (1024 ** 3), 3),
                "percent": round(100.0 * size / total, 2) if total else 0.0,
            }
            for key, size in totals.items()
        },
        "largest_storages": largest,
        "classification_note": (
            "Field-name heuristic for sizing only; not a validated evaluation split."
        ),
    }
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
