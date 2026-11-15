"""Найти расхождение в runtime-снимке после восстановления (round-trip)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "tests"))
from test_v19_spiking_language import tiny_core  # noqa: E402


def diff(a, b, path="core", out=None):
    if out is None:
        out = []
    if isinstance(a, torch.Tensor):
        if not torch.equal(a, b):
            out.append((path, "tensor", "max|d|=" + str(
                float((a - b).abs().max()) if a.shape == b.shape else "shape")))
        return out
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                out.append((f"{path}.{key}", "только в B", ""))
            elif key not in b:
                out.append((f"{path}.{key}", "только в A", ""))
            else:
                diff(a[key], b[key], f"{path}.{key}", out)
        return out
    if isinstance(a, (list, tuple)):
        for index, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{index}]", out)
        return out
    if a != b:
        out.append((path, repr(a), repr(b)))
    return out


def main() -> int:
    torch.set_num_threads(1)
    core = tiny_core()
    core.enable_integrated_organism()
    x = torch.ones(64) * 0.2
    before = core.export_runtime_state()
    for _ in range(core.life.cfg.ticks_per_token):
        core.life.tick(core, x)
    expected = core.export_runtime_state()
    core.load_runtime_state(before)
    core.life.present(core, x)
    actual = core.export_runtime_state()
    rows = diff(expected, actual)
    print(f"расхождений: {len(rows)}")
    for path, a, b in rows[:25]:
        print(f"  {path}: A={a} B={b}")
    return 1 if rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
