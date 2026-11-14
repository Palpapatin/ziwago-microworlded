"""Замер патологии метапластического щита на реальном чекпоинте.

Читает из zip-контейнера torch ТОЛЬКО нужные storages (по ключу из pickle),
не поднимая модель целиком: память ~750 МБ вместо 4.5 ГБ.

Что измеряется
--------------
1. СТАРАЯ семантика (`magnitude_v1`, матрица u_slow): какая доля строк
   заморожена (attenuation = 1/(1+κ·u_slow) < 1e-3) и как |w| замороженных
   строк соотносится с |w| живых. Патология 2026-11-12: замороженные строки
   имели МЕДИАННЫЙ |w| ВЫШЕ живых (1.597 против 1.214) — щит замораживал
   именно накопленный смысл.
2. НОВАЯ семантика (`reinforcement_v2`, вектор на строку): что получается
   после миграции (трасса обнуляется) и какой потолок затухания задан
   `trace_cap`.

Запуск:
    python tools/probes/shield_pathtology_probe.py <checkpoint.pt> [--kappa 18.0]
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.probes.ckpt_layout import _StorageStub, _Unpickler, _walk  # noqa: E402

DTYPES = {
    "FloatStorage": torch.float32, "DoubleStorage": torch.float64,
    "HalfStorage": torch.float16, "BFloat16Storage": torch.bfloat16,
    "LongStorage": torch.int64, "IntStorage": torch.int32,
    "ByteStorage": torch.uint8, "BoolStorage": torch.bool,
}


def _tensor_meta(obj: Dict[str, Any], path: str, wanted: str):
    """Найти тензор по подстроке пути: вернуть (storage_key, dtype, size, stride, offset)."""
    found: List[Tuple[str, Dict[str, Any]]] = []
    _walk(obj, "", found, set())
    hits = [(p, m) for p, m in found if wanted.lower() in p.lower()]
    if not hits:
        raise SystemExit(f"тензор '{wanted}' не найден")
    # Предпочитаем «плоский» путь (без core_runtime_state) — он короче.
    hits.sort(key=lambda pm: (pm[0].startswith("core_runtime_state"), len(pm[0])))
    path_name, meta = hits[0]
    return path_name, meta


def _materialize(zf: zipfile.ZipFile, prefix: str, key: str, dtype_name: str,
                 shape: Tuple[int, ...], stride: Tuple[int, ...], offset: int) -> torch.Tensor:
    name = f"{prefix}data/{key}"
    raw = zf.read(name)
    dtype = DTYPES.get(dtype_name)
    if dtype is None:
        raise SystemExit(f"неизвестный dtype {dtype_name}")
    buf = torch.frombuffer(bytearray(raw), dtype=dtype)
    if offset:
        buf = buf[offset:]
    if len(stride) == 1:
        return buf[: shape[0]].clone()
    if len(stride) == 2 and stride[1] == 1:
        rows, cols = shape
        if stride[0] == cols:
            return buf[: rows * cols].reshape(rows, cols).clone()
        # Срез столбцов: хранилище шире, чем представление (stride[0] > cols).
        if stride[0] >= cols:
            full = buf[: rows * stride[0]].reshape(rows, stride[0])
            return full[:, :cols].clone()
    raise SystemExit(f"неподдерживаемая геометрия shape={shape} stride={stride}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--kappa", type=float, default=18.0)
    ap.add_argument("--trace-cap", type=float, default=4.0)
    ap.add_argument("--frozen-threshold", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    with zipfile.ZipFile(ckpt) as zf:
        pkl_name = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        prefix = pkl_name[: -len("data.pkl")]
        with zf.open(pkl_name) as fh:
            obj = _Unpickler(fh).load()

        slow_path, slow_meta = _tensor_meta(obj, "u_slow", "shield_state.attributes.u_slow")
        w_path, w_meta = _tensor_meta(obj, "w_dend_out", "w_dend_out")
        print(f"u_slow: {slow_path} shape={slow_meta['shape']} key={slow_meta['key']}")
        print(f"w_dend_out: {w_path} shape={w_meta['shape']} key={w_meta['key']}")

        u_slow = _materialize(zf, prefix, slow_meta["key"], slow_meta["dtype"],
                              tuple(slow_meta["shape"]), tuple(slow_meta["stride"]), 0)
        w = _materialize(zf, prefix, w_meta["key"], w_meta["dtype"],
                         tuple(w_meta["shape"]), tuple(w_meta["stride"]), 0)

    report: Dict[str, Any] = {"checkpoint": str(ckpt), "kappa_shield": args.kappa}

    # ── 1. Старая семантика: матрица (d_dend, n_cols) с историей величины ──
    if u_slow.ndim == 2:
        attenuation = 1.0 / (1.0 + args.kappa * u_slow)
        row_atten = attenuation.min(dim=1).values          # строка заморожена, если хоть одна колонка заморожена
        row_frozen = row_atten < args.frozen_threshold
        w_row = w.abs().mean(dim=1)
        frozen_w = w_row[row_frozen]
        alive_w = w_row[~row_frozen]
        report["legacy"] = {
            "layout": "matrix",
            "shape": list(u_slow.shape),
            "u_slow_max": float(u_slow.max()),
            "u_slow_mean": float(u_slow.mean()),
            "frozen_rows": int(row_frozen.sum()),
            "rows": int(row_frozen.numel()),
            "frozen_fraction": float(row_frozen.float().mean()),
            "median_abs_w_frozen": float(frozen_w.median()) if frozen_w.numel() else None,
            "median_abs_w_alive": float(alive_w.median()) if alive_w.numel() else None,
            "mean_abs_w_frozen": float(frozen_w.mean()) if frozen_w.numel() else None,
            "mean_abs_w_alive": float(alive_w.mean()) if alive_w.numel() else None,
        }
        if frozen_w.numel() and alive_w.numel():
            report["legacy"]["median_ratio_frozen_over_alive"] = (
                float(frozen_w.median()) / float(alive_w.median()))
        del attenuation, row_atten
    else:
        report["legacy"] = {"layout": "row_trace", "shape": list(u_slow.shape)}

    # ── 2. Новая семантика: что будет после миграции ─────────────────────
    floor = 1.0 / (1.0 + args.kappa * args.trace_cap)
    report["new_semantics"] = {
        "semantics": "reinforcement_v2",
        "layout": "row_trace",
        "trace_bytes": int(u_slow.shape[0] * 4) if u_slow.ndim == 1 else int(u_slow.shape[0] * 4),
        "legacy_trace_bytes": int(u_slow.numel() * 4),
        "attenuation_floor": floor,
        "note": ("legacy-трасса копила величину веса → в новой семантике она "
                 "сбрасывается (отморозка ошибочно замороженного)"),
    }
    del u_slow
    del w

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"записано: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
