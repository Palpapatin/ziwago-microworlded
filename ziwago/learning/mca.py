"""Metaplastic Credit Assignment (MCA): per-synapse susceptibility m in [0, 1].

Blueprint MATH-MCA-001 / §3.2, with the blueprint's bound correction: m relaxes
exponentially toward a *bounded* activity-target instead of integrating the raw
unbounded (w - w0)^2 source. The activity proxy of synapse (r, c) is its squared
contribution to the local logit, (d_r * w_rc)^2, saturating through a half-saturation
constant. The BCM-style threshold theta0 * (1 + beta_m * m) requires theta0 > 0 to
have any effect. Continuous relaxation is normalized by dt; spike impulses are
added once and clamped, never dt-scaled. Storage is budgeted and evictions reported.
No global backprop.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch


class MetaplasticSusceptibility:
    def __init__(self, max_entries: int = 4096, tau_m: float = 1000.0,
                 beta_m: float = 4.0, theta0: float = 0.1, half_sat: float = 1.0):
        if (int(max_entries) <= 0 or not math.isfinite(tau_m) or tau_m <= 0
                or not math.isfinite(beta_m) or beta_m < 0
                or not math.isfinite(theta0) or theta0 <= 0
                or not math.isfinite(half_sat) or half_sat <= 0):
            raise ValueError("MCA requires max_entries>0, tau_m>0, beta_m>=0, theta0>0, half_sat>0.")
        self.max_entries = int(max_entries)
        self.tau_m = float(tau_m)
        self.beta_m = float(beta_m)
        self.theta0 = float(theta0)
        self.half_sat = float(half_sat)
        self.m: Dict[Tuple[int, int], float] = {}
        self.observations = 0
        self.evicted_entries = 0
        self.manual_kicks = 0

    def activity_target(self, contribution_sq: torch.Tensor) -> torch.Tensor:
        return contribution_sq / (contribution_sq + self.half_sat)

    def threshold(self, m: torch.Tensor) -> torch.Tensor:
        return self.theta0 * (1.0 + self.beta_m * m)

    def observe(self, w: torch.Tensor, d_rows: torch.Tensor, error: torch.Tensor,
                rows: Sequence[int], target_col: int, dt: float,
                top_columns: int = 32) -> Dict[str, int]:
        """Observe local activity; weights, drives and errors must be finite and aligned."""
        if (w.ndim != 2 or error.ndim != 1 or d_rows.ndim != 1
                or not bool(torch.isfinite(w).all())
                or not bool(torch.isfinite(error).all())
                or not bool(torch.isfinite(d_rows).all())):
            raise ValueError("w must be finite and 2D; error/drive must be finite 1D tensors.")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")
        rows_t = torch.as_tensor(list(rows), dtype=torch.long, device=w.device)
        if d_rows.numel() != rows_t.numel():
            raise ValueError("rows and d_rows must have the same length.")
        if rows_t.numel() == 0:
            return {"selected": 0, "evicted": 0}
        # error покрывает ВЕДУЩИЕ (активные) столбцы w; ёмкость w может быть шире.
        if error.numel() > w.shape[1] or not 0 <= int(target_col) < error.numel():
            raise ValueError("error must cover the leading columns of w and target_col must be inside it.")
        if bool(((rows_t < 0) | (rows_t >= w.shape[0])).any()):
            raise ValueError("rows must index existing rows of w.")
        if error.device != w.device or d_rows.device != w.device:
            raise ValueError("w, error and d_rows must be on the same device.")
        k = min(max(1, int(top_columns)), error.numel())
        cols = torch.topk(error.abs(), k=k).indices
        if int(target_col) not in cols.tolist():
            cols = torch.cat((cols, torch.tensor([int(target_col)], dtype=torch.long)))
        w_sel = w[rows_t.unsqueeze(1), cols.unsqueeze(0)]
        contrib_sq = (d_rows.unsqueeze(1) * w_sel).square()
        target = self.activity_target(contrib_sq)
        decay = math.exp(-float(dt) / self.tau_m)
        flat_target = target.flatten()
        keep = min(self.max_entries, flat_target.numel())
        sel = torch.topk(flat_target, k=keep).indices
        row_sel = rows_t[sel // cols.numel()]
        col_sel = cols[sel % cols.numel()]
        added = 0
        for r, c, t in zip(row_sel.tolist(), col_sel.tolist(), flat_target[sel].tolist()):
            key = (int(r), int(c))
            old = self.m.get(key)
            if old is None:
                added += 1
                old = 0.0
            self.m[key] = float(t) + (old - float(t)) * decay
        self.observations += 1
        evicted = 0
        if len(self.m) > self.max_entries:
            ordered = sorted(self.m.items(), key=lambda kv: kv[1])
            for key, _ in ordered[: len(self.m) - self.max_entries]:
                del self.m[key]
                evicted += 1
            self.evicted_entries += evicted
        return {"selected": int(keep), "added": added, "evicted": evicted}

    def spike_kick(self, keys: Sequence[Tuple[int, int]], amount: float) -> int:
        """Manual experimental impulse API; no live spike source calls this method."""
        if not math.isfinite(amount):
            raise ValueError("spike kick must be finite.")
        applied = 0
        for key in keys:
            key = (int(key[0]), int(key[1]))
            old = self.m.get(key, 0.0)
            self.m[key] = max(0.0, min(1.0, old + float(amount)))
            applied += 1
        self.manual_kicks += applied
        return applied

    def row_susceptibility(self, rows: Sequence[int], n_active: int) -> torch.Tensor:
        rows_t = torch.as_tensor(list(rows), dtype=torch.long)
        agg: Dict[int, float] = {}
        for (r, c), value in self.m.items():
            if 0 <= c < n_active:
                agg[r] = max(agg.get(r, 0.0), value)
        gains = [1.0 / (1.0 + self.beta_m * agg.get(int(r), 0.0)) for r in rows_t.tolist()]
        return torch.tensor(gains, dtype=torch.float32)

    def telemetry(self) -> Dict[str, object]:
        values = list(self.m.values())
        return {
            "entries": len(self.m), "max_entries": self.max_entries,
            "observations": self.observations, "evicted_entries": self.evicted_entries,
            "manual_kicks": self.manual_kicks,
            "mean_m": sum(values) / len(values) if values else 0.0,
            "max_m": max(values) if values else 0.0,
            "theta0": self.theta0, "beta_m": self.beta_m, "tau_m": self.tau_m,
        }

    def export_state_dict(self) -> Dict[str, object]:
        return {
            "max_entries": self.max_entries, "tau_m": self.tau_m, "beta_m": self.beta_m,
            "theta0": self.theta0, "half_sat": self.half_sat,
            "observations": self.observations, "evicted_entries": self.evicted_entries,
            "manual_kicks": self.manual_kicks,
            "m": [[r, c, v] for (r, c), v in self.m.items()],
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        loaded = type(self)(int(state["max_entries"]), float(state["tau_m"]),
                            float(state["beta_m"]), float(state["theta0"]), float(state["half_sat"]))
        loaded.observations = int(state.get("observations", 0))
        loaded.evicted_entries = int(state.get("evicted_entries", 0))
        loaded.manual_kicks = int(state.get("manual_kicks", 0))
        for r, c, v in state["m"]:
            loaded.m[(int(r), int(c))] = float(v)
        if len(loaded.m) > loaded.max_entries:
            raise ValueError("MCA state exceeds its declared budget.")
        self.__dict__.update(loaded.__dict__)