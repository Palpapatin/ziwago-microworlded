"""Budgeted local Fisher importance and anchor regularization.

No global backprop: the score gradient of a local categorical scorer is
g_rv = a * d_r * (1[y=v] - p_v). Importance is an exponentially weighted
estimate of g^2 over observed active synapses; anchors are captured when an
entry first becomes important. Storage is capped and evictions are reported.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch


class SparseFisherAnchor:
    def __init__(self, max_entries: int = 4096, decay: float = 0.9):
        if int(max_entries) <= 0 or not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("max_entries must be positive and decay in [0, 1).")
        self.max_entries = int(max_entries)
        self.decay = float(decay)
        self.fisher: Dict[Tuple[int, int], float] = {}
        self.anchor: Dict[Tuple[int, int], float] = {}
        self.observations = 0
        self.evicted_entries = 0

    @staticmethod
    def score_gradient(d_rows: torch.Tensor, error: torch.Tensor, a: float) -> torch.Tensor:
        return float(a) * d_rows.detach().float().unsqueeze(1) * error.detach().float().unsqueeze(0)

    @staticmethod
    def conditional_fisher(d_rows: torch.Tensor, probs: torch.Tensor, a: float) -> torch.Tensor:
        drive = (float(a) * d_rows.detach().float()).square().unsqueeze(1)
        p = probs.detach().float().unsqueeze(0)
        return drive * p * (1.0 - p)

    def observe(self, w: torch.Tensor, d_rows: torch.Tensor, error: torch.Tensor, a: float,
                rows: Sequence[int], target_col: int, top_columns: int = 32) -> Dict[str, int]:
        """Observe one local categorical update; all scores, drives and weights must be finite."""
        if (w.ndim != 2 or error.ndim != 1 or d_rows.ndim != 1
                or not bool(torch.isfinite(w).all())
                or not bool(torch.isfinite(error).all())
                or not bool(torch.isfinite(d_rows).all())
                or not math.isfinite(a)):
            raise ValueError("w must be finite and 2D; error/drive must be finite 1D tensors; scale finite.")
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
        g = self.score_gradient(d_rows, error[cols], a)
        flat = g.abs().flatten()
        keep = min(self.max_entries, flat.numel())
        sel = torch.topk(flat, k=keep).indices
        row_sel = rows_t[sel // cols.numel()]
        col_sel = cols[sel % cols.numel()]
        g_sel = g.flatten()[sel]
        added = 0
        for r, c, gv in zip(row_sel.tolist(), col_sel.tolist(), g_sel.tolist()):
            key = (int(r), int(c))
            value = float(gv) * float(gv)
            old = self.fisher.get(key)
            self.fisher[key] = value if old is None else self.decay * old + (1.0 - self.decay) * value
            if key not in self.anchor:
                self.anchor[key] = float(w[int(r), int(c)].detach().item())
                added += 1
        self.observations += 1
        evicted = 0
        if len(self.fisher) > self.max_entries:
            ordered = sorted(self.fisher.items(), key=lambda kv: kv[1])
            for key, _ in ordered[: len(self.fisher) - self.max_entries]:
                del self.fisher[key]
                self.anchor.pop(key, None)
                evicted += 1
            self.evicted_entries += evicted
        return {"selected": int(keep), "added": added, "evicted": evicted}

    def proximal_pull(self, w: torch.Tensor, eta: float, lambda_reg: float) -> Dict[str, float]:
        if not math.isfinite(eta) or eta < 0 or not math.isfinite(lambda_reg) or lambda_reg < 0:
            raise ValueError("eta and lambda_reg must be finite and nonnegative.")
        applied, max_disp = 0, 0.0
        strength = float(eta) * float(lambda_reg)
        if strength == 0.0:
            return {"applied": 0, "max_displacement": 0.0}
        with torch.no_grad():
            for (r, c), importance in self.fisher.items():
                if importance <= 0.0:
                    continue
                anchor = self.anchor[(r, c)]
                old = float(w[r, c].item())
                denom = 1.0 + strength * importance
                new = (old + strength * importance * anchor) / denom
                w[r, c] = new
                applied += 1
                max_disp = max(max_disp, abs(new - old))
        return {"applied": applied, "max_displacement": max_disp}

    def refresh_anchor(self, w: torch.Tensor) -> int:
        with torch.no_grad():
            for (r, c) in self.fisher:
                self.anchor[(r, c)] = float(w[r, c].item())
        return len(self.fisher)

    def telemetry(self) -> Dict[str, object]:
        values = list(self.fisher.values())
        return {
            "entries": len(self.fisher), "max_entries": self.max_entries,
            "observations": self.observations, "evicted_entries": self.evicted_entries,
            "mean_fisher": sum(values) / len(values) if values else 0.0,
            "max_fisher": max(values) if values else 0.0,
        }

    def export_state_dict(self) -> Dict[str, object]:
        return {
            "max_entries": self.max_entries, "decay": self.decay,
            "observations": self.observations, "evicted_entries": self.evicted_entries,
            "fisher": [[r, c, f] for (r, c), f in self.fisher.items()],
            "anchor": [[r, c, a] for (r, c), a in self.anchor.items()],
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        loaded = type(self)(int(state["max_entries"]), float(state["decay"]))
        loaded.observations = int(state.get("observations", 0))
        loaded.evicted_entries = int(state.get("evicted_entries", 0))
        for r, c, f in state["fisher"]:
            loaded.fisher[(int(r), int(c))] = float(f)
        for r, c, a in state["anchor"]:
            loaded.anchor[(int(r), int(c))] = float(a)
        if len(loaded.fisher) > loaded.max_entries:
            raise ValueError("Fisher state exceeds its declared budget.")
        self.__dict__.update(loaded.__dict__)