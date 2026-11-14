"""Developmental Curriculum Controller (DCC), blueprint §3.5 / ARCH-DCC-001.

Per-zone ontogeny with the blueprint's corrections to the card:
- age prior m_z = 1 - exp(-t/tau_z) orders capabilities, but closure is decided by
  an independent readiness criterion, not by age alone;
- the hyperplasticity window eta = eta_max * 4 m (1-m) is zero at m=0 and m=1 by
  construction (no hidden floor); the adaptation reserve is an explicit plastic
  quota, not a silent eta clamp;
- the E/I score is bounded, exp(-|log((E+eps)/(I+eps))|), and gated by activity so
  that E=I=0 (silence) is NOT treated as mature equilibrium;
- ITPC is gated by a sufficient event count; readiness components live in [0,1];
- closure needs a stable hysteresis window plus retention and free plastic capacity;
  zones can be reopened addressably with a serialized reason, keeping their age.
The discrete L4->L2/3->L5 pathway-enablement stage machine is a pending integration
sub-step; this module provides the readiness signal that ACL consumes.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch


class DevelopmentalReadinessController:
    def __init__(self, n_zones: int, tau_mature=1000.0, eta_max: float = 1.0,
                 g_inh_max: float = 1.0, beta: float = 2.0, sigma_0: float = 1.0,
                 k_sigma: float = 2.3026, theta_r: float = 0.60, window_w: int = 50,
                 itpc_threshold: float = 0.55, itpc_min_events: int = 20,
                 ei_eps: float = 1e-6, ei_activity_floor: float = 1.0,
                 min_plastic_fraction: float = 0.25,
                 weights=(0.4, 0.4, 0.2)):
        if (int(n_zones) <= 0 or not math.isfinite(eta_max) or eta_max < 0
                or not math.isfinite(g_inh_max) or g_inh_max < 0 or not math.isfinite(beta) or beta < 0
                or not math.isfinite(sigma_0) or sigma_0 < 0 or not math.isfinite(k_sigma)
                or not 0 <= theta_r <= 1 or int(window_w) <= 0
                or not 0 <= itpc_threshold <= 1 or int(itpc_min_events) <= 0
                or not math.isfinite(ei_eps) or ei_eps <= 0
                or not math.isfinite(ei_activity_floor) or ei_activity_floor <= 0
                or not 0 <= min_plastic_fraction <= 1):
            raise ValueError("invalid DCC configuration")
        w_c, w_p, w_e = (float(w) for w in weights)
        if min(w_c, w_p, w_e) < 0 or not math.isclose(w_c + w_p + w_e, 1.0, abs_tol=1e-6):
            raise ValueError("readiness weights must be nonnegative and sum to 1")
        self.n_zones = int(n_zones)
        if torch.is_tensor(tau_mature):
            tau = tau_mature.clone().float().reshape(-1)
        elif isinstance(tau_mature, (list, tuple)):
            tau = torch.tensor(tau_mature, dtype=torch.float32)
        else:
            tau = torch.full((self.n_zones,), float(tau_mature))
        if tau.numel() != self.n_zones or not torch.all(tau > 0) or not torch.isfinite(tau).all():
            raise ValueError("tau_mature must be positive per zone")
        self.tau_mature = tau
        self.eta_max, self.g_inh_max, self.beta = float(eta_max), float(g_inh_max), float(beta)
        self.sigma_0, self.k_sigma = float(sigma_0), float(k_sigma)
        self.theta_r, self.window_w = float(theta_r), int(window_w)
        self.itpc_threshold, self.itpc_min_events = float(itpc_threshold), int(itpc_min_events)
        self.ei_eps, self.ei_activity_floor = float(ei_eps), float(ei_activity_floor)
        self.min_plastic_fraction = float(min_plastic_fraction)
        self.w_c, self.w_p, self.w_e = w_c, w_p, w_e
        self.age = torch.zeros(self.n_zones)
        self.maturity = torch.zeros(self.n_zones)
        self.eta = torch.zeros(self.n_zones)
        self.g_inh = torch.zeros(self.n_zones)
        self.sigma_noise = torch.full((self.n_zones,), self.sigma_0)
        self.readiness = torch.zeros(self.n_zones)
        self.consecutive_ready = torch.zeros(self.n_zones, dtype=torch.long)
        self.closed = torch.zeros(self.n_zones, dtype=torch.bool)
        self.closure_age = torch.zeros(self.n_zones)
        self.closure_reason: List[str] = [""] * self.n_zones

    def maturity_prior(self, age: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.exp(-age / self.tau_mature)

    def eta_window(self, m: torch.Tensor) -> torch.Tensor:
        return self.eta_max * 4.0 * m * (1.0 - m)

    def ei_score(self, exc: torch.Tensor, inh: torch.Tensor) -> torch.Tensor:
        ratio = (exc + self.ei_eps) / (inh + self.ei_eps)
        balance = torch.exp(-torch.abs(torch.log(ratio)))
        activity = ((exc + inh) / self.ei_activity_floor).clamp(0.0, 1.0)
        return balance * activity

    def itpc_gate(self, itpc: torch.Tensor, n_events: torch.Tensor) -> torch.Tensor:
        sufficiency = (n_events.float() / self.itpc_min_events).clamp(0.0, 1.0)
        return itpc.clamp(0.0, 1.0) * sufficiency

    def readiness_score(self, itpc_gated: torch.Tensor, pred_accuracy: torch.Tensor,
                        ei: torch.Tensor) -> torch.Tensor:
        return (self.w_c * itpc_gated + self.w_p * pred_accuracy.clamp(0.0, 1.0)
                + self.w_e * ei).clamp(0.0, 1.0)

    def _observe(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for name in ("itpc", "pred_accuracy", "exc", "inh", "n_events"):
            t = torch.as_tensor(obs[name], dtype=torch.float32).reshape(-1)
            if t.numel() != self.n_zones:
                raise ValueError(f"observation {name} must have n_zones entries")
            out[name] = t
        return out

    def tick(self, obs: Dict[str, torch.Tensor], dt: float,
             retention_ok: bool = True) -> torch.Tensor:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        o = self._observe(obs)
        self.age = self.age + float(dt)
        self.maturity = self.maturity_prior(self.age)
        self.eta = self.eta_window(self.maturity)
        self.g_inh = self.g_inh_max * self.maturity.pow(self.beta)
        self.sigma_noise = self.sigma_0 * torch.exp(-self.k_sigma * self.maturity)
        itpc_gated = self.itpc_gate(o["itpc"], o["n_events"])
        ei = self.ei_score(o["exc"], o["inh"])
        self.readiness = self.readiness_score(itpc_gated, o["pred_accuracy"], ei)
        ready_now = (self.readiness >= self.theta_r) & (itpc_gated >= self.itpc_threshold)
        self.consecutive_ready = torch.where(
            ready_now, self.consecutive_ready + 1,
            torch.clamp(self.consecutive_ready - 1, min=0))
        if retention_ok:
            self._close_ready_zones()
        return self.readiness

    def _close_ready_zones(self) -> None:
        candidates = (self.consecutive_ready >= self.window_w) & ~self.closed
        n_closed_max = math.floor((1.0 - self.min_plastic_fraction) * self.n_zones)
        room = n_closed_max - int(self.closed.sum())
        if room <= 0 or not bool(candidates.any()):
            return
        idx = candidates.nonzero().flatten()
        order = idx[torch.argsort(self.readiness[idx], descending=True)]
        for z in order[:room].tolist():
            self.closed[z] = True
            self.closure_age[z] = self.age[z]
            self.closure_reason[z] = "readiness_window"

    def effective_plasticity(self) -> torch.Tensor:
        return self.eta * (~self.closed).float()

    def reopen(self, indices: Sequence[int], reason: str = "distribution_shift") -> int:
        idx = torch.as_tensor(list(indices), dtype=torch.long)
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= self.n_zones):
            raise ValueError("zone index out of range")
        self.closed[idx] = False
        self.consecutive_ready[idx] = 0
        for z in idx.tolist():
            self.closure_reason[z] = f"reopened:{reason}"
        return int(idx.numel())

    def telemetry(self) -> Dict[str, object]:
        return {
            "n_zones": self.n_zones, "mean_maturity": float(self.maturity.mean()),
            "mean_readiness": float(self.readiness.mean()),
            "mean_eta": float(self.eta.mean()),
            "closed_zones": int(self.closed.sum()),
            "plastic_zones": int((~self.closed).sum()),
            "min_plastic_fraction": self.min_plastic_fraction,
            "theta_r": self.theta_r, "window_w": self.window_w,
        }

    def export_state_dict(self) -> Dict[str, object]:
        return {
            "n_zones": self.n_zones, "tau_mature": self.tau_mature.clone(),
            "eta_max": self.eta_max, "g_inh_max": self.g_inh_max, "beta": self.beta,
            "sigma_0": self.sigma_0, "k_sigma": self.k_sigma, "theta_r": self.theta_r,
            "window_w": self.window_w, "itpc_threshold": self.itpc_threshold,
            "itpc_min_events": self.itpc_min_events, "ei_eps": self.ei_eps,
            "ei_activity_floor": self.ei_activity_floor,
            "min_plastic_fraction": self.min_plastic_fraction,
            "weights": (self.w_c, self.w_p, self.w_e),
            "age": self.age.clone(), "maturity": self.maturity.clone(),
            "eta": self.eta.clone(), "g_inh": self.g_inh.clone(),
            "sigma_noise": self.sigma_noise.clone(), "readiness": self.readiness.clone(),
            "consecutive_ready": self.consecutive_ready.clone(), "closed": self.closed.clone(),
            "closure_age": self.closure_age.clone(), "closure_reason": list(self.closure_reason),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        loaded = type(self)(
            int(state["n_zones"]), state["tau_mature"], float(state["eta_max"]),
            float(state["g_inh_max"]), float(state["beta"]), float(state["sigma_0"]),
            float(state["k_sigma"]), float(state["theta_r"]), int(state["window_w"]),
            float(state["itpc_threshold"]), int(state["itpc_min_events"]),
            float(state["ei_eps"]), float(state["ei_activity_floor"]),
            float(state["min_plastic_fraction"]), tuple(state["weights"]))
        for name in ("age", "maturity", "eta", "g_inh", "sigma_noise", "readiness",
                     "consecutive_ready", "closed", "closure_age"):
            tensor = torch.as_tensor(state[name]).clone()
            if tensor.shape != (loaded.n_zones,):
                raise ValueError(f"DCC {name} shape mismatch")
            setattr(loaded, name, tensor)
        loaded.closure_reason = list(state["closure_reason"])
        self.__dict__.update(loaded.__dict__)