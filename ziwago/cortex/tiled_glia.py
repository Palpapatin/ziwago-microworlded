"""Tiled astrocytic glial control (ACL), blueprint §3.4 / PHY-GLIA-002.

One astrocyte per tile of neurons. Fixes the defects of the legacy microdomain:
a single exponential relaxation (measured tau == tau_C, no hidden second decay),
drive normalized by tile size, a conservative normalized diffusion laplacian with
a stability-limited step, and a bounded sigmoid gate applied at one declared point.
The cumulative integral G is *exposure*, not learnedness. PNN closure (HARD)
requires an external readiness signal (DCC, pending) and retention, and a plastic
tile quota forbids global freezing; tiles can be reopened addressably (ChABC).
Modes: off (gate bypassed), phantom (state computed, gate neutral), on (active).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch


class TiledGlialControl:
    PLASTIC, SOFT, HARD = 0, 1, 2

    def __init__(self, n_tiles: int, tile_size: int = 16, tau_c: float = 2500.0,
                 rho: float = 0.05, diffusion: float = 0.1, mu_c: float = 0.2,
                 sigma_c: float = 0.15, k_g: float = 4.0, beta_g: float = -1.0,
                 theta1: float = 50.0, theta2: float = 250.0,
                 readiness_threshold: float = 0.5, min_plastic_fraction: float = 0.25,
                 mode: str = "on"):
        if (int(n_tiles) <= 0 or int(tile_size) <= 0 or not math.isfinite(tau_c) or tau_c <= 0
                or not math.isfinite(rho) or rho < 0 or not math.isfinite(diffusion) or diffusion < 0
                or not math.isfinite(sigma_c) or sigma_c <= 0
                or not math.isfinite(theta1) or theta1 < 0 or not math.isfinite(theta2) or theta2 <= theta1
                or not 0 <= readiness_threshold <= 1 or not 0 <= min_plastic_fraction <= 1
                or mode not in ("off", "phantom", "on")):
            raise ValueError("invalid tiled glial configuration")
        self.n_tiles = int(n_tiles)
        self.tile_size = int(tile_size)
        self.tau_c = float(tau_c)
        self.rho = float(rho)
        self.diffusion = float(diffusion)
        self.mu_c, self.sigma_c, self.k_g, self.beta_g = float(mu_c), float(sigma_c), float(k_g), float(beta_g)
        self.theta1, self.theta2 = float(theta1), float(theta2)
        self.readiness_threshold = float(readiness_threshold)
        self.min_plastic_fraction = float(min_plastic_fraction)
        self.mode = mode
        self.C = torch.zeros(self.n_tiles, dtype=torch.float32)
        self.G = torch.zeros(self.n_tiles, dtype=torch.float32)
        self.age = torch.zeros(self.n_tiles, dtype=torch.float32)
        self.state = torch.zeros(self.n_tiles, dtype=torch.long)

    def reaction_target(self, mean_spikes: torch.Tensor, iota: float = 0.0) -> torch.Tensor:
        return (self.rho * mean_spikes + float(iota)).clamp_min(0.0)

    def reaction(self, C: torch.Tensor, target: torch.Tensor, dt: float) -> torch.Tensor:
        return target + (C - target) * math.exp(-float(dt) / self.tau_c)

    def diffuse(self, C: torch.Tensor, alpha: float) -> torch.Tensor:
        a = max(0.0, min(0.5, float(alpha)))
        if self.n_tiles <= 2 or a == 0.0:
            return C.clone()
        lap = 0.5 * (torch.roll(C, 1, 0) + torch.roll(C, -1, 0)) - C
        return C + a * lap

    def gate_from_calcium(self, C: torch.Tensor) -> torch.Tensor:
        x = self.k_g * (C - self.mu_c) / self.sigma_c + self.beta_g
        return torch.sigmoid(x)

    def _tile_mean(self, spikes: torch.Tensor) -> torch.Tensor:
        if spikes.shape == (self.n_tiles, self.tile_size):
            return spikes.mean(dim=1)
        if spikes.numel() == self.n_tiles * self.tile_size:
            return spikes.reshape(self.n_tiles, self.tile_size).mean(dim=1)
        raise ValueError("spikes must align with n_tiles * tile_size")

    def step(self, spikes: torch.Tensor, dt: float, awake: bool = True,
             iota: float = 0.0, readiness: Optional[torch.Tensor] = None,
             retention_ok: bool = True) -> torch.Tensor:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if self.mode == "off":
            return torch.ones(self.n_tiles, dtype=torch.float32)
        mean_spikes = self._tile_mean(spikes)
        target = self.reaction_target(mean_spikes, iota)
        self.C = self.reaction(self.C, target, dt)
        self.C = self.diffuse(self.C, self.diffusion).clamp_min(0.0)
        gate = self.gate_from_calcium(self.C)
        if awake:
            self.G = self.G + self.C * float(dt)
            self.age = self.age + float(dt)
            self.update_states(readiness, retention_ok)
        if self.mode == "phantom":
            return torch.ones(self.n_tiles, dtype=torch.float32)
        return gate

    def update_states(self, readiness: Optional[torch.Tensor], retention_ok: bool) -> None:
        r = torch.zeros(self.n_tiles) if readiness is None else torch.as_tensor(readiness, dtype=torch.float32)
        hard = (self.G >= self.theta2) & (r >= self.readiness_threshold) & bool(retention_ok)
        soft = (self.G >= self.theta1) & ~hard
        new_state = self.state.clone()
        new_state[soft] = self.SOFT
        new_state[hard] = self.HARD
        n_min_plastic = math.ceil(self.min_plastic_fraction * self.n_tiles)
        plastic = new_state == self.PLASTIC
        deficit = n_min_plastic - int(plastic.sum())
        if deficit > 0:
            nonplastic = (~plastic).nonzero().flatten()
            order = nonplastic[torch.argsort(self.G[nonplastic])]
            new_state[order[:deficit]] = self.PLASTIC
        self.state = new_state

    def reopen(self, indices: Sequence[int]) -> int:
        idx = torch.as_tensor(list(indices), dtype=torch.long)
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= self.n_tiles):
            raise ValueError("tile index out of range")
        self.G[idx] = 0.0
        self.state[idx] = self.PLASTIC
        return int(idx.numel())

    def telemetry(self) -> Dict[str, object]:
        return {
            "n_tiles": self.n_tiles, "tile_size": self.tile_size, "mode": self.mode,
            "tau_c": self.tau_c, "mean_C": float(self.C.mean()), "max_C": float(self.C.max()),
            "mean_G": float(self.G.mean()), "max_G": float(self.G.max()),
            "plastic_tiles": int((self.state == self.PLASTIC).sum()),
            "soft_tiles": int((self.state == self.SOFT).sum()),
            "hard_tiles": int((self.state == self.HARD).sum()),
        }

    def export_state_dict(self) -> Dict[str, object]:
        return {
            "n_tiles": self.n_tiles, "tile_size": self.tile_size, "tau_c": self.tau_c,
            "rho": self.rho, "diffusion": self.diffusion, "mu_c": self.mu_c,
            "sigma_c": self.sigma_c, "k_g": self.k_g, "beta_g": self.beta_g,
            "theta1": self.theta1, "theta2": self.theta2,
            "readiness_threshold": self.readiness_threshold,
            "min_plastic_fraction": self.min_plastic_fraction, "mode": self.mode,
            "C": self.C.clone(), "G": self.G.clone(), "age": self.age.clone(),
            "state": self.state.clone(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        loaded = type(self)(
            int(state["n_tiles"]), int(state["tile_size"]), float(state["tau_c"]),
            float(state["rho"]), float(state["diffusion"]), float(state["mu_c"]),
            float(state["sigma_c"]), float(state["k_g"]), float(state["beta_g"]),
            float(state["theta1"]), float(state["theta2"]),
            float(state["readiness_threshold"]), float(state["min_plastic_fraction"]),
            str(state["mode"]))
        for name in ("C", "G", "age", "state"):
            tensor = torch.as_tensor(state[name]).clone()
            if tensor.shape != (loaded.n_tiles,):
                raise ValueError(f"glial {name} shape mismatch")
            setattr(loaded, name, tensor)
        self.__dict__.update(loaded.__dict__)