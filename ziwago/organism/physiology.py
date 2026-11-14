"""Reduced, deliberately time-compressed v20 physiology (not biological calibration).

One NeuroChemistry owner, advanced in milliseconds, never by wave count. Loads
are supplied by the caller, including during sleep; there is no invented rest
load. ATP is a fractional pool, not a concentration of lactate. Extracellular
zonal adenosine and the global sleep-pressure filter are distinct states.

Legacy ``update``/``ecb_update`` remain inherited for compatibility, but must not
be called alongside ``advance``. The v20 stepping entry point is ``advance``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real

import torch

from ..chemistry.modulators import ChemState, NeuroChemistry
from ..config import ChemistryCfg


def _scalar(name: str, value, *, signed: bool = False) -> float:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.is_complex() or value.dtype == torch.bool:
            raise ValueError(f"{name} must be a real scalar")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value) or (not signed and value < 0.0):
        raise ValueError(f"{name} must be finite" + ("" if signed else " and nonnegative"))
    return value


def _relax(value: float, target: float, tau_ms: float, dt_ms: float) -> float:
    """Exact constant-target exponential; expm1 also handles very small dt."""
    return value + (target - value) * -math.expm1(-dt_ms / tau_ms)


@dataclass
class PhysiologyConfig:
    """Explicit engineering constants; all times are compressed model ms.

    For constant effective load L, E' = r(1-E) - burn*L*E and
    A' = production*L*(1-A) - clearance*A. Thus both zonal pools stay
    in [0, 1] without an energy floor. Ca contributes an additive measured
    load. Control signals novelty/attention/stress/drive saturate at one;
    loads and Ca are not clipped. Normal chemistry time constants come from
    ChemistryCfg, except the explicitly compressed orexin time below.
    Legacy ado/lactate/bout gains are deliberately unused.
    """

    atp_recovery_tau_ms: float = 6000.0
    atp_burn_per_ms: float = 0.0002
    calcium_load_gain: float = 0.25
    sleep_recovery_multiplier: float = 3.0
    ado_production_per_ms: float = 0.0004
    ado_clearance_tau_ms: float = 6000.0
    sleep_ado_clearance_multiplier: float = 3.0
    process_s_tau_ms: float = 2000.0
    process_s_sleep_tau_ms: float = 1500.0
    process_s_wake_threshold: float = 0.25
    atp_wake_threshold: float = 0.8
    vlpo_tau_ms: float = 250.0
    ox_tau_ms: float = 500.0
    vlpo_pressure_gain: float = 12.0
    vlpo_ox_inhibition: float = 3.0
    ox_vlpo_inhibition: float = 4.0
    vlpo_sleep_threshold: float = 0.6
    vlpo_wake_threshold: float = 0.3
    ox_wake_threshold: float = 0.35
    ox_baseline: float = 0.6
    ox_drive_gain: float = 0.4
    nrem_min_ms: float = 2000.0
    rem_duration_ms: float = 1000.0
    # The nonlinear switch and pressure-coupled ACh source require substeps.
    integration_step_ms: float = 10.0
    phasic_reward_gain: float = 0.05
    phasic_limit: float = 2.0
    da_tonic_target: float = 0.5
    ne_baseline: float = 0.3
    ne_novelty_gain: float = 1.0
    ach_baseline: float = 0.4
    ach_attention_gain: float = 0.8
    ht_target: float = 0.3
    cort_baseline: float = 0.1
    cort_stress_gain: float = 1.0
    calcium_half_saturation: float = 0.5
    ecb_target_max: float = 1.2
    no_target_max: float = 1.0
    ach_ado_half_saturation: float = 0.6  # Hypothesized normalized pressure scale, not a measured constant.

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            setattr(self, name, _scalar(name, value))
        for name in (
            "atp_recovery_tau_ms", "ado_clearance_tau_ms", "process_s_tau_ms",
            "process_s_sleep_tau_ms", "vlpo_tau_ms", "ox_tau_ms", "nrem_min_ms",
            "rem_duration_ms", "integration_step_ms", "calcium_half_saturation",
            "sleep_recovery_multiplier", "sleep_ado_clearance_multiplier",
            "phasic_limit", "ach_ado_half_saturation",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "process_s_wake_threshold", "atp_wake_threshold", "vlpo_sleep_threshold",
            "vlpo_wake_threshold", "ox_wake_threshold", "ox_baseline",
        ):
            if not 0.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")
        if self.vlpo_wake_threshold >= self.vlpo_sleep_threshold:
            raise ValueError("VLPO wake threshold must be below sleep threshold")
        if self.ox_wake_threshold >= self.ox_baseline:
            raise ValueError("orexin must be able to recover at zero drive")


@dataclass
class V20ChemState(ChemState):
    da_phasic: float = 0.0  # Signed impulse trace; never folded into tonic da.


class OrganismPhysiology(NeuroChemistry):
    """Single, snapshot-safe chemistry owner with a hysteretic sleep switch.

    Zonal tensors are detached CPU float64 vectors (one entry per zone).
    ``tick`` counts positive-duration advance calls, not integration substeps.
    Zero-duration advances validate their inputs but are otherwise no-ops.
    """

    def __init__(self, n_zones: int, cfg: PhysiologyConfig | None = None,
                 chemistry_cfg: ChemistryCfg | None = None):
        if isinstance(n_zones, bool) or not isinstance(n_zones, Integral) or n_zones <= 0:
            raise ValueError("n_zones must be a positive integer")
        if cfg is not None and not isinstance(cfg, PhysiologyConfig):
            raise TypeError("cfg must be PhysiologyConfig")
        if chemistry_cfg is not None and not isinstance(chemistry_cfg, ChemistryCfg):
            raise TypeError("chemistry_cfg must be ChemistryCfg")
        self.physiology_cfg = cfg if cfg is not None else PhysiologyConfig()
        self.physiology_cfg.__post_init__()
        super().__init__(chemistry_cfg if chemistry_cfg is not None else ChemistryCfg())
        for name in (
            "da_tau_phasic", "da_tau_tonic", "ne_tau", "ach_tau", "ht_tau",
            "cort_tau_nongenomic", "ecb_tau", "no_tau", "ecb_ic50", "theta_stress",
        ):
            if _scalar(name, getattr(self.cfg, name)) <= 0.0:
                raise ValueError(f"chemistry {name} must be positive")
        for name in ("alpha_da", "beta_ne", "kappa_ecb"):
            _scalar(name, getattr(self.cfg, name))
        for name in ("sleep_threshold", "theta_energy", "ox_collapse"):
            value = _scalar(name, getattr(self.cfg, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"chemistry {name} must lie between zero and one")
        p = self.physiology_cfg
        if p.process_s_wake_threshold >= self.cfg.sleep_threshold:
            raise ValueError("sleep-pressure wake threshold must be below sleep threshold")
        if p.atp_wake_threshold <= self.cfg.theta_energy:
            raise ValueError("ATP wake threshold must exceed plasticity threshold")
        if p.ox_wake_threshold <= self.cfg.ox_collapse:
            raise ValueError("orexin wake threshold must exceed collapse threshold")
        self.n_zones = int(n_zones)
        self.s = V20ChemState(ox=p.ox_baseline)
        self.phase = "awake"
        self.time_ms = 0.0
        self.tick = 0
        self.phase_elapsed_ms = 0.0
        self.zone_atp = torch.ones(self.n_zones, dtype=torch.float64)
        self.zone_ado = torch.zeros(self.n_zones, dtype=torch.float64)
        self.zone_load = torch.zeros(self.n_zones, dtype=torch.float64)
        self.vlpo = 0.0
        self.phase_events: list[dict] = []
        self.reward_events = 0
        self.last_reward: dict | None = None
        self.last_inputs: dict = {}

    @property
    def process_s(self) -> float:
        # One authoritative pressure scalar, also visible through legacy s.ado.
        return self.s.ado

    @process_s.setter
    def process_s(self, value: float) -> None:
        self.s.ado = value

    def reward(self, rpe: float) -> dict:
        """Apply one signed, bounded DA impulse; do not advance any clock/pool."""
        rpe = _scalar("rpe", rpe, signed=True)
        p = self.physiology_cfg
        before = self.s.da_phasic
        self.s.da_phasic = max(-p.phasic_limit, min(
            p.phasic_limit, before + p.phasic_reward_gain * rpe))
        self.reward_events += 1
        self.last_reward = {"rpe": rpe, "delta": self.s.da_phasic - before,
                            "time_ms": self.time_ms, "tick": self.tick}
        return self.bundle()

    def request_sleep(self) -> dict:
        """Enter NREM explicitly, recording provenance without resetting pools.

        Repeated NREM requests are recorded but do not restart the dwell timer.
        """
        self._transition("nrem", "requested", "requested")
        return self.bundle()

    def _transition(self, phase: str, reason: str, provenance: str = "natural") -> None:
        self.phase_events.append({
            "event": "transition" if phase != self.phase else "sleep_request",
            "from": self.phase, "to": phase, "reason": reason,
            "provenance": provenance, "time_ms": self.time_ms, "tick": self.tick,
            "phase_elapsed_ms": self.phase_elapsed_ms,
            "process_s": self.process_s, "atp_mean": float(self.zone_atp.mean()),
            "vlpo": self.vlpo, "ox": self.s.ox,
        })
        if phase != self.phase:
            self.phase = phase
            self.phase_elapsed_ms = 0.0
        self.is_awake = phase == "awake"

    def _recovered(self) -> bool:
        p = self.physiology_cfg
        return (self.process_s <= p.process_s_wake_threshold
                and float(self.zone_atp.mean()) >= p.atp_wake_threshold
                and self.vlpo <= p.vlpo_wake_threshold
                and self.s.ox >= p.ox_wake_threshold)

    def _check_phase(self) -> None:
        p = self.physiology_cfg
        if self.phase == "awake":
            if (self.process_s >= self.cfg.sleep_threshold
                    and self.vlpo >= p.vlpo_sleep_threshold
                    and self.s.ox <= self.cfg.ox_collapse):
                self._transition("nrem", "sleep_pressure")
        elif self.phase == "nrem":
            if self.phase_elapsed_ms >= p.nrem_min_ms and self._recovered():
                self._transition("rem", "nrem_recovered")
        elif self.phase_elapsed_ms >= p.rem_duration_ms:
            # A bounded REM cannot force an unrecovered organism awake.
            if self._recovered():
                self._transition("awake", "rem_recovered")
            else:
                self._transition("nrem", "rem_recovery_lost")

    def _switch_targets(self, pressure: float, vlpo: float, ox: float,
                        drive: float) -> tuple[float, float]:
        p = self.physiology_cfg
        x = (p.vlpo_pressure_gain * (pressure - self.cfg.sleep_threshold)
             - p.vlpo_ox_inhibition * ox)
        # Stable sigmoid even for large configured gains.
        v_target = 1.0 / (1.0 + math.exp(-x)) if x >= 0.0 else math.exp(x) / (1.0 + math.exp(x))
        ox_source = min(1.0, p.ox_baseline + p.ox_drive_gain * min(1.0, drive))
        return v_target, ox_source * math.exp(-p.ox_vlpo_inhibition * vlpo)

    def _ach_pressure_suppression(self, pressure: float) -> float:
        """Instantaneous source factor; no extra state or per-call ACh kick."""
        ratio = pressure / self.physiology_cfg.ach_ado_half_saturation
        # Multiplication saturates instead of raising on tiny valid pressure scales.
        return 1.0 / (1.0 + ratio * ratio)

    def _integrate(self, effective_load: torch.Tensor, inputs: dict, dt: float) -> None:
        p, c, s = self.physiology_cfg, self.cfg, self.s
        sleeping = self.phase != "awake"
        recovery = (p.sleep_recovery_multiplier if sleeping else 1.0) / p.atp_recovery_tau_ms
        rate = recovery + p.atp_burn_per_ms * effective_load
        target = recovery / rate
        self.zone_atp = target + (self.zone_atp - target) * torch.exp(-rate * dt)
        s.lactate = float(self.zone_atp.mean())

        clearance = (p.sleep_ado_clearance_multiplier if sleeping else 1.0) / p.ado_clearance_tau_ms
        production = p.ado_production_per_ms * effective_load
        ado_rate = production + clearance
        ado_target = production / ado_rate
        ado_offset = self.zone_ado - ado_target
        ado_decay = torch.exp(-ado_rate * dt)
        old_pressure = self.process_s
        if sleeping:
            self.process_s *= math.exp(-dt / p.process_s_sleep_tau_ms)
        else:
            # Exact convolution of the zonal exponential sources with the
            # global linear pressure filter, rather than an endpoint Euler step.
            k = 1.0 / p.process_s_tau_ms
            decay = math.exp(-k * dt)
            delta = k - ado_rate
            coincident = delta.abs() < 1e-12
            safe_delta = torch.where(coincident, torch.ones_like(delta), delta)
            convolution = k * (ado_decay - decay) / safe_delta
            convolution = torch.where(coincident, torch.full_like(delta, k * dt * decay), convolution)
            mean_target = float(ado_target.mean())
            pressure = (mean_target + (old_pressure - mean_target) * decay
                        + float((ado_offset * convolution).mean()))
            self.process_s = max(0.0, min(1.0, pressure))
        self.zone_ado = ado_target + ado_offset * ado_decay
        pressure_mid = (old_pressure + self.process_s) * 0.5

        # Exponential midpoint for the mutually inhibiting nonlinear switch.
        # The source targets are always finite and bounded, not per-call kicks.
        v_target, ox_target = self._switch_targets(old_pressure, self.vlpo, s.ox, inputs["drive"])
        v_mid = _relax(self.vlpo, v_target, p.vlpo_tau_ms, dt * 0.5)
        ox_mid = _relax(s.ox, ox_target, p.ox_tau_ms, dt * 0.5)
        v_target, ox_target = self._switch_targets(
            pressure_mid, v_mid, ox_mid, inputs["drive"])
        self.vlpo = _relax(self.vlpo, v_target, p.vlpo_tau_ms, dt)
        s.ox = _relax(s.ox, ox_target, p.ox_tau_ms, dt)
        s.da = _relax(s.da, p.da_tonic_target, c.da_tau_tonic, dt)
        s.da_phasic *= math.exp(-dt / c.da_tau_phasic)
        s.ne = _relax(s.ne, p.ne_baseline + p.ne_novelty_gain * min(1.0, inputs["novelty"]), c.ne_tau, dt)
        # Suppress the midpoint source, not the already relaxed concentration.
        ach_source = p.ach_baseline + p.ach_attention_gain * min(1.0, inputs["attention"])
        ach_target = ach_source * self._ach_pressure_suppression(pressure_mid)
        s.ach = _relax(s.ach, ach_target, c.ach_tau, dt)
        s.ht = _relax(s.ht, p.ht_target, c.ht_tau, dt)
        s.cort = _relax(s.cort, p.cort_baseline + p.cort_stress_gain * min(1.0, inputs["stress"]), c.cort_tau_nongenomic, dt)
        calcium = inputs["ca_mean"]
        ca_signal = 1.0 / (1.0 + p.calcium_half_saturation / calcium) if calcium > 0.0 else 0.0
        s.ecb = _relax(s.ecb, p.ecb_target_max * ca_signal, c.ecb_tau, dt)
        s.no = _relax(s.no, p.no_target_max * ca_signal, c.no_tau, dt)

    def advance(self, zone_load: torch.Tensor, *, ca_mean: float = 0.0,
                novelty: float = 0.0, attention: float = 0.0, stress: float = 0.0,
                drive: float = 0.0, dt_ms: float = 1.0) -> dict:
        """Advance measured physiology; validate every input before mutation."""
        if not isinstance(zone_load, torch.Tensor):
            raise TypeError("zone_load must be a tensor")
        if zone_load.ndim != 1 or zone_load.shape[0] != self.n_zones:
            raise ValueError(f"zone_load must have shape ({self.n_zones},)")
        if zone_load.layout != torch.strided or zone_load.is_complex() or zone_load.dtype == torch.bool:
            raise ValueError("zone_load must be a dense real numeric tensor")
        load = zone_load.detach().to(device="cpu", dtype=torch.float64).clone()
        if not bool(torch.isfinite(load).all()) or bool((load < 0.0).any()):
            raise ValueError("zone_load must be finite and nonnegative")
        inputs = {name: _scalar(name, value) for name, value in (
            ("ca_mean", ca_mean), ("novelty", novelty), ("attention", attention),
            ("stress", stress), ("drive", drive),
        )}
        dt = _scalar("dt_ms", dt_ms)
        end_time = self.time_ms + dt
        effective_load = load + self.physiology_cfg.calcium_load_gain * inputs["ca_mean"]
        p = self.physiology_cfg
        if (not math.isfinite(end_time) or not bool(torch.isfinite(effective_load).all())
                or not bool(torch.isfinite(effective_load * max(p.atp_burn_per_ms, p.ado_production_per_ms)).all())):
            raise ValueError("inputs overflow the physiology clock or rates")
        if dt == 0.0:
            return self.bundle()

        self.zone_load = load
        self.last_inputs = {**inputs, "dt_ms": dt}
        self.tick += 1
        remaining = dt
        while remaining > 0.0:
            step = min(remaining, p.integration_step_ms)
            if self.phase == "rem":
                step = min(step, p.rem_duration_ms - self.phase_elapsed_ms)
            elif self.phase == "nrem" and self.phase_elapsed_ms < p.nrem_min_ms:
                step = min(step, p.nrem_min_ms - self.phase_elapsed_ms)
            self._integrate(effective_load, inputs, step)
            self.time_ms += step
            self.phase_elapsed_ms += step
            remaining -= step
            self._check_phase()
        self.time_ms = end_time
        self.ticks_awake += int(self.is_awake)
        self.ticks_asleep += int(not self.is_awake)
        return self.bundle()

    def f_mod(self, energy_ok: float = 1.0) -> float:
        """A genuinely zero gate in sleep, energy deficit or high pressure.

        The optional legacy energy_ok can further suppress, never override,
        measured mean ATP. No positive floor is applied to the resulting gate.
        """
        energy_ok = _scalar("energy_ok", energy_ok)
        energy = float(self.zone_atp.mean())
        if (self.phase != "awake" or not self.is_awake
                or energy <= self.cfg.theta_energy
                or self.process_s >= self.cfg.sleep_threshold):
            return 0.0
        chemical = max(0.0, self.cfg.alpha_da * self.s.da + self.cfg.beta_ne * self.s.ne)
        ecb_gate = max(0.0, 1.0 - self.cfg.kappa_ecb * self.s.ecb)
        return chemical * ecb_gate * self.phi_cort() * energy * min(1.0, energy_ok)

    def bundle(self) -> dict:
        result = {name: float(getattr(self.s, name)) for name in (
            "ach", "ne", "da", "da_phasic", "ht", "ox", "ado", "cort", "ecb", "no", "lactate",
        )}
        result.update(p_release=float(self.p_release()), f_mod=float(self.f_mod()),
                      awake=bool(self.is_awake), phase=self.phase,
                      time_ms=self.time_ms, tick=self.tick)
        return result

    def telemetry(self) -> dict:
        """Detached JSON-safe values; no tensors, owners or core references."""
        return {
            **self.bundle(), "phase_elapsed_ms": self.phase_elapsed_ms,
            "process_s": self.process_s, "vlpo": self.vlpo,
            # Current-state factor, not the last substep's midpoint target.
            "ach_pressure_suppression": self._ach_pressure_suppression(self.process_s),
            "n_zones": self.n_zones,
            "zone_atp": self.zone_atp.tolist(), "zone_ado": self.zone_ado.tolist(),
            "zone_load": self.zone_load.tolist(), "state": asdict(self.s),
            "physiology_config": asdict(self.physiology_cfg),
            "chemistry_config": asdict(self.cfg),
            "last_inputs": dict(self.last_inputs),
            "reward_events": self.reward_events,
            "last_reward": None if self.last_reward is None else dict(self.last_reward),
            "phase_events": [dict(event) for event in self.phase_events],
        }
