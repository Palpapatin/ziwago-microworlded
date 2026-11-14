"""Clock-driven contextual Go/NoGo selection with one outstanding credit ticket.

This module owns neither the organism clock nor dopamine delivery. ``step`` uses
only its supplied dt; the caller may deliver a *consumed* settlement's RPE once.
``committed`` means a ticket is outstanding, including held decisions;
``new_commit`` distinguishes the initial commitment from subsequent reads.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real

import torch


# Explicit numerical/size budgets, also exposed by telemetry; no evidence floor.
_CONFIG_RANGES = {
    "decision_threshold": (1e-6, 10.0),
    "drift_gain": (0.0, 1000.0),       # accumulator units / second
    "leak_rate": (0.0, 1000.0),        # inverse seconds
    "sigma": (0.0, 100.0),            # accumulator units / sqrt(second)
    "tau_gpe_ms": (0.1, 10000.0),
    "tau_stn_ms": (0.1, 10000.0),
    "tau_gpi_ms": (0.1, 10000.0),
    "context_gain": (0.0, 10.0),
    "conflict_gain": (0.0, 4.0),
    "uncertainty_gain": (0.0, 4.0),
    "ado_gain": (0.0, 4.0),
    "ht_gain": (0.0, 4.0),
    "ox_gain": (0.0, 4.0),
    "learning_rate": (0.0, 1.0),
    "critic_learning_rate": (0.0, 1.0),
    "initial_weight": (0.0, 10.0),
    "weight_max": (1e-6, 10.0),
    "critic_weight_max": (1e-6, 10.0),
    "reward_bound": (1e-6, 100.0),
    "credit_horizon_ms": (1e-3, 3600000.0),
    "max_dt_ms": (1e-6, 1000.0),
}
_MODULATOR_DEFAULTS = {"da": 0.5, "ach": 0.5, "ado": 0.0, "ox": 0.6, "ht": 0.3}
_MODULATOR_RANGES = {
    "da": (0.0, 2.0), "ach": (0.0, 2.0), "ht": (0.0, 2.0),
    "ado": (0.0, 1.0), "ox": (0.0, 1.0),
}
_MAX_ACTIONS = 256
_MAX_CONTEXT_DIM = 4096


def _number(name, value, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, not {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _boolean(name, value):
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _integer(name, value, low, high):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


@dataclass
class ActionGateConfig:
    """Local numerical parameters; all times are ms except explicitly named rates.

    Mutable (rather than a frozen dataclass) for ``load_runtime_tree``. A gate
    copies the supplied config and validates it again before step/settle.
    """

    decision_threshold: float = 0.04
    drift_gain: float = 5.0
    leak_rate: float = 2.0
    sigma: float = 0.0
    tau_gpe_ms: float = 10.0
    tau_stn_ms: float = 5.0
    tau_gpi_ms: float = 10.0
    context_gain: float = 1.0
    conflict_gain: float = 0.5
    uncertainty_gain: float = 0.5
    ado_gain: float = 1.0
    ht_gain: float = 0.25
    ox_gain: float = 0.25
    learning_rate: float = 0.05
    critic_learning_rate: float = 0.05
    initial_weight: float = 0.1
    weight_max: float = 2.0
    critic_weight_max: float = 2.0
    reward_bound: float = 1.0
    credit_horizon_ms: float = 2000.0
    max_dt_ms: float = 1000.0

    def __post_init__(self):
        self.validate()
        for name in _CONFIG_RANGES:
            setattr(self, name, float(getattr(self, name)))

    def validate(self):
        for name, bounds in _CONFIG_RANGES.items():
            _number(name, getattr(self, name), *bounds)
        if self.initial_weight > self.weight_max:
            raise ValueError("initial_weight must not exceed weight_max")


class ContextualActionGate:
    """Small CPU gate with detached, data-only learned and transient state.

    Context may be signed; its L1 norm is capped at one for bounded projections.
    Evidence is divided by max(1, max(evidence)), never thresholded for admission:
    every strictly positive entry is eligible. Go/NoGo synapses are nonnegative,
    their sigmoid activities feed GPe -> STN -> GPi, and GPi inhibits the DDM.

    Modulators used here: da/ach/ht in [0,2], ado/ox in [0,1], awake: bool.
    Other bundle entries must be finite real scalars and are ignored explicitly
    in telemetry. Sleep is controlled by ``awake=False``, not inferred here.

    Learning requires learn=True at commitment AND settlement and an enabled
    ``learning_enabled`` flag at both times. A frozen/evaluation decision cannot
    become trainable later. No weights change during step, expiry or cancel.
    """

    def __init__(self, n_actions: int = 7, context_dim: int = 64, seed: int = 20,
                 cfg: ActionGateConfig | None = None):
        self.n_actions = _integer("n_actions", n_actions, 1, _MAX_ACTIONS)
        self.context_dim = _integer("context_dim", context_dim, 1, _MAX_CONTEXT_DIM)
        self.seed = _integer("seed", seed, 0, 2**63 - 1)
        if cfg is not None and not isinstance(cfg, ActionGateConfig):
            raise TypeError("cfg must be ActionGateConfig or None")
        self.cfg = ActionGateConfig(**asdict(cfg)) if cfg is not None else ActionGateConfig()
        self.learning_enabled = True
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        shape = (self.n_actions, self.context_dim)
        self.w_go = torch.full(shape, self.cfg.initial_weight, dtype=torch.float64)
        self.w_nogo = self.w_go.clone()
        self.w_v = torch.zeros(self.context_dim, dtype=torch.float64)
        self.acc = torch.zeros(self.n_actions, dtype=torch.float64)
        self.gpe = torch.full_like(self.acc, 0.5)
        self.stn = torch.full_like(self.acc, 0.25)
        self.gpi = torch.full_like(self.acc, 0.5)
        self.pending = None
        self.next_decision_id = 1
        self.elapsed_ms = 0.0
        self.counts = {name: 0 for name in ("steps", "committed", "settled", "expired", "cancelled", "updates")}
        self.last_outcome = {"reason": "initial", "consumed": False, "rpe": 0.0, "action": -1,
                             "decision_id": None, "learned": False}
        self.last_step = {"action": -1, "committed": False, "decision_id": None,
                          "new_commit": False, "threshold": self.cfg.decision_threshold,
                          "conflict": 0.0, "bg_release": 0.0, "expired": False,
                          "expired_decision_id": None, "stopped": False, "sleeping": False,
                          "eligible_actions": 0, "evidence_scale": 1.0, "ignored_modulators": []}

    @staticmethod
    def _vector(name, value, size):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a Tensor")
        if value.layout != torch.strided or value.is_complex() or value.dtype == torch.bool:
            raise TypeError(f"{name} must be a dense real numeric Tensor")
        if tuple(value.shape) != (size,):
            raise ValueError(f"{name} must have shape [{size}]")
        value = value.detach().to(device="cpu", dtype=torch.float64).clone()
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
        return value

    @staticmethod
    def _features(context):
        # Two-stage division avoids overflow even for large finite contexts.
        scaled = context / max(1.0, float(context.abs().max()))
        return scaled / max(1.0, float(scaled.abs().sum()))

    @staticmethod
    def _modulators(modulators):
        if not isinstance(modulators, dict):
            raise TypeError("modulators must be dict")
        result = dict(_MODULATOR_DEFAULTS, awake=True)
        ignored = []
        for name, value in modulators.items():
            if not isinstance(name, str):
                raise TypeError("modulator names must be strings")
            if name == "awake":
                result[name] = _boolean("awake", value)
            elif name in _MODULATOR_RANGES:
                result[name] = _number(name, value, *_MODULATOR_RANGES[name])
            else:
                _number(name, value)
                ignored.append(name)
        return result, sorted(ignored)

    def _integrate(self, features, evidence, hyperdirect, mods, dt_s, hold, blocked):
        """Exact affine lowpasses + leaky DDM drift for piecewise-constant input.

        Five-dimensional homogeneous coordinates also handle equal time constants
        and zero leak without singular divisions. With sigma=0 the integration is
        subdivision-invariant up to roundoff, until a sampled commitment boundary.
        All three pathway targets are convex combinations in [0,1].
        """
        c = self.cfg
        go = torch.sigmoid(c.context_gain * (self.w_go @ features) + mods["da"] - 0.5)
        nogo = torch.sigmoid(c.context_gain * (self.w_nogo @ features) + 0.5 - mods["da"])
        re, rs, ri = (1000.0 / tau for tau in (c.tau_gpe_ms, c.tau_stn_ms, c.tau_gpi_ms))
        matrix = torch.zeros((self.n_actions, 5, 5), dtype=torch.float64)
        matrix[:, 0, 0], matrix[:, 0, 4] = -re, re * (1.0 - nogo)
        matrix[:, 1, 0], matrix[:, 1, 1] = -0.5 * rs, -rs
        matrix[:, 1, 4] = rs * (0.5 + 0.5 * hyperdirect)
        matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2] = -0.25 * ri, 0.25 * ri, -ri
        matrix[:, 2, 4] = ri * (0.5 + 0.25 * (nogo - go))
        if not hold:
            drift = torch.zeros_like(evidence) if blocked else c.drift_gain * evidence
            matrix[:, 3, 2], matrix[:, 3, 3], matrix[:, 3, 4] = -drift, -c.leak_rate, drift
        acc = self.acc if hold else self.acc.masked_fill(evidence == 0.0, 0.0)
        state = torch.stack((self.gpe, self.stn, self.gpi, acc, torch.ones_like(acc)), dim=-1)
        updated = (torch.matrix_exp(matrix * dt_s) @ state.unsqueeze(-1)).squeeze(-1)
        # Only physiological states are bounded; accumulator has no admission floor.
        return updated[:, 0].clamp(0.0, 1.0), updated[:, 1].clamp(0.0, 1.0), \
            updated[:, 2].clamp(0.0, 1.0), updated[:, 3]

    @torch.no_grad()
    def step(self, context: torch.Tensor, evidence: torch.Tensor, *, modulators: dict,
             dt_ms: float = 1.0, uncertainty: float = 0.0, stop: bool = False,
             learn: bool = False) -> dict:
        """Advance supplied time; hold credit until settle/cancel or age > horizon.

        Stop/sleep prohibit new decisions and noise draws, but advance lowpasses
        and credit age. A held ticket still reports its original action/id even
        when that action has no current evidence. Expiry consumes without reward,
        returns expired=True, and never recommits in the same step.
        """
        # Validate every argument even when stopped, pending, or about to expire.
        self.cfg.validate()
        context = self._vector("context", context, self.context_dim)
        evidence = self._vector("evidence", evidence, self.n_actions)
        if bool((evidence < 0.0).any()):
            raise ValueError("evidence must be nonnegative")
        dt_ms = _number("dt_ms", dt_ms, 0.0, self.cfg.max_dt_ms)
        if dt_ms == 0.0:
            raise ValueError("dt_ms must be positive")
        uncertainty = _number("uncertainty", uncertainty, 0.0, 1.0)
        stop, learn = _boolean("stop", stop), _boolean("learn", learn)
        enabled = _boolean("learning_enabled", self.learning_enabled)
        mods, ignored = self._modulators(modulators)

        c = self.cfg
        features = self._features(context)
        eligible = evidence > 0.0
        eligible_count = int(eligible.sum())
        evidence_scale = max(1.0, float(evidence.max()))
        evidence = evidence / evidence_scale
        conflict = 0.0
        if eligible_count > 1:
            p = evidence[eligible] / evidence[eligible].sum()
            p = p[p > 0.0]
            conflict = max(0.0, min(1.0, float(-(p * p.log()).sum()) / math.log(eligible_count)))
        threshold = c.decision_threshold * (
            1.0 + c.conflict_gain * conflict + c.uncertainty_gain * uncertainty
            + c.ado_gain * mods["ado"] + c.ht_gain * mods["ht"]
        ) / (1.0 + c.ox_gain * mods["ox"])
        blocked = stop or not mods["awake"]
        hyperdirect = max(conflict, uncertainty, float(blocked))
        hold = self.pending is not None
        gpe, stn, gpi, acc = self._integrate(
            features, evidence, hyperdirect, mods, dt_ms / 1000.0, hold, blocked)
        if not hold and not blocked and eligible_count and c.sigma > 0.0:
            noise = torch.randn(eligible_count, generator=self.generator, dtype=torch.float64)
            acc[eligible] += noise * (c.sigma * math.sqrt(dt_ms / 1000.0))
        self.gpe, self.stn, self.gpi, self.acc = gpe, stn, gpi, acc
        self.elapsed_ms += dt_ms
        self.counts["steps"] += 1
        expired_id = None
        new_commit = False
        if hold:
            self.pending["age_ms"] += dt_ms
            if self.pending["age_ms"] > c.credit_horizon_ms:
                expired_id = self.pending["decision_id"]
                self._consume("expired")
        elif not blocked and eligible_count:
            winner = int(self.acc.masked_fill(~eligible, -math.inf).argmax())
            if float(self.acc[winner]) >= threshold:
                self.pending = {
                    "context": context.clone(), "features": features.clone(),
                    "predicted_value": float(self.w_v @ features), "action": winner,
                    "decision_id": self.next_decision_id, "age_ms": 0.0,
                    "learnable": bool(learn and enabled),
                    "plasticity_scale": 2.0 * mods["ach"] / (1.0 + mods["ado"]),
                }
                self.next_decision_id += 1
                self.counts["committed"] += 1
                new_commit = True
        action = self.pending["action"] if self.pending is not None else -1
        release = 0.0
        if not blocked and expired_id is None:
            if action >= 0:
                release = float(1.0 - self.gpi[action])
            elif eligible_count:
                release = float((1.0 - self.gpi[eligible]).max())
        self.last_step = {
            "action": action, "committed": self.pending is not None,
            "decision_id": self.pending["decision_id"] if self.pending is not None else None,
            "new_commit": new_commit, "threshold": threshold, "conflict": conflict,
            "bg_release": release, "expired": expired_id is not None,
            "expired_decision_id": expired_id, "stopped": stop, "sleeping": not mods["awake"],
            "eligible_actions": eligible_count, "evidence_scale": evidence_scale,
            "ignored_modulators": ignored,
        }
        return dict(self.last_step, ignored_modulators=list(ignored))

    @staticmethod
    def _noop():
        return {"consumed": False, "rpe": 0.0, "action": -1, "decision_id": None,
                "learned": False, "reason": "no_matching_pending"}

    def _consume(self, reason, rpe=0.0, learned=False):
        pending = self.pending
        self.last_outcome = {
            "consumed": True, "rpe": float(rpe), "action": pending["action"],
            "decision_id": pending["decision_id"], "learned": learned, "reason": reason,
        }
        self.pending = None
        self.acc.zero_()
        self.counts[reason] += 1
        return dict(self.last_outcome)

    @torch.no_grad()
    def settle(self, decision_id: int, reward: float, *, learn: bool = True) -> dict:
        """Consume exactly once: RPE = reward - critic value frozen at commitment.

        Only the selected action's contextual D1/D2 row and the critic may learn.
        No bootstrapping from a later distractor; no second DA/RPE on stale ids.
        Even no-op settlements validate controls before checking the ticket.
        """
        self.cfg.validate()
        _integer("decision_id", decision_id, 1, 2**63 - 1)
        reward = _number("reward", reward, -self.cfg.reward_bound, self.cfg.reward_bound)
        learn = _boolean("learn", learn)
        enabled = _boolean("learning_enabled", self.learning_enabled)
        if self.pending is None or self.pending["decision_id"] != decision_id:
            return self._noop()
        p, c = self.pending, self.cfg
        rpe = reward - p["predicted_value"]
        do_learn = bool(learn and enabled and p["learnable"] and p["plasticity_scale"] > 0.0)
        changed = False
        if do_learn:
            delta = rpe * p["plasticity_scale"] * p["features"]
            action = p["action"]
            go = (self.w_go[action] + c.learning_rate * delta).clamp(0.0, c.weight_max)
            nogo = (self.w_nogo[action] - c.learning_rate * delta).clamp(0.0, c.weight_max)
            critic = (self.w_v + c.critic_learning_rate * delta).clamp(-c.critic_weight_max, c.critic_weight_max)
            changed = not (torch.equal(go, self.w_go[action]) and torch.equal(nogo, self.w_nogo[action])
                           and torch.equal(critic, self.w_v))
            self.w_go[action].copy_(go)
            self.w_nogo[action].copy_(nogo)
            self.w_v.copy_(critic)
            self.counts["updates"] += int(changed)
        return self._consume("settled", rpe, changed)

    def cancel(self) -> dict:
        """Consume the pending ticket without reward or learning; repeated calls no-op."""
        return self._noop() if self.pending is None else self._consume("cancelled")

    def telemetry(self) -> dict:
        """Independent JSON-safe snapshot; no tensors, callbacks, or parent links."""
        p = self.pending
        return {
            "config": {name: float(value) for name, value in asdict(self.cfg).items()},
            "budgets": {
                "n_actions": self.n_actions, "context_dim": self.context_dim,
                "max_actions": _MAX_ACTIONS, "max_context_dim": _MAX_CONTEXT_DIM,
                "max_pending": 1, "max_dt_ms": self.cfg.max_dt_ms,
                "credit_horizon_ms": self.cfg.credit_horizon_ms,
                "learned_parameters": self.w_go.numel() + self.w_nogo.numel() + self.w_v.numel(),
                "dtype": "float64", "device": "cpu", "context_l1_cap": 1.0,
                "evidence_scaling": "divide_by_max(1,max(evidence))",
                "evidence_admission": "strictly_positive", "parameter_ranges": dict(_CONFIG_RANGES),
            },
            "seed": self.seed, "elapsed_ms": self.elapsed_ms, "learning_enabled": self.learning_enabled,
            "counts": dict(self.counts), "outcome": dict(self.last_outcome),
            "pending": None if p is None else {
                key: p[key] for key in ("decision_id", "action", "age_ms", "predicted_value",
                                        "learnable", "plasticity_scale")
            },
            "last_step": dict(self.last_step, ignored_modulators=list(self.last_step["ignored_modulators"])),
            "gpe": self.gpe.tolist(), "stn": self.stn.tolist(), "gpi": self.gpi.tolist(),
            "accumulator": self.acc.tolist(),
        }
