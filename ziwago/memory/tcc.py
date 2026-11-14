"""TCC — toroidal continuous-attractor code for relations (blueprint §3.8).

Bounded relational-transfer EXPERIMENT, not a physical arena and not a drop-in
replacement for the hippocampus. The existing `Hippocampus.encode_ec` is a 1-D
grid-like code cos(w*proj+phi); this module is a genuine 2-D continuous attractor
(CANN) on the torus T^2 = (S^1)^2, so wrap, bump stability, interference and
reversible decoding can be measured rather than assumed.

Honest boundaries (blueprint §3.8):
- A commutative phase-sum of two relation codes cannot represent a directed /
  non-commutative relation. `phase_sum_bind` is order-independent by construction;
  direction needs an extra typed operator (`typed_directional_bind`, PhaseVSA-style
  role tags). Both are provided so the limitation is demonstrated, not hidden.
- Gridness / a stable bump does NOT prove textual causality. Decoding accuracy here
  is geometric nearest-anchor recall on synthetic vectors, not a language benchmark.
- Opt-in (enabled=False), standalone, NOT wired into live generation. No global
  backprop (fixed random projection + stored anchors), no n-grams, checkpoints and
  loader untouched. Scale only after a measured advantage over the VSA baseline.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_TWO_PI = 2.0 * math.pi


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle, _TWO_PI)


def toroidal_distance(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    """Geodesic distance on T^2 with wrap: each coordinate difference is taken on
    the circle, so 0 and 2*pi-eps are neighbours (distance eps, not 2*pi-eps)."""
    d1 = abs((p[0] - q[0] + math.pi) % _TWO_PI - math.pi)
    d2 = abs((p[1] - q[1] + math.pi) % _TWO_PI - math.pi)
    return math.sqrt(d1 * d1 + d2 * d2)


def phase_sum_bind(theta_a: Tuple[float, float],
                   theta_b: Tuple[float, float]) -> Tuple[float, float]:
    """Commutative toroidal phase-sum. bind(a,b) == bind(b,a) exactly, so this
    operator CANNOT encode direction A->B vs B->A (blueprint §3.8 limitation)."""
    return ((theta_a[0] + theta_b[0]) % _TWO_PI,
            (theta_a[1] + theta_b[1]) % _TWO_PI)


def _to_phasor(vec: torch.Tensor) -> torch.Tensor:
    v = vec.detach().float().flatten()
    ang = torch.tanh(v / v.norm().clamp(min=1e-6) * math.pi)
    return torch.complex(torch.cos(ang), torch.sin(ang))


def typed_directional_bind(vec_a: torch.Tensor, vec_b: torch.Tensor,
                           role_agent: torch.Tensor,
                           role_patient: torch.Tensor) -> torch.Tensor:
    """Non-commutative typed binding (PhaseVSA-style). Role tags are distinct types
    and the conjugate lands on the patient, so bind(A,B) != bind(B,A): direction is
    representable only because of the extra typed operator, not the phase-sum."""
    za, zb = _to_phasor(vec_a), _to_phasor(vec_b)
    ra, rp = _to_phasor(role_agent), _to_phasor(role_patient)
    return ra * za * rp * zb.conj()


class ToroidalRelationAttractor:
    """Rate-based CANN on a discretized T^2 with a first-Fourier-mode toroidal
    kernel W = -w_inh + w_exc*(cos(dphi1) + cos(dphi2)). The kernel is periodic by
    construction (exact wrap) and applied in O(N^2) via the Fourier decomposition
    instead of an N^4 dense matmul. A localized bump is a stable fixed point of the
    leaky sigmoid dynamics; decoding reads the activity-weighted circular mean and
    returns the nearest stored anchor.

    Honest amplitude (2026-11-13): the drive is the MEAN-FIELD application of the
    kernel, not a dense sum: the DC mode multiplies the mean activity (`-w_inh·mean(a)`,
    so a constant field gives exactly −w_inh), the first harmonic is applied with its
    exact projection (2/N per axis, so a first-harmonic profile `a` gives exactly
    `w_exc·a`). Both `w_inh` and `w_exc` are therefore the coefficients that actually
    multiply their mode; the old code realised only `w_exc/2` for the first harmonic.
    To keep the calibrated dynamics identical the default `w_exc` is 3.0 (it used to be
    declared 6.0 while the realised coefficient was 3.0). For a constant field the
    declared kernel would give `-w_inh·N²` with a dense sum — that is NOT what this
    operator does and not what the dynamics are calibrated for.
    """

    def __init__(self, grid_size: int = 24, d_vec: int = 32,
                 w_inh: float = 3.0, w_exc: float = 3.0, beta: float = 8.0,
                 theta: float = 0.3, dt: float = 0.5, bump_width: float = 0.6,
                 seed: int = 0, enabled: bool = False):
        if (int(grid_size) < 4 or int(d_vec) < 1
                or not all(math.isfinite(x) for x in (w_inh, w_exc, beta, theta, dt, bump_width))
                or w_inh < 0 or w_exc < 0 or beta <= 0 or not 0 < dt <= 1 or bump_width <= 0):
            raise ValueError("invalid TCC configuration")
        self.grid_size = int(grid_size)
        self.d_vec = int(d_vec)
        self.w_inh, self.w_exc = float(w_inh), float(w_exc)
        self.beta, self.theta, self.dt = float(beta), float(theta), float(dt)
        self.bump_width = float(bump_width)
        self.enabled = bool(enabled)
        g = torch.Generator().manual_seed(int(seed))
        idx = torch.arange(self.grid_size, dtype=torch.float32)
        phi = _TWO_PI * idx / self.grid_size
        self.c1, self.s1 = torch.cos(phi), torch.sin(phi)
        self.c2, self.s2 = self.c1.clone(), self.s1.clone()
        self.phi = phi
        self.proj = torch.randn(2, self.d_vec, generator=g) / math.sqrt(self.d_vec)
        self.activity = torch.zeros(self.grid_size, self.grid_size)
        self.anchors: Dict[str, Tuple[float, float]] = {}
        self.anchor_vecs: Dict[str, torch.Tensor] = {}
        self.last_report: Dict[str, object] = {}

    # ── geometry ────────────────────────────────────────────────────────────
    def encode_vector(self, vec: torch.Tensor) -> Tuple[float, float]:
        v = torch.as_tensor(vec, dtype=torch.float32).flatten()
        if v.numel() != self.d_vec or not torch.isfinite(v).all():
            raise ValueError("relation vector must be finite with d_vec entries")
        proj = self.proj @ v
        return (float(_wrap(proj[0])), float(_wrap(proj[1])))

    def _bump_profile(self, theta1: float, theta2: float,
                      width: float) -> torch.Tensor:
        d1 = torch.abs(_wrap(self.phi - theta1 + math.pi) - math.pi)
        d2 = torch.abs(_wrap(self.phi - theta2 + math.pi) - math.pi)
        dist2 = d1[:, None] ** 2 + d2[None, :] ** 2
        return torch.exp(-dist2 / (2.0 * width * width))

    def place_bump(self, theta1: float, theta2: float,
                   width: Optional[float] = None) -> None:
        w = self.bump_width if width is None else float(width)
        if not math.isfinite(w) or w <= 0:
            raise ValueError("bump width must be finite and positive")
        self.activity = self._bump_profile(theta1, theta2, w)

    def bump_center(self) -> Tuple[float, float]:
        a = self.activity
        total = float(a.sum())
        if total <= 1e-9:
            return (0.0, 0.0)
        t1 = math.atan2(float((a * self.s1[:, None]).sum()),
                        float((a * self.c1[:, None]).sum())) % _TWO_PI
        t2 = math.atan2(float((a * self.s2[None, :]).sum()),
                        float((a * self.c2[None, :]).sum())) % _TWO_PI
        return (t1, t2)

    def localization(self) -> float:
        """Resultant length of the activity-weighted phase distribution in [0,1];
        1 = a single sharp bump, 0 = uniform / silent (no localized code)."""
        a = self.activity
        total = float(a.sum())
        if total <= 1e-9:
            return 0.0
        r1 = math.hypot(float((a * self.c1[:, None]).sum()),
                        float((a * self.s1[:, None]).sum())) / total
        r2 = math.hypot(float((a * self.c2[None, :]).sum()),
                        float((a * self.s2[None, :]).sum())) / total
        return float(min(1.0, 0.5 * (r1 + r2)))

    # ── dynamics ────────────────────────────────────────────────────────────
    def recurrent_drive(self) -> torch.Tensor:
        """Рекуррентный вход каждой единицы: точное применение ядра
        `-w_inh + w_exc·(cos Δφ1 + cos Δφ2)` через проекцию первой гармоники.

        Публичный метод (не `_recurrent_drive`): амплитуда ядра измерима снаружи —
        иначе заявленные w_inh/w_exc нельзя проверить, не читая код.
        """
        a = self.activity
        mean_a = a.mean()
        m_c1 = (a * self.c1[:, None]).mean()
        m_s1 = (a * self.s1[:, None]).mean()
        m_c2 = (a * self.c2[None, :]).mean()
        m_s2 = (a * self.s2[None, :]).mean()
        # 2/N по каждой оси — точная проекция: для a = cos(φ−θ) даёт ровно w_exc·a
        return (-self.w_inh * mean_a
                + 2.0 * self.w_exc * (self.c1[:, None] * m_c1 + self.s1[:, None] * m_s1
                                      + self.c2[None, :] * m_c2 + self.s2[None, :] * m_s2))

    # обратная совместимость
    _recurrent_drive = recurrent_drive

    def step(self, external: Optional[torch.Tensor] = None, steps: int = 1) -> torch.Tensor:
        if int(steps) < 1:
            raise ValueError("steps must be a positive integer")
        if not self.enabled:
            # opt-in выключен — динамика НЕ идёт (раньше флаг ни на что не влиял)
            self.last_report = {"disabled": True, "activity_sum": float(self.activity.sum())}
            return self.activity
        drive_in = None
        if external is not None:
            drive_in = torch.as_tensor(external, dtype=torch.float32)
            if drive_in.shape != self.activity.shape:
                raise ValueError("external input must match the grid shape")
        for _ in range(int(steps)):
            drive = self.recurrent_drive()
            if drive_in is not None:
                drive = drive + drive_in
            target = torch.sigmoid(self.beta * (drive - self.theta))
            self.activity = (1.0 - self.dt) * self.activity + self.dt * target
        return self.activity

    def reset(self) -> None:
        self.activity = torch.zeros(self.grid_size, self.grid_size)
        self.last_report = {}

    # ── relational memory ───────────────────────────────────────────────────
    def store_relation(self, label: str, vec: torch.Tensor) -> Tuple[float, float]:
        if not label:
            raise ValueError("relation label must be non-empty")
        theta = self.encode_vector(vec)
        self.anchors[label] = theta
        self.anchor_vecs[label] = torch.as_tensor(vec, dtype=torch.float32).flatten().clone()
        return theta

    def decode(self) -> Dict[str, object]:
        center = self.bump_center()
        silent = float(self.activity.sum()) <= 1e-9
        if not self.anchors or silent:
            self.last_report = {"decoded": None, "center": center, "distance": None,
                                "localization": self.localization()}
            return self.last_report
        best_label, best_dist = None, float("inf")
        for label, theta in self.anchors.items():
            d = toroidal_distance(center, theta)
            if d < best_dist:
                best_label, best_dist = label, d
        self.last_report = {"decoded": best_label, "center": center,
                            "distance": float(best_dist),
                            "localization": self.localization()}
        return self.last_report

    def recall(self, label: str, settle_steps: int = 12,
               anchor_gain: float = 0.0) -> Dict[str, object]:
        """Place a bump at a stored relation, optionally settle under dynamics, and
        decode. anchor_gain>0 pins the bump with a weak persistent cue; 0 = free."""
        if label not in self.anchors:
            raise KeyError(f"unknown relation {label!r}")
        t1, t2 = self.anchors[label]
        self.place_bump(t1, t2)
        external = None
        if anchor_gain > 0:
            external = anchor_gain * self._bump_profile(t1, t2, self.bump_width)
        if settle_steps > 0:
            self.step(external=external, steps=settle_steps)
        return self.decode()

    # ── diagnostics required by blueprint §3.8 ──────────────────────────────
    def bump_stability(self, label: str, steps: int = 40,
                       anchor_gain: float = 0.0) -> Dict[str, float]:
        t1, t2 = self.anchors[label]
        self.place_bump(t1, t2)
        start = self.bump_center()
        loc0 = self.localization()
        external = (anchor_gain * self._bump_profile(t1, t2, self.bump_width)
                    if anchor_gain > 0 else None)
        self.step(external=external, steps=steps)
        end = self.bump_center()
        return {"center_drift": toroidal_distance(start, end),
                "target_drift": toroidal_distance(end, (t1, t2)),
                "localization_before": loc0,
                "localization_after": self.localization(),
                "steps": float(steps), "anchor_gain": float(anchor_gain)}

    def interference(self, label_a: str, label_b: str,
                     steps: int = 30) -> Dict[str, float]:
        """Two bumps coexist when far apart and merge/compete when close. Returns
        the surviving localization and how far the decoded center is from each."""
        ta, tb = self.anchors[label_a], self.anchors[label_b]
        sep = toroidal_distance(ta, tb)
        self.activity = (self._bump_profile(*ta, self.bump_width)
                         + self._bump_profile(*tb, self.bump_width))
        self.step(steps=steps)
        center = self.bump_center()
        return {"separation": sep, "localization_after": self.localization(),
                "dist_to_a": toroidal_distance(center, ta),
                "dist_to_b": toroidal_distance(center, tb)}

    def aliasing(self, tol: float = 0.15) -> Dict[str, object]:
        """Distinct labels whose encoded phase points collide within tol on T^2:
        a many-to-one projection aliases relations (honest failure mode)."""
        labels = list(self.anchors)
        collisions: List[Tuple[str, str]] = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if toroidal_distance(self.anchors[labels[i]], self.anchors[labels[j]]) < tol:
                    collisions.append((labels[i], labels[j]))
        return {"n_relations": len(labels), "n_collisions": len(collisions),
                "collisions": collisions, "tol": float(tol)}

    # ── serialization / telemetry ───────────────────────────────────────────
    def telemetry(self) -> Dict[str, object]:
        return {"grid_size": self.grid_size, "d_vec": self.d_vec,
                "n_relations": len(self.anchors), "enabled": self.enabled,
                "w_inh": self.w_inh, "w_exc": self.w_exc, "beta": self.beta,
                "theta": self.theta, "dt": self.dt, "bump_width": self.bump_width,
                "localization": self.localization()}

    def export_state_dict(self) -> Dict[str, object]:
        return {"grid_size": self.grid_size, "d_vec": self.d_vec,
                "w_inh": self.w_inh, "w_exc": self.w_exc, "beta": self.beta,
                "theta": self.theta, "dt": self.dt, "bump_width": self.bump_width,
                "enabled": self.enabled, "proj": self.proj.clone(),
                "activity": self.activity.clone(),
                "anchors": dict(self.anchors),
                "anchor_vecs": {k: v.clone() for k, v in self.anchor_vecs.items()}}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        loaded = type(self)(int(state["grid_size"]), int(state["d_vec"]),
                            float(state["w_inh"]), float(state["w_exc"]),
                            float(state["beta"]), float(state["theta"]),
                            float(state["dt"]), float(state["bump_width"]),
                            enabled=bool(state["enabled"]))
        loaded.proj = torch.as_tensor(state["proj"]).clone()
        if loaded.proj.shape != (2, loaded.d_vec):
            raise ValueError("TCC projection shape mismatch")
        loaded.activity = torch.as_tensor(state["activity"]).clone()
        if loaded.activity.shape != (loaded.grid_size, loaded.grid_size):
            raise ValueError("TCC activity shape mismatch")
        loaded.anchors = dict(state["anchors"])
        loaded.anchor_vecs = {k: torch.as_tensor(v).clone()
                              for k, v in state["anchor_vecs"].items()}
        self.__dict__.update(loaded.__dict__)


def _vsa_nn_baseline(vecs: Dict[str, torch.Tensor],
                     probe: torch.Tensor) -> Tuple[str, float]:
    """Plain cosine nearest-neighbour VSA recall — the cheap baseline TCC must beat
    on quality AND cost before it is scaled (blueprint §3.8 acceptance)."""
    q = F.normalize(probe.flatten(), dim=0)
    best_label, best_sim = None, -2.0
    for label, v in vecs.items():
        sim = float(torch.dot(q, F.normalize(v.flatten(), dim=0)))
        if sim > best_sim:
            best_label, best_sim = label, sim
    return best_label, best_sim


def run_acceptance_experiment(n_relations: int = 8, grid_size: int = 24,
                              d_vec: int = 32, seed: int = 0,
                              settle_steps: int = 12) -> Dict[str, object]:
    """Bounded acceptance harness (blueprint §3.8): wrap, bump stability,
    interference, reversible decoding, held-out alternative-path composition,
    aliasing,     the commutative limitation, and a paired noisy-retrieval quality+cost
    comparison against a cosine-VSA baseline. Exact-recall results are sanity
    checks; only corrupted probes count toward the advantage flag. Synthetic
    vectors only — this is NOT a language benchmark and gridness is not evidence
    of textual causality. Scale only when the reported advantage is True."""
    g = torch.Generator().manual_seed(seed)
    tcc = ToroidalRelationAttractor(grid_size=grid_size, d_vec=d_vec, seed=seed,
                                    enabled=True)
    labels = [f"rel_{i}" for i in range(n_relations)]
    vecs = {lab: F.normalize(torch.randn(d_vec, generator=g), dim=0)
            for lab in labels}
    for lab in labels:
        tcc.store_relation(lab, vecs[lab])

    # wrap: toroidal metric must treat 0 and 2*pi-eps as neighbours
    wrap_err = abs(toroidal_distance((0.0, 0.0), (_TWO_PI - 1e-3, 0.0)) - 1e-3)

    # reversible decoding on stored relations (free settle, no anchor pin)
    t0 = time.perf_counter()
    hits = 0
    for lab in labels:
        if tcc.recall(lab, settle_steps=settle_steps)["decoded"] == lab:
            hits += 1
    tcc_decode_time = time.perf_counter() - t0
    decode_acc = hits / max(1, n_relations)

    # bump stability (free) and pinned
    stab_free = tcc.bump_stability(labels[0], steps=40, anchor_gain=0.0)
    stab_pin = tcc.bump_stability(labels[0], steps=40, anchor_gain=0.5)

    # interference: nearest vs farthest pair by toroidal separation
    pairs = [(a, b, toroidal_distance(tcc.anchors[a], tcc.anchors[b]))
             for i, a in enumerate(labels) for b in labels[i + 1:]]
    pairs.sort(key=lambda x: x[2])
    near = tcc.interference(pairs[0][0], pairs[0][1])
    far = tcc.interference(pairs[-1][0], pairs[-1][1])

    # held-out alternative-path composition: a new relation built as the phase-sum
    # of two stored ones, decoded by geometry (commutative => direction lost).
    a_lab, b_lab = labels[0], labels[1]
    comp = phase_sum_bind(tcc.anchors[a_lab], tcc.anchors[b_lab])
    tcc.place_bump(*comp)
    heldout = tcc.decode()

    # commutative limitation: phase-sum is order-independent; typed bind is not.
    role_a = F.normalize(torch.randn(d_vec, generator=g), dim=0)
    role_p = F.normalize(torch.randn(d_vec, generator=g), dim=0)
    sum_ab = phase_sum_bind(tcc.anchors[a_lab], tcc.anchors[b_lab])
    sum_ba = phase_sum_bind(tcc.anchors[b_lab], tcc.anchors[a_lab])
    commutative = sum_ab == sum_ba
    typed_ab = typed_directional_bind(vecs[a_lab], vecs[b_lab], role_a, role_p)
    typed_ba = typed_directional_bind(vecs[b_lab], vecs[a_lab], role_a, role_p)
    typed_sim = float(torch.dot(F.normalize(typed_ab.real, dim=0),
                                F.normalize(typed_ba.real, dim=0)))

    alias = tcc.aliasing()

    # Quality/cost baseline: cosine-VSA NN on the same vectors / same probes.
    # Wall-clock timings are reported as diagnostics, not used as acceptance criteria.
    t0 = time.perf_counter()
    base_hits = 0
    for lab in labels:
        if _vsa_nn_baseline(vecs, vecs[lab])[0] == lab:
            base_hits += 1
    baseline_time = time.perf_counter() - t0
    baseline_acc = base_hits / max(1, n_relations)

    probe_noise_scale = 0.5
    noisy_tcc_hits = noisy_base_hits = 0
    for lab in labels:
        noisy_query = F.normalize(
            vecs[lab] + probe_noise_scale * torch.randn(d_vec, generator=g), dim=0)
        if _vsa_nn_baseline(vecs, noisy_query)[0] == lab:
            noisy_base_hits += 1
        tcc.place_bump(*tcc.encode_vector(noisy_query))
        tcc.step(steps=settle_steps)
        if tcc.decode()["decoded"] == lab:
            noisy_tcc_hits += 1
    noisy_decode_acc = noisy_tcc_hits / max(1, n_relations)
    noisy_baseline_acc = noisy_base_hits / max(1, n_relations)

    tcc_params = tcc.proj.numel() + sum(v.numel() for v in tcc.anchor_vecs.values())
    baseline_params = sum(v.numel() for v in vecs.values())
    tcc_state = tcc.export_state_dict()
    tcc_tensor_bytes = sum(
        value.numel() * value.element_size()
        for value in (tcc_state["proj"], tcc_state["activity"],
                      *tcc_state["anchor_vecs"].values())
    )
    baseline_tensor_bytes = sum(v.numel() * v.element_size() for v in vecs.values())
    advantage = bool(noisy_decode_acc > noisy_baseline_acc
                     and tcc_tensor_bytes <= baseline_tensor_bytes)

    return {
        "n_relations": n_relations, "grid_size": grid_size, "d_vec": d_vec,
        "wrap_error": wrap_err,
        "decode_accuracy": decode_acc, "decode_time_s": tcc_decode_time,
        "bump_stability_free": stab_free, "bump_stability_pinned": stab_pin,
        "interference_near": near, "interference_far": far,
        "heldout_decoded": heldout["decoded"], "heldout_distance": heldout["distance"],
        "phase_sum_commutative": commutative,
        "typed_directional_similarity": typed_sim,
        "aliasing": alias,
        "baseline_accuracy": baseline_acc, "baseline_time_s": baseline_time,
        "quality_metric": "paired_noisy_retrieval_accuracy",
        "probe_noise_scale": probe_noise_scale,
        "noisy_decode_accuracy": noisy_decode_acc,
        "noisy_baseline_accuracy": noisy_baseline_acc,
        "tcc_params": int(tcc_params), "baseline_params": int(baseline_params),
        "cost_metric": "persistent_tensor_bytes",
        "tcc_persistent_tensor_bytes": int(tcc_tensor_bytes),
        "baseline_persistent_tensor_bytes": int(baseline_tensor_bytes),
        "advantage_over_baseline": advantage,
        "synthetic_only": True, "textual_causality_proven": False,
    }