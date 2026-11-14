"""CPU-small, deterministic contract tests for the standalone v20 owner.

No runtime/core construction, GPU tensors, checkpoints on disk or training.
Run in the main worker's exclusive test slot, not beside a training run.
"""
from __future__ import annotations

from dataclasses import asdict
import io
import json
import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ziwago.chemistry.modulators import NeuroChemistry
from ziwago.config import ChemistryCfg
from ziwago.organism.physiology import (
    OrganismPhysiology, PhysiologyConfig, V20ChemState,
)


def _snapshot(body):
    return (json.dumps(body.telemetry(), sort_keys=True, allow_nan=False),
            body.ticks_awake, body.ticks_asleep)


@pytest.mark.parametrize("rpe", [-2.0, 2.0])
def test_reward_is_one_signed_impulse_and_has_100ms_decay(rpe):
    body = OrganismPhysiology(2)
    before = asdict(body.s)
    atp, ado = body.zone_atp.clone(), body.zone_ado.clone()
    result = body.reward(rpe)
    impulse = rpe * body.physiology_cfg.phasic_reward_gain
    assert result["da_phasic"] == pytest.approx(impulse)
    assert body.time_ms == body.tick == body.phase_elapsed_ms == 0
    assert body.s.da == before["da"]
    assert {k: v for k, v in asdict(body.s).items() if k != "da_phasic"} == {
        k: v for k, v in before.items() if k != "da_phasic"
    }
    torch.testing.assert_close(body.zone_atp, atp, rtol=0, atol=0)
    torch.testing.assert_close(body.zone_ado, ado, rtol=0, atol=0)
    for elapsed in (50.0, 100.0, 150.0):
        body.advance(torch.zeros(2), dt_ms=50.0)
        assert body.s.da_phasic == pytest.approx(impulse * math.exp(-elapsed / 100.0), abs=1e-14)
        assert body.s.da == before["da"]
    assert body.reward_events == 1  # advance does not replay the RPE
    trace = body.s.da_phasic
    body.reward(-rpe)
    assert body.s.da_phasic == pytest.approx(trace - impulse)
    assert body.time_ms == 150.0
    assert body.tick == 3


def test_tonic_da_relaxes_independently_and_impulse_is_bounded():
    body = OrganismPhysiology(1)
    body.s.da = 0.9
    body.reward(1e100)
    assert body.s.da_phasic == body.physiology_cfg.phasic_limit
    body.advance(torch.zeros(1), dt_ms=100.0)
    assert body.s.da == pytest.approx(0.5 + 0.4 * math.exp(-100.0 / body.cfg.da_tau_tonic))
    body.reward(-1e100)
    assert body.s.da_phasic == -body.physiology_cfg.phasic_limit


def test_dt_refinement_and_continuous_sources_have_finite_exponential_targets():
    coarse, fine, finer = (OrganismPhysiology(2) for _ in range(3))
    load = torch.tensor([0.8, 0.1], dtype=torch.float64)
    inputs = dict(ca_mean=0.4, novelty=0.6, attention=0.7, stress=0.3, drive=1.0)
    for body, step, count in ((coarse, 1000.0, 1), (fine, 10.0, 100), (finer, 2.0, 500)):
        body.reward(-0.5)
        for _ in range(count):
            body.advance(load, dt_ms=step, **inputs)
        assert body.time_ms == 1000.0
        assert body.phase == "awake"
    assert (coarse.tick, fine.tick, finer.tick) == (1, 100, 500)
    for body in (fine, finer):
        torch.testing.assert_close(body.zone_atp, coarse.zone_atp, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(body.zone_ado, coarse.zone_ado, rtol=1e-11, atol=1e-12)
        assert body.process_s == pytest.approx(coarse.process_s, abs=1e-11)
        # Coupled midpoint axes converge; unchanged axes and pools remain analytic.
        assert body.vlpo == pytest.approx(coarse.vlpo, abs=1e-5)
        assert body.s.ox == pytest.approx(coarse.s.ox, abs=1e-5)
        for name in ("da", "da_phasic", "ne", "ht", "cort", "ecb", "no"):
            assert getattr(body.s, name) == pytest.approx(getattr(coarse.s, name), abs=1e-12)
    # Identical 10 ms substeps must give identical ACh regardless of call count.
    assert fine.s.ach == pytest.approx(coarse.s.ach, rel=0, abs=1e-12)
    p, c = coarse.physiology_cfg, coarse.cfg
    for name, initial, target, tau in (
        ("ne", 0.3, p.ne_baseline + p.ne_novelty_gain * 0.6, c.ne_tau),
        ("cort", 0.1, p.cort_baseline + p.cort_stress_gain * 0.3, c.cort_tau_nongenomic),
        ("ecb", 0.0, p.ecb_target_max * 0.4 / (0.4 + p.calcium_half_saturation), c.ecb_tau),
        ("no", 0.0, p.no_target_max * 0.4 / (0.4 + p.calcium_half_saturation), c.no_tau),
    ):
        expected = target + (initial - target) * math.exp(-1000.0 / tau)
        assert getattr(coarse.s, name) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("sleeping", [False, True])
def test_ach_coupled_source_converges_at_second_order(sleeping):
    # With S(t)=S0*exp(-t/tau_s), choosing tau_ach=tau_s/2 gives an analytic reference.
    steps = (20.0, 10.0, 5.0)
    duration, pressure, attention = 1000.0, 0.5, 0.7
    errors = []
    for step in steps:
        p = PhysiologyConfig(integration_step_ms=step, ach_ado_half_saturation=0.3)
        tau_s = p.process_s_sleep_tau_ms if sleeping else p.process_s_tau_ms
        tau = tau_s / 2.0
        body = OrganismPhysiology(1, cfg=p, chemistry_cfg=ChemistryCfg(ach_tau=tau))
        body.process_s = pressure
        if sleeping:
            body.request_sleep()
        initial = body.s.ach
        source = p.ach_baseline + p.ach_attention_gain * attention
        q, x = (pressure / p.ach_ado_half_saturation) ** 2, duration / tau
        decay = math.exp(-x)
        expected = (source + (initial - source) * decay
                    - source * q * decay * math.log1p(math.expm1(x) / (1.0 + q)))
        body.advance(torch.zeros(1), attention=attention, dt_ms=duration)
        assert body.phase == ("nrem" if sleeping else "awake")
        assert body.time_ms == duration
        assert body.process_s == pytest.approx(pressure * math.exp(-duration / tau_s),
                                               rel=0, abs=1e-12)
        errors.append(abs(body.s.ach - expected))
    assert errors[0] > errors[1] > errors[2] > 64 * math.ulp(expected)
    for i in range(len(errors) - 1):
        # Bound the relative higher-order remainder by h/tau, not an absolute epsilon.
        assert errors[i] / errors[i + 1] == pytest.approx(
            (steps[i] / steps[i + 1]) ** 2, rel=steps[i] / tau)


def test_ach_uses_midpoint_pressure_and_same_load_has_lower_ach_at_higher_pressure():
    low, high = OrganismPhysiology(2), OrganismPhysiology(2)
    load = torch.tensor([0.2, 0.8], dtype=torch.float64)
    inputs = dict(ca_mean=0.4, novelty=0.6, attention=0.7, stress=0.3, drive=1.0)
    for body, pressure in ((low, 0.1), (high, 0.8)):
        body.process_s = pressure
        initial = body.s.ach
        p = body.physiology_cfg
        dt = p.integration_step_ms  # Exactly one substep, no phase change.
        body.advance(load, dt_ms=dt, **inputs)
        midpoint = (pressure + body.process_s) / 2.0
        source = p.ach_baseline + p.ach_attention_gain * inputs["attention"]
        target = source / (1.0 + (midpoint / p.ach_ado_half_saturation) ** 2)
        expected = initial + (target - initial) * -math.expm1(-dt / body.cfg.ach_tau)
        assert body.s.ach == pytest.approx(expected, rel=0, abs=1e-14)
        assert min(initial, target) <= body.s.ach <= max(initial, target)
        assert body.phase == "awake"
    torch.testing.assert_close(low.zone_atp, high.zone_atp, rtol=0, atol=0)
    torch.testing.assert_close(low.zone_ado, high.zone_ado, rtol=0, atol=0)
    assert high.process_s > low.process_s
    assert high.s.ach < low.s.ach


@pytest.mark.parametrize("pressure", [0.0, 0.3, 0.6])
def test_ach_steady_pressure_has_exact_bounded_target(pressure):
    body = OrganismPhysiology(1)
    p, c = body.physiology_cfg, body.cfg
    # Zonal Ado and Process S start at the constant-load awake equilibrium.
    load = pressure / ((1.0 - pressure) * p.ado_production_per_ms * p.ado_clearance_tau_ms)
    body.zone_ado.fill_(pressure)
    body.process_s = pressure
    attention = 2.0  # Attention still saturates at one before suppression.
    source = p.ach_baseline + p.ach_attention_gain * min(attention, 1.0)
    factor = 1.0 / (1.0 + (pressure / p.ach_ado_half_saturation) ** 2)
    target, initial, elapsed = source * factor, body.s.ach, 0.0
    for tau_count in (1, 3, 8):
        dt = tau_count * c.ach_tau
        elapsed += dt
        body.advance(torch.tensor([load], dtype=torch.float64), attention=attention, dt_ms=dt)
        expected = target + (initial - target) * math.exp(-elapsed / c.ach_tau)
        assert body.phase == "awake"
        assert body.process_s == pytest.approx(pressure, rel=0, abs=1e-12)
        assert body.s.ach == pytest.approx(expected, rel=0, abs=1e-12)
        assert body.telemetry()["ach_pressure_suppression"] == pytest.approx(factor, rel=0, abs=1e-12)
    # Steady-target tolerance is precisely the remaining exponential transient.
    assert body.s.ach == pytest.approx(
        target, rel=0, abs=abs(initial - target) * math.exp(-elapsed / c.ach_tau) + 1e-12)


@pytest.mark.parametrize("theta,initial,attention", [
    (0.6, 0.0, 0.0),
    (0.6, 0.5, 2.0),
    (0.6, 2.0, 1.0),
    (1e-300, 0.5, 1.0),
    (1e300, 0.5, 1.0),
])
def test_ach_stays_finite_nonnegative_and_bounded(theta, initial, attention):
    body = OrganismPhysiology(1, cfg=PhysiologyConfig(ach_ado_half_saturation=theta))
    p = body.physiology_cfg
    body.process_s = 1.0
    body.zone_ado.fill_(1.0)
    body.s.ach = initial
    upper = max(initial, p.ach_baseline + p.ach_attention_gain * min(1.0, attention))
    for dt in (1.0, 100.0, 1000.0):
        body.advance(torch.ones(1), attention=attention, dt_ms=dt)
        assert math.isfinite(body.s.ach)
        assert 0.0 <= body.s.ach <= upper
        assert 0.0 <= body.telemetry()["ach_pressure_suppression"] <= 1.0


@pytest.mark.parametrize("sleeping", [False, True])
def test_ach_recovers_as_pressure_releases_without_changing_attention(sleeping):
    body = OrganismPhysiology(1)
    p = body.physiology_cfg
    attention, pressure = 0.7, 0.5
    source = p.ach_baseline + p.ach_attention_gain * attention
    body.process_s = pressure
    body.s.ach = source / (1.0 + (pressure / p.ach_ado_half_saturation) ** 2)
    initial = body.s.ach
    if sleeping:
        body.request_sleep()
        assert body.s.ach == initial  # Requesting sleep is not an ACh kick.
    tau_s = p.process_s_sleep_tau_ms if sleeping else p.process_s_tau_ms
    for elapsed in (500.0, 1000.0, 1500.0):
        before_ach, before_pressure = body.s.ach, body.process_s
        body.advance(torch.zeros(1), attention=attention, dt_ms=500.0)
        target = source / (1.0 + (body.process_s / p.ach_ado_half_saturation) ** 2)
        assert body.phase == ("nrem" if sleeping else "awake")
        assert body.process_s == pytest.approx(pressure * math.exp(-elapsed / tau_s), rel=0, abs=1e-12)
        assert 0.0 < body.process_s < before_pressure
        assert initial <= before_ach < body.s.ach < target < source


@pytest.mark.parametrize("load,kwargs", [
    ([0.0, 1.0], {}),
    (torch.zeros(1), {}),
    (torch.zeros(1, 2), {}),
    (torch.tensor([0.0, -0.1]), {}),
    (torch.tensor([0.0, float("nan")]), {}),
    (torch.tensor([float("inf"), 0.0]), {}),
    (torch.tensor([0j, 1j]), {}),
    (torch.tensor([True, False]), {}),
    (torch.zeros(2), {"dt_ms": -1.0}),
    (torch.zeros(2), {"dt_ms": float("nan")}),
    (torch.zeros(2), {"dt_ms": float("inf")}),
    (torch.zeros(2), {"ca_mean": float("nan")}),
    (torch.zeros(2), {"ca_mean": -0.1}),
    (torch.zeros(2), {"novelty": torch.ones(1)}),
    (torch.zeros(2), {"attention": float("inf")}),
    (torch.zeros(2), {"stress": -0.1}),
    (torch.zeros(2), {"drive": float("nan")}),
])
def test_invalid_advance_is_atomic(load, kwargs):
    body = OrganismPhysiology(2)
    body.advance(torch.tensor([0.2, 0.8]), ca_mean=0.3, dt_ms=25.0)
    body.reward(-1.0)
    before = _snapshot(body)
    with pytest.raises((ValueError, TypeError)):
        body.advance(load, **kwargs)
    assert _snapshot(body) == before


@pytest.mark.parametrize("rpe", [float("nan"), float("inf"), -float("inf"), [1.0], torch.ones(1)])
def test_invalid_reward_is_atomic(rpe):
    body = OrganismPhysiology(1)
    body.reward(-0.5)
    before = _snapshot(body)
    with pytest.raises((ValueError, TypeError)):
        body.reward(rpe)
    assert _snapshot(body) == before


def test_zero_duration_is_a_validated_noop_and_inputs_are_detached():
    body = OrganismPhysiology(2)
    load = torch.tensor([0.2, 0.9], requires_grad=True)
    body.advance(load, dt_ms=25.0)
    assert body.zone_load.device.type == "cpu"
    assert body.zone_load.dtype == torch.float64
    assert not body.zone_load.requires_grad
    assert not body.zone_atp.requires_grad
    saved_load = body.zone_load.clone()
    with torch.no_grad():
        load.fill_(100.0)
    torch.testing.assert_close(body.zone_load, saved_load, rtol=0, atol=0)
    body.process_s = 0.8  # Nontrivial suppression must not kick ACh at zero dt.
    before = _snapshot(body)
    body.advance(torch.ones(2), novelty=0.9, attention=1.0, dt_ms=0.0)
    assert _snapshot(body) == before
    with pytest.raises(ValueError):
        body.advance(torch.tensor([0.0, -1.0]), dt_ms=0.0)
    assert _snapshot(body) == before


def test_energy_and_zonal_adenosine_follow_measured_load_and_ca_then_recover():
    body = OrganismPhysiology(3)
    p = body.physiology_cfg
    load = torch.tensor([0.0, 1.0, 4.0], dtype=torch.float64)
    body.advance(load, ca_mean=0.5, dt_ms=1200.0)
    assert body.phase == "awake"
    effective = load + p.calcium_load_gain * 0.5
    recovery = 1.0 / p.atp_recovery_tau_ms
    rate = recovery + p.atp_burn_per_ms * effective
    target = recovery / rate
    expected = target + (1.0 - target) * torch.exp(-rate * 1200.0)
    torch.testing.assert_close(body.zone_atp, expected, rtol=1e-12, atol=1e-12)
    production = p.ado_production_per_ms * effective
    rate_ado = production + 1.0 / p.ado_clearance_tau_ms
    expected_ado = production / rate_ado * (1.0 - torch.exp(-rate_ado * 1200.0))
    torch.testing.assert_close(body.zone_ado, expected_ado, rtol=1e-12, atol=1e-12)
    assert 1.0 > body.zone_atp[0] > body.zone_atp[1] > body.zone_atp[2] > 0.0
    assert body.zone_ado[0] < body.zone_ado[1] < body.zone_ado[2]
    assert body.s.lactate == float(body.zone_atp.mean())
    assert body.s.ado == body.process_s
    assert body.process_s != pytest.approx(float(body.zone_ado.mean()))
    previous_atp, previous_ado = body.zone_atp.clone(), body.zone_ado.clone()
    body.advance(torch.zeros(3), dt_ms=800.0)
    expected_recovery = 1.0 + (previous_atp - 1.0) * math.exp(-800.0 / p.atp_recovery_tau_ms)
    torch.testing.assert_close(body.zone_atp, expected_recovery, rtol=1e-12, atol=1e-12)
    assert bool((body.zone_atp > previous_atp).all())
    assert bool((body.zone_ado < previous_ado).all())


def test_sleep_recovery_is_three_times_faster_but_load_is_not_invented():
    awake, asleep = OrganismPhysiology(1), OrganismPhysiology(1)
    for body in (awake, asleep):
        body.advance(torch.ones(1), dt_ms=1000.0)
    depleted = float(asleep.zone_atp[0])
    asleep.request_sleep()
    for body in (awake, asleep):
        body.advance(torch.zeros(1), dt_ms=500.0)
    assert asleep.phase == "nrem"
    assert asleep.zone_atp[0] > awake.zone_atp[0]
    assert float(asleep.zone_atp[0]) == pytest.approx(
        1.0 + (depleted - 1.0) * math.exp(-3.0 * 500.0 / 6000.0))
    # A caller that continues supplying work during sleep still burns ATP.
    asleep.advance(torch.full((1,), 1e4), dt_ms=100.0)
    assert asleep.s.lactate < 0.001  # no arbitrary lactate/ATP floor
    assert asleep.bundle()["f_mod"] == 0.0


def test_plasticity_gate_uses_mean_atp_and_really_closes():
    body = OrganismPhysiology(2)
    normal = body.f_mod()
    assert normal > 0.0
    body.zone_atp.fill_(0.5)
    assert body.f_mod() == pytest.approx(normal * 0.5)
    body.zone_atp.fill_(body.cfg.theta_energy)
    assert body.f_mod(energy_ok=1.0) == body.bundle()["f_mod"] == 0.0
    body.zone_atp.fill_(1.0)
    body.process_s = body.cfg.sleep_threshold
    assert body.s.ado == body.process_s
    assert body.f_mod() == 0.0
    body.process_s = 0.0
    assert body.f_mod(energy_ok=0.0) == 0.0
    body.s.da = body.s.ne = 0.0
    assert body.f_mod() == 0.0
    body.s.da, body.s.ne = 0.5, 0.3
    body.s.ecb = 2.0 / body.cfg.kappa_ecb
    assert body.f_mod() == 0.0
    body.s.ecb = 0.0
    body.request_sleep()
    assert body.f_mod() == 0.0
    body.advance(torch.zeros(2), dt_ms=body.physiology_cfg.nrem_min_ms)
    assert body.phase == "rem"
    assert body.f_mod() == 0.0


def test_natural_sleep_then_nrem_rem_and_recovered_wake():
    body = OrganismPhysiology(2)
    for _ in range(200):
        body.advance(torch.ones(2), drive=1.0, dt_ms=100.0)
        if body.phase == "nrem":
            break
    else:
        pytest.fail("sustained measured load did not produce natural sleep within 20,000 model ms")
    assert not body.is_awake
    assert body.f_mod() == 0.0
    sleep_event = body.phase_events[0]
    assert sleep_event["reason"] == "sleep_pressure"
    assert sleep_event["process_s"] >= body.cfg.sleep_threshold
    assert sleep_event["vlpo"] >= body.physiology_cfg.vlpo_sleep_threshold
    assert sleep_event["ox"] <= body.cfg.ox_collapse
    for _ in range(100):
        body.advance(torch.zeros(2), dt_ms=100.0)
        if body.phase == "awake":
            break
    else:
        pytest.fail("zero measured load did not permit natural recovery within 10,000 model ms")
    events = body.phase_events
    assert [event["to"] for event in events] == ["nrem", "rem", "awake"]
    assert all(event["provenance"] == "natural" for event in events)
    assert events[1]["phase_elapsed_ms"] >= body.physiology_cfg.nrem_min_ms
    assert events[2]["phase_elapsed_ms"] == pytest.approx(body.physiology_cfg.rem_duration_ms)
    assert events[2]["time_ms"] - events[1]["time_ms"] == pytest.approx(body.physiology_cfg.rem_duration_ms)
    # Wake does not reset pressure or refill ATP to one.
    assert 0.0 < events[2]["process_s"] <= body.physiology_cfg.process_s_wake_threshold
    assert body.physiology_cfg.atp_wake_threshold <= events[2]["atp_mean"] < 1.0
    assert body.is_awake and body.bundle()["awake"]
    assert body.f_mod() > 0.0


def test_explicit_sleep_preserves_pools_and_repeated_request_preserves_dwell():
    body = OrganismPhysiology(2)
    body.advance(torch.tensor([1.0, 2.0]), ca_mean=0.2, dt_ms=500.0)
    state = asdict(body.s)
    atp, ado, load = body.zone_atp.clone(), body.zone_ado.clone(), body.zone_load.clone()
    clock = (body.time_ms, body.tick)
    body.request_sleep()
    assert body.phase == "nrem" and not body.is_awake
    assert asdict(body.s) == state
    assert (body.time_ms, body.tick) == clock
    for before, after in ((atp, body.zone_atp), (ado, body.zone_ado), (load, body.zone_load)):
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert body.phase_events[-1]["provenance"] == "requested"
    body.advance(torch.zeros(2), dt_ms=100.0)
    body.request_sleep()
    assert body.phase_elapsed_ms == 100.0
    assert body.phase_events[-1]["event"] == "sleep_request"
    assert body.phase_events[-1]["from"] == body.phase_events[-1]["to"] == "nrem"


def test_minimum_nrem_and_bounded_rem_do_not_force_unrecovered_wake():
    body = OrganismPhysiology(1)
    p = body.physiology_cfg
    body.request_sleep()
    body.advance(torch.zeros(1), dt_ms=p.nrem_min_ms - 1.0)
    assert body.phase == "nrem"
    body.advance(torch.zeros(1), dt_ms=1.0)
    assert body.phase == "rem"
    body.advance(torch.full((1,), 1e4), dt_ms=p.rem_duration_ms)
    assert body.phase == "nrem" and not body.is_awake
    assert body.phase_events[-1]["reason"] == "rem_recovery_lost"
    assert body.phase_events[-1]["phase_elapsed_ms"] == p.rem_duration_ms
    assert not any(event["to"] == "awake" for event in body.phase_events)
    assert body.f_mod() == 0.0


def test_bundle_and_telemetry_are_json_safe_detached_and_keep_legacy_owner():
    chemistry_cfg = ChemistryCfg()
    body = OrganismPhysiology(2, chemistry_cfg=chemistry_cfg)
    assert isinstance(body, NeuroChemistry)
    assert isinstance(body.s, V20ChemState)
    assert body.cfg is chemistry_cfg
    assert body.update.__func__ is NeuroChemistry.update
    body.advance(torch.tensor([0.1, 0.4]), dt_ms=12.5)
    body.request_sleep()
    bundle = body.bundle()
    assert set(bundle) == {
        "ach", "ne", "da", "da_phasic", "ht", "ox", "ado", "cort", "ecb", "no",
        "lactate", "p_release", "f_mod", "awake", "phase", "time_ms", "tick",
    }
    assert isinstance(bundle["awake"], bool)
    assert isinstance(bundle["tick"], int)
    json.dumps(bundle, allow_nan=False)
    report = body.telemetry()
    json.dumps(report, allow_nan=False)
    theta = body.physiology_cfg.ach_ado_half_saturation
    assert report["physiology_config"]["ach_ado_half_saturation"] == theta == 0.6
    assert report["ach_pressure_suppression"] == pytest.approx(
        1.0 / (1.0 + (body.process_s / theta) ** 2), rel=0, abs=1e-15)
    assert "ach_pressure_suppression" not in vars(body)  # Derived, not cached.
    body.process_s = theta
    assert body.telemetry()["ach_pressure_suppression"] == 0.5
    assert report["ach_pressure_suppression"] > 0.5  # Detached earlier value.
    report["zone_atp"][0] = -1.0
    report["physiology_config"]["atp_recovery_tau_ms"] = -1.0
    report["physiology_config"]["ach_ado_half_saturation"] = -1.0
    report["phase_events"][0]["reason"] = "changed"
    assert body.zone_atp[0] >= 0.0
    assert body.physiology_cfg.atp_recovery_tau_ms == 6000.0
    assert body.physiology_cfg.ach_ado_half_saturation == theta
    assert body.phase_events[0]["reason"] == "requested"
    assert "owner" not in vars(body) and "core" not in vars(body)


@pytest.mark.parametrize("fresh", [True, False])
def test_runtime_tree_round_trip_restores_mutable_dataclasses_and_continuation(fresh):
    from ziwago.cortex.moe_cortex import export_runtime_tree, load_runtime_tree

    body = OrganismPhysiology(2, cfg=PhysiologyConfig(integration_step_ms=7.5,
                                                   ach_ado_half_saturation=0.35),
                             chemistry_cfg=ChemistryCfg(alpha_da=0.8))
    body.physiology_cfg.atp_recovery_tau_ms = 6500.0  # restore uses setattr too
    body.reward(-2.0)
    body.advance(torch.tensor([0.4, 1.2]), ca_mean=0.3, dt_ms=175.0)
    body.request_sleep()
    body.advance(torch.zeros(2), dt_ms=17.5)
    payload = export_runtime_tree(body)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, map_location="cpu", weights_only=True)
    restored = load_runtime_tree(OrganismPhysiology(1) if fresh else None, saved)
    assert isinstance(restored, OrganismPhysiology)
    assert isinstance(restored.physiology_cfg, PhysiologyConfig)
    assert isinstance(restored.cfg, ChemistryCfg)
    assert isinstance(restored.s, V20ChemState)
    assert restored.n_zones == 2
    assert restored.physiology_cfg.ach_ado_half_saturation == 0.35
    assert _snapshot(restored) == _snapshot(body)
    assert restored.zone_atp.data_ptr() != body.zone_atp.data_ptr()
    for item in (body, restored):
        item.advance(torch.tensor([0.1, 0.0]), attention=0.4, dt_ms=103.25)
        item.reward(0.25)
    assert _snapshot(restored) == _snapshot(body)
    torch.testing.assert_close(restored.zone_atp, body.zone_atp, rtol=0, atol=0)
    torch.testing.assert_close(restored.zone_ado, body.zone_ado, rtol=0, atol=0)


@pytest.mark.parametrize("fresh", [True, False])
def test_legacy_config_and_runtime_state_use_default_ach_pressure_scale(fresh):
    from ziwago.cortex.moe_cortex import export_runtime_tree, load_runtime_tree

    body = OrganismPhysiology(1)
    body.process_s = 0.5
    body.advance(torch.zeros(1), attention=0.7, dt_ms=25.0)
    legacy_config = asdict(body.physiology_cfg)
    legacy_config.pop("ach_ado_half_saturation")
    assert PhysiologyConfig(**legacy_config).ach_ado_half_saturation == 0.6

    payload = export_runtime_tree(body)
    payload["attributes"]["physiology_cfg"]["attributes"].pop("ach_ado_half_saturation")
    current = OrganismPhysiology(1, cfg=PhysiologyConfig(ach_ado_half_saturation=0.9)) if fresh else None
    restored = load_runtime_tree(current, payload)
    # Old object snapshots inherit the dataclass default, even when restoring
    # into an owner that previously used another theta. No ACh state is reset.
    assert restored.physiology_cfg.ach_ado_half_saturation == 0.6
    assert _snapshot(restored) == _snapshot(body)
    for item in (body, restored):
        item.advance(torch.zeros(1), attention=0.7, dt_ms=100.0)
    assert _snapshot(restored) == _snapshot(body)


@pytest.mark.parametrize("theta", [
    float("nan"), float("inf"), -float("inf"), 0.0, -0.1, True, False,
])
def test_invalid_ach_pressure_half_saturation_is_rejected(theta):
    with pytest.raises((ValueError, TypeError), match="ach_ado_half_saturation"):
        PhysiologyConfig(ach_ado_half_saturation=theta)
    # The owner also validates a config mutated after dataclass construction.
    cfg = PhysiologyConfig()
    cfg.ach_ado_half_saturation = theta
    with pytest.raises((ValueError, TypeError), match="ach_ado_half_saturation"):
        OrganismPhysiology(1, cfg=cfg)


@pytest.mark.parametrize("n_zones", [0, -1, 1.5, True])
def test_invalid_zone_count_is_rejected(n_zones):
    with pytest.raises(ValueError):
        OrganismPhysiology(n_zones)


@pytest.mark.parametrize("kwargs", [
    {"atp_recovery_tau_ms": 0.0},
    {"atp_burn_per_ms": -0.1},
    {"integration_step_ms": float("nan")},
    {"rem_duration_ms": 0.0},
    {"vlpo_wake_threshold": 0.9},
])
def test_invalid_physiology_constants_are_rejected(kwargs):
    with pytest.raises(ValueError):
        PhysiologyConfig(**kwargs)
