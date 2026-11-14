"""CPU-only action-clock/credit contracts; no organism or training run required."""
from __future__ import annotations

import io
import json
import math

import pytest
import torch

from ziwago.action.contextual import ActionGateConfig, ContextualActionGate


def _gate(**config):
    return ContextualActionGate(n_actions=3, context_dim=4, cfg=ActionGateConfig(**config))


def _inputs():
    return torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])


def _commit(gate, context=None, evidence=None, learn=True):
    default_context, default_evidence = _inputs()
    out = gate.step(default_context if context is None else context,
                    default_evidence if evidence is None else evidence,
                    modulators={}, dt_ms=50.0, learn=learn)
    assert out["new_commit"] and out["committed"]
    return out


def _tree(gate):
    from ziwago.cortex.moe_cortex import export_runtime_tree
    return export_runtime_tree(gate)


def _same_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and left.device == right.device
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _same_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _same_tree(a, b)
    else:
        assert type(left) is type(right) and left == right


def _weights(gate):
    return tuple(value.clone() for value in (gate.w_go, gate.w_nogo, gate.w_v))


def test_deterministic_dt_subdivision_preserves_lowpasses_and_accumulator():
    coarse, fine = _gate(decision_threshold=1.0), _gate(decision_threshold=1.0)
    for gate in (coarse, fine):
        gate.w_go[1, 0] = 0.8
        gate.w_nogo[0, 1] = 0.7
    context = torch.tensor([1.0, -0.5, 0.2, 0.0])
    evidence = torch.tensor([0.4, 1.0, 0.0])
    for gate, dt, steps in ((coarse, 4.0, 2), (fine, 0.5, 16)):
        before = _weights(gate)
        for _ in range(steps):
            out = gate.step(context, evidence, modulators={"da": 0.8}, dt_ms=dt, uncertainty=0.2)
            assert not out["committed"]
        _same_tree(before, _weights(gate))
    for attr in ("acc", "gpe", "stn", "gpi"):
        torch.testing.assert_close(getattr(coarse, attr), getattr(fine, attr), rtol=1e-10, atol=1e-12)
    assert coarse.elapsed_ms == fine.elapsed_ms == 8.0
    assert coarse.last_step["threshold"] == fine.last_step["threshold"]


def test_default_deterministic_decisions_take_tens_of_ms():
    times = []
    for dt in (1.0, 0.5):
        gate = _gate()
        context, evidence = _inputs()
        for _ in range(int(100 / dt)):
            out = gate.step(context, evidence, modulators={}, dt_ms=dt)
            if out["committed"]:
                break
        assert out["action"] == 1 and out["decision_id"] == 1
        assert 10.0 <= gate.elapsed_ms <= 100.0
        times.append(gate.elapsed_ms)
    assert abs(times[0] - times[1]) <= 1.0  # commitment is sampled at clock boundaries


def test_zero_evidence_is_masked_even_with_stale_accumulator_and_noise():
    gate = _gate(sigma=0.2)
    context, evidence = _inputs()
    gate.acc[0] = 1000.0
    rng = gate.generator.get_state().clone()
    out = gate.step(context, torch.zeros(3), modulators={}, dt_ms=5.0)
    assert out["action"] == -1 and not out["committed"] and out["bg_release"] == 0.0
    assert torch.count_nonzero(gate.acc) == 0
    assert torch.equal(rng, gate.generator.get_state())
    gate.acc[0], gate.acc[1] = 1000.0, 1.0
    out = gate.step(context, evidence, modulators={})
    assert out["action"] == 1 and gate.acc[0] == gate.acc[2] == 0.0


def test_small_positive_evidence_has_no_admission_floor_and_single_action_is_finite():
    gate = ContextualActionGate(1, 1, cfg=ActionGateConfig(decision_threshold=1e-6))
    out = gate.step(torch.ones(1), torch.tensor([1e-5]), modulators={}, dt_ms=1000.0)
    assert out["action"] == 0 and out["committed"] and out["conflict"] == 0.0
    json.dumps(gate.telemetry(), allow_nan=False)


@pytest.mark.parametrize("reward", [-1.0, 1.0])
@pytest.mark.parametrize("context_sign", [-1.0, 1.0])
def test_settle_uses_frozen_original_context_critic_and_selected_action(reward, context_sign):
    gate = _gate()
    context, evidence = _inputs()
    context *= context_sign
    original = context.double().clone()
    gate.w_v[0] = 0.25
    before_selection = _weights(gate)
    out = _commit(gate, context, evidence)
    _same_tree(before_selection, _weights(gate))  # no learning in step
    predicted = 0.25 * context_sign
    context.zero_()  # caller mutation must not corrupt the ticket
    assert torch.equal(gate.pending["context"], original)
    distractor = torch.tensor([0.0, 0.0, 1.0, 0.0])
    rng = gate.generator.get_state().clone()
    held = gate.step(distractor, torch.tensor([0.0, 0.0, 100.0]),
                     modulators={"ach": 0.0}, dt_ms=3.0, learn=False)
    assert held["action"] == out["action"] == 1
    assert held["decision_id"] == out["decision_id"] and not held["new_commit"]
    assert torch.equal(gate.generator.get_state(), rng)
    gate.w_v.fill_(0.75)  # current critic differs from the prediction on the ticket
    before_go, before_nogo, before_v = _weights(gate)
    result = gate.settle(out["decision_id"], reward)
    rpe = reward - predicted
    assert result["consumed"] and result["learned"] and result["action"] == 1
    assert result["rpe"] == pytest.approx(rpe)
    expected_go, expected_nogo = before_go.clone(), before_nogo.clone()
    expected_go[1] = (expected_go[1] + gate.cfg.learning_rate * rpe * original).clamp(0, gate.cfg.weight_max)
    expected_nogo[1] = (expected_nogo[1] - gate.cfg.learning_rate * rpe * original).clamp(0, gate.cfg.weight_max)
    torch.testing.assert_close(gate.w_go, expected_go)
    torch.testing.assert_close(gate.w_nogo, expected_nogo)
    torch.testing.assert_close(gate.w_v, before_v + gate.cfg.critic_learning_rate * rpe * original)
    assert gate.pending is None and torch.count_nonzero(gate.acc) == 0
    snapshot = _tree(gate)
    duplicate = gate.settle(out["decision_id"], reward)
    assert not duplicate["consumed"] and duplicate["rpe"] == 0.0
    _same_tree(snapshot, _tree(gate))


def test_old_or_wrong_id_cannot_consume_a_new_pending_ticket():
    gate = _gate()
    first = _commit(gate)
    gate.settle(first["decision_id"], 1.0)
    second = _commit(gate)
    assert second["decision_id"] > first["decision_id"]
    snapshot = _tree(gate)
    for decision_id in (first["decision_id"], second["decision_id"] + 1):
        result = gate.settle(decision_id, -1.0)
        assert not result["consumed"] and result["rpe"] == 0.0
        _same_tree(snapshot, _tree(gate))


@pytest.mark.parametrize("mode", ["settle_false", "frozen_settle", "step_false", "frozen_commit", "ach_zero"])
def test_frozen_or_disabled_learning_consumes_without_weight_changes(mode):
    gate = _gate()
    if mode == "frozen_commit":
        gate.learning_enabled = False
    context, evidence = _inputs()
    before = _weights(gate)
    out = gate.step(context, evidence, modulators={"ach": 0.0} if mode == "ach_zero" else {},
                    dt_ms=50.0, learn=mode != "step_false")
    assert out["committed"]
    gate.learning_enabled = mode != "frozen_settle"
    result = gate.settle(out["decision_id"], 1.0, learn=mode != "settle_false")
    assert result["consumed"] and not result["learned"]
    assert gate.counts["updates"] == 0
    _same_tree(before, _weights(gate))
    snapshot = _tree(gate)
    assert gate.settle(out["decision_id"], 1.0)["rpe"] == 0.0
    _same_tree(snapshot, _tree(gate))


@pytest.mark.parametrize("stop,modulators", [(True, {}), (False, {"awake": False}), (True, {"awake": False})])
def test_stop_and_sleep_block_new_decisions_but_retain_pending_and_advance_age(stop, modulators):
    gate = _gate(sigma=0.05)
    context, evidence = _inputs()
    rng = gate.generator.get_state().clone()
    blocked = gate.step(context, evidence * 100.0, modulators=modulators, dt_ms=10.0, stop=stop)
    assert not blocked["committed"] and blocked["bg_release"] == 0.0
    assert torch.equal(rng, gate.generator.get_state())
    assert float(gate.stn.min()) > 0.25  # hyperdirect stop/sleep pathway advances
    gate.acc[1] = 1.0  # guaranteed commitment on the next eligible waking sample
    out = gate.step(context, evidence, modulators={}, learn=True)
    assert out["committed"]
    credit = {key: value.clone() if isinstance(value, torch.Tensor) else value
              for key, value in gate.pending.items()}
    rng = gate.generator.get_state().clone()
    held = gate.step(context.roll(1), torch.zeros(3), modulators=modulators, dt_ms=7.0, stop=stop)
    assert held["committed"] and not held["new_commit"]
    assert held["action"] == out["action"] and held["decision_id"] == out["decision_id"]
    assert held["bg_release"] == 0.0 and gate.pending["age_ms"] == 7.0
    assert gate.elapsed_ms == 18.0
    credit["age_ms"] = 7.0
    _same_tree(credit, gate.pending)
    assert torch.equal(rng, gate.generator.get_state())


def test_expiry_uses_strict_horizon_during_sleep_without_negative_reward_or_recommit():
    gate = _gate(credit_horizon_ms=10.0)
    out = _commit(gate)
    before = _weights(gate)
    context, evidence = _inputs()
    rng = gate.generator.get_state().clone()
    held = gate.step(context, evidence, modulators={"awake": False}, stop=True, dt_ms=10.0)
    assert held["committed"] and not held["expired"] and gate.pending["age_ms"] == 10.0
    expired = gate.step(context, evidence, modulators={"awake": False}, dt_ms=0.5)
    assert expired["expired"] and expired["expired_decision_id"] == out["decision_id"]
    assert not expired["committed"] and expired["action"] == -1
    assert gate.last_outcome["reason"] == "expired" and gate.last_outcome["rpe"] == 0.0
    assert gate.pending is None and torch.count_nonzero(gate.acc) == 0
    assert torch.equal(rng, gate.generator.get_state())
    _same_tree(before, _weights(gate))
    snapshot = _tree(gate)
    assert not gate.settle(out["decision_id"], -1.0)["consumed"]
    _same_tree(snapshot, _tree(gate))
    assert _commit(gate)["decision_id"] > out["decision_id"]


def test_cancel_consumes_without_learning_and_is_idempotent():
    gate = _gate()
    out = _commit(gate)
    before = _weights(gate)
    cancelled = gate.cancel()
    assert cancelled["consumed"] and cancelled["action"] == out["action"]
    assert cancelled["decision_id"] == out["decision_id"] and cancelled["rpe"] == 0.0
    assert not cancelled["learned"] and cancelled["reason"] == "cancelled"
    assert gate.pending is None and torch.count_nonzero(gate.acc) == 0
    _same_tree(before, _weights(gate))
    snapshot = _tree(gate)
    assert not gate.cancel()["consumed"]
    assert not gate.settle(out["decision_id"], 1.0)["consumed"]
    _same_tree(snapshot, _tree(gate))


def test_conflict_uncertainty_and_neuromodulators_reach_pathways_and_threshold():
    context, evidence = _inputs()
    baseline, conflict, uncertain, da_high = (_gate() for _ in range(4))
    a = baseline.step(context, evidence, modulators={})
    b = conflict.step(context, torch.ones(3), modulators={})
    u = uncertain.step(context, evidence, modulators={}, uncertainty=1.0)
    d = da_high.step(context, evidence, modulators={"da": 2.0, "ado": 0.8, "ht": 1.0, "ox": 0.0})
    assert a["conflict"] == 0.0 and b["conflict"] == pytest.approx(1.0)
    assert b["threshold"] > a["threshold"] and u["threshold"] > a["threshold"]
    assert torch.all(conflict.stn > baseline.stn)
    assert torch.all(da_high.gpi < baseline.gpi) and d["threshold"] > a["threshold"]


def test_private_rng_does_not_touch_or_depend_on_global_rng():
    global_before = torch.random.get_rng_state().clone()
    try:
        first, second = _gate(sigma=0.3, decision_threshold=10.0), _gate(sigma=0.3, decision_threshold=10.0)
        assert torch.equal(global_before, torch.random.get_rng_state())
        context, evidence = _inputs()
        for _ in range(4):
            before = torch.random.get_rng_state().clone()
            a = first.step(context, evidence, modulators={}, dt_ms=2.0)
            assert torch.equal(before, torch.random.get_rng_state())
            torch.rand(11)  # unrelated global consumer between equal private streams
            before = torch.random.get_rng_state().clone()
            b = second.step(context, evidence, modulators={}, dt_ms=2.0)
            assert torch.equal(before, torch.random.get_rng_state())
            assert a == b and torch.equal(first.acc, second.acc)
        different = ContextualActionGate(3, 4, seed=21, cfg=first.cfg)
        different.step(context, evidence, modulators={}, dt_ms=8.0)
        assert not torch.equal(first.generator.get_state(), different.generator.get_state())
    finally:
        torch.random.set_rng_state(global_before)


def test_noise_uses_sqrt_seconds_and_only_eligible_channels():
    gate = _gate(drift_gain=0.0, leak_rate=0.0, sigma=0.2, decision_threshold=10.0)
    expected_rng = torch.Generator().set_state(gate.generator.get_state())
    expected = torch.randn(1, generator=expected_rng, dtype=torch.float64) * 0.2 * math.sqrt(0.004)
    context, evidence = _inputs()
    gate.step(context, evidence, modulators={}, dt_ms=4.0)
    torch.testing.assert_close(gate.acc[1:2], expected, rtol=1e-12, atol=1e-12)
    assert gate.acc[0] == gate.acc[2] == 0.0
    assert torch.equal(expected_rng.get_state(), gate.generator.get_state())


@pytest.mark.parametrize("override", [
    {"context": torch.ones(5)}, {"context": torch.ones(1, 4)},
    {"context": torch.tensor([float("nan"), 0.0, 0.0, 0.0])},
    {"context": torch.ones(4, dtype=torch.complex64)},
    {"evidence": torch.ones(4)}, {"evidence": [0.0, 1.0, 0.0]},
    {"evidence": torch.tensor([0.0, -0.1, 0.0])},
    {"evidence": torch.tensor([0.0, float("inf"), 0.0])},
    {"evidence": torch.tensor([0.0, float("nan"), 0.0])},
    {"dt_ms": 0.0}, {"dt_ms": -1.0}, {"dt_ms": 1001.0}, {"dt_ms": float("nan")},
    {"dt_ms": True}, {"uncertainty": -0.01}, {"uncertainty": 1.01},
    {"uncertainty": float("inf")}, {"stop": 1}, {"learn": "yes"},
    {"modulators": None}, {"modulators": {"awake": 0}},
    {"modulators": {"da": float("nan")}}, {"modulators": {"da": 2.1}},
    {"modulators": {"ado": -0.1}}, {"modulators": {"ox": 1.1}},
    {"modulators": {"ach": "0.5"}}, {"modulators": {"unused_axis": float("inf")}},
])
def test_invalid_step_controls_do_not_mutate_pending_rng_clock_or_weights(override):
    gate = _gate(credit_horizon_ms=1.0)
    _commit(gate)
    context, evidence = _inputs()
    kwargs = dict(context=context, evidence=evidence, modulators={"awake": False},
                  dt_ms=2.0, uncertainty=0.0, stop=True, learn=True)
    kwargs.update(override)
    snapshot = _tree(gate)
    with pytest.raises((TypeError, ValueError)):
        gate.step(**kwargs)
    _same_tree(snapshot, _tree(gate))  # invalid call must not expire its ticket either


@pytest.mark.parametrize("decision_id,reward,learn", [
    (True, 1.0, True), (1.0, 1.0, True), (0, 1.0, True),
    (1, float("nan"), True), (1, float("inf"), True), (1, 1.1, True),
    (1, -1.1, True), (1, True, True), (1, 1.0, 1), (999, float("nan"), True),
])
def test_invalid_settlement_is_atomic_even_for_nonmatching_id(decision_id, reward, learn):
    gate = _gate()
    _commit(gate)
    snapshot = _tree(gate)
    with pytest.raises((TypeError, ValueError)):
        gate.settle(decision_id, reward, learn=learn)
    _same_tree(snapshot, _tree(gate))


@pytest.mark.parametrize("config", [
    {"decision_threshold": 0.0}, {"drift_gain": -1.0}, {"sigma": -0.1},
    {"leak_rate": float("nan")}, {"tau_stn_ms": 0.0}, {"context_gain": float("inf")},
    {"learning_rate": 1.1}, {"credit_horizon_ms": 0.0}, {"max_dt_ms": 1001.0},
    {"initial_weight": 0.2, "weight_max": 0.1}, {"sigma": True},
])
def test_config_has_explicit_finite_parameter_bounds(config):
    with pytest.raises((TypeError, ValueError)):
        ActionGateConfig(**config)


@pytest.mark.parametrize("kwargs", [
    {"n_actions": 0}, {"n_actions": 257}, {"n_actions": True},
    {"context_dim": 0}, {"context_dim": 4097}, {"context_dim": 4.0},
    {"seed": -1}, {"seed": True}, {"cfg": {}},
])
def test_constructor_validates_size_seed_and_config(kwargs):
    with pytest.raises((TypeError, ValueError)):
        ContextualActionGate(**kwargs)


def test_config_is_copied_and_revalidated_before_mutation():
    cfg = ActionGateConfig()
    gate = ContextualActionGate(3, 4, cfg=cfg)
    cfg.sigma = 1.0
    assert gate.cfg.sigma == 0.0
    out = _commit(gate)
    gate.cfg.leak_rate = -1.0
    snapshot = _tree(gate)
    with pytest.raises(ValueError):
        gate.step(*_inputs(), modulators={})
    with pytest.raises(ValueError):
        gate.settle(out["decision_id"], 1.0)
    _same_tree(snapshot, _tree(gate))


def test_large_finite_inputs_remain_bounded_and_weights_stay_nonnegative():
    gate = _gate(learning_rate=1.0, critic_learning_rate=1.0, weight_max=0.2,
                 critic_weight_max=0.2)
    huge = torch.finfo(torch.float64).max
    context = torch.tensor([huge, -huge, huge, -huge], dtype=torch.float64)
    evidence = torch.tensor([0.0, huge, huge / 2.0], dtype=torch.float64)
    out = _commit(gate, context, evidence)
    result = gate.settle(out["decision_id"], -1.0)
    assert result["learned"]
    assert torch.isfinite(gate.w_go).all() and torch.isfinite(gate.w_nogo).all()
    assert torch.all((gate.w_go >= 0.0) & (gate.w_go <= 0.2))
    assert torch.all((gate.w_nogo >= 0.0) & (gate.w_nogo <= 0.2))
    assert torch.all(gate.w_v.abs() <= 0.2)
    json.dumps(gate.telemetry(), allow_nan=False)


def test_runtime_tree_roundtrip_preserves_learned_pending_and_rng_state():
    from ziwago.cortex.moe_cortex import export_runtime_tree, load_runtime_tree

    gate = _gate(sigma=0.001)
    first = _commit(gate)
    gate.settle(first["decision_id"], 0.8)
    second = _commit(gate)
    context, evidence = _inputs()
    gate.step(context.roll(1), evidence.roll(1), modulators={}, dt_ms=3.0)
    snapshot = export_runtime_tree(gate)
    buffer = io.BytesIO()
    torch.save(snapshot, buffer)
    buffer.seek(0)
    payload = torch.load(buffer, map_location="cpu", weights_only=True)
    # Both reconstruction paths are used by the existing runtime-tree loader.
    restored = load_runtime_tree(ContextualActionGate(1, 1, seed=99), payload)
    reconstructed = load_runtime_tree(None, payload)
    _same_tree(snapshot, export_runtime_tree(restored))
    _same_tree(snapshot, export_runtime_tree(reconstructed))
    assert restored.pending["context"].data_ptr() != gate.pending["context"].data_ptr()
    assert restored.w_go.data_ptr() != gate.w_go.data_ptr()
    assert restored.pending["age_ms"] == 3.0
    assert gate.settle(second["decision_id"], -0.5) == restored.settle(second["decision_id"], -0.5)
    for _ in range(3):
        a = gate.step(context, evidence, modulators={}, dt_ms=1.5, learn=True)
        b = restored.step(context, evidence, modulators={}, dt_ms=1.5, learn=True)
        assert a == b
    _same_tree(export_runtime_tree(gate), export_runtime_tree(restored))


def test_telemetry_is_json_safe_independent_and_reports_budgets_and_outcomes():
    gate = _gate()
    json.dumps(gate.telemetry(), allow_nan=False)
    context, evidence = _inputs()
    out = gate.step(context, evidence, modulators={"awake": True, "ne": 0.3,
                    "f_mod": 1.2, "da_phasic": -0.2}, dt_ms=50.0, learn=True)
    telemetry = gate.telemetry()
    json.dumps(telemetry, allow_nan=False)
    assert telemetry["pending"]["decision_id"] == out["decision_id"]
    assert telemetry["budgets"]["max_pending"] == 1
    assert telemetry["budgets"]["learned_parameters"] == 2 * 3 * 4 + 4
    assert telemetry["budgets"]["evidence_admission"] == "strictly_positive"
    assert telemetry["last_step"]["ignored_modulators"] == ["da_phasic", "f_mod", "ne"]
    telemetry["pending"]["age_ms"] = -999.0
    telemetry["config"]["sigma"] = 99.0
    telemetry["last_step"]["ignored_modulators"].clear()
    assert gate.pending["age_ms"] == 0.0 and gate.cfg.sigma == 0.0
    assert gate.last_step["ignored_modulators"]
    gate.settle(out["decision_id"], 1.0)
    telemetry = gate.telemetry()
    assert telemetry["pending"] is None and telemetry["outcome"]["reason"] == "settled"
    assert telemetry["counts"]["updates"] == 1
    json.dumps(telemetry, allow_nan=False)
