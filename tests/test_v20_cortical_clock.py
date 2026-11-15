"""Opt-in v20 cortical clock contracts; small real two-column circuits."""
from copy import deepcopy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ziwago.config import USSynapseCfg
from ziwago.core.synapse import GenesisSynapse
from ziwago.cortex import moe_cortex as runtime_paths
from ziwago.cortex.moe_cortex import SparselyGatedCorticalMoE


def _moe(single_active=True):
    cortex = SparselyGatedCorticalMoE(
        n_columns=2, neurons_per_column=32, n_in=32, seed=20,
    )
    # Keep the legacy constructor's >=2 policy unchanged. Its existing mutable
    # routing setting permits exercising dormancy on a real 2x32 circuit.
    if single_active:
        cortex.top_k_active = 1
    for col in cortex.columns:
        col.pyr.cfg.stochastic = False
    return cortex


def _context(tick, **overrides):
    ctx = dict(
        tick=tick, zone_atp=torch.tensor([0.91, 0.72]),
        zone_ado=torch.tensor([0.12, 0.35]), ach=0.8, ne=0.67,
        da=0.9, da_phasic=-0.4, p_release=0.63, nmda_gain=1.3,
        histamine=0.4, bdnf=0.35, is_sleep_tick=False,
        learning_enabled=True,
    )
    ctx.update(overrides)
    return ctx


def _bias(cid):
    return torch.tensor([40.0, -40.0]) if cid == 0 else torch.tensor([-40.0, 40.0])


def _syn(n_pre=2, n_post=1):
    syn = GenesisSynapse(n_pre, n_post, USSynapseCfg(density=1.0), seed=20)
    syn.w_slow.zero_()
    syn.w_fast.zero_()
    syn.externally_clocked = True
    return syn


def _assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            _assert_tree_equal(left, right)
    else:
        assert a == b


def test_het_syn_matrices_are_allocated_only_when_the_feature_is_used():
    syn = _syn(n_pre=4, n_post=3)
    assert not hasattr(syn, "tau_syn_het")
    assert not hasattr(syn, "het_trace")

    syn.advance_time(dt=1.0, s_pre=torch.ones(1, 4), s_post=torch.zeros(1, 3))
    assert not hasattr(syn, "tau_syn_het")
    assert not hasattr(syn, "het_trace")

    syn.enable_het_syn = True
    trace = syn.step_het_syn_traces(torch.ones(1, 4), adapt_tau=False)
    assert syn.tau_syn_het.shape == (3, 4)
    assert syn.het_trace.shape == (3, 4)
    assert torch.isfinite(syn.tau_syn_het).all()
    assert torch.isfinite(trace).all()


def test_slow_weight_initialization_matches_the_original_seeded_values():
    n_pre, n_post, seed, init_scale = 7, 5, 23, 0.4
    cfg = USSynapseCfg(density=0.6)
    generator = torch.Generator().manual_seed(seed)
    expected_mask = torch.rand(n_post, n_pre, generator=generator) < cfg.density
    expected_weights = (
        torch.rand(n_post, n_pre, generator=generator) * init_scale - init_scale / 2.0
    ) * (1.0 if cfg.sign == "exc" else -1.0) * expected_mask

    syn = GenesisSynapse(n_pre, n_post, cfg, seed=seed, init_scale=init_scale)
    assert torch.equal(syn.mask, expected_mask)
    assert torch.equal(syn.w_slow, expected_weights)


def test_branch_kernels_are_shared_for_matching_dimensions():
    first = _syn(n_pre=4, n_post=3)
    second = GenesisSynapse(4, 2, USSynapseCfg(density=1.0), name="other", seed=21)
    idx = torch.arange(4, dtype=torch.float32)
    dist = (idx[:, None] - idx[None, :]).abs()
    expected = torch.exp(-0.5 * (dist / 1.5) ** 2)
    expected.fill_diagonal_(0.0)
    expected = expected / expected.sum(dim=1, keepdim=True).clamp(min=1e-6)

    assert first.branch_kernel is second.branch_kernel
    torch.testing.assert_close(first.branch_kernel, expected, rtol=0.0, atol=0.0)


def test_pinned_block_mask_is_allocated_only_when_lanes_are_pinned():
    syn = _syn()
    assert not hasattr(syn, "pinned_block_mask")

    syn.enable_scaffold = True
    syn.learn(
        error=torch.ones(1, 1),
        s_pre=torch.ones(1, 2),
        s_post=torch.ones(1, 1),
        f_mod=1.0,
    )
    assert not hasattr(syn, "pinned_block_mask")

    syn.pin_top_block_sparse_lanes()
    assert syn.pinned_block_mask.shape == syn.mask.shape


def test_active_sparse_driver_matches_dense_projection_without_cached_indices():
    syn = GenesisSynapse(4, 3, USSynapseCfg(density=1.0), seed=12)
    syn.enable_block_sparse = True
    syn.w_slow.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10.0)
    syn.w_fast.fill_(0.1)
    syn.u_syn.fill_(0.8)
    x = torch.tensor([[1.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0]])
    effective_weights = (syn.w_slow + syn.w_fast) * syn.mask * syn.u_syn

    actual = syn.drive(x)
    expected = x @ effective_weights.t()

    torch.testing.assert_close(actual, expected)
    for field in ("csr_row_ptr", "csr_col_idx", "csc_col_ptr", "csc_row_idx", "_dirty_blocks"):
        assert not hasattr(syn, field)


def test_dynamic_state_export_round_trips_without_sparse_index_cache():
    cortex = _moe()
    state = cortex.export_dynamic_state_dict()

    cortex.load_dynamic_state_dict(state)

    for column in cortex.columns:
        for syn in (column.w_ff, column.w_fb, column.w_rec):
            assert not hasattr(syn, "_dirty_blocks")


def test_epoch_kl_snapshot_is_explicit_and_still_measures_weight_drift():
    syn = _syn()
    assert not hasattr(syn, "w_epoch0_snapshot")
    with pytest.raises(RuntimeError, match="Capture the epoch-0"):
        syn.kl_divergence_bits()

    syn.capture_epoch0_snapshot()
    syn.w_slow[0, 0] = 1.0
    assert syn.kl_divergence_bits() > 0.0


def test_routing_telemetry_restores_router_state_even_after_second_sweep():
    cortex = _moe()
    cortex.lag_plv_router.step_phases_and_compute_causal_matrix(torch.tensor([0.2, 0.7]))
    before = cortex.export_state_dict()

    report = cortex.routing_telemetry()

    assert report["reference_sweep_invariance_check"]["is_reference_invariant"]
    _assert_tree_equal(before, cortex.export_state_dict())


@pytest.mark.parametrize("fault", ["tick", "dt"])
def test_context_clock_validation_precedes_all_mutation(fault):
    cortex = _moe()
    ctx = _context(1)
    if fault == "tick":
        ctx["tick"] = 0
    else:
        cortex.columns[1].pyr.cfg.dt = 0.5
    before = cortex.export_state_dict()
    with pytest.raises(AssertionError):
        cortex.step(torch.ones(32), organism_context=ctx)
    _assert_tree_equal(before, cortex.export_state_dict())


def test_global_local_ticks_and_all_synapses_advance_through_dormancy():
    cortex = _moe()
    cortex.global_clock_tick = 40
    cortex.column_last_active_tick = [40, 40]
    expected_pre = [torch.zeros(32), torch.zeros(32)]
    expected_bcm = [0.0, 0.0]
    for col in cortex.columns:
        col.tick_ladder.tick = 3  # stale local clock must align, not increment to 4
        col.pyr.theta.fill_(-100.0)  # ensure a real measured post-activity signal
        for syn in (col.w_ff, col.w_rec, col.w_fb):
            syn.tag.fill_(1.0)
            syn.eligibility.fill_(0.5)
    x = torch.full((32,), 0.25)
    schedule = [0, 1, 1, 1, 0]
    for elapsed, active in enumerate(schedule, 1):
        tick = 40 + elapsed
        ctx = _context(tick, is_sleep_tick=(elapsed == 5))
        with patch.object(cortex.columns[active].minicolumn_hierarchy, "update_tiers",
                          wraps=cortex.columns[active].minicolumn_hierarchy.update_tiers) as tiers:
            out = cortex.step(x, learn=False, organism_context=ctx, routing_bias=_bias(active))
        assert out.active_expert_ids == [active]
        assert cortex.global_clock_tick == tick
        assert tiers.call_args.kwargs["ladder_step"].tick == tick
        assert tiers.call_args.kwargs["ladder_step"].run_class_4_sleep == ctx["is_sleep_tick"]
        for cid, col in enumerate(cortex.columns):
            assert col.tick_ladder.tick == tick
            assert col._mod_ctx["clock_tick"] == tick
            assert col._mod_ctx["is_sleep_tick"] == ctx["is_sleep_tick"]
            assert torch.equal(col.pyr.energy, torch.full_like(col.pyr.energy, float(ctx["zone_atp"][cid])))
            assert torch.equal(col.pyr.ado_i, torch.full_like(col.pyr.ado_i, float(ctx["zone_ado"][cid])))
            expected_pre[cid] *= 1.0 - 1.0 / col.w_ff.cfg.tau_x
            if cid == active:
                expected_pre[cid] += x
            torch.testing.assert_close(col.w_ff.z_pre, expected_pre[cid])
            measured = float(col._last_s.mean()) ** 2 if cid == active else 0.0
            alpha = 1.0 / col.pyr.cfg.tau_ca
            expected_bcm[cid] = (1.0 - alpha) * expected_bcm[cid] + alpha * measured
            assert col._bcm_post_activity == pytest.approx(expected_bcm[cid])
            for syn in (col.w_ff, col.w_rec, col.w_fb):
                assert syn.externally_clocked
                assert syn.theta_m == pytest.approx(syn.cfg.theta_m0 + expected_bcm[cid])
                torch.testing.assert_close(syn.tag, torch.full_like(syn.tag, 0.9995 ** elapsed))
                torch.testing.assert_close(syn.eligibility, torch.full_like(syn.eligibility, 0.5 * 0.999 ** elapsed))
                torch.testing.assert_close(syn.spine_age, elapsed * syn.mask.float())
        if elapsed == 5:
            assert out.rehydrated_columns_count == 1
    assert all(value > 0.0 for value in expected_bcm)
    assert cortex.column_last_active_tick == [45, 44]


def test_real_context_reaches_neurons_and_owns_recovery_and_spike_energy():
    cortex = _moe(single_active=False)
    for col in cortex.columns:
        col.pyr.enable_elif_metabolic_gate = True
        col.pyr.theta.fill_(-100.0)
    ctx = _context(1, zone_atp=torch.tensor([0.2, 0.8]),
                   zone_ado=torch.tensor([0.35, 0.7]), p_release=0.41)
    with patch.object(cortex.columns[0].pyr, "step", wraps=cortex.columns[0].pyr.step) as neuron_step, \
            patch.object(cortex.bio_rwkv, "step", wraps=cortex.bio_rwkv.step) as rwkv_step:
        cortex.step(torch.ones(32), learn=False, organism_context=ctx)
    assert rwkv_step.call_args.kwargs["ne_phasic"] == ctx["ne"]
    for key in ("ach", "ne", "nmda_gain", "histamine"):
        assert neuron_step.call_args.kwargs[key] == ctx[key]
    assert neuron_step.call_args.kwargs["p_rel"] == ctx["p_release"]
    assert cortex.columns[0].pyr.last_down_phase_fraction == 1.0
    assert cortex.columns[1]._last_s.sum() > 0  # real spikes would burn locally owned ATP
    for cid, col in enumerate(cortex.columns):
        assert col.pyr.external_energy
        assert col._mod_ctx["da_phasic"] == ctx["da_phasic"]
        assert col._mod_ctx["lambda_atp"] == pytest.approx(1.0 - float(ctx["zone_atp"][cid]))
        assert col._mod_ctx["a1_gain"] == pytest.approx(1.0 / (1.0 + (float(ctx["zone_ado"][cid]) / 0.35) ** 2))
        torch.testing.assert_close(col.pyr.energy, torch.full_like(col.pyr.energy, float(ctx["zone_atp"][cid])), rtol=0, atol=0)
        torch.testing.assert_close(col.pyr.ado_i, torch.full_like(col.pyr.ado_i, float(ctx["zone_ado"][cid])), rtol=0, atol=0)


def test_a1_quenches_complete_ff_and_recurrent_conductance_once():
    plain = _moe()
    plain.columns[0]._last_s = torch.full((1, 32), 0.3)
    plain.columns[0].w_rec.w_slow.fill_(0.1)
    quenched = deepcopy(plain)
    captured = []
    for cortex, ado in ((plain, 0.0), (quenched, 0.35)):
        ctx = _context(1, zone_ado=torch.full((2,), ado), p_release=0.37)
        with patch.object(cortex.columns[0].pyr, "step", wraps=cortex.columns[0].pyr.step) as call:
            cortex.step(torch.ones(32), learn=False, organism_context=ctx, routing_bias=_bias(0))
        captured.append(call.call_args)
    assert captured[0].args[0].abs().sum() > 0
    gain = quenched.columns[0]._mod_ctx["a1_gain"]
    torch.testing.assert_close(captured[1].args[0], captured[0].args[0] * gain)
    torch.testing.assert_close(captured[1].args[1], captured[0].args[1])  # FB is not A1-quenched
    assert captured[0].kwargs["p_rel"] == captured[1].kwargs["p_rel"] == 0.37


def test_broadcast_enters_real_apical_conductance_before_forward():
    plain = _moe()
    plain.columns[0].w_fb.w_slow.fill_(0.2)
    broadcasted = deepcopy(plain)
    ctx = _context(1)
    plain.step(torch.zeros(32), learn=False, organism_context=ctx, routing_bias=_bias(0))
    broadcast = torch.full((32,), 0.4)
    with patch.object(broadcasted.columns[0].pyr, "step", wraps=broadcasted.columns[0].pyr.step) as call:
        broadcasted.step(torch.zeros(32), learn=False, organism_context=ctx,
                         routing_bias=_bias(0), broadcast=broadcast)
    col = broadcasted.columns[0]
    torch.testing.assert_close(col._last_x_fb, broadcast.unsqueeze(0))
    assert call.call_args.args[1].sum() > 0
    assert col.pyr.last_i_fb.sum() > plain.columns[0].pyr.last_i_fb.sum()
    assert col.pyr.v_a.mean() > plain.columns[0].pyr.v_a.mean()


def test_routing_bias_changes_raw_logit_selection_and_softmax():
    cortex = _moe(single_active=False)
    cortex.w_router.zero_()
    ids, weights = cortex.route_top_k_columns(torch.zeros(32), routing_bias=torch.tensor([-3.0, 3.0]))
    assert ids == [1, 0]
    torch.testing.assert_close(weights, torch.softmax(torch.tensor([-3.0, 3.0]), dim=0))
    single = _moe()
    single.w_router.zero_()
    ids, weights = single.route_top_k_columns(torch.zeros(32), routing_bias=torch.tensor([-3.0, 3.0]))
    assert ids == [1]
    assert weights.tolist() == [0.0, 1.0]


def test_external_rehydration_uses_only_missed_ticks_and_does_not_recover_atp():
    external = _moe()
    col = external.columns[0]
    col.pyr.v_s.fill_(-40.0)
    col.pyr.v_d.fill_(-40.0)
    col.pyr.v_a.fill_(-40.0)
    col.pyr.energy.fill_(0.37)
    col.pyr.ado_i.fill_(0.6)
    col.dendrite.branch_v.fill_(-20.0)
    col.dendrite.plateau.fill_(5.0)
    legacy = deepcopy(external)
    col.pyr.external_energy = True
    for cortex, missed in ((external, 3.0), (legacy, 4.0)):
        cortex._analytically_rehydrate_column(0, 4)
        c = cortex.columns[0]
        cfg = c.pyr.cfg
        torch.testing.assert_close(c.pyr.v_s, torch.full_like(c.pyr.v_s, cfg.e_l + 30.0 * math.exp(-missed / (cfg.c_m / cfg.g_l))))
        torch.testing.assert_close(c.pyr.v_d, torch.full_like(c.pyr.v_d, cfg.e_l + 30.0 * math.exp(-missed / cfg.tau_d)))
        torch.testing.assert_close(c.pyr.v_a, c.pyr.v_d)
        torch.testing.assert_close(c.dendrite.plateau, torch.full_like(c.dendrite.plateau, 5.0 - missed))
    torch.testing.assert_close(col.pyr.energy, torch.full_like(col.pyr.energy, 0.37), rtol=0, atol=0)
    torch.testing.assert_close(col.pyr.ado_i, torch.full_like(col.pyr.ado_i, 0.6), rtol=0, atol=0)
    torch.testing.assert_close(legacy.columns[0].pyr.energy, torch.full_like(col.pyr.energy, 0.37 + 0.008 * 4))


@pytest.mark.parametrize("f_mod", [0.0, 1.0])
def test_teaching_does_not_repeat_elapsed_decay_or_forward_traces(f_mod):
    syn = _syn()
    syn.enable_het_syn = True
    syn.tag.fill_(1.0)
    syn.eligibility.fill_(0.8)
    syn.r_err.fill_(0.4)
    syn.spine_age.fill_(5.0)
    syn.spine_health.fill_(0.4)
    syn.z_pre.fill_(0.6)
    syn.z_post.fill_(0.4)
    syn.u_syn.fill_(0.5)
    syn.w_fast.fill_(0.3)
    syn.w_slow.fill_(0.2)
    pre, post = torch.ones(1, 2), torch.zeros(1, 1)
    syn.advance_time(dt=1.0, s_pre=pre, s_post=post)
    expected = {name: getattr(syn, name).clone() for name in (
        "tag", "eligibility", "r_err", "spine_age", "z_pre", "z_post", "u_syn",
        "w_fast", "w_slow", "het_trace", "pre_silence_ticks", "tau_syn_het",
    )}
    for _ in range(2):
        syn.learn(torch.zeros_like(post), pre, post, f_mod=f_mod, da_phasic=0.0)
        for name, value in expected.items():
            torch.testing.assert_close(getattr(syn, name), value, rtol=0, atol=0)
    torch.testing.assert_close(syn.tag, torch.full_like(syn.tag, 0.9995))
    torch.testing.assert_close(syn.eligibility, torch.full_like(syn.eligibility, 0.8 * 0.999))
    torch.testing.assert_close(syn.spine_age, torch.full_like(syn.spine_age, 6.0))
    torch.testing.assert_close(syn.z_pre, torch.full_like(syn.z_pre, 0.6 * 0.95 + 1.0))
    torch.testing.assert_close(syn.z_post, torch.full_like(syn.z_post, 0.4 * 0.95))
    expected_health = 0.4 * 0.998 + (2 * 0.025 * 0.3 * 0.9995 if f_mod else 0.0)
    torch.testing.assert_close(syn.spine_health, torch.full_like(syn.spine_health, expected_health))


def test_external_learning_keeps_activity_driven_resource_spend():
    syn = _syn()
    syn.u_syn.fill_(0.5)
    pre, post = torch.ones(1, 2), torch.ones(1, 1)
    syn.advance_time(dt=1.0, s_pre=pre, s_post=post)
    syn.learn(torch.zeros_like(post), pre, post, 1.0, da_phasic=0.0)
    torch.testing.assert_close(syn.u_syn, torch.full_like(syn.u_syn, 0.5 + 0.5 / 500.0 - 0.15))
    torch.testing.assert_close(syn.z_pre, torch.ones_like(syn.z_pre))
    torch.testing.assert_close(syn.z_post, torch.ones_like(syn.z_post))


def test_nonlearning_clock_advances_traces_without_decaying_learned_weights():
    syn = _syn()
    syn.w_slow.fill_(0.4)
    syn.w_fast.fill_(0.2)
    slow_before = syn.w_slow.clone()
    fast_before = syn.w_fast.clone()

    syn.advance_time(
        s_pre=torch.ones(1, 2),
        s_post=torch.ones(1, 1),
        decay_weights=False,
    )

    assert torch.equal(syn.w_slow, slow_before)
    assert torch.equal(syn.w_fast, fast_before)
    assert torch.equal(syn.z_pre, torch.ones_like(syn.z_pre))
    assert torch.equal(syn.z_post, torch.ones_like(syn.z_post))


def test_signed_external_da_reaches_delayed_eligibility_even_in_silence():
    changes = {}
    for da in (-1.0, 0.0, 1.0, None):
        syn = _syn()
        syn.eligibility.fill_(1.0)
        syn.advance_time()
        syn.learn(torch.zeros(1, 1), torch.zeros(1, 2), torch.zeros(1, 1),
                  f_mod=0.7, da_phasic=da)
        changes[da] = syn.w.clone()
    assert (changes[-1.0] < 0).all()
    assert (changes[1.0] > 0).all()
    assert torch.equal(changes[0.0], torch.zeros_like(changes[0.0]))
    assert torch.equal(changes[None], changes[0.0])  # tonic f_mod is not phasic DA
    immediate = _syn()
    immediate.advance_time(s_pre=torch.ones(1, 2), s_post=torch.ones(1, 1))
    immediate.learn(torch.ones(1, 1), torch.ones(1, 2), torch.ones(1, 1),
                    f_mod=0.7, da_phasic=0.0)
    assert (immediate.w > 0).all()  # immediate supervised error survives zero DA


def test_active_forward_and_apical_teaching_share_one_elapsed_tick():
    cortex = _moe()
    col = cortex.columns[0]
    ctx = _context(1, zone_atp=torch.ones(2))
    with patch.object(col.w_ff, "advance_time", wraps=col.w_ff.advance_time) as advance, \
            patch.object(col.w_ff, "learn", wraps=col.w_ff.learn) as teaching:
        cortex.step(torch.ones(32), organism_context=ctx, routing_bias=_bias(0))
        traces = col.w_ff.z_pre.clone()
        col.inject_apical_error(torch.ones(1, 32))
    assert advance.call_count == 1
    assert teaching.call_count == 2
    assert teaching.call_args.kwargs["ach"] == ctx["ach"]
    assert teaching.call_args.kwargs["da_phasic"] == ctx["da_phasic"]
    assert teaching.call_args.kwargs["lambda_atp"] == 0.0
    torch.testing.assert_close(col.w_ff.z_pre, traces, rtol=0, atol=0)
    torch.testing.assert_close(col.w_ff.spine_age, col.w_ff.mask.float())


def test_learning_disabled_blocks_teaching_but_not_clock():
    cortex = _moe()
    col = cortex.columns[0]
    ctx = _context(1, learning_enabled=False)
    with patch.object(col.w_ff, "learn", wraps=col.w_ff.learn) as teaching:
        cortex.step(torch.ones(32), learn=True, organism_context=ctx, routing_bias=_bias(0))
        before = col.pyr.v_a.clone()
        result = col.inject_apical_error(torch.ones(1, 32))
    assert teaching.call_count == 0
    assert torch.equal(col.pyr.v_a, before)
    assert torch.count_nonzero(result) == 0
    torch.testing.assert_close(col.w_ff.spine_age, col.w_ff.mask.float())
    torch.testing.assert_close(col.w_ff.z_pre, torch.ones_like(col.w_ff.z_pre))


def test_tagged_consolidation_is_selective_conservative_bounded_and_rng_free():
    syn = _syn(3, 3)
    syn.mask[1, 0] = False
    syn.w_slow.copy_(torch.tensor([[1.99, 0.2, 0.3], [0.4, -1.99, 0.5], [2.0, 0.1, -0.1]]))
    syn.w_fast.copy_(torch.tensor([[0.6, 0.5, 0.4], [0.3, -0.6, 0.2], [0.6, 0.6, -0.6]]))
    syn.tag.copy_(torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]))
    cue = torch.tensor([1.0, -1.0, 0.0])
    hit = syn.mask & syn.tag.ne(0) & cue.ne(0).unsqueeze(0)
    slow, fast, tags = syn.w_slow.clone(), syn.w_fast.clone(), syn.tag.clone()
    resource = syn.u_syn.clone()
    rng = torch.random.get_rng_state().clone()
    with patch.object(syn, "drive", side_effect=AssertionError("replay must not forward")):
        count = syn.consolidate_tagged(cue)
    assert count == int(hit.sum()) == 4
    assert torch.equal(torch.random.get_rng_state(), rng)
    torch.testing.assert_close(syn.w_slow + syn.w_fast, slow + fast)
    assert torch.equal(syn.w_slow[~hit], slow[~hit])
    assert torch.equal(syn.w_fast[~hit], fast[~hit])
    assert torch.equal(syn.tag[~hit], tags[~hit])
    assert torch.count_nonzero(syn.tag[hit]) == 0
    assert torch.equal(syn.u_syn, resource)
    assert syn.w_slow.max() <= syn.cfg.w_max
    assert syn.w_slow.min() >= syn.cfg.w_min
    assert syn.w_slow[0, 0] == syn.cfg.w_max
    assert syn.w_slow[1, 1] == syn.cfg.w_min
    assert syn.w_fast[2, 0] == fast[2, 0]  # saturated slow component consumes no fast weight
    assert syn.consolidate_tagged(cue) == 0


@pytest.mark.parametrize("factor", [0.0, 0.1, 1.0, 5.0])
def test_tagged_consolidation_clips_fraction_without_reversing_fast_weight(factor):
    syn = _syn()
    syn.w_fast.fill_(0.5)
    syn.tag.fill_(1.0)
    syn.consolidate_tagged(torch.ones(2), factor=factor)
    torch.testing.assert_close(syn.w_slow, torch.full_like(syn.w_slow, 0.5 * min(factor, 1.0)))
    torch.testing.assert_close(syn.w, torch.full_like(syn.w, 0.5))
    assert (syn.w_fast >= 0).all()
    if factor == 0.0:
        assert torch.equal(syn.tag, torch.ones_like(syn.tag))


@pytest.mark.parametrize("da", [None, -0.8, 0.8])
def test_legacy_synapse_formulas_are_unchanged(da):
    syn = GenesisSynapse(1, 1, USSynapseCfg(density=1.0), seed=20)
    assert not syn.externally_clocked
    syn.w_slow.zero_()
    syn.learn(torch.tensor([[0.4]]), torch.tensor([[0.3]]), torch.tensor([[0.1]]),
              f_mod=0.7, da_phasic=da)
    hom = 0.4 * (0.6 * 0.3)
    eligibility = hom * 0.75
    da_gate = 0.7 if da is None else max(0.05, 0.7 * (0.5 + max(0.0, da)))
    dw = syn.cfg.eta * 0.7 * (0.65 * hom + 0.35 * da_gate * eligibility)
    assert syn.w_fast.item() == pytest.approx(0.7 * dw * 0.9995)
    assert syn.w_slow.item() == pytest.approx(0.3 * dw * syn.cfg.decay)
    assert syn.eligibility.item() == pytest.approx(eligibility)
    assert syn.z_pre.item() == pytest.approx(0.3)
    assert syn.z_post.item() == pytest.approx(0.1)
    assert syn.spine_age.item() == 1.0
    spent_resource = 1.0 - 0.15 * 0.1 * 0.3
    assert syn.u_syn.item() == pytest.approx(spent_resource + (1.0 - spent_resource) / 500.0)


def test_v19_default_api_and_explicit_none_are_identical():
    default = _moe(single_active=False)
    explicit_none = deepcopy(default)
    assert default.top_k_active == 2
    for _ in range(2):
        left = default.step(torch.ones(32))
        right = explicit_none.step(torch.ones(32), organism_context=None, broadcast=None, routing_bias=None)
        assert left.active_expert_ids == right.active_expert_ids
        torch.testing.assert_close(left.combined_spikes, right.combined_spikes, rtol=0, atol=0)
    _assert_tree_equal(default.export_state_dict(), explicit_none.export_state_dict())
    for col in default.columns:
        assert not col.pyr.external_energy
        assert col.tick_ladder.tick == default.global_clock_tick == 2
        for syn in (col.w_ff, col.w_rec, col.w_fb):
            assert not syn.externally_clocked
            assert syn.theta_m == syn.cfg.theta_m0


def test_omitting_context_releases_external_clock_ownership():
    cortex = _moe()
    cortex.step(torch.ones(32), organism_context=_context(1), routing_bias=_bias(0))
    cortex.step(torch.ones(32), learn=False, routing_bias=_bias(0))
    for col in cortex.columns:
        assert not col.pyr.external_energy
        assert "clock_tick" not in col._mod_ctx
        assert not col.w_ff.externally_clocked
