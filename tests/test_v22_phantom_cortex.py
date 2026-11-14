from __future__ import annotations

import torch

from ziwago.config import USNeuronCfg, USSynapseCfg, ZoneCfg
from ziwago.core.dendrite import DendriticTree
from ziwago.cortex.column import CorticalColumn
from ziwago.organism.admc import AllostaticCuriosityController


def _column():
    col = CorticalColumn(
        zcfg=ZoneCfg(name="SENS", g=0.5, n_neurons=8, n_pv=2, n_sst=2, n_vip=2),
        n_in=4,
        n_fb=8,
        ncfg=USNeuronCfg(stochastic=False),
        scfg=USSynapseCfg(density=1.0),
        seed=13,
    )
    col.reset_state(1)
    col._mod_ctx = {"ach": 0.5, "ne": 0.3}
    return col


def test_local_sigma_is_connected_to_column_learning():
    low = _column()
    high = _column()
    x_ff = torch.ones(1, 4)
    x_fb = torch.ones(1, 8)

    low.local_sigma = 0.75
    high.local_sigma = 1.25

    low.step(x_ff, x_fb, f_mod=1.0, vip_mod=0.0, learn=True)
    high.step(x_ff, x_fb, f_mod=1.0, vip_mod=0.0, learn=True)

    assert low.last_phase1_stats["local_sigma_plasticity_gain"] < high.last_phase1_stats["local_sigma_plasticity_gain"]
    assert high.last_phase1_stats["local_sigma_plasticity_gain"] > 1.0


def test_dendritic_somatic_prediction_is_used_by_learning_when_error_is_missing():
    tree = DendriticTree(n_neurons=3, n_in=5, n_branch=3, seed=7, enable_deep_7level=False)
    x = torch.randn(2, 5)
    spikes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)

    tree.step(x, spikes)
    pred = tree.somatic_prediction()
    err = tree.somatic_error()
    assert torch.allclose(err, spikes - pred, atol=1e-6)

    alpha_before = tree.alpha.clone()
    tree.learn(error=None, x_in=x)
    assert not torch.equal(alpha_before, tree.alpha)


def test_admc_regulation_is_applied_to_generation_limits():
    c = AllostaticCuriosityController(
        tau_allo=100.0,
        allo_threshold=1.0,
        allo_width=0.5,
        depth_reduce_max=0.5,
        conf_add_max=0.3,
        phrase_reduce_max=0.5,
    )
    c.observe_error(10.0, dt=1.0)
    c.observe_error(3.0, dt=1.0)
    c.allo_load = 10.0
    c.regulation()

    policy = c.apply_generation_policy(12, 0.35, 24)
    assert policy["graph_depth"] < 12
    assert policy["phrase_length"] < 24
    assert policy["confidence_threshold"] >= 0.35
    assert c.telemetry()["prev_error"] == 3.0


def test_column_cost_tracking_is_zero_when_learning_is_off():
    col = _column()
    col.step(torch.zeros(1, 4), torch.zeros(1, 8), f_mod=1.0, vip_mod=0.0, learn=False)
    stats = col.phase1_telemetry()
    assert stats["measured_total_weight_rows_updated"] == 0
    assert stats["measured_total_weight_updates"] == 0
