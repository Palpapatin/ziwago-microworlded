from __future__ import annotations

import torch
from unittest.mock import patch

from ziwago.config import USNeuronCfg, USSynapseCfg, ZoneCfg
from ziwago.core.dendrite import DendriticTree
from ziwago.core.synapse import GenesisSynapse
from ziwago.cortex.astrocyte import AstrocyteMicrodomain
from ziwago.cortex.column import CorticalColumn


def _column():
    col = CorticalColumn(
        zcfg=ZoneCfg(name="SENS", g=0.5, n_neurons=8, n_pv=2, n_sst=2, n_vip=2),
        n_in=4,
        n_fb=8,
        ncfg=USNeuronCfg(stochastic=False),
        scfg=USSynapseCfg(density=1.0),
        seed=11,
    )
    col.reset_state(1)
    col._mod_ctx = {"ach": 0.5, "ne": 0.5}
    return col


def test_astrocyte_increases_stp_gain_under_active_dendritic_state_and_reduces_it_under_quiet_state():
    quiet = _column()
    active = _column()

    quiet.astrocyte.step(torch.zeros(1, quiet.dendrite.n))
    active.astrocyte.step(torch.ones(1, active.dendrite.n))

    quiet.w_ff.set_stp_gain(float(quiet.astrocyte.step(torch.zeros(1, quiet.dendrite.n)).mean().item()))
    active.w_ff.set_stp_gain(float(active.astrocyte.step(torch.ones(1, active.dendrite.n)).mean().item()))
    quiet.w_rec.set_stp_gain(quiet.w_ff.stp_gain)
    active.w_rec.set_stp_gain(active.w_ff.stp_gain)

    assert quiet.w_ff.stp_gain < active.w_ff.stp_gain
    assert quiet.w_rec.stp_gain < active.w_rec.stp_gain
    assert active.w_ff.stp_gain > 1.0


def test_stp_gain_changes_the_learning_update_magnitude():
    syn = GenesisSynapse(n_pre=4, n_post=2, cfg=USSynapseCfg(), seed=5)
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    err = torch.tensor([[1.0, -0.5], [0.0, 0.25]], dtype=torch.float32)
    syn.set_stp_gain(1.0)
    before = syn.learn(error=err, s_pre=x, s_post=torch.ones_like(err), f_mod=1.0)
    syn.set_stp_gain(2.0)
    after = syn.learn(error=err, s_pre=x, s_post=torch.ones_like(err), f_mod=1.0)
    assert abs(after.dw_mean) > abs(before.dw_mean)


def test_astrocyte_theta_ca_shift_is_applied_to_dendrite_and_ca_is_initialized_in_ctor():
    astro = AstrocyteMicrodomain(n_neurons=2)
    assert astro.ca is not None
    assert astro.ca.shape == (1, 2)

    tree = DendriticTree(n_neurons=2, n_in=4, n_branch=3, seed=1)
    base = float(tree.theta_ca)
    astro.step(torch.tensor([[1.0, 0.0]]))
    shift = astro.apply_theta_ca_shift(tree)
    assert float(tree.theta_ca) == base + shift
    assert shift < 0.0


def test_astrocyte_threshold_is_absolute_and_reset_returns_to_baseline():
    tree = DendriticTree(n_neurons=2, n_in=4, n_branch=3, seed=1)
    base = tree.theta_ca_base
    for shift in (-0.5, -1.0, -0.5, 0.0):
        assert tree.apply_theta_ca_shift(shift) == base + shift
    tree.apply_theta_ca_shift(-2.0)
    tree.reset_state(1)
    assert tree.theta_ca == base


def test_column_routes_same_tick_neuron_error_to_dendritic_learning():
    col = _column()
    x_ff = torch.ones(1, 4)
    x_fb = torch.zeros(1, 8)
    with patch.object(col.dendrite, "learn", wraps=col.dendrite.learn) as learn:
        out = col.step(x_ff, x_fb, f_mod=1.0, vip_mod=0.0, learn=True)

    learning_error = learn.call_args.args[0]
    torch.testing.assert_close(
        learning_error,
        out.error * col.last_local_plasticity_gain,
        rtol=0.0,
        atol=0.0,
    )
