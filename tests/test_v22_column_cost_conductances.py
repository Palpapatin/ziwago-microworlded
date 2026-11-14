"""Проверки измеренного обучения колонки и проводимостей её сна."""
from __future__ import annotations

import torch

from ziwago.config import USNeuronCfg, USSynapseCfg, ZoneCfg
from ziwago.core.multirate import BennaFusiCascade, MinicolumnStateHierarchy, TIER_COLD
from ziwago.cortex.column import CorticalColumn


def _column():
    col = CorticalColumn(
        zcfg=ZoneCfg(name="SENS", g=0.5, n_neurons=8, n_pv=2, n_sst=2, n_vip=2),
        n_in=4, n_fb=8, ncfg=USNeuronCfg(stochastic=False),
        scfg=USSynapseCfg(density=1.0), seed=3)
    col.reset_state(1)
    return col


def _quiet_column_step(col, learn=True):
    col._mod_ctx = {"ach": 0.5, "ne": 0.3}
    x_ff = torch.zeros(1, 4)
    x_fb = torch.zeros(1, 8)
    return col.step(x_ff, x_fb, f_mod=1.0, vip_mod=0.0, learn=learn)


def test_column_reports_actual_synaptic_rows_changed_and_respects_learning_flag():
    col = _column()
    # Known nonzero teaching signal and inputs make changes observable.
    col._mod_ctx = {"ach": 0.5, "ne": 0.3}
    col.w_ff.learning_enabled = True
    col.w_fb.learning_enabled = True
    col.w_rec.learning_enabled = True
    x_ff = torch.ones(1, 4)
    x_fb = torch.ones(1, 8)
    col.pyr.theta.fill_(-100.0)
    _ = col.step(x_ff, x_fb, f_mod=1.0, vip_mod=0.0, learn=True)

    matrices = (col.w_ff, col.w_fb, col.w_rec)
    actual_rows = 0
    actual_weights = 0
    for syn in matrices:
        # Rows changed in either fast or slow matrix are counted once per synapse.
        changed = ((syn.w_fast != 0.0) | (syn.w_slow != syn.w_epoch0_snapshot)) & syn.mask
        actual_rows += int(changed.any(dim=1).sum().item())
        actual_weights += int(changed.sum().item())
    assert col.last_phase1_stats["measured_synaptic_rows_updated"] == actual_rows
    assert col.last_phase1_stats["measured_synaptic_weight_updates"] == actual_weights
    assert actual_rows > 0

    frozen = _column()
    _quiet_column_step(frozen, learn=False)
    assert frozen.last_phase1_stats["measured_synaptic_rows_updated"] == 0
    assert frozen.last_phase1_stats["measured_synaptic_weight_updates"] == 0
    assert frozen.phase1_telemetry()["measured_total_weight_rows_updated"] == 0


def test_multirate_wakeup_uses_real_sleep_conductances_and_changes_result():
    h_zero = MinicolumnStateHierarchy(n_neurons=8, tile_size=8)
    h_nonzero = MinicolumnStateHierarchy(n_neurons=8, tile_size=8)
    for h, conductance in ((h_zero, 0.0), (h_nonzero, 0.7)):
        h.tile_tier.fill_(TIER_COLD)
        h.tile_t_sleep.zero_()
        h.v_sleep.fill_(-60.0)
        h.g_ampa_sleep.fill_(conductance)
        h.g_nmda_sleep.fill_(conductance * 0.5)
        h.g_gaba_sleep.fill_(conductance * 0.25)
        h.tile_sleep_conductances_known.fill_(True)

    args = dict(
        tick=20,
        afferent_drive=torch.ones(1, 8),
        spikes=torch.zeros(1, 8),
        v_s=torch.full((1, 8), -60.0),
        adapt=torch.zeros(1, 8),
        g_ampa=torch.zeros(1, 8),
        g_nmda=torch.zeros(1, 8),
        g_gaba=torch.zeros(1, 8),
    )
    h_zero.update_tiers(**args)
    h_nonzero.update_tiers(**args)
    assert h_zero.last_wakeup["conductances_known"]
    assert h_nonzero.last_wakeup["conductances_known"]
    assert h_nonzero.last_wakeup["g_ampa_sleep_used"] > 0.0
    assert not torch.allclose(h_zero.pending_wakeup_v, h_nonzero.pending_wakeup_v)


def test_column_passes_live_sleep_conductances_to_hierarchy():
    col = _column()
    received = {}
    original = col.minicolumn_hierarchy.update_tiers

    def capture(**kwargs):
        received.update(kwargs)
        return original(**kwargs)

    col.minicolumn_hierarchy.update_tiers = capture
    _quiet_column_step(col, learn=False)
    assert torch.is_tensor(received["g_ampa"])
    assert torch.is_tensor(received["g_nmda"])
    assert torch.is_tensor(received["g_gaba"])


def test_cascade_is_not_owned_or_snapshotted_by_column():
    col = _column()
    assert not hasattr(col, "benna_fusi")
    assert "НЕ подключён" in BennaFusiCascade.__doc__
    assert not any("benna_fusi" in key for key in vars(col))
