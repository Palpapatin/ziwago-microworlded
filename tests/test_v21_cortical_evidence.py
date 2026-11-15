from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from test_v19_spiking_language import tiny_core
from test_v20_integration import assert_tree_equal
from ziwago.organism.integrated import IntegrationConfig


@pytest.fixture
def core():
    torch.set_num_threads(1)
    organism = tiny_core()
    organism.enable_integrated_organism()
    return organism


def test_real_token_window_matches_all_physical_subticks(core):
    x = torch.ones(64) * 0.2
    before = core.export_runtime_state()
    outputs = []
    counts = []
    for _ in range(core.life.cfg.ticks_per_token):
        out = core.life.tick(core, x)
        outputs.append(out)
        active = set(out.active_expert_ids)
        counts.append(torch.tensor([
            float(core.moe_cortex._last_col_spikes[i].sum()) if i in active else 0.0
            for i in range(core.moe_cortex.n_columns)
        ]))
    expected_state = core.export_runtime_state()
    core.load_runtime_state(before)
    result = core.life.present(core, x)
    torch.testing.assert_close(result.combined_spikes,
                               torch.stack([o.combined_spikes for o in outputs]).mean(0))
    torch.testing.assert_close(result.combined_l5_drive,
                               torch.stack([o.combined_l5_drive for o in outputs]).mean(0))
    torch.testing.assert_close(result.spike_count_window, torch.stack(counts))
    assert core.chemistry.tick == len(outputs)
    import unittest
    assert_tree_equal(unittest.TestCase(), expected_state, core.export_runtime_state())


def _controlled_window(core, active_first=True):
    sample = core.life.tick(core, torch.zeros(64))
    early_spikes = torch.zeros_like(sample.combined_spikes)
    early_l5 = torch.zeros_like(sample.combined_l5_drive)
    if active_first:
        early_spikes[:, 0] = 1.0
        early_l5[:, 0] = 1.0
    early = replace(sample, combined_spikes=early_spikes, combined_l5_drive=early_l5,
                    active_expert_ids=[0], active_expert_names=["early"],
                    gating_weights=torch.tensor([1.0, 0.0]),
                    spike_count_window=torch.tensor([[float(active_first), 0.0]]))
    late = replace(sample, combined_spikes=torch.zeros_like(early_spikes),
                   combined_l5_drive=torch.zeros_like(early_l5),
                   active_expert_ids=[1], active_expert_names=["late"],
                   gating_weights=torch.tensor([0.0, 1.0]),
                   spike_count_window=torch.zeros(1, 2))
    sequence = iter([early, late, late, late])

    def tick(_core, _x):
        out = next(sequence)
        for i in range(core.moe_cortex.n_columns):
            core.moe_cortex._last_col_spikes[i] = (
                out.combined_spikes.clone() if i in out.active_expert_ids
                else torch.zeros_like(out.combined_spikes))
        return out

    with patch.object(core.life, "tick", side_effect=tick):
        return core.life.present(core, torch.zeros(64))


def test_silent_final_tick_retains_earlier_evidence_and_expert_credit(core):
    out = _controlled_window(core)
    assert out.active_expert_ids == [0, 1]
    torch.testing.assert_close(out.gating_weights, torch.tensor([0.25, 0.75]))
    assert out.combined_spikes[0, 0] == 0.25
    assert out.combined_l5_drive[0, 0] == 0.25
    assert out.spike_count_window[:, 0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert out.spike_count_window[:, 1].count_nonzero() == 0
    loop = core.spiking_language_loop
    loop.ensure_tokens([1, 2])
    loop.w_projector[:, 0].fill_(1.0)
    loop.w_projector[:, 1].fill_(-1.0)
    scores = loop.predict(out, [1, 2])
    assert scores[0] > scores[1]
    assert loop.last_features.norm() > 0
    assert torch.equal(loop.last_features, loop.cortical_features(out))
    assert loop.predict(out, [1, 2], mode="disabled").count_nonzero() == 0


def test_completely_silent_window_does_not_fabricate_features(core):
    out = _controlled_window(core, active_first=False)
    loop = core.spiking_language_loop
    loop.ensure_tokens([1, 2])
    loop.w_projector.fill_(1.0)
    assert loop.predict(out, [1, 2]).count_nonzero() == 0
    assert loop.last_features.count_nonzero() == 0
    assert out.spike_count_window.count_nonzero() == 0


def test_inference_and_learning_use_the_same_window_features(core):
    out = _controlled_window(core)
    loop = core.spiking_language_loop
    loop.ensure_tokens([1, 2])
    loop.predict(out, [1, 2])
    features = loop.last_features.clone()
    before = loop.w_projector.clone()
    error = -(features @ before).softmax(0)
    error[1] += 1
    loop.learn(out, 2, [1, 2])
    torch.testing.assert_close(loop.w_projector, before + loop.learning_rate * torch.outer(features, error))
    assert torch.equal(loop.last_features, features)


def test_raster_records_subticks_not_only_last_tick_and_is_bounded(core):
    out = _controlled_window(core)
    loop = core.spiking_language_loop
    loop.last_spike_raster = torch.ones(511, core.moe_cortex.n_columns)
    with patch.object(core.life, "present", return_value=out):
        loop.step(torch.zeros(64), [], cortical_step=core._cortical_step)
    assert loop.last_spike_raster.shape == (512, core.moe_cortex.n_columns)
    assert torch.equal(loop.last_spike_raster[-4:], out.spike_count_window)
    assert loop.total_steps == 1


def test_single_tick_preserves_evidence(core):
    core.life.cfg = IntegrationConfig(ticks_per_token=1)
    x = torch.ones(64) * 0.2
    snapshot = core.export_runtime_state()
    raw = core.life.tick(core, x)
    core.load_runtime_state(snapshot)
    window = core.life.present(core, x)
    assert torch.equal(raw.combined_spikes, window.combined_spikes)
    assert torch.equal(raw.combined_l5_drive, window.combined_l5_drive)
    assert window.spike_count_window.shape[0] == 1


def test_language_snapshot_replays_window_and_restores_prediction_telemetry(core):
    loop = core.spiking_language_loop
    loop.ensure_tokens([1, 2])
    snapshot = core.export_language_dynamic_state()
    initial_features = loop.last_features.clone()
    x = torch.ones(64) * 0.2
    a, out_a = loop.step(x, [1, 2], cortical_step=core._cortical_step)
    features = loop.last_features.clone()
    raster = loop.last_spike_raster.clone()
    core.load_language_dynamic_state(snapshot)
    assert torch.equal(loop.last_features, initial_features)
    b, out_b = loop.step(x, [1, 2], cortical_step=core._cortical_step)
    assert torch.equal(a, b)
    assert torch.equal(out_a.spike_count_window, out_b.spike_count_window)
    assert torch.equal(features, loop.last_features)
    assert torch.equal(raster, loop.last_spike_raster)


def test_language_snapshot_restores_workspace_history_telemetry_and_next_result(core):
    gw = core.son_column.gw
    gw.energy_hist[:] = [0.125, 0.25]
    gw.telemetry["energy"] = 0.25
    gw.telemetry["test_marker"] = "saved branch"
    snapshot = core.export_language_dynamic_state()

    candidates = [
        torch.linspace(0.0, 1.0, gw.cfg.d_gw),
        torch.linspace(1.0, 0.0, gw.cfg.d_gw),
    ]
    expected = gw.compete(candidates, ne=0.2)
    expected_history = list(gw.energy_hist)
    expected_telemetry = dict(gw.telemetry)

    gw.energy_hist[:] = [-1.0]
    gw.telemetry["energy"] = -1.0
    gw.telemetry["test_marker"] = "discarded branch"
    core.load_language_dynamic_state(snapshot)

    assert gw.energy_hist == [0.125, 0.25]
    assert gw.telemetry["energy"] == 0.25
    assert gw.telemetry["test_marker"] == "saved branch"
    assert gw.telemetry["energy_consumer_connected"] is False
    actual = gw.compete(candidates, ne=0.2)
    torch.testing.assert_close(actual.broadcast, expected.broadcast, rtol=0, atol=0)
    assert actual.energy == expected.energy
    assert actual.ignition == expected.ignition
    assert actual.coherence_weights == expected.coherence_weights
    assert gw.energy_hist == expected_history
    assert gw.telemetry == expected_telemetry
