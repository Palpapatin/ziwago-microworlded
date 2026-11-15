from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from test_v19_spiking_language import tiny_core
from ziwago.organism.integrated import IntegrationConfig


def _assert_tree_equal(left, right, path="root"):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), f"tensor mismatch at {path}"
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), f"keys mismatch at {path}"
        for key in left:
            _assert_tree_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), f"length mismatch at {path}"
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_tree_equal(a, b, f"{path}[{index}]")
    else:
        assert left == right, f"value mismatch at {path}: {left!r} != {right!r}"


def _evaluate_candidate(core, parent_state, candidate):
    core.load_language_dynamic_state(parent_state)
    output = core.life.tick(core, candidate)
    return (
        output.combined_spikes.clone(),
        output.gating_weights.clone(),
        core.export_language_dynamic_state(),
    )


def test_two_candidate_branch_order_restores_column_state_and_next_step():
    torch.set_num_threads(1)
    core = tiny_core()
    core.enable_integrated_organism(config=IntegrationConfig(ticks_per_token=1))
    core.learning_enabled = False
    core.life.learning_enabled = False
    parent_state = core.export_language_dynamic_state()
    candidates = {
        "first": torch.full((64,), 0.1),
        "second": torch.full((64,), 0.7),
    }

    forward_order = {
        name: _evaluate_candidate(core, parent_state, candidates[name])
        for name in ("first", "second")
    }
    reverse_order = {
        name: _evaluate_candidate(core, parent_state, candidates[name])
        for name in ("second", "first")
    }

    for name in ("first", "second"):
        _assert_tree_equal(forward_order[name], reverse_order[name])


def test_language_snapshot_restores_recurrent_resource_without_integrated_clock():
    torch.set_num_threads(1)
    core = tiny_core()
    column = core.moe_cortex.columns[0]
    core.moe_cortex.integrated_clock_enabled = False
    column._vol_eject = torch.zeros(column.zcfg.n_neurons)
    before = column.w_rec.u_syn.clone()
    state = core.export_language_dynamic_state()

    column.w_rec.u_syn.add_(0.25)
    core.load_language_dynamic_state(state)

    assert torch.equal(column.w_rec.u_syn, before)


def test_beam_factor_two_is_independent_of_candidate_iteration_order():
    torch.set_num_threads(1)
    core = tiny_core()
    core.enable_integrated_organism(config=IntegrationConfig(ticks_per_token=1))
    core.learning_enabled = False
    core.life.learning_enabled = False
    core_state = core.export_language_dynamic_state()
    token_output = SimpleNamespace(
        active_expert_names=["Col0_SYNTAX_BA44", "Col1_LEXICAL_SEMANTICS"],
        gating_weights=torch.tensor([0.6, 0.4]),
    )
    core._language_token_distribution = lambda q, prefix, pos: (
        [0, 1], torch.tensor([1.0, 0.9]), token_output, torch.ones(64))
    barrier = core.moe_cortex.discrete_pc_barrier
    args = dict(
        q_vec=torch.ones(64),
        l5_vec=torch.ones(64),
        max_tokens=2,
        beam_width=2,
        branch_factor=2,
    )

    with patch.object(
        barrier, "evaluate_discrete_pin_and_stranger_check",
        return_value={"stranger_check_triggered": False},
    ):
        forward = core._synthesize_neural_beam_search(**args)
        _assert_tree_equal(core_state, core.export_language_dynamic_state())

        original_argsort = torch.argsort

        def reverse_candidate_iteration(values, *positional, **kwargs):
            ordered = original_argsort(values, *positional, **kwargs)
            return ordered.flip(0) if values.numel() == 2 else ordered

        with patch.object(torch, "argsort", side_effect=reverse_candidate_iteration):
            reverse = core._synthesize_neural_beam_search(**args)

    _assert_tree_equal(forward, reverse)
    _assert_tree_equal(core_state, core.export_language_dynamic_state())
