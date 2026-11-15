"""Causal wiring tests for the one-tick-delayed MetaHead replay channel."""
from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from test_v19_spiking_language import tiny_core
from ziwago.organism.integrated import IntegratedOrganism, IntegrationConfig


def _bare_life(enabled=True):
    life = IntegratedOrganism.__new__(IntegratedOrganism)
    life.cfg = IntegrationConfig(meta_replay_priority_enabled=enabled)
    life._meta_replay_priority_pending = 0.0
    life._last_meta_replay_priority_output = 0.0
    life._last_meta_replay_priority_applied = 0.0
    life.meta_replay_priority_credit_count = 0
    life.replay = []
    life.replay_evictions = 0
    life.confidence = 0.5
    life.learning_enabled = True
    life.hippo = None
    life._last_hippo = None
    return life


def _credit(life):
    core = SimpleNamespace(
        learning_enabled=True,
        chemistry=SimpleNamespace(is_awake=True),
        moe_cortex=SimpleNamespace(columns=[
            SimpleNamespace(_last_x_ff=torch.tensor([0.2, 0.8])),
        ]),
    )
    output = SimpleNamespace(active_expert_ids=[0])
    life.record_credit(core, torch.tensor([1.0, 2.0]), output, loss=0.25)


def test_meta_priority_is_applied_to_the_next_replay_credit():
    life = _bare_life()

    life._begin_meta_replay_priority_tick()
    life._stage_meta_replay_priority(SimpleNamespace(gates={"replay_priority": 0.4}))
    _credit(life)

    assert life.replay[0]["base_priority"] == pytest.approx(0.75)
    assert life.replay[0]["meta_replay_priority"] == 0.0
    assert life.replay[0]["priority"] == pytest.approx(0.75)

    life._begin_meta_replay_priority_tick()
    life._stage_meta_replay_priority(SimpleNamespace(gates={"replay_priority": 0.8}))
    _credit(life)

    assert life.replay[1]["base_priority"] == pytest.approx(0.75)
    assert life.replay[1]["meta_replay_priority"] == pytest.approx(0.4)
    assert life.replay[1]["priority"] == pytest.approx(1.15)
    assert life.meta_replay_priority_credit_count == 2
    # ПРИМЕНЁННОЕ значение — ровно то, что попало в запись реплея (0.4):
    # канал MetaHead работает с задержкой в один внутренний тик.
    assert life._last_meta_replay_priority_applied == pytest.approx(0.4)
    # А 0.8 приготовлено для СЛЕДУЮЩЕЙ записи и ещё не применено.
    assert life._last_meta_replay_priority_output == pytest.approx(0.8)


def test_meta_priority_switch_disables_only_the_added_priority():
    life = _bare_life(enabled=False)
    life._meta_replay_priority_pending = 0.6
    life._begin_meta_replay_priority_tick()
    life._stage_meta_replay_priority(SimpleNamespace(gates={"replay_priority": 0.9}))
    _credit(life)

    assert life._meta_replay_priority_pending == 0.0
    assert life.replay[0]["priority"] == pytest.approx(0.75)
    assert life.replay[0]["meta_replay_priority"] == 0.0
    assert life.meta_replay_priority_credit_count == 0


@pytest.mark.parametrize("ticks_per_token", [1, 4])
@pytest.mark.parametrize("enabled", [True, False])
def test_present_credit_delay_changes_the_next_replay_selection(ticks_per_token, enabled):
    torch.set_num_threads(1)
    core = tiny_core()
    core.enable_integrated_organism(config=IntegrationConfig(
        ticks_per_token=ticks_per_token,
        meta_replay_priority_enabled=enabled,
        dual_hippocampus_enabled=False,
    ))
    life = core.life
    meta_outputs = (
        [0.4, 0.8] if ticks_per_token == 1
        else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )
    outputs = iter(meta_outputs)

    def deterministic_meta_step(*args, **kwargs):
        return SimpleNamespace(
            confidence=0.5,
            gates={"replay_priority": next(outputs, 0.0)},
        )

    core.son_column.meta_head.step = deterministic_meta_step
    cues = [torch.full((64,), 1.0), torch.full((64,), 2.0)]
    losses = [0.3, 0.2]
    expected_meta = [0.0, 0.4] if ticks_per_token == 1 else [0.3, 0.7]
    recorded = []
    for cue, loss in zip(cues, losses):
        cortical_out = life.present(core, cue)
        life.record_credit(core, cue, cortical_out, loss=loss)
        recorded.append(life.replay[-1])

    expected_base = [0.8, 0.7]
    for item, base, meta_priority in zip(recorded, expected_base, expected_meta):
        assert item["base_priority"] == pytest.approx(base)
        expected_addition = meta_priority if enabled else 0.0
        assert item["meta_replay_priority"] == pytest.approx(expected_addition)
        assert item["priority"] == pytest.approx(base + expected_addition)

    core.chemistry.request_sleep()
    life.last_replay_ms = core.chemistry.time_ms - life.cfg.replay_interval_ms
    life.present(core, torch.zeros(64))

    assert life.replay_events == 1
    assert len(life.replay) == 1
    selected = recorded[0] if life.replay[0] is recorded[1] else recorded[1]
    expected_selected_cue = 2.0 if enabled else 1.0
    assert torch.allclose(selected["cue"], torch.full((64,), expected_selected_cue))


def test_tick_computes_priority_after_replay_selection_for_next_tick():
    function = ast.parse(textwrap.dedent(inspect.getsource(IntegratedOrganism.tick))).body[0]
    calls = [
        (node.lineno, node.func.attr)
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    meta_line = next(
        node.lineno for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "meta_head"
    )
    stage_line = next(line for line, name in calls if name == "_stage_meta_replay_priority")
    replay_selection_line = next(
        node.lineno for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "max"
        and node.args
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == "range"
    )
    assert replay_selection_line < meta_line < stage_line


def test_meta_replay_priority_config_requires_a_boolean():
    with pytest.raises(ValueError, match="must be a boolean"):
        IntegrationConfig(meta_replay_priority_enabled=1)


@pytest.mark.parametrize("value", [-0.1, float("nan")])
def test_invalid_meta_replay_priority_is_rejected(value):
    life = _bare_life()
    with pytest.raises(ValueError, match="finite and non-negative"):
        life._stage_meta_replay_priority(SimpleNamespace(gates={"replay_priority": value}))
