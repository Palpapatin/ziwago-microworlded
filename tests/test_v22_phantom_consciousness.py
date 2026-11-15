"""M-track phantom audit for consciousness/thalamus.

This file keeps the legacy import path working and adds explicit checks for the
honest 'not connected' cases required by the phantom-effect contract.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from ziwago.consciousness.meta_head import MetaHead, MetaHeadCfg
from ziwago.consciousness.workspace import GlobalWorkspace
from ziwago.config import WorkspaceCfg
from ziwago.thalamus.relay import ThalamusFO

from test_v22_consciousness_thalamus import *  # noqa: F401,F403
from test_v19_spiking_language import tiny_core


def test_meta_head_reports_honest_gate_connection_state():
    meta = MetaHead(MetaHeadCfg(d_hidden=16), d_gw=8, seed=7)
    assert meta.telemetry()["connected_gate_channels"] == []
    tel = meta.telemetry(replay_priority_connected=True)
    assert tel["gates_connected"] is False
    assert "apply_gates" in tel["gate_contract"]
    assert "bg_threshold" in tel["gate_keys"]
    assert tel["connected_gate_channels"] == ["replay_priority"]
    assert "IntegratedOrganism.record_credit" in tel["live_consumer_points"][0]
    channels = tel["gate_channel_status"]
    assert channels["replay_priority"]["connected"] is True
    for name, reason in (
        ("bg_threshold", "no_approved_action_gate_consumer_contract"),
        ("beta_gw_boost", "no_approved_workspace_ignition_consumer_contract"),
        ("trn_gain", "no_approved_trn_gain_consumer_contract"),
    ):
        assert channels[name]["connected"] is False
        assert channels[name]["consumer"] is None
        assert channels[name]["reason"] == reason
    assert tel["novelty_unit_consumed"] is True
    assert tel["novelty_unit_consumer"] == "MetaHead.replay_priority"
    assert tel["novelty_unit_direct_consumer_connected"] is False


def test_meta_head_reports_replay_priority_disabled_reason():
    meta = MetaHead(MetaHeadCfg(d_hidden=16), d_gw=8, seed=7)
    tel = meta.telemetry(replay_priority_connected=False)
    assert tel["gate_channel_status"]["replay_priority"] == {
        "connected": False,
        "consumer": "IntegratedOrganism.record_credit",
        "reason": "disabled_by_meta_replay_priority_enabled",
    }
    assert tel["live_consumer_points"] == []
    assert tel["novelty_unit_consumed"] is False
    assert tel["novelty_unit_consumer"] is None


def test_workspace_tracks_slot_indexing_for_energy_history():
    gw = GlobalWorkspace(WorkspaceCfg(d_gw=8, n_slots=4))
    del gw.telemetry
    candidates = [torch.randn(8), torch.randn(8), torch.randn(8)]
    out = gw.compete(candidates, ne=0.5)
    assert len(out.slot_energy_traces) == len(out.slot_energy_trace_slots)
    assert all(len(trace) >= 2 for trace in out.slot_energy_traces)
    assert set(out.slot_energy_trace_slots) <= {0, 1, 2, 3}
    assert len(gw.energy_hist) == 1
    assert gw.telemetry["energy"] == out.energy
    assert gw.telemetry["energy_consumer_connected"] is False
    assert gw.telemetry["energy_consumer_points"] == []
    assert gw.telemetry["energy_consumer_reason"] == (
        "diagnostic_only_no_approved_consumer_contract"
    )


def test_live_integrated_telemetry_exposes_gw_energy_as_unconsumed():
    torch.set_num_threads(1)
    core = tiny_core()
    core.enable_integrated_organism()
    core.life.tick(core, torch.ones(64) * 0.3)

    report = core.life.telemetry(core)
    meta = report["meta_head"]
    workspace = report["global_workspace"]
    assert "predicted_loss" in meta
    assert "surprise" in meta
    assert "epistemic" in meta
    assert meta["latest_gates"]["replay_priority"] == meta["replay_priority"]
    assert meta["gate_channel_status"]["beta_gw_boost"]["connected"] is False
    assert workspace["energy"] == report["last_tick"]["gw_energy"]
    assert workspace["energy_consumer_connected"] is False
    assert workspace["energy_consumer_points"] == []
    assert workspace["energy_consumer_reason"] == (
        "diagnostic_only_no_approved_consumer_contract"
    )


def test_relay_reports_connected_phrase_boundary_and_unwired_spindle():
    fo = ThalamusFO(4)
    assert fo.BOUNDARY_CONNECTED is True
    assert fo.SPINDLE_CONNECTED is False
    assert fo.telemetry["boundary_connected"] is True
    assert fo.telemetry["spindle_connected"] is False
    assert fo.telemetry["boundary_burst"] is False

    source = (ROOT / "ziwago" / "organism" / "flat.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    organism = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ZiwagoPureSynapticDialogueOrganismV14"
    )
    generate = next(
        node for node in organism.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_synthesize_single_beam_from_weights_only"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "mark_phrase_boundary"
        for node in ast.walk(generate)
    )

    out = fo.spindle(1000, 10.0)
    assert out.shape == (4,)
    assert "spindle" in fo.telemetry
