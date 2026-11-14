"""M-track phantom audit for consciousness/thalamus.

This file keeps the legacy import path working and adds explicit checks for the
honest 'not connected' cases required by the phantom-effect contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ziwago.consciousness.meta_head import MetaHead, MetaHeadCfg
from ziwago.consciousness.workspace import GlobalWorkspace
from ziwago.config import WorkspaceCfg
from ziwago.thalamus.relay import ThalamusFO

from test_v22_consciousness_thalamus import *  # noqa: F401,F403


def test_meta_head_reports_honest_gate_connection_state():
    meta = MetaHead(MetaHeadCfg(d_hidden=16), d_gw=8, seed=7)
    tel = meta.telemetry()
    assert tel["gates_connected"] is False
    assert "apply_gates" in tel["gate_contract"]
    assert "bg_threshold" in tel["gate_keys"]


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


def test_relay_marks_unwired_boundary_and_spindle_as_honest_non_live_hooks():
    fo = ThalamusFO(4)
    assert fo.BOUNDARY_CONNECTED is False
    assert fo.SPINDLE_CONNECTED is False
    assert fo.telemetry["boundary_connected"] is False
    assert fo.telemetry["spindle_connected"] is False
    assert fo.telemetry["boundary_burst"] is False
    out = fo.spindle(1000, 10.0)
    assert out.shape == (4,)
    assert "spindle" in fo.telemetry
