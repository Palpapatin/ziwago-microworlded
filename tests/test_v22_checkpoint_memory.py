from __future__ import annotations

import torch

from ziwago.cortex.moe_cortex import load_runtime_tree
from ziwago.organism.flat import release_redundant_runtime_payload_fields


def test_runtime_restore_reuses_matching_model_tensor_storage():
    current = torch.zeros((1024, 1024), dtype=torch.float32)
    source = torch.full((1024, 1024), 2.5, dtype=torch.float32)
    current_ptr = current.data_ptr()

    restored = load_runtime_tree(current, source, _reuse_current_tensors=True)

    assert restored is current
    assert restored.data_ptr() == current_ptr
    assert torch.equal(restored, source)


def test_runtime_restore_preserves_shared_tensor_identity():
    shared = torch.arange(16, dtype=torch.float32)
    current = {"left": torch.zeros_like(shared), "right": torch.ones_like(shared)}
    source = {"left": shared, "right": shared}

    restored = load_runtime_tree(current, source, _reuse_current_tensors=True)

    assert restored["left"] is restored["right"]
    assert torch.equal(restored["left"], shared)


def test_runtime_restore_does_not_preserve_unrecorded_target_aliases():
    aliased = torch.zeros(16, dtype=torch.float32)
    current = {"left": aliased, "right": aliased}
    source = {
        "left": torch.full((16,), 1.0),
        "right": torch.full((16,), 2.0),
    }

    restored = load_runtime_tree(current, source, _reuse_current_tensors=True)

    assert restored["left"] is not restored["right"]
    assert torch.all(restored["left"] == 1.0)
    assert torch.all(restored["right"] == 2.0)


def test_runtime_payload_prunes_only_flat_duplicates_not_overlay_fields():
    payload = {
        "core_runtime_state": {"core": {}},
        "w_dend_out": torch.zeros((4, 4)),
        "w_core_dend_out": torch.zeros((4, 4)),
        "w_swr_dend_out": torch.zeros((4, 4)),
        "w_basal_branch": torch.ones((2, 2)),
        "w_apical_branch": torch.ones((2, 2)),
        "pos_phase_vecs": torch.ones((3, 2)),
        "w_episode_trace": torch.ones((2, 3)),
        "w_episode_trace_len": torch.ones((2,), dtype=torch.int32),
        "v19_state": {"preserved": True},
    }

    removed = release_redundant_runtime_payload_fields(payload)

    assert removed["fields"] == 3
    assert "w_dend_out" not in payload
    assert "w_core_dend_out" not in payload
    assert "w_swr_dend_out" not in payload
    assert payload["w_basal_branch"].shape == (2, 2)
    assert payload["w_apical_branch"].shape == (2, 2)
    assert payload["pos_phase_vecs"].shape == (3, 2)
    assert payload["w_episode_trace"].shape == (2, 3)
    assert payload["w_episode_trace_len"].shape == (2,)
    assert payload["v19_state"] == {"preserved": True}


def test_legacy_flat_checkpoint_without_runtime_snapshot_is_untouched():
    legacy = {"w_dend_out": torch.ones((2, 2)), "readout_config": {"d_vec": 2}}
    original = legacy["w_dend_out"]

    removed = release_redundant_runtime_payload_fields(legacy)

    assert removed["fields"] == 0
    assert legacy["w_dend_out"] is original
    assert legacy["readout_config"] == {"d_vec": 2}
