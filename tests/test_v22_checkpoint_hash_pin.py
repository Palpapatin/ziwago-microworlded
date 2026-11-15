from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ziwago.organism_core as flatmod
from ziwago.evaluation.gates import CheckpointCompatibilityError


def test_hash_pinned_loader_consumes_the_verified_snapshot(tmp_path):
    loader = object.__new__(flatmod.ZiwagoOrganism)
    loader.root = tmp_path
    snapshot_dir = tmp_path / "Produce"
    snapshot_dir.mkdir()
    source = tmp_path / "source.pt"
    original_payload = {"value": torch.tensor([7])}
    replacement_payload = {"value": torch.tensor([99])}
    torch.save(original_payload, source)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    original_load = torch.load
    loaded_paths = []

    def replace_source_before_read(path, **kwargs):
        loaded_paths.append(Path(path))
        torch.save(replacement_payload, source)
        return original_load(path, **kwargs)

    with mock.patch.object(flatmod.torch, "load", side_effect=replace_source_before_read):
        value = loader._consume_checkpoint_payload(
            source, expected, lambda payload: int(payload["value"].item()))

    assert value == 7
    assert loaded_paths and loaded_paths[0] != source
    assert int(torch.load(source, weights_only=True)["value"].item()) == 99
    assert list(snapshot_dir.iterdir()) == []


def test_unrestricted_legacy_pickle_requires_explicit_trust_and_hash(tmp_path):
    loader = object.__new__(flatmod.ZiwagoOrganism)
    loader.root = tmp_path
    source = tmp_path / "legacy.pt"
    torch.save({"value": torch.tensor([1])}, source)

    with pytest.raises(CheckpointCompatibilityError, match="explicit trust"):
        loader._consume_checkpoint_payload(
            source, None, lambda payload: payload,
            weights_only=False,
        )

    with pytest.raises(CheckpointCompatibilityError, match="expected SHA-256"):
        loader.load_genesis_checkpoint(
            source, trusted_legacy_pickle=True,
        )
