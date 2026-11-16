from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_ziwago_v19 as runner


class _AcceptanceAssistant:
    def __init__(self):
        self.genesis = SimpleNamespace(
            authority=SimpleNamespace(records=[]),
            document_ranker=object(),
            document_ranker_metadata={},
        )
        self.organism = SimpleNamespace(language_memory_records=[])

    @contextmanager
    def evaluation_context(self):
        yield


def test_open_book_acceptance_enforces_ranker_identity_ledger():
    assistant = _AcceptanceAssistant()
    cases = [{"family_id": "heldout", "prompt": "question"}]

    with pytest.raises(ValueError, match="complete selected-identity ledger"):
        runner.evaluate_frozen_acceptance(assistant, [], cases, "open_book")


def test_progress_json_replace_failure_preserves_previous_file(tmp_path):
    progress = tmp_path / "progress.json"
    progress.write_text('{"stage":"previous"}', encoding="utf-8")

    with mock.patch.object(runner.os, "replace", side_effect=OSError("replace failed")):
        with pytest.raises(OSError, match="replace failed"):
            runner._write_json_atomic(progress, {"stage": "next"})

    assert progress.read_text(encoding="utf-8") == '{"stage":"previous"}'
    assert list(tmp_path.iterdir()) == [progress]


@pytest.mark.parametrize(
    ("version", "filename"),
    [(19, "ziwago_v19_0_accepted.pt"), (20, "ziwago_v20_0_accepted.pt")],
)
def test_accepted_checkpoint_path_matches_runtime_version(tmp_path, version, filename):
    assert runner._accepted_checkpoint_path(tmp_path, version) == (
        tmp_path / "checkpoints" / filename
    )


def test_accepted_checkpoint_path_rejects_unknown_runtime(tmp_path):
    with pytest.raises(ValueError, match="Unsupported runtime"):
        runner._accepted_checkpoint_path(tmp_path, 21)
