"""Регрессия: архивированные dead-модули не возвращаются в пакет."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


DEAD_MODULES = (
    "ziwago/core/e2e_predictive.py",
    "ziwago/motor/syllabary_decoder.py",
    "ziwago/embodied/arena.py",
    "ziwago/learning/phase.py",
    "ziwago/learning/predictive.py",
    "ziwago/learning/sync.py",
    "ziwago/consciousness/metrics.py",
)
ARCHIVE_MANIFEST = Path(r"D:\Ziwago18-storage\Archive\_dead_20261114\MANIFEST.json")


def test_dead_modules_are_absent_from_ziwago_package():
    package = Path(__file__).resolve().parents[1] / "ziwago"
    for relative in DEAD_MODULES:
        assert not (package / relative).exists(), f"dead module returned: {relative}"


def test_archive_manifest_contains_verified_sha256_for_every_dead_module():
    assert ARCHIVE_MANIFEST.is_file(), f"archive manifest missing: {ARCHIVE_MANIFEST}"
    manifest = json.loads(ARCHIVE_MANIFEST.read_text(encoding="utf-8"))
    entries = {entry["file"]: entry for entry in manifest["files"]}
    assert set(DEAD_MODULES) <= set(entries)
    for relative in DEAD_MODULES:
        entry = entries[relative]
        archive_path = ARCHIVE_MANIFEST.parent / entry["archive_file"]
        assert archive_path.is_file(), f"archive copy missing: {relative}"
        payload = archive_path.read_bytes()
        assert len(payload) == entry["size_bytes"], relative
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"], relative
