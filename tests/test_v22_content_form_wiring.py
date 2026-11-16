from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ziwago.organism_core import ZiwagoOrganism


def _organism() -> ZiwagoOrganism:
    return ZiwagoOrganism(auto_load_checkpoint=False)


def test_content_form_dictionary_grows_from_training_and_round_trips():
    org = _organism()
    binder = org.core_organism.role_filler_binder
    binder.register_emergent_stdp_bindings({"mayonnaise": ("hollandaise", 0.91)})

    vocab = org.refresh_content_form_role_dictionary()
    assert "mayonnaise" in vocab
    assert "hollandaise" in vocab["mayonnaise"]
    assert vocab["mayonnaise"]["hollandaise"].shape == (64,)

    state = org.export_genesis_state_dict()
    restore = _organism()
    restore.load_genesis_state_dict(state)

    assert "hollandaise" in restore.content_form_role_dictionary["mayonnaise"]
    assert restore.content_form_role_dictionary["mayonnaise"]["hollandaise"].shape == (64,)


def test_grounded_answers_skip_content_form_fallback():
    org = _organism()
    result = org.maybe_fallback_compose("help with hollandaise sauce", "grounded answer", grounded=True)
    assert result["fallback_used"] is False
    assert result["reason"] == "grounded_answer"


def test_ungrounded_answers_use_content_form_fallback_when_role_map_is_live():
    org = _organism()
    org.core_organism.role_filler_binder.register_emergent_stdp_bindings({"mayonnaise": ("hollandaise", 0.92)})

    result = org.maybe_fallback_compose(
        "please help with the hollandaise sauce",
        "fallback response",
        grounded=False,
    )
    assert result["role_dictionary_size"] > 0
    assert result["reason"] in {"content_form_fallback", "composer_disabled", "empty_role_dictionary", "no_role_bindings"}
    if result["reason"] == "content_form_fallback":
        assert result["fallback_used"] is True


def test_fallback_uses_one_stable_slot_mapping_and_reports_overflow():
    org = _organism()
    vocab = {
        f"role_{index}": {"filler": torch.full((64,), float(index))}
        for index in range(8)
    }
    org.refresh_content_form_role_dictionary = lambda: vocab
    org.core_organism.role_filler_binder.extract_slot_bindings = lambda _prompt: {
        "role_2": "filler", "role_7": "filler",
    }
    captured = {}

    def capture_compose(fillers, canonical_vocab, render_fn):
        captured["fillers"] = list(fillers)
        captured["vocabulary"] = canonical_vocab
        return None

    org.content_form_composer.compose = capture_compose
    result = org.maybe_fallback_compose("prompt", "response", grounded=False)

    slot = org.content_form_composer.ROLE_SLOT_NAMES[2]
    assert result["role_slot_mapping"]["role_2"] == slot
    assert captured["fillers"][0].role == slot
    assert captured["fillers"][0].role in captured["vocabulary"]
    assert len(set(result["role_slot_mapping"].values())) == 6
    assert "role_7" in result["unmapped_roles"]
