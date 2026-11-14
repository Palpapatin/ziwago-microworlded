"""Stage F: content/form composer with causal theta-gamma temporal binding.

Verifies the П-5 liveliness criterion: the TTFS multiplexer causally participates
in composition (decoded vectors drive recovery and form), not merely exists.
Opt-in default off is neutral; loss/corruption refuse or change the output rather
than fabricating fillers. No n-grams, no global backprop, no checkpoint changes.
"""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from ziwago.cognition.language.content_form import ContentFormComposer, RoleFiller
from ziwago.cortex.genesis_engine import ThetaGammaSpikingPhaseMultiplexer

AGENT, ACTION, OBJECT = ContentFormComposer.ROLE_SLOT_NAMES[:3]


def _vec(seed: int, d: int = 8) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(d, generator=g), dim=0) * 0.4


def _vocab(d: int = 8):
    return {
        AGENT: {"astrocyte": _vec(1, d), "neuron": _vec(2, d)},
        ACTION: {"release": _vec(3, d), "inhibit": _vec(4, d)},
        OBJECT: {"serine": _vec(5, d), "glutamate": _vec(6, d)},
    }


def _fillers(vocab):
    return [
        RoleFiller(AGENT, "astrocyte", vocab[AGENT]["astrocyte"], "user"),
        RoleFiller(ACTION, "release", vocab[ACTION]["release"], "user"),
        RoleFiller(OBJECT, "serine", vocab[OBJECT]["serine"], "document"),
    ]


def _render(c):
    return f"{c[AGENT]} {c[ACTION]} {c[OBJECT]}."


class _LossyMultiplexer(ThetaGammaSpikingPhaseMultiplexer):
    """Injects event loss after encoding: a dropped role decodes to silence."""
    def __init__(self, drop_role=None, drop_all=False, **kw):
        super().__init__(**kw)
        self.drop_role = drop_role
        self.drop_all = drop_all

    def encode_roles_into_theta_spike_train(self, role_vectors):
        raster, windows = super().encode_roles_into_theta_spike_train(role_vectors)
        if self.drop_all:
            raster.zero_()
        elif self.drop_role is not None:
            start, end = windows[self.drop_role]
            raster[start:end].zero_()
        return raster, windows


class _CorruptMultiplexer(ThetaGammaSpikingPhaseMultiplexer):
    """Transmits a different filler vector for one role (content corruption)."""
    def __init__(self, corrupt_role=None, corrupt_vec=None, **kw):
        super().__init__(**kw)
        self.corrupt_role = corrupt_role
        self.corrupt_vec = corrupt_vec

    def encode_roles_into_theta_spike_train(self, role_vectors):
        rv = dict(role_vectors)
        if self.corrupt_role is not None:
            rv[self.corrupt_role] = self.corrupt_vec
        return super().encode_roles_into_theta_spike_train(rv)


class ContentFormTests(unittest.TestCase):
    def _composer(self, **kw):
        params = dict(d_vec=8, enabled=True)
        params.update(kw)
        return ContentFormComposer(**params)

    def test_default_off_is_neutral(self):
        c = ContentFormComposer(d_vec=8)  # enabled=False
        vocab = _vocab()
        self.assertIsNone(c.compose(_fillers(vocab), vocab, _render))
        self.assertEqual(c.last_report, {})

    def test_clean_compose_preserves_roles_and_provenance(self):
        c = self._composer()
        vocab = _vocab()
        out = c.compose(_fillers(vocab), vocab, _render)
        self.assertTrue(out["composed"])
        self.assertEqual(out["content"], {AGENT: "astrocyte", ACTION: "release", OBJECT: "serine"})
        self.assertEqual(out["text"], "astrocyte release serine.")
        self.assertEqual(out["closure_score"], 1.0)
        self.assertEqual(out["recovery_fidelity"], 1.0)
        self.assertEqual(out["availability_fraction"], 1.0)
        self.assertEqual(out["composition_mode"], "compose")
        self.assertEqual(out["provenance"], {AGENT: "user", ACTION: "user", OBJECT: "document"})

    def test_event_loss_causally_refuses_filler(self):
        # П-5: dropping one role's gamma slot makes its decoded vector silence,
        # so recovery fails and composition refuses -- the frame causally matters.
        mux = _LossyMultiplexer(d_vec=8, drop_role=OBJECT)
        c = self._composer(multiplexer=mux)
        vocab = _vocab()
        out = c.compose(_fillers(vocab), vocab, _render)
        self.assertFalse(out["composed"])
        self.assertEqual(out["reason"], "filler_lost")
        self.assertEqual(out["role"], OBJECT)

    def test_availability_gate_refuses_on_loss(self):
        mux = _LossyMultiplexer(d_vec=8, drop_role=OBJECT)
        c = self._composer(multiplexer=mux, availability_threshold=0.9)
        vocab = _vocab()
        out = c.compose(_fillers(vocab), vocab, _render)
        self.assertFalse(out["composed"])
        self.assertEqual(out["reason"], "temporal_loss")
        self.assertLess(out["availability_fraction"], 0.9)

    def test_silence_refuses_not_fabricates(self):
        mux = _LossyMultiplexer(d_vec=8, drop_all=True)
        c = self._composer(multiplexer=mux)
        vocab = _vocab()
        out = c.compose(_fillers(vocab), vocab, _render)
        self.assertFalse(out["composed"])
        self.assertEqual(out["reason"], "temporal_loss")
        self.assertEqual(out["availability_fraction"], 0.0)

    def test_corruption_form_follows_decoded_content(self):
        # П-5: transmitting a different agent vector changes the rendered form and
        # lowers fidelity -- output follows the decoded frame, not the intended plan.
        vocab = _vocab()
        mux = _CorruptMultiplexer(d_vec=8, corrupt_role=AGENT, corrupt_vec=vocab[AGENT]["neuron"])
        c = self._composer(multiplexer=mux)
        out = c.compose(_fillers(vocab), vocab, _render)
        self.assertTrue(out["composed"])
        self.assertEqual(out["content"][AGENT], "neuron")
        self.assertEqual(out["text"], "neuron release serine.")
        self.assertLess(out["recovery_fidelity"], 1.0)
        self.assertEqual(out["closure_score"], 1.0)

    def test_content_form_separation(self):
        c = self._composer()
        vocab = _vocab()
        fillers = _fillers(vocab)
        render2 = lambda cc: f"{cc[OBJECT]} ({cc[AGENT]}/{cc[ACTION]})"
        out1 = c.compose(fillers, vocab, _render)
        out2 = c.compose(fillers, vocab, render2)
        self.assertEqual(out1["content"], out2["content"])      # same content
        self.assertNotEqual(out1["text"], out2["text"])          # different form
        self.assertEqual(out2["text"], "serine (astrocyte/release)")
        self.assertEqual(out2["closure_score"], 1.0)

    def test_closure_failure_refuses(self):
        c = self._composer()
        vocab = _vocab()
        drop_object = lambda cc: f"{cc[AGENT]} {cc[ACTION]}."   # omits OBJECT
        out = c.compose(_fillers(vocab), vocab, drop_object)
        self.assertFalse(out["composed"])
        self.assertEqual(out["reason"], "closure_failed")
        self.assertLess(out["closure_score"], 1.0)

    def test_replay_vs_compose(self):
        c = self._composer()
        vocab = _vocab()
        signature = frozenset({(AGENT, "astrocyte"), (ACTION, "release"), (OBJECT, "serine")})
        replay = c.compose(_fillers(vocab), vocab, _render, known_episodes={signature})
        self.assertEqual(replay["composition_mode"], "replay")
        novel = c.compose(_fillers(vocab), vocab, _render, known_episodes=set())
        self.assertEqual(novel["composition_mode"], "compose")

    def test_empty_content_refuses(self):
        c = self._composer()
        out = c.compose([], _vocab(), _render)
        self.assertFalse(out["composed"])
        self.assertEqual(out["reason"], "empty_content")

    def test_duplicate_and_unknown_role_rejected(self):
        c = self._composer()
        vocab = _vocab()
        dup = [RoleFiller(AGENT, "astrocyte", vocab[AGENT]["astrocyte"]),
               RoleFiller(AGENT, "neuron", vocab[AGENT]["neuron"])]
        with self.assertRaises(ValueError):
            c.compose(dup, vocab, _render)
        unknown = [RoleFiller("GAMMA_9_UNKNOWN", "x", _vec(1))]
        with self.assertRaises(ValueError):
            c.compose(unknown, vocab, _render)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            ContentFormComposer(d_vec=8, availability_threshold=1.5)
        with self.assertRaises(ValueError):
            ContentFormComposer(d_vec=8, value_scale=0.0)
        with self.assertRaises(ValueError):
            ContentFormComposer(d_vec=8, multiplexer=ThetaGammaSpikingPhaseMultiplexer(d_vec=16))

    def test_recover_silence_returns_none(self):
        c = self._composer()
        label, cos = c._recover(torch.zeros(8), _vocab()[AGENT])
        self.assertIsNone(label)
        self.assertLess(cos, c.recovery_threshold)

    def test_export_load_roundtrip_and_guard(self):
        c = self._composer(availability_threshold=0.7, recovery_threshold=0.6)
        vocab = _vocab()
        c.compose(_fillers(vocab), vocab, _render)
        state = c.export_state_dict()
        fresh = ContentFormComposer(d_vec=8)
        fresh.load_state_dict(state)
        self.assertTrue(fresh.enabled)
        self.assertEqual(fresh.availability_threshold, 0.7)
        self.assertEqual(fresh.recovery_threshold, 0.6)
        out = fresh.compose(_fillers(vocab), vocab, _render)
        self.assertTrue(out["composed"])
        self.assertEqual(out["text"], "astrocyte release serine.")
        with self.assertRaises(ValueError):
            fresh.load_state_dict({**state, "availability_threshold": 2.0})


if __name__ == "__main__":
    unittest.main()