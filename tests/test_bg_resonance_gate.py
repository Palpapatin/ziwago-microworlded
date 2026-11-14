"""М-3: оживлённый гиппокампальный NoGo-гейт базальных ганглиев (opt-in).

Измерено до правки (прибор живости, AGENTS.md §7):
  * TD-критик УЖЕ жив в ContextualActionGate (settle → rpe → w_v меняется);
  * ca3_resonance в legacy BasalGangliaActionSelector.select_action БЫЛ мёртв
    (параметр игнорировался: resonance 0.0 и 1.0 → один и тот же action);
  * в v20 evidence-пути резонанс argmax-инвариантен для direct-vs-clarify
    (0.6+0.4r > 0.5(1-r) при всех r≥0) — отрицательный результат, см. §8.4.

Контракты гейта:
  * default off → резонанс игнорируется, v19/v20 bit-identical;
  * on + низкий резонанс + default fallback → ACTIVE_INFERENCE_CLARIFY («не знаю»);
  * on + высокий резонанс → DIRECT_SYNAPTIC_RESPONSE;
  * специфичные действия (teaching/episodic/...) гейтом не трогаются;
  * границы и валидация порога.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ziwago_assistant import BasalGangliaActionSelector
from ziwago.action.contextual import ContextualActionGate


class ResonanceGateTests(unittest.TestCase):
    def test_default_off_ignores_resonance(self):
        sel = BasalGangliaActionSelector()
        self.assertFalse(sel.resonance_gate_enabled)
        a0, s0 = sel.select_action("what is quux", False, False, False, 0.0)
        a1, s1 = sel.select_action("what is quux", False, False, False, 1.0)
        self.assertEqual(a0, a1)
        self.assertEqual(a0, sel.ACTION_DIRECT_SYNAPTIC_RESPONSE)
        self.assertEqual(s0, s1)

    def test_low_resonance_triggers_clarify(self):
        sel = BasalGangliaActionSelector(resonance_gate_enabled=True, unknown_resonance=0.3)
        action, _ = sel.select_action("what is quux", False, False, False, 0.1)
        self.assertEqual(action, sel.ACTION_ACTIVE_INFERENCE_CLARIFY)

    def test_high_resonance_keeps_direct(self):
        sel = BasalGangliaActionSelector(resonance_gate_enabled=True, unknown_resonance=0.3)
        action, _ = sel.select_action("what is neurons", False, False, False, 0.9)
        self.assertEqual(action, sel.ACTION_DIRECT_SYNAPTIC_RESPONSE)

    def test_threshold_is_strict_less_than(self):
        sel = BasalGangliaActionSelector(resonance_gate_enabled=True, unknown_resonance=0.3)
        at, _ = sel.select_action("q", False, False, False, 0.3)
        below, _ = sel.select_action("q", False, False, False, 0.2999)
        self.assertEqual(at, sel.ACTION_DIRECT_SYNAPTIC_RESPONSE)
        self.assertEqual(below, sel.ACTION_ACTIVE_INFERENCE_CLARIFY)

    def test_specific_actions_not_overridden(self):
        sel = BasalGangliaActionSelector(resonance_gate_enabled=True, unknown_resonance=0.3)
        teach, _ = sel.select_action("remember that x", False, True, False, 0.0)
        episodic, _ = sel.select_action("what did I say", True, False, False, 0.0)
        multi, _ = sel.select_action("step by step", False, False, True, 0.0)
        self.assertEqual(teach, sel.ACTION_ONLINE_DA_CONSOLIDATION)
        self.assertEqual(episodic, sel.ACTION_EPISODIC_MEMORY_RECALL)
        self.assertEqual(multi, sel.ACTION_MULTI_HOP_OR_PARAGRAPH_SYNTHESIS)

    def test_scores_reflect_flags_not_gate(self):
        sel = BasalGangliaActionSelector(resonance_gate_enabled=True)
        action, scores = sel.select_action("what is quux", False, False, False, 0.0)
        self.assertEqual(action, sel.ACTION_ACTIVE_INFERENCE_CLARIFY)
        self.assertFalse(any(scores.values()))  # все флаги ложны, гейт их не меняет
        self.assertNotIn(sel.ACTION_DIRECT_SYNAPTIC_RESPONSE, scores)  # fallback вне scores

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            BasalGangliaActionSelector(unknown_resonance=-0.1)
        with self.assertRaises(ValueError):
            BasalGangliaActionSelector(unknown_resonance=1.1)


class TDCriticAliveTests(unittest.TestCase):
    """TD-критик — живой владелец ContextualActionGate, не дублируется."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_settle_updates_critic_and_returns_rpe(self):
        gate = ContextualActionGate(n_actions=7, context_dim=8, seed=3)
        ctx = torch.ones(8) * 0.3
        evidence = torch.tensor([0., 0., 0., 0., 0., 0., 1.])
        wv_before = gate.w_v.clone()
        decision = None
        for _ in range(200):
            out = gate.step(ctx, evidence,
                            modulators={"da": 0.6, "ach": 0.6, "ado": 0.1, "ox": 0.6,
                                        "ht": 0.3, "awake": True}, dt_ms=1.0, learn=True)
            if out["committed"]:
                decision = out
                break
        self.assertIsNotNone(decision)
        result = gate.settle(decision["decision_id"], 1.0, learn=True)
        self.assertTrue(result["consumed"])
        self.assertTrue(result["learned"])
        self.assertAlmostEqual(result["rpe"], 1.0, places=6)
        self.assertFalse(torch.equal(wv_before, gate.w_v))

    def test_settlement_is_one_shot(self):
        gate = ContextualActionGate(n_actions=7, context_dim=8, seed=3)
        ctx = torch.ones(8) * 0.3
        evidence = torch.tensor([0., 0., 0., 0., 0., 0., 1.])
        decision = None
        for _ in range(200):
            out = gate.step(ctx, evidence,
                            modulators={"da": 0.6, "ach": 0.6, "ado": 0.1, "ox": 0.6,
                                        "ht": 0.3, "awake": True}, dt_ms=1.0, learn=True)
            if out["committed"]:
                decision = out
                break
        first = gate.settle(decision["decision_id"], 1.0, learn=True)
        second = gate.settle(decision["decision_id"], 1.0, learn=True)
        self.assertTrue(first["consumed"])
        self.assertFalse(second["consumed"])  # one-shot: no second RPE on stale id


if __name__ == "__main__":
    unittest.main()