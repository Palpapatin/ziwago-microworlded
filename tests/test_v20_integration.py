"""v19.5 integration contracts on the small real cortex; no master checkpoints."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test_v19_spiking_language import tiny_core
from train_ziwago_v14_pure_synaptic_dialogue import TokenizedDialogPair


def assert_tree_equal(case, a, b):
    if isinstance(a, torch.Tensor):
        case.assertEqual(a.dtype, b.dtype)
        case.assertTrue(torch.equal(a, b))
    elif isinstance(a, dict):
        case.assertEqual(set(a), set(b))
        for key in a:
            assert_tree_equal(case, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        case.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_tree_equal(case, x, y)
    else:
        case.assertEqual(a, b)


class V20IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_single_clock_owner_and_token_subticks(self):
        core = tiny_core()
        core.enable_integrated_organism()
        core.step(torch.ones(64) * 0.1, 3)
        self.assertEqual(core.chemistry.tick, 3)
        self.assertEqual(core.moe_cortex.global_clock_tick, 3)
        self.assertEqual(core.chemistry.time_ms, 3.0)
        self.assertEqual(core.life.last_tick["tick"], 3)
        before = core.moe_cortex.global_clock_tick
        core._cortical_step(torch.ones(64) * 0.1)
        self.assertEqual(core.moe_cortex.global_clock_tick - before, core.life.cfg.ticks_per_token)
        self.assertFalse(core.life.migration["chemistry_reinitialized"])
        self.assertTrue(core.moe_cortex.integrated_clock_enabled)

    def test_training_writes_weights_and_replay_credit_without_ngrams(self):
        core = tiny_core()
        core.enable_integrated_organism()
        pair = TokenizedDialogPair("lab", [1, 3], [1, 2, 3, 4, 5, 6, 7])
        report = core.train_wave_with_neuromodulator_reward([pair])
        self.assertEqual(report["dendritic_pairs_trained"], 1)
        self.assertEqual(report["physiology_blocked_pairs"], 0)
        self.assertGreater(report["spiking_token_updates"], 0)
        self.assertEqual(core.synaptic_bigram_g, {})
        self.assertEqual(core.synaptic_trigram_g, {})
        self.assertGreater(len(core.life.replay), 0)
        self.assertIn("organism", report)

    def test_sleep_closes_plasticity_and_replays_tagged_cues(self):
        core = tiny_core()
        core.enable_integrated_organism()
        pair = TokenizedDialogPair("lab", [1, 3], [1, 2, 3, 4, 5, 6, 7])
        core.train_wave_with_neuromodulator_reward([pair])
        vocab_before = list(core.dendritic_plateau_readout.col_to_tok)
        core.chemistry.request_sleep()
        self.assertEqual(core.chemistry.f_mod(), 0.0)
        blocked = core.train_wave_with_neuromodulator_reward(
            [TokenizedDialogPair("lab", [1, 3], [2, 3, 4])])
        self.assertEqual(blocked["physiology_blocked_pairs"], 1)
        self.assertEqual(blocked["dendritic_pairs_trained"], 0)
        self.assertEqual(core.dendritic_plateau_readout.col_to_tok, vocab_before)
        for _ in range(96):
            core.life.tick(core, torch.zeros(64))
        self.assertGreater(core.life.replay_events, 0)
        self.assertEqual(core.chemistry.f_mod(), 0.0)

    def test_frozen_evaluation_restores_full_state_even_on_exception(self):
        core = tiny_core()
        core.enable_integrated_organism()
        core.step(torch.ones(64) * 0.2, 5)
        before = core.export_runtime_state()
        with self.assertRaises(ValueError):
            with core.evaluation_context():
                core.step(torch.ones(64) * 0.4, 7)
                raise ValueError("probe")
        assert_tree_equal(self, before, core.export_runtime_state())

    def test_runtime_restore_replays_next_present_identically(self):
        core = tiny_core()
        core.enable_integrated_organism()
        x = torch.ones(64) * 0.2
        snapshot = core.export_runtime_state()

        for _ in range(core.life.cfg.ticks_per_token):
            core.life.tick(core, x)
        expected = core.export_runtime_state()

        core.load_runtime_state(snapshot)
        core.life.present(core, x)
        assert_tree_equal(self, expected, core.export_runtime_state())

    def test_dynamic_snapshot_roundtrip_covers_v20_elapsed_fields(self):
        core = tiny_core()
        core.enable_integrated_organism()
        core.step(torch.ones(64) * 0.2, 4)
        snapshot = core.export_language_dynamic_state()
        core.step(torch.ones(64) * 0.3, 6)
        core.load_language_dynamic_state(snapshot)
        assert_tree_equal(self, snapshot, core.export_language_dynamic_state())

    def test_action_credit_is_one_shot_and_signed(self):
        core = tiny_core()
        core.enable_integrated_organism()
        life = core.life
        life.action_context = torch.ones(64) * 0.1
        life.action_evidence = torch.tensor([0., 0., 0., 0., 0., 0., 1.])
        decision = None
        for _ in range(life.cfg.decision_budget_ms):
            life.tick(core, life.action_context)
            if life.action_result["committed"]:
                decision = dict(life.action_result)
                break
        self.assertIsNotNone(decision)
        self.assertEqual(decision["action"], 6)
        events_before = core.chemistry.reward_events
        first = life.settle(core, decision["decision_id"], 1.0)
        second = life.settle(core, decision["decision_id"], 1.0)
        self.assertTrue(first["consumed"])
        self.assertFalse(second["consumed"])
        self.assertEqual(core.chemistry.reward_events, events_before + 1)
        self.assertGreater(core.chemistry.s.da_phasic, 0.0)

    def test_broadcast_and_relay_are_causal_signals(self):
        core = tiny_core()
        core.enable_integrated_organism()
        norms = []
        for _ in range(8):
            core.life.tick(core, torch.ones(64) * 0.3)
            norms.append((core.life.last_relay_norm, core.life.last_broadcast_norm))
        self.assertTrue(any(b > 0.0 for _, b in norms))
        self.assertTrue(all(r < float(torch.ones(64).norm()) for r, _ in norms))


if __name__ == "__main__":
    unittest.main()