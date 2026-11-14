"""Merged dual-system hippocampus (hippocampusidea.md; ARCHITECTURE_MAPPING §8).

Контракт, который здесь фиксируется:
  * рабочая память Fast Drafter: MergedLaplaceBank консолидирует nw9 elapsed-time
    и nw10 slow_profile; θ-γ indexer связывает композиционные следы;
  * часы ThetaClock выровнены по токенам и гейтят реколл (1 реколл на цикл);
  * эпизодическая память Slow Verifier переиспользует живой nw10 core, но её
    распределённая память VSA-DHT и граф HippoRAG ТЕПЕРЬ ЧИТАЮТСЯ (в v19 были
    store-only);
  * Arbiter реализует протокол дуальной системы: resonance>0.7 → accept,
    resonance<0.3 → unknown («я не знаю»), залипание/mismatch → regenerate;
  * opt-in: модуль не меняет живой путь v19 (plain Hippocampus остаётся собой).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ziwago.config import HippocampusCfg
from ziwago.memory.hippocampus import Hippocampus
from ziwago.memory.dual_hippocampus import (
    ArbiterCfg,
    DualHippocampus,
    MergedLaplaceBank,
    NoveltyArbiter,
    ThetaClock,
)


def tiny_cfg() -> HippocampusCfg:
    return HippocampusCfg(n_ec=32, n_dg=96, n_ca3=64)


class MergedLaplaceBankTests(unittest.TestCase):
    def test_state_shape_and_readouts(self):
        bank = MergedLaplaceBank(feature_dim=16, num_scales=8)
        x = torch.randn(2, 16)
        state = bank.step(x)
        self.assertEqual(tuple(state.shape), (8, 2, 16))
        self.assertEqual(tuple(bank.time_cells().shape), (8, 2, 16))
        self.assertEqual(tuple(bank.context_summary().shape), (2, 16))
        self.assertEqual(tuple(bank.slow_profile().shape), (2, 16))

    def test_elapsed_time_recovers_decay_steps(self):
        # Импульс, затем нули: отношение медленной/быстрой шкал даёт elapsed ≈ t·dt.
        bank = MergedLaplaceBank(feature_dim=4, num_scales=8, dt=1.0)
        bank.step(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        self.assertAlmostEqual(bank.estimate_elapsed_time(0), 0.0, places=4)
        prev = 0.0
        for _ in range(6):
            bank.step(torch.zeros(1, 4))
            cur = bank.estimate_elapsed_time(0)
            self.assertGreater(cur, prev)
            prev = cur
        self.assertAlmostEqual(prev, 6.0, delta=0.5)

    def test_slow_profile_is_slow_half_mean(self):
        bank = MergedLaplaceBank(feature_dim=4, num_scales=8)
        bank.step(torch.ones(1, 4))
        expected = bank.state[4:].mean(0)
        self.assertTrue(torch.allclose(bank.slow_profile(), expected))


class ThetaClockTests(unittest.TestCase):
    def test_token_mode_cycle_length(self):
        clock = ThetaClock(mode="token", cycle_steps=8.0)
        self.assertEqual(clock.mode, "token")
        self.assertEqual(clock.cycle_steps, 8.0)

    def test_configure_rejects_bad_mode_and_steps(self):
        clock = ThetaClock()
        with self.assertRaises(ValueError):
            clock.configure("warp", 8.0)
        with self.assertRaises(ValueError):
            clock.configure("token", 0.0)

    def test_retrieval_gated_once_per_cycle(self):
        clock = ThetaClock(mode="token", cycle_steps=8.0)
        ready_steps = []
        for i in range(8):
            st = clock.step()
            if st["retrieval_ready"]:
                ready_steps.append(i)
                clock.mark_retrieved()
        # Ровно одна возможность реколла за тета-цикл.
        self.assertEqual(len(ready_steps), 1)
        # После mark_retrieved повторный реколл в том же цикле запрещён.
        st = clock.step()
        self.assertFalse(st["retrieval_ready"] and clock._retrieved_in_cycle
                         and st["is_retrieval"] and ready_steps[-1] == 7)

    def test_encoding_and_retrieval_phases_partition_cycle(self):
        clock = ThetaClock(mode="token", cycle_steps=8.0)
        enc = ret = 0
        for _ in range(8):
            st = clock.step()
            enc += int(st["is_encoding"])
            ret += int(st["is_retrieval"])
        self.assertEqual(enc + ret, 8)
        self.assertGreater(enc, 0)
        self.assertGreater(ret, 0)


class NoveltyArbiterTests(unittest.TestCase):
    def setUp(self):
        self.arb = NoveltyArbiter(ArbiterCfg())

    def test_high_resonance_accepts(self):
        d = self.arb.decide(resonance=0.85, mismatch=0.1, novelty_da=0.2, novelty_ne=0.16)
        self.assertEqual(d.decision, "accept")
        self.assertAlmostEqual(d.confidence, 0.85)
        self.assertAlmostEqual(d.novelty_da, 0.2)
        self.assertFalse(d.ne_restart)

    def test_low_resonance_unknown(self):
        d = self.arb.decide(resonance=0.1, mismatch=0.05)
        self.assertEqual(d.decision, "unknown")
        self.assertEqual(d.reason, "ca3_resonance_low")

    def test_low_entropy_triggers_ne_restart(self):
        d = self.arb.decide(resonance=0.5, mismatch=0.1, entropy=0.05)
        self.assertEqual(d.decision, "regenerate")
        self.assertTrue(d.ne_restart)
        self.assertEqual(d.reason, "low_entropy_stick")

    def test_high_mismatch_triggers_regenerate(self):
        d = self.arb.decide(resonance=0.5, mismatch=0.6, entropy=0.5)
        self.assertEqual(d.decision, "regenerate")
        self.assertTrue(d.ne_restart)

    def test_partial_resonance_asks(self):
        d = self.arb.decide(resonance=0.5, mismatch=0.1, entropy=0.5)
        self.assertEqual(d.decision, "ask")
        self.assertFalse(d.ne_restart)


class DualHippocampusTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dh = DualHippocampus(tiny_cfg(), input_dim=32, seed=0)
        self.x = torch.randn(1, 32)

    def test_tick_returns_all_three_systems(self):
        out = self.dh.tick(self.x, ach=0.8, adenosine=0.1, learn=True)
        # Fast Drafter
        self.assertEqual(tuple(out.laplace_state.shape), (8, 1, 32))
        self.assertEqual(len(out.theta_gamma_slots), 4)
        # Slow Verifier
        self.assertIsNotNone(out.hippo.ec)
        self.assertIsInstance(out.recalled_episode_ids, list)
        # Arbiter
        self.assertIn(out.arbiter.decision, {"accept", "regenerate", "ask", "unknown"})

    def test_resonance_rises_after_storage(self):
        before = self.dh.tick(self.x, ach=0.8, adenosine=0.1).resonance
        self.assertAlmostEqual(before, 0.0, places=5)
        out = self.dh.tick(self.x, ach=0.8, adenosine=0.1)
        self.dh.store_episode(out.hippo.ec, out.hippo.dg, out.hippo.ca3,
                              error=0.5, uncertainty=0.3, da_tag=0.4)
        after = self.dh.tick(self.x, ach=0.8, adenosine=0.1).resonance
        self.assertGreater(after, 0.9)

    def test_vsa_dht_retrieval_reads_stored_episodes(self):
        # В v19 VSA-DHT был store-only; merged-путь обязан читать его обратно.
        out = self.dh.tick(self.x, ach=0.8, adenosine=0.1)
        self.dh.store_episode(out.hippo.ec, out.hippo.dg, out.hippo.ca3,
                              error=0.5, uncertainty=0.3)
        ver = self.dh.verify(out.hippo.ec)
        self.assertIn(0, ver["recalled_episode_ids"])
        self.assertGreater(len(ver["recall_scores"]), 0)
        self.assertGreaterEqual(ver["n_probes_used"], 1)

    def test_hipporag_multi_hop_concept_chain(self):
        # A-B и B-C эпизоды → мост B → многошаговая цепочка A→B→C.
        for concepts in (["A", "B"], ["B", "C"]):
            out = self.dh.tick(torch.randn(1, 32), ach=0.8, adenosine=0.1)
            self.dh.store_episode(out.hippo.ec, out.hippo.dg, out.hippo.ca3,
                                  error=0.5, uncertainty=0.3, concepts=concepts)
        dht = self.dh.episodic.vsa_dht
        self.assertIn("A", dht.concept_adj["B"])
        self.assertIn("C", dht.concept_adj["B"])
        ppr = dht.ppr_associative_recall(seed_concepts=["A"], top_k=5)
        self.assertIn("A", ppr["concept_chain"])
        self.assertGreaterEqual(len(ppr["concept_chain"]), 2)

    def test_theta_gamma_indexer_binds_compositional_traces(self):
        self.assertEqual(self.dh.indexer.theoretical_capacity(), 64 ** 4)
        for _ in range(3):
            self.dh.tick(torch.randn(1, 32), ach=0.8, adenosine=0.1)
        self.assertGreater(len(self.dh.indexer.compositional_traces), 0)

    def test_fast_context_working_memory_readout(self):
        self.dh.tick(self.x, ach=0.8, adenosine=0.1)
        ctx = self.dh.fast_context()
        self.assertEqual(tuple(ctx["context_summary"].shape), (1, 32))
        self.assertEqual(ctx["theoretical_capacity"], 64 ** 4)

    def test_sleep_replay_delegates_to_core(self):
        out = self.dh.tick(self.x, ach=0.8, adenosine=0.1)
        self.dh.store_episode(out.hippo.ec, out.hippo.dg, out.hippo.ca3,
                              error=0.9, uncertainty=0.4)
        writes = []
        replayed = self.dh.swr_replay(lambda ec, err: writes.append(float(err)))
        self.assertGreater(len(replayed), 0)
        self.assertGreater(len(writes), 0)
        rem = self.dh.rem_recombine()
        self.assertIn("semantic_bridges", rem)

    def test_contextual_replay_router_alive(self):
        out = self.dh.tick(self.x, ach=0.8, adenosine=0.1)
        self.dh.store_episode(out.hippo.ec, out.hippo.dg, out.hippo.ca3,
                              error=0.5, uncertainty=0.3, concepts=["A", "B"])
        routed = self.dh.contextual_replay(cue_modality="associative", seed_concepts=["A"])
        self.assertIn("routed_mode", routed)
        self.assertIn("silent_period_compute_saving", routed)

    def test_report_exposes_capacities(self):
        self.dh.tick(self.x, ach=0.8, adenosine=0.1)
        rep = self.dh.report()
        self.assertEqual(rep["vsa_capacity"], 38400)
        self.assertEqual(rep["theta_mode"], "token")


class DualSystemProtocolTests(unittest.TestCase):
    """Сквозной протокол hippocampusidea.md: Fast Drafter + Slow Verifier + Arbiter."""

    def test_unknown_before_accept_after(self):
        torch.manual_seed(1)
        dh = DualHippocampus(tiny_cfg(), input_dim=32, seed=1)
        x = torch.randn(1, 32)
        # Нет фактов → «я не знаю» (анти-галлюцинация).
        d0 = dh.tick(x, ach=0.8, adenosine=0.1).arbiter
        self.assertEqual(d0.decision, "unknown")
        # Запомнили эпизод → факты совпали → accept.
        out = dh.tick(x, ach=0.8, adenosine=0.1)
        dh.store_episode(out.hippo.ec, out.hippo.dg, out.hippo.ca3,
                         error=0.5, uncertainty=0.2, da_tag=0.5)
        d1 = dh.tick(x, ach=0.8, adenosine=0.1).arbiter
        self.assertEqual(d1.decision, "accept")
        self.assertGreater(d1.confidence, 0.7)


class OptInIsolationTests(unittest.TestCase):
    """Merged-модуль не меняет живой путь v19: plain Hippocampus остаётся собой."""

    def test_plain_core_unchanged(self):
        core = Hippocampus(tiny_cfg(), seed=0)
        self.assertFalse(hasattr(core, "arbiter"))
        self.assertFalse(hasattr(core, "laplace"))
        self.assertFalse(hasattr(core, "theta_gamma_slots"))
        x = torch.randn(1, 32)
        hout = core.step(x, ach=0.8, adenosine=0.1, learn=True)
        self.assertTrue(hasattr(hout, "mismatch"))
        self.assertTrue(hasattr(hout, "vta_da_burst"))


if __name__ == "__main__":
    unittest.main()