"""Контракты модуля нарративного Я NSM (ARCH-NSM-001, этап E часть 3).

Проверяют математику автобиографии/настроения и границы additive prior:
нейтральное пустое состояние, bounded M(t), целочисленный косинус SBC-VSA без
FFT, mood-congruent retrieval, amygdala-boost, bipolar-binding self-vector,
поэлементно ограниченный prior (NSM-4: без hard token clamping), кольцевой
буфер и сериализацию..Reference-autograd не используется; всё локально.
"""
from __future__ import annotations

import math
import unittest

import torch

from ziwago.memory.nsm import NarrativeSelfModule


def _ctx(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.where(torch.randn(d, generator=g) >= 0.0, 1.0, -1.0)


class NSMMathTests(unittest.TestCase):
    def test_empty_state_is_neutral(self):
        nsm = NarrativeSelfModule(d_vec=16, max_episodes=8)
        self.assertEqual(nsm.mood(100.0), 0.0)
        self.assertTrue(torch.all(nsm.self_vector(100.0) == 0.0))
        self.assertTrue(torch.all(nsm.additive_prior(100.0) == 0.0))
        idx, score = nsm.retrieve(_ctx(16, 1), 100.0)
        self.assertEqual(idx.numel(), 0)
        self.assertEqual(score.numel(), 0)

    def test_episode_validation_and_clamping(self):
        nsm = NarrativeSelfModule(d_vec=8, max_episodes=4)
        with self.assertRaises(ValueError):
            nsm.record_episode(torch.zeros(7), 0.0, 0.0, 0.5)   # wrong dim
        with self.assertRaises(ValueError):
            nsm.record_episode(torch.zeros(8), -1.0, 0.0, 0.5)  # negative time
        nsm.record_episode(torch.zeros(8), 0.0, 5.0, 9.0)       # clamp v, s
        self.assertAlmostEqual(float(nsm.v[0]), 1.0, places=6)
        self.assertAlmostEqual(float(nsm.s[0]), 1.0, places=6)
        nsm.record_episode(torch.zeros(8), 1.0, -5.0, -9.0)
        self.assertAlmostEqual(float(nsm.v[1]), -1.0, places=6)
        self.assertAlmostEqual(float(nsm.s[1]), 0.0, places=6)

    def test_mood_bounded_and_exact_two_episodes(self):
        nsm = NarrativeSelfModule(d_vec=8, max_episodes=8, tau_m=3600.0)
        nsm.record_episode(_ctx(8, 1), 0.0, 1.0, 0.5)
        nsm.record_episode(_ctx(8, 2), 100.0, -1.0, 0.5)
        t = 100.0
        d0 = math.exp(-100.0 / 3600.0)
        d1 = math.exp(0.0)
        expected = (1.0 * d0 + (-1.0) * d1) / (d0 + d1 + nsm.eps)
        self.assertAlmostEqual(nsm.mood(t), expected, delta=1e-4)
        self.assertGreaterEqual(nsm.mood(t), -1.0)
        self.assertLessEqual(nsm.mood(t), 1.0)

    def test_mood_recency_weighting(self):
        nsm = NarrativeSelfModule(d_vec=8, max_episodes=8, tau_m=10.0)
        nsm.record_episode(_ctx(8, 1), 0.0, -1.0, 0.5)     # old negative
        nsm.record_episode(_ctx(8, 2), 1000.0, 1.0, 0.5)   # recent positive
        m = nsm.mood(1000.0)
        self.assertGreater(m, 0.9)   # recent positive dominates after decay
        self.assertLessEqual(m, 1.0)

    def test_mood_extreme_stays_bounded(self):
        nsm = NarrativeSelfModule(d_vec=8, max_episodes=64, tau_m=3600.0)
        for k in range(64):
            nsm.record_episode(_ctx(8, k), float(k), 1.0, 1.0)
        self.assertLessEqual(nsm.mood(100.0), 1.0)
        for k in range(64):
            nsm.record_episode(_ctx(8, 100 + k), float(200 + k), -1.0, 1.0)
        self.assertGreaterEqual(nsm.mood(300.0), -1.0)

    def test_retrieval_integer_cosine_exact_score(self):
        nsm = NarrativeSelfModule(d_vec=16, max_episodes=8, alpha=0.5,
                                  tau_ep=86400.0, tau_m=3600.0)
        h = _ctx(16, 7)
        nsm.record_episode(h, 0.0, 0.5, 0.5)
        idx, score = nsm.retrieve(h, 0.0, top_k=1)   # q == h → cos = 1
        self.assertEqual(int(idx[0]), 0)
        m = nsm.mood(0.0)                            # single episode → M ≈ v
        expected = 1.0 * (1.0 + 0.5 * 0.5) * 1.0 * (1.0 + m * 0.5)
        self.assertAlmostEqual(float(score[0]), expected, delta=1e-4)

    def test_retrieval_orthogonal_query_zero_cosine(self):
        nsm = NarrativeSelfModule(d_vec=16, max_episodes=8)
        h = torch.ones(16)
        nsm.record_episode(h, 0.0, 0.5, 0.5)
        q = torch.ones(16)
        q[:8] = -1.0    # orthogonal bipolar → cos = 0
        idx, score = nsm.retrieve(q, 0.0, top_k=1)
        self.assertAlmostEqual(float(score[0]), 0.0, delta=1e-6)

    def test_retrieval_mood_congruence(self):
        nsm = NarrativeSelfModule(d_vec=16, max_episodes=8, tau_m=3600.0)
        h = _ctx(16, 3)
        nsm.record_episode(h, 0.0, 1.0, 0.5)    # positive episode
        nsm.record_episode(h, 0.0, -1.0, 0.5)   # negative episode, same context
        # Force positive mood via a recent strong positive episode.
        nsm.record_episode(_ctx(16, 9), 100.0, 1.0, 1.0)
        m = nsm.mood(100.0)
        self.assertGreater(m, 0.0)
        idx, score = nsm.retrieve(h, 100.0, top_k=2)
        # Among the two same-context episodes, positive valence ranks first.
        vals = {int(i): float(nsm.v[int(i)]) for i in idx}
        self.assertEqual(vals[int(idx[0])], 1.0)

    def test_retrieval_amygdala_emotional_boost(self):
        nsm = NarrativeSelfModule(d_vec=16, max_episodes=8, alpha=0.5,
                                  tau_m=1.0)
        h = _ctx(16, 5)
        nsm.record_episode(h, 0.0, 0.0, 0.5)    # neutral
        nsm.record_episode(h, 0.0, 1.0, 0.5)    # emotional, same context/time
        # tau_m tiny → mood ≈ 0 at large t, isolating the amygdala factor.
        idx, score = nsm.retrieve(h, 1e6, top_k=2)
        self.assertEqual(int(idx[0]), 1)        # emotional episode wins
        self.assertGreater(float(score[0]), float(score[1]))

    def test_self_vector_normalized_bipolar_binding(self):
        nsm = NarrativeSelfModule(d_vec=16, max_episodes=8)
        h = _ctx(16, 11)
        nsm.record_episode(h, 0.0, 0.5, 0.5)
        vs = nsm.self_vector(0.0)
        self.assertAlmostEqual(float(vs.norm()), 1.0, delta=1e-5)
        expected = nsm._bipolar(h) * nsm.role_key
        expected = expected / expected.norm()
        self.assertTrue(torch.allclose(vs, expected, atol=1e-5))

    def test_additive_prior_bounded_and_opt_in(self):
        nsm = NarrativeSelfModule(d_vec=16, max_episodes=64, prior_bound=0.5)
        for k in range(64):
            nsm.record_episode(_ctx(16, k), float(k), 1.0, 1.0)
        prior = nsm.additive_prior(100.0, gain=10.0)   # huge gain → must clamp
        self.assertTrue(torch.all(prior <= 0.5 + 1e-6))
        self.assertTrue(torch.all(prior >= -0.5 - 1e-6))
        self.assertTrue(torch.all(nsm.additive_prior(100.0, gain=0.0) == 0.0))

    def test_ring_buffer_eviction_keeps_recent(self):
        nsm = NarrativeSelfModule(d_vec=8, max_episodes=4, tau_m=3600.0)
        for k in range(6):
            val = 1.0 if k >= 2 else -1.0
            nsm.record_episode(_ctx(8, k), float(k), val, 0.5)
        self.assertEqual(nsm.count, 4)
        # Oldest two (negative) evicted; remaining valences all positive.
        active = nsm._active()
        self.assertTrue(torch.all(nsm.v[active] == 1.0))
        self.assertGreater(nsm.mood(10.0), 0.99)


class NSMIntegrationTests(unittest.TestCase):
    def test_prior_is_soft_additive_not_token_clamping(self):
        # prior_bound по умолчанию 0.5 << logit_scale 12.0 живого ридаута:
        # prior не способен форсировать токен (NSM-4), только мягко сместить bias.
        nsm = NarrativeSelfModule(d_vec=32, max_episodes=32)
        self.assertLess(nsm.prior_bound, 12.0)
        for k in range(32):
            nsm.record_episode(_ctx(32, k), float(k), 1.0, 1.0)
        prior = nsm.additive_prior(50.0, gain=1.0)
        self.assertLessEqual(float(prior.abs().max()), nsm.prior_bound + 1e-6)

    def test_export_load_roundtrip_and_shape_guard(self):
        nsm = NarrativeSelfModule(d_vec=16, max_episodes=8, seed=42)
        for k in range(5):
            nsm.record_episode(_ctx(16, k), float(k), 0.3 * k - 0.6, 0.5)
        sd = nsm.export_state_dict()
        other = NarrativeSelfModule(d_vec=16, max_episodes=8, seed=99)
        other.load_state_dict(sd)
        self.assertAlmostEqual(nsm.mood(10.0), other.mood(10.0), places=6)
        self.assertTrue(torch.allclose(nsm.self_vector(10.0),
                                       other.self_vector(10.0), atol=1e-6))
        self.assertTrue(torch.allclose(nsm.additive_prior(10.0),
                                       other.additive_prior(10.0), atol=1e-6))
        bad = NarrativeSelfModule(d_vec=8, max_episodes=8)
        with self.assertRaises(ValueError):
            bad.load_state_dict(sd)


if __name__ == "__main__":
    unittest.main()