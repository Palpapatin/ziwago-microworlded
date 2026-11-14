"""Контракты ADMC (CONSC-ADMC-001, этап E часть 4): learning-progress/curiosity.

Проверяют прогресс обучения ΔE, решение noisy-TV, метаболический гейт АТФ,
точную формулу цели J, bounded leaky-интеграл аллостатического груза и bounded
сигналы регуляции когнитивной нагрузки. Ветвь ACh уже в physiology.py и здесь
не дублируется. Reference-autograd не используется; всё локально.
"""
from __future__ import annotations

import math
import unittest

from ziwago.organism.admc import AllostaticCuriosityController


class ADMCMathTests(unittest.TestCase):
    def test_first_observation_zero_progress(self):
        c = AllostaticCuriosityController(tau_ref=100.0)
        self.assertEqual(c.observe_error(10.0, dt=1.0), 0.0)
        self.assertAlmostEqual(c.error_ref, 10.0, places=6)
        with self.assertRaises(ValueError):
            c.observe_error(1.0, dt=0.0)

    def test_learning_progress_positive_on_decreasing_error(self):
        c = AllostaticCuriosityController(tau_ref=100.0)
        c.observe_error(10.0, dt=1.0)
        p = c.observe_error(9.0, dt=1.0)
        self.assertGreater(p, 0.0)
        p2 = c.observe_error(8.0, dt=1.0)
        self.assertGreater(p2, p)   # sustained learning → growing progress

    def test_noisy_tv_flat_error_zero_progress_negative_objective(self):
        c = AllostaticCuriosityController(tau_ref=100.0, lambda_atp=0.5)
        for _ in range(50):
            p = c.observe_error(10.0, dt=1.0)   # flat high error = noise
        self.assertAlmostEqual(p, 0.0, delta=1e-6)
        j = c.objective_value(p, atp_cost=1.0)
        self.assertLess(j, 0.0)                 # J = 0 - λ·cost < 0 → avoid noise
        drive = c.curiosity_drive(p, 1.0, atp_level=1.0)
        self.assertLess(drive, 0.0)

    def test_objective_exact_formula(self):
        c = AllostaticCuriosityController(lambda_atp=0.5)
        self.assertAlmostEqual(c.objective_value(2.0, 1.0), 2.0 - 0.5 * 1.0, places=6)
        # negative cost is clamped to zero (no reward for free energy)
        self.assertAlmostEqual(c.objective_value(2.0, -5.0), 2.0, places=6)

    def test_exploration_gate_bounded_monotonic(self):
        c = AllostaticCuriosityController(atp_safe=0.5, atp_width=0.1)
        g_low = c.exploration_gate(0.0)
        g_mid = c.exploration_gate(0.5)
        g_high = c.exploration_gate(1.0)
        self.assertAlmostEqual(g_mid, 0.5, places=6)
        self.assertLess(g_low, g_mid)
        self.assertLess(g_mid, g_high)
        for g in (g_low, g_mid, g_high):
            self.assertGreaterEqual(g, 0.0)
            self.assertLessEqual(g, 1.0)

    def test_curiosity_drive_metabolic_conservation(self):
        c = AllostaticCuriosityController(atp_safe=0.5, atp_width=0.1,
                                          lambda_atp=0.1)
        progress = 5.0   # strong learning progress
        drive_high = c.curiosity_drive(progress, 0.1, atp_level=1.0)
        drive_low = c.curiosity_drive(progress, 0.1, atp_level=0.0)
        self.assertGreater(drive_high, 0.0)
        # Conservation = strong relative suppression (gate sigmoid(-5) ≈ 0.0067),
        # not absolute zero: a large progress still leaks a tiny gated drive.
        self.assertGreaterEqual(drive_low, 0.0)
        self.assertLess(drive_low, 0.01 * drive_high)
        self.assertLess(drive_low, drive_high)

    def test_curiosity_gain_opt_in_zero(self):
        c = AllostaticCuriosityController(curiosity_gain=0.0)
        self.assertEqual(c.curiosity_drive(5.0, 0.1, atp_level=1.0), 0.0)

    def test_allostatic_leaky_integral_exact(self):
        c = AllostaticCuriosityController(tau_allo=1000.0, w_ado=1.0,
                                          w_pat=1.0, w_err=1.0, allo_max=1000.0)
        l1 = c.accumulate_allostatic(ado=1.0, pat_flag=False, error=0.0, dt=1.0)
        self.assertAlmostEqual(l1, 1.0, delta=1e-6)
        l2 = c.accumulate_allostatic(ado=1.0, pat_flag=False, error=0.0, dt=1.0)
        expected = 1.0 * math.exp(-1.0 / 1000.0) + 1.0
        self.assertAlmostEqual(l2, expected, delta=1e-6)
        with self.assertRaises(ValueError):
            c.accumulate_allostatic(1.0, False, 0.0, dt=0.0)

    def test_allostatic_bounded_at_max(self):
        c = AllostaticCuriosityController(allo_max=5.0, tau_allo=1000.0)
        for _ in range(100):
            load = c.accumulate_allostatic(ado=1000.0, pat_flag=True,
                                           error=1000.0, dt=1.0)
        self.assertLessEqual(load, 5.0 + 1e-9)
        self.assertGreaterEqual(load, 0.0)

    def test_allostatic_decays_without_source(self):
        c = AllostaticCuriosityController(tau_allo=1000.0, allo_max=100.0)
        c.allo_load = 10.0
        load = c.accumulate_allostatic(ado=0.0, pat_flag=False, error=0.0, dt=100.0)
        self.assertAlmostEqual(load, 10.0 * math.exp(-100.0 / 1000.0), delta=1e-6)
        self.assertLess(load, 10.0)
        self.assertGreater(load, 0.0)


class ADMCRegulationTests(unittest.TestCase):
    def test_regulation_neutral_below_threshold(self):
        c = AllostaticCuriosityController(allo_threshold=1.0)
        reg = c.regulation()   # allo_load = 0
        self.assertEqual(reg["strain"], 0.0)
        self.assertAlmostEqual(reg["graph_depth_factor"], 1.0, places=6)
        self.assertAlmostEqual(reg["confidence_threshold_add"], 0.0, places=6)
        self.assertAlmostEqual(reg["phrase_length_factor"], 1.0, places=6)
        self.assertFalse(reg["conservation_mode"])

    def test_regulation_bounded_above_threshold(self):
        c = AllostaticCuriosityController(allo_threshold=1.0, allo_width=1.0,
                                          allo_max=100.0, depth_reduce_max=0.5,
                                          conf_add_max=0.3, phrase_reduce_max=0.5)
        c.allo_load = 100.0
        reg = c.regulation()
        self.assertTrue(reg["conservation_mode"])
        self.assertGreater(reg["strain"], 0.0)
        self.assertLessEqual(reg["strain"], 1.0)   # saturates to 1.0 in float64
        self.assertGreaterEqual(reg["graph_depth_factor"], 0.5 - 1e-9)
        self.assertLessEqual(reg["graph_depth_factor"], 1.0)
        self.assertGreaterEqual(reg["confidence_threshold_add"], 0.0)
        self.assertLessEqual(reg["confidence_threshold_add"], 0.3 + 1e-9)
        self.assertGreaterEqual(reg["phrase_length_factor"], 0.5 - 1e-9)
        self.assertLessEqual(reg["phrase_length_factor"], 1.0)

    def test_step_integrates_all_signals(self):
        c = AllostaticCuriosityController()
        c.observe_error(10.0, dt=1.0)
        out = c.step(error_now=8.0, atp_cost=0.1, atp_level=1.0,
                     ado=0.5, pat_flag=False, dt=1.0)
        for key in ("progress", "objective", "exploration_gate",
                    "curiosity_drive", "allostatic_load", "strain",
                    "graph_depth_factor", "confidence_threshold_add",
                    "phrase_length_factor", "conservation_mode"):
            self.assertIn(key, out)
        self.assertGreater(out["progress"], 0.0)
        self.assertGreater(out["allostatic_load"], 0.0)

    def test_export_load_roundtrip_and_guard(self):
        c = AllostaticCuriosityController(allo_max=50.0)
        for e in (10.0, 9.0, 8.0):
            c.step(error_now=e, atp_cost=0.1, atp_level=1.0,
                   ado=0.5, pat_flag=False, dt=1.0)
        sd = c.export_state_dict()
        other = AllostaticCuriosityController(allo_max=50.0)
        other.load_state_dict(sd)
        self.assertAlmostEqual(c.error_ref, other.error_ref, places=6)
        self.assertAlmostEqual(c.allo_load, other.allo_load, places=6)
        self.assertAlmostEqual(c.last_progress, other.last_progress, places=6)
        bad = AllostaticCuriosityController(allo_max=99.0)
        with self.assertRaises(ValueError):
            bad.load_state_dict(sd)


if __name__ == "__main__":
    unittest.main()