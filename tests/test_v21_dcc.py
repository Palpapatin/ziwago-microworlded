"""Этап E: DCC — контроллер зрелости/readiness (blueprint §3.5 / ARCH-DCC-001).

Проверяются исправления карточки: eta=4m(1-m) ноль на краях без скрытого floor;
bounded E/I-score с гейтом активности (E=I=0 не зрелое равновесие); ITPC гейтится
числом событий; возрастной prior задаёт порядок, но закрытие решает независимая
readiness; гистерезис закрытия, retention, квота пластичных зон (резерв адаптации),
адресное переоткрытие с сохранением возраста, сериализация.
"""
import math
import unittest

import torch

from ziwago.cortex.dcc import DevelopmentalReadinessController


def ready_obs(n):
    return {"itpc": torch.ones(n), "pred_accuracy": torch.ones(n),
            "exc": torch.ones(n), "inh": torch.ones(n),
            "n_events": torch.full((n,), 20.0)}


def silent_obs(n):
    return {"itpc": torch.zeros(n), "pred_accuracy": torch.zeros(n),
            "exc": torch.zeros(n), "inh": torch.zeros(n),
            "n_events": torch.zeros(n)}


class DCCMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_eta_window_zero_at_edges_and_peaks_mid(self):
        dcc = DevelopmentalReadinessController(n_zones=3, eta_max=1.0)
        eta = dcc.eta_window(torch.tensor([0.0, 0.5, 1.0]))
        torch.testing.assert_close(eta, torch.tensor([0.0, 1.0, 0.0]), rtol=1e-6, atol=1e-7)

    def test_maturity_prior_orders_zones_by_tau(self):
        dcc = DevelopmentalReadinessController(n_zones=2, tau_mature=[100.0, 1000.0])
        m0 = dcc.maturity_prior(torch.zeros(2))
        self.assertTrue(torch.all(m0 == 0.0))
        dcc.tick(ready_obs(2), dt=100.0)
        self.assertGreater(float(dcc.maturity[0]), float(dcc.maturity[1]))
        self.assertAlmostEqual(float(dcc.maturity[0]), 1 - math.exp(-1.0), places=5)
        self.assertTrue(torch.all(dcc.maturity >= 0.0) and torch.all(dcc.maturity <= 1.0))

    def test_ei_score_bounded_and_silence_is_not_mature(self):
        dcc = DevelopmentalReadinessController(n_zones=4, ei_activity_floor=1.0)
        exc = torch.tensor([0.0, 1.0, 10.0, 1.0])
        inh = torch.tensor([0.0, 1.0, 1.0, 0.0])
        score = dcc.ei_score(exc, inh)
        self.assertTrue(torch.all(score >= 0.0) and torch.all(score <= 1.0))
        self.assertAlmostEqual(float(score[0]), 0.0, places=6)  # E=I=0 silence
        self.assertAlmostEqual(float(score[1]), 1.0, places=5)  # balanced active
        self.assertLess(float(score[2]), 0.2)                   # E >> I
        self.assertLess(float(score[3]), 0.2)                   # I = 0

    def test_itpc_gated_by_event_count(self):
        dcc = DevelopmentalReadinessController(n_zones=3, itpc_min_events=20)
        itpc = torch.ones(3)
        gated = dcc.itpc_gate(itpc, torch.tensor([0.0, 10.0, 20.0]))
        torch.testing.assert_close(gated, torch.tensor([0.0, 0.5, 1.0]), rtol=1e-6, atol=1e-7)

    def test_readiness_is_bounded_weighted_sum(self):
        dcc = DevelopmentalReadinessController(n_zones=2, weights=(0.4, 0.4, 0.2))
        r = dcc.readiness_score(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]),
                                torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(r, torch.tensor([1.0, 0.0]), rtol=1e-6, atol=1e-7)
        with self.assertRaises(ValueError):
            DevelopmentalReadinessController(n_zones=2, weights=(0.5, 0.5, 0.5))


class DCCDevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_learning_from_birth_state(self):
        dcc = DevelopmentalReadinessController(n_zones=4)
        self.assertEqual(float(dcc.age.sum()), 0.0)
        self.assertEqual(float(dcc.maturity.sum()), 0.0)
        self.assertEqual(float(dcc.eta.sum()), 0.0)
        self.assertEqual(int(dcc.closed.sum()), 0)
        self.assertEqual(float(dcc.effective_plasticity().sum()), 0.0)
        dcc.tick(ready_obs(4), dt=50.0)
        self.assertGreater(float(dcc.maturity.min()), 0.0)
        self.assertGreater(float(dcc.eta.min()), 0.0)

    def test_closure_forbidden_in_silence(self):
        dcc = DevelopmentalReadinessController(n_zones=4, window_w=5, min_plastic_fraction=0.0)
        for _ in range(30):
            dcc.tick(silent_obs(4), dt=50.0)
        self.assertEqual(int(dcc.closed.sum()), 0)
        self.assertEqual(int(dcc.consecutive_ready.sum()), 0)
        self.assertEqual(float(dcc.readiness.sum()), 0.0)

    def test_hysteresis_requires_stable_window(self):
        dcc = DevelopmentalReadinessController(n_zones=4, window_w=5, min_plastic_fraction=0.0)
        for _ in range(4):
            dcc.tick(ready_obs(4), dt=50.0)
        self.assertEqual(int(dcc.closed.sum()), 0)
        dcc.tick(ready_obs(4), dt=50.0)
        self.assertEqual(int(dcc.closed.sum()), 4)
        self.assertTrue(all(r == "readiness_window" for r in dcc.closure_reason))

    def test_retention_blocks_closure(self):
        dcc = DevelopmentalReadinessController(n_zones=4, window_w=3, min_plastic_fraction=0.0)
        for _ in range(10):
            dcc.tick(ready_obs(4), dt=50.0, retention_ok=False)
        self.assertEqual(int(dcc.closed.sum()), 0)

    def test_plastic_reserve_forbids_global_closure(self):
        quarter = DevelopmentalReadinessController(n_zones=4, window_w=3, min_plastic_fraction=0.25)
        for _ in range(5):
            quarter.tick(ready_obs(4), dt=50.0)
        self.assertGreaterEqual(int((~quarter.closed).sum()), 1)
        self.assertLessEqual(int(quarter.closed.sum()), 3)
        half = DevelopmentalReadinessController(n_zones=4, window_w=3, min_plastic_fraction=0.5)
        for _ in range(5):
            half.tick(ready_obs(4), dt=50.0)
        self.assertGreaterEqual(int((~half.closed).sum()), 2)

    def test_effective_plasticity_zero_when_closed(self):
        dcc = DevelopmentalReadinessController(n_zones=4, window_w=3, min_plastic_fraction=0.0)
        for _ in range(5):
            dcc.tick(ready_obs(4), dt=50.0)
        self.assertEqual(float(dcc.effective_plasticity().sum()), 0.0)

    def test_addressable_reopening_preserves_age_and_others(self):
        dcc = DevelopmentalReadinessController(n_zones=4, window_w=3, min_plastic_fraction=0.0)
        for _ in range(5):
            dcc.tick(ready_obs(4), dt=50.0)
        age_before = dcc.age.clone()
        maturity_before = dcc.maturity.clone()
        n = dcc.reopen([0, 1], reason="distribution_shift")
        self.assertEqual(n, 2)
        self.assertFalse(bool(dcc.closed[0]))
        self.assertTrue(bool(dcc.closed[2]))
        self.assertEqual(int(dcc.consecutive_ready[0]), 0)
        self.assertEqual(dcc.closure_reason[0], "reopened:distribution_shift")
        torch.testing.assert_close(dcc.age, age_before, rtol=0, atol=0)
        torch.testing.assert_close(dcc.maturity, maturity_before, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            dcc.reopen([4])

    def test_export_load_round_trip_and_shape_guard(self):
        dcc = DevelopmentalReadinessController(n_zones=4, window_w=3, tau_mature=[100.0, 200.0, 300.0, 400.0])
        for _ in range(5):
            dcc.tick(ready_obs(4), dt=50.0)
        dcc.reopen([0])
        state = dcc.export_state_dict()
        restored = DevelopmentalReadinessController(n_zones=4)
        restored.load_state_dict(state)
        torch.testing.assert_close(restored.maturity, dcc.maturity, rtol=0, atol=0)
        torch.testing.assert_close(restored.closed, dcc.closed, rtol=0, atol=0)
        self.assertEqual(restored.closure_reason, dcc.closure_reason)
        self.assertEqual(restored.telemetry(), dcc.telemetry())
        bad = dict(state)
        bad["age"] = torch.zeros(3)
        with self.assertRaises(ValueError):
            DevelopmentalReadinessController(n_zones=4).load_state_dict(bad)


if __name__ == "__main__":
    unittest.main()