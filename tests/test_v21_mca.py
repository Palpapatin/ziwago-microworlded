"""Этап D: MCA — пер-синаптическая восприимчивость m in [0,1].

Blueprint §3.2 / MATH-MCA-001 с исправлением границ: m экспоненциально релаксирует
к bounded activity-target (не интегрирует сырой (w-w0)^2). Порог theta0*(1+beta_m*m)
требует theta0>0. Непрерывная релаксация нормирована по dt; спайковый импульс
добавляется один раз и зажимается, без dt-масштаба. Живой путь без backward.
"""
import math
import unittest

import torch
import torch.nn.functional as F

from ziwago.cognition.language.readout import DendriticPlateauMatrixReadout
from ziwago.learning.mca import MetaplasticSusceptibility


class MCAMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_activity_target_is_bounded_and_monotonic(self):
        mca = MetaplasticSusceptibility(half_sat=1.0)
        sq = torch.tensor([0.0, 0.1, 1.0, 10.0, 1e6])
        target = mca.activity_target(sq)
        self.assertTrue(torch.all(target >= 0.0))
        self.assertTrue(torch.all(target < 1.0))
        self.assertAlmostEqual(float(target[0]), 0.0, places=7)
        self.assertTrue(torch.all(target[1:] >= target[:-1]))
        self.assertGreater(float(target[-1]), 0.999)

    def test_threshold_requires_positive_theta0_and_scales_with_m(self):
        with self.assertRaises(ValueError):
            MetaplasticSusceptibility(theta0=0.0)
        mca = MetaplasticSusceptibility(theta0=0.1, beta_m=4.0)
        m = torch.tensor([0.0, 0.5, 1.0])
        theta = mca.threshold(m)
        torch.testing.assert_close(theta, torch.tensor([0.1, 0.3, 0.5]), rtol=1e-6, atol=1e-7)
        self.assertGreater(float(theta[2]), float(theta[0]))

    def test_observe_relaxation_is_dt_normalized_and_bounded(self):
        w = torch.tensor([[2.0, 0.0]])
        d_rows = torch.tensor([1.0])
        error = torch.tensor([0.5, 0.5])
        expected_target = 4.0 / 5.0
        mca1 = MetaplasticSusceptibility(tau_m=1.0, half_sat=1.0)
        mca1.observe(w, d_rows, error, [0], 0, dt=1.0, top_columns=2)
        self.assertAlmostEqual(mca1.m[(0, 0)], expected_target * (1 - math.exp(-1.0)), places=6)
        mca10 = MetaplasticSusceptibility(tau_m=1.0, half_sat=1.0)
        mca10.observe(w, d_rows, error, [0], 0, dt=10.0, top_columns=2)
        self.assertGreater(mca10.m[(0, 0)], mca1.m[(0, 0)])
        self.assertAlmostEqual(mca10.m[(0, 0)], expected_target * (1 - math.exp(-10.0)), places=6)
        for _ in range(50):
            mca1.observe(w, d_rows, error, [0], 0, dt=1.0, top_columns=2)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in mca1.m.values()))
        self.assertAlmostEqual(mca1.m[(0, 0)], expected_target, places=4)

    def test_spike_kick_added_once_and_clamped(self):
        mca = MetaplasticSusceptibility()
        mca.spike_kick([(0, 0)], 0.3)
        self.assertAlmostEqual(mca.m[(0, 0)], 0.3, places=7)
        mca.spike_kick([(0, 0)], 0.3)
        self.assertAlmostEqual(mca.m[(0, 0)], 0.6, places=7)
        mca.spike_kick([(0, 0)], 5.0)
        self.assertAlmostEqual(mca.m[(0, 0)], 1.0, places=7)
        mca.spike_kick([(0, 0)], -5.0)
        self.assertAlmostEqual(mca.m[(0, 0)], 0.0, places=7)

    def test_budget_eviction_is_reported_and_bounded(self):
        mca = MetaplasticSusceptibility(max_entries=2, tau_m=1.0, half_sat=1.0)
        w = torch.ones(1, 4)
        d_rows = torch.tensor([1.0])
        mca.observe(w, d_rows, torch.tensor([10.0, 9.0, 0.1, 0.1]), [0], 0, dt=1.0, top_columns=2)
        self.assertEqual(len(mca.m), 2)
        stats = mca.observe(w, d_rows, torch.tensor([0.1, 0.1, 10.0, 9.0]), [0], 2, dt=1.0, top_columns=2)
        self.assertLessEqual(len(mca.m), 2)
        self.assertGreater(stats["evicted"], 0)
        self.assertGreater(mca.evicted_entries, 0)

    def test_row_susceptibility_decreases_with_m(self):
        mca = MetaplasticSusceptibility(beta_m=4.0)
        mca.m = {(0, 0): 0.5, (0, 1): 0.9, (1, 0): 0.0}
        gain = mca.row_susceptibility([0, 1, 2], n_active=2)
        self.assertAlmostEqual(float(gain[0]), 1.0 / (1.0 + 4.0 * 0.9), places=6)
        self.assertAlmostEqual(float(gain[1]), 1.0, places=6)
        self.assertAlmostEqual(float(gain[2]), 1.0, places=6)
        self.assertLess(float(gain[0]), float(gain[1]))

    def test_export_load_round_trip_and_budget_guard(self):
        mca = MetaplasticSusceptibility(max_entries=8, tau_m=2.0, half_sat=1.0)
        mca.observe(torch.randn(2, 3), torch.tensor([0.7, -0.4]), torch.tensor([0.6, -0.3, 0.1]),
                    [0, 1], 0, dt=1.0, top_columns=3)
        state = mca.export_state_dict()
        restored = MetaplasticSusceptibility(max_entries=8)
        restored.load_state_dict(state)
        self.assertEqual(restored.m, mca.m)
        self.assertEqual(restored.telemetry(), mca.telemetry())
        bad = dict(state)
        bad["max_entries"] = 1
        with self.assertRaises(ValueError):
            MetaplasticSusceptibility(max_entries=8).load_state_dict(bad)


class MCAReadoutIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _readout(self):
        ro = DendriticPlateauMatrixReadout(d_vec=8, d_dend=16, max_pos=4, seed=3)
        for token in range(8):
            ro._ensure_token_col(token)
        gen = torch.Generator().manual_seed(5)
        ro.w_dend_out[:, :8] = torch.randn(16, 8, generator=gen)
        return ro

    def _steps(self):
        gen = torch.Generator().manual_seed(9)
        q = F.normalize(torch.randn(8, generator=gen), dim=0)
        out = []
        for target in (0, 3, 5, 1, 6):
            d = torch.zeros(16)
            d[[0, 1, 2, 7]] = torch.rand(4, generator=gen) + 0.3
            out.append((d, q, target))
        return out

    def test_default_on_builds_susceptibility_and_explicit_opt_out_is_neutral(self):
        """13.11: MCA включён по умолчанию; opt-out остался явным."""
        ro = self._readout()
        for d, q, target in self._steps():
            stats = ro._learn_readout_token(d, q, target, f_mod=1.0)
        self.assertGreater(stats["mca_entries_selected"], 0)
        self.assertIsNotNone(ro._mca)

        ro_off = self._readout()
        ro_off.mca_enabled = False
        for d, q, target in self._steps():
            stats_off = ro_off._learn_readout_token(d, q, target, f_mod=1.0)
        self.assertEqual(stats_off["mca_entries_selected"], 0)
        self.assertEqual(stats_off["mca_min_row_susceptibility"], 1.0)
        self.assertIsNone(ro_off._mca)

    def test_enabled_mca_reduces_drift_from_initial_weights(self):
        ro_off = self._readout()
        ro_on = self._readout()
        ro_on.mca_enabled = True
        ro_on.mca_beta_m = 4.0
        ro_on.mca_tau_m = 1.0
        ro_on.mca_dt = 1.0
        ro_on.mca_half_sat = 1.0
        ro_on.mca_top_columns = 8
        initial = ro_off.w_dend_out.clone()
        saw_signal = False
        for d, q, target in self._steps():
            ro_off._learn_readout_token(d, q, target, f_mod=1.0)
            stats = ro_on._learn_readout_token(d, q, target, f_mod=1.0)
            if stats["mca_entries_selected"] > 0:
                saw_signal = True
                self.assertLess(stats["mca_min_row_susceptibility"], 1.0)
        self.assertTrue(saw_signal)
        drift_off = float((ro_off.w_dend_out - initial).norm())
        drift_on = float((ro_on.w_dend_out - initial).norm())
        self.assertLess(drift_on, drift_off)
        self.assertGreater(ro_on._mca.telemetry()["entries"], 0)


if __name__ == "__main__":
    unittest.main()