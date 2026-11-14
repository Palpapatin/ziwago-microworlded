"""Этап D: локальный Fisher/anchor без global backprop.

Математическая ссылка (независимый маленький autograd-тест, не живой контур):
для локального категориального скорера z_v = a * sum_r d_r W[r,v] и
cross-entropy по цели y градиент dL/dW[r,v] = a*d_r*(p_v - 1[y=v]), то есть
score_gradient = a*d_r*(1[y=v]-p_v) = -W.grad. Условный Fisher модели
E_y~p[g^2] = (a*d_r)^2 * p_v*(1-p_v). Живой язык surrogate/backprop не использует.
"""
import unittest

import torch
import torch.nn.functional as F

from ziwago.cognition.language.readout import DendriticPlateauMatrixReadout
from ziwago.learning.local_fisher import SparseFisherAnchor


class FisherMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_score_gradient_matches_reference_autograd(self):
        torch.manual_seed(0)
        d = torch.zeros(6)
        d[[0, 2, 4]] = torch.tensor([0.7, -0.4, 1.3])
        w = torch.randn(6, 5, requires_grad=True)
        a = 12.0
        target = 3
        logits = a * (d @ w)
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target]))
        loss.backward()
        error = -torch.softmax(logits.detach(), dim=0)
        error[target] += 1.0
        sg = SparseFisherAnchor.score_gradient(d, error, a)
        torch.testing.assert_close(sg, -w.grad, rtol=1e-5, atol=1e-6)

    def test_observe_decay_zero_stores_squared_gradient(self):
        anchor = SparseFisherAnchor(max_entries=100, decay=0.0)
        d_rows = torch.tensor([0.7, -0.4])
        error = torch.tensor([0.6, -0.3, 0.1])
        a = 12.0
        w = torch.randn(2, 3)
        stats = anchor.observe(w, d_rows, error, a, [0, 1], target_col=0, top_columns=3)
        self.assertEqual(stats["selected"], 6)
        expected = SparseFisherAnchor.score_gradient(d_rows, error, a).square()
        self.assertEqual(len(anchor.fisher), 6)
        for (r, c), value in anchor.fisher.items():
            self.assertAlmostEqual(value, float(expected[r, c]), places=5)
        for (r, c), value in anchor.anchor.items():
            self.assertAlmostEqual(value, float(w[r, c]), places=6)

    def test_conditional_fisher_equals_expected_squared_gradient(self):
        d_rows = torch.tensor([0.7, -0.4, 1.1])
        probs = torch.softmax(torch.randn(5), dim=0)
        a = 12.0
        cf = SparseFisherAnchor.conditional_fisher(d_rows, probs, a)
        drive = (a * d_rows).square().unsqueeze(1)
        expected = drive * probs.unsqueeze(0) * (1.0 - probs.unsqueeze(0))
        torch.testing.assert_close(cf, expected, rtol=1e-6, atol=1e-7)
        # E_y~p[(a*d*(1[y=v]-p_v))^2] = (a*d)^2 * p_v*(1-p_v): проверяем Монте-Карло.
        g = torch.Generator().manual_seed(1)
        samples = torch.multinomial(probs, 20000, replacement=True, generator=g)
        acc = torch.zeros(3, 5)
        for y in samples.tolist():
            err = -probs.clone()
            err[y] += 1.0
            acc += SparseFisherAnchor.score_gradient(d_rows, err, a).square()
        torch.testing.assert_close(acc / 20000, cf, rtol=0.08, atol=0.05)

    def test_proximal_pull_moves_weight_toward_anchor(self):
        anchor = SparseFisherAnchor(max_entries=4, decay=0.0)
        anchor.fisher[(0, 0)] = 1.0
        anchor.anchor[(0, 0)] = 0.0
        w = torch.tensor([[2.0]])
        pull = anchor.proximal_pull(w, eta=1.0, lambda_reg=1.0)
        self.assertEqual(pull["applied"], 1)
        self.assertAlmostEqual(float(w[0, 0]), 1.0, places=6)
        self.assertLess(abs(float(w[0, 0]) - 0.0), abs(2.0 - 0.0))
        self.assertAlmostEqual(pull["max_displacement"], 1.0, places=6)
        zero = anchor.proximal_pull(w, eta=0.0, lambda_reg=1.0)
        self.assertEqual(zero["applied"], 0)

    def test_budget_eviction_is_reported_and_bounded(self):
        anchor = SparseFisherAnchor(max_entries=2, decay=0.9)
        d_rows = torch.tensor([1.0])
        w = torch.zeros(1, 4)
        anchor.observe(w, d_rows, torch.tensor([10.0, 9.0, 0.1, 0.1]), 12.0, [0], 0, top_columns=2)
        self.assertEqual(len(anchor.fisher), 2)
        stats = anchor.observe(w, d_rows, torch.tensor([0.1, 0.1, 10.0, 9.0]), 12.0, [0], 2, top_columns=2)
        self.assertLessEqual(len(anchor.fisher), 2)
        self.assertGreater(stats["evicted"], 0)
        self.assertGreater(anchor.evicted_entries, 0)
        self.assertEqual(set(anchor.fisher), set(anchor.anchor))

    def test_export_load_round_trip_and_budget_guard(self):
        anchor = SparseFisherAnchor(max_entries=8, decay=0.5)
        d_rows = torch.tensor([0.7, -0.4])
        anchor.observe(torch.randn(2, 3), d_rows, torch.tensor([0.6, -0.3, 0.1]), 12.0, [0, 1], 0, top_columns=3)
        state = anchor.export_state_dict()
        restored = SparseFisherAnchor(max_entries=8, decay=0.5)
        restored.load_state_dict(state)
        self.assertEqual(restored.fisher, anchor.fisher)
        self.assertEqual(restored.anchor, anchor.anchor)
        self.assertEqual(restored.telemetry(), anchor.telemetry())
        bad = dict(state)
        bad["max_entries"] = 1
        with self.assertRaises(ValueError):
            SparseFisherAnchor(max_entries=8).load_state_dict(bad)


class FisherReadoutIntegrationTests(unittest.TestCase):
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

    def test_default_on_builds_anchor_and_explicit_opt_out_is_neutral(self):
        """13.11: Фишер включён по умолчанию; opt-out остался явным."""
        ro = self._readout()
        for d, q, target in self._steps():
            stats = ro._learn_readout_token(d, q, target, f_mod=1.0)
        self.assertGreater(stats["fisher_entries_selected"], 0)
        self.assertIsNotNone(ro._fisher_anchor)

        ro_off = self._readout()
        ro_off.fisher_anchor_enabled = False
        for d, q, target in self._steps():
            stats_off = ro_off._learn_readout_token(d, q, target, f_mod=1.0)
        self.assertEqual(stats_off["fisher_entries_selected"], 0)
        self.assertEqual(stats_off["fisher_proximal_applied"], 0)
        self.assertIsNone(ro_off._fisher_anchor)

    def test_enabled_fisher_reduces_drift_from_initial_weights(self):
        ro_off = self._readout()
        ro_on = self._readout()
        ro_on.fisher_anchor_enabled = True
        ro_on.fisher_lambda = 1.0
        ro_on.fisher_eta = 1.0
        ro_on.fisher_decay = 0.5
        ro_on.fisher_top_columns = 8
        initial = ro_off.w_dend_out.clone()
        saw_signal = False
        for d, q, target in self._steps():
            ro_off._learn_readout_token(d, q, target, f_mod=1.0)
            stats = ro_on._learn_readout_token(d, q, target, f_mod=1.0)
            if stats["fisher_entries_selected"] > 0:
                saw_signal = True
                self.assertGreater(stats["fisher_proximal_applied"], 0)
        self.assertTrue(saw_signal)
        drift_off = float((ro_off.w_dend_out - initial).norm())
        drift_on = float((ro_on.w_dend_out - initial).norm())
        self.assertLess(drift_on, drift_off)
        self.assertGreater(ro_on._fisher_anchor.telemetry()["entries"], 0)


if __name__ == "__main__":
    unittest.main()