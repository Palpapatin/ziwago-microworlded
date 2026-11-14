"""Юнит-тесты арбитража «префикс ↔ эпизод».

Арбитраж — часть ядра, поэтому он должен быть зафиксирован тестом: иначе
порог или правило срабатывания можно сдвинуть вместе с результатом.
Тесты чистые: модель и чекпоинты не нужны.
"""
from __future__ import annotations

import unittest

import torch

from train_ziwago_v14_pure_synaptic_dialogue import ZiwagoPureSynapticDialogueOrganismV14 as Org


class TestConflictSignal(unittest.TestCase):
    def _signal(self, logits):
        return Org._language_conflict_signal(None, logits)

    def test_peaked_distribution_is_low_conflict(self):
        logits = torch.full((512,), -8.0)
        logits[17] = 8.0
        self.assertLess(self._signal(logits), 0.05)

    def test_flat_distribution_is_high_conflict(self):
        logits = torch.zeros(512)
        self.assertGreater(self._signal(logits), 0.95)

    def test_intermediate_distribution_is_between(self):
        logits = torch.full((512,), -2.0)
        logits[3] = 1.0
        value = self._signal(logits)
        self.assertGreater(value, 0.0)
        self.assertLess(value, 1.0)

    def test_empty_logits_is_maximal_conflict(self):
        self.assertEqual(self._signal(torch.zeros(0)), 1.0)

    def test_signal_is_bounded(self):
        for scale in (0.0, 1.0, 50.0):
            logits = torch.randn(256) * scale
            value = self._signal(logits)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


class TestConflictThreshold(unittest.TestCase):
    def _threshold(self, ach):
        # Состояние химии лежит в chemistry.s (ChemState) — так и в коде.
        stub = type("S", (), {
            "chemistry": type("C", (), {"s": type("St", (), {"ach": ach})()})(),
            "language_conflict_threshold_base": 0.035,
            "language_conflict_ach_gain": 0.03,
            "language_conflict_ach_full": 1.2,
            "neuromodulation_enabled": False,
        })()
        return Org._language_conflict_threshold(stub)

    def test_threshold_increases_with_ach(self):
        self.assertLess(self._threshold(0.0), self._threshold(0.6))
        self.assertLess(self._threshold(0.6), self._threshold(1.2))

    def test_threshold_stays_inside_the_measured_separation(self):
        # зазор измерен на 16 условиях: on<=0.0, off>=0.0358.
        # порог обязан лежать ВНУТРИ [0.035, 0.065] при любом состоянии.
        for ach in (0.0, 0.3, 0.6, 0.9, 1.2, 2.0):
            threshold = self._threshold(ach)
            self.assertGreaterEqual(threshold, 0.035)
            self.assertLessEqual(threshold, 0.065)

    def test_ach_is_clamped(self):
        self.assertEqual(self._threshold(-5.0), self._threshold(0.0))
        self.assertEqual(self._threshold(99.0), self._threshold(1.2))


class TestArbitrationDefaults(unittest.TestCase):
    def test_arbitration_is_disabled_by_default(self):
        self.assertFalse(Org.language_conflict_arbitration)

    def test_defaults_match_the_measured_calibration(self):
        self.assertEqual(Org.language_conflict_abstain_steps, 3)
        self.assertEqual(Org.language_conflict_threshold_base, 0.035)
        self.assertEqual(Org.language_conflict_ach_gain, 0.03)


if __name__ == "__main__":
    unittest.main()
