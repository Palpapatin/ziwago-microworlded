"""Инфраструктура обучения v20+ (ARCHITECTURE_MAPPING §8, П-2; opt-in).

Контракт, который здесь фиксируется:
  * У-1 surrogate: forward = ступенька Хевисайда, backward = калиброванная
    псевдопроизводная (FastSigmoid/Arctan); градиент течёт через спайк;
  * У-3 EWC: штраф растёт при отклонении от снапшота, rebaseline после сна
    обнуляет его, Fisher нормируется на max;

  * opt-in: пакет learning не импортирует nw8/nw9 и не трогает живой ридеаут v19.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ziwago.learning.ewc import EWCRegularizer
from ziwago.learning.surrogate import hebbian_step, spike_fn


class SurrogateTests(unittest.TestCase):
    def test_forward_is_heaviside(self):
        x = torch.tensor([-2.0, -0.01, 0.0, 0.01, 2.0])
        for mode in ("fast_sigmoid", "arctan"):
            s = spike_fn(x, mode, allow_experimental=True)
            self.assertTrue(torch.equal(s, torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])))

    def test_fast_sigmoid_backward_analytic(self):
        k = 10.0
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
        spike_fn(x, "fast_sigmoid", k, allow_experimental=True).sum().backward()
        expected = 1.0 / (1.0 + k * x.abs()) ** 2
        self.assertTrue(torch.allclose(x.grad, expected, atol=1e-6))

    def test_arctan_backward_analytic(self):
        a = 8.0
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
        spike_fn(x, "arctan", a, allow_experimental=True).sum().backward()
        f = (math.pi / 2.0) * a * x
        expected = (a / 2.0) / (1.0 + f * f)
        self.assertTrue(torch.allclose(x.grad, expected, atol=1e-6))

    def test_hebbian_step_monotonic_bounded(self):
        x = torch.linspace(-3, 3, 50)
        for mode in ("fast_sigmoid", "arctan"):
            y = hebbian_step(x, mode, k=5.0, allow_experimental=True)
            self.assertTrue(bool((y >= 0).all() and (y <= 1).all()))
            self.assertTrue(bool((y[1:] >= y[:-1]).all()))

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            spike_fn(torch.zeros(3), "relu", allow_experimental=True)


class EWCTests(unittest.TestCase):
    def test_requires_grad_validation(self):
        p = torch.zeros(4, requires_grad=False)
        with self.assertRaises(ValueError):
            EWCRegularizer({"w": p})

    def test_unarmed_penalty_zero(self):
        ewc = EWCRegularizer({"w": nn.Parameter(torch.ones(4))})
        self.assertFalse(ewc.is_armed)
        self.assertEqual(float(ewc.penalty()), 0.0)

    def test_fisher_nonzero_and_normalized(self):
        p = nn.Parameter(torch.ones(4))
        ewc = EWCRegularizer({"w": p}, lam=10.0)
        # Постоянный ненулевой градиент: d/dp (2·p).sum() = 2.
        mass = ewc.estimate_fisher(lambda: (p * 2.0).sum(), n_samples=4, allow_backward=True)
        self.assertTrue(ewc.is_armed)
        self.assertGreater(mass, 0.0)
        self.assertLessEqual(float(ewc.fisher["w"].max()), 1.0 + 1e-6)

    def test_penalty_grows_then_rebaseline_resets(self):
        p = nn.Parameter(torch.ones(4))
        ewc = EWCRegularizer({"w": p}, lam=10.0)
        ewc.estimate_fisher(lambda: (p * 2.0).sum(), n_samples=4, allow_backward=True)
        self.assertAlmostEqual(float(ewc.penalty().detach()), 0.0, places=5)
        p.data.add_(0.5)
        pen = float(ewc.penalty().detach())
        self.assertGreater(pen, 0.0)
        # lam · sum(F·(Δ)²) = 10 · 4 · 0.25 = 10 (F нормирован к 1).
        self.assertAlmostEqual(pen, 10.0, places=4)
        ewc.rebaseline()
        self.assertAlmostEqual(float(ewc.penalty().detach()), 0.0, places=5)



class OptInIsolationTests(unittest.TestCase):
    def test_learning_does_not_import_nw8_nw9(self):
        import ziwago.learning as L
        pkg_dir = Path(L.__file__).parent
        for py in pkg_dir.glob("*.py"):
            src = py.read_text(encoding="utf-8")
            self.assertNotIn("neuroweave8", src, f"{py.name} ссылается на nw8")
            self.assertNotIn("neuroweave9", src, f"{py.name} ссылается на nw9")

    def test_live_readout_contract_intact(self):
        # Живой ридеаут v19 запрещает surrogate/backprop (spiking_language.py:3);
        # пакет learning не должен становиться его зависимостью.
        import ziwago.cortex.spiking_language as sl
        src = Path(sl.__file__).read_text(encoding="utf-8")
        self.assertIn("No surrogate neurons", src)
        self.assertNotIn("from ..learning", src)
        self.assertNotIn("learning.surrogate", src)


if __name__ == "__main__":
    unittest.main()