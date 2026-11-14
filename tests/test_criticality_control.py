"""М-2: гомеостаз критичности (таламический гейн) — ЕДИНЫЙ КОНТУР (13.11.2026).

Контракты:
  * контроллер включён по умолчанию (единый контур: биология — часть
    организма с рождения), но без оценки σ̂ гейн ≡ 1 и реле не тронут;
  * гейн причинно масштабирует таламический реле (не декоративная телеметрия);
  * знак регуляции верен: σ̂ < target → гейн растёт, σ̂ > target → падает;
  * границы гейна соблюдаются, нет оценки → нет регуляции;
  * выключатель по-прежнему существует, но это ЯВНЫЙ opt-out, а не дефолт.
Прибор — малая реальная кора (tiny_core), без мастер-чекпоинтов.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test_v19_spiking_language import tiny_core
from ziwago.organism.integrated import IntegratedOrganism, IntegrationConfig


def _organism(**overrides):
    cfg = IntegrationConfig(**overrides)
    return IntegratedOrganism(n_zones=2, n_input=64, n_neurons=96, seed=19, cfg=cfg)


class ConfigValidationTests(unittest.TestCase):
    def test_default_is_on_but_gain_is_unit_without_estimate(self):
        cfg = IntegrationConfig()
        # Единый контур: контроллер включён по умолчанию.
        self.assertTrue(cfg.criticality_control_enabled)
        self.assertEqual(cfg.criticality_target, 0.97)
        org = _organism()
        # Но без измеренной σ̂ регуляции нет — гейн остаётся единичным.
        self.assertEqual(org.criticality_gain, 1.0)
        self.assertFalse(org.criticality.get("controller_enabled", False))

    def test_target_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            IntegrationConfig(criticality_target=0.0)
        with self.assertRaises(ValueError):
            IntegrationConfig(criticality_target=1.6)

    def test_negative_rate_rejected(self):
        with self.assertRaises(ValueError):
            IntegrationConfig(criticality_gain_rate=-0.1)

    def test_gain_bounds_must_straddle_one(self):
        with self.assertRaises(ValueError):
            IntegrationConfig(criticality_gain_min=1.2, criticality_gain_max=1.5)
        with self.assertRaises(ValueError):
            IntegrationConfig(criticality_gain_min=0.5, criticality_gain_max=0.9)


class UpdateRuleTests(unittest.TestCase):
    """Прямая проверка _update_criticality_gain с заданной оценкой σ̂."""

    def test_disabled_never_regulates(self):
        org = _organism(criticality_control_enabled=False)
        org.criticality = {"available": True, "estimate": 0.5}
        org._update_criticality_gain()
        self.assertEqual(org.criticality_gain, 1.0)
        self.assertFalse(org.criticality.get("controller_enabled", False))

    def test_no_estimate_no_regulation(self):
        org = _organism(criticality_control_enabled=True)
        org.criticality = {"available": False, "estimate": None}
        org._update_criticality_gain()
        self.assertEqual(org.criticality_gain, 1.0)
        self.assertFalse(org.criticality.get("controller_enabled", False))

    def test_subcritical_raises_gain(self):
        org = _organism(criticality_control_enabled=True)
        org.criticality = {"available": True, "estimate": 0.90}
        org._update_criticality_gain()
        self.assertGreater(org.criticality_gain, 1.0)
        self.assertTrue(org.criticality["controller_enabled"])
        self.assertEqual(org.criticality["gain"], org.criticality_gain)

    def test_supercritical_lowers_gain(self):
        org = _organism(criticality_control_enabled=True)
        org.criticality = {"available": True, "estimate": 1.05}
        org._update_criticality_gain()
        self.assertLess(org.criticality_gain, 1.0)
        self.assertTrue(org.criticality["controller_enabled"])

    def test_at_target_holds_unit_gain(self):
        org = _organism(criticality_control_enabled=True)
        org.criticality = {"available": True, "estimate": org.cfg.criticality_target}
        for _ in range(50):
            org._update_criticality_gain()
        self.assertAlmostEqual(org.criticality_gain, 1.0, places=6)

    def test_gain_respects_upper_bound(self):
        org = _organism(criticality_control_enabled=True)
        org.criticality = {"available": True, "estimate": 0.0}
        for _ in range(200):
            org._update_criticality_gain()
        self.assertGreater(org.criticality_gain, 1.4)
        self.assertLessEqual(org.criticality_gain, org.cfg.criticality_gain_max)

    def test_gain_respects_lower_bound(self):
        org = _organism(criticality_control_enabled=True)
        org.criticality = {"available": True, "estimate": 3.0}
        for _ in range(200):
            org._update_criticality_gain()
        self.assertLess(org.criticality_gain, 0.6)
        self.assertGreaterEqual(org.criticality_gain, org.cfg.criticality_gain_min)


class CausalRelayTests(unittest.TestCase):
    """Гейн причинно масштабирует реле на живой коре, а не только в отчёте."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _relay_norm(self, enabled, gain):
        core = tiny_core()
        cfg = IntegrationConfig(criticality_control_enabled=enabled)
        core.enable_integrated_organism(config=cfg)
        core.life.criticality_gain = gain
        core.life.tick(core, torch.ones(64) * 0.3)
        return core.life.last_relay_norm

    def test_enabled_unit_gain_is_bit_identical_to_off(self):
        off = self._relay_norm(enabled=False, gain=1.0)
        on = self._relay_norm(enabled=True, gain=1.0)
        self.assertEqual(off, on)

    def test_gain_scales_relay_causally(self):
        base = self._relay_norm(enabled=False, gain=1.0)
        scaled = self._relay_norm(enabled=True, gain=1.4)
        self.assertGreater(base, 0.0)
        self.assertAlmostEqual(scaled, base * 1.4, places=5)

    def test_telemetry_exposes_gain(self):
        core = tiny_core()
        core.enable_integrated_organism(config=IntegrationConfig(criticality_control_enabled=True))
        core.life.criticality_gain = 1.23
        report = core.life.telemetry(core)
        self.assertEqual(report["criticality_gain"], 1.23)
        self.assertTrue(report["config"]["criticality_control_enabled"])


class LiveMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_integrated_monitor_receives_live_spike_totals_each_tick(self):
        core = tiny_core()
        core.enable_integrated_organism(config=IntegrationConfig())
        generator = torch.Generator().manual_seed(27)
        for _ in range(4):
            core.life.tick(core, torch.randn(64, generator=generator))
        history = core.life._criticality_monitor.activity
        self.assertEqual(len(history), 4)
        self.assertAlmostEqual(history[-1], core.life.last_tick["spikes"], places=6)

    def test_measured_criticality_reaches_live_homeostatic_controller(self):
        core = tiny_core()
        core.enable_integrated_organism(config=IntegrationConfig())
        monitor = core.life._criticality_monitor
        monitor.k_max = 10_000
        monitor.sigma_hat = 1.5
        monitor.sigma_hat_ready = True
        generator = torch.Generator().manual_seed(28)
        for _ in range(32):
            core.life.tick(core, torch.randn(64, generator=generator))
        telemetry = core.life.telemetry(core)["homeostasis"]
        self.assertEqual(core.life.homeostatic._tick, 1)
        self.assertTrue(telemetry["connected"])
        self.assertEqual(telemetry["reason"], "measured")
        self.assertEqual(telemetry["report"]["depression"], len(core.moe_cortex.columns))


if __name__ == "__main__":
    unittest.main()