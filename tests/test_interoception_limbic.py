"""М-6/М-8 (адаптация П-3): виртуальная интероцепция Insula–ACC + лимбика BLA/CeA.

БЕЗ физической арены и слогового декодирования (решение П-3: арена и syllabary
убьют производительность текстовой модели). Оживлены два мёртвых в v20 модуля
(живы только в GenesisOrganism): core/interoception.py + emotion/limbic.py.

Измерено прибором живости ДО правки (AGENTS.md §7):
  * IntegratedOrganism не нёс ни intero, ни limbic (мёртвый провод в v20);
  * причинные актуаторы интероцепции: met_gain (neuron.py:180) меняет v_s
    -40.17→-40.83; process_s (physiology.py:179) меняет f_mod 0.65→0.0;
  * причинный актуатор лимбики: valence→da_phasic (3-й фактор, synapse.py:502,
    external clock) знаковый: da 0→1.06e-5, +1→1.92e-5, -0.5→6.3e-6 dW.

Отрицательный результат (честно, НЕ оживлён): arousal→NE. В свежем v20-режиме
ne инвариантен на таламическом реле (тоническое насыщение, norm 2.40 при ne 0.2
и 1.2) и на нейроне (stochastic off, v_s -49.009 идентичен). Это мёртвый провод
по AGENTS.md §7 — лимбика НЕ модулирует ne, только da_phasic.

Контракты:
  * default off → intero/limbic=None, met_gain≡1.0, ctx не меняется, бит-идентично;
  * интероцепция: allostatic_drive→met_gain (recoverable EMA, bounded) + process_s;
  * лимбика: valence→da_phasic причинно меняет пластичность (при learn on);
  * телеметрия exposes self_state/intero_error/allostatic_drive и valence/arousal/
    drives (нет «тихих режекторов»).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_v19_spiking_language import tiny_core
from ziwago.organism.integrated import IntegratedOrganism, IntegrationConfig


def _run(ticks=6, energy=None, plastic=False, **cfg):
    core = tiny_core()
    core.enable_integrated_organism(config=IntegrationConfig(**cfg))
    core.life.training_active = plastic
    for _ in range(ticks):
        if energy is not None:
            for col in core.moe_cortex.columns:
                col.pyr.energy.fill_(energy)
        core.life.tick(core, torch.ones(64) * 0.3)
    return core


class ConfigValidationTests(unittest.TestCase):
    def test_invalid_allostatic_threshold_rejected(self):
        for bad in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                IntegrationConfig(intero_allostatic_threshold=bad)

    def test_negative_gains_rejected(self):
        for name in ("intero_sleep_gain", "intero_met_gain_scale", "limbic_da_valence_gain"):
            with self.assertRaises(ValueError):
                IntegrationConfig(**{name: -0.1})


class OptInIsolationTests(unittest.TestCase):
    """ЕДИНЫЙ КОНТУР (13.11): интероцепция и лимбика включены по умолчанию.

    Раньше они были выключены флагом (модули написаны, но инертны). Теперь
    они создаются всегда, а отключение остаётся явным opt-out.
    """

    def test_default_on_creates_owners(self):
        org = IntegratedOrganism(2, 64, 32, cfg=IntegrationConfig())
        self.assertIsNotNone(org.intero)
        self.assertIsNotNone(org.limbic)

    def test_explicit_opt_out_creates_nothing(self):
        org = IntegratedOrganism(2, 64, 32, cfg=IntegrationConfig(
            interoception_enabled=False, limbic_enabled=False))
        self.assertIsNone(org.intero)
        self.assertIsNone(org.limbic)

    def test_telemetry_enabled_by_default(self):
        core = _run()
        tel = core.life.telemetry(core)
        self.assertTrue(tel["interoception"]["enabled"])
        self.assertTrue(tel["limbic"]["enabled"])

    def test_opt_out_telemetry_disabled(self):
        core = _run(interoception_enabled=False, limbic_enabled=False)
        tel = core.life.telemetry(core)
        self.assertEqual(tel["interoception"], {"enabled": False})
        self.assertEqual(tel["limbic"], {"enabled": False})

    def test_met_gain_responds_to_deficit_when_enabled(self):
        """С интероцепцией включённой дефицит энергии ДОЛЖЕН поднимать met_gain."""
        core = _run(energy=0.40)
        gains = [float(col.pyr.met_gain) for col in core.moe_cortex.columns]
        self.assertTrue(any(g > 1.0 for g in gains),
                        f"при дефиците met_gain обязан расти, получено {gains}")

    def test_enabled_constructs_owners(self):
        org = IntegratedOrganism(2, 64, 32, cfg=IntegrationConfig(
            interoception_enabled=True, limbic_enabled=True))
        self.assertIsNotNone(org.intero)
        self.assertIsNotNone(org.limbic)
        self.assertEqual(org.intero.total_dim, 2 * 16)
        self.assertEqual(org.limbic.input_dim, 64)
        self.assertEqual(org.limbic.pfc_dim, org.ho.n_in)

    def test_flags_are_independent(self):
        only_intero = IntegratedOrganism(2, 64, 32, cfg=IntegrationConfig(
            interoception_enabled=True, limbic_enabled=False))
        self.assertIsNotNone(only_intero.intero)
        self.assertIsNone(only_intero.limbic)
        only_limbic = IntegratedOrganism(2, 64, 32, cfg=IntegrationConfig(
            interoception_enabled=False, limbic_enabled=True))
        self.assertIsNone(only_limbic.intero)
        self.assertIsNotNone(only_limbic.limbic)


class InteroceptionCausalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_met_gain_rises_with_allostatic_drive(self):
        core = _run(energy=0.40, interoception_enabled=True)
        self.assertGreater(core.life._last_intero.allostatic_drive, 0.35)
        for col in core.moe_cortex.columns:
            self.assertGreater(float(col.pyr.met_gain), 1.001)

    def test_met_gain_causal_on_vs_off(self):
        off = _run(energy=0.40, interoception_enabled=False)
        on = _run(energy=0.40, interoception_enabled=True)
        off_mg = float(off.moe_cortex.columns[0].pyr.met_gain)
        on_mg = float(on.moe_cortex.columns[0].pyr.met_gain)
        self.assertEqual(off_mg, 1.0)
        self.assertGreater(on_mg, off_mg)

    def test_met_gain_recoverable(self):
        # EMA возвращается к 1 при низком драйве (в v20 нет колоночного decay).
        core = tiny_core()
        core.enable_integrated_organism(config=IntegrationConfig(interoception_enabled=True))
        for col in core.moe_cortex.columns:
            col.pyr.met_gain = 1.5
            col.pyr.energy.fill_(1.0)
        core.life.tick(core, torch.ones(64) * 0.3)
        for col in core.moe_cortex.columns:
            self.assertLess(float(col.pyr.met_gain), 1.5)

    def test_met_gain_bounded_under_chronic_load(self):
        core = _run(ticks=40, energy=0.30, interoception_enabled=True)
        for col in core.moe_cortex.columns:
            self.assertLessEqual(float(col.pyr.met_gain), 2.0)

    def test_sleep_pressure_raised_causally(self):
        off = _run(energy=0.40, interoception_enabled=False)
        on = _run(energy=0.40, interoception_enabled=True)
        self.assertGreater(float(on.chemistry.process_s), float(off.chemistry.process_s) + 1e-6)

    def test_telemetry_exposes_self_state(self):
        core = _run(energy=0.40, interoception_enabled=True)
        it = core.life.telemetry(core)["interoception"]
        self.assertTrue(it["enabled"])
        for key in ("allostatic_drive", "intero_error", "self_state", "state_mean"):
            self.assertIn(key, it)
        self.assertIn("intero_error", it["self_state"])
        self.assertIn("allostatic_drive", it["self_state"])


class LimbicCausalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_da_phasic_augmented_by_valence(self):
        off = _run(limbic_enabled=False)
        on = _run(limbic_enabled=True)
        da_off = off.moe_cortex.columns[0]._mod_ctx["da_phasic"]
        da_on = on.moe_cortex.columns[0]._mod_ctx["da_phasic"]
        self.assertNotAlmostEqual(da_on, da_off, places=6)
        valence = on.life._last_limbic.valence
        self.assertAlmostEqual(da_on, da_off + 0.5 * valence, places=6)

    def test_zero_gain_is_bit_identical(self):
        off = _run(limbic_enabled=False)
        zero = _run(limbic_enabled=True, limbic_da_valence_gain=0.0)
        self.assertEqual(off.moe_cortex.columns[0]._mod_ctx["da_phasic"],
                         zero.moe_cortex.columns[0]._mod_ctx["da_phasic"])

    def test_plasticity_diverges_causally(self):
        off = _run(ticks=8, plastic=True, limbic_enabled=False)
        on = _run(ticks=8, plastic=True, limbic_enabled=True)
        w_off = off.moe_cortex.columns[0].w_ff.w_fast
        w_on = on.moe_cortex.columns[0].w_ff.w_fast
        self.assertGreater(float((w_on - w_off).abs().mean()), 1e-9)

    def test_valence_is_signed_and_bounded(self):
        core = _run(limbic_enabled=True)
        out = core.life._last_limbic
        self.assertGreaterEqual(out.valence, -1.0)
        self.assertLessEqual(out.valence, 1.0)
        self.assertGreaterEqual(out.arousal, 0.0)
        self.assertLessEqual(out.arousal, 1.0)

    def test_arousal_ne_not_wired_negative_result(self):
        # Отрицательный результат: лимбика НЕ модулирует ne (мёртвый провод в v20).
        off = _run(limbic_enabled=False)
        on = _run(limbic_enabled=True)
        self.assertEqual(off.moe_cortex.columns[0]._mod_ctx["ne"],
                         on.moe_cortex.columns[0]._mod_ctx["ne"])

    def test_telemetry_exposes_emotion(self):
        core = _run(limbic_enabled=True)
        lt = core.life.telemetry(core)["limbic"]
        self.assertTrue(lt["enabled"])
        for key in ("valence", "arousal", "d_energy", "d_safety", "d_novelty",
                    "appetitive", "aversive"):
            self.assertIn(key, lt)


if __name__ == "__main__":
    unittest.main()