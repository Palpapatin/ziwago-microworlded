"""§8.1 миграция: merged DualHippocampus в живом IntegratedOrganism за флагом v20.

Целевой контейнер — НЕ organism/genesis.py (как писал ранний план §8.1): genesis
архивирован как мёртвый (§8.7), живой оркестратор v20 — IntegratedOrganism.
Подключено по тому же opt-in контракту, что М-4/М-8 (volume/intero/limbic):

  * default off → hippo=None, da_phasic/ne не меняются, бит-идентично v19;
  * причинный актуатор: новизна CA1 (VTA DA burst) → da_phasic (3-й фактор
    пластичности, synapse.py:502, измерен причинным в М-8);
  * эпизодическая память оживает через record_credit (живой хук v14:3055):
    resonance/VSA-DHT retrieval читаются, а не только пишутся (§8.1);
  * отрицательный результат (честно, НЕ оживлён): NE-burst → ne. В свежем v20
    ne инвариантен (тоническое насыщение реле + stochastic off нейрона) — мёртвый
    провод по AGENTS.md §7, тот же результат, что в лимбике (М-8);
  * телеметрия exposes resonance/mismatch/decision/novelty/recall/capacities
    (нет «тихих режекторов», AGENTS.md §6).
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
from ziwago.memory.hippocampus import Hippocampus


def _run(ticks=4, plastic=False, seed=0, record=False, **cfg):
    """Новизна каждый такт (случайный EC-вход из отдельного генератора), чтобы
    CA1-компаратор давал ненулевой mismatch → VTA burst. Одинаковые входы для
    off/on гарантированы выделенным генератором, независимым от RNG конструкции."""
    core = tiny_core()
    core.enable_integrated_organism(config=IntegrationConfig(**cfg))
    core.life.training_active = plastic
    g = torch.Generator().manual_seed(seed)
    out = None
    for _ in range(ticks):
        x = torch.randn(64, generator=g)
        out = core.life.tick(core, x)
        if record:
            core.life.record_credit(core, x, out, loss=0.5)
    return core


class ConfigValidationTests(unittest.TestCase):
    def test_invalid_n_ec_rejected(self):
        for bad in (0, -4, 3.5, True):
            with self.assertRaises(ValueError):
                IntegrationConfig(hippo_n_ec=bad)

    def test_invalid_theta_mode_rejected(self):
        with self.assertRaises(ValueError):
            IntegrationConfig(hippo_theta_mode="warp")

    def test_invalid_theta_cycle_steps_rejected(self):
        for bad in (0.0, -8.0, float("inf")):
            with self.assertRaises(ValueError):
                IntegrationConfig(hippo_theta_cycle_steps=bad)

    def test_negative_novelty_gain_rejected(self):
        for bad in (-0.1, float("nan")):
            with self.assertRaises(ValueError):
                IntegrationConfig(hippo_novelty_da_gain=bad)


class OptInIsolationTests(unittest.TestCase):
    """ЕДИНЫЙ КОНТУР (13.11): дуальный гиппокамп включён по умолчанию.

    Раньше он был выключен флагом, то есть написанный и измеренный модуль не
    участвовал в живом контуре. Теперь он создаётся всегда; отключение —
    явный opt-out.
    """

    def test_default_on_creates_owner(self):
        org = IntegratedOrganism(2, 64, 32, cfg=IntegrationConfig())
        self.assertIsNotNone(org.hippo)
        self.assertIsNone(org._last_hippo)   # ещё ни одного тика

    def test_explicit_opt_out_creates_nothing(self):
        org = IntegratedOrganism(2, 64, 32, cfg=IntegrationConfig(dual_hippocampus_enabled=False))
        self.assertIsNone(org.hippo)
        self.assertIsNone(org._last_hippo)

    def test_telemetry_enabled_by_default(self):
        core = _run()
        self.assertTrue(core.life.telemetry(core)["dual_hippocampus"]["enabled"])

    def test_opt_out_telemetry_disabled(self):
        core = _run(dual_hippocampus_enabled=False)
        self.assertEqual(core.life.telemetry(core)["dual_hippocampus"], {"enabled": False})

    def test_enabled_constructs_owner(self):
        org = IntegratedOrganism(2, 64, 32, cfg=IntegrationConfig(dual_hippocampus_enabled=True))
        self.assertIsNotNone(org.hippo)
        self.assertIsInstance(org.hippo.episodic, Hippocampus)
        self.assertEqual(org.hippo.n_ec, 64)
        self.assertEqual(org.hippo.theta.mode, "token")

    def test_flags_are_independent(self):
        only_hippo = IntegratedOrganism(2, 64, 32, cfg=IntegrationConfig(
            dual_hippocampus_enabled=True, limbic_enabled=False, interoception_enabled=False))
        self.assertIsNotNone(only_hippo.hippo)
        self.assertIsNone(only_hippo.limbic)
        self.assertIsNone(only_hippo.intero)
        only_limbic = IntegratedOrganism(2, 64, 32, cfg=IntegrationConfig(
            dual_hippocampus_enabled=False, limbic_enabled=True, interoception_enabled=False))
        self.assertIsNone(only_limbic.hippo)
        self.assertIsNotNone(only_limbic.limbic)
        self.assertIsNone(only_limbic.intero)

    def test_off_da_phasic_and_ne_unchanged(self):
        off = _run(dual_hippocampus_enabled=False, limbic_enabled=False)
        self.assertIsNone(off.life.hippo)
        # Без флага актуатор не проводится: da_phasic — только базовая химия.
        base = off.moe_cortex.columns[0]._mod_ctx["da_phasic"]
        self.assertAlmostEqual(base, off.chemistry.bundle()["da_phasic"], places=7)


class HippocampusCausalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_novelty_da_augments_da_phasic(self):
        off = _run(dual_hippocampus_enabled=False)
        on = _run(dual_hippocampus_enabled=True)
        da_off = off.moe_cortex.columns[0]._mod_ctx["da_phasic"]
        da_on = on.moe_cortex.columns[0]._mod_ctx["da_phasic"]
        burst = on.life._last_hippo.hippo.vta_da_burst
        self.assertGreater(burst, 0.0)
        self.assertAlmostEqual(da_on, da_off + 0.5 * burst, places=6)

    def test_zero_gain_is_bit_identical(self):
        """Нулевой вклад новизны ⇒ DA-канал совпадает с выключенным гиппокампом."""
        off = _run(ticks=6, plastic=True, dual_hippocampus_enabled=False)
        zero = _run(ticks=6, plastic=True, dual_hippocampus_enabled=True,
                    hippo_novelty_da_gain=0.0)
        self.assertEqual(off.moe_cortex.columns[0]._mod_ctx["da_phasic"],
                         zero.moe_cortex.columns[0]._mod_ctx["da_phasic"])

    def test_plasticity_diverges_causally(self):
        off = _run(ticks=8, plastic=True, dual_hippocampus_enabled=False)
        on = _run(ticks=8, plastic=True, dual_hippocampus_enabled=True)
        w_off = off.moe_cortex.columns[0].w_ff.w_fast
        w_on = on.moe_cortex.columns[0].w_ff.w_fast
        self.assertGreater(float((w_on - w_off).abs().mean()), 1e-9)

    def test_ne_not_wired_negative_result(self):
        # Отрицательный результат: гиппокамп НЕ модулирует ne (мёртвый провод в v20).
        off = _run(dual_hippocampus_enabled=False)
        on = _run(dual_hippocampus_enabled=True)
        self.assertEqual(off.moe_cortex.columns[0]._mod_ctx["ne"],
                         on.moe_cortex.columns[0]._mod_ctx["ne"])

    def test_telemetry_exposes_state(self):
        core = _run(dual_hippocampus_enabled=True)
        ht = core.life.telemetry(core)["dual_hippocampus"]
        self.assertTrue(ht["enabled"])
        for key in ("resonance", "mismatch", "decision", "novelty_da", "novelty_ne",
                    "recalled_episode_ids", "concept_chain", "tick_n", "vsa_capacity",
                    "n_ec", "novelty_da_gain", "theta_mode"):
            self.assertIn(key, ht)
        self.assertIn(ht["decision"], {"accept", "regenerate", "ask", "unknown"})


class EpisodicMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_record_credit_stores_and_exact_episode_resonates(self):
        core = tiny_core()
        core.enable_integrated_organism(config=IntegrationConfig(dual_hippocampus_enabled=True))
        core.life.training_active = True
        g = torch.Generator().manual_seed(3)
        x = torch.randn(64, generator=g)
        out = core.life.tick(core, x)
        stored_ec = core.life._last_hippo.hippo.ec.detach().clone()
        self.assertAlmostEqual(core.life._last_hippo.resonance, 0.0, places=5)
        core.life.record_credit(core, x, out, loss=0.5)
        self.assertEqual(len(core.life.hippo.episodic.episodes), 1)
        verified = core.life.hippo.verify(stored_ec)
        self.assertGreater(verified["resonance"], 0.5)

    def test_vsa_dht_retrieval_reads_stored_episodes(self):
        """Записанный эпизод должен читаться поиском (после открытия θ-гейта).

        Трек B (13.11) ввёл настоящий θ-гейт: реколл разрешён ОДИН раз за
        θ-цикл (8 токенов), поэтому поиск запускается не с первого тика.
        """
        core = tiny_core()
        core.enable_integrated_organism(config=IntegrationConfig(dual_hippocampus_enabled=True))
        core.life.training_active = True
        g = torch.Generator().manual_seed(5)
        x = torch.randn(64, generator=g)
        out = core.life.tick(core, x)
        core.life.record_credit(core, x, out, loss=0.5)
        recalled = []
        for _ in range(12):                      # проходим θ-цикл целиком
            core.life.tick(core, x)
            again = core.life._last_hippo
            recalled.extend(again.recalled_episode_ids)
        self.assertIn(0, recalled)
        self.assertGreater(len(core.life._last_hippo.recall_scores), 0)

    def test_off_stores_no_episodes(self):
        core = _run(record=True, dual_hippocampus_enabled=False)
        self.assertIsNone(core.life.hippo)
        self.assertEqual(core.life.telemetry(core)["dual_hippocampus"], {"enabled": False})


if __name__ == "__main__":
    unittest.main()