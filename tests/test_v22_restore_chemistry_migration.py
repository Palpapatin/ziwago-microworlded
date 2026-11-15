"""Регрессии восстановления химии и маршрутизации действий (15.11.2026).

Обе поломки нашлись на живом прогоне восстановленного кандидата v19 и молча
ломали ВСЮ генерацию:

1. В кандидате v19 сохранено `ach=11.56` при номинальном ~0.4. Кламп оси
   был добавлен в `NeuroChemistry.update()`, но НЕ в путь восстановления, и
   валидатор action-контекста отвергал состояние: `ach must be in [0.0, 2.0]`
   → каждый запрос падал.
2. Резонанс CA3 у обученной модели доходит до 2.5, и компонента evidence
   `(1 - resonance) * 0.5` уходила в минус → `Invalid action context or
   evidence` → маршрутизация падала.

Контракт: миграция ВИДИМА (отчёт по каждой оси), а маршрутизация никогда не
строит отрицательный evidence.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from ziwago.chemistry.modulators import NeuroChemistry
from ziwago.config import ChemistryCfg


class ChemistryRestoreMigrationTests(unittest.TestCase):
    def test_legacy_ach_is_clamped_with_a_visible_report(self):
        chem = NeuroChemistry(ChemistryCfg())
        chem.s.ach = 11.5591          # значение из кандидата v19
        chem.s.da = 2.0
        chem.s.ado = 0.00404
        report = chem.normalize_restored_state()
        self.assertIn("ach", report, "миграция обязана быть видна, а не молчаливой")
        self.assertAlmostEqual(report["ach"]["raw"], 11.5591, places=4)
        self.assertAlmostEqual(report["ach"]["clamped"], 2.0, places=6)
        self.assertAlmostEqual(float(chem.s.ach), 2.0, places=6)
        # В границах состояния — отчёт пуст, состояние не трогаем.
        chem.s.ach = 0.4
        self.assertEqual(chem.normalize_restored_state(), {})

    def test_bounds_match_the_update_clamp(self):
        chem = NeuroChemistry(ChemistryCfg())
        for name in ("da", "ne", "ach", "ht", "cort", "ado", "ox", "lactate", "ecb"):
            self.assertIn(name, NeuroChemistry.STATE_BOUNDS)
        chem.s.ado = 5.0
        chem.s.ecb = 9.0
        report = chem.normalize_restored_state()
        self.assertAlmostEqual(float(chem.s.ado), 1.0, places=6)
        self.assertAlmostEqual(float(chem.s.ecb), 1.2, places=6)
        self.assertEqual(sorted(report), ["ado", "ecb"])


class ActionEvidenceTests(unittest.TestCase):
    def test_evidence_is_never_negative_for_strong_resonance(self):
        from ziwago_assistant import ZiwagoBiologicalSNNAssistant
        for resonance in (0.0, 0.3, 1.0, 2.5, 1e6, float("nan")):
            with self.subTest(resonance=resonance):
                evidence = ZiwagoBiologicalSNNAssistant._action_evidence(resonance, True)
                self.assertEqual(len(evidence), 7)
                self.assertTrue(all(v >= 0.0 for v in evidence), evidence)
                self.assertTrue(all(v == v for v in evidence), "NaN в evidence недопустим")

    def test_resonance_gate_saturates(self):
        from ziwago_assistant import ZiwagoBiologicalSNNAssistant
        low = ZiwagoBiologicalSNNAssistant._action_evidence(0.0, True)
        high = ZiwagoBiologicalSNNAssistant._action_evidence(5.0, True)
        self.assertAlmostEqual(low[6], 0.6, places=6)
        self.assertAlmostEqual(high[6], 1.0, places=6)
        self.assertAlmostEqual(low[4], 0.5, places=6)
        self.assertAlmostEqual(high[4], 0.0, places=6)


class ColumnRestoreCompletionTests(unittest.TestCase):
    """Новые поля живых объектов не ломают восстановление старых снимков."""

    def test_astrocyte_is_created_lazily_for_restored_columns(self):
        from ziwago.cortex.column import CorticalColumn
        from ziwago.config import USNeuronCfg, USSynapseCfg, ZoneCfg
        column = CorticalColumn(
            ZoneCfg(name="Z", n_neurons=8, n_pv=2, n_sst=1, n_vip=1, g=0.4),
            16, 4, USNeuronCfg(), USSynapseCfg(), seed=1)
        del column.astrocyte          # как в снимке v19: поля нет
        astro = column._ensure_astrocyte()
        self.assertIsNotNone(astro)
        self.assertIsNotNone(astro.ca, "ca должен быть инициализирован, а не None")
        self.assertIs(column._ensure_astrocyte(), astro, "создание должно быть идемпотентным")

    def test_restore_migrations_are_reported(self):
        from ziwago.organism.flat import ZiwagoPureSynapticDialogueOrganismV14 as Organism
        self.assertTrue(hasattr(Organism, "apply_restore_migrations"),
                        "миграции после восстановления должны быть одной точкой")

    def test_legacy_dendrite_threshold_baseline_migration_is_reported(self):
        from test_v19_spiking_language import tiny_core

        core = tiny_core()
        dendrite = core.moe_cortex.columns[0].dendrite
        dendrite.theta_ca = -31.0
        del dendrite.theta_ca_base

        report = core.apply_restore_migrations()

        self.assertEqual(dendrite.theta_ca_base, -31.0)
        self.assertEqual(report["dendrite_theta_ca_baseline"]["columns"], [0])

    def test_accumulated_legacy_threshold_is_reconciled_to_restored_calcium(self):
        from test_v19_spiking_language import tiny_core

        core = tiny_core()
        column = core.moe_cortex.columns[0]
        column.dendrite.theta_ca_base = -25.0
        column.dendrite.theta_ca = -33.0
        column.astrocyte.ca.fill_(0.5)

        report = core.apply_restore_migrations()

        self.assertAlmostEqual(column.dendrite.theta_ca, -27.0, places=6)
        self.assertEqual(report["dendrite_theta_ca_reconciled"][0]["previous"], -33.0)


if __name__ == "__main__":
    unittest.main()
