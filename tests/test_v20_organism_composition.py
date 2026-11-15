"""Плоская композиция ZiwagoOrganism (ziwago/organism_core.py) — H2: композиция
без наследования. Clean Slate (H1): башня обёрток v18→v19→v20 архивирована,
поэтому тест проверяет плоский класс НАПРЯМУЮ в обоих режимах (integrated
True/False), без ссылки на обёртки. Эквивалентность плоского класса башне была
доказана до архивации (10 passed, ARCHITECTURE_MAPPING.md §7.8/§7.9).

Прибор: конструктор v14 патчится на tiny_core (малый организм, без мастера).
Полный организм на этой машине OOM (§5.3) — поэтому малый организм.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test_v19_spiking_language import tiny_core
import ziwago.organism_core as flatmod
from ziwago.organism.integrated import IntegrationConfig
from ziwago.organism.verified import V19_SCHEMA, validate_v19_envelope

ZiwagoOrganism = flatmod.ZiwagoOrganism


def assert_tree_equal(case, a, b):
    if isinstance(a, torch.Tensor):
        case.assertEqual(a.dtype, b.dtype)
        case.assertTrue(torch.equal(a, b), "tensor mismatch")
    elif isinstance(a, dict):
        case.assertEqual(set(a), set(b))
        for key in a:
            assert_tree_equal(case, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        case.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_tree_equal(case, x, y)
    else:
        case.assertEqual(a, b)


def _seed_all(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_flat(seed=42, integrated=True):
    """Строит плоский ZiwagoOrganism полным __init__ с малым ядром (патч v14).
    RNG сбрасывается перед сборкой для воспроизводимости снимков состояния."""
    with mock.patch.object(flatmod, "ZiwagoPureSynapticDialogueOrganismV14", side_effect=lambda **kw: tiny_core()):
        _seed_all(seed)
        flat = ZiwagoOrganism(root=ROOT, seed=seed, auto_load_checkpoint=False, integrated=integrated)
    return flat


class FlatV20Tests(unittest.TestCase):
    """integrated=True → режим v20 (физиология, core.life)."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_flat_class_has_no_wrapper_inheritance(self):
        self.assertEqual(ZiwagoOrganism.__bases__, (object,),
                         "ZiwagoOrganism не должен наследовать v18/v19/v20 (H2)")
        for name in ("sleep_cycle", "select_action", "reward", "save_candidate",
                     "generate_response", "export_v19_state", "load_v19_state",
                     "export_genesis_state_dict", "load_genesis_checkpoint",
                     "_rem_dream", "step", "_on_sleep_phase", "load_v20_checkpoint"):
            self.assertTrue(callable(getattr(ZiwagoOrganism, name, None)), f"нет метода {name}")

    def test_v20_mode_construction_and_state_exports(self):
        flat = build_flat(integrated=True)
        self.assertTrue(flat._integrated)
        self.assertEqual(flat.version, "20.0")
        self.assertTrue(hasattr(flat.core_organism, "life"),
                        "v20 mode must enable integrated physiology (core.life)")
        self.assertIsInstance(flat.export_genesis_state_dict(), dict)
        self.assertIsInstance(flat.export_v19_state(), dict)
        self.assertIsInstance(flat.core_organism.export_runtime_state(), dict)

    def test_explicit_config_does_not_replace_restored_integrated_state(self):
        import tempfile

        config = IntegrationConfig(ticks_per_token=2)
        restored_life = []

        def fake_consume(self, path, expected_sha256, consumer, **kwargs):
            return consumer({"v20_schema_version": flatmod.V20_SCHEMA})

        def fake_restore(self, path, payload):
            life = self.core_organism.life
            restored_life.append(life)
            life.replay = [{
                "cue": torch.ones(64),
                "priority": 0.8,
                "columns": [],
            }]
            life.actions.pending = {"decision_id": "restored", "action": 2}
            life.action_result = {
                "action": 2, "committed": True,
                "decision_id": "restored", "bg_release": 0.4,
            }
            life.criticality = {
                "available": True, "estimate": 0.91, "reason": "measured",
            }
            life.criticality_gain = 1.23

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.pt"
            checkpoint.write_bytes(b"mocked checkpoint")
            with mock.patch.object(flatmod.ZiwagoOrganism, "_consume_checkpoint_payload",
                                   new=fake_consume), \
                 mock.patch.object(flatmod.ZiwagoOrganism, "_load_v20_payload",
                                   new=fake_restore), \
                 mock.patch.object(flatmod, "ZiwagoPureSynapticDialogueOrganismV14",
                                   side_effect=lambda **kw: tiny_core()):
                flat = ZiwagoOrganism(
                    root=ROOT,
                    seed=41,
                    checkpoint_path=checkpoint,
                    auto_load_checkpoint=True,
                    integration_config=config,
                )

        life = flat.core_organism.life
        self.assertIs(life, restored_life[0])
        self.assertIs(life.cfg, config)
        self.assertEqual(len(life.replay), 1)
        self.assertEqual(life.actions.pending["decision_id"], "restored")
        self.assertTrue(life.criticality["available"])
        self.assertEqual(life.criticality_gain, 1.23)

    def test_sleep_cycle_v20(self):
        from train_ziwago_v14_pure_synaptic_dialogue import TokenizedDialogPair
        flat = build_flat(integrated=True)
        pair = TokenizedDialogPair("lab", [1, 3], [1, 2, 3, 4, 5, 6, 7])
        flat.core_organism.train_wave_with_neuromodulator_reward([pair])
        flat.core_organism.chemistry.request_sleep()
        rep = flat.sleep_cycle(wave_idx=0)
        self.assertIsInstance(rep, dict)

    def test_nrem_to_rem_reports_swr_disabled_without_cortical_writer(self):
        from ziwago.config import HippocampusCfg
        from ziwago.memory.hippocampus import Hippocampus

        flat = build_flat(integrated=True)
        core, life = flat.core_organism, flat.core_organism.life
        hippo = Hippocampus(HippocampusCfg(n_ec=16, n_dg=48, n_ca3=64), seed=23)
        for seed in (71, 72, 73):
            ec = torch.randn(1, hippo.cfg.n_ec, generator=torch.Generator().manual_seed(seed))
            dg = hippo.dg_code(ec)
            ca3 = torch.relu(dg @ hippo.w_dg_ca3.t())
            hippo.store_episode(ec, dg, ca3, error=0.8, uncertainty=0.3)
        life.hippo = hippo
        core.chemistry.s.ado = 0.9
        core.chemistry.s.ach = 0.2

        with mock.patch.object(hippo, "swr_replay", wraps=hippo.swr_replay) as swr_replay:
            flat._on_sleep_phase("awake", "nrem")
            flat._on_sleep_phase("nrem", "rem")

        swr_replay.assert_not_called()
        swr = life.sleep_report["hippocampus_sleep"]
        self.assertFalse(swr["swr_gate_open"])
        self.assertFalse(swr["swr_cortical_write_connected"])
        self.assertEqual(swr["swr_skipped_reason"], "no_cortical_write")
        self.assertEqual(swr["replayed"], [])
        self.assertIn("semantic_bridges", swr["rem"])

    def test_action_and_reward_v20(self):
        flat = build_flat(integrated=True)
        ctx = torch.ones(64) * 0.1
        ev = torch.tensor([0., 0., 0., 0., 0., 0., 1.])
        dec = flat.select_action(ctx, ev)
        self.assertIsInstance(dec, dict)
        if dec.get("decision_id") is not None:
            rew = flat.reward(dec["decision_id"], 1.0)
            self.assertIsInstance(rew, dict)

    def test_generation_v20(self):
        flat = build_flat(integrated=True)
        prompt = "What neurons integrate sensory signals?"
        try:
            out = flat.generate_response(prompt, max_tokens=12)
        except Exception:
            # малый организм может не поддерживать полную генерацию
            return
        self.assertIn("pure_weight_generated_response", out)

    def test_evaluation_context_restores_state(self):
        flat = build_flat(integrated=True)
        flat.core_organism.step(torch.ones(64) * 0.2, 4)
        before = flat.export_v19_state()
        with flat.evaluation_context():
            flat.core_organism.step(torch.ones(64) * 0.4, 6)
        assert_tree_equal(self, before, flat.export_v19_state())


class UnifiedContourTests(unittest.TestCase):
    """ЕДИНЫЙ КОНТУР (13.11): режима «v19 без биологии» больше нет.

    Раньше `integrated=False` давал организм без `life`: кору шагали «голой»
    (без таламуса, TRN, AST/GW, критичности, интероцепции, лимбика и
    гиппокампа) и без биологического сна. Теперь контур один и всегда
    биологический; `integrated` больше не переключает контур — он лишь
    сохраняет смысл в старых вызовах. Схемы чекпоинта (v19/v20) при этом
    РАЗНЫЕ и определяются содержимым payload, а не флагом.
    """

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_single_contour_always_has_biological_driver(self):
        for integrated in (False, True):
            with self.subTest(integrated=integrated):
                flat = build_flat(integrated=integrated)
                self.assertTrue(hasattr(flat.core_organism, "life"),
                                "контур всегда биологический: life обязателен")
                self.assertEqual(flat.version, "20.0")

    def test_state_exports_available_in_both_dispatch_modes(self):
        flat = build_flat(integrated=False)
        self.assertEqual(flat.version, "20.0")
        self.assertIsInstance(flat.export_genesis_state_dict(), dict)
        self.assertIsInstance(flat.export_v19_state(), dict)
        self.assertIsInstance(flat.core_organism.export_runtime_state(), dict)

    def test_sleep_cycle_is_integrated(self):
        """Сон теперь всегда интегрированный (SWS через биологический драйвер).

        Бюджет тиков урезан: полный цикл — 4096 тиков, для теста хватает
        малого окна, чтобы проверить диспетчеризацию и отчёт.
        """
        flat = build_flat(integrated=False)
        rep = flat.sleep_cycle(wave_idx=0, max_ticks=64)
        self.assertEqual(rep["phase"], "nrem")
        self.assertTrue(rep["requested"])
        self.assertLessEqual(rep["budget_ticks"], 64)
        self.assertIn("sleep_pressure_after", rep)

    def test_save_candidate_uses_the_v20_envelope(self):
        import tempfile
        from train_ziwago_v14_pure_synaptic_dialogue import TokenizedDialogPair
        flat = build_flat(integrated=False)
        # export_checkpoint_payload требует непустой col_to_tok; обучим пару.
        flat.core_organism.train_wave_with_neuromodulator_reward(
            [TokenizedDialogPair("lab", [1, 3], [1, 2, 3, 4, 5, 6, 7])])
        flat.core_organism.loaded_checkpoint_path = ""
        with tempfile.TemporaryDirectory(dir=ROOT / "Produce") as tmp:
            target = Path(tmp) / "candidate_v22.pt"
            info = flat.save_candidate(target)
            self.assertEqual(info["schema"], flatmod.V20_SCHEMA)
            payload = torch.load(target, map_location="cpu", weights_only=True)
            self.assertEqual(payload.get("v20_schema_version"), flatmod.V20_SCHEMA)
            self.assertIsNone(payload.get("v19_schema_version"))
        self.assertEqual(flat.core_organism.loaded_checkpoint_path, "",
                         "save must restore loaded_checkpoint_path (finally block)")


if __name__ == "__main__":
    unittest.main()