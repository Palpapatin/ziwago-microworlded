"""М-4: оживлённая объёмная химия NO/H2S + пул агентов (opt-in).

Измерено до правки (прибор живости, AGENTS.md §7):
  * живая колонка УЖЕ читает эффекторы из mod_ctx (column.py: nmda_gain,
    histamine, vth_shift, gaba_tonic, rate_ceiling, ahp_block, bdnf) и
    _no_ltp_field, но в v20 IntegratedOrganism они НЕ заполнялись → нейтральные
    дефолты (nmda_gain=1.0, _no_ltp_field=None) = мёртвый провод;
  * ExtendedChemistry/VolumeTransmission живы только в GenesisOrganism.

Контракты:
  * default off → xchem/volume=None, ctx не меняется, бит-идентично v19/v20;
  * on → эффекторы в mod_ctx причинно меняют дендритные токи (i_ff/i_fb/v_a);
  * NO LTP поле раздаётся активным колонкам, форма и диапазон верны;
  * телеметрия exposes все концентрации агентов (нет «тихих режекторов»).

Отрицательный результат (честно): спайковый выход и Ca-след в свежей tiny_core
НЕ меняются (ca = ca*(1-dt/tau)+s, neuron.py:279 — Ca управляется спайками, а
спайки задаются навязанным соматическим драйвом). Причинность доказана на
дендритных токах, не на Ca. На обученных весах эффект дойдёт до спайков.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_v19_spiking_language import tiny_core
from ziwago.organism.integrated import IntegratedOrganism, IntegrationConfig


def _run(enabled, ticks=6):
    core = tiny_core()
    core.enable_integrated_organism(config=IntegrationConfig(volume_chemistry_enabled=enabled))
    for _ in range(ticks):
        core.life.tick(core, torch.ones(64) * 0.3)
    return core


class OptInIsolationTests(unittest.TestCase):
    def test_default_on_creates_owners(self):
        """ЕДИНЫЙ КОНТУР (13.11): объёмная химия включена по умолчанию.

        Раньше модуль был написан, но выключен флагом по умолчанию — при
        живом `OrganismPhysiology` (единственный владелец DA/NE/ACh) это был
        «призрак». Теперь он создаётся всегда, а отключение — явный opt-out.
        """
        org = IntegratedOrganism(2, 64, 32, cfg=IntegrationConfig())
        self.assertIsNotNone(org.xchem)
        self.assertIsNotNone(org.volume)
        self.assertEqual(org.volume.n, 2 * 32)

    def test_explicit_opt_out_creates_nothing(self):
        """Явное отключение по-прежнему возможно и ничего не создаёт."""
        org = IntegratedOrganism(2, 64, 32, cfg=IntegrationConfig(volume_chemistry_enabled=False))
        self.assertIsNone(org.xchem)
        self.assertIsNone(org.volume)
        self.assertEqual(org._update_extended_chemistry({"ne": 0.3, "ach": 0.4, "p_release": 1.0},
                                                        torch.ones(64)), {})

    def test_off_ctx_and_columns_unchanged(self):
        core = _run(False)
        col = core.moe_cortex.columns[0]
        self.assertNotIn("nmda_gain", col._mod_ctx)
        self.assertFalse(hasattr(col, "_no_ltp_field"))
        self.assertEqual(core.life.telemetry(core)["volume_chemistry"], {"enabled": False})

    def test_enabled_constructs_owners(self):
        org = IntegratedOrganism(2, 64, 32, cfg=IntegrationConfig(volume_chemistry_enabled=True))
        self.assertIsNotNone(org.xchem)
        self.assertIsNotNone(org.volume)
        self.assertEqual(org.volume.n, 2 * 32)


class CausalEffectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_effectors_populate_mod_ctx(self):
        core = _run(True)
        mc = core.moe_cortex.columns[0]._mod_ctx
        for key in ("nmda_gain", "histamine", "vth_shift", "gaba_tonic", "rate_ceiling", "bdnf"):
            self.assertIn(key, mc)
        self.assertNotAlmostEqual(mc["nmda_gain"], 1.0, places=3)

    def test_dendritic_currents_change_causally(self):
        off = _run(False).moe_cortex.columns[0].pyr
        on = _run(True).moe_cortex.columns[0].pyr
        # p_release<1 масштабирует AMPA (i_ff), nmda_gain<1 масштабирует NMDA (i_fb)
        self.assertGreater(float(off.last_i_ff.abs().mean()), float(on.last_i_ff.abs().mean()))
        self.assertNotAlmostEqual(float(off.last_i_fb.abs().mean()),
                                  float(on.last_i_fb.abs().mean()), places=5)
        self.assertNotAlmostEqual(float(off.v_a.mean()), float(on.v_a.mean()), places=4)

    def test_p_release_scaled_by_agent_pool(self):
        off = _run(False).moe_cortex.columns[0]._mod_ctx["p_rel"]
        on = _run(True).moe_cortex.columns[0]._mod_ctx["p_rel"]
        self.assertLess(on, off)  # NPY/Dyn/AEA/Gal супрессия P_release

    def test_relay_upstream_unchanged(self):
        self.assertEqual(_run(False).life.last_relay_norm, _run(True).life.last_relay_norm)


class VolumeFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_no_ltp_field_distributed_to_columns(self):
        core = _run(True)
        for col in core.moe_cortex.columns:
            self.assertTrue(hasattr(col, "_no_ltp_field"))
            self.assertEqual(tuple(col._no_ltp_field.shape), (core.life.n_neurons,))
            self.assertGreaterEqual(float(col._no_ltp_field.min()), 0.0)
            self.assertLessEqual(float(col._no_ltp_field.max()), 1.0)

    def test_telemetry_exposes_agents_and_fields(self):
        core = _run(True)
        vt = core.life.telemetry(core)["volume_chemistry"]
        self.assertTrue(vt["enabled"])
        self.assertEqual(len(vt["agents"]), 22)
        for key in ("no_mean", "h2s_mean", "ecb_mean", "nmda_gain", "p_release_scale"):
            self.assertIn(key, vt)
        self.assertTrue(vt["field_consumers"]["volume_ecb"]["connected"])
        self.assertFalse(vt["field_consumers"]["volume_h2s"]["connected"])

    def test_ejection_is_causally_driven_by_ecb_and_h2s_is_not_routed(self):
        def run(ecb_level):
            core = tiny_core()
            core.enable_integrated_organism(
                config=IntegrationConfig(volume_chemistry_enabled=True))
            col = core.moe_cortex.columns[0]
            volume = core.life.volume
            volume.ecb_field[:32].fill_(ecb_level)
            volume.attract_dur[:32].fill_(200.0)
            col._last_s = torch.ones(1, 32)
            core.life._distribute_volume_fields(
                core, SimpleNamespace(active_expert_ids=[0]))
            before = col.w_rec.u_syn.clone()
            col.step(torch.ones(1, 64) * 0.1, torch.zeros(1, 32),
                     f_mod=0.0, vip_mod=0.0, learn=False)
            return col, before

        control, control_before = run(0.0)
        ecb, ecb_before = run(0.8)
        self.assertLess(float(control._vol_eject.max()), float(ecb._vol_eject.max()))
        self.assertGreater(float(ecb._vol_eject.max()), 0.0)
        self.assertTrue(bool((control.w_rec.u_syn <= control_before).all()))
        self.assertTrue(bool((ecb.w_rec.u_syn[:32] < control.w_rec.u_syn[:32]).any()))
        self.assertTrue(bool((ecb.w_rec.u_syn < ecb_before).any()))
        self.assertFalse(hasattr(ecb, "_vol_h2s"))
        self.assertFalse(hasattr(ecb, "_vol_ecb_field"))


if __name__ == "__main__":
    unittest.main()