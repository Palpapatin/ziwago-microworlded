"""Интеграция живой проводки ACL-гейта в ридаут (этап E, часть 5).

Единая объявленная точка — множитель row_gain в _apply_local_update. ACL-гейт
тайловых астроцитов (drive из d_nmda) подмешивается туда же, где MCA; readiness
от DCC гейтит HARD-консолидацию. Opt-in: default off → goldens не меняются.
Проверяется причинное действие гейта на пластичность, а не только наличие модуля.
"""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from ziwago.cognition.language.readout import DendriticPlateauMatrixReadout


class ACLWiringTests(unittest.TestCase):
    def _readout(self, d_dend: int = 32) -> DendriticPlateauMatrixReadout:
        ro = DendriticPlateauMatrixReadout(d_vec=8, d_dend=d_dend, max_pos=4, seed=3)
        for token in range(8):
            ro._ensure_token_col(token)
        gen = torch.Generator().manual_seed(5)
        ro.w_dend_out[:, :8] = torch.randn(d_dend, 8, generator=gen)
        return ro

    def _steps(self, d_dend: int = 32):
        gen = torch.Generator().manual_seed(9)
        q = F.normalize(torch.randn(8, generator=gen), dim=0)
        out = []
        for target in (0, 3, 5, 1, 6):
            d = torch.zeros(d_dend)
            d[[0, 1, 2, 7]] = torch.rand(4, generator=gen) + 0.3   # tile 0 active
            out.append((d, q, target))
        return out

    def test_default_on_creates_modules_and_explicit_opt_out_is_neutral(self):
        """13.11: ACL и DCC включены по умолчанию; opt-out остался явным."""
        ro = self._readout()
        for d, q, target in self._steps():
            stats = ro._learn_readout_token(d, q, target, f_mod=1.0)
        self.assertTrue(stats["acl_enabled"])
        self.assertTrue(stats["dcc_enabled"])
        self.assertIsNotNone(ro._acl)
        self.assertIsNotNone(ro._dcc)

        ro_off = self._readout()
        ro_off.acl_enabled = False
        ro_off.dcc_enabled = False
        for d, q, target in self._steps():
            stats_off = ro_off._learn_readout_token(d, q, target, f_mod=1.0)
        self.assertEqual(stats_off["acl_min_gate"], 1.0)
        self.assertEqual(stats_off["acl_mean_gate"], 1.0)
        self.assertFalse(stats_off["acl_enabled"])
        self.assertFalse(stats_off["dcc_enabled"])
        self.assertIsNone(ro_off._acl)
        self.assertIsNone(ro_off._dcc)

    def test_acl_enabled_causally_suppresses_drift(self):
        ro_off = self._readout()
        ro_off.acl_enabled = False
        ro_on = self._readout()
        ro_on.acl_enabled = True
        ro_on.acl_tile_size = 8
        initial = ro_off.w_dend_out.clone()
        for d, q, target in self._steps():
            ro_off._learn_readout_token(d, q, target, f_mod=1.0)
            ro_on._learn_readout_token(d, q, target, f_mod=1.0)
        drift_off = float((ro_off.w_dend_out - initial).norm())
        drift_on = float((ro_on.w_dend_out - initial).norm())
        self.assertGreater(drift_off, 0.0)
        self.assertLess(drift_on, drift_off)   # gate < 1 suppresses plasticity
        self.assertIsNotNone(ro_on._acl)

    def test_acl_gate_bounded_and_reported(self):
        ro = self._readout()
        ro.acl_enabled = True
        ro.acl_tile_size = 8
        for d, q, target in self._steps():
            stats = ro._learn_readout_token(d, q, target, f_mod=1.0)
            self.assertGreaterEqual(stats["acl_min_gate"], 0.0)
            self.assertLessEqual(stats["acl_min_gate"], stats["acl_mean_gate"])
            self.assertLessEqual(stats["acl_mean_gate"], 1.0)

    def test_acl_tile_mapping_active_vs_weak_tile(self):
        ro = self._readout(d_dend=32)
        ro.acl_enabled = True
        ro.acl_tile_size = 8
        d = torch.zeros(32)
        d[[0, 1, 2, 7]] = 1.0
        q = F.normalize(torch.randn(8, generator=torch.Generator().manual_seed(9)), dim=0)
        ro._learn_readout_token(d, q, 0, f_mod=1.0)   # creates _acl
        # Calibrate the gate and isolate tiles (diffusion off): this tests the
        # row->tile mapping wiring, not ACL diffusion (covered in test_v21_acl).
        # Steady C: tile0 ~0.165 (mean drive 0.5), tile1 ~0.004 (mean drive 0.0125);
        # mu_c=0.02 sits between them so the active tile opens and the weak stays shut.
        ro._acl.mu_c = 0.02
        ro._acl.sigma_c = 0.02
        ro._acl.rho = 1.0
        ro._acl.diffusion = 0.0
        d2 = torch.zeros(32)
        d2[[0, 1, 2, 3]] = 1.0    # tile 0 strong drive
        d2[[8, 9]] = 0.05         # tile 1 weak drive
        rows = torch.nonzero(d2).flatten()
        for _ in range(20):
            gain = ro._acl_row_gain(d2, rows, 0.5)
        g_tile0 = float(gain[:4].mean())
        g_tile1 = float(gain[4:].mean())
        self.assertGreater(g_tile0, g_tile1)   # active tile opens, weak stays closed
        self.assertLessEqual(g_tile0, 1.0)
        self.assertGreaterEqual(g_tile1, 0.0)

    def test_dcc_readiness_wired_only_when_enabled(self):
        ro = self._readout()
        ro.acl_enabled = True
        ro.acl_tile_size = 8
        ro.dcc_enabled = True
        for d, q, target in self._steps():
            ro._learn_readout_token(d, q, target, f_mod=1.0)
        self.assertIsNotNone(ro._dcc)
        n_tiles = (32 + 8 - 1) // 8
        self.assertEqual(ro._dcc.n_zones, n_tiles)
        r = ro._dcc.readiness
        self.assertTrue(torch.all(r >= 0.0) and torch.all(r <= 1.0))
        # Без явного флага DCC не создаётся (opt-out, а не дефолт).
        ro2 = self._readout()
        ro2.acl_enabled = True
        ro2.acl_tile_size = 8
        ro2.dcc_enabled = False
        for d, q, target in self._steps():
            ro2._learn_readout_token(d, q, target, f_mod=1.0)
        self.assertIsNone(ro2._dcc)

    def test_acl_and_mca_compose_at_declared_point(self):
        ro_acl = self._readout()
        ro_acl.acl_enabled = True
        ro_acl.acl_tile_size = 8
        ro_both = self._readout()
        ro_both.acl_enabled = True
        ro_both.acl_tile_size = 8
        ro_both.mca_enabled = True
        ro_both.mca_tau_m = 1.0
        ro_both.mca_dt = 1.0
        ro_both.mca_top_columns = 8
        initial = ro_acl.w_dend_out.clone()
        for d, q, target in self._steps():
            ro_acl._learn_readout_token(d, q, target, f_mod=1.0)
            ro_both._learn_readout_token(d, q, target, f_mod=1.0)
        drift_acl = float((ro_acl.w_dend_out - initial).norm())
        drift_both = float((ro_both.w_dend_out - initial).norm())
        # Both gates are <= 1, so composing them cannot increase drift.
        self.assertLessEqual(drift_both, drift_acl + 1e-6)


if __name__ == "__main__":
    unittest.main()