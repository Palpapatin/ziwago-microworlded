"""Причинная проверка проводки Fisher, MCA, ACL и DCC в малом ридеуте."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ziwago.cognition.language.readout import DendriticPlateauMatrixReadout


def _readout(n_tokens: int = 6, d_dend: int = 32, seed: int = 5):
    ro = DendriticPlateauMatrixReadout(d_vec=8, d_dend=d_dend, max_pos=n_tokens, seed=seed)
    for token in range(n_tokens):
        ro._ensure_token_col(token)
    ro.w_dend_out[:, :n_tokens] = torch.randn(ro.d_dend, n_tokens, generator=torch.Generator().manual_seed(seed)) * 0.1
    return ro


def _train_once(ro, n_tokens: int = 6, target_col: int = 0, f_mod: float = 1.0):
    """Один шаг обучения ридеута: NMDA-активность + ошибка + пластичность."""
    d = torch.zeros(ro.d_dend)
    d[1] = 0.9
    d[5] = 0.4
    q = torch.nn.functional.normalize(torch.ones(ro.d_vec), dim=0)
    stats = ro._learn_readout_token(d, q, target_col, f_mod)
    return stats


def _drift(ro, steps=5):
    initial = ro.w_dend_out[:, :len(ro.col_to_tok)].clone()
    for _ in range(steps):
        _train_once(ro)
    return float((ro.w_dend_out[:, :len(ro.col_to_tok)] - initial).norm())


class ReadoutWiringTests(unittest.TestCase):
    def test_modules_are_created_by_default(self):
        ro = _readout()
        stats = _train_once(ro)
        for name in ("fisher", "mca", "acl", "dcc"):
            self.assertTrue(stats[f"{name}_enabled"])
            self.assertTrue(stats[f"{name}_created"])
        self.assertFalse(stats["dcc_g_inh_connected"])
        self.assertFalse(stats["dcc_sigma_noise_connected"])

    def test_explicit_opt_out_creates_nothing(self):
        ro = _readout()
        ro.fisher_anchor_enabled = False
        ro.mca_enabled = False
        ro.acl_enabled = False
        ro.dcc_enabled = False
        stats = _train_once(ro)
        for name in ("fisher", "mca", "acl", "dcc"):
            self.assertFalse(stats[f"{name}_enabled"])
            self.assertFalse(stats[f"{name}_created"])

    def test_fisher_anchor_changes_the_update(self):
        """Якорь Фишера меняет веса (proximal pull) — измеримо, а не декларативно."""
        on = _readout()
        off = _readout()
        for ro in (on, off):
            ro.fisher_anchor_enabled = True
            ro.fisher_eta = 1.0
            ro.fisher_lambda = 10.0
            _train_once(ro)
        off.fisher_anchor_enabled = False
        drift_on = _drift(on, steps=3)
        drift_off = _drift(off, steps=3)
        self.assertGreater(drift_on, 0.0)
        self.assertNotAlmostEqual(drift_on, drift_off, places=6)

    def test_mca_changes_the_update(self):
        """MCA (метапластическая чувствительность) меняет скорость обучения.

        Эффект накапливается: у первой строки ещё нет истории наблюдений, и её
        чувствительность равна 1. Поэтому сравниваем НЕСКОЛЬКО шагов: после
        них метапластическая оценка m накоплена и гейт уже не равен 1.
        """
        on = _readout()
        off = _readout()
        for ro in (on, off):
            ro.mca_tau_m = 1.0
            ro.mca_dt = 1.0
        off.mca_enabled = False
        drift_on = _drift(on)
        drift_off = _drift(off)
        self.assertGreater(on._mca.telemetry()["mean_m"], 0.0)
        self.assertLess(drift_on, drift_off)

    def test_acl_gate_changes_the_update(self):
        """ACL (тайлованная глия) ограничивает пластичность своих тайлов."""
        on = _readout()
        for ro in (on,):
            ro.acl_tile_size = 8
            ro.acl_tau_c = 1.0
        off = _readout()
        off.acl_enabled = False
        off.acl_tile_size = 8
        drift_on = _drift(on)
        drift_off = _drift(off)
        self.assertNotAlmostEqual(drift_on, drift_off, places=6)
        self.assertGreaterEqual(on._acl.telemetry()["mean_C"], 0.0)

    def test_dcc_plasticity_is_applied_not_only_reported(self):
        """Выход DCC реально ограничивает обучение, а не только пишется в отчёт."""
        on = _readout()
        off = _readout()
        on.acl_enabled = False
        on.acl_tile_size = 8
        off.acl_enabled = False
        off.acl_tile_size = 8
        off.dcc_enabled = False
        on.dcc_tau_mature = 1.0
        drift_on = _drift(on, steps=2)
        drift_off = _drift(off, steps=2)
        self.assertIsNotNone(on._dcc)
        self.assertLess(on._dcc.effective_plasticity().mean().item(), 1.0)
        self.assertNotAlmostEqual(drift_on, drift_off, places=6)

    def test_dcc_without_acl_still_modulates_plasticity(self):
        on = _readout()
        off = _readout()
        on.acl_enabled = False
        off.acl_enabled = False
        off.dcc_enabled = False
        on.dcc_tau_mature = 1.0
        drift_on = _drift(on, steps=2)
        drift_off = _drift(off, steps=2)
        self.assertLess(on._dcc.effective_plasticity().mean().item(), 1.0)
        self.assertNotAlmostEqual(drift_on, drift_off, places=6)

    def test_module_presence_and_state_round_trip(self):
        ro = _readout()
        _train_once(ro)
        restored = _readout()
        restored.load_state_dict(ro.export_state_dict())
        for name in ("fisher", "mca", "acl", "dcc"):
            self.assertIsNotNone(getattr(restored, f"_{name if name != 'fisher' else 'fisher_anchor'}"))
            self.assertTrue(restored._readout_module_telemetry[f"{name}_created"])

    def test_telemetry_reports_effect_not_only_presence(self):
        ro = _readout()
        stats = _train_once(ro)
        for key in ("acl_min_gate", "acl_mean_gate", "dcc_min_plasticity",
                    "dcc_mean_plasticity", "mca_min_row_susceptibility"):
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], float)


if __name__ == "__main__":
    unittest.main()
