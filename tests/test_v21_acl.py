"""Этап E: ACL — тайловый глиальный контроль (blueprint §3.4 / PHY-GLIA-002).

Проверяется исправленная физика: одно затухание с измеримой tau=tau_C, drive
нормирован на размер тайла, консервативная нормированная диффузия, bounded
sigmoid-гейт, режимы off/phantom/on, экспозиционный интеграл G, PNN-переходы с
readiness/retention и квотой пластичных тайлов (без глобального замораживания),
адресное переоткрытие и сериализация. Живая проводка гейта — отдельный подшаг.
"""
import math
import unittest

import torch

from ziwago.cortex.tiled_glia import TiledGlialControl


class ACLPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_measured_tau_equals_configured_single_decay(self):
        glia = TiledGlialControl(n_tiles=4, tau_c=2500.0, mode="on")
        c0 = torch.ones(4)
        target = torch.zeros(4)
        dt = 100.0
        c1 = glia.reaction(c0, target, dt)
        torch.testing.assert_close(c1, c0 * math.exp(-dt / 2500.0), rtol=1e-6, atol=1e-7)
        # Effective tau recovered from two time points equals tau_c (no hidden decay).
        c2 = glia.reaction(c1, target, dt)
        tau_eff = -2 * dt / math.log(float(c2[0]) / float(c0[0]))
        self.assertAlmostEqual(tau_eff, 2500.0, delta=0.5)

    def test_drive_is_tile_size_independent(self):
        a = TiledGlialControl(n_tiles=4, tile_size=16, mode="on")
        b = TiledGlialControl(n_tiles=4, tile_size=32, mode="on")
        spikes_a = torch.full((4, 16), 0.5)
        spikes_b = torch.full((4, 32), 0.5)
        for _ in range(5):
            a.step(spikes_a, dt=50.0)
            b.step(spikes_b, dt=50.0)
        torch.testing.assert_close(a.C, b.C, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(a.G, b.G, rtol=1e-6, atol=1e-7)

    def test_diffusion_conserves_mean_and_is_stability_limited(self):
        glia = TiledGlialControl(n_tiles=8, diffusion=0.4)
        c = torch.rand(8) + 0.1
        for alpha in (0.0, 0.1, 0.5, 5.0):
            out = glia.diffuse(c, alpha)
            self.assertAlmostEqual(float(out.mean()), float(c.mean()), places=6)
        # Clamped coefficient keeps the explicit step stable (no amplification).
        out = glia.diffuse(c, 5.0)
        self.assertLessEqual(float(out.max()), float(c.max()) + 1e-6)
        self.assertGreaterEqual(float(out.min()), float(c.min()) - 1e-6)

    def test_calcium_gate_and_exposure_are_bounded(self):
        glia = TiledGlialControl(n_tiles=4, mode="on")
        gen = torch.Generator().manual_seed(0)
        prev_g = glia.G.clone()
        for _ in range(20):
            spikes = torch.rand(4, 16, generator=gen)
            gate = glia.step(spikes, dt=50.0)
            self.assertGreaterEqual(float(glia.C.min()), 0.0)
            self.assertTrue(torch.all(gate >= 0.0) and torch.all(gate <= 1.0))
            self.assertTrue(torch.all(glia.G >= prev_g - 1e-6))
            prev_g = glia.G.clone()
        dose = glia.gate_from_calcium(torch.tensor([0.0, 0.1, 0.2, 0.3]))
        self.assertTrue(torch.all(dose[1:] > dose[:-1]))
        self.assertTrue(torch.all(dose > 0.0) and torch.all(dose < 1.0))
        extreme = glia.gate_from_calcium(torch.tensor([0.0, 10.0]))
        self.assertTrue(torch.all(extreme >= 0.0) and torch.all(extreme <= 1.0))

    def test_off_phantom_on_modes(self):
        spikes = torch.full((4, 16), 0.8)
        off = TiledGlialControl(n_tiles=4, mode="off")
        gate_off = off.step(spikes, dt=50.0)
        self.assertTrue(torch.all(gate_off == 1.0))
        self.assertEqual(float(off.C.sum()), 0.0)
        phantom = TiledGlialControl(n_tiles=4, mode="phantom")
        gate_ph = phantom.step(spikes, dt=50.0)
        self.assertTrue(torch.all(gate_ph == 1.0))
        self.assertGreater(float(phantom.C.sum()), 0.0)
        on = TiledGlialControl(n_tiles=4, mode="on")
        gate_on = on.step(spikes, dt=50.0)
        self.assertLess(float(gate_on.mean()), 1.0)


class ACLConsolidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _glia(self, **kw):
        glia = TiledGlialControl(n_tiles=4, theta1=50.0, theta2=250.0, **kw)
        glia.G = torch.full((4,), 300.0)
        return glia

    def test_hard_closure_requires_readiness_and_retention(self):
        no_readiness = self._glia(min_plastic_fraction=0.0)
        no_readiness.update_states(None, retention_ok=True)
        self.assertEqual(int((no_readiness.state == TiledGlialControl.HARD).sum()), 0)
        no_retention = self._glia(min_plastic_fraction=0.0)
        no_retention.update_states(torch.ones(4), retention_ok=False)
        self.assertEqual(int((no_retention.state == TiledGlialControl.HARD).sum()), 0)
        allowed = self._glia(min_plastic_fraction=0.0)
        allowed.update_states(torch.ones(4), retention_ok=True)
        self.assertEqual(int((allowed.state == TiledGlialControl.HARD).sum()), 4)

    def test_plastic_quota_forbids_global_freezing(self):
        glia = self._glia(min_plastic_fraction=0.25)
        glia.update_states(torch.ones(4), retention_ok=True)
        self.assertGreaterEqual(int((glia.state == TiledGlialControl.PLASTIC).sum()), 1)
        half = self._glia(min_plastic_fraction=0.5)
        half.update_states(torch.ones(4), retention_ok=True)
        self.assertGreaterEqual(int((half.state == TiledGlialControl.PLASTIC).sum()), 2)

    def test_addressable_reopening_resets_exposure(self):
        glia = self._glia(min_plastic_fraction=0.0)
        glia.update_states(torch.ones(4), retention_ok=True)
        self.assertEqual(int((glia.state == TiledGlialControl.HARD).sum()), 4)
        n = glia.reopen([0, 1])
        self.assertEqual(n, 2)
        self.assertEqual(float(glia.G[0]), 0.0)
        self.assertEqual(float(glia.G[1]), 0.0)
        self.assertEqual(int(glia.state[0]), TiledGlialControl.PLASTIC)
        self.assertEqual(int(glia.state[2]), TiledGlialControl.HARD)
        with self.assertRaises(ValueError):
            glia.reopen([4])

    def test_sleep_does_not_accumulate_exposure(self):
        glia = TiledGlialControl(n_tiles=4, mode="on")
        spikes = torch.full((4, 16), 0.8)
        glia.step(spikes, dt=50.0, awake=True)
        g_awake = glia.G.clone()
        glia.step(spikes, dt=50.0, awake=False)
        torch.testing.assert_close(glia.G, g_awake, rtol=0, atol=0)

    def test_export_load_round_trip_and_shape_guard(self):
        glia = TiledGlialControl(n_tiles=4, mode="on")
        gen = torch.Generator().manual_seed(1)
        for _ in range(5):
            glia.step(torch.rand(4, 16, generator=gen), dt=50.0, readiness=torch.ones(4))
        state = glia.export_state_dict()
        restored = TiledGlialControl(n_tiles=4)
        restored.load_state_dict(state)
        torch.testing.assert_close(restored.C, glia.C, rtol=0, atol=0)
        torch.testing.assert_close(restored.G, glia.G, rtol=0, atol=0)
        torch.testing.assert_close(restored.state, glia.state, rtol=0, atol=0)
        self.assertEqual(restored.telemetry(), glia.telemetry())
        bad = dict(state)
        bad["C"] = torch.zeros(3)
        with self.assertRaises(ValueError):
            TiledGlialControl(n_tiles=4).load_state_dict(bad)


if __name__ == "__main__":
    unittest.main()