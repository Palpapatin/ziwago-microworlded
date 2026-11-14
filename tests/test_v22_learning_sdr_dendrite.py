"""Трек D: честность обучения и локальных вычислений (v22).

Каждый тест написан так, чтобы на ИСХОДНОМ коде падать (красный) и становиться
зелёным только после честной правки. Проверяются:

  * `core/sdr.py`      — Hebbian-обновление микро-канала должно попадать в тот же
                          канал, по которому выбирался победитель (раньше — мимо);
                          обучение должно быть достижимым и наблюдаемым;
                          аннотации `List`/`Dict` должны разрешаться;
  * `core/dendrite.py` — соматическая ошибка e_soma = S_i − φ(V*_D,i) должна
                          считаться модулем и потребляться правилом обучения;
                          привязка афферента к ветви должна действовать в форварде;
  * `cortex/astrocyte.py` + `core/synapse.py` — астроцитарный модуль должен быть
                          безопасным без ручного reset_state() и реально модулировать
                          состояние синапса `stp_gain`;
  * `learning/*`       — явный запрет суррогатных градиентов, честный EWC-контракт,
                          экспорт действительно подключённых модулей, измеримый
                          эффект Fisher/MCA на уровне модуля.

Тесты быстрые (секунды), без обучения и без чекпоинтов.
"""
from __future__ import annotations

import math
import sys
import typing
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ziwago
from ziwago.config import USSynapseCfg
from ziwago.core.dendrite import DendriticTree
from ziwago.core.sdr import (
    CompositionalVSABPEEncoder,
    HashedCorticalSDREncoder,
)
from ziwago.core.synapse import GenesisSynapse
from ziwago.cortex.astrocyte import AstrocyteMicrodomain


# ---------------------------------------------------------------------------
# core/sdr.py — Hebbian-обновление микро-каналов
# ---------------------------------------------------------------------------
def _expected_winners(enc: HashedCorticalSDREncoder, token_ids: torch.Tensor):
    """Независимый пересчёт окна/фазы скорера (документированный алгоритм).

    Возвращает (winner_bins, win_chan, phase_key, top_vals) — те самые индексы,
    по которым энкодер ЧИТАЕТ веса при выборе победителей.
    """
    t_long = token_ids.long().view(-1, 1) + 1
    h1_raw = t_long * enc.hash_primes.unsqueeze(0) + enc.hash_offsets.unsqueeze(0)
    h1 = (h1_raw ^ (h1_raw >> 13) ^ (h1_raw >> 7)).abs()
    h2_raw = t_long * enc.hash_primes2.unsqueeze(0) + (enc.hash_offsets.unsqueeze(0) * 31)
    h2 = ((h2_raw ^ (h2_raw >> 11) ^ (h2_raw >> 5)).abs() * 2 + 1)
    start_bins = h1 % enc.bins_per_module
    step_bins = h2 % enc.bins_per_module
    w_idx = enc.window_arange.view(1, 1, -1)
    win_bins = (start_bins.unsqueeze(-1) + w_idx * step_bins.unsqueeze(-1)) % enc.bins_per_module
    phase_key = (((h1 >> 8) ^ (h2 >> 4)) % enc.d_micro).unsqueeze(-1)
    mod_idx = torch.arange(enc.n_modules, dtype=torch.long).view(1, enc.n_modules, 1)
    local_scores = enc.micro_weights[mod_idx, win_bins, (phase_key + w_idx * 5) % enc.d_micro]
    top_vals, top_local_idx = local_scores.topk(enc.active_per_module, dim=-1)
    winner_bins = torch.gather(win_bins, dim=-1, index=top_local_idx)
    win_chan = (phase_key + top_local_idx * 5) % enc.d_micro
    return winner_bins, win_chan, phase_key, top_vals


class SDRHebbianWriteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _encoder(self) -> HashedCorticalSDREncoder:
        return HashedCorticalSDREncoder(
            vocab_size=64, n_sdr=64, n_modules=2, active_per_module=3,
            d_micro=32, window_size=8, seed=7,
        )

    def test_hebbian_update_hits_the_channel_used_for_scoring(self):
        """Обновляться должен ТОТ микро-канал, по которому выбирался победитель."""
        enc = self._encoder()
        seed_reference = self._encoder()
        tokens = torch.tensor([1, 5, 9, 17])
        before = enc.micro_weights.clone()
        self.assertTrue(torch.equal(before, seed_reference.micro_weights),
                        "до обучения матрица должна совпадать с детерминированной seed-инициализацией")
        winner_bins, win_chan, _phase_key, top_vals = _expected_winners(enc, tokens)

        enc.encode_tokens(tokens, learn=True, lr=0.5)

        after = enc.micro_weights
        self.assertFalse(torch.equal(after, seed_reference.micro_weights),
                         "после learn=True seed-матрица должна измениться")
        changed = {(int(m), int(b), int(c)) for m, b, c in (after != before).nonzero().tolist()}
        expected: dict[tuple[int, int, int], float] = {}
        for b in range(tokens.shape[0]):
            for m in range(enc.n_modules):
                for k in range(enc.active_per_module):
                    key = (int(m), int(winner_bins[b, m, k]), int(win_chan[b, m, k]))
                    expected[key] = expected.get(key, 0.0) + 0.5 * (1.0 - float(top_vals[b, m, k]))
        self.assertTrue(expected, "скор должен что-то выбрать")
        self.assertEqual(changed, set(expected))
        for key, delta in expected.items():
            m, b, c = key
            self.assertAlmostEqual(
                float(after[m, b, c]), float(before[m, b, c]) + delta, places=6,
                msg=f"канал {key} должен получить ровно Hebbian-приращение",
            )

    def test_learn_false_does_not_touch_micro_weights(self):
        enc = self._encoder()
        before = enc.micro_weights.clone()
        enc.encode_tokens(torch.tensor([1, 2, 3]), learn=False)
        self.assertTrue(torch.equal(enc.micro_weights, before))

    def test_training_path_is_observable(self):
        """Путь обучения должен быть достижим и наблюдаем (счётчик обновлений)."""
        enc = self._encoder()
        self.assertEqual(int(getattr(enc, "learn_updates", -1)), 0)
        enc.encode_tokens(torch.tensor([1, 2]), learn=True)
        self.assertEqual(int(enc.learn_updates), 1)
        enc.encode_tokens(torch.tensor([3]), learn=True)
        self.assertEqual(int(enc.learn_updates), 2)

    def test_state_dict_carries_learned_micro_weights(self):
        enc = self._encoder()
        enc.encode_tokens(torch.tensor([4, 8]), learn=True)
        state = enc.export_state_dict()
        restored = self._encoder()
        restored.load_state_dict(state)
        self.assertTrue(torch.equal(restored.micro_weights, enc.micro_weights))

    def test_type_hints_resolve(self):
        """Аннотации `List`/`Dict` должны разрешаться (иначе латентный NameError)."""
        hints = typing.get_type_hints(CompositionalVSABPEEncoder.decompose_to_base_tokens)
        self.assertIn("return", hints)
        hints = typing.get_type_hints(CompositionalVSABPEEncoder.memory_footprint_report)
        self.assertIn("return", hints)
        hints = typing.get_type_hints(CompositionalVSABPEEncoder._vsa_bind_bins)
        self.assertIn("return", hints)


# ---------------------------------------------------------------------------
# core/dendrite.py — соматическая ошибка и привязка афферентов
# ---------------------------------------------------------------------------
class DendriticSomaticErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _tree(self, **kw) -> DendriticTree:
        params = dict(n_neurons=3, n_in=5, n_branch=3, seed=1)
        params.update(kw)
        return DendriticTree(**params)

    def test_somatic_error_is_computed_by_the_module(self):
        """e_soma = S_i − φ(V*_D,i) считается модулем, а не приходит извне."""
        tree = self._tree()
        x = torch.randn(2, 5)
        spikes = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        tree.step(x, spikes)
        err = tree.somatic_error()
        pred = tree.somatic_prediction()
        self.assertEqual(tuple(err.shape), (2, 3))
        self.assertEqual(tuple(pred.shape), (2, 3))
        self.assertTrue(torch.allclose(err, spikes - pred, atol=1e-6))
        fired = spikes > 0
        self.assertTrue(bool((err[fired] > 0).all()), "у вспыхнувшего нейрона ошибка > 0")
        self.assertTrue(bool((err[~fired] < 0).all()), "у молчавшего нейрона ошибка < 0")

    def test_learn_without_external_error_uses_internal_somatic_error(self):
        tree = self._tree()
        x = torch.randn(2, 5)
        spikes = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        tree.step(x, spikes)
        alpha_before = tree.alpha.clone()
        magnitude = tree.learn(error=None, x_in=x)
        self.assertGreater(float(magnitude), 0.0)
        self.assertFalse(torch.equal(tree.alpha, alpha_before))

    def test_alpha_update_follows_sign_of_internal_somatic_error(self):
        """Внутренняя соматическая ошибка реально потребляется правилом обучения."""
        tree = self._tree()
        x = torch.randn(2, 5)
        spikes = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        tree.step(x, spikes)
        err = tree.somatic_error().mean(0)
        alpha_before = tree.alpha.clone()
        tree.learn(error=None, x_in=x)
        d_alpha = (tree.alpha - alpha_before).mean(1)
        for i in range(3):
            if float(err[i]) > 0:
                self.assertGreater(float(d_alpha[i]), 0.0, f"нейрон {i}: e>0 ⇒ усиление α")
            else:
                self.assertLess(float(d_alpha[i]), 0.0, f"нейрон {i}: e<0 ⇒ ослабление α")

    def test_afferent_branch_binding_is_active_in_forward(self):
        """Афферент, привязанный к другой ветви, не должен питать эту ветвь."""
        tree = DendriticTree(n_neurons=2, n_in=4, n_branch=3, seed=0,
                              enable_deep_7level=False)
        tree.w_syn.zero_()
        tree.w_syn[0, 1] = 50.0            # вес «ветвь 0 ← афферент 1»
        tree.mask = torch.tensor([0, 1, 2, 0])   # афферент 1 привязан к ветви 1
        spikes = torch.zeros(1, 2)

        x_hit = torch.zeros(1, 4)
        x_hit[0, 1] = 1.0
        x_zero = torch.zeros(1, 4)

        tree.reset_state(1)
        out_hit = tree.step(x_hit, spikes)
        tree.reset_state(1)
        out_zero = tree.step(x_zero, spikes)

        self.assertTrue(
            torch.allclose(out_hit, out_zero, atol=1e-6),
            "привязка афферента обязана действовать в форварде",
        )

    def test_routed_afferent_reaches_its_own_branch(self):
        tree = DendriticTree(n_neurons=2, n_in=4, n_branch=3, seed=0,
                              enable_deep_7level=False)
        tree.w_syn.zero_()
        tree.w_syn[1, 1] = 50.0            # вес «ветвь 1 ← афферент 1»
        tree.mask = torch.tensor([0, 1, 2, 0])
        x_hit = torch.zeros(1, 4)
        x_hit[0, 1] = 1.0
        tree.reset_state(1)
        out_hit = tree.step(x_hit, torch.zeros(1, 2))
        tree.reset_state(1)
        out_zero = tree.step(torch.zeros(1, 4), torch.zeros(1, 2))
        self.assertFalse(torch.allclose(out_hit, out_zero, atol=1e-6))

    def test_theta_ca_shift_is_applicable_to_dendrite_state(self):
        tree = self._tree()
        base = float(tree.theta_ca)
        new = tree.apply_theta_ca_shift(-4.0)
        self.assertAlmostEqual(float(tree.theta_ca), base - 4.0, places=6)
        self.assertAlmostEqual(float(new), float(tree.theta_ca), places=6)


# ---------------------------------------------------------------------------
# cortex/astrocyte.py + core/synapse.py — реальное состояние stp_gain
# ---------------------------------------------------------------------------
class AstrocyteStpGainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _synapse(self) -> GenesisSynapse:
        return GenesisSynapse(n_pre=4, n_post=2, cfg=USSynapseCfg(), seed=3)

    def test_astrocyte_is_usable_without_manual_reset(self):
        astro = AstrocyteMicrodomain(n_neurons=4)
        self.assertIsNotNone(astro.ca)
        spikes = torch.zeros(1, 4)
        spikes[0, 0] = 1.0
        gate = astro.step(spikes)
        self.assertEqual(tuple(gate.shape), (1, 4))
        self.assertGreaterEqual(float(gate.min()), 1.0)
        self.assertLessEqual(float(gate.max()), 1.5)
        self.assertGreater(float(gate[0, 0]), 1.0)
        # смена размера батча не должна ломать модуль
        gate2 = astro.step(torch.zeros(3, 4))
        self.assertEqual(tuple(gate2.shape), (3, 4))

    def test_synapse_stp_gain_is_real_state(self):
        syn = self._synapse()
        self.assertAlmostEqual(float(syn.stp_gain), 1.0, places=6)
        x = torch.tensor([1.0, 0.0, 0.0, 0.0])
        base = syn.drive(x).clone()
        syn.set_stp_gain(1.25)
        self.assertAlmostEqual(float(syn.stp_gain), 1.25, places=6)
        scaled = syn.drive(x)
        self.assertTrue(torch.allclose(scaled, base * 1.25, atol=1e-6))
        with self.assertRaises(ValueError):
            syn.set_stp_gain(float("nan"))

    def test_astrocyte_modulates_real_synapse_state(self):
        astro = AstrocyteMicrodomain(n_neurons=2)
        spikes = torch.zeros(1, 2)
        spikes[0, 0] = 1.0
        gate = astro.step(spikes)
        syn = self._synapse()
        syn.mask[:, 0] = True
        syn.w_slow[:, 0] = torch.tensor([0.5, 0.25])
        x = torch.tensor([1.0, 0.0, 0.0, 0.0])
        before = syn.drive(x).clone()
        touched = astro.apply_to_synapses([syn], gate)
        self.assertEqual(int(touched), 1)
        self.assertGreater(float(syn.stp_gain), 1.0)
        self.assertFalse(torch.equal(syn.drive(x), before))

    def test_astrocyte_theta_ca_shift_applies_to_dendrite(self):
        astro = AstrocyteMicrodomain(n_neurons=2)
        astro.step(torch.tensor([[1.0, 0.0]]))
        tree = DendriticTree(n_neurons=2, n_in=4, n_branch=3, seed=0)
        base = float(tree.theta_ca)
        applied = astro.apply_theta_ca_shift(tree)
        self.assertAlmostEqual(float(tree.theta_ca), base + applied, places=6)
        self.assertLess(applied, 0.0)


# ---------------------------------------------------------------------------
# learning/* — контракты пакета
# ---------------------------------------------------------------------------
class LearningPackageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_package_exports_the_modules_wired_into_the_live_readout(self):
        import ziwago.learning as L
        self.assertIn("SparseFisherAnchor", L.__all__)
        self.assertIn("MetaplasticSusceptibility", L.__all__)
        self.assertTrue(hasattr(L, "SparseFisherAnchor"))
        self.assertTrue(hasattr(L, "MetaplasticSusceptibility"))

    def test_surrogate_requires_explicit_experimental_permission(self):
        from ziwago.learning.surrogate import hebbian_step, spike_fn
        x = torch.tensor([-1.0, 0.5, 2.0])
        with self.assertRaises(RuntimeError) as ctx:
            spike_fn(x)
        self.assertIn("surrogate", str(ctx.exception).lower())
        with self.assertRaises(RuntimeError):
            hebbian_step(x)
        # явное разрешение эксперимента — работает
        out = spike_fn(x, "fast_sigmoid", 10.0, allow_experimental=True)
        self.assertTrue(torch.equal(out, torch.tensor([0.0, 1.0, 1.0])))

    def test_ewc_requires_explicit_backward_permission(self):
        from ziwago.learning.ewc import EWCRegularizer
        p = nn.Parameter(torch.ones(4))
        ewc = EWCRegularizer({"w": p})
        with self.assertRaises(RuntimeError) as ctx:
            ewc.estimate_fisher(lambda: (p * 2.0).sum(), n_samples=2)
        self.assertIn("backward", str(ctx.exception).lower())
        mass = ewc.estimate_fisher(lambda: (p * 2.0).sum(), n_samples=2,
                                   allow_backward=True)
        self.assertGreater(float(mass), 0.0)

    def test_mca_spike_kick_is_accounted_as_manual_api(self):
        from ziwago.learning.mca import MetaplasticSusceptibility
        mca = MetaplasticSusceptibility()
        mca.spike_kick([(0, 0)], 0.3)
        self.assertEqual(int(mca.telemetry()["manual_kicks"]), 1)

    def test_mca_rejects_non_finite_weights(self):
        from ziwago.learning.mca import MetaplasticSusceptibility
        mca = MetaplasticSusceptibility()
        w = torch.zeros(1, 3)
        w[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            mca.observe(w, torch.tensor([1.0]), torch.tensor([0.5, 0.1]), [0], 0, dt=1.0)

    def test_fisher_rejects_non_finite_logit_scale(self):
        from ziwago.learning.local_fisher import SparseFisherAnchor
        anchor = SparseFisherAnchor()
        with self.assertRaises(ValueError):
            anchor.observe(torch.zeros(1, 3), torch.tensor([1.0]),
                           torch.tensor([0.5, 0.1]), float("nan"), [0], 0)

    def test_fisher_anchor_changes_weight_trajectory_at_module_level(self):
        """Измеримый эффект: якорь Fisher меняет дрейф весов (без ридеута)."""
        from ziwago.learning.local_fisher import SparseFisherAnchor
        torch.manual_seed(0)
        w_plain = torch.zeros(3, 4)
        w_anchor = w_plain.clone()
        anchor = SparseFisherAnchor(max_entries=64, decay=0.5)
        d_rows = torch.tensor([0.8, -0.4, 0.6])
        errors = [torch.tensor([0.9, -0.6, 0.2, 0.1]),
                  torch.tensor([0.7, 0.5, -0.3, 0.2]),
                  torch.tensor([0.4, -0.2, 0.8, -0.1])]
        for err in errors:
            anchor.observe(w_anchor, d_rows, err, 12.0, [0, 1, 2], 0, top_columns=4)
            grad = 12.0 * d_rows.unsqueeze(1) * err.unsqueeze(0)
            w_plain = w_plain + 0.05 * grad
            w_anchor = w_anchor + 0.05 * grad
            anchor.proximal_pull(w_anchor, eta=1.0, lambda_reg=1.0)
        drift_plain = float((w_plain - torch.zeros(3, 4)).norm())
        drift_anchor = float((w_anchor - torch.zeros(3, 4)).norm())
        self.assertGreater(anchor.telemetry()["entries"], 0)
        self.assertNotAlmostEqual(drift_plain, drift_anchor, places=6)

    def test_mca_gain_changes_weight_trajectory_at_module_level(self):
        """Измеримый эффект: гейт MCA меняет дрейф весов (без ридеута)."""
        from ziwago.learning.mca import MetaplasticSusceptibility
        torch.manual_seed(0)
        w_plain = torch.zeros(3, 4)
        w_gated = w_plain.clone()
        mca = MetaplasticSusceptibility(max_entries=64, tau_m=1.0, beta_m=4.0)
        d_rows = torch.tensor([0.8, -0.4, 0.6])
        errors = [torch.tensor([0.9, -0.6, 0.2, 0.1]),
                  torch.tensor([0.7, 0.5, -0.3, 0.2]),
                  torch.tensor([0.4, -0.2, 0.8, -0.1])]
        for err in errors:
            mca.observe(w_gated, d_rows, err, [0, 1, 2], 0, dt=1.0, top_columns=4)
            gain = mca.row_susceptibility([0, 1, 2], n_active=4)
            grad = 12.0 * d_rows.unsqueeze(1) * err.unsqueeze(0)
            w_plain = w_plain + 0.05 * grad
            w_gated = w_gated + 0.05 * grad * gain.unsqueeze(1)
        drift_plain = float((w_plain - torch.zeros(3, 4)).norm())
        drift_gated = float((w_gated - torch.zeros(3, 4)).norm())
        self.assertGreater(mca.telemetry()["entries"], 0)
        self.assertNotAlmostEqual(drift_plain, drift_gated, places=6)


    def test_live_multiplexer_is_the_functional_owner_of_gamma_slots(self):
        """Сравнение с живым модулем: у него те же γ-слоты, и он рабочий."""
        from ziwago.cortex.genesis_engine import ThetaGammaSpikingPhaseMultiplexer

        mux = ThetaGammaSpikingPhaseMultiplexer(d_vec=8, n_gamma_slots=6, bins_per_gamma=5)
        role_vec = torch.full((8,), 0.5)
        raster, _windows = mux.encode_roles_into_theta_spike_train({"GAMMA_0_AGENT": role_vec})
        decoded = mux.decode_roles_from_theta_spike_train(raster)
        self.assertTrue(torch.allclose(decoded["GAMMA_0_AGENT"], role_vec, atol=1e-5))
        self.assertEqual(len(mux.ROLE_SLOT_NAMES[:mux.n_gamma_slots]), 6)

        self.assertEqual(mux.theta_period_ms, 6 * 5)


if __name__ == "__main__":
    unittest.main()
