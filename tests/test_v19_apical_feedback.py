"""Тесты апикальной гиппокампальной обратной связи (CA3 → кора).

Контракт, который здесь фиксируется:
  * канал ВЫКЛЮЧЕН по умолчанию и не меняет ни одного логита;
  * эпизод опознаётся строго по полюсу CA3 (чужая схема → тишина);
  * гиппокампальная траектория хранит ПОРЯДОК ответа, а не мешок слов;
  * апикальный код — функция элемента последовательности, а не позиции,
    поэтому кортикальный readout обучается обобщаемому переходу;
  * реплей обучает ТОЛЬКО эпизодическую матрицу `w_swr_dend_out`,
    базовый ридеаут `w_dend_out` неприкосновенен.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from test_v19_spiking_language import tiny_core
from train_ziwago_v14_pure_synaptic_dialogue import DendriticPlateauMatrixReadout


def readout_with_episode(seed: int = 1405) -> DendriticPlateauMatrixReadout:
    ro = DendriticPlateauMatrixReadout(d_vec=64, d_dend=256, max_pos=16, seed=seed)
    pole = F.normalize(torch.randn(64, generator=torch.Generator().manual_seed(seed)), dim=0)
    ro.w_ca3_attractors = pole.unsqueeze(0).clone()
    ro.w_ca3_sdr = torch.zeros((1, 2048))
    ro.w_ca3_salience = torch.tensor([1.5])
    ro.col_to_tok = [10, 11, 12, 13]
    ro.tok_to_col = {10: 0, 11: 1, 12: 2, 13: 3}
    ro.w_dend_out = torch.zeros(256, 4)
    ro.w_core_dend_out = torch.zeros(256, 4)
    ro.w_swr_dend_out = torch.zeros(256, 4)
    return ro


class ApicalFeedbackTests(unittest.TestCase):
    def test_episode_is_resolved_only_by_matching_pole(self):
        ro = readout_with_episode()
        pole = ro.w_ca3_attractors[0]
        index, resonance = ro.resolve_episode_pole(pole)
        self.assertEqual(index, 0)
        self.assertGreater(resonance, 0.99)
        stranger = F.normalize(torch.randn(64, generator=torch.Generator().manual_seed(7)), dim=0)
        index, resonance = ro.resolve_episode_pole(stranger)
        self.assertIsNone(index)
        self.assertLess(resonance, ro.apical_min_resonance)

    def test_trace_stores_order_and_roundtrips(self):
        ro = readout_with_episode()
        ro.upsert_episode_trace(0, [10, 11, 12])
        self.assertEqual(ro.episode_trace(0), [10, 11, 12])
        self.assertEqual(int(ro.w_episode_trace_len[0]), 3)
        # перезапись заменяет содержимое, а не дописывает
        ro.upsert_episode_trace(0, [13, 10])
        self.assertEqual(ro.episode_trace(0), [13, 10])
        # неизвестный эпизод даёт пустую траекторию, а не исключение
        self.assertEqual(ro.episode_trace(7), [])

    def test_trace_is_truncated_to_position_budget(self):
        ro = readout_with_episode()
        ro.upsert_episode_trace(0, list(range(10, 10 + ro.max_pos + 5)))
        self.assertEqual(len(ro.episode_trace(0)), ro.max_pos)

    def test_apical_code_depends_on_cue_only(self):
        ro = readout_with_episode()
        cue = F.normalize(torch.randn(64, generator=torch.Generator().manual_seed(3)), dim=0)
        first = ro.apical_episode_code(cue)
        second = ro.apical_episode_code(cue)
        self.assertTrue(torch.equal(first, second))
        # разреженность и нормировка сохраняются
        self.assertEqual(int((first != 0).sum()), ro.apical_topk)
        self.assertAlmostEqual(float(first.norm()), 1.0, places=5)
        other = F.normalize(torch.randn(64, generator=torch.Generator().manual_seed(4)), dim=0)
        self.assertFalse(torch.equal(first, ro.apical_episode_code(other)))

    def test_readout_components_are_unchanged_without_apical_channel(self):
        ro = readout_with_episode()
        d = torch.zeros(256)
        d[[1, 5, 9]] = 0.5
        components = ro._readout_components(d, ro.w_ca3_attractors[0], 4)
        self.assertEqual(set(components), {"dendritic", "core", "swr", "bank"})

    def test_replay_touches_only_episodic_matrix(self):
        ro = readout_with_episode()
        basal_before = ro.w_dend_out.clone()
        core_before = ro.w_core_dend_out.clone()
        cue = F.normalize(torch.randn(64, generator=torch.Generator().manual_seed(5)), dim=0)
        stats = ro.imprint_apical_replay([cue, cue], [10, 11], f_mod=1.5)
        self.assertEqual(stats["apical_replay_updates"], 2)
        self.assertTrue(torch.equal(basal_before, ro.w_dend_out))
        self.assertTrue(torch.equal(core_before, ro.w_core_dend_out))
        self.assertGreater(float(ro.w_swr_dend_out.abs().max()), 0.0)

    def test_replay_is_inert_without_learning_or_signal(self):
        ro = readout_with_episode()
        before = ro.w_swr_dend_out.clone()
        cue = F.normalize(torch.randn(64, generator=torch.Generator().manual_seed(6)), dim=0)
        ro.learning_enabled = False
        stats = ro.imprint_apical_replay([cue], [10], f_mod=1.5)
        self.assertEqual(stats["apical_replay_updates"], 0)
        self.assertTrue(torch.equal(before, ro.w_swr_dend_out))

    def test_checkpoint_roundtrip_preserves_trace(self):
        ro = readout_with_episode()
        ro.upsert_episode_trace(0, [10, 11, 12])
        state = ro.export_state_dict()
        self.assertIn("w_episode_trace", state)
        restored = DendriticPlateauMatrixReadout(d_vec=64, d_dend=256, max_pos=16, seed=1405)
        restored.load_state_dict(state)
        self.assertEqual(restored.episode_trace(0), [10, 11, 12])

    def test_old_checkpoint_without_trace_loads_and_keeps_channel_silent(self):
        ro = readout_with_episode()
        state = ro.export_state_dict()
        state.pop("w_episode_trace")
        state.pop("w_episode_trace_len")
        restored = DendriticPlateauMatrixReadout(d_vec=64, d_dend=256, max_pos=16, seed=1405)
        restored.load_state_dict(state)
        self.assertEqual(restored.episode_trace(0), [])

    def test_organism_channel_is_off_by_default(self):
        core = tiny_core()
        self.assertFalse(core.apical_feedback_enabled)
        ro = core.dendritic_plateau_readout
        pole = F.normalize(torch.randn(64, generator=torch.Generator().manual_seed(11)), dim=0)
        ro.w_ca3_attractors = pole.unsqueeze(0).clone()
        ro.upsert_episode_trace(0, [1, 2, 3])
        # при выключенном канале контекст апикальной связи не собирается вовсе
        self.assertIsNone(core._language_apical_context(pole))

    def test_organism_cue_follows_trajectory(self):
        core = tiny_core()
        core.apical_feedback_enabled = True
        ro = core.dendritic_plateau_readout
        pole = F.normalize(torch.randn(64, generator=torch.Generator().manual_seed(12)), dim=0)
        ro.w_ca3_attractors = pole.unsqueeze(0).clone()
        ro.upsert_episode_trace(0, [1, 2, 3])
        context = core._language_apical_context(pole)
        self.assertIsNotNone(context)
        self.assertEqual(context["pole_idx"], 0)
        # позиция 0: куда двигаться — сам стимул (полюс CA3)
        cue0 = core._language_apical_cue(context, 0)
        self.assertTrue(torch.equal(cue0, ro.w_ca3_attractors[0]))
        # позиция 1: предыдущий элемент траектории
        cue1 = core._language_apical_cue(context, 1)
        self.assertTrue(torch.equal(cue1, core.get_vec64(1)))
        # за пределами траектории канал молчит
        self.assertIsNone(core._language_apical_cue(context, 9))

    def test_episode_conflict_is_bounded_and_zero_for_identical_evidence(self):
        core = tiny_core()
        logits = torch.tensor([2.0, 1.0, 0.5, 0.0])
        self.assertAlmostEqual(core._language_js_divergence(
            logits.softmax(0), logits.softmax(0)), 0.0, places=6)
        opposite = torch.tensor([0.0, 0.5, 1.0, 2.0])
        value = core._language_js_divergence(logits.softmax(0), opposite.softmax(0))
        self.assertGreater(value, 0.0)
        self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
