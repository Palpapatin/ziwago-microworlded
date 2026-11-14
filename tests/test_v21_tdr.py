"""Candidate-first TDR contracts; tiny fixtures, no checkpoints."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ziwago.cognition.language.readout import DendriticPlateauMatrixReadout
from ziwago.cognition.language.tdr import TokenCandidateIndex
from test_v19_spiking_language import tiny_core


class TinyEncoder:
    n_sdr = 16

    def encode_token_id(self, token_id: int) -> torch.Tensor:
        sdr = torch.zeros(self.n_sdr)
        bits = {int(token_id) % 16, (int(token_id) * 3 + 1) % 16, (int(token_id) * 5 + 2) % 16}
        sdr[list(bits)] = 1.0
        return sdr

    def project_token_to_cortex(self, token_id: int) -> torch.Tensor:
        vec = torch.zeros(4)
        vec[int(token_id) % 4] = 1.0
        return vec


class Encoder64:
    n_sdr = 64

    def encode_token_id(self, token_id: int) -> torch.Tensor:
        sdr = torch.zeros(self.n_sdr)
        t = int(token_id)
        sdr[[t % 64, (t * 7 + 3) % 64, (t * 13 + 5) % 64]] = 1.0
        return sdr

    def project_token_to_cortex(self, token_id: int) -> torch.Tensor:
        vec = torch.zeros(64)
        t = int(token_id)
        vec[t % 64] = 1.0
        vec[(t * 3 + 1) % 64] = 0.5
        return vec


class TDRIndexTests(unittest.TestCase):
    def setUp(self):
        self.enc = TinyEncoder()
        self.idx = TokenCandidateIndex(self.enc)
        self.tokens = list(range(8))

    def test_full_pool_returns_every_active_column(self):
        proposal = self.idx.propose(torch.ones(4), [0], self.tokens, budget=None)
        self.assertEqual(proposal.cols, list(range(8)))
        self.assertEqual(proposal.tokens, self.tokens)
        self.assertEqual(proposal.report["mode"], "full")

    def test_pool_a_recalls_sdr_overlap_before_scoring(self):
        proposal = self.idx.propose(torch.zeros(4), [0], self.tokens, budget=3, pool_a=3, pool_b=0)
        self.assertIn(0, proposal.cols)
        self.assertIn(2, proposal.cols)  # shares bit 2 with token 0
        self.assertGreater(proposal.report["pool_a_candidates"], 0)
        self.assertEqual(proposal.report["pool_b_candidates"], 0)
        self.assertLessEqual(len(proposal.cols), 3)

    def test_pool_b_uses_encoder_projection_cosine(self):
        q = self.enc.project_token_to_cortex(3)
        proposal = self.idx.propose(q, [], self.tokens, budget=2, pool_a=0, pool_b=2)
        self.assertIn(3, proposal.cols)
        self.assertGreater(proposal.report["pool_b_candidates"], 0)
        self.assertEqual(proposal.report["pool_a_candidates"], 0)

    def test_training_target_is_forced_but_inference_target_is_not(self):
        q = self.enc.project_token_to_cortex(3)
        infer = self.idx.propose(q, [], self.tokens, budget=1, pool_a=0, pool_b=1)
        self.assertEqual(infer.cols, [3])
        self.assertEqual(infer.report["forced_target"], 0)
        train = self.idx.propose(q, [], self.tokens, budget=1, pool_a=0, pool_b=1, target_token=0)
        self.assertEqual(train.cols, [0])
        self.assertEqual(train.report["forced_target"], 1)

    def test_zero_query_signal_refuses_without_fabricating_candidates(self):
        proposal = self.idx.propose(torch.zeros(4), [], self.tokens, budget=4)
        self.assertEqual(proposal.cols, [])
        self.assertEqual(proposal.report["reason"], "no_query_signal")

    def test_vocab_rebuild_detects_same_length_replacement(self):
        self.idx.ensure_vocab([0, 1, 2])
        self.idx.ensure_vocab([5, 6, 7])
        self.assertEqual(self.idx.token_ids, [5, 6, 7])
        self.assertEqual(set(self.idx.token_to_col), {5, 6, 7})


class TDRReadoutTests(unittest.TestCase):
    def test_candidate_components_equal_full_components_indexed(self):
        ro = DendriticPlateauMatrixReadout(d_vec=8, d_dend=16, max_pos=4, seed=3)
        for token in range(8):
            ro._ensure_token_col(token)
        gen = torch.Generator().manual_seed(5)
        ro.w_dend_out[:, :8] = torch.randn(16, 8, generator=gen)
        ro.w_core_dend_out[:, :8] = torch.randn(16, 8, generator=gen)
        ro.w_swr_dend_out[:, :8] = torch.randn(16, 8, generator=gen)
        q = F.normalize(torch.randn(8, generator=gen), dim=0)
        ro.w_swr_poles = q.unsqueeze(0)
        d = torch.zeros(16)
        d[[0, 1, 2]] = 1.0
        for token in ro.col_to_tok:
            ro.bank_router.banks[0][(0, token)] = 0.3 + 0.1 * token
        full = ro._readout_components(d, q, 8)
        cols = [1, 3, 5]
        sub = ro._readout_components(d, q, 8, cols)
        for name in full:
            self.assertTrue(torch.equal(sub[name], full[name][cols]), name)

    def test_invalid_candidate_columns_are_rejected(self):
        ro = DendriticPlateauMatrixReadout(d_vec=8, d_dend=16, max_pos=4, seed=3)
        for token in range(4):
            ro._ensure_token_col(token)
        d = torch.ones(16)
        q = F.normalize(torch.ones(8), dim=0)
        with self.assertRaises(ValueError):
            ro._readout_components(d, q, 4, [0, 0])
        with self.assertRaises(ValueError):
            ro._readout_components(d, q, 4, [4])


class TDRIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _core(self):
        core = tiny_core()
        core.vsa_encoder = Encoder64()
        ro = core.dendritic_plateau_readout
        for token in range(8):
            ro._ensure_token_col(token)
        core.spiking_language_loop.ensure_tokens(ro.col_to_tok)
        gen = torch.Generator().manual_seed(11)
        ro.w_dend_out[:, :8] = torch.randn(ro.d_dend, 8, generator=gen)
        loop = core.spiking_language_loop
        loop.w_projector[:, :8] = torch.randn(loop.d_features, 8, generator=gen) * 0.1
        return core

    def test_disabled_tdr_preserves_full_vocabulary_path(self):
        core = self._core()
        # TDR включён по умолчанию (13.11) — проверяем ЯВНЫЙ opt-out: полный
        # путь словаря сохраняется, бюджет не режет кандидатов.
        core.tdr_enabled = False
        q = F.normalize(torch.randn(64, generator=torch.Generator().manual_seed(12)), dim=0)
        ids, logits, _, _ = core._language_token_distribution(q, [], 0)
        self.assertEqual(ids, core.dendritic_plateau_readout.col_to_tok)
        self.assertEqual(logits.numel(), 8)
        self.assertIsNone(core._last_tdr_report)

    def test_enabled_tdr_scores_only_selected_columns_and_matches_full_indexed(self):
        core = self._core()
        q = F.normalize(torch.randn(64, generator=torch.Generator().manual_seed(12)), dim=0)
        # ЕДИНЫЙ КОНТУР (13.11): логиты зависят не только от матриц коры, но и
        # от состояния биологического драйвера (химия и её часы, SON-колонка,
        # сам драйвер). Поэтому для честного сравнения восстанавливаем ВСЕ
        # участников контура из одного снимка. Снимок берём по частям, а не
        # всего организма: в этом тесте `vsa_encoder` — локальный класс из
        # теста, а экспорт runtime-дерева допускает только объекты пакета.
        from ziwago.cortex.moe_cortex import export_runtime_tree, load_runtime_tree
        moe_state = core.moe_cortex.export_state_dict()
        loop_state = core.spiking_language_loop.export_dynamic_state_dict()
        chem_state = export_runtime_tree(core.chemistry)
        son_state = export_runtime_tree(core.son_column)
        life_state = export_runtime_tree(core.life)
        ids_full, logits_full, _, _ = core._language_token_distribution(q, [], 0, use_tdr=False)
        core.moe_cortex.load_state_dict(moe_state)
        core.spiking_language_loop.load_dynamic_state_dict(loop_state)
        load_runtime_tree(core.chemistry, chem_state)
        load_runtime_tree(core.son_column, son_state)
        load_runtime_tree(core.life, life_state)
        core.tdr_enabled = True
        core.tdr_candidate_budget = 4
        core.tdr_pool_a = 0
        core.tdr_pool_b = 4
        ids_sub, logits_sub, _, _ = core._language_token_distribution(q, [], 0)
        self.assertLessEqual(len(ids_sub), 4)
        self.assertEqual(logits_sub.numel(), len(ids_sub))
        cols = [core.dendritic_plateau_readout.tok_to_col[t] for t in ids_sub]
        torch.testing.assert_close(logits_sub, logits_full[cols], rtol=0, atol=0)
        report = core._last_tdr_report
        self.assertEqual(report["mode"], "candidate_pool")
        self.assertEqual(report["selected_candidates"], len(ids_sub))
        self.assertEqual(report["budget"], 4)

    def test_zero_signal_returns_empty_pool_and_evaluation_stays_full(self):
        core = self._core()
        core.tdr_enabled = True
        core.tdr_candidate_budget = 4
        ids, logits, _, _ = core._language_token_distribution(torch.zeros(64), [], 0)
        self.assertEqual(ids, [])
        self.assertEqual(logits.numel(), 0)
        self.assertEqual(core._last_tdr_report["reason"], "no_query_signal")
        q = F.normalize(torch.randn(64, generator=torch.Generator().manual_seed(13)), dim=0)
        ids_eval, logits_eval, _, _ = core._language_token_distribution(q, [], 0, use_tdr=False)
        self.assertEqual(ids_eval, core.dendritic_plateau_readout.col_to_tok)
        self.assertEqual(logits_eval.numel(), 8)


if __name__ == "__main__":
    unittest.main()