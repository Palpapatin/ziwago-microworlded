"""Compatibility and behavioral checks for the extracted readout components."""

import inspect
import unittest
from unittest.mock import patch

import torch

from ziwago.cognition.language.readout import DendriticPlateauMatrixReadout
from ziwago.cognition.language.parts.banks import ReadoutBanks
from ziwago.cognition.language.parts.readout_math import ReadoutMath
from ziwago.cognition.language.parts.regulators import ReadoutRegulators


class ReadoutPartsTests(unittest.TestCase):
    def test_sparse_logits_preserve_sparse_candidate_scoring(self):
        readout = DendriticPlateauMatrixReadout(d_vec=4, d_dend=6, max_pos=3, seed=7)
        weights = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        activation = torch.tensor([0.0, 0.5, 0.0, -0.25, 0.0, 1.0])

        logits = readout._sparse_logits(activation, weights, 4, [3, 1])

        torch.testing.assert_close(logits, activation[[1, 3, 5]] @ weights[[1, 3, 5]][:, [3, 1]])
        self.assertEqual(readout.last_sparse_rows.tolist(), [1, 3, 5])
        extracted = ReadoutMath.sparse_logits(readout, activation, weights, 4, [3, 1])
        torch.testing.assert_close(extracted, logits)
        self.assertIsInstance(readout._regulators, ReadoutRegulators)
        readout.mca_enabled = False
        self.assertFalse(readout._regulators.mca_enabled)

    def test_public_scoring_signatures_are_unchanged(self):
        expected = {
            "compute_dendritic_plateau_state": (
                "self", "v_prev2", "v_prev1", "v_cur", "pos_idx", "q_vec"),
            "predict_next_token_logits": (
                "self", "v_prev2", "v_prev1", "v_cur", "pos_idx", "q_vec", "candidate_cols"),
            "counterfactual_credit_report": (
                "self", "v_prev2", "v_prev1", "v_cur", "pos_idx", "q_vec",
                "target_token", "auxiliary_components"),
            "resolve_episode_pole": ("self", "q_vec"),
            "clean_up_stimulus_in_ca3_attractor": (
                "self", "q_raw", "q_sdr2048", "beta", "recent_poles"),
            "discover_second_hop_ca3_pole": (
                "self", "first_pole_vec", "stimulus_sdr2048", "recent_poles"),
            "imprint_response_sequence": (
                "self", "q_vec", "response_vecs", "response_ids", "f_mod",
                "stimulus_ids", "response_content_ids"),
            "imprint_response_batch_vectorized": (
                "self", "batch_items", "auxiliary_logits", "memory_query_vecs"),
        }
        for name, parameters in expected.items():
            self.assertEqual(tuple(inspect.signature(
                getattr(DendriticPlateauMatrixReadout, name)).parameters), parameters)

    def test_bank_domain_selection_keeps_target_within_budget(self):
        readout = DendriticPlateauMatrixReadout(d_vec=4, d_dend=6, max_pos=3, seed=7)
        for token in (10, 11, 12, 13):
            readout._ensure_token_col(token)
        selected = {}

        class Router:
            max_token_ids = 2

            def learn_sparse(self, q_vec, d_nmda, target_token, tokens, **kwargs):
                selected.update(target_token=target_token, tokens=tokens, **kwargs)
                return {"accepted": True}

        readout.bank_router = Router()
        components = {
            "dendritic": torch.tensor([0.0, 0.0, 0.0, 0.0]),
            "core": torch.zeros(4),
            "swr": torch.zeros(4),
            "bank": torch.tensor([0.0, 0.0, 2.0, 1.0]),
        }
        with patch.object(readout, "_readout_components", return_value=components), \
                patch.object(ReadoutBanks, "learn_bank", wraps=ReadoutBanks.learn_bank) as learn_bank:
            result = readout._learn_bank(torch.ones(4), torch.ones(6), target_col=0)

        learn_bank.assert_called_once()
        self.assertTrue(result["accepted"])
        self.assertEqual(selected["tokens"], [12, 10])
        self.assertEqual(selected["target_token"], 10)
        self.assertEqual(len(selected["tokens"]), 2)


if __name__ == "__main__":
    unittest.main()
