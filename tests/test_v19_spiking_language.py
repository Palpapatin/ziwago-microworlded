"""Small-module v19 contracts; no master checkpoints or rewritten goldens."""
from __future__ import annotations

import io
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ziwago.organism.physiology import OrganismPhysiology
from ziwago.cortex.moe_cortex import SparselyGatedCorticalMoE
from ziwago.cortex.son_reflection import SONReflectiveColumn
from ziwago.cortex.spiking_language import SpikingLanguageLoop
from train_ziwago_v14_pure_synaptic_dialogue import (
    BA44AttractorSyntaxBridge, ConversationalDialogueState, DendriticPlateauMatrixReadout,
    PredictiveCodingCorticalReadout, ThetaGammaRoleFillerBinder, TokenizedDialogPair,
    ZiwagoPureSynapticDialogueOrganismV14,
)
import ziwago.app.tokenizer as gpt2_tokenizer  # B.2: владелец BPE worker-глобалов
import train_ziwago_v9_gpt2vocab as v9_wikitext  # _tokenize_wikitext_chunk остаётся в v9-тренере
from train_ziwago_v14_pure_synaptic_dialogue import _bpe_tokenize_raw_pairs_worker


def small_moe(n_in=8, selective=False):
    moe = SparselyGatedCorticalMoE(n_columns=2, neurons_per_column=32, n_in=n_in, seed=19)
    for col in moe.columns:
        col.pyr.cfg.stochastic = False
        col.pyr.cfg.c_m = 0.1
        if selective:
            # A controlled *real* cortical afferent experiment, not mocked spikes.
            col.dendrite.w_syn.zero_()
            col.dendrite.alpha.zero_()
            col.w_ff.w_slow.zero_()
            col.w_ff.w_fast.zero_()
            col.w_ff.mask.fill_(True)
            col.w_ff.enable_block_sparse = False
            for category in range(4):
                col.w_ff.w_slow[8 * category:8 * (category + 1), category] = 5.0
            col.w_rec.w_slow.zero_()
            col.w_fb.w_slow.zero_()
    return moe


def tiny_core():
    core = ZiwagoPureSynapticDialogueOrganismV14.__new__(ZiwagoPureSynapticDialogueOrganismV14)
    core.root, core.seed = ROOT, 19
    core.moe_cortex = small_moe(64)
    core.spiking_language_loop = SpikingLanguageLoop(core.moe_cortex, seed=20)
    core.son_column = SONReflectiveColumn(n_son_neurons=32, n_observed_columns=2, d_vec=64, seed=19)
    core.chemistry = OrganismPhysiology(core.moe_cortex.n_columns)
    core.pc_readout = PredictiveCodingCorticalReadout(vocab_size=32, d_token=64, d_state=256, seed=19)
    core.dendritic_plateau_readout = DendriticPlateauMatrixReadout(d_vec=64, d_dend=64, max_pos=16, seed=19)
    core.dialogue_state = ConversationalDialogueState()
    core.role_filler_binder = ThetaGammaRoleFillerBinder()
    core.ba44_syntax_bridge = BA44AttractorSyntaxBridge()
    core.vsa_encoder = SimpleNamespace(hashed_sdr=SimpleNamespace(micro_weights=torch.ones(4)),
                                       base_active_bins=torch.ones(4, dtype=torch.uint8), w_proj=torch.eye(64))
    # TDR (13.11) включён по умолчанию и обращается к энкодеру за SDR и
    # корковой проекцией токена — заглушка должна уметь то же, что и живой
    # энкодер, иначе тест проверял бы не тот контур.
    def _encode_token_id(token_id: int) -> torch.Tensor:
        sdr = torch.zeros(4)
        sdr[int(token_id) % 4] = 1.0
        return sdr
    def _project_token_to_cortex(token_id: int) -> torch.Tensor:
        vec = torch.zeros(64)
        vec[int(token_id) % 64] = 1.0
        return vec
    core.vsa_encoder.encode_token_id = _encode_token_id
    core.vsa_encoder.project_token_to_cortex = _project_token_to_cortex
    words = ["What", " neurons", " integrate", " sensory", " signals", " basal", " dendrites", ".",
             " quux", " corge", " not", " inhibit", " apical", " predictions", " bottom-up", " top-down"]
    core._id_to_token_raw = dict(enumerate(words))
    core.eos_token_id = 31
    core._id_to_token_raw[core.eos_token_id] = "<|endoftext|>"
    gen = torch.Generator().manual_seed(19)
    core._vec64_cache = {i: F.normalize(torch.randn(64, generator=gen), dim=0) for i in range(32)}
    core._vec64_cache[262] = core._vec64_cache[0].clone()
    core.encode_text = lambda text: [next((i for i, word in enumerate(words)
                                         if word.strip().lower() == w.strip(".,?!").lower()), 8)
                                    for w in text.split()] or [0]
    core.w_stimulus_to_onset = torch.zeros(64, 32)
    core.onset_pair_synapses, core.onset_pair_ctx_vec = {}, {}
    core.synaptic_bigram_g, core.synaptic_trigram_g = {}, {}
    core.synaptic_trigram_ctx, core.synaptic_fourgram_g = {}, {}
    core.use_ngrams = False
    core.total_pairs_trained, core.total_tokens_trained, core.cumulative_da_reward = 0, 0, 0.0
    core.language_memory_records = []
    core.external_grounding_verifier_fn = None
    core.learning_enabled, core._evaluation_depth = True, 0
    from ziwago.cortex.genesis_engine import ThetaGammaSpikingPhaseMultiplexer
    core.dialogue_writes_enabled = True
    core.online_learning_enabled = True
    core.temporal_codec = ThetaGammaSpikingPhaseMultiplexer(n_gamma_slots=1, bins_per_gamma=81)
    core.temporal_mode = "normal"
    core.temporal_generator = torch.Generator().manual_seed(19)
    core.last_temporal_report = {}
    core.last_ca3_top1_resonance, core.last_multi_hop_used = 1.0, False
    core.tokenizer_sha256 = "a" * 64
    # ЕДИНЫЙ КОНТУР (13.11): фикстура собирает организм вручную (минуя
    # __init__), поэтому биологический драйвер нужно поднять явно — иначе
    # тесты проверяли бы кору «без мозга» (голый шаг без таламуса, TRN,
    # AST/GW и критичности), то есть не тот контур, что живёт в проекте.
    core.enable_integrated_organism()
    return core


class V19Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_tree_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_tree_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_tree_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_learned_heldout_cortical_mapping_and_independent_ablations(self):
        moe = small_moe(selective=True)
        loop = SpikingLanguageLoop(moe, learning_rate=0.6)
        tokens, labels = [101, 102, 103, 104], [103, 101, 104, 102]
        loop.ensure_tokens(tokens)
        initial = moe.export_state_dict()
        before = loop.w_projector.clone()
        # Train on 0.9/1.1 amplitudes; hold out 0.7/1.3 and nuisance afferents.
        for epoch in range(18):
            for category in range(4):
                moe.load_state_dict(initial)
                x = torch.eye(8)[category] * (0.9 if epoch % 2 else 1.1)
                loop.step(x, tokens, target_token=labels[category], learn=True)
        self.assertFalse(torch.equal(before, loop.w_projector))
        measurements = {}
        for mode in ("normal", "disabled", "shuffled", "constant"):
            correct, losses = 0, []
            for amplitude in (0.7, 1.3):
                for category in range(4):
                    moe.load_state_dict(initial)
                    x = torch.eye(8)[category] * amplitude
                    x[7] = 0.03  # held-out afferent, never part of a fixed token lookup
                    scores, out = loop.step(x, tokens, mode=mode)
                    self.assertGreater(float(out.combined_spikes.sum()), 0)
                    target = tokens.index(labels[category])
                    correct += int(int(scores.argmax()) == target)
                    losses.append(float(-scores.log_softmax(0)[target]))
            measurements[mode] = (correct / 8, sum(losses) / 8)
        self.assertEqual(measurements["normal"][0], 1.0, measurements)
        for mode in ("disabled", "shuffled", "constant"):
            self.assertLess(measurements[mode][0], measurements["normal"][0], measurements)
            self.assertGreater(measurements[mode][1], measurements["normal"][1] + 0.25, measurements)

    def test_timing_channel_drives_query_and_sparse_token_logits(self):
        core = tiny_core()
        gen = torch.Generator().manual_seed(1907)
        ro = core.dendritic_plateau_readout
        for token in range(5):
            ro._ensure_token_col(token)
        ro.w_dend_out[:, :5] = torch.randn(ro.d_dend, 5, generator=gen)
        vector = F.normalize(torch.randn(64, generator=gen), dim=0)
        clean = core._transmit_temporal_vector(vector)
        self.assertGreater(float(clean @ vector), .999)
        scores = ro.predict_next_token_scores(clean, clean, clean, 0, clean)
        core.temporal_mode = "shuffle"
        noisy = core._transmit_temporal_vector(vector)
        changed = ro.predict_next_token_scores(noisy, noisy, noisy, 0, noisy)
        self.assertLess(float(noisy @ vector), .5)
        self.assertTrue(any(abs(scores[t] - changed[t]) > 1e-4 for t in scores))
        core.temporal_mode = "drop"
        core._transmit_temporal_vector(vector)
        self.assertGreater(core.last_temporal_report["loss_fraction"], .1)

    def test_teacher_likelihood_measures_tokens_without_runtime_mutation(self):
        core = tiny_core()
        core.train_wave_with_neuromodulator_reward([TokenizedDialogPair("lab", [1, 3], [1, 2, 3, 4, 5, 6, 7])])
        before = core.export_runtime_state()
        report = core.teacher_forced_likelihood("neurons sensory", "neurons integrate sensory signals")
        self.assertTrue(report["available"])
        self.assertEqual(report["token_count"], 4)
        self.assertGreaterEqual(report["nll_sum"], 0)
        self.assert_tree_equal(before, core.export_runtime_state())
        self.assertFalse(core.teacher_forced_likelihood("neurons", "quux")["available"])

    def test_both_real_spikes_and_l5_change_token_scores(self):
        moe = small_moe(selective=True)
        loop = SpikingLanguageLoop(moe)
        loop.ensure_tokens([1, 2])
        out = moe.step(torch.eye(8)[3], learn=False)
        loop.w_projector[:, 0] = torch.arange(loop.d_features).float() / loop.d_features
        loop.w_projector[:, 1] = -loop.w_projector[:, 0]
        reference = loop.predict(out, [1, 2])
        for field in ("combined_spikes", "combined_l5_drive"):
            altered = SimpleNamespace(**vars(out))
            setattr(altered, field, torch.zeros_like(getattr(out, field)))
            self.assertFalse(torch.allclose(reference, loop.predict(altered, [1, 2])))

    def test_teacher_error_reaches_real_apical_compartments_and_synapses(self):
        moe = small_moe(selective=True)
        loop = SpikingLanguageLoop(moe)
        loop.ensure_tokens([1, 2])
        out = moe.step(torch.eye(8)[2], learn=False)
        before = [(c.pyr.v_a.clone(), c.w_sst_in.clone(), c.dendrite.w_syn.clone()) for c in moe.columns]
        result = loop.learn(out, 2, [1, 2], task_modulation=1.5)
        self.assertGreater(result["apical_error_norm"], 0)
        for cid in out.active_expert_ids:
            col = moe.columns[cid]
            self.assertFalse(torch.equal(before[cid][0], col.pyr.v_a))
            self.assertFalse(torch.equal(before[cid][1], col.w_sst_in))
            self.assertFalse(torch.equal(before[cid][2], col.dendrite.w_syn))

    def test_spiking_learning_uses_shared_counterfactual_error(self):
        moe = small_moe(selective=True)
        loop = SpikingLanguageLoop(moe)
        loop.ensure_tokens([1, 2])
        out = moe.step(torch.eye(8)[2], learn=False)
        features = loop.cortical_features(out)
        before = loop.w_projector.clone()
        auxiliary = torch.tensor([4.0, -4.0])
        total = features @ before + auxiliary
        expected_error = -torch.softmax(total, dim=0)
        expected_error[0] += 1.0
        expected_loss = float(-torch.log_softmax(total, dim=0)[0])
        result = loop.learn(out, 1, [1, 2], task_modulation=1.5, auxiliary_logits=auxiliary)
        expected_update = torch.outer(features, expected_error) * loop.learning_rate * 1.5
        self.assertEqual(result["loss"], expected_loss)
        torch.testing.assert_close(loop.last_token_error, expected_error)
        torch.testing.assert_close(loop.w_projector, before + expected_update)

    def test_invalid_ablation_rejected_and_no_ablated_learning(self):
        loop = SpikingLanguageLoop(small_moe())
        with self.assertRaises(ValueError):
            loop.set_mode("fake")
        loop.ensure_tokens([1, 2])
        before = loop.w_projector.clone()
        for mode in ("disabled", "shuffled", "constant"):
            loop.set_mode(mode)
            loop.step(torch.ones(8), [1, 2], target_token=1, learn=True)
        self.assertTrue(torch.equal(before, loop.w_projector))

    def test_sparse_inference_matches_dense_reference_without_epsilon_or_cache(self):
        ro = DendriticPlateauMatrixReadout(d_vec=8, d_dend=128, max_pos=8)
        for tid in range(5):
            ro._ensure_token_col(tid)
        gen = torch.Generator().manual_seed(5)
        ro.w_dend_out[:, :5] = torch.randn(128, 5, generator=gen)
        ro.w_core_dend_out[:, :5] = torch.randn(128, 5, generator=gen)
        q = F.normalize(torch.randn(8, generator=gen), dim=0)
        d = ro.compute_dendritic_plateau_state(q, q, q, 0, q)
        self.assertLessEqual(int(d.count_nonzero()), 32)
        dense = d @ (ro.w_dend_out[:, :5] + 0.35 * ro.w_core_dend_out[:, :5])
        scores = ro.predict_next_token_scores(q, q, q, 0, q)
        torch.testing.assert_close(torch.tensor([scores[t] for t in range(5)]), dense)
        self.assertIsNone(getattr(ro, "_w_eff_cache", None))
        zero = torch.zeros(8)
        self.assertEqual(int(ro.compute_dendritic_plateau_state(zero, zero, zero, 0, zero).count_nonzero()), 0)

    def test_sparse_update_numerical_reference_and_shield_really_used(self):
        ro = DendriticPlateauMatrixReadout(d_vec=8, d_dend=128, max_pos=8)
        for token in (1, 2, 3):
            ro._ensure_token_col(token)
        q = F.normalize(torch.arange(1, 9).float(), dim=0)
        d = ro.compute_dendritic_plateau_state(q, q, q, 0, q)
        rows = d.nonzero().flatten()
        ro.w_dend_out[:, :3].fill_(0.5)
        ro.shield.protect_rows(ro.w_dend_out, rows, 3)
        old = ro.w_dend_out[:, :3].clone()
        err = torch.tensor([0.5, -0.2, -0.3])
        # Трасса щита — на строку (reinforcement_v2): затухание построчное,
        # неактивные строки не гасятся (их трасса нулевая).
        prior = ro.shield.u_slow.clone()
        attenuation = torch.ones(ro.d_dend)
        attenuation[rows] = 1.0 / (1.0 + ro.shield.kappa_shield * prior[rows])
        expected = old + torch.outer(d, err) * 0.8 * attenuation.unsqueeze(1)
        with patch.object(ro.shield, "apply_sparse_update", wraps=ro.shield.apply_sparse_update) as apply:
            ro._apply_local_update(ro.w_dend_out, d, err, 0.8)
            self.assertEqual(apply.call_count, 1)
            self.assertLessEqual(apply.call_args.args[2].shape[0], 32)
        torch.testing.assert_close(ro.w_dend_out[:, :3], expected)
        self.assertLess(float((expected - old).norm()), float((torch.outer(d, err) * 0.8).norm()))

    def test_sparse_training_protects_and_router_changes_readout(self):
        ro = DendriticPlateauMatrixReadout(d_vec=8, d_dend=64, max_pos=8)
        q = F.normalize(torch.arange(1, 9).float(), dim=0)
        ids, vecs = [1, 2, 1, 2], [q] * 4
        with patch.object(ro.shield, "protect_rows", wraps=ro.shield.protect_rows) as protect:
            ro.imprint_response_sequence(q, vecs, ids, f_mod=2.0, stimulus_ids=[3])
            self.assertEqual(protect.call_count, 4)
        d = ro.compute_dendritic_plateau_state(q, q, q, 0, q)
        self.assertGreater(float(ro.bank_router.predict_sparse(q, d, ro.col_to_tok).abs().sum()), 0)
        full = ro.predict_next_token_scores(q, q, q, 0, q)
        ro.router_residual_bound = 0.0
        shared = ro.predict_next_token_scores(q, q, q, 0, q)
        self.assertTrue(any(abs(full[t] - shared[t]) > 1e-6 for t in full))
        self.assertTrue(all(abs(full[t] - shared[t]) <= 0.250001 for t in full))

    def test_bank_candidate_budget_handles_large_active_vocabulary(self):
        ro = DendriticPlateauMatrixReadout(d_vec=8, d_dend=64, max_pos=8)
        for tid in range(300):
            ro._ensure_token_col(tid)
        q = F.normalize(torch.ones(8), dim=0)
        d = ro.compute_dendritic_plateau_state(q, q, q, 0, q)
        ro._learn_bank(q, d, 299, torch.zeros(300))
        self.assertEqual(len(ro.predict_next_token_scores(q, q, q, 0, q)), 48)

    def test_module_state_roundtrip_is_data_only_and_exact(self):
        core = tiny_core()
        core.train_wave_with_neuromodulator_reward([TokenizedDialogPair("synthetic_train", [1, 3], [1, 2, 3, 4, 5, 6, 7])])
        state = core.export_runtime_state()
        buf = io.BytesIO()
        torch.save(state, buf)
        buf.seek(0)
        data = torch.load(buf, weights_only=True)
        restored = tiny_core()
        restored.load_runtime_state(data)
        self.assert_tree_equal(state, restored.export_runtime_state())
        x = F.normalize(torch.ones(64), dim=0)
        expected = core.moe_cortex.step(x, learn=False).combined_spikes
        actual = restored.moe_cortex.step(x, learn=False).combined_spikes
        self.assertTrue(torch.equal(expected, actual))

    def test_evaluation_disables_learning_writes_and_restores_all_state_and_rng(self):
        core = tiny_core()
        q = F.normalize(torch.ones(64), dim=0)
        core.spiking_language_loop.ensure_tokens([1, 2])
        state = core.export_runtime_state()
        with core.evaluation_context():
            intrinsic = core.moe_cortex.columns[0].pyr.theta_base.clone()
            core.moe_cortex.step(q, learn=True)
            self.assertTrue(torch.equal(intrinsic, core.moe_cortex.columns[0].pyr.theta_base))
            core.spiking_language_loop.step(q, [1, 2], target_token=2, learn=True)
            core.role_filler_binder.register_emergent_stdp_bindings({"role": ("novel", 0.9)})
            core.dialogue_state.record_turn_and_poles(q, [q], "observed", "generated")
            core.dendritic_plateau_readout.imprint_swr_composite_episode(q, [q] * 4, [1, 2, 1, 2])
            self.assertEqual(core.language_memory_records, [])
            self.assertEqual(core.dialogue_state.turn_count, 0)
            self.assertEqual(core.role_filler_binder.dynamic_stdp_role_map, {})
            self.assertEqual(core.spiking_language_loop.total_updates, 0)
            son_before = core.son_column.meta_head.w_out.clone()
            core.son_column.observe_and_reflect_step(torch.ones(2), torch.zeros(2), torch.zeros(2),
                                                      torch.ones(2) / 2, q, False, 1, learn=True)
            self.assertTrue(torch.equal(son_before, core.son_column.meta_head.w_out))
            core.last_telemetry = {"mutated": True}
            core._vec64_cache[31] = q
            random.random(); np.random.random(); torch.rand(3)
            torch.rand(3, generator=core.spiking_language_loop.generator)
            with core.evaluation_context():
                core.moe_cortex.bio_rwkv.step(q, learn=True)
        self.assert_tree_equal(state, core.export_runtime_state())
        with self.assertRaises(RuntimeError):
            with core.evaluation_context():
                core.moe_cortex.w_router.add_(1)
                raise RuntimeError("evaluation failed")
        self.assert_tree_equal(state, core.export_runtime_state())

    def test_provenance_ledger_uses_training_only_and_never_generated_answers(self):
        core = tiny_core()
        train = TokenizedDialogPair("synthetic_train", [1, 3], [1, 2, 3, 4, 5, 6, 7])
        heldout = TokenizedDialogPair("heldout", [8, 9], [8, 9, 8, 9], split="test")
        core.train_wave_with_neuromodulator_reward([train, heldout])
        self.assertEqual(len(core.language_memory_records), 1)
        record = core.language_memory_records[0]
        self.assertEqual(record["source"], "synthetic_train")
        self.assertEqual((record["split"], record["status"]), ("train", "asserted"))
        self.assertIn("observed_at", record)
        with core.evaluation_context():
            self.assertEqual(core.train_wave_with_neuromodulator_reward([train])["pairs"], 0)
        self.assertEqual(len(core.language_memory_records), 1)

    def test_checkpoint_persists_loop_ledger_shield_bank_and_moe(self):
        core = tiny_core()
        core.train_wave_with_neuromodulator_reward([TokenizedDialogPair("source", [1, 3], [1, 2, 3, 4, 5, 6, 7], 2.5)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "small.pt"
            core.save_self_contained_checkpoint(path)
            restored = tiny_core()
            hook = lambda p, r: {"grounded": False}
            restored.external_grounding_verifier_fn = hook
            restored.load_self_contained_checkpoint(path)
            self.assertIs(restored.external_grounding_verifier_fn, hook)
            self.assert_tree_equal(core.export_runtime_state(), restored.export_runtime_state())
            generated = restored.pure_neural_generate_response(
                "What sensory signals?", max_tokens=8, enable_active_inference=False)
            self.assertIn("pure_weight_generated_response", generated)
            self.assertTrue(str(generated["pure_weight_generated_response"]).strip())

    def test_external_grounding_hook_is_mandatory_and_fail_closed(self):
        core = tiny_core()
        answer = "Pyramidal neurons receive sensory signals on basal dendrites."
        calls = []
        core.external_grounding_verifier_fn = lambda p, r: calls.append((p, r)) or {"grounded": False, "score": 0.0}
        self.assertFalse(core.evaluate_semantic_pole_closure_and_grounding("Pyramidal neurons?", answer, 999)["grounded"])
        self.assertEqual(len(calls), 1)
        core.external_grounding_verifier_fn = lambda p, r: {"verified": True, "score": 0.95}
        self.assertTrue(core.evaluate_semantic_pole_closure_and_grounding("Pyramidal neurons?", answer)["grounded"])
        core.external_grounding_verifier_fn = lambda p, r: 0.95
        self.assertFalse(core.evaluate_semantic_pole_closure_and_grounding("Pyramidal neurons?", answer)["grounded"])

    def test_default_grounding_rejects_domain_only_and_polarity_conflicts(self):
        core = tiny_core()
        self.assertFalse(core.evaluate_semantic_pole_closure_and_grounding(
            "How do astrocytes activate receptors?", "Pyramidal neurons receive sensory signals on basal dendrites.", 999)["grounded"])
        self.assertFalse(core.evaluate_semantic_pole_closure_and_grounding(
            "Neurons do not inhibit sensory signals.", "Neurons inhibit sensory signals on basal dendrites.", 999)["grounded"])
        self.assertTrue(core._strict_content_polarity_verifier(
            "Pyramidal neurons integrate sensory signals.", "Pyramidal neurons integrate sensory signals on basal dendrites.")["grounded"])

    def test_dynamic_role_binding_has_no_preloaded_family_exceptions(self):
        binder = ThetaGammaRoleFillerBinder()
        self.assertEqual(binder.extract_slot_bindings("fix runny hollandaise and errors in my manuscript"), {})
        binder.register_emergent_stdp_bindings({"mechanism": ("quux", 0.8)})
        self.assertEqual(binder.extract_slot_bindings("explain quux"), {"mechanism": "quux"})

    def test_think_and_disabled_active_inference_cannot_bypass_verifier_or_write_swr(self):
        core = tiny_core()
        core.dendritic_plateau_readout.imprint_response_sequence(
            core._vec64_cache[1], [core._vec64_cache[t] for t in [1, 2, 3, 4, 5, 6, 7]],
            [1, 2, 3, 4, 5, 6, 7], stimulus_ids=[1, 3])
        core._attempt_endogenous_cortical_think_reformulation = lambda **kwargs: {
            "response_text": "Pyramidal neurons receive sensory signals on basal dendrites.",
            "think_trace": "external graph", "bindings": {}}
        calls = []
        core.external_grounding_verifier_fn = lambda p, r: calls.append(r) or {"grounded": False}
        before = core.dendritic_plateau_readout.w_swr_dend_out.clone()
        result = core.pure_neural_generate_response("What sensory signals?", max_tokens=8, enable_active_inference=False)
        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(result["active_inference_clarification_triggered"])
        self.assertFalse(result["endogenous_cortical_think_triggered"])
        self.assertFalse(result["swr_consolidated_into_nmda_matrix"])
        self.assertTrue(torch.equal(before, core.dendritic_plateau_readout.w_swr_dend_out))
        self.assertFalse(any(t["is_grounded_conclusion"] for t in core.dialogue_state.episodic_turns))


class ReadoutCapacityTests(unittest.TestCase):
    assert_tree_equal = V19Tests.assert_tree_equal
    WEIGHT_NAMES = ("w_dend_out", "w_core_dend_out", "w_swr_dend_out")

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _assert_capacity(self, ro, tokens):
        capacity = ((tokens + 255) // 256) * 256
        expected_bytes = 3 * ro.d_dend * capacity * 4
        allocated = 0
        for name in self.WEIGHT_NAMES:
            weights = getattr(ro, name)
            self.assertEqual(weights.shape, (ro.d_dend, capacity))
            self.assertEqual(weights.dtype, torch.float32)
            allocated += weights.untyped_storage().nbytes()
        self.assertEqual(allocated, expected_bytes)
        state = ro.export_state_dict()
        self.assertEqual(sum(state[name].untyped_storage().nbytes() for name in self.WEIGHT_NAMES),
                         3 * ro.d_dend * tokens * 4)
        for name in self.WEIGHT_NAMES:
            self.assertEqual(state[name].shape, (ro.d_dend, tokens))

    def test_empty_and_1_256_257_token_allocations_preserve_weights_and_lazy_shield(self):
        ro = DendriticPlateauMatrixReadout(d_vec=8, d_dend=64, max_pos=8, seed=19)
        self._assert_capacity(ro, 0)
        self.assertEqual(ro.shield.u_slow.untyped_storage().nbytes(), 0)
        previous_count = 0
        for count in (1, 256, 257):
            with self.subTest(tokens=count):
                previous = {name: getattr(ro, name)[:, :previous_count].clone() for name in self.WEIGHT_NAMES}
                old_slow = ro.shield.u_slow.clone()
                for col in range(previous_count, count):
                    self.assertEqual(ro._ensure_token_col(10001 + 17 * col), col)
                self._assert_capacity(ro, count)
                self.assertTrue(torch.equal(ro.shield.u_slow, old_slow))
                for index, name in enumerate(self.WEIGHT_NAMES):
                    weights = getattr(ro, name)
                    self.assertTrue(torch.equal(weights[:, :previous_count], previous[name]))
                    self.assertEqual(int(weights[:, previous_count:].count_nonzero()), 0)
                    weights[:, :count] = (torch.arange(64 * count).reshape(64, count).float() + index + 1) / 256
                ro.shield._ensure(ro.w_dend_out, count)
                # Трасса щита — вектор на строку: рост словаря колонок её не трогает,
                # накопленные значения переживают перевыделение.
                self.assertEqual(tuple(ro.shield.u_slow.shape), (64,))
                expected_slow = torch.zeros(64)
                keep = min(int(old_slow.numel()), 64)
                if keep:
                    expected_slow[:keep] = old_slow[:keep]
                self.assertTrue(torch.equal(ro.shield.u_slow, expected_slow))
                ro.shield.u_slow[:] = torch.arange(64).float() / 1024
                identities = [id(getattr(ro, name)) for name in self.WEIGHT_NAMES]
                self.assertEqual(ro._ensure_token_col(ro.col_to_tok[-1]), count - 1)
                self.assertEqual(identities, [id(getattr(ro, name)) for name in self.WEIGHT_NAMES])
                self.assertEqual(ro.col_to_tok, [10001 + 17 * col for col in range(count)])
                self.assertEqual(ro.tok_to_col, {token: col for col, token in enumerate(ro.col_to_tok)})
                previous_count = count

    def test_boundary_logits_are_bit_exact_dense_reference_and_exclude_padding(self):
        q = torch.eye(1, 8)[0]
        d = torch.zeros(64)
        d[[0, 11, 63]] = torch.tensor([1.0, 0.5, 0.25])
        for count in (0, 1, 256, 257):
            with self.subTest(tokens=count):
                ro = DendriticPlateauMatrixReadout(d_vec=8, d_dend=64, max_pos=8, seed=19)
                for col in range(count):
                    ro._ensure_token_col(10001 + 17 * col)
                for index, name in enumerate(self.WEIGHT_NAMES):
                    weights = getattr(ro, name)
                    # Dyadic weights make the dense/sparse reduction exactly representable.
                    weights[:, :count] = ((torch.arange(64 * count).reshape(64, count) % 29).float()
                                          - 14 + index) / 256
                    weights[:, count:].fill_(1000000.0 * (index + 1))
                ro.w_swr_poles = q.unsqueeze(0)
                dense = (d @ ro.w_dend_out[:, :count]
                         + 0.35 * (d @ ro.w_core_dend_out[:, :count])
                         + 1.15 * (d @ ro.w_swr_dend_out[:, :count]))
                with patch.object(ro, "compute_dendritic_plateau_state", return_value=d):
                    logits = ro.predict_next_token_logits(q, q, q, 0, q)
                    scores = ro.predict_next_token_scores(q, q, q, 0, q)
                self.assertEqual(logits.shape, (count,))
                self.assertTrue(torch.equal(logits, dense))
                self.assertEqual(len(scores), min(48, count))
                for token, score in scores.items():
                    self.assertEqual(score, float(dense[ro.tok_to_col[token]]))
                self.assertIsNone(getattr(ro, "_w_eff_cache", None))

    def test_compact_checkpoint_load_uses_same_block_and_retains_active_vocab_shield_bank(self):
        gen = torch.Generator().manual_seed(19257)
        q = F.normalize(torch.randn(8, generator=gen), dim=0)
        d = torch.zeros(64)
        d[[1, 7, 19]] = torch.tensor([0.6, 0.8, 0.25])
        for count in (0, 1, 256, 257):
            with self.subTest(tokens=count):
                ro = DendriticPlateauMatrixReadout(d_vec=8, d_dend=64, max_pos=8, seed=19)
                for col in range(count):
                    ro._ensure_token_col(10001 + 17 * col)
                for name in self.WEIGHT_NAMES:
                    weights = getattr(ro, name)
                    weights[:, :count] = torch.randn(64, count, generator=gen) * 0.2
                    weights[:, count:].fill_(12345.0)
                ro.shield._ensure(ro.w_dend_out, count)
                ro.shield.u_slow.copy_(torch.rand(64, generator=gen))
                ro.shield.protect_rows(ro.w_dend_out, d.nonzero().flatten(), count)
                ro.w_swr_poles = q.unsqueeze(0).clone()
                ro.w_ca3_attractors = q.unsqueeze(0).clone()
                ro.w_ca3_sdr = ro._compute_cortical_sdr2048([3, 4], normalize=False).unsqueeze(0)
                ro.w_ca3_salience = torch.tensor([2.5])
                if count >= 2:
                    ro.bank_router.learn_sparse(
                        q, d, ro.col_to_tok[0], ro.col_to_tok[:3], all_token_ids=ro.col_to_tok,
                        base_logits=torch.zeros(len(ro.col_to_tok)),
                        residual_gain=ro.logit_scale * ro.router_residual_bound)
                state = ro.export_state_dict()
                buffer = io.BytesIO()
                torch.save(state, buffer)
                buffer.seek(0)
                checkpoint = torch.load(buffer, weights_only=True)
                restored = DendriticPlateauMatrixReadout(d_vec=8, d_dend=64, max_pos=8, seed=999)
                for token in range(257):
                    restored._ensure_token_col(token)
                restored.load_state_dict(checkpoint)
                self._assert_capacity(restored, count)
                V19Tests.assert_tree_equal(self, state, restored.export_state_dict())
                self.assertEqual(restored.col_to_tok, ro.col_to_tok)
                self.assertEqual(restored.tok_to_col, ro.tok_to_col)
                self.assertTrue(torch.equal(restored.shield.u_slow, ro.shield.u_slow))
                self.assertEqual(restored.shield.n_cols, count)
                for name in self.WEIGHT_NAMES:
                    weights = getattr(restored, name)
                    self.assertTrue(torch.equal(weights[:, :count], state[name]))
                    self.assertEqual(int(weights[:, count:].count_nonzero()), 0)
                expected = ro.predict_next_token_logits(q, q, q, 0, q)
                actual = restored.predict_next_token_logits(q, q, q, 0, q)
                self.assertTrue(torch.equal(expected, actual))


class V19TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        worker_patch = patch.multiple(gpt2_tokenizer, _WORKER_VOCAB=None,
                                      _WORKER_MERGES=None, _WORKER_CACHE={})
        worker_patch.start()
        cls.addClassCleanup(worker_patch.stop)
        # Same local assets used by the v14 constructor, without building the organism.
        gpt2_tokenizer._init_bpe_worker(str(ROOT / "data" / "gpt2" / "vocab.json"),
                                      str(ROOT / "data" / "gpt2" / "merges.txt"))
        cls.vocab = gpt2_tokenizer._WORKER_VOCAB
        cls.merges = gpt2_tokenizer._WORKER_MERGES
        cls.core = ZiwagoPureSynapticDialogueOrganismV14.__new__(ZiwagoPureSynapticDialogueOrganismV14)
        cls.core._id_to_token_raw = {tid: token for token, tid in cls.vocab.items()}

    def encode_raw(self, text):
        cache = {}
        return [tid for piece in gpt2_tokenizer._GPT2_PAT.findall(text)
                for tid in gpt2_tokenizer._bpe_encode_piece(piece, self.vocab, self.merges, cache)]

    def test_gpt2_byte_alphabet_is_reversible(self):
        encoder, decoder = gpt2_tokenizer._GPT2_BYTE_ENCODER, gpt2_tokenizer._GPT2_BYTE_DECODER
        self.assertEqual(set(encoder), set(range(256)))
        self.assertEqual(len(set(encoder.values())), 256)
        self.assertEqual([decoder[encoder[b]] for b in range(256)], list(range(256)))
        self.assertEqual({b: encoder[b] for b in (0, 9, 10, 13, 32, 255)},
                         {0: "Ā", 9: "ĉ", 10: "Ċ", 13: "č", 32: "Ġ", 255: "ÿ"})
        self.assertTrue(set(encoder.values()).issubset(self.vocab))
        self.assertEqual(len(self.vocab), 50257)

    def test_known_gpt2_ascii_ids_are_unchanged(self):
        cases = {"!": [0], "Hello": [15496], "hello": [31373],
                 "Hello, world!": [15496, 11, 995, 0], " hello world!": [23748, 995, 0],
                 " the quick brown fox": [262, 2068, 7586, 21831],
                 "  ": [220, 220], "\t\n": [197, 198]}
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.encode_raw(text), expected)
                self.assertEqual(self.core.decode_tokens(expected), text)
        self.assertEqual(self.core.encode_text("hello world!"), [23748, 995, 0])
        self.assertEqual(self.core.encode_text("the"), [262])

    def test_unicode_punctuation_and_whitespace_roundtrip(self):
        samples = ["Hello, мир! Привет, café — 你好; e\u0301 \U0001d11e.\tNext\n  end  ",
                   " \t\n\r  Привет\u00a0мир\u2003１２٣?!  ",
                   "Latin АБВ, русский Ёж; apostrophe's and I'm.\r\nnext",
                   "", " \t\r\n  "]
        for text in samples:
            with self.subTest(text=text):
                self.assertEqual("".join(gpt2_tokenizer._GPT2_PAT.findall(text)), text)
                self.assertEqual(self.core.decode_tokens(self.encode_raw(text)), text)
                if text.strip():
                    self.assertEqual(self.core.decode_tokens(self.core.encode_text(text)),
                                     " " + text.strip())

    def test_empty_input_has_no_fabricated_tokens(self):
        for text in ("", " ", "\t\n\r", "\u00a0\u2003"):
            self.assertEqual(self.core.encode_text(text), [])
        self.assertEqual(self.encode_raw(""), [])
        self.assertEqual(self.core.decode_tokens([]), "")
        self.assertEqual(gpt2_tokenizer._bpe_encode_piece("", {"": 262}, {}, {}), [])
        with patch.multiple(gpt2_tokenizer, _WORKER_VOCAB=None, _WORKER_MERGES=None):
            self.assertEqual(self.core.encode_text(""), [])

    def test_sequence_decode_joins_split_utf8_tokens(self):
        text = "Жé你\U0001d11e"
        ids = [self.vocab[gpt2_tokenizer._GPT2_BYTE_ENCODER[b]] for b in text.encode("utf-8")]
        self.assertEqual(self.core.decode_tokens(ids), text)
        self.assertNotEqual("".join(self.core.decode_token(tid) for tid in ids), text)
        for tid, expected in ((262, " the"), (15496, "Hello"), (198, "\n"), (197, "\t"), (0, "!")):
            self.assertEqual(self.core.decode_token(tid), expected)

    def test_plain_ascii_and_raw_space_vocabularies(self):
        vocab = {"What": 0, " What": 1, " neurons": 2, " integrate": 3,
                 " sensory": 4, "!": 5, "\t": 6, "\n": 7, " ": 8}
        core = ZiwagoPureSynapticDialogueOrganismV14.__new__(ZiwagoPureSynapticDialogueOrganismV14)
        core._id_to_token_raw = {tid: token for token, tid in vocab.items()}
        text = "What neurons integrate sensory!\t\n"
        ids = [tid for piece in gpt2_tokenizer._GPT2_PAT.findall(text)
               for tid in gpt2_tokenizer._bpe_encode_piece(piece, vocab, {}, {})]
        self.assertEqual(ids, [0, 2, 3, 4, 5, 6, 7])
        self.assertEqual(core.decode_tokens(ids), text)
        self.assertEqual(core.decode_token(2), " neurons")
        with patch.multiple(gpt2_tokenizer, _WORKER_VOCAB=vocab, _WORKER_MERGES={}, _WORKER_CACHE={}):
            self.assertEqual(core.encode_text(text), [1, 2, 3, 4, 5])
        vocab = {char: tid for tid, char in enumerate(" abc")}
        vocab.update({"ab": 10, " ab": 11})
        self.assertEqual(gpt2_tokenizer._bpe_encode_piece(" abc", vocab,
                         {("a", "b"): 0, (" ", "ab"): 1}, {}), [11, vocab["c"]])
        # Raw whole-token fixtures need not contain a standalone space token.
        self.assertEqual(gpt2_tokenizer._bpe_encode_piece(" neurons", {" neurons": 9}, {}, {}), [9])

    def test_missing_symbols_raise_instead_of_inventing_id_zero(self):
        for piece, vocab in (("?", {"!": 0}), ("ab", {"a": 8, "!": 0}), ("Ж", {"!": 0})):
            cache = {}
            with self.subTest(piece=piece), self.assertRaises(KeyError):
                gpt2_tokenizer._bpe_encode_piece(piece, vocab, {}, cache)
            self.assertNotIn(piece, cache)

    def test_active_worker_encoding_parity(self):
        stimulus = "User: \tHello, мир! café?"
        response = "Assistant: \nПривет, world! 你好."
        clean = lambda text: " ".join(text.replace("User:", "").replace("Assistant:", "").strip().split())
        rows = _bpe_tokenize_raw_pairs_worker([("utf8", stimulus, response, 0.75)])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], ("utf8", self.core.encode_text(clean(stimulus)),
                                  self.core.encode_text(clean(response)), 0.75))
        self.assertEqual(self.core.decode_tokens(rows[0][1]), " " + clean(stimulus))
        self.assertEqual(self.core.decode_tokens(rows[0][2]), " " + clean(response))
        self.assertEqual(_bpe_tokenize_raw_pairs_worker([("empty", "", response, 1.0)]), [])
        line = "Привет мир это русский текст, café и 你好; words stay here."
        flat, paragraphs = v9_wikitext._tokenize_wikitext_chunk([line])
        self.assertEqual(flat, self.encode_raw(line))
        self.assertEqual(paragraphs, [flat])
        self.assertEqual(self.core.decode_tokens(flat), line)

    def test_training_ledger_decodes_whole_token_sequences(self):
        core = tiny_core()
        core._id_to_token_raw.update({tid: gpt2_tokenizer._GPT2_BYTE_ENCODER[b]
                                      for tid, b in zip((1, 2), "Ж".encode("utf-8"))})
        core.train_wave_with_neuromodulator_reward([
            TokenizedDialogPair("utf8", [1, 2, 3], [1, 2, 3, 4, 5, 6, 7])])
        self.assertEqual(core.language_memory_records[-1]["prompt"], "Ж sensory")
        self.assertEqual(core.language_memory_records[-1]["response"], "Ж sensory signals basal dendrites.")


if __name__ == "__main__":
    unittest.main()
