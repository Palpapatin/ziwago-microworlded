"""Sparse cortical bank selection and training delegation."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class ReadoutBanks:
    """Selects the active token domain and delegates sparse bank learning."""

    @staticmethod
    def learn_bank(
        readout,
        q_vec: torch.Tensor,
        d_nmda: torch.Tensor,
        target_col: int,
        auxiliary_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        if not readout.learning_enabled:
            return {"accepted": False, "reason": "learning_disabled"}
        n_active = len(readout.col_to_tok)
        components = readout._readout_components(d_nmda, q_vec, n_active)
        base_logits = readout.logit_scale * (
            components["dendritic"] + components["core"] + components["swr"])
        if auxiliary_logits is not None:
            base_logits = base_logits + auxiliary_logits.to(base_logits)
        logits = base_logits + readout.logit_scale * components["bank"]
        budget = min(n_active, readout.bank_router.max_token_ids)
        cols = torch.topk(logits, k=budget).indices.tolist()
        if target_col not in cols:
            cols[-1] = target_col
        tokens = [readout.col_to_tok[c] for c in cols]
        return readout.bank_router.learn_sparse(
            q_vec,
            d_nmda,
            readout.col_to_tok[target_col],
            tokens,
            learning_rate=0.1,
            all_token_ids=readout.col_to_tok,
            base_logits=base_logits,
            residual_gain=readout.logit_scale * readout.router_residual_bound,
        )

    @staticmethod
    def imprint_response_sequence(
        readout,
        q_vec: torch.Tensor,
        response_vecs: List[torch.Tensor],
        response_ids: List[int],
        f_mod: float = 1.5,
        stimulus_ids: Optional[List[int]] = None,
        response_content_ids: Optional[List[int]] = None,
    ) -> None:
        if not readout.learning_enabled or len(response_ids) < 2:
            return
        if not math.isfinite(f_mod):
            raise ValueError("Readout modulation must be finite.")
        with torch.no_grad():
            q_norm = F.normalize(q_vec, p=2, dim=0)
            if stimulus_ids is not None or readout.w_ca3_attractors.shape[0] == 0:
                s_sdr_a = readout._compute_cortical_sdr2048(
                    stimulus_ids if stimulus_ids is not None else response_ids, normalize=False)
                s_sdr_b = readout._compute_cortical_sdr2048(
                    response_content_ids if response_content_ids is not None else response_ids,
                    normalize=False)
                bi_sdr = torch.clamp(s_sdr_a + 0.95 * s_sdr_b, max=1.2)
                readout.sdr_column_usage.add_((bi_sdr > 0).float() * 0.25)
                if readout.w_ca3_attractors.shape[0] == 0:
                    readout.w_ca3_attractors = q_norm.unsqueeze(0).clone()
                    readout.w_ca3_sdr = bi_sdr.unsqueeze(0).clone()
                    readout.w_ca3_salience = torch.tensor([float(f_mod)], dtype=torch.float32)
                else:
                    sims_existing = readout.w_ca3_attractors @ q_norm
                    max_sim, max_idx = torch.max(sims_existing, dim=0)
                    if float(max_sim.item()) >= 0.985:
                        p_idx = int(max_idx.item())
                        readout.w_ca3_sdr[p_idx] = torch.max(readout.w_ca3_sdr[p_idx], bi_sdr)
                        readout.w_ca3_salience[p_idx] = max(
                            float(readout.w_ca3_salience[p_idx].item()), float(f_mod))
                    elif readout.w_ca3_attractors.shape[0] < 4096:
                        readout.w_ca3_attractors = torch.cat(
                            [readout.w_ca3_attractors, q_norm.unsqueeze(0)], dim=0)
                        readout.w_ca3_sdr = torch.cat(
                            [readout.w_ca3_sdr, bi_sdr.unsqueeze(0)], dim=0)
                        readout.w_ca3_salience = torch.cat(
                            [readout.w_ca3_salience,
                             torch.tensor([float(f_mod)], dtype=torch.float32)], dim=0)
                    else:
                        p_idx = int(max_idx.item())
                        readout.w_ca3_attractors[p_idx] = F.normalize(
                            0.92 * readout.w_ca3_attractors[p_idx] + 0.08 * q_norm,
                            p=2, dim=0)
                        readout.w_ca3_sdr[p_idx] = torch.clamp(
                            0.90 * readout.w_ca3_sdr[p_idx] + 0.25 * bi_sdr, max=1.2)
                        readout.w_ca3_salience[p_idx] = max(
                            float(readout.w_ca3_salience[p_idx].item()), float(f_mod))

            for tid in response_ids:
                readout._ensure_token_col(tid)
            v_zero = q_norm
            d_list: List[torch.Tensor] = []
            col_targets: List[int] = []
            for pos in range(len(response_ids)):
                col_targets.append(readout.tok_to_col[int(response_ids[pos])])
                vp2 = v_zero if pos < 3 else response_vecs[pos - 3]
                vp1 = v_zero if pos < 2 else response_vecs[pos - 2]
                vc = v_zero if pos < 1 else response_vecs[pos - 1]
                d_list.append(readout.compute_dendritic_plateau_state(
                    vp2, vp1, vc, pos, q_norm))
            for d_nmda, target_col in zip(d_list, col_targets):
                readout._learn_readout_token(d_nmda, q_norm, target_col, f_mod)
            readout._w_eff_cache = None

    @staticmethod
    def imprint_response_batch_vectorized(
        readout,
        batch_items: List[Tuple[torch.Tensor, List[torch.Tensor], List[int], float, List[int], List[int]]],
        auxiliary_logits: Optional[List[List[torch.Tensor]]] = None,
        memory_query_vecs: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, object]:
        credit_stats = {
            "readout_token_updates": 0, "protected_core_token_updates": 0,
            "readout_step_limited": 0, "readout_max_effective_gain": 0.0,
            "readout_step_factor_sum": 0.0, "readout_step_factor_max": 0.0,
            "bank_token_updates": 0, "bank_token_rejections": 0,
            "bank_backtracking_steps": 0, "bank_limited_mask_updates": 0,
            "bank_update_tokens_requested": 0, "bank_update_tokens_selected": 0,
            "bank_gradient_entries_selected": 0, "bank_memory_budget_rejections": 0,
            "bank_capacity_rejections": 0, "bank_objective_rejections": 0,
            "bank_no_signal_rejections": 0, "bank_other_rejections": 0,
            "bank_pruned_entries": 0,
        }
        if not readout.learning_enabled or not batch_items:
            return credit_stats
        if any(not math.isfinite(item[3]) for item in batch_items):
            raise ValueError("Readout modulation must be finite.")
        if auxiliary_logits is not None and len(auxiliary_logits) != len(batch_items):
            raise ValueError("Auxiliary batches must align with response batches.")
        if memory_query_vecs is not None and len(memory_query_vecs) != len(batch_items):
            raise ValueError("Memory queries must align with response batches.")
        with torch.no_grad():
            for _, _, response_ids, _, _, _ in batch_items:
                for tid in response_ids:
                    readout._ensure_token_col(tid)
            d_all: List[torch.Tensor] = []
            t_all: List[int] = []
            f_all: List[float] = []
            q_all: List[torch.Tensor] = []
            aux_all: List[Optional[torch.Tensor]] = []
            for item_idx, (q_vec, response_vecs, response_ids, f_mod,
                           s_content, r_content) in enumerate(batch_items):
                item_aux = auxiliary_logits[item_idx] if auxiliary_logits is not None else None
                if item_aux is not None and len(item_aux) != len(response_ids):
                    raise ValueError("Auxiliary logits must align with response positions.")
                if len(response_ids) < 2:
                    continue
                q_norm = F.normalize(q_vec, p=2, dim=0)
                memory_q = (q_norm if memory_query_vecs is None else
                            F.normalize(memory_query_vecs[item_idx], dim=0))
                s_sdr_a = readout._compute_cortical_sdr2048(s_content, normalize=False)
                s_sdr_b = readout._compute_cortical_sdr2048(r_content, normalize=False)
                bi_sdr = torch.clamp(s_sdr_a + 0.95 * s_sdr_b, max=1.2)
                readout.sdr_column_usage.add_((bi_sdr > 0).float() * 0.25)
                if readout.w_ca3_attractors.shape[0] == 0:
                    readout.w_ca3_attractors = memory_q.unsqueeze(0).clone()
                    readout.w_ca3_sdr = bi_sdr.unsqueeze(0).clone()
                    readout.w_ca3_salience = torch.tensor([float(f_mod)], dtype=torch.float32)
                else:
                    sims_existing = readout.w_ca3_attractors @ memory_q
                    max_sim, max_idx = torch.max(sims_existing, dim=0)
                    if float(max_sim.item()) >= 0.985:
                        p_idx = int(max_idx.item())
                        readout.w_ca3_sdr[p_idx] = torch.max(readout.w_ca3_sdr[p_idx], bi_sdr)
                        readout.w_ca3_salience[p_idx] = max(
                            float(readout.w_ca3_salience[p_idx].item()), float(f_mod))
                    elif readout.w_ca3_attractors.shape[0] < 4096:
                        readout.w_ca3_attractors = torch.cat(
                            [readout.w_ca3_attractors, memory_q.unsqueeze(0)], dim=0)
                        readout.w_ca3_sdr = torch.cat(
                            [readout.w_ca3_sdr, bi_sdr.unsqueeze(0)], dim=0)
                        readout.w_ca3_salience = torch.cat(
                            [readout.w_ca3_salience,
                             torch.tensor([float(f_mod)], dtype=torch.float32)], dim=0)
                    else:
                        p_idx = int(max_idx.item())
                        readout.w_ca3_attractors[p_idx] = F.normalize(
                            0.92 * readout.w_ca3_attractors[p_idx] + 0.08 * memory_q,
                            p=2, dim=0)
                        readout.w_ca3_sdr[p_idx] = torch.clamp(
                            0.90 * readout.w_ca3_sdr[p_idx] + 0.25 * bi_sdr, max=1.2)
                        readout.w_ca3_salience[p_idx] = max(
                            float(readout.w_ca3_salience[p_idx].item()), float(f_mod))
                for tid in response_ids:
                    readout._ensure_token_col(tid)
                v_zero = q_norm
                for pos in range(len(response_ids)):
                    t_all.append(readout.tok_to_col[int(response_ids[pos])])
                    f_all.append(float(f_mod))
                    q_all.append(q_norm)
                    vp2 = v_zero if pos < 3 else response_vecs[pos - 3]
                    vp1 = v_zero if pos < 2 else response_vecs[pos - 2]
                    vc = v_zero if pos < 1 else response_vecs[pos - 1]
                    d_all.append(readout.compute_dendritic_plateau_state(
                        vp2, vp1, vc, pos, q_norm))
                    aux_all.append(item_aux[pos] if item_aux is not None else None)
            if not d_all:
                return credit_stats
            for d_nmda, target_col, f_mod, q_norm, aux in zip(
                    d_all, t_all, f_all, q_all, aux_all):
                token_stats = readout._learn_readout_token(
                    d_nmda, q_norm, target_col, f_mod, aux)
                for key in credit_stats:
                    if key == "readout_max_effective_gain":
                        credit_stats[key] = max(credit_stats[key], token_stats[key])
                    else:
                        credit_stats[key] += token_stats[key]
            readout._w_eff_cache = None
        return credit_stats
