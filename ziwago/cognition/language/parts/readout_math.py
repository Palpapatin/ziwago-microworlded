"""Numerical readout and scoring operations for the language readout."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


class ReadoutMath:
    """Stateless operations over the readout's established state."""

    @staticmethod
    def resolve_episode_pole(readout, q_vec: torch.Tensor):
        if readout.w_ca3_attractors.shape[0] == 0:
            return None, 0.0
        q_norm = F.normalize(q_vec.detach().float(), p=2, dim=0)
        sims = readout.w_ca3_attractors @ q_norm
        value, index = torch.max(sims, dim=0)
        resonance = float(value.item())
        if resonance < float(readout.apical_min_resonance):
            return None, resonance
        return int(index.item()), resonance

    @staticmethod
    def get_linked_next_episode_pole(readout, pole_vec: torch.Tensor):
        if readout.w_ca3_attractors.shape[0] == 0 or not readout.w_ca3_episode_links:
            return None
        with torch.no_grad():
            p1 = F.normalize(pole_vec, p=2, dim=0)
            sims = readout.w_ca3_attractors @ p1
            max_sim, max_idx = torch.max(sims, dim=0)
            if float(max_sim.item()) < 0.85:
                return None
            idx1 = int(max_idx.item())
            idx2 = readout.w_ca3_episode_links.get(idx1)
            if idx2 is not None and 0 <= idx2 < readout.w_ca3_attractors.shape[0]:
                return readout.w_ca3_attractors[idx2]
            return None

    @staticmethod
    def clean_up_stimulus_in_ca3_attractor(
        readout,
        q_raw: torch.Tensor,
        q_sdr2048: Optional[torch.Tensor] = None,
        beta: float = 18.0,
        recent_poles: Optional[List[torch.Tensor]] = None,
    ):
        q_norm = F.normalize(q_raw, p=2, dim=0)
        readout.last_ca3_no_match = False
        readout.last_consolidated_boost_applied = False
        if readout.w_ca3_attractors.shape[0] == 0:
            readout.last_top_pole_idx = -1
            readout.last_top_pole_resonance = 0.0
            readout.last_ca3_no_match = True
            return q_norm, []
        with torch.no_grad():
            cos_sim = (readout.w_ca3_attractors @ q_norm).clamp(min=0.0)
            sdr_drive = torch.zeros_like(cos_sim)
            raw_sdr_overlap = torch.zeros_like(cos_sim, dtype=torch.bool)
            if q_sdr2048 is not None and readout.w_ca3_sdr.shape[0] == cos_sim.shape[0]:
                idf_col = 1.0 / torch.sqrt(readout.sdr_column_usage)
                q_weighted = F.normalize(q_sdr2048 * idf_col + 1e-8, p=2, dim=0)
                row_weighted = readout.w_ca3_sdr * idf_col
                row_l2 = row_weighted.norm(p=2, dim=1).clamp(min=1e-8)
                sdr_drive = (row_weighted @ q_weighted) / row_l2
                raw_sdr_overlap = sdr_drive > 0.0125
                sdr_drive = torch.where(
                    cos_sim > 0.01, sdr_drive, torch.zeros_like(sdr_drive))

            has_sdr_overlap = (sdr_drive > 0.0125).float()
            turn_relevance = cos_sim * (0.30 + 0.25 * has_sdr_overlap) + 0.60 * sdr_drive
            if readout.w_ca3_salience.shape[0] == turn_relevance.shape[0]:
                salience_gate = 1.0 + 0.15 * torch.tanh(readout.w_ca3_salience - 1.0)
                sims = turn_relevance * salience_gate
            else:
                sims = turn_relevance
            if recent_poles:
                for rp in recent_poles[-3:]:
                    rp_norm = F.normalize(rp, p=2, dim=0)
                    recent_match = (readout.w_ca3_attractors @ rp_norm) > 0.92
                    weak_retrigger = sdr_drive < 0.155
                    habituation_mask = recent_match & weak_retrigger
                    sims = torch.where(habituation_mask, sims * 0.45, sims)

            k = min(3, int(sims.shape[0]))
            top_v, top_i = torch.topk(sims, k=k)
            best_score = float(top_v[0].item())
            if best_score < 0.12:
                readout.last_top_pole_idx = -1
                readout.last_top_pole_resonance = 0.0
                readout.last_top_pole_salience = 1.0
                readout.last_consolidated_boost_applied = False
                readout.last_ca3_no_match = (
                    not bool(raw_sdr_overlap.any().item())
                    and not bool((cos_sim > 0.01).any().item()))
                return q_norm, []

            poles = [
                (float(v.item()), readout.w_ca3_attractors[int(idx.item())])
                for v, idx in zip(top_v, top_i)
            ]
            best_p_idx = int(top_i[0].item())
            readout.last_top_pole_idx = best_p_idx
            readout.last_top_pole_resonance = round(best_score, 4)
            readout.last_top_pole_salience = (
                round(float(readout.w_ca3_salience[best_p_idx].item()), 4)
                if readout.w_ca3_salience.shape[0] > best_p_idx
                else 1.0
            )
            return poles[0][1], poles

    @staticmethod
    def discover_second_hop_ca3_pole(
        readout,
        first_pole_vec: torch.Tensor,
        stimulus_sdr2048: torch.Tensor,
        recent_poles: Optional[List[torch.Tensor]] = None,
    ):
        if (readout.w_ca3_attractors.shape[0] < 2
                or readout.w_ca3_sdr.shape[0] != readout.w_ca3_attractors.shape[0]):
            return None
        with torch.no_grad():
            p1_norm = F.normalize(first_pole_vec, p=2, dim=0)
            cos_p1 = readout.w_ca3_attractors @ p1_norm
            p1_idx = int(torch.argmax(cos_p1).item())
            sdr_p1 = readout.w_ca3_sdr[p1_idx]
            delta_sdr_b = torch.clamp(sdr_p1 - 0.85 * stimulus_sdr2048, min=0.0)
            if float(delta_sdr_b.sum().item()) < 1e-4:
                delta_sdr_b = sdr_p1
            idf_col = 1.0 / torch.sqrt(readout.sdr_column_usage)
            b_weighted = F.normalize(delta_sdr_b * idf_col + 1e-8, p=2, dim=0)
            row_weighted = readout.w_ca3_sdr * idf_col
            row_l2 = row_weighted.norm(p=2, dim=1).clamp(min=1e-8)
            hop2_drive = (row_weighted @ b_weighted) / row_l2
            if readout.w_ca3_salience.shape[0] == hop2_drive.shape[0]:
                sal_gate = 1.0 + 0.15 * torch.tanh(readout.w_ca3_salience - 1.0)
                hop2_scores = hop2_drive * sal_gate
                k_top_sal = min(12, int(readout.w_ca3_salience.shape[0]))
                sal_thresh = float(torch.topk(
                    readout.w_ca3_salience, k=k_top_sal).values[-1].item())
                consolidated_mask = readout.w_ca3_salience >= sal_thresh
                if bool(consolidated_mask.any().item()):
                    hop2_scores = torch.where(
                        consolidated_mask, hop2_scores, torch.full_like(hop2_scores, -1e9))
            else:
                hop2_scores = hop2_drive
            too_close = cos_p1 >= 0.85
            hop2_scores = torch.where(
                too_close, torch.full_like(hop2_scores, -1e9), hop2_scores)
            if recent_poles:
                for rp in recent_poles[-2:]:
                    rp_norm = F.normalize(rp, p=2, dim=0)
                    recent_match = (readout.w_ca3_attractors @ rp_norm) > 0.92
                    hop2_scores = torch.where(
                        recent_match, hop2_scores * 0.65, hop2_scores)
            best_val, best_idx = torch.max(hop2_scores, dim=0)
            score = float(best_val.item())
            if score < 0.18:
                return None
            return score, readout.w_ca3_attractors[int(best_idx.item())]

    @staticmethod
    def sparse_logits(
        readout,
        d_nmda: torch.Tensor,
        w: torch.Tensor,
        n_active: int,
        candidate_cols: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        rows = torch.nonzero(d_nmda, as_tuple=False).flatten()
        readout.last_sparse_rows = rows.clone()
        sub = w[:, :n_active].index_select(0, rows)
        cols = readout._validate_candidate_cols(candidate_cols, n_active)
        if cols is not None:
            sub = sub.index_select(1, cols.to(sub.device))
        return d_nmda.index_select(0, rows) @ sub

    @staticmethod
    def readout_components(
        readout,
        d_nmda: torch.Tensor,
        q_norm: torch.Tensor,
        n_active: int,
        candidate_cols: Optional[Sequence[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        r = readout
        cols = r._validate_candidate_cols(candidate_cols, n_active)
        if cols is not None and cols.numel() == 0:
            empty = d_nmda.new_empty(0)
            return {name: empty.clone() for name in ("dendritic", "core", "swr", "bank")}
        dendritic = ReadoutMath.sparse_logits(r, d_nmda, r.w_dend_out, n_active, candidate_cols)
        core = torch.zeros_like(dendritic)
        swr = torch.zeros_like(dendritic)
        if r.w_core_dend_out.shape[1] >= n_active:
            core = r.core_readout_gain * ReadoutMath.sparse_logits(
                r, d_nmda, r.w_core_dend_out, n_active, candidate_cols)
        if r.w_swr_poles.shape[0] > 0 and r.w_swr_dend_out.shape[1] >= n_active:
            swr_sim = float((r.w_swr_poles @ q_norm).max().item())
            if swr_sim > 0.90:
                swr = (swr_sim * 1.15) * ReadoutMath.sparse_logits(
                    r, d_nmda, r.w_swr_dend_out, n_active, candidate_cols)
        bank_tokens = (r.col_to_tok[:n_active] if candidate_cols is None
                       else [r.col_to_tok[int(c)] for c in candidate_cols])
        bank_logits = r.bank_router.predict_sparse(q_norm, d_nmda, bank_tokens).to(dendritic)
        if bank_logits.shape != dendritic.shape:
            raise ValueError("Sparse bank prediction must align with token_ids")
        bank = r.router_residual_bound * torch.tanh(bank_logits)
        return {"dendritic": dendritic, "core": core, "swr": swr, "bank": bank}

    @staticmethod
    def compute_cortical_sdr2048(
        token_ids: List[int], normalize: bool = True
    ) -> torch.Tensor:
        sdr = torch.zeros(2048, dtype=torch.float32)
        for tid in token_ids:
            t = int(tid)
            b0 = (t * 2654435761) % 2048
            b1 = (t * 1597334677 + 31) % 2048
            b2 = (t * 381271489 + 97) % 2048
            sdr[b0] = 1.0
            sdr[b1] = 1.0
            sdr[b2] = 1.0
        if normalize:
            return F.normalize(sdr + 1e-8, p=2, dim=0)
        return sdr

    @staticmethod
    def apical_episode_code(readout, cue64: torch.Tensor) -> torch.Tensor:
        r = readout
        drive = F.relu(cue64 @ r.w_apical_branch)
        k = min(int(r.apical_topk), r.d_dend)
        top_vals, top_idx = torch.topk(drive, k=k)
        sparse = torch.zeros_like(drive)
        sparse.scatter_(0, top_idx, top_vals * top_vals)
        return F.normalize(sparse, p=2, dim=0)

    @staticmethod
    def apical_logits(
        readout,
        d_apical: Optional[torch.Tensor],
        gate: float = 1.0,
        candidate_cols: Optional[Sequence[int]] = None,
    ) -> Optional[torch.Tensor]:
        r = readout
        if d_apical is None or gate <= 0.0:
            return None
        n_active = len(r.col_to_tok)
        if n_active == 0 or r.w_swr_dend_out.shape[1] < n_active:
            return None
        return float(gate) * ReadoutMath.sparse_logits(
            r, d_apical, r.w_swr_dend_out, n_active, candidate_cols)

    @staticmethod
    def compute_cortical_moe_branch_profile(
        readout,
        q_vec: torch.Tensor,
        expert_gates: torch.Tensor,
        expert_names: Sequence[str],
    ) -> Dict[str, float]:
        r = readout
        with torch.no_grad():
            q_norm = F.normalize(q_vec, p=2, dim=0)
            d_probe = r.compute_dendritic_plateau_state(q_norm, q_norm, q_norm, 0, q_norm)
            n_cols = min(16, int(expert_gates.shape[0]))
            bank_sz = r.d_dend // n_cols
            profile: Dict[str, float] = {}
            for c_i in range(n_cols):
                g_val = float(expert_gates[c_i].item())
                if g_val > 0.01:
                    branch_slice = d_probe[c_i * bank_sz : (c_i + 1) * bank_sz]
                    b_energy = float(branch_slice.norm().item()) * (1.0 + 0.5 * g_val)
                    col_name = str(expert_names[c_i]) if c_i < len(expert_names) else f"Col{c_i}"
                    profile[col_name] = round(b_energy, 4)
            return profile

    @staticmethod
    def compute_dendritic_plateau_state(
        readout,
        v_prev2: torch.Tensor,
        v_prev1: torch.Tensor,
        v_cur: torch.Tensor,
        pos_idx: int,
        q_vec: torch.Tensor,
    ) -> torch.Tensor:
        r = readout
        p_vec = r.pos_phase_vecs[min(pos_idx, r.max_pos - 1)]
        b_holo1 = torch.roll(v_prev1, shifts=2, dims=0) * torch.roll(v_cur, shifts=1, dims=0) * p_vec * q_vec
        b_holo2 = torch.roll(v_prev2, shifts=3, dims=0) * torch.roll(v_cur, shifts=1, dims=0) * torch.roll(q_vec, shifts=4, dims=0)
        b_holo3 = v_cur * p_vec * torch.roll(q_vec, shifts=2, dims=0)
        b_holo4 = torch.roll(v_prev1, shifts=1, dims=0) * p_vec * q_vec
        holo_in = F.normalize(torch.cat([b_holo1, b_holo2, b_holo3, b_holo4], dim=0), p=2, dim=0)
        raw_drive = F.relu(holo_in @ r.w_basal_branch)
        top_vals, top_idx = torch.topk(raw_drive, k=min(32, r.d_dend))
        sparse_d = torch.zeros_like(raw_drive)
        sparse_d.scatter_(0, top_idx, top_vals * top_vals)
        return F.normalize(sparse_d, p=2, dim=0)

    @staticmethod
    def predict_next_token_logits(
        readout,
        v_prev2: torch.Tensor,
        v_prev1: torch.Tensor,
        v_cur: torch.Tensor,
        pos_idx: int,
        q_vec: torch.Tensor,
        candidate_cols: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        r = readout
        n_active = len(r.col_to_tok)
        if n_active == 0 or (candidate_cols is not None and len(candidate_cols) == 0):
            return r.w_dend_out.new_empty(0)
        with torch.no_grad():
            q_norm = F.normalize(q_vec, p=2, dim=0)
            d_nmda = r.compute_dendritic_plateau_state(v_prev2, v_prev1, v_cur, pos_idx, q_norm)
            components = r._readout_components(d_nmda, q_norm, n_active, candidate_cols)
            return sum(components.values())

    @staticmethod
    def counterfactual_credit_report(
        readout,
        v_prev2: torch.Tensor,
        v_prev1: torch.Tensor,
        v_cur: torch.Tensor,
        pos_idx: int,
        q_vec: torch.Tensor,
        target_token: int,
        auxiliary_components: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, object]:
        r = readout
        if int(target_token) not in r.tok_to_col:
            raise ValueError("Target token is absent from the active vocabulary.")
        with torch.no_grad():
            q_norm = F.normalize(q_vec, p=2, dim=0)
            d_nmda = r.compute_dendritic_plateau_state(v_prev2, v_prev1, v_cur, pos_idx, q_norm)
            n_active = len(r.col_to_tok)
            components = {name: r.logit_scale * value for name, value in
                          r._readout_components(d_nmda, q_norm, n_active).items()}
            for name, value in (auxiliary_components or {}).items():
                if value.numel() != n_active:
                    raise ValueError("Auxiliary credit components must align with the active vocabulary.")
                components[name] = value.to(d_nmda)
            total = sum(components.values())
            target_col = r.tok_to_col[int(target_token)]
            winner_col = int(total.argmax())
            target_logit = total[target_col]
            winner_logit = total[winner_col]
            report = {
                "target_token": int(target_token),
                "winner_token": r.col_to_tok[winner_col],
                "target_rank": int((total > target_logit).sum()) + 1,
                "target_margin": float(target_logit - winner_logit),
                "target_nll": float(-torch.log_softmax(total, dim=0)[target_col]),
                "components": {},
            }
            for name, values in components.items():
                without = total - values
                report["components"][name] = {
                    "target_logit": float(values[target_col]),
                    "winner_logit": float(values[winner_col]),
                    "target_minus_winner": float(values[target_col] - values[winner_col]),
                    "rms": float(values.square().mean().sqrt()),
                    "centered_rms": float((values - values.mean()).square().mean().sqrt()),
                    "min": float(values.min()), "max": float(values.max()),
                    "target_nll_without": float(-torch.log_softmax(without, dim=0)[target_col]),
                    "target_margin_without": float(without[target_col] - without.max()),
                }
            return report

    @staticmethod
    def predict_next_token_scores(
        readout,
        v_prev2: torch.Tensor,
        v_prev1: torch.Tensor,
        v_cur: torch.Tensor,
        pos_idx: int,
        q_vec: torch.Tensor,
    ) -> Dict[int, float]:
        r = readout
        logits = r.predict_next_token_logits(v_prev2, v_prev1, v_cur, pos_idx, q_vec)
        top_v, top_i = torch.topk(logits, k=min(48, logits.numel()))
        return {r.col_to_tok[idx]: float(val) for val, idx in zip(top_v.tolist(), top_i.tolist())}
