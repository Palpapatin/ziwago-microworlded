"""Plasticity regulators and their checkpoint-owned state for the readout."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from ziwago.cortex.next16_core import FusiZenkeCascadeMetaplasticityShield


class ReadoutRegulators:
    """Owns the shield, local plasticity regulators, flags, and telemetry."""

    def __init__(self, d_dend: int):
        self.shield = FusiZenkeCascadeMetaplasticityShield(d_dend=int(d_dend), n_cols=0)
        self.fisher_anchor_enabled = True
        self.fisher_lambda = 0.1
        self.fisher_eta = 1.0
        self.fisher_max_entries = 4096
        self.fisher_top_columns = 32
        self.fisher_decay = 0.9
        self.fisher_anchor = None
        self.mca_enabled = True
        self.mca_tau_m = 1000.0
        self.mca_beta_m = 4.0
        self.mca_theta0 = 0.1
        self.mca_half_sat = 1.0
        self.mca_max_entries = 4096
        self.mca_top_columns = 32
        self.mca_dt = 1.0
        self.mca = None
        self.acl_enabled = True
        self.acl_tile_size = 16
        self.acl_tau_c = 2500.0
        self.acl_dt = 50.0
        self.acl_readiness_threshold = 0.5
        self.acl_min_plastic_fraction = 0.25
        self.acl_inh_tone = 0.5
        self.acl_itpc_half_sat = 0.5
        self.acl = None
        self.dcc_enabled = True
        self.dcc_tau_mature = 1000.0
        self.dcc_theta_r = 0.60
        self.dcc_window_w = 50
        self.dcc = None
        self.telemetry: Dict[str, object] = {}

    def apply_local_update(
        self,
        readout,
        w: torch.Tensor,
        d_nmda: torch.Tensor,
        error: torch.Tensor,
        scale: float,
        protect: bool = False,
        row_gain: Optional[torch.Tensor] = None,
    ) -> None:
        if not readout.learning_enabled:
            return
        rows = torch.nonzero(d_nmda, as_tuple=False).flatten()
        n_active = error.numel()
        delta_rows = torch.outer(d_nmda.index_select(0, rows), error) * float(scale)
        if row_gain is not None:
            delta_rows = delta_rows * row_gain.to(delta_rows).unsqueeze(1)
        self.shield.apply_sparse_update(w, rows, delta_rows, n_active)
        if protect:
            self.shield.protect_rows(w, rows, n_active)

    def dcc_readiness(
        self,
        readout,
        d_nmda: torch.Tensor,
        rows: torch.Tensor,
        target_prob: float,
        n_tiles: int,
        ts: int,
    ) -> torch.Tensor:
        """Readiness от DCC для HARD-консолидации ACL."""
        from ziwago.cortex.dcc import DevelopmentalReadinessController

        if self.dcc is None:
            self.dcc = DevelopmentalReadinessController(
                n_zones=n_tiles,
                tau_mature=self.dcc_tau_mature,
                theta_r=self.dcc_theta_r,
                window_w=self.dcc_window_w,
            )
        tile_of_row = rows // ts
        n_events = torch.bincount(tile_of_row, minlength=n_tiles).to(d_nmda.dtype)
        exc = torch.zeros(n_tiles, dtype=d_nmda.dtype, device=d_nmda.device)
        exc.index_add_(0, tile_of_row, d_nmda.index_select(0, rows))
        exc = exc / ts
        itpc = (exc / (exc + self.acl_itpc_half_sat)).clamp(0.0, 1.0)
        inh = torch.full_like(exc, self.acl_inh_tone)
        pred = torch.full_like(exc, float(target_prob))
        obs = {"itpc": itpc, "pred_accuracy": pred, "exc": exc,
               "inh": inh, "n_events": n_events}
        return self.dcc.tick(obs, dt=self.acl_dt)

    def acl_row_gain(
        self, readout, d_nmda: torch.Tensor, rows: torch.Tensor, target_prob: float
    ) -> torch.Tensor:
        """ACL-гейт на строки в единой объявленной точке row_gain."""
        from ziwago.cortex.tiled_glia import TiledGlialControl

        ts = self.acl_tile_size
        n_tiles = (readout.d_dend + ts - 1) // ts
        if self.acl is None:
            self.acl = TiledGlialControl(
                n_tiles=n_tiles,
                tile_size=ts,
                tau_c=self.acl_tau_c,
                mode="on",
                readiness_threshold=self.acl_readiness_threshold,
                min_plastic_fraction=self.acl_min_plastic_fraction,
            )
        padded = torch.zeros(n_tiles * ts, dtype=d_nmda.dtype, device=d_nmda.device)
        padded[: readout.d_dend] = d_nmda
        spikes = padded.view(n_tiles, ts)
        readiness = (
            self.dcc_readiness(readout, d_nmda, rows, target_prob, n_tiles, ts)
            if bool(getattr(self, "dcc_enabled", True)) else None
        )
        gate_tiles = self.acl.step(spikes, dt=self.acl_dt, awake=True, readiness=readiness)
        return gate_tiles.index_select(0, rows // ts)

    def learn_token(
        self,
        readout,
        d_nmda: torch.Tensor,
        q_norm: torch.Tensor,
        target_col: int,
        f_mod: float,
        auxiliary_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        r = readout
        stats: Dict[str, object] = {
            "readout_token_updates": 0, "protected_core_token_updates": 0,
            "readout_step_limited": 0, "readout_max_effective_gain": 0.0,
            "readout_step_factor_sum": 0.0, "readout_step_factor_max": 0.0,
            "bank_token_updates": 0, "bank_token_rejections": 0,
            "bank_backtracking_steps": 0, "bank_limited_mask_updates": 0,
            "bank_update_tokens_requested": 0, "bank_update_tokens_selected": 0,
            "bank_gradient_entries_selected": 0, "bank_memory_budget_rejections": 0,
            "bank_capacity_rejections": 0, "bank_objective_rejections": 0,
            "bank_no_signal_rejections": 0, "bank_other_rejections": 0,
            "bank_pruned_entries": 0, "fisher_entries_selected": 0,
            "fisher_entries_added": 0, "fisher_entries_evicted": 0,
            "fisher_proximal_applied": 0, "fisher_proximal_max_displacement": 0.0,
            "mca_entries_selected": 0, "mca_entries_added": 0,
            "mca_entries_evicted": 0, "mca_min_row_susceptibility": 1.0,
            "acl_min_gate": 1.0, "acl_mean_gate": 1.0,
            "dcc_min_plasticity": 1.0, "dcc_mean_plasticity": 1.0,
            "fisher_enabled": False, "mca_enabled": False,
            "acl_enabled": False, "dcc_enabled": False,
            "fisher_created": False, "mca_created": False,
            "acl_created": False, "dcc_created": False,
            "dcc_g_inh_connected": False, "dcc_sigma_noise_connected": False,
        }

        def set_module_presence():
            stats.update(
                fisher_enabled=bool(self.fisher_anchor_enabled),
                mca_enabled=bool(self.mca_enabled),
                acl_enabled=bool(self.acl_enabled),
                dcc_enabled=bool(self.dcc_enabled),
                fisher_created=self.fisher_anchor is not None,
                mca_created=self.mca is not None,
                acl_created=self.acl is not None,
                dcc_created=self.dcc is not None,
            )
            self.telemetry = dict(stats)

        if not r.learning_enabled:
            set_module_presence()
            return stats
        if not math.isfinite(f_mod):
            raise ValueError("Readout modulation must be finite.")
        if f_mod <= 0:
            set_module_presence()
            return stats
        n_active = len(r.col_to_tok)
        components = r._readout_components(d_nmda, q_norm, n_active)
        logits = r.logit_scale * sum(
            components.values(), torch.zeros_like(next(iter(components.values()))))
        if auxiliary_logits is not None:
            if auxiliary_logits.numel() != n_active:
                raise ValueError("Auxiliary logits must align with the active vocabulary.")
            logits = logits + auxiliary_logits.to(logits)
        error = -torch.softmax(logits, dim=0)
        error[target_col] += 1.0
        rows = torch.nonzero(d_nmda, as_tuple=False).flatten()
        if not rows.numel():
            set_module_presence()
            return stats

        fisher_stats = None
        if self.fisher_anchor_enabled:
            from ziwago.learning.local_fisher import SparseFisherAnchor

            if self.fisher_anchor is None:
                self.fisher_anchor = SparseFisherAnchor(
                    self.fisher_max_entries, self.fisher_decay)
            fisher_stats = self.fisher_anchor.observe(
                r.w_dend_out, d_nmda.index_select(0, rows), error, r.logit_scale,
                rows.tolist(), target_col, self.fisher_top_columns)

        mca_stats = None
        row_gain = None
        if self.mca_enabled:
            from ziwago.learning.mca import MetaplasticSusceptibility

            if self.mca is None:
                self.mca = MetaplasticSusceptibility(
                    self.mca_max_entries, self.mca_tau_m, self.mca_beta_m,
                    self.mca_theta0, self.mca_half_sat)
            mca_stats = self.mca.observe(
                r.w_dend_out, d_nmda.index_select(0, rows), error,
                rows.tolist(), target_col, self.mca_dt, self.mca_top_columns)
            row_gain = self.mca.row_susceptibility(rows.tolist(), n_active)
            mca_stats["min_row_susceptibility"] = (
                float(row_gain.min()) if row_gain.numel() else 1.0)

        acl_stats = None
        if self.acl_enabled:
            target_prob = 1.0 - float(error[target_col])
            acl_gain = self.acl_row_gain(r, d_nmda, rows, target_prob)
            row_gain = acl_gain if row_gain is None else row_gain * acl_gain.to(row_gain)
            acl_stats = {
                "min_gate": float(acl_gain.min()) if acl_gain.numel() else 1.0,
                "mean_gate": float(acl_gain.mean()) if acl_gain.numel() else 1.0,
            }

        dcc_stats = None
        if self.dcc_enabled:
            if self.dcc is None:
                from ziwago.cortex.dcc import DevelopmentalReadinessController

                n_tiles = (r.d_dend + self.acl_tile_size - 1) // self.acl_tile_size
                self.dcc = DevelopmentalReadinessController(
                    n_zones=n_tiles, tau_mature=self.dcc_tau_mature,
                    theta_r=self.dcc_theta_r, window_w=self.dcc_window_w)
            if self.dcc.age.sum().item() == 0:
                n_tiles = self.dcc.n_zones
                tile_size = max(1, int(self.acl_tile_size))
                tile_of_row = rows // tile_size
                n_events = torch.bincount(tile_of_row, minlength=n_tiles).to(d_nmda.dtype)
                exc = torch.zeros(n_tiles, dtype=d_nmda.dtype, device=d_nmda.device)
                exc.index_add_(0, tile_of_row, d_nmda.index_select(0, rows))
                exc = exc / tile_size
                itpc = (exc / (exc + self.acl_itpc_half_sat)).clamp(0.0, 1.0)
                inh = torch.full_like(exc, self.acl_inh_tone)
                pred = torch.full_like(exc, 1.0 - float(error[target_col]))
                self.dcc.tick({"itpc": itpc, "pred_accuracy": pred, "exc": exc,
                               "inh": inh, "n_events": n_events}, dt=self.acl_dt)
            plasticity = self.dcc.effective_plasticity()
            tile_of_row = (rows // max(1, int(self.acl_tile_size))).clamp_max(
                max(0, int(plasticity.numel()) - 1))
            zone_gain = plasticity.index_select(0, tile_of_row)
            if row_gain is None:
                row_gain = zone_gain.to(d_nmda.dtype)
            else:
                row_gain = row_gain * zone_gain.to(row_gain.dtype)
            dcc_stats = {
                "min_plasticity": float(zone_gain.min()) if zone_gain.numel() else 1.0,
                "mean_plasticity": float(zone_gain.mean()) if zone_gain.numel() else 1.0,
            }

        self.shield._ensure(r.w_dend_out, n_active)
        slow = self.shield.u_slow.index_select(0, rows).unsqueeze(1)
        gain = (d_nmda.index_select(0, rows).square().unsqueeze(1)
                / (1 + self.shield.kappa_shield * slow)).sum(0)
        protected = f_mod >= 2.0
        plastic_scale = r.logit_scale * 0.85 * float(f_mod)
        core_scale = r.logit_scale * r.core_readout_gain * 0.65 * float(f_mod) if protected else 0.0
        max_gain = float(gain.max()) * r.logit_scale * (
            plastic_scale + r.core_readout_gain * core_scale)
        step = min(1.0, 2.0 / max_gain) if max_gain > 0 else 1.0
        self.apply_local_update(
            r, r.w_dend_out, d_nmda, error, plastic_scale * step, row_gain=row_gain)
        if protected:
            self.apply_local_update(
                r, r.w_core_dend_out, d_nmda, error, core_scale * step, row_gain=row_gain)
            self.shield.protect_rows(r.w_dend_out, rows, n_active)
        if fisher_stats is not None:
            pull = self.fisher_anchor.proximal_pull(
                r.w_dend_out, self.fisher_eta, self.fisher_lambda)
            fisher_stats.update(
                proximal_applied=pull["applied"],
                proximal_max_displacement=pull["max_displacement"])

        bank_stats = r._learn_bank(q_norm, d_nmda, target_col, auxiliary_logits)
        stats.update(
            readout_token_updates=1,
            protected_core_token_updates=int(protected),
            readout_step_limited=int(step < 1.0),
            readout_max_effective_gain=max_gain * step,
            readout_step_factor_sum=step,
            readout_step_factor_max=step if step > stats.get("readout_step_factor_max", 0.0)
            else stats.get("readout_step_factor_max", 0.0),
            bank_token_updates=int(bank_stats["accepted"]),
            bank_token_rejections=int(not bank_stats["accepted"]),
            bank_backtracking_steps=bank_stats.get("backtracking_steps", 0),
            bank_limited_mask_updates=int(bank_stats.get("update_mask_limited", False)),
            bank_update_tokens_requested=bank_stats.get("update_tokens_requested", 0),
            bank_update_tokens_selected=bank_stats.get("update_tokens_selected", 0),
            bank_gradient_entries_selected=bank_stats.get("gradient_entries_selected", 0),
            bank_pruned_entries=bank_stats.get("pruned_entries", 0))
        if fisher_stats is not None:
            stats.update(
                fisher_entries_selected=fisher_stats.get("selected", 0),
                fisher_entries_added=fisher_stats.get("added", 0),
                fisher_entries_evicted=fisher_stats.get("evicted", 0),
                fisher_proximal_applied=fisher_stats.get("proximal_applied", 0),
                fisher_proximal_max_displacement=fisher_stats.get(
                    "proximal_max_displacement", 0.0))
        if mca_stats is not None:
            stats.update(
                mca_entries_selected=mca_stats.get("selected", 0),
                mca_entries_added=mca_stats.get("added", 0),
                mca_entries_evicted=mca_stats.get("evicted", 0),
                mca_min_row_susceptibility=mca_stats.get(
                    "min_row_susceptibility", 1.0))
        if acl_stats is not None:
            stats.update(
                acl_min_gate=acl_stats.get("min_gate", 1.0),
                acl_mean_gate=acl_stats.get("mean_gate", 1.0))
        if dcc_stats is not None:
            stats.update(
                dcc_min_plasticity=dcc_stats.get("min_plasticity", 1.0),
                dcc_mean_plasticity=dcc_stats.get("mean_plasticity", 1.0))
        stats.update(
            fisher_enabled=bool(self.fisher_anchor_enabled),
            mca_enabled=bool(self.mca_enabled),
            acl_enabled=bool(self.acl_enabled),
            dcc_enabled=bool(self.dcc_enabled),
            fisher_created=self.fisher_anchor is not None,
            mca_created=self.mca is not None,
            acl_created=self.acl is not None,
            dcc_created=self.dcc is not None,
            dcc_g_inh_connected=False,
            dcc_sigma_noise_connected=False)
        self.telemetry = dict(stats)
        if not bank_stats["accepted"]:
            reason_key = {
                "memory_budget_exhausted": "bank_memory_budget_rejections",
                "bank_budget_exhausted": "bank_capacity_rejections",
                "objective_not_improved": "bank_objective_rejections",
                "no_learning_signal": "bank_no_signal_rejections",
            }.get(bank_stats.get("reason"), "bank_other_rejections")
            stats[reason_key] = 1
        return stats

    def export_state_dict(self) -> Dict[str, object]:
        return {
            "fisher_enabled": bool(self.fisher_anchor_enabled),
            "fisher_created": self.fisher_anchor is not None,
            "fisher_state": self.fisher_anchor.export_state_dict()
            if self.fisher_anchor is not None else None,
            "mca_enabled": bool(self.mca_enabled),
            "mca_created": self.mca is not None,
            "mca_state": self.mca.export_state_dict() if self.mca is not None else None,
            "acl_enabled": bool(self.acl_enabled),
            "acl_created": self.acl is not None,
            "acl_state": self.acl.export_state_dict() if self.acl is not None else None,
            "dcc_enabled": bool(self.dcc_enabled),
            "dcc_created": self.dcc is not None,
            "dcc_state": self.dcc.export_state_dict() if self.dcc is not None else None,
        }

    def load_state_dict(self, module_presence: Dict[str, object]) -> None:
        from ziwago.cortex.dcc import DevelopmentalReadinessController
        from ziwago.cortex.tiled_glia import TiledGlialControl
        from ziwago.learning.local_fisher import SparseFisherAnchor
        from ziwago.learning.mca import MetaplasticSusceptibility

        self.fisher_anchor_enabled = bool(module_presence.get("fisher_enabled", True))
        fisher_state = module_presence.get("fisher_state")
        self.fisher_anchor = None
        if isinstance(fisher_state, dict):
            self.fisher_anchor = SparseFisherAnchor()
            self.fisher_anchor.load_state_dict(fisher_state)
        self.mca_enabled = bool(module_presence.get("mca_enabled", True))
        mca_state = module_presence.get("mca_state")
        self.mca = None
        if isinstance(mca_state, dict):
            self.mca = MetaplasticSusceptibility()
            self.mca.load_state_dict(mca_state)
        self.acl_enabled = bool(module_presence.get("acl_enabled", True))
        acl_state = module_presence.get("acl_state")
        self.acl = None
        if isinstance(acl_state, dict):
            self.acl = TiledGlialControl(
                n_tiles=int(acl_state["n_tiles"]),
                tile_size=int(acl_state["tile_size"]))
            self.acl.load_state_dict(acl_state)
        self.dcc_enabled = bool(module_presence.get("dcc_enabled", True))
        dcc_state = module_presence.get("dcc_state")
        self.dcc = None
        if isinstance(dcc_state, dict):
            self.dcc = DevelopmentalReadinessController(
                n_zones=int(dcc_state["n_zones"]))
            self.dcc.load_state_dict(dcc_state)
        self.telemetry = {
            f"{name}_{suffix}": bool(module_presence.get(f"{name}_{suffix}", False))
            for name in ("fisher", "mca", "acl", "dcc")
            for suffix in ("enabled", "created")
        }
