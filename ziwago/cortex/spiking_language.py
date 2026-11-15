"""Local supervised language readout of the existing multicompartment MoE.

No surrogate neurons, global backpropagation, or synthetic firing-rate features:
spikes and L5 output come directly from SparselyGatedCorticalMoE.step.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from .moe_cortex import export_runtime_paths, export_runtime_tree, load_runtime_paths, load_runtime_tree


class SpikingLanguageLoop:
    MODES = frozenset(("normal", "disabled", "shuffled", "constant"))

    def __init__(self, moe_cortex, seed: int = 19, learning_rate: float = 0.1):
        self.moe_cortex = moe_cortex
        self.learning_rate = float(learning_rate)
        self.mode = "normal"
        self.learning_enabled = True
        self.n_spikes = int(moe_cortex.neurons_per_column)
        self.n_l5 = self.n_spikes - 2 * self.n_spikes // 3
        self.d_features = self.n_spikes + self.n_l5
        self.token_ids = []
        self.token_to_col = {}
        self.w_projector = torch.empty(self.d_features, 0)
        self.generator = torch.Generator().manual_seed(seed)
        self.shuffle_indices = torch.randperm(self.d_features, generator=self.generator)
        self.constant_features = F.normalize(torch.ones(self.d_features), dim=0)
        self.last_features = torch.zeros(self.d_features)
        self.last_token_error = torch.empty(0)
        self.last_apical_error = torch.zeros(self.n_spikes)
        self.total_steps = 0
        self.total_updates = 0
        self.last_loss = 0.0
        self.last_spike_raster = torch.empty((0, moe_cortex.n_columns))

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unknown cortical ablation mode: {mode}")
        self.mode = mode

    def ensure_tokens(self, token_ids: Sequence[int]) -> None:
        if not self.learning_enabled:
            return
        new = [int(t) for t in dict.fromkeys(token_ids) if int(t) not in self.token_to_col]
        if not new:
            return
        start = len(self.token_ids)
        for i, tid in enumerate(new, start):
            self.token_to_col[tid] = i
            self.token_ids.append(tid)
        init = torch.randn(self.d_features, len(new), generator=self.generator) * 0.001
        self.w_projector = torch.cat((self.w_projector, init.to(self.w_projector.device)), dim=1)

    def cortical_features(self, cortical_out, mode: Optional[str] = None) -> torch.Tensor:
        mode = self.mode if mode is None else mode
        if mode not in self.MODES:
            raise ValueError(f"Unknown cortical ablation mode: {mode}")
        spikes = cortical_out.combined_spikes.detach().float().mean(dim=0)
        l5 = cortical_out.combined_l5_drive.detach().float().mean(dim=0)
        features = F.normalize(torch.cat((F.normalize(spikes, dim=0), F.normalize(l5, dim=0))), dim=0)
        if features.numel() != self.d_features:
            raise ValueError("MoE cortical output dimensions changed")
        if mode == "disabled":
            return torch.zeros_like(features)
        if mode == "constant":
            return self.constant_features.to(features)
        if mode == "shuffled":
            return features.index_select(0, self.shuffle_indices.to(features.device))
        return features

    def predict(self, cortical_out, token_ids: Sequence[int], mode: Optional[str] = None) -> torch.Tensor:
        features = self.cortical_features(cortical_out, mode)
        self.last_features = features.clone()
        scores = torch.zeros(len(token_ids), device=features.device)
        known = [(i, self.token_to_col[int(t)]) for i, t in enumerate(token_ids) if int(t) in self.token_to_col]
        if known:
            positions, cols = zip(*known)
            scores[list(positions)] = features @ self.w_projector[:, list(cols)].to(features)
        return scores

    def learn(self, cortical_out, target_token: int, token_ids: Optional[Sequence[int]] = None,
              task_modulation: float = 1.0, auxiliary_logits: Optional[torch.Tensor] = None,
              modulators: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        if not self.learning_enabled or self.mode != "normal" or task_modulation <= 0.0:
            return {"loss": self.last_loss, "apical_error_norm": 0.0, "updated": False}
        candidate_ids = list(token_ids) if token_ids is not None else [int(target_token)]
        if auxiliary_logits is not None and int(target_token) not in candidate_ids:
            raise ValueError("Counterfactual candidates must include the target token.")
        self.ensure_tokens(candidate_ids + [int(target_token)])
        features = self.cortical_features(cortical_out)
        with torch.no_grad():
            if auxiliary_logits is None:
                weights = self.w_projector.to(features)
                logits = features @ weights
                target_position = self.token_to_col[int(target_token)]
                columns = None
            else:
                if auxiliary_logits.numel() != len(candidate_ids):
                    raise ValueError("Auxiliary logits must align with candidate token_ids.")
                columns = torch.tensor([self.token_to_col[int(t)] for t in candidate_ids],
                                       dtype=torch.long, device=self.w_projector.device)
                weights = self.w_projector.index_select(1, columns).to(features)
                logits = features @ weights + auxiliary_logits.to(features)
                target_position = candidate_ids.index(int(target_token))
            error = -torch.softmax(logits, dim=0)
            error[target_position] += 1.0
            self.last_loss = float(-torch.log_softmax(logits, dim=0)[target_position])
            # Local transpose credit assignment at the token synapse; no global BPTT.
            credit = weights @ error
            apical = credit[:self.n_spikes].clone()
            apical[2 * self.n_spikes // 3:] += credit[self.n_spikes:]
            apical = apical.clamp(-1.5, 1.5)
            mod = min(2.5, max(0.0, float(task_modulation)))
            # Нейрохимия доходит до синапса СВОИМИ каналами, а не одним
            # скаляром: ацетилхолин (мускариновый гейт колонки) и фазовая
            # дофаминовая передача (D1/D2-гейт) идут в апикальную инъекцию
            # ошибки, где для них уже есть места в интерфейсе колонки.
            ach = float((modulators or {}).get("ach", 0.5))
            da_phasic = (modulators or {}).get("da_phasic")
            for cid in cortical_out.active_expert_ids:
                col = self.moe_cortex.columns[cid]
                gate = float(cortical_out.gating_weights[cid])
                eff = col.inject_apical_error(apical * gate, f_mod=mod,
                                              ach=ach, da_phasic=da_phasic)
                # Existing injection learns FF/REC; explicitly train the apical branch gate too.
                col.dendrite.learn(eff, col._last_x_ff, apical_error=eff, f_mod=mod * gate)
            update = torch.outer(features.to(self.w_projector), error.to(self.w_projector))
            if columns is None:
                self.w_projector.add_(update, alpha=self.learning_rate * mod)
            else:
                self.w_projector.index_add_(1, columns, update, alpha=self.learning_rate * mod)
            self.last_features = features.clone()
            self.last_token_error = error.clone()
            self.last_apical_error = apical.clone()
            self.total_updates += 1
        return {"loss": self.last_loss, "apical_error_norm": float(apical.norm()), "updated": True}

    def step(self, x_in: torch.Tensor, token_ids: Sequence[int], target_token: Optional[int] = None,
             learn: bool = False, task_modulation: float = 1.0, mode: Optional[str] = None,
             cortical_step=None):
        do_learn = bool(learn and self.learning_enabled and self.mode == "normal"
                        and (mode is None or mode == "normal"))
        # Unsupervised MoE plasticity is not a substitute for the task-specific apical error.
        cortical_out = (self.moe_cortex.step(x_in=x_in, learn=False, f_mod=task_modulation)
                        if cortical_step is None else cortical_step(x_in))
        scores = self.predict(cortical_out, token_ids, mode)
        counts = getattr(cortical_out, "spike_count_window", None)
        if counts is None:
            active = set(cortical_out.active_expert_ids)
            counts = torch.tensor([[float(self.moe_cortex._last_col_spikes[i].sum()) if i in active else 0.0
                                    for i in range(self.moe_cortex.n_columns)]])
        self.last_spike_raster = torch.cat((self.last_spike_raster, counts.to(self.last_spike_raster)), dim=0)[-512:]
        self.total_steps += 1
        if do_learn and target_token is not None:
            self.learn(cortical_out, target_token, token_ids, task_modulation)
        return scores, cortical_out

    def export_dynamic_state_dict(self) -> Dict[str, object]:
        return export_runtime_paths(self, [(name,) for name in (
            "generator", "last_features", "last_token_error", "last_apical_error",
            "total_steps", "total_updates", "last_loss", "last_spike_raster",
        )])

    def load_dynamic_state_dict(self, state: Dict[str, object]) -> None:
        load_runtime_paths(self, state)

    def export_state_dict(self) -> Dict[str, object]:
        return export_runtime_tree({k: v for k, v in vars(self).items() if k != "moe_cortex"})

    def load_state_dict(self, state: Dict[str, object],
                        remove_missing_attributes: bool = False) -> None:
        restored = load_runtime_tree(
            {k: v for k, v in vars(self).items() if k != "moe_cortex"},
            state,
            _remove_missing_attributes=remove_missing_attributes,
        )
        for key in list(vars(self)):
            if key != "moe_cortex" and key not in restored and not callable(getattr(self, key)):
                delattr(self, key)
        for key, val in restored.items():
            setattr(self, key, val)
