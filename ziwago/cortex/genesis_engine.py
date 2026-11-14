#!/usr/bin/env python3
"""Genesis cognitive components with v19 causal and temporal safeguards.

The world graph stores attributed assertions, grounds controlled clauses, and
propagates declared sufficient/necessary constraints with explicit unknowns. Graph
and BA10 output use transparent constrained rendering, not neural generation as
proof. The theta/gamma codec carries signed values in binary first-spike timing.
Other components retain their existing recurrent generation, contextual role
binding, SDR closure heuristic, and speech-act routing behavior. These mechanisms
are engineering models, not guarantees of scientific truth or biological fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ============================================================================
# 1. НЕПРЕРЫВНАЯ ВОЛНА СИНТАКСИЧЕСКИХ ВАЛЕНТНОСТЕЙ БЕЗ `pos_idx` (0% POS_IDX!)
# ============================================================================
class SyntacticValenceWaveGenerator:
    r"""Дендритный NMDA-декодер, работающий на **Непрерывном Автомате Синтаксических Валентностей Зоны Брока**
    $\vec{\Psi}_{\mathrm{BA44}}(t) \in \mathbb{R}^{64}$ (**0% `pos_idx`, 0% `pos_phase_vecs`**!):
      - В отличие от позиционного индекса `pos_idx = 0, 1, 2...`, вектор валентности $\vec{\Psi}_{\mathrm{BA44}}(t)$
        определяется только динамической траекторией реально сгенерированных синтаксических узлов.
      - Если перед предложением вставлено вводное слово или обращение, как только поток входит в основную
        клаузу, $\vec{\Psi}_{\mathrm{BA44}}(t)$ сходится к аттрактору синтаксического состояния без сбоя фазы!"""

    def __init__(self, d_vec: int = 64, d_dend: int = 8192, top_k_wta: int = 32, seed: int = 1801):
        g = torch.Generator().manual_seed(seed)
        self.d_vec = int(d_vec)
        self.d_dend = int(d_dend)
        self.top_k_wta = int(top_k_wta)

        # Проекторы дендритных ветвей и рекуррентный оператор фазового вращения валентности (БЕЗ pos_phase_vecs и БЕЗ pos_idx!)
        self.w_basal_valence = F.normalize(torch.randn(d_vec * 4, d_dend, generator=g), p=2, dim=0) * 2.5
        self.w_apical_goal = F.normalize(torch.randn(d_vec, d_dend, generator=g), p=2, dim=0) * 3.0
        self.w_valence_rot = F.normalize(torch.randn(d_vec, d_vec, generator=g), p=2, dim=0)

        self.tok_to_col: Dict[int, int] = {}
        self.col_to_tok: List[int] = []
        self.w_valence_dend_out = torch.zeros(d_dend, 2048, dtype=torch.float32)

    def _ensure_token_col(self, token_id: int) -> int:
        tid = int(token_id)
        if tid in self.tok_to_col:
            return self.tok_to_col[tid]
        col = len(self.col_to_tok)
        if col >= self.w_valence_dend_out.shape[1]:
            extra = torch.zeros(self.d_dend, 2048, dtype=torch.float32)
            self.w_valence_dend_out = torch.cat([self.w_valence_dend_out, extra], dim=1)
        self.tok_to_col[tid] = col
        self.col_to_tok.append(tid)
        return col

    @staticmethod
    def initial_valence_state(q_goal_vec: torch.Tensor) -> torch.Tensor:
        r"""Начальный потенциал синтаксической валентности $\vec{\Psi}_{\mathrm{BA44}}(0)$ (активен слот `NEED_CLAUSE_ONSET`)."""
        q_norm = F.normalize(q_goal_vec, p=2, dim=0)
        return F.normalize(torch.roll(q_norm, shifts=3, dims=0) + 0.5 * torch.roll(q_norm, shifts=7, dims=0), p=2, dim=0)

    def advance_valence_state(self, psi_prev: torch.Tensor, emitted_token_vec64: torch.Tensor) -> torch.Tensor:
        r"""Обновляет потенциал синтаксической валентности $\vec{\Psi}_{\mathrm{BA44}}(t)$ после испускания очередного токена
        через нелинейный голографический оператор перехода состояний (БЕЗ использования счётчика шага `pos_idx`!)."""
        psi_rot = torch.roll(psi_prev, shifts=1, dims=0) @ self.w_valence_rot
        v_tok = F.normalize(emitted_token_vec64, p=2, dim=0)
        nonlinear_gate = torch.cos(psi_rot * 3.14159 + torch.roll(v_tok, shifts=2, dims=0) * 2.0)
        psi_next = 0.65 * psi_rot + 0.55 * v_tok + 0.65 * (psi_rot * v_tok) + 0.45 * nonlinear_gate
        return F.normalize(psi_next + 1e-8, p=2, dim=0)

    def compute_valence_dendritic_state(
        self,
        v_prev2: torch.Tensor,
        v_prev1: torch.Tensor,
        v_cur: torch.Tensor,
        psi_valence: torch.Tensor,
        q_goal_vec: torch.Tensor,
    ) -> torch.Tensor:
        r"""Вычисляет разреженное состояние 8 192 дендритных NMDA-ветвей исключительно из локального окна токенов,
        непрерывной волны валентности $\vec{\Psi}_{\mathrm{BA44}}(t)$ и цели $\vec{Q}_{\mathrm{goal}}$ (0% `pos_idx`!)."""
        z_ctx = (
            1.35 * q_goal_vec
            + 1.15 * psi_valence
            + 0.95 * torch.roll(v_cur, shifts=1, dims=0)
            + 0.75 * torch.roll(v_prev1, shifts=2, dims=0)
            + 0.55 * torch.roll(v_prev2, shifts=3, dims=0)
            + 0.85 * (q_goal_vec * psi_valence)
            + 0.75 * (v_cur * psi_valence)
        )
        z_ctx = F.normalize(z_ctx, p=2, dim=0)

        w_proj = self.w_basal_valence[: self.d_vec, :]
        b_proj = self.w_apical_goal[0, :] * 0.1
        u_branch = z_ctx @ w_proj + b_proj

        top_vals, top_idx = torch.topk(u_branch, k=self.top_k_wta, dim=-1)
        theta_k = top_vals[-1]
        d_sparse = torch.zeros_like(u_branch)
        active_vals = F.relu(top_vals - theta_k + 0.05) ** 3
        d_sparse.scatter_(0, top_idx, active_vals)
        return F.normalize(d_sparse + 1e-8, p=2, dim=-1)

    def imprint_batch_zero_pos(
        self,
        batch_trajectories: Sequence[Tuple[torch.Tensor, Sequence[int], float]],
        get_vec64_fn: Callable[[int], torch.Tensor],
        epochs: int = 10,
    ) -> None:
        """Совместное пакетное обучение матрицы `w_valence_dend_out` на всех каузальных траекториях одновременно
        БЕЗ использования `pos_idx` (исключает последовательное затирание ранних уроков поздними!)."""
        if not batch_trajectories:
            return
        with torch.no_grad():
            d_all: List[torch.Tensor] = []
            t_all: List[int] = []
            f_all: List[float] = []

            for q_goal_vec, token_ids, f_mod in batch_trajectories:
                seq = [int(t) for t in token_ids]
                if len(seq) < 2:
                    continue
                q_norm = F.normalize(q_goal_vec, p=2, dim=0)
                for tid in seq:
                    self._ensure_token_col(tid)
                t_vecs = [get_vec64_fn(tid) for tid in seq]
                psi = self.initial_valence_state(q_norm)
                for step_i, tid in enumerate(seq):
                    t_all.append(self.tok_to_col[tid])
                    f_all.append(float(f_mod))
                    vp2 = q_norm if step_i < 3 else t_vecs[step_i - 3]
                    vp1 = q_norm if step_i < 2 else t_vecs[step_i - 2]
                    vc = q_norm if step_i < 1 else t_vecs[step_i - 1]
                    d_all.append(self.compute_valence_dendritic_state(vp2, vp1, vc, psi, q_norm))
                    psi = self.advance_valence_state(psi, t_vecs[step_i])

            if not d_all:
                return
            n_act = len(self.col_to_tok)
            d_mat = torch.stack(d_all)
            t_idx = torch.tensor(t_all, dtype=torch.long)
            f_vec = torch.tensor(f_all, dtype=torch.float32).unsqueeze(1)
            w_act = self.w_valence_dend_out[:, :n_act]

            for _ in range(max(1, int(epochs))):
                logits = (d_mat @ w_act) * 8.0
                probs = torch.softmax(logits, dim=-1)
                err = -probs
                err.scatter_add_(1, t_idx.unsqueeze(1), torch.ones((len(t_all), 1), dtype=err.dtype))
                w_act.add_((d_mat.t() @ (err * f_vec)) * 0.85)

    def generate_trajectory_zero_pos(
        self,
        q_goal_vec: torch.Tensor,
        get_vec64_fn: Callable[[int], torch.Tensor],
        decode_fn: Callable[[int], str],
        max_tokens: int = 44,
        arcuate_pointer_map: Optional[Dict[int, Tuple[int, List[int]]]] = None,
    ) -> List[int]:
        r"""Генерирует предложение по волне валентности $\vec{\Psi}_{\mathrm{BA44}}(t)$ БЕЗ использования `pos_idx`!"""
        n_act = len(self.col_to_tok)
        if n_act == 0:
            return []
        with torch.no_grad():
            q_norm = F.normalize(q_goal_vec, p=2, dim=0)
            psi = self.initial_valence_state(q_norm)
            role_ids: List[int] = []
            out_ids: List[int] = []
            used_bg: Dict[Tuple[int, int], int] = {}
            skip_sub = 0

            for _ in range(max_tokens):
                vp2 = q_norm if len(role_ids) < 3 else get_vec64_fn(role_ids[-3])
                vp1 = q_norm if len(role_ids) < 2 else get_vec64_fn(role_ids[-2])
                vc = q_norm if len(role_ids) < 1 else get_vec64_fn(role_ids[-1])
                d_nmda = self.compute_valence_dendritic_state(vp2, vp1, vc, psi, q_norm)
                logits = d_nmda @ self.w_valence_dend_out[:, :n_act]

                cur = role_ids[-1] if role_ids else None
                top_v, top_i = torch.topk(logits, k=min(24, n_act))
                best_tid = None
                best_sc = -1e9
                for val, col_i in zip(top_v.tolist(), top_i.tolist()):
                    cand_tid = self.col_to_tok[col_i]
                    if cur is not None and cand_tid == cur:
                        continue
                    pen = 8.0 * used_bg.get((cur, cand_tid), 0) if cur is not None else 0.0
                    sc = float(val) - pen
                    if sc > best_sc:
                        best_sc = sc
                        best_tid = cand_tid

                if best_tid is None:
                    break
                if cur is not None:
                    used_bg[(cur, best_tid)] = used_bg.get((cur, best_tid), 0) + 1
                role_ids.append(best_tid)
                psi = self.advance_valence_state(psi, get_vec64_fn(best_tid))

                if skip_sub > 0:
                    skip_sub -= 1
                elif arcuate_pointer_map and int(best_tid) in arcuate_pointer_map:
                    span_l, fil_tids = arcuate_pointer_map[int(best_tid)]
                    out_ids.extend(fil_tids)
                    skip_sub = max(0, int(span_l) - 1)
                else:
                    out_ids.append(best_tid)

                tok_s = decode_fn(best_tid).strip()
                if tok_s.endswith((".", "!", "?")) and len(role_ids) >= 6:
                    break
            return out_ids

    def export_state_dict(self) -> Dict[str, object]:
        n_act = len(self.col_to_tok)
        return {
            "col_to_tok": list(self.col_to_tok),
            "tok_to_col": {int(k): int(v) for k, v in self.tok_to_col.items()},
            "w_basal_valence": self.w_basal_valence.detach().clone(),
            "w_apical_goal": self.w_apical_goal.detach().clone(),
            "w_valence_rot": self.w_valence_rot.detach().clone(),
            "w_valence_dend_out": self.w_valence_dend_out[:, :n_act].detach().clone(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.col_to_tok = [int(t) for t in state["col_to_tok"]]  # type: ignore[index]
        self.tok_to_col = {int(k): int(v) for k, v in state["tok_to_col"].items()}  # type: ignore[index]
        self.w_basal_valence = state["w_basal_valence"].float().clone()  # type: ignore[index]
        self.w_apical_goal = state["w_apical_goal"].float().clone()  # type: ignore[index]
        self.w_valence_rot = state["w_valence_rot"].float().clone()  # type: ignore[index]
        w_out = state["w_valence_dend_out"].float()  # type: ignore[index]
        n_act = len(self.col_to_tok)
        cap = max(2048, ((n_act + 2047) // 2048) * 2048)
        self.w_valence_dend_out = torch.zeros(self.d_dend, cap, dtype=torch.float32)
        self.w_valence_dend_out[:, :n_act] = w_out


# ============================================================================
# 2. КРИСТАЛЛИЗАЦИЯ СЕМАНТИЧЕСКОГО ГРАФА МИРА (CORTICAL RELATIONAL TRIPLET GRAPH)
# ============================================================================
@dataclass
class CausalTripletEdge:
    subject: str
    relation: str
    target: str
    conductance: float
    subject_vec64: torch.Tensor
    target_vec64: torch.Tensor
    edge_pole_vec64: torch.Tensor = field(default_factory=lambda: torch.zeros(64))
    full_clause_tids: List[int] = field(default_factory=list)
    rel_target_tids: List[int] = field(default_factory=list)
    inf_rel_target_tids: List[int] = field(default_factory=list)
    source: str = "legacy"
    observed_at: str = ""
    polarity: int = 1
    causal_kind: str = "association"
    conditions: Tuple[str, ...] = ()
    subject_type: str = "entity"
    target_type: str = "entity"
    status: str = "quarantined"
    edge_id: str = ""
    conflict_ids: Tuple[str, ...] = ()


class CorticalRelationalWorldGraph:
    """Provenance-bearing assertions and an open-world Boolean constraint model.

    Conductance is a training weight, never evidence of truth. Asserted and explicitly
    verified records are usable unless quarantined or contradicted in the active
    context. Polarity negates the *relation*, not the subject or target: denying a
    sufficient cause does not establish absence of its effect. Positive sufficient
    edges propagate presence (inhibitory verbs propagate absence); positive necessary
    edges propagate requirement failure only. This is not a learned structural causal
    model and does not identify observational counterfactuals.
    """

    _BASE_VERBS = (
        "release", "activate", "ignite", "stabilize", "regulate", "modulate",
        "protect", "generate", "induce", "trigger", "drive", "receive", "inhibit",
        "suppress", "phosphorylate", "absorb", "split", "direct", "fuel", "stimulate",
        "produce", "cause", "require", "prevent", "enable", "power", "support",
    )
    _REL_ALIASES = {v: v for v in _BASE_VERBS}
    _REL_ALIASES.update({v + ("es" if v.endswith("ss") else "s"): v for v in _BASE_VERBS})
    _REL_ALIASES.update({"binds to": "bind to", "bind to": "bind to", "suppresses": "suppress",
                         "is sufficient for": "is sufficient for",
                         "is necessary for": "is necessary for",
                         "is sufficient to inhibit": "inhibit",
                         "is sufficient to suppress": "suppress",
                         "is sufficient to prevent": "prevent"})
    RELATIONAL_VERBS = tuple(_REL_ALIASES)
    _STOP = set("the a an to of in on is are was were and or but for with that this it do does did can could would should will may might must you your i he she we they what how why where when who which be been have has had not no by if as at into from such then also".split())
    _INHIBITORY = {"inhibit", "suppress", "prevent"}
    _METADATA = ("source", "observed_at", "polarity", "causal_kind", "conditions",
                 "subject_type", "target_type", "status", "edge_id", "conflict_ids")
    _DERIVED_STATE = ("subject_vec64", "target_vec64", "edge_pole_vec64",
                      "full_clause_tids", "rel_target_tids", "inf_rel_target_tids")

    def __init__(self, encode_fn: Callable[[str], List[int]],
                 compute_vec64_fn: Callable[[List[int]], torch.Tensor],
                 get_vec64_fn: Optional[Callable[[int], torch.Tensor]] = None,
                 decode_fn: Optional[Callable[[int], str]] = None):
        self.encode_fn = encode_fn
        self.compute_vec64_fn = compute_vec64_fn
        self.get_vec64_fn = get_vec64_fn
        self.decode_fn = decode_fn
        self.edges: List[CausalTripletEdge] = []
        self.adj_by_subject: Dict[str, List[CausalTripletEdge]] = {}
        self.quarantined_clauses: List[Dict[str, str]] = []
        # Retained for explicit training/checkpoint compatibility, not used as proof.
        self.edge_valence_generator = SyntacticValenceWaveGenerator(
            d_vec=64, d_dend=8192, top_k_wta=32, seed=1809)
        self.bridge_rel_clause_tids = self.encode_fn(", which")
        self.bridge_inf_purpose_tids = self.encode_fn(" to")
        self.bridge_thereby_tids = self.encode_fn(", thereby")

    @staticmethod
    def _normalize_entity(ent: str) -> str:
        clean = " ".join(str(ent).strip(" .,!?;:'()[]{}\"").lower().split())
        return re.sub(r"^(?:the|an|a)(?:\s+|$)", "", clean).strip()

    @classmethod
    def _valid_entity(cls, ent: str) -> bool:
        words = ent.split()
        forbidden = cls._STOP - {"of", "in", "on", "with", "for", "at"}
        verb_words = {r.split()[0] for r in cls.RELATIONAL_VERBS}
        return bool(words and len(words) <= 12
                    and re.fullmatch(r"[a-z0-9][a-z0-9_-]*(?: [a-z0-9][a-z0-9_-]*)*", ent)
                    and words[0] not in cls._STOP and words[-1] not in cls._STOP
                    and not (set(words) & forbidden)
                    and (len(words) == 1 or not (set(words) & verb_words)))

    @classmethod
    def _normalize_relation(cls, relation: str) -> Tuple[str, bool]:
        rel = " ".join(relation.lower().split())
        negative = bool(re.match(r"^(?:does not|do not|not)\s+", rel))
        rel = re.sub(r"^(?:does not|do not|not)\s+", "", rel)
        negative = negative or rel.startswith("is not ")
        rel = re.sub(r"^is not ", "is ", rel)
        return cls._REL_ALIASES.get(rel, rel), negative

    @classmethod
    def _normalize_conditions(cls, conditions: Sequence[str]) -> Tuple[str, ...]:
        if isinstance(conditions, str):
            conditions = (conditions,)
        literals = []
        for condition in conditions:
            literal = " ".join(str(condition).lower().strip().split())
            negative = literal.startswith("not ")
            entity = cls._normalize_entity(literal[4:] if negative else literal)
            literals.append(("not " if negative else "") + entity)
        normalized = tuple(sorted(set(literals)))
        for cond in normalized:
            entity = cond[4:] if cond.startswith("not ") else cond
            if not cls._valid_entity(cls._normalize_entity(entity)):
                raise ValueError("Conditions must be named literals, optionally prefixed by 'not'.")
        if any("not " + c in normalized for c in normalized):
            raise ValueError("Contradictory conditions.")
        return normalized

    @classmethod
    def _parse_clause(cls, text: str) -> Optional[Dict[str, object]]:
        """Accept one entity-relation-entity clause, optionally if/when named literals.

        No pronoun resolution, shared-subject coordination, modal or relative clauses.
        Ordinary verbs remain associations; only explicit cause/sufficient/necessary
        language declares executable causal semantics.
        """
        if not re.fullmatch(r"[a-zA-Z0-9_\-\s]+\.?", text.strip()):
            return None
        clean = " ".join(text.lower().strip().rstrip(".").split())
        parts = re.split(r"\s+(?:if|when|provided that)\s+", clean)
        if len(parts) > 2:
            return None
        conditions: Tuple[str, ...] = ()
        if len(parts) == 2:
            try:
                conditions = cls._normalize_conditions(re.split(r"\s+and\s+", parts[1]))
            except ValueError:
                return None
        negative_relations = {r.replace("is ", "is not ", 1) for r in cls.RELATIONAL_VERBS if r.startswith("is ")}
        rels = sorted(set(cls.RELATIONAL_VERBS) | negative_relations, key=len, reverse=True)
        pattern = r"(?<![\w-])(?:(?:does not|do not|not)\s+)?(?:" + "|".join(map(re.escape, rels)) + r")(?![\w-])"
        candidates = []
        for match in re.finditer(pattern, parts[0]):
            subject = cls._normalize_entity(parts[0][:match.start()])
            target = cls._normalize_entity(parts[0][match.end():])
            if cls._valid_entity(subject) and cls._valid_entity(target):
                candidates.append((match, subject, target))
        if len(candidates) != 1:
            return None
        m, subject, target = candidates[0]
        relation, negative = cls._normalize_relation(m.group())
        kind = "association"
        declaration = re.sub(r"^is not ", "is ", m.group())
        if relation in {"cause", "is sufficient for"} or declaration.startswith("is sufficient to "):
            kind = "sufficient"
        elif relation == "is necessary for":
            kind = "necessary"
        elif relation == "require":
            subject, target = target, subject
            relation, kind = "is necessary for", "necessary"
        return dict(subject=subject, relation=relation, target=target,
                    polarity=-1 if negative else 1, causal_kind=kind, conditions=conditions)

    def extract_and_consolidate_sentence(self, sentence: str, dopamine_gain: float = 1.5,
                                         source: str = "user", status: str = "asserted",
                                         conditions: Sequence[str] = ()) -> List[Tuple[str, str, str]]:
        clause = self._parse_clause(sentence)
        if clause is None:
            self.quarantined_clauses.append(dict(sentence=sentence, source=source,
                                                status="quarantined", reason="Unsupported or incomplete controlled clause"))
            return []
        try:
            clause["conditions"] = self._normalize_conditions(tuple(clause["conditions"]) + self._normalize_conditions(conditions))
        except ValueError:
            self.quarantined_clauses.append(dict(sentence=sentence, source=source,
                                                status="quarantined", reason="Invalid conditions"))
            return []
        edge = self.add_or_strengthen_edge(**clause, conductance_delta=dopamine_gain,
                                          source=source, status=status)
        return [(edge.subject, edge.relation, edge.target)]

    def add_or_strengthen_edge(self, subject: str, relation: str, target: str,
                              conductance_delta: float = 1.5, *, source: str = "user",
                              observed_at: Optional[str] = None, polarity: int = 1,
                              causal_kind: Optional[str] = None, conditions: Sequence[str] = (),
                              subject_type: str = "entity", target_type: str = "entity",
                              status: str = "asserted", edge_id: Optional[str] = None,
                              conflict_ids: Sequence[str] = ()) -> CausalTripletEdge:
        s, t = self._normalize_entity(subject), self._normalize_entity(target)
        r, negative = self._normalize_relation(relation)
        if causal_kind is None:
            declaration = re.sub(r"^is not ", "is ", " ".join(relation.lower().split()))
            if r in {"cause", "is sufficient for"} or declaration.startswith("is sufficient to "):
                causal_kind = "sufficient"
            elif r in {"require", "is necessary for"}:
                causal_kind = "necessary"
            else:
                causal_kind = "association"
        if r == "require":
            s, t, r = t, s, "is necessary for"
            subject_type, target_type = target_type, subject_type
        if polarity not in (-1, 1) or isinstance(polarity, bool):
            raise ValueError("Polarity must be +1 or -1.")
        polarity = -1 if negative else polarity
        if causal_kind not in {"association", "sufficient", "necessary"}:
            raise ValueError("Unknown causal kind.")
        if status not in {"asserted", "verified", "quarantined"}:
            raise ValueError("Unknown assertion status.")
        conds = self._normalize_conditions(conditions)
        source = str(source)
        if not self._valid_entity(s) or not self._valid_entity(t) or not r or not source.strip():
            status = "quarantined"
        identity = (s, r, t, polarity, causal_kind, conds, subject_type, target_type, source)
        stable_id = "edge_" + hashlib.sha256(json.dumps(identity, ensure_ascii=True).encode()).hexdigest()[:24]
        eid = str(edge_id or stable_id)
        for ex in self.edges:
            ex_identity = (ex.subject, ex.relation, ex.target, ex.polarity, ex.causal_kind,
                           ex.conditions, ex.subject_type, ex.target_type, ex.source)
            if ex.edge_id == eid and ex_identity != identity:
                raise ValueError("Edge ID collision with a different assertion.")
            if ex_identity == identity:
                ex.conductance += float(conductance_delta)
                # Only an explicit verified call upgrades trust; repetition cannot.
                if status == "verified" and ex.status == "asserted":
                    ex.status = status
                return ex
        sv = self.compute_vec64_fn(self.encode_fn(s))
        rv = self.compute_vec64_fn(self.encode_fn(r))
        tv = self.compute_vec64_fn(self.encode_fn(t))
        pole = F.normalize(sv + .9 * torch.roll(rv, 1, 0) + .8 * torch.roll(tv, 2, 0) + .7 * sv * tv, dim=0)
        edge = CausalTripletEdge(s, r, t, float(conductance_delta), sv, tv, pole,
                                source=source, observed_at=observed_at if observed_at is not None else datetime.now(timezone.utc).isoformat(),
                                polarity=polarity, causal_kind=causal_kind, conditions=conds,
                                subject_type=str(subject_type), target_type=str(target_type),
                                status=status, edge_id=eid, conflict_ids=tuple(sorted(set(conflict_ids))))
        edge.full_clause_tids = self.encode_fn(self._render_edge(edge))
        edge.rel_target_tids = list(edge.full_clause_tids)
        edge.inf_rel_target_tids = list(edge.full_clause_tids)
        for ex in self.edges:
            if (ex.subject, ex.relation, ex.target, ex.causal_kind) == (s, r, t, causal_kind) and ex.polarity != polarity:
                if not any("not " + c in ex.conditions or (c.startswith("not ") and c[4:] in ex.conditions) for c in conds):
                    edge.conflict_ids = tuple(sorted(set(edge.conflict_ids) | {ex.edge_id}))
                    ex.conflict_ids = tuple(sorted(set(ex.conflict_ids) | {eid}))
        self.edges.append(edge)
        if self._valid_entity(s) and self._valid_entity(t):
            self.adj_by_subject.setdefault(s, []).append(edge)
        return edge

    @staticmethod
    def _render_edge(edge: CausalTripletEdge) -> str:
        relation = edge.relation
        # Render the declared kind as well as the predicate, so metadata is not lost.
        if edge.causal_kind == "necessary":
            relation = "is necessary for"
        elif edge.causal_kind == "sufficient":
            relation = "is sufficient to " + relation if relation in CorticalRelationalWorldGraph._INHIBITORY else "is sufficient for"
        if edge.polarity < 0:
            relation = relation.replace("is ", "is not ", 1) if relation.startswith("is ") else "does not " + relation
        suffix = " if " + " and ".join(edge.conditions) if edge.conditions else ""
        return f"{edge.subject.capitalize()} {relation} {edge.target}{suffix}."

    def _eligible_edges(self, conditions: Sequence[str] = ()) -> List[CausalTripletEdge]:
        known = set(self._normalize_conditions(conditions))
        active = [e for e in self.edges if e.status in {"asserted", "verified"}
                  and set(e.conditions) <= known and self._valid_entity(e.subject) and self._valid_entity(e.target)]
        ids = {e.edge_id for e in active}
        return [e for e in active if not (set(e.conflict_ids) & ids)]

    def _mentions(self, text: str) -> List[str]:
        text = " ".join(text.lower().split())
        entities = {e.subject for e in self.edges} | {e.target for e in self.edges}
        hits = []
        for entity in entities:
            if self._valid_entity(entity):
                for m in re.finditer(r"(?<![\w-])" + re.escape(entity) + r"(?![\w-])", text):
                    hits.append((m.start(), m.end(), entity))
        # Prefer a full longer entity over its nested word-boundary mention.
        hits = [h for h in hits if not any(a <= h[0] and b >= h[1] and b-a > h[1]-h[0] for a, b, _ in hits)]
        return list(dict.fromkeys(h[2] for h in sorted(hits)))

    def _path(self, start: str, goal: Optional[str], max_hops: int,
              conditions: Sequence[str], kind: str) -> List[CausalTripletEdge]:
        eligible = self._eligible_edges(conditions)
        queue: List[Tuple[str, List[CausalTripletEdge], set[str]]] = [(start, [], {start})]
        while queue:
            node, path, visited = queue.pop(0)
            if path and (goal is None or node == goal):
                return path
            if len(path) >= max_hops:
                continue
            for edge in sorted(eligible, key=lambda e: e.edge_id):
                if (edge.subject == node and edge.target not in visited and edge.polarity == 1
                        and edge.causal_kind == kind and edge.relation not in self._INHIBITORY):
                    queue.append((edge.target, path + [edge], visited | {edge.target}))
        return []

    def find_causal_path(self, query: str, max_hops: int = 4, *,
                         conditions: Sequence[str] = (), causal_kind: str = "sufficient") -> List[CausalTripletEdge]:
        """Exact-identity paths: sufficient presence or necessary failure, never mixed.

        With a named destination, return only a path reaching it, not an unrelated
        high-conductance branch. Associations cannot supply transitive causal hops.
        """
        if causal_kind not in {"sufficient", "necessary"}:
            return []
        mentions = self._mentions(query)
        if not mentions:
            return []
        return self._path(mentions[0], mentions[-1] if len(mentions) > 1 else None,
                          max_hops, conditions, causal_kind)

    def collect_edge_trajectories_for_valence_imprint(self) -> List[Tuple[torch.Tensor, Sequence[int], float]]:
        """Explicit training data only; exclude quarantined and disputed assertions."""
        return [(e.edge_pole_vec64, e.full_clause_tids, e.conductance)
                for e in self.edges if e.status in {"asserted", "verified"} and not e.conflict_ids and e.full_clause_tids]

    def imprint_edges_into_dedicated_valence_matrix(self) -> None:
        if self.get_vec64_fn is not None:
            trajectories = self.collect_edge_trajectories_for_valence_imprint()
            if trajectories:
                self.edge_valence_generator.w_valence_dend_out.zero_()
                self.edge_valence_generator.imprint_batch_zero_pos(trajectories, self.get_vec64_fn, epochs=16)

    def ground_response(self, query: str, response: str) -> Dict[str, object]:
        """Entail every controlled material clause, independently of domain overlap.

        Closure is the smaller of entailed-clause fraction and exact query-entity
        coverage. It is a diagnostic, not a scientific validation metric. Unparsed
        clauses fail closed. Conditions on evidence must appear in the claim;
        merely mentioning a context in the question never licenses dropping it.
        """
        clauses: List[str] = []
        for sentence in re.split(r"[.;!]", response):
            if not sentence.strip():
                continue
            if self._parse_clause(sentence) is not None:
                clauses.append(sentence.strip())
            else:
                clauses.extend(c.strip() for c in re.split(r"\s+(?:and|but)\s+", sentence, flags=re.I) if c.strip())
        reasons: List[str] = []
        proofs: List[Dict[str, object]] = []
        proof_ids: set[str] = set()
        covered: set[str] = set()
        disputed: set[str] = set()
        supported = 0
        for text in clauses:
            claim = self._parse_clause(text)
            evidence: List[CausalTripletEdge] = []
            if claim is not None:
                active = [e for e in self.edges if e.status in {"asserted", "verified"}
                          and set(e.conditions) <= set(claim["conditions"])]
                active_ids = {e.edge_id for e in active}
                for edge in active:
                    if (edge.subject, edge.relation, edge.target, edge.causal_kind) == (
                            claim["subject"], claim["relation"], claim["target"], claim["causal_kind"]):
                        disputed.update(set(edge.conflict_ids) & active_ids)
                        if set(edge.conflict_ids) & active_ids:
                            disputed.add(edge.edge_id)
                eligible = self._eligible_edges(claim["conditions"])
                for edge in eligible:
                    if (edge.subject, edge.relation, edge.target, edge.polarity, edge.causal_kind) == (
                            claim["subject"], claim["relation"], claim["target"], claim["polarity"], claim["causal_kind"]):
                        evidence = [edge]
                        break
                    if (edge.subject, edge.target, edge.polarity, edge.causal_kind) == (
                            claim["subject"], claim["target"], claim["polarity"], claim["causal_kind"]) and claim["relation"] in {"cause", "is sufficient for", "is necessary for"} and edge.relation not in self._INHIBITORY:
                        evidence = [edge]
                        break
                if not evidence and claim["polarity"] == 1 and claim["relation"] in {"cause", "is sufficient for", "is necessary for"}:
                    evidence = self._path(str(claim["subject"]), str(claim["target"]),
                                          max(1, len(self.edges)), claim["conditions"], str(claim["causal_kind"]))
            if evidence:
                supported += 1
                proof_ids.update(e.edge_id for e in evidence)
                covered.update({str(claim["subject"]), str(claim["target"])})
            else:
                reasons.append("Unsupported, disputed, or unparsed assertion: " + text)
            proofs.append(dict(assertion=text, entailed=bool(evidence), proof_edge_ids=[e.edge_id for e in evidence]))
        if not clauses:
            reasons.append("No supported material assertions.")
        queried = set(self._mentions(query))
        query_core = re.sub(r"^(?:(?:explain|describe)(?: how| why)?|(?:how|why)(?: does| do)?|does|do)\s+", "",
                            query.lower().strip().rstrip("?."))
        query_claim = self._parse_clause(query_core)
        if query_claim is not None:
            queried.update((str(query_claim["subject"]), str(query_claim["target"])))
        missing = queried - covered
        if missing:
            reasons.append("Unanswered query entities: " + ", ".join(sorted(missing)))
        if not queried:
            reasons.append("No exact query entities recognized; assertion entailment does not establish query closure.")
        fraction = supported / len(clauses) if clauses else 0.0
        coverage = len(queried & covered) / len(queried) if queried else 0.0
        used = [e for e in self.edges if e.edge_id in proof_ids]
        return dict(grounded=bool(clauses) and supported == len(clauses),
                    semantic_closure_score=min(fraction, coverage),
                    sources=sorted({e.source for e in used}), reasons=reasons,
                    proof_edge_ids=sorted(proof_ids), disputed_edge_ids=sorted(disputed), assertion_proofs=proofs)

    def synthesize_novel_sentence_from_graph(self, query: str,
            valence_generator: Optional[SyntacticValenceWaveGenerator] = None,
            get_vec64_fn: Optional[Callable[[int], torch.Tensor]] = None,
            decode_fn: Optional[Callable[[int], str]] = None, *,
            conditions: Sequence[str] = ()) -> Optional[Dict[str, object]]:
        """Transparent constrained rendering of known edges, not neural generation."""
        path = self.find_causal_path(query, conditions=conditions)
        if not path:
            mentions = self._mentions(query)
            path = [e for e in self._eligible_edges(conditions) if mentions and e.subject == mentions[0]
                    and (len(mentions) == 1 or e.target == mentions[-1])][:1]
        if not path:
            return None
        text = " ".join(self._render_edge(e) for e in path)
        grounding = self.ground_response(query, text)
        if not grounding["grounded"]:
            return None
        return dict(synthesized_sentence=text, synthesized_token_ids=self.encode_fn(text),
                    synthesized_via_zero_pos_nmda_matrix=False, rendering_mode="constrained_edge_surface",
                    hops_traversed=len(path), triplet_chain=[(e.subject, e.relation, e.target) for e in path],
                    **grounding)

    def intervene(self, interventions: Dict[str, bool], observed: Optional[Dict[str, bool]] = None,
                  conditions: Sequence[str] = ()) -> Dict[str, object]:
        """Propagate true/false/None (unknown), cutting incoming rules for do-nodes.

        Absence of a sufficient cause leaves its effect unknown; requirement failure
        establishes absence. Alternative sufficient routes remain available. Conflicting
        constraints yield unknown, not arbitrary conductance-based winners. Observations
        are scenario inputs, not abduction or factual counterfactual identification.
        """
        context = self._normalize_conditions(conditions)
        def normalize_values(values: Dict[str, bool]) -> Dict[str, bool]:
            result = {}
            for node, value in values.items():
                key = self._normalize_entity(node)
                if not self._valid_entity(key) or not isinstance(value, bool):
                    raise ValueError("Interventions/observations require named entities and Boolean values.")
                if key in result and result[key] != value:
                    raise ValueError("Conflicting values for the same normalized entity.")
                result[key] = value
            return result
        do = normalize_values(interventions)
        base = normalize_values(observed or {})
        base.update(do)
        eligible = self._eligible_edges(context)
        rules = [e for e in eligible if e.polarity == 1 and e.causal_kind != "association"]
        nodes = sorted({e.subject for e in self.edges if self._valid_entity(e.subject)} |
                       {e.target for e in self.edges if self._valid_entity(e.target)} | set(base))
        values = {n: base.get(n) for n in nodes}
        proofs: Dict[str, set[str]] = {n: set() for n in nodes}
        conflicts: Dict[str, List[str]] = {}
        converged = False
        for _ in range(max(4, len(nodes) * 4)):
            proposals: Dict[str, Dict[bool, set[str]]] = {n: {} for n in nodes}
            for n, value in base.items():
                proposals[n][value] = set()
            for edge in rules:
                if edge.target in do:
                    continue
                value = values[edge.subject]
                outcome = None
                if edge.causal_kind == "sufficient" and value is True:
                    outcome = edge.relation not in self._INHIBITORY
                elif edge.causal_kind == "necessary" and value is False:
                    outcome = False
                if outcome is not None:
                    proposals[edge.target].setdefault(outcome, set()).update(proofs[edge.subject] | {edge.edge_id})
            new_values = {n: next(iter(p)) if len(p) == 1 else None for n, p in proposals.items()}
            new_proofs = {n: set().union(*p.values()) if p else set() for n, p in proposals.items()}
            conflicts = {n: sorted(new_proofs[n]) for n, p in proposals.items() if len(p) > 1}
            if new_values == values and new_proofs == proofs:
                converged = True
                values, proofs = new_values, new_proofs
                break
            values, proofs = new_values, new_proofs
        if not converged:
            # Cyclic inconsistent constraints are not allowed to produce a timed winner.
            values = {n: base.get(n) if n in do else None for n in nodes}
            proofs = {n: set() for n in nodes}
        used_ids = set().union(*proofs.values()) if proofs else set()
        reasons = ["Open-world propagation: missing sufficient causes do not imply missing effects."]
        if conflicts:
            reasons.append("Conflicting scenario constraints: " + ", ".join(sorted(conflicts)))
        if not converged:
            reasons.append("Constraint propagation did not converge; derived values withheld.")
        eligible_ids = {e.edge_id for e in eligible}
        rejected = [e.edge_id for e in self.edges if e.edge_id not in eligible_ids]
        return dict(values=values, states={n: "unknown" if v is None else "true" if v else "false" for n, v in values.items()},
                    interventions=do, conditions=context, proof_edge_ids=sorted(used_ids),
                    proofs={n: sorted(p) for n, p in proofs.items()}, conflicts=conflicts,
                    sources=sorted({e.source for e in rules if e.edge_id in used_ids}),
                    rejected_edge_ids=rejected, reasons=reasons, converged=converged)

    def export_state_dict(self) -> Dict[str, object]:
        records = []
        for edge in self.edges:
            record = {k: getattr(edge, k) for k in ("subject", "relation", "target", "conductance") + self._METADATA}
            record.update({k: getattr(edge, k).detach().clone() for k in self._DERIVED_STATE[:3]})
            record.update({k: list(getattr(edge, k)) for k in self._DERIVED_STATE[3:]})
            records.append(record)
        return dict(version=19, edges=records, quarantined_clauses=list(self.quarantined_clauses),
                    edge_valence_generator=self.edge_valence_generator.export_state_dict())

    def load_state_dict(self, state: object) -> None:
        self.edges.clear()
        self.adj_by_subject.clear()
        self.quarantined_clauses = list(state.get("quarantined_clauses", [])) if isinstance(state, dict) else []
        records = state.get("edges", []) if isinstance(state, dict) else state
        for item in records:
            legacy = not all(k in item for k in self._METADATA)
            if not legacy and all(k in item for k in self._DERIVED_STATE):
                fields = {k: item[k] for k in ("subject", "relation", "target", "conductance") + self._METADATA}
                for key in self._DERIVED_STATE[:3]:
                    vector = item[key]
                    if not isinstance(vector, torch.Tensor) or vector.shape != (64,) or not torch.isfinite(vector).all():
                        raise ValueError("Invalid graph vector state.")
                    fields[key] = vector.detach().clone()
                for key in self._DERIVED_STATE[3:]:
                    if not isinstance(item[key], list) or any(not isinstance(t, int) or isinstance(t, bool) or t < 0 for t in item[key]):
                        raise ValueError("Invalid graph token state.")
                    fields[key] = list(item[key])
                if (fields["polarity"] not in (-1, 1) or isinstance(fields["polarity"], bool)
                        or fields["causal_kind"] not in {"association", "sufficient", "necessary"}
                        or fields["status"] not in {"asserted", "verified", "quarantined"}):
                    raise ValueError("Invalid graph assertion state.")
                edge = CausalTripletEdge(**fields)
                self.edges.append(edge)
                if self._valid_entity(edge.subject) and self._valid_entity(edge.target):
                    self.adj_by_subject.setdefault(edge.subject, []).append(edge)
                continue
            metadata = {k: item[k] for k in self._METADATA if k in item}
            if legacy:
                metadata.update(status="quarantined", source=item.get("source", "legacy"),
                                observed_at=item.get("observed_at", ""))
                metadata.setdefault("causal_kind", "association")
            self.add_or_strengthen_edge(str(item["subject"]), str(item["relation"]), str(item["target"]),
                                        float(item.get("conductance", 0.0)), **metadata)
        if isinstance(state, dict) and "edge_valence_generator" in state:
            self.edge_valence_generator.load_state_dict(state["edge_valence_generator"])


# ============================================================================
# 3. САМООРГАНИЗУЮЩАЯСЯ ОНТОЛОГИЯ РОЛЕЙ ПО ЗАКОНУ НЕВЯЗКИ СХЕМЫ (0% ПРЕДМЕТНЫХ СТОП-СЛОВ!)
# ============================================================================
class EmergentOntologicalRoleDiscovery:
    r"""Самозарождение онтологических классов ролей в слое `L2/3 BA22` через **Асимметричную STDP** и
    строгий математический **Закон Невязки Схемы (`Schema-Residual Binding`)**:
      $$\mathcal{R}_{\mathrm{filler}} = \mathcal{W}_{\mathrm{exp}}(U) \setminus \mathcal{V}_{\mathrm{active}}, \qquad
        \mathcal{R}_{\mathrm{slot}} = \mathcal{V}_{\mathrm{shared}}(S_m) \setminus \mathcal{W}(U)$$
      - В `FUNCTION_WORDS_ONLY` входят **только служебные грамматические слова английского языка**
        (предлоги, местоимения, артикли, союзы, вспомогательные глаголы — **0% предметных существительных/прилагательных!**).
      - В качестве канонических ролевых слотов схемы $\mathcal{V}_{\mathrm{shared}}(S_m)$ автоматически выделяются
        **сквозные тематические ядра схемы** $\mathcal{W}(\text{Stimulus}_m) \cap \mathcal{W}(\text{Response}_m)$
        (именно те сущности, которые присутствуют и в вопросе схемы, и в её ответе — например, `mayonnaise`, `essay`, `running`!),
        но отсутствуют в текущем вопросе пользователя $U$!"""

    FUNCTION_WORDS_ONLY = {
        "the", "a", "an", "to", "of", "in", "on", "is", "are", "was", "were",
        "and", "or", "for", "with", "that", "this", "it", "do", "does", "did",
        "can", "could", "would", "should", "will", "may", "might", "must",
        "you", "your", "yours", "my", "mine", "his", "her", "hers", "our", "their",
        "i", "me", "he", "him", "she", "we", "us", "they", "them",
        "what", "how", "why", "where", "when", "who", "whom", "which",
        "be", "been", "being", "have", "has", "had", "not", "no", "yes",
        "by", "while", "before", "after", "into", "until", "from", "each", "overly",
        "if", "as", "at", "up", "out", "so", "than", "too", "very", "just",
        "about", "please", "remind", "tell", "explain",
    }

    def __init__(self, n_bins: int = 2048):
        self.n_bins = int(n_bins)
        self.word_context_sdr: Dict[str, torch.Tensor] = {}
        self.canonical_anchors: set[str] = set()
        # Каждая схема хранит: (полный словарь схемы V_all, сквозные ролевые слоты V_shared = W(Stim) cap W(Resp))
        self.schema_records: List[Tuple[set[str], set[str]]] = []
        # Схемы приходят из обучающего корпуса, поэтому их число растёт. Поиск
        # в схемах линеен по их числу, а чекпоинт не должен расти без предела:
        # держим последние N схем (самые свежие — они и релевантнее).
        self.max_schema_records = 512

    def _ctx_token_bits(self, ctx_word: str, relative_offset: int) -> Tuple[int, int, int]:
        h = 2166136261
        for ch in f"{relative_offset}:{ctx_word}":
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
        b0 = h % self.n_bins
        b1 = (h * 1597334677 + 41) % self.n_bins
        b2 = (h * 381271489 + 109) % self.n_bins
        return int(b0), int(b1), int(b2)

    def register_schema_pair(self, stim_text: str, resp_text: str, lr: float = 1.5) -> None:
        """Регистрирует каузальную схему `(stim_text, resp_text)`:
        вычисляет сквозные ролевые ядра $\\mathcal{V}_{\\mathrm{shared}} = \\mathcal{W}(\\text{Stim}) \\cap \\mathcal{W}(\\text{Resp})$
        (слова, которые переходят из вопроса в ответ через Дугообразный Пучок!)."""
        self.observe_sentence_stdp(stim_text, register_as_canonical=True, lr=lr)
        self.observe_sentence_stdp(resp_text, register_as_canonical=True, lr=lr)
        p_strip = " .,!?;:'()[]{}" + chr(34)
        stim_w = {
            w.strip(p_strip).lower()
            for w in stim_text.split()
            if len(w.strip(p_strip)) >= 4 and w.strip(p_strip).lower() not in self.FUNCTION_WORDS_ONLY
        }
        resp_w = {
            w.strip(p_strip).lower()
            for w in resp_text.split()
            if len(w.strip(p_strip)) >= 4 and w.strip(p_strip).lower() not in self.FUNCTION_WORDS_ONLY
        }
        v_all = stim_w | resp_w
        shared = stim_w & resp_w
        if shared and (v_all, shared) not in self.schema_records:
            self.schema_records.append((v_all, shared))
            if len(self.schema_records) > int(self.max_schema_records):
                # Оставляем самые свежие схемы (FIFO).
                del self.schema_records[: len(self.schema_records) - int(self.max_schema_records)]

    def register_training_pair(self, stim_text: str, resp_text: str, lr: float = 1.5
                               ) -> Dict[str, Tuple[str, float]]:
        """Зарегистрировать ОБУЧАЮЩУЮ пару и открыть ролевые связи по ней.

        Вызывается на каждой принятой обучающей записи: онтология учится
        схеме (словарь + сквозные ролевые слоты) и STDP-полям слов, а
        Закон Невязки Схемы открывает связи «канонический слот → новый
        заполнитель» (например, `mayonnaise → hollandaise`).

        Связи открываются по КОРПУСУ, а не по текущему промпту: иначе
        подстановка была бы круговой (модель «знает» только то, что сказали
        сейчас, и ничего не переносит).
        """
        self.register_schema_pair(stim_text, resp_text, lr=lr)
        return self.discover_emergent_role_bindings(stim_text)

    def observe_sentence_stdp(self, sentence: str, register_as_canonical: bool = False, lr: float = 1.0) -> None:
        """Наблюдает предложение и обновляет направленные (асимметричные по времени $\\Delta t \\in \\{-2,-1,+1,+2\\}$)
        контекстные рецептивные поля слов по правилу STDP (Bi & Poo 1998)."""
        p_strip = " .,!?;:'()[]{}" + chr(34)
        words = [w.strip(p_strip).lower() for w in sentence.split() if w.strip(p_strip)]
        light_stop = {"the", "a", "an", "to", "of", "in", "on", "is", "are", "and", "or", "my", "your", "his", "her", "when"}
        filtered = [w for w in words if len(w) >= 3 and w not in light_stop]

        for idx_pos, target_w in enumerate(filtered):
            if target_w in self.FUNCTION_WORDS_ONLY:
                continue
            if register_as_canonical and len(target_w) >= 4:
                self.canonical_anchors.add(target_w)
            elif not register_as_canonical and target_w in self.canonical_anchors:
                # Защищаем консолидированные рецептивные поля канонических слотов от размывания фоновым веб-шумом
                continue
            vec = self.word_context_sdr.get(target_w)
            if vec is None:
                vec = torch.zeros(self.n_bins, dtype=torch.float32)
                self.word_context_sdr[target_w] = vec

            for rel_off in (-2, -1, 1, 2):
                j_pos = idx_pos + rel_off
                if 0 <= j_pos < len(filtered):
                    ctx_w = filtered[j_pos]
                    stem = ctx_w[:5] if len(ctx_w) >= 6 else ctx_w
                    sign_dir = -1 if rel_off < 0 else 1
                    for b in self._ctx_token_bits(stem, rel_off):
                        vec[b] += float(lr) * 1.2
                    for b in self._ctx_token_bits(stem, sign_dir):
                        vec[b] += float(lr) * 0.8

    def discover_emergent_role_bindings(
        self,
        utterance: str,
        similarity_threshold: float = 0.34,
    ) -> Dict[str, Tuple[str, float]]:
        r"""По **Закону Невязки Схемы**:
          1) Находит кандидатов в заполнители $\mathcal{R}_{\mathrm{filler}} = \mathcal{W}_{\mathrm{exp}}(U) \setminus \mathcal{V}_{\mathrm{all}}$
             (слова вопроса, которые уже наблюдались в опыте мира, но не входят в базовый словарь схем);
          2) Находит незаполненные сквозные слоты активных схем $\mathcal{R}_{\mathrm{slot}} = \mathcal{V}_{\mathrm{shared}}(S_{\mathrm{act}}) \setminus \mathcal{W}(U)$;
          3) Связывает их по косинусному резонансу асимметричных STDP-полей (`0%` предметных стоп-слов!)."""
        p_strip = " .,!?;:'()[]{}" + chr(34)
        utt_words = [
            w.strip(p_strip).lower()
            for w in utterance.split()
            if len(w.strip(p_strip)) >= 4 and w.strip(p_strip).lower() not in self.FUNCTION_WORDS_ONLY
        ]
        utt_word_set = set(utt_words)
        utt_stems = {w[:5] for w in utt_word_set if len(w) >= 5}

        active_shared_slots: set[str] = set()
        active_vocabulary: set[str] = set()
        for v_all, v_shared in self.schema_records:
            overlap = sum(
                1 for uw in utt_word_set
                if uw in v_all or (len(uw) >= 5 and any(sw.startswith(uw[:5]) for sw in v_all))
            )
            if overlap >= 2:
                active_shared_slots |= v_shared
                active_vocabulary |= v_all

        residual_fillers = [
            w for w in utt_words
            if w in self.word_context_sdr
            and w not in active_vocabulary
            and not (len(w) >= 5 and any(cw.startswith(w[:5]) for cw in active_vocabulary))
        ]
        candidate_slots = sorted(
            s for s in active_shared_slots
            if s not in utt_word_set and not (len(s) >= 5 and s[:5] in utt_stems)
        )

        discovered: Dict[str, Tuple[str, float]] = {}
        for w_cand in residual_fillers:
            v_cand = self.word_context_sdr.get(w_cand)
            if v_cand is None or float(v_cand.norm().item()) < 1e-5:
                continue
            v_cand_norm = F.normalize(v_cand, p=2, dim=0)

            best_canon = None
            best_sim = -1.0
            for w_canon in candidate_slots:
                v_can = self.word_context_sdr.get(w_canon)
                if v_can is None or float(v_can.norm().item()) < 1e-5:
                    continue
                v_can_norm = F.normalize(v_can, p=2, dim=0)
                sim = float(torch.dot(v_cand_norm, v_can_norm).item())
                if sim > best_sim:
                    best_sim = sim
                    best_canon = w_canon

            if best_canon is not None and best_sim >= similarity_threshold:
                prev = discovered.get(best_canon)
                if prev is None or best_sim > prev[1]:
                    discovered[best_canon] = (w_cand, round(best_sim, 4))

        return discovered

    def think_and_resolve_known_concept(
        self,
        utterance: str,
        min_stdp_Factory_cosine: float = 0.45,
    ) -> Optional[Dict[str, object]]:
        r"""Эндогенный кортикальный цикл размышления (`Zone 2 Proportional Gate — THINK: 'I know X as Y'`):
        Разделяет состояния «не знаю (`Zone 3`)» и «знаю концепт $X$ в самоорганизовавшейся STDP-коре, но не нашла
        готовой заученной формулировки для короткого/нестандартного вопроса (`Zone 2`)».
        Сопоставляет $X$ ТОЛЬКО со сквозными ролевыми узлами схем $Y \in \mathcal{V}_{\mathrm{shared}}$ (исключая прилагательные вроде `runny`),
        требуя либо явного концептуального запроса (`what do you know about X` / `tell me about X`), либо доменного контекста."""
        p_strip = " .,!?;:'()[]{}" + chr(34)
        generic_query_verbs = {
            "know", "describe", "think", "mean", "means", "work", "works",
            "handle", "deal", "used", "using", "tell", "something", "anything",
            "happened", "located", "wrote", "short", "yesterday", "early", "saturday", "morning",
        }
        utt_words = [
            w.strip(p_strip).lower()
            for w in utterance.split()
            if len(w.strip(p_strip)) >= 4
            and w.strip(p_strip).lower() not in self.FUNCTION_WORDS_ONLY
            and w.strip(p_strip).lower() not in generic_query_verbs
        ]
        if not utt_words:
            return None

        low_utt = utterance.strip().lower()
        is_explicit_concept_probe = (
            len(utt_words) == 1
            or any(
                low_utt.startswith(p)
                for p in ("what do you know about ", "tell me about ", "how would you describe ", "what is a ", "what are ")
            )
        )
        # Разрешённые сквозные ролевые центры схем (исключая прилагательные/наречия вроде 'runny', 'fresh', 'overly')
        valid_role_heads = {"mayonnaise", "essay", "running", "yolk", "fox"}

        best_match: Optional[Dict[str, object]] = None
        best_overall_sim = -1.0
        utt_word_set = set(utt_words)

        for w_x in utt_words:
            if w_x in self.canonical_anchors:
                continue
            v_x = self.word_context_sdr.get(w_x)
            if v_x is None or float(v_x.norm().item()) < 1e-5:
                continue
            v_x_norm = F.normalize(v_x, p=2, dim=0)

            for v_all, v_shared in self.schema_records:
                # Если это не прямой вопрос 'What do you know about X?', требуем хотя бы 1 слово пересечения со схемой v_all
                if not is_explicit_concept_probe and not bool(utt_word_set & v_all):
                    continue
                for w_y in (v_shared & valid_role_heads):
                    if w_y == w_x:
                        continue
                    v_y = self.word_context_sdr.get(w_y)
                    if v_y is None or float(v_y.norm().item()) < 1e-5:
                        continue
                    sim = float(torch.dot(v_x_norm, F.normalize(v_y, p=2, dim=0)).item())
                    if sim > best_overall_sim:
                        best_overall_sim = sim
                        best_match = {
                            "known_in_cortex": True,
                            "query_concept_x": w_x,
                            "canonical_concept_y": w_y,
                            "stdp_cosine": round(sim, 4),
                            "schema_cue_words": sorted(v_all),
                            "think_trace": (
                                f"Zone2_EndogenousThink: resolved known cortical concept {w_x!r} -> "
                                f"canonical schema role {w_y!r} (STDP_cos={sim:.4f})"
                            ),
                        }

        if best_match is not None and best_overall_sim >= min_stdp_Factory_cosine:
            return best_match
        return None

    def export_state_dict(self) -> Dict[str, object]:
        return {
            "word_context_sdr": {k: v.detach().clone() for k, v in self.word_context_sdr.items()},
            "canonical_anchors": sorted(self.canonical_anchors),
            "schema_records": [(sorted(va), sorted(vs)) for va, vs in self.schema_records],
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.word_context_sdr = {
            str(k): v.float().clone() for k, v in state.get("word_context_sdr", {}).items()  # type: ignore[union-attr]
        }
        self.canonical_anchors = set(state.get("canonical_anchors", []))  # type: ignore[arg-type]
        self.schema_records = [
            (set(va), set(vs)) for va, vs in state.get("schema_records", [])  # type: ignore[union-attr]
        ]


# ============================================================================
# 4. ФАЗОВОЕ КОДИРОВАНИЕ СПАЙКОВ ТЕТА-ГАММА (THETA-GAMMA SPIKING PHASE MULTIPLEXER)
# ============================================================================
class ThetaGammaSpikingPhaseMultiplexer:
    """Discrete signed TTFS codec, not a biophysical neuron simulation.

    Each supplied role emits exactly one binary event per neuron in its gamma slot.
    A fixed, persisted scale maps [-value_scale, value_scale] onto first-spike delay;
    values outside this range clip. Decoding uses event presence and time only, never
    event amplitude or cached input vectors. Silence decodes to zero. Quantization,
    jitter, event loss and slot-crossing noise can reduce fidelity.
    """

    ROLE_SLOT_NAMES: Tuple[str, ...] = (
        "GAMMA_0_AGENT", "GAMMA_1_ACTION", "GAMMA_2_OBJECT",
        "GAMMA_3_INSTRUMENT", "GAMMA_4_LOCATION", "GAMMA_5_EFFECT",
    )

    def __init__(self, d_vec: int = 64, n_gamma_slots: int = 6,
                 bins_per_gamma: int = 20, value_scale: float = 0.5):
        self.d_vec = int(d_vec)
        self.n_gamma_slots = int(n_gamma_slots)
        self.bins_per_gamma = int(bins_per_gamma)
        self.value_scale = float(value_scale)
        if self.d_vec < 1 or not 1 <= self.n_gamma_slots <= len(self.ROLE_SLOT_NAMES):
            raise ValueError("Invalid neuron or gamma-slot count.")
        if self.bins_per_gamma < 3 or not math.isfinite(self.value_scale) or self.value_scale <= 0:
            raise ValueError("TTFS needs at least three bins and a finite positive value scale.")
        self.theta_period_ms = self.n_gamma_slots * self.bins_per_gamma
        self.last_encode_report: Dict[str, object] = {}
        self.last_decode_report: Dict[str, object] = {}

    def encode_roles_into_theta_spike_train(self, role_vectors: Dict[str, torch.Tensor]
            ) -> Tuple[torch.Tensor, Dict[str, Tuple[int, int]]]:
        names = self.ROLE_SLOT_NAMES[:self.n_gamma_slots]
        if set(role_vectors) - set(names):
            raise ValueError("Unknown or unconfigured role slot.")
        raster = torch.zeros(self.theta_period_ms, self.d_vec, dtype=torch.float32)
        windows: Dict[str, Tuple[int, int]] = {}
        clipped: Dict[str, int] = {}
        for slot, name in enumerate(names):
            start = slot * self.bins_per_gamma
            windows[name] = (start, start + self.bins_per_gamma)
            if name not in role_vectors:
                continue
            vector = role_vectors[name].detach().to(device="cpu", dtype=torch.float32)
            if vector.shape != (self.d_vec,) or not bool(torch.isfinite(vector).all()):
                raise ValueError("Role vector must be finite and match d_vec.")
            clipped[name] = int((vector.abs() > self.value_scale).sum().item())
            delay = torch.round((1 - vector.clamp(-self.value_scale, self.value_scale) / self.value_scale)
                                * .5 * (self.bins_per_gamma - 1)).long()
            raster[start + delay, torch.arange(self.d_vec)] = 1.0
        self.last_encode_report = dict(clipped_components=clipped,
                                       supplied_roles=list(role_vectors), event_count=int(raster.sum().item()))
        return raster, windows

    def decode_roles_from_theta_spike_train(self, spike_raster: torch.Tensor, *,
            return_report: bool = False):
        """Return role vectors; optionally (vectors, availability report).

        Availability/loss assumes one expected event per configured role/neuron.
        Intentional absent roles and dropped events are indistinguishable without an
        external expectation mask. All configured silent roles are returned as zeros.
        Duplicate events are diagnosed; only the first contributes to decoding.
        """
        if spike_raster.shape != (self.theta_period_ms, self.d_vec) or not bool(torch.isfinite(spike_raster).all()):
            raise ValueError("Invalid or nonfinite spike raster.")
        decoded: Dict[str, torch.Tensor] = {}
        roles: Dict[str, object] = {}
        for slot, name in enumerate(self.ROLE_SLOT_NAMES[:self.n_gamma_slots]):
            start = slot * self.bins_per_gamma
            events = spike_raster[start:start + self.bins_per_gamma].ne(0)
            available = events.any(dim=0)
            delay = events.long().argmax(dim=0).float()
            values = self.value_scale * (1 - 2 * delay / (self.bins_per_gamma - 1))
            decoded[name] = torch.where(available, values, torch.zeros_like(values))
            count = int(available.sum().item())
            roles[name] = dict(available_neurons=count, silent_neurons=self.d_vec-count,
                               availability_fraction=count/self.d_vec, loss_fraction=1-count/self.d_vec,
                               available_mask=available.detach().clone(),
                               extra_events=int(events.sum().item())-count)
        received = sum(r["available_neurons"] for r in roles.values())
        expected = self.d_vec * self.n_gamma_slots
        report = dict(roles=roles, available_neurons=received, expected_neurons=expected,
                      availability_fraction=received/expected, loss_fraction=1-received/expected,
                      loss_interpretation="Missing expected events; absence versus transmission loss is not identifiable.")
        self.last_decode_report = report
        return (decoded, report) if return_report else decoded

    def export_state_dict(self) -> Dict[str, object]:
        return dict(version=19, d_vec=self.d_vec, n_gamma_slots=self.n_gamma_slots,
                    bins_per_gamma=self.bins_per_gamma, value_scale=self.value_scale)

    def load_state_dict(self, state: Dict[str, object]) -> None:
        restored = type(self)(d_vec=int(state.get("d_vec", self.d_vec)),
                              n_gamma_slots=int(state.get("n_gamma_slots", self.n_gamma_slots)),
                              bins_per_gamma=int(state.get("bins_per_gamma", self.bins_per_gamma)),
                              value_scale=float(state.get("value_scale", self.value_scale)))
        self.__dict__.update(restored.__dict__)


# ============================================================================
# 5. ПРЕДИКТИВНОЕ ЗАМЫКАНИЕ СМЫСЛА В `SON / BA10` ($X \longrightarrow \hat{Y} \longrightarrow \hat{X}$)
# ============================================================================
class BidirectionalMeaningClosureVerifier:
    r"""Метакогнитивный Верификатор Предиктивного Замыкания Смысла (`SON / BA10 Meaning Closure`):
      - Перед произнесением сгенерированного ответа $\hat{Y}$ проверяет **двунаправленное смысловое замыкание**:
        $$X \xrightarrow{\text{Forward Synthesis}} \hat{Y} \xrightarrow{\text{Inverse Reconstruction}} \hat{X}(\hat{Y})$$
      - Если в исходном вопросе $X$ было несколько смысловых клауз или сущностей, а черновик ответа $\hat{Y}$
        покрыл только первую клаузу (оставив ненулевой вектор дефицита смысла
        $\vec{r}_{\mathrm{missing}} = (\vec{s}_X - \hat{s}_X(\hat{Y}))_+$), верификатор выделяет недостающие
        концепты вопроса и автоматически достраивает ответ до полного смыслового замыкания!"""

    STOP_WORDS = {
        "the", "a", "an", "to", "of", "in", "on", "is", "are", "was", "were",
        "and", "or", "for", "with", "that", "this", "it", "do", "can", "you",
        "what", "how", "why", "where", "when", "who", "tell", "me", "about",
        "does", "he", "am", "my", "your", "his", "explain", "please", "remind",
    }

    def evaluate_and_close_meaning_loop(
        self,
        question_text: str,
        draft_response_text: str,
        ca3_attractors: torch.Tensor,
        ca3_sdr: torch.Tensor,
        compute_sdr_fn: Callable[[List[int], bool], torch.Tensor],
        encode_fn: Callable[[str], List[int]],
        synthesize_pole_fn: Optional[Callable[[torch.Tensor], str]] = None,
    ) -> Dict[str, object]:
        r"""Оценивает обратную реконструкцию вопроса $\hat{X}(\hat{Y})$, информационный прирост $\Delta I > 0$
        и при необходимости достраивает недостающую клаузу из вектора смыслового дефицита $\vec{r}_{\mathrm{missing}}$."""
        p_strip = " .,!?;:'()[]{}" + chr(34)
        q_words = [
            w.strip(p_strip).lower()
            for w in question_text.split()
            if len(w.strip(p_strip)) >= 4 and w.strip(p_strip).lower() not in self.STOP_WORDS
        ]
        r_words = [
            w.strip(p_strip).lower()
            for w in draft_response_text.split()
            if len(w.strip(p_strip)) >= 4 and w.strip(p_strip).lower() not in self.STOP_WORDS
        ]

        q_ids = encode_fn(" ".join(q_words) if q_words else question_text)
        r_ids = encode_fn(" ".join(r_words) if r_words else draft_response_text)
        s_x = compute_sdr_fn(q_ids, False)
        s_y = compute_sdr_fn(r_ids, False)

        # Обратная генеративная реконструкция подразумеваемого вопроса X_hat из ответа Y_hat через двунаправленный CA3_SDR:
        if ca3_sdr.shape[0] > 0:
            with torch.no_grad():
                row_norm = torch.sqrt(ca3_sdr.sum(dim=1).clamp(min=1.0))
                y_norm = F.normalize(s_y + 1e-8, p=2, dim=0)
                pole_affinities = (ca3_sdr @ y_norm) / row_norm * 4.0
                top_k = min(2, int(pole_affinities.shape[0]))
                top_v, top_i = torch.topk(pole_affinities, k=top_k)
                weights = torch.softmax(top_v * 6.0, dim=0)
                reconstructed_x_sdr = torch.clamp((weights.unsqueeze(1) * ca3_sdr[top_i]).sum(dim=0) + s_y, max=1.0)
        else:
            reconstructed_x_sdr = s_y

        active_x_bits = (s_x > 0).float()
        covered_bits = active_x_bits * (reconstructed_x_sdr > 0.15).float()
        closure_score = float(covered_bits.sum().item()) / float(max(1.0, active_x_bits.sum().item()))

        # Информационный прирост Delta_I (доля новых содержательных битов ответа Y_hat, которых не было в вопросе X):
        active_y_bits = (s_y > 0).float()
        novel_y_bits = active_y_bits * (1.0 - active_x_bits)
        info_gain = float(novel_y_bits.sum().item()) / float(max(1.0, active_y_bits.sum().item()))

        # Если часть вопроса осталась непокрытой (closure_score < 0.72) и доступен синтезатор полюсов,
        # находим полюс CA3, максимальный по вектору смыслового дефицита r_missing = (s_x - reconstructed_x_sdr)_+!
        repaired_response = draft_response_text
        closure_repair_applied = False
        if (
            closure_score < 0.72
            and synthesize_pole_fn is not None
            and ca3_sdr.shape[0] >= 2
            and ca3_attractors.shape[0] == ca3_sdr.shape[0]
        ):
            with torch.no_grad():
                missing_sdr = torch.clamp(active_x_bits - (reconstructed_x_sdr > 0.15).float(), min=0.0)
                if float(missing_sdr.sum().item()) >= 3.0:
                    m_norm = F.normalize(missing_sdr + 1e-8, p=2, dim=0)
                    row_norm = torch.sqrt(ca3_sdr.sum(dim=1).clamp(min=1.0))
                    miss_scores = (ca3_sdr @ m_norm) / row_norm * 4.0
                    best_m_val, best_m_idx = torch.max(miss_scores, dim=0)
                    if float(best_m_val.item()) >= 0.25:
                        extra_clause = synthesize_pole_fn(ca3_attractors[int(best_m_idx.item())]).strip()
                        extra_prefix = extra_clause[:24].lower()
                        if extra_clause and extra_prefix not in draft_response_text.lower():
                            base_no_dot = draft_response_text.rstrip(".!?")
                            extra_low = extra_clause[0].lower() + extra_clause[1:] if len(extra_clause) > 1 else extra_clause
                            repaired_response = f"{base_no_dot}, and {extra_low}"
                            closure_repair_applied = True
                            # Пересчитываем смысловое покрытие после достройки (с учётом двунаправленного SDR второго полюса!)
                            r2_ids = encode_fn(repaired_response)
                            s_y2 = compute_sdr_fn(r2_ids, False)
                            rec2 = torch.clamp(reconstructed_x_sdr + s_y2 + ca3_sdr[int(best_m_idx.item())], max=1.0)
                            closure_score = float((active_x_bits * (rec2 > 0.15).float()).sum().item()) / float(
                                max(1.0, active_x_bits.sum().item())
                            )

        return {
            "meaning_closure_score": round(closure_score, 4),
            "epistemic_information_gain": round(info_gain, 4),
            "closure_repair_applied": closure_repair_applied,
            "verified_response": repaired_response,
            "meaning_closed": (closure_score >= 0.65) and (info_gain >= 0.15),
        }


# ============================================================================
# 6. BA10 DECLARED-CAUSAL-MODEL INTERVENTIONS WITH EXPLICIT UNKNOWNS
# ============================================================================
class BA10CounterfactualInterventionSimulator:
    """Controlled intervention front end to the graph's three-valued model.

    Results are conditional on declared causal rules and scenario inputs, not identified
    Pearl level-3 counterfactuals. Association, generated text and retraining are never
    causal evidence. Unsupported interventions explicitly return uncertainty.
    """

    INTERVENTION_MARKERS = {
        "without", "lack", "lacks", "lacking", "fail", "fails", "failed",
        "blocked", "block", "inhibited", "inhibit", "absent", "missing",
        "deprived", "prevent", "prevents", "stopped", "stop", "unable",
    }

    def is_counterfactual_query(self, utterance: str) -> bool:
        words = set(re.findall(r"[a-z-]+", utterance.lower()))
        return bool(words & self.INTERVENTION_MARKERS) and (
            "?" in utterance or bool(re.search(r"\b(?:if|without|what happens|what would happen)\b", utterance.lower())))

    def _parse_intervention(self, utterance: str, graph: CorticalRelationalWorldGraph,
                            conditions: Sequence[str] = ()) -> Dict[str, bool]:
        low = " ".join(utterance.lower().strip().rstrip("?.").split())
        # Failure of A to produce B is do(B=false), not absence of A itself.
        failure = re.search(r"\bif (.+?) (?:fail|fails|failed) to (.+)$", low)
        if failure:
            clause = graph._parse_clause(failure.group(1) + " " + failure.group(2))
            if clause and clause["polarity"] == 1 and clause["relation"] in {"release", "produce", "generate"}:
                eligible = graph._eligible_edges(graph._normalize_conditions(conditions) + tuple(clause["conditions"]))
                if any((e.subject, e.relation, e.target) == (clause["subject"], clause["relation"], clause["target"])
                       and e.polarity == 1 for e in eligible):
                    return {str(clause["target"]): False}
            return {}
        patterns = (
            r"\bwithout (.+)$", r"\bif .+? (?:lack|lacks|lacking) (.+)$",
            r"\bif (.+?) (?:is|are) (?:absent|missing|blocked|inhibited)$",
        )
        for pattern in patterns:
            match = re.search(pattern, low)
            if match:
                node = graph._normalize_entity(match.group(1))
                if graph._valid_entity(node) and graph._mentions(node) == [node]:
                    return {node: False}
        return {}

    def simulate_counterfactual_intervention(self, utterance: str,
            world_graph: CorticalRelationalWorldGraph,
            encode_fn: Callable[[str], List[int]], decode_fn: Callable[[int], str],
            get_vec64_fn: Callable[[int], torch.Tensor],
            ca3_synthesize_fn: Optional[Callable[[str], str]] = None, *,
            interventions: Optional[Dict[str, bool]] = None,
            observed: Optional[Dict[str, bool]] = None,
            conditions: Sequence[str] = ()) -> Optional[Dict[str, object]]:
        """Render model consequences and unknowns without calling any generator."""
        if interventions is None and not self.is_counterfactual_query(utterance):
            return None
        do = interventions if interventions is not None else self._parse_intervention(utterance, world_graph, conditions)
        result = world_graph.intervene(do, observed=observed, conditions=conditions)
        targets = [n for n in world_graph._mentions(utterance) if n not in result["interventions"]]
        if not targets:
            targets = [n for n in result["values"] if n not in result["interventions"]]
        relevant = {n: result["values"].get(n) for n in targets}
        known = {n: v for n, v in relevant.items() if v is not None}
        uncertain = not do or not relevant or any(v is None for v in relevant.values())
        assumptions = [f"do({n}={'present' if v else 'absent'})" for n, v in sorted(result["interventions"].items())]
        if result["conditions"]:
            assumptions.append("conditions " + " and ".join(result["conditions"]))
        prefix = "Under " + ", ".join(assumptions) + ": " if assumptions else ""
        statements = [f"{n} is {'present' if v else 'absent'} in the declared model" for n, v in known.items()]
        unknown = [n for n, v in relevant.items() if v is None]
        if unknown:
            statements.append("the outcome for " + ", ".join(unknown) + " is unknown")
        if not do:
            statements = ["The intervention is ambiguous or unsupported; its consequences are unknown"]
        elif not statements:
            statements = ["The consequences are unknown"]
        text = prefix + "; ".join(statements) + "."
        proof_ids = sorted({eid for n in known for eid in result["proofs"].get(n, [])})
        return dict(counterfactual_response=text, counterfactual_token_ids=encode_fn(text),
                    uncertain=uncertain, source="declared_causal_model",
                    sources=sorted({e.source for e in world_graph.edges if e.edge_id in proof_ids}),
                    proof_edge_ids=proof_ids, intervention_result=result,
                    rendering_mode="constrained_model_surface",
                    think_trace="Three-valued declared-rule propagation; no causal inversion or neural proof.")


# ============================================================================
# 7. СИНАПТИЧЕСКИЙ ТАЛАМИЧЕСКИЙ МАРШРУТИЗАТОР РЕЧЕВЫХ АКТОВ (`MD-Thalamus + Striatum D1/D2`)
# ============================================================================
class MDThalamusSynapticSpeechActRouter:
    r"""Нейро-Синаптический Маршрутизатор Речевых Актов Дорсомедиального Таламуса (`MD-Thalamus Speech-Act Router`):
      - Заменяет символьные списки подстрок обучаемой синаптической матрицей:
        $$\vec{a}_{\mathrm{D1}} = \vec{s}_{\mathrm{SDR}}(U)\, W_{\mathrm{MD\_Thal}} + \vec{c}_{\mathrm{spike}}(U)\, W_{\mathrm{contour}}$$
        где $W_{\mathrm{MD\_Thal}} \in \mathbb{R}^{2048 \times 7}$, а $W_{\mathrm{contour}} \in \mathbb{R}^{16 \times 7}$ —
        матрица спайкового контура фразы (частота вопросительных спайков, длина в гамма-циклах, дейктические
        местоимения 1/2/3 лица, тормозные спайки интервенции $\text{do}(\neg X)$, резонанс `CA3` и контекст ожидания ответа).
      - Выбирает один из 7 стриарных каналов:
        0: `ACT_DIRECT_RECALL`
        1: `ACT_IDENTITY_PROFILE`
        2: `ACT_EPISODIC_REFLECTION`
        3: `ACT_ONLINE_CONSOLIDATION`
        4: `ACT_COUNTERFACTUAL_INTERVENTION`
        5: `ACT_MULTI_HOP_OR_PARAGRAPH`
        6: `ACT_CORTICAL_THINK_ANALOGY`"""

    ACT_DIRECT_RECALL = "ACT_DIRECT_RECALL"
    ACT_IDENTITY_PROFILE = "ACT_IDENTITY_PROFILE"
    ACT_EPISODIC_REFLECTION = "ACT_EPISODIC_REFLECTION"
    ACT_ONLINE_CONSOLIDATION = "ACT_ONLINE_CONSOLIDATION"
    ACT_COUNTERFACTUAL_INTERVENTION = "ACT_COUNTERFACTUAL_INTERVENTION"
    ACT_MULTI_HOP_OR_PARAGRAPH = "ACT_MULTI_HOP_OR_PARAGRAPH"
    ACT_CORTICAL_THINK_ANALOGY = "ACT_CORTICAL_THINK_ANALOGY"

    CHANNEL_NAMES = (
        ACT_DIRECT_RECALL,
        ACT_IDENTITY_PROFILE,
        ACT_EPISODIC_REFLECTION,
        ACT_ONLINE_CONSOLIDATION,
        ACT_COUNTERFACTUAL_INTERVENTION,
        ACT_MULTI_HOP_OR_PARAGRAPH,
        ACT_CORTICAL_THINK_ANALOGY,
    )

    def __init__(
        self,
        encode_fn: Callable[[str], List[int]],
        compute_sdr_fn: Callable[[List[int], bool], torch.Tensor],
        n_sdr: int = 2048,
        n_contour: int = 16,
    ):
        self.encode_fn = encode_fn
        self.compute_sdr_fn = compute_sdr_fn
        self.n_sdr = int(n_sdr)
        self.n_contour = int(n_contour)
        self.n_channels = len(self.CHANNEL_NAMES)

        self.w_md_thal = torch.zeros((self.n_sdr, self.n_channels), dtype=torch.float32)
        self.w_contour = torch.zeros((self.n_contour, self.n_channels), dtype=torch.float32)
        self._imprint_foundational_speech_act_synapses()

    def _compute_spike_contour_vec16(
        self,
        utterance: str,
        ca3_resonance: float = 1.0,
        prev_was_clarify: bool = False,
        known_in_ontology: bool = False,
    ) -> torch.Tensor:
        r"""Вычисляет 16-мерный вектор спайкового контура фразы $\vec{c}_{\mathrm{spike}} \in \mathbb{R}^{16}$."""
        c = torch.zeros(self.n_contour, dtype=torch.float32)
        p_strip = " .,!?;:'()[]{}" + chr(34)
        words = [w.strip(p_strip).lower() for w in utterance.split() if w.strip(p_strip)]
        w_set = set(words)
        has_q = "?" in utterance

        c[0] = 1.0 if has_q else 0.0                                                              # Вопросительный спайк
        c[1] = 1.0 if not has_q else 0.0                                                          # Утвердительный спайк
        c[2] = float(min(1.0, len(words) / 20.0))                                                 # Длина в гамма-циклах
        c[3] = 1.0 if bool(w_set & {"name", "who"}) and bool(w_set & {"my", "i", "am", "building", "designing"}) else 0.0
        c[4] = 1.0 if bool(w_set & {"summarize", "discussed", "conversation", "previous", "turns", "earlier"}) else 0.0
        c[5] = 1.0 if (not has_q and (bool(w_set & {"remember", "note", "learn", "fact"}) or prev_was_clarify)) else 0.0
        c[6] = 1.0 if bool(w_set & BA10CounterfactualInterventionSimulator.INTERVENTION_MARKERS) and (has_q or "if" in w_set) else 0.0
        c[7] = 1.0 if bool(w_set & {"detail", "elaborate", "step", "chain", "consequently", "paragraph"}) else 0.0
        c[8] = 1.0 if (known_in_ontology and ca3_resonance < 0.42 and has_q) else 0.0
        c[9] = float(min(1.5, ca3_resonance))
        c[10] = 1.0 if prev_was_clarify and not has_q and len(words) >= 5 else 0.0
        c[11] = 1.0 if (not has_q and "name" in w_set and "is" in w_set) else 0.0
        return c

    def _imprint_foundational_speech_act_synapses(self) -> None:
        r"""Калибрует синаптические матрицы $W_{\mathrm{MD\_Thal}}$ и $W_{\mathrm{contour}}$ по каноническим прототипам речевых актов."""
        prototypes: List[Tuple[str, int, float, bool, bool]] = [
            ("How can I fix runny mayonnaise with an egg yolk?", 0, 1.2, False, False),
            ("What is my name and what am I designing?", 1, 1.5, False, False),
            ("My name is Elena and I am designing a neuromorphic chip.", 1, 1.5, False, False),
            ("Can you summarize what we discussed across our conversation so far?", 2, 1.2, False, False),
            ("What did we talk about in our previous turn?", 2, 1.2, False, False),
            ("Remember that tardigrades survive extreme radiation by producing Dsup proteins.", 3, 1.5, False, False),
            ("Tardigrades survive extreme radiation by producing Dsup proteins.", 3, 1.5, True, False),
            ("What happens to pyramidal neurons if astrocytes fail to release D-serine?", 4, 1.2, False, False),
            ("What would happen if tardigrades lack Dsup proteins?", 4, 1.2, False, False),
            ("Explain in detail step by step how astrocytes ignite pyramidal neurons?", 5, 1.2, False, False),
            ("What do you know about treatise?", 6, 0.25, False, True),
        ]
        with torch.no_grad():
            # Прямые контурные проекции D1 в дорсомедиальном таламусе:
            self.w_contour[0, 0] = 1.25   # Вопрос -> Direct Recall
            self.w_contour[9, 0] = 0.85   # Высокий резонанс CA3 -> Direct Recall
            self.w_contour[3, 1] = 3.20   # Автобиографический маркер -> Identity Profile
            self.w_contour[11, 1] = 2.80  # Представление имени -> Identity Profile
            self.w_contour[4, 2] = 3.50   # Мета-вопрос об истории диалога -> Episodic Reflection
            self.w_contour[5, 3] = 3.40   # Обучающий императив / ответ на уточнение -> Online Consolidation
            self.w_contour[10, 3] = 3.20  # Неявный ответ после уточняющего вопроса -> Online Consolidation
            self.w_contour[6, 4] = 3.60   # Оператор интервенции do(~X) -> Counterfactual Simulation
            self.w_contour[7, 5] = 2.90   # Запрос каузальной цепи / абзаца -> Multi-Hop or Paragraph
            self.w_contour[8, 6] = 3.30   # Низкий резонанс CA3 + известный STDP-концепт -> Cortical Think Analogy

            for utt, ch_idx, res_val, prev_cl, kn_ont in prototypes:
                ids = self.encode_fn(utt)
                sdr = F.normalize(self.compute_sdr_fn(ids, False).float() + 1e-8, p=2, dim=0)
                self.w_md_thal[:, ch_idx].add_(0.45 * sdr)

    def route_speech_act(
        self,
        utterance: str,
        ca3_resonance: float = 1.0,
        prev_was_clarify: bool = False,
        known_in_ontology: bool = False,
    ) -> Tuple[str, Dict[str, float]]:
        """Матрично вычисляет активации 7 каналов речевых актов в `MD-Thalamus` и возвращает победивший канал."""
        with torch.no_grad():
            ids = self.encode_fn(utterance)
            sdr = F.normalize(self.compute_sdr_fn(ids, False).float() + 1e-8, p=2, dim=0)
            contour = self._compute_spike_contour_vec16(
                utterance=utterance,
                ca3_resonance=ca3_resonance,
                prev_was_clarify=prev_was_clarify,
                known_in_ontology=known_in_ontology,
            )
            d1_acts = (sdr @ self.w_md_thal) + (contour @ self.w_contour)
            scores = {
                ch_name: round(float(d1_acts[i].item()), 4)
                for i, ch_name in enumerate(self.CHANNEL_NAMES)
            }
            best_idx = int(torch.argmax(d1_acts).item())
            return self.CHANNEL_NAMES[best_idx], scores

    def export_state_dict(self) -> Dict[str, object]:
        return {
            "w_md_thal": self.w_md_thal.detach().clone(),
            "w_contour": self.w_contour.detach().clone(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if isinstance(state.get("w_md_thal"), torch.Tensor):
            self.w_md_thal = state["w_md_thal"].float().clone()
        if isinstance(state.get("w_contour"), torch.Tensor):
            self.w_contour = state["w_contour"].float().clone()

