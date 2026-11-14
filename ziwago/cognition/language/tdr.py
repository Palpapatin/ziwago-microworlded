"""Candidate-first token retrieval (TDR) for the language readout.

The index is derived from encoder-owned token codes and the active vocabulary.
Pool A scores SDR overlap through an inverted index; pool B scores cosine
similarity in the encoder's cortical projection space. A full pool reproduces
the original scorer; a truncated pool is reported, never silently substituted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass
class CandidateProposal:
    cols: List[int]
    tokens: List[int]
    report: Dict[str, object]


class TokenCandidateIndex:
    def __init__(self, encoder):
        self.encoder = encoder
        self.token_ids: List[int] = []
        self.token_to_col: Dict[int, int] = {}
        self._sdr_bits: List[List[int]] = []
        self._inverted: Dict[int, List[int]] = {}
        self._vsa: Optional[torch.Tensor] = None

    def ensure_vocab(self, token_ids: Sequence[int]) -> None:
        ids = [int(t) for t in token_ids]
        if ids == self.token_ids:
            return
        keep = len(self.token_ids)
        if len(ids) < keep or ids[:keep] != self.token_ids:
            self.token_ids, self.token_to_col = [], {}
            self._sdr_bits, self._inverted, self._vsa = [], {}, None
            keep = 0
        vecs = []
        for col, tid in enumerate(ids[keep:], keep):
            sdr = self.encoder.encode_token_id(tid)
            bits = [int(b) for b in torch.nonzero(sdr > 0.0).flatten().tolist()]
            self._sdr_bits.append(bits)
            for bit in bits:
                self._inverted.setdefault(bit, []).append(col)
            vec = self.encoder.project_token_to_cortex(tid).detach().float()
            vecs.append(F.normalize(vec, p=2, dim=0))
            self.token_to_col[tid] = col
        self.token_ids = ids
        if vecs:
            mat = torch.stack(vecs)
            self._vsa = mat if self._vsa is None else torch.cat((self._vsa, mat), dim=0)

    def _query_bits(self, prefix: Sequence[int]) -> List[int]:
        bits = set()
        for tid in prefix:
            t = int(tid)
            col = self.token_to_col.get(t)
            if col is not None:
                bits.update(self._sdr_bits[col])
            else:
                sdr = self.encoder.encode_token_id(t)
                bits.update(int(b) for b in torch.nonzero(sdr > 0.0).flatten().tolist())
        return sorted(bits)

    def propose(
        self,
        q_vec: torch.Tensor,
        prefix: Sequence[int],
        token_ids: Sequence[int],
        *,
        budget: Optional[int] = None,
        pool_a: int = 64,
        pool_b: int = 64,
        target_token: Optional[int] = None,
    ) -> CandidateProposal:
        self.ensure_vocab(token_ids)
        n = len(self.token_ids)
        report: Dict[str, object] = {
            "enabled": True, "vocabulary_size": n, "budget": budget,
            "pool_a_budget": int(pool_a), "pool_b_budget": int(pool_b),
            "pool_a_candidates": 0, "pool_b_candidates": 0,
            "union_candidates": 0, "selected_candidates": 0,
            "forced_target": 0, "query_bits": 0, "mode": "candidate_pool",
            "reason": "ok",
        }
        if n == 0:
            report.update(mode="empty", reason="empty_vocabulary")
            return CandidateProposal([], [], report)
        q = F.normalize(q_vec.detach().float(), p=2, dim=0)
        qbits = self._query_bits(prefix)
        report["query_bits"] = len(qbits)
        if not qbits and float(q.norm()) == 0.0:
            report.update(mode="empty", reason="no_query_signal")
            return CandidateProposal([], [], report)
        if budget is None or int(budget) >= n:
            report.update(mode="full", selected_candidates=n)
            return CandidateProposal(list(range(n)), list(self.token_ids), report)

        counts: Dict[int, int] = {}
        for bit in qbits:
            for col in self._inverted.get(bit, ()):
                counts[col] = counts.get(col, 0) + 1
        a_cols = sorted(counts, key=lambda c: (-counts[c], c))[: max(0, int(pool_a))]
        b_cols: List[int] = []
        if int(pool_b) > 0 and self._vsa is not None and float(q.norm()) > 0.0:
            sims = self._vsa @ q
            b_cols = torch.topk(sims, k=min(int(pool_b), n)).indices.tolist()
        union = list(dict.fromkeys(a_cols + b_cols))
        report.update(pool_a_candidates=len(a_cols), pool_b_candidates=len(b_cols),
                      union_candidates=len(union))
        forced_col = None
        if target_token is not None:
            forced_col = self.token_to_col.get(int(target_token))
            if forced_col is not None and forced_col not in union:
                union.append(forced_col)
                report["forced_target"] = 1
        limit = max(0, int(budget))
        if len(union) > limit:
            union = union[:limit]
            if forced_col is not None and forced_col not in union:
                union[-1] = forced_col
        cols = sorted(union)
        report["selected_candidates"] = len(cols)
        if not cols:
            report.update(mode="empty", reason="empty_candidate_pool")
        return CandidateProposal(cols, [self.token_ids[c] for c in cols], report)