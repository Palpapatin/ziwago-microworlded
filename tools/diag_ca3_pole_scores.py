#!/usr/bin/env python3
"""Read-only diagnostic: dump CA3 scoring internals for competing poles."""
from __future__ import annotations
import sys
from pathlib import Path
import json
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ziwago_assistant import ZiwagoBiologicalSNNAssistant

CKPT = Path("D:/Ziwago18-storage/Produce/checkpoints/ziwago_v19_candidate_1794007671713945800.pt")
QUESTION = "Who managed the Destiny's Child group?"


def main():
    assistant = ZiwagoBiologicalSNNAssistant(checkpoint_path=CKPT)
    core = assistant.organism
    ro = core.dendritic_plateau_readout
    ids = core.encode_text(QUESTION)
    q_raw = core.compute_stimulus_vec64(ids)
    q_norm = F.normalize(q_raw, p=2, dim=0)
    sdr = ro._compute_cortical_sdr2048(ids, normalize=False)

    with torch.no_grad():
        cos_sim = (ro.w_ca3_attractors @ q_norm).clamp(min=0.0)
        idf_col = 1.0 / torch.sqrt(ro.sdr_column_usage)
        q_weighted = F.normalize(sdr * idf_col + 1e-8, p=2, dim=0)
        row_weighted = ro.w_ca3_sdr * idf_col
        row_l2 = row_weighted.norm(p=2, dim=1).clamp(min=1e-8)
        sdr_drive_raw = (row_weighted @ q_weighted) / row_l2
        sdr_drive = torch.where(cos_sim > 0.01, sdr_drive_raw, torch.zeros_like(sdr_drive_raw))
        has_overlap = (sdr_drive > 0.0125).float()
        turn_relevance = cos_sim * (0.30 + 0.25 * has_overlap) + 0.60 * sdr_drive

        salience_gate = 1.0 + 0.15 * torch.tanh(ro.w_ca3_salience - 1.0)
        consolidated_boost = torch.where(
            (sdr_drive > 0.0875) & (ro.w_ca3_salience >= 4.50) & (cos_sim > 0.30),
            torch.full_like(salience_gate, 1.45),
            torch.ones_like(salience_gate),
        )
        sims = turn_relevance * salience_gate * consolidated_boost

        top_v, top_i = torch.topk(sims, k=10)
        rows = []
        for v, idx in zip(top_v, top_i):
            i = int(idx.item())
            rows.append({
                "pole": i,
                "final_score": round(float(v), 4),
                "cos_sim": round(float(cos_sim[i]), 4),
                "sdr_drive_raw": round(float(sdr_drive_raw[i]), 4),
                "sdr_drive_guarded": round(float(sdr_drive[i]), 4),
                "has_overlap": bool(has_overlap[i]),
                "turn_relevance": round(float(turn_relevance[i]), 4),
                "salience": round(float(ro.w_ca3_salience[i]), 4),
                "salience_gate": round(float(salience_gate[i]), 4),
                "consolidated_boost": round(float(consolidated_boost[i]), 4),
            })

        # Also check what the correct pole (371) looks like
        for i in [318, 371, 370, 372, 369]:
            if i < cos_sim.shape[0]:
                rows.append({
                    "pole": i,
                    "final_score": round(float(sims[i]), 4),
                    "cos_sim": round(float(cos_sim[i]), 4),
                    "sdr_drive_raw": round(float(sdr_drive_raw[i]), 4),
                    "sdr_drive_guarded": round(float(sdr_drive[i]), 4),
                    "has_overlap": bool(has_overlap[i]),
                    "turn_relevance": round(float(turn_relevance[i]), 4),
                    "salience": round(float(ro.w_ca3_salience[i]), 4),
                    "salience_gate": round(float(salience_gate[i]), 4),
                    "consolidated_boost": round(float(consolidated_boost[i]), 4),
                    "forced_check": True,
                })

    # Deduplicate by pole index
    seen = set()
    unique = []
    for r in rows:
        if r["pole"] not in seen:
            seen.add(r["pole"])
            unique.append(r)

    print(json.dumps({"question": QUESTION, "total_poles": int(cos_sim.shape[0]),
                      "top10_and_key_poles": unique}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()