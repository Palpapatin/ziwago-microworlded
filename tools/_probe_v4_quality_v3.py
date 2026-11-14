#!/usr/bin/env python3
"""Read-only probe: how many existing v4 rows would actually pass the trainer's gates?

Applies the exact filters from tools/diag_v19_decoder_credit.py:train_pilot:
  * 2 <= response BPE tokens <= max_response_tokens (64) and < readout.max_pos (64)
  * no EOS control token inside the response
  * evaluate_grammar_and_artifacts(response, min_words=3).grammar_coherent
  * authority.ground(prompt, response).grounded
This script only reads; it never writes into data/hf or Produce.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ziwago.evaluation.gates import evaluate_grammar_and_artifacts
from train_ziwago_v9_gpt2vocab import _GPT2_PAT, _bpe_encode_piece, _init_bpe_worker
import train_ziwago_v9_gpt2vocab as tokenizer

TARGET = Path(r"D:\Ziwago18-storage\data\hf\squad\language_v19_v4\train.jsonl")
ACCEPT = Path(r"D:\Ziwago18-storage\data\hf\squad\language_v19_v4\acceptance.json")


def bpe_len(text: str) -> int:
    total = 0
    for piece in _GPT2_PAT.findall(" " + text):
        total += len(_bpe_encode_piece(piece, tokenizer._WORKER_VOCAB, tokenizer._WORKER_MERGES,
                                       tokenizer._WORKER_CACHE))
    return total


def main() -> int:
    _init_bpe_worker(str(ROOT / "data/gpt2/vocab.json"), str(ROOT / "data/gpt2/merges.txt"))
    rows = [json.loads(line) for line in TARGET.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = json.loads(ACCEPT.read_text(encoding="utf-8"))
    manifest = json.loads((TARGET.parent / "manifest.json").read_text(encoding="utf-8"))

    print("rows:", len(rows), "acceptance cases:", len(cases))
    print("row keys:", sorted(rows[0].keys()))
    print("acceptance keys:", sorted(cases[0].keys()))
    print("manifest keys:", sorted(manifest.keys()))
    print("manifest per_source:", json.dumps(manifest.get("per_source"), ensure_ascii=False))
    print("manifest cross_split_family_overlap:", manifest.get("cross_split_family_overlap"))

    train_families = {r["family_id"] for r in rows}
    accept_families = {c["family_id"] for c in cases}
    print("train families:", len(train_families), "accept families:", len(accept_families),
          "overlap:", len(train_families & accept_families))
    print("acceptance has expected_concepts:",
          sum(1 for c in cases if isinstance(c.get("expected_concepts"), list) and c["expected_concepts"]),
          "/", len(cases))

    stats = Counter()
    per_source_pass = Counter()
    per_source_total = Counter()
    for row in rows:
        src = row["source"]
        per_source_total[src] += 1
        response = row["response"]
        tokens = bpe_len(response)
        if not 2 <= tokens <= 63:
            stats["token_budget"] += 1
            continue
        if not evaluate_grammar_and_artifacts(response, min_words=3)["grammar_coherent"]:
            stats["grammar"] += 1
            continue
        stats["passes_static_gates"] += 1
        per_source_pass[src] += 1

    print("\nstatic gate results over all", len(rows), "train rows:")
    print("  rejected token_budget:", stats["token_budget"])
    print("  rejected grammar    :", stats["grammar"])
    print("  passes static gates :", stats["passes_static_gates"])
    print("\nper source (pass/total):")
    for src in sorted(per_source_total):
        print(f"  {src:<22} {per_source_pass[src]:>4} / {per_source_total[src]:>4}")
    print("\nfirst-300 selection composition:")
    head = rows[:300]
    print(" ", Counter(r["source"] for r in head))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
