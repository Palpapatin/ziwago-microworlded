#!/usr/bin/env python3
"""Смоук-сплит: малая, но честная приёмка из объединённого скоупа.

Зачем: приёмка v3 оказалась неизмеримой — её эталоны содержат токены вне
выученного словаря модели (`teacher_forced_likelihood` → "Reference exceeds the
learned vocabulary or positional window"). Чтобы приёмка вообще считалась,
она должна собираться из тех же строк, на которых модель обучалась. Здесь
берётся срез объединённого скоупа: первые N строк — train, следующие M —
acceptance, без пересечения по промптам и семействам.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HF = Path(r"D:\Ziwago18-storage\data\hf\squad")
SRC = HF / "language_v19_v6_union"
OUT = HF / "language_v19_v7_smoke"
TRAIN, ACCEPT = 24, 6


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    rows = [json.loads(line) for line in (SRC / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    train, accept = rows[:TRAIN], rows[TRAIN:TRAIN + ACCEPT]
    train_families = {r["family_id"] for r in train}
    accept_families = {r["family_id"] for r in accept}
    if train_families & accept_families:
        raise SystemExit("Family overlap between train and acceptance.")
    if {r["prompt"] for r in train} & {r["prompt"] for r in accept}:
        raise SystemExit("Prompt overlap between train and acceptance.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train), encoding="utf-8")
    # Формат приёмки требует prompt / family_id / expected_concepts / reference.
    cases = [{
        "prompt": row["prompt"],
        "expected_concepts": [row["response"]],
        "family_id": row["family_id"],
        "reference": row["response"],
        "split": "acceptance",
        "reference_context_not_provided_to_model": True,
        "provenance": row.get("provenance", {}),
    } for row in accept]
    (OUT / "acceptance.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "schema": "ziwago.language_curriculum.v1",
        "license": "cc-by-sa-4.0",
        "train_records": len(train),
        "acceptance_records": len(accept),
        "train_documents": len(train_families),
        "acceptance_documents": len(accept_families),
        "cross_split_family_overlap": len(train_families & accept_families),
        "selection_policy": "Срез объединённого скоупа v2 ∪ v3 без пересечения по семействам "
                            "и промптам; строки уже прошли бюджет ответа и грамматический гейт "
                            "сборщика, поэтому измеримы моделью.",
        "training_performed": False,
        "eligible_for_future_training": True,
        "source_sha256": {"union_train": sha256_file(SRC / "train.jsonl")},
        "prepared_files": {},
    }
    manifest["prepared_files"][str(OUT / "train.jsonl")] = sha256_file(OUT / "train.jsonl")
    manifest["prepared_files"][str(OUT / "acceptance.json")] = sha256_file(OUT / "acceptance.json")
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(OUT), "train": len(train), "acceptance": len(accept),
                      "train_families": len(train_families), "acceptance_families": len(accept_families)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
