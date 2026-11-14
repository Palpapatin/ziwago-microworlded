#!/usr/bin/env python3
"""Сборка объединённого скоупа обучения: v2 (фикстуры) + обученная часть v3.

Зачем
----
Линейка чекпоинтов v19 обучалась по очереди на двух куррикулах: ранние
кандидаты — на `language_v19_v2` (24 строки), поздние (bankfixed) — на
`language_v19_v3` (первые 150 строк). Порядок строк у этих куррикул РАЗНЫЙ, а
заранее зарегистрированные фикстуры compose-зонда привязаны к индексам v2.

Гиппокампальный реплей должен пройти по тем вопросам, эпизоды которых реально
есть в памяти модели, — то есть по объединению двух скоупов. Этот скрипт
собирает ровно такое объединение:

  * сначала ВСЕ строки v2 — чтобы индексы предзарегистрированных фикстур
    остались теми же и замер был сравним с предыдущим;
  * затем строки v3, которые ещё не встречались, в пределах реально обученного
    диапазона v3 (первые 150).

Ничего не генерируется, ничего не переписывается: это объединение двух
существующих куррикул с явной provenance и SHA-привязкой в манифесте.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HF = Path(r"D:\Ziwago18-storage\data\hf\squad")
V2 = HF / "language_v19_v2"
V3 = HF / "language_v19_v3"
OUT = HF / "language_v19_v6_union"
TRAINED_V3_RANGE = 150


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    rows_v2 = load_rows(V2 / "train.jsonl")
    rows_v3 = load_rows(V3 / "train.jsonl")
    seen = {row["prompt"] for row in rows_v2}
    merged = list(rows_v2)
    added = 0
    for row in rows_v3[:TRAINED_V3_RANGE]:
        if row["prompt"] in seen:
            continue
        seen.add(row["prompt"])
        merged.append(row)
        added += 1
    OUT.mkdir(parents=True, exist_ok=True)
    train_path = OUT / "train.jsonl"
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged)
    train_path.write_text(payload, encoding="utf-8")
    train_sha = sha256_file(train_path)
    families = {row.get("family_id") for row in merged}
    manifest = {
        "schema": "ziwago.language_curriculum.v1",
        "license": "cc-by-sa-4.0",
        "train_records": len(merged),
        "train_documents": len({row.get("family_id") for row in merged}),
        "acceptance_records": 0,
        "acceptance_documents": 0,
        "composition": {
            "rule": "language_v19_v2 целиком, затем непересекающиеся строки "
                    f"language_v19_v3 в пределах индексов 0..{TRAINED_V3_RANGE - 1}",
            "v2_records": len(rows_v2),
            "v3_records_added": added,
            "v3_index_range": [0, TRAINED_V3_RANGE],
            "duplicate_prompts_skipped": len(rows_v3[:TRAINED_V3_RANGE]) - added,
        },
        "source_sha256": {
            "language_v19_v2_train": sha256_file(V2 / "train.jsonl"),
            "language_v19_v3_train": sha256_file(V3 / "train.jsonl"),
            "language_v19_v2_manifest": sha256_file(V2 / "manifest.json"),
            "language_v19_v3_manifest": sha256_file(V3 / "manifest.json"),
        },
        "cross_curriculum_family_overlap": len(
            {row.get("family_id") for row in rows_v2}
            & {row.get("family_id") for row in rows_v3[:TRAINED_V3_RANGE]}),
        "selection_policy": "Объединение двух ранее существовавших куррикул без "
                            "генерации и без изменения формулировок. Порядок: v2 первыми, "
                            "чтобы предзарегистрированные фикстуры сохранили индексы.",
        "training_performed": False,
        "eligible_for_future_training": True,
        "prepared_files": {str(train_path): train_sha},
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_dir": str(OUT), "train_records": len(merged),
        "v2_records": len(rows_v2), "v3_records_added": added,
        "distinct_families": len(families), "train_sha256": train_sha,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
