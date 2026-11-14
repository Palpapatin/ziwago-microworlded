#!/usr/bin/env python3
"""Приёмка в формате v19-тренера из готовой корпусной приёмки.

Зачем: у куррикул v5b строки и train, и acceptance имеют единый формат
(`prompt`/`response`/`family_id`), а v19-тренер ждёт приёмочные случаи вида
`prompt` + `expected_concepts` + `reference`. Кроме того, приёмка v3 собиралась
БЕЗ бюджета ответа, который применяется к обучению, и потому её эталоны
выходили за пределы выученного словаря — измеримость падала в ноль
("Reference exceeds the learned vocabulary or positional window").

Здесь приёмка приводится к каноническому формату БЕЗ изменения содержания:
ожидаемым концептом становится сам ответ (он же эталон), а бюджет ответа
проверяется явно — строки вне бюджета отбрасываются, а не подгоняются.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HF = Path(r"D:\Ziwago18-storage\data\hf\squad")
SRC = HF / "language_v19_v5b"
OUT = HF / "language_v19_v8_v19format"
MAX_RESPONSE_TOKENS = 63


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    rows = [json.loads(line) for line in (SRC / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    cases_raw = json.loads((SRC / "acceptance.json").read_text(encoding="utf-8"))

    def normalized(text: str) -> str:
        return " ".join(text.casefold().split())

    train_prompts = {normalized(r["prompt"]) for r in rows}
    cases, dropped, leaked = [], [], []
    for case in cases_raw:
        tokens = int(case.get("response_bpe_tokens") or 0)
        if not 2 <= tokens <= MAX_RESPONSE_TOKENS:
            dropped.append({"family_id": case.get("family_id"), "response_bpe_tokens": tokens})
            continue
        if normalized(case["prompt"]) in train_prompts:
            # Промпт приёмки встречается в обучении — такой случай непригоден:
            # модель видела вопрос. Отсекаем, а не подгоняем.
            leaked.append({"family_id": case.get("family_id"), "prompt": case["prompt"][:120]})
            continue
        cases.append({
            "prompt": case["prompt"],
            "expected_concepts": [case["response"]],
            "family_id": case["family_id"],
            "reference": case["response"],
            "split": "acceptance",
            "reference_context_not_provided_to_model": True,
            "provenance": case.get("provenance", {}),
        })
    if not cases:
        raise SystemExit("No acceptance case survived the response budget.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (OUT / "acceptance.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    train_families = {r["family_id"] for r in rows}
    accept_families = {c["family_id"] for c in cases}
    if train_families & accept_families:
        raise SystemExit("Train/acceptance family overlap.")
    manifest = {
        "schema": "ziwago.language_curriculum.v1",
        "license": "mixed: see per-source README",
        "train_records": len(rows),
        "acceptance_records": len(cases),
        "train_documents": len(train_families),
        "acceptance_documents": len(accept_families),
        "cross_split_family_overlap": 0,
        "acceptance_dropped_by_response_budget": len(dropped),
        "acceptance_dropped_examples": dropped[:10],
        "acceptance_dropped_for_prompt_leak": len(leaked),
        "acceptance_leak_examples": leaked[:10],
        "selection_policy": "Обучение и приёмка из одного корпуса (language_v19_v5b), "
                            "приёмка приведена к формату v19 и отфильтрована тем же "
                            f"бюджетом ответа (≤{MAX_RESPONSE_TOKENS} токенов), что и обучение.",
        "training_performed": False,
        "eligible_for_future_training": True,
        "source_sha256": {"v5b_train": sha256_file(SRC / "train.jsonl"),
                          "v5b_acceptance": sha256_file(SRC / "acceptance.json")},
        "prepared_files": {},
    }
    manifest["prepared_files"][str(OUT / "train.jsonl")] = sha256_file(OUT / "train.jsonl")
    manifest["prepared_files"][str(OUT / "acceptance.json")] = sha256_file(OUT / "acceptance.json")
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(OUT), "train": len(rows), "acceptance": len(cases),
                      "dropped_by_budget": len(dropped)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
