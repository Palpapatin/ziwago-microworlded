from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ziwago.evaluation.gates import evaluate_grammar_and_artifacts
from train_ziwago_v9_gpt2vocab import _init_bpe_worker, _GPT2_PAT, _bpe_encode_piece
import train_ziwago_v9_gpt2vocab as tokenizer


REVISION = "7b6d24c440a36b6815f21b70d25016731768db1f"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized(text):
    return " ".join(text.lower().split())


def source_sentence(record):
    context = record["context"]
    answers = record["answers"]
    if not answers["text"]:
        return None
    answer = answers["text"][0]
    start = int(answers["answer_start"][0])
    if context[start:start + len(answer)] != answer:
        raise ValueError("SQuAD answer offset does not match its context.")
    sentence_start = 0
    for match in re.finditer(r'[.!?](?!\d)(?:["\u201d])?(?=\s|$)', context):
        preceding = context[sentence_start:match.start()].split()
        last_word = preceding[-1] if preceding else ""
        if match.group().startswith(".") and (last_word in {"Dr", "Mr", "Mrs", "Ms", "Prof", "St", "Jr", "Sr", "vs", "e.g", "i.e", "U.S", "U.K"}
                                              or len(last_word) == 1 and last_word.isupper()):
            continue
        if sentence_start <= start and start + len(answer) <= match.end():
            raw = context[sentence_start:match.end()]
            sentence = raw.strip()
            offset = sentence_start + len(raw) - len(raw.lstrip())
            if not (6 <= len(sentence.split()) <= 22):
                return None
            if not evaluate_grammar_and_artifacts(sentence, min_words=3)["grammar_coherent"]:
                return None
            ids = []
            for piece in _GPT2_PAT.findall(" " + sentence):
                ids.extend(_bpe_encode_piece(piece, tokenizer._WORKER_VOCAB, tokenizer._WORKER_MERGES, tokenizer._WORKER_CACHE))
            if not 2 <= len(ids) <= 64:
                return None
            return sentence, answer, offset, offset + len(sentence), len(ids)
        sentence_start = match.end()
    return None


def prepare(root=ROOT, max_train=24, max_acceptance=6):
    destination = root / "data" / "hf" / "squad" / "language_v19_v2"
    if destination.exists():
        raise FileExistsError("Prepared curriculum exists; it will not be overwritten.")
    vocab = root / "data" / "gpt2" / "vocab.json"
    merges = root / "data" / "gpt2" / "merges.txt"
    _init_bpe_worker(str(vocab), str(merges))
    original = root / "data" / "hf" / "squad" / "plain_text"
    source_paths = {split: original / f"{split}-00000-of-00001.parquet" for split in ("train", "validation")}
    hashes = {split: digest(path) for split, path in source_paths.items()}
    train_questions = set()
    train_titles = set()
    train = []
    title_counts = {}
    selected_sentences = set()
    for batch in pq.ParquetFile(source_paths["train"]).iter_batches(batch_size=1024):
        for record in batch.to_pylist():
            train_questions.add(normalized(record["question"]))
            train_titles.add(record["title"])
            if len(train) >= max_train or title_counts.get(record["title"], 0) >= 2:
                continue
            selected = source_sentence(record)
            if selected is None:
                continue
            sentence, answer, start, end, token_count = selected
            if (record["title"], sentence) in selected_sentences:
                continue
            selected_sentences.add((record["title"], sentence))
            train.append({"prompt": record["question"], "response": sentence,
                          "source": "rajpurkar/squad:train", "family_id": f'squad:article:{record["title"]}',
                          "split": "train", "evidence_kind": "document_span", "truth_verified_externally": False,
                          "provenance": {"repository": "rajpurkar/squad", "revision": REVISION,
                                         "path": str(source_paths["train"].resolve()), "sha256": hashes["train"],
                                         "question_id": record["id"], "title": record["title"],
                                         "context_sha256": hashlib.sha256(record["context"].encode("utf-8")).hexdigest(),
                                         "span_start": start, "span_end": end}, "response_bpe_tokens": token_count})
            title_counts[record["title"]] = title_counts.get(record["title"], 0) + 1
    cases = []
    acceptance_titles = set()
    skipped_duplicates = 0
    for batch in pq.ParquetFile(source_paths["validation"]).iter_batches(batch_size=1024):
        for record in batch.to_pylist():
            if normalized(record["question"]) in train_questions:
                skipped_duplicates += 1
                continue
            if record["title"] in train_titles:
                raise ValueError("Official SQuAD article split overlap.")
            if len(cases) >= max_acceptance or record["title"] in acceptance_titles:
                continue
            selected = source_sentence(record)
            if selected is None:
                continue
            sentence, answer, start, end, _ = selected
            cases.append({"prompt": record["question"], "expected_concepts": [answer],
                          "family_id": f'squad:article:{record["title"]}', "reference": sentence,
                          "split": "acceptance", "reference_context_not_provided_to_model": True,
                          "provenance": {"repository": "rajpurkar/squad", "revision": REVISION,
                                         "path": str(source_paths["validation"].resolve()), "sha256": hashes["validation"],
                                         "question_id": record["id"], "title": record["title"],
                                         "span_start": start, "span_end": end}})
            acceptance_titles.add(record["title"])
    if len(train) != max_train or len(cases) != max_acceptance:
        raise ValueError("Insufficient bounded source sentences for the requested experiment.")
    destination.mkdir()
    train_path = destination / "train.jsonl"
    acceptance_path = destination / "acceptance.json"
    with train_path.open("x", encoding="utf-8") as stream:
        for record in train:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    with acceptance_path.open("x", encoding="utf-8") as stream:
        json.dump(cases, stream, ensure_ascii=False, indent=2)
    manifest = {"schema": "ziwago.language_curriculum.v1", "license": "cc-by-sa-4.0",
                "train_records": len(train), "train_documents": len(title_counts),
                "acceptance_records": len(cases), "acceptance_documents": len(acceptance_titles),
                "all_official_train_validation_duplicate_questions_excluded": skipped_duplicates,
                "cross_split_article_overlap": 0, "cross_split_normalized_question_overlap": 0,
                "semantic_topic_independence_certified": False, "cross_corpus_contamination_audited": False,
                "evaluation_policy": "Frozen closed-book probe; reference contexts and answers never become model sources or lessons. A refusal is safe but not QA success.",
                "selection_policy": "At most two complete short source sentences per training article, one per acceptance article; no generated answers or mid-sentence truncation.",
                "training_performed": False, "source_sha256": hashes,
                "prepared_files": {str(p.resolve()): digest(p) for p in (train_path, acceptance_path)}}
    with (destination / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return {"destination": str(destination.resolve()), **manifest}


def prepare_openbook(root=ROOT):
    root = Path(root)
    original = root / "data/hf/squad/language_v19_v2"
    destination = original.parent / "openbook_v19"
    if destination.exists():
        raise FileExistsError("Prepared open-book acceptance exists; it will not be overwritten.")
    manifest = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("eligible_for_future_training") is False:
        raise ValueError("The original curriculum is invalid.")
    for name in ("train.jsonl", "acceptance.json"):
        path = original / name
        if manifest["prepared_files"].get(str(path.resolve())) != digest(path):
            raise ValueError("Original curriculum file hash mismatch.")
    train = [json.loads(line) for line in (original / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    original_cases = json.loads((original / "acceptance.json").read_text(encoding="utf-8"))
    source_paths = {split: root / f"data/hf/squad/plain_text/{split}-00000-of-00001.parquet"
                    for split in ("train", "validation")}
    hashes = {split: digest(path) for split, path in source_paths.items()}
    if hashes != manifest["source_sha256"]:
        raise ValueError("Pinned SQuAD source hashes mismatch.")
    train_questions, train_titles = set(), set()
    for batch in pq.ParquetFile(source_paths["train"]).iter_batches(batch_size=1024):
        for record in batch.to_pylist():
            train_questions.add(normalized(record["question"]))
            train_titles.add(record["title"])
    wanted = {case["provenance"]["question_id"] for case in original_cases}
    found, row_number, duplicates = {}, 0, 0
    for batch in pq.ParquetFile(source_paths["validation"]).iter_batches(batch_size=1024):
        for record in batch.to_pylist():
            row_number += 1
            if normalized(record["question"]) in train_questions:
                duplicates += 1
                continue
            if record["title"] in train_titles:
                raise ValueError("Official SQuAD article split overlap.")
            if record["id"] in wanted:
                if record["id"] in found:
                    raise ValueError("Duplicate validation question ID.")
                found[record["id"]] = record, row_number
    if len(wanted) != len(original_cases) or set(found) != wanted:
        raise ValueError("Original acceptance questions missing or not independent.")
    cases = []
    for case in original_cases:
        provenance = case["provenance"]
        record, row_number = found[provenance["question_id"]]
        document = record["context"]
        start, end = provenance["span_start"], provenance["span_end"]
        if (provenance["revision"] != REVISION or provenance["sha256"] != hashes["validation"]
                or record["question"] != case["prompt"] or record["title"] != provenance["title"]
                or document[start:end] != case["reference"]
                or not all(concept in document for concept in case["expected_concepts"])):
            raise ValueError("Original acceptance evidence mismatch.")
        cases.append({**case, "document": document, "evaluation_mode": "open_book",
                      "reference_context_not_provided_to_model": False, "context_is_ephemeral": True,
                      "provenance": {**provenance, "validation_row": row_number,
                                     "context_sha256": hashlib.sha256(document.encode("utf-8")).hexdigest()}})
    if {item["family_id"] for item in train} & {case["family_id"] for case in cases}:
        raise ValueError("Train/acceptance article overlap.")
    destination.mkdir()
    acceptance_path = destination / "acceptance.json"
    with acceptance_path.open("x", encoding="utf-8") as stream:
        json.dump(cases, stream, ensure_ascii=False, indent=2)
    result = {"schema": "ziwago.openbook_acceptance.v1", "license": manifest["license"],
              "evaluation_mode": "open_book", "certification_scope": "extractive_document_qa",
              "acceptance_records": len(cases), "eligible_for_future_training": False,
              "previously_evaluated_questions": True, "original_questions_unchanged": True,
              "original_train_path": str((original / "train.jsonl").resolve()),
              "source_sha256": hashes, "original_prepared_files": manifest["prepared_files"],
              "all_official_train_validation_duplicate_questions_excluded": duplicates,
              "cross_split_article_overlap": 0, "cross_split_normalized_question_overlap": 0,
              "semantic_topic_independence_certified": False, "cross_corpus_contamination_audited": False,
              "training_performed": False,
              "evaluation_policy": "Full validation passages are ephemeral QA inputs, not lessons or persistent sources. References and expected concepts are evaluator-only. This reuses the previously examined questions, not a fresh production benchmark.",
              "prepared_files": {str(acceptance_path.resolve()): digest(acceptance_path)}}
    with (destination / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return {"destination": str(destination.resolve()), **result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--open-book", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_openbook(args.root) if args.open_book else prepare(args.root), ensure_ascii=False, indent=2))
