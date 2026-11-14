from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    ("OpenAssistant/oasst2", "oasst2", "179dd21fc55192153d94adb0e0ce8f69e222bf75", ["apache-2.0"],
     "Human-reviewed conversation style; attributed assertions, not verified facts."),
    ("databricks/databricks-dolly-15k", "dolly15k", "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a", ["cc-by-sa-3.0"],
     "Instruction following; prefer document-backed categories for factual tasks."),
    ("rajpurkar/squad", "squad", "7b6d24c440a36b6815f21b70d25016731768db1f", ["cc-by-sa-4.0"],
     "Extractive QA with reference passages; original validation is evaluation-only."),
    ("Salesforce/wikitext", "wikitext103", "b08601e04326c79dfdd32d625aee71d232d685c3", ["cc-by-sa-3.0", "gfdl"],
     "General language modelling; Wikipedia text is not externally verified truth."),
]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(root=ROOT):
    from huggingface_hub import HfApi

    hf_root = root / "data" / "hf"
    wiki_root = root / "data" / "wikitext-103-raw"
    paired_root = hf_root / "oasst2" / "paired_v19"
    manifest_path = hf_root / "download_manifest.json"
    if manifest_path.exists() or wiki_root.exists() or paired_root.exists():
        raise FileExistsError("Prepared destination exists; existing output will not be overwritten.")
    if shutil.disk_usage(root).free < 9_000_000_000:
        raise OSError("Insufficient disk reserve for dataset preparation.")

    api = HfApi(token=False)
    datasets = []
    for repo_id, folder, revision, licenses, purpose in DATASETS:
        info = api.dataset_info(repo_id, revision=revision, files_metadata=True)
        remote_files = {file.rfilename: file for file in info.siblings}
        files = []
        for path in sorted((hf_root / folder).rglob("*")):
            if not path.is_file() or ".cache" in path.parts:
                continue
            remote = remote_files[path.relative_to(hf_root / folder).as_posix()]
            digest = sha256(path)
            if path.stat().st_size != remote.size:
                raise ValueError(f"Dataset size differs from the pinned revision: {path}")
            if remote.lfs:
                if digest != remote.lfs.sha256:
                    raise ValueError(f"Dataset SHA-256 differs from the pinned revision: {path}")
                verification = "pinned_hub_lfs_sha256"
            else:
                blob = path.read_bytes()
                oid = hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest()
                if oid != remote.blob_id:
                    raise ValueError(f"Dataset card differs from the pinned revision: {path}")
                verification = "pinned_hub_git_blob_sha1"
            entry = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                     "sha256": digest, "integrity_verified_against": verification}
            if path.suffix == ".parquet":
                parquet = pq.ParquetFile(path)
                entry["rows"] = sum(batch.num_rows for batch in parquet.iter_batches(batch_size=8192))
                if entry["rows"] != parquet.metadata.num_rows:
                    raise ValueError(f"Incomplete Parquet scan: {path}")
            files.append(entry)
        datasets.append({"repo_id": repo_id, "revision": revision, "licenses": licenses,
                         "license_card": f"data/hf/{folder}/README.md", "purpose": purpose, "files": files})

    wiki_root.mkdir()
    wiki_outputs = []
    for split, filename in [("train", "wiki.train.raw"), ("validation", "wiki.valid.raw"), ("test", "wiki.test.raw")]:
        sources = sorted((hf_root / "wikitext103" / "wikitext-103-raw-v1").glob(split + "-*.parquet"))
        if not sources:
            raise FileNotFoundError(f"WikiText {split} split is missing.")
        output = wiki_root / filename
        count = 0
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            for source in sources:
                for batch in pq.ParquetFile(source).iter_batches(batch_size=8192, columns=["text"]):
                    for text in batch.column(0).to_pylist():
                        if not isinstance(text, str):
                            raise ValueError("WikiText contains a non-text record.")
                        stream.write(text)
                        if not text.endswith("\n"):
                            stream.write("\n")
                        count += 1
        expected = sum(pq.ParquetFile(path).metadata.num_rows for path in sources)
        if count != expected:
            raise ValueError("WikiText row count differs after conversion.")
        wiki_outputs.append({"path": output.relative_to(root).as_posix(), "split": split, "rows": count,
                             "bytes": output.stat().st_size, "sha256": sha256(output),
                             "source_files": [path.relative_to(root).as_posix() for path in sources],
                             "transform": "Concatenate text in shard/row order; append LF only where absent; never mix splits."})

    pairs, trees, stats = {}, {}, {}
    revision = DATASETS[0][2]
    for split in ("train", "validation"):
        source = next((hf_root / "oasst2" / "data").glob(split + "-*.parquet"))
        source_hash = sha256(source)
        messages = pq.read_table(source).to_pylist()
        lookup = {message["message_id"]: message for message in messages}
        rows = []
        for row_index, message in enumerate(messages):
            if message["role"] != "assistant" or message["rank"] != 0:
                continue
            parent = lookup.get(message["parent_id"])
            if not parent or parent["role"] != "prompter" or parent["parent_id"] is not None:
                continue
            if not all(item["lang"] in ("en", "ru") and item["deleted"] is False
                       and item["synthetic"] is False and item["review_result"] is True
                       and isinstance(item["text"], str) and item["text"].strip()
                       for item in (message, parent)):
                continue
            rows.append({"prompt": parent["text"].strip(), "response": message["text"].strip(),
                         "source": f"hf:OpenAssistant/oasst2@{revision}#message:{message['message_id']}",
                         "family_id": f"oasst2:tree:{message['message_tree_id']}", "split": split,
                         "language": message["lang"], "evidence_kind": "human_reviewed_source_assertion",
                         "truth_verified_externally": False,
                         "provenance": {"repo_id": "OpenAssistant/oasst2", "revision": revision,
                                        "path": source.relative_to(root).as_posix(), "sha256": source_hash,
                                        "row_index_zero_based": row_index, "message_id": message["message_id"],
                                        "parent_message_id": parent["message_id"]}})
        pairs[split] = rows
        trees[split] = {message["message_tree_id"] for message in messages}
        stats[split] = {"source_messages": len(messages), "source_trees": len(trees[split]),
                        "synthetic_messages": sum(message["synthetic"] is True for message in messages)}

    if trees["train"] & trees["validation"]:
        raise ValueError("Original OASST2 tree splits overlap.")
    normalize = lambda value: " ".join(value.casefold().split())
    train_prompts = {normalize(row["prompt"]) for row in pairs["train"]}
    validation_prompts, filtered, removed = set(), [], 0
    for row in pairs["validation"]:
        key = normalize(row["prompt"])
        if key in train_prompts or key in validation_prompts:
            removed += 1
            continue
        validation_prompts.add(key)
        filtered.append(row)
    pairs["validation"] = filtered
    paired_root.mkdir()
    paired_outputs = []
    for split, rows in pairs.items():
        output = paired_root / f"{split}_en_ru.jsonl"
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        paired_outputs.append({"path": output.relative_to(root).as_posix(), "split": split,
                               "rows": len(rows), "languages": dict(Counter(row["language"] for row in rows)),
                               "bytes": output.stat().st_size, "sha256": sha256(output)})

    squad = {}
    for split in ("train", "validation"):
        source = next((hf_root / "squad" / "plain_text").glob(split + "-*.parquet"))
        table = pq.read_table(source, columns=["title", "id", "question"])
        squad[split] = {"titles": set(table.column("title").to_pylist()),
                        "ids": set(table.column("id").to_pylist()),
                        "prompts": {normalize(text) for text in table.column("question").to_pylist()}}
    with (hf_root / "dolly15k" / "databricks-dolly-15k.jsonl").open(encoding="utf-8") as stream:
        dolly = [json.loads(line) for line in stream if line.strip()]
    if not all(isinstance(row.get(field), str) for row in dolly
               for field in ("instruction", "context", "response", "category")):
        raise ValueError("Dolly record schema is invalid.")

    manifest = {
        "schema": "ziwago.hf_dataset_acquisition.v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "cli": {"name": "hf", "version": "2.4.0",
                "launcher": "C:/Users/yarma/AppData/Roaming/Python/Python314/Scripts/hf.exe"},
        "downloaded_bytes": sum(file["bytes"] for dataset in datasets for file in dataset["files"]),
        "datasets": datasets, "prepared_outputs": wiki_outputs + paired_outputs,
        "oasst2_selection": {
            "rule": "EN/RU, non-synthetic, not deleted, reviewed root prompter and reviewed rank-0 assistant; original tree splits preserved.",
            "source_statistics": stats, "validation_duplicate_prompts_removed": removed,
            "family_id_scope": "Conversation-tree isolation only, not proof of semantic topic independence.",
            "train_validation_tree_overlap": 0,
            "train_validation_normalized_prompt_overlap_after_filter": len(train_prompts & validation_prompts)},
        "dolly_statistics": {"rows": len(dolly), "categories": dict(Counter(row["category"] for row in dolly)),
                             "with_reference_context": sum(bool(row["context"].strip()) for row in dolly)},
        "squad_official_split_audit": {
            "article_title_overlap": len(squad["train"]["titles"] & squad["validation"]["titles"]),
            "question_id_overlap": len(squad["train"]["ids"] & squad["validation"]["ids"]),
            "normalized_question_overlap": len(squad["train"]["prompts"] & squad["validation"]["prompts"]),
            "status": "Exclude duplicate prompts before building an independent acceptance suite."},
        "restrictions": {
            "training_performed": False, "acceptance_performed": False, "calibration_fitted": False,
            "full_corpus_training_certified": False,
            "semantic_independence": "Unverified: WikiText, Dolly and SQuAD all include Wikipedia; cross-corpus document/topic contamination audit required before mixed training.",
            "validation_and_test_usage": "Evaluation-only; never use heldout answers as repair lessons or source assertions.",
            "current_trainer": "OASST train has the v19 record fields. Dolly/SQuAD require context-aware adaptation; WikiText raw requires a language-modelling phase. No production acceptance suite created.",
            "license_obligations": "Preserve attribution and comply with Apache, CC BY-SA and GFDL terms; downloaded cards retain publisher notices."},
        "disk_free_bytes_after_preparation": shutil.disk_usage(root).free,
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return {"manifest": str(manifest_path), "downloaded_bytes": manifest["downloaded_bytes"],
            "prepared_outputs": manifest["prepared_outputs"], "oasst2_selection": manifest["oasst2_selection"],
            "disk_free_bytes": shutil.disk_usage(root).free}


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
