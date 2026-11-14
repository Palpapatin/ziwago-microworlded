"""Download pinned research corpora to D: and verify immutable source hashes."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("D:/Ziwago18-storage/data/hf")
ENVIRONMENT = {
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "HF_HOME": str(ROOT / ".additional_hf_cache"),
    "HF_HUB_CACHE": str(ROOT / ".additional_hf_cache/hub"),
    "HF_XET_CACHE": str(ROOT / ".additional_hf_cache/xet"),
}
os.environ.update(ENVIRONMENT)

import pyarrow.parquet as pq
from huggingface_hub import HfApi
from huggingface_hub import __version__ as hub_version

CLI = Path("C:/Users/yarma/AppData/Roaming/Python/Python314/Scripts/hf.exe")
MANIFEST = ROOT / "additional_download_manifest.json"
IDENTITY = "ziwago19-pinned-additional-language-research-v1"


def corpus(key, repo, revision, directory, license_name, caveats, files, **extra):
    return {
        "key": key,
        "repo_id": repo,
        "revision": revision,
        "directory": directory,
        "license": license_name,
        "caveats": caveats,
        "selection": [{"repo_file": name, "split": split} for name, split in files],
        **extra,
    }


CORPORA = [
    corpus(
        "tinystories", "roneneldan/TinyStories",
        "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64", "tinystories",
        "CDLA-Sharing-1.0",
        ["Synthetic GPT-4 stories; not verified facts.",
         "Only V2 GPT4 train/valid: original duplicate versions excluded."],
        [("TinyStoriesV2-GPT4-train.txt", "train"),
         ("TinyStoriesV2-GPT4-valid.txt", "validation")],
    ),
    corpus(
        "daily_dialog", "li2017dailydialog/daily_dialog",
        "469b7118f52336674f1d78693797e9fcc7e47d2a", "daily_dialog",
        "CC-BY-NC-SA-4.0",
        ["Research/noncommercial only; attribution and share-alike obligations.",
         "Pinned Parquet conversion; no card available at this revision.",
         "Publisher main-card license verified separately; conversion metadata has no license field."],
        [("default/train/0000.parquet", "train"),
         ("default/validation/0000.parquet", "validation"),
         ("default/test/0000.parquet", "test")],
        license_evidence={"repo_id": "li2017dailydialog/daily_dialog",
                          "revision": "ffde34acefcd956529c39c4fc78d993b5b7f7520",
                          "card_git_blob": "f39810cbfc27426c6451fc5184a07d516e433259",
                          "publisher_card_license": ["cc-by-nc-sa-4.0"]},
    ),
    corpus(
        "personamem_v1", "bowen-upenn/PersonaMem-v1",
        "a8076d5608c93ba2a28983cd78aa99b01a163ae7", "personamem_v1_benchmark",
        "MIT",
        ["Benchmark-only; not authorized as training material.",
         "128k shared contexts and questions only; benchmark must remain held out."],
        [("shared_contexts_128k.jsonl", "benchmark_contexts"),
         ("questions_128k.csv", "benchmark_questions")],
        training_eligible=False,
    ),
    corpus(
        "npc_dialogue_v2", "chimbiwide/NPC-Dialogue_v2",
        "098599c78366d56e15ba49db0b0484fc0931f302", "npc_dialogue_v2",
        "Apache-2.0",
        ["Creator-published Gemini-synthetic fictional dialogue; not verified facts."],
        [("npc_dialogue.jsonl", "unsplit")],
    ),
    corpus(
        "ultrachat200k_train_sft", "HuggingFaceH4/ultrachat_200k",
        "8049631c405ae6576f93f445c6b8166f76f5505a", "ultrachat200k/train_sft",
        "MIT",
        ["Synthetic dialogue; not verified facts.",
         "Only three train_sft shards; no train_gen or test files in this root."],
        [("data/train_sft-00000-of-00003-a3ecf92756993583.parquet", "train_sft"),
         ("data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet", "train_sft"),
         ("data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet", "train_sft")],
    ),
    corpus(
        "ultrachat200k_eval", "HuggingFaceH4/ultrachat_200k",
        "8049631c405ae6576f93f445c6b8166f76f5505a", "ultrachat200k_eval",
        "MIT",
        ["Synthetic held-out test_sft, not training material.",
         "Separate root outside ultrachat200k so recursive training readers cannot consume it."],
        [("data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet", "test_sft")],
        training_eligible=False,
    ),
    corpus(
        "openwebtext_subset", "Skylion007/openwebtext",
        "79d93d786212f7344586290adb811d4ae6a1762c", "openwebtext_subset",
        "CC0-1.0 (packaging)",
        ["subset4of80: ONLY the first four of eighty Parquet shards.",
         "CC0 packaging does not waive underlying source website rights.",
         "No official evaluation split; no full-corpus or preflight completion claim."],
        [(f"plain_text/train-{i:05d}-of-00080.parquet", "train_subset4of80")
         for i in range(4)],
        subset="subset4of80", full_corpus=False, official_eval=False,
    ),
]
PLAN_SHA256 = hashlib.sha256(
    json.dumps(CORPORA, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def now():
    return datetime.now(UTC).isoformat()


def inspect_parent(path):
    """Use ls before creating anything, including nested download/cache parents."""
    ls = shutil.which("ls")
    if not ls:
        raise RuntimeError("Git Bash ls is required for parent-directory checks")
    subprocess.run([ls, "-d", path.as_posix()], check=True)
    if not path.is_dir() or path.is_symlink() or path.is_junction():
        raise RuntimeError(f"Unsafe/non-directory parent: {path}")
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError(f"Parent escapes authorized storage: {path}")


def ensure_directory(path):
    if path.exists():
        if not path.is_dir() or path.is_symlink() or path.is_junction():
            raise RuntimeError(f"Unsafe directory: {path}")
        return
    ensure_directory(path.parent)
    inspect_parent(path.parent)
    path.mkdir()


def check_regular(path):
    if path.is_symlink() or path.is_junction() or not path.is_file():
        raise RuntimeError(f"Refusing non-regular file: {path}")


def file_hashes(path):
    check_regular(path)
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            git_blob.update(chunk)
    return size, sha256.hexdigest(), git_blob.hexdigest()


def verify_integrity(path, entry):
    size, digest, git_blob = file_hashes(path)
    entry.update({"observed_size": size, "observed_sha256": digest,
                  "observed_content_git_blob_sha1": git_blob})
    if size != entry["expected_size"]:
        entry["status"] = "integrity_failed"
        raise RuntimeError(f"Refusing size mismatch: {path}: {size} != {entry['expected_size']}")
    if entry["lfs_sha256"]:
        if digest != entry["lfs_sha256"]:
            entry["status"] = "integrity_failed"
            raise RuntimeError(f"Refusing LFS SHA256 mismatch: {path}")
        method = "pinned_HfApi_LFS_SHA256_and_size"
    else:
        if git_blob != entry["git_blob_id"]:
            entry["status"] = "integrity_failed"
            raise RuntimeError(f"Refusing pinned Git blob mismatch: {path}")
        method = "pinned_HfApi_native_Git_blob_SHA1_and_size_plus_local_SHA256"
    return {"actual_size": size, "sha256": digest,
            "content_git_blob_sha1": git_blob, "integrity_method": method}


def validate_content(path):
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        rows = 0
        batches = 0
        for batch in parquet.iter_batches(batch_size=256, use_threads=False):
            batch.validate(full=True)
            decoded = batch.to_pylist()  # Materialize every column/value, not just footer counts.
            for row in decoded:
                if not isinstance(row, dict) or len(row) != len(batch.schema):
                    raise ValueError(f"Invalid decoded Parquet row in {path}")
                rows += 1
            batches += 1
        if rows != parquet.metadata.num_rows:
            raise ValueError(f"Parquet row count mismatch: {path}")
        return {"format": "parquet", "rows": rows, "metadata_rows": parquet.metadata.num_rows,
                "row_groups": parquet.metadata.num_row_groups, "batches": batches,
                "schema": str(parquet.schema_arrow), "all_columns_decoded": True,
                "method": "all_batches_validate_full_and_every_row_to_pylist"}
    if path.suffix == ".jsonl":
        rows = 0
        blanks = 0
        with path.open("r", encoding="utf-8-sig", errors="strict") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    blanks += 1
                    continue
                value = json.loads(line, parse_constant=lambda x, number=number: (_ for _ in ()).throw(
                    ValueError(f"Non-finite JSON constant {x} at line {number}")))
                if not isinstance(value, dict):
                    raise TypeError(f"JSONL row is not an object: {path}:{number}")
                rows += 1
        return {"format": "jsonl", "rows": rows, "blank_lines": blanks,
                "method": "stream_parse_every_nonblank_line_strict_utf8_and_json_object"}
    if path.suffix == ".csv":
        csv.field_size_limit(64 * 1024 * 1024)
        rows = 0
        with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as source:
            reader = csv.reader(source, strict=True)
            header = next(reader)
            if not header or len(set(header)) != len(header):
                raise ValueError(f"Invalid CSV header: {path}")
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(f"CSV column count mismatch: {path}:{reader.line_num}")
                rows += 1
        return {"format": "csv", "rows": rows, "columns": header,
                "method": "stream_csv_strict_all_rows_and_column_counts"}
    if path.suffix == ".txt":
        stories = 0
        empty_records = 0
        empty_record_lines = []
        lines = 0
        delimiters = 0
        has_text = False
        with path.open("r", encoding="utf-8", errors="strict") as source:
            for line in source:
                lines += 1
                pieces = line.split("<|endoftext|>")
                for i, piece in enumerate(pieces):
                    has_text = has_text or bool(piece.strip())
                    if i < len(pieces) - 1:
                        delimiters += 1
                        if not has_text:
                            empty_records += 1
                            if len(empty_record_lines) < 20:
                                empty_record_lines.append(lines)
                        stories += 1  # Count every upstream framed record, including empty ones.
                        has_text = False
        if has_text:
            stories += 1
        return {"format": "tinystories_text", "rows": stories, "lines": lines,
                "nonempty_stories": stories - empty_records, "empty_story_records": empty_records,
                "first_empty_record_line_numbers": empty_record_lines,
                "delimiter_count": delimiters, "trailing_unterminated_story": has_text,
                "upstream_records_preserved_unmodified": True,
                "caveat": "Empty upstream framed records are explicitly counted; no filtering, rewriting, or training performed.",
                "method": "stream_strict_utf8_all_lines_and_all_endoftext_framed_records_including_empty"}
    if path.name.lower() == "readme.md":
        with path.open("r", encoding="utf-8", errors="strict") as source:
            for _ in source:
                pass
        return {"format": "dataset_card", "rows": None,
                "method": "strict_utf8_entire_card_plus_pinned_Git_blob"}
    raise ValueError(f"Unapproved file type: {path}")


def cache_paths(directory, entry):
    metadata = directory / ".cache/huggingface/download" / (entry["repo_file"] + ".metadata")
    short = base64.urlsafe_b64encode(hashlib.sha1(metadata.name.encode()).digest()).decode()
    etag = entry["lfs_sha256"] or entry["git_blob_id"]
    return metadata, short, etag


def inspect_destination(directory, entries):
    allowed_data = {entry["repo_file"] for entry in entries}
    cache = directory / ".cache/huggingface"
    allowed_cache = {".gitignore", "CACHEDIR.TAG", ".gitignore.lock"}
    incomplete_patterns = []
    for entry in entries:
        metadata, short, etag = cache_paths(directory, entry)
        allowed_cache.update({metadata.relative_to(cache).as_posix(),
                              metadata.with_suffix(".lock").relative_to(cache).as_posix()})
        incomplete_patterns.append((metadata.parent, re.compile(
            rf"{re.escape(short)}\.{re.escape(etag)}(?:\.[0-9a-f]{{8}})?\.incomplete"), entry))
        if metadata.exists():
            check_regular(metadata)
            fields = metadata.read_text(encoding="utf-8").splitlines()
            if len(fields) != 3 or fields[:2] != [entry["revision"], etag]:
                raise RuntimeError(f"Refusing mismatched preexisting HF metadata: {metadata}")
            float(fields[2])
    partials = []
    if directory.exists():
        for path in directory.rglob("*"):
            if path.is_symlink() or path.is_junction():
                raise RuntimeError(f"Refusing linked destination entry: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(directory).as_posix()
            if relative in allowed_data:
                continue
            if path.is_relative_to(cache):
                cache_relative = path.relative_to(cache).as_posix()
                if cache_relative in allowed_cache:
                    continue
                # Permit only the exact revision's metadata tree; verify selected entries.
                if cache_relative == f"trees/{entries[0]['revision']}.json":
                    tree = json.loads(path.read_text(encoding="utf-8"))
                    if tree.get("format_version") != 1:
                        raise RuntimeError(f"Unknown HF tree cache format: {path}")
                    for entry in entries:
                        item = tree.get("files", {}).get(entry["repo_file"])
                        if item is None:
                            continue  # Some downloads cache a partial listing.
                        if (item.get("size") != entry["expected_size"]
                                or item.get("blob_id") != entry["git_blob_id"]
                                or item.get("lfs_sha256") != entry["lfs_sha256"]):
                            raise RuntimeError(f"Mismatched pinned HF tree metadata: {path}")
                    continue
                for parent, pattern, entry in incomplete_patterns:
                    if path.parent == parent and pattern.fullmatch(path.name):
                        size = path.stat().st_size
                        if size > entry["expected_size"]:
                            raise RuntimeError(f"Oversized pinned temporary file: {path}")
                        partials.append({"path": str(path), "bytes": size,
                                         "repo_file": entry["repo_file"],
                                         "status": "known_pinned_partial_not_byte_resumed_by_hf_2_4"})
                        break
                else:
                    raise RuntimeError(f"Refusing unknown preexisting cache file: {path}")
                continue
            raise RuntimeError(f"Refusing unplanned preexisting destination file: {path}")
    return partials


def update_summary(manifest):
    actual_bytes = 0
    present_bytes = 0
    payload_rows = 0
    verified_files = 0
    for record in manifest["corpora"]:
        splits = {}
        for entry in record.get("files", []):
            present_bytes += entry.get("observed_size", entry.get("actual_size", 0))
            if entry.get("status") == "verified":
                actual_bytes += entry.get("actual_size", 0)
                verified_files += 1
                if entry["split"] != "card":
                    payload_rows += entry["validation"]["rows"]
            if entry["split"] != "card":
                splits.setdefault(entry["split"], []).append(entry)
        record["splits"] = {}
        for name, entries in splits.items():
            complete = all(e.get("status") == "verified" for e in entries)
            inventory = [{"repo_file": e["repo_file"], "size": e.get("actual_size"),
                          "sha256": e.get("sha256"), "rows": e.get("validation", {}).get("rows")}
                         for e in sorted(entries, key=lambda e: e["repo_file"])]
            record["splits"][name] = {
                "status": "verified" if complete else "incomplete",
                "bytes": sum(e.get("actual_size", 0) for e in entries),
                "rows": sum((e.get("validation", {}).get("rows") or 0) for e in entries),
                "sha256_inventory": hashlib.sha256(json.dumps(
                    inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest() if complete else None,
                "inventory": inventory,
            }
    manifest["summary"] = {"actual_verified_bytes_including_cards": actual_bytes,
                           "actual_present_bytes_including_cards": present_bytes,
                           "payload_records_across_distinct_formats": payload_rows,
                           "verified_files": verified_files,
                           "planned_files": sum(len(c.get("files", [])) for c in manifest["corpora"])}


def save_manifest(manifest):
    update_summary(manifest)
    manifest["updated_at"] = now()
    inspect_parent(MANIFEST.parent)
    if MANIFEST.exists():
        check_regular(MANIFEST)
        previous = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if previous.get("identity") != IDENTITY or previous.get("plan_sha256") != PLAN_SHA256:
            raise RuntimeError("Refusing to overwrite an unrelated additional manifest")
    temporary = MANIFEST.with_suffix(".json.tmp")
    if temporary.exists():
        raise RuntimeError(f"Refusing to overwrite preexisting manifest temporary: {temporary}")
    with temporary.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(manifest, output, indent=2, ensure_ascii=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, MANIFEST)


def prepare_metadata(manifest):
    api = HfApi(token=False)
    seen = {}
    for spec, record in zip(CORPORA, manifest["corpora"]):
        key = (spec["repo_id"], spec["revision"])
        if key not in seen:
            seen[key] = api.repo_info(repo_id=key[0], revision=key[1],
                                      repo_type="dataset", files_metadata=True)
        info = seen[key]
        if info.sha != spec["revision"]:
            raise RuntimeError(f"Pinned revision did not resolve exactly: {spec['repo_id']}")
        available = {s.rfilename: s for s in info.siblings}
        card_names = [name for name in available if name.lower() == "readme.md"]
        if not card_names and spec["key"] != "daily_dialog":
            raise RuntimeError(f"Expected card missing: {spec['repo_id']}")
        selection = spec["selection"] + [{"repo_file": name, "split": "card"} for name in card_names]
        record["resolved_revision"] = info.sha
        record["hub_card_license"] = info.card_data.to_dict().get("license") if info.card_data else None
        record["card_status"] = "selected_pinned_card" if card_names else "unavailable_at_pinned_revision"
        record["files"] = []
        for selected in selection:
            sibling = available.get(selected["repo_file"])
            if sibling is None or sibling.size is None or not sibling.blob_id:
                raise RuntimeError(f"Missing pinned size/blob metadata: {selected['repo_file']}")
            if sibling.lfs and (sibling.lfs.size != sibling.size or not sibling.lfs.sha256):
                raise RuntimeError(f"Inconsistent LFS metadata: {selected['repo_file']}")
            record["files"].append({
                **selected, "revision": spec["revision"],
                "path": str(ROOT / spec["directory"] / selected["repo_file"]),
                "expected_size": sibling.size, "git_blob_id": sibling.blob_id,
                "lfs_sha256": sibling.lfs.sha256 if sibling.lfs else None,
                "metadata_storage": "lfs" if sibling.lfs else "native_git_blob",
                "status": "pending",
            })
        record["status"] = "metadata_verified"


def run_group(record, timeout, verify_only):
    directory = Path(record["local_directory"])
    entries = record["files"]
    record["preexisting_pinned_partials"] = inspect_destination(directory, entries)
    missing = []
    # Validate ALL existing planned destinations before the CLI can write anything.
    for entry in entries:
        path = Path(entry["path"])
        if path.exists():
            try:
                entry.update(verify_integrity(path, entry))
                entry["status"] = "integrity_verified"
                entry["validation"] = validate_content(path)
                entry["status"] = "verified"
                entry["acquisition"] = "reused_exact_pinned_verified_destination"
            except Exception as exc:  # noqa: BLE001 - retain failed verification without replacing files
                if entry["status"] != "integrity_failed":
                    entry["status"] = "validation_failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
        else:
            entry["status"] = "incomplete_missing"
            missing.append(entry)
    if any(e["status"] in ("integrity_failed", "validation_failed") for e in entries):
        record["status"] = "failed"
        return
    if missing and not verify_only:
        ensure_directory(directory)
        for entry in missing:
            ensure_directory(Path(entry["path"]).parent)
            ensure_directory(cache_paths(directory, entry)[0].parent)
        ensure_directory(directory / ".cache/huggingface/trees")
        for name in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE"):
            ensure_directory(Path(ENVIRONMENT[name]))
            inspect_parent(Path(ENVIRONMENT[name]))
        # Check cache parent before HF creates its bookkeeping files.
        inspect_parent(directory / ".cache/huggingface")
        command = [str(CLI), "download", record["repo_id"],
                   *[e["repo_file"] for e in missing], "--repo-type", "dataset",
                   "--revision", record["revision"], "--local-dir", str(directory),
                   "--max-workers", "4", "--json"]
        record["cli_command"] = command
        record["cli_started_at"] = now()
        print(json.dumps({"event": "hf_download", "key": record["key"], "command": command}), flush=True)
        try:
            completed = subprocess.run(command, env=os.environ.copy(), timeout=timeout,
                                       capture_output=True, check=False)
            record["cli_returncode"] = completed.returncode
            record["cli_stdout"] = completed.stdout.decode("utf-8", errors="replace")
            record["cli_stderr"] = completed.stderr.decode("utf-8", errors="replace")[-16000:]
            try:
                record["cli_json"] = json.loads(record["cli_stdout"])
            except json.JSONDecodeError:
                record["cli_json"] = None
            if completed.returncode:
                record["cli_error"] = "HF CLI returned a nonzero exit status; inspect exact state before rerunning."
        except subprocess.TimeoutExpired as exc:
            record["cli_error"] = "timeout; no automatic retry; inspect completed/partial files before rerunning"
            record["cli_returncode"] = None
            record["cli_stdout"] = (exc.stdout or b"").decode("utf-8", errors="replace")
            record["cli_stderr"] = (exc.stderr or b"").decode("utf-8", errors="replace")[-16000:]
        record["cli_finished_at"] = now()
        # Never blindly retry: inspect exact final and pinned temporary state now.
        record["post_download_pinned_partials"] = inspect_destination(directory, entries)
        for entry in missing:
            path = Path(entry["path"])
            if not path.exists():
                entry["status"] = "incomplete_missing"
                continue
            try:
                entry.update(verify_integrity(path, entry))
                entry["status"] = "integrity_verified"
                entry["validation"] = validate_content(path)
                entry["status"] = "verified"
                entry["acquisition"] = "hf_cli_downloaded_pinned_file"
            except Exception as exc:  # noqa: BLE001 - record each failed file and inspect the remaining files
                if entry["status"] != "integrity_failed":
                    entry["status"] = "validation_failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
        if any(e["status"] in ("integrity_failed", "validation_failed") for e in entries):
            record["status"] = "failed"
            return
    record["status"] = "verified" if all(e["status"] == "verified" for e in entries) else "incomplete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Inspect/validate exact local state; no downloads")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds per HF CLI group; no automatic retries")
    args = parser.parse_args()
    inspect_parent(ROOT)
    if not CLI.is_file() or hub_version != "2.4.0":
        raise RuntimeError("Expected installed HF 2.4.0 launcher/dependency")
    prior = None
    if MANIFEST.exists():
        check_regular(MANIFEST)
        prior = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if prior.get("identity") != IDENTITY or prior.get("plan_sha256") != PLAN_SHA256:
            raise RuntimeError("Refusing an unrelated preexisting additional manifest")
        if prior.get("status") == "running":
            raise RuntimeError("Existing running acquisition: inspect process/state first, do not blindly resume")
    manifest = {
        "identity": IDENTITY, "plan_sha256": PLAN_SHA256, "schema_version": 1,
        "created_at": prior["created_at"] if prior else now(), "status": "metadata_preflight",
        "training_performed": False, "evaluation_performed": False, "preparation_performed": False,
        "legacy_manifest_modified": False, "full_corpus_preflight_claim": False,
        "environment": ENVIRONMENT, "hub_version": hub_version, "hf_launcher": str(CLI),
        "metadata_api": "HfApi(token=False).repo_info(repo_type=dataset, revision=exact_SHA, files_metadata=True)",
        "resume_policy": "Skip only pinned hash+size+full-content-validated complete files; refuse mismatches. HF2.4 does not reuse partial bytes.",
        "native_git_exception": "Files with no LFS metadata are verified against pinned Git blob SHA1 plus size; local SHA256 recorded. No invented LFS SHA256 claims.",
        "split_hash_definition": "SHA256 of UTF8 json.dumps(sorted per-file inventory by repo_file, sort_keys=True, separators=(',', ':')); fields: repo_file,size,sha256,rows.",
        "deferred": {"BookCorpus": "unknown rights", "Cornell": "unknown rights",
                     "PersonaChat": "unknown rights", "Gutenberg": "not currently needed"},
        "corpora": [{**spec, "local_directory": str(ROOT / spec["directory"]),
                      "training_performed": False, "evaluation_performed": False,
                      "status": "pending_metadata"} for spec in CORPORA],
    }
    if prior:
        manifest["previous_run_status"] = prior["status"]
        manifest["previous_cli_runs"] = prior.get("previous_cli_runs", []) + [
            {"key": c["key"], "status": c["status"],
             **{name: value for name, value in c.items() if name.startswith("cli_")}}
            for c in prior["corpora"] if c.get("cli_command")]
    try:
        prepare_metadata(manifest)
        expected_bytes = sum(e["expected_size"] for c in manifest["corpora"] for e in c["files"])
        manifest["planned_bytes_including_cards"] = expected_bytes
        manifest["free_bytes_before"] = shutil.disk_usage(ROOT).free
        if manifest["free_bytes_before"] < expected_bytes * 2 + 1024 ** 3:
            raise RuntimeError("Insufficient D: headroom for safe downloads")
        # Global destination refusal check before any download, not just per group.
        for record in manifest["corpora"]:
            directory = Path(record["local_directory"])
            inspect_destination(directory, record["files"])
            for entry in record["files"]:
                if Path(entry["path"]).exists():
                    verify_integrity(Path(entry["path"]), entry)
        manifest["status"] = "running"
        save_manifest(manifest)
        for record in manifest["corpora"]:
            try:
                run_group(record, args.timeout, args.verify_only)
            except Exception as exc:  # noqa: BLE001 - persist failure state without retrying or replacing data
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                print(json.dumps({"event": "group_failed", "key": record["key"], "error": record["error"]}), flush=True)
            save_manifest(manifest)
            print(json.dumps({"event": "group_finished", "key": record["key"],
                              "status": record["status"], "splits": record["splits"]}), flush=True)
        manifest["status"] = "complete" if all(c["status"] == "verified" for c in manifest["corpora"]) else "incomplete"
        manifest["free_bytes_after"] = shutil.disk_usage(ROOT).free
    except Exception as exc:  # noqa: BLE001 - retain an independent acquisition failure record
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
    save_manifest(manifest)
    print(json.dumps({"manifest": str(MANIFEST), "status": manifest["status"],
                      "summary": manifest["summary"], "error": manifest.get("error")}), flush=True)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
