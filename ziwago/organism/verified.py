"""verified — source-grounded acceptance machinery (бывшие helper-классы v19).

Вынесено из train_ziwago_v19_verified_runtime.py (#30, 2026-11-12): классы
EmpiricalConfidenceCalibration / SourceGroundingAuthority / IndependentHypothesisSleep,
функции file_sha256 / stable_hash / content_words / _document_qa_from_state /
validate_v19_envelope / validate_verified_envelope / accept_candidate и V19_SCHEMA.
Тела перенесены verbatim (marker-based surgery). v19 ре-экспортирует имена обратно,
поэтому существующие импорты и старые снимки целы. Плоский ZiwagoOrganism
(ziwago/organism_core.py) импортирует отсюда — НЕ из модуля-обёртки v19.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch

from ziwago.cortex.document_qa import DocumentEvidenceRanker, validate_training_metadata
from ziwago.evaluation.gates import CheckpointCompatibilityError


V19_SCHEMA = "ziwago.checkpoint.v19.0"


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def content_words(text: str) -> set[str]:
    stop = {"a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "what", "why",
            "how", "can", "could", "please", "tell", "me", "about", "and", "or", "to", "of", "in",
            "on", "for", "with", "that", "this", "it", "i", "you", "my", "your", "by", "as"}
    return {w for w in re.findall(r"[\w]+(?:[-'][\w]+)*", text.lower()) if w not in stop}


def _document_qa_from_state(state: dict | None):
    if state is None:
        return None, None
    if type(state) is not dict or set(state) != {"ranker", "metadata"}:
        raise CheckpointCompatibilityError("Malformed document QA checkpoint state.")
    try:
        validate_training_metadata(state["metadata"])
        ranker = DocumentEvidenceRanker()
        ranker.load_state_dict(state["ranker"])
        if not ranker.trained or ranker.support_threshold is None:
            raise ValueError("Document QA checkpoint requires trained, calibrated weights")
    except ValueError as exc:
        raise CheckpointCompatibilityError(str(exc)) from exc
    return ranker, copy.deepcopy(state["metadata"])


def validate_v19_envelope(payload: dict) -> None:
    validate_verified_envelope(payload, version=19)


def validate_verified_envelope(payload: dict, version: int) -> None:
    if not isinstance(payload, dict) or payload.get(f"v{version}_schema_version") != f"ziwago.checkpoint.v{version}.0":
        raise CheckpointCompatibilityError(f"Missing v{version} runtime envelope.")
    runtime, state = payload.get("core_runtime_state"), payload.get(f"v{version}_state")
    runtime_fields = {"core", "moe_cortex", "spiking_language_loop", "rng", "bpe_cache"}
    state_fields = {"authority", "bcm_gate", "larkum", "analogies", "temporal", "calibration", "genesis"}
    if (not isinstance(runtime, dict) or runtime.get("runtime_version") != 19
            or not all(isinstance(runtime.get(k), dict) for k in runtime_fields)):
        raise CheckpointCompatibilityError("Missing complete v19 core runtime state.")
    if (not isinstance(state, dict) or state.get("version") != f"{version}.0"
            or not all(isinstance(state.get(k), dict) for k in state_fields)):
        raise CheckpointCompatibilityError("Missing complete v19 experimental state.")
    core_fields = {"dendritic_plateau_readout", "dialogue_state", "vsa_encoder", "chemistry",
                   "son_column", "pc_readout", "ba44_syntax_bridge", "temporal_generator", "temporal_codec",
                   "temporal_mode", "learning_enabled", "online_learning_enabled", "dialogue_writes_enabled"}
    rng_fields = {"python", "torch", "cuda", "numpy"}
    genesis_fields = {"valence_wave_generator", "world_graph", "emergent_ontology", "thalamic_router", "dynamic_stdp_role_map"}
    if (not core_fields.issubset(runtime["core"]) or not rng_fields.issubset(runtime["rng"])
            or not genesis_fields.issubset(state["genesis"])
            or not {"w_projector", "generator", "last_spike_raster"}.issubset(runtime["spiking_language_loop"])):
        raise CheckpointCompatibilityError("Incomplete v19 mechanism snapshot.")
    _document_qa_from_state(state.get("document_qa"))
    graph = state["genesis"]["world_graph"]
    if not isinstance(graph, dict) or graph.get("version") != 19 or not isinstance(graph.get("edges"), list):
        raise CheckpointCompatibilityError("Missing v19 graph state.")
    edge_fields = {"subject_vec64", "target_vec64", "edge_pole_vec64", "full_clause_tids", "rel_target_tids", "inf_rel_target_tids"}
    if any(not isinstance(edge, dict) or not edge_fields.issubset(edge) for edge in graph["edges"]):
        raise CheckpointCompatibilityError("Graph restoration would require neural recomputation.")


class EmpiricalConfidenceCalibration:
    def __init__(self, bins: int = 10, min_observations: int = 5):
        self.bins = bins
        self.min_observations = min_observations
        self.counts = [0] * bins
        self.correct = [0] * bins
        self.split_hash = None

    def fit(self, observations: Sequence[dict], split_name: str = "calibration") -> None:
        if split_name != "calibration" or not observations:
            raise ValueError("A nonempty independent calibration split is required.")
        counts, correct = [0] * self.bins, [0] * self.bins
        for item in observations:
            score, outcome = item["score"], item["correct"]
            if not isinstance(score, (int, float)) or not 0 <= score <= 1 or not isinstance(outcome, bool):
                raise ValueError("Calibration requires finite probabilities and boolean labels.")
            idx = min(self.bins - 1, int(score * self.bins))
            counts[idx] += 1
            correct[idx] += int(outcome)
        self.counts, self.correct = counts, correct
        self.split_hash = stable_hash(observations)

    def estimate(self, score: Optional[float]) -> dict:
        if score is None or not 0 <= score <= 1:
            return {"uncertainty": None, "status": "unavailable", "observations": 0}
        idx = min(self.bins - 1, int(score * self.bins))
        n = self.counts[idx]
        if n < self.min_observations:
            return {"uncertainty": None, "status": "insufficient_calibration", "observations": n}
        accuracy = self.correct[idx] / n
        return {"uncertainty": 1.0 - accuracy, "status": "empirical_bin_accuracy", "observations": n,
                "calibration_split_hash": self.split_hash}

    def export_state_dict(self) -> dict:
        return {"bins": self.bins, "min_observations": self.min_observations,
                "counts": list(self.counts), "correct": list(self.correct), "split_hash": self.split_hash}

    def load_state_dict(self, state: dict) -> None:
        self.bins, self.min_observations = state["bins"], state["min_observations"]
        self.counts, self.correct = list(state["counts"]), list(state["correct"])
        self.split_hash = state["split_hash"]


class SourceGroundingAuthority:
    def __init__(self, organism, world_graph):
        self.organism = organism
        self.world_graph = world_graph
        self.records: list[dict] = []
        self.quarantine: list[dict] = []

    def add_record(self, prompt: str, response: str, source: str, split: str = "train",
                   status: str = "asserted", family_id: str = "", provenance: Optional[dict] = None) -> dict:
        record = {"prompt": prompt, "response": response, "source": source, "split": split,
                  "status": status, "family_id": family_id, "provenance": copy.deepcopy(provenance or {}),
                  "observed_at": datetime.now(timezone.utc).isoformat()}
        record["record_id"] = stable_hash({k: record[k] for k in ("prompt", "response", "source", "split", "provenance")})
        existing = next((r for r in self.records if r["record_id"] == record["record_id"]), None)
        if existing is not None:
            return existing
        negatives = {"not", "no", "never", "cannot"}
        for other in self.records:
            left, right = content_words(other["response"]), content_words(response)
            same_prompt = " ".join(other["prompt"].casefold().split()) == " ".join(prompt.casefold().split())
            if same_prompt and left - negatives == right - negatives and bool(left & negatives) != bool(right & negatives):
                other["status"] = record["status"] = "quarantined"
                self.quarantine.append({"record_ids": [other["record_id"], record["record_id"]], "reason": "Opposing source assertions."})
        self.records.append(record)
        return record

    def ground(self, query: str, response: str) -> dict:
        normalized = lambda s: " ".join(re.findall(r"[\w'-]+", s.lower()))
        clauses = [s.strip() for s in re.split(r"[.!?;]+", response) if s.strip()]
        records = self.records + getattr(self.organism, "language_memory_records", [])
        for record in records:
            if record.get("status") != "quarantined":
                continue
            fact_text = normalized(str(record.get("response", "")))
            for clause in clauses:
                clause_text = normalized(clause)
                if clause_text == fact_text or (clause_text in fact_text and len(content_words(clause)) >= 3 and len(clause_text.split()) >= 5):
                    return {**self._reject("A source assertion is quarantined."), "evidence_kind": "conflicting_sources"}
        graph_check = self.world_graph.ground_response(query, response)
        if graph_check.get("disputed_edge_ids"):
            return {**graph_check, "grounded": False, "evidence_kind": "conflicting_sources", "truth_verified_externally": False}
        if graph_check.get("grounded") and graph_check.get("semantic_closure_score", 0.0) >= 0.8:
            return {**graph_check, "evidence_kind": "asserted_graph", "truth_verified_externally": False}
        qw, rw = content_words(query), content_words(response)
        if not rw:
            return self._reject("No material answer.")
        selected = []
        covered: set[str] = set()
        for clause in clauses:
            cw = content_words(clause)
            matched = None
            for record in records:
                if record.get("status") not in ("asserted", "verified") or record.get("split", "train") != "train":
                    continue
                fact = str(record.get("response", ""))
                fw = content_words(fact)
                anchors = content_words(str(record.get("prompt", ""))) | fw
                if not qw or not qw.intersection(anchors):
                    continue
                if len(qw) >= 3 and len(qw & anchors) / len(qw) < 0.5:
                    continue
                if not cw.issubset(fw):
                    continue
                negative = re.compile(r"\b(?:not|never|cannot|without|no|lack|lacks)\b", re.I)
                if bool(negative.search(clause)) != bool(negative.search(fact)):
                    continue
                # Word sets are insufficient evidence for exchanging subject and object.
                clause_text, fact_text = normalized(clause), normalized(fact)
                if clause_text != fact_text and (clause_text not in fact_text or len(cw) < 3 or len(clause_text.split()) < 5):
                    continue
                matched = record
                break
            if matched is None:
                return self._reject("A clause has no source-supported assertion.")
            selected.append({"source": matched.get("source"), "record_id": matched.get("record_id"),
                             "status": matched.get("status"), "observed_at": matched.get("observed_at"),
                             "provenance": matched.get("provenance", {})})
            covered.update(cw)
        return {"grounded": True, "semantic_closure_score": len(covered & rw) / len(rw),
                "sources": selected, "reasons": [], "evidence_kind": "source_text",
                "truth_verified_externally": all(s["status"] == "verified" for s in selected)}

    def ground_analogy(self, dreamer, source: str, target: str, match: dict) -> dict:
        if match.get("ambiguous") or not match.get("proof") or source not in dreamer.graphs or target not in dreamer.graphs:
            return self._reject("Ambiguous or unsupported graph correspondence.")
        mapping = match["mapping"]
        sg, tg = dreamer.graphs[source], dreamer.graphs[target]
        if set(mapping) != set(sg["node_types"]) or set(mapping.values()) != set(tg["node_types"]):
            return self._reject("Graph correspondence omits nodes.")
        if len(set(mapping.values())) != len(mapping) or len(match["proof"]) != len(sg["edges"]):
            return self._reject("Graph correspondence is incomplete or not one-to-one.")
        if (sorted(stable_hash(p["source_edge"]) for p in match["proof"]) != sorted(map(stable_hash, sg["edges"]))
                or sorted(stable_hash(p["target_edge"]) for p in match["proof"]) != sorted(map(stable_hash, tg["edges"]))):
            return self._reject("Graph proof repeats or omits evidence.")
        for node, counterpart in mapping.items():
            if sg["node_types"].get(node) != tg["node_types"].get(counterpart):
                return self._reject("Node types disagree.")
        for proof in match["proof"]:
            se, te = proof["source_edge"], proof["target_edge"]
            if se not in sg["edges"] or te not in tg["edges"]:
                return self._reject("A correspondence edge has no source.")
            if mapping[se["subject"]] != te["subject"] or mapping[se["object"]] != te["object"]:
                return self._reject("Correspondence reverses roles.")
            if any(se[k] != te[k] for k in ("relation", "polarity", "causal_kind", "type")):
                return self._reject("Correspondence changes edge semantics.")
        return {"grounded": True, "semantic_closure_score": 1.0, "reasons": [],
                "sources": [{"model": source, "hash": stable_hash(sg)}, {"model": target, "hash": stable_hash(tg)}],
                "evidence_kind": "formal_graph_correspondence", "truth_verified_externally": False}

    def ground_simulation(self, query: str, simulation: dict, simulator) -> dict:
        expected = simulator.simulate_counterfactual_intervention(
            query, self.world_graph, self.organism.encode_text, self.organism.decode_token, self.organism.get_vec64)
        if expected is None or any(simulation.get(k) != expected[k] for k in (
                "counterfactual_response", "proof_edge_ids", "intervention_result", "uncertain")):
            return self._reject("Intervention output does not match the declared model.")
        return {"grounded": True, "semantic_closure_score": 1.0, "reasons": expected["intervention_result"]["reasons"],
                "sources": expected["sources"], "proof_edge_ids": expected["proof_edge_ids"],
                "evidence_kind": "conditional_declared_model", "model_uncertain": expected["uncertain"],
                "truth_verified_externally": False}

    @staticmethod
    def _reject(reason: str) -> dict:
        return {"grounded": False, "semantic_closure_score": 0.0, "sources": [], "reasons": [reason],
                "truth_verified_externally": False}

    def export_state_dict(self) -> dict:
        return {"records": copy.deepcopy(self.records), "quarantine": copy.deepcopy(self.quarantine)}

    def load_state_dict(self, state: dict) -> None:
        self.records = copy.deepcopy(state["records"])
        self.quarantine = copy.deepcopy(state.get("quarantine", []))


class IndependentHypothesisSleep:
    def __init__(self, runtime):
        self.runtime = runtime

    def generate_hypotheses(self) -> list[dict]:
        graph = self.runtime.world_graph
        hypotheses = []
        for edge in graph.edges:
            if edge.status not in ("asserted", "verified") or edge.conflict_ids:
                continue
            reverse = copy.copy(edge)
            reverse.subject, reverse.target = edge.target, edge.subject
            negative = copy.copy(edge)
            negative.polarity = -1
            eligible = graph._eligible_edges(edge.conditions)
            signature = lambda e: (e.subject, e.relation, e.target, e.polarity, e.causal_kind)
            for kind, candidate in (("role_permutation", reverse), ("negation", negative)):
                hypotheses.append({"kind": kind, "query": f"{candidate.subject} {candidate.target}",
                                   "candidate": graph._render_edge(candidate),
                                   "should_be_supported": any(signature(e) == signature(candidate) for e in eligible)})
            absent = {edge.subject}
            while True:
                expanded = absent | {e.target for e in eligible
                                     if e.subject in absent and e.polarity == 1 and e.causal_kind == "necessary"}
                if expanded == absent:
                    break
                absent = expanded
            nodes = {e.subject for e in graph.edges if graph._valid_entity(e.subject)} | {
                e.target for e in graph.edges if graph._valid_entity(e.target)}
            hypotheses.append({"kind": "alternative_cause", "query": edge.target,
                               "candidate": None, "intervention": {edge.subject: False}, "conditions": edge.conditions,
                               "expected_values": {node: False if node in absent else None for node in nodes},
                               "evidence_kind": "declared_constraint_control"})
        return hypotheses

    def _check_hypotheses(self, hypotheses: Sequence[dict]) -> list[dict]:
        checks = []
        for hypothesis in hypotheses:
            if hypothesis["kind"] == "alternative_cause":
                outcome = self.runtime.world_graph.intervene(hypothesis["intervention"], conditions=hypothesis["conditions"])
                checks.append({**hypothesis, "outcome": outcome,
                               "passed": outcome["values"] == hypothesis["expected_values"]})
            else:
                support = self.runtime.authority.ground(hypothesis["query"], hypothesis["candidate"])
                checks.append({**hypothesis, "support": support,
                               "passed": bool(support["grounded"]) == hypothesis["should_be_supported"]})
        return checks

    def consolidate(self, validator: Optional[Callable] = None) -> dict:
        if validator is None:
            return {"accepted": False, "mutated": False, "reason": "Independent acceptance validator required."}
        with self.runtime.evaluation_context():
            before = validator(self.runtime)
        if (not before.get("passed", False) or not before.get("split_hash") or not before.get("results")
                or not all(row.get("passed") is True for row in before["results"])):
            return {"accepted": False, "mutated": False, "reason": "Baseline acceptance failed or split is unidentified.",
                    "baseline": before}
        hypotheses = self.generate_hypotheses()
        checks = self._check_hypotheses(hypotheses)
        if not all(c["passed"] for c in checks):
            return {"accepted": False, "mutated": False, "reason": "Source-based hypothesis controls failed.", "hypotheses": checks}
        with tempfile.TemporaryDirectory(prefix="ziwago-v19-sleep-", dir=self.runtime.root / "Produce") as temp:
            snapshot = Path(temp) / "snapshot.pt"
            original_path = self.runtime.core_organism.loaded_checkpoint_path
            self.runtime.save_candidate(snapshot)
            from ziwago.app.data import TokenizedDialogPair
            core = self.runtime.core_organism
            replayed_sources, accepted = 0, False
            try:
                for record in self.runtime.authority.records:
                    if record.get("split") != "train" or record.get("status") not in ("asserted", "verified"):
                        continue
                    if not self.runtime.authority.ground(record["prompt"], record["response"])["grounded"]:
                        continue
                    q_ids, r_ids = core.encode_text(record["prompt"]), core.encode_text(record["response"])
                    if (not q_ids or not 2 <= len(r_ids) < core.dendritic_plateau_readout.max_pos
                            or core.eos_token_id in r_ids):
                        continue
                    pair = TokenizedDialogPair(record["source"], q_ids, r_ids, quality_reward_hint=1.0)
                    core.train_wave_with_neuromodulator_reward([pair], wave_idx=1)
                    replayed_sources += 1
                    if replayed_sources >= 32:
                        break
                readout = core.dendritic_plateau_readout
                n_cols = len(readout.col_to_tok)
                for start in range(0, readout.d_dend, 128):
                    block = readout.w_dend_out[start:start + 128, :n_cols]
                    rows = torch.nonzero(block.abs().sum(dim=1) > 0).flatten() + start
                    readout.shield.protect_rows(readout.w_dend_out, rows, n_cols)
                with self.runtime.evaluation_context():
                    after = validator(self.runtime)
                checks = self._check_hypotheses(hypotheses)
                accepted = (bool(after.get("passed")) and bool(after.get("results")) and after.get("split_hash") == before["split_hash"]
                            and len(after["results"]) == len(before["results"])
                            and all(row.get("passed") is True for row in after["results"])
                            and all(c["passed"] for c in checks))
            finally:
                if not accepted:
                    self.runtime.load_v19_checkpoint(snapshot)
                    core.loaded_checkpoint_path = original_path
        return {"accepted": accepted, "mutated": accepted, "hypotheses": checks,
                "source_records_replayed": replayed_sources if accepted else 0,
                "baseline": before, "candidate": after, "self_generated_facts_consolidated": 0}



def accept_candidate(path: Path, report: dict, destination: Path) -> dict:
    path, destination = Path(path), Path(destination)
    report = copy.deepcopy(report)
    if not report.get("passed") or not report.get("split_hash") or not report.get("results"):
        raise ValueError("Independent nonempty acceptance suite must pass.")
    if not all(row.get("passed") for row in report["results"]):
        raise ValueError("Acceptance contains failed results.")
    if destination.name != "ziwago_v19_0_accepted.pt":
        raise ValueError("Promotion destination must be the separate v19 accepted checkpoint.")
    if destination.exists():
        raise FileExistsError(destination)
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    with tempfile.TemporaryFile(dir=destination.parent) as snapshot:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                snapshot.write(chunk)
        if report.get("checkpoint_sha256") != digest.hexdigest():
            raise ValueError("Acceptance is not bound to the candidate hash.")
        snapshot.seek(0)
        payload = torch.load(snapshot, map_location="cpu", weights_only=True)
    validate_v19_envelope(payload)
    payload["acceptance"] = report
    payload["certification"] = "accepted_for_declared_suite_only"
    fd, temporary = tempfile.mkstemp(prefix=".accepted-v19-", suffix=".pt", dir=destination.parent)
    os.close(fd)
    temp = Path(temporary)
    try:
        torch.save(payload, temp)
        with temp.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.link(temp, destination)
    finally:
        temp.unlink(missing_ok=True)
    return {"path": str(destination), "sha256": file_sha256(destination), "acceptance": report}
