#!/usr/bin/env python3
"""Interactive and programmatic ZIWAGO v19 assistant."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re
import time
from typing import Optional

import torch
import torch.nn.functional as F

from ziwago.evaluation.gates import OPEN_BOOK_MAX_SENTENCES, evaluate_grammar_and_artifacts, split_sentences
from train_ziwago_v14_pure_synaptic_dialogue import TokenizedDialogPair
from ziwago.organism_core import ROOT, ZiwagoOrganism

_QA_STOP = frozenset({"the", "a", "an", "to", "of", "in", "on", "is", "are", "was", "were",
                      "and", "or", "for", "with", "that", "this", "it", "do", "can", "you",
                      "what", "how", "why", "where", "when", "who", "does", "did", "has", "have"})


class BasalGangliaActionSelector:
    ACTION_EPISODIC_MEMORY_RECALL = "ACTION_EPISODIC_MEMORY_RECALL"
    ACTION_ONLINE_DA_CONSOLIDATION = "ACTION_ONLINE_DA_CONSOLIDATION"
    ACTION_COUNTERFACTUAL_SIMULATION = "ACTION_COUNTERFACTUAL_SIMULATION"
    ACTION_ENDOGENOUS_CORTICAL_THINK = "ACTION_ENDOGENOUS_CORTICAL_THINK"
    ACTION_ACTIVE_INFERENCE_CLARIFY = "ACTION_ACTIVE_INFERENCE_CLARIFY"
    ACTION_MULTI_HOP_OR_PARAGRAPH_SYNTHESIS = "ACTION_MULTI_HOP_OR_PARAGRAPH_SYNTHESIS"
    ACTION_DIRECT_SYNAPTIC_RESPONSE = "ACTION_DIRECT_SYNAPTIC_RESPONSE"

    def __init__(self, resonance_gate_enabled: bool = False, unknown_resonance: float = 0.3):
        # Opt-in гиппокампальный NoGo-гейт (гиперпрямой путь STN): при низком
        # CA3-резонансе (< unknown_resonance) уверенный DIRECT_SYNAPTIC_RESPONSE
        # заменяется на ACTIVE_INFERENCE_CLARIFY («не знаю» / уточнение).
        # Default off → select_action игнорирует резонанс как раньше (bit-identical).
        if not 0.0 <= unknown_resonance <= 1.0:
            raise ValueError("unknown_resonance must lie in [0, 1]")
        self.resonance_gate_enabled = bool(resonance_gate_enabled)
        self.unknown_resonance = float(unknown_resonance)

    def select_action(self, utterance: str, is_episodic_query: bool, is_teaching_or_identity: bool,
                      wants_multi_hop_or_paragraph: bool, ca3_resonance: float,
                      active_inference_triggered: bool = False,
                      endogenous_think_triggered: bool = False, counterfactual_triggered: bool = False):
        choices = [(is_teaching_or_identity, self.ACTION_ONLINE_DA_CONSOLIDATION),
                   (is_episodic_query, self.ACTION_EPISODIC_MEMORY_RECALL),
                   (counterfactual_triggered, self.ACTION_COUNTERFACTUAL_SIMULATION),
                   (endogenous_think_triggered, self.ACTION_ENDOGENOUS_CORTICAL_THINK),
                   (active_inference_triggered, self.ACTION_ACTIVE_INFERENCE_CLARIFY),
                   (wants_multi_hop_or_paragraph, self.ACTION_MULTI_HOP_OR_PARAGRAPH_SYNTHESIS)]
        action = next((name for enabled, name in choices if enabled), self.ACTION_DIRECT_SYNAPTIC_RESPONSE)
        if (self.resonance_gate_enabled and action == self.ACTION_DIRECT_SYNAPTIC_RESPONSE
                and ca3_resonance < self.unknown_resonance):
            action = self.ACTION_ACTIVE_INFERENCE_CLARIFY
        return action, {name: enabled for enabled, name in choices}


class ZiwagoBiologicalSNNAssistant:
    def __init__(self, root: Path = ROOT, checkpoint_path: Optional[Path] = None, workers: int = 4,
                 cpu_threads: int = 8, seed: int = 42, knowledge_dir: Optional[Path] = None,
                 auto_load_checkpoint: bool = True, expected_checkpoint_sha256: Optional[str] = None,
                 runtime_version: int = 19, resonance_action_gate: bool = False,
                 unknown_resonance: float = 0.3):
        if runtime_version not in (19, 20):
            raise ValueError("runtime_version must be 19 or 20")
        self.root = Path(root)
        # Flat composition (H2): one class, two modes via the integrated flag.
        # runtime_version=19 → integrated=False (no physiology); 20 → integrated=True.
        self.genesis = ZiwagoOrganism(root=self.root, checkpoint_path=checkpoint_path,
                                      workers=workers, cpu_threads=cpu_threads, seed=seed,
                                      knowledge_dir=knowledge_dir, auto_load_checkpoint=auto_load_checkpoint,
                                      expected_checkpoint_sha256=expected_checkpoint_sha256,
                                      integrated=(runtime_version == 20))
        self.organism = self.genesis.core_organism
        self.basal_ganglia = BasalGangliaActionSelector(
            resonance_gate_enabled=resonance_action_gate, unknown_resonance=unknown_resonance)
        self.loaded_checkpoint_info = self.genesis.loaded_checkpoint_info
        self.knowledge_distillery = self.genesis.knowledge_distillery
        self.bcm_gate = self.genesis.bcm_gate
        self.larkum_microcircuit = self.genesis.larkum_microcircuit
        self.metaplasticity_shield = self.organism.dendritic_plateau_readout.shield
        self.isomorphism_dreamer = self.genesis.isomorphism_dreamer
        self.consciousness_dashboard = self.genesis.consciousness_dashboard
        self.organism.dialogue_writes_enabled = False

    @contextmanager
    def evaluation_context(self):
        with self.genesis.evaluation_context():
            yield self

    def teach_multi_sentence_episode(self, teaching_text: str, source: str = "user") -> dict:
        if self.genesis.evaluation_depth:
            raise RuntimeError("Teaching is forbidden during evaluation.")
        clean = re.sub(r"^(?:remember that|note that|learn that|fact:)\s*", "", teaching_text.strip(), flags=re.I)
        sentences = [s + "." if not s.endswith((".", "!", "?")) else s
                     for s, _, _ in split_sentences(clean) if len(s.split()) >= 3]
        tokens, trained, edges = 0, 0, 0
        poles, records = [], []
        for sentence in sentences:
            ids = self.organism.encode_text(sentence)
            sdr = self.organism.dendritic_plateau_readout._compute_cortical_sdr2048(ids, normalize=False)
            grammar = evaluate_grammar_and_artifacts(sentence, min_words=3)
            gate = self.bcm_gate.evaluate_neuromodulatory_gate(sdr, grammar_ok=bool(grammar["grammar_coherent"]))
            record = self.genesis.authority.add_record(sentence, sentence, source)
            records.append(record["record_id"])
            edges += len(self.genesis.world_graph.extract_and_consolidate_sentence(sentence, source=source, status="asserted"))
            if not gate["accepted"] or not self.genesis.authority.ground(sentence, sentence)["grounded"]:
                continue
            pair = TokenizedDialogPair(source, ids, ids, quality_reward_hint=2.0)
            self.organism.train_wave_with_neuromodulator_reward([pair], wave_idx=1)
            tokens += len(ids)
            trained += 1
            poles.append(self.organism.compute_stimulus_vec64(ids))
        return {"sentences_imprinted": trained, "tokens_imprinted": tokens, "episode_links_created": 0,
                "source_assertions_recorded": len(records), "records": records, "graph_edges_added": edges,
                "pole_vecs": poles, "clean_fact": " ".join(sentences), "truth_verified_externally": False}

    def _finalize(self, prompt: str, result: dict, started: float, action: str,
                  check: Optional[dict] = None, remember: bool = True) -> dict:
        text = str(result.get("pure_weight_generated_response", ""))
        check = check if check is not None else self.genesis.authority.ground(prompt, text)
        grounded = bool(check.get("grounded"))
        if not grounded:
            if text and not self.genesis.evaluation_depth:
                self.genesis.authority.quarantine.append({"query": prompt, "candidate": text, "reasons": check.get("reasons", [])})
            text = "I do not have source-supported knowledge for this question."
            result["active_inference_clarification_triggered"] = True
            action = self.basal_ganglia.ACTION_ACTIVE_INFERENCE_CLARIFY
        passport = result.setdefault("response_passport", {})
        passport.update({"version": getattr(self.genesis, "version", "19.0"), "grounding_enforced": True, "grounded": grounded,
                         "grounding": check, "sources": check.get("sources", []), "selected_action": action})
        trace = getattr(self.organism, "last_cortical_trajectory_trace", [])
        spikes = getattr(getattr(self.organism, "spiking_language_loop", None), "last_spike_raster", None)
        passport["consciousness_metrics"] = self.consciousness_dashboard.compute_turn_consciousness_metrics(
            ca3_resonance=float(result.get("ca3_top1_resonance", 0.0)),
            epistemic_uncertainty=1.0 - float(check.get("semantic_closure_score", 0.0)),
            cortical_trajectory_trace=trace, spike_raster=spikes,
            total_columns=self.organism.moe_cortex.n_columns)
        calibration = self.genesis.confidence_calibration.estimate(result.get("prediction_confidence"))
        passport["confidence_calibration"] = calibration
        if hasattr(self.organism, "life"):
            result["organism"] = self.organism.life.telemetry(self.organism)
        result.update({"stimulus": prompt, "version": getattr(self.genesis, "version", "19.0"), "pure_weight_generated_response": text,
                       "basal_ganglia_selected_action": action, "epistemic_uncertainty": calibration["uncertainty"],
                       "uncertainty_status": calibration["status"],
                       "grounded": grounded, "swr_consolidated_into_nmda_matrix": False,
                       "assistant_turn_latency_ms": round((time.perf_counter() - started) * 1000, 2)})
        if remember and not self.genesis.evaluation_depth:
            vec = self.organism.compute_stimulus_vec64(self.organism.encode_text(prompt))
            self.organism.dialogue_state.record_turn_and_poles(
                stimulus_vec64=vec, activated_poles=[vec] if grounded else [], utterance=prompt,
                response_text=text, is_grounded_conclusion=grounded, role_filler_bindings={},
                was_clarification_request=not grounded, response_passport=passport)
        return result

    @staticmethod
    def _action_evidence(resonance: float, enable_active_inference: bool) -> list:
        """Веса действий для action-контекста (7 компонент, все ≥ 0).

        Резонанс CA3 у обученной модели БЫВАЕТ > 1 (до 2.5), а evidence —
        нормированные веса: отрицательная компонента отвергалась валидатором
        action-контекста, и маршрутизация падала на каждом запросе с сильным
        резонансом. Поэтому резонанс прижимается в [0, 1]
        (max/min устойчивы к NaN: nan → 0.0).
        """
        resonance_gate = min(1.0, max(0.0, float(resonance)))
        return [0., 0., 0., 0.,
                (1.0 - resonance_gate) * 0.5 if enable_active_inference else 0.,
                0., 0.6 + 0.4 * resonance_gate]

    def chat(self, user_utterance: str, max_tokens: int = 64, enable_active_inference: bool = True) -> dict:
        started = time.perf_counter()
        prompt = user_utterance.strip()
        if not prompt:
            raise ValueError("Empty utterance.")
        self.organism.last_cortical_trajectory_trace = []
        loop = self.organism.spiking_language_loop
        loop.last_spike_raster = loop.last_spike_raster[:0]
        low = prompt.lower()
        explicit_teach = bool(re.match(r"^(?:remember that|note that|learn that|fact:)\s", prompt, re.I))
        identity = re.search(r"\bmy name is\s+([\w'-]+)\b", prompt, re.I) if "?" not in prompt else None
        history = self.organism.dialogue_state.is_episodic_history_query(prompt)
        cf = self.genesis.counterfactual_simulator.is_counterfactual_query(prompt)
        analogy = bool(re.search(r"\b(?:compare|analogy|analogous|isomorphism)\b", low))
        multi = "causal chain" in low or "step by step" in low
        action, scores = self.basal_ganglia.select_action(prompt, history, explicit_teach or bool(identity), multi, 0.0,
                                                         endogenous_think_triggered=analogy,
                                                         counterfactual_triggered=cf)
        result = {"basal_ganglia_d1_scores": scores}
        if hasattr(self.organism, "life"):
            life = self.organism.life
            abandoned = life.actions.cancel()
            q = self.organism.compute_stimulus_vec64(self.organism.encode_text(prompt))
            poles = self.organism.dendritic_plateau_readout.w_ca3_attractors
            resonance = max(0.0, float((torch.nn.functional.normalize(poles, dim=-1) @ q).max())) if poles.numel() else 0.0
            actions = [self.basal_ganglia.ACTION_EPISODIC_MEMORY_RECALL,
                       self.basal_ganglia.ACTION_ONLINE_DA_CONSOLIDATION,
                       self.basal_ganglia.ACTION_COUNTERFACTUAL_SIMULATION,
                       self.basal_ganglia.ACTION_ENDOGENOUS_CORTICAL_THINK,
                       self.basal_ganglia.ACTION_ACTIVE_INFERENCE_CLARIFY,
                       self.basal_ganglia.ACTION_MULTI_HOP_OR_PARAGRAPH_SYNTHESIS,
                       self.basal_ganglia.ACTION_DIRECT_SYNAPTIC_RESPONSE]
            if explicit_teach or identity:
                evidence = [0., 1., 0., 0., 0., 0., 0.]
            elif history or cf or analogy or multi:
                evidence = [float(history), 0., float(cf), float(analogy), 0., float(multi), 0.]
            else:
                evidence = self._action_evidence(resonance, enable_active_inference)
            decision = self.genesis.select_action(q, evidence)
            result.update({"organism_decision": decision, "abandoned_decision": abandoned,
                           "ca3_top1_resonance": resonance})
            if not decision["committed"] or decision["action"] == 4:
                result["pure_weight_generated_response"] = ""
                result["organism_deferred"] = True
                return self._finalize(prompt, result, started, self.basal_ganglia.ACTION_ACTIVE_INFERENCE_CLARIFY)
            action = actions[decision["action"]]
            history, cf, analogy, multi = (decision["action"] == i for i in (0, 2, 3, 5))
        if identity:
            if self.genesis.evaluation_depth:
                raise RuntimeError("Identity writes are forbidden during evaluation.")
            name = identity.group(1)
            self.organism.dialogue_state.user_name = name
            record = self.genesis.authority.add_record("What is my name?", f"Your name is {name}.", "user_identity")
            result["pure_weight_generated_response"] = f"Your name is {name}."
            check = {"grounded": True, "semantic_closure_score": 1.0, "sources": [record],
                     "evidence_kind": "user_assertion", "truth_verified_externally": False}
            return self._finalize(prompt, result, started, action, check)
        if explicit_teach:
            teaching = self.teach_multi_sentence_episode(prompt)
            result.update({"pure_weight_generated_response": "Recorded your source-attributed assertion; external truth is unverified.",
                           "episode_teaching_telemetry": {k: v for k, v in teaching.items() if k != "pole_vecs"},
                           "online_dopamine_imprinted_tokens": teaching["tokens_imprinted"]})
            check = {"grounded": bool(teaching["source_assertions_recorded"]), "semantic_closure_score": 1.0,
                     "sources": teaching["records"], "evidence_kind": "observed_write", "truth_verified_externally": False}
            return self._finalize(prompt, result, started, action, check, remember=False)
        if re.search(r"\b(?:my name|what.*name)\b", low):
            name = self.organism.dialogue_state.user_name
            result["pure_weight_generated_response"] = f"Your name is {name}." if name else ""
            return self._finalize(prompt, result, started, action)
        if history:
            facts = [r for r in self.genesis.authority.records
                     if r.get("source", "").startswith("user") and r.get("status") in ("asserted", "verified")
                     and self.genesis.authority.ground(r["prompt"], r["response"])["grounded"]]
            result["pure_weight_generated_response"] = " ".join(r["response"] for r in facts[-8:])
            check = {"grounded": bool(facts), "semantic_closure_score": 1.0 if facts else 0.0,
                     "sources": [{"source": r["source"], "record_id": r["record_id"]} for r in facts[-8:]],
                     "evidence_kind": "source_attributed_history", "truth_verified_externally": False}
            return self._finalize(prompt, result, started, action, check)
        if cf:
            simulation = self.genesis.counterfactual_simulator.simulate_counterfactual_intervention(
                utterance=prompt, world_graph=self.genesis.world_graph, encode_fn=self.organism.encode_text,
                decode_fn=self.organism.decode_token, get_vec64_fn=self.organism.get_vec64)
            if simulation:
                check = self.genesis.authority.ground_simulation(prompt, simulation, self.genesis.counterfactual_simulator)
                result.update({"pure_weight_generated_response": simulation["counterfactual_response"],
                               "counterfactual_simulation_triggered": True, "counterfactual_telemetry": simulation,
                               "model_prediction_uncertain": simulation["uncertain"], "generation_kind": "constrained_model_surface"})
                return self._finalize(prompt, result, started, action, check, remember=False)
            result["pure_weight_generated_response"] = ""
            return self._finalize(prompt, result, started, action)
        if analogy:
            match = self.isomorphism_dreamer.resolve_cross_domain_isomorphism(prompt)
            if match:
                source, target = match["domains_bridged"]
                check = self.genesis.authority.ground_analogy(self.isomorphism_dreamer, source, target, match)
                pairs = "; ".join(f"{node} corresponds to {other}" for node, other in sorted(match["mapping"].items()))
                result.update({"pure_weight_generated_response": f"Typed graph correspondence between {source} and {target}: {pairs}.",
                               "isomorphism_telemetry": match, "generation_kind": "constrained_graph_correspondence"})
                return self._finalize(prompt, result, started, action, check, remember=False)
            result["pure_weight_generated_response"] = ""
            return self._finalize(prompt, result, started, action)
        raw = self.genesis.generate_response(prompt, max_tokens=max_tokens, enable_active_inference=enable_active_inference,
                                            enable_multi_hop=multi)
        result.update(raw)
        if not hasattr(self.organism, "life"):
            resonance = float(result.get("ca3_top1_resonance", 1.0))
            try:
                emergent = self.genesis.sync_emergent_ontology_into_role_binder(prompt)
            except Exception:
                emergent = {}
            thal_act, thal_scores = self.genesis.thalamic_router.route_speech_act(
                utterance=prompt, ca3_resonance=resonance,
                prev_was_clarify=False, known_in_ontology=bool(emergent))
            result["thalamic_speech_act"] = thal_act
            result["thalamic_d1_scores"] = thal_scores
            result["emergent_stdp_role_discoveries"] = emergent
            try:
                q_vec = self.organism.compute_stimulus_vec64(self.organism.encode_text(prompt))
                larkum = self.larkum_microcircuit.step_microcircuit(q_vec, q_vec, learn_sst=False)
                result["larkum_bac_burst"] = bool(larkum["bac_burst_flag"])
                result["consciousness_metrics"] = self.consciousness_dashboard.compute_turn_consciousness_metrics(
                    ca3_resonance=resonance,
                    epistemic_uncertainty=float(result.get("epistemic_uncertainty", 0.0)),
                    cortical_trajectory_trace=result.get("cortical_trajectory_trace", []))
            except Exception as exc:
                result["larkum_bac_burst"] = False
                result["consciousness_metrics"] = {"unavailable": f"{type(exc).__name__}: {exc}"}
            action, scores = self.basal_ganglia.select_action(
                prompt, history, explicit_teach or bool(identity), multi, resonance,
                endogenous_think_triggered=analogy, counterfactual_triggered=cf)
            result["basal_ganglia_d1_scores"] = scores
        result["basal_ganglia_selected_action"] = action
        if not self.genesis.authority.ground(prompt, str(result.get("pure_weight_generated_response", ""))).get("grounded"):
            graph = self.genesis.world_graph.synthesize_novel_sentence_from_graph(prompt)
            if graph:
                result.update({"pure_weight_generated_response": graph["synthesized_sentence"],
                               "graph_telemetry": graph, "generation_kind": "constrained_graph_rendering"})
        return self._finalize(prompt, result, started, action)

    def chat_with_document(self, question: str, document: str, max_sentences: int = OPEN_BOOK_MAX_SENTENCES) -> dict:
        started = time.perf_counter()
        question = question.strip()
        if not question or not document.strip():
            raise ValueError("Question and document must be non-empty.")
        if type(max_sentences) is not int or not 1 <= max_sentences <= OPEN_BOOK_MAX_SENTENCES:
            raise ValueError("max_sentences must fit the extractive sentence budget.")
        self.organism.last_cortical_trajectory_trace = []
        loop = self.organism.spiking_language_loop
        loop.last_spike_raster = loop.last_spike_raster[:0]
        if not self.genesis.evaluation_depth:
            with self.evaluation_context():
                return self.chat_with_document(question, document, max_sentences)
        sentences = split_sentences(document)
        ranker = self.genesis.document_ranker
        assessment = None
        if ranker is not None:
            assessment = ranker.assess(question, [sentence for sentence, _, _ in sentences])
            scored = [(score, sentence, start, end) for score, (sentence, start, end)
                      in zip(assessment["scores"], sentences)]
            scored.sort(key=lambda row: row[0], reverse=True)
            selected = [scored[0][1:]] if scored and assessment["supported"] else []
            ranking_method = "learned_ordered_question_sentence_interaction"
        else:
            ro = self.organism.dendritic_plateau_readout
            q_ids = self.organism.encode_text(question)
            q_sdr = ro._compute_cortical_sdr2048(q_ids, normalize=False)
            idf = 1.0 / torch.sqrt(ro.sdr_column_usage)
            q_weighted = F.normalize(q_sdr * idf + 1e-8, p=2, dim=0)
            p_strip = " .,!?;:'()[]{}" + chr(34)
            q_words = {w.strip(p_strip).lower() for w in question.split()
                       if len(w.strip(p_strip)) >= 3 and w.strip(p_strip).lower() not in _QA_STOP}
            scored = []
            for sent, start, end in sentences:
                s_ids = self.organism.encode_text(sent)
                s_sdr = ro._compute_cortical_sdr2048(s_ids, normalize=False)
                s_weighted = F.normalize(s_sdr * idf + 1e-8, p=2, dim=0)
                sdr_score = float(torch.dot(q_weighted, s_weighted))
                s_words = {w.strip(p_strip).lower() for w in sent.split()
                           if len(w.strip(p_strip)) >= 3 and w.strip(p_strip).lower() not in _QA_STOP}
                overlap = len(q_words & s_words) / max(1, len(q_words))
                scored.append((sdr_score + 0.3 * overlap, sent, start, end))
            scored.sort(key=lambda x: x[0], reverse=True)
            selected = [(s, st, en) for sc, s, st, en in scored[:max_sentences] if sc > 0.05]
            ranking_method = "legacy_sdr_lexical"
        response = " ".join(s for s, _, _ in selected)
        grounded = bool(selected)
        sources = [{"source": "user_document", "text": s, "start": st, "end": en} for s, st, en in selected]
        result = {"pure_weight_generated_response": response, "generation_kind": "extractive_document_qa",
                  "document_sentences_considered": len(sentences), "document_sentences_selected": len(selected),
                  "document_ranking_method": ranking_method, "document_ranker_assessment": assessment,
                  "semantic_relation_verified": False,
                  "sentence_scores": [{"sentence": s, "score": round(sc, 4)} for sc, s, _, _ in scored[:5]]}
        check = {"grounded": grounded, "semantic_closure_score": 1.0 if grounded else 0.0,
                 "sources": sources, "evidence_kind": "verbatim_document_span", "truth_verified_externally": False}
        return self._finalize(question, result, started, "ACTION_DOCUMENT_QA", check, remember=False)

    def sleep_and_consolidate(self, save_checkpoint_path: Optional[Path] = None, validator=None) -> dict:
        if self.genesis.evaluation_depth:
            raise RuntimeError("Sleep is forbidden during evaluation.")
        result = (self.genesis.sleep_cycle() if hasattr(self.organism, "life")
                  else self.genesis.sleep_engine.consolidate(validator))
        if save_checkpoint_path is not None and (hasattr(self.organism, "life") or result.get("accepted")):
            result["saved_checkpoint"] = self.genesis.save_candidate(Path(save_checkpoint_path))
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--knowledge-dir", type=Path)
    parser.add_argument("--runtime-version", type=int, choices=(19, 20), default=19)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    assistant = ZiwagoBiologicalSNNAssistant(checkpoint_path=args.checkpoint, knowledge_dir=args.knowledge_dir,
                                            workers=args.workers, cpu_threads=args.cpu_threads,
                                            runtime_version=args.runtime_version)
    print(f"ZIWAGO v{args.runtime_version} experimental runtime; /exit, /status, /sleep")
    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt == "/exit":
            break
        if prompt == "/status":
            print(json.dumps({"version": assistant.genesis.version, "checkpoint": assistant.loaded_checkpoint_info,
                              "organism": assistant.organism.life.telemetry(assistant.organism)
                              if hasattr(assistant.organism, "life") else None}, ensure_ascii=False))
        elif prompt == "/sleep":
            print(json.dumps(assistant.sleep_and_consolidate(), ensure_ascii=False))
        elif prompt:
            print(assistant.chat(prompt)["pure_weight_generated_response"])


if __name__ == "__main__":
    main()
