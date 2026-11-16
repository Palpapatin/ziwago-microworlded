"""Flat composition organism (v20-equivalent) — H2: composition, not inheritance.

Merges the wrapper-inheritance tower
(`ZiwagoGenesisDreamOrganismV18` -> `ZiwagoVerifiedOrganismV19` -> `ZiwagoOrganismV20`)
into a single flat class that composes cortex/hippocampus/basal-ganglia and the
genesis/verified modules as independent attributes. Method bodies are copied
verbatim from the most-derived version of each method; there is NO wrapper
inheritance (base is `object` only).

NOTE: this module is named `organism_core` (not `organism`) because
`ziwago/organism/` already exists as a package (integrated.py, physiology.py);
a sibling `organism.py` module would collide with it.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import io
import math
import mmap
import os
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ziwago.cognition.language.content_form import ContentFormComposer, RoleFiller
from ziwago.cortex.genesis_engine import (
    BA10CounterfactualInterventionSimulator,
    BidirectionalMeaningClosureVerifier,
    CorticalRelationalWorldGraph,
    EmergentOntologicalRoleDiscovery,
    MDThalamusSynapticSpeechActRouter,
    SyntacticValenceWaveGenerator,
    ThetaGammaSpikingPhaseMultiplexer,
)
from ziwago.cortex.next16_core import (
    ConsciousnessMetricsDashboard, CrossDomainIsomorphismDreamer,
    DynamicNeuromodulatoryBCMGate, KnowledgeDistillery, TwoCompartmentLarkumMicrocircuit,
)
from ziwago.evaluation.gates import (
    MASTER_CKPT_REL_PATH, CheckpointCompatibilityError, resolve_master_checkpoint_path,
)
from ziwago.organism.integrated import IntegrationConfig
from ziwago.organism.physiology import PhysiologyConfig
from ziwago.organism.flat import ZiwagoPureSynapticDialogueOrganismV14
from ziwago.app.curriculum import FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH
from ziwago.organism.sleep import execute_sws_synaptic_homeostasis_and_pruning
from ziwago.organism.verified import (
    EmpiricalConfidenceCalibration, SourceGroundingAuthority, IndependentHypothesisSleep,
    file_sha256, _document_qa_from_state, validate_v19_envelope, validate_verified_envelope,
    V19_SCHEMA,
)


V20_SCHEMA = "ziwago.checkpoint.v20.0"
_READONLY_CHECKPOINT_MAPPING_LOCK = threading.RLock()


class CandidatePublishError(RuntimeError):
    pass


def _link_candidate_temp(temp: Path, destination: Path) -> None:
    try:
        os.link(temp, destination)
    except OSError as exc:
        raise CandidatePublishError(
            f"Could not publish candidate at {destination}; complete checkpoint retained at {temp}"
        ) from exc


@contextmanager
def _readonly_checkpoint_mapping():
    """Отображать файл чекпоинта как PAGE_READONLY, а не torch-Windows WRITECOPY.

    torch.load(mmap=True) на Windows вызывает UntypedStorage.from_file(shared=False),
    который создаёт PAGE_WRITECOPY-секцию: она коммитит ВЕСЬ файл (~4.5 ГБ) в Job
    commit независимо от объёма чтения. Поверх организма (~3.9 ГБ) это ~8.4 ГБ ->
    OOM или ACCESS_VIOLATION при нехватке commit на создание секции (замер diag6:
    WRITECOPY peak 7.12 ГБ; diag7/b1_readonly: READONLY peak 2.32/6.00 ГБ, без AV).

    Python mmap.ACCESS_READ даёт настоящую file-backed PAGE_READONLY-страницу
    (evictable, ~0 commit). torch.frombuffer разделяет память без копии и удерживает
    буфер живым через DataPtr, поэтому отображение переживает выход из контекста.
    Вызовы загрузчика сериализованы; подмена from_file восстанавливается после
    torch.load. Не запускайте посторонний torch.load параллельно с restore.
    """
    with _READONLY_CHECKPOINT_MAPPING_LOCK:
        original = torch.UntypedStorage.from_file
        keepalive = []

        def _from_file(filename, shared, size):
            with open(filename, "rb") as handle:
                mapping = mmap.mmap(handle.fileno(), size, access=mmap.ACCESS_READ)
            keepalive.append(mapping)
            return torch.frombuffer(mapping, dtype=torch.uint8).untyped_storage()

        torch.UntypedStorage.from_file = _from_file
        try:
            yield
        finally:
            torch.UntypedStorage.from_file = original


class ZiwagoOrganism:
    """Flat composition organism (v20-equivalent). Composes cortex/hippocampus/basal-ganglia
    and genesis/verified modules as independent attributes — NO wrapper inheritance (H2)."""

    def __init__(self, root: Path = ROOT, checkpoint_path: Optional[Path] = None,
                 workers: int = 4, cpu_threads: int = 8, seed: int = 42,
                 knowledge_dir: Optional[Path] = None, auto_load_checkpoint: bool = True,
                 expected_checkpoint_sha256: Optional[str] = None,
                 integration_config: IntegrationConfig | None = None,
                 physiology_config: PhysiologyConfig | None = None,
                 integrated: bool = True):
        # `integrated` больше НЕ выбирает контур: контур всегда биологический
        # (13.11.2026, ЕДИНЫЙ КОНТУР). Параметр оставлен для совместимости
        # вызовов; на что он влияет — только на предпочтительную СХЕМУ
        # сохранения/загрузки в очень старых сценариях.
        self._integrated = bool(integrated)
        # ── v18 __init__ body (lines 100-141); v18 auto-load block OMITTED ──
        self.root = Path(root)
        self.core_organism = ZiwagoPureSynapticDialogueOrganismV14(
            root=self.root,
            workers=workers,
            cpu_threads=cpu_threads,
            seed=seed,
            physiology_config=physiology_config,
        )
        self.core_organism.use_ngrams = False

        # 1. Двигатель синтаксической волны без pos_idx (0% pos_idx!)
        self.valence_wave_generator = SyntacticValenceWaveGenerator(d_vec=64, d_dend=8192, top_k_wta=32, seed=seed + 1801)

        # 2. Кристаллизующийся граф каузальных триплетов мира (E1 -> R -> E2) с собственным без-позиционным NMDA-декодером рёбер
        self.world_graph = CorticalRelationalWorldGraph(
            encode_fn=self.core_organism.encode_text,
            compute_vec64_fn=self.core_organism.compute_stimulus_vec64,
            get_vec64_fn=self.core_organism.get_vec64,
            decode_fn=self.core_organism.decode_token,
        )

        # 3. Самоорганизующаяся онтология ролей через дистрибутивную STDP (0% ручных списков!)
        self.emergent_ontology = EmergentOntologicalRoleDiscovery(n_bins=2048)

        # 4. Тета-Гамма спайковый фазовый мультиплексор (120 мс = 6 x 20 мс TTFS)
        self.theta_gamma_multiplexer = ThetaGammaSpikingPhaseMultiplexer(d_vec=64, n_gamma_slots=6, bins_per_gamma=20)
        self.content_form_composer = ContentFormComposer(d_vec=64, enabled=True, multiplexer=self.theta_gamma_multiplexer)
        self.content_form_role_dictionary: Dict[str, Dict[str, torch.Tensor]] = {}

        # 5. Верификатор двунаправленного замыкания смысла X -> Y_hat -> X_hat
        self.meaning_closure_verifier = BidirectionalMeaningClosureVerifier()

        # 6. Префронтальный Контрфактический Симулятор Интервенций BA10 (do(~X) => ~Y => ~Z)
        self.counterfactual_simulator = BA10CounterfactualInterventionSimulator()

        # 7. Нейро-Синаптический Таламический Маршрутизатор Речевых Актов (MD-Thalamus + Striatum D1/D2)
        self.thalamic_router = MDThalamusSynapticSpeechActRouter(
            encode_fn=self.core_organism.encode_text,
            compute_sdr_fn=self.core_organism.dendritic_plateau_readout._compute_cortical_sdr2048,
        )

        self._bootstrap_trajectories: List[Tuple[torch.Tensor, Sequence[int], float]] = []
        self.loaded_genesis_state_from_ckpt: bool = False
        self.loaded_checkpoint_info: Optional[Dict[str, object]] = None
        self.core_organism.external_think_resolver_fn = self._resolve_cortical_think_concept

        # ── v19 __init__ body (lines 386-422), incl. its own load dispatch ──
        self.version = "19.0"
        self.authority = SourceGroundingAuthority(self.core_organism, self.world_graph)
        self.knowledge_distillery = KnowledgeDistillery(knowledge_dir or self.root / "data" / "Knowledge")
        self.bcm_gate = DynamicNeuromodulatoryBCMGate()
        self.larkum_microcircuit = TwoCompartmentLarkumMicrocircuit()
        self.isomorphism_dreamer = CrossDomainIsomorphismDreamer(self.knowledge_distillery)
        self.consciousness_dashboard = ConsciousnessMetricsDashboard()
        self.confidence_calibration = EmpiricalConfidenceCalibration()
        self.sleep_engine = IndependentHypothesisSleep(self)
        self.document_ranker = None
        self.document_ranker_metadata = None
        self.evaluation_depth = 0
        self.baseline_sha256 = None
        self.core_organism.external_grounding_verifier_fn = self.authority.ground
        self.core_organism.external_think_resolver_fn = self._resolve_cortical_think_concept
        self.core_organism.online_learning_enabled = False
        # Configure the live driver before restoring a checkpoint. An explicit
        # config may replace the fresh driver, but must not replace restored state.
        self.core_organism.enable_integrated_organism(integration_config, physiology_config)
        # Clean Slate (H1): no master fallback. checkpoint_path=None → fresh organism
        # (v20 trains from scratch); the archived v18 master is never auto-loaded.
        if auto_load_checkpoint and checkpoint_path:
            path = Path(checkpoint_path)
            if not path.is_file():
                raise CheckpointCompatibilityError(f"Checkpoint missing: {path}")
            def restore_payload(payload):
                if payload.get("v20_schema_version") is not None:
                    self._load_v20_payload(path, payload)
                elif payload.get("v19_schema_version") == V19_SCHEMA:
                    self._load_v19_payload(path, payload)
                else:
                    self.core_organism.load_self_contained_checkpoint(path, preloaded_payload=payload)
                    state = payload.get("genesis_v18_state")
                    if isinstance(state, dict):
                        self.load_genesis_state_dict(state)
                    self.baseline_sha256 = expected_checkpoint_sha256 or file_sha256(path)
                    self.core_organism.dialogue_state.reset_session()
                    self.loaded_checkpoint_info = {
                        "migration": "legacy assertions quarantined",
                        "source_sha256": self.baseline_sha256,
                    }
                    self._register_foundation_sources()
                    self.ingest_parsed_knowledge()
            self._consume_checkpoint_payload(
                path, expected_checkpoint_sha256, restore_payload)

        # ── ЕДИНЫЙ КОНТУР (13.11.2026) ───────────────────────────────────────
        # Контур организма всегда биологический: таламус/TRN/AST/GW/критичность/
        # интероцепция/лимбика/гиппокамп включены при рождении ядра
        # (flat.py: ZiwagoPureSynapticDialogueOrganismV14.__init__ вызывает
        # enable_integrated_organism). Переключателя «v19 / v20 контур» больше
        # нет: остались только ДВА ФОРМАТА СОСТОЯНИЯ чекпоинта (v19_schema /
        # v20_schema), которые выбираются по содержимому payload, а не флагом.
        # Версия контура — всегда 20.0; версия схемы чекпоинта — отдельное поле.
        self.version = "20.0"
        self.core_organism.external_sleep_phase_fn = self._on_sleep_phase
        self.core_organism.life.sleep_report = getattr(self.core_organism.life, "sleep_report", {})

    # ── v18 methods (not overridden) ──────────────────────────────────────────
    def sync_emergent_ontology_into_role_binder(self, utterance: str) -> Dict[str, Tuple[str, float]]:
        """Единый контур самоорганизации (Unified STDP Ontology Fusion):
        Открывает новые ролевые аналогии по Закону Невязки Схемы в `EmergentOntologicalRoleDiscovery`
        и мгновенно регистрирует их в `ThetaGammaRoleFillerBinder.dynamic_stdp_role_map`."""
        disc = self.emergent_ontology.discover_emergent_role_bindings(utterance)
        if disc:
            self.core_organism.role_filler_binder.register_emergent_stdp_bindings(disc)
        return disc

    def train_wave_with_neuromodulator_reward(self, pairs, wave_idx: int = 1) -> Dict[str, object]:
        """Обучение волной + накопление ролевых связей ПО КОРПУСУ.

        До 2026-11-13 `sync_emergent_ontology_into_role_binder` не вызывался
        НИ ОДНОЙ раз в production-коде: ролевая карта
        (`role_filler_binder.dynamic_stdp_role_map`) оставалась пустой, а
        живые пути `extract_slot_bindings` (flat.py) всегда возвращали `{}`.
        То есть весь аппарат переноса каузальной рамы на новый заполнитель
        (`hollandaise` → `mayonnaise`) был инертен, хотя схемы онтологии
        регистрировались только из БУТСТРЕП-урок, а не из обучающего корпуса.

        Теперь после обучения каждой пары её текст регистрируется в
        `EmergentOntologicalRoleDiscovery` (схема + STDP-поля слов), а
        найденные связи — в биндер ролей. Это НЕ круговая подстановка:
        карта растёт по обучающим репликам и применяется к НОВОЙ реплике.
        """
        result = self.core_organism.train_wave_with_neuromodulator_reward(pairs, wave_idx=wave_idx)
        registered = 0
        schemas_added = 0
        decode_errors = []
        for pair_index, pair in enumerate(pairs):
            try:
                stim_text = self.core_organism.decode_tokens(list(pair.stimulus_ids))
                resp_text = self.core_organism.decode_tokens(list(pair.response_ids))
            except Exception as exc:  # noqa: BLE001 — failure is retained in wave telemetry
                decode_errors.append({
                    "pair_index": pair_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                continue
            if not stim_text or not resp_text:
                continue
            before = len(self.emergent_ontology.schema_records)
            disc = self.emergent_ontology.register_training_pair(stim_text, resp_text, lr=1.5)
            schemas_added += len(self.emergent_ontology.schema_records) - before
            if disc:
                self.core_organism.role_filler_binder.register_emergent_stdp_bindings(disc)
                registered += len(disc)
        result["role_bindings_registered"] = registered
        result["schema_records_added"] = schemas_added
        result["role_pair_decode_errors"] = decode_errors
        result["role_map_size"] = len(self.core_organism.role_filler_binder.dynamic_stdp_role_map)
        return result

    def loaded_genesis_state_from_lessons_or_ckpt(self) -> bool:
        return self.loaded_genesis_state_from_ckpt and len(self.valence_wave_generator.col_to_tok) > 0

    def _bootstrap_genesis_modules_from_lessons(self) -> None:
        """Кристаллизует базовые каузальные уроки в без-индексный генератор валентностей `SyntacticValenceWaveGenerator`,
        дистрибутивную STDP-онтологию `EmergentOntologicalRoleDiscovery` (по Закону Невязки Схемы)
        и граф мира `CorticalRelationalWorldGraph`."""
        batch_trajs: List[Tuple[torch.Tensor, Sequence[int], float]] = []
        need_schema_reg = len(self.emergent_ontology.schema_records) == 0
        for ep_lessons in FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH.values():
            for _, stim_str, resp_str, f_mod in ep_lessons:
                s_ids = self.core_organism.encode_text(stim_str)
                r_ids = self.core_organism.encode_text(resp_str)
                p_strip = " .,!?;:'()[]{}" + chr(34)
                stop_w = self.core_organism.CONTENT_STOP
                c_ids = [
                    tid for tid in s_ids
                    if len(self.core_organism.decode_token(tid).strip(p_strip)) >= 3
                    and self.core_organism.decode_token(tid).strip(p_strip).lower() not in stop_w
                ] or s_ids
                q_pole, _ = self.core_organism.dendritic_plateau_readout.clean_up_stimulus_in_ca3_attractor(
                    self.core_organism.compute_stimulus_vec64(c_ids),
                    q_sdr2048=self.core_organism.dendritic_plateau_readout._compute_cortical_sdr2048(c_ids, normalize=False),
                )
                batch_trajs.append((q_pole, r_ids, float(f_mod)))

                if need_schema_reg:
                    self.emergent_ontology.register_schema_pair(stim_str, resp_str, lr=1.5)

        self._bootstrap_trajectories = list(batch_trajs)
        self.valence_wave_generator.w_valence_dend_out.zero_()
        self.valence_wave_generator.imprint_batch_zero_pos(
            batch_trajectories=self._bootstrap_trajectories,
            get_vec64_fn=self.core_organism.get_vec64,
            epochs=14,
        )

    def export_genesis_state_dict(self) -> Dict[str, object]:
        """Экспортирует полное состояние всех модулей `Ziwago-Genesis v18.1` (100% независимость от `seed`!)."""
        return {
            "valence_wave_generator": self.valence_wave_generator.export_state_dict(),
            "world_graph": self.world_graph.export_state_dict(),
            "emergent_ontology": self.emergent_ontology.export_state_dict(),
            "thalamic_router": self.thalamic_router.export_state_dict(),
            "dynamic_stdp_role_map": dict(self.core_organism.role_filler_binder.dynamic_stdp_role_map),
            "content_form_role_dictionary": {
                role: {label: vec.detach().cpu().tolist() for label, vec in slots.items()}
                for role, slots in self.content_form_role_dictionary.items()
            },
        }

    def load_genesis_state_dict(self, state: Dict[str, object]) -> None:
        """Восстанавливает полное состояние модулей `Ziwago-Genesis v18.1`."""
        if "valence_wave_generator" in state:
            self.valence_wave_generator.load_state_dict(state["valence_wave_generator"])  # type: ignore[arg-type]
        if "world_graph" in state:
            self.world_graph.load_state_dict(state["world_graph"])  # type: ignore[arg-type]
        if "emergent_ontology" in state:
            self.emergent_ontology.load_state_dict(state["emergent_ontology"])  # type: ignore[arg-type]
        if "thalamic_router" in state and isinstance(state["thalamic_router"], dict):
            self.thalamic_router.load_state_dict(state["thalamic_router"])  # type: ignore[arg-type]
        if "dynamic_stdp_role_map" in state and isinstance(state["dynamic_stdp_role_map"], dict):
            self.core_organism.role_filler_binder.dynamic_stdp_role_map = {
                str(k): str(v) for k, v in state["dynamic_stdp_role_map"].items()
            }
        if "content_form_role_dictionary" in state and isinstance(state["content_form_role_dictionary"], dict):
            restored: Dict[str, Dict[str, torch.Tensor]] = {}
            for role, slots in state["content_form_role_dictionary"].items():
                if not isinstance(slots, dict):
                    continue
                restored[str(role)] = {str(label): torch.as_tensor(vec, dtype=torch.float32)
                                      for label, vec in slots.items()}
            self.content_form_role_dictionary = restored
            self.core_organism.role_filler_binder.role_dictionary = restored

    def load_genesis_checkpoint(
        self,
        path: Path,
        strict_schema: bool = True,
        *,
        expected_checkpoint_sha256: Optional[str] = None,
        trusted_legacy_pickle: bool = False,
    ) -> Dict[str, object]:
        """Загружает единый самодостаточный чекпоинт ядра `v16.2` и всех 5 столпов `Genesis v18.1` за один вызов `torch.load`!"""
        ckpt_p = resolve_master_checkpoint_path(explicit_path=Path(path))
        if trusted_legacy_pickle and expected_checkpoint_sha256 is None:
            raise CheckpointCompatibilityError(
                "Trusted legacy pickle loading requires an expected SHA-256."
            )

        def restore_payload(payload):
            val_info = self.core_organism.load_self_contained_checkpoint(
                ckpt_p,
                strict_schema=strict_schema,
                preloaded_payload=payload,
            )
            gen_state = payload.get("genesis_v18_state")
            if isinstance(gen_state, dict):
                self.load_genesis_state_dict(gen_state)
                self.loaded_genesis_state_from_ckpt = True
            self.loaded_checkpoint_info = val_info
            return val_info

        return self._consume_checkpoint_payload(
            ckpt_p,
            expected_checkpoint_sha256,
            restore_payload,
            weights_only=not trusted_legacy_pickle,
            trusted_legacy_pickle=trusted_legacy_pickle,
        )

    # ── v19 methods (not overridden by v20) ───────────────────────────────────
    def _rem_dream(self) -> Dict[str, object]:
        """REM: сравнение выученных графов между собой.

        Сновидение работает по NAMED-графам: без регистрации графов в dreamer
        сон был пустым (в живом пути `register_graph` не вызывался ни разу, то
        есть REM не с чем было сравнивать). Здесь графы мира раскладываются по
        семействам-источникам и сравниваются попарно: структурное сходство
        разных областей знания — это и есть содержание REM.

        ГРАНИЦЫ ЖЁСТКИЕ, и это не косметика: выравнивание в dreamer —
        перебор с возвратом, экспоненциальный по числу узлов. Без ограничений
        фаза сна съедала бы всё время прогона (замерено: зависание на графах
        реального размера). Поэтому: не более 8 графов, не более 6 узлов в
        графе, не более 24 пар, узлы отбираются детерминированно.
        """
        max_graphs, max_nodes, max_pairs = 8, 6, 24
        dreamer = self.isomorphism_dreamer
        edges = list(getattr(self.world_graph, "edges", []) or [])
        families: Dict[str, list] = {}
        for edge in edges:
            if str(getattr(edge, "status", "")) not in ("asserted", "verified"):
                continue
            family = str(getattr(edge, "source", "") or "world").split("|")[0]
            families.setdefault(family, []).append(edge)
        registered = 0
        selected_families = sorted(families, key=lambda family: (-len(families[family]), family))
        skipped_families = max(0, len(selected_families) - max_graphs)
        edge_cap_rejections = 0
        registration_rejections = 0
        for family in selected_families[:max_graphs]:
            family_edges = families[family]
            if len(family_edges) < 2:
                continue
            node_types: Dict[str, str] = {}
            dream_edges = []
            for edge in family_edges:
                subject = str(edge.subject)
                target = str(edge.target)
                new_nodes = {subject, target} - node_types.keys()
                if len(node_types) + len(new_nodes) > max_nodes:
                    edge_cap_rejections += 1
                    continue
                node_types[subject] = str(getattr(edge, "subject_type", "entity") or "entity")
                node_types[target] = str(getattr(edge, "target_type", "entity") or "entity")
                dream_edges.append({
                    "subject": subject, "object": target, "relation": str(edge.relation),
                    "polarity": str(getattr(edge, "polarity", 1)),
                    "causal_kind": str(getattr(edge, "causal_kind", "association")),
                    "type": "association",
                })
            if len(dream_edges) < 2:
                continue
            try:
                dreamer.register_graph(family, dream_edges, node_types)
                registered += 1
            except ValueError:
                registration_rejections += 1
        analogies = []
        names = sorted(dreamer.graphs)[:max_graphs]
        pairs = 0
        for i, source in enumerate(names):
            for target in names[i + 1:]:
                if pairs >= max_pairs:
                    break
                pairs += 1
                match = dreamer.match_graphs(source, target)
                if match and not match.get("ambiguous"):
                    analogies.append({"source": source, "target": target,
                                       "mapping": match["mapping"], "score": match["score"]})
        return {"graphs_registered": registered, "families_skipped_by_graph_cap": skipped_families,
                "edges_skipped_by_node_cap": edge_cap_rejections, "node_cap": max_nodes,
                "graph_registration_rejections": registration_rejections,
                "pairs_compared": pairs,
                "analogies_found": len(analogies), "analogies": analogies[:8]}

    def _register_foundation_sources(self) -> None:
        for lessons in FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH.values():
            for lesson in lessons:
                _, prompt, response, _ = lesson
                self.authority.add_record(prompt, response, "bundled_foundation", status="asserted")

    def ingest_parsed_knowledge(self) -> dict:
        if self.evaluation_depth:
            raise RuntimeError("Knowledge ingestion is forbidden during evaluation.")
        records, edges = 0, 0
        for item in self.knowledge_distillery.records:
            source = f"parsed_knowledge:{item['source']}:{item['source_hash']}:{item['line']}"
            provenance = {k: item[k] for k in ("record_id", "source", "source_hash", "line", "kind")}
            self.authority.add_record(f"What does {item['section']} state?", item["content"], source,
                                      status="asserted", provenance=provenance)
            records += 1
            edges += len(self.world_graph.extract_and_consolidate_sentence(item["content"], source=source, status="asserted"))
        return {"source_records": records, "graph_edges": edges, "neural_training_performed": False}

    def ingest_scientific_corpus_into_world_graph(self, sentences: Sequence[str], dopamine_gain: float = 2.0,
                                                source: str = "local_corpus") -> int:
        if self.evaluation_depth:
            raise RuntimeError("Corpus ingestion is forbidden during evaluation.")
        return sum(len(self.world_graph.extract_and_consolidate_sentence(
            sentence, dopamine_gain=dopamine_gain, source=source, status="asserted")) for sentence in sentences)

    def _resolve_cortical_think_concept(self, utterance: str) -> Optional[dict]:
        graph = self.world_graph.synthesize_novel_sentence_from_graph(utterance)
        if graph is None or not self.authority.ground(utterance, graph["synthesized_sentence"]).get("grounded"):
            return None
        chain = graph["triplet_chain"]
        return {"known_in_cortex": True, "query_concept_x": chain[0][0],
                "canonical_concept_y": chain[0][2], "direct_graph_sentence": graph["synthesized_sentence"],
                "think_trace": "Source-supported exact-entity graph traversal.", "sources": graph.get("sources", [])}

    @contextmanager
    def evaluation_context(self):
        self.evaluation_depth += 1
        state = self.export_v19_state()
        try:
            with self.core_organism.evaluation_context():
                try:
                    yield self
                finally:
                    self.load_v19_state(state)
        finally:
            self.evaluation_depth -= 1

    def refresh_content_form_role_dictionary(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Build and cache the live `role -> {label -> vector}` dictionary from the learned role map.

        This is the live fallback vocabulary for `ContentFormComposer`: the role map is learned
        from corpus pairs and then materialized into the same vector space used by the composer.
        """
        def _encode_word(word: str) -> torch.Tensor:
            if not isinstance(word, str):
                return torch.zeros(64, dtype=torch.float32)
            token_ids = self.core_organism.encode_text(word.strip())
            if not token_ids:
                return torch.zeros(64, dtype=torch.float32)
            return self.core_organism.compute_stimulus_vec64(token_ids)

        vocab = self.core_organism.role_filler_binder.build_role_dictionary(_encode_word, d_vec=64)
        self.content_form_role_dictionary = vocab
        return vocab

    def maybe_fallback_compose(self, prompt: str, response_text: str, grounded: bool) -> Dict[str, object]:
        """Content-form fallback: only when the neural answer is not grounded.

        This keeps exact-match grounded responses untouched and exposes the fallback decision in
        the telemetry instead of silently fabricating form.
        """
        report: Dict[str, object] = {
            "fallback_used": False,
            "grounded": bool(grounded),
            "reason": "grounded_answer",
            "role_dictionary_size": 0,
            "role_slot_mapping": {},
            "unmapped_roles": [],
            "prompt": prompt,
        }
        if grounded:
            return report

        vocab = self.refresh_content_form_role_dictionary()
        report["role_dictionary_size"] = len(vocab)
        if not vocab:
            report["reason"] = "empty_role_dictionary"
            return report

        bindings = self.core_organism.role_filler_binder.extract_slot_bindings(prompt)
        if not bindings:
            report["reason"] = "no_role_bindings"
            return report

        slot_names = self.content_form_composer.ROLE_SLOT_NAMES[:self.content_form_composer.multiplexer.n_gamma_slots]
        canonical_vocab: Dict[str, Dict[str, torch.Tensor]] = {}
        role_slot_mapping = {}
        unmapped_roles = set()
        for idx, role in enumerate(sorted(vocab.keys())):
            if idx >= len(slot_names):
                unmapped_roles.add(role)
                continue
            slot_name = slot_names[idx]
            canonical_vocab[slot_name] = dict(vocab[role])
            role_slot_mapping[role] = slot_name

        fillers: List[RoleFiller] = []
        for role, label in sorted(bindings.items()):
            vec = vocab.get(role, {}).get(label)
            if vec is None:
                continue
            slot_name = role_slot_mapping.get(role)
            if slot_name is None:
                unmapped_roles.add(role)
                continue
            fillers.append(RoleFiller(role=slot_name, label=label, vector=vec, source="learned_role_map"))
        report["role_slot_mapping"] = role_slot_mapping
        report["unmapped_roles"] = sorted(unmapped_roles)
        if not fillers:
            report["reason"] = "role_slot_limit_exceeded" if unmapped_roles else "role_vector_missing"
            return report

        def _render(mapping: Dict[str, str]) -> str:
            if not mapping:
                return response_text
            return " | ".join(f"{role}={mapping.get(role, '')}" for role in sorted(mapping))

        composed = self.content_form_composer.compose(fillers, canonical_vocab, _render)
        if composed is None:
            report["reason"] = "composer_disabled"
            return report
        if composed.get("composed"):
            report.update({
                "fallback_used": True,
                "reason": "content_form_fallback",
                "composition_mode": composed.get("composition_mode"),
                "content": composed.get("content"),
                "text": composed.get("text"),
            })
        else:
            report.update({
                "fallback_used": False,
                "reason": composed.get("reason", "composer_refused"),
                "availability_fraction": composed.get("availability_fraction"),
                "recovery_cosine": composed.get("recovery_cosine"),
            })
        return report

    def generate_response(self, prompt: str, max_tokens: int = 64, **kwargs) -> dict:
        discoveries = self.emergent_ontology.discover_emergent_role_bindings(prompt)
        bindings = {canonical: filler for canonical, (filler, _) in discoveries.items()}
        result = self.core_organism.pure_neural_generate_response(
            prompt, max_tokens=max_tokens, use_ngrams=False, role_filler_bindings=bindings, **kwargs)
        result["emergent_stdp_role_discoveries"] = discoveries
        grounded = bool(result.get("grounded")) or bool(result.get("is_grounded_conclusion"))
        result["content_form_fallback"] = self.maybe_fallback_compose(
            prompt,
            str(result.get("response_text", "")),
            grounded=grounded,
        )
        result["fallback_used"] = bool(result["content_form_fallback"].get("fallback_used"))
        return result

    def generate_genesis_response(self, user_utterance: str, max_tokens: int = 64, **kwargs) -> dict:
        return self.generate_response(user_utterance, max_tokens=max_tokens)

    def load_document_ranker(self, path: Path) -> dict:
        if self.evaluation_depth:
            raise RuntimeError("Replacing document ranker weights is forbidden during evaluation.")
        snapshot = Path(path).read_bytes()
        payload = torch.load(io.BytesIO(snapshot), map_location="cpu", weights_only=True)
        ranker, metadata = _document_qa_from_state({"ranker": payload["ranker"], "metadata": payload["metadata"]})
        metadata["ranker_checkpoint_sha256"] = hashlib.sha256(snapshot).hexdigest()
        self.document_ranker, self.document_ranker_metadata = ranker, metadata
        return {"sha256": metadata["ranker_checkpoint_sha256"], "parameters": sum(p.numel() for p in ranker.model.parameters()),
                "support_threshold": ranker.support_threshold, "training_performed": False}

    def export_v19_state(self) -> dict:
        document_qa = ({"ranker": self.document_ranker.export_state_dict(),
                        "metadata": copy.deepcopy(self.document_ranker_metadata)} if self.document_ranker is not None else None)
        return {"document_qa": document_qa, "authority": self.authority.export_state_dict(), "bcm_gate": self.bcm_gate.export_state_dict(),
                "larkum": self.larkum_microcircuit.export_state_dict(),
                "analogies": self.isomorphism_dreamer.export_state_dict(),
                "temporal": self.theta_gamma_multiplexer.export_state_dict(),
                "calibration": self.confidence_calibration.export_state_dict(),
                "genesis": self.export_genesis_state_dict(), "baseline_sha256": self.baseline_sha256,
                "version": self.version}

    def load_v19_state(self, state: dict) -> None:
        ranker, metadata = _document_qa_from_state(state.get("document_qa"))
        self.authority.load_state_dict(state["authority"])
        self.bcm_gate.load_state_dict(state["bcm_gate"])
        self.larkum_microcircuit.load_state_dict(state["larkum"])
        self.isomorphism_dreamer.load_state_dict(state["analogies"])
        self.theta_gamma_multiplexer.load_state_dict(state["temporal"])
        self.confidence_calibration.load_state_dict(state["calibration"])
        self.load_genesis_state_dict(state["genesis"])
        self.baseline_sha256 = state.get("baseline_sha256")
        self.document_ranker, self.document_ranker_metadata = ranker, metadata

    def save_genesis_checkpoint(self, path: Path, extra_metadata: Optional[dict] = None, use_fp16: bool = False) -> dict:
        return self.save_candidate(path, extra_metadata, use_fp16)

    def _load_v19_payload(self, path: Path, payload: dict) -> None:
        validate_v19_envelope(payload)
        self.core_organism.load_self_contained_checkpoint(path, preloaded_payload=payload)
        self.load_v19_state(payload["v19_state"])
        self.loaded_checkpoint_info = {"schema": V19_SCHEMA, "certification": payload.get("certification")}

    def _consume_checkpoint_payload(
        self,
        path: Path,
        expected_checkpoint_sha256: Optional[str],
        consumer: Callable[[dict], object],
        *,
        weights_only: bool = True,
        trusted_legacy_pickle: bool = False,
    ):
        path = Path(path)
        if not callable(consumer):
            raise TypeError("consumer must be callable.")
        if not weights_only and (
            not trusted_legacy_pickle or expected_checkpoint_sha256 is None
        ):
            raise CheckpointCompatibilityError(
                "Unrestricted pickle loading requires explicit trust and an expected SHA-256."
            )

        if expected_checkpoint_sha256 is None:
            # READONLY-mmap avoids materializing the full checkpoint in RAM.
            with _readonly_checkpoint_mapping():
                payload = torch.load(
                    path, map_location="cpu", weights_only=weights_only, mmap=True)
            try:
                return consumer(payload)
            finally:
                payload = None
                gc.collect()

        expected = str(expected_checkpoint_sha256).lower()
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            raise CheckpointCompatibilityError("Expected checkpoint SHA-256 must be 64 hex characters.")

        snapshot_dir = self.root / "Produce"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = None
        payload = None
        try:
            digest = hashlib.sha256()
            with tempfile.NamedTemporaryFile(
                dir=snapshot_dir, suffix=".pt", delete=False
            ) as snapshot:
                snapshot_path = Path(snapshot.name)
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                        snapshot.write(chunk)
            if digest.hexdigest() != expected:
                raise CheckpointCompatibilityError(
                    "Checkpoint changed before loading the evaluation snapshot.")
            # Load the exact file whose bytes were hashed; never reopen the source path.
            with _readonly_checkpoint_mapping():
                payload = torch.load(
                    snapshot_path, map_location="cpu", weights_only=weights_only, mmap=True)
            return consumer(payload)
        finally:
            payload = None
            gc.collect()
            if snapshot_path is not None and snapshot_path.exists():
                snapshot_path.unlink()

    def load_v19_checkpoint(self, path: Path, expected_checkpoint_sha256: Optional[str] = None) -> dict:
        path = Path(path)
        self._consume_checkpoint_payload(
            path, expected_checkpoint_sha256,
            lambda payload: self._load_v19_payload(path, payload))
        return self.loaded_checkpoint_info

    # ── v20 methods (most-derived) ────────────────────────────────────────────
    def step(self, sensory_input=None, dt_ms=1.0):
        return self.core_organism.step(sensory_input, dt_ms)

    def _on_sleep_phase(self, old_phase, new_phase):
        core = self.core_organism
        life = core.life
        if new_phase == "nrem":
            life.sleep_report = {"started_ms": core.chemistry.time_ms}
        if old_phase == "nrem" and new_phase == "rem":
            readout = core.dendritic_plateau_readout
            with torch.no_grad():
                before = float(readout.w_dend_out.norm())
                active = len(readout.col_to_tok)
                readout.w_dend_out[:, :active].mul_(0.992)
                after = float(readout.w_dend_out.norm())
            life.sleep_report.update({"sws_homeostasis": {"norm_before": before, "norm_after": after,
                                                         "downscale_factor": 0.992},
                "ca3_apoptosis_repair": core.execute_sws_ca3_apoptotic_pole_sanitization()})
            # Hippocampal sleep reports SWR and REM status in the live transition.
            # No production consumer currently defines a justified cortical_write
            # update for replayed episodes; keep SWR disabled rather than inventing
            # a training target. Hippocampus reports `no_cortical_write`.
            hippo = getattr(life, "hippo", None)
            if hippo is not None and hasattr(hippo, "sleep_cycle"):
                chem_s = core.chemistry.s
                life.sleep_report["hippocampus_sleep"] = hippo.sleep_cycle(
                    cortical_write=None,
                    adenosine=float(getattr(chem_s, "ado", 0.0)),
                    ach=float(getattr(chem_s, "ach", 0.0)),
                    rem=True)
            if active:
                life.sleep_report["slow_reservoir_pct"] = float(
                    readout.shield.consolidate_into_slow_reservoir(readout.w_dend_out, active))
            life.sleep_report["rem"] = self._rem_dream()
            life.sleep_report["rem_facts_promoted"] = False

    def sleep_cycle(self, wave_idx=0, downscale_factor=0.992, *, requested=True, max_ticks=None):
        # Единый контур: сон всегда интегрированный (SWS → REM через
        # `life`), легаси-сон без биологии больше не используется.
        if self.evaluation_depth or not self.core_organism.learning_enabled:
            raise RuntimeError("Sleep consolidation is forbidden during evaluation")
        if downscale_factor != 0.992:
            raise ValueError("v20 SWS downscale is fixed at the reported 0.992")
        core, life = self.core_organism, self.core_organism.life
        budget = life.cfg.max_sleep_ticks if max_ticks is None else max_ticks
        if not isinstance(budget, int) or isinstance(budget, bool) or not 0 < budget <= life.cfg.max_sleep_ticks:
            raise ValueError("Sleep tick budget must be a positive integer within max_sleep_ticks")
        start = core.chemistry.time_ms
        if core.chemistry.is_awake:
            if not requested:
                return {"wave_idx": wave_idx, "elapsed_ms": 0.0, "completed": True,
                        "requested": False, "phase": "awake", "budget_ticks": budget}
            core.chemistry.request_sleep()
            self._on_sleep_phase("awake", "nrem")
        for _ in range(budget):
            life.tick(core, torch.zeros(core.moe_cortex.n_in))
            if core.chemistry.is_awake:
                break
        return {**life.sleep_report, "wave_idx": wave_idx, "requested": requested,
                "elapsed_ms": core.chemistry.time_ms - start,
                "completed": core.chemistry.is_awake, "phase": core.chemistry.phase,
                "budget_ticks": budget, "budget_exhausted": not core.chemistry.is_awake,
                "sleep_pressure_after": core.chemistry.s.ado,
                "replay_events": life.replay_events, "replay_synapses": life.replay_synapses}

    def _sleep_cycle_legacy(self, wave_idx: int = 0, downscale_factor: float = 0.992) -> Dict[str, object]:
        """v19 sleep phase (integrated=False): SWS → consolidation → REM, no physiology.

        Verbatim v19 wrapper body; SWS comes from the extracted clean module
        `ziwago.organism.sleep` instead of the old v15 engine import.
        """
        core = self.core_organism
        telemetry: Dict[str, object] = {"wave_idx": int(wave_idx)}

        # ── SWS: гомеостаз и прунинг слабых синапсов ───────────────────────────
        telemetry["sws_homeostasis"] = execute_sws_synaptic_homeostasis_and_pruning(
            core, downscale_factor=downscale_factor)
        # ── SWS: ремонт апоптотических полюсов CA3 (A2/A4) ───────────────────
        telemetry["ca3_apoptosis_repair"] = core.execute_sws_ca3_apoptotic_pole_sanitization()
        # ── Консолидация защищённого ядра в базовую матрицу ───────────────────
        readout = core.dendritic_plateau_readout
        n_act = len(readout.col_to_tok)
        if n_act > 0:
            with torch.no_grad():
                base = readout.w_dend_out[:, :n_act]
                protected = readout.w_core_dend_out[:, :n_act]
                base.add_(0.08 * protected)
            telemetry["core_to_basal_transfer"] = True
            telemetry["slow_reservoir_pct"] = round(
                float(readout.shield.consolidate_into_slow_reservoir(readout.w_dend_out, n_act)), 4)
        # ── Сон снимает давление сна и возвращает бодрствование ──────────────
        chemistry = core.chemistry
        chemistry.s.ado = max(0.0, float(chemistry.s.ado) * 0.35)
        chemistry.is_awake = True
        telemetry["sleep_pressure_after"] = round(float(chemistry.s.ado), 4)
        # ── REM: сновидение по изоморфизмам выученных графов ───────────────────
        telemetry["rem"] = self._rem_dream()
        return telemetry

    def select_action(self, context, evidence):
        core, life = self.core_organism, self.core_organism.life
        context = torch.as_tensor(context).detach().float().flatten()
        evidence = torch.as_tensor(evidence).detach().float().flatten()
        if (context.numel() != life.n_input or evidence.numel() != 7
                or not bool(torch.isfinite(context).all()) or not bool(torch.isfinite(evidence).all())
                or bool((evidence < 0).any())):
            raise ValueError("Invalid action context or evidence")
        life.action_context = context.clone()
        life.action_evidence = evidence.clone()
        start = core.chemistry.time_ms
        for _ in range(life.cfg.decision_budget_ms):
            life.tick(core, context)
            if life.action_result["committed"] or not core.chemistry.is_awake:
                break
        return {**life.action_result, "elapsed_ms": core.chemistry.time_ms - start,
                "budget_ms": life.cfg.decision_budget_ms,
                "budget_exhausted": not life.action_result["committed"]}

    def reward(self, decision_id: int, outcome: float, *, learn=True):
        if not math.isfinite(outcome):
            raise ValueError("Outcome must be finite")
        return self.core_organism.life.settle(self.core_organism, decision_id, outcome, learn=learn)

    def save_candidate(self, path: Path, extra_metadata=None, use_fp16=False):
        # Единый контур: кандидат всегда сохраняется в схеме v20 (она несёт
        # и биологическое состояние `life`). Старая схема v19 читается, но
        # больше не порождается — иначе формат снова разойдётся с архитектурой.
        path = Path(path).resolve()
        if path == (self.root / MASTER_CKPT_REL_PATH).resolve() or "master" in path.name.lower():
            raise PermissionError("Candidate saving cannot overwrite any master")
        if path.exists():
            raise FileExistsError(path)
        if not path.parent.is_dir():
            raise FileNotFoundError(path.parent)
        payload, _ = self.core_organism.export_checkpoint_payload(use_fp16=use_fp16)
        payload["core_runtime_state"]["core"]["loaded_checkpoint_path"] = str(path)
        payload.update({"v20_schema_version": V20_SCHEMA, "v20_state": self.export_v19_state(),
                        "v20_metadata": extra_metadata or {}, "certification": "experimental_unaccepted"})
        validate_verified_envelope(payload, version=20)
        fd, temporary = tempfile.mkstemp(prefix=".v20-", suffix=".pt", dir=path.parent)
        os.close(fd)
        temp = Path(temporary)
        preserve_temp = False
        try:
            torch.save(payload, temp)
            with temp.open("r+b") as stream:
                os.fsync(stream.fileno())
            try:
                _link_candidate_temp(temp, path)
            except CandidatePublishError:
                preserve_temp = True
                raise
        finally:
            if not preserve_temp:
                temp.unlink(missing_ok=True)
        return {"path": str(path), "sha256": file_sha256(path), "schema": V20_SCHEMA, "accepted": False}

    def _save_candidate_legacy(self, path: Path, extra_metadata: Optional[dict] = None, use_fp16: bool = False) -> dict:
        """v19 candidate save (integrated=False): v19 envelope, restores loaded_checkpoint_path.

        Verbatim v19 wrapper body; helpers (V19_SCHEMA, validate_v19_envelope,
        file_sha256) come from ziwago.organism.verified.
        """
        path = Path(path).resolve()
        if path == (self.root / MASTER_CKPT_REL_PATH).resolve() or "master" in path.name.lower():
            raise PermissionError("Candidate saving cannot overwrite any master.")
        if path.exists():
            raise FileExistsError(path)
        if not path.parent.is_dir():
            raise FileNotFoundError(path.parent)
        fd, temporary = tempfile.mkstemp(prefix=".v19-", suffix=".pt", dir=path.parent)
        os.close(fd)
        temp = Path(temporary)
        previous_path = self.core_organism.loaded_checkpoint_path
        preserve_temp = False
        try:
            payload, _ = self.core_organism.export_checkpoint_payload(use_fp16=use_fp16)
            payload["core_runtime_state"]["core"]["loaded_checkpoint_path"] = str(path)
            payload.update({"v19_schema_version": V19_SCHEMA, "v19_state": self.export_v19_state(),
                            "v19_metadata": extra_metadata or {}, "certification": "experimental_unaccepted"})
            validate_v19_envelope(payload)
            torch.save(payload, temp)
            with temp.open("r+b") as stream:
                os.fsync(stream.fileno())
            try:
                _link_candidate_temp(temp, path)
            except CandidatePublishError:
                preserve_temp = True
                raise
        finally:
            self.core_organism.loaded_checkpoint_path = previous_path
            if not preserve_temp:
                temp.unlink(missing_ok=True)
        return {"path": str(path), "sha256": file_sha256(path), "schema": V19_SCHEMA, "accepted": False}

    def _load_v20_payload(self, path, payload):
        validate_verified_envelope(payload, version=20)
        core_state = payload["core_runtime_state"]["core"]
        if "life" not in core_state or "chemistry" not in core_state:
            raise CheckpointCompatibilityError("Missing v20 integrated state")
        self.core_organism.load_self_contained_checkpoint(path, preloaded_payload=payload)
        self.load_v19_state(payload["v20_state"])
        self.version = "20.0"
        self.loaded_checkpoint_info = {"schema": V20_SCHEMA, "certification": payload.get("certification")}

    def load_v20_checkpoint(self, path, expected_checkpoint_sha256=None):
        path = Path(path)
        self._consume_checkpoint_payload(
            path, expected_checkpoint_sha256,
            lambda payload: self._load_v20_payload(path, payload))
        return self.loaded_checkpoint_info