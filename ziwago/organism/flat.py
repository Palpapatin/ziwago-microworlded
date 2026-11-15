"""Плоский организм v14/v19 — слой ziwago/organism (B.2, устранение инверсии слоёв).

ZiwagoPureSynapticDialogueOrganismV14 + SixDialogCorporaStreamer +
_bpe_tokenize_raw_pairs_worker перенесены verbatim из
train_ziwago_v14_pure_synaptic_dialogue.py (этап B.2). Тренер v14 ре-экспортирует
их для обратной совместимости; ziwago/ больше не импортирует train_*.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from contextlib import contextmanager
import datetime
import gc
import gzip
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Dict, Iterator, List, Optional, Tuple

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from ziwago.organism.physiology import OrganismPhysiology
from ziwago.config import ChemistryCfg
from ziwago.core.sdr import CompositionalVSABPEEncoder
from ziwago.cortex.moe_cortex import (
    SparselyGatedCorticalMoE, export_runtime_paths, export_runtime_tree,
    load_runtime_paths, load_runtime_tree,
)
from ziwago.cortex.spiking_language import SpikingLanguageLoop
from ziwago.cortex.genesis_engine import ThetaGammaSpikingPhaseMultiplexer
from ziwago.cortex.next16_core import FusiZenkeCascadeMetaplasticityShield, CorticalColumnarSubBankRouter
from ziwago.cortex.son_reflection import SONReflectiveColumn
from ziwago.evaluation.gates import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
    compute_tokenizer_sha256,
    evaluate_grammar_and_artifacts,
    validate_self_contained_checkpoint,
)
from ziwago.app.tokenizer import _GPT2_PAT, _bpe_encode_piece, _init_bpe_worker
import ziwago.app.tokenizer as _gpt2_mod
from ziwago.cognition.language.predictive_readout import PredictiveCodingCorticalReadout
from ziwago.app.corpora import PairedDialogueStreamer
from ziwago.cognition.memory.role_filler import ThetaGammaRoleFillerBinder
from ziwago.cognition.dialogue import ConversationalDialogueState
from ziwago.cognition.language.syntax import BA44AttractorSyntaxBridge
from ziwago.cognition.language.readout import DendriticPlateauMatrixReadout
from ziwago.app.data import TokenizedDialogPair

def _bpe_tokenize_raw_pairs_worker(
    raw_batch: List[Tuple[str, str, str, float]]
) -> List[Tuple[str, List[int], List[int], float]]:
    """Выполняется параллельно на 32 ядрах CPU: кодирует пары `(source, stim_str, resp_str, quality)` в BPE ID."""
    vocab = _gpt2_mod._WORKER_VOCAB
    merges = _gpt2_mod._WORKER_MERGES
    cache = _gpt2_mod._WORKER_CACHE
    assert vocab is not None and merges is not None

    out: List[Tuple[str, List[int], List[int], float]] = []
    for src, sa, sb, q_hint in raw_batch:
        sa_clean = " ".join(sa.replace("User:", "").replace("Assistant:", "").strip().split())
        sb_clean = " ".join(sb.replace("User:", "").replace("Assistant:", "").strip().split())
        if not sa_clean or not sb_clean:
            continue
        if len(sb_clean.split()) > 42:
            continue

        ids_a: List[int] = []
        for p in _GPT2_PAT.findall(" " + sa_clean):
            ids_a.extend(_bpe_encode_piece(p, vocab, merges, cache))

        ids_b: List[int] = []
        for p in _GPT2_PAT.findall(" " + sb_clean):
            ids_b.extend(_bpe_encode_piece(p, vocab, merges, cache))

        if 4 <= len(ids_a) <= 64 and 5 <= len(ids_b) <= 56:
            out.append((src, ids_a, ids_b, float(q_hint)))
    return out


class SixDialogCorporaStreamer:
    """Потоково читает все 6 папок в `data/dialog/` (`ultrachat`, `oasst2`, `personachat`, `personamem`,
    `npc_dialogue`, `daily_dialog` + `cornell`) и параллельно токенизирует их на 32 ядрах CPU."""

    ALL_SOURCES = (
        "ultrachat",
        "oasst2",
        "personachat",
        "personamem",
        "npc_dialogue",
        "daily_dialog_cornell",
    )

    def __init__(self, root: Path, workers: int = 32):
        self.root = root
        self.dialog_dir = root / "data" / "dialog"
        self.workers = max(1, min(int(workers), os.cpu_count() or 32))

        self.vocab_path = root / "data" / "gpt2" / "vocab.json"
        self.merges_path = root / "data" / "gpt2" / "merges.txt"
        _init_bpe_worker(str(self.vocab_path), str(self.merges_path))

        self._pool = ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_init_bpe_worker,
            initargs=(str(self.vocab_path), str(self.merges_path)),
        )
        self._cornell_helper = PairedDialogueStreamer(root=root)
        self._cursors: Dict[str, int] = {k: 0 for k in self.ALL_SOURCES}

    def _sample_ultrachat(self, n_pairs: int) -> List[Tuple[str, str, str, float]]:
        uc_dir = self.dialog_dir / "ultrachat" / "data"
        files = sorted(uc_dir.glob("*.parquet")) if uc_dir.exists() else []
        if not files:
            return []
        out: List[Tuple[str, str, str, float]] = []
        f_idx = self._cursors["ultrachat"] % len(files)
        pf = pq.ParquetFile(files[f_idx], memory_map=True)
        skip_b = (self._cursors["ultrachat"] // len(files)) % 25
        self._cursors["ultrachat"] += 1

        for b_i, batch in enumerate(pf.iter_batches(batch_size=128, columns=["messages"])):
            if b_i < skip_b:
                continue
            for msgs in batch.column("messages").to_pylist():
                if not msgs or len(msgs) < 2:
                    continue
                for m_a, m_b in zip(msgs, msgs[1:]):
                    if m_a.get("role") == "user" and m_b.get("role") == "assistant":
                        ca = str(m_a.get("content", "")).strip()
                        cb = str(m_b.get("content", "")).strip()
                        if 15 <= len(ca) <= 320 and len(cb) >= 20:
                            out.append(("ultrachat", ca, cb, 1.25))
                            if len(out) >= n_pairs:
                                return out
        return out

    def _sample_oasst2(self, n_pairs: int) -> List[Tuple[str, str, str, float]]:
        oasst_file = self.dialog_dir / "oasst2" / "2023-11-05_oasst2_ready.trees.jsonl.gz"
        if not oasst_file.exists():
            return []
        out: List[Tuple[str, str, str, float]] = []
        start_line = self._cursors["oasst2"]

        def _extract_tree_pairs(node: dict) -> None:
            if len(out) >= n_pairs:
                return
            if node.get("lang") == "en" and node.get("role") == "prompter":
                p_txt = str(node.get("text", "")).strip()
                for rep in node.get("replies", []) or []:
                    if rep.get("lang") == "en" and rep.get("role") == "assistant":
                        r_txt = str(rep.get("text", "")).strip()
                        if len(p_txt) >= 12 and len(r_txt) >= 15:
                            out.append(("oasst2", p_txt, r_txt, 1.35))
                            if len(out) >= n_pairs:
                                return
                        for child in rep.get("replies", []) or []:
                            _extract_tree_pairs(child)
            else:
                for rep in node.get("replies", []) or []:
                    _extract_tree_pairs(rep)

        with gzip.open(oasst_file, "rt", encoding="utf-8", errors="ignore") as f:
            for l_i, ln in enumerate(f):
                if l_i < start_line:
                    continue
                self._cursors["oasst2"] = l_i + 1
                try:
                    tree = json.loads(ln)
                    if "prompt" in tree:
                        _extract_tree_pairs(tree["prompt"])
                except Exception:
                    continue
                if len(out) >= n_pairs:
                    break
            else:
                self._cursors["oasst2"] = 0
        return out

    def _sample_personachat(self, n_pairs: int) -> List[Tuple[str, str, str, float]]:
        pc_file = self.dialog_dir / "personachat" / "data" / "train-00000-of-00001.parquet"
        if not pc_file.exists():
            return []
        out: List[Tuple[str, str, str, float]] = []
        pf = pq.ParquetFile(pc_file, memory_map=True)
        skip_b = self._cursors["personachat"] % 20
        self._cursors["personachat"] += 1

        for b_i, batch in enumerate(pf.iter_batches(batch_size=64, columns=["utterances"])):
            if b_i < skip_b:
                continue
            for utt_list in batch.column("utterances").to_pylist():
                if not utt_list:
                    continue
                for u_item in utt_list:
                    hist = u_item.get("history", [])
                    cands = u_item.get("candidates", [])
                    if hist and cands:
                        stim = str(hist[-1]).strip()
                        resp = str(cands[-1]).strip()  # в PersonaChat последний кандидат — истинный ответ!
                        if len(stim) >= 10 and len(resp) >= 10:
                            out.append(("personachat", stim, resp, 1.15))
                            if len(out) >= n_pairs:
                                return out
        return out

    def _sample_personamem(self, n_pairs: int) -> List[Tuple[str, str, str, float]]:
        pm_file = self.dialog_dir / "personamem" / "shared_contexts_32k.jsonl"
        if not pm_file.exists():
            pm_file = self.dialog_dir / "personamem" / "shared_contexts_128k.jsonl"
        if not pm_file.exists():
            return []
        out: List[Tuple[str, str, str, float]] = []
        start_l = self._cursors["personamem"]
        with pm_file.open("r", encoding="utf-8", errors="ignore") as f:
            for l_i, ln in enumerate(f):
                if l_i < start_l:
                    continue
                self._cursors["personamem"] = l_i + 1
                try:
                    d = json.loads(ln)
                    msgs = next(iter(d.values()))
                    for m_a, m_b in zip(msgs, msgs[1:]):
                        if m_a.get("role") == "user" and m_b.get("role") == "assistant":
                            ca = str(m_a.get("content", "")).strip()
                            cb = str(m_b.get("content", "")).strip()
                            if len(ca) >= 15 and len(cb) >= 15:
                                out.append(("personamem", ca, cb, 1.20))
                                if len(out) >= n_pairs:
                                    return out
                except Exception:
                    continue
            else:
                self._cursors["personamem"] = 0
        return out

    def _sample_npc_dialogue(self, n_pairs: int) -> List[Tuple[str, str, str, float]]:
        npc_file = self.dialog_dir / "npc_dialogue" / "rpg-quests-dialogue.jsonl"
        if not npc_file.exists():
            return []
        out: List[Tuple[str, str, str, float]] = []
        start_l = self._cursors["npc_dialogue"]
        with npc_file.open("r", encoding="utf-8", errors="ignore") as f:
            for l_i, ln in enumerate(f):
                if l_i < start_l:
                    continue
                self._cursors["npc_dialogue"] = l_i + 1
                try:
                    obj = json.loads(ln)
                    msgs = obj.get("messages", [])
                    for m_a, m_b in zip(msgs, msgs[1:]):
                        if m_a.get("role") == "user" and m_b.get("role") == "assistant":
                            ca = str(m_a.get("content", "")).strip()
                            cb = str(m_b.get("content", "")).strip()
                            if len(ca) >= 6 and len(cb) >= 12:
                                out.append(("npc_dialogue", ca, cb, 1.10))
                                if len(out) >= n_pairs:
                                    return out
                except Exception:
                    continue
            else:
                self._cursors["npc_dialogue"] = 0
        return out

    def _sample_daily_dialog_cornell(self, n_pairs: int) -> List[Tuple[str, str, str, float]]:
        pairs = self._cornell_helper.stream_paired_dialogues(n_pairs=n_pairs)
        return [("daily_dialog_cornell", p.stimulus_text, p.response_text, 1.15) for p in pairs]

    def stream_mixed_dialogue_wave(
        self,
        total_pairs: int = 3000,
    ) -> Tuple[List[TokenizedDialogPair], Dict[str, int], float]:
        """Собирает сбалансированную волну диалоговых пар из всех 6 источников `data/dialog/`
        и токенизирует их параллельно на 32 ядрах CPU."""
        t0 = time.perf_counter()
        per_src = max(1, total_pairs // 6)

        raw_all: List[Tuple[str, str, str, float]] = []
        raw_all.extend(self._sample_ultrachat(per_src))
        raw_all.extend(self._sample_oasst2(per_src))
        raw_all.extend(self._sample_personachat(per_src))
        raw_all.extend(self._sample_personamem(per_src))
        raw_all.extend(self._sample_npc_dialogue(per_src))
        raw_all.extend(self._sample_daily_dialog_cornell(total_pairs - len(raw_all)))

        random.shuffle(raw_all)
        chunk_sz = max(1, (len(raw_all) + self.workers - 1) // self.workers)
        batches = [raw_all[i : i + chunk_sz] for i in range(0, len(raw_all), chunk_sz)]
        tok_lists = list(self._pool.map(_bpe_tokenize_raw_pairs_worker, batches))

        pairs_out: List[TokenizedDialogPair] = []
        counts: Dict[str, int] = {k: 0 for k in self.ALL_SOURCES}
        for sub in tok_lists:
            for src, ids_a, ids_b, q_h in sub:
                pairs_out.append(TokenizedDialogPair(src, ids_a, ids_b, q_h))
                counts[src] = counts.get(src, 0) + 1

        return pairs_out, counts, round(time.perf_counter() - t0, 3)


def drop_legacy_shield_trace(payload: Dict[str, object]) -> Dict[str, object]:
    """Не материализовать legacy-трассу щита (семантика `magnitude_v1`).

    Старая трасса — матрица (d_dend, n_cols): 372.8 МБ на чекпоинте v19 и
    столько же приватной памяти при восстановлении. В новой семантике
    (`reinforcement_v2`) она всё равно СБРАСЫВАЕТСЯ при миграции (старая
    величина веса в новой семантике означала бы «заморожено по старому
    правилу»), поэтому копировать её в память бессмысленно: подменяем
    нулевым вектором на строку ДО восстановления.

    Итог: минус ~373 МБ приватной памяти на пике загрузки чекпоинта,
    семантика результата не меняется (трасса и так обнуляется).
    """
    report: Dict[str, object] = {"dropped": 0, "bytes_saved": 0}

    def visit(node) -> None:
        if isinstance(node, dict):
            for key, value in list(node.items()):
                if key == "u_slow" and isinstance(value, torch.Tensor) and value.ndim == 2:
                    rows = int(value.shape[0])
                    node[key] = torch.zeros((rows,), dtype=value.dtype)
                    report["dropped"] = int(report["dropped"]) + 1
                    report["bytes_saved"] = int(report["bytes_saved"]) + int(
                        value.numel() * value.element_size())
                else:
                    visit(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                visit(item)

    visit(payload)
    return report


_REDUNDANT_RUNTIME_PAYLOAD_FIELDS = (
    "col_to_tok", "tok_to_col", "w_ca3_episode_links", "shield_state",
    "bank_router_state", "router_residual_bound", "vsa_encoder_state",
    "moe_cortex_state", "son_motor_state", "chemistry_state",
    "w_ba44_bridge", "son_w_basal", "son_w_apical", "w_stimulus_to_onset",
    "w_readout", "token_embed", "conversational_state", "spiking_language_state",
    "language_memory_records", "w_dend_out", "w_core_dend_out", "w_swr_dend_out",
    "w_ca3_attractors", "w_ca3_sdr", "w_ca3_salience", "sdr_column_usage",
)


def release_redundant_runtime_payload_fields(payload: Dict[str, object]) -> Dict[str, int]:
    """Release flat legacy copies when the full runtime snapshot is authoritative.

    The positional/branch/episode fields used by ``apply_checkpoint_overlay`` are
    intentionally retained. This avoids dropping compatibility state while
    releasing other top-level aliases before restoration starts.
    """
    if not isinstance(payload.get("core_runtime_state"), dict):
        return {"fields": 0}
    removed = 0
    for key in _REDUNDANT_RUNTIME_PAYLOAD_FIELDS:
        if key in payload:
            del payload[key]
            removed += 1
    if removed:
        gc.collect()
    return {"fields": removed}


# ============================================================================
# 3. ОРГАНИЗМ С ЧИСТОЙ НЕЙРОННОЙ ГЕНЕРАЦИЕЙ ИЗ СИНАПСОВ + ПЕРЕКЛЮЧАТЕЛЬ N-ГРАММ (ON / OFF)
# ============================================================================
class ZiwagoPureSynapticDialogueOrganismV14:
    """v14-compatible core with v19 local cortical language learning.

    Tokens are synthesized from learned synapses and actual MoE activity. The
    separate language_memory_records ledger contains observed training pairs
    with provenance; it is not a sentence-retrieval decoding path. Grounding is
    verified independently of cortical confidence, and evaluation is read-only.
    """

    eos_token_id = 50256

    CONTENT_STOP: frozenset = frozenset({
        "the", "a", "an", "to", "of", "in", "on", "is", "are", "was", "were",
        "and", "or", "for", "with", "that", "this", "it", "do", "can", "you",
        "what", "how", "why", "where", "when", "who", "tell", "me", "about",
        "am", "my", "has", "become", "now", "have", "been", "until", "into",
        "from", "yes", "one", "not", "but", "does", "he", "remind", "please", "fix",
    })

    PRONOUN_INVERSION: Dict[str, str] = {
        "i": "you",
        "i'm": "you are",
        "am": "are",
        "my": "your",
        "mine": "yours",
        "me": "you",
        "myself": "yourself",
    }

    def __init__(self, root: Path, workers: int = 32, cpu_threads: int = 32, seed: int = 42,
                 physiology_config=None, integration_config=None):
        torch.set_num_threads(max(1, int(cpu_threads)))
        torch.manual_seed(seed)
        random.seed(seed)
        self.root = root
        self.seed = int(seed)

        vocab_path = root / "data" / "gpt2" / "vocab.json"
        merges_path = root / "data" / "gpt2" / "merges.txt"
        _init_bpe_worker(str(vocab_path), str(merges_path))

        # 1. Стример всех 6 диалоговых корпусов в data/dialog/
        self.dialog_streamer = SixDialogCorporaStreamer(root=root, workers=workers)

        # 2. VSA-BPE энкодер (50,257 токенов GPT-2) + 16 колонок MoE (16,384 нейрона) + 17-я колонка SON (1,024 нейрона)
        self.vsa_encoder = CompositionalVSABPEEncoder(
            vocab_path=str(vocab_path),
            merges_path=str(merges_path),
            n_base=2048,
            n_sdr=2048,
            n_modules=8,
            active_per_module=5,
            d_out=64,
            seed=seed,
        )
        self.moe_cortex = SparselyGatedCorticalMoE(
            n_columns=16,
            neurons_per_column=1024,
            n_in=64,
            n_branch_per_neuron=7,
            top_k_active=2,
            seed=seed,
        )
        self.son_column = SONReflectiveColumn(
            n_son_neurons=1024,
            n_observed_columns=16,
            d_vec=64,
            seed=seed + 1400,
        )

        # 3. Единый владелец химии (B.3): OrganismPhysiology создаётся ОДИН раз при
        # рождении и больше никогда не заменяется. integrated=True/False — конфигурация
        # контура (enable_integrated_organism), а не подмена объекта (потеря состояния).
        self.chemistry = OrganismPhysiology(self.moe_cortex.n_columns, cfg=physiology_config,
                                            chemistry_cfg=ChemistryCfg())

        # 4. Матричный декодер коры Predictive Coding [256, 50257] + Нелинейный Дендритный Плато-Декодер [8192, V_act] (для режима --ngrams off!)
        self.pc_readout = PredictiveCodingCorticalReadout(vocab_size=50257, d_token=64, d_state=256, seed=seed)
        self.dendritic_plateau_readout = DendriticPlateauMatrixReadout(
            d_vec=64, d_dend=8192, max_pos=int(self.language_position_window), seed=seed + 1405)
        self.use_ngrams: bool = False  # Чистая Матрица Дендритных Плато 8192 NMDA-ветвей по умолчанию (0% N-грамм!)
        self.dialogue_state = ConversationalDialogueState(d_vec=64)
        self.role_filler_binder = ThetaGammaRoleFillerBinder()
        self.last_ca3_top1_resonance: float = 1.0
        self.last_multi_hop_used: bool = False
        self.tokenizer_sha256: str = compute_tokenizer_sha256(vocab_path, merges_path)

        g = torch.Generator().manual_seed(seed + 1401)
        # Матрица инициации ответа (каким первым/вторым словом начать ответ на вектор стимула Q(A))
        self.w_stimulus_to_onset = F.normalize(torch.randn(64, 50257, generator=g), p=2, dim=0) * 0.1
        self.onset_pair_synapses: Dict[int, Dict[int, float]] = {}  # first_tok -> {second_tok: conductance}
        self.onset_pair_ctx_vec: Dict[Tuple[int, int], torch.Tensor] = {}

        # 5. Чисто синаптические проводимости переходов (БЕЗ хранения предложений или doc_uid!)
        self.synaptic_bigram_g: Dict[int, Dict[int, float]] = {}
        self.synaptic_trigram_g: Dict[Tuple[int, int], Dict[int, float]] = {}
        # Средний вектор вопроса Q(A) \in R^64, при котором данный синаптический переход получал дофаминовое подкрепление:
        self.synaptic_trigram_ctx: Dict[Tuple[int, int, int], torch.Tensor] = {}
        self.synaptic_fourgram_g: Dict[Tuple[int, int, int], Dict[int, float]] = {}

        # Построитель декодирования отдельных BPE-токенов
        with open(vocab_path, "r", encoding="utf-8") as vf:
            v_map = json.load(vf)
        self._id_to_token_raw: Dict[int, str] = {int(v): str(k) for k, v in v_map.items()}

        self._vec64_cache: Dict[int, torch.Tensor] = {}
        for bid in range(2048):
            self._vec64_cache[bid] = self.vsa_encoder.project_token_to_cortex(bid)

        # Синтаксический Мост Зоны Брока BA44 между аттракторами CA3 (Col0_SYNTAX_BA44 Syntax Bridge)
        self.ba44_syntax_bridge = BA44AttractorSyntaxBridge(d_vec=64, seed=seed + 1444)
        self.ba44_syntax_bridge.initialize_synaptic_connectors(
            encode_fn=self.encode_text,
            decode_fn=self.decode_token,
            get_vec64_fn=self.get_vec64,
            vocab_map=v_map,
        )

        self.total_pairs_trained: int = 0
        self.total_tokens_trained: int = 0
        self.cumulative_da_reward: float = 0.0
        self.spiking_language_loop = SpikingLanguageLoop(self.moe_cortex, seed=seed + 19)
        self.language_memory_records: List[Dict[str, object]] = []
        self.external_grounding_verifier_fn = None
        self.learning_enabled = True
        self._evaluation_depth = 0
        self.dialogue_writes_enabled = True
        self.online_learning_enabled = True
        self.temporal_codec = ThetaGammaSpikingPhaseMultiplexer(n_gamma_slots=1, bins_per_gamma=81)
        self.temporal_mode = "normal"
        self.temporal_generator = torch.Generator().manual_seed(seed + 1901)
        self.last_temporal_report = {}

        # ── ЕДИНЫЙ КОНТУР (13.11.2026) ─────────────────────────────────────
        # Биология (таламус FO/HO, TRN, AST/GW, мета-голова, активная
        # критичность, интероцепция, лимбика, гиппокамп) — не отдельный
        # «режим v20», а часть организма С РОЖДЕНИЯ. Раньше было два контура
        # на одном субстрате: v19 шагал кору «голой»
        # (`moe_cortex.step(x_in=x, f_mod=1.0, learn=False)` — без таламуса,
        # без TRN, без пластичности в коре), а v20 — через IntegratedOrganism.
        # Теперь контур один и всегда биологический; версия чекпоинта
        # (v19/v20) различает только ФОРМАТ состояния, а не способ шагать.
        self.enable_integrated_organism(integration_config, physiology_config)

    @property
    def spiking_language(self):
        return self.spiking_language_loop

    def export_runtime_state(self) -> Dict[str, object]:
        """Data-only full core snapshot for checkpoints and transactional evaluation."""
        import numpy as np
        excluded = {"root", "dialog_streamer", "moe_cortex", "spiking_language_loop",
                     "restore_migrations"}
        core = {k: v for k, v in vars(self).items() if k not in excluded and not k.startswith("external_") and not callable(v)}
        np_state = np.random.get_state()
        return {
            "runtime_version": 19,
            "core": export_runtime_tree(core),
            "moe_cortex": self.moe_cortex.export_state_dict(),
            "spiking_language_loop": self.spiking_language_loop.export_state_dict(),
            "rng": {
                "python": export_runtime_tree(random.getstate()),
                "torch": torch.get_rng_state().clone(),
                "cuda": [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
                "numpy": {"name": np_state[0], "keys": torch.from_numpy(np_state[1].astype(np.int64)),
                          "pos": np_state[2], "has_gauss": np_state[3], "cached_gaussian": np_state[4]},
            },
            "bpe_cache": export_runtime_tree(_gpt2_mod._WORKER_CACHE),
        }

    def load_runtime_state(self, state: Dict[str, object],
                           remove_missing_attributes: bool = False) -> None:
        import numpy as np
        if state.get("runtime_version") != 19:
            raise ValueError("Unsupported core runtime snapshot")
        excluded = {"root", "dialog_streamer", "moe_cortex", "spiking_language_loop",
                     "restore_migrations"}
        core_state = state["core"]
        # Checkpoint loads preserve newer fields; evaluation rollback opts into strict cleanup.
        for key, value in list(vars(self).items()):
            if (key not in core_state and (key.startswith("_last_") or remove_missing_attributes)
                    and key not in excluded
                    and not key.startswith("external_") and not callable(value)):
                delattr(self, key)
        memo = {}
        for key, val in core_state.items():
            setattr(self, key, load_runtime_tree(
                getattr(self, key, None), val, memo, _reuse_current_tensors=True,
                _remove_missing_attributes=remove_missing_attributes))
        if hasattr(self, "dendritic_plateau_readout"):
            router = getattr(self.dendritic_plateau_readout, "bank_router", None)
            if router is not None and not hasattr(router, "prune_threshold"):
                router.prune_threshold = 1e-4
        self.moe_cortex.load_state_dict(
            state["moe_cortex"], remove_missing_attributes=remove_missing_attributes)
        self.spiking_language_loop.load_state_dict(
            state["spiking_language_loop"], remove_missing_attributes=remove_missing_attributes)
        self.spiking_language_loop.moe_cortex = self.moe_cortex
        rng = state["rng"]
        random.setstate(load_runtime_tree(None, rng["python"]))
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])
        ns = rng["numpy"]
        np.random.set_state((ns["name"], ns["keys"].cpu().numpy().astype(np.uint32),
                             ns["pos"], ns["has_gauss"], ns["cached_gaussian"]))
        cache = load_runtime_tree(_gpt2_mod._WORKER_CACHE, state["bpe_cache"])
        _gpt2_mod._WORKER_CACHE.clear()
        _gpt2_mod._WORKER_CACHE.update(cache)

    def export_language_dynamic_state(self) -> Dict[str, object]:
        import numpy as np
        np_state = np.random.get_state()
        son_paths = [(name,) for name in (
            "current_stimulus_vec", "running_thought_vec", "v_son", "r_son", "epistemic_drive",
            "prev_free_energy", "total_son_steps", "total_motor_stops_triggered",
            "total_vte_simulations", "total_dissonance_interventions", "elig_son",
        )]
        son_paths.extend(("gw", name) for name in (
            "x", "gate", "ignition", "last_coherence", "energy_hist", "telemetry",
        ))
        son_paths.extend(("meta_head", name) for name in (
            "w_out", "b_out", "elig", "gw_centroid", "h_state", "trace_gw", "err_hist", "conf_hist",
        ))
        return {
            "runtime_version": 19,
            "core": export_runtime_paths(self, [(name,) for name in (
                "last_cortical_trajectory_trace", "last_reafferent_stranger_interventions",
                "last_arcuate_token_injections", "last_temporal_report", "temporal_generator",
            )]),
            "moe_cortex": self.moe_cortex.export_dynamic_state_dict(),
            "spiking_language_loop": self.spiking_language_loop.export_dynamic_state_dict(),
            "son_column": export_runtime_paths(self.son_column, son_paths),
            "organism": export_runtime_paths(self, [("life",), ("chemistry",),
                ("son_column", "ast"), ("son_column", "organism_reflection")]) if hasattr(self, "life") else None,
            "temporal_codec": export_runtime_paths(self.temporal_codec, [
                ("last_encode_report",), ("last_decode_report",),
            ]),
            "rng": {
                "python": export_runtime_tree(random.getstate()),
                "torch": torch.get_rng_state().clone(),
                "cuda": [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
                "numpy": {"name": np_state[0], "keys": torch.from_numpy(np_state[1].astype(np.int64)),
                          "pos": np_state[2], "has_gauss": np_state[3], "cached_gaussian": np_state[4]},
            },
        }

    def load_language_dynamic_state(self, state: Dict[str, object]) -> None:
        import numpy as np
        if state.get("runtime_version") != 19:
            raise ValueError("Unsupported language dynamic snapshot")
        load_runtime_paths(self, state["core"])
        self.moe_cortex.load_dynamic_state_dict(state["moe_cortex"])
        self.spiking_language_loop.load_dynamic_state_dict(state["spiking_language_loop"])
        load_runtime_paths(self.son_column, state["son_column"])
        if state.get("organism") is not None:
            load_runtime_paths(self, state["organism"])
        load_runtime_paths(self.temporal_codec, state["temporal_codec"])
        rng = state["rng"]
        random.setstate(load_runtime_tree(None, rng["python"]))
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])
        ns = rng["numpy"]
        np.random.set_state((ns["name"], ns["keys"].cpu().numpy().astype(np.uint32),
                             ns["pos"], ns["has_gauss"], ns["cached_gaussian"]))

    @contextmanager
    def evaluation_context(self):
        """Read-only evaluation, including nested contexts and exceptional exits.

        Suppress local plasticity and evidence/role writes *during* evaluation;
        restore all dynamic compartments, telemetry, caches and RNG afterwards.
        """
        import inspect
        snapshot = self.export_runtime_state()
        patched = []
        visited = set()
        mutators = ("learn", "train_", "imprint_", "consolidate", "structural_plasticity",
                    "promote_", "register_", "record_turn", "link_sequential", "protect_rows",
                    "apply_sparse_update", "apply_metaplasticity", "inject_apical_error", "local_stdp")

        def disable(obj):
            if id(obj) in visited:
                return
            visited.add(id(obj))
            if isinstance(obj, dict):
                for child in obj.values():
                    disable(child)
                return
            if isinstance(obj, (list, tuple)):
                for child in obj:
                    disable(child)
                return
            if not hasattr(obj, "__dict__") or isinstance(obj, torch.Tensor):
                return
            module = type(obj).__module__
            if not (module.startswith("ziwago.") or module.startswith("train_ziwago_")):
                return
            for child in list(vars(obj).values()):
                if not callable(child):
                    disable(child)
            if hasattr(obj, "learning_enabled"):
                obj.learning_enabled = False
            # Intrinsic Triesch plasticity and routing homeostasis have no learn
            # argument in the legacy neuron/router APIs; freeze their learned buffers.
            frozen = ()
            step_name = "step"
            if type(obj).__name__ == "USNeuronPool":
                frozen = ("r_ip", "ip_gain", "theta_base")
            elif type(obj).__name__ == "MetaHead":
                frozen = ("w_out", "b_out", "elig", "gw_centroid")
            elif isinstance(obj, SparselyGatedCorticalMoE):
                frozen = ("homeostatic_expert_theta",)
                step_name = "route_top_k_columns"
            if frozen:
                original = getattr(obj, step_name)
                had_instance = step_name in vars(obj)
                patched.append((obj, step_name, had_instance, vars(obj).get(step_name)))

                def read_only_step(*args, _obj=obj, _original=original, _frozen=frozen, **kwargs):
                    saved = {k: getattr(_obj, k).clone() for k in _frozen if hasattr(_obj, k)}
                    try:
                        return _original(*args, **kwargs)
                    finally:
                        for k, val in saved.items():
                            setattr(_obj, k, val)

                setattr(obj, step_name, read_only_step)
            for name in dir(type(obj)):
                if name.startswith(mutators) and callable(getattr(obj, name, None)):
                    had_instance = name in vars(obj)
                    previous = vars(obj).get(name)
                    patched.append((obj, name, had_instance, previous))
                    setattr(obj, name, lambda *args, **kwargs: None)
                elif callable(getattr(obj, name, None)):
                    method = getattr(obj, name)
                    try:
                        signature = inspect.signature(method)
                    except (TypeError, ValueError):
                        continue
                    if "learn" in signature.parameters:
                        had_instance = name in vars(obj)
                        patched.append((obj, name, had_instance, vars(obj).get(name)))

                        def no_learning_call(*args, _method=method, _signature=signature, **kwargs):
                            bound = _signature.bind_partial(*args, **kwargs)
                            bound.arguments["learn"] = False
                            return _method(*bound.args, **bound.kwargs)

                        setattr(obj, name, no_learning_call)

        try:
            # Traverse components, not the core methods (training's guarded return contract remains intact).
            for key, component in list(vars(self).items()):
                if key not in ("dialog_streamer", "root") and not callable(component):
                    disable(component)
            self.learning_enabled = False
            self._evaluation_depth = getattr(self, "_evaluation_depth", 0) + 1
            with torch.no_grad():
                yield self
        finally:
            for obj, name, had_instance, previous in reversed(patched):
                if had_instance:
                    setattr(obj, name, previous)
                else:
                    delattr(obj, name)
            self.load_runtime_state(snapshot, remove_missing_attributes=True)

    def _record_dialogue_turn(self, *args, **kwargs) -> None:
        if self.dialogue_writes_enabled and not self._evaluation_depth:
            self.dialogue_state.record_turn_and_poles(*args, **kwargs)

    def _transmit_temporal_vector(self, vector: torch.Tensor) -> torch.Tensor:
        if self.temporal_mode == "bypass":
            return vector
        if self.temporal_mode not in {"normal", "jitter", "drop", "shuffle"}:
            raise ValueError("Unknown temporal ablation mode")
        name = self.temporal_codec.ROLE_SLOT_NAMES[0]
        events, _ = self.temporal_codec.encode_roles_into_theta_spike_train({name: vector})
        if self.temporal_mode == "jitter":
            times, neurons = events.nonzero(as_tuple=True)
            noise = torch.randint(-2, 3, times.shape, generator=self.temporal_generator)
            events.zero_()
            events[(times + noise).clamp(0, events.shape[0] - 1), neurons] = 1
        elif self.temporal_mode == "drop":
            events *= torch.rand(events.shape, generator=self.temporal_generator) >= 0.3
        elif self.temporal_mode == "shuffle":
            events = events[torch.randperm(events.shape[0], generator=self.temporal_generator)]
        decoded, report = self.temporal_codec.decode_roles_from_theta_spike_train(events, return_report=True)
        self.last_temporal_report = {"mode": self.temporal_mode, "event_count": int(events.count_nonzero()),
                                     "availability_fraction": report["availability_fraction"], "loss_fraction": report["loss_fraction"]}
        return F.normalize(decoded[name].to(vector), dim=0)

    def decode_token(self, tid: int) -> str:
        return self.decode_tokens([tid])

    def decode_tokens(self, ids: List[int]) -> str:
        """Decode UTF-8 only after joining bytes, including split multibyte tokens."""
        raw = "".join(self._id_to_token_raw.get(int(tid), "") for tid in ids)
        token_bytes = bytearray()
        for char in raw:
            if char in _gpt2_mod._GPT2_BYTE_DECODER:
                token_bytes.append(_gpt2_mod._GPT2_BYTE_DECODER[char])
            else:
                # Literal ASCII whitespace in small raw-token test vocabularies.
                token_bytes.extend(char.encode("utf-8"))
        return token_bytes.decode("utf-8", errors="replace")

    def encode_text(self, text: str) -> List[int]:
        text = text.strip()
        if not text:
            return []
        assert _gpt2_mod._WORKER_VOCAB is not None and _gpt2_mod._WORKER_MERGES is not None
        ids: List[int] = []
        for p in _GPT2_PAT.findall(" " + text):
            ids.extend(
                _bpe_encode_piece(
                    p,
                    _gpt2_mod._WORKER_VOCAB,
                    _gpt2_mod._WORKER_MERGES,
                    _gpt2_mod._WORKER_CACHE,
                )
            )
        return ids

    def get_vec64(self, tid: int) -> torch.Tensor:
        t = int(tid)
        if t in self._vec64_cache:
            return self._vec64_cache[t]
        v = self.vsa_encoder.project_token_to_cortex(t)
        if len(self._vec64_cache) < 50257:
            self._vec64_cache[t] = v
        return v

    def compute_stimulus_vec64(self, token_ids: List[int]) -> torch.Tensor:
        """Интегрирует входные токены реплики собеседника с ацетилхолиновым (ACh) фокусом на содержательных словах."""
        if not token_ids:
            return self.get_vec64(262)
        stop_words = {
            "the", "a", "an", "to", "of", "in", "on", "is", "are", "was", "were",
            "and", "or", "for", "with", "that", "this", "it", "do", "can", "you",
            "what", "how", "why", "where", "when", "who", "tell", "me", "about",
        }
        p_strip = " .,!?;:'()[]{}" + chr(34)
        weighted_sum = torch.zeros(64)
        total_w = 0.0

        for pos, tid in enumerate(token_ids):
            v = self.get_vec64(tid)
            w_str = self.decode_token(tid).strip(p_strip).lower()
            # Редкие и именные слова получают высокий холинергический вес ACh (3.5x), служебные — низкий (0.25x)
            if len(w_str) >= 4 and w_str not in stop_words:
                ach_w = 3.5 * (1.0 + 0.15 * (pos / float(max(1, len(token_ids)))))
            else:
                ach_w = 0.25
            weighted_sum = weighted_sum + ach_w * v
            total_w += ach_w

        analog = F.normalize(weighted_sum / max(1e-5, total_w), p=2, dim=-1)
        return self._transmit_temporal_vector(analog)

    def enable_integrated_organism(self, config=None, physiology_config=None):
        from ziwago.organism.integrated import IntegratedOrganism
        # Явно переданный конфиг → драйвер ПЕРЕСОЗДАЁТСЯ (иначе второй
        # вызов с другим конфигом молча игнорировался бы). Без конфига —
        # идемпотентно: уже созданный драйвер не трогаем.
        if hasattr(self, "life") and config is None:
            return
        if hasattr(self, "life"):
            del self.life
        # B.3: химия — единый владелец OrganismPhysiology, созданный при рождении.
        # Режим integrated конфигурирует контур (life/часы/обучение), но НЕ заменяет
        # объект химии: восстановленное из чекпоинта состояние сохраняется.
        # physiology_config принят для совместимости API; химия уже сконфигурирована
        # при рождении (см. __init__), поэтому здесь не пересоздаётся.
        self.chemistry.tick = self.moe_cortex.global_clock_tick
        self.chemistry.time_ms = float(self.chemistry.tick)
        self.life = IntegratedOrganism(self.moe_cortex.n_columns, self.moe_cortex.n_in,
                                       self.moe_cortex.neurons_per_column, self.seed, config)
        self.moe_cortex.integrated_clock_enabled = True
        for col in self.moe_cortex.columns:
            for syn in (col.w_ff, col.w_rec, col.w_fb):
                syn.learning_enabled = True
        self.life.migration = {"chemistry_reinitialized": False, "from_core_clock": self.chemistry.tick,
                               "knowledge_weights_preserved": True, "single_owner_from_birth": True}
        self.use_ngrams = False
        self.neuromodulation_enabled = True

    def _cortical_step(self, x):
        # ЕДИНЫЙ КОНТУР: шаг коры всегда идёт через биологический драйвер
        # (`life` = IntegratedOrganism: таламус → TRN → AST/GW → мета-голова →
        # критичность → реле → колонки). «Голый» путь v19
        # (`moe_cortex.step(learn=False)`) удалён: он не учил кору и не имел
        # таламуса, TRN и глобального рабочего пространства.
        if not hasattr(self, "life"):
            raise RuntimeError("The organism contour is missing its biological driver "
                               "(life). This is an invariant violation, not a mode switch.")
        return self.life.present(self, x)

    def step(self, sensory_input=None, dt_ms=1.0):
        if not hasattr(self, "life"):
            raise RuntimeError("The shared organism clock requires the v20 runtime")
        if (not math.isfinite(dt_ms) or dt_ms <= 0 or int(dt_ms) != dt_ms
                or dt_ms > self.life.cfg.max_step_ticks):
            raise ValueError("dt_ms must be an integer number of 1 ms neural steps within the tick budget")
        x = torch.zeros(self.moe_cortex.n_in) if sensory_input is None else sensory_input
        for _ in range(int(dt_ms)):
            self.life.tick(self, x)
        return self.life.telemetry(self)

    def _prepare_language_sequence(self, q_vec: torch.Tensor, clause_poles=None):
        poles = [q_vec] if clause_poles is None else clause_poles
        if not hasattr(self, "life"):
            self.moe_cortex.homeostatic_expert_theta.zero_()
        first = self._cortical_step(q_vec)
        self.last_cortical_trajectory_trace = [list(first.active_expert_names)]
        for pole in poles[1:]:
            out = self._cortical_step(pole)
            self.last_cortical_trajectory_trace.append(list(out.active_expert_names))
        motor = F.normalize(torch.roll(poles[-1], shifts=1, dims=-1), p=2, dim=-1)
        out = self._cortical_step(motor)
        if list(out.active_expert_names) != self.last_cortical_trajectory_trace[-1]:
            self.last_cortical_trajectory_trace.append(list(out.active_expert_names))
        return first

    def _prepare_language_query(
        self, prompt: str, q_raw: torch.Tensor, *, use_ngrams: bool = False,
        enable_multi_hop: bool = False, role_filler_bindings: Optional[Dict[str, str]] = None,
    ):
        poles = ([q_raw] if use_ngrams else self.compute_clause_attractor_poles(
            prompt, enable_multi_hop=enable_multi_hop, role_filler_bindings=role_filler_bindings))
        q_vec = poles[0]
        first = self._prepare_language_sequence(q_vec, poles)
        return q_vec, poles, first

    def _language_pc_logits(
        self, q_vec: torch.Tensor, prefix: List[int], base_logits: torch.Tensor, l5: torch.Tensor,
        candidate_tokens: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, List[int]]:
        ro = self.dendritic_plateau_readout
        pc_logits = torch.zeros_like(base_logits)
        candidate_ids: List[int] = []
        if prefix and base_logits.numel():
            top_cols = base_logits.topk(min(48, base_logits.numel())).indices
            tokens = ro.col_to_tok if candidate_tokens is None else candidate_tokens
            candidate_ids = [tokens[i] for i in top_cols.tolist()]
            pc = self.pc_readout.score_candidates(
                prev2_id=prefix[-3] if len(prefix) >= 3 else prefix[-2] if len(prefix) >= 2 else prefix[-1],
                prev1_id=prefix[-2] if len(prefix) >= 2 else prefix[-1],
                cur_id=prefix[-1], topic_vec64=q_vec, l5_vec64=l5, candidate_ids=candidate_ids)
            pc_logits[top_cols] = base_logits.new_tensor([2.2 * pc.get(t, 0.0) for t in candidate_ids])
        return pc_logits, candidate_ids

    # ── Арбитраж «префикс ↔ эпизод» (гиппокампально-неокортикальный гейт) ──
    # ВЫКЛЮЧЕН по умолчанию: поведение не меняется, пока флаг не включён явно.
    # Биологическая опора: Dudai & Marlat (1997) memory gating и Kumford et al.
    # (2007) hippocampal–neocortical gating — при расхождении входа и извлечённой
    # памяти приоритет решает ацетилхолин: высокий АЦх — режим кодирования
    # (вход принимается, порог терпимости выше), низкий — режим извлечения
    # (расхождение означает, что продолжать нечего).
    language_conflict_arbitration = False
    language_conflict_abstain_steps = 3
    # Порог и прирост порога от АЦх выбраны по ИЗМЕРЕННОМУ разделению
    # (v19_compose_conflict_diag3_20261109.json, 16 условий):
    #   на траектории (дословный префикс / перефраз вопроса) максимальная
    #   длина серии шагов выше порога = 1 при пороге 0.035 → арбитраж не срабатывает;
    #   вне траектории (перефраз ответа) = 13 → срабатывает всегда.
    # Диапазон 0.035…0.065 (нижняя граница зазора + вклад АЦх) даёт полное
    # разделение при любом состоянии.
    language_conflict_threshold_base = 0.035
    language_conflict_ach_gain = 0.03
    language_conflict_ach_full = 1.2

    # ── Сквозная нейромодуляция языкового контура ────────────────────────────
    # Модуляция в проекте уже реализована: chemistry.s (ACh/NE/DA/5-HT/окситоцин/
    # кортизол/аденозин), f_mod — трёхфакторный химический гейт, phi_cort — окно
    # пластичности. Но языковой контур её обходил: выбор токена не зависел от
    # состояния организма. Здесь химия включается в решение токена.
    # ВЫКЛЮЧЕНО по умолчанию — поведение обученной модели не меняется.
    # Сквозная нейромодуляция включена. Гейты НОРМИРОВАНЫ относительно опорного
    # состояния модели (`neuromodulatory_gains`): в опорной точке все гейи равны
    # 1.0, поэтому поведение обученной модели совпадает с выключенной
    # модуляцией, а ЛЮБОЕ отклонение состояния (стресс, сон, новизна, награда)
    # меняет генерацию. Это и есть «разные состояния → разная речь».
    neuromodulation_enabled = True

    # ── Апикальная гиппокампальная обратная связь в языковой контур ────────────
    # Связывает память и речь: аттрактор CA3, извлечённый по вопросу, возвращает
    # в кору содержание эпизода (через SDR-привязку полюса), и кора может
    # воспроизвести ответ даже когда локальный префикс ушёл с траектории.
    # ВЫКЛЮЧЕНО по умолчанию — поведение ранее обученных моделей не меняется.
    apical_feedback_enabled = False

    # Ёмкость позиционного кода ответа: сколько шагов позиции ридеаут вообще
    # различает. Исторически 64 — из этого следовал ВНЕШНИЙ обрез ответов до 63
    # токенов (куррикула, гейт допуска, max_response_tokens). Окно расширяемое:
    # при загрузке чекпоинта с меньшим окном позиции достраиваются, а не
    # урезаются. Нерасширенные позиции ещё не обучены — за них отвечает
    # тренировка, а не конструкция.
    # Переопределяется переменной окружения ZIWAGO_POSITION_WINDOW (например, 256),
    # чтобы окно менялось без правки кода; фактическое значение пишется в отчёты.
    language_position_window = int(os.environ.get("ZIWAGO_POSITION_WINDOW", "64") or 64)

    def measure_living_state(self) -> Dict[str, float]:
        """Живое состояние организма — источник входов нейрохимии.

        Нейрохимия не должна получать константы «по расписанию»: тогда она
        декорация, а не часть организма. Здесь каждый вход ИЗМЕРЕН:
          * activity — плотность спайков последнего растра (реальная активность);
          * spikes  — суммарное число спайков (расход энергии);
          * novelty — ошибка предсказания (сюрприз; NE реагирует на новизну);
          * attention — концентрация маршрутизации (ACh = внимание);
          * drive   — фактическая потребность в бодрствовании (орексин);
          * stress  — рассогласование ожидания и результата (кортизол).
        """
        loop = getattr(self, "spiking_language_loop", None)
        activity = 0.0
        spikes = 0.0
        novelty = 0.0
        attention = 0.0
        stress = 0.0
        if loop is not None:
            raster = getattr(loop, "last_spike_raster", None)
            if raster is not None and raster.numel():
                values = raster.float()
                spikes = float(values.sum())
                # Активность — доля вовлечённых колонок, а не счётчик спайков:
                # величина должна быть нормирована, иначе она насыщается.
                activity = min(1.0, float((values > 0).float().mean()))
                # Внимание — концентрация маршрутизации: доля спайков в трёх
                # самых активных колонках (размытость роутинга = низкое внимание).
                column_totals = values.sum(dim=0)
                total = float(column_totals.sum())
                if total > 0:
                    top = torch.topk(column_totals, min(3, column_totals.numel())).values
                    attention = min(1.0, float(top.sum()) / total)
            apical = getattr(loop, "last_apical_error", None)
            if apical is not None and apical.numel():
                novelty = min(1.0, float(apical.float().abs().mean()) / 1.5)
            loss = float(getattr(loop, "last_loss", 0.0) or 0.0)
            stress = min(1.0, max(0.0, loss - math.log(max(2, len(loop.token_to_col) or 2))))
        # Потребность в бодрствовании: чем выше активность и дофамин, тем
        # сильнее удерживается wake-аттрактор (орексин). Аденозин, в свою
        # очередь, ПОДАВЛЯЕТ орексинергические нейроны — поэтому давление сна
        # снижает потребность в бодрствовании, и wake-аттрактор может рухнуть
        # (Borbely two-process: именно так засыпание и наступает).
        drive = min(1.0, max(0.0, (0.6 * activity + 0.4 * float(self.chemistry.s.da))
                             * (1.0 - 0.9 * float(self.chemistry.s.ado))))
        return {"activity": activity, "spikes": min(1.0, spikes / 64.0), "novelty": novelty,
                "attention": attention, "drive": drive, "stress": stress}

    def modulator_bundle(self) -> Dict[str, float]:
        """Единый пакет модуляторов: химия и нейрон читают ОДНО и то же состояние.

        Это то, что превращает набор разрозненных слоёв в единый организм:
        пластичность, генерация и выбор действия читают один и тот же
        модуляторный вектор, а не каждый — свой скаляр.
        """
        if hasattr(self, "life"):
            return self.chemistry.bundle()
        s = self.chemistry.s
        return {
            "ach": float(s.ach), "ne": float(s.ne), "da": float(s.da),
            # Фазовая DA — знаковый импульс, а не тоническая. В v20 синапс
            # переводится в режим `externally_clocked`, где гейт DA берётся
            # ИМЕННО из знаковой фазы (0 — нет импульса, ± — предупреждающее
            # значение). Тоническая DA в этот слот означала бы, что фазовая
            # модуляция никогда не работает.
            "da_phasic": float(s.da),  # v19 (life is None): фазовых импульсов нет — тонический fallback, как в NeuroChemistry
            "ht": float(s.ht), "cort": float(s.cort),
            "ecb": float(s.ecb), "ado": float(s.ado), "ox": float(s.ox),
            "lactate": float(s.lactate), "no": float(s.no),
            "f_mod": float(self.chemistry.f_mod(energy_ok=1.0)),
            "p_release": float(self.chemistry.p_release()),
            "awake": bool(self.chemistry.is_awake),
        }

    def _raw_neuromod_gains(self) -> Dict[str, float]:
        """Абсолютные усиления контуров из текущего состояния химии.

        Биологические соответствия (уже принятые в проекте):
          ACh  ↑ — холинергическая активация коры и усиление NMDA-пластичности;
                  высокий ACh = режим кодирования, вход неокортекса доминирует;
          NE   ↑ — гейн-контроль и повышение сигнал/шум (в коре NE уже гасит
                  шум: ne_snr_damp в core/neuron.py);
          DA   ↑ — дофаминергическая gates пластичности и сигнал вознаграждения;
          5-HT ↑ — терпение/ожидание, снижает импульсивность выбора.
        """
        try:
            s = self.chemistry.s
            ach = float(getattr(s, "ach", 0.5))
            ne = float(getattr(s, "ne", 0.3))
            da = float(getattr(s, "da", 0.5))
            ht = float(getattr(s, "ht", 0.3))
        except Exception:
            ach, ne, da, ht = 0.5, 0.3, 0.5, 0.3
        # нормировка к номинальным значениям ChemState (da .5 / ne .3 / ach .5 / ht .3)
        ach_n = ach / 0.5
        ne_n = ne / 0.3
        da_n = da / 0.5
        ht_n = ht / 0.3
        return {
            # корковый (спайковый) канал: усиливается холинергически и норадреналином
            "cortical_gain": float(max(0.5, min(2.0, 0.55 * ach_n + 0.45 * ne_n))),
            # гиппокампальный (дендритный) канал: усиливается при низком ACh
            # (режим извлечения), когда вход неокортекса отодвинут
            "readout_gain": float(max(0.5, min(2.0, 1.0 / max(0.25, 0.35 * ach_n + 0.65)))),
            # порог решения: норадреналин (новизна) поднимает, 5-HT (терпение) снижает
            "threshold_shift": float(max(-0.02, min(0.02, 0.012 * ne_n - 0.008 * ht_n))),
            # дофаминергическая/серотонинергическая управляемая эксплуатация
            "exploit_bias": float(max(0.5, min(1.5, 0.6 * da_n + 0.4 * ht_n))),
        }

    def neuromodulatory_gains(self) -> Dict[str, float]:
        """Усиления относительно ОПОРНОГО состояния (химия на момент загрузки).

        Относительность нужна для честности: поведение обученной модели при
        включённой модуляции и опорном состоянии совпадает с выключенной,
        а любые отклонения состояния (стресс, сон, новизна, награда) меняют
        поведение — это и есть «разные состояния → разная генерация».
        """
        raw = self._raw_neuromod_gains()
        ref = getattr(self, "_neuromod_reference", None)
        if not ref:
            ref = raw
            self._neuromod_reference = dict(raw)  # первый вызов фиксирует «дом» модели
        gains = {}
        for key, value in raw.items():
            base = float(ref.get(key, value)) or 1.0
            if key == "threshold_shift":
                gains[key] = float(value - float(ref.get(key, 0.0)))
            else:
                gains[key] = float(value / base)
        return gains

    def _language_conflict_signal(self, logits: torch.Tensor) -> float:
        """Конфликт «входящий префикс ↔ извлечённый эпизод» через уверенность.

        Дендритный канал несёт извлечённый эпизод, корковый — входящий префикс.
        Когда они расходятся, распределение по токенам становится плоским: модель
        не знает, что сказать дальше. Тот же механизм (плоская salience) раньше
        выбирал неверный полюс. 0.0 — полная уверенность, 1.0 — полная неопределённость.
        """
        if logits is None or not logits.numel():
            return 1.0
        probs = logits.softmax(0)
        p1 = float(probs.max())
        n = int(logits.numel())
        uniform = 1.0 / max(1, n)
        confidence = (p1 - uniform) / max(1e-9, 1.0 - uniform)
        return max(0.0, min(1.0, 1.0 - confidence))

    def _language_conflict_threshold(self) -> float:
        """Порог конфликта, модулируемый состоянием организма.

        АЦх — режим кодирования/извлечения (Dudai & Marlat): высокий АЦх — вход
        неокортекса принимается, порог терпимости выше; низкий — расхождение
        означает, что продолжать нечего, пород строже. Дополнительно порог
        сдвигается норадреналином (новизна) и 5-HT (терпение) — то есть
        модуляция здесь та же, что и на всём остальном организме.
        """
        try:
            ach = float(getattr(self.chemistry.s, "ach", 0.5))
        except Exception:
            ach = 0.5
        ach_norm = max(0.0, min(1.0, ach / max(1e-9, float(self.language_conflict_ach_full))))
        threshold = float(self.language_conflict_threshold_base) + float(self.language_conflict_ach_gain) * ach_norm
        if bool(getattr(self, "neuromodulation_enabled", False)):
            threshold += float(self.neuromodulatory_gains().get("threshold_shift", 0.0))
        return threshold

    def _language_apical_context(self, q_vec: torch.Tensor) -> Optional[Dict[str, object]]:
        """Эпизод, извлечённый по вектору запроса (для апикальной обратной связи).

        Вектор запроса не меняется в течение генерации ответа, поэтому извлечение
        выполняется один раз и кэшируется по содержимому вектора.
        """
        if not bool(getattr(self, "apical_feedback_enabled", False)):
            return None
        ro = self.dendritic_plateau_readout
        key = hash(q_vec.detach().cpu().numpy().tobytes())
        cache = getattr(self, "_apical_episode_cache", None)
        if isinstance(cache, dict) and cache.get("key") == key:
            return cache
        pole_idx, resonance = ro.resolve_episode_pole(q_vec)
        trace = ro.episode_trace(pole_idx) if pole_idx is not None else []
        entry: Dict[str, object] = {
            "key": key, "pole_idx": pole_idx,
            "resonance": resonance, "trace": trace,
        }
        self._apical_episode_cache = entry
        return entry

    def _language_apical_cue(self, ctx: Dict[str, object], pos: int) -> Optional[torch.Tensor]:
        """«Куда двигаться»: элемент гиппокампальной траектории для позиции `pos`.

        pos == 0 → сам стимул (полюс CA3): переход «стимул → первый токен ответа»
                   (то же, что делает матрица `w_stimulus_to_onset`);
        pos > 0  → элемент траектории на позиции pos-1: переход «токен → следующий токен».
        """
        ro = self.dendritic_plateau_readout
        trace = ctx.get("trace") or []
        if pos == 0:
            pole_idx = ctx.get("pole_idx")
            if pole_idx is None or pole_idx >= int(ro.w_ca3_attractors.shape[0]):
                return None
            return ro.w_ca3_attractors[int(pole_idx)]
        if pos - 1 >= len(trace):
            return None
        tid = int(trace[pos - 1])
        if tid < 0:
            return None
        return self.get_vec64(tid)

    @staticmethod
    def _language_js_divergence(p_a: torch.Tensor, p_b: torch.Tensor) -> float:
        """Расхождение распределений коркового и гиппокампального свидетельств."""
        p_a = p_a.clamp_min(1e-12)
        p_b = p_b.clamp_min(1e-12)
        m = 0.5 * (p_a + p_b)
        kl_a = float((p_a * (p_a.log() - m.log())).sum())
        kl_b = float((p_b * (p_b.log() - m.log())).sum())
        return max(0.0, min(1.0, 0.5 * (kl_a + kl_b)))

    def _language_token_components(self, q_vec: torch.Tensor, prefix: List[int], pos: int,
                                   candidate_cols: Optional[List[int]] = None):
        ro = self.dendritic_plateau_readout
        previous = [q_vec if len(prefix) < lag else self.get_vec64(prefix[-lag]) for lag in (3, 2, 1)]
        basal_logits = ro.logit_scale * ro.predict_next_token_logits(*previous, pos, q_vec, candidate_cols=candidate_cols)
        token_ids = ro.col_to_tok if candidate_cols is None else [ro.col_to_tok[c] for c in candidate_cols]
        token_input = self._transmit_temporal_vector(F.normalize(q_vec + previous[-1], dim=0))
        cortical, out = self.spiking_language_loop.step(
            token_input, token_ids, learn=False,
            cortical_step=self._cortical_step if hasattr(self, "life") else None)
        cortical = cortical.to(basal_logits)
        # Конфликт измеряется на НЕМОДУЛИРОВАННЫХ потоках: это мера
        # эпистемической неопределённости (расхождение доказательств), и она
        # не должна зависеть от состояния организма — иначе калибровка порога
        # рассыпается при смене состояния.
        # Конфликт считается по КОРКОВОМУ свидетельству (базовый readout + спайки).
        # Апикальный (гиппокампальный) канал сюда не входит намеренно: иначе
        # память, разрешая неопределённость, маскировала бы сам конфликт и
        # существующая калибровка порога перестала бы быть измеренной величиной.
        conflict = self._language_conflict_signal(basal_logits + cortical)
        self._language_last_conflict = conflict
        # ── Апикальная гиппокампальная обратная связь ──────────────────────────
        # Гиппокамп говорит «куда двигаться» (элемент траектории эпизода), кора
        # решает «как сказать». Канал подключается ТОЛЬКО когда кора
        # неуверена: гиппокамп — это ФАЛБЭК, а не второй источник мнения.
        # Иначе он вмешивается в работу префикса и ломает верный replay
        # (измерено: 4/4 → 3/4 при постоянном гейте).
        # Биологическая опора: Dudai & Marlat (1997) — гиппокампальная система
        # вовлекается, когда неокортикальная перегружена; Kumford et al. (2007) —
        # переключение между гиппокампом и корой по надёжности свидетельств.
        d_apical = None
        apical_logits = None
        episode_conflict = None
        apical_gate = 0.0
        apical_ctx = self._language_apical_context(q_vec)
        if apical_ctx is not None:
            ramp = ((float(conflict) - float(ro.apical_conflict_gate_low))
                    / max(1e-6, float(ro.apical_conflict_gate_high) - float(ro.apical_conflict_gate_low)))
            ramp = max(0.0, min(1.0, ramp))
            if ramp > 0.0:
                cue = self._language_apical_cue(apical_ctx, pos)
                if cue is not None:
                    d_apical = ro.apical_episode_code(cue)
                    apical_raw = ro.apical_logits(d_apical, 1.0, candidate_cols=candidate_cols)
                    if apical_raw is not None:
                        raw = ro.logit_scale * apical_raw
                        episode_conflict = self._language_js_divergence(
                            (basal_logits + cortical).softmax(0), raw.softmax(0))
                        apical_gate = (ramp * float(ro.apical_gate_gain)
                                       * float(apical_ctx.get("resonance") or 0.0))
                        apical_logits = raw * apical_gate
        self._language_last_episode_conflict = episode_conflict
        self._language_last_apical_gate = apical_gate
        readout_logits = basal_logits if apical_logits is None else basal_logits + apical_logits
        # Сквозная нейромодуляция: состояние организма вмешивается в выбор токена.
        # Каналы разделены по происхождению: дендритный несёт извлечённый эпизод,
        # корковый — входящий префикс. Гейны задаются состоянием химии.
        if bool(getattr(self, "neuromodulation_enabled", False)):
            gains = self.neuromodulatory_gains()
            readout_logits = readout_logits * float(gains.get("readout_gain", 1.0))
            cortical = cortical * float(gains.get("cortical_gain", 1.0))
        l5 = F.normalize(F.adaptive_avg_pool1d(out.combined_l5_drive.mean(0).view(1, 1, -1), 64).flatten(), dim=0)
        pc_logits, candidates = self._language_pc_logits(q_vec, prefix, readout_logits + cortical, l5,
                                                         candidate_tokens=token_ids)
        return {"readout": readout_logits, "spiking": cortical, "pc": pc_logits}, out, l5, candidates

    # Candidate-first TDR: ВКЛЮЧЁН (13.11, единый контур). Раньше флаг был
    # False, и при обзорной рисовке TDR стоял в схеме живого ридеута — то есть
    # модуль был описан, покрыт тестами и не работал. Теперь индекс кандидатов
    # реально скорит только отобранные колонки (бюджет и пулы — в отчёте),
    # а полный путь остаётся как контрольный. Явный opt-out — `tdr_enabled=False`.
    tdr_enabled = True
    tdr_candidate_budget = 128
    tdr_pool_a = 64
    tdr_pool_b = 64

    def _tdr_index(self):
        from ziwago.cognition.language.tdr import TokenCandidateIndex
        idx = getattr(self, "_tdr_index_cache", None)
        if idx is None or idx.encoder is not self.vsa_encoder:
            idx = TokenCandidateIndex(self.vsa_encoder)
            self._tdr_index_cache = idx
        return idx

    def _language_token_distribution(self, q_vec: torch.Tensor, prefix: List[int], pos: int,
                                     use_tdr: bool = True):
        ro = self.dendritic_plateau_readout
        candidate_cols = None
        candidate_tokens = list(ro.col_to_tok)
        self._last_tdr_report = None
        if use_tdr and bool(getattr(self, "tdr_enabled", False)) and ro.col_to_tok:
            proposal = self._tdr_index().propose(
                q_vec, prefix, ro.col_to_tok, budget=int(self.tdr_candidate_budget),
                pool_a=int(self.tdr_pool_a), pool_b=int(self.tdr_pool_b))
            candidate_cols, candidate_tokens = proposal.cols, proposal.tokens
            self._last_tdr_report = proposal.report
        components, out, l5, _ = self._language_token_components(q_vec, prefix, pos, candidate_cols=candidate_cols)
        logits = sum(components.values())
        conflict = float(getattr(self, "_language_last_conflict", 0.0))
        trace = getattr(self, "_language_conflict_trace", None)
        if trace is not None:
            trace.append(conflict)
        # Арбитраж живёт здесь, а не в цикле: единственный источник решения для
        # всех путей генерации (прямой greedy, ассистент, проба).
        if self.language_conflict_arbitration:
            if conflict >= self._language_conflict_threshold():
                self._language_conflict_streak = getattr(self, "_language_conflict_streak", 0) + 1
            else:
                self._language_conflict_streak = 0
            self._language_abstain_requested = (
                self._language_conflict_streak >= int(self.language_conflict_abstain_steps))
        else:
            self._language_conflict_streak = 0
            self._language_abstain_requested = False
        return candidate_tokens, logits, out, l5

    def teacher_forced_likelihood(self, prompt: str, reference: str) -> dict:
        with self.evaluation_context():
            ro = self.dendritic_plateau_readout
            targets = self.encode_text(reference)
            if not targets or len(targets) > ro.max_pos or any(t not in ro.tok_to_col for t in targets):
                return {"available": False, "reason": "Reference exceeds the learned vocabulary or positional window."}
            q_raw = self.compute_stimulus_vec64(self.encode_text(prompt))
            q_vec, _, _ = self._prepare_language_query(prompt, q_raw, use_ngrams=self.use_ngrams)
            nll, errors = 0.0, 0
            for pos, target in enumerate(targets):
                _, logits, _, _ = self._language_token_distribution(q_vec, targets[:pos], pos, use_tdr=False)
                col = ro.tok_to_col[target]
                nll -= float(logits.log_softmax(0)[col])
                errors += int(int(logits.argmax()) != col)
            termination = {"available": False, "reason": "EOS is outside the learned vocabulary or positional window."}
            if self.eos_token_id in ro.tok_to_col and len(targets) < ro.max_pos:
                _, logits, _, _ = self._language_token_distribution(q_vec, targets, len(targets), use_tdr=False)
                col = ro.tok_to_col[self.eos_token_id]
                winner = ro.col_to_tok[int(logits.argmax())]
                termination = {"available": True, "position": len(targets), "target_token": self.eos_token_id,
                               "winner_token": winner, "correct": winner == self.eos_token_id,
                               "target_nll": -float(logits.log_softmax(0)[col])}
            return {"available": True, "nll_sum": nll, "token_count": len(targets), "token_errors": errors,
                    "termination": termination,
                    "measurement": "teacher_forced_decoder_distribution_before_repetition_controls"}

    # ------------------------------------------------------------------
    # ТРЁХФАКТОРНОЕ ОБУЧЕНИЕ ЧЕРЕЗ НЕЙРОМЕДИАТОРЫ (DA / ACh / NE / 5-HT) + SON
    # ------------------------------------------------------------------
    def train_wave_with_neuromodulator_reward(self, pairs: List[TokenizedDialogPair],
                                             wave_idx: int = 1) -> Dict[str, object]:
        life = getattr(self, "life", None)
        if life is None:
            return self._train_wave_with_neuromodulator_reward(pairs, wave_idx)
        previous = life.training_active
        life.training_active = True
        try:
            result = self._train_wave_with_neuromodulator_reward(pairs, wave_idx)
            result["organism"] = life.telemetry(self)
            return result
        finally:
            life.training_active = previous

    def _train_wave_with_neuromodulator_reward(
        self,
        pairs: List[TokenizedDialogPair],
        wave_idx: int = 1,
    ) -> Dict[str, object]:
        """Обучает синаптические матрицы организма на волне диалоговых пар `(Stimulus A -> Response B)`
        с модуляцией от биологических нейромедиаторов (`DA`, `ACh`, `NE`, `5-HT`, `ATP`) и удаляет сырые пары после записи!"""
        t0 = time.perf_counter()
        empty_wave = {"pairs": 0, "tokens": 0, "tokens_trained": 0, "pc_targets_trained": 0,
                      "dendritic_pairs_trained": 0, "dendritic_tokens_trained": 0,
                      "spiking_token_updates": 0, "physiology_blocked_pairs": 0,
                      "pc_shortlist_hits": 0, "pc_target_extensions": 0, "pc_shortlist_recall": None,
                      "readout_token_updates": 0, "protected_core_token_updates": 0,
                      "readout_step_limited": 0, "readout_max_effective_gain": 0.0, "readout_step_factor_sum": 0.0, "readout_step_factor_max": 0.0,
                      "response_tokens_trained": 0, "eos_targets_trained": 0,
                      "bank_token_updates": 0, "bank_token_rejections": 0, "bank_backtracking_steps": 0,
                 "bank_limited_mask_updates": 0, "bank_update_tokens_requested": 0,
                 "bank_update_tokens_selected": 0, "bank_gradient_entries_selected": 0,
                 "bank_memory_budget_rejections": 0, "bank_capacity_rejections": 0,
                 "bank_objective_rejections": 0, "bank_no_signal_rejections": 0,
                 "bank_other_rejections": 0, "bank_pruned_entries": 0}
        if not getattr(self, "learning_enabled", True):
            return {**empty_wave, "learning_disabled": True}
        pairs = [p for p in pairs if getattr(p, "split", "train") == "train" and len(p.response_ids) >= 2]
        if not pairs:
            return empty_wave
        if any(len(p.response_ids) >= self.dendritic_plateau_readout.max_pos for p in pairs):
            raise ValueError("Training responses must leave a positional slot for EOS.")
        if any(self.eos_token_id in p.response_ids for p in pairs):
            raise ValueError("Training response_ids must contain content only, without EOS.")
        if any(not math.isfinite(p.quality_reward_hint) or p.quality_reward_hint < 0 for p in pairs):
            raise ValueError("Training quality must be finite and nonnegative.")
        life = getattr(self, "life", None)
        if life is not None and self.chemistry.f_mod() <= 0.0:
            self.step(self.compute_stimulus_vec64(pairs[0].stimulus_ids), life.cfg.ticks_per_token)
            return {**empty_wave, "physiology_blocked_pairs": len(pairs),
                    "plasticity_gate": self.chemistry.f_mod(),
                    "observed_bpe_tokens": sum(len(p.stimulus_ids) + len(p.response_ids) for p in pairs)}
        readout = self.dendritic_plateau_readout
        for pair in pairs:
            for tid in [*pair.response_ids, self.eos_token_id]:
                readout._ensure_token_col(tid)
        self.spiking_language_loop.ensure_tokens(readout.col_to_tok)
        updates_before = self.spiking_language_loop.total_updates

        auxiliary_batch_items: List[List[torch.Tensor]] = []
        memory_query_vecs: List[torch.Tensor] = []
        pc_targets_trained = 0
        pc_shortlist_hits = 0
        pc_target_extensions = 0
        pc_losses: List[float] = []

        onset_q_vecs: List[torch.Tensor] = []
        onset_first_ids: List[int] = []
        dend_batch_items: List[Tuple[torch.Tensor, List[torch.Tensor], List[int], float, List[int], List[int]]] = []

        da_levels: List[float] = []
        ach_levels: List[float] = []
        f_mods: List[float] = []
        wave_tokens = 0
        physiology_blocked_pairs = 0

        for pair in pairs:
            s_ids = pair.stimulus_ids
            r_ids = pair.response_ids
            if life is None:
                wave_tokens += len(s_ids) + len(r_ids)
                self.total_tokens_trained += len(s_ids) + len(r_ids)
                self.total_pairs_trained += 1

            q_vec = self.compute_stimulus_vec64(s_ids)
            r_vec = self.compute_stimulus_vec64(r_ids)

            life = getattr(self, "life", None)
            if life is None:
                sem_align = float(torch.dot(q_vec, r_vec).item())
                rpe = (pair.quality_reward_hint - float(self.chemistry.s.da)) + 0.45 * max(0.0, sem_align + 0.3)
                living = self.measure_living_state()
                chem_state = self.chemistry.update(**living, rpe=rpe, dt_ms=1.0)
                self.chemistry.s.lactate = max(0.85, self.chemistry.s.lactate)
                f_mod = max(0.25, min(2.2, self.chemistry.f_mod(energy_ok=1.0) * 2.2))
            else:
                rpe = life.training_reward(self, pair.quality_reward_hint)
                chem_state = self.chemistry.s
                f_mod = min(2.2, self.chemistry.f_mod(energy_ok=1.0) * 2.2)

            da_levels.append(chem_state.da)
            ach_levels.append(chem_state.ach)
            f_mods.append(f_mod)
            self.cumulative_da_reward += rpe

            # Provenance is observed training data, never reverse-invented from weights.
            self.language_memory_records.append({
                "prompt": self.decode_tokens(s_ids).strip(),
                "response": self.decode_tokens(r_ids).strip(),
                "source": pair.source_dataset, "split": "train", "status": "asserted",
                "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
            memory_query = q_vec.clone()
            prompt = self.decode_tokens(s_ids).strip()
            q_vec, _, _ = self._prepare_language_query(prompt, q_vec, use_ngrams=self.use_ngrams)
            auxiliary_items: List[torch.Tensor] = []
            response_targets = [*r_ids, self.eos_token_id]
            token_gates = []
            for pos, target_token in enumerate(response_targets):
                prefix = response_targets[:pos]
                components, cortical_out, l5, pc_candidates = self._language_token_components(q_vec, prefix, pos)
                if life is not None:
                    f_mod = min(2.2, self.chemistry.f_mod(energy_ok=1.0) * 2.2)
                    if f_mod <= 0.0:
                        break
                token_gates.append(f_mod)
                if prefix:
                    train_candidates = list(pc_candidates)
                    if target_token in train_candidates:
                        pc_shortlist_hits += 1
                    else:
                        train_candidates.append(target_token)
                        pc_target_extensions += 1
                    candidate_cols = torch.tensor([readout.tok_to_col[t] for t in train_candidates], dtype=torch.long)
                    base_logits = components["readout"] + components["spiking"]
                    pc_losses.append(self.pc_readout.train_counterfactual_candidate_step(
                        prev2_id=prefix[-3] if len(prefix) >= 3 else prefix[-2] if len(prefix) >= 2 else prefix[-1],
                        prev1_id=prefix[-2] if len(prefix) >= 2 else prefix[-1],
                        cur_id=prefix[-1], target_id=target_token, topic_vec64=q_vec, l5_vec64=l5,
                        candidate_ids=train_candidates, auxiliary_logits=base_logits.index_select(0, candidate_cols),
                        lr=0.1 * (f_mod if life is not None else 1.0)))
                    pc_targets_trained += 1
                self.spiking_language_loop.learn(
                    cortical_out, target_token, readout.col_to_tok, task_modulation=f_mod,
                    auxiliary_logits=components["readout"] + components["pc"],
                    modulators=self.modulator_bundle())
                if life is not None:
                    life.record_credit(self, q_vec, cortical_out, self.spiking_language_loop.last_loss)
                auxiliary_items.append(components["spiking"] + components["pc"])
            if len(auxiliary_items) != len(response_targets):
                physiology_blocked_pairs += 1
                continue
            f_mod = min(token_gates)
            memory_query_vecs.append(memory_query)
            auxiliary_batch_items.append(auxiliary_items)

            # 2. Трёхфакторное обучение синаптического моста завязки ответа (Q(A) -> первые 2 токена B)
            t0_id = r_ids[0]
            t1_id = r_ids[1]
            onset_q_vecs.append(q_vec * (f_mod if life is not None else 1.0))
            onset_first_ids.append(t0_id)

            r_vecs_seq = [self.get_vec64(tid) for tid in response_targets]
            p_strip_s = " .,!?;:'()[]{}" + chr(34)
            stop_s = self.CONTENT_STOP
            s_content = [
                tid for tid in s_ids
                if len(self.decode_token(tid).strip(p_strip_s)) >= 3
                and self.decode_token(tid).strip(p_strip_s).lower() not in stop_s
            ] or s_ids
            r_content = [
                tid for tid in r_ids
                if len(self.decode_token(tid).strip(p_strip_s)) >= 3
                and self.decode_token(tid).strip(p_strip_s).lower() not in stop_s
            ] or r_ids
            dend_batch_items.append(
                (q_vec, r_vecs_seq, response_targets, f_mod * pair.quality_reward_hint, s_content, r_content)
            )

            op_map = self.onset_pair_synapses.setdefault(t0_id, {})
            op_map[t1_id] = min(5.0, op_map.get(t1_id, 0.0) + 0.65 * f_mod)
            op_key = (t0_id, t1_id)
            if op_key in self.onset_pair_ctx_vec:
                self.onset_pair_ctx_vec[op_key] = F.normalize(0.75 * self.onset_pair_ctx_vec[op_key] + 0.25 * q_vec, p=2, dim=-1)
            else:
                self.onset_pair_ctx_vec[op_key] = q_vec

            # 3. Трёхфакторное обучение дендритных проводимостей внутри ответа B, модулированное Q(A) и дофамином f_mod!
            for pos in range(len(response_targets) - 1
                             if (self.use_ngrams or not hasattr(self, "life")) else 0):
                cur = response_targets[pos]
                nxt = response_targets[pos + 1]

                bg_map = self.synaptic_bigram_g.setdefault(cur, {})
                bg_map[nxt] = min(4.5, bg_map.get(nxt, 0.0) + 0.45 * f_mod)

                if pos >= 1:
                    prev1 = r_ids[pos - 1]
                    tg_map = self.synaptic_trigram_g.setdefault((prev1, cur), {})
                    tg_map[nxt] = min(5.5, tg_map.get(nxt, 0.0) + 0.78 * f_mod)

                    # Обновляем синаптический вектор контекста вопроса mu_{prev1, cur, nxt} (пластическое усреднение Хебба):
                    t_key = (prev1, cur, nxt)
                    if t_key in self.synaptic_trigram_ctx:
                        self.synaptic_trigram_ctx[t_key] = F.normalize(
                            0.80 * self.synaptic_trigram_ctx[t_key] + 0.20 * q_vec, p=2, dim=-1
                        )
                    elif len(self.synaptic_trigram_ctx) < 450_000:
                        self.synaptic_trigram_ctx[t_key] = q_vec

                if pos >= 2 and len(self.synaptic_fourgram_g) < 350_000:
                    prev2 = r_ids[pos - 2]
                    prev1 = r_ids[pos - 1]
                    fg_map = self.synaptic_fourgram_g.setdefault((prev2, prev1, cur), {})
                    fg_map[nxt] = min(6.0, fg_map.get(nxt, 0.0) + 0.92 * f_mod)

        # 3B. Векторизованное пакетное обучение матрицы 8,192 дендритных NMDA-ветвей и двунаправленного CA3
        readout_stats = self.dendritic_plateau_readout.imprint_response_batch_vectorized(
            dend_batch_items, auxiliary_logits=auxiliary_batch_items, memory_query_vecs=memory_query_vecs)

        # 4. Пакетное обновление матрицы инициации ответа W_stimulus_to_onset [64, 50257]
        if onset_q_vecs:
            with torch.no_grad():
                q_mat_on = torch.stack(onset_q_vecs)  # [N, 64]
                f_t = torch.tensor(onset_first_ids, dtype=torch.long)
                self.w_stimulus_to_onset.index_add_(1, f_t, q_mat_on.t() * 0.35)
                uniq_f = torch.unique(f_t)
                self.w_stimulus_to_onset.index_copy_(
                    1, uniq_f, F.normalize(self.w_stimulus_to_onset.index_select(1, uniq_f), p=2, dim=0)
                )

        pc_loss = sum(pc_losses) / len(pc_losses) if pc_losses else self.pc_readout.last_pc_loss

        # Уничтожаем ссылки на входной пакет пар — в памяти остаются ТОЛЬКО веса!
        del pairs

        elapsed = max(1e-4, time.perf_counter() - t0)
        # ── Давление сна от бута активности ────────────────────────────────────
        # Аденозин накапливается не по тику, а за bout активности: тик внутри
        # волны — это единица обработки, а бут — это ~минуты биологического
        # времени непрерывной работы. Без бутового члена гейт пластичности
        # H(θ_S−S) недостижим в любом разумном числе тиков (τ_аденозина — часы),
        # и сон как фаза организма не существует вовсе.
        bout_activity = self.measure_living_state()["activity"]
        if not hasattr(self, "life"):
            self.chemistry.s.ado = min(
                1.0, float(self.chemistry.s.ado)
                + bout_activity * float(self.chemistry.cfg.ado_bout_gain))
        return {
            "wave_idx": wave_idx,
            "physiology_blocked_pairs": physiology_blocked_pairs,
            "tokens_trained": wave_tokens,
            "bout_activity": round(float(bout_activity), 6),
            "sleep_pressure": round(float(self.chemistry.s.ado), 6),
            "awake": bool(self.chemistry.is_awake),
            "dendritic_pairs_trained": len(dend_batch_items),
            "dendritic_tokens_trained": sum(len(item[2]) for item in dend_batch_items),
            "response_tokens_trained": sum(len(item[2]) - 1 for item in dend_batch_items),
            "eos_targets_trained": len(dend_batch_items),
            "spiking_token_updates": self.spiking_language_loop.total_updates - updates_before,
            "pc_targets_trained": pc_targets_trained,
            "pc_shortlist_hits": pc_shortlist_hits,
            "pc_target_extensions": pc_target_extensions,
            "pc_shortlist_recall": pc_shortlist_hits / pc_targets_trained if pc_targets_trained else None,
            "pc_training_objective": "target-inclusive shortlist contrast, not full-vocabulary likelihood",
            **readout_stats,
            "readout_training_objective": "shared shield-projected gradient with full-support bank; gain bound assumes fixed SWR/auxiliary contributions",
            "bank_training_objective": "full-vocabulary bounded-residual NLL with sparse update mask and atomic backtracking",
            "throughput_tok_per_sec": round(wave_tokens / elapsed, 1),
            "mean_dopamine_DA": round(sum(da_levels) / float(max(1, len(da_levels))), 4),
            "mean_acetylcholine_ACh": round(sum(ach_levels) / float(max(1, len(ach_levels))), 4),
            "mean_plasticity_gate_f_mod": round(sum(f_mods) / float(max(1, len(f_mods))), 4),
            "min_dopamine_DA": round(min(da_levels), 4) if da_levels else None,
            "max_dopamine_DA": round(max(da_levels), 4) if da_levels else None,
            "min_plasticity_gate_f_mod": round(min(f_mods), 4) if f_mods else None,
            "max_plasticity_gate_f_mod": round(max(f_mods), 4) if f_mods else None,
            "f_mod_saturated_at_ceiling": bool(f_mods and min(f_mods) >= 2.19),
            "da_saturated_at_ceiling": bool(da_levels and min(da_levels) >= 1.99),
            "pc_readout_loss_nats": round(pc_loss, 4),
            "synaptic_trigrams_count": len(self.synaptic_trigram_g),
            "synaptic_fourgrams_count": len(self.synaptic_fourgram_g),
            "elapsed_sec": round(elapsed, 3),
        }

    def _online_dopamine_imprint_from_user(self, user_utterance: str) -> int:
        """Когда собеседник в диалоге сообщает новые сведения о себе (`I / my`) ИЛИ явно обучает модель новому
        причинно-следственному факту (`'Remember that ...'`, `'Note that ...'`, `'In <domain>, ...'`),
        холинергический/дофаминовый всплеск новизны (`ACh + DA burst`) консолидирует этот факт напрямую в
        синаптическую матрицу 8 192 дендритных NMDA-ветвей `DendriticPlateauMatrixReadout` и полюса `CA3`!"""
        if not self.learning_enabled or not self.online_learning_enabled:
            return 0
        p_strip = " .,!?;:'()[]{}" + chr(34)
        low_u = user_utterance.strip().lower()
        has_identity_fact = any(p in low_u for p in ("my name is", "i am building", "i'm building", "call me "))
        is_declarative_teaching = any(
            low_u.startswith(pref) for pref in ("remember that ", "note that ", "learn that ", "fact: ")
        )
        is_q = ("?" in user_utterance) or any(
            low_u.startswith(q) for q in ("what", "who", "how", "why", "where", "when", "remind", "do", "can", "explain")
        )
        if is_q or (not has_identity_fact and not is_declarative_teaching):
            return 0

        if is_declarative_teaching and not has_identity_fact:
            clean_fact = user_utterance.strip()
            for pref in ("Remember that ", "remember that ", "Note that ", "note that ", "Learn that ", "learn that ", "Fact: ", "fact: "):
                if clean_fact.startswith(pref):
                    clean_fact = clean_fact[len(pref) :].strip()
                    break
            if clean_fact and clean_fact[0].islower():
                clean_fact = clean_fact[0].upper() + clean_fact[1:]
            if clean_fact and clean_fact[-1] not in ".!?":
                clean_fact += "."
            s_ids = self.encode_text(clean_fact)
            r_ids = self.encode_text(clean_fact)
            if len(r_ids) < 4:
                return 0
            teach_pairs = [TokenizedDialogPair("online_declarative_teaching_da_burst", s_ids, r_ids, quality_reward_hint=2.8)]
            for _ in range(4):
                self.train_wave_with_neuromodulator_reward(teach_pairs, wave_idx=1)
            return len(r_ids)

        first_sent = user_utterance.split(".")[0].strip() + "."
        inv_words: List[str] = []
        for w in first_sent.split():
            cl = w.strip(p_strip).lower()
            if cl in ("hi", "hello", "hey"):
                continue
            if cl in self.PRONOUN_INVERSION:
                rep = self.PRONOUN_INVERSION[cl]
                if len(inv_words) == 0 and w.strip(p_strip)[0].isupper():
                    rep = rep[0].upper() + rep[1:]
                suf = w[len(w.rstrip(p_strip)) :]
                inv_words.append(rep + suf)
            else:
                inv_words.append(w)

        inv_text = " ".join(inv_words)
        s_ids = self.encode_text(user_utterance)
        r_ids = self.encode_text(inv_text)
        if len(r_ids) < 4:
            return 0

        # Создаём временные пары для прокачки через синаптические матрицы с высоким дофамином (quality_reward_hint = 3.2)
        self.dialogue_state.register_user_identity(user_utterance, self.compute_stimulus_vec64(s_ids))
        cur_name = (self.dialogue_state.user_name or "user").strip()
        q_self = self.encode_text(f"Identity profile of {cur_name}: {user_utterance}")
        q_self_vec = self.compute_stimulus_vec64(q_self)
        self.dialogue_state.register_user_identity(user_utterance, q_self_vec)
        temp_pairs = [
            TokenizedDialogPair("online_user_da_burst", s_ids, r_ids, quality_reward_hint=3.2),
            TokenizedDialogPair("online_user_da_burst", q_self, r_ids, quality_reward_hint=3.2),
        ]
        for _ in range(4):
            self.train_wave_with_neuromodulator_reward(temp_pairs, wave_idx=1)
        return len(r_ids)

    # ------------------------------------------------------------------
    # 100% ЧИСТЫЙ ПОСЛОВНЫЙ СИНТЕЗ ОТВЕТА ТОЛЬКО ИЗ СИНАПТИЧЕСКИХ ВЕСОВ + SON VTE
    # ------------------------------------------------------------------
    def _synthesize_single_beam_from_weights_only(
        self,
        q_vec: torch.Tensor,
        l5_vec: torch.Tensor,
        onset_pair: Tuple[int, int],
        temperature_jitter: float = 0.0,
        max_tokens: int = 36,
        use_ngrams: Optional[bool] = None,
        arcuate_pointer_map: Optional[Dict[int, Tuple[int, List[int]]]] = None,
    ) -> Tuple[List[int], float, str]:
        """Синтезирует одну траекторию ответа шаг за шагом:
          - Включает **Дугообразный Пучок (`Arcuate Fasciculus Synaptic Pointer`, Point 1)**:
            поддерживает раздельный внутренний след ролей коры (`role_state_ids` для `vp2, vp1, vc`) и моторный выход (`out_ids`).
            Когда дендритные плато возбуждают канонический ролевой токен из `arcuate_pointer_map`, в `out_ids` напрямую
            копируются BPE-токены сущности собеседника (`filler_tids`) без строкового `.replace()`!
          - Включает **Замкнутую Сенсомоторную Реафферентацию (`MoE + Stranger's Check`, Point 3)**:
            на каждом шаге проверяет топ-1 и топ-2 кандидатов через `DiscretePredictiveCodingBarrier`."""
        ngrams_enabled = self.use_ngrams if use_ngrams is None else bool(use_ngrams)

        # Cortical activity and SON observations are refreshed for each emitted token.
        if not ngrams_enabled:
            role_state_ids: List[int] = []
            out_ids: List[int] = []
            used_bigrams: Dict[Tuple[int, int], int] = {}
        else:
            role_state_ids = [onset_pair[0], onset_pair[1]]
            out_ids = [onset_pair[0], onset_pair[1]]
            used_bigrams = {(onset_pair[0], onset_pair[1]): 1}

        cooldown: Dict[int, int] = {}
        total_syn_support = 0.0
        finish_reason = "token_budget"
        skip_canonical_subwords = 0

        start_pos = len(role_state_ids)
        for pos_i in range(start_pos, min(max_tokens, self.dendritic_plateau_readout.max_pos)):
            cur = role_state_ids[-1] if len(role_state_ids) >= 1 else None
            prev1 = role_state_ids[-2] if len(role_state_ids) >= 2 else cur
            prev2 = role_state_ids[-3] if len(role_state_ids) >= 3 else prev1

            cands: Dict[int, float] = {}

            # А. Матричное предсказание Нелинейного Дендритного Плато-Декодера (8,192 NMDA-ветви, Top-32 Sparse WTA, 0% N-грамм!)
            vp2 = q_vec if len(role_state_ids) < 3 else self.get_vec64(role_state_ids[-3])
            vp1 = q_vec if len(role_state_ids) < 2 else self.get_vec64(role_state_ids[-2])
            vc = q_vec if len(role_state_ids) < 1 else self.get_vec64(role_state_ids[-1])
            if not ngrams_enabled:
                token_ids, logits, token_moe, l5_vec = self._language_token_distribution(q_vec, role_state_ids, pos_i)
                cands = dict(zip(token_ids, logits.tolist()))
            else:
                dend_scores = self.dendritic_plateau_readout.predict_next_token_scores(vp2, vp1, vc, pos_i, q_vec)
                cortical_token_ids = list(dict.fromkeys(list(dend_scores) + self.spiking_language_loop.token_ids))
                token_input = self._transmit_temporal_vector(F.normalize(q_vec + vc, p=2, dim=0))
                cortical_scores, token_moe = self.spiking_language_loop.step(
                    token_input, cortical_token_ids, learn=False,
                    cortical_step=self._cortical_step if hasattr(self, "life") else None)
                cands = dict(zip(cortical_token_ids, cortical_scores.tolist()))
                for cid, d_sc in dend_scores.items():
                    cands[cid] = cands.get(cid, 0.0) + 0.45 * d_sc
                l5_vec = F.normalize(F.adaptive_avg_pool1d(token_moe.combined_l5_drive.mean(0).view(1, 1, -1), 64).flatten(), dim=0)
            self.last_cortical_trajectory_trace = getattr(self, "last_cortical_trajectory_trace", [])
            self.last_cortical_trajectory_trace.append(list(token_moe.active_expert_names))
            col_rates = torch.tensor([float(self.moe_cortex._last_col_spikes[i].mean()) for i in range(self.moe_cortex.n_columns)])
            col_som_err = torch.tensor([float(c.dendrite.last_somatic_fep_error) for c in self.moe_cortex.columns])
            col_api_err = torch.tensor([float(c.dendrite.last_apical_fep_error) for c in self.moe_cortex.columns])
            col_gates = token_moe.gating_weights

            # Б. Если режим N-грамм ВКЛЮЧЁН (ON), добавляем проводимости 4-грамм, 3-грамм и 2-грамм
            if ngrams_enabled and cur is not None and prev1 is not None and prev2 is not None:
                if (prev2, prev1, cur) in self.synaptic_fourgram_g:
                    for nxt, g4 in self.synaptic_fourgram_g[(prev2, prev1, cur)].items():
                        ctx_v = self.synaptic_trigram_ctx.get((prev1, cur, nxt))
                        cos_q = float(torch.dot(q_vec, ctx_v).item()) if ctx_v is not None else 0.5
                        cands[nxt] = cands.get(nxt, 0.0) + g4 * (1.5 + 3.2 * max(0.0, cos_q))

                if (prev1, cur) in self.synaptic_trigram_g:
                    for nxt, g3 in self.synaptic_trigram_g[(prev1, cur)].items():
                        ctx_v = self.synaptic_trigram_ctx.get((prev1, cur, nxt))
                        cos_q = float(torch.dot(q_vec, ctx_v).item()) if ctx_v is not None else 0.3
                        cands[nxt] = cands.get(nxt, 0.0) + g3 * (1.0 + 2.8 * max(0.0, cos_q))

                if len(cands) < 4 and cur in self.synaptic_bigram_g:
                    for nxt, g2 in self.synaptic_bigram_g[cur].items():
                        cands[nxt] = cands.get(nxt, 0.0) + 0.45 * g2

            if not cands:
                finish_reason = "empty_distribution"
                break

            # В. Вклад матрицы PredictiveCodingCorticalReadout W_readout [256, 50257]
            if ngrams_enabled and cur is not None and prev1 is not None and prev2 is not None:
                top_cand_ids = [cid for cid, _ in sorted(cands.items(), key=lambda kv: kv[1], reverse=True)[:48]]
                pc_scores = self.pc_readout.score_candidates(
                    prev2_id=prev2,
                    prev1_id=prev1,
                    cur_id=cur,
                    topic_vec64=q_vec,
                    l5_vec64=l5_vec,
                    candidate_ids=top_cand_ids,
                )
                for cid, pc_s in pc_scores.items():
                    cands[cid] = cands.get(cid, 0.0) + 2.2 * pc_s

            # Г. Рефрактерность и анти-циклический фильтр
            for cid in list(cands.keys()):
                if len(role_state_ids) >= 2 and cid == cur == role_state_ids[-2]:
                    cands[cid] = -1e9
                    continue
                if cur is not None:
                    bg_cnt = used_bigrams.get((cur, cid), 0)
                    if bg_cnt > 0:
                        cands[cid] -= 8.0 * bg_cnt
                cd = cooldown.get(cid, 0)
                if cd > 0 and cid != cur:
                    cands[cid] -= 3.5 * cd
                if temperature_jitter > 0.0:
                    cands[cid] += random.gauss(0.0, temperature_jitter)

            sorted_cands = sorted(cands.items(), key=lambda kv: kv[1], reverse=True)
            best_nxt, best_score = sorted_cands[0]
            if best_score < -1e5:
                finish_reason = "repetition_barrier"
                break

            # Г2. Замкнутая петля реафферентации MoE + DiscretePredictiveCodingBarrier (Stranger's Check, Point 3):
            if len(sorted_cands) >= 2 and cur is not None:
                rep_p = float(used_bigrams.get((cur, best_nxt), 0)) * 0.65
                barrier_res = self.moe_cortex.discrete_pc_barrier.evaluate_discrete_pin_and_stranger_check(
                    continuous_error=max(0.0, 1.0 - max(0.0, best_score) * 0.04),
                    primary_synaptic_support=max(0.1, float(best_score)),
                    stranger_contrast_support=max(0.05, float(sorted_cands[1][1])),
                    recent_repetition_pressure=rep_p,
                )
                if bool(barrier_res.get("stranger_check_triggered", False)) and sorted_cands[1][1] > -1e4:
                    best_nxt, best_score = sorted_cands[1]
                    self.last_reafferent_stranger_interventions = getattr(self, "last_reafferent_stranger_interventions", 0) + 1

            # Арбитраж «префикс ↔ эпизод»: при устойчивом конфликте входа и
            # извлечённого эпизода система воздерживается вместо того, чтобы
            # излучать бессвязный поток. Решение принимает _language_token_distribution.
            if bool(getattr(self, "_language_abstain_requested", False)):
                finish_reason = "conflict_abstain"
                break

            if best_nxt == self.eos_token_id:
                finish_reason = "learned_eos"
                break
            if cur is not None:
                used_bigrams[(cur, best_nxt)] = used_bigrams.get((cur, best_nxt), 0) + 1
            role_state_ids.append(best_nxt)

            # Г3. Аксональная инъекция Дугообразного Пучка (Arcuate Fasciculus Token-ID Pointer, Point 1):
            if skip_canonical_subwords > 0:
                skip_canonical_subwords -= 1
            elif arcuate_pointer_map and int(best_nxt) in arcuate_pointer_map:
                span_len, filler_tids = arcuate_pointer_map[int(best_nxt)]
                out_ids.extend(filler_tids)
                skip_canonical_subwords = max(0, int(span_len) - 1)
                self.last_arcuate_token_injections = getattr(self, "last_arcuate_token_injections", 0) + 1
            else:
                out_ids.append(best_nxt)

            total_syn_support += max(0.1, min(3.0, best_score * 0.1))
            for k in list(cooldown.keys()):
                if cooldown[k] <= 1:
                    del cooldown[k]
                else:
                    cooldown[k] -= 1
            tok_s = self.decode_token(best_nxt).strip().lower()
            if tok_s not in (".", ",", "!", "?", "the", "a", "to", "and", "of", "in", "is", "are", "you", "your"):
                cooldown[best_nxt] = 6

            # Д. Рефлексивное наблюдение 17-й колонки SON и проверка естественного стоп-крана SON_MOTOR_STOP
            is_end_punct = tok_s.endswith((".", "!", "?"))
            # ГРАНИЦА ФРАЗЫ → ТАЛАМУС (14.11, трек M): язык знает, где кончилась
            # фраза, а ThalamusFO — единственный, кто умеет отдать корне фазовый
            # маркер начала новой синтаксической мысли (`boundary_burst`).
            # До этого маркер помечался, но из живого пути НЕ доходил: `fo.step()`
            # вызывался без флага, `mark_phrase_boundary()` не звался нигде.
            fo = getattr(getattr(self, "life", None), "fo", None)
            if is_end_punct and fo is not None and hasattr(fo, "mark_phrase_boundary"):
                fo.mark_phrase_boundary()
            verdict = self.son_column.observe_and_reflect_step(
                col_firing_rates=col_rates,
                col_somatic_errors=col_som_err,
                col_apical_errors=col_api_err,
                col_gating_weights=col_gates,
                candidate_token_vec64=self.get_vec64(best_nxt),
                is_sentence_boundary_punct=is_end_punct,
                tokens_generated_so_far=len(role_state_ids),
                syntax_transition_support=1.0 if len(role_state_ids) < 16 else 0.15,
                domain_conflict_energy=0.0,
                learn=False,
            )
            if is_end_punct and (len(role_state_ids) >= 2 or verdict.son_motor_stop_triggered):
                finish_reason = "son_eos" if verdict.son_motor_stop_triggered else "sentence_boundary_control"
                break

        mean_sup = total_syn_support / float(max(1, len(out_ids)))
        return out_ids, mean_sup, finish_reason

    def _synthesize_neural_beam_search(
        self,
        q_vec: torch.Tensor,
        l5_vec: torch.Tensor,
        onset_pair: Tuple[int, int] = (0, 0),
        max_tokens: int = 36,
        arcuate_pointer_map: Optional[Dict[int, Tuple[int, List[int]]]] = None,
        beam_width: int = 3,
        branch_factor: int = 3,
    ) -> List[Dict[str, object]]:
        beam_width = max(1, int(beam_width))
        branch_factor = max(1, int(branch_factor))
        initial_state = self.export_language_dynamic_state()
        initial = {
            "role_state_ids": [], "out_ids": [], "used_bigrams": {}, "cooldown": {},
            "skip_canonical_subwords": 0, "log_probability": 0.0, "control_penalty": 0.0,
            "total_syn_support": 0.0, "normalized_score": 0.0, "stopped_by_son": False,
            "stopped_by_learned_eos": False,
            "finished": False, "finish_reason": "active", "last_son_free_energy": 0.0,
            "dynamic_state": initial_state,
        }
        active: List[Dict[str, object]] = [initial]
        completed: List[Dict[str, object]] = []

        def normalized_score(beam: Dict[str, object]) -> float:
            length = max(1, len(beam["role_state_ids"]))
            return (float(beam["log_probability"]) + float(beam["control_penalty"])) / float(length ** 0.7)

        def rank_beams(beams: List[Dict[str, object]]) -> List[Dict[str, object]]:
            return sorted(beams, key=lambda beam: (
                -float(beam["normalized_score"]),
                -float(beam["log_probability"]),
                tuple(beam["out_ids"]),
            ))

        try:
            token_limit = min(int(max_tokens), self.dendritic_plateau_readout.max_pos)
            for pos_i in range(token_limit):
                children: Dict[Tuple[int, ...], Dict[str, object]] = {}
                for parent in active:
                    self.load_language_dynamic_state(parent["dynamic_state"])
                    role_state_ids = list(parent["role_state_ids"])
                    used_bigrams = dict(parent["used_bigrams"])
                    cur = role_state_ids[-1] if role_state_ids else None
                    token_ids, logits, token_moe, l5_vec = self._language_token_distribution(
                        q_vec, role_state_ids, pos_i)
                    self.last_cortical_trajectory_trace = getattr(self, "last_cortical_trajectory_trace", [])
                    self.last_cortical_trajectory_trace.append(list(token_moe.active_expert_names))
                    col_rates = torch.tensor([
                        float(self.moe_cortex._last_col_spikes[i].mean())
                        for i in range(self.moe_cortex.n_columns)
                    ])
                    col_som_err = torch.tensor([
                        float(c.dendrite.last_somatic_fep_error) for c in self.moe_cortex.columns
                    ])
                    col_api_err = torch.tensor([
                        float(c.dendrite.last_apical_fep_error) for c in self.moe_cortex.columns
                    ])
                    if not token_ids or not logits.numel():
                        parent["dynamic_state"] = self.export_language_dynamic_state()
                        parent["finished"] = True
                        parent["finish_reason"] = "empty_distribution"
                        completed.append(parent)
                        continue

                    log_probabilities = logits.log_softmax(0)
                    controlled = logits.clone()
                    local_penalties = torch.zeros_like(logits)
                    for idx, cid in enumerate(token_ids):
                        if len(role_state_ids) >= 2 and cid == cur == role_state_ids[-2]:
                            controlled[idx] = -1e9
                            continue
                        penalty = 0.0
                        if cur is not None:
                            bg_cnt = int(used_bigrams.get((cur, cid), 0))
                            if bg_cnt > 0:
                                penalty -= 8.0 * bg_cnt
                        cd = int(parent["cooldown"].get(cid, 0))
                        if cd > 0 and cid != cur:
                            penalty -= 3.5 * cd
                        local_penalties[idx] = penalty
                        controlled[idx] += penalty

                    order = torch.argsort(controlled, descending=True).tolist()
                    if not order or float(controlled[order[0]]) < -1e5:
                        parent["dynamic_state"] = self.export_language_dynamic_state()
                        parent["finished"] = True
                        parent["finish_reason"] = "repetition_barrier"
                        completed.append(parent)
                        continue

                    best_idx = order[0]
                    best_nxt = int(token_ids[best_idx])
                    best_score = float(controlled[best_idx])
                    if len(order) >= 2 and cur is not None:
                        second_idx = order[1]
                        rep_p = float(used_bigrams.get((cur, best_nxt), 0)) * 0.65
                        barrier_res = self.moe_cortex.discrete_pc_barrier.evaluate_discrete_pin_and_stranger_check(
                            continuous_error=max(0.0, 1.0 - max(0.0, best_score) * 0.04),
                            primary_synaptic_support=max(0.1, best_score),
                            stranger_contrast_support=max(0.05, float(controlled[second_idx])),
                            recent_repetition_pressure=rep_p,
                        )
                        if (bool(barrier_res.get("stranger_check_triggered", False))
                                and float(controlled[second_idx]) > -1e4):
                            controlled[best_idx] = -1e9
                            self.last_reafferent_stranger_interventions = getattr(
                                self, "last_reafferent_stranger_interventions", 0) + 1
                            order = torch.argsort(controlled, descending=True).tolist()

                    post_distribution_state = self.export_language_dynamic_state()
                    expanded = False
                    for token_idx in order[:branch_factor]:
                        if float(controlled[token_idx]) < -1e5:
                            continue
                        expanded = True
                        self.load_language_dynamic_state(post_distribution_state)
                        best_nxt = int(token_ids[token_idx])
                        best_score = float(controlled[token_idx])
                        next_role_ids = role_state_ids + [best_nxt]
                        next_out_ids = list(parent["out_ids"])
                        next_bigrams = dict(used_bigrams)
                        next_cooldown = dict(parent["cooldown"])
                        skip_canonical = int(parent["skip_canonical_subwords"])
                        arc_injections = 0
                        if cur is not None:
                            next_bigrams[(cur, best_nxt)] = next_bigrams.get((cur, best_nxt), 0) + 1
                        is_eos = best_nxt == self.eos_token_id
                        if not is_eos:
                            if skip_canonical > 0:
                                skip_canonical -= 1
                            elif arcuate_pointer_map and best_nxt in arcuate_pointer_map:
                                span_len, filler_tids = arcuate_pointer_map[best_nxt]
                                next_out_ids.extend(filler_tids)
                                skip_canonical = max(0, int(span_len) - 1)
                                arc_injections = 1
                                self.last_arcuate_token_injections = getattr(
                                    self, "last_arcuate_token_injections", 0) + 1
                            else:
                                next_out_ids.append(best_nxt)

                        for tid in list(next_cooldown):
                            if next_cooldown[tid] <= 1:
                                del next_cooldown[tid]
                            else:
                                next_cooldown[tid] -= 1
                        son_stop = False
                        boundary_stop = False
                        son_free_energy = float(parent["last_son_free_energy"])
                        if not is_eos:
                            tok_s = self.decode_token(best_nxt).strip().lower()
                            if tok_s not in (".", ",", "!", "?", "the", "a", "to", "and", "of", "in", "is", "are", "you", "your"):
                                next_cooldown[best_nxt] = 6
                            is_end_punct = tok_s.endswith((".", "!", "?"))
                            verdict = self.son_column.observe_and_reflect_step(
                                col_firing_rates=col_rates,
                                col_somatic_errors=col_som_err,
                                col_apical_errors=col_api_err,
                                col_gating_weights=token_moe.gating_weights,
                                candidate_token_vec64=self.get_vec64(best_nxt),
                                is_sentence_boundary_punct=is_end_punct,
                                tokens_generated_so_far=len(next_role_ids),
                                syntax_transition_support=1.0 if len(next_role_ids) < 16 else 0.15,
                                domain_conflict_energy=0.0,
                                learn=False,
                            )
                            son_stop = bool(is_end_punct and verdict.son_motor_stop_triggered)
                            boundary_stop = bool(is_end_punct and len(next_role_ids) >= 2)
                            son_free_energy = float(verdict.son_free_energy_F)
                        stopped = is_eos or son_stop or boundary_stop
                        finish_reason = ("learned_eos" if is_eos else "son_eos" if son_stop
                                         else "sentence_boundary_control" if boundary_stop else "active")
                        child = {
                            "role_state_ids": next_role_ids,
                            "out_ids": next_out_ids,
                            "used_bigrams": next_bigrams,
                            "cooldown": next_cooldown,
                            "skip_canonical_subwords": skip_canonical,
                            "log_probability": float(parent["log_probability"]) + float(log_probabilities[token_idx]),
                            "control_penalty": float(parent["control_penalty"]) + float(local_penalties[token_idx]),
                            "total_syn_support": float(parent["total_syn_support"]) + max(0.1, min(3.0, best_score * 0.1)),
                            "stopped_by_son": son_stop,
                            "stopped_by_learned_eos": is_eos,
                            "finished": stopped,
                            "finish_reason": finish_reason,
                            "last_son_free_energy": son_free_energy,
                            "arcuate_injections": int(parent.get("arcuate_injections", 0)) + arc_injections,
                            "dynamic_state": self.export_language_dynamic_state(),
                        }
                        child["normalized_score"] = normalized_score(child)
                        key = tuple(next_role_ids)
                        previous = children.get(key)
                        if previous is None or float(child["normalized_score"]) > float(previous["normalized_score"]):
                            children[key] = child
                    if not expanded:
                        parent["dynamic_state"] = post_distribution_state
                        parent["finished"] = True
                        parent["finish_reason"] = "stranger_barrier"
                        completed.append(parent)

                next_active = []
                for child in children.values():
                    if bool(child["finished"]):
                        completed.append(child)
                    else:
                        next_active.append(child)
                active = rank_beams(next_active)[:beam_width]
                completed = rank_beams(completed)[:beam_width * 2]
                if not active:
                    break

            results = rank_beams(completed)[:beam_width]
            if len(results) < beam_width:
                results.extend(rank_beams(active)[:beam_width - len(results)])
            if not results:
                results = [initial]
            for beam in results:
                if not beam["finished"] and beam["finish_reason"] == "active":
                    beam["finish_reason"] = "token_budget"
                beam["normalized_score"] = normalized_score(beam)
                beam["mean_syn_support"] = float(beam["total_syn_support"]) / float(max(1, len(beam["out_ids"])))
            return results
        finally:
            self.load_language_dynamic_state(initial_state)

    def compute_clause_attractor_poles(
        self,
        user_utterance: str,
        enable_multi_hop: bool = False,
        role_filler_bindings: Optional[Dict[str, str]] = None,
    ) -> List[torch.Tensor]:
        """Разбивает длинный незнакомый стимул на смысловые клаузы (по знакам `,`, `;`, `—`, `--`, `while`, `before`, `after`, `and`),
        выполняет каноническую ролевую проекцию `ThetaGammaRoleFillerBinder`, отсекает дискурсивные вводные конструкции
        (`'Remind me,'`, `'Tell me,'`) и проецирует каждую клаузу на ближайший релевантный полюс `W_CA3`.
        При `enable_multi_hop = True` (или запросе причинной цепочки) выполняет 2-хоповый транзитивный вывод ($A \\implies B \\implies C$)."""
        self.last_multi_hop_used = False
        self.last_ca3_top1_resonance = 1.0
        self.last_identity_profile_used = False
        self.last_consolidated_schema_boost_used = False
        self.last_selected_ca3_pole_indices: List[int] = []
        self.last_selected_ca3_pole_saliences: List[float] = []
        self.last_selected_ca3_pole_resonances: List[float] = []

        # 0. Каноническая проекция переменных слотов Role x Filler (например, hollandaise -> mayonnaise для поиска схемы в CA3):
        slot_bindings = (self.role_filler_binder.extract_slot_bindings(user_utterance)
                         if role_filler_bindings is None else role_filler_bindings)
        canonical_utt = self.role_filler_binder.rewrite_utterance_to_canonical_for_ca3(user_utterance, slot_bindings)

        # 1. Явная маршрутизация рабочей памяти самоидентификации (ziwago_v12_conversational_mind.py):
        domains = self.dialogue_state.classify_utterance_domains(canonical_utt)
        if (
            self.dialogue_state.is_identity_query(canonical_utt)
            and self.dialogue_state.identity_pole_vec is not None
            and domains == ["self_identity"]
        ):
            self.last_ca3_top1_resonance = 2.5
            self.last_identity_profile_used = True
            self._record_pole_telemetry_from_vectors([self.dialogue_state.identity_pole_vec], [2.5])
            return [self.dialogue_state.identity_pole_vec]

        # 2. Очистка вводных дискурсивных оборотов перед клаузальным разбиением (чтобы 'Remind me,' не порождала ложный полюс!):
        cleaned_utt = canonical_utt.strip()
        lower_utt = cleaned_utt.lower()
        discourse_prefixes = (
            "remind me,",
            "tell me,",
            "by the way,",
            "excuse me,",
            "step by step,",
            "in detail,",
            "explain in detail,",
            "hi, how are you doing?",
            "hi!",
            "hello,",
        )
        for pref in discourse_prefixes:
            if lower_utt.startswith(pref):
                rest = cleaned_utt[len(pref) :].strip()
                if len(rest.split()) >= 3:
                    cleaned_utt = rest
                    break

        raw_clauses = [
            c.strip()
            for c in cleaned_utt.replace("—", "|")
            .replace("--", "|")
            .replace(";", "|")
            .replace(",", "|")
            .replace(" while ", "|")
            .replace(" before ", "|")
            .replace(" after ", "|")
            .replace(" and ", "|")
            .split("|")
            if len(c.strip().split()) >= 2
        ]
        if not raw_clauses:
            raw_clauses = [cleaned_utt]

        stop_words = self.CONTENT_STOP
        p_strip = " .,!?;:'()[]{}" + chr(34)

        # ТЕТА-ФАЗОВАЯ ПРЕЦЕССИЯ О'КИФА-ЛИСМАНА (BA46 Theta-Phase Temporal Queue):
        clause_candidates: List[Tuple[int, float, torch.Tensor]] = []
        max_resonance_seen = 0.0
        for tau_phase, cl in enumerate(raw_clauses):
            if self.dialogue_state.is_identity_query(cl) and self.dialogue_state.identity_pole_vec is not None:
                self.last_identity_profile_used = True
                clause_candidates.append((tau_phase, 2.5, self.dialogue_state.identity_pole_vec))
                max_resonance_seen = max(max_resonance_seen, 2.5)
                continue
            cl_ids = self.encode_text(cl)
            content_ids = [
                tid for tid in cl_ids
                if len(self.decode_token(tid).strip(p_strip)) >= 3
                and self.decode_token(tid).strip(p_strip).lower() not in stop_words
            ]
            if not content_ids:
                continue
            cl_vec = self.compute_stimulus_vec64(content_ids)
            cl_sdr = self.dendritic_plateau_readout._compute_cortical_sdr2048(content_ids, normalize=False)
            pole_vec, poles_info = self.dendritic_plateau_readout.clean_up_stimulus_in_ca3_attractor(
                cl_vec,
                q_sdr2048=cl_sdr,
                recent_poles=self.dialogue_state.recent_pole_vectors,
            )
            if getattr(self.dendritic_plateau_readout, "last_consolidated_boost_applied", False):
                self.last_consolidated_schema_boost_used = True
            if poles_info:
                max_resonance_seen = max(max_resonance_seen, float(poles_info[0][0]))
            if not poles_info or poles_info[0][0] < 0.25:
                continue
            clause_candidates.append((tau_phase, poles_info[0][0], pole_vec))

        # Отбираем уникальные полюса с самым высоким резонансом CA3, но возвращаем их в ХРОНОЛОГИЧЕСКОМ порядке фаз tau_phase!
        clause_candidates.sort(key=lambda item: item[1], reverse=True)
        chosen_with_phase: List[Tuple[int, float, torch.Tensor]] = []
        for tau_p, sc, p_vec in clause_candidates:
            if not any(float(torch.dot(p_vec, ex).item()) > 0.88 for _, _, ex in chosen_with_phase):
                chosen_with_phase.append((tau_p, sc, p_vec))
            if len(chosen_with_phase) >= 2:
                break
        chosen_with_phase.sort(key=lambda item: item[0])
        selected_poles: List[torch.Tensor] = [p_vec for _, _, p_vec in chosen_with_phase]
        selected_res: List[float] = [round(float(sc), 4) for _, sc, _ in chosen_with_phase]

        if len(selected_poles) < 2 and len(domains) >= 2:
            all_content_ids = [
                tid for tid in self.encode_text(cleaned_utt)
                if len(self.decode_token(tid).strip(p_strip)) >= 3
                and self.decode_token(tid).strip(p_strip).lower() not in stop_words
            ]
            sub_poles: List[Tuple[int, float, torch.Tensor]] = []
            for w_start in range(0, max(1, len(all_content_ids) - 2), 2):
                sub_ids = all_content_ids[w_start : w_start + 4]
                if len(sub_ids) >= 2:
                    sv = self.compute_stimulus_vec64(sub_ids)
                    ssdr = self.dendritic_plateau_readout._compute_cortical_sdr2048(sub_ids, normalize=False)
                    pv, p_info = self.dendritic_plateau_readout.clean_up_stimulus_in_ca3_attractor(
                        sv,
                        q_sdr2048=ssdr,
                        recent_poles=self.dialogue_state.recent_pole_vectors,
                    )
                    if getattr(self.dendritic_plateau_readout, "last_consolidated_boost_applied", False):
                        self.last_consolidated_schema_boost_used = True
                    if p_info:
                        max_resonance_seen = max(max_resonance_seen, float(p_info[0][0]))
                    if p_info and p_info[0][0] >= 0.28:
                        sub_poles.append((w_start, p_info[0][0], pv))
            sub_poles.sort(key=lambda kv: kv[1], reverse=True)
            for w_pos, sc, p_vec in sub_poles:
                if not any(float(torch.dot(p_vec, ex).item()) > 0.88 for ex in selected_poles):
                    selected_poles.append(p_vec)
                    selected_res.append(round(float(sc), 4))
                if len(selected_poles) >= 2:
                    break

        if not selected_poles:
            if self.dialogue_state.identity_pole_vec is not None and self.dialogue_state.is_identity_query(canonical_utt):
                self.last_identity_profile_used = True
                selected_poles.append(self.dialogue_state.identity_pole_vec)
                selected_res.append(2.5)
                max_resonance_seen = 2.5
            else:
                u_ids = self.encode_text(cleaned_utt)
                c_ids = [
                    tid for tid in u_ids
                    if len(self.decode_token(tid).strip(p_strip)) >= 4
                    and self.decode_token(tid).strip(p_strip).lower() not in stop_words
                ] or u_ids
                p_fallback, fb_info = self.dendritic_plateau_readout.clean_up_stimulus_in_ca3_attractor(
                    self.compute_stimulus_vec64(c_ids),
                    q_sdr2048=self.dendritic_plateau_readout._compute_cortical_sdr2048(c_ids, normalize=False),
                    recent_poles=self.dialogue_state.recent_pole_vectors,
                )
                if getattr(self.dendritic_plateau_readout, "last_consolidated_boost_applied", False):
                    self.last_consolidated_schema_boost_used = True
                fb_sc = float(fb_info[0][0]) if fb_info else 0.0
                max_resonance_seen = max(max_resonance_seen, fb_sc)
                selected_poles.append(p_fallback)
                selected_res.append(round(fb_sc, 4))

        self.last_ca3_top1_resonance = round(max_resonance_seen, 4)

        # 3. МНОГОШАГОВЫЙ ТРАНЗИТИВНЫЙ ВЫВОД В CA3 (Multi-Hop Cortical Rollout, A => B => C):
        multi_hop_triggers = ("step by step", "causal chain", "consequently", "how does that lead to", "trace the chain")
        wants_multi_hop = enable_multi_hop or any(t in lower_utt for t in multi_hop_triggers)
        if wants_multi_hop and len(selected_poles) == 1:
            u_ids_all = self.encode_text(cleaned_utt)
            s_sdr0 = self.dendritic_plateau_readout._compute_cortical_sdr2048(u_ids_all, normalize=False)
            hop2 = self.dendritic_plateau_readout.discover_second_hop_ca3_pole(
                first_pole_vec=selected_poles[0],
                stimulus_sdr2048=s_sdr0,
                recent_poles=self.dialogue_state.recent_pole_vectors,
            )
            if hop2 is not None:
                selected_poles.append(hop2[1])
                selected_res.append(round(float(hop2[0]), 4))
                self.last_multi_hop_used = True

        self._record_pole_telemetry_from_vectors(selected_poles, selected_res)
        return selected_poles

    def _record_pole_telemetry_from_vectors(
        self,
        pole_vecs: List[torch.Tensor],
        resonances: List[float],
    ) -> None:
        """Записывает точные индексы и значимости выбранных полюсов `w_ca3_attractors` для проверяемого паспорта ответа."""
        self.last_selected_ca3_pole_indices = []
        self.last_selected_ca3_pole_saliences = []
        self.last_selected_ca3_pole_resonances = list(resonances)
        attr = self.dendritic_plateau_readout.w_ca3_attractors
        sal = self.dendritic_plateau_readout.w_ca3_salience
        if attr.shape[0] == 0:
            return
        with torch.no_grad():
            for pv in pole_vecs:
                p_norm = F.normalize(pv, p=2, dim=0)
                similarities = attr @ p_norm
                idx = int(torch.argmax(similarities).item())
                if float(similarities[idx].item()) < 0.9999:
                    self.last_selected_ca3_pole_indices.append(-1)
                    self.last_selected_ca3_pole_saliences.append(1.0)
                    continue
                self.last_selected_ca3_pole_indices.append(idx)
                s_val = float(sal[idx].item()) if sal.shape[0] > idx else 1.0
                self.last_selected_ca3_pole_saliences.append(round(s_val, 4))

    def evaluate_semantic_pole_closure_and_grounding(
        self,
        question_text: str,
        response_text: str,
        pole_salience: float = 1.0,
    ) -> Dict[str, object]:
        """Предохранители A1 (Финальный санитайзер генерации) и A2 (Семантический критерий отбора полюсов >= 0.80):
        Заменяет поверхностный `grammar_coherent` строгой семантической проверкой:
          1) Синтаксическая и пунктуационная чистота (`evaluate_grammar_and_artifacts`);
          2) Запрет утечки чужой идентичности из тренировочного корпуса (`Alex` без представления в текущей сессии);
          3) Двунаправленное семантическое замыкание `semantic_closure_score >= 0.80` (пересечение концептуальных основ
             `q_stems & r_stems`, совпадение канонического домена схемы при `salience >= 4.50` или подтверждённая
             самоидентификация пользователя)."""
        grammar_ok = bool(evaluate_grammar_and_artifacts(response_text)["grammar_coherent"])
        verifier = getattr(self, "external_grounding_verifier_fn", None)
        try:
            evidence = (verifier(question_text, response_text) if callable(verifier)
                        else self._strict_content_polarity_verifier(question_text, response_text))
            if not isinstance(evidence, dict):
                raise ValueError("Grounding verifier must return a dictionary")
            verified = bool(evidence.get("grounded", evidence.get("verified", False)))
            score = float(evidence.get("semantic_closure_score", evidence.get("score", 1.0 if verified else 0.0)))
            if not math.isfinite(score):
                raise ValueError("Nonfinite grounding score")
        except Exception as exc:
            evidence = {"grounded": False, "reason": f"verifier_failed:{type(exc).__name__}"}
            verified, score = False, 0.0
        return {
            "grounded": bool(grammar_ok and verified and score >= 0.80),
            "semantic_closure_score": round(min(1.0, max(0.0, score)) if grammar_ok else 0.0, 4),
            "grammar_coherent": grammar_ok,
            "reason": str(evidence.get("reason", "external_verifier")) if grammar_ok else "grammar_incoherent",
            "grounding_verifier_result": evidence,
        }

    @staticmethod
    def _strict_content_polarity_verifier(prompt: str, response: str) -> Dict[str, object]:
        """Conservative lexical fallback, not an assertion of world truth.

        Domain membership and pole strength are never factual evidence. Hosts can
        replace this fallback with a source-backed external verifier.
        """
        import re
        stop = {"what", "how", "why", "where", "when", "who", "does", "do", "is", "are", "was", "were",
                "the", "a", "an", "in", "of", "to", "with", "can", "you", "your", "my", "his", "her",
                "and", "or", "for", "from", "that", "this", "have", "has", "explain", "please", "tell",
                "me", "about", "know", "as", "i", "it", "we", "our"}
        q_words = set(re.findall(r"[a-z0-9]+(?:'[a-z]+)?", prompt.lower()))
        r_words = set(re.findall(r"[a-z0-9]+(?:'[a-z]+)?", response.lower()))
        negatives = {"not", "never", "no", "cannot", "can't", "isn't", "doesn't", "without"}
        positive = {"increase", "increases", "activate", "activates", "promote", "promotes", "enable", "enables"}
        negative = {"decrease", "decreases", "inhibit", "inhibits", "suppress", "suppresses", "prevent", "prevents"}
        polarity_ok = bool(q_words & negatives) == bool(r_words & negatives)
        polarity_ok = polarity_ok and not ((q_words & positive and r_words & negative)
                                            or (q_words & negative and r_words & positive))
        q_content = q_words - stop - negatives
        r_content = r_words - stop - negatives
        overlap = len(q_content & r_content)
        coverage = overlap / max(1, len(q_content))
        grounded = bool(polarity_ok and overlap >= min(2, len(q_content)) and q_content and coverage >= 0.80)
        return {"grounded": grounded, "semantic_closure_score": coverage if grounded else 0.0,
                "reason": "strict_content_polarity_overlap" if grounded else "content_or_polarity_mismatch",
                "polarity_match": polarity_ok, "content_overlap": overlap}

    def execute_sws_ca3_apoptotic_pole_sanitization(self) -> Dict[str, int]:
        """Предохранитель A2 + A4 (Vectorized SWS Apoptotic Pole Sanitization):
        Векторизованно проверяет все полюса `w_ca3_attractors` путём пакетной генерации их траекторий из `w_dend_out`
        и семантического само-замыкания (`self_closure >= 0.45` со своим собственным `w_ca3_sdr` + `grammar_coherent`).
        Апоптотически удаляет повреждённые фоновые полюса («окрошку» из OpenWebText), чтобы они не засоряли `CA3` и `w_core_dend_out`."""
        ro = self.dendritic_plateau_readout
        n_before = int(ro.w_ca3_attractors.shape[0])
        n_act = len(ro.col_to_tok)
        if n_before == 0 or n_act == 0:
            return {"poles_before": n_before, "poles_after": n_before, "pruned_garbage_poles": 0}

        if not getattr(self, "learning_enabled", True):
            return {"poles_before": n_before, "poles_after": n_before, "pruned_garbage_poles": 0}
        keep_indices: List[int] = []
        with torch.no_grad():
            q_mat = F.normalize(ro.w_ca3_attractors, p=2, dim=-1)
            vec_bank = torch.stack([self.get_vec64(ro.col_to_tok[c]) for c in range(n_act)])

            hist_vecs: List[torch.Tensor] = []
            gen_cols: List[torch.Tensor] = []
            for pos_idx in range(40):
                vp2 = q_mat if pos_idx < 3 else hist_vecs[-3]
                vp1 = q_mat if pos_idx < 2 else hist_vecs[-2]
                vc = q_mat if pos_idx < 1 else hist_vecs[-1]
                p_vec = ro.pos_phase_vecs[pos_idx].unsqueeze(0)
                b1 = torch.roll(vp1, shifts=2, dims=-1) * torch.roll(vc, shifts=1, dims=-1) * p_vec * q_mat
                b2 = torch.roll(vp2, shifts=3, dims=-1) * torch.roll(vc, shifts=1, dims=-1) * torch.roll(q_mat, shifts=4, dims=-1)
                b3 = vc * p_vec * torch.roll(q_mat, shifts=2, dims=-1)
                b4 = torch.roll(vp1, shifts=1, dims=-1) * p_vec * q_mat
                holo_in = F.normalize(torch.cat([b1, b2, b3, b4], dim=-1), p=2, dim=-1)
                raw_d = F.relu(holo_in @ ro.w_basal_branch)
                top_v, top_i = torch.topk(raw_d, k=min(32, ro.d_dend), dim=-1)
                d_sparse = torch.zeros_like(raw_d)
                d_sparse.scatter_(1, top_i, top_v * top_v)
                d_nmda = F.normalize(d_sparse, p=2, dim=-1)
                logits = torch.stack([ro._sparse_logits(d, ro.w_dend_out, n_act)
                                      + 0.35 * ro._sparse_logits(d, ro.w_core_dend_out, n_act) for d in d_nmda])
                best_c = torch.argmax(logits, dim=-1)
                gen_cols.append(best_c)
                hist_vecs.append(vec_bank.index_select(0, best_c))

            gen_mat = torch.stack(gen_cols, dim=1).tolist()
            for idx, cols in enumerate(gen_mat):
                tids = [ro.col_to_tok[c] for c in cols]
                cut: List[int] = []
                for t in tids:
                    cut.append(t)
                    if len(cut) >= 8 and self.decode_token(t).strip().endswith((".", "!", "?")):
                        break
                txt = self.decode_tokens(cut).strip()
                if not bool(evaluate_grammar_and_artifacts(txt, min_words=5)["grammar_coherent"]):
                    continue
                if idx < ro.w_ca3_sdr.shape[0]:
                    gen_sdr = ro._compute_cortical_sdr2048(cut, normalize=True)
                    pole_sdr = F.normalize(ro.w_ca3_sdr[idx].float() + 1e-8, p=2, dim=0)
                    self_closure = float(torch.dot(gen_sdr, pole_sdr).item())
                    if self_closure >= 0.45:
                        keep_indices.append(idx)
                else:
                    keep_indices.append(idx)

            if keep_indices and len(keep_indices) < n_before:
                k_t = torch.tensor(keep_indices, dtype=torch.long)
                ro.w_ca3_attractors = ro.w_ca3_attractors.index_select(0, k_t).contiguous()
                if ro.w_ca3_sdr.shape[0] == n_before:
                    ro.w_ca3_sdr = ro.w_ca3_sdr.index_select(0, k_t).contiguous()
                if ro.w_ca3_salience.shape[0] == n_before:
                    ro.w_ca3_salience = ro.w_ca3_salience.index_select(0, k_t).contiguous()
                ro.w_ca3_episode_links.clear()

        n_after = int(ro.w_ca3_attractors.shape[0])
        return {
            "poles_before": n_before,
            "poles_after": n_after,
            "pruned_garbage_poles": n_before - n_after,
        }

    def _detect_tpj_shift_mode(self, stimulus_ids: List[int]) -> str:
        stim_set = set(int(t) for t in stimulus_ids)
        has_1st = bool(stim_set & self.ba44_syntax_bridge.tpj_1st_trigger_ids)
        if not has_1st and bool(stim_set & self.ba44_syntax_bridge.tpj_3rd_fem_trigger_ids):
            return "feminine_3rd_she_her"
        if not has_1st and bool(stim_set & self.ba44_syntax_bridge.tpj_3rd_trigger_ids):
            return "masculine_3rd_he_his"
        return "none"

    def _build_response_passport(
        self,
        stimulus_ids: List[int],
        proportional_gate_zone: str,
        son_vte_grounding: str,
        think_trace: str = "",
        episodic_recall: bool = False,
        endogenous_cortical_think: bool = False,
        counterfactual_simulation: bool = False,
        thalamic_speech_act: str = "ACT_DIRECT_RECALL",
        arcuate_injections: int = 0,
        ba44_bridge: str = "",
    ) -> Dict[str, object]:
        """Формирует детерминированный проверяемый Паспорт Ответа (`response_passport`),
        фиксирующий путь к мастер-чекпоинту, индексы и значимости выбранных полюсов `CA3`,
        динамическую траекторию 16 колонок коры (`cortical_trajectory_trace`), речевой акт `MD-Thalamus`
        и точный список всех сработавших биофизических механизмов."""
        return {
            "master_ckpt": str(getattr(self, "loaded_checkpoint_path", "")),
            "selected_ca3_pole_indices": list(getattr(self, "last_selected_ca3_pole_indices", [])),
            "selected_ca3_pole_saliences": list(getattr(self, "last_selected_ca3_pole_saliences", [])),
            "selected_ca3_pole_resonances": list(getattr(self, "last_selected_ca3_pole_resonances", [])),
            "proportional_gate_zone": proportional_gate_zone,
            "thalamic_speech_act": thalamic_speech_act,
            "cortical_trajectory_trace": list(getattr(self, "last_cortical_trajectory_trace", [])),
            "think_trace": think_trace,
            "mechanisms_triggered": {
                "identity_profile": bool(getattr(self, "last_identity_profile_used", False)),
                "tpj_shift": self._detect_tpj_shift_mode(stimulus_ids),
                "consolidated_schema_boost": bool(getattr(self, "last_consolidated_schema_boost_used", False)),
                "son_vte_grounding": son_vte_grounding,
                "episodic_recall": bool(episodic_recall),
                "endogenous_cortical_think": bool(endogenous_cortical_think),
                "counterfactual_simulation": bool(counterfactual_simulation),
                "arcuate_synaptic_pointer": bool(arcuate_injections > 0),
                "ba44_syntax_bridge": ba44_bridge,
            },
        }

    def _attempt_endogenous_cortical_think_reformulation(
        self,
        user_utterance: str,
        slot_bindings: Dict[str, str],
        max_tokens: int = 64,
    ) -> Optional[Dict[str, object]]:
        """Пропорциональный Эпистемический Гейт (`Zone 2 — THINK: 'I know X as Y'`):
        Разделяет «не знаю (`Zone 3`)» и «знаю концепт $X$ в самоорганизовавшейся коре/слотах/графе, но не нашла
        готовой заученной формулировки в `CA3` для короткого или нестандартного вопроса (`Zone 2`)».
        Вместо того чтобы просить пользователя `share the key mechanism`, запускает внутренний цикл `THINK`,
        находит канонический концепт $Y$, синтезирует его кортикальную схему с подстановкой $X$ и отвечает:
        `'I know <X> as <Y>: <synthesized_explanation>'`."""
        resolved_x: Optional[str] = None
        resolved_y: Optional[str] = None
        schema_cue_str: Optional[str] = None
        think_trace: str = ""

        ext_fn = getattr(self, "external_think_resolver_fn", None)
        if callable(ext_fn):
            ext_info = ext_fn(user_utterance)
            if isinstance(ext_info, dict) and bool(ext_info.get("known_in_cortex")):
                if "direct_graph_sentence" in ext_info:
                    gx = str(ext_info["query_concept_x"])
                    gy = str(ext_info["canonical_concept_y"])
                    gsent = str(ext_info["direct_graph_sentence"])
                    body_low = gsent[0].lower() + gsent[1:] if len(gsent) > 1 else gsent
                    result = {
                        "response_text": f"I know {gx} as {gy}: {body_low}",
                        "think_trace": str(ext_info.get("think_trace", f"Zone2_WorldGraphThink: {gx} -> {gy}")),
                        "bindings": {gy: gx},
                    }
                    check = self.evaluate_semantic_pole_closure_and_grounding(user_utterance, result["response_text"])
                    return result if bool(check["grounded"]) else None
                resolved_x = str(ext_info["query_concept_x"])
                resolved_y = str(ext_info["canonical_concept_y"])
                cues = ext_info.get("schema_cue_words", [])
                schema_cue_str = " ".join(cues[:6]) if isinstance(cues, list) and cues else resolved_y
                think_trace = str(ext_info.get("think_trace", f"Zone2_STDPThink: {resolved_x} -> {resolved_y}"))

        if resolved_x is None and slot_bindings:
            for canon_y, user_x in slot_bindings.items():
                resolved_y = canon_y
                resolved_x = user_x
                schema_cue_str = canon_y
                think_trace = f"Zone2_RoleSlotThink: resolved known entity {user_x!r} -> canonical role {canon_y!r}"
                break

        if resolved_x is None or resolved_y is None or not schema_cue_str:
            return None

        cue_ids = self.encode_text(schema_cue_str)
        cue_vec = self.compute_stimulus_vec64(cue_ids)
        cue_sdr = self.dendritic_plateau_readout._compute_cortical_sdr2048(cue_ids, normalize=False)
        pole_y, pole_info = self.dendritic_plateau_readout.clean_up_stimulus_in_ca3_attractor(
            cue_vec,
            q_sdr2048=cue_sdr,
        )
        res_score = float(pole_info[0][0]) if pole_info else 0.0
        self._record_pole_telemetry_from_vectors([pole_y], [round(res_score, 4)])

        merged_bindings = dict(slot_bindings)
        merged_bindings[resolved_y] = resolved_x
        arc_map = self.role_filler_binder.build_arcuate_token_pointer_map(merged_bindings, self.encode_text)
        synth_ids, _, _ = self._synthesize_single_beam_from_weights_only(
            q_vec=pole_y,
            l5_vec=F.normalize(torch.roll(pole_y, shifts=1, dims=-1), p=2, dim=-1),
            onset_pair=(0, 0),
            temperature_jitter=0.0,
            max_tokens=max_tokens,
            use_ngrams=False,
            arcuate_pointer_map=arc_map,
        )
        u_ids = self.encode_text(user_utterance)
        synth_ids = self.ba44_syntax_bridge.apply_tpj_deictic_perspective_shift(
            clause_ids=synth_ids,
            stimulus_ids=u_ids,
            decode_fn=self.decode_token,
        )
        body_txt = self.decode_tokens(synth_ids).strip()
        if not body_txt or not bool(evaluate_grammar_and_artifacts(body_txt)["grammar_coherent"]):
            return None
        body_low = body_txt[0].lower() + body_txt[1:] if len(body_txt) > 1 else body_txt
        result = {
            "response_text": f"I know {resolved_x} as {resolved_y}: {body_low}",
            "think_trace": think_trace,
            "bindings": merged_bindings,
            "pole_vec": pole_y,
        }
        check = self.evaluate_semantic_pole_closure_and_grounding(user_utterance, result["response_text"])
        return result if bool(check["grounded"]) else None

    def pure_neural_generate_response(
        self,
        user_utterance: str,
        max_tokens: int = 64,
        use_ngrams: Optional[bool] = None,
        enable_multi_hop: bool = False,
        enable_active_inference: bool = False,
        enable_episode_chain: bool = False,
        role_filler_bindings: Optional[Dict[str, str]] = None,
    ) -> Dict[str, object]:
        r"""Генерирует ответ собеседнику в выбранном режиме (`use_ngrams = False` [OFF, 100% 8192-NMDA Matrix] по умолчанию):
          - При `use_ngrams = False` (OFF): 0% обращений к словарям N-грамм — весь ответ от 1-го до последнего слова
            вычисляется матрицей 8,192 нелинейных дендритных NMDA-ветвей `DendriticPlateauMatrixReadout` + `W_CA3`!
          - Включает Пропорциональный Эпистемический Гейт с эндогенным режимом `THINK` (`Zone 1: Direct Recall`,
            `Zone 2: 'I know X as Y'`, `Zone 3: True Unknown Clarify`) и детерминированный `response_passport`."""
        t0 = time.perf_counter()
        self.last_cortical_trajectory_trace = []
        self.spiking_language_loop.last_spike_raster = torch.empty((0, self.moe_cortex.n_columns))
        ngrams_enabled = self.use_ngrams if use_ngrams is None else bool(use_ngrams)
        self.last_arcuate_token_injections = 0
        self.last_reafferent_stranger_interventions = 0
        imprinted = self._online_dopamine_imprint_from_user(user_utterance)

        u_ids = self.encode_text(user_utterance)
        q_raw_vec = self.compute_stimulus_vec64(u_ids)

        # А. Проверка мета-запроса к Эпизодическому Графу Рабочей Памяти Диалога (BA46 Episodic Graph):
        if self.dialogue_state.is_episodic_history_query(user_utterance):
            ep_resp = self.dialogue_state.synthesize_episodic_memory_response(user_utterance)
            passport = self._build_response_passport(
                stimulus_ids=u_ids,
                proportional_gate_zone="ZONE_1_EPISODIC_RECALL",
                son_vte_grounding="grounded_episodic_working_memory",
                think_trace=f"BA46_EpisodicGraph: recalled {len(self.dialogue_state.episodic_turns)} turns",
                episodic_recall=True,
            )
            self._record_dialogue_turn(
                stimulus_vec64=q_raw_vec,
                activated_poles=[q_raw_vec],
                utterance=user_utterance,
                response_text=ep_resp,
                role_filler_bindings={},
                selected_ca3_pole_indices=[],
                response_passport=passport,
            )
            return {
                "stimulus": user_utterance,
                "ngrams_mode": "OFF (Episodic Working Memory Graph Recall)",
                "active_moe_columns": ["Col9_PFC_BA46", "Col15_METACOG_BA10"],
                "conversational_domain": "episodic_working_memory",
                "remembered_user_name": self.dialogue_state.user_name,
                "role_filler_bindings_applied": {},
                "arcuate_token_pointer_injections": 0,
                "reafferent_stranger_check_interventions": 0,
                "episode_chain_sentences_generated": 1,
                "multi_hop_rollout_used": False,
                "active_inference_clarification_triggered": False,
                "endogenous_cortical_think_triggered": False,
                "ca3_top1_resonance": 1.0,
                "epistemic_uncertainty": 0.0,
                "cortical_moe_branch_profile": {},
                "ba44_syntax_bridge_connector": "",
                "swr_consolidated_into_nmda_matrix": False,
                "online_dopamine_imprinted_tokens": imprinted,
                "vte_beams_evaluated": [],
                "selected_beam_idx": 0,
                "stopped_naturally_by_son_eos": True,
                "generated_tokens_count": len(self.encode_text(ep_resp)),
                "pure_weight_generated_response": ep_resp,
                "response_passport": passport,
                "stored_sentence_chains_in_class": 0,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        slot_bindings = self.role_filler_binder.extract_slot_bindings(user_utterance)
        slot_bindings.update(role_filler_bindings or {})
        arcuate_pointer_map = self.role_filler_binder.build_arcuate_token_pointer_map(slot_bindings, self.encode_text)
        q_vec, ca3_clause_poles, moe_out = self._prepare_language_query(
            user_utterance, q_raw_vec, use_ngrams=ngrams_enabled,
            enable_multi_hop=enable_multi_hop, role_filler_bindings=slot_bindings)

        if imprinted > 0:
            self.last_identity_profile_used = True

        col_names = [f"{c.zcfg.name}_{idx}" for idx, c in enumerate(self.moe_cortex.columns)]
        cortical_branch_profile = self.dendritic_plateau_readout.compute_cortical_moe_branch_profile(
            q_vec=q_vec,
            expert_gates=moe_out.gating_weights,
            expert_names=col_names,
        )

        # Б. ПРОПОРЦИОНАЛЬНЫЙ ЭПИСТЕМИЧЕСКИЙ ГЕЙТ ПРИ НИЗКОМ РЕЗОНАНСЕ CA3 (< 0.38):
        epistemic_uncertainty = round(max(0.0, 1.0 - min(1.0, self.last_ca3_top1_resonance / 0.55)), 4)
        if (
            enable_active_inference
            and imprinted == 0
            and self.last_ca3_top1_resonance < 0.38
            and not self.dialogue_state.is_identity_query(user_utterance)
        ):
            think_res = self._attempt_endogenous_cortical_think_reformulation(
                user_utterance=user_utterance,
                slot_bindings=slot_bindings,
                max_tokens=max_tokens,
            )
            if think_res is not None and bool(self.evaluate_semantic_pole_closure_and_grounding(
                user_utterance, str(think_res["response_text"]))["grounded"]):
                t_resp = str(think_res["response_text"])
                t_trace = str(think_res["think_trace"])
                t_bind = dict(think_res.get("bindings", slot_bindings))  # type: ignore[arg-type]
                passport = self._build_response_passport(
                    stimulus_ids=u_ids,
                    proportional_gate_zone="ZONE_2_KNOWN_REFORMULATE_THINK",
                    son_vte_grounding="known_reformulated_i_know_x_as_y",
                    think_trace=t_trace,
                    endogenous_cortical_think=True,
                    arcuate_injections=len(t_bind),
                )
                self._record_dialogue_turn(
                    stimulus_vec64=q_raw_vec,
                    activated_poles=[think_res.get("pole_vec", q_vec)],  # type: ignore[list-item]
                    utterance=user_utterance,
                    response_text=t_resp,
                    role_filler_bindings=t_bind,
                    selected_ca3_pole_indices=list(self.last_selected_ca3_pole_indices),
                    response_passport=passport,
                )
                return {
                    "stimulus": user_utterance,
                    "ngrams_mode": "OFF (Proportional Gate Zone 2: Endogenous Cortical THINK 'I know X as Y')",
                    "active_moe_columns": moe_out.active_expert_names,
                    "conversational_domain": self.dialogue_state.active_domain,
                    "remembered_user_name": self.dialogue_state.user_name,
                    "role_filler_bindings_applied": t_bind,
                    "arcuate_token_pointer_injections": len(t_bind),
                    "reafferent_stranger_check_interventions": self.last_reafferent_stranger_interventions,
                    "episode_chain_sentences_generated": 1,
                    "multi_hop_rollout_used": False,
                    "active_inference_clarification_triggered": False,
                    "endogenous_cortical_think_triggered": True,
                    "ca3_top1_resonance": self.last_ca3_top1_resonance,
                    "epistemic_uncertainty": epistemic_uncertainty,
                    "cortical_moe_branch_profile": cortical_branch_profile,
                    "ba44_syntax_bridge_connector": "",
                    "swr_consolidated_into_nmda_matrix": False,
                    "online_dopamine_imprinted_tokens": 0,
                    "vte_beams_evaluated": [],
                    "selected_beam_idx": 0,
                    "stopped_naturally_by_son_eos": True,
                    "generated_tokens_count": len(self.encode_text(t_resp)),
                    "pure_weight_generated_response": t_resp,
                    "response_passport": passport,
                    "stored_sentence_chains_in_class": 0,
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

            p_strip = " .,!?;:'()[]{}" + chr(34)
            stop_q = {"what", "how", "why", "where", "when", "who", "does", "do", "is", "are", "the", "a", "an", "in", "of", "to", "with"}
            novel_terms = [
                w.strip(p_strip)
                for w in user_utterance.split()
                if len(w.strip(p_strip)) >= 4 and w.strip(p_strip).lower() not in stop_q
            ][:3]
            topic_str = " ".join(novel_terms) if novel_terms else "this new topic"
            clarify_resp = (
                f"I notice you are asking about {topic_str}, which has high epistemic uncertainty in my current CA3 memory; "
                f"could you share the key mechanism or principle so I can consolidate it into my dendritic synapses?"
            )
            abductive_trace = f"Zone3_AbductiveClarify: no cortical/STDP trace for {topic_str!r} (CA3_res={self.last_ca3_top1_resonance:.3f})"
            passport = self._build_response_passport(
                stimulus_ids=u_ids,
                proportional_gate_zone="ZONE_3_UNKNOWN_CLARIFY",
                son_vte_grounding="ungrounded_clarify",
                think_trace=abductive_trace,
            )
            self._record_dialogue_turn(
                stimulus_vec64=q_raw_vec,
                activated_poles=[q_raw_vec],
                utterance=user_utterance,
                response_text=clarify_resp,
                role_filler_bindings=slot_bindings,
                topic_tags=[t.lower() for t in novel_terms],
                was_clarification_request=True,
                selected_ca3_pole_indices=list(self.last_selected_ca3_pole_indices),
                response_passport=passport,
            )
            return {
                "stimulus": user_utterance,
                "ngrams_mode": "OFF (SON Active Inference Epistemic Clarification)",
                "active_moe_columns": moe_out.active_expert_names,
                "conversational_domain": self.dialogue_state.active_domain,
                "remembered_user_name": self.dialogue_state.user_name,
                "role_filler_bindings_applied": slot_bindings,
                "arcuate_token_pointer_injections": 0,
                "reafferent_stranger_check_interventions": self.last_reafferent_stranger_interventions,
                "episode_chain_sentences_generated": 1,
                "multi_hop_rollout_used": False,
                "active_inference_clarification_triggered": True,
                "endogenous_cortical_think_triggered": False,
                "ca3_top1_resonance": self.last_ca3_top1_resonance,
                "epistemic_uncertainty": epistemic_uncertainty,
                "cortical_moe_branch_profile": cortical_branch_profile,
                "ba44_syntax_bridge_connector": "",
                "swr_consolidated_into_nmda_matrix": False,
                "online_dopamine_imprinted_tokens": 0,
                "vte_beams_evaluated": [],
                "selected_beam_idx": 0,
                "stopped_naturally_by_son_eos": True,
                "generated_tokens_count": len(self.encode_text(clarify_resp)),
                "pure_weight_generated_response": clarify_resp,
                "response_passport": passport,
                "stored_sentence_chains_in_class": 0,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        l5_vec = F.normalize(torch.roll(q_vec, shifts=1, dims=-1), p=2, dim=-1)
        self.son_column.reset_for_new_stimulus(q_vec, initial_epistemic_drive=1.0)

        if ngrams_enabled:
            scored_onsets: List[Tuple[float, Tuple[int, int]]] = []
            for op_pair, ctx_v in self.onset_pair_ctx_vec.items():
                cos_on = float(torch.dot(q_vec, ctx_v).item())
                g_on = self.onset_pair_synapses.get(op_pair[0], {}).get(op_pair[1], 1.0)
                scored_onsets.append((cos_on * 3.5 + 0.25 * g_on, op_pair))
            scored_onsets.sort(key=lambda kv: kv[0], reverse=True)
            top_onsets = [pair for _, pair in scored_onsets[:3]] or [(262, 318)]
        else:
            top_onsets = [(0, 0)]

        vte_candidates: List[Tuple[str, List[int], torch.Tensor, float, float]] = []
        beams_stopped: List[bool] = []
        beam_dynamic_states: Dict[int, Dict[str, object]] = {}
        beam_sequence_scores: Dict[int, float] = {}
        beam_grounding_results: Dict[int, Dict[str, object]] = {}
        beam_finish_reasons: Dict[int, str] = {}
        ba44_bridge_used = ""
        for b_idx, on_pair in enumerate(top_onsets):
            jitter = 0.0 if (b_idx == 0 or not ngrams_enabled) else (0.03 * b_idx)
            ptr_arg = arcuate_pointer_map if b_idx == 0 else arcuate_pointer_map
            if not ngrams_enabled and len(ca3_clause_poles) > 1:
                clause_parts: List[List[int]] = []
                total_sup = 0.0
                stopped_flag = False
                for p_pole in ca3_clause_poles:
                    part_ids, part_sup, part_stop = self._synthesize_single_beam_from_weights_only(
                        q_vec=p_pole,
                        l5_vec=F.normalize(torch.roll(p_pole, shifts=1, dims=-1), p=2, dim=-1),
                        onset_pair=(0, 0),
                        temperature_jitter=jitter,
                        max_tokens=max_tokens,
                        use_ngrams=False,
                        arcuate_pointer_map=ptr_arg,
                    )
                    part_ids = self.ba44_syntax_bridge.apply_tpj_deictic_perspective_shift(
                        clause_ids=part_ids,
                        stimulus_ids=u_ids,
                        decode_fn=self.decode_token,
                    )
                    if clause_parts:
                        part_txt_check = self.decode_tokens(part_ids).strip()
                        cl_sem = self.evaluate_semantic_pole_closure_and_grounding(
                            question_text=user_utterance,
                            response_text=part_txt_check,
                            pole_salience=max(self.last_selected_ca3_pole_saliences, default=1.0),
                        )
                        if not bool(cl_sem["grounded"]):
                            continue
                    if not clause_parts or part_ids != clause_parts[0]:
                        clause_parts.append(part_ids)
                        total_sup += part_sup
                        stopped_flag = stopped_flag or part_stop == "son_eos"

                if len(clause_parts) >= 2:
                    bridge_phrase, bridge_ids, _ = self.ba44_syntax_bridge.select_bridge_tokens_between_attractors(
                        pole1_vec=ca3_clause_poles[0],
                        pole2_vec=ca3_clause_poles[1],
                        stimulus_ids=u_ids,
                        get_vec64_fn=self.get_vec64,
                    )
                    ba44_bridge_used = bridge_phrase
                    beam_ids = self.ba44_syntax_bridge.bridge_two_attractor_clauses_in_ba44(
                        clause1_ids=clause_parts[0],
                        clause2_ids=clause_parts[1],
                        bridge_token_ids=bridge_ids,
                        decode_fn=self.decode_token,
                        bridge_phrase=bridge_phrase,
                        stimulus_ids=u_ids,
                    )
                    beam_ids = self.ba44_syntax_bridge.apply_tpj_deictic_perspective_shift(
                        clause_ids=beam_ids,
                        stimulus_ids=u_ids,
                        decode_fn=self.decode_token,
                    )
                else:
                    beam_ids = clause_parts[0] if clause_parts else []
                syn_sup = total_sup / max(1, len(clause_parts))
            else:
                if not ngrams_enabled:
                    neural_beams = self._synthesize_neural_beam_search(
                        q_vec=q_vec,
                        l5_vec=l5_vec,
                        onset_pair=on_pair,
                        max_tokens=max_tokens,
                        arcuate_pointer_map=ptr_arg,
                        beam_width=3,
                        branch_factor=3,
                    )
                    prepared_beams: List[Tuple[Dict[str, object], List[int], Dict[str, object]]] = []
                    for neural_beam in neural_beams:
                        candidate_ids = self.ba44_syntax_bridge.apply_tpj_deictic_perspective_shift(
                            clause_ids=list(neural_beam["out_ids"]),
                            stimulus_ids=u_ids,
                            decode_fn=self.decode_token,
                        )
                        if bool(neural_beam["finished"]):
                            grounding = self.evaluate_semantic_pole_closure_and_grounding(
                                question_text=user_utterance,
                                response_text=self.decode_tokens(candidate_ids).strip(),
                                pole_salience=max(self.last_selected_ca3_pole_saliences, default=1.0),
                            )
                        else:
                            grounding = {
                                "grounded": False,
                                "semantic_closure_score": 0.0,
                                "reason": "unfinished_beam",
                            }
                        prepared_beams.append((neural_beam, candidate_ids, grounding))
                    grounded_beams = [item for item in prepared_beams if bool(item[2]["grounded"])]
                    for neural_idx, (neural_beam, beam_ids, grounding) in enumerate(grounded_beams or prepared_beams):
                        candidate_idx = len(vte_candidates)
                        syn_sup = float(neural_beam.get("mean_syn_support", 0.0))
                        beam_vec = self.compute_stimulus_vec64(beam_ids)
                        vte_candidates.append((
                            f"ngrams_OFF_stateful_neural_beam_{b_idx}_{neural_idx}",
                            beam_ids, beam_vec, syn_sup, 0.05,
                        ))
                        beams_stopped.append(bool(neural_beam["stopped_by_son"]))
                        beam_dynamic_states[candidate_idx] = neural_beam["dynamic_state"]  # type: ignore[assignment]
                        beam_sequence_scores[candidate_idx] = float(neural_beam["normalized_score"])
                        beam_grounding_results[candidate_idx] = grounding
                        beam_finish_reasons[candidate_idx] = str(neural_beam["finish_reason"])
                    continue
                beam_ids, syn_sup, finish_reason = self._synthesize_single_beam_from_weights_only(
                    q_vec=q_vec,
                    l5_vec=l5_vec,
                    onset_pair=on_pair,
                    temperature_jitter=jitter,
                    max_tokens=max_tokens,
                    use_ngrams=ngrams_enabled,
                    arcuate_pointer_map=ptr_arg,
                )
                stopped_flag = finish_reason == "son_eos"
                beam_finish_reasons[len(vte_candidates)] = finish_reason
                beam_ids = self.ba44_syntax_bridge.apply_tpj_deictic_perspective_shift(
                    clause_ids=beam_ids,
                    stimulus_ids=u_ids,
                    decode_fn=self.decode_token,
                )
            beam_vec = self.compute_stimulus_vec64(beam_ids)
            mode_lbl = "ngrams_ON_beam" if ngrams_enabled else "ngrams_OFF_pure_dendritic_matrix_beam"
            vte_candidates.append((f"{mode_lbl}_{b_idx}", beam_ids, beam_vec, syn_sup, 0.05))
            beams_stopped.append(stopped_flag)

        best_idx, vte_logs = self.son_column.evaluate_vte_inner_speech_trajectories(
            stimulus_vec64=q_vec,
            candidate_trajectories=vte_candidates,
        )
        if beam_sequence_scores and len(beam_sequence_scores) == len(vte_candidates):
            for log in vte_logs:
                candidate_idx = int(log["trajectory_idx"])
                sequence_score = beam_sequence_scores[candidate_idx]
                combined_score = float(log["son_free_energy_F"]) + 0.35 * math.tanh(-sequence_score / 4.0)
                log["sequence_normalized_score"] = round(sequence_score, 6)
                log["combined_selection_score"] = round(combined_score, 6)
                log["pre_vte_grounded"] = bool(beam_grounding_results[candidate_idx]["grounded"])
                log["beam_finish_reason"] = beam_finish_reasons[candidate_idx]
            best_idx = min(
                range(len(vte_logs)),
                key=lambda idx: (float(vte_logs[idx]["combined_selection_score"]), idx),
            )
        self.son_column.note_vte_selection(best_idx)
        if best_idx in beam_dynamic_states:
            total_vte_simulations = self.son_column.total_vte_simulations
            self.load_language_dynamic_state(beam_dynamic_states[best_idx])
            self.son_column.total_vte_simulations = total_vte_simulations
        chosen_ids = list(vte_candidates[best_idx][1])
        sentences_generated = 1

        # Г. ИЕРАРХИЧЕСКАЯ ЦЕПОЧКА ЭПИЗОДОВ В CA3 (Sequential Attractor Chains, Point 4):
        wants_paragraph = enable_episode_chain or any(
            m in user_utterance.lower() for m in ("in detail", "elaborate", "full paragraph", "two sentences")
        )
        if wants_paragraph and not ngrams_enabled and len(ca3_clause_poles) == 1:
            next_ep_pole = self.dendritic_plateau_readout.get_linked_next_episode_pole(ca3_clause_poles[0])
            if next_ep_pole is not None:
                sent2_ids, _, _ = self._synthesize_single_beam_from_weights_only(
                    q_vec=next_ep_pole,
                    l5_vec=F.normalize(torch.roll(next_ep_pole, shifts=1, dims=-1), p=2, dim=-1),
                    onset_pair=(0, 0),
                    temperature_jitter=0.0,
                    max_tokens=max_tokens,
                    use_ngrams=False,
                    arcuate_pointer_map=arcuate_pointer_map,
                )
                if sent2_ids:
                    first_s2 = self.decode_token(sent2_ids[0])
                    if not first_s2.startswith(" "):
                        space_ids = self.encode_text(" " + first_s2.strip())
                        if space_ids:
                            sent2_ids = space_ids + sent2_ids[1:]
                    chosen_ids.extend(sent2_ids)
                    sentences_generated = 2

        response_text = self.decode_tokens(chosen_ids).strip()
        top_pole_sal = max(self.last_selected_ca3_pole_saliences, default=1.0)
        sem_check = self.evaluate_semantic_pole_closure_and_grounding(
            question_text=user_utterance,
            response_text=response_text,
            pole_salience=top_pole_sal,
        )
        has_grounding = bool(sem_check["grounded"])
        if beam_grounding_results:
            has_grounding = has_grounding and bool(
                beam_grounding_results.get(best_idx, {}).get("grounded", False))
        closure_score_val = float(sem_check["semantic_closure_score"])  # type: ignore[arg-type]

        # Г2. ФИНАЛЬНЫЙ САНИТАЙЗЕР ГЕНЕРАЦИИ И СЕМАНТИЧЕСКИЙ ГЕЙТ ЗАМЫКАНИЯ (A1 + A2: closure >= 0.80):
        if not has_grounding:
            # Disabling active inference never disables the verifier/return safety gate.
            think_res = self._attempt_endogenous_cortical_think_reformulation(
                user_utterance=user_utterance,
                slot_bindings=slot_bindings,
                max_tokens=max_tokens,
            )
            if think_res is not None and bool(self.evaluate_semantic_pole_closure_and_grounding(
                user_utterance, str(think_res["response_text"]))["grounded"]):
                t_resp = str(think_res["response_text"])
                t_trace = str(think_res["think_trace"])
                t_bind = dict(think_res.get("bindings", slot_bindings))  # type: ignore[arg-type]
                passport = self._build_response_passport(
                    stimulus_ids=u_ids,
                    proportional_gate_zone="ZONE_2_KNOWN_REFORMULATE_THINK",
                    son_vte_grounding="known_reformulated_i_know_x_as_y",
                    think_trace=t_trace,
                    endogenous_cortical_think=True,
                    arcuate_injections=len(t_bind),
                )
                self._record_dialogue_turn(
                    stimulus_vec64=q_raw_vec,
                    activated_poles=[think_res.get("pole_vec", q_vec)],  # type: ignore[list-item]
                    utterance=user_utterance,
                    response_text=t_resp,
                    role_filler_bindings=t_bind,
                    is_grounded_conclusion=True,
                    selected_ca3_pole_indices=list(self.last_selected_ca3_pole_indices),
                    response_passport=passport,
                )
                return {
                    "stimulus": user_utterance,
                    "ngrams_mode": "OFF (Proportional Gate Zone 2: Endogenous Cortical THINK 'I know X as Y')",
                    "active_moe_columns": moe_out.active_expert_names,
                    "conversational_domain": self.dialogue_state.active_domain,
                    "remembered_user_name": self.dialogue_state.user_name,
                    "role_filler_bindings_applied": t_bind,
                    "arcuate_token_pointer_injections": len(t_bind),
                    "reafferent_stranger_check_interventions": self.last_reafferent_stranger_interventions,
                    "episode_chain_sentences_generated": 1,
                    "multi_hop_rollout_used": False,
                    "active_inference_clarification_triggered": False,
                    "endogenous_cortical_think_triggered": True,
                    "ca3_top1_resonance": self.last_ca3_top1_resonance,
                    "epistemic_uncertainty": max(epistemic_uncertainty, 0.35),
                    "cortical_moe_branch_profile": cortical_branch_profile,
                    "ba44_syntax_bridge_connector": "",
                    "swr_consolidated_into_nmda_matrix": False,
                    "online_dopamine_imprinted_tokens": 0,
                    "vte_beams_evaluated": vte_logs,
                    "selected_beam_idx": best_idx,
                    "stopped_naturally_by_son_eos": True,
                    "generated_tokens_count": len(self.encode_text(t_resp)),
                    "pure_weight_generated_response": t_resp,
                    "response_passport": passport,
                    "stored_sentence_chains_in_class": 0,
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

            p_strip = " .,!?;:'()[]{}" + chr(34)
            stop_q = {"what", "how", "why", "where", "when", "who", "does", "do", "is", "are", "the", "a", "an", "in", "of", "to", "with", "can", "you"}
            novel_terms = [
                w.strip(p_strip)
                for w in user_utterance.split()
                if len(w.strip(p_strip)) >= 4 and w.strip(p_strip).lower() not in stop_q
            ][:3]
            topic_str = " ".join(novel_terms) if novel_terms else "this new topic"
            clarify_resp = (
                f"I notice you are asking about {topic_str}, which has high epistemic uncertainty in my current CA3 memory; "
                f"could you share the key mechanism or principle so I can consolidate it into my dendritic synapses?"
            )
            abductive_trace = (
                f"Zone3_AbductiveClarify: draft rejected by Semantic Sanitizer A1/A2 "
                f"(reason={sem_check['reason']}, closure={closure_score_val:.2f}) and {topic_str!r} not in STDP ontology"
            )
            passport = self._build_response_passport(
                stimulus_ids=u_ids,
                proportional_gate_zone="ZONE_3_UNKNOWN_CLARIFY",
                son_vte_grounding="ungrounded_clarify",
                think_trace=abductive_trace,
            )
            self._record_dialogue_turn(
                stimulus_vec64=q_raw_vec,
                activated_poles=[q_raw_vec],
                utterance=user_utterance,
                response_text=clarify_resp,
                role_filler_bindings=slot_bindings,
                topic_tags=[t.lower() for t in novel_terms],
                was_clarification_request=True,
                is_grounded_conclusion=False,
                selected_ca3_pole_indices=list(self.last_selected_ca3_pole_indices),
                response_passport=passport,
            )
            return {
                "stimulus": user_utterance,
                "ngrams_mode": "OFF (SON VTE Inner-Speech Epistemic Clarification)",
                "active_moe_columns": moe_out.active_expert_names,
                "conversational_domain": self.dialogue_state.active_domain,
                "remembered_user_name": self.dialogue_state.user_name,
                "role_filler_bindings_applied": slot_bindings,
                "arcuate_token_pointer_injections": 0,
                "reafferent_stranger_check_interventions": self.last_reafferent_stranger_interventions,
                "episode_chain_sentences_generated": 1,
                "multi_hop_rollout_used": False,
                "active_inference_clarification_triggered": True,
                "endogenous_cortical_think_triggered": False,
                "ca3_top1_resonance": self.last_ca3_top1_resonance,
                "epistemic_uncertainty": max(epistemic_uncertainty, 0.75),
                "cortical_moe_branch_profile": cortical_branch_profile,
                "ba44_syntax_bridge_connector": "",
                "swr_consolidated_into_nmda_matrix": False,
                "online_dopamine_imprinted_tokens": 0,
                "vte_beams_evaluated": vte_logs,
                "selected_beam_idx": best_idx,
                "stopped_naturally_by_son_eos": True,
                "generated_tokens_count": len(self.encode_text(clarify_resp)),
                "pure_weight_generated_response": clarify_resp,
                "response_passport": passport,
                "stored_sentence_chains_in_class": 0,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        arc_inj = max(0, self.last_arcuate_token_injections // max(1, len(top_onsets)))
        passport = self._build_response_passport(
            stimulus_ids=u_ids,
            proportional_gate_zone="ZONE_1_DIRECT_RECALL",
            son_vte_grounding="grounded_direct" if has_grounding else "ungrounded_suppressed_from_memory",
            think_trace=f"Zone1_DirectRecall: poles={self.last_selected_ca3_pole_indices} sal={self.last_selected_ca3_pole_saliences} closure={closure_score_val:.2f}",
            arcuate_injections=arc_inj,
            ba44_bridge=ba44_bridge_used,
        )
        # Предохранитель A3: записываем ход в эпизодическую память и SWR ТОЛЬКО если ответ прошёл семантическое заземление!
        self._record_dialogue_turn(
            stimulus_vec64=q_raw_vec,
            activated_poles=ca3_clause_poles,
            utterance=user_utterance,
            response_text=response_text,
            role_filler_bindings=slot_bindings,
            is_grounded_conclusion=has_grounding,
            selected_ca3_pole_indices=list(self.last_selected_ca3_pole_indices),
            response_passport=passport,
        )

        # A generated answer is not an observation, even if a verifier accepts it.
        # SWR learning is permitted only through explicit source-backed training.
        swr_consolidated = False

        return {
            "stimulus": user_utterance,
            "ngrams_mode": "ON" if ngrams_enabled else "OFF (Pure 8192-Dendritic NMDA Matrix + BA44 Syntax Bridge)",
            "active_moe_columns": moe_out.active_expert_names,
            "conversational_domain": self.dialogue_state.active_domain,
            "remembered_user_name": self.dialogue_state.user_name,
            "role_filler_bindings_applied": slot_bindings,
            "arcuate_token_pointer_injections": arc_inj,
            "reafferent_stranger_check_interventions": self.last_reafferent_stranger_interventions,
            "episode_chain_sentences_generated": sentences_generated,
            "multi_hop_rollout_used": self.last_multi_hop_used,
            "active_inference_clarification_triggered": False,
            "endogenous_cortical_think_triggered": False,
            "ca3_top1_resonance": self.last_ca3_top1_resonance,
            "epistemic_uncertainty": epistemic_uncertainty,
            "cortical_moe_branch_profile": cortical_branch_profile,
            "ba44_syntax_bridge_connector": ba44_bridge_used,
            "swr_consolidated_into_nmda_matrix": swr_consolidated,
            "online_dopamine_imprinted_tokens": imprinted,
            "vte_beams_evaluated": vte_logs,
            "selected_beam_idx": best_idx,
            "stopped_naturally_by_son_eos": beams_stopped[best_idx],
            "stopped_by_learned_eos": beam_finish_reasons.get(best_idx) == "learned_eos",
            "generation_finish_reason": beam_finish_reasons.get(best_idx, "composite_clause_controls"),
            "generated_tokens_count": len(chosen_ids),
            "pure_weight_generated_response": response_text,
            "response_passport": passport,
            "stored_sentence_chains_in_class": 0,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    # ------------------------------------------------------------------
    # САМОДОСТАТОЧНЫЙ КОНТРАКТ СОХРАНЕНИЯ И ЗАГРУЗКИ ЧЕКПОИНТА (ROUND-TRIP SAFE)
    # ------------------------------------------------------------------
    def export_checkpoint_payload(
        self,
        extra_metadata: Optional[Dict[str, object]] = None,
        use_fp16: bool = False,
    ) -> Tuple[Dict[str, object], Dict[str, object]]:
        """Build a validated checkpoint with one authoritative copy of each tensor."""
        snapshot = self.export_runtime_state()
        saved_core = snapshot["core"]
        saved_dend = saved_core["dendritic_plateau_readout"]["attributes"]
        saved_vsa = saved_core["vsa_encoder"]["attributes"]
        saved_son = saved_core["son_column"]["attributes"]
        saved_pc = saved_core["pc_readout"]["attributes"]
        n_active = len(saved_dend["col_to_tok"])
        # Состояние химии живёт в chemistry.s (ChemState). Раньше читались
        # несуществующие атрибуты самой chemistry (da/ne/ach/sht/...) ⇒ в чекпоинт
        # писался ПУСТОЙ словарь, и нейромодуляция терялась при каждой загрузке:
        # модель всегда работала на значениях ChemState по умолчанию.
        chem_s = self.chemistry.s
        chemistry_state: Dict[str, float] = {
            k: float(getattr(chem_s, k))
            for k in ("da", "ne", "ach", "ht", "ox", "ado", "cort", "lactate",
                      "ecb", "no", "h2s", "zn", "mg")
            if hasattr(chem_s, k)
        }

        payload: Dict[str, object] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "tokenizer_sha256": self.tokenizer_sha256,
            "seed": self.seed,
            "total_neurons": self.moe_cortex.total_neurons + self.son_column.n_son,
            "stored_sentence_chains": 0,
            "total_pairs_trained": self.total_pairs_trained,
            "total_tokens_trained": self.total_tokens_trained,
            "readout_config": {key: saved_dend[key] for key in ("d_vec", "d_dend", "max_pos")},
            "col_to_tok": saved_dend["col_to_tok"],
            "tok_to_col": saved_dend["tok_to_col"],
            "w_ca3_episode_links": saved_dend["w_ca3_episode_links"],
            "shield_state": saved_dend["shield"],
            "bank_router_state": saved_dend["bank_router"],
            "router_residual_bound": saved_dend["router_residual_bound"],
            "vsa_encoder_state": {
                "micro_weights": saved_vsa["hashed_sdr"]["attributes"]["micro_weights"],
                "base_active_bins": saved_vsa["base_active_bins"], "w_proj": saved_vsa["w_proj"]},
            "moe_cortex_state": snapshot["moe_cortex"],
            "son_motor_state": {key: saved_son[key] for key in
                                ("w_basal_son", "w_apical_son", "w_rec_son", "v_son")},
            "chemistry_state": chemistry_state,
            "w_ba44_bridge": saved_core["ba44_syntax_bridge"]["attributes"]["w_ba44_bridge"],
            "son_w_basal": saved_son["w_basal_son"],
            "son_w_apical": saved_son["w_apical_son"],
            "w_stimulus_to_onset": saved_core["w_stimulus_to_onset"].half() if use_fp16 else saved_core["w_stimulus_to_onset"],
            "w_readout": saved_pc["w_readout"].half() if use_fp16 else saved_pc["w_readout"],
            "token_embed": saved_pc["token_embed"].half() if use_fp16 else saved_pc["token_embed"],
            "conversational_state": self.dialogue_state.state_dict(),
            "spiking_language_state": snapshot["spiking_language_loop"],
            "language_memory_records": saved_core["language_memory_records"],
            "core_runtime_state": snapshot,
        }
        for key in ("pos_phase_vecs", "w_basal_branch", "w_apical_branch",
                    "w_episode_trace", "w_episode_trace_len", "w_swr_poles",
                    "w_ca3_attractors", "w_ca3_sdr", "w_ca3_salience", "sdr_column_usage"):
            payload[key] = saved_dend[key]
        for key in ("w_dend_out", "w_core_dend_out", "w_swr_dend_out"):
            active = saved_dend[key][:, :n_active]
            payload[key] = active.half() if use_fp16 else active
        if extra_metadata:
            payload.update(extra_metadata)

        val_info = validate_self_contained_checkpoint(
            payload,
            expected_tokenizer_sha256=self.tokenizer_sha256,
            strict_schema=True,
        )
        return payload, val_info

    def save_self_contained_checkpoint(
        self,
        ckpt_path: Path,
        extra_metadata: Optional[Dict[str, object]] = None,
        use_fp16: bool = False,
    ) -> Dict[str, object]:
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload, val_info = self.export_checkpoint_payload(extra_metadata, use_fp16)
        tmp_ckpt = ckpt_path.with_suffix(".pt.tmp")
        torch.save(payload, str(tmp_ckpt))
        os.replace(tmp_ckpt, ckpt_path)
        self.loaded_checkpoint_path = str(ckpt_path)
        return val_info

    def drop_legacy_shield_trace(self, payload: Dict[str, object]) -> Dict[str, object]:
        """Не материализовать legacy-трассу щита (см. модульную функцию)."""
        return drop_legacy_shield_trace(payload)

    def apply_restore_migrations(self) -> Dict[str, object]:
        """МИГРАЦИИ ПОСЛЕ ВОССТАНОВЛЕНИЯ (единая точка).

        Восстановление ставит только те поля, что были в снимке. Всё новое,
        добавленное в живые объекты после этого чекпоинта, в снимке отсутствует
        — и модель падала на каждом шаге (три разные поломки подряд):
          * `astrocyte` у колонок (трек O, 14.11) → AttributeError;
          * `ach=11.56` в химии кандидата v19 при номинальном ~0.4 → валидатор
            action-контекста: «ach must be in [0.0, 2.0]»;
          * новые поля регуляторов в ридеуте.

        Здесь все такие разрывы закрываются ОДИН раз и ОТЧЁТНО пишутся в
        `restore_migrations`: миграция должна быть видна, а не «тихой».
        """
        report: Dict[str, object] = {}
        chem_report = self.chemistry.normalize_restored_state()
        if chem_report:
            report["chemistry_state"] = chem_report
        # ЧАСЫ: владелец такта — биологический драйвер (химия). В снимке v19
        # драйвера не было, поэтому часы коры и химии в нём не синхронизированы:
        # после восстановления контракт «+1 за такт» падал
        # (`context=1 != cortex=12`). Приводим кору к часам владельца.
        chem_tick = int(getattr(self.chemistry, "tick", 0))
        cortex_tick = int(getattr(self.moe_cortex, "global_clock_tick", chem_tick))
        if cortex_tick != chem_tick:
            self.moe_cortex.global_clock_tick = chem_tick
            report["clock_sync"] = {"cortex_tick_before": cortex_tick,
                                    "cortex_tick_after": chem_tick,
                                    "owner": "chemistry.tick"}
        completed = 0
        for column in getattr(self.moe_cortex, "columns", []):
            if hasattr(column, "_ensure_astrocyte") and not hasattr(column, "astrocyte"):
                column._ensure_astrocyte()
                completed += 1
        if completed:
            report["columns_completed"] = completed
        self.restore_migrations = report
        return report

    def load_self_contained_checkpoint(
        self,
        ckpt_path: Path,
        strict_schema: bool = True,
        preloaded_payload: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Загружает и валидирует самодостаточный чекпоинт (за один вызов `torch.load`!), восстанавливая проекторы
        `vsa_encoder_state`, `DendriticPlateauMatrixReadout`, `BA44AttractorSyntaxBridge`, `moe_cortex_state`,
        `son_motor_state`, `chemistry_state` и `ConversationalDialogueState` для 100% побайтового Round-Trip при ЛЮБОМ `seed`!"""
        ckpt_path = Path(ckpt_path)
        if preloaded_payload is None:
            if not ckpt_path.exists():
                raise CheckpointCompatibilityError(f"Checkpoint file not found: {ckpt_path}")
            # mmap=True: чекпоинт остаётся на диске и подтягивается страницами.
            # Без этого пик загрузки ~9 ГБ (полная копия + восстановленное
            # состояние) на машине в 15.9 ГБ вызывает OOM всей системы.
            payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        else:
            payload = preloaded_payload
        val_info = validate_self_contained_checkpoint(
            payload,
            expected_tokenizer_sha256=self.tokenizer_sha256,
            strict_schema=strict_schema,
        )
        self.loaded_checkpoint_path = str(ckpt_path)
        if isinstance(payload.get("core_runtime_state"), dict):
            # Плоские дубликаты тензоров верхнего уровня (w_dend_out и прочие)
            # в v19-пути не используются — состояние берётся из
            # core_runtime_state. Ключи positional/branch/episode, используемые
            # checkpoint overlay, сохраняются.
            release_redundant_runtime_payload_fields(payload)
            shield_report = self.drop_legacy_shield_trace(payload)
            self.load_runtime_state(payload["core_runtime_state"])
            # Миграции после восстановления (новые поля, границы осей химии):
            # единая точка + видимый отчёт.
            self.apply_restore_migrations()
            if shield_report["dropped"]:
                # ПОСЛЕ восстановления: load_runtime_state удаляет атрибуты,
                # которых нет в снимке.
                self.legacy_shield_trace_dropped = shield_report
            readout = self.dendritic_plateau_readout
            # Наложение слоёв, которых нет в runtime-снимке (гиппокампальная
            # траектория эпизода, расширенное окно позиций): восстановление
            # снимка удаляет атрибуты, которых в нём не было.
            readout.apply_checkpoint_overlay(payload)
            readout.resize_output_capacity(len(readout.col_to_tok))
            self.loaded_checkpoint_path = str(ckpt_path)
            return val_info

        # 1. Восстанавливаем проекторы VSA-энкодера и перестраиваем кэш 64-мерных векторов (полная независимость от seed!)
        vsa_state = payload.get("vsa_encoder_state")
        if isinstance(vsa_state, dict):
            if isinstance(vsa_state.get("micro_weights"), torch.Tensor):
                self.vsa_encoder.hashed_sdr.micro_weights = vsa_state["micro_weights"].float().clone()
            if isinstance(vsa_state.get("base_active_bins"), torch.Tensor):
                self.vsa_encoder.base_active_bins = vsa_state["base_active_bins"].to(torch.uint8).clone()
            if isinstance(vsa_state.get("w_proj"), torch.Tensor):
                self.vsa_encoder.w_proj = vsa_state["w_proj"].float().clone()
            self._vec64_cache.clear()
            for bid in range(2048):
                self._vec64_cache[bid] = self.vsa_encoder.project_token_to_cortex(bid)
            for idx, trig_tid in self.ba44_syntax_bridge.connector_trigger_ids.items():
                self.ba44_syntax_bridge.connector_trigger_vecs[idx] = self.get_vec64(trig_tid)

        # 2. Восстанавливаем матрицу дендритных плато, проекторы pos_phase_vecs, w_basal_branch, w_apical_branch и CA3
        shield_report = self.drop_legacy_shield_trace(payload)
        if shield_report["dropped"]:
            self.legacy_shield_trace_dropped = shield_report
        self.dendritic_plateau_readout.load_state_dict(payload)

        # 3. Восстанавливаем состояние коры, мотора SON и нейрохимии
        moe_state = payload.get("moe_cortex_state")
        if isinstance(moe_state, dict):
            if "state" in moe_state:
                self.moe_cortex.load_state_dict(moe_state)
            if isinstance(moe_state.get("homeostatic_expert_theta"), torch.Tensor):
                self.moe_cortex.homeostatic_expert_theta = moe_state["homeostatic_expert_theta"].float().clone()
            last_spikes = moe_state.get("last_col_spikes")
            if isinstance(last_spikes, dict):
                for k, v in last_spikes.items():
                    if isinstance(v, torch.Tensor):
                        self.moe_cortex._last_col_spikes[int(k)] = v.float().clone()
            som_errs = moe_state.get("col_somatic_errors")
            api_errs = moe_state.get("col_apical_errors")
            if isinstance(som_errs, list) and isinstance(api_errs, list):
                for c_i, col in enumerate(self.moe_cortex.columns):
                    if c_i < len(som_errs):
                        col.dendrite.last_somatic_fep_error = float(som_errs[c_i])
                    if c_i < len(api_errs):
                        col.dendrite.last_apical_fep_error = float(api_errs[c_i])

        son_state = payload.get("son_motor_state")
        if isinstance(son_state, dict):
            if isinstance(son_state.get("w_basal_son"), torch.Tensor):
                self.son_column.w_basal_son = son_state["w_basal_son"].float().clone()
            if isinstance(son_state.get("w_apical_son"), torch.Tensor):
                self.son_column.w_apical_son = son_state["w_apical_son"].float().clone()
            if isinstance(son_state.get("w_rec_son"), torch.Tensor):
                self.son_column.w_rec_son = son_state["w_rec_son"].float().clone()
            if isinstance(son_state.get("v_son"), torch.Tensor):
                self.son_column.v_son = son_state["v_son"].float().clone()
        else:
            if isinstance(payload.get("son_w_basal"), torch.Tensor):
                self.son_column.w_basal_son = payload["son_w_basal"].float().clone()
            if isinstance(payload.get("son_w_apical"), torch.Tensor):
                self.son_column.w_apical_son = payload["son_w_apical"].float().clone()

        chem_state = payload.get("chemistry_state")
        if isinstance(chem_state, dict):
            # Состояние химии — в chemistry.s. Раньше сеттер шёл в сам chemistry,
            # где таких атрибутов нет ⇒ восстановление было молчаливым no-op.
            # Легаси-ключи старого формата переводим в актуальные.
            legacy = {"sht": "ht", "orexin": "ox", "adenosine": "ado", "cortisol": "cort"}
            chem_s = self.chemistry.s
            for k, v in chem_state.items():
                key = legacy.get(str(k), str(k))
                if hasattr(chem_s, key):
                    value = float(v)
                    # Аксиальные модуляторы ограничены физиологическим диапазоном
                    # (иначе, например, ach копился до 11.56 и все холинергические
                    # гейты насыщались). Приводим к диапазону при загрузке.
                    if key in ("da", "ne", "ach", "ht"):
                        value = max(0.0, min(2.0, value))
                    setattr(chem_s, key, value)
                elif hasattr(self.chemistry, key):
                    setattr(self.chemistry, key, float(v))
        # Опорное состояние модуляции: модель «у себя дома». Отклонения от него
        # (стресс, сон, новизна, награда) — это и есть разные состояния организма.
        self._neuromod_reference = self._raw_neuromod_gains()

        if isinstance(payload.get("w_ba44_bridge"), torch.Tensor):
            self.ba44_syntax_bridge.w_ba44_bridge = payload["w_ba44_bridge"].float().clone()
        if isinstance(payload.get("w_stimulus_to_onset"), torch.Tensor):
            self.w_stimulus_to_onset = payload["w_stimulus_to_onset"].float().clone()
        if isinstance(payload.get("w_readout"), torch.Tensor):
            self.pc_readout.w_readout = payload["w_readout"].float().clone()
        if isinstance(payload.get("token_embed"), torch.Tensor):
            self.pc_readout.token_embed = payload["token_embed"].float().clone()
        if isinstance(payload.get("conversational_state"), dict):
            self.dialogue_state.load_state_dict(payload["conversational_state"])
        self.total_pairs_trained = int(payload.get("total_pairs_trained", payload.get("total_elite_pairs_consolidated", 0)))
        self.total_tokens_trained = int(payload.get("total_tokens_trained", payload.get("total_consolidated_tokens", 0)))
        # Legacy checkpoints contain no provenance: do not invent a ledger from weights.
        self.language_memory_records = load_runtime_tree([], payload.get("language_memory_records", []))
        loop_state = payload.get("spiking_language_state")
        if isinstance(loop_state, dict):
            self.spiking_language_loop.load_state_dict(loop_state)
        runtime = payload.get("core_runtime_state")
        if isinstance(runtime, dict):
            # Копия обязательна: mmap-отображение на Windows доступно только
            # для чтения, а модель пишет в веса на месте.
            self.load_runtime_state(runtime)
            # Те же миграции, что и в v19-пути (новые поля, границы химии).
            self.apply_restore_migrations()
        self.loaded_checkpoint_path = str(ckpt_path)
        return val_info


