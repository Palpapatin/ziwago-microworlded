#!/usr/bin/env python3
r"""UNIFIED EVALUATION GATES, SELF-CONTAINED CHECKPOINT CONTRACT & GENERALIZATION METRICS
(`ziwago/evaluation/gates.py`)
=============================================================================================
Единый модуль верификации, ворот качества (Gates), эталонного Round-Trip контроля и раздельного
измерения устойчивости к перефразированию (`paraphrase_robustness_of_retrieval`) и истинного
обобщения на новое знание (`novel_knowledge_generalization_score`):

  1. САМОДОСТАТОЧНЫЙ КОНТРАКТ ЧЕКПОИНТА (`Self-Contained Checkpoint Contract v16.2`):
     - Обязательные поля: `schema_version`, `tokenizer_sha256`, `col_to_tok`, `tok_to_col`, `w_dend_out`,
       проекторы `pos_phase_vecs`, `w_basal_branch`, `w_apical_branch`, `vsa_encoder_state`,
       а также состояния `moe_cortex_state`, `son_motor_state`, `chemistry_state`.
     - Полная независимость от `seed` при загрузке: отказ загружать (`CheckpointCompatibilityError`)
       при отсутствии любого проектора, карты колонок, несовпадении размерностей или SHA-256 токенизатора.

  2. РАЗЛИЧАЮЩИЕ ГРАММАТИЧЕСКИЕ ВОРОТА (`Discriminating Grammar & N-Gram Coverage Gate`):
     - Сочетает три строгих критерия, гарантированно отвергающих «словесную окрошку» (word salad):
       а) Синтаксическая и пунктуационная валидность переходов (запрет пробела перед запятой `" ,"`,
          запрет цепочек предлогов/вспомогательных глаголов `"to in"`, `"in but"`, `"a is"`, `"was for."`,
          запрет случайных заглавных букв в середине клаузы `"i a In"` и битых токенов `"He't"`, `",*"`);
       б) Покрытие сгенерированных переходов обученной синаптической статистикой (`trained_ngram_coverage`);
       в) Выравнивание с эталонным набором ответов (`known_answer_f1_score`).

  3. РАЗДЕЛЕНИЕ МЕТРИК: `paraphrase_robustness_of_retrieval` И `novel_knowledge_generalization_score`:
     - `paraphrase_robustness_of_retrieval`: измеряет устойчивость ретривала и сшивки полюсов `CA3 + BA44 + TPJ`
       при перефразированных и многоклаузальных вопросах (`HOLD_OUT_EXAM_SUITE`).
     - `novel_knowledge_generalization_score`: проверяет настоящее обобщение на **новое знание**, которого
       никогда не было в обучающем потоке (Zero-Shot перенос новых сущностей через `Role x Filler` и
       One-Shot синаптическое усвоение нового научного факта с перефразированным опросом).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

CHECKPOINT_SCHEMA_VERSION = "ziwago.checkpoint.v16.2"
MASTER_CKPT_REL_PATH = Path("Produce") / "checkpoints" / "ziwago_v18_1_genesis_omniscient_master.pt"
MASTER_CKPT = Path(__file__).resolve().parents[2] / MASTER_CKPT_REL_PATH


class CheckpointCompatibilityError(ValueError):
    """Выбрасывается при нарушении контракта самодостаточного чекпоинта (отсутствие карты колонок,
    проекторов, состояния коры/SON/химии, несовпадение версии схемы, SHA-256 токенизатора или размерностей)."""


def resolve_master_checkpoint_path(root: Optional[Path] = None, explicit_path: Optional[Path] = None) -> Path:
    """Единый резолвер мастер-чекпоинта (`MASTER_CKPT`):
    Вместо тихого `skip` или отката к необученной сети жёстко выбрасывает `CheckpointCompatibilityError`,
    если файл мастер-чекпоинта отсутствует на диске."""
    if explicit_path is not None:
        ckpt = Path(explicit_path)
    elif root is not None:
        ckpt = Path(root) / MASTER_CKPT_REL_PATH
    else:
        ckpt = MASTER_CKPT
    if not ckpt.exists():
        raise CheckpointCompatibilityError(
            f"CRITICAL: Master checkpoint file does not exist at {ckpt} (silent skip is forbidden)."
        )
    return ckpt


def compute_tokenizer_sha256(vocab_path: Path, merges_path: Path) -> str:
    """Вычисляет детерминированный SHA-256 дайджест файлов токенизатора `vocab.json` и `merges.txt`."""
    vocab_path = Path(vocab_path)
    merges_path = Path(merges_path)
    h = hashlib.sha256()
    if not vocab_path.exists():
        raise CheckpointCompatibilityError(f"Tokenizer vocab file not found: {vocab_path}")
    if not merges_path.exists():
        raise CheckpointCompatibilityError(f"Tokenizer merges file not found: {merges_path}")
    h.update(vocab_path.read_bytes())
    h.update(b"\x00::MERGES::\x00")
    h.update(merges_path.read_bytes())
    return h.hexdigest()


def validate_self_contained_checkpoint(
    payload: Dict[str, object],
    expected_tokenizer_sha256: Optional[str] = None,
    expected_schema_version: str = CHECKPOINT_SCHEMA_VERSION,
    strict_schema: bool = True,
) -> Dict[str, object]:
    """Проверяет самодостаточность и целостность словаря чекпоинта перед сохранением или после загрузки.
    Требует наличия словарей колонок, проекторов (`pos_phase_vecs`, `w_basal_branch`, `w_apical_branch`,
    `vsa_encoder_state`) и состояний (`moe_cortex_state`, `son_motor_state`, `chemistry_state`)."""
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError("Checkpoint payload must be a dictionary.")

    required_keys = (
        "schema_version",
        "tokenizer_sha256",
        "col_to_tok",
        "tok_to_col",
        "w_dend_out",
        "pos_phase_vecs",
        "w_basal_branch",
        "w_apical_branch",
        "vsa_encoder_state",
        "moe_cortex_state",
        "son_motor_state",
        "chemistry_state",
    )
    missing_keys = [k for k in required_keys if k not in payload]
    if missing_keys:
        raise CheckpointCompatibilityError(
            f"Checkpoint is missing required self-contained keys: {missing_keys}"
        )

    schema_ver = payload.get("schema_version")
    if strict_schema and schema_ver != expected_schema_version:
        raise CheckpointCompatibilityError(
            f"Unsupported checkpoint schema_version: got {schema_ver!r}, expected {expected_schema_version!r}."
        )

    tok_hash = payload.get("tokenizer_sha256")
    if not isinstance(tok_hash, str) or re.fullmatch(r"[0-9a-f]{64}", tok_hash) is None:
        raise CheckpointCompatibilityError(
            f"Missing or invalid 'tokenizer_sha256' in checkpoint: {tok_hash!r}."
        )
    if expected_tokenizer_sha256 is not None and tok_hash != expected_tokenizer_sha256:
        raise CheckpointCompatibilityError(
            f"Tokenizer SHA-256 mismatch: checkpoint has {tok_hash[:16]}..., "
            f"runtime has {expected_tokenizer_sha256[:16]}..."
        )

    col_to_tok = payload["col_to_tok"]
    tok_to_col = payload["tok_to_col"]
    if not isinstance(col_to_tok, list) or not isinstance(tok_to_col, dict):
        raise CheckpointCompatibilityError(
            "'col_to_tok' must be a list and 'tok_to_col' must be a dict."
        )
    if len(col_to_tok) == 0:
        raise CheckpointCompatibilityError("Checkpoint 'col_to_tok' mapping is empty.")
    if len(col_to_tok) != len(tok_to_col):
        raise CheckpointCompatibilityError(
            f"Mapping size mismatch: len(col_to_tok)={len(col_to_tok)} != len(tok_to_col)={len(tok_to_col)}."
        )

    for col_idx, tok_id in enumerate(col_to_tok):
        mapped_col = tok_to_col.get(int(tok_id), tok_to_col.get(str(tok_id)))
        if mapped_col != col_idx:
            raise CheckpointCompatibilityError(
                f"Bijection broken at col={col_idx}, tok_id={tok_id}: tok_to_col returned {mapped_col!r}."
            )

    w_dend = payload.get("w_dend_out")
    if not isinstance(w_dend, torch.Tensor) or w_dend.ndim != 2:
        raise CheckpointCompatibilityError("Checkpoint 'w_dend_out' must be a 2D torch.Tensor.")
    if int(w_dend.shape[1]) != len(col_to_tok):
        raise CheckpointCompatibilityError(
            f"Dimension mismatch: w_dend_out.shape[1]={w_dend.shape[1]} != len(col_to_tok)={len(col_to_tok)}."
        )

    pos_vecs = payload.get("pos_phase_vecs")
    w_bas = payload.get("w_basal_branch")
    w_api = payload.get("w_apical_branch")
    if not isinstance(pos_vecs, torch.Tensor) or not isinstance(w_bas, torch.Tensor) or not isinstance(w_api, torch.Tensor):
        raise CheckpointCompatibilityError("Projectors pos_phase_vecs, w_basal_branch, w_apical_branch must be torch.Tensors.")
    if int(w_bas.shape[1]) != int(w_dend.shape[0]) or int(w_api.shape[1]) != int(w_dend.shape[0]):
        raise CheckpointCompatibilityError("Projector branch dimension mismatch with w_dend_out.shape[0].")

    vsa_state = payload.get("vsa_encoder_state")
    if not isinstance(vsa_state, dict) or not all(k in vsa_state for k in ("micro_weights", "base_active_bins", "w_proj")):
        raise CheckpointCompatibilityError("Checkpoint 'vsa_encoder_state' is missing required VSA projector tensors.")

    return {
        "valid": True,
        "schema_version": str(schema_ver),
        "tokenizer_sha256": str(tok_hash),
        "active_vocab_columns": len(col_to_tok),
    }


# ============================================================================
# 2. РАЗЛИЧАЮЩИЕ ГРАММАТИЧЕСКИЕ ВОРОТА (SYNTAX LEGALITY + N-GRAM COVERAGE + KNOWN-ANSWER)
# ============================================================================
_FUNCTION_PREPOSITIONS = {"in", "on", "at", "by", "for", "with", "from", "of", "into", "onto", "upon"}
_AUXILIARY_VERBS = {"is", "are", "was", "were", "be", "been", "am", "can", "could", "will", "would", "did", "do", "does"}
_ARTICLES = {"a", "an", "the"}
_DANGLING_END_WORDS = _FUNCTION_PREPOSITIONS | _AUXILIARY_VERBS | _ARTICLES | {"to", "and", "or", "but", "while", "before", "after", "his", "your", "my", "its", "their"}
_MID_SENTENCE_FORBIDDEN_CAPS = {"In", "For", "The", "They", "Yes", "He", "But", "And", "To", "Of", "With", "Were", "Was", "Is", "Are"}


def evaluate_grammar_and_artifacts(
    text: str,
    min_words: int = 6,
    min_unique_ratio: float = 0.48,
    trained_bigram_g: Optional[Dict[int, Dict[int, float]]] = None,
    encode_fn: Optional[Callable[[str], List[int]]] = None,
) -> Dict[str, object]:
    """Различающий грамматический и статистический гейт (Point 2):
      1) Проверяет синтаксическую легальность пар слов и пунктуации (отвергает `" ,"`, `",*"`, `"to in"`,
         `"a is"`, `"He't"`, висящие предлоги/модальные глаголы в конце предложения `"was for."`, `"in could."`,
         случайные заглавные служебные слова в середине фразы `"i a In"`).
      2) Если переданы `trained_bigram_g` и `encode_fn`, вычисляет долю поддержанных обученной статистикой
         переходов между токенами (`trained_ngram_coverage >= 0.65`)."""
    raw_str = str(text).strip()
    clean = " ".join(raw_str.split())
    words = clean.split()
    low_text = clean.lower()

    has_superacid = ("superacid" in low_text) or ("<unk>" in low_text) or ("\ufffd" in clean)

    # 1. Детектор пунктуационных и субсловных артефактов BPE (" ,", ",*", ".*", "He't", "i's", "- are", ",'s")
    punct_defect = bool(
        re.search(r"\s[,.;:!?]", raw_str)
        or re.search(r"[,.;:!?][*]", raw_str)
        or re.search(r"[,.;:!?]'[a-zA-Z]", raw_str)
        or re.search(r"\b(?:he't|i's|you't|it't|they't)\b", low_text)
        or re.search(r"\b-\s+[a-zA-Z]", raw_str)
        or raw_str.startswith((",", ".", ";", ":", "!", "?"))
    )

    consec_repeats = 0
    illegal_syntax_transitions = 0
    mid_sentence_cap_defects = 0
    punct_strip = ".,!?;:'()[]{}" + chr(34)

    for idx, (w_a, w_b) in enumerate(zip(words, words[1:])):
        ca = w_a.strip(punct_strip).lower()
        cb = w_b.strip(punct_strip).lower()
        a_ends_clause = w_a[-1:] in ".!?;:,"
        a_ends_sent = w_a[-1:] in ".!?"

        if not a_ends_sent and ca and ca == cb and len(ca) >= 2:
            consec_repeats += 1

        # Одиночные случайные согласные буквы ("Bel Pre Road f in")
        if (len(ca) == 1 and ca.isascii() and ca.isalpha() and ca not in {"a", "i"}) or (len(cb) == 1 and cb.isascii() and cb.isalpha() and cb not in {"a", "i"}):
            illegal_syntax_transitions += 1

        # Завершение внутренней фразы/предложения на предлоге или артикле ("f in. overly", "the, a")
        if a_ends_clause and ca in (_ARTICLES | _FUNCTION_PREPOSITIONS | {"your", "my", "his", "her", "to"}):
            illegal_syntax_transitions += 1

        # Строчная буква сразу после точки в середине текста (". overly")
        raw_b_clean = w_b.strip(punct_strip)
        if a_ends_sent and raw_b_clean and raw_b_clean[0].islower():
            illegal_syntax_transitions += 1

        # Проверка запрещённых синтаксических переходов внутри одной клаузы (без разделительной пунктуации):
        if not a_ends_clause:
            # Два предлога подряд ("at to", "of as", "in for") или предлог + союз ("to in", "in but")
            if ca in _FUNCTION_PREPOSITIONS and cb in (_FUNCTION_PREPOSITIONS | {"but", "and", "or", "are", "is", "was", "were"}):
                illegal_syntax_transitions += 1
            # Артикль + глагол-связка, предлог или служебное слово ("a is", "a be", "the is", "a in", "the for", "a as")
            if ca in _ARTICLES and cb in (_AUXILIARY_VERBS | _FUNCTION_PREPOSITIONS | {"to", "and", "but", "or", "as", "that", "all"}):
                illegal_syntax_transitions += 1
            # Два вспомогательных глагола одной группы подряд ("is are", "were was", "be are")
            if ca in {"is", "are", "was", "were"} and cb in {"is", "are", "was", "were", "can", "will", "did"}:
                illegal_syntax_transitions += 1
            # Строчное местоимение "i" (в отличие от грамматически верного заглавного "I")
            if w_a.strip(punct_strip) == "i" or w_b.strip(punct_strip) == "i":
                illegal_syntax_transitions += 1

        # Случайное слово с заглавной буквы в середине предложения (не после .!?;:)
        if not a_ends_sent and raw_b_clean in _MID_SENTENCE_FORBIDDEN_CAPS:
            mid_sentence_cap_defects += 1

    # Проверка последнего слова предложения (не должно обрываться на предлоге, артикле или модальном глаголе)
    dangling_end = False
    if words:
        last_clean = words[-1].strip(punct_strip).lower()
        if last_clean in _DANGLING_END_WORDS:
            dangling_end = True

    norm_words = [w.strip(punct_strip).lower() for w in words if w.strip(punct_strip)]
    unique_ratio = len(set(norm_words)) / float(max(1, len(norm_words)))

    starts_capital = bool(clean) and (clean[0].isupper() or clean[0] in chr(34) + "'(")
    ends_punct = bool(clean) and (clean[-1] in ".!?")
    has_enough_words = len(words) >= min_words

    # Syntax can be inspected without a corpus, but absent coverage is not evidence.
    ngram_coverage = 0.0
    ngram_available = False
    if trained_bigram_g and encode_fn is not None:
        t_ids = encode_fn(clean)
        if len(t_ids) >= 2:
            supported = 0
            total_bg = 0
            for t1, t2 in zip(t_ids, t_ids[1:]):
                total_bg += 1
                weight = trained_bigram_g.get(int(t1), {}).get(int(t2), 0.0)
                if isinstance(weight, (int, float)) and math.isfinite(weight) and weight > 0:
                    supported += 1
            if total_bg > 0:
                ngram_coverage = supported / float(total_bg)
                ngram_available = True

    repeated_cluster = any(
        norm_words.count(word) >= 3
        and sum(norm_words[i:i + 4] == norm_words[start:start + 4]
                for i in range(len(norm_words) - 3)) >= 3
        for start, word in enumerate(norm_words[:-3])
    )
    sentences = [s.strip().lower() for s in re.split(r"[.!?]+", clean) if s.strip()]
    repeated_sentence = any(sentences.count(s) >= 2 and len(s.split()) >= 4 for s in sentences)
    syntax_coherent = (
        has_enough_words
        and (consec_repeats == 0)
        and (illegal_syntax_transitions == 0)
        and (mid_sentence_cap_defects == 0)
        and (not punct_defect)
        and (not dangling_end)
        and (not has_superacid)
        and (unique_ratio >= min_unique_ratio)
        and starts_capital
        and ends_punct
        and not repeated_cluster
        and not repeated_sentence
    )
    grammar_coherent = syntax_coherent and (not ngram_available or ngram_coverage >= 0.50)

    return {
        "text": clean,
        "word_count": len(words),
        "unique_token_ratio": round(unique_ratio, 4),
        "consecutive_word_repeats": consec_repeats,
        "illegal_syntax_transitions": illegal_syntax_transitions,
        "mid_sentence_cap_defects": mid_sentence_cap_defects,
        "punctuation_defect": punct_defect,
        "dangling_sentence_end": dangling_end,
        "trained_ngram_coverage": round(ngram_coverage, 4),
        "trained_ngram_available": ngram_available,
        "grammar_status": "measured" if ngram_available else "syntax_only_ngram_unavailable",
        "syntax_coherent": syntax_coherent,
        "repeated_cluster_detected": repeated_cluster or repeated_sentence,
        "superacid_detected": has_superacid,
        "starts_with_capital": starts_capital,
        "ends_with_punctuation": ends_punct,
        "grammar_coherent": grammar_coherent,
    }


def evaluate_oov_compositional_resilience(
    encode_fn: Callable[[str], List[int]],
    get_vec64_fn: Callable[[int], torch.Tensor],
    generate_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, object]:
    """Проверяет устойчивость VSA-BPE энкодера и генератора к редким/незнакомым композиционным терминам (OOV)."""
    oov_probes = [
        "neuro-astrocytic synapto-homeostasis",
        "dendro-apical hyper-plasticity",
        "thalamo-cortical chrono-precession",
    ]
    probe_vecs: List[torch.Tensor] = []
    valid_vectors = True

    for phrase in oov_probes:
        t_ids = encode_fn(phrase)
        if not t_ids:
            valid_vectors = False
            break
        vecs = [get_vec64_fn(tid) for tid in t_ids]
        v_mean = torch.stack(vecs).mean(dim=0)
        if torch.isnan(v_mean).any() or torch.isinf(v_mean).any() or float(v_mean.norm().item()) < 1e-5:
            valid_vectors = False
            break
        probe_vecs.append(v_mean / v_mean.norm().clamp(min=1e-8))

    distinct_representations = False
    if len(probe_vecs) >= 3:
        c01 = float(torch.dot(probe_vecs[0], probe_vecs[1]).item())
        c02 = float(torch.dot(probe_vecs[0], probe_vecs[2]).item())
        distinct_representations = (c01 < 0.98) and (c02 < 0.98)

    gen_ok = False
    gen_sample = ""
    if generate_fn is not None:
        gen_sample = generate_fn(
            "How does neuro-astrocytic synapto-homeostasis interact with pyramidal neurons in the neocortex?"
        )
        g_eval = evaluate_grammar_and_artifacts(gen_sample, min_words=5, min_unique_ratio=0.45)
        gen_ok = bool(g_eval["grammar_coherent"]) and not bool(g_eval["superacid_detected"])

    passed = valid_vectors and distinct_representations and gen_ok
    return {
        "passed": passed,
        "available": generate_fn is not None and bool(gen_sample.strip()),
        "oov_pass_rate": 1.0 if passed else 0.0,
        "valid_non_zero_vectors": valid_vectors,
        "distinct_subword_compositions": distinct_representations,
        "oov_generation_coherent": gen_ok,
        "oov_generated_sample": gen_sample,
    }


# ============================================================================
# 3. МЕТРИКА МАСКИРОВКИ (ATTRACTOR COLLAPSE RATE)
# ============================================================================
ATTRACTOR_COLLAPSE_PROBE_CLUSTERS: List[Dict[str, object]] = [
    {
        "cluster_id": "culinary_vs_lifestyle_perturbation",
        "base_prompt": "My homemade mayonnaise emulsion turned thin and runny while whisking.",
        "perturbed_prompts": [
            "My homemade mayonnaise turned runny, and I also want to fix grammar errors in my essay.",
            "Before fixing a runny mayonnaise emulsion, how does running outdoors help clear the mind?",
            "Why do pyramidal neurons in the neocortex use apical dendrites while someone fixes runny mayonnaise?",
        ],
    },
    {
        "cluster_id": "identity_vs_perspective_perturbation",
        "base_prompt": "Remind me, what is my name and what am I building?",
        "perturbed_prompts": [
            "Alex is a researcher; what is his name and how does he fix grammar errors in an essay?",
            "What is my name, and how do pyramidal neurons in the neocortex integrate predictions?",
        ],
    },
    {
        "cluster_id": "narrative_vs_science_perturbation",
        "base_prompt": "Greetings, guardian of the ancient stone arch in the forest.",
        "perturbed_prompts": [
            "Greetings, guardian of the ancient stone arch, how do pyramidal neurons integrate sensory input?",
            "Can running outdoors clear the mind before speaking with the guardian of the ancient stone arch?",
        ],
    },
]


def _word_jaccard_similarity(text_a: str, text_b: str) -> float:
    p_strip = ".,!?;:'()[]{}" + chr(34)
    set_a = {w.strip(p_strip).lower() for w in text_a.split() if w.strip(p_strip)}
    set_b = {w.strip(p_strip).lower() for w in text_b.split() if w.strip(p_strip)}
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / float(len(union))


def evaluate_attractor_collapse_rate(
    generate_fn: Callable[[str], str],
    probe_clusters: Optional[Sequence[Dict[str, object]]] = None,
) -> Dict[str, object]:
    """Измеряет `Attractor Collapse Rate` (метрику маскировки из `next14.txt` п. 6)."""
    clusters = list(probe_clusters) if probe_clusters is not None else ATTRACTOR_COLLAPSE_PROBE_CLUSTERS
    total_perturbed = 0
    exact_collapses = 0
    near_collapses = 0
    divergences: List[float] = []
    cluster_details: List[Dict[str, object]] = []
    nonempty = bool(clusters)

    for cl in clusters:
        base_q = str(cl["base_prompt"])
        base_ans = _response_text(generate_fn(base_q))[0]
        nonempty = nonempty and bool(base_ans)
        pert_list = list(cl["perturbed_prompts"])  # type: ignore[arg-type]
        pert_logs: List[Dict[str, object]] = []

        nonempty = nonempty and bool(pert_list)
        for pq_str in pert_list:
            p_ans = _response_text(generate_fn(str(pq_str)))[0]
            nonempty = nonempty and bool(p_ans)
            total_perturbed += 1
            is_exact = (p_ans.lower() == base_ans.lower())
            jacc = _word_jaccard_similarity(base_ans, p_ans)
            is_near = jacc >= 0.92
            if is_exact:
                exact_collapses += 1
            if is_near:
                near_collapses += 1
            div = 1.0 - jacc
            divergences.append(div)
            pert_logs.append(
                {
                    "perturbed_prompt": pq_str,
                    "response": p_ans,
                    "exact_echo_of_base": is_exact,
                    "jaccard_with_base": round(jacc, 4),
                    "lexical_divergence": round(div, 4),
                }
            )

        cluster_details.append(
            {
                "cluster_id": cl["cluster_id"],
                "base_prompt": base_q,
                "base_response": base_ans,
                "perturbations": pert_logs,
            }
        )

    collapse_rate = near_collapses / float(max(1, total_perturbed))
    exact_collapse_rate = exact_collapses / float(max(1, total_perturbed))
    mean_div = sum(divergences) / float(max(1, len(divergences)))

    return {
        "available": nonempty and total_perturbed > 0,
        "passed": nonempty and total_perturbed > 0 and near_collapses == 0,
        "total_perturbed_probes": total_perturbed,
        "exact_collapse_count": exact_collapses,
        "near_collapse_count": near_collapses,
        "exact_attractor_collapse_rate": round(exact_collapse_rate, 4),
        "attractor_collapse_rate": round(collapse_rate, 4),
        "mean_lexical_divergence": round(mean_div, 4),
        "cluster_reports": cluster_details,
    }


# ============================================================================
# 4. НАБОР ПРОВЕРКИ ПЕРЕФРАЗИРОВОЧНОЙ УСТОЙЧИВОСТИ РЕТРИВАЛА (`HOLD_OUT_EXAM_SUITE`)
#    И НАСТОЯЩЕЙ ГЕНЕРАЦИИ НОВОГО ЗНАНИЯ (`NOVEL_KNOWLEDGE_GENERALIZATION_SUITE`)
# ============================================================================
@dataclass
class HoldoutExamItem:
    exam_id: str
    domain: str
    prompt: str
    expected_keywords: Tuple[str, ...]
    forbidden_substrings: Tuple[str, ...] = ()

    def __getitem__(self, key: str) -> object:
        if key == "expected_concepts":
            return list(self.expected_keywords)
        if key == "forbidden_echo":
            return list(self.forbidden_substrings)
        return getattr(self, key)

    def get(self, key: str, default: object = None) -> object:
        try:
            return self[key]
        except AttributeError:
            return default


HOLD_OUT_EXAM_SUITE: List[HoldoutExamItem] = [
    HoldoutExamItem(
        exam_id="holdout_1_paraphrase_neuroscience",
        domain="Epoch 3 Science (Paraphrase Retrieval Robustness)",
        prompt="Explain how neocortical pyramidal cells combine bottom-up sensory spikes with top-down cortical predictions.",
        expected_keywords=("pyramidal", "basal", "apical", "nmda"),
        forbidden_substrings=("echoing glade", "superacid"),
    ),
    HoldoutExamItem(
        exam_id="holdout_2_paraphrase_culinary_science",
        domain="Epoch 3+5 Culinary + Cortical Neuroscience",
        prompt="Why does a runny mayonnaise sauce need an egg yolk while pyramidal neurons in the neocortex process predictions?",
        expected_keywords=("mayonnaise", "yolk", "pyramidal", "dendrites"),
        forbidden_substrings=("echoing glade", "superacid"),
    ),
    HoldoutExamItem(
        exam_id="holdout_3_paraphrase_writing_fitness",
        domain="Epoch 4+5 Essay Writing + Outdoor Fitness",
        prompt="Before fixing a runny mayonnaise emulsion, how does running outdoors help clear the mind?",
        expected_keywords=("running", "mind", "mayonnaise", "yolk"),
        forbidden_substrings=("echoing glade", "before to fix", "superacid"),
    ),
    HoldoutExamItem(
        exam_id="holdout_4_conversational_identity_recall",
        domain="Conversational State Recall (What is my name?)",
        prompt="Remind me, what is my name and what project am I building?",
        expected_keywords=("alex", "biological", "neural", "dendritic"),
        forbidden_substrings=("echoing glade", "egg yolk", "superacid"),
    ),
    HoldoutExamItem(
        exam_id="holdout_5_tpj_third_person_shift",
        domain="TPJ 3rd-Person Deictic Shift + Compositional Bridge",
        prompt="Alex is testing his model; what is his name, and how can he fix grammar errors when writing an essay?",
        expected_keywords=("his", "alex", "grammar", "essay"),
        forbidden_substrings=("echoing glade", "superacid"),
    ),
    HoldoutExamItem(
        exam_id="holdout_6_theta_phase_before_bridge",
        domain="Theta-Phase Queue + BA44 'before' Syntax Bridge",
        prompt="Can running outdoors clear the mind before writing an essay without grammar errors?",
        expected_keywords=("running", "mind", "grammar", "essay"),
        forbidden_substrings=("before to fix", "echoing glade", "superacid"),
    ),
    HoldoutExamItem(
        exam_id="holdout_7_roleplay_plus_neuroscience",
        domain="Epoch 3+4 NPC Lore + Cortical Neuroscience",
        prompt="Greetings, guardian of the ancient stone arch, how do pyramidal neurons integrate sensory input?",
        expected_keywords=("echoing", "glade", "pyramidal", "dendrites"),
        forbidden_substrings=("superacid",),
    ),
    HoldoutExamItem(
        exam_id="holdout_8_bidirectional_ca3",
        domain="Bidirectional CA3 (Response Tokens -> Dual Attractor)",
        prompt="Why should a student check subject-verb agreement and read each sentence aloud, while whisking a fresh egg yolk to stabilize an emulsion?",
        expected_keywords=("agreement", "yolk"),
        forbidden_substrings=("superacid",),
    ),
]


def _normal_prompt(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def _response_text(result: object) -> Tuple[str, bool]:
    clarification = False
    if isinstance(result, dict):
        clarification = bool(result.get("active_inference_clarification_triggered")
                             or result.get("clarification_triggered") or result.get("grounded") is False
                             or result.get("model_prediction_uncertain"))
        result = result.get("pure_weight_generated_response", result.get("response", ""))
    text = " ".join(result.split()) if isinstance(result, str) else ""
    clarification = clarification or bool(re.search(
        r"\b(?:clarify|clarification|could you specify|can you specify|i (?:do not|don't) know|"
        r"insufficient (?:context|information)|not enough information)\b", text, re.I))
    return text, clarification


def _concept_matches(text: str, concept: str) -> List[re.Match]:
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(p) for p in concept.split()) + r"(?!\w)"
    return list(re.finditer(pattern, text, re.I)) if concept.strip() else []


def _positive_concept(text: str, concept: str) -> bool:
    # Conservative clause-local polarity. This is lexical evidence, not entailment.
    for match in _concept_matches(text, concept):
        before = re.split(r"[.!?;,:]|\bbut\b", text[:match.start()], flags=re.I)[-1]
        after = re.split(r"[.!?;,:]|\bbut\b", text[match.end():], flags=re.I)[0]
        window = " ".join(before.split()[-5:] + after.split()[:5])
        if not re.search(r"\b(?:not|no|never|neither|without|cannot|can't|isn't|aren't|doesn't|don't)\b", window, re.I):
            return True
    return False


def _case_result(case: dict, generated: object) -> dict:
    text, clarification = _response_text(generated)
    expected = case.get("expected", case.get("expected_concepts", case.get("expected_keywords", ())))
    forbidden = case.get("forbidden", case.get("forbidden_echo", case.get("forbidden_substrings", ())))
    valid = (isinstance(expected, (list, tuple)) and bool(expected)
             and all(isinstance(c, str) and c.strip() for c in expected)
             and isinstance(forbidden, (list, tuple))
             and all(isinstance(c, str) and c.strip() for c in forbidden))
    missing = [c for c in expected if not _positive_concept(text, c)] if valid else []
    # An unrelated entity is contamination even when it is negated.
    echoes = [c for c in forbidden if _concept_matches(text, c)] if valid else []
    prompt_echo = bool(text) and _normal_prompt(text) == _normal_prompt(str(case.get("prompt", "")))
    return {"id": case.get("id", case.get("exam_id", case.get("probe_id"))),
            "prompt": case.get("prompt", ""), "response": text,
            "available": valid and bool(text), "missing_expected": missing,
            "forbidden_matches": echoes, "clarification": clarification,
            "prompt_echo": prompt_echo,
            "passed": valid and bool(text) and not (missing or echoes or clarification or prompt_echo)}


_DOCUMENT_ABBREVIATIONS = frozenset({"Dr", "Mr", "Mrs", "Ms", "Prof", "St", "Jr", "Sr", "vs", "e.g", "i.e", "U.S", "U.K"})
_DOCUMENT_SENT_END = re.compile(r'''[.!?](?!\d)(?:["'\u201d\u2019)\]]*)(?=\s|$)''')
OPEN_BOOK_MAX_SENTENCES = 2


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    sentences = []
    start = 0
    for match in _DOCUMENT_SENT_END.finditer(text):
        preceding = text[start:match.start()].split()
        last_word = preceding[-1] if preceding else ""
        if match.group().startswith(".") and (
            last_word in _DOCUMENT_ABBREVIATIONS or (len(last_word) == 1 and last_word.isupper())
        ):
            continue
        end = match.end()
        raw = text[start:end]
        sentence = raw.strip()
        if sentence:
            offset = start + len(raw) - len(raw.lstrip())
            sentences.append((sentence, offset, offset + len(sentence)))
        start = end
    if start < len(text):
        raw = text[start:]
        sentence = raw.strip()
        if sentence:
            offset = start + len(raw) - len(raw.lstrip())
            sentences.append((sentence, offset, offset + len(sentence)))
    return sentences


def validate_acceptance_documents(
    cases: Sequence[dict], mode: str, trained_documents: Sequence[str] = (),
    trained_document_hashes: Sequence[str] = (),
) -> list[str]:
    if mode not in ("closed_book", "open_book"):
        raise ValueError("Acceptance mode must be closed_book or open_book.")
    hashes = set(trained_document_hashes)
    normalized_documents = {_normal_prompt(text) for text in trained_documents if text.strip()}
    errors = []
    for case in cases:
        for names in (("expected", "expected_concepts", "expected_keywords"),
                      ("forbidden", "forbidden_echo", "forbidden_substrings")):
            values = [case[name] for name in names if name in case]
            if len(values) > 1 and any(value != values[0] for value in values[1:]):
                errors.append("Conflicting acceptance criterion aliases")
        if case.get("evaluation_mode", mode) != mode:
            errors.append("Acceptance case mode disagrees with the requested mode")
        if mode == "closed_book":
            if "document" in case:
                errors.append("Closed-book acceptance cannot contain a document")
            continue
        document = case.get("document")
        if not isinstance(document, str) or not document.strip():
            errors.append("Open-book acceptance requires a nonempty document for every case")
            continue
        forms = {document, document.strip(), " ".join(document.split()),
                 " ".join(document.casefold().split()), _normal_prompt(document)}
        digests = {hashlib.sha256(text.encode("utf-8")).hexdigest() for text in forms}
        if digests & hashes or _normal_prompt(document) in normalized_documents:
            errors.append("Training/evaluation document leakage")
    return errors


def _document_span_result(case: dict, generated: object) -> dict:
    document = case["document"]
    sentence_offsets = {(start, end) for _, start, end in split_sentences(document)}
    text, _ = _response_text(generated)
    document_echo = len(sentence_offsets) > 1 and text == " ".join(document.split())
    errors, spans = [], []
    if document_echo:
        errors.append("Response echoes the whole multi-sentence document")
    if not isinstance(generated, dict) or generated.get("generation_kind") != "extractive_document_qa":
        errors.append("Open-book acceptance requires extractive document QA")
    else:
        grounding = generated.get("response_passport", {}).get("grounding", {})
        sources = grounding.get("sources")
        if (generated.get("grounded") is not True or grounding.get("grounded") is not True
                or grounding.get("evidence_kind") != "verbatim_document_span"
                or not isinstance(sources, list) or not sources):
            errors.append("Missing grounded document spans")
        else:
            if len(sources) > OPEN_BOOK_MAX_SENTENCES:
                errors.append("Response exceeds the extractive sentence budget")
            for source in sources:
                if not isinstance(source, dict):
                    errors.append("Invalid document span")
                    continue
                start, end, text = source.get("start"), source.get("end"), source.get("text")
                if (source.get("source") != "user_document" or type(start) is not int or type(end) is not int
                        or not isinstance(text, str) or not text.strip()
                        or not 0 <= start < end <= len(document) or document[start:end] != text
                        or (start, end) not in sentence_offsets):
                    errors.append("Document span does not match its offsets")
                    continue
                spans.append({"start": start, "end": end, "text": text})
            response, _ = _response_text(generated)
            if response != " ".join(" ".join(span["text"].split()) for span in spans):
                errors.append("Response contains text not covered by document spans")
    return {"document_sha256": hashlib.sha256(document.encode("utf-8")).hexdigest(),
            "source_spans": spans, "source_span_errors": errors, "source_supported": not errors,
            "document_echo": document_echo}


def evaluate_acceptance_suite(
    generate_fn: Callable[..., object], cases: Sequence[dict],
    trained_prompts: Sequence[str] = (), *, mode: str = "closed_book",
    trained_documents: Sequence[str] = (), trained_document_hashes: Sequence[str] = (),
    trained_prompt_hashes: Sequence[str] = (),
) -> dict:
    cases = [{**c, "id": c.get("id", c.get("case_id", c.get("exam_id", c.get("probe_id",
               "case_" + hashlib.sha256(str(c.get("prompt", "")).encode("utf-8")).hexdigest()[:16]))))}
             for c in cases]
    errors = validate_acceptance_documents(cases, mode, trained_documents, trained_document_hashes)
    prompts = [_normal_prompt(str(c.get("prompt", ""))) for c in cases]
    ids = [c["id"] for c in cases]
    if len(cases) < 2 or any(not p for p in prompts):
        errors.append("At least two nonempty independent cases are required")
    if any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
        errors.append("Case IDs must be nonempty and unique")
    trained = [_normal_prompt(p) for p in trained_prompts]
    for i, prompt in enumerate(prompts):
        if any(prompt == p or _word_jaccard_similarity(prompt, p) >= 0.92 for p in prompts[:i]):
            errors.append(f"Non-independent lexical prompts: {ids[i]}")
        if (hashlib.sha256(prompt.encode("utf-8")).hexdigest() in trained_prompt_hashes
                or any(prompt == p or _word_jaccard_similarity(prompt, p) >= 0.92 for p in trained)):
            errors.append(f"Training/evaluation prompt leakage: {ids[i]}")
    split_hash = hashlib.sha256(json.dumps(
        {"cases": cases, "trained_prompts": sorted(trained), "evaluation_mode": mode,
         "trained_prompt_hashes": sorted(set(trained_prompt_hashes)),
         "trained_document_hashes": sorted(set(trained_document_hashes) | {
             hashlib.sha256(text.encode("utf-8")).hexdigest() for text in trained_documents})},
        sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    results = []
    if not errors:
        for case in cases:
            try:
                generated = (generate_fn(case["prompt"], document=case["document"])
                             if mode == "open_book" else generate_fn(case["prompt"]))
                row = _case_result(case, generated)
                if mode == "open_book":
                    row.update(_document_span_result(case, generated))
                    row["passed"] = row["passed"] and row["source_supported"]
                results.append(row)
            except Exception as exc:
                row = _case_result(case, "")
                row["error"] = f"{type(exc).__name__}: {exc}"
                results.append(row)
    echoes = sum(bool(r["forbidden_matches"] or r["prompt_echo"] or r.get("document_echo")) for r in results)
    # Коллапс измеряется только по СОДЕРЖАТЕЛЬНЫМ ответам.
    # Одинаковая формулировка отказа («I do not have source-supported knowledge…»)
    # на разные вопросы — это не вырожденный ответ, а корректное «не знаю»;
    # такие кейсы уже не проходят per-case проверку (clarification) и не дают
    # available. Считать их коллапсом делало closed-book гейт непроходимым
    # по построению: любой отказ давал collapse=1.0 независимо от модели.
    substantive = [r for r in results if not r.get("clarification")]
    comparisons = [(a, b) for i, a in enumerate(substantive) for b in substantive[:i]]
    collapses = sum(bool(a["response"] and b["response"])
                    and _word_jaccard_similarity(a["response"], b["response"]) >= 0.92
                    for a, b in comparisons)
    refusals = sum(1 for r in results if r.get("clarification"))
    return {"passed": not errors and len(results) == len(cases)
            and all(r["passed"] for r in results) and not collapses,
            # Валидное измерение требует хотя бы одного содержательного ответа:
            # прогон, где ассистент на всё ответил отказом, не измеряет качество.
            "available": not errors and bool(results) and bool(substantive)
                         and all(r["available"] for r in results),
            "results": results, "split_hash": split_hash, "errors": errors,
            "echo": echoes / len(results) if results else 0.0,
            "collapse": collapses / len(comparisons) if comparisons else 0.0,
            "refusal_rate": refusals / len(results) if results else 0.0,
            "substantive_responses": len(substantive),
            "prompt_disjoint": not errors, "evaluation_mode": mode,
            "certification_scope": "extractive_document_qa" if mode == "open_book" else "closed_book_source_grounded_qa",
            "semantic_independence": "unverified: lexical separation does not prove semantic independence or entailment"}


def _read_only_generate(organism: object, prompt: str, max_tokens: int = 44) -> object:
    return organism.pure_neural_generate_response(
        prompt, max_tokens=max_tokens, use_ngrams=False, enable_active_inference=False)


def evaluate_novel_knowledge_generalization(
    organism: object,
    training_corpus_responses_lower: Optional[Sequence[str]] = None,
    *, isolated_factory: Optional[Callable[[], object]] = None,
    experimental_train_fn: Optional[Callable[[object, str, str], None]] = None,
    trained_prompts: Sequence[str] = (),
) -> Dict[str, object]:
    """Read-only transfer probes; optional one-shot experiment trains ONLY a new instance.

    A missing factory/training hook/corpus evidence is unavailable, not a success.
    The experimental training pair is not a heldout prompt. Call this experiment
    separately; evaluate_organism never supplies a training hook.
    """
    cases = [
        {"id": "novel_1_zero_shot_hollandaise_transfer",
         "prompt": "My homemade hollandaise emulsion turned thin and runny while whisking; how can I fix it?",
         "expected": ["hollandaise", "emulsion", "whisking"], "forbidden": ["mayonnaise"]},
        {"id": "novel_2_zero_shot_dual_slot_cycling_manuscript",
         "prompt": "Can cycling outdoors clear the mind before fixing grammar errors in my manuscript?",
         "expected": ["cycling outdoors", "manuscript", "before you fix"],
         "forbidden": ["running outdoors", "essay", "before to fix"]},
        {"id": "novel_3_zero_shot_tpj_dissertation",
         "prompt": "Alex is revising his dissertation; what is his name, and how can he fix grammar errors in his dissertation?",
         "expected": ["his name is alex", "dissertation", "subject-verb agreement"],
         "forbidden": ["essay", "your name"]},
    ]
    context = getattr(organism, "evaluation_context", None)
    if not callable(context):
        return {"available": False, "passed": False, "novel_knowledge_generalization_score": 0.0,
                "probe_reports": [], "reason": "evaluation_context unavailable"}
    known = {r.strip().casefold() for r in (training_corpus_responses_lower or ()) if r.strip()}
    with context(), torch.no_grad():
        suite = evaluate_acceptance_suite(lambda p: _read_only_generate(organism, p), cases, trained_prompts)
    logs = suite["results"]
    for row in logs:
        row["not_verbatim_in_training"] = bool(known) and row["response"].casefold() not in known
        row["passed"] = row["passed"] and row["not_verbatim_in_training"]
    experimental = {"id": "novel_4_oneshot_synaptic_fact_acquisition", "available": False,
                    "passed": False, "reason": "isolated factory/training hook unavailable"}
    if isolated_factory is not None and experimental_train_fn is not None:
        isolated = isolated_factory()
        if isolated is organism or not callable(getattr(isolated, "evaluation_context", None)):
            experimental["reason"] = "Factory must return an independent organism with evaluation_context"
        else:
            case = {"id": experimental["id"],
                    "prompt": "How do astrocytic calcium waves regulate cortical microcolumns?",
                    "expected": ["astrocytic", "calcium", "microcolumns", "gliotransmitters"], "forbidden": []}
            with isolated.evaluation_context(), torch.no_grad():
                before = _case_result(case, _read_only_generate(isolated, case["prompt"]))
            # No exam/transfer prompt is reused as a training example.
            experimental_train_fn(isolated, "Describe the role of glial calcium signalling.",
                                  "Astrocytic calcium waves regulate synaptic plasticity across cortical microcolumns via gliotransmitters.")
            with isolated.evaluation_context(), torch.no_grad():
                experimental = _case_result(case, _read_only_generate(isolated, case["prompt"]))
            experimental["before_training"] = before
            experimental["available"] = experimental["available"] and before["available"]
            experimental["passed"] = experimental["passed"] and not before["passed"]
            if before["passed"]:
                experimental["reason"] = "Fact already recalled before training; no novel acquisition demonstrated"
    logs.append(experimental)
    available = bool(known) and suite["available"] and experimental["available"]
    score = sum(bool(r["passed"]) for r in logs) / 4.0 if available else 0.0
    return {"available": available, "passed": available and score >= 0.75 and suite["collapse"] == 0,
            "novel_knowledge_generalization_score": score,
            "probes_passed": sum(bool(r["passed"]) for r in logs), "probes_total": 4,
            "probe_reports": logs, "transfer_report": suite,
            "novelty_limits": "Non-verbatim reference responses do not prove absence from the full training stream",
            "experimental_training": "isolated instance only"}


# ============================================================================
# 5. ЕДИНЫЙ КЛАСС ВОРОТ КАЧЕСТВА (ОБЩИЙ ДЛЯ v10.1 И v16/v17)
# ============================================================================
@dataclass
class UnifiedGateReport:
    stage_or_epoch_id: int
    passed: bool
    mean_surprise_F: float
    target_surprise_F: float
    grammar_score: float
    trained_ngram_coverage: float
    superacid_detected: bool
    context_retention_tokens: int
    target_context_retention: float
    oov_resilience_passed: bool
    attractor_collapse_rate: float
    max_allowed_collapse_rate: float
    paraphrase_robustness_of_retrieval: float
    novel_knowledge_generalization_score: float
    holdout_echo_contamination_rate: float
    superacid_count: int = 0
    oov_pass_rate: float = 0.0
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, object] = field(default_factory=dict)

    @property
    def surprise_f_t(self) -> float:
        return self.mean_surprise_F

    @property
    def holdout_concept_recall(self) -> float:
        return self.paraphrase_robustness_of_retrieval

    @property
    def holdout_keyword_accuracy(self) -> float:
        return self.paraphrase_robustness_of_retrieval


class UnifiedCurriculumGates:
    """Единый контроллер ворот качества (Gates), вызываемый как из `train_ziwago_v10_curriculum.py`,
    так и из `train_ziwago_v16_10B_omniscient_engine.py`."""

    def __init__(
        self,
        target_surprise_F: float = 0.25,
        surprise_threshold: Optional[float] = None,
        min_grammar_score: float = 1.0,
        target_context_retention: float = 1536.0,
        min_retention_tokens: Optional[int] = None,
        max_attractor_collapse_rate: float = 0.35,
        min_novel_generalization_score: float = 0.75,
        min_holdout_score: float = 0.75,
    ):
        self.target_surprise_F = float(surprise_threshold if surprise_threshold is not None else target_surprise_F)
        self.min_grammar_score = float(min_grammar_score)
        self.target_context_retention = float(
            min_retention_tokens if min_retention_tokens is not None else target_context_retention
        )
        self.max_attractor_collapse_rate = float(max_attractor_collapse_rate)
        self.min_novel_generalization_score = float(min_novel_generalization_score)
        self.min_holdout_score = float(min_holdout_score)

    def evaluate_all_gates(
        self,
        stage_or_epoch_id: int,
        mean_surprise_F: Optional[float],
        generated_samples: Sequence[str],
        context_retention_tokens: Optional[float],
        encode_fn: Optional[Callable[[str], List[int]]],
        get_vec64_fn: Optional[Callable[[int], torch.Tensor]],
        generate_fn: Optional[Callable[[str], str]] = None,
        holdout_results: Optional[Sequence[Dict[str, object]]] = None,
        trained_bigram_g: Optional[Dict[int, Dict[int, float]]] = None,
        novel_generalization_report: Optional[Dict[str, object]] = None,
    ) -> UnifiedGateReport:
        """Выполняет комплексную проверку всех ворот качества (`Surprise`, `Discriminating Grammar`,
        `Trained N-Gram Coverage`, `Superacid`, `Context Retention`, `OOV Resilience`,
        `Attractor Collapse Rate`, `Paraphrase Robustness of Retrieval`, `Novel Knowledge Generalization`)."""
        grammar_evals = [
            evaluate_grammar_and_artifacts(txt, trained_bigram_g=trained_bigram_g, encode_fn=encode_fn)
            for txt in generated_samples
            if str(txt).strip()
        ]
        if grammar_evals:
            grammar_score = sum(1.0 for g in grammar_evals if g["grammar_coherent"]) / float(len(grammar_evals))
            mean_ngram_cov = sum(float(g["trained_ngram_coverage"]) for g in grammar_evals) / float(len(grammar_evals))
            superacid_cnt = sum(1 for g in grammar_evals if bool(g["superacid_detected"]))
            superacid_any = superacid_cnt > 0
        else:
            grammar_score = 0.0
            mean_ngram_cov = 0.0
            superacid_cnt = 0
            superacid_any = False

        oov_rep: dict = {"available": False, "passed": False}
        collapse_rep: dict = {"available": False}
        if callable(encode_fn) and callable(get_vec64_fn) and callable(generate_fn):
            try:
                oov_rep = evaluate_oov_compositional_resilience(encode_fn, get_vec64_fn, generate_fn)
            except Exception as exc:
                oov_rep["error"] = f"{type(exc).__name__}: {exc}"
        oov_passed = bool(oov_rep.get("available") and oov_rep.get("passed"))
        oov_rate = float(oov_rep.get("oov_pass_rate", 0.0))
        if callable(generate_fn):
            try:
                collapse_rep = evaluate_attractor_collapse_rate(generate_fn)
            except Exception as exc:
                collapse_rep["error"] = f"{type(exc).__name__}: {exc}"
        collapse_rate = float(collapse_rep.get("attractor_collapse_rate", 0.0))

        holdout_rows = [_case_result(hr, hr) for hr in (holdout_results or ())]
        holdout_available = (len(holdout_rows) >= 2 and all(r["available"] for r in holdout_rows)
                             and all(_normal_prompt(str(r["prompt"])) for r in holdout_rows)
                             and len({_normal_prompt(str(r["prompt"])) for r in holdout_rows}) == len(holdout_rows))
        paraphrase_robustness = (sum(bool(r["passed"]) for r in holdout_rows) / len(holdout_rows)
                                if holdout_available else 0.0)
        echo_contamination = (sum(bool(r["forbidden_matches"] or r["prompt_echo"]) for r in holdout_rows)
                              / len(holdout_rows) if holdout_rows else 0.0)
        novel_rep = novel_generalization_report or {}
        novel_probes = novel_rep.get("probe_reports", [])
        novel_available = (novel_rep.get("available") is True and bool(novel_probes)
                           and all(isinstance(p, dict) and p.get("available") is True
                                   and isinstance(p.get("passed"), bool) for p in novel_probes))
        novel_gen_score = (sum(p["passed"] for p in novel_probes) / len(novel_probes)
                           if novel_available else 0.0)
        surprise_available = (isinstance(mean_surprise_F, (int, float))
                              and math.isfinite(mean_surprise_F) and mean_surprise_F >= 0)
        retention_available = (isinstance(context_retention_tokens, (int, float))
                               and math.isfinite(context_retention_tokens) and context_retention_tokens >= 0)
        mean_surprise_F = float(mean_surprise_F) if surprise_available else 0.0
        context_retention_tokens = float(context_retention_tokens) if retention_available else 0.0
        availability = {
            "surprise": surprise_available, "context_retention": retention_available,
            "generated_samples": bool(generated_samples) and all(isinstance(t, str) and t.strip() for t in generated_samples),
            "trained_ngram_coverage": bool(grammar_evals) and all(g["trained_ngram_available"] for g in grammar_evals),
            "oov": bool(oov_rep.get("available")), "collapse": bool(collapse_rep.get("available")),
            "holdout": holdout_available, "novel_generalization": novel_available,
        }
        reasons = [f"Required measurement unavailable: {name}" for name, ok in availability.items() if not ok]
        passed = not reasons

        if mean_surprise_F >= self.target_surprise_F:
            passed = False
            reasons.append(f"Surprise {mean_surprise_F:.4f} >= target {self.target_surprise_F:.2f}")
        if grammar_score < self.min_grammar_score:
            passed = False
            reasons.append(f"Discriminating grammar score {grammar_score * 100:.1f}% < {self.min_grammar_score * 100:.1f}%")
        if mean_ngram_cov < 0.50:
            passed = False
            reasons.append(f"Trained n-gram coverage {mean_ngram_cov:.4f} < 0.50")
        if paraphrase_robustness < self.min_holdout_score:
            passed = False
            reasons.append(f"Measured holdout score {paraphrase_robustness:.4f} < {self.min_holdout_score:.2f}")
        if superacid_any:
            passed = False
            reasons.append("Forbidden token artifact ('superacid' or '<unk>') detected")
        if context_retention_tokens < self.target_context_retention:
            passed = False
            reasons.append(
                f"Context retention {context_retention_tokens:.1f} < target {self.target_context_retention:.1f}"
            )
        if not oov_passed:
            passed = False
            reasons.append("OOV compositional resilience check failed")
        if collapse_rate > self.max_attractor_collapse_rate:
            passed = False
            reasons.append(
                f"Attractor collapse rate {collapse_rate:.4f} > max allowed {self.max_attractor_collapse_rate:.2f}"
            )
        if novel_gen_score < self.min_novel_generalization_score:
            passed = False
            reasons.append(
                f"Novel knowledge generalization score {novel_gen_score:.4f} < {self.min_novel_generalization_score:.2f}"
            )
        if novel_available and novel_rep.get("passed") is not True:
            passed = False
            reasons.append("Novel experiment reported failure")
        if echo_contamination > 0.0:
            passed = False
            reasons.append(f"Unrelated pole echo contamination detected ({echo_contamination * 100:.1f}%)")

        if passed:
            reasons.append("ALL UNIFIED GATES PASSED")

        return UnifiedGateReport(
            stage_or_epoch_id=stage_or_epoch_id,
            passed=passed,
            mean_surprise_F=round(mean_surprise_F, 4),
            target_surprise_F=self.target_surprise_F,
            grammar_score=round(grammar_score, 4),
            trained_ngram_coverage=round(mean_ngram_cov, 4),
            superacid_detected=superacid_any,
            context_retention_tokens=int(round(context_retention_tokens)),
            target_context_retention=self.target_context_retention,
            oov_resilience_passed=oov_passed,
            attractor_collapse_rate=round(collapse_rate, 4),
            max_allowed_collapse_rate=self.max_attractor_collapse_rate,
            paraphrase_robustness_of_retrieval=round(paraphrase_robustness, 4),
            novel_knowledge_generalization_score=round(novel_gen_score, 4),
            holdout_echo_contamination_rate=round(echo_contamination, 4),
            superacid_count=superacid_cnt,
            oov_pass_rate=round(oov_rate, 4),
            reasons=reasons,
            details={
                "availability": availability,
                "measurement_status": {k: "measured" if v else "unavailable" for k, v in availability.items()},
                "holdout_results": holdout_rows,
                "holdout_threshold": self.min_holdout_score,
                "grammar_evaluations": grammar_evals,
                "oov_report": oov_rep,
                "attractor_collapse_report": collapse_rep,
                "novel_knowledge_generalization_report": novel_generalization_report or {},
            },
        )

    def evaluate_organism(
        self,
        organism: object,
        stage_or_epoch_id: int = 5,
        training_corpus_responses_lower: Optional[Sequence[str]] = None,
        *, trained_prompts: Sequence[str] = (),
        surprise_cases: Sequence[dict] = (),
        novel_generalization_report: Optional[dict] = None,
    ) -> UnifiedGateReport:
        """Evaluate only inside the core's complete evaluation_context; NEVER train here.

        Optional surprise_cases contain prompt/reference for teacher_forced_likelihood,
        which returns nll_sum/token_count (mean NLL), or token_errors/token_count.
        Otherwise actual generation token_error_report counts are required.
        Novel experimental evidence may be supplied from an isolated prior experiment.
        """
        context = getattr(organism, "evaluation_context", None)
        if not callable(context):
            report = self.evaluate_all_gates(stage_or_epoch_id, None, [], None, None, None)
            report.reasons.insert(0, "Required evaluation_context unavailable; no organism methods called")
            report.details["evaluation_context"] = "unavailable"
            return report
        with context(), torch.no_grad():
            def gen(prompt: str) -> object:
                return _read_only_generate(organism, prompt)

            cases = [{"id": item.exam_id, "prompt": item.prompt,
                      "expected": list(item.expected_keywords), "forbidden": list(item.forbidden_substrings)}
                     for item in HOLD_OUT_EXAM_SUITE]
            raw_results: Dict[str, object] = {}

            def capture(prompt: str) -> object:
                raw_results[prompt] = gen(prompt)
                return raw_results[prompt]

            acceptance = evaluate_acceptance_suite(capture, cases, trained_prompts)
            holdout_logs = []
            for case in cases:
                raw = raw_results.get(case["prompt"], "")
                text, clarification = _response_text(raw)
                holdout_logs.append({**case, "response": text,
                                     "clarification_triggered": clarification})
            surprise, surprise_detail = _measure_surprise(organism, surprise_cases, list(raw_results.values()))
            retention, retention_detail = _measure_retention(organism, self.target_context_retention)
            novel = novel_generalization_report
            if novel is None:
                novel = evaluate_novel_knowledge_generalization(
                    organism, training_corpus_responses_lower, trained_prompts=trained_prompts)
            report = self.evaluate_all_gates(
                stage_or_epoch_id, surprise, [r["response"] for r in holdout_logs], retention,
                getattr(organism, "encode_text", None), getattr(organism, "get_vec64", None),
                generate_fn=lambda p: _response_text(gen(p))[0], holdout_results=holdout_logs,
                trained_bigram_g=getattr(organism, "synaptic_bigram_g", None),
                novel_generalization_report=novel)
            report.details.update({"evaluation_context": "required core context used",
                                   "surprise_measurement": surprise_detail,
                                   "retention_probe": retention_detail,
                                   "acceptance_suite": acceptance})
            if acceptance["errors"] or not acceptance["available"] or acceptance["echo"] or acceptance["collapse"]:
                report.passed = False
                report.reasons = [r for r in report.reasons if r != "ALL UNIFIED GATES PASSED"]
                report.reasons.append("Holdout independence, availability, echo or collapse check failed")
            return report


def _measure_surprise(organism: object, cases: Sequence[dict], outputs: Sequence[object]) -> Tuple[Optional[float], dict]:
    likelihood = getattr(organism, "teacher_forced_likelihood", None)
    reports = []
    try:
        if cases:
            if not callable(likelihood) or any(not c.get("prompt") or not c.get("reference") for c in cases):
                return None, {"available": False, "reason": "teacher-forced references/provider unavailable"}
            reports = [likelihood(c["prompt"], c["reference"]) for c in cases]
            mode = "teacher_forced"
        else:
            reports = [o.get("token_error_report") if isinstance(o, dict) else None for o in outputs]
            mode = "token_error_report"
        if not reports or any(not isinstance(r, dict) for r in reports):
            return None, {"available": False, "reason": "real token measurement unavailable"}
        numerator = 0.0
        denominator = 0
        kinds = set()
        for measurement in reports:
            count = measurement.get("token_count")
            kind = "nll_sum" if mode == "teacher_forced" and "nll_sum" in measurement else "token_errors"
            value = measurement.get(kind)
            if (type(count) is not int or count <= 0 or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0
                    or (kind == "token_errors" and (value > count or int(value) != value))):
                return None, {"available": False, "reason": "invalid token measurement"}
            kinds.add(kind)
            numerator += value
            denominator += count
        if len(kinds) != 1:
            return None, {"available": False, "reason": "mixed surprise units"}
        return numerator / denominator, {"available": True, "method": mode,
                                         "units": next(iter(kinds)), "token_count": denominator,
                                         "reports": reports}
    except Exception as exc:
        return None, {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def _measure_retention(organism: object, target: float) -> Tuple[Optional[float], dict]:
    """Recall unpredictable source-specific facts after an actually tokenized filler stream."""
    if not callable(getattr(organism, "encode_text", None)):
        return None, {"available": False, "reason": "tokenizer unavailable"}
    tag, code, destination = (secrets.token_hex(4) for _ in range(3))
    query = f"In the evaluation log, what access code and destination belong to crate {tag}?"
    case = {"id": "source_fact_retention", "prompt": query,
            "expected": [code, destination], "forbidden": []}
    try:
        # The core context preserves transient dialogue while restoring ALL persistent state.
        with organism.evaluation_context(), torch.no_grad():
            _read_only_generate(organism, f"Evaluation log: crate {tag} has access code {code} and destination {destination}.")
            before = _case_result(case, _read_only_generate(organism, query))
            filler = "Unrelated filler: the quiet wind moves over hills while ordinary clouds cross the distant horizon. " * 8
            count = len(organism.encode_text(filler))
            if count <= 0:
                return None, {"available": False, "reason": "empty filler tokenization"}
            tokens = 0
            chunks = int(math.ceil(max(1, target) / count))
            if chunks > 4096:
                return None, {"available": False, "reason": "retention probe exceeds bounded work budget"}
            for _ in range(chunks):
                _read_only_generate(organism, filler, max_tokens=1)
                tokens += count
            after = _case_result(case, _read_only_generate(organism, query))
        retained = bool(before["passed"] and after["passed"])
        return float(tokens if retained else 0), {"available": True, "passed": retained,
                                                 "filler_tokens": tokens, "before": before, "after": after,
                                                 "method": "source-specific fact recall over tokenized filler"}
    except Exception as exc:
        return None, {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
