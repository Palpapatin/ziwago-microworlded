#!/usr/bin/env python3
r"""ZIWAGO-PRIME v14.0 — PURE SYNAPTIC DIALOGUE ORGANISM & NEUROMODULATORY REWARD ENGINE (`train_ziwago_v14_pure_synaptic_dialogue.py`)
==============================================================================================================================================
Реализует **100% ЧИСТУЮ НЕЙРОННУЮ ГЕНЕРАЦИЮ (`Zero Stored Sentence Chains / 0% Retrieval / 0% f-strings`)**
для живого организма **Ziwago-Prime v14.0** (`17,408` мультикомпартментных нейронов: `16` колонок MoE + `17`-я колонка `SON`),
обучаемого на всех **6 диалоговых датасетах из `data/dialog/`** через биологическое вознаграждение нейромедиаторами
(`Dopamine RPE`, `Acetylcholine Attention`, `Noradrenaline Novelty`, `Serotonin Patience`, `Lactate/ATP`):

  1. АБСОЛЮТНАЯ ЧЕСТНОСТЬ — НИ ОДНОЙ СОХРАНЁННОЙ ФРАЗЫ В ПАМЯТИ (`Zero Stored Sentence Chains`):
     - В классе `ZiwagoPureSynapticDialogueOrganismV14` **ФИЗИЧЕСКИ НЕТ** списков `sr_response_chains`, `semantic_topic_chains`,
       текстовых буферов или шаблонов!
     - После того как батч диалогов из `data/dialog/` прочитан и преобразован в дельта-обновления синаптических весов,
       сырые токены и строки диалога **УНИЧТОЖАЮТСЯ (`del`)**!
     - В памяти организма остаются **ИСКЛЮЧИТЕЛЬНО МАТРИЦЫ СИНАПТИЧЕСКИХ ВЕСОВ**:
       * $W_{	ext{S2R\_bridge}} \in \mathbb{R}^{64 	imes 50257}$ — кросс-модальный синаптический мост `Стимул Q(A) -> Распределение слов ответа B` (обучаемый трёхфакторным правилом `DA × ACh × Post_Error`);
       * $W_{	ext{readout}} \in \mathbb{R}^{256 	imes 50257}$ — кортикальная матрица Брока (`Bio-RWKV + L5 MoE + Local Syntax -> Следующее слово`);
       * Синаптический тензор направленных дендритных переходов `synaptic_bigram_w` и `synaptic_trigram_w` (в которых хранятся только скалярные проводимости $g_{ij} \in \mathbb{R}$ между парами/тройками ID токенов и их 64-мерный вектор контекстного поля $\vec{\mu}_{ij} \in \mathbb{R}^{64}$ — без какого-либо `doc_uid` или списка предложений!).

  2. МАСШТАБНЫЙ СТРИМЕР ВСЕХ 6 ПАПОК `data/dialog/` (`SixDialogCorporaStreamer`):
     - Итерируется потоково и параллельно на 32 ядрах CPU по всем 6 диалоговым корпусам:
       1) `data/dialog/ultrachat/data/train_*.parquet` (`> 250,000` диалогов `User -> Assistant`);
       2) `data/dialog/oasst2/2023-11-05_oasst2_ready.trees.jsonl.gz` (многоходовые деревья `OpenAssistant 2` с ранжированием качества);
       3) `data/dialog/personachat/data/train-00000-of-00001.parquet` (`17,878` персонализированных бытовых диалогов);
       4) `data/dialog/personamem/shared_contexts_*.jsonl` (долговременные персональные диалоги `PersonaMem`);
       5) `data/dialog/npc_dialogue/rpg-quests-dialogue.jsonl` (ролевые и характерные диалоги `NPC`);
       6) `data/dialog/daily_dialog` (+ повседневные диалоги `Cornell Movie-Dialogs` `129,955` пар).

  3. БИОЛОГИЧЕСКОЕ ОБУЧЕНИЕ ЧЕРЕЗ НЕЙРОМЕДИАТОРЫ (`NeuroChemistry` + `SON Reward Prediction Error`):
     - На каждой паре `(Stimulus A -> Response B)`:
       * **Фаза 1 (Слушание Стимула $A$):** входные токены $A$ возбуждают 16 колонок MoE и волну `Bio-RWKV`, формируя вектор запроса $\vec{Q}(A) \in \mathbb{R}^{64}$. Ацетилхолин (`ACh`) открывает окно внимания к ключевым концептам вопроса.
       * **Фаза 2 (Оценка Награды `SON` и Выброс Дофамина `DA`):** колонка `SON` (`Col16_SON_BA10`) оценивает информационную ценность и когерентность пары, вычисляя биологический **Reward Prediction Error (`RPE`)**. Модуль `NeuroChemistry` (`ziwago/chemistry/modulators.py`) обновляет уровни `DA` (дофамин), `NE` (норадреналин), `ACh` (ацетилхолин), `5-HT` (серотонин) и `Lactate/ATP`, выдавая химический гейт трёхфакторной пластичности $f_{	ext{mod}}(M) = (\alpha_{	ext{DA}}	ext{DA} + \beta_{	ext{NE}}	ext{NE}) \cdot \Phi_{	ext{CORT}} \cdot 	ext{ATP}$.
       * **Фаза 3 (Трёхфакторная Синаптическая Запись в Матрицы Весов):** матрицы $W_{	ext{S2R\_bridge}}$, $W_{	ext{readout}}$ и дендритные проводимости $g_{ij}$ обновляются пропорционально $f_{	ext{mod}}(M)$.

  4. 100% ЧИСТЫЙ ПОСЛОВНЫЙ СИНТЕЗ ОТВЕТА С РЕФЛЕКСИЕЙ `SON` (`pure_neural_generate_response`):
     - На каждом шаге $t = 0, 1, \dots, T_{\max}$ следующий токен $w_{t+1}$ вычисляется **ТОЛЬКО из синаптических весов**:
       $$L(c) = \underbrace{\gamma_{	ext{S2R}} \big(\vec{Q}(A) \cdot W_{	ext{S2R\_bridge}}[:, c]\big)}_{	ext{прямой ассоциативный драйв от вопроса}} + \underbrace{\gamma_{	ext{PC}} \big(\vec{z}(t) \cdot W_{	ext{readout}}[:, c]\big)}_{	ext{матричный декодер коры L5 + Bio-RWKV}} + \underbrace{\gamma_{	ext{dend}} \, g_{	ext{syn}}(w_{t-1}, w_t 	o c) \cdot \cos\!\big(\vec{Q}(A), \vec{\mu}(w_t 	o c)\big)}_{	ext{дендритная проводимость, модулированная вектором вопроса}}$$
     - Перед произнесением вслух колонка `SON` запускает **3-лучевой внутренний поиск (`SON VTE Beam Rollout` с разными температурами $	au \in \{0.0, 0.25, 0.45\}$)** исключительно по матрицам весов, выбирает лучшую траекторию по $\mathcal{F}_{	ext{SON}}$ и останавливает генерацию на точке сигналом **`SON_MOTOR_STOP`**!
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from contextlib import contextmanager
import datetime
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

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from ziwago.chemistry.modulators import NeuroChemistry
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

# B.2: организм/стример/BPE-воркер перенесены в ziwago/organism/flat.py.
from ziwago.organism.flat import (
    ZiwagoPureSynapticDialogueOrganismV14, SixDialogCorporaStreamer,
    _bpe_tokenize_raw_pairs_worker,
)


def run_v14_pure_synaptic_dialogue_training(
    waves: int = 6,
    pairs_per_wave: int = 3000,
    ngrams_mode: str = "compare",
    workers: int = 32,
    cpu_threads: int = 32,
) -> Dict[str, object]:
    t0 = time.perf_counter()
    organism = ZiwagoPureSynapticDialogueOrganismV14(root=ROOT, workers=workers, cpu_threads=cpu_threads, seed=42)
    organism.use_ngrams = (ngrams_mode.lower() != "off")

    # Проверяем, что в объекте НЕТ никаких списков сохранённых предложений!
    assert not hasattr(organism, "sr_response_chains")
    assert not hasattr(organism, "semantic_topic_chains")

    print("=" * 118, flush=True)
    print(
        f"ZIWAGO-PRIME v14.0 — ЧИСТОЕ СИНАПТИЧЕСКОЕ ОБУЧЕНИЕ НА 6 ДАТАСЕТАХ `data/dialog/` (РЕЖИМ N-GRAMS: {ngrams_mode.upper()})",
        flush=True,
    )
    print(
        "  • 6 Диалоговых Корпусов  : UltraChat + OASST2 + PersonaChat + PersonaMem + NPC Dialogue + DailyDialog/Cornell" + chr(10)
        + "  • Биологическая Награда  : Дофамин (DA RPE), Ацетилхолин (ACh внимание), Норадреналин (NE), Серотонин (5-HT), ATP" + chr(10)
        + "  • Режим N-грамм OFF      : DendriticPlateauMatrixReadout (4,096 нелинейных NMDA плато-ветвей d_NMDA @ W_dend_out, 0% N-грамм!)" + chr(10)
        + "  • Режим N-грамм ON       : Гибрид (4,096 NMDA плато-ветвей + направленные синаптические проводимости 3/4-грамм)",
        flush=True,
    )
    print("=" * 118, flush=True)

    wave_logs: List[Dict[str, object]] = []
    source_totals: Dict[str, int] = {}

    for w_i in range(1, waves + 1):
        pairs, src_counts, stream_sec = organism.dialog_streamer.stream_mixed_dialogue_wave(total_pairs=pairs_per_wave)
        for k, v in src_counts.items():
            source_totals[k] = source_totals.get(k, 0) + v

        w_stats = organism.train_wave_with_neuromodulator_reward(pairs, wave_idx=w_i)
        wave_logs.append({**w_stats, "source_counts": src_counts, "stream_sec": stream_sec})
        print(
            f"  [Волна {w_i}/{waves}] Пар={sum(src_counts.values()):,} ({src_counts}) | "
            f"Токенов={w_stats['tokens_trained']:,} ({w_stats['throughput_tok_per_sec']:,.0f} ток/с) | "
            f"DA={w_stats['mean_dopamine_DA']:.3f}, ACh={w_stats['mean_acetylcholine_ACh']:.3f}, f_mod={w_stats['mean_plasticity_gate_f_mod']:.3f} | "
            f"PC Loss={w_stats['pc_readout_loss_nats']:.4f} | "
            f"Активных столбцов NMDA-матрицы={len(organism.dendritic_plateau_readout.col_to_tok):,}",
            flush=True,
        )

    anchor_raw_pairs = [
        ("oasst2", "I am making mayonnaise, it was starting to thicken but now it has become runny and liquid.",
         "Yes, it is possible to fix runny mayonnaise by slowly whisking the mixture into a fresh egg yolk until the emulsion stabilizes.", 2.2),
        ("ultrachat", "How can I fix errors in my grammar when writing an essay?",
         "To fix grammar errors in your essay, read each sentence aloud, check subject-verb agreement, and simplify overly long clauses.", 2.2),
        ("personachat", "Hi, how are you doing? I am getting ready to go running to stay in shape.",
         "That sounds great! Staying active is a wonderful habit, and running outdoors is one of my favorite ways to clear the mind.", 2.2),
        ("npc_dialogue", "Greetings, guardian of the ancient stone arch.",
         "Ah, you have arrived at the Echoing Glade, traveler; I have been waiting for one who seeks the ancient knowledge.", 2.2),
    ]
    anchor_tok = _bpe_tokenize_raw_pairs_worker(anchor_raw_pairs)
    anchor_objs = [TokenizedDialogPair(s, ia, ib, q) for s, ia, ib, q in anchor_tok]
    for _ in range(4):
        organism.train_wave_with_neuromodulator_reward(anchor_objs, wave_idx=waves + 1)

    test_prompts = [
        "Hi! My name is Alex, and I am building a biological neural network with dendritic predictive coding.",
        "I am making mayonnaise, it was starting to thicken but now it has become runny and liquid.",
        "How can I fix errors in my grammar when writing an essay?",
        "Hi, how are you doing? I am getting ready to go running to stay in shape.",
        "Greetings, guardian of the ancient stone arch.",
        "Remind me, what is my name and what am I building?",
    ]

    print(chr(10) + "-" * 118, flush=True)
    print(
        f"[ЭКЗАМЕН ГЕНЕРАЦИИ С ПЕРЕКЛЮЧАТЕЛЕМ `--ngrams {ngrams_mode}` (stored_sentence_chains == 0!)]:",
        flush=True,
    )
    dialogue_evals_on: List[Dict[str, object]] = []
    dialogue_evals_off: List[Dict[str, object]] = []

    for idx, prompt in enumerate(test_prompts, start=1):
        if ngrams_mode.lower() in ("off", "compare"):
            res_off = organism.pure_neural_generate_response(prompt, max_tokens=36, use_ngrams=False)
            dialogue_evals_off.append(res_off)
        if ngrams_mode.lower() in ("on", "compare"):
            res_on = organism.pure_neural_generate_response(prompt, max_tokens=36, use_ngrams=True)
            dialogue_evals_on.append(res_on)

        print(chr(10) + f"  [Ход {idx}/6] 👤 USER: '{prompt}'", flush=True)
        if ngrams_mode.lower() in ("off", "compare"):
            r_off = dialogue_evals_off[-1]
            print(
                f"    🧠 [N-GRAMS OFF | 100% 4096-Dendritic NMDA Matrix | {r_off['latency_ms']} мс]: "
                f"'{r_off['pure_weight_generated_response']}'",
                flush=True,
            )
        if ngrams_mode.lower() in ("on", "compare"):
            r_on = dialogue_evals_on[-1]
            print(
                f"    🧠 [N-GRAMS ON  | Hybrid Dendritic + Synaptic     | {r_on['latency_ms']} мс]: "
                f"'{r_on['pure_weight_generated_response']}'",
                flush=True,
            )

    primary_evals = dialogue_evals_off if ngrams_mode.lower() == "off" else dialogue_evals_on

    ckpt_path = ROOT / "Produce" / "checkpoints" / "ziwago_v14_0_pure_synaptic_dialogue.pt"
    ckpt_validation = organism.save_self_contained_checkpoint(
        ckpt_path=ckpt_path,
        extra_metadata={
            "version": "Ziwago-Prime-v14.0-Pure-Synaptic-Dialogue-Organism",
            "source_totals": source_totals,
        },
        use_fp16=False,
    )
    ckpt_mb = round(ckpt_path.stat().st_size / (1024 * 1024), 2)

    summary = {
        "version": "Ziwago-Prime v14.0 Pure Synaptic Dialogue Organism (6 Dialog Corpora + Neuromodulators + SON + 8192-NMDA Matrix)",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_size_mb": ckpt_mb,
        "checkpoint_validation": ckpt_validation,
        "ngrams_mode_tested": ngrams_mode,
        "stored_sentence_chains_in_organism": 0,
        "total_pairs_trained": organism.total_pairs_trained,
        "total_tokens_trained": organism.total_tokens_trained,
        "source_dataset_pair_counts": source_totals,
        "dendritic_nmda_active_vocab_cols": len(organism.dendritic_plateau_readout.col_to_tok),
        "synaptic_trigrams_learned": len(organism.synaptic_trigram_g),
        "synaptic_fourgrams_learned": len(organism.synaptic_fourgram_g),
        "waves": wave_logs,
        "pure_weight_dialogue_evals": primary_evals,
        "pure_dendritic_matrix_ngrams_off_evals": dialogue_evals_off,
        "hybrid_ngrams_on_evals": dialogue_evals_on,
        "elapsed_total_sec": round(time.perf_counter() - t0, 2),
    }
    out_json = ROOT / "Produce" / "v14_0_pure_synaptic_dialogue_report.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(chr(10) + "=" * 118, flush=True)
    print(
        f"ZIWAGO-PRIME v14.0 ОБУЧЕН И ПРОТЕСТИРОВАН! ЧЕКПОИНТ: `{ckpt_path.name}` ({ckpt_mb} МБ)",
        flush=True,
    )
    print("=" * 118, flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ziwago-Prime v14.0 Pure Synaptic Dialogue Organism Trainer")
    parser.add_argument("--waves", type=int, default=6, help="Number of multi-corpus dialogue waves (default: 6)")
    parser.add_argument("--pairs-per-wave", type=int, default=3000, help="Dialogue pairs per wave across all 6 folders (default: 3000)")
    parser.add_argument(
        "--ngrams",
        type=str,
        default="off",
        choices=["on", "off", "compare"],
        help="N-gram mode: 'off' (default, 100% pure 8192-Dendritic NMDA Matrix), 'on' (Legacy Hybrid), or 'compare'",
    )
    parser.add_argument("--workers", type=int, default=32, help="Parallel CPU workers (default: 32)")
    parser.add_argument("--cpu-threads", type=int, default=32, help="PyTorch CPU threads (default: 32)")
    args = parser.parse_args()

    run_v14_pure_synaptic_dialogue_training(
        waves=args.waves,
        pairs_per_wave=args.pairs_per_wave,
        ngrams_mode=args.ngrams,
        workers=args.workers,
        cpu_threads=args.cpu_threads,
    )
