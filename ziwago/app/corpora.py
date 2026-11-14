"""corpora — потоковые загрузчики диалоговых корпусов (Cornell Movie-Dialogs и др.).

Вынесено из train_ziwago_v13_son_reflective_organism.py (#31, 2026-11-12):
StimulusResponsePair + PairedDialogueStreamer (verbatim, marker-based surgery).
Зависит только от GPT2-BPE токенизатора v9 (фундаментальный, остаётся) + pyarrow + stdlib;
НЕ тянет v13-движок (ZiwagoReflectiveOrganismV13) и v11.5. v14 импортирует отсюда;
v13 ре-экспортирует для обратной совместимости.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow.parquet as pq

from ziwago.app.tokenizer import _GPT2_PAT, _bpe_encode_piece, _init_bpe_worker
import ziwago.app.tokenizer as _gpt2_mod


@dataclass
class StimulusResponsePair:
    pair_uid: str
    domain: str
    stimulus_text: str
    response_text: str
    stimulus_ids: List[int]
    response_ids: List[int]


class PairedDialogueStreamer:
    """Извлекает чистые диалоговые пары `(Stimulus A -> Response B)` без театральных меток `CHARACTER :`
    из `Cornell Movie-Dialogs` (`129,955` пар), `BookCorpus`, `TinyStories`, `WikiText-103` и любых
    пользовательских диалоговых датасетов (`--extra-dialog-path`)."""

    def __init__(self, root: Path, extra_dialog_path: Optional[Path] = None):
        self.root = root
        self.data_dir = root / "data"
        self.extra_dialog_path = extra_dialog_path

        vocab_path = self.data_dir / "gpt2" / "vocab.json"
        merges_path = self.data_dir / "gpt2" / "merges.txt"
        _init_bpe_worker(str(vocab_path), str(merges_path))

        self._cornell_lines_map: Dict[str, str] = {}
        self._cornell_conv_cursor: int = 0
        self._load_cornell_lines_index()

    def _encode(self, text: str) -> List[int]:
        assert _gpt2_mod._WORKER_VOCAB is not None and _gpt2_mod._WORKER_MERGES is not None
        ids: List[int] = []
        for p in _GPT2_PAT.findall(" " + text.strip()):
            if p.isascii():
                ids.extend(
                    _bpe_encode_piece(
                        p,
                        _gpt2_mod._WORKER_VOCAB,
                        _gpt2_mod._WORKER_MERGES,
                        _gpt2_mod._WORKER_CACHE,
                    )
                )
        return ids

    def _load_cornell_lines_index(self) -> None:
        pq_path = self.data_dir / "cornell" / "data" / "movie_lines-00000-of-00001.parquet"
        if not pq_path.exists():
            return
        pf = pq.ParquetFile(pq_path, memory_map=True)
        for b in pf.iter_batches(batch_size=50_000, columns=["lineID", "utterance"]):
            d = b.to_pydict()
            for lid, utt in zip(d["lineID"], d["utterance"]):
                if lid and utt:
                    # Очищаем реплику без имени персонажа, чтобы в ответах были только чистые мысли!
                    clean_u = " ".join(str(utt).strip().split())
                    if len(clean_u) >= 12:
                        self._cornell_lines_map[str(lid).strip()] = clean_u

    def stream_paired_dialogues(self, n_pairs: int = 2500) -> List[StimulusResponsePair]:
        """Потоково собирает `n_pairs` чистых пар `(Stimulus A -> Response B)`."""
        pairs: List[StimulusResponsePair] = []

        # 1. Если передан внешний датасет диалогов (--extra-dialog-path), сначала читаем из него
        if self.extra_dialog_path is not None and self.extra_dialog_path.exists():
            ext_p = self.extra_dialog_path
            if ext_p.suffix == ".jsonl":
                with ext_p.open("r", encoding="utf-8", errors="ignore") as f:
                    for idx, ln in enumerate(f):
                        if len(pairs) >= n_pairs // 2:
                            break
                        obj = json.loads(ln)
                        sa = str(obj.get("prompt") or obj.get("stimulus") or obj.get("question") or obj.get("input") or "").strip()
                        sb = str(obj.get("response") or obj.get("answer") or obj.get("output") or obj.get("target") or "").strip()
                        if sa and sb:
                            ids_a, ids_b = self._encode(sa), self._encode(sb)
                            if len(ids_a) >= 4 and len(ids_b) >= 4:
                                pairs.append(StimulusResponsePair(f"extra_{idx:06d}", "cornell_dialogs", sa, sb, ids_a, ids_b))

        # 2. Читаем реальные разговорные пары (A -> B) из movie_conversations.txt (129,955 пар)
        conv_file = self.data_dir / "cornell" / "movie_conversations.txt"
        if conv_file.exists():
            with conv_file.open("r", encoding="latin-1", errors="ignore") as f:
                for line_idx, ln in enumerate(f):
                    if line_idx < self._cornell_conv_cursor:
                        continue
                    self._cornell_conv_cursor = line_idx + 1
                    parts = ln.strip().split(" +++$+++ ")
                    if len(parts) < 4:
                        continue
                    try:
                        lids = ast.literal_eval(parts[3])
                    except Exception:
                        continue
                    for a_id, b_id in zip(lids, lids[1:]):
                        ua = self._cornell_lines_map.get(str(a_id))
                        ub = self._cornell_lines_map.get(str(b_id))
                        if not ua or not ub:
                            continue
                        wa, wb = ua.split(), ub.split()
                        if 5 <= len(wa) <= 36 and 6 <= len(wb) <= 38:
                            ids_a = self._encode(ua)
                            ids_b = self._encode(ub)
                            if len(ids_a) >= 5 and len(ids_b) >= 6:
                                pairs.append(
                                    StimulusResponsePair(
                                        pair_uid=f"cornell_pair_{a_id}_{b_id}",
                                        domain="cornell_dialogs",
                                        stimulus_text=ua,
                                        response_text=ub,
                                        stimulus_ids=ids_a,
                                        response_ids=ids_b,
                                    )
                                )
                                if len(pairs) >= n_pairs:
                                    return pairs
        return pairs

