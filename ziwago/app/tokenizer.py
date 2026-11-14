"""GPT-2 BPE токенизатор — фундаментальный слой ziwago/app (B.2).

Примитивы BPE (_GPT2_PAT / _bpe_encode_piece / _init_bpe_worker / _WORKER_* /
byte encoder-decoder) перенесены verbatim из train_ziwago_v9_gpt2vocab.py.
Единый владелец process-local worker-глобалов: _init_bpe_worker переприсваивает
их, потребители читают через модуль (import ziwago.app.tokenizer as _gpt2_mod).
train_ziwago_v9_gpt2vocab ре-экспортирует их (PEP 562 __getattr__) для совместимости.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import regex as re

_WORKER_VOCAB: Optional[Dict[str, int]] = None
_WORKER_MERGES: Optional[Dict[Tuple[str, str], int]] = None
_WORKER_CACHE: Dict[str, List[int]] = {}
_GPT2_PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def _bytes_to_unicode() -> Dict[int, str]:
    """GPT-2's reversible alphabet for all 256 UTF-8 byte values."""
    byte_values = list(range(ord("!"), ord("~") + 1)) + list(range(161, 173)) + list(range(174, 256))
    codepoints = byte_values.copy()
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, map(chr, codepoints)))


_GPT2_BYTE_ENCODER = _bytes_to_unicode()
_GPT2_BYTE_DECODER = {char: value for value, char in _GPT2_BYTE_ENCODER.items()}


def _init_bpe_worker(vocab_path: str, merges_path: str) -> None:
    """Инициализация воркера в пуле из 32 процессов: загружает `vocab.json` и `merges.txt`."""
    global _WORKER_VOCAB, _WORKER_MERGES, _WORKER_CACHE
    with open(vocab_path, "r", encoding="utf-8") as f:
        _WORKER_VOCAB = json.load(f)
    _WORKER_MERGES = {}
    with open(merges_path, "r", encoding="utf-8") as f:
        for rank, ln in enumerate(f):
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(" ")
            if len(parts) == 2:
                _WORKER_MERGES[(parts[0], parts[1])] = rank
    _WORKER_CACHE = {}


def _bpe_encode_piece(piece: str, vocab: Dict[str, int], merges: Dict[Tuple[str, str], int], cache: Dict[str, List[int]]) -> List[int]:
    """Сверхбыстрое GPT-2 BPE-кодирование одного слова/фрагмента с мемоизацией:
    1) если слово с префиксом `Ġ` уже есть в `vocab.json` (94% слов WikiText-103!) — возвращает его ID за O(1)!
    2) иначе выполняет стандартное слияние пар по рангам `merges.txt`."""
    if not piece:
        return []
    if piece in cache:
        return cache[piece]

    g_piece = "".join(_GPT2_BYTE_ENCODER[value] for value in piece.encode("utf-8"))
    # Tiny plain-ASCII vocabularies may use literal spaces, unlike GPT-2.
    if piece.isascii() and "Ġ" not in vocab and (piece in vocab or " " in vocab):
        g_piece = piece
    if g_piece in vocab:
        res = [vocab[g_piece]]
        cache[piece] = res
        return res

    # Разбиваем на символы и сливаем по дереву merges.txt
    word = tuple(g_piece)
    if len(word) == 1:
        res = [vocab[word[0]]]
        cache[piece] = res
        return res

    while len(word) > 1:
        best_pair = None
        best_rank = 1_000_000_000
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            rk = merges.get(pair, 1_000_000_000)
            if rk < best_rank:
                best_rank = rk
                best_pair = pair
        if best_pair is None or best_rank == 1_000_000_000:
            break
        first, second = best_pair
        new_word: List[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = tuple(new_word)

    ids = [vocab[tok] for tok in word]
    if len(cache) < 65536:
        cache[piece] = ids
    return ids
