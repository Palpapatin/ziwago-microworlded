"""Структуры данных диалоговых корпусов (слой app/data, ARCHITECTURE_MAPPING §7).

Выделено из монолита v14 (`train_ziwago_v14_pure_synaptic_dialogue.py`), этап 4c.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class TokenizedDialogPair:
    source_dataset: str
    stimulus_ids: List[int]
    response_ids: List[int]
    quality_reward_hint: float = 1.0
    split: str = "train"