"""Тета-гамма связыватель синтаксических ролей и переменных слотов (BA44/BA46).

Выделен из монолита v14 (`train_ziwago_v14_pure_synaptic_dialogue.py`) в слой
cognition/memory (ARCHITECTURE_MAPPING §7, этап 4c). Поведение сохранено дословно.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Tuple

import torch


class ThetaGammaRoleFillerBinder:
    """Тета-Гамма Связыватель Синтаксических Ролей и Переменных Слотов (`BA44/BA46 Theta-Frame x BA22 Gamma-Filler`):
      - В коре человека медленная тета-волна (4–8 Гц) из префронтальной коры и зоны Брока удерживает
        абстрактную причинно-следственную раму (`Role / Schema`), а быстрые гамма-осцилляции (40–80 Гц)
        подставляют в слоты этой рамы конкретные объекты из текущей реплики пользователя (`Filler`).
      - Позволяет модели переносить выученный каузальный закон на НОВЫЕ объекты собеседника
        (например, `hollandaise` / `vinaigrette` вместо `mayonnaise`, `mustard` вместо `egg yolk`,
        `manuscript` / `thesis` / `dissertation` вместо `essay`, `cycling` / `hiking` вместо `running`)!"""

    # Compatibility names only: unknown-role transfer must come from learned bindings.
    CANONICAL_ROLE_FAMILIES: Dict[str, Tuple[str, ...]] = {}
    SLOT_DOMAIN_CONTEXT_MARKERS: Dict[str, Tuple[str, ...]] = {}

    def __init__(self) -> None:
        # Единый динамический банк ролевых аналогий, самоорганизующийся из латеральной STDP-коры (EmergentOntologicalRoleDiscovery):
        self.dynamic_stdp_role_map: Dict[str, str] = {}
        self.role_dictionary: Dict[str, Dict[str, torch.Tensor]] = {}

    def build_role_dictionary(self, encode_fn: Callable[[str], torch.Tensor | List[int] | Tuple[int, ...]],
                             d_vec: int = 64) -> Dict[str, Dict[str, torch.Tensor]]:
        """Build the live `role -> {label -> vector}` dictionary from the learned filler map.

        `dynamic_stdp_role_map` stores `filler -> canonical_role`; this method reverses
        the direction and encodes each filler into the same vector space as the composer.
        """
        from ziwago.cognition.language.content_form import ContentFormComposer

        vocab = ContentFormComposer.build_role_dictionary_from_bindings(
            self.dynamic_stdp_role_map,
            encode_fn,
            d_vec=d_vec,
        )
        self.role_dictionary = vocab
        return vocab

    def export_state_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {"dynamic_stdp_role_map": dict(self.dynamic_stdp_role_map)}
        payload["role_dictionary"] = {
            role: {label: vec.detach().cpu().tolist() for label, vec in slots.items()}
            for role, slots in self.role_dictionary.items()
        }
        return payload

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if isinstance(state.get("dynamic_stdp_role_map"), dict):
            self.dynamic_stdp_role_map = {str(k): str(v) for k, v in state["dynamic_stdp_role_map"].items()}
        if isinstance(state.get("role_dictionary"), dict):
            loaded: Dict[str, Dict[str, torch.Tensor]] = {}
            for role, slots in state["role_dictionary"].items():
                if not isinstance(slots, dict):
                    continue
                loaded[str(role)] = {str(label): torch.as_tensor(vec, dtype=torch.float32)
                                    for label, vec in slots.items()}
            self.role_dictionary = loaded

    def register_emergent_stdp_bindings(self, discovered_map: Dict[str, Tuple[str, float]]) -> int:
        """Сливает самоорганизовавшиеся STDP-аналогии (`canonical_slot -> (novel_filler, sim)`) напрямую в активный ролевой биндер."""
        added = 0
        for canon_slot, (novel_word, sim_val) in discovered_map.items():
            if float(sim_val) >= 0.25 and novel_word and novel_word != canon_slot:
                self.dynamic_stdp_role_map[novel_word.lower()] = canon_slot.lower()
                added += 1
        if added:
            self.role_dictionary = {}
        return added

    def _has_domain_context_for_slot(self, canonical_slot: str, words_set: set[str], low_utt: str) -> bool:
        """Предохранитель A5 (Context-Gated Role x Filler Binding):
        Запрещает внеконтекстные подстановки в `Zone 1` (например, `'paper on quantum gravity'` -> `'essay'`),
        требуя наличия доменных маркеров соответствующей схемы в реплике; открытые концептуальные запросы
        (`'What do you know about treatise?'`) направляются в `Zone 2 (THINK: 'I know X as Y')`."""
        ctx_markers = self.SLOT_DOMAIN_CONTEXT_MARKERS.get(canonical_slot, ())
        if not ctx_markers:
            return True
        return any(m in words_set or (" " in m and m in low_utt) for m in ctx_markers)

    def extract_slot_bindings(self, utterance: str) -> Dict[str, str]:
        """Извлекает из реплики пользователя отображения `canonical_slot -> user_filler`,
        используя самоорганизовавшиеся STDP-аналогии (`dynamic_stdp_role_map`), семантические семейства ролей
        и открытые синтаксические шаблоны при строгом соблюдении доменного контекста (A5)."""
        low = utterance.lower().strip()
        p_strip = " .,!?;:'()[]{}" + chr(34)
        words_clean = [w.strip(p_strip) for w in utterance.split() if w.strip(p_strip)]
        words_low = [w.lower() for w in words_clean]
        bindings: Dict[str, str] = {}

        # 0. Приоритетная проверка по самоорганизовавшимся STDP-ролям L2/3 BA22 (Unified STDP Ontology Fusion):
        words_set = set(words_low)
        for novel_fil, canon_slot in self.dynamic_stdp_role_map.items():
            if canon_slot not in low and novel_fil in words_set and self._has_domain_context_for_slot(canon_slot, words_set, low):
                bindings[canon_slot] = novel_fil

        # Multiword fillers are learned too; no family or benchmark-specific inference.
        for novel_fil, canon_slot in self.dynamic_stdp_role_map.items():
            if " " in novel_fil and novel_fil in low and canon_slot not in low:
                bindings[canon_slot] = novel_fil
        return bindings

    def rewrite_utterance_to_canonical_for_ca3(self, utterance: str, bindings: Dict[str, str]) -> str:
        """Проецирует пользовательские заполнители (`user_filler`) обратно в канонические ролевые маркеры (`canonical_slot`)
        при поиске полюса в `CA3`, чтобы гиппокамп узнавал каузальную раму независимо от конкретного существительного."""
        if not bindings:
            return utterance
        rewritten = utterance
        for canonical_slot, user_filler in sorted(bindings.items(), key=lambda item: -len(item[1])):
            rewritten = re.sub(r"(?<!\w)" + re.escape(user_filler) + r"(?!\w)",
                               lambda _: canonical_slot, rewritten, flags=re.I)
        return rewritten

    def build_arcuate_token_pointer_map(
        self,
        bindings: Dict[str, str],
        encode_fn,
    ) -> Dict[int, Tuple[int, List[int]]]:
        """Дугообразный пучок (`Arcuate Fasciculus Synaptic Pointer`, `BA22 Wernicke -> BA44 Broca`):
        Строит прямую синаптическую проекцию на уровне BPE Token ID `canonical_head_tid -> (canonical_span_len, filler_tids)`.
        Когда в пословном цикле генерации дендритная матрица возбуждает головной токен канонического слота `canonical_head_tid`,
        аксональный указатель напрямую копирует BPE-токены `filler_tids` пользователя в моторный буфер `out_ids`,
        удерживая внутренний рекуррентный след коры на канонической роли (БЕЗ строкового `.replace()`!)."""
        if not bindings:
            return {}
        pointer_map: Dict[int, Tuple[int, List[int]]] = {}
        for canonical_slot, user_filler in bindings.items():
            # BPE-токены с ведущим пробелом (" mayonnaise" -> " hollandaise") и с заглавной буквы
            for prefix, cap in ((" ", False), (" ", True), ("", True)):
                c_str = prefix + (canonical_slot.capitalize() if cap else canonical_slot)
                f_str = prefix + (user_filler.capitalize() if cap else user_filler)
                c_tids = encode_fn(c_str)
                f_tids = encode_fn(f_str)
                if c_tids and f_tids:
                    pointer_map[int(c_tids[0])] = (len(c_tids), [int(t) for t in f_tids])
        return pointer_map

    def apply_slot_bindings_to_response(self, response_text: str, bindings: Dict[str, str]) -> str:
        """Резервная проверка многословных артиклевых конструкций (например, 'a fresh egg yolk' -> 'fresh mustard')."""
        if not bindings or not response_text:
            return response_text
        out = response_text
        for canonical_slot, user_filler in bindings.items():
            out = out.replace(canonical_slot, user_filler)
            out = out.replace(canonical_slot.capitalize(), user_filler.capitalize())
        return out