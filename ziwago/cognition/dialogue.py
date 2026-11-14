"""Явное состояние диалога и эпизодический граф рабочей памяти (BA46).

Выделен из монолита v14 (`train_ziwago_v14_pure_synaptic_dialogue.py`) в слой
cognition (ARCHITECTURE_MAPPING §7, этап 4c). Поведение сохранено дословно.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class ConversationalDialogueState:
    """Явное состояние диалога, эпизодический граф рабочей памяти и маршрутизатор доменов
    (интеграция `ziwago_v12_conversational_mind.py` + `EpisodicDialogueGraph` v17):
      1) Отслеживает вектор истории диалога `dialogue_history_vec64`, текущий домен реплики и факты о собеседнике
         (`user_name`, `identity_pole_vec`), чтобы вопросы типа `'what is my name'` или `'what is his name'`
         разрешались через префронтальную рабочую память без захвата случайными полюсами вроде `'Echoing Glade'`.
      2) Ведёт эпизодический граф всех ходов беседы (`episodic_turns`, до 128 ходов), позволяя отвечать
         на мета-вопросы о самом разговоре (`'What did we discuss in our previous turn?'`, `'Summarize our conversation'`).
      3) Хранит историю недавно активированных полюсов `CA3` (`recent_pole_vectors`) для латерального
         габитуационного торможения (Habituation), предотвращая повтор одного и того же полюса на нерелевантных ходах."""

    DOMAIN_MARKERS: Dict[str, Tuple[str, ...]] = {
        "self_identity": ("my name is", "what is my name", "who am i", "what am i building", "what project am i", "call me"),
        "culinary_emulsion": (
            "mayonnaise", "hollandaise", "vinaigrette", "aioli", "egg", "yolk", "mustard",
            "emulsion", "whisk", "runny", "sauce", "liquid",
        ),
        "essay_grammar": (
            "grammar", "essay", "manuscript", "thesis", "dissertation", "sentence",
            "clause", "subject-verb", "writing", "revising", "errors",
        ),
        "fitness_lifestyle": (
            "running", "cycling", "hiking", "swimming", "jogging", "outdoors",
            "stay in shape", "active", "habit", "clear the mind", "clear my mind",
        ),
        "neuroscience": ("pyramidal", "neurons", "neocortex", "dendrites", "apical", "basal", "nmda", "synapto", "predictions"),
        "fantasy_lore": ("guardian", "stone arch", "ancient", "echoing glade", "traveler"),
    }

    def __init__(self, d_vec: int = 64):
        self.d_vec = int(d_vec)
        self.turn_count: int = 0
        self.user_name: Optional[str] = None
        self.identity_pole_vec: Optional[torch.Tensor] = None
        self.dialogue_history_vec64: torch.Tensor = torch.zeros(self.d_vec, dtype=torch.float32)
        self.active_domain: str = "general_dialogue"
        self.recent_pole_vectors: List[torch.Tensor] = []
        self.episodic_turns: List[Dict[str, object]] = []

    def classify_utterance_domains(self, utterance: str) -> List[str]:
        low = utterance.lower()
        matched: List[str] = []
        for dom, markers in self.DOMAIN_MARKERS.items():
            if any(m in low for m in markers):
                matched.append(dom)
        if not matched and self.user_name and self.user_name.lower() in low and ("name" in low or "building" in low):
            matched.append("self_identity")
        return matched or ["general_dialogue"]

    def is_identity_query(self, utterance: str) -> bool:
        low = utterance.lower()
        identity_patterns = (
            "what is my name",
            "what my name is",
            "remember my name",
            "who am i",
            "what am i building",
            "what i am building",
            "what am i designing",
            "what i am designing",
            "what project am i",
            "what is his name",
            "what his name is",
            "what is her name",
            "what her name is",
        )
        if any(p in low for p in identity_patterns):
            return True
        if self.user_name and self.user_name.lower() in low and (
            "his name" in low or "her name" in low or "my name" in low
        ):
            return True
        return False

    def is_episodic_history_query(self, utterance: str) -> bool:
        low = utterance.lower()
        episodic_markers = (
            "what did we discuss",
            "what have we discussed",
            "what topics have we",
            "what topics we have",
            "talked about in our",
            "in our previous turn",
            "what did i ask you earlier",
            "summarize our conversation",
            "summarize what topics",
            "our dialogue so far",
            "our conversation so far",
            "our chat today",
        )
        return any(m in low for m in episodic_markers)

    def reset_session(self) -> None:
        """Сбрасывает оперативную сессию диалога при старте нового интерактивного чата, чтобы имя из тренировочного
        чекпоинта (`Alex`) не протекало в разговор с новым пользователем."""
        self.turn_count = 0
        self.user_name = None
        self.identity_pole_vec = None
        self.dialogue_history_vec64.zero_()
        self.active_domain = "general_dialogue"
        self.recent_pole_vectors = []
        self.episodic_turns = []

    def synthesize_episodic_memory_response(self, utterance: str) -> str:
        """Синтезирует ответ из эпизодического графа рабочей памяти `BA46` ТОЛЬКО по заземлённым выводам (`A3`:
        ungrounded-ответы и уточняющие запросы `Clarify` никогда не попадают в сводку как выводы!)."""
        grounded_turns = [
            ep for ep in self.episodic_turns
            if bool(ep.get("is_grounded_conclusion", True)) and not bool(ep.get("was_clarification_request", False))
        ]
        if not grounded_turns and not self.episodic_turns:
            return "We have just started our conversation, and no prior turns are recorded in working memory yet."
        low = utterance.lower()
        if "previous turn" in low or "last turn" in low:
            last_ep = grounded_turns[-1] if grounded_turns else self.episodic_turns[-1]
            prev_dom = str(last_ep.get("domain", "general_dialogue")).replace("_", " ")
            prev_resp = str(last_ep.get("response", "")).strip()
            return f"In our previous turn on {prev_dom}, we discussed that {prev_resp[0].lower() + prev_resp[1:] if len(prev_resp) > 1 else prev_resp}"
        domains_seen: List[str] = []
        topics_seen: List[str] = []
        for ep in grounded_turns:
            d_str = str(ep.get("domain", "")).replace("_", " ")
            if d_str and d_str not in domains_seen:
                domains_seen.append(d_str)
            for t_tag in ep.get("topic_tags", []):  # type: ignore[union-attr]
                if t_tag and t_tag not in topics_seen and len(topics_seen) < 6:
                    topics_seen.append(str(t_tag))
        if not domains_seen:
            domains_seen = ["epistemic exploration"]
        name_part = f"With {self.user_name}, we" if self.user_name else "We"
        topics_clause = f" (exploring {', '.join(topics_seen[:5])})" if topics_seen else ""
        if grounded_turns:
            last_resp = str(grounded_turns[-1].get("response", "")).strip()
            return (
                f"{name_part} have completed {len(grounded_turns)} grounded dialogue turns covering "
                f"{', '.join(domains_seen)}{topics_clause}, most recently concluding that "
                f"{last_resp[0].lower() + last_resp[1:] if len(last_resp) > 1 else last_resp}"
            )
        return (
            f"{name_part} have explored {', '.join(domains_seen)}{topics_clause}, "
            f"and no verified conclusions have been consolidated into working memory yet."
        )

    def register_user_identity(self, utterance: str, pole_vec: torch.Tensor) -> None:
        p_strip = " .,!?;:'()[]{}" + chr(34)
        words = utterance.split()
        for i, w in enumerate(words):
            if w.strip(p_strip).lower() == "name" and i + 2 < len(words) and words[i + 1].strip(p_strip).lower() == "is":
                cand = words[i + 2].strip(p_strip)
                if cand:
                    self.user_name = cand
                    break
        self.identity_pole_vec = F.normalize(pole_vec.detach().clone(), p=2, dim=0)

    def record_turn_and_poles(
        self,
        stimulus_vec64: torch.Tensor,
        activated_poles: List[torch.Tensor],
        utterance: str,
        response_text: str = "",
        role_filler_bindings: Optional[Dict[str, str]] = None,
        topic_tags: Optional[List[str]] = None,
        was_clarification_request: bool = False,
        is_grounded_conclusion: bool = True,
        selected_ca3_pole_indices: Optional[List[int]] = None,
        response_passport: Optional[Dict[str, object]] = None,
    ) -> None:
        """Предохранитель A3:
        Запрещает запись `ungrounded`-ответов в эпизодическую память и в буфер `recent_pole_vectors` (для REM/SWR реплея).
        Уточняющие ходы (`was_clarification_request=True`) сохраняются только как маркер ожидания ответа собеседника
        с `is_grounded_conclusion=False`, никогда не попадая в итоговую сводку выводов."""
        if not is_grounded_conclusion and not was_clarification_request:
            return
        self.turn_count += 1
        doms = self.classify_utterance_domains(utterance)
        self.active_domain = doms[0]
        if is_grounded_conclusion and not was_clarification_request:
            if self.turn_count == 1 or float(self.dialogue_history_vec64.norm().item()) < 1e-6:
                self.dialogue_history_vec64 = F.normalize(stimulus_vec64.detach().clone(), p=2, dim=0)
            else:
                self.dialogue_history_vec64 = F.normalize(
                    0.70 * self.dialogue_history_vec64 + 0.30 * stimulus_vec64.detach(), p=2, dim=0
                )
            for p in activated_poles:
                self.recent_pole_vectors.append(F.normalize(p.detach().clone(), p=2, dim=0))
            if len(self.recent_pole_vectors) > 6:
                self.recent_pole_vectors = self.recent_pole_vectors[-6:]

        if topic_tags is None:
            p_strip = " .,!?;:'()[]{}" + chr(34)
            stop_t = {"what", "how", "why", "where", "when", "who", "does", "do", "is", "are", "the", "a", "an", "in", "of", "to", "with", "can", "you", "my", "your"}
            topic_tags = [
                w.strip(p_strip).lower()
                for w in utterance.split()
                if len(w.strip(p_strip)) >= 4 and w.strip(p_strip).lower() not in stop_t
            ][:4]

        self.episodic_turns.append(
            {
                "turn_id": self.turn_count,
                "domain": self.active_domain,
                "domains": doms,
                "utterance": utterance,
                "response": response_text if (is_grounded_conclusion and not was_clarification_request) else "",
                "role_filler_bindings": dict(role_filler_bindings) if role_filler_bindings else {},
                "topic_tags": list(topic_tags),
                "was_clarification_request": bool(was_clarification_request),
                "is_grounded_conclusion": bool(is_grounded_conclusion and not was_clarification_request),
                "selected_ca3_pole_indices": list(selected_ca3_pole_indices) if selected_ca3_pole_indices else [],
                "response_passport": dict(response_passport) if response_passport else {},
            }
        )
        if len(self.episodic_turns) > 128:
            self.episodic_turns = self.episodic_turns[-128:]

    def state_dict(self) -> Dict[str, object]:
        return {
            "turn_count": self.turn_count,
            "user_name": self.user_name,
            "identity_pole_vec": self.identity_pole_vec,
            "dialogue_history_vec64": self.dialogue_history_vec64,
            "active_domain": self.active_domain,
            "recent_pole_vectors": self.recent_pole_vectors,
            "episodic_turns": list(self.episodic_turns),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.turn_count = int(state.get("turn_count", 0))  # type: ignore[arg-type]
        self.user_name = state.get("user_name")  # type: ignore[assignment]
        id_vec = state.get("identity_pole_vec")
        self.identity_pole_vec = id_vec.float() if isinstance(id_vec, torch.Tensor) else None
        hist_vec = state.get("dialogue_history_vec64")
        if isinstance(hist_vec, torch.Tensor):
            self.dialogue_history_vec64 = hist_vec.float()
        self.active_domain = str(state.get("active_domain", "general_dialogue"))
        rec = state.get("recent_pole_vectors", [])
        if isinstance(rec, list):
            self.recent_pole_vectors = [t.float() for t in rec if isinstance(t, torch.Tensor)]
        eps = state.get("episodic_turns", [])
        if isinstance(eps, list):
            self.episodic_turns = [dict(e) for e in eps if isinstance(e, dict)]