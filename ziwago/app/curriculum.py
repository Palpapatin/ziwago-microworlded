"""Базовые онтогенетические уроки по эпохам (слой L4: данные обучения).

Вынесено из train_ziwago_v16_10B_omniscient_engine.py (#31, 2026-11-12):
чистый дата-литерал, единственный живой v16-символ плоского организма v20
(ziwago/organism_core.py). Не тянет за собой движок v16.
"""
from __future__ import annotations

from typing import Dict, List, Tuple


FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH: Dict[int, List[Tuple[str, str, str, float]]] = {
    1: [
        (
            "tinystories",
            "Once upon a time, a curious little fox found a shiny brass key near the old oak tree.",
            "The little fox used the brass key to unlock the hidden wooden chest and shared the golden berries with her friends.",
            2.3,
        ),
    ],
    2: [
        (
            "bookcorpus",
            "The autumn wind swept across the cobblestone courtyard as the cathedral bells began to chime at dusk.",
            "Shadows lengthened beneath the stone arches while the old scholar closed his leather-bound journal and stepped into the lantern light.",
            2.3,
        ),
    ],
    3: [
        (
            "wikitext103",
            "How do biological pyramidal neurons in the neocortex integrate bottom-up sensory input with top-down predictions?",
            "Pyramidal neurons receive bottom-up sensory spikes on basal dendrites and top-down cortical predictions on apical tuft NMDA compartments.",
            2.4,
        ),
    ],
    4: [
        (
            "npc_dialogue",
            "Greetings, guardian of the ancient stone arch.",
            "Ah, you have arrived at the Echoing Glade, traveler; I have been waiting for one who seeks the ancient knowledge.",
            2.3,
        ),
        (
            "personachat",
            "Hi, how are you doing? I am getting ready to go running to stay in shape.",
            "That sounds great! Staying active is a wonderful habit, and running outdoors is one of my favorite ways to clear the mind.",
            2.3,
        ),
    ],
    5: [
        (
            "oasst2",
            "I am making mayonnaise, it was starting to thicken but now it has become runny and liquid.",
            "Yes, it is possible to fix runny mayonnaise by slowly whisking the mixture into a fresh egg yolk until the emulsion stabilizes.",
            2.3,
        ),
        (
            "ultrachat",
            "How can I fix errors in my grammar when writing an essay?",
            "To fix grammar errors in your essay, read each sentence aloud, check subject-verb agreement, and simplify overly long clauses.",
            2.3,
        ),
    ],
}
