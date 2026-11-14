"""Ролевая карта должна накапливаться из обучающего корпуса (Шаг 2, 2026-11-13).

До 2026-11-13 `sync_emergent_ontology_into_role_binder` не вызывался ни разу
в production-коде: `dynamic_stdp_role_map` оставалась пустой, живые пути
`extract_slot_bindings` всегда возвращали `{}`, и перенос каузальной рамы на
новый заполнитель (`mayonnaise → hollandaise`) был инертен.

Тест проверяет цепочку: обучающая пара → схема онтологии → STDP-поля слов →
Закон Невязки Схемы → ролевая связь → подстановка заполнителя биндером.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ziwago.cortex.genesis_engine import EmergentOntologicalRoleDiscovery
from ziwago.cognition.memory.role_filler import ThetaGammaRoleFillerBinder


def _ontology() -> EmergentOntologicalRoleDiscovery:
    return EmergentOntologicalRoleDiscovery(n_bins=512)


def test_role_binding_is_discovered_from_training_corpus_not_from_prompt():
    ontology = _ontology()
    binder = ThetaGammaRoleFillerBinder()

    # 1. Обучающая пара со сквозным ядром схемы — `mayonnaise`.
    disc = ontology.register_training_pair(
        "I am making mayonnaise and it has become runny",
        "Add more mayonnaise to thicken the sauce")
    assert ontology.schema_records, "обучающая пара должна зарегистрировать схему"
    assert binder.dynamic_stdp_role_map == {}, "до обучения карта пуста"

    # 2. Второе наблюдение мира: заполнитель ВИДЕН, но схемы на него нет
    #    (в ответе слово не повторяется — общих ядер нет). Именно такие слова
    #    Закон Невязки Схемы объявляет кандидатами в заполнители.
    ontology.register_training_pair(
        "I am making hollandaise and it has become runny",
        "Add more egg yolk to thicken the sauce")

    # 3. Новая реплика с новым заполнителем: связь открывается из корпуса.
    discovery = ontology.discover_emergent_role_bindings(
        "I am making hollandaise and it has become runny")
    assert discovery, "из корпуса должна открыться связь с новым заполнителем"
    slot, (filler, similarity) = next(iter(discovery.items()))
    assert slot == "mayonnaise"
    assert filler == "hollandaise"
    assert similarity >= 0.34

    # 4. Связь попадает в карту и используется живым биндером.
    binder.register_emergent_stdp_bindings(discovery)
    assert binder.dynamic_stdp_role_map["hollandaise"] == "mayonnaise"
    bindings = binder.extract_slot_bindings("please help with the hollandaise sauce")
    assert bindings == {"mayonnaise": "hollandaise"}

    # 5. Подстановка заполнителя в ответ (резервная проверка многословных форм).
    rewritten = binder.rewrite_utterance_to_canonical_for_ca3(
        "please help with the hollandaise sauce", bindings)
    assert "mayonnaise" in rewritten
    assert "hollandaise" not in rewritten


def test_role_map_stays_empty_without_training_corpus():
    """Без обучения карта пуста — никакой подстановки «из воздуха»."""
    ontology = _ontology()
    binder = ThetaGammaRoleFillerBinder()
    assert binder.dynamic_stdp_role_map == {}
    assert binder.extract_slot_bindings("please help with the hollandaise sauce") == {}
    assert ontology.schema_records == []


def test_schema_records_are_bounded():
    ontology = _ontology()
    ontology.max_schema_records = 8
    for index in range(40):
        ontology.register_training_pair(
            f"problem number {index} needs careful mayonnaise handling",
            f"answer number {index} explains mayonnaise handling carefully")
    assert len(ontology.schema_records) <= 8
