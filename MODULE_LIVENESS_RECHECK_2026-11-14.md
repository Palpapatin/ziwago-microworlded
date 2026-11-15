# Перепроверка живых модулей Ziwago18 — 2026-11-14

## Итог и границы доказательств

Это свежая проверка текущего production-контура по исходникам, AST-карте
конструирования и точечным тестам. Это **не** полное динамическое измерение
обученного кандидата: `tools/probes/module_map_probe.py` требует `--checkpoint`,
а в прикреплённом worktree не найден чекпоинт-кандидат или каталог `Produce`.
Также доступной RAM недостаточно для требуемой проектом полной загрузки
чекпоинта (ориентир — не менее 10 ГБ свободной памяти). Проба намеренно не
запускалась без чекпоинта или с непроверенными весами.

Поэтому:

* ниже приведены только механизмы с прослеживаемыми владельцем и точкой
  production-вызова либо с подтверждённым отсутствием такой точки в активном
  v20-контуре;
* наличие вызова/телеметрии отдельно от причинного эффекта не считается
  доказательством SHIPPED;
* тесты на `tiny_core` доказывают локальные контракты, но не качество поведения
  на обученном чекпоинте;
* **BLOCKED** — не новый статус жизненного цикла, а пометка, что требуемый
  динамический вывод нельзя сделать из имеющихся доказательств.

Статусы для этой перепроверки:

* **SHIPPED** — владелец, вызов в активном production-пути, измеряемый эффект и
  тестовый причинный контроль (on/off или эквивалентное контролируемое
  возмущение) подтверждены для названного узкого механизма.
* **PHANTOM** — владелец и/или production-вызов есть, но эффект/контроль не
  доказан; сюда же относится созданный, но не вызываемый механизм.
* **OFF** — существует явный выключатель, и проверено, что соответствующий
  consumer отключён по текущей конфигурации.
* **DEAD** — у указанного механизма нет вызова из текущего активного
  production-пути. Это не означает, что файл или класс можно удалить.

## Срез механизмов

| Механизм и фаза | Владелец → production-вызов | Проверенный эффект и on/off-контроль | Статус |
|---|---|---|---|
| Интегрированное обучение, физиологический гейт (обучение/сон) | `ZiwagoOrganism.enable_integrated_organism()` создаёт `IntegratedOrganism` (`ziwago/organism/flat.py:834-850`); training-path вызывает `record_credit()` (`flat.py:1484-1489`); тики проходят через `IntegratedOrganism.tick()` (`ziwago/organism/integrated.py:235-292`). | `tests/test_v20_integration.py::test_training_writes_weights_and_replay_credit_without_ngrams` проверяет обновление обучения и кредит реплея; `::test_sleep_closes_plasticity_and_replays_tagged_cues` подтверждает закрытие пластичности во сне и последующий реплей. Малый core, не обученный кандидат. | **SHIPPED** — только этот проверенный training/sleep-gate контракт. |
| Критичность → гомеостатический регулятор (живой тик) | Monitor и `HomeostaticController` создаются в `IntegratedOrganism` (`ziwago/organism/integrated.py:159-174`); вызовы измерения, регуляции и применения relay gain — `integrated.py:269-277`, `:342-359`. | `tests/test_criticality_control.py::CausalRelayTests::test_gain_scales_relay_causally` сравнивает relay при gain 1.0/1.4; `::test_enabled_unit_gain_is_bit_identical_to_off` — enabled/off контроль; `LiveMeasurementTests` подтверждают путь измерения к controller. | **SHIPPED** — локальный механизм на tiny core; не оценка эффекта на чекпоинте. |
| MetaHead → `replay_priority` (обучение → последующий сон/replay) | `MetaHead.step()` вызывается в тике, выход стадируется; кредит записывается на training-path (`ziwago/organism/integrated.py:319-325`, `:586`; `ziwago/organism/flat.py:1488-1489`). | `tests/test_v22_meta_replay_priority.py` проверяет `present() → record_credit()` с задержкой в тик при `ticks_per_token=1` и `4`, раздельное сохранение базового и MetaHead приоритета, а также изменение следующего replay-выбора on/off (`::test_meta_priority_is_consumed_by_replay_credit_after_one_tick`, `::test_present_credit_delay_changes_the_next_replay_selection`). | **SHIPPED** — только канал `replay_priority`; это не доказывает влияние остальных выходов MetaHead. |
| Metaplastic shield: локальное ограничение обучения и sleep-decay трассы | Shield принадлежит readout; в конце NREM→REM вызывается `consolidate_into_slow_reservoir()` (`ziwago/organism_core.py:671-673`). Реализация уменьшает `u_slow` и считает долю активных весов выше порога, а не переносит веса (`ziwago/cortex/next16_core.py:547-567`). | `tests/test_v19_research_modules.py::test_shield_sleep_decay_releases_unused_protection` измеряет спад `u_slow` и восстановление attenuation; `::test_shield_roundtrip_synthetic_retention_and_validation` сравнивает защищённый и незащищённый сдвиг весов. Здесь проверены decay/защита, не перенос в «медленный резервуар». | **SHIPPED** — узко для трассы защиты/её decay. Название метода не является доказательством передачи знаний. |
| Гиппокамп: production sleep-cycle / SWR replay | `life.hippo.sleep_cycle()` вызывается на NREM→REM (`ziwago/organism_core.py:664-670`); `Hippocampus.sleep_cycle()` требует callback записи для SWR (`ziwago/memory/hippocampus.py:749-785`). Production-вызов callback не передаёт (`organism_core.py:667-670`). | Условия SWR и причина `no_cortical_write` проверяемы (`hippocampus.py:771-785`); тесты `tests/test_v22_memory_honesty.py::test_sleep_cycle_swr_requires_adenosine_gate` и `::test_sleep_cycle_rem_runs_only_with_enough_episodes` вызывают метод с тестовыми аргументами. Это не production on/off-доказательство. При текущем вызове SWR-запись не происходит из-за `cortical_write=None`. | **PHANTOM** — sleep-вызов есть, но production SWR-эффекта нет; REM-рекомбинация не равна корковому replay. |
| REM dreamer (сон/консолидация графов) | Dreamer создаётся `ZiwagoOrganism`; `_rem_dream()` регистрирует графы и вызывается на NREM→REM (`ziwago/organism_core.py:353-420`, `:674`). | Результат попадает в `sleep_report["rem"]`, и отдельно записано `rem_facts_promoted=False` (`organism_core.py:674-675`). В этой проверке нет причинного контроля, что REM меняет память/веса или последующую генерацию. | **PHANTOM** — вызов установлен, эффект для организма не доказан. |
| Активный SWS в интегрированном контуре | При NREM→REM активный callback масштабирует веса `w_dend_out[:, :active]` коэффициентом `0.992` (`ziwago/organism_core.py:649-658`). | Отчёт фиксирует нормы до/после, но это телеметрия одного вызова; в проверенных тестах нет достаточного production on/off-контроля, связывающего downscale с последующим качеством памяти/генерации. | **PHANTOM** — операция вызывается, причинный эффект на целевой исход не установлен. |
| `execute_sws_synaptic_homeostasis_and_pruning` / старый модуль сна | Импорт и вызов находятся внутри `_sleep_cycle_legacy()` (`ziwago/organism_core.py:706-717`); активный `sleep_cycle()` вместо него использует `life.tick()` и фазовый callback (`organism_core.py:677-704`). Поиск production-вызовов не нашёл вызова legacy-метода. | Нет активного вызова и нет применимого production on/off-контроля для этого helper в unified v20-пути. | **DEAD** — только этот legacy-вызов в активном контуре; не удалять по одному AST-результату. |
| Защищённое ядро → базальные веса во сне | `w_core_dend_out` добавляется к `w_dend_out` только в `_sleep_cycle_legacy()` (`ziwago/organism_core.py:720-730`); активный `sleep_cycle()` этот код не вызывает. | Для активного v20 sleep-path измеренного переноса нет; метрика `slow_reservoir_pct` из другого метода не подтверждает core-to-basal transfer. | **DEAD** — перенос отсутствует в текущем активном sleep-пути. |
| BCM gate для явного обучения фактами (assistant teach-path) | Владелец — `ZiwagoOrganism.bcm_gate`; `teach_multi_sentence_episode()` вызывает `evaluate_neuromodulatory_gate()` и пропускает/блокирует обучение по `accepted` (`ziwago/organism_core.py:165`; `ziwago_assistant.py:94-115`). | В свежих проверенных тестах не найден причинный on/off-контроль, показывающий, что именно BCM меняет приемлемый результат/обучение при одинаковом входе. Checkpoint roundtrip не является таким контролем. | **PHANTOM** — production-вызов есть, эффект/контроль не подтверждён. |
| Larkum microcircuit в языковом контуре | Экземпляр создаётся и сериализуется (`ziwago/organism_core.py:165-166`, `:585-596`). Единственный найденный assistant call находится в ветке `if not hasattr(self.organism, "life")` (`ziwago_assistant.py:264-286`), тогда как текущий конструктор всегда включает integrated contour (`ziwago/organism/flat.py:518`, `:834-850`). | Свежий вызов в текущем production-контуре не обнаружен; тест `tests/test_v19_research_modules.py::test_bcm_and_larkum_checkpoint_resume_exactly` — только roundtrip состояния. | **PHANTOM** — владелец есть, live call и эффект не подтверждены. |
| ThalamusFO phrase-boundary marker (генерация) | FO принадлежит `IntegratedOrganism`; генератор вызывает `mark_phrase_boundary()` (`ziwago/organism/flat.py:1880-1883`), следующий тик проходит через `fo.step()` (`ziwago/organism/integrated.py:269-272`). | `tests/test_v22_phantom_consciousness.py::test_relay_reports_connected_phrase_boundary_and_unwired_spindle` подтверждает production call статически и флаг подключения, но не причинное отличие конечной генерации с/без маркера. | **PHANTOM** — проводка есть, пользовательский эффект пока не доказан. |
| ThalamusFO spindle | FO владеет `spindle()`, но `SPINDLE_CONNECTED=False` (`ziwago/thalamus/relay.py:44-49`); production-вызов не обнаружен. | Тест вызывает метод напрямую для формы/телеметрии, но нет production consumer. | **OFF** — consumer явно выключен; нет основания подключать без точки сна и causal-test. |
| MetaHead остальные gate-каналы (`bg_threshold`, `beta_gw_boost`, `trn_gain`) | `MetaHead` вызывается в live tick; только `replay_priority` проходит в очередь. Общий `GATES_CONNECTED=False` (`ziwago/organism/integrated.py:319-325`; `tests/test_v22_phantom_consciousness.py:26-35`). | Тесты контракта `apply_gates` не доказывают, что остальные гейты воздействуют на production consumers. Их live-потребители не установлены. | **PHANTOM** — не подключать по факту наличия вычисляемых выходов. |
| BA44 `learn_bridge_transition()` | Владелец моста конструируется и сам мост используется при выборе; production-вызов learner не найден (статическая проверка, отражённая в `GLOBAL_AUDIT_2026-11-14.md`, § P1). | Нет теста обучения на валидированных независимых парах с on/off сравнением. | **DEAD** — только production-обучение переходам; сам мост/выбор не классифицируется этим статусом. |

## Что в старой карте устарело

`MODULE_MAP_2026-11-13.md` и построенная из него
`MODULE_MAP_TABLE_2026-11-13.md` — не текущая карта. Их классификация выводила
SHIPPED преимущественно из числа вызовов, PHANTOM — из отсутствия вызова,
OFF — из старого условия конструктора, DEAD — из отсутствия конструктора.
Это не удовлетворяет нынешнему контракту «владелец + live call + измеримый
эффект + on/off-контроль».

Подтверждённые расхождения:

* текущий `flat.py` всегда включает `IntegratedOrganism`; integrated tick
  содержит вызовы TRN/FO, гиппокампа, MetaHead и гомеостаза. Следовательно,
  старые DEAD-метки для этих файлов/классов нельзя переносить в текущий статус;
* у конфигурации integrated contour есть включённые по умолчанию опции
  критичности и hippocampus (`ziwago/organism/integrated.py:41-53`), что
  противоречит утверждению старой карты, что всё выключено по умолчанию;
* phrase-boundary и hippocampus sleep-cycle уже имеют production-вызовы, но
  вызов сам по себе не поднимает их до SHIPPED;
* обратный пример: `execute_sws_synaptic_homeostasis_and_pruning` и
  core-to-basal перенос остались в `_sleep_cycle_legacy`, который не вызывается
  текущим активным `sleep_cycle()`.

## Выполненные проверки и блокировки

Каждый тестовый процесс запускался отдельно через `tools/limited_run.py` с
RAM/CPU cap; тяжёлое тестирование параллельно не выполнялось.

| Проверка | Результат |
|---|---|
| `tests/test_v20_integration.py` | 7 passed; max RSS 1.56 GiB. |
| `tests/test_v20_physiology.py` | 67 passed; max RSS 1.56 GiB. |
| `tests/test_v20_cortical_clock.py` | 24 passed; max RSS 1.56 GiB. |
| `tests/test_v20_organism_composition.py -k "generation_v20 or sleep_cycle_v20 or sleep_cycle_is_integrated"` | 3 passed, 7 deselected; max RSS 2.25 GiB. Тест `generation_v20` перехватывает исключения, поэтому этот PASS не доказывает успешную генерацию. |
| `tests/test_v22_consciousness_thalamus.py tests/test_v22_phantom_consciousness.py` | 39 passed; max RSS 1.57 GiB. |
| `tests/test_v22_memory_honesty.py` | 29 passed; max RSS 1.52 GiB. |
| `tests/test_criticality_control.py` | 16 passed; max RSS 1.53 GiB. |
| `tests/test_v22_astrocyte_wiring.py tests/test_v22_column_cost_conductances.py` | 7 passed; max RSS 1.52 GiB. |
| `tests/test_integrated_hippocampus.py` | 19 passed; max RSS 1.54 GiB. |
| Выборочная объединённая тяжёлая выборка тестов | Остановлена жёстким лимитом около 2.39 GiB; процесс завершился Windows access violation при создании организма. Это ресурсно заблокированный запуск, не утверждение о провале assertions. |
| `tools/probes/module_map_probe.py` | **Не запускалась**: обязателен `--checkpoint`; подходящий кандидат отсутствует в worktree. Полное динамическое покрытие training/chat/generation и фазы сна остаётся BLOCKED. |

В рамках первичной перепроверки production-код не менялся. Последующая задача
по MetaHead и энергии GW отражена ниже. Для окончательной полной карты всё ещё
необходимы обученный checkpoint и машина с достаточным свободным RAM; затем
нужно отдельно профильтровать обучение, генерацию, сон и консолидацию и
выполнить live on/off-контроль для каждого оставшегося PHANTOM.

## Последующая задача: MetaHead и энергия GW

В следующей задаче на `2026-11-14` разрывы для этих двух выходов обработаны без
подключения неподтверждённых гейтов:

* `MetaHead.telemetry()` теперь перечисляет статус и причину отдельно для каждого
  канала. В live-пути потребляется только `replay_priority`; для `bg_threshold`,
  `beta_gw_boost` и `trn_gain` указаны `connected: false` и причина отсутствия
  утверждённого контракта потребителя. `GATES_CONNECTED` остаётся `False`.
  Телеметрия `IntegratedOrganism` теперь также передаёт последние измеренные
  confidence, predicted loss, surprise, novelty, uncertainty, calibration и
  полный набор вычисленных gates. `novelty_unit` обозначен как потребляемый
  внутри `replay_priority`, без отдельного consumer.
* Энергия `GlobalWorkspace` по-прежнему не влияет на действие, ignition или
  обновление модели. Владелец выставляет `energy_consumer_connected: false`,
  пустой список точек потребления и причину
  `diagnostic_only_no_approved_consumer_contract`; integrated telemetry
  экспортирует это состояние вместе с текущей энергией. Новый consumer не
  придуман и не подключён.
* Проверки `tests/test_v22_phantom_consciousness.py -k
  "meta_head_reports or workspace_tracks or live_integrated_telemetry"` прошли:
  4 passed. `tests/test_v22_meta_replay_priority.py` ранее прошёл 8 тестов до
  дополнительного расширения диагностической телеметрии; финальный повтор
  остановился на лимите памяти во время pytest collection/cache, не на assertion.
  Повтор всего `test_v22_phantom_consciousness.py` также упёрся в MemoryError
  при AST-разборе большого `flat.py`: 21 passed, 2 теста не завершили assertions.

Таким образом, остальные выходы MetaHead и энергия GW теперь явно обозначены
как не подключённые, а не представлены как действующие. Их статус повышать до
SHIPPED нельзя, пока не появится обоснованный consumer-контракт и причинный
тест на его эффект.
