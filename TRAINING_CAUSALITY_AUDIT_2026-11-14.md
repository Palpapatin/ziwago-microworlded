# Причинность обучения Ziwago18 — 2026-11-14

## Краткий вывод

Проверен локальный причинный контур обучения по коду и уже имеющимся
контрольным тестам. Это не новый запуск pytest и не измерение на обученном
чекпоинте: на момент проверки доступно 1.77 GiB RAM из 15.94 GiB (88.9 %
занято), тогда как проектный предел — не более 85 % занятости, а загрузчику
чекпоинта требуется ориентировочно не менее 10 GiB свободной памяти. В
прикреплённом worktree подходящего обученного чекпоинта также нет.

Итог по уровню доказательств:

* **Причинно подтверждены на малом/синтетическом контроле:** signed phasic DA
  меняет синаптические веса; обучение коркового контура меняет `w_projector`,
  синапсы и apical-ответ, а learned-контур улучшает held-out классификацию;
  Fisher и ACL меняют величину дрейфа readout-весов; общий production-тренер
  пишет веса и реплей-кредит, а sleep-гейт закрывает обучение.
* **Проводка или вычисление есть, но причинный эффект канала не установлен:**
  ACh как независимый фактор на итоговых весах; изолированные MCA и DCC;
  BCM как фактор изменения результата обучения; MetaHead replay priority на
  итоговой метрике обучения после сна; выборка и консолидация реплея в
  сравнении on/off.
* **Не являются обучающими факторами в текущем контуре:** энергия Global
  Workspace — диагностическая; вычисляемые MetaHead `bg_threshold`,
  `beta_gw_boost`, `trn_gain` не подключены к потребителям; BA44
  `learn_bridge_transition()` не вызывается в активном production training
  path. BA44 на ответах самой модели не обучался и не тестировался.

## Production-путь и модель обновлений

В `ZiwagoOrganism._train_wave_with_neuromodulator_reward()` глобальный флаг
обучения может полностью остановить волну; в integrated-пути нулевая
физиологическая модуляция блокирует пару, а токеновый цикл повторно проверяет
`f_mod`. При открытом гейте `f_mod` масштабирует PC counterfactual step,
spiking `task_modulation`, контекст начала ответа, dendritic-batch reward,
onset-синапсы и отдельные legacy n-граммные таблицы. Сама функция вызывается
как основной training path в `ziwago/organism/flat.py:1356-1393`,
токеновые обновления — `:1460-1490`, пакетные/контекстные обновления —
`:1498-1571`.

Интегрированный live-контур задаёт `plastic` из `learning_enabled`,
`training_active` и `f_mod > 0`, передаёт его коре и исполнительным модулям
(`ziwago/organism/integrated.py:252-292`). Нейрокорковый learner отдельно
возвращается без обновлений при выключенном обучении или нулевой
`task_modulation` (`ziwago/cortex/spiking_language.py:86-90`), иначе передаёт
ошибку вектору-проектору и apical teaching (`:119-136`). Поэтому наличие
вычисленного сигнала само по себе не означает, что он реально изменил вес:
важны активная ветка, не нулевой gate, строки/колонки с сигналом и конечный
контроль весов.

Интегрированный путь намеренно не обновляет bigram/trigram таблицы:
`flat.py:1529-1532` ограничивает их legacy/no-life режимом; проверка
`tests/test_v20_integration.py::test_training_writes_weights_and_replay_credit_without_ngrams`
утверждает, что при integrated training таблицы пусты, но веса и очередь
реплея обновляются.

## Матрица причинных свидетельств

| Механизм/канал | Кодовой потребитель и возможный эффект | Причинное свидетельство и его границы | Оценка |
|---|---|---|---|
| `learning_enabled` / physiology `f_mod` | Гейтирует волну и токены; масштабирует несколько семейств updates (`flat.py:1377-1393`, `:1463-1489`, `:1517-1571`). | `tests/test_v20_integration.py::test_training_writes_weights_and_replay_credit_without_ngrams` проверяет, что интегрированное обучение пишет веса/credit. `::test_sleep_closes_plasticity_and_replays_tagged_cues` подтверждает `f_mod=0`, отсутствие обученной dendritic-пары и subsequent replay. `tests/test_v20_physiology.py::test_plasticity_gate_uses_mean_atp_and_really_closes` проверяет закрытие гейта. Это доказательство gate/open-close, но не matched-run количественного эффекта разных положительных значений `f_mod`. | **Подтверждён узкий gate-контракт; эффект величины не измерен.** |
| Учительская ошибка → корковая пластичность | `SpikingLanguageLoop.learn()` обновляет `w_projector` и направляет ошибку в apical compartments / синапсы (`spiking_language.py:119-136`). | `tests/test_v19_spiking_language.py::test_teacher_error_reaches_real_apical_compartments_and_synapses` фиксирует изменения `v_a`, `w_sst_in`, `dendrite.w_syn`; `::test_spiking_learning_uses_shared_counterfactual_error` проверяет точное outer-product обновление; `::test_learned_heldout_cortical_mapping_and_independent_ablations` сравнивает normal/disabled/shuffled/constant и held-out loss/accuracy. Синтетический supervised-контроль, не оценка ответа на реальном обученном checkpoint. | **Причинность подтверждена локально.** |
| Внешний phasic DA в синапсе | Отдельный signed DA-множитель eligibility передаётся в `Synapse.learn()` (`ziwago/core/synapse.py:490-499`, `:538-552`). | `tests/test_v20_cortical_clock.py::test_signed_external_da_reaches_delayed_eligibility_even_in_silence`: на одинаковом исходном eligibility DA -1/0/+1 даёт отрицательное/нулевое/положительное изменение `w`; `None` равен нулю, а немедленная supervised ошибка остаётся при DA=0. Это чистый причинный локальный тест. | **Причинность подтверждена на синапсе.** |
| ACh как независимый фактор | Умножает eligibility и влияет на cholinergic tag (`synapse.py:524-529`, `:538-544`); организм передаёт `ach` отдельно (`spiking_language.py:119-131`). | `tests/test_v20_cortical_clock.py::test_active_forward_and_apical_teaching_share_one_elapsed_tick` доказывает передачу ACh/DA аргументов, но не разницу весов ACh-on/off. Нет найденного изолированного matched-state теста, который меняет только ACh и измеряет последующий вес/результат. | **Проводка подтверждена; эффект ACh не доказан.** |
| Fisher anchor в readout | Наблюдает обновления, затем делает proximal pull к якорю (`ziwago/cognition/language/parts/regulators.py:199-208`, `:287-292`). | `tests/test_v21_local_fisher.py::FisherReadoutIntegrationTests::test_enabled_fisher_reduces_drift_from_initial_weights` сравнивает on/off при одних шагах и проверяет, что drift-on меньше drift-off. Это показывает влияние на сохранение весов, не самостоятельное повышение качества обучения. | **Причинность подтверждена для дрейфа.** |
| ACL row-gate в readout | Умножает row gain на строковый update (`regulators.py:226-234`, `:281-285`). | `tests/test_v21_acl_wiring.py::test_acl_enabled_causally_suppresses_drift` сравнивает on/off-весовой дрейф. | **Причинность подтверждена для масштаба обновления.** |
| MCA susceptibility | Собирает наблюдения и формирует `row_gain`, который умножается на update (`regulators.py:210-224`, `:281-285`). | Найден composition-тест MCA+ACL, но не изолированное сравнение MCA on/off конечных весов/метрики. Наличие `row_gain` не заменяет такого сравнения. | **Вычисляется/включено; независимая причинность не доказана.** |
| DCC readiness | Формирует зональный `zone_gain`, умножает его на `row_gain` (`regulators.py:236-269`, `:281-285`). | Нет изолированного DCC on/off matched-run теста конечного update или результата. Наличие readiness-метрик не доказывает, что DCC меняет активный итоговый вес в типовом обучающем состоянии. | **Проводка есть; независимая причинность не доказана.** |
| MetaHead `replay_priority` | Прибавляется к базовому приоритету replay-credit; в NREM выбранный cue вызывает `consolidate_tagged()` (`integrated.py:257-263`, `:333-339`, `:586-600`). | `tests/test_v22_meta_replay_priority.py::test_present_credit_delay_changes_the_next_replay_selection` для ticks/token 1 и 4 проверяет задержку, сохранение base/meta priorities и смену следующего cue on/off. Это подтверждает причинность выбора, не то, что выбранный реплей меняет веса лучше или улучшает следующую train/eval loss. `test_sleep_closes_plasticity_and_replays_tagged_cues` проверяет replay events, но не MetaHead on/off-веса. | **Причинность выбора подтверждена; downstream learning effect не измерен.** |
| BCM gate при явном assistant teach | `evaluate_neuromodulatory_gate()` допускает/блокирует teach path в `ziwago_assistant.py:94-115` (см. лiveness-аудит). | Checkpoint roundtrip не является контролем эффекта; для одинакового входа нет найденного BCM on/off сравнения весов или результата. | **Production gate есть; причинность обучения не доказана.** |
| Энергия Global Workspace | `energy` записывается в историю/телеметрию; `energy_consumer_connected=False`, consumer points пусты и причина diagnostic-only (`ziwago/consciousness/workspace.py:57-82`). | Нет production-потребителя в training loss/update и нет веса, который мог бы зависеть от этой энергии. | **Не является фактором обучения; только телеметрия.** |
| MetaHead `bg_threshold`, `beta_gw_boost`, `trn_gain` | Выходы вычисляются, но актуальный consumers для них не подключены; статусы канала в текущей MetaHead telemetry. | Нет пути от этих сигналов к учительской ошибке, learning rate, выбору строки/веса или evaluation result. | **Не являются факторами обучения сейчас.** |
| BA44 `learn_bridge_transition()` | Функция существует в `ziwago/cognition/language/syntax.py:234-274`, активный production training call не найден. | Отдельный контроль независимых человеческих/авторских переходов не проводился. В соответствии с запросом learner не вызывался, обучение на собственных ответах исключено. | **Не подключён к обучению; эффект отсутствует/не заявляется.** |

## Исполнение и блокировки

В этом проходе новые pytest-процессы не запускались. На снимке ресурсов было
1.77 GiB available и 88.9 % RAM utilization; это превышает проектный порог
85 % занятости и не подходит ни для полноценного тестового набора, ни для
checkpoint probe. Не запускал `tools/probes/module_map_probe.py` без checkpoint
и не пытался загрузить полный checkpoint. Это ресурсная блокировка, а не
результат теста.

Ранее в рамках текущей перепроверки модулей отдельные тесты, включая
`test_v20_integration.py` (7 passed), `test_v20_physiology.py` (67 passed) и
`test_v20_cortical_clock.py` (24 passed), уже выполнялись через
`tools/limited_run.py`; см. `MODULE_LIVENESS_RECHECK_2026-11-14.md:80-98`.
Они являются существующими лёгкими причинными свидетельствами, но здесь не
перезапускались. Аудит не утверждает, что они проверяли все перечисленные
каналы.

## Следующие проверки при доступности ресурсов

1. Сначала повторить адресные причинные unit tests отдельно через
   `tools/limited_run.py`, когда utilization ниже 85 %; не запускать набор
   параллельно с обучением.
2. Добавить/запустить малые matched-state тесты, меняющие ровно один фактор:
   ACh на delayed eligibility, MCA-only, DCC-only, BCM gate on/off, и
   MetaHead replay-priority off/on до/после сна с явными snapshots конкретных
   весов и общей независимой loss-метрикой.
3. Для утверждений о поведении обученного организма дождаться разрешённого
   checkpoint и машины с достаточной RAM; запускать только ограниченным
   `tools/limited_run.py`. Не заменять это тестом на fresh/random core.
4. BA44 не обучать на сгенерированных моделью ответах. Возможен отдельный тест
   только на заранее заданных независимых transition labels, если будет
   утверждён источник разметки и потребитель эффекта.

Production-код в рамках этого анализа не менялся; изменения только в этом
документе.
