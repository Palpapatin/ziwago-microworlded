# Стоимость фаз и восстановление runtime-состояния

## Итог по состоянию на 2026-11-15

Ранее были изменены SWR gating, selective Workspace snapshot и evaluation
rollback; подробная история и их отдельные проверки приведены ниже. В текущем
проходе закрыты найденные дефекты состояния и безопасного восстановления:

* Checkpoint загружается из тех же проверенных SHA-256 байтов, что были
  верифицированы; unrestricted legacy pickle требует явного trust и ожидаемого
  хеша.
* `IntegrationConfig` применяется до checkpoint restore, поэтому создание
  integrated driver не заменяет восстановленные replay/action/criticality state.
* Router sweep-телеметрия восстанавливает состояние и при исключении.
* Дендритный Ca-порог вычисляется как baseline + текущий астроцитарный сдвиг;
  restore migration исправляет накопленный legacy-порог и записывает это в отчёт.
* Beam rollback сохраняет динамику колонок, включая recurrent resource `u_syn`
  даже без active ejection/integrated clock; inference не затухает выученные
  веса через внешний синаптический clock.
* Химическая телеметрия различает подключённый volume-eCB косвенный путь и
  неподключённый volume-H₂S; проверен same-tick обучающий сигнал дендрита.

**Текущая проверка:** объединённый focused-набор завершился с результатом
`74 passed, 1 warning, 8 subtests passed in 81.38s`. Единственное предупреждение
— PyTorch сообщает, что buffer checkpoint mmap read-only, хотя tensor API
теоретически допускает запись; hash-pin регрессионный тест прошёл. Точная команда
и границы этого результата приведены в конце документа. Эти тесты используют
маленькие synthetic/tiny-core состояния, не обученный v19 checkpoint.

**Граница доказанного восстановления:** тесты проверяют байты hash-pinned
загрузки, явный `IntegrationConfig` при restore, отсутствие побочных эффектов
router telemetry, baseline/сброс Ca-порога, парный chemical effect и beam
кандидаты в обратном порядке с `branch_factor=2`. Это подтверждает только
зафиксированные поля и сценарии тестов, не полную эквивалентность произвольного
runtime-состояния или обученного checkpoint.

**Не проверено:** генерация на обученном checkpoint; полный/большой набор тестов;
причинные SWR-тесты в текущем прогоне; wall-clock snapshot/load и фиксированные
wall-clock sleep/replay. Исторические generation/snapshot миллисекунды ниже —
только ориентир от 2026-11-12. Текущий отдельный microbenchmark phase-router
приведён ниже и не является end-to-end профилем. Оптимизацию beam snapshot
не внедряли.

### Контекст исторических замеров доступной RAM

Показатели ниже сняты/сообщены в разные моменты и не противоречат друг другу.
Это измерения доступной памяти на Windows-хосте с 15.94 GiB физической RAM;
точное время самого опроса не сохранилось для первых двух записей.

| Доступно | Контекст времени и рабочее дерево |
|---:|---|
| 0.93 GiB (94.2% занято) | Снимок при подготовке первоначального отчёта 2026-11-14; точное время опроса не записано. Основное дерево `C:\Users\yarma\Desktop\Ziwago18`. |
| 1.12 GiB (93.0% занято) | Сообщено в статусе Delta, полученном 2026-11-14 23:17:59Z; точное время измерения не записано. Delta worktree, его точный путь в сохранённой заметке отсутствует. |
| около 8.05 GiB | Сообщено после tiny-core проверки в статусе от 2026-11-15 06:11:19Z (08:11:19 +02); это доступная память в тот момент, не постоянная характеристика машины. Основное дерево `C:\Users\yarma\Desktop\Ziwago18`. |

У всех трёх значений контекст — одна локальная Windows-машина; менялась
загруженность системы и в одном случае использовался отдельный Delta worktree.

## Историческая стоимость — ориентир 2026-11-12

Это **не повторный замер**. Источник значений: таблицы и измерения в
`INNOVATION_PROPOSALS_2026-11-12.md:57-95`. Для повторного тяжёлого запуска
этот документ задаёт запуск частей отдельно через `tools/limited_run.py` с
`--ram-pct 55 --cpu-pct 40`; значения нельзя считать текущей производительностью
или повторно воспроизведёнными здесь.

| Операция | Baseline от 2026-11-12 | Интерпретация/ограничение |
|---|---:|---|
| Полный снимок dynamic-state | 3.75 MB, 756 тензоров | Один snapshot на тестовом checkpoint/runtime; это не размер полного checkpoint. |
| Экспорт снимка в generation | 12.9 ms/вызов, 137 вызовов, 12.7 % времени | Измерено на 12-token generation; суммарно около 1.77 s по округлённым значениям таблицы. |
| Загрузка/восстановление снимка | 10.2 ms/вызов, 138 вызовов, 10.1 % времени | Суммарно около 1.41 s; экспорт плюс load — 22.8 % латентности. |
| Генерация целиком | 13.8 s на 12 токенов | Контекст измерения из прежнего efficiency probe; не новый acceptance-порог. |
| `token_components` | 109.9 ms/вызов, 34 вызова, 27.0 % | Дендритный readout + PC; основной активный вычислительный блок по прежнему профилю. |
| `spiking_loop_step` | 97.9 ms/вызов, 34 вызова, 24.0 % | Спайковая кора; второй крупнейший блок профиля. |
| `cortical_step` | 68.2 ms/вызов, 2 вызова, 1.0 % | Мало вызовов и малая доля времени по прежней трассе. |
| Sleep / SWS / REM | **Времени не измерено** | `sleep_cycle()` считает `budget_ticks` и elapsed model-time, это не wall-clock. |
| Replay / tagged consolidation | **Времени не измерено** | Есть `replay_events` и `replay_synapses`, нет длительности/CPU профиля на событие. |

Прежний profiling report также фиксировал 877,946 вызовов `export_runtime_tree`
и 720,698 вызовов `_get_runtime_path` при beam search. В коде
`_synthesize_neural_beam_search()` сохраняет состояние для initial beam,
post-distribution, child beams и откатывает его при переходе между ветками
(`ziwago/organism/flat.py:1903-1944`, `:2013-2026`, `:2087-2091`,
`:2119-2126`). Это согласуется с измеренной долей snapshot/load, однако число
вызовов и доля относятся к прежней конкретной трассе, а не к каждой генерации.

Сон и replay структурно ограничены счётчиками/бюджетом: `sleep_cycle()` принимает
`max_ticks` внутри `max_sleep_ticks` (`ziwago/organism_core.py:677-704`), а
переход NREM→REM повторно использует readout и hippocampus
(`organism_core.py:644-675`). Replay выбирает приоритетный cue и применяет
`consolidate_tagged()` (`ziwago/organism/integrated.py:255-263`,
`:333-339`). Ни ограничение числа тиков, ни число обновлённых синапсов не дают
измерения времени. Нужен отдельный `perf_counter`-профиль фиксированного числа
тиков/replay-событий.

## Экспорт, импорт, восстановление поведения и телеметрии

### Уже покрытые контракты

В исходниках найдены следующие причинные/round-trip проверки. Первоначально,
при аудите 2026-11-14, их не запускали из-за нехватки RAM; позднее перечисленные
ниже проверки были частично воспроизведены точечным набором:

* `tests/test_runtime_paths.py` сравнивает селективный export с эталонным
  обходом и проверяет wire format, порядок путей, алиасы тензоров, сохранение
  снимка при последующей мутации исходного тензора и замену поддеревьев.
* `tests/test_v20_integration.py::test_dynamic_snapshot_roundtrip_covers_v20_elapsed_fields`
  экспортирует dynamic-state, меняет состояние, импортирует snapshot и
  сравнивает весь снимок повторно; `::test_frozen_evaluation_restores_full_state_even_on_exception`
  проверяет точное восстановление после мутации и исключения.
* `tests/test_v21_cortical_evidence.py::test_language_snapshot_replays_window_and_restores_prediction_telemetry`
  проверяет повторяемость output/spike window и восстановление `last_features`,
  spike raster и других prediction telemetry.
* `tests/test_v20_organism_composition.py::test_evaluation_context_restores_state`
  сравнивает полное v19 runtime состояние до/после evaluation context; ранее
  данный метод также покрыт тестом восстановления при исключении выше.

Эти проверки доказывают только зафиксированные поля и тестовые сценарии. Они
не являются измерением скорости и не доказывают полную эквивалентность всего
runtime дерева на checkpoint-модели.

### Найденный telemetry gap в selective language snapshot

`export_language_dynamic_state()` сохраняет в `son_column` выбранные поля
Workspace (`x`, `gate`, `ignition`, `last_coherence`) и MetaHead, но не включает
`son_column.gw.energy_hist` и `son_column.gw.telemetry`
(`ziwago/organism/flat.py:587-616`). Workspace при каждом `compete()` меняет
оба этих диагностических поля (`ziwago/consciousness/workspace.py:58-82`,
`:91-96`, `:171-173`). Поэтому selective snapshot не восстанавливает историю
энергии и последнее значение/consumer-телеметрию Workspace. В полном
evaluation snapshot поля могут сохраняться другим путём, но beam-local
`export_language_dynamic_state()` этот контракт не обеспечивает.

В текущем Workspace energy consumer явно отсутствует; это телеметрический,
а не обучающий канал. Отсутствие этих полей поэтому не доказано как причина
смены logits, но **точное восстановление telemetry нарушено**. Добавлен
targeted test: записать две разные энергии и telemetry в Workspace,
экспортировать dynamic-state, изменить эти поля, импортировать снимок и
сравнить `energy_hist`/`telemetry` плюс повторный `GWOut` на одинаковых входах.
Этот тест прошёл в целевом прогоне 2026-11-15; он не доказывает полную
эквивалентность runtime-дерева на обученном checkpoint.

## Оптимизация и критерии

Подтверждённое место для последующей оптимизации — повторный полный snapshot
веток beam search, а не sleep/replay: только оно имеет измеренную долю
latency. Предложение прежнего отчёта — сохранять/восстанавливать только
действительно изменяемые компоненты и RNG вместо полного dynamic tree
(`INNOVATION_PROPOSALS_2026-11-12.md:247-268`). Эта замена не сделана: нельзя
предполагать, что пропущенные поля не меняются или что telemetry/поведение
останутся прежними.

Перед оптимизацией обязательны следующие ограниченные проверки:

1. Один и тот же фиксированный вход и seed на baseline и кандидатной версии:
   текст, beam sequences, `selected_beam_idx`, finish reason и generation
   telemetry должны совпасть; любое различие — отчёт, а не скрытый tolerance.
2. Snapshot/load должны сохранять полный тестируемый dynamic state, RNG,
   telemetry и поведение следующего шага. Тест на Workspace
   `energy_hist`/`telemetry` и следующий GW результат прошёл на tiny-core;
   повторить его на штатном/обученном организме ещё не удалось.
3. Повторить профиль 12 токенов, записать checkpoint/schema, beam width,
   branch factor, max tokens, thread counts, RAM/CPU caps, wall time,
   RSS, export/load count и длительности по этапам.
4. Измерить отдельно sleep и replay при фиксированных `max_ticks`,
   replay queue length/selected items и одинаковом исходном runtime; записать
   wall-clock, число replay-событий/синапсов, телеметрию фаз и состояние до/после.
5. Каждый тяжёлый процесс запускать отдельно через
   `tools/limited_run.py`; не запускать pytest параллельно.

Новый wall-clock baseline для snapshot/load, sleep или replay не получен.
Оптимизацию снимков следует отложить до повторяемого профиля и возможности
проверить round-trip и generation equivalence на одном и том же состоянии.

## Дополнение после проверки sleep/SWR — 2026-11-14

Повторная проверка активного пути `nrem → rem` подтвердила: владелец вызывает
`hippo.sleep_cycle()` без `cortical_write`
(`ziwago/organism_core.py:644-675`). У hippocampus есть параметрический callback
и метод `swr_replay`, но другого production-модуля, который задаёт обоснованный
supervised cortical update для replay-эпизодов, не найдено. Поэтому callback
не был изобретён и SWR остаётся отключённым. `Hippocampus.sleep_cycle()` теперь
считает отсутствие writer закрытым SWR-гейтом и возвращает явный флаг
`swr_cortical_write_connected` вместе с причиной `no_cortical_write`.
Добавлены причинные assertions для прямого SWR-gate и активного обработчика
NREM→REM, включая проверку, что `swr_replay` не вызывается без consumer.

Selective language snapshot расширен полями Workspace `energy_hist` и
`telemetry`; добавлен round-trip test, проверяющий восстановление истории,
произвольного telemetry marker и равенство следующего GW output.

На момент передачи Delta-результата, полученного 2026-11-14 23:17:59Z,
сообщалось 1.12 GiB available из 15.94 GiB (93.0 % занято); точное время
системного опроса и путь Delta worktree не сохранились. SWR-gate/handler-specific
тесты тогда не запускались и в целевой набор 2026-11-15 ниже не включены.
Workspace snapshot test позже прошёл. Сопоставленный on/off replay на обученном
состоянии и wall-clock замеры sleep, replay, snapshot/load по-прежнему
не выполнялись.

## Дополнение: проверка восстановления и генерации — 2026-11-15

При проверке незавершённой работы агента tiny-core round-trip probe обнаружил
130 отличий после восстановления и повторения тех же тиков. Сам probe при этом
возвращал код 0 даже при наличии отличий; это исправлено: теперь он завершается
с ошибкой, если состояние различается. После исправления probe показал
**0 расхождений**. Для обычной загрузки checkpoint сохранена миграционная
совместимость: core restore сохраняет неизвестные верхнеуровневые поля, а
generic loader по умолчанию очищает только отсутствующие `_last_*`. Только
evaluation rollback включает строгую очистку отсутствующих non-callable полей
в точных классах; подклассовые extras остаются защищены. Самостоятельные
component-specific `load_state_dict` сохраняют свои прежние правила.

Исходная команда первого успешного прогона из заметок не сохранилась. После
исправления evaluation rollback 2026-11-15 выполнен этот воспроизводимый набор:
предыдущие 13 проверок плюс regression test с `last_telemetry`
(маленький checkpoint с production-генерацией, restore→`present()`, evaluation
restore, Workspace snapshot и миграции):

```powershell
python tools\limited_run.py --ram-pct 35 --ram-free-pct 50 --cpu-pct 40 --log "$env:TEMP\audit_restore_tests_lasttelemetry_20261115_final.log" -- python -m pytest tests\test_organism_load.py::LoadFidelityTests::test_runtime_state_roundtrip_byte_identical tests\test_v19_spiking_language.py::V19Tests::test_checkpoint_persists_loop_ledger_shield_bank_and_moe tests\test_v20_integration.py::V20IntegrationTests::test_frozen_evaluation_restores_full_state_even_on_exception tests\test_v20_integration.py::V20IntegrationTests::test_runtime_restore_replays_next_present_identically tests\test_v20_integration.py::V20IntegrationTests::test_dynamic_snapshot_roundtrip_covers_v20_elapsed_fields tests\test_v20_organism_composition.py::FlatV20Tests::test_evaluation_context_restores_state tests\test_v21_cortical_evidence.py::test_language_snapshot_restores_workspace_history_telemetry_and_next_result tests\test_v22_restore_chemistry_migration.py tests\test_v19_spiking_language.py::V19Tests::test_evaluation_disables_learning_writes_and_restores_all_state_and_rng -q
```

Фактический результат: `14 passed, 6 subtests passed in 10.98s`; код завершения
0, пик 1.97 GiB при лимите 3.22 GiB. Отдельный targeted запуск теста с
`last_telemetry` также прошёл: `1 passed in 3.50s`, код 0, пик 1.54 GiB при
лимите 3.33 GiB.

Tiny-core probe повторён отдельно:

```powershell
python tools\limited_run.py --ram-pct 35 --ram-free-pct 50 --cpu-pct 40 --log 'D:\Ziwago18-storage\Produce\logs\audit_state_diff_repro_20261115.log' -- python tools\probes\_state_diff.py
```

Результат: `расхождений: 0`, код завершения 0; пик 1.51 GiB при лимите
3.50 GiB. Probe сравнивает состояние после одинаковых тиков и восстановления
с вызовом `present()`.

Историческая попытка целевого набора включала
`V19Tests.test_evaluation_disables_learning_writes_and_restores_all_state_and_rng`
и завершилась `1 failed, 12 passed, 6 subtests passed in 15.05s` (пик 1.97 GiB,
лимит 3.43 GiB): `last_telemetry` оставалось после restore. При исправлении
этого top-level поля повторный запуск того же теста выявил вложенное
`last_local_plasticity_gain`, которое также оставалось после restore. Строгий
rollback закрывает обе точки; тест включён в набор выше и проходит. Это не
доказательство точного восстановления любого возможного runtime-поля.

Также ранее отдельно сообщалось:

* восстановление tiny-core и повтор `present()` побайтно совпадают со
  snapshot-трассой: `test_runtime_restore_replays_next_present_identically`;
* сохранение/загрузка маленького self-contained checkpoint и последующая
  production-генерация проходят, ответ не пустой:
  `test_checkpoint_persists_loop_ledger_shield_bank_and_moe`;
* restore/migration, exact-state, frozen evaluation, Workspace snapshot и
  `last_telemetry` regression: **14 passed, 6 subtests passed** (команда выше);
* `_state_diff.py`: `расхождений: 0`.

Это проверка tiny-core и маленького синтетического checkpoint, не обученного
v19-кандидата. Известный checkpoint
`D:\Ziwago18-storage\Produce\checkpoints\ziwago_v19_bankfixed_20261109.pt`
имеет размер около 4.54 GB; его полная загрузка/генерация требует примерно
10 GB свободной RAM. В отчёте после tiny-core проверки, переданном
2026-11-15 06:11:19Z (08:11:19 +02), сообщалось около 8.05 GiB доступной RAM
на основном worktree; это исторический моментный замер, а не текущий уровень.
Поэтому фактическую генерацию восстановленного обученного кандидата
**не запускали**. Агентское `loaded_ok: true` не трактуется как проверка
генерации. Следующий шаг — повторить ограниченную генерацию этого кандидата
только после наличия ≥10 GiB свободной RAM и выполнить её через
`tools/limited_run.py`; до этого точный ответ/качество кандидата остаётся
непроверенным.

## Проверка инвентаризации checkpoint для Э-7 — 2026-11-15

Добавлен read-only прибор
[`probe_checkpoint_storage_categories.py`](./tools/probes/probe_checkpoint_storage_categories.py).
Он открывает checkpoint через существующий Windows READONLY mmap, считает
уникальные tensor storages и классифицирует их по именам полей. Прогон
`ziwago_v19_bankfixed_20261109.pt` через `tools/limited_run.py` с лимитом RAM
2.80 GiB завершился успешно; пик процесса — 2.33 GiB.

Результат прибора:

| Категория-предположение | Уникальные байты | GiB | Доля |
|---|---:|---:|---:|
| Core | 2,717,471,332 | 2.531 | 60.77 % |
| Plasticity | 1,752,989,696 | 1.633 | 39.20 % |
| Telemetry | 1,419,804 | 0.001 | 0.03 % |
| Всего | 4,471,880,832 | 4.165 | 100 % |

Крупнейшее хранилище, помеченное пластичностью по имени поля, — `shield.u_slow`,
372,801,536 байт (355.5 MiB); матрицы `w_dend_out`, `w_core_dend_out` и
`w_swr_dend_out` имеют по 377,487,360 байт. Это инвентаризация storage, а не
доказательство безопасного core-only restore: категория вычисляется эвристикой
имени, некоторые счетчики обучения и runtime-поля пока могут учитываться как
core, а загрузчик восстанавливает полное runtime-дерево.

Поэтому Э-7 пока не реализован: отделение sidecar-файла само по себе не
устраняет выделение пластических матриц в живом организме. Для продолжения
нужны конкретная карта evaluation-необходимых полей, core-only путь с явной
миграционной телеметрией и проверка побайтного fingerprint ядра плюс
production-генерации на старом checkpoint. Мастер и существующий
`ziwago_v19_bankfixed_20261109.pt` не изменялись.

## Дополнение: фиксы обзора и текущий phase-router baseline — 2026-11-15

Точный targeted набор текущего прохода:

```powershell
python -m pytest -q tests/test_v20_cortical_clock.py tests/test_v20_organism_composition.py tests/test_v22_astrocyte_wiring.py tests/test_v22_restore_chemistry_migration.py tests/test_volume_chemistry.py tests/test_v22_beam_state_isolation.py tests/test_v22_checkpoint_hash_pin.py
```

Фактический результат: `74 passed, 1 warning, 8 subtests passed in 81.38s`,
код завершения 0. Предупреждение — `torch.frombuffer` сообщает о read-only
buffer в `_readonly_checkpoint_mapping`; проверенный loader и его TOCTOU-тест
успешны. Запуск покрывает hash-pinned загрузку и требование явного trust для
legacy pickle, restore с `IntegrationConfig`, транзакционную router telemetry,
Ca-baseline и migration старого накопленного порога, beam branch-order
independence и recurrent-resource restore без active ejection/integrated clock,
inference decay, volume eCB/H₂S routing и дендритный teaching signal. Это
focused набор, не полный suite.

Отдельный замер текущего CPU phase-router выполнен на Windows-хосте
`AMD Ryzen 7 3700X`, PyTorch `2.14.1+cu130`, с 15.94 GiB физической RAM.
CUDA доступна, но не использовалась: router измерялся на CPU, `torch` ограничен
одним потоком. Конфигурация: 16 колонок, ring length 64, seed 7; 20 прогревочных
шагов и 5 пакетов по 200 шагов. Медиана — **29.4884 ms/step**, диапазон пакетов
**29.3403–29.5312 ms/step**. Это локальный microbenchmark текущей реализации,
не end-to-end generation профиль и не сравнение оптимизированного кандидата.
Повторить измерение можно таким PowerShell-командным блоком:

```powershell
@'
import statistics
import time
import torch
from ziwago.cortex.moe_cortex import ReferenceInvariantLagPLVRouter

torch.set_num_threads(1)
router = ReferenceInvariantLagPLVRouter(n_columns=16, ring_len=64, seed=7)
rates = torch.linspace(0.01, 0.2, 16)
for _ in range(20):
    router.step_phases_and_compute_causal_matrix(rates)
measurements = []
steps = 200
for _ in range(5):
    start = time.perf_counter()
    for _ in range(steps):
        router.step_phases_and_compute_causal_matrix(rates)
    measurements.append((time.perf_counter() - start) * 1000.0 / steps)
print({
    "device": "cpu",
    "n_columns": 16,
    "steps_per_batch": steps,
    "batches": len(measurements),
    "median_ms_per_step": round(statistics.median(measurements), 4),
    "range_ms_per_step": [round(min(measurements), 4), round(max(measurements), 4)],
    "cuda_available": torch.cuda.is_available(),
})
'@ | python -
```

Векторизация не выполнялась: этот baseline лишь фиксирует измеряемую стоимость.
Snapshot/load, sleep и replay всё ещё не имеют текущего wall-clock замера; для
них нужен отдельный воспроизводимый сценарий и сравнение точного состояния до
оптимизации.
