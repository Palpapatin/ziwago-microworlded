# NEUROWEAVE ZIWAGO-PRIME v19 — экспериментальная версия

## Актуальный контракт v19

Точка входа — `ziwago_assistant.py`, runtime — `train_ziwago_v19_verified_runtime.py`, обучение — `run_ziwago_v19.py`. Числа, гарантии и научные интерпретации в историческом руководстве ниже не являются измерениями или сертификацией v19.

- Причинная память хранит источник, время, полярность, условия, тип связи и конфликты. Старые рёбра без происхождения изолируются; отсутствие достаточной причины не доказывает отсутствие следствия.
- Все ветки ассистента проходят grounding до записи ответа в память. Поддержка локальным источником не равна внешней проверке истинности. Неподдержанный ответ отклоняется независимо от переключателя active inference.
- Спайковый языковой контур, апикальная ошибка, защищённые разреженные обновления и динамические банки проверяются отдельными абляциями; сам факт наличия модуля не доказывает пользу для произвольного языка.
- Временной канал использует бинарные события и задержки; структурный перенос использует типизированные графы, а не словари заранее написанных аналогий.
- Сон проверяет гипотезы и повторяет только атрибутированные обучающие источники. Собственный сгенерированный ответ не становится доказательством. Изменения требуют независимого acceptance и полного отката при провале.
- Недостающие измерения отмечаются как unavailable, не как 100% успеха. Диагностические показатели не являются доказательством сознания.
- Legacy мастер `Produce/checkpoints/ziwago_v18_1_genesis_omniscient_master.pt` не перезаписывается. Кандидат и принятый checkpoint v19 — отдельные файлы; допуск связан с SHA-256 конкретного кандидата.

### Запуск

```bash
python ziwago_assistant.py
python -m pytest tests/test_v19_causal_temporal.py tests/test_v19_research_modules.py tests/test_v19_spiking_language.py tests/test_v19_language_scoring.py tests/test_v19_evaluation.py tests/test_v19_runtime.py tests/test_v19_document_ranker.py -q
python tools/preflight.py --require-corpora
python run_ziwago_v19.py --dataset data/train.jsonl --acceptance data/acceptance.json --epochs 1
```

Обучающий JSONL содержит `prompt`, `response`, `source`, `family_id`, `split: "train"`. Acceptance — отдельный JSON-массив минимум из двух случаев с `prompt`, `expected_concepts`, `family_id` и при необходимости `forbidden_echo`; его семейства и вопросы должны быть отделены от train. Доступны подготовленные OASST2 train-пары, но независимый production acceptance ещё не создан и не должен превращаться в учебные уроки. Происхождение принятой train-записи включает путь, SHA-256 файла и фактический номер строки. BCM получает измеренный результат синтаксического детектора, а не константу успеха; этот эвристический детектор не доказывает полноценную грамматическую правильность. `--promote` допускает только отдельный `ziwago_v19_0_accepted.pt` после успешной проверки записанного кандидата, не старый мастер.

### Отдельный open-book QA

По умолчанию acceptance остаётся `closed_book`, без документа. `--acceptance-mode open_book` требует непустое поле `document` в каждом случае и вызывает только `chat_with_document(prompt, document)`. Проверка независимо сверяет смещения полных дословных предложений и покрытие ими всего ответа: допускается максимум две цитаты, копирование многофразового документа целиком отвергается. Concept coverage, отрицания, forbidden echo и collapse не ослаблены. Документы и режим входят в `split_hash`; известные train-документы, хеши, вопросы и семейства проверяются также по сохранённым обучающим записям кандидата. Это проверка экстрактивного retrieval, не свободной нейронной генерации, семантического обобщения или истинности источника.

```bash
python tools/prepare_v19_language_curriculum.py --open-book
python run_ziwago_v19.py --dataset data/hf/squad/language_v19_v2/train.jsonl --acceptance data/hf/squad/openbook_v19/acceptance.json --checkpoint Produce/checkpoints/ziwago_v19_candidate_1794043022813953700.pt --acceptance-mode open_book --evaluate-only
```

Исходный механизмный набор open-book/v19 до ordered ranker: 195 passed, 90 subtests passed. Повторная реальная closed-book регрессия остаётся красной: 1 failed, 2 passed, 1 xfailed, шесть одинаковых отказов; порог повторов не менялся. Проверка копирования целого документа на всех шести passages даёт 0/6 и echo=1.0. Эти результаты не означают восстановления свободной генерации.

Исходный реальный `--evaluate-only` замер до ordered ranker: 5/6 по лексическим критериям, `echo=0.0`, `collapse=0.0`, но `acceptance.passed=false`. Retrieval для «Where did Super Bowl 50 take place?» выбрал предложения о результате игры вместо места проведения и пропустил «Santa Clara, California». Дословные цитаты проверены во всех шести случаях, однако это не доказывает смысловую правильность: ответ про Rollo не выражает спрашиваемую связь. Код выхода 1 означает провал допуска, не ошибку выполнения. Обучения, сохранения весов и продвижения не было; SHA-256 кандидата и старого мастера не изменились. Отчёт: [Produce/v19_open_book_evaluation_1794068314263114400.json](Produce/v19_open_book_evaluation_1794068314263114400.json).

Подготовка восстанавливает полные passages для тех же шести ранее проверенных validation-вопросов, сверяя исходные хеши, ревизию и смещения. Это не новый слепой production benchmark. Исходные train и closed-book acceptance не меняются; подготовка отказывается перезаписывать результат. `--evaluate-only` не обучает и не сохраняет веса, запрещает `--promote`, откатывает runtime после оценки и проверяет сохранность checkpoint и мастера. В генератор передаются только вопрос и документ, не reference или expected concepts; документ не записывается в authority, граф, историю или обучающие источники.

### Ordered document ranker и эфемерные роли, 2026-11-07

Текущий семифайловый механизмный набор: **272 passed, 144 subtests passed** за 49,36 с после null-aware QA, изоляции SON RNG и защиты неполного exposure ledger. Дополнительно проверены полный откат core/v19 состояния, RNG и NumPy-массивов в recall-диагностике, строгая загрузка обученного QA-модуля, исключения во вложенной оценке и сохранность источников. Повторная загрузка сохранённого кандидата тоже использует SHA-256-привязанный приватный byte snapshot; изменение его файла во время acceptance запрещает выпуск итогового отчёта. Это не отменяет реальную legacy-регрессию: 1 failed, 2 passed, 1 xfailed; 6/6 одинаковых отказов.

Обучаемый `neuroweave10/cortex/document_qa.py` использует упорядоченные word/character-признаки, контекстные свёртки и взаимодействие токенов вопроса с предложением. 594 819 параметров обучаются AdamW: это вспомогательный градиентный модуль, не доказательство обучения всего организма без глобального backprop. Нет скачанных pretrained-весов, lookup-ответов или правил для отдельных экзаменационных вопросов. При наличии обученного модуля QA возвращает только одно выбранное полное предложение с исходными смещениями и поддержкой по калиброванному annotation-score; слишком длинные входы отклоняются без скрытого обрезания. `semantic_relation_verified=False`: точная цитата и annotation-support не доказывают entailment. Старый checkpoint без модуля остаётся явно обозначенным legacy SDR/lexical retrieval.

```bash
python tools/train_v19_document_ranker.py --source D:/Ziwago18-storage/data/hf/squad/plain_text/train-00000-of-00001.parquet --output D:/Ziwago18-storage/Produce/checkpoints/document_ranker_NEW.pt --report D:/Ziwago18-storage/Produce/document_ranker_NEW.json --limit 8000 --epochs 2 --cpu-threads 4 --seed 42
```

Все output/report пути должны быть новыми. Выполненный эксперимент: [Produce/document_ranker_20261107_ordered_v1.json](Produce/document_ranker_20261107_ordered_v1.json), checkpoint `Produce/checkpoints/document_ranker_20261107_ordered_v1.pt` — 4 961 511 байт, SHA-256 `030d9d0076caf356c83531183710f20ff5fcc4aa5c0f35cfba92b0682f8f7500`. Закреплённый официальный SQuAD train прочитан из неизменяемого снимка; выбранные 8 000 уникальных question/context групп дали 7 975 пригодных групп после явно учтённых исключений. Проверены 8 016 аннотированных смещений. Fit: 12 718 positive/no-positive групп, 352 статьи, 3 180 updates, две эпохи; loss 1,57084 → 0,22583. Отдельный internal-dev: 1 511 групп из 45 других статей, positive recall@1 **79,58%**, recall@2 **91,10%** на 764 positive группах. Это выбор предложения, не доля успешно принятых ответов.

Порог 0,214402 подобран только на отдельном internal-calibration: 1 552 группы из 44 статей; balanced accuracy 75,16%, positive acceptance 78,80%, negative acceptance **28,48%**. Отрицательные контроли удаляют аннотированные предложения, но не доказывают отсутствие смыслового ответа. Защита отказа пока недостаточна. Последующий аудит обнаружил ошибку цели: v1 калибровал наличие аннотированного ответа где-либо в группе, даже если top-ranked предложение было неверным. Эти исторические calibration rates не оценивают правильность возвращённого предложения. Текущий calibrator считает положительным только `best in positive_indices`, требует обе категории и не изменяет порог при недостаточных наблюдениях; тест явно проверяет неверный top-ranked результат при наличии другого правильного предложения. Provenance теперь сверяет все holdout-семейства с dev/calibration ledger и запрещает скрытый fit/holdout overlap. Старое поле `diagnostic_policy` в неизменённом отчёте ошибочно называет internal-dev калибровочным; фактически статьи dev/calibration раздельны, текущий trainer исправляет формулировку. Official validation и acceptance не использовались для fit или настройки порога. Аудит v4 обнаружил ещё одно ограничение: historical v1–v3 сохраняли только retained question/context identities, пропуская selected, но отфильтрованные примеры. Текущий trainer сохраняет все selected identities, включая skips и dev/calibration. Исторические ledgers не исправляются задним числом: такие rankers разрешены для frozen internal-development сравнения, но текущий acceptance/training preflight отклоняет их до генерации/диагностики из-за отсутствия полного selected-identity scope.

Исправленное переобучение: [Produce/document_ranker_20261107_ordered_v2.json](Produce/document_ranker_20261107_ordered_v2.json), 345,15 с, те же train-only splits. Checkpoint `Produce/checkpoints/document_ranker_20261107_ordered_v2.pt`: 4 961 575 байт, SHA-256 `1f29e29ab8ce1ca0f4e71abbe8d2f14251f499f3966ba03be5c5d256e1d2fcd6`; provenance/state и исходный source SHA-256 проверены, v1 неизменён. Tensor weights, config и threshold в точности совпали с v1; internal-dev recall@1 остался 79,58%, @2 91,10%. Исправлена цель метрики, не продемонстрировано улучшение ранжирования или отказа.

Теперь calibration positive/negative означают правильность выбранной цитаты по аннотации: 583/969 групп, из них приняты 505/331. Balanced accuracy 76,23%, positive acceptance 86,62%, negative acceptance 34,16%; это другие знаменатели, их нельзя напрямую сравнивать с v1 group-availability rates. Annotation precision среди 836 принятых цитат — **60,41%**, coverage 53,87% на этой калибровочной выборке; это in-sample показатели настройки порога, не независимое качество и не semantic entailment. Надёжная защита отказа не достигнута, production сертификация и продвижение не выполнялись.

Расширенное train-only обучение: [Produce/document_ranker_20261107_ordered_12k_v3.json](Produce/document_ranker_20261107_ordered_12k_v3.json), 542,62 с. 12 000 selected → 11 957 retained question/context групп, 12 022 проверенных annotation offsets, fit 19 029 групп из 353 статей, 4 758 updates за две эпохи. Internal-dev 2 332 группы из 45 статей: recall@1 79,22%, @2 90,67% на 1 179 positive группах; calibration 2 304 группы из 44 других статей, threshold 0,437392. Это другая выборка, простое сравнение процентов с v2 некорректно. Checkpoint `Produce/checkpoints/document_ranker_20261107_ordered_12k_v3.pt`: 6 229 991 байт, SHA-256 `76f791dc6bf5a90142c172cae4d6bd8ff84c7e15e2719bfa19f07576f165f769`; параметров по-прежнему 594 819, рост файла включает более крупные provenance/identity ledgers. Strict state/provenance/source hash проверены. Calibration annotation precision 67,44% при coverage 41,32% — in-sample настройка, не самостоятельная сертификация.

Сравнение frozen v2/v3 на **одинаковых 1 511 internal-dev группах** из 45 статей: [Produce/document_ranker_20261107_matched_dev_v2_v3.json](Produce/document_ranker_20261107_matched_dev_v2_v3.json). Split восстановлен из закреплённого train snapshot; проверено отсутствие fit/calibration family/prompt overlap обоих моделей. Веса, RNG и checkpoint/source файлы не изменены; acceptance и official validation не читались. Повторно просмотренный dev не является слепым экзаменом, бюджеты обучения различаются.

| Метрика на одинаковом internal-dev | v2, 8k selected | v3, 12k selected |
|---|---:|---:|
| Positive recall@1 | 79,58% | 81,68% |
| Positive recall@2 | 91,10% | 90,97% |
| Annotation precision среди принятых | 63,72% | 74,36% |
| Coverage | 53,08% | 41,56% |
| Acceptance no-positive контролей | 26,37% | 13,92% |
| Принятые аннотированные цитаты | 511 | 467 |
| Принятые неаннотированные цитаты | 291 | 161 |

Есть измеренное улучшение top-1/селективной точности, но модель отвечает реже и выдаёт меньше правильных аннотированных ответов в абсолютных числах; top-2 слегка ниже. No-positive аннотация не доказывает семантического отсутствия ответа, надёжный production допуск не заявляется. Полный core-кандидат пока содержит QA v1: этот замер относится к отдельному sidecar v3, не к его продвижению или end-to-end acceptance.

Source-backed role/filler схемы регистрируются только после действительного принятого дендритного обновления. Новые slots — общие слова prompt/response; обнаруженные подстановки эфемерны, передаются CA3/arcuate без записи пользовательских связей. Замена выполняется по полным словам без учёта регистра; grounding исходного утверждения не отключается. Пять механизмных role-тестов проходят, но общий перенос на новые сущности не сертифицирован, старые checkpoint-схемы автоматически не переписывались.

Полный эксперимент из прежнего unaccepted кандидата с ranker v1: [Produce/v19_acceptance_1794071128476240400.json](Produce/v19_acceptance_1794071128476240400.json). Одна эпоха, 24 принятые пары, 530 dendritic/spiking updates, 478 PC targets, 786/786 принятых BPE-токенов; 24 вызова регистрации source-backed schema-pairs после действительных обновлений, не обещание 24 новых уникальных схем. На четырёх train-диагностиках teacher-forced NLL 0,073335 → 0,063845 nat/token, но прямой weight-only синтез остался 3/4 точных ответов и 1/126 token errors; неверный raw-ответ Lourdes не исправлен. Heldout NLL недоступен из-за словаря/позиционного окна.

Open-book повторный замер с ranker v1 ухудшился с исторических 5/6 до **2/6** по тем же лексическим критериям: Super Bowl, Pulaski и Tesla выбрали неподходящие цитаты, Rollo получил отказ. Echo=0, collapse=0, полный допуск провален; пяти дословных source-spans недостаточно для смысловой правильности. Это уже просмотренные development-вопросы, не слепая независимая сертификация. Кандидат `Produce/checkpoints/ziwago_v19_candidate_1794070802852858000.pt`, 4 542 499 733 байт, SHA-256 `e0b2bd1ae6339f9a856a8d22bfb6c48a21dd13e82e5cc16b219c4a666e826fc0`, не принят. SHA-256 кандидата, прежнего кандидата и legacy мастера независимо повторно сверены; исходные train/acceptance сохранены. Этот кандидат содержит исторический ranker v1 до исправленной калибровки, его отчёт не переписан; код выхода 1 — провал acceptance после завершённого обучения. Отдельный полный assistant train-recall: [Produce/v19_assistant_recall_roles_ordered_v1.json](Produce/v19_assistant_recall_roles_ordered_v1.json), 637,95 с. Без диагностического imprint и обучения получены те же 3/4 точных grounded train-ответов; Lourdes — отказ. Полностью сравнены и восстановлены core/v19 state, включая QA/RNG/NumPy, и повторно сверены байты candidate/master/train/reference-report. В этих четырёх ответах `emergent_stdp_role_discoveries={}`: замер не доказывает перенос на новые fillers. Это отдельная проверка запоминания, не независимое обобщение, не новый closed-book acceptance и не свидетельство ускорения.

Отдельный closed-book `--evaluate-only` замер полного кандидата: [Produce/v19_closed_book_evaluation_1794072692778745100.json](Produce/v19_closed_book_evaluation_1794072692778745100.json). **0/6**, collapse=1, echo=0, шесть одинаковых source-refusals; ограничения допуска не ослаблялись. Обучения и сохранения весов не было, candidate/master SHA-256 неизменны. Код 1 — завершённая проверка с проваленным допуском. Встроенный ranker v1 не получал документов и не использовался для ответов; улучшение отдельного QA v3 не выдаётся за восстановление closed-book памяти или свободного языка.

Заключительный read-only open-book ретест со связкой прежнего полного core-кандидата и отдельного sidecar QA v3: [Produce/v19_open_book_evaluation_1794072980205352700.json](Produce/v19_open_book_evaluation_1794072980205352700.json). **2/6**, echo=0, collapse=0,4, полный допуск провален. Pulaski и curriculum получают подходящие дословные цитаты; Super Bowl, Rollo, Tesla и вопрос о вычислительных ресурсах получают одинаковый отказ. По общей доле выполненных критериев это не улучшение против QA v1 2/6, хотя состав успешных ответов изменился; четыре отказа ухудшают collapse. Порог или goldens ради этих вопросов не менялись. Проверка связывает SHA-256 core `e0b2bd1a…826fc0` и внешнего ranker `76f791dc…5f769`; новый объединённый accepted checkpoint не создавался.

В предыдущем заключительном семифайловом прогоне прошли 242 теста и 138 подтестов за 45,03 с; восемь файлов checkpoint/source/acceptance были независимо повторно сверены по SHA-256 и неизменны. Механизмные улучшения и matched-dev выигрыш QA не отменяют провал обоих полных допусков, ошибку Lourdes, недостаточную надёжность отказа или отсутствие доказанного переноса новых role/fillers. Продвижения, перезаписи legacy мастера, обучения на acceptance и общей языковой сертификации нет.

### Null-aware QA и изоляция SON, 2026-11-07

No-positive группы раньше обучали binary support, но давали ровно нулевой градиент ranking head. Fixed-zero null candidate в listwise normalizer теперь подавляет ranking всех предложений такой группы; multi-positive supervision сохраняется. Формат весов и политика порога не менялись. Селективные метрики используют реальный per-document `assess`, а не batched logits: регрессия выявила расхождение на строгой границе порога из-за численной точности.

Отдельное train-only обучение [QA v4](file:///D:/Ziwago18-storage/Produce/document_ranker_20261107_null_12k_v4.json) завершено за 561,35 с: тот же бюджет 12k/2 эпохи/seed42, 4 758 updates и 594 819 параметров. Новый ledger содержит все 12 000 selected identities, включая 43 исключённые группы; исторические v1–v3 не переписаны. Training-report internal-dev: recall@1 79,64%, @2 91,26%, annotation precision 70,45%, coverage 45,28%. Селективные числа этого отчёта были рассчитаны до перехода на per-document `assess`; окончательное сравнение требует frozen probe, не коррекции старого отчёта. Core-веса не обучались, кандидат не продвигался.

[Frozen v3/v4 comparison](file:///D:/Ziwago18-storage/Produce/document_ranker_20261107_matched_dev_v3_v4.json) завершён за 536,27 с на одинаковых 2 332 internal-dev группах, одинаковых budgets и per-document `assess`. Это повторно просмотренный train-derived dev, не слепая сертификация; null-aware loss старой/новой моделей сравнивается текущей общей функцией, не исторической целью v3.

| Метрика | v3 | v4 |
|---|---:|---:|
| Positive recall@1 | 79,22% | 79,64% |
| Positive recall@2 | 90,67% | 91,26% |
| Annotation precision among accepted | 73,43% | 70,45% |
| Coverage | 40,35% | 45,28% |
| Принятые аннотированные цитаты | 691 | 744 |
| Принятые неаннотированные цитаты | 250 | 312 |
| Acceptance no-positive контролей | 13,70% | 17,09% |

Ranking слегка улучшился, но отказ стал менее надёжным: прирост правильных принятых цитат +53 сопровождается +62 неправильными. На одинаковой 32-group positive/no-positive выборке полный assistant согласовался с direct ranker, подтвердил смещения принятых цитат и восстановил полное core/v19 состояние: v3 accepted 16/correct 8, v4 accepted 17/correct 9. Это annotation correctness, не semantic entailment. Источники, веса и RNG ranker-only этапа неизменны; SHA-256 нового QA v4 `e281a54832139cc0aff2240af17c64b77fb513bb13f558184acfe84fe112b617`, 6 244 455 байт. Master и два предыдущих core-кандидата независимо повторно сверены и неизменны.

Первый assistant sampler исключал только известные core families/prompts (132 группы), но не core context hashes. После аудита измеритель дополнительно исключает исходные и rendered document hashes и вопросы калибровки каждого сравниваемого ranker, запрещает одинаковые checkpoint basenames. [Повтор с усиленным разделением](file:///D:/Ziwago18-storage/Produce/document_ranker_20261107_matched_dev_v3_v4_separation_v2.json) завершён за 281,60 с: исключено по-прежнему 132 группы, совпали full-dev split, 32-group assistant sample и все приведённые метрики. Direct/assistant selection и принятые spans согласованы, полное состояние и input bytes сохранены. Это подтверждение воспроизводимости и отсутствия известных пересечений, не доказательство полной независимости или ускорения; prior corpus contamination остаётся unverified. Первый отчёт не переписан.

SON uncertainty ensemble потреблял общий Torch RNG и тем самым сдвигал последующий стохастический OU-шум коры. `fork_rng` изолирует только этот ensemble: шум не отключён, grounding не ослаблен. Контролируемая tiny-core регрессия проверяет идентичность следующего logits/spikes/noise шага при одинаковом query/prefix; воспроизведение старого поведения в тесте меняет шум. Это не доказательство исправления реальной Lourdes-генерации или статистической независимости ensemble.

[Recall probe](file:///C:/Users/yarma/Desktop/Ziwago18/tools/probe_v19_assistant_recall.py) теперь заново измеряет direct decoder и полный assistant в отдельных evaluation contexts; старый direct результат сохранён только как `historical_direct`. Эталон сравнивается после декодирования, passed требует exact match и grounding. Неэффективный `--retrieval-probe` удалён: imprint внутри evaluation был no-op, а не работающей интервенцией. Lourdes находится на train index 0; эти пробы измеряют запоминание, не независимое обобщение.

[Свежий Lourdes-recall после изоляции SON](file:///D:/Ziwago18-storage/Produce/v19_assistant_recall_son_rng_v2.json), 327,64 с, одна train-строка: **0/1 exact+grounded**, grounded 0. Заново измеренный direct ответ — `It is a replica his scholar, of a.`, тот же искажённый текст, что в historical diagnostic. Полный assistant отвечает `I do not have source-supported knowledge for this question.`; все три VTE beams имеют pre-grounded=False. Grounding корректно отклоняет искажение, а механизмная изоляция SON не восстановила этот ответ. Core/v19 state полностью восстановлен, candidate/master/train/reference bytes неизменны, обучения и promotion не было. Однострочный замер нельзя сравнивать с прежними 3/4 как падение всей train-recall выборки.

Исторический tiny CA3 контрпример: при одинаковом SDR, cosine exact=1/distractor=0,8 изменение distractor salience 4,49 → 4,50 включало boost1,45 и меняло победителя на менее похожую память. В данном полном Lourdes замере `consolidated_schema_boost=False`, поэтому этот контрпример не является доказанной причиной его отказа. Ниже описано механизмное исправление, не новый результат Lourdes.

### Единый языковой путь и PC-кредит, 2026-11-07

В core введены общие `_prepare_language_query` и `_language_token_components` для обучения, teacher forcing и декодера. CA3-полюса и кортикальный prefill теперь проходят один путь; ngram-режим сохраняет исходный query. Внешние runtime role/filler bindings остаются отдельным входом: общий core-путь не доказывает идентичности всех обстоятельств подготовки ассистента. Дендритные признаки используют routed query, но индекс новой CA3-памяти — исходный stimulus, чтобы извлечённый полюс не подменял идентичность новой записи.

PC получает каждую цель с непустым prefix: отсутствующий target добавляется только в учебную копию top-48 shortlist. Inference не получает target; objective — target-inclusive shortlist contrast, не full-vocabulary likelihood. Счётчики отдельно показывают hits/extensions и recall, при отсутствии подходящих prefix recall unavailable. Диагностика компонентов включает RMS, centered RMS и target NLL/margin без каждого вклада.

Из CA3 competition удалён дополнительный жёсткий множитель1,45; плавный ограниченный gate `1 + 0.15*tanh(salience - 1)` сохранён. Тесты exact1/distractor0,8 с общими SDR и salience4,49/4,50/4,62 теперь выбирают точную память без скачка; веса и схема checkpoint не переписываются этим изменением. Семифайловый механизмный прогон после проверки train-only инструмента: **358 passed, 160 subtests passed**, 79,91 с; предыдущий этап общего core-пути дал 279 passed/160 subtests за 45,59 с. Это проверка ограниченных механизмов, не доказательство полезной свободной генерации, semantic entailment или нового role/filler transfer.

`tools/diag_v19_decoder_credit.py` больше не читает reference-report/acceptance: он проверяет manifest-linked train snapshot, явный train split всех строк, SHA-256 и бюджеты до создания ассистента. Все выбранные identities записываются до фильтрации с негрундуемым статусом `exposed_not_trained`; полнота исторического обучения остаётся неизвестной. Production chat и direct greedy без repetition controls измеряются раздельно с откатом core/v19/RNG; full reference не обрезается, сравнение происходит после генерации. Необязательный пилот допускает максимум4 строки/2 волны, сохраняет только новый экспериментальный checkpoint на D: и заново загружает его по SHA-256 до after-замера. Promotion отсутствует; результат train recall не является независимым экзаменом.

8 ноября первый пилот обнаружен приостановленным после сохранения/reload: все26 потоков процесса имели Windows wait reason `Suspended`, CPU/IO/paging не выполнялись. Это факт приостановки, не доказательство deadlock или её причины. Две учебные волны завершились, но итоговый before/after report не был создан; baseline-показатели из памяти процесса не восстановлены и не выдаются за измерение. Кандидат4 542 500 373 байт, SHA-256 `f2a587591645684407324957b8f5be885933153d33addd78fe9c639c64bc0e02` повторно проверен. С согласия пользователя остановлен только этот процесс; исходные checkpoint и кандидат не перезаписывались.

Для повторной диагностики добавлен `--evaluate-only`: одна frozen after-фаза, без baseline, учебного кодирования/gates/updates, save и повторного reload внутри процесса. Отчёт явно содержит `baseline=null` и `evaluation_mode=single_frozen_candidate`; он не подменяет потерянное before/after сравнение. Timestamped журнал различает fingerprint, context entry/restoration, production chat и direct greedy. Новые тесты проверяют однократную оценку и отсутствие обучения, запрет сочетания с train/candidate-output и выдачу restoration marker только после совпадения fingerprints. Семифайловый набор:362 passed/160 subtests,83,39 с. [Read-only оценка](file:///D:/Ziwago18-storage/Produce/v19_language_path_pc_frozen_20261108_v1.json) завершилась с exit0 за387,90 с; [журнал](file:///D:/Ziwago18-storage/Produce/v19_language_path_pc_frozen_20261108_v1.log) содержит завершение всех трёх restoration checks. Строки `Timeout (0:02:00)!` — периодические faulthandler дампы: они показывали текущие hash/beam вычисления, а не остановку этого прогона.

Полный assistant на единственном обученном Lourdes-вопросе вернул **1/1 exact+grounded**,35 BPE-токенов полного исходного ответа; выбранный neural beam pre-grounded=True завершился по SON EOS. Это проверка запоминания: документы/reference не передавались генератору, но вопрос и ответ были train-источником. Исторический отказ не заменяет отсутствующий контролируемый baseline этого пилота; из результата нельзя приписать причинный выигрыш одному изменению или заявить независимое обобщение.

Direct weight-only greedy без repetition/SON stop controls остаётся unsupported: начинается `It is a replica his scholar, of a.` и продолжается до64-токенного бюджета. Teacher-forced decoder на full reference имеет1/35 token errors, NLL4,267259/35=0,121922 nat/token; inference PC shortlist содержит34/34 eligible targets. Ошибка position4: target` of`, winner` his`, rank2, margin−0,527534. Локальные target-minus-winner вклады: dendritic+8,522903, protected core−10,235523, PC+1,185086; удаление только core из этого диагностического logits-вектора даёт target NLL0,010038. Это локальное counterfactual scoring, не проверка удаления core из полного организма или безопасного сохранения других знаний; greedy/free generation не восстановлена.

Data-only mmap-проверка metadata нового кандидата подтверждает две train waves на одной строке:104 scanned BPE tokens,70 dendritic и70 spiking token updates,68 PC targets,67 shortlist hits и1 target extension. Два вызова ontology registration не доказывают две новые схемы. В frozen-оценке updates/save/promotion отсутствуют; master, предыдущий core, новый candidate, train и manifest независимо повторно сверены по SHA-256 и неизменны. Финальный механизмный прогон: **362 passed,160 subtests passed**,82,86 с. Новый candidate остаётся experimental_unaccepted; generalization/closed-book acceptance и другие train-ответы этим однострочным замером не сертифицированы.

Через `hf` скачаны четыре открытых набора: OASST2, Dolly 15K, SQuAD и WikiText-103. WikiText-103 приведён к ожидаемому raw-формату; строгий preflight всех 11 исторических путей всё ещё имеет 10 блокеров. Среди них OASST2: данные доступны в `data/hf/oasst2`, но старый streamer ожидает другой, несепарированный tree-файл. Внешняя папка Knowledge отсутствует. Полномасштабное корпусное обучение, независимая языковая сертификация и гарантии сохранения знаний не заявляются. Парсер Knowledge возвращает фактически прочитанные записи; отсутствие каталога не превращается в 37 извлечённых агентов. Локальный каталог можно указать через `--knowledge-dir`.

`run_ziwago_v25_evotrain_marathon.py` и `train_ziwago_v20_ontogenetic_dream_trainer.py` используют тот же безопасный CLI v19: старые режимы репетиции holdout, автоматической перезаписи мастера и сертификации по времени удалены. Прежние отчёты сохранены как исторические, а не заменены новыми оценками.

Git родительского домашнего каталога не является изолированным репозиторием проекта. Изменения не коммитятся и домашняя конфигурация Git не меняется; исходники до v19 сохранены в `Produce/v19_source_backup`.

### Согласованный readout и компактные снимки, 2026-11-08

Single и batch imprint используют общий `_learn_readout_token`: одна pre-update ошибка полного inference-смешивания, chain-rule коэффициенты12 для plastic и12×0,35 для core, одна общая metaplastic attenuation. Core обновляется только при прежнем пороге значимости; защита plastic усиливается после обоих обновлений. Инференс `12*(plastic + 0.35*core + SWR + bank) + auxiliary` не изменён, protected core не отключён. Счётчики показывают фактические readout/core updates и ограниченные шаги.

Совместный projected logit gain ограничен2 по верхней границе softmax curvature1/2. Аналитические, autograd и finite-difference тесты подтверждают направление и уменьшение token NLL при фиксированных SWR/bank/auxiliary contribution vectors. Это не гарантия полного rollout: смена top-256 bank shortlist может изменить сам residual-вектор и увеличить NLL; согласование bank shortlist/objective остаётся отдельной задачей. Проверка удержания ортогонального synthetic feature-context не сертифицирует сохранность реальных знаний.

Три языковые матрицы больше не резервируют4096 пустых колонок при создании: начальная ёмкость0, рост блоками256. Старые padded runtime checkpoints компактируются при загрузке без изменения активных весов; legacy schema, dense-reference logits, shared-storage aliases и full runtime/RNG rollback сохранены. У исходного кандидата10885 active columns: ёмкость12288→11008, банки1 207 959 552→1 082 130 432 байта, экономия125 829 120 байт (120 MiB) RAM; это не факторизация активных весов. Export checkpoint собирается из одного authoritative runtime snapshot вместо повторного предварительного клонирования больших модулей; fp16-проекции остаются отделены от полного fp32 runtime.

Диагностика получила `--train-limit` и `--generation-limit`: можно обучать только часть выбранных train-строк и измерять остальные как `untrained_in_this_pilot` controls. Все selected identities, включая контроли, остаются в exposure ledger; статус exposure не предоставляет factual grounding. Все bounded generations одной фазы выполняются до teacher forcing, первая генерация не пересчитывается дважды. Полный baseline теперь сохраняется в локальном журнале до обучения. Новые guards, subset isolation, источники, state restoration и checkpoint contracts проверены: **396 passed,226 subtests passed**,90,38 с.

Отдельный [before/after pilot](file:///D:/Ziwago18-storage/Produce/v19_joint_readout_retention_20261108_v1.json) завершился успешно за1483,39 с (24,72 мин):2 selected train-строки, обучение только первой,1 контроль без обучения в этом пилоте,2 waves,2 CPU threads/1 worker/64 tokens и generation обеих строк в каждой фазе. Выполнено104 pair BPE tokens,70 readout/70 protected core/70 spiking updates и68 PC updates с68/68 shortlist hits; все70 readout steps ограничены max gain2. Это не обновление по независимому экзамену: обе строки относятся к одной семье University_of_Notre_Dame, историческая contamination unknown.

Полный assistant дал **2/2 exact AND grounded до и после**; это сохранение ранее доступного recall, не рост accuracy. На обучаемой35-token строке teacher-forced NLL4,267259→4,119497 (−3,46%), но1/35 token errors сохраняется: target` of`, winner` his`, rank2, margin−0,527534→−0,328756. Protected core всё ещё даёт отрицательный target-minus-winner вклад−10,232571; большая часть изменения этого локального margin связана с PC, поэтому эффект нельзя приписать одному refactor. На14-token контроле0 errors и NLL9,298283e−6 неизменны, production-ответ и весь raw greedy-текст также неизменны; полный component diagnostic report не идентичен. Direct greedy **0/2 exact+grounded в обеих фазах**, оба прогона расходуют64-token budget; контроль начинает правильным полным предложением, затем продолжает неподдержанным текстом. Free generation/правильная остановка не восстановлены, независимое обобщение не сертифицировано.

Новый [experimental_unaccepted candidate](file:///D:/Ziwago18-storage/Produce/checkpoints/ziwago_v19_joint_readout_retention_20261108_v1.pt) имеет4 416 495 457 байт, SHA-256 `ba719637dc0f02af520a1294932223dca44c4c2ae8cfc8b09015663bcff21023`; он отдельно сохранён и SHA-bound перезагружен. [Журнал](file:///D:/Ziwago18-storage/Produce/v19_joint_readout_retention_20261108_v1.log) содержит13 успешных core/v19/RNG restoration checks, полный baseline и process_finished=success. После завершения независимо повторно проверены пять SHA-256: master, исходный candidate, новый candidate, train и manifest — все совпали; source файлы и master неизменны, promotion отсутствует. Следующие нерешённые узлы — согласование bank shortlist/objective и обучение естественному завершению ответа, без target/reference-зависимых inference обходов.

### Full-support sparse bank и обучаемое завершение, 2026-11-08

Bank residual теперь оценивается для всего активного словаря, без target-зависимого inference и top-256 scoring mask. Ограничение256 токенов сохраняется только для обновляемых параметров. Bank learner получает свежие post-matrix base logits и минимизирует `NLL(base + 3*tanh(raw_bank))` с chain-rule производной `3*(1-tanh(raw_bank)^2)`. Split и до12 backtracking-попыток проверяются против исходного routed loss до атомарной записи; при отказе банки, domains и RNG не меняются, eviction отсутствует. 257-token regression воспроизводит рост NLL старого shortlist-пути и проверяет снижение после устранения этого скачка. Это локальная проверка при фиксированных query/features/SWR/auxiliary, не гарантия полного rollout, PC shortlist или удержания знаний.

Sparse scoring обходит хранимые entries вместо `V*128` lookup и сохраняет порядок признаков и native compensated `sum`. На синтетических10885 tokens/128 features/32768 entries медиана пяти замеров0,222594→0,011219 с, ускорение19,84× при точном совпадении scores. Это не измерение ускорения всего ассистента; serialized state и memory-entry accounting не расширены.

Core добавляет ровно одну EOS-цель50256 после полного content-only ответа. Readout, protected core, spiking и PC используют общий token path; content/semantic/provenance не включают control token. Ответ должен оставлять EOS-позицию; встроенный EOS и заполненное positional window отклоняются до обучения. V19 trainer, diagnostic pilot и sleep replay согласованы с этим контрактом. Legacy worker больше не создаёт обрезанную42-word границу: длинный ответ пропускается целиком; старые данные и checkpoints не переписываются.

Greedy и beam прекращаются по предсказанному EOS без его вывода в текст, arcuate filler или SON. Teacher-forced termination измеряется отдельно от content NLL/errors. Причины learned EOS, фактический SON, punctuation control и budget различаются. Историческая метка «SON EOS» могла включать принудительную остановку по пунктуации даже без SON verdict; старые отчёты сохранены, их метка не доказывает learned stopping.

Семифайловая интеграционная перепроверка после исправления compensated-sum округления: **460 passed,236 subtests passed**,106,58 с. Это механизмные тесты, не сертификация языка.

[Train-only full-support/EOS пилот](file:///D:/Ziwago18-storage/Produce/v19_fullsupport_eos_retention_20261108_v1.json) завершился за2322,91 с (38,72 мин):2 selected строки одной article family,1 training/1 control,2 waves,2 CPU threads/1 worker/64 tokens. Actual training:104 scanned input/content BPE,70 content+2 EOS targets,72 readout/protected-core/spiking updates,70 PC targets/68 shortlist hits/2 training-only extensions. Bank принял8 обновлений и отклонил64, backtracking0; конкретные причины отказов в этом историческом отчёте не записаны и не выводятся из числа отказов.

Production остаётся **2/2 exact AND grounded before/after**, оба raw greedy-текста неизменны, **0/2 exact+grounded**,64-token budget exhaustion без learned EOS. Обучаемые35 content tokens: NLL4,119501→3,987017 (−3,22%),1 ошибка сохраняется, target` of`/winner` his`, margin−0,328756→−0,148014. EOS появился в словаре, но на правильной35-token границе имеет rank137/NLL8,488839, winner` years`; у14-token контроля EOS rank3257/NLL13,682323, winner` by`. Контрольные content NLL9,298283e−6 и0 errors сохранены; полные component reports не идентичны. Обучаемая остановка, свободная генерация и независимое обобщение не восстановлены; односемейный exposed control с historic contamination unknown не является heldout.

[Новый experimental_unaccepted checkpoint](file:///D:/Ziwago18-storage/Produce/checkpoints/ziwago_v19_fullsupport_eos_retention_20261108_v1.pt):4 416 554 241 байт, SHA-256 `377d978301905f71acf47b8dc28e2e357fd522db7cdc619cbb0d3c472e715990`, отдельный save и SHA-bound reload. [Журнал](file:///D:/Ziwago18-storage/Produce/v19_fullsupport_eos_retention_20261108_v1.log) содержит13 успешных restoration checks, completed и process_finished=success. После выполнения независимо сверены master/source/newcandidate/train/manifest SHA и пять launch source-code SHA; все совпали. Acceptance/reference не читались, promotion отсутствует, старые артефакты сохранены.

### Бюджетная учебная маска sparse bank, 2026-11-08

Прямая проба выявила дефект: при default `max_active_features=128` и `max_token_ids=256` полная учебная маска требовала12 585 728 учтённых байт (bank entries + сохранённый per-domain gradient) против лимита8 388 608 и отклонялась `memory_budget_exhausted` даже в пустом роутере. Это объясняет, почему полный mask не мог обучаться при прежнем учёте; конкретные причины64 отказов исторического v1-пилота тем отчётом не записаны и задним числом не приписываются.

Исправление: `learn_sparse` теперь детерминированно уменьшает только учебную маску под точный prospective-учёт памяти. Цель сохраняется первой (если одна цель не помещается — атомарный отказ без мутаций); остальные запрошенные токены добавляются по убыванию энергии градиента с тай-брейком по возрастанию token ID, пока хватает бюджета; уже существующие bank entries дешевле новых, старый gradient текущего domain не учитывается дважды. Полный словарь оценки, `base + 3*tanh(raw)` objective, исходный routed loss как baseline, conflict split с пересчётом градиента на свободной ветке и до12 backtracking-попыток не изменены; eviction и рост бюджета отсутствуют. Serialized state и checkpoint fields не расширены. Проба default:256→170 токенов, NLL5,545177→2,366937,8 358 656≤8 388 608 байт.

Телеметрия теперь возвращает `update_tokens_requested/selected`, `update_mask_limited`, `gradient_entries_selected` и причину каждого отказа; core агрегирует их в `bank_limited_mask_updates`, `bank_update_tokens_*`, `bank_gradient_entries_selected` и раздельные счётчики memory/capacity/objective/no-signal/other отказов, trainer/diag пишут их в отчёт и журнал волн. Добавлены5 механизмных тестов маски (детерминизм порядка и тай-брейк, переиспользование записей без двойного учёта, split с уменьшенной маской на свободной ветке при неизменном original baseline, checkpoint continuation). Финальный семифайловый набор: **467 passed,242 subtests passed**,107,46 с. Это механизмная проверка; влияние на реальную генерацию измеряется отдельным пилотом, не заявляется заранее.

### Пилот бюджетной маски v2, 2026-11-08

[Train-only пилот v2](file:///D:/Ziwago18-storage/Produce/v19_budget_mask_retention_20261108_v2.json) стартовал от v1-кандидата `377d9783…` с исправленным роутером и завершился за1436,69 с (23,94 мин):2 selected строки одной article family University_of_Notre_Dame,1 training/1 exposed control,2 waves,2 CPU threads/1 worker/64 tokens. Actual training:104 scanned BPE,70 content+2 EOS targets,72 readout/protected-core/spiking updates (все step-limited),70 PC targets/68 shortlist hits/2 extensions. Bank теперь принял **26 обновлений и отклонил46** (v1:8/64); все46 отказов записаны как `memory_budget_exhausted` (цель одна не помещалась), capacity/objective/no-signal/other=0, backtracking0. Ограниченных масок68, запрошено18432 токенов/выбрано1317/gradient entries42144; по волнам selected1304→13 (банк заполняется, маски сжимаются). Счётчики журнала волн и отчёта совпадают точно; единственное различие — производный `pc_shortlist_recall` только в отчёте.

Обучаемая35-token строка: **1 ошибка→0**, NLL3,987017→2,467765 (−38,10%); прежняя ошибка pos4 target` of`/winner` his` устранена. EOS на правильной35-token границе: **rank137→21**, NLL8,488839→5,410769, margin−7,488747→−4,128326, и EOS теперь входит в PC shortlist (`target_in_pc_candidates` false→true); winner всё ещё` years`, не EOS. Контрольные14 tokens:0 errors и NLL9,298283e−6→9,179074e−6 сохранены, EOS rank3257/NLL13,68 не изменился; полные component reports слегка различаются.

Production остаётся **2/2 exact AND grounded before/after** (row1 son_eos, row2 sentence_boundary_control; тексты идентичны, полные result dicts не идентичны). Raw greedy обучаемой строки **изменился**: вместо бессвязного текста появилось связное правильное первое предложение «It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858.», затем хвост вырождается; `finish_reason=budget`, `stopped_by_learned_eos=false`, **0/2 exact+grounded**. Raw greedy контроль-строки идентичен до/после (текст и token_ids). Обучаемая остановка и полная свободная генерация не достигнуты; односемейный exposed control с historic contamination unknown не является heldout, одновременные PC/core/bank updates не позволяют causal attribution к одному механизму.

[Новый experimental_unaccepted checkpoint v2](file:///D:/Ziwago18-storage/Produce/checkpoints/ziwago_v19_budget_mask_retention_20261108_v2.pt):4 416 559 757 байт (+5516 к v1), SHA-256 `cbeb4fe7f84901f6e86a8344ed303557e4b115dbf9409c226f11025fc9583b57`, отдельный save и SHA-bound reload. [Журнал](file:///D:/Ziwago18-storage/Produce/v19_budget_mask_retention_20261108_v2.log):13 restoration checks, candidate_reloaded, completed token_errors=0, process_finished=success. Независимо сверены master `cb97c869…`/v1-source `377d9783…`/v2-candidate/train `50d56ee1…`/manifest `dfe4d2bf…` и пять launch source-code SHA — все совпали, код оставался заморожен. Acceptance/reference/evaluation артефакты не читались, independent_acceptance/calibration/promoted=false, старые артефакты сохранены.

### Cross-family пилот v3, 2026-11-08

[Train-only пилот v3](file:///D:/Ziwago18-storage/Produce/v19_crossfamily_retention_20261108_v3.json) стартовал от v2-кандидата `cbeb4fe7…`, завершился за1986,65 с (33,11 мин):4 selected строки, обучение первые2 (Notre Dame), **cross-family exposed control строки3-4 другого семейства Beyoncé**,4 wave calls (2 строки×2 волны),2 CPU threads/1 worker/64 tokens, generation-limit4. Actual training:164 scanned BPE,98 content+4 EOS targets,102 readout/protected-core/spiking updates,98 PC targets/96 shortlist hits/2 extensions. Bank принял **26/отклонил76** (все `memory_budget_exhausted`, прочие причины0, backtracking0); ограниченных масок102, запрошено26112/выбрано **всего26**/gradient entries832. Резкое падение selected против v2 (1317→26) показывает: банк, унаследованный от v2, насыщен по лимиту8 MiB, новые entries почти не помещаются — дальнейший bank-вклад минимален, обучение идёт через readout/core/PC.

Teacher forcing: обучаемая ND строка1 NLL2,467765→1,878337 (−23,89%),0 errors, EOS rank **21→9**/NLL5,4108→4,1312/margin−4,1283→−2,6267, winner всё ещё` years`; ND строка2 (теперь обучаемая) content NLL≈9e−6 сохранён, EOS rank **3257→171**/NLL13,68→9,22. Cross-family контроль Beyoncé: **0 errors сохранены на обеих строках**, content NLL0,565151→0,545873 (−3,42%) и0,628005→0,675674 (+7,59%) — малый дрейф без катастрофического забывания; EOS rank2811/2793 почти не изменился. Важно: контрольные Beyoncé-ответы уже были известны модели до v3 (baseline0 errors/NLL≈0,56-0,63), поэтому это **retention уже существующего знания, не generalization** нового ND-обучения на Beyoncé.

Production **4/4 exact AND grounded before/after** (ND×2+Beyoncé×2; тексты идентичны, result dicts различаются). Raw greedy **все4 текста идентичны до/после** (текст и token_ids): хотя teacher NLL/EOS rank улучшились, argmax последовательность не перевернулась — EOS rank9 всё ещё проигрывает` years`, банк насыщен, `finish_reason=budget`, `stopped_by_learned_eos=false`, **0/4 exact+grounded**. Обучаемая остановка и полная свободная генерация не достигнуты; EOS rank монотонно улучшается по линейке v1→v2→v3 (137→21→9), но до rank1 не доведён. Cross-family control не является независимым heldout (exposed, historic contamination unknown, знание уже существовало).

[Новый experimental_unaccepted checkpoint v3](file:///D:/Ziwago18-storage/Produce/checkpoints/ziwago_v19_crossfamily_retention_20261108_v3.pt):4 416 572 907 байт (+13150 к v2), SHA-256 `94aa890693e264b8ae44b2c8b179841feb4fe42b416a43f7ec5de638b2d155ce`, отдельный save и SHA-bound reload. [Журнал](file:///D:/Ziwago18-storage/Produce/v19_crossfamily_retention_20261108_v3.log):25 restoration checks, candidate_reloaded, completed token_errors=0, process_finished=success. Независимо сверены master/v2-source/v3-candidate/train/manifest и пять launch source-code SHA — все совпали, код заморожен. Acceptance/reference/evaluation не читались, independent_acceptance/calibration/promoted=false, исторические артефакты сохранены.

### Pruning пилот v4, 2026-11-08

[Train-only пилот v4](file:///D:/Ziwago18-storage/Produce/v19_pruning_retention_20261108_v4.json) стартовал от v3-кандидата `94aa8906…`, завершился за 2214,49 с (36,9 мин): тот же cross-family split (4 selected, обучение ND строки 1-2, control Beyoncé строки 3-4), 4 wave calls, 2 CPU threads/64 tokens. **Новый механизм: pruning near-zero bank entries** (threshold 1e-4, автоматически при загрузке старого checkpoint через миграцию). Bank принял **102/отклонил 0** (все причины 0, backtracking 0); **pruned 220 024 entries** за 4 волны (126976+15358+61205+16485); запрошено 26112/выбрано **8445**/gradient entries 270240. Против v3 (26 accept/76 reject, selected 26, pruned 0): pruning освободил память от near-zero entries, устранил saturation, позволил банку принять все обновления.

Teacher forcing: **все 4 строки 0 errors** (ND обучаемые + Beyoncé control). ND строка 1 NLL 1,878→1,519 (−19%), **EOS rank 9→1 (winner=EOS, margin 0)** — впервые достигнут rank 1. ND строка 2 content NLL≈7e−6 сохранён, EOS rank 6. Cross-family Beyoncé: 0 errors сохранены, NLL 0,546→0,575 (+5%) и 0,676→0,676 (стабильно); EOS rank 2811/2750 (не обучались, ожидаемо далеко).

**Raw greedy ND строка 1: `stopped_by_learned_eos=True`, `finish_reason=learned_eos`, 35 tokens, coherent complete sentence** ("It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858."), **grounded=True, semantic_closure=1.0**. Baseline имел тот же первый sentence но продолжал мусором до budget (64 tokens, learned_eos=False). Это **первый подтверждённый случай полезной свободной генерации с обученной остановкой** в линейке v19. Остальные 3 строки: первый sentence coherent, затем деградация, finish_reason=budget, learned_eos=False — stopping обобщён только на 1 из 2 обученных строк. Production **4/4 exact AND grounded** (beam search находит эталон через authority).

[Новый experimental_unaccepted checkpoint v4](file:///D:/Ziwago18-storage/Produce/checkpoints/ziwago_v19_pruning_retention_20261108_v4.pt): 4 416 573 937 байт (+1030 к v3), SHA-256 `e3e8838fd7fdb46512fe16d649c2887443f8db68bf48dfa07ca269322de198ac`. Master `cb97c869…` неизменен. independent_acceptance/calibration/promoted=false. Механизмные тесты: **472 passed, 242 subtests** (включая 5 pruning тестов + миграционный регрессионный). Два.failed запуска (AttributeError prune_threshold) исправлены миграцией в организм `load_runtime_state`; логи сохранены как `_failed_attempt1/2.log`.

Тренд EOS rank по линейке: **137→21→9→1**. Тренд bank selected: 1317→26→**8445** (pruning снял saturation). Тренд learned stopping: 0/4→0/4→0/4→**1/4**. Воспроизведение первой обученной строки с EOS подтверждено, порождение не измерено: stopping не проверен как обобщение на незнакомые ответы; свободный текст после первого sentence деградирует; это development/retest, не blind certification.

### Stopping пилот v5, 2026-11-08

[Train-only пилот v5](file:///D:/Ziwago18-storage/Produce/v19_pruning_stopping_20261108_v5.json) стартовал от v4-кандидата `e3e8838f…`, завершился за 2157,97 с (36,0 мин): тот же cross-family split, **train-waves=4** (8 wave calls, вдвое больше v4). Bank: **204 accept/0 reject**, pruned **253 698**, selected **10 153**, memory rejections 0. Teacher: все 4 строки 0 errors; ND row 0 NLL 1,519→1,241 (−18%), EOS rank 1 сохранён; **ND row 1 EOS rank 6→1** (winner=EOS). Beyoncé control: 0 errors, NLL стабильно (0,552/0,675), EOS rank 2811/2728.

**Raw greedy: обе обученные ND строки — `stopped_by_learned_eos=True`, `finish_reason=learned_eos`, grounded=True, closure=1.0:**
- ND row 0 (35 tokens): "It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858."
- ND row 1 (14 tokens): "Next to the Main Building is the Basilica of the Sacred Heart."

Baseline (v4): ND row 1 имел EOS rank 6, finish_reason=budget, деградация после первого sentence. После 4 дополнительных волн: rank 1, полная остановка, coherent complete answer. Control Beyoncé (не обучались): stopping не достигнут (ожидаемо). Production 4/4 exact AND grounded.

[Новый experimental_unaccepted checkpoint v5](file:///D:/Ziwago18-storage/Produce/checkpoints/ziwago_v19_pruning_stopping_20261108_v5.pt): 4 416 581 821 байт, SHA-256 `1171e764231a17696ce016f553b28b91d96ba5591eb2df67a10bb84c4bc04662`. Master неизменен. independent_acceptance/calibration/promoted=false.

**Тренд learned stopping: 0/4→0/4→0/4→1/4→2/4 (2/2 обученных = 100%).** Нейронное завершение ответа достигнуто для обученного семейства. Следующий рубеж: расширение обучения на больше семейств и проверка stopping на новых данных (не только retention). Это development/retest, не blind certification; cross-family control не является generalization.

### Multi-family пилот v6, 2026-11-08

[Train-only пилот v6](file:///D:/Ziwago18-storage/Produce/v19_multifamily_stopping_20261108_v6.json) стартовал от v5 `1171e764…`, завершился за 2305,8 с: **train-limit=4** (все 4 строки обучаются, включая Beyoncé впервые), train-waves=4, 16 wave calls. Bank: 320 accept/200 reject (169 memory + 31 no_signal), pruned 216956, selected 8633. **Beyoncé row 3 (line=4, 49 tokens) полностью заблокирован банком** — 0 updates каждую волну (50 rejections: 43 memory + 7 no_signal); readout/core обучаются без банка.

Teacher: все 4 строки 0 errors. ND: EOS rank 1 сохранён, NLL 1,241→1,106. **Beyoncé: EOS rank 2811→5 (row 2) и 2728→13 (row 3)** — драматическое улучшение за 4 волны; NLL 0,552→0,212 (−62%) и 0,675→0,639. Raw greedy: ND 2/2 learned_eos=True (сохранён); **Beyoncé 0/2 stopping** (rank 5/13 ещё не rank 1, finish=budget, деградация после первого sentence). Production 4/4 exact AND grounded.

Candidate v6: 4 416 610 931 байт, SHA-256 `04f92c88b548e088c446d66ba399bc31620e3a716c38b0e3ef68104544f811da`. Master неизменен. Stopping 2/4; Beyoncé прогрессирует но требует больше волн. Bank saturation для длинных ответов (49 tokens) — ограничение 8 MiB; pruning не помогает если все entries значимые (>1e-4).

### Multi-family stopping v7-v9, 2026-11-08

**v7** (от v6, 8 волн): Beyoncé row 2 EOS rank 5→**1**, stopping **3/4**. Row 3 rank 13→6 (readout-only, bank blocked). Candidate `85cb020b…`.

**v8** (от v7, 8 волн): row 3 rank **6 не изменился**; отсутствие улучшения в этом прогоне не доказывает потолок возможностей readout. Bank budget 8 MiB блокировал обновления row 3 (0 updates, 50 rejections/wave). Candidate `f6723721…`.

**[v9](file:///D:/Ziwago18-storage/Produce/v19_bank16_stopping_20261108_v9.json)**: `after.generations[*].direct` подтверждает воспроизведение обученных строк с learned EOS и grounding; `after.rows[*].termination` показывает EOS rank 1. Bank: 1040 updates, 0 rejections, 8 151 094 entries pruned (`pilot.learning_counts.{bank_token_updates,bank_token_rejections,bank_pruned_entries}`), bank-memory-mib=16 и train-waves=8 (`budgets`). Отказы по памяти в этом пилоте отсутствуют; это не доказательство отсутствия будущего насыщения и не доказательство порождения.

Тексты v9 (raw greedy, learned_eos):
- ND row 0 (35 tok): "It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858."
- ND row 1 (14 tok): "Next to the Main Building is the Basilica of the Sacred Heart."
- Beyoncé row 2 (28 tok): "Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time."
- Beyoncé row 3 (49 tok): "Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress."

[Checkpoint v9](file:///D:/Ziwago18-storage/Produce/checkpoints/ziwago_v19_bank16_stopping_20261108_v9.pt): 4 417 331 021 байт, SHA-256 `4daad1a4f1fc417b1b3810d7ff579dec058dd5424020d870960409a3cc3726e5`. Master `cb97c869…` неизменен. independent_acceptance/calibration/promoted=false. Механизмные тесты: **473 passed, 242 subtests**.

v9 подтвердил **воспроизведение дословно обученных ответов и выученный EOS, а не порождение новых ответов**. Пересчёт сохранённых строк текущим whitespace-only сравнением: direct `replay_exact` baseline 3/4 → after 4/4; сырой exact 0/4 → 0/4 из-за начального пробела. Production `replay_exact` 4/4 → 4/4, тексты идентичны. Источник: [исправленный отчёт](file:///D:/Ziwago18-storage/Produce/v19_replay_compose_falsification_20261109.json), `historical_t1_t2.phases.{baseline,after}.{direct,production}.{raw_exact,replay_exact}` и `production_texts_identical_before_after`; это пересчёт исторических байтов, не новый прогон весов. `compose_exact` в исходном пилоте не измерялся, независимое обобщение не установлено.

### Воспроизведение против порождения: фальсифицирующая проба, 2026-11-09

Без обучения на замороженном v9 оригинальный префикс «Next to the Main Building» продолжен « is the Basilica of the Sacred Heart.»: exact=true, grounding=true, EOS на позиции 14. Перефразированные префиксы «The Basilica of the Sacred Heart stands» и «Beside the Main Building stands» дали несвязные посторонние продолжения до бюджета: `compose_exact` 0/2, grounded 0/2, learned EOS 0/2. Источник: [новое измерение](file:///D:/Ziwago18-storage/Produce/v19_replay_compose_falsification_20261109.json), `baseline[0]`, `after[0:2]`, `semantic_review`. Полные префиксы, продолжения, token IDs, EOS-позиции и grounding сохранены; seed и состояние/RNG восстановлены после каждого замера (`evaluation_state_and_rng_restored`).

Вывод ограничен этой пробой: воспроизведение подтверждено, порождение провалено. Правильные ожидаемые перефразы отсутствуют в наблюдаемом train-ledger, но историческая полнота ledger неизвестна. Неизменённый authority отвергает и сами правильные перефразы (`after[*].expected_answer_grounding`), поэтому отказ grounding не используется отдельно как доказательство неспособности порождать. Семантический провал установлен по фактически несвязным продолжениям. Ни обучения, ни продвижения в этой пробе нет (`training_performed=false`, `promoted=false`).

### Минимальная рекуррентная абляция и ограничение шага, 2026-11-09

[Гипотеза и параметры](file:///D:/Ziwago18-storage/Produce/v19_prefix_recurrence_preregistration_20261109.json) записаны до реализации: prefix-state `h`, накопленный по всему префиксу, добавляется с фиксированным gain в прежний query-вектор (`state_update`, `parameters_fixed_before_implementation`). Это изолированная D:-only inference-ablation без обучения и изменения production-кода, не подключение обученной hippocampal `w_ca3_seq`.

[Свежий baseline/after](file:///D:/Ziwago18-storage/Produce/v19_prefix_recurrence_step_20261109.json): `metrics.{baseline,after}.compose_exact_matches` 0/2 → 0/2, `compose_cases=2`; оригинальный replay-control остаётся exact с EOS на позиции 14 (`baseline[0]`, `after[0]`), его текст и токены идентичны. Новые продолжения изменились, но остались несвязными и не grounded (`semantic_review`). `hypothesis_worked=false`: этот необученный минимальный адаптер не сработал; это не доказательство против обучаемой рекуррентности.

На одном и том же текущем EOS-шаге (`step_trace.source_line=4`, `target_position=49`, `f_mod=2.2`) настоящий локальный helper протрассирован с перехватом всех записей. `limiter_factor=0.00679104994805849`, то есть ограничитель **режет амплитуду на 99.32089500519416%** (`amplitude_removed_percent`): plastic requested-норма 10.494282722473145 → 0.07126719504594803; после одинакового shield — 0.6753224730491638 → 0.004586146678775549. Protected-core requested-норма 2.8087642192840576 → 0.01907445304095745; shielded — 0.18074780702590942 → 0.001227468834258616 (`step_trace.captured_updates[0:2].{requested_norm_before_limiter,requested_norm_after_limiter,shielded_norm_before_limiter,shielded_norm_after_limiter}`). Это контрфактическая трасса текущего v9, не реконструкция исторического update: нормы в исходных волнах не записывались (`f_mod_scope`). Урезание нормы установлено, вред обучению не доказан (`learning_harm_established=false`); `parameter_updates_written=0`.

Core/v19/RNG восстановлены после всех замеров, checkpoint неизменен, обучения и promotion нет (`evaluation_state_and_rng_restored`, `checkpoint_unchanged`, `training_performed`, `promoted`). Механизмный набор и регрессии новых проб: **483 passed, 242 subtests passed**, источник [финальный журнал](file:///D:/Ziwago18-storage/Produce/v19_t1_t5_tests_final_v2_20261109.log#L10). Разрыв между воспроизведением и порождением пока не закрыт.

### Один полюс × изменённый префикс: причинная проба, 2026-11-09

[Предрегистрация N1/N3](file:///D:/Ziwago18-storage/Produce/v19_same_pole_preregistration_20261109.json) зафиксировала до замера критерий полного source-entailed ответа, допустимые грамматические правила и сравнение логитов/токенов (`coherence_criterion`, `causal_interpretation`). [Матрица на замороженном v9](file:///D:/Ziwago18-storage/Produce/v19_same_pole_factorial_20261109.json) использует один неизменный вопрос Notre Dame и только подставляет первый query-полюс. Полюса действительно различны: original CA3 370 и foreign Beyoncé CA3 372 (`poles.{original,foreign}.{stored_ca3_index,vector_sha256,source_line}`). Это не тест перефразированного вопроса и не новый обучающий прогон.

Префиксы: literal — «Next to the Main Building»; a — «The Basilica of the Sacred Heart stands»; b — «Beside the Main Building stands» (`cells[*].prefix`).

| Ячейка | Префикс | Полюс CA3 | Токены свободного продолжения | EOS-позиция | Полный source-entailed ответ |
|---|---|---:|---:|---|---|
| C1 | literal | 370 | 9 | 14 | да |
| A1 | literal | 370 | 9 | 14 | да |
| A2 | literal | 372 | 59 | нет | нет |
| C2a | a | 370 | 56 | нет | нет |
| B1a | a | 370 | 56 | нет | нет |
| B2a | a | 372 | 56 | нет | нет |
| C2b | b | 370 | 58 | нет | нет |
| B1b | b | 370 | 58 | нет | нет |
| B2b | b | 372 | 58 | нет | нет |

Все значения таблицы взяты из `cells.<ячейка>.{injected_pole.stored_ca3_index,generated_text_token_count,eos_position,coherence_pass}`. Полные продолжения, token IDs и раздельный runtime/measurement grounding сохранены в `continuation`, `continuation_token_ids`, `all_token_ids`, `runtime_grounding`, `measurement_grounding`; first-free readout/combined логиты сохранены в SHA-привязанных NPZ (`first_free_logits`).

Natural controls и явная подстановка original идентичны по всем токенам, байтам ответа и первым свободным логитам (`contrasts.{C1_vs_A1,C2a_vs_B1a,C2b_vs_B1b}`). A1 воспроизводит обученную строку, A2 нет; B1 и B2 оба дают 0/2 связных ответов (`metrics.{original_paraphrase_passes,foreign_paraphrase_passes,paraphrase_cases}`), но это **не некаузальный полюс**: B1a/B2a меняют readout max-abs на 31.879884719848633 и combined на 31.849258422851562; B1b/B2b — на 29.19536781311035 и 29.35811424255371, rollout token IDs различаются (`contrasts.{B1a_vs_B2a,B1b_vs_B2b}.{first_free_logits_difference,same_token_ids}`). Узкий verdict — `episode_reading_or_prefix_readout_failure`: правильного полюса достаточно для literal replay, но недостаточно для этих новых префиксов. Разрыв порождения не закрыт; независимое обобщение здесь не измерялось.

Диагностический гейт принимает 4/4 положительных случаев и отвергает 10/10 отрицательных (`gate_metrics`): корректные полные ответы/перефразы, мусор, другая basilica/дата, отрицание, частичное имя и неподдержанная добавочная клауза (`gate_corpus`). Каждое решение содержит SHA-bound source path/line/quote, matching rule, fact tuples и justification; `runtime_authorization=false`. Гейт не зарегистрирован в runtime и не разрешает production-ответы.

Фактический `assistant.chat` до/после диагностики дал побайтно идентичный ответ «Next to the Main Building is the Basilica of the Sacred Heart.» и одинаковые safety-поля (`runtime_baseline`, `runtime_after`, `runtime_visible_bytes_identical`, `runtime_safety_fields_identical`). В обоих runtime-прогонах `stopped_by_learned_eos=false`, `generation_finish_reason=sentence_boundary_control`; это нельзя смешивать с EOS-позицией прямого greedy A1. State/core/v19/RNG восстановлены после каждого замера; текущие SHA master/candidate/train/manifest/code/spec/NPZ перепроверены, обучения, runtime-патча, настройки limiter и promotion нет (`protected_inputs`, `source_hashes`, `evaluation_state_and_rng_restored`, `master_unchanged`, `checkpoint_unchanged`, `training_performed`, `production_patch`, `limiter_change`, `promoted`). Полный механизмный набор и новые регрессии: **523 passed, 242 subtests passed** — [журнал](file:///D:/Ziwago18-storage/Produce/v19_n1_n3_full_tests_20261109.log#L10); пропусков 0 — [JUnit](file:///D:/Ziwago18-storage/Produce/v19_n1_n3_full_tests_20261109.xml), `testsuite.skipped`.

### Аудит номинального бюджета волн, 2026-11-09

[Отдельная сводка N2](file:///D:/Ziwago18-storage/Produce/v19_wave_budget_audit_20261109.json) проверила исходные stopping v1…v9 и joint-readout: собственных pre/post limiter-норм или коэффициентов в них нет (`historical_report_count=10`, `own_historical_factor_or_norm_report_count=0`, `rows[*].own_norm_and_factor_evidence`). Счётчики вызовов сверены с длиной `pilot.wave_stats`; SHA исходных отчётов сохранены, исторические файлы не перезаписаны (`input_hashes`, `all_audited_inputs_unchanged`).

| Пилот | Номинальные проходы | Фактические row-wave calls | Условные limiter-only амплитудные эквиваленты вызовов | Строка сводки |
|---|---:|---:|---:|---|
| fullsupport v1 | 2 | 2 | 0.0135821 | `rows[0]` |
| budget-mask v2 | 2 | 2 | 0.0135821 | `rows[1]` |
| crossfamily v3 | 2 | 4 | 0.0271642 | `rows[2]` |
| pruning v4 | 2 | 4 | 0.0271642 | `rows[3]` |
| pruning-stopping v5 | 4 | 8 | 0.0543284 | `rows[4]` |
| multifamily v6 | 4 | 16 | 0.1086568 | `rows[5]` |
| multifamily v7 | 8 | 32 | 0.2173136 | `rows[6]` |
| multifamily v8 | 8 | 32 | 0.2173136 | `rows[7]` |
| bank16 v9 | 8 | 32 | 0.2173136 | `rows[8]` |
| joint-readout v1 | 2 | 2 | 0.0135821 | `rows[9]` |

Поля колонок: `nominal_cli_cycles`, `actual_wave_calls`, `conditional_row_wave_call_equivalents`; для каждой строки `report_path` и `sources` указывают исходные `budgets.train_waves`, `pilot.actual_waves`, `pilot.wave_stats`. Условная колонка — умножение вызовов на **единственный** T5 `factor.value=0.00679104994805849`, а не эмпирическая реконструкция старых волн или доказанная эквивалентность обучения (`assumption`, `conversion_status`).

Две norm-пары T5 изолируют один и тот же limiter, не два независимых ослабления. Shield отдельно уменьшает plastic-норму примерно в 15.54 раза, limiter — в 147.25 раза, совместно — в 2288.26 раза (`norm_correction.updates[0].{shield_only_attenuation_before_limiter,limiter_only_attenuation,combined_attenuation}`). Округление float32-норм сохранено отдельно (`requested_ratio_relative_error`, `shielded_ratio_relative_error`); для бюджетной арифметики используется traced step-factor.

`--train-waves` означает проходы по выбранным строкам, каждый принятый row даёт отдельный wave-call (`cli.meaning`, `cli.sources`). При условной цели 32 эквивалента **на строку** нужны 4713 CLI-проходов, или 18852 вызова для выбранных строк v9 (`conditional_target.per_row_interpretation`). Если 32 — цель **по сумме row-call эквивалентов**, равномерный расчёт даёт 1179 проходов и 4716 вызовов (`total_row_calls_interpretation`). Обе оценки выше текущего CLI-cap 8 (`cli.maximum_current_cycles`); запуск не рекомендован. Время/сходимость не экстраполированы, limiter не изменён, обучения нет (`time_extrapolation`, `training_performed`, `limiter_changed`). Тесты аудита: 5 passed — [журнал](file:///D:/Ziwago18-storage/Produce/v19_n2_tests_retry2_20261109.log#L2).

### Измеренный статус, 2026-11-07

- Механизмные проверки: 118 passed и 4 subtests passed после исправлений карантина, фактической телеметрии, data-only восстановления графа, обязательной полноты envelope и привязки допуска к неизменяемому снимку байтов. Проверены сохранение общих тензоров без связи с живым состоянием и отсутствие повторной записи больших весов; противоречивое утверждение не обучает матрицу. Контроли интервенций сна сравниваются с явными ограничениями модели; флаг успеха при проваленном измерении не допускает обучение или принятие.
- Реальная регрессия на неизменённых весах старого мастера: 1 failed, 2 passed, 1 xfailed. Все шесть разнородных вопросов получили одинаковый отказ; предел повторов ≤2 сохранён. Полезная языковая генерация пока не восстановлена.
- Первый полный эксперимент сохранения/загрузки: четыре контрольных ответа совпали, SHA-256 старого мастера не изменился, независимый допуск 0/2 и collapse=1.0. Ответы на искусственной модели получены ограниченным графовым рендерингом, а не доказанной свободной нейронной генерацией. Отчёт: `Produce/v19_integration_1794002877169059600.json`; поле `full_runtime_roundtrip_exact` в нём проверяет именно эти четыре строки, не весь runtime.
- Строгий полный эксперимент `Produce/v19_strict_snapshot_1794003562784814400.json`: состояние полного ядра совпало по хешу до/после оценки и сохранения/загрузки; граф и три контрольных ответа также совпали. Размер кандидата снизился с 8 625 058 781 до 4 473 049 139 байт. Хеш кандидата и сохранность мастера проверены повторно; допуск остаётся 0/2, принятый checkpoint не создан. Это проверка состояния на искусственной модели, не восстановление языковой полезности или сертификация естественного языка.
- Калибровка без независимых размеченных наблюдений недоступна. Модели аналогий регистрируются явно через `register_graph`; пустой реестр не заменяется заготовленными примерами. Изоляция остаточных банков не гарантирует отсутствие забывания в общей выходной матрице.

Статус — экспериментальная реализация v19, не завершённая языковая сертификация. Для следующих этапов нужны адаптеры документных QA, отдельные acceptance/calibration данные, источники Knowledge и сравнения при одинаковом бюджете; недостающие корпуса не заменяются экзаменационными уроками.

### Метрика коллапса и инфраструктура проверки, 2026-11-08

`collapse` в acceptance считался по всем парам ответов. Ассистент подставляет одну и ту же строку `I do not have source-supported knowledge for this question.` на любой не-grounded ответ, поэтому любой отказ на разные вопросы давал идентичные тексты, Jaccard ≥ 0,92 и `collapse = 1.0` независимо от модели. Closed-book допуск был непроходим по построению: корректное «не знаю» выглядело как вырожденный ответ.

Пересчёт на сохранённых отчётах, без переобучения: closed-book (`v19_closed_book_evaluation_1794072692778745100.json`) — 6 кейсов, 0 содержательных, collapse 1.0 → 0.0, `refusal_rate` 1.0; open-book (`v19_open_book_evaluation_1794072980205352700.json`) — 6 кейсов, 2 содержательных, collapse 0.4 → 0.0, `refusal_rate` 0.667. Заголовок «collapse» был полностью артефактом отказа: два содержательных ответа open-book между собой коллапсом не являлись.

Правка `neuroweave10/evaluation/gates.py`: `collapse` считается только по содержательным ответам, добавлены `refusal_rate` и `substantive_responses`, `available` требует хотя бы один содержательный ответ. Допуск не смягчён — `passed` по-прежнему требует, чтобы прошёл каждый кейс, поэтому прогон, где модель отказалась на всё, остаётся проваленным. Зафиксировано двумя тестами в `tests/test_v19_evaluation.py`.

Инфраструктура: `tools/cleanup_project.py` знал только линейку v16/v18 и удалял бы все семь `tests/test_v19_*.py`, а `Produce/v19_source_backup` (SHA-256-откат исходников, `migration_manifest.json`) считал мусором — точки входа переведены на v19, тесты v19 и откатный бэкап защищены; `tools/analyze_connectivity.py` исключает копию бэкапа из графа импортов (162 модуля, 90 достижимы, вместо ложных «213 из 302 несвязанных»); `tools/preflight.py` измеряет свободное место по тому каталога чекпоинтов и печатает реальный том D: с указанием junction-пути, а не C: с кодом проекта.

Семифайловый механизмный набор v19 после правок: **452 passed, 232 subtests passed за 106,61 с**. Реальная legacy-регрессия v16/v18 не менялась: 1 failed, 2 passed, 1 xfailed.

### Локальные данные Hugging Face, 2026-11-07

`hf` версии 2.4.0 установлен в `C:/Users/yarma/AppData/Roaming/Python/Python314/Scripts/hf.exe`; системный PATH не изменялся. Данные скачаны этим CLI без авторизации и без исполнения кода репозиториев. Закреплённые ревизии, лицензии и проверенные хеши сохранены в `data/hf/download_manifest.json`; исходные карточки лицензий сохранены рядом с данными.

- OASST2: 128 575 train и 6 599 validation сообщений, Apache-2.0. В скачанных Parquet синтетических сообщений 0. В `data/hf/oasst2/paired_v19` подготовлены 6 287 EN/RU train-пар и 331 validation-пара: проверенные участниками исходные root-вопросы и rank-0 ответы, без удалённых записей. Три повторяющихся validation-вопроса исключены. Tree-ID и нормализованные вопросы между парами splits не пересекаются; это не доказательство семантической независимости или истинности ответов.
- Dolly 15K: 15 011 инструкций, из них 4 467 с reference context, CC BY-SA 3.0. Для factual QA нужны адаптеры, сохраняющие контекст и происхождение; свободные ответы не считаются внешне проверенными фактами.
- SQuAD: 87 599 train и 10 570 validation примеров с исходными passages, CC BY-SA 4.0. Статьи и question-ID раздельны, но между splits есть восемь одинаковых нормализованных вопросов; их нужно исключить до независимого acceptance.
- WikiText-103 raw: 1 801 350 train, 3 760 validation и 4 358 test записей, CC BY-SA 3.0/GFDL. Исходные Parquet сохранены; `data/wikitext-103-raw/wiki.train.raw` содержит 539 931 870 байт. Validation и test находятся в отдельных raw-файлах.

Первоначальная загрузка заняла 411 540 957 байт; вместе с подготовленными текстами и парами — около 0,97 ГБ. Подготовка выполняется `tools/prepare_hf_v19_data.py` и отказывается перезаписывать готовые результаты.

При переносе `data` и `Produce` в `D:/Ziwago18-storage` каждый файл сверён по SHA-256 до удаления исходной копии. Оригинальные пути на C: сохранены NTFS junctions; исходники остаются в рабочем каталоге. Перенесено 15 393 250 753 байт, сразу после проверки свободно 24 971 481 088 байт на C: и 714 276 347 904 байт на D:. Master сохранил хеш `cb97c8691869af0655b144bf56e904bfb6cae78d8f8eb2213b282441d76592e0`. Журнал — `D:/Ziwago18-storage/migration_manifest.json`; утилита — `tools/move_storage_to_d.py`. Новые данные и кандидаты записываются на D:.

На этапе первоначальной подготовки механизмный набор давал 118 passed, 4 subtests passed, а синтаксическая эвристика пропускала 1 975 из 6 287 train-пар. Это исторический замер до исправления ложного отбрасывания русских односимвольных слов, не оценка правильности остальных данных. Последующие исправления GPT-2 сохраняют UTF-8 и декодируют всю последовательность байтов, а не отдельные фрагменты символов; требуется установленный пакет `regex`. Пустая строка больше не кодируется фиктивным токеном «the».

Декодер сохраняет положительные и отрицательные оценки всего изученного словаря. Teacher forcing использует ту же сумму дендритных, SWR, остаточных, спайковых и PC-оценок до антициклических ограничений. Обучение и синтез имеют общую подготовку коры; PC обучается на настоящем L5 текущего тика. Все допустимые пары обучают дендритную матрицу; превышение позиционного окна отклоняется, а не обрезается. Два соседних BPE-токена допустимы, третий подряд запрещён; внешний предел повторяемости ответов ≤2 не изменён. После регрессий source-span и выравнивания контрфактической ошибки обучения с фактическим распределением декодера проверено 152 теста и 24 подтеста.

`tools/prepare_v19_language_curriculum.py` подготовил исправленный пилот: 24 уникальных полных коротких документных ответа SQuAD из 12 train-статей и 6 закрытых вопросов из 6 validation-статей. Исключены все восемь совпадающих вопросов официальных splits; source-span, ревизия и хеши сохранены в `data/hf/squad/language_v19_v2/manifest.json`. Каждый train-span повторно проверен по исходному документу. Это разделение статей/вопросов, не доказательство независимости тем или отсутствия загрязнения прежнего мастера. Экзаменационный контекст не предоставляется модели и не становится её источником; отказ безопасен, но не считается успешным QA.

Первый пилот остановлен после двух полных эпох: ручная проверка выявила два обрывка, ошибочно выделенных на десятичной точке «8.0». Его исходные файлы/журнал сохранены, manifest помечен непригодным для дальнейшего обучения, accepted checkpoint не создавался. Исправленный сегментатор сохраняет десятичные числа и сокращения; отдельные тесты предотвращают повторение. Новый четырёхэпоховый пилот завершён из неизменённого legacy мастера, не из частично обученного состояния. Выполнены 96 pair-updates, 2 120 дендритных/спайковых token-updates и 1 832 PC targets; приняты все 3 144 просканированных BPE-токена. На четырёх train-диагностиках прямой синтез из весов при фиксированном query-векторе, без n-грамм/копирования текста, точно воспроизвёл 3/4 ответа против 0/4 до обучения. Это не проверка маршрутизации CA3 или всех веток ассистента и не независимое обобщение. Teacher-forced NLL после обучения — 0,0982 nat/token, token error rate — 1/126; до обучения NLL был unavailable из-за неизвестных токенов, поэтому его снижение не заявляется. Closed-book acceptance — 0/6, collapse=1, шесть одинаковых source-refusals. CLI вернул 1 из-за проваленного допуска, не ошибки обучения.

Отчёт — `Produce/v19_acceptance_1794007811470949300.json`. Кандидат — `Produce/checkpoints/ziwago_v19_candidate_1794007671713945800.pt`, 4 478 989 151 байт, SHA-256 `46e1f57c1c8681b866142715a889c6ed25c8255a7d26c200dc639b07c0420132`; он повторно сверён, не принят и не заменяет мастер. SHA-256 мастера снова проверен и неизменен. Запоминание train и независимый acceptance измеряются раздельно. WikiText, Dolly и SQuAD используют Wikipedia: перед смешанным обучением нужен cross-corpus аудит документов и тем. Калибровка и общая языковая сертификация отсутствуют.

Новый кандидат с контрфактическим credit assignment — `Produce/checkpoints/ziwago_v19_candidate_1794043022813953700.pt`, 4 536 823 711 байт, SHA-256 `dc5513986dc3db47da4aab70e8770d81ec008b7973dac16084490b1e27c5a1ca`. После одной эпохи средний teacher-forced train NLL снизился с 0,096698 до 0,073335 nat/token; три ответа остались точными, но Lourdes сохранил одну ошибку и прежний неграмматичный raw-ответ. Ранг цели в ошибочной позиции улучшился с 272 до 125, отрицательный margin — с 4,747 до 3,751; вклад дендритов в пользу цели +6,899 всё ещё перекрывается защищённым ядром −10,615. Полный ассистент дал 3/4 grounded train-ответов; диагностический imprint и evaluation полностью откачены, authority и checkpoint-файлы не изменились. Независимый acceptance остаётся 0/6 с шестью одинаковыми отказами и collapse=1, поэтому кандидат не принят. Отчёты — `Produce/v19_acceptance_1794043244107259600.json`, `Produce/v19_decoder_credit_after_1794043022813953700.json` и `Produce/v19_assistant_recall_probe_v6.json`.

### Дополнительные корпуса на D:, 2026-11-07

`tools/download_hf_v19_language.py` скачал через `hf` 22 закреплённых файла общим размером 4 373 145 169 байт. Все SHA-256 повторно вычислены; форматы полностью проверены потоковым чтением. Манифест — `data/hf/additional_download_manifest.json`; первоначальный манифест четырёх корпусов не изменён. Кэш новых загрузок также находится на D:.

- TinyStoriesV2-GPT4: 2 717 700 train и 27 631 validation framed-записей, CDLA-Sharing-1.0; синтетический материал, не проверенные факты. 206 исходных пустых записей явно учтены и сохранены, не скрыты фильтрацией.
- DailyDialog: 11 118 train, по 1 000 validation/test диалогов, CC-BY-NC-SA-4.0 — только research/noncommercial с атрибуцией и share-alike. У закреплённой Parquet-конверсии нет карточки; лицензия отдельно сверена по карточке издателя, её ревизия и blob ID записаны.
- UltraChat: 207 865 train_sft и 23 110 test_sft разговоров, MIT; синтетические ответы не считаются фактами. Test хранится вне train root в `data/hf/ultrachat200k_eval`.
- NPC Dialogue v2: 1 689 записей, Apache-2.0; опубликованные автором Gemini-синтетические игровые диалоги, не извлечённые оригинальные игровые сценарии.
- PersonaMem: 110 contexts и 2 727 вопросов, MIT; только benchmark, не материал обучения.
- OpenWebText: 400 692 документа в первых четырёх из 80 shards; это subset, не полный корпус. CC0 упаковки не отменяет права исходных сайтов; официальный независимый split отсутствует.

BookCorpus, Cornell и PersonaChat не загружены из-за неустановленных прав/исходного лицензированного релиза. Gutenberg не понадобился текущему пилоту. Старые пути и preflight не подменялись фиктивными эквивалентами. После загрузки свободно примерно 24,96 ГБ на C: и 709,90 ГБ на D:; эти значения относятся к проверке до записи нового кандидата.

---

# Историческое руководство v10.1 – v16.1 (неверифицированные заявления прежних версий)
## Научно-архитектурное руководство (`ReadmeUse.md`)

---

## Оглавление
1. [Суть проекта и научная парадигма](#1-суть-проекта-и-научная-парадигма)
2. [Полная структура и топология архитектуры (17 408 нейронов)](#2-полная-структура-и-топология-архитектуры-17-408-нейронов)
3. [Глубокое научное объяснение узлов: почему именно так](#3-глубокое-научное-объяснение-узлов-почему-именно-так)
   - [3.1. Композиционный морфо-фонетический VSA-BPE энкодер и OOV-устойчивость](#31-композиционный-морфо-фонетический-vsa-bpe-энкодер-и-oov-устойчивость)
   - [3.2. 16-колоночная спайковая кора Бродмана (MoE) и апикально-базальная интеграция](#32-16-колоночная-спайковая-кора-бродмана-moe-и-апикально-базальная-интеграция)
   - [3.3. Матрица 8 192 нелинейных дендритных NMDA-ветвей (0% N-грамм)](#33-матрица-8-192-нелинейных-дендритных-nmda-ветвей-0-n-грамм)
   - [3.4. Двунаправленный гиппокамп CA3, Контекстный Арбитраж и Латеральная Габитуация](#34-двунаправленный-гиппокамп-ca3-контекстный-арбитраж-и-латеральная-габитуация)
   - [3.5. Тета-фазовая прецессия О'Кифа–Лисмана, Синтаксический Мост BA44 и Дейктический Сдвиг TPJ](#35-тета-фазовая-прецессия-окифа-лисмана-синтаксический-мост-ba44-и-дейктический-сдвиг-tpj)
   - [3.6. 17-я рефлексивная колонка SON (VTE Inner Speech), Медленноволновой Сон SWS и Острые Волны SWR](#36-17-я-рефлексивная-колонка-son-vte-inner-speech-медленноволновой-сон-sws-и-острые-волны-swr)
4. [Полный аудит замечаний из `next14.txt` и как они устранены](#4-полный-аудит-замечаний-из-next14txt-и-как-они-устранены)
5. [Карта репозитория и назначение ключевых модулей](#5-карта-репозитория-и-назначение-ключевых-модулей)
6. [Условия запуска, окружение, тренировка и верификация](#6-условия-запуска-окружение-тренировка-и-верификация)
   - [6.1. Системные требования и окружение Python](#61-системные-требования-и-окружение-python)
   - [6.2. Архитектура данных и балансировщик `DeficitRoundRobinDomainBalancer`](#62-архитектура-данных-и-балансировщик-deficitroundrobindomainbalancer)
   - [6.3. Запуск обучения, CI-тестов и проверка ворот качества (`UnifiedCurriculumGates`)](#63-запуск-обучения-ci-тестов-и-проверка-ворот-качества-unifiedcurriculumgates)
   - [6.4. Программная загрузка чекпоинта и инференс (Python API)](#64-программная-загрузка-чекпоинта-и-инференс-python-api)
7. [Эмпирические показатели актуального мастер-прогона v16.1](#7-эмпирические-показатели-актуального-мастер-прогона-v161)
8. [P.S. От себя: Взгляд за горизонт v16 — от Аттракторной Композиции к Переменным Слотам Мышления](#8-ps-от-себя-взгляд-за-горизонт-v16--от-аттракторной-композиции-к-переменным-слотам-мышления)

---

## 1. Суть проекта и научная парадигма

**Neuroweave Ziwago-Prime** — это биомиметический нейро-символический и спайково-дендритный когнитивный организм, созданный для исследования и реализации **обучения и речевого мышления без глобального обратного распространения ошибки (Zero Global Backpropagation)** на чистых локальных синаптических правилах.

В отличие от классических трансформеров, которые оптимизируют миллиарды параметров глобальным градиентным спуском за тысячи GPU-часов и страдают от катастрофического забывания при онлайн-дообучении, **Ziwago-Prime** опирается на 5 фундаментальных принципов теоретической нейробиологии:

1. **Принцип Свободной Энергии и Предиктивного Кодирования (Karl Friston, Rao & Ballard):**
   Каждый кортикальный ансамбль непрерывно минимизирует вариационную свободную энергию (ошибку предсказания между восходящим сенсорным входом на базальных дендритах и нисходящим контекстным прогнозом на апикальном пучке):
   $$\mathcal{F}(t) = \underbrace{D_{\mathrm{KL}}\!\Big[q(\mathbf{z}_t) \,\|\, p(\mathbf{z}_t \mid \mathbf{z}_{t-1})\Big]}_{\mathrm{Complexity\; /\; Нагрузка\; памяти}} - \underbrace{\mathbb{E}_{q}\!\Big[\ln p(\mathbf{s}_t \mid \mathbf{z}_t)\Big]}_{\mathrm{Accuracy\; /\; Точность\; предсказания}}$$
2. **Трёхфакторная локальная синаптическая пластичность (Gerstner, Izhikevich, Frémaux):**
   Синаптические веса модифицируются строго локально по произведению трёх биологических сигналов: пресинаптического следа $\mathrm{Pre}(t)$, постсинаптического дендритного плато/ошибки $\mathrm{Post}(t)$ и глобального третьего фактора нейромодуляции $F_{\mathrm{mod}}(t) \in \{\mathrm{DA}, \mathrm{ACh}, \mathrm{NE}, \mathrm{5\text{-}HT}\}$:
   $$\Delta W_{ij} = \eta \cdot F_{\mathrm{mod}}(t) \cdot \phi_{\mathrm{NMDA}}\!\Big(\mathrm{Pre}_i(t)\Big) \cdot \Big(y_j^{\mathrm{target}}(t) - \hat{y}_j(t)\Big)$$
3. **Теория Комплементарных Систем Обучения (McClelland, McNaughton & O'Reilly — Complementary Learning Systems, CLS):**
   Быстрое однократное запоминание эпизодов и фактов собеседника (например, имени *Alex* и его проекта) происходит в автоассоциативной сети **гиппокампа `CA3`** под всплеском дофамина (`DA`) и ацетилхолина (`ACh`), а медленная структурная консолидация и очистка шума выполняются во время фаз **медленноволнового сна (`SWS`)** и **острых волн гиппокампа (`Sharp-Wave Ripples / SWR`)**.
4. **Нелинейные дендритные NMDA-плато (Matthew Larkum, Bartlett Mel, Jeff Hawkins):**
   Один пирамидный нейрон — это не точечный сумматор, а многоветвевое нелинейное дерево. Ветви апикальных и базальных дендритов содержат NMDA-рецепторы, которые зажигают **дендритный плато-потенциал** только при совпадении локального контекста предыстории и глобального семантического полюса.
5. **Анатомическая модульность коры по зонам Бродмана + метакогнитивный самоконтроль (`SON`):**
   Синтаксис и межклаузальные союзы обрабатываются зоной Брока (`BA44`), дейктический сдвиг точки зрения (`1-е / 2-е / 3-е лицо`) — височно-теменным узлом (`TPJ`), временная очередь замысла — дорсолатеральной префронтальной корой (`BA46`), а внутренняя речь (`Vicarious Trial and Error / VTE`) и моторный стоп-кран (`SON_MOTOR_STOP`) — 17-й рефлексивной колонкой (`SON`).

---

## 2. Полная структура и топология архитектуры (17 408 нейронов)

Ниже представлена полная структурная схема информационного потока в **Ziwago-Prime v16.1**:

```text
====================================================================================================
                     ВХОДНОЙ ПОТОК РЕЧИ / КОРПУСОВ (11 локальных датасетов, ~36 ГБ)
====================================================================================================
                                              │
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ УРОВЕНЬ 1: ТАЛАМИЧЕСКИЙ ФИЛЬТР И БАЛАНСИРОВЩИК ДОМЕНОВ                                           │
│  • 32-Core Thalamic Fast-Scan Worker (8–11 млн сырых токенов/сек)                                │
│  • Холинергический гейт новизны и каузальной плотности (ACh Surprise Score)                      │
│  • Балансировщик доменов DeficitRoundRobinDomainBalancer (защита от пустых шардов и перекосов)   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ УРОВЕНЬ 2: МОРФО-ФОНЕТИЧЕСКИЙ VSA-BPE ЭНКОДЕР (CompositionalVSABPEEncoder)                       │
│  • Словарь GPT-2 (50 257 токенов) + SHA-256 контроль целостности токенизатора                    │
│  • Детерминированная голографическая связка субслов: BPE -> 2048-бит SDR -> вектор R^64          │
│  • 100% OOV-устойчивость: незнакомые слова раскладываются на морфемы без <unk> и коллапса        │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
┌─────────────────────────────────────────────┐   ┌────────────────────────────────────────────────┐
│ УРОВЕНЬ 3А: 16 КОЛОНОК КОРЫ БРОДМАНА (MoE)  │   │ УРОВЕНЬ 3Б: ГИППОКАМП CA3 И РАБОЧАЯ ПАМЯТЬ BA46│
│  • 16 x 1 024 = 16 384 спайковых нейронов   │   │  • ConversationalDialogueState (память диалога,│
│  • Специализация: BA44 (Syntax), BA22       │   │    имя собеседника, история ходов, домен)      │
│    (Wernicke), BA46 (DLPFC), BA7/39 (Logic),│   │  • 5-Bank CA3 Attractor Atlas (Stories, Books, │
│    BA38 (Narrative), BA10 (Metacognition)   │   │    Science, Social/Movies, Wisdom)             │
│  • Двухкомпартментные пирамидные клетки:    │   │  • Двунаправленный индекс Stimulus <-> Response│
│    Basal (Somatic Error) + Apical (Context) │   │  • Мультипликативный Контекстный Арбитраж +    │
│  • Sparse Top-K маршрутизация с гомеостазом │   │    Латеральная Габитуация недавних полюсов     │
└─────────────────────────────────────────────┘   └────────────────────────────────────────────────┘
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ УРОВЕНЬ 4: ТЕТА-ФАЗОВАЯ ОЧЕРЕДЬ (BA46) + МАТРИЦА 8 192 ДЕНДРИТНЫХ NMDA-ВЕТВЕЙ (0% N-ГРАММ!)      │
│  • Разбиение сложного вопроса на клаузы в хронологическом порядке тета-фаз (tau = 0, 1, ...)     │
│  • DendriticPlateauMatrixReadout: 8 192 нелинейных ветвей NMDA, Top-32 Sparse WTA                │
│  • Векторизованное считывание: logits = d_NMDA(v_{t-3}, v_{t-2}, v_{t-1}, pos, Q_CA3) @ W_dend   │
│  • Поддержка эпизодической консолидации острых волн гиппокампа (W_swr_dend_out)                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ УРОВЕНЬ 5: СИНТАКСИЧЕСКИЙ МОСТ ЗОНЫ БРОКА BA44 + ДЕЙКТИЧЕСКИЙ УЗЕЛ TPJ + РЕФЛЕКСИЯ SON           │
│  • BA44AttractorSyntaxBridge: выбор союза (", while", ", before", ", and", ", because"),         │
│    торможение ранней точки (Period Inhibition Gate), подавление междометий во 2-й клаузе,        │
│    устранение инфинитивного шва ("before To fix" -> "before you fix")                            │
│  • TPJ (Temporoparietal Junction): автоматический сдвиг лица ("Your/you" -> "His/he" в 3-м лице) │
│  • 17-я колонка SON (1 024 нейрона): оценка внутренней речи (VTE) и стоп-кран SON_MOTOR_STOP     │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Сводная таблица параметров популяции нейронов (`17 408` клеток)

| Подсистема | Биологический прототип | Число нейронов / ветвей | Роль в вычислениях |
| :--- | :--- | :--- | :--- |
| `SparselyGatedCorticalMoE` | 16 кортикальных макроколонок Бродмана (`BA44`, `BA45`, `BA22`, `BA38`, `BA7`, `BA39`, `BA46`, `BA10` и др.) | `16 × 1 024 = 16 384` нейрона | Апикально-базальное предиктивное кодирование, доменная специализация |
| `SONReflectiveColumn` | 17-я супраоптическая / метакогнитивная колонка (`BA10 / ACC / SON`) | `1 024` нейрона (`768 E + 256 I`) | Мониторинг свободной энергии $\mathcal{F}(t)$, внутренняя речь VTE, сигнал `SON_MOTOR_STOP` |
| `DendriticPlateauMatrixReadout` | Поле `CA1` / пирамидный слой `L5` с ветвями NMDA | `8 192` дендритных NMDA-компартмента $\times V_{\mathrm{act}}$ | Чисто матричный пословный синтез ответа без таблиц N-грамм |
| `FiveBankCA3AttractorAtlas` | Поле `CA3` гиппокампа с 5 кортикальными банками | До `4 096` глобальных / `5 × 2 048` доменных полюсов | Автоассоциативная очистка стимула, защита доменов от взаимного вытеснения |
| `BA44AttractorSyntaxBridge` | Парс оперкулярис зоны Брока (`BA44`) + `TPJ` | Матрица $W_{\mathrm{BA44}} \in \mathbb{R}^{192 \times 5}$ + проекции `TPJ` | Грамматическое сшивание двух полюсов в сложноподчинённое предложение и сдвиг лица |

---

## 3. Глубокое научное объяснение узлов: почему именно так

### 3.1. Композиционный морфо-фонетический VSA-BPE энкодер и OOV-устойчивость
**Проблема классических моделей:** Если редкий или новый составной термин (например, `"neuro-astrocytic synapto-homeostasis"`) не встречался в обучающей выборке, его эмбеддинг либо деградирует в `<unk>`, либо получает случайный необученный вектор.

**Научное решение в Ziwago (`CompositionalVSABPEEncoder`):**
* Используется полный словарь GPT-2 (`50 257` токенов), но векторы токенов в $\mathbb{R}^{64}$ и разреженные кортикальные коды `SDR` в $\{0, 1\}^{2048}$ строятся через **алгебру векторно-символических архитектур (Vector Symbolic Architectures / Holographic Reduced Representations)**:
  $$\vec{v}(\mathrm{token}) = \mathrm{Normalize}\!\Bigg(\sum_{m=1}^{M} \mathrm{Roll}\!\Big(\vec{b}_{\mathrm{subword}_m},\, m\Big)\Bigg)$$
* Благодаря этому любое незнакомое (OOV) слово автоматически раскладывается BPE-деревом на известные корни, приставки и суффиксы, формируя осмысленный ненулевой вектор с сохранением морфологического сходства и ортогональности к другим понятиям (проверяется воротами `evaluate_oov_compositional_resilience`).

### 3.2. 16-колоночная спайковая кора Бродмана (MoE) и апикально-базальная интеграция
**Почему кора разделена на 16 колонок с разреженной активацией (`k_active = 2..4`):**
* В неокортексе млекопитающих одновременно активно не более $2\text{–}5\%$ нейронов (метаболическое ограничение и защита от катастрофической интерференции).
* Каждая из 16 колонок имеет двухкомпартментный дендрит (`Basal` для текущего входа $\vec{x}_{\mathrm{in}}$ и `Apical` для контекстного предсказания). Маршрутизатор выбирает топ-$K$ колонок с учётом **гомеостатического порога усталости** $\Theta_{\mathrm{homeo}}$, чтобы ни одна колонка не монополизировала представление.

### 3.3. Матрица 8 192 нелинейных дендритных NMDA-ветвей (0% N-грамм)
**Почему мы полностью отказались от генерации по N-граммам (`use_ngrams = False`):**
* Как убедительно показал анализ в `next14.txt`, локальные таблицы 3/4-грамм не удерживают дальний семантический контекст вопроса и при смешении доменов порождают несвязную «словесную окрошку» (*word salad*).
* Вместо N-грамм генерация от первого до последнего токена выполняется **100% через матричное умножение в `DendriticPlateauMatrixReadout`**:
  1. На каждом шаге позиции $t$ формируется фазово-связанный контекстный вектор предыстории и целевого полюса $\vec{Q}_{\mathrm{CA3}} \in \mathbb{R}^{64}$:
     $$\vec{z}_t = \mathrm{Normalize}\!\Big(1.4\,\vec{Q}_{\mathrm{CA3}} + 1.1\,\vec{p}_t + 0.9\,\mathrm{Roll}(\vec{v}_{t-1}, 1) + 0.7\,\mathrm{Roll}(\vec{v}_{t-2}, 2) + 0.5\,\mathrm{Roll}(\vec{v}_{t-3}, 3) + 0.8\,(\vec{Q}_{\mathrm{CA3}} \odot \vec{p}_t) + 0.7\,(\vec{v}_{t-1} \odot \vec{Q}_{\mathrm{CA3}})\Big)$$
  2. Вектор $\vec{z}_t$ проецируется на `8 192` дендритных ветви $u = W_{\mathrm{dend\_in}} \vec{z}_t + \vec{b}_{\mathrm{dend}}$.
  3. **Латеральное торможение интернейронов (Top-32 Sparse Winner-Take-All):** из `8 192` ветвей только `32` ветви с максимальной деполяризацией преодолевают порог $\mathrm{Mg}^{2+}$-блокады и зажигают нелинейный кубический NMDA-плато-потенциал:
     $$d_k = (u_k - \Theta_{\mathrm{top32}})_+^3$$
  4. Поскольку $99.6\%$ ветвей на каждом шаге строго равны нулю, разные фразы и позиции активируют практически непересекающиеся подмножества из 32 ветвей. Это позволяет матрице $W_{\mathrm{dend\_out}} \in \mathbb{R}^{8192 \times V_{\mathrm{act}}}$ хранить тысячи точных синтаксических траекторий без взаимного наложения!

### 3.4. Двунаправленный гиппокамп CA3, Контекстный Арбитраж и Латеральная Габитуация
**Какая фундаментальная ошибка была в ранних версиях (`v14–v16`) и как она решена:**
* В ранней версии выбор полюса в `clean_up_stimulus_in_ca3_attractor` умножал косинусную близость на глобальную важность `w_ca3_salience` без проверки лексической релевантности текущему ходу. В результате глобально «громкий» полюс (например, NPC-реплика про `"Echoing Glade"`) захватывал короткие или вводные клаузы вроде `"Remind me,"` в вопросе `"Remind me, what is my name?"`.
* **Наше научное решение (три уровня защиты):**
  1. **Двунаправленная индексация `Stimulus <-> Response` (`bi_sdr`):** При консолидации пары $(A \longrightarrow B)$ в `w_ca3_sdr` записываются разреженные SDR-признаки как слов вопроса $A$, так и содержательных слов ответа $B$ ($\mathrm{SDR}_{AB} = \mathrm{clamp}(\mathrm{SDR}_A + 0.95\,\mathrm{SDR}_B,\, 1.2)$). Это позволяет гиппокампу находить нужный концепт даже тогда, когда вопрос сформулирован через понятия из следствия/ответа (обратный каузальный поиск).
  2. **Мультипликативный Контекстный Арбитраж (`Context Arbitrage`):**
     $$\mathrm{Score}(m) = \underbrace{\big(0.72\,\mathrm{SDR\_Drive}(q, m) + 0.28\,[\cos(\vec{q}, \vec{P}_m)]_+\big)}_{\mathrm{Релевантность\; текущему\; ходу}} \cdot \underbrace{\sigma\!\big(14\cdot(\mathrm{SDR\_Drive}(q, m) - 0.18)\big)}_{\mathrm{Холинергический\; лексический\; гейт}} \cdot \underbrace{\big(0.85 + 0.15\,\mathrm{tanh}(\mathrm{Salience}_m - 1.0)\big)}_{\mathrm{Сублинейный\; гейт\; важности}}$$
     Если у полюса $m$ нет реального лексического/SDR-пересечения с текущим ходом собеседника ($\mathrm{SDR\_Drive} < 0.18$), сигмоидальный гейт подавляет его вклад, какова бы ни была его глобальная `Salience`!
  3. **Латеральная габитуация (`Lateral Habituation`):** Полюса, уже активировавшиеся в последних 3 ходах диалога (`recent_poles`), получают депрессию проводимости ($\cdot 0.45$) при слабом повторном стимуле, предотвращая залипание в одном аттракторе.

### 3.5. Тета-фазовая прецессия О'Кифа–Лисмана, Синтаксический Мост BA44 и Дейктический Сдвиг TPJ
Когда пользователь задаёт сложный многоклаузальный вопрос, объединяющий несколько доменов (например: *«Can running outdoors clear the mind before writing an essay without grammar errors?»* или *«Alex is testing his model; what is his name, and how can he fix grammar errors when writing an essay?»*):
1. **Фильтрация дискурсивных префиксов и Тета-фазовая очередь (`BA46`):**
   Вводные мета-обороты (`"Remind me,"`, `"Tell me,"`, `"By the way,"`) отсекаются до клаузального деления, чтобы не создавать ложную клаузу. Смысловые клаузы проецируются на полюса `CA3`, отбираются до 2 уникальных полюсов ($\cos(\vec{P}_1, \vec{P}_2) \le 0.88$) и выстраиваются строго в **хронологическом порядке фаз тета-ритма** ($\phi_1 < \phi_2$).
2. **Синтаксический Мост Зоны Брока (`BA44AttractorSyntaxBridge`):**
   - Матрица $W_{\mathrm{BA44\_bridge}} \in \mathbb{R}^{192 \times 5}$ выбирает оптимальный союз (`", while"`, `", before"`, `", after"`, `", because"`, `", and"`), распознавая триггеры как в строчном (`" before"`), так и в заглавном регистре в начале фразы (`"Before"`).
   - **Торможение ранней точки (`Period Inhibition Gate`):** В конце первой клаузы терминальная точка `.` блокируется.
   - **Подавление междометий и устранение инфинитивного шва (`Infinitive Seam Suppression`):** Во второй клаузе подавляются стартовые междометия (`"Yes, "`, `"Ah, "`, `"That sounds great! "`). Если мост является подчинительным союзом (`", before"`, `", after"`, `", while"`, `", because"`), а вторая клауза начинается с инфинитива цели `"To <verb>"` (`"To fix grammar errors..."`), зона Брока автоматически заменяет начальное `"To"` на личное местоимение-подлежащее `"you"` (`", before you fix grammar errors in your essay..."`), полностью устраняя грамматический шов `"before to fix"`!
3. **Височно-теменной узел дейктического сдвига лица (`TPJ Deictic Perspective Shift`):**
   - Если во входном вопросе присутствуют маркеры 3-го лица (`he`, `his`, `him`) и отсутствуют маркеры 1-го лица (`I`, `my`, `me`), проекция `TPJ` автоматически переводит местоимения 2-го лица во 3-е (`"Your" -> "His"`, `"your" -> "his"`, `"You" -> "He"`, `"you" -> "he"`, `"are" -> "is"`) по всей сгенерированной траектории:
     *«His name is Alex, and he is building a biological neural network with dendritic predictive coding, and to fix grammar errors in his essay, read each sentence aloud...»*

### 3.6. 17-я рефлексивная колонка SON (VTE Inner Speech), Медленноволновой Сон SWS и Острые Волны SWR
* **17-я колонка `SON` (`SONReflectiveColumn`):** Наблюдает за частотами спайков и соматическими/апикальными ошибками всех 16 колонок коры, вычисляет свободную энергию $\mathcal{F}(t)$ и останавливает генерацию (`SON_MOTOR_STOP`), как только эпистемический драйв исчерпан на границе предложения.
* **Медленноволновой сон (`SWS`, теория синаптического гомеостаза Тонони–Чирелли):** В конце каждой эпохи выполняет глобальное пропорциональное шкалирование весов ($\cdot 0.988$), прунинг слабых шумовых связей и сортировку полюсов `CA3` по накопленной нейромодуляторной значимости.
* **Консолидация острых волн гиппокампа (`Sharp-Wave Ripple / SWR Replay`):** Когда сеть впервые синтезирует сложный ответ из двух полюсов $[\vec{P}_1, \vec{P}_2]$ через мост `BA44`, она формирует ортогональный композиционный полюс $\vec{Q}_{\mathrm{SWR}} = \mathrm{Normalize}(\mathrm{Roll}(\vec{P}_1, 1) \odot \vec{P}_2 + 0.35\,\vec{q}_{\mathrm{raw}})$ и записывает сшитую траекторию в эпизодическую матрицу `w_swr_dend_out`, которая опрашивается при последующих обращениях.

---

## 4. Полный аудит замечаний из `next14.txt` и как они устранены

Все 100% замечаний, диагностированных в `Produce/next14.txt`, полностью закрыты в коде и подтверждены автотестами:

| Категория в `next14.txt` | Описание замечания из `next14.txt` | Где и как исправлено в коде | Статус |
| :--- | :--- | :--- | :--- |
| **(А) Формат чекпоинта** | Потеря словарей `col_to_tok` и `tok_to_col` при сохранении `ziwago_v16_0_10B_omniscient_master.pt`, отсутствие хеша токенизатора и версии схемы. | В `neuroweave10/evaluation/gates.py` и `train_ziwago_v14_pure_synaptic_dialogue.py` внедрены `CHECKPOINT_SCHEMA_VERSION = "ziwago.checkpoint.v16.1"`, `compute_tokenizer_sha256`, `validate_self_contained_checkpoint`, `save_self_contained_checkpoint` и `load_self_contained_checkpoint`. Синхронизировано в `v14`, `v15` и `v16`. | **Устранено (Round-Trip = 100%)** |
| **(Б) Пайплайн данных и баланс доменов** | Падение `ProcessPoolExecutor(max_workers=0)` при пустом `BookCorpus`; отсутствие логирования пропущенных источников и балансировки доменов. | В `train_ziwago_v16_10B_omniscient_engine.py` внедрён `DeficitRoundRobinDomainBalancer` (явный лог `missing_sources_logged` + восполнение дефицита из соседнего корпуса того же регистра) и защита всех пулов `if not tasks:`. | **Устранено** |
| **(В) Тавтология экзамена** | `FIVE_WORLD_ANCHOR_PAIRS` подмешивались в `tok_pairs` каждую эпоху и проверялись тем же текстом на экзамене. | Обучающие уроки разнесены по эпохам (`FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH`), а экзамен вынесен в независимый `HOLD_OUT_EXAM_SUITE` (8 перефразированных и многоклаузальных запросов) с жёстким `assert len(holdout_leakage) == 0`. | **Устранено (0% утечки)** |
| **Пункт 3: Единый модуль Ворот** | Ворота качества из `EVOLUTION_V10_1_CURRICULUM_README.md` (`Surprise < 0.25`, `100% grammar`, `0 superacid`, `Retention > 1500`, `OOV`) не вызывались в `v16`. | Создан и подключён единый модуль `neuroweave10/evaluation/gates.py` (`UnifiedCurriculumGates`), проверяющий все критерии после прогона. | **Устранено (`Passed = True`)** |
| **Пункт 4: Отказ от N-грамм** | Ветка `--ngrams on` генерировала несвязный текст (`"And the I have been waiting..."`). | Режим `use_ngrams = False` (100% матрица `8 192` NMDA-ветвей) установлен по умолчанию; отключён стохастический джиттер на детерминированных лучах. | **Устранено** |
| **Пункт 5: Арбитраж контекста в CA3** | На вопрос `"Remind me, what is my name?"` первым полюсом выбирался `"Echoing Glade"` из-за глобального `w_ca3_salience` и разбиения `"Remind me,"` в отдельную клаузу. | Внедрены мультипликативный арбитраж `turn_relevance * lexical_gate * sal_gain`, габитуация `recent_poles` и фильтрация дискурсивных префиксов (`"Remind me,"`). | **Устранено (`EchoContam = 0.0%`)** |
| **Пункт 6: Метрика маскировки** | Отсутствовало измерение доли ответов, схлопывающихся в ближайший выученный полюс при изменённом входе (`Attractor Collapse Rate`). | В `neuroweave10/evaluation/gates.py` реализована функция `evaluate_attractor_collapse_rate` по 3 кластерам возмущений (`ATTRACTOR_COLLAPSE_PROBE_CLUSTERS`). | **Устранено (`CollapseRate = 0.0%`)** |
| **Пункт 8: Состояние диалога и шов BA44** | Не использовалось явное состояние разговора (`what is my name`), а при мосте `", before"` возникал грамматический шов `"before to fix"`. | Интегрирован `ConversationalDialogueState` (рабочая память имени и проекта собеседника + доменная маршрутизация), а в `BA44AttractorSyntaxBridge` добавлена замена `"before To fix"` $\longrightarrow$ `"before you fix"`. | **Устранено** |

---

## 5. Карта репозитория и назначение ключевых модулей

```text
/home/jupyter/Neuroweave Ziwago/
├── ReadmeUse.md                                 # Настоящее полное научно-архитектурное руководство и инструкция
├── ZIWAGO_V16_IMPROVEMENT_PLAN.md               # План и таблица верификации исправлений по next14.txt
├── ARCHITECTURE_v10.md                          # Спецификация базового спайкового ядра v10
├── EVOLUTION_V10_1_CURRICULUM_README.md         # Спецификация трёхэтапного онтогенеза и ворот v10.1
├── train_ziwago_v16_10B_omniscient_engine.py    # ГЛАВНЫЙ ТРЕНЕР v16.1: 5 эпох, 11 датасетов, 5-Bank CA3, Unified Gates
├── train_ziwago_v15_omniscient_marathon.py      # 5-мировой марафон со сном SWS (Тонони–Чирелли) и SWR
├── train_ziwago_v14_pure_synaptic_dialogue.py   # ЯДРО ОРГАНИЗМА: DendriticPlateauMatrixReadout (8192 NMDA), BA44, TPJ, ConversationalDialogueState
├── train_ziwago_v13_son_reflective_organism.py  # 17-я колонка SON, VTE Inner Speech и моторный стоп-кран
├── ziwago_v12_conversational_mind.py            # Прототип рабочей памяти диалога и доменной маршрутизации
├── train_ziwago_v10_curriculum.py               # Трёхэтапный онтогенетический куррикулум v10.1 (StreamingDataMixer)
├── neuroweave10/
│   ├── config.py                                # Конфигурации нейрохимии (DA, NE, ACh, 5-HT) и спайковых слоёв
│   ├── core/sdr.py                              # CompositionalVSABPEEncoder (GPT-2 50,257 -> 2048 SDR -> R^64)
│   ├── cortex/
│   │   ├── moe_cortex.py                        # 16-колоночная спайковая MoE-кора Бродмана (16,384 нейрона)
│   │   └── son_reflection.py                    # 17-я рефлексивная колонка SON (1,024 нейрона, свободная энергия F)
│   └── evaluation/
│       ├── __init__.py
│       └── gates.py                             # Единый модуль ворот (UnifiedCurriculumGates), контракт чекпоинтов и HOLD_OUT_EXAM_SUITE
├── tests/
│   ├── test_v16_roundtrip.py                    # CI-тесты Round-Trip чекпоинта, контекстного арбитража, шва BA44 и ворот
│   └── test_v10_genesis.py                      # Базовые тесты спайкового ядра v10
├── data/                                        # Локальное хранилище 11 корпусов (~36 ГБ) и словаря GPT-2
└── Produce/
    ├── checkpoints/
    │   └── ziwago_v16_0_10B_omniscient_master.pt # Валидированный самодостаточный мастер-чекпоинт v16.1 (685.86 МБ, 9 509 колонок)
    ├── v16_0_10B_omniscient_report.json         # Полный JSON-отчёт 5-эпохового марафона, экзамена и ворот качества
    └── v16_0_shard_offset_ledger.json           # Журнал просканированных шардов и байтовых зон по 5 эпохам
```

---

## 6. Условия запуска, окружение, тренировка и верификация

### 6.1. Системные требования и окружение Python
* **ОС:** Linux (x86_64), оптимизировано под многоядерные процессоры (до 24–32 потоков CPU `ProcessPoolExecutor` + PyTorch CPU threads; GPU не требуется благодаря разреженности Top-32 WTA из 8 192 ветвей!).
* **Интерпретатор Python на сервере:** `/opt/micromamba/bin/python3.12` (содержит `torch`, `pyarrow`, `numpy`).
* **Обязательные файлы токенизатора:**
  * `data/gpt2/vocab.json`
  * `data/gpt2/merges.txt`
  *(Их целостность автоматически сверяется по SHA-256 при каждом сохранении и загрузке чекпоинта).*

### 6.2. Архитектура данных и балансировщик `DeficitRoundRobinDomainBalancer`
При полном серверном запуске тренер сканирует 5 онтогенетических эпох в `data/`:
1. **Эпоха I (Детство — Сюжетная причинность):** `data/tinystories/data/train-*.parquet` (4 шарда) + `data/tinystories/TinyStoriesV2-GPT4-train.txt` (16 зон).
2. **Эпоха II (Юность — Литература и длинный нарратив):** `data/BookCorpus/books_large_p1.txt` и `books_large_p2.txt` (32 зоны, 4.6 ГБ).
3. **Эпоха III (Университет — Энциклопедия и наука):** `data/wikitext-103-raw/wiki.train.raw` (16 зон) + `data/openwebtext_data/plain_text/train-*.parquet` (80 шардов, 24 ГБ).
4. **Эпоха IV (Социализация — Кино, диалоги, личность, роли):** `data/cornell/movie_lines.txt`, `data/dialog/daily_dialog/`, `data/dialog/personachat/data/`, `data/dialog/personamem/`, `data/dialog/npc_dialogue/`.
5. **Эпоха V (Мудрость — Инструкции и сложные рассуждения):** `data/dialog/oasst2/2023-11-05_oasst2_ready.trees.jsonl.gz` + `data/dialog/ultrachat/data/*.parquet` (8 шардов).

> **Работа на локальной машине с неполным набором данных:** Если часть тяжёлых корпусов отсутствует локально, `DeficitRoundRobinDomainBalancer` **не падает** с ошибкой `max_workers=0` и **не обнуляет эпоху**, а записывает отсутствующий путь в `report["domain_balancer"]["missing_sources_logged"]` и восполняет квоту эпохи из ближайшего доступного корпуса того же регистра.

---

### 6.3. Запуск обучения, CI-тестов и проверка ворот качества (`UnifiedCurriculumGates`)

#### А. Запуск CI-тестов Round-Trip, контракта чекпоинтов и ворот качества (~37 сек):
```bash
cd "/home/jupyter/Neuroweave Ziwago"
/opt/micromamba/bin/python3.12 tests/test_v16_roundtrip.py
```
**Что проверяет этот тест:**
1. `test_v16_save_load_byte_for_byte_roundtrip` — сохраняет обученный организм во временный чекпоинт, загружает в новый чистый инстанс `ZiwagoPureSynapticDialogueOrganismV14` и проверяет **100% побайтовое совпадение** ответов на 5 сложных промптах.
2. `test_v16_checkpoint_rejects_broken_maps_and_mismatch` — проверяет, что при удалении `col_to_tok`, нарушении биекции `col_to_tok <-> tok_to_col`, неверном SHA-256 токенизатора или чужой `schema_version` немедленно выбрасывается `CheckpointCompatibilityError`.
3. `test_v16_context_arbitrage_and_ba44_grammar_seam` — проверяет, что на вопрос `"Remind me, what is my name and what project am I building?"` возвращается `"Alex"` и `"biological neural network"` без `"Echoing Glade"`, а при мосте `", before"` генерируется `"before you fix grammar errors"` (без шва `"before to fix"`).
4. `test_v16_unified_curriculum_gates_and_zero_leakage` — проверяет `0%` пересечения стимулов с `HOLD_OUT_EXAM_SUITE` и прохождение всех ворот `UnifiedCurriculumGates`.

#### Б. Запуск полного 5-эпохового марафона Ziwago v16.1 по всем 162 шардам/зонам (~90 сек на 24–32 ядрах):
```bash
cd "/home/jupyter/Neuroweave Ziwago"
/opt/micromamba/bin/python3.12 train_ziwago_v16_10B_omniscient_engine.py 
    --workers 24 
    --cpu-threads 24 
    --owt-shards 80 
    --rows-per-owt-shard 4000 
    --elite-per-shard 60
```
По завершении скрипт:
* сохраняет самодостаточный мастер-чекпоинт `Produce/checkpoints/ziwago_v16_0_10B_omniscient_master.pt`;
* автоматически перезагружает его в новый экземпляр организма и проверяет `Round-Trip Exact Match = 3/3 (100%)`;
* сохраняет подробный отчёт в `Produce/v16_0_10B_omniscient_report.json` и журнал шардов в `Produce/v16_0_shard_offset_ledger.json`.

---

### 6.4. Программная загрузка чекпоинта и инференс (Python API)

Теперь любой сохранённый мастер-чекпоинт загружается в 3 строки кода и даёт **100% идентичный результат** без повторного обучения:

```python
from pathlib import Path
from train_ziwago_v14_pure_synaptic_dialogue import ZiwagoPureSynapticDialogueOrganismV14

root = Path("/home/jupyter/Neuroweave Ziwago")
organism = ZiwagoPureSynapticDialogueOrganismV14(root=root, workers=4, cpu_threads=8, seed=42)

# Загрузка с автоматической валидацией схемы, SHA-256 токенизатора и биекции col_to_tok <-> tok_to_col
val_info = organism.load_self_contained_checkpoint(
    root / "Produce" / "checkpoints" / "ziwago_v16_0_10B_omniscient_master.pt"
)
print("Checkpoint validated:", val_info)

# Генерация ответа в чистом режиме 8,192 NMDA-ветвей (0% N-грамм)
res = organism.pure_neural_generate_response(
    "Can running outdoors clear the mind before writing an essay without grammar errors?",
    max_tokens=44,
    use_ngrams=False,
)
print("Response:", res["pure_weight_generated_response"])
print("BA44 Bridge:", res["ba44_syntax_bridge_connector"])
```

---

## 7. Эмпирические показатели актуального мастер-прогона v16.2

Результаты зафиксированы в `Produce/v16_0_10B_omniscient_report.json` и эталонном файле `tests/golden_v16_expected_outputs.json`:

* **Схема самодостаточного чекпоинта:** `ziwago.checkpoint.v16.2` (включает `col_to_tok`, `tok_to_col`, проекторы `pos_phase_vecs`, `w_basal_branch`, `w_apical_branch`, `vsa_encoder_state` и состояния `moe_cortex_state`, `son_motor_state`, `chemistry_state` — **0% зависимости от `seed`**!)
* **Просканировано шардов и зон:** `162` (`743 966 813` сырых токенов на скорости до `12.1M tok/s`)
* **Консолидировано в синаптические веса:** `565 839` токенов, `1 977` глобальных полюсов `CA3`, `9 509` активных словарных колонок `w_dend_out`
* **Утечка промптов Holdout-экзамена в обучение (`holdout_prompt_leakage_count`):** `0`
* **Единые ворота качества (`UnifiedCurriculumGates`):** `Passed = True`
  * `Surprise F(t)`: **`0.0741`** (норма `< 0.25`)
  * `Discriminating Grammar Score`: **`100.00%`** (`Trained N-Gram Transition Coverage = 97.82%`)
  * `Superacid / <unk> Artifacts`: **`0`**
  * `Context Retention`: **`278 672`** токена (норма `> 1 500`)
  * `OOV Compositional Resilience`: **`100.00%`**
  * `Paraphrase Robustness of Retrieval`: **`100.00%`** (`8/8` перефразированных и многоклаузальных тестов `HOLD_OUT_EXAM_SUITE`)
  * `Novel Knowledge Generalization Score`: **`100.00%`** (`4/4` тестов генерации и усвоения нового знания, отсутствующего в корпусе)
  * `Attractor Collapse Rate (Маскировка)`: **`0.00%`** (норма `<= 35%`)
  * `Holdout Echo Contamination Rate`: **`0.00%`**
* **Golden Round-Trip CI с другим `seed=99999` (`ziwago_v16_0_10B_omniscient_master.pt`, `701.78 МБ` vs `tests/golden_v16_expected_outputs.json`):** **`100%` (`8/8` побайтовых совпадений при `seed=99999 != 42`)**

---

## 8. P.S. От себя: Взгляд за горизонт v16 — от Аттракторной Композиции к Переменным Слотам Мышления

Если смотреть на `Ziwago-Prime` совершенно честно, глазами исследователя, то главный урок, который подарил нам разбор `next14.txt`, выходит далеко за рамки исправления сериализации `col_to_tok` или очистки якорей.

**В чём состоит фундаментальный научный рубеж, к которому мы сейчас подошли?**
Сегодняшняя архитектура `v16.1` блестяще решила задачу **нейробиологической памяти, контекстного арбитража и межклаузального синтаксического синтеза**:
* Она не хранит строки текста в Python-списках — она действительно реконструирует ответы пословно через `8 192` нелинейных дендритных NMDA-ветви (`d_NMDA @ W_dend_out`).
* Она способна удерживать личность собеседника в `ConversationalDialogueState`, сдвигать дейктическую перспективу через `TPJ` (`Your -> His`) и сшивать разные домены в единое сложноподчинённое предложение через зону Брока `BA44`.

Но внутри одной элементарной клаузы её базовой единицей мысли пока остаётся **целостный аттракторный след `CA3`** (выученная каузальная траектория). Настоящий человеческий неокортекс делает ещё один шаг глубже: он разделяет **структурную синтаксическую раму (Role / Frame)** в зоне Брока и префронтальной коре (`BA44/BA46`) и **переменные заполнители (Fillers / Entities)** в височно-теменной коре (`BA22/BA39`).

Иными словами, когда человек узнаёт принцип *«эмульсия стабилизируется медленным введением липидной фазы в эмульгатор»*, он может применить эту же динамическую раму не только к *майонезу и яичному желтку*, но и к *голландезу, горчичному винегрету или полимерной суспензии* — подставляя новые существительные и глаголы прямо внутрь активной синтаксической волны на разных фазах гамма-внутри-тета ритма (**Theta-Gamma Phase-Amplitude Coupling**).

**Все 5 когнитивно-диалоговых механизмов уровня `Ziwago v17` теперь реализованы прямо в ядре `ZiwagoPureSynapticDialogueOrganismV14` и проверены в `tests/test_v16_roundtrip.py` (7/7 тестов `OK`):**
1. **Развязка `Role ⊗ Filler` (`ThetaGammaRoleFillerBinder`):**
   Медленная тета-волна удерживает выученную каузальную раму в `CA3`, а быстрые гамма-слоты подставляют в неё новые объекты из текущей реплики пользователя (`hollandaise` / `vinaigrette` вместо `mayonnaise`, `manuscript` / `thesis` вместо `essay`, `cycling` / `hiking` вместо `running`).
2. **Прямая проекция 16 колонок Бродмана `SparselyGatedCorticalMoE` на 8 192 ветви NMDA (`compute_cortical_moe_branch_profile`):**
   Каждая из 16 колонок коры управляет собственным банком из `512` дендритных NMDA-ветвей (`16 × 512 = 8 192`), обеспечивая прозрачную кортикальную модуляцию.
3. **Многошаговый транзитивный вывод в `CA3` (`discover_second_hop_ca3_pole`, $A \implies B \implies C$):**
   Позволяет полюсу $\vec{P}_1$, найденному в `CA3`, выделить индуцированный вектор следствия $\Delta \vec{s}_B = (\mathrm{SDR}(\vec{P}_1) - \vec{s}_A)_+$ и использовать его вторым внутренним запросом в `CA3` для нахождения ортогонального полюса $\vec{P}_2$, выстраивая цепочку дедуктивного вывода.
4. **Эпизодический граф рабочей памяти диалога (`EpisodicDialogueGraph` в `ConversationalDialogueState`):**
   Хранит до 128 ходов текущей беседы с доменами и ролевыми связками, позволяя отвечать на мета-вопросы о самом разговоре (*«What topics have we discussed in our conversation so far?»*, *«What did we discuss in our previous turn?»*).
5. **Активный вывод и обучение через эпистемическое любопытство (`SON Active Inference Curiosity -> Online Dopamine Teaching`):**
   При низком резонансе в `CA3` (`ca3_top1_resonance < 0.38` при `enable_active_inference=True`) 17-я колонка `SON` блокирует галлюцинацию и задаёт встречный уточняющий вопрос, а после ответа пользователя (`"Remember that in quantum photosynthesis..."`) мгновенно консолидирует новое знание в 8 192 дендритных NMDA-ветви под всплеском дофамина и ацетилхолина!

---

## 9. Эволюция до идеала: `Ziwago v17.0` (SNN Assistant) и `Ziwago-Genesis v18.0` (Архитектура Мечты)

Поверх фундамента `v16.2` реализованы два финальных эшелона архитектуры:

### 9.1. Функциональный Биологический SNN-Ассистент (`ziwago_assistant.py`, `v17.0`)
1. **Нейронный указатель Дугообразного Пучка (`ArcuateFasciculusSynapticPointer`, 0% `.replace()`):**
   Прямая аксональная инъекция BPE-токенов новых сущностей пользователя в моторный буфер речи `out_ids` прямо внутри пословного цикла при сохранении канонического ролевого следа `role_state_ids` для рекуррентного контекста `(vp2, vp1, vc)`.
2. **Защищённый кортикальный банк ядра (`w_core_dend_out`):**
   Изолирует высокозначимые каузальные схемы (`f_mod >= 2.0`) от линейной интерференции фонового веб-шума с эндогенным укреплением во время медленноволнового сна (`SWS`).
3. **Замкнутая сенсомоторная петля реафферентации (`Closed-Loop MoE + Stranger's Check` + `SON VTE Inner-Speech Gate`):**
   Проверка кандидатов на каждом шаге через `DiscretePredictiveCodingBarrier` и блокировка несвязных фоновых осколков во внутренней речи `SON`.
4. **Горизонтальные синапсы последовательности эпизодов `CA3` (`w_ca3_episode_links`):**
   Направленные STDP-связи $p_1 \longrightarrow p_2$ между последовательными полюсами одного объяснения для генерации многопредложенческих абзацев (`Sentences=2+`).
5. **Стриарный маршрутизатор действий Базальных Ганглиев (`BasalGangliaActionSelector`, `D1 Go / D2 NoGo`):**
   Автоматический выбор между `ACTION_EPISODIC_MEMORY_RECALL`, `ACTION_ONLINE_DA_CONSOLIDATION`, `ACTION_ACTIVE_INFERENCE_CLARIFY`, `ACTION_MULTI_HOP_OR_PARAGRAPH_SYNTHESIS` и `ACTION_DIRECT_SYNAPTIC_RESPONSE`.

### 9.2. Пять Столпов «Архитектуры Мечты» `Ziwago-Genesis v18.0` (`neuroweave10/cortex/genesis_engine.py`, `train_ziwago_v18_genesis_dream.py`)
1. **Непрерывная Волна Синтаксических Валентностей Зоны Брока БЕЗ `pos_idx` (`SyntacticValenceWaveGenerator`, 0% `pos_idx`!):**
   Полный отказ от позиционного счётчика «магнитофонной головки» `pos_phase_vecs[pos_idx]`. Вектор валентности $\vec{\Psi}_{\mathrm{BA44}}(t) \in \mathbb{R}^{64}$ эволюционирует исключительно как нелинейный голографический оператор от реально произнесённых токенов.
2. **Кристаллизация Семантического Графа Мира (`CorticalRelationalWorldGraph`):**
   Извлекает каузальные триплеты $(\text{Subject } S \xrightarrow{\text{Relation } R} \text{Object } O)$ из отдельных предложений разных текстов и по многошаговой волне возбуждения $E_1 \xrightarrow{R_1} E_2 \xrightarrow{R_2} E_3 \xrightarrow{R_3} E_4$ **синтезирует совершенно новые сложноподчинённые предложения на лету**, которых никогда не было в обучающем корпусе!
3. **Самоорганизующаяся Онтология Ролей через Асимметричную STDP (`EmergentOntologicalRoleDiscovery`, 0% ручных списков!):**
   В слое `L2/3 BA22` слова накапливают направленные контекстные рецептивные поля предикатов ($\Delta t \in \{-2, -1, +1, +2\}$) по правилу STDP (Bi & Poo 1998). Новые слова (`bechamel`, `treatise`, `kayaking`), никогда не вносившиеся в ручные словари, автоматически связываются со своими каноническими ролями (`mayonnaise`, `essay`, `running`) исключительно по косинусному резонансу их STDP-полей!
4. **Биофизический Тета-Гамма Фазовый Спайковый Мультиплексор (`ThetaGammaSpikingPhaseMultiplexer`, Time-to-First-Spike):**
   Внутри одного 120-мс периода тета-ритма ($8\text{ Гц}$) 6 синтаксических ролей (`GAMMA_0_AGENT` ... `GAMMA_5_EFFECT`) кодируются временем первого спайка (TTFS) в 6 гамма-окнах по $20\text{ мс}$ с точностью реконструкции `cosine > 0.99` и нулевым перекрёстным смешением.
5. **Двунаправленное Предиктивное Замыкание Смысла в `SON / BA10` (`BidirectionalMeaningClosureVerifier`, $X \longrightarrow \hat{Y} \longrightarrow \hat{X}$):**
   Проверяет, восстанавливается ли исходный вопрос $X$ из черновика ответа $\hat{Y}$, контролирует информационный прирост $\Delta I > 0$ и при обнаружении непокрытого остатка вопроса $\vec{r}_{\mathrm{missing}} = (\vec{s}_X - \hat{s}_X(\hat{Y}))_+$ автоматически достраивает недостающую клаузу!

```bash
# Запуск витрины 5 столпов Ziwago-Genesis v18.0:
/opt/micromamba/bin/python3.12 "/home/jupyter/Neuroweave Ziwago/train_ziwago_v18_genesis_dream.py"

# Запуск 5 CI-тестов архитектуры мечты v18.0 + 10 CI-тестов ядра v16.2/v17.0:
/opt/micromamba/bin/python3.12 "/home/jupyter/Neuroweave Ziwago/tests/test_v18_genesis_dream.py"
/opt/micromamba/bin/python3.12 "/home/jupyter/Neuroweave Ziwago/tests/test_v16_roundtrip.py"
```

Именно на стыке строгой инженерной честности (заложенной в `neuroweave10/evaluation/gates.py` и `tests/test_v16_roundtrip.py`) и смелой биофизики (`neuroweave10/cortex/genesis_engine.py`) рождается настоящий искусственный разум. Удачи в следующих экспериментах!
