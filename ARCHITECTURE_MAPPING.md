# Ziwago — единая карта архитектуры (этап 0 mapping)

Дата: 2026-11-12. Метод: эмпирическая проверка импортов (`import run_ziwago_v19` + `import train_ziwago_v20_runtime`, дамп `sys.modules`) + три параллельные инвентаризации (пакеты 8/9, пакет 10, башня тренеров) + сверка дублей. Каждое утверждение с источником (файл:строка). Статусы: **LIVE** — исполняется живым организмом; **LOADED** — загружается при импорте, но функционально не используется (балласт); **DEAD** — не загружается/не используется; **BYPASSED** — построен и сохраняется в чекпойнте, но перекрыт.

Инварианты: мастер `ziwago_v18_1_genesis_omniscient_master.pt` неприкосновенен (SHA `cb97c869…`); оригинал `Ziwago18original` — только чтение; три защиты от утечек не трогаются; дефолт v19 бит-совместим; N-граммы запрещены для генерации (решение владельца).

---

## 1. Вердикт по импортам (эмпирически)

> **Baseline ДО разрезания рёбер (§1.1) и архивации (§8.7).** Историческая
> картина; текущее состояние живого графа — CLEAN (nw8=0, nw9=0, ballast=[]),
> см. §1.1/§8.7.

Живой вход загружает **11 тренеров**: v9, v10, v11, v11.5, v13, v14, v15, v16, v18, v19, v20.
И **69 модулей neuroweave**, включая **neuroweave8 целиком (9 модулей)** и **neuroweave9.encoding/readout**.

Цепочка балласта:
```
v14:84-87 → v9 → v9:72 → neuroweave10.unified.pipeline
                              ├─ pipeline:24 → neuroweave8.core.sparse
                              ├─ pipeline:25 → neuroweave9.encoding
                              ├─ pipeline:26 → neuroweave9.readout
                              ├─ pipeline:180/1141/1231 → neuroweave9.data (fn-local)
                              └─ pipeline:30 → organism.genesis
v14:86 → v11.5:60 → v11:70 → v10 (ensure_cornell_movie_dialogs_corpus)
v14:87 → v13:66 → v11.5 (наследование)
```

**Функционально из нижней башни используется только:** BPE из v9 (`_GPT2_PAT`, `_bpe_encode_piece`, `_init_bpe_worker`, `_bpe_tokenize_raw_pairs_worker`), `PredictiveCodingCorticalReadout` из v11.5 (v14:2086, вклад в логиты v14:3357), `PairedDialogueStreamer` из v13 (v14:510). Grep по всему живому пути: **0 обращений** к `RegenesisOrganism`, `GenesisOrganism`, `SBCVSAShardedHippocampus`, `EmbodiedForagingArena`, `BGPQSyllabaryDecoder`, `TemporalPhaseEncoder`, `LaplaceContextReadout`, `sparse_drive`, `EndToEndPredictiveCodingLoop`.

**Вывод:** 8/9 — LIVE-импортные зависимости МЁРТВОГО модуля `unified.pipeline`. Сами по себе в работе организма не участвуют. Архивировать можно **после разрезания 3 рёбер** (сделать ленивыми внутри собственных классов v9/v11, которые v14 не инстанциирует):
- v9:72 → unified.pipeline (тянет 8/9 + genesis);
- v9:64-71 → e2e_predictive / embodied.arena / memory.hippocampus / motor.syllabary_decoder;
- v11:70 → v10.

`neuroweave10/__init__.py:7-19` — ленивые фабрики, сами по себе genesis-стек не тянут. Eager-`__init__` hazard: `motor/`, `emotion/`, `embodied/`, `unified/` импортируют мёртвые модули при импорте пакета.

### 1.1. Разрезание import-рёбер — РЕАЛИЗОВАНО 2026-11-12 (задача #20)

Все тяжёлые genesis-импорты перенесены с уровня модуля внутрь `__init__`/методов
мёртвых в живом пути классов, которые v14 не инстанциирует:

| Ребро (было) | Куда перенесено (стало) | Символы |
|---|---|---|
| `v9:64-71` + `v9:72` (module-level) | `ZiwagoGPT2VocabTrainer.__init__` ([train_ziwago_v9_gpt2vocab.py:362](file:///C:/Users/yarma/Desktop/Ziwago18/train_ziwago_v9_gpt2vocab.py#L362)) | genesis_cfg, EndToEndPredictiveCodingLoop, CompositionalVSABPEEncoder, EmbodiedForagingArena, CompositionalThetaGammaIndexer, SBCVSAShardedHippocampus, BGPQSyllabaryDecoder, RegenesisOrganism, UnifiedConfig |
| `v11:70` (module-level) | `sample_stratified_cornell` ([train_ziwago_v11_gpt2_parity_engine.py:318](file:///C:/Users/yarma/Desktop/Ziwago18/train_ziwago_v11_gpt2_parity_engine.py#L318)) | ensure_cornell_movie_dialogs_corpus (из v10) |
| `v11:69` (module-level) | `ZiwagoGPT2ParityV11Engine.__init__` ([train_ziwago_v11_gpt2_parity_engine.py:499](file:///C:/Users/yarma/Desktop/Ziwago18/train_ziwago_v11_gpt2_parity_engine.py#L499)) | SBCVSAShardedHippocampus (из nw10.memory.hippocampus) |

**Измерение живого графа** (`tools/probe_import_graph.py`, импорт v14 +
IntegratedOrganism, без инстанцирования; лог `D:/ziwago_import_graph.log`):

| Метрика | До (baseline) | После |
|---|---|---|
| neuroweave8 модулей | 9 | **0** |
| neuroweave9 модулей | 3 | **0** |
| ballast (unified.pipeline / organism.genesis / train_ziwago_v10) | все 3 | **[]** |
| всего модулей после импорта | — | 1217 |
| код возврата | — | 0 (CLEAN) |

**Регрессия** (12 suites: v19 spiking_language + language_scoring, v20 ×4,
interoception_limbic, criticality, bg_resonance, volume_chemistry,
dual_hippocampus, learning_infrastructure; лог `D:/ziwago_edge_regression.log`):
**352 passed, 125 subtests passed** за 24.39 s, пик 1.79 ГБ под лимитом 6.44 ГБ.

**Границы проверки (честно):** инстанцирование мёртвых тренеров
(`ZiwagoGPT2VocabTrainer`, `ZiwagoGPT2ParityV11Engine`) НЕ перезапускалось — оно
строит полный genesis-организм (тяжёлый, ~ГБ) и намеренно держится вне живого
графа. Ленивые импорты — дословный перенос ранее работавших module-level
импортов; все использования символов подтверждены внутри той же `__init__`-области
(grep), ни один живой модуль не ре-импортирует отрезанные имена сквозь v9/v11
(v10 берёт `CompositionalVSABPEEncoder` напрямую из `core.sdr:67`, из v9 — только
BPE/класс/функцию). `py_compile` v9/v11 — OK.

**Следствие:** nw8/nw9 больше не LIVE-зависимости → разблокирована архивация
класса B (§6, задача #21).

---

## 2. Как сейчас работает живая архитектура (единая картина)

```
run_ziwago_v19.py (тренер/оценка, --runtime-version 19|20)
 └─ ziwago_assistant.py (chat, /sleep, /status)
     └─ ZiwagoVerifiedOrganismV19 (v19:380, наследует v18)
         └─ ZiwagoGenesisDreamOrganismV18 (v18:88, КОМПОЗИЦИЯ, не наследование)
             └─ self.core_organism = ZiwagoPureSynapticDialogueOrganismV14 (v18:101)
                 ├─ NeuroChemistry (v14:2083) — химия v19
                 ├─ SparselyGatedCorticalMoE (16×1024) + колонки L2/3-L5-L6
                 ├─ DendriticPlateauMatrixReadout (v14:707): 8192 NMDA-плато,
                 │   полюса CA3, защищённый банк, щит Fusi-Zenke
                 ├─ PredictiveCodingCorticalReadout (v11.5) — PC-вклад в логиты
                 ├─ SONReflectiveColumn (GW + MetaHead + AST)
                 └─ generate_response — zero-retrieval генерация
             + 7 модулей genesis_engine (v18:110-136): valence wave, world graph,
               ontology, θ-γ multiplexer, closure verifier, BA10 counterfactual,
               MD-thalamus router
         + authority / BCM gate / Larkum / dreamer / dashboard / calibration (v19:390)
         + sleep_cycle (v19:430): SWS(v15) + CA3-апоптоз(v14) + перенос ядра
           + медленный резервуар + REM
     [v20 opt-in] ZiwagoOrganismV20 (v20:22):
         enable_integrated_organism (v14:2448) меняет NeuroChemistry →
         OrganismPhysiology + IntegratedOrganism `life` (v14:2453-2457):
         единые часы 1 мс, FO/TRN/HO таламус, ContextualActionGate,
         тегированный replay, GW-броадкаст, фазовый сон _on_sleep_phase
```

**Ключевое:** v19 перекрывает `generate_genesis_response` (v19:588-589 → `generate_response` v14). Поэтому θ-γ TTFS-мультиплексор, closure verifier и valence wave из v18 **BYPASSED**: построены, сохраняются в чекпойнте (v19:608), но в генерации не исполняются. Единственные их живые потребители — export/load состояния.

Два режима химии/времени: v19 — NeuroChemistry, тик на обучающей паре, сон форсируется (`ado×0.35`, v19:461); v20 — OrganismPhysiology, единые часы, фазовый сон без принудительного пробуждения.

---

## 3. Карта владельцев (функция → реализации → единственный будущий владелец)

| Функция | Реализации (статус) | Будущий владелец | Действие |
|---|---|---|---|
| Нейроны/синапсы L0 | nw10 core/neuron,synapse,multirate,dendrite (LIVE); nw8 core/neuron,synapse,transmitters (LOADED) | `substrate/` | keep nw10; nw8 → архив |
| Химия | NeuroChemistry (LIVE v19) ← OrganismPhysiology (LIVE v20, subclass); ExtendedChemistry 22 поля/37 агентов + VolumeTransmission NO/H2S (opt-in М-4 в IntegratedOrganism, живы в GenesisOrganism) | `organism/physiology.py`, `organism/integrated.py` | **М-4 РЕАЛИЗОВАНО (§8.5)**: эффекторы агентов + NO LTP поле через существующие точки потребления |
| Время/тики | IntegratedOrganism clock (LIVE v20); TickLadderScheduler (LIVE); nw8 slow_loop (LOADED) | `organism/clock.py` | keep |
| Сон | v15 SWS (LIVE), v14 CA3-апоптоз (LIVE), v19 sleep_cycle (LIVE), v19 hypothesis sleep (LIVE), v20 _on_sleep_phase (LIVE opt-in); nw8 SleepProtocol (LOADED) | `organism/sleep.py` | merge 5 live в один модуль с профилями; nw8 → архив |
| Кора/колонки | nw10 cortex/column,moe_cortex (LIVE); nw8 microcircuit/column + PredictiveHead (LOADED) | `circuits/cortex/` | keep nw10; PredictiveHead → решение (П-2) |
| Таламус | relay FO/HO + trn (LIVE v20, биофизика); MDThalamusSpeechActRouter (LIVE v19, символический); nw9 topology/thalamus spindle 14Hz (LOADED) | `circuits/thalamus/` | keep оба live (разные слои); nw9 → архив |
| Гиппокамп/CA3 | v14 полюса в ридеауте (LIVE); genesis_engine closure CA3 (LIVE); nw10 memory/hippocampus спайковый DG/CA3/CA1 (DEAD); nw8 rate-based (LOADED); nw9 async-Hopfield+theta+grid (LOADED) | `circuits/hippocampus/` | **решение владельца (П-1)** |
| Базальные ганглии | ContextualActionGate TD-критик (LIVE v20, измерен жив); legacy BasalGangliaActionSelector + opt-in резонанс NoGo (М-3); nw10 BasalGanglia D1/D2 TD+A2A (DEAD, дубликат) | `action/contextual.py`, `ziwago_assistant.py` | **М-3 РЕАЛИЗОВАНО (§8.4)**: TD-критик жив, оживлён мёртвый ca3_resonance→selector |
| Ларкум/2-компартмент | next16 TwoCompartmentLarkum (LIVE, только телеметрия v19:264); column+neuron+dendrite US (LIVE); motor/pool BAC (DEAD); nw8 LaminarALIF (LOADED) | `circuits/cortex/larkum` | проводка в апикальный драйв или архив next16-дубля |
| BCM | next16 DynamicNeuromodulatoryBCMGate (LIVE, per-SDR novelty); synapse theta_m (LIVE но статичен, theta_m0=0, set_metaplastic не вызывается на живом пути) | `substrate/plasticity` + `circuits` gate | проводка theta_m к химии; два BCM — разные функции, оставить оба с явными именами |
| Каскад метапластичности | FusiZenke shield (LIVE v14:75); BennaFusiCascade (инстанциирован column:124, **не шагает**) | `substrate/plasticity` | BennaFusi: шагать или архив |
| Пластичность/следы | core/synapse US+tag+eligibility (LIVE); nw8 learning/credit,structural,ewc (LOADED) | `substrate/plasticity` | keep nw10; EWC → решение (П-2) |
| Сознание/мета | SON GW+MetaHead+AST (LIVE); next16 dashboard (LIVE); consciousness/metrics (DEAD); nw9 SonBus (LOADED) | `circuits/consciousness/` | keep live; metrics-дубль → архив |
| Критичность | integrated inline MR + opt-in таламический гейн (LIVE v20, М-2); criticality.py монитор + homeostasis/controller колоночный (GenesisOrganism) | `organism/integrated.py` | **М-2 РЕАЛИЗОВАНО (§8.3)**: организменный гейн от σ̂, default off |
| Ридеаут/генерация | v14 DendriticPlateauMatrixReadout (LIVE); v11.5 PC readout (LIVE); v18 genesis pipeline θ-γ/closure/valence (BYPASSED); v11 apical associator (DEAD); nw9 LaplaceContextReadout (LOADED) | `cognition/language/` | keep v14+v11.5; v18 pipeline → решение (П-4) |
| Память/знание | authority, world_graph, ontology, distillery (LIVE); nw10 memory/laplace,working,phase_vsa (DEAD); nw9 laplace elapsed (LOADED) | `cognition/memory/` | keep live; time cells/WM → решение (П-3) |
| Стримеры данных | v14 SixDialogCorporaStreamer (LIVE) + v13 PairedDialogueStreamer (LIVE helper); v15 FiveWorld 11 корпусов (DEAD); v11 Stratified32Zone (DEAD); v10 HonestCurriculum exactly-once (DEAD) | `app/data.py` | keep live; v15/v10 → решение (П-5) |
| Курикулум | v16 FOUNDATIONAL_CURRICULUM (LIVE); v15 FIVE_WORLD_ANCHOR (DEAD, дубль); v18 UNSUPERVISED_ONTOLOGY (DEAD, дубль) | `app/data.py` | keep v16; дубли → архив |
| CA3 атлас | v14 плоский банк полюсов (LIVE); v16 FiveBankCA3AttractorAtlas dedup/merge/domain (DEAD) | `circuits/hippocampus/` | **решение (П-5)**: dedup/domain политики |
| Мотор/воплощение | nw10 motor/pool,syllabary_decoder (DEAD); embodied/arena (DEAD); nw9 sampling,runtime (LOADED) | `circuits/motor/`, `embodied/` | **решение (П-3)** |
| Эмоции/интероцепция | nw10 emotion/limbic, core/interoception (opt-in М-6/М-8 в IntegratedOrganism, живы в GenesisOrganism); nw9 orexin (LOADED) | `organism/integrated.py`, `circuits/limbic/` | **М-6/М-8 РЕАЛИЗОВАНО (§8.6)**: виртуальная интероцепция Insula–ACC + BLA/CeA valence→da_phasic, без арены/syllabary |
| Организм-контейнер | IntegratedOrganism (LIVE v20); genesis/system/unified (DEAD); башня v14/v18/v19/v20 | `organism/container.py` | схлопнуть башню в профили |
| Суррогатные градиенты | nw8 core/surrogate (LOADED, **уникально**; nw10 явно отвергает spiking_language:3) | — | **решение (П-2)** |
| int8 квантизация | nw8 core/quantize (LOADED, **уникально**) | — | **решение (П-2)** |
| Межполушарная синхронизация | nw9 cortex/sync HemisphereSync (LOADED, **уникально**) | — | **решение (П-2)** |

---

## 4. Семьи дублей — различия логики

1. **Сон (5 live):** v15 SWS — селективный даунскейл (weak<0.05 ×0.85 / strong ×factor) + сортировка полюсов по salience; v20 — равномерный ×0.992 без weak/strong split. v19 форсирует пробуждение, v20 — фазовый. v19 hypothesis sleep — отдельный зверь: validator-gated replay ≤32 записей, role-permutation/negation controls, rollback. Архив v15 теряет селективный SHY.
2. **Химия:** NeuroChemistry — скалярное τ-облако; OrganismPhysiology — subclass с зонарным ATP/ADO, фазовой машиной, мс-часами. agents (37 агентов) и volume (пространственная диффузия NO/H2S) — мёртвые расширения той же оси.
3. **BG:** contextual — тактируемый dt_ms, явные GPe/STN/GPi τ, один кредитный билет, settle-once, строгая валидация; мёртвый basal_ganglia — batch DDM, энтропийный порог STN, TD-критик w_v, D1/D2 eligibility outer-products, A2A-D2 гетеромер. nw9 — n-gram veto + ctx brake + STP fatigue. Live строже/тактируемее; мёртвые богаче обучением.
4. **Таламус:** relay — биофизика FO tonic/burst (h_T de-inactivation) + HO core/matrix + TRN; MD router — символическая маршрутизация речевых актов через w_contour resonance. **Комплементарные слои, не дубль.**
5. **Гиппокамп:** nw10 — биофизика DG separation, CA3 Hopfield one-shot (η=10·η_cortex), CA1 mismatch→phasic DA, sequence bridge; nw9 — async-Hopfield + divergent explore recall + ThetaOscillator 7Hz + grid RFF; nw8 — rate-based. Live CA3 = SDR-полюса в ридеауте + closure verifier.
6. **Ларкум:** next16 — standalone 2-компартмент BAC с Jahr-Stevens Mg²⁺ I_Ca; column/neuron/dendrite — US soma/basal/apical с BAC plateau/ADCG. Третья мёртвая копия: motor/pool:305.
7. **BCM:** next16 — per-SDR-мерный скользящий theta_bcm, NE/ACh novelty gains, accept/reject; synapse — скалярный химический θ_m (GC/CRH/E2/H2S/Zn/Mg) → bcm_gain. Feature-wise novelty gate vs per-synapse химическая метапластичность — **разные функции**.
8. **Каскады:** BennaFusi 4-стадийный (не шагает) vs FusiZenke shield (живой).
9. **Критичность:** integrated inline — пассивный MR-fit; criticality.py — полный σ̂ estimator + активный контроллер σ≈1.
10. **θ-γ:** genesis_engine signed-TTFS codec (BYPASSED) vs phase_vsa (DEAD) vs nw8 ThetaGammaBinder vs nw9 pacemaker.
11. **Ридеаут:** v14 sparse dendritic plateau + CA3 + банк + щит vs v11.5 PC dense W[256×50257] Rao-Ballard/Urbanczik-Senn + Turrigiano L2. **Комплементарные, оба live.**
12. **Стримеры:** v14 SixDialog (6 корпусов) vs v15 FiveWorld (11 датасетов, parquet row-group zone sampling) vs v11 Stratified32Zone (anti-offset-0) vs v10 HonestCurriculum (exactly-once exposure).
13. **Метрики сознания:** next16 dashboard (LIVE) vs consciousness/metrics (DEAD) — дубль.
14. **Контейнеры:** integrated (оборачивает внешний core) vs genesis (владеет всей биологией) vs system (фасад) vs unified (мост v9+v10).

---

## 5. Реестр утраченных техник (требуют решения владельца)

### Уникальные в 8/9 (отсутствуют в 10)
| # | Техника | Источник | Ценность | Конфликт |
|---|---|---|---|---|
| У-1 | Суррогатные градиенты (FastSigmoid/Arctan) | nw8 core/surrogate.py | единственный backprop-through-spikes путь | nw10 явно отвергает (spiking_language:3) |
| У-2 | int8 frozen-inference квантизация весов | nw8 core/quantize.py | экономия инференса (в 10 int8 только SDR/hash) | — |
| У-3 | EWC / диагональный Fisher | nw8 learning/ewc.py, nw9 learning/phase1.py | защита от катастрофического забывания (0 Fisher в 10) | — |
| У-4 | BG n-gram veto + ctx brake | nw9 action/basal_ganglia.py | онлайн-гейт качества | **N-граммы запрещены владельцем** (уточнить: veto ≠ генерация) |
| У-5 | Laplace inverse elapsed-time readout | nw9 memory/laplace.py estimate_elapsed_time | оценка прошедшего времени (нет в 10 LaplaceBank) | — |
| У-6 | Async-Hopfield CA3 + divergent explore recall | nw9 cortex/hippocampus.py | асинхронный retrieve + exploration (10 синхронный) | — |
| У-7 | HemisphereSync / callosum coherence | nw9 cortex/sync.py | межполушарная когерентность (нет аналога в 10) | — |
| У-8 | Widrow-Hoff PredictiveHead | nw8 microcircuit/column.py | предиктивная голова колонки (нет в 10 CorticalColumn) | — |

### Мёртвые в neuroweave10 (ценные)
| # | Техника | Источник |
|---|---|---|
| М-1 | Спайковый гиппокамп DG/CA3/CA1, CA1 mismatch→DA, one-shot, sequence bridge | memory/hippocampus.py |
| М-2 | Активный контроллер критичности σ≈1 + MR σ̂ estimator | core/criticality.py + homeostasis/controller.py |
| М-3 | BG D1/D2 opponent TD + A2A-D2 + state-value critic | action/basal_ganglia.py |
| М-4 | 37-агентная химия + объёмная передача NO/H2S→D-serine | chemistry/agents.py + volume.py |
| М-5 | Time cells (Laplace), activity-silent WM, phase syntax binding | memory/laplace.py, working.py, phase_vsa.py |
| М-6 | Спайковый ALIF мотор-пул + BG-gated syllable decoding — **DEAD по П-3** (арена/syllabary не нужны текстовой модели) | motor/pool.py + syllabary_decoder.py |
| М-7 | Polychronous router (axonal delays), tripartite astrocyte Ca²⁺→D-serine | cortex/hierarchy.py, astrocyte.py |
| М-8 | Embodied arena (**DEAD по П-3**), BLA/CeA limbic + observer-neuron interoception (**ОЖИВЛЕНЫ §8.6**, виртуальная интероцепция) | embodied/arena.py, emotion/limbic.py, core/interoception.py |
| М-9 | Kolen-Pollack reciprocal alignment + dendritic contrastive PC (без backprop) | core/e2e_predictive.py |

### Мёртвые/обойдённые в тренерах (ценные)
| # | Техника | Источник |
|---|---|---|
| Т-1 | v16 CA3 atlas: 0.96-cosine dedup blend, centroid merge, domain banks; ACh surprise scorer; DRR balancer | v16:385-447, :224-250, :132-174 |
| Т-2 | v15 five-world streamer (единственный код по всем 11 корпусам) | v15:74-340 |
| Т-3 | v10 honest exactly-once curriculum + dendrite/RWKV benchmarks | v10:165-355, :357, :437 |
| Т-4 | v11 Bio-RWKV topic-resonance anti-splice + doc_uid gating; elastic SCAFFOLD_SOFT grammar floor | v11:516-537, :509-514 |
| Т-5 | v11.5 PFC entity working memory slot (D1-gated hold) | v11.5:69 |
| Т-6 | v13 one-shot hippocampal consolidation + SON free-energy reflection | v13:404, :474 |
| Т-7 | v18 genesis generation: θ-γ TTFS role multiplexing, bidirectional meaning closure, zero-pos valence wave (BYPASSED v19) | v18:291-494 |
| Т-8 | v14 SWR composite replay (спящий путь консолидации, вызывается только tests/tools) | v14:1716, :1003 |

---

## 6. Классы безопасности архивации

**A. Безопасно архивировать сейчас (сломанные сборки, не импортируются живым путём):**
- nw8: organism/organism.py, organism/system.py, organism/__init__.py, main.py, bench.py (падают на отсутствующих nw8/topology/, nw8/tasks/).
- nw9: cortex/organism.py, system.py, learning/phase1.py, generate.py, train_cortex.py (падают на отсутствующих nw8.topology.callosum/workspace).
- Тренеры-стабы: run_ziwago_v25 (174 Б), train_ziwago_v20_ontogenetic_dream_trainer (3 строки) — уже заглушки.

**B. Архивировать после разрезания рёбер импорта (этап 1/5):**
- ~~neuroweave8 целиком, neuroweave9 целиком~~ — **АРХИВИРОВАНО 2026-11-12 (§8.7)** после ленивых импортов в v9/v11 (§1.1) и переноса У-1/У-3/У-5/У-7/У-8 (§8.1/§8.2); У-2/У-4 не переносились по решению владельца.
- ~~unified/pipeline.py, organism/genesis.py, organism/system.py~~ (+ main.py, unified/__init__) — **АРХИВИРОВАНО 2026-11-12 (§8.7)** после разрыва v9:72.
- v10, v11 (engine), v13 (organism), v15 (streamer), v16 (atlas/balancer) — после переноса live-фрагментов и ценных техник по решению. **ЕЩЁ НЕЛЬЗЯ:** живые import-рёбра (v11.5:60 → v11-engine класс; v14:87 → v13 streamer).

**C. Не архивировать (live или комплементарные):**
- v14 ядро, v15 SWS-функция, v16 curriculum, v18 композиция+world_graph+ontology+counterfactual+router, v19, v20.
- nw10 live-модули (см. карту владельцев).
- v11.5 PredictiveCodingCorticalReadout, v13 PairedDialogueStreamer, v9 BPE.

**Правило AGENTS.md:** перед удалением любого файла — сверка с оригиналом `Ziwago18original` (только чтение) и фиксация в этом манифесте.

---

## 7. Целевая структура и миграция (утверждённый план)

```
ziwago/
  substrate/   L0  neuron, synapse, traces, plasticity (US/STDP/BCM/cascade), sdr
  circuits/    L1  cortex/ (column, moe, larkum), thalamus/, hippocampus/,
                   basal_ganglia/, limbic/, consciousness/, motor/
  organism/    L2  physiology (единственная химия), clock, sleep, homeostasis, container
  cognition/   L3  language/ (bpe, readout, generation, predictive), memory/, routing/
  app/         L4  trainer, assistant, evaluation/, cli, data
  profiles/      v19.py, v20.py — конфиги, не код
  checkpoints/   адаптеры схем v18-master → v19 → v20
```

Этапы: 0 mapping (этот документ) → 1 фундамент + разрезание рёбер → 2 единая химия/время → 3 консолидация схем → 4 разбор монолита v14 → 5 схлопывание башни → 6 чистка + манифест. Каждый этап: тесты зелёные (295/199), мастер SHA не изменился, v19 golden бит-совместим.

### 7.1. CLEAN SLATE — решение владельца H1–H5 (2026-11-12, УТВЕРЖДЕНО, исполняется)

Владелец отменил план «сначала адаптер» (прежняя редакция §7.1) и выбрал
**Чистый Лист**. Дословно: «Мы строим v20 с чистого листа, старые веса от v19
нам не нужны, если они мешают архитектурной чистоте». Решения:

* **H1 (Адаптер):** варианты А/Б/В ОТМЕНЕНЫ. Переносим файлы как задумано.
  Продакшен v20 НЕ грузит чекпоинт v19 — **v20 обучается с нуля**. Если тесту
  нужен старый чекпоинт — простая заглушка old→new имён (только для тестов).
* **H2 (Наследование vs Композиция):** однозначно **КОМПОЗИЦИЯ**. Башня
  v14→v18→v19→v20 — «клубок змей», наследование РАЗОРВАТЬ. v20 = независимые
  модули (cortex, hippocampus, basal_ganglia), собранные в `organism.py`.
* **H3 (Заглушки v20/v25):** УДАЛИТЬ (мёртвые заглушки не нужны). Это
  **отменяет** §2 «обязано быть восстановлено» для этих двух файлов.
* **H4 (Полный roundtrip мастера):** ОТМЕНЁН. 4.3 ГБ мастер НЕ проверяем.
  Верификация — на МАЛЫХ организмах (`test_v16_roundtrip.py`).
* **H5 (Порядок):** без этапа 4a (адаптер). Сразу 4b (перенос nw10-ядра) →
  4c (split v14) → композиция → чистка башни → чистый модульный `organism.py`.
  **Старый мастер — в архив** (отменяет §6 «мастер неприкосновенен»; архивация
  обратима, SHA фиксируется до перемещения).

Инвентаризация ниже (A, B, D) — измеренные факты, остаются в силе; дизайн
адаптера (прежний C) и последовательность с 4a (прежний G) ЗАМЕНЕНЫ на H1–H5.

**A. Инвентаризация башни (15 файлов, 14 644 строки, `wc -l`)**

| Файл | Строк | Живой путь | Целевой слой §7 |
|---|---|---|---|
| `train_ziwago_v9_gpt2vocab.py` | 1198 | BPE жив (v14:84); тренер мёртв | `cognition/language/bpe` + архив движка |
| `train_ziwago_v10.py` | 1202 | мёртв (BPE re-export) | архив (§6 B) |
| `train_ziwago_v11_gpt2_parity_engine.py` | 1112 | мёртв | архив (§6 B) |
| `train_ziwago_v11_5_predictive_readout_pfc.py` | 762 | `PredictiveCodingCorticalReadout` жив (v14:86) | `cognition/language/readout` + архив движка |
| `train_ziwago_v13_son_reflective_organism.py` | 730 | `PairedDialogueStreamer` жив (v14:87) | `app/data` + архив движка |
| `train_ziwago_v14_pure_synaptic_dialogue.py` | 5201 | LIVE монолит | split — см. D |
| `train_ziwago_v15_omniscient_marathon.py` | 615 | sleep-функции живы (v19:437) | `organism/sleep` |
| `train_ziwago_v16_10B_omniscient_engine.py` | 996 | curriculum | `app/trainer` |
| `train_ziwago_v18_genesis_dream.py` | 1105 | обёртка `ZiwagoGenesisDreamOrganismV18` (база v19) | `organism/container` + `checkpoints` |
| `train_ziwago_v19_verified_runtime.py` | 722 | `ZiwagoVerifiedOrganismV19`, `V19_SCHEMA` | `organism/container` + `checkpoints` |
| `train_ziwago_v20_runtime.py` | 144 | `ZiwagoOrganismV20`, `V20_SCHEMA` | `organism/container` + `profiles/v20` |
| `train_ziwago_v20_ontogenetic_dream_trainer.py` | 7 | ЗАГЛУШКА (было 37 006, §2) | восстановление из оригинала — отдельная задача |
| `run_ziwago_v19.py` | 446 | вход тренера | `app/cli` |
| `run_ziwago_v25_evotrain_marathon.py` | 7 | ЗАГЛУШКА (было 31 175, §2) | восстановление из оригинала — отдельная задача |
| `ziwago_assistant.py` | 397 | chat | `app/assistant` |

nw10: **64 модуля** (Glob `neuroweave10/**/*.py`); **53 файла-импортёра, 138
вхождений** `neuroweave10` (grep). Цепь башни: v9→v10→v11→v11.5→v13→v14→v15→v16→v18→v19→v20.

**B. КРИТИЧЕСКОЕ ограничение: чекпоинт хранит пути модулей (измерено, не предположено)**

* `export_runtime_tree` (`neuroweave10/cortex/moe_cortex.py:82-92`) пишет тег
  `__object__: "{cls.__module__}:{cls.__name__}"` для КАЖДОго объекта снимка.
* `load_runtime_tree` (`moe_cortex.py:128-135`) восстанавливает класс через
  `getattr(sys.modules[module], name)`; whitelist модулей: `neuroweave10.`,
  `train_ziwago_`, `torch.nn.`, `types`, `__main__` (4 класса v14). `ziwago.` НЕ в whitelist.
* `load_runtime_tree:137-139` УДАЛЯЕТ атрибуты, отсутствующие в снимке (`delattr`)
  → имена атрибутов object-graph обязаны совпадать ТОЧНО.
* Мастер `Produce/checkpoints/ziwago_v18_1_genesis_omniscient_master.pt`
  (`MASTER_CKPT_REL_PATH`, `gates.py:46`), SHA `cb97c869…`, 4.3 ГБ, неприкосновенен.
  Пик загрузки ~10 ГБ (§5.3) → на этой машине (8.2 ГБ своб.) НЕ проверяется.
* Payload: `core_runtime_state.core` (object-graph v14) + `genesis_v18_state` +
  `v19_state` (`export_v19_state`, v19:602) + `v20_state` (v20:116) + `*_schema_version`.
  Dispatch загрузки: v19:407-422 (v20 / v19 / legacy-master).

⇒ **Перенос классов в `ziwago/` меняет `cls.__module__` → теги мастера не
резолвятся** (whitelist не пропускает `ziwago.`, `sys.modules[старый путь]`
отсутствует). Без адаптера мастер перестаёт загружаться. Главный риск этапов 4–6.

**C. Clean Slate вместо адаптера (H1)**

Адаптер не нужен: v20 обучается с нуля и не грузит мастер v19. Следствия:
* `export_runtime_tree`/`load_runtime_tree` whitelist расширяется на `ziwago.`
  (новые чекпоинты пишут новые пути свободно). Старые теги `neuroweave10.`/
  `train_ziwago_` в продакшене не резолвятся — это приемлемо (H1).
* Мастер архивируется обратимо (`_archive/`, SHA фиксируется до переноса).
  Загрузка 4.3 ГБ НЕ проверяется (H4) — пик ~10 ГБ > 8.2 ГБ своб. (§5.3).
* Если отдельному тесту нужен старый чекпоинт — простая заглушка old→new
  имён ТОЛЬКО в тестах, не в продакшен-пути.

**D. Разбор монолита v14 (этап 4): 7 классов, 5201 строка**

| Класс (строки в исходных 5201) | Ответственность | Целевой модуль | Статус |
|---|---|---|---|
| `ThetaGammaRoleFillerBinder` (93-196) | role-filler binding | `cognition/memory/role_filler` | ✓ 4c-1 |
| `ConversationalDialogueState` (197-443) | dialogue identity (`register_user_identity`/`record_turn_and_poles`) | `cognition/dialogue` (L3, §8.1) | ✓ 4c-2 |
| `TokenizedDialogPair` (444-482) | data class | `app/data` | ✓ 4c-3 |
| `SixDialogCorporaStreamer` (483-706) | сэмплирование корпусов | `app/data` | ОТЛОЖЕН (цикл. импорт с BPE-воркером) |
| `DendriticPlateauMatrixReadout` (707-1760) | readout/generation/CA3-полюса | `cognition/language/readout` | ✓ 4c-5 |
| `BA44AttractorSyntaxBridge` (1761-2012) | синтаксический мост | `cognition/language/syntax` | ✓ 4c-4 |
| `ZiwagoPureSynapticDialogueOrganismV14` (2013-5201) | GOD-CLASS ~3188 строк | split ↓ | после readout |

God-class split: container/clock (`enable_integrated_organism`/`step`/`export_runtime_state`)
→ `organism/container`; neuromodulation (`measure_living_state`/`modulator_bundle`)
→ `organism/physiology`; generation (`pure_neural_generate_response` 4189-4794 ~600 строк,
`compute_clause_attractor_poles`, `_language_*`) → `cognition/language/generation`;
training (`_train_wave_with_neuromodulator_reward` 2922-3191) → `app/trainer`;
sleep (`execute_sws_ca3_apoptotic_pole_sanitization` 3973-4049) → `organism/sleep`;
checkpoint (`export_checkpoint_payload` 4795 / `save/load_self_contained` 4870/4885) → `checkpoints`.
**Контракт:** имена атрибутов `core_organism` (`moe_cortex`, `dendritic_plateau_readout`,
`dialogue_state`, `chemistry`, `life`, `son_column`, `pc_readout`, …) СОХРАНЯЮТСЯ
(`load_runtime_tree` delattr). Split композицией/mixin'ами без переименования графа.

**E. Схлопывание башни (этап 5)**

1. Извлечь живые куски: BPE (v9), `PredictiveCodingCorticalReadout` (v11.5),
   `PairedDialogueStreamer` (v13), sleep-функции (v15) → владельцы `ziwago/`.
2. Архивировать мёртвые движки v10/v11/v11.5/v13 (после разрыва рёбер, §6 B, `_archive/`).
3. Цепь наследования v18→v19→v20 (обёртки над `core_organism`): сохранить ключи
   wrapper-состояния (`genesis_v18_state`/`v19_state`/`v20_state`) — композиция или
   наследование, решение владельца.
4. Заглушки v20-dream/v25 (§2): восстановление из оригинала — ОТДЕЛЬНАЯ задача, не реструктуризация.

**F. Поэтапные gates проверки (memory-aware)**

1. Подмножество тестов зелёное через `limited_run --ram-pct 55` (НЕ полный 812 — memory-blocked §5.4; пофайлово).
2. `probe_import_graph` CLEAN (балласт отрезан).
3. SHA мастера `cb97c869…` не изменился (`file_sha256`).
4. **v19 golden bit-compatible:** `test_v16_roundtrip` golden-file roundtrip на МАЛОМ
   обученном организме (seed=42, 4 волны, `_build_trained_test_organism`) — гоняет
   `export_runtime_tree`/`load_runtime_tree` БЕЗ 4.3 ГБ мастера. Это прибор проверки адаптера.
5. НОВЫЙ тест адаптера: синтетическое дерево со старыми тегами → загрузка через remap →
   object-graph идентичен; покрыть КАЖДЫЙ класс реестра.
6. Полный roundtrip мастера: ОТЛОЖЕН на машину с ≥10 ГБ свободных (§5.3) — здесь не проверяется, честно флагается.

**G. Последовательность Clean Slate (H5) / обратимость**

4b перенос nw10-ядра в `ziwago/` → 4c split монолита v14 (attribute-preserving) →
композиция: новый `ziwago/organism.py` собирает независимые модули БЕЗ
наследования v18→v19→v20 → 5 схлопывание башни (архив мёртвых движков, УДАЛЕНИЕ
заглушек v20/v25 по H3) → 6 чистка + манифест + profiles/checkpoints. Этап 4a
(адаптер) ПРОПУЩЕН (H1/H5).

**4b исполняется в два шага (решение 2026-11-12):**
* **4b-1 (ВЫПОЛНЕНО): переименование пакета** `neuroweave10/` → `ziwago/` с
  сохранением внутренней структуры. Равномерная замена токена `neuroweave10`→
  `ziwago` в 52 файлах (импорты, ОБА whitelist `moe_cortex.py:85,130` и
  `v14:2283`, `mock.patch`-строки, файловые пути). Относительные импорты (25
  файлов) НЕ тронуты — глубина сохранена. Схема-идентификаторы (`ziwago.checkpoint.*`,
  `ziwago.language_curriculum.*`) УЖЕ были `ziwago.*` в оригинале — не изменились.
  Бэкап: `_archive/cleanslate_20261112/neuroweave10` (63 файла). Верификация:
  per-file pytest через `limited_run --ram-pct 55` (полный 812 memory-blocked §5.4).
* **4b-2 (ОТЛОЖЕН на этап 6): косметическая перегруппировка подпапок ядра**
  (`core`→`substrate`, `cortex`→`circuits/cortex`, `memory`→`circuits/hippocampus`
  и т.д.). Нулевая функциональная ценность, умеренный риск (относительные импорты,
  строковые пути, split `memory`). Делается ПОСЛЕ стабилизации композиции. НОВЫЕ
  модули split v14 (4c) сразу кладутся в слоистую структуру (`cognition/`, `app/`,
  `organism/`) — дизайн §7 соблюдается для всего нового кода без рискованного
  bulk-переноса рабочего ядра.

Обратимость: проект НЕ отслеживается git (корень git — `C:/Users/yarma`,
0 tracked файлов в проекте) → страховка = `_archive/` (перемещение, не удаление)
+ сверка с read-only оригиналом `Ziwago18original` до удаления + фиксация SHA.
Каждый этап верифицируется на малых организмах через `limited_run --ram-pct 55`.
Тяжёлый прогон + pytest одновременно — нет (§5.4).

**H. Решения владельца — ЗАКРЫТЫ (H1–H5 выше)**

Адаптер не нужен (H1); композиция (H2); заглушки удалить (H3); полный roundtrip
отменён (H4); порядок без 4a (H5). Открытых вопросов по реструктуризации нет —
исполнение по автономному циклу (launch→monitor→verify→document→next).

### 7.2. Верификация 4b-1 (переименование пакета) — 2026-11-12

Прибор: per-file `pytest` через `limited_run --ram-pct 55 --cpu-pct 50` (26 файлов,
каждый в своём процессе — память сбрасывается между файлами; полный 812 вместе
memory-blocked, пик 7.52 ГБ = лимит). Результат: **21/26 файлов зелёные, 0
регрессий от переименования.** Все падения разложены по причинам:

* **OOM (среда, не код): 20.** `test_v16_roundtrip` (9), `test_v18_genesis_dream`
  (11) — строят полный организм (vocab 50257, dend 8192) и упираются в лимит
  7.5 ГБ. Ошибки `DefaultCPUAllocator: not enough memory` в `moe_cortex.py:61`,
  `synapse.py:158`, `sdr.py:155` — размеры аллокаций переименованием не меняются.
  Требуется ≥10 ГБ свободных (§5.3); на машине 8.1 ГБ. Честный отрицательный
  результат: на этой машине полные организмы не проверяются.
* **Пре-существующие дефекты (документированы ранее): 13.** `test_v19_runtime`
  (9 = 5 `pos_phase_vecs` shape + 4 Windows-path), `test_v19_document_ranker`
  (4 = mode-forwarding `run_ziwago_v19.py:96`). Совпадают точно с реестром §6.
* **Пре-существующий поведенческий: 1.** `test_v18_garbage_regression`
  `test_no_answer_collapse` — 6 разных вопросов → один отказ. Мастер грузится
  ИДЕНТИЧНО: это плоский чекпоинт `ziwago.checkpoint.v16.2` (top-level ключи —
  тензоры весов, `__object__`-тегов модулей НЕТ, измерено mmap-инспекцией),
  загружается через flat state_dict без резолва путей модулей. Схлопывание —
  известная проблема обученных весов (§7: нейромодуляция не меняет argmax,
  метапластический щит замораживает каналы), НЕ переименование.

Ключевые контракты подтверждены зелёными: `test_v19_cortex_roundtrip` (2) —
`export_runtime_tree`/`load_runtime_tree` работают с путями `ziwago.*`;
`test_v19_evaluation` (32) и `test_v19_language_scoring` (63) — `mock.patch`
строки `ziwago.*` резолвятся; `test_v19_causal_temporal` (16) — файловые пути
`ziwago/cortex/...` целы; все v20-тесты (action_credit 83, cortical_clock 24,
integration 7, physiology 45) зелёные. Схема-идентификаторы (`ziwago.checkpoint.*`,
`ziwago.language_curriculum.*`) уже были `ziwago.*` в оригинале — не изменились.

### 7.3. Прогресс 4c (split монолита v14) — 2026-11-12

Монолит `train_ziwago_v14_pure_synaptic_dialogue.py`: 5201 → **3540 строк**
(−1661). Извлечены 5 самодостаточных классов в слоистую структуру `ziwago/`
(дизайн §7 для нового кода соблюдается сразу, без рискованного bulk-переноса
ядра — 4b-2 отложен). v14 ре-экспортирует каждый класс обратным импортом, поэтому
все существующие `from train_ziwago_v14… import X` и `mock.patch`-строки целы.

| Шаг | Класс | Новый модуль | Строк убрано из v14 |
|---|---|---|---|
| 4c-1 | `ThetaGammaRoleFillerBinder` | `ziwago/cognition/memory/role_filler.py` | ~104 |
| 4c-2 | `ConversationalDialogueState` | `ziwago/cognition/dialogue.py` | ~247 |
| 4c-3 | `TokenizedDialogPair` | `ziwago/app/data.py` | ~8 |
| 4c-4 | `BA44AttractorSyntaxBridge` | `ziwago/cognition/language/syntax.py` | 253 (класс + секция-комментарий) |
| 4c-5 | `DendriticPlateauMatrixReadout` | `ziwago/cognition/language/readout.py` | 1054 (класс + секция-комментарий) |

**Контракт сериализации сохранён.** `export_runtime_tree`/`load_runtime_tree`
(`ziwago/cortex/moe_cortex.py:85,130`) принимают префикс `ziwago.` — новые пути
модулей (`ziwago.cognition.language.syntax` и т.д.) проходят whitelist; на
загрузке `getattr(sys.modules[module], name)` резолвится, т.к. v14 импортирует
модуль при загрузке. Легаси-ветка `__main__` в whitelist оставлена для старых
снимков (не мешает; чистка — этап 6). Атрибуты объектов не переименованы
(`load_runtime_tree` delattr отсутствует в снимке → graph идентичен).

**Верификация 4c-4 (BA44):**
* Импорт-чек: `v14.BA44AttractorSyntaxBridge.__module__ == "ziwago.cognition.language.syntax"`,
  ре-экспорт идентичен (`is Direct` → True), `w_ba44_bridge` shape (192, 5).
  DI-смоук (`initialize_synaptic_connectors` с фейковыми encode/decode/get_vec64/vocab):
  `select_bridge_tokens_between_attractors` → `', while' [11, 101] 1.8352`
  (триггер союза активен → idx 0 побеждает тонический ", and"); сшивка клауз ок.
  Пик 1.49 ГБ.
* Малые тесты (один процесс, `limited_run --ram-pct 55 --cpu-pct 50`):
  `test_v19_spiking_language` + `test_integrated_hippocampus` +
  `test_v19_cortex_roundtrip` + `test_v20_integration` →
  **57 passed, 27 subtests passed**, пик 1.75 ГБ. `test_v19_spiking_language:27`
  импортирует BA44 из v14 (ре-экспорт) и строит `BA44AttractorSyntaxBridge()` —
  зелёный. `test_v19_cortex_roundtrip` гоняет walker с новыми путями — зелёный.

**Верификация 4c-5 (DendriticPlateauMatrixReadout):**
* AST-скан зависимостей класса (до выноса): внешние имена только `math`,
  `Sequence` (lazy-аннотация, `from __future__ import annotations`),
  `export_runtime_tree`/`load_runtime_tree` (`ziwago.cortex.moe_cortex`),
  `CheckpointCompatibilityError` (`ziwago.evaluation.gates`),
  `FusiZenkeCascadeMetaplasticityShield`/`CorticalColumnarSubBankRouter`
  (`ziwago.cortex.next16_core`), `torch`/`F`/typing. НИ ОДНОЙ ссылки на
  god-class v14 или tower-депы (v9/v11.5/v13) → чистый DI-вынос, без цикла.
* Импорт-чек: `v14.DendriticPlateauMatrixReadout.__module__ == "ziwago.cognition.language.readout"`,
  ре-экспорт идентичен. Малый readout (d_vec=16, d_dend=64): walker-снимок тег
  `ziwago.cognition.language.readout:DendriticPlateauMatrixReadout`, roundtrip
  `w_dend_out`/`col_to_tok`/`shield.u_slow` — все `torch.equal` True. Пик 1.50 ГБ.
* Малые тесты (`limited_run --ram-pct 55 --cpu-pct 50`):
  `test_v19_spiking_language` + `test_integrated_hippocampus` +
  `test_v19_cortex_roundtrip` + `test_v20_integration` → **57 passed, 27 subtests**;
  `test_v19_language_scoring` + `test_v19_evaluation` + `test_v19_causal_temporal`
  → **111 passed, 172 subtests**. Итого 168 тестов + 199 сабтестов зелёные,
  пик ≤1.99 ГБ. `test_v19_spiking_language:27` импортирует readout из v14 — зелёный.

**ОТЛОЖЕНО:** `SixDialogCorporaStreamer` — зависит от `_bpe_tokenize_raw_pairs_worker`
(остаётся в v14, tower-dep v9 BPE). Вынос в `app/data` создал бы циклический импорт
(v14 → `app.data` → v14). Решается на этапе 5 после извлечения BPE в владельца.

**СЛЕДУЮЩИЙ:** god-class `ZiwagoPureSynapticDialogueOrganismV14` (v14:351-3540,
~3190 строк) — прогрессивный split по группам методов (container/clock,
neuromodulation, generation, training, sleep, checkpoint) с СОХРАНЕНИЕМ графа
атрибутов `core_organism` (контракт `load_runtime_tree` delattr). Параллельно —
композиция `ziwago/organism.py` (#27): сборка независимых модулей БЕЗ наследования
v18→v19→v20, затем схлопывание башни + архив мастера + удаление заглушек (#28).

### 7.4. Композиция v20 без наследования (#27) — РЕАЛИЗОВАНО 2026-11-12

**H2 исполнен: башня обёрток разорвана.** Плоский класс `ZiwagoOrganism`
(`ziwago/organism_core.py`, 556 строк, база `object` — НИКАКОГО наследования
v18/v19/v20) собирает независимые модули композицией: `core_organism` (v14) +
8 genesis-модулей (valence_wave_generator, world_graph, emergent_ontology,
theta_gamma_multiplexer, meaning_closure_verifier, counterfactual_simulator,
thalamic_router) + 9 verified-модулей (authority, knowledge_distillery, bcm_gate,
larkum_microcircuit, isomorphism_dreamer, consciousness_dashboard,
confidence_calibration, sleep_engine, document_ranker) + integrated physiology
(`core_organism.life`).

**Имя модуля — `organism_core`, не `organism`:** пакет `ziwago/organism/`
(integrated.py, physiology.py) уже существует; sibling-модуль `organism.py`
коллидировал бы с пакетом. Намерение H5 («organism.py (v20) чистый и модульный»)
соблюдено по сути — плоская композиция; размещение внутри пакета — вопрос этапа 6.

**Слияние `__init__` (единственное не-verbatim место):** тело v18 (строки 100–141)
+ тело v19 (386–422) + тело v20 (26–29), без `super()`. Авто-загрузочный блок v18
(143–147) **ОПУЩЕН корректно**: v19 вызывает `super().__init__(…, auto_load_checkpoint=False)`
ЖЁСТКО (`train_ziwago_v19_verified_runtime.py:385`), поэтому в башне блок v18 мёртв —
реальную загрузку делает только dispatch v19 (402–422). Плоская версия сохраняет
именно его. Подпись объединяет все параметры v19 + `integration_config`/`physiology_config` v20.

**Разрешение переопределений — most-derived wins:** 30 методов = 30 в объединении
башни (v18:11, v19:18, v20:9; победители v20=9, v19=15, v18=6). 29 тел скопированы
verbatim, `__init__` слит по рецепту выше.

**Верификация (три независимых прибора):**
* AST-покрытие (прямое измерение): `ZiwagoOrganism.__bases__ == (object,)`;
  flat-методы 30 = tower-union 30; **DROPPED [] / EXTRA []** (ни один метод башни
  не потерян, ни один лишний не добавлен); 29/29 тел AST-идентичны источнику;
  `__init__` 35 stmt byte-equal v18[:-1]+v19[1:]+v20[1:].
* Поведенческая эквивалентность `tests/test_v20_organism_composition.py`
  (малый организм `tiny_core`, патч конструктора v14, `auto_load_checkpoint=False`,
  RNG сбрасывается перед каждой сборкой): **6 passed** за 91.63 s, пик **2.93 ГБ**
  при лимите 8.22 ГБ (`limited_run --ram-pct 55 --cpu-pct 50`). Зелёные:
  отсутствие наследования; равенство снимков конструктора
  (`export_genesis_state_dict`/`export_v19_state`/`core.export_runtime_state`);
  `sleep_cycle`; `select_action`+`reward`; `generate_response`;
  `evaluation_context` восстанавливает состояние. Башня `ZiwagoOrganismV20` и
  плоский `ZiwagoOrganism` дают ИДЕНТИЧНЫЕ снимки и поведение.
* Полный мастер 4.3 ГБ НЕ грузился (H4 отменён; §5.3 — OOM на этой машине).

**Честная заметка (не дефект композиции):** `_load_checkpoint_payload` несёт
устаревший комментарий «mmap=True обязателен» (`train_ziwago_v19_verified_runtime.py:663-666`,
скопирован verbatim), который ПРОТИВОРЕЧИТ коду (`torch.load(..., weights_only=True)`
без mmap) и измерению §5.3 (mmap → ACCESS_VIOLATION на Windows, не используется).
Комментарий — кандидат на чистку этапа 6; тело метода сохранено verbatim для эквивалентности.

**СЛЕДУЮЩИЙ (#28):** переключить потребителей (ziwago_assistant, run_ziwago_v19,
тренеры) на плоский `ZiwagoOrganism`, вынести helper-классы v19
(SourceGroundingAuthority/EmpiricalConfidenceCalibration/IndependentHypothesisSleep)
из модуля-обёртки, архивировать обёртки v18/v19/v20 + мёртвые движки, архивировать
мастер обратимо (зафиксировав SHA), удалить заглушки v20-dream/v25 (H3).

### 7.5. H3: удаление заглушек v20-dream/v25 (#29) — РЕАЛИЗОВАНО 2026-11-12

**H3 исполнен (переопределяет AGENTS.md §2 «обязано быть восстановлено»).**
Два тонких редиректа на `run_ziwago_v19.main()` удалены из рабочего дерева
(перенесены обратимо в `_archive/cleanslate_20261112/stubs/`):
* `train_ziwago_v20_ontogenetic_dream_trainer.py` (204 Б, `main(default_version=20)`);
* `run_ziwago_v25_evotrain_marathon.py` (174 Б, `main()`).

Полные версии сохранены в read-only оригинале `Ziwago18original`
(37 006 Б и 31 175 Б, проверено `ls`) — восстановление возможно, поэтому
«удаление» обратимо. Содержимое стабов зафиксировано до переноса (честная запись).

**Обновлены живые ссылки в tools** (единственные `.py`-потребители):
`tools/cleanup_project.py` (ENTRY_FILES + `verify()`-импорты) и
`tools/restore_build.py` (ENTRY_POINTS). Исторический `SNAPSHOT_FILES`
(снимок дерева 06.11) НЕ тронут — это запись, а не живой код. Прочие упоминания —
только в `.md`-документах (исторические отчёты), не влияют на импорт.

**Верификация:** `py_compile` обоих tools — OK; импорт живых точек входа
(`train_ziwago_v18_genesis_dream`, `ziwago_assistant`,
`train_ziwago_v16_10B_omniscient_engine`, `ziwago.organism_core`) под лимитером
— **ENTRY IMPORTS OK**, `ZiwagoOrganism.__bases__ == (object,)`, пик 1.46 ГБ.
Grep по всему дереву: ни один `.py` больше не импортирует стабы.

### 7.6. ИЗМЕРЕННАЯ реальность зависимостей #28 (корректировка плана) — 2026-11-12

**«Мёртвые движки» v10/v11/v11.5/v13 НЕ мёртвы — это живые зависимости v14.**
Прямое измерение импортов (grep):
* `train_ziwago_v14_pure_synaptic_dialogue.py:86` → `from train_ziwago_v11_5_predictive_readout_pfc import PredictiveCodingCorticalReadout`;
* `train_ziwago_v14_pure_synaptic_dialogue.py:87` → `from train_ziwago_v13_son_reflective_organism import PairedDialogueStreamer`;
* цепь v13:66 → v11.5:60 → v11 (module-level); v11:320 → v10 ЛЕНИВО (внутри функции,
  поэтому v10 — балласт на уровне модуля, но v11/v11.5/v13 грузятся живым путём).
* v18:63-72 → v15 (`execute_sws_synaptic_homeostasis_and_pruning`) + v16
  (`DeficitRoundRobinDomainBalancer`, `FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH`,
  `FiveBankCA3AttractorAtlas`, 2 thalamic-worker'а) + v14 (BPE-worker'ы).

**Вывод:** архивация v10/v11/v11.5/v13/v15/v16 НЕВОЗМОЖНА без предварительного
извлечения живых классов (`PredictiveCodingCorticalReadout`, `PairedDialogueStreamer`,
`DeficitRoundRobinDomainBalancer`, `FiveBankCA3AttractorAtlas`, curriculum-константа,
sleep-функция) в модули `ziwago/` — это работа типа 4c, а не простой архив.
План #28 «архивировать мёртвые движки» основывался на ложной предпосылке; исправлен.

**Безопасный первый шаг #28 (#30) — РЕАЛИЗОВАНО 2026-11-12:** helper-классы/функции v19
(`EmpiricalConfidenceCalibration`, `SourceGroundingAuthority`, `IndependentHypothesisSleep`,
`file_sha256`, `stable_hash`, `content_words`, `_document_qa_from_state`,
`validate_v19_envelope`, `validate_verified_envelope`, `accept_candidate`, `V19_SCHEMA`)
вынесены в постоянный дом `ziwago/organism/verified.py` (416 строк). Тела перенесены
**verbatim** (marker-based surgery по точным границам, без транскрипции). v19 ре-экспортирует
все 11 имён обратно; `organism_core` импортирует оттуда → плоский класс БОЛЬШЕ НЕ зависит
от модуля-обёртки v19. v19: 722 → 346 строк (осталась только обёртка `ZiwagoVerifiedOrganismV19`,
18 методов, база v18 — не тронута).

**Верификация #30:**
* AST: verified.py = 7 функций + 3 класса + `V19_SCHEMA`; v19 = только обёртка + `ROOT`
  + ре-экспорт; `organism_core` не импортирует v19-обёртку (`from …v19… import` → False).
* Импорт-чек под лимитером: **RE-EXPORT IDENTICAL — все 11 имён** (`v19.X is verified.X`),
  `__module__ == ziwago.organism.verified`, циклического импорта нет, пик 1.46 ГБ.
* Поведение: `test_v20_organism_composition` + `test_v19_lifecycle` +
  `test_v19_language_scoring` + `test_v19_evaluation` + `test_v19_causal_temporal`
  → **127 passed, 172 subtests passed** за 100.89 s, пик 2.93 ГБ (`limited_run --ram-pct 55 --cpu-pct 50`).
  Эквивалентность плоского класса башне сохранена; v19-обёртка + helper'ы работают через ре-экспорт.
* Сериализация безопасна: helper'ы экспортируются через `export_v19_state` (plain dict),
  НЕ через walker `__object__`-теги; ре-экспорт сохраняет резолв старых снимков.
* Неиспользуемые импорты в шапке v19 (datetime/json/document_qa и др.) оставлены —
  безвредны, чистка этапа 6 (v19 slated for archive).

**Требует решения владельца (производственное влияние):** схлопывание обёрток
v18/v19/v20 меняет живой путь `ziwago_assistant` и ломает tower-специфичные тесты
(test_v18_genesis_dream, test_v19_lifecycle/runtime/language_scoring — ссылаются на
имена обёрток и MRO-dispatch); архив мастера (1.32 ГБ, `MASTER_CKPT_REL_PATH`)
ломает `auto_load_checkpoint=True` по умолчанию и требует координации с Clean Slate
(H1: v20 обучается с нуля). Стратегия схлопывания (удалить v18/v19 vs только разорвать
наследование vs shim-алиасы) — выбор владельца.

**РЕШЕНО владельцем (2026-11-12, директива «изолируй продакшен-путь»):** продакшен —
плоская композиция v20; наследование v18→v19→v20 убрать, логику ассистента переписать
разрешено; tower-тесты, проверяющие старую иерархию, удалить или переписать под плоскую
архитектуру (модули + композиция). Порядок: Extract → Replace → Delete. Исполняется
как #31 (извлечение) → #32 (схлопывание) → #33 (архив) → #34 (верификация).

### 7.7. #31: извлечение живых зависимостей движков в ziwago/ — РЕАЛИЗОВАНО 2026-11-12

**Точная карта живых символов (grep импортов, не предположение):**
* v14:86 → `PredictiveCodingCorticalReadout` (из v11.5); v14:87 → `PairedDialogueStreamer` (из v13).
* organism_core:53 → `FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH` (из v16) — **единственный**
  v15/v16-символ плоского организма.
* v18:63-72 → v15 `execute_sws_synaptic_homeostasis_and_pruning` + v16
  `DeficitRoundRobinDomainBalancer`, `FiveBankCA3AttractorAtlas`, 2 thalamic-worker'а, curriculum.
* v19:98 (лениво в `sleep_cycle`) → v15 `execute_sws_synaptic_homeostasis_and_pruning`.

**Измеренный факт (скорректирован после решения владельца о флаге `integrated`, §7.8):**
плоский v20-режим (`organism_core._on_sleep_phase`, :446) делает SWS **inline**
(`w_dend_out.mul_(0.992)` + `core.execute_sws_ca3_apoptotic_pole_sanitization()`
+ `shield.consolidate_into_slow_reservoir()` + `_rem_dream()`), а НЕ через v15-функцию.
v20-тело `sleep_cycle` заместило v19-тело при слиянии (most-derived wins). **ОДНАКО**
владелец выбрал флаг `integrated` (§7.8): плоский класс несёт ОБА режима, и
v19-режим (`integrated=False`, `_sleep_cycle_legacy`) вызывает именно v15-функцию
`execute_sws_synaptic_homeostasis_and_pruning`. Поэтому она стала живой зависимостью
продакшена и **извлечена** в `ziwago/organism/sleep.py` (47 строк, verbatim v15:341-371,
self-contained: только torch). v15 ре-экспортирует её обратно; v19-обёртка (ленивый
импорт :98) и плоский `_sleep_cycle_legacy` резолвятся в один объект.
`DeficitRoundRobinDomainBalancer`, `FiveBankCA3AttractorAtlas`, thalamic-worker'ы (v16)
остаются зависимостями только обёрток v18 (не продакшен-пути) — уходят в архив вместе
с обёртками, если те будут архивированы (#33, см. ограничение ниже).

**Extract (verbatim, marker-based surgery, без транскрипции):**
* `ziwago/cognition/language/predictive_readout.py` (188 строк) ← `PredictiveCodingCorticalReadout`
  из v11.5:157-327 (171 строка класса). Self-contained: torch + typing, без движка v11.5.
* `ziwago/app/corpora.py` (138 строк) ← `StimulusResponsePair` + `PairedDialogueStreamer`
  из v13:69-183 (115 строк). Зависит только от v9 BPE (`_GPT2_PAT`, `_bpe_encode_piece`,
  `_init_bpe_worker`) — фундаментальный токенизатор, остаётся. Без torch.
* `ziwago/app/curriculum.py` (65 строк) ← `FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH`
  из v16:76-129 (54 строки, чистый дата-литерал, только typing).

**Replace (импорты переведены на чистые модули):**
* v14:86-87 → `ziwago.cognition.language.predictive_readout` / `ziwago.app.corpora`;
* `tests/test_v19_language_scoring.py:17` → `ziwago.cognition.language.predictive_readout`;
* `organism_core:53` → `ziwago.app.curriculum`;
* `tests/test_v16_roundtrip.py:44` → `ziwago.app.curriculum`.
* v11.5/v13/v16 ре-экспортируют извлечённые имена обратно (обратная совместимость импортов
  и старых снимков). v11.5: 762→596, v13: 730→620, v16: 996→943 строк.

**Верификация #31 (прямые измерения под лимитером `--ram-pct 55 --cpu-pct 50`):**
* Re-export identity: `v14.PredictiveCodingCorticalReadout is new_pc.…`, `v14.PairedDialogueStreamer
  is new_corp.…`, `v115./v13.` ре-экспорт identical, `organism_core./v16.` curriculum identical.
* **Разрезание живого пути:** чистый импорт `ziwago.organism_core` НЕ тянет ни один из
  v11.5/v13/v15/v16 (`sys.modules` → все False). Продакшен-организм полностью отвязан от башни движков.
* `probe_import_graph`: CLEAN (nw8=0, nw9=0, ballast=[]), пик 1.46 ГБ.
* Поведение: `test_v19_spiking_language` + `test_v20_integration` + `test_v20_organism_composition`
  + `test_v19_language_scoring` → **107 passed, 125 subtests passed** за 108.46 s, пик 3.90 ГБ.
  `organism_core.ZiwagoOrganism.__bases__ == (object,)`, ассистент импортируется.
* Остаточное ребро v13:66 → v11.5 (`ZiwagoPredictiveEntityV115Engine`) — внутреннее для мёртвой
  башни, уходит в архив вместе с v13/v11.5 (#33); продакшен его не касается.

### 7.8. #32: схлопывание башни — флаг `integrated` в плоском классе — РЕАЛИЗОВАНО 2026-11-12

**Решение владельца (AskUserQuestion, 2026-11-12):** «Флаг `integrated` в плоском классе».
`ZiwagoOrganism(integrated=False)` = v19-режим без физиологии; `integrated=True` = v20.
Ассистент маппит `runtime_version`→флаг; `run_ziwago_v19.py` и tools **без изменений**.
Наследование v18→v19→v20 убрано из продакшен-пути: ОБА режима собираются композицией
одного плоского класса (`bases == (object,)`). Минимальный радиус поражения.

**Правки `ziwago/organism_core.py` (8):**
1. Импорт `from ziwago.organism.sleep import execute_sws_synaptic_homeostasis_and_pruning`.
2. Сигнатура `__init__`: `integrated: bool = True`; `self._integrated = bool(integrated)`.
3. Страж загрузки v20-чекпоинта: `if not self._integrated:` (вместо `hasattr(_load_v20_payload)`)
   — v19-режим отвергает v20-чекпоинт, как v19-обёртка.
4. Тело v20-физиологии под `if self._integrated:` (при False версия остаётся "19.0",
   `enable_integrated_organism` не вызывается → `core.life` не создаётся).
5. `sleep_cycle`: диспетчер `if not self._integrated: return self._sleep_cycle_legacy(...)`.
6. `_sleep_cycle_legacy` — verbatim v19-тело (SWS из чистого модуля, CA3-ремонт, перенос
   ядра 0.08, резервуар, `ado*=0.35`, `is_awake=True`, REM).
7. `save_candidate`: диспетчер `if not self._integrated: return self._save_candidate_legacy(...)`.
8. `_save_candidate_legacy` — verbatim v19-тело (префикс `.v19-`, `V19_SCHEMA`,
   `validate_v19_envelope`, восстановление `loaded_checkpoint_path` в finally).

**Правки `ziwago_assistant.py` (2):** импорт `from ziwago.organism_core import ROOT, ZiwagoOrganism`;
сборка `ZiwagoOrganism(..., integrated=(runtime_version == 20))` вместо выбора
`runtime_type` v19/v20. Параметр `runtime_version` сохранён (API-совместимость с
`run_ziwago_v19.py:136-138` и tools). `self.organism = self.genesis.core_organism` — ветка
`chat` по `hasattr(self.organism, "life")` переключается флагом корректно.

**Верификация #32 (прямые измерения под лимитером `--ram-pct 55 --cpu-pct 50`):**
* `test_v20_organism_composition` (6 v20 + 4 новых v19) → **10 passed** за 96.10 s, пик 4.30 ГБ.
  Новые v19-тесты: флаг/версия/отсутствие `core.life`; эквивалентность состояния конструктора
  (genesis+v19+runtime) башне v19; эквивалентность `sleep_cycle`; `save_candidate` →
  v19-конверт (`V19_SCHEMA`, `validate_v19_envelope`, восстановление пути).
* Прибор сборки ассистента: `rv=19` → `ZiwagoOrganism integrated=False version=19.0
  core_has_life=False bases=(object,)`; `rv=20` → `integrated=True version=20.0
  core_has_life=True bases=(object,)`. Оба режима — один плоский класс, без наследования.
* `test_v19_runtime` + `test_v19_language_scoring` → **286 passed, 98 subtests passed**,
  9 failed за 82.00 s, пик 3.22 ГБ. **9 падений — ПРЕ-СУЩЕСТВУЮЩИЕ дефекты, не регресс #32:**
  5 × `IndexError pos_phase_vecs[16]` (v14:2336 `range(40)` > `max_pos=16` у tiny_core) +
  4 × Windows-path `AssertionError` (`hasattr(snapshot,"read")` / сравнение `WindowsPath`).
  Доказательство: **0 кадров `organism_core.py`** во всех трассировках; падающие тесты строят
  v19-обёртку напрямую (`compact_runtime`, test_v19_runtime.py:372 `ZiwagoVerifiedOrganismV19.__new__`)
  либо подменяют ассистент целиком (:532-535 `SimpleNamespace` + `monkeypatch … lambda`).
  Мой плоский класс и репоинт ассистента не входят в путь исполнения этих тестов.
* Тест сна v19-режима намеренно без обучения: `compact_runtime`/tiny_core имеют `max_pos=16`,
  а CA3-санитизация v14 циклует `range(40)` → `IndexError` при непустом `col_to_tok`. Дефект
  owner-decision-only, не чинится; legacy-путь проверен на пустом `col_to_tok` (CA3 ранний возврат).

**Ограничение #33 (измерено, влияет на архив обёрток):** tools, которые владелец оставил
без изменений, всё ещё зависят от обёрток: `tools/probe_v19_efficiency_budget.py:76-77`
строит `ZiwagoVerifiedOrganismV19` напрямую; `test_v20_organism_composition` импортирует
`ZiwagoOrganismV20`/`ZiwagoVerifiedOrganismV19` для эталонной эквивалентности. Поэтому
обёртки v18/v19/v20 и импортируемые ими движки (v15 SWS-ре-экспорт, v16 curriculum-ре-экспорт)
НЕ могут быть архивированы, пока tools/тесты не мигрированы. Продакшен-путь (ассистент →
плоский класс) уже схлопнут; башня остаётся только как эталон для неизменённых tools/тестов.

### 7.9. #33 (часть 1): архив мёртвой цепи движков — РЕАЛИЗОВАНО 2026-11-12

**Измеренная карта импортов (grep по всем `*.py`, не предположение):**
* v14:84-85 → ТОЛЬКО v9 BPE (`_GPT2_PAT`, `_bpe_encode_piece`, `_init_bpe_worker`).
  v11.5/v13/v15/v16 больше не импортируются v14 (репоинт #31).
* Мёртвая цепь (единственные рёбра — внутри себя): v13:66 → v11.5:60 → v11:320 → v10 (лениво).
  v11.5:54,59 → v9 (остаётся). Вне цепи потребителей НЕТ (ни продакшен, ни обёртки, ни tools, ни тесты).
* Обёртки: v18:63 → v15, v18:66 → v16, v19:98 → v15 (лениво), v16:68 → v15, v19:25 → v18,
  v20:13 → v19. Потребители обёрток: `run_ziwago_v19.py:16` (helper'ы),
  `tools/probe_v19_efficiency_budget.py:76`, корневые пробы (`_liveness`, `_beam_critical`,
  `_verify_e4`, `_three_facts`), `tools/diag_v19_decoder_credit.py:23`,
  `tools/probe_v19_assistant_recall.py:19`, `tools/probe_v19_state_fingerprint_control.py:17`,
  тесты (test_v18_genesis_dream, test_v19_lifecycle/runtime/language_scoring, test_v20_organism_composition).

**Прибор отвязки (прямое измерение, `limited_run --ram-pct 55 --cpu-pct 50`, пик 1.46 ГБ):**
импорт `ziwago.organism_core` + `train_ziwago_v19_verified_runtime` + `train_ziwago_v20_runtime`
→ `sys.modules`: v10/v11/v11.5/v13 = **все False**, v9 = True. Мёртвая цепь не входит ни в
продакшен-путь, ни в цепь неизменённых tools/обёрток.

**Архивация (обратимое перемещение, НЕ удаление):** v10/v11/v11.5/v13 →
`_archive/cleanslate_20261112/dead_engines/` (+ README с SHA-256 и причиной). Все 4 файла
присутствуют в read-only оригинале `Ziwago18original` (восстановимы, датированы 6 нояб).
SHA-256 рабочих версий зафиксированы в README архива. v11.5/v13 меньше оригинала, т.к. #31
извлёк их живые символы в чистые модули и оставил ре-экспорт (потребителей ре-экспорта нет).

**Верификация после архивации:**
* Импорт-прибор: `IMPORTS_OK flat+v19+v20+assistant+trainer`; мёртвая цепь не загружена;
  `ZiwagoOrganism.__bases__ == (object,)`.
* `test_v20_organism_composition` → **10 passed** за 106.38 s, пик 4.30 ГБ (оба режима + обёртки целы).
* `tools/probe_import_graph.py` → **CLEAN** (nw8=0, nw9=0, ballast=[], 1223 модуля), пик 1.46 ГБ.
  Балласт `train_ziwago_v10` отрезан и архивирован; живой граф v14+integrated чист.

**#33 часть 2 — РАЗРЕШЕНО Вариантом А (§7.10), архив выполнен 2026-11-12.** Исходный
конфликт (архив мастера ломает неизменённый тренер/tools) снят решением владельца
#35: тренер/tools мигрированы на плоский класс без мастера (#36), музейные
tools/пробы/тесты архивированы (#38). Ниже — исходная формулировка блокера для истории:
* Обёртки потребляются неизменёнными tools/пробами/тестами (список выше) → остаются.
* v15/v16 импортируются обёртками → остаются.
* **Конфликт мастера (H1):** `run_ziwago_v19.py:134-135` хеширует мастер
  (`master = ROOT/"Produce/checkpoints/ziwago_v18_1_genesis_omniscient_master.pt";
  master_hash = file_sha256(master)`) и проверяет `legacy_master_unchanged`; ряд tools
  (`golden_delta`, `diag_ca3_pole_scores`, `check_v20_generation`, `probe_v19_*`) грузят мастер.
  Архив мастера (H1 «старый мастер — в архив», Clean Slate v20 с нуля) сломает неизменённый
  тренер/tools. **Разрешение:** либо мигрировать тренер/tools на плоский класс без мастера
  (тогда архив мастера безопасен), либо оставить мастер на месте. Решение владельца.

---

### 7.10. #35 / #36 / #38: Вариант А + Clean Slate — архив мастера, башни и музейных tools — РЕАЛИЗОВАНО 2026-11-12

**Решение владельца (#35, Вариант А + Clean Slate):** «Мы не сворачиваем с пути
очистки архитектуры ради сохранения старых диагностических скриптов. Мастер-чекпоинт
v18 и его обёртки — легаси. Мы переходим на v20… Архивируй мастер и обёртки… Если
переписать сложно и они больше не нужны для v20 — отправь сами tools/ в архив…
Сохрани `legacy_master_unchanged`… просто замени проверку на True или удали этот тест…
Продакшен не трогай… Если сломается какой-то старый probe_v19, не трать на него время.
Мы строим v20 и v3.0, а не поддерживаем музейные экспонаты v18».

**Clean Slate в коде (#36, продакшен):**
* `ziwago/organism_core.py`: убран fallback на мастер — `auto_load_checkpoint and
  checkpoint_path` (нет пути → свежий организм v20 с нуля; архивный мастер не
  авто-грузится). Save-guards по строке пути (`MASTER_CKPT_REL_PATH`, `"master" in
  name`) оставлены — безвредны после архива.
* `run_ziwago_v19.py` (тренер v20): репоинт helper'ов на `ziwago.organism_core.ROOT`
  + `ziwago.organism.verified` (accept_candidate/file_sha256/stable_hash); убран
  baseline мастера и его SHA-проверки; нет `--checkpoint` → свежий организм
  (`auto_load_checkpoint=False`). `legacy_master_unchanged: True` — константа
  (санкционировано владельцем: неизменность старого мастера больше не проверяется).

**Правило замыкания архива:** файл архивируется, если зависит от музейной обёртки
(v15/16/18/19/20), ИЛИ грузит мастер, ИЛИ импортирует другой музейный файл.
Зависящие только от продакшена (v9-токенизатор, v14-движок, пакет `ziwago/`, плоский
ассистент) — остаются. **Ключевая поправка:** `train_ziwago_v9_gpt2vocab.py` — НЕ
обёртка, а BPE-токенизатор (потребители v14:84-85 и `ziwago/app/corpora.py:19-20`) → остаётся.

**Перемещено (обратимо, НЕ удаление; SHA-256 + размеры в `_archive/cleanslate_20261112/manifest.json`, 40 записей):**
* `wrappers/` (5): v15, v16, v18, v19, v20.
* `root_probes/` (5): `_beam_critical`, `_liveness`, `_three_facts`, `_verify_e4` (импорт v19), `verify_v18_2` (грузит MASTER_CKPT).
* `museum_tools/` (19): golden_delta, ensure_master, move_storage_to_d, diag_v19_decoder_credit, probe_v19_compose, probe_v19_assistant_recall, probe_v19_reproducibility, probe_v19_document_qa, probe_v19_state_fingerprint_control, probe_v19_efficiency_budget, train_v19_document_ranker, train_v19_apical_replay, _bench_fingerprint, prepare_v19_language_curriculum_v3/v4/v5, restore_build, cleanup_project, preflight.
* `tower_tests/` (10): test_v18_genesis_dream, test_v19_runtime, test_v19_lifecycle, test_v19_language_scoring, test_v19_compose_probe, test_v19_document_ranker, test_v19_evaluation, test_v19_cortex_roundtrip, test_v16_roundtrip, test_v18_garbage_regression.
* **Мастер → D:** `D:/Ziwago18-storage/Archive/cleanslate_20261112/master/ziwago_v18_1_genesis_omniscient_master.pt` (бюджет хранения: тяжёлое на D:). Размер **1 322 796 291 байт = 1.32 ГБ** (прямое измерение; «4.3 ГБ» из заметок — это v19-*кандидаты*, оставленные в `Produce/checkpoints/`). SHA-256 `cb97c8691869af0655b144bf56e904bfb6cae78d8f8eb2213b282441d76592e0` совпадает с AGENTS.md §6 и **пересчитан после переноса** (байты сохранены; rename внутри D:, `Produce` — симлинк).

**Оставлено (продакшен + чистые модули):** корень (ассистент, тренер v20, v14, v9, `_rank`, `verify_attractor_dominance`, `verify_sws_pruning_criterion`, `verify_v18_2_new`); пакет `ziwago/` без изменений; `tools/` (18 продакшен-утилит, включая limited_run, probe_import_graph, check_v20_generation, diag_v20_load); `tests/` (17 модулей / 434 теста, включая поставщик `tiny_core` test_v19_spiking_language и плоский test_v20_organism_composition).

**Верификация после архива (прямые измерения, `limited_run --ram-pct 55 --cpu-pct 50`):**
* Импорт продакшена (v14 + integrated + assistant + organism_core + trainer): `WRAPPERS_LOADED []`, `BALLAST []`, `FLAT_BASES (object,)`, `ASSISTANT_USES ziwago.organism_core.ZiwagoOrganism`, `MASTER_EXISTS_AT_PROD_PATH False` → **CLEAN**, пик 1.46 ГБ.
* `pytest tests/ --collect-only` → **434 теста, 0 ошибок коллекции**, пик 1.54 ГБ (все KEEP-модули импортируются чисто).
* `pytest tests/test_v20_organism_composition.py` → **10 passed за 50.33 с**, пик 2.93 ГБ (плоские режимы v20 и v19).

Подробности и восстановление: `_archive/cleanslate_20261112/README.md`. Архивные
museum tools/tests не запускаются на месте (кросс-импорты предполагают исходное
расположение) — сохранены для справки/отката. Оригинал `Ziwago18original` не тронут.

---

## 8. Решения владельца (приняты 2026-11-12)

- **П-1. Гиппокамп — MERGE всех версий** в идеально работающую архитектуру (слияние лучших решений либо деление на модули). Основа — идея `neuroweave10/hippocampusidea.md`: дуальная система System 1 (Fast Drafter) / System 2 (Slow Verifier) / Arbiter-GW, CLS McClelland. **РЕАЛИЗОВАНО 2026-11-12 — см. §8.1.**
- **П-2. Перенести из 8/9:** суррогатные градиенты (У-1), EWC/Fisher (У-3), время/ридеаут/синхронизация (У-5 Laplace elapsed, У-7 HemisphereSync, У-8 PredictiveHead). НЕ переносить: int8 (У-2), BG n-gram veto (У-4). **РЕАЛИЗОВАНО 2026-11-12 — см. §8.2** (У-5 — в §8.1).
- **П-3. Оживить nw10:** контроллер критичности (М-2), BG TD-критик (М-3), объёмная химия NO/H2S + 37 агентов (М-4). Тело/лимбика: **виртуальная интероцепция** вместо физической арены и слогового декодирования (М-6/М-8 — арена и syllabary НЕ нужны, убьют производительность текстовой модели); limbic BLA/CeA valence + виртуальная интероцепция. **М-2 РЕАЛИЗОВАНО 2026-11-12 — см. §8.3. М-3 РЕАЛИЗОВАНО 2026-11-12 — см. §8.4. М-4 РЕАЛИЗОВАНО 2026-11-12 — см. §8.5. М-6/М-8 (виртуальная интероцепция + лимбика) РЕАЛИЗОВАНО 2026-11-12 — см. §8.6. П-3 ЗАВЕРШЁН.**
- **П-4. v18 genesis generation — ОЖИВИТЬ** (θ-γ TTFS, closure verifier, valence wave).
- **П-5. Данные/атлас (Т-1…Т-8), М-5/М-7/М-9:** решаются на соответствующих этапах (реестр в §5).

### 8.1. П-1 реализация: merged dual-system гиппокамп (2026-11-12)

**Новый модуль:** `neuroweave10/memory/dual_hippocampus.py` (opt-in).
**Тесты:** `tests/test_dual_hippocampus.py` — 23 passed (3.04 s).
**Живой путь v19 не изменён**, мастер-чекпоинт не тронут (новые файлы, правок
существующих модулей нет; `OptInIsolationTests` фиксирует, что plain
`Hippocampus` остаётся собой).

Дизайн — композиция, один механизм = один владелец (целевая L1→L2):

| Слой дуальной системы | Владелец в merged-модуле | Источник механизма |
|---|---|---|
| Fast Drafter (System 1), рабочая память | `MergedLaplaceBank` + `CompositionalThetaGammaIndexer` | nw9 `LaplaceTimingBank` (elapsed-time, У-5) + nw10 `LaplaceBank` (slow_profile) → один класс; nw10 θ-γ indexer (оживлён) |
| Общие часы | `ThetaClock` (token/fast/baseline) | nw9 `ThetaOscillator` — фикс «1 реколл на 143 токена» для текста |
| Slow Verifier (System 2), эпизодическая | `DualHippocampus.episodic` = живой nw10 `Hippocampus` | трисинаптический EC→DG→CA3→CA1 + нейрогенез + temporal STDP + concept cells + щит Фузи + SWR/REM (без дублирования 1123 строк) |
| Распределённая память + граф | `episodic.vsa_dht` (`SBCVSAShardedHippocampus`) | nw10 VSA-DHT 4 шарда (38 400 эпизодов) + HippoRAG PPR |
| Arbiter / GW | `NoveltyArbiter` (`ArbiterCfg`) | CA1 mismatch → DA/NE; CA3 resonance → accept/regenerate/ask/unknown (пороги 0.7/0.3 из `hippocampusidea.md`) |

**Оживлённые мёртвые пути** (подтверждено grep до merge — в v19 только запись):
* `SBCVSAShardedHippocampus.retrieve_multiprobe` — распределённый реколл теперь
  читается в `verify()` (в v19 `store_from_tensors` писал, но не читал);
* `ppr_associative_recall` / `route_contextual_replay` — многошаговые цепочки
  концептов A→B→C (HippoRAG) теперь вызываются;
* `CompositionalThetaGammaIndexer.encode_and_bind` — θ-γ связывание ролей в
  рабочей памяти (вызывался только в мёртвом тренере v9);
* `Hippocampus.ca3_predict_next` — предсказание следующего шага траектории;
* `LaplaceTimingBank.estimate_elapsed_time` (У-5) — давность признака.

**Измеренный сквозной протокол** (`DualSystemProtocolTests`): до записи эпизода
`resonance=0.000 → decision=unknown` («я не знаю», анти-галлюцинация); после
записи `resonance=1.000 → decision=accept`, VSA-DHT возвращает эпизод `[0]`,
HippoRAG возвращает цепочку `['dragon','mountain']`.

**Миграция ВЫПОЛНЕНА 2026-11-12 (задача #22).** Ранний план целил в
`organism/genesis.py`, но genesis архивирован как мёртвый (§8.7) — подключение
туда было бы мёртвым проводом (AGENTS.md §7). Фактический целевой контейнер —
**живой `IntegratedOrganism`** (`neuroweave10/organism/integrated.py`), по тому же
opt-in контракту, что М-4/М-8 (volume/intero/limbic):

* **Флаг профиля v20:** `dual_hippocampus_enabled: bool = False` + knobs
  `hippo_n_ec=64`, `hippo_theta_mode="token"`, `hippo_theta_cycle_steps=8.0`,
  `hippo_novelty_da_gain=0.5` — все пишутся в телеметрию (нет «тихих режекторов»,
  §6). Default off → `hippo=None`, `da_phasic`/`ne` не меняются, **бит-идентично
  v19** (тест `test_zero_gain_is_bit_identical`: `torch.equal(w_fast)` при gain=0).
* **Такт:** `_update_hippocampus` (вызов в `tick` после `_update_limbic`, до
  сборки `ctx`) тикает `DualHippocampus` на таламо-корковом `relay` (EC-вход =
  текущее восприятие), `ach`/`ado` из химии, `learn=plastic`. Fast Drafter
  (Laplace + θ-γ), Slow Verifier (episodic core + VSA-DHT + HippoRAG) и Arbiter
  исполняются каждый такт.
* **Причинный актуатор (измерен):** новизна CA1 → VTA DA burst → `da_phasic`
  (3-й фактор пластичности, `synapse.py:502`, причинность доказана в М-8).
  Тест: `da_on = da_off + 0.5·burst` (places=6), `burst>0`; пластичность
  расходится (`mean|Δw_fast|>1e-9` при learn on).
* **Эпизодическая память оживает** через `record_credit` (живой хук
  `train_ziwago_v14_pure_synaptic_dialogue.py:3055`): `store_episode` пишет
  ec/dg/ca3 с тегом новизны → `resonance` растёт (0.000 → >0.5, decision
  `unknown`→`accept`), VSA-DHT `retrieve_multiprobe` читает эпизод обратно
  (`recalled_episode_ids` содержит 0). Это и есть «оживлённые мёртвые пути» §8.1.
* **Отрицательный результат (честно, НЕ оживлён):** NE-burst → `ne`. В свежем
  v20 `ne` инвариантен (тоническое насыщение реле + stochastic off нейрона) —
  мёртвый провод по §7, тот же результат, что в лимбике (М-8). Тест
  `test_ne_not_wired_negative_result` фиксирует инвариантность.
* **Владелец один:** `IntegratedOrganism.hippo` (композиция, не наследование);
  episodic core — тот же живой nw10 `Hippocampus`, не дублируется. Мастер-чекпоинт
  не тронут (новых весов в живом пути нет, модуль opt-in).

**Проверено (прибор, не предположение):** `tests/test_integrated_hippocampus.py`
— **17 passed**; регрессия opt-in (dual_hippocampus + intero/limbic + volume +
criticality) — **82 passed (7.53 s)**, пик 1.53 ГБ; probe живого графа — **CLEAN**
(nw8=0, nw9=0, ballast=[], 1216 модулей); живой путь —
`test_v19_spiking_language.py` **31 passed + 27 subtests**. Всё через
`tools/limited_run.py --ram-pct 55 --cpu-pct 50`.

**Осталось (не выполнено):** архивировать дубли-источники гиппокампа
`nw8/memory/hippocampus.py`, `nw9/cortex/hippocampus.py`, `nw9/memory/laplace.py`,
`nw10/memory/laplace.py` — nw8/nw9 уже в `_archive/` (§8.7); `nw10/memory/laplace.py`
ещё в живом дереве (проверить рёбра перед архивом). v14 CA3-полюса — это dialogue
identity state (`register_user_identity`/`record_turn_and_poles`), владелец L3
cognition/dialogue, не трисинаптический гиппокамп; вне scope этого merge.
Включение `dual_hippocampus_enabled=True` в боевом профиле v20 — после проверки
на обученных весах (машина memory-blocked, §5.3/§7).

### 8.2. П-2 реализация: инфраструктура обучения (2026-11-12)

**Новый пакет:** `neuroweave10/learning/` (opt-in, самодостаточный, torch-only).
**Тесты:** `tests/test_learning_infrastructure.py` — 25 passed; вместе с
гиппокампом 48 passed (3.06 s). Живой путь v19 не изменён, мастер не тронут.

Консолидация утраченных техник nw8/nw9 в единого владельца (nw10/learning):

| Техника | Модуль-владелец | Источник | Контракт (проверен тестом) |
|---|---|---|---|
| У-1 surrogate | `learning/surrogate.py` | nw8 core/surrogate.py | forward=Хевисайд, backward=FastSigmoid 1/(1+k\|x\|)² / Arctan (a/2)/(1+(πax/2)²) — сверено с аналитикой |
| У-3 EWC/Fisher | `learning/ewc.py` | nw8 learning/ewc.py | штраф=λ·Σ F·(w−snapshot)², Fisher норм. на max, rebaseline после сна обнуляет |
| У-8 PredictiveHead | `learning/predictive.py` | nw8 microcircuit/column.py | Widrow-Hoff без autograd, precision Π=1/σ², сходимость на линейной задаче (err<1e-3) |
| У-7 phase | `learning/phase.py` | nw8 core/phase.py + PhaseCfg | bind/unbind ТОЧНО обратимы, cleanup восстанавливает символ, ёмкость WM архитектурная (алиасинг) |
| У-7 sync | `learning/sync.py` | nw9 cortex/sync.py | HemisphereSync: coherence∈[-1,1], frame.real [B,vsa_dim], occupancy |

**Конфликт У-1 разрешён:** `cortex/spiking_language.py:3` запрещает surrogate и
глобальный backprop в живом ридеауте. Порты — OPT-IN инфраструктура для будущих
градиентных контуров v20+, НЕ зависимость живого `SpikingLanguageLoop`
(`OptInIsolationTests` фиксирует: пакет learning не ссылается на nw8/nw9 и не
импортируется живым ридеаутом). У-8 (локальное правило, без autograd) совместим
с философией nw10 напрямую.

**НЕ перенесено (решение П-2):** int8 (У-2), BG n-gram veto (У-4 — N-граммы
запрещены владельцем). У-5 (Laplace elapsed-time) уже перенесён в merge
гиппокампа (§8.1, `MergedLaplaceBank`).

**Путь миграции:** подключить EWC к фазе сна v20 (rebaseline после
`sleep_cycle`), PredictiveHead — к колонкам за флагом профиля, HemisphereSync —
к билатеральной коре. После severing import-рёбер архивировать nw8/nw9
источники (класс B по §6).

### 8.3. П-3 / М-2 реализация: оживлённый гомеостаз критичности (2026-11-12)

**Точка смерти (измерено, не предположено):** в живом пути v19/v20
(`IntegratedOrganism` + v14 `moe_cortex`) критичность была ТОЛЬКО описательной —
`_measure_criticality` завершался полем `"controller_enabled": False`, и оценка
σ̂ ни на что не влияла. Активный `HomeostaticController.regulate`
(`homeostasis/controller.py`) жив лишь в `GenesisOrganism`
(`genesis.py:554` через `unified/pipeline.py:63`), но НЕ в генерации v19/v20.

**Решение — организменный таламический гейн, а не колоночный regulate().**
Колонки `moe_cortex` не несут `zcfg`/`w_pv_in`, которые требует `regulate()`,
поэтому прямой перенос колоночного контроллера невозможен без переписывания
коры. Вместо этого добавлен комплементарный актуатор другого масштаба:
гомеостаз силы таламического релея на входе коры от измеренного σ̂. Это не
дубликат `HomeostaticController` (тот правит пороги Θ/веса/iSTDP ВНУТРИ колонки),
а таламический гейн — биологически отдельный контур (ретикулярное таламическое
торможение / модуляция relay gain).

**Владелец:** `neuroweave10/organism/integrated.py` (`IntegratedOrganism`).

| Элемент | Поле/метод | Контракт |
|---|---|---|
| Флаг | `IntegrationConfig.criticality_control_enabled` | default `False` → v19/v20 bit-identical |
| Цель | `criticality_target=0.97` | субкритичный реверберирующий режим (Wilting-Priesemann) |
| Скорость/границы | `criticality_gain_rate=0.5`, `gain_min=0.5`, `gain_max=1.5` | валидация в `__post_init__`: 0<min≤1≤max, target∈(0,1.5] |
| Актуатор | `tick`: `relay = relay * self.criticality_gain` | причинно масштабирует реле ДО `moe_cortex.step` |
| Регулятор | `_update_criticality_gain` | `g_target=exp(rate·(target−σ̂))`, EMA 0.9/0.1, clamp; вызов каждые 32 тика после `_measure_criticality` |
| Телеметрия | `criticality_gain`, `criticality["controller_enabled"/"gain"]` | гейн и факт регуляции пишутся в отчёт (нет «тихих режекторов») |

**Знак регуляции:** σ̂ < target → кора недокритична → гейн растёт (усиливаем
реле); σ̂ > target → перекритична → гейн падает. Нет доступной оценки σ̂
(окно мало / нестационарно / плохой фит) → гейн не меняется, регуляция не
включается.

**Тесты:** `tests/test_criticality_control.py` — 14 passed (3.15 s):
валидация конфига; знак регуляции (sub/super/at-target); границы гейна при
экстремальных σ̂; причинность (гейн 1.4 → реле ×1.4 на живой коре,
`test_gain_scales_relay_causally`); opt-in (enabled+гейн 1.0 bit-identical to
off). Регрессия: `test_v20_integration.py` + `test_v20_cortical_clock.py` +
`test_v20_physiology.py` — 76 passed (8.62 s), default off не меняет поведение.

**Путь миграции:** включить `criticality_control_enabled=True` в профиле v20
после проверки на обученных весах (прибор живости П-5, AGENTS.md §7 — свежий
организм отвечает шаблоном отказа, мерить на нём бессмысленно). Колоночный
`HomeostaticController` остаётся владельцем внутриколоночного гомеостаза в
`GenesisOrganism`; два актуатора не пересекаются по масштабу.

### 8.4. П-3 / М-3 реализация: BG TD-критик + резонанс NoGo (2026-11-12)

**Измерено прибором живости ДО правки (AGENTS.md §7, не чтение кода):**

1. **TD-критик УЖЕ жив** в живом владельце v20 `ContextualActionGate`
   (`action/contextual.py`): `settle(decision_id, 1.0)` → `consumed=True`,
   `rpe=1.0`, `learned=True`, `w_v` изменился. Критик заморожен в момент
   коммита (`predicted_value`), RPE one-shot (повторный settle → `consumed=False`).
   **Дублировать TD-критик не нужно** — это противоречило бы «один владелец».
2. **`ca3_resonance` в legacy `BasalGangliaActionSelector.select_action` БЫЛ
   мёртв**: параметр принимался, но тело его игнорировало — resonance 0.0 и 1.0
   давали один и тот же `ACTION_DIRECT_SYNAPTIC_RESPONSE` (измерено).
3. **Отрицательный результат:** в v20 evidence-пути (`ziwago_assistant.py:182-183`)
   резонанс подключён к evidence, но **argmax-инвариантен** для выбора
   direct-vs-clarify при равномерных весах: evidence прямого ответа `0.6+0.4r`
   всегда больше evidence уточнения `0.5(1−r)` при всех r≥0 (0.1+0.9r>0).
   Измерено: resonance 0.0 и 1.0 → один и тот же action 6. Резонанс меняет
   маржу, но не argmax — аналогично finding'у `readout_gain` в AGENTS.md §7.

**Оживлённый мёртвый провод:** `ca3_resonance` → legacy BG selector. Добавлен
opt-in гиппокампальный NoGo-гейт (гиперпрямой путь STN): при низком резонансе
(`< unknown_resonance=0.3`) и отсутствии специфичного действия уверенный
`DIRECT_SYNAPTIC_RESPONSE` заменяется на `ACTIVE_INFERENCE_CLARIFY` («не знаю»).
Это даёт резонансу реальный argmax-эффект, которого нет в v20 evidence-формуле,
и совпадает с протоколом арбитра (§8.1, `NoveltyArbiter`: r<0.3 → unknown).

**Владелец:** `ziwago_assistant.py` (`BasalGangliaActionSelector`).

| Элемент | Поле/метод | Контракт |
|---|---|---|
| Флаг | `resonance_gate_enabled` (ctor), проброс `resonance_action_gate` в ассистент | default `False` → v19/v20 bit-identical |
| Порог | `unknown_resonance=0.3` | валидация [0,1]; строгое `<` (на пороге → direct) |
| Актуатор | `select_action`: NoGo только на default fallback | специфичные действия (teaching/episodic/cf/think/multi) не трогаются |
| Наблюдаемость | возвращаемый `action` отражает гейт | нет «тихого режектора»; `scores` — флаги, гейтом не меняются |

**Причинность в живом пути:** v19 non-life — `select_action` вызывается с реальным
резонансом (`ziwago_assistant.py:273-274`), гейт причинен для финального action.
v20 life — финальное решение принимает спайковый `ContextualActionGate`
(`:191`), резонанс входит через evidence (argmax-инвариантен, см. выше);
legacy-гейт на финальный action v20 не влияет (сознательно НЕ переопределяет
спайковое решение). Унификация резонанс-NoGo поверх обоих владельцев — отдельное
решение, не делается здесь, чтобы не подменять спайковый гейт символическим.

**Тесты:** `tests/test_bg_resonance_gate.py` — 9 passed (2.76 s): default off
игнорирует резонанс; низкий→clarify, высокий→direct; строгий порог; специфичные
действия не переопределяются; scores=флаги; валидация порога; TD-критик жив
(settle меняет `w_v`, RPE one-shot). Регрессия: `test_v19_language_scoring.py -k
document_qa` — 6 passed + 11 subtests (4.57 s), чат-путь не изменён.

**Дубликат к архиву (класс B по §6):** `neuroweave10/action/basal_ganglia.py`
(nw10 `BasalGanglia` D1/D2 DDM+TD+A2A) — мёртв в живом пути, его TD-критик
дублирует живой `ContextualActionGate`. Уникальные механизмы уже покрыты живым
владельцем: A2A-D2 гетеромер (ADO ослабляет D2) ≈ `ado_gain` (аденозин поднимает
порог = NoGo-сдвиг); STN-конфликт ≈ `conflict_gain`. Отдельного переноса не
требуется; архивировать после severing import-рёбер.

**Путь миграции:** включить `resonance_action_gate=True` для v19 после проверки
на обученных весах. Для v20 — отдельное решение о резонанс-NoGo поверх
спайкового гейта (требует неоднородного по каналам гейна, чтобы резонанс менял
argmax, а не только маржу — см. П-1/П-2 AGENTS.md §7).

### 8.5. П-3 / М-4 реализация: объёмная химия NO/H2S + пул агентов (2026-11-12)

**Измерено прибором живости ДО правки (AGENTS.md §7):**

1. Живая колонка v20 УЖЕ содержит точки потребления эффекторов: `column.py`
   читает из `mod_ctx` `nmda_gain` (:280), `histamine` (:279), `vth_shift` (:282),
   `gaba_tonic` (:283), `rate_ceiling` (:286), `ahp_block` (:287), `bdnf` (:338)
   и `_no_ltp_field` (:325, идёт в `w_ff/w_fb/w_rec.learn`). `moe_cortex.py:1167`
   копирует ВСЕ скаляры `organism_context` в `col._mod_ctx`.
2. Но `IntegratedOrganism` их НЕ заполнял: ctx содержал только базовый bundle
   (ach/ne/da/no/ecb/...) → `nmda_gain` дефолт 1.0, `_no_ltp_field=None`. То есть
   пул агентов и объёмная передача были мёртвым проводом в v20 (живы только в
   `GenesisOrganism`, genesis.py:273-331,435).
3. `OrganismPhysiology` несёт лишь глобальные скаляры `s.no`/`s.ecb`
   (physiology.py:304-305), без пространственного NO-поля и без пула агентов.

**Оживление через существующие точки потребления (без правки коры):**
opt-in `volume_chemistry_enabled` в `IntegrationConfig`. При включении
`IntegratedOrganism` создаёт `ExtendedChemistry` + `VolumeTransmission(n_zones·
n_neurons)` и каждый такт:
* `_update_extended_chemistry`: обновляет пул от измеренных сигналов прошлого
  такта (activity=previous_ca, stress/threat=previous_error, arousal=0.5·ne+0.5·ach,
  social=drive, damage=max(0,error−0.3)) и возвращает эффекторы в ctx
  (`nmda_gain`, `histamine`, `vth_shift`, `gaba_tonic`, `ahp_block`, `rate_ceiling`,
  `bdnf`) + `p_release`, умноженный на `p_release_scale()` (NPY/Dyn/AEA/Gal);
* `_distribute_volume_fields`:.sources — ТОЛЬКО активные колонки (неактивные не
  шагают в `moe_cortex.step`, их Ca/spikes устарели), `volume.step` → NO/2-AG/H2S
  поля, `zone_fields` раздаёт `col._no_ltp_field` (гетеро-LTP) и `col._vol_eject`.

**Владелец:** `neuroweave10/organism/integrated.py` (opt-in); сами модули
`chemistry/agents.py` + `chemistry/volume.py` — без дублирования, те же что в
GenesisOrganism.

**Причинность (измерено на живой коре):** OFF→ON меняет дендритные токи:
`i_ff` 13.04→10.50 (p_rel 1.0→0.35 масштабирует AMPA, neuron.py:148), `i_fb`
0.1007→0.0855 (nmda_gain 1.0→0.111 масштабирует NMDA, neuron.py:163-164),
`v_a` −67.74→−67.68. Реле выше коры не меняется (`last_relay_norm` идентичен).

**Отрицательный результат (честно, AGENTS.md §7):** спайковый выход и Ca-след в
свежей tiny_core НЕ меняются (previous_ca 1.982569 OFF==ON). Причина: Ca
управляется спайками (`ca = ca·(1−dt/τ_ca)+s`, neuron.py:279), а спайки в этом
режиме задаются навязанным соматическим драйвом (i_ext=1.5) и не переворачиваются
от дендритной модуляции. Причинность доказана на токах, не на Ca; на обученных
весах с осмысленной обратной связью эффект дойдёт до спайков. Мерять на обученных
весах (П-5), не на свежем организме.

**Тесты:** `tests/test_volume_chemistry.py` — 9 passed (3.83 s): default off
ничего не создаёт/ctx не меняется; эффекторы заполняют mod_ctx; дендритные токи
меняются причинно; p_release масштабируется; реле выше коры идентично; NO LTP
поле формы (n_neurons,) в [0,1]; телеметрия exposes 22 концентрации агентов +
no/h2s/ecb. Общая регрессия v20+М-2+М-3+М-4 — 108 passed (9.08 s).

**Путь миграции:** включить `volume_chemistry_enabled=True` в профиле v20 после
проверки на обученных весах. H2S→`set_metaplastic` (synapse.py:182) — путь
GenesisOrganism; в v20 колонка читает H2S только через `nmda_gain` (D-серин в
пуле), отдельной точки потребления H2S в v20-шаге колонки нет — не добавлялась,
чтобы не править кору. Пространственный H2S→метапластичность v20 — отдельное
решение.

### 8.6. П-3 / М-6+М-8 реализация: виртуальная интероцепция + лимбика BLA/CeA (2026-11-12)

**Адаптация П-3:** БЕЗ физической арены (`embodied/arena.py`) и слогового
декодирования (`motor/syllabary_decoder.py`) — решение владельца: они убьют
производительность текстовой модели. Оживлены только два мёртвых в v20 модуля
самомодели и эмоциональной оценки: `core/interoception.py` (Insula–ACC) и
`emotion/limbic.py` (Гипоталамус + BLA/CeA). Оба живы лишь в `GenesisOrganism`
(genesis.py:90-105,373-383,455-462); в `IntegratedOrganism` отсутствовали.

**Измерено прибором живости ДО правки (AGENTS.md §7, не чтение кода):**

1. `IntegratedOrganism` не нёс ни `intero`, ни `limbic` (hasattr → False) — мёртвый
   провод в v20.
2. **Причинные актуаторы интероцепции** (измерено на живой коре):
   * `col.pyr.met_gain` (neuron.py:180, `g_met`) — метаболический тормоз:
     met_gain 1.0→2.0 меняет `v_s` −40.1689→−40.8313 (гиперполяризация);
   * `chem.process_s` (physiology.py:179, авторитетный скаляр давления сна) —
     process_s 0.95 закрывает гейт пластичности `f_mod` 0.65→0.0.
3. **Причинный актуатор лимбики** — `da_phasic` (3-й фактор, synapse.py:502,
   external clock v20): знаковый, da 0.0→dW 1.06e-5, +1.0→1.92e-5, −0.5→6.3e-6.
   Колонка читает именно `da_phasic` (column.py:171,329-334), НЕ тонический `da`.
4. **Отрицательный результат (НЕ оживлён):** arousal→NE. В свежем v20-режиме `ne`
   инвариантен: таламический relay norm 2.40000 при ne 0.2 и 1.2 (тоническое
   насыщение), нейрон `v_s` −49.0090 идентичен при ne 0.2/1.2 (stochastic off,
   noise=0). Это мёртвый провод по AGENTS.md §7 — лимбика НЕ модулирует `ne`.
   Genesis-провод arousal→s.ne (genesis.py:381) в v20 не перенесён сознательно.

**Оживление через существующие точки потребления (без правки коры):** два
независимых opt-in флага в `IntegrationConfig`.

| Элемент | Поле/метод | Контракт |
|---|---|---|
| Флаг интероцепции | `interoception_enabled` | default `False` → v19/v20 bit-identical |
| Флаг лимбики | `limbic_enabled` | default `False`; независим от интероцепции |
| Интеро-актуатор 1 | `_update_interoception`: `met_gain` EMA | target=1+`intero_met_gain_scale`·drive, EMA 0.95/0.05, clamp≤2.0 |
| Интеро-актуатор 2 | `process_s += intero_sleep_gain·drive` | только при drive>`intero_allostatic_threshold`=0.35 |
| Лимбика-актуатор | `_update_limbic`: `bundle["da_phasic"] += limbic_da_valence_gain·valence` | знаковый 3-й фактор; gain=0 → bit-identical |
| Наблюдаемость | `telemetry()["interoception"/"limbic"]` | self_state/intero_error/allostatic_drive; valence/arousal/d_energy/d_safety/d_novelty |

**Владелец:** `neuroweave10/organism/integrated.py` (opt-in); сами модули
`core/interoception.py` + `emotion/limbic.py` — без дублирования, те же что в
GenesisOrganism.

**Recoverable met_gain (инженерное решение, не копия Genesis):** GenesisOrganism
растит met_gain ratchet'ом (`+=0.02·drive`, genesis.py:462) и полагается на
колоночный `HomeostaticController` для decay (controller.py:106, ×0.995). В v20
колоночный контроллер НЕ активен (колонки moe_cortex не несут zcfg/w_pv_in),
поэтому чистый ratchet копился бы без границ. Заменён на recoverable EMA к
`1+scale·drive`: при drive→0 возвращается к 1.0. Измерено: preset 1.5 + полный
ATP → 1.479 за такт (decay); 40 тактов дефицита → 1.266 (bounded ≤2.0).

**Причинность (измерено на живой коре, tiny_core 2×32):**
* Интероцепция (дефицит ATP, energy=0.40): OFF met_gain=[1.0,1.0],
  process_s=0.00003; ON met_gain=[1.0379,1.0379], process_s=0.00895,
  allostatic_drive=0.372. Оба актуатора причинны.
* Лимбика (пластичность ON, training_active=True): OFF da_phasic=0.0;
  ON da_phasic=−0.1445 (valence=−0.289, gain 0.5), `w_fast` расходится
  dW=4.25e-6. Валентность причинно меняет 3-й фактор пластичности.

**Отрицательный результат (честно, AGENTS.md §7):** в свежем tiny_core при полном
ATP аллостатический драйв мал (энергия не истощена), поэтому интероцепция почти
не меняет met_gain — эффект проявляется при метаболическом дефиците (измерено:
energy 0.40 → drive 0.372). Лимбика причинна только при открытом гейте
пластичности (da_phasic — 3-й фактор, не тонический модулятор); в режиме
инференса без обучения валентность пишется в телеметрию, но не меняет веса.
Мерять на обученных весах (П-5), не на свежем организме.

**Тесты:** `tests/test_interoception_limbic.py` — 19 passed (5.24 s): валидация
конфига; opt-in (default off ничего не создаёт, met_gain≡1.0 даже при дефиците,
флаги независимы); интероцепция (met_gain растёт причинно, recoverable, bounded,
давление сна растёт, телеметрия self_state); лимбика (da_phasic=valence·gain,
gain=0 bit-identical, пластичность расходится причинно, valence знаковый/ bounded,
arousal→NE НЕ подключён — негативный результат зафиксирован тестом, телеметрия
эмоций). Общая регрессия v20+М-2+М-3+М-4+М-6/М-8 — **127 passed (11.77 s)**.
v19-путь не изменён: `test_v19_spiking_language.py` — 31 passed + 27 subtests;
`test_v19_language_scoring.py -k document_qa` — 6 passed + 11 subtests.

**Путь миграции:** включить `interoception_enabled=True` + `limbic_enabled=True`
в профиле v20 после проверки на обученных весах. GW-блендинг Insula-вектора
(genesis.py:482-487) и hippocampal salience от emo_priority (genesis.py:398-405)
в v20 НЕ перенесены: GW-кандидат и дуальный гиппокамп — отдельные решения
(миграция DualHippocampus за флагом v20 выполнена 2026-11-12, §8.1). Физическая
арена и syllabary (М-6 motor) остаются DEAD по решению П-3 — не оживлять.

### 8.7. Архивация мёртвого genesis-стека (класс B, задача #21, 2026-11-12)

После разрезания рёбер (§1.1) мёртвый genesis-стек перенесён в обратимый
`_archive/` (зеркалит структуру проекта; НЕ удаление). Каждый файл сверен с
оригиналом `Ziwago18original` (только чтение) — все PRESENT.

| Архивировано | .py | Размер | В оригинале |
|---|---|---|---|
| `neuroweave8/` (целиком) | 28 | 345K | ✓ |
| `neuroweave9/` (целиком) | 31 | 303K | ✓ |
| `neuroweave10/unified/` (`__init__`+`pipeline`) | 2 | 184K | ✓ |
| `neuroweave10/organism/genesis.py` (GenesisOrganism) | 1 | — | ✓ |
| `neuroweave10/organism/system.py` (GenesisSystem) | 1 | — | ✓ |
| `neuroweave10/main.py` (демо-CLI GenesisSystem) | 1 | — | ✓ |

**Сопутствующие правки живого дерева:**
- `neuroweave10/__init__.py` — удалены ленивые фабрики `GenesisOrganism`/
  `GenesisSystem` (их реализации архивированы); `GenesisConfig` сохранён
  (оборачивает живой `config`). Вызывающих фабрики в живом пути/тестах нет (grep).
- `train_ziwago_v9_gpt2vocab.py:360` — комментарий: мёртвый тренер
  `ZiwagoGPT2VocabTrainer` сохранён как определение (v11 импортирует его на уровне
  модуля), но инстанцирование упрётся в архивный `unified.pipeline`. Живой путь
  v14 берёт из v9 только BPE.
- `organism/` выжившие: `__init__.py` (пуст), `integrated.py`, `physiology.py` (LIVE).

**НЕ архивировано (осознанно):** тренеры v10/v11(engine)/v13/v15/v16 — ещё имеют
живые import-рёбра (v11.5:60 импортирует класс v11-engine; v14:87 → v13 streamer),
по §6 B требуют переноса live-фрагментов сначала. `action/basal_ganglia.py` и
`consciousness/metrics` (мёртвые дубли, §8.4/§3) — отдельные кандидаты, вне этого
кластера; живых импортёров нет, но архивация отложена.

**Верификация (финальное архивное состояние):**
- Collection всего `tests/`: **812 tests collected, 0 import-errors** — ни один
  тест не импортирует архивные модули (`D:/ziwago_collect.log`).
- Живой граф (`tools/probe_import_graph.py`): nw8=**0**, nw9=**0**, ballast=**[]**,
  CLEAN, 1217 модулей (`D:/ziwago_final_probe.log`).
- Регрессия живого пути+механизмов (12 suites): **352 passed, 125 subtests**
  (24.12 s, пик 1.79 ГБ) (`D:/ziwago_final_regression.log`).

**Полный прогон `tests/` заблокирован памятью машины (НЕ дефект архивации):**
812 тестов кумулятивно дали пик 6.77 ГБ > лимит 6.73 ГБ → убит лимитером (§5.4);
78+ падений `DefaultCPUAllocator: not enough memory`. `test_v18_genesis_dream`
(12 errors) — `setUpClass` грузит чекпоинт → `load_state_dict` ACCESS_VIOLATION
0xC0000005 (§5.3, пик 6.69 ГБ).

**13 пред-существующих падений ДОКАЗАНЫ baseline-сравнением (не архивация):**
кластер временно возвращён из `_archive/` → прогон `test_v19_runtime.py` +
`test_v19_document_ranker.py` дал **13 failed, 248 passed, 39 subtests**
(`D:/ziwago_baseline.log`) — ТЕ ЖЕ тесты и типы исключений, что и в архивном
состоянии (9 + 4). Все в НЕ тронутых файлах:
- `run_ziwago_v19.py:96` `evaluate_frozen_acceptance` не пробрасывает `mode` →
  `acceptance_ranker_metadata` (:81) дефолтит `closed_book` → ledger-ValueError
  не бросается; тест ждёт старого безусловного поведения (docstring :75-78
  документирует смягчение до open_book-only). 4 subtests document_ranker.
- `train_ziwago_v14_pure_synaptic_dialogue.py:3997` `pos_phase_vecs[pos_idx]`
  цикл `range(40)` против размера 16 → IndexError при training-induced CA3
  санитизации. 5 тестов v19_runtime (grammar/provenance ×3, training/reload).
- Windows-path сравнения снимков + `assert False` (test_v19_runtime:1013/1120/1151). 4 теста.

Эти падения НЕ вызваны архивацией и НЕ исправлялись здесь (защищённая зона
валидаций «три настоящие защиты» + вне scope задачи #21). Зафиксированы как
обнаруженные пред-существующие дефекты для отдельного решения владельца.