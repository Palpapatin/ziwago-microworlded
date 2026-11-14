# Ziwago ← Alexandria: BLUEPRINT РЕФАКТОРИНГА (Legacy Tower → Clean Connectome)

**Дата:** 2026-11-13 · **Проект:** `C:\Users\yarma\Desktop\Ziwago18`
**Рамка (принята владельцем):** Alexandria Sapience v3.0 — это **идеальный Clean Connectome**
(целевая спецификация, база знаний, НЕ исполняемый код). Ziwago v20 — это **рабочая Legacy Tower**
(живой организм, измеренный). Мы **не сравниваем их напрямую** — мы используем Alexandria как
**чертёж для рефакторинга** Ziwago.

**Метод.** Для каждого модуля и каждого решения:
`предписание Alexandria → текущее состояние Ziwago (измеренная живость + file:line) → конкретный разрыв → действие рефакторинга → приоритет → критерий приёмки`.

**Доказательная база.**
- [ARCHITECTURE_MAPPING.md](file:///C:/Users/yarma/Desktop/Ziwago18/ARCHITECTURE_MAPPING.md) — авторитетная карта живости (LIVE/DEAD/LOADED/BYPASSED), решения владельца, семьи дублей.
- [INNOVATION_PROPOSALS_2026-11-12.md](file:///C:/Users/yarma/Desktop/Ziwago18/INNOVATION_PROPOSALS_2026-11-12.md) — измеренные находки A/B/C/D + предложения Э/П/И.
- Глубокая верификация кода: 4 параллельных прогона (кора+ридаут; память+обучение+химия; организм+действие+сознание+таламус; генезис+язык+мотор+данные). Каждое утверждение ниже проверено по факту кода и вызывающим, не по классификации.

**Правила документа (AGENTS.md).** Каждое число — с источником. Отрицательный результат показывается прямо. N-граммы запрещены для генерации. Ответы не выдумываются. Три защиты от утечек не трогаются. Оригинал `Ziwago18original` — только чтение.

---

## 0. Главная находка верификации (меняет постановку рефакторинга)

> **Почти всё, что Alexandria описывает как «новый v3», УЖЕ существует в Ziwago — но как DEAD / opt-in-OFF / BYPASSED код, либо с неверной математикой/параметрами. Структурно отсутствует меньшинство.**

Это означает: рефакторинг — это **преимущественно оживление и проводка уже написанного**, а не строительство с нуля. Три класса действий доминируют:

1. **Оживить мёртвое** (ACL-астроцит, EWC/Fisher, BennaFusi, гиппокамп, θ-γ-closure, valence-generation) — код есть, вызывающих нет.
2. **Починить измеренные дефекты живого пути** (находки A/B/C/D) — код работает, но работает неправильно.
3. **Закрыть разомкнутые петли** (MetaHead-гейты → BG/TRN не подключены; hypothesis-sleep — no-op; Larkum — телеметрия) — сигнал вычисляется, но не потребляется.

И только затем — **достроить genuinely-absent** (TDR-декодер, NSM-самость, DCC-онтогенез, SCL-профили, ADMC-цель, TCC-тор).

### 0.1 Таксономия действий рефакторинга

| Класс | Смысл | Примеры в Ziwago |
|---|---|---|
| **WIRED-LIVE** | исполняется, уточнить | NeuroChemistry, SON GW, FusiZenke-щит, WorldGraph, v14-ридаут |
| **DEAD / OPT-IN-OFF** | код есть, не исполняется | AstrocyteMicrodomain, EWC, BennaFusi, hippocampus, laplace/working/phase_vsa, surrogate/predictive/sync, agents/volume (OFF), dual_hippocampus (OFF) |
| **BYPASSED** | построен+сохранён, перекрыт | v18 genesis-генерация: 6-slot θ-γ, closure-verifier, valence-wave |
| **WRONG** | исполняется, но математика/параметры ≠ спеке | MCA τ_tag/статичный m; щит magnitude вместо Fisher; ACh без ado; CSGL declared вместо learned |
| **ABSENT** | нет реализации | TDR-декодер, NSM mood/v_self, DCC m_z, SCL-профили, ADMC J-цель, TCC-тор |

### 0.2 Приоритизация по измеренным находкам (что блокирует интеллект)

| Находка (источник: INNOVATION_PROPOSALS) | Измерение | Какие модули蓝图 критичны |
|---|---|---|
| **A** — нейромодуляция не может изменить ответ | `readout_gain` argmax-инвариантен (0.56…1.90 → argmax 393); корковый канал пуст | **TDR + П-1 + П-2** |
| **D** — корковый канал пуст по построению | `last_features.norm=0.0`; `F.normalize(0)=0` (spiking_language:65); `w_projector` выучен (std 0.00225 vs 0.027) | **П-2 → TDR** |
| **B** — щит замораживает смысловые каналы | 34.5% строк attenuation<1e-3; замороженные держат 1.32× веса; `u_slow` max 866.5 | **FSP + MCA + Э-6 + И-5** |
| **C** — стоимость структурно лишняя | 87.5% коры спит (2191 МБ); 22.8% латентности — снимки; пик загрузки ~10 ГБ → OOM | **SCL + Э-1..Э-4 + Э-7** |
| **compose_prefix=0/4** (GLOBAL_PLAN_2026-11-09.md:33) | композиция провалена при replay 4/4, compose_question 4/4 | **CSGL + TDR (form-layer) + И-2** |

**Критический путь:** `Э-7/SCL (снять OOM) → П-2/П-1/TDR (состояние→речь) → Э-6/FSP/MCA (обучение) → проводка петель → оживление мёртвого`.

---

## 1. Архитектурный уровень (каркас)

**Что уже соответствует чертежу.** Плоская композиция `ZiwagoOrganism`
([ziwago/organism_core.py:65](file:///C:/Users/yarma/Desktop/Ziwago18/ziwago/organism_core.py#L65), base=`object`, без wrapper-наследования) — это и есть Alexandria-принцип «единый коннектом, не башня». Флаг `integrated` (:75) переключает v19 (`NeuroChemistry`, schema 19.0) ↔ v20 (`OrganismPhysiology`+`core.life`, schema 20.0). Genesis-модули (:90-116) и v19-модули (:124-139) — независимые атрибуты. **Каркас рефакторинга уже готов; проблема не в каркасе, а в наполнении и проводке.**

**Что архитектурно отсутствует против чертежа:**
- **SCL** — нет именованных масштабных профилей в production (пресеты `config.py:309-329` — мёртвый контейнер). Живой MoE = 16×1024 = 16384 нейрона, `top_k=2` (v14:406-413) — по числу нейронов это Tiny, но кора весит 2504 МБ (находка C), чекпоинт 4.224 ГБ.
- **DCC** — нет зонарного онтогенеза; все зоны созревают одновременно.
- **PCI / three-tier subjectivity** — мета-уровень вычисляет гейты, но не замкнут на действие; PCI-прокси недоступен.
- **Единые часы** — v20 ms-clock есть (`physiology.py:307`), но синаптические τ считаются в тиках (`dt=1.0`, column.py:157), не в.wall-seconds → разрыв MCA/FSP/NSM τ-масштабов.

**Архитектурное действие №1 (сквозное):** ввести **единую временную базу** (ms-clock v20 как канон) и **явный SCL-профиль Tiny** как production-конфиг. Без этого все τ-зависимые модули (MCA, FSP, NSM, DCC, ACL) нельзя калибровать против чертежа.

---

## 2. Per-module blueprint — v3 (реализуемо сейчас)

Сводная таблица (детали ниже):

| # | Модуль Alexandria | Класс Ziwago | Приоритет | Суть действия |
|---|---|---|---|---|
| 1 | **TDR** гибридный декодер | ABSENT + находки A/D | **P0** | построить 2-стадийный декодер + ε-отказ |
| 2 | **MCA** 4-компонентный синапс | WRONG (τ_tag, статичный m) | **P1** | починить τ, оживить m/BennaFusi |
| 3 | **ACL** вычислительный астроцит | DEAD + WRONG | P2 | оживить + sigmoid g_i + PNN G_i |
| 4 | **FSP** EWC×Sleep | WRONG + DEAD (Fisher) | **P0/P1** | Fisher вместо magnitude + NREM rebaseline |
| 5 | **DCC** онтогенез | ABSENT | P2 | m_z + η/σ-созревание + R_z closure |
| 6 | **NSM** нарративная самость | ABSENT | P2 | mood M(t) + v_self → TDR λ_self |
| 7 | **SCL** масштабные профили | ABSENT (production) | **P0** | Tiny-профиль + Э-7 split |
| 8 | **TCC** тороидальный CANN | ABSENT + DEAD hippocampus | P3 | тор/реляции; spatial — горизонт |
| 9 | **ADMC** аллостатическое любопытство | ABSENT (цель) + WRONG (ACh) | P2 | ACh-ado формула + J-цель |
| 10 | **CSGL** причинно-символьное заземление | WRONG (declared) + BYPASSED closure | **P1** | open-roles + learned SCM + closure в генерацию |
| 11 | **PCI / three-tier** субъективность | PARTIAL (петли разомкнуты) | **P1** | MetaHead→BG/TRN + PCI-растр |

---

### 2.1 TDR — гибридный токенный декодер 〔ABSENT · P0〕

**Предписание Alexandria.** Двухстадийный декод: этап 1 — пул кандидатов `K_A=64` (инвертированный индекс по активным битам SDR) ∪ `K_B=64` (VSA-косинус), `≤128` вместо полного словаря 50257; этап 2 — точный скоринг `s(v)=⟨h,v_w⟩/(‖h‖‖v_w‖)+λ_prior·log P₀(v)+λ_self·⟨v_self,v_w⟩`, bounded-softmax с **токеном отказа ε** (`θ_conf=0.42`, `T=0.7`); если ε доминирует — токен НЕ порождается (anti-hallucination gate, синхронизирован с DDM-аккумулятором). VSA-словарь weight-tied с энкодером.

**Текущее состояние Ziwago (верифицировано).**
- [ziwago/cognition/language/readout.py:698](file:///C:/Users/yarma/Desktop/Ziwago18/ziwago/cognition/language/readout.py#L698) `DendriticPlateauMatrixReadout.predict_next_token_logits` суммирует по **вему зарегистрированному словарю** `n_active=len(col_to_tok)` (:706), затем полный softmax (v14:1865). Это dense sparse-row матрица, НЕ 2-стадийный пул.
- [ziwago/cognition/language/predictive_readout.py:186](file:///C:/Users/yarma/Desktop/Ziwago18/ziwago/cognition/language/predictive_readout.py#L186) `PredictiveCodingCorticalReadout.score_candidates` — это **бонус-перескоринг top-48** (v14:854-860, `2.2·pc`), не гейт пула. `token_embed`/`w_readout` (:41-42) — независимые случайные матрицы, **не weight-tied VSA**.
- **Токена отказа ε нет.** Воздержание = строковый шаблон (`ziwago_assistant.py:129`) или `conflict_abstain` по JS-дивергенции (`language_conflict_arbitration=False`, v14:870). `θ_conf=0.42` существует только как contour-порог MD-роутера (genesis_engine:1429), не как уверенность декодера.
- **Находка A подтверждена в коде:** v14:1191 `readout_logits *= readout_gain` (argmax-инвариантный скаляр), :1192 `cortical *= cortical_gain` (канал пуст — находка D).

**Разрыв.** Нет пула кандидатов K_A/K_B; нет weight-tied VSA-словаря; нет ε/θ_conf/T; нет DDM-синхронизации; нет `λ_prior·log P₀` и `λ_self·⟨v_self,v_w⟩` (IDF в PC — это train-time LR-вес, не inference-приор).

**Действие рефакторинга.**
1. Построить TDR как **замену финального softmax-стадия** v14:1865: инвертированный индекс по активным битам `d_NMDA` (уже top-32, readout:584) → K_A; VSA-косинус по `CompositionalVSABPEEncoder` (v9, уже в живом пути) → K_B; union ≤128.
2. **Weight-tie** VSA-словарь с энкодером (TDR-5) — переиспользовать существующий roll-bind readout.
3. Добавить **ε-токен** с `θ_conf=0.42`, `T=0.7`; синхронизировать с `ContextualActionGate` DDM (contextual.py) — сброс аккумулятора только при успешной эмиссии (TDR-4).
4. `λ_self·⟨v_self,v_w⟩` подключается к **NSM v_self** (§2.6) — TDR и NSM связаны.
5. После оживления коркового канала (**П-2**) и неоднородных гейнов (**П-1**) `cortical_gain` наконец сможет сдвигать argmax.

**Приоритет P0** — критический путь «состояние→речь» (находки A/D) + `compose_prefix=0/4`.
**Критерий приёмки.** `compose_prefix>0/4`; argmax сдвигается при пертурбации химии (прибор П-5); воздержание через ε измеримо; FLOPs/токен ↓ (Alexandria заявляет ~396× аналитически — **это ASSERTED, не измерено**; наш критерий — собственное измерение латентности, не чужое число).

---

### 2.2 MCA — 4-компонентный синаптический вектор 〔WRONG · P1〕

**Предписание Alexandria.** Каждый синапс несёт `(w, e=eligibility, tag, m=metaplasticity)`; `τ_e∈[0.1,2]s`, `τ_tag~3600-7200s`, `τ_m~10⁴-10⁵s`. Цели ≥87% one-shot, −64% ATP.

**Текущее состояние (верифицировано).** [ziwago/core/synapse.py:44](file:///C:/Users/yarma/Desktop/Ziwago18/ziwago/core/synapse.py#L44) `GenesisSynapse` **уже имеет 4 компоненты**: `w_slow+w_fast` (:59-60), `eligibility` (:66), `tag` (:69), `theta_m` (:74). Но:
- `τ_e≈1000` тиков (`1−1/1000`, :422/:497) ✓ по порядку; **`τ_tag≈2000` тиков** (:421/:464) — против цели 3600-7200 **секунд**.
- Живые часы `dt=1.0` на корковый тик (column.py:157) → **тики, не wall-seconds**. Разрыв масштабов ~3 порядка.
- **`m` (theta_m) СТАТИЧЕН** `=theta_m0=0`; `set_metaplastic` (:182) **никогда не вызывается в живом пути** (только `_archive/.../genesis.py:328`). Нет медленного `τ_m~10⁴-10⁵s`.
- **BennaFusiCascade** ([ziwago/cortex/column.py:124](file:///C:/Users/yarma/Desktop/Ziwago18/ziwago/cortex/column.py#L124), 4 стадии, taus 10/80/640/5120) инстанциирован, но **никогда не шагает** (grep-подтверждено).
- SI-консолидация (:730-791, `Ω+=relu(ω)/((Δw)²+ξ)`) жива, но отдельна от EWC-Fisher.

**Разрыв.** τ_tag на 3 порядка короче цели; нет пер-синаптического медленного m; каскад мёртв; временная база — тики.

**Действие рефакторинга.**
1. Перевести синаптические τ на **ms-clock v20** (архитектурное действие №1).
2. Подключить `set_metaplastic` к химии (`theta_m` из `modulator_bundle`) — оживить m.
3. **BennaFusi: шагать в column learn-path ИЛИ архивировать** (решение владельца; сейчас это мёртвый балласт).
4. Разделить **зрелость** (`u_slow`) и **значимость** (`|w|`) — это Э-6, см. FSP §2.4.

**Приоритет P1** (обучение, находка B).
**Критерий приёмки.** τ_tag в секундах соответствует профилю; m динамилен; one-shot replay ↑; (совместно с FSP) доля attenuation<1e-3 ↓.

---

### 2.3 ACL — вычислительный астроцит 〔DEAD + WRONG · P2〕

**Предписание Alexandria.** `τ_C≈2-5s`, `ρ=0.05` латеральная диффузия, `g_i=σ(4(C−0.2)/0.15−1)`, `f_NMDA=0.3+0.7·g_i`, **кумулятивный PNN-гейт** `G_i=∫C·1_awake` с порогами `θ1=50s/θ2=250s` → состояния PLASTIC/SOFT/HARD. Заявлено <0.8% FLOPs.

**Текущее состояние (верифицировано).** [ziwago/cortex/astrocyte.py](file:///C:/Users/yarma/Desktop/Ziwago18/ziwago/cortex/astrocyte.py) `AstrocyteMicrodomain(n_neurons, tau_ca=2500.0, gamma_gli=0.65, theta_ca=0.20, diffusion=0.05, ne_boost=1.5)`:
- `τ_C=2.5s` ✓, `ρ=0.05` ✓ — **параметры уже совпадают**.
- Но `glio=tanh(ca/theta_ca)` — **tanh, не sigmoid** `σ(4(C−0.2)/0.15−1)`; нет `f_NMDA`; **нет кумулятивного PNN G_i**.
- **Модуль DEAD в живом пути** — конструируется только `_archive/neuroweave10/organism/genesis.py:95`. Живой «астро» — `AstroNMDALinearRecurrence` (bio-RWKV, moe_cortex:1023), **другой механизм**.

**Разрыв.** Мёртв + неверная активация + отсутствует PNN-гейт пластичности.

**Действие рефакторинга.**
1. Оживить `AstrocyteMicrodomain` в column.step — он уже модулирует `stp_gain` L4→L2/3 и `theta_ca_shift` (прямое действие на физику связей, не канал декодера).
2. Заменить `tanh → sigmoid g_i`; добавить `f_NMDA=0.3+0.7·g_i`.
3. Добавить **кумулятивный интеграл бодрствования G_i** → PLASTIC/SOFT/HARD. Это биологическая опора для **Э-6** (щит защищает от перезаписи, не от роста) и **DCC** (HARD = закрытое критическое окно).

**Приоритет P2** (оживление мёртвого; поддерживает обучение + онтогенез).
**Критерий приёмки.** glio-гейт измеримо модулирует stp_gain в живом тике; переходы G_i наблюдаемы; overhead <1% FLOPs (собственный замер).

---

### 2.4 FSP — Fusi-Zenke × Sleep Plasticity 〔WRONG + DEAD · P0/P1〕

**Предписание Alexandria.** `F_HARD=F∞≈10⁶ / F_PLASTIC·0.99`; в NREM-сне Fisher-rebaseline `F←(1−ρ)F+ρ·E[(∂ln p/∂θ)²]`, `ρ∈[0.1,0.3]`. EWC (диагональный Fisher важности) + фазовая консолидация.

**Текущее состояние (верифицировано).**
- [ziwago/learning/ewc.py](file:///C:/Users/yarma/Desktop/Ziwago18/ziwago/learning/ewc.py) `EWCRegularizer(lam=50)`, `estimate_fisher(n_samples=64)`, `rebaseline()` — **DEAD, никогда не armed/estimated** в production («0 Fisher», У-3).
- Живой щит [ziwago/cortex/next16_core.py:363](file:///C:/Users/yarma/Desktop/Ziwago18/ziwago/cortex/next16_core.py#L363) `FusiZenkeCascadeMetaplasticityShield(lambda_cons=0.85, delta_decay=0.002, kappa_shield=18)` — **magnitude-based**, не Fisher: `consolidate` (:404) `u_slow.mul_(1−0.002).add_(0.85·mag·(mag>0.05))`; `apply_sparse_update` (:441) `w/(1+18·u_slow)`.
- **Находка B подтверждена:** высоко-|w| строки накапливают u_slow (max 866.5) → 34.5% строк attenuation<1e-3; замороженные держат 1.32× веса живых. **Щит карает за знание.**
- Сон: 5 реализаций; v20 uniform `w_dend_out.mul_(0.992)` (organism_core:464), v19 селективный SWS (sleep.py:30, weak<0.05 ×0.85). **Нет NREM Fisher-rebaseline.**
- **Hypothesis-sleep — NO-OP в production:** [ziwago/organism/verified.py:323](file:///C:/Users/yarma/Desktop/Ziwago18/ziwago/organism/verified.py#L323) `consolidate(validator=None)` → `{"accepted":False,"reason":"Independent acceptance validator required."}`; assistant `/sleep` (:356) передаёт None.

**Разрыв.** Fisher никогда не оценивается/rebaseline-ится; щит magnitude вместо Fisher; сон без FSP-фазового протокола; hypothesis-sleep заблокирован валидатором.

**Действие рефакторинга.**
1. **Сначала контрольный эксперимент И-5** (`kappa_shield=0` vs `18`) — изолировать вклад щита в «модель не учится» (P0, дёшево).
2. Arm EWC на живых строках ридеаута; оценивать Fisher **локальным прокси** (surrogate запрещён в живом цикле, spiking_language:3 → использовать уже существующую Rao-Ballard/PC-ошибку predictive_readout или `eligibility²` как диагональный Fisher).
3. Заменить magnitude `u_slow` на **Fisher-важность**; добавить NREM-rebaseline `ρ` в sleep_cycle.
4. Починить validator-gating, чтобы hypothesis-sleep реально исполнялся (или явно задокументировать no-op как решение).
5. Э-6: разделить зрелость/значимость (см. MCA).

**Приоритет P0/P1** — находка B есть кандидатная корневая причина «не учится».
**Критерий приёмки.** И-5: delta replay/compose при kappa=0; профиль attenuation ↓ с 34.5%; Fisher ненулевой после сна; `max|w|` замороженных ≤ живых.

---

### 2.5 DCC — онтогенез / критические окна 〔ABSENT · P2〕

**Предписание Alexandria.** Зонарная зрелость `m_z=1−e^(−t/τ_z)`, `τ_SENS<τ_ASSOC<τ_HIPPO<τ_PFC`; `g_I∝m²`; `η=η_max·4m(1−m)` (перевёрнутая U); `σ=σ0·e^(−2.3m)`; гейт закрытия `R_z=0.4·ITPC+0.4·A+0.2·(1−|EI−1|)≥0.60`, удерживается 50 θ-циклов.

**Текущее состояние (верифицировано).** **Отсутствует в живом пути.** `ZoneCfg` τ-градиент (config.py:101) — мёртв (только GenesisOrganism). Нет m_z, нет staged-созревания, нет closure-гейта. Курикулум — 7 hardcoded уроков ([ziwago/app/curriculum.py](file:///C:/Users/yarma/Desktop/Ziwago18/ziwago/app/curriculum.py), 5 эпох) — staging **по контенту, не по зрелости зон**. `f_mod` v19 floor-clamp 0.25 (v14:1341) vs v20 true-zero (physiology:354).

**Разрыв.** Нет расписания развития; все зоны созревают одновременно; нет закрытия критических окон.

**Действие рефакторинга.**
1. Добавить пер-зонное состояние зрелости `m_z`, управляемое ms-clock v20.
2. Связать `η` (learning rate) и `σ` (шум) с `m_z` (перевёрнутая U / экспоненциальный спад).
3. Гейт закрытия пластичности зоны по `R_z`; **соединить с ACL G_i** (HARD = закрытое окно) — §2.3.
4. Это принципиальная версия курикулума: не «7 уроков», а «зоры созревают в порядке τ».

**Приоритет P2** (после разблокировки обучения).
**Критерий приёмки.** Кривые созревания зон наблюдаемы; η перевёрнутая-U; события закрытия логируются; нет регрессии replay.

---

### 2.6 NSM — нарративная самость 〔ABSENT · P2〕

**Предписание Alexandria.** Событие `e_k=(h,t,v,s)`; настроение `M(t)∈[−1,1]`, `τ_M≈3600-7200s`; валентность `v=tanh(∫da_phasic)`; персистентный self-вектор `v_self`, `K=1024`; **только аддитивный bias** в генерацию.

**Текущее состояние (верифицировано).** **Отсутствует.** [ziwago/emotion/limbic.py:135](file:///C:/Users/yarma/Desktop/Ziwago18/ziwago/emotion/limbic.py#L135) `valence=clamp(app−av,[−1,1])` → `da_phasic+=0.5·valence` (integrated:400-401, opt-in `limbic_enabled=False`) — **быстрый аддитивный, не персистентное настроение/самость**. Нет `v_self`, нет медленного интеграла `M(t)`. В химии нет mood-поля.

**Разрыв.** Нет персистентной самости/настроения; валентность моментальная, не интегрированная.

**Действие рефакторинга.**
1. Добавить медленный интегратор настроения `M(t)` (τ_M часы) поверх существующего `da_phasic`.
2. Построить `v_self` как персистентный VSA-вектор (K=1024) из накопленных эпизодов.
3. Подать `λ_self·⟨v_self,v_w⟩` в **TDR-скоринг** (§2.1) — NSM и TDR связаны одним членом.

**Приоритет P2** (зависит от TDR для bias-канала).
**Критерий приёмки.** M(t) персистентен между репликами; v_self измеримо сдвигает скоринг; нет роста галлюцинаций.

---

### 2.7 SCL — масштабные профили 〔ABSENT в production · P0〕

**Предписание Alexandria.** `state=30·N` байт; **Tiny** 16k нейронов/2.56M синапсов/1.8 ГБ; **Medium** 500k/80M/24.5 ГБ; **Full** 10M/10¹⁰/480 ГБ. Явные именованные конфигурации.

**Текущее состояние (верифицировано).** Пресеты `genesis_cfg tiny/poc/full` (config.py:309-329) — **мёртвый контейнер, не production**. Живой MoE = 16×1024 = 16384 нейрона, `top_k=2` (v14:406-413) — по числу нейронов Tiny, но кора 2504 МБ (находка C), чекпоинт 4.224 ГБ, пик загрузки ~10 ГБ → OOM (AGENTS.md §5.3). Нет именованных production-профилей, нет бюджета `30·N`.

**Разрыв.** Нет явной масштабной конфигурации; память структурно потрачена (87.5% спящей коры, 74.6% бухгалтерия обучения).

**Действие рефакторинга.**
1. Объявить **SCL Tiny production-профилем** — единственным, что влезает в 15.9 ГБ RAM / 6 ГБ VRAM. Medium/Full — **горизонт, не строить** (24.5/480 ГБ не влезают).
2. **Э-7 (P0):** разделить чекпоинт `*.core.pt` + `*.plasticity.pt` + телеметрия → пик загрузки 10→~5 ГБ. Это **самый большой разблокировщик проекта** (снимает OOM-блокировку приёмки v20).
3. Э-1 горячие/холодные колонки; Э-2 прореживание строк ридеаута (CSR); Э-3/Э-4 снимки только изменяемого + accessor-замыкания.

**Приоритет P0** (аппаратная разблокировка).
**Критерий приёмки.** Пик загрузки ≤5 ГБ; `state_fingerprint` побайтно идентичен; Tiny-профиль именован и подключён.

---

### 2.8 TCC — тороидальный CANN 〔ABSENT + DEAD hippocampus · P3〕

**Предписание Alexandria.** Grid/place на торе `T²`, `λ ratio≈1.42`, 8 модулей; gridness≥0.68, path-integration≤0.08 rad, transitive inference>94.2%.

**Текущее состояние (верифицировано).** [ziwago/memory/hippocampus.py:53](file:///C:/Users/yarma/Desktop/Ziwago18/ziwago/memory/hippocampus.py#L53) EC-grid = **1-D cos** (`grid_freq/grid_phase`), не тор; нет 8 модулей, нет λ≈1.42, нет метрик. **DEAD в production** (только dual_hippocampus opt-in + v9, который v14 не инстанциирует). `dual_hippocampus.MergedLaplaceBank` (:43) — **TIME-клетки, не SPACE**. `laplace.py`/`working.py`/`phase_vsa.py` — все DEAD.

**Разрыв.** Нет пространственно-реляционного тороидального кода; гиппокампальный стек мёртв.

**Действие рефакторинга.** Это территория решения **П-1** (гиппокамп). Либо оживить nw10 спайковый гиппокамп (М-1) с тор-модулем, либо оставить живые CA3-полюса + closure и добавить TCC как реляционный модуль. **Для текстовой модели spatial path-integration — низкий приоритет; ценна transitive inference (реляции) для CSGL.**

**Приоритет P3** (оживление мёртвого; spatial-часть — горизонт для текста).
**Критерий приёмки.** gridness/transitive метрики если строится; иначе — явное решение владельца.

---

### 2.9 ADMC — аллостатическое любопытство 〔ABSENT цель + WRONG ACh · P2〕

**Предписание Alexandria.** `J=ΔE_progress − λ_ATP·Cost`; `[ACh]=[ACh]₀/(1+([Ado]/θ)²)` — ACh подавляется аденозином (давление сна).

**Текущее состояние (верифицировано).** Теневая цена энергии **есть**: `interoception.py:26/105 allostatic_drive`; `lambda_atp=1−atp` (moe_cortex:1172); метаболический LTP-гейт `synapse.py:509-511` `1/(1+0.65·λ_ATP)`. Но **нет J-цели**; **ACh=0.4+0.8·attention** (physiology:85/:299) — **без ado-подавления**.

**Разрыв.** Нет цели любопытства; ACh не связан с давлением сна.

**Действие рефакторинга.**
1. Добавить **ACh-ado формулу** в physiology (однострочная правка; биологически верно — любопытство падает с усталостью).
2. Определить `J` как exploration-бонус в action-gate / выборе курикулума.

**Приоритет P2** (малая проводка, поддерживает сон + исследование).
**Критерий приёмки.** ACh падает при росте ado; J логируется; исследование измеримо.

---

### 2.10 CSGL — причинно-символьное заземление 〔WRONG + BYPASSED · P1〕

**Предписание Alexandria.** 3 уровня Pearl (ассоциация/интервенция/контрфактуал); Corr2Cause≥88.5%, Counterfactual≥91.8%, Hallucination<0.4%.

**Текущее состояние (верифицировано).** [ziwago/cortex/genesis_engine.py:273](file:///C:/Users/yarma/Desktop/Ziwago18/ziwago/cortex/genesis_engine.py#L273) `CorticalRelationalWorldGraph` **LIVE** (assistant:109/239/293): `intervene` (:672) — 3-значное do-style распространение, `ground_response` (:574) closure fail-closed. Но docstring: **«not a learned structural causal model and does not identify observational counterfactuals»**. `BA10CounterfactualInterventionSimulator` (:1268) **LIVE**, но `source="declared_causal_model"`, только 3 глагола (release/produce/generate, :1291-1299), **«not identified Pearl level-3»**. `EmergentOntologicalRoleDiscovery` (:795) **LIVE**, но **HARDCODED** `valid_role_heads={"mayonnaise","essay","running","yolk","fox"}` (:992) — ломает open-world. `BidirectionalMeaningClosureVerifier` (:1156) — **НОЛЬ живых вызывающих**. Нет абдукции, нет learned SCM, ассоциации не дают причинных hops. Нет benchmark-harness.

**Разрыв.** Только ассоциация + declared-интервенция; нет выученной причинной структуры, нет абдукции, hardcoded роли, closure-verifier обойдён.

**Действие рефакторинга.**
1. Убрать hardcoded role-heads → open-world role discovery.
2. Подключить `BidirectionalMeaningClosureVerifier` в генерацию — это **П-4** (задача #19).
3. Апгрейд WorldGraph с declared на **learned edge weights** (STDP на причинных hops) — И-4 «учить связи, не токены».
4. Абдукция (level-3) — горизонт.
5. Построить **Corr2Cause/CF benchmark-harness** для измерения (сейчас не измерено; Alexandria-числа ASSERTED).

**Приоритет P1** (композиционность `compose_prefix=0/4` + галлюцинации).
**Критерий приёмки.** closure-verifier исполняется в генерации; role-heads открыты; CF-benchmark существует + baseline измерен.

---

### 2.11 PCI / three-tier subjectivity 〔PARTIAL, петли разомкнуты · P1〕

**Предписание Alexandria.** `PCI_SNN=CLZ(R)/(|R|·H(p))≥0.52`; трёхуровневая субъективность.

**Текущее состояние (верифицировано).** SON GW ([ziwago/consciousness/workspace.py:34](file:///C:/Users/yarma/Desktop/Ziwago18/ziwago/consciousness/workspace.py#L34)) + MetaHead (meta_head.py:40) + AST (attention_schema.py:32) **LIVE**. Но:
- **MetaHead-гейты** (:135-140 `bg_threshold/beta_gw_boost/replay_priority/trn_gain`) вычисляются, но потребляются **ТОЛЬКО в `_archive/.../genesis.py`** — production BG/TRN их **не читают** (grep-подтверждено).
- **PCI-прокси недоступен:** assistant:137 dashboard не передаёт `perturbed_spike_raster` → next16:668 `None`.
- **AST шагает только v20** (integrated:253); в v19 inert (нет вызывающего).
- `consciousness/metrics.py:87` — **DEAD-дубль** next16:642.

**Разрыв.** Мета-познание вычисляет гейты, но не замкнуто на действие; PCI неизмерим; AST полу-жив.

**Действие рефакторинга.**
1. Подключить MetaHead-гейты → `ContextualActionGate.threshold` + TRN gain + replay priority (**замкнуть мета→действие петлю**).
2. Подать perturbed-растр в dashboard → включить PCI-прокси.
3. Шагать AST в v19 ИЛИ явно задокументировать v20-only.
4. Архивировать metrics.py-дубль.

**Приоритет P1** — закрытие существующих петель, дёшево, высокая ценность («проводка, не строительство»).
**Критерий приёмки.** MetaHead-гейты измеримо меняют BG/TRN; PCI-прокси ≠ None; AST жив в обоих режимах.

---

## 3. Per-module blueprint — v3.1 Singularity (горизонт)

| Модуль | Спека | Вердикт для Ziwago | Обоснование |
|---|---|---|---|
| **Aither** | sub-Landauer 2.87e-22 J, photonic-memristive | **ГОРИЗОНТ, не строить** | требует фотонно-мемристорного железа; наша RTX 1660 Ti не подходит |
| **TMM** | persistent homology Ripser β0/β1/β2 | **PORTABLE, опционально** | чистая математика; топологическая метрика на world-graph/растре; низкий приоритет |
| **HVF** | quaternion affect q=a+bi+cj+dk (ссылается на наш da_phasic) | **PORTABLE, зависит от NSM** | расширение limbic-валентности до кватерниона; после NSM |
| **ONE** | de-novo neurogenesis GRN K=2, >85% HARD | **ГОРИЗОНТ, не строить** | структурный рост конфликтует с фиксированными SCL-профилями + schema чекпоинта |
| **PCI** | phenocentric core ≥0.52 | **реализуемо как прокси сейчас** | см. §2.11 — подать perturbed-растр |
| **LMP** | glossomotor v_gesture=v_ac⊗v_art⊗v_som >96% | **OWNER-DEAD П-3, не строить** | мотор/арена/syllabary не нужны текстовой модели (решение владельца) |
| **Aesthetic** | I_aes=−dL(W)/dt | **ГОРИЗОНТ** | нужен loss-landscape, которого нет (no global backprop); опциональный curiosity-сигнал позже |

**Вывод по v3.1:** из 7 только **PCI-прокси** реализуем сейчас, **TMM/HVF** переносимы опционально. Остальные 4 — горизонт или owner-dead. **Не тратить ресурс машины на горизонт.**

---

## 4. Per-decision blueprint —映射 существующих решений П/Э/И

| Решение | Статус (MAPPING/INNOVATION) | Модуль蓝图 которому служит | Действие рефакторинга |
|---|---|---|---|
| **П-1** неоднородные по каналам гейны | предложение | TDR / находка A | `readout_gain` скаляр → пер-канальный/пер-строчный гейн; иначе argmax не сдвинуть |
| **П-2** живые корковые признаки | предложение | TDR / находка D | `normalize+eps` в spiking_language:65; внешний clock; оживить `cortical_gain` |
| **П-3** оживить criticality/BG/volume/interoception/limbic | **СДЕЛАНО §8.3-8.6**, но все opt-in OFF | ADMC/NSM/PCI |选择性 включить ПОСЛЕ верификации прибором П-5 |
| **П-4** оживить v18 genesis-генерацию (θ-γ/closure/valence) | **PENDING #19** | CSGL/TDR form-layer | нюанс верификации: 6-slot θ-γ BYPASSED, но 1-slot `temporal_codec` LIVE (v14:476); closure-verifier 0 вызывающих; valence-generation 0 вызывающих (test_v19_causal_temporal:212 assert) |
| **П-5** данные/атлас | частично | SCL | `app/data.py:16` только `TokenizedDialogPair`; `SixDialogCorporaStreamer` НЕ мигрирован (v14:129, circular BPE import) |
| **Э-1** горячие/холодные колонки | предложение | SCL / находка C | dynamic vs static разделение; 2191 МБ спящей коры |
| **Э-2** прореживание строк CSR | предложение | SCL / находка C | 606/8192 строк (7.4%); 344→25.4 МБ |
| **Э-3** снимок только изменяемого | предложение | SCL / находка C | 22.8% латентности; learn=False → веса не меняются |
| **Э-4** accessor-замыкания | предложение | SCL / находка C | 720698 вызовов `_get_runtime_path` → список замыканий |
| **Э-5** лимитер потолок→цель | предложение | FSP/MCA обучение | `readout_step_limited=20` всегда режет; множитель 0.211 → средний шаг ↑4.7× |
| **Э-6** щит защита-от-перезаписи | предложение | FSP/MCA / находка B | разделить u_slow-зрелость и |w|-значимость; клиппинг нормы строки |
| **Э-7** split чекпоинта | предложение **P0** | SCL | core/plasticity/telemetry; пик 10→~5 ГБ; снимает OOM-блокировку |
| **И-1** апикал в дендритное состояние | **негатив уже измерен** | MCA/Larkum | в логиты replay 4/4→3/4; в состояние — проверять |
| **И-2** два слоя content/form | предложение | TDR/CSGL | `w_ca3_sdr [526,2048]`, arcuate; compose_prefix=0/4 |
| **И-3** относительный позиционный код | предложение | TDR | Alexandria GENESIS 0% pos_idx |
| **И-4** учить связи не токены | предложение | CSGL/MCA | learned SCM edges |
| **И-5** kappa_shield=0 эксперимент | предложение **P0** | FSP / находка B | контрольный: изолировать вклад щита |

---

## 5. Дорожная карта рефакторинга (волны)

**Волна 0 — Разблокировка (P0, сначала измерить).**
- Э-7 split чекпоинта → пик ≤5 ГБ (снимает OOM, AGENTS.md §5.3).
- SCL Tiny как production-профиль.
- И-5 контрольный эксперимент `kappa_shield=0` vs `18` (изолирует находку B).
- *Критерий:* загрузка проходит на этой машине; state_fingerprint побайтно; delta replay/compose от kappa измерена.

**Волна 1 — Состояние→речь (P0, находки A/D).**
- П-2 оживить корковые признаки (`normalize+eps`, внешний clock).
- П-1 неоднородные по каналам гейны.
- TDR 2-стадийный декодер + ε-отказ + weight-tied VSA.
- *Критерий:* argmax сдвигается при пертурбации химии; `compose_prefix>0/4`; ε-воздержание измеримо.

**Волна 2 — Обучение (P1, находка B).**
- Э-6/FSP: Fisher вместо magnitude + NREM-rebaseline.
- MCA: τ на ms-clock, оживить m, BennaFusi шагать-или-архив.
- Э-5 лимитер потолок→цель.
- *Критерий:* attenuation<1e-3 ↓ с 34.5%; max|w| замороженных ≤ живых; one-shot ↑.

**Волна 3 — Замкнуть петли (P1, проводка).**
- MetaHead-гейты → BG/TRN/replay; PCI-растр; AST в v19.
- П-4 closure-verifier в генерацию; CSGL open-roles.
- Починить hypothesis-sleep validator (или задокументировать no-op).
- *Критерий:* гейты измеримо меняют действие; PCI≠None; closure исполняется.

**Волна 4 — Оживить мёртвое选择性 (P2, после верификации).**
- ACL астроцит (sigmoid g_i + PNN G_i); ADMC ACh-ado + J; NSM mood/v_self; DCC m_z; dual_hippocampus.
- Включать opt-in по одному, проверяя прибором П-5 (живость механизма), не чтением кода.
- *Критерий:* каждый оживлённый механизм даёт ненулевой измеримый вклад; нет регрессии.

**Горизонт (не строить сейчас):** TCC spatial, v3.1 Aither/ONE/LMP/Aesthetic; TMM/HVF опционально.

---

## 6. Ограничения владельца (что НЕ делать при рефакторинге)

- **N-граммы запрещены для генерации** (`use_ngrams=False`, organism_core:87/v14:803). N-gram-признаки `document_qa.py` — retrieval-only, допустимы.
- **Мотор/арена/syllabary — DEAD по П-3** (не нужны текстовой модели). LMP не строить.
- **Три защиты от утечек** (пересечение семейств, утечка промптов, утечка документов) — не трогать.
- **Оригинал `Ziwago18original` — только чтение.** Перед удалением — сверить с оригиналом, зафиксировать в манифесте.
- **Мастер архивирован (Variant A)**, SHA `cb97c869…`, 1.32 ГБ, `D:/Ziwago18-storage/Archive/cleanslate_20261112/master/`.
- **Ответы не выдумываются.** Свежий организм отвечает шаблоном отказа (v14:4352-4355) — поведение измерять только на обученных весах.
- **Аппарат:** RAM/VRAM/CPU ≤85%; тяжёлые процессы только через `tools/limited_run.py --ram-pct 55 --cpu-pct 50`; не запускать тяжёлое обучение + pytest одновременно; длинные процессы отсоединённо (~40 мин лимит рантайма). Хранение тяжёлых артефактов на `D:`.
- **Предсуществующие дефекты НЕ чинить без решения владельца:** `range(40)` vs `max_pos=16` IndexError (v14:2332); Windows-path assertions; garbage_regression answer-collapse.
- **Каждое число — с источником; отрицательный результат прямо; пропущенный тест = дефект.** Alexandria-числа (396× TDR, 87% one-shot, gridness 0.68 и т.д.) — **ASSERTED, не измерены**; использовать как цели, но приёмку строить на собственных замерах.

---

## 7. Итог

Рефакторинг Ziwago по чертежу Alexandria — это **не переписывание**, а **три слоя работы**:
1. **Разблокировать** (Э-7/SCL — снять OOM; И-5 — измерить вклад щита).
2. **Починить живое** (находки A/D → TDR/П-1/П-2; находка B → FSP/MCA/Э-6).
3. **Замкнуть и оживить** (MetaHead→BG/TRN, closure-verifier, ACL/EWC/BennaFusi/гиппокамп — код уже написан).

Genuinely-absent (TDR-декодер, NSM, DCC, ADMC-цель, TCC-тор) — достраивать **после** того, как живое начало учиться и говорить. Горизонт v3.1 — не трогать, кроме PCI-прокси.

**Главный принцип蓝图:** Alexandria даёт *что* должно быть; Ziwago уже содержит *большую часть* этого в мёртвом/опт-ин виде. Ценность рефакторинга — в **проводке и оживлении**, а не в новом строительстве. Это согласуется с диагнозом проекта: «проблема — проводка» (INNOVATION_PROPOSALS §0).