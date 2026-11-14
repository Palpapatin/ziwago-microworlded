# Трек C: сознание, таламус, язык и метрики — итоговый отчёт

Что это: отчёт о правках восьми модулей «сознания/таламуса/языка/метрик» — что изменено, чем доказано, где остались врезки вне зоны. Кому: владельцам `organism/integrated.py`, `organism/flat.py`, `action/basal_ganglia.py` и принимающему трек.

**Область и границы.** Правились только 8 файлов из `ziwago/` (§1). Спецификация `tests/test_v22_consciousness_thalamus.py` (19 тестов) не менялась. Полный `pytest tests\` не запускался (запрещено заданием), живой прогон организма и загрузка чекпоинтов не выполнялись. Параллельно в дереве работал трек B (`ziwago/memory/*.py`, `tests/test_v22_memory_honesty.py`) — эти файлы не наши и ниже не фигурируют.

**Версия.** 1.0 от 14.11.2026 — первый выпуск; имя файла дано по дате трека (13.11.2026), содержимое фиксирует состояние на 14.11.2026. Окружение прогонов: pytest 9.1.1, Python 3.13.15.

## 1. Что изменено (file:line до/после) и почему

Объём правок по `diff_all.md` (эталон 13.11 ~23:5x → текущее): `meta_head.py` 121 строка (+107/−14), `workspace.py` 58 (+42/−16), `metrics.py` 134 (+116/−18), `relay.py` 97 (+88/−9), `trn.py` 31 (+27/−4), `son_reflection.py` 54 (+53/−1), `syntax.py` 169 (+150/−19), `attention_schema.py` 0 (+0/−0).

### 1.1 `ziwago/consciousness/attention_schema.py` — 0 строк в диффе

**Было.** `gain = softmax(logits) · alpha_ast` — формула зафиксирована комментарием теста (`tests/test_v22_consciousness_thalamus.py:83`). При `alpha_ast=0.1` все модули получали ≈0.02, и TRN видел константу `0.5 + 0.02` (`trn.py:52`), а не профиль внимания.

**Стало.** Файл был частично исправлен предыдущим, прерванным проходом ДО старта трека (mtime 13.11 23:22), поэтому текущая версия принята как база и в дифф не попала. Действующая формула — `attention_schema.py:73-86`: логиты нормируются по размаху и умножаются на `competition_margin`, softmax даёт резкий профиль, `rel = p / p_max`, итог `floor + (1−floor)·rel` при `floor = alpha_ast`.

**Почему.** Победитель должен доходить до TRN как 1.0, а проигравшие — как доля от него, иначе «приоритет внимания» — константа. Краснота старой формулы доказана не прогоном, а реконструкцией (`old_ast_evidence.py`): `max(gain) = 0.02638` при требовании `>= 0.9`, `max−min = 0.0080` при требовании `> 0.5`; текущая даёт `max = 1.0`, `min = 0.1448`.

### 1.2 `ziwago/consciousness/meta_head.py` — гейты, новизна, честное имя метрики

**Было.** `meta_head.py:125-132`: `m_ratio` — это `corr(conf, 1−err)` по последним 64 пробам, а в докстринге модуля (`:13`) названо «M-ratio = meta-d'/d' — прокси метакогнитивной чувствительности». `:135-140`: гейты `bg_threshold`, `beta_gw_boost`, `replay_priority`, `trn_gain` считались и складывались в `MetaOut.gates`, но во всём пакете их не читал никто; `integrated.py:282` брал только `meta.confidence`. `:117-118`: третий выход `out[2]` (новизна) обучался, но в `MetaOut` не попадал — мёртвый вес. `apply_gates` отсутствовал.

**Стало.** `meta_head.py:68-82` — `GateApplication` (`beta_gw`, `replay_priority`, `trn_gain: torch.Tensor`, `bg_threshold`) с докстрингом, куда идёт каждое поле. `:214-238` — публичный контракт `apply_gates(out, trn_gain, beta_gw, replay_priority)`: аддитивные гейты складываются с базой потребителя, мультипликативный `trn_gain` умножает тензор той же формы. `:59`, `:148` — поле `novelty_unit`; `:204` — `replay_priority = epistemic + 0.5·surprise + 0.5·novelty_unit`. `:63`, `:198-200` — `calibration_corr` вместо `m_ratio` (поле `m_ratio` удалено).

**Почему.** Гейт, который считается и никуда не доходит, — не гейт; теперь есть один публичный контракт переноса без правки чужих файлов. Обучаемый выход обязан менять поведение (приоритет реплея), а не только матрицу весов. Корреляция калибровки — не meta-d'/d' (для meta-d' нужна 2AFC-декомпозиция и ROC), поэтому переименована по смыслу.

### 1.3 `ziwago/consciousness/meta_head.py` — калибровка уверенности (добивка пункта 2)

**Было (замерено).** `corr(conf, 1−err) = +0.2937` при пороге теста 0.3, `ECE = 0.2917`; `conf ∈ [0.0050, 0.9952]` при целях 0.9/0.1; логит `−5.30…+5.34` при нужных ±2.2; `‖w_out[conf]‖ = 2.06` (клип ±3 почти достигнут). Причина: след накапливал САМУ ошибку (`elig ← 0.75·elig + local_err⊗h`) — это НЧ-фильтр цели с τ≈4 пробы, вес уезжал в клип уже после достижения цели, а на смене блока 3–4 пробы из 8 уходили на разряд. Вторая причина: преактивация `h·w` росла вместе с `‖h‖`, и гейн уверенности определялся нормой активности, а не свидетельствами.

**Стало.** `meta_head.py:115-125` — `_readout` делит преактивацию на `max(1, ‖h‖)` (делительный гомеостаз) и зажимает логит в `_LOGIT_MAX = 3.0` (`:113`), то есть `p ∈ [0.047, 0.953]` по построению. `:176-187` — трёхфакторное правило: `Δw = lr·outer(local_err, trace_pre)`, след `trace_pre` (`:104`) живёт на пресинаптической активности, ошибка берётся текущая. `self.elig` удалён, внешних читателей `.elig` в дереве нет.

**Почему и что вышло.** Ошибка больше не накапливается: при `out = target` апдейт останавливается сам (фиксированная точка вместо клипа). Тот же сценарий, 48 проб блоками по 8: `corr 0.2937 → 0.5948`, `ECE 0.2917 → 0.1197` (−59 %), `conf 0.137…0.864`. Устойчивость на 8 парах сидов: `corr 0.5096…0.6082`, `ECE 0.0790…0.1365`. Остаток объясним и не подгонялся: `corr` с лагом 1 равен 0.8779 — это ровно одна проба, физический минимум, потому что уверенность читается ДО апдейта на этой же пробе; в метрику лаг не входит, окно не подбиралось, тест не ослаблялся.

### 1.4 `ziwago/consciousness/workspace.py` — NE-порог, Lyapunov-невязка, согласованный ignition

**Было.** `workspace.py:87`: `s_th_eff = c.s_th * 0.75` — от NE не зависит, хотя докстринг (`:6`) обещал сдвиг порога. `:10`: «E(ξ) = −½ξᵀXᵀXξ — Lyapunov-энергия (монотонно ↓)», а на реализованном шаге энергия не сравнивалась с предыдущей и хранилась только последняя (`:116`). `:89-91`: слоты писались по сглаженному `a`, а `ignition` и `winner_zone` считались по несглаженному `a_raw`.

**Стало.** `:97` — `s_th_eff = c.s_th * (0.75 − 0.25·ne)`. `:116-140` — энергия объявлена невязкой `‖ξ − h(ξ)‖²` (`h = X·softmax(βXᵀξ)`), шаг принимается только при `e_new <= trace[-1] + 1e-12`, иначе итерации останавливаются на предыдущем ξ; `energy = Σ` финальных невязок, `energy_hist` пишется без обрезки (`:140`, включая вызов без кандидатов `:65`). `:101-102` — `ignition` и `winner_zone` считаются по тем же `slot_acts`, которыми пишутся слоты. В `GWOut` добавлены `slot_activations` и `slot_energy_traces` (`:37-38`).

**Почему именно невязка.** Квадратичная форма на реализованном шаге `0.75·ξ + 0.25·h(ξ)` строго возрастает: измерено `dE/dt|₀ = +3.932e-4 / +2.960e-4 / +3.881e-4` (кандидаты теста, ne=0.5) и `ΔE > 0` вдоль траектории; Ramsauer-энергия даёт `+1.03e-3` за шаг, чистый шаг `ξ←X·softmax` — `+4.13e-3`. Единственная измеренно убывающая функция — невязка неподвижной точки: монотонна на всех 10 слотах в 4 конфигурациях, например `3.3e-7 → 1.9e-7 → 1.1e-7 → 6e-8`. Поэтому объявлена энергия, а не изменён шаг: динамика `content`/`broadcast` осталась прежней. NE-сдвиг даёт `Δignition = 0.025605` (0.950273 → 0.975878) на кандидатах теста и 0.961108 → 0.981244 на живом контуре (d=128).

### 1.5 `ziwago/thalamus/relay.py` — граница фразы, веретено, телеметрия

**Было.** `relay.py:33` — `self.gate = torch.ones(n)` плоским атрибутом. `:36` — параметр `boundary_burst` в `step()`, который живой вызов `integrated.py:228` не передаёт, поэтому `b_drive` (`:49`) всегда 0 и обещанный фазовый маркер не наступал никогда. `:69-75` — `spindle()` считал осцилляцию и возвращал её, не трогая гейт (мёртвый код вне тестов). Телеметрии не было вовсе.

**Стало.** `:76` — `mark_phrase_boundary()` ставит одноразовый `_boundary_pending`; `:92-93` — `boundary = boundary_burst or _boundary_pending`, флаг гасится в том же `step` (маркер не залипает). `:62-73` — `gate` стал свойством: геттер отдаёт `_gate × фаза веретена`, сеттер пишет только базу, поэтому внешняя запись `integrated.py:227` не теряется и не накапливается. `:137-156` — `spindle()` ставит фазу множителем на один шаг и пишет `telemetry["spindle"]`. `:30-38`, `:52-60`, `:122-133` — `_TELEMETRY_DEFAULTS`, самовосстанавливающийся `_telemetry()`, поля `boundary_burst`, `b_drive`, `boundary_markers`, `mode`, `h_t_mean`, честный `burst_cause ∈ {boundary, vigilance, silence, none}` и `spindle_applied`.

**Почему.** Дефект был не в формуле, а в том, что единственный живой вызов не мог включить burst; решение о границе перенесено владельцу релея, сигнатура `step` не изменилась. Фаза одноразовая намеренно: если бы произведение оставалось в гейте, серия вызовов за сон умножила бы гейт до нуля — «тихий режектор». Терпимость к старому снимку нужна потому, что `load_runtime_tree` (`moe_cortex.py:181`) удаляет атрибуты, которых не было в чекпоинте.

### 1.6 `ziwago/thalamus/trn.py` — произведение вместо суммы

**Было.** `trn.py:9` — докстринг обещал `g_k = g_L6 · g_AST · (1 − lateral_inh)`, код этого не делал. `:52` — `x = x * (0.5 + ast_priority.clamp(0,1))`: константа 0.5 плюс зависимость от абсолютного масштаба приоритета, и AST дополнительно разбавлялся EMA (`:53`). `:62` — `self.gate = (self.act + 0.5*x_raw − lateral).clamp(0,1)`: аддитивная смесь, где униполярная засветка L6 складывается со всеми секторами одинаково.

**Стало.** `:9-23` — докстринг с явной формулой; `:67` — `act` = EMA 0.85/0.15 от чистого high-pass L6 (AST из драйва убран, иначе приоритет учитывался бы дважды); `:76-83` — `g_AST,k = clamp(ast_k,0,∞) / max_j clamp(ast_j,0,∞)` (при `peak ≤ 1e-8` — единицы); `:85` — `gate = (act · g_ast · (1 − lateral).clamp(0,1)).clamp(0,1)`. Сохранены high-pass, `w_lat`, электротоника, сигнатура `step(l6_drive, ast_priority=None, vigilance=0.5)`.

**Почему.** Произведение делает гейт конкурентным, нормировка на максимум снимает зависимость от уровня приоритета (важна форма профиля). Удаление `0.5·x_raw` — прямое следствие замера: аддитивный член сжимал разрыв 6.9:1 до 1.15:1. После правки профиль AST даёт `[0.1448, 0.1448, 1.0000, 0.1448, 0.1448]`, отношение 6.906; `max|gate_a − gate_b| = 0.3257` при пороге 0.05; удвоение приоритета меняет гейт на 0.0 (`atol 1e-6`); униформный приоритет 0.5 даёт ровно тот же гейт, что `None`.

### 1.7 `ziwago/cortex/son_reflection.py` — реальный индекс VTE и D1 Go/No-Go

**Было.** `son_reflection.py:250` — `selected_vte_trajectory_idx=0` литералом, хотя `evaluate_vte_inner_speech_trajectories` честно считал `best_idx` (argmin `F_SON`) и никуда его не сохранял. `grep "D1"` по эталону давал ровно одно совпадение — модульный докстринг (строка 28): гейт был обещан в тексте и отсутствовал в коде.

**Стало.** `:125-127` — состояние `last_vte_best_idx`, `last_vte_logs`, `last_vte_d1_go_gate` (0 = «симуляции ещё не было»). `:256` — вердикт репортит `int(self.last_vte_best_idx)`. `:298` — веса минимальной модели `w_align_go, w_coh_go, w_conf_go = 2.2, 0.9, 2.0`; `:315` — `go_arg = 2.2·cos_qa + 0.9·min(2.0, syn_coh) − 2.0·conflict_e`, `d1_go_raw = sigmoid(clamp(go_arg, ±12))`; в логи пишутся `d1_go_raw` и `d1_go_gate = 0.0` (No-Go по умолчанию); `:338-341` — гейт открывается только победителю. `:345-355` — `note_vte_selection(idx)` для внешней пересинхронизации.

**Почему.** Индекс выбора — измерение, а не константа: он приходит из того же argmin, что показан в логах, поэтому расхождение видно сразу. Гейт положителен только у победителя, у альтернатив ровно `0.0` — это торможение, а не «мягкое» ослабление; сырое значение сохранено как `d1_go_raw`, чтобы вклад торможения был виден. Замер на тестовом входе: `best_idx 2`, `F` по траекториям `0.5129 / 1.4350 / −0.7000`, `gate = 0.982014` у победителя и `0.0` у остальных; при `conflict+2.0` тот же победитель даёт `0.5`, то есть гейт не константа.

### 1.8 `ziwago/cognition/language/syntax.py` — мост BA44 учится на реальном переходе

**Было.** `syntax.py:91-92` — `w_ba44_bridge` писался один раз при инициализации и только на вырожденном пробнике `probe_192 = normalize([t_vec, t_vec, t_vec])` (один и тот же вектор союза во всех трёх третях); пар «полюс1→полюс2» матрица не видела, слов `learn`/`update` в файле не было. `:201` — `ba44_current = bridge_in @ w[:, idx] + tonic_bias + 0.85·trig_active`: константы (`tonic_bias = 0.25`, `0.85` за триггер) сильнее косинуса, гуляющего в [−1, 1]. `:190` — третья треть входа на выборе была вектором самого союза, а не стимула.

**Стало.** `:55-58` — параметры отчёта `BRIDGE_LR = 0.35`, `TRIGGER_BONUS = 0.10`, `TONIC_BIAS = 0.03`, `CONNECTOR_AND_IDX = 4`; `:82-84` — телеметрия `bridge_learn_events`, `bridge_last_cos`; `:196-211` — единый конструктор входа `_bridge_input` с по-третными L2-нормами; `:213-231` — `_stimulus_context_vec` (средний вектор стимула); `:234-273` — `learn_bridge_transition`, хеббовский шаг с ренормировкой столбца `W[:,k] ← normalize((1−lr)·W[:,k] + lr·x)`; `:276-316` — `score_bridges` с полями `matrix_drive` (`0.5(cos+1)`), `matrix_cos`, `trigger_active`, `tonic_bias`, `total_drive`; `:318-338` — выбор по argmax `total_drive` при прежних сигнатуре и возврате.

**Почему.** Матрица получила владельца события. Один конструктор входа обязателен: первый замер дал расхождение 0.9913 при научении против 0.7228 при выборе, потому что сырой `get_vec64` (норма ≈8) забирал 87 % нормы 192-вектора в обучении и лишь треть при выборе. После нормировки по третям `cos_after` последнего шага 0.9926 и `matrix_cos` при выборе 0.9926 совпадают. Драйв переставлен честно: матрица — главный член (0…1), триггер и приор стали поправками. Замер после 6 вызовов: `cos_after` 0.5303 → 0.9926, `matrix_drive` целевого столбца 0.9963 против 0.4217…0.5273 у остальных, нормы столбцов 1.0, `select → (', after', [11, 103], 0.9963)`; контроль `lr=0.05` за 6 вызовов даёт 0.3791 — метрика зависит от шага.

### 1.9 `ziwago/consciousness/metrics.py` — строгие метрики вместо имитации

**Было.** `metrics.py:24-29` — `lempel_ziv_complexity = len(zlib.compress(raw, 9)) / len(raw)`: точечного возмущения нет вовсе, хотя докстринг (`:6-7`) обещал «сложность после точечного возмущения (Casali 2013)»; на пустом входе возвращался `0.0`. `:115-130` — `report(self, ffr, rdi=0.0)` возвращал аргументы вызывающего как «измеренные»; при пустой истории подставлялся `torch.zeros(1,1)`, и `pci_lz = 9.0` (вне шкалы 0..1). О несовместимости с живым `next16_core.ConsciousnessMetricsDashboard` в докстринге не было ни слова.

**Стало.** `:1-40` — докстринг «это НЕ дубль живого дашборда» со ссылкой на `next16_core.py:846` и точными формулами обоих расхождений (`broadcast_reach` — там `(raster>0).any(dim=0).sum()/n_cols`, `:877`; `ignition` — там `(raster>0).sum(dim=1).max()/n_cols`, `:876`). `:41-42` — `_PCI_CORE_BINS = 2`, `_PCI_MAX_BINS = 4096`; `:45-74` — приведение потока к матрице и `_evoked_response` (инверсия ядра 2 такта × 3 канала в точке первого фронта, конус `2 + lag//8` каналов, AND с субстратом); `:77-87` — `_lz76_phrases`; `:90-118` — `lempel_ziv_complexity -> Optional[float]` с нормировкой `min(1, c/(n/log2 n))` и `None` на пустом входе; `:204-228` — `report()` без аргументов, при `n_samples == 0` все шесть метрик `None`, ключи `ffr`/`rdi` удалены.

**Почему.** Эхо аргументов — не измерение (вызовов `report(...)` с аргументами в дереве нет, единственный потребитель — тест). Возмущение теперь применяется физически, шкала нормирована: замеры `0.106695` (тест-вектор) против наивного `0.03125`, `1.0` на случайном потоке, `0.141321` на периодике, `0.052734` на нулях, `0.734647` на растре 200×8, `None` на пустом входе; повторный вызов даёт то же число (ГПСЧ в модуле нет), один изменённый бит — `0.088618`.

## 2. Сводная таблица: пункт → было → стало → тест → красный до правки

| № | Пункт | Было | Стало | Тест | Красный до правки |
|---|---|---|---|---|---|
| 1 | `attention_schema.py` | `gain = softmax(logits)·alpha_ast` ≈0.02 на модуль | профиль `floor + (1−floor)·rel`, победитель 1.0 | `test_ast_gain_is_competitive_priority_not_flat_softmax` | **нет** — в baseline единственный зелёный; старая формула ловится реконструкцией: `assert float(gain.max()) >= 0.9` → 0.02638 |
| 2 | `meta_head.py` (гейты) | гейты считаются, `gates` не читает никто; `novelty_unit` не выходит из головы | `apply_gates` → `GateApplication`; `novelty_unit` в `replay_priority` | `test_meta_head_exposes_consumable_gate_contract` | **да** — `AttributeError: 'MetaHead' object has no attribute 'apply_gates'` |
| 3 | `meta_head.py` (калибровка) | `corr = 0.2937`, `ECE = 0.2917`, conf 0.005…0.995 | `corr = 0.5948`, `ECE = 0.1197`, conf 0.137…0.864 | `test_meta_head_calibration_metric_is_named_by_its_real_meaning` | **да** — `assert 0.29369770140864826 > 0.3` |
| 4 | `workspace.py` | `s_th_eff = s_th·0.75`; квадратичная энергия не монотонна; `ignition` по `a_raw` | `s_th·(0.75−0.25·ne)`; энергия — невязка `‖ξ−h(ξ)‖²`; `slot_activations` | `test_workspace_threshold_depends_on_norepinephrine`, `..._energy_is_monotone_and_full_history_is_kept`, `..._ignition_matches_slot_activations` | **да** — `assert 0.9502728581428528 > (0.9502728581428528 + 0.01)`; `AssertionError: нет следов энергии по шагам` |
| 5 | `relay.py` | `boundary_burst` не передаётся; `spindle()` не влияет на гейт; нет телеметрии | `mark_phrase_boundary()`; `gate` = свойство; `telemetry` | `test_fo_relay_has_explicit_phrase_boundary_and_honest_telemetry`, `test_fo_relay_spindle_is_not_dead_code` | **да** — `AttributeError: 'ThalamusFO' object has no attribute 'telemetry'`; `spindle() должен влиять на гейт` |
| 6 | `trn.py` | `(act + 0.5·x_raw − lateral)`; `0.5 + ast` | `act · g_AST · (1−lateral)` | `test_trn_gate_competes_by_ast_priority_and_differs_between_profiles`, `..._is_scale_invariant_to_global_ast_priority_scale` | **да** — `assert 0.5990042686462402 > (2.0 * 0.521517276763916)`; `assert torch.allclose(gate_a, gate_b, atol=1e-6)` → False |
| 7 | `son_reflection.py` | `selected_vte_trajectory_idx=0`; D1 Go только в докстринге | `last_vte_best_idx`; `d1_go_raw`/`d1_go_gate`; `note_vte_selection` | `test_son_reports_real_selected_vte_trajectory_index`, `test_son_has_minimal_dopamine_d1_go_gate` | **да** — `assert 0 == 2`; `assert 'd1_go_gate' in {...}` → отсутствует |
| 8 | `syntax.py` | матрица инициализирована `[t,t,t]`; `tonic 0.25` + `0.85·trig` | `learn_bridge_transition`, `score_bridges`, `TRIGGER_BONUS 0.10`, `TONIC_BIAS 0.03` | `test_ba44_bridge_learns_on_real_transition_pairs_and_drives_selection` | **да** — `AttributeError: 'BA44AttractorSyntaxBridge' object has no attribute 'learn_bridge_transition'` |
| 9 | `metrics.py` | LZ = zlib/n без возмущения; `report(ffr, rdi)`; пустая история → 9.0 | LZ76 отклика, `None` на пустом, `report()` без аргументов | `test_metrics_report_has_no_echo_of_caller_inputs`, `..._empty_history_reports_no_data_not_nine`, `..._lz_complexity_uses_honest_perturbation`, `..._module_warns_about_incompatibility_with_live_dashboard` | **да** — `TypeError: ... report() missing 1 required positional argument: 'ffr'`; `assert 0.03125 != 0.03125` |

**Про пункт 1 отдельно.** Единственный пункт, который трек не делал: файл был исправлен предыдущим прерванным проходом (13.11 23:22), в диффе 0 строк, текущая версия принята как база. Поэтому «падал ли» — **нет**: в baseline-прогоне этот тест зелёный (`18 failed, 1 passed`, зелёный ровно он). Утверждение «старая формула была красной» опирается на реконструкцию, а не на прогон, и это единственное место отчёта, где краснота доказана косвенно.

## 3. Точки врезки вне зоны

Все врезки ниже — не сделаны: это чужие файлы. Приведены строка и код на момент отчёта.

### 3.1 `ziwago/organism/integrated.py`

* `:116` — `self.ast_priority = torch.full((n_input,), 0.5)`: дефолт униформен, поэтому `g_AST ≡ 1` и поведение TRN до первого `ast.step` идентично `None`.
* `:225-226` — `gate = self.trn.step((x.abs() + self.l6_feedback.abs()).clamp(0, 1), self.ast_priority, vigilance=bundle["ach"] + bundle["ne"])`: вызов уже совместим с новой формулой, менять не нужно. Для телеметрии конкуренции писать `float(self.trn.gate.max())`, `float(self.trn.gate.min())` в отчёт тика.
* `:227` — `self.fo.gate = gate`: попадает в сеттер релея, база сохраняется; строка работает как раньше.
* `:228` — `relay = self.fo.step(x.unsqueeze(0), ach=bundle["ach"], ne=bundle["ne"])[0]`: `boundary_burst` не передаётся. Перед этой строкой нужен `self.fo.mark_phrase_boundary()`, когда языковой контур закрыл клаузу (решение о границе — у языка, не у релея).
* `:237` — `self.last_relay_norm = float(relay.norm())`: рядом добавить `self.last_relay_telemetry = dict(self.fo.telemetry)` и писать поле в JSON-отчёт.
* `:276` — `gw = son.gw.compete([resample(self.l5_feedback, self.n_input), x], ne=bundle["ne"])`: `beta_boost` не передаётся. Подключать как `beta_boost=app.beta_gw - <база>`; база живёт в конфиге рабочего пространства (`son.gw.cfg.beta_gw`), сама формула — `workspace.py:95`. Ограничение порядка: `compete` (`:276`) идёт раньше `meta.step` (`:279`), поэтому в том же тике `app` ещё не существует — либо переносить расчёт гейтов выше `:276`, либо брать `app` предыдущего тика (какой вариант верен для живого контура — не проверено).
* `:279` — `meta = son.meta_head.step(self.broadcast, gw.content, self.previous_error, learn=plastic)`: сразу после — `app = son.meta_head.apply_gates(meta, trn_gain=<гейны TRN [n]>, beta_gw=<база>, replay_priority=0.0)`; `app` положить в `son.organism_reflection` (`:287`, сейчас там `{"gw": gw, "meta": meta}`).
* `:286` — `self.ast_priority = resample(ast.gain, self.n_input).clamp(0, 1)`: `gain` уже в [0,1], clamp ничего не режет — путь AST → TRN замкнут без правок.
* `:303` — `"gw_ignition": gw.ignition, "confidence": self.confidence`: рядом просится `"gw_energy": gw.energy` (теперь невязка ≥ 0, знак изменился).
* `:529` — `self.replay.append({"cue": ..., "priority": float(loss) + 1.0 - self.confidence, ...})`: добавить `+ app.replay_priority`; потребитель приоритета — `:215`, `max(range(len(self.replay)), key=lambda i: self.replay[i]["priority"])`. `record_credit` вызывается не из тика, поэтому `app` нужно хранить на организме (`self.last_meta_gates`) — иначе добавки не будет.

### 3.2 `ziwago/organism/flat.py`

* `:2855` — `bridge_phrase, bridge_ids, _ = self.ba44_syntax_bridge.select_bridge_tokens_between_attractors(...)`: рядом (после `:2862` `bridge_two_attractor_clauses_in_ba44`) вызывать `learn_bridge_transition(ca3_clause_poles[0], ca3_clause_poles[1], self.compute_stimulus_vec64(u_ids), connector_idx=<argmax score_bridges>)` (`compute_stimulus_vec64` — `flat.py:775`). Индекс выбранного моста `select_...` не возвращает, сигнатуру менять нельзя — брать из `score_bridges` в той же точке.
* `:3171` — `"ba44_syntax_bridge_connector": ba44_bridge_used`: туда же по правилу «каждое число с источником» просится `total_drive`/`matrix_drive` выбранного моста.
* `:3236` — `"w_ba44_bridge": saved_core["ba44_syntax_bridge"]["attributes"]["w_ba44_bridge"]`: в payload нет `bridge_learn_events`/`bridge_last_cos`; восстановление — `:3411-3412` (`self.ba44_syntax_bridge.w_ba44_bridge = payload["w_ba44_bridge"].float().clone()`). Без этого счётчики научения теряются на чекпоинте.
* `:2945` — `best_idx, vte_logs = self.son_column.evaluate_vte_inner_speech_trajectories(...)`; `:2958-2961` — пересчёт `best_idx = min(range(len(vte_logs)), key=lambda idx: (float(vte_logs[idx]["combined_selection_score"]), idx))`. В этом пути вердикт сообщит argmin-F, а не фактически выбранный план; сразу после `:2961` нужен `self.son_column.note_vte_selection(best_idx)` (метод готов, `son_reflection.py:345`).

### 3.3 `ziwago/action/basal_ganglia.py`

* `:36` — `class BasalGanglia:` с докстрингом «Go/NoGo DDM + STN + TD-критик»: место, где `d1_go_gate` должен складываться с дофаминовым входом выбора действия (стриатум → GPi/SNr → таламус). Сейчас гейт живёт только в логах VTE. Конкретной строки врезки назвать нельзя: сигнатура BG не читает `vte_logs`, и это **не проверено** — нужен владелец файла.

### 3.4 Прочие точки

* `ziwago/organism_core.py:554` — `def sleep_cycle(...)`, тик внутри — `:573-574` (`for _ in range(budget): life.tick(core, torch.zeros(core.moe_cortex.n_in))`). Здесь перед тиком релея нужно `self.fo.spindle(t_ms, 10.0)`; точная строка внутри `life.tick` — у владельца модуля `life`. Заметка W4 ссылается на `train_ziwago_v19_verified_runtime.py::sleep_cycle`, но этого файла в текущем дереве нет (сон вызывается как `runtime.sleep_cycle(...)`, `run_ziwago_v19.py:342`) — ссылку на старый файл считать устаревшей.
* `ziwago/cortex/son_reflection.py:255` — `trn_column_gain_modulation=trn_gains`: сюда идёт `trn_gains * app.trn_gain` (тензор той же формы); `MetaHead` вызывается в `:185`, `Beta`-путь не менялся.
* `ziwago/consciousness/metrics.py` — внешних вызовов нет, в карте модулей числится PHANTOM (`MODULE_MAP_2026-11-13.md:143`). Для живого контура использовать `next16_core.ConsciousnessMetricsDashboard` (строка 846), а не этот класс: определения `broadcast_reach` и `ignition` несовместимы, числа сравнивать нельзя.

## 4. Команды проверки и вывод

Команды ниже выполнялись из `C:\Users\yarma\Desktop\Ziwago18` на Python 3.13.15 / pytest 9.1.1; выводы приведены так, как зафиксированы в прогонах трека.

### 4.1 Базовая линия ДО правок

```
python -m pytest tests\test_v22_consciousness_thalamus.py -q -p no:cacheprovider
18 failed, 1 passed, 2 warnings in 2.40s
```

Зелёный ровно один: `test_ast_gain_is_competitive_priority_not_flat_softmax`. Полные тексты падений — `baseline_tests.txt` (цитаты assert'ов выше взяты оттуда; для пункта 8 в файле только summary-строка `FAILED ...::test_ba44_bridge_learns_on_real_transition_pairs_and_drives_selection`, падение воспроизведено отдельным прогоном до правки с `-k "ba44_bridge_learns"`).

### 4.2 Итоговый прогон

```
python -m pytest tests\test_v22_consciousness_thalamus.py -q -p no:cacheprovider
19 passed, 2 warnings in 2.53s
```

Повторный прогон — `19 passed`. Два warning'а предсуществующие: `SyntaxWarning` на неизменённых строках докстрингов (`\vec`, `\mathcal`); baseline давал тот же `\mathcal`-warning.

Подпрогоны по зонам (фактические выводы из заметок): `-k "trn_gate or ast_gain_is_competitive"` → `3 passed, 16 deselected`; `-k "workspace"` → `3 passed`; `-k "test_fo_relay"` → `2 passed`; `-k "meta_head"` → `4 passed` (было `1 failed, 3 passed`); `-k "son_reports_real_selected_vte_trajectory_index or son_has_minimal_dopamine_d1_go_gate"` → `2 passed`; `-k "ba44_bridge_learns"` → `1 passed`; `-k "metrics_report_has_no_echo or metrics_empty_history or metrics_lz_complexity or metrics_module_warns"` → `4 passed`.

### 4.3 Проверка импортов

Зафиксированный результат проверки (скрипт трека, `import_check.py`):

```
python C:\Users\yarma\AppData\Local\Temp\track_c\import_check.py
OK   ziwago.consciousness.meta_head
OK   ziwago.consciousness.workspace
OK   ziwago.consciousness.metrics
OK   ziwago.thalamus.relay
OK   ziwago.thalamus.trn
OK   ziwago.cortex.son_reflection
OK   ziwago.cognition.language.syntax
OK   ziwago.organism.integrated
FAIL ziwago.organism.flat ModuleNotFoundError No module named 'pyarrow'
```

`ziwago.organism.flat` не импортируется в этой среде — **предсуществующий дефект окружения**, не следствие правок: такое же падение зафиксировано в `REPORT_TRACK_B_MEMORY_HONESTY_2026-11-13.md:212`. Модуль `flat.py` в этом треке не менялся.

### 4.4 Измерения, которыми доказаны числа

```
python C:\Users\yarma\AppData\Local\Temp\track_c\old_ast_evidence.py   # старая формула AST (max 0.02638)
python C:\Users\yarma\AppData\Local\Temp\track_c\w7_verify.py            # LZ на 5 потоках
python C:\Users\yarma\AppData\Local\Temp\track_c\probe_meta3.py           # новизна: обученный/необученный юнит
python C:\Users\yarma\AppData\Local\Temp\track_c\probe_ws2.py … probe_ws5.py  # dE/dt и невязка Хопфилда
```

Замер калибровки на 8 парах сидов (`corr 0.5096…0.6082`, `ECE 0.0790…0.1365`) выполнен
одноразовым скриптом в консоли — файл не сохранён; сами числа зафиксированы в
`notes\W2b_calibration.md` (§«ECE ДО → ПОСЛЕ», «Устойчивость»), там же — таблица
`conf`/логит/`‖w_out‖` до и после. Повторяемость подтверждается тем, что сценарий
целиком воспроизводится тестом `-k "meta_head"` (4 passed).

Реконструкция старой формулы AST (результат прогона скрипта): `max(gain) = 0.02638` при требовании теста `>= 0.9`, `max−min = 0.0080` при требовании `> 0.5`; новая версия — `max = 1.0`, `min = 0.1448`. Диагностика Хопфилда (`dE/dt|₀ > 0`) и невязки — скрипты `probe_ws2.py`…`probe_ws5.py` в том же каталоге.

### 4.5 Что не запускалось

* Полный `pytest tests\` — запрещён заданием; проверен только файл спецификации.
* Живой прогон организма, загрузка чекпоинтов, обучение — запрещены; влияние правок на логиты и текст не измерялось.
* Регрессия чужих тестов (например, `tests/test_v19_spiking_language.py`, где тоже создаётся `BA44AttractorSyntaxBridge`) не запускалась — только чтение.
* Импорт `ziwago.organism.flat` не проходит по причине окружения (§4.3), поэтому врезки в §3.2 проверены чтением кода, а не исполнением.
* Влияние новых гейтов на живой контур не измерялось: `trn_gain = 1 − 0.5·conf` теперь меняется в диапазоне ≈0.95…0.53 (было 1.0…0.50) — сдвиг небольшой, но на реальных `mean_cort_err` не замерялся.
* Веса D1 Go (`2.2 / 0.9 / 2.0`) — ручная минимальная модель: не обучались и не калибровались, поведение при большом числе траекторий не измерялось.
* Старый чекпоинт BA44 (столбцы, выровненные на `[t, t, t]`) с новой формулой входа ожидаемо даёт `matrix_drive ≈ 0.5` до первого научения — на чекпоинте не проверено.
* Стоимость не профилировалась: LZ76 квадратичен по длине окна (бенчмарка на предельных 4096 бинах нет), добавленный `norm()` в `_readout` на профиле не снимался.

### 4.6 Отрицательные результаты, которые не спрятаны

* Пункт 1 трек не делал: 0 строк в диффе, краснота доказана реконструкцией, а не прогоном.
* Калибровка не «подкручена порогом»: первый честный вариант (переименование + гейты) оставлял `corr = 0.2937`, поэтому переделано само правило обучения, а тест не ослаблялся и окно не подбиралось.
* Квадратичная энергия Хопфилда Lyapunov-функцией не является: измерено её возрастание на реализованном шаге, поэтому объявлена невязка, а шаг оставлен прежним.
* «Контрастная» (гетеросинаптическая) часть правила BA44 не реализована: обновляется только целевой столбец, соседние не подавляются — для требования теста достаточно, но в строгом смысле контрастности нет.
* В `metrics.py` `causal_density`, `ignition_gamma`, `ece` по-прежнему возвращают `0.0` при недостатке данных — тот же анти-паттерн «константа вместо измерения», но это вне пункта 9 и вне диффа.
* В карте модулей `AttentionSchema` числится PHANTOM (`MODULE_MAP_2026-11-13.md:143`), при этом `integrated.py:283` вызывает `son.ast.step(...)`. Расхождение карты и кода этим треком не проверялось.
