# Трек B — честная память (v22), отчёт 2026-11-13

Зона: `ziwago/memory/{hippocampus,dual_hippocampus,laplace,working,phase_vsa,tcc}.py`
+ новый `tests/test_v22_memory_honesty.py` (29 тестов). Другие файлы не тронуты, git не
использовался, обучение и чекпоинты не запускались, полный pytest по каталогу не запускался.

Все восемь заявленных дефектов **подтверждены замером до правки** (не «по чтению кода»).
Отчёт прошёл один круг независимого ревью (`审阅者`, run 88b58bd3): три замечания из
«must-fix» были признаны справедливыми и переделаны — это отмечено в §2.1.

---

## 1. Что изменено

### 1.1 `hippocampus.py`

| Где | Было | Стало | Почему |
|---|---|---|---|
| `ca3_store` :192 | `w += max(eta_ca3*10, 0.08)·ξξᵀ/N` → инкремент 7.8e-4, насыщение \|w\|≈0.011–0.016 | `w += ξξᵀ/N` (одно предъявление = аттрактор единичного масштаба: `Wξ = ξ`); 1-D паттерн поддержан (`p = ca3.mean(0) if dim==2 else ca3`) | при зажиме 0.25·cue рекуррентный член был в 12–16 раз слабее ключа → «достраивание» было копированием знака; 1-D ключ ломался на втором вызове (`outer: Expected 1-D … got 0-D`) |
| `ca3_retrieve` :244 | `tanh(2·W·a + 0.35·W_seq·a + 0.25·cue)`, без метрик | `tanh(2·W_auto·a + 2·W_assoc·cue + 0.35·W_seq·a + 0.10·cue)`; пишет `last_ca3_metrics`; при высоком аденозине больше recall-итераций | ключ стал слабым смещением; вклад каждого канала измеряем |
| `ca3_store_pair` :224 (**новый**) | — | гетероассоциация `w_ca3_assoc`: `ξ_target ξ_cueᵀ/N`, цель дополнительно автоассоциируется | требование «запись пары (A,B) → по признаку A вернуть B»: автоассоциация этого не может по определению |
| `ca3_metrics` :280 (**новый**) | — | `recurrent_drive_rms`, `assoc_drive_rms`, `sequence_drive_rms`, `cue_bias_rms`, `attractor_share`, `w_rms`, `stored_patterns`, `capacity` | п.1(в) «видно по метрике»: на обученных весах `attractor_share = 0.875` |
| `dg_code` :127 | возвращал маску | дополнительно `dg_usage += act.sum(0)` | нужен реальный счётчик применений гранул |
| `neurogenesis` :141 | `torch.randn(...)` без `generator` (глобальный RNG), выбор по возрасту `argsort(dg_age)[:n_new]` | `generator=self._rng`; выбор: сначала `dg_usage == 0`, затем `dg_usage ≤ медианы` (по возрасту внутри группы) | (а) результат не зависит от чужого `torch.manual_seed`; (б) выученные EC→DG-строки не затираются |
| `step` :578 | `adenosine` принимался и игнорировался | `encode_gain = max(0.25, 1−0.75·ado)` входит в `enc_gate`; в `HippoOut` — `adenosine`, `encode_gain`, `ec_gated`; обновляется `_context_ema`; аденозин удлиняет recall-итерации CA3 | ACh/аденозиновая оппозиция (Hasselmo 1999), аргумент перестал быть мёртвым; CA1-компаратор видит другой вход (mismatch 0.712 → 0.613) |
| `swr_replay` :682 | `torch.multinomial(probs, 1)` без генератора | `generator=self._rng` | порядок реплея зависел от чужого RNG, вопреки комментарию о единственном засеянном RNG |
| `sleep_cycle` :749 (**новый**) | — | SWR при `adenosine ≥ 0.55` и `ach ≤ 0.40` и наличии `cortical_write`; REM при `rem=True` и ≥2 эпизодов; в телеметрии — оба гейта открыто/закрыто, причина отказа, число реплеев | врезка сна, которой не было ни в одном цикле |
| `modality_codes` :628 (**новый**) | — | 4 модальности: EC / DG / CA3 / контекст (`_context_ema`) | вместо 4 кодов из одного вектора |
| `store_episode` :656 | `vsa_dht.store_from_tensors(ec,dg,ca3)` (адрес записи ≠ адрес запроса) | `store_composite_episode(modality_codes(ec))` — **канонический** адрес из того же конвейера, что и запрос | иначе шарды 2 и 3 не совпадали при чтении; переданные `dg/ca3` остаются в самом эпизоде, но адрес не зависит от gated-версии |
| SBC-VSA `episode_slot_codes` :913/992 (**новое**) | — | исходные (до bind) коды каждого шарда | без них «целостность реплея» можно мерить только сравнением unbind с самим собой |
| `index_size` :935 (**новый**) | — | число проиндексированных записей (цена полного свипа) | знаменатель для измеренной экономии |
| `retrieve_multiprobe` :1232 | — | `top_shard_hits` — сколько РАЗНЫХ шардов подтвердили эпизод | прямая метрика мультимодальности (было максимум 2 из 4) |
| `two_clock_replay_step` :1276 | — | `episode_id`, `recovered_shard_indices` | нужны для честной проверки целостности |
| `route_contextual_replay` :1314 | `sequence_retrieval_accuracy = 0.96 if ids else 0.0`; `saving = 1 − min(0.45, выдуманный коэффициент)`; параметра запроса не было вообще | добавлен `query_slot_codes` (запрос); `sequence_retrieval_accuracy = None` + `sequence_retrieval_measurement="unavailable:no_ground_truth"`; `replay_unbind_fidelity` (проверка целостности адреса, падает при порче памяти); `silent_period_compute_saving = 1 − inspected_records/index_size` (измеренные просмотренные записи LSH); `replay_inactivation_threshold` стал реальным рычагом (сколько эпизодов разворачивается) | константы в полях «accuracy»/«saving» — брак; и «экономия» из отношения параметров тоже (ревью, §2.1) |
| `store_from_tensors` :1210 | 4-й шард = `ec + 0.5·ca3` | параметр `temporal` (без него шард 3 честно вырождается в EC) | явная модальность вместо смеси |

### 1.2 `dual_hippocampus.py`

| Где | Было | Стало |
|---|---|---|
| `verify` :381 | 4 кода из одного `ec` (шарды 0,2,3 — копии); `mark_retrieved()` не вызывался; PPR искал в обход гейта | `episodic.modality_codes(ec)`; `theta_gate=True` → `retrieval_allowed()` (спад θ + не более 1 раза за цикл), после реколла — `mark_retrieved()`; при закрытом гейте PPR получает `query_slot_codes=None` (второй readout тоже закрыт); в отчёте — `theta_gate{allowed,is_retrieval,retrieved_in_cycle,theta}` |
| `ThetaClock.retrieval_allowed` :205 (**новый**) | — | сам гейт (раньше — мёртвая пара `retrieval_ready`/`mark_retrieved`) |
| `tick` :437 | `verify(ec)` без гейта | `verify(ec, theta_gate=True)` |
| `sleep_cycle` :503 (**новый**) | — | делегирует `episodic.sleep_cycle` |
| `contextual_replay` :517 | без запроса | `query_ec` → `modality_codes` → `query_slot_codes` |
| `report` :530 | — | `theta_gate_blocks` / `theta_recalls` — телеметрия врезанного гейта |

### 1.3 `laplace.py`

* `step` :43 — ленивая инициализация по форме входа (было `TypeError: unsupported operand type(s) for *: 'Tensor' and 'NoneType'`).
* `slow_profile` :55 / `time_cells` :61 — `RuntimeError("LaplaceBank: состояние пуст — вызовите step(x) или reset_state(batch) …")` вместо `TypeError: 'NoneType' object is not subscriptable`.

### 1.4 `working.py`

* `probe` :77 — отклик только по фацилитации: `(facil @ ĉ)·(1 + 0.5·bump)`; k-WTA по `topk` → **не более** k (ничья больше не расширяет выборку), нулевой отклик не считается активностью → пустой зонд даёт 0 (было `bump·0.5` = эхо последней записи).
* `tick` :59 — при `active_hold=True` бамп удерживается нормированным рекуррентным током `(facil @ bump)/max(facil.sum(0))` с утечкой `dt/τ_nmda` (было только пассивное затухание).
* `WMOut.active_count` — читаемое число активных единиц.

### 1.5 `phase_vsa.py`

| Где | Было | Стало |
|---|---|---|
| `theta_phase` :224 | `2π·role/n_slots` (`theta_hz`/`gamma_hz` не использовались) | `2π·((t_ms·theta_hz/1000) mod 1)`; `slot_clock` — θ-γ мультиплексирование; `gamma_index` — γ-окно |
| выбор роли :177 | `trans_prior` + `argmax` по таблице; победитель всегда есть | `_wlc_step`: Лотка–Вольтерра `du_k = u_k(σ_k − Σ_j A_kj u_j)`, `A` асимметрична от `w_trans` (свой преемник тормозится слабо); без стимула победителя НЕТ (`gamma_slot = -1`); **число подшагов = γ/θ** (`_wlc_substeps` :172), т.е. γ-частота задаёт глубину конкуренции внутри θ-слота (pop₃ 0.347 при 40 Гц → 0.173 при 160 Гц) |
| `learn_role` :196 | `w_lex_role @ x_norm` без `.to(device)` → падение на CUDA | `_ensure_device()` переносит `roles/w_angle/w_lex_role/w_trans/role_state/slots/фреймы` на устройство и тип входа |
| `slot_age` :224 | инкрементировался и нигде не читался | скорость распада слота зависит от возраста: `slots *= slot_decay^(dt·(1+age/τ_age))`; возраст отдаётся в `PhaseStoryOut.slot_age` (за 10 тактов тишины старый слот 0.679 против молодого 2.296) |

### 1.6 `tcc.py`

* `recurrent_drive` :157 (**публичный**, бывший `_recurrent_drive`) — среднее поле: DC-мода умножает среднюю активность (постоянное поле → ровно `−w_inh`), первая гармоника — точная проекция (`2/N` по оси → профиль `cos` даёт ровно `w_exc·a`). Раньше первая гармоника фактически умножалось на `w_exc/2`; чтобы калибровка динамики не поехала, дефолт `w_exc` снижен 6.0 → 3.0. Docstring исправлен: это **не** плотная сумма `−w_inh·N²`.
* `step` :178 — `enabled=False` действительно выключает динамику (активность не меняется, `last_report["disabled"]=True`).

---

## 2. Таблица: было → стало → тест → падал ли до правки

| № | Было (измерено) | Стало (измерено) | Тест | Красный до правки |
|---|---|---|---|---|
| 1 | 25% шум → ошибка 10.2%, выход = знак ключа (`dist(out,cue)=0.25`); `recurrent_drive_rms` 0.015 против `cue_bias` 0.25 | `dist(out,A)=0.000`, `dist(out,cue)=0.250`; `recurrent_drive_rms=0.914`, `cue_bias=0.100`, `attractor_share=0.875` | `test_ca3_pattern_completion_restores_stored_pattern`, `test_ca3_recurrent_drive_dominates_cue_bias` | **да** (`0.1016 != 0.0`; ratio 0.06 < 3) |
| 1b | матрицы ассоциации не было | `dist(out,B)=0.000`, `dist(out,A)=0.500`, в т.ч. с 20% шумом ключа | `test_ca3_association_recall_returns_partner_not_cue` | **да** (`AttributeError: ca3_store_pair`) |
| 1c | 1-D ключ: `ca3_store` падал на втором вызове | два 1-D `ca3_store` + два `ca3_store_pair`, затем точное извлечение | `test_ca3_accepts_one_dimensional_patterns_twice` | **да** (`RuntimeError: outer … got 0-D`) |
| 2 | глобальный RNG: результат зависел от `torch.manual_seed`; строка выбиралась по возрасту и затиралась | одинаковый `w_ec_dg` при разных глобальных seed; самая старая, но используемая (usage=1000) строка не тронута | `test_neurogenesis_uses_seeded_generator`, `test_neurogenesis_does_not_erase_used_granule` | **да** (тензоры различны; `torch.equal` False) |
| 2b | счётчика применений не было | `dg_usage.sum() == dg.sum()`, ненулевых ровно столько, сколько активных гранул | `test_dg_usage_counts_real_activation` | **да** (`AttributeError: dg_usage`) |
| 3 | `sequence_retrieval_accuracy = 0.96` (литерал); `silent_period_compute_saving = 0.9116` и **не менялась** при 1 и 20 эпизодах (2/16 = 10/80 = 0.125) | accuracy = `None` + флаг «нет ground truth»; `replay_unbind_fidelity` = 1.0 на чистом round-trip и 0.0 при порче всех ключей; saving = `1 − inspected/index_size`: 0.0 (1 эпизод, 8/8) → 0.625 (3, 9/24) → 0.925 (20, 12/160), без запроса `None` | `test_sequence_retrieval_accuracy_is_not_invented`, `test_replay_unbind_fidelity_drops_when_replay_content_corrupted`, `test_silent_period_saving_is_measured_from_inspected_records` | **да** (0.96 ≠ None; KeyError на `replay_unbind_fidelity`; KeyError на `silent_compute_records_inspected`) |
| 4 | `adenosine` не использовался; `sleep_cycle` не существовал | `encode_gain` 1.0 → 0.325, амплитуда `ec_gated` падает, `mismatch` 0.712 → 0.613; SWR только при ado ≥ 0.55 и ach ≤ 0.40, REM только по запросу и при ≥2 эпизодах | `test_adenosine_gates_encoding_in_step`, `test_sleep_cycle_swr_requires_adenosine_gate`, `test_sleep_cycle_rem_runs_only_with_enough_episodes` | **да** (`AttributeError: encode_gain` / `sleep_cycle`) |
| 5 | 4 кода из одного `ec`; мультимодальный поиск подтверждался 2 шардами из 4; `mark_retrieved` вызывали только тесты; PPR искал в обход гейта (ревью: 9 поисков на 8 токенов) | 4 разные модальности; точный запрос подтверждается 4/4 шардами; за 8 токенов ровно 1 реколл и 2 поиска (оба на разрешённом такте); за 16 — 2 реколла | `test_four_shard_codes_come_from_four_modalities`, `test_exact_query_scores_all_four_shards`, `test_theta_gate_allows_one_recall_per_cycle` | **да** (2 ≠ 4; 8 реколлов за цикл) |
| 6 | `step` → `TypeError`; `slow_profile` → `TypeError` | `step` сам инициализирует состояние; readout без состояния — `RuntimeError("…состояние пуст…")` | `test_laplace_step_initialises_state`, `test_laplace_readout_without_state_raises_clear_error` | **да** (`TypeError`) |
| 7 | пустой зонд → 8 из 8 единиц последней записи; ничья при k=4 → 16 из 16; реальный паттерн из 3 единиц → 32 из 32; бамп за 200 тактов удержания падал до 13.4% | пустой зонд 0; ничья ровно 4; реальный паттерн 3 (≤ k); удержание 100% против 0.003% пассивного | `test_empty_probe_does_not_echo_last_write`, `test_kwta_returns_exactly_k_units_on_a_tie`, `test_kwta_is_at_most_k_on_a_real_pattern`, `test_nmda_hold_sustains_bump_beyond_passive_decay` | **да** (8 ≠ 0; 16 ≠ 4; 32 ≠ 3; 1.07 vs 8) |
| 8a | `theta_hz`/`gamma_hz` ни на что не влияли (фазы при 8 и 32 Гц совпадали) | фаза и γ-индекс следуют за часами; γ/θ меняет глубину конкуренции (pop₃ 0.347 → 0.173) | `test_theta_clock_drives_phase_and_gamma_clock_is_reported` | **да** (`ps == pf`; `role_state` совпадал) |
| 8b | без стимула роль назначалась (role=2), победитель всегда был | `gamma_slot == -1`; слоты не пишутся; при почти равных драйвах проигравший жив (0.347 против 0.756) | `test_silent_input_has_no_role_winner`, `test_role_populations_coexist_before_winner` | **да** (0 ≠ −1; 0.1 < 0.25) |
| 8c | `learn_role` падал на не-float32 входе (прокси CUDA-бага) | работает, веса переезжают на тип/устройство входа | `test_learn_role_accepts_other_dtype` | **да** (`RuntimeError: addmv … Double, Float`) |
| 8d | `slot_age` рос, слоты не старели | старый слот (age=200) за 10 тактов 0.679 против молодого 2.296 | `test_slots_decay_depends_on_age` | **да** (значения совпадали) |
| 8e | ядро при `w_exc=2.5` давало коэффициент 1.25 (вдвое меньше заявленного); DC-мода давала −3.0 (совпадала) | DC ровно −3.0, первая гармоника ровно 2.5 | `test_tcc_kernel_modes_match_declared_weights` | **да** (1.25 ≠ 2.5) |
| 8f | `enabled=False`, а динамика шла (сумма 504) | при `enabled=False` активность не меняется (0.0) | `test_tcc_disabled_gate_freezes_dynamics` | **да** (504 ≠ 0) |

### 2.1 Что исправлено после ревью (и почему это важно)

1. **`sequence_retrieval_accuracy` был бы checksum'ом.** Ревьюер справедливо заметил, что
   `unbind(bind(f,r),r)==f` — точная модульная инверсия, поэтому 1.0 гарантировано, и мерило
   оно не извлечение. Переименовано в `replay_unbind_fidelity` (честно: целостность адреса),
   а `sequence_retrieval_accuracy` теперь `None` + флаг «нет ground truth» — именно тот
   вариант, который разрешён заданием.
2. **`silent_period_compute_saving` сводился к `1 − n_replay/n_stored`.** Заменён на
   измеренную цену join-а: `1 − inspected_records/index_size`, где `inspected_records` —
   реально просмотренные записи LSH (зависит от заполненности корзин: 8/8, 9/24, 12/160),
   а без запроса — `None`. Заодно добавлен сам запрос (`query_slot_codes`), которого в
   функции не было вообще.
3. **`slot_age`/`gamma_hz` были телеметрией.** `slot_age` теперь задаёт скорость распада
   слота, `gamma_hz` — число подшагов конкуренции (γ/θ); оба проверены сравнением выходов,
   а не наличием поля.
4. **θ-гейт обходился вторым путём PPR** — при закрытом гейте PPR больше не получает код
   запроса (поисков на заблокированном такте: 0).
5. **1-D `ca3_store` падал на втором вызове**, **`swr_replay` использовал глобальный RNG**,
   **`w_inh` в docstring TCC выдавался за коэффициент плотной суммы** — исправлено/уточнено.

---

## 3. Точки врезки вне моей зоны

1. **Цикл сна** — `ziwago/organism_core.py:533` `_on_sleep_phase`, ветка
   `old_phase == "nrem" and new_phase == "rem"`, сразу перед
   `life.sleep_report["rem"] = self._rem_dream()` (:551):

```python
hippo = getattr(core.life, "hippo", None)       # DualHippocampus, integrated.py:161
if hippo is not None and hasattr(hippo, "sleep_cycle"):
    life.sleep_report["hippocampus_sleep"] = hippo.sleep_cycle(
        lambda ec, err: None,                   # cortical_write: консолидация в кору
        adenosine=float(core.chemistry.s.ado),
        ach=0.2,                                # или bundle["ach"] текущего такта
        rem=True,
    )
```
`DualHippocampus.sleep_cycle` (`dual_hippocampus.py:503`) — обёртка над
`Hippocampus.sleep_cycle` (`hippocampus.py:749`). Гейты (`adenosine_threshold`, `ach_max`),
причина отказа (`swr_skipped_reason`, `rem_skipped_reason`) и число реплеев возвращаются в
телеметрии — их надо писать в отчёт прогона (правило «никаких тихих режекторов»).

2. **Статус по §8 — честно:** `sleep_cycle` пока **PHANTOM** (владелец есть, production-вызова
   нет), потому что врезка лежит в чужой зоне (`organism_core.py`). `working.py`,
   `laplace.py`, `phase_vsa.py`, `tcc.py` остаются **DEAD** — дефекты исправлены, но
   владельца в живом организме у них нет; это отдельная работа (владелец + фантом-контроль).
   `Hippocampus`/`DualHippocampus` живые через `IntegratedOrganism.hippo`.

3. **`integrated.py:445` `_update_hippocampus`** — править не нужно, но учесть: `tick()`
   теперь открывает реколл не каждый такт, а один раз за θ-цикл (`theta_cycle_steps=8`
   по умолчанию). Для реколла на каждый токен: `theta_mode="token"`, `theta_cycle_steps=1.0`
   (но это уже не θ-гейт).

4. **`MODULE_MAP_TABLE_2026-11-13.md`** — требует регенерации (`tools/probes/build_module_map.py`,
   не моя зона): у `Hippocampus`/`DualHippocampus` появились вызываемые механизмы
   (`sleep_cycle`, `ca3_store_pair`, `modality_codes`, `index_size`), у `phase_vsa`/`tcc` — новые ветки.

5. **Отрицательный результат, который надо показать прямо:** аденозин меняет амплитуду EC
   и `mismatch`, но **не меняет дискретный DG/CA3-код** (замер: `dg` и `ca3` побитово равны при
   ado=0 и 0.9) — k-WTA по рангу и бинаризация по среднему инвариантны к общему масштабу входа.
   То есть в этой архитектуре аденозин не может изменить адрес памяти, только сигнал новизны.

6. **`vta_da_burst` = 1.0 на любом входе** (mismatch 1.9–3.2 при пороге 0.3; до правки 1.919,
   после 1.960 средний mismatch — сигнал не сдвинут, но сам DA насыщен). По §8 это константа в
   поле метрики, питающая `da_phasic`; кандидат на отдельную задачу (менять формулу DA —
   значит менять живой путь, вне списка дефектов трека B).

---

## 4. Команды проверки и вывод

```
> python -m pytest tests/test_v22_memory_honesty.py -q        # ДО правок (первая редакция файла)
25 failed, 2 passed in 2.91s
  (два прошедших теста были усилены: старая формула давала role_state[3]=0.1 < 0.25,
   а старая константа 0.96 ≠ 0.0 — проверено прямым замером старого поведения)

> python -m pytest tests/test_v22_memory_honesty.py -q        # ПОСЛЕ правок
29 passed in 3.43s

> python -m pytest tests/test_v22_memory_honesty.py tests/test_dual_hippocampus.py tests/test_v21_tcc.py -q
69 passed in 2.58s          # смежные тесты моих файлов не сломаны

> python -m py_compile ziwago/memory/{hippocampus,dual_hippocampus,laplace,working,phase_vsa,tcc}.py
compile ok
> python -c "import sys; sys.path.insert(0,'.'); import ziwago.organism.integrated …"
imports ok
```

Сводка замеров «после» (прибор — скрипты в `%TEMP%`, вывод дословно):

```
CA3 completion hamming=0.000 copy_dist=0.250
  {'recurrent_drive_rms': 0.914301, 'cue_bias_rms': 0.1, 'attractor_share': 0.8752}
CA3 pair A->B: hamming(out,B)=0.000 hamming(out,A)=0.500
route: {'sequence_retrieval_accuracy': None, 'sequence_retrieval_measurement': 'unavailable:no_ground_truth',
        'replay_unbind_fidelity': 1.0, 'replay_unbind_n': 16,
        'silent_period_compute_saving': 0.625, 'silent_compute_records_inspected': 9,
        'silent_compute_records_full_sweep': 24}
working: zero-cue 0, real-pattern k=8 -> 3, tie k=4 -> 4
tcc: DC drive -3.0, harmonic ratio 2.5 = declared 2.5
phase_vsa: silent -> gamma_slot -1; coexist pop2=0.756 pop3=0.347 (winner=2)
dual: exact top_shard_hits 4/4, реколлов за 8 токенов = 1, retrieve_multiprobe вызовов = 2
sleep_cycle: swr_gate_open=True (ado 0.8, ach 0.2), replayed [0,1,2], rem выполнен
```

Живой прогон `Hippocampus.step` × 300 тактов (n_ec=64/n_dg=256/n_ca3=128):
mismatch 1.124…3.156 (среднее 1.96; до правки 1.919), `|w_ca3|max` 0.0105 → 0.1053,
`attractor_share` 0.557.

## 5. Что не проверено

* **CUDA-путь** `phase_vsa.learn_role`: машина без GPU (`torch 2.14.1+cpu`), device-баг проверен
  dtype-прокси (float64-вход → прежний `RuntimeError: addmv … Double, Float`), а не CUDA-тензором.
* `tests/test_integrated_hippocampus.py` не собирается в этом окружении
  (`ModuleNotFoundError: pyarrow` при импорте `train_ziwago_v14_pure_synaptic_dialogue`) — дефект
  окружения, не правки; сквозной `integrated._update_hippocampus` проверен только импортом и
  тестами `DualHippocampus.tick`.
* Влияние θ-гейта и аденозинового гейта на качество ответов живого организма не измерялось
  (нужен тяжёлый прогон с чекпоинтом — на этой машине не проходит, §7 AGENTS.md).
* `sleep_cycle` в `organism_core.py` не врезан (чужая зона) — проверен только сам метод.
