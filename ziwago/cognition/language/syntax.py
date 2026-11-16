"""Синтаксический мост зоны Брока BA44 между аттракторами CA3 (слой L3 language).

Извлечено из монолита train_ziwago_v14_pure_synaptic_dialogue.py (4c split, ARCHITECTURE_MAPPING §7).
Класс самодостаточен: зависимости (encode/decode/get_vec64/vocab) внедряются через
initialize_synaptic_connectors, глобальных связей с v14 нет.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class BA44AttractorSyntaxBridge:
    r"""Реализует биофизический **Синтаксический Мост Зоны Брока (`Col0_SYNTAX_BA44` Syntax Bridge)**
    между двумя последовательными смысловыми аттракторами `CA3` ($\vec{Q}_1 \to \vec{Q}_2$):

      ПРИМЕЧАНИЕ О НАУЧЕНИИ: `learn_bridge_transition` — только примитив для внешне
      размеченной пары (два вектора перехода + независимо заданный индекс союза).
      В production-контуре нет источника такой метки и этот примитив не вызывается;
      сгенерированный или самостоятельно выбранный мост не является целевой меткой.

      1. ПОЧЕМУ БЕЗ МОСТА ПОЛУЧАЛАСЬ СКЛЕЙКА ДВУХ ОТДЕЛЬНЫХ ПРЕДЛОЖЕНИЙ (АТТРАКТОРНАЯ КОНКАТЕНАЦИЯ):
         Когда первый аттрактор $\vec{Q}_1$ доходил до конца своей клаузы, он генерировал терминальную точку `.`
         (или `!`), а второй аттрактор $\vec{Q}_2$ стартовал с заглавной буквы и вводного междометия (`"Yes,"`, `"Ah,"`, `"That sounds great!"`).

      2. КАК РАБОТАЕТ СИНТАКСИЧЕСКИЙ МОСТ `Col0_SYNTAX_BA44` + `SON`:
         - **Обучаемая синаптическая матрица дискурсивных связок $W_{\text{BA44\_bridge}} \in \mathbb{R}^{192 \times K_{\text{conj}}}$**:
           Принимает 192-мерный вектор перехода $[\vec{Q}_1(64),\; \vec{Q}_2(64),\; \vec{v}_{\text{rel\_stimulus}}(64)]$
           и возбуждает синаптические ансамбли подчинительных и сочинительных союзов зоны Брока:
           * `", while"` (одновременность двух процессов),
           * `", and"` (соединение двух фактов),
           * `", before"` / `", after"` (временная последовательность),
           * `", whereas"` / `", as"` (причинно-сопоставительная связь).
         - **Торможение ранней точки (`Period Inhibition Gate` от `SON`)**:
           Когда `Pole_1` доходит до финального токена, но в префронтальной очереди ещё ждёт невысказанный `Pole_2`
           (ненулевой `Epistemic Drive` в `SON`), тормозные интернейроны `BA44` **блокируют точку `.`/`!`** в конце `Pole_1`
           и вместо неё генерируют через $W_{\text{BA44\_bridge}}$ синтаксический мост (например, `", while"` или `", and"`)!
         - **Латеральное подавление повторных приветствий, заглавных букв и инфинитивных швов (`Interjection, Capital & Infinitive Seam Suppression`)**:
           При входе во второй аттрактор `Pole_2` после союза (`", while"` / `", before"` / `", and"`) зона Брока `BA44`
           подавляет стартовые междометия диалога (`"Yes,"`, `"Ah,"`, `"That sounds great!"`), переводит регистр
           первого слова второй клаузы в строчный, а после подчинительных союзов (`", before"`, `", after"`, `", while"`, `", because"`)
           заменяет начальный инфинитивный маркер `"To <verb>"` на личное подлежащее `"you <verb>"` (или `"he <verb>"`),
           устраняя грамматический шов `"before to fix"` -> `"before you fix"`!"""

    BRIDGE_CONNECTORS = (
        ("while", ", while"),
        ("before", ", before"),
        ("after", ", after"),
        ("because", ", because"),
        ("and", ", and"),
    )

    # Параметры научения моста (все — числа, которые попадают в отчёт score_bridges):
    #  - шаг столбца: локальное хеббовское притяжение к направлению реального перехода;
    #  - бонус за лексический триггер союза в стимуле — ПОПРАВКА к матричному вкладу,
    #    а не доминанта (было 0.85 против косинуса в [-1, 1]);
    #  - остаточный приор сочинительного моста ", and" (CONNECTOR_AND_IDX), когда ни
    #    матрица, ни триггер ничего не говорят.
    BRIDGE_LR = 0.35
    TRIGGER_BONUS = 0.10
    TONIC_BIAS = 0.03
    CONNECTOR_AND_IDX = 4

    def __init__(self, d_vec: int = 64, seed: int = 1444):
        g = torch.Generator().manual_seed(seed)
        self.d_vec = int(d_vec)
        self.n_connectors = len(self.BRIDGE_CONNECTORS)
        # Синаптическая матрица зоны Брока BA44: [3 * 64 = 192, n_connectors]
        self.w_ba44_bridge = F.normalize(torch.randn(d_vec * 3, self.n_connectors, generator=g), p=2, dim=0)
        self.connector_token_seqs: Dict[int, List[int]] = {}
        self.connector_trigger_vecs: Dict[int, torch.Tensor] = {}
        self.connector_trigger_ids: Dict[int, int] = {}
        self.connector_trigger_id_sets: Dict[int, set[int]] = {}
        self.lowercase_token_map: Dict[int, int] = {}
        self.tpj_2nd_to_3rd_map: Dict[int, int] = {}
        self.tpj_2nd_to_3rd_fem_map: Dict[int, int] = {}
        self.tpj_3rd_trigger_ids: set[int] = set()
        self.tpj_3rd_fem_trigger_ids: set[int] = set()
        self.tpj_1st_trigger_ids: set[int] = set()
        self.you_subject_token_id: int = 345
        self.he_subject_token_id: int = 339
        self.she_subject_token_id: int = 673
        self.interjection_prefix_tokens: set[str] = {"yes", "ah", "oh", "well", "hey", "hi", "hello", "sure", "certainly"}
        # Телеметрия онлайн-научения моста: сколько раз столбец обновлялся на реальном
        # переходе и какой косинус был у входа с этим столбцом сразу после обновления.
        self.bridge_learn_events: Dict[int, int] = {}
        self.bridge_last_cos: Dict[int, float] = {}

    def initialize_synaptic_connectors(self, encode_fn, decode_fn, get_vec64_fn, vocab_map: Dict[str, int]) -> None:
        r"""Задаёт начальные столбцы из лексических векторов союзов (`while`, `before`, `after`, `because`, `and`),
        строит отображение заглавных токенов в строчные и инициализирует **TPJ Deictic Shift**.
        Лексическая инициализация не является меткой перехода и не вызывает `learn_bridge_transition`."""
        with torch.no_grad():
            comma_id = int(vocab_map.get(",", 11))
            for idx, (trig_word, bridge_phrase) in enumerate(self.BRIDGE_CONNECTORS):
                t_ids = encode_fn(" " + trig_word)
                cap_word = trig_word[0].upper() + trig_word[1:]
                cap_ids_space = encode_fn(" " + cap_word)
                cap_ids_start = encode_fn(cap_word)
                # Токен ',' (без пробела перед запятой!) + токен ' while' / ' and' / ' before'
                self.connector_token_seqs[idx] = [comma_id] + t_ids
                self.connector_trigger_ids[idx] = int(t_ids[0])
                trig_set = {int(t_ids[0])}
                if cap_ids_space:
                    trig_set.add(int(cap_ids_space[0]))
                if cap_ids_start:
                    trig_set.add(int(cap_ids_start[0]))
                self.connector_trigger_id_sets[idx] = trig_set
                t_vec = get_vec64_fn(t_ids[0])
                self.connector_trigger_vecs[idx] = t_vec
                probe_192 = F.normalize(torch.cat([t_vec, t_vec, t_vec], dim=0), p=2, dim=0)
                self.w_ba44_bridge[:, idx] = F.normalize(0.15 * self.w_ba44_bridge[:, idx] + 0.85 * probe_192, p=2, dim=0)

            for tok_str, tid in vocab_map.items():
                if len(tok_str) >= 2 and tok_str.startswith("Ġ") and tok_str[1].isupper():
                    low_key = "Ġ" + tok_str[1].lower() + tok_str[2:]
                    if low_key in vocab_map:
                        self.lowercase_token_map[int(tid)] = int(vocab_map[low_key])
                elif len(tok_str) >= 2 and tok_str[0].isupper():
                    low_space_key = "Ġ" + tok_str[0].lower() + tok_str[1:]
                    if low_space_key in vocab_map:
                        self.lowercase_token_map[int(tid)] = int(vocab_map[low_space_key])

            you_ids = encode_fn(" you")
            if you_ids:
                self.you_subject_token_id = int(you_ids[0])
            he_ids = encode_fn(" he")
            if he_ids:
                self.he_subject_token_id = int(he_ids[0])
            she_ids = encode_fn(" she")
            if she_ids:
                self.she_subject_token_id = int(she_ids[0])

            # Инициализируем синаптическую проекцию TPJ (Temporoparietal Junction) для сдвига 2-го лица -> 3-е лицо (мужской и женский род)
            deictic_pairs = [
                ("Your", "His"),
                (" your", " his"),
                ("You", "He"),
                (" you", " he"),
                (" are", " is"),
            ]
            for src_w, tgt_w in deictic_pairs:
                s_ids = encode_fn(src_w)
                t_ids = encode_fn(tgt_w)
                if len(s_ids) == 1 and len(t_ids) == 1:
                    self.tpj_2nd_to_3rd_map[int(s_ids[0])] = int(t_ids[0])

            deictic_fem_pairs = [
                ("Your", "Her"),
                (" your", " her"),
                ("You", "She"),
                (" you", " she"),
                (" are", " is"),
            ]
            for src_w, tgt_w in deictic_fem_pairs:
                s_ids = encode_fn(src_w)
                t_ids = encode_fn(tgt_w)
                if len(s_ids) == 1 and len(t_ids) == 1:
                    self.tpj_2nd_to_3rd_fem_map[int(s_ids[0])] = int(t_ids[0])

            for w3 in (" he", "He", " him", " his", "His"):
                ids3 = encode_fn(w3)
                if ids3:
                    self.tpj_3rd_trigger_ids.add(int(ids3[0]))
            for w3f in (" she", "She", " her", "Her"):
                ids3f = encode_fn(w3f)
                if ids3f:
                    self.tpj_3rd_fem_trigger_ids.add(int(ids3f[0]))
            for w1 in (" I", "I", " my", "My", " me", " am"):
                ids1 = encode_fn(w1)
                if ids1:
                    self.tpj_1st_trigger_ids.add(int(ids1[0]))

    def apply_tpj_deictic_perspective_shift(
        self,
        clause_ids: List[int],
        stimulus_ids: List[int],
        decode_fn,
    ) -> List[int]:
        """Височно-теменной узел (`TPJ Deictic Perspective Shift`):
          - Если во входном стимуле активированы рецепторы 3-го лица (`he / his / him` или `she / her`) и НЕТ рецепторов 1-го лица (`I / my / me`),
            то при генерации клаузы самоидентификации или совета (`"Your name is..."`, `"in your..."`)
            синапсы `TPJ` переключают местоимения во 3-е лицо мужского или женского рода."""
        stim_set = set(int(t) for t in stimulus_ids)
        has_1st = bool(stim_set & self.tpj_1st_trigger_ids)
        is_third_masc = bool(stim_set & self.tpj_3rd_trigger_ids) and not has_1st
        is_third_fem = bool(stim_set & self.tpj_3rd_fem_trigger_ids) and not has_1st

        out: List[int] = []
        for pos, tid in enumerate(clause_ids):
            t_cur = int(tid)
            if pos > 0 and decode_fn(t_cur).strip() == "You" and t_cur in self.lowercase_token_map:
                t_cur = self.lowercase_token_map[t_cur]
            if is_third_fem and t_cur in self.tpj_2nd_to_3rd_fem_map:
                t_cur = self.tpj_2nd_to_3rd_fem_map[t_cur]
            elif is_third_masc and t_cur in self.tpj_2nd_to_3rd_map:
                t_cur = self.tpj_2nd_to_3rd_map[t_cur]
            out.append(t_cur)
        return out

    def _bridge_input(self, pole1_vec, pole2_vec, stim_vec64) -> torch.Tensor:
        r"""Единственная формула входа моста, общая для научения и для оценки:
        $\vec{x} = \mathrm{normalize}\big([\hat{Q}_1(64),\; \hat{Q}_2(64),\; \hat{v}_{\text{stim}}(64)]\big)$,
        где каждая треть сначала приводится к единичной норме.
        Нормировка ПО-третям обязательна: иначе сырой вектор стимула ($\|v\|\approx 8$ против
        единичных полюсов) забирает 87 % нормы входа, и косинус при научении и при выборе
        считается по разным геометриям (измерено: 0.99 после научения → 0.72 при выборе).
        Общий конструктор обязателен: иначе столбец учится на одном входе, а выбирается по другому."""
        w = self.w_ba44_bridge
        parts: List[torch.Tensor] = []
        for name, vec in (("pole1_vec", pole1_vec), ("pole2_vec", pole2_vec), ("stim_vec64", stim_vec64)):
            t = torch.as_tensor(vec).detach().to(dtype=w.dtype, device=w.device).reshape(-1)
            if int(t.numel()) != self.d_vec:
                raise ValueError(f"{name}: ожидалось {self.d_vec} компонент, получено {int(t.numel())}")
            parts.append(F.normalize(t, p=2, dim=0))
        return F.normalize(torch.cat(parts, dim=0), p=2, dim=0)

    def _stimulus_context_vec(self, stimulus_ids, get_vec64_fn) -> torch.Tensor:
        """Третья треть входа моста на фазе выбора: средний вектор стимула (та же
        конструкция, что и в learn_bridge_transition). Токен без вектора не голосует;
        пустой стимул даёт нулевой вектор, и мост тогда решают только полюса."""
        w = self.w_ba44_bridge
        acc = torch.zeros(self.d_vec, dtype=w.dtype, device=w.device)
        n_used = 0
        if get_vec64_fn is not None and stimulus_ids:
            for tid in stimulus_ids:
                try:
                    v = torch.as_tensor(get_vec64_fn(int(tid))).detach().to(dtype=w.dtype, device=w.device).reshape(-1)
                except (IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                    logging.getLogger(__name__).warning(
                        "Skipped BA44 stimulus token %s because its vector lookup failed (%s: %s).",
                        tid, type(exc).__name__, exc,
                    )
                    continue
                if int(v.numel()) != self.d_vec:
                    logging.getLogger(__name__).warning(
                        "Skipped BA44 stimulus token %s with vector size %d; expected %d.",
                        tid, int(v.numel()), self.d_vec,
                    )
                    continue
                acc = acc + v
                n_used += 1
        if n_used == 0:
            return acc
        return F.normalize(acc / float(n_used), p=2, dim=0)

    def learn_bridge_transition(
        self,
        pole1_vec,
        pole2_vec,
        stim_vec64,
        connector_idx: int,
        lr: Optional[float] = None,
    ) -> dict:
        r"""Низкоуровневое обновление столбца BA44 на внешне размеченной паре перехода
        $\vec{Q}_1 \to \vec{Q}_2$. `connector_idx` должен поступать из независимой
        разметки, а не из генерации/самостоятельного выбора моста. В текущем production
        источника такой разметки и вызова этого метода нет.

        Обновление выполняется локальным хеббовским правилом без backprop: столбец
        подтягивается к направлению перехода и нормируется заново, поэтому
        $\cos(\vec{x}, W[:, k]) \to 1$ за несколько предъявлений:

        $$W[:,k] \leftarrow \mathrm{normalize}\big((1-\lambda)\,W[:,k] + \lambda\,\vec{x}\big)$$

        Возвращает телеметрию шага (числа измерения, а не константы)."""
        idx = int(connector_idx)
        if idx < 0:
            idx += self.n_connectors
        if not (0 <= idx < self.n_connectors):
            raise IndexError(f"connector_idx {connector_idx} вне диапазона [0, {self.n_connectors})")
        rate = float(self.BRIDGE_LR if lr is None else lr)
        if not (0.0 < rate <= 1.0):
            raise ValueError(f"lr должен быть в (0, 1], получено {rate}")

        bridge_in = self._bridge_input(pole1_vec, pole2_vec, stim_vec64)
        with torch.no_grad():
            updated = (1.0 - rate) * self.w_ba44_bridge[:, idx] + rate * bridge_in
            norm = float(updated.norm().item())
            if norm > 0.0:
                updated = updated / norm
            self.w_ba44_bridge[:, idx] = updated
            cos_after = float(torch.dot(bridge_in, self.w_ba44_bridge[:, idx]).item())
            self.bridge_last_cos[idx] = cos_after
            self.bridge_learn_events[idx] = int(self.bridge_learn_events.get(idx, 0)) + 1

        return {
            "connector_idx": idx,
            "lr": rate,
            "cos_after": cos_after,
            "learn_events": int(self.bridge_learn_events[idx]),
            "production_connection": "disconnected",
            "supervision_source": "externally_supplied_connector_label",
        }

    def bridge_learning_telemetry(self) -> dict:
        """Отчёт о внешних вызовах примитива; production-обучение остаётся отключённым."""
        return {
            "production_connection": "disconnected",
            "supervision_source": "externally_supplied_connector_label",
            "connectors": [
                {
                    "connector_idx": idx,
                    "connector_name": name,
                    "learn_events": int(self.bridge_learn_events.get(idx, 0)),
                    "last_cos": self.bridge_last_cos.get(idx),
                }
                for idx, (name, _) in enumerate(self.BRIDGE_CONNECTORS)
            ],
        }

    def score_bridges(
        self,
        pole1_vec,
        pole2_vec,
        stimulus_ids,
        get_vec64_fn,
    ) -> List[dict]:
        r"""Честная таблица драйвов моста BA44: по строке на союз, все величины измеримы.

        * `matrix_drive` — вклад ОБУЧЕННОЙ матрицы, приведённый к $[0,1]$ монотонным
          отображением $0.5(\cos+1)$; `matrix_cos` — сырой косинус (знак не теряется);
        * `trigger_active` — лексический триггер союза в стимуле (поправка +`TRIGGER_BONUS`);
        * `tonic_bias` — остаточный приор `", and"`;
        * `total_drive` — то, что решает выбор: `matrix_drive + TRIGGER_BONUS*trigger_active + tonic_bias`,
          где матрица — главный член, а не тонет под константами."""
        bridge_in = self._bridge_input(
            pole1_vec,
            pole2_vec,
            self._stimulus_context_vec(stimulus_ids, get_vec64_fn),
        )
        stim_id_set = set(int(t) for t in (stimulus_ids or []))
        rows: List[dict] = []
        with torch.no_grad():
            for idx in range(self.n_connectors):
                cos = float(torch.dot(bridge_in, self.w_ba44_bridge[:, idx]).item())
                matrix_drive = float(min(1.0, max(0.0, 0.5 * (cos + 1.0))))
                trig_set = self.connector_trigger_id_sets.get(idx, set())
                # Триггер-словом задаются только подчинительные союзы (индексы 0..3);
                # сочинительный ", and" (индекс 4) триггера не имеет — у него tonic_bias.
                trigger_active = 1.0 if (idx < self.CONNECTOR_AND_IDX and bool(trig_set & stim_id_set)) else 0.0
                tonic_bias = float(self.TONIC_BIAS) if idx == self.CONNECTOR_AND_IDX else 0.0
                rows.append({
                    "connector_idx": idx,
                    "connector_name": self.BRIDGE_CONNECTORS[idx][0],
                    "matrix_cos": cos,
                    "matrix_drive": matrix_drive,
                    "trigger_active": trigger_active,
                    "tonic_bias": tonic_bias,
                    "total_drive": float(matrix_drive + self.TRIGGER_BONUS * trigger_active + tonic_bias),
                })
        return rows

    def select_bridge_tokens_between_attractors(
        self,
        pole1_vec: torch.Tensor,
        pole2_vec: torch.Tensor,
        stimulus_ids: List[int],
        get_vec64_fn,
    ) -> Tuple[str, List[int], float]:
        r"""Вычисляет через синаптическую матрицу $W_{\text{BA44\_bridge}}$ оптимальный союзный мост
        между первым аттрактором $\vec{Q}_1$ и вторым аттрактором $\vec{Q}_2$.

        Выбор идёт по `score_bridges`: решает матричный вклад (`matrix_drive`), а триггер
        и `tonic_bias` — только поправки. Сигнатура и возврат (phrase, token_ids, score) прежние;
        `score` теперь равен `total_drive` выбранного моста (шкала $[0, 1.13]$, не «сырая проводка»)."""
        rows = self.score_bridges(pole1_vec, pole2_vec, stimulus_ids, get_vec64_fn)
        best = rows[0]
        for row in rows[1:]:
            if float(row["total_drive"]) > float(best["total_drive"]):
                best = row
        best_idx = int(best["connector_idx"])
        conn_name, conn_phrase = self.BRIDGE_CONNECTORS[best_idx]
        return conn_phrase, list(self.connector_token_seqs[best_idx]), round(float(best["total_drive"]), 4)

    def bridge_two_attractor_clauses_in_ba44(
        self,
        clause1_ids: List[int],
        clause2_ids: List[int],
        bridge_token_ids: List[int],
        decode_fn,
        bridge_phrase: str = "",
        stimulus_ids: Optional[List[int]] = None,
    ) -> List[int]:
        """Сшивает два аттракторных следа `clause1_ids` и `clause2_ids` через синтаксический мост `Col0_SYNTAX_BA44`:
          1) Тормозит терминальную точку `.` / `!` в конце `clause1_ids` (Period Inhibition Gate);
          2) Вставляет синаптический мост союза `bridge_token_ids` (`", while"`, `", and"`, `", before"`);
          3) Подавляет стартовые междометия (`"Yes, "`, `"Ah, "`, `"That sounds great! "`) в начале `clause2_ids`,
             устраняет грамматический шов подчинительного союза с инфинитивом (`"before To fix"` -> `"before you fix"`)
             и переводит первое слово второй клаузы в строчный регистр!"""
        if not clause1_ids:
            return clause2_ids
        if not clause2_ids:
            return clause1_ids

        # 1. В первой клаузе подавляем вводное междометие с восклицательным знаком ("That sounds great! "), если за ней идёт вторая клауза,
        #    и тормозим финальную точку "." на конце первой клаузы!
        c1 = list(clause1_ids)
        for i in range(min(6, len(c1))):
            if decode_fn(c1[i]).strip().endswith("!"):
                c1 = c1[i + 1 :]
                break
        while c1 and decode_fn(c1[-1]).strip() in (".", "!", ";", ","):
            c1.pop()

        # 2. Во второй клаузе подавляем стартовые диалоговые междометия ("Yes, ", "Ah, ", "That sounds great! ")
        c2 = list(clause2_ids)
        for i in range(min(6, len(c2))):
            if decode_fn(c2[i]).strip().endswith("!"):
                c2 = c2[i + 1 :]
                break
        if len(c2) >= 2:
            w0 = decode_fn(c2[0]).strip(" .,!?;:").lower()
            w1 = decode_fn(c2[1]).strip()
            if w0 in self.interjection_prefix_tokens and w1 == ",":
                c2 = c2[2:]

        # 3. Устранение грамматического шва подчинительного союза с инфинитивом цели ("before To fix..." -> "before you fix..."):
        if c2:
            first_w = decode_fn(c2[0]).strip()
            is_subordinating = any(sub in bridge_phrase for sub in ("before", "after", "while", "because"))
            if first_w.lower() == "to" and is_subordinating:
                c2[0] = self.you_subject_token_id
            elif first_w.lower() != "i" and c2[0] in self.lowercase_token_map:
                c2[0] = self.lowercase_token_map[c2[0]]

        return c1 + bridge_token_ids + c2