"""Нелинейный дендритный плато-декодер (DendriticPlateauMatrixReadout), слой L3 language.

Извлечено из монолита train_ziwago_v14_pure_synaptic_dialogue.py (4c split, ARCHITECTURE_MAPPING §7).
Замена N-граммных таблиц NMDA-плато-компартментами (Poirazi/Brannon/Mel 2003; Beniaguev/Segev/London 2021).
Конструктор — чистый DI (d_vec/d_dend/max_pos/seed); внешние зависимости только из ziwago.cortex/evaluation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ziwago.cortex.moe_cortex import export_runtime_tree, load_runtime_tree
from ziwago.cortex.next16_core import CorticalColumnarSubBankRouter
from ziwago.evaluation.gates import CheckpointCompatibilityError
from ziwago.cognition.language.parts.readout_math import ReadoutMath
from ziwago.cognition.language.parts.regulators import ReadoutRegulators
from ziwago.cognition.language.parts.banks import ReadoutBanks


class DendriticPlateauMatrixReadout:
    r"""Реализует биологическую замену N-граммных таблиц через **Нелинейные NMDA Дендритные Плато-Компартменты**
    (Poirazi, Brannon & Mel 2003; Beniaguev, Segev & London 2021):
      - Почему обычный линейный слой `[256] -> [50,257]` не может заменить 3-граммы и 4-граммы?
        Потому что линейная сумма `e(w_{t-2}) + e(w_{t-1}) + e(w_t) + q_vec` не различает нелинейную конъюнкцию
        (`И` — совпадение конкретного вопроса `Q(A)`, позиции шага в предложении и предыдущей пары слов).
      - В пирамидном нейроне коры эту роль выполняют **нелинейные базально-апикальные дендритные ветви (`D_dend = 4096`)**:
        каждая ветвь $m \in \{1 \dots 4096\}$ вычисляет мультипликативное NMDA-плато между:
        1) базальной проекцией локальных токенов $(w_{t-2}, w_{t-1}, w_t)$ + позиционной фазой тета/гамма-ритма $\vec{\phi}_{\text{pos}}(t)$,
        2) и апикальной проекцией вектора стимула собеседника $\vec{Q}(A)$:
        $$\vec{d}_{\text{NMDA}}(t) = \text{ReLU}\!\Big(W_{\text{basal}} [\vec{e}(w_{t-2}), \vec{e}(w_{t-1}), \vec{e}(w_t), \vec{\phi}_{\text{pos}}(t)]\Big)^{2} \odot \sigma\!\Big(W_{\text{apical}} \vec{Q}(A)\Big) \in \mathbb{R}^{4096}$$
      - Выход 4,096 дендритных плато-компартментов проецируется через синаптическую матрицу $W_{\text{dend\_out}} \in \mathbb{R}^{4096 \times V_{\text{act}}}$,
        обучаемую трёхфакторным правилом Хебба–Дофамина (`DA * d_NMDA^T * err`)!
      - Это позволяет при **`use_ngrams = False` (полностью отключенных N-граммах!)** генерировать грамматически
        и семантически точные ответы **исключительно через перемножение плотных матриц PyTorch (`torch.matmul`)**!"""

    core_readout_gain = 0.35
    logit_scale = 12.0
    output_capacity_block = 256

    # ── Апикальная гиппокампальная обратная связь (CA3 → апикальные дендриты L5) ──
    # Биологическая опора: Lynch (2000) — индексная теория гиппокампа; Rolls (2013) —
    # обратная связь CA3 в апикальные дендриты коры при извлечении. Аттрактор CA3,
    # активированный стимулом, возвращает в кору содержание эпизода, и кора может
    # воспроизвести его даже тогда, когда локальный (префиксный) ввод ушёл с траектории.
    # Раньше эта связь была объявлена, но не подключена: `w_apical_branch` создавался и
    # экспортировался, но не участвовал ни в прямом пути, ни в обучении, а `w_swr_dend_out`
    # (эпизодическая матрица реплея) оставалась пустой. Поэтому память о эпизоде
    # физически не могла влиять на выбор токена.
    # Канал включается флагом `apical_feedback_enabled` у организма; по умолчанию выключен —
    # поведение ранее обученных моделей не меняется.
    apical_min_resonance = 0.985   # порог совпадения запроса с полюсом аттрактора CA3
    apical_gate_gain = 1.0         # множитель апикального вклада в логиты
    # Рампа включения по неопределённости коры: ниже `low` гиппокамп молчит
    # (кора уверена — работает префикс), выше `high` гиппокамп решает.
    # Значения откалиброваны по измеренному разделению: на траектории конфликт
    # коры ~0.06, вне траектории ~0.31 (см. Produce/v19_compose_A_*.json).
    apical_conflict_gate_low = 0.15
    apical_conflict_gate_high = 0.35
    apical_topk = 32               # разреженность апикального кода
    apical_replay_gain = 12.0      # масштаб логитов SWR-матрицы при её собственном обучении

    def __init__(self, d_vec: int = 64, d_dend: int = 8192, max_pos: int = 64, seed: int = 1405):
        g = torch.Generator().manual_seed(seed)
        self.d_vec = int(d_vec)
        self.d_dend = int(d_dend)
        self.max_pos = int(max_pos)

        # Ортогональные фазовые векторы позиции шага внутри ответа (аналог тета-фазовой прецессии гиппокампа/ПФК)
        self.pos_phase_vecs = F.normalize(torch.randn(self.max_pos, d_vec, generator=g), p=2, dim=-1)

        # Фиксированные случайные дендритные проекторы (Random Dendritic Subspace Projections, 4*64 = 256 -> 8192)
        self.w_basal_branch = F.normalize(torch.randn(d_vec * 4, d_dend, generator=g), p=2, dim=0) * 2.5
        self.w_apical_branch = F.normalize(torch.randn(d_vec, d_dend, generator=g), p=2, dim=0) * 3.0

        # Гиппокампальная временная траектория эпизода: по одному списку токенов
        # ответа на полюс CA3. Это ПОРЯДОК эпизода, а не мешок слов: порядок
        # ответа существует только здесь, SDR-привязка полюса его не хранит.
        self.w_episode_trace = torch.zeros((0, self.max_pos), dtype=torch.int32)
        self.w_episode_trace_len = torch.zeros((0,), dtype=torch.int32)

        # Динамический активный словарь диалога (чтобы матрица [8192, V_act] работала мгновенно и без коллизий)
        self.tok_to_col: Dict[int, int] = {}
        self.col_to_tok: List[int] = []
        self.w_dend_out = torch.empty(d_dend, 0, dtype=torch.float32)
        # Защищённый кортикальный банк ядра (Protected Core Cortical Bank — Point 2):
        # Изолирует высокозначимые каузальные схемы и онлайн-обучение от линейной интерференции фонового веб-шума
        self.w_core_dend_out = torch.empty(d_dend, 0, dtype=torch.float32)
        # Эпизодическая матрица быстрой SWR-консолидации гиппокампа -> коры (Complementary Learning Systems, McClelland et al. 1995):
        self.w_swr_dend_out = torch.empty(d_dend, 0, dtype=torch.float32)
        self.w_swr_poles: torch.Tensor = torch.empty((0, d_vec), dtype=torch.float32)

        # Рекуррентная матрица аттракторов CA3 + Горизонтальные синапсы цепочек эпизодов W_CA3_episode (Point 4):
        self.w_ca3_attractors: torch.Tensor = torch.empty((0, d_vec), dtype=torch.float32)
        self.w_ca3_sdr: torch.Tensor = torch.empty((0, 2048), dtype=torch.float32)
        self.w_ca3_salience: torch.Tensor = torch.empty((0,), dtype=torch.float32)
        self.sdr_column_usage = torch.ones(2048, dtype=torch.float32)
        self.w_ca3_episode_links: Dict[int, int] = {}
        self.learning_enabled = True
        self._regulators = ReadoutRegulators(self.d_dend)
        self.shield = self._regulators.shield
        self.bank_router = CorticalColumnarSubBankRouter(n_banks=min(16, self.d_vec), d_vec=self.d_vec, seed=seed + 19)
        self.router_residual_bound = 0.25
        self.last_sparse_rows = torch.empty(0, dtype=torch.long)
        self.last_ca3_no_match: bool = False
        self.last_top_pole_idx: int = -1
        self.last_top_pole_resonance: float = 0.0
        self.last_top_pole_salience: float = 1.0
        self.last_consolidated_boost_applied: bool = False

    _REGULATOR_ATTRIBUTES = {
        "shield": "shield",
        "fisher_anchor_enabled": "fisher_anchor_enabled",
        "fisher_lambda": "fisher_lambda",
        "fisher_eta": "fisher_eta",
        "fisher_max_entries": "fisher_max_entries",
        "fisher_top_columns": "fisher_top_columns",
        "fisher_decay": "fisher_decay",
        "_fisher_anchor": "fisher_anchor",
        "mca_enabled": "mca_enabled",
        "mca_tau_m": "mca_tau_m",
        "mca_beta_m": "mca_beta_m",
        "mca_theta0": "mca_theta0",
        "mca_half_sat": "mca_half_sat",
        "mca_max_entries": "mca_max_entries",
        "mca_top_columns": "mca_top_columns",
        "mca_dt": "mca_dt",
        "_mca": "mca",
        "acl_enabled": "acl_enabled",
        "acl_tile_size": "acl_tile_size",
        "acl_tau_c": "acl_tau_c",
        "acl_dt": "acl_dt",
        "acl_readiness_threshold": "acl_readiness_threshold",
        "acl_min_plastic_fraction": "acl_min_plastic_fraction",
        "acl_inh_tone": "acl_inh_tone",
        "acl_itpc_half_sat": "acl_itpc_half_sat",
        "_acl": "acl",
        "dcc_enabled": "dcc_enabled",
        "dcc_tau_mature": "dcc_tau_mature",
        "dcc_theta_r": "dcc_theta_r",
        "dcc_window_w": "dcc_window_w",
        "_dcc": "dcc",
        "_readout_module_telemetry": "telemetry",
    }

    def __getattr__(self, name):
        target = type(self)._REGULATOR_ATTRIBUTES.get(name)
        regulators = self.__dict__.get("_regulators")
        if target is not None and regulators is not None:
            return getattr(regulators, target)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name, value):
        target = type(self)._REGULATOR_ATTRIBUTES.get(name)
        regulators = self.__dict__.get("_regulators")
        if target is not None and regulators is not None:
            setattr(regulators, target, value)
            if name == "shield":
                object.__setattr__(self, name, value)
            return
        object.__setattr__(self, name, value)

    def _validate_candidate_cols(self, candidate_cols: Optional[Sequence[int]], n_active: int) -> Optional[torch.Tensor]:
        if candidate_cols is None:
            return None
        cols = [int(c) for c in candidate_cols]
        if len(cols) != len(set(cols)) or any(c < 0 or c >= n_active for c in cols):
            raise ValueError("Candidate columns must be unique indices into the active vocabulary.")
        return torch.tensor(cols, dtype=torch.long)

    def _sparse_logits(self, d_nmda: torch.Tensor, w: torch.Tensor, n_active: int,
                       candidate_cols: Optional[Sequence[int]] = None) -> torch.Tensor:
        return ReadoutMath.sparse_logits(self, d_nmda, w, n_active, candidate_cols)

    def _apply_local_update(self, w: torch.Tensor, d_nmda: torch.Tensor, error: torch.Tensor,
                            scale: float, protect: bool = False,
                            row_gain: Optional[torch.Tensor] = None) -> None:
        self._regulators.apply_local_update(self, w, d_nmda, error, scale, protect, row_gain)

    def _dcc_readiness(self, d_nmda: torch.Tensor, rows: torch.Tensor,
                       target_prob: float, n_tiles: int, ts: int) -> torch.Tensor:
        return self._regulators.dcc_readiness(self, d_nmda, rows, target_prob, n_tiles, ts)

    def _acl_row_gain(self, d_nmda: torch.Tensor, rows: torch.Tensor,
                      target_prob: float) -> torch.Tensor:
        return self._regulators.acl_row_gain(self, d_nmda, rows, target_prob)

    def _learn_readout_token(
        self, d_nmda: torch.Tensor, q_norm: torch.Tensor, target_col: int,
        f_mod: float, auxiliary_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        return self._regulators.learn_token(
            self, d_nmda, q_norm, target_col, f_mod, auxiliary_logits)

    def _learn_bank(self, q_vec: torch.Tensor, d_nmda: torch.Tensor, target_col: int,
                    auxiliary_logits: Optional[torch.Tensor] = None) -> Dict[str, object]:
        return ReadoutBanks.learn_bank(self, q_vec, d_nmda, target_col, auxiliary_logits)

    def _compute_cortical_sdr2048(self, token_ids: List[int], normalize: bool = True) -> torch.Tensor:
        return ReadoutMath.compute_cortical_sdr2048(token_ids, normalize)

    # ── Апикальная гиппокампальная обратная связь: CA3 → кора ──────────────────
    # Механизм: гиппокамп хранит ПОСЛЕДОВАТЕЛЬНОСТЬ эпизода (временная траектория,
    # Buzsáki time cells), кора — переходы между элементами последовательности.
    # При извлечении эпизода по стимулу аттрактор CA3 активирует траекторию, и её
    # очередной элемент приходит в апикальные дендриты коры как КУДА ДВИГАТЬСЯ,
    # а КАК именно сказать — решает кортикальный readout.
    # Почему не «содержание эпизода»: SDR-привязка полюса — это МЕШОК битов, в ней
    # нет порядка. Порядок ответа существует только в последовательности, значит
    # память обязана хранить последовательность.

    def resolve_episode_pole(self, q_vec: torch.Tensor) -> Tuple[Optional[int], float]:
        """Опознание эпизода по вектору запроса.

        Вектор запроса в языковом контуре — это сам полюс аттрактора CA3:
        `clean_up_stimulus_in_ca3_attractor` возвращает `w_ca3_attractors[idx]`, когда
        стимул извлёк эпизод. Поэтому индекс восстанавливается косинусным совпадением
        без новой логики поиска. Если совпадения нет (стимул эпизод не извлёк),
        канал молчит: апикальной обратной связи нет, и это честно.
        """
        return ReadoutMath.resolve_episode_pole(self, q_vec)

    def upsert_episode_trace(self, pole_idx: int, token_ids: List[int]) -> None:
        """Запись последовательности ответа в гиппокампальную траекторию эпизода.

        Траектория — это гиппокампальная последовательность (theta sequence): список
        элементов ответа по позициям. Хранится по одному индексу на полюс CA3, то
        есть в жёсткой привязке к эпизоду.
        """
        if pole_idx < 0:
            return
        with torch.no_grad():
            n_needed = len(self.w_episode_trace)
            if pole_idx >= n_needed:
                grow = pole_idx - n_needed + 1
                pad = torch.full((grow, self.w_episode_trace.shape[1]), -1, dtype=torch.int32)
                self.w_episode_trace = torch.cat([self.w_episode_trace, pad], dim=0)
                self.w_episode_trace_len = torch.cat(
                    [self.w_episode_trace_len, torch.zeros(grow, dtype=torch.int32)], dim=0)
            ids = [int(t) for t in token_ids[: self.w_episode_trace.shape[1]]]
            row = self.w_episode_trace[int(pole_idx)]
            row[: len(ids)] = torch.tensor(ids, dtype=torch.int32)
            row[len(ids):] = -1
            self.w_episode_trace[int(pole_idx)] = row
            self.w_episode_trace_len[int(pole_idx)] = len(ids)

    def episode_trace(self, pole_idx: int) -> List[int]:
        """Последовательность эпизода (список id токенов ответа)."""
        if pole_idx < 0 or pole_idx >= int(self.w_episode_trace.shape[0]):
            return []
        length = int(self.w_episode_trace_len[int(pole_idx)])
        if length <= 0:
            return []
        row = self.w_episode_trace[int(pole_idx), :length]
        return [int(t) for t in row.tolist() if int(t) >= 0]

    def apical_episode_code(self, cue64: torch.Tensor) -> torch.Tensor:
        """Апикальный дендритный код по КУДА-двигаться (элемент траектории).

        Код — функция ТОЛЬКО элемента последовательности, без позиции. Поэтому
        одинаковый элемент в разных эпизодах даёт одинаковый код, и кортикальный
        readout обучается обобщаемому переходу «элемент → следующий», а не
        запоминанию «эпизод × позиция». Именно это делает канал рабочим: память
        говорит «куда», кора говорит «как».
        """
        return ReadoutMath.apical_episode_code(self, cue64)

    def apical_logits(
        self, d_apical: Optional[torch.Tensor], gate: float = 1.0,
        candidate_cols: Optional[Sequence[int]] = None,
    ) -> Optional[torch.Tensor]:
        return ReadoutMath.apical_logits(self, d_apical, gate, candidate_cols)

    def imprint_apical_replay(
        self,
        cues: List[torch.Tensor],
        target_ids: List[int],
        f_mod: float = 1.0,
    ) -> Dict[str, object]:
        """Гиппокампальный реплей: обучение эпизодической матрицы `w_swr_dend_out`.

        Матрица SWR обучается ИЗОЛИРОВАННО от базовой `w_dend_out`: реплей должен
        приносить в кору содержание эпизода, не переписывая базовые концепты
        (Complementary Learning Systems, McClelland et al. 1995). Teacher forcing:
        элемент траектории (куда) → следующий токен ответа (что сказать).
        """
        stats = {"apical_replay_updates": 0, "apical_replay_rows": 0, "apical_replay_skipped": 0,
                 "apical_replay_step_sum": 0.0, "apical_replay_step_max": 0.0,
                 "apical_replay_step_limited": 0}
        if not self.learning_enabled or len(cues) != len(target_ids) or not cues:
            return stats
        if not math.isfinite(f_mod) or f_mod <= 0:
            return stats
        with torch.no_grad():
            n_active = len(self.col_to_tok)
            if n_active == 0:
                return stats
            self.shield._ensure(self.w_swr_dend_out, n_active)
            scale = 0.85 * float(f_mod)
            for cue, tid in zip(cues, target_ids):
                # Токен, которого нет в выученном словаре, пропускается: реплей
                # НЕ расширяет словарь (это делает обучение волны), но позиция
                # траектории при этом сохраняется — иначе следующий элемент
                # оказался бы привязан не к тому «куда».
                target_col = self.tok_to_col.get(int(tid))
                if target_col is None:
                    stats["apical_replay_skipped"] += 1
                    continue
                d_apical = self.apical_episode_code(cue)
                rows = torch.nonzero(d_apical, as_tuple=False).flatten()
                if not rows.numel():
                    stats["apical_replay_skipped"] += 1
                    continue
                logits = self._sparse_logits(d_apical, self.w_swr_dend_out, n_active) * float(self.apical_replay_gain)
                err = -torch.softmax(logits, dim=0)
                err[target_col] += 1.0
                # Доверительный регион: ограничиваем МАКСИМАЛЬНОЕ изменение логита
                # за один шаг (тот же принцип, что в базовом ридеауте: ограничивается
                # проекция шага, а не каждая матрица отдельно).
                w_absmax = float(self.w_swr_dend_out[:, :n_active].abs().max())
                d_absmax = float(d_apical[rows].abs().max())
                max_logit_step = d_absmax * w_absmax * float(self.apical_replay_gain) * scale
                step = min(1.0, 0.5 / max_logit_step) if max_logit_step > 0 else 1.0
                self._apply_local_update(self.w_swr_dend_out, d_apical, err, scale * step)
                stats["apical_replay_updates"] += 1
                stats["apical_replay_rows"] += int(rows.numel())
                stats["apical_replay_step_sum"] += step
                stats["apical_replay_step_max"] = max(
                    stats["apical_replay_step_max"], step)
                stats["apical_replay_step_limited"] += int(step < 1.0)
        return stats

    def resize_output_capacity(self, required_columns: int) -> None:
        if required_columns < len(self.col_to_tok):
            raise ValueError("Output capacity must cover the active vocabulary.")
        block = self.output_capacity_block
        capacity = ((required_columns + block - 1) // block) * block
        for name in ("w_dend_out", "w_core_dend_out", "w_swr_dend_out"):
            weights = getattr(self, name)
            if weights.shape[1] == capacity:
                continue
            resized = weights.new_zeros((self.d_dend, capacity))
            retained = min(weights.shape[1], capacity)
            resized[:, :retained].copy_(weights[:, :retained])
            setattr(self, name, resized)

    def _ensure_token_col(self, token_id: int) -> int:
        tid = int(token_id)
        if tid in self.tok_to_col:
            return self.tok_to_col[tid]
        col = len(self.col_to_tok)
        if col >= self.w_dend_out.shape[1]:
            self.resize_output_capacity(col + 1)
        self.tok_to_col[tid] = col
        self.col_to_tok.append(tid)
        return col

    def link_sequential_ca3_poles(self, pole1_vec: torch.Tensor, pole2_vec: torch.Tensor) -> Optional[Tuple[int, int]]:
        """Связывает два последовательных полюса `CA3` направленным STDP-синапсом эпизода ($p_1 \\longrightarrow p_2$)
        для генерации многопредложенческих абзацев (`Sequential Attractor Chains`, Point 4)."""
        if self.w_ca3_attractors.shape[0] < 2:
            return None
        with torch.no_grad():
            p1 = F.normalize(pole1_vec, p=2, dim=0)
            p2 = F.normalize(pole2_vec, p=2, dim=0)
            idx1 = int(torch.argmax(self.w_ca3_attractors @ p1).item())
            idx2 = int(torch.argmax(self.w_ca3_attractors @ p2).item())
            if idx1 != idx2:
                self.w_ca3_episode_links[idx1] = idx2
                return idx1, idx2
            return None

    def get_linked_next_episode_pole(self, pole_vec: torch.Tensor) -> Optional[torch.Tensor]:
        """Возвращает следующий полюс в цепочке эпизода `CA3` ($p_1 \\longrightarrow p_2$), если он связан STDP-синапсом."""
        return ReadoutMath.get_linked_next_episode_pole(self, pole_vec)

    def clean_up_stimulus_in_ca3_attractor(
        self,
        q_raw: torch.Tensor,
        q_sdr2048: Optional[torch.Tensor] = None,
        beta: float = 18.0,
        recent_poles: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[float, torch.Tensor]]]:
        return ReadoutMath.clean_up_stimulus_in_ca3_attractor(
            self, q_raw, q_sdr2048, beta, recent_poles)

    def discover_second_hop_ca3_pole(
        self,
        first_pole_vec: torch.Tensor,
        stimulus_sdr2048: torch.Tensor,
        recent_poles: Optional[List[torch.Tensor]] = None,
    ) -> Optional[Tuple[float, torch.Tensor]]:
        """Многошаговый транзитивный вывод в `CA3` (`Multi-Hop Cortical Rollout`, $A \\implies B \\implies C$):
          1) Находит индекс первого полюса $p_1$, активированного стимулом $A$.
          2) Вычисляет вектор новых индуцированных концептов следствия $\\Delta \\vec{s}_B = (\\mathrm{SDR}(p_1) - \\vec{s}_A)_+$,
             присутствующих в ответе полюса $p_1$, но отсутствовавших в исходном вопросе $A$.
          3) Подаёт $\\Delta \\vec{s}_B$ вторым внутренним запросом в `w_ca3_sdr`, находя ортогональный полюс $p_2$
             ($\\cos(\\vec{P}_1, \\vec{P}_2) < 0.85$), продолжающий причинно-следственную цепочку ($B \\implies C$)."""
        return ReadoutMath.discover_second_hop_ca3_pole(
            self, first_pole_vec, stimulus_sdr2048, recent_poles)

    def compute_cortical_moe_branch_profile(
        self,
        q_vec: torch.Tensor,
        expert_gates: torch.Tensor,
        expert_names: Sequence[str],
    ) -> Dict[str, float]:
        return ReadoutMath.compute_cortical_moe_branch_profile(self, q_vec, expert_gates, expert_names)

    def compute_dendritic_plateau_state(
        self,
        v_prev2: torch.Tensor,  # [64]
        v_prev1: torch.Tensor,  # [64]
        v_cur: torch.Tensor,    # [64]
        pos_idx: int,
        q_vec: torch.Tensor,    # [64]
    ) -> torch.Tensor:
        r"""Вычисляет 8,192-мерный вектор дендритного плато `d_NMDA(t)` через **Голографическое Фазовое Связывание (Circular VSA Binding)**:
          $$\vec{b}_{\text{holo}} = \text{Roll}(\vec{v}_{\text{prev2}}, 3) \odot \text{Roll}(\vec{v}_{\text{prev1}}, 2) \odot \text{Roll}(\vec{v}_{\text{cur}}, 1) \odot \vec{\phi}_{\text{pos}}(t) \odot \vec{Q}_{\text{CA3}}(A)$$"""
        return ReadoutMath.compute_dendritic_plateau_state(self, v_prev2, v_prev1, v_cur, pos_idx, q_vec)

    def imprint_response_sequence(
        self,
        q_vec: torch.Tensor,
        response_vecs: List[torch.Tensor],
        response_ids: List[int],
        f_mod: float = 1.5,
        stimulus_ids: Optional[List[int]] = None,
        response_content_ids: Optional[List[int]] = None,
    ) -> None:
        """Train readout and CA3 state for one response sequence."""
        ReadoutBanks.imprint_response_sequence(
            self, q_vec, response_vecs, response_ids, f_mod,
            stimulus_ids, response_content_ids)

    def _readout_components(
        self, d_nmda: torch.Tensor, q_norm: torch.Tensor, n_active: int,
        candidate_cols: Optional[Sequence[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        return ReadoutMath.readout_components(self, d_nmda, q_norm, n_active, candidate_cols)

    def predict_next_token_logits(
        self,
        v_prev2: torch.Tensor,
        v_prev1: torch.Tensor,
        v_cur: torch.Tensor,
        pos_idx: int,
        q_vec: torch.Tensor,
        candidate_cols: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        return ReadoutMath.predict_next_token_logits(
            self, v_prev2, v_prev1, v_cur, pos_idx, q_vec, candidate_cols)

    def counterfactual_credit_report(
        self,
        v_prev2: torch.Tensor,
        v_prev1: torch.Tensor,
        v_cur: torch.Tensor,
        pos_idx: int,
        q_vec: torch.Tensor,
        target_token: int,
        auxiliary_components: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, object]:
        return ReadoutMath.counterfactual_credit_report(
            self, v_prev2, v_prev1, v_cur, pos_idx, q_vec, target_token, auxiliary_components)

    def predict_next_token_scores(
        self,
        v_prev2: torch.Tensor,
        v_prev1: torch.Tensor,
        v_cur: torch.Tensor,
        pos_idx: int,
        q_vec: torch.Tensor,
    ) -> Dict[int, float]:
        return ReadoutMath.predict_next_token_scores(self, v_prev2, v_prev1, v_cur, pos_idx, q_vec)

    def export_state_dict(self, use_fp16: bool = False) -> Dict[str, object]:
        """Экспортирует самодостаточное состояние дендритной матрицы, защищённого ядра `w_core_dend_out`,
        проекторов (`pos_phase_vecs`, `w_basal_branch`, `w_apical_branch`), аттракторов CA3 и цепочек эпизодов."""
        n_active = len(self.col_to_tok)
        w_dend_slice = self.w_dend_out[:, :n_active].detach().clone()
        w_core_slice = self.w_core_dend_out[:, :n_active].detach().clone()
        w_swr_slice = self.w_swr_dend_out[:, :n_active].detach().clone()
        if use_fp16:
            w_dend_slice = w_dend_slice.half()
            w_core_slice = w_core_slice.half()
            w_swr_slice = w_swr_slice.half()
        return {
            "readout_config": {"d_vec": self.d_vec, "d_dend": self.d_dend, "max_pos": self.max_pos},
            "col_to_tok": list(self.col_to_tok),
            "tok_to_col": {int(k): int(v) for k, v in self.tok_to_col.items()},
            "pos_phase_vecs": self.pos_phase_vecs.detach().clone(),
            "w_basal_branch": self.w_basal_branch.detach().clone(),
            "w_apical_branch": self.w_apical_branch.detach().clone(),
            "w_episode_trace": self.w_episode_trace.detach().clone(),
            "w_episode_trace_len": self.w_episode_trace_len.detach().clone(),
            "w_dend_out": w_dend_slice,
            "w_core_dend_out": w_core_slice,
            "w_swr_dend_out": w_swr_slice,
            "w_swr_poles": self.w_swr_poles.detach().clone(),
            "w_ca3_attractors": self.w_ca3_attractors.detach().clone(),
            "w_ca3_sdr": self.w_ca3_sdr.detach().clone(),
            "w_ca3_salience": self.w_ca3_salience.detach().clone(),
            "sdr_column_usage": self.sdr_column_usage.detach().clone(),
            "w_ca3_episode_links": {int(k): int(v) for k, v in self.w_ca3_episode_links.items()},
            "shield_state": export_runtime_tree(self.shield),
            "bank_router_state": export_runtime_tree(self.bank_router),
            "router_residual_bound": self.router_residual_bound,
            "module_presence": self._regulators.export_state_dict(),
        }

    def apply_checkpoint_overlay(self, payload: Dict[str, object]) -> None:
        """Наложение слоёв, добавленных после runtime-снимка, поверх восстановленного состояния.

        v19-путь загрузки восстанавливает организм ЦЕЛИКОМ из runtime-снимка и
        удаляет атрибуты, которых в снимке нет (`load_runtime_state`). Поэтому слои,
        появившиеся позже снимка — гиппокампальная траектория эпизода и
        расширенное окно позиций — восстанавливаются отдельно, из
        верхнеуровневых ключей payload. Для старых чекпоинтов их нет, и канал
        апикальной обратной связи просто молчит.
        """
        config = payload.get("readout_config", {}) or {}
        if "max_pos" in config:
            self.max_pos = max(int(self.max_pos), int(config["max_pos"]))
        pos_vecs = payload.get("pos_phase_vecs")
        if isinstance(pos_vecs, torch.Tensor):
            loaded_pos = pos_vecs.float().clone()
            if int(loaded_pos.shape[0]) < self.max_pos:
                # Достраиваем фазовые векторы расширенного окна. Детерминированно
                # (свой генератор), чтобы повторная загрузка давала то же самое.
                g_pad = torch.Generator().manual_seed(9001 + self.max_pos)
                extra = F.normalize(torch.randn(
                    self.max_pos - int(loaded_pos.shape[0]), self.d_vec, generator=g_pad), p=2, dim=-1)
                loaded_pos = torch.cat([loaded_pos, extra], dim=0)
            self.pos_phase_vecs = loaded_pos
        w_api = payload.get("w_apical_branch")
        if isinstance(w_api, torch.Tensor):
            self.w_apical_branch = w_api.float().clone()
        w_bas = payload.get("w_basal_branch")
        if isinstance(w_bas, torch.Tensor):
            self.w_basal_branch = w_bas.float().clone()
        trace = payload.get("w_episode_trace")
        trace_len = payload.get("w_episode_trace_len")
        if isinstance(trace, torch.Tensor) and isinstance(trace_len, torch.Tensor):
            loaded_trace = trace.to(torch.int32).clone()
            if int(loaded_trace.shape[1]) < self.max_pos:
                pad = torch.full((loaded_trace.shape[0], self.max_pos - int(loaded_trace.shape[1])),
                                 -1, dtype=torch.int32)
                loaded_trace = torch.cat([loaded_trace, pad], dim=1)
            self.w_episode_trace = loaded_trace
            self.w_episode_trace_len = trace_len.to(torch.int32).clone()
            n_trace = min(int(self.w_episode_trace.shape[0]), int(self.w_episode_trace_len.shape[0]))
            self.w_episode_trace = self.w_episode_trace[:n_trace].contiguous()
            self.w_episode_trace_len = self.w_episode_trace_len[:n_trace].contiguous()
        elif not hasattr(self, "w_episode_trace"):
            self.w_episode_trace = torch.zeros((0, self.max_pos), dtype=torch.int32)
            self.w_episode_trace_len = torch.zeros((0,), dtype=torch.int32)

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Восстанавливает состояние дендритной матрицы, защищённого ядра, проекторов, карт колонок и аттракторов CA3."""
        config = state.get("readout_config", {})
        for key in ("d_vec", "d_dend"):
            if key in config:
                setattr(self, key, int(config[key]))
        # Окно позиций НЕ сужается под чекпоинт: если текущая конфигурация шире,
        # чем у сохранённой модели, позиции достраиваются (см. ниже). Иначе
        # загрузка старого чекпоинта молча возвращала бы внешний обрез ответов.
        if "max_pos" in config:
            self.max_pos = max(int(self.max_pos), int(config["max_pos"]))
        col_to_tok_raw = state.get("col_to_tok")
        tok_to_col_raw = state.get("tok_to_col")
        if not isinstance(col_to_tok_raw, list) or not isinstance(tok_to_col_raw, dict):
            raise CheckpointCompatibilityError("Invalid or missing col_to_tok / tok_to_col in DendriticPlateauMatrixReadout state.")
        self.col_to_tok = [int(t) for t in col_to_tok_raw]
        self.tok_to_col = {int(k): int(v) for k, v in tok_to_col_raw.items()}
        n_active = len(self.col_to_tok)
        block = self.output_capacity_block
        cap = ((n_active + block - 1) // block) * block

        pos_vecs = state.get("pos_phase_vecs")
        w_bas = state.get("w_basal_branch")
        w_api = state.get("w_apical_branch")
        # Окно позиций, фазовые векторы, апикальная ветвь и гиппокампальная
        # траектория восстанавливаются единым наложением (см. apply_checkpoint_overlay).
        self.apply_checkpoint_overlay(state)

        w_dend = state["w_dend_out"]
        assert isinstance(w_dend, torch.Tensor)
        self.w_dend_out = torch.zeros(self.d_dend, cap, dtype=torch.float32)
        self.w_dend_out[:, :n_active] = w_dend.float()

        w_core = state.get("w_core_dend_out")
        self.w_core_dend_out = torch.zeros(self.d_dend, cap, dtype=torch.float32)
        if isinstance(w_core, torch.Tensor) and int(w_core.shape[1]) == n_active:
            self.w_core_dend_out[:, :n_active] = w_core.float()

        w_swr = state.get("w_swr_dend_out")
        self.w_swr_dend_out = torch.zeros(self.d_dend, cap, dtype=torch.float32)
        if isinstance(w_swr, torch.Tensor) and int(w_swr.shape[1]) == n_active:
            self.w_swr_dend_out[:, :n_active] = w_swr.float()

        swr_poles = state.get("w_swr_poles")
        if isinstance(swr_poles, torch.Tensor):
            self.w_swr_poles = swr_poles.float().clone()
        ca3_attr = state.get("w_ca3_attractors")
        if isinstance(ca3_attr, torch.Tensor):
            self.w_ca3_attractors = ca3_attr.float().clone()
        ca3_sdr = state.get("w_ca3_sdr")
        if isinstance(ca3_sdr, torch.Tensor):
            self.w_ca3_sdr = ca3_sdr.float().clone()
        ca3_sal = state.get("w_ca3_salience")
        if isinstance(ca3_sal, torch.Tensor):
            self.w_ca3_salience = ca3_sal.float().clone()
        if self.w_ca3_attractors.shape[0] > 0:
            n_ca3 = min(
                int(self.w_ca3_attractors.shape[0]),
                int(self.w_ca3_sdr.shape[0]),
                int(self.w_ca3_salience.shape[0]),
            )
            self.w_ca3_attractors = self.w_ca3_attractors[:n_ca3].contiguous()
            self.w_ca3_sdr = self.w_ca3_sdr[:n_ca3].contiguous()
            self.w_ca3_salience = self.w_ca3_salience[:n_ca3].contiguous()
        sdr_usage = state.get("sdr_column_usage")
        if isinstance(sdr_usage, torch.Tensor):
            self.sdr_column_usage = sdr_usage.float().clone()
        ep_links = state.get("w_ca3_episode_links")
        if isinstance(ep_links, dict):
            self.w_ca3_episode_links = {int(k): int(v) for k, v in ep_links.items()}
        if isinstance(state.get("shield_state"), dict):
            self.shield = load_runtime_tree(self.shield, state["shield_state"])
        if isinstance(state.get("bank_router_state"), dict):
            self.bank_router = load_runtime_tree(self.bank_router, state["bank_router_state"])
            if not hasattr(self.bank_router, "prune_threshold"):
                self.bank_router.prune_threshold = 1e-4
        self.router_residual_bound = float(state.get("router_residual_bound", 0.25))
        module_presence = state.get("module_presence", {})
        if isinstance(module_presence, dict):
            self._regulators.load_state_dict(module_presence)
        self._w_eff_cache = None

    def imprint_response_batch_vectorized(
        self,
        batch_items: List[Tuple[torch.Tensor, List[torch.Tensor], List[int], float, List[int], List[int]]],
        auxiliary_logits: Optional[List[List[torch.Tensor]]] = None,
        memory_query_vecs: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, object]:
        """Train a batch through sparse readout banks."""
        return ReadoutBanks.imprint_response_batch_vectorized(
            self, batch_items, auxiliary_logits, memory_query_vecs)

    def imprint_swr_composite_episode(
        self,
        q_swr_pole: torch.Tensor,
        response_vecs: List[torch.Tensor],
        response_ids: List[int],
        f_mod: float = 2.0,
    ) -> None:
        """Консолидация острых волн гиппокампа (`Sharp-Wave Ripple / SWR Replay`, Complementary Learning Systems):
        Записывает новую сложноподчинённую траекторию (сшитую из двух полюсов `CA3` через мост `BA44`)
        в эпизодическую матрицу дендритных плато `w_swr_dend_out` без интерференции с базовыми концептами `w_dend_out`!"""
        if not self.learning_enabled or len(response_ids) < 4:
            return
        with torch.no_grad():
            q_norm = F.normalize(q_swr_pole, p=2, dim=0)
            if self.w_swr_poles.shape[0] == 0:
                self.w_swr_poles = q_norm.unsqueeze(0).clone()
            elif float((self.w_swr_poles @ q_norm).max().item()) < 0.985:
                self.w_swr_poles = torch.cat([self.w_swr_poles, q_norm.unsqueeze(0)], dim=0)

            seq_ids = response_ids[: self.max_pos]
            for tid in seq_ids:
                self._ensure_token_col(tid)
            n_active = len(self.col_to_tok)
            v_zero = q_norm

            d_list: List[torch.Tensor] = []
            col_targets: List[int] = []
            for pos in range(len(seq_ids)):
                col_targets.append(self.tok_to_col[int(seq_ids[pos])])
                vp2 = v_zero if pos < 3 else response_vecs[pos - 3]
                vp1 = v_zero if pos < 2 else response_vecs[pos - 2]
                vc = v_zero if pos < 1 else response_vecs[pos - 1]
                d_list.append(self.compute_dendritic_plateau_state(vp2, vp1, vc, pos, q_norm))

            for d_nmda, target_col in zip(d_list, col_targets):
                logits = self._sparse_logits(d_nmda, self.w_swr_dend_out, n_active) * 12.0
                err = -torch.softmax(logits, dim=0)
                err[target_col] += 1.0
                self._apply_local_update(self.w_swr_dend_out, d_nmda, err, 0.85 * float(f_mod))
