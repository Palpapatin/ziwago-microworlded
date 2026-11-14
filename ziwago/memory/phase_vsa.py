"""Фазово-голографический синтаксический биндер (Ziwago-Prime v2.5: Grammar & Roles).

Реализует нейробиологический контур синтаксиса (поля Брока BA44/45 ↔ SMA ↔ Таламус):
1. Синтаксический слот-фазовый счётчик внутри предложения (Ding et al. 2016;
   Lisman & Jensen 2013):
   Каждая позиция в предложении (после точки/запятой/начала фразы) активирует
   свою жесткую структурную роль R_k ∈ C^D на единичной окружности:
     Role 0: DET / INIT (артикль / зачин: "the", "a", "once")
     Role 1: ADJ / MOD  (прилагательное / модификатор: "little", "big", "happy")
     Role 2: SUBJ       (субъект / герой: "girl", "dog", "cat", "lily")
     Role 3: VERB       (действие / предикат: "sat", "ran", "went", "saw")
     Role 4: PREP / AUX (предлог / связка: "to", "in", "with", "on")
     Role 5: OBJ        (объект / место: "park", "house", "tree", "ball")
     Role 6: PUNCT      (граница клаузы / точка / запятая)

2. Двойной выход:
   - `role_vec` [B, n_slots]: one-hot + фазовый след ОЖИДАЕМОЙ СЛЕДУЮЩЕЙ синтаксической
     роли (для базально-апикального гейтинга моторной коры — в слоте Действия могут
     побеждать только глаголы, а в слоте Субъекта — существительные!).
   - `context_vec` [B, D]: распакованный через PhaseVSA (Frame ⊙ conj(R_k))
     голографический вектор содержания текущего предложения (кто именно был в Role 0..2),
     удерживающий семантическую связь между Субъектом и Предикатом!

Честные границы (2026-11-13):
  * `theta_hz`/`gamma_hz` — реальные часы: θ-фаза выводится из `t_ms*theta_hz`, γ-индекс
    из `t_ms*gamma_hz` (раньше оба параметра не использовались, а `t_ms` был мёртв);
  * роль выбирается конкуренцией Лотки–Вольтерры с асимметричным торможением
    (`_wlc_step`), а не argmax по таблице переходов: популяции сосуществуют, а без
    стимула победителя НЕТ (`gamma_slot == -1`);
  * `slot_age` работает: буфер слотов пассивен и стареет (`slot_decay`), возраст
    обнуляется только записью в свой слот.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch


@dataclass
class PhaseStoryOut:
    theta_phase: float          # текущая фаза θ ∈ [0, 2π) из часов theta_hz
    gamma_slot: int             # роль-победитель ∈ [0, K-1]; -1 = победителя нет
    next_role: int              # индекс ОЖИДАЕМОЙ СЛЕДУЮЩЕЙ роли ∈ [0, K-1]
    role_vec: torch.Tensor      # [B, K] вектор активной синтаксической роли (BA44/SMA)
    context_vec: torch.Tensor   # [B, D] вещественный фазовый контекст (PhaseVSA unbind)
    salience: float             # сила записи в текущем слоте
    boundary_reset: bool        # флаг сброса фразы (для таламического LTS-burst)
    slot_clock: int = 0         # слот θ-γ мультиплексирования из часов
    gamma_index: int = -1       # индекс γ-окна из часов gamma_hz
    role_population: Optional[torch.Tensor] = None  # состояние конкуренции BA44 [K]
    slot_age: Optional[torch.Tensor] = None         # возраст слотов клаузы [K]


class ThetaGammaStoryBinder:
    """Синтаксический тета-гамма мультиплексор + эмерджентный гетероклинический
    аттрактор ролей Брока BA44/45 (Winnerless Competition, Rabinovich et al. 2001)
    + комплексная PhaseVSA память ролей.
    """

    def __init__(self, dim: int, n_slots: int = 7,
                 theta_hz: float = 8.0, gamma_hz: float = 40.0,
                 decay: float = 0.96, seed: int = 42):
        self.dim = dim
        self.n_slots = n_slots
        self.theta_hz = theta_hz
        self.gamma_hz = gamma_hz
        self.decay = decay
        g = torch.Generator().manual_seed(seed)
        # Ортогональные фазовые векторы синтаксических ролей R_k ∈ C^D на единичной окружности
        role_angles = (torch.rand(n_slots, dim, generator=g) * 2.0 - 1.0) * math.pi
        self.roles = torch.complex(torch.cos(role_angles), torch.sin(role_angles))
        # Проекция спайкового вектора в фазовые углы филлера
        self.w_angle = torch.randn(dim, dim, generator=g) / math.sqrt(max(1, dim))
        # P2: Гетероклиническая матрица синтаксических переходов W_trans[next, cur] (BA44/SMA):
        # задаёт асимметричную конкуренцию без победителя (Winnerless Competition) и допускает
        # естественные грамматические переходы с пропуском опциональных модификаторов
        # (например, DET(0) → SUBJ(2) без ADJ(1), или VERB(3) → OBJ(5)/PUNCT(6))!
        self.w_trans = torch.full((n_slots, n_slots), -0.35)
        self.w_trans.fill_diagonal_(-1.20)  # самоторможение роли (не застревать в одной роли)
        for k in range(n_slots):
            self.w_trans[(k + 1) % n_slots, k] = 1.05
        if n_slots >= 7:
            self.w_trans[2, 0] = 0.82   # DET(0) → SUBJ(2) (пропуск опционального ADJ)
            self.w_trans[3, 2] = 1.15   # SUBJ(2) → VERB(3) (ядро предикации)
            self.w_trans[5, 3] = 0.80   # VERB(3) → OBJ(5) (прямое дополнение без предлога)
            self.w_trans[6, 3] = 0.72   # VERB(3) → PUNCT(6) (непереходный глагол в конце фразы)
            self.w_trans[6, 5] = 1.10   # OBJ(5) → PUNCT(6)
            self.w_trans[0, 6] = 1.20   # PUNCT(6) → DET/INIT(0)
            self.w_trans[2, 6] = 0.78   # PUNCT(6) → SUBJ(2) (начало фразы с имени собственного)
        # Пластичная матрица распознавания синтаксической роли по сенсорному/кортикальному следу [K, D]
        self.w_lex_role = torch.zeros(n_slots, dim)
        # Текущее состояние популяции синтаксических нейронов BA44 [K] — не one-hot,
        # а непрерывная популяция конкуренции (может быть несколько активных ролей).
        self.role_state = torch.zeros(n_slots)
        self.role_state[0] = 1.0
        self.prev_role: int = n_slots - 1
        # Голографический фрейм текущего предложения и глобального сюжета [D]
        self.clause_frame = torch.zeros(dim, dtype=torch.complex64)
        self.story_frame = torch.zeros(dim, dtype=torch.complex64)
        # Память субъекта истории (Role 0..2), которая переживает границы предложений
        self.subject_trace = torch.zeros(dim)
        # Буфер слотов текущего предложения [K, D]
        self.slots = torch.zeros(n_slots, dim)
        self.slot_age = torch.zeros(n_slots)
        self.t_ms: float = 0.0
        self.clause_pos: int = 0
        self._last_boundary: bool = False
        self._last_x_norm: Optional[torch.Tensor] = None
        self._gamma_index: int = -1
        # Параметры конкуренции (Lotka–Volterra): самоторможение 1.0, латеральное — wlc_inh
        self.wlc_inh: float = 1.4
        self.wlc_dt: float = 0.15      # шаг интеграции на один γ-такт
        self.u_winner_min: float = 0.05
        self.silence_floor: float = 1e-3
        self.slot_decay: float = 0.97
        self.slot_tau_age: float = 50.0   # старение слота ускоряет его распад

    # ------------------------------------------------------------------
    def _ensure_device(self, device, dtype) -> None:
        """Все буферы живут там же, где вход (CUDA-падение `learn_role`)."""
        cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.roles = self.roles.to(device=device, dtype=cdtype)
        self.w_angle = self.w_angle.to(device=device, dtype=dtype)
        self.w_lex_role = self.w_lex_role.to(device=device, dtype=dtype)
        self.w_trans = self.w_trans.to(device=device, dtype=dtype)
        self.role_state = self.role_state.to(device=device, dtype=dtype)
        self.slots = self.slots.to(device=device, dtype=dtype)
        self.slot_age = self.slot_age.to(device=device, dtype=dtype)
        self.subject_trace = self.subject_trace.to(device=device, dtype=dtype)
        self.clause_frame = self.clause_frame.to(device=device, dtype=cdtype)
        self.story_frame = self.story_frame.to(device=device, dtype=cdtype)

    def reset_state(self) -> None:
        self.clause_frame.zero_()
        self.story_frame.zero_()
        self.subject_trace.zero_()
        self.slots.zero_()
        self.slot_age.zero_()
        self.role_state.zero_()
        self.role_state[0] = 1.0
        self.prev_role = self.n_slots - 1
        self.t_ms = 0.0
        self.clause_pos = 0
        self._last_boundary = False
        self._last_x_norm = None
        self._gamma_index = -1

    def notify_boundary(self) -> None:
        """Сигнал конца предложения / клаузы (точка, восклицательный знак, запятая):
        сбрасывает синтаксический аттрактор ролей в Role 0 (Субъект/Зачин)
        и переносит тему предложения в долговременный story_frame."""
        self.story_frame = 0.85 * self.story_frame + 0.15 * self.clause_frame
        mag = self.story_frame.abs().clamp(min=1e-6)
        self.story_frame = self.story_frame / mag
        self.clause_frame.mul_(0.25)
        self.role_state.zero_()
        self.role_state[0] = 1.0
        self.prev_role = self.n_slots - 1
        self.clause_pos = 0
        self._last_boundary = True

    def _to_phasor(self, x: torch.Tensor) -> torch.Tensor:
        """Спайковый/активностный вектор [D] → комплексный фазор exp(i·ψ) ∈ C^D."""
        rms = x.pow(2).mean().sqrt().clamp(min=1e-6)
        ang = torch.tanh((x / rms) @ self.w_angle.t()) * math.pi
        return torch.complex(torch.cos(ang), torch.sin(ang))

    # ------------------------------------------------------------------
    def _wlc_substeps(self) -> int:
        """Сколько γ-тактов укладывается в один θ-такт (γ/θ): столько подшагов
        конкуренции успевает пройти внутри одного θ-слота (Buzsáki 2010; Lisman 2013)."""
        return max(1, int(round(float(self.gamma_hz) / max(1e-6, float(self.theta_hz)))))

    def _wlc_step(self, u: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
        """Winnerless competition: du_k = u_k·(σ_k − Σ_j A_kj·u_j).

        A_kk = 1 (самоторможение), A_kj = wlc_inh·(1 − W_trans[k,j]/1.2) для k≠j:
        активная роль j тормозит «чужие» роли сильно, а своего естественного
        преемника — слабо. Поэтому популяции сосуществуют, а траектория может
        уходить по цепочке (метастабильные состояния), не имея единственного
        устойчивого победителя. Число подшагов задаётся отношением γ/θ: чем больше
        γ-тактов на θ-цикл, тем глубже конкуренция успевает сойтись внутри слота.
        """
        wt = (self.w_trans.clamp(min=0.0) / 1.2).to(u.device)
        a = self.wlc_inh * (1.0 - wt)
        a.fill_diagonal_(1.0)
        for _ in range(self._wlc_substeps()):
            inhib = a @ u
            du = u * (drive - inhib)
            u = (u + self.wlc_dt * du).clamp(min=0.0, max=1.0)
        return u

    def learn_role(self, x_spikes: torch.Tensor, role_idx: int,
                   f_mod: float = 1.0, eta: float = 0.08) -> None:
        """P2: Локальное обучение синтаксического распознавателя BA44 (w_lex_role)
        и гетероклинической матрицы переходов (w_trans) по опыту."""
        if f_mod <= 0.0 or not (0 <= role_idx < self.n_slots):
            return
        x_mean = x_spikes.detach().mean(0) if x_spikes.dim() == 2 else x_spikes.detach()
        if x_mean.numel() != self.dim:
            if x_mean.numel() > self.dim:
                x_mean = x_mean[:self.dim]
            else:
                x_mean = torch.nn.functional.pad(x_mean, (0, self.dim - x_mean.numel()))
        # Матрица распознавания обязана жить на устройстве/типе входа, иначе CUDA-вход
        # падал «expected scalar type Float but found Double» ещё до первого шага.
        self._ensure_device(x_mean.device, x_mean.dtype)
        x_norm = x_mean / x_mean.norm().clamp(min=1e-6)
        lr = eta * max(0.2, min(2.0, float(f_mod)))
        target = torch.full((self.n_slots,), -0.15, device=x_norm.device, dtype=x_norm.dtype)
        target[role_idx] = 1.0
        pred = self.w_lex_role @ x_norm
        self.w_lex_role = (self.w_lex_role + lr * torch.outer(target - pred, x_norm)).clamp(-1.5, 2.0)
        # STDP синтаксического перехода prev_role → role_idx
        if 0 <= self.prev_role < self.n_slots and self.prev_role != role_idx:
            self.w_trans[role_idx, self.prev_role] = min(
                1.6, float(self.w_trans[role_idx, self.prev_role]) + 0.03 * lr
            )

    # ------------------------------------------------------------------
    def step(self, x_spikes: torch.Tensor, ach: float = 0.5,
             ne: float = 0.3, dt: float = 1.0,
             is_boundary: bool = False,
             role_hint: Optional[torch.Tensor] = None,
             learn: bool = True) -> PhaseStoryOut:
        """Один шаг: распознаёт синтаксическую роль текущего слова через
        гетероклинический аттрактор Winnerless Competition (W_trans + лексический вход),
        связывает слово с его ролью в PhaseVSA и выдаёт вектор ожидаемых следующих ролей."""
        self._ensure_device(x_spikes.device, x_spikes.dtype)
        self.t_ms += dt
        b_sz = x_spikes.shape[0] if x_spikes.dim() == 2 else 1
        x_mean = x_spikes.mean(0) if x_spikes.dim() == 2 else x_spikes
        if x_mean.numel() != self.dim:
            if x_mean.numel() > self.dim:
                x_mean = x_mean[:self.dim]
            else:
                x_mean = torch.nn.functional.pad(x_mean, (0, self.dim - x_mean.numel()))
        x_norm = x_mean.detach() / x_mean.detach().norm().clamp(min=1e-6)
        self._last_x_norm = x_norm
        act_level = float(x_mean.abs().mean().item())

        # --- часы: θ задаёт фазу, γ — индекс гамма-окна -------------------
        frac = (self.t_ms * self.theta_hz * 1e-3) % 1.0
        theta_phase = 2.0 * math.pi * frac
        slot_clock = int(frac * self.n_slots) % self.n_slots
        self._gamma_index = int(self.t_ms * self.gamma_hz * 1e-3) % self.n_slots

        # Пассивное старение буфера: скорость забывания зависит от ВОЗРАСТА слота
        # (старый след распадается быстрее — интерференция рабочей памяти).
        age_gain = 1.0 + self.slot_age / max(1e-6, self.slot_tau_age)
        self.slots *= (self.slot_decay ** (dt * age_gain)).unsqueeze(1)
        self.slot_age = self.slot_age + dt

        hint = None
        if role_hint is not None and role_hint.numel() == self.n_slots:
            rh = role_hint.detach().flatten().to(device=x_norm.device, dtype=x_norm.dtype)
            if float(rh.max().item()) > 0.0:
                hint = rh / rh.max().clamp(min=1e-6)

        if is_boundary:
            self.notify_boundary()
            cur_role = self.n_slots - 1  # PUNCT / BOUNDARY роль
            next_role = 0                # После границы предложения ждём Role 0 (Зачин/Субъект)
            boundary_flag = True
            next_drive = torch.zeros(self.n_slots, device=x_spikes.device)
            next_drive[0] = 1.0
            if self.n_slots >= 3:
                next_drive[2] = 0.45     # Допускаем старт предложения сразу с Субъекта
        else:
            if self.clause_pos >= self.n_slots:
                self.notify_boundary()
            trans_prior = torch.relu(self.w_trans[:, self.prev_role])
            # Лексико-кортикальное распознавание роли текущего слова
            lex_cue = torch.relu(self.w_lex_role @ x_norm)
            if hint is not None:
                lex_cue = 0.45 * lex_cue + 0.85 * hint
            non_punct_slots = max(1, self.n_slots - 1)

            if act_level <= self.silence_floor and hint is None:
                # Нет стимула — нет входа в конкуренцию: победителя быть не может.
                cur_role = -1
                self.role_state.mul_(0.5)          # пассивный распад популяции
            else:
                allowed_mask = (trans_prior[:non_punct_slots] > 0.10).to(lex_cue.dtype)
                drive = torch.zeros(self.n_slots, device=lex_cue.device, dtype=lex_cue.dtype)
                drive[:non_punct_slots] = (
                    lex_cue[:non_punct_slots] * allowed_mask * 1.60
                    + 0.20 * trans_prior[:non_punct_slots])
                # Топ-даун приор ожидаемой роли (а не жёсткое присвоение «по таблице»)
                prior_role = int(trans_prior[:non_punct_slots].argmax().item())
                drive[prior_role] += 0.25
                if self.clause_pos == 0:
                    drive[0] += 0.60               # зачин клаузы: сильный, но не жёсткий приор
                self.role_state = self._wlc_step(self.role_state.to(drive.dtype), drive)
                cur_role = (int(self.role_state.argmax().item())
                            if float(self.role_state.max().item()) >= self.u_winner_min else -1)

            if cur_role >= 0:
                self.clause_pos = cur_role + 1
                if learn and act_level > 0.01:
                    self.learn_role(x_mean, cur_role, f_mod=0.5 + 0.5 * ach)
                self.prev_role = cur_role
                next_drive = torch.relu(self.w_trans[:, cur_role]).to(x_spikes.device)
                next_role = (int(next_drive.argmax().item())
                             if float(next_drive.max().item()) > 0.0
                             else self.clause_pos % self.n_slots)
            else:
                next_drive = trans_prior.to(x_spikes.device)
                next_role = (int(next_drive.argmax().item())
                             if float(next_drive.max().item()) > 0.0 else 0)
            boundary_flag = self._last_boundary
            self._last_boundary = False

        salience = act_level * (0.5 + 0.5 * ach + 0.5 * ne)
        if cur_role >= 0 and salience > 0.01:
            self.slots[cur_role] = 0.2 * self.slots[cur_role] + 0.8 * x_mean
            self.slot_age[cur_role] = 0.0
            # Если это группа Субъекта (Role 0..2), фиксируем субъект предложения
            if cur_role <= 2 and not is_boundary:
                self.subject_trace = 0.4 * self.subject_trace + 0.6 * x_mean
            filler = self._to_phasor(x_mean)
            bound = self.roles[cur_role] * filler
            # Жесткое голографическое связывание (PhaseVSA binding)
            self.clause_frame = self.clause_frame * self.decay + bound
            mag = self.clause_frame.abs().clamp(min=1e-6)
            self.clause_frame = self.clause_frame / mag

        # Распаковка (unbind) содержания ролей через сопряжённые векторы ролей conj(R_k):
        # Мы распаковываем Субъект (Role 0..2) и связываем его с ожидаемой следующей ролью!
        unbound_clause = (self.clause_frame.unsqueeze(0) * self.roles.conj()).real  # [K, D]
        unbound_story = (self.story_frame.unsqueeze(0) * self.roles.conj()).real    # [K, D]
        # Контекст для следующего слова: комбинация распакованного Субъекта клаузы
        # и фазовой сигнатуры ожидаемой роли R_{next_role}.real
        subj_unbound = 0.5 * unbound_clause[:3].mean(0) + 0.5 * self.subject_trace
        role_probe = self.roles[next_role].real
        ctx_1d = 0.55 * subj_unbound + 0.25 * unbound_story.mean(0) + 0.20 * role_probe
        ctx = ctx_1d.unsqueeze(0).expand(b_sz, -1).contiguous()

        # Вектор синтаксической роли [B, K]: указывает моторному пулу следующую роль
        # и плавное перекрытие текущего гамма-окна
        r_vec = torch.zeros(b_sz, self.n_slots, device=x_spikes.device, dtype=ctx.dtype)
        r_vec[:, next_role] = 1.0
        if 0 <= cur_role < self.n_slots:
            r_vec[:, cur_role] += 0.35

        return PhaseStoryOut(theta_phase=theta_phase, gamma_slot=cur_role,
                             next_role=next_role, role_vec=r_vec,
                             context_vec=ctx, salience=salience,
                             boundary_reset=boundary_flag,
                             slot_clock=slot_clock,
                             gamma_index=self._gamma_index,
                             role_population=self.role_state.clone(),
                             slot_age=self.slot_age.clone())
