"""Глобальное рабочее пространство (§II.4.1): Modern Hopfield Network.

Слоты K=4 (Cowan 2001) — когерентные пакеты информации d_GW.
Ignition: нелинейная фазовая транзиция (Dehaene & Changeux 2011):
  A = σ(γ_ign·(S - S_th)),  S = метрика внимания кандидата
  [NE]↑ опускает порог: s_th_eff = s_th·(0.75 − 0.25·[NE]) ⇒ резче транзиция
  (ignition выше на тех же кандидатах).

Извлечение (Ramsauer et al. 2021):
  ξ_new = 0.75·ξ + 0.25·h(ξ),  h(ξ) = X·softmax(β_GW·Xᵀξ) — шаг к аттрактору
  E(ξ)  = ‖ξ − h(ξ)‖²           — Lyapunov-невязка (монотонно ↓): ноль только в
          неподвижной точке Хопфида. Шаг принимается лишь если E не возросла,
          иначе итерации останавливаются на предыдущем ξ (спуск по Ляпунову).
  Квадратичная форма −½ξᵀXᵀXξ Lyapunov-функцией этого шага не является: измерено
  dE/dt|₀ > 0 (шаг уводит от главного собственного направления XᵀX).

Бродкаст: чтение K-слотов всеми зонами (дешёвое чтение).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from ..config import WorkspaceCfg


@dataclass
class GWOut:
    broadcast: torch.Tensor      # [d_gw] симфонический аккорд (все зоны вместе)
    content: List[torch.Tensor]  # активные партитуры слотов (SENS, ASSOC, PFC, CA1)
    ignition: float              # A ∈ [0,1] глобальная синхронизация
    winner_zone: int             # ведущая солирующая зона (для обратной совместимости AST)
    energy: float                # Lyapunov-энергия аккорда (сумма невязок слотов)
    coherence_weights: Optional[List[float]] = None  # веса фазовой когерентности зон
    slot_activations: List[float] = field(default_factory=list)      # A_i слотов, по которым пишутся слоты
    slot_energy_traces: List[List[float]] = field(default_factory=list)  # энергии по шагам, на слот
    # Номера слотов, которым соответствуют трейсы выше. Без них индексы
    # «сдвинуты»: слоты с gate <= 1e-3 пропускаются, и по трейсу нельзя
    # было сказать, какому слоту принадлежит энергия.
    slot_energy_trace_slots: List[int] = field(default_factory=list)


class GlobalWorkspace:
    """Симфоническое Рабочее Пространство (Ziwago-Prime v4, Узел 2):
    Параллельная таламо-кортикальная интеграция через Modern Hopfield резонанс
    и фазовую когерентность LFP (Communication-Through-Coherence, Fries 2015)
    без подавления слабых сигналов."""

    def __init__(self, cfg: WorkspaceCfg):
        self.cfg = cfg
        self.x = torch.zeros(cfg.n_slots, cfg.d_gw)   # слоты-партитуры зон
        self.gate = torch.zeros(cfg.n_slots)          # резонансный вес каждого потока
        self.ignition = 0.0
        self.last_coherence: List[float] = []
        self.energy_hist: List[float] = []            # по значению на каждый compete
        self.telemetry: dict = {
            "energy": 0.0,
            "energy_consumer_connected": False,
            "energy_consumer_points": [],
        }

    def _telemetry(self) -> dict:
        tel = getattr(self, "telemetry", None)
        if not isinstance(tel, dict):
            tel = {
                "energy": 0.0,
                "energy_consumer_connected": False,
                "energy_consumer_points": [],
            }
            self.telemetry = tel
        else:
            tel.setdefault("energy_consumer_connected", False)
            tel.setdefault("energy_consumer_points", [])
        return tel

    # ------------------------------------------------------------------
    def compete(self, candidates: List[torch.Tensor],
                ne: float,
                lfp_phases: Optional[List[float]] = None,
                beta_boost: float = 0.0) -> GWOut:
        """Интегрирует параллельные потоки от всех зон (SENS, ASSOC, PFC, CA1)
        в единый симфонический аккорд без уничтожения слабых сигналов."""
        c = self.cfg
        if not candidates:
            self.x *= c.decay
            self.energy_hist.append(0.0)      # вызов compete без кандидатов тоже в истории
            self._telemetry()["energy"] = 0.0
            return GWOut(torch.zeros(c.d_gw), [], 0.0, -1, 0.0, [])
        stack = torch.stack([c_.detach().float().flatten()[:c.d_gw]
                             for c_ in candidates])
        stack = torch.nn.functional.pad(
            stack, (0, max(0, c.d_gw - stack.shape[1])))[:, :c.d_gw]
        n_cand = stack.shape[0]

        # 1. Нормировка энергии пакета зоны + фазовая когерентность LFP (PLV):
        # разреженные бинарные спайки коры (SENS, ASSOC, PFC) и плотный вектор CA1
        # приводятся к сбалансированной шкале громкости оркестра (~0.8..1.2)!
        raw_norms = stack.norm(dim=1, keepdim=True).clamp(min=1e-6)
        has_signal = (raw_norms.squeeze(1) > 1e-4).float()
        stack_normed = (stack / raw_norms) * has_signal.unsqueeze(1)

        mean_field = stack_normed.mean(dim=0, keepdim=True)
        cos_coh = torch.nn.functional.cosine_similarity(
            stack_normed, mean_field.expand_as(stack_normed), dim=1).clamp(min=0.0)
        if lfp_phases is not None and len(lfp_phases) == n_cand:
            import math
            mean_phi = sum(lfp_phases) / max(1, len(lfp_phases))
            plv = torch.tensor([0.5 * (1.0 + math.cos((p - mean_phi) * 0.5)) for p in lfp_phases],
                               dtype=stack.dtype, device=stack.device)
            coh_gain = (0.55 + 0.45 * cos_coh) * (0.65 + 0.35 * plv)
        else:
            coh_gain = 0.65 + 0.35 * cos_coh

        # 2. Мягкая активация каждого параллельного потока (БЕЗ argmax-убийства остальных зон!):
        # фазовая когерентность LFP управляет весом партии каждой зоны в Симфоническом Аккорде!
        scores = has_signal * coh_gain
        beta = (c.beta_gw + float(beta_boost)) * (1.0 + 0.5 * ne)
        # [NE]↑ ⇒ порог ниже ⇒ резче транзиция и выше ignition на тех же кандидатах.
        s_th_eff = c.s_th * (0.75 - 0.25 * float(ne))
        a_raw = torch.sigmoid(c.gamma_ign * 0.5 * (scores - s_th_eff)) * has_signal
        a = torch.maximum(a_raw, 0.25 * coh_gain * has_signal)
        # ignition и winner_zone — строго по тем активациям, которыми пишутся слоты.
        slot_acts = [float(v) for v in a[:min(n_cand, c.n_slots)]]
        self.ignition = max(slot_acts) if slot_acts else 0.0
        lead_zone = (int(max(range(len(slot_acts)), key=lambda i: slot_acts[i]))
                     if slot_acts and self.ignition > 0.5 else -1)
        self.last_coherence = [round(float(w), 4) for w in a.tolist()]

        # 3. Параллельная запись: каждая зона обновляет свой собственный слот в симфонии!
        for i in range(min(n_cand, c.n_slots)):
            # Плавное вливание нового сигнала зоны в её партитуру слота i
            alpha_slot = float(max(0.25, a[i].item()))
            self.x[i] = (1.0 - alpha_slot) * (self.x[i] * c.decay) + alpha_slot * stack_normed[i]
            self.gate[i] = max(float(self.gate[i] * c.decay), float(a[i].item()))

        # 4. Извлечение как спуск по Ляпунову: E(ξ)=‖ξ−h(ξ)‖² (см. docstring модуля).
        #    Новое ξ принимается только если E не возросла; иначе — стоп на предыдущем.
        def _hopfield_residual(xi_: torch.Tensor) -> float:
            h_ = self.x.t() @ torch.softmax(beta * (self.x @ xi_), dim=0)
            return float(((xi_ - h_) ** 2).sum())

        slot_energy_traces: List[List[float]] = []
        slot_energy_trace_slots: List[int] = []
        slot_energies: List[float] = []
        for s in range(c.n_slots):
            if self.gate[s] <= 1e-3:
                continue
            xi = self.x[s].clone()
            trace = [_hopfield_residual(xi)]
            for _ in range(c.retrieval_steps):
                logits = beta * (self.x @ xi)
                xi_harm = self.x.t() @ torch.softmax(logits, dim=0)
                xi_new = 0.75 * xi + 0.25 * xi_harm
                e_new = _hopfield_residual(xi_new)
                if e_new > trace[-1] + 1e-12:   # энергия возросла ⇒ останов итераций
                    break
                xi = xi_new
                trace.append(e_new)
            self.x[s] = xi
            slot_energy_traces.append(trace)
            slot_energy_trace_slots.append(int(s))
            slot_energies.append(trace[-1])
        energy = float(sum(slot_energies))
        self.energy_hist.append(energy)         # полная история, без обрезки
        self._telemetry()["energy"] = energy

        # 5. Симфонический бродкаст: взвешенная сумма всех одновременно звучащих партий
        broadcast = (self.x * self.gate.unsqueeze(1)).sum(0)
        return GWOut(broadcast=broadcast,
                     content=[self.x[i] for i in range(c.n_slots)
                              if self.gate[i] > 1e-3],
                     ignition=self.ignition, winner_zone=lead_zone,
                     energy=energy, coherence_weights=self.last_coherence,
                     slot_activations=slot_acts,
                     slot_energy_traces=slot_energy_traces,
                     slot_energy_trace_slots=slot_energy_trace_slots)
