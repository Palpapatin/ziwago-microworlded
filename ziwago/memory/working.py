"""Рабочая память: activity-silent хранилище (Lundqvist 2018) + NMDA-бамп.

PFC держит информацию не в персистентном огне, а в кратковременной
синаптической фацилитации STP (u_syn-прочность потенциированных связей)
— энергетически дешевле: спайков нет, память жива в весах-мостах.
Пробный спайк («зонд») вызывает повторную активацию именно того
паттерна — чтение. Во время активного удержания медленный NMDA
(τ≈100 мс, Compte 2000) поддерживает бамп рекуррентным током `facil @ bump`,
а не хранением копии последней записи.

Честные границы (2026-11-13):
  * чтение (`probe`) идёт ТОЛЬКО по фацилитации: пустой зонд не возвращает
    ничего, поэтому «эхо последней записи» невозможно по построению;
  * k-WTA выбирает не более k единиц: topk по значениям (а не порог `>= k-го`,
    который на ничьей отдавал больше k); нулевой отклик активностью не считается.
  * NMDA-бамп затухает с τ≈100 мс; при `active_hold=True` он удерживается
    нормированным рекуррентным током и без внешнего входа не гаснет.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class WMOut:
    reactivated: torch.Tensor    # [B,N] считанный паттерн
    stored_strength: float
    active_count: int = 0


class SilentWorkingMemory:
    """STP-хранилище: запись через фацилитацию, чтение — пробный спайк."""

    def __init__(self, n: int, tau_facil: float = 3000.0,
                 nmda_tau: float = 100.0, probe_gain: float = 0.5,
                 threshold: float = 0.0):
        self.n = n
        self.tau_facil = tau_facil      # фацилитация живёт секунды
        self.nmda_tau = nmda_tau        # бамп-удержание 100 мс
        self.probe_gain = float(probe_gain)
        self.threshold = float(threshold)
        self.facil = torch.zeros(n, n)  # STP-потенциация
        self.bump = torch.zeros(n)      # активный NMDA-бамп
        self.writes = 0

    # ------------------------------------------------------------------
    def write(self, pattern: torch.Tensor, strength: float = 1.0) -> None:
        """Спайк-паттерн [B,N] → фацилитация его внутренних связей."""
        p = pattern.float().mean(0) if pattern.dim() == 2 else pattern.float()
        self.facil += strength * torch.outer(p, p)
        self.facil.clamp_(0.0, 3.0)
        self.bump = (self.bump + strength * p).clamp(0.0, 1.0)
        self.writes += 1

    # ------------------------------------------------------------------
    def tick(self, dt: float = 1.0, active_hold: bool = False) -> None:
        """Фацилитация тает медленно; бамп держится на NMDA-τ.

        При `active_hold=True` бамп поддерживается рекуррентным током через
        фацилитированные связи (нормировка на максимальную колонку — иначе
        амплитуда зависела бы от числа записей). Без удержания — пассивная утечка.
        """
        self.facil *= 1.0 - dt / self.tau_facil
        if active_hold and float(self.facil.sum()) > 0.0:
            scale = self.facil.sum(0).max().clamp(min=1e-6)
            drive = (self.facil @ self.bump) / scale
            leak = min(1.0, dt / self.nmda_tau)
            self.bump = self.bump * (1.0 - leak) + leak * drive
        else:
            self.bump *= 1.0 - dt / 20.0   # без удержания бамп гаснет
        self.bump.clamp_(0.0, 1.0)

    # ------------------------------------------------------------------
    def probe(self, cue: torch.Tensor, k: int = 8) -> WMOut:
        """Пробный спайк → реактивация сохранённого паттерна (ровно k-WTA).

        Отклик = фацилитация × нормированный зонд (NMDA-бамп даёт только гейн,
        а не аддитивное эхо). Пустой зонд → нулевой отклик → ноль активных единиц.
        """
        c = cue.float().mean(0) if cue.dim() == 2 else cue.float()
        c = c / c.norm().clamp(min=1e-6)
        resp = (self.facil @ c) * (1.0 + self.probe_gain * self.bump)
        k_eff = max(1, min(int(k), self.n))
        top = torch.topk(resp, k_eff)
        mask = torch.zeros_like(resp)
        mask[top.indices] = (top.values > self.threshold).float()
        return WMOut(mask.unsqueeze(0), float(self.facil.abs().mean()),
                     int(mask.sum().item()))
