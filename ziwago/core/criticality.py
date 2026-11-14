"""Мониторинг критичности и лавин (§1.5, §II.1.5; math.md §5).

MR-оценщик σ̂ (Wilting & Priesemann 2018): автокорреляции
r_k = m^k → фит log r_k = k·log m̂ по истории активности
(устойчив к субсэмплингу). Целевой режим по in-vivo данным:
σ̂ ≈ 0.95-0.99 («ревербирующий», слегка субкритичный), τ_int=Δt/(1−m̂)
конечен и перестраиваем — не σ=1.

Лавины: эпизод A(t)>θ_aval между тихими кадрами; P(S)~S^{-3/2},
P(T)~T^{-2}; новизна — лавина выше скользящего квантиля.
Динамический диапазон: Δ = 10·log10(F_0.9/F_0.1) (Kinouchi-Copelli).

ЧЕСТНОСТЬ ОЦЕНКИ (трек A, v22):
  * σ̂ — ИЗМЕРЕНИЕ (MR-оценка по суррогатной автокорреляции). Никакого клампа
    к «~1»: оценка может быть и 0.6, и 1.4, и это её честное значение.
  * «Цель σ≈1» — ОТДЕЛЬНАЯ величина (`target` / `HomeostasisCfg.target_sigma`),
    с которой сравнивается измеренное σ̂ (см. `ziwago/homeostasis/controller.py`).
  * Единица активности во всей очереди автокорреляции — СУММА спайков зоны за тик
    (то же, что кладётся в `zone_activity`); статистика `zone_activity_mean`
    использует ту же единицу, а не среднее по нейронам.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import torch


@dataclass
class Avalanche:
    size: float
    duration_ms: int
    novel: bool


class CriticalityMonitor:
    """σ̂ по автокорреляционной регрессии + статистика лавин."""

    def __init__(self, k_max: int = 20, novel_quantile: float = 0.95,
                 target_sigma: float = 0.97):
        self.k_max = k_max
        self.novel_q = novel_quantile
        self.target = target_sigma
        self.activity: Deque[float] = deque(maxlen=2000)
        self.aval_sizes: Deque[float] = deque(maxlen=512)
        self.aval_durs: Deque[int] = deque(maxlen=512)
        self._cur_size, self._cur_dur = 0.0, 0
        self._in_avalanche = False
        # Априорная инициализация (НЕ измерение): реальная оценка появляется после
        # накопления достаточной истории активности.
        self.sigma_hat = target_sigma
        self.sigma_hat_ready: bool = False
        self.theta_aval = 1.0
        self.zone_activity: Dict[str, Deque[float]] = {}
        self.zone_sigmas: Dict[str, float] = {}
        # Готовность измерения по зонам (False — пока данных мало)
        self.zone_sigma_ready: Dict[str, bool] = {}
        # Средняя активность зоны в той же единице, что и очередь (СУММА спайков за тик)
        self.zone_activity_mean: Dict[str, float] = {}

    def push(self, spikes_now: torch.Tensor,
             spikes_prev: torch.Tensor) -> float:
        a_t1 = float(spikes_now.sum())
        self.activity.append(a_t1)
        if len(self.activity) > self.k_max + 2:
            estimate = self._mr_estimate(self.activity)
            if estimate is not None:
                self.sigma_hat = estimate
                self.sigma_hat_ready = True
        if a_t1 > self.theta_aval:
            if not self._in_avalanche:
                self._in_avalanche, self._cur_size, self._cur_dur = True, 0.0, 0
            self._cur_size += a_t1
            self._cur_dur += 1
        elif self._in_avalanche:
            self._in_avalanche = False
            self.aval_sizes.append(self._cur_size)
            self.aval_durs.append(self._cur_dur)
        return self.sigma_hat

    def push_zones(self, zone_spikes: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Зональный мониторинг критичности (SENS, ASSOC, PFC отдельно).

        σ̂ зоны — МЕРКА: MR-оценка по суррогатной автокорреляции активности зоны.
        Раньше оценка смешивалась с «энергетическим тонусом» (`act_ratio` ∈ [0.92, 1.08])
        и клампилась в [0.88, 1.06], из-за чего любая зона «притягивалась» к ~1 и
        переставала быть измерением. Теперь:
          * единица активности — СУММА спайков зоны за тик (та же, что в очереди);
          * до накопления достаточной истории σ̂ остаётся априорным (target) и
            помечается как НЕ измеренное (`zone_sigma_ready[zone] == False`);
          * после — σ̂ зоны равна MR-оценке без клампа к цели.
        """
        for name, sp in zone_spikes.items():
            if name not in self.zone_activity:
                self.zone_activity[name] = deque(maxlen=600)
                self.zone_sigmas[name] = self.target          # априор, не измерение
                self.zone_sigma_ready[name] = False
                self.zone_activity_mean[name] = 0.0
            q = self.zone_activity[name]
            # ЕДИНИЦА: сумма спайков зоны за тик (не среднее по нейронам!)
            act_total = float(sp.sum().item()) if sp.numel() > 0 else 0.0
            q.append(act_total)
            self.zone_activity_mean[name] = sum(q) / max(1, len(q))
            if len(q) > self.k_max + 2:
                # МЕРКА: MR-оценка автокорреляции, без клампа к «~1»
                estimate = self._mr_estimate(q)
                if estimate is not None:
                    self.zone_sigmas[name] = estimate
                    self.zone_sigma_ready[name] = True
            # При недостатке данных значение остаётся априорным (target):
            # подменять измерение «тонусом» нельзя — это была бы выдумка.
        return dict(self.zone_sigmas)

    def push_local_columns(self, local_sigmas: Dict[str, float]) -> float:
        """D3 (archnext.txt): Вычисление критичности без глобального readout спайков!
        Каждая колонка передаёт только одно скалярное число local_sigma (O(1) на колонку),
        а общемозговая критичность получается усреднением без синхронизации спайковых массивов."""
        if not local_sigmas:
            return self.sigma_hat
        for zname, lsig in local_sigmas.items():
            self.zone_sigmas[zname] = float(lsig)
            # локальные σ̂ приходят из колонок как ИЗМЕРЕНИЕ (а не априор)
            self.zone_sigma_ready[zname] = True
        mean_loc = float(sum(local_sigmas.values()) / len(local_sigmas))
        self.sigma_hat = float(0.85 * self.sigma_hat + 0.15 * mean_loc)
        self.sigma_hat_ready = True
        return mean_loc

    # ------------------------------------------------------------------
    # Явная ЦЕЛЬ σ̂ (отделена от измерения)
    # ------------------------------------------------------------------
    @property
    def target_sigma(self) -> float:
        """Целевой режим по in-vivo данным (σ̂ ≈ 0.95–0.99, «ревербирующий»).

        Это ЦЕЛЬ регуляции, а не измерение: она сравнивается с измеренным σ̂
        (см. `ziwago/homeostasis/controller.py`), но никогда не подмешивается в него.
        """
        return float(self.target)

    def sigma_error(self, zone: Optional[str] = None) -> float:
        """Отклонение измеренного σ̂ от цели (σ̂_измеренное − цель).

        Для зоны используется измерение только если оно готово
        (`zone_sigma_ready`), иначе — глобальное σ̂.
        """
        if zone is not None and self.zone_sigma_ready.get(zone, False):
            measured = float(self.zone_sigmas.get(zone, self.sigma_hat))
        else:
            measured = float(self.sigma_hat)
        return measured - float(self.target)

    def _mr_estimate(self, act_buf: Optional[Deque[float]] = None) -> Optional[float]:
        """r_k = автокорреляция на лаге k; fit log r_k = k log m с робастной защитой."""
        import numpy as np
        buf = self.activity if act_buf is None else act_buf
        a = np.asarray(list(buf), dtype=float)
        a = a - a.mean()
        v = (a * a).mean() + 1e-12
        if v < 1e-9:
            return None
        rs, ks = [], []
        for k in range(1, min(self.k_max, len(a) - 1) + 1):
            r_k = float((a[:-k] * a[k:]).mean() / v)
            if r_k > 1e-6:
                rs.append(np.log(r_k)); ks.append(k)
        if len(ks) < 2:
            return None
        slope_ols = float(np.polyfit(ks, rs, 1)[0])
        # Робастная медианная оценка наклона Theil-Sen для защиты от нестационарных выбросов
        pair_slopes = [
            (rs[j] - rs[i]) / max(1, ks[j] - ks[i])
            for i in range(len(ks))
            for j in range(i + 1, len(ks))
        ]
        slope_med = float(np.median(pair_slopes)) if pair_slopes else slope_ols
        slope = 0.65 * slope_ols + 0.35 * slope_med
        return float(np.clip(np.exp(slope), 0.0, 3.0))

    def novel_avalanche(self) -> bool:
        if len(self.aval_sizes) < 8:
            return False
        import numpy as np
        return float(self.aval_sizes[-1]) >= float(
            np.quantile(list(self.aval_sizes), self.novel_q))

    def dynamic_range(self, responses: List[float],
                      inputs: List[float]) -> float:
        """Δ = 10·log10(F(0.9)/F(0.1)) (Kinouchi & Copelli 2006)."""
        if len(responses) < 4:
            return 0.0
        import numpy as np
        r = np.sort(np.asarray(responses, dtype=float))
        f90, f10 = r[int(0.9 * (len(r) - 1))], r[int(0.1 * (len(r) - 1))]
        return float(10.0 * np.log10((f90 + 1e-9) / (f10 + 1e-9)))

    def stats(self) -> Dict[str, float]:
        return {"sigma_hat": self.sigma_hat,
                "sigma_hat_ready": 1.0 if self.sigma_hat_ready else 0.0,
                "sigma_target": float(self.target),
                "sigma_error": float(self.sigma_hat - float(self.target)),
                "n_zones_measured": float(sum(1 for v in self.zone_sigma_ready.values() if v)),
                "n_avalanches": len(self.aval_sizes),
                "aval_mean_size": (sum(self.aval_sizes)
                                   / max(1, len(self.aval_sizes))),
                "aval_mean_dur": (sum(self.aval_durs)
                                  / max(1, len(self.aval_durs)))}
