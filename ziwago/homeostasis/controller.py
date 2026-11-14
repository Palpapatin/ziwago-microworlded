"""Гомеостатический контроллер (§1.5): регуляция по измеренной критичности.

Механизмы:
  MR-оценщик σ̂ → если σ>1: синаптическая депрессия (Levina),
                 ↓весов; σ<1: потенциация/нейрогенез;
  Intrinsic plasticity: ΔΘ = η_IP·(x - H_IP) — порог к целевой частоте;
  Vogels 2011 iSTDP: ингибиторные веса следят за пост-частотой ρ₀;
  SORN-нормировка: Σw_in ≈ const (синаптическое масштабирование).

ВЛАДЕНИЕ ИЗМЕРЕНИЕМ (трек A, v22):
  * ИЗМЕРЕНИЕ σ̂ принадлежит `ziwago/core/criticality.py` (`CriticalityMonitor`).
    Этот модуль — её ПОТРЕБИТЕЛЬ: он ничего не «оценивает» сам и не подменяет
    измерение априором.
  * «Цель σ≈1» — отдельная величина (`HomeostasisCfg.target_sigma` /
    `CriticalityMonitor.target`), а не кламп внутри оценки.
  * Управляющая переменная регулятора (`zone_effective_sigma`) хранится ЗДЕСЬ и
    не затирает измерение в мониторе.

ПЕРЕНОСИМОСТЬ (трек A, v22):
  `regulate()` принимает любые реальные структуры проекта:
    * объект с `.columns` — `moe_cortex.SparselyGatedCorticalMoE` (production);
    * объект с `.cols`    — старая `cortex/hierarchy.py` (в production не импортируется);
    * сам список колонок  — `CorticalColumn` (`cortex/column.py`), у каждой есть
      `.name`, `.pyr` (`USNeuronPool`), `.w_ff/.w_fb/.w_rec` (`GenesisSynapse`),
      `.w_pv_in`, `.zcfg`.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ..config import HomeostasisCfg
from ..core.criticality import CriticalityMonitor


class HomeostaticController:
    """Регуляция критичности по всем зонам.

    Единственный владелец σ̂-регуляции: измерение — `CriticalityMonitor`
    (`ziwago/core/criticality.py`), регуляция — здесь.
    """

    def __init__(self, cfg: HomeostasisCfg,
                 monitor: Optional[CriticalityMonitor] = None):
        self.cfg = cfg
        self.monitor = monitor if monitor is not None else CriticalityMonitor(
            cfg.mr_kmax, target_sigma=cfg.target_sigma)
        self._prev_spikes: Dict[str, torch.Tensor] = {}
        self._tick = 0
        # Управляющая переменная регулятора (НЕ измерение): накапливает коррекцию
        self.zone_effective_sigma: Dict[str, float] = {}
        # Отчёт по зонам (измерение / управление / цель)
        self.zone_report: Dict[str, Dict[str, float]] = {}
        self.last_report: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # ИЗМЕРЕНИЕ (единственный источник истины — core/criticality.py)
    # ------------------------------------------------------------------
    @property
    def sigma_hat(self) -> float:
        """Измеренное σ̂ (MR-оценка) из монитора."""
        return float(self.monitor.sigma_hat)

    @property
    def target_sigma(self) -> float:
        """Целевой режим (явная величина, а не кламп оценки)."""
        return float(self.cfg.target_sigma)

    def observe(self, spikes_now: torch.Tensor,
                spikes_prev: Optional[torch.Tensor] = None) -> float:
        """Скормить монитору активность (глобальная σ̂). Возвращает измеренную σ̂."""
        if spikes_prev is None:
            spikes_prev = torch.zeros_like(spikes_now)
        return float(self.monitor.push(spikes_now, spikes_prev))

    def observe_zones(self, zone_spikes: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Скормить монитору зональную активность. Возвращает измеренные σ̂ зон."""
        return self.monitor.push_zones(zone_spikes)

    # ------------------------------------------------------------------
    # РЕГУЛЯЦИЯ
    # ------------------------------------------------------------------
    @staticmethod
    def resolve_columns(source) -> List:
        """Достаёт список колонок из любой реальной структуры проекта.

        Поддерживаются: `.columns` (moe_cortex), `.cols` (старая иерархия),
        а также готовый список/кортеж колонок.
        """
        if source is None:
            return []
        if isinstance(source, (list, tuple)):
            return list(source)
        for attr in ("columns", "cols"):
            cols = getattr(source, attr, None)
            if cols is not None:
                return list(cols)
        raise TypeError(
            "regulate(): ожидался объект с .columns (moe_cortex), .cols (иерархия) "
            f"или список колонок, получено {type(source).__name__}")

    def regulate(self, source, sigma: Optional[float] = None,
                 dt_ms: float = 1.0) -> Dict[str, float]:
        """Применить регуляцию по разделённым шкалам времени (Menesse et al. 2024; §1.5):
          1. Быстрая шкала (каждый такт, ~10-50 мс): ресурс везикул u_syn (Levina 2007),
             а синапс-специфичный iSTDP Вогельса работает локально в CorticalColumn.step().
          2. Средняя шкала (покомпонентная IP, ~100-500 мс): индивидуальные пороги Θ_i
             следуют за медленным кальциевым следом нейрона ca_i с учётом τ_m(g) зоны.
          3. Медленная шкала (раз в wnorm_every тактов): мультипликативное синаптическое
             масштабирование Тёрриджано на w_slow без разрушения выученной структуры.

        `source` — колонки (`.columns` / `.cols` / список). `sigma=None` ⇒ берётся
        измеренная σ̂ из монитора.
        """
        c = self.cfg
        self._tick += 1
        columns = self.resolve_columns(source)
        measured_sigma = self.sigma_hat if sigma is None else float(sigma)
        target = float(c.target_sigma)
        report = {"sigma": measured_sigma,
                  "sigma_target": target,
                  "sigma_error": measured_sigma - target,
                  "depression": 0.0,
                  "potentiation": 0.0,
                  "ip": 0.0,
                  "istdp": 0.0}
        do_wnorm = (c.wnorm_every > 0 and self._tick % c.wnorm_every == 0)
        for col in columns:
            pool = col.pyr
            # Измеренная σ̂ зоны (только если это реальное измерение, а не априор!)
            zone_measured = None
            if col.name in self.monitor.zone_sigmas and \
                    self.monitor.zone_sigma_ready.get(col.name, False):
                zone_measured = float(self.monitor.zone_sigmas[col.name])
            # Управляющая переменная: накапливает коррекцию между тиками
            if col.name not in self.zone_effective_sigma:
                self.zone_effective_sigma[col.name] = (
                    zone_measured if zone_measured is not None else measured_sigma)
            eff_sigma = float(self.zone_effective_sigma[col.name])
            # Коэффициент темпорального масштабирования зоны (Murray/Hasson):
            # медленные зоны (PFC) меняют порог IP плавнее, чем быстрые (SENS)
            tau_scale = min(1.0, 20.0 / max(20.0, float(col.zcfg.tau_m))) ** 0.35

            # 1) Быстрая шкала (Levina et al. 2007): модуляция кратковременного ресурса u_syn
            #    при отклонении σ от критичности (без ежетактового стирания w_slow!)
            if eff_sigma > c.target_sigma + c.sigma_tol:
                for w in (col.w_ff, col.w_fb, col.w_rec):
                    w.u_syn.mul_(0.996)
                pool.theta = (pool.theta + 0.06).clamp(-56.0, -42.0)
                col.w_pv_in = (col.w_pv_in * 1.015).clamp(0.05, 3.0)
                self.zone_effective_sigma[col.name] = float(
                    eff_sigma - 0.12 * (eff_sigma - c.target_sigma))
                report["depression"] += 1.0
            elif eff_sigma < c.target_sigma - c.sigma_tol:
                for w in (col.w_ff, col.w_fb, col.w_rec):
                    w.u_syn = (w.u_syn + 0.012).clamp(max=1.0)
                pool.theta = (pool.theta - 0.08).clamp(-56.0, -42.0)
                col.w_pv_in = (col.w_pv_in * 0.985).clamp(0.05, 3.0)
                self.zone_effective_sigma[col.name] = float(
                    eff_sigma + 0.12 * (c.target_sigma - eff_sigma))
                report["potentiation"] += 1.0

            # 2) Средняя шкала — покомпонентная Информационная IP Триша (Triesch 2005):
            #    оптимизирует экспоненциальное распределение частот при фиксированном бюджете
            if hasattr(pool, "ca") and pool.ca is not None:
                rate_vec = pool.ca.mean(0)  # [N]
                rate_mean = float(rate_vec.mean().item())
                mu = max(0.02, float(c.ip_target))
                rate_err = rate_vec - mu
                ent_grad = mu * ((2.0 + 1.0 / mu) * rate_vec - (rate_vec ** 2) / mu - 1.0).clamp(-0.3, 0.3)
                d_theta = c.ip_rate * tau_scale * (0.75 * rate_err + 0.25 * ent_grad)
                pool.theta = (pool.theta + d_theta.unsqueeze(0)).clamp(-56.0, -42.0)
                report["ip"] += float(d_theta.abs().mean().item())
            else:
                rate_mean = 0.0

            # 3) Отчёт по тормозному балансу iSTDP (сам матричный iSTDP выполняется
            #    локально в CorticalColumn.step() без дублирующего скалярного сдвига!)
            d_inh = c.inh_stdp_rate * (rate_mean - 0.06)
            report["istdp"] += abs(d_inh)

            # 4) Медленная шкала (раз в wnorm_every тактов) — мультипликативное
            #    синаптическое масштабирование Тёрриджано (сохраняет пропорции w_slow)
            if do_wnorm:
                sigma_factor = 0.995 if eff_sigma > c.target_sigma + c.sigma_tol else (
                    1.005 if eff_sigma < c.target_sigma - c.sigma_tol else 1.0
                )
                for w in (col.w_ff, col.w_fb, col.w_rec):
                    nrm = w.w_slow.norm(dim=1, keepdim=True).clamp(min=1e-6)
                    tgt = getattr(w, "target_row_norm",
                                  torch.full_like(nrm, 0.8)).clamp(min=0.25, max=3.0)
                    scale = (tgt / nrm).clamp(0.85, 1.15) * sigma_factor
                    w.w_slow.mul_(0.9 * 1.0 + 0.1 * scale).clamp_(w.cfg.w_min, w.cfg.w_max)

            # 5) Гомеостаз критичности → метаболический тормоз KATP/GIRK:
            if eff_sigma < c.target_sigma - c.sigma_tol:
                pool.met_gain = max(0.1, getattr(pool, "met_gain", 1.0) * 0.995)
            elif eff_sigma > c.target_sigma + c.sigma_tol:
                pool.met_gain = min(2.0, getattr(pool, "met_gain", 1.0) * 1.005)
            report["met_gain"] = float(getattr(pool, "met_gain", 1.0))
            self.zone_report[col.name] = {
                "sigma_measured": (float(zone_measured) if zone_measured is not None
                                   else float(measured_sigma)),
                "sigma_control": eff_sigma,
                "sigma_target": float(c.target_sigma),
                "measured_ready": 1.0 if zone_measured is not None else 0.0,
            }
        self.last_report = report
        return report
