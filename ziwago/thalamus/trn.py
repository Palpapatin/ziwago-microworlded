"""Таламическая ретикулярная ядро TRN (§II.1.3).

Секторный GABA-шлюз: каждый сектор TRN контролирует свой релейный канал.
Управление:
  L6-кортикоталамический — выбор «что пропустить»;
  PFC/AST — нисходящий приоритет внимания (searchlight, Crick 1984);
  латеральное торможение WTM между секторами.

Гейт сектора k — ПРОИЗВЕДЕНИЕ, а не сумма (конкурирует приоритет, а не
масштаб):

    g_k = g_L6,k · g_AST,k · max(0, 1 − lateral_inh_k)

    g_L6,k        — high-pass L6: EMA( clamp(l6_drive_k − 0.5·l6_dc_k, 0, 1) )
                    плюс электротоническое выравнивание фазы через `elec_sync`
                    (`g_elec * (osc @ w_elec.t() - osc)`), так что фактическая
                    L6-гамма в коде включает программно добавленный фазовый член;
    g_AST,k       — ПРОФИЛЬ приоритета: clamp(ast_k,0,∞) / max_j clamp(ast_j,0,∞);
                    нормировка на максимум делает гейт инвариантным к общему
                    масштабу (удвоение всего вектора не меняет g_k);
                    ast_priority=None или нулевой профиль ⇒ g_AST,k = 1;
    lateral_inh_k — Σ_j w_lat[k,j] · g_L6,j (ГАМК-торможение соседей).

g_AST ≤ 1, g_L6 ≤ 1, lateral_inh ≥ 0 ⇒ gate ∈ [0, 1].
Итоговый гейт FO: Π g_k.
"""
from __future__ import annotations

from typing import Optional

import torch


class TRN:
    """Ретикулярное ядро: посекторный GABA-гейт + Mexican-hat торможение
    + электротонический синцитий (щелевые контакты) + AC-фильтр L6 (Genesis)."""

    def __init__(self, n_sectors: int):
        self.n = n_sectors
        self.act = torch.zeros(n_sectors)      # активность секторов r
        self.l6_dc = torch.zeros(n_sectors)    # медленное среднее L6 (DC-уровень)
        self.elec_dc = torch.zeros(n_sectors)  # медленное среднее r для gap-junctions
        idx = torch.arange(n_sectors).float()
        dist = (idx[:, None] - idx[None, :]).abs()
        dist = torch.minimum(dist, n_sectors - dist)
        # Латеральное ГАМК-торможение внутри TRN
        w_lat = torch.exp(-0.5 * (dist / 1.2) ** 2) * 0.3
        w_lat.fill_diagonal_(0.0)
        self.w_lat = w_lat
        # Электротонические щелевые контакты (нормированная диффузия соседей)
        w_elec = torch.exp(-0.5 * (dist / 1.0) ** 2)
        w_elec.fill_diagonal_(0.0)
        self.w_elec = w_elec / w_elec.sum(dim=1, keepdim=True).clamp(min=1e-6)
        self.gate = torch.ones(n_sectors)      # выходной гейт на релей

    def step(self, l6_drive: torch.Tensor,
             ast_priority: Optional[torch.Tensor] = None,
             vigilance: float = 0.5) -> torch.Tensor:
        """l6_drive [S] — кортикоталамический запрос каждого сектора.

        Возвращает gate [S]: 0=закрыт, 1=открыт."""
        x_raw = l6_drive.clamp(0.0, 1.0)
        # High-pass фильтр L6 (Genesis l6_ac_frac=0.6): убирает постоянную
        # тоническую засветку L6, оставляя информативную модуляцию
        self.l6_dc = 0.995 * self.l6_dc + 0.005 * x_raw
        x = (x_raw - 0.5 * self.l6_dc).clamp(0.0, 1.0)
        # g_L6: AST сюда не входит — приоритет входит в гейт множителем ниже,
        # иначе он учитывался бы дважды и разбавлял бы конкуренцию
        self.act = self.act * 0.85 + x * 0.15
        lateral = self.act @ self.w_lat        # торможение соседей
        # Щелевые контакты выравнивают только осцилляторную фазу (act - elec_dc),
        # закрываясь при высокой бодрости (vigilance)
        self.elec_dc = 0.99 * self.elec_dc + 0.01 * self.act
        osc = self.act - self.elec_dc
        g_elec = 0.4 / (1.0 + (max(0.0, vigilance) / 0.25) ** 2)
        elec_sync = g_elec * (osc @ self.w_elec.t() - osc)
        self.act = (self.act + 0.1 * elec_sync).clamp(0.0, 1.0)
        # g_AST — профиль внимания: деление на максимум убирает зависимость
        # от абсолютного масштаба приоритета (важна форма, а не уровень)
        if ast_priority is None:
            g_ast = torch.ones_like(self.act)
        else:
            a = ast_priority.clamp(min=0.0)
            peak = a.max()
            g_ast = a / peak if float(peak) > 1e-8 else torch.ones_like(a)
        # g_k = g_L6 · g_AST · (1 − lateral_inh)
        self.gate = (self.act * g_ast * (1.0 - lateral).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        return self.gate
