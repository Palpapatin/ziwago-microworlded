"""Фаза медленноволнового сна (SWS / SHY Тонони–Чирелли) — слой организма.

Вынесено из train_ziwago_v15_omniscient_marathon.py (#32, 2026-11-12):
единственная живая v15-функция, нужная v19-режиму плоского организма
(ZiwagoOrganism(integrated=False)). Self-contained: только torch.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch

if TYPE_CHECKING:
    from ziwago.organism.flat import ZiwagoPureSynapticDialogueOrganismV14


def execute_sws_synaptic_homeostasis_and_pruning(
    organism: ZiwagoPureSynapticDialogueOrganismV14,
    downscale_factor: float = 0.985,
) -> Dict[str, float]:
    r"""Фаза медленноволнового сна Тонони–Чирелли (`Slow-Wave Sleep Synaptic Homeostasis / SHY`):
      1. Пропорционально ослабляет фоновый синаптический шум в `w_dend_out` (`w_dend_out *= downscale_factor`),
         сохраняя пиковые консолидированные NMDA-связи;
      2. Сортирует полюса аттракторов `W_CA3` по их дофаминовой значимости (`w_ca3_salience`),
         чтобы фундаментальные законы и высоконаграждённые диалоги никогда не вытеснялись шумом!"""
    readout = organism.dendritic_plateau_readout
    with torch.no_grad():
        before_norm = float(readout.w_dend_out.norm().item())
        # Мягкое гомеостатическое подавление слабого синаптического шума (< 0.05) при сохранении сильных связей
        weak_mask = readout.w_dend_out.abs() < 0.05
        readout.w_dend_out[weak_mask] *= 0.85
        readout.w_dend_out[~weak_mask] *= downscale_factor
        after_norm = float(readout.w_dend_out.norm().item())

        # Сортировка полюсов CA3 по убыванию дофаминовой значимости (Synaptic Salience Sorting)
        n_poles = int(readout.w_ca3_attractors.shape[0])
        if n_poles > 1 and readout.w_ca3_salience.shape[0] == n_poles:
            sorted_idx = torch.argsort(readout.w_ca3_salience, descending=True)
            readout.w_ca3_attractors = readout.w_ca3_attractors.index_select(0, sorted_idx)
            readout.w_ca3_sdr = readout.w_ca3_sdr.index_select(0, sorted_idx)
            readout.w_ca3_salience = readout.w_ca3_salience.index_select(0, sorted_idx)

    return {
        "ca3_poles_sorted": n_poles,
        "w_dend_out_norm_before": round(before_norm, 2),
        "w_dend_out_norm_after": round(after_norm, 2),
    }
