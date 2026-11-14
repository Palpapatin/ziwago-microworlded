"""Локальные алгоритмы, подключаемые к текущему языковому readout.

Только Fisher-anchor и MCA реально вызываются readout при включённых opt-in
флагах. EWC/surrogate требуют запрещённого живому пути backprop, а phase/sync и
predictive остаются отдельными экспериментальными модулями и не реэкспортируются.
"""
from __future__ import annotations

from .local_fisher import SparseFisherAnchor
from .mca import MetaplasticSusceptibility

__all__ = [
    "SparseFisherAnchor",
    "MetaplasticSusceptibility",
]