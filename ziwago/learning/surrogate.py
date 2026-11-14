"""Экспериментальные суррогатные градиенты спайка, запрещены в живом языке.

Любой вызов требует ``allow_experimental=True``. Модуль НЕ используется живым ридеаутом v19:
`cortex/spiking_language.py:3` явно запрещает surrogate-нейроны и глобальный
backprop («spikes and L5 output come directly from SparselyGatedCorticalMoE.step»).
Модуль предоставляет единственный backprop-through-spikes путь для БУДУЩИХ
градиентных контуров (v20+), не нарушая локальных правил живого пути.

Две калиброванные псевдопроизводные:
    FastSigmoid: d/dx = 1 / (1 + k|x|)^2             (левое полушарие: точность)
    Arctan:      d/dx = (a/2) / (1 + (pi a x /2)^2)  (правое: тяжёлые хвосты)

Крутизна k — медленный параметр организма (мета-контроллер меняет её между
эпизодами, «5-HT управляет резкостью обучения»); внутри такта k постоянна,
поэтому горячий контур остаётся чисто тензорным.
"""
from __future__ import annotations

import math

import torch


class _SpikeSigmoid(torch.autograd.Function):
    """Forward: ступенька Хевисайда. Backward: FastSigmoid-псевдопроизводная."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, k: float):
        ctx.save_for_backward(x)
        ctx.k = float(k)
        return (x >= 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        denom = 1.0 + ctx.k * x.abs()
        return grad_out / (denom * denom), None


class _SpikeArctan(torch.autograd.Function):
    """Forward: ступенька. Backward: Arctan с тяжёлыми хвостами."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float):
        ctx.save_for_backward(x)
        ctx.alpha = float(alpha)
        return (x >= 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        a = ctx.alpha
        f = (math.pi / 2.0) * a * x
        return grad_out * (a / 2.0) / (1.0 + f * f), None


def spike_fn(x: torch.Tensor, mode: str = "fast_sigmoid", k: float = 10.0, *,
             allow_experimental: bool = False) -> torch.Tensor:
    """Пороговая функция с суррогатным градиентом.

    x — «расстояние до порога» (V - theta), НЕ сам потенциал.
    """
    if not allow_experimental:
        raise RuntimeError(
            "Surrogate gradients are forbidden in the live language path; "
            "pass allow_experimental=True only in an explicitly permitted experiment."
        )
    if mode == "fast_sigmoid":
        return _SpikeSigmoid.apply(x, k)
    if mode == "arctan":
        return _SpikeArctan.apply(x, k)
    raise ValueError(f"unknown surrogate mode: {mode!r}")


def hebbian_step(x: torch.Tensor, mode: str = "fast_sigmoid", k: float = 10.0, *,
                 allow_experimental: bool = False) -> torch.Tensor:
    """Дифференцируемый «мягкий спайк» для локальных правил (без autograd-функции).

    Используется там, где нужен именно градиент, а не событие: локальное
    предсказание в колонке (predictive coding). Сигмоида с температурой 1/k —
    монотонная аппроксимация ступеньки.
    """
    if not allow_experimental:
        raise RuntimeError(
            "Surrogate/hebbian soft spikes are forbidden in the live language path; "
            "pass allow_experimental=True only in an explicitly permitted experiment."
        )
    if mode == "arctan":
        return 0.5 + torch.atan((math.pi / 2.0) * k * x) / math.pi
    return torch.sigmoid(k * x)