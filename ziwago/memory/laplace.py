"""Спектральный банк временных констант (REGENESIS, Закон 4).

Клетки времени Shankar & Howard (2012): геометрический спектр
экспоненциальных интеграторов τ_k = τ_min·c^k — дискретное
преобразование Лапласа входного потока. В v10 медленные шкалы
модулируют адаптацию порога нейронов L5: возбудимость клетки зависит
от длинного контекста, а не от отдельного вектора в декодере.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


class LaplaceBank:
    """O(K·D) спектральная память; медленный профиль → Θ-адаптация L5."""

    def __init__(self, feature_dim: int, num_scales: int = 8,
                 tau_min: float = 4.0, tau_max: float = 512.0):
        self.feature_dim = int(feature_dim)
        self.num_scales = max(2, int(num_scales))
        taus = torch.exp(torch.linspace(
            math.log(tau_min), math.log(tau_max), self.num_scales))
        self.taus = taus
        self.alphas = torch.exp(-1.0 / taus).view(-1, 1, 1)  # [K,1,1]
        self.state: Optional[torch.Tensor] = None            # [K,B,F]

    def reset_state(self, batch: int, device=None, dtype=torch.float32) -> None:
        self.state = torch.zeros(self.num_scales, batch,
                                 self.feature_dim, device=device,
                                 dtype=dtype)

    def _require_state(self) -> torch.Tensor:
        if self.state is None:
            raise RuntimeError(
                "LaplaceBank: состояние пуст — вызовите step(x) или reset_state(batch) "
                "перед slow_profile()/time_cells()")
        return self.state

    # ------------------------------------------------------------------
    def step(self, x: torch.Tensor) -> torch.Tensor:
        """x [B,F] → обновлённое состояние [K,B,F].

        Ленивая инициализация по форме входа: без неё первый же step падал
        TypeError (`self.alphas * None`).
        """
        if self.state is None or self.state.shape[1] != x.shape[0] \
                or self.state.device != x.device:
            self.reset_state(x.shape[0], x.device, x.dtype)
        self.state = self.alphas.to(self.state.device) * self.state + x.detach().unsqueeze(0)
        return self.state

    def slow_profile(self) -> torch.Tensor:
        """Медленная половина спектра → [B,F]: длинный контекст."""
        state = self._require_state()
        k0 = self.num_scales // 2
        return state[k0:].mean(0)

    def time_cells(self) -> torch.Tensor:
        """Разностный фильтр Поста-Виддера — когда появился вход."""
        state = self._require_state()
        l = torch.roll(state, 1, 0)
        r = torch.roll(state, -1, 0)
        return torch.relu(state - 0.5 * (l + r))
