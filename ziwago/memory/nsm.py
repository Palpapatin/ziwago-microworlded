"""Модуль нарративного Я (NSM, ARCH-NSM-001): автобиография и настроение.

Эпизод e_k = (h_k, t_k, v_k, s_k): bipolar VSA-контекст скрытого состояния,
абсолютное время, аффективная валентность v_k = tanh(∫ da_phasic) ∈ [-1, 1] и
салиентность s_k ∈ [0, 1]. Настроение M(t) — экспоненциально взвешенное среднее
валентностей (часовой масштаб τ_M), строго ограничено [-1, 1] (NSM-3). Вектор
самости v_self — нормированная сумма bipolar-binding эпизодов с ключом роли
(XOR в биполярном домене, без FFT — NSM-2).

Влияние на порождение речи — ТОЛЬКО bounded additive prior (NSM-4): модуль
выдаёт мягкое аддитивное смещение, зажатое поэлементно в ±prior_bound; жёсткое
форсирование токенов (hard token clamping) запрещено и не реализуется. Пустое
состояние нейтрально: M = 0, v_self = 0, prior = 0.

Живая проводка prior в дендритный bias ридаута и адаптер эпизодов гиппокампа —
отдельный pending-подшаг; здесь проверена математика модуля и границы prior.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


class NarrativeSelfModule:
    """Автобиографический таймлайн, настроение и вектор самости (opt-in)."""

    def __init__(self, d_vec: int = 64, max_episodes: int = 1024,
                 alpha: float = 0.5, tau_ep: float = 86400.0,
                 tau_m: float = 3600.0, eps: float = 1e-8,
                 prior_bound: float = 0.5, seed: int = 2026):
        if d_vec <= 0:
            raise ValueError("d_vec must be positive")
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if tau_ep <= 0.0 or tau_m <= 0.0:
            raise ValueError("tau_ep and tau_m must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if prior_bound < 0.0:
            raise ValueError("prior_bound must be non-negative")
        self.d_vec = int(d_vec)
        self.max_episodes = int(max_episodes)
        self.alpha = float(alpha)
        self.tau_ep = float(tau_ep)
        self.tau_m = float(tau_m)
        self.eps = float(eps)
        self.prior_bound = float(prior_bound)

        g = torch.Generator().manual_seed(int(seed))
        # Ортогональный ключ текущей роли (биполярный, фиксированный).
        self.role_key = torch.where(torch.randn(d_vec, generator=g) >= 0.0, 1.0, -1.0)
        # Фиксированные проекции additive prior (обучение — pending-проводка).
        self.w_self = torch.randn(d_vec, d_vec, generator=g) / math.sqrt(d_vec)
        self.w_mood = torch.randn(d_vec, generator=g) / math.sqrt(d_vec)

        # Кольцевой буфер эпизодов (NSM-5: последние K = max_episodes).
        self.H = torch.zeros(max_episodes, d_vec)
        self.t = torch.zeros(max_episodes)
        self.v = torch.zeros(max_episodes)
        self.s = torch.zeros(max_episodes)
        self.count = 0
        self.ptr = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _bipolar(x: torch.Tensor) -> torch.Tensor:
        """SBC-VSA биполярный код {-1, +1}^D (sign с конвенцией sign(0)=+1)."""
        return torch.where(x >= 0.0, 1.0, -1.0)

    def record_episode(self, h, t: float, v: float, s: float) -> None:
        """Зафиксировать полный эпизод (NSM-1: без усечённых выжимок)."""
        h = torch.as_tensor(h, dtype=torch.float32).reshape(-1)
        if h.numel() != self.d_vec:
            raise ValueError("episode context dimension mismatch")
        t = float(t)
        if t < 0.0:
            raise ValueError("episode timestamp must be non-negative")
        v = max(-1.0, min(1.0, float(v)))
        s = max(0.0, min(1.0, float(s)))
        i = self.ptr
        self.H[i] = h
        self.t[i] = t
        self.v[i] = v
        self.s[i] = s
        self.ptr = (self.ptr + 1) % self.max_episodes
        self.count = min(self.count + 1, self.max_episodes)

    def _active(self) -> torch.Tensor:
        """Индексы активных слотов в хронологическом порядке (старый → новый)."""
        if self.count == 0:
            return torch.zeros(0, dtype=torch.long)
        if self.count < self.max_episodes:
            return torch.arange(self.count)
        return torch.cat([torch.arange(self.ptr, self.max_episodes),
                          torch.arange(0, self.ptr)])

    # ------------------------------------------------------------------
    def mood(self, t: float) -> float:
        """M(t) = Σ v_k e^{-(t-t_k)/τ_M} / (Σ e^{-(t-t_k)/τ_M} + ε) ∈ [-1, 1]."""
        idx = self._active()
        if idx.numel() == 0:
            return 0.0
        dt = (float(t) - self.t[idx]).clamp_min(0.0)
        decay = torch.exp(-dt / self.tau_m)
        num = float((self.v[idx] * decay).sum())
        den = float(decay.sum()) + self.eps
        return max(-1.0, min(1.0, num / den))

    def retrieve(self, q, t: float, top_k: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """Скоринг извлечения (ARCH-NSM-001 §2.2), целочисленный косинус без FFT.

        Score = [cos(h_q, h_k)]_+ · (1 + α|v_k|) · e^{-(t-t_k)/τ_ep} · (1 + M(t)·v_k).
        Возвращает (индексы эпизодов, scores) по убыванию релевантности.
        """
        idx = self._active()
        if idx.numel() == 0 or int(top_k) <= 0:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0)
        q = torch.as_tensor(q, dtype=torch.float32).reshape(-1)
        if q.numel() != self.d_vec:
            raise ValueError("query dimension mismatch")
        bq = self._bipolar(q)
        bH = self._bipolar(self.H[idx])
        cos = (bH @ bq) / self.d_vec                 # integer cosine SBC-VSA
        cos_pos = cos.clamp_min(0.0)
        dt = (float(t) - self.t[idx]).clamp_min(0.0)
        recency = torch.exp(-dt / self.tau_ep)
        v = self.v[idx]
        amygdala = 1.0 + self.alpha * v.abs()        # эмоциональные помнятся лучше
        congruence = 1.0 + self.mood(t) * v          # mood-congruent memory ∈ [0, 2]
        score = cos_pos * amygdala * recency * congruence
        k = min(int(top_k), idx.numel())
        top = torch.topk(score, k)
        return idx[top.indices], top.values

    def self_vector(self, t: float) -> torch.Tensor:
        """v_self = Normalize(Σ w(Δt)·(h_k ⊗ k_role)), binding = XOR (без FFT).

        w(Δt) = e^{-Δt/τ_ep}·(1 + α|v_k|): приоритет недавним и эмоциональным.
        Пустое состояние → нулевой вектор (нейтральная субъектная позиция).
        """
        idx = self._active()
        if idx.numel() == 0:
            return torch.zeros(self.d_vec)
        dt = (float(t) - self.t[idx]).clamp_min(0.0)
        w = torch.exp(-dt / self.tau_ep) * (1.0 + self.alpha * self.v[idx].abs())
        bound = self._bipolar(self.H[idx]) * self.role_key.unsqueeze(0)
        acc = (w.unsqueeze(1) * bound).sum(0)
        norm = float(acc.norm())
        if norm <= self.eps:
            return torch.zeros(self.d_vec)
        return acc / norm

    def additive_prior(self, t: float, gain: float = 1.0) -> torch.Tensor:
        """Bounded additive bias (NSM-4): clamp(gain·(W_self v_self + w_mood M), ±bound).

        gain = 0 → нейтральный ноль (opt-in off). Пустое состояние → ноль.
        Поэлементное ограничение ±prior_bound исключает hard token clamping.
        """
        if float(gain) == 0.0:
            return torch.zeros(self.d_vec)
        vs = self.self_vector(t)
        m = self.mood(t)
        raw = float(gain) * (self.w_self @ vs + self.w_mood * m)
        return raw.clamp(-self.prior_bound, self.prior_bound)

    # ------------------------------------------------------------------
    def telemetry(self, t: float) -> dict:
        vs = self.self_vector(t)
        return {
            "n_episodes": int(self.count),
            "mood": round(self.mood(t), 6),
            "self_norm": round(float(vs.norm()), 6),
            "prior_bound": self.prior_bound,
        }

    def export_state_dict(self) -> dict:
        return {
            "d_vec": self.d_vec, "max_episodes": self.max_episodes,
            "alpha": self.alpha, "tau_ep": self.tau_ep, "tau_m": self.tau_m,
            "eps": self.eps, "prior_bound": self.prior_bound,
            "role_key": self.role_key.clone(),
            "w_self": self.w_self.clone(), "w_mood": self.w_mood.clone(),
            "H": self.H.clone(), "t": self.t.clone(),
            "v": self.v.clone(), "s": self.s.clone(),
            "count": self.count, "ptr": self.ptr,
        }

    def load_state_dict(self, sd: dict) -> None:
        if int(sd["d_vec"]) != self.d_vec or int(sd["max_episodes"]) != self.max_episodes:
            raise ValueError("NSM state shape mismatch")
        self.alpha = float(sd["alpha"])
        self.tau_ep = float(sd["tau_ep"])
        self.tau_m = float(sd["tau_m"])
        self.eps = float(sd["eps"])
        self.prior_bound = float(sd["prior_bound"])
        self.role_key = sd["role_key"].clone()
        self.w_self = sd["w_self"].clone()
        self.w_mood = sd["w_mood"].clone()
        self.H = sd["H"].clone()
        self.t = sd["t"].clone()
        self.v = sd["v"].clone()
        self.s = sd["s"].clone()
        self.count = int(sd["count"])
        self.ptr = int(sd["ptr"])