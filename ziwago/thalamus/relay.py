"""Таламус: FO-релей (§II.1.2) и HO-ядро (§II.1.4).

FO (first order, LGN/VPL): передаёт сенсорику в L4. Два режима:
  tonic — линейный релей высокой точности ([ACh]↑, [NE]↑);
  burst — нелинейный T-type Ca2+-burst «wake-up call»
          (молчание афферентов → деполяризация после гиперполяризации
          → нейрон «кричит»). Режим выбирается модуляторами и паузой
          входа (Sherman 2001).

HO (higher order, Pulvinar/MD): релей кортико-кортикального драйва —
копия L5-эфферентов идёт через HO в следующую зону (трансталамический
путь). WTA выбирает «сильнейшие сообщения»; базальные ганглии
разблокируют релей через дизингибицию (GPi/SNr → таламус).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


class ThalamusFO:
    """First-order таламус с переключением tonic/burst, T-type Ca2+ инактивацией h_T
    и TRN-гейтом (Sherman 2001, Genesis TcRelayCore)."""

    # Дефолты телеметрии: восстановление старого снимка (load_runtime_tree удаляет
    # поля, которых не было в чекпоинте) не должно оставлять объект без отчёта.
    _TELEMETRY_DEFAULTS = {
        "boundary_burst": False,
        "boundary_connected": False,
        "b_drive": 0.0,
        "boundary_markers": 0,
        "mode": "tonic",
        "spindle": None,
        "spindle_connected": False,
        "h_t_mean": 0.0,
        "burst_cause": "none",
        "spindle_applied": False,
    }

    # Честные пометки подключения (трек M, phantom-effect):
    # BOUNDARY_CONNECTED — flat language loop помечает конец фразы на FO,
    #   после чего следующий integrated tick потребляет одноразовый маркер.
    # SPINDLE_CONNECTED — spindle() пока вызывается только тестами; живой цикл
    #   сна ещё не задаёт осцилляцию перед тиком релея.
    BOUNDARY_CONNECTED = True
    SPINDLE_CONNECTED = False

    def __init__(self, n: int, name: str = "FO"):
        self.n, self.name = n, name
        self.mode = "tonic"
        self.silence_ms = torch.zeros(n)  # длительность молчания канала
        self.burst_ms = torch.zeros(n)    # оставшиеся такты burst-режима
        self.h_t = torch.zeros(n)         # доступность T-type Ca2+ каналов [0..1]
        self._gate = torch.ones(n)        # TRN-гейт [0..1]; снаружи — свойство gate
        self._spindle_osc: Optional[torch.Tensor] = None  # фаза веретена на один шаг
        self._boundary_pending = False    # одноразовый маркер границы фразы
        self.telemetry: dict = dict(self._TELEMETRY_DEFAULTS)
        self._telemetry()

    def _telemetry(self) -> dict:
        """Единый словарь телеметрии; чинится, если его снесло восстановление снимка."""
        tel = getattr(self, "telemetry", None)
        if not isinstance(tel, dict):
            tel = dict(self._TELEMETRY_DEFAULTS)
            self.telemetry = tel
        for key, default in self._TELEMETRY_DEFAULTS.items():
            tel.setdefault(key, default)
        tel["boundary_connected"] = bool(self.BOUNDARY_CONNECTED)
        tel["spindle_connected"] = bool(self.SPINDLE_CONNECTED)
        return tel

    @property
    def gate(self) -> torch.Tensor:
        """TRN-гейт с учётом веретена: база × текущая фаза осцилляции."""
        osc = getattr(self, "_spindle_osc", None)
        if osc is None:
            return self._gate
        return self._gate * osc

    @gate.setter
    def gate(self, value: torch.Tensor) -> None:
        # Внешний владелец (TRN) задаёт базу гейта; веретено — множитель поверх неё,
        # поэтому запись снаружи не теряется и не накапливается.
        self._gate = value

    def mark_phrase_boundary(self) -> None:
        """Явная граница фразы/клаузы: следующий step() обязан выдать burst-маркер.

        Маркер одноразовый — гасится в самом step(), чтобы не залипать в burst."""
        self._boundary_pending = True

    def step(self, x: torch.Tensor, ach: float, ne: float,
             boundary_burst: bool = False) -> torch.Tensor:
        """x [B,N] сенсорика → релейный выход в L4.

        LTS-вспышка (T-type Ca2+ burst) срабатывает только:
          - после длительного молчания (>50 мс) или во сне/пробуждении (vigilance < 0.5),
          - либо на границе предложения/клаузы или смене темы (`boundary_burst=True`),
            давая коре чёткий фазовый маркер начала новой синтаксической мысли!"""
        # Граница фразы: явный флаг или одноразовый маркер mark_phrase_boundary().
        # Маркер гасится здесь же — иначе relay залипнет в burst навсегда.
        boundary = bool(boundary_burst or getattr(self, "_boundary_pending", False))
        self._boundary_pending = False
        active = (x.abs().sum(0) > 1e-3).float()
        self.silence_ms = torch.where(active > 0, torch.zeros_like(active),
                                      self.silence_ms + 1.0)
        vigilance = ach + ne
        # В молчании, при низкой бодрости или на паузе конца предложения (boundary)
        # кальциевые каналы Т-типа де-инактивируются (h_T → 1)
        b_drive = 0.8 if boundary else 0.0
        h_inf = torch.clamp((self.silence_ms / 50.0)
                            + max(0.0, 0.6 - vigilance) + b_drive, 0.0, 1.0)
        self.h_t = 0.88 * self.h_t + 0.12 * h_inf
        if vigilance < 0.5 or self.silence_ms.mean() > 50.0 or boundary:
            self.mode = "burst"
        else:
            self.mode = "tonic"
        if self.mode == "burst":
            # Burst: нелинейная амплификация на границе фразы или после молчания
            gain = torch.where(self.silence_ms > 50.0,
                               torch.full_like(x[0], 3.0),
                               1.0 + 1.2 * self.h_t)
            out = torch.tanh(x * 2.0) * gain
            self.h_t = torch.where(active > 0, self.h_t * 0.25, self.h_t)
        else:
            # Внутри предложения (tonic) — чистый линейный релей без хаотичных вспышек!
            out = x
        spindle_applied = getattr(self, "_spindle_osc", None) is not None
        out = out * self.gate.unsqueeze(0)
        self._spindle_osc = None  # фаза веретена действует ровно на этот шаг
        tel = self._telemetry()
        if boundary:
            tel["boundary_markers"] = int(tel["boundary_markers"]) + 1
        tel.update({
            "boundary_burst": boundary,
            "b_drive": float(b_drive),
            "mode": self.mode,
            "h_t_mean": float(self.h_t.mean()),
            "spindle_applied": spindle_applied,
            "burst_cause": ("boundary" if boundary
                            else "vigilance" if vigilance < 0.5
                            else "silence" if self.silence_ms.mean() > 50.0
                            else "none"),
        })
        return out

    def spindle(self, t_ms: int, freq_hz: float = 10.0) -> torch.Tensor:
        """Сонное веретено: TRN-пачки ~10 Гц во время сна (NREM).

        Возвращает осциллирующий вектор фазы и ставит его множителем к TRN-гейту:
        ближайший step() прогоняет вход через осцилляцию (база гейта не портится —
        произведение снимается после шага), так что таламические реле пульсируют,
        запуская SWR-обвязку гиппокампа синхронно с корой. Факт вызова пишется
        в телеметрию, чтобы снаружи было видно, что веретено не декоративно."""
        phase = 2.0 * math.pi * freq_hz * t_ms / 1000.0
        osc = torch.full((self.n,), 0.5 + 0.5 * math.sin(phase))
        self._spindle_osc = osc
        self._telemetry()["spindle"] = {
            "t_ms": int(t_ms),
            "freq_hz": float(freq_hz),
            "phase_rad": float(phase),
            "osc_mean": float(osc.mean()),
        }
        return osc


class ThalamicHO:
    """Higher-order ядро (Pulvinar / MD): Core + Matrix параллельная маршрутизация
    (Jones 2001; Ziwago-Prime v4, Узел 2).

    * Core-путь (Parvalbumin+): выделяет ведущие L5-драйверы (winner_mask);
    * Matrix-путь (Calbindin+): сохраняет и синхронизирует слабые сигналы
      по фазовой когерентности LFP, не обнуляя тонкие детали контекста!"""
    def __init__(self, n_in: int, n_out: int, name: str = "HO"):
        self.n_in, self.n_out, self.name = n_in, n_out, name
        self.w = torch.randn(n_out, n_in) * 0.6
        self.w_matrix = torch.randn(n_out, n_in) * 0.25
        self.winner_mask = torch.zeros(n_in)
        self.open_gate = 0.0              # BG-гейт
        self.matrix_coherence: float = 1.0

    def step(self, l5: torch.Tensor, bg_release: float,
             lfp_coherence: float = 1.0) -> torch.Tensor:
        """l5 [B,N_in] драйв → Core + Matrix выход в следующую зону.

        bg_release: дизингибиция от BG (0=закрыт, 1=открыт)."""
        self.open_gate = bg_release
        self.matrix_coherence = float(max(0.2, min(1.5, lfp_coherence)))
        if bg_release < 0.5:
            return torch.zeros(l5.shape[0], self.n_out, device=l5.device)
        # Core-путь: отмечает топ-K сильнейших каналов для диагностики
        k = max(1, self.n_in // 4)
        thr = l5.topk(k, dim=1).values[:, -1:]
        core_mask = (l5 >= thr).float()
        self.winner_mask = core_mask.mean(0)
        # Matrix-путь: пропускает ВСЕ каналы (включая слабые фоновые обертоны),
        # модулируя их фазовой когерентностью LFP!
        core_drive = (l5 * (0.65 + 0.35 * core_mask)) @ self.w.t()
        matrix_drive = (l5 @ self.w_matrix.t()) * (0.35 * self.matrix_coherence)
        return torch.tanh(core_drive + matrix_drive) * self.open_gate
