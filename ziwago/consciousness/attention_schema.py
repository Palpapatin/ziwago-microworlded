"""Схема внимания AST (§II.4.3): внутренняя модель внимания системы.

AST строит контроллер, отвечающий: «что сейчас в фокусе, с какой силой,
насколько это точно и почему». Схема — не само внимание, а модель для
нисходящего контроля (Graziano 2013).

Состояние: current_focus (какой модуль), probability (вероятностное
распределение внимания), priority (КОНКУРЕНТНЫЙ приоритет ��0..1]),
expected_accuracy, reason (триггер: novelty/error/drive), prediction.
Обучение: L_schema = MSE(pred, фактический фокус) + MSE(prob, факт. ошибка).
Выход: конкурентный приоритет внимания → TRN (см. `ziwago/thalamus/trn.py`).

v22 — устранено «притворство» гейна:
  БЫЛО:  gain = softmax(logits) * alpha_ast(0.1) ≈ 0.02 на модуль, и TRN
         получал константный множитель (0.5 + 0.02) — приоритет внимания
         НЕ соревновался (см. `thalamus/trn.py`).
  СТАЛО: gain — конкурентный приоритет в [0,1]:
         * логиты нормируются по размаху (масштаб-инвариантно: форма
           профиля важнее абсолютного разброса весов),
         * softmax по нормированным логитам даёт резкий профиль
           (победитель ≈ 0.83 при margin=3.0),
         * rel = p / p_max  → победитель получает 1.0,
         * alpha_ast трактуется как БАЗОВОЕ открытие реле поисковым светом
           (floor): ни один сектор не закрывается полностью, но проигравшие
           получают ≈alpha_ast от максимума.
  Итог: выигрыш внимания доходит до TRN с правильным знаком (победитель
  открыт сильнее) и масштабом (0..1), а не как константа.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from ..config import AttentionSchemaCfg


@dataclass
class ASTOut:
    focus: int
    gain: torch.Tensor          # [n_modules] КОНКУРЕНТНЫЙ приоритет TRN (победитель = 1.0)
    expected_accuracy: float
    reason: str
    prediction: int
    report: Dict                # интроспекция
    probability: Optional[torch.Tensor] = None  # softmax-распределение (для отчёта/обучения)


class AttentionSchema:
    """Внутренняя схема внимания с предиктивным обучением."""

    MODULES = ["SENS", "ASSOC", "PFC", "HPC", "BG"]

    def __init__(self, cfg: AttentionSchemaCfg, seed: int = 0,
                 competition_margin: float = 3.0):
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        self.w_pred = torch.randn(cfg.n_modules, cfg.d_schema,
                                  generator=g) * 0.1
        self.state = torch.zeros(cfg.d_schema)
        self.prob = torch.ones(cfg.n_modules) / max(1, cfg.n_modules)
        self.priority = torch.ones(cfg.n_modules)
        self.exp_acc = 0.5
        self.focus = 0
        self.reason = "baseline"
        # Резкость конкуренции: во сколько раз «логарифмический разрыв»
        # между лидером и аутсайдерами нормируется до softmax-профиля.
        self.competition_margin = float(competition_margin)

    # ------------------------------------------------------------------
    def _competitive_priority(self, logits: torch.Tensor) -> torch.Tensor:
        """Конкурентный приоритет внимания в [0,1].

        Логиты нормируются по размаху (масштаб-инвариантно), затем softmax
        задаёт резкий профиль, и победитель приводится к 1.0. Итог:
        winner = 1.0, проигравшие ≈ alpha_ast (базовое открытие)."""
        spread = float(logits.max() - logits.min())
        if spread > 1e-6:
            scaled = (logits - logits.min()) / spread * self.competition_margin
        else:
            scaled = torch.zeros_like(logits)
        p = torch.softmax(scaled, dim=0)
        rel = p / p.max().clamp(min=1e-8)
        floor = float(min(max(self.cfg.alpha_ast, 0.0), 1.0))
        return floor + (1.0 - floor) * rel

    # ------------------------------------------------------------------
    def step(self, novelty: float, error: float, drive: float,
             actual_gain: float, actual_focus: int,
             learn: bool = True) -> ASTOut:
        c = self.cfg
        # Обновление состояния схемы от признаков
        feat = torch.tensor([novelty, error, drive, actual_gain,
                             float(actual_focus) / c.n_modules])
        feat = torch.nn.functional.pad(feat, (0, max(
            0, c.d_schema - feat.numel())))[:c.d_schema]
        self.state = 0.8 * self.state + 0.2 * feat
        # Фокус: argmax модуля по схеме
        logits = self.w_pred @ self.state
        self.focus = int(logits.argmax())
        self.prediction = int(logits.topk(2).indices[-1])
        # Причина фокуса: доминирующий признак
        self.reason = ["novelty", "error", "drive",
                       "baseline"][int(torch.tensor(
                           [novelty, error, drive, 0.0]).argmax())]
        # Вероятностное распределение внимания (для обучения) и КОНКУРЕНТНЫЙ
        # приоритет (для TRN) — разные величины, обе честно названы.
        self.prob = torch.softmax(logits, dim=0)
        self.priority = self._competitive_priority(logits)
        tgt = torch.zeros(c.n_modules)
        tgt[actual_focus] = 1.0
        if learn:
            # L_schema: предсказание фокуса и гейна
            self.w_pred += c.lr * torch.outer(tgt - self.prob, self.state)
            self.exp_acc += c.lr * (actual_gain - self.exp_acc)
        report = {
            "attention_state": {
                "current_focus": self.MODULES[self.focus],
                "gain": [round(float(g), 3) for g in self.priority],
                "attention_probability": [round(float(p), 3) for p in self.prob],
                "expected_accuracy": round(self.exp_acc, 3),
                "reason": self.reason,
                "schema_loss": float((tgt - self.prob).pow(2).mean())
                             if learn else 0.0,
                "prediction": self.MODULES[self.prediction],
            }
        }
        return ASTOut(focus=self.focus,
                      gain=self.priority,
                      expected_accuracy=self.exp_acc,
                      reason=self.reason, prediction=self.prediction,
                      report=report,
                      probability=self.prob)
