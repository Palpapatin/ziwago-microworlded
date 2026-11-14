"""Мета-когнитивная голова HOT (§II.4.2).

Наблюдает GW и выдаёт:
  confidence     — «насколько система уверена в текущем решении»
  predicted_loss — L̂: предсказанная ошибка следующего шага
  surprise       — |фактический err - L̂| (мета-сюрприз)
  novelty        — расстояние GW-контента к эпизодической базе
  uncertainty    — декомпозиция epistemic (дисперсия по ансамблю
                   извлечений GW) / aleatoric (шум входа)

Обучение: L_meta = MSE + λ·Brier (калибровка уверенности).
calibration_corr = corr(confidence, 1−err) — мера КАЛИБРОВКИ уверенности,
НЕ meta-d'/d': это согласованность уверенности с фактической точностью на
последних 64 пробах, а не отношение чувствительностей meta-d' и d'.

Калибровка (2026-11-14): уверенность обязана быть вероятностью, а не 0/1.
Два дефекта, из-за которых она насыщалась (0.995 вместо цели 0.9, logit
+10.2 вместо +4.4, фазовое отставание 3–4 пробы из 8), устранены физиологично:

1. Readout нормирован по масштабу популяционного состояния (делительный
   гомеостаз, `_readout`): преактивация делится на max(1, ‖h‖), поэтому гейн
   выхода не растёт вместе с нормой состояния, плюс физиологичный потолок
   логита `_LOGIT_MAX` — p ∈ [0.047, 0.953] по построению.
2. Пластичность — ТРЁХФАКТОРНОЕ правило: след пресинаптической активности
   (`trace_pre`, «холинергический тег») × текущая локальная ошибка `local_err`
   (дофаминовый 3-й фактор). Раньше след накапливал САМУ ОШИБКУ
   (`elig ← 0.75·elig + local_err⊗h`), что превращало вес в НЧ-фильтр цели
   с τ≈4 пробы: он уползал до клипа ±3 даже когда цель уже достигнута
   (псевдоградиент не знает о насыщении), а на смене цели тратил 3–4 пробы
   на разряд — отсюда перерегулирование и фазовое отставание.

Замер (тот же сценарий, 48 проб, блоки 8 через 8): corr 0.2937 → 0.595, ECE
0.2917 → 0.1197, conf 0.005…0.995 → 0.137…0.864. Остаточный лаг — ровно одна
проба: уверенность читается ДО апдейта на этой же пробе, поэтому физический
минимум запаздывания = 1 проба.

Выход управляет поведением: confidence < θ_low гейтит действия,
увеличивает β_GW (крутизну ignition) и приоритет реплея. Все три обучаемых
выхода (confidence, L̂, novelty_unit) сведены в gates; перенос гейтов в
параметры потребителей — публичный контракт `apply_gates()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional

import torch

from ..config import MetaHeadCfg


@dataclass
class MetaOut:
    confidence: float
    predicted_loss: float
    surprise: float
    novelty: float
    novelty_unit: float     # обучаемый выходной юнит новизны [0,1] → replay_priority
    epistemic: float
    aleatoric: float
    # corr(confidence, 1−err) — КАЛИБРОВКА уверенности, не meta-d'/d'
    calibration_corr: float
    gates: Dict[str, float]  # → поведение/BG/TRN


@dataclass
class GateApplication:
    """Гейты MetaHead, перенесённые в параметры потребителей.

    beta_gw          — β_GW с добавкой гейта (крутизна ignition): вход
                       `beta_boost` в `GlobalWorkspace.compete()`
                       (workspace.py:86 `beta = (c.beta_gw + beta_boost)·(1+0.5·ne)`).
    replay_priority  — приоритет реплея с добавкой новизны/сюрприза/эпистемики.
    trn_gain         — множитель гейна TRN, тензор той же формы, что вход
                       (→ `SONReflectionVerdict.trn_column_gain_modulation`,
                       son_reflection.py:249).
    bg_threshold     — 0/1 ворот допуска действия в базальные ганглии.
    """
    beta_gw: float
    replay_priority: float
    trn_gain: torch.Tensor
    bg_threshold: float


class MetaHead:
    """Биофизическая мета-когнитивная популяция (BA10 / префронтальный полюс,
    Genesis level4/meta_head.py): фиксированная рекуррентная проекция в спайковый
    резервуар + чисто локальная синаптическая пластичность без backprop."""

    # Честная телеметрия подключения (трек M, phantom-effect): живой путь
    # (organism/integrated.py:294) вызывает только step(); apply_gates() и
    # gates НЕ потребляются ⇒ гейты считаются и выбрасываются.
    # Точка врезки: сразу после `meta = son.meta_head.step(...)` (integrated.py:294)
    # вызвать apply_gates(...) и раздать app.* потребителям; затем поставить True.
    GATES_CONNECTED = False

    def __init__(self, cfg: MetaHeadCfg, d_gw: int, seed: int = 0):
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        # Фиксированная случайная дендритная экспансия из GW в популяцию BA10 (не обучается backprop'ом!)
        self.w_in = torch.randn(cfg.d_hidden, d_gw, generator=g) / max(1.0, d_gw ** 0.5)
        self.w_rec = torch.randn(cfg.d_hidden, cfg.d_hidden, generator=g) * (0.25 / max(1.0, cfg.d_hidden ** 0.5))
        self.w_rec.fill_diagonal_(0.0)
        # Локальные синаптические веса 3 популяционных выходов: [confidence, predicted_loss, novelty]
        self.w_out = torch.zeros(3, cfg.d_hidden)
        self.b_out = torch.tensor([0.0, 0.0, -0.5])
        # Состояние популяции BA10 и след пресинаптического «тега» (eligibility)
        self.h_state = torch.zeros(cfg.d_hidden)
        self.trace_gw = torch.zeros(d_gw)
        self.trace_pre = torch.zeros(cfg.d_hidden)
        # Бегущий центроид опыта для оценки истинной новизны
        self.gw_centroid = torch.zeros(d_gw)
        self.err_hist: List[float] = []
        self.conf_hist: List[float] = []

    # ------------------------------------------------------------------
    # Потолок логита: p ∈ [1/(1+e^3), 1/(1+e^-3)] = [0.047, 0.953]. Уверенность
    # остаётся вероятностью при любых весах и любой норме состояния.
    _LOGIT_MAX: float = 3.0

    def _readout(self, h: torch.Tensor) -> torch.Tensor:
        """[conf, L̂, nov] — логистический readout с делительным гомеостазом.

        Преактивация делится на масштаб состояния max(1, ‖h‖): гейн выхода не
        растёт вместе с нормой популяционной активности (иначе рост ‖h‖ сам по
        себе загонял уверенность в 0/1, независимо от свидетельств). При слабом
        драйве (‖h‖ < 1) деления нет — «нет свидетельств, нет уверенности».
        """
        scale = max(1.0, float(h.norm()))
        z = (h / scale) @ self.w_out.t() + self.b_out
        return torch.sigmoid(z.clamp(-self._LOGIT_MAX, self._LOGIT_MAX))

    def step(self, gw_broadcast: torch.Tensor,
             gw_content: List[torch.Tensor],
             actual_error: float,
             reward: float = 0.0,
             sensory_frame: Optional[torch.Tensor] = None,
             learn: bool = True) -> MetaOut:
        c = self.cfg
        gw = gw_broadcast.detach().float().flatten()
        if gw.numel() != self.trace_gw.numel():
            gw = torch.nn.functional.pad(gw, (0, max(0, self.trace_gw.numel() - gw.numel())))[:self.trace_gw.numel()]
        if sensory_frame is not None:
            sf = sensory_frame.detach().float().flatten()
            sf_pad = torch.nn.functional.pad(sf, (0, max(0, gw.numel() - sf.numel())))[:gw.numel()]
            gw = 0.6 * gw + 0.4 * sf_pad
        self.trace_gw = self.trace_gw * 0.65 + gw * 0.35

        # Рекуррентный шаг популяции BA10 (без градиентов и без скрытого backprop)
        drive = self.trace_gw @ self.w_in.t() + self.h_state @ self.w_rec.t()
        self.h_state = 0.65 * self.h_state + 0.35 * torch.tanh(drive)
        out = self._readout(self.h_state)
        conf, l_hat = float(out[0]), float(out[1])
        novelty_unit = float(out[2])

        # Истинная новизна опыта: отклонение от выученного центроида знакомых состояний
        nov_dist = float((gw - self.gw_centroid).pow(2).mean().sqrt().item())
        nov = min(1.0, math.tanh(nov_dist * 2.5))
        if learn:
            self.gw_centroid = 0.90 * self.gw_centroid + 0.10 * gw

        # Эпистемическая неопределённость (BA10): чувствительность популяции к мембранному
        # шуму + незнакомость паттерна относительно выученного следа (Genesis MetaHead)
        # Observation must not advance the cortical noise stream.
        with torch.random.fork_rng(devices=[]):
            ens = [self._readout(torch.tanh(drive + torch.randn_like(drive) * 0.45))
                   for _ in range(c.n_ensemble)]
        ens_var = float(torch.stack(ens).var(0, unbiased=False).mean())
        epistemic = float(min(1.0, ens_var * 15.0 + 0.35 * nov * min(1.0, actual_error)))
        # Алеаторная: шум входа — дисперсия по слотам
        if gw_content:
            aleatoric = float(torch.stack([x.float().flatten()
                                           for x in gw_content])
                              .var(0, unbiased=False).mean())
        else:
            aleatoric = 0.5
        surprise = abs(actual_error - l_hat)
        self.err_hist.append(actual_error)
        self.conf_hist.append(conf)

        # Трёхфакторное правило (локальное, без backprop в w_in):
        #   Δw = lr · local_err(3-й фактор) ⊗ trace_pre(тег)
        # След живёт на ПРЕсинаптической активности, ошибка берётся текущая.
        # Если след накапливает саму ошибку (outer(local_err, h)), вес становится
        # НЧ-фильтром цели (τ≈4 пробы) и уползает до клипа ±3 уже после
        # достижения цели — перерегулирование и лаг 3–4 пробы из 8.
        err_clipped = min(1.0, max(0.0, float(actual_error)))
        target = torch.tensor([1.0 - err_clipped, err_clipped, nov])
        local_err = target - out  # [3]
        if learn:
            self.trace_pre = 0.75 * self.trace_pre + self.h_state
            lr_eff = max(c.lr * 15.0, 0.15) * (1.0 + 0.5 * abs(float(reward)) + 0.5 * surprise)
            self.w_out = (self.w_out + lr_eff * torch.outer(local_err, self.trace_pre)).clamp(-3.0, 3.0)
            self.b_out = (self.b_out + lr_eff * 0.5 * local_err).clamp(-2.5, 2.5)

        # КАЛИБРОВКА, не meta-d'/d': corr(уверенность, 1−err) — согласованность
        # уверенности с фактической точностью за последние 64 пробы.
        if len(self.err_hist) > 8:
            import numpy as np
            c_arr = np.asarray(self.conf_hist[-64:])
            e_arr = 1.0 - np.asarray(self.err_hist[-64:])
            m = float(np.corrcoef(c_arr, e_arr)[0, 1]) \
                if c_arr.std() > 1e-6 and e_arr.std() > 1e-6 else 0.0
            calibration_corr = m if np.isfinite(m) else 0.0
        else:
            calibration_corr = 0.0
        gates = {
            "bg_threshold": 1.0 if conf < c.theta_low else 0.0,
            "beta_gw_boost": 0.3 if conf < c.theta_uncertain else 0.0,
            "replay_priority": epistemic + 0.5 * surprise + 0.5 * novelty_unit,
            "trn_gain": 1.0 - conf * 0.5,
        }
        return MetaOut(confidence=conf, predicted_loss=l_hat,
                       surprise=surprise, novelty=nov,
                       novelty_unit=novelty_unit,
                       epistemic=epistemic, aleatoric=aleatoric,
                       calibration_corr=calibration_corr, gates=gates)

    # ------------------------------------------------------------------
    def apply_gates(self, out: MetaOut, trn_gain: torch.Tensor,
                    beta_gw: float, replay_priority: float) -> GateApplication:
        """Перенести гейты MetaOut в параметры потребителей (публичный контракт).

        Аддитивные гейты складываются с базовым значением потребителя,
        мультипликативный (trn_gain) — умножает тензор гейнов той же формы.

        Точка врезки в organism/integrated.py: сразу после
        `meta = son.meta_head.step(...)` (:279) вызвать
        `app = son.meta_head.apply_gates(meta, trn_gain=<гейны TRN>,
         beta_gw=self.cfg.beta_gw, replay_priority=<приоритет реплея>)` и
        передать `app.bg_threshold` в ворота BG/действия, `app.beta_gw` — в
        `gw.compete(beta_boost=...)`, `app.replay_priority` — в очередь реплея,
        `app.trn_gain` — в `SONReflectionVerdict.trn_column_gain_modulation`.
        """
        g = out.gates
        trn_t = trn_gain if torch.is_tensor(trn_gain) \
            else torch.as_tensor(trn_gain, dtype=torch.float32)
        return GateApplication(
            beta_gw=float(beta_gw) + float(g["beta_gw_boost"]),
            replay_priority=float(replay_priority) + float(g["replay_priority"]),
            trn_gain=trn_t * float(g["trn_gain"]),
            bg_threshold=float(g["bg_threshold"]),
        )

    def telemetry(self, recent: Optional[MetaOut] = None) -> Dict[str, object]:
        """Честная телеметрия: не подключённые гейты явно помечены как отключённые."""
        payload: Dict[str, object] = {
            "gates_connected": bool(self.GATES_CONNECTED),
            "gate_contract": "apply_gates(out, trn_gain, beta_gw, replay_priority)",
            "gate_keys": ["bg_threshold", "beta_gw_boost", "replay_priority", "trn_gain"],
            "novelty_unit_consumed": False,
            "live_consumer_points": [],
        }
        if recent is not None:
            payload.update({
                "confidence": float(recent.confidence),
                "novelty_unit": float(recent.novelty_unit),
                "replay_priority": float(recent.gates["replay_priority"]),
                "bg_threshold": float(recent.gates["bg_threshold"]),
            })
        return payload
