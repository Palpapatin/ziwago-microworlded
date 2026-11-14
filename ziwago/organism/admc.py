"""Аллостатический драйв и метаболическое любопытство (ADMC, CONSC-ADMC-001).

Цель learning-progress/curiosity поверх готовой ветви ACh (аденозин подавляет
ACh и инициирует сон — `OrganismPhysiology._ach_pressure_suppression`). Здесь
реализованы недостающие контуры мотивации:

  * Прогресс обучения ΔE = E_ref − E_now (EWMA-опора ошибки предсказания):
    положительный, когда ошибка снижается (сжатие/обучение), ≈0 для плоской
    ошибки. Это решает проблему «шумного телевизора»: чисто стохастичный стимул
    держит ошибку высокой, но прогресс ≈0, поэтому J = ΔE − λ_ATP·Cost < 0 и
    организм теряет интерес к шуму (Schmidhuber 2010; Oudeyer & Kaplan 2007).
  * Метаболический гейт: исследование разрешено только в безопасной зоне АТФ;
    при истощении гейт → 0 (консервация энергии).
  * Аллостатический груз L_allo — leaky-интеграл хронического напряжения
    (аденозин + PAT + ошибка), ограничен; при превышении порога выдаёт bounded
    сигналы регуляции когнитивной нагрузки (глубина графа, порог уверенности,
    длина фразы). Сигналы советующие, не hard clamping токенов.

Все входы (error/ATP/Ado/PAT) — измеряемые величины; живая проводка входов
физиологии и регуляции декодера/графа — отдельный pending-подшаг. Без global
backprop, без n-грамм. Opt-in: curiosity_gain=0 → нулевой драйв.
"""
from __future__ import annotations

import math
from typing import Optional


def _sigmoid(x: float) -> float:
    """Устойчивая сигмоида для больших по модулю x."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class AllostaticCuriosityController:
    """Прогресс обучения, метаболическое любопытство и аллостатический груз."""

    def __init__(self, tau_ref: float = 100.0, lambda_atp: float = 0.5,
                 atp_safe: float = 0.5, atp_width: float = 0.1,
                 tau_allo: float = 1000.0, w_ado: float = 1.0,
                 w_pat: float = 1.0, w_err: float = 1.0,
                 allo_threshold: float = 1.0, allo_width: float = 1.0,
                 allo_max: float = 100.0, depth_reduce_max: float = 0.5,
                 conf_add_max: float = 0.3, phrase_reduce_max: float = 0.5,
                 curiosity_gain: float = 1.0):
        if tau_ref <= 0.0 or tau_allo <= 0.0:
            raise ValueError("tau_ref and tau_allo must be positive")
        if lambda_atp < 0.0:
            raise ValueError("lambda_atp must be non-negative")
        if atp_width <= 0.0 or allo_width <= 0.0:
            raise ValueError("atp_width and allo_width must be positive")
        if allo_max <= 0.0 or allo_threshold < 0.0:
            raise ValueError("allo_max must be positive, allo_threshold non-negative")
        for name, val in (("depth_reduce_max", depth_reduce_max),
                          ("phrase_reduce_max", phrase_reduce_max)):
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if conf_add_max < 0.0:
            raise ValueError("conf_add_max must be non-negative")
        if min(w_ado, w_pat, w_err) < 0.0:
            raise ValueError("allostatic weights must be non-negative")
        self.tau_ref = float(tau_ref)
        self.lambda_atp = float(lambda_atp)
        self.atp_safe = float(atp_safe)
        self.atp_width = float(atp_width)
        self.tau_allo = float(tau_allo)
        self.w_ado = float(w_ado)
        self.w_pat = float(w_pat)
        self.w_err = float(w_err)
        self.allo_threshold = float(allo_threshold)
        self.allo_width = float(allo_width)
        self.allo_max = float(allo_max)
        self.depth_reduce_max = float(depth_reduce_max)
        self.conf_add_max = float(conf_add_max)
        self.phrase_reduce_max = float(phrase_reduce_max)
        self.curiosity_gain = float(curiosity_gain)

        self.error_ref = None      # EWMA-опора ошибки (инициализируется на первом тике)
        self.prev_error = None
        self.last_progress = 0.0
        self.allo_load = 0.0
        self.last_regulation: Optional[dict] = None
        self.last_policy: Optional[dict] = None

    # ------------------------------------------------------------------
    def observe_error(self, error_now: float, dt: float) -> float:
        """Прогресс обучения ΔE = E_ref − E_now; обновляет EWMA-опору."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        e = max(0.0, float(error_now))
        if self.error_ref is None:
            self.error_ref = e
            self.prev_error = e
            self.last_progress = 0.0
            return 0.0
        progress = self.error_ref - e
        decay = math.exp(-dt / self.tau_ref)
        self.error_ref = self.error_ref * decay + e * (1.0 - decay)
        self.prev_error = e
        self.last_progress = progress
        return progress

    def exploration_gate(self, atp_level: float) -> float:
        """Метаболическое разрешение исследования ∈ [0, 1] (сигмоида по АТФ)."""
        return _sigmoid((float(atp_level) - self.atp_safe) / self.atp_width)

    def objective_value(self, progress: float, atp_cost: float) -> float:
        """J = ΔE_progress − λ_ATP·Cost_ATP (эпистемическая ценность нетто)."""
        return float(progress) - self.lambda_atp * max(0.0, float(atp_cost))

    def curiosity_drive(self, progress: float, atp_cost: float,
                        atp_level: float) -> float:
        """Gated-драйв = gain·gate(ATP)·J. Низкий АТФ → 0 (консервация)."""
        return (self.curiosity_gain * self.exploration_gate(atp_level)
                * self.objective_value(progress, atp_cost))

    def accumulate_allostatic(self, ado: float, pat_flag: bool,
                              error: float, dt: float) -> float:
        """L_allo = leaky-интеграл (w_ado·Ado + w_pat·PAT + w_err·E), bounded."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        decay = math.exp(-dt / self.tau_allo)
        source = (self.w_ado * max(0.0, float(ado))
                  + self.w_pat * (1.0 if pat_flag else 0.0)
                  + self.w_err * max(0.0, float(error)))
        self.allo_load = max(0.0, min(self.allo_max,
                                      self.allo_load * decay + source * dt))
        return self.allo_load

    def regulation(self) -> dict:
        """Bounded сигналы регуляции когнитивной нагрузки при L_allo > порога."""
        excess = max(0.0, self.allo_load - self.allo_threshold)
        strain = 1.0 - math.exp(-excess / self.allo_width)   # ∈ [0, 1)
        reg = {
            "allostatic_load": self.allo_load,
            "strain": strain,
            "graph_depth_factor": 1.0 - self.depth_reduce_max * strain,
            "confidence_threshold_add": self.conf_add_max * strain,
            "phrase_length_factor": 1.0 - self.phrase_reduce_max * strain,
            "conservation_mode": self.allo_load > self.allo_threshold,
        }
        self.last_regulation = reg
        return reg

    def apply_generation_policy(self, graph_depth: float, confidence_threshold: float,
                               phrase_length: float) -> dict:
        """Применяет bounded ADMC-регуляцию к параметрам генерации."""
        reg = self.regulation()
        graph_depth_eff = max(1, int(round(float(graph_depth) * reg["graph_depth_factor"])))
        confidence_eff = max(0.0, float(confidence_threshold) + reg["confidence_threshold_add"])
        phrase_eff = max(1, int(round(float(phrase_length) * reg["phrase_length_factor"])))
        policy = {
            "graph_depth": graph_depth_eff,
            "confidence_threshold": confidence_eff,
            "phrase_length": phrase_eff,
            "graph_depth_factor": reg["graph_depth_factor"],
            "confidence_threshold_add": reg["confidence_threshold_add"],
            "phrase_length_factor": reg["phrase_length_factor"],
            "conservation_mode": reg["conservation_mode"],
        }
        self.last_policy = policy
        return policy

    def step(self, error_now: float, atp_cost: float, atp_level: float,
             ado: float, pat_flag: bool, dt: float) -> dict:
        """Один такт ADMC: прогресс, цель, драйв, аллостатический груз, регуляция."""
        progress = self.observe_error(error_now, dt)
        drive = self.curiosity_drive(progress, atp_cost, atp_level)
        self.accumulate_allostatic(ado, pat_flag, error_now, dt)
        reg = self.regulation()
        return {
            "progress": progress,
            "objective": self.objective_value(progress, atp_cost),
            "exploration_gate": self.exploration_gate(atp_level),
            "curiosity_drive": drive,
            "prev_error": self.prev_error,
            **reg,
        }

    # ------------------------------------------------------------------
    def telemetry(self) -> dict:
        return {
            "error_ref": self.error_ref,
            "prev_error": self.prev_error,
            "last_progress": self.last_progress,
            "allostatic_load": self.allo_load,
            "lambda_atp": self.lambda_atp,
            "atp_safe": self.atp_safe,
            "allo_threshold": self.allo_threshold,
            "allo_max": self.allo_max,
            "curiosity_gain": self.curiosity_gain,
            "last_policy": getattr(self, "last_policy", None),
        }

    def export_state_dict(self) -> dict:
        return {
            "config": {
                "tau_ref": self.tau_ref, "lambda_atp": self.lambda_atp,
                "atp_safe": self.atp_safe, "atp_width": self.atp_width,
                "tau_allo": self.tau_allo, "w_ado": self.w_ado,
                "w_pat": self.w_pat, "w_err": self.w_err,
                "allo_threshold": self.allo_threshold, "allo_width": self.allo_width,
                "allo_max": self.allo_max, "depth_reduce_max": self.depth_reduce_max,
                "conf_add_max": self.conf_add_max, "phrase_reduce_max": self.phrase_reduce_max,
                "curiosity_gain": self.curiosity_gain,
            },
            "error_ref": self.error_ref, "prev_error": self.prev_error,
            "last_progress": self.last_progress, "allo_load": self.allo_load,
            "last_policy": self.last_policy,
        }

    def load_state_dict(self, sd: dict) -> None:
        cfg = sd["config"]
        if float(cfg["allo_max"]) != self.allo_max:
            raise ValueError("ADMC allo_max mismatch")
        for key, val in cfg.items():
            setattr(self, key, float(val))
        self.error_ref = sd["error_ref"]
        self.prev_error = sd["prev_error"]
        self.last_progress = float(sd["last_progress"])
        self.allo_load = float(sd["allo_load"])
        self.last_policy = sd.get("last_policy")