"""Merged dual-system hippocampus (hippocampusidea.md; ARCHITECTURE_MAPPING §8).

Консолидация четырёх реализаций гиппокампа (nw8 rate-based, nw9 async-Hopfield +
theta-timing + explore-recall, nw10 трисинаптический + concept cells + VSA-DHT) в
один модуль, обслуживающий дуальную систему мышления (Kahneman System 1/2):

    Fast Drafter (System 1)  <- WorkingMemory: MergedLaplaceBank + θ-γ indexer
    Slow Verifier (System 2) <- EpisodicCore: nw10 Hippocampus (EC→DG→CA3→CA1,
                                нейрогенез, temporal STDP, concept cells, щит Фузи,
                                SWR/REM) + VSA-DHT multiprobe + HippoRAG PPR
    Arbiter / GW             <- NoveltyArbiter: CA1 mismatch → DA/NE,
                                CA3 resonance → accept / regenerate / ask / unknown

Принципы (целевая архитектура L1→L2):
  * композиция вместо наследования; один механизм = один владелец;
  * живой nw10 episodic core переиспользуется, а не дублируется;
  * оживают пути, которые в v19 только записывались, но не читались
    (VSA-DHT retrieval, HippoRAG PPR, θ-γ indexer, Laplace elapsed-time);
  * opt-in: модуль не меняет живой путь v19 и не трогает мастер-чекпоинт.

CLS (McClelland, McNaughton & O'Reilly 1995): быстрая эпизодическая память (CA3)
и медленная неокортикальная память работают как дополняющие системы.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

import torch

from ..config import HippocampusCfg
from .hippocampus import (
    CompositionalThetaGammaIndexer,
    Hippocampus,
    HippoOut,
)


# ---------------------------------------------------------------------------
# Fast Drafter (System 1): рабочая память
# ---------------------------------------------------------------------------
class MergedLaplaceBank:
    """Консолидация nw9 LaplaceTimingBank + nw10 LaplaceBank в один владелец.

    Спектр клеток времени Shankar & Howard (2012): геометрический спектр
    экспоненциальных интеграторов τ_k = τ_min·c^k — дискретное преобразование
    Лапласа входного потока, память O(K·D). Объединяет:
      * time_cells      — разностный фильтр Поста-Виддера (когда появился вход);
      * context_summary — средний профиль (nw9);
      * slow_profile    — медленная половина спектра → длинный контекст (nw10);
      * estimate_elapsed_time — оценка давности признака по отношению шкал (nw9, У-5).
    """

    def __init__(self, feature_dim: int, num_scales: int = 8,
                 tau_min: float = 4.0, tau_max: float = 512.0, dt: float = 1.0):
        self.feature_dim = int(feature_dim)
        self.num_scales = max(2, int(num_scales))
        self.tau_min = max(1.0, float(tau_min))
        self.tau_max = max(self.tau_min * 2.0, float(tau_max))
        self.dt = float(dt)
        taus = torch.exp(torch.linspace(
            math.log(self.tau_min), math.log(self.tau_max), self.num_scales))
        self.taus = taus
        self.alphas = torch.exp(-self.dt / taus).view(-1, 1, 1)  # [K,1,1]
        self.state: Optional[torch.Tensor] = None                # [K,B,F]

    def reset_state(self, batch: int = 1, device=None, dtype=torch.float32) -> None:
        self.state = torch.zeros(self.num_scales, batch, self.feature_dim,
                                 device=device, dtype=dtype)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """x [B,F] → обновлённое состояние [K,B,F]."""
        if (self.state is None or self.state.shape[1] != x.shape[0]
                or self.state.device != x.device):
            self.reset_state(x.shape[0], x.device, x.dtype)
        self.state = self.alphas * self.state + x.detach().unsqueeze(0)
        return self.state

    def time_cells(self) -> torch.Tensor:
        if self.state is None:
            return torch.zeros(self.num_scales, 1, self.feature_dim)
        left = torch.roll(self.state, 1, 0)
        right = torch.roll(self.state, -1, 0)
        return torch.relu(self.state - 0.5 * (left + right))

    def context_summary(self) -> torch.Tensor:
        if self.state is None:
            return torch.zeros(1, self.feature_dim)
        return self.state.mean(0)

    def slow_profile(self) -> torch.Tensor:
        if self.state is None:
            return torch.zeros(1, self.feature_dim)
        k0 = self.num_scales // 2
        return self.state[k0:].mean(0)

    def estimate_elapsed_time(self, feature_idx: int, batch_idx: int = 0) -> float:
        """Давность признака: отношение медленной/быстрой шкал → log / (1/τ_fast - 1/τ_slow)."""
        if self.state is None:
            return 0.0
        col = self.state[:, batch_idx, feature_idx].clamp_min(1e-8)
        k_fast, k_slow = 1, self.num_scales - 1
        inv_diff = (1.0 / float(self.taus[k_fast])) - (1.0 / float(self.taus[k_slow]))
        if abs(inv_diff) < 1e-9:
            return 0.0
        ratio = float((col[k_slow] / col[k_fast]).item())
        if ratio <= 1.0:
            return 0.0
        return max(0.0, math.log(ratio) / inv_diff)


class ThetaClock:
    """Тета-осциллятор гиппокампа (~7 Гц) с выравниванием по токенам (порт nw9).

    Биология: тета-цикл (~143 мс) делится на пик (кодирование, высокий ACh) и
    спад/впадину (извлечение, низкий ACh). Проблема текста: при dt=1 мс и 1 шаг =
    1 токен реколл разрешён лишь раз в 143 токена — контекст коры (~32 токена)
    к тому моменту мёртв. Режимы (рантайм, веса не затрагиваются):
      * baseline — 143 шага на цикл (исходные 7 Гц);
      * fast     — цикл = cycle_steps шагов при тех же 7 Гц (масштаб времени);
      * token    — фаза строго от счётчика токенов (1 токен = 1/cycle_steps цикла).
    """

    def __init__(self, freq_hz: float = 7.0, dt: float = 1.0,
                 mode: str = "token", cycle_steps: float = 8.0):
        self.freq_hz = float(freq_hz)
        self.base_dt = float(dt)
        self.dt = float(dt)
        self.t = 0.0
        self._retrieved_in_cycle = False
        self._last_phase_sign = 1.0
        self._token_index = 0
        self._mode = "baseline"
        self._cycle_steps = 0.0
        self.configure(mode, cycle_steps)

    def configure(self, mode: str = "baseline", cycle_steps: float = 0.0) -> None:
        if mode == "baseline":
            self._mode = "baseline"
            self.dt = self.base_dt
            self._cycle_steps = 0.0
        elif mode in ("fast", "token"):
            steps = float(cycle_steps)
            if steps <= 0.0:
                raise ValueError("cycle_steps должен быть > 0")
            self._mode = mode
            self._cycle_steps = steps
            self.dt = self.base_dt if mode == "token" else (1000.0 / self.freq_hz) / steps
        else:
            raise ValueError(f"неизвестный режим тета-тайминга: {mode}")
        self.reset_state()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def cycle_steps(self) -> float:
        if self._mode == "baseline":
            return 1000.0 / (self.freq_hz * self.base_dt)
        return self._cycle_steps

    def reset_state(self) -> None:
        self.t = 0.0
        self._retrieved_in_cycle = False
        self._last_phase_sign = 1.0
        self._token_index = 0

    def step(self) -> Dict[str, float]:
        if self._mode == "token":
            self._token_index += 1
            frac = (self._token_index % self._cycle_steps) / self._cycle_steps
            theta = math.sin(2.0 * math.pi * frac)
            phase = 2.0 * math.pi * frac
        else:
            self.t += self.dt
            two_pi_f = 2.0 * math.pi * self.freq_hz * 1e-3
            theta = math.sin(two_pi_f * self.t)
            phase = (two_pi_f * self.t) % (2.0 * math.pi)
        if theta > 0.0 and self._last_phase_sign <= 0.0:
            self._retrieved_in_cycle = False
        self._last_phase_sign = theta
        is_retrieval = theta <= 0.0
        return {
            "theta": theta,
            "phase": phase,
            "is_retrieval": is_retrieval,
            "is_encoding": not is_retrieval,
            "retrieval_ready": is_retrieval and not self._retrieved_in_cycle,
        }

    def mark_retrieved(self) -> None:
        self._retrieved_in_cycle = True

    def current_theta(self) -> float:
        if self._mode == "token":
            frac = (self._token_index % self._cycle_steps) / self._cycle_steps
            return math.sin(2.0 * math.pi * frac)
        return math.sin(2.0 * math.pi * self.freq_hz * 1e-3 * self.t)

    def is_retrieval_phase(self) -> bool:
        return self.current_theta() <= 0.0

    def retrieval_allowed(self) -> bool:
        """Реколл разрешён на спаде θ и не чаще одного раза за цикл.

        Это и есть гейт «1 реколл за тета-цикл»: без него `mark_retrieved` был
        мёртвым методом, который вызывали только в тестах.
        """
        return bool(self.is_retrieval_phase() and not self._retrieved_in_cycle)


# ---------------------------------------------------------------------------
# Arbiter / Global Workspace: новизна + резонанс → решение дуальной системы
# ---------------------------------------------------------------------------
@dataclass
class ArbiterCfg:
    """Пороги протокола Arbiter из hippocampusidea.md."""
    accept_resonance: float = 0.7    # CA3 resonance > 0.7 → принять черновик
    unknown_resonance: float = 0.3   # CA3 resonance < 0.3 → «я не знаю»
    stick_entropy: float = 0.15      # энтропия ниже → залипание → NE+2-AG рестарт
    mismatch_regenerate: float = 0.3  # CA1 mismatch выше → перегенерация


@dataclass
class ArbiterDecision:
    decision: str          # accept | regenerate | ask | unknown
    confidence: float
    resonance: float
    mismatch: float
    novelty_da: float
    novelty_ne: float
    ne_restart: bool       # сигнал NE+2-AG для выхода из залипания
    reason: str


class NoveltyArbiter:
    """SON-подобный арбитр дуальной системы (hippocampusidea.md).

    Принимает черновик Fast Drafter и вердикт Slow Verifier:
      * resonance ≥ 0.7              → accept (факты совпали);
      * resonance < 0.3              → unknown («я не знаю», анти-галлюцинация);
      * 0.3…0.7 + залипание/mismatch → regenerate (NE+2-AG рестарт);
      * иначе                        → ask (уточняющий вопрос).
    Новизна CA1 (mismatch) даёт фазический DA (VTA) и NE (LC).
    """

    def __init__(self, cfg: Optional[ArbiterCfg] = None):
        self.cfg = cfg or ArbiterCfg()

    def decide(self, resonance: float, mismatch: float,
               novelty_da: float = 0.0, novelty_ne: float = 0.0,
               entropy: Optional[float] = None) -> ArbiterDecision:
        c = self.cfg
        r = float(max(0.0, min(1.0, resonance)))
        m = float(max(0.0, mismatch))
        ne_restart = False
        if r >= c.accept_resonance:
            decision, confidence, reason = "accept", r, "ca3_resonance_high"
        elif r < c.unknown_resonance:
            decision, confidence, reason = "unknown", 1.0 - r, "ca3_resonance_low"
        else:
            stuck = entropy is not None and float(entropy) < c.stick_entropy
            if stuck:
                decision, confidence, reason = "regenerate", 0.5, "low_entropy_stick"
                ne_restart = True
            elif m > c.mismatch_regenerate:
                decision, confidence, reason = "regenerate", 0.5, "ca1_mismatch_high"
                ne_restart = True
            else:
                decision, confidence, reason = "ask", 0.5, "partial_resonance"
        return ArbiterDecision(
            decision=decision, confidence=float(confidence), resonance=r,
            mismatch=m, novelty_da=float(novelty_da), novelty_ne=float(novelty_ne),
            ne_restart=ne_restart, reason=reason,
        )


# ---------------------------------------------------------------------------
# DualHippocampus: единый фасад дуальной системы
# ---------------------------------------------------------------------------
@dataclass
class DualHippoOut:
    # Fast Drafter (System 1) — рабочая память
    laplace_state: torch.Tensor
    time_cells: torch.Tensor
    context_summary: torch.Tensor
    slow_profile: torch.Tensor
    elapsed_time: float
    theta_gamma_slots: Tuple[int, ...]
    # Slow Verifier (System 2) — эпизодическая память
    hippo: HippoOut
    resonance: float
    recalled_episode_ids: List[int]
    recall_scores: List[float]
    concept_chain: List[str]
    predicted_next: Optional[torch.Tensor]
    # Arbiter / GW
    arbiter: ArbiterDecision
    theta: Dict[str, float] = field(default_factory=dict)


class DualHippocampus:
    """Консолидированный гиппокамп дуальной системы (opt-in поверх живого nw10 core).

    Не меняет живой путь v19: episodic core — тот же nw10 Hippocampus, но теперь
    его распределённая память (VSA-DHT) и граф HippoRAG читаются, а не только
    пишутся; рабочая память (Laplace + θ-γ) и арбитр новизны дают интерфейсы
    Fast Drafter / Slow Verifier / Arbiter из hippocampusidea.md.
    """

    def __init__(self, cfg: HippocampusCfg, input_dim: int, seed: int = 0,
                 theta_mode: str = "token", theta_cycle_steps: float = 8.0,
                 laplace_scales: int = 8, arbiter_cfg: Optional[ArbiterCfg] = None):
        self.cfg = cfg
        self.input_dim = int(input_dim)
        self.n_ec = int(cfg.n_ec)
        # Slow Verifier: живой трисинаптический core (без дублирования)
        self.episodic = Hippocampus(cfg, seed)
        # Fast Drafter: рабочая память
        self.laplace = MergedLaplaceBank(self.n_ec, num_scales=laplace_scales)
        self.indexer = CompositionalThetaGammaIndexer(
            n_slots=4, k_per_slot=64, d_slot=max(4, self.n_ec // 4), seed=seed + 7)
        # Общие часы гиппокампа (token-aligned)
        self.theta = ThetaClock(mode=theta_mode, cycle_steps=theta_cycle_steps)
        # Arbiter / GW
        self.arbiter = NoveltyArbiter(arbiter_cfg)
        self.tick_n = 0
        # Телеметрия θ-гейта: сколько тактов гейт закрыл и сколько реколлов прошло
        self.theta_gate_blocks = 0
        self.theta_recalls = 0

    # -- служебное ---------------------------------------------------------
    def _cosine(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.detach().float().flatten()
        b = b.detach().float().flatten()
        n = min(a.numel(), b.numel())
        if n == 0:
            return 0.0
        a, b = a[:n], b[:n]
        denom = float(a.norm().item() * b.norm().item())
        if denom < 1e-8:
            return 0.0
        return float(torch.dot(a, b).item() / denom)

    def _resonance(self, query_ec: torch.Tensor) -> float:
        """CA3 resonance: максимальное косинусное сходство запроса с известным эпизодом."""
        best = 0.0
        for ep in self.episodic.episodes:
            best = max(best, self._cosine(query_ec, ep[0]))
        return best

    def _bind_context(self, ec: torch.Tensor, salience: float, learn: bool) -> Tuple[int, ...]:
        """Квантует EC в ролевые слоты θ-γ indexer и связывает композиционный след."""
        x = ec.detach().float().flatten()
        d_slot = self.indexer.d_slot
        slot_ids: List[int] = []
        for m in range(self.indexer.n_slots):
            chunk = x[m * d_slot:(m + 1) * d_slot]
            if chunk.numel() < d_slot:
                chunk = torch.nn.functional.pad(chunk, (0, d_slot - chunk.numel()))
            chunk = chunk / chunk.norm().clamp(min=1e-6)
            sims = self.indexer.codebooks[m] @ chunk
            slot_ids.append(int(sims.argmax().item()))
        return self.indexer.encode_and_bind(tuple(slot_ids), salience=salience, learn=learn)

    # -- Fast Drafter (System 1) -------------------------------------------
    def fast_context(self) -> Dict[str, object]:
        """Контекст рабочей памяти для Fast Drafter (Laplace + θ-γ)."""
        return {
            "laplace_state": self.laplace.state,
            "time_cells": self.laplace.time_cells(),
            "context_summary": self.laplace.context_summary(),
            "slow_profile": self.laplace.slow_profile(),
            "compositional_traces": len(self.indexer.compositional_traces),
            "theoretical_capacity": self.indexer.theoretical_capacity(),
        }

    # -- Slow Verifier (System 2) ------------------------------------------
    def verify(self, query_ec: torch.Tensor,
               concepts: Optional[List[str]] = None,
               theta_gate: bool = False) -> Dict[str, object]:
        """Эпизодическая верификация: resonance + распределённый реколл + HippoRAG.

        Оживает мёртвый путь v19: VSA-DHT multiprobe (38 400 эпизодов) и
        Personalized PageRank по графу концептов (многошаговые цепочки A→B→C).
        θ-гейт (`theta_gate=True`, живой путь из `tick`): реколл только на спаде θ и
        не чаще одного раза за цикл — после успешного реколла вызывается `mark_retrieved`.
        Четыре кода шардов берутся из РАЗНЫХ модальностей (`episodic.modality_codes`),
        а не из одного и того же вектора `ec` с разными смещениями сэмплирования.
        """
        ec = query_ec.detach().float()
        if ec.dim() == 1:
            ec = ec.unsqueeze(0)
        resonance = self._resonance(ec)
        dht = self.episodic.vsa_dht
        mismatch = float(self.episodic._last_mismatch) if hasattr(self.episodic, "_last_mismatch") else 0.25
        slot_codes = self.episodic.modality_codes(ec)
        gate_allowed = True
        if theta_gate:
            gate_allowed = self.theta.retrieval_allowed()
            if not gate_allowed:
                self.theta_gate_blocks += 1
        if gate_allowed:
            recall = dht.retrieve_multiprobe(slot_codes, ca1_mismatch=mismatch, top_k=5)
            if theta_gate and recall["top_episode_ids"]:
                self.theta.mark_retrieved()
                self.theta_recalls += 1
        else:
            recall = {"top_episode_ids": [], "top_scores": [],
                      "n_probes_used": 0, "inspected_records": 0}
        ppr = dht.ppr_associative_recall(
            seed_concepts=concepts,
            query_slot_codes=slot_codes if gate_allowed else None,
            top_k=5)
        predicted = None
        if self.episodic.n_stored > 0:
            dg_q = self.episodic.dg_code(ec)
            ca3_q = torch.relu(dg_q @ self.episodic.w_dg_ca3.t())
            predicted = self.episodic.ca3_predict_next(ca3_q.mean(0))
        return {
            "resonance": resonance,
            "recalled_episode_ids": recall["top_episode_ids"],
            "recall_scores": recall["top_scores"],
            "n_probes_used": recall["n_probes_used"],
            "concept_chain": ppr["concept_chain"],
            "multi_hop_chain_ids": ppr["multi_hop_chain_ids"],
            "predicted_next": predicted,
            "theta_gate": {"allowed": bool(gate_allowed),
                           "is_retrieval": bool(self.theta.is_retrieval_phase()),
                           "retrieved_in_cycle": bool(self.theta._retrieved_in_cycle),
                           "theta": round(float(self.theta.current_theta()), 4)},
        }

    # -- единый такт -------------------------------------------------------
    def tick(self, ec_in: torch.Tensor, ach: float, adenosine: float,
             learn: bool = True, valence: float = 0.0, salience_boost: float = 0.0,
             concepts: Optional[List[str]] = None,
             entropy: Optional[float] = None) -> DualHippoOut:
        """Один когнитивный такт: часы → рабочая память → эпизодический core →
        верификация → арбитр. Возвращает интерфейсы всех трёх систем."""
        self.tick_n += 1
        theta_st = self.theta.step()
        x = ec_in.detach().float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x_ec = torch.nn.functional.pad(x, (0, max(0, self.n_ec - x.shape[1])))[:, :self.n_ec]
        # Fast Drafter: рабочая память
        self.laplace.step(x_ec)
        # Slow Verifier: трисинаптический core (живой nw10)
        hout = self.episodic.step(x_ec, ach, adenosine, learn=learn,
                                  valence=valence, salience_boost=salience_boost)
        self.episodic._last_mismatch = hout.mismatch
        slots = self._bind_context(hout.ec, salience=0.35 + salience_boost, learn=learn)
        # Верификация (оживлённый VSA-DHT + HippoRAG). θ-гейт включён именно на
        # автоматическом пути: не более одного реколла за тета-цикл.
        ver = self.verify(hout.ec, concepts=concepts, theta_gate=True)
        # Arbiter
        decision = self.arbiter.decide(
            resonance=ver["resonance"], mismatch=hout.mismatch,
            novelty_da=hout.vta_da_burst, novelty_ne=hout.lc_ne_burst, entropy=entropy)
        elapsed = self.laplace.estimate_elapsed_time(0)
        return DualHippoOut(
            laplace_state=self.laplace.state,
            time_cells=self.laplace.time_cells(),
            context_summary=self.laplace.context_summary(),
            slow_profile=self.laplace.slow_profile(),
            elapsed_time=elapsed,
            theta_gamma_slots=slots,
            hippo=hout,
            resonance=ver["resonance"],
            recalled_episode_ids=ver["recalled_episode_ids"],
            recall_scores=ver["recall_scores"],
            concept_chain=ver["concept_chain"],
            predicted_next=ver["predicted_next"],
            arbiter=decision,
            theta=theta_st,
        )

    # -- эпизодическая запись / сон (делегирование живому core) ------------
    def store_episode(self, ec: torch.Tensor, dg: torch.Tensor, ca3: torch.Tensor,
                      error: float, uncertainty: float, da_tag: float = 0.0,
                      concepts: Optional[List[str]] = None) -> None:
        self.episodic.store_episode(ec, dg, ca3, error, uncertainty, da_tag=da_tag)
        if concepts:
            dht = self.episodic.vsa_dht
            ep_id = dht._last_stored_ep_id
            if ep_id is not None:
                dht.episode_to_concepts[ep_id] = list(concepts)
                for cname in concepts:
                    dht.concept_to_episodes.setdefault(cname, []).append(ep_id)
                for i in range(len(concepts) - 1):
                    dht.bind_concept_edge(concepts[i], concepts[i + 1], weight=0.90)

    def swr_replay(self, cortical_write, learn: bool = True,
                   direction: str = "auto") -> List[int]:
        return self.episodic.swr_replay(cortical_write, learn=learn, direction=direction)

    def rem_recombine(self, cortical_bridge_fn=None) -> Dict[str, float]:
        return self.episodic.rem_recombine(cortical_bridge_fn=cortical_bridge_fn)

    def sleep_cycle(self, cortical_write=None, *, adenosine: float = 0.0,
                    ach: float = 0.2, rem: bool = True,
                    cortical_bridge_fn=None,
                    swr_adenosine_threshold: float = 0.55) -> Dict[str, object]:
        """Точка врезки сна дуальной системы (делегирует живому episodic core)."""
        return self.episodic.sleep_cycle(
            cortical_write, adenosine=adenosine, ach=ach, rem=rem,
            cortical_bridge_fn=cortical_bridge_fn,
            swr_adenosine_threshold=swr_adenosine_threshold)

    def neurogenesis(self, novelty_level: float = 0.5, cortisol: float = 0.2,
                     bdnf: float = 0.35) -> int:
        return self.episodic.neurogenesis(novelty_level, cortisol, bdnf)

    def contextual_replay(self, cue_modality: str = "temporal_sequence",
                          seed_concepts: Optional[List[str]] = None,
                          query_ec: Optional[torch.Tensor] = None) -> Dict[str, object]:
        """Оживлённый маршрутизатор реплея VSA-DHT (F#15/F#19/F#21).

        `query_ec` — код запроса (EC): без него маршрутизация идёт только по сидам,
        а метрики извлечения честно возвращают `None` (измерять нечего).
        """
        query_codes = self.episodic.modality_codes(query_ec) if query_ec is not None else None
        return self.episodic.vsa_dht.route_contextual_replay(
            cue_modality=cue_modality, seed_concepts=seed_concepts,
            query_slot_codes=query_codes)

    def report(self) -> Dict[str, object]:
        dht = self.episodic.vsa_dht
        return {
            "tick_n": self.tick_n,
            "episodes": len(self.episodic.episodes),
            "ca3_stored": self.episodic.n_stored,
            "vsa_episodes": len(dht.episodes_vsa),
            "vsa_capacity": dht.total_capacity(),
            "concept_cells_active": int((self.episodic.concept_salience > 0.2).sum().item()),
            "semantic_bridges": int((self.episodic.w_semantic_graph > 0.05).sum().item()),
            "compositional_traces": len(self.indexer.compositional_traces),
            "theta_mode": self.theta.mode,
            "theta_cycle_steps": self.theta.cycle_steps,
            "theta_gate_blocks": int(self.theta_gate_blocks),
            "theta_recalls": int(self.theta_recalls),
        }