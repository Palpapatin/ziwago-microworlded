"""Clocked orchestration of the existing language cortex; model time is milliseconds."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

import torch
import torch.nn.functional as F

from ..action.contextual import ContextualActionGate
from ..thalamus.relay import ThalamusFO, ThalamicHO
from ..thalamus.trn import TRN


@dataclass
class IntegrationConfig:
    ticks_per_token: int = 4
    max_step_ticks: int = 4096
    max_sleep_ticks: int = 4096
    replay_capacity: int = 128
    replay_interval_ms: int = 40
    replay_transfer: float = 0.1
    meta_replay_priority_enabled: bool = True
    broadcast_gain: float = 0.15
    higher_order_gain: float = 0.1
    baseline_load: float = 0.05
    spike_reference_per_ms: float = 0.05
    current_reference: float = 100.0
    decision_budget_ms: int = 128
    criticality_window: int = 512
    # ── 13.11.2026: ЕДИНЫЙ КОНТУР ────────────────────────────────────────────
    # Раньше эти пять флагов были выключены, и «интегрированная архитектура»
    # оставалась набором переключателей: в живом пути не работал ни таламус,
    # ни TRN, ни AST/GW, ни интероцепция, ни лимбика, ни гиппокамп, ни
    # активная критичность. При этом причинные проводы у них ЕСТЬ и были
    # измерены: лимбика и гиппокамп пишут в `da_phasic` (третий фактор
    # пластичности), объёмная химия — в `organism_context` → `col._mod_ctx`
    # (nmda_gain/vth_shift/gaba_tonic/ahp_block/rate_ceiling/bdnf), а
    # критичность — в силу таламического релея. То есть модули были написаны
    # и подключены, но выключены; теперь контур один и биология включена.
    criticality_control_enabled: bool = True
    criticality_target: float = 0.97
    criticality_gain_rate: float = 0.5
    criticality_gain_min: float = 0.5
    criticality_gain_max: float = 1.5
    volume_chemistry_enabled: bool = True
    interoception_enabled: bool = True
    limbic_enabled: bool = True
    intero_allostatic_threshold: float = 0.35
    intero_sleep_gain: float = 0.008
    intero_met_gain_scale: float = 0.5
    limbic_da_valence_gain: float = 0.5
    dual_hippocampus_enabled: bool = True
    hippo_n_ec: int = 64
    hippo_theta_mode: str = "token"
    hippo_theta_cycle_steps: float = 8.0
    hippo_novelty_da_gain: float = 0.5

    def __post_init__(self):
        if not isinstance(self.meta_replay_priority_enabled, bool):
            raise ValueError("meta_replay_priority_enabled must be a boolean")
        for name in ("ticks_per_token", "max_step_ticks", "max_sleep_ticks", "replay_capacity",
                     "replay_interval_ms", "decision_budget_ms", "criticality_window"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("broadcast_gain", "higher_order_gain", "baseline_load", "replay_transfer"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.ticks_per_token > self.max_step_ticks or self.criticality_window < 32:
            raise ValueError("Token ticks must fit the step budget; the measurement window must be at least 32")
        for name in ("spike_reference_per_ms", "current_reference"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 < self.criticality_target <= 1.5:
            raise ValueError("criticality_target must lie in (0, 1.5]")
        if not math.isfinite(self.criticality_gain_rate) or self.criticality_gain_rate < 0:
            raise ValueError("criticality_gain_rate must be finite and non-negative")
        if not 0 < self.criticality_gain_min <= 1 <= self.criticality_gain_max:
            raise ValueError("criticality gain bounds must satisfy 0 < min <= 1 <= max")
        if not 0 <= self.intero_allostatic_threshold <= 1:
            raise ValueError("intero_allostatic_threshold must lie in [0, 1]")
        for name in ("intero_sleep_gain", "intero_met_gain_scale", "limbic_da_valence_gain"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(self.hippo_n_ec, int) or isinstance(self.hippo_n_ec, bool) or self.hippo_n_ec < 1:
            raise ValueError("hippo_n_ec must be a positive integer")
        if self.hippo_theta_mode not in ("baseline", "fast", "token"):
            raise ValueError("hippo_theta_mode must be baseline, fast or token")
        if not math.isfinite(self.hippo_theta_cycle_steps) or self.hippo_theta_cycle_steps <= 0:
            raise ValueError("hippo_theta_cycle_steps must be positive and finite")
        if not math.isfinite(self.hippo_novelty_da_gain) or self.hippo_novelty_da_gain < 0:
            raise ValueError("hippo_novelty_da_gain must be finite and non-negative")


def resample(vector: torch.Tensor, width: int) -> torch.Tensor:
    return F.interpolate(vector.reshape(1, 1, -1), size=width,
                         mode="linear", align_corners=False).reshape(-1)


def resample_sector_gain(vector: torch.Tensor, width: int) -> torch.Tensor:
    """Раскладка секторного профиля по БЛОКАМ каналов — без размытия.

    Линейная интерполяция 5 секторов в 128 каналов превращала профиль
    «победитель/проигравший = 6.9» в плато (max/4-й = 1.035): конкуренция
    приоритетов внимания терялась ИМЕННО на живом пути, хотя сам гейн AST
    был нормирован правильно. Сектор — это модуль коры, значит он занимает
    блок каналов, а внутри блока приоритет постоянен: контраст профиля
    сохраняется, а не растворяется.
    """
    n = int(vector.numel())
    if n <= 0:
        return torch.zeros(width)
    idx = (torch.arange(width, dtype=torch.long) * n) // max(1, width)
    return vector.detach().float().reshape(-1).index_select(0, idx.clamp_(0, n - 1))


class IntegratedOrganism:
    def __init__(self, n_zones: int, n_input: int, n_neurons: int,
                 seed: int = 20, cfg: IntegrationConfig | None = None):
        self.cfg = cfg or IntegrationConfig()
        self.n_zones, self.n_input, self.n_neurons = n_zones, n_input, n_neurons
        self.fo = ThalamusFO(n_input)
        self.trn = TRN(n_input)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 200)
            self.ho = ThalamicHO(n_neurons - 2 * n_neurons // 3, n_input)
        self.actions = ContextualActionGate(n_actions=7, context_dim=n_input, seed=seed + 201)
        self.previous_load = torch.full((n_zones,), self.cfg.baseline_load)
        self.previous_ca = 0.0
        self.previous_error = 0.0
        self.previous_attention = 0.0
        self.l6_feedback = torch.zeros(n_input)
        self.l5_feedback = torch.zeros(self.ho.n_in)
        self.broadcast = torch.zeros(n_input)
        self.ast_priority = torch.full((n_input,), 0.5)
        self.confidence = 0.5
        self.replay = []
        self._meta_replay_priority_pending = 0.0
        self._meta_replay_priority_for_credit = 0.0
        self._last_meta_replay_priority_output = 0.0
        self._last_meta_replay_priority_applied = 0.0
        self.meta_replay_priority_credit_count = 0
        self.replay_evictions = 0
        self.replay_events = 0
        self.replay_synapses = 0
        self.last_replay_ms = 0.0
        self.learning_enabled = True
        self.training_active = False
        self.reward_baseline = 0.0
        self.reward_events = 0
        self.action_context = torch.zeros(n_input)
        self.action_evidence = torch.zeros(7)
        self.action_result = {"action": -1, "committed": False, "decision_id": None, "bg_release": 0.0}
        self.last_relay_norm = 0.0
        self.last_broadcast_norm = 0.0
        self.last_tick = {}
        self.criticality = {"available": False, "estimate": None, "reason": "insufficient_window"}
        self.criticality_gain = 1.0
        # ЕДИНЫЙ ВЛАДЕЛЕЦ σ̂ (13.11): `core/criticality.py`. Инлайн-регрессия,
        # которая жила здесь раньше, клампила σ̂ к «~1» — «σ̂≈1» было
        # свойством формулы, а не измерением.
        from ..core.criticality import CriticalityMonitor
        from ..config import HomeostasisCfg
        from ..homeostasis.controller import HomeostaticController
        self._criticality_monitor = CriticalityMonitor(
            target_sigma=self.cfg.criticality_target)
        self._last_spike_total = 0.0
        self._prev_spike_total = 0.0
        # Организменный гомеостаз: измерение (монитор) отдельно от управления
        # (контроллер). Контроллер больше не затирает измерение.
        self.homeostatic = HomeostaticController(
            HomeostasisCfg(target_sigma=self.cfg.criticality_target),
            monitor=self._criticality_monitor)
        self.homeostatic_status = {
            "connected": False,
            "reason": "measurement_unavailable",
            "report": {},
        }
        self.phase_events = []
        self.xchem = None
        self.volume = None
        if self.cfg.volume_chemistry_enabled:
            from ..chemistry.agents import ExtendedChemistry
            from ..chemistry.volume import VolumeTransmission
            self.xchem = ExtendedChemistry()
            self.volume = VolumeTransmission(n_zones * n_neurons)
        self.intero = None
        self.limbic = None
        self._limbic_fe_ema = None
        self._last_intero = None
        self._last_limbic = None
        if self.cfg.interoception_enabled:
            from ..core.interoception import InteroceptiveField
            self.intero = InteroceptiveField([f"z{i}" for i in range(n_zones)],
                                             n_int=16, seed=seed + 202)
        if self.cfg.limbic_enabled:
            from ..emotion.limbic import LimbicSystem
            self.limbic = LimbicSystem(n_input, self.ho.n_in, n_bla=16, seed=seed + 203)
        self.hippo = None
        self._last_hippo = None
        if self.cfg.dual_hippocampus_enabled:
            from ..config import HippocampusCfg
            from ..memory.dual_hippocampus import DualHippocampus
            self.hippo = DualHippocampus(
                HippocampusCfg(n_ec=self.cfg.hippo_n_ec), input_dim=n_input,
                seed=seed + 204, theta_mode=self.cfg.hippo_theta_mode,
                theta_cycle_steps=self.cfg.hippo_theta_cycle_steps)

    def present(self, core, x: torch.Tensor):
        spike_sum = l5_sum = gate_sum = None
        counts = []
        experts = {}
        for _ in range(self.cfg.ticks_per_token):
            result = self.tick(core, x)
            if spike_sum is None:
                spike_sum = torch.zeros_like(result.combined_spikes)
                l5_sum = torch.zeros_like(result.combined_l5_drive)
                gate_sum = torch.zeros_like(result.gating_weights)
            spike_sum.add_(result.combined_spikes)
            l5_sum.add_(result.combined_l5_drive)
            for cid, name in zip(result.active_expert_ids, result.active_expert_names):
                experts[cid] = name
                gate_sum[cid] += result.gating_weights[cid]
            counts.append(result.spike_count_window)
        duration = self.cfg.ticks_per_token
        ids = sorted(experts)
        # A silent final millisecond must not erase earlier evidence or its contributing experts.
        return replace(result, combined_spikes=spike_sum / duration,
                       combined_l5_drive=l5_sum / duration,
                       gating_weights=gate_sum / duration,
                       active_expert_ids=ids,
                       active_expert_names=[experts[i] for i in ids],
                       spike_count_window=torch.cat(counts, dim=0))

    @torch.no_grad()
    def tick(self, core, x: torch.Tensor):
        x = x.detach().float().flatten()
        if x.numel() != self.n_input or not bool(torch.isfinite(x).all()):
            raise ValueError("Sensory input must be a finite vector of the configured width")
        self._begin_meta_replay_priority_tick()
        chem = core.chemistry
        old_phase = chem.phase
        bundle = chem.advance(self.previous_load, ca_mean=self.previous_ca,
                              novelty=self.previous_error, attention=self.previous_attention,
                              stress=self.previous_error, drive=min(1.0, float(x.norm())))
        if chem.phase != old_phase:
            self.phase_events.append({"time_ms": chem.time_ms, "from": old_phase, "to": chem.phase})
            self.phase_events = self.phase_events[-32:]
            callback = getattr(core, "external_sleep_phase_fn", None)
            if callback is not None and core.learning_enabled and self.learning_enabled:
                callback(old_phase, chem.phase)
        plastic = bool(self.learning_enabled and core.learning_enabled and self.training_active
                       and bundle["f_mod"] > 0)
        replay_item = None
        if chem.phase != "awake":
            x = torch.zeros_like(x)
            if (chem.phase == "nrem" and self.replay and self.learning_enabled and core.learning_enabled
                    and chem.time_ms - self.last_replay_ms >= self.cfg.replay_interval_ms):
                index = max(range(len(self.replay)), key=lambda i: self.replay[i]["priority"])
                replay_item = self.replay.pop(index)
                x = replay_item["cue"].clone()
                self.last_replay_ms = chem.time_ms

        action_modulators = {key: bundle[key] for key in ("da", "ach", "ado", "ox", "ht", "awake")}
        self.action_result = self.actions.step(
            self.action_context, self.action_evidence, modulators=action_modulators,
            uncertainty=1.0 - self.confidence, dt_ms=1.0,
            learn=bool(self.learning_enabled and core.learning_enabled and bundle["f_mod"] > 0))
        gate = self.trn.step((x.abs() + self.l6_feedback.abs()).clamp(0, 1),
                             self.ast_priority, vigilance=bundle["ach"] + bundle["ne"])
        self.fo.gate = gate
        relay = self.fo.step(x.unsqueeze(0), ach=bundle["ach"], ne=bundle["ne"])[0]
        higher = self.ho.step(self.l5_feedback.unsqueeze(0),
                              bg_release=float(self.action_result["bg_release"]))[0]
        relay = relay + self.cfg.higher_order_gain * higher
        if self.cfg.criticality_control_enabled:
            relay = relay * self.criticality_gain
        self._update_limbic(bundle, relay, plastic)
        self._update_hippocampus(bundle, relay, plastic)
        broadcast = resample(self.broadcast, self.n_neurons) * self.cfg.broadcast_gain
        self.last_relay_norm = float(relay.norm())
        self.last_broadcast_norm = float(broadcast.norm())
        ctx = {**bundle, "zone_atp": chem.zone_atp, "zone_ado": chem.zone_ado,
               "is_sleep_tick": chem.phase != "awake", "learning_enabled": plastic}
        ctx.update(self._update_extended_chemistry(bundle, x))
        routing_bias = resample(self.ast_priority, self.n_zones) - 0.5
        # `learn=plastic`, а не `learn=False`: иначе вычисленный выше
        # `ctx["learning_enabled"]` остаётся мёртвым, и правило обучения
        # Bio-RWKV (w_v/w_k) вместе с блочно-разрежённой STDP не включаются
        # даже при открытом гейте пластичности.
        out = core.moe_cortex.step(relay, learn=plastic, organism_context=ctx,
                                   broadcast=broadcast, routing_bias=routing_bias)
        active = set(out.active_expert_ids)
        load, counts, ca = [], [], []
        l6 = []
        for cid, col in enumerate(core.moe_cortex.columns):
            if cid in active:
                spikes = float(col._last_s.float().mean())
                current = sum(float(getattr(col.pyr, name).abs().mean())
                              for name in ("last_i_ff", "last_i_fb", "last_i_inh"))
                # The imposed somatic drive and noise also have an energetic cost.
                current += 1.5 + float(col.pyr.noise.abs().mean())
                load.append(self.cfg.baseline_load + spikes / self.cfg.spike_reference_per_ms
                            + current / self.cfg.current_reference)
                counts.append(float(col._last_s.sum()))
                ca.append(float(col.pyr.ca.mean()))
                l6.append(resample(col._last_s[0, col.layer_slices["L6"]], self.n_input))
            else:
                load.append(self.cfg.baseline_load)
                counts.append(0.0)
        self.previous_load = torch.tensor(load)
        self.previous_ca = sum(ca) / max(1, len(ca))
        self.previous_error = min(1.0, max(0.0, out.mean_apical_fep_error + out.mean_somatic_fep_error))
        self.previous_attention = float(out.gating_weights.max())
        self.l6_feedback = 0.95 * self.l6_feedback + 0.05 * torch.stack(l6).mean(0)
        self.l5_feedback = out.combined_l5_drive.detach().mean(0).clone()
        self._distribute_volume_fields(core, out)
        self._update_interoception(core, out, plastic)
        son = core.son_column
        gw = son.gw.compete([resample(self.l5_feedback, self.n_input), x], ne=bundle["ne"])
        self.broadcast = (gw.broadcast.detach().clone() if gw.ignition > 0.5
                          else torch.zeros_like(self.broadcast))
        meta = son.meta_head.step(self.broadcast, gw.content, self.previous_error, learn=plastic)
        self._stage_meta_replay_priority(meta)
        son.meta_head.err_hist = son.meta_head.err_hist[-self.cfg.criticality_window:]
        son.meta_head.conf_hist = son.meta_head.conf_hist[-self.cfg.criticality_window:]
        self.confidence = meta.confidence
        ast = son.ast.step(self.previous_error, self.previous_error, float(bundle["ox"]),
                           self.previous_attention, int(out.active_expert_ids[0]) % son.ast.cfg.n_modules,
                           learn=plastic)
        self.ast_priority = resample_sector_gain(ast.gain, self.n_input).clamp(0, 1)
        son.organism_reflection = {"gw": gw, "meta": meta}
        if replay_item is not None:
            hits = 0
            for item in replay_item["columns"]:
                syn = core.moe_cortex.columns[item["id"]].w_ff
                hits += syn.consolidate_tagged(item["pre"], factor=self.cfg.replay_transfer)
            self.replay_events += 1
            self.replay_synapses += hits
        self._prev_spike_total = self._last_spike_total
        self._last_spike_total = float(sum(counts))
        self._measure_criticality()
        if int(chem.tick) % 32 == 0:
            if not self.cfg.criticality_control_enabled:
                self.homeostatic_status = {
                    "connected": False, "reason": "disabled", "report": {},
                }
            elif self.criticality["available"]:
                report = self.homeostatic.regulate(core.moe_cortex)
                self.homeostatic_status = {
                    "connected": True, "reason": "measured", "report": dict(report),
                }
            else:
                self.homeostatic_status = {
                    "connected": False,
                    "reason": self.criticality["reason"],
                    "report": {},
                }
            self._update_criticality_gain()
        self.last_tick = {"time_ms": chem.time_ms, "tick": chem.tick, "phase": chem.phase,
                          "load": load, "spikes": sum(counts), "plasticity_gate": bundle["f_mod"],
                          "relay_norm": self.last_relay_norm, "broadcast_norm": self.last_broadcast_norm,
                          "gw_ignition": gw.ignition, "confidence": self.confidence,
                          # Энергия GW честно считается, но наружу НЕ отдавалась —
                          # ещё один «призрак»: значение было, а потребителя не было.
                          "gw_energy": float(gw.energy),
                          "meta_replay_priority": self._last_meta_replay_priority_output,
                          "meta_replay_priority_for_credit": self._meta_replay_priority_for_credit,
                          "replay": replay_item is not None, "action": dict(self.action_result)}
        out.spike_count_window = torch.tensor([counts])
        return out

    def _measure_criticality(self):
        """σ̂ — ИЗМЕРЕНИЕ, единственный источник: `core/criticality.py`.

        До 13.11 здесь была собственная инлайн-регрессия по автокорреляции —
        дубль измерения (и он клампил σ̂ к «~1», то есть «σ̂≈1» было
        свойством формулы, а не измерением). Теперь единственный владелец
        σ̂ — `CriticalityMonitor`; регулятор (`HomeostaticController`)
        сравнивает измеренное значение с ЯВНОЙ ЦЕЛЬЮ.
        """
        from ..core.criticality import CriticalityMonitor
        if self._criticality_monitor is None:
            self._criticality_monitor = CriticalityMonitor(
                target_sigma=self.cfg.criticality_target)
        self._criticality_monitor.push(torch.tensor([float(self._last_spike_total)]),
                                       torch.tensor([float(self._prev_spike_total)]))
        est = self._criticality_monitor.sigma_hat
        ready = bool(self._criticality_monitor.sigma_hat_ready)
        self.criticality = {
            "available": ready,
            "estimate": float(est) if ready else None,
            "reason": "measured" if ready else "insufficient_window",
            "sigma_target": float(self.cfg.criticality_target),
            "sigma_error": (float(est - self.cfg.criticality_target)
                            if ready else None),
        }
        return self.criticality

    def _update_criticality_gain(self):
        # Организменный гомеостаз таламического гейна от измеренного σ̂.
        # Комплементарен колоночному HomeostaticController: тот правит пороги/веса
        # внутри колонки, этот — силу таламического релея на входе коры.
        # σ̂ < target → кора недокритична → гейн растёт (усиливаем реле);
        # σ̂ > target → перекритична → гейн падает. Default off → гейн ≡ 1.
        if not self.cfg.criticality_control_enabled:
            return
        est = self.criticality.get("estimate")
        if not self.criticality.get("available") or est is None:
            return
        c = self.cfg
        g_target = math.exp(c.criticality_gain_rate * (c.criticality_target - float(est)))
        g_target = max(c.criticality_gain_min, min(c.criticality_gain_max, g_target))
        self.criticality_gain = max(c.criticality_gain_min, min(c.criticality_gain_max,
                                    0.9 * self.criticality_gain + 0.1 * g_target))
        self.criticality["controller_enabled"] = True
        self.criticality["gain"] = self.criticality_gain

    def _update_extended_chemistry(self, bundle, x):
        # Opt-in (М-4): пул 37 агентов от измеренных сигналов прошлого такта.
        # Эффекторы попадают в organism_context → col._mod_ctx, где их уже читает
        # живая колонка (column.py: nmda_gain/histamine/vth_shift/gaba_tonic/
        # rate_ceiling/ahp_block/bdnf). Default off → пустой dict, ctx не меняется.
        if self.xchem is None:
            return {}
        activity = float(self.previous_ca)
        stress = self.previous_error
        arousal = min(1.0, 0.5 * float(bundle["ne"]) + 0.5 * float(bundle["ach"]))
        drive = min(1.0, float(x.norm()))
        damage = max(0.0, self.previous_error - 0.30)
        self.xchem.update(spikes=activity, stress=stress, arousal=arousal,
                          social=drive, threat=stress, damage=damage)
        return {"histamine": self.xchem.l.histamine,
                "nmda_gain": self.xchem.nmda_gain(),
                "vth_shift": self.xchem.vth_shift(),
                "gaba_tonic": self.xchem.gaba_tonic(),
                "ahp_block": self.xchem.ahp_block(),
                "rate_ceiling": self.xchem.rate_ceiling(),
                "bdnf": self.xchem.l.bdnf,
                "p_release": float(bundle["p_release"]) * self.xchem.p_release_scale()}

    def _distribute_volume_fields(self, core, out):
        # Opt-in (М-4): объёмная передача NO/2-AG/H2S. Источники — ТОЛЬКО активные
        # колонки (неактивные не шагают в moe_cortex.step, их Ca/spikes устарели).
        # Поле гетеро-LTP раздаётся колонкам в _no_ltp_field, который читает
        # column.py при learn. Default off → поля не создаются, атрибут не ставится.
        if self.volume is None:
            return
        n_pc = self.n_neurons
        ca_all = torch.zeros(1, self.n_zones * n_pc)
        sp_all = torch.zeros(1, self.n_zones * n_pc)
        for cid in out.active_expert_ids:
            col = core.moe_cortex.columns[cid]
            off = cid * n_pc
            ca_all[:, off:off + n_pc] = col.pyr.ca.detach()
            sp_all[:, off:off + n_pc] = col._last_s.detach()
        self.volume.step(ca_all, sp_all)
        for cid, col in enumerate(core.moe_cortex.columns):
            zf = self.volume.zone_fields(cid * n_pc, n_pc)
            col._no_ltp_field = zf["no_ltp"]
            col._vol_eject = zf["eject"]

    def _volume_telemetry(self):
        if self.xchem is None:
            return {"enabled": False}
        return {"enabled": True, "nmda_gain": self.xchem.nmda_gain(),
                "p_release_scale": self.xchem.p_release_scale(),
                "agents": asdict(self.xchem.l),
                "no_mean": float(self.volume.no.mean()),
                "h2s_mean": float(self.volume.h2s.mean()),
                "ecb_mean": float(self.volume.ecb_field.mean())}

    def _update_limbic(self, bundle, relay, plastic):
        # Opt-in (М-8): BLA/CeA валентность → фазический DA (3-й фактор пластичности).
        # Измерено: da_phasic причинен и знаковый в v20 (synapse.py:502, external clock):
        # da 0→1.06e-5, +1→1.92e-5, -0.5→6.3e-6 dW. Arousal→NE НЕ оживлён: в свежем
        # v20-режиме ne инвариантен на реле (тоническое насыщение) и нейроне (stochastic
        # off) — мёртвый провод по AGENTS.md §7, не добавляется.
        if self.limbic is None:
            return
        fe = float(self.previous_error)
        if self._limbic_fe_ema is None:
            self._limbic_fe_ema = fe
            intrinsic_rpe = 0.0
        else:
            prev = self._limbic_fe_ema
            intrinsic_rpe = max(-0.5, min(0.8, (prev - fe) * 5.0))
            self._limbic_fe_ema = 0.70 * prev + 0.30 * fe
        out = self.limbic.step(relay, self.l5_feedback, free_energy=fe,
                               intrinsic_rpe=intrinsic_rpe,
                               atp_mean=float(bundle["lactate"]), ext_reward=0.0,
                               learn=plastic)
        self._last_limbic = out
        bundle["da_phasic"] = (float(bundle["da_phasic"])
                               + self.cfg.limbic_da_valence_gain * out.valence)

    def _update_hippocampus(self, bundle, relay, plastic):
        # Opt-in (§8.1): merged dual-system гиппокамп. EC-вход — текущий
        # таламо-корковый relay (восприятие); CA1-компаратор сравнивает его с
        # реконструкцией из памяти → новизна. Причинный актуатор: novelty DA
        # (VTA burst) → da_phasic (3-й фактор пластичности, synapse.py:502,
        # измерен причинным в М-8). NE-burst НЕ проводится в ne: в свежем v20
        # ne инвариантен (тоническое насыщение реле + stochastic off нейрона) —
        # мёртвый провод по AGENTS.md §7, тот же отрицательный результат, что в
        # лимбике (М-8). Default off → hippo=None, ctx не меняется, бит-идентично.
        if self.hippo is None:
            return
        out = self.hippo.tick(
            relay, ach=float(bundle["ach"]), adenosine=float(bundle["ado"]),
            learn=plastic, valence=0.0, salience_boost=0.0, concepts=None, entropy=None)
        self._last_hippo = out
        bundle["da_phasic"] = (float(bundle["da_phasic"])
                               + self.cfg.hippo_novelty_da_gain * out.hippo.vta_da_burst)

    def _update_interoception(self, core, out, plastic):
        # Opt-in (М-8): виртуальная интероцепция Insula–ACC (без физической арены и
        # слогового декодирования). Читает висцеральное состояние колонк (v_s/energy/
        # ca/rate) и регулирует два причинных актуатора (измерено на живой коре):
        #   allostatic_drive → met_gain (метаболический тормоз, neuron.py:180, v_s);
        #   allostatic_drive → process_s (давление сна, physiology.py:179, f_mod).
        # met_gain — recoverable EMA к 1+scale*drive: в v20 нет колоночного
        # HomeostaticController (controller.py:106 decay), поэтому чистый ratchet
        # GenesisOrganism копился бы без границ; EMA возвращается к 1 при drive→0.
        if self.intero is None:
            return
        active = set(out.active_expert_ids)
        zone_states = {}
        for cid, col in enumerate(core.moe_cortex.columns):
            p = col.pyr
            rate = float(col._last_s.float().mean()) if cid in active else 0.0
            zone_states[f"z{cid}"] = torch.tensor([
                float(p.v_s.mean()) / 100.0 + 0.5, float(p.energy.mean()),
                float(p.ca.mean()), rate])
        io = self.intero.read(zone_states, pfc_spikes=self.l5_feedback, learn=plastic)
        self._last_intero = io
        target = 1.0 + self.cfg.intero_met_gain_scale * io.allostatic_drive
        for col in core.moe_cortex.columns:
            mg = float(getattr(col.pyr, "met_gain", 1.0))
            col.pyr.met_gain = min(2.0, 0.95 * mg + 0.05 * target)
        if io.allostatic_drive > self.cfg.intero_allostatic_threshold:
            chem = core.chemistry
            chem.process_s = min(1.0, chem.process_s
                                 + self.cfg.intero_sleep_gain * io.allostatic_drive)

    def _intero_telemetry(self):
        if self.intero is None:
            return {"enabled": False}
        io = self._last_intero
        if io is None:
            return {"enabled": True, "allostatic_drive": 0.0, "intero_error": 0.0,
                    "self_state": {}, "state_mean": 0.0}
        return {"enabled": True, "allostatic_drive": io.allostatic_drive,
                "intero_error": io.intero_error, "self_state": io.summary,
                "state_mean": float(io.state_vec.mean())}

    def _limbic_telemetry(self):
        if self.limbic is None:
            return {"enabled": False}
        out = self._last_limbic
        if out is None:
            return {"enabled": True}
        return {"enabled": True, "valence": out.valence, "arousal": out.arousal,
                "d_energy": out.d_energy, "d_safety": out.d_safety,
                "d_novelty": out.d_novelty, "appetitive": out.appetitive,
                "aversive": out.aversive}

    def _hippo_telemetry(self):
        if self.hippo is None:
            return {"enabled": False}
        rep = self.hippo.report()
        out = self._last_hippo
        base = {"enabled": True, "tick_n": rep["tick_n"], "n_ec": self.cfg.hippo_n_ec,
                "novelty_da_gain": self.cfg.hippo_novelty_da_gain,
                "theta_mode": rep["theta_mode"], "theta_cycle_steps": rep["theta_cycle_steps"],
                "episodes": rep["episodes"], "vsa_episodes": rep["vsa_episodes"],
                "vsa_capacity": rep["vsa_capacity"],
                "compositional_traces": rep["compositional_traces"]}
        if out is None:
            return {**base, "resonance": 0.0, "decision": None}
        return {**base, "resonance": out.resonance, "mismatch": out.hippo.mismatch,
                "decision": out.arbiter.decision, "reason": out.arbiter.reason,
                "novelty_da": out.hippo.vta_da_burst, "novelty_ne": out.hippo.lc_ne_burst,
                "recalled_episode_ids": out.recalled_episode_ids,
                "concept_chain": out.concept_chain, "elapsed_time": out.elapsed_time}

    def record_credit(self, core, cue, cortical_out, loss):
        if not self.learning_enabled or not core.learning_enabled or not core.chemistry.is_awake:
            return
        if len(self.replay) >= self.cfg.replay_capacity:
            self.replay.pop(0)
            self.replay_evictions += 1
        base_priority = float(loss) + 1.0 - self.confidence
        meta_priority = (self._meta_replay_priority_for_credit
                         if self.cfg.meta_replay_priority_enabled else 0.0)
        self.replay.append({"cue": cue.detach().clone(),
                            "priority": base_priority + meta_priority,
                            "base_priority": base_priority,
                            "meta_replay_priority": meta_priority,
                            "columns": [{"id": cid, "pre": core.moe_cortex.columns[cid]._last_x_ff.detach().clone().flatten()}
                                        for cid in cortical_out.active_expert_ids]})
        self._last_meta_replay_priority_applied = meta_priority
        if self.cfg.meta_replay_priority_enabled:
            self.meta_replay_priority_credit_count += 1
        if self.hippo is not None and self._last_hippo is not None:
            h = self._last_hippo.hippo
            self.hippo.store_episode(h.ec, h.dg, h.ca3, error=float(loss),
                                     uncertainty=1.0 - self.confidence,
                                     da_tag=float(self._last_hippo.arbiter.novelty_da))

    def training_reward(self, core, quality: float):
        rpe = float(quality) - self.reward_baseline
        self.reward_baseline += 0.05 * rpe
        core.chemistry.reward(rpe)
        self.reward_events += 1
        return rpe

    def settle(self, core, decision_id: int, reward: float, learn=True):
        result = self.actions.settle(decision_id, reward,
                                     learn=bool(learn and self.learning_enabled and core.learning_enabled))
        if result["consumed"]:
            core.chemistry.reward(result["rpe"])
            self.action_evidence.zero_()
        return result

    def telemetry(self, core):
        reflection = getattr(core.son_column, "organism_reflection", {})
        recent_meta = reflection.get("meta") if isinstance(reflection, dict) else None
        return {"config": asdict(self.cfg), "physiology": core.chemistry.telemetry(),
                "last_tick": dict(self.last_tick), "action": self.actions.telemetry(),
                "replay_pending": len(self.replay), "replay_events": self.replay_events,
                "replay_synapses": self.replay_synapses, "replay_evictions": self.replay_evictions,
                "meta_replay_priority": {
                    "enabled": self.cfg.meta_replay_priority_enabled,
                    "consumer": "IntegratedOrganism.record_credit",
                    "delay_ticks": 1,
                    "pending_for_next_tick": self._meta_replay_priority_pending,
                    "applied_to_credit": self._last_meta_replay_priority_applied,
                    "credit_count": self.meta_replay_priority_credit_count,
                },
                "meta_head": core.son_column.meta_head.telemetry(
                    recent=recent_meta,
                    replay_priority_connected=self.cfg.meta_replay_priority_enabled),
                "global_workspace": dict(core.son_column.gw.telemetry),
                "reward_events": self.reward_events, "criticality": dict(self.criticality),
                "homeostasis": {"enabled": self.cfg.criticality_control_enabled,
                                **self.homeostatic_status},
                "criticality_gain": self.criticality_gain,
                "volume_chemistry": self._volume_telemetry(),
                "interoception": self._intero_telemetry(),
                "limbic": self._limbic_telemetry(),
                "dual_hippocampus": self._hippo_telemetry(),
                "phase_events": list(self.phase_events),
                "energy_accounting": "normalized cortical currents, spikes and somatic drive; not biochemical ATP units"}

    def _begin_meta_replay_priority_tick(self):
        """Latch the previous tick's MetaHead output for the next credit."""
        self._meta_replay_priority_for_credit = (
            self._meta_replay_priority_pending
            if self.cfg.meta_replay_priority_enabled else 0.0
        )

    def _stage_meta_replay_priority(self, meta):
        """Stage only replay_priority; other MetaHead gates remain unconsumed."""
        if not self.cfg.meta_replay_priority_enabled:
            self._meta_replay_priority_pending = 0.0
            self._last_meta_replay_priority_output = 0.0
            return
        priority = float(meta.gates["replay_priority"])
        if not math.isfinite(priority) or priority < 0.0:
            raise ValueError("MetaHead replay_priority must be finite and non-negative")
        self._meta_replay_priority_pending = priority
        self._last_meta_replay_priority_output = priority
