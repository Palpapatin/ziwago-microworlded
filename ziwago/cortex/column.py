"""Кортикальная колонка V10 (§II.2.1-2.2).

Микроцепь:
  PYR (US-нейроны): базальный дендрит ← AMPA-FF от предыдущей зоны;
                    апикальный ← NMDA-FB от следующей зоны + GW;
  PV — перисоматическое торможение, k-WTA на сому (быстрая реакция);
  SST — дендритное торможение на апикальный вход (гейтинг FB);
  VIP — тормозит SST → дизингибиция апикала. g = g_0/(1+([VIP]/K_d)^n).

Обучение: трёхфакторное правило с внутренней ошибкой зоны
error_i = S_i - φ(V*_D,i) (Rao & Ballard 1999 внутри дендрита).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..config import USNeuronCfg, USSynapseCfg, ZoneCfg
from ..core.multirate import MinicolumnStateHierarchy, TickLadderScheduler
from ..core.neuron import USNeuronPool, NeuronOut
from ..core.synapse import GenesisSynapse
from .lfp import EphapticLFP


def barbas_layer_slices(n_exc: int, gradient: float,
                        l5_idx: torch.Tensor) -> dict[str, slice]:
    """Структурная модель Барбас (Barbas 1986; Genesis level2/column.py):
    Гранулярная сенсорная кора (g=0.0) имеет выраженный слой L4,
    агранулярная префронтальная кора (g=0.9) — толстые слои L2/3 и L5/6."""
    n_sup = int(l5_idx[0].item()) if l5_idx.numel() > 0 else (2 * n_exc // 3)
    l4_frac = 0.45 - 0.35 * max(0.0, min(1.0, float(gradient)))
    n_l4 = max(1, int(round(n_sup * l4_frac)))
    n_l23 = max(1, n_sup - n_l4)
    n_l4 = max(0, n_sup - n_l23)
    n_deep = n_exc - n_sup
    n_l5 = max(1, int(round(n_deep * 0.65)))
    return {
        "L4": slice(0, n_l4),
        "L23": slice(n_l4, n_sup),
        "L5": slice(n_sup, n_sup + n_l5),
        "L6": slice(n_sup + n_l5, n_exc),
    }


@dataclass
class ColumnOut:
    s: torch.Tensor           # спайки пирамидальных
    error: torch.Tensor       # внутренняя ошибка зоны
    l5_drive: torch.Tensor    # выход L5-подмножества → HO/BG
    v_s: torch.Tensor
    ca: torch.Tensor          # Ca2+ → 2-AG/NO
    rate_d: torch.Tensor
    local_sigma: float = 0.97 # D3: локальный коэффициент ветвления колонки O(1) без глобального readout
    local_sigma_gain: float = 1.0


class CorticalColumn:
    """Зона на градиенте: PYR + PV + SST + VIP + LFP-Эфаптика + FF/FB/REC-синапсы."""

    def __init__(self, zcfg: ZoneCfg, n_in: int, n_fb: int,
                 ncfg: USNeuronCfg, scfg: USSynapseCfg, seed: int = 0):
        self.zcfg = zcfg
        self.name = zcfg.name
        n = zcfg.n_neurons
        self.pyr = USNeuronPool(n, ncfg, name=zcfg.name)
        self.pv = torch.zeros(zcfg.n_pv)
        self.sst = torch.zeros(zcfg.n_sst)
        self.vip = torch.zeros(zcfg.n_vip)
        # Мембранные потенциалы спайковых популяций интернейронов (LIF)
        self.v_pv = torch.zeros(zcfg.n_pv)
        self.v_sst = torch.zeros(zcfg.n_sst)
        self.v_vip = torch.zeros(zcfg.n_vip)
        self.vip_drive = 0.0                      # [VIP] для дизингибиции
        # Синапсы: FF (пред→basal), FB (след→apical), рекуррентные, к интернейронам
        self.w_ff = GenesisSynapse(n_in, n, scfg, f"{zcfg.name}_ff", seed,
                                   init_scale=1.2)
        self.w_fb = GenesisSynapse(n_fb, n, scfg, f"{zcfg.name}_fb", seed + 1)
        self.w_rec = GenesisSynapse(n, n, scfg, f"{zcfg.name}_rec", seed + 2)
        g_rng = torch.Generator().manual_seed(seed + 10)
        self.w_pv = torch.rand(n, zcfg.n_pv, generator=g_rng) * 0.35      # PYR→PV
        self.w_pv_in = torch.rand(zcfg.n_pv, n, generator=g_rng) * 0.4    # PV→PYR (soma)
        self.w_sst_in = torch.rand(zcfg.n_sst, n, generator=g_rng) * 0.3  # SST→PYR (apical)
        self.w_vip_in = torch.rand(zcfg.n_vip, zcfg.n_sst, generator=g_rng) * 0.5
        self.kd_vip, self.n_vip_hill = 0.4, 2.0
        # L5-подмножество (нижняя треть пула) → HO-драйверы (сохраняет интерфейс)
        self.l5_idx = torch.arange(2 * n // 3, n)
        # Ламинарные слои по градиенту Барбас + профили входа FF/FB
        g_val = float(getattr(zcfg, "g", getattr(zcfg, "gradient_pos", 0.0)))
        self.layer_slices = barbas_layer_slices(n, g_val, self.l5_idx)
        self.ff_layer_gain = torch.ones(1, n)
        self.fb_layer_gain = torch.ones(1, n)
        if self.layer_slices["L4"].stop > self.layer_slices["L4"].start:
            self.ff_layer_gain[:, self.layer_slices["L4"]] = 1.35
            self.fb_layer_gain[:, self.layer_slices["L4"]] = 0.35
        self.fb_layer_gain[:, self.layer_slices["L23"]] = 1.15
        self.fb_layer_gain[:, self.layer_slices["L5"]] = 1.10
        # Ламинарный LFP и эфаптическое поле (Уровень 6)
        self.lfp = EphapticLFP(n, self.layer_slices)
        # Базальное 7-уровневое дендритное дерево Poirazi / Beniaguev (v10.0: 7 ветвей с локальными Ca2+/NMDA-спайками)
        from ..core.dendrite import DendriticTree
        from .astrocyte import AstrocyteMicrodomain
        self.dendrite = DendriticTree(n, n_in, n_branch=7, seed=seed + 3)
        self.astrocyte = AstrocyteMicrodomain(n, tau_ca=2500.0, gamma_gli=0.65,
                                            theta_ca=0.20, diffusion=0.05, ne_boost=1.5)        # D3 (archnext.txt): O(1) Локальный оценщик критичности колонки без глобального readout
        self.local_sigma: float = 0.97
        self._loc_act_mean: float = 0.06
        self._loc_act_prev1: float = 0.06
        self._loc_act_prev2: float = 0.06
        self._loc_c0: float = 1e-4
        self._loc_c1: float = 0.97e-4
        self._loc_c2: float = 0.94e-4
        # --- Фаза 1 v8.0 (snnscience.docx §3.1, §3.2, §3.5, §3.7): Многоскоростное ядро ---
        self.tick_ladder = TickLadderScheduler(stride_2=10, stride_3=100, degenerate=False)
        self.minicolumn_hierarchy = MinicolumnStateHierarchy(
            n_neurons=n,
            tile_size=16,
            e_l=ncfg.e_l,
            tau_m=ncfg.c_m / max(1e-4, ncfg.g_l),
            tau_adapt=ncfg.tau_adapt,
        )
        self.last_phase1_stats: dict[str, float] = {}
        self._bcm_post_activity: float = 0.0

    def reset_state(self, batch: int, device=None, dtype=torch.float32) -> None:
        self.pyr.reset_state(batch, device, dtype)
        self.dendrite.reset_state(batch, device, dtype)
        self.astrocyte.reset_state(batch, device=device, dtype=dtype)
        self.lfp.reset_state(batch)
        self.v_pv.zero_()
        self.v_sst.zero_()
        self.v_vip.zero_()
        self.local_sigma = 0.97
        self._loc_act_mean = 0.06
        self._loc_act_prev1 = 0.06
        self._loc_act_prev2 = 0.06
        self._loc_c0 = 1e-4
        self._loc_c1 = 0.97e-4
        self._loc_c2 = 0.94e-4

    def _advance_synaptic_clock(self, x_ff: torch.Tensor, x_fb: torch.Tensor,
                                x_rec: torch.Tensor, s_post: torch.Tensor,
                                decay_weights: bool = True) -> None:
        """One v20 tick, after forward (zero activity for dormant columns).

        BCM uses a column-local mean-square population rate, smoothed on the
        existing configurable calcium timescale rather than a global threshold.
        """
        alpha = min(1.0 / self.pyr.cfg.tau_ca, 1.0)
        measured = float(s_post.mean().item()) ** 2
        self._bcm_post_activity = (
            (1.0 - alpha) * getattr(self, "_bcm_post_activity", 0.0) + alpha * measured
        )
        for syn, pre in ((self.w_ff, x_ff), (self.w_fb, x_fb), (self.w_rec, x_rec)):
            syn.theta_m = syn.cfg.theta_m0 + self._bcm_post_activity
            syn.advance_time(dt=1.0, s_pre=pre, s_post=s_post,
                             decay_weights=decay_weights)

    # ------------------------------------------------------------------
    def _ensure_astrocyte(self):
        """Астроцит создаётся ЛЕНИВО и всегда доступен.

        Восстановление старого чекпоинта не знает про новые поля: снимок
        v19 не содержит `astrocyte`, и код падал с
        `AttributeError: 'CorticalColumn' object has no attribute 'astrocyte'`
        на каждом шаге — то есть модель не отвечала ни на что. Ленивое
        создание чинит это и для любых будущих новых полей колонки.
        """
        astro = getattr(self, "astrocyte", None)
        if astro is None:
            from .astrocyte import AstrocyteMicrodomain
            n = int(getattr(self.zcfg, "n_neurons", 0) or getattr(self.pyr, "n", 1))
            astro = AstrocyteMicrodomain(n, tau_ca=2500.0, gamma_gli=0.65,
                                        theta_ca=0.20, diffusion=0.05, ne_boost=1.5)
            self.astrocyte = astro
        return astro

    def step(self, x_ff: torch.Tensor, x_fb: torch.Tensor,
             f_mod: float, vip_mod: float,
             ltp_field: Optional[torch.Tensor] = None,
             hipp_drive: Optional[torch.Tensor] = None,
             learn: bool = True, dt: float = 1.0) -> ColumnOut:
        """x_ff [B,n_in] → basal AMPA; x_fb [B,n_fb] → apical NMDA.
        hipp_drive — CA1→L5 через NMDA+Mg²⁺-гейт (прайминг, не спайк)."""
        mod_ctx = self._mod_ctx if isinstance(
            getattr(self, "_mod_ctx", None), dict) else {}
        ach_val = float(mod_ctx.get("ach", 0.5))
        ne_val = float(mod_ctx.get("ne", 0.3))
        da_phasic = mod_ctx.get("da_phasic", None)
        external = "clock_tick" in mod_ctx
        if external:
            assert dt == 1.0 and self.pyr.cfg.dt == 1.0
            learn = learn and bool(mod_ctx["learning_enabled"])

        # --- Уровень 2 (Hasselmo 2006) + Инновация 6: ACh, Окситоцин, Опиоиды и D1/D2 ---
        # Никотиновые рецепторы α4β2 + окситоцин усиливают восходящий вход (FF) в L4,
        # мускариновые рецепторы M1/M4 подавляют старую реверберацию при высоком ACh,
        # а баланс D1/D2 управляет глубиной аттракторов w_rec и NMDA-проводимостью!
        snr_boost = float(mod_ctx.get("snr_boost", 1.0))
        d1_boost = float(mod_ctx.get("d1_nmda_boost", 1.0))
        d2_flex = float(mod_ctx.get("d2_flexibility", 0.0))
        net_reset = float(mod_ctx.get("network_reset", 0.0))
        nic_ff_gain = (1.0 + 0.35 * ach_val) * (1.0 + 0.04 * max(0.0, snr_boost - 1.25))
        # D1 углубляет рекуррентный аттрактор (в особенности в PFC), а D2 и Network Reset
        # ослабляют удержание старого аттрактора для быстрого переключения контекста
        is_pfc = float(getattr(self.zcfg, "g", 0.0)) >= 0.7
        d1_rec_gain = (1.0 + 0.15 * max(0.0, d1_boost - 1.3)) if is_pfc else 1.0
        musc_rec_gate = max(0.15, (1.0 - 0.35 * min(1.0, ach_val)) * d1_rec_gain * (1.0 - 0.25 * d2_flex) * (1.0 - 0.5 * net_reset))

        # 2-AG выталкивание из застрявшего аттрактора (VolumeTransmission eject)
        vol_eject = getattr(self, "_vol_eject", None)
        if vol_eject is not None and vol_eject.numel() == self.zcfg.n_neurons:
            if float(vol_eject.max().item()) > 0.0:
                self.w_rec.u_syn.mul_(1.0 - 0.35 * vol_eject.unsqueeze(1).clamp(0.0, 1.0)).clamp_(0.05, 1.0)

        apical_hint = (
            (self.pyr.v_a - self.pyr.cfg.e_l) / 20.0
            if hasattr(self.pyr, "v_a") and self.pyr.v_a.numel() == self.zcfg.n_neurons
            else None
        )
        g_ampa = self.w_ff.drive(x_ff, apical_subspace_hint=apical_hint).clamp(0.0, None) * 4.0 * nic_ff_gain * self.ff_layer_gain
        g_nmda = self.w_fb.drive(x_fb, apical_subspace_hint=apical_hint).clamp(0.0, None) * self.fb_layer_gain * (1.0 + 0.10 * max(0.0, d1_boost - 1.3))
        # Закон 4: гиппокамп→L5 по апикальному NMDA-каналу; Mg-блок
        # в нейроне гарантирует, что без базального входа CA1 только
        # подготавливает кору (суб-пороговый прайминг).
        if hipp_drive is None:
            hipp_drive = getattr(self, "_hipp_in", None)
        if hipp_drive is not None:
            hd = hipp_drive
            if hd.dim() == 1:
                hd = hd.unsqueeze(0)
            g_nmda = g_nmda.clone()
            g_nmda[:, self.l5_idx] += hd[:, :self.l5_idx.numel()]
        g_rec = (self.w_rec.drive(self._last_s, apical_subspace_hint=apical_hint) * musc_rec_gate
                 if hasattr(self, "_last_s") else torch.zeros_like(g_ampa))

        # --- Спайковые популяции интернейронов PV, SST и VIP (Уровень 2 и 6) ---
        pyr_s = self._last_s if hasattr(self, "_last_s") \
            else torch.zeros(x_ff.shape[0], self.zcfg.n_neurons)
        # 1) VIP-интернейроны возбуждаются никотиновым ACh и верхнеуровневым vip_mod
        vip_in = pyr_s.mean(0)[:self.zcfg.n_vip].mean() * 0.4 + vip_mod + 0.6 * ach_val
        self.v_vip = self.v_vip * 0.85 + vip_in * 0.25
        vip_s = (self.v_vip > 0.45).float()
        self.v_vip = self.v_vip * (1.0 - vip_s * 0.5)
        self.vip = 0.8 * self.vip + 0.2 * (0.5 * vip_s + 0.5 * self.v_vip.clamp(0.0, 1.5))
        vip_mean = float(self.vip.mean().item())
        self.vip_drive = vip_mean

        # 2) SST-интернейроны (клетки Мартинотти): тормозят апикальный дендрит,
        #    но подавляются активностью VIP: g_SST = g0 / (1 + ([VIP]/Kd)^n)
        sst_gate = 1.0 / (1.0 + (vip_mean / self.kd_vip) ** self.n_vip_hill)
        self._last_sst_gate = sst_gate
        sst_drive = (g_nmda.mean(0) @ self.w_sst_in.t() * 0.15) - (self.vip @ self.w_vip_in) * 0.4
        self.v_sst = (self.v_sst * 0.88 + sst_drive * 0.20).clamp(min=0.0)
        self.sst = self.v_sst * sst_gate
        g_sst = (self.sst.unsqueeze(0) @ self.w_sst_in) * sst_gate
        g_nmda = g_nmda * sst_gate              # SST гейтит апикал

        # 3) PV-интернейроны (быстрые корзинчатые клетки): перисоматический k-WTA тормоз
        pv_drive = (pyr_s @ self.w_pv).clamp(min=0.0)
        self.v_pv = self.v_pv * 0.80 + pv_drive.mean(0) * 0.30
        self.pv = self.v_pv
        k = max(1, int(0.3 * self.zcfg.n_neurons))
        thr = pyr_s.topk(k, dim=1).values[:, -1:]
        g_pv = (pv_drive @ self.w_pv_in) * (pyr_s < thr).float()

        # Бёрлин E/I: торможение ∝ остаточной ошибке кодирования × опиоидная дизингибиция
        disinh = float(mod_ctx.get("disinhibition", 1.0))
        ei_scale = (1.0 + 0.5 * float(getattr(self, "_last_resid", 0.0))) * (0.7 + 0.3 * disinh)
        g_gaba = ((g_pv + g_sst) * ei_scale).clamp(0.0, 5.0)

        # --- Уровень 4: Локальный тормозной iSTDP Вогельса (Vogels et al., Science 2011) ---
        # Автоматически подстраивает w_pv_in на каждом такте, удерживая целевую частоту
        # пирамидальных клеток ρ₀≈0.06 и эмерджентную критичность σ≈1.0 без внешней руки!
        interneuron_weight_updates = 0
        interneuron_rows_updated = 0
        if learn and pv_drive.abs().max() > 0:
            rho0 = 0.06
            post_err = pyr_s.mean(0) - rho0                  # [N_pyr]
            pre_pv = pv_drive.mean(0).clamp(0.0, 2.0)        # [N_pv]
            dw_istdp = 0.0015 * torch.outer(pre_pv, post_err)
            w_pv_before = self.w_pv_in
            self.w_pv_in = (self.w_pv_in + dw_istdp).clamp(0.05, 3.0)
            changed = self.w_pv_in != w_pv_before
            interneuron_weight_updates = int(changed.sum().item())
            interneuron_rows_updated = int(changed.any(dim=1).sum().item())

        # Базальный вход идёт через 7-уровневые ветви Poirazi/Beniaguev: суперлинейная сумма + Ca2+/NMDA-плато + ADCG
        g_tree = self.dendrite.step(x_ff, pyr_s, apical_potential=self.pyr.v_a, dt=dt)
        g_ampa = (g_ampa + g_tree).clamp(0.0, None)
        # A1 quenches the complete FF (including tree) and REC conductance once;
        # the independent release-probability path remains inside pyr.step.
        if external:
            a1_gain = float(mod_ctx["a1_gain"])
            g_ampa = g_ampa * a1_gain
            g_rec = g_rec * a1_gain
        # Тонический фон + эфаптическое поле LFP (Уровень 6)
        i_ext = torch.full_like(g_ampa, 1.5)
        eph_mv = self.lfp.ephaptic_mv if self.lfp.ephaptic_mv.shape == g_ampa.shape else None
        out = self.pyr.step(g_ampa + g_rec * 0.5, g_nmda, g_gaba, i_ext,
                            ach=ach_val,
                            ne=ne_val,
                            histamine=mod_ctx.get("histamine", 0.0),
                            nmda_gain=mod_ctx.get("nmda_gain", 1.0),
                            p_rel=mod_ctx.get("p_rel", 1.0),
                            vth_shift=mod_ctx.get("vth_shift", 0.0),
                            gaba_tonic=mod_ctx.get("gaba_tonic", 1.0),
                            g_gaba_d=g_sst,
                            ephaptic_mv=eph_mv,
                            rate_ceiling=max(1.0, float(mod_ctx.get("rate_ceiling", 1.0)) - 0.30),
                            ahp_block=mod_ctx.get("ahp_block", None) if float(mod_ctx.get("ahp_block", 0.0)) > 0.65 else None)
        # Обновляем ламинарный LFP и эфаптический диполь для следующего такта
        self.lfp.step(self.pyr.last_i_ff, self.pyr.last_i_fb,
                      self.pyr.last_i_inh, dt=dt)
        if external:
            self._advance_synaptic_clock(x_ff, x_fb, pyr_s, out.s,
                                         decay_weights=learn)
        self._last_s = out.s
        self._last_x_ff = x_ff.detach()
        self._last_x_fb = x_fb.detach()

        astro = self._ensure_astrocyte()
        astro_gate = astro.step(out.s, ne=ne_val)
        astro_gain = float(astro_gate.mean().item())
        for syn in (self.w_ff, self.w_fb, self.w_rec):
            syn.set_stp_gain(astro_gain)
        if hasattr(self.dendrite, "apply_theta_ca_shift"):
            self.dendrite.apply_theta_ca_shift(astro.theta_ca_shift())

        # --- D3 (archnext.txt): O(1) Локальная оценка критичности колонки без глобального readout ---
        # Использует 2-лаговую экспоненциальную автоковариацию (Local MR Estimator, Wilting & Priesemann 2018)
        # и локальный баланс возбуждения/торможения внутри самой колонки:
        act_c = float((0.5 * out.s.mean() + 0.5 * out.rate_d.mean()).item())
        alpha_c = 0.025
        self._loc_act_mean = (1.0 - alpha_c) * self._loc_act_mean + alpha_c * act_c
        d0 = act_c - self._loc_act_mean
        d1 = self._loc_act_prev1 - self._loc_act_mean
        d2 = self._loc_act_prev2 - self._loc_act_mean
        self._loc_c0 = (1.0 - alpha_c) * self._loc_c0 + alpha_c * (d0 * d0) + 1e-8
        self._loc_c1 = (1.0 - alpha_c) * self._loc_c1 + alpha_c * abs(d0 * d1) + 0.97e-8
        self._loc_c2 = (1.0 - alpha_c) * self._loc_c2 + alpha_c * abs(d1 * d2) + 0.94e-8
        self._loc_act_prev2 = self._loc_act_prev1
        self._loc_act_prev1 = act_c
        r1_loc = min(1.08, max(0.86, self._loc_c1 / max(1e-8, self._loc_c0)))
        r2_loc = min(1.08, max(0.86, (self._loc_c2 / max(1e-8, self._loc_c1))))
        mr_loc = 0.5 * (r1_loc + r2_loc)
        act_tone = 0.935 + 0.65 * min(0.18, max(0.0, act_c))
        inst_loc_sigma = 0.40 * mr_loc + 0.60 * act_tone
        self.local_sigma = float(0.88 * self.local_sigma + 0.12 * inst_loc_sigma)
        local_plasticity_gain = float(max(0.6, min(1.8, 0.75 + 0.9 * self.local_sigma)))
        self.last_local_plasticity_gain = local_plasticity_gain

        # --- P1 (Mikulasch et al. 2023): Дендритное предиктивное обучение SST→Apical ---
        if learn and float(self.sst.abs().max()) > 1e-5:
            apical_mismatch = ((self.pyr.v_a - self.pyr.cfg.e_l) / 25.0).mean(0).clamp(-1.0, 1.0)
            dw_sst = 0.001 * torch.outer(self.sst.clamp(0.0, 2.0), apical_mismatch)
            self.w_sst_in = (self.w_sst_in + dw_sst).clamp(0.02, 1.5)

        if ltp_field is None:
            ltp_field = getattr(self, "_no_ltp_field", None)
        synaptic_rows_updated = 0
        synaptic_weight_updates = 0
        if learn:
            lam_atp = float(mod_ctx.get("lambda_atp", 0.0))
            effective_error = out.error * local_plasticity_gain
            effective_f_mod = max(0.0, float(f_mod) * local_plasticity_gain)
            for syn, pre in ((self.w_ff, x_ff), (self.w_fb, x_fb),
                             (self.w_rec, pyr_s if external else self._last_s)):
                before = syn.w_fast + syn.w_slow
                syn.learn(effective_error, pre, out.s, effective_f_mod, ltp_field, dt,
                          ach=ach_val, da_phasic=da_phasic, lambda_atp=lam_atp)
                changed = ((syn.w_fast + syn.w_slow) != before) & syn.mask
                synaptic_rows_updated += int(changed.any(dim=1).sum().item())
                synaptic_weight_updates += int(changed.sum().item())
            ap_err_step = ((self.pyr.v_a - self.pyr.cfg.e_l) / 25.0).clamp(-1.5, 1.5)
            # The tree records the prior-cycle soma target in step(); this
            # same-cycle neuron error is the owning column's teaching signal.
            self.dendrite.learn(effective_error, x_ff, apical_error=ap_err_step,
                                f_mod=effective_f_mod)
            # Узел 4: Бодрствующий структурный морфогенез при высокой ошибке и всплеске BDNF
            bdnf_val = float(mod_ctx.get("bdnf", 0.35))
            err_inst = float(out.error.abs().mean().item())
            if err_inst > 0.16 and bdnf_val > 0.42:
                self.w_ff.structural_plasticity(error_level=err_inst, bdnf=bdnf_val)
                self.w_rec.structural_plasticity(error_level=err_inst, bdnf=bdnf_val)
        # --- Фаза 1 v8.0: Шаг логарифмической лестницы тиков и коалесированных ярусов HOT/WARM/COLD ---
        if external:
            self.tick_ladder.tick = int(mod_ctx["clock_tick"]) - 1
        ladder_step = self.tick_ladder.step(
            is_sleep_tick=bool(mod_ctx.get("is_sleep_tick", False)),
            lambda_atp=float(mod_ctx.get("lambda_atp", 0.0)),
        )
        batch_size = int(out.s.shape[0]) if out.s.dim() > 1 else 1
        self.last_phase1_stats = self.minicolumn_hierarchy.update_tiers(
            tick=ladder_step.tick,
            afferent_drive=g_ampa + g_nmda,
            spikes=out.s,
            v_s=self.pyr.v_s,
            adapt=self.pyr.theta,
            ladder_step=ladder_step,
            measured_neuron_updates=batch_size * self.zcfg.n_neurons,
            measured_synaptic_rows_updated=synaptic_rows_updated + interneuron_rows_updated,
            measured_synaptic_weight_updates=synaptic_weight_updates + interneuron_weight_updates,
            g_ampa=g_ampa,
            g_nmda=g_nmda,
            g_gaba=g_gaba,
        )
        self.last_phase1_stats.update(
            measured_neuron_step_elements=int(out.s.numel()),
            measured_neuron_updates=batch_size * self.zcfg.n_neurons,
            measured_synaptic_rows_updated=synaptic_rows_updated,
            measured_synaptic_weight_updates=synaptic_weight_updates,
            measured_interneuron_rows_updated=interneuron_rows_updated,
            measured_interneuron_weight_updates=interneuron_weight_updates,
            measured_total_weight_rows_updated=synaptic_rows_updated + interneuron_rows_updated,
            measured_total_weight_updates=synaptic_weight_updates + interneuron_weight_updates,
            local_sigma=self.local_sigma,
            local_sigma_plasticity_gain=local_plasticity_gain,
        )
        return ColumnOut(s=out.s, error=out.error,
                         l5_drive=out.s[:, self.l5_idx],
                         v_s=out.v_s, ca=out.ca, rate_d=out.rate_d,
                         local_sigma=self.local_sigma,
                         local_sigma_gain=local_plasticity_gain)

    def enable_phase1_v8(self, enable_scaffold: bool = True,
                         enable_block_sparse: bool = True,
                         promote_core_quantile: Optional[float] = None) -> None:
        """Включает механизмы Фазы 1 v8.0 (MC-MREI, CT-PNN Scaffold, PDSR Block-Sparse)."""
        self.tick_ladder.degenerate = False
        self.minicolumn_hierarchy.degenerate = False
        for w in (self.w_ff, self.w_fb, self.w_rec):
            w.enable_scaffold = bool(enable_scaffold)
            w.enable_block_sparse = bool(enable_block_sparse)
            if promote_core_quantile is not None:
                w.promote_scaffold_first(weight_quantile=float(promote_core_quantile))

    def phase1_telemetry(self) -> dict[str, object]:
        """Возвращает сводную телеметрию Фазы 1 v8.0 по колонке."""
        return {
            "zone": self.name,
            "local_sigma": float(self.local_sigma),
            "local_sigma_plasticity_gain": float(self.last_phase1_stats.get("local_sigma_plasticity_gain", self.local_sigma)),
            "state_compression_ratio": round(
                float(self.last_phase1_stats.get("compression_ratio", 1.0)), 3
            ),
            "ladder_compression_ratio": round(self.tick_ladder.ladder_compression_ratio(), 3),
            "hot_frac": round(float(self.last_phase1_stats.get("hot_frac", 1.0)), 3),
            "warm_frac": round(float(self.last_phase1_stats.get("warm_frac", 0.0)), 3),
            "cold_frac": round(float(self.last_phase1_stats.get("cold_frac", 0.0)), 3),
            "exact_wakeups": int(self.minicolumn_hierarchy.total_exact_wakeups),
            "measured_neuron_step_elements": int(self.last_phase1_stats.get("measured_neuron_step_elements", 0)),
            "measured_neuron_updates": int(self.last_phase1_stats.get("measured_neuron_updates", 0)),
            "measured_synaptic_rows_updated": int(self.last_phase1_stats.get("measured_synaptic_rows_updated", 0)),
            "measured_synaptic_weight_updates": int(self.last_phase1_stats.get("measured_synaptic_weight_updates", 0)),
            "measured_interneuron_rows_updated": int(self.last_phase1_stats.get("measured_interneuron_rows_updated", 0)),
            "measured_interneuron_weight_updates": int(self.last_phase1_stats.get("measured_interneuron_weight_updates", 0)),
            "measured_total_weight_rows_updated": int(self.last_phase1_stats.get("measured_total_weight_rows_updated", 0)),
            "measured_total_weight_updates": int(self.last_phase1_stats.get("measured_total_weight_updates", 0)),
            "flop_efficiency_ff": round(float(self.w_ff.last_flop_efficiency), 4),
            "scaffold_ff": self.w_ff.scaffold_stats(),
        }

    def inject_apical_error(self, apical_err: torch.Tensor,
                            f_mod: float = 1.0,
                            ach: float = 0.5,
                            da_phasic: Optional[float] = None) -> torch.Tensor:
        """P1 (Sacramento et al. 2018 / Urbanczik-Senn 2014):
        Инжекция нисходящего целевого сигнала ошибки в апикальные дендриты колонки.
        Деполяризует/гиперполяризует V_A, обучает базальные FF/REC/ветви Poirazi
        и возвращает ошибку для следующей нижележащей зоны."""
        mod_ctx = getattr(self, "_mod_ctx", {})
        external = isinstance(mod_ctx, dict) and "clock_tick" in mod_ctx
        if external:
            if not mod_ctx["learning_enabled"]:
                return torch.zeros_like(self.pyr.v_a)
            ach = float(mod_ctx.get("ach", ach))
            da_phasic = mod_ctx.get("da_phasic", None)
        if f_mod <= 0.0 or not hasattr(self, "_last_x_ff"):
            return torch.zeros(1, self.zcfg.n_neurons)
        if apical_err.dim() == 1:
            apical_err = apical_err.unsqueeze(0)
        # Гейтинг через VIP/SST: при высоком внимании/новизне (VIP↑ → sst_gate↓)
        # апикальный дендрит открыт для приёма обучающей ошибки сверху!
        sst_g = float(getattr(self, "_last_sst_gate", 0.5))
        apical_open = 1.0 - 0.5 * sst_g
        eff_err = (apical_err * apical_open).clamp(-1.5, 1.5)
        # Сдвигаем потенциал апикального дендрита V_A
        self.pyr.v_a = (self.pyr.v_a + eff_err * 6.0).clamp(-100.0, 40.0)
        self.pyr.apical_error = eff_err
        # SST-интернейроны адаптируются к апикальной ошибке (Mikulasch et al. 2023)
        sst_act = self.sst.clamp(min=0.01, max=2.0)
        self.w_sst_in = (self.w_sst_in + 0.002 * f_mod * torch.outer(
            sst_act, eff_err.mean(0)
        )).clamp(0.02, 1.5)
        # Локальное трёхфакторное обновление синапсов колонки по апикальному градиенту
        s_post = getattr(self, "_last_s", torch.zeros_like(eff_err))
        lam_atp = float(mod_ctx["lambda_atp"]) if external else 0.0
        self.w_ff.learn(eff_err, self._last_x_ff, s_post, f_mod * 0.65,
                        ach=ach, da_phasic=da_phasic, lambda_atp=lam_atp)
        self.w_rec.learn(eff_err, s_post, s_post, f_mod * 0.45,
                         ach=ach, da_phasic=da_phasic, lambda_atp=lam_atp)
        self.dendrite.learn(eff_err, self._last_x_ff, f_mod=f_mod * 0.65)
        return eff_err

    def l6_output(self) -> float:
        """Выход кортикоталамического слоя L6 для управления сектором TRN."""
        if not hasattr(self, "_last_s"):
            return 0.5
        sl_l6 = self.layer_slices.get("L6", slice(0, 0))
        if sl_l6.stop > sl_l6.start:
            val = float(self._last_s[:, sl_l6].mean().item())
        else:
            val = float(self._last_s.mean().item())
        return max(0.15, min(1.0, 0.35 + 1.5 * val))

    def consolidate(self, factor: float = 0.5,
                    spindle_gain: float = 1.0,
                    error_level: Optional[float] = None,
                    microglia_boost: float = 1.0,
                    bdnf: float = 0.35) -> None:
        """Узел 1 + 4: Сонная консолидация w_fast → w_slow под таламическими веретёнами
        и структурная пластичность по градиенту ошибки зоны + микроглиальный P2X гейт и BDNF."""
        for w in (self.w_ff, self.w_fb, self.w_rec):
            w.consolidate(factor, spindle_gain=spindle_gain)
            w.structural_plasticity(error_level=error_level,
                                    microglia_boost=microglia_boost,
                                    bdnf=bdnf)
