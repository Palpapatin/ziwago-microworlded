"""Регресс-тесты трека A: честность multi-rate ядра и σ̂-критичности (v22).

Каждый тест написан так, чтобы ПАДАТЬ на исходном коде (до правки трека A) и
проходить после неё. Проверено: на исходном коде файл даёт красный прогон
(см. раздел «Было → Стало» в отчёте трека A).

Покрытые пункты:
  1. multirate.py: приоритет операторов в `BennaFusiCascade.step` (ветви
     run_class_2_medium / run_class_3_slow никогда не исполнялись).
  2. multirate.py: тавтологичная «проверка причинного конуса»
     (`verify_causal_cone_transport`) + её вызов из живого пути `update_tiers`.
  3. multirate.py: «сжатие арифметики» — целевая модель стоимости, а не измерение.
  4. multirate.py: проводимости сна доходят до `exact_lazy_propagate`, а аналитический
     перенос НЕ перезаписывает живое состояние нейрона.
  5. criticality.py: σ̂ зоны — МЕРКА (MR-оценка), не кламп к ~1; очередь активности
     и статистика согласованы (сумма спайков, а не среднее по нейронам).
  6. homeostasis/controller.py: контроллер работает с реальными структурами
     (`moe_cortex.columns`), измерение σ̂ принадлежит `core/criticality.py`.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ziwago.config import HomeostasisCfg, USNeuronCfg, USSynapseCfg, ZoneCfg
from ziwago.core.criticality import CriticalityMonitor
from ziwago.core.multirate import (
    TIER_COLD,
    BennaFusiCascade,
    MinicolumnStateHierarchy,
    TickLadderScheduler,
    TickLadderStep,
)
from ziwago.homeostasis.controller import HomeostaticController


# ---------------------------------------------------------------------------
# Вспомогательные конструкции
# ---------------------------------------------------------------------------

def _ladder_step(tick: int, class2: bool, class3: bool, sleep: bool = False) -> TickLadderStep:
    return TickLadderStep(
        tick=tick,
        run_class_1_fast=True,
        run_class_2_medium=class2,
        run_class_3_slow=class3,
        run_class_4_sleep=sleep,
        eff_stride_2=10,
        eff_stride_3=100,
    )


def _filled_cascade(shape=(3, 3), n_stages: int = 4) -> BennaFusiCascade:
    cas = BennaFusiCascade(shape=shape, n_stages=n_stages, tau_base=10.0)
    for k in range(n_stages):
        cas.u[k] = torch.full(shape, float(k + 1))
    return cas


def _quiet_kwargs(n: int = 8) -> dict:
    return {
        "afferent_drive": torch.zeros(1, n),
        "spikes": torch.zeros(1, n),
        "v_s": torch.full((1, n), -65.0),
        "adapt": torch.full((1, n), -50.0),
    }


def _tiny_column(name: str = "SENS"):
    """Настоящая CorticalColumn минимального размера (8 нейронов)."""
    from ziwago.cortex.column import CorticalColumn

    col = CorticalColumn(
        zcfg=ZoneCfg(name=name, g=0.5, n_neurons=8, n_pv=2, n_sst=2, n_vip=2),
        n_in=4,
        n_fb=8,
        ncfg=USNeuronCfg(),
        scfg=USSynapseCfg(),
        seed=3,
    )
    col.reset_state(1)
    return col


class _Cortex:
    """Stand-in за `moe_cortex.SparselyGatedCorticalMoE`: у него атрибут `.columns`."""

    def __init__(self, columns):
        self.columns = list(columns)


def _ar1_series(phi: float = 0.6, n: int = 400, seed: int = 7) -> np.ndarray:
    """Субкритическая серия активности (AR(1)) — ожидаемая MR-оценка m̂ ≈ 0.6."""
    rng = np.random.default_rng(seed)
    a = np.zeros(n)
    for i in range(1, n):
        a[i] = phi * a[i - 1] + rng.standard_normal()
    return np.abs(a)


# ---------------------------------------------------------------------------
# Пункт 1 — приоритет операторов: ступень обновления каскада
# ---------------------------------------------------------------------------

def test_cascade_depth_is_limited_when_class2_is_not_running():
    """Класс II выключен → обновляется только резервуар 0 (Класс I)."""
    cas = _filled_cascade()
    before = [u.clone() for u in cas.u]
    cas.step(dt=1.0, ladder_step=_ladder_step(1, class2=False, class3=False))
    assert not torch.equal(cas.u[0], before[0]), "резервуар 0 (Класс I) обязан обновляться"
    assert torch.equal(cas.u[1], before[1]), "без Класса II резервуар 1 обязан быть заморожен"
    assert torch.equal(cas.u[2], before[2]), "без Класса III резервуар 2 обязан быть заморожен"
    assert torch.equal(cas.u[3], before[3]), "без Класса III резервуар 3 обязан быть заморожен"


def test_cascade_depth_is_limited_to_two_stages_when_only_class2_runs():
    """Класс II включён, Класс III выключен → обновляются резервуары 0 и 1."""
    cas = _filled_cascade()
    before = [u.clone() for u in cas.u]
    cas.step(dt=1.0, ladder_step=_ladder_step(10, class2=True, class3=False))
    assert not torch.equal(cas.u[0], before[0])
    assert not torch.equal(cas.u[1], before[1]), "Класс II обязан двигать резервуар 1"
    assert torch.equal(cas.u[2], before[2]), "без Класса III резервуар 2 заморожен"
    assert torch.equal(cas.u[3], before[3]), "без Класса III резервуар 3 заморожен"


def test_cascade_updates_all_stages_when_class3_runs():
    """Класс III включён → глубокие резервуары обновляются."""
    cas = _filled_cascade()
    before = [u.clone() for u in cas.u]
    cas.step(dt=1.0, ladder_step=_ladder_step(100, class2=True, class3=True))
    for k in range(4):
        assert not torch.equal(cas.u[k], before[k]), f"резервуар {k} обязан обновляться"


def test_cascade_ladder_step_carries_its_own_degenerate_flag():
    """Вырожденный режим лестницы обязан быть виден в самом шаге (иначе — hasattr-хак)."""
    step = _ladder_step(1, class2=False, class3=False)
    assert step.degenerate is False
    # в вырожденном режиме все классы идут каждый тик → все резервуары движутся
    sched = TickLadderScheduler(stride_2=10, stride_3=100, degenerate=True)
    deg_step = sched.step()
    assert deg_step.degenerate is True
    cas = _filled_cascade()
    before = [u.clone() for u in cas.u]
    cas.step(dt=1.0, ladder_step=deg_step)
    for k in range(4):
        assert not torch.equal(cas.u[k], before[k]), f"резервуар {k} обязан обновляться"


# ---------------------------------------------------------------------------
# Пункт 2 — причинный конус: настоящее возмущение + вызов из живого пути
# ---------------------------------------------------------------------------

def test_causal_cone_check_detects_a_real_perturbation():
    """Возмущение начальных условий обязано оставлять ненулевой хвост в окне ДО t_pert."""
    h = MinicolumnStateHierarchy(n_neurons=8, tile_size=8)
    res = h.verify_causal_cone_transport(
        total_horizon_ms=40.0, perturbation_time_ms=20.0, perturbation_amp=1.5)
    assert res["pre_causal_cone_error"] > 0.0, (
        "разные начальные условия обязаны давать ненулевую разницу в окне до возмущения")
    assert res["pre_causal_cone_leak_ratio"] > 0.0
    assert res["inside_causal_cone_response"] > 0.01, "возмущение обязано доходить до окна после t_pert"
    assert res["causality_preserved"] == 0.0, (
        "экспоненциальный транспорт имеет бесконечную память: строгая изоляция конуса ломается")
    assert res["perturbation_registered"] == 1.0


def test_causal_cone_check_is_causal_without_perturbation():
    """Без возмущения траектории совпадают ⇒ причинный конус не нарушен."""
    h = MinicolumnStateHierarchy(n_neurons=8, tile_size=8)
    res = h.verify_causal_cone_transport(
        total_horizon_ms=40.0, perturbation_time_ms=20.0, perturbation_amp=0.0)
    assert res["pre_causal_cone_error"] == 0.0
    assert res["causality_preserved"] == 1.0
    assert res["inside_causal_cone_response"] == 0.0


def test_causal_cone_check_runs_inside_the_live_update_tiers_path():
    """Живой путь (`update_tiers` ← cortex/column.py) обязан выполнять проверку конуса."""
    h = MinicolumnStateHierarchy(n_neurons=8, tile_size=8)
    stats = h.update_tiers(tick=1, **_quiet_kwargs(8))
    assert isinstance(h.last_causal_cone_check, dict)
    assert "causal_cone_leak_ratio" in stats
    assert stats["causal_cone_leak_ratio"] > 0.0
    assert stats["causal_cone_preserved"] == 0.0


# ---------------------------------------------------------------------------
# Пункт 3 — «сжатие арифметики» как целевая модель стоимости, а не измерение
# ---------------------------------------------------------------------------

def test_state_cost_ratio_is_labelled_as_a_target_cost_model():
    h = MinicolumnStateHierarchy(n_neurons=8, tile_size=8)
    stats = h.update_tiers(tick=1, **_quiet_kwargs(8))
    assert "target_cost_compression_ratio" in stats, "метрика обязана иметь честное имя"
    assert stats["compression_ratio"] == stats["target_cost_compression_ratio"], (
        "legacy-ключ обязан оставаться тем же числом, что и явная целевая модель")
    hot, warm, cold = stats["hot_frac"], stats["warm_frac"], stats["cold_frac"]
    # целевая модель: HOT = 4 (Класс I) + 4 (Класс II), WARM = 8, COLD = 0.05
    expected_per_neuron = hot * 8.0 + warm * 8.0 + cold * 0.05
    assert stats["target_cost_per_neuron"] == pytest.approx(expected_per_neuron)
    assert stats["target_cost_compression_ratio"] == pytest.approx(15.0 / expected_per_neuron)


def test_cost_accounting_separates_measured_counters_from_the_target_model():
    n = 8
    h = MinicolumnStateHierarchy(n_neurons=n, tile_size=n)
    stats = h.update_tiers(tick=1, **_quiet_kwargs(n))
    # реально выполненная работа модуля измерима: на тихом тике она нулевая
    assert stats["measured_exact_wakeup_neurons"] == 0.0
    assert stats["measured_exact_wakeups"] == 0.0
    assert stats["measured_state_vars_touched"] == 0.0
    assert h.total_exact_wakeups == 0
    # модель обещает экономию, фактически выполнено 0 переменных состояния ⇒
    # «коэффициент сжатия» — целевая модель стоимости, а НЕ измерение
    assert stats["target_cost_compression_ratio"] > 1.0
    assert stats["compression_ratio"] == stats["target_cost_compression_ratio"]


def test_ladder_cost_ratio_is_a_target_model_recomputed_from_the_schedule():
    sched = TickLadderScheduler(stride_2=10, stride_3=100)
    for _ in range(10):
        sched.step()
    # 10 тиков: 4 переменных Класса I на каждом + 4 переменных Класса II на 10-м тике
    assert sched.target_cost_compression_ratio() == pytest.approx(150.0 / 44.0)
    assert sched.ladder_compression_ratio() == pytest.approx(150.0 / 44.0)


# ---------------------------------------------------------------------------
# Пункт 4 — проводимости сна и «точное» пробуждение
# ---------------------------------------------------------------------------

def test_sleep_conductances_are_captured_and_reach_the_analytic_transfer():
    n = 8
    h = MinicolumnStateHierarchy(n_neurons=n, tile_size=n)
    kwargs = _quiet_kwargs(n)
    g_ampa = torch.full((1, n), 0.4)
    g_nmda = torch.full((1, n), 0.25)
    g_gaba = torch.full((1, n), 0.1)
    for t in range(1, 25):
        h.update_tiers(tick=t, g_ampa=g_ampa, g_nmda=g_nmda, g_gaba=g_gaba, **kwargs)
    assert (h.tile_tier == TIER_COLD).all(), "тихий тайл обязан уснуть"
    assert torch.allclose(h.g_ampa_sleep, g_ampa[0]), "проводимость AMPA обязана быть захвачена"
    assert torch.allclose(h.g_nmda_sleep, g_nmda[0]), "проводимость NMDA обязана быть захвачена"
    assert torch.allclose(h.g_gaba_sleep, g_gaba[0]), "проводимость GABA обязана быть захвачена"
    assert bool(h.tile_sleep_conductances_known.all())

    h.update_tiers(tick=25, afferent_drive=torch.full((1, n), 1.0), spikes=torch.zeros(1, n),
                   v_s=kwargs["v_s"], adapt=kwargs["adapt"],
                   g_ampa=g_ampa, g_nmda=g_nmda, g_gaba=g_gaba)
    assert h.total_exact_wakeups == 1
    assert h.last_wakeup["conductances_known"] is True
    assert h.last_wakeup["g_ampa_sleep_used"] == pytest.approx(0.4, rel=1e-5)


def test_analytic_wakeup_without_conductances_is_reported_as_such():
    """Нет данных о проводимостях сна ⇒ перенос не объявляется «точным»."""
    n = 8
    h = MinicolumnStateHierarchy(n_neurons=n, tile_size=n)
    kwargs = _quiet_kwargs(n)
    for t in range(1, 25):
        h.update_tiers(tick=t, **kwargs)
    assert (h.tile_tier == TIER_COLD).all()
    h.update_tiers(tick=25, afferent_drive=torch.full((1, n), 1.0), spikes=torch.zeros(1, n),
                   v_s=kwargs["v_s"], adapt=kwargs["adapt"])
    assert h.total_exact_wakeups == 1
    assert h.last_wakeup["conductances_known"] is False


def test_analytic_wakeup_does_not_overwrite_live_neuron_state():
    """Результат аналитики не имеет права перезаписывать живое состояние нейрона."""
    n = 8
    h = MinicolumnStateHierarchy(n_neurons=n, tile_size=n)
    kwargs = _quiet_kwargs(n)
    for t in range(1, 25):
        h.update_tiers(tick=t, **kwargs)
    assert (h.tile_tier == TIER_COLD).all()
    v_before = kwargs["v_s"].clone()
    adapt_before = kwargs["adapt"].clone()
    h.update_tiers(tick=25, afferent_drive=torch.full((1, n), 1.0), spikes=torch.zeros(1, n),
                   v_s=kwargs["v_s"], adapt=kwargs["adapt"])
    assert torch.equal(kwargs["v_s"], v_before), "живой V_S не должен перезаписываться аналитикой"
    assert torch.equal(kwargs["adapt"], adapt_before), "живой порог не должен перезаписываться аналитикой"
    assert h.total_exact_wakeups == 1
    # аналитический перенос остаётся доступен как явный кандидат
    assert "v_restored" in h.last_wakeup
    assert not torch.allclose(h.last_wakeup["v_restored"], v_before[0])


def test_wakeup_with_and_without_conductances_gives_different_transfer():
    n = 8
    g_ampa = torch.full((1, n), 0.4)
    g_nmda = torch.full((1, n), 0.25)
    g_gaba = torch.full((1, n), 0.1)

    def _run(with_conductances: bool):
        h = MinicolumnStateHierarchy(n_neurons=n, tile_size=n)
        kwargs = _quiet_kwargs(n)
        extra = {}
        if with_conductances:
            extra = {"g_ampa": g_ampa, "g_nmda": g_nmda, "g_gaba": g_gaba}
        for t in range(1, 25):
            h.update_tiers(tick=t, **kwargs, **extra)
        h.update_tiers(tick=25, afferent_drive=torch.full((1, n), 1.0), spikes=torch.zeros(1, n),
                       v_s=kwargs["v_s"], adapt=kwargs["adapt"], **extra)
        return h.last_wakeup["v_restored"]

    assert not torch.allclose(_run(True), _run(False))


# ---------------------------------------------------------------------------
# Пункт 5 — σ̂ как измерение, а не кламп к ~1
# ---------------------------------------------------------------------------

def test_zone_sigma_is_the_mr_measurement_not_a_clamp_to_one():
    mon = CriticalityMonitor(k_max=20)
    series = _ar1_series()
    for v in series:
        mon.push_zones({"SENS": torch.full((8,), float(v))})
    measured = mon.zone_sigmas["SENS"]
    expected = mon._mr_estimate(mon.zone_activity["SENS"])
    assert expected < 0.9, "фикстура должна быть заведомо субкритической"
    assert measured == pytest.approx(expected), "σ̂ зоны обязана быть MR-оценкой, а не смесью с «тонусом»"
    assert not (0.94 <= measured <= 1.11), "σ̂ не должна притягиваться клампом к ~1"


def test_zone_sigma_is_not_fabricated_before_enough_data():
    mon = CriticalityMonitor(k_max=20)
    mon.push_zones({"PFC": torch.tensor([1.0, 2.0, 0.0, 3.0])})
    assert mon.zone_sigma_ready["PFC"] is False, "до накопления данных σ̂ не считается измеренной"
    assert mon.zone_sigmas["PFC"] == pytest.approx(mon.target)


def test_zone_activity_queue_and_statistics_use_the_same_sum_based_unit():
    mon = CriticalityMonitor(k_max=4)
    mon.push_zones({"PFC": torch.tensor([1.0, 2.0, 0.0, 3.0])})
    assert list(mon.zone_activity["PFC"]) == [6.0], "очередь автокорреляции хранит СУММУ спайков"
    assert mon.zone_activity_mean["PFC"] == pytest.approx(6.0), (
        "статистика активности обязана быть в той же единице, что и очередь (сумма, не среднее по нейронам)")
    assert mon.zone_activity_mean["PFC"] != pytest.approx(1.5)


def test_constant_activity_does_not_become_a_criticality_measurement():
    mon = CriticalityMonitor(k_max=4)
    for _ in range(12):
        mon.push(torch.tensor([3.0]), torch.tensor([3.0]))
        mon.push_zones({"PFC": torch.tensor([3.0])})
    assert mon.sigma_hat_ready is False
    assert mon.zone_sigma_ready["PFC"] is False


def test_global_sigma_hat_is_a_measurement_and_keeps_its_target_separate():
    mon = CriticalityMonitor(k_max=20)
    for v in _ar1_series(n=300):
        mon.push(torch.tensor([float(v)]), torch.tensor([0.0]))
    expected = mon._mr_estimate(mon.activity)
    assert expected is not None
    assert mon.sigma_hat == pytest.approx(expected)
    assert mon.target == pytest.approx(0.97), "цель σ≈1 обязана быть отдельной величиной"


# ---------------------------------------------------------------------------
# Пункт 6 — контроллер σ̂-регуляции: рабочий, переносимый, единственный владелец
# ---------------------------------------------------------------------------

def test_controller_regulates_a_real_moe_style_cortex():
    ctrl = HomeostaticController(HomeostasisCfg())
    col = _tiny_column("SENS")
    cortex = _Cortex([col])
    u_before = float(col.w_ff.u_syn.mean())
    theta_before = float(col.pyr.theta.mean())
    rep = ctrl.regulate(cortex, sigma=1.5)
    assert rep["sigma"] == pytest.approx(1.5)
    assert rep["depression"] == 1.0, "перекритичность обязана вызывать депрессию"
    assert float(col.w_ff.u_syn.mean()) < u_before
    assert float(col.pyr.theta.mean()) > theta_before
    assert rep["sigma_target"] == pytest.approx(ctrl.cfg.target_sigma)


def test_controller_potentiates_when_sigma_is_subcritical():
    ctrl = HomeostaticController(HomeostasisCfg())
    col = _tiny_column("PFC")
    theta_before = float(col.pyr.theta.mean())
    pv_before = float(col.w_pv_in.mean())
    rep = ctrl.regulate([col], sigma=0.5)
    assert rep["potentiation"] == 1.0
    # субкритичность → порог снижается (больше спайков), торможение PV ослабевает
    assert float(col.pyr.theta.mean()) < theta_before
    assert float(col.w_pv_in.mean()) < pv_before
    # ресурс u_syn ограничен сверху единицей (он и так насыщен) — регулятор не обязан его двигать
    assert float(col.w_ff.u_syn.mean()) <= 1.0
    assert float(col.pyr.met_gain) < 1.0, "субкритичность → метаболический тормоз ослабевает"


def test_controller_takes_sigma_from_the_monitor_when_not_given():
    ctrl = HomeostaticController(HomeostasisCfg())
    col = _tiny_column("SENS")
    for i in range(60):
        ctrl.observe(torch.tensor([float(i % 7)]))
    rep = ctrl.regulate([col], sigma=None)
    assert rep["sigma"] == pytest.approx(ctrl.monitor.sigma_hat)


def test_controller_does_not_overwrite_the_monitor_measurement():
    ctrl = HomeostaticController(HomeostasisCfg())
    col = _tiny_column("SENS")
    ctrl.observe_zones({"SENS": torch.tensor([4.0, 1.0, 0.0, 2.0])})
    measured_before = ctrl.monitor.zone_sigmas.get("SENS")
    ctrl.regulate([col], sigma=1.5)
    assert ctrl.monitor.zone_sigmas.get("SENS") == measured_before, (
        "измерение не должно подменяться управляющей переменной")
    assert ctrl.zone_effective_sigma.get("SENS") is not None


def test_controller_accepts_legacy_hierarchy_attribute_cols():
    ctrl = HomeostaticController(HomeostasisCfg())
    col = _tiny_column("SENS")

    class _Legacy:
        cols = [col]

    rep = ctrl.regulate(_Legacy(), sigma=1.5)
    assert rep["depression"] == 1.0
