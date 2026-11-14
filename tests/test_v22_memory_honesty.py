"""v22 — честность памяти: восемь дефектов, каждый с причинным тестом.

Правило файла: тест проверяет ПРИЧИНУ, а не константу. Каждый тест падал (или
давал неверное значение) на коде до правки 2026-11-13 и проходит после неё.
Никаких сравнений с литералом ради прохождения: измеряется поведение
(достраивание паттерна, сохранность выученной строки, доля реального реплея,
реакция на выключатель).

Запуск:  python -m pytest tests/test_v22_memory_honesty.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ziwago.config import HippocampusCfg
from ziwago.memory.hippocampus import Hippocampus, SBCVSAShardedHippocampus
from ziwago.memory.dual_hippocampus import DualHippocampus
from ziwago.memory.laplace import LaplaceBank
from ziwago.memory.phase_vsa import ThetaGammaStoryBinder
from ziwago.memory.tcc import ToroidalRelationAttractor
from ziwago.memory.working import SilentWorkingMemory


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def tiny_cfg(**kw) -> HippocampusCfg:
    base = dict(n_ec=16, n_dg=48, n_ca3=64)
    base.update(kw)
    return HippocampusCfg(**base)


def ca3_of(h: Hippocampus, ec: torch.Tensor) -> torch.Tensor:
    """CA3-паттерн из EC-входа (ключ для ca3_retrieve должен быть размером n_ca3)."""
    return torch.relu(h.dg_code(ec) @ h.w_dg_ca3.t())


def bipolar(n: int, seed: int, p_one: float = 0.5) -> torch.Tensor:
    """Биполярный паттерн ±1 в форме [1, N] (формат ca3_store)."""
    g = torch.Generator().manual_seed(seed)
    return torch.where(torch.rand(1, n, generator=g) < p_one,
                       torch.ones(1, n), -torch.ones(1, n))


def as_bits(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).float()


def hamming(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((as_bits(a) != as_bits(b)).float().mean().item())


def flip_fraction(x: torch.Tensor, frac: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    out = x.clone()
    idx = torch.randperm(x.numel(), generator=g)[: max(1, int(frac * x.numel()))]
    flat = out.flatten()
    flat[idx] = -flat[idx]
    return flat.view_as(out)


# ===========================================================================
# 1. CA3-Хопфилд: обучение должно накапливаться, retrieval — достраивать
# ===========================================================================
def test_ca3_pattern_completion_restores_stored_pattern():
    """Зашумлённый на 25% ключ должен вернуть ИСХОДНЫЙ паттерн, а не копию ключа."""
    cfg = tiny_cfg()
    h = Hippocampus(cfg, seed=0)
    a = bipolar(cfg.n_ca3, seed=11)
    b = bipolar(cfg.n_ca3, seed=12)
    c = bipolar(cfg.n_ca3, seed=13)
    for pat in (a, b, c):
        h.ca3_store(pat)

    cue = flip_fraction(a, 0.25, seed=99)
    out = h.ca3_retrieve(cue)

    assert hamming(out, a) == 0.0, "паттерн не достроен: сеть не исправила шум"
    assert hamming(out, cue) > 0.2, "выход совпал с ключом — это копирование, не аттрактор"
    assert hamming(out, b) > 0.4 and hamming(out, c) > 0.4


def test_ca3_association_recall_returns_partner_not_cue():
    """Пара (A→B): по признаку A извлекается B, а не A и не sign(A)."""
    cfg = tiny_cfg()
    h = Hippocampus(cfg, seed=1)
    a = bipolar(cfg.n_ca3, seed=21)
    b = bipolar(cfg.n_ca3, seed=22)
    h.ca3_store_pair(a, b)

    out = h.ca3_retrieve(a)
    assert hamming(out, b) == 0.0, "ассоциация A→B не извлекла B"
    assert hamming(out, a) > 0.4, "вернулся сам ключ A (автоассоциация вместо ассоциации)"

    # частичный ключ тоже должен вести к B
    out_noisy = h.ca3_retrieve(flip_fraction(a, 0.2, seed=77))
    assert hamming(out_noisy, b) <= 0.1


def test_ca3_recurrent_drive_dominates_cue_bias():
    """Метрика живости: вклад обученной матрицы обязан превышать зажим 0.25*cue."""
    cfg = tiny_cfg()
    h = Hippocampus(cfg, seed=2)
    for seed in (31, 32, 33):
        h.ca3_store(bipolar(cfg.n_ca3, seed=seed))

    h.ca3_retrieve(bipolar(cfg.n_ca3, seed=31))
    m = h.ca3_metrics()
    assert m["stored_patterns"] == 3
    assert m["recurrent_drive_rms"] > 3.0 * m["cue_bias_rms"]
    assert m["w_rms"] > 0.0


# ===========================================================================
# 2. Нейрогенез: засеянный генератор и сохранность выученных строк
# ===========================================================================
def test_neurogenesis_uses_seeded_generator():
    cfg = tiny_cfg()
    h1 = Hippocampus(cfg, seed=3)
    h2 = Hippocampus(cfg, seed=3)
    before = h1.w_ec_dg.clone()

    torch.manual_seed(1)
    n1 = h1.neurogenesis(novelty_level=1.0, bdnf=1.0)
    torch.manual_seed(2)
    n2 = h2.neurogenesis(novelty_level=1.0, bdnf=1.0)

    assert n1 == n2 and n1 >= 1
    assert not torch.equal(before, h1.w_ec_dg), "нейрогенез не тронул матрицу"
    assert torch.equal(h1.w_ec_dg, h2.w_ec_dg), "нейрогенез зависит от глобального RNG"


def test_neurogenesis_does_not_erase_used_granule():
    """Самая старая, но реально используемая гранула не должна затираться."""
    cfg = tiny_cfg()
    h = Hippocampus(cfg, seed=4)
    row = int(h.dg_usage.argmin().item())          # наименее используемая
    h.w_ec_dg[row] = torch.linspace(-1.0, 1.0, cfg.n_ec)
    h.dg_age[row] = 10_000.0                       # при этом самая старая
    h.dg_usage[row] = 1_000.0                      # но выученная и нужная
    learned = h.w_ec_dg[row].clone()

    n_new = h.neurogenesis(novelty_level=1.0, bdnf=1.0)
    assert n_new >= 1
    assert torch.equal(h.w_ec_dg[row], learned), "выученная EC→DG-строка затёрта шумом"


def test_dg_usage_counts_real_activation():
    cfg = tiny_cfg()
    h = Hippocampus(cfg, seed=5)
    x = torch.randn(1, cfg.n_ec, generator=torch.Generator().manual_seed(6))
    dg = h.dg_code(h.encode_ec(x))
    assert float(h.dg_usage.sum()) == pytest.approx(float(dg.sum()), abs=1e-6)
    assert int((h.dg_usage > 0).sum()) == int((dg.sum(0) > 0).sum())


# ===========================================================================
# 3. Метрики: измерение вместо константы
# ===========================================================================
def _dht_with_episodes(n: int = 3):
    dht = SBCVSAShardedHippocampus(n_shards=4, n_blocks=16, block_len=64,
                                   shard_capacity=64, seed=7)
    all_codes = []
    for i in range(n):
        vec = torch.randn(32, generator=torch.Generator().manual_seed(100 + i))
        slots = [dht.vector_to_block_code(vec * (1.0 + 0.01 * s), shard_idx=s)
                 for s in range(4)]
        dht.store_composite_episode(slots, salience=1.0, ca1_mismatch=0.2)
        all_codes.append(slots)
    return dht, all_codes


def test_sequence_retrieval_accuracy_is_not_invented():
    """Ground truth для «правильной последовательности» в этой функции нет —
    честный ответ `None` + флаг, а не литеральная константа 0.96."""
    dht, codes = _dht_with_episodes(3)
    out = dht.route_contextual_replay(cue_modality="temporal_sequence",
                                      query_slot_codes=codes[0])
    assert out["sequence_retrieval_accuracy"] is None
    assert out["sequence_retrieval_measurement"] == "unavailable:no_ground_truth"
    # зато целостность адреса при реплее измерена
    assert out["replay_unbind_n"] > 0
    assert out["replay_unbind_fidelity"] == pytest.approx(1.0)


def test_replay_unbind_fidelity_drops_when_replay_content_corrupted():
    """Порча ВСЕХ сохранённых ключей обязана обнулить метрику целостности."""
    dht, codes = _dht_with_episodes(3)
    for ep in list(dht.episodes_vsa):
        bound, sal = dht.episodes_vsa[ep]
        dht.episodes_vsa[ep] = (
            tuple(tuple((c + 1) % dht.block_len for c in code) for code in bound), sal)

    out = dht.route_contextual_replay(cue_modality="temporal_sequence",
                                      query_slot_codes=codes[0])
    assert out["replay_unbind_n"] > 0
    assert out["replay_unbind_fidelity"] == 0.0


def test_silent_period_saving_is_measured_from_inspected_records():
    """Экономия — доля реально просмотренных записей индекса, а не 1−n_replay/n_stored."""
    small, small_codes = _dht_with_episodes(1)
    r1 = small.route_contextual_replay(query_slot_codes=small_codes[0])
    assert r1["silent_compute_records_inspected"] > 0
    assert (r1["silent_compute_records_full_sweep"]
            == small.index_size())
    assert r1["silent_period_compute_saving"] == pytest.approx(0.0)  # сокращать нечего

    big, big_codes = _dht_with_episodes(20)
    r2 = big.route_contextual_replay(query_slot_codes=big_codes[0])
    assert r2["silent_compute_records_full_sweep"] > 5 * r1["silent_compute_records_full_sweep"]
    assert r2["silent_compute_records_inspected"] < r2["silent_compute_records_full_sweep"]
    assert r2["silent_period_compute_saving"] > 0.8

    # без запроса извлекать нечего — метрика честно недоступна, а не «100% экономии»
    r3 = big.route_contextual_replay()
    assert r3["silent_period_compute_saving"] is None


# ===========================================================================
# 4. Точка врезки сна + аденозин в step()
# ===========================================================================
def test_adenosine_gates_encoding_in_step():
    cfg = tiny_cfg()
    h = Hippocampus(cfg, seed=8)
    x = torch.randn(1, cfg.n_ec, generator=torch.Generator().manual_seed(9))
    awake = h.step(x, ach=0.8, adenosine=0.0, learn=False)
    asleep = h.step(x, ach=0.8, adenosine=0.9, learn=False)
    assert awake.encode_gain > asleep.encode_gain
    assert asleep.adenosine == pytest.approx(0.9)
    # гейт применён к тому паттерну, который реально входит в DG/CA3
    assert float(asleep.ec_gated.abs().mean()) < float(awake.ec_gated.abs().mean())
    # и это видит CA1-компаратор (отклик памяти, а не только масштаб входа)
    assert asleep.mismatch != awake.mismatch


def test_sleep_cycle_swr_requires_adenosine_gate():
    cfg = tiny_cfg()
    h = Hippocampus(cfg, seed=10)
    for seed in (41, 42, 43):
        ec = torch.randn(1, cfg.n_ec, generator=torch.Generator().manual_seed(seed))
        h.store_episode(ec, h.dg_code(ec), ca3_of(h, ec), error=0.6,
                        uncertainty=0.3)
    writes: list[float] = []
    shallow = h.sleep_cycle(lambda ec, err: writes.append(float(err)),
                            adenosine=0.10, ach=0.2, rem=False)
    assert shallow["swr_gate_open"] is False
    assert shallow["replayed"] == [] and writes == []

    deep = h.sleep_cycle(lambda ec, err: writes.append(float(err)),
                         adenosine=0.90, ach=0.2, rem=False)
    assert deep["swr_gate_open"] is True
    assert len(deep["replayed"]) >= 1 and len(writes) >= 1
    assert deep["rem"] == {}, "REM не запрошен — фаза не должна выполняться"


def test_sleep_cycle_rem_runs_only_with_enough_episodes():
    cfg = tiny_cfg()
    h = Hippocampus(cfg, seed=11)
    for seed in (51, 52):
        ec = torch.randn(1, cfg.n_ec, generator=torch.Generator().manual_seed(seed))
        h.store_episode(ec, h.dg_code(ec), ca3_of(h, ec), error=0.9,
                        uncertainty=0.4)
    out = h.sleep_cycle(None, adenosine=0.9, ach=0.1, rem=True)
    assert set(out["rem"]) >= {"semantic_bridges", "rem_chimera_similarity"}
    assert out["sleep_cycle"] == 1


# ===========================================================================
# 5. dual_hippocampus: четыре модальности и θ-гейт реколла
# ===========================================================================
def test_four_shard_codes_come_from_four_modalities():
    cfg = tiny_cfg()
    h = Hippocampus(cfg, seed=12)
    ec = torch.randn(1, cfg.n_ec, generator=torch.Generator().manual_seed(13))
    h.step(ec, ach=0.8, adenosine=0.1, learn=False)
    base = h.modality_codes(ec)

    other_dg = torch.zeros(1, cfg.n_dg)
    other_dg[0, :4] = 1.0
    other_ca3 = torch.zeros(1, cfg.n_ca3)
    other_ca3[0, :4] = 1.0
    swapped = h.modality_codes(ec, dg=other_dg, ca3=other_ca3)
    assert swapped[0] == base[0], "EC-код не должен зависеть от dg/ca3"
    assert swapped[1] != base[1]
    assert swapped[2] != base[2]

    # контекстный шард обязан реагировать на историю, а не на сам ec
    before = base[3]
    h._context_ema = torch.zeros(cfg.n_ec)
    after = h.modality_codes(ec)[3]
    assert after != before, "временной шард — копия EC, а не контекст"


def test_exact_query_scores_all_four_shards():
    cfg = tiny_cfg()
    h = Hippocampus(cfg, seed=14)
    ec = torch.randn(1, cfg.n_ec, generator=torch.Generator().manual_seed(15))
    h.step(ec, ach=0.8, adenosine=0.1, learn=False)
    h.store_episode(ec, h.dg_code(ec), ca3_of(h, ec), error=0.5, uncertainty=0.2)

    res = h.vsa_dht.retrieve_multiprobe(h.modality_codes(ec), top_k=3)
    assert res["top_episode_ids"][0] == 0
    # эпизод обязан подтвердиться ВСЕМИ четырьмя шардами (раньше — только двумя,
    # потому что три кода строились из одного вектора ec)
    assert res["top_shard_hits"][0] == 4
    assert res["top_scores"][0] > 0.9


def test_theta_gate_allows_one_recall_per_cycle():
    cfg = tiny_cfg()
    dh = DualHippocampus(cfg, input_dim=cfg.n_ec, seed=16,
                         theta_mode="token", theta_cycle_steps=8.0)
    x = torch.randn(1, cfg.n_ec, generator=torch.Generator().manual_seed(17))
    out = dh.tick(x, ach=0.8, adenosine=0.1)
    dh.store_episode(out.hippo.ec, out.hippo.dg, out.hippo.ca3, error=0.5, uncertainty=0.2)

    recalls = [len(o.recalled_episode_ids) > 0
               for o in (dh.tick(x, ach=0.8, adenosine=0.1) for _ in range(8))]
    assert sum(recalls) == 1, f"θ-гейт не работает: реколлов {sum(recalls)} за цикл"

    # гейт закрыт и сразу после реколла, и вне фазы извлечения
    blocked = dh.verify(x, theta_gate=True)
    assert blocked["theta_gate"]["allowed"] is False
    assert blocked["recalled_episode_ids"] == []

    # два полных цикла — ровно два реколла, а не 16
    twice = [len(o.recalled_episode_ids) > 0
             for o in (dh.tick(x, ach=0.8, adenosine=0.1) for _ in range(16))]
    assert sum(twice) == 2


# ===========================================================================
# 6. LaplaceBank без reset_state
# ===========================================================================
def test_laplace_step_initialises_state():
    bank = LaplaceBank(feature_dim=8)
    state = bank.step(torch.ones(1, 8))
    assert tuple(state.shape) == (bank.num_scales, 1, 8)
    assert float(bank.slow_profile().abs().sum()) > 0.0


def test_laplace_readout_without_state_raises_clear_error():
    bank = LaplaceBank(feature_dim=8)
    with pytest.raises(RuntimeError, match="пуст"):
        bank.slow_profile()
    with pytest.raises(RuntimeError, match="пуст"):
        bank.time_cells()


# ===========================================================================
# 7. Рабочая память: чтение по фацилитации, ровно k, NMDA-удержание
# ===========================================================================
def test_empty_probe_does_not_echo_last_write():
    wm = SilentWorkingMemory(n=64)
    a = torch.zeros(1, 64)
    a[0, :8] = 1.0
    wm.write(a)
    out = wm.probe(torch.zeros(1, 64))
    assert float(out.reactivated.sum()) == 0.0, "пустой зонд вернул эхо последней записи"


def test_kwta_returns_exactly_k_units_on_a_tie():
    wm = SilentWorkingMemory(n=16)
    wm.facil = torch.ones(16, 16)          # полная ничья по всем откликам
    out = wm.probe(torch.ones(1, 16), k=4)
    assert float(out.reactivated.sum()) == 4.0


def test_kwta_is_at_most_k_on_a_real_pattern():
    """На реальном паттерне активных ровно столько, сколько есть отклика (≤ k)."""
    wm = SilentWorkingMemory(n=32)
    p = torch.zeros(1, 32)
    p[0, :3] = 1.0                         # запомнили 3 единицы
    wm.write(p)
    out = wm.probe(p, k=8)
    assert out.active_count == 3, "k-WTA выдумал активность там, где её нет"


def test_ca3_accepts_one_dimensional_patterns_twice():
    """1-D ключ (естественная форма паттерна) не должен ломаться на втором вызове."""
    cfg = tiny_cfg()
    h = Hippocampus(cfg, seed=17)
    a = bipolar(cfg.n_ca3, seed=61).squeeze(0)
    b = bipolar(cfg.n_ca3, seed=62).squeeze(0)
    c = bipolar(cfg.n_ca3, seed=63).squeeze(0)
    d = bipolar(cfg.n_ca3, seed=64).squeeze(0)
    h.ca3_store(a)
    h.ca3_store(b)
    assert h.n_stored == 2
    h.ca3_store_pair(c, d)
    h.ca3_store_pair(a, b)
    assert hamming(h.ca3_retrieve(c), d) == 0.0


def test_nmda_hold_sustains_bump_beyond_passive_decay():
    wm = SilentWorkingMemory(n=64)
    a = torch.zeros(1, 64)
    a[0, :8] = 1.0
    wm.write(a)
    held = wm.bump.clone()

    wm_passive = SilentWorkingMemory(n=64)
    wm_passive.write(a)
    for _ in range(200):
        wm_passive.tick(dt=1.0, active_hold=False)

    for _ in range(200):
        wm.tick(dt=1.0, active_hold=True)

    assert float(wm_passive.bump.sum()) < 0.2 * float(held.sum()), "пассивное затухание не работает"
    assert float(wm.bump.sum()) > 0.9 * float(held.sum()), "NMDA-бамп не удержан рекуррентностью"


# ===========================================================================
# 8. phase_vsa: θ/γ-часы, winnerless competition, device, возраст слотов
# ===========================================================================
def _story_binder(**kw) -> ThetaGammaStoryBinder:
    params = dict(dim=16, n_slots=7, theta_hz=8.0, gamma_hz=40.0, seed=0)
    params.update(kw)
    return ThetaGammaStoryBinder(**params)


def test_theta_clock_drives_phase_and_gamma_clock_is_reported():
    x = torch.zeros(1, 16)
    x[0, :4] = 1.0
    slow = _story_binder(theta_hz=8.0, gamma_hz=40.0)
    fast = _story_binder(theta_hz=32.0, gamma_hz=40.0)
    ps = [slow.step(x).theta_phase for _ in range(5)]
    pf = [fast.step(x).theta_phase for _ in range(5)]
    assert ps != pf, "theta_hz не влияет на фазу — часы не подключены"

    g_slow = _story_binder(theta_hz=8.0, gamma_hz=40.0)
    g_fast = _story_binder(theta_hz=8.0, gamma_hz=160.0)
    gs = [g_slow.step(x).gamma_index for _ in range(40)]
    gf = [g_fast.step(x).gamma_index for _ in range(40)]
    assert gs != gf, "gamma_hz не влияет ни на что"
    # и не только на телеметрию: отношение γ/θ задаёт глубину конкуренции в слоте
    assert not torch.allclose(g_slow.role_state, g_fast.role_state)


def test_silent_input_has_no_role_winner():
    b = _story_binder()
    out = b.step(torch.zeros(1, 16))
    assert out.gamma_slot == -1, "роль назначена без стимула — это таблица, не конкуренция"
    assert float(b.slots.abs().sum()) == 0.0


def test_role_populations_coexist_before_winner():
    b = _story_binder()
    cue = torch.zeros(1, 16)
    cue[0, :6] = 1.0
    b.learn_role(cue, 2)
    b.clause_pos = 1
    b.role_state = torch.zeros(b.n_slots)
    b.role_state[2] = 0.6
    b.role_state[3] = 0.5
    out = b.step(cue, role_hint=torch.tensor([0.0, 0.0, 1.0, 0.98, 0.0, 0.0, 0.0]))
    assert float(b.role_state[3]) > 0.25, "проигравшая популяция раздавлена — это argmax"
    assert out.gamma_slot in (2, 3)


def test_learn_role_accepts_other_dtype():
    """device-баг: w_lex_role не приводился к устройству/типу входа (CUDA падала)."""
    b = _story_binder()
    x = torch.rand(1, 16, dtype=torch.float64)
    b.learn_role(x, 2)                      # не должно бросать
    assert b.w_lex_role.dtype == torch.float64


def test_slots_decay_depends_on_age():
    """slot_age реально управляет скоростью забывания слота (старый — быстрее)."""
    def make() -> ThetaGammaStoryBinder:
        b = _story_binder(n_slots=7)
        cue = torch.zeros(1, 16)
        cue[0, :4] = 1.0
        b.step(cue)
        return b

    young, old = make(), make()
    slot = int(young.slots.abs().sum(1).argmax().item())
    old.slot_age[slot] = 200.0            # тот же слот, но уже старый
    for _ in range(10):
        young.step(torch.zeros(1, 16))
        old.step(torch.zeros(1, 16))
    assert float(old.slots[slot].abs().sum()) < 0.7 * float(young.slots[slot].abs().sum())
    assert float(old.slot_age[slot]) > 200.0


# ===========================================================================
# 9. tcc: амплитуда ядра и выключатель
# ===========================================================================
def test_tcc_kernel_modes_match_declared_weights():
    """Обе моды ядра: DC даёт ровно −w_inh (среднее поле), первая гармоника — ровно w_exc."""
    tcc = ToroidalRelationAttractor(grid_size=24, d_vec=8, w_inh=3.0, w_exc=2.5,
                                    enabled=True, seed=0)
    tcc.activity = torch.ones(24, 24)
    assert float(tcc.recurrent_drive().median()) == pytest.approx(-3.0, rel=1e-6)

    profile = torch.cos(tcc.phi - 0.7)[:, None].repeat(1, tcc.grid_size)
    tcc.activity = profile.clone()
    drive = tcc.recurrent_drive()
    mask = profile.abs() > 0.9
    ratio = drive[mask] / profile[mask]
    assert float(ratio.median()) == pytest.approx(tcc.w_exc, rel=1e-4)


def test_tcc_disabled_gate_freezes_dynamics():
    ext = torch.ones(24, 24) * 5.0
    off = ToroidalRelationAttractor(grid_size=24, d_vec=8, enabled=False, seed=0)
    off.step(external=ext, steps=3)
    assert float(off.activity.abs().sum()) == 0.0, "enabled=False не выключает динамику"

    on = ToroidalRelationAttractor(grid_size=24, d_vec=8, enabled=True, seed=0)
    on.step(external=ext, steps=3)
    assert float(on.activity.abs().sum()) > 0.0
