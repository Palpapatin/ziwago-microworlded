"""v22 — контракты честности модулей сознания/таламуса (трек C).

Каждый тест на ИСХОДНОМ коде (до правок) красный: он либо падает с
AttributeError/KeyError (несуществующий публичный контракт), либо ловит
фиктивное значение (константа, эхо-аргумент, число вне шкалы, «притворство»
в докстринге).

Покрытие:
  1. AST-приоритет конкурирует и доходит до TRN с правильным знаком/масштабом;
  2. гейты MetaHead потребляемы (публичный контракт apply_gates);
  3. третий выход MetaHead (новизна) обучается И потребляется; m_ratio → калибровка;
  4. GW: порог зависит от NE, энергия монотонна, история энергий полная,
     ignition согласован с активациями слотов;
  5. FO-реле: явный «граница фразы» + честная телеметрия, spindle() не мёртв;
  6. TRN: формула мультипликативна и конкурентна (докстринг == код);
  7. SON: реальный индекс выбранной траектории + минимальный D1 Go-гейт;
  8. BA44: мост обучается на настоящих парах переходов и влияет на выбор;
  9. consciousness/metrics.py — строгий: без эха, LZ с возмущением, «нет данных».
"""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path


import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ziwago.cortex.next16_core import ConsciousnessMetricsDashboard
from ziwago.consciousness.attention_schema import AttentionSchema
from ziwago.consciousness.meta_head import MetaHead
from ziwago.consciousness.workspace import GlobalWorkspace
from ziwago.cognition.language.syntax import BA44AttractorSyntaxBridge
from ziwago.config import AttentionSchemaCfg, MetaHeadCfg, WorkspaceCfg
from ziwago.cortex.son_reflection import SONReflectiveColumn
from ziwago.thalamus.relay import ThalamusFO
from ziwago.thalamus.trn import TRN

N_MODULES = 5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ast_profile(peak_module: int, n_modules: int = N_MODULES):
    """AST с заранее заданным профилем внимания (фокус = peak_module)."""
    ast = AttentionSchema(AttentionSchemaCfg(d_schema=32, n_modules=n_modules), seed=7)
    with torch.no_grad():
        ast.w_pred.zero_()
        ast.w_pred[peak_module] = 1.0
    return ast.step(novelty=0.4, error=0.2, drive=0.3,
                    actual_gain=0.5, actual_focus=peak_module, learn=False)


def _trn_steady(n_sectors: int = N_MODULES, drive_value: float = 0.6) -> TRN:
    """TRN в установившемся состоянии (без аст-приоритета)."""
    trn = TRN(n_sectors)
    drive = torch.full((n_sectors,), float(drive_value))
    for _ in range(25):
        trn.step(drive, None, vigilance=1.0)
    return trn


def _random_unit(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


# ---------------------------------------------------------------------------
# 1. AST-приоритет конкурирует; TRN получает разный вклад
# ---------------------------------------------------------------------------
def test_ast_gain_is_competitive_priority_not_flat_softmax():
    out = _ast_profile(2)
    gain = out.gain
    assert gain.shape == (N_MODULES,)
    assert int(gain.argmax()) == 2
    # На исходном коде gain = softmax(logits)*alpha_ast ≈ 0.02 на модуль:
    assert float(gain.max()) >= 0.9
    assert float(gain.max() - gain.min()) > 0.5
    assert 0.0 <= float(gain.min()) and float(gain.max()) <= 1.0


def test_trn_gate_competes_by_ast_priority_and_differs_between_profiles():
    out_a = _ast_profile(2)
    out_b = _ast_profile(4)
    drive = torch.full((N_MODULES,), 0.6)

    trn_a = _trn_steady()
    gate_a = trn_a.step(drive, out_a.gain, vigilance=1.0).clone()
    trn_b = _trn_steady()
    gate_b = trn_b.step(drive, out_b.gain, vigilance=1.0).clone()

    # Победитель внимания должен РЕАЛЬНО доминировать (конкуренция, а не константа):
    assert float(gate_a[2]) > 2.0 * float(gate_a[4])
    assert float(gate_b[4]) > 2.0 * float(gate_b[2])
    # Разные профили внимания ⇒ разный (не равный) вклад в TRN:
    assert not torch.allclose(gate_a, gate_b)
    assert float((gate_a - gate_b).abs().max()) > 0.05


def test_trn_gate_is_scale_invariant_to_global_ast_priority_scale():
    """Форма гейна g_k = g_L6·g_AST·(1−lateral) ⇒ важен ПРОФИЛЬ приоритетов.

    На исходном коде множитель `0.5 + ast_priority` зависит от абсолютного
    масштаба ⇒ удвоение приоритета меняет гейт (константа съедает конкуренцию).
    """
    out = _ast_profile(2)
    drive = torch.full((N_MODULES,), 0.6)
    trn_a = _trn_steady()
    gate_a = trn_a.step(drive, out.gain, vigilance=1.0).clone()
    trn_b = _trn_steady()
    gate_b = trn_b.step(drive, out.gain * 2.0, vigilance=1.0).clone()
    assert torch.allclose(gate_a, gate_b, atol=1e-6)


# ---------------------------------------------------------------------------
# 2-3. MetaHead: гейты потребляемы, новизна потребляется, калибровка честно названа
# ---------------------------------------------------------------------------
def _meta_after_steps(n_steps: int = 14, err: float = 0.4, seed: int = 3):
    meta = MetaHead(MetaHeadCfg(d_hidden=32), d_gw=64, seed=seed)
    gw = torch.randn(64, generator=torch.Generator().manual_seed(11)) * 0.5
    out = None
    for _ in range(n_steps):
        out = meta.step(gw, [gw], err, learn=True)
    return meta, out


def test_meta_head_exposes_consumable_gate_contract():
    meta, out = _meta_after_steps()
    app = meta.apply_gates(out, trn_gain=torch.ones(4), beta_gw=8.0, replay_priority=0.5)
    assert app is not None
    assert app.beta_gw == pytest.approx(8.0 + out.gates["beta_gw_boost"])
    assert app.replay_priority == pytest.approx(0.5 + out.gates["replay_priority"])
    assert torch.allclose(app.trn_gain, torch.ones(4) * out.gates["trn_gain"])
    assert app.bg_threshold == out.gates["bg_threshold"]


def test_meta_head_gates_flip_with_confidence_and_reach_consumer():
    meta = MetaHead(MetaHeadCfg(d_hidden=32), d_gw=64, seed=3)
    gw = torch.randn(64, generator=torch.Generator().manual_seed(11)) * 0.5
    out = None
    for _ in range(40):
        out = meta.step(gw, [gw], 0.95, learn=True)  # низкая уверенность
    assert out is not None
    assert out.confidence < meta.cfg.theta_low
    assert out.gates["bg_threshold"] == 1.0
    assert out.gates["beta_gw_boost"] > 0.0
    app = meta.apply_gates(out, trn_gain=torch.ones(4), beta_gw=8.0, replay_priority=0.0)
    assert app.beta_gw > 8.0
    assert app.bg_threshold == 1.0


def test_meta_head_novelty_unit_is_trained_and_consumed():
    meta, out = _meta_after_steps()
    assert hasattr(out, "novelty_unit")
    novelty_unit = float(out.novelty_unit)
    assert 0.0 <= novelty_unit <= 1.0
    # Обучающийся юнит новизны влияет на приоритет реплея (а не висит мёртвым):
    assert out.gates["replay_priority"] == pytest.approx(
        out.epistemic + 0.5 * out.surprise + 0.5 * novelty_unit)
    # Юнит действительно обучается: с обучением и без — разные значения.
    meta_untrained, out_untrained = _meta_after_steps(seed=3)
    meta_untrained.learn_disabled = True
    gw = torch.randn(64, generator=torch.Generator().manual_seed(11)) * 0.5
    for _ in range(14):
        out_untrained = meta_untrained.step(gw, [gw], 0.4, learn=False)
    assert float(out_untrained.novelty_unit) != pytest.approx(novelty_unit)


def test_meta_head_calibration_metric_is_named_by_its_real_meaning():
    meta = MetaHead(MetaHeadCfg(d_hidden=32), d_gw=64, seed=5)
    gw = torch.randn(64, generator=torch.Generator().manual_seed(13)) * 0.5
    out = None
    for i in range(48):
        err = 0.1 if (i // 8) % 2 == 0 else 0.9
        out = meta.step(gw, [gw], err, learn=True)
    assert out is not None
    assert not hasattr(out, "m_ratio")
    assert hasattr(out, "calibration_corr")
    # Уверенность реально отслеживает точность ⇒ корреляция(conf, 1−err) > 0.
    assert float(out.calibration_corr) > 0.3


# ---------------------------------------------------------------------------
# 4. Global Workspace
# ---------------------------------------------------------------------------
def _gw_candidates(dim: int = 32, n: int = 3):
    return [_random_unit(dim, 100 + i) for i in range(n)]


def test_workspace_threshold_depends_on_norepinephrine():
    cands = _gw_candidates()
    gw_low = GlobalWorkspace(WorkspaceCfg(d_gw=32, n_slots=4))
    gw_high = GlobalWorkspace(WorkspaceCfg(d_gw=32, n_slots=4))
    out_low = gw_low.compete(cands, ne=0.0)
    out_high = gw_high.compete(cands, ne=1.0)
    # NE должен сдвигать порог ignition (иначе обещание «[NE] сдвигает порог» ложно):
    assert float(out_high.ignition) > float(out_low.ignition) + 0.01


def test_workspace_energy_is_monotone_and_full_history_is_kept():
    gw = GlobalWorkspace(WorkspaceCfg(d_gw=32, n_slots=4))
    cands = _gw_candidates()
    out = gw.compete(cands, ne=0.5)
    assert getattr(out, "slot_energy_traces", None), "нет следов энергии по шагам"
    for trace in out.slot_energy_traces:
        assert len(trace) >= 2
        for prev, nxt in zip(trace, trace[1:]):
            assert nxt <= prev + 1e-6, "энергия Хопфилда должна не возрастать"
    assert hasattr(gw, "energy_hist")
    assert len(gw.energy_hist) == 1
    gw.compete(cands, ne=0.5)
    gw.compete(cands, ne=0.5)
    assert len(gw.energy_hist) == 3, "история энергий должна храниться полностью"


def test_workspace_ignition_matches_slot_activations():
    gw = GlobalWorkspace(WorkspaceCfg(d_gw=32, n_slots=4))
    out = gw.compete(_gw_candidates(), ne=0.5)
    acts = list(getattr(out, "slot_activations", []))
    assert acts, "нет активаций слотов"
    assert float(out.ignition) == pytest.approx(max(acts), abs=1e-6)
    if out.winner_zone >= 0:
        assert int(max(range(len(acts)), key=lambda i: acts[i])) == int(out.winner_zone)


# ---------------------------------------------------------------------------
# 5. Таламическое FO-реле
# ---------------------------------------------------------------------------
def test_fo_relay_has_explicit_phrase_boundary_and_honest_telemetry():
    fo = ThalamusFO(8)
    x = torch.randn(1, 8)
    # Контроль: без границы фразы при высокой бодрости — тонический режим.
    fo.step(x, ach=0.7, ne=0.7)
    assert fo.mode == "tonic"
    assert fo.telemetry["boundary_burst"] is False

    fo2 = ThalamusFO(8)
    fo2.step(x, ach=0.7, ne=0.7)
    fo2.mark_phrase_boundary()
    fo2.step(x, ach=0.7, ne=0.7)
    assert fo2.mode == "burst", "явная граница фразы обязана давать фазовый маркер"
    assert fo2.telemetry["boundary_burst"] is True
    assert float(fo2.telemetry["b_drive"]) > 0.0
    assert int(fo2.telemetry["boundary_markers"]) == 1


def test_fo_relay_spindle_is_not_dead_code():
    fo = ThalamusFO(4)
    fo.gate = torch.ones(4)
    osc = fo.spindle(1000, 10.0)
    assert osc.shape == (4,)
    assert not torch.allclose(fo.gate, torch.ones(4)), "spindle() должен влиять на гейт"
    assert "spindle" in fo.telemetry


# ---------------------------------------------------------------------------
# 7. SON-рефлексия
# ---------------------------------------------------------------------------
def _son_column() -> SONReflectiveColumn:
    return SONReflectiveColumn(n_son_neurons=32, n_observed_columns=16,
                                d_vec=64, seed=5)


def _son_step(son: SONReflectiveColumn):
    ones = torch.ones(16) * 0.4
    zeros = torch.zeros(16)
    return son.observe_and_reflect_step(
        col_firing_rates=ones, col_somatic_errors=zeros, col_apical_errors=zeros,
        col_gating_weights=ones, candidate_token_vec64=torch.randn(64),
        is_sentence_boundary_punct=False, tokens_generated_so_far=3)


def test_son_reports_real_selected_vte_trajectory_index():
    son = _son_column()
    q = _random_unit(64, 42)
    trajs = [
        ("orth", [1, 2, 3], _random_unit(64, 43), 1.0, 0.5),
        ("anti", [4, 5, 6], -q, 0.2, 0.9),
        ("aligned", [7, 8, 9], q.clone(), 2.0, 0.0),
    ]
    best_idx, logs = son.evaluate_vte_inner_speech_trajectories(q, trajs)
    assert best_idx == 2
    verdict = _son_step(son)
    assert int(verdict.selected_vte_trajectory_idx) == 2


def test_flat_generation_notifies_son_after_overriding_vte_selection():
    source = (ROOT / "ziwago" / "organism" / "flat.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    organism = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ZiwagoPureSynapticDialogueOrganismV14"
    )
    generate = next(
        node for node in organism.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "pure_neural_generate_response"
    )
    override_lines = [
        node.lineno for node in ast.walk(generate)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "best_idx"
                for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "min"
    ]
    notify_lines = [
        node.lineno for node in ast.walk(generate)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "note_vte_selection"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "best_idx"
    ]
    assert len(override_lines) == 1
    assert len(notify_lines) == 1
    assert notify_lines[0] > override_lines[0]


def test_son_has_minimal_dopamine_d1_go_gate():
    son = _son_column()
    q = _random_unit(64, 42)
    trajs = [
        ("orth", [1, 2, 3], _random_unit(64, 43), 1.0, 0.5),
        ("anti", [4, 5, 6], -q, 0.2, 0.9),
        ("aligned", [7, 8, 9], q.clone(), 2.0, 0.0),
    ]
    best_idx, logs = son.evaluate_vte_inner_speech_trajectories(q, trajs)
    assert "d1_go_gate" in logs[0]
    for idx, log in enumerate(logs):
        if idx == best_idx:
            assert float(log["d1_go_gate"]) > 0.0
        else:
            assert float(log["d1_go_gate"]) == 0.0


# ---------------------------------------------------------------------------
# 8. Синтаксический мост BA44
# ---------------------------------------------------------------------------
def _fake_bridge_env():
    vocab = {",": 11, " while": 101, " before": 102, " after": 103,
             " because": 104, " and": 105}
    reverse = {v: k for k, v in vocab.items()}

    def encode_fn(text: str):
        return [vocab[text]] if text in vocab else []

    def decode_fn(tid: int) -> str:
        return reverse.get(int(tid), "")

    vecs = {}

    def get_vec64_fn(tid: int) -> torch.Tensor:
        tid = int(tid)
        if tid not in vecs:
            g = torch.Generator().manual_seed(9000 + tid)
            vecs[tid] = torch.randn(64, generator=g)
        return vecs[tid].clone()

    return vocab, encode_fn, decode_fn, get_vec64_fn


def test_ba44_bridge_learns_on_real_transition_pairs_and_drives_selection():
    vocab, encode_fn, decode_fn, get_vec64_fn = _fake_bridge_env()
    bridge = BA44AttractorSyntaxBridge(d_vec=64, seed=1444)
    bridge.initialize_synaptic_connectors(encode_fn, decode_fn, get_vec64_fn, vocab)

    pole1 = _random_unit(64, 201)
    pole2 = _random_unit(64, 202)
    stim_id = 500
    stim_vec = get_vec64_fn(stim_id)

    before = bridge.w_ba44_bridge.clone()
    for _ in range(6):
        bridge.learn_bridge_transition(pole1, pole2, stim_vec, connector_idx=2)
    after = bridge.w_ba44_bridge
    assert not torch.allclose(before, after), "матрица моста должна обучаться"

    rows = bridge.score_bridges(pole1, pole2, [stim_id], get_vec64_fn)
    assert len(rows) == bridge.n_connectors
    target = [r for r in rows if r["connector_idx"] == 2][0]
    others = [r for r in rows if r["connector_idx"] != 2]
    assert float(target["matrix_drive"]) > 0.5
    assert float(target["matrix_drive"]) > max(float(r["matrix_drive"]) for r in others)

    phrase, token_ids, score = bridge.select_bridge_tokens_between_attractors(
        pole1, pole2, [stim_id], get_vec64_fn)
    assert phrase == ", after"
    assert token_ids == [11, 103]


# ---------------------------------------------------------------------------
# 9. Строгие метрики сознания
# ---------------------------------------------------------------------------
def test_live_metrics_do_not_echo_legacy_inputs_and_report_measured_traces():
    dash = ConsciousnessMetricsDashboard()
    raster = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    rep = dash.compute_turn_consciousness_metrics(
        ca3_resonance=0.7, epistemic_uncertainty=0.2,
        spike_raster=raster, perturbed_spike_raster=1.0 - raster,
        total_columns=2)
    assert "ca3_resonance" not in rep
    assert "epistemic_uncertainty" not in rep
    assert rep["broadcast_reach"] == 1.0
    assert rep["gwt_ignition_score"] == 0.5
    assert rep["iit_pci_complexity_proxy"] is not None


def test_live_metrics_return_unavailable_for_missing_measurements():
    rep = ConsciousnessMetricsDashboard().compute_turn_consciousness_metrics()
    assert rep["gwt_ignition_score"] is None
    assert rep["broadcast_reach"] is None
    assert "gwt_ignition_score" in rep["unavailable_measurements"]
