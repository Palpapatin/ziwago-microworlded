"""Focused, synthetic v19 research-module checks (no main-runtime dependencies)."""

from __future__ import annotations

import copy
import io
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ziwago.cortex.next16_core import (
    ConsciousnessMetricsDashboard,
    CorticalColumnarSubBankRouter,
    CrossDomainIsomorphismDreamer,
    DynamicNeuromodulatoryBCMGate,
    FusiZenkeCascadeMetaplasticityShield,
    KnowledgeDistillery,
    TwoCompartmentLarkumMicrocircuit,
)
from ziwago.cortex.moe_cortex import export_runtime_tree, load_runtime_tree


def checkpoint(module, restored):
    stream = io.BytesIO()
    torch.save(module.export_state_dict(), stream)
    stream.seek(0)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    return restored


def test_missing_knowledge_is_unavailable_without_foundation_fixtures(tmp_path):
    distillery = KnowledgeDistillery(tmp_path / "absent")
    assert not distillery.available
    assert distillery.status == "unavailable"
    assert distillery.extracted_chemical_agents_count == 0
    assert distillery.extracted_records_count == 0
    assert distillery.parsed_files_summary == {}
    assert distillery.get_knowledge_core_lessons() == []
    assert distillery.get_knowledge_world_graph_facts() == []
    assert KnowledgeDistillery().knowledge_dir == ROOT / "data" / "Knowledge"


@pytest.mark.parametrize("text", [
    "| **1. Fakeamine** | receptor |\n",
    "| Agent | Effect |\n| --- | --- |\n| **1. Fakeamine** |\n",
    "| Agent | Effect |\n| -- | --- |\n| **1. Fakeamine** | made up |\n",
    "| Agent | Effect |\n| --- | --- |\n| **1. Fakeamine** | |\n",
    "| Agent | Agent |\n| --- | --- |\n| Fakeamine | Fakeamine |\n",
])
def test_malformed_chemical_tables_do_not_extract(tmp_path, text):
    (tmp_path / "chemical.md").write_text(text, encoding="utf-8")
    distillery = KnowledgeDistillery(tmp_path)
    assert distillery.available
    assert distillery.extracted_chemical_agents_count == 0
    assert distillery.records == []
    assert distillery.extraction_errors


def test_knowledge_content_tables_hashes_and_unique_chemical_count(tmp_path):
    chemical = tmp_path / "chemical.md"
    chemical.write_text(
        "# Novel agents\n| Agent | Effect |\n| --- | --- |\n"
        "| **1. Vexorin** | Opens the amber channel |\n"
        "| **2. Quenolate** | Inhibits the violet pump |\n"
        "| **3. Vexorin** | Stabilizes the amber channel |\n",
        encoding="utf-8",
    )
    (tmp_path / "unfamiliar.txt").write_text(
        "# Unfamiliar mechanism\nZorba activates the frindle relay.\n"
        "The relay transmits the signal.\n\n```\nNot a fact.\n```\n", encoding="utf-8",
    )
    (tmp_path / "agents.csv").write_text("Agent,Effect\nQuenolate,Slows transport\nTormide,Accelerates decay\n", encoding="utf-8")
    distillery = KnowledgeDistillery(tmp_path)
    assert distillery.extracted_chemical_agents_count == 3
    assert len(distillery.records) == 7
    assert all(len(r["source_hash"]) == 64 and len(r["record_id"]) == 64 for r in distillery.records)
    lessons = distillery.get_knowledge_core_lessons()
    facts = distillery.get_knowledge_world_graph_facts()
    assert [lesson[2] for lesson in lessons] == facts
    assert any("Zorba activates" in fact for fact in facts)
    assert not any("glutamate" in fact.lower() or "Not a fact" in fact for fact in facts)
    prior_hash = next(r["source_hash"] for r in distillery.records if r["source"] == "chemical.md")
    chemical.write_text(chemical.read_text(encoding="utf-8").replace("amber", "silver"), encoding="utf-8")
    rescanned = KnowledgeDistillery(tmp_path)
    assert next(r["source_hash"] for r in rescanned.records if r["source"] == "chemical.md") != prior_hash
    lessons.clear()
    assert distillery.get_knowledge_core_lessons()


def test_bad_utf8_and_csv_report_errors_without_claiming_extraction(tmp_path):
    (tmp_path / "chemical.md").write_bytes(b"\xff\xfe")
    (tmp_path / "bad.csv").write_text('Agent,Effect\n"unfinished,entry', encoding="utf-8")
    distillery = KnowledgeDistillery(tmp_path)
    assert distillery.records == []
    assert distillery.extracted_chemical_agents_count == 0
    assert len(distillery.extraction_errors) == 2


def test_bcm_and_larkum_checkpoint_resume_exactly():
    generator = torch.Generator().manual_seed(19)
    gate = DynamicNeuromodulatoryBCMGate(d_sdr=8, tau_bcm=17)
    circuit = TwoCompartmentLarkumMicrocircuit(d_vec=8, theta_ca=0.2)
    for _ in range(4):
        x, y = torch.rand(8, generator=generator), torch.rand(8, generator=generator)
        gate.evaluate_neuromodulatory_gate(x)
        circuit.step_microcircuit(x, y)
    restored_gate = checkpoint(gate, DynamicNeuromodulatoryBCMGate())
    restored_circuit = checkpoint(circuit, TwoCompartmentLarkumMicrocircuit())
    x, y = torch.rand(8, generator=generator), torch.rand(8, generator=generator)
    assert gate.evaluate_neuromodulatory_gate(x) == restored_gate.evaluate_neuromodulatory_gate(x)
    first, second = circuit.step_microcircuit(x, y), restored_circuit.step_microcircuit(x, y)
    for key, value in first.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, second[key], rtol=0, atol=0)
        else:
            assert value == second[key]
    state = gate.export_state_dict()
    state["theta_bcm"].zero_()
    assert gate.theta_bcm.any()


@pytest.mark.parametrize("module", ["bcm", "larkum", "shield"])
def test_state_load_rejects_nonfinite_data_without_mutation(module):
    if module == "bcm":
        instance, key = DynamicNeuromodulatoryBCMGate(d_sdr=4), "theta_bcm"
    elif module == "larkum":
        instance, key = TwoCompartmentLarkumMicrocircuit(d_vec=4), "w_sst_pred"
    else:
        instance, key = FusiZenkeCascadeMetaplasticityShield(d_dend=4, n_cols=2), "u_slow"
        # Трасса щита ленивая: без аллокации вектор пуст и в него нечего писать.
        instance._ensure(torch.zeros(4, 2), 2)
    original = instance.export_state_dict()
    invalid = copy.deepcopy(original)
    invalid[key].flatten()[0] = float("nan")
    with pytest.raises(ValueError):
        instance.load_state_dict(invalid)
    torch.testing.assert_close(instance.export_state_dict()[key], original[key], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_shield_lazy_masked_dense_and_gathered_updates(dtype):
    shield = FusiZenkeCascadeMetaplasticityShield()
    assert shield.n_cols == 0 and shield.u_slow.numel() == 0
    weights = torch.full((4, 6), 0.5, dtype=dtype)
    shield.protect_rows(weights, [1, 3], active_cols=3)
    # Трасса щита — вектор на строку (reinforcement_v2), не матрица на колонку.
    assert shield.u_slow.shape == (4,)
    assert shield.u_slow.dtype == dtype
    assert shield.u_slow[[1, 3]].sum() > 0
    assert shield.u_slow[[0, 2]].sum() == 0
    before = weights.clone()
    delta = torch.ones_like(weights) * 0.1
    prior = shield.u_slow.clone()
    attenuation = 1.0 / (1.0 + shield.kappa_shield * prior)
    shield.apply_metaplasticity_shielded_update(weights, delta, 3)
    assert torch.equal(weights[:, 3:], before[:, 3:])
    expected = before.clone()
    expected[:, :3] = before[:, :3] + delta[:, :3] * attenuation.unsqueeze(1)
    torch.testing.assert_close(weights, expected, rtol=0.03, atol=0.004)
    before = weights.clone()
    prior = shield.u_slow.clone()
    shield.apply_sparse_update(weights, [1, 1], torch.full((2, 3), 0.1, dtype=dtype), 3)
    assert torch.equal(weights[[0, 2, 3]], before[[0, 2, 3]])
    # Затухание берётся по ПРЕДШЕСТВУЮЩЕЙ трассе: событие не гасит само себя.
    expected = before[1, :3] + 0.2 / (1 + shield.kappa_shield * prior[1])
    torch.testing.assert_close(weights[1, :3], expected, rtol=0.03, atol=0.004)
    shield.apply_sparse_update(weights, [], torch.empty((0, 6)), 6)
    assert shield.u_slow.shape == (4,)


def test_shield_freezes_by_reinforcement_history_not_by_weight_magnitude():
    """Щит замораживает по ИСПОЛЬЗОВАНИЮ, а не по величине веса.

    Регрессия на патологию, зафиксированную 2026-11-12: старая трасса
    копила |w|, поэтому крупный вес означал заморозку (у замороженных строк
    медианный |w| был 1.597 против 1.214 у живых), а пустые строки
    оставались пластичными.
    """
    shield = FusiZenkeCascadeMetaplasticityShield(d_dend=2)
    weights = torch.tensor([[10.0, 10.0], [0.01, 0.01]])
    shield._ensure(weights, 2)
    for _ in range(5):
        shield.apply_sparse_update(weights, [1], torch.full((1, 2), 0.1), 2)
    attenuation = 1.0 / (1.0 + shield.kappa_shield * shield.u_slow)
    assert float(shield.u_slow[0]) == 0.0
    assert float(shield.u_slow[1]) > 0.0
    assert float(attenuation[0]) == pytest.approx(1.0, abs=1e-6)
    assert float(attenuation[1]) < 1.0
    # Крупный вес без использования остаётся нетронутым.
    assert torch.equal(weights[0], torch.tensor([10.0, 10.0]))


def test_shield_update_is_not_self_suppressed_and_attenuation_has_floor():
    shield = FusiZenkeCascadeMetaplasticityShield(d_dend=1)
    weights = torch.zeros(1, 4)
    shield._ensure(weights, 4)
    delta = torch.full((1, 4), 0.1)
    shield.apply_sparse_update(weights, [0], delta, 4)
    # Первое событие не гасит само себя.
    torch.testing.assert_close(weights, delta)
    for _ in range(500):
        shield.apply_sparse_update(weights, [0], delta, 4)
    floor = 1.0 / (1.0 + shield.kappa_shield * shield.trace_cap)
    attenuation = 1.0 / (1.0 + shield.kappa_shield * shield.u_slow)
    assert float(shield.u_slow[0]) <= shield.trace_cap + 1e-6
    assert float(attenuation[0]) >= floor - 1e-9
    assert float(attenuation[0]) > 1e-3


def test_shield_sleep_decay_releases_unused_protection():
    shield = FusiZenkeCascadeMetaplasticityShield(d_dend=1)
    weights = torch.zeros(1, 4)
    shield._ensure(weights, 4)
    shield.protect_rows(weights, [0], 4)
    protected = float(shield.u_slow[0])
    assert protected > 0
    # Сон глохнет трассу на долю sleep_decay за цикл (не на всю сразу).
    shield.consolidate_into_slow_reservoir(weights, active_cols=4)
    assert float(shield.u_slow[0]) == pytest.approx(protected * (1.0 - shield.sleep_decay), rel=1e-5)
    for _ in range(200):
        shield.consolidate_into_slow_reservoir(weights, active_cols=4)
    assert float(shield.u_slow[0]) < 1e-3
    attenuation = 1.0 / (1.0 + shield.kappa_shield * shield.u_slow)
    assert float(attenuation[0]) == pytest.approx(1.0, abs=1e-3)


def test_shield_migrates_legacy_magnitude_matrix_to_row_trace():
    legacy_state = {
        "d_dend": 4, "n_cols": 3, "lambda_cons": 0.85, "delta_decay": 0.002,
        "kappa_shield": 18.0, "u_slow": torch.full((4, 3), 5.0),
    }
    restored = FusiZenkeCascadeMetaplasticityShield()
    report = restored.load_state_dict(legacy_state)
    assert report["u_slow_reset"] is True
    assert report["u_slow_layout_from"] == "matrix"
    assert report["u_slow_semantics_from"] == "magnitude_v1"
    assert restored.u_slow.shape == (4,)
    assert float(restored.u_slow.abs().sum()) == 0.0
    # Ленивая миграция: общий путь восстановления ставит атрибут напрямую.
    lazy = FusiZenkeCascadeMetaplasticityShield(d_dend=4, n_cols=3)
    lazy.u_slow = torch.full((4, 3), 5.0)
    lazy.u_slow_semantics = "magnitude_v1"
    lazy._ensure(torch.zeros(4, 3), 3)
    assert lazy.u_slow.shape == (4,)
    assert float(lazy.u_slow.abs().sum()) == 0.0
    assert lazy.last_migration["u_slow_reset"] is True


def test_shield_roundtrip_synthetic_retention_and_validation():
    shield = FusiZenkeCascadeMetaplasticityShield(d_dend=3)
    original = torch.tensor([[1.0, -1.0, 8.0], [0.2, 0.1, 9.0], [0.3, 0.1, 7.0]])
    content = shield.consolidate_into_slow_reservoir(original, active_cols=2, theta_consolidation=25)
    # Консолидация ИЗМЕРЯЕТ долю содержательных элементов и не пишет величину
    # веса в трассу (иначе щит снова стал бы замораживать крупное).
    expected_content = 100.0 * float((original[:, :2].abs() > 0.05).float().mean().item())
    assert content == pytest.approx(expected_content)
    restored = checkpoint(shield, FusiZenkeCascadeMetaplasticityShield())
    protected, unprotected = original.clone(), original.clone()
    for _ in range(20):
        delta = torch.tensor([[-0.05, 0.05]])
        restored.apply_sparse_update(protected, [0], delta, 2)
        unprotected[0, :2].add_(delta[0])
    # История подкреплений ограничивает обучение, но НЕ обнуляет его.
    protected_shift = (protected[0, :2] - original[0, :2]).norm()
    unprotected_shift = (unprotected[0, :2] - original[0, :2]).norm()
    assert float(protected_shift) < float(unprotected_shift)
    assert float(protected_shift) > 0.0
    assert float(restored.u_slow[0]) > 0.0
    assert torch.equal(protected[:, 2], original[:, 2])
    assert torch.equal(protected[1:], original[1:])
    with pytest.raises(ValueError):
        restored.apply_sparse_update(protected, [3], torch.zeros(1, 2), 2)
    with pytest.raises(ValueError):
        restored.apply_sparse_update(protected, [0.5], torch.zeros(1, 2), 2)
    with pytest.raises(ValueError):
        restored.apply_metaplasticity_shielded_update(protected, torch.zeros(3, 1), 2)
    with pytest.raises(ValueError):
        restored.protect_rows(protected, [0], 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_shield_and_stateful_microcircuits():
    weights = torch.ones((8, 5), device="cuda", dtype=torch.float16)
    shield = FusiZenkeCascadeMetaplasticityShield()
    shield.protect_rows(weights, [1, 4], 4)
    shield.apply_sparse_update(weights, [1], torch.ones((1, 4)), 4)
    assert shield.u_slow.device == weights.device and shield.u_slow.dtype == weights.dtype
    restored = checkpoint(shield, FusiZenkeCascadeMetaplasticityShield())
    restored.apply_sparse_update(weights, [4], torch.ones((1, 4)), 4)
    assert restored.u_slow.device == weights.device
    gate = DynamicNeuromodulatoryBCMGate(d_sdr=8)
    gate.evaluate_neuromodulatory_gate(torch.rand(8, device="cuda"))
    circuit = TwoCompartmentLarkumMicrocircuit(d_vec=8)
    assert circuit.step_microcircuit(torch.rand(8, device="cuda"), torch.rand(8, device="cuda"))["u_soma"].is_cuda
    router = CorticalColumnarSubBankRouter()
    pole = torch.randn(64, device="cuda")
    features = torch.tensor([1., 0., 0.5], device="cuda", dtype=torch.float16)
    assert router.learn_sparse(pole, features, 4, [4, 12], all_token_ids=[4, 12],
                               base_logits=torch.zeros(2, device="cuda"), residual_gain=3)["accepted"]
    predicted = router.predict_sparse(pole, features, [4, 12])
    assert predicted.is_cuda and predicted.dtype == features.dtype
    restored_router = checkpoint(router, CorticalColumnarSubBankRouter())
    torch.testing.assert_close(restored_router.predict_sparse(pole, features, [4, 12]), predicted)


def poles():
    return torch.eye(64)[0], torch.eye(64)[1], torch.eye(64)[2]


def test_sparse_banks_measured_conflict_isolation_and_fixed_budget():
    router = CorticalColumnarSubBankRouter(n_banks=2, memory_budget_bytes=12000)
    a, b, c = poles()
    features = torch.tensor([0.0, 1.0, 0.0, 0.5])
    tokens = [7, 11]
    for _ in range(12):
        assert router.learn_sparse(a, features, 7, tokens, all_token_ids=tokens,
                                   base_logits=torch.zeros(2), residual_gain=3)["accepted"]
    old_a = router.predict_sparse(a, features, tokens).clone()
    first = router.learn_sparse(b, features, 11, tokens, all_token_ids=tokens,
                                    base_logits=torch.zeros(2), residual_gain=3)
    assert first["accepted"] and first["allocated_branch"]
    assert first["gradient_conflict"] < -0.9
    assert router.route_pole_to_sub_bank(a) != router.route_pole_to_sub_bank(b)
    for _ in range(12):
        assert router.learn_sparse(b, features, 11, tokens, all_token_ids=tokens,
                                    base_logits=torch.zeros(2), residual_gain=3)["accepted"]
    torch.testing.assert_close(router.predict_sparse(a, features, tokens), old_a, rtol=0, atol=0)
    assert router.predict_sparse(b, features, tokens).argmax().item() == 1
    assert router.allocated_banks == 2
    assert router.memory_usage_bytes <= router.memory_budget_bytes
    distribution = router.compute_bank_distribution(torch.stack([a, a, b]))
    assert sorted(distribution.values()) == [1, 2]
    assert sum(map(len, router.banks)) == 8  # Only nonzero feature/candidate pairs.
    # Put a third pole nearer B, then demand A's opposing label. No unsafe eviction.
    c = b + c * 0.01
    old_b = router.predict_sparse(b, features, tokens).clone()
    rejected = router.learn_sparse(c, features, 7, tokens, all_token_ids=tokens,
                                   base_logits=torch.zeros(2), residual_gain=3)
    assert not rejected["accepted"] and rejected["reason"] == "bank_budget_exhausted"
    torch.testing.assert_close(router.predict_sparse(b, features, tokens), old_b, rtol=0, atol=0)


def test_nonconflicting_banks_share_until_measured_conflict():
    router = CorticalColumnarSubBankRouter(n_banks=4)
    a, b, _ = poles()
    features = torch.tensor([1.0, 0.0])
    assert router.learn_sparse(a, features, 3, [3, 4], all_token_ids=[3, 4],
                               base_logits=torch.zeros(2), residual_gain=3)["accepted"]
    result = router.learn_sparse(b, features, 3, [3, 4], all_token_ids=[3, 4],
                                 base_logits=torch.zeros(2), residual_gain=3)
    assert result["accepted"] and not result["allocated_branch"]
    assert router.allocated_banks == 1
    assert router.route_pole_to_sub_bank(a) == router.route_pole_to_sub_bank(b)


def test_bank_budget_rejection_is_transactional_and_candidates_are_sparse():
    router = CorticalColumnarSubBankRouter(n_banks=2, memory_budget_bytes=1850, max_active_features=2)
    a, _, _ = poles()
    assert router.learn_sparse(a, torch.tensor([1.0]), 7, [7, 11], all_token_ids=[7, 11],
                               base_logits=torch.zeros(2), residual_gain=3)["accepted"]
    before = router.predict_sparse(a, torch.tensor([1.0]), [7, 11]).clone()
    used = router.memory_usage_bytes
    result = router.learn_sparse(a, torch.tensor([1.0, 1.0, 0.5]), 7, [7, 11, 19],
                                 all_token_ids=[7, 11, 19], base_logits=torch.zeros(3), residual_gain=3)
    assert not result["accepted"] and result["reason"] == "memory_budget_exhausted"
    assert router.memory_usage_bytes == used
    torch.testing.assert_close(router.predict_sparse(a, torch.tensor([1.0]), [7, 11]), before)
    assert router.predict_sparse(a, torch.tensor([1.0]), []).shape == (0,)
    with pytest.raises(ValueError):
        router.learn_sparse(a, torch.tensor([1.0]), 99, [7, 11], all_token_ids=[7, 11],
                            base_logits=torch.zeros(2), residual_gain=3)
    with pytest.raises(ValueError):
        router.predict_sparse(a, torch.tensor([1.0]), [7, 7])


def test_banks_sparse_coo_checkpoint_and_continued_learning():
    gen = torch.Generator().manual_seed(91)
    a, b = torch.randn(64, generator=gen), torch.randn(64, generator=gen)
    sparse = torch.sparse_coo_tensor(torch.tensor([[3, 9000]]), torch.tensor([0.5, 1.0]), (10000,), check_invariants=True)
    router = CorticalColumnarSubBankRouter(n_banks=3, memory_budget_bytes=10000)
    router.learn_sparse(a, sparse, 20001, [20001, 39001], all_token_ids=[20001, 39001],
                        base_logits=torch.zeros(2), residual_gain=3)
    router.learn_sparse(b, sparse, 39001, [20001, 39001], all_token_ids=[20001, 39001],
                        base_logits=torch.zeros(2), residual_gain=3)
    restored = checkpoint(router, CorticalColumnarSubBankRouter())
    for pole in (a, b):
        torch.testing.assert_close(router.predict_sparse(pole, sparse, [20001, 39001]),
                                   restored.predict_sparse(pole, sparse, [20001, 39001]), rtol=0, atol=0)
    kwargs = {"all_token_ids": [20001, 39001], "base_logits": torch.zeros(2), "residual_gain": 3}
    assert router.learn_sparse(b, sparse, 39001, [20001, 39001], **kwargs) == restored.learn_sparse(b, sparse, 39001, [20001, 39001], **kwargs)
    assert restored.memory_usage_bytes == router.memory_usage_bytes
    invalid = restored.export_state_dict()
    invalid["memory_budget_bytes"] = 400
    with pytest.raises(ValueError):
        restored.load_state_dict(invalid)


def assert_bank_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual.keys() - {"domains"}:
        assert actual[key] == expected[key]
    assert actual["domains"].keys() == expected["domains"].keys()
    for key, domain in actual["domains"].items():
        prior = expected["domains"][key]
        assert domain.keys() == prior.keys()
        torch.testing.assert_close(domain["pole"], prior["pole"], rtol=0, atol=0)
        for field in domain.keys() - {"pole"}:
            assert domain[field] == prior[field]


@pytest.mark.parametrize("mask_size", [1, 3])
def test_bank_full_vocabulary_bounded_chain_rule_matches_autograd(mask_size):
    router = CorticalColumnarSubBankRouter(n_banks=2, prune_threshold=0.0)
    pole, _, _ = poles()
    tokens = [7 * i + 3 for i in reversed(range(321))]
    target_position = 289
    target = tokens[target_position]
    mask = [target, tokens[3], tokens[299]][:mask_size]
    features = torch.tensor([0.7, 0., -1.3], dtype=torch.float64, requires_grad=True)
    weights = torch.stack([torch.linspace(-0.8, 0.9, len(tokens), dtype=torch.float64),
                           torch.linspace(0.3, -0.1, len(tokens), dtype=torch.float64)]).requires_grad_()
    router.banks[0] = {(idx, token): float(weights.detach()[row, pos])
                       for row, idx in enumerate([0, 2]) for pos, token in enumerate(tokens)}
    original = dict(router.banks[0])
    base = torch.linspace(-1.5, 1.8, len(tokens), dtype=torch.float64, requires_grad=True)
    scores = base + 3 * (features[[0, 2]] @ weights).tanh()
    loss = scores.logsumexp(0) - scores[target_position]
    expected_gradient, = torch.autograd.grad(loss, weights)
    result = router.learn_sparse(pole, features, target, mask, learning_rate=0.02,
                                 all_token_ids=tokens, base_logits=base, residual_gain=3)
    assert result["accepted"] and not result["allocated_branch"]
    assert result["objective"] == "full_vocabulary_bounded_residual"
    assert result["loss"] == pytest.approx(float(loss.detach()), abs=1e-12)
    assert result["loss_after"] < result["loss"]
    domain = next(iter(router.domains.values()))
    assert len(domain["gradient"]) == 2 * len(mask)
    for row, idx in enumerate([0, 2]):
        for pos, token in enumerate(tokens):
            if token in mask:
                grad = float(expected_gradient[row, pos])
                assert domain["gradient"][(idx, token)] == pytest.approx(grad, abs=1e-12)
                assert router.banks[0][(idx, token)] == pytest.approx(
                    original[(idx, token)] - result["learning_rate"] * grad, abs=1e-12)
            else:
                assert router.banks[0][(idx, token)] == original[(idx, token)]
    actual_scores = base.detach() + 3 * router.predict_sparse(pole, features, tokens).tanh()
    actual_loss = actual_scores.logsumexp(0) - actual_scores[target_position]
    assert result["loss_after"] == pytest.approx(float(actual_loss), abs=1e-12)
    assert features.grad is None and base.grad is None


def test_bank_harmful_split_rejected_against_original_loss_atomically():
    router = CorticalColumnarSubBankRouter(n_banks=2)
    a, b, _ = poles()
    tokens = list(range(300))
    base = torch.full((300,), -20., dtype=torch.float64)
    base[:2] = 0
    features = torch.ones(1, dtype=torch.float64)
    # A conflicting stored gradient need not mean that replacing its bank helps B.
    router.banks[0] = {(0, 0): 2., (0, 1): -2.}
    kwargs = {"all_token_ids": tokens, "base_logits": base, "residual_gain": 3}
    assert router.learn_sparse(a, features, 1, [0, 1], learning_rate=1e-5, **kwargs)["accepted"]
    before = router.export_state_dict()
    rng = torch.random.get_rng_state().clone()
    original_scores = base + 3 * router.predict_sparse(b, features, tokens).tanh()
    preloss = float(original_scores.logsumexp(0) - original_scores[0])
    split_scores = base.clone()
    split_p = split_scores.softmax(0)
    split_p[0] -= 1
    first_weights = -0.1 * 3 * split_p[:2]
    split_scores[:2] += 3 * first_weights.tanh()
    split_loss = float(split_scores.logsumexp(0) - split_scores[0])
    assert split_loss < float(base.logsumexp(0) - base[0])  # Easier split-only objective descends.
    assert split_loss > preloss
    result = router.learn_sparse(b, features, 0, [0, 1], **kwargs)
    assert not result["accepted"] and result["reason"] == "objective_not_improved"
    assert result["gradient_conflict"] < -0.99
    assert result["bank"] == 0 and not result["allocated_branch"]
    assert result["loss"] == pytest.approx(preloss, abs=1e-12)
    assert result["loss_after"] > result["loss"]
    assert result["backtracking_steps"] == 12 and result["learning_rate"] == 0
    assert_bank_state_equal(router.export_state_dict(), before)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert router.banks[1] == {} and router.allocated_banks == 1


def test_bank_accepted_full_vocabulary_split_recomputes_gradient_and_retains_old_bank():
    router = CorticalColumnarSubBankRouter(n_banks=2)
    a, b, _ = poles()
    tokens = list(range(321))
    features = torch.tensor([0., 1., 0.5], dtype=torch.float64)
    base = torch.full((321,), -10., dtype=torch.float64)
    base[:2] = torch.tensor([0.2, -0.1], dtype=torch.float64)
    kwargs = {"all_token_ids": tokens, "base_logits": base, "residual_gain": 3}
    for _ in range(12):
        assert router.learn_sparse(a, features, 1, [0, 1], **kwargs)["accepted"]
    before = router.predict_sparse(a, features, tokens).clone()
    old_weights = dict(router.banks[0])
    original_scores = base + 3 * before.tanh()
    expected_loss = float(original_scores.logsumexp(0) - original_scores[0])
    free_errors = base.softmax(0)
    free_errors[0] -= 1
    result = router.learn_sparse(b, features, 0, [0, 1], **kwargs)
    assert result["accepted"] and result["allocated_branch"]
    assert result["bank"] == 1 and result["gradient_conflict"] < -0.9
    assert result["loss"] == pytest.approx(expected_loss, abs=1e-12)
    assert result["loss_after"] < result["loss"]
    assert router.banks[0] == old_weights and router.banks[0] is not router.banks[1]
    for idx in [1, 2]:
        for token in [0, 1]:
            grad = float(features[idx] * 3 * free_errors[token])
            assert router.banks[1][(idx, token)] == pytest.approx(-result["learning_rate"] * grad)
    assert len(router.banks[1]) == 4
    torch.testing.assert_close(router.predict_sparse(a, features, tokens), before, rtol=0, atol=0)
    new_scores = base + 3 * router.predict_sparse(b, features, tokens).tanh()
    assert result["loss_after"] == pytest.approx(float(new_scores.logsumexp(0) - new_scores[0]), abs=1e-12)


def test_bank_backtracking_checks_finite_prospective_weights_deterministically():
    a, _, _ = poles()
    features = torch.tensor([1e100], dtype=torch.float64)
    tokens = list(range(300))
    kwargs = {"all_token_ids": tokens, "base_logits": torch.zeros(300), "residual_gain": 3}
    first, second = CorticalColumnarSubBankRouter(), CorticalColumnarSubBankRouter()
    result = first.learn_sparse(a, features, 0, [0, 1], learning_rate=1e209, **kwargs)
    assert result == second.learn_sparse(a, features, 0, [0, 1], learning_rate=1e209, **kwargs)
    assert result["accepted"] and 0 < result["backtracking_steps"] < 12
    assert result["learning_rate"] == 1e209 * 0.5 ** result["backtracking_steps"]
    assert result["loss_after"] < result["loss"]
    assert_bank_state_equal(first.export_state_dict(), second.export_state_dict())
    assert all(torch.isfinite(torch.tensor(v, dtype=torch.float64)) for v in first.banks[0].values())


@pytest.mark.parametrize("reverse_entries", [False, True])
def test_bank_sparse_scoring_preserves_feature_sum_order_and_full_token_identity(reverse_entries):
    features = {8: 1.0, 3: 1.0, 0: 1.0, 5: -0.25}
    tokens = [11, 3, 99, 7]
    entries = [((0, 11), 1.0), ((3, 11), -1e16), ((8, 11), 1e16),
               ((5, 3), 0.8), ((8, 7), -0.125), ((55, 3), 100.0), ((8, 1000), 5.0)]
    bank = dict(reversed(entries) if reverse_entries else entries)
    before = dict(bank)
    expected = [sum(value * bank.get((idx, token), 0.0) for idx, value in features.items()) for token in tokens]
    assert expected[0] == 1.0
    assert CorticalColumnarSubBankRouter._scores(bank, features, tokens) == expected
    assert CorticalColumnarSubBankRouter._scores(bank, features, tokens[::-1]) == expected[::-1]
    assert CorticalColumnarSubBankRouter._scores({}, features, tokens) == [0.0] * len(tokens)
    assert bank == before


def test_bank_sparse_scoring_preserves_native_compensated_sum_exactly():
    features = {idx: 1 / (idx + 1) for idx in range(128)}
    tokens = list(range(321))
    bank = {(idx, token): 0.001 * (idx + 1) for idx in range(128) for token in range(256)}
    expected = [sum(value * bank.get((idx, token), 0.0) for idx, value in features.items()) for token in tokens]
    assert CorticalColumnarSubBankRouter._scores(bank, features, tokens) == expected


def test_bank_prediction_full_vocab_target_free_no_allocation_rng_and_roundtrip():
    router = CorticalColumnarSubBankRouter(prune_threshold=0.0)
    a, b, _ = poles()
    tokens = [1001 + 3 * i for i in range(513)]
    features = torch.zeros(1000, dtype=torch.float64)
    features[:150] = torch.arange(1., 151., dtype=torch.float64) / 150
    rng = torch.random.get_rng_state().clone()
    before = router.export_state_dict()
    assert torch.equal(router.predict_sparse(b, features, tokens), torch.zeros(513, dtype=torch.float64))
    assert_bank_state_equal(router.export_state_dict(), before)
    kwargs = {"all_token_ids": tokens, "base_logits": torch.zeros(513), "residual_gain": 3}
    result = router.learn_sparse(a, features, tokens[-1], [tokens[-1], tokens[0]], **kwargs)
    assert result["accepted"] and len(router.banks[0]) == 128 * 2
    before = router.export_state_dict()
    predictions = router.predict_sparse(b, features, tokens)
    assert predictions[-1] > 0 and predictions[0] < 0
    assert torch.equal(predictions[1:-1], torch.zeros(511, dtype=torch.float64))
    assert_bank_state_equal(router.export_state_dict(), before)
    assert router.predict_sparse(b, features, tokens[:-1]).tolist() == predictions[:-1].tolist()
    assert router.predict_sparse(b, features, list(reversed(tokens))).tolist() == predictions.flip(0).tolist()
    restored = checkpoint(router, CorticalColumnarSubBankRouter())
    assert_bank_state_equal(restored.export_state_dict(), before)
    torch.testing.assert_close(restored.predict_sparse(b, features, tokens), predictions, rtol=0, atol=0)
    assert router.learn_sparse(a, features, tokens[-1], [tokens[-1]], **kwargs) == restored.learn_sparse(
        a, features, tokens[-1], [tokens[-1]], **kwargs)
    assert_bank_state_equal(router.export_state_dict(), restored.export_state_dict())
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("invalid", [
    "base_shape", "base_size", "base_nan", "base_inf", "base_integer", "base_not_tensor",
    "mask_subset", "mask_target", "mask_duplicate", "mask_negative", "mask_budget",
    "vocab_duplicate", "vocab_negative", "vocab_fractional", "gain_zero", "gain_negative",
    "gain_nan", "gain_inf", "learning_rate_nan", "learning_rate_zero",
])
def test_bank_full_objective_validation_precedes_mutation(invalid):
    router = CorticalColumnarSubBankRouter()
    pole, _, _ = poles()
    features = torch.ones(1)
    tokens = list(range(300))
    kwargs = {"all_token_ids": tokens, "base_logits": torch.zeros(300), "residual_gain": 3}
    assert router.learn_sparse(pole, features, 0, [0, 1], **kwargs)["accepted"]
    before = router.export_state_dict()
    mask = [0, 1]
    if invalid == "base_shape":
        kwargs["base_logits"] = torch.zeros(1, 300)
    elif invalid == "base_size":
        kwargs["base_logits"] = torch.zeros(299)
    elif invalid in {"base_nan", "base_inf"}:
        kwargs["base_logits"][299] = float(invalid.split("_")[1])
    elif invalid == "base_integer":
        kwargs["base_logits"] = torch.zeros(300, dtype=torch.int64)
    elif invalid == "base_not_tensor":
        kwargs["base_logits"] = [0.] * 300
    elif invalid.startswith("mask_"):
        mask = {"mask_subset": [0, 300], "mask_target": [1], "mask_duplicate": [0, 0],
                "mask_negative": [0, -1], "mask_budget": list(range(257))}[invalid]
    elif invalid.startswith("vocab_"):
        kwargs["all_token_ids"] = list(tokens)
        kwargs["all_token_ids"][-1] = {"vocab_duplicate": 0, "vocab_negative": -1,
                                      "vocab_fractional": 299.5}[invalid]
    elif invalid.startswith("gain_"):
        kwargs["residual_gain"] = {"gain_zero": 0, "gain_negative": -3,
                                   "gain_nan": float("nan"), "gain_inf": float("inf")}[invalid]
    else:
        kwargs["learning_rate"] = float("nan") if invalid == "learning_rate_nan" else 0
    with pytest.raises(ValueError):
        router.learn_sparse(pole, features, 0, mask, **kwargs)
    assert_bank_state_equal(router.export_state_dict(), before)


@pytest.mark.parametrize("missing", ["all_token_ids", "base_logits", "residual_gain"])
def test_bank_full_objective_arguments_are_mandatory(missing):
    router = CorticalColumnarSubBankRouter()
    pole, _, _ = poles()
    kwargs = {"all_token_ids": [0, 1], "base_logits": torch.zeros(2), "residual_gain": 3}
    del kwargs[missing]
    with pytest.raises(TypeError):
        router.learn_sparse(pole, torch.ones(1), 0, [0, 1], **kwargs)
    assert not router.domains and not any(router.banks)


def test_bank_default_full_mask_reduced_to_budget_keeps_target_and_descends():
    router = CorticalColumnarSubBankRouter()
    pole = torch.eye(64)[0]
    tokens = list(range(256))
    kwargs = {"all_token_ids": tokens, "base_logits": torch.zeros(256), "residual_gain": 3}
    result = router.learn_sparse(pole, torch.ones(128), 0, tokens, **kwargs)
    assert result["accepted"] and result["update_mask_limited"]
    assert result["update_tokens_requested"] == 256
    assert 0 < result["update_tokens_selected"] < 256
    assert result["loss_after"] < result["loss"]
    assert router.memory_usage_bytes <= router.memory_budget_bytes
    domain = next(iter(router.domains.values()))
    selected_tokens = {entry[1] for entry in domain["gradient"]}
    assert 0 in selected_tokens  # Target always retained.
    assert len(domain["gradient"]) == result["gradient_entries_selected"]
    for idx in range(128):
        for token in tokens:
            assert ((idx, token) in router.banks[0]) == (token in selected_tokens)


def test_bank_mask_selection_deterministic_under_order_and_energy_ties():
    pole = torch.eye(64)[0]
    # Empty 2-bank router: 256 header + 768 new domain; each 1-feature token costs 384.
    # Budget 2176 admits exactly target + two others; equal energies tie-break to lowest IDs.
    def run(mask):
        router = CorticalColumnarSubBankRouter(n_banks=2, memory_budget_bytes=2176)
        result = router.learn_sparse(pole, torch.ones(1), 9, mask,
                                     all_token_ids=[1, 3, 5, 9], base_logits=torch.zeros(4), residual_gain=3)
        return router, result
    router_a, result_a = run([5, 3, 9, 1])
    router_b, result_b = run([1, 9, 3, 5])
    assert result_a["accepted"] and result_a["update_tokens_selected"] == 3
    selected = {entry[1] for entry in next(iter(router_a.domains.values()))["gradient"]}
    assert selected == {9, 1, 3}  # Target 9, then lowest-ID ties 1 and 3.
    assert result_a == result_b
    assert_bank_state_equal(router_a.export_state_dict(), router_b.export_state_dict())


def test_bank_mask_reuses_existing_entries_without_double_charging_old_gradient():
    pole = torch.eye(64)[0]
    # After first call usage is 1792; replacing the 2-entry domain gradient and adding two
    # new feature entries lands exactly at 2560 only if the old gradient is not charged twice.
    router = CorticalColumnarSubBankRouter(n_banks=2, memory_budget_bytes=2560)
    assert router.learn_sparse(pole, torch.tensor([1.0]), 7, [7, 11],
                               all_token_ids=[7, 11], base_logits=torch.zeros(2), residual_gain=3)["accepted"]
    assert router.memory_usage_bytes == 1792
    result = router.learn_sparse(pole, torch.tensor([1.0, 1.0]), 7, [7, 11],
                                 all_token_ids=[7, 11], base_logits=torch.zeros(2), residual_gain=3)
    assert result["accepted"] and not result["update_mask_limited"]
    assert router.memory_usage_bytes == 2560
    domain = next(iter(router.domains.values()))
    assert len(domain["gradient"]) == 4 and domain["updates"] == 2


def test_bank_split_refits_reduced_mask_on_free_branch_and_keeps_original_baseline():
    a, b, _ = poles()
    router = CorticalColumnarSubBankRouter(n_banks=2, memory_budget_bytes=3328)
    features = torch.ones(1)
    kwargs = {"all_token_ids": [0, 1, 2, 3], "base_logits": torch.zeros(4), "residual_gain": 3}
    for _ in range(12):
        assert router.learn_sparse(a, features, 1, [0, 1], **kwargs)["accepted"]
    old_bank = dict(router.banks[0])
    result = router.learn_sparse(b, features, 0, [0, 1, 2, 3], **kwargs)
    assert result["accepted"] and result["allocated_branch"] and result["bank"] == 1
    assert result["gradient_conflict"] < router.conflict_threshold
    assert result["update_tokens_requested"] == 4 and result["update_tokens_selected"] == 2
    assert result["update_mask_limited"] and result["loss_after"] < result["loss"]
    assert router.banks[0] == old_bank  # Original routed bank untouched by the split.
    selected = {entry[1] for entry in router.banks[1]}
    assert selected == {0, 1}  # Target 0, then lowest-ID energy tie 1.
    assert router.memory_usage_bytes <= router.memory_budget_bytes


def test_bank_reduced_mask_checkpoint_continued_learning_matches():
    pole = torch.eye(64)[0]
    tokens = list(range(256))
    kwargs = {"all_token_ids": tokens, "base_logits": torch.zeros(256), "residual_gain": 3}
    router = CorticalColumnarSubBankRouter()
    assert router.learn_sparse(pole, torch.ones(128), 0, tokens, **kwargs)["accepted"]
    restored = checkpoint(router, CorticalColumnarSubBankRouter())
    assert router.learn_sparse(pole, torch.ones(128), 5, tokens, **kwargs) == restored.learn_sparse(
        pole, torch.ones(128), 5, tokens, **kwargs)
    assert_bank_state_equal(router.export_state_dict(), restored.export_state_dict())
    torch.testing.assert_close(router.predict_sparse(pole, torch.ones(128), tokens),
                               restored.predict_sparse(pole, torch.ones(128), tokens), rtol=0, atol=0)


def test_bank_pruning_removes_near_zero_entries_frees_memory_and_preserves_objective():
    router = CorticalColumnarSubBankRouter(n_banks=2, prune_threshold=1e-3)
    pole = torch.eye(64)[0]
    tokens = list(range(64))
    features = torch.ones(4, dtype=torch.float64)
    router.banks[0] = {(0, 0): 0.5, (1, 0): 0.4}
    router.banks[0].update({(0, t): 1e-5 for t in range(10, 60)})
    usage_before = router.memory_usage_bytes
    kwargs = {"all_token_ids": tokens, "base_logits": torch.zeros(64, dtype=torch.float64), "residual_gain": 3}
    result = router.learn_sparse(pole, features, 0, [0, 1, 2], **kwargs)
    assert result["accepted"]
    assert result["pruned_entries"] >= 50
    assert result["loss_after"] <= result["loss"]
    assert all(abs(w) > 1e-3 for w in router.banks[0].values())
    assert (0, 0) in router.banks[0] and (1, 0) in router.banks[0]
    assert router.memory_usage_bytes < usage_before


def test_bank_pruning_disabled_keeps_near_zero_entries():
    router = CorticalColumnarSubBankRouter(n_banks=2, prune_threshold=0.0)
    pole = torch.eye(64)[0]
    tokens = list(range(64))
    features = torch.ones(4, dtype=torch.float64)
    router.banks[0] = {(0, 0): 0.5, (0, 50): 1e-6}
    kwargs = {"all_token_ids": tokens, "base_logits": torch.zeros(64, dtype=torch.float64), "residual_gain": 3}
    result = router.learn_sparse(pole, features, 0, [0, 1, 2], **kwargs)
    assert result["accepted"] and result["pruned_entries"] == 0
    assert router.banks[0][(0, 50)] == 1e-6


def test_bank_pruning_checkpoint_roundtrip_and_continuation_match():
    pole = torch.eye(64)[0]
    tokens = list(range(64))
    kwargs = {"all_token_ids": tokens, "base_logits": torch.zeros(64), "residual_gain": 3}
    router = CorticalColumnarSubBankRouter(prune_threshold=1e-3)
    assert router.learn_sparse(pole, torch.ones(8), 0, tokens, **kwargs)["accepted"]
    restored = checkpoint(router, CorticalColumnarSubBankRouter())
    assert restored.prune_threshold == router.prune_threshold == 1e-3
    assert_bank_state_equal(router.export_state_dict(), restored.export_state_dict())
    assert router.learn_sparse(pole, torch.ones(8), 5, tokens, **kwargs) == restored.learn_sparse(
        pole, torch.ones(8), 5, tokens, **kwargs)
    assert_bank_state_equal(router.export_state_dict(), restored.export_state_dict())


def test_bank_pruning_relieves_saturation_so_new_target_fits():
    pole = torch.eye(64)[0]
    tokens = list(range(8))
    features = torch.ones(1, dtype=torch.float64)
    kwargs = {"all_token_ids": tokens, "base_logits": torch.zeros(8, dtype=torch.float64), "residual_gain": 3}
    budget = 128 + 192 * 20
    seed = {(0, t): 1e-5 for t in range(10, 30)}
    without = CorticalColumnarSubBankRouter(n_banks=1, memory_budget_bytes=budget, prune_threshold=0.0)
    without.banks[0] = dict(seed)
    with_prune = CorticalColumnarSubBankRouter(n_banks=1, memory_budget_bytes=budget, prune_threshold=1e-2)
    with_prune.banks[0] = dict(seed)
    assert without.memory_usage_bytes == with_prune.memory_usage_bytes == budget
    rejected = without.learn_sparse(pole, features, 0, [0], **kwargs)
    assert not rejected["accepted"] and rejected["reason"] == "memory_budget_exhausted"
    accepted = with_prune.learn_sparse(pole, features, 0, [0], **kwargs)
    assert accepted["accepted"] and accepted["pruned_entries"] == 20
    assert accepted["loss_after"] <= accepted["loss"]
    assert with_prune.memory_usage_bytes <= budget
    assert all(abs(w) > 1e-2 for w in with_prune.banks[0].values())


def test_bank_pruning_migration_from_old_checkpoint_without_threshold():
    router = CorticalColumnarSubBankRouter(prune_threshold=1e-4)
    state = export_runtime_tree(router)
    del state["attributes"]["prune_threshold"]
    restored = load_runtime_tree(CorticalColumnarSubBankRouter(), state)
    assert not hasattr(restored, "prune_threshold")
    if not hasattr(restored, "prune_threshold"):
        restored.prune_threshold = 1e-4
    pole = torch.eye(64)[0]
    tokens = list(range(8))
    kwargs = {"all_token_ids": tokens, "base_logits": torch.zeros(8), "residual_gain": 3}
    result = restored.learn_sparse(pole, torch.ones(8), 0, tokens, **kwargs)
    assert result["accepted"] and "pruned_entries" in result


def edge(subject, obj, relation="drives", **attributes):
    return {"subject": subject, "object": obj, "relation": relation,
            "polarity": "positive", "causal_kind": "activation", "type": "signal", **attributes}


def graph_pair():
    dreamer = CrossDomainIsomorphismDreamer()
    source_edges = [edge("zul", "vek"), edge("vek", "nox"), edge("nox", "pir"),
                    edge("zul", "pir", "suppresses", polarity="negative", causal_kind="inhibition")]
    target_edges = [edge("gamma", "delta"), edge("alpha", "beta"), edge("beta", "gamma")]
    dreamer.register_graph("Veldora", source_edges, dict.fromkeys(["zul", "vek", "nox", "pir"], "unit"))
    dreamer.register_graph("Threxal", target_edges, dict.fromkeys(["delta", "beta", "alpha", "gamma"], "unit"))
    return dreamer


def test_novel_name_structural_heldout_transfer_and_graph_roundtrip():
    dreamer = graph_pair()
    result = dreamer.transfer_relation("Veldora", "Threxal", "zul", "suppresses", "pir")
    assert result is not None
    assert result["mapping"] == {"zul": "alpha", "vek": "beta", "nox": "gamma", "pir": "delta"}
    assert result["edge"] == edge("alpha", "delta", "suppresses", polarity="negative", causal_kind="inhibition")
    assert len(result["proof"]) == 3
    assert all(p["source_edge"]["relation"] != "suppresses" for p in result["proof"])
    assert result["status"] == "structural_prediction"
    assert dreamer.match_graphs("Veldora", "Threxal") is None
    restored = checkpoint(dreamer, CrossDomainIsomorphismDreamer())
    assert restored.transfer_relation("Veldora", "Threxal", "zul", "suppresses", "pir") == result
    target = copy.deepcopy(restored.graphs["Threxal"])
    target["edges"].append(result["edge"])
    restored.register_graph("Threxal", target["edges"], target["node_types"])
    assert restored.match_graphs("Veldora", "Threxal")["score"] == 1.0
    assert restored.resolve_cross_domain_isomorphism("Compare Veldora with Threxal")["mapping"] == result["mapping"]
    assert restored.resolve_cross_domain_isomorphism("Explain Veldora and Threxal") is None
    assert restored.resolve_cross_domain_isomorphism("Compare BCM and insulin homeostasis") is None


@pytest.mark.parametrize("mutation", ["direction", "node_type", "polarity", "causal_kind", "edge_type", "relation"])
def test_graph_alignment_rejects_direction_type_and_polarity_mismatch(mutation):
    dreamer = graph_pair()
    target = copy.deepcopy(dreamer.graphs["Threxal"])
    if mutation == "direction":
        target["edges"][0]["subject"], target["edges"][0]["object"] = target["edges"][0]["object"], target["edges"][0]["subject"]
    elif mutation == "node_type":
        target["node_types"]["delta"] = "incompatible"
    else:
        key = "type" if mutation == "edge_type" else mutation
        target["edges"][0][key] = "incompatible"
    dreamer.register_graph("Threxal", target["edges"], target["node_types"])
    assert dreamer.transfer_relation("Veldora", "Threxal", "zul", "suppresses", "pir") is None


@pytest.mark.parametrize("attribute", ["polarity", "causal_kind", "type", "direction"])
def test_heldout_target_edge_cannot_validate_a_contradiction(attribute):
    dreamer = graph_pair()
    target = copy.deepcopy(dreamer.graphs["Threxal"])
    contradiction = edge("alpha", "delta", "suppresses", polarity="negative", causal_kind="inhibition")
    contradiction[attribute] = "reverse" if attribute == "direction" else "incompatible"
    target["edges"].append(contradiction)
    dreamer.register_graph("Threxal", target["edges"], target["node_types"])
    assert dreamer.transfer_relation("Veldora", "Threxal", "zul", "suppresses", "pir") is None


def test_ambiguous_or_evidenceless_graphs_do_not_transfer():
    dreamer = CrossDomainIsomorphismDreamer()
    types = dict.fromkeys("abcd", "unit")
    dreamer.register_graph("source", [edge("a", "b"), edge("c", "d"), edge("a", "c", "touches")], types)
    dreamer.register_graph("target", [edge("a", "b"), edge("c", "d")], types)
    assert dreamer.transfer_relation("source", "target", "a", "touches", "c") is None
    dreamer.register_graph("single", [edge("a", "b")], {"a": "input", "b": "output"})
    dreamer.register_graph("blank", [], {"a": "input", "b": "output"})
    assert dreamer.transfer_relation("single", "blank", "a", "drives", "b") is None
    assert dreamer.transfer_relation("missing", "blank", "a", "drives", "b") is None
    with pytest.raises(ValueError):
        dreamer.register_graph("untyped", [edge("a", "missing")], {"a": "input"})


def test_dashboard_empty_legacy_inputs_are_not_measurements():
    metrics = ConsciousnessMetricsDashboard().compute_turn_consciousness_metrics(0.99, 0.0, [])
    assert all(value is None for key, value in metrics.items() if key != "unavailable_measurements")
    assert "iit_pci_complexity_proxy" in metrics["unavailable_measurements"]
    assert "hot_calibration_brier" in metrics["unavailable_measurements"]
    assert "rpt_recurrence_index" in metrics["unavailable_measurements"]


def test_dashboard_real_traces_compute_actual_values_without_criticality_clamp():
    raster = torch.tensor([[1., 0, 0], [2., 0, 0], [4., 0, 0]])
    perturbed = raster.clone()
    perturbed[1, 1] = 1
    metrics = ConsciousnessMetricsDashboard().compute_turn_consciousness_metrics(
        0.01, 0.99, [], spike_raster=raster, perturbed_spike_raster=perturbed,
        basal_energy=[3, 1], apical_energy=[1, 3], confidence_outcomes=[[0.9, 1], [0.8, 0]],
        attention_predictions=[[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]], attention_targets=[0, 1, 1],
    )
    assert metrics["priesemann_subcritical_sigma"] == 2.0
    assert metrics["gwt_ignition_score"] == pytest.approx(1 / 3)
    assert metrics["broadcast_reach"] == pytest.approx(1 / 3)
    assert metrics["rpt_recurrence_index"] == 0.5
    assert metrics["hot_metacognitive_confidence"] == pytest.approx(0.85)
    assert metrics["hot_calibration_brier"] == pytest.approx(0.325)
    assert metrics["hot_calibration_score"] == pytest.approx(0.675)
    assert metrics["ast_attention_schema_alignment"] == pytest.approx(2 / 3)
    assert metrics["iit_pci_complexity_proxy"] > 0
    assert metrics["unavailable_measurements"] == []


def test_dashboard_perturbational_compressibility_depends_on_measured_response():
    dashboard = ConsciousnessMetricsDashboard()
    raster = torch.zeros((256, 16))
    random_response = torch.randint(0, 2, raster.shape, generator=torch.Generator().manual_seed(19))
    constant = dashboard.compute_turn_consciousness_metrics(spike_raster=raster, perturbed_spike_raster=torch.ones_like(raster))
    random = dashboard.compute_turn_consciousness_metrics(spike_raster=raster, perturbed_spike_raster=random_response)
    assert constant["iit_pci_complexity_proxy"] < random["iit_pci_complexity_proxy"]
    assert constant["iit_pci_complexity_proxy"] < 0.32  # No fabricated minimum.
    no_effect = dashboard.compute_turn_consciousness_metrics(spike_raster=raster, perturbed_spike_raster=raster)
    assert no_effect["iit_pci_complexity_proxy"] == 0.0
    assert no_effect["priesemann_subcritical_sigma"] is None
    unperturbed = dashboard.compute_turn_consciousness_metrics(spike_raster=random_response)
    assert unperturbed["iit_pci_complexity_proxy"] is None


def test_dashboard_malformed_measurements_stay_unavailable():
    result = ConsciousnessMetricsDashboard().compute_turn_consciousness_metrics(
        spike_raster=[1, 2], perturbed_spike_raster=[[1]], basal_energy=[0], apical_energy=[0],
        confidence_outcomes=[[1.5, 1]], attention_predictions=[float("nan")], attention_targets=[1],
    )
    assert all(value is None for key, value in result.items() if key != "unavailable_measurements")
    result = ConsciousnessMetricsDashboard().compute_turn_consciousness_metrics(
        cortical_trajectory_trace=[["A"], ["A", "B"]], total_columns=4,
    )
    assert result["broadcast_reach"] == 0.5
    assert result["gwt_ignition_score"] == 0.5
    assert result["rpt_recurrence_index"] is None
