"""TCC acceptance tests (blueprint §3.8): wrap, bump stability, interference,
reversible decoding, the commutative phase-sum limitation, aliasing, held-out
composition, and an honest quality+cost comparison against a cosine-VSA baseline.

Synthetic vectors only — gridness / a stable bump is NOT evidence of textual
causality. The module is opt-in and standalone (not wired into live generation).
"""
import math

import pytest
import torch
import torch.nn.functional as F

from ziwago.memory.tcc import (
    ToroidalRelationAttractor,
    phase_sum_bind,
    run_acceptance_experiment,
    toroidal_distance,
    typed_directional_bind,
)

TWO_PI = 2.0 * math.pi


def _vec(seed, d=32):
    return F.normalize(torch.randn(d, generator=torch.Generator().manual_seed(seed)), dim=0)


def _tcc(**kw):
    params = dict(grid_size=24, d_vec=32, seed=0, enabled=True)
    params.update(kw)
    return ToroidalRelationAttractor(**params)


def _stored(n=6, d=32, **kw):
    tcc = _tcc(d_vec=d, **kw)
    for i in range(n):
        tcc.store_relation(f"rel_{i}", _vec(i, d))
    return tcc


# ── wrap ────────────────────────────────────────────────────────────────────
def test_wrap_is_exact_on_torus():
    eps = 1e-3
    assert toroidal_distance((0.0, 0.0), (TWO_PI - eps, 0.0)) == pytest.approx(eps, abs=1e-9)
    assert toroidal_distance((0.0, 0.0), (0.0, TWO_PI - eps)) == pytest.approx(eps, abs=1e-9)
    # antipodal point is pi away on one axis, not 3*pi
    assert toroidal_distance((0.0, 0.0), (math.pi, 0.0)) == pytest.approx(math.pi, abs=1e-9)


def test_encode_vector_wraps_into_range():
    tcc = _tcc()
    for seed in range(8):
        t1, t2 = tcc.encode_vector(_vec(seed) * 50.0)  # large drive must still wrap
        assert 0.0 <= t1 < TWO_PI and 0.0 <= t2 < TWO_PI


# ── bump stability (genuine CANN) ───────────────────────────────────────────
def test_free_bump_is_stable_without_input():
    tcc = _stored()
    stab = tcc.bump_stability("rel_0", steps=60, anchor_gain=0.0)
    assert stab["localization_after"] > 0.5          # bump persists, not silence
    assert stab["target_drift"] < 0.1                # stays at the encoded phase
    assert stab["localization_after"] <= stab["localization_before"] + 1e-6


def test_bump_stability_is_caused_by_recurrent_excitation():
    # П-5: with recurrent excitation removed the bump must collapse — stability is
    # not a trivial artifact of initialization.
    tcc = _stored(w_exc=0.0)
    stab = tcc.bump_stability("rel_0", steps=60, anchor_gain=0.0)
    assert stab["localization_after"] < 0.1


def test_pinned_bump_tracks_anchor():
    tcc = _stored()
    stab = tcc.bump_stability("rel_1", steps=40, anchor_gain=0.5)
    assert stab["localization_after"] > 0.5
    assert stab["target_drift"] < 0.1


# ── reversible decoding ─────────────────────────────────────────────────────
def test_stored_relations_decode_reversibly():
    tcc = _stored(n=8)
    hits = sum(tcc.recall(f"rel_{i}", settle_steps=12)["decoded"] == f"rel_{i}"
               for i in range(8))
    assert hits == 8


def test_decode_reads_activity_not_storage_order():
    # П-5: place the bump at rel_3's phase after storing rel_0..5; decoding must
    # follow the bump, not the most recently stored anchor.
    tcc = _stored(n=6)
    t1, t2 = tcc.anchors["rel_3"]
    tcc.place_bump(t1, t2)
    tcc.step(steps=12)
    assert tcc.decode()["decoded"] == "rel_3"


def test_silence_decodes_to_none():
    tcc = _stored()
    tcc.reset()
    report = tcc.decode()
    assert report["decoded"] is None
    assert tcc.localization() == 0.0


# ── interference (honest single-bump attractor) ─────────────────────────────
def test_superposed_relations_merge_into_single_bump():
    tcc = _stored(n=8)
    # pick the farthest pair by toroidal separation
    labels = list(tcc.anchors)
    pairs = [(a, b, toroidal_distance(tcc.anchors[a], tcc.anchors[b]))
             for i, a in enumerate(labels) for b in labels[i + 1:]]
    a, b, sep = max(pairs, key=lambda x: x[2])
    res = tcc.interference(a, b, steps=30)
    # a single bump survives (winner-take-all), centered between the two anchors:
    # this bounded CANN does NOT hold two independent relations at once.
    assert res["localization_after"] > 0.5
    assert res["dist_to_a"] < sep and res["dist_to_b"] < sep
    assert res["dist_to_a"] > 0.01 and res["dist_to_b"] > 0.01  # a blend, not clean recall


# ── commutative limitation vs typed operator ────────────────────────────────
def test_phase_sum_is_commutative_and_loses_direction():
    tcc = _stored(n=2)
    a, b = tcc.anchors["rel_0"], tcc.anchors["rel_1"]
    assert phase_sum_bind(a, b) == phase_sum_bind(b, a)  # cannot encode A->B vs B->A


def test_typed_binding_is_non_commutative():
    va, vb = _vec(1), _vec(2)
    ra, rp = _vec(10), _vec(11)
    ab = typed_directional_bind(va, vb, ra, rp)
    ba = typed_directional_bind(vb, va, ra, rp)
    sim = float(torch.dot(F.normalize(ab.real, dim=0), F.normalize(ba.real, dim=0)))
    assert sim < 0.99  # direction is representable only with the typed operator


# ── aliasing / held-out ─────────────────────────────────────────────────────
def test_aliasing_is_reported_not_hidden():
    tcc = _tcc()
    v = _vec(0)
    tcc.store_relation("a", v)
    tcc.store_relation("b", v)  # identical vector -> identical phase -> collision
    alias = tcc.aliasing(tol=0.15)
    assert alias["n_collisions"] >= 1
    assert ("a", "b") in alias["collisions"]


def test_heldout_composition_decodes_geometrically():
    tcc = _stored(n=4)
    comp = phase_sum_bind(tcc.anchors["rel_0"], tcc.anchors["rel_1"])
    tcc.place_bump(*comp)
    report = tcc.decode()
    assert report["decoded"] in tcc.anchors        # nearest stored anchor
    assert report["distance"] is not None


# ── acceptance harness: honest baseline comparison ──────────────────────────
def test_acceptance_reports_no_advantage_and_boundaries():
    report = run_acceptance_experiment(n_relations=8, grid_size=24, d_vec=32, seed=0)
    assert report["wrap_error"] < 1e-9
    assert report["decode_accuracy"] == 1.0
    assert report["bump_stability_free"]["localization_after"] > 0.5
    assert report["phase_sum_commutative"] is True
    assert report["typed_directional_similarity"] < 0.99
    # honest negative result: TCC does not beat the cheap cosine-VSA baseline on
    # quality AND persistent tensor cost, so it must not be scaled yet.
    assert report["advantage_over_baseline"] is False
    assert report["baseline_accuracy"] == 1.0
    assert report["decode_accuracy"] == report["baseline_accuracy"]
    assert report["quality_metric"] == "paired_noisy_retrieval_accuracy"
    assert report["noisy_baseline_accuracy"] < report["baseline_accuracy"]
    assert report["noisy_decode_accuracy"] < report["noisy_baseline_accuracy"]
    assert report["cost_metric"] == "persistent_tensor_bytes"
    assert (report["tcc_persistent_tensor_bytes"]
            > report["baseline_persistent_tensor_bytes"])
    assert report["synthetic_only"] is True
    assert report["textual_causality_proven"] is False


# ── config / serialization / opt-in ─────────────────────────────────────────
def test_default_is_opt_in_off():
    assert ToroidalRelationAttractor().enabled is False


def test_config_validation():
    with pytest.raises(ValueError):
        ToroidalRelationAttractor(grid_size=2)
    with pytest.raises(ValueError):
        ToroidalRelationAttractor(w_exc=-1.0)
    with pytest.raises(ValueError):
        ToroidalRelationAttractor(dt=0.0)
    with pytest.raises(ValueError):
        ToroidalRelationAttractor(bump_width=0.0)
    with pytest.raises(ValueError):
        _tcc().encode_vector(torch.zeros(31))  # wrong d_vec
    with pytest.raises(ValueError):
        _tcc().store_relation("", _vec(0))


def test_export_load_roundtrip_preserves_decoding():
    tcc = _stored(n=6)
    tcc.recall("rel_2", settle_steps=12)
    state = tcc.export_state_dict()
    fresh = ToroidalRelationAttractor()
    fresh.load_state_dict(state)
    assert fresh.anchors == tcc.anchors
    assert torch.allclose(fresh.proj, tcc.proj)
    assert torch.allclose(fresh.activity, tcc.activity)
    assert fresh.decode()["decoded"] == tcc.decode()["decoded"]
    with pytest.raises(ValueError):
        bad = dict(state)
        bad["proj"] = torch.zeros(2, 99)
        ToroidalRelationAttractor().load_state_dict(bad)