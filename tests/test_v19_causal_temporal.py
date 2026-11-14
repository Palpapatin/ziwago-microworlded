"""Independent synthetic v19 regression tests: no corpus, trainer or checkpoint."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

MODULE = Path(__file__).resolve().parents[1] / "ziwago" / "cortex" / "genesis_engine.py"
SPEC = importlib.util.spec_from_file_location("v19_genesis_under_test", MODULE)
ENGINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ENGINE
SPEC.loader.exec_module(ENGINE)


def graph():
    # Character encoding and deterministic vectors; no neural/corpus dependencies.
    def vector(ids):
        out = torch.zeros(64)
        for index, token in enumerate(ids):
            out[(token + index) % 64] += 1
        return F.normalize(out, dim=0)
    return ENGINE.CorticalRelationalWorldGraph(lambda text: list(map(ord, text)), vector)


class CausalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.g = graph()

    def test_negation_is_not_part_of_subject_and_never_inverts_effect(self):
        self.g.extract_and_consolidate_sentence("The alpha does not activate the beta.", source="lab")
        edge = self.g.edges[0]
        self.assertEqual((edge.subject, edge.relation, edge.target, edge.polarity),
                         ("alpha", "activate", "beta", -1))
        self.assertEqual(edge.status, "asserted")
        self.assertTrue(self.g.ground_response("alpha beta", "Alpha not activates beta.")["grounded"])
        self.assertFalse(self.g.ground_response("alpha beta", "Alpha activates beta.")["grounded"])
        self.assertIsNone(self.g.intervene({"alpha": True})["values"]["beta"])
        self.assertEqual(self.g.extract_and_consolidate_sentence("Not alpha activates beta."), [])

    def test_articles_and_ambiguous_clauses_are_quarantined_not_nodes(self):
        malformed = ("The activates beta.", "Alpha activates an.", "Alpha activates.",
                     "Alpha activates beta and inhibits gamma.", "Alpha may activate beta.",
                     "Alpha activates beta which activates gamma.", "Alpha activates beta?",
                     "It activates this.", "Alpha activates beta if warm and not warm.")
        for sentence in malformed:
            self.assertEqual(self.g.extract_and_consolidate_sentence(sentence), [])
        self.assertEqual(self.g.edges, [])
        self.assertEqual(len(self.g.quarantined_clauses), len(malformed))
        invalid = self.g.add_or_strengthen_edge("the", "activates", "beta")
        self.assertEqual(invalid.status, "quarantined")
        self.assertNotIn("", self.g.adj_by_subject)
        self.assertEqual(self.g.extract_and_consolidate_sentence("An alpha activates a beta."),
                         [("alpha", "activate", "beta")])

    def test_exact_boundaries_no_substring_or_shared_word_hops(self):
        first = self.g.add_or_strengthen_edge("alpha", "cause", "beta protein", causal_kind="sufficient")
        self.g.add_or_strengthen_edge("beta", "cause", "gamma", 999, causal_kind="sufficient")
        self.assertEqual(self.g.find_causal_path("alpha gamma"), [])
        self.assertEqual(self.g.find_causal_path("alphabet beta protein"), [])
        last = self.g.add_or_strengthen_edge("beta protein", "cause", "gamma", causal_kind="sufficient")
        self.assertEqual([e.edge_id for e in self.g.find_causal_path("alpha gamma")],
                         [first.edge_id, last.edge_id])
        self.assertEqual(self.g.find_causal_path("alpha gamma-ray"), [first])

    def test_provenance_repetition_conflicts_and_context(self):
        positive = self.g.add_or_strengthen_edge("alpha", "activates", "beta", 2, source="one")
        repeated = self.g.add_or_strengthen_edge("the alpha", "activate", "beta", 100, source="one")
        self.assertIs(positive, repeated)
        self.assertEqual(positive.status, "asserted")
        self.assertTrue(positive.observed_at)
        negative = self.g.add_or_strengthen_edge("alpha", "does not activate", "beta", source="two")
        self.assertIn(negative.edge_id, positive.conflict_ids)
        self.assertIn(positive.edge_id, negative.conflict_ids)
        for response in ("Alpha activates beta.", "Alpha does not activate beta."):
            self.assertFalse(self.g.ground_response("alpha beta", response)["grounded"])
        self.assertEqual(self.g.ground_response("alpha beta", "Alpha activates beta.")["sources"], [])
        conditional = graph()
        conditional.extract_and_consolidate_sentence("Alpha causes beta if warm.")
        conditional.extract_and_consolidate_sentence("Alpha does not cause beta if not warm.")
        self.assertEqual(conditional.edges[0].conflict_ids, ())
        self.assertIs(conditional.intervene({"alpha": True}, conditions=("warm",))["values"]["beta"], True)
        self.assertIsNone(conditional.intervene({"alpha": True}, conditions=("not warm",))["values"]["beta"])

    def test_alternative_sufficient_routes_and_open_world_absence(self):
        self.g.add_or_strengthen_edge("alpha", "cause", "gamma", causal_kind="sufficient")
        alternate = self.g.add_or_strengthen_edge("beta", "cause", "gamma", causal_kind="sufficient", source="alternate")
        result = self.g.intervene({"alpha": False}, observed={"beta": True})
        self.assertIs(result["values"]["gamma"], True)
        self.assertEqual(result["proofs"]["gamma"], [alternate.edge_id])
        self.assertEqual(result["sources"], ["alternate"])
        self.assertIsNone(self.g.intervene({"alpha": False, "beta": False})["values"]["gamma"])
        self.assertIs(self.g.intervene({"gamma": False}, observed={"beta": True})["values"]["gamma"], False)

    def test_necessary_failure_presence_and_incompatible_routes(self):
        requirement = self.g.add_or_strengthen_edge("fuel", "is necessary for", "engine", causal_kind="necessary")
        self.g.add_or_strengthen_edge("engine", "is necessary for", "motion", causal_kind="necessary")
        absent = self.g.intervene({"fuel": False})
        self.assertIs(absent["values"]["motion"], False)
        self.assertIn(requirement.edge_id, absent["proofs"]["motion"])
        self.assertIsNone(self.g.intervene({"fuel": True})["values"]["engine"])
        self.assertEqual(self.g.find_causal_path("fuel motion"), [])
        self.assertEqual(len(self.g.find_causal_path("fuel motion", causal_kind="necessary")), 2)
        self.g.add_or_strengthen_edge("battery", "cause", "engine", causal_kind="sufficient")
        incompatible = self.g.intervene({"fuel": False}, {"battery": True})
        self.assertIsNone(incompatible["values"]["engine"])
        self.assertIn("engine", incompatible["conflicts"])
        self.assertIsNone(incompatible["values"]["motion"])

    def test_explicit_causal_language_defaults_and_negative_conditions(self):
        cause = self.g.add_or_strengthen_edge("alpha", "causes", "beta")
        self.assertEqual((cause.causal_kind, cause.status), ("sufficient", "asserted"))
        requirement = self.g.add_or_strengthen_edge("gamma", "requires", "beta")
        self.assertEqual((requirement.subject, requirement.target, requirement.causal_kind),
                         ("beta", "gamma", "necessary"))
        self.g.extract_and_consolidate_sentence("Fuel is necessary for engine.")
        self.assertEqual(self.g.edges[-1].subject, "fuel")
        denial = self.g.add_or_strengthen_edge("switch", "is not sufficient to inhibit", "lamp",
                                               conditions=("not warm",))
        self.assertEqual((denial.relation, denial.causal_kind, denial.polarity), ("inhibit", "sufficient", -1))
        self.assertTrue(self.g.ground_response("switch lamp", "Switch is not sufficient to inhibit lamp if not warm.")["grounded"])
        self.assertIsNone(self.g.intervene({"switch": True}, conditions=("not warm",))["values"]["lamp"])
        with self.assertRaises(ValueError):
            self.g.intervene({"alpha": True, "the alpha": False})
        with self.assertRaises(ValueError):
            self.g.add_or_strengthen_edge("alpha", "activate", "beta", conditions=("warm", "not warm"))

    def test_conditions_filter_paths_and_inference(self):
        self.g.extract_and_consolidate_sentence("Alpha causes beta if warm and dry.", source="conditional")
        self.g.extract_and_consolidate_sentence("Beta causes gamma.")
        self.assertEqual(self.g.find_causal_path("alpha gamma"), [])
        self.assertEqual(self.g.find_causal_path("alpha gamma", conditions=("warm",)), [])
        self.assertEqual(len(self.g.find_causal_path("alpha gamma", conditions=("dry", "warm"))), 2)
        self.assertIsNone(self.g.intervene({"alpha": True})["values"]["gamma"])
        self.assertIs(self.g.intervene({"alpha": True}, conditions=("warm", "dry"))["values"]["gamma"], True)
        self.assertFalse(self.g.ground_response("alpha gamma if warm and dry", "Alpha causes gamma.")["grounded"])
        self.assertTrue(self.g.ground_response("alpha gamma", "Alpha causes gamma if dry and warm.")["grounded"])

    def test_association_and_negative_causality_are_not_executable(self):
        self.g.add_or_strengthen_edge("alpha", "activate", "beta")
        self.g.add_or_strengthen_edge("beta", "cause", "gamma", causal_kind="sufficient", polarity=-1)
        self.assertEqual(self.g.find_causal_path("alpha gamma"), [])
        self.assertIsNone(self.g.intervene({"alpha": False, "beta": True})["values"]["gamma"])
        self.g.add_or_strengthen_edge("switch", "inhibit", "lamp", causal_kind="sufficient")
        self.assertIs(self.g.intervene({"switch": True})["values"]["lamp"], False)
        rendered = self.g.synthesize_novel_sentence_from_graph("switch lamp")
        self.assertTrue(rendered["grounded"])

    def test_all_material_clauses_grounded_not_domain_overlap(self):
        self.g.extract_and_consolidate_sentence("Alpha activates beta.", source="lesson")
        self.g.extract_and_consolidate_sentence("Gamma protects delta.", source="manual")
        query = "Explain alpha beta and gamma delta"
        complete = self.g.ground_response(query, "Alpha activates beta and gamma protects delta.")
        self.assertTrue(complete["grounded"])
        self.assertEqual(complete["semantic_closure_score"], 1)
        self.assertEqual(complete["sources"], ["lesson", "manual"])
        self.assertEqual(len(complete["proof_edge_ids"]), 2)
        partial = self.g.ground_response(query, "Alpha activates beta.")
        self.assertTrue(partial["grounded"])
        self.assertEqual(partial["semantic_closure_score"], .5)
        for response in ("Alpha activates beta; gamma activates delta.",
                         "Alpha activates beta. Delta is immortal.", "Alpha activates delta.",
                         "Alpha activates beta and gamma protects delta and magic."):
            self.assertFalse(self.g.ground_response(query, response)["grounded"])
        self.assertFalse(self.g.ground_response(query, "")["grounded"])
        unrelated = self.g.ground_response("Describe quantum gravity", "Alpha activates beta.")
        self.assertEqual(unrelated["semantic_closure_score"], 0)
        unknown_target = self.g.ground_response("Does alpha activate epsilon?", "Alpha activates beta.")
        self.assertLess(unknown_target["semantic_closure_score"], 1)

    def test_legacy_migration_and_metadata_roundtrip_without_training(self):
        with patch.object(self.g, "imprint_edges_into_dedicated_valence_matrix", side_effect=AssertionError("training")):
            self.g.load_state_dict([dict(subject="alpha", relation="activates", target="beta", conductance=9999)])
        self.assertEqual(self.g.edges[0].status, "quarantined")
        self.assertFalse(self.g.ground_response("alpha beta", "Alpha activates beta.")["grounded"])
        current = self.g.add_or_strengthen_edge("gamma", "cause", "delta", causal_kind="sufficient",
                  source="explicit", status="verified", conditions=("warm",), subject_type="signal",
                  target_type="event", observed_at="2026-01-02T03:04:05Z")
        self.g.add_or_strengthen_edge("gamma", "cause", "delta", causal_kind="sufficient", polarity=-1,
                                     source="opposition", conditions=("warm",))
        state = self.g.export_state_dict()
        restored = graph()
        restored.load_state_dict(state)
        for before, after in zip(state["edges"], restored.export_state_dict()["edges"], strict=True):
            self.assertEqual(set(before), set(after))
            for key, value in before.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(after[key], value, rtol=0, atol=0)
                else:
                    self.assertEqual(after[key], value)
        clone = next(e for e in restored.edges if e.edge_id == current.edge_id)
        self.assertEqual((clone.status, clone.subject_type, clone.target_type), ("verified", "signal", "event"))
        self.assertEqual(clone.conflict_ids, current.conflict_ids)

    def test_surface_synthesis_has_real_proofs_and_no_neural_claim(self):
        first = self.g.add_or_strengthen_edge("alpha", "cause", "beta", causal_kind="sufficient")
        self.g.add_or_strengthen_edge("beta", "cause", "gamma", causal_kind="sufficient")
        with patch.object(self.g.edge_valence_generator, "generate_trajectory_zero_pos", side_effect=AssertionError("generation")):
            response = self.g.synthesize_novel_sentence_from_graph("alpha gamma")
        self.assertFalse(response["synthesized_via_zero_pos_nmda_matrix"])
        self.assertEqual(response["rendering_mode"], "constrained_edge_surface")
        self.assertEqual(response["hops_traversed"], 2)
        self.assertEqual(len(response["proof_edge_ids"]), 2)
        self.assertIn(first.edge_id, response["proof_edge_ids"])
        self.assertEqual(response["synthesized_token_ids"], list(map(ord, response["synthesized_sentence"])))

    def test_ba10_uncertainty_and_required_failure_without_generator(self):
        self.g.extract_and_consolidate_sentence("Alpha releases beta.")
        self.g.add_or_strengthen_edge("beta", "is necessary for", "gamma", causal_kind="necessary")
        simulator = ENGINE.BA10CounterfactualInterventionSimulator()
        def forbidden(*args, **kwargs):
            raise AssertionError("Generator/training callback used as proof")
        args = (self.g, self.g.encode_fn, forbidden, forbidden, forbidden)
        with patch.object(self.g, "imprint_edges_into_dedicated_valence_matrix", side_effect=forbidden):
            result = simulator.simulate_counterfactual_intervention("What happens to gamma if alpha fails to release beta?", *args)
            self.assertEqual(result["intervention_result"]["interventions"], {"beta": False})
            self.assertIs(result["intervention_result"]["values"]["gamma"], False)
            self.assertTrue(result["proof_edge_ids"])
            # Alpha presence is not resolved by failure to produce beta.
            self.assertTrue(result["uncertain"])
            unknown = simulator.simulate_counterfactual_intervention("What happens to beta without alpha?", *args)
            self.assertTrue(unknown["uncertain"])
            self.assertIn("unknown", unknown["counterfactual_response"])
            self.assertIsNone(simulator.simulate_counterfactual_intervention("Describe alpha.", *args))
            explicit = simulator.simulate_counterfactual_intervention("What happens to gamma?", *args,
                                                                      interventions={"beta": False})
            self.assertFalse(explicit["uncertain"])
            self.assertIn("gamma is absent", explicit["counterfactual_response"])
        legacy = graph()
        legacy.load_state_dict([dict(subject="alpha", relation="cause", target="beta", conductance=100)])
        result = simulator.simulate_counterfactual_intervention("What happens to beta without alpha?", legacy,
                                                               legacy.encode_fn, forbidden, forbidden, forbidden)
        self.assertTrue(result["uncertain"])
        self.assertEqual(result["proof_edge_ids"], [])


class TemporalTests(unittest.TestCase):
    def test_binary_signed_timing_silence_and_amplitude_invariance(self):
        mux = ENGINE.ThetaGammaSpikingPhaseMultiplexer(d_vec=4, n_gamma_slots=1, bins_per_gamma=11, value_scale=1)
        name = mux.ROLE_SLOT_NAMES[0]
        vector = torch.tensor([1., -1., 0., .4])
        raster, windows = mux.encode_roles_into_theta_spike_train({name: vector})
        self.assertEqual(set(raster.unique().tolist()), {0., 1.})
        self.assertEqual(raster.sum(dim=0).tolist(), [1., 1., 1., 1.])
        self.assertEqual(windows[name], (0, 11))
        decoded, report = mux.decode_roles_from_theta_spike_train(raster, return_report=True)
        torch.testing.assert_close(decoded[name], vector)
        self.assertEqual(report["availability_fraction"], 1)
        modified = raster * torch.tensor([.01, -5., 1000., 2.])
        torch.testing.assert_close(mux.decode_roles_from_theta_spike_train(modified)[name], decoded[name])
        raster[0, 3] = 42  # An extra earlier event wins; no amplitude accumulation.
        duplicate, report = mux.decode_roles_from_theta_spike_train(raster, return_report=True)
        self.assertEqual(duplicate[name][3].item(), 1)
        self.assertEqual(report["roles"][name]["extra_events"], 1)
        silent, report = mux.decode_roles_from_theta_spike_train(torch.zeros_like(raster), return_report=True)
        self.assertEqual(silent[name].tolist(), [0., 0., 0., 0.])
        self.assertEqual(report["loss_fraction"], 1)

    def test_temporal_clean_jitter_drop_and_shuffle_degradation(self):
        mux = ENGINE.ThetaGammaSpikingPhaseMultiplexer()
        generator = torch.Generator().manual_seed(1907)
        roles = {name: F.normalize(torch.randn(64, generator=generator), dim=0) for name in mux.ROLE_SLOT_NAMES}
        raster, _ = mux.encode_roles_into_theta_spike_train(roles)
        self.assertEqual(int(raster.sum().item()), 384)
        def cosine(events):
            decoded = mux.decode_roles_from_theta_spike_train(events)
            return sum(F.cosine_similarity(roles[n], decoded[n], dim=0).item() for n in roles) / len(roles)
        clean = cosine(raster)
        jittered = torch.zeros_like(raster)
        shuffled = torch.zeros_like(raster)
        for slot in range(mux.n_gamma_slots):
            start = slot * mux.bins_per_gamma
            times, neurons = raster[start:start + mux.bins_per_gamma].nonzero(as_tuple=True)
            noise = torch.randint(-1, 2, times.shape, generator=generator)
            jittered[start + (times + noise).clamp(0, mux.bins_per_gamma-1), neurons] = 1
            shuffled[start:start + mux.bins_per_gamma] = raster[start:start + mux.bins_per_gamma][
                torch.randperm(mux.bins_per_gamma, generator=generator)]
        dropped = raster * (torch.rand(raster.shape, generator=generator) > .3)
        jitter, drop, shuffle = cosine(jittered), cosine(dropped), cosine(shuffled)
        self.assertGreater(clean, .99)
        self.assertGreater(jitter, .9)
        self.assertLess(jitter, clean)
        self.assertGreater(drop, .7)
        self.assertLess(drop, clean)
        self.assertLess(shuffle, .3)
        self.assertLess(shuffle, drop)
        _, loss = mux.decode_roles_from_theta_spike_train(dropped, return_report=True)
        self.assertGreater(loss["loss_fraction"], .2)
        self.assertLess(loss["loss_fraction"], .4)
        print(f"TTFS cosine clean={clean:.4f} jitter={jitter:.4f} drop={drop:.4f} shuffle={shuffle:.4f}")

    def test_configuration_roundtrip_and_input_validation(self):
        mux = ENGINE.ThetaGammaSpikingPhaseMultiplexer(8, 2, 41, .75)
        names = mux.ROLE_SLOT_NAMES[:2]
        roles = {n: torch.linspace(-.5, .5, 8) for n in names}
        raster, _ = mux.encode_roles_into_theta_spike_train(roles)
        fresh = ENGINE.ThetaGammaSpikingPhaseMultiplexer()
        fresh.load_state_dict(mux.export_state_dict())
        self.assertEqual(fresh.export_state_dict(), mux.export_state_dict())
        for name, vector in mux.decode_roles_from_theta_spike_train(raster).items():
            torch.testing.assert_close(fresh.decode_roles_from_theta_spike_train(raster)[name], vector)
        with self.assertRaises(ValueError):
            mux.encode_roles_into_theta_spike_train({names[0]: torch.zeros(7)})
        with self.assertRaises(ValueError):
            mux.encode_roles_into_theta_spike_train({"unknown": torch.zeros(8)})
        with self.assertRaises(ValueError):
            mux.decode_roles_from_theta_spike_train(torch.zeros(20, 8))
        with self.assertRaises(ValueError):
            ENGINE.ThetaGammaSpikingPhaseMultiplexer(n_gamma_slots=7)
        clipped = mux.encode_roles_into_theta_spike_train({names[0]: torch.ones(8)})[0]
        self.assertEqual(mux.last_encode_report["clipped_components"][names[0]], 8)
        self.assertEqual(int(clipped.sum().item()), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
