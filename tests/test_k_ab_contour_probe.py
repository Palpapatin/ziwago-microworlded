from pathlib import Path

from tools.probes.ab_contour_probe import (
    build_plan_report,
    detect_contour_identity,
)


ROOT = Path(__file__).resolve().parents[1]


def test_plan_marks_unavailable_metrics_without_inventing_values():
    report = build_plan_report()

    assert report["execution"] == {
        "performed": False,
        "model_loaded": False,
        "organism_constructed": False,
        "training_performed": False,
        "heavy_probes_run": False,
    }
    assert report["readiness"]["ab_possible"] is False
    for mode in ("closed_book", "open_book", "overall_acceptance"):
        metric = report["acceptance"][mode]
        assert metric["status"] == "not_measured"
        assert metric["value"] is None
        assert metric["reason"]
    for condition in report["compose"]["conditions"].values():
        assert condition["status"] == "not_measured"
        assert condition["case_count"] is None
        assert condition["passes"] is None
        assert condition["rate"] is None
        assert condition["reason"]
    assert report["memory_profile"]["peak_commit_bytes"] is None
    assert report["memory_profile"]["peak_private_bytes"] is None
    for phase in report["module_call_map"]["phases"].values():
        assert phase["status"] == "not_measured"
        assert phase["call_count"] is None
        assert phase["unique_functions"] is None
        assert phase["reason"]


def test_static_detector_finds_current_unified_contour_and_checkpoint_formats():
    identity = detect_contour_identity(
        (ROOT / "ziwago" / "organism_core.py").read_text(encoding="utf-8"),
        (ROOT / "ziwago" / "organism" / "flat.py").read_text(encoding="utf-8"),
    )

    assert identity["status"] == "single_unified_contour"
    assert identity["integrated_parameter_present"] is True
    assert identity["integrated_argument_selects_biology"] is False
    assert identity["unconditional_integrated_setup"] is True
    assert identity["checkpoint_version_semantics"]["status"] == (
        "checkpoint_formats_not_biological_contours"
    )
    assert any(
        row["finding"] == "integrated is stored as compatibility metadata"
        for row in identity["evidence"]
    )
    assert any(
        row["finding"] == "integrated biological setup is called during core construction"
        for row in identity["evidence"]
    )


def test_static_detector_does_not_call_a_real_integrated_switch_unified():
    organism_source = """
class ZiwagoOrganism:
    def __init__(self, integrated=True):
        if integrated:
            self.core = IntegratedCore()
        else:
            self.core = LegacyCore()
"""
    flat_source = """
class ZiwagoPureSynapticDialogueOrganismV14:
    def __init__(self):
        pass
"""

    identity = detect_contour_identity(organism_source, flat_source)

    assert identity["status"] == "distinct_or_undetermined"
    assert identity["integrated_argument_selects_biology"] is None
