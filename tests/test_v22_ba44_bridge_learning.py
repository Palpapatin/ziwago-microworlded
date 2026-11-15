import ast
from pathlib import Path

import torch
import torch.nn.functional as F

from ziwago.cognition.language.syntax import BA44AttractorSyntaxBridge


ROOT = Path(__file__).resolve().parents[1]


def test_bridge_learning_uses_external_label_and_reports_disconnected_production():
    bridge = BA44AttractorSyntaxBridge(d_vec=64, seed=1444)
    pole1 = F.normalize(torch.randn(64, generator=torch.Generator().manual_seed(1)), dim=0)
    pole2 = F.normalize(torch.randn(64, generator=torch.Generator().manual_seed(2)), dim=0)
    stimulus = F.normalize(torch.randn(64, generator=torch.Generator().manual_seed(3)), dim=0)
    externally_labeled_connector = 2
    before = bridge.w_ba44_bridge.clone()
    before_cos = torch.dot(
        bridge._bridge_input(pole1, pole2, stimulus),
        before[:, externally_labeled_connector],
    )

    result = bridge.learn_bridge_transition(
        pole1,
        pole2,
        stimulus,
        connector_idx=externally_labeled_connector,
    )

    after = bridge.w_ba44_bridge
    after_cos = torch.dot(
        bridge._bridge_input(pole1, pole2, stimulus),
        after[:, externally_labeled_connector],
    )
    assert not torch.allclose(before[:, externally_labeled_connector], after[:, externally_labeled_connector])
    assert torch.allclose(before[:, :2], after[:, :2])
    assert torch.allclose(before[:, 3:], after[:, 3:])
    assert after_cos > before_cos
    assert result["learn_events"] == 1
    assert result["production_connection"] == "disconnected"
    assert result["supervision_source"] == "externally_supplied_connector_label"

    telemetry = bridge.bridge_learning_telemetry()
    assert telemetry["production_connection"] == "disconnected"
    assert telemetry["supervision_source"] == "externally_supplied_connector_label"
    connector = telemetry["connectors"][externally_labeled_connector]
    assert connector["learn_events"] == 1
    assert connector["last_cos"] == result["cos_after"]


def test_production_does_not_reference_external_only_bridge_learning_primitive():
    method_references = []
    for source_file in (ROOT / "ziwago").rglob("*.py"):
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "learn_bridge_transition":
                method_references.append(f"{source_file}:{node.lineno}")
            elif isinstance(node, ast.Name) and node.id == "learn_bridge_transition":
                method_references.append(f"{source_file}:{node.lineno}")

    assert method_references == []
