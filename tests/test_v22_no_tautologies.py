from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from typing import get_type_hints

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ziwago.core.criticality import CriticalityMonitor
from ziwago.core.interoception import InteroceptiveField
from ziwago.core.sdr import HashedCorticalSDREncoder, LCASparseEncoder
from ziwago.cortex.next16_core import FusiZenkeCascadeMetaplasticityShield


class TautologyGuardTests(unittest.TestCase):
    def _iter_banned(self, source: str):
        tree = ast.parse(source)
        offenders = []

        def same_expr(left, right):
            return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)

        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                expr = node.test
                if isinstance(expr, ast.Constant) and expr.value in (True, False, None, 1, 0):
                    offenders.append((node.lineno, "literal assert"))
                    continue
                if isinstance(expr, ast.Compare) and len(expr.ops) == 1:
                    if isinstance(expr.ops[0], (ast.Eq, ast.Is)) and same_expr(expr.left, expr.comparators[0]):
                        offenders.append((node.lineno, "self-comparison"))
                if isinstance(expr, ast.BoolOp):
                    if all(isinstance(v, ast.Constant) and v.value in (True, False) for v in expr.values):
                        offenders.append((node.lineno, "boolean tautology"))
            elif isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
                if same_expr(node.left, node.comparators[0]):
                    offenders.append((node.lineno, "self-comparison"))
        return offenders

    def test_tests_directory_has_no_tautologies(self):
        test_root = ROOT / "tests"
        offenders = []
        for py in sorted(test_root.glob("*.py")):
            if py.name == Path(__file__).name:
                continue
            source = py.read_text(encoding="utf-8")
            offenders.extend((py.name, lineno, reason) for lineno, reason in self._iter_banned(source))
        if offenders:
            details = "; ".join(f"{path}:{lineno}:{reason}" for path, lineno, reason in offenders[:10])
            self.fail(f"tautology guard caught banned patterns in tests: {details}")

    def test_public_core_annotations_resolve(self):
        for cls in (CriticalityMonitor, InteroceptiveField, LCASparseEncoder, HashedCorticalSDREncoder):
            hints = get_type_hints(cls)
            self.assertIsInstance(hints, dict)

    def test_metaplasticity_shield_method_annotations_resolve(self):
        for method in (
            FusiZenkeCascadeMetaplasticityShield.stats,
            FusiZenkeCascadeMetaplasticityShield.decay_trace,
        ):
            self.assertIn("return", get_type_hints(method))


if __name__ == "__main__":
    unittest.main()
