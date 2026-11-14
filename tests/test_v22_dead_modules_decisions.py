from __future__ import annotations

import unittest

from ziwago.config import GenesisConfig, MotorCfg
from ziwago.cortex.hierarchy import CorticalHierarchy
from ziwago.motor.pool import MotorPool


class DeadModuleDecisionTests(unittest.TestCase):
    def test_cortical_hierarchy_archived_by_default(self):
        with self.assertRaisesRegex(RuntimeError, "archived.*live path"):
            CorticalHierarchy(GenesisConfig())

    def test_cortical_hierarchy_requires_explicit_opt_in(self):
        h = CorticalHierarchy(GenesisConfig(), allow_experimental=True)
        self.assertIsNotNone(h)

    def test_motor_pool_archived_by_default(self):
        with self.assertRaisesRegex(RuntimeError, "archived.*live path"):
            MotorPool(4, 2, 2, 2, MotorCfg())

    def test_motor_pool_requires_explicit_opt_in(self):
        pool = MotorPool(4, 2, 2, 2, MotorCfg(), allow_experimental=True)
        self.assertEqual(pool.vocab, 4)


if __name__ == "__main__":
    unittest.main()
