"""Unit tests for survival utility edge cases."""

from __future__ import annotations

import math
import unittest

from trifusesurv2.utils.survival import comparable_pair_count, concordance_index, set_seed


class SurvivalUtilsTest(unittest.TestCase):
    def test_concordance_index_returns_nan_when_undefined(self):
        self.assertTrue(math.isnan(concordance_index([1.0], [1.0], [0.5])))
        self.assertTrue(math.isnan(concordance_index([1.0, 1.0], [1.0, 0.0], [0.5, 0.4])))
        self.assertTrue(math.isnan(concordance_index([1.0, 2.0], [0.0, 0.0], [0.5, 0.4])))

    def test_comparable_pair_count_matches_defined_pairs(self):
        self.assertEqual(comparable_pair_count([1.0, 2.0, 3.0], [1.0, 0.0, 1.0], [0.8, 0.2, 0.4]), 2)
        self.assertEqual(comparable_pair_count([1.0, 1.0], [1.0, 0.0], [0.8, 0.2]), 0)

    def test_set_seed_accepts_deterministic_mode(self):
        set_seed(123, deterministic=True)
        set_seed(123, deterministic=False)


if __name__ == "__main__":
    unittest.main()
