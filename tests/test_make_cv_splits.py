"""Unit tests for time-aware CV split stratification."""

from __future__ import annotations

import unittest

import numpy as np

from trifusesurv2.preprocessing.make_cv_splits import (
    _build_strata,
    stratified_kfold_indices,
    make_fold_splits,
)


class BuildStrataTest(unittest.TestCase):
    def test_event_only_fallback_when_times_is_none(self):
        events = np.array([0, 1, 0, 1, 0, 1])
        strata = _build_strata(events, times=None, n_time_bins=4)
        np.testing.assert_array_equal(strata, events)

    def test_event_only_fallback_when_n_bins_zero(self):
        events = np.array([0, 1, 0, 1])
        times = np.array([100.0, 200.0, 300.0, 400.0])
        strata = _build_strata(events, times=times, n_time_bins=0)
        np.testing.assert_array_equal(strata, events)

    def test_composite_strata_distinct_for_different_bins(self):
        events = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        times = np.array([10.0, 100.0, 500.0, 1000.0, 10.0, 100.0, 500.0, 1000.0])
        strata = _build_strata(events, times=times, n_time_bins=2)
        # Events and censored should have distinct strata
        event_strata = set(strata[:4].tolist())
        censor_strata = set(strata[4:].tolist())
        self.assertTrue(event_strata.isdisjoint(censor_strata))

    def test_too_few_events_falls_back(self):
        events = np.array([1, 0, 0, 0, 0])
        times = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        strata = _build_strata(events, times=times, n_time_bins=4)
        # Only 1 event, too few for 4 bins — should fall back to event-only
        np.testing.assert_array_equal(strata, events)


class TimeAwareKFoldTest(unittest.TestCase):
    def test_time_aware_folds_are_balanced(self):
        rng = np.random.default_rng(42)
        n = 100
        events = rng.integers(0, 2, size=n)
        times = rng.exponential(500, size=n)

        folds = stratified_kfold_indices(events, k=4, seed=1, times=times, n_time_bins=4)
        self.assertEqual(len(folds), 4)
        all_indices = sorted(idx for fold in folds for idx in fold)
        self.assertEqual(all_indices, list(range(n)))

    def test_backward_compat_no_times(self):
        events = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        folds = stratified_kfold_indices(events, k=2, seed=1)
        self.assertEqual(len(folds), 2)
        all_indices = sorted(idx for fold in folds for idx in fold)
        self.assertEqual(all_indices, list(range(8)))

    def test_make_fold_splits_with_times(self):
        rng = np.random.default_rng(42)
        n = 50
        events = rng.integers(0, 2, size=n)
        times = rng.exponential(365, size=n)

        splits = make_fold_splits(events, cv_folds=4, val_frac=0.2, split_seed=1,
                                  times=times, n_time_bins=4)
        self.assertEqual(len(splits), 4)
        for s in splits:
            self.assertIn("train", s)
            self.assertIn("val", s)
            self.assertIn("test", s)


if __name__ == "__main__":
    unittest.main()
