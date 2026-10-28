"""Fixed-horizon supporting scores on fabricated survival outcomes only."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sksurv.metrics import brier_score
from sksurv.nonparametric import CensoringDistributionEstimator

from test_paired_survival_evaluation import fixture
from trifusesurv2.evaluation.horizon_brier import supporting_horizon_brier
from trifusesurv2.evaluation.paired import (
    _validated_predictions, evaluate_predictions, outcome_array, validate_outcomes, validate_spec,
)


def horizon_fixture(horizons=(5.5, 8.5), changed_seed=False):
    predictions, outcomes, spec = fixture()
    spec['evaluation']['secondary_horizons'] = list(horizons)
    rates = dict(zip(['001', 'NA', 'a', 'b', 'c', 'd'], range(1, 7)))
    extra = []
    for (_, _, pid), part in predictions.groupby(['mode', 'seed', 'Patient_ID']):
        for horizon in horizons:
            if horizon not in part['evaluation_time'].astype(float).tolist():
                row = part.iloc[0].to_dict()
                row['evaluation_time'] = str(horizon)
                row['survival_probability'] = str(np.exp(-rates[pid] * horizon / 40))
                extra.append(row)
    predictions = pd.concat([predictions, pd.DataFrame(extra)], ignore_index=True)
    if changed_seed:
        mask = predictions['mode'].eq('F3') & predictions['seed'].eq('42')
        selected = predictions.loc[mask].copy()
        selected['survival_probability'] = [str(np.exp(-(7 - rates[pid]) * float(t) / 40))
                                            for pid, t in zip(selected.Patient_ID, selected.evaluation_time)]
        selected['risk_score'] = [str(1 - np.exp(-(7 - rates[pid]) * 4 / 40)) for pid in selected.Patient_ID]
        predictions.loc[mask] = selected
    return predictions, outcomes, spec


def validated_arrays(predictions, outcomes, spec):
    ev = validate_spec(spec)
    outcomes = validate_outcomes(outcomes, 'OS')
    target, fits, _ = _validated_predictions(predictions, outcomes, ev)
    training = outcomes.loc[outcomes['split'].eq('train')]
    return outcome_array(training.observed_time, training.event), outcome_array(target.observed_time, target.event), fits, ev


class SupportingHorizonBrierTests(unittest.TestCase):
    def test_outside_ibs_grid_matches_reference_and_manual_fixed_training_bootstrap(self):
        predictions, outcomes, spec = horizon_fixture()
        report = evaluate_predictions(predictions, outcomes, spec)
        supporting = report['supporting_horizon_brier']
        self.assertEqual(set(supporting['horizons']), {'4.0', '5.5', '8.5'})
        self.assertEqual(supporting['bootstrap']['draws_sha256'], report['bootstrap']['draws_sha256'])
        y_train, y_test, fits, ev = validated_arrays(predictions, outcomes, spec)
        horizon = 8.5
        probabilities = fits[('F2', '42')][:, ev['required_times'].index(horizon)]
        result = supporting['horizons']['8.5']
        self.assertEqual(result['scores']['F2/42'], float(brier_score(y_train, y_test, probabilities, [horizon])[1][0]))
        censor = CensoringDistributionEstimator().fit(y_train)
        g_horizon = censor.predict_proba([horizon])[0]

        def manual(indices):
            times, events, survival = y_test['time'][indices], y_test['event'][indices], probabilities[indices]
            cases = events & (times <= horizon)
            controls = times > horizon
            event_term = np.sum(survival[cases] ** 2 / censor.predict_proba(times[cases])) if cases.any() else 0.
            return float((event_term + np.sum((1 - survival[controls]) ** 2 / g_horizon)) / len(indices))

        self.assertAlmostEqual(result['scores']['F2/42'], manual(np.arange(len(y_test))))
        rng = np.random.default_rng(ev['bootstrap_seed'])
        expected = []
        for _ in range(ev['bootstrap_replicates']):
            indices = rng.integers(0, len(y_test), size=len(y_test))
            times = y_test['time'][indices]
            if times.min() <= horizon < times.max() and y_test['event'][indices].any():
                expected.append(manual(indices))
        self.assertEqual(result['bootstrap']['valid'], len(expected))
        self.assertGreater(result['bootstrap']['invalid'], 0)
        np.testing.assert_allclose(result['mode_summary']['F2']['percentile_95_ci'], np.percentile(expected, [2.5, 97.5]))
        for record in supporting['horizons'].values():
            contrast = record['paired_contrasts']['F2->F3']
            self.assertEqual(contrast['candidate_minus_comparator'], 0)
            self.assertEqual(contrast['percentile_95_ci'], [0, 0])
            self.assertEqual(contrast['favorable_direction'], 'negative')
        json.dumps(report, allow_nan=False)

    def test_supporting_failure_preserves_every_existing_primary_output(self):
        predictions, outcomes, spec = horizon_fixture((5.5, 10.))
        with patch('trifusesurv2.evaluation.horizon_brier.supporting_horizon_brier', return_value={}):
            before = evaluate_predictions(predictions, outcomes, spec)
        after = evaluate_predictions(predictions, outcomes, spec)
        supporting = after.pop('supporting_horizon_brier')
        before.pop('supporting_horizon_brier')
        self.assertEqual(json.dumps(before, sort_keys=True, allow_nan=False), json.dumps(after, sort_keys=True, allow_nan=False))
        unavailable = supporting['horizons']['10.0']
        self.assertEqual(unavailable['status'], 'unassessable')
        self.assertEqual(unavailable['scores'], {})
        self.assertEqual(unavailable['mode_summary'], {})
        self.assertEqual(unavailable['paired_contrasts'], {})
        self.assertEqual(unavailable['unassessable_reason']['type'], 'ValueError')
        self.assertIn('within follow-up', unavailable['unassessable_reason']['message'])
        self.assertEqual(unavailable['support']['evaluation_max_observed_time'], 9)
        self.assertFalse(unavailable['support']['within_evaluation_follow_up'])
        self.assertEqual(unavailable['bootstrap']['not_evaluated'], spec['evaluation']['bootstrap_replicates'])
        self.assertEqual(unavailable['bootstrap']['invalid'], 0)
        self.assertEqual(supporting['horizons']['5.5']['status'], 'assessable')

    def test_supporting_validity_does_not_require_primary_bootstrap_validity(self):
        predictions, outcomes, spec = horizon_fixture()
        report = evaluate_predictions(predictions, outcomes, spec)
        supporting = report['supporting_horizon_brier']
        primary_valid = report['mode_summary']['F2']['ibs']['valid_bootstrap_replicates']
        horizon_valid = supporting['horizons']['5.5']['bootstrap']['valid']
        self.assertGreater(horizon_valid, primary_valid)

    def test_per_seed_contrasts_are_averaged_without_pooling_predictions(self):
        predictions, outcomes, spec = horizon_fixture(changed_seed=True)
        report = evaluate_predictions(predictions, outcomes, spec)
        record = report['supporting_horizon_brier']['horizons']['5.5']
        difference = record['scores']['F3/42'] - record['scores']['F2/42']
        contrast = record['paired_contrasts']['F2->F3']
        self.assertNotEqual(difference, 0)
        self.assertAlmostEqual(contrast['candidate_minus_comparator'], difference / 3)
        self.assertGreater(contrast['seed_difference_sample_sd'], 0)
        self.assertEqual(contrast['per_seed_differences']['142'], 0)

    def test_insufficient_horizon_draws_keep_point_scores_without_interval(self):
        predictions, outcomes, spec = horizon_fixture()
        spec['evaluation']['bootstrap_min_valid_fraction'] = 1.
        report = evaluate_predictions(predictions, outcomes, spec)
        record = report['supporting_horizon_brier']['horizons']['8.5']
        self.assertEqual(record['status'], 'assessable')
        self.assertGreater(record['bootstrap']['invalid'], 0)
        self.assertIsNone(record['mode_summary']['F2']['percentile_95_ci'])
        self.assertEqual(record['paired_contrasts']['F2->F3']['ci_status'], 'insufficient_valid_replicates')

    def test_unknown_or_inconsistent_reference_errors_are_not_labeled_support_failures(self):
        arrays = validated_arrays(*horizon_fixture())
        for error in ('malformed survival matrix', 'all samples are censored',
                      'all times must be within follow-up time of test data: [bogus]'):
            with self.subTest(error=error), patch('sksurv.metrics.brier_score', side_effect=ValueError(error)), \
                    self.assertRaisesRegex(ValueError, error.split(':')[0]):
                supporting_horizon_brier(*arrays)

    def test_censoring_support_and_prediction_contract_failures_remain_fatal(self):
        predictions, outcomes, spec = horizon_fixture((13.,))
        with self.assertRaisesRegex(ValueError, 'training censoring support'):
            evaluate_predictions(predictions, outcomes, spec)
        predictions, outcomes, spec = horizon_fixture()
        missing = predictions.loc[~predictions['evaluation_time'].astype(float).eq(8.5)]
        with self.assertRaisesRegex(ValueError, 'required evaluation time'):
            evaluate_predictions(missing, outcomes, spec)

    def test_all_censored_draws_are_counted_and_deterministic_baseline_has_no_seed_sd(self):
        y_train, _, _, ev = validated_arrays(*horizon_fixture((6.,)))
        y_test = outcome_array([2, 3, 6, 9], [1, 0, 0, 0])
        ev.update(modes=['C0'], mode_seeds={'C0': ['deterministic']}, contrasts=[])
        fits = {('C0', 'deterministic'): np.exp(-np.asarray(ev['required_times'])[None, :] / np.arange(10, 50, 10)[:, None])}
        report = supporting_horizon_brier(y_train, y_test, fits, ev)
        record = report['horizons']['4.0']
        self.assertIn('ValueError: all samples are censored', record['bootstrap']['invalid_reasons'])
        self.assertIsNone(record['mode_summary']['C0']['sample_sd'])
        self.assertEqual(record['paired_contrasts'], {})


if __name__ == '__main__':
    unittest.main()
