"""Synthetic engineering checks; these are not clinical/GPU qualification."""
import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
from sksurv.nonparametric import CensoringDistributionEstimator

from trifusesurv2.evaluation.curves import hazards_to_survival, write_survival_curves
from trifusesurv2.evaluation.paired import (
    calibration_table, censoring_support_table, evaluate_predictions, outcome_array,
    outcome_inventory_sha256, read_literal_csv, reference_metrics, validate_spec,
)


def fixture():
    spec = {"status": "frozen", "endpoint_definition": {"status": "confirmed",
            "time_origin": "synthetic baseline", "event_definition": "synthetic death",
            "censoring_rule": "synthetic last follow-up", "time_unit": "days"},
            "evaluation": {"endpoint": "OS", "fold": 0, "split": "test",
            "modes": ["F2", "F3"], "seeds": [42, 142, 242], "primary_horizon": 4.,
            "secondary_horizons": [6.], "concordance_tau": 7., "ibs_times": [2., 3., 4., 6.],
            "calibration_bins": [0., .5, 1.], "bootstrap_replicates": 80,
            "bootstrap_seed": 913, "bootstrap_min_valid_fraction": .5, "tied_tol": 1e-8}}
    outcomes = pd.DataFrame({"Patient_ID": [f"tr{i}" for i in range(12)] + ["val"] + ["001", "NA", "a", "b", "c", "d"],
                            "split": ["train"] * 12 + ["val"] + ["test"] * 6,
                            "endpoint": ["OS"] * 19,
                            "observed_time": list(range(1, 13)) + [8] + [2, 3, 5, 6, 8, 9],
                            "event": [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1] + [0] + [1, 0, 1, 1, 0, 1]})
    times = np.array(spec["evaluation"]["ibs_times"])
    probabilities = np.exp(-np.arange(1, 7)[:, None] * times[None, :] / 40)
    spec["evaluation"]["outcome_inventory_sha256"] = outcome_inventory_sha256(outcomes, "OS")
    frames = []
    target = outcomes.loc[outcomes["split"].eq("test")]
    with tempfile.TemporaryDirectory() as temp:
        for mode in spec["evaluation"]["modes"]:
            for seed in spec["evaluation"]["seeds"]:
                path = Path(temp) / f"{mode}_{seed}.csv"
                write_survival_curves(path, patient_ids=target["Patient_ID"].tolist(),
                    observed_times=target["observed_time"], events=target["event"],
                    evaluation_times=times, survival_probabilities=probabilities,
                    split="test", fold=0, seed=seed, mode=mode, endpoint="OS",
                    checkpoint_sha256="a" * 64, model_contract_sha256="b" * 64, risk_horizon=4.)
                frames.append(read_literal_csv(path))
    return pd.concat(frames, ignore_index=True), outcomes, spec


class PairedEvaluationTests(unittest.TestCase):
    def test_identical_models_are_zero_with_shared_draws_and_literal_ids(self):
        predictions, outcomes, spec = fixture()
        result = evaluate_predictions(predictions, outcomes, spec)
        for metric in result["paired_contrasts"]["F2->F3"].values():
            self.assertEqual(metric["candidate_minus_comparator"], 0)
            self.assertEqual(metric["seed_difference_sample_sd"], 0)
            self.assertEqual(metric["percentile_95_ci"], [0, 0])
        self.assertGreater(result["bootstrap"]["invalid"], 0)
        self.assertIn("001", predictions["Patient_ID"].tolist())
        self.assertIn("NA", predictions["Patient_ID"].tolist())

    def test_metric_backend_matches_reference_with_censoring_and_ties(self):
        y_train = outcome_array([1, 2, 2, 4, 6, 7, 9, 12], [1, 1, 0, 1, 0, 1, 0, 1])
        y_test = outcome_array([2, 2, 4, 5, 8], [1, 0, 1, 0, 1])
        times = np.array([2., 3., 6.])
        survival = np.exp(-np.array([3, 3, 2, 1, 1])[:, None] * times / 20)
        risk = 1 - survival[:, 1]
        actual = reference_metrics(y_train, y_test, survival, times, risk=risk, concordance_tau=7.)
        expected_c = concordance_index_ipcw(y_train, y_test, risk, tau=7.)
        self.assertEqual(actual["ipcw_concordance"], expected_c[0])
        self.assertEqual(actual["ibs"], integrated_brier_score(y_train, y_test, survival, times))
        self.assertGreater(actual["concordance_pair_counts"]["tied_risk"], 0)
        # Independent Brier formula makes use of the TRAINING censoring reference explicit.
        censor = CensoringDistributionEstimator().fit(y_train)
        numerator = denominator = 0.
        for i in range(len(y_test)):
            if not y_test["event"][i] or y_test["time"][i] >= 7:
                continue
            weight = 1 / censor.predict_proba([y_test["time"][i]])[0] ** 2
            for j in range(len(y_test)):
                comparable = y_test["time"][j] > y_test["time"][i] or (
                    y_test["time"][j] == y_test["time"][i] and not y_test["event"][j])
                if comparable:
                    denominator += weight
                    numerator += weight * (.5 if abs(risk[i] - risk[j]) <= 1e-8 else float(risk[i] > risk[j]))
        self.assertAlmostEqual(actual["ipcw_concordance"], numerator / denominator)
        manual = []
        for j, time in enumerate(times):
            cases = y_test["event"] & (y_test["time"] <= time)
            controls = y_test["time"] > time
            score = np.sum(survival[cases, j] ** 2 / censor.predict_proba(y_test["time"][cases]))
            score += np.sum((1 - survival[controls, j]) ** 2 / censor.predict_proba([time])[0])
            manual.append(score / len(y_test))
        np.testing.assert_allclose(actual["brier"], manual, rtol=1e-12)
        self.assertAlmostEqual(actual["ibs"], np.trapz(manual, times) / (times[-1] - times[0]))

    def test_uncensored_risk_direction_and_tied_risk(self):
        train = outcome_array([1, 2, 3, 4, 5, 10], [1] * 6)
        test = outcome_array([1, 2, 3, 4, 5], [1] * 5)
        survival = np.full((5, 2), .5)
        for risk, expected in (([5, 4, 3, 2, 1], 1.), ([1, 2, 3, 4, 5], 0.), ([1] * 5, .5)):
            actual = reference_metrics(train, test, survival, [1., 4.], risk=risk, concordance_tau=5.)
            self.assertEqual(actual["ipcw_concordance"], expected)

    def test_rejects_invalid_prediction_contracts(self):
        predictions, outcomes, spec = fixture()
        cases = []
        cases.append(pd.concat([predictions, predictions.iloc[:1]], ignore_index=True))
        cases.append(predictions.loc[~predictions["Patient_ID"].eq("001")])
        cases.append(predictions.loc[~predictions["seed"].eq("242")])
        cases.append(predictions.loc[~predictions["evaluation_time"].astype(float).eq(3)])
        for column, value in (("survival_probability", "NaN"), ("checkpoint_sha256", "unknown"),
                              ("risk_score", ".999"), ("observed_time", "999"), ("event", "2")):
            bad = predictions.copy()
            bad.loc[0, column] = value
            cases.append(bad)
        bad = predictions.copy()
        bad.loc[1, "survival_probability"] = "1"
        cases.append(bad)
        for bad in cases:
            with self.subTest(case=len(bad)), self.assertRaises(ValueError):
                evaluate_predictions(bad, outcomes, spec)

    def test_unconfirmed_endpoint_cannot_be_scored(self):
        _, _, spec = fixture()
        spec["endpoint_definition"]["time_origin"] = ""
        with self.assertRaisesRegex(ValueError, "time_origin"):
            validate_spec(spec)

    def test_joint_patient_dropping_and_censoring_reference_changes_fail(self):
        predictions, outcomes, spec = fixture()
        with self.assertRaisesRegex(ValueError, "frozen cohort"):
            evaluate_predictions(predictions.loc[~predictions["Patient_ID"].eq("001")],
                                 outcomes.loc[~outcomes["Patient_ID"].eq("001")], spec)
        outcomes.loc[outcomes["Patient_ID"].eq("tr0"), "event"] = 0
        with self.assertRaisesRegex(ValueError, "frozen cohort"):
            evaluate_predictions(predictions, outcomes, spec)
        spec["status"] = "draft"
        with self.assertRaisesRegex(ValueError, "frozen"):
            validate_spec(spec)

    def test_outcome_hash_preserves_decimal_csv_round_trip(self):
        _, outcomes, _ = fixture()
        outcomes["observed_time"] = outcomes["observed_time"].astype(float)
        outcomes.loc[0, "observed_time"] = 0.12345678901234567
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outcomes.csv"
            outcomes.to_csv(path, index=False)
            self.assertEqual(outcome_inventory_sha256(outcomes, "OS"),
                             outcome_inventory_sha256(read_literal_csv(path), "OS"))

    def test_cli_support_and_unverified_artifact_status(self):
        predictions, outcomes, spec = fixture()
        spec["evaluation"]["bootstrap_replicates"] = 4
        script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_paired_survival.py"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "spec.json").write_text(json.dumps(spec))
            predictions.to_csv(tmp / "predictions.csv", index=False)
            outcomes.to_csv(tmp / "outcomes.csv", index=False)
            base = [sys.executable, str(script), "--spec", str(tmp / "spec.json"),
                    "--outcomes", str(tmp / "outcomes.csv")]
            subprocess.run([*base, "--support-only", "--out-dir", str(tmp / "support")],
                           check=True, capture_output=True, text=True)
            support = json.loads((tmp / "support" / "development_support.json").read_text())
            self.assertEqual({row["split"] for row in support["support"]}, {"train", "val"})
            subprocess.run([*base, "--predictions", str(tmp / "predictions.csv"),
                            "--out-dir", str(tmp / "scores")], check=True, capture_output=True, text=True)
            scores = json.loads((tmp / "scores" / "paired_evaluation.json").read_text())
            self.assertEqual(scores["status"], "engineering_evaluation_unverified_artifacts")
            self.assertIn("source_tree_sha256", scores["source_identity"])
            self.assertEqual(scores["dependencies"]["scikit-survival"], "0.25.0")

    def test_contrast_averages_per_seed_metrics(self):
        predictions, outcomes, spec = fixture()
        selected = predictions["mode"].eq("F3") & predictions["seed"].eq("42")
        part = predictions.loc[selected].copy()
        ids = ["001", "NA", "a", "b", "c", "d"]
        rates = {pid: rate for pid, rate in zip(ids, range(6, 0, -1))}
        part["survival_probability"] = [str(np.exp(-rates[pid] * float(time) / 40))
                                           for pid, time in zip(part["Patient_ID"], part["evaluation_time"])]
        part["risk_score"] = [str(1 - np.exp(-rates[pid] * 4 / 40)) for pid in part["Patient_ID"]]
        predictions.loc[selected, :] = part
        result = evaluate_predictions(predictions, outcomes, spec)
        contrast = result["paired_contrasts"]["F2->F3"]["ipcw_concordance"]
        delta42 = result["scores"]["F3/42"]["ipcw_concordance"] - result["scores"]["F2/42"]["ipcw_concordance"]
        self.assertNotEqual(delta42, 0)
        self.assertAlmostEqual(contrast["candidate_minus_comparator"], delta42 / 3)
        self.assertGreater(contrast["seed_difference_sample_sd"], 0)

    def test_support_ignores_test_outcomes_and_rejects_zero_censoring_support(self):
        _, outcomes, _ = fixture()
        before = censoring_support_table(outcomes, "OS", [4., 6.])
        outcomes.loc[outcomes["split"].eq("test"), "observed_time"] = -999
        self.assertEqual(before, censoring_support_table(outcomes, "OS", [4., 6.]))
        unsupported = censoring_support_table(outcomes, "OS", [100.])
        self.assertTrue(all(not row["censoring_supported"] for row in unsupported))

    def test_calibration_does_not_label_early_censoring_as_event_free(self):
        train = outcome_array([1, 2, 3, 5, 8, 10], [1, 0, 1, 1, 0, 1])
        test = outcome_array([1, 2, 4, 7], [1, 0, 1, 1])
        rows = calibration_table(train, test, np.full(4, .5), 5., [0., 1.])
        expected = (1 / 1.0 + 1 / .8) / 4
        self.assertAlmostEqual(rows[0]["ipcw_observed_event_probability"], expected)
        self.assertEqual(rows[0]["censored_by_horizon"], 1)

    def test_curve_grid_uses_physical_bin_width_and_rejects_extrapolation(self):
        got = hazards_to_survival([[.2, .5]], [0, 5, 10, 15, 20], 10)
        np.testing.assert_allclose(got, [[1, np.sqrt(.8), .8, .8 * np.sqrt(.5), .4]])
        with self.assertRaisesRegex(ValueError, "coverage"):
            hazards_to_survival([[.2, .5]], [21], 10)

    def test_distinct_baseline_seed_inventory_is_allowed_without_fake_pairing(self):
        predictions, outcomes, spec = fixture()
        spec["evaluation"]["modes"].append("C0")
        spec["evaluation"]["mode_seeds"] = {"F2": [42, 142, 242], "F3": [42, 142, 242], "C0": ["deterministic"]}
        spec["evaluation"]["contrasts"] = [["F2", "F3"]]
        baseline = predictions.loc[predictions["mode"].eq("F2") & predictions["seed"].eq("42")].copy()
        baseline["mode"], baseline["seed"] = "C0", "deterministic"
        report = evaluate_predictions(pd.concat([predictions, baseline]), outcomes, spec)
        self.assertIn("C0/deterministic", report["scores"])
        self.assertIsNone(report["mode_summary"]["C0"]["ibs"]["sample_sd"])


if __name__ == "__main__":
    unittest.main()
