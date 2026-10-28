"""Supporting fixed-horizon Brier scores after the shared evaluation gates.

The primary evaluator owns protocol, outcome, prediction and training-support
validation. This additive report reuses its validated arrays and frozen patient
draws, without changing primary metric acceptance or tuning any time point.
"""
from collections import Counter
import hashlib

import numpy as np


def _support_failure(error, y_train, y_test, horizon):
    """Recognize only independently verified support errors from reference 0.25."""
    if type(error) is not ValueError:
        return False
    message = str(error)
    if message == "all samples are censored":
        return not bool(y_test["event"].any())
    if message.startswith("all times must be within follow-up time of test data: ["):
        return not float(y_test["time"].min()) <= horizon < float(y_test["time"].max())
    if message.startswith("time must be smaller than largest observed time point: "):
        return max(float(y_test["time"].max()), horizon) > float(y_train["time"].max())
    return False


def _interval(values, evaluation):
    sufficient = len(values) / evaluation["bootstrap_replicates"] >= evaluation["bootstrap_min_valid_fraction"]
    return {"valid_bootstrap_replicates": len(values),
            "percentile_95_ci": np.percentile(values, [2.5, 97.5]).tolist() if sufficient else None,
            "ci_status": "available" if sufficient else "insufficient_valid_replicates"}


def supporting_horizon_brier(y_train, y_test, fits, evaluation):
    """Report Brier at explicit horizons, using training-reference IPCW only.

    ``fits`` contains complete, already validated survival matrices in the
    evaluator's literal patient and required-time order. Reference support
    failures are reported per horizon; unexpected failures still propagate.
    """
    from sksurv.metrics import brier_score
    from sksurv.nonparametric import CensoringDistributionEstimator
    from .paired import require_reference

    require_reference()
    horizons = sorted(set(float(h) for h in [evaluation["primary_horizon"], *evaluation["secondary_horizons"]]))
    censoring = CensoringDistributionEstimator().fit(y_train)
    n = len(y_test)
    all_indices = np.arange(n)
    report, mode_draws, contrast_draws, reasons = {}, {}, {}, {}

    def scores(horizon, indices):
        column = evaluation["required_times"].index(horizon)
        values = {}
        for fit, matrix in fits.items():
            _, observed = brier_score(y_train, y_test[indices], matrix[indices, column], [horizon])
            score = float(observed[0])
            if not np.isfinite(score):
                raise ValueError("supporting Brier reference returned a nonfinite score")
            values["/".join(fit)] = score
        return values

    for horizon in horizons:
        key = str(horizon)
        # These are diagnostics, never a rule to alter the cohort or horizons.
        record = {"horizon": horizon, "status": "assessable", "scores": {},
            "mode_summary": {}, "paired_contrasts": {}, "unassessable_reason": None,
            "support": {"patients": n, "training_patients": len(y_train),
                "training_censoring_survival": float(censoring.predict_proba([horizon])[0]),
                "training_max_observed_time": float(y_train["time"].max()),
                "evaluation_min_observed_time": float(y_test["time"].min()),
                "evaluation_max_observed_time": float(y_test["time"].max()),
                "within_evaluation_follow_up": bool(y_test["time"].min() <= horizon < y_test["time"].max()),
                "events_by_horizon": int(np.sum(y_test["event"] & (y_test["time"] <= horizon))),
                "censored_by_horizon": int(np.sum(~y_test["event"] & (y_test["time"] <= horizon))),
                "observed_beyond_horizon": int(np.sum(y_test["time"] > horizon))}}
        try:
            record["scores"] = scores(horizon, all_indices)
        except ValueError as error:
            if not _support_failure(error, y_train, y_test, horizon):
                raise
            record.update(status="unassessable", unassessable_reason={
                "stage": "full_cohort_reference_brier", "type": type(error).__name__, "message": str(error)})
        report[key] = record
        mode_draws[key] = {mode: [] for mode in evaluation["modes"]}
        contrast_draws[key] = {f"{a}->{b}": [] for a, b in evaluation["contrasts"]}
        reasons[key] = Counter()

    # Regenerate the exact primary patient draws with a separate RNG. Horizon
    # support must not change the primary scores, accepted draws, or intervals.
    rng = np.random.default_rng(evaluation["bootstrap_seed"])
    digest = hashlib.sha256()
    for _ in range(evaluation["bootstrap_replicates"]):
        indices = rng.integers(0, n, size=n)
        digest.update(indices.astype("<i8").tobytes())
        for horizon in horizons:
            key = str(horizon)
            if report[key]["status"] != "assessable":
                continue
            try:
                replicate = scores(horizon, indices)
            except ValueError as error:
                if not _support_failure(error, y_train, y_test[indices], horizon):
                    raise
                reasons[key][f"{type(error).__name__}: {str(error)}"] += 1
                continue
            for mode in evaluation["modes"]:
                mode_draws[key][mode].append(float(np.mean([
                    replicate[f"{mode}/{seed}"] for seed in evaluation["mode_seeds"][mode]])))
            for a, b in evaluation["contrasts"]:
                contrast_draws[key][f"{a}->{b}"].append(float(np.mean([
                    replicate[f"{b}/{seed}"] - replicate[f"{a}/{seed}"]
                    for seed in evaluation["mode_seeds"][a]])))

    for key, record in report.items():
        evaluated = evaluation["bootstrap_replicates"] if record["status"] == "assessable" else 0
        record["bootstrap"] = {"evaluated": evaluated, "invalid": sum(reasons[key].values()),
            "valid": evaluated - sum(reasons[key].values()), "invalid_reasons": dict(reasons[key]),
            "not_evaluated": evaluation["bootstrap_replicates"] - evaluated}
        if record["status"] != "assessable":
            continue
        for mode in evaluation["modes"]:
            values = [record["scores"][f"{mode}/{seed}"] for seed in evaluation["mode_seeds"][mode]]
            record["mode_summary"][mode] = {"mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                **_interval(mode_draws[key][mode], evaluation)}
        for a, b in evaluation["contrasts"]:
            per_seed = {seed: record["scores"][f"{b}/{seed}"] - record["scores"][f"{a}/{seed}"]
                        for seed in evaluation["mode_seeds"][a]}
            record["paired_contrasts"][f"{a}->{b}"] = {
                "candidate_minus_comparator": float(np.mean(list(per_seed.values()))),
                "per_seed_differences": per_seed,
                "seed_difference_sample_sd": float(np.std(list(per_seed.values()), ddof=1)) if len(per_seed) > 1 else None,
                "favorable_direction": "negative", **_interval(contrast_draws[key][f"{a}->{b}"], evaluation)}
    return {"role": "supporting", "metric": "IPCW Brier score", "censoring_reference": "training",
            "lower_is_better": True, "horizons": report,
            "bootstrap": {"requested": evaluation["bootstrap_replicates"], "draws_sha256": digest.hexdigest(),
                "method": "same patient indices as primary across modes/seeds; support counted independently by horizon",
                "uncertainty_scope": "held-out patient sampling conditional on fixed trained models and training censoring reference"}}
