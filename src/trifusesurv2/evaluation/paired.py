"""Paired survival evaluation on a fixed patient inventory and time grid.

The scoring backend is the version-pinned scikit-survival reference, not the
trainer's historical diagnostic IBS. Bootstrap draws resample held-out patients
once and reuse those indices for every mode and seed. They do not refit models.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import importlib.metadata
import json

import numpy as np
import pandas as pd

from .curves import CURVE_COLUMNS, require_digest

REFERENCE_VERSION = "0.25.0"


def require_reference():
    actual = importlib.metadata.version("scikit-survival")
    if actual != REFERENCE_VERSION:
        raise ValueError(f"reference version mismatch: require scikit-survival {REFERENCE_VERSION}, got {actual}")
    return actual


def read_literal_csv(path):
    # Disables pandas NA heuristics (e.g. literal 'NA') and leading-zero removal.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def outcome_array(times, events):
    from sksurv.util import Surv
    times, events = np.asarray(times, dtype=float), np.asarray(events, dtype=float)
    if not np.isfinite(times).all() or np.any(times < 0) or not np.isin(events, [0, 1]).all():
        raise ValueError("outcomes require finite nonnegative times and binary events")
    return Surv.from_arrays(events.astype(bool), times, name_event="event", name_time="time")


def validate_outcomes(frame, endpoint):
    required = {"Patient_ID", "split", "endpoint", "observed_time", "event"}
    if not required <= set(frame):
        raise ValueError(f"outcomes missing columns: {sorted(required - set(frame))}")
    out = frame.copy()
    if not out["endpoint"].eq(endpoint).all():
        raise ValueError("outcome endpoint does not match the declared endpoint")
    ids = out["Patient_ID"]
    if any(not isinstance(x, str) or not x or x.strip() != x for x in ids) or ids.duplicated().any():
        raise ValueError("duplicate, missing, or modified patient IDs in outcomes")
    if not set(out["split"]) <= {"train", "val", "test"}:
        raise ValueError("unknown outcome split")
    # Python's float parser preserves CSV shortest-round-trip representations;
    # pandas.to_numeric can round long decimal strings differently by one ULP.
    out["observed_time"] = [float(value) for value in out["observed_time"]]
    out["event"] = [float(value) for value in out["event"]]
    outcome_array(out["observed_time"], out["event"])
    return out


def outcome_inventory_sha256(frame, endpoint):
    """Canonical binding of all development/held-out labels and split membership."""
    out = validate_outcomes(frame, endpoint).sort_values("Patient_ID")
    records = [{"Patient_ID": row.Patient_ID, "split": row.split, "endpoint": row.endpoint,
                "observed_time": float(row.observed_time), "event": int(row.event)}
               for row in out.itertuples(index=False)]
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_spec(spec, *, frozen=True):
    if frozen:
        if spec.get("status") != "frozen" or spec.get("endpoint_definition", {}).get("status") != "confirmed":
            raise ValueError("scientific evaluation requires a frozen protocol and confirmed endpoint definition")
        definition = spec["endpoint_definition"]
        for key in ("time_origin", "event_definition", "censoring_rule", "time_unit"):
            if not isinstance(definition.get(key), str) or not definition[key].strip():
                raise ValueError(f"confirmed endpoint definition is missing {key}")
    ev = copy.deepcopy(spec.get("evaluation", {}))
    required = {"endpoint", "fold", "split", "modes", "seeds", "primary_horizon",
                "secondary_horizons", "concordance_tau", "ibs_times", "calibration_bins",
                "bootstrap_replicates", "bootstrap_seed", "bootstrap_min_valid_fraction", "tied_tol"}
    if not required <= set(ev):
        raise ValueError(f"evaluation specification missing: {sorted(required - set(ev))}")
    if frozen:
        require_digest(ev.get("outcome_inventory_sha256"), "evaluation.outcome_inventory_sha256")
    grid = np.asarray(ev["ibs_times"], dtype=float)
    horizons = np.asarray([ev["primary_horizon"], *ev["secondary_horizons"]], dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all() or grid[0] <= 0 or np.any(np.diff(grid) <= 0):
        raise ValueError("IBS grid must contain at least two increasing finite positive times")
    if not np.isfinite(horizons).all() or np.any(horizons <= 0):
        raise ValueError("invalid declared horizons")
    tau = float(ev["concordance_tau"])
    if not np.isfinite(tau) or tau <= 0 or ev["split"] not in {"val", "test"}:
        raise ValueError("invalid concordance truncation or evaluation split")
    if not ev["modes"] or len(ev["modes"]) != len(set(ev["modes"])):
        raise ValueError("modes must be unique and nonempty")
    mode_seeds = ev.get("mode_seeds", {mode: ev["seeds"] for mode in ev["modes"]})
    if set(mode_seeds) != set(ev["modes"]):
        raise ValueError("mode_seeds must exactly cover modes")
    for mode, seeds in mode_seeds.items():
        seeds = [str(x) for x in seeds]
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError(f"seeds must be unique and nonempty for {mode}")
        mode_seeds[mode] = seeds
    ev["mode_seeds"] = mode_seeds
    contrasts = ev.get("contrasts", [ev["modes"][:2]] if len(ev["modes"]) >= 2 else [])
    for pair in contrasts:
        if len(pair) != 2 or pair[0] == pair[1] or any(mode not in mode_seeds for mode in pair):
            raise ValueError("invalid ordered comparator/candidate contrast")
        if mode_seeds[pair[0]] != mode_seeds[pair[1]]:
            raise ValueError("paired contrasts require the same ordered seed inventory")
    ev["contrasts"] = contrasts
    bins = np.asarray(ev["calibration_bins"], dtype=float)
    if len(bins) < 2 or bins[0] != 0 or bins[-1] != 1 or not np.isfinite(bins).all() or np.any(np.diff(bins) <= 0):
        raise ValueError("calibration_bins must be prespecified increasing risk edges from 0 to 1")
    if isinstance(ev["bootstrap_replicates"], bool) or int(ev["bootstrap_replicates"]) != ev["bootstrap_replicates"] or ev["bootstrap_replicates"] < 1:
        raise ValueError("bootstrap_replicates must be a positive integer")
    if not 0 < float(ev["bootstrap_min_valid_fraction"]) <= 1 or not np.isfinite(ev["tied_tol"]) or ev["tied_tol"] < 0:
        raise ValueError("invalid bootstrap acceptance fraction or tie tolerance")
    ev["required_times"] = sorted(set(grid.tolist() + horizons.tolist()))
    return ev


def censoring_support_table(outcomes, endpoint, horizons):
    """Training/validation support only; test outcomes and predictions are unused."""
    from sksurv.nonparametric import CensoringDistributionEstimator
    require_reference()
    development = outcomes.loc[outcomes["split"].isin(["train", "val"])].copy()
    development = validate_outcomes(development, endpoint)
    train = development.loc[development["split"].eq("train")]
    if train.empty:
        raise ValueError("training outcomes are required for censoring support")
    y_train = outcome_array(train["observed_time"], train["event"])
    estimator = CensoringDistributionEstimator().fit(y_train)
    result = []
    for horizon in sorted(set(float(x) for x in horizons)):
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("support horizon must be finite and positive")
        try:
            g = float(estimator.predict_proba([horizon])[0])
            supported = bool(g > 0 and horizon < float(y_train["time"].max()))
        except ValueError:
            g, supported = None, False
        for split in ("train", "val"):
            part = development.loc[development["split"].eq(split)]
            t, e = part["observed_time"].to_numpy(float), part["event"].to_numpy(int)
            result.append({"split": split, "horizon": horizon, "n": len(part),
                           "at_risk_at_horizon": int(np.sum(t >= horizon)),
                           "observed_beyond_horizon": int(np.sum(t > horizon)),
                           "events_by_horizon": int(np.sum((e == 1) & (t <= horizon))),
                           "censored_by_horizon": int(np.sum((e == 0) & (t <= horizon))),
                           "training_censoring_survival": g, "censoring_supported": supported})
    return result


def reference_metrics(y_train, y_test, survival, times, *, risk, concordance_tau, tied_tol=1e-8):
    """Delegate to the pinned reference; unsupported follow-up fails explicitly."""
    from sksurv.metrics import brier_score, concordance_index_ipcw, integrated_brier_score
    require_reference()
    c = concordance_index_ipcw(y_train, y_test, risk, tau=concordance_tau, tied_tol=tied_tol)
    ibs = integrated_brier_score(y_train, y_test, survival, times)
    _, brier = brier_score(y_train, y_test, survival, times)
    if not np.isfinite([c[0], ibs, *brier]).all():
        raise ValueError("reference metrics are nonfinite (unsupported censoring/comparable pairs)")
    return {"ipcw_concordance": float(c[0]), "ibs": float(ibs),
            "brier": [float(x) for x in brier],
            "concordance_pair_counts": dict(zip(("concordant", "discordant", "tied_risk", "tied_time"), map(int, c[1:])))}


def calibration_table(y_train, y_test, risks, horizon, bins):
    """Fixed probability-bin IPCW calibration, with training KM censoring.

    Observed risk is the Horvitz--Thompson average I(event<=h)/G_train(T).
    It can exceed one in a small bin; it is reported raw, never clipped. This
    is a descriptive calibration table, not a logistic regression on censored
    binary labels. Censored-before-horizon people remain in each bin's n.
    """
    from sksurv.nonparametric import CensoringDistributionEstimator
    censor = CensoringDistributionEstimator().fit(y_train)
    if float(censor.predict_proba([horizon])[0]) <= 0:
        raise ValueError("calibration horizon has zero censoring support")
    cases = y_test["event"] & (y_test["time"] <= horizon)
    weights = np.zeros(len(y_test), dtype=float)
    if cases.any():
        g = censor.predict_proba(y_test["time"][cases])
        if np.any(g <= 0):
            raise ValueError("calibration events have zero censoring support")
        weights[cases] = 1.0 / g
    rows = []
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (risks >= low) & ((risks <= high) if i == len(bins) - 2 else (risks < high))
        n = int(mask.sum())
        rows.append({"risk_lower": float(low), "risk_upper": float(high), "n": n,
                     "predicted_event_probability": float(risks[mask].mean()) if n else None,
                     "ipcw_observed_event_probability": float(weights[mask].mean()) if n else None,
                     "observed_events_by_horizon": int((cases & mask).sum()),
                     "observed_beyond_horizon": int((mask & (y_test["time"] > horizon)).sum()),
                     "censored_by_horizon": int((mask & ~y_test["event"] & (y_test["time"] <= horizon)).sum())})
    return rows


def _validated_predictions(predictions, outcomes, ev):
    missing = set(CURVE_COLUMNS) - set(predictions)
    if missing:
        raise ValueError(f"prediction columns missing: {sorted(missing)}")
    data = predictions.copy()
    for field in ("Patient_ID", "seed", "mode", "endpoint", "split"):
        if any(not isinstance(x, str) or not x or x.strip() != x for x in data[field]):
            raise ValueError(f"{field} must contain literal nonempty strings")
    for col in ("fold", "observed_time", "event", "evaluation_time", "survival_probability", "risk_horizon", "risk_score"):
        data[col] = [float(value) for value in data[col]]
        if not np.isfinite(data[col]).all():
            raise ValueError(f"nonfinite {col}")
    for col, value in (("fold", ev["fold"]), ("endpoint", ev["endpoint"]), ("split", ev["split"])):
        if not data[col].eq(value).all():
            raise ValueError(f"prediction {col} differs from the protocol")
    expected_fits = {(mode, str(seed)) for mode in ev["modes"] for seed in ev["mode_seeds"][mode]}
    actual_fits = set(zip(data["mode"], data["seed"]))
    if actual_fits != expected_fits:
        raise ValueError(f"mode/seed inventory mismatch: missing={sorted(expected_fits - actual_fits)}, extra={sorted(actual_fits - expected_fits)}")
    target = outcomes.loc[outcomes["split"].eq(ev["split"])].sort_values("Patient_ID")
    ids = target["Patient_ID"].tolist()
    if len(ids) < 2:
        raise ValueError("at least two held-out patients are required")
    expected_ids = set(ids)
    fits, identities = {}, {}
    for mode, seed in sorted(expected_fits):
        part = data.loc[data["mode"].eq(mode) & data["seed"].eq(seed)].copy()
        if set(part["Patient_ID"]) != expected_ids:
            raise ValueError(f"patient inventory mismatch in {mode}/{seed}")
        if part.duplicated(["Patient_ID", "evaluation_time"]).any():
            raise ValueError(f"duplicate patient/time rows in {mode}/{seed}")
        for col in ("checkpoint_sha256", "model_contract_sha256"):
            if part[col].nunique() != 1:
                raise ValueError(f"mixed {col} within {mode}/{seed}")
            require_digest(part[col].iloc[0], col)
        matrices = []
        for pid in ids:
            curve = part.loc[part["Patient_ID"].eq(pid)].sort_values("evaluation_time")
            for col in ("observed_time", "event", "risk_horizon", "risk_score"):
                if curve[col].nunique() != 1:
                    raise ValueError(f"inconsistent {col} in one patient's curve")
            truth = target.loc[target["Patient_ID"].eq(pid)].iloc[0]
            if curve["observed_time"].iloc[0] != truth["observed_time"] or curve["event"].iloc[0] != truth["event"]:
                raise ValueError(f"outcome mismatch for {pid} in {mode}/{seed}")
            p = curve["survival_probability"].to_numpy(float)
            if np.any((p < 0) | (p > 1)) or np.any(np.diff(p) > 1e-12):
                raise ValueError("survival curves must be in [0,1] and nonincreasing")
            if curve["evaluation_time"].min() < 0 or not set(ev["required_times"]) <= set(curve["evaluation_time"]):
                raise ValueError("missing explicit required evaluation time; interpolation is forbidden")
            selected = curve.set_index("evaluation_time").loc[ev["required_times"], "survival_probability"].to_numpy(float)
            risk = 1.0 - selected[ev["required_times"].index(float(ev["primary_horizon"]))]
            if curve["risk_horizon"].iloc[0] != ev["primary_horizon"] or not np.isclose(curve["risk_score"].iloc[0], risk, rtol=0, atol=1e-10):
                raise ValueError("scalar risk must equal 1-S(primary_horizon); reversed or inconsistent risk")
            matrices.append(selected)
        fits[(mode, seed)] = np.asarray(matrices)
        identities[f"{mode}/{seed}"] = {col: part[col].iloc[0] for col in ("checkpoint_sha256", "model_contract_sha256")}
    return target, fits, identities


def evaluate_predictions(predictions, outcomes, spec):
    """Evaluate complete fits and paired patient-bootstrap differences.

    Missing or undefined bootstrap replicates are counted, never replaced. CIs
    are unavailable when fewer than the prespecified fraction are valid. The
    training reference and model seeds are fixed in this conditional inference.
    """
    require_reference()
    ev = validate_spec(spec)
    outcomes = validate_outcomes(outcomes, ev["endpoint"])
    if outcome_inventory_sha256(outcomes, ev["endpoint"]) != ev["outcome_inventory_sha256"]:
        raise ValueError("outcome inventory differs from frozen cohort/splits/labels/censoring reference")
    if "fold" in outcomes and not pd.to_numeric(outcomes["fold"]).eq(ev["fold"]).all():
        raise ValueError("outcome fold mismatch")
    train = outcomes.loc[outcomes["split"].eq("train")]
    y_train = outcome_array(train["observed_time"], train["event"])
    support = censoring_support_table(outcomes, ev["endpoint"], [*ev["required_times"], ev["concordance_tau"]])
    if not all(row["censoring_supported"] for row in support):
        raise ValueError("protocol evaluation time lacks training censoring support")
    target, fits, identities = _validated_predictions(predictions, outcomes, ev)
    y_test = outcome_array(target["observed_time"], target["event"])
    grid_indices = [ev["required_times"].index(float(t)) for t in ev["ibs_times"]]
    horizon_index = ev["required_times"].index(float(ev["primary_horizon"]))

    def score(matrix, indices):
        return reference_metrics(y_train, y_test[indices], matrix[indices][:, grid_indices], ev["ibs_times"],
                                 risk=1 - matrix[indices, horizon_index], concordance_tau=ev["concordance_tau"], tied_tol=ev["tied_tol"])

    all_indices = np.arange(len(y_test))
    scores, calibration = {}, {}
    for fit, matrix in fits.items():
        key = "/".join(fit)
        scores[key] = score(matrix, all_indices)
        calibration[key] = {str(float(h)): calibration_table(y_train, y_test,
                             1 - matrix[:, ev["required_times"].index(float(h))], h, ev["calibration_bins"])
                            for h in [ev["primary_horizon"], *ev["secondary_horizons"]]}

    metric_names = ("ipcw_concordance", "ibs")
    bootstrap_values = {f"{a}->{b}": {m: [] for m in metric_names} for a, b in ev["contrasts"]}
    bootstrap_modes = {mode: {m: [] for m in metric_names} for mode in ev["modes"]}
    reasons = Counter()
    rng = np.random.default_rng(ev["bootstrap_seed"])
    draw_digest = hashlib.sha256()
    for _ in range(ev["bootstrap_replicates"]):
        indices = rng.integers(0, len(y_test), size=len(y_test))
        draw_digest.update(indices.astype("<i8").tobytes())
        try:
            # One shared draw across ALL fits; no mode-specific salvage.
            replicate = {fit: score(matrix, indices) for fit, matrix in fits.items()}
        except (ValueError, ZeroDivisionError) as exc:
            reasons[f"{type(exc).__name__}: {str(exc)}"] += 1
            continue
        for mode in ev["modes"]:
            for metric in metric_names:
                bootstrap_modes[mode][metric].append(float(np.mean([
                    replicate[(mode, seed)][metric] for seed in ev["mode_seeds"][mode]])))
        for a, b in ev["contrasts"]:
            for metric in metric_names:
                value = np.mean([replicate[(b, seed)][metric] - replicate[(a, seed)][metric]
                                 for seed in ev["mode_seeds"][a]])
                bootstrap_values[f"{a}->{b}"][metric].append(float(value))

    contrasts = {}
    for a, b in ev["contrasts"]:
        metrics = {}
        for metric in metric_names:
            per_seed = {seed: scores[f"{b}/{seed}"][metric] - scores[f"{a}/{seed}"][metric]
                        for seed in ev["mode_seeds"][a]}
            values = bootstrap_values[f"{a}->{b}"][metric]
            valid = len(values)
            sufficient = valid / ev["bootstrap_replicates"] >= ev["bootstrap_min_valid_fraction"]
            metrics[metric] = {"candidate_minus_comparator": float(np.mean(list(per_seed.values()))),
                               "per_seed_differences": per_seed,
                               "seed_difference_sample_sd": float(np.std(list(per_seed.values()), ddof=1)) if len(per_seed) > 1 else None,
                               "valid_bootstrap_replicates": valid,
                               "percentile_95_ci": np.percentile(values, [2.5, 97.5]).tolist() if sufficient else None,
                               "ci_status": "available" if sufficient else "insufficient_valid_replicates",
                               "favorable_direction": "positive" if metric == "ipcw_concordance" else "negative"}
        contrasts[f"{a}->{b}"] = metrics
    mode_summary = {mode: {metric: {"mean": float(np.mean([scores[f"{mode}/{seed}"][metric] for seed in ev["mode_seeds"][mode]])),
                                    "sample_sd": float(np.std([scores[f"{mode}/{seed}"][metric] for seed in ev["mode_seeds"][mode]], ddof=1)) if len(ev["mode_seeds"][mode]) > 1 else None}
                                  for metric in metric_names} for mode in ev["modes"]}
    for mode, values_by_metric in bootstrap_modes.items():
        for metric, values in values_by_metric.items():
            sufficient = len(values) / ev["bootstrap_replicates"] >= ev["bootstrap_min_valid_fraction"]
            mode_summary[mode][metric].update({
                "valid_bootstrap_replicates": len(values),
                "percentile_95_ci": np.percentile(values, [2.5, 97.5]).tolist() if sufficient else None,
                "ci_status": "available" if sufficient else "insufficient_valid_replicates",
            })
    from .horizon_brier import supporting_horizon_brier
    horizon_brier = supporting_horizon_brier(y_train, y_test, fits, ev)
    return {"schema_version": 1, "reference": {"library": "scikit-survival", "version": REFERENCE_VERSION},
            "status": "evaluated", "evaluation": ev, "n_patients": len(y_test),
            "outcome_inventory_sha256": ev["outcome_inventory_sha256"],
            "patient_inventory_sha256": hashlib.sha256(json.dumps(target["Patient_ID"].tolist(), separators=(",", ":")).encode()).hexdigest(),
            "fit_identities": identities, "scores": scores, "mode_summary": mode_summary,
            "calibration": calibration, "support": support, "paired_contrasts": contrasts,
            "supporting_horizon_brier": horizon_brier,
            "bootstrap": {"requested": ev["bootstrap_replicates"], "invalid": sum(reasons.values()),
                          "invalid_reasons": dict(reasons), "draws_sha256": draw_digest.hexdigest(),
                          "method": "same patient indices across modes/seeds; mean per-seed differences; percentile interval",
                          "uncertainty_scope": "held-out patient sampling conditional on fixed trained models and training censoring reference"},
            "limitations": ["No independent external validation is established.",
                            "Calibration bins use training-reference IPCW; their small-sample observed probability may exceed one.",
                            "Calibration tables are descriptive; no calibration confidence intervals are estimated.",
                            "Historical trainer IBS remains a separate unvalidated diagnostic."]}
