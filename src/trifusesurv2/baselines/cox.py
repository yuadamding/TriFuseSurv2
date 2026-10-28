"""Deterministic C0/C1 fits with held-out data excluded from model selection."""
from __future__ import annotations

import json
import warnings

import numpy as np

from .features import TabularPreprocessor


def survival_target(frame, time_column: str, event_column: str) -> np.ndarray:
    import pandas as pd
    from sksurv.util import Surv

    times = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    events = pd.to_numeric(frame[event_column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(times).all() or np.any(times < 0) or not np.isin(events, [0, 1]).all():
        raise ValueError("Invalid eligible survival outcomes; times must be finite/nonnegative and events binary.")
    return Surv.from_arrays(events.astype(bool), times)


def survival_probabilities(model, matrix: np.ndarray, times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or np.any(times < 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("Prediction times must be finite, nonnegative and strictly increasing.")
    if times[-1] > float(model.unique_times_[-1]):
        raise ValueError("Prediction horizon exceeds training follow-up support; no extrapolation is permitted.")
    predictions = np.asarray([function(times) for function in model.predict_survival_function(matrix)])
    if (not np.isfinite(predictions).all() or np.any(predictions < 0)
            or np.any(predictions > 1) or np.any(np.diff(predictions, axis=1) > 1e-10)):
        raise ValueError("Cox survival predictions are invalid or nonmonotone.")
    return predictions


def fit_cox_baseline(train, validation, *, numeric: list[str], categorical: list[str],
                     time_column: str, event_column: str, alpha_grid: list[float],
                     validation_times: list[float], ties: str = "efron", n_iter: int = 100,
                     tol: float = 1e-9) -> tuple:
    """Select a training-fitted ridge Cox model by minimum validation IBS.

    There is intentionally no test-frame argument. All candidates use the same
    training-fitted encoding, censoring estimate and fixed validation time grid.
    Exact score ties choose the larger alpha (stronger regularization).
    """
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import integrated_brier_score

    alphas = [float(alpha) for alpha in alpha_grid]
    if (not alphas or len(set(alphas)) != len(alphas) or not np.isfinite(alphas).all()
            or min(alphas) <= 0):
        raise ValueError("Predeclared alpha grid must contain unique positive finite penalties.")
    if ties != "efron" or n_iter < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("Baseline contract requires Efron ties and positive solver controls.")
    times = np.asarray(validation_times, dtype=float)
    if len(times) < 2 or np.any(np.diff(times) <= 0) or not np.isfinite(times).all():
        raise ValueError("Validation IBS requires at least two increasing finite times.")
    preprocessor = TabularPreprocessor(numeric, categorical).fit(train)
    x_train, x_validation = preprocessor.transform(train), preprocessor.transform(validation)
    y_train = survival_target(train, time_column, event_column)
    y_validation = survival_target(validation, time_column, event_column)
    if not y_train["event"].any():
        raise ValueError("Cox training requires an observed event.")
    # All candidates must succeed. Silently dropping a failed member would
    # change the prespecified search inventory and hide convergence problems.
    selected, scores, best_key = None, [], None
    for alpha in alphas:
        model = CoxPHSurvivalAnalysis(alpha=alpha, ties=ties, n_iter=n_iter, tol=tol)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(x_train, y_train)
        if caught:
            messages = [f"{item.category.__name__}: {item.message}" for item in caught]
            raise ValueError(f"Cox candidate alpha={alpha} raised fit warnings: {messages}")
        if not np.isfinite(model.coef_).all():
            raise ValueError(f"Cox candidate alpha={alpha} produced nonfinite coefficients.")
        prediction = survival_probabilities(model, x_validation, times)
        score = float(integrated_brier_score(y_train, y_validation, prediction, times))
        if not np.isfinite(score):
            raise ValueError(f"Cox candidate alpha={alpha} has nonfinite validation IBS.")
        scores.append({"alpha": alpha, "validation_ibs": score})
        key = (score, -alpha)
        if best_key is None or key < best_key:
            selected, best_key = model, key
    selection = {"metric": "validation_ibs", "censoring_reference": "training",
                 "times": times.tolist(), "candidates": scores,
                 "selected_alpha": float(selected.alpha), "tie_break": "larger_alpha",
                 "refit_on_validation": False, "deterministic": True}
    return selected, preprocessor, selection


def serializable_model(model, preprocessor: TabularPreprocessor) -> dict:
    """Portable, inspectable checkpoint; no executable pickle is needed.

    S(t|x) = exp(-H0(t) * exp(x beta)); H0 is right-continuous and zero
    before the first recorded time. Prediction beyond training support is not
    part of the contract.
    """
    result = {"schema_version": 1, "estimator": "sksurv.CoxPHSurvivalAnalysis",
              "alpha": float(model.alpha), "ties": model.ties,
              "coefficients": model.coef_.tolist(), "preprocessor": preprocessor.to_dict(),
              "baseline_times": model.cum_baseline_hazard_.x.tolist(),
              "baseline_cumulative_hazard": model.cum_baseline_hazard_.y.tolist(),
              "risk_direction": "larger 1-S(t) means greater event risk"}
    # Guard checkpoint serializability without coercing missing values to null.
    json.dumps(result, allow_nan=False)
    return result
