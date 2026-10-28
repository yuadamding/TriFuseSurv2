"""Traceable survival exports without importing the training stack."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np


CURVE_COLUMNS = (
    "Patient_ID", "split", "fold", "seed", "mode", "endpoint", "observed_time",
    "event", "evaluation_time", "survival_probability", "checkpoint_sha256",
    "model_contract_sha256", "risk_horizon", "risk_score",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_digest(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def hazards_to_survival(hazards, evaluation_times, time_bin_width):
    """Piecewise exponential S(t), using the existing model's bin convention.

    Unlike the old diagnostic utility, this rejects extrapolation beyond K*bw.
    A hazard is the conditional event probability in [k*bw, (k+1)*bw).
    """
    hazards = np.asarray(hazards, dtype=float)
    times = np.asarray(evaluation_times, dtype=float)
    width = float(time_bin_width)
    if hazards.ndim != 2 or hazards.shape[1] == 0 or not np.isfinite(hazards).all():
        raise ValueError("hazards must be a finite, nonempty patient by bin matrix")
    if np.any((hazards < 0) | (hazards > 1)) or not np.isfinite(width) or width <= 0:
        raise ValueError("invalid hazards or bin width")
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("evaluation times must be a finite vector")
    if np.any(times < 0) or np.any(times > hazards.shape[1] * width):
        raise ValueError("evaluation time outside trained hazard coverage; no extrapolation")
    # Match the v2 model's hazards_to_risk numerical hazard bounds.
    log_remaining = np.log1p(-np.clip(hazards, 1e-7, 1 - 1e-7))
    exposure = np.clip(times[:, None] / width - np.arange(hazards.shape[1]), 0, 1)
    return np.exp(log_remaining @ exposure.T)


def write_survival_curves(path, *, patient_ids, observed_times, events,
                          evaluation_times, survival_probabilities, split, fold,
                          seed, mode, endpoint, checkpoint_sha256,
                          model_contract_sha256, risk_horizon, metadata=None):
    """Write long-form probabilities and a hash-bound provenance sidecar.

    Patient IDs must already be literal strings. No ID normalization, inference,
    horizon interpolation, clipping of predictions, or missing-patient dropping.
    """
    ids = list(patient_ids)
    if any(not isinstance(x, str) or not x or x != x.strip() for x in ids):
        raise ValueError("patient IDs must be nonempty, unmodified literal strings")
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("patient IDs must be unique and nonempty")
    times = np.asarray(evaluation_times, dtype=float)
    probs = np.asarray(survival_probabilities, dtype=float)
    observed = np.asarray(observed_times, dtype=float)
    event = np.asarray(events, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or times[0] < 0:
        raise ValueError("evaluation times must be finite, nonnegative, strictly increasing")
    if observed.shape != (len(ids),) or event.shape != observed.shape:
        raise ValueError("outcome and patient counts differ")
    if not np.isfinite(observed).all() or np.any(observed < 0) or not np.isin(event, [0, 1]).all():
        raise ValueError("invalid observed outcomes")
    if probs.shape != (len(ids), len(times)) or not np.isfinite(probs).all():
        raise ValueError("survival matrix has invalid shape or nonfinite values")
    if np.any((probs < 0) | (probs > 1)) or np.any(np.diff(probs, axis=1) > 1e-12):
        raise ValueError("survival must be in [0,1] and nonincreasing")
    horizon = float(risk_horizon)
    index = np.flatnonzero(times == horizon)
    if len(index) != 1:
        raise ValueError("risk horizon must be an explicit exported evaluation time")
    for name, value in (("checkpoint_sha256", checkpoint_sha256), ("model_contract_sha256", model_contract_sha256)):
        require_digest(value, name)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURVE_COLUMNS)
        writer.writeheader()
        for i, pid in enumerate(ids):
            for j, time in enumerate(times):
                writer.writerow(dict(zip(CURVE_COLUMNS, (
                    pid, split, fold, seed, mode, endpoint, observed[i], int(event[i]),
                    time, probs[i, j], checkpoint_sha256, model_contract_sha256,
                    horizon, 1.0 - probs[i, index[0]],
                ))))
    receipt = {"schema_version": 1, "csv_sha256": file_sha256(path),
               "patients": len(ids), "evaluation_times": times.tolist(),
               "checkpoint_sha256": checkpoint_sha256,
               "model_contract_sha256": model_contract_sha256,
               "metadata": metadata or {}}
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt
