"""Explicit clinical coding and train-fitted tabular preprocessing.

No endpoint column is used to fit or transform features. Geometry comes from
the reference manifest, with absence and unavailable measurements kept distinct.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trifusesurv2.preprocessing.compartments import validate_region_manifest
from trifusesurv2.utils.clinical import PROGNOSTIC_CLINICAL_COLS


GEOMETRY_COLUMNS = (
    "pt_volume_mm3", "ln_volume_mm3", "pt_ln_centroid_distance_mm",
    "pt_ln_min_surface_distance_mm", "ln_voxel_rms_spread_mm",
    "ln_left_flag", "ln_right_flag", "ln_bilateral_flag",
)
GEOMETRY_STATES = ("pt_annotation_state", "ln_annotation_state")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def clinical_columns(policy: dict, available_columns: list[str]) -> tuple[list[str], list[str]]:
    """Require confirmed coding and a documented nonredundant stage choice.

    NSTAGE's meaning and relationship to TNM are not inferred from its name.
    """
    if policy.get("status") != "confirmed":
        raise ValueError("Clinical encoding/staging policy is not confirmed.")
    if not str(policy.get("prediction_time", "")).strip():
        raise ValueError("Clinical prediction-time availability must be documented.")
    numeric, categorical = list(policy.get("numeric", [])), list(policy.get("categorical", []))
    chosen = numeric + categorical
    if not chosen or len(chosen) != len(set(chosen)):
        raise ValueError("Clinical columns must be nonempty, unique and explicitly typed.")
    available = list(available_columns)
    if len(available) != len(set(available)) or not set(available) <= set(PROGNOSTIC_CLINICAL_COLS):
        raise ValueError("Fusion clinical inventory must use pretreatment prognostic columns; TX is excluded.")
    redundant = policy.get("redundant_columns", {})
    if set(chosen) & set(redundant) or set(chosen) | set(redundant) != set(available):
        raise ValueError("Every fusion clinical column must be retained or explicitly redundant.")
    for name, evidence in redundant.items():
        if name not in {"T", "N", "M", "NSTAGE"}:
            raise ValueError("Only documented staging redundancy may remove a clinical information source.")
        represented = evidence.get("represented_by", [])
        if evidence.get("confirmed") is not True or not represented or not set(represented) <= set(chosen):
            raise ValueError(f"Unconfirmed redundancy for {name}.")
        if not str(evidence.get("justification", "")).strip():
            raise ValueError(f"Missing redundancy justification for {name}.")
    stages = set(chosen) & {"T", "N", "M", "NSTAGE"}
    if "NSTAGE" in stages and stages & {"T", "N", "M"}:
        raise ValueError("The requested nonredundant staging representation is unresolved "
                         "for NSTAGE together with TNM; NSTAGE's meaning is not inferred from its name.")
    if not str(policy.get("coding_source", "")).strip():
        raise ValueError("Clinical encoding requires a coding-source reference.")
    return numeric, categorical


def geometry_features(manifest: dict, *, patient_id: str, scan_id: str | None = None) -> dict:
    validate_region_manifest(manifest)
    if manifest["patient_id"] != patient_id or (scan_id is not None and manifest["scan_id"] != scan_id):
        raise ValueError("Geometry patient/scan identity mismatch.")
    geometry = manifest["physical_geometry"]
    values = dict(zip(geometry["names"], geometry["values"]))
    validity = dict(zip(geometry["names"], geometry["valid"]))
    pt_present = manifest["regions"]["pt_intra"]["status"] == "present"
    ln_present = manifest["regions"]["ln_intra"]["status"] == "present"
    for name in ("pt_ln_centroid_distance_mm", "pt_ln_min_surface_distance_mm"):
        if validity[name] and not (pt_present and ln_present):
            raise ValueError("A qualified PT-LN distance requires both lesions to be present.")
    if validity["ln_voxel_rms_spread_mm"] and not ln_present:
        raise ValueError("Qualified LN spread requires a present LN annotation.")
    flags = ("ln_left_flag", "ln_right_flag", "ln_bilateral_flag")
    if any(validity[name] for name in flags):
        midline = geometry.get("midline_lps_x_mm")
        if (not all(validity[name] for name in flags) or not ln_present or midline is None
                or not np.isfinite(midline) or any(values[name] not in (0, 1) for name in flags)):
            raise ValueError("Laterality needs binary flags, present LN anatomy and a qualified LPS midline.")
        if bool(values["ln_bilateral_flag"]) != (bool(values["ln_left_flag"]) and bool(values["ln_right_flag"])):
            raise ValueError("LN bilateral flag disagrees with left/right measurements.")
    result = {}
    for name in GEOMETRY_COLUMNS:
        value = float(values[name]) if validity[name] else np.nan
        # log1p is in the declared reference unit mm^3, not a learned transform.
        result[name] = float(np.log1p(value)) if name.endswith("volume_mm3") else value
    for lesion, key in zip(("pt", "ln"), GEOMETRY_STATES):
        result[key] = manifest["regions"][f"{lesion}_intra"]["status"]
    return result


def attach_geometry(frame: pd.DataFrame, *, inventory: dict, manifest_column: str,
                    metadata_directory: Path, scan_column: str | None = None) -> pd.DataFrame:
    if set(inventory) != set(frame.index):
        raise ValueError("Manifest inventory must exactly match the eligible patient inventory.")
    records = []
    for patient_id, row in frame.iterrows():
        entry = inventory[patient_id]
        path = Path(entry["path"])
        if not path.is_absolute():
            path = metadata_directory / path
        declared = Path(str(row[manifest_column]))
        if not declared.is_absolute():
            declared = metadata_directory / declared
        if path.resolve() != declared.resolve() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Manifest path/hash mismatch for patient {patient_id!r}.")
        manifest = json.loads(path.read_text())
        scan_id = str(row[scan_column]) if scan_column else None
        records.append(geometry_features(manifest, patient_id=patient_id, scan_id=scan_id))
    return frame.join(pd.DataFrame(records, index=frame.index), validate="one_to_one")


def _numeric(series: pd.Series, name: str) -> np.ndarray:
    # Empty cells are missing; a nonempty unparseable code is a policy error.
    missing = series.isna() | series.astype(str).str.strip().eq("")
    parsed = pd.to_numeric(series.mask(missing), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    if np.any(~missing.to_numpy() & ~np.isfinite(parsed)):
        raise ValueError(f"Nonfinite or unparseable numeric coding in {name}; supply a qualified policy.")
    return parsed


def _category(value) -> str | None:
    return None if pd.isna(value) or not str(value).strip() else str(value).strip()


class TabularPreprocessor:
    """Median imputation, z scaling and categories learned from training only.

    Missing numeric values have separate indicators. Categorical missing and
    unseen values have distinct slots. Constant training columns are removed;
    a held-out-only level cannot introduce an untrained coefficient.
    """

    def __init__(self, numeric: list[str], categorical: list[str]):
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.parameters: dict = {}
        self.categories: dict = {}
        self.keep: list[int] | None = None
        self.feature_names: list[str] = []

    def fit(self, train: pd.DataFrame) -> "TabularPreprocessor":
        if not len(train):
            raise ValueError("Cannot fit preprocessing without training patients.")
        if set(self.numeric) & set(self.categorical):
            raise ValueError("Numeric and categorical columns overlap.")
        for name in self.numeric:
            values = _numeric(train[name], name)
            finite = np.isfinite(values)
            median = float(np.median(values[finite])) if finite.any() else 0.0
            imputed = np.where(finite, values, median)
            scale = float(imputed.std())
            self.parameters[name] = {"median": median, "mean": float(imputed.mean()),
                                     "scale": scale if scale > 1e-12 else 1.0,
                                     "observed_training_count": int(finite.sum())}
        for name in self.categorical:
            self.categories[name] = sorted({str(value) for raw in train[name]
                                            if (value := _category(raw)) is not None})
        matrix, names = self._raw_transform(train)
        self.keep = np.flatnonzero(np.ptp(matrix, axis=0) > 1e-12).tolist()
        if not self.keep:
            raise ValueError("All training baseline features are constant or unavailable.")
        self.feature_names = [names[index] for index in self.keep]
        return self

    def _raw_transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        columns, names = [], []
        for name in self.numeric:
            values = _numeric(frame[name], name)
            finite = np.isfinite(values)
            parameters = self.parameters[name]
            columns.extend([(np.where(finite, values, parameters["median"]) - parameters["mean"])
                            / parameters["scale"], (~finite).astype(float)])
            names.extend([name, f"{name}::unavailable"])
        for name in self.categorical:
            levels = self.categories[name]
            values = [_category(value) for value in frame[name]]
            for level in levels:
                columns.append(np.asarray([value == level for value in values], dtype=float))
                names.append(f"{name}::level={json.dumps(level)}")
            columns.extend([np.asarray([value is None for value in values], dtype=float),
                            np.asarray([value is not None and value not in levels for value in values], dtype=float)])
            names.extend([f"{name}::missing", f"{name}::unseen"])
        matrix = np.column_stack(columns) if columns else np.zeros((len(frame), 0))
        if not np.isfinite(matrix).all():
            raise ValueError("Preprocessed baseline matrix contains nonfinite values.")
        return matrix, names

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.keep is None:
            raise ValueError("Preprocessor has not been fitted.")
        return self._raw_transform(frame)[0][:, self.keep]

    def to_dict(self) -> dict:
        return {"numeric": self.numeric, "categorical": self.categorical,
                "parameters": self.parameters, "categories": self.categories,
                "keep": self.keep, "feature_names": self.feature_names,
                "fit_scope": "training_only", "numeric_imputation": "training_median"}
