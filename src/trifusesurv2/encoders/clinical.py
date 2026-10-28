"""Semantic clinical token encoder for TriFuseSurv2."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import pandas as pd

from trifusesurv2.schema import CLINICAL_TOKEN_GROUPS

CLINICAL_SCHEMA: dict[str, str] = {
    "AGE": "numeric",
    "KFCF": "numeric",
    "T": "ordinal",
    "N": "ordinal",
    "M": "ordinal",
    "NSTAGE": "ordinal",
    "T_RAW": "categorical",
    "N_RAW": "categorical",
    "M_RAW": "categorical",
    "NSTAGE_RAW": "categorical",
    "SMOKE": "ordinal",
    "ALCOHOL": "ordinal",
    "HPV": "ordinal",
    "PATHOLOGY": "categorical",
    "SEX": "categorical",
    "RACE": "categorical",
    "TX": "categorical",
}

RAW_STAGE_SOURCE_COLUMNS: dict[str, str] = {
    "T_RAW": "T",
    "N_RAW": "N",
    "M_RAW": "M",
    "NSTAGE_RAW": "NSTAGE",
}


def normalize_raw_stage_value(col: str, val: Any) -> str:
    """Preserve TNM/stage subclass semantics as stable categorical strings."""

    if val is None or pd.isna(val):
        return ""
    s = str(val).strip().upper()
    if s == "":
        return ""
    s = re.sub(r"\s+", "", s)
    col_u = str(col).upper()
    if col_u == "NSTAGE_RAW":
        s = s.replace("STAGE", "").replace("STG", "")
        if s and not s.startswith(("0", "I", "V")):
            m = re.search(r"(IV|III|II|I|V|0|[0-9]+[A-Z]*)", s)
            if m is not None:
                s = m.group(1)
    elif col_u in {"T_RAW", "N_RAW", "M_RAW"}:
        prefix = col_u[0]
        m = re.search(rf"({prefix}(?:IS|X|[0-9]+[A-Z]*))", s)
        if m is not None:
            s = m.group(1)
        elif s and not s.startswith(prefix):
            s = f"{prefix}{s}"
    return s


def parse_ordinal_value(col: str, val: Any) -> float:
    """Parse an ordinal or semi-structured value to float.

    This ordinal path is intentionally lossy: TNM sub-stages (e.g. T4a → 4,
    N2b → 2) and Roman numeral suffixes (IVA → 4) collapse to the coarse
    numeric stage. The parallel *_RAW categorical features preserve subclass
    labels for the burden token.
    """

    if val is None or pd.isna(val):
        return np.nan
    s = str(val).strip().upper()
    if s == "":
        return np.nan
    if col.upper() in {"NSTAGE", "STAGE"}:
        s = s.replace("STAGE", "").replace("STG", "").strip()
    m = re.search(r"(-?\d+(\.\d+)?)", s)
    if m is not None:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    roman_map = {"0": 0.0, "I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0, "V": 5.0}
    if s in roman_map:
        return roman_map[s]
    roman_prefix = re.match(r"^(IV|III|II|I|V|0)[A-Z]*$", s)
    if roman_prefix is not None:
        return roman_map[roman_prefix.group(1)]
    if s in {"YES", "Y", "POS", "POSITIVE"}:
        return 1.0
    if s in {"NO", "N", "NEG", "NEGATIVE"}:
        return 0.0
    if col.upper() in {"T", "N", "M"} and "IS" in s:
        return 0.0
    return np.nan


@dataclass(frozen=True)
class NumericFeatureSpec:
    name: str
    mean: float
    std: float
    ordinal: bool


@dataclass(frozen=True)
class CategoricalFeatureSpec:
    name: str
    mapping: dict[str, int]
    unk_index: int


@dataclass(frozen=True)
class GroupEncodingSpec:
    name: str
    numeric: tuple[NumericFeatureSpec, ...]
    categorical: tuple[CategoricalFeatureSpec, ...]
    dim: int


class SemanticClinicalTokenEncoder:
    """Encode clinical data as semantic tokens instead of one flat vector."""

    def __init__(
        self,
        group_specs: "OrderedDict[str, GroupEncodingSpec]",
        *,
        columns: Sequence[str] | None = None,
    ):
        self.group_specs = OrderedDict(group_specs)
        self.group_names = tuple(group_specs.keys())
        self.token_dims = {name: spec.dim for name, spec in group_specs.items()}
        self.max_token_dim = max(self.token_dims.values(), default=0)
        self.output_dim = self.max_token_dim
        # Keep legacy/default state unchanged; old pickled encoders have no
        # column restriction. Explicit policies travel with the saved encoder.
        if columns is not None:
            self._allowed_columns = self._validate_columns(columns)

    @staticmethod
    def _validate_columns(columns: Sequence[str]) -> tuple[str, ...]:
        if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
            raise ValueError("[CLIN2] columns must be a sequence of source-column names")
        values = tuple(columns)
        if any(not isinstance(col, str) or not col or col != col.strip() for col in values):
            raise ValueError("[CLIN2] columns must contain nonempty, unpadded names")
        if len(set(values)) != len(values):
            raise ValueError("[CLIN2] columns must not contain duplicate names")
        return values

    @property
    def allowed_columns(self) -> tuple[str, ...] | None:
        """Requested source columns, or unrestricted legacy group behavior."""
        return getattr(self, "_allowed_columns", None)

    @property
    def column_policy(self) -> dict[str, Any]:
        """JSON-compatible input policy, also retained in checkpoint state."""
        if self.allowed_columns is None:
            return {"mode": "all_token_group_sources"}
        return {"mode": "explicit_source_columns", "columns": list(self.allowed_columns)}

    def _allowed_row(self, row: pd.Series) -> pd.Series:
        if self.allowed_columns is None:
            return row
        return row.loc[row.index.isin(self.allowed_columns)]

    @staticmethod
    def _is_numeric_column(series: pd.Series, threshold: float = 0.8) -> bool:
        coerced = pd.to_numeric(series, errors="coerce")
        valid = coerced.notna().sum()
        frac = valid / max(len(series), 1)
        return frac >= threshold

    @staticmethod
    def _column_series(df: pd.DataFrame, col: str) -> pd.Series | None:
        source = RAW_STAGE_SOURCE_COLUMNS.get(str(col).upper())
        if source is not None and source in df.columns:
            return df[source].apply(lambda v, c=col: normalize_raw_stage_value(c, v))
        if col in df.columns:
            if str(col).upper() in RAW_STAGE_SOURCE_COLUMNS:
                return df[col].apply(lambda v, c=col: normalize_raw_stage_value(c, v))
            return df[col]
        return None

    @staticmethod
    def _row_value(row: pd.Series, col: str) -> Any:
        source = RAW_STAGE_SOURCE_COLUMNS.get(str(col).upper())
        if source is not None and source in row.index:
            return normalize_raw_stage_value(col, row.get(source, None))
        if col in row.index:
            val = row.get(col, None)
            if str(col).upper() in RAW_STAGE_SOURCE_COLUMNS:
                return normalize_raw_stage_value(col, val)
            return val
        return None

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        token_groups: OrderedDict[str, tuple[str, ...]] | None = None,
        *,
        columns: Sequence[str] | None = None,
    ) -> "SemanticClinicalTokenEncoder":
        """Fit train-only statistics using the permitted source columns.

        A selected TNM source (for example ``T``) still produces its ordinal
        and derived ``T_RAW`` features. Excluding ``NSTAGE`` and
        ``NSTAGE_RAW`` excludes both representations. Group identities remain
        fixed, including groups with no available permitted measurements.
        ``None`` preserves the original all-group-column behavior.
        """
        groups = OrderedDict(token_groups or CLINICAL_TOKEN_GROUPS)
        allowed_columns = None if columns is None else cls._validate_columns(columns)
        if allowed_columns is not None:
            df = df.loc[:, df.columns.isin(allowed_columns)]
        specs: "OrderedDict[str, GroupEncodingSpec]" = OrderedDict()

        for group_name, cols in groups.items():
            numeric_specs = []
            categorical_specs = []

            for col in cols:
                series = cls._column_series(df, col)
                if series is None:
                    continue
                non_na = series.dropna()
                if len(non_na) == 0:
                    continue

                schema = CLINICAL_SCHEMA.get(col, "auto")
                if schema in ("numeric", "ordinal"):
                    treat_as_numeric = True
                elif schema == "categorical":
                    treat_as_numeric = False
                else:
                    treat_as_numeric = cls._is_numeric_column(series)

                if treat_as_numeric:
                    if schema == "ordinal":
                        vals = series.apply(lambda v, c=col: parse_ordinal_value(c, v))
                        numeric_series = pd.to_numeric(vals, errors="coerce")
                        ordinal = True
                    else:
                        numeric_series = pd.to_numeric(series, errors="coerce")
                        ordinal = False
                    valid = numeric_series.notna()
                    if valid.sum() == 0:
                        continue
                    mean = float(numeric_series[valid].mean())
                    std = float(numeric_series[valid].std())
                    std = std if std > 1e-6 else 1.0
                    numeric_specs.append(NumericFeatureSpec(col, mean, std, ordinal))
                else:
                    if str(col).upper() in RAW_STAGE_SOURCE_COLUMNS:
                        cats = sorted({normalize_raw_stage_value(col, v) for v in non_na.values if normalize_raw_stage_value(col, v) != ""})
                    else:
                        cats = sorted({str(v).strip() for v in non_na.values if str(v).strip() != ""})
                    if not cats:
                        continue
                    mapping = {cat: idx for idx, cat in enumerate(cats)}
                    categorical_specs.append(CategoricalFeatureSpec(col, mapping, len(cats)))

            dim = (2 * len(numeric_specs)) + sum(len(spec.mapping) + 1 for spec in categorical_specs)
            specs[group_name] = GroupEncodingSpec(
                name=group_name,
                numeric=tuple(numeric_specs),
                categorical=tuple(categorical_specs),
                dim=dim,
            )

        if not specs or all(spec.dim <= 0 for spec in specs.values()):
            raise ValueError("[CLIN2] no usable clinical groups were derived from the provided DataFrame")

        return cls(specs, columns=allowed_columns)

    def _encode_numeric(self, row: pd.Series, spec: NumericFeatureSpec) -> list[float]:
        val = row.get(spec.name, np.nan)
        if spec.ordinal:
            x = parse_ordinal_value(spec.name, val)
        else:
            try:
                x = float(val)
            except Exception:
                x = np.nan
        if np.isnan(x):
            return [0.0, 1.0]
        return [float((x - spec.mean) / spec.std), 0.0]

    def _numeric_is_present(self, row: pd.Series, spec: NumericFeatureSpec) -> bool:
        val = row.get(spec.name, np.nan)
        if spec.ordinal:
            x = parse_ordinal_value(spec.name, val)
        else:
            try:
                x = float(val)
            except Exception:
                x = np.nan
        return not np.isnan(x)

    def _encode_categorical(self, row: pd.Series, spec: CategoricalFeatureSpec) -> list[float]:
        dim = len(spec.mapping) + 1
        vec = np.zeros(dim, dtype=np.float32)
        val = self._row_value(row, spec.name)
        if val is None or pd.isna(val):
            idx = spec.unk_index
        else:
            key = (
                normalize_raw_stage_value(spec.name, val)
                if str(spec.name).upper() in RAW_STAGE_SOURCE_COLUMNS
                else str(val).strip()
            )
            idx = spec.mapping.get(key, spec.unk_index)
        vec[idx] = 1.0
        return vec.tolist()

    def _categorical_is_present(self, row: pd.Series, spec: CategoricalFeatureSpec) -> bool:
        val = self._row_value(row, spec.name)
        if val is None or pd.isna(val):
            return False
        if str(spec.name).upper() in RAW_STAGE_SOURCE_COLUMNS:
            return normalize_raw_stage_value(spec.name, val) != ""
        return str(val).strip() != ""

    def encode_row_tokens(self, row: pd.Series) -> "OrderedDict[str, np.ndarray]":
        row = self._allowed_row(row)
        out: "OrderedDict[str, np.ndarray]" = OrderedDict()
        for name, spec in self.group_specs.items():
            feats: list[float] = []
            for numeric_spec in spec.numeric:
                feats.extend(self._encode_numeric(row, numeric_spec))
            for categorical_spec in spec.categorical:
                feats.extend(self._encode_categorical(row, categorical_spec))
            out[name] = np.asarray(feats, dtype=np.float32)
        return out

    def encode_row_token_matrix(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Return padded clinical token matrix and group-presence mask."""

        row = self._allowed_row(row)
        tokens = self.encode_row_tokens(row)
        mat = np.zeros((len(self.group_names), self.max_token_dim), dtype=np.float32)
        presence = np.zeros((len(self.group_names),), dtype=np.float32)
        for idx, name in enumerate(self.group_names):
            vec = tokens[name]
            mat[idx, : vec.shape[0]] = vec
            spec = self.group_specs[name]
            group_present = any(self._numeric_is_present(row, numeric_spec) for numeric_spec in spec.numeric)
            group_present = group_present or any(
                self._categorical_is_present(row, categorical_spec) for categorical_spec in spec.categorical
            )
            presence[idx] = 1.0 if group_present else 0.0
        return mat, presence

    def encode_frame_token_matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mats = []
        pres = []
        for _, row in df.iterrows():
            mat, p = self.encode_row_token_matrix(row)
            mats.append(mat)
            pres.append(p)
        if not mats:
            return (
                np.zeros((0, len(self.group_names), self.max_token_dim), dtype=np.float32),
                np.zeros((0, len(self.group_names)), dtype=np.float32),
            )
        return np.stack(mats, axis=0), np.stack(pres, axis=0)

    @property
    def token_count(self) -> int:
        return len(self.group_names)

    def available_columns(self) -> list[str]:
        cols: list[str] = []
        for spec in self.group_specs.values():
            cols.extend(feature.name for feature in spec.numeric)
            cols.extend(feature.name for feature in spec.categorical)
        return cols
