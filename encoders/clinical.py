"""Semantic clinical token encoder for TriFuseSurv2."""

from __future__ import annotations

from collections import OrderedDict
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
    "SMOKE": "ordinal",
    "ALCOHOL": "ordinal",
    "HPV": "ordinal",
    "PATHOLOGY": "categorical",
    "SEX": "categorical",
    "RACE": "categorical",
    "TX": "categorical",
}


def parse_ordinal_value(col: str, val: Any) -> float:
    """Parse an ordinal or semi-structured value to float."""

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

    def __init__(self, group_specs: "OrderedDict[str, GroupEncodingSpec]"):
        self.group_specs = OrderedDict(group_specs)
        self.group_names = tuple(group_specs.keys())
        self.token_dims = {name: spec.dim for name, spec in group_specs.items()}
        self.max_token_dim = max(self.token_dims.values(), default=0)
        self.output_dim = self.max_token_dim

    @staticmethod
    def _is_numeric_column(series: pd.Series, threshold: float = 0.8) -> bool:
        coerced = pd.to_numeric(series, errors="coerce")
        valid = coerced.notna().sum()
        frac = valid / max(len(series), 1)
        return frac >= threshold

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        token_groups: OrderedDict[str, tuple[str, ...]] | None = None,
    ) -> "SemanticClinicalTokenEncoder":
        groups = OrderedDict(token_groups or CLINICAL_TOKEN_GROUPS)
        specs: "OrderedDict[str, GroupEncodingSpec]" = OrderedDict()

        for group_name, cols in groups.items():
            numeric_specs = []
            categorical_specs = []

            for col in cols:
                if col not in df.columns:
                    continue
                series = df[col]
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
                    cats = sorted({str(v).strip() for v in non_na.values if str(v).strip() != ""})
                    if not cats:
                        continue
                    mapping = {cat: idx for idx, cat in enumerate(cats)}
                    categorical_specs.append(CategoricalFeatureSpec(col, mapping, len(cats)))

            dim = (2 * len(numeric_specs)) + sum(len(spec.mapping) + 1 for spec in categorical_specs)
            if dim <= 0:
                continue
            specs[group_name] = GroupEncodingSpec(
                name=group_name,
                numeric=tuple(numeric_specs),
                categorical=tuple(categorical_specs),
                dim=dim,
            )

        if not specs:
            raise ValueError("[CLIN2] no usable clinical groups were derived from the provided DataFrame")

        return cls(specs)

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

    def _encode_categorical(self, row: pd.Series, spec: CategoricalFeatureSpec) -> list[float]:
        dim = len(spec.mapping) + 1
        vec = np.zeros(dim, dtype=np.float32)
        val = row.get(spec.name, None)
        if val is None or pd.isna(val):
            idx = spec.unk_index
        else:
            idx = spec.mapping.get(str(val).strip(), spec.unk_index)
        vec[idx] = 1.0
        return vec.tolist()

    def encode_row_tokens(self, row: pd.Series) -> "OrderedDict[str, np.ndarray]":
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

        tokens = self.encode_row_tokens(row)
        mat = np.zeros((len(self.group_names), self.max_token_dim), dtype=np.float32)
        presence = np.zeros((len(self.group_names),), dtype=np.float32)
        for idx, name in enumerate(self.group_names):
            vec = tokens[name]
            mat[idx, : vec.shape[0]] = vec
            presence[idx] = 1.0
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
