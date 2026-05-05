"""Habitat-preserving radiomics token encoder for TriFuseSurv2."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

DEFAULT_GROUP_PREFIXES: "OrderedDict[str, str]" = OrderedDict(
    [
        ("pt_intra", "PT_intratumor__"),
        ("pt_peri", "PT_peritumor_10mm__"),
        ("ln_intra", "LN_intratumor__"),
        ("ln_peri", "LN_peritumor_10mm__"),
    ]
)

DEFAULT_PRESENCE_COLUMNS: dict[str, str] = {
    "pt_intra": "present__PT_intratumor",
    "pt_peri": "present__PT_peritumor_10mm",
    "ln_intra": "present__LN_intratumor",
    "ln_peri": "present__LN_peritumor_10mm",
}


def _pad_or_trunc_1d(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.shape[0] == dim:
        return x
    if x.shape[0] > dim:
        return x[:dim].copy()
    out = np.zeros(dim, dtype=np.float32)
    out[: x.shape[0]] = x
    return out


@dataclass(frozen=True)
class GroupPCASpec:
    name: str
    input_dim: int
    output_dim: int
    lower: np.ndarray
    upper: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    pca_mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray


class HabitatRadiomicsTokenEncoder:
    """Preserve habitat-specific radiomics tokens instead of flattening them.

    The defaults are intentionally aligned to what the current package exports
    today: PT/LN intra plus 10 mm peri habitats. Multiscale peri radiomics
    should only be introduced once real 3 mm / 6 mm features are exported.
    """

    def __init__(
        self,
        *,
        group_names: tuple[str, ...],
        pca_specs: dict[str, GroupPCASpec],
        patient_vectors: dict[str, dict[str, np.ndarray]],
        presence_map: dict[str, np.ndarray],
    ):
        self.group_names = tuple(group_names)
        self.group_index = {name: idx for idx, name in enumerate(self.group_names)}
        self.pca_specs = dict(pca_specs)
        self.patient_vectors = dict(patient_vectors)
        self.presence_map = dict(presence_map)
        self.token_dims = {name: spec.output_dim for name, spec in self.pca_specs.items()}
        self.max_token_dim = max(self.token_dims.values(), default=0)
        self.output_dim = self.max_token_dim

    @staticmethod
    def normalize_patient_id(pid: Any) -> str:
        s = str(pid).strip()
        s = re.sub(r"(_radio|_radiomics|_rad)$", "", s, flags=re.IGNORECASE)
        m = re.match(r"^([A-Za-z]+)0*([0-9]+)$", s)
        if m:
            return f"{m.group(1).upper()}{int(m.group(2))}"
        return s.upper()

    @classmethod
    def fit_from_wide_csv(
        cls,
        *,
        radiomics_csv: str | Path,
        train_ids: list[str],
        all_ids: list[str],
        total_pcs_per_group: int = 16,
        group_prefixes: OrderedDict[str, str] | None = None,
        presence_columns: dict[str, str] | None = None,
        require_presence_columns: bool = True,
        random_state: int = 1,
    ) -> "HabitatRadiomicsTokenEncoder":
        group_prefixes = OrderedDict(group_prefixes or DEFAULT_GROUP_PREFIXES)
        presence_columns = dict(presence_columns or DEFAULT_PRESENCE_COLUMNS)

        df = pd.read_csv(radiomics_csv)
        id_col = "case_id" if "case_id" in df.columns else ("patient_id" if "patient_id" in df.columns else None)
        if id_col is None:
            raise ValueError(f"[RAD2] Missing case identifier column in {radiomics_csv}")

        if require_presence_columns:
            missing_presence = [presence_columns[g] for g in group_prefixes if presence_columns.get(g, "") not in df.columns]
            if missing_presence:
                raise ValueError(
                    "[RAD2] Missing explicit radiomics presence columns: "
                    + ", ".join(sorted(dict.fromkeys(missing_presence)))
                )

        df = df.copy()
        df["patient_id_norm"] = df[id_col].map(cls.normalize_patient_id)
        dup_mask = df["patient_id_norm"].duplicated(keep=False)
        if dup_mask.any():
            dup_ids = df.loc[dup_mask, "patient_id_norm"].drop_duplicates().tolist()
            raise ValueError(f"[RAD2] duplicate normalized radiomics IDs: {dup_ids[:10]}")

        train_ids_norm = [cls.normalize_patient_id(x) for x in train_ids]
        all_ids_norm = list(dict.fromkeys(cls.normalize_patient_id(x) for x in all_ids))
        wanted = set(all_ids_norm)

        feature_cols: dict[str, list[str]] = {}
        for name, prefix in group_prefixes.items():
            cols = [c for c in df.columns if c.startswith(prefix) and not c.endswith("__error")]
            usable = []
            for col in cols:
                series = pd.to_numeric(df[col], errors="coerce")
                if series.notna().any():
                    usable.append(col)
            feature_cols[name] = sorted(usable)

        patient_vectors_raw: dict[str, dict[str, np.ndarray]] = {}
        presence_map: dict[str, np.ndarray] = {}
        group_names = tuple(group_prefixes.keys())
        for _, row in df.iterrows():
            pid = str(row["patient_id_norm"])
            if pid not in wanted:
                continue
            patient_vectors_raw[pid] = {}
            presence_bits = np.zeros((len(group_names),), dtype=np.float32)

            for idx, group_name in enumerate(group_names):
                cols = feature_cols[group_name]
                if cols:
                    vals = pd.to_numeric(pd.Series([row[c] for c in cols]), errors="coerce").to_numpy(dtype=np.float32)
                    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    vals = np.zeros((0,), dtype=np.float32)
                patient_vectors_raw[pid][group_name] = vals

                present_col = presence_columns.get(group_name, "")
                if present_col and present_col in df.columns:
                    present_raw = pd.to_numeric(pd.Series([row[present_col]]), errors="coerce").iloc[0]
                    if pd.isna(present_raw):
                        raise ValueError(
                            f"[RAD2] Non-numeric presence value for patient {pid} group {group_name}: "
                            f"{row[present_col]!r}"
                        )
                    present = float(present_raw) > 0.0
                else:
                    if require_presence_columns:
                        raise ValueError(f"[RAD2] Missing presence column for group {group_name}")
                    present = bool(np.any(np.abs(vals) > 0))
                presence_bits[idx] = 1.0 if present else 0.0

            presence_map[pid] = presence_bits

        pca_specs: dict[str, GroupPCASpec] = {}
        for group_name in group_names:
            train_rows = []
            idx = group_names.index(group_name)
            for pid in train_ids_norm:
                if pid not in patient_vectors_raw or presence_map[pid][idx] < 0.5:
                    continue
                vec = patient_vectors_raw[pid][group_name]
                if vec.size > 0:
                    train_rows.append(vec)

            if not train_rows:
                import warnings
                warnings.warn(
                    f"[RAD2] radiomics group '{group_name}' has no present training samples; "
                    f"it will be absent from the fitted encoder. If this group is required, "
                    f"check the presence columns and training ID overlap.",
                    stacklevel=2,
                )
                continue
            x = np.stack(train_rows, axis=0).astype(np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            n_samples, dim = x.shape
            n_comp = max(1, min(int(total_pcs_per_group), n_samples, dim))
            feature_lower = np.percentile(x, 1.0, axis=0).astype(np.float32)
            feature_upper = np.percentile(x, 99.0, axis=0).astype(np.float32)
            feature_upper = np.maximum(feature_upper, feature_lower).astype(np.float32)
            x_clipped = np.clip(x, feature_lower, feature_upper).astype(np.float32)
            feature_mean = x_clipped.mean(axis=0).astype(np.float32)
            feature_scale = x_clipped.std(axis=0).astype(np.float32)
            feature_scale = np.where(feature_scale > 1e-6, feature_scale, 1.0).astype(np.float32)
            x_scaled = ((x_clipped - feature_mean) / feature_scale).astype(np.float32)
            pca = PCA(n_components=n_comp, svd_solver="full", random_state=int(random_state))
            pca.fit(x_scaled)
            pca_specs[group_name] = GroupPCASpec(
                name=group_name,
                input_dim=dim,
                output_dim=n_comp,
                lower=feature_lower,
                upper=feature_upper,
                mean=feature_mean,
                scale=feature_scale,
                pca_mean=pca.mean_.astype(np.float32),
                components=pca.components_.astype(np.float32),
                explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
            )

        if not pca_specs:
            raise ValueError("[RAD2] no usable radiomics groups were available to fit PCA")

        patient_vectors: dict[str, dict[str, np.ndarray]] = {}
        for pid, raw_groups in patient_vectors_raw.items():
            token_map: dict[str, np.ndarray] = {}
            for group_name in group_names:
                spec = pca_specs.get(group_name)
                if spec is None:
                    continue
                raw = raw_groups.get(group_name, np.zeros((0,), dtype=np.float32))
                if raw.size == 0:
                    token = np.zeros((spec.output_dim,), dtype=np.float32)
                else:
                    raw = _pad_or_trunc_1d(raw, spec.input_dim)
                    clipped = np.clip(raw, spec.lower, spec.upper).astype(np.float32)
                    scaled = ((clipped - spec.mean) / spec.scale).astype(np.float32)
                    token = (spec.components @ (scaled - spec.pca_mean)).astype(np.float32)
                token_map[group_name] = token
            patient_vectors[pid] = token_map

        return cls(
            group_names=group_names,
            pca_specs=pca_specs,
            patient_vectors=patient_vectors,
            presence_map=presence_map,
        )

    def _token_for_group(self, pid: str, group_name: str) -> tuple[np.ndarray, float]:
        pid_norm = self.normalize_patient_id(pid)
        token_map = self.patient_vectors.get(pid_norm, {})
        if group_name in token_map:
            idx = self.group_index[group_name]
            presence = float(self.presence_map.get(pid_norm, np.zeros((len(self.group_names),), dtype=np.float32))[idx])
            return token_map[group_name], presence

        dim = self.max_token_dim
        return np.zeros((dim,), dtype=np.float32), 0.0

    def encode_patient_tokens(self, pid: str) -> "OrderedDict[str, np.ndarray]":
        out: "OrderedDict[str, np.ndarray]" = OrderedDict()
        for group_name in self.group_names:
            token, _ = self._token_for_group(pid, group_name)
            out[group_name] = _pad_or_trunc_1d(token, self.max_token_dim)
        return out

    def encode_patient_token_matrix(self, pid: str) -> tuple[np.ndarray, np.ndarray]:
        mat = np.zeros((len(self.group_names), self.max_token_dim), dtype=np.float32)
        presence = np.zeros((len(self.group_names),), dtype=np.float32)
        for idx, group_name in enumerate(self.group_names):
            token, present = self._token_for_group(pid, group_name)
            token = _pad_or_trunc_1d(token, self.max_token_dim)
            mat[idx, : token.shape[0]] = token
            presence[idx] = float(present)
        return mat, presence

    @property
    def token_count(self) -> int:
        return len(self.group_names)
