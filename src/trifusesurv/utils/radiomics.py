"""Shared radiomics feature encoder for TriFuseSurv."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def _pad_or_trunc_1d(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.shape[0] == dim:
        return x
    if x.shape[0] > dim:
        return x[:dim].copy()
    out = np.zeros(dim, dtype=np.float32)
    out[: x.shape[0]] = x
    return out


class RadiomicsEncoder:
    """PCA-based radiomics encoder with per-group aggregation.

    Output vector: [PCs (total_pc_dim)] + [presence bits (G)]
    where G = number of ROI groups (typically 4: PT_intra, PT_peri, LN_intra, LN_peri).
    """

    def __init__(
        self,
        patient_vectors: Dict[str, np.ndarray],
        *,
        group_names: Optional[List[str]] = None,
        group_n_comp: Optional[Dict[str, int]] = None,
        pc_slices: Optional[Dict[str, slice]] = None,
        total_pc_dim: int = 0,
    ):
        self.patient_vectors = {
            str(k): np.asarray(v, dtype=np.float32).reshape(-1)
            for k, v in patient_vectors.items()
        }
        if not self.patient_vectors:
            raise RuntimeError("[RAD] empty patient_vectors")
        first = next(iter(self.patient_vectors.values()))
        self.output_dim = int(first.shape[0])

        self.group_names = list(group_names or [])
        self.group_n_comp = dict(group_n_comp or {})
        self.pc_slices = dict(pc_slices or {})
        self.total_pc_dim = int(total_pc_dim)

    @staticmethod
    def build_radiomics_path(lid: str, radiomics_root: str) -> Optional[Path]:
        lid = str(lid)
        roots = [Path(radiomics_root), Path("radiomics") / radiomics_root]
        basenames = [f"{lid}_radio_radiomics.csv", f"{lid}_radiomics.csv", f"{lid}.csv"]
        for root in roots:
            for bn in basenames:
                p = root / bn
                if p.is_file():
                    return p
        return None

    @staticmethod
    def _extract_feature_cols(df: pd.DataFrame) -> List[str]:
        meta_cols = {
            "roi_name", "case_id", "tumor_id", "tumor_class", "tumor_origin_roi",
            "region", "peritumor_radius_mm",
            "qc_voxel_volume_mm3", "qc_roi_voxels", "qc_roi_volume_ml",
        }
        feature_cols = [
            c for c in df.columns
            if (not c.startswith("diagnostics_")) and c not in meta_cols
        ]
        return sorted(feature_cols)

    @staticmethod
    def _aggregate_patient_groups(
        df: pd.DataFrame, feature_cols: Optional[List[str]],
    ):
        if feature_cols is None:
            feature_cols = RadiomicsEncoder._extract_feature_cols(df)
        else:
            feature_cols = list(feature_cols)
            for c in feature_cols:
                if c not in df.columns:
                    df[c] = np.nan

        df_feat = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        idx = df.index
        roi = (
            df["roi_name"].astype(str)
            if "roi_name" in df.columns
            else pd.Series(["PT_intratumor"] * len(df), index=idx)
        )

        tumor_class = (
            df["tumor_class"].astype(str)
            if "tumor_class" in df.columns
            else roi.map(lambda s: "PT" if str(s).upper().startswith("PT") else "LN")
        )
        region = (
            df["region"].astype(str)
            if "region" in df.columns
            else roi.map(lambda s: "peritumor" if "peri" in str(s).lower() else "intratumor")
        )

        tumor_class_u = tumor_class.str.upper()
        region_l = region.str.lower()
        is_pt = tumor_class_u == "PT"
        is_ln = ~is_pt

        group_specs = [
            ("PT_intra", is_pt & (region_l == "intratumor")),
            ("PT_peri", is_pt & (region_l == "peritumor")),
            ("LN_intra", is_ln & (region_l == "intratumor")),
            ("LN_peri", is_ln & (region_l == "peritumor")),
        ]

        n_feat = len(feature_cols)
        group_vectors = {}
        presence_bits = []
        for gname, mask in group_specs:
            if mask.any():
                sub = df_feat.loc[mask]
                mean = sub.mean(axis=0).to_numpy(dtype=np.float32)
                std = sub.std(axis=0).to_numpy(dtype=np.float32)
                presence_bits.append(1.0)
            else:
                mean = np.zeros(n_feat, dtype=np.float32)
                std = np.zeros(n_feat, dtype=np.float32)
                presence_bits.append(0.0)
            mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
            std = np.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
            group_vectors[gname] = np.concatenate([mean, std], axis=0)

        presence = np.asarray(presence_bits, dtype=np.float32)
        group_names = [g[0] for g in group_specs]
        return group_vectors, presence, feature_cols, group_names

    @classmethod
    def fit(
        cls,
        train_ids: Sequence[str],
        all_ids: Sequence[str],
        radiomics_root: str,
        total_pcs: int,
        seed: int,
    ) -> "RadiomicsEncoder":
        all_ids = list(dict.fromkeys([str(x) for x in all_ids]))
        train_ids = [str(x) for x in train_ids]

        # Pick feature columns from TRAIN only (avoids data leakage)
        feature_cols = None
        for lid in train_ids:
            p = cls.build_radiomics_path(lid, radiomics_root)
            if p is None:
                continue
            try:
                df0 = pd.read_csv(p)
            except Exception:
                continue
            feature_cols = cls._extract_feature_cols(df0)
            if feature_cols:
                break

        if feature_cols is None:
            raise RuntimeError("[RAD] Could not determine feature columns from TRAIN radiomics CSVs.")

        group_names = None
        group_vectors: Dict[str, Dict[str, np.ndarray]] = {}
        presence_map: Dict[str, np.ndarray] = {}

        for lid in all_ids:
            p = cls.build_radiomics_path(lid, radiomics_root)
            if p is None:
                continue
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            gvecs, presence, _, gnames = cls._aggregate_patient_groups(df, feature_cols)
            if group_names is None:
                group_names = gnames
                group_vectors = {g: {} for g in group_names}
            for g, vec in gvecs.items():
                group_vectors[g][lid] = vec.astype(np.float32)
            presence_map[lid] = presence.astype(np.float32)

        if group_names is None or not presence_map:
            raise RuntimeError("[RAD] No usable radiomics CSVs found.")

        available_ids = [lid for lid in all_ids if lid in presence_map]
        G = len(group_names)
        group_index = {g: i for i, g in enumerate(group_names)}
        pcs_per_group_target = max(1, int(total_pcs) // max(1, G))

        group_pca_means = {}
        group_pca_components = {}
        group_n_comp: Dict[str, int] = {}

        for g in group_names:
            rows = []
            for lid in train_ids:
                pres = presence_map.get(lid)
                if pres is None or pres[group_index[g]] < 0.5:
                    continue
                vec = group_vectors[g].get(lid)
                if vec is not None:
                    rows.append(vec)
            if not rows:
                group_pca_means[g] = None
                group_pca_components[g] = None
                group_n_comp[g] = 0
                continue
            X = np.stack(rows, axis=0).astype(np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            n_samples, dim = X.shape
            n_comp = max(1, min(pcs_per_group_target, n_samples, dim))
            pca = PCA(n_components=n_comp, svd_solver="full", random_state=int(seed))
            pca.fit(X)
            group_pca_means[g] = pca.mean_.astype(np.float32)
            group_pca_components[g] = pca.components_.astype(np.float32)
            group_n_comp[g] = int(n_comp)

        total_pc_dim = sum(group_n_comp[g] for g in group_names)
        final_dim = total_pc_dim + G  # PCs + presence bits

        # Build PC slices for SHAP grouping
        pc_slices: Dict[str, slice] = {}
        offset = 0
        for g in group_names:
            nc = group_n_comp[g]
            if nc > 0:
                pc_slices[g] = slice(offset, offset + nc)
                offset += nc

        patient_vectors = {}
        for lid in available_ids:
            pcs_chunks = []
            pres_vec = presence_map[lid]
            for g in group_names:
                n_comp = group_n_comp[g]
                mean = group_pca_means[g]
                comp = group_pca_components[g]
                if n_comp <= 0 or mean is None or comp is None:
                    continue
                if pres_vec[group_index[g]] < 0.5:
                    pcs = np.zeros((n_comp,), dtype=np.float32)
                else:
                    x_raw = group_vectors[g].get(lid, None)
                    if x_raw is None:
                        pcs = np.zeros((n_comp,), dtype=np.float32)
                    else:
                        x_raw = np.nan_to_num(
                            np.asarray(x_raw, dtype=np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0,
                        )
                        pcs = (comp @ (x_raw - mean)).astype(np.float32)
                pcs_chunks.append(pcs)

            pcs_all = (
                np.concatenate(pcs_chunks, axis=0)
                if pcs_chunks
                else np.zeros((total_pc_dim,), dtype=np.float32)
            )
            pcs_all = _pad_or_trunc_1d(pcs_all, total_pc_dim)
            presence = _pad_or_trunc_1d(pres_vec.astype(np.float32), G)

            full_vec = np.concatenate([pcs_all, presence], axis=0)
            full_vec = _pad_or_trunc_1d(full_vec, final_dim)
            patient_vectors[lid] = full_vec

        return cls(
            patient_vectors,
            group_names=group_names,
            group_n_comp=group_n_comp,
            pc_slices=pc_slices,
            total_pc_dim=total_pc_dim,
        )

    def encode_patient(self, lid: str) -> np.ndarray:
        vec = self.patient_vectors.get(str(lid), None)
        if vec is None:
            return np.zeros(self.output_dim, dtype=np.float32)
        return _pad_or_trunc_1d(vec, self.output_dim)
