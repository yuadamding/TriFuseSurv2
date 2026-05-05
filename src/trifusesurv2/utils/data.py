"""Shared datasets and augmentation for contour-aware TriFuseSurv survival."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk

import torch
from torch.utils.data import Dataset

from trifusesurv2.utils.clinical import ClinicalEncoder
from trifusesurv2.utils.radiomics import RadiomicsEncoder
from trifusesurv2.schema import NODE_TOPOLOGY_FEATURES


@lru_cache(maxsize=65536)
def _resolve_preprocessed_case_path_cached(raw: str, data_root: str, patient_id: str) -> str:
    if raw == "":
        return raw
    if os.path.isfile(raw):
        return raw

    root = Path(data_root) if data_root else None
    raw_path = Path(raw)
    candidates = []

    def add_candidate(candidate: Path):
        s = str(candidate)
        if s not in candidates:
            candidates.append(s)

    if root is not None:
        if not raw_path.is_absolute():
            add_candidate(root / raw_path)

        anchor = root.name
        parts = raw_path.parts
        if anchor in parts:
            idx = parts.index(anchor)
            suffix = Path(*parts[idx + 1 :])
            add_candidate(root / suffix)

        if patient_id:
            add_candidate(root / str(patient_id) / raw_path.name)

        # Last-resort recovery for slightly different layouts under the current
        # preprocessed root, for example after moving the cohort between systems.
        basename = raw_path.name
        if patient_id:
            for path_obj in root.glob(f"**/{patient_id}/{basename}"):
                add_candidate(path_obj)
        for path_obj in root.glob(f"**/{basename}"):
            if patient_id and str(patient_id) not in path_obj.parts:
                continue
            add_candidate(path_obj)

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return raw


def resolve_preprocessed_case_path(path: str, *, data_root: Optional[str] = None, patient_id: Optional[str] = None) -> str:
    raw = str(path or "").strip()
    return _resolve_preprocessed_case_path_cached(
        raw,
        str(Path(data_root)) if data_root else "",
        str(patient_id or ""),
    )


def rand_flip_3d(ct: np.ndarray, m1: np.ndarray, m2: np.ndarray, p: float = 0.5):
    if random.random() < p:
        ct = np.flip(ct, 0).copy()
        m1 = np.flip(m1, 0).copy()
        m2 = np.flip(m2, 0).copy()
    if random.random() < p:
        ct = np.flip(ct, 1).copy()
        m1 = np.flip(m1, 1).copy()
        m2 = np.flip(m2, 1).copy()
    if random.random() < p:
        ct = np.flip(ct, 2).copy()
        m1 = np.flip(m1, 2).copy()
        m2 = np.flip(m2, 2).copy()
    return ct, m1, m2


def rand_intensity(ct: np.ndarray, p: float = 0.3):
    if random.random() < p:
        ct = ct + 0.02 * np.random.randn(*ct.shape).astype(np.float32)
    if random.random() < p:
        scale = float(np.clip(1.0 + 0.05 * np.random.randn(), 0.9, 1.1))
        ct = ct * scale
    return np.clip(ct, 0.0, 1.0).astype(np.float32)


class _BasePreprocessedSurvivalDataset(Dataset):
    """Shared NIfTI loading and tabular encoding for survival datasets."""

    def __init__(
        self,
        meta,
        *,
        id_col: str,
        time_col: str,
        event_col: str,
        multi_time_cols: Optional[Tuple[str, ...]] = None,
        multi_event_cols: Optional[Tuple[str, ...]] = None,
        ct_col: str,
        mask_pt_col: str,
        mask_ln_col: str,
        clinical_encoder: Optional[ClinicalEncoder],
        radiomics_encoder: Optional[RadiomicsEncoder],
        use_radiomics: bool = True,
        strict_files: bool = True,
        expected_dhw: Optional[Tuple[int, int, int]] = None,
        data_root: Optional[str] = None,
        mode: str = "eval",
        spatial_augment: bool = True,
    ):
        self.meta = meta.reset_index(drop=True)
        self.id_col = id_col
        self.time_col = time_col
        self.event_col = event_col
        self.multi_time_cols = tuple(multi_time_cols or ())
        self.multi_event_cols = tuple(multi_event_cols or ())
        self.ct_col = ct_col
        self.mask_pt_col = mask_pt_col
        self.mask_ln_col = mask_ln_col
        self.clinical_encoder = clinical_encoder
        self.radiomics_encoder = radiomics_encoder
        self.use_radiomics = bool(use_radiomics)
        self.strict_files = bool(strict_files)
        self.expected_dhw = tuple(expected_dhw) if expected_dhw is not None else None
        self.data_root = str(Path(data_root)) if data_root else None
        self.mode = mode
        self.spatial_augment = bool(spatial_augment)

    def _load_nii(self, path: str) -> np.ndarray:
        img = sitk.ReadImage(str(path))
        return sitk.GetArrayFromImage(img).astype(np.float32)

    def _zeros_like_expected(self) -> np.ndarray:
        shape = self.expected_dhw if self.expected_dhw is not None else (128, 256, 256)
        return np.zeros(shape, dtype=np.float32)

    def __len__(self):
        return len(self.meta)

    def _load_case(self, idx: int):
        row = self.meta.iloc[idx]
        pid = str(row[self.id_col])

        ct_path_raw = str(row[self.ct_col])
        pt_path_raw = str(row[self.mask_pt_col])
        ln_path_raw = str(row[self.mask_ln_col])

        ct_path = resolve_preprocessed_case_path(ct_path_raw, data_root=self.data_root, patient_id=pid)
        pt_path = resolve_preprocessed_case_path(pt_path_raw, data_root=self.data_root, patient_id=pid)
        ln_path = resolve_preprocessed_case_path(ln_path_raw, data_root=self.data_root, patient_id=pid)

        if (not os.path.isfile(ct_path)) or (not os.path.isfile(pt_path)) or (not os.path.isfile(ln_path)):
            if self.strict_files:
                raise RuntimeError(
                    f"Missing ct/pt/ln mask for pid={pid}: "
                    f"ct={ct_path} (raw={ct_path_raw}) "
                    f"pt={pt_path} (raw={pt_path_raw}) "
                    f"ln={ln_path} (raw={ln_path_raw})"
                )
            ct = self._zeros_like_expected()
            pt = self._zeros_like_expected()
            ln = self._zeros_like_expected()
        else:
            ct = self._load_nii(ct_path)
            pt = (self._load_nii(pt_path) > 0.5).astype(np.float32)
            ln = (self._load_nii(ln_path) > 0.5).astype(np.float32)

        if self.expected_dhw is not None:
            if tuple(ct.shape) != self.expected_dhw:
                raise RuntimeError(f"[SHAPE] pid={pid} CT {tuple(ct.shape)} != expected {self.expected_dhw}")
            if tuple(pt.shape) != self.expected_dhw:
                raise RuntimeError(f"[SHAPE] pid={pid} PT {tuple(pt.shape)} != expected {self.expected_dhw}")
            if tuple(ln.shape) != self.expected_dhw:
                raise RuntimeError(f"[SHAPE] pid={pid} LN {tuple(ln.shape)} != expected {self.expected_dhw}")

        if self.mode == "train" and self.spatial_augment:
            ct, pt, ln = rand_flip_3d(ct, pt, ln)
            ct = rand_intensity(ct)
        elif self.mode == "train":
            ct = rand_intensity(ct)

        t = float(row[self.time_col])
        e = float(row[self.event_col])

        t_multi = []
        e_multi = []
        if self.multi_time_cols and self.multi_event_cols:
            for tcol, ecol in zip(self.multi_time_cols, self.multi_event_cols):
                t_multi.append(float(row[tcol]) if tcol in row.index else float("nan"))
                e_multi.append(float(row[ecol]) if ecol in row.index else float("nan"))
        else:
            t_multi.append(float(t))
            e_multi.append(float(e))

        if self.clinical_encoder is not None and self.clinical_encoder.output_dim > 0:
            clin_t = torch.tensor(self.clinical_encoder.encode_row(row), dtype=torch.float32)
        else:
            clin_t = torch.zeros(0, dtype=torch.float32)

        if self.use_radiomics and self.radiomics_encoder is not None and self.radiomics_encoder.output_dim > 0:
            rad_t = torch.tensor(self.radiomics_encoder.encode_patient(pid), dtype=torch.float32)
        else:
            rad_t = torch.zeros(0, dtype=torch.float32)

        return ct, pt, ln, t, e, np.asarray(t_multi, dtype=np.float32), np.asarray(e_multi, dtype=np.float32), clin_t, rad_t, pid

class PreprocessedContourAwareDataset(_BasePreprocessedSurvivalDataset):
    """CT-only dataset with PT/LN masks kept as localization labels."""

    def __getitem__(self, idx):
        ct, pt, ln, t, e, t_multi, e_multi, clin_t, rad_t, pid = self._load_case(idx)
        return (
            torch.tensor(ct[None, ...], dtype=torch.float32),
            torch.tensor(pt[None, ...], dtype=torch.float32),
            torch.tensor(ln[None, ...], dtype=torch.float32),
            torch.tensor(t, dtype=torch.float32),
            torch.tensor(e, dtype=torch.float32),
            torch.tensor(t_multi, dtype=torch.float32),
            torch.tensor(e_multi, dtype=torch.float32),
            clin_t,
            rad_t,
            pid,
        )


def _topology_json_candidates(root: Path, pid: str) -> list[Path]:
    slug = str(pid)
    return [
        root / f"{slug}.json",
        root / f"{slug}_topology.json",
        root / f"{slug}_node_topology.json",
        root / slug / "node_topology.json",
        root / slug / f"{slug}_node_topology.json",
    ]


def _read_node_topology_payload(root: Optional[str], pid: str) -> Optional[dict]:
    if not root:
        return None
    root_path = Path(root)
    for path in _topology_json_candidates(root_path, pid):
        if path.is_file():
            return json.loads(path.read_text())
    return None


def _topology_vector_from_payload(payload: Optional[dict]) -> tuple[np.ndarray, float]:
    vec = np.zeros((len(NODE_TOPOLOGY_FEATURES),), dtype=np.float32)
    if not payload:
        return vec, 0.0
    summary = payload.get("topology_summary", payload)
    present_any = False
    for idx, name in enumerate(NODE_TOPOLOGY_FEATURES):
        val = summary.get(name, np.nan) if isinstance(summary, dict) else np.nan
        try:
            x = float(val)
        except Exception:
            x = np.nan
        if np.isfinite(x):
            vec[idx] = x
            present_any = True
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec.astype(np.float32), float(present_any)


def _node_matrix_from_payload(payload: Optional[dict], *, max_nodes: int, node_dim: int) -> tuple[np.ndarray, np.ndarray]:
    mat = np.zeros((int(max_nodes), int(node_dim)), dtype=np.float32)
    presence = np.zeros((int(max_nodes),), dtype=np.float32)
    if not payload or int(max_nodes) <= 0 or int(node_dim) <= 0:
        return mat, presence
    nodes = payload.get("node_instances", [])
    for row_idx, node in enumerate(nodes[: int(max_nodes)]):
        vals: list[float] = []
        vals.append(float(node.get("voxel_count", 0.0)))
        vals.append(float(node.get("volume_mm3", 0.0)))
        centroid = node.get("centroid_xyz_mm", [0.0, 0.0, 0.0])
        vals.extend(float(x) for x in list(centroid)[:3])
        laterality = str(node.get("laterality", "unknown")).lower()
        vals.extend([
            1.0 if laterality == "left" else 0.0,
            1.0 if laterality == "right" else 0.0,
            1.0 if laterality == "midline" else 0.0,
            0.0 if laterality == "unknown" else 1.0,
        ])
        arr = np.asarray(vals, dtype=np.float32)
        keep = min(int(node_dim), int(arr.shape[0]))
        mat[row_idx, :keep] = arr[:keep]
        presence[row_idx] = 1.0
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat, presence


@dataclass
class NodeTopologyScaler:
    """Fold-wise scaler for compact node/topology numeric tokens."""

    topology_mean: np.ndarray
    topology_scale: np.ndarray
    node_mean: np.ndarray
    node_scale: np.ndarray

    @staticmethod
    def _precondition_topology(vec: np.ndarray) -> np.ndarray:
        out = np.asarray(vec, dtype=np.float32).copy()
        for name in ("node_count", "node_total_volume_mm3", "node_largest_volume_mm3"):
            if name in NODE_TOPOLOGY_FEATURES:
                idx = NODE_TOPOLOGY_FEATURES.index(name)
                out[idx] = np.log1p(max(float(out[idx]), 0.0))
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    @staticmethod
    def _precondition_nodes(mat: np.ndarray) -> np.ndarray:
        out = np.asarray(mat, dtype=np.float32).copy()
        if out.shape[-1] > 0:
            out[..., 0] = np.log1p(np.maximum(out[..., 0], 0.0))
        if out.shape[-1] > 1:
            out[..., 1] = np.log1p(np.maximum(out[..., 1], 0.0))
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    @classmethod
    def fit(
        cls,
        *,
        node_topology_dir: str,
        train_ids: list[str],
        max_nodes: int,
        node_dim: int,
    ) -> "NodeTopologyScaler":
        topo_rows: list[np.ndarray] = []
        node_rows: list[np.ndarray] = []
        for pid in train_ids:
            payload = _read_node_topology_payload(node_topology_dir, str(pid))
            topo, topo_present = _topology_vector_from_payload(payload)
            if topo_present > 0.5:
                topo_rows.append(cls._precondition_topology(topo))
            nodes, node_presence = _node_matrix_from_payload(payload, max_nodes=max_nodes, node_dim=node_dim)
            nodes = cls._precondition_nodes(nodes)
            for row in nodes[node_presence > 0.5]:
                node_rows.append(row.astype(np.float32))

        topo_dim = len(NODE_TOPOLOGY_FEATURES)
        topo_stack = np.stack(topo_rows, axis=0) if topo_rows else np.zeros((1, topo_dim), dtype=np.float32)
        node_stack = (
            np.stack(node_rows, axis=0)
            if node_rows and int(node_dim) > 0
            else np.zeros((1, int(max(node_dim, 0))), dtype=np.float32)
        )
        topo_mean = topo_stack.mean(axis=0).astype(np.float32)
        topo_scale = topo_stack.std(axis=0).astype(np.float32)
        topo_scale = np.where(topo_scale > 1e-6, topo_scale, 1.0).astype(np.float32)
        node_mean = node_stack.mean(axis=0).astype(np.float32)
        node_scale = node_stack.std(axis=0).astype(np.float32)
        node_scale = np.where(node_scale > 1e-6, node_scale, 1.0).astype(np.float32)
        return cls(topo_mean, topo_scale, node_mean, node_scale)

    def transform_topology(self, vec: np.ndarray) -> np.ndarray:
        x = self._precondition_topology(vec)
        return ((x - self.topology_mean) / self.topology_scale).astype(np.float32)

    def transform_nodes(self, mat: np.ndarray) -> np.ndarray:
        x = self._precondition_nodes(mat)
        if x.shape[-1] == 0:
            return x
        return ((x - self.node_mean) / self.node_scale).astype(np.float32)


def _pad_or_trunc_matrix_dim(mat: np.ndarray, dim: int) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if int(dim) <= 0 or mat.shape[-1] == int(dim):
        return mat
    if mat.shape[-1] > int(dim):
        return mat[:, : int(dim)].copy()
    out = np.zeros((mat.shape[0], int(dim)), dtype=np.float32)
    out[:, : mat.shape[-1]] = mat
    return out


class PreprocessedHabitatOOFDataset(_BasePreprocessedSurvivalDataset):
    """OOF/evaluation dataset that emits v2 grouped clinical/radiomics tokens.

    This dataset intentionally does not use the legacy flat ``ClinicalEncoder``
    or ``RadiomicsEncoder`` outputs.  It returns semantic clinical token
    matrices, habitat radiomics token matrices, optional node tokens, and an
    optional topology token for the 2.0.11 habitat-aligned model.
    """

    def __init__(
        self,
        meta,
        *,
        id_col: str,
        time_col: str,
        event_col: str,
        multi_time_cols: Optional[Tuple[str, ...]] = None,
        multi_event_cols: Optional[Tuple[str, ...]] = None,
        ct_col: str,
        mask_pt_col: str,
        mask_ln_col: str,
        clinical_token_encoder,
        radiomics_token_encoder=None,
        use_radiomics: bool = True,
        node_topology_dir: Optional[str] = None,
        node_topology_scaler: Optional[NodeTopologyScaler] = None,
        max_nodes: int = 0,
        node_token_dim: int = 0,
        clinical_token_dim: int = 0,
        radiomics_token_dim: int = 0,
        strict_files: bool = True,
        expected_dhw: Optional[Tuple[int, int, int]] = None,
        data_root: Optional[str] = None,
        mode: str = "eval",
        spatial_augment: Optional[bool] = None,
    ):
        if bool(node_topology_dir) and bool(spatial_augment):
            raise ValueError(
                "Spatial augmentation with node_topology_dir is disabled until node centroids/laterality "
                "and topology summaries are transformed with the image."
            )
        allow_spatial_augment = (not bool(node_topology_dir)) if spatial_augment is None else bool(spatial_augment)
        super().__init__(
            meta,
            id_col=id_col,
            time_col=time_col,
            event_col=event_col,
            multi_time_cols=multi_time_cols,
            multi_event_cols=multi_event_cols,
            ct_col=ct_col,
            mask_pt_col=mask_pt_col,
            mask_ln_col=mask_ln_col,
            clinical_encoder=None,
            radiomics_encoder=None,
            use_radiomics=use_radiomics,
            strict_files=strict_files,
            expected_dhw=expected_dhw,
            data_root=data_root,
            mode=mode,
            spatial_augment=allow_spatial_augment,
        )
        self.clinical_token_encoder = clinical_token_encoder
        self.radiomics_token_encoder = radiomics_token_encoder
        self.node_topology_dir = str(node_topology_dir or "")
        self.node_topology_scaler = node_topology_scaler
        self.max_nodes = int(max_nodes)
        self.node_token_dim = int(node_token_dim)
        self.clinical_token_dim = int(clinical_token_dim)
        self.radiomics_token_dim = int(radiomics_token_dim)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        ct, pt, ln, t, e, t_multi, e_multi, _clin_t, _rad_t, pid = self._load_case(idx)

        clin_mat, clin_presence = self.clinical_token_encoder.encode_row_token_matrix(row)
        clin_mat = _pad_or_trunc_matrix_dim(clin_mat, self.clinical_token_dim)
        if self.use_radiomics and self.radiomics_token_encoder is not None:
            rad_mat, rad_presence = self.radiomics_token_encoder.encode_patient_token_matrix(pid)
            rad_mat = _pad_or_trunc_matrix_dim(rad_mat, self.radiomics_token_dim)
        else:
            rad_mat = np.zeros((0, int(self.radiomics_token_dim)), dtype=np.float32)
            rad_presence = np.zeros((0,), dtype=np.float32)

        payload = _read_node_topology_payload(self.node_topology_dir, pid)
        topology_vec, topology_present = _topology_vector_from_payload(payload)
        node_mat, node_presence = _node_matrix_from_payload(
            payload,
            max_nodes=self.max_nodes,
            node_dim=self.node_token_dim,
        )
        if self.node_topology_scaler is not None:
            topology_vec = self.node_topology_scaler.transform_topology(topology_vec)
            node_mat = self.node_topology_scaler.transform_nodes(node_mat)

        return {
            "x": torch.tensor(ct[None, ...], dtype=torch.float32),
            "mask_pt": torch.tensor(pt[None, ...], dtype=torch.float32),
            "mask_ln": torch.tensor(ln[None, ...], dtype=torch.float32),
            "t": torch.tensor(t, dtype=torch.float32),
            "e": torch.tensor(e, dtype=torch.float32),
            "t_all": torch.tensor(t_multi, dtype=torch.float32),
            "e_all": torch.tensor(e_multi, dtype=torch.float32),
            "clinical_tokens": torch.tensor(clin_mat, dtype=torch.float32),
            "clinical_presence": torch.tensor(clin_presence, dtype=torch.float32),
            "radiomics_tokens": torch.tensor(rad_mat, dtype=torch.float32),
            "radiomics_presence": torch.tensor(rad_presence, dtype=torch.float32),
            "node_tokens": torch.tensor(node_mat, dtype=torch.float32),
            "node_presence": torch.tensor(node_presence, dtype=torch.float32),
            "topology_token": torch.tensor(topology_vec[None, :], dtype=torch.float32),
            "topology_presence": torch.tensor([topology_present], dtype=torch.float32),
            "pid": pid,
        }


class PreprocessedHabitatSurvivalDataset(PreprocessedHabitatOOFDataset):
    """Training/evaluation dataset for the v2 habitat-aligned survival path.

    The implementation is shared with the OOF explanation dataset so training,
    validation, prediction, and Grad-CAM all consume the same grouped token
    structure instead of drifting back to legacy flat clinical/radiomics vectors.
    """
