"""Shared dataset and augmentation for TriFuseSurv multimodal survival."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import SimpleITK as sitk

import torch
from torch.utils.data import Dataset

from trifusesurv.utils.clinical import ClinicalEncoder, ClinicalEncoderCompact
from trifusesurv.utils.radiomics import RadiomicsEncoder


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PreprocessedMoEDataset(Dataset):
    """Dataset for multimodal survival training and evaluation.

    Returns (x_img, time, event, clinical, radiomics, patient_id)
    and optionally (pt_mask_present, ln_mask_present) if track_mask_presence=True.
    """

    def __init__(
        self,
        meta,
        *,
        ct_col: str,
        mask_pt_col: str,
        mask_ln_col: str,
        time_col: str,
        event_col: str,
        id_col: str,
        clinical_encoder: Optional[ClinicalEncoder | ClinicalEncoderCompact],
        radiomics_encoder: Optional[RadiomicsEncoder],
        target_shape=(128, 256, 256),
        mode: str = "eval",
        track_mask_presence: bool = False,
    ):
        self.meta = meta.reset_index(drop=True)
        self.ct_col = ct_col
        self.mask_pt_col = mask_pt_col
        self.mask_ln_col = mask_ln_col
        self.time_col = time_col
        self.event_col = event_col
        self.id_col = id_col
        self.clinical_encoder = clinical_encoder
        self.radiomics_encoder = radiomics_encoder
        self.target_shape = tuple(target_shape)
        self.mode = mode
        self.track_mask_presence = track_mask_presence

    def __len__(self):
        return len(self.meta)

    def _load_nii(self, path: str) -> np.ndarray:
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        # pad/crop to target shape
        out = np.zeros(self.target_shape, dtype=np.float32)
        slices_src = []
        slices_dst = []
        for i in range(3):
            s = arr.shape[i]
            t = self.target_shape[i]
            if s >= t:
                start = (s - t) // 2
                slices_src.append(slice(start, start + t))
                slices_dst.append(slice(0, t))
            else:
                start = (t - s) // 2
                slices_src.append(slice(0, s))
                slices_dst.append(slice(start, start + s))
        out[slices_dst[0], slices_dst[1], slices_dst[2]] = arr[slices_src[0], slices_src[1], slices_src[2]]
        return out

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        pid = str(row[self.id_col])

        ct = self._load_nii(row[self.ct_col])
        pt = self._load_nii(row[self.mask_pt_col]).clip(0, 1)
        ln = self._load_nii(row[self.mask_ln_col]).clip(0, 1)

        if self.mode == "train":
            ct, pt, ln = rand_flip_3d(ct, pt, ln)
            ct = rand_intensity(ct)

        x = np.stack([ct, pt, ln], axis=0)
        t = float(row[self.time_col])
        e = float(row[self.event_col])

        if self.clinical_encoder is not None and self.clinical_encoder.output_dim > 0:
            clin_t = torch.tensor(self.clinical_encoder.encode_row(row), dtype=torch.float32)
        else:
            clin_t = torch.zeros(0, dtype=torch.float32)

        if self.radiomics_encoder is not None:
            rad_t = torch.tensor(self.radiomics_encoder.encode_patient(pid), dtype=torch.float32)
        else:
            rad_t = torch.zeros(0, dtype=torch.float32)

        result = (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(t, dtype=torch.float32),
            torch.tensor(e, dtype=torch.float32),
            clin_t,
            rad_t,
            pid,
        )

        if self.track_mask_presence:
            pt_present = bool(float(pt.sum()) > 0.0)
            ln_present = bool(float(ln.sum()) > 0.0)
            return result + (pt_present, ln_present)

        return result
