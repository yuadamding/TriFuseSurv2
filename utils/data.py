"""Shared datasets and augmentation for TriFuseSurv multimodal survival."""

from __future__ import annotations

import os
import random
from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk

import torch
from torch.utils.data import Dataset

from trifusesurv.utils.clinical import ClinicalEncoder, ClinicalEncoderCompact
from trifusesurv.utils.radiomics import RadiomicsEncoder


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
        ct_col: str,
        mask_pt_col: str,
        mask_ln_col: str,
        clinical_encoder: Optional[ClinicalEncoder | ClinicalEncoderCompact],
        radiomics_encoder: Optional[RadiomicsEncoder],
        use_radiomics: bool = True,
        strict_files: bool = True,
        expected_dhw: Optional[Tuple[int, int, int]] = None,
        mode: str = "eval",
        track_mask_presence: bool = False,
    ):
        self.meta = meta.reset_index(drop=True)
        self.id_col = id_col
        self.time_col = time_col
        self.event_col = event_col
        self.ct_col = ct_col
        self.mask_pt_col = mask_pt_col
        self.mask_ln_col = mask_ln_col
        self.clinical_encoder = clinical_encoder
        self.radiomics_encoder = radiomics_encoder
        self.use_radiomics = bool(use_radiomics)
        self.strict_files = bool(strict_files)
        self.expected_dhw = tuple(expected_dhw) if expected_dhw is not None else None
        self.mode = mode
        self.track_mask_presence = bool(track_mask_presence)

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

        ct_path = str(row[self.ct_col])
        pt_path = str(row[self.mask_pt_col])
        ln_path = str(row[self.mask_ln_col])

        if (not os.path.isfile(ct_path)) or (not os.path.isfile(pt_path)) or (not os.path.isfile(ln_path)):
            if self.strict_files:
                raise RuntimeError(
                    f"Missing ct/pt/ln mask for pid={pid}: ct={ct_path} pt={pt_path} ln={ln_path}"
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

        if self.mode == "train":
            ct, pt, ln = rand_flip_3d(ct, pt, ln)
            ct = rand_intensity(ct)

        t = float(row[self.time_col])
        e = float(row[self.event_col])

        if self.clinical_encoder is not None and self.clinical_encoder.output_dim > 0:
            clin_t = torch.tensor(self.clinical_encoder.encode_row(row), dtype=torch.float32)
        else:
            clin_t = torch.zeros(0, dtype=torch.float32)

        if self.use_radiomics and self.radiomics_encoder is not None and self.radiomics_encoder.output_dim > 0:
            rad_t = torch.tensor(self.radiomics_encoder.encode_patient(pid), dtype=torch.float32)
        else:
            rad_t = torch.zeros(0, dtype=torch.float32)

        return ct, pt, ln, t, e, clin_t, rad_t, pid


class PreprocessedMoEDataset(_BasePreprocessedSurvivalDataset):
    """Legacy dataset that exposes CT and contour masks as a 3-channel image input."""

    def __getitem__(self, idx):
        ct, pt, ln, t, e, clin_t, rad_t, pid = self._load_case(idx)
        x = np.stack([ct, pt, ln], axis=0)

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


class PreprocessedContourAwareDataset(_BasePreprocessedSurvivalDataset):
    """CT-only dataset with PT/LN masks kept as localization labels."""

    def __getitem__(self, idx):
        ct, pt, ln, t, e, clin_t, rad_t, pid = self._load_case(idx)
        return (
            torch.tensor(ct[None, ...], dtype=torch.float32),
            torch.tensor(pt[None, ...], dtype=torch.float32),
            torch.tensor(ln[None, ...], dtype=torch.float32),
            torch.tensor(t, dtype=torch.float32),
            torch.tensor(e, dtype=torch.float32),
            clin_t,
            rad_t,
            pid,
        )
