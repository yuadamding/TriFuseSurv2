#!/usr/bin/env python3
"""
trifusesurv.multimodal_survival.export_grouped_shap

OOF grouped SHAP (permutation Shapley) computed directly from fold checkpoints,
for the PT/LN SwinUNETR-token MoE discrete-time survival model.

Core guarantees (what you asked):
1) SHAP is computed for SURVIVAL probability at horizon t:
      f(.) = S(t) = 1 - Risk(t)
   so every feature’s SHAP is “with respect to survival prediction”.

2) Image SHAP is PT vs LN lesion tokens:
      IMG:PT = tokens [1,2] (PT_intra, PT_peri)
      IMG:LN = tokens [3,4] (LN_intra, LN_peri)
   (GLOBAL token 0 is not included in PT/LN.)

3) Validity / explainability:
   For each patient:
      sum_j SHAP_j ≈ S_full(t) - S_base(t)
   (checked per patient; script raises if violated).

4) Your requirement on availability:
   - Every patient has a finite IMG:PT SHAP value.
   - For every patient whose LN mask is non-empty, IMG:LN SHAP is finite.
   We also export flags:
      pt_mask_present / ln_mask_present (from masks)
      pt_token_present / ln_token_present (from model pres flags)

Other supported features (kept):
- LoRA checkpoints: auto-inject by scanning *.lora_A.weight
- EMA/SWA overlay: --weights {last,ema,swa}
- Split mask patch-embed support
- Radiomics encoder: PCs + presence bits (NO counts), align dim to ckpt if needed
- Clinical encoder: ckpt-dim matched deterministic encoder (fixes strict-load mismatch)

Outputs:
- fold_XX/shap_oof_grouped_foldXX.npz
- oof_shap_grouped_all.npz
- oof_shap_grouped_wide.csv
- oof_group_importance.csv
- oof_group_values_wide.csv (if --export_group_values)
- manifest.json
"""

from __future__ import annotations

import os, json, contextlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Sequence, Set

import numpy as np
import pandas as pd
import SimpleITK as sitk

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from trifusesurv.models.survival_model import SwinUNETRTokenMoEDiscrete
from trifusesurv.models.lora import LoRALinear, inject_lora_from_state_dict
from trifusesurv.utils.clinical import (
    ClinicalEncoderCompact, CLINICAL_SCHEMA, DEFAULT_CLINICAL_COLS,
    ENDPOINT_MAP, parse_ordinal_value,
)
from trifusesurv.utils.radiomics import RadiomicsEncoder, _pad_or_trunc_1d
from trifusesurv.utils.survival import set_seed, seed_worker


# =============================================================================
# Device / AMP helpers
# =============================================================================


def parse_device(device_str: str) -> torch.device:
    dev = str(device_str).strip().lower()
    if dev in ("", "auto"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dev == "cpu":
        return torch.device("cpu")
    if dev == "cuda":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dev.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device but CUDA is not available.")
        return torch.device(dev)
    raise ValueError(f"--device must be cpu|cuda|cuda:N|auto, got: {device_str}")


def bind_cuda_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.set_device(int(device.index) if device.index is not None else 0)


def make_autocast(device: torch.device, enabled: bool):
    amp_enabled = bool(enabled and device.type == "cuda")
    if not amp_enabled:
        return lambda: contextlib.nullcontext()
    try:
        return lambda: torch.amp.autocast("cuda", enabled=True)
    except Exception:
        return lambda: torch.cuda.amp.autocast(enabled=True)


# LoRALinear, inject_lora_from_state_dict -> imported from trifusesurv.models.lora


def sd_has_lora(sd: Dict[str, torch.Tensor]) -> bool:
    return any(k.endswith(".lora_A.weight") for k in sd.keys())


# =============================================================================
# EMA/SWA overlay helpers
# =============================================================================
def strip_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith("module."):
            k2 = k2[len("module."):]
        if k2.startswith("model."):
            k2 = k2[len("model."):]
        out[k2] = v
    return out


def _extract_tensor_mapping(d: Any) -> Optional[Dict[str, torch.Tensor]]:
    if not isinstance(d, dict):
        return None
    if d and all(torch.is_tensor(v) for v in d.values()):
        return d
    for key in ("shadow", "avg", "averaged", "weights", "state_dict", "model_state", "params"):
        v = d.get(key, None)
        if isinstance(v, dict) and v and all(torch.is_tensor(x) for x in v.values()):
            return v
    return None


def apply_partial_state_dict(model: nn.Module, partial: Dict[str, torch.Tensor], *, tag: str) -> int:
    msd = model.state_dict()
    matched = {}
    for k, v in partial.items():
        if k in msd and tuple(msd[k].shape) == tuple(v.shape):
            matched[k] = v
    if not matched:
        print(f"[{tag}] no matching tensors to apply.")
        return 0
    msd.update(matched)
    model.load_state_dict(msd, strict=True)
    print(f"[{tag}] applied {len(matched)} tensor(s).")
    return int(len(matched))


# =============================================================================
# IO / splits / utilities
# =============================================================================
def _read_id_list(path: str) -> List[str]:
    with open(path, "r") as f:
        ids = [ln.strip() for ln in f.read().splitlines()]
    return [x for x in ids if x]


def load_precomputed_splits(cv_folds: int, *, splits_dir: str = "", splits_csv: str = "") -> Dict[int, Dict[str, List[str]]]:
    if bool(splits_dir) == bool(splits_csv):
        raise ValueError("Provide exactly one of --splits_dir or --splits_csv.")
    out: Dict[int, Dict[str, List[str]]] = {}

    if splits_dir:
        for f in range(int(cv_folds)):
            fold_dir = os.path.join(splits_dir, f"fold_{f:02d}")
            tr = _read_id_list(os.path.join(fold_dir, "train_ids.txt"))
            va = _read_id_list(os.path.join(fold_dir, "val_ids.txt"))
            te = _read_id_list(os.path.join(fold_dir, "test_ids.txt"))
            out[f] = {"train": tr, "val": va, "test": te}
        return out

    df = pd.read_csv(splits_csv, dtype={"patient_id": str, "split": str})
    need = {"patient_id", "fold", "split"}
    if not need.issubset(df.columns):
        raise ValueError(f"--splits_csv must contain {sorted(need)}; got {list(df.columns)}")
    df = df.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(int)
    df["split"] = df["split"].astype(str).str.lower()

    for f in range(int(cv_folds)):
        dff = df[df["fold"] == f]
        if dff.empty:
            raise ValueError(f"--splits_csv has no rows for fold={f}")
        out[f] = {
            "train": dff.loc[dff["split"] == "train", "patient_id"].tolist(),
            "val":   dff.loc[dff["split"] == "val",   "patient_id"].tolist(),
            "test":  dff.loc[dff["split"] == "test",  "patient_id"].tolist(),
        }
    return out


def select_df_by_ids(meta: pd.DataFrame, ids: List[str], id_col: str, strict: bool, tag: str) -> pd.DataFrame:
    ids = [str(x) for x in ids]
    have = set(meta[id_col].astype(str).tolist())
    missing = [x for x in ids if x not in have]
    if missing:
        msg = f"[SPLIT][{tag}] {len(missing)} id(s) missing in meta. First: {missing[:10]}"
        if strict:
            raise RuntimeError(msg)
        print("[WARN]", msg, "-> dropping missing")
        ids = [x for x in ids if x in have]
    return meta[meta[id_col].astype(str).isin(set(ids))].copy().reset_index(drop=True)


def read_nii(path: str, dtype=np.float32) -> np.ndarray:
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (D,H,W)
    arr = np.asarray(arr, dtype=dtype)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def parse_ckpt_fold_list(items: List[str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for it in items:
        if ":" not in it:
            raise ValueError(f"--ckpt_fold must be FOLD:/path/to/ckpt.pt, got: {it}")
        a, b = it.split(":", 1)
        f = int(a)
        p = b.strip()
        if not p:
            raise ValueError(f"Empty path in --ckpt_fold {it}")
        out[f] = p
    return out


# _pad_or_trunc_1d -> imported from trifusesurv.utils.radiomics


# =============================================================================
# Clinical encoding (ckpt-dim matched)
# =============================================================================
def build_global_categorical_maps(meta: pd.DataFrame, clinical_cols: List[str]) -> Dict[str, Dict[str, int]]:
    maps: Dict[str, Dict[str, int]] = {}
    for col in clinical_cols:
        schema = CLINICAL_SCHEMA.get(col, "auto")
        if schema in ("numeric", "ordinal"):
            continue
        if col not in meta.columns:
            continue
        s = meta[col]
        non_na = s.dropna().astype(str).map(lambda x: x.strip())
        non_na = non_na[non_na != ""]
        if len(non_na) == 0:
            continue
        cats = sorted(set(non_na.tolist()))
        maps[col] = {c: i for i, c in enumerate(cats)}
    return maps


# _ClinSpec, ClinicalEncoder -> imported as ClinicalEncoderCompact from trifusesurv.utils.clinical


def clinical_raw_value_global(row: Optional[pd.Series], col: str, cat_maps: Dict[str, Dict[str, int]]) -> float:
    if row is None or col not in row.index:
        return float("nan")
    v = row.get(col, None)
    if v is None or pd.isna(v):
        return float("nan")

    schema = CLINICAL_SCHEMA.get(col, "auto")
    if schema == "numeric":
        try:
            return float(v)
        except Exception:
            return float("nan")
    if schema == "ordinal":
        try:
            x = parse_ordinal_value(col, v)
            return float(x) if np.isfinite(x) else float("nan")
        except Exception:
            return float("nan")

    s = str(v).strip()
    if s == "":
        return float("nan")
    mp = cat_maps.get(col, {})
    if s not in mp:
        return float("nan")
    return float(mp[s])


# RadiomicsEncoderMeta -> imported as RadiomicsEncoder from trifusesurv.utils.radiomics


@torch.no_grad()
def rad_group_l2(rad_x: Optional[torch.Tensor], idx_list: Optional[List[int]]) -> float:
    if rad_x is None or idx_list is None or len(idx_list) == 0:
        return float("nan")
    idx = torch.as_tensor(sorted(set(int(i) for i in idx_list)), device=rad_x.device, dtype=torch.long)
    sub = rad_x.index_select(1, idx)
    v = torch.linalg.norm(sub, dim=1)
    return float(v.item())


@torch.no_grad()
def img_group_token_norm(tok_x: torch.Tensor, pres_x: torch.Tensor, idxs: List[int]) -> float:
    if tok_x is None or pres_x is None or not idxs:
        return float("nan")
    idx = torch.as_tensor(sorted(set(int(i) for i in idxs)), device=tok_x.device, dtype=torch.long)
    sub = tok_x.index_select(1, idx)  # (B,n,D)
    pres = pres_x.to(torch.bool).index_select(1, idx).unsqueeze(-1)  # (B,n,1)
    sub = sub * pres.to(sub.dtype)
    norms = torch.linalg.norm(sub, dim=2)  # (B,n)
    return float(norms.sum(dim=1).item())


# =============================================================================
# Dataset (eval)
# =============================================================================
class PreprocessedMoEDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        id_col: str,
        time_col: str,
        event_col: str,
        ct_col: str,
        mask_pt_col: str,
        mask_ln_col: str,
        clinical_encoder: Optional[ClinicalEncoderCompact],
        radiomics_encoder: Optional[RadiomicsEncoder],
        use_radiomics: bool,
        strict_files: bool,
        expected_dhw: Optional[Tuple[int,int,int]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.id_col = id_col
        self.time_col = time_col
        self.event_col = event_col
        self.ct_col = ct_col
        self.mask_pt_col = mask_pt_col
        self.mask_ln_col = mask_ln_col
        self.clin_enc = clinical_encoder
        self.rad_enc = radiomics_encoder
        self.use_radiomics = bool(use_radiomics)
        self.strict_files = bool(strict_files)
        self.expected_dhw = tuple(expected_dhw) if expected_dhw is not None else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        pid = str(row[self.id_col])

        ct_path = str(row[self.ct_col])
        pt_path = str(row[self.mask_pt_col])
        ln_path = str(row[self.mask_ln_col])

        if (not os.path.isfile(ct_path)) or (not os.path.isfile(pt_path)) or (not os.path.isfile(ln_path)):
            if self.strict_files:
                raise RuntimeError(f"Missing ct/pt/ln mask for pid={pid}: ct={ct_path} pt={pt_path} ln={ln_path}")
            shape = self.expected_dhw if self.expected_dhw is not None else (128, 256, 256)
            ct = np.zeros(shape, dtype=np.float32)
            pt = np.zeros(shape, dtype=np.float32)
            ln = np.zeros(shape, dtype=np.float32)
        else:
            ct = read_nii(ct_path, dtype=np.float32)
            pt = (read_nii(pt_path, dtype=np.float32) > 0.5).astype(np.float32)
            ln = (read_nii(ln_path, dtype=np.float32) > 0.5).astype(np.float32)

        if self.expected_dhw is not None:
            if tuple(ct.shape) != self.expected_dhw:
                raise RuntimeError(f"[SHAPE] pid={pid} CT {tuple(ct.shape)} != expected {self.expected_dhw}")
            if tuple(pt.shape) != self.expected_dhw:
                raise RuntimeError(f"[SHAPE] pid={pid} PT {tuple(pt.shape)} != expected {self.expected_dhw}")
            if tuple(ln.shape) != self.expected_dhw:
                raise RuntimeError(f"[SHAPE] pid={pid} LN {tuple(ln.shape)} != expected {self.expected_dhw}")

        pt_mask_present = bool(float(pt.sum()) > 0.0)
        ln_mask_present = bool(float(ln.sum()) > 0.0)

        x = torch.from_numpy(np.stack([ct, pt, ln], axis=0).astype(np.float32))  # (3,D,H,W)
        t = torch.tensor(float(row[self.time_col]), dtype=torch.float32)
        e = torch.tensor(float(row[self.event_col]), dtype=torch.float32)

        if self.clin_enc is not None and self.clin_enc.output_dim > 0:
            clin_vec = self.clin_enc.encode_row(row)
            clin_vec = _pad_or_trunc_1d(clin_vec, int(self.clin_enc.output_dim))
            clin_t = torch.from_numpy(clin_vec).float()
        else:
            clin_t = torch.zeros(0, dtype=torch.float32)

        if self.use_radiomics and (self.rad_enc is not None) and (self.rad_enc.output_dim > 0):
            rad_vec = self.rad_enc.encode_patient(pid)
            rad_t = torch.from_numpy(rad_vec).float()
        else:
            rad_t = torch.zeros(0, dtype=torch.float32)

        return x, t, e, clin_t, rad_t, pid, pt_mask_present, ln_mask_present


# =============================================================================
# Checkpoint helpers
# =============================================================================
def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model_state", "model", "state_dict", "net", "network"):
            v = ckpt.get(key, None)
            if isinstance(v, dict) and v and all(torch.is_tensor(x) for x in v.values()):
                return v
        if ckpt and all(torch.is_tensor(x) for x in ckpt.values()):
            return ckpt
    raise RuntimeError("Could not find a state_dict in checkpoint.")


def ckpt_uses_split_patch(sd: Dict[str, torch.Tensor]) -> bool:
    for k in sd.keys():
        if "patch_embed.proj.conv_ct." in k or "patch_embed.proj.conv_mask." in k:
            return True
    return False


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(v)


def infer_backbone_runtime_overrides(ckpt: Any, sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    ck_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    if isinstance(ck_args, dict):
        if "force_presence_from_raw_masks" in ck_args:
            out["force_presence_from_raw_masks"] = _coerce_bool(ck_args["force_presence_from_raw_masks"])
        if "raw_mask_threshold" in ck_args:
            out["raw_mask_threshold"] = float(ck_args["raw_mask_threshold"])
        if "fallback_peri_to_intra" in ck_args:
            out["fallback_peri_to_intra"] = _coerce_bool(ck_args["fallback_peri_to_intra"])

    if not out and sd_has_lora(sd):
        out.update(
            force_presence_from_raw_masks=True,
            raw_mask_threshold=0.5,
            fallback_peri_to_intra=True,
        )
    return out


def infer_dims_from_sd(sd: Dict[str, torch.Tensor]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if "img_tok_ln.weight" in sd:
        out["fused_dim"] = int(sd["img_tok_ln.weight"].numel())
    if "img_backbone.token_type" in sd:
        out["img_token_dim"] = int(sd["img_backbone.token_type"].shape[1])
    if "img_backbone.token_mlp.0.3.weight" in sd:
        out["token_mlp_hidden_dim"] = int(sd["img_backbone.token_mlp.0.0.weight"].shape[0])
    if "fuse_projs.0.1.weight" in sd:
        out["img_proj_hidden_dim"] = int(sd["fuse_projs.0.1.weight"].shape[0])
    if "img_tok_ffn.1.weight" in sd:
        out["img_tok_ffn_hidden_dim"] = int(sd["img_tok_ffn.1.weight"].shape[0])
    if "img_post_mlp.1.weight" in sd:
        out["img_post_hidden_dim"] = int(sd["img_post_mlp.1.weight"].shape[0])
    if "gate_mlp.2.weight" in sd:
        out["gate_hidden_dim"] = int(sd["gate_mlp.2.weight"].shape[0])
    if "rad_proj.2.weight" in sd:
        out["rad_hidden_dim"] = int(sd["rad_proj.2.weight"].shape[0])
    if "rad_proj.0.weight" in sd:
        out["radiomics_dim_ckpt"] = int(sd["rad_proj.0.weight"].numel())

    if "clin_proj.0.weight" in sd:
        out["clinical_dim_ckpt"] = int(sd["clin_proj.0.weight"].numel())
    elif "clin_proj.2.weight" in sd:
        out["clinical_dim_ckpt"] = int(sd["clin_proj.2.weight"].shape[1])

    return out


def strict_load(model: nn.Module, sd: Dict[str, torch.Tensor], *, fold: int, ckpt_path: str):
    try:
        model.load_state_dict(sd, strict=True)
        print(f"[CKPT][fold {fold:02d}] strict load OK")
        return
    except RuntimeError as e:
        msg = (
            f"[CKPT][fold {fold:02d}] STRICT load FAILED for {ckpt_path}\n"
            f"First error:\n{e}\n"
        )
        raise RuntimeError(msg)


# SwinUNETRTokenMoEDiscrete -> imported from trifusesurv.models.survival_model


# =============================================================================
# Group specs + SHAP
# =============================================================================
@dataclass
class GroupSpec:
    name: str
    kind: str
    img_token_indices: Optional[List[int]] = None
    vec_indices: Optional[List[int]] = None
    value_indices: Optional[List[int]] = None


def slice_to_idx(s: slice) -> List[int]:
    return list(range(int(s.start), int(s.stop)))


def build_group_specs(
    *,
    clinical_cols: List[str],
    clin_groups: Dict[str, List[int]],
    clin_dim: int,
    rad_pc_slices: Dict[str, slice],
    rad_total_pc_dim: int,
    rad_group_names: Optional[List[str]],
    rad_dim: int,
) -> List[GroupSpec]:
    out: List[GroupSpec] = []

    # PT vs LN lesion tokens only (GLOBAL token 0 excluded)
    out.append(GroupSpec("IMG:PT", "img", img_token_indices=[1, 2]))
    out.append(GroupSpec("IMG:LN", "img", img_token_indices=[3, 4]))

    if int(rad_dim) > 0:
        req = {"PT_intra", "PT_peri", "LN_intra", "LN_peri"}
        if not req.issubset(set(rad_pc_slices.keys())):
            missing = sorted(list(req - set(rad_pc_slices.keys())))
            raise RuntimeError(f"[RAD] pc_slices missing keys: {missing}")

        G = int(len(rad_group_names) if (rad_group_names is not None) else 0)
        if G <= 0:
            rad_group_names = ["PT_intra", "PT_peri", "LN_intra", "LN_peri"]
            G = 4

        group_index = {str(g): i for i, g in enumerate(list(rad_group_names))}
        for k in ("PT_intra", "PT_peri", "LN_intra", "LN_peri"):
            if k not in group_index:
                raise RuntimeError(f"[RAD] rad_group_names missing key {k}. Got: {list(rad_group_names)}")

        pc_off = int(rad_total_pc_dim)
        pres_off = pc_off

        def pres_idx(g: str) -> int:
            return pres_off + int(group_index[g])

        pc_pt_intra = slice_to_idx(rad_pc_slices["PT_intra"])
        pc_pt_peri  = slice_to_idx(rad_pc_slices["PT_peri"])
        pc_ln_intra = slice_to_idx(rad_pc_slices["LN_intra"])
        pc_ln_peri  = slice_to_idx(rad_pc_slices["LN_peri"])
        pc_ln = pc_ln_intra + pc_ln_peri

        rad_pt_intra_vec = sorted(set(pc_pt_intra + [pres_idx("PT_intra")]))
        rad_pt_peri_vec  = sorted(set(pc_pt_peri  + [pres_idx("PT_peri")]))
        rad_ln_vec       = sorted(set(pc_ln + [pres_idx("LN_intra"), pres_idx("LN_peri")]))

        out.append(GroupSpec("RAD:PT-intra", "rad", vec_indices=rad_pt_intra_vec, value_indices=pc_pt_intra))
        out.append(GroupSpec("RAD:PT-peri",  "rad", vec_indices=rad_pt_peri_vec,  value_indices=pc_pt_peri))
        out.append(GroupSpec("RAD:LN",       "rad", vec_indices=rad_ln_vec,       value_indices=pc_ln))

    for col in clinical_cols:
        idxs = clin_groups.get(col, []) or []
        idxs = [int(i) for i in idxs if 0 <= int(i) < int(clin_dim)]
        out.append(GroupSpec(f"CLIN:{col}", "clin", vec_indices=idxs))

    return out


def _copy_cols_inplace_(dst: Optional[torch.Tensor], src: Optional[torch.Tensor], cols: Optional[Sequence[int]]):
    if dst is None or src is None or cols is None:
        return
    cols = [int(i) for i in cols if int(i) >= 0]
    if len(cols) == 0:
        return
    idx = torch.as_tensor(sorted(set(cols)), device=dst.device, dtype=torch.long)
    dst.index_copy_(1, idx, src.index_select(1, idx))


class HeadFromTokens(nn.Module):
    def __init__(self, base: SwinUNETRTokenMoEDiscrete):
        super().__init__()
        self.base = base

    @torch.no_grad()
    def forward_survival(
        self,
        tok: torch.Tensor,
        pres: torch.Tensor,
        clinical: Optional[torch.Tensor],
        radiomics: Optional[torch.Tensor],
        *,
        horizon_days: float,
    ) -> torch.Tensor:
        """
        Returns survival probability S(t) = 1 - Risk(t).
        This is the scalar target used for SHAP.
        """
        b = self.base
        B, E, _ = tok.shape
        pres_bool = pres.to(torch.bool)

        fused = [b.fuse_projs[i](tok[:, i, :]) for i in range(E)]
        stacked = torch.stack(fused, dim=1)
        stacked = stacked * pres_bool.unsqueeze(-1).to(stacked.dtype)

        q = b.img_tok_ln(stacked)
        attn_out, _ = b.img_attn(q, q, q, key_padding_mask=(~pres_bool))
        attn_out = attn_out * pres_bool.unsqueeze(-1).to(attn_out.dtype)
        attn_out = (attn_out + stacked) * pres_bool.unsqueeze(-1).to(attn_out.dtype)
        attn_out = (attn_out + b.img_tok_ffn(attn_out)) * pres_bool.unsqueeze(-1).to(attn_out.dtype)

        gate_logits = b.gate_mlp(attn_out.reshape(B, -1).float()).float()
        neg_inf = torch.finfo(gate_logits.dtype).min
        gate_logits = gate_logits.masked_fill(~pres_bool, neg_inf)
        all_abs = ~pres_bool.any(dim=1)
        if all_abs.any():
            gate_logits = gate_logits.clone()
            gate_logits[all_abs, 0] = 0.0
        gate = torch.softmax(gate_logits, dim=1).to(attn_out.dtype)

        fused_img = (attn_out * gate.unsqueeze(-1)).sum(dim=1)
        fused_img = fused_img + b.img_post_mlp(fused_img)

        chunks = [fused_img]
        if b.use_clin and clinical is not None and clinical.numel() > 0:
            chunks.append(b.clin_proj(clinical))
        if b.use_rad and radiomics is not None and radiomics.numel() > 0:
            chunks.append(b.rad_proj(radiomics))

        h = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
        logits = b.surv_head(h)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        risk = b.hazards_to_risk(logits, horizon_days=float(horizon_days))
        surv = (1.0 - risk).clamp(0.0, 1.0)
        return surv


@torch.no_grad()
def permutation_shapley_survival(
    head: HeadFromTokens,
    tok_x: torch.Tensor, pres_x: torch.Tensor,
    clin_x: Optional[torch.Tensor], rad_x: Optional[torch.Tensor],
    tok_base: torch.Tensor, pres_base: torch.Tensor,
    clin_base: Optional[torch.Tensor], rad_base: Optional[torch.Tensor],
    groups: List[GroupSpec],
    *,
    horizon_days: float,
    n_perm: int,
    rng: np.random.RandomState,
    additivity_tol: float = 5e-5,
) -> Tuple[np.ndarray, float, float]:
    """
    Shapley for survival probability f(.) = S(t).
    Returns: shap (P,), base_surv, full_surv
    """
    P = len(groups)
    base_surv = float(head.forward_survival(tok_base, pres_base, clin_base, rad_base, horizon_days=horizon_days).item())
    full_surv = float(head.forward_survival(tok_x, pres_x, clin_x, rad_x, horizon_days=horizon_days).item())

    shap = np.zeros((P,), dtype=np.float64)
    perm = np.arange(P, dtype=np.int64)

    tok_cur = tok_base.clone()
    pres_cur = pres_base.clone()
    clin_cur = None if clin_base is None else clin_base.clone()
    rad_cur  = None if rad_base  is None else rad_base.clone()

    for _ in range(int(n_perm)):
        rng.shuffle(perm)

        tok_cur.copy_(tok_base)
        pres_cur.copy_(pres_base)
        if clin_cur is not None and clin_base is not None:
            clin_cur.copy_(clin_base)
        if rad_cur is not None and rad_base is not None:
            rad_cur.copy_(rad_base)

        prev = float(head.forward_survival(tok_cur, pres_cur, clin_cur, rad_cur, horizon_days=horizon_days).item())

        for j in perm.tolist():
            g = groups[j]
            if g.kind == "img":
                for ti in g.img_token_indices or []:
                    tok_cur[:, ti, :].copy_(tok_x[:, ti, :])
                    pres_cur[:, ti].copy_(pres_x[:, ti])
            elif g.kind == "clin":
                _copy_cols_inplace_(clin_cur, clin_x, g.vec_indices)
            elif g.kind == "rad":
                _copy_cols_inplace_(rad_cur, rad_x, g.vec_indices)
            else:
                raise ValueError(g.kind)

            cur = float(head.forward_survival(tok_cur, pres_cur, clin_cur, rad_cur, horizon_days=horizon_days).item())
            shap[j] += (cur - prev)
            prev = cur

    shap /= float(n_perm)
    shap = shap.astype(np.float32)

    resid = float(shap.sum() - (full_surv - base_surv))
    if abs(resid) > float(additivity_tol):
        raise RuntimeError(f"[SHAP] additivity check failed: sum(shap)-(full-base)={resid:.6e}")

    return shap, base_surv, full_surv


# =============================================================================
# Args + main
# =============================================================================
def parse_args():
    import argparse
    p = argparse.ArgumentParser()

    p.add_argument("--meta_csv", required=True)
    p.add_argument("--id_col", type=str, default="patient_id")
    p.add_argument("--ct_col", type=str, default="ct_out_path")
    p.add_argument("--mask_pt_col", type=str, default="mask_pt_out_path")
    p.add_argument("--mask_ln_col", type=str, default="mask_ln_out_path")

    p.add_argument("--endpoint", type=str, default="OS", choices=["OS", "DSS", "DFS"])
    p.add_argument("--time_col", type=str, default="")
    p.add_argument("--event_col", type=str, default="")

    p.add_argument("--keep_bad_status", action="store_true")
    p.add_argument("--keep_unmatched_survival", action="store_true")

    p.add_argument("--splits_dir", type=str, default="")
    p.add_argument("--splits_csv", type=str, default="")
    p.add_argument("--cv_folds", type=int, default=4)
    p.add_argument("--strict_splits", action="store_true")

    p.add_argument("--ckpt_fold", action="append", default=[], help="Repeat: FOLD:/path/to/ckpt.pt")
    p.add_argument("--out_dir", type=str, default="runs/shap_oof_grouped")

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--n_perm", type=int, default=32)
    p.add_argument("--bg_size", type=int, default=64)

    p.add_argument("--weights", type=str, default="last", choices=["last", "ema", "swa"])

    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_scope", type=str, default="both", choices=["pt", "ln", "both"])

    p.add_argument("--export_group_values", dest="export_group_values", action="store_true")
    p.add_argument("--no_export_group_values", dest="export_group_values", action="store_false")
    p.set_defaults(export_group_values=True)

    p.add_argument("--sanity_check", action="store_true")
    p.add_argument("--no_sanity_check", dest="sanity_check", action="store_false")
    p.set_defaults(sanity_check=True)

    p.add_argument("--clinical_cols", type=str, nargs="*", default=DEFAULT_CLINICAL_COLS)
    p.add_argument("--use_radiomics", action="store_true")
    p.add_argument("--radiomics_root", type=str, default="radiomics_features/radiomics_features")
    p.add_argument("--radiomics_pca_total_components", type=int, default=100)

    p.add_argument("--img_size", type=int, nargs=3, default=[128, 256, 256])
    p.add_argument("--feature_size", type=int, default=96)
    p.add_argument("--depths", type=int, nargs=4, default=[2, 2, 18, 2])
    p.add_argument("--num_heads", type=int, nargs=4, default=[3, 6, 12, 24])
    p.add_argument("--drop_rate", type=float, default=0.0)
    p.add_argument("--attn_drop_rate", type=float, default=0.0)
    p.add_argument("--dropout_path_rate", type=float, default=0.0)
    p.add_argument("--use_checkpoint", action="store_true")

    p.add_argument("--img_token_dim", type=int, default=0)
    p.add_argument("--token_mlp_dropout", type=float, default=0.40)
    p.add_argument("--token_mlp_hidden_dim", type=int, default=0)
    p.add_argument("--token_dropout", type=float, default=0.10)

    p.add_argument("--attn_mask_bias", type=float, default=2.0)
    p.add_argument("--use_multiscale", action="store_true")
    p.add_argument("--mask_interp", type=str, default="nearest", choices=["nearest", "trilinear"])
    p.add_argument("--min_roi_frac", type=float, default=1e-5)
    p.add_argument("--min_roi_voxels_deep", type=int, default=8)

    p.add_argument("--pt_shell_radius", type=int, default=5)
    p.add_argument("--ln_shell_radius", type=int, default=5)

    p.add_argument("--shell_body_from_ct", action="store_true")
    p.add_argument("--body_ct_thr", type=str, default="auto")
    p.add_argument("--body_ct_thr_hu", type=float, default=-500.0)
    p.add_argument("--body_close_r", type=int, default=2)
    p.add_argument("--body_max_frac", type=float, default=0.995)

    p.add_argument("--strict_swinvit_layout", dest="strict_swinvit_layout", action="store_true")
    p.add_argument("--no_strict_swinvit_layout", dest="strict_swinvit_layout", action="store_false")
    p.set_defaults(strict_swinvit_layout=True)
    p.add_argument("--debug_swinvit_layout", action="store_true")

    p.add_argument("--time_bin_width_days", type=float, default=180.0)
    p.add_argument("--risk_horizon_days", type=float, default=365.0)  # horizon for S(t)

    args = p.parse_args()

    if args.time_col == "" or args.event_col == "":
        tcol, ecol = ENDPOINT_MAP[args.endpoint]
        if args.time_col == "":
            args.time_col = tcol
        if args.event_col == "":
            args.event_col = ecol

    if bool(args.splits_dir) and bool(args.splits_csv):
        raise ValueError("Provide only one of --splits_dir or --splits_csv.")
    if not args.ckpt_fold:
        raise ValueError("You must provide --ckpt_fold for all folds.")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    device = parse_device(args.device)
    bind_cuda_device(device)
    autocast_ctx = make_autocast(device, enabled=bool(args.amp))
    print(f"[info] device={device} amp={bool(args.amp and device.type=='cuda')} weights={args.weights} shap_target=S(t)")

    ckpt_map = parse_ckpt_fold_list(args.ckpt_fold)
    missing_folds = [f for f in range(int(args.cv_folds)) if f not in ckpt_map]
    if missing_folds:
        raise RuntimeError(f"Missing checkpoint(s) for folds: {missing_folds}")

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.meta_csv, dtype={args.id_col: str})
    meta[args.id_col] = meta[args.id_col].astype(str)

    if (not args.keep_bad_status) and ("status" in meta.columns):
        meta = meta[meta["status"].astype(str).str.lower() == "ok"].copy()
    if (not args.keep_unmatched_survival) and ("survival_matched" in meta.columns):
        sm = meta["survival_matched"]
        if sm.dtype == bool:
            meta = meta[meta["survival_matched"]].copy()
        else:
            meta = meta[sm.astype(str).str.lower().isin(["true", "1", "t", "yes"])].copy()

    meta[args.time_col] = pd.to_numeric(meta[args.time_col], errors="coerce")
    meta[args.event_col] = pd.to_numeric(meta[args.event_col], errors="coerce")
    meta = meta.dropna(subset=[args.time_col, args.event_col]).copy()
    meta = meta[meta[args.time_col] > 0].copy()
    meta[args.event_col] = meta[args.event_col].astype(int)

    global_cat_maps = build_global_categorical_maps(meta, list(args.clinical_cols))
    splits = load_precomputed_splits(args.cv_folds, splits_dir=args.splits_dir, splits_csv=args.splits_csv)
    expected_dhw = tuple(int(x) for x in args.img_size)

    manifest = {
        "meta_csv": str(Path(args.meta_csv).resolve()),
        "splits_dir": args.splits_dir,
        "splits_csv": args.splits_csv,
        "cv_folds": int(args.cv_folds),
        "ckpt_map": {str(k): str(Path(v).resolve()) for k, v in ckpt_map.items()},
        "weights": str(args.weights),
        "lora_alpha": float(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_scope": str(args.lora_scope),
        "shap_target": "survival_probability",
        "horizon_days": float(args.risk_horizon_days),
        "time_bin_width_days": float(args.time_bin_width_days),
        "n_perm": int(args.n_perm),
        "bg_size": int(args.bg_size),
        "clinical_cols": list(args.clinical_cols),
        "use_radiomics": bool(args.use_radiomics),
        "radiomics_root": str(args.radiomics_root),
        "radiomics_pca_total_components": int(args.radiomics_pca_total_components),
        "export_group_values": bool(args.export_group_values),
        "sanity_check": bool(args.sanity_check),
        "folds": {},
    }

    all_ids: List[str] = []
    all_folds: List[int] = []
    all_shap: List[np.ndarray] = []
    all_gvals: List[np.ndarray] = []
    all_pt_mask: List[int] = []
    all_ln_mask: List[int] = []
    all_pt_tok: List[int] = []
    all_ln_tok: List[int] = []

    group_names_ref: Optional[List[str]] = None
    seen_ids: Set[str] = set()

    for fold in range(int(args.cv_folds)):
        fold_dir = out_root / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        tr_df = select_df_by_ids(meta, splits[fold]["train"], args.id_col, args.strict_splits, f"fold{fold:02d}/train")
        va_df = select_df_by_ids(meta, splits[fold]["val"],   args.id_col, args.strict_splits, f"fold{fold:02d}/val")
        te_df = select_df_by_ids(meta, splits[fold]["test"],  args.id_col, args.strict_splits, f"fold{fold:02d}/test")
        te_row_map = {str(r[args.id_col]): r for _, r in te_df.iterrows()}

        ckpt_path = ckpt_map[fold]
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"[fold {fold:02d}] checkpoint not found: {ckpt_path}")

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        num_bins = int(ck.get("num_time_bins", -1))
        if num_bins <= 0:
            raise RuntimeError(f"[fold {fold:02d}] checkpoint missing num_time_bins: {ckpt_path}")

        sd_raw = strip_prefixes(extract_state_dict(ck))
        dims = infer_dims_from_sd(sd_raw)

        # ---- clinical encoder matched to ckpt dim (if clinical branch exists) ----
        clin_dim_ckpt = int(dims.get("clinical_dim_ckpt", 0))
        has_clin_keys = any(k.startswith("clin_proj.") for k in sd_raw.keys())
        if has_clin_keys and clin_dim_ckpt <= 0:
            raise RuntimeError(f"[fold {fold:02d}] ckpt has clin_proj.* but clinical_dim_ckpt not inferred.")

        clin_enc: Optional[ClinicalEncoderCompact] = None
        clin_groups: Dict[str, List[int]] = {}
        clin_dim = 0
        clin_plan: Dict[str, str] = {}

        if has_clin_keys and clin_dim_ckpt > 0:
            clin_enc = ClinicalEncoderCompact.fit(
                tr_df,
                list(args.clinical_cols),
                global_cat_maps=global_cat_maps,
                target_dim=int(clin_dim_ckpt),
            )
            clin_groups = clin_enc.feature_groups()
            clin_dim = int(clin_enc.output_dim)
            clin_plan = {s["col"]: s["kind"] for s in clin_enc.specs}

        # ---- radiomics encoder ----
        rad_enc = None
        rad_dim = 0
        rad_pc_slices: Dict[str, slice] = {}

        if args.use_radiomics:
            all_ids_fold = pd.concat([tr_df[args.id_col], va_df[args.id_col], te_df[args.id_col]]).astype(str).unique().tolist()
            train_ids = tr_df[args.id_col].astype(str).tolist()
            rad_enc = RadiomicsEncoder.fit(
                train_ids=train_ids,
                all_ids=all_ids_fold,
                radiomics_root=args.radiomics_root,
                total_pcs=int(args.radiomics_pca_total_components),
                seed=int(args.seed),
            )
            rad_dim = int(rad_enc.output_dim)
            rad_pc_slices = rad_enc.pc_slices
            print(f"[RAD][fold {fold:02d}] radiomics_dim={rad_dim} total_pc_dim={rad_enc.total_pc_dim}")

        tr_ds = PreprocessedMoEDataset(
            tr_df,
            id_col=args.id_col, time_col=args.time_col, event_col=args.event_col,
            ct_col=args.ct_col, mask_pt_col=args.mask_pt_col, mask_ln_col=args.mask_ln_col,
            clinical_encoder=clin_enc, radiomics_encoder=rad_enc,
            use_radiomics=args.use_radiomics, strict_files=True,
            expected_dhw=expected_dhw,
        )
        te_ds = PreprocessedMoEDataset(
            te_df,
            id_col=args.id_col, time_col=args.time_col, event_col=args.event_col,
            ct_col=args.ct_col, mask_pt_col=args.mask_pt_col, mask_ln_col=args.mask_ln_col,
            clinical_encoder=clin_enc, radiomics_encoder=rad_enc,
            use_radiomics=args.use_radiomics, strict_files=True,
            expected_dhw=expected_dhw,
        )

        tr_loader = DataLoader(tr_ds, batch_size=1, shuffle=False, num_workers=int(args.workers),
                               pin_memory=(device.type == "cuda"))
        te_loader = DataLoader(te_ds, batch_size=1, shuffle=False, num_workers=int(args.workers),
                               pin_memory=(device.type == "cuda"))

        fused_dim = int(dims.get("fused_dim", 512))
        img_token_dim = int(dims.get("img_token_dim", 0)) if int(dims.get("img_token_dim", 0)) > 0 else (
            int(args.img_token_dim) if int(args.img_token_dim) > 0 else fused_dim
        )

        img_proj_hidden_dim = int(dims.get("img_proj_hidden_dim", 2 * fused_dim))
        img_tok_ffn_hidden_dim = int(dims.get("img_tok_ffn_hidden_dim", 2 * fused_dim))
        img_post_hidden_dim = int(dims.get("img_post_hidden_dim", 2 * fused_dim))
        gate_hidden_dim = int(dims.get("gate_hidden_dim", fused_dim))
        rad_hidden_dim = int(dims.get("rad_hidden_dim", max(512, 2 * fused_dim)))

        rad_dim_ckpt = int(dims.get("radiomics_dim_ckpt", 0))
        if args.use_radiomics and rad_dim_ckpt > 0 and rad_dim > 0 and rad_dim != rad_dim_ckpt:
            print(f"[RAD][fold {fold:02d}][WARN] encoder rad_dim={rad_dim} but ckpt expects {rad_dim_ckpt} -> overriding")
            rad_dim = int(rad_dim_ckpt)
            if rad_enc is not None:
                rad_enc.output_dim = int(rad_dim_ckpt)

        token_mlp_hidden_dim = int(dims.get("token_mlp_hidden_dim", 0))
        if token_mlp_hidden_dim <= 0:
            token_mlp_hidden_dim = int(args.token_mlp_hidden_dim) if int(args.token_mlp_hidden_dim) > 0 else int(2 * img_token_dim)

        backbone_cfg = {
            "img_size": tuple(args.img_size),
            "feature_size": int(args.feature_size),
            "depths": tuple(args.depths),
            "num_heads": tuple(args.num_heads),
            "drop_rate": float(args.drop_rate),
            "attn_drop_rate": float(args.attn_drop_rate),
            "dropout_path_rate": float(args.dropout_path_rate),
            "normalize": True,
            "use_checkpoint": bool(args.use_checkpoint),

            "token_dim": int(img_token_dim),
            "token_mlp_dropout": float(args.token_mlp_dropout),
            "token_mlp_hidden_dim": int(token_mlp_hidden_dim),

            "attn_mask_bias": float(args.attn_mask_bias),
            "use_multiscale": bool(args.use_multiscale),
            "mask_interp": str(args.mask_interp),
            "min_roi_frac": float(args.min_roi_frac),
            "min_roi_voxels_deep": int(args.min_roi_voxels_deep),
            "token_dropout": float(args.token_dropout),

            "pt_shell_radius": int(args.pt_shell_radius),
            "ln_shell_radius": int(args.ln_shell_radius),
            "shell_body_from_ct": bool(args.shell_body_from_ct),
            "body_ct_thr": str(args.body_ct_thr),
            "body_ct_thr_hu": float(args.body_ct_thr_hu),
            "body_close_r": int(args.body_close_r),
            "body_max_frac": float(args.body_max_frac),

            "strict_swinvit_layout": bool(args.strict_swinvit_layout),
            "debug_swinvit_layout": bool(args.debug_swinvit_layout),
        }
        backbone_cfg.update(infer_backbone_runtime_overrides(ck, sd_raw))

        model = SwinUNETRTokenMoEDiscrete(
            num_time_bins=num_bins,
            time_bin_width_days=float(args.time_bin_width_days),
            fused_dim=fused_dim,
            backbone_cfg=backbone_cfg,
            clinical_dim=int(clin_dim),
            radiomics_dim=int(rad_dim),
            img_proj_hidden_dim=img_proj_hidden_dim,
            img_tok_ffn_hidden_dim=img_tok_ffn_hidden_dim,
            img_post_hidden_dim=img_post_hidden_dim,
            img_attn_heads=4,
            gate_hidden_dim=gate_hidden_dim,
            rad_hidden_dim=rad_hidden_dim,
            rad_proj_dropout_p=0.0,
        ).to(device)

        if ckpt_uses_split_patch(sd_raw):
            print(f"[PATCH][fold {fold:02d}] enabling split mask patch-embed before load")
            model.img_backbone.enable_mask_patch_embed_training(verbose=True)

        n_lora = 0
        if sd_has_lora(sd_raw):
            n_lora = inject_lora_from_state_dict(
                model,
                sd_raw,
                lora_alpha=float(args.lora_alpha),
                lora_dropout=float(args.lora_dropout),
                scope=str(args.lora_scope),
                verbose=True,
            )

        strict_load(model, sd_raw, fold=fold, ckpt_path=ckpt_path)

        which = str(args.weights).lower().strip()
        if which in ("ema", "swa"):
            blob = ck.get(which, None)
            mapping = _extract_tensor_mapping(blob)
            if mapping is None:
                print(f"[{which.upper()}][fold {fold:02d}] not found/recognized -> using LAST weights")
            else:
                mapping = strip_prefixes(mapping)
                apply_partial_state_dict(model, mapping, tag=f"{which.upper()}[fold {fold:02d}]")

        model.eval()
        head = HeadFromTokens(model).to(device).eval()

        groups = build_group_specs(
            clinical_cols=list(args.clinical_cols),
            clin_groups=clin_groups,
            clin_dim=int(clin_dim),
            rad_pc_slices=rad_pc_slices if rad_enc is not None else {},
            rad_total_pc_dim=int(rad_enc.total_pc_dim) if (rad_enc is not None) else 0,
            rad_group_names=list(rad_enc.group_names) if (rad_enc is not None) else None,
            rad_dim=int(rad_dim),
        )
        group_names = [g.name for g in groups]
        if group_names_ref is None:
            group_names_ref = group_names
        elif group_names != group_names_ref:
            raise RuntimeError(f"[fold {fold:02d}] group_names differ across folds.")

        name2idx = {nm: j for j, nm in enumerate(group_names)}
        idx_img_pt = name2idx["IMG:PT"]
        idx_img_ln = name2idx["IMG:LN"]

        # background baselines for clin/rad
        bg_clin, bg_rad = [], []
        bg_count = 0
        for x, _, _, clin, rad, _, _, _ in tr_loader:
            if bg_count >= int(args.bg_size):
                break
            x = x.to(device, non_blocking=True)
            clin = clin.to(device) if (clin is not None and clin.numel() > 0) else None
            rad  = rad.to(device)  if (rad  is not None and rad.numel()  > 0) else None

            if clin is not None and clin.numel() > 0:
                bg_clin.append(clin.detach().cpu().numpy())
            if rad is not None and rad.numel() > 0:
                bg_rad.append(rad.detach().cpu().numpy())
            bg_count += 1

        clin_base = None
        if clin_dim > 0:
            m = np.mean(np.concatenate(bg_clin, axis=0), axis=0, keepdims=True).astype(np.float32) if bg_clin else np.zeros((1, clin_dim), dtype=np.float32)
            clin_base = torch.from_numpy(m).to(device)
        rad_base = None
        if rad_dim > 0:
            m = np.mean(np.concatenate(bg_rad, axis=0), axis=0, keepdims=True).astype(np.float32) if bg_rad else np.zeros((1, rad_dim), dtype=np.float32)
            rad_base = torch.from_numpy(m).to(device)

        rng = np.random.RandomState(args.seed + 12345 + 97 * fold)

        fold_ids, fold_shap, fold_base_surv, fold_full_surv, fold_gvals = [], [], [], [], []
        fold_pt_mask, fold_ln_mask, fold_pt_tok, fold_ln_tok = [], [], [], []
        sanity_done = False

        for x, _, _, clin, rad, pid, pt_mask_present, ln_mask_present in te_loader:
            pid = str(pid[0])
            if pid in seen_ids:
                raise RuntimeError(f"[OOF] duplicate patient_id across folds: {pid}")
            seen_ids.add(pid)

            row_raw = te_row_map.get(pid, None)

            x = x.to(device, non_blocking=True)
            clin_x = clin.to(device) if (clin is not None and clin.numel() > 0) else None
            rad_x  = rad.to(device)  if (rad  is not None and rad.numel()  > 0) else None

            with torch.no_grad():
                with autocast_ctx():
                    tok_x, pres_x = model.img_backbone(x)
            tok_x = tok_x.float()
            pres_x = pres_x.to(torch.bool)

            # token presence flags (model-level)
            pt_token_present = bool(pres_x[0, 1].item() or pres_x[0, 2].item())
            ln_token_present = bool(pres_x[0, 3].item() or pres_x[0, 4].item())

            # BASELINE: keep patient GLOBAL token (0) fixed; remove lesion tokens (1..4)
            tok_base = tok_x.clone()
            pres_base = pres_x.clone()
            tok_base[:, 1:5, :].zero_()
            pres_base[:, 1:5] = False

            if bool(args.sanity_check) and (not sanity_done):
                with torch.no_grad():
                    logits_full = model(x, clin_x, rad_x, return_gate=False)
                    risk_full = float(model.hazards_to_risk(logits_full, horizon_days=float(args.risk_horizon_days)).item())
                    surv_full = 1.0 - risk_full
                    surv_head = float(head.forward_survival(tok_x, pres_x, clin_x, rad_x, horizon_days=float(args.risk_horizon_days)).item())
                diff = abs(surv_full - surv_head)
                print(f"[SANITY][fold {fold:02d}] |S(full)-S(head)| = {diff:.6e}")
                if diff > 1e-3:
                    raise RuntimeError(f"[SANITY] head-from-tokens mismatch too large: {diff}")
                sanity_done = True

            shap_vec, base_surv, full_surv = permutation_shapley_survival(
                head=head,
                tok_x=tok_x, pres_x=pres_x,
                clin_x=clin_x, rad_x=rad_x,
                tok_base=tok_base, pres_base=pres_base,
                clin_base=clin_base, rad_base=rad_base,
                groups=groups,
                horizon_days=float(args.risk_horizon_days),
                n_perm=int(args.n_perm),
                rng=rng,
            )

            # --- Required validity checks ---
            shap_pt = float(shap_vec[idx_img_pt])
            shap_ln = float(shap_vec[idx_img_ln])

            if not np.isfinite(shap_pt):
                raise RuntimeError(f"[INVALID] pid={pid} IMG:PT SHAP is not finite: {shap_pt}")

            # Only require LN SHAP to be finite when LN mask exists (your requirement).
            if bool(ln_mask_present) and (not np.isfinite(shap_ln)):
                raise RuntimeError(f"[INVALID] pid={pid} LN mask present but IMG:LN SHAP not finite: {shap_ln}")

            # Useful warning: LN mask exists but model LN token absent (thresholding / ROI rules)
            if bool(ln_mask_present) and (not ln_token_present):
                print(f"[WARN] pid={pid} LN mask present but LN tokens absent by model pres (LN SHAP likely ~0).")

            if args.export_group_values:
                gvals = []
                for g in groups:
                    if g.kind == "img":
                        idxs = g.img_token_indices or []
                        gvals.append(img_group_token_norm(tok_x, pres_x, idxs))
                    elif g.kind == "rad":
                        gvals.append(rad_group_l2(rad_x, (g.value_indices if (g.value_indices is not None) else g.vec_indices)))
                    elif g.kind == "clin":
                        col = g.name.split(":", 1)[1] if ":" in g.name else g.name
                        gvals.append(clinical_raw_value_global(row_raw, col, global_cat_maps))
                    else:
                        gvals.append(float("nan"))
                fold_gvals.append(np.asarray(gvals, dtype=np.float32))

            fold_ids.append(pid)
            fold_shap.append(shap_vec)
            fold_base_surv.append(float(base_surv))
            fold_full_surv.append(float(full_surv))

            fold_pt_mask.append(int(bool(pt_mask_present)))
            fold_ln_mask.append(int(bool(ln_mask_present)))
            fold_pt_tok.append(int(pt_token_present))
            fold_ln_tok.append(int(ln_token_present))

        S_fold = np.stack(fold_shap, axis=0).astype(np.float32)
        base_s = np.asarray(fold_base_surv, dtype=np.float32)
        full_s = np.asarray(fold_full_surv, dtype=np.float32)
        V_fold = np.stack(fold_gvals, axis=0).astype(np.float32) if (args.export_group_values and fold_gvals) else None

        pt_mask_arr = np.asarray(fold_pt_mask, dtype=np.int8)
        ln_mask_arr = np.asarray(fold_ln_mask, dtype=np.int8)
        pt_tok_arr  = np.asarray(fold_pt_tok, dtype=np.int8)
        ln_tok_arr  = np.asarray(fold_ln_tok, dtype=np.int8)

        fold_npz = fold_dir / f"shap_oof_grouped_fold{fold:02d}.npz"
        np_save_kwargs = dict(
            patient_id=np.asarray(fold_ids, dtype=object),
            fold=np.full((len(fold_ids),), fold, dtype=np.int32),
            group_names=np.asarray(group_names, dtype=object),
            shap_values=S_fold,
            base_survival=base_s,
            full_survival=full_s,
            pt_mask_present=pt_mask_arr,
            ln_mask_present=ln_mask_arr,
            pt_token_present=pt_tok_arr,
            ln_token_present=ln_tok_arr,
            ckpt_path=str(Path(ckpt_path).resolve()),
            num_time_bins=np.int32(num_bins),
            horizon_days=np.float32(args.risk_horizon_days),
            time_bin_width_days=np.float32(args.time_bin_width_days),
            n_perm=np.int32(args.n_perm),
            weights=str(args.weights),
            lora_alpha=np.float32(args.lora_alpha),
            lora_injected=np.int32(n_lora),
            export_group_values=np.bool_(args.export_group_values),
        )
        if V_fold is not None:
            np_save_kwargs["group_values"] = V_fold
        np.savez_compressed(fold_npz, **np_save_kwargs)
        print(f"[fold {fold:02d}] saved {len(fold_ids)} OOF rows -> {fold_npz}")

        manifest["folds"][str(fold)] = {
            "n_test": int(len(fold_ids)),
            "fold_npz": str(fold_npz),
            "ckpt_path": str(Path(ckpt_path).resolve()),
            "clinical_dim_ckpt": int(clin_dim_ckpt),
            "clinical_dim_used": int(clin_dim),
            "clinical_plan": clin_plan,
            "radiomics_dim_used": int(rad_dim),
            "radiomics_dim_ckpt": int(rad_dim_ckpt),
            "inferred_dims": dims,
            "lora_injected": int(n_lora),
        }

        all_ids.extend(fold_ids)
        all_folds.extend([fold] * len(fold_ids))
        all_shap.append(S_fold)
        if V_fold is not None:
            all_gvals.append(V_fold)

        all_pt_mask.extend(pt_mask_arr.tolist())
        all_ln_mask.extend(ln_mask_arr.tolist())
        all_pt_tok.extend(pt_tok_arr.tolist())
        all_ln_tok.extend(ln_tok_arr.tolist())

    # ---- pooled outputs ----
    S_all = np.concatenate(all_shap, axis=0).astype(np.float32)
    group_names = group_names_ref or []

    out_npz = out_root / "oof_shap_grouped_all.npz"
    pooled_save = dict(
        patient_id=np.asarray(all_ids, dtype=object),
        fold=np.asarray(all_folds, dtype=np.int32),
        group_names=np.asarray(group_names, dtype=object),
        shap_values=S_all,
        mean_abs_shap=np.mean(np.abs(S_all), axis=0).astype(np.float32),
        shap_target="survival_probability",
        horizon_days=np.float32(args.risk_horizon_days),
        time_bin_width_days=np.float32(args.time_bin_width_days),
        n_perm=np.int32(args.n_perm),
        weights=str(args.weights),
        lora_alpha=np.float32(args.lora_alpha),
        export_group_values=np.bool_(args.export_group_values),
        pt_mask_present=np.asarray(all_pt_mask, dtype=np.int8),
        ln_mask_present=np.asarray(all_ln_mask, dtype=np.int8),
        pt_token_present=np.asarray(all_pt_tok, dtype=np.int8),
        ln_token_present=np.asarray(all_ln_tok, dtype=np.int8),
    )
    if args.export_group_values and all_gvals:
        V_all = np.concatenate(all_gvals, axis=0).astype(np.float32)
        pooled_save["group_values"] = V_all
    np.savez_compressed(out_npz, **pooled_save)

    wide = pd.DataFrame({"patient_id": all_ids, "fold": all_folds})
    for j, nm in enumerate(group_names):
        wide[nm] = S_all[:, j]
    wide["pt_mask_present"] = all_pt_mask
    wide["ln_mask_present"] = all_ln_mask
    wide["pt_token_present"] = all_pt_tok
    wide["ln_token_present"] = all_ln_tok

    wide_csv = out_root / "oof_shap_grouped_wide.csv"
    wide.to_csv(wide_csv, index=False)

    mean_abs = np.mean(np.abs(S_all), axis=0).astype(np.float32)
    imp = pd.DataFrame({"feature": group_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    imp_csv = out_root / "oof_group_importance.csv"
    imp.to_csv(imp_csv, index=False)

    gv_csv = None
    if args.export_group_values and all_gvals:
        V_all = np.concatenate(all_gvals, axis=0).astype(np.float32)
        wide_v = pd.DataFrame({"patient_id": all_ids, "fold": all_folds})
        for j, nm in enumerate(group_names):
            wide_v[nm] = V_all[:, j]
        wide_v["pt_mask_present"] = all_pt_mask
        wide_v["ln_mask_present"] = all_ln_mask
        wide_v["pt_token_present"] = all_pt_tok
        wide_v["ln_token_present"] = all_ln_tok
        gv_csv = out_root / "oof_group_values_wide.csv"
        wide_v.to_csv(gv_csv, index=False)
        print(f"[OK] wrote group values CSV: {gv_csv}")

    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] wrote:\n  {out_npz}\n  {wide_csv}\n  {imp_csv}\n  {out_root/'manifest.json'}")
    if gv_csv is not None:
        print(f"  {gv_csv}")
    print("[Top 10 mean|SHAP| for S(t)]")
    print(imp.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
