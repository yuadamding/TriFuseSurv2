#!/usr/bin/env python3
"""
trifusesurv.multimodal_survival.explain_tokens

CV SHAP for moe_discrete_swinunetr.py trained models, with fold-specific checkpoints.

Explains scalar risk@horizon using token-level SHAP:
  - 5 image tokens: GLOBAL, PT_intra, PT_peri, LN_intra, LN_peri
  - clinical features
  - radiomics features (fold-specific PCA)

CRITICAL:
  We do NOT pass `pres` into SHAP. Presence is used via boolean masking (non-differentiable),
  which breaks GradientExplainer ("tensor not used in graph"). Instead:
    - mask tokens: tok := tok * pres[...,None]
    - derive presence internally from tok==0 (treated as constant)

Robustness:
  - handles SHAP returning extra singleton output dims (e.g., (N,1,D) or (1,N,D))
  - always squeezes SHAP arrays to expected shapes before plotting/exporting

CUDA_VISIBLE_DEVICES=1 python -m trifusesurv.multimodal_survival.explain_tokens --train_script trifusesurv.multimodal_survival.train --ckpt_fold 0:runs/moe_discrete_swinunetr/cv4_best_fold00_test2/fold_00/last.pt --ckpt_fold 1:runs/moe_discrete_swinunetr/cv4_best_fold01_test2/fold_01/last.pt --ckpt_fold 2:runs/moe_discrete_swinunetr/cv4_best_fold02_test2/fold_02/last.pt --ckpt_fold 3:runs/moe_discrete_swinunetr/cv4_best_fold03_test2/fold_03/last.pt --meta_csv OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv --splits_dir runs/opscc_splits_os_seed1 --cv_folds 4 --strict_splits --endpoint OS --ct_col ct_out_path --mask_pt_col mask_primary_out_path --mask_ln_col mask_nodal_out_path --img_size 128 256 256 --time_bin_width_days 180 --risk_horizon_days 1095 --use_radiomics --radiomics_root cohort_radiomics_patient_wide.csv --radiomics_pca_total_components 50 --device cuda:0 --amp --weights ema --n_background 16 --n_explain 200 --explainer gradient --out_dir runs/shap_cv --strict_load --pool_clinical
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import argparse
import importlib.util
import contextlib
import json
import random
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


DEFAULT_TRAIN_MODULE = "trifusesurv.multimodal_survival.train"


def resolve_path(p: str) -> Path:
    pp = Path(p).expanduser()
    if pp.is_absolute():
        return pp
    if pp.exists():
        return pp
    return Path.cwd() / pp


def load_train_module(train_script_path: str):
    target = str(train_script_path).strip() or DEFAULT_TRAIN_MODULE
    if target.endswith(".py") or "/" in target or "\\" in target:
        p = resolve_path(target)
        if not p.is_file():
            raise FileNotFoundError(f"--train_script not found: {train_script_path} (resolved: {p})")

        spec = importlib.util.spec_from_file_location("moe_train_mod", str(p))
        mod = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore
        return mod

    try:
        return importlib.import_module(target)
    except ModuleNotFoundError:
        p = resolve_path(target)
        if not p.is_file():
            raise FileNotFoundError(f"--train_script not found as module or file: {train_script_path} (resolved: {p})")

        spec = importlib.util.spec_from_file_location("moe_train_mod", str(p))
        mod = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore
        return mod


# =============================================================================
# CLI helpers for fold->ckpt mapping
# =============================================================================
def _flatten_list_of_lists(x: List[List[str]]) -> List[str]:
    out: List[str] = []
    for g in x:
        out.extend(list(g))
    return out


def parse_ckpt_fold_items(items: Sequence[str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for s in items:
        s = str(s).strip()
        if not s:
            continue
        if ":" in s:
            k, v = s.split(":", 1)
        elif "=" in s:
            k, v = s.split("=", 1)
        else:
            raise ValueError(f"--ckpt_fold entry must be like 0:/path/to/last.pt (or 0=/path), got: {s}")
        out[int(k.strip())] = v.strip()
    return out


def load_ckpt_map_json(path: str) -> Dict[int, str]:
    p = resolve_path(path)
    if not p.is_file():
        raise FileNotFoundError(f"--ckpt_map_json not found: {path} (resolved: {p})")
    with open(p, "r") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("--ckpt_map_json must contain a JSON object mapping fold->path.")
    return {int(k): str(v) for k, v in obj.items()}


# =============================================================================
# Feature name builders
# =============================================================================
def clinical_feature_names(clin_enc) -> List[str]:
    names: List[str] = []
    for col in getattr(clin_enc, "numeric_cols", []):
        names.append(f"{col}_z")
        names.append(f"{col}_missing")

    cat_cols = getattr(clin_enc, "cat_cols", [])
    cat_maps = getattr(clin_enc, "cat_maps", {})
    cat_dims = getattr(clin_enc, "cat_dims", {})
    for col in cat_cols:
        mapping = cat_maps.get(col, {})
        inv = sorted(mapping.items(), key=lambda kv: kv[1])  # (cat, idx)
        for cat, _ in inv:
            names.append(f"{col}={cat}")
        names.append(f"{col}=UNK")
        _ = int(cat_dims.get(col, len(inv) + 1))

    out_dim = int(getattr(clin_enc, "output_dim", len(names)))
    if len(names) != out_dim:
        names = [f"clin_{i:03d}" for i in range(out_dim)]
    return names


def radiomics_feature_names(rad_dim: int) -> List[str]:
    rad_dim = int(rad_dim)
    if rad_dim <= 0:
        return []
    tail = 7  # presence(4) + counts(3)
    pc_dim = max(0, rad_dim - tail)
    names = [f"rad_pc_{i:03d}" for i in range(pc_dim)]
    pres_names = ["presence_PT_intra", "presence_PT_peri", "presence_LN_intra", "presence_LN_peri"]
    cnt_names = ["count_0", "count_1", "count_2"]
    if rad_dim >= tail:
        names += pres_names + cnt_names
    else:
        names = [f"rad_{i:03d}" for i in range(rad_dim)]
    if len(names) != rad_dim:
        names = [f"rad_{i:03d}" for i in range(rad_dim)]
    return names


# =============================================================================
# SHAP shape normalization
# =============================================================================
def _squeeze_to(arr: np.ndarray, target_ndim: int, name: str) -> np.ndarray:
    """
    SHAP sometimes returns arrays with singleton output dims, e.g.:
      token: (N,1,5,D) or (1,N,5,D) or (N,5,D,1)
      clin : (N,1,D)   or (1,N,D)   or (N,D,1)

    This function aggressively removes singleton dims until target_ndim is reached.
    Raises if it can't.
    """
    a = np.asarray(arr)

    # repeatedly remove singleton dims (but keep order stable)
    # 1) common patterns
    if a.ndim == target_ndim + 1 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == target_ndim + 1 and a.shape[1] == 1:
        a = a[:, 0, ...]
    while a.ndim > target_ndim and a.shape[-1] == 1:
        a = a[..., 0]

    # 2) final squeeze of any remaining singleton dims
    if a.ndim > target_ndim:
        a = np.squeeze(a)

    if a.ndim != target_ndim:
        raise RuntimeError(f"[SHAP][{name}] Unexpected ndim after squeeze: got {a.ndim}, expected {target_ndim}. shape={a.shape}")

    return a


# =============================================================================
# Token-head wrapper (NO pres input)
# =============================================================================
class TokenHead(nn.Module):
    """
    Takes tok_masked: (B,5,Dtok) where absent tokens are already zeroed.
    Presence is derived internally from tok_masked==0 (detached mask).
    """
    def __init__(self, trained_model):
        super().__init__()
        mm = trained_model

        self.num_time_bins = int(mm.num_time_bins)
        self.time_bin_width_days = float(mm.time_bin_width_days)
        self.fused_dim = int(mm.fused_dim)
        self.num_experts = int(mm.num_experts)

        self.fuse_projs = mm.fuse_projs
        self.img_attn = mm.img_attn
        self.gate_mlp = mm.gate_mlp

        self.use_clin = bool(getattr(mm, "use_clin", False))
        self.use_rad = bool(getattr(mm, "use_rad", False))
        self.clin_proj = mm.clin_proj
        self.rad_proj = mm.rad_proj
        self.surv_head = mm.surv_head

    def hazards_to_risk(self, hazards_logits: torch.Tensor, horizon_days: float) -> torch.Tensor:
        bw = float(self.time_bin_width_days)
        t = float(horizon_days)
        hazards = torch.sigmoid(hazards_logits.float()).clamp(1e-7, 1.0 - 1e-7)
        K = hazards.shape[1]
        max_covered = K * bw
        if t >= max_covered:
            logS = torch.log1p(-hazards).sum(dim=1)
            return (1.0 - torch.exp(logS)).clamp(0.0, 1.0)

        k = int(np.floor(t / bw))
        k = max(0, min(k, K - 1))
        frac = (t - (k * bw)) / bw

        log1m = torch.log1p(-hazards)
        cum = torch.cumsum(log1m, dim=1)
        if k == 0:
            logS_t = log1m[:, 0] * float(t / bw)
        else:
            logS_t = cum[:, k - 1] + log1m[:, k] * float(frac)
        return (1.0 - torch.exp(logS_t)).clamp(0.0, 1.0)

    def forward(
        self,
        tok_masked: torch.Tensor,  # (B,5,Dtok)
        clinical: Optional[torch.Tensor] = None,
        radiomics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = tok_masked.size(0)

        # derive presence from token content (detached => constant mask)
        pres_bool = (tok_masked.detach().abs().sum(dim=2) > 0)  # (B,5)
        if pres_bool.shape[1] >= 1:
            pres_bool[:, 0] = True  # GLOBAL always allowed

        fused_list = [self.fuse_projs[i](tok_masked[:, i, :]) for i in range(self.num_experts)]
        stacked = torch.stack(fused_list, dim=1)  # (B,5,D)
        stacked = stacked * pres_bool.unsqueeze(-1).to(stacked.dtype)

        attn_out, _ = self.img_attn(
            stacked, stacked, stacked,
            key_padding_mask=(~pres_bool),
        )
        attn_out = attn_out * pres_bool.unsqueeze(-1).to(attn_out.dtype)

        gate_logits = self.gate_mlp(attn_out.reshape(B, -1).float()).float()
        neg_inf = torch.finfo(gate_logits.dtype).min
        gate_logits = gate_logits.masked_fill(~pres_bool, neg_inf)
        all_abs = ~pres_bool.any(dim=1)
        if all_abs.any():
            gate_logits = gate_logits.clone()
            gate_logits[all_abs, 0] = 0.0
        gate = torch.softmax(gate_logits, dim=1).to(attn_out.dtype)

        fused_img = (attn_out * gate.unsqueeze(-1)).sum(dim=1)
        chunks = [fused_img]

        if self.use_clin and clinical is not None and clinical.numel() > 0:
            chunks.append(self.clin_proj(clinical.to(fused_img.device)))
        if self.use_rad and radiomics is not None and radiomics.numel() > 0:
            chunks.append(self.rad_proj(radiomics.to(fused_img.device)))

        h = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
        logits = self.surv_head(h)
        return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)


class RiskWrapper(nn.Module):
    def __init__(self, token_head: TokenHead, risk_horizon_days: float):
        super().__init__()
        self.head = token_head
        self.horizon = float(risk_horizon_days)

    def forward(self, tok_masked, *rest):
        logits = self.head(tok_masked, *rest)
        risk = self.head.hazards_to_risk(logits, horizon_days=self.horizon)
        return risk.unsqueeze(1)


# =============================================================================
# Checkpoint utilities
# =============================================================================
def extract_model_state(ck: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ck, dict) and "model_state" in ck and isinstance(ck["model_state"], dict):
        sd = ck["model_state"]
    elif isinstance(ck, dict) and all(isinstance(v, torch.Tensor) for v in ck.values()):
        sd = ck
    else:
        raise RuntimeError("Checkpoint format not recognized (missing model_state and not a raw state_dict).")

    if any(str(k).startswith("module.") for k in sd.keys()):
        sd = {str(k)[7:]: v for k, v in sd.items()}
    return sd


def ckpt_uses_split_patch_embed(sd: Dict[str, torch.Tensor]) -> bool:
    for k in sd.keys():
        ks = str(k)
        if "patch_embed.proj.conv_ct." in ks or "patch_embed.proj.conv_mask." in ks:
            return True
        if "patch_embed.proj.conv_mask_pt." in ks or "patch_embed.proj.conv_mask_ln." in ks:
            return True
        if "img_backbone._pt_patch_split." in ks or "img_backbone._ln_patch_split." in ks:
            return True
        if "img_backbone._patch_split." in ks:
            return True
    return False


def sd_has_lora(sd: Dict[str, torch.Tensor]) -> bool:
    for k in sd.keys():
        ks = str(k)
        if ".lora_A." in ks or ".lora_B." in ks:
            return True
    return False


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
    return out


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(v)


def infer_backbone_runtime_overrides(ck: Any) -> Dict[str, Any]:
    if not isinstance(ck, dict):
        return {}
    ck_args = ck.get("args", {})
    if not isinstance(ck_args, dict):
        return {}

    out: Dict[str, Any] = {}
    if "force_presence_from_raw_masks" in ck_args:
        out["force_presence_from_raw_masks"] = _coerce_bool(ck_args["force_presence_from_raw_masks"])
    if "raw_mask_threshold" in ck_args:
        out["raw_mask_threshold"] = float(ck_args["raw_mask_threshold"])
    if "fallback_peri_to_intra" in ck_args:
        out["fallback_peri_to_intra"] = _coerce_bool(ck_args["fallback_peri_to_intra"])
    return out


def infer_image_encoder_mode(ck: Any, sd: Dict[str, torch.Tensor]) -> str:
    if isinstance(ck, dict):
        ck_args = ck.get("args", {})
        if isinstance(ck_args, dict):
            mode = str(ck_args.get("image_encoder_mode", "")).strip().lower()
            if mode in ("shared_mask", "shared_roi"):
                return "shared_mask"
            if mode in ("dual_backbone", "legacy_dual"):
                return "dual_backbone"
    for k in sd.keys():
        ks = str(k)
        if ".backbone_shared." in ks or "img_backbone._patch_split." in ks:
            return "shared_mask"
    return "dual_backbone"


# =============================================================================
# Extraction helpers
# =============================================================================
@torch.no_grad()
def extract_tok_pres_clin_rad(full_model, loader, device, autocast_ctx, max_n: int):
    full_model.eval()
    mm = full_model.module if isinstance(full_model, nn.DataParallel) else full_model

    tok_list, pres_list = [], []
    clin_list, rad_list = [], []
    ids: List[str] = []

    for x, _, _, clin, rad, pid in loader:
        if max_n > 0 and len(ids) >= max_n:
            break

        x = x.to(device, non_blocking=True)
        with autocast_ctx():
            tok, pres = mm.img_backbone(x)

        tok_list.append(tok.float().cpu().numpy())
        pres_list.append(pres.float().cpu().numpy())

        if clin is not None and clin.numel() > 0:
            clin_list.append(clin.float().cpu().numpy())
        if rad is not None and rad.numel() > 0:
            rad_list.append(rad.float().cpu().numpy())

        ids.extend([str(p) for p in pid])

    tok_np = np.nan_to_num(np.concatenate(tok_list, axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    pres_np = np.nan_to_num(np.concatenate(pres_list, axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    clin_np = np.nan_to_num(np.concatenate(clin_list, axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32) if clin_list else None
    rad_np  = np.nan_to_num(np.concatenate(rad_list, axis=0),  nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32) if rad_list  else None

    return tok_np, pres_np, clin_np, rad_np, ids


def _torch(x: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


def _sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, replace=False, random_state=seed).reset_index(drop=True)


# =============================================================================
# Fold SHAP
# =============================================================================
def compute_fold_shap(
    train_mod,
    *,
    fold: int,
    args,
    meta: pd.DataFrame,
    split: Dict[str, List[str]],
    ckpt_path: Path,
    out_dir: Path,
    device,
    autocast_ctx,
) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import shap
    except Exception as e:
        raise RuntimeError("SHAP is not installed. Install: pip install shap") from e

    out_dir.mkdir(parents=True, exist_ok=True)

    tr_df = train_mod.select_df_by_ids(meta, split["train"], args.id_col, args.strict_splits, f"fold{fold:02d}/train")
    va_df = train_mod.select_df_by_ids(meta, split["val"],   args.id_col, args.strict_splits, f"fold{fold:02d}/val")
    te_df = train_mod.select_df_by_ids(meta, split["test"],  args.id_col, args.strict_splits, f"fold{fold:02d}/test")

    # encoders
    clinical_cols = args.clinical_cols or list(getattr(train_mod, "DEFAULT_CLINICAL_COLS", []))
    clin_enc = train_mod.ClinicalEncoder.fit(tr_df, clinical_cols)
    clin_dim = int(clin_enc.output_dim)

    rad_enc = None
    rad_dim = 0
    if args.use_radiomics:
        all_ids = pd.concat([tr_df[args.id_col], va_df[args.id_col], te_df[args.id_col]]).astype(str).unique().tolist()
        train_ids = tr_df[args.id_col].astype(str).tolist()
        rad_enc = train_mod.RadiomicsEncoder.fit(
            train_ids, all_ids, args.radiomics_root, int(args.radiomics_pca_total_components), int(args.seed)
        )
        rad_dim = int(rad_enc.output_dim)

    # bg/explain
    bg_df = _sample_df(tr_df, int(args.n_background), seed=int(args.seed + 1000 + fold))
    ex_df = _sample_df(te_df, int(args.n_explain), seed=int(args.seed + 2000 + fold))

    expected_dhw = tuple(int(x) for x in args.img_size)

    # time/event cols
    time_col, event_col = args.time_col, args.event_col
    if time_col == "" or event_col == "":
        tcol, ecol = train_mod.ENDPOINT_MAP[args.endpoint]
        time_col = tcol if time_col == "" else time_col
        event_col = ecol if event_col == "" else event_col

    bg_ds = train_mod.PreprocessedMoEDataset(
        bg_df,
        id_col=args.id_col, time_col=time_col, event_col=event_col,
        ct_col=args.ct_col, mask_pt_col=args.mask_pt_col, mask_ln_col=args.mask_ln_col,
        mode="eval",
        clinical_encoder=clin_enc, radiomics_encoder=rad_enc,
        use_radiomics=bool(args.use_radiomics),
        strict_files=bool(args.strict_files),
        expected_dhw=expected_dhw,
    )
    ex_ds = train_mod.PreprocessedMoEDataset(
        ex_df,
        id_col=args.id_col, time_col=time_col, event_col=event_col,
        ct_col=args.ct_col, mask_pt_col=args.mask_pt_col, mask_ln_col=args.mask_ln_col,
        mode="eval",
        clinical_encoder=clin_enc, radiomics_encoder=rad_enc,
        use_radiomics=bool(args.use_radiomics),
        strict_files=bool(args.strict_files),
        expected_dhw=expected_dhw,
    )

    bg_loader = torch.utils.data.DataLoader(bg_ds, batch_size=1, shuffle=False, num_workers=int(args.workers),
                                           pin_memory=(device.type == "cuda"))
    ex_loader = torch.utils.data.DataLoader(ex_ds, batch_size=1, shuffle=False, num_workers=int(args.workers),
                                           pin_memory=(device.type == "cuda"))

    # load checkpoint
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = extract_model_state(ck)
    if sd_has_lora(sd):
        raise RuntimeError(
            "LoRA checkpoint detected in shap_joint_model_tokens.py. "
            "This token-level SHAP entry point reconstructs the standard survival model only; "
            "use the grouped SHAP export scripts for LoRA checkpoints."
        )

    num_time_bins = int(ck.get("num_time_bins", -1)) if isinstance(ck, dict) else -1
    if num_time_bins <= 0:
        for k in sd.keys():
            if k.endswith("surv_head.3.bias"):
                num_time_bins = int(sd[k].numel())
                break
    if num_time_bins <= 0:
        raise RuntimeError(f"[fold {fold:02d}] Could not determine num_time_bins from checkpoint.")

    dims = infer_dims_from_sd(sd)
    fused_dim = int(dims.get("fused_dim", args.fused_dim))
    img_token_dim = int(dims.get("img_token_dim", fused_dim))
    token_mlp_hidden_dim = int(dims.get("token_mlp_hidden_dim", 2 * img_token_dim))
    img_proj_hidden_dim = int(dims.get("img_proj_hidden_dim", 2 * fused_dim))
    img_tok_ffn_hidden_dim = int(dims.get("img_tok_ffn_hidden_dim", 2 * fused_dim))
    img_post_hidden_dim = int(dims.get("img_post_hidden_dim", 2 * fused_dim))
    gate_hidden_dim = int(dims.get("gate_hidden_dim", fused_dim))
    rad_hidden_dim = int(dims.get("rad_hidden_dim", max(512, 2 * fused_dim)))

    # build model (must match training)
    image_encoder_mode = infer_image_encoder_mode(ck, sd)
    print(f"[fold {fold:02d}] image_encoder_mode={image_encoder_mode}")
    backbone_cfg = dict(
        img_size=tuple(args.img_size),
        image_encoder_mode=image_encoder_mode,
        feature_size=int(args.feature_size),
        depths=tuple(args.depths),
        num_heads=tuple(args.num_heads),
        drop_rate=float(args.drop_rate),
        attn_drop_rate=float(args.attn_drop_rate),
        dropout_path_rate=float(args.dropout_path_rate),
        normalize=True,
        use_checkpoint=bool(args.use_checkpoint),
        token_dim=img_token_dim,
        token_mlp_dropout=float(args.token_mlp_dropout),
        token_mlp_hidden_dim=token_mlp_hidden_dim,
        attn_mask_bias=float(args.attn_mask_bias),
        use_multiscale=bool(args.use_multiscale),
        mask_interp=str(args.mask_interp),
        min_roi_frac=float(args.min_roi_frac),
        min_roi_voxels_deep=int(args.min_roi_voxels_deep),
        token_dropout=float(args.token_dropout),
        pt_shell_radius=int(args.pt_shell_radius),
        ln_shell_radius=int(args.ln_shell_radius),
        shell_body_from_ct=bool(args.shell_body_from_ct),
        body_ct_thr=str(args.body_ct_thr),
        body_ct_thr_hu=float(args.body_ct_thr_hu),
        body_close_r=int(args.body_close_r),
        body_max_frac=float(args.body_max_frac),
        strict_swinvit_layout=bool(args.strict_swinvit_layout),
        debug_swinvit_layout=bool(args.debug_swinvit_layout),
    )
    backbone_cfg.update(infer_backbone_runtime_overrides(ck))

    model = train_mod.SwinUNETRTokenMoEDiscrete(
        num_time_bins=num_time_bins,
        time_bin_width_days=float(args.time_bin_width_days),
        fused_dim=fused_dim,
        backbone_cfg=backbone_cfg,
        clinical_dim=int(clin_dim),
        radiomics_dim=int(rad_dim),
        expert_dropout_p=float(args.expert_dropout_p),
        proj_dropout_p=float(args.proj_dropout_p),
        attn_dropout_p=float(args.attn_dropout_p),
        gate_dropout_p=float(args.gate_dropout_p),
        surv_dropout_p=float(args.surv_dropout_p),
        clinical_noise_std=float(args.clinical_noise_std),
        radiomics_noise_std=float(args.radiomics_noise_std),
        modality_dropout_clin_p=float(args.modality_dropout_clin_p),
        modality_dropout_rad_p=float(args.modality_dropout_rad_p),
        img_proj_hidden_dim=img_proj_hidden_dim,
        img_tok_ffn_hidden_dim=img_tok_ffn_hidden_dim,
        img_post_hidden_dim=img_post_hidden_dim,
        gate_hidden_dim=gate_hidden_dim,
        rad_hidden_dim=rad_hidden_dim,
    ).to(device)

    # patch-embed structure
    if ckpt_uses_split_patch_embed(sd):
        model.enable_mask_patch_embed_training(verbose=False)

    # load weights
    if args.strict_load:
        model.load_state_dict(sd, strict=True)
    else:
        model.load_state_dict(sd, strict=False)

    model.eval()
    mm = model.module if isinstance(model, nn.DataParallel) else model

    # weights mode
    weight_ctx = contextlib.nullcontext()
    if args.weights == "ema" and isinstance(ck, dict) and ck.get("ema") is not None:
        ema = train_mod.H.EMAWeights(mm, decay=0.0, track_trainable_only=True)
        ema.load_state_dict(ck["ema"], model=mm)
        weight_ctx = ema.apply_to(mm)
        print(f"[fold {fold:02d}] using EMA weights")
    elif args.weights == "swa" and isinstance(ck, dict) and ck.get("swa") is not None:
        swa = train_mod.H.SWAWeights(mm, track_trainable_only=True)
        swa.load_state_dict(ck["swa"], model=mm)
        weight_ctx = swa.apply_to(mm)
        print(f"[fold {fold:02d}] using SWA weights")
    else:
        print(f"[fold {fold:02d}] using LAST weights")

    with weight_ctx:
        model.eval()

        # extract tokens + presence
        tok_bg, pres_bg, clin_bg, rad_bg, _ = extract_tok_pres_clin_rad(model, bg_loader, device, autocast_ctx, max_n=int(args.n_background))
        tok_ex, pres_ex, clin_ex, rad_ex, ids_ex = extract_tok_pres_clin_rad(model, ex_loader, device, autocast_ctx, max_n=int(args.n_explain))

        tok_bg_masked = tok_bg * pres_bg[..., None]
        tok_ex_masked = tok_ex * pres_ex[..., None]

        token_head = TokenHead(mm).to(device).eval()
        risk_model = RiskWrapper(token_head, risk_horizon_days=float(args.risk_horizon_days)).to(device).eval()

        use_clin = (clin_dim > 0) and (clin_bg is not None) and (clin_ex is not None)
        use_rad  = (rad_dim  > 0) and (rad_bg  is not None) and (rad_ex  is not None)

        # SHAP inputs (NO pres)
        Xbg: List[torch.Tensor] = [_torch(tok_bg_masked, device)]
        Xex: List[torch.Tensor] = [_torch(tok_ex_masked, device)]
        if use_clin:
            Xbg.append(_torch(clin_bg, device))
            Xex.append(_torch(clin_ex, device))
        if use_rad:
            Xbg.append(_torch(rad_bg, device))
            Xex.append(_torch(rad_ex, device))

        # SHAP
        with torch.amp.autocast("cuda", enabled=False):
            if args.explainer == "gradient":
                explainer = shap.GradientExplainer(risk_model, Xbg)
            else:
                explainer = shap.DeepExplainer(risk_model, Xbg)
            shap_vals = explainer.shap_values(Xex)

        # normalize shap output: list per input
        if isinstance(shap_vals, list) and len(shap_vals) > 0 and isinstance(shap_vals[0], list):
            shap_in = shap_vals[0]  # first output
        elif isinstance(shap_vals, list):
            shap_in = shap_vals
        else:
            shap_in = [shap_vals]

        # inputs are: tok_masked, (clin?), (rad?)
        idx = 0
        shap_tok = _squeeze_to(np.asarray(shap_in[idx]), 3, "tok"); idx += 1
        shap_clin = _squeeze_to(np.asarray(shap_in[idx]), 2, "clin") if use_clin else None
        idx += 1 if use_clin else 0
        shap_rad = _squeeze_to(np.asarray(shap_in[idx]), 2, "rad") if use_rad else None

        # sanity checks
        if shap_tok.shape != tok_ex_masked.shape:
            raise RuntimeError(f"[fold {fold:02d}] SHAP tok shape mismatch: shap={shap_tok.shape} input={tok_ex_masked.shape}")

        if use_clin and shap_clin is not None and clin_ex is not None and shap_clin.shape != clin_ex.shape:
            raise RuntimeError(f"[fold {fold:02d}] SHAP clin shape mismatch: shap={shap_clin.shape} input={clin_ex.shape}")
        if use_rad and shap_rad is not None and rad_ex is not None and shap_rad.shape != rad_ex.shape:
            raise RuntimeError(f"[fold {fold:02d}] SHAP rad shape mismatch: shap={shap_rad.shape} input={rad_ex.shape}")

        # optional raw dump
        if args.save_npz:
            np.savez_compressed(
                out_dir / "shap_values.npz",
                ids=np.array(ids_ex, dtype=object),
                shap_tok=shap_tok.astype(np.float32),
                shap_clin=(shap_clin.astype(np.float32) if shap_clin is not None else np.zeros((len(ids_ex), 0), dtype=np.float32)),
                shap_rad=(shap_rad.astype(np.float32) if shap_rad is not None else np.zeros((len(ids_ex), 0), dtype=np.float32)),
                tok_input=tok_ex_masked.astype(np.float32),
                clin_input=(clin_ex.astype(np.float32) if clin_ex is not None else np.zeros((len(ids_ex), 0), dtype=np.float32)),
                rad_input=(rad_ex.astype(np.float32) if rad_ex is not None else np.zeros((len(ids_ex), 0), dtype=np.float32)),
            )

        # token importance
        tok_names = ["GLOBAL", "PT_intra", "PT_peri", "LN_intra", "LN_peri"]
        tok_shap_sum = shap_tok.sum(axis=2)  # (N,5)
        tok_imp = np.mean(np.abs(tok_shap_sum), axis=0).reshape(-1)  # (5,)
        pd.DataFrame({"token": tok_names, "mean_abs_shap": tok_imp.tolist()}).to_csv(out_dir / "shap_importance_tokens.csv", index=False)

        plt.figure()
        order = np.argsort(-tok_imp)
        plt.bar([tok_names[i] for i in order], tok_imp[order])
        plt.ylabel("Mean(|SHAP|) (risk@horizon)")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_tokens_bar.png", dpi=300)
        plt.close()

        # clinical plots
        clin_names = clinical_feature_names(clin_enc)
        if use_clin and shap_clin is not None and clin_ex is not None and shap_clin.size > 0:
            X = clin_ex.astype(np.float32)
            S = shap_clin.astype(np.float32)

            plt.figure()
            shap.summary_plot(S, X, feature_names=clin_names, show=False, max_display=int(args.max_display))
            plt.tight_layout()
            plt.savefig(out_dir / "shap_summary_clinical_beeswarm.png", dpi=300)
            plt.close()

            plt.figure()
            shap.summary_plot(S, X, feature_names=clin_names, show=False, plot_type="bar", max_display=int(args.max_display))
            plt.tight_layout()
            plt.savefig(out_dir / "shap_summary_clinical_bar.png", dpi=300)
            plt.close()

            imp = np.mean(np.abs(S), axis=0).reshape(-1)
            pd.DataFrame({"feature": clin_names, "mean_abs_shap": imp.tolist()}).sort_values("mean_abs_shap", ascending=False)\
                .to_csv(out_dir / "shap_importance_clinical.csv", index=False)

        # radiomics plots
        rad_names = radiomics_feature_names(rad_dim)
        if use_rad and shap_rad is not None and rad_ex is not None and shap_rad.size > 0:
            X = rad_ex.astype(np.float32)
            S = shap_rad.astype(np.float32)

            plt.figure()
            shap.summary_plot(S, X, feature_names=rad_names, show=False, max_display=int(args.max_display))
            plt.tight_layout()
            plt.savefig(out_dir / "shap_summary_radiomics_beeswarm.png", dpi=300)
            plt.close()

            plt.figure()
            shap.summary_plot(S, X, feature_names=rad_names, show=False, plot_type="bar", max_display=int(args.max_display))
            plt.tight_layout()
            plt.savefig(out_dir / "shap_summary_radiomics_bar.png", dpi=300)
            plt.close()

            imp = np.mean(np.abs(S), axis=0).reshape(-1)
            pd.DataFrame({"feature": rad_names, "mean_abs_shap": imp.tolist()}).sort_values("mean_abs_shap", ascending=False)\
                .to_csv(out_dir / "shap_importance_radiomics.csv", index=False)

        return {
            "fold": int(fold),
            "tok_imp": tok_imp.astype(float),
            "clin_names": clin_names,
            "clin_X": (clin_ex.astype(np.float32) if (use_clin and clin_ex is not None) else None),
            "clin_S": (shap_clin.astype(np.float32) if (use_clin and shap_clin is not None) else None),
        }


# =============================================================================
# Args
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train_script", type=str, default=DEFAULT_TRAIN_MODULE)

    # repeatable fold checkpoints
    p.add_argument("--ckpt_fold", action="append", nargs="+", default=[],
                   help="Repeatable fold:path pairs. Example: --ckpt_fold 0:/a/last.pt --ckpt_fold 1:/b/last.pt")
    p.add_argument("--ckpt_map_json", type=str, default="")

    p.add_argument("--folds", type=int, nargs="*", default=[])

    # data
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

    p.add_argument("--strict_files", dest="strict_files", action="store_true")
    p.add_argument("--no_strict_files", dest="strict_files", action="store_false")
    p.set_defaults(strict_files=True)

    # clinical / radiomics
    p.add_argument("--clinical_cols", type=str, nargs="*", default=None)
    p.add_argument("--use_radiomics", action="store_true")
    p.add_argument(
        "--radiomics_root",
        type=str,
        default="radiomics_features/radiomics_features",
        help="Radiomics source: directory of per-patient CSVs or a patient-wide CSV file.",
    )
    p.add_argument("--radiomics_pca_total_components", type=int, default=100)

    # model arch
    p.add_argument("--img_size", type=int, nargs=3, default=[256, 256, 128])
    p.add_argument("--feature_size", type=int, default=48)
    p.add_argument("--depths", type=int, nargs=4, default=[2, 2, 2, 2])
    p.add_argument("--num_heads", type=int, nargs=4, default=[3, 6, 12, 24])
    p.add_argument("--drop_rate", type=float, default=0.10)
    p.add_argument("--attn_drop_rate", type=float, default=0.10)
    p.add_argument("--dropout_path_rate", type=float, default=0.20)
    p.add_argument("--use_checkpoint", action="store_true")

    p.add_argument("--fused_dim", type=int, default=512)
    p.add_argument("--token_mlp_dropout", type=float, default=0.30)
    p.add_argument("--attn_mask_bias", type=float, default=2.0)
    p.add_argument("--use_multiscale", action="store_true")
    p.add_argument("--mask_interp", type=str, default="nearest", choices=["nearest", "trilinear"])
    p.add_argument("--min_roi_frac", type=float, default=1e-5)
    p.add_argument("--min_roi_voxels_deep", type=int, default=8)
    p.add_argument("--token_dropout", type=float, default=0.05)

    p.add_argument("--pt_shell_radius", type=int, default=3)
    p.add_argument("--ln_shell_radius", type=int, default=3)

    p.add_argument("--shell_body_from_ct", action="store_true")
    p.add_argument("--body_ct_thr", type=str, default="auto")
    p.add_argument("--body_ct_thr_hu", type=float, default=-500.0)
    p.add_argument("--body_close_r", type=int, default=2)
    p.add_argument("--body_max_frac", type=float, default=0.995)

    p.add_argument("--strict_swinvit_layout", dest="strict_swinvit_layout", action="store_true")
    p.add_argument("--no_strict_swinvit_layout", dest="strict_swinvit_layout", action="store_false")
    p.set_defaults(strict_swinvit_layout=True)
    p.add_argument("--debug_swinvit_layout", action="store_true")

    # head hparams (only to rebuild object)
    p.add_argument("--expert_dropout_p", type=float, default=0.10)
    p.add_argument("--proj_dropout_p", type=float, default=0.20)
    p.add_argument("--attn_dropout_p", type=float, default=0.10)
    p.add_argument("--gate_dropout_p", type=float, default=0.20)
    p.add_argument("--surv_dropout_p", type=float, default=0.40)
    p.add_argument("--clinical_noise_std", type=float, default=0.02)
    p.add_argument("--radiomics_noise_std", type=float, default=0.02)
    p.add_argument("--modality_dropout_clin_p", type=float, default=0.20)
    p.add_argument("--modality_dropout_rad_p", type=float, default=0.20)

    p.add_argument("--time_bin_width_days", type=float, default=180.0)
    p.add_argument("--risk_horizon_days", type=float, default=3 * 365.0)

    # SHAP
    p.add_argument("--n_background", type=int, default=16)
    p.add_argument("--n_explain", type=int, default=200)
    p.add_argument("--max_display", type=int, default=25)
    p.add_argument("--explainer", type=str, default="gradient", choices=["gradient", "deep"])
    p.add_argument("--weights", type=str, default="ema", choices=["last", "ema", "swa"])
    p.add_argument("--save_npz", action="store_true")

    p.add_argument("--pool_clinical", action="store_true")

    # runtime
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strict_load", action="store_true")

    args = p.parse_args()

    if bool(args.splits_dir) == bool(args.splits_csv):
        raise ValueError("Provide exactly one of --splits_dir or --splits_csv.")

    return args


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    train_mod = load_train_module(args.train_script)

    device = train_mod.parse_device(args.device)
    train_mod.bind_cuda_device(device)
    _, autocast_ctx = train_mod.make_amp(device, enabled=bool(args.amp))
    print(f"[info] device={device} amp={bool(args.amp and device.type=='cuda')}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.meta_csv, dtype={args.id_col: str})
    if args.id_col not in meta.columns:
        raise RuntimeError(f"--id_col {args.id_col} not found in meta_csv.")
    meta[args.id_col] = meta[args.id_col].astype(str)

    if (not args.keep_bad_status) and ("status" in meta.columns):
        meta = meta[meta["status"].astype(str).str.lower() == "ok"].copy()
    if (not args.keep_unmatched_survival) and ("survival_matched" in meta.columns):
        sm = meta["survival_matched"]
        if sm.dtype == bool:
            meta = meta[sm].copy()
        else:
            meta = meta[sm.astype(str).str.lower().isin(["true", "1", "t", "yes"])].copy()

    if args.time_col == "" or args.event_col == "":
        tcol, ecol = train_mod.ENDPOINT_MAP[args.endpoint]
        if args.time_col == "":
            args.time_col = tcol
        if args.event_col == "":
            args.event_col = ecol

    meta[args.time_col] = pd.to_numeric(meta[args.time_col], errors="coerce")
    meta[args.event_col] = pd.to_numeric(meta[args.event_col], errors="coerce")
    meta = meta.dropna(subset=[args.time_col, args.event_col]).copy()
    meta = meta[meta[args.time_col] > 0].copy()
    meta[args.event_col] = meta[args.event_col].astype(int)

    splits = train_mod.load_precomputed_splits(int(args.cv_folds), splits_dir=args.splits_dir, splits_csv=args.splits_csv)

    ckpt_map: Dict[int, str] = {}
    if args.ckpt_map_json:
        ckpt_map.update(load_ckpt_map_json(args.ckpt_map_json))
    ckpt_items = _flatten_list_of_lists(args.ckpt_fold)
    if ckpt_items:
        ckpt_map.update(parse_ckpt_fold_items(ckpt_items))

    folds = [int(f) for f in args.folds] if args.folds else list(range(int(args.cv_folds)))

    for f in folds:
        if f not in ckpt_map:
            raise RuntimeError(f"Missing checkpoint for fold={f}. Provide --ckpt_fold {f}:/path/to/last.pt")
        ckpt_p = resolve_path(ckpt_map[f])
        if not ckpt_p.is_file():
            raise RuntimeError(f"Checkpoint not found for fold={f}: {ckpt_map[f]} (resolved: {ckpt_p})")
        ckpt_map[f] = str(ckpt_p)

    token_rows = []
    pooled_clin_names = None
    pooled_clin_X = []
    pooled_clin_S = []

    for f in folds:
        print(f"\n=== SHAP fold {f:02d} ===")
        res = compute_fold_shap(
            train_mod,
            fold=int(f),
            args=args,
            meta=meta,
            split=splits[int(f)],
            ckpt_path=Path(ckpt_map[int(f)]),
            out_dir=out_root / f"fold_{int(f):02d}",
            device=device,
            autocast_ctx=autocast_ctx,
        )

        tok_imp = res["tok_imp"]
        for name, val in zip(["GLOBAL", "PT_intra", "PT_peri", "LN_intra", "LN_peri"], tok_imp):
            token_rows.append({"fold": int(f), "token": name, "mean_abs_shap": float(val)})

        if args.pool_clinical:
            cn = res.get("clin_names")
            X = res.get("clin_X")
            S = res.get("clin_S")
            if cn is not None and X is not None and S is not None and X.size > 0 and S.size > 0:
                if pooled_clin_names is None:
                    pooled_clin_names = list(cn)
                    pooled_clin_X.append(X)
                    pooled_clin_S.append(S)
                else:
                    if list(cn) == list(pooled_clin_names):
                        pooled_clin_X.append(X)
                        pooled_clin_S.append(S)
                    else:
                        print(f"[CV][WARN] clinical feature names differ in fold {f:02d}; skipping pooled clinical for this fold.")

    # CV token summary
    df_tok = pd.DataFrame(token_rows)
    if not df_tok.empty:
        summary = (
            df_tok.groupby("token")["mean_abs_shap"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "cv_mean_abs_shap", "std": "cv_sd_abs_shap"})
        )
        summary.to_csv(out_root / "cv_token_importance.csv", index=False)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure()
        order = np.argsort(-summary["cv_mean_abs_shap"].to_numpy())
        plt.bar(summary.loc[order, "token"].tolist(), summary.loc[order, "cv_mean_abs_shap"].to_numpy())
        plt.ylabel("CV mean(|SHAP|) (risk@horizon)")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(out_root / "cv_token_bar.png", dpi=300)
        plt.close()

        print(f"\n[CV] wrote: {out_root/'cv_token_importance.csv'} and {out_root/'cv_token_bar.png'}")

    # pooled clinical (only if consistent)
    if args.pool_clinical and pooled_clin_names is not None and pooled_clin_X and pooled_clin_S:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        Xp = np.concatenate(pooled_clin_X, axis=0)
        Sp = np.concatenate(pooled_clin_S, axis=0)

        out_cv = out_root / "cv_pooled"
        out_cv.mkdir(parents=True, exist_ok=True)

        plt.figure()
        shap.summary_plot(Sp, Xp, feature_names=pooled_clin_names, show=False, max_display=int(args.max_display))
        plt.tight_layout()
        plt.savefig(out_cv / "shap_cv_clinical_beeswarm.png", dpi=300)
        plt.close()

        plt.figure()
        shap.summary_plot(Sp, Xp, feature_names=pooled_clin_names, show=False, plot_type="bar", max_display=int(args.max_display))
        plt.tight_layout()
        plt.savefig(out_cv / "shap_cv_clinical_bar.png", dpi=300)
        plt.close()

        imp = np.mean(np.abs(Sp), axis=0).reshape(-1)
        pd.DataFrame({"feature": pooled_clin_names, "mean_abs_shap": imp.tolist()}).sort_values("mean_abs_shap", ascending=False)\
            .to_csv(out_cv / "shap_cv_importance_clinical.csv", index=False)

        print(f"[CV] wrote pooled clinical SHAP to: {out_cv}")

    print("\n[done]")


if __name__ == "__main__":
    main()
