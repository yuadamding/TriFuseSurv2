#!/usr/bin/env python3
"""
trifusesurv.segmentation.train (TRAIN ON ALL SAMPLES; NO VAL/TEST; AMP-SAFE)

Uses preprocess_for_swinunetr_survival.py outputs:
- Predictor: ct_out_path (ct.nii.gz, float32 in [0,1])
- Response:  --mask_col points to a mask NIfTI (union/primary/nodal) (0/1)

OOM-safe:
- DOES NOT call SwinUNETR full forward (no decoder).
- Uses SwinUNETR.swinViT encoder features + tiny head on deepest feature map.
- Loss computed at deep resolution using downsampled GT.

TRAINING POLICY (as requested):
- NO validation set and NO test set.
- All selected samples are used for training.
- We still print training loss + training dice for monitoring.

Modes:
- --train_mode all (default): one model trained on ALL usable samples -> out_dir/all/
- --train_mode cv: for each fold, trains on (train+val+test) IDs -> out_dir/fold_XX/
  (folds are ONLY output organization; NOT for splitting)

Coverage of ROI:
- Default loss is Tversky with beta>alpha (penalize FN more => better coverage).
- Optional positive oversampling based on voxel-count columns (useful for nodal).

CRITICAL correctness fixes:
- Loss is computed in FP32 (outside autocast) to prevent AMP overflow & skipped steps.
- GT downsample uses adaptive_max_pool3d (positives never vanish due to nearest downsample).
- LazyConv3d head is materialized BEFORE optimizer construction.

CUDA_VISIBLE_DEVICES=0 \
python -m trifusesurv.segmentation.train \
  --meta_csv /rsrch8/home/bcb/yding4/radiomics/improve/OPSCC_preprocessed_128/cohort_preprocessed.csv \
  --train_mode all \
  --out_dir runs/seg_overfit_pt_big_stable \
  --mask_col mask_primary_out_path \
  --img_size 128 256 256 \
  --feature_size 96 \
  --depths 2 2 18 2 \
  --num_heads 3 6 12 24 \
  --drop_rate 0 --attn_drop_rate 0 --dropout_path_rate 0 \
  --epochs 100 --batch_size 1 --workers 16 --amp --use_checkpoint \
  --device cuda:0 \
  --lr 5e-5 --wd 0 \
  --grad_clip 1.0 \
  --loss bce_dice --max_pos_weight 200 \
  --pos_oversample 1


CUDA_VISIBLE_DEVICES=1 \
python -m trifusesurv.segmentation.train \
  --meta_csv /rsrch8/home/bcb/yding4/radiomics/improve/OPSCC_preprocessed_128/cohort_preprocessed.csv \
  --train_mode all \
  --out_dir runs/seg_overfit_ln_big_stable \
  --mask_col mask_nodal_out_path \
  --img_size 128 256 256 \
  --feature_size 96 \
  --depths 2 2 18 2 \
  --num_heads 3 6 12 24 \
  --drop_rate 0 --attn_drop_rate 0 --dropout_path_rate 0 \
  --epochs 100 --batch_size 1 --workers 16 --amp --use_checkpoint \
  --device cuda:0 \
  --lr 5e-5 --wd 0 \
  --grad_clip 1.0 \
  --loss bce_dice --max_pos_weight 200 \
  --pos_oversample 1




"""

from __future__ import annotations

import argparse
import os
import random
from typing import List, Dict, Tuple, Optional, Sequence

import numpy as np
import pandas as pd
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from trifusesurv.models.swinunetr_backbone_utils import (
    build_swinunetr_backbone,
    swinvit_features,
    convert_swinvit_feats_to_channel_first,
    _expected_channels,
)


# ---------------------------
# Seeds / device / AMP
# ---------------------------
def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    s = torch.initial_seed() % (2**32)
    np.random.seed(s)
    random.seed(s)


def parse_device(device_str: str) -> torch.device:
    dev = str(device_str).strip().lower()
    if dev == "" or dev == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif dev == "cpu":
        device = torch.device("cpu")
    elif dev == "cuda":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif dev.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device but CUDA is not available.")
        device = torch.device(dev)
    else:
        raise ValueError(f"--device must be cpu|cuda|cuda:N (or empty), got: {device_str}")

    if device.type == "cuda":
        torch.cuda.set_device(int(device.index) if device.index is not None else 0)
    return device


def make_amp(device: torch.device, enabled: bool):
    """
    Returns (scaler, autocast_ctx).
    Loss will be computed in FP32 regardless; autocast is for forward only.
    """
    amp_enabled = bool(enabled and device.type == "cuda")
    if not amp_enabled:
        return None, (lambda: torch.cuda.amp.autocast(enabled=False))  # harmless null-autocast

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=True)
        return scaler, autocast_ctx
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=True)
        return scaler, autocast_ctx


# ---------------------------
# Splits I/O (cv mode uses these only to enumerate IDs)
# ---------------------------
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
            out[f] = {
                "train": _read_id_list(os.path.join(fold_dir, "train_ids.txt")),
                "val":   _read_id_list(os.path.join(fold_dir, "val_ids.txt")),
                "test":  _read_id_list(os.path.join(fold_dir, "test_ids.txt")),
            }
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


# ---------------------------
# NIfTI reading + augmentation
# ---------------------------
def read_nii(path: str, dtype=np.float32) -> np.ndarray:
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (D,H,W)
    arr = np.asarray(arr, dtype=dtype)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def rand_flip(ct: np.ndarray, m: np.ndarray, p: float = 0.5):
    if random.random() < p:
        ct = np.flip(ct, 0).copy(); m = np.flip(m, 0).copy()
    if random.random() < p:
        ct = np.flip(ct, 1).copy(); m = np.flip(m, 1).copy()
    if random.random() < p:
        ct = np.flip(ct, 2).copy(); m = np.flip(m, 2).copy()
    return ct, m


def rand_intensity(ct: np.ndarray, p: float = 0.3):
    if random.random() < p:
        ct = ct + 0.02 * np.random.randn(*ct.shape).astype(np.float32)
    if random.random() < p:
        scale = float(np.clip(1.0 + 0.05 * np.random.randn(), 0.9, 1.1))
        ct = ct * scale
    return np.clip(ct, 0.0, 1.0).astype(np.float32)


def _find_first_existing_path(meta: pd.DataFrame, col: str) -> Optional[str]:
    if col not in meta.columns:
        return None
    for p in meta[col].astype(str).tolist():
        if p and os.path.isfile(p):
            return p
    return None


def resolve_img_size_against_data(meta: pd.DataFrame, ct_col: str, img_size_arg: Sequence[int]) -> Tuple[Tuple[int,int,int], Tuple[int,int,int]]:
    arg = tuple(int(x) for x in img_size_arg)
    p0 = _find_first_existing_path(meta, ct_col)
    if p0 is None:
        print("[WARN] Could not find any existing CT path to validate --img_size. Using as-is:", arg)
        return arg, arg

    shp = tuple(read_nii(p0, dtype=np.float32).shape)  # (D,H,W)
    if shp == arg:
        return arg, shp
    if shp == tuple(reversed(arg)):
        print(f"[img_size][WARN] Data shape is {shp} (D,H,W) but --img_size={arg}. "
              f"Interpreting --img_size as (H,W,D) and flipping to (D,H,W)={shp}.")
        return shp, shp

    raise RuntimeError(
        f"[img_size] Mismatch: data CT shape (D,H,W)={shp} but --img_size={arg} (and reversed={tuple(reversed(arg))}). "
        f"Fix your CLI --img_size to match the NIfTI array shape."
    )


# ---------------------------
# Dataset
# ---------------------------
class SegPreprocessedDataset(Dataset):
    def __init__(
        self,
        meta: pd.DataFrame,
        ids: List[str],
        *,
        id_col: str,
        ct_col: str,
        mask_col: str,
        train: bool,
        strict_files: bool,
        expected_dhw: Optional[Tuple[int,int,int]] = None,
    ):
        self.meta = meta.set_index(id_col, drop=False)
        self.ids = [str(x) for x in ids]
        self.ct_col = ct_col
        self.mask_col = mask_col
        self.train = bool(train)
        self.strict_files = bool(strict_files)
        self.expected_dhw = tuple(expected_dhw) if expected_dhw is not None else None

        ok, miss = [], []
        for pid in self.ids:
            if pid not in self.meta.index:
                miss.append(pid); continue
            ct_p = str(self.meta.loc[pid, self.ct_col])
            m_p = str(self.meta.loc[pid, self.mask_col])
            if os.path.isfile(ct_p) and os.path.isfile(m_p):
                ok.append(pid)
            else:
                miss.append(pid)

        if miss:
            msg = f"[seg] {len(miss)} id(s) missing meta/files. First: {miss[:10]}"
            if self.strict_files:
                raise RuntimeError(msg)
            print("[WARN]", msg, "-> dropping")
        self.ids = ok

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i: int):
        pid = self.ids[i]
        ct = read_nii(str(self.meta.loc[pid, self.ct_col]), dtype=np.float32)
        m  = read_nii(str(self.meta.loc[pid, self.mask_col]), dtype=np.float32)
        m = (m > 0.5).astype(np.float32)

        if self.expected_dhw is not None:
            if tuple(ct.shape) != self.expected_dhw:
                raise RuntimeError(f"[seg][SHAPE] pid={pid} ct {tuple(ct.shape)} != expected {self.expected_dhw} (D,H,W).")
            if tuple(m.shape) != self.expected_dhw:
                raise RuntimeError(f"[seg][SHAPE] pid={pid} mask {tuple(m.shape)} != expected {self.expected_dhw} (D,H,W).")

        if self.train:
            ct, m = rand_flip(ct, m, p=0.5)
            ct = rand_intensity(ct, p=0.3)

        x = torch.from_numpy(ct).unsqueeze(0)  # (1,D,H,W)
        y = torch.from_numpy(m).unsqueeze(0)   # (1,D,H,W)
        return x, y, pid


# ---------------------------
# Losses (coverage/recall friendly)
# ---------------------------
def dice_per_sample_from_logits(logits: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits).clamp(0, 1)
    y = y.float()
    inter = (p * y).sum(dim=(2,3,4))
    den = p.sum(dim=(2,3,4)) + y.sum(dim=(2,3,4)) + eps
    return ((2.0 * inter + eps) / den).squeeze(1)  # (B,)


def bce_dice_loss(logits: torch.Tensor, y: torch.Tensor, max_pos_weight: float = 200.0) -> torch.Tensor:
    y = y.float()
    pos = float(y.sum().detach().cpu().item())
    tot = float(y.numel())
    neg = max(tot - pos, 0.0)
    if pos > 0:
        pw = min(max(neg / max(pos, 1.0), 1.0), float(max_pos_weight))
        pos_weight = logits.new_tensor(pw)
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
    else:
        bce = F.binary_cross_entropy_with_logits(logits, y)
    d = dice_per_sample_from_logits(logits, y).mean()
    return bce + (1.0 - d)


def tversky_loss_from_logits(logits: torch.Tensor, y: torch.Tensor, alpha: float = 0.2, beta: float = 0.8, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits).clamp(0, 1)
    y = y.float()
    TP = (p * y).sum(dim=(2,3,4))
    FP = (p * (1.0 - y)).sum(dim=(2,3,4))
    FN = ((1.0 - p) * y).sum(dim=(2,3,4))
    TI = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
    return (1.0 - TI).mean()


# ---------------------------
# Model: encoder-only seg head
# ---------------------------
class SwinViTEncoderSeg(nn.Module):
    """
    x -> swinViT feats -> deepest feat -> 1x1 head -> logits_low
    """
    def __init__(
        self,
        *,
        img_size,
        feature_size,
        depths,
        num_heads,
        drop_rate,
        attn_drop_rate,
        dropout_path_rate,
        use_checkpoint,
        strict_layout: bool,
        debug_layout: bool,
    ):
        super().__init__()
        self.normalize = True
        self.strict_layout = bool(strict_layout)
        self.debug_layout = bool(debug_layout)
        self._checked = False
        self._expected_c = _expected_channels(int(feature_size), max_pow=6)

        self.backbone = build_swinunetr_backbone(
            img_size=tuple(img_size),
            in_channels=1,
            out_channels=2,
            feature_size=int(feature_size),
            depths=tuple(depths),
            num_heads=tuple(num_heads),
            drop_rate=float(drop_rate),
            attn_drop_rate=float(attn_drop_rate),
            dropout_path_rate=float(dropout_path_rate),
            normalize=True,
            use_checkpoint=bool(use_checkpoint),
            spatial_dims=3,
        )
        self.head = nn.LazyConv3d(1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = swinvit_features(self.backbone, x, self.normalize)
        feats = convert_swinvit_feats_to_channel_first(
            feats,
            self._expected_c,
            strict=self.strict_layout,
            print_shapes=(self.debug_layout and (not self._checked)),
            tag="swinViT",
        )
        self._checked = True
        fdeep = feats[-2]
        return self.head(fdeep)


# ---------------------------
# Train on ALL selected IDs (no val/test)
# ---------------------------
def run_one_training_all(
    args,
    meta: pd.DataFrame,
    ids_all: List[str],
    out_dir: str,
    device: torch.device,
    expected_dhw: Tuple[int,int,int],
):
    os.makedirs(out_dir, exist_ok=True)

    scaler, autocast_ctx = make_amp(device, enabled=bool(args.amp))
    amp_enabled = bool(args.amp and device.type == "cuda")

    ds = SegPreprocessedDataset(
        meta, ids_all,
        id_col=args.id_col, ct_col=args.ct_col, mask_col=args.mask_col,
        train=True, strict_files=args.strict_files, expected_dhw=expected_dhw
    )
    print(f"[seg] out_dir={out_dir} TRAIN(all)={len(ds)}")

    # optional positive oversampling
    sampler = None
    if args.pos_oversample > 1.0:
        vox_col = args.pos_voxels_col.strip()
        weights = np.ones((len(ds),), dtype=np.float64)

        if vox_col and (vox_col in meta.columns):
            meta_idx = meta.set_index(args.id_col, drop=False)
            pos_mask = []
            for pid in ds.ids:
                v = meta_idx.loc[str(pid), vox_col] if str(pid) in meta_idx.index else 0
                try:
                    pos_mask.append(float(v) > 0)
                except Exception:
                    pos_mask.append(False)
            pos_mask = np.asarray(pos_mask, dtype=bool)
            weights[pos_mask] = float(args.pos_oversample)
            print(f"[seg][sampler] pos_oversample={args.pos_oversample} using {vox_col}: pos={int(pos_mask.sum())} neg={int((~pos_mask).sum())}")

            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(weights),
                num_samples=len(weights),
                replacement=True,
            )
        else:
            print(f"[seg][sampler][WARN] pos_oversample>1 but pos_voxels_col='{vox_col}' not found; oversampling disabled.")

    g = torch.Generator()
    g.manual_seed(int(args.seed + 777))

    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(args.workers),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(int(args.workers) > 0),
        worker_init_fn=seed_worker,
        generator=g,
    )

    model = SwinViTEncoderSeg(
        img_size=tuple(expected_dhw),
        feature_size=args.feature_size,
        depths=tuple(args.depths),
        num_heads=tuple(args.num_heads),
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint,
        strict_layout=args.strict_swinvit_layout,
        debug_layout=args.debug_swinvit_layout,
    ).to(device)

    # ---- materialize LazyConv3d BEFORE optimizer ----
    model.eval()
    first = next(iter(loader), None)
    if first is None:
        raise RuntimeError("Empty training loader; cannot warmup lazy head.")
    x0, _, _ = first
    x0 = x0.to(device, non_blocking=True)
    with torch.no_grad():
        with autocast_ctx():
            _ = model(x0)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.wd))

    best = -1e9
    best_path = os.path.join(out_dir, "seg_best.pt")
    last_path = os.path.join(out_dir, "seg_last.pt")
    log_path = os.path.join(out_dir, "train_log.csv")

    prev_head_mean = None

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        dice_all = []
        dice_pos = []

        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            # forward under autocast
            with autocast_ctx():
                logits_low = model(x)  # (B,1,d,h,w) maybe fp16 under AMP

            # ---- FP32 loss (CRITICAL for AMP stability) ----
            logits_f = logits_low.float()

            # downsample GT with max-pool to preserve positives
            y_low = F.adaptive_max_pool3d(y.float(), output_size=logits_f.shape[2:]).clamp(0, 1)

            if args.loss == "tversky":
                loss = tversky_loss_from_logits(
                    logits_f, y_low,
                    alpha=float(args.tversky_alpha),
                    beta=float(args.tversky_beta),
                )
            else:
                loss = bce_dice_loss(logits_f, y_low, max_pos_weight=float(args.max_pos_weight))

            if amp_enabled and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                opt.step()

            losses.append(float(loss.detach().cpu().item()))

            # dice monitoring in FP32
            d = dice_per_sample_from_logits(logits_f.detach(), y_low.detach())
            dice_all.extend([float(v) for v in d.cpu().numpy().tolist()])

            gt_sum = y_low.detach().sum(dim=(2,3,4)).squeeze(1).cpu().numpy()
            for vv, s in zip(d.cpu().numpy(), gt_sum):
                if float(s) > 0:
                    dice_pos.append(float(vv))

        tr_loss = float(np.mean(losses)) if losses else float("nan")
        tr_d_all = float(np.mean(dice_all)) if dice_all else float("nan")
        tr_d_pos = float(np.mean(dice_pos)) if dice_pos else float("nan")

        # checkpoint
        ckpt = {
            "epoch": int(epoch),
            "mode": "all_train",
            "model_state": model.backbone.state_dict(),
            "head_state": model.head.state_dict(),
            "args": vars(args),
            "train_loss": tr_loss,
            "train_dice_all": tr_d_all,
            "train_dice_pos": tr_d_pos,
            "img_size_dhw": tuple(expected_dhw),
        }
        torch.save(ckpt, last_path)

        # log
        pd.DataFrame([{
            "epoch": int(epoch),
            "train_loss": tr_loss,
            "train_dice_all": tr_d_all,
            "train_dice_pos": tr_d_pos,
        }]).to_csv(log_path, mode="a", header=(not os.path.isfile(log_path)), index=False)

        # best selection without val
        score = tr_d_pos if np.isfinite(tr_d_pos) else (tr_d_all if np.isfinite(tr_d_all) else (-tr_loss))
        if np.isfinite(score) and score > best:
            best = float(score)
            torch.save(ckpt, best_path)

        # debug: head update + scaler scale
        with torch.no_grad():
            head_mean = float(model.head.weight.abs().mean().detach().cpu().item())
        delta = 0.0 if prev_head_mean is None else (head_mean - prev_head_mean)
        prev_head_mean = head_mean
        scale_val = float(scaler.get_scale()) if (scaler is not None and amp_enabled) else 1.0

        print(f"[seg] epoch {epoch:03d} | train_loss={tr_loss:.4f} | train_dice_all={tr_d_all:.4f} | train_dice_pos={tr_d_pos:.4f}")
        print(f"[debug] head |mean|={head_mean:.12e} delta={delta:.12e} amp_scale={scale_val:.1f}")

    print(f"[seg] done | best_score={best:.4f} -> {best_path}")


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--meta_csv", required=True)
    p.add_argument("--id_col", type=str, default="patient_id")
    p.add_argument("--ct_col", type=str, default="ct_out_path")
    p.add_argument("--mask_col", type=str, default="mask_union_out_path")

    p.add_argument("--keep_bad_status", action="store_true")
    p.add_argument("--keep_unmatched_survival", action="store_true")

    p.add_argument("--train_mode", type=str, default="all", choices=["all", "cv"])

    # cv mode (ID enumeration only)
    p.add_argument("--splits_dir", type=str, default="")
    p.add_argument("--splits_csv", type=str, default="")
    p.add_argument("--cv_folds", type=int, default=4)
    p.add_argument("--debug_fold", type=int, default=-1)

    # strictness
    p.add_argument("--strict_files", dest="strict_files", action="store_true", help="Error if any ct/mask is missing.")
    p.add_argument("--no_strict_files", dest="strict_files", action="store_false", help="Drop missing ct/mask rows.")
    p.set_defaults(strict_files=False)

    p.add_argument("--out_dir", type=str, default="runs/seg_pretrain_swinunetr_from_preprocessed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="")

    # model cfg
    p.add_argument("--img_size", type=int, nargs=3, default=[256, 256, 128])
    p.add_argument("--feature_size", type=int, default=48)
    p.add_argument("--depths", type=int, nargs=4, default=[2,2,2,2])
    p.add_argument("--num_heads", type=int, nargs=4, default=[3,6,12,24])
    p.add_argument("--drop_rate", type=float, default=0.10)
    p.add_argument("--attn_drop_rate", type=float, default=0.10)
    p.add_argument("--dropout_path_rate", type=float, default=0.20)

    p.add_argument("--use_checkpoint", action="store_true")

    p.add_argument("--strict_swinvit_layout", dest="strict_swinvit_layout", action="store_true")
    p.add_argument("--no_strict_swinvit_layout", dest="strict_swinvit_layout", action="store_false")
    p.set_defaults(strict_swinvit_layout=True)
    p.add_argument("--debug_swinvit_layout", action="store_true")

    # optim
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad_clip", type=float, default=1.0)

    # ROI coverage knobs
    p.add_argument("--loss", type=str, default="tversky", choices=["tversky", "bce_dice"])
    p.add_argument("--tversky_alpha", type=float, default=0.2)
    p.add_argument("--tversky_beta", type=float, default=0.8)
    p.add_argument("--max_pos_weight", type=float, default=200.0)

    # positive oversampling
    p.add_argument("--pos_oversample", type=float, default=1.0)
    p.add_argument("--pos_voxels_col", type=str, default="")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = parse_device(args.device)
    print(f"[seg] device={device} amp={bool(args.amp and device.type=='cuda')} train_mode={args.train_mode}")

    meta = pd.read_csv(args.meta_csv, dtype={args.id_col: str})
    meta[args.id_col] = meta[args.id_col].astype(str)

    # filter like training scripts
    if (not args.keep_bad_status) and ("status" in meta.columns):
        meta = meta[meta["status"].astype(str).str.lower() == "ok"].copy()
    if (not args.keep_unmatched_survival) and ("survival_matched" in meta.columns):
        sm = meta["survival_matched"]
        if sm.dtype == bool:
            meta = meta[sm].copy()
        else:
            meta = meta[sm.astype(str).str.lower().isin(["true","1","t","yes"])].copy()

    meta = meta.copy()
    set_seed(args.seed)

    img_size_dhw, data_shape = resolve_img_size_against_data(meta, args.ct_col, args.img_size)
    print(f"[seg] data_shape(D,H,W)={data_shape} | using img_size(D,H,W)={img_size_dhw}")

    # infer pos_voxels_col if not provided
    if not args.pos_voxels_col.strip():
        mc = args.mask_col.lower()
        if "primary" in mc:
            args.pos_voxels_col = "mask_primary_voxels"
        elif "nodal" in mc:
            args.pos_voxels_col = "mask_nodal_voxels"
        else:
            args.pos_voxels_col = "mask_union_voxels"
        print(f"[seg] inferred pos_voxels_col={args.pos_voxels_col}")

    if args.train_mode == "all":
        ids_all = meta[args.id_col].astype(str).unique().tolist()
        out_run_dir = os.path.join(args.out_dir, "all")
        run_one_training_all(args, meta, ids_all, out_run_dir, device, expected_dhw=img_size_dhw)
        return

    # cv mode: train on union of train/val/test IDs (no split)
    if not (args.splits_dir or args.splits_csv):
        raise ValueError("cv mode requires --splits_dir or --splits_csv.")
    splits = load_precomputed_splits(args.cv_folds, splits_dir=args.splits_dir, splits_csv=args.splits_csv)
    folds = [int(args.debug_fold)] if int(args.debug_fold) >= 0 else list(range(int(args.cv_folds)))

    for f in folds:
        set_seed(args.seed + 100 * int(f))
        ids = splits[int(f)]["train"] + splits[int(f)]["val"] + splits[int(f)]["test"]
        ids = list(dict.fromkeys([str(x) for x in ids]))
        out_fold_dir = os.path.join(args.out_dir, f"fold_{int(f):02d}")
        run_one_training_all(args, meta, ids, out_fold_dir, device, expected_dhw=img_size_dhw)


if __name__ == "__main__":
    main()
