#!/usr/bin/env python3
"""
trifusesurv.segmentation.evaluate

Evaluate seg-pretrained SwinUNETR encoder-only model (CT -> union mask).

- Loads ckpt from pretrain_swinunetr_seg.py (expects keys: model_state, head_state, args)
- Computes Dice on:
    * ALL cases (empty-empty contributes 1.0 by definition)
    * POS-only cases (gt has >0 voxels)  <-- this is the meaningful metric for tumor masks
- Evaluates on val/test/both for one fold or all folds.

Example:
CUDA_VISIBLE_DEVICES=0 \
python -m trifusesurv.segmentation.evaluate \
  --meta_csv OPSCC_preprocessed_128/cohort_preprocessed.csv \
  --splits_dir runs/opscc_splits_os_seed1 \
  --cv_folds 4 --debug_fold 0 --strict_splits \
  --ckpt_dir runs/seg_pretrain_swinunetr_from_preprocessed --ckpt_name seg_best.pt \
  --split both --device cuda:0 --amp --workers 8 --thr 0.5
  
"""

from __future__ import annotations

import argparse
import os
import random
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from trifusesurv.models.swinunetr_backbone_utils import (
    build_swinunetr_backbone,
    swinvit_features,
    convert_swinvit_feats_to_channel_first,
    _expected_channels,
)


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

    df = pd.read_csv(splits_csv)
    need = {"patient_id", "fold", "split"}
    if not need.issubset(df.columns):
        raise ValueError(f"--splits_csv must contain {sorted(need)}; got {list(df.columns)}")

    df = df.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(int)
    df["split"] = df["split"].astype(str).str.lower()

    for f in range(int(cv_folds)):
        dff = df[df["fold"] == f]
        out[f] = {
            "train": dff.loc[dff["split"] == "train", "patient_id"].tolist(),
            "val":   dff.loc[dff["split"] == "val",   "patient_id"].tolist(),
            "test":  dff.loc[dff["split"] == "test",  "patient_id"].tolist(),
        }
    return out


def read_nii(path: str, dtype=np.float32) -> np.ndarray:
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (D,H,W)
    arr = np.asarray(arr, dtype=dtype)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


class SegPreprocessedDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, ids: List[str], *, id_col: str, ct_col: str, mask_col: str, strict: bool):
        self.meta = meta.set_index(id_col, drop=False)
        self.ids = [str(x) for x in ids]
        self.ct_col = ct_col
        self.mask_col = mask_col
        self.strict = bool(strict)

        ok, miss = [], []
        for pid in self.ids:
            if pid not in self.meta.index:
                miss.append(pid); continue
            ct_p = str(self.meta.loc[pid, self.ct_col])
            m_p  = str(self.meta.loc[pid, self.mask_col])
            if os.path.isfile(ct_p) and os.path.isfile(m_p):
                ok.append(pid)
            else:
                miss.append(pid)

        if miss:
            msg = f"[eval] {len(miss)} id(s) missing meta/files. First: {miss[:10]}"
            if self.strict:
                raise RuntimeError(msg)
            print("[WARN]", msg, "-> dropping")
        self.ids = ok

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i: int):
        pid = self.ids[i]
        ct = read_nii(str(self.meta.loc[pid, self.ct_col]), dtype=np.float32)       # [0,1]
        m  = read_nii(str(self.meta.loc[pid, self.mask_col]), dtype=np.float32)
        m  = (m > 0.5).astype(np.float32)
        x = torch.from_numpy(ct).unsqueeze(0)   # (1,D,H,W)
        y = torch.from_numpy(m).unsqueeze(0)    # (1,D,H,W)
        return x, y, pid


class SwinViTEncoderSeg(nn.Module):
    """
    Same encoder-only segmentation head used in pretraining:
      x -> swinViT feats -> deepest feat -> 1x1 head -> logits_low
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.normalize = True
        feature_size = int(cfg["feature_size"])
        self._expected_c = _expected_channels(feature_size, max_pow=6)
        self.strict_layout = bool(cfg.get("strict_swinvit_layout", False))
        self.debug_layout = bool(cfg.get("debug_swinvit_layout", False))
        self._checked = False

        self.backbone = build_swinunetr_backbone(
            img_size=tuple(cfg["img_size"]),
            in_channels=1,
            out_channels=2,
            feature_size=feature_size,
            depths=tuple(cfg["depths"]),
            num_heads=tuple(cfg["num_heads"]),
            drop_rate=float(cfg["drop_rate"]),
            attn_drop_rate=float(cfg["attn_drop_rate"]),
            dropout_path_rate=float(cfg["dropout_path_rate"]),
            normalize=True,
            use_checkpoint=bool(cfg.get("use_checkpoint", False)),
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
        fdeep = feats[-1]  # (B,C,d,h,w)
        return self.head(fdeep)  # (B,1,d,h,w)


def dice_binary(p: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # p,y: (B,1,*,*,*) float in {0,1} for y, {0,1} for p
    inter = (p * y).sum(dim=(2,3,4))
    den = p.sum(dim=(2,3,4)) + y.sum(dim=(2,3,4)) + eps
    return ((2.0 * inter + eps) / den).squeeze(1)  # (B,)


@torch.no_grad()
def eval_split(model: nn.Module, loader: DataLoader, device: torch.device, autocast_ctx, thr: float = 0.5):
    model.eval()
    dice_all = []
    dice_pos = []
    dice_empty = []
    n_pos = 0
    n_empty = 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast_ctx():
            logits_low = model(x)  # (B,1,d,h,w)

        # compare at low-res (same as training objective)
        y_low = F.interpolate(y, size=logits_low.shape[2:], mode="nearest")
        p_low = (torch.sigmoid(logits_low) >= thr).float()

        d = dice_binary(p_low, y_low)  # (B,)
        dice_all.extend(d.detach().cpu().numpy().tolist())

        ysum = y_low.sum(dim=(2,3,4)).squeeze(1)  # (B,)
        pos_mask = (ysum > 0)
        empty_mask = ~pos_mask
        if pos_mask.any():
            dice_pos.extend(d[pos_mask].detach().cpu().numpy().tolist())
            n_pos += int(pos_mask.sum().item())
        if empty_mask.any():
            dice_empty.extend(d[empty_mask].detach().cpu().numpy().tolist())
            n_empty += int(empty_mask.sum().item())

    out = {
        "n_total": int(len(dice_all)),
        "n_pos_gt": int(n_pos),
        "n_empty_gt": int(n_empty),
        "dice_all_mean": float(np.mean(dice_all)) if dice_all else float("nan"),
        "dice_pos_mean": float(np.mean(dice_pos)) if dice_pos else float("nan"),
        "dice_empty_mean": float(np.mean(dice_empty)) if dice_empty else float("nan"),
    }
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--meta_csv", required=True)
    p.add_argument("--id_col", type=str, default="patient_id")
    p.add_argument("--ct_col", type=str, default="ct_out_path")
    p.add_argument("--mask_col", type=str, default="mask_union_out_path")

    p.add_argument("--keep_bad_status", action="store_true")
    p.add_argument("--keep_unmatched_survival", action="store_true")

    p.add_argument("--splits_dir", type=str, default="")
    p.add_argument("--splits_csv", type=str, default="")
    p.add_argument("--cv_folds", type=int, default=4)
    p.add_argument("--debug_fold", type=int, default=-1)
    p.add_argument("--strict_splits", action="store_true")

    p.add_argument("--ckpt", type=str, default="", help="Single checkpoint path (overrides ckpt_dir/name).")
    p.add_argument("--ckpt_dir", type=str, default="", help="Directory containing fold_XX/ckpt_name.")
    p.add_argument("--ckpt_name", type=str, default="seg_best.pt")

    p.add_argument("--split", type=str, default="both", choices=["val", "test", "both"])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)

    p.add_argument("--device", type=str, default="", help="cpu|cuda|cuda:N (default cuda:0)")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--thr", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()

    dev = str(args.device).strip().lower()
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
        raise ValueError(f"--device must be cpu|cuda|cuda:N (or empty), got: {args.device}")
    if device.type == "cuda":
        torch.cuda.set_device(int(device.index) if device.index is not None else 0)

    amp_enabled = bool(args.amp and device.type == "cuda")
    try:
        autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=amp_enabled)
    except Exception:
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=amp_enabled)

    meta = pd.read_csv(args.meta_csv)
    meta[args.id_col] = meta[args.id_col].astype(str)

    if (not args.keep_bad_status) and ("status" in meta.columns):
        meta = meta[meta["status"].astype(str).str.lower() == "ok"].copy()
    if (not args.keep_unmatched_survival) and ("survival_matched" in meta.columns):
        sm = meta["survival_matched"]
        if sm.dtype == bool:
            meta = meta[sm].copy()
        else:
            meta = meta[sm.astype(str).str.lower().isin(["true","1","t","yes"])].copy()

    splits = load_precomputed_splits(args.cv_folds, splits_dir=args.splits_dir, splits_csv=args.splits_csv)
    folds = [int(args.debug_fold)] if int(args.debug_fold) >= 0 else list(range(int(args.cv_folds)))

    for f in folds:
        ckpt_path = args.ckpt
        if ckpt_path == "":
            if args.ckpt_dir == "":
                raise ValueError("Provide --ckpt or --ckpt_dir.")
            ckpt_path = os.path.join(args.ckpt_dir, f"fold_{f:02d}", args.ckpt_name)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("args", {})
        # fill required keys if missing
        cfg = dict(cfg)
        cfg.setdefault("img_size", [256, 256, 128])
        cfg.setdefault("feature_size", 48)
        cfg.setdefault("depths", [2,2,2,2])
        cfg.setdefault("num_heads", [3,6,12,24])
        cfg.setdefault("drop_rate", 0.10)
        cfg.setdefault("attn_drop_rate", 0.10)
        cfg.setdefault("dropout_path_rate", 0.20)
        cfg.setdefault("use_checkpoint", False)
        cfg.setdefault("strict_swinvit_layout", False)
        cfg.setdefault("debug_swinvit_layout", False)

        model = SwinViTEncoderSeg(cfg).to(device)
        # load backbone/head
        if "model_state" in ckpt:
            model.backbone.load_state_dict(ckpt["model_state"], strict=False)
        if "head_state" in ckpt and ckpt["head_state"] is not None:
            model.head.load_state_dict(ckpt["head_state"], strict=False)

        # build loaders
        val_ids = splits[f]["val"]
        test_ids = splits[f]["test"]

        if args.split in ("val", "both"):
            va_ds = SegPreprocessedDataset(meta, val_ids, id_col=args.id_col, ct_col=args.ct_col, mask_col=args.mask_col, strict=args.strict_splits)
            va_loader = DataLoader(va_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.workers),
                                   pin_memory=(device.type=="cuda"), drop_last=False, worker_init_fn=seed_worker)
            res = eval_split(model, va_loader, device, autocast_ctx, thr=float(args.thr))
            print(f"[fold {f:02d}][VAL] ckpt={ckpt_path} thr={args.thr} -> {res}")

        if args.split in ("test", "both"):
            te_ds = SegPreprocessedDataset(meta, test_ids, id_col=args.id_col, ct_col=args.ct_col, mask_col=args.mask_col, strict=args.strict_splits)
            te_loader = DataLoader(te_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.workers),
                                   pin_memory=(device.type=="cuda"), drop_last=False, worker_init_fn=seed_worker)
            res = eval_split(model, te_loader, device, autocast_ctx, thr=float(args.thr))
            print(f"[fold {f:02d}][TEST] ckpt={ckpt_path} thr={args.thr} -> {res}")


if __name__ == "__main__":
    main()
