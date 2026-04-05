#!/usr/bin/env python3
"""
trifusesurv.segmentation.gradcam  (UPDATED for latest SwinViTEncoderSeg)

Generate Grad-CAM for ALL samples (or one sample) from ONE checkpoint for the seg-pretrained encoder-only model.

Inputs:
- meta_csv: cohort_preprocessed.csv (or cohort_preprocessed_with_clin.csv)
  must include:
    patient_id, ct_out_path, mask_union_out_path
  optionally:
    mask_primary_out_path, mask_nodal_out_path

- ckpt_path: seg-pretrain checkpoint (seg_best.pt) produced by pretrain_swinunetr_seg.py
  expected keys:
    model_state (backbone), head_state (head), args (cfg), optional img_size_dhw

Outputs:
out_dir/
  all/<patient_id>/
    cam_low.nii.gz        (optional, low-res)
    cam_up.nii.gz         (upsampled to CT voxel grid)
    gt_union.nii.gz       (the target mask used for CAM scoring)
    gt_primary.nii.gz     (if available)
    gt_nodal.nii.gz       (if available)
    overlay_axial.png     (optional)
    overlay_coronal.png   (optional)
    overlay_sagittal.png  (optional)

Important updates:
- Low-res GT uses adaptive_max_pool3d (matches latest pretrain; prevents ROI vanish).
- CAM score uses FP32 logits (stable under AMP).
- --feat_index controls which swinViT feature is used (default -1).

CUDA_VISIBLE_DEVICES=0 \
python -m trifusesurv.segmentation.gradcam \
  --meta_csv /rsrch8/home/bcb/yding4/radiomics/improve/OPSCC_preprocessed_128/cohort_preprocessed.csv \
  --ckpt_path runs/seg_overfit_pt_big_stable/all/seg_best.pt \
  --mask_target_col mask_primary_out_path \
  --out_dir runs/gradcam_pt \
  --cam_target gt --slice_mode cam_roi \
  --device cuda:0 --amp \
  --feat_index -2 \
  --save_png --save_low --skip_existing
  
CUDA_VISIBLE_DEVICES=2 \
python -m trifusesurv.segmentation.gradcam \
  --meta_csv /rsrch8/home/bcb/yding4/radiomics/improve/OPSCC_preprocessed_128/cohort_preprocessed.csv \
  --ckpt_path runs/seg_overfit_ln_big_stable/all/seg_best.pt \
  --mask_target_col mask_nodal_out_path \
  --out_dir runs/gradcam_ln \
  --cam_target gt --slice_mode cam_roi \
  --device cuda:0 --amp \
   --feat_index -2 \
  --save_png --save_low --skip_existing

"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from trifusesurv.models.swinunetr_backbone_utils import (
    build_swinunetr_backbone,
    swinvit_features,
    convert_swinvit_feats_to_channel_first,
    _expected_channels,
)


# -----------------------
# Device / AMP
# -----------------------
def set_device_from_arg(device_str: str) -> torch.device:
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


def make_autocast(device: torch.device, enabled: bool):
    amp_enabled = bool(enabled and device.type == "cuda")
    if not amp_enabled:
        return (lambda: torch.cuda.amp.autocast(enabled=False))
    try:
        return (lambda: torch.amp.autocast("cuda", enabled=True))
    except Exception:
        return (lambda: torch.cuda.amp.autocast(enabled=True))


# -----------------------
# SITK helpers
# -----------------------
def read_sitk(path: str) -> sitk.Image:
    return sitk.ReadImage(path)


def arr_from_sitk(img: sitk.Image, dtype=np.float32) -> np.ndarray:
    a = sitk.GetArrayFromImage(img)  # (Z,Y,X) == (D,H,W)
    a = np.asarray(a, dtype=dtype)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def sitk_from_arr_like(arr_zyx: np.ndarray, ref: sitk.Image, dtype=sitk.sitkFloat32, allow_size_mismatch: bool = False) -> sitk.Image:
    out = sitk.GetImageFromArray(arr_zyx.astype(np.float32))  # arr Z,Y,X; SITK size X,Y,Z

    if out.GetSize() == ref.GetSize():
        out.CopyInformation(ref)
        return sitk.Cast(out, dtype)

    if not allow_size_mismatch:
        raise RuntimeError(f"Size mismatch: ref={ref.GetSize()} vs out={out.GetSize()} (enable allow_size_mismatch)")

    out.SetDirection(ref.GetDirection())
    out.SetOrigin(ref.GetOrigin())

    ref_size = ref.GetSize()
    out_size = out.GetSize()
    ref_sp = ref.GetSpacing()

    sx = ref_sp[0] * (ref_size[0] / max(1, out_size[0]))
    sy = ref_sp[1] * (ref_size[1] / max(1, out_size[1]))
    sz = ref_sp[2] * (ref_size[2] / max(1, out_size[2]))
    out.SetSpacing((float(sx), float(sy), float(sz)))

    return sitk.Cast(out, dtype)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn = float(x.min())
    mx = float(x.max())
    return (x - mn) / (mx - mn + 1e-8)


# -----------------------
# Model (encoder-only seg head) with activations/grads
# -----------------------
class SwinViTEncoderSegForCAM(nn.Module):
    """
    x -> swinViT -> feats -> select feats[feat_index] -> 1x1 head -> logits_low
    Stores selected feature activations & retains grad for CAM.
    """
    def __init__(self, cfg: dict, feat_index: int = -1):
        super().__init__()
        self.normalize = True
        self.feat_index = int(feat_index)

        fs = int(cfg["feature_size"])
        self._expected_c = _expected_channels(fs, max_pow=6)

        self.strict_layout = bool(cfg.get("strict_swinvit_layout", True))
        self.debug_layout = bool(cfg.get("debug_swinvit_layout", False))
        self._checked = False

        self.backbone = build_swinunetr_backbone(
            img_size=tuple(cfg["img_size"]),
            in_channels=1,
            out_channels=2,
            feature_size=fs,
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
        self._acts = None

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

        idx = self.feat_index
        if not (-len(feats) <= idx < len(feats)):
            raise RuntimeError(f"feat_index={idx} out of range for {len(feats)} features")

        f = feats[idx]  # (B,C,d,h,w)
        if f.requires_grad:
            f.retain_grad()
        self._acts = f
        return self.head(f)  # (B,1,d,h,w)


def compute_gradcam_3d(acts: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
    # compute CAM in FP32 for stability
    a = acts.float()
    g = grads.float()
    w = g.mean(dim=(2, 3, 4), keepdim=True)      # (B,C,1,1,1)
    cam = (w * a).sum(dim=1, keepdim=True)       # (B,1,d,h,w)
    return F.relu(cam)


# -----------------------
# Plotting with GT ROI contours
# -----------------------
def overlay_and_save(ct2d: np.ndarray, cam2d: np.ndarray,
                     union2d: np.ndarray | None,
                     primary2d: np.ndarray | None,
                     nodal2d: np.ndarray | None,
                     out_png: str, title: str):
    plt.figure(figsize=(6, 6))
    plt.imshow(ct2d, cmap="gray", vmin=0.0, vmax=1.0)
    plt.imshow(cam2d, cmap="jet", alpha=0.40, vmin=0.0, vmax=1.0)

    if union2d is not None:
        try:
            plt.contour(union2d.astype(float), levels=[0.5], colors=["lime"], linewidths=1.2)
        except Exception:
            pass
    if primary2d is not None:
        try:
            plt.contour(primary2d.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.0)
        except Exception:
            pass
    if nodal2d is not None:
        try:
            plt.contour(nodal2d.astype(float), levels=[0.5], colors=["magenta"], linewidths=1.0)
        except Exception:
            pass

    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def best_indices(cam: np.ndarray, union: np.ndarray | None, mode: str = "cam_roi") -> tuple[int, int, int]:
    if union is None or float(union.sum()) <= 0:
        mode = "cam"
    if mode == "cam":
        z = int(np.argmax(cam.sum(axis=(1, 2))))
        y = int(np.argmax(cam.sum(axis=(0, 2))))
        x = int(np.argmax(cam.sum(axis=(0, 1))))
        return z, y, x
    if mode == "roi":
        z = int(np.argmax(union.sum(axis=(1, 2))))
        y = int(np.argmax(union.sum(axis=(0, 2))))
        x = int(np.argmax(union.sum(axis=(0, 1))))
        return z, y, x
    w = cam * union
    z = int(np.argmax(w.sum(axis=(1, 2))))
    y = int(np.argmax(w.sum(axis=(0, 2))))
    x = int(np.argmax(w.sum(axis=(0, 1))))
    return z, y, x


# -----------------------
# Single-case Grad-CAM
# -----------------------
def gradcam_one_case(
    *,
    pid: str,
    ct_sitk: sitk.Image,
    ct: np.ndarray,
    target_mask: np.ndarray,
    primary: np.ndarray | None,
    nodal: np.ndarray | None,
    model: SwinViTEncoderSegForCAM,
    device: torch.device,
    autocast_ctx,
    cam_target: str,
    pred_thr: float,
    slice_mode: str,
    out_case_dir: str,
    save_png: bool,
    save_low: bool,
):
    os.makedirs(out_case_dir, exist_ok=True)

    x = torch.from_numpy(ct).unsqueeze(0).unsqueeze(0).to(device)             # (1,1,D,H,W)
    y = torch.from_numpy(target_mask).unsqueeze(0).unsqueeze(0).to(device)    # (1,1,D,H,W)

    model.zero_grad(set_to_none=True)

    with torch.enable_grad():
        with autocast_ctx():
            logits_low = model(x)   # (1,1,d,h,w) possibly fp16 under amp
        logits_f = logits_low.float()

        # IMPORTANT: match latest pretrain downsampling behavior (preserve positives)
        y_low = F.adaptive_max_pool3d(y.float(), output_size=logits_f.shape[2:]).clamp(0, 1)

        if cam_target == "gt":
            if float(y_low.sum().item()) > 0:
                score = (logits_f * y_low).sum() / (y_low.sum() + 1e-6)
            else:
                score = logits_f.max()
        elif cam_target == "pred":
            p = torch.sigmoid(logits_f)
            m_pred = (p >= float(pred_thr)).float()
            score = (logits_f * m_pred).sum() / (m_pred.sum() + 1e-6) if float(m_pred.sum().item()) > 0 else logits_f.max()
        else:
            score = logits_f.max()

        score.backward()

    acts = model._acts
    grads = model._acts.grad
    if grads is None:
        raise RuntimeError(f"[{pid}] No gradients captured for selected feature map.")

    cam_low = compute_gradcam_3d(acts, grads)  # (1,1,d,h,w)
    cam_low_np = normalize01(cam_low.detach().cpu().numpy()[0, 0])

    cam_up = F.interpolate(cam_low.detach(), size=ct.shape, mode="trilinear", align_corners=False)
    cam_up_np = normalize01(cam_up.detach().cpu().numpy()[0, 0])

    # save CAMs
    if save_low:
        cam_low_img = sitk_from_arr_like(cam_low_np, ref=ct_sitk, dtype=sitk.sitkFloat32, allow_size_mismatch=True)
        sitk.WriteImage(cam_low_img, os.path.join(out_case_dir, "cam_low.nii.gz"), useCompression=True)

    cam_up_img = sitk_from_arr_like(cam_up_np, ref=ct_sitk, dtype=sitk.sitkFloat32, allow_size_mismatch=False)
    sitk.WriteImage(cam_up_img, os.path.join(out_case_dir, "cam_up.nii.gz"), useCompression=True)

    # save target mask as gt_union.nii.gz (naming kept for compatibility)
    sitk.WriteImage(
        sitk_from_arr_like(target_mask, ref=ct_sitk, dtype=sitk.sitkUInt8, allow_size_mismatch=False),
        os.path.join(out_case_dir, "gt_union.nii.gz"),
        useCompression=True
    )

    # save extra masks if available
    if primary is not None:
        sitk.WriteImage(sitk_from_arr_like(primary, ref=ct_sitk, dtype=sitk.sitkUInt8, allow_size_mismatch=False),
                        os.path.join(out_case_dir, "gt_primary.nii.gz"), useCompression=True)
    if nodal is not None:
        sitk.WriteImage(sitk_from_arr_like(nodal, ref=ct_sitk, dtype=sitk.sitkUInt8, allow_size_mismatch=False),
                        os.path.join(out_case_dir, "gt_nodal.nii.gz"), useCompression=True)

    # overlays
    if save_png:
        z_best, y_best, x_best = best_indices(cam_up_np, target_mask if float(target_mask.sum()) > 0 else None, mode=slice_mode)

        overlay_and_save(
            ct[z_best], cam_up_np[z_best],
            target_mask[z_best] if target_mask is not None else None,
            primary[z_best] if primary is not None else None,
            nodal[z_best] if nodal is not None else None,
            os.path.join(out_case_dir, "overlay_axial.png"),
            title=f"{pid} axial z={z_best} target={cam_target} feat={model.feat_index}"
        )
        overlay_and_save(
            ct[:, y_best, :], cam_up_np[:, y_best, :],
            target_mask[:, y_best, :] if target_mask is not None else None,
            primary[:, y_best, :] if primary is not None else None,
            nodal[:, y_best, :] if nodal is not None else None,
            os.path.join(out_case_dir, "overlay_coronal.png"),
            title=f"{pid} coronal y={y_best} target={cam_target} feat={model.feat_index}"
        )
        overlay_and_save(
            ct[:, :, x_best], cam_up_np[:, :, x_best],
            target_mask[:, :, x_best] if target_mask is not None else None,
            primary[:, :, x_best] if primary is not None else None,
            nodal[:, :, x_best] if nodal is not None else None,
            os.path.join(out_case_dir, "overlay_sagittal.png"),
            title=f"{pid} sagittal x={x_best} target={cam_target} feat={model.feat_index}"
        )


# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--meta_csv", required=True)
    p.add_argument("--id_col", type=str, default="patient_id")
    p.add_argument("--ct_col", type=str, default="ct_out_path")

    # target mask used for CAM scoring + saved as gt_union.nii.gz (name kept for compatibility)
    p.add_argument("--mask_target_col", type=str, default="mask_union_out_path")

    # extra overlays (optional)
    p.add_argument("--mask_primary_col", type=str, default="mask_primary_out_path")
    p.add_argument("--mask_nodal_col", type=str, default="mask_nodal_out_path")

    # filtering (match training scripts)
    p.add_argument("--keep_bad_status", action="store_true")
    p.add_argument("--keep_unmatched_survival", action="store_true")

    # single patient mode
    p.add_argument("--patient_id", type=str, default="")

    # checkpoint (one checkpoint)
    p.add_argument("--ckpt_path", type=str, default="")

    # feature level for head/CAM
    p.add_argument("--feat_index", type=int, default=-1,
                   help="Which swinViT feature to use for head + CAM. Default -1 (deepest). Use -2 if your pretrain uses feats[-2].")

    # CAM options
    p.add_argument("--cam_target", type=str, default="gt", choices=["gt", "pred", "max"])
    p.add_argument("--pred_thr", type=float, default=0.5)
    p.add_argument("--slice_mode", type=str, default="cam_roi", choices=["cam", "roi", "cam_roi"])

    p.add_argument("--device", type=str, default="")
    p.add_argument("--amp", action="store_true")

    p.add_argument("--out_dir", type=str, default="runs/gradcam_pretrain_all")
    p.add_argument("--save_png", action="store_true")
    p.add_argument("--save_low", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--max_cases", type=int, default=-1)
    p.add_argument("--empty_cache_every", type=int, default=0)

    return p.parse_args()


def main():
    args = parse_args()

    device = set_device_from_arg(args.device)
    autocast_ctx = make_autocast(device, enabled=bool(args.amp))

    # load meta
    meta = pd.read_csv(args.meta_csv, dtype={args.id_col: str})
    meta[args.id_col] = meta[args.id_col].astype(str)

    if (not args.keep_bad_status) and ("status" in meta.columns):
        meta = meta[meta["status"].astype(str).str.lower() == "ok"].copy()
    if (not args.keep_unmatched_survival) and ("survival_matched" in meta.columns):
        sm = meta["survival_matched"]
        if sm.dtype == bool:
            meta = meta[sm].copy()
        else:
            meta = meta[sm.astype(str).str.lower().isin(["true", "1", "t", "yes"])].copy()

    meta = meta.set_index(args.id_col, drop=False)

    # ids
    if args.patient_id.strip():
        ids = [args.patient_id.strip()]
        base_out = os.path.join(args.out_dir, "single")
    else:
        ids = meta[args.id_col].astype(str).tolist()
        base_out = os.path.join(args.out_dir, "all")
    os.makedirs(base_out, exist_ok=True)

    # checkpoint
    if not args.ckpt_path:
        raise ValueError("Provide --ckpt_path")
    ckpt_path = args.ckpt_path
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = dict(ckpt.get("args", {}))
    # prefer explicit saved size if present
    if "img_size_dhw" in ckpt and ckpt["img_size_dhw"] is not None:
        cfg["img_size"] = list(tuple(int(x) for x in ckpt["img_size_dhw"]))
    cfg.setdefault("img_size", [128, 256, 256])
    cfg.setdefault("feature_size", 48)
    cfg.setdefault("depths", [2, 2, 2, 2])
    cfg.setdefault("num_heads", [3, 6, 12, 24])
    cfg.setdefault("drop_rate", 0.10)
    cfg.setdefault("attn_drop_rate", 0.10)
    cfg.setdefault("dropout_path_rate", 0.20)
    cfg.setdefault("use_checkpoint", False)
    cfg.setdefault("strict_swinvit_layout", True)
    cfg.setdefault("debug_swinvit_layout", False)

    model = SwinViTEncoderSegForCAM(cfg, feat_index=int(args.feat_index)).to(device)
    model.eval()

    # load backbone/head weights
    if "model_state" in ckpt:
        model.backbone.load_state_dict(ckpt["model_state"], strict=False)
    elif "state_dict" in ckpt:
        model.backbone.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        raise RuntimeError("Checkpoint has no 'model_state'/'state_dict' for backbone.")

    # warmup Lazy head + load head_state
    warm_pid = None
    for pid in ids:
        if pid not in meta.index:
            continue
        ct_p = str(meta.loc[pid, args.ct_col])
        m_p = str(meta.loc[pid, args.mask_target_col]) if args.mask_target_col in meta.columns else ""
        if os.path.isfile(ct_p) and os.path.isfile(m_p):
            warm_pid = pid
            break
    if warm_pid is None:
        raise RuntimeError("No valid patient found to warmup Lazy head (ct/mask missing).")

    ct_w = arr_from_sitk(read_sitk(str(meta.loc[warm_pid, args.ct_col])), dtype=np.float32)
    x0 = torch.from_numpy(ct_w).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        with autocast_ctx():
            _ = model(x0)

    if "head_state" in ckpt and ckpt["head_state"] is not None:
        model.head.load_state_dict(ckpt["head_state"], strict=False)

    # run
    n_done = 0
    for pid in ids:
        pid = str(pid)
        if pid not in meta.index:
            continue

        case_dir = os.path.join(base_out, pid)
        cam_up_path = os.path.join(case_dir, "cam_up.nii.gz")
        if args.skip_existing and os.path.isfile(cam_up_path):
            continue

        ct_path = str(meta.loc[pid, args.ct_col])
        tgt_path = str(meta.loc[pid, args.mask_target_col])
        if not (os.path.isfile(ct_path) and os.path.isfile(tgt_path)):
            continue

        ct_sitk = read_sitk(ct_path)
        ct = arr_from_sitk(ct_sitk, dtype=np.float32)
        ct = np.clip(ct, 0.0, 1.0)

        target_mask = (arr_from_sitk(read_sitk(tgt_path), dtype=np.float32) > 0.5).astype(np.float32)

        primary = None
        nodal = None
        if args.mask_primary_col in meta.columns:
            pth = str(meta.loc[pid, args.mask_primary_col])
            if os.path.isfile(pth):
                primary = (arr_from_sitk(read_sitk(pth), dtype=np.float32) > 0.5).astype(np.float32)
        if args.mask_nodal_col in meta.columns:
            pth = str(meta.loc[pid, args.mask_nodal_col])
            if os.path.isfile(pth):
                nodal = (arr_from_sitk(read_sitk(pth), dtype=np.float32) > 0.5).astype(np.float32)

        gradcam_one_case(
            pid=pid,
            ct_sitk=ct_sitk,
            ct=ct,
            target_mask=target_mask,
            primary=primary,
            nodal=nodal,
            model=model,
            device=device,
            autocast_ctx=autocast_ctx,
            cam_target=args.cam_target,
            pred_thr=float(args.pred_thr),
            slice_mode=args.slice_mode,
            out_case_dir=case_dir,
            save_png=bool(args.save_png),
            save_low=bool(args.save_low),
        )

        n_done += 1
        if (n_done % 10) == 0:
            print(f"[progress] done={n_done} / total~{len(ids)} (latest pid={pid})")

        if args.empty_cache_every and device.type == "cuda":
            if (n_done % int(args.empty_cache_every)) == 0:
                torch.cuda.empty_cache()

        if args.max_cases > 0 and n_done >= int(args.max_cases):
            break

    print(f"[DONE] generated={n_done} cases -> {base_out}")


if __name__ == "__main__":
    main()
