"""Core utilities for TriFuseSurv2 2.1.5 v2 Grad-CAM generation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from trifusesurv2.schema import IMAGE_HABITATS

SOFTWARE_VERSION = "2.1.5"
TARGET_COMMIT_SHA = "daaaa363020b7e27b93981f62dfa17821489e1ea"
MODEL_CLASS = "ContourAwareHabitatSurvivalModel"


def checkpoint_args_to_dict(ck: dict[str, Any]) -> dict[str, Any]:
    raw = ck.get("args", {})
    if isinstance(raw, dict):
        return dict(raw)
    if hasattr(raw, "__dict__"):
        return dict(vars(raw))
    return {}


def normalized_state_dict(ck: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = ck.get("model_state", ck.get("model_state_dict", ck.get("state_dict", ck)))
    out = {}
    for key, value in state.items():
        k = str(key)
        if k.startswith("module."):
            k = k[len("module.") :]
        out[k] = value
    return out


def assert_v213_v2_checkpoint(
    ck: dict[str, Any],
    *,
    checkpoint_path: str | Path = "",
    require_commit: bool = False,
) -> None:
    """Fail unless a checkpoint looks like the 2.1.5 habitat-aligned v2 model."""

    args = checkpoint_args_to_dict(ck)
    state = normalized_state_dict(ck)
    keys = set(state.keys())

    missing_prefixes = [
        prefix
        for prefix in (
            "backbone.",
            "habitat_model.",
            "habitat_model.habitat_fusers.",
            "habitat_model.sequence_encoder.",
            "habitat_model.survival_heads.",
        )
        if not any(k.startswith(prefix) for k in keys)
    ]
    if missing_prefixes:
        raise RuntimeError(
            f"{checkpoint_path}: expected a {MODEL_CLASS} v2 checkpoint; missing key prefixes {missing_prefixes}"
        )

    old_prefixes = ("gate_mlp.", "fuse_projs.", "img_post_mlp.", "surv_heads.", "img_backbone.")
    old_hits = [p for p in old_prefixes if any(k.startswith(p) for k in keys)]
    if old_hits:
        raise RuntimeError(
            f"{checkpoint_path}: checkpoint has v1/MoE key prefixes {old_hits}; use the v1 Grad-CAM script instead."
        )

    model_version = str(args.get("model_version", "")).strip().lower()
    if model_version != "v2":
        raise RuntimeError(
            f"{checkpoint_path}: checkpoint args model_version={model_version!r}; expected 'v2'. "
            "A repository version of 2.1.5 is not sufficient if the run used --model_version v1."
        )

    commit_sha = str(ck.get("commit_sha", args.get("commit_sha", args.get("git_commit", "")))).strip()
    if require_commit and commit_sha != TARGET_COMMIT_SHA:
        raise RuntimeError(
            f"{checkpoint_path}: checkpoint commit_sha={commit_sha}, expected {TARGET_COMMIT_SHA} for this Grad-CAM compatibility profile."
        )


assert_v211_v2_checkpoint = assert_v213_v2_checkpoint


def horizon_bin_index(num_time_bins: int, time_bin_width_days: float, horizon_days: float) -> int:
    k = int(math.ceil(float(horizon_days) / float(time_bin_width_days)) - 1)
    return max(0, min(k, int(num_time_bins) - 1))


def select_survival_target(
    model,
    logits: dict[str, torch.Tensor],
    *,
    endpoint: str,
    horizon_days: float,
    target_type: str = "cumulative_risk",
) -> torch.Tensor:
    endpoint = str(endpoint).upper()
    ep_logits = logits[endpoint].float()
    if str(target_type) == "hazard_logit":
        k = horizon_bin_index(model.num_time_bins, model.time_bin_width_days, horizon_days)
        return ep_logits[:, k].sum()
    return model.hazards_to_risk(ep_logits, float(horizon_days)).sum()


def risk_vector(model, logits: dict[str, torch.Tensor], *, endpoint: str, horizon_days: float) -> torch.Tensor:
    return model.hazards_to_risk(logits[str(endpoint).upper()].float(), float(horizon_days))


def signed_gradcam_3d(activation: torch.Tensor, gradient: torch.Tensor, output_shape: tuple[int, int, int]) -> torch.Tensor:
    """Signed 3D Grad-CAM, upsampled to ``output_shape``."""

    if activation.dim() != 5 or gradient.dim() != 5:
        raise ValueError(f"activation/gradient must be 5D [B,C,D,H,W], got {activation.shape} / {gradient.shape}")
    weights = gradient.float().mean(dim=(2, 3, 4), keepdim=True)
    cam = (weights * activation.float()).sum(dim=1, keepdim=True)
    return F.interpolate(cam, size=tuple(int(x) for x in output_shape), mode="trilinear", align_corners=False)


def cam_from_features(features: list[torch.Tensor], output_shape: tuple[int, int, int]) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    cams = []
    shapes = []
    for feat in features:
        grad = feat.grad
        if grad is None:
            continue
        cam = signed_gradcam_3d(feat, grad, output_shape)
        cams.append(cam)
        shapes.append("x".join(str(int(v)) for v in feat.shape))
    if not cams:
        raise RuntimeError("No CAM features had gradients. Ensure return_cam_features=True and gradients are enabled.")
    mean_cam = torch.stack(cams, dim=0).mean(dim=0)
    scale_np = [c.detach().cpu().numpy()[0, 0].astype(np.float32) for c in cams]
    return mean_cam.detach().cpu().numpy()[0, 0].astype(np.float32), scale_np, shapes


def isolate_image_habitat_gradient(image_tokens: torch.Tensor, habitat_idx: int) -> torch.Tensor:
    iso = torch.zeros(
        image_tokens.shape[0],
        image_tokens.shape[1],
        1,
        device=image_tokens.device,
        dtype=image_tokens.dtype,
    )
    iso[:, int(habitat_idx), :] = 1.0
    return image_tokens.detach() * (1.0 - iso) + image_tokens * iso


def split_signed_cam(cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cam = np.asarray(cam, dtype=np.float32)
    return np.maximum(cam, 0.0).astype(np.float32), np.maximum(-cam, 0.0).astype(np.float32)


def normalize_component(arr: np.ndarray, *, support: Optional[np.ndarray] = None, eps: float = 1e-8) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if support is not None:
        supp = np.asarray(support, dtype=np.float32) > 0.05
        arr = arr * supp.astype(np.float32)
        valid = supp & (arr > 0)
    else:
        valid = arr > 0
    out = np.zeros_like(arr, dtype=np.float32)
    if not bool(valid.any()):
        return out
    vals = arr[valid]
    lo = float(vals.min())
    hi = float(vals.max())
    if hi <= lo + eps:
        out[valid] = 1.0
    else:
        out[valid] = (vals - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def supports_from_backbone_aux(aux: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    pt = aux["pt_used"].detach().clamp(0, 1)
    ln = aux["ln_used"].detach().clamp(0, 1)
    pt_shell = aux["pt_shell"].detach().clamp(0, 1)
    ln_shell = aux["ln_shell"].detach().clamp(0, 1)
    body = aux.get("body", torch.ones_like(pt)).detach().clamp(0, 1)
    habitat_union = (pt + pt_shell + ln + ln_shell).clamp(0, 1)
    peri_overlap = torch.minimum(pt_shell, ln_shell).clamp(0, 1)
    pt_peri_disjoint = (pt_shell - pt - ln - ln_shell).clamp(0, 1)
    ln_peri_disjoint = (ln_shell - ln - pt - pt_shell).clamp(0, 1)
    habitat_union_disjoint = (pt + ln + pt_peri_disjoint + ln_peri_disjoint).clamp(0, 1)
    return {
        "full_volume": torch.ones_like(body),
        "body": body,
        "pt_intra": pt,
        "pt_peri": pt_shell,
        "pt_peri_disjoint": pt_peri_disjoint,
        "ln_intra": ln,
        "ln_peri": ln_shell,
        "peri_overlap": peri_overlap,
        "ln_peri_disjoint": ln_peri_disjoint,
        "pt_ln_union": (pt + ln).clamp(0, 1),
        "habitat_union": habitat_union,
        "habitat_union_disjoint": habitat_union_disjoint,
        "off_habitat_body": (body - habitat_union).clamp(0, 1),
        "off_habitat_body_disjoint": (body - habitat_union_disjoint).clamp(0, 1),
    }


def cam_mass_summary(cam_pos: np.ndarray, supports: dict[str, np.ndarray], *, denominator: str = "body") -> dict[str, float]:
    den_support = np.asarray(supports[denominator], dtype=np.float32)
    den = float((cam_pos * den_support).sum())
    out: dict[str, float] = {}
    for name, support in supports.items():
        support_np = np.asarray(support, dtype=np.float32)
        mass = float((cam_pos * support_np).sum())
        out[f"cam_mass_{name}"] = mass
        out[f"cam_fraction_{name}"] = mass / max(den, 1e-8)
    return out


def save_nifti(cam: np.ndarray, out_path: Path, ref_path: Optional[Path] = None, *, clip: bool = False) -> dict[str, Any]:
    import SimpleITK as sitk

    arr = np.nan_to_num(np.asarray(cam, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if clip:
        arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    out_img = sitk.GetImageFromArray(arr)
    copied = False
    ref_size = None
    if ref_path is not None and Path(ref_path).is_file():
        ref_img = sitk.ReadImage(str(ref_path))
        ref_size = tuple(int(x) for x in ref_img.GetSize())
        if tuple(out_img.GetSize()) == tuple(ref_img.GetSize()):
            out_img.CopyInformation(ref_img)
            copied = True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_img, str(out_path), True)
    return {
        "geometry_copied": bool(copied),
        "cam_shape_zyx": "x".join(str(int(x)) for x in arr.shape),
        "cam_size_xyz": "x".join(str(int(x)) for x in out_img.GetSize()),
        "reference_size_xyz": "" if ref_size is None else "x".join(str(int(x)) for x in ref_size),
    }


def _normalize01(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo + eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def make_overlay_png(
    ct: np.ndarray,
    cam: np.ndarray,
    out_png: Path,
    *,
    title: str,
    mask_pt: Optional[np.ndarray] = None,
    mask_ln: Optional[np.ndarray] = None,
    body_mask: Optional[np.ndarray] = None,
    max_slices: int = 12,
    cmap: str = "inferno",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ct = np.asarray(ct, dtype=np.float32)
    cam = np.asarray(cam, dtype=np.float32)
    ct_disp = _normalize01(ct)
    body = np.ones_like(cam, dtype=bool) if body_mask is None else (np.asarray(body_mask, dtype=np.float32) > 0.05)
    scores = (cam * body.astype(np.float32)).reshape(cam.shape[0], -1).sum(axis=1)
    if mask_pt is not None:
        scores += 0.5 * (np.asarray(mask_pt) > 0.5).reshape(cam.shape[0], -1).sum(axis=1)
    if mask_ln is not None:
        scores += 0.5 * (np.asarray(mask_ln) > 0.5).reshape(cam.shape[0], -1).sum(axis=1)
    n = max(1, min(int(max_slices), int(cam.shape[0])))
    if float(scores.max()) > 0:
        idx = np.sort(np.argsort(scores)[-n:])
    else:
        idx = np.linspace(0, cam.shape[0] - 1, num=n, dtype=int)
    cols = 4
    rows = int((len(idx) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.8))
    axes = np.asarray(axes).reshape(-1)
    positives = cam[(cam > 0) & body]
    threshold = 0.25 if not positives.size else max(0.25, float(np.percentile(positives, 85.0)))
    for ax_i, z in enumerate(idx):
        ax = axes[ax_i]
        ax.imshow(ct_disp[z], cmap="gray", interpolation="nearest")
        overlay = np.ma.masked_where((cam[z] <= threshold) | (~body[z]), cam[z])
        ax.imshow(overlay, cmap=cmap, vmin=0.0, vmax=1.0, alpha=0.45, interpolation="nearest")
        if mask_pt is not None and float((np.asarray(mask_pt)[z] > 0.5).sum()) > 0:
            ax.contour(np.asarray(mask_pt)[z] > 0.5, levels=[0.5], colors=["#00D1FF"], linewidths=0.8)
        if mask_ln is not None and float((np.asarray(mask_ln)[z] > 0.5).sum()) > 0:
            ax.contour(np.asarray(mask_ln)[z] > 0.5, levels=[0.5], colors=["#FFD23F"], linewidths=0.8)
        ax.set_title(f"z={int(z)}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(idx) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def append_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    write_header = not path.is_file()
    if not write_header:
        with path.open("r", newline="") as f:
            existing = next(csv.reader(f), [])
        if existing != fieldnames:
            raise RuntimeError(f"Existing manifest schema differs for {path}. Use a new output directory.")
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def pid_slug(pid: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(pid))


def image_habitat_names(model: Any = None) -> tuple[str, ...]:
    if model is not None:
        habitat_model = getattr(model, "habitat_model", model)
        names = getattr(habitat_model, "image_habitats", None)
        if names:
            return tuple(str(x) for x in names)
    return tuple(IMAGE_HABITATS)
