"""ROI focus metrics for contour-aware survival training."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


ROI_FOCUS_METRIC_NAMES = (
    "gt_present_frac",
    "prob_mass_inside_gt",
    "prob_gt_mean",
    "support_precision",
    "support_recall",
    "support_dice",
    "support_iou",
    "support_empty_when_gt_present",
    "support_volume_ratio",
)


def _resize_like(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if mask.shape == ref.shape:
        return mask
    if mask.ndim != 5 or ref.ndim != 5:
        raise ValueError(f"ROI focus tensors must be 5D, got mask={tuple(mask.shape)} ref={tuple(ref.shape)}")
    if mask.shape[:2] != ref.shape[:2]:
        raise ValueError(
            "ROI focus tensors must have matching batch/channel dimensions, "
            f"got mask={tuple(mask.shape)} ref={tuple(ref.shape)}"
        )
    return F.interpolate(mask.float(), size=tuple(int(x) for x in ref.shape[2:]), mode="nearest")


def _present_mean(values: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    if bool(present.any().item()):
        return values[present].mean()
    return values.new_tensor(float("nan"))


def roi_focus_metrics(
    *,
    pred_prob: torch.Tensor,
    target_mask: torch.Tensor,
    used_support: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Measure whether predicted/support ROI masks focus on the target ROI.

    Metrics are batch means. Overlap-style metrics are averaged only over
    samples with a non-empty ground-truth ROI; ``gt_present_frac`` is averaged
    over the full batch.
    """

    prob = pred_prob.float().clamp(0, 1)
    target = _resize_like(target_mask, prob).float().clamp(0, 1)
    support_src = pred_prob if used_support is None else _resize_like(used_support, prob)
    support = (support_src.float() >= float(threshold)).float()
    target_bin = (target >= float(threshold)).float()

    prob_f = prob.flatten(1)
    support_f = support.flatten(1)
    target_f = target_bin.flatten(1)

    target_sum = target_f.sum(dim=1)
    support_sum = support_f.sum(dim=1)
    present = target_sum > float(eps)

    tp = (support_f * target_f).sum(dim=1)
    union = (support_f + target_f - support_f * target_f).sum(dim=1)
    prob_inside = (prob_f * target_f).sum(dim=1)
    prob_mass = prob_f.sum(dim=1)

    support_precision = tp / support_sum.clamp_min(float(eps))
    support_recall = tp / target_sum.clamp_min(float(eps))
    support_dice = (2.0 * tp) / (support_sum + target_sum).clamp_min(float(eps))
    support_iou = tp / union.clamp_min(float(eps))
    prob_mass_inside_gt = prob_inside / prob_mass.clamp_min(float(eps))
    prob_gt_mean = prob_inside / target_sum.clamp_min(float(eps))
    support_empty = (support_sum <= float(eps)).to(prob.dtype)
    support_volume_ratio = support_sum / target_sum.clamp_min(float(eps))

    return {
        "gt_present_frac": present.float().mean(),
        "prob_mass_inside_gt": _present_mean(prob_mass_inside_gt, present),
        "prob_gt_mean": _present_mean(prob_gt_mean, present),
        "support_precision": _present_mean(support_precision, present),
        "support_recall": _present_mean(support_recall, present),
        "support_dice": _present_mean(support_dice, present),
        "support_iou": _present_mean(support_iou, present),
        "support_empty_when_gt_present": _present_mean(support_empty, present),
        "support_volume_ratio": _present_mean(support_volume_ratio, present),
    }


def finite_scalar(value: torch.Tensor) -> Optional[float]:
    """Detach a scalar tensor if finite; otherwise return None."""

    scalar = float(value.detach().float().cpu().item())
    if math.isfinite(scalar):
        return scalar
    return None
