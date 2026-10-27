#!/usr/bin/env python3
"""Contour-aware SwinUNETR ROI token backbone.

Recommended end-to-end survival image path:
- CT-only shared SwinUNETR encoder
- PT/LN localization heads on a configurable higher-resolution Swin feature
- ROI tokenization from explicit PT/LN support masks
- optional teacher forcing with GT PT/LN masks during training

Tokens (B,6,Dtok):
  0: GLOBAL
  1: PT_INTRA
  2: PT_PERI
  3: LN_INTRA
  4: LN_PERI
  5: SHAPE_SPATIAL
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from trifusesurv2.models.swinunetr_backbone_utils import (
    _expected_channels,
    build_swinunetr_backbone,
    convert_swinvit_feats_to_channel_first,
    swinvit_features,
)
from trifusesurv2.schema import PHYSICAL_GEOMETRY_FEATURES, REGION_STATUS_CODES


def masked_mean(feat: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    w = mask.clamp(0, 1)
    denom = w.sum(dim=(2, 3, 4)).clamp_min(eps)
    num = (feat * w).sum(dim=(2, 3, 4))
    return num / denom


def ct_stats_in_mask(
    ct: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
    *,
    include_volume: bool = True,
) -> torch.Tensor:
    w = mask.clamp(0, 1)
    denom = w.sum(dim=(2, 3, 4)).clamp_min(eps)
    mu = (ct * w).sum(dim=(2, 3, 4)) / denom
    second = ((ct * ct) * w).sum(dim=(2, 3, 4)) / denom
    var = (second - mu * mu).clamp_min(0.0)
    sd = torch.sqrt(var + 1e-8)
    parts = [mu, sd]
    if bool(include_volume):
        parts.append(w.mean(dim=(2, 3, 4)))
    return torch.cat(parts, dim=1)


def ct_stats_global(ct: torch.Tensor, body: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    if body is None:
        mu = ct.mean(dim=(2, 3, 4))
        sd = ct.std(dim=(2, 3, 4)).clamp_min(1e-8)
        frac = ct.new_ones((ct.shape[0], 1))
        return torch.cat([mu, sd, frac], dim=1)

    w = body.clamp(0, 1)
    denom = w.sum(dim=(2, 3, 4)).clamp_min(eps)
    mu = (ct * w).sum(dim=(2, 3, 4)) / denom
    second = ((ct * ct) * w).sum(dim=(2, 3, 4)) / denom
    var = (second - mu * mu).clamp_min(0.0)
    sd = torch.sqrt(var + 1e-8)
    frac = w.mean(dim=(2, 3, 4))
    return torch.cat([mu, sd, frac], dim=1)


def mask_centroid_zyx(mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Weighted centroid in normalized z/y/x coordinates without materializing grids."""

    w = mask.clamp(0, 1)
    B, _, D, H, W = w.shape
    denom = w.sum(dim=(2, 3, 4)).squeeze(1).clamp_min(eps)
    z_coord = torch.linspace(-1.0, 1.0, D, device=w.device, dtype=w.dtype)
    y_coord = torch.linspace(-1.0, 1.0, H, device=w.device, dtype=w.dtype)
    x_coord = torch.linspace(-1.0, 1.0, W, device=w.device, dtype=w.dtype)
    z_mass = w.sum(dim=(3, 4)).squeeze(1)
    y_mass = w.sum(dim=(2, 4)).squeeze(1)
    x_mass = w.sum(dim=(2, 3)).squeeze(1)
    cz = (z_mass * z_coord.view(1, D)).sum(dim=1) / denom
    cy = (y_mass * y_coord.view(1, H)).sum(dim=1) / denom
    cx = (x_mass * x_coord.view(1, W)).sum(dim=1) / denom
    return torch.stack([cz, cy, cx], dim=1).view(B, 3)


def shape_spatial_features(
    pt_mask: torch.Tensor,
    pt_shell: torch.Tensor,
    ln_mask: torch.Tensor,
    ln_shell: torch.Tensor,
    body: Optional[torch.Tensor],
    pt_present: torch.Tensor,
    ln_present: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compact ROI burden and PT-LN relation features for survival prediction."""

    pt_frac = pt_mask.clamp(0, 1).mean(dim=(2, 3, 4))
    ln_frac = ln_mask.clamp(0, 1).mean(dim=(2, 3, 4))
    pt_shell_frac = pt_shell.clamp(0, 1).mean(dim=(2, 3, 4))
    ln_shell_frac = ln_shell.clamp(0, 1).mean(dim=(2, 3, 4))
    if body is None:
        body_frac = pt_frac.new_ones(pt_frac.shape)
    else:
        body_frac = body.clamp(0, 1).mean(dim=(2, 3, 4))

    pt_cent = mask_centroid_zyx(pt_mask, eps=eps)
    ln_cent = mask_centroid_zyx(ln_mask, eps=eps)
    both_present = (pt_present & ln_present).to(pt_mask.dtype).view(-1, 1)
    delta = (ln_cent - pt_cent) * both_present
    distance = torch.linalg.vector_norm(delta, ord=2, dim=1, keepdim=True)
    ratio_ln_pt = torch.log1p(ln_frac / pt_frac.clamp_min(eps))

    presence = torch.cat(
        [
            pt_present.to(pt_mask.dtype).view(-1, 1),
            ln_present.to(pt_mask.dtype).view(-1, 1),
            both_present,
        ],
        dim=1,
    )
    return torch.cat(
        [
            pt_frac,
            ln_frac,
            pt_shell_frac,
            ln_shell_frac,
            body_frac,
            ratio_ln_pt,
            pt_cent,
            ln_cent,
            delta,
            distance,
            presence,
        ],
        dim=1,
    )


class AttnPool3D(nn.Module):
    def __init__(
        self,
        mask_bias: float = 2.0,
        temperature: float = 1.0,
        mask_mode: str = "masked",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.attn = nn.LazyConv3d(1, kernel_size=1, bias=False)
        self.mask_bias = float(mask_bias)
        self.temperature = float(temperature)
        self.mask_mode = str(mask_mode).strip().lower()
        self.eps = float(eps)
        if self.mask_mode not in {"masked", "weighted", "bias"}:
            raise ValueError(f"Unknown attention pooling mask_mode: {mask_mode}")

    def forward(self, feat: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.attn(feat).flatten(2)
        support = None
        if mask is not None:
            mask_flat = mask.clamp(0, 1).flatten(2).to(dtype=logits.dtype, device=logits.device)
            if self.mask_mode == "bias":
                logits = logits + self.mask_bias * mask_flat
            else:
                support = mask_flat > self.eps
                has_support = support.any(dim=-1, keepdim=True)
                if self.mask_mode == "weighted":
                    logits = logits + torch.log(mask_flat.clamp_min(self.eps))
                logits = logits.masked_fill(~support, torch.finfo(logits.dtype).min)
                # Empty ROIs must not become global pooling. Keep their softmax
                # finite, then explicitly remove every unsupported weight below.
                logits = torch.where(has_support, logits, torch.zeros_like(logits))
        w = torch.softmax(logits / self.temperature, dim=-1)
        if support is not None:
            w = w * support.to(w.dtype)
        feat_flat = feat.flatten(2)
        return (feat_flat * w).sum(dim=-1)


def interp_mask(mask: torch.Tensor, size: Tuple[int, int, int], mode: str) -> torch.Tensor:
    """Resize a mask without discarding small ROIs during spatial reduction.

    Point sampling, including trilinear interpolation, can miss an entire ROI
    between feature-grid sample locations. Max-pool every reduced axis so a
    feature cell retains support whenever its native region contains ROI.
    The requested interpolation mode applies to any remaining upsampling.
    """
    if mode not in {"nearest", "trilinear"}:
        raise ValueError(f"Unknown mask_interp mode: {mode}")
    size = tuple(int(x) for x in size)
    reduced_size = tuple(min(int(src), dst) for src, dst in zip(mask.shape[2:], size))
    if reduced_size != tuple(mask.shape[2:]):
        mask = F.adaptive_max_pool3d(mask, output_size=reduced_size)
    if tuple(mask.shape[2:]) == size:
        return mask
    if mode == "nearest":
        return F.interpolate(mask, size=size, mode="nearest")
    return F.interpolate(mask, size=size, mode="trilinear", align_corners=False)


def binary_close(m: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return m
    k = 2 * r + 1
    d = (F.max_pool3d(m, kernel_size=k, stride=1, padding=r) > 0).float()
    e = 1.0 - (F.max_pool3d((1.0 - d).clamp(0, 1), kernel_size=k, stride=1, padding=r) > 0).float()
    return e


def soft_shell_around_mask(
    mask: torch.Tensor,
    *,
    radius: int,
    thickness_mm: float = 0.0,
    voxel_spacing_dhw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Binary Euclidean support for the explicit, unverified legacy path.

    Verified runs consume persisted compartments instead. Spacing alone does
    not qualify acquisition geometry. Voxel-radius mode uses unit spacing;
    physical mode requires spacing and never silently falls back to voxels.
    """
    from trifusesurv2.preprocessing.compartments import physical_shell
    import numpy as np

    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError("Shell input must be [B,1,D,H,W]")
    physical = float(thickness_mm) > 0
    width = float(thickness_mm) if physical else float(radius)
    if width <= 0:
        return torch.zeros_like(mask)
    if physical and voxel_spacing_dhw is None:
        raise ValueError("Physical shells require voxel spacing; use persisted verified regions.")
    if physical:
        spacing = torch.as_tensor(voxel_spacing_dhw).detach().cpu().numpy()
        spacing = np.atleast_2d(spacing)
        if spacing.shape == (1, 3):
            spacing = np.repeat(spacing, mask.shape[0], axis=0)
        if spacing.shape != (mask.shape[0], 3):
            raise ValueError("Voxel spacing must be [B,3] in D,H,W order")
    else:
        spacing = np.ones((mask.shape[0], 3))
    masks = (mask.detach().float().cpu().numpy()[:, 0] > 0.5)
    shells = np.stack([
        physical_shell(masks[i], spacing_dhw=spacing[i], width_mm=width)
        for i in range(mask.shape[0])
    ])[:, None]
    return torch.as_tensor(shells, dtype=mask.dtype, device=mask.device)


class ContourAwareROITokenBackbone(nn.Module):
    def __init__(
        self,
        *,
        img_size: Tuple[int, int, int],
        feature_size: int = 48,
        depths: Tuple[int, int, int, int] = (2, 2, 2, 2),
        num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24),
        drop_rate: float = 0.10,
        attn_drop_rate: float = 0.10,
        dropout_path_rate: float = 0.20,
        normalize: bool = True,
        use_checkpoint: bool = False,
        token_dim: int = 512,
        token_mlp_dropout: float = 0.30,
        token_mlp_hidden_dim: int = 0,
        attn_mask_bias: float = 2.0,
        attn_pool_mask_mode: str = "masked",
        use_multiscale: bool = True,
        mask_interp: str = "nearest",
        loc_feature_from_end: int = 4,
        roi_support_threshold: float = 0.5,
        roi_support_fallback_threshold: float = 0.05,
        roi_support_fallback_relmax: float = 0.5,
        include_roi_volume: bool = True,
        include_shell_volume: bool = True,
        min_roi_frac: float = 1e-5,
        min_roi_voxels_deep: int = 8,
        token_dropout: float = 0.05,
        pt_shell_radius: int = 3,
        ln_shell_radius: int = 3,
        pt_shell_thickness_mm: float = 10.0,
        ln_shell_thickness_mm: float = 10.0,
        shell_body_from_ct: bool = True,
        body_ct_thr: Union[str, float] = "auto",
        body_ct_thr_hu: float = -500.0,
        body_close_r: int = 2,
        body_max_frac: float = 0.995,
        strict_swinvit_layout: bool = True,
        debug_swinvit_layout: bool = False,
        force_presence_from_raw_masks: bool = False,
        raw_mask_threshold: float = 0.5,
        fallback_peri_to_intra: bool = False,
        require_verified_regions: bool = False,
        sync_sanitize_checks: bool = False,
    ):
        super().__init__()
        self.normalize = bool(normalize)
        self.use_multiscale = bool(use_multiscale)
        self.mask_interp = str(mask_interp)
        self.loc_feature_from_end = int(max(1, loc_feature_from_end))
        self.roi_support_threshold = float(roi_support_threshold)
        self.roi_support_fallback_threshold = float(roi_support_fallback_threshold)
        self.roi_support_fallback_relmax = float(roi_support_fallback_relmax)
        self.include_roi_volume = bool(include_roi_volume)
        self.include_shell_volume = bool(include_shell_volume)
        self.min_roi_frac = float(min_roi_frac)
        self.min_roi_voxels_deep = int(max(min_roi_voxels_deep, 0))
        self.token_dropout = float(max(token_dropout, 0.0))
        self.pt_shell_radius = int(pt_shell_radius)
        self.ln_shell_radius = int(ln_shell_radius)
        self.pt_shell_thickness_mm = float(max(0.0, pt_shell_thickness_mm))
        self.ln_shell_thickness_mm = float(max(0.0, ln_shell_thickness_mm))
        self.shell_body_from_ct = bool(shell_body_from_ct)
        self.body_ct_thr = body_ct_thr
        self.body_ct_thr_hu = float(body_ct_thr_hu)
        self.body_close_r = int(body_close_r)
        self.body_max_frac = float(body_max_frac)
        self.strict_swinvit_layout = bool(strict_swinvit_layout)
        self.debug_swinvit_layout = bool(debug_swinvit_layout)
        self._checked_layout = False
        self._expected_c = _expected_channels(int(feature_size), max_pow=6)
        self.force_presence_from_raw_masks = bool(force_presence_from_raw_masks)
        self.raw_mask_threshold = float(raw_mask_threshold)
        if fallback_peri_to_intra:
            raise ValueError("Peritumoral-to-interior substitution is unsupported: an empty shell must remain empty.")
        self.require_verified_regions = bool(require_verified_regions)
        self.sync_sanitize_checks = bool(sync_sanitize_checks)
        self._warned_sanitized = set()

        self.backbone_shared = build_swinunetr_backbone(
            img_size=tuple(img_size),
            in_channels=1,
            out_channels=2,
            feature_size=int(feature_size),
            depths=tuple(depths),
            num_heads=tuple(num_heads),
            drop_rate=float(drop_rate),
            attn_drop_rate=float(attn_drop_rate),
            dropout_path_rate=float(dropout_path_rate),
            normalize=self.normalize,
            use_checkpoint=bool(use_checkpoint),
            spatial_dims=3,
        )

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.attn_pool = AttnPool3D(mask_bias=float(attn_mask_bias), mask_mode=str(attn_pool_mask_mode))
        self.loc_pt_head = nn.LazyConv3d(1, kernel_size=1, bias=True)
        self.loc_ln_head = nn.LazyConv3d(1, kernel_size=1, bias=True)
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
        self.max_tokens = 6
        self.token_dim = int(token_dim)

        token_mlp_hidden_dim = int(token_mlp_hidden_dim)
        if token_mlp_hidden_dim > 0:
            self.token_mlp = nn.ModuleList([
                nn.Sequential(
                    nn.LazyLinear(token_mlp_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(token_mlp_dropout)),
                    nn.Linear(token_mlp_hidden_dim, self.token_dim),
                    nn.GELU(),
                    nn.Dropout(float(token_mlp_dropout)),
                    nn.LayerNorm(self.token_dim),
                ) for _ in range(self.max_tokens)
            ])
        else:
            self.token_mlp = nn.ModuleList([
                nn.Sequential(
                    nn.LazyLinear(self.token_dim),
                    nn.GELU(),
                    nn.Dropout(float(token_mlp_dropout)),
                    nn.LayerNorm(self.token_dim),
                ) for _ in range(self.max_tokens)
            ])

        self.token_type = nn.Parameter(torch.zeros(self.max_tokens, self.token_dim))
        nn.init.normal_(self.token_type, std=0.02)

    @property
    def out_dim(self) -> int:
        return int(self.token_dim)

    @property
    def num_tokens(self) -> int:
        return int(self.max_tokens)

    def iter_encoder_backbones(self):
        return [("backbone_shared", self.backbone_shared)]

    def _soft_shell(self, mask: torch.Tensor, radius: int) -> torch.Tensor:
        return soft_shell_around_mask(mask, radius=int(radius))

    def _roi_support_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Forward-pass hard ROI support with a soft straight-through gradient."""

        soft = mask.clamp(0, 1)
        threshold = float(self.roi_support_threshold)
        if threshold <= 0.0:
            hard = (soft > 0).to(dtype=soft.dtype)
        else:
            hard = (soft >= threshold).to(dtype=soft.dtype)

        flat_hard = hard.flatten(1)
        empty = flat_hard.sum(dim=1) <= 0.0
        rel = float(self.roi_support_fallback_relmax)
        fallback_floor = float(self.roi_support_fallback_threshold)
        if rel > 0.0 or fallback_floor > 0.0:
            flat_soft = soft.flatten(1)
            maxv = flat_soft.max(dim=1).values.view(-1, 1, 1, 1, 1)
            fallback_thr = maxv.new_full(maxv.shape, fallback_floor)
            if rel > 0.0:
                fallback_thr = torch.maximum(fallback_thr, maxv * rel)
            fallback = (soft >= fallback_thr).to(dtype=soft.dtype)
            min_peak = max(fallback_floor, threshold * rel)
            has_signal = (maxv.flatten() >= min_peak).to(torch.bool)
            use_fallback = empty & has_signal
            hard = torch.where(use_fallback.view(-1, 1, 1, 1, 1), fallback, hard)

        return hard.detach() - soft.detach() + soft

    @staticmethod
    def _apply_native_support_floor(
        support: torch.Tensor,
        native_mask: Optional[torch.Tensor],
        alpha: float,
    ) -> torch.Tensor:
        if native_mask is None or float(alpha) <= 0.0:
            return support
        return torch.maximum(support, native_mask.clamp(0, 1).to(device=support.device, dtype=support.dtype))

    def _ct_looks_hu(self, ct: torch.Tensor) -> bool:
        cmin = float(ct.amin().item())
        cmax = float(ct.amax().item())
        return (cmax > 50.0) or (cmin < -50.0)

    def _auto_body_thr(self, ct: torch.Tensor) -> float:
        return float(self.body_ct_thr_hu) if self._ct_looks_hu(ct) else 0.02

    def _deep_present(self, mask: torch.Tensor, deep_size: Tuple[int, int, int]) -> torch.Tensor:
        if self.min_roi_voxels_deep <= 0:
            return mask.new_ones((mask.shape[0],), dtype=torch.bool)
        m_ds = interp_mask(mask, size=deep_size, mode=self.mask_interp).clamp(0, 1)
        s = m_ds.sum(dim=(2, 3, 4)).squeeze(1)
        return s >= float(self.min_roi_voxels_deep)

    @staticmethod
    def _raw_present(mask01: torch.Tensor, thr: float) -> torch.Tensor:
        return (mask01 > float(thr)).flatten(1).any(dim=1)

    def _presence_from_mask(self, mask01: torch.Tensor, deep_size: Tuple[int, int, int]) -> torch.Tensor:
        mean_present = mask01.mean(dim=(2, 3, 4)).squeeze(1) > self.min_roi_frac
        return mean_present & self._deep_present(mask01, deep_size)

    def enable_mask_patch_embed_training(self, verbose: bool = True):
        if verbose:
            print("[PATCH][INFO] contour-aware CT-only encoder has no mask input channels; ignoring mask patch-embed training request.")

    def _sanitize_tensor(
        self,
        tensor: torch.Tensor,
        *,
        name: str,
        posinf: float = 0.0,
        neginf: float = 0.0,
        clamp_abs: float = 0.0,
    ) -> torch.Tensor:
        if not self.sync_sanitize_checks:
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=posinf, neginf=neginf)
        elif not torch.isfinite(tensor).all().item():
            if name not in self._warned_sanitized:
                bad = int((~torch.isfinite(tensor)).sum().item())
                print(f"[WARN][CONTOUR] sanitized {bad} non-finite value(s) in {name}", flush=True)
                self._warned_sanitized.add(name)
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=posinf, neginf=neginf)
        if float(clamp_abs) > 0.0:
            tensor = tensor.clamp(min=-float(clamp_abs), max=float(clamp_abs))
        return tensor

    def _sync_backbone_eval(self):
        self.backbone_shared.eval()

    def forward(
        self,
        x_img: torch.Tensor,
        *,
        mask_pt: Optional[torch.Tensor] = None,
        mask_ln: Optional[torch.Tensor] = None,
        voxel_spacing_dhw: Optional[torch.Tensor] = None,
        region_masks: Optional[torch.Tensor] = None,
        region_validity: Optional[torch.Tensor] = None,
        region_states: Optional[torch.Tensor] = None,
        physical_geometry: Optional[torch.Tensor] = None,
        teacher_force_alpha: float = 0.0,
        return_aux: bool = False,
        return_cam_features: bool = False,
    ):
        if x_img.ndim != 5 or x_img.size(1) != 1:
            raise ValueError(f"Expected contour-aware x_img (B,1,D,H,W), got {tuple(x_img.shape)}")

        B = x_img.size(0)
        ct = x_img[:, 0:1]
        verified = region_masks is not None
        if self.require_verified_regions and not verified:
            raise ValueError("Verified compartment masks are required; regenerate inputs with region manifests.")
        if verified:
            if tuple(region_masks.shape) != (B, 4, *ct.shape[2:]):
                raise ValueError("region_masks must be [B,4,D,H,W] on the CT grid")
            if region_validity is None or region_states is None or physical_geometry is None:
                raise ValueError("Region masks require validity, explicit states, and physical geometry")
            if region_validity.shape != (B, 4) or region_states.shape != (B, 4):
                raise ValueError("Region validity/states must be [B,4]")
            if not ((region_validity == 0) | (region_validity == 1)).all():
                raise ValueError("Region validity must be binary")
            if not torch.isfinite(region_states).all() or not (region_states == region_states.long()).all():
                raise ValueError("Region states must be integer codes")
            if not torch.isfinite(region_masks).all() or not ((region_masks == 0) | (region_masks == 1)).all():
                raise ValueError("Verified compartment masks must be finite binary supports")
            region_masks = region_masks.to(device=ct.device, dtype=ct.dtype)
            region_validity = region_validity.to(device=ct.device, dtype=torch.bool)
            region_states = region_states.to(device=ct.device, dtype=torch.long)
            if not ((region_states >= 0) & (region_states < len(REGION_STATUS_CODES))).all():
                raise ValueError("Unknown compartment state")
            has_support = region_masks.flatten(2).any(dim=2)
            if (has_support != (region_states == REGION_STATUS_CODES["present"])).any():
                raise ValueError("Compartment states must agree with measured tissue support")
            expected_validity = (region_states == REGION_STATUS_CODES["present"]) | (region_states == REGION_STATUS_CODES["absent"])
            if (region_validity != expected_validity).any():
                raise ValueError("Missing/invalid compartments cannot have valid support")
            interiors = region_masks[:, 0:1] + region_masks[:, 2:3]
            if (interiors * (region_masks[:, 1:2] + region_masks[:, 3:4])).any():
                raise ValueError("Exterior compartments must exclude both tumor interiors")
            gdim = len(PHYSICAL_GEOMETRY_FEATURES)
            physical_geometry = physical_geometry.to(device=ct.device, dtype=torch.float32)
            if physical_geometry.shape != (B, 2 * gdim) or not torch.isfinite(physical_geometry).all():
                raise ValueError("Physical geometry must contain finite values and validity flags")

        if mask_pt is not None:
            mask_pt = mask_pt.to(device=ct.device, dtype=ct.dtype).clamp(0, 1)
        if mask_ln is not None:
            mask_ln = mask_ln.to(device=ct.device, dtype=ct.dtype).clamp(0, 1)

        body = None
        if self.shell_body_from_ct and not verified:
            thr = self._auto_body_thr(ct) if (isinstance(self.body_ct_thr, str) and self.body_ct_thr == "auto") else float(self.body_ct_thr)
            body = (ct > thr).float()
            if self.body_close_r > 0:
                body = binary_close(body, self.body_close_r)
            frac = body.mean(dim=(2, 3, 4)).squeeze(1)
            valid = frac >= 0.02
            if 0.0 < self.body_max_frac < 1.0:
                valid = valid & (frac <= self.body_max_frac)
            body = body * valid.to(dtype=body.dtype, device=body.device).view(-1, 1, 1, 1, 1)

        feats = swinvit_features(self.backbone_shared, ct, self.normalize)
        feats = convert_swinvit_feats_to_channel_first(
            feats,
            self._expected_c,
            strict=self.strict_swinvit_layout,
            print_shapes=(self.debug_swinvit_layout and (not self._checked_layout)),
            tag="swinViT-SHARED",
        )
        feats = [self._sanitize_tensor(f, name=f"swin_feat_{i}") for i, f in enumerate(feats)]
        self._checked_layout = True

        use_feats = list(feats[-4:]) if (self.use_multiscale and len(feats) >= 4) else [feats[-1]]
        fdeep = use_feats[-1]
        loc_idx = max(0, len(use_feats) - min(self.loc_feature_from_end, len(use_feats)))
        floc = use_feats[loc_idx]
        deep_size = tuple(int(x) for x in fdeep.shape[2:])

        loc_pt_logits = self._sanitize_tensor(
            self.loc_pt_head(floc), name="loc_pt_logits", posinf=30.0, neginf=-30.0, clamp_abs=30.0
        )
        loc_ln_logits = self._sanitize_tensor(
            self.loc_ln_head(floc), name="loc_ln_logits", posinf=30.0, neginf=-30.0, clamp_abs=30.0
        )
        loc_pt_prob = torch.sigmoid(loc_pt_logits)
        loc_ln_prob = torch.sigmoid(loc_ln_logits)

        pt_prob = F.interpolate(loc_pt_prob, size=tuple(int(x) for x in ct.shape[2:]), mode="trilinear", align_corners=False).clamp(0, 1)
        ln_prob = F.interpolate(loc_ln_prob, size=tuple(int(x) for x in ct.shape[2:]), mode="trilinear", align_corners=False).clamp(0, 1)

        alpha = float(max(0.0, min(1.0, teacher_force_alpha)))
        pt_soft_used = pt_prob
        ln_soft_used = ln_prob
        if alpha > 0.0 and mask_pt is not None and mask_ln is not None:
            pt_soft_used = (1.0 - alpha) * pt_prob + alpha * mask_pt
            ln_soft_used = (1.0 - alpha) * ln_prob + alpha * mask_ln
        pt_soft_used = pt_soft_used.clamp(0, 1)
        ln_soft_used = ln_soft_used.clamp(0, 1)
        pt_used = self._roi_support_mask(pt_soft_used)
        ln_used = self._roi_support_mask(ln_soft_used)
        pt_used = self._apply_native_support_floor(pt_used, mask_pt, alpha)
        ln_used = self._apply_native_support_floor(ln_used, mask_ln, alpha)
        if verified:
            # These persisted supports also identify the corresponding radiomics.
            # Neither predicted masks nor teacher forcing may change their meaning.
            pt_used, ln_used = region_masks[:, 0:1], region_masks[:, 2:3]
            pt_soft_used, ln_soft_used = pt_used, ln_used

        raw_pt_source = mask_pt if (mask_pt is not None and alpha > 0.0) else pt_soft_used
        raw_ln_source = mask_ln if (mask_ln is not None and alpha > 0.0) else ln_soft_used
        pt_present_raw = self._raw_present(raw_pt_source, self.raw_mask_threshold)
        ln_present_raw = self._raw_present(raw_ln_source, self.raw_mask_threshold)

        presence_logits = self._sanitize_tensor(
            self.presence_head(fdeep), name="presence_logits", posinf=30.0, neginf=-30.0, clamp_abs=30.0
        )
        pt_presence_logits = presence_logits[:, 0]
        ln_presence_logits = presence_logits[:, 1]

        pres_global = torch.ones(B, device=ct.device, dtype=torch.bool)
        pres_pt_intra = self._presence_from_mask(pt_used, deep_size)
        pres_ln_intra = self._presence_from_mask(ln_used, deep_size)

        if verified:
            pt_shell, ln_shell = region_masks[:, 1:2], region_masks[:, 3:4]
        else:
            pt_shell = soft_shell_around_mask(pt_used, radius=self.pt_shell_radius,
                thickness_mm=self.pt_shell_thickness_mm, voxel_spacing_dhw=voxel_spacing_dhw)
            ln_shell = soft_shell_around_mask(ln_used, radius=self.ln_shell_radius,
                thickness_mm=self.ln_shell_thickness_mm, voxel_spacing_dhw=voxel_spacing_dhw)
            exterior = 1.0 - torch.maximum(pt_used, ln_used)
            pt_shell, ln_shell = pt_shell * exterior, ln_shell * exterior
            if body is not None:
                pt_shell, ln_shell = pt_shell * body, ln_shell * body

        pres_pt_peri = self._presence_from_mask(pt_shell, deep_size)
        pres_ln_peri = self._presence_from_mask(ln_shell, deep_size)
        pres_shape_spatial = pres_pt_intra | pres_ln_intra
        pres = torch.stack(
            [pres_global, pres_pt_intra, pres_pt_peri, pres_ln_intra, pres_ln_peri, pres_shape_spatial],
            dim=1,
        )

        if verified:
            measured = region_masks.flatten(2).sum(-1) > 0
            measured &= region_validity & (region_states == REGION_STATUS_CODES["present"])
            pres[:, 1:5] = measured
            pres_pt_intra, pres_ln_intra = measured[:, 0], measured[:, 2]
            # Physical validity/missingness is information even without a lesion.
            pres[:, 5] = True
        elif self.force_presence_from_raw_masks:
            pres = pres.clone()
            pres[:, 0] = True
            pres[:, 1] = pt_present_raw
            pres[:, 3] = ln_present_raw
            pres[:, 5] = pt_present_raw | ln_present_raw

        token_inputs: List[torch.Tensor] = []

        g_vecs = [self.gap(f).flatten(1) for f in use_feats]
        g = torch.cat(g_vecs, dim=1)
        g = torch.cat([g, ct_stats_global(ct, body=body)], dim=1)
        if body is not None:
            body_deep = interp_mask(body, size=fdeep.shape[2:], mode="nearest")
            g = torch.cat([g, self.attn_pool(fdeep, body_deep)], dim=1)
        else:
            g = torch.cat([g, self.attn_pool(fdeep, None)], dim=1)
        token_inputs.append(g)

        vecs = [ct_stats_in_mask(ct, pt_used, include_volume=self.include_roi_volume)]
        for f in use_feats:
            vecs.append(masked_mean(f, interp_mask(pt_used, size=f.shape[2:], mode=self.mask_interp)))
        token_inputs.append(torch.cat(vecs + [self.attn_pool(fdeep, interp_mask(pt_used, size=fdeep.shape[2:], mode=self.mask_interp))], dim=1))

        vecs = [ct_stats_in_mask(ct, pt_shell, include_volume=self.include_shell_volume)]
        for f in use_feats:
            vecs.append(masked_mean(f, interp_mask(pt_shell, size=f.shape[2:], mode=self.mask_interp)))
        token_inputs.append(torch.cat(vecs + [self.attn_pool(fdeep, interp_mask(pt_shell, size=fdeep.shape[2:], mode=self.mask_interp))], dim=1))

        vecs = [ct_stats_in_mask(ct, ln_used, include_volume=self.include_roi_volume)]
        for f in use_feats:
            vecs.append(masked_mean(f, interp_mask(ln_used, size=f.shape[2:], mode=self.mask_interp)))
        token_inputs.append(torch.cat(vecs + [self.attn_pool(fdeep, interp_mask(ln_used, size=fdeep.shape[2:], mode=self.mask_interp))], dim=1))

        vecs = [ct_stats_in_mask(ct, ln_shell, include_volume=self.include_shell_volume)]
        for f in use_feats:
            vecs.append(masked_mean(f, interp_mask(ln_shell, size=f.shape[2:], mode=self.mask_interp)))
        token_inputs.append(torch.cat(vecs + [self.attn_pool(fdeep, interp_mask(ln_shell, size=fdeep.shape[2:], mode=self.mask_interp))], dim=1))

        token_inputs.append(
            shape_spatial_features(
                pt_used,
                pt_shell,
                ln_used,
                ln_shell,
                body,
                pres_pt_intra,
                pres_ln_intra,
            )
        )
        if verified:
            values, valid = physical_geometry.split(gdim, dim=1)
            if not ((valid == 0) | (valid == 1)).all():
                raise ValueError("Physical geometry validity must be binary")
            # Fixed unit scaling, not a fit on validation/test statistics.
            scaled = torch.sign(values) * torch.log1p(values.abs()) * valid
            state_flags = F.one_hot(region_states, num_classes=len(REGION_STATUS_CODES)).flatten(1).float()
            token_inputs[-1] = torch.cat([token_inputs[-1], scaled, valid, state_flags], dim=1)

        hs: List[torch.Tensor] = []
        for i in range(self.max_tokens):
            h = self.token_mlp[i](token_inputs[i]) + self.token_type[i].unsqueeze(0)
            if i > 0:
                absent = ~pres[:, i]
                h = h.masked_fill(absent.unsqueeze(1), 0.0)
            hs.append(h)

        tok_img = torch.stack(hs, dim=1)
        tok_img = torch.nan_to_num(tok_img, nan=0.0, posinf=0.0, neginf=0.0)

        if self.training and self.token_dropout > 0:
            pres2m = pres.clone()
            tok2m = tok_img.clone()
            for tok_i in (1, 2, 3, 4):
                drop = (torch.rand(B, device=x_img.device) < self.token_dropout) & pres2m[:, tok_i]
                pres2m[:, tok_i] = pres2m[:, tok_i] & (~drop)
                tok2m[:, tok_i, :] = tok2m[:, tok_i, :] * (~drop).to(dtype=tok2m.dtype).unsqueeze(1)
            pres, tok_img = pres2m, tok2m

        if not return_aux and not return_cam_features:
            return tok_img, pres

        aux: Dict[str, torch.Tensor] = {
            "loc_pt_logits": loc_pt_logits,
            "loc_ln_logits": loc_ln_logits,
            "pt_prob": pt_prob,
            "ln_prob": ln_prob,
            "pt_soft_used": pt_soft_used,
            "ln_soft_used": ln_soft_used,
            "pt_used": pt_used,
            "ln_used": ln_used,
            "pt_shell": pt_shell,
            "ln_shell": ln_shell,
            "body": body if body is not None else torch.ones_like(ct),
            "pt_presence_logits": pt_presence_logits,
            "ln_presence_logits": ln_presence_logits,
        }
        if return_cam_features:
            cam_features: List[torch.Tensor] = []
            for feat in use_feats:
                if feat.requires_grad:
                    feat.retain_grad()
                    cam_features.append(feat)
            aux["cam_features"] = cam_features
        return tok_img, pres, aux
