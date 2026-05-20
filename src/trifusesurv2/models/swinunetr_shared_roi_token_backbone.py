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

import math
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
        raw_logits = logits
        if mask is not None:
            mask_flat = mask.clamp(0, 1).flatten(2).to(dtype=logits.dtype, device=logits.device)
            has_support = mask_flat.sum(dim=-1, keepdim=True) > self.eps
            if self.mask_mode == "bias":
                logits = logits + self.mask_bias * mask_flat
            elif self.mask_mode == "weighted":
                logits = logits + torch.log(mask_flat.clamp_min(self.eps))
                logits = torch.where(has_support, logits, raw_logits)
            else:
                support = mask_flat > self.eps
                masked_logits = logits.masked_fill(~support, torch.finfo(logits.dtype).min)
                logits = torch.where(has_support, masked_logits, logits)
        w = torch.softmax(logits / self.temperature, dim=-1)
        feat_flat = feat.flatten(2)
        return (feat_flat * w).sum(dim=-1)


def interp_mask(mask: torch.Tensor, size: Tuple[int, int, int], mode: str) -> torch.Tensor:
    if mode == "nearest":
        return F.interpolate(mask, size=size, mode="nearest")
    if mode == "trilinear":
        return F.interpolate(mask, size=size, mode="trilinear", align_corners=False)
    raise ValueError(f"Unknown mask_interp mode: {mode}")


def binary_close(m: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return m
    k = 2 * r + 1
    d = (F.max_pool3d(m, kernel_size=k, stride=1, padding=r) > 0).float()
    e = 1.0 - (F.max_pool3d((1.0 - d).clamp(0, 1), kernel_size=k, stride=1, padding=r) > 0).float()
    return e


def _radius_tuple_from_spacing(
    *,
    thickness_mm: float,
    spacing_dhw,
    fallback_radius: int,
) -> Tuple[int, int, int]:
    if float(thickness_mm) <= 0.0:
        r = int(max(0, fallback_radius))
        return r, r, r
    try:
        vals = [float(x) for x in list(spacing_dhw)[:3]]
    except Exception:
        vals = []
    if len(vals) != 3 or any((not math.isfinite(v)) or v <= 0.0 for v in vals):
        r = int(max(0, fallback_radius))
        return r, r, r
    return tuple(max(1, int(math.ceil(float(thickness_mm) / max(v, 1e-6)))) for v in vals)


def soft_shell_around_mask(
    mask: torch.Tensor,
    *,
    radius: int,
    thickness_mm: float = 0.0,
    voxel_spacing_dhw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Differentiable hard-support shell, optionally using physical mm spacing.

    ``voxel_spacing_dhw`` is in array order (z/y/x or D/H/W). When a positive
    thickness is supplied, each sample is dilated by ceil(thickness_mm / spacing)
    per axis. This preserves the existing max-pool morphology while making the
    shell thickness physically meaningful after ROI-crop resampling.
    """

    m = mask.clamp(0, 1)
    fallback_radius = int(max(0, radius))
    if float(thickness_mm) <= 0.0 or voxel_spacing_dhw is None:
        if fallback_radius <= 0:
            return m.new_zeros(m.shape)
        k = 2 * fallback_radius + 1
        dil = F.max_pool3d(m, kernel_size=k, stride=1, padding=fallback_radius)
        return (dil - m).clamp(0, 1)

    spacing = voxel_spacing_dhw
    if torch.is_tensor(spacing):
        spacing_cpu = spacing.detach().float().cpu()
    else:
        spacing_cpu = torch.as_tensor(spacing, dtype=torch.float32)
    if spacing_cpu.ndim == 1:
        spacing_cpu = spacing_cpu.view(1, -1)
    if spacing_cpu.ndim != 2 or int(spacing_cpu.shape[1]) < 3:
        return soft_shell_around_mask(m, radius=fallback_radius)
    if int(spacing_cpu.shape[0]) == 1 and int(m.shape[0]) > 1:
        spacing_cpu = spacing_cpu.expand(int(m.shape[0]), -1)
    if int(spacing_cpu.shape[0]) != int(m.shape[0]):
        return soft_shell_around_mask(m, radius=fallback_radius)

    radii = [
        _radius_tuple_from_spacing(
            thickness_mm=float(thickness_mm),
            spacing_dhw=spacing_cpu[i, :3].tolist(),
            fallback_radius=fallback_radius,
        )
        for i in range(int(m.shape[0]))
    ]
    if all(r == radii[0] for r in radii):
        rz, ry, rx = radii[0]
        if max(rz, ry, rx) <= 0:
            return m.new_zeros(m.shape)
        dil = F.max_pool3d(
            m,
            kernel_size=(2 * rz + 1, 2 * ry + 1, 2 * rx + 1),
            stride=1,
            padding=(rz, ry, rx),
        )
        return (dil - m).clamp(0, 1)

    shells = []
    for i, (rz, ry, rx) in enumerate(radii):
        one = m[i : i + 1]
        if max(rz, ry, rx) <= 0:
            shells.append(one.new_zeros(one.shape))
            continue
        dil = F.max_pool3d(
            one,
            kernel_size=(2 * rz + 1, 2 * ry + 1, 2 * rx + 1),
            stride=1,
            padding=(rz, ry, rx),
        )
        shells.append((dil - one).clamp(0, 1))
    return torch.cat(shells, dim=0)


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
        ln_shell_thickness_mm: float = 0.0,
        shell_body_from_ct: bool = True,
        body_ct_thr: Union[str, float] = "auto",
        body_ct_thr_hu: float = -500.0,
        body_close_r: int = 2,
        body_max_frac: float = 0.995,
        strict_swinvit_layout: bool = True,
        debug_swinvit_layout: bool = False,
        force_presence_from_raw_masks: bool = False,
        raw_mask_threshold: float = 0.5,
        fallback_peri_to_intra: bool = True,
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
        self.fallback_peri_to_intra = bool(fallback_peri_to_intra)
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
        teacher_force_alpha: float = 0.0,
        return_aux: bool = False,
        return_cam_features: bool = False,
    ):
        if x_img.ndim != 5 or x_img.size(1) != 1:
            raise ValueError(f"Expected contour-aware x_img (B,1,D,H,W), got {tuple(x_img.shape)}")

        B = x_img.size(0)
        ct = x_img[:, 0:1]

        if mask_pt is not None:
            mask_pt = mask_pt.to(device=ct.device, dtype=ct.dtype).clamp(0, 1)
        if mask_ln is not None:
            mask_ln = mask_ln.to(device=ct.device, dtype=ct.dtype).clamp(0, 1)

        body = None
        if self.shell_body_from_ct:
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

        pt_shell = soft_shell_around_mask(
            pt_used,
            radius=self.pt_shell_radius,
            thickness_mm=self.pt_shell_thickness_mm,
            voxel_spacing_dhw=voxel_spacing_dhw,
        )
        ln_shell = soft_shell_around_mask(
            ln_used,
            radius=self.ln_shell_radius,
            thickness_mm=self.ln_shell_thickness_mm,
            voxel_spacing_dhw=voxel_spacing_dhw,
        )
        if body is not None:
            pt_shell = pt_shell * body
            ln_shell = ln_shell * body

        if self.fallback_peri_to_intra:
            pt_shell_sum = pt_shell.sum(dim=(2, 3, 4)).squeeze(1)
            ln_shell_sum = ln_shell.sum(dim=(2, 3, 4)).squeeze(1)
            bad_pt_peri = pres_pt_intra & (pt_shell_sum <= 0.0)
            bad_ln_peri = pres_ln_intra & (ln_shell_sum <= 0.0)
            pt_shell = torch.where(bad_pt_peri.view(-1, 1, 1, 1, 1), pt_used, pt_shell)
            ln_shell = torch.where(bad_ln_peri.view(-1, 1, 1, 1, 1), ln_used, ln_shell)

        pres_pt_peri = self._presence_from_mask(pt_shell, deep_size)
        pres_ln_peri = self._presence_from_mask(ln_shell, deep_size)
        pres_shape_spatial = pres_pt_intra | pres_ln_intra
        pres = torch.stack(
            [pres_global, pres_pt_intra, pres_pt_peri, pres_ln_intra, pres_ln_peri, pres_shape_spatial],
            dim=1,
        )

        if self.force_presence_from_raw_masks:
            pres = pres.clone()
            pres[:, 0] = True
            pres[:, 1] = pt_present_raw
            pres[:, 2] = pt_present_raw
            pres[:, 3] = ln_present_raw
            pres[:, 4] = ln_present_raw
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
