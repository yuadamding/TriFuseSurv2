#!/usr/bin/env python3
"""
model/swinunetr_ptln_intra_peri_token_backbone.py

Two-backbone SwinUNETR token backbone:
- backbone_pt: pretrained on PT segmentation
- backbone_ln: pretrained on LN segmentation

Input x_img: (B,3,D,H,W) = [CT, PT_mask, LN_mask]

Tokens (B,5,Dtok):
  0: GLOBAL
  1: PT_INTRA
  2: PT_PERI
  3: LN_INTRA
  4: LN_PERI

Mask-channel patch-embed trainable ONLY:
- Replace patch_embed.proj (Conv3d in_ch=2) with split conv:
    out = Conv_ct(ct) + Conv_mask(mask)
  Conv_ct frozen (initialized from pretrained channel-0)
  Conv_mask trainable (initialized zeros)

UPDATE (drop-in):
- Added `token_mlp_hidden_dim` to optionally make token MLP 2-layer:
    LazyLinear(hidden) -> GELU -> Dropout -> Linear(hidden->token_dim) -> GELU -> Dropout -> LayerNorm

NEW (guaranteed presence):
- Optional hard guarantee that PT/LN tokens are present whenever the corresponding RAW masks are non-empty:
    - if PT mask has any voxel > raw_mask_threshold: pres[:,1]=pres[:,2]=True
    - if LN mask has any voxel > raw_mask_threshold: pres[:,3]=pres[:,4]=True
- Optional peri-shell fallback:
    - if shell becomes empty but intra mask exists, use intra mask as peri mask
  This prevents "mask present but token absent" caused by:
    - min_roi_frac / min_roi_voxels_deep thresholds
    - downsampling masks erasing small ROIs
    - body restriction removing peri shells
"""

from __future__ import annotations

import contextlib
from typing import Tuple, Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from trifusesurv.models.swinunetr_backbone_utils import (
    build_swinunetr_backbone,
    swinvit_features,
    convert_swinvit_feats_to_channel_first,
    _expected_channels,
)


# ---------------------------
# helpers (copied from your union-shell backbone)
# ---------------------------
def masked_mean(feat: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    w = mask.clamp(0, 1)
    denom = w.sum(dim=(2, 3, 4)).clamp_min(eps)
    num = (feat * w).sum(dim=(2, 3, 4))
    return num / denom


def ct_stats_in_mask(ct: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    w = mask.clamp(0, 1)
    denom = w.sum(dim=(2, 3, 4)).clamp_min(eps)
    mu = (ct * w).sum(dim=(2, 3, 4)) / denom
    second = ((ct * ct) * w).sum(dim=(2, 3, 4)) / denom
    var = (second - mu * mu).clamp_min(0.0)
    sd = torch.sqrt(var + 1e-8)
    vol = w.mean(dim=(2, 3, 4))
    return torch.cat([mu, sd, vol], dim=1)


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


class AttnPool3D(nn.Module):
    def __init__(self, mask_bias: float = 2.0, temperature: float = 1.0):
        super().__init__()
        self.attn = nn.LazyConv3d(1, kernel_size=1, bias=False)
        self.mask_bias = float(mask_bias)
        self.temperature = float(temperature)

    def forward(self, feat: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.attn(feat)
        if mask is not None:
            logits = logits + self.mask_bias * mask.clamp(0, 1)
        w = torch.softmax((logits / self.temperature).flatten(2), dim=-1)
        feat_flat = feat.flatten(2)
        return (feat_flat * w).sum(dim=-1)


def interp_mask(mask: torch.Tensor, size: Tuple[int, int, int], mode: str) -> torch.Tensor:
    if mode == "nearest":
        return F.interpolate(mask, size=size, mode="nearest")
    if mode == "trilinear":
        return F.interpolate(mask, size=size, mode="trilinear", align_corners=False)
    raise ValueError(f"Unknown mask_interp mode: {mode}")


def binary_dilate(m: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return m
    k = 2 * r + 1
    return (F.max_pool3d(m, kernel_size=k, stride=1, padding=r) > 0).float()


def binary_close(m: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return m
    k = 2 * r + 1
    d = (F.max_pool3d(m, kernel_size=k, stride=1, padding=r) > 0).float()
    e = 1.0 - (F.max_pool3d((1.0 - d).clamp(0, 1), kernel_size=k, stride=1, padding=r) > 0).float()
    return e


# ---------------------------
# patch-embed split conv: trainable mask channel only
# ---------------------------
class SplitMaskPatchEmbedConv3d(nn.Module):
    """
    Replacement for patch_embed.proj (Conv3d with in_channels=2).
    out = Conv_ct(ct) + Conv_mask(mask)
    Conv_ct frozen (copied from original weight[:,0:1])
    Conv_mask trainable (zeros init)
    """
    def __init__(self, conv2: nn.Conv3d):
        super().__init__()
        if not isinstance(conv2, nn.Conv3d):
            raise TypeError("SplitMaskPatchEmbedConv3d expects nn.Conv3d.")
        if int(conv2.in_channels) != 2:
            raise ValueError(f"Expected in_channels=2, got {conv2.in_channels}")
        if int(conv2.groups) != 1:
            raise ValueError(f"Expected groups=1 for patch embed conv, got {conv2.groups}")

        bias = conv2.bias is not None
        self.conv_ct = nn.Conv3d(
            in_channels=1,
            out_channels=conv2.out_channels,
            kernel_size=conv2.kernel_size,
            stride=conv2.stride,
            padding=conv2.padding,
            dilation=conv2.dilation,
            groups=1,
            bias=bias,
        )
        self.conv_mask = nn.Conv3d(
            in_channels=1,
            out_channels=conv2.out_channels,
            kernel_size=conv2.kernel_size,
            stride=conv2.stride,
            padding=conv2.padding,
            dilation=conv2.dilation,
            groups=1,
            bias=False,
        )

        with torch.no_grad():
            self.conv_ct.weight.copy_(conv2.weight[:, 0:1].contiguous())
            if bias:
                self.conv_ct.bias.copy_(conv2.bias)
            self.conv_mask.weight.zero_()

        for p in self.conv_ct.parameters():
            p.requires_grad = False
        for p in self.conv_mask.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.size(1) != 2:
            raise ValueError(f"Expected x (B,2,D,H,W), got {tuple(x.shape)}")
        ct = x[:, 0:1]
        m  = x[:, 1:2]
        return self.conv_ct(ct) + self.conv_mask(m)


def _replace_patch_embed_proj_with_split(backbone: nn.Module, verbose: bool = True) -> SplitMaskPatchEmbedConv3d:
    """
    Find first module named *.patch_embed.proj that is Conv3d(in_ch=2), replace with SplitMaskPatchEmbedConv3d.
    Returns the split module.
    """
    target_name = None
    target_conv = None
    for name, mod in backbone.named_modules():
        if name.endswith("patch_embed.proj") and isinstance(mod, nn.Conv3d) and int(mod.in_channels) == 2:
            target_name = name
            target_conv = mod
            break
    if target_name is None or target_conv is None:
        raise RuntimeError("Could not find patch_embed.proj Conv3d(in_ch=2) inside SwinUNETR backbone.")

    parent_path, attr = target_name.rsplit(".", 1)
    parent = backbone
    for part in parent_path.split("."):
        parent = getattr(parent, part)

    split = SplitMaskPatchEmbedConv3d(target_conv).to(device=target_conv.weight.device, dtype=target_conv.weight.dtype)
    setattr(parent, attr, split)

    if verbose:
        print(f"[PATCH] Replaced {target_name} with SplitMaskPatchEmbedConv3d (train mask channel only).")

    return split


# ---------------------------
# backbone
# ---------------------------
class SwinUNETRPTLNIntraPeriTokenBackbone(nn.Module):
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
        use_multiscale: bool = True,
        mask_interp: str = "nearest",
        min_roi_frac: float = 1e-5,
        min_roi_voxels_deep: int = 8,
        token_dropout: float = 0.05,

        # peri shells
        pt_shell_radius: int = 3,
        ln_shell_radius: int = 3,
        shell_body_from_ct: bool = True,
        body_ct_thr: Union[str, float] = "auto",
        body_ct_thr_hu: float = -500.0,
        body_close_r: int = 2,
        body_max_frac: float = 0.995,

        strict_swinvit_layout: bool = True,
        debug_swinvit_layout: bool = False,

        # --- NEW: force token presence when raw masks are non-empty ---
        force_presence_from_raw_masks: bool = False,
        raw_mask_threshold: float = 0.5,
        fallback_peri_to_intra: bool = True,
    ):
        super().__init__()
        self.normalize = bool(normalize)
        self.use_multiscale = bool(use_multiscale)
        self.mask_interp = str(mask_interp)
        self.min_roi_frac = float(min_roi_frac)
        self.min_roi_voxels_deep = int(max(min_roi_voxels_deep, 0))
        self.token_dropout = float(max(token_dropout, 0.0))

        self.pt_shell_radius = int(pt_shell_radius)
        self.ln_shell_radius = int(ln_shell_radius)

        self.shell_body_from_ct = bool(shell_body_from_ct)
        self.body_ct_thr = body_ct_thr
        self.body_ct_thr_hu = float(body_ct_thr_hu)
        self.body_close_r = int(body_close_r)
        self.body_max_frac = float(body_max_frac)

        self.strict_swinvit_layout = bool(strict_swinvit_layout)
        self.debug_swinvit_layout = bool(debug_swinvit_layout)
        self._checked_layout_pt = False
        self._checked_layout_ln = False
        self._expected_c = _expected_channels(int(feature_size), max_pow=6)

        # --- NEW: forced presence controls ---
        self.force_presence_from_raw_masks = bool(force_presence_from_raw_masks)
        self.raw_mask_threshold = float(raw_mask_threshold)
        self.fallback_peri_to_intra = bool(fallback_peri_to_intra)

        # Two SwinUNETR backbones, each takes (CT, mask) => in_channels=2
        self.backbone_pt = build_swinunetr_backbone(
            img_size=tuple(img_size),
            in_channels=2,
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
        self.backbone_ln = build_swinunetr_backbone(
            img_size=tuple(img_size),
            in_channels=2,
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

        # attention pool
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.attn_pool = AttnPool3D(mask_bias=float(attn_mask_bias))

        # tokens: GLOBAL + PT_INTRA + PT_PERI + LN_INTRA + LN_PERI
        self.max_tokens = 5
        self.token_dim = int(token_dim)

        # token MLP: optionally 2-layer for higher capacity
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

        # handles to the split patch-embed modules (if enabled)
        self._pt_patch_split: Optional[SplitMaskPatchEmbedConv3d] = None
        self._ln_patch_split: Optional[SplitMaskPatchEmbedConv3d] = None

    @property
    def out_dim(self) -> int:
        return int(self.token_dim)

    @property
    def num_tokens(self) -> int:
        return int(self.max_tokens)

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
        # mask01: (B,1,D,H,W)
        return (mask01 > float(thr)).flatten(1).any(dim=1)

    def enable_mask_patch_embed_training(self, verbose: bool = True):
        """
        Freeze ALL backbone params, but make mask-channel patch_embed trainable only by replacing proj with split conv.
        """
        self._pt_patch_split = _replace_patch_embed_proj_with_split(self.backbone_pt, verbose=verbose)
        self._ln_patch_split = _replace_patch_embed_proj_with_split(self.backbone_ln, verbose=verbose)

        for p in self.backbone_pt.parameters():
            p.requires_grad = False
        for p in self.backbone_ln.parameters():
            p.requires_grad = False

        for p in self._pt_patch_split.conv_mask.parameters():
            p.requires_grad = True
        for p in self._ln_patch_split.conv_mask.parameters():
            p.requires_grad = True

        self.backbone_pt.eval()
        self.backbone_ln.eval()

        if verbose:
            n_pt = sum(p.requires_grad for p in self.backbone_pt.parameters())
            n_ln = sum(p.requires_grad for p in self.backbone_ln.parameters())
            print(f"[PATCH] PT backbone trainable params: {n_pt} (should be small, mask patch only)")
            print(f"[PATCH] LN backbone trainable params: {n_ln} (should be small, mask patch only)")

    def _sync_backbones_eval(self):
        self.backbone_pt.eval()
        self.backbone_ln.eval()

    def forward(self, x_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_img: (B,3,D,H,W) = [CT, PT_mask, LN_mask]
        returns:
          tok_img: (B,5,Dtok)
          pres:    (B,5 bool)
        """
        if x_img.ndim != 5 or x_img.size(1) < 3:
            raise ValueError(f"Expected x_img (B,3,D,H,W), got {tuple(x_img.shape)}")

        self._sync_backbones_eval()

        B = x_img.size(0)
        ct = x_img[:, 0:1]
        pt = x_img[:, 1:2].clamp(0, 1)
        ln = x_img[:, 2:3].clamp(0, 1)

        # --- NEW: raw mask presence (source-of-truth guarantee) ---
        pt_present_raw = self._raw_present(pt, self.raw_mask_threshold)  # (B,)
        ln_present_raw = self._raw_present(ln, self.raw_mask_threshold)  # (B,)

        # body mask (optional)
        body = None
        if self.shell_body_from_ct:
            thr = self._auto_body_thr(ct) if (isinstance(self.body_ct_thr, str) and self.body_ct_thr == "auto") else float(self.body_ct_thr)
            body = (ct > thr).float()
            if self.body_close_r > 0:
                body = binary_close(body, self.body_close_r)
            frac = body.mean(dim=(2, 3, 4)).squeeze(1)
            bad = (frac < 0.02)
            if 0.0 < self.body_max_frac < 1.0:
                bad = bad | (frac > self.body_max_frac)
            if bad.any():
                body = body.clone()
                body[bad] = 0.0
                if float(bad.float().mean().item()) > 0.5:
                    body = None

        xin_pt = torch.cat([ct, pt], dim=1)
        xin_ln = torch.cat([ct, ln], dim=1)

        feats_pt = swinvit_features(self.backbone_pt, xin_pt, self.normalize)
        feats_ln = swinvit_features(self.backbone_ln, xin_ln, self.normalize)

        feats_pt = convert_swinvit_feats_to_channel_first(
            feats_pt,
            self._expected_c,
            strict=self.strict_swinvit_layout,
            print_shapes=(self.debug_swinvit_layout and (not self._checked_layout_pt)),
            tag="swinViT-PT",
        )
        self._checked_layout_pt = True

        feats_ln = convert_swinvit_feats_to_channel_first(
            feats_ln,
            self._expected_c,
            strict=self.strict_swinvit_layout,
            print_shapes=(self.debug_swinvit_layout and (not self._checked_layout_ln)),
            tag="swinViT-LN",
        )
        self._checked_layout_ln = True

        use_feats_pt = list(feats_pt[-4:]) if (self.use_multiscale and len(feats_pt) >= 4) else [feats_pt[-1]]
        use_feats_ln = list(feats_ln[-4:]) if (self.use_multiscale and len(feats_ln) >= 4) else [feats_ln[-1]]

        fdeep_pt = use_feats_pt[-1]
        fdeep_ln = use_feats_ln[-1]
        deep_size_pt = tuple(int(x) for x in fdeep_pt.shape[2:])
        deep_size_ln = tuple(int(x) for x in fdeep_ln.shape[2:])

        # presence bits (original logic)
        pres_global = torch.ones(B, device=x_img.device, dtype=torch.bool)

        pres_pt_intra = (pt.mean(dim=(2, 3, 4)).squeeze(1) > self.min_roi_frac) & self._deep_present(pt, deep_size_pt)
        pres_ln_intra = (ln.mean(dim=(2, 3, 4)).squeeze(1) > self.min_roi_frac) & self._deep_present(ln, deep_size_ln)

        # shells
        pt_shell = (binary_dilate(pt, self.pt_shell_radius) - pt).clamp(0, 1)
        ln_shell = (binary_dilate(ln, self.ln_shell_radius) - ln).clamp(0, 1)
        if body is not None:
            pt_shell = pt_shell * body
            ln_shell = ln_shell * body

        # --- NEW: if shell collapses but intra exists, fall back peri mask to intra mask ---
        if self.force_presence_from_raw_masks and self.fallback_peri_to_intra:
            pt_shell_sum = pt_shell.sum(dim=(2, 3, 4)).squeeze(1)
            ln_shell_sum = ln_shell.sum(dim=(2, 3, 4)).squeeze(1)
            pt_shell_empty = (pt_shell_sum <= 0.0)
            ln_shell_empty = (ln_shell_sum <= 0.0)

            bad_pt_peri = pt_present_raw & pt_shell_empty
            bad_ln_peri = ln_present_raw & ln_shell_empty
            if bad_pt_peri.any():
                pt_shell = pt_shell.clone()
                pt_shell[bad_pt_peri] = pt[bad_pt_peri]
            if bad_ln_peri.any():
                ln_shell = ln_shell.clone()
                ln_shell[bad_ln_peri] = ln[bad_ln_peri]

        pres_pt_peri = (pt_shell.mean(dim=(2, 3, 4)).squeeze(1) > self.min_roi_frac) & self._deep_present(pt_shell, deep_size_pt)
        pres_ln_peri = (ln_shell.mean(dim=(2, 3, 4)).squeeze(1) > self.min_roi_frac) & self._deep_present(ln_shell, deep_size_ln)

        pres = torch.stack([pres_global, pres_pt_intra, pres_pt_peri, pres_ln_intra, pres_ln_peri], dim=1)

        # --- NEW: hard guarantee presence based on raw masks ---
        if self.force_presence_from_raw_masks:
            pres = pres.clone()
            pres[:, 0] = True
            pres[:, 1] = pt_present_raw
            pres[:, 2] = pt_present_raw
            pres[:, 3] = ln_present_raw
            pres[:, 4] = ln_present_raw

        token_inputs: List[torch.Tensor] = []

        # GLOBAL token: use PT features (same architecture; avoids third forward)
        g_vecs = [self.gap(f).flatten(1) for f in use_feats_pt]
        g = torch.cat(g_vecs, dim=1)
        g = torch.cat([g, ct_stats_global(ct, body=body)], dim=1)
        if body is not None:
            body_deep = interp_mask(body, size=fdeep_pt.shape[2:], mode="nearest")
            g = torch.cat([g, self.attn_pool(fdeep_pt, body_deep)], dim=1)
        else:
            g = torch.cat([g, self.attn_pool(fdeep_pt, None)], dim=1)
        token_inputs.append(g)

        # PT_INTRA token
        vecs = [ct_stats_in_mask(ct, pt)]
        for f in use_feats_pt:
            m_ds = interp_mask(pt, size=f.shape[2:], mode=self.mask_interp)
            vecs.append(masked_mean(f, m_ds))
        pt_deep = interp_mask(pt, size=fdeep_pt.shape[2:], mode=self.mask_interp)
        vecs.append(self.attn_pool(fdeep_pt, pt_deep))
        token_inputs.append(torch.cat(vecs, dim=1))

        # PT_PERI token
        vecs = [ct_stats_in_mask(ct, pt_shell)]
        for f in use_feats_pt:
            m_ds = interp_mask(pt_shell, size=f.shape[2:], mode=self.mask_interp)
            vecs.append(masked_mean(f, m_ds))
        ptp_deep = interp_mask(pt_shell, size=fdeep_pt.shape[2:], mode=self.mask_interp)
        vecs.append(self.attn_pool(fdeep_pt, ptp_deep))
        token_inputs.append(torch.cat(vecs, dim=1))

        # LN_INTRA token
        vecs = [ct_stats_in_mask(ct, ln)]
        for f in use_feats_ln:
            m_ds = interp_mask(ln, size=f.shape[2:], mode=self.mask_interp)
            vecs.append(masked_mean(f, m_ds))
        ln_deep = interp_mask(ln, size=fdeep_ln.shape[2:], mode=self.mask_interp)
        vecs.append(self.attn_pool(fdeep_ln, ln_deep))
        token_inputs.append(torch.cat(vecs, dim=1))

        # LN_PERI token
        vecs = [ct_stats_in_mask(ct, ln_shell)]
        for f in use_feats_ln:
            m_ds = interp_mask(ln_shell, size=f.shape[2:], mode=self.mask_interp)
            vecs.append(masked_mean(f, m_ds))
        lnp_deep = interp_mask(ln_shell, size=fdeep_ln.shape[2:], mode=self.mask_interp)
        vecs.append(self.attn_pool(fdeep_ln, lnp_deep))
        token_inputs.append(torch.cat(vecs, dim=1))

        # project to token_dim and apply type embedding; zero out absent tokens
        hs: List[torch.Tensor] = []
        for i in range(self.max_tokens):
            h = self.token_mlp[i](token_inputs[i]) + self.token_type[i].unsqueeze(0)
            if i > 0:
                absent = ~pres[:, i]
                if absent.any():
                    h = h.masked_fill(absent.unsqueeze(1), 0.0)
            hs.append(h)

        tok_img = torch.stack(hs, dim=1)  # (B,5,D)
        tok_img = torch.nan_to_num(tok_img, nan=0.0, posinf=0.0, neginf=0.0)

        # token dropout on ROI tokens (not GLOBAL)
        # NOTE: This is TRAINING regularization. It can temporarily drop tokens even if masks exist.
        if self.training and self.token_dropout > 0:
            pres2m = pres.clone()
            tok2m = tok_img
            for tok_i in (1, 2, 3, 4):
                if pres2m[:, tok_i].any():
                    drop = (torch.rand(B, device=x_img.device) < self.token_dropout) & pres2m[:, tok_i]
                    if drop.any():
                        pres2m[drop, tok_i] = False
                        tok2m = tok2m.clone()
                        tok2m[drop, tok_i, :] = 0.0
            pres, tok_img = pres2m, tok2m

        return tok_img, pres
