"""Wrapper model bridging a contour-aware image backbone to the habitat-aligned head.

The established TriFuseSurv pipeline feeds raw CT images [B,1,D,H,W] through
``ContourAwareROITokenBackbone`` which internally produces habitat tokens
[B,N,out_dim] + presence [B,N].  ``HabitatAlignedSurvivalModel`` expects
pre-computed tokens.  This wrapper owns both and exposes a forward() signature
compatible with the existing training loop call pattern:

    logits = model(x_img, clinical, radiomics, mask_pt=..., mask_ln=..., ...)

Usage
-----
The backbone is injected as any ``nn.Module`` whose ``forward`` returns
``(tokens [B,N,D], presence [B,N])`` (or ``(tokens, presence, aux)`` when
``return_aux=True``).  It must also expose ``num_tokens: int`` and
``out_dim: int`` properties.  The concrete ``ContourAwareROITokenBackbone``
from ``trifusesurv2.models`` satisfies this contract.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn

from trifusesurv2.models.habitat_survival import HabitatAlignedSurvivalModel
from trifusesurv2.schema import (
    CLINICAL_TOKEN_GROUPS,
    IMAGE_HABITATS,
    RADIOLOGY_HABITATS,
)


@runtime_checkable
class ImageTokenBackbone(Protocol):
    """Structural protocol for contour-aware image backbones."""

    @property
    def num_tokens(self) -> int: ...  # noqa: E704

    @property
    def out_dim(self) -> int: ...  # noqa: E704

    def forward(
        self,
        x_img: torch.Tensor,
        *,
        mask_pt: Optional[torch.Tensor],
        mask_ln: Optional[torch.Tensor],
        voxel_spacing_dhw: Optional[torch.Tensor] = None,
        teacher_force_alpha: float,
        return_aux: bool,
        return_cam_features: bool = False,
    ) -> Any: ...  # noqa: E704


class ContourAwareHabitatSurvivalModel(nn.Module):
    """End-to-end model: CT image → backbone tokens → habitat-aligned survival.

    This replaces ``SwinUNETRTokenMoEDiscrete`` for the v2 pathway. It keeps the
    same backbone but replaces the late-fusion MoE gating with habitat-aligned
    cross-attention fusion and structured survival heads.

    Parameters
    ----------
    backbone : nn.Module
        Must satisfy ``ImageTokenBackbone`` protocol.
    num_time_bins : int
        Discrete time bins for survival heads.
    time_bin_width_days : float
        Width of each discrete time bin in days (for hazards_to_risk).
    radiomics_token_dim, clinical_token_dim : int
        Dimensions of per-group token vectors from the v2 encoders. 0 = disabled.
    node_token_dim, topology_dim : int
        Dimensions for node-set and topology tokens. 0 = disabled.
    model_dim, num_heads, transformer_layers, dropout :
        Forwarded to ``HabitatAlignedSurvivalModel``.
    """

    def __init__(
        self,
        *,
        backbone: nn.Module,
        num_time_bins: int,
        time_bin_width_days: float,
        radiomics_token_dim: int = 0,
        clinical_token_dim: int = 0,
        node_token_dim: int = 0,
        topology_dim: int = 0,
        model_dim: int = 256,
        num_heads: int = 8,
        transformer_layers: int = 2,
        dropout: float = 0.1,
        image_habitats: tuple[str, ...] = IMAGE_HABITATS,
        radiomics_habitats: tuple[str, ...] = RADIOLOGY_HABITATS,
        clinical_groups: "dict[str, tuple[str, ...]] | None" = None,
        habitat_clinical_context: "dict[str, tuple[str, ...]] | None" = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.model_version = "v2"
        self.image_encoder_mode = "contour_aware"
        self.num_time_bins = int(num_time_bins)
        self.time_bin_width_days = float(time_bin_width_days)

        from collections import OrderedDict
        clin_groups = OrderedDict(clinical_groups or CLINICAL_TOKEN_GROUPS)

        self.habitat_model = HabitatAlignedSurvivalModel(
            image_token_dim=int(backbone.out_dim),
            radiomics_token_dim=radiomics_token_dim,
            clinical_token_dim=clinical_token_dim,
            num_time_bins=num_time_bins,
            model_dim=model_dim,
            num_heads=num_heads,
            transformer_layers=transformer_layers,
            dropout=dropout,
            node_token_dim=node_token_dim,
            topology_dim=topology_dim,
            image_habitats=image_habitats,
            radiomics_habitats=radiomics_habitats,
            clinical_groups=clin_groups,
            habitat_clinical_context=habitat_clinical_context,
        )

    def hazards_to_risk(self, hazards_logits: torch.Tensor, horizon_days: float) -> torch.Tensor:
        """Convert discrete-time hazard logits to cumulative risk at a horizon.

        Compatible with the ``evaluate_model`` / ``predict_risk_scores`` helpers
        in the training pipeline.
        """
        bw = self.time_bin_width_days
        if bw <= 0.0:
            raise ValueError(f"time_bin_width_days must be > 0, got {bw}")
        t = float(horizon_days)
        B, K = hazards_logits.shape[0], hazards_logits.shape[1]
        if K <= 0 or t <= 0.0:
            return hazards_logits.new_zeros((B,), dtype=torch.float32)
        hazards = torch.sigmoid(hazards_logits.float()).clamp(1e-7, 1.0 - 1e-7)
        max_covered = K * bw
        if t >= max_covered:
            logS = torch.log1p(-hazards).sum(dim=1)
            return (1.0 - torch.exp(logS)).clamp(0.0, 1.0)
        k = int(math.floor(t / bw))
        k = max(0, min(k, K - 1))
        within = t - (k * bw)
        frac = within / bw
        log1m = torch.log1p(-hazards)
        cum = torch.cumsum(log1m, dim=1)
        if k == 0:
            logS_t = log1m[:, 0] * float(t / bw)
        else:
            logS_prev = cum[:, k - 1]
            logS_t = logS_prev + log1m[:, k] * float(frac)
        return (1.0 - torch.exp(logS_t)).clamp(0.0, 1.0)

    @staticmethod
    def _apply_image_habitat_dropout(
        image_tokens: torch.Tensor,
        image_presence: Optional[torch.Tensor],
        *,
        p: float,
        keep_global: bool,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Structured dropout over image habitat tokens for v2 robustness."""
        prob = max(0.0, min(1.0, float(p)))
        if not training or prob <= 0.0:
            if image_presence is None:
                image_presence = torch.ones(
                    image_tokens.shape[:2],
                    dtype=torch.bool,
                    device=image_tokens.device,
                )
            return image_tokens, image_presence

        presence = image_presence.to(torch.bool) if image_presence is not None else torch.ones(
            image_tokens.shape[:2],
            dtype=torch.bool,
            device=image_tokens.device,
        )
        keep = torch.rand(presence.shape, device=presence.device) >= prob
        if keep_global and keep.shape[1] > 0:
            keep[:, 0] = True

        dropped_presence = presence & keep
        had_any = presence.any(dim=1)
        empty_after = had_any & (~dropped_presence.any(dim=1))
        if empty_after.any():
            first_present = presence.to(torch.int64).argmax(dim=1)
            rows = torch.nonzero(empty_after, as_tuple=False).flatten()
            dropped_presence[rows, first_present[rows]] = True

        dropped_tokens = image_tokens * dropped_presence.unsqueeze(-1).to(image_tokens.dtype)
        return dropped_tokens, dropped_presence

    def forward(
        self,
        x_img: torch.Tensor,
        clinical_tokens: Optional[torch.Tensor] = None,
        radiomics_tokens: Optional[torch.Tensor] = None,
        *,
        mask_pt: Optional[torch.Tensor] = None,
        mask_ln: Optional[torch.Tensor] = None,
        voxel_spacing_dhw: Optional[torch.Tensor] = None,
        teacher_force_alpha: float = 0.0,
        clinical_presence: Optional[torch.Tensor] = None,
        radiomics_presence: Optional[torch.Tensor] = None,
        node_tokens: Optional[torch.Tensor] = None,
        node_presence: Optional[torch.Tensor] = None,
        topology_token: Optional[torch.Tensor] = None,
        topology_presence: Optional[torch.Tensor] = None,
        return_gate: bool = False,
        return_aux: bool = False,
        return_attention: bool = False,
        return_cam_features: bool = False,
        image_habitat_dropout_p: float = 0.0,
        keep_global_image_habitat: bool = True,
    ):
        """Forward pass: raw CT → backbone → habitat model → survival logits.

        The first three positional arguments (x_img, clinical_tokens,
        radiomics_tokens) follow the same ordering as the v1 model so that
        the training loop call pattern ``model(x, clin, rad, ...)`` works
        with minimal changes.

        Returns
        -------
        logits : dict[str, Tensor]
            Per-endpoint hazard logits {endpoint: [B, num_time_bins]}.
        If ``return_aux``, also returns a dict of intermediate representations.

        Raises
        ------
        NotImplementedError
            If ``return_gate=True``.  This model uses habitat-aligned cross-
            attention, not MoE gating.  Use ``return_aux=True`` to inspect
            habitat tokens and attention patterns instead.
        """
        request_bb_aux = bool(return_aux or return_cam_features)
        backbone_kwargs: dict[str, Any] = {
            "mask_pt": mask_pt,
            "mask_ln": mask_ln,
            "voxel_spacing_dhw": voxel_spacing_dhw,
            "teacher_force_alpha": float(teacher_force_alpha),
            "return_aux": request_bb_aux,
        }
        if return_cam_features:
            backbone_kwargs["return_cam_features"] = True
        bb_out = self.backbone(x_img, **backbone_kwargs)
        if request_bb_aux:
            image_tokens, image_presence, bb_aux = bb_out
        else:
            image_tokens, image_presence = bb_out
            bb_aux = None

        image_tokens, image_presence = self._apply_image_habitat_dropout(
            image_tokens,
            image_presence,
            p=float(image_habitat_dropout_p),
            keep_global=bool(keep_global_image_habitat),
            training=bool(self.training),
        )

        habitat_result = self.habitat_model(
            image_tokens=image_tokens,
            image_presence=image_presence,
            radiomics_tokens=radiomics_tokens,
            radiomics_presence=radiomics_presence,
            clinical_tokens=clinical_tokens,
            clinical_presence=clinical_presence,
            node_tokens=node_tokens,
            node_presence=node_presence,
            topology_token=topology_token,
            topology_presence=topology_presence,
            return_aux=return_aux,
            return_attention=return_attention,
        )

        if return_aux:
            logits, hab_aux = habitat_result
            aux = {**hab_aux}
            if bb_aux is not None:
                aux["backbone_aux"] = bb_aux
        else:
            logits = habitat_result
            aux = None

        if return_gate:
            raise NotImplementedError(
                "ContourAwareHabitatSurvivalModel uses habitat-aligned cross-attention, "
                "not MoE gating. Gate weights are not available. Use return_aux=True "
                "to inspect habitat_tokens and attention patterns instead."
            )

        if aux is not None:
            return logits, aux
        return logits

    def forward_from_image_tokens(
        self,
        *,
        image_tokens: torch.Tensor,
        image_presence: torch.Tensor,
        clinical_tokens: Optional[torch.Tensor] = None,
        radiomics_tokens: Optional[torch.Tensor] = None,
        clinical_presence: Optional[torch.Tensor] = None,
        radiomics_presence: Optional[torch.Tensor] = None,
        node_tokens: Optional[torch.Tensor] = None,
        node_presence: Optional[torch.Tensor] = None,
        topology_token: Optional[torch.Tensor] = None,
        topology_presence: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        return_attention: bool = False,
    ):
        """Forward the v2 habitat model from precomputed image tokens.

        This helper is intended for explainability methods such as Grad-CAM
        that need one backbone pass, then repeated downstream passes with
        selected image-token gradient paths isolated or ablated.
        """
        return self.habitat_model(
            image_tokens=image_tokens,
            image_presence=image_presence,
            radiomics_tokens=radiomics_tokens,
            radiomics_presence=radiomics_presence,
            clinical_tokens=clinical_tokens,
            clinical_presence=clinical_presence,
            node_tokens=node_tokens,
            node_presence=node_presence,
            topology_token=topology_token,
            topology_presence=topology_presence,
            return_aux=return_aux,
            return_attention=return_attention,
        )
