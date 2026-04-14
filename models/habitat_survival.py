"""Habitat-aligned, node-aware survival model for TriFuseSurv2."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn

from trifusesurv2.data.batch import HabitatBatch
from trifusesurv2.schema import (
    CLINICAL_TOKEN_GROUPS,
    DEFAULT_HABITAT_CLINICAL_CONTEXT,
    IMAGE_HABITATS,
    RADIOLOGY_HABITATS,
    SURVIVAL_ENDPOINTS,
    TREATMENT_AWARE_HABITAT_CLINICAL_CONTEXT,
    TREATMENT_AWARE_CLINICAL_TOKEN_GROUPS,
    habitat_index_map,
)


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute a stable masked mean over token dimension 1."""

    w = mask.to(tokens.dtype).unsqueeze(-1)
    denom = w.sum(dim=1).clamp(min=eps)
    return (tokens * w).sum(dim=1) / denom


class ResidualMLP(nn.Module):
    """Small residual MLP block."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class HabitatCrossAttentionBlock(nn.Module):
    """Cross-attend a habitat query to a small multimodal context set."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(model_dim)
        self.context_norm = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = ResidualMLP(model_dim, hidden_dim=2 * model_dim, dropout=dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(query).unsqueeze(1)
        kv = self.context_norm(context)
        safe_mask = context_mask.to(torch.bool)
        empty_rows = ~safe_mask.any(dim=1)
        if empty_rows.any():
            safe_mask = safe_mask.clone()
            kv = kv.clone()
            safe_mask[empty_rows, 0] = True
            kv[empty_rows, 0, :] = 0.0
        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=(~safe_mask))
        fused = query + attn_out.squeeze(1)
        return self.ffn(fused)


class PTNodeCrossAttention(nn.Module):
    """Bipartite cross-attention giving PT habitats direct access to node context.

    After per-habitat fusion, PT habitats (pt_intra, pt_peri) still lack direct
    node/topology information — they only meet it later in the global sequence
    encoder. This block injects node-set and topology context directly into PT
    habitat representations before the final encoder, strengthening the
    primary-nodal coupling signal.
    """

    def __init__(self, model_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(model_dim)
        self.context_norm = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = ResidualMLP(model_dim, hidden_dim=2 * model_dim, dropout=dropout)

    def forward(
        self,
        pt_tokens: torch.Tensor,
        pt_mask: torch.Tensor,
        node_context: torch.Tensor,
        node_context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attend PT habitat tokens to node-set context.

        Args:
            pt_tokens: [B, num_pt_habitats, model_dim] — fused PT tokens.
            pt_mask: [B, num_pt_habitats] — presence mask for PT habitats.
            node_context: [B, N_ctx, model_dim] — node tokens (+ optional topology).
            node_context_mask: [B, N_ctx] — presence mask for context tokens.

        Returns:
            Updated PT tokens [B, num_pt_habitats, model_dim].
        """
        B, N_pt, D = pt_tokens.shape
        q = self.query_norm(pt_tokens)
        kv = self.context_norm(node_context)

        safe_ctx_mask = node_context_mask.to(torch.bool)
        empty_ctx = ~safe_ctx_mask.any(dim=1)
        if empty_ctx.any():
            safe_ctx_mask = safe_ctx_mask.clone()
            kv = kv.clone()
            safe_ctx_mask[empty_ctx, 0] = True
            kv[empty_ctx, 0, :] = 0.0

        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=(~safe_ctx_mask))

        pt_mask_f = pt_mask.to(pt_tokens.dtype).unsqueeze(-1)  # [B, N_pt, 1]
        updated = pt_tokens + attn_out * pt_mask_f  # [B, N_pt, D]

        # Batched masked FFN: only run on present tokens to avoid spurious
        # gradients through LayerNorm bias for absent (zero) inputs.
        flat = updated.reshape(B * N_pt, D)
        flat_mask = pt_mask.reshape(B * N_pt).bool()
        out_flat = torch.zeros_like(flat)
        if flat_mask.any():
            out_flat[flat_mask] = self.ffn(flat[flat_mask])
        return out_flat.reshape(B, N_pt, D) * pt_mask_f


class StructuredSurvivalHeads(nn.Module):
    """Shared-trunk multitask endpoint heads.

    This is intentionally a conservative structured head: shared prognostic
    representation plus endpoint-specific residual logits. It is not yet a full
    competing-risk or multistate event model.
    """

    def __init__(self, model_dim: int, num_time_bins: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shared_logits = nn.Linear(hidden_dim, num_time_bins)
        self.endpoint_residuals = nn.ModuleDict(
            {
                endpoint: nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_time_bins),
                )
                for endpoint in SURVIVAL_ENDPOINTS
            }
        )

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        trunk = self.shared(latent)
        shared_logits = self.shared_logits(trunk)
        return {
            endpoint: shared_logits + residual_head(trunk)
            for endpoint, residual_head in self.endpoint_residuals.items()
        }


class HabitatAlignedSurvivalModel(nn.Module):
    """Habitat-aligned fusion model with optional node-set support.

    Defaults are aligned to what the established package can actually emit
    today: 5 image habitats and 4 radiomics habitats.
    """

    def __init__(
        self,
        *,
        image_token_dim: int,
        radiomics_token_dim: int,
        clinical_token_dim: int,
        num_time_bins: int,
        model_dim: int = 256,
        num_heads: int = 8,
        transformer_layers: int = 2,
        dropout: float = 0.1,
        node_token_dim: int = 0,
        topology_dim: int = 0,
        image_habitats: tuple[str, ...] = IMAGE_HABITATS,
        radiomics_habitats: tuple[str, ...] = RADIOLOGY_HABITATS,
        clinical_groups: OrderedDict[str, tuple[str, ...]] = CLINICAL_TOKEN_GROUPS,
        habitat_clinical_context: dict[str, tuple[str, ...]] | None = None,
    ):
        super().__init__()
        self.image_habitats = tuple(image_habitats)
        self.radiomics_habitats = tuple(radiomics_habitats)
        self.clinical_groups = tuple(clinical_groups.keys())
        default_clinical_context = (
            TREATMENT_AWARE_HABITAT_CLINICAL_CONTEXT
            if tuple(self.clinical_groups) == tuple(TREATMENT_AWARE_CLINICAL_TOKEN_GROUPS.keys())
            else DEFAULT_HABITAT_CLINICAL_CONTEXT
        )
        self.habitat_clinical_context = dict(habitat_clinical_context or default_clinical_context)

        self.image_index = habitat_index_map(self.image_habitats)
        self.radiomics_index = habitat_index_map(self.radiomics_habitats)
        self.clinical_index = habitat_index_map(self.clinical_groups)

        self.image_proj = nn.Linear(image_token_dim, model_dim)
        self.radiomics_proj = (
            nn.ModuleDict({name: nn.Linear(radiomics_token_dim, model_dim) for name in self.radiomics_habitats})
            if radiomics_token_dim > 0
            else None
        )
        self.clinical_proj = (
            nn.ModuleDict({name: nn.Linear(clinical_token_dim, model_dim) for name in self.clinical_groups})
            if clinical_token_dim > 0
            else None
        )
        self.node_proj = nn.Linear(node_token_dim, model_dim) if node_token_dim > 0 else None
        self.topology_proj = nn.Linear(topology_dim, model_dim) if topology_dim > 0 else None

        self.node_encoder = (
            nn.MultiheadAttention(model_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            if node_token_dim > 0
            else None
        )

        self._pt_habitat_indices = [
            i for i, h in enumerate(self.image_habitats) if h.startswith("pt_")
        ]
        self.pt_node_cross_attn = (
            PTNodeCrossAttention(model_dim=model_dim, num_heads=num_heads, dropout=dropout)
            if node_token_dim > 0
            else None
        )

        self.habitat_fusers = nn.ModuleDict(
            {
                habitat: HabitatCrossAttentionBlock(model_dim=model_dim, num_heads=num_heads, dropout=dropout)
                for habitat in self.image_habitats
            }
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=2 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.prognosis_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.normal_(self.prognosis_token, mean=0.0, std=0.02)

        self.output_norm = nn.LayerNorm(model_dim)
        self.survival_heads = StructuredSurvivalHeads(
            model_dim=model_dim,
            num_time_bins=num_time_bins,
            hidden_dim=model_dim,
            dropout=dropout,
        )

        self._init_weights()
        # Restore prognosis token init (the sweep above would have overwritten it)
        nn.init.normal_(self.prognosis_token, mean=0.0, std=0.02)

    def _init_weights(self) -> None:
        """Xavier uniform for Linear layers, ones/zeros for LayerNorm."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _project_optional_tokens(
        self,
        tokens: Optional[torch.Tensor],
        presence: Optional[torch.Tensor],
        proj: Optional[nn.Module],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if tokens is None:
            return None, None
        if proj is None:
            raise ValueError("Received optional tokens but the corresponding projection layer is disabled.")
        if tokens.dim() != 3:
            raise ValueError(f"Expected optional tokens with shape [B,N,D], got {tuple(tokens.shape)}")
        x = proj(tokens)
        if presence is None:
            presence = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        else:
            presence = presence.to(torch.bool)
            if tuple(presence.shape[:2]) != tuple(x.shape[:2]):
                raise ValueError(
                    f"Optional token presence shape {tuple(presence.shape)} does not match token shape {tuple(x.shape)}"
                )
        return x, presence

    def _project_grouped_tokens(
        self,
        tokens: Optional[torch.Tensor],
        presence: Optional[torch.Tensor],
        names: tuple[str, ...],
        proj_map: Optional[nn.ModuleDict],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if tokens is None:
            return None, None
        if proj_map is None:
            raise ValueError("Received grouped tokens but the corresponding projection layers are disabled.")
        if tokens.dim() != 3:
            raise ValueError(f"Expected grouped tokens with shape [B,N,D], got {tuple(tokens.shape)}")
        if tokens.shape[1] != len(names):
            raise ValueError(f"Expected {len(names)} grouped tokens for {names}, got {tokens.shape[1]}")
        out = []
        for idx, name in enumerate(names):
            out.append(proj_map[name](tokens[:, idx, :]))
        stacked = torch.stack(out, dim=1)
        if presence is None:
            presence = torch.ones(stacked.shape[:2], dtype=torch.bool, device=stacked.device)
        else:
            presence = presence.to(torch.bool)
            if tuple(presence.shape[:2]) != tuple(stacked.shape[:2]):
                raise ValueError(
                    f"Grouped token presence shape {tuple(presence.shape)} does not match token shape "
                    f"{tuple(stacked.shape)}"
                )
        return stacked, presence

    def _radiomics_index_for_habitat(self, habitat: str) -> Optional[int]:
        return self.radiomics_index.get(habitat, None)

    def _clinical_indices_for_habitat(self, habitat: str) -> list[int]:
        names = self.habitat_clinical_context.get(habitat, ())
        return [self.clinical_index[name] for name in names if name in self.clinical_index]

    def _validate_primary_inputs(
        self,
        image_tokens: torch.Tensor,
        image_presence: Optional[torch.Tensor],
    ) -> None:
        if image_tokens.dim() != 3:
            raise ValueError(f"Expected image_tokens [B,N,D], got shape {tuple(image_tokens.shape)}")
        if image_tokens.shape[1] != len(self.image_habitats):
            raise ValueError(
                f"Expected {len(self.image_habitats)} image habitats {self.image_habitats}, "
                f"got token count {image_tokens.shape[1]}"
            )
        if image_presence is not None and tuple(image_presence.shape[:2]) != tuple(image_tokens.shape[:2]):
            raise ValueError(
                f"image_presence shape {tuple(image_presence.shape)} does not match image_tokens "
                f"shape {tuple(image_tokens.shape)}"
            )

    @property
    def is_treatment_aware(self) -> bool:
        return tuple(self.clinical_groups) == tuple(TREATMENT_AWARE_CLINICAL_TOKEN_GROUPS.keys())

    def _encode_node_pool(
        self,
        node_tokens: Optional[torch.Tensor],
        node_presence: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if node_tokens is None or self.node_encoder is None:
            return None, None, None
        mask = node_presence.to(torch.bool) if node_presence is not None else torch.ones(
            node_tokens.shape[:2], dtype=torch.bool, device=node_tokens.device
        )
        pooled_mask = mask.any(dim=1)
        safe_tokens = node_tokens
        safe_mask = mask
        empty_rows = ~safe_mask.any(dim=1)
        if empty_rows.any():
            safe_tokens = safe_tokens.clone()
            safe_mask = safe_mask.clone()
            safe_mask[empty_rows, 0] = True
            safe_tokens[empty_rows, 0, :] = 0.0
        encoded, _ = self.node_encoder(safe_tokens, safe_tokens, safe_tokens, key_padding_mask=(~safe_mask))
        pooled = masked_mean(encoded, safe_mask)
        if empty_rows.any():
            encoded = encoded.clone()
            pooled = pooled.clone()
            encoded[empty_rows, :, :] = 0.0
            pooled[empty_rows, :] = 0.0
        return encoded, pooled, pooled_mask

    def forward(
        self,
        *,
        image_tokens: torch.Tensor,
        image_presence: Optional[torch.Tensor] = None,
        radiomics_tokens: Optional[torch.Tensor] = None,
        radiomics_presence: Optional[torch.Tensor] = None,
        clinical_tokens: Optional[torch.Tensor] = None,
        clinical_presence: Optional[torch.Tensor] = None,
        node_tokens: Optional[torch.Tensor] = None,
        node_presence: Optional[torch.Tensor] = None,
        topology_token: Optional[torch.Tensor] = None,
        topology_presence: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._validate_primary_inputs(image_tokens, image_presence)
        image = self.image_proj(image_tokens)
        image_mask = image_presence.to(torch.bool) if image_presence is not None else torch.ones(
            image.shape[:2], dtype=torch.bool, device=image.device
        )
        image = image * image_mask.unsqueeze(-1).to(image.dtype)

        radiomics, radiomics_mask = self._project_grouped_tokens(
            radiomics_tokens,
            radiomics_presence,
            self.radiomics_habitats,
            self.radiomics_proj,
        )
        clinical, clinical_mask = self._project_grouped_tokens(
            clinical_tokens,
            clinical_presence,
            self.clinical_groups,
            self.clinical_proj,
        )
        node, node_mask = self._project_optional_tokens(node_tokens, node_presence, self.node_proj)

        topology = None
        topology_mask = None
        if topology_token is not None:
            if self.topology_proj is None:
                raise ValueError("Received topology tokens but topology projection is disabled.")
            topology = self.topology_proj(topology_token)
            if topology.dim() == 2:
                topology = topology.unsqueeze(1)
            if topology_presence is None:
                topology_mask = torch.ones(topology.shape[:2], dtype=torch.bool, device=topology.device)
            else:
                topology_mask = topology_presence.to(torch.bool)
                if topology_mask.dim() == 1:
                    topology_mask = topology_mask.unsqueeze(1)
                if tuple(topology_mask.shape[:2]) != tuple(topology.shape[:2]):
                    raise ValueError(
                        f"Topology presence shape {tuple(topology_mask.shape)} does not match token shape "
                        f"{tuple(topology.shape)}"
                    )

        node_encoded, node_pooled, node_pooled_mask = self._encode_node_pool(node, node_mask)

        fused_habitats = []
        fused_presence = []
        for habitat_idx, habitat_name in enumerate(self.image_habitats):
            query = image[:, habitat_idx, :]

            context_parts = [query]
            context_masks = [image_mask[:, habitat_idx]]

            rad_idx = self._radiomics_index_for_habitat(habitat_name)
            if radiomics is not None and rad_idx is not None and rad_idx < radiomics.shape[1]:
                context_parts.append(radiomics[:, rad_idx, :])
                context_masks.append(radiomics_mask[:, rad_idx])

            if clinical is not None:
                for clin_idx in self._clinical_indices_for_habitat(habitat_name):
                    if clin_idx < clinical.shape[1]:
                        context_parts.append(clinical[:, clin_idx, :])
                        context_masks.append(clinical_mask[:, clin_idx])

            if topology is not None and (habitat_name.startswith("ln_") or habitat_name == "global"):
                context_parts.append(topology[:, 0, :])
                context_masks.append(topology_mask[:, 0])

            if node_pooled is not None and habitat_name.startswith("ln_"):
                context_parts.append(node_pooled)
                context_masks.append(node_pooled_mask)

            context = torch.stack(context_parts, dim=1)
            context_mask = torch.stack(context_masks, dim=1)
            fused = self.habitat_fusers[habitat_name](query, context, context_mask)
            fused = fused * image_mask[:, habitat_idx].unsqueeze(-1).to(fused.dtype)
            fused_habitats.append(fused)
            fused_presence.append(image_mask[:, habitat_idx])

        habitat_seq = torch.stack(fused_habitats, dim=1)
        habitat_mask = torch.stack(fused_presence, dim=1)

        if self.pt_node_cross_attn is not None and node_encoded is not None and self._pt_habitat_indices:
            pt_idx = self._pt_habitat_indices
            pt_tokens = habitat_seq[:, pt_idx, :]
            pt_mask = habitat_mask[:, pt_idx]

            ctx_parts = [node_encoded]
            ctx_masks = [node_mask]
            if topology is not None:
                ctx_parts.append(topology)
                ctx_masks.append(topology_mask)
            node_ctx = torch.cat(ctx_parts, dim=1)
            node_ctx_mask = torch.cat(ctx_masks, dim=1)

            pt_updated = self.pt_node_cross_attn(pt_tokens, pt_mask, node_ctx, node_ctx_mask)

            habitat_seq = habitat_seq.clone()
            for out_idx, hab_idx in enumerate(pt_idx):
                habitat_seq[:, hab_idx, :] = pt_updated[:, out_idx, :]

        seq_parts = [self.prognosis_token.expand(image.shape[0], -1, -1), habitat_seq]
        seq_masks = [
            torch.ones((image.shape[0], 1), dtype=torch.bool, device=image.device),
            habitat_mask,
        ]
        if node_encoded is not None:
            seq_parts.append(node_encoded)
            seq_masks.append(node_mask)
        if topology is not None:
            seq_parts.append(topology)
            seq_masks.append(topology_mask)

        seq = torch.cat(seq_parts, dim=1)
        seq_mask = torch.cat(seq_masks, dim=1)
        encoded = self.sequence_encoder(seq, src_key_padding_mask=(~seq_mask))
        latent = self.output_norm(encoded[:, 0, :])
        logits = self.survival_heads(latent)

        if not return_aux:
            return logits

        aux = {
            "latent": latent,
            "habitat_tokens": habitat_seq,
            "habitat_presence": habitat_mask,
            "node_tokens": node_encoded,
            "node_presence": node_mask,
            "topology_token": topology,
            "topology_presence": topology_mask,
        }
        return logits, aux

    def forward_batch(
        self,
        batch: HabitatBatch,
        *,
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Forward pass using the structured TriFuseSurv2 batch interface.

        Calls ``batch.validate()`` before forwarding to catch shape mismatches
        early rather than letting them surface as cryptic tensor errors.
        """
        batch.validate(
            num_image_habitats=len(self.image_habitats),
            num_radiomics_habitats=len(self.radiomics_habitats) if batch.radiomics is not None else None,
            num_clinical_groups=len(self.clinical_groups) if batch.clinical is not None else None,
        )
        return self.forward(
            image_tokens=batch.image.tokens,
            image_presence=batch.image.presence,
            radiomics_tokens=None if batch.radiomics is None else batch.radiomics.tokens,
            radiomics_presence=None if batch.radiomics is None else batch.radiomics.presence,
            clinical_tokens=None if batch.clinical is None else batch.clinical.tokens,
            clinical_presence=None if batch.clinical is None else batch.clinical.presence,
            node_tokens=None if batch.nodes is None else batch.nodes.tokens,
            node_presence=None if batch.nodes is None else batch.nodes.presence,
            topology_token=None if batch.topology is None else batch.topology.tokens,
            topology_presence=None if batch.topology is None else batch.topology.presence,
            return_aux=return_aux,
        )
