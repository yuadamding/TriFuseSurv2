"""Structured batch containers for TriFuseSurv2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TokenBlock:
    """Token matrix with optional presence mask."""

    tokens: torch.Tensor
    presence: Optional[torch.Tensor] = None
    region_ids: Optional[tuple[tuple[str, ...], ...]] = None


@dataclass
class SurvivalTargets:
    """Multitask discrete-time survival targets."""

    times: torch.Tensor
    events: torch.Tensor


def _validate_token_block(block: TokenBlock, name: str, *, expected_n: Optional[int] = None) -> None:
    """Validate token shape and presence mask consistency."""
    t = block.tokens
    if t.dim() != 3:
        raise ValueError(f"{name}.tokens must be [B, N, D], got shape {tuple(t.shape)}")
    if expected_n is not None and t.shape[1] != expected_n:
        raise ValueError(f"{name}.tokens expected N={expected_n}, got {t.shape[1]}")
    if block.region_ids is not None:
        if len(block.region_ids) != t.shape[0] or any(len(row) != t.shape[1] for row in block.region_ids):
            raise ValueError(f"{name}.region_ids must match token batch/group shape")
    if block.presence is not None:
        p = block.presence
        if p.dim() != 2 or p.shape[0] != t.shape[0] or p.shape[1] != t.shape[1]:
            raise ValueError(
                f"{name}.presence shape {tuple(p.shape)} does not match "
                f"tokens shape {tuple(t.shape[:2])}"
            )


@dataclass
class HabitatBatch:
    """Single model batch for habitat-aligned survival."""

    image: TokenBlock
    radiomics: Optional[TokenBlock] = None
    clinical: Optional[TokenBlock] = None
    nodes: Optional[TokenBlock] = None
    topology: Optional[TokenBlock] = None
    survival: Optional[SurvivalTargets] = None
    verified_regions: bool = False

    def validate(
        self,
        *,
        num_image_habitats: Optional[int] = None,
        num_radiomics_habitats: Optional[int] = None,
        num_clinical_groups: Optional[int] = None,
    ) -> None:
        """Check shape consistency across all present modalities.

        Call this in the dataset, collate function, or model entry point
        so errors surface early and clearly.
        """
        B = self.image.tokens.shape[0]
        _validate_token_block(self.image, "image", expected_n=num_image_habitats)

        if self.radiomics is not None:
            _validate_token_block(self.radiomics, "radiomics", expected_n=num_radiomics_habitats)
            if self.radiomics.tokens.shape[0] != B:
                raise ValueError(f"radiomics batch size {self.radiomics.tokens.shape[0]} != image batch size {B}")

        if self.verified_regions and self.radiomics is not None:
            from trifusesurv2.schema import IMAGE_HABITATS, RADIOLOGY_HABITATS
            _validate_token_block(self.image, "image", expected_n=len(IMAGE_HABITATS))
            _validate_token_block(self.radiomics, "radiomics", expected_n=len(RADIOLOGY_HABITATS))
            if self.image.region_ids is None or self.radiomics.region_ids is None:
                raise ValueError("Verified fusion requires image and radiomics region identities.")
            for b in range(B):
                for j, name in enumerate(RADIOLOGY_HABITATS):
                    i = IMAGE_HABITATS.index(name)
                    image_id = self.image.region_ids[b][i]
                    if not image_id or image_id != self.radiomics.region_ids[b][j]:
                        raise ValueError(f"Image/radiomics correspondence mismatch for {name} in row {b}")

        if self.clinical is not None:
            _validate_token_block(self.clinical, "clinical", expected_n=num_clinical_groups)
            if self.clinical.tokens.shape[0] != B:
                raise ValueError(f"clinical batch size {self.clinical.tokens.shape[0]} != image batch size {B}")

        if self.nodes is not None:
            _validate_token_block(self.nodes, "nodes")
            if self.nodes.tokens.shape[0] != B:
                raise ValueError(f"nodes batch size {self.nodes.tokens.shape[0]} != image batch size {B}")

        if self.topology is not None:
            t = self.topology.tokens
            if t.dim() not in (2, 3):
                raise ValueError(f"topology.tokens must be [B, D] or [B, 1, D], got shape {tuple(t.shape)}")
            if t.shape[0] != B:
                raise ValueError(f"topology batch size {t.shape[0]} != image batch size {B}")
            if self.topology.presence is not None:
                p = self.topology.presence
                if p.shape[0] != B:
                    raise ValueError(
                        f"topology.presence batch size {p.shape[0]} != image batch size {B}"
                    )

        if self.survival is not None:
            if self.survival.times.shape[0] != B:
                raise ValueError(f"survival.times batch size {self.survival.times.shape[0]} != image batch size {B}")
            if self.survival.events.shape[0] != B:
                raise ValueError(f"survival.events batch size {self.survival.events.shape[0]} != image batch size {B}")

    def to(self, device: torch.device | str) -> "HabitatBatch":
        def move(block: Optional[TokenBlock]) -> Optional[TokenBlock]:
            if block is None:
                return None
            presence = block.presence.to(device) if block.presence is not None else None
            return TokenBlock(tokens=block.tokens.to(device), presence=presence, region_ids=block.region_ids)

        survival = None
        if self.survival is not None:
            survival = SurvivalTargets(
                times=self.survival.times.to(device),
                events=self.survival.events.to(device),
            )
        return HabitatBatch(
            image=move(self.image),
            radiomics=move(self.radiomics),
            clinical=move(self.clinical),
            nodes=move(self.nodes),
            topology=move(self.topology),
            survival=survival,
            verified_regions=self.verified_regions,
        )
