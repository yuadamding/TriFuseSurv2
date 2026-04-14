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
            return TokenBlock(tokens=block.tokens.to(device), presence=presence)

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
        )
