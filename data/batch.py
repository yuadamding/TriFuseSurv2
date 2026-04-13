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


@dataclass
class HabitatBatch:
    """Single model batch for habitat-aligned survival."""

    image: TokenBlock
    radiomics: Optional[TokenBlock] = None
    clinical: Optional[TokenBlock] = None
    nodes: Optional[TokenBlock] = None
    topology: Optional[TokenBlock] = None
    survival: Optional[SurvivalTargets] = None

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
