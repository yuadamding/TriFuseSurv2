"""Identity and deterministic availability rule for the specified F3 control.

The metadata functions deliberately have no tensor dependency. The intervention
adds no parameter and consumes no random number; it changes Boolean local
radiomics edges while retaining each key's anatomical identity and projection.
"""

from __future__ import annotations


ROUTING_INTERVENTION_ID = "anatomical_mismatch_availability_cycle_v1"
ROUTING_DESIGN_SHA256 = "c9c30676ca2059ead03f4ec7584fe0a688e7b7c9e916bd02ccc6956072ea66df"
ROUTING_EXPORT_LABEL = "F3_mismatch_availability_cycle_v1"
REGIONAL_ORDER = ("pt_intra", "pt_peri", "ln_intra", "ln_peri")


def validate_routing_intervention(value: str | None, *, fusion_mode: str = "F3") -> str | None:
    """Accept the explicit versioned F3 intervention or its inactive default."""
    if value is None:
        return None
    if value != ROUTING_INTERVENTION_ID:
        raise ValueError(f"Unknown routing_intervention: {value!r}")
    if str(fusion_mode).upper() != "F3":
        raise ValueError("The availability-cycle routing intervention requires fusion_mode=F3.")
    return value


def routing_intervention_metadata(value: str | None) -> dict[str, str]:
    """Only an active intervention adds fields to historical F3 contracts."""
    validate_routing_intervention(value)
    if value is None:
        return {}
    return {
        "routing_intervention": ROUTING_INTERVENTION_ID,
        "routing_intervention_design_sha256": ROUTING_DESIGN_SHA256,
        "routing_intervention_export_label": ROUTING_EXPORT_LABEL,
    }


def availability_cycle_assignment(image_presence, radiomics_presence):
    """Return the next canonical member of each joint-availability class.

    Both inputs are actual Boolean prelocal masks of shape ``[B, 4]`` in
    ``REGIONAL_ORDER``. The result is a bijection in every row. A singleton
    wraps to itself. All operations remain on the masks' device; there are no
    data-dependent CPU transfers, random draws or feature/outcome inputs.
    """
    import torch

    if (image_presence.dtype != torch.bool or radiomics_presence.dtype != torch.bool
            or image_presence.ndim != 2 or image_presence.shape[1] != 4
            or tuple(radiomics_presence.shape) != tuple(image_presence.shape)
            or image_presence.device != radiomics_presence.device):
        raise ValueError("Routing presence must be Boolean [B,4] masks on the same device.")
    signature = image_presence.to(torch.int64) * 2 + radiomics_presence.to(torch.int64)
    indices = torch.arange(4, device=image_presence.device)
    # For each query inspect the next 1,2,3,4 canonical positions. The query
    # itself is last and guarantees a match for a singleton class.
    candidates = (indices[:, None] + indices[None, :] + 1) % 4
    same_class = signature[:, candidates] == signature[:, :, None]
    first = same_class.to(torch.int64).argmax(dim=-1)
    return candidates[None].expand(image_presence.shape[0], -1, -1).gather(
        2, first[:, :, None]
    ).squeeze(-1)
