"""Token ablation utilities for v2 habitat-aligned explanations."""

from __future__ import annotations

from typing import Optional

import torch

from trifusesurv2.explain.gradcam_v2_core import risk_vector


def _clone_presence(p: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if p is None else p.clone().to(torch.bool)


def _model_group_names(model, attr: str, count: int) -> tuple[str, ...]:
    habitat_model = getattr(model, "habitat_model", model)
    names = tuple(str(x) for x in getattr(habitat_model, attr, ()))
    if len(names) >= int(count):
        return names[: int(count)]
    fallback = tuple(f"{attr}_{i}" for i in range(len(names), int(count)))
    return names + fallback


@torch.no_grad()
def compute_v2_ablation_package(
    model,
    *,
    image_tokens: torch.Tensor,
    image_presence: torch.Tensor,
    clinical_tokens: Optional[torch.Tensor],
    clinical_presence: Optional[torch.Tensor],
    radiomics_tokens: Optional[torch.Tensor],
    radiomics_presence: Optional[torch.Tensor],
    node_tokens: Optional[torch.Tensor],
    node_presence: Optional[torch.Tensor],
    topology_token: Optional[torch.Tensor],
    topology_presence: Optional[torch.Tensor],
    endpoint: str,
    horizon_days: float,
) -> dict[str, float]:
    """Compute full-vs-ablated cumulative-risk deltas for v2 token groups."""

    logits = model.forward_from_image_tokens(
        image_tokens=image_tokens,
        image_presence=image_presence,
        clinical_tokens=clinical_tokens,
        clinical_presence=clinical_presence,
        radiomics_tokens=radiomics_tokens,
        radiomics_presence=radiomics_presence,
        node_tokens=node_tokens,
        node_presence=node_presence,
        topology_token=topology_token,
        topology_presence=topology_presence,
        return_aux=False,
    )
    risk_full = risk_vector(model, logits, endpoint=endpoint, horizon_days=horizon_days)
    out: dict[str, float] = {"risk_full": float(risk_full.detach().cpu()[0])}

    for idx, name in enumerate(_model_group_names(model, "image_habitats", image_tokens.shape[1])):
        tok = image_tokens.clone()
        pres = image_presence.clone().to(torch.bool)
        tok[:, idx, :] = 0.0
        pres[:, idx] = False
        logits_i = model.forward_from_image_tokens(
            image_tokens=tok,
            image_presence=pres,
            clinical_tokens=clinical_tokens,
            clinical_presence=clinical_presence,
            radiomics_tokens=radiomics_tokens,
            radiomics_presence=radiomics_presence,
            node_tokens=node_tokens,
            node_presence=node_presence,
            topology_token=topology_token,
            topology_presence=topology_presence,
            return_aux=False,
        )
        risk_without = float(risk_vector(model, logits_i, endpoint=endpoint, horizon_days=horizon_days).detach().cpu()[0])
        out[f"risk_without_image_{name}"] = risk_without
        out[f"delta_risk_without_image_{name}"] = out["risk_full"] - risk_without

    if radiomics_tokens is not None and radiomics_tokens.numel() > 0:
        for idx, name in enumerate(_model_group_names(model, "radiomics_habitats", radiomics_tokens.shape[1])):
            tok = radiomics_tokens.clone()
            pres = _clone_presence(radiomics_presence)
            tok[:, idx, :] = 0.0
            if pres is not None:
                pres[:, idx] = False
            logits_i = model.forward_from_image_tokens(
                image_tokens=image_tokens,
                image_presence=image_presence,
                clinical_tokens=clinical_tokens,
                clinical_presence=clinical_presence,
                radiomics_tokens=tok,
                radiomics_presence=pres,
                node_tokens=node_tokens,
                node_presence=node_presence,
                topology_token=topology_token,
                topology_presence=topology_presence,
                return_aux=False,
            )
            risk_without = float(risk_vector(model, logits_i, endpoint=endpoint, horizon_days=horizon_days).detach().cpu()[0])
            out[f"risk_without_radiomics_{name}"] = risk_without
            out[f"delta_risk_without_radiomics_{name}"] = out["risk_full"] - risk_without

    if clinical_tokens is not None and clinical_tokens.numel() > 0:
        for idx, name in enumerate(_model_group_names(model, "clinical_groups", clinical_tokens.shape[1])):
            tok = clinical_tokens.clone()
            pres = _clone_presence(clinical_presence)
            tok[:, idx, :] = 0.0
            if pres is not None:
                pres[:, idx] = False
            logits_i = model.forward_from_image_tokens(
                image_tokens=image_tokens,
                image_presence=image_presence,
                clinical_tokens=tok,
                clinical_presence=pres,
                radiomics_tokens=radiomics_tokens,
                radiomics_presence=radiomics_presence,
                node_tokens=node_tokens,
                node_presence=node_presence,
                topology_token=topology_token,
                topology_presence=topology_presence,
                return_aux=False,
            )
            risk_without = float(risk_vector(model, logits_i, endpoint=endpoint, horizon_days=horizon_days).detach().cpu()[0])
            out[f"risk_without_clinical_{name}"] = risk_without
            out[f"delta_risk_without_clinical_{name}"] = out["risk_full"] - risk_without

    if node_tokens is not None and node_tokens.numel() > 0:
        logits_i = model.forward_from_image_tokens(
            image_tokens=image_tokens,
            image_presence=image_presence,
            clinical_tokens=clinical_tokens,
            clinical_presence=clinical_presence,
            radiomics_tokens=radiomics_tokens,
            radiomics_presence=radiomics_presence,
            node_tokens=torch.zeros_like(node_tokens),
            node_presence=torch.zeros_like(node_presence).bool() if node_presence is not None else None,
            topology_token=topology_token,
            topology_presence=topology_presence,
            return_aux=False,
        )
        risk_without = float(risk_vector(model, logits_i, endpoint=endpoint, horizon_days=horizon_days).detach().cpu()[0])
        out["risk_without_node_tokens"] = risk_without
        out["delta_risk_without_node_tokens"] = out["risk_full"] - risk_without

    if topology_token is not None and topology_token.numel() > 0:
        logits_i = model.forward_from_image_tokens(
            image_tokens=image_tokens,
            image_presence=image_presence,
            clinical_tokens=clinical_tokens,
            clinical_presence=clinical_presence,
            radiomics_tokens=radiomics_tokens,
            radiomics_presence=radiomics_presence,
            node_tokens=node_tokens,
            node_presence=node_presence,
            topology_token=torch.zeros_like(topology_token),
            topology_presence=torch.zeros_like(topology_presence).bool() if topology_presence is not None else None,
            return_aux=False,
        )
        risk_without = float(risk_vector(model, logits_i, endpoint=endpoint, horizon_days=horizon_days).detach().cpu()[0])
        out["risk_without_topology"] = risk_without
        out["delta_risk_without_topology"] = out["risk_full"] - risk_without

    return out
