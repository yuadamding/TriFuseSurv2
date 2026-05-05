"""Attention export helpers for v2 habitat-aligned explanations."""

from __future__ import annotations

from typing import Any

import torch


def _tensor_to_list(x) -> Any:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x


def attention_aux_to_jsonable(aux: dict[str, Any]) -> dict[str, Any]:
    """Convert v2 attention aux tensors into JSON-safe nested structures."""

    out: dict[str, Any] = {}
    habitat_attention = aux.get("habitat_attention", {}) or {}
    out["habitat_attention"] = {}
    for habitat, item in habitat_attention.items():
        weights = item.get("weights")
        context_names = list(item.get("context_names", []))
        context_mask = item.get("context_mask")
        mean_by_context = None
        if torch.is_tensor(weights):
            w = weights.detach().cpu()
            # [B, H, 1, C] for per-habitat cross-attention.
            if w.numel() and w.dim() >= 4:
                mean_by_context = w.mean(dim=(0, 1, 2)).tolist()
        out["habitat_attention"][str(habitat)] = {
            "context_names": context_names,
            "weights": _tensor_to_list(weights),
            "mean_attention_by_context": mean_by_context,
            "context_mask": _tensor_to_list(context_mask),
        }

    pt_node = aux.get("pt_node_attention")
    if pt_node is None:
        out["pt_node_attention"] = None
    else:
        weights = pt_node.get("weights")
        mean_by_pt_and_context = None
        if torch.is_tensor(weights):
            w = weights.detach().cpu()
            # [B, H, N_pt, N_ctx].
            if w.numel() and w.dim() >= 4:
                mean_by_pt_and_context = w.mean(dim=(0, 1)).tolist()
        out["pt_node_attention"] = {
            "pt_habitat_names": list(pt_node.get("pt_habitat_names", [])),
            "node_context_names": list(pt_node.get("node_context_names", [])),
            "weights": _tensor_to_list(weights),
            "mean_attention_by_pt_and_context": mean_by_pt_and_context,
            "context_mask": _tensor_to_list(pt_node.get("context_mask")),
        }
    return out
