"""TriFuseSurv2: habitat-aligned, node-aware OPSCC survival modeling."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from trifusesurv2.schema import (
    CLINICAL_TOKEN_GROUPS,
    DEFAULT_HABITAT_CLINICAL_CONTEXT,
    ENDPOINT_MAP,
    FUTURE_IMAGE_HABITATS_MM,
    IMAGE_HABITATS,
    PROGNOSTIC_CLINICAL_TOKEN_GROUPS,
    RADIOLOGY_HABITATS,
    SURVIVAL_ENDPOINTS,
    TREATMENT_AWARE_HABITAT_CLINICAL_CONTEXT,
    TREATMENT_AWARE_CLINICAL_TOKEN_GROUPS,
)

__all__ = [
    "CLINICAL_TOKEN_GROUPS",
    "ContourAwareHabitatSurvivalModel",
    "DEFAULT_HABITAT_CLINICAL_CONTEXT",
    "ENDPOINT_MAP",
    "FUTURE_IMAGE_HABITATS_MM",
    "HabitatAlignedSurvivalModel",
    "HabitatBatch",
    "HabitatRadiomicsTokenEncoder",
    "IMAGE_HABITATS",
    "ImageTokenBackbone",
    "PROGNOSTIC_CLINICAL_TOKEN_GROUPS",
    "PTNodeCrossAttention",
    "RADIOLOGY_HABITATS",
    "SemanticClinicalTokenEncoder",
    "SurvivalTargets",
    "SURVIVAL_ENDPOINTS",
    "TREATMENT_AWARE_HABITAT_CLINICAL_CONTEXT",
    "TREATMENT_AWARE_CLINICAL_TOKEN_GROUPS",
    "TokenBlock",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "HabitatBatch": ("trifusesurv2.data.batch", "HabitatBatch"),
    "SurvivalTargets": ("trifusesurv2.data.batch", "SurvivalTargets"),
    "TokenBlock": ("trifusesurv2.data.batch", "TokenBlock"),
    "SemanticClinicalTokenEncoder": ("trifusesurv2.encoders.clinical", "SemanticClinicalTokenEncoder"),
    "HabitatRadiomicsTokenEncoder": ("trifusesurv2.encoders.radiomics", "HabitatRadiomicsTokenEncoder"),
    "HabitatAlignedSurvivalModel": ("trifusesurv2.models.habitat_survival", "HabitatAlignedSurvivalModel"),
    "PTNodeCrossAttention": ("trifusesurv2.models.habitat_survival", "PTNodeCrossAttention"),
    "ContourAwareHabitatSurvivalModel": ("trifusesurv2.models.contour_habitat_survival", "ContourAwareHabitatSurvivalModel"),
    "ImageTokenBackbone": ("trifusesurv2.models.contour_habitat_survival", "ImageTokenBackbone"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
