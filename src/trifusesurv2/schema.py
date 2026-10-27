"""Shared schema and semantic constants for TriFuseSurv2."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Sequence

SURVIVAL_ENDPOINTS: tuple[str, ...] = ("OS", "DSS", "DFS")

ENDPOINT_MAP: dict[str, tuple[str, str]] = {
    "OS": ("OS.TIME", "OS.EVENT"),
    "DSS": ("DSS.TIME", "DSS.EVENT"),
    "DFS": ("DFS.TIME", "DFS.EVENT"),
}

# Established image habitats from the current contour-aware backbone.
IMAGE_HABITATS: tuple[str, ...] = (
    "global",
    "pt_intra",
    "pt_peri",
    "ln_intra",
    "ln_peri",
    "shape_spatial",
)

# Future extension point: explicit multiscale peri habitats in millimeters.
FUTURE_IMAGE_HABITATS_MM: tuple[str, ...] = (
    "global",
    "pt_intra",
    "pt_peri_3mm",
    "pt_peri_6mm",
    "pt_peri_10mm",
    "ln_intra",
    "ln_peri_3mm",
    "ln_peri_10mm",
    "shape_spatial",
)

# Established radiomics habitats from the current package.
RADIOLOGY_HABITATS: tuple[str, ...] = (
    "pt_intra",
    "pt_peri",
    "ln_intra",
    "ln_peri",
)

# Region states describe anatomy/annotation, not padded-token availability.
REGION_STATUS_CODES: dict[str, int] = {"present": 0, "absent": 1, "missing": 2, "invalid": 3}
REGION_MANIFEST_VERSION = 1
PHYSICAL_GEOMETRY_FEATURES: tuple[str, ...] = (
    "pt_volume_mm3",
    "ln_volume_mm3",
    "pt_ln_centroid_distance_mm",
    "pt_ln_min_surface_distance_mm",
    "ln_voxel_rms_spread_mm",
    "ln_left_flag",
    "ln_right_flag",
    "ln_bilateral_flag",
    "node_count",
    "largest_node_volume_mm3",
)

# Order of the original descriptors; these are explicitly crop dependent.
CROP_GEOMETRY_FEATURES: tuple[str, ...] = (
    "crop_pt_voxel_fraction", "crop_ln_voxel_fraction",
    "crop_pt_peri_voxel_fraction", "crop_ln_peri_voxel_fraction",
    "crop_body_voxel_fraction", "crop_log1p_ln_to_pt_fraction_ratio",
    "crop_pt_centroid_z", "crop_pt_centroid_y", "crop_pt_centroid_x",
    "crop_ln_centroid_z", "crop_ln_centroid_y", "crop_ln_centroid_x",
    "crop_ln_minus_pt_z", "crop_ln_minus_pt_y", "crop_ln_minus_pt_x",
    "crop_pt_ln_centroid_distance", "pt_present", "ln_present", "both_present",
)

# Prognostic default: no treatment token unless explicitly requested.
PROGNOSTIC_CLINICAL_TOKEN_GROUPS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    [
        ("biology", ("HPV", "PATHOLOGY")),
        ("burden", ("T", "N", "M", "NSTAGE", "T_RAW", "N_RAW", "M_RAW", "NSTAGE_RAW")),
        ("host", ("AGE", "SEX", "RACE", "KFCF", "SMOKE", "ALCOHOL")),
    ]
)

TREATMENT_AWARE_CLINICAL_TOKEN_GROUPS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    [
        ("biology", ("HPV", "PATHOLOGY")),
        ("burden", ("T", "N", "M", "NSTAGE", "T_RAW", "N_RAW", "M_RAW", "NSTAGE_RAW")),
        ("host", ("AGE", "SEX", "RACE", "KFCF", "SMOKE", "ALCOHOL")),
        ("treatment", ("TX",)),
    ]
)

CLINICAL_TOKEN_GROUPS = PROGNOSTIC_CLINICAL_TOKEN_GROUPS

DEFAULT_HABITAT_CLINICAL_CONTEXT: Mapping[str, tuple[str, ...]] = {
    "global": ("biology", "burden", "host"),
    "pt_intra": ("biology", "burden"),
    "pt_peri": ("biology", "host"),
    "ln_intra": ("burden", "host"),
    "ln_peri": ("burden", "host"),
    "shape_spatial": ("burden", "host"),
}

TREATMENT_AWARE_HABITAT_CLINICAL_CONTEXT: Mapping[str, tuple[str, ...]] = {
    **DEFAULT_HABITAT_CLINICAL_CONTEXT,
    "global": ("biology", "burden", "host", "treatment"),
}

NODE_TOPOLOGY_FEATURES: tuple[str, ...] = (
    "node_count",
    "node_total_volume_mm3",
    "node_largest_volume_mm3",
    "node_bilateral_flag",
    "node_laterality_known_flag",
    "node_centroid_spread_mm",
    "pt_ln_distance_known_flag",
    "pt_ln_min_distance_mm",
    "pt_ln_mean_distance_mm",
)


@dataclass(frozen=True)
class TokenSpec:
    """Lightweight token metadata."""

    name: str
    source: str
    dim: int


@dataclass(frozen=True)
class NodeTopologySummary:
    """Compact nodal topology summary."""

    node_count: float
    node_total_volume_mm3: float
    node_largest_volume_mm3: float
    node_bilateral_flag: float
    node_laterality_known_flag: float
    node_centroid_spread_mm: float
    pt_ln_distance_known_flag: float
    pt_ln_min_distance_mm: float
    pt_ln_mean_distance_mm: float

    def to_vector(self) -> tuple[float, ...]:
        return (
            self.node_count,
            self.node_total_volume_mm3,
            self.node_largest_volume_mm3,
            self.node_bilateral_flag,
            self.node_laterality_known_flag,
            self.node_centroid_spread_mm,
            self.pt_ln_distance_known_flag,
            self.pt_ln_min_distance_mm,
            self.pt_ln_mean_distance_mm,
        )


def habitat_index_map(names: Sequence[str]) -> dict[str, int]:
    """Create a name->index map with stable first-seen behavior."""

    out: dict[str, int] = {}
    for idx, name in enumerate(names):
        out.setdefault(str(name), idx)
    return out
