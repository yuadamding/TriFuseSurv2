"""Node-instance and topology utilities for TriFuseSurv2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import SimpleITK as sitk

from trifusesurv2.schema import NodeTopologySummary


@dataclass(frozen=True)
class NodeInstanceSummary:
    """Per-node summary extracted from a connected nodal component."""

    label_id: int
    voxel_count: int
    volume_mm3: float
    centroid_xyz_mm: tuple[float, float, float]
    bbox_xyzwh: tuple[int, int, int, int, int, int]
    laterality: str


def _mask_centroid_xyz_mm(mask_img: sitk.Image) -> Optional[tuple[float, float, float]]:
    mask = sitk.Cast(mask_img > 0, sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    labels = list(stats.GetLabels())
    if not labels:
        return None
    centroid = stats.GetCentroid(labels[0])
    return (float(centroid[0]), float(centroid[1]), float(centroid[2]))


def extract_node_instances(
    mask_img: sitk.Image,
    *,
    min_voxels: int = 10,
    midline_x_mm: Optional[float] = None,
    midline_tolerance_mm: float = 2.0,
) -> list[NodeInstanceSummary]:
    """Split a nodal mask into connected-component instances.

    `midline_x_mm` should come from a stable anatomical reference. If it is not
    provided, laterality is marked as `unknown` instead of guessing from a crop.
    """

    binary = sitk.Cast(mask_img > 0, sitk.sitkUInt8)
    cc = sitk.ConnectedComponent(binary)
    cc = sitk.RelabelComponent(cc, sortByObjectSize=True)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    spacing = mask_img.GetSpacing()
    voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])

    out: list[NodeInstanceSummary] = []
    for label in stats.GetLabels():
        voxel_count = int(stats.GetNumberOfPixels(label))
        if voxel_count < int(min_voxels):
            continue
        centroid = stats.GetCentroid(label)
        bbox = stats.GetBoundingBox(label)
        x_mm = float(centroid[0])
        if midline_x_mm is None:
            laterality = "unknown"
        elif x_mm < (float(midline_x_mm) - float(midline_tolerance_mm)):
            laterality = "left"
        elif x_mm > (float(midline_x_mm) + float(midline_tolerance_mm)):
            laterality = "right"
        else:
            laterality = "midline"
        out.append(
            NodeInstanceSummary(
                label_id=int(label),
                voxel_count=voxel_count,
                volume_mm3=float(voxel_count) * voxel_volume_mm3,
                centroid_xyz_mm=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                bbox_xyzwh=tuple(int(v) for v in bbox),
                laterality=laterality,
            )
        )
    return out


def summarize_node_topology(
    node_instances: Iterable[NodeInstanceSummary],
    *,
    pt_mask_img: Optional[sitk.Image] = None,
) -> NodeTopologySummary:
    """Build a compact nodal-topology summary token."""

    nodes = list(node_instances)
    if not nodes:
        return NodeTopologySummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    centroids = np.asarray([node.centroid_xyz_mm for node in nodes], dtype=np.float32)
    volumes = np.asarray([node.volume_mm3 for node in nodes], dtype=np.float32)
    centroid_mean = centroids.mean(axis=0, keepdims=True)
    centroid_spread = float(np.linalg.norm(centroids - centroid_mean, axis=1).mean()) if len(nodes) > 1 else 0.0
    laterality_known_flag = 1.0 if all(node.laterality != "unknown" for node in nodes) else 0.0
    laterality_set = {node.laterality for node in nodes if node.laterality in {"left", "right"}}
    bilateral_flag = 1.0 if laterality_known_flag > 0.5 and {"left", "right"}.issubset(laterality_set) else 0.0

    pt_centroid = _mask_centroid_xyz_mm(pt_mask_img) if pt_mask_img is not None else None
    if pt_centroid is None:
        min_pt_ln = 0.0
        mean_pt_ln = 0.0
    else:
        pt_centroid_arr = np.asarray(pt_centroid, dtype=np.float32)[None, :]
        dists = np.linalg.norm(centroids - pt_centroid_arr, axis=1)
        min_pt_ln = float(dists.min()) if dists.size else 0.0
        mean_pt_ln = float(dists.mean()) if dists.size else 0.0

    return NodeTopologySummary(
        node_count=float(len(nodes)),
        node_total_volume_mm3=float(volumes.sum()),
        node_largest_volume_mm3=float(volumes.max()),
        node_bilateral_flag=bilateral_flag,
        node_laterality_known_flag=laterality_known_flag,
        node_centroid_spread_mm=centroid_spread,
        pt_ln_min_distance_mm=min_pt_ln,
        pt_ln_mean_distance_mm=mean_pt_ln,
    )


def topology_summary_to_vector(summary: NodeTopologySummary) -> np.ndarray:
    """Convert topology summary to a float32 vector."""

    return np.asarray(summary.to_vector(), dtype=np.float32)
