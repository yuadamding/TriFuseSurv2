"""Node-instance and topology utilities for TriFuseSurv2."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

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
    roi_source_name: str = ""


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
) -> tuple[list[NodeInstanceSummary], sitk.Image]:
    """Split a nodal mask into connected-component instances.

    Parameters
    ----------
    mask_img : sitk.Image
        Binary or integer nodal mask.
    min_voxels : int
        Minimum voxel count to retain a node.
    midline_x_mm : float or None
        Stable anatomical midline. If None, laterality is ``"unknown"``.
    midline_tolerance_mm : float
        Tolerance band around midline for the ``"midline"`` label.

    Returns
    -------
    nodes : list[NodeInstanceSummary]
    instance_label_map : sitk.Image
        Integer label map (same geometry as mask_img) with one label per node.
        Labels are sorted by component size (1 = largest).

    Notes
    -----
    To attach original RTSTRUCT ROI names, use ``map_roi_sources_by_overlap``
    after extraction.  Do not assume a positional correspondence between
    component labels and ROI ordering — ``RelabelComponent`` reorders by size.
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
    return out, cc


def map_roi_sources_by_overlap(
    nodes: list[NodeInstanceSummary],
    label_map: sitk.Image,
    roi_masks: dict[str, sitk.Image],
) -> list[NodeInstanceSummary]:
    """Annotate node instances with original ROI source names via voxel overlap.

    For each node (connected component in ``label_map``), find the ROI mask
    with the highest voxel overlap and assign its name as ``roi_source_name``.

    Parameters
    ----------
    nodes : list[NodeInstanceSummary]
        Nodes from ``extract_node_instances``.
    label_map : sitk.Image
        Integer label map from ``extract_node_instances``.
    roi_masks : dict[str, sitk.Image]
        Mapping from ROI name to its binary mask (same geometry as label_map).

    Returns
    -------
    list[NodeInstanceSummary]
        New node instances with ``roi_source_name`` filled in.
    """
    from dataclasses import replace

    if not roi_masks or not nodes:
        return list(nodes)

    label_arr = sitk.GetArrayFromImage(label_map)
    roi_arrays = {
        name: (sitk.GetArrayFromImage(mask) > 0).astype(np.uint8)
        for name, mask in roi_masks.items()
    }

    label_to_best_roi: dict[int, str] = {}
    for node in nodes:
        lid = node.label_id
        component_mask = (label_arr == lid)
        best_name = ""
        best_overlap = 0
        for roi_name, roi_arr in roi_arrays.items():
            overlap = int((component_mask & (roi_arr > 0)).sum())
            if overlap > best_overlap:
                best_overlap = overlap
                best_name = roi_name
        label_to_best_roi[lid] = best_name

    return [
        replace(node, roi_source_name=label_to_best_roi.get(node.label_id, ""))
        for node in nodes
    ]


def summarize_node_topology(
    node_instances: Iterable[NodeInstanceSummary],
    *,
    pt_mask_img: Optional[sitk.Image] = None,
) -> NodeTopologySummary:
    """Build a compact nodal-topology summary token."""

    nodes = list(node_instances)
    if not nodes:
        return NodeTopologySummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan, np.nan)

    centroids = np.asarray([node.centroid_xyz_mm for node in nodes], dtype=np.float32)
    volumes = np.asarray([node.volume_mm3 for node in nodes], dtype=np.float32)
    centroid_mean = centroids.mean(axis=0, keepdims=True)
    centroid_spread = float(np.linalg.norm(centroids - centroid_mean, axis=1).mean()) if len(nodes) > 1 else 0.0
    laterality_known_flag = 1.0 if all(node.laterality != "unknown" for node in nodes) else 0.0
    laterality_set = {node.laterality for node in nodes if node.laterality in {"left", "right"}}
    bilateral_flag = 1.0 if laterality_known_flag > 0.5 and {"left", "right"}.issubset(laterality_set) else 0.0

    pt_centroid = _mask_centroid_xyz_mm(pt_mask_img) if pt_mask_img is not None else None
    if pt_centroid is None:
        pt_ln_distance_known_flag = 0.0
        min_pt_ln = np.nan
        mean_pt_ln = np.nan
    else:
        pt_centroid_arr = np.asarray(pt_centroid, dtype=np.float32)[None, :]
        dists = np.linalg.norm(centroids - pt_centroid_arr, axis=1)
        pt_ln_distance_known_flag = 1.0 if dists.size else 0.0
        min_pt_ln = float(dists.min()) if dists.size else np.nan
        mean_pt_ln = float(dists.mean()) if dists.size else np.nan

    return NodeTopologySummary(
        node_count=float(len(nodes)),
        node_total_volume_mm3=float(volumes.sum()),
        node_largest_volume_mm3=float(volumes.max()),
        node_bilateral_flag=bilateral_flag,
        node_laterality_known_flag=laterality_known_flag,
        node_centroid_spread_mm=centroid_spread,
        pt_ln_distance_known_flag=pt_ln_distance_known_flag,
        pt_ln_min_distance_mm=min_pt_ln,
        pt_ln_mean_distance_mm=mean_pt_ln,
    )


def topology_summary_to_vector(summary: NodeTopologySummary, *, fill_missing: float = 0.0) -> np.ndarray:
    """Convert topology summary to a float32 vector.

    Missing scalar values are kept as NaN in the summary object itself so they
    are not conflated with true zeros. For model-safe vector export, callers can
    use the default `fill_missing=0.0` together with the explicit known/unknown
    flags already included in the summary.
    """

    vec = np.asarray(summary.to_vector(), dtype=np.float32)
    return np.nan_to_num(vec, nan=float(fill_missing), posinf=float(fill_missing), neginf=float(fill_missing))


def topology_summary_feature_presence(summary: NodeTopologySummary) -> np.ndarray:
    """Return a feature-level presence mask for the topology summary vector."""

    vec = np.asarray(summary.to_vector(), dtype=np.float32)
    return np.isfinite(vec).astype(np.float32)


def _safe_float(v: Any) -> Any:
    """Convert numpy/float to JSON-safe value, preserving NaN as None."""
    if isinstance(v, (np.floating, float)):
        if np.isnan(v) or np.isinf(v):
            return None
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v


def serialize_node_topology(
    nodes: list[NodeInstanceSummary],
    summary: NodeTopologySummary,
    *,
    image_metadata: Optional[dict[str, Any]] = None,
    crop_transform: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Serialize node instances and topology summary to a JSON-compatible dict.

    Parameters
    ----------
    nodes : list[NodeInstanceSummary]
        Per-node metadata from ``extract_node_instances``.
    summary : NodeTopologySummary
        Aggregate topology from ``summarize_node_topology``.
    image_metadata : dict, optional
        Original image geometry: spacing, origin, direction, size.
    crop_transform : dict, optional
        Crop parameters used during preprocessing (bbox, margin, target spacing).

    Returns
    -------
    dict
        JSON-serializable dict containing all node topology information.
    """
    node_dicts = []
    for node in nodes:
        d = asdict(node)
        d["centroid_xyz_mm"] = list(d["centroid_xyz_mm"])
        d["bbox_xyzwh"] = list(d["bbox_xyzwh"])
        node_dicts.append(d)

    summary_dict = {}
    for field_name, val in zip(
        ("node_count", "node_total_volume_mm3", "node_largest_volume_mm3",
         "node_bilateral_flag", "node_laterality_known_flag",
         "node_centroid_spread_mm", "pt_ln_distance_known_flag",
         "pt_ln_min_distance_mm", "pt_ln_mean_distance_mm"),
        summary.to_vector(),
    ):
        summary_dict[field_name] = _safe_float(val)

    out: dict[str, Any] = {
        "node_instances": node_dicts,
        "topology_summary": summary_dict,
    }
    if image_metadata is not None:
        out["image_metadata"] = {
            k: [float(x) for x in v] if isinstance(v, (list, tuple)) else v
            for k, v in image_metadata.items()
        }
    if crop_transform is not None:
        out["crop_transform"] = crop_transform
    return out


def write_node_topology_json(
    path: str | Path,
    nodes: list[NodeInstanceSummary],
    summary: NodeTopologySummary,
    **kwargs: Any,
) -> None:
    """Write node topology to a JSON file."""
    data = serialize_node_topology(nodes, summary, **kwargs)
    Path(path).write_text(json.dumps(data, indent=2))


def read_node_topology_json(path: str | Path) -> dict[str, Any]:
    """Read a node topology JSON file."""
    return json.loads(Path(path).read_text())
