"""Lesion-independent physical sampling followed only by integer crop/padding.

The caller must supply a complete physical lattice policy. No spacing is selected
here. Source FOV enclosure depends on acquisition geometry, never on contours.
Each input is resampled once on that full grid before any crop is selected.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import re
from typing import Any, Mapping

import numpy as np
import SimpleITK as sitk

from .compartments import ReferenceGeometry, assert_same_geometry, mask_checksum

SOURCE_CENTER_RETENTION_RULE = "source voxel centers in half-open output voxel-cell FOV; resampling loss recorded separately"


@dataclass(frozen=True)
class CommonGridPolicy:
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    direction: tuple[float, ...]
    units: str = "mm"
    coordinate_system: str = "LPS"

    def __post_init__(self):
        ReferenceGeometry((1, 1, 1), self.spacing_xyz, self.origin_xyz, self.direction,
                          self.units, self.coordinate_system)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CommonGridPolicy":
        fields = {"schema_version", "spacing_xyz", "origin_xyz", "direction", "units", "coordinate_system"}
        if set(value) != fields or value["schema_version"] != 1:
            raise ValueError("Common-grid configuration requires exactly schema_version=1, spacing_xyz, origin_xyz, direction, units and coordinate_system.")
        return cls(tuple(float(x) for x in value["spacing_xyz"]), tuple(float(x) for x in value["origin_xyz"]),
                   tuple(float(x) for x in value["direction"]), str(value["units"]), str(value["coordinate_system"]))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "spacing_xyz": list(map(float, self.spacing_xyz)),
                "origin_xyz": list(map(float, self.origin_xyz)), "direction": list(map(float, self.direction)),
                "units": self.units, "coordinate_system": self.coordinate_system}

    @property
    def policy_id(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass
class CommonGridResult:
    images: dict[str, sitk.Image]
    mask_images: dict[str, sitk.Image]
    masks: dict[str, np.ndarray]
    geometry: ReferenceGeometry
    full_geometry: ReferenceGeometry
    source_fov_mask: np.ndarray
    metadata: dict[str, Any]


def _policy(value):
    return value if isinstance(value, CommonGridPolicy) else CommonGridPolicy.from_dict(value)


def _reference(value):
    return value if isinstance(value, ReferenceGeometry) else ReferenceGeometry.from_dict(value)


def _snap_integer(values):
    values = np.asarray(values, dtype=float)
    return np.where(np.abs(values - np.rint(values)) < 1e-8, np.rint(values), values)


def _indices_xyz(points_lps, geometry):
    return ((np.asarray(points_lps) - np.asarray(geometry.origin_xyz)) @ np.asarray(geometry.direction).reshape(3, 3)) / np.asarray(geometry.spacing_xyz)


def full_grid_geometry(reference_geometry, policy) -> tuple[ReferenceGeometry, tuple[int, int, int]]:
    """Enclose source voxel-cell FOV on the fixed lattice, including rotated FOVs."""
    reference, policy = _reference(reference_geometry), _policy(policy)
    corners_dhw = np.asarray(list(itertools.product(*[(-.5, size - .5) for size in reference.shape_dhw])))
    corners = _indices_xyz(reference.points_lps(corners_dhw), policy)
    lower = np.floor(_snap_integer(corners.min(axis=0) + .5)).astype(np.int64)
    upper = np.ceil(_snap_integer(corners.max(axis=0) - .5)).astype(np.int64)
    upper = np.maximum(upper, lower)
    origin = np.asarray(policy.origin_xyz) + np.asarray(policy.direction).reshape(3, 3) @ (lower * policy.spacing_xyz)
    geometry = ReferenceGeometry(tuple(map(int, (upper - lower + 1)[::-1])), policy.spacing_xyz,
                                 tuple(origin), policy.direction, policy.units, policy.coordinate_system)
    return geometry, tuple(map(int, lower))


def _array_image(array, geometry):
    image = sitk.GetImageFromArray(np.asarray(array))
    image.SetSpacing(geometry.spacing_xyz)
    image.SetOrigin(geometry.origin_xyz)
    image.SetDirection(geometry.direction)
    return image


def _resample(image, geometry, *, mask, default):
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(list(reversed(geometry.shape_dhw)))
    resampler.SetOutputSpacing(geometry.spacing_xyz)
    resampler.SetOutputOrigin(geometry.origin_xyz)
    resampler.SetOutputDirection(geometry.direction)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if mask else sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(default))
    resampler.SetOutputPixelType(sitk.sitkUInt8 if mask else sitk.sitkFloat32)
    return sitk.GetArrayFromImage(resampler.Execute(image))


def _source_center_indices(mask, reference, full):
    coordinates = np.argwhere(mask)
    # Work in chunks to avoid several simultaneous N-by-3 floating point arrays.
    for offset in range(0, len(coordinates), 262144):
        points = reference.points_lps(coordinates[offset:offset + 262144])
        indices = _indices_xyz(points, full)[:, ::-1]
        yield np.floor(_snap_integer(indices + .5)).astype(np.int64)


def _padding(start, shape, full_shape):
    return np.minimum(np.maximum(-start, 0), shape), np.minimum(np.maximum(start + shape - full_shape, 0), shape)


def _copy_canvas(array, start, shape, *, default=0):
    result = np.full(tuple(shape), default, dtype=array.dtype)
    lower = np.maximum(start, 0)
    upper = np.minimum(start + shape, np.asarray(array.shape))
    if np.all(upper > lower):
        src = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
        dst = tuple(slice(int(a), int(b)) for a, b in zip(lower - start, upper - start))
        result[dst] = array[src]
    return result


def sample_common_grid(images, masks, reference_geometry, policy, output_shape_dhw, *,
                       crop_start_dhw=None, context_margin_mm=0.0, reference_region_ids=None,
                       intensity_default_values=None) -> CommonGridResult:
    """Map full acquisition once; retain source centers and all mapped supports.

    ``crop_start_dhw`` is an integer offset relative to the full source grid, and
    may be negative to request padding. Crop overflow fails instead of resizing.
    Source-center retention is distinct from resampling loss of a small region.
    """
    reference, policy = _reference(reference_geometry), _policy(policy)
    shape_raw = np.asarray(output_shape_dhw)
    if shape_raw.shape != (3,) or not np.isfinite(shape_raw).all() or np.any(shape_raw <= 0) or np.any(shape_raw != np.floor(shape_raw)):
        raise ValueError("output_shape_dhw requires three positive integers.")
    shape = shape_raw.astype(np.int64)
    if not images:
        raise ValueError("At least one source intensity image is required.")
    if not np.isfinite(context_margin_mm) or context_margin_mm < 0:
        raise ValueError("context_margin_mm must be finite and nonnegative.")
    full, lattice_start = full_grid_geometry(reference, policy)
    source_masks = {}
    for name, mask in masks.items():
        array = np.asarray(mask)
        if array.shape != reference.shape_dhw or not np.isfinite(array).all() or not np.isin(array, [0, 1]).all():
            raise ValueError(f"{name}: source support must be binary on the reference grid.")
        source_masks[name] = array.astype(bool)
    if reference_region_ids is not None and set(reference_region_ids) != set(source_masks):
        raise ValueError("Reference region identities must name every supplied source compartment exactly once.")
    defaults = {name: float((intensity_default_values or {}).get(name, 0.0)) for name in images}
    if not np.isfinite(list(defaults.values())).all():
        raise ValueError("Intensity defaults must be finite.")
    # Every interpolation sees the same full-source geometry, independent of ROI.
    full_images = {}
    for name, image in images.items():
        assert_same_geometry(reference, image)
        if not np.isfinite(sitk.GetArrayViewFromImage(image)).all():
            raise ValueError(f"{name}: source intensity image contains nonfinite values.")
        full_images[name] = _resample(image, full, mask=False, default=defaults[name])
    full_fov = _resample(_array_image(np.ones(reference.shape_dhw, dtype=np.uint8), reference), full,
                         mask=True, default=0).astype(bool)
    full_masks = {name: _resample(_array_image(array.astype(np.uint8), reference), full, mask=True, default=0).astype(bool) & full_fov
                  for name, array in source_masks.items()}
    lower, upper = np.asarray(full.shape_dhw), np.zeros(3, dtype=np.int64)
    has_support = False
    for name, array in source_masks.items():
        for coordinates in _source_center_indices(array, reference, full):
            if len(coordinates):
                lower = np.minimum(lower, coordinates.min(axis=0))
                upper = np.maximum(upper, coordinates.max(axis=0) + 1)
                has_support = True
        coordinates = np.argwhere(full_masks[name])
        if len(coordinates):
            lower = np.minimum(lower, coordinates.min(axis=0))
            upper = np.maximum(upper, coordinates.max(axis=0) + 1)
            has_support = True
    if not has_support:
        lower = (np.asarray(full.shape_dhw) - 1) // 2
        upper = lower + 1
    margin = np.ceil(float(context_margin_mm) / np.asarray(full.spacing_dhw)).astype(np.int64)
    lower, upper = lower - margin, upper + margin
    if np.any(upper - lower > shape):
        raise ValueError(f"Common-grid canvas overflow: required DHW {(upper-lower).tolist()} exceeds requested {shape.tolist()}; choose a larger canvas, never resize.")
    if crop_start_dhw is None:
        start = np.floor((lower + upper - shape) / 2).astype(np.int64)
    else:
        raw_start = np.asarray(crop_start_dhw)
        if raw_start.shape != (3,) or not np.isfinite(raw_start).all() or np.any(raw_start != np.floor(raw_start)):
            raise ValueError("crop_start_dhw must contain three integer lattice offsets.")
        start = raw_start.astype(np.int64)
        if np.any(start > lower) or np.any(start + shape < upper):
            raise ValueError("Requested integer crop would lose required source centers, support or context.")
    origin = full.points_lps(start)
    geometry = ReferenceGeometry(tuple(map(int, shape)), full.spacing_xyz, tuple(origin), full.direction)
    pad_before, pad_after = _padding(start, shape, np.asarray(full.shape_dhw))
    output_images = {name: _array_image(_copy_canvas(array, start, shape, default=defaults[name]), geometry)
                     for name, array in full_images.items()}
    output_masks = {name: _copy_canvas(array, start, shape).astype(bool) for name, array in full_masks.items()}
    output_fov = _copy_canvas(full_fov, start, shape).astype(bool)
    measurements = {}
    for name, array in source_masks.items():
        retained = sum(int(np.all((coordinates >= start) & (coordinates < start + shape), axis=1).sum())
                       for coordinates in _source_center_indices(array, reference, full))
        reference_count, full_count, image_count = int(array.sum()), int(full_masks[name].sum()), int(output_masks[name].sum())
        if retained != reference_count or image_count != full_count:
            raise ValueError(f"{name}: integer crop loses reference centers or mapped support.")
        reference_volume = reference_count * reference.voxel_volume_mm3
        image_volume = image_count * geometry.voxel_volume_mm3
        measurements[name] = {
            "reference_region_id": (reference_region_ids or {}).get(name),
            "reference_mask_sha256": mask_checksum(array),
            "image_mask_sha256": mask_checksum(output_masks[name]),
            "reference_voxels": reference_count, "retained_reference_voxels": retained,
            "retained_reference_fraction": retained / reference_count if reference_count else 1.0,
            "full_grid_voxels": full_count, "image_voxels": image_count,
            "reference_volume_mm3": reference_volume, "full_grid_volume_mm3": full_count * full.voxel_volume_mm3,
            "image_volume_mm3": image_volume,
            "image_to_reference_volume_ratio": image_volume / reference_volume if reference_count else None,
            "disappeared_on_full_grid": bool(reference_count and not full_count),
        }
    metadata = {
        "policy": "common_grid", "policy_version": 1, "configuration": policy.to_dict(), "policy_id": policy.policy_id,
        "source_geometry": reference.to_dict(), "full_grid_geometry": full.to_dict(),
        "full_grid_lattice_start_xyz": list(lattice_start), "image_geometry": geometry.to_dict(),
        "crop_start_dhw": start.tolist(), "output_shape_dhw": shape.tolist(),
        "pad_before_dhw": pad_before.tolist(), "pad_after_dhw": pad_after.tolist(),
        "context_margin_mm": float(context_margin_mm), "required_bounds_dhw": [lower.tolist(), upper.tolist()],
        "empty_source_support": not has_support, "regions": measurements,
        "intensity_interpolator": "linear_float32_once_on_full_source_grid",
        "support_interpolator": "nearest_neighbor_once_on_full_source_grid",
        "intensity_default_values": defaults,
        "source_center_retention_rule": SOURCE_CENTER_RETENTION_RULE,
        "source_fov": {"full_grid_voxels": int(full_fov.sum()), "image_voxels": int(output_fov.sum()),
                       "image_mask_sha256": mask_checksum(output_fov), "padded_support_is_zero": True},
    }
    return CommonGridResult(output_images, {name: _array_image(array.astype(np.uint8), geometry) for name, array in output_masks.items()},
                            output_masks, geometry, full, output_fov, metadata)


def validate_sampling_metadata(manifest):
    """Validate a persisted common-grid policy and its crop/source accounting."""
    metadata = manifest["sampling"]
    if metadata.get("policy") != "common_grid" or metadata.get("policy_version") != 1:
        raise ValueError("Unsupported common-grid sampling metadata.")
    policy = CommonGridPolicy.from_dict(metadata["configuration"])
    if metadata["policy_id"] != policy.policy_id:
        raise ValueError("Common-grid policy checksum mismatch.")
    reference = ReferenceGeometry.from_dict(manifest["reference_geometry"])
    assert_same_geometry(reference, metadata["source_geometry"])
    full, lattice_start = full_grid_geometry(reference, policy)
    assert_same_geometry(full, metadata["full_grid_geometry"])
    if metadata["full_grid_lattice_start_xyz"] != list(lattice_start):
        raise ValueError("Full-grid lattice offset differs from source geometry and policy.")
    raw_start = np.asarray(metadata["crop_start_dhw"])
    if raw_start.shape != (3,) or not np.isfinite(raw_start).all() or np.any(raw_start != np.floor(raw_start)):
        raise ValueError("Persisted crop offset is not an integer lattice offset.")
    start = raw_start.astype(np.int64)
    image = ReferenceGeometry.from_dict(manifest["image"]["geometry"])
    expected = ReferenceGeometry(image.shape_dhw, full.spacing_xyz, tuple(full.points_lps(start)), full.direction)
    assert_same_geometry(expected, image)
    assert_same_geometry(image, metadata["image_geometry"])
    if list(image.shape_dhw) != metadata["output_shape_dhw"]:
        raise ValueError("Sampling canvas shape differs from image geometry.")
    before, after = _padding(start, np.asarray(image.shape_dhw), np.asarray(full.shape_dhw))
    if before.tolist() != metadata["pad_before_dhw"] or after.tolist() != metadata["pad_after_dhw"]:
        raise ValueError("Padding declaration differs from integer crop geometry.")
    bounds = np.asarray(metadata["required_bounds_dhw"])
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds != np.floor(bounds)) or np.any(bounds[1] <= bounds[0]):
        raise ValueError("Required crop bounds must be integer intervals.")
    if np.any(start > bounds[0]) or np.any(start + image.shape_dhw < bounds[1]):
        raise ValueError("Common-grid canvas does not contain declared required support bounds.")
    if metadata.get("intensity_interpolator") != "linear_float32_once_on_full_source_grid" or metadata.get("support_interpolator") != "nearest_neighbor_once_on_full_source_grid":
        raise ValueError("Unsupported common-grid interpolation contract.")
    if metadata.get("source_center_retention_rule") != SOURCE_CENTER_RETENTION_RULE:
        raise ValueError("Unsupported common-grid source-center retention rule.")
    margin = metadata.get("context_margin_mm")
    if not isinstance(margin, (int, float)) or not np.isfinite(margin) or margin < 0:
        raise ValueError("Persisted context margin must be finite and nonnegative.")
    defaults = metadata.get("intensity_default_values")
    channels = {name for name in ("image", "radiomics_image") if name in manifest}
    if not isinstance(defaults, Mapping) or not channels.issubset(defaults) or not defaults:
        raise ValueError("Persisted intensity defaults must cover every bound intensity channel.")
    if any(not isinstance(value, (int, float)) or not np.isfinite(value) for value in defaults.values()):
        raise ValueError("Persisted intensity defaults must be finite.")
    fov = metadata["source_fov"]
    fov_counts = {}
    limits = {"full_grid_voxels": math.prod(full.shape_dhw),
              "image_voxels": math.prod((np.asarray(image.shape_dhw) - before - after).tolist())}
    for name, limit in limits.items():
        count = fov.get(name)
        if isinstance(count, bool) or not isinstance(count, (int, float)) or not np.isfinite(count) or count != math.floor(count) or not 0 <= count <= limit:
            raise ValueError("Persisted source-FOV counts must be nonnegative integers within the unpadded grid.")
        fov_counts[name] = int(count)
    if fov_counts["image_voxels"] > fov_counts["full_grid_voxels"] or fov.get("padded_support_is_zero") is not True:
        raise ValueError("Persisted source-FOV crop/padding accounting is inconsistent.")
    if not isinstance(fov.get("image_mask_sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", fov["image_mask_sha256"]) is None:
        raise ValueError("Persisted source-FOV support must declare a SHA256 checksum.")
    if set(metadata["regions"]) != set(manifest["regions"]):
        raise ValueError("Common-grid measurements must declare every manifest region.")
    for name, record in manifest["regions"].items():
        measured = metadata["regions"][name]
        if measured["reference_region_id"] != record["region_id"] or measured["reference_mask_sha256"] != record["mask_sha256"]:
            raise ValueError(f"{name}: common-grid reference region identity differs.")
        if measured["image_mask_sha256"] != record["image_mask_sha256"]:
            raise ValueError(f"{name}: common-grid image support checksum differs.")
        if measured["reference_voxels"] != record["voxel_count"] or measured["retained_reference_voxels"] != record["voxel_count"] or measured["retained_reference_fraction"] != 1.0:
            raise ValueError(f"{name}: common-grid crop did not retain every reference center.")
        if measured["image_voxels"] != measured["full_grid_voxels"]:
            raise ValueError(f"{name}: common-grid crop did not retain every mapped voxel.")
        counts = np.asarray([measured[key] for key in ("reference_voxels", "retained_reference_voxels", "full_grid_voxels", "image_voxels")])
        if not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts != np.floor(counts)):
            raise ValueError(f"{name}: common-grid support counts must be nonnegative integers.")
        if measured["image_voxels"] > fov_counts["image_voxels"]:
            raise ValueError(f"{name}: mapped support exceeds the declared available source FOV.")
        reference_volume = record["voxel_count"] * reference.voxel_volume_mm3
        if not np.isclose(measured["reference_volume_mm3"], reference_volume):
            raise ValueError(f"{name}: common-grid reference volume is inconsistent.")
        volume = measured["image_voxels"] * image.voxel_volume_mm3
        if not np.isclose(volume, measured["image_volume_mm3"]) or not np.isclose(volume, measured["full_grid_volume_mm3"]):
            raise ValueError(f"{name}: common-grid image volume is inconsistent.")
        ratio = volume / reference_volume if reference_volume else None
        if (ratio is None) != (measured["image_to_reference_volume_ratio"] is None) or (ratio is not None and not np.isclose(ratio, measured["image_to_reference_volume_ratio"])):
            raise ValueError(f"{name}: common-grid volume ratio is inconsistent.")
        if bool(measured["disappeared_on_full_grid"]) != bool(record["voxel_count"] and not measured["full_grid_voxels"]):
            raise ValueError(f"{name}: common-grid disappearance flag is inconsistent.")
        crop = record["crop_completeness"]
        if crop["retained_reference_voxels"] != measured["retained_reference_voxels"] or not np.isclose(crop["image_volume_mm3"], volume):
            raise ValueError(f"{name}: crop report differs from common-grid accounting.")
