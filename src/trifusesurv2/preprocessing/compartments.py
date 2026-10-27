"""Authoritative, spacing-aware anatomical compartments on qualified grids.

Arrays use D/H/W (z/y/x); physical coordinates use SimpleITK LPS x/y/z in
declared millimeters. Distance is Euclidean distance between voxel centers,
not a subvoxel surface reconstruction. Grid direction must be orthonormal:
rotations/reflections are supported; shear must first be resampled onto an
explicitly qualified reference grid. Spacing alone never qualifies geometry.

Shells use ``0 < distance_to_lesion <= width_mm``, exclude every available PT
and LN interior, and intersect a supplied, documented valid-tissue mask. PT/LN
shell overlap is allowed and measured. Unknown exclusion anatomy invalidates
the shell. There is no replacement of an empty shell with tumor interior.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from trifusesurv2.schema import (
    PHYSICAL_GEOMETRY_FEATURES,
    RADIOLOGY_HABITATS,
    REGION_MANIFEST_VERSION,
    REGION_STATUS_CODES,
)


@dataclass(frozen=True)
class ReferenceGeometry:
    """An orthogonal physical sampling grid, with explicit units and orientation."""

    shape_dhw: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    units: str = "mm"
    coordinate_system: str = "LPS"

    def __post_init__(self):
        if len(self.shape_dhw) != 3 or any(int(x) != x or x <= 0 for x in self.shape_dhw):
            raise ValueError("Reference shape must contain three positive integer dimensions.")
        if len(self.spacing_xyz) != 3 or not np.isfinite(self.spacing_xyz).all() or min(self.spacing_xyz) <= 0:
            raise ValueError("Reference spacing must contain three finite positive values.")
        if len(self.origin_xyz) != 3 or not np.isfinite(self.origin_xyz).all():
            raise ValueError("Reference origin must contain three finite values.")
        if len(self.direction) != 9 or not np.isfinite(self.direction).all():
            raise ValueError("Reference direction must be a finite 3x3 matrix.")
        direction = np.asarray(self.direction).reshape(3, 3)
        if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-6, rtol=0.0):
            raise ValueError("Reference direction must be orthonormal; resample sheared grids before generating regions.")
        if self.units != "mm" or self.coordinate_system != "LPS":
            raise ValueError("Reference geometry requires declared mm units and LPS coordinates.")

    @property
    def spacing_dhw(self) -> tuple[float, float, float]:
        return tuple(reversed(self.spacing_xyz))

    @property
    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.spacing_xyz))

    @classmethod
    def from_image(cls, image) -> "ReferenceGeometry":
        if int(image.GetDimension()) != 3 or int(image.GetNumberOfComponentsPerPixel()) != 1:
            raise ValueError("Reference image must be scalar and three dimensional.")
        return cls(
            shape_dhw=tuple(reversed(image.GetSize())),
            spacing_xyz=tuple(image.GetSpacing()),
            origin_xyz=tuple(image.GetOrigin()),
            direction=tuple(image.GetDirection()),
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "ReferenceGeometry":
        return cls(
            shape_dhw=tuple(record["shape_dhw"]), spacing_xyz=tuple(record["spacing_xyz"]),
            origin_xyz=tuple(record["origin_xyz"]), direction=tuple(record["direction"]),
            units=str(record["units"]), coordinate_system=str(record["coordinate_system"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape_dhw": list(self.shape_dhw), "spacing_xyz": list(self.spacing_xyz),
            "origin_xyz": list(self.origin_xyz), "direction": list(self.direction),
            "units": self.units, "coordinate_system": self.coordinate_system,
        }

    def points_lps(self, indices_dhw: np.ndarray) -> np.ndarray:
        points_xyz = np.asarray(indices_dhw, dtype=np.float64)[..., ::-1] * np.asarray(self.spacing_xyz)
        return points_xyz @ np.asarray(self.direction).reshape(3, 3).T + np.asarray(self.origin_xyz)


def _geometry(value) -> ReferenceGeometry:
    if isinstance(value, ReferenceGeometry):
        return value
    if isinstance(value, Mapping):
        return ReferenceGeometry.from_dict(value)
    return ReferenceGeometry.from_image(value)


def assert_same_geometry(reference, other, *, atol: float = 1e-5) -> None:
    """Reject silent alignment by array shape alone (tolerance is in mm/cosines)."""
    reference, other = _geometry(reference), _geometry(other)
    if reference.shape_dhw != other.shape_dhw:
        raise ValueError("Reference and input image shapes differ.")
    for name in ("spacing_xyz", "origin_xyz", "direction"):
        if not np.allclose(getattr(reference, name), getattr(other, name), atol=atol, rtol=0.0):
            raise ValueError(f"Reference and input image {name} differ; explicit resampling is required.")


def _binary(mask, *, shape=None, name="mask") -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 3 or (shape is not None and tuple(array.shape) != tuple(shape)):
        raise ValueError(f"{name} must be a 3D array on the declared reference grid.")
    if not np.isfinite(array).all() or not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must contain finite binary values; threshold contours explicitly first.")
    return np.ascontiguousarray(array, dtype=np.bool_)


def mask_checksum(mask: np.ndarray) -> str:
    """SHA-256 of shape, canonical uint8 binary dtype, and C-order support bytes."""
    array = _binary(mask).astype(np.uint8)
    header = json.dumps({"shape_dhw": list(array.shape), "dtype": "uint8", "order": "C"}, sort_keys=True)
    return hashlib.sha256(header.encode("utf-8") + b"\n" + array.tobytes(order="C")).hexdigest()


def physical_shell(mask, *, spacing_dhw, width_mm, excluded=None, valid_tissue=None) -> np.ndarray:
    """Binary Euclidean exterior shell; callers separately qualify full geometry.

    This low-level routine has no affine or acquisition information. Use
    ``build_compartments`` for qualified, provenance-bound scientific inputs.
    """
    mask = _binary(mask)
    spacing = np.asarray(spacing_dhw, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("spacing_dhw must contain three finite positive values.")
    if not np.isfinite(width_mm) or float(width_mm) < 0:
        raise ValueError("Shell width must be finite and nonnegative.")
    excluded = mask if excluded is None else _binary(excluded, shape=mask.shape, name="excluded") | mask
    valid = np.ones(mask.shape, dtype=bool) if valid_tissue is None else _binary(valid_tissue, shape=mask.shape, name="valid_tissue")
    if not mask.any() or float(width_mm) == 0:
        return np.zeros_like(mask)
    distance = distance_transform_edt(~mask, sampling=spacing)
    return (distance > 0.0) & (distance <= float(width_mm)) & (~excluded) & valid


def _source_value(source: str | Mapping[str, str], lesion: str) -> str:
    value = source.get(lesion, "") if isinstance(source, Mapping) else source
    if not str(value).strip():
        raise ValueError(f"A declared {lesion} contour source/version is required.")
    return str(value).strip()


def region_identity(record: Mapping[str, Any], *, mask_sha256=None, geometry=None) -> str:
    """Bind a region's anatomy, source, acquisition, grid and exact support.

    Supplying output support/grid creates a distinct ``image_region_id`` after
    crop/resampling, while the original ``region_id`` remains reference provenance.
    """
    identity = {key: record[key] for key in (
        "patient_id", "scan_id", "lesion_id", "compartment_id", "contour_source",
        "contour_version", "shell_definition", "status", "support_valid",
    )}
    identity["mask_sha256"] = record["mask_sha256"] if mask_sha256 is None else str(mask_sha256)
    identity["reference_geometry"] = record["reference_geometry"] if geometry is None else _geometry(geometry).to_dict()
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def physical_geometry_checksum(physical: Mapping[str, Any]) -> str:
    """Bind measured values and their calculation/source provenance together."""
    payload = {key: value for key, value in physical.items() if key != "provenance_sha256"}
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _reference_boundary_truncated(mask: np.ndarray, geometry: ReferenceGeometry, width_mm: float) -> bool:
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return False
    spacing = np.asarray(geometry.spacing_dhw)
    low = (coordinates.min(axis=0) + 0.5) * spacing
    high = (np.asarray(mask.shape) - coordinates.max(axis=0) - 0.5) * spacing
    return bool(np.any(low <= width_mm) or np.any(high <= width_mm))


def _physical_geometry(pt, ln, geometry, *, midline_lps_x=None):
    values = dict.fromkeys(PHYSICAL_GEOMETRY_FEATURES, 0.0)
    valid = dict.fromkeys(PHYSICAL_GEOMETRY_FEATURES, False)
    centroids = {}
    for name, mask in (("pt", pt), ("ln", ln)):
        if mask is None:
            centroids[name] = None
            continue
        values[f"{name}_volume_mm3"] = float(mask.sum()) * geometry.voxel_volume_mm3
        valid[f"{name}_volume_mm3"] = True
        points = geometry.points_lps(np.argwhere(mask))
        centroids[name] = points.mean(axis=0).tolist() if len(points) else None
        if name == "ln" and len(points):
            values["ln_voxel_rms_spread_mm"] = float(np.sqrt(np.mean(np.sum((points - points.mean(axis=0)) ** 2, axis=1))))
            valid["ln_voxel_rms_spread_mm"] = True
            if midline_lps_x is not None:
                left = bool(np.any(points[:, 0] > float(midline_lps_x)))
                right = bool(np.any(points[:, 0] < float(midline_lps_x)))
                for feature, result in (("ln_left_flag", left), ("ln_right_flag", right), ("ln_bilateral_flag", left and right)):
                    values[feature], valid[feature] = float(result), True
    if centroids.get("pt") is not None and centroids.get("ln") is not None:
        values["pt_ln_centroid_distance_mm"] = float(np.linalg.norm(np.subtract(centroids["pt"], centroids["ln"])))
        valid["pt_ln_centroid_distance_mm"] = True
        if np.any(pt & ln):
            distance = 0.0
        else:
            pt_surface = pt & (~binary_erosion(pt, border_value=0))
            ln_surface = ln & (~binary_erosion(ln, border_value=0))
            distance = float(distance_transform_edt(~pt_surface, sampling=geometry.spacing_dhw)[ln_surface].min())
        values["pt_ln_min_surface_distance_mm"] = distance
        valid["pt_ln_min_surface_distance_mm"] = True
    return {
        "names": list(PHYSICAL_GEOMETRY_FEATURES),
        "values": [values[name] for name in PHYSICAL_GEOMETRY_FEATURES],
        "valid": [valid[name] for name in PHYSICAL_GEOMETRY_FEATURES],
        "centroids_lps_mm": centroids,
        "midline_lps_x_mm": None if midline_lps_x is None else float(midline_lps_x),
        "distance_discretization": "Euclidean distance between 6-connected boundary voxel centers; overlapping interiors have distance zero",
        "node_instances": "unavailable; aggregate connected components are not counted as anatomical nodes",
    }


def build_compartments(
    pt, ln, valid_tissue, geometry, *, patient_id: str, scan_id: str,
    contour_source: str | Mapping[str, str], contour_version: str | Mapping[str, str],
    valid_tissue_source: str, shell_width_mm: float = 10.0, midline_lps_x: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    geometry = _geometry(geometry)
    for name, value in (("patient_id", patient_id), ("scan_id", scan_id), ("valid_tissue_source", valid_tissue_source)):
        if not str(value).strip():
            raise ValueError(f"{name} must be explicitly declared.")
    if float(shell_width_mm) != 10.0:
        raise ValueError("The qualified initial compartment definition uses one fixed 10 mm shell.")
    if midline_lps_x is not None and not np.isfinite(midline_lps_x):
        raise ValueError("An anatomical midline must be finite when supplied.")
    body = _binary(valid_tissue, shape=geometry.shape_dhw, name="valid_tissue")
    pt = None if pt is None else _binary(pt, shape=geometry.shape_dhw, name="PT")
    ln = None if ln is None else _binary(ln, shape=geometry.shape_dhw, name="LN")
    empty = np.zeros(geometry.shape_dhw, dtype=bool)
    excluded = (empty if pt is None else pt) | (empty if ln is None else ln)
    masks, records = {}, {}
    exclusion_known = pt is not None and ln is not None
    for habitat in RADIOLOGY_HABITATS:
        lesion = "pt" if habitat.startswith("pt_") else "ln"
        source_mask = pt if lesion == "pt" else ln
        is_peri = habitat.endswith("_peri")
        status = "present"
        if source_mask is None:
            support = empty.copy()
            status = "missing"
        elif not source_mask.any():
            # An explicitly absent lesion has no exterior shell, regardless of
            # whether a different lesion's annotation is available.
            support = empty.copy()
            status = "absent"
        elif is_peri and not exclusion_known:
            support = empty.copy()
            status = "invalid"
        elif is_peri:
            support = physical_shell(source_mask, spacing_dhw=geometry.spacing_dhw, width_mm=shell_width_mm,
                                     excluded=excluded, valid_tissue=body)
        else:
            support = source_mask.copy()
        if status == "present" and not support.any():
            status = "absent"
        support_valid = status in {"present", "absent"}
        definition = {
            "kind": "exterior_euclidean" if is_peri else "annotated_interior",
            "width_mm": float(shell_width_mm) if is_peri else 0.0,
            "distance_discretization": "voxel_center",
            "exclude_interiors": ["pt_intra", "ln_intra"] if is_peri else [],
            "shell_overlap_rule": "allow_and_measure",
            "valid_tissue_source": str(valid_tissue_source),
            "valid_tissue_mask_sha256": mask_checksum(body) if is_peri else None,
            "all_tumor_exclusion_known": exclusion_known if is_peri else None,
        }
        record = {
            "patient_id": str(patient_id), "scan_id": str(scan_id),
            "lesion_id": "primary" if lesion == "pt" else "aggregate_nodal",
            "compartment_id": habitat, "contour_source": _source_value(contour_source, lesion),
            "contour_version": _source_value(contour_version, lesion),
            "reference_geometry": geometry.to_dict(), "shell_definition": definition,
            "mask_sha256": mask_checksum(support), "voxel_count": int(support.sum()),
            "physical_volume_mm3": float(support.sum()) * geometry.voxel_volume_mm3,
            "status": status, "presence": bool(support.any()), "support_valid": support_valid,
            "annotation_available": source_mask is not None,
            "feature_validity": {"image": bool(support_valid and support.any()), "radiomics": None},
            "reference_boundary_truncated": _reference_boundary_truncated(source_mask, geometry, shell_width_mm)
                if is_peri and source_mask is not None else (None if source_mask is None else False),
            "crop_completeness": None,
            "radiomics_config": None,
        }
        record["region_id"] = region_identity(record)
        masks[habitat], records[habitat] = support, record
    overlap = masks["pt_peri"] & masks["ln_peri"]
    physical = _physical_geometry(pt, ln, geometry, midline_lps_x=midline_lps_x)
    physical.update({
        "calculation_version": "voxel_center_v1",
        "reference_region_ids": {name: records[name]["region_id"] for name in RADIOLOGY_HABITATS},
        "reference_geometry": geometry.to_dict(),
    })
    physical["provenance_sha256"] = physical_geometry_checksum(physical)
    manifest = {
        "schema_version": REGION_MANIFEST_VERSION,
        "patient_id": str(patient_id), "scan_id": str(scan_id),
        "reference_geometry": geometry.to_dict(), "regions": records,
        "valid_tissue": {"source": str(valid_tissue_source), "mask_sha256": mask_checksum(body)},
        "shell_overlap": {"rule": "allow_and_measure", "voxel_count": int(overlap.sum()),
                          "physical_volume_mm3": float(overlap.sum()) * geometry.voxel_volume_mm3},
        "physical_geometry": physical,
        "intensity_pipelines": {"image": None, "radiomics": None},
        "scope": "Aggregation support correspondence; learned embeddings may include encoder receptive-field context.",
    }
    validate_region_manifest(manifest, masks=masks, geometry=geometry)
    return masks, manifest


def validate_region_manifest(manifest: Mapping[str, Any], masks=None, geometry=None) -> None:
    """Validate identities and optional loaded support before model/radiomics use."""
    if manifest.get("schema_version") != REGION_MANIFEST_VERSION:
        raise ValueError("Unsupported region manifest schema version.")
    reference = ReferenceGeometry.from_dict(manifest["reference_geometry"])
    records = manifest["regions"]
    if set(records) != set(RADIOLOGY_HABITATS):
        raise ValueError("Region manifest must declare all four anatomical compartments.")
    physical = manifest["physical_geometry"]
    if physical["names"] != list(PHYSICAL_GEOMETRY_FEATURES):
        raise ValueError("Physical geometry feature names/order differ from the shared schema.")
    if len(physical["values"]) != len(PHYSICAL_GEOMETRY_FEATURES) or len(physical["valid"]) != len(PHYSICAL_GEOMETRY_FEATURES):
        raise ValueError("Physical geometry values and validity must have the declared length.")
    if not np.isfinite(physical["values"]).all():
        raise ValueError("Physical geometry must use finite zero-filled values plus explicit validity flags.")
    validity = np.asarray(physical["valid"])
    values = np.asarray(physical["values"], dtype=float)
    if values.shape != (len(PHYSICAL_GEOMETRY_FEATURES),) or validity.shape != values.shape:
        raise ValueError("Physical geometry values and validity must be one-dimensional feature vectors.")
    if not np.isin(validity, [0, 1]).all():
        raise ValueError("Physical geometry validity must contain only boolean or 0/1 flags.")
    if np.any(values[~validity.astype(bool)] != 0.0):
        raise ValueError("Unavailable physical geometry features must be zero filled.")
    if np.any(values < 0.0):
        raise ValueError("Physical volume, distance, spread, flag and count features cannot be negative.")
    if physical.get("calculation_version") != "voxel_center_v1":
        raise ValueError("Physical geometry calculation version is not supported.")
    expected_region_ids = {name: records[name]["region_id"] for name in RADIOLOGY_HABITATS}
    if physical.get("reference_region_ids") != expected_region_ids:
        raise ValueError("Physical geometry source region identities do not match this manifest.")
    assert_same_geometry(reference, physical["reference_geometry"])
    if physical.get("provenance_sha256") != physical_geometry_checksum(physical):
        raise ValueError("Physical geometry provenance checksum mismatch.")
    feature_values = dict(zip(PHYSICAL_GEOMETRY_FEATURES, values))
    feature_validity = dict(zip(PHYSICAL_GEOMETRY_FEATURES, validity.astype(bool)))
    for lesion in ("pt", "ln"):
        record = records[f"{lesion}_intra"]
        feature = f"{lesion}_volume_mm3"
        known = record["status"] in {"present", "absent"}
        if feature_validity[feature] != known:
            raise ValueError(f"{lesion}: physical volume validity disagrees with reference annotation availability.")
        if known and not np.isclose(feature_values[feature], float(record["physical_volume_mm3"]), rtol=1e-8, atol=1e-8):
            raise ValueError(f"{lesion}: physical volume differs from the authoritative reference compartment.")
    if feature_validity["node_count"] or feature_validity["largest_node_volume_mm3"]:
        raise ValueError("Physical geometry v1 does not qualify anatomical node instances.")
    loaded = {}
    for name in RADIOLOGY_HABITATS:
        record = records[name]
        if record["patient_id"] != manifest["patient_id"] or record["scan_id"] != manifest["scan_id"] or record["compartment_id"] != name:
            raise ValueError(f"{name}: patient, scan or anatomical identity mismatch.")
        assert_same_geometry(reference, record["reference_geometry"])
        definition = record["shell_definition"]
        peri = name.endswith("_peri")
        if (
            definition.get("kind") != ("exterior_euclidean" if peri else "annotated_interior")
            or definition.get("width_mm") != (10.0 if peri else 0.0)
            or definition.get("distance_discretization") != "voxel_center"
            or definition.get("exclude_interiors") != (["pt_intra", "ln_intra"] if peri else [])
            or definition.get("shell_overlap_rule") != "allow_and_measure"
        ):
            raise ValueError(f"{name}: unsupported anatomical compartment definition.")
        if record["status"] not in REGION_STATUS_CODES:
            raise ValueError(f"{name}: unknown region state.")
        if record["support_valid"] != (record["status"] in {"present", "absent"}):
            raise ValueError(f"{name}: region state and support validity disagree.")
        if bool(record["presence"]) != (record["status"] == "present"):
            raise ValueError(f"{name}: region state and anatomical presence disagree.")
        if record["region_id"] != region_identity(record):
            raise ValueError(f"{name}: reference region identity/checksum mismatch.")
        expected_volume = float(record["voxel_count"]) * reference.voxel_volume_mm3
        if not np.isclose(float(record["physical_volume_mm3"]), expected_volume, rtol=1e-8, atol=1e-8):
            raise ValueError(f"{name}: physical volume is inconsistent with the reference grid.")
        support_geometry, checksum = reference, record["mask_sha256"]
        if "image_mask_sha256" in record:
            support_geometry = ReferenceGeometry.from_dict(record["image_geometry"])
            checksum = record["image_mask_sha256"]
            expected_id = region_identity(record, mask_sha256=checksum, geometry=support_geometry)
            if record.get("image_region_id") != expected_id:
                raise ValueError(f"{name}: image support identity/checksum mismatch.")
        if geometry is not None:
            assert_same_geometry(support_geometry, geometry)
        if masks is not None:
            support = _binary(masks[name], shape=support_geometry.shape_dhw, name=name)
            if mask_checksum(support) != checksum:
                raise ValueError(f"{name}: loaded support checksum differs from the manifest.")
            if record["status"] != "present" and support.any():
                raise ValueError(f"{name}: absent/unavailable/invalid anatomy contains measured support.")
            if "image_mask_sha256" not in record and int(support.sum()) != record["voxel_count"]:
                raise ValueError(f"{name}: reference voxel count differs from measured support.")
            if record["feature_validity"].get("image") and not support.any():
                raise ValueError(f"{name}: valid image features cannot be assigned to empty support.")
            loaded[name] = support
    if loaded:
        tumor = loaded["pt_intra"] | loaded["ln_intra"]
        if np.any(tumor & (loaded["pt_peri"] | loaded["ln_peri"])):
            raise ValueError("Peritumoral support overlaps annotated tumor interior.")


def load_region_manifest(path: str | Path, *, masks=None, geometry=None) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text())
    validate_region_manifest(manifest, masks=masks, geometry=geometry)
    return manifest


def write_region_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    validate_region_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
