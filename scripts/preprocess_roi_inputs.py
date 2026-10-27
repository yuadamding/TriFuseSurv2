#!/usr/bin/env python3
"""Create fixed-size ROI-crop inputs for TriFuseSurv2 survival training.

Verified mode defines physical 10 mm compartments on the reference acquisition
before cropping and writes their model-grid supports and provenance manifest.
The output metadata rewrites CT/PT/LN paths and adds region_manifest_path.
Legacy mode retains the historical voxel-margin crop without verified regions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PACKAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import SimpleITK as sitk

from trifusesurv2.utils.data import resolve_preprocessed_case_path, validate_normalized_ct


def _parse_margin(values: Sequence[int]) -> tuple[int, int, int]:
    vals = [int(v) for v in values]
    if len(vals) == 1:
        return vals[0], vals[0], vals[0]
    if len(vals) == 3:
        return vals[0], vals[1], vals[2]
    raise ValueError("--margin_voxels expects either one integer or three integers: D H W")


def _safe_id(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("Patient IDs must not be missing.")
    raw = str(value).strip()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    if safe in {"", ".", ".."}:
        raise ValueError(f"Invalid patient ID for an output directory: {value!r}")
    return safe


def _validate_output_ids(values: Sequence[Any]) -> None:
    seen: dict[str, str] = {}
    for value in values:
        safe = _safe_id(value)
        if safe in seen:
            raise ValueError(
                f"Patient IDs {seen[safe]!r} and {str(value)!r} resolve to the same "
                f"output directory {safe!r}; IDs must be unique after path sanitization."
            )
        seen[safe] = str(value)


def _file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _output_fingerprint(path: Path, output_size_dhw: tuple[int, int, int]) -> dict[str, Any]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    size = tuple(reversed(reader.GetSize()))
    if size != output_size_dhw:
        raise ValueError(f"{path}: image size {size} does not match requested {output_size_dhw}")
    return {
        **_file_fingerprint(path),
        "size_dhw": list(size),
        "spacing_xyz": list(reader.GetSpacing()),
        "origin_xyz": list(reader.GetOrigin()),
        "direction": list(reader.GetDirection()),
    }


def _relpath(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return str(path)


def _bbox_from_mask(
    mask: np.ndarray,
    *,
    margin_dhw: tuple[int, int, int],
    output_size_dhw: tuple[int, int, int],
    expand_to_aspect: bool,
    fail_empty_roi: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    shape = np.asarray(mask.shape, dtype=np.int64)
    coords = np.argwhere(mask)
    if coords.size == 0:
        if fail_empty_roi:
            raise ValueError("empty PT/LN ROI mask")
        return np.zeros(3, dtype=np.int64), shape.copy(), "empty_roi_full_volume"

    margin = np.asarray(margin_dhw, dtype=np.int64)
    start = np.maximum(coords.min(axis=0) - margin, 0)
    end = np.minimum(coords.max(axis=0) + 1 + margin, shape)
    if not expand_to_aspect:
        return start.astype(np.int64), end.astype(np.int64), "ok"

    target = np.asarray(output_size_dhw, dtype=np.float64)
    size = np.maximum(end - start, 1).astype(np.float64)
    scale = float(np.max(size / target))
    desired = np.ceil(scale * target).astype(np.int64)
    desired = np.maximum(desired, (end - start).astype(np.int64))
    desired = np.minimum(desired, shape)

    center = (start.astype(np.float64) + end.astype(np.float64)) / 2.0
    start2 = np.floor(center - desired.astype(np.float64) / 2.0).astype(np.int64)
    start2 = np.maximum(start2, 0)
    start2 = np.minimum(start2, shape - desired)
    end2 = start2 + desired
    return start2.astype(np.int64), end2.astype(np.int64), "ok"


def _crop_image(img: sitk.Image, start_dhw: np.ndarray, end_dhw: np.ndarray) -> sitk.Image:
    size_dhw = end_dhw - start_dhw
    index_xyz = [int(start_dhw[2]), int(start_dhw[1]), int(start_dhw[0])]
    size_xyz = [int(size_dhw[2]), int(size_dhw[1]), int(size_dhw[0])]
    return sitk.RegionOfInterest(img, size_xyz, index_xyz)


def _copy_geometry(src: sitk.Image, dst: sitk.Image) -> sitk.Image:
    dst.SetOrigin(src.GetOrigin())
    dst.SetSpacing(src.GetSpacing())
    dst.SetDirection(src.GetDirection())
    return dst


def _resample_to_size(img: sitk.Image, output_size_dhw: tuple[int, int, int], *, is_mask: bool) -> sitk.Image:
    output_size_xyz = [int(output_size_dhw[2]), int(output_size_dhw[1]), int(output_size_dhw[0])]
    input_size = np.asarray(img.GetSize(), dtype=np.float64)
    output_size = np.asarray(output_size_xyz, dtype=np.float64)
    input_spacing = np.asarray(img.GetSpacing(), dtype=np.float64)
    output_spacing = tuple((input_spacing * input_size / output_size).tolist())

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(output_size_xyz)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    if not is_mask:
        resampler.SetOutputPixelType(sitk.sitkFloat32)
    out = resampler.Execute(img)

    if not is_mask:
        return sitk.Cast(out, sitk.sitkFloat32)

    arr = (sitk.GetArrayFromImage(out) > 0.5).astype(np.uint8)
    return _copy_geometry(out, sitk.GetImageFromArray(arr))


def _resolve_required_path(row: Mapping[str, Any], col: str, *, meta_dir: Path, id_col: str) -> Path:
    pid = str(row.get(id_col, ""))
    raw = str(row.get(col, "")).strip()
    resolved = Path(resolve_preprocessed_case_path(raw, data_root=str(meta_dir), patient_id=pid))
    if not resolved.is_file():
        raise FileNotFoundError(f"{col} not found for patient_id={pid}: {resolved} (raw={raw})")
    return resolved


def _process_one(job: Mapping[str, Any]) -> dict[str, Any]:
    if job.get("region_mode", "legacy") == "verified":
        return _process_verified(job)
    row = job["row"]
    index = int(job["index"])
    id_col = str(job["id_col"])
    pid = str(row[id_col])
    pid_safe = _safe_id(pid)
    meta_dir = Path(job["meta_dir"])
    out_root = Path(job["out_root"])
    out_csv_dir = Path(job["out_csv_dir"])
    output_size_dhw = tuple(int(x) for x in job["output_size_dhw"])
    margin_dhw = tuple(int(x) for x in job["margin_dhw"])
    overwrite = bool(job["overwrite"])
    expand_to_aspect = bool(job["expand_to_aspect"])
    fail_empty_roi = bool(job["fail_empty_roi"])

    ct_path = _resolve_required_path(row, str(job["ct_col"]), meta_dir=meta_dir, id_col=id_col)
    pt_path = _resolve_required_path(row, str(job["mask_pt_col"]), meta_dir=meta_dir, id_col=id_col)
    ln_path = _resolve_required_path(row, str(job["mask_ln_col"]), meta_dir=meta_dir, id_col=id_col)

    case_dir = out_root / pid_safe
    if case_dir.resolve().parent != out_root.resolve():
        raise ValueError(f"Patient output directory resolves outside out_root: {case_dir}")
    ct_out = case_dir / "ct_roi.nii.gz"
    pt_out = case_dir / "mask_primary_roi.nii.gz"
    ln_out = case_dir / "mask_nodal_roi.nii.gz"
    provenance_path = case_dir / "roi_crop_provenance.json"
    outputs = (ct_out, pt_out, ln_out)
    inputs = {
        "schema_version": 1,
        "patient_id": pid,
        "output_size_dhw": list(output_size_dhw),
        "margin_dhw": list(margin_dhw),
        "expand_to_aspect": expand_to_aspect,
        "fail_empty_roi": fail_empty_roi,
        "sources": {
            name: {"path": str(path.resolve()), **_file_fingerprint(path)}
            for name, path in (("ct", ct_path), ("pt", pt_path), ("ln", ln_path))
        },
    }

    ct_img = sitk.ReadImage(str(ct_path))
    pt_img = sitk.ReadImage(str(pt_path))
    ln_img = sitk.ReadImage(str(ln_path))
    validate_normalized_ct(sitk.GetArrayViewFromImage(ct_img), source=str(ct_path))
    if ct_img.GetSize() != pt_img.GetSize() or ct_img.GetSize() != ln_img.GetSize():
        raise ValueError(
            f"patient_id={pid} CT/PT/LN image sizes differ: "
            f"ct={ct_img.GetSize()} pt={pt_img.GetSize()} ln={ln_img.GetSize()}"
        )

    pt = sitk.GetArrayFromImage(pt_img) > 0.5
    ln = sitk.GetArrayFromImage(ln_img) > 0.5
    union = np.logical_or(pt, ln)
    start, end, status = _bbox_from_mask(
        union,
        margin_dhw=margin_dhw,
        output_size_dhw=output_size_dhw,
        expand_to_aspect=expand_to_aspect,
        fail_empty_roi=fail_empty_roi,
    )

    inputs["bbox_zyx"] = [[int(x) for x in start], [int(x) for x in end]]
    existing = any(path.exists() for path in (*outputs, provenance_path))
    if existing and not overwrite:
        try:
            provenance = json.loads(provenance_path.read_text())
            if provenance.get("inputs") != inputs:
                raise ValueError("source images or crop settings differ from the recorded inputs")
            actual_outputs = {path.name: _output_fingerprint(path, output_size_dhw) for path in outputs}
            if provenance.get("outputs") != actual_outputs:
                raise ValueError("output contents differ from the recorded hashes or geometry")
        except (OSError, RuntimeError, ValueError, AttributeError) as exc:
            raise RuntimeError(
                f"Cannot reuse ROI crops for patient_id={pid}: {exc}. "
                "Use a new --out_root or pass --overwrite to explicitly regenerate this output."
            ) from exc
    else:
        case_dir.mkdir(parents=True, exist_ok=True)
        # Prepare the complete case before replacing its outputs; publish provenance last.
        with TemporaryDirectory(prefix=".roi-crop-", dir=case_dir) as temporary:
            stage = Path(temporary)
            for img, output, is_mask in ((ct_img, ct_out, False), (pt_img, pt_out, True), (ln_img, ln_out, True)):
                cropped = _crop_image(img, start, end)
                sitk.WriteImage(_resample_to_size(cropped, output_size_dhw, is_mask=is_mask), str(stage / output.name))
            provenance = {
                "inputs": inputs,
                "outputs": {path.name: _output_fingerprint(stage / path.name, output_size_dhw) for path in outputs},
            }
            staged_provenance = stage / provenance_path.name
            staged_provenance.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
            for path in outputs:
                os.replace(stage / path.name, path)
            os.replace(staged_provenance, provenance_path)

    return {
        "index": index,
        "pid": pid,
        "ct_out_path": _relpath(ct_out, out_csv_dir),
        "mask_primary_out_path": _relpath(pt_out, out_csv_dir),
        "mask_nodal_out_path": _relpath(ln_out, out_csv_dir),
        "roi_crop_status": status,
        "roi_crop_bbox_zyx": json.dumps([[int(x) for x in start], [int(x) for x in end]]),
        "roi_crop_margin_voxels": json.dumps([int(x) for x in margin_dhw]),
        "roi_crop_output_size_dhw": json.dumps([int(x) for x in output_size_dhw]),
        "roi_crop_source_ct": str(ct_path),
    }


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None or pd.isna(value) or not str(value).strip():
        raise ValueError(f"Verified preprocessing requires metadata field {key}.")
    return str(value).strip()


def _annotation_available(row: Mapping[str, Any], lesion: str) -> bool:
    value = row.get(f"{lesion}_annotation_available", True)
    if pd.isna(value):
        raise ValueError(f"{lesion}_annotation_available must be explicitly true or false.")
    if str(value).strip().lower() in {"0", "false", "no"}:
        return False
    if str(value).strip().lower() in {"1", "true", "yes"}:
        return True
    raise ValueError(f"Invalid {lesion}_annotation_available value: {value!r}")


def _array_image(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    return _copy_geometry(reference, sitk.GetImageFromArray(array.astype(np.uint8)))


def _verified_binary_mask(image: sitk.Image, *, source: Path) -> np.ndarray:
    array = sitk.GetArrayFromImage(image)
    if array.ndim != 3 or not np.isfinite(array).all() or not np.isin(array, (0, 1)).all():
        raise ValueError(f"Verified mask source must contain finite binary 0/1 values on a 3D grid: {source}")
    return array.astype(bool)


def _process_verified(job: Mapping[str, Any]) -> dict[str, Any]:
    from trifusesurv2.preprocessing.compartments import (
        ReferenceGeometry, assert_same_geometry, build_compartments,
        mask_checksum, region_identity, validate_region_manifest,
    )
    from trifusesurv2.preprocessing.region_io import validate_deployment, load_region_manifest
    row = job["row"]
    pid = _required_text(row, str(job["id_col"]))
    scan_id = _required_text(row, str(job.get("scan_id_col", "scan_id")))
    contour_source = _required_text(row, "contour_source")
    contour_version = _required_text(row, "contour_version")
    intensity = _required_text(row, "ct_intensity_transform")
    meta_dir, out_root = Path(job["meta_dir"]), Path(job["out_root"])
    out_csv_dir = Path(job["out_csv_dir"])
    shape = tuple(map(int, job["output_size_dhw"]))
    ct_path = _resolve_required_path(row, str(job["ct_col"]), meta_dir=meta_dir, id_col=str(job["id_col"]))
    ct = sitk.ReadImage(str(ct_path))
    geometry = ReferenceGeometry.from_image(ct)
    validate_normalized_ct(sitk.GetArrayViewFromImage(ct), source=str(ct_path))
    sources = {"ct": {"path": str(ct_path.resolve()), **_file_fingerprint(ct_path)}}
    contours = {}
    for lesion, col in (("pt", job["mask_pt_col"]), ("ln", job["mask_ln_col"])):
        if not _annotation_available(row, lesion):
            contours[lesion] = None
            sources[lesion] = None
            continue
        path = _resolve_required_path(row, str(col), meta_dir=meta_dir, id_col=str(job["id_col"]))
        image = sitk.ReadImage(str(path))
        assert_same_geometry(geometry, image)
        contours[lesion] = _verified_binary_mask(image, source=path)
        sources[lesion] = {"path": str(path.resolve()), **_file_fingerprint(path)}
    tissue_policy = str(job.get("valid_tissue_policy", "mask"))
    if tissue_policy not in {"mask", "full_fov"}:
        raise ValueError(f"Unknown valid tissue policy: {tissue_policy}")
    if tissue_policy == "full_fov":
        tissue = np.ones(geometry.shape_dhw, dtype=bool)
        tissue_source = "explicit_full_field_of_view; includes all imaged voxels without tissue classification"
    else:
        path = _resolve_required_path(row, str(job.get("valid_tissue_col", "valid_tissue_out_path")), meta_dir=meta_dir, id_col=str(job["id_col"]))
        image = sitk.ReadImage(str(path))
        assert_same_geometry(geometry, image)
        tissue = _verified_binary_mask(image, source=path)
        tissue_source = _required_text(row, "valid_tissue_definition")
        sources["valid_tissue"] = {"path": str(path.resolve()), **_file_fingerprint(path)}
    midline = row.get("midline_lps_x_mm")
    midline = None if midline is None or pd.isna(midline) else float(midline)
    masks, manifest = build_compartments(
        contours["pt"], contours["ln"], tissue, geometry,
        patient_id=pid, scan_id=scan_id, contour_source=contour_source,
        contour_version=contour_version, valid_tissue_source=tissue_source, midline_lps_x=midline,
    )
    segmentation = row.get("segmentation_provenance", "")
    if isinstance(segmentation, str) and segmentation.strip():
        manifest["segmentation_provenance"] = json.loads(segmentation)
    validate_deployment(manifest, str(job.get("deployment_mode", "expert")))
    margin_mm = float(job.get("margin_mm", 10.0))
    if not np.isfinite(margin_mm) or margin_mm < 10.0:
        raise ValueError("Verified --margin_mm must be finite and at least the fixed 10 mm shell width.")
    margin = tuple(np.ceil(margin_mm / np.asarray(geometry.spacing_dhw)).astype(int))
    union = np.zeros(geometry.shape_dhw, dtype=bool)
    for lesion in contours.values():
        if lesion is not None:
            union |= lesion
    start, end, status = _bbox_from_mask(
        union, margin_dhw=margin, output_size_dhw=shape,
        expand_to_aspect=bool(job["expand_to_aspect"]), fail_empty_roi=bool(job["fail_empty_roi"]),
    )
    # The reference shells are defined before cropping. Ensure all available
    # declared tissue survives; FOV-boundary truncation remains separately recorded.
    crop_slices = tuple(slice(int(a), int(b)) for a, b in zip(start, end))
    for name, support in masks.items():
        if int(support[crop_slices].sum()) != int(support.sum()):
            raise ValueError(f"{pid}/{name}: physical crop would truncate its authoritative support.")
    ct_crop = _resample_to_size(_crop_image(ct, start, end), shape, is_mask=False)
    image_geometry = ReferenceGeometry.from_image(ct_crop)
    radiomics_ct = ct
    radiomics_transform = intensity
    raw_col = str(job.get("radiomics_ct_col", "radiomics_ct_path"))
    raw_value = row.get(raw_col)
    if raw_value is not None and not pd.isna(raw_value) and str(raw_value).strip():
        raw_scan = _required_text(row, "radiomics_scan_id")
        if raw_scan != scan_id:
            raise ValueError("Radiomics and normalized CT must declare the same acquisition scan_id.")
        path = _resolve_required_path(row, raw_col, meta_dir=meta_dir, id_col=str(job["id_col"]))
        radiomics_ct = sitk.ReadImage(str(path))
        assert_same_geometry(geometry, radiomics_ct)
        if not np.isfinite(sitk.GetArrayViewFromImage(radiomics_ct)).all():
            raise ValueError("Radiomics CT must contain finite intensities.")
        radiomics_transform = _required_text(row, "radiomics_intensity_transform")
        sources["radiomics_ct"] = {"path": str(path.resolve()), **_file_fingerprint(path)}
    inputs = {
        "schema_version": 2, "patient_id": pid, "scan_id": scan_id,
        "sources": sources, "margin_mm": margin_mm, "output_size_dhw": list(shape),
        "bbox_zyx": [start.tolist(), end.tolist()], "contour_source": contour_source,
        "contour_version": contour_version, "valid_tissue_policy": tissue_policy,
        "valid_tissue_definition": tissue_source, "ct_intensity_transform": intensity,
        "radiomics_intensity_transform": radiomics_transform,
        "segmentation_provenance": manifest.get("segmentation_provenance"),
        "midline_lps_x_mm": midline,
    }
    case_dir = out_root / _safe_id(pid)
    if case_dir.resolve().parent != out_root.resolve():
        raise ValueError("Patient output directory must remain inside out_root.")
    manifest_path = case_dir / "region_manifest.json"
    provenance_path = case_dir / "roi_crop_provenance.json"
    ct_out, pt_out, ln_out = (case_dir / n for n in ("ct_roi.nii.gz", "mask_primary_roi.nii.gz", "mask_nodal_roi.nii.gz"))
    expected_names = ["ct_roi.nii.gz", "ct_radiomics.nii.gz", "mask_primary_roi.nii.gz", "mask_nodal_roi.nii.gz"]
    expected_names += [f"region_{name}.nii.gz" for name in masks]
    expected_names += ["region_manifest.json"]
    existing = any((case_dir / name).exists() for name in expected_names) or provenance_path.exists()
    if existing and not bool(job["overwrite"]):
        try:
            receipt = json.loads(provenance_path.read_text())
            if receipt.get("inputs") != inputs:
                raise ValueError("source/configuration differs")
            actual = {name: _file_fingerprint(case_dir / name) for name in expected_names}
            if receipt.get("outputs") != actual:
                raise ValueError("output files differ from recorded fingerprints")
            persisted = load_region_manifest(manifest_path, pid, deployment_mode=str(job.get("deployment_mode", "expert")))
            if any(persisted["regions"][name]["region_id"] != manifest["regions"][name]["region_id"] for name in masks):
                raise ValueError("reference compartment identities differ from the current definition")
            if persisted["physical_geometry"] != manifest["physical_geometry"]:
                raise ValueError("physical geometry differs from the current definition")
        except (OSError, ValueError, AttributeError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Cannot reuse verified regions for {pid}: {exc}. Use a new --out_root or --overwrite.") from exc
    else:
        case_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".regions-", dir=case_dir) as temporary:
            stage = Path(temporary)
            sitk.WriteImage(ct_crop, str(stage / ct_out.name))
            raw_crop = _resample_to_size(_crop_image(radiomics_ct, start, end), shape, is_mask=False)
            sitk.WriteImage(raw_crop, str(stage / "ct_radiomics.nii.gz"))
            for name, support in masks.items():
                image = _resample_to_size(_crop_image(_array_image(support, ct), start, end), shape, is_mask=True)
                arr = sitk.GetArrayFromImage(image) > 0.5
                path = stage / f"region_{name}.nii.gz"
                sitk.WriteImage(image, str(path))
                if name in {"pt_intra", "ln_intra"}:
                    sitk.WriteImage(image, str(stage / (pt_out.name if name == "pt_intra" else ln_out.name)))
                record = manifest["regions"][name]
                image_hash = mask_checksum(arr)
                record.update(image_mask_path=path.name, image_mask_sha256=image_hash,
                              image_geometry=image_geometry.to_dict(),
                              image_region_id=region_identity(record, mask_sha256=image_hash, geometry=image_geometry))
                reference_count = int(support.sum())
                image_volume = int(arr.sum()) * image_geometry.voxel_volume_mm3
                record["feature_validity"]["image"] = bool(record["support_valid"] and arr.any())
                record["crop_completeness"] = {
                    "fraction": 1.0, "retained_reference_voxels": reference_count,
                    "reference_voxels": reference_count,
                    "reference_volume_mm3": record["physical_volume_mm3"],
                    "retained_reference_volume_mm3": reference_count * geometry.voxel_volume_mm3,
                    "image_volume_mm3": image_volume,
                    "image_to_reference_volume_ratio": image_volume / record["physical_volume_mm3"] if reference_count else None,
                    "reference_boundary_truncated": record["reference_boundary_truncated"],
                    "resampling": "nearest-neighbor support representation; volume is measured, not assumed conserved",
                }
            for key, name, transform in (("image", ct_out.name, intensity), ("radiomics_image", "ct_radiomics.nii.gz", radiomics_transform)):
                manifest[key] = {"path": name, "sha256": _file_fingerprint(stage / name)["sha256"],
                                 "geometry": image_geometry.to_dict(), "intensity_transform": transform,
                                 "scan_id": scan_id, "spatial_transform": "physical ROI crop followed by linear intensity resampling"}
            manifest["intensity_pipelines"] = {"image": intensity, "radiomics": radiomics_transform}
            manifest["source_files"] = sources
            manifest["crop"] = {"margin_mm": margin_mm, "bbox_zyx": [start.tolist(), end.tolist()], "retains_all_available_reference_support": True}
            validate_region_manifest(manifest)
            (stage / manifest_path.name).write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
            receipt = {"inputs": inputs, "outputs": {name: _file_fingerprint(stage / name) for name in expected_names}}
            (stage / provenance_path.name).write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
            for name in expected_names:
                os.replace(stage / name, case_dir / name)
            os.replace(stage / provenance_path.name, provenance_path)
    return {
        "index": int(job["index"]), "pid": pid,
        "ct_out_path": _relpath(ct_out, out_csv_dir), "mask_primary_out_path": _relpath(pt_out, out_csv_dir),
        "mask_nodal_out_path": _relpath(ln_out, out_csv_dir), "region_manifest_path": _relpath(manifest_path, out_csv_dir),
        "roi_crop_status": status, "roi_crop_bbox_zyx": json.dumps([start.tolist(), end.tolist()]),
        "roi_crop_margin_voxels": json.dumps(list(map(int, margin))), "roi_crop_margin_mm": margin_mm,
        "roi_crop_output_size_dhw": json.dumps(list(shape)), "roi_crop_source_ct": str(ct_path),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta_csv", default="OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv")
    p.add_argument("--out_root", default="OPSCC_preprocessed_roi_96x128x128")
    p.add_argument("--out_csv", default="")
    p.add_argument("--id_col", default="patient_id")
    p.add_argument("--ct_col", default="ct_out_path")
    p.add_argument("--mask_pt_col", default="mask_primary_out_path")
    p.add_argument("--mask_ln_col", default="mask_nodal_out_path")
    p.add_argument("--output_size", type=int, nargs=3, default=[96, 128, 128], metavar=("D", "H", "W"))
    p.add_argument(
        "--margin_voxels",
        type=int,
        nargs="+",
        default=[24],
        help="Legacy mode only: one isotropic margin or three margins in D H W voxel order.",
    )
    p.add_argument("--region_mode", choices=("verified", "legacy"), default="verified")
    p.add_argument("--margin_mm", type=float, default=10.0, help="Verified physical crop margin; at least the fixed 10 mm shell.")
    p.add_argument("--scan_id_col", default="scan_id")
    p.add_argument("--valid_tissue_col", default="valid_tissue_out_path")
    p.add_argument("--valid_tissue_policy", choices=("mask", "full_fov"), default="mask")
    p.add_argument("--radiomics_ct_col", default="radiomics_ct_path")
    p.add_argument("--deployment_mode", choices=("expert", "automated", "contour_variation"), default="expert")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail_empty_roi", action="store_true")
    p.add_argument(
        "--no_expand_to_aspect",
        action="store_true",
        help="Do not expand the ROI bounding box toward the output-size aspect ratio before resampling.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    meta_csv = Path(args.meta_csv).resolve()
    out_root = Path(args.out_root).resolve()
    out_csv = Path(args.out_csv).resolve() if str(args.out_csv).strip() else out_root / "cohort_preprocessed_stage2_roi.csv"

    output_size_dhw = tuple(int(x) for x in args.output_size)
    margin_dhw = _parse_margin(args.margin_voxels)
    if any(x <= 0 for x in output_size_dhw):
        raise ValueError(f"--output_size must be positive, got {output_size_dhw}")
    if any(x < 0 for x in margin_dhw):
        raise ValueError(f"--margin_voxels must be nonnegative, got {margin_dhw}")

    df = pd.read_csv(meta_csv, dtype={args.id_col: str})
    if args.id_col not in df.columns:
        raise ValueError(f"Missing patient identifier column: {args.id_col}")
    for col in (
        args.ct_col,
        args.mask_pt_col,
        args.mask_ln_col,
        "roi_crop_status",
        "roi_crop_bbox_zyx",
        "roi_crop_margin_voxels",
        "roi_crop_output_size_dhw",
        "roi_crop_source_ct",
    ):
        if col not in df.columns:
            df[col] = ""

    runnable_indices: list[int] = []
    if "status" in df.columns:
        status_ok = df["status"].astype(str).str.lower().eq("ok")
        skipped = int((~status_ok).sum())
        if skipped:
            df.loc[~status_ok, "roi_crop_status"] = "skipped_non_ok_status"
        runnable_indices = [int(i) for i in df.index[status_ok]]
    else:
        runnable_indices = [int(i) for i in df.index]

    _validate_output_ids(df.loc[runnable_indices, args.id_col].tolist())
    df[args.id_col] = df[args.id_col].astype(str)
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    jobs = [
        {
            "index": int(i),
            "region_mode": args.region_mode, "margin_mm": args.margin_mm,
            "scan_id_col": args.scan_id_col, "valid_tissue_col": args.valid_tissue_col,
            "valid_tissue_policy": args.valid_tissue_policy, "radiomics_ct_col": args.radiomics_ct_col,
            "deployment_mode": args.deployment_mode,
            "row": df.loc[i].to_dict(),
            "id_col": args.id_col,
            "ct_col": args.ct_col,
            "mask_pt_col": args.mask_pt_col,
            "mask_ln_col": args.mask_ln_col,
            "meta_dir": str(meta_csv.parent),
            "out_root": str(out_root),
            "out_csv_dir": str(out_csv.parent),
            "output_size_dhw": output_size_dhw,
            "margin_dhw": margin_dhw,
            "overwrite": bool(args.overwrite),
            "expand_to_aspect": not bool(args.no_expand_to_aspect),
            "fail_empty_roi": bool(args.fail_empty_roi),
        }
        for i in runnable_indices
    ]

    print(
        f"[roi-preprocess] meta_csv={meta_csv} rows={len(df)} runnable={len(jobs)} "
        f"out_root={out_root} output_size_dhw={output_size_dhw} margin_dhw={margin_dhw}"
    )

    results: list[dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for n, job in enumerate(jobs, start=1):
            result = _process_one(job)
            results.append(result)
            if n == 1 or n == len(jobs) or n % 25 == 0:
                print(f"[roi-preprocess] processed {n}/{len(jobs)} patient_id={result['pid']}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_one, job) for job in jobs]
            for n, fut in enumerate(as_completed(futures), start=1):
                result = fut.result()
                results.append(result)
                if n == 1 or n == len(jobs) or n % 25 == 0:
                    print(f"[roi-preprocess] processed {n}/{len(jobs)} patient_id={result['pid']}")

    for result in results:
        idx = int(result["index"])
        df.at[idx, args.ct_col] = result["ct_out_path"]
        df.at[idx, args.mask_pt_col] = result["mask_primary_out_path"]
        df.at[idx, args.mask_ln_col] = result["mask_nodal_out_path"]
        if "region_manifest_path" in result:
            df.at[idx, "region_manifest_path"] = result["region_manifest_path"]
            df.at[idx, "roi_crop_margin_mm"] = result["roi_crop_margin_mm"]
        for key in (
            "roi_crop_status",
            "roi_crop_bbox_zyx",
            "roi_crop_margin_voxels",
            "roi_crop_output_size_dhw",
            "roi_crop_source_ct",
        ):
            df.at[idx, key] = result[key]

    df.to_csv(out_csv, index=False)
    print(f"[roi-preprocess] wrote {out_csv}")
    print(f"[roi-preprocess] train with: META_CSV={_relpath(out_csv, Path.cwd())} IMG_SIZE=\"{' '.join(map(str, output_size_dhw))}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
