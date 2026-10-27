#!/usr/bin/env python3
"""Audit declared physical sampling across manifests without opening image files.

This is a metadata diagnostic, not mask-byte verification, feature extraction,
sampling-policy qualification, or an outcome analysis. No grid is selected.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trifusesurv2.preprocessing.compartments import ReferenceGeometry, assert_same_geometry, validate_region_manifest
from trifusesurv2.schema import RADIOLOGY_HABITATS

SCOPE = "manifest metadata only; image/mask bytes and extracted features were not inspected"
NUMERIC_METRICS = tuple(f"reference_spacing_{axis}_mm" for axis in "xyz") + tuple(
    f"model_spacing_{axis}_mm" for axis in "xyz"
) + tuple(f"model_extent_{axis}_mm" for axis in "xyz") + (
    "model_voxel_volume_mm3", "model_spacing_anisotropy",
) + tuple(f"{region}_volume_ratio" for region in RADIOLOGY_HABITATS)


def number(value, name, *, optional=False, minimum=0.0):
    if optional and value in (None, ""):
        return None
    result = float(value)
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite" + (f" and >= {minimum}" if minimum is not None else ""))
    return result


def inspect_manifest(path, *, patient_id=None, scan_id=None, attributes=None):
    """Return one patient row and four region rows, using JSON declarations only."""
    path = Path(path).resolve(strict=True)
    payload = path.read_bytes()
    manifest = json.loads(payload)
    validate_region_manifest(manifest)
    if patient_id is not None and patient_id != str(manifest["patient_id"]):
        raise ValueError("Metadata patient identifier differs from manifest patient_id.")
    if scan_id is not None and scan_id != str(manifest["scan_id"]):
        raise ValueError("Metadata acquisition identifier differs from manifest scan_id.")
    reference = ReferenceGeometry.from_dict(manifest["reference_geometry"])
    model = ReferenceGeometry.from_dict(manifest["image"]["geometry"])
    if str(manifest["image"]["scan_id"]) != str(manifest["scan_id"]):
        raise ValueError("Model image and reference acquisition identities differ.")
    patient = {
        "patient_id": str(manifest["patient_id"]), "scan_id": str(manifest["scan_id"]),
        "manifest_path": str(path), "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "validation_scope": SCOPE, **(attributes or {}),
        "model_shape_dhw": "x".join(map(str, model.shape_dhw)),
        "model_voxel_volume_mm3": model.voxel_volume_mm3,
        "model_spacing_anisotropy": max(model.spacing_xyz) / min(model.spacing_xyz),
    }
    for index, axis in enumerate("xyz"):
        patient[f"reference_spacing_{axis}_mm"] = reference.spacing_xyz[index]
        patient[f"model_spacing_{axis}_mm"] = model.spacing_xyz[index]
        patient[f"model_extent_{axis}_mm"] = model.spacing_xyz[index] * model.shape_dhw[2 - index]
    regions = []
    for habitat in RADIOLOGY_HABITATS:
        record = manifest["regions"][habitat]
        assert_same_geometry(model, record["image_geometry"])
        crop = record.get("crop_completeness") or {}
        reference_volume = number(record["physical_volume_mm3"], "reference volume")
        model_volume = number(crop.get("image_volume_mm3"), "model volume")
        fraction = number(crop.get("fraction"), "retained reference fraction", optional=True)
        if fraction is not None and fraction > 1 + 1e-9:
            raise ValueError(f"{habitat}: retained fraction exceeds one.")
        ratio = model_volume / reference_volume if reference_volume > 0 else None
        declared_ratio = number(crop.get("image_to_reference_volume_ratio"), "declared volume ratio", optional=True)
        if (ratio is None) != (declared_ratio is None) or (ratio is not None and not np.isclose(ratio, declared_ratio)):
            raise ValueError(f"{habitat}: declared model/reference volume ratio is inconsistent.")
        if record["status"] != "present" and model_volume > 0:
            raise ValueError(f"{habitat}: absent/unavailable anatomy has declared model support.")
        if bool(record["feature_validity"]["image"]) != bool(record["support_valid"] and model_volume > 0):
            raise ValueError(f"{habitat}: image feature validity contradicts declared model support.")
        disappeared = record["status"] == "present" and reference_volume > 0 and model_volume == 0
        region = {
            "patient_id": patient["patient_id"], "scan_id": patient["scan_id"], "region": habitat,
            "reference_region_id": record["region_id"], "model_region_id": record["image_region_id"],
            "reference_status": record["status"], "support_valid": record["support_valid"],
            "image_features_valid": record["feature_validity"]["image"],
            "reference_volume_mm3": reference_volume, "declared_model_volume_mm3": model_volume,
            "model_reference_volume_ratio": ratio, "declared_support_disappeared": disappeared,
            "retained_reference_fraction": fraction,
            "crop_truncated": None if fraction is None else fraction < 1 - 1e-9,
            "shell_source_fov_truncated": record["reference_boundary_truncated"] if habitat.endswith("_peri") else None,
        }
        regions.append(region)
        patient[f"{habitat}_volume_ratio"] = ratio
        patient[f"{habitat}_reference_volume_mm3"] = reference_volume
    patient["declared_disappearance_count"] = sum(row["declared_support_disappeared"] for row in regions)
    patient["crop_truncated_region_count"] = sum(row["crop_truncated"] is True for row in regions)
    patient["crop_unknown_region_count"] = sum(row["crop_truncated"] is None for row in regions)
    patient["shell_source_fov_flag_count"] = sum(row["shell_source_fov_truncated"] is True for row in regions)
    return patient, regions


def distribution(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(values):
        return {"n": 0}
    quantiles = np.quantile(values, [0, .05, .25, .5, .75, .95, 1])
    return {"n": len(values), **dict(zip(("minimum", "p05", "p25", "median", "p75", "p95", "maximum"), quantiles.tolist()))}


def aggregate(patients, regions):
    return {
        "patient_scan_count": len(patients),
        "metrics": {name: distribution(row.get(name) for row in patients) for name in NUMERIC_METRICS},
        "regions": {
            habitat: {
                "reference_states": dict(Counter(row["reference_status"] for row in regions if row["region"] == habitat)),
                "reference_present_count": sum(row["reference_status"] == "present" for row in regions if row["region"] == habitat),
                "declared_disappearance_count": sum(row["declared_support_disappeared"] for row in regions if row["region"] == habitat),
                "crop_truncated_count": sum(row["crop_truncated"] is True for row in regions if row["region"] == habitat),
                "crop_unknown_count": sum(row["crop_truncated"] is None for row in regions if row["region"] == habitat),
                "shell_source_fov_flag_count": sum(row["shell_source_fov_truncated"] is True for row in regions if row["region"] == habitat),
                "shell_source_fov_unknown_count": sum(row["shell_source_fov_truncated"] is None for row in regions if row["region"] == habitat) if habitat.endswith("_peri") else 0,
            } for habitat in RADIOLOGY_HABITATS
        },
    }


def summarize(patients, regions, *, categorical_fields=(), numeric_burden=False):
    summary = {"scope": SCOPE, "overall": aggregate(patients, regions), "stratified": {}}
    for field in categorical_fields:
        groups = []
        for value in sorted({row.get(field, "") for row in patients}):
            subset = [row for row in patients if row.get(field, "") == value]
            identities = {(row["patient_id"], row["scan_id"]) for row in subset}
            subset_regions = [row for row in regions if (row["patient_id"], row["scan_id"]) in identities]
            groups.append({"value": value or None, **aggregate(subset, subset_regions)})
        summary["stratified"][field] = groups
    if numeric_burden:
        associations = {}
        for name in NUMERIC_METRICS:
            pairs = [(row.get("nodal_burden_numeric"), row.get(name)) for row in patients]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            result = {"n": len(pairs), "spearman_r": None}
            if len(pairs) >= 3:
                x, y = np.asarray(pairs).T
                if np.ptp(x) > 0 and np.ptp(y) > 0:
                    result["spearman_r"] = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
            associations[name] = result
        summary["nodal_burden_descriptive_associations"] = associations
        summary["nodal_burden_numeric_missing_count"] = sum(row.get("nodal_burden_numeric") is None for row in patients)
    summary["interpretation"] = (
        "Descriptive summaries only; no outcomes, significance tests, causal claims, chosen spacing, or pass thresholds. "
        "Reference crop retention does not restore shell coverage outside the source FOV. "
        "Nodal burden is caller-declared metadata, not a connected-component node count. "
        "Representative image review and qualified feature extraction remain required."
    )
    return summary


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--metadata-csv", type=Path)
    source.add_argument("--manifest", type=Path, nargs="+")
    parser.add_argument("--data-root", type=Path, help="Relative manifest path base; defaults to metadata CSV directory.")
    parser.add_argument("--id_col", "--patient-id-col", default="Patient_ID")
    parser.add_argument("--scan-id-col", default="scan_id")
    parser.add_argument("--manifest-col", default="region_manifest_path")
    parser.add_argument("--institution-col")
    parser.add_argument("--protocol-col")
    parser.add_argument("--nodal-burden-col")
    parser.add_argument("--nodal-burden-kind", choices=("numeric", "categorical"), default="numeric")
    parser.add_argument("--out-dir", required=True, type=Path, help="New audit report directory.")
    args = parser.parse_args(argv)
    entries, categorical = [], []
    if args.metadata_csv:
        from trifusesurv2.preprocessing.identifiers import resolve_identifier_column
        with args.metadata_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            id_col = resolve_identifier_column(fields, args.id_col)
            manifest_col = resolve_identifier_column(fields, args.manifest_col)
            scan_col = resolve_identifier_column(fields, args.scan_id_col, required=False)
            extra_columns = {}
            for destination, requested in (("institution", args.institution_col), ("acquisition_protocol", args.protocol_col), ("nodal_burden", args.nodal_burden_col)):
                if requested:
                    extra_columns[destination] = resolve_identifier_column(fields, requested)
                    if destination != "nodal_burden" or args.nodal_burden_kind == "categorical":
                        categorical.append(destination)
            for row in reader:
                if not row[id_col].strip() or not row[manifest_col].strip():
                    raise ValueError("Every metadata row must declare patient and manifest identifiers.")
                attributes = {key: row[column] for key, column in extra_columns.items()}
                if args.nodal_burden_col and args.nodal_burden_kind == "numeric":
                    try:
                        attributes["nodal_burden_numeric"] = number(attributes["nodal_burden"], "nodal burden", minimum=None)
                    except (TypeError, ValueError):
                        attributes["nodal_burden_numeric"] = None
                path = Path(row[manifest_col])
                if not path.is_absolute():
                    path = (args.data_root or args.metadata_csv.resolve().parent) / path
                entries.append((path, row[id_col], row[scan_col] if scan_col else None, attributes))
    else:
        entries = [(path, None, None, {}) for path in args.manifest]
    patients, regions, errors, seen = [], [], [], set()
    for path, pid, scan, attributes in entries:
        try:
            patient, patient_regions = inspect_manifest(path, patient_id=pid, scan_id=scan, attributes=attributes)
            identity = (patient["patient_id"], patient["scan_id"])
            if identity in seen:
                raise ValueError("Duplicate patient/scan row would bias audit summaries.")
            seen.add(identity)
            patients.append(patient)
            regions.extend(patient_regions)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append({"patient_id": pid, "scan_id": scan, "manifest_path": str(path), "error": str(exc)})
    summary = summarize(patients, regions, categorical_fields=categorical,
                        numeric_burden=bool(args.nodal_burden_col and args.nodal_burden_kind == "numeric"))
    summary.update(requested_patient_scans=len(entries), failed_patient_scans=len(errors),
                   metadata_fields={"patient": args.id_col, "scan": args.scan_id_col, "institution": args.institution_col,
                                    "protocol": args.protocol_col, "nodal_burden": args.nodal_burden_col})
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.out_dir / "patients.csv", patients)
    write_csv(args.out_dir / "regions.csv", regions)
    write_csv(args.out_dir / "errors.csv", errors)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Manifest-only sampling audit: {len(patients)} accepted, {len(errors)} failed; reports: {args.out_dir}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
