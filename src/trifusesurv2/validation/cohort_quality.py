"""Join retained cohort evidence without loading images or extracting features.

All identifiers stay literal strings. Outcome exclusion and unavailable anatomy
are independent axes. Reviews are imported as evidence, never synthesized.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from .pilot import checked_file, exact_ids, finite, literal, read_csv, read_json, sha256, source_record, write_json, report_sources

HABITATS = ("pt_intra", "pt_peri", "ln_intra", "ln_peri")
PREFIXES = dict(zip(HABITATS, ("PT_intratumor__", "PT_peritumor_10mm__", "LN_intratumor__", "LN_peritumor_10mm__")))
MISSING = {"", "na", "nan", "null", "none"}


def indexed_rows(path: Path) -> tuple[dict[str, dict], str]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"Empty patient table: {path}")
    columns = list(rows[0])
    candidates = [key for key in columns if re.sub(r"[^a-z0-9]", "", key.casefold()) == "patientid"]
    if len(candidates) != 1:
        raise ValueError(f"Expected one unambiguous Patient_ID/patient_id column: {path}")
    key = candidates[0]
    exact_ids([row[key] for row in rows])
    return {row[key]: row for row in rows}, key


def outcome(row: dict, endpoint="OS") -> dict:
    time, event = row.get(endpoint + ".TIME"), row.get(endpoint + ".EVENT")
    missing = lambda value: value is None or str(value).strip().casefold() in MISSING
    if missing(time) and missing(event):
        return {"valid": False, "state": "paired_missing", "time": None, "event": None,
                "reason": f"primary_{endpoint}_time_and_event_both_missing"}
    if missing(time) or missing(event):
        return {"valid": False, "state": "invalid", "time": time, "event": event, "reason": "incomplete_primary_time_event_pair"}
    try:
        time, event = finite(time), finite(event)
        if time <= 0 or event not in (0, 1):
            raise ValueError("Nonpositive time or nonbinary event")
    except (TypeError, ValueError):
        return {"valid": False, "state": "invalid", "time": row.get(endpoint + ".TIME"),
                "event": row.get(endpoint + ".EVENT"), "reason": "malformed_primary_time_event_pair"}
    return {"valid": True, "state": "observed", "time": time, "event": int(event), "reason": "primary_OS_observed"}


class PathMapper:
    """Explicit read-only prefix mapping; never edits retained evidence paths."""

    def __init__(self, mappings=()):
        self.mappings = sorted(((Path(old), Path(new).absolute()) for old, new in mappings), key=lambda pair: len(pair[0].parts), reverse=True)
        if len({str(old) for old, _ in self.mappings}) != len(self.mappings):
            raise ValueError("Duplicate source path prefix")
        if any(not old.is_absolute() or not new.is_absolute() for old, new in self.mappings):
            raise ValueError("Path maps must use absolute prefixes")

    def resolve(self, value, base: Path | None = None) -> Path:
        path = Path(value)
        if not path.is_absolute():
            if base is None:
                raise ValueError("Relative evidence path has no declared base")
            path = base / path
        for old, new in self.mappings:
            if path.is_relative_to(old):
                path = new / path.relative_to(old)
                break
        # '..' paths must not evade manifest-root or mapped-root containment.
        if ".." in path.parts:
            raise ValueError("Parent traversal is not permitted in evidence paths")
        return path


def distribution(values) -> dict:
    values = sorted(finite(value) for value in values if value is not None)
    if not values:
        return {"n": 0, "minimum": None, "median": None, "maximum": None, "p05": None, "p95": None}
    def quantile(q):
        position = (len(values) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)
    return {"n": len(values), "minimum": values[0], "median": quantile(.5), "maximum": values[-1], "p05": quantile(.05), "p95": quantile(.95)}


def qualified_volume_distribution(rows: list[dict], field: str) -> dict:
    """Describe known-present reference anatomy, retaining resampling loss.

    Missing/invalid compartments can carry zero placeholders in manifests.
    Those values are not measured anatomical zero volumes. A model-grid zero
    for a qualified, present reference region is measured disappearance and
    remains in the model-volume and ratio distributions.
    """
    qualified = [row for row in rows if row.get("anatomical_state") == "present"
                 and row.get("support_valid") is True and row.get("geometry_declaration_status") == "passed"
                 and row.get("reference_volume_mm3") is not None and finite(row["reference_volume_mm3"]) > 0]
    result = distribution(row.get(field) for row in qualified)
    qualified_ids = {id(row) for row in qualified}
    return {**result, "population": "present_reference_anatomy_with_valid_support_and_passed_geometry_declarations",
            "population_count": len(qualified), "total_compartment_observations": len(rows),
            "excluded_from_population_count": len(rows) - len(qualified),
            "excluded_anatomical_states": dict(Counter(row.get("anatomical_state", "unassessable")
                                                       for row in rows if id(row) not in qualified_ids)),
            "missing_measurement_count": len(qualified) - result["n"],
            "model_grid_disappearance_zeros_retained": True}


def region_rows(manifest: dict, radiomics: dict, *, config_sha256: str, ratio_bounds=(.8, 1.2)) -> list[dict]:
    """Validate declarations and return all four states, including unavailable ones."""
    from trifusesurv2.preprocessing.compartments import validate_region_manifest, assert_same_geometry, ReferenceGeometry
    from trifusesurv2.preprocessing.sampling import validate_sampling_metadata
    from trifusesurv2.preprocessing.region_io import validate_radiomics_binding
    validate_region_manifest(manifest)
    validate_sampling_metadata(manifest)
    validate_radiomics_binding(radiomics, manifest)
    if radiomics.get("radiomics_config_sha256") != config_sha256:
        raise ValueError("Radiomics extraction configuration differs from development receipt")
    if manifest["image"]["scan_id"] != manifest["scan_id"]:
        raise ValueError("Model image acquisition identity differs")
    model_geometry = ReferenceGeometry.from_dict(manifest["image"]["geometry"])
    output = []
    for habitat in HABITATS:
        record = manifest["regions"][habitat]
        crop = record["crop_completeness"]
        assert_same_geometry(model_geometry, record["image_geometry"])
        reference, model = finite(record["physical_volume_mm3"]), finite(crop["image_volume_mm3"])
        if reference < 0 or model < 0:
            raise ValueError("Negative support volume")
        ratio = model / reference if reference > 0 else None
        declared_ratio = crop.get("image_to_reference_volume_ratio")
        if (ratio is None) != (declared_ratio is None) or (ratio is not None and not math.isclose(ratio, finite(declared_ratio), rel_tol=1e-6, abs_tol=1e-9)):
            raise ValueError("Declared model/reference volume ratio differs")
        fraction = finite(crop["fraction"]) if crop.get("fraction") is not None else None
        if fraction is not None and not 0 <= fraction <= 1 + 1e-9:
            raise ValueError("Invalid retained reference fraction")
        image_available = record["feature_validity"]["image"]
        if not isinstance(image_available, bool) or image_available != bool(record["support_valid"] and model > 0):
            raise ValueError("Image availability contradicts model support")
        if record["status"] != "present" and model > 0:
            raise ValueError("Unavailable/absent anatomy has positive model support")
        disappeared = record["status"] == "present" and reference > 0 and model == 0
        status = radiomics[f"feature_status__{habitat}"]
        features = [value for key, value in radiomics.items() if key.startswith(PREFIXES[habitat])]
        if status == "valid" and (not features or any(not math.isfinite(float(value)) for value in features)):
            raise ValueError(f"{habitat}: valid radiomics has missing/nonfinite features")
        error = radiomics.get(f"feature_error__{habitat}", "")
        extraction_failure = bool(error) or (status == "invalid" and image_available)
        shell_unknown = habitat.endswith("_peri") and record["shell_definition"].get("all_tumor_exclusion_known") is False
        flags = []
        if disappeared:
            flags.append("resampling_disappearance")
        if ratio is not None and not ratio_bounds[0] <= ratio <= ratio_bounds[1]:
            flags.append("unusual_model_reference_volume_ratio")
        if extraction_failure:
            flags.append("radiomics_extraction_failure")
        if record.get("reference_boundary_truncated") is True:
            flags.append("source_boundary_truncation")
        if fraction is not None and fraction < 1 - 1e-9:
            flags.append("crop_truncation")
        output.append({"patient_id": manifest["patient_id"], "scan_id": manifest["scan_id"], "compartment": habitat,
                       "anatomical_state": record["status"], "model_support_state": "invalid" if disappeared else record["status"],
                       "annotation_available": record["annotation_available"], "support_valid": record["support_valid"],
                       "image_feature_available": image_available, "radiomics_feature_available": status == "valid",
                       "reference_volume_mm3": reference, "model_volume_mm3": model, "model_reference_volume_ratio": ratio,
                       "resampling_disappearance": disappeared, "source_fov_truncated": record.get("reference_boundary_truncated"),
                       "retained_reference_fraction": fraction, "shell_exclusion_anatomy_unknown": shell_unknown,
                       "extraction_status": status, "extraction_error": error,
                       "extraction_failure": extraction_failure, "extracted_finite_feature_count": len(features) if status == "valid" else 0,
                       "sampling_policy": manifest["sampling"]["policy"], "sampling_policy_id": manifest["sampling"]["policy_id"],
                       "radiomics_config_sha256": radiomics["radiomics_config_sha256"], "radiomics_software": radiomics["radiomics_software"],
                       "reference_region_id": record["region_id"], "model_region_id": record["image_region_id"],
                       "support_sha256": record["image_mask_sha256"], "geometry_declaration_status": "passed",
                       "review_reasons": ";".join(flags)})
    return output


def summarize_cohort(*, metadata_csv: Path, source_metadata_csv: Path, radiomics_csv: Path,
                     development_receipt: Path, builder_receipt: Path, path_mappings=(),
                     sampling_audit_dir: Path | None = None, visual_review_receipt: Path | None = None,
                     representative_receipt: Path | None = None,
                     expected_counts=(613, 519, 94), ratio_bounds=(.8, 1.2), ordinary_sample_size=5) -> dict:
    if not 0 < ratio_bounds[0] < ratio_bounds[1] or ordinary_sample_size < 0:
        raise ValueError("Invalid review sampling configuration")
    mapper = PathMapper(path_mappings)
    sources = {name: source_record(checked_file(path)) for name, path in {
        "metadata": metadata_csv, "source_metadata": source_metadata_csv, "radiomics": radiomics_csv,
        "development_receipt": development_receipt, "builder_receipt": builder_receipt}.items()}
    development, builder = read_json(development_receipt), read_json(builder_receipt)
    for name, field in (("metadata", "metadata_csv_sha256"), ("source_metadata", "source_csv_sha256"), ("radiomics", "radiomics_csv_sha256")):
        if sources[name]["sha256"] != development.get(field):
            raise ValueError(f"{name} bytes differ from development receipt")
    if development.get("status") != "prepared_development_only" or development.get("errors") or development.get("audit_exit_code") != 0:
        raise ValueError("Development receipt does not describe a completed, audited delivery")
    if development.get("selection_uses_outcomes") is not False:
        raise ValueError("Preparation selection was not declared outcome independent")
    metadata, _ = indexed_rows(metadata_csv)
    source, source_id_col = indexed_rows(source_metadata_csv)
    radiomics, _ = indexed_rows(radiomics_csv)
    exact_ids(list(metadata), development["selected_patient_ids"])
    exact_ids(list(radiomics), list(metadata))
    exact_ids(list(development["source_manifests"]), list(metadata))
    if set(metadata) - set(source):
        raise ValueError("Prepared patient absent from original source metadata")
    eligibility_ref = builder["outcome_eligibility_receipt"]
    eligibility_path = checked_file(mapper.resolve(eligibility_ref["path"], builder_receipt.parent), digest=eligibility_ref["sha256"])
    sources["outcome_eligibility"] = source_record(eligibility_path)
    prior_eligibility = read_json(eligibility_path)
    exact_ids(prior_eligibility["prepared_patient_ids"], list(metadata))
    if prior_eligibility["prepared_metadata"]["sha256"] != sources["metadata"]["sha256"] or prior_eligibility["prepared_radiomics"]["sha256"] != sources["radiomics"]["sha256"]:
        raise ValueError("Eligibility was defined for different prepared bytes")
    splits = builder["split_patient_ids"]
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Expected frozen train/val/test inventories")
    split_by_id = {}
    for part, ids in splits.items():
        exact_ids(ids)
        for pid in ids:
            if pid in split_by_id:
                raise ValueError("Patient repeated across splits")
            split_by_id[pid] = part
    exact_ids(list(split_by_id), builder["selected_patient_ids"])
    exact_ids(list(split_by_id), prior_eligibility["eligible_patient_ids"])
    visual = read_json(visual_review_receipt) if visual_review_receipt else {}
    if visual_review_receipt:
        sources["visual_review"] = source_record(checked_file(visual_review_receipt))
    reviewed, review_applicability = {}, {}
    historical_panels = visual.get("qc_overlays", {}) if visual.get("status") == "passed_development_geometry_review" else {}
    prior_manifests = {}
    if representative_receipt is not None:
        prior_ref = visual.get("representative_receipt", {})
        checked_file(representative_receipt, digest=prior_ref["sha256"])
        sources["representative_receipt"] = source_record(representative_receipt)
        prior_manifests = read_json(representative_receipt).get("source_manifests", {})
    for pid, old_panel in historical_panels.items():
        current_panel = development.get("qc_overlays", {}).get(pid, {})
        result = {"status": "unassessable", "panel_hash_matches": False, "manifest_hash_matches": False}
        try:
            panel_path = checked_file(mapper.resolve(current_panel["path"], development_receipt.parent), digest=current_panel["sha256"])
            result["panel_hash_matches"] = sha256(panel_path) == old_panel["sha256"]
            result["manifest_hash_matches"] = prior_manifests.get(pid, {}).get("sha256") == development["source_manifests"][pid]["sha256"]
            result["status"] = "historical_evidence_matches" if result["panel_hash_matches"] and result["manifest_hash_matches"] else "not_applicable_to_current_manifest"
            if result["status"] == "historical_evidence_matches":
                reviewed[pid] = old_panel
        except (KeyError, ValueError, OSError) as error:
            result["reason"] = str(error)
        review_applicability[pid] = result
    audit_patients, audit_regions, audit_errors = {}, {}, {}
    if sampling_audit_dir is not None:
        for name in ("patients", "regions", "errors"):
            path = checked_file(sampling_audit_dir / f"{name}.csv")
            sources[f"sampling_audit_{name}"] = source_record(path)
            # Empty errors.csv may be an empty file, following the old writer.
            rows = read_csv(path) if path.stat().st_size > 2 else []
            for row in rows:
                pid = literal(row["patient_id"])
                if name == "patients":
                    if pid in audit_patients:
                        raise ValueError("Duplicate sampling patient audit")
                    audit_patients[pid] = row
                elif name == "regions":
                    key = (pid, row["region"])
                    if key in audit_regions:
                        raise ValueError("Duplicate sampling region audit")
                    audit_regions[key] = row
                else:
                    audit_errors.setdefault(pid, []).append(row)
        unknown = (set(audit_patients) | set(audit_errors) | {pid for pid, _ in audit_regions}) - set(metadata)
        if unknown:
            raise ValueError("Sampling audit contains foreign patients")
    patients, regions, eligibility, errors, queue = [], [], [], [], []
    for pid, row in metadata.items():
        scan = literal(row["scan_id"])
        if radiomics[pid].get("scan_id") != scan:
            raise ValueError(f"{pid}: radiomics scan identity differs")
        primary = outcome(row)
        if outcome(source[pid]) != primary:
            raise ValueError(f"{pid}: primary outcome changed from original source metadata")
        if primary["state"] == "invalid":
            errors.append({"patient_id": pid, "type": "invalid_outcome", "reason": primary["reason"]})
        expected_split = split_by_id.get(pid)
        if primary["valid"] != (expected_split is not None):
            errors.append({"patient_id": pid, "type": "eligibility_mismatch", "reason": "Frozen split membership differs from OS-valid inclusion"})
        eligibility.append({"patient_id": pid, "scan_id": scan, "include": primary["valid"], "reason": primary["reason"],
                            "split": expected_split or "excluded", "policy_version": "frozen_pilot_OS_v1",
                            "anatomy_unavailability_is_exclusion": False,
                            "frozen_inventory_agrees": primary["valid"] == (expected_split is not None)})
        patient = {"patient_id": pid, "scan_id": scan, "source_patient_id_column": source_id_col,
                   "source_metadata_path": development["source_csv"], "source_metadata_sha256": development["source_csv_sha256"],
                   "source_history_status": row.get("source_history_status", "unavailable"),
                   "contour_source": row.get("contour_source", "unavailable"), "contour_version": row.get("contour_version", "unavailable"),
                   "ct_intensity_transform": row.get("ct_intensity_transform", "unavailable"),
                   "valid_tissue_definition": row.get("valid_tissue_definition", "unavailable"),
                   "os_time": primary["time"], "os_event": primary["event"], "os_valid": primary["valid"], "os_state": primary["state"],
                   "split": expected_split or "excluded", "geometry_declaration_status": "unassessable"}
        for name in ("ct_series_uid", "rtstruct_referenced_series_uid", "ct_num_slices", "ct_spacing_xyz", "ct_size_xyz",
                     "manufacturer", "manufacturer_model_name", "convolution_kernel", "slice_thickness", "acquisition_date"):
            patient[name] = source[pid].get(name) or row.get(name) or "unavailable"
        patient["unresolved_provenance"] = ";".join(development.get("limitations", []))
        record = development["source_manifests"][pid]
        manifest_path = mapper.resolve(record["path"], development_receipt.parent)
        declared_path = mapper.resolve(row["region_manifest_path"], metadata_csv.parent)
        patient.update(manifest_path=str(manifest_path), manifest_sha256=record["sha256"])
        patient_reasons = []
        try:
            if manifest_path.absolute() != declared_path.absolute():
                raise ValueError("Metadata and development receipt name different manifests")
            manifest = read_json(checked_file(manifest_path, digest=record["sha256"]))
            if manifest["patient_id"] != pid or manifest["scan_id"] != scan:
                raise ValueError("Manifest patient or acquisition identity differs")
            if manifest["sampling"]["policy_id"] != development["sampling_policy_id"] or row["sampling_policy_id"] != development["sampling_policy_id"]:
                raise ValueError("Sampling policy differs from frozen development delivery")
            if pid in audit_patients and audit_patients[pid].get("manifest_sha256") != record["sha256"]:
                raise ValueError("Retained sampling audit binds a different manifest")
            current = region_rows(manifest, radiomics[pid], config_sha256=development["radiomics_config_sha256"], ratio_bounds=ratio_bounds)
            patient["geometry_declaration_status"] = "passed"
            for region in current:
                region["manifest_sha256"] = record["sha256"]
                region["sampling_audit_status"] = "retained_declaration_audit" if (pid, region["compartment"]) in audit_regions else "not_available; current_declaration_checks_only"
                patient_reasons.extend(filter(None, region["review_reasons"].split(";")))
            regions.extend(current)
        except (ValueError, KeyError, TypeError, OSError) as error:
            patient["geometry_declaration_status"] = "failed"
            patient_reasons.append("failed_geometry_or_feature_binding")
            errors.append({"patient_id": pid, "type": "geometry_or_feature_binding", "reason": str(error)})
            # Preserve all four observations, without inventing absent anatomy.
            for habitat in HABITATS:
                regions.append({"patient_id": pid, "scan_id": scan, "compartment": habitat,
                                "anatomical_state": "unassessable", "model_support_state": "unassessable",
                                "geometry_declaration_status": "failed", "review_reasons": "failed_geometry_or_feature_binding",
                                "extraction_status": radiomics[pid].get(f"feature_status__{habitat}", "unassessable"),
                                "extraction_error": str(error), "manifest_sha256": record["sha256"]})
        if pid in audit_errors:
            patient_reasons.append("retained_sampling_audit_failure")
            errors.append({"patient_id": pid, "type": "retained_sampling_audit_failure", "reason": json.dumps(audit_errors[pid], sort_keys=True)})
        patient["review_reasons"] = ";".join(sorted(set(patient_reasons)))
        patients.append(patient)
    actual_counts = (len(patients), sum(row["os_valid"] for row in patients), sum(row["os_state"] == "paired_missing" for row in patients))
    recorded_counts = (builder["prepared_patient_count"], builder["eligible_patient_count"], builder["excluded_primary_missing_count"])
    if actual_counts != tuple(expected_counts) or actual_counts != recorded_counts or len(metadata) != development["prepared_cases"]:
        errors.append({"patient_id": "", "type": "cohort_counts", "reason": f"Observed={actual_counts}, expected={tuple(expected_counts)}, builder={recorded_counts}"})
    actual_exclusions = {row["patient_id"]: row["reason"] for row in eligibility if not row["include"]}
    old_exclusions = {literal(row["patient_id"]): row["reason"] for row in prior_eligibility["exclusions"]}
    if len(old_exclusions) != len(prior_eligibility["exclusions"]) or actual_exclusions != old_exclusions:
        errors.append({"patient_id": "", "type": "exclusion_reconciliation", "reason": "Patient-level exclusions differ from frozen outcome eligibility receipt"})
    ordinary = sorted((row for row in patients if not row["review_reasons"]), key=lambda row: (hashlib.sha256(("cohort_qc_v1:" + row["patient_id"]).encode()).hexdigest(), row["patient_id"]))[:ordinary_sample_size]
    ordinary_ids = {row["patient_id"] for row in ordinary}
    for patient in patients:
        pid = patient["patient_id"]
        if not patient["review_reasons"] and pid not in ordinary_ids:
            continue
        overlay = development.get("qc_overlays", {}).get(pid, {})
        legacy = reviewed.get(pid, {})
        # A representative overlay can concern a different preparation. Import
        # that observation as historical; never close the new review queue.
        queue.append({"patient_id": pid, "scan_id": patient["scan_id"], "reasons": patient["review_reasons"] or "ordinary_deterministic_sample",
                      "review_status": "pending", "reviewer": "", "reviewed_at_utc": "", "review_decision": "",
                      "current_overlay_path": overlay.get("path", ""), "current_overlay_sha256": overlay.get("sha256", ""),
                      "current_overlay_status": overlay.get("status", "not_available"),
                      "historical_review_status": visual.get("status", "") if legacy else "not_available",
                      "historical_review_applicability": review_applicability.get(pid, {}).get("status", "not_available"),
                      "historical_overlay_path": legacy.get("path", ""), "historical_overlay_sha256": legacy.get("sha256", ""),
                      "historical_review_limitation": visual.get("limitation", "") if legacy else ""})
    grouped = {habitat: [row for row in regions if row["compartment"] == habitat] for habitat in HABITATS}
    summaries = {}
    for habitat, rows in grouped.items():
        summaries[habitat] = {"count": len(rows), "anatomical_states": dict(Counter(row["anatomical_state"] for row in rows)),
                              "image_available": sum(row.get("image_feature_available") is True for row in rows),
                              "radiomics_available": sum(row.get("radiomics_feature_available") is True for row in rows),
                              "disappearances": sum(row.get("resampling_disappearance") is True for row in rows),
                              "source_fov_truncated": sum(row.get("source_fov_truncated") is True for row in rows),
                              "extraction_failures": sum(row.get("extraction_failure") is True for row in rows),
                              **{key: qualified_volume_distribution(rows, key) for key in ("reference_volume_mm3", "model_volume_mm3", "model_reference_volume_ratio")}}
    by_region = {(row["patient_id"], row["compartment"]): row for row in regions}
    intersections = {"pt_shell_unavailable_with_unknown_nodal_exclusion": [], "ln_interior_disappeared_with_measurable_shell": []}
    for pid in metadata:
        pt_shell, ln, ln_shell = (by_region[(pid, habitat)] for habitat in ("pt_peri", "ln_intra", "ln_peri"))
        if (pt_shell.get("shell_exclusion_anatomy_unknown") and ln.get("annotation_available") is False
                and pt_shell.get("anatomical_state") == "invalid" and pt_shell.get("annotation_available") is True
                and pt_shell.get("image_feature_available") is False):
            intersections["pt_shell_unavailable_with_unknown_nodal_exclusion"].append(pid)
        if ln.get("resampling_disappearance") and ln_shell.get("image_feature_available"):
            intersections["ln_interior_disappeared_with_measurable_shell"].append(pid)
    return {"schema_version": 1, "status": "failed" if errors else "accounted_review_pending" if queue else "accounted",
            "accounting_gate": "failed" if errors else "passed", "review_gate": "pending" if queue else "no_cases_requested",
            "clinical_qualification": "not_established", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sources": sources, "path_mappings": [{"source": str(old), "local": str(new)} for old, new in mapper.mappings],
            "counts": dict(zip(("prepared", "eligible", "paired_missing_primary_outcomes"), actual_counts)),
            "split_counts": dict(Counter(row["split"] for row in eligibility)), "compartments": summaries,
            "intersections": {key: {"count": len(ids), "patient_ids": ids} for key, ids in intersections.items()},
            "review_policy": {"ratio_bounds": list(ratio_bounds), "ordinary_sample_size": ordinary_sample_size,
                              "threshold_purpose": "review prioritization only; no patient exclusion", "ordinary_sampling": "SHA256(cohort_qc_v1:literal_patient_id), outcome independent"},
            "measurement_policy": {"sampling_policy_id": development["sampling_policy_id"], "radiomics_config_sha256": development["radiomics_config_sha256"],
                                   "proposal": "Retain frozen measurement policy and OS eligibility for development; compartment missingness is represented, not an outcome exclusion. Version inventories with original assignments if later adjudication requires exclusions."},
            "limitations": development.get("limitations", []),
            "scope": "Retained metadata, manifest declarations and tabular radiomics; historical PNG checksums where supplied. No CT/mask-byte validation, new feature extraction, new visual review, or clinical source qualification.",
            "historical_review": {"receipt": sources.get("visual_review"),
                                  "findings": visual.get("findings", []) if historical_panels and len(reviewed) == len(historical_panels) else [],
                                  "patient_ids": sorted(reviewed), "applicability_by_patient": review_applicability,
                                  "scope": "Historical findings are imported only when both copied panel and manifest hashes match; no new whole-cohort review is implied."},
            "patients": patients, "regions": regions, "eligibility": eligibility, "review_queue": queue, "errors": errors,
            "postprocessor": source_record(Path(__file__)),
            "postprocessor_sources": report_sources("scripts/summarize_cohort_quality.py", "src/trifusesurv2/validation/cohort_quality.py",
                "src/trifusesurv2/validation/pilot.py", "src/trifusesurv2/preprocessing/compartments.py",
                "src/trifusesurv2/preprocessing/sampling.py", "src/trifusesurv2/preprocessing/region_io.py")}


def write_csv(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["patient_id", "status"]
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_cohort_report(report: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=False)
    for key, filename in (("patients", "cohort_qc_patients.csv"), ("regions", "cohort_qc_regions.csv"),
                          ("eligibility", "cohort_eligibility.csv"), ("review_queue", "cohort_review_queue.csv"), ("errors", "cohort_qc_errors.csv")):
        write_csv(out_dir / filename, report[key])
    summary = {key: value for key, value in report.items() if key not in {"patients", "regions", "eligibility", "review_queue"}}
    summary["review_queue_count"] = len(report["review_queue"])
    summary["outputs"] = {path.name: source_record(path) for path in out_dir.glob("*.csv")}
    write_json(out_dir / "cohort_qualification.json", summary)
    counts = report["counts"]
    lines = ["# Cohort accounting and qualification", "", f"Accounting: **{report['accounting_gate']}**. Review: **{report['review_gate']}**. Clinical qualification is not established.", "",
             f"{counts['prepared']} prepared patients; {counts['eligible']} OS-valid; {counts['paired_missing_primary_outcomes']} excluded for paired-missing OS outcomes. These exclusions do not imply missing primary tumors.", "",
             "Frozen split counts: " + json.dumps(report["split_counts"], sort_keys=True) + ".", "",
             "| Compartment | States | Image available | Radiomics available | Disappeared | Source FOV flag |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for habitat, value in report["compartments"].items():
        lines.append(f"| {habitat} | {json.dumps(value['anatomical_states'], sort_keys=True)} | {value['image_available']} | {value['radiomics_available']} | {value['disappearances']} | {value['source_fov_truncated']} |")
    lines.extend(["", "Volume and ratio distributions describe only present reference anatomy with valid support and passed geometry declarations. Missing/invalid/absent anatomy is excluded from these distributions; true model-grid zeros from disappearance of known-present regions remain included. Exact population and measurement denominators are in cohort_qualification.json.", ""])
    for name, value in report["intersections"].items():
        lines.append(f"- {name}: {value['count']} patients.")
    lines.extend(["", f"The queue contains {len(report['review_queue'])} cases. Every new queue decision remains pending; retained representative reviews are linked as historical evidence only.", "",
                  report["measurement_policy"]["proposal"], "", report["scope"], ""])
    lines.extend("- " + str(item) for item in report["limitations"])
    if report["historical_review"]["findings"]:
        lines.extend(["", "Retained representative review findings:", ""])
        lines.extend("- " + str(item) for item in report["historical_review"]["findings"])
    if report["errors"]:
        lines.extend(["", f"{len(report['errors'])} accounting/quality errors are listed in cohort_qc_errors.csv; do not promote this report as a passed cohort gate."])
    (out_dir / "cohort_qualification.md").write_text("\n".join(lines) + "\n")
