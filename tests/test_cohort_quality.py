"""Fabricated metadata and tiny image fixtures; no cohort processing or extraction."""
import copy
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import SimpleITK as sitk

from trifusesurv2.preprocessing.region_io import canonical_json_sha256
from trifusesurv2.validation.cohort_quality import (
    HABITATS, PREFIXES, PathMapper, indexed_rows, outcome, region_rows, qualified_volume_distribution,
    summarize_cohort, write_cohort_report, write_csv,
)
from trifusesurv2.validation.pilot import sha256, write_json

SPEC = importlib.util.spec_from_file_location("quality_fixture_preprocess", Path(__file__).resolve().parents[1] / "scripts/preprocess_roi_inputs.py")
PREPROCESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPROCESS)


def make_case(root, pid, *, node_available=False):
    source = root / "inputs" / pid; source.mkdir(parents=True)
    shape = (32, 32, 32)
    pt = np.zeros(shape, dtype=np.uint8); pt[14:17, 14:17, 14:17] = 1
    ln = np.zeros(shape, dtype=np.uint8); ln[18:20, 18:20, 18:20] = 1
    row = {"patient_id": pid, "scan_id": "scan-" + pid, "contour_source": "synthetic", "contour_version": "fixture-v1",
           "ct_intensity_transform": "synthetic normalized", "radiomics_scan_id": "scan-" + pid,
           "radiomics_intensity_transform": "synthetic HU", "ln_annotation_available": node_available}
    for key, array in (("ct", np.full(shape, .5, dtype=np.float32)), ("pt", pt), ("ln", ln), ("hu", np.full(shape, -400, dtype=np.float32))):
        path = source / f"{key}.mha"; sitk.WriteImage(sitk.GetImageFromArray(array), str(path)); row[key] = str(path)
    row["radiomics_ct_path"] = row["hu"]
    grid = {"schema_version": 1, "spacing_xyz": [1., 1., 1.], "origin_xyz": [0., 0., 0.],
            "direction": [1., 0., 0., 0., 1., 0., 0., 0., 1.], "units": "mm", "coordinate_system": "LPS"}
    output = root / "prepared"
    job = dict(row=row, index=0, id_col="patient_id", meta_dir=str(source), out_root=str(output), out_csv_dir=str(output),
               output_size_dhw=(32, 32, 32), region_mode="verified", sampling_policy="common_grid", grid_config=grid,
               margin_mm=10., valid_tissue_policy="full_fov", overwrite=False, expand_to_aspect=True,
               fail_empty_roi=True, ct_col="ct", mask_pt_col="pt", mask_ln_col="ln")
    prepared = PREPROCESS._process_one(job)
    manifest_path = output / prepared["region_manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    config = {"setting": {"binWidth": 25}, "featureClass": {"firstorder": ["Mean"]}}
    rad = {"patient_id": pid, "scan_id": row["scan_id"], "sampling_policy_id": manifest["sampling"]["policy_id"],
           "radiomics_image_sha256": manifest["radiomics_image"]["sha256"], "radiomics_intensity_transform": "synthetic HU",
           "radiomics_config_json": json.dumps(config), "radiomics_config_sha256": canonical_json_sha256(config), "radiomics_software": "fabricated fixture (no extractor run)"}
    for habitat in HABITATS:
        record = manifest["regions"][habitat]
        rad[f"region_id__{habitat}"] = record["image_region_id"]
        rad[f"support_sha256__{habitat}"] = record["image_mask_sha256"]
        rad[f"feature_status__{habitat}"] = "valid" if record["feature_validity"]["image"] else record["status"]
        rad[PREFIXES[habitat] + "fixture_value"] = "1.0" if record["feature_validity"]["image"] else ""
    return {"patient_id": pid, "scan_id": row["scan_id"], "region_manifest_path": prepared["region_manifest_path"],
            "sampling_policy_id": manifest["sampling"]["policy_id"]}, manifest_path, manifest, rad


@pytest.fixture
def cohort(tmp_path):
    records, radiomics, manifests = [], [], {}
    for i, pid in enumerate(("001", "1", "NA", "excluded")):
        row, path, manifest, rad = make_case(tmp_path, pid)
        row.update({"OS.TIME": "" if i == 3 else str(100 + i), "OS.EVENT": "" if i == 3 else str(i % 2)})
        records.append(row); radiomics.append(rad)
        manifests[pid] = {"path": str(path.relative_to(tmp_path / "prepared")), "sha256": sha256(path)}
    root = tmp_path / "prepared"
    metadata, original, features = root / "metadata.csv", root / "original.csv", root / "radiomics.csv"
    write_csv(metadata, records); write_csv(original, records); write_csv(features, radiomics)
    development = root / "development.json"
    dev = {"status": "prepared_development_only", "errors": [], "audit_exit_code": 0, "selection_uses_outcomes": False,
           "metadata_csv_sha256": sha256(metadata), "source_csv": str(original), "source_csv_sha256": sha256(original),
           "radiomics_csv_sha256": sha256(features), "selected_patient_ids": [row["patient_id"] for row in records],
           "source_manifests": manifests, "sampling_policy_id": records[0]["sampling_policy_id"], "prepared_cases": 4,
           "radiomics_config_sha256": radiomics[0]["radiomics_config_sha256"], "limitations": ["fabricated fixture only"]}
    write_json(development, dev)
    eligibility = root / "eligibility.json"
    write_json(eligibility, {"prepared_patient_ids": list(manifests), "eligible_patient_ids": ["001", "1", "NA"],
                            "prepared_metadata": {"sha256": sha256(metadata)}, "prepared_radiomics": {"sha256": sha256(features)},
                            "exclusions": [{"patient_id": "excluded", "reason": "primary_OS_time_and_event_both_missing"}]})
    builder = root / "builder.json"
    write_json(builder, {"outcome_eligibility_receipt": {"path": str(eligibility), "sha256": sha256(eligibility)},
                         "split_patient_ids": {"train": ["001"], "val": ["1"], "test": ["NA"]},
                         "selected_patient_ids": ["001", "1", "NA"], "prepared_patient_count": 4,
                         "eligible_patient_count": 3, "excluded_primary_missing_count": 1})
    return dict(metadata_csv=metadata, source_metadata_csv=original, radiomics_csv=features,
                development_receipt=development, builder_receipt=builder, expected_counts=(4, 3, 1))


def test_all_patients_and_compartments_accounted_without_loading_images(cohort, tmp_path):
    with patch.object(sitk, "ReadImage", side_effect=AssertionError("Metadata report must not load images")):
        report = summarize_cohort(**cohort)
    assert report["accounting_gate"] == "passed", report["errors"]
    assert report["counts"] == {"prepared": 4, "eligible": 3, "paired_missing_primary_outcomes": 1}
    assert len(report["regions"]) == 16
    assert {row["patient_id"] for row in report["patients"]} == {"001", "1", "NA", "excluded"}
    assert report["intersections"]["pt_shell_unavailable_with_unknown_nodal_exclusion"]["count"] == 4
    assert all(row["include"] for row in report["eligibility"] if row["patient_id"] != "excluded")
    assert all(row["review_status"] == "pending" and not row["review_decision"] for row in report["review_queue"])
    output = tmp_path / "report"; write_cohort_report(report, output)
    assert len(indexed_rows(output / "cohort_eligibility.csv")[0]) == 4
    with pytest.raises(FileExistsError):
        write_cohort_report(report, output)


def test_failed_manifest_is_four_unknown_observations_not_absent_tumor(cohort):
    path = cohort["development_receipt"].parent / "001" / "region_manifest.json"
    # Locate the receipt-bound manifest without assuming the builder's directory layout.
    dev = json.loads(cohort["development_receipt"].read_text())
    path = cohort["development_receipt"].parent / dev["source_manifests"]["001"]["path"]
    path.write_text("{}")
    report = summarize_cohort(**cohort)
    assert report["accounting_gate"] == "failed"
    assert len(report["regions"]) == 16
    assert {row["anatomical_state"] for row in report["regions"] if row["patient_id"] == "001"} == {"unassessable"}
    assert next(row for row in report["eligibility"] if row["patient_id"] == "001")["include"]


def test_corrupt_input_table_fails_before_joins(cohort):
    with cohort["radiomics_csv"].open("a") as handle:
        handle.write("changed\n")
    with pytest.raises(ValueError, match="bytes differ"):
        summarize_cohort(**cohort)


def test_duplicate_literal_ids_rejected(tmp_path):
    path = tmp_path / "ids.csv"; write_csv(path, [{"Patient_ID": "001"}, {"Patient_ID": "001"}])
    with pytest.raises(ValueError, match="duplicate"):
        indexed_rows(path)


def test_paired_missing_outcomes_differ_from_malformed_pairs():
    assert outcome({"OS.TIME": "NA", "OS.EVENT": ""})["state"] == "paired_missing"
    for row in ({"OS.TIME": "20", "OS.EVENT": ""}, {"OS.TIME": "NaN", "OS.EVENT": "1"},
                {"OS.TIME": "0", "OS.EVENT": "1"}, {"OS.TIME": "20", "OS.EVENT": "2"}):
        assert outcome(row)["state"] == "invalid"


def test_path_mapping_keeps_source_bytes_and_rejects_traversal(tmp_path):
    mapper = PathMapper([("/remote/project", str(tmp_path))])
    assert mapper.resolve("/remote/project/a.json") == tmp_path / "a.json"
    assert mapper.resolve("a.json", tmp_path) == tmp_path / "a.json"
    with pytest.raises(ValueError, match="traversal"):
        mapper.resolve("../outside", tmp_path)


def test_volume_distributions_exclude_unknown_zeros_but_retain_known_disappearance():
    present = {"anatomical_state": "present", "support_valid": True, "geometry_declaration_status": "passed",
               "reference_volume_mm3": 10., "model_volume_mm3": 8., "model_reference_volume_ratio": .8}
    disappeared = {**present, "reference_volume_mm3": 2., "model_volume_mm3": 0., "model_reference_volume_ratio": 0.,
                   "image_feature_available": False, "resampling_disappearance": True}
    unavailable = [{**present, "anatomical_state": state, "support_valid": state == "absent",
                    "reference_volume_mm3": 0., "model_volume_mm3": 0., "model_reference_volume_ratio": None}
                   for state in ("missing", "invalid", "absent")]
    rows = [present, disappeared, *unavailable]
    reference = qualified_volume_distribution(rows, "reference_volume_mm3")
    assert reference["n"] == reference["population_count"] == 2
    assert reference["median"] == 6.
    assert reference["total_compartment_observations"] == 5
    assert reference["excluded_from_population_count"] == 3
    assert reference["excluded_anatomical_states"] == {"missing": 1, "invalid": 1, "absent": 1}
    model = qualified_volume_distribution(rows, "model_volume_mm3")
    assert model["n"] == 2 and model["minimum"] == 0. and model["median"] == 4.
    assert model["missing_measurement_count"] == 0
    ratio = qualified_volume_distribution(rows, "model_reference_volume_ratio")
    assert ratio["minimum"] == 0. and ratio["median"] == .4
    missing_only = qualified_volume_distribution(unavailable, "reference_volume_mm3")
    assert missing_only["n"] == 0 and missing_only["median"] is None
