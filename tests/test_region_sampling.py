"""Metadata-only diagnostics and a regression demonstrating crop-size sensitivity."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from trifusesurv2.preprocessing.compartments import ReferenceGeometry, build_compartments, mask_checksum, region_identity

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = script("audit_region_sampling")
preprocess = script("preprocess_roi_inputs")


class SamplingAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def manifest(self, *, patient="001", scan="002", disappear=False, crop_fraction=1., spacing=1.):
        geometry = ReferenceGeometry((31, 31, 31), (1., 1., 1.))
        model_geometry = ReferenceGeometry((31, 31, 31), (spacing,) * 3)
        pt = np.zeros(geometry.shape_dhw, dtype=bool)
        pt[15, 15, 15] = True
        masks, manifest = build_compartments(pt, np.zeros_like(pt), np.ones_like(pt), geometry,
            patient_id=patient, scan_id=scan, contour_source="synthetic", contour_version="v1",
            valid_tissue_source="synthetic full FOV")
        manifest["image"] = {"geometry": model_geometry.to_dict(), "scan_id": scan,
                             "path": "DO_NOT_OPEN_CT.nii.gz", "sha256": "not-payload-verified"}
        for name, support in masks.items():
            record = manifest["regions"][name]
            if disappear and name == "pt_intra":
                support = np.zeros_like(support)
            checksum = mask_checksum(support)
            record.update(image_geometry=model_geometry.to_dict(), image_mask_sha256=checksum,
                          image_mask_path="DO_NOT_OPEN_MASK.nii.gz",
                          image_region_id=region_identity(record, mask_sha256=checksum, geometry=model_geometry))
            volume = float(support.sum()) * model_geometry.voxel_volume_mm3
            record["feature_validity"]["image"] = bool(support.any())
            record["crop_completeness"] = {
                "fraction": crop_fraction, "image_volume_mm3": volume,
                "image_to_reference_volume_ratio": volume / record["physical_volume_mm3"] if record["voxel_count"] else None,
            }
        path = self.root / f"{patient}-{scan}.json"
        path.write_text(json.dumps(manifest))
        return path, manifest

    def test_audit_never_opens_images_and_distinguishes_absence_from_loss(self):
        path, _ = self.manifest(disappear=True, crop_fraction=.9)
        with patch.object(sitk, "ReadImage", side_effect=AssertionError("must not read image payload")):
            patient, regions = audit.inspect_manifest(path)
        self.assertEqual(patient["patient_id"], "001")
        self.assertEqual(patient["declared_disappearance_count"], 1)
        self.assertEqual(patient["crop_truncated_region_count"], 4)
        by_region = {row["region"]: row for row in regions}
        self.assertTrue(by_region["pt_intra"]["declared_support_disappeared"])
        self.assertFalse(by_region["ln_intra"]["declared_support_disappeared"])
        self.assertIsNone(by_region["ln_intra"]["model_reference_volume_ratio"])
        self.assertIn("not inspected", patient["validation_scope"])

    def test_inconsistent_declared_volume_ratio_is_rejected(self):
        path, manifest = self.manifest()
        manifest["regions"]["pt_intra"]["crop_completeness"]["image_to_reference_volume_ratio"] = 99.
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "volume ratio is inconsistent"):
            audit.inspect_manifest(path)

    def test_metadata_alias_and_numeric_looking_ids_survive_csv_reports(self):
        path, _ = self.manifest()
        metadata = self.root / "metadata.csv"
        with metadata.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["patient_id", "scan_id", "region_manifest_path", "site", "protocol"])
            writer.writeheader()
            writer.writerow(dict(patient_id="001", scan_id="002", region_manifest_path=path.name, site="01", protocol="0003"))
        output = self.root / "audit"
        self.assertEqual(audit.main(["--metadata-csv", str(metadata), "--id_col", "Patient_ID",
                                    "--institution-col", "site", "--protocol-col", "protocol", "--out-dir", str(output)]), 0)
        with (output / "patients.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual((rows[0]["patient_id"], rows[0]["scan_id"], rows[0]["institution"], rows[0]["acquisition_protocol"]),
                         ("001", "002", "01", "0003"))
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["stratified"]["institution"][0]["value"], "01")

    def test_stratification_and_descriptive_burden_are_outcome_free(self):
        patients, regions = [], []
        for index in range(3):
            path, _ = self.manifest(patient=str(index), spacing=float(index + 1))
            patient, patient_regions = audit.inspect_manifest(path, attributes={
                "institution": "A" if index < 2 else "B", "nodal_burden_numeric": float(index)})
            patients.append(patient)
            regions.extend(patient_regions)
        summary = audit.summarize(patients, regions, categorical_fields=("institution",), numeric_burden=True)
        self.assertEqual(summary["stratified"]["institution"][0]["patient_scan_count"], 2)
        association = summary["nodal_burden_descriptive_associations"]["model_spacing_x_mm"]
        self.assertEqual(association["n"], 3)
        self.assertAlmostEqual(association["spearman_r"], 1.)
        self.assertIn("no outcomes", summary["interpretation"])

    def test_corrupt_or_duplicate_manifest_is_reported_not_counted_twice(self):
        path, _ = self.manifest()
        output = self.root / "duplicates"
        result = audit.main(["--manifest", str(path), str(path), "--out-dir", str(output)])
        self.assertEqual(result, 1)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["overall"]["patient_scan_count"], 1)
        self.assertEqual(summary["failed_patient_scans"], 1)

    def test_fixed_shape_resizing_changes_unchanged_primary_sampling_with_remote_node(self):
        # This characterizes the existing crop-size policy; it is not qualification
        # of that policy or a proposal for the synthetic fixture's chosen spacing.
        z, y, x = np.indices((64, 64, 64))
        pt = (z - 24) ** 2 + (y - 24) ** 2 + (x - 24) ** 2 <= 9
        nearby = (z - 24) ** 2 + (y - 24) ** 2 + (x - 34) ** 2 <= 4
        remote = (z - 50) ** 2 + (y - 48) ** 2 + (x - 52) ** 2 <= 4
        original_pt = pt.copy()
        ct = (.5 + .2 * np.sin(x * .7) + .15 * np.cos(y * .9)).astype(np.float32)
        measurements = []
        for ln in (nearby, nearby | remote):
            start, end, _ = preprocess._bbox_from_mask(pt | ln, margin_dhw=(10, 10, 10),
                output_size_dhw=(24, 32, 32), expand_to_aspect=True, fail_empty_roi=True)
            image = preprocess._resample_to_size(preprocess._crop_image(sitk.GetImageFromArray(ct), start, end),
                                                (24, 32, 32), is_mask=False)
            mask = preprocess._resample_to_size(preprocess._crop_image(sitk.GetImageFromArray(pt.astype(np.uint8)), start, end),
                                               (24, 32, 32), is_mask=True)
            support = sitk.GetArrayFromImage(mask) > 0
            measurements.append((image.GetSpacing(), float(support.sum()) * np.prod(image.GetSpacing()),
                                 float(sitk.GetArrayFromImage(image)[support].mean())))
        np.testing.assert_array_equal(pt, original_pt)
        self.assertNotEqual(measurements[0][0], measurements[1][0])
        self.assertNotAlmostEqual(measurements[0][1], measurements[1][1])
        self.assertNotAlmostEqual(measurements[0][2], measurements[1][2])


if __name__ == "__main__":
    unittest.main()
