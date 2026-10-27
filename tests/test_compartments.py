"""Scientific checks for shared physical support and its persisted identity."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk

from trifusesurv2.preprocessing.compartments import (
    ReferenceGeometry, assert_same_geometry, build_compartments, load_region_manifest,
    mask_checksum, physical_geometry_checksum, physical_shell, region_identity,
    validate_region_manifest, write_region_manifest,
)
from trifusesurv2.schema import PHYSICAL_GEOMETRY_FEATURES


class CompartmentTests(unittest.TestCase):
    def geometry(self, shape=(31, 31, 31), spacing=(1.0, 1.0, 1.0), direction=None, origin=(0., 0., 0.)):
        kwargs = {} if direction is None else {"direction": direction}
        return ReferenceGeometry(shape, spacing, origin, **kwargs)

    def build(self, pt, ln, geometry=None, body=None, **kwargs):
        shape = pt.shape if pt is not None else ln.shape
        geometry = self.geometry(shape) if geometry is None else geometry
        body = np.ones(shape, dtype=bool) if body is None else body
        return build_compartments(
            pt, ln, body, geometry, patient_id="001", scan_id="scan-A",
            contour_source="reader-1", contour_version="v1", valid_tissue_source="explicit-test-mask",
            **kwargs,
        )

    def test_anisotropic_shell_is_euclidean_not_rectangular(self):
        pt = np.zeros((15, 15, 15), dtype=bool)
        pt[7, 7, 7] = True
        geometry = self.geometry(pt.shape, spacing=(2., 3., 4.))
        masks, manifest = self.build(pt, np.zeros_like(pt), geometry)
        shell = masks["pt_peri"]
        self.assertTrue(shell[7, 7, 12])  # exactly 10 mm along x
        self.assertFalse(shell[7, 7, 13])  # 12 mm along x
        self.assertFalse(shell[9, 9, 9])  # sqrt(4²+6²+8²) > 10 mm
        self.assertFalse(shell[7, 7, 7])
        coords = np.argwhere(shell)
        distances = np.linalg.norm((coords - [7, 7, 7]) * geometry.spacing_dhw, axis=1)
        self.assertTrue(np.all((distances > 0) & (distances <= 10)))
        self.assertEqual(manifest["regions"]["pt_intra"]["physical_volume_mm3"], 24.0)

    def test_shells_exclude_both_interiors_and_measure_overlap(self):
        pt = np.zeros((31, 31, 31), dtype=bool)
        ln = np.zeros_like(pt)
        pt[15, 15, 12] = True
        ln[15, 15, 18] = True
        body = np.ones_like(pt)
        body[:, :, :8] = False
        masks, manifest = self.build(pt, ln, body=body)
        for name in ("pt_peri", "ln_peri"):
            self.assertFalse(np.any(masks[name] & (pt | ln)))
            self.assertFalse(np.any(masks[name] & ~body))
        overlap = masks["pt_peri"] & masks["ln_peri"]
        self.assertGreater(int(overlap.sum()), 0)
        self.assertEqual(manifest["shell_overlap"]["voxel_count"], int(overlap.sum()))

    def test_empty_shell_stays_empty_without_semantic_substitution(self):
        pt = np.zeros((5, 5, 5), dtype=bool)
        pt[2, 2, 2] = True
        masks, manifest = self.build(pt, np.zeros_like(pt), body=pt)
        self.assertTrue(masks["pt_intra"].any())
        self.assertFalse(masks["pt_peri"].any())
        record = manifest["regions"]["pt_peri"]
        self.assertEqual(record["status"], "absent")
        self.assertTrue(record["support_valid"])
        self.assertFalse(record["feature_validity"]["image"])

    def test_absent_and_missing_annotations_are_distinct(self):
        pt = np.zeros((31, 31, 31), dtype=bool)
        pt[15, 15, 15] = True
        _, known = self.build(pt, np.zeros_like(pt))
        masks, unknown = self.build(pt, None)
        self.assertEqual(known["regions"]["ln_intra"]["status"], "absent")
        self.assertEqual(unknown["regions"]["ln_intra"]["status"], "missing")
        self.assertEqual(unknown["regions"]["pt_peri"]["status"], "invalid")
        self.assertFalse(unknown["regions"]["pt_peri"]["support_valid"])
        self.assertFalse(masks["pt_peri"].any())
        _, no_node_missing_pt = self.build(None, np.zeros_like(pt))
        self.assertEqual(no_node_missing_pt["regions"]["ln_peri"]["status"], "absent")

    def test_geometry_checks_origin_direction_and_shear(self):
        image = sitk.Image([7, 8, 9], sitk.sitkUInt8)
        image.SetSpacing((2., 3., 4.))
        reference = ReferenceGeometry.from_image(image)
        self.assertEqual(reference.shape_dhw, (9, 8, 7))
        assert_same_geometry(reference, image)
        shifted = sitk.Image(image)
        shifted.SetOrigin((1., 0., 0.))
        with self.assertRaisesRegex(ValueError, "origin_xyz"):
            assert_same_geometry(reference, shifted)
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            self.geometry(direction=(1., .2, 0., 0., 1., 0., 0., 0., 1.))
        with self.assertRaisesRegex(ValueError, "positive"):
            self.geometry(spacing=(1., 0., 1.))

    def test_rotated_grid_preserves_physical_distances_and_defines_laterality(self):
        pt = np.zeros((31, 31, 31), dtype=bool)
        ln = np.zeros_like(pt)
        pt[15, 15, 15] = True
        ln[15, 15, 17] = True
        identity = self.geometry(pt.shape, origin=(-16., -15., -15.))
        reflected = self.geometry(pt.shape, direction=(-1., 0., 0., 0., 1., 0., 0., 0., 1.), origin=(16., -15., -15.))
        _, first = self.build(pt, ln, identity, midline_lps_x=0.)
        _, second = self.build(pt, ln, reflected, midline_lps_x=0.)
        values_a = dict(zip(PHYSICAL_GEOMETRY_FEATURES, first["physical_geometry"]["values"]))
        values_b = dict(zip(PHYSICAL_GEOMETRY_FEATURES, second["physical_geometry"]["values"]))
        self.assertEqual(values_a["pt_ln_centroid_distance_mm"], 2.0)
        self.assertEqual(values_a["pt_ln_min_surface_distance_mm"], 2.0)
        self.assertEqual(values_a["pt_ln_centroid_distance_mm"], values_b["pt_ln_centroid_distance_mm"])
        self.assertEqual(values_a["ln_left_flag"], 1.)
        self.assertEqual(values_b["ln_right_flag"], 1.)
        _, unknown = self.build(pt, ln, identity)
        validity = dict(zip(PHYSICAL_GEOMETRY_FEATURES, unknown["physical_geometry"]["valid"]))
        self.assertFalse(validity["ln_left_flag"])
        self.assertFalse(validity["node_count"])
        self.assertFalse(validity["largest_node_volume_mm3"])

    def test_reference_boundary_truncation_is_separate_from_crop_retention(self):
        pt = np.zeros((31, 31, 31), dtype=bool)
        pt[1, 15, 15] = True
        _, truncated = self.build(pt, np.zeros_like(pt))
        self.assertTrue(truncated["regions"]["pt_peri"]["reference_boundary_truncated"])
        self.assertIsNone(truncated["regions"]["pt_peri"]["crop_completeness"])
        pt[:] = False
        pt[15, 15, 15] = True
        _, complete = self.build(pt, np.zeros_like(pt))
        self.assertFalse(complete["regions"]["pt_peri"]["reference_boundary_truncated"])

    def test_manifest_roundtrip_and_tamper_rejection(self):
        pt = np.zeros((31, 31, 31), dtype=bool)
        pt[15, 15, 15] = True
        masks, manifest = self.build(pt, np.zeros_like(pt))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "regions.json"
            write_region_manifest(path, manifest)
            self.assertEqual(load_region_manifest(path, masks=masks), manifest)
        altered = copy.deepcopy(manifest)
        altered["regions"]["pt_intra"]["scan_id"] = "another-scan"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            validate_region_manifest(altered)
        altered_masks = {name: mask.copy() for name, mask in masks.items()}
        altered_masks["pt_intra"][14, 15, 15] = True
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_region_manifest(manifest, masks=altered_masks)

    def test_image_representation_has_its_own_support_identity(self):
        pt = np.zeros((31, 31, 31), dtype=bool)
        pt[15, 15, 15] = True
        masks, manifest = self.build(pt, np.zeros_like(pt))
        image_geometry = self.geometry((15, 15, 15), spacing=(2., 2., 2.))
        image_masks = {}
        for name, mask in masks.items():
            smaller = mask[1::2, 1::2, 1::2]
            image_masks[name] = smaller
            record = manifest["regions"][name]
            record["image_geometry"] = image_geometry.to_dict()
            record["image_mask_sha256"] = mask_checksum(smaller)
            record["image_region_id"] = region_identity(record, mask_sha256=record["image_mask_sha256"], geometry=image_geometry)
            record["feature_validity"]["image"] = bool(smaller.any())
            self.assertNotEqual(record["region_id"], record["image_region_id"])
        validate_region_manifest(manifest, masks=image_masks, geometry=image_geometry)
        manifest["regions"]["pt_intra"]["image_region_id"] = manifest["regions"]["pt_intra"]["region_id"]
        with self.assertRaisesRegex(ValueError, "image support identity"):
            validate_region_manifest(manifest)

    def test_binary_checksum_is_canonical_and_geometry_specific(self):
        array = np.ones((2, 3, 4), dtype=bool)
        self.assertEqual(mask_checksum(array), mask_checksum(array.astype(np.float32)))
        self.assertNotEqual(mask_checksum(array), mask_checksum(array.reshape(3, 2, 4)))
        with self.assertRaisesRegex(ValueError, "binary"):
            mask_checksum(array.astype(float) * .5)
        self.assertFalse(physical_shell(array, spacing_dhw=(1, 1, 1), width_mm=0).any())

    def test_physical_features_bind_source_regions_and_calculation(self):
        pt = np.zeros((31, 31, 31), dtype=bool)
        pt[15, 15, 15] = True
        _, manifest = self.build(pt, np.zeros_like(pt))
        shifted = np.roll(pt, 1, axis=0)
        _, other = self.build(shifted, np.zeros_like(pt))
        altered = copy.deepcopy(manifest)
        altered["physical_geometry"] = other["physical_geometry"]
        with self.assertRaisesRegex(ValueError, "source region identities"):
            validate_region_manifest(altered)
        altered = copy.deepcopy(manifest)
        altered["physical_geometry"]["values"][0] = 2.0
        with self.assertRaisesRegex(ValueError, "provenance checksum"):
            validate_region_manifest(altered)
        physical = altered["physical_geometry"]
        physical["provenance_sha256"] = physical_geometry_checksum(physical)
        with self.assertRaisesRegex(ValueError, "authoritative reference compartment"):
            validate_region_manifest(altered)

    def test_physical_features_require_binary_validity_and_zero_unknowns(self):
        pt = np.zeros((31, 31, 31), dtype=bool)
        pt[15, 15, 15] = True
        _, manifest = self.build(pt, None)
        altered = copy.deepcopy(manifest)
        altered["physical_geometry"]["valid"][0] = 2
        with self.assertRaisesRegex(ValueError, "0/1"):
            validate_region_manifest(altered)
        altered = copy.deepcopy(manifest)
        altered["physical_geometry"]["values"][1] = 1.0
        with self.assertRaisesRegex(ValueError, "zero filled"):
            validate_region_manifest(altered)

    def test_reference_voxel_count_must_match_loaded_support(self):
        pt = np.zeros((31, 31, 31), dtype=bool)
        pt[15, 15, 15] = True
        masks, manifest = self.build(pt, np.zeros_like(pt))
        record = manifest["regions"]["pt_intra"]
        record["voxel_count"] = 2
        record["physical_volume_mm3"] = 2.0
        physical = manifest["physical_geometry"]
        physical["values"][0] = 2.0
        physical["provenance_sha256"] = physical_geometry_checksum(physical)
        with self.assertRaisesRegex(ValueError, "reference voxel count"):
            validate_region_manifest(manifest, masks=masks)

    def test_absent_region_cannot_gain_model_support(self):
        pt = np.zeros((31, 31, 31), dtype=bool)
        pt[15, 15, 15] = True
        masks, manifest = self.build(pt, np.zeros_like(pt))
        record = manifest["regions"]["ln_intra"]
        record["image_geometry"] = manifest["reference_geometry"]
        record["image_mask_sha256"] = mask_checksum(pt)
        record["image_region_id"] = region_identity(record, mask_sha256=record["image_mask_sha256"],
                                                      geometry=record["image_geometry"])
        masks["ln_intra"] = pt
        with self.assertRaisesRegex(ValueError, "absent/unavailable/invalid"):
            validate_region_manifest(manifest, masks=masks)


if __name__ == "__main__":
    unittest.main()
