"""Common-lattice engineering acceptance; synthetic spacing is not a cohort policy."""
import copy
import itertools
import unittest

import numpy as np
import SimpleITK as sitk

from trifusesurv2.preprocessing.compartments import ReferenceGeometry, build_compartments, mask_checksum, region_identity
from trifusesurv2.preprocessing.sampling import CommonGridPolicy, full_grid_geometry, sample_common_grid, validate_sampling_metadata


def image_from_array(array, geometry):
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(geometry.spacing_xyz)
    image.SetOrigin(geometry.origin_xyz)
    image.SetDirection(geometry.direction)
    return image


class CommonGridTests(unittest.TestCase):
    def policy(self, *, spacing=(1.5, 1.25, 2.), origin=(.25, -.125, .5)):
        return CommonGridPolicy.from_dict({"schema_version": 1, "spacing_xyz": list(spacing),
            "origin_xyz": list(origin), "direction": [1., 0., 0., 0., 1., 0., 0., 0., 1.],
            "units": "mm", "coordinate_system": "LPS"})

    def case(self, *, distant=False):
        geometry = ReferenceGeometry((64, 64, 64), (1., 1., 1.))
        z, y, x = np.indices(geometry.shape_dhw)
        pt = (z - 24) ** 2 + (y - 24) ** 2 + (x - 24) ** 2 <= 9
        ln = (z - 24) ** 2 + (y - 24) ** 2 + (x - 34) ** 2 <= 4
        if distant:
            ln |= (z - 50) ** 2 + (y - 48) ** 2 + (x - 52) ** 2 <= 4
        masks, manifest = build_compartments(pt, ln, np.ones_like(pt), geometry,
            patient_id="synthetic", scan_id="synthetic-scan", contour_source="synthetic",
            contour_version="v1", valid_tissue_source="explicit synthetic FOV")
        ct = (.5 + .2 * np.sin(x * .7) + .15 * np.cos(y * .9)).astype(np.float32)
        images = {"image": image_from_array(ct, geometry),
                  "radiomics_image": image_from_array(ct * 1400 - 1000, geometry)}
        return images, masks, geometry, manifest

    def sample(self, case, *, shape=(48, 64, 64), **kwargs):
        images, masks, geometry, manifest = case
        return sample_common_grid(images, masks, geometry, self.policy(), shape,
            reference_region_ids={name: record["region_id"] for name, record in manifest["regions"].items()}, **kwargs)

    def regional_measurements(self, result, name, image="image"):
        coordinates = np.argwhere(result.masks[name])
        global_indices_xyz = (coordinates + np.asarray(result.metadata["crop_start_dhw"]))[:, ::-1]
        global_indices_xyz += np.asarray(result.metadata["full_grid_lattice_start_xyz"])
        order = np.lexsort(global_indices_xyz.T[::-1])
        values = sitk.GetArrayFromImage(result.images[image])[result.masks[name]]
        return global_indices_xyz[order], values[order], result.geometry.points_lps(coordinates)[order]

    def assert_same_measurements(self, first, second, name):
        for image in ("image", "radiomics_image"):
            first_indices, first_values, first_points = self.regional_measurements(first, name, image)
            second_indices, second_values, second_points = self.regional_measurements(second, name, image)
            np.testing.assert_array_equal(first_indices, second_indices)
            np.testing.assert_array_equal(first_values, second_values)
            np.testing.assert_allclose(first_points, second_points, rtol=0, atol=1e-12)

    def test_distant_node_preserves_exact_primary_support_and_values(self):
        near, distant = self.case(), self.case(distant=True)
        for name in ("pt_intra", "pt_peri"):
            np.testing.assert_array_equal(near[1][name], distant[1][name])
        first, second = self.sample(near), self.sample(distant)
        self.assertEqual(first.full_geometry, second.full_geometry)
        self.assertEqual(first.metadata["policy_id"], second.metadata["policy_id"])
        self.assertNotEqual(first.metadata["crop_start_dhw"], second.metadata["crop_start_dhw"])
        for name in ("pt_intra", "pt_peri"):
            self.assert_same_measurements(first, second, name)
            self.assertEqual(first.metadata["regions"][name]["image_volume_mm3"], second.metadata["regions"][name]["image_volume_mm3"])

    def test_integer_crop_offsets_and_larger_buckets_do_not_resample(self):
        case = self.case()
        first = self.sample(case)
        start = np.asarray(first.metadata["crop_start_dhw"]) + [1, -1, 2]
        shifted = self.sample(case, crop_start_dhw=start)
        larger = self.sample(case, shape=(52, 68, 68))
        for result in (shifted, larger):
            self.assertEqual(result.geometry.spacing_xyz, first.geometry.spacing_xyz)
            for name in case[1]:
                self.assert_same_measurements(first, result, name)
            self.assertTrue(any(result.metadata["pad_before_dhw"]) or any(result.metadata["pad_after_dhw"]))
            for mask in result.masks.values():
                self.assertFalse(np.any(mask & ~result.source_fov_mask))

    def test_canvas_overflow_and_fractional_offsets_fail_without_resize(self):
        case = self.case()
        with self.assertRaisesRegex(ValueError, "canvas overflow"):
            self.sample(case, shape=(8, 8, 8))
        with self.assertRaisesRegex(ValueError, "integer lattice offsets"):
            self.sample(case, crop_start_dhw=(0, 0, .25))
        with self.assertRaisesRegex(ValueError, "lose required"):
            self.sample(case, crop_start_dhw=(100, 100, 100))

    def test_small_support_disappearance_is_distinct_from_crop_loss(self):
        geometry = ReferenceGeometry((33, 33, 33), (1., 1., 1.))
        pt = np.zeros(geometry.shape_dhw, dtype=bool)
        pt[16, 16, 16] = True
        masks, manifest = build_compartments(pt, np.zeros_like(pt), np.ones_like(pt), geometry,
            patient_id="synthetic", scan_id="small", contour_source="synthetic", contour_version="v1",
            valid_tissue_source="explicit synthetic FOV")
        result = sample_common_grid({"image": image_from_array(pt.astype(np.float32), geometry)}, masks, geometry,
            self.policy(spacing=(8., 8., 8.), origin=(4., 4., 4.)), (8, 8, 8),
            reference_region_ids={name: record["region_id"] for name, record in manifest["regions"].items()})
        record = result.metadata["regions"]["pt_intra"]
        self.assertTrue(record["disappeared_on_full_grid"])
        self.assertEqual(record["image_voxels"], 0)
        self.assertEqual(record["retained_reference_voxels"], 1)
        self.assertEqual(record["retained_reference_fraction"], 1.)
        self.assertFalse(result.metadata["regions"]["ln_intra"]["disappeared_on_full_grid"])

    def test_rotated_reflected_anisotropic_sources_align_in_world_coordinates(self):
        angle = np.deg2rad(30)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
        for reflection in (1., -1.):
            with self.subTest(reflection=reflection):
                direction = rotation @ np.diag([reflection, 1., 1.])
                geometry = ReferenceGeometry((31, 33, 35), (1.25, 1.5, 2.), (-20., -30., -30.), tuple(direction.ravel()))
                coordinates = np.indices(geometry.shape_dhw).reshape(3, -1).T
                points = geometry.points_lps(coordinates)
                values = (.4 + points @ np.array([.001, .002, .003])).reshape(geometry.shape_dhw).astype(np.float32)
                mask = np.zeros(geometry.shape_dhw, dtype=bool)
                mask[13:18, 14:19, 15:20] = True
                result = sample_common_grid({"image": image_from_array(values, geometry)}, {"pt_intra": mask},
                                            geometry, self.policy(), (32, 32, 32))
                output_points = result.geometry.points_lps(np.argwhere(result.masks["pt_intra"]))
                sampled = sitk.GetArrayFromImage(result.images["image"])[result.masks["pt_intra"]]
                np.testing.assert_allclose(sampled, .4 + output_points @ np.array([.001, .002, .003]), rtol=0, atol=1e-7)
                source_indices = ((output_points - geometry.origin_xyz) @ direction) / geometry.spacing_xyz
                source_indices = np.floor(source_indices + .5).astype(int)[:, ::-1]
                self.assertTrue(mask[tuple(source_indices.T)].all())
                corners = np.asarray(list(itertools.product(*[(-.5, size - .5) for size in geometry.shape_dhw])))
                full = result.full_geometry
                projected = ((geometry.points_lps(corners) - full.origin_xyz) @ np.asarray(full.direction).reshape(3, 3)) / full.spacing_xyz
                self.assertTrue(np.all(projected >= -.5 - 1e-8))
                self.assertTrue(np.all(projected <= np.asarray(full.shape_dhw[::-1]) - .5 + 1e-8))

    def test_policy_is_explicit_canonical_and_independent_of_source(self):
        policy = self.policy()
        config = policy.to_dict()
        self.assertEqual(CommonGridPolicy.from_dict(dict(reversed(list(config.items())))).policy_id, policy.policy_id)
        self.assertNotEqual(self.policy(origin=(0., 0., 0.)).policy_id, policy.policy_id)
        for field in ("spacing_xyz", "direction", "origin_xyz"):
            incomplete = dict(config)
            incomplete.pop(field)
            with self.assertRaisesRegex(ValueError, "requires exactly"):
                CommonGridPolicy.from_dict(incomplete)
        invalid = dict(config, direction=[1., .2, 0., 0., 1., 0., 0., 0., 1.])
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            CommonGridPolicy.from_dict(invalid)
        invalid = dict(config, spacing_xyz=[0., 1., 1.])
        with self.assertRaisesRegex(ValueError, "positive"):
            CommonGridPolicy.from_dict(invalid)

    def test_persisted_policy_phase_and_support_accounting_reject_tampering(self):
        case = self.case()
        result = self.sample(case)
        manifest = copy.deepcopy(case[3])
        manifest["sampling"] = result.metadata
        manifest["image"] = {"geometry": result.geometry.to_dict()}
        for name, record in manifest["regions"].items():
            measured = result.metadata["regions"][name]
            record["image_geometry"] = result.geometry.to_dict()
            record["image_mask_sha256"] = mask_checksum(result.masks[name])
            record["image_region_id"] = region_identity(record, mask_sha256=record["image_mask_sha256"], geometry=result.geometry)
            record["crop_completeness"] = {"retained_reference_voxels": measured["retained_reference_voxels"],
                                           "image_volume_mm3": measured["image_volume_mm3"]}
        validate_sampling_metadata(manifest)
        changed = copy.deepcopy(manifest)
        changed["sampling"]["policy"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            validate_sampling_metadata(changed)
        changed = copy.deepcopy(manifest)
        changed["sampling"]["configuration"]["spacing_xyz"][0] = 9.
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_sampling_metadata(changed)
        changed = copy.deepcopy(manifest)
        changed["image"]["geometry"]["origin_xyz"][0] += .25
        with self.assertRaisesRegex(ValueError, "origin_xyz"):
            validate_sampling_metadata(changed)
        changed = copy.deepcopy(manifest)
        changed["sampling"]["regions"]["pt_intra"]["reference_region_id"] = None
        with self.assertRaisesRegex(ValueError, "reference region identity"):
            validate_sampling_metadata(changed)
        mutations = (
            ("source_fov", "image_voxels", -1),
            ("source_fov", "image_voxels", 10 ** 12),
            ("source_fov", "padded_support_is_zero", False),
            ("source_fov", "image_mask_sha256", "not-a-checksum"),
            ("intensity_default_values", "image", float("nan")),
            (None, "context_margin_mm", -1),
            (None, "source_center_retention_rule", "different rule"),
        )
        for section, field, value in mutations:
            with self.subTest(section=section, field=field, value=value):
                changed = copy.deepcopy(manifest)
                target = changed["sampling"] if section is None else changed["sampling"][section]
                target[field] = value
                with self.assertRaises(ValueError):
                    validate_sampling_metadata(changed)


if __name__ == "__main__":
    unittest.main()
