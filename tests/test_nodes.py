"""Unit tests for node topology utilities."""

from __future__ import annotations

import json
import tempfile
import unittest

import importlib.util

try:
    _HAVE_DEPS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "SimpleITK"))
except Exception:
    _HAVE_DEPS = False

if _HAVE_DEPS:
    import numpy as np
    import SimpleITK as sitk
    from trifusesurv2.preprocessing.nodes import (
        extract_node_instances,
        map_roi_sources_by_overlap,
        summarize_node_topology,
        topology_summary_feature_presence,
        topology_summary_to_vector,
        serialize_node_topology,
        write_node_topology_json,
        read_node_topology_json,
    )


@unittest.skipUnless(_HAVE_DEPS, "numpy/SimpleITK are not available in this runtime")
class NodeTopologyTest(unittest.TestCase):
    def test_extract_and_summarize(self):
        arr = np.zeros((8, 8, 8), dtype=np.uint8)
        arr[2:4, 2:4, 2:4] = 1
        arr[5:7, 5:7, 5:7] = 1
        mask = sitk.GetImageFromArray(arr)
        mask.SetSpacing((1.0, 1.0, 1.0))

        pt = np.zeros((8, 8, 8), dtype=np.uint8)
        pt[1:3, 1:3, 1:3] = 1
        pt_mask = sitk.GetImageFromArray(pt)
        pt_mask.SetSpacing((1.0, 1.0, 1.0))

        nodes, label_map = extract_node_instances(mask, min_voxels=1, midline_x_mm=3.5)
        self.assertEqual(len(nodes), 2)
        self.assertIsInstance(label_map, sitk.Image)
        label_arr = sitk.GetArrayFromImage(label_map)
        self.assertGreater(label_arr.max(), 0)

        summary = summarize_node_topology(nodes, pt_mask_img=pt_mask)
        self.assertEqual(summary.node_count, 2.0)
        self.assertGreater(summary.node_total_volume_mm3, 0.0)
        self.assertEqual(summary.node_laterality_known_flag, 1.0)
        self.assertEqual(summary.pt_ln_distance_known_flag, 1.0)

    def test_missing_pt_geometry_is_not_conflated_with_zero_distance(self):
        arr = np.zeros((8, 8, 8), dtype=np.uint8)
        arr[2:4, 2:4, 2:4] = 1
        mask = sitk.GetImageFromArray(arr)
        mask.SetSpacing((1.0, 1.0, 1.0))

        nodes, _ = extract_node_instances(mask, min_voxels=1, midline_x_mm=None)
        summary = summarize_node_topology(nodes, pt_mask_img=None)

        self.assertEqual(summary.pt_ln_distance_known_flag, 0.0)
        self.assertTrue(np.isnan(summary.pt_ln_min_distance_mm))
        self.assertTrue(np.isnan(summary.pt_ln_mean_distance_mm))

        vec = topology_summary_to_vector(summary)
        pres = topology_summary_feature_presence(summary)
        self.assertEqual(vec.shape[0], pres.shape[0])
        self.assertEqual(float(vec[-3]), 0.0)
        self.assertEqual(float(pres[-2]), 0.0)
        self.assertEqual(float(pres[-1]), 0.0)

    def test_roi_source_name_via_overlap(self):
        arr = np.zeros((8, 8, 8), dtype=np.uint8)
        arr[2:4, 2:4, 2:4] = 1  # component A
        arr[5:7, 5:7, 5:7] = 1  # component B
        mask = sitk.GetImageFromArray(arr)
        mask.SetSpacing((1.0, 1.0, 1.0))

        roi_a = np.zeros((8, 8, 8), dtype=np.uint8)
        roi_a[2:4, 2:4, 2:4] = 1
        roi_a_img = sitk.GetImageFromArray(roi_a)
        roi_a_img.SetSpacing((1.0, 1.0, 1.0))

        roi_b = np.zeros((8, 8, 8), dtype=np.uint8)
        roi_b[5:7, 5:7, 5:7] = 1
        roi_b_img = sitk.GetImageFromArray(roi_b)
        roi_b_img.SetSpacing((1.0, 1.0, 1.0))

        nodes, label_map = extract_node_instances(mask, min_voxels=1, midline_x_mm=3.5)
        self.assertEqual(len(nodes), 2)
        # Before annotation, all names are empty
        self.assertTrue(all(n.roi_source_name == "" for n in nodes))

        annotated = map_roi_sources_by_overlap(
            nodes, label_map,
            roi_masks={"GTV_LN_L2": roi_a_img, "GTV_LN_R3": roi_b_img},
        )
        self.assertEqual(len(annotated), 2)
        names = {n.roi_source_name for n in annotated}
        self.assertEqual(names, {"GTV_LN_L2", "GTV_LN_R3"})

    def test_serialize_and_write_json(self):
        arr = np.zeros((8, 8, 8), dtype=np.uint8)
        arr[2:4, 2:4, 2:4] = 1
        mask = sitk.GetImageFromArray(arr)
        mask.SetSpacing((1.0, 1.0, 1.0))

        nodes, _ = extract_node_instances(mask, min_voxels=1)
        summary = summarize_node_topology(nodes)

        data = serialize_node_topology(
            nodes, summary,
            image_metadata={"spacing": [1.0, 1.0, 1.0], "origin": [0.0, 0.0, 0.0]},
            crop_transform={"bbox": [0, 0, 0, 8, 8, 8], "margin_mm": 20.0},
        )
        self.assertIn("node_instances", data)
        self.assertIn("topology_summary", data)
        self.assertIn("image_metadata", data)
        self.assertIn("crop_transform", data)

        json_str = json.dumps(data)
        self.assertNotIn("NaN", json_str)
        self.assertNotIn("Infinity", json_str)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        write_node_topology_json(path, nodes, summary)
        loaded = read_node_topology_json(path)
        self.assertEqual(len(loaded["node_instances"]), len(nodes))

    def test_nan_serialized_as_null(self):
        arr = np.zeros((8, 8, 8), dtype=np.uint8)
        arr[2:4, 2:4, 2:4] = 1
        mask = sitk.GetImageFromArray(arr)
        mask.SetSpacing((1.0, 1.0, 1.0))

        nodes, _ = extract_node_instances(mask, min_voxels=1)
        summary = summarize_node_topology(nodes, pt_mask_img=None)
        data = serialize_node_topology(nodes, summary)
        self.assertIsNone(data["topology_summary"]["pt_ln_min_distance_mm"])
        self.assertIsNone(data["topology_summary"]["pt_ln_mean_distance_mm"])


if __name__ == "__main__":
    unittest.main()
