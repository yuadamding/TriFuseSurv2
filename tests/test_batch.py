"""Unit tests for HabitatBatch validation and device transfer."""

from __future__ import annotations

import unittest
import importlib.util

try:
    _HAVE_DEPS = importlib.util.find_spec("torch") is not None
except Exception:
    _HAVE_DEPS = False

if _HAVE_DEPS:
    import torch
    from trifusesurv2.data.batch import (
        HabitatBatch,
        SurvivalTargets,
        TokenBlock,
    )


@unittest.skipUnless(_HAVE_DEPS, "torch is not available in this runtime")
class HabitatBatchValidateTest(unittest.TestCase):
    def test_valid_batch_passes(self):
        batch = HabitatBatch(
            image=TokenBlock(tokens=torch.randn(2, 5, 8)),
            radiomics=TokenBlock(
                tokens=torch.randn(2, 4, 6),
                presence=torch.ones(2, 4),
            ),
            clinical=TokenBlock(
                tokens=torch.randn(2, 3, 10),
                presence=torch.ones(2, 3),
            ),
            survival=SurvivalTargets(
                times=torch.tensor([100.0, 200.0]),
                events=torch.tensor([1.0, 0.0]),
            ),
        )
        batch.validate(num_image_habitats=5, num_radiomics_habitats=4, num_clinical_groups=3)

    def test_wrong_image_token_dim_raises(self):
        batch = HabitatBatch(image=TokenBlock(tokens=torch.randn(2, 5)))
        with self.assertRaises(ValueError):
            batch.validate()

    def test_mismatched_batch_sizes_raises(self):
        batch = HabitatBatch(
            image=TokenBlock(tokens=torch.randn(2, 5, 8)),
            radiomics=TokenBlock(tokens=torch.randn(3, 4, 6)),
        )
        with self.assertRaises(ValueError):
            batch.validate()

    def test_wrong_presence_shape_raises(self):
        batch = HabitatBatch(
            image=TokenBlock(
                tokens=torch.randn(2, 5, 8),
                presence=torch.ones(2, 3),
            ),
        )
        with self.assertRaises(ValueError):
            batch.validate()

    def test_wrong_num_habitats_raises(self):
        batch = HabitatBatch(image=TokenBlock(tokens=torch.randn(2, 5, 8)))
        with self.assertRaises(ValueError):
            batch.validate(num_image_habitats=4)

    def test_topology_2d_accepted(self):
        batch = HabitatBatch(
            image=TokenBlock(tokens=torch.randn(2, 5, 8)),
            topology=TokenBlock(tokens=torch.randn(2, 9)),
        )
        batch.validate()

    def test_topology_3d_accepted(self):
        batch = HabitatBatch(
            image=TokenBlock(tokens=torch.randn(2, 5, 8)),
            topology=TokenBlock(tokens=torch.randn(2, 1, 9)),
        )
        batch.validate()

    def test_none_optional_modalities_pass(self):
        batch = HabitatBatch(image=TokenBlock(tokens=torch.randn(2, 5, 8)))
        batch.validate()


@unittest.skipUnless(_HAVE_DEPS, "torch is not available in this runtime")
class HabitatBatchToDeviceTest(unittest.TestCase):
    def test_to_moves_all_tensors(self):
        batch = HabitatBatch(
            image=TokenBlock(tokens=torch.randn(2, 5, 8), presence=torch.ones(2, 5)),
            clinical=TokenBlock(tokens=torch.randn(2, 3, 6)),
            survival=SurvivalTargets(times=torch.tensor([1.0, 2.0]), events=torch.tensor([1.0, 0.0])),
        )
        moved = batch.to("cpu")
        self.assertEqual(moved.image.tokens.device.type, "cpu")
        self.assertIsNotNone(moved.image.presence)
        self.assertIsNone(moved.radiomics)
        self.assertIsNotNone(moved.survival)

    def test_to_preserves_none(self):
        batch = HabitatBatch(image=TokenBlock(tokens=torch.randn(1, 5, 8)))
        moved = batch.to("cpu")
        self.assertIsNone(moved.radiomics)
        self.assertIsNone(moved.nodes)
        self.assertIsNone(moved.survival)


if __name__ == "__main__":
    unittest.main()
