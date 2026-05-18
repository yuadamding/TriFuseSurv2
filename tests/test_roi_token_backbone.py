"""Focused tests for ROI-token spatial support behavior."""

from __future__ import annotations

import importlib.util
import unittest

try:
    _HAVE_DEPS = (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("monai") is not None
    )
except Exception:
    _HAVE_DEPS = False

if _HAVE_DEPS:
    import torch
    import torch.nn as nn

    from trifusesurv2.models.swinunetr_shared_roi_token_backbone import (
        AttnPool3D,
        ContourAwareROITokenBackbone,
        ct_stats_in_mask,
    )


@unittest.skipUnless(_HAVE_DEPS, "torch/monai are not available in this runtime")
class ROITokenBackboneTest(unittest.TestCase):
    def test_attention_pool_masked_mode_excludes_outside_voxels(self):
        feat = torch.arange(4, dtype=torch.float32).view(1, 1, 1, 1, 4)
        mask = torch.tensor([[[[[0.0, 0.0, 1.0, 1.0]]]]], dtype=torch.float32)
        pool = AttnPool3D(mask_mode="masked")
        pool.attn = nn.Conv3d(1, 1, kernel_size=1, bias=False)
        nn.init.zeros_(pool.attn.weight)

        masked = pool(feat, mask)
        unmasked = pool(feat, None)
        empty_fallback = pool(feat, torch.zeros_like(mask))

        self.assertAlmostEqual(float(masked.item()), 2.5, places=5)
        self.assertAlmostEqual(float(unmasked.item()), 1.5, places=5)
        self.assertAlmostEqual(float(empty_fallback.item()), 1.5, places=5)

    def test_roi_support_mask_uses_hard_forward_support_with_fallback(self):
        holder = type("BackboneHolder", (), {})()
        holder.roi_support_threshold = 0.5
        holder.roi_support_fallback_threshold = 0.05
        holder.roi_support_fallback_relmax = 0.5
        holder.raw_mask_threshold = 0.5

        soft = torch.tensor([[[[[0.01, 0.20, 0.40]]]]], dtype=torch.float32, requires_grad=True)
        support = ContourAwareROITokenBackbone._roi_support_mask(holder, soft)

        self.assertEqual(support.detach().flatten().tolist(), [0.0, 1.0, 1.0])
        support.sum().backward()
        self.assertTrue(torch.isfinite(soft.grad).all().item())

    def test_native_mask_is_always_included_when_guidance_is_active(self):
        support = torch.tensor([[[[[1.0, 0.0, 0.0]]]]], dtype=torch.float32)
        native = torch.tensor([[[[[0.0, 1.0, 0.0]]]]], dtype=torch.float32)

        guided = ContourAwareROITokenBackbone._apply_native_support_floor(support, native, alpha=0.25)
        unguided = ContourAwareROITokenBackbone._apply_native_support_floor(support, native, alpha=0.0)

        self.assertEqual(guided.flatten().tolist(), [1.0, 1.0, 0.0])
        self.assertEqual(unguided.flatten().tolist(), [1.0, 0.0, 0.0])

    def test_ct_stats_can_ablate_explicit_mask_volume(self):
        ct = torch.tensor([[[[[1.0, 3.0, 5.0]]]]], dtype=torch.float32)
        mask = torch.tensor([[[[[1.0, 1.0, 0.0]]]]], dtype=torch.float32)

        with_volume = ct_stats_in_mask(ct, mask, include_volume=True)
        without_volume = ct_stats_in_mask(ct, mask, include_volume=False)

        self.assertEqual(tuple(with_volume.shape), (1, 3))
        self.assertEqual(tuple(without_volume.shape), (1, 2))
        self.assertAlmostEqual(float(with_volume[0, 0].item()), 2.0, places=5)
        self.assertAlmostEqual(float(without_volume[0, 0].item()), 2.0, places=5)
        self.assertAlmostEqual(float(with_volume[0, 2].item()), 2.0 / 3.0, places=5)


if __name__ == "__main__":
    unittest.main()
