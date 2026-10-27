"""Regression checks for small and absent ROI feature pooling (CPU only)."""

import unittest

import torch

from trifusesurv2.models.contour_habitat_survival import ContourAwareHabitatSurvivalModel
from trifusesurv2.models.swinunetr_shared_roi_token_backbone import (
    AttnPool3D,
    ContourAwareROITokenBackbone,
    interp_mask,
)


class ROIPoolingTests(unittest.TestCase):
    @staticmethod
    def _uniform_pool(mode, features):
        pool = AttnPool3D(mask_mode=mode)
        pool(features)  # Materialize the lazy convolution.
        with torch.no_grad():
            pool.attn.weight.zero_()
        return pool

    def test_small_roi_survives_documented_crop_downsampling(self):
        mask = torch.zeros(2, 1, 96, 128, 128)
        mask[0, :, 24:28, 24:28, 24:28] = 1.0
        for mode in ("nearest", "trilinear"):
            with self.subTest(mode=mode):
                reduced = interp_mask(mask, (3, 4, 4), mode)
                expected = torch.zeros_like(reduced)
                expected[0, 0, 0, 0, 0] = 1.0
                torch.testing.assert_close(reduced, expected)

    def test_roi_pooling_excludes_features_outside_reduced_support(self):
        native = torch.zeros(1, 1, 96, 128, 128)
        native[:, :, 24:28, 24:28, 24:28] = 1.0
        mask = interp_mask(native, (3, 4, 4), "trilinear")
        for mode in ("masked", "weighted"):
            with self.subTest(mode=mode):
                features = torch.arange(1.0, 49.0).reshape(1, 1, 3, 4, 4).requires_grad_()
                pool = self._uniform_pool(mode, features)
                result = pool(features, mask)
                torch.testing.assert_close(result, torch.tensor([[1.0]]))
                perturbed = features.detach() + 1000.0 * (1.0 - mask)
                torch.testing.assert_close(pool(perturbed, mask), result)
                result.sum().backward()
                self.assertTrue(bool(torch.isfinite(features.grad).all()))
                self.assertEqual(float(features.grad[mask == 0].abs().sum()), 0.0)

    def test_empty_support_is_zero_with_finite_backward(self):
        for mode in ("masked", "weighted"):
            with self.subTest(mode=mode):
                features = torch.randn(2, 3, 2, 2, 2, requires_grad=True)
                pool = self._uniform_pool(mode, features)
                # Exercise an empty row alongside a present row in one batch.
                mask = torch.zeros(2, 1, 2, 2, 2)
                mask[1, 0, 0, 0, 0] = 1.0
                result = pool(features, mask)
                torch.testing.assert_close(result[0], torch.zeros(3))
                torch.testing.assert_close(result[1], features[1, :, 0, 0, 0])
                result.sum().backward()
                self.assertTrue(bool(torch.isfinite(features.grad).all()))
                self.assertTrue(bool(torch.isfinite(pool.attn.weight.grad).all()))
                self.assertEqual(float(features.grad[0].abs().sum()), 0.0)

    def test_weighted_pooling_respects_nonzero_weights(self):
        features = torch.tensor([[[[[2.0, 6.0, 1000.0]]]]])
        mask = torch.tensor([[[[[0.25, 0.75, 0.0]]]]])
        pool = self._uniform_pool("weighted", features)
        torch.testing.assert_close(pool(features, mask), torch.tensor([[5.0]]))

    def test_unmasked_pooling_still_uses_all_features(self):
        features = torch.arange(1.0, 9.0).reshape(1, 1, 2, 2, 2)
        pool = self._uniform_pool("masked", features)
        torch.testing.assert_close(pool(features), torch.tensor([[4.5]]))

    def test_small_real_model_forward_and_backward(self):
        torch.manual_seed(17)
        backbone = ContourAwareROITokenBackbone(
            img_size=(32, 32, 32), feature_size=12, depths=(1, 1, 1, 1),
            num_heads=(3, 6, 12, 24), token_dim=16, token_mlp_dropout=0.0,
            token_dropout=0.0, drop_rate=0.0, attn_drop_rate=0.0,
            dropout_path_rate=0.0, min_roi_voxels_deep=0,
            mask_interp="trilinear", body_ct_thr=0.02, body_close_r=0,
            pt_shell_thickness_mm=0.0, pt_shell_radius=1, ln_shell_radius=1,
        )
        model = ContourAwareHabitatSurvivalModel(
            backbone=backbone, num_time_bins=4, time_bin_width_days=120,
            radiomics_token_dim=4, clinical_token_dim=4,
            node_token_dim=9, topology_dim=9,
            model_dim=16, num_heads=4, transformer_layers=1, dropout=0.0,
        )
        image = torch.rand(2, 1, 32, 32, 32)
        pt = torch.zeros_like(image)
        pt[:, :, 12:16, 12:16, 12:16] = 1.0
        ln = torch.zeros_like(image)
        ln[0, :, 20:24, 20:24, 20:24] = 1.0
        logits, aux = model(
            image, torch.randn(2, 3, 4), torch.randn(2, 4, 4),
            mask_pt=pt, mask_ln=ln, voxel_spacing_dhw=torch.ones(2, 3),
            teacher_force_alpha=1.0,
            clinical_presence=torch.tensor([[1, 1, 1], [0, 0, 0]]),
            radiomics_presence=torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]]),
            node_tokens=torch.randn(2, 2, 9),
            node_presence=torch.tensor([[1, 0], [0, 0]]),
            topology_token=torch.randn(2, 9),
            topology_presence=torch.tensor([1, 0]), return_aux=True,
        )
        self.assertEqual(set(logits), {"OS", "DSS", "DFS"})
        for output in logits.values():
            self.assertEqual(tuple(output.shape), (2, 4))
            self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertFalse(bool(aux["habitat_presence"][1, 3]))
        self.assertFalse(bool(aux["habitat_presence"][1, 4]))
        sum(output.square().mean() for output in logits.values()).backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in gradients))


if __name__ == "__main__":
    unittest.main()
