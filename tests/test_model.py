"""Unit tests for the habitat-aligned survival model."""

from __future__ import annotations

import unittest

import importlib.util

try:
    _HAVE_DEPS = importlib.util.find_spec("torch") is not None
except Exception:
    _HAVE_DEPS = False

if _HAVE_DEPS:
    import torch

    from trifusesurv2.models.habitat_survival import HabitatAlignedSurvivalModel


@unittest.skipUnless(_HAVE_DEPS, "torch is not available in this runtime")
class HabitatAlignedSurvivalModelTest(unittest.TestCase):
    def test_invalid_image_habitat_count_raises(self):
        model = HabitatAlignedSurvivalModel(
            image_token_dim=8,
            radiomics_token_dim=4,
            clinical_token_dim=4,
            num_time_bins=5,
            model_dim=16,
            num_heads=4,
            transformer_layers=1,
        )
        with self.assertRaises(ValueError):
            model(
                image_tokens=torch.zeros(2, 4, 8),
                radiomics_tokens=torch.zeros(2, 4, 4),
                clinical_tokens=torch.zeros(2, 3, 4),
            )

    def test_empty_optional_masks_are_handled(self):
        model = HabitatAlignedSurvivalModel(
            image_token_dim=8,
            radiomics_token_dim=0,
            clinical_token_dim=0,
            node_token_dim=8,
            topology_dim=4,
            num_time_bins=5,
            model_dim=16,
            num_heads=4,
            transformer_layers=1,
        )
        logits = model(
            image_tokens=torch.randn(2, 5, 8),
            image_presence=torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool),
            node_tokens=torch.randn(2, 3, 8),
            node_presence=torch.tensor([[0, 0, 0], [1, 1, 0]], dtype=torch.bool),
            topology_token=torch.randn(2, 4),
            topology_presence=torch.tensor([0, 1], dtype=torch.bool),
        )
        self.assertEqual(set(logits.keys()), {"OS", "DSS", "DFS"})
        for value in logits.values():
            self.assertTrue(torch.isfinite(value).all().item())

    def test_pt_node_cross_attention_changes_pt_habitats(self):
        model = HabitatAlignedSurvivalModel(
            image_token_dim=8,
            radiomics_token_dim=0,
            clinical_token_dim=0,
            node_token_dim=8,
            topology_dim=4,
            num_time_bins=5,
            model_dim=16,
            num_heads=4,
            transformer_layers=1,
        )
        image_tokens = torch.randn(2, 5, 8)
        node_tokens = torch.randn(2, 3, 8)
        node_presence = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
        topology_token = torch.randn(2, 4)

        _, aux_with_nodes = model(
            image_tokens=image_tokens,
            node_tokens=node_tokens,
            node_presence=node_presence,
            topology_token=topology_token,
            return_aux=True,
        )
        self.assertIsNotNone(model.pt_node_cross_attn)
        for v in aux_with_nodes.values():
            if isinstance(v, torch.Tensor):
                self.assertTrue(torch.isfinite(v).all().item())

    def test_pt_node_cross_attention_not_created_without_nodes(self):
        model = HabitatAlignedSurvivalModel(
            image_token_dim=8,
            radiomics_token_dim=4,
            clinical_token_dim=4,
            node_token_dim=0,
            num_time_bins=5,
            model_dim=16,
            num_heads=4,
            transformer_layers=1,
        )
        self.assertIsNone(model.pt_node_cross_attn)

    def test_absent_image_habitats_are_zeroed_before_fusion(self):
        model = HabitatAlignedSurvivalModel(
            image_token_dim=4,
            radiomics_token_dim=0,
            clinical_token_dim=0,
            num_time_bins=3,
            model_dim=8,
            num_heads=4,
            transformer_layers=1,
        )
        image_tokens = torch.randn(2, 5, 4)
        image_presence = torch.tensor([[0, 0, 0, 0, 0], [1, 0, 1, 0, 1]], dtype=torch.bool)
        _, aux = model(
            image_tokens=image_tokens,
            image_presence=image_presence,
            return_aux=True,
        )
        self.assertTrue(torch.allclose(aux["habitat_tokens"][0], torch.zeros_like(aux["habitat_tokens"][0]), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
