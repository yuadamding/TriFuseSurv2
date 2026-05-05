"""Unit tests for the ContourAwareHabitatSurvivalModel wrapper."""

from __future__ import annotations

import unittest
import importlib.util

try:
    _HAVE_DEPS = importlib.util.find_spec("torch") is not None
except Exception:
    _HAVE_DEPS = False

if _HAVE_DEPS:
    from typing import Optional

    import torch
    import torch.nn as nn

    from trifusesurv2.models.contour_habitat_survival import (
        ContourAwareHabitatSurvivalModel,
        ImageTokenBackbone,
    )

    class _MockBackbone(nn.Module):
        """Minimal backbone satisfying the ImageTokenBackbone protocol."""

        def __init__(self, out_dim: int = 32, num_tokens: int = 5):
            super().__init__()
            self._out_dim = out_dim
            self._num_tokens = num_tokens
            self.proj = nn.Linear(1, out_dim * num_tokens)

        @property
        def num_tokens(self) -> int:
            return self._num_tokens

        @property
        def out_dim(self) -> int:
            return self._out_dim

        def forward(
            self,
            x_img: torch.Tensor,
            *,
            mask_pt: Optional[torch.Tensor] = None,
            mask_ln: Optional[torch.Tensor] = None,
            teacher_force_alpha: float = 0.0,
            return_aux: bool = False,
            return_cam_features: bool = False,
        ):
            B = x_img.shape[0]
            flat = self.proj(x_img.new_ones(B, 1))
            tokens = flat.view(B, self._num_tokens, self._out_dim)
            presence = torch.ones(B, self._num_tokens, device=x_img.device)
            if return_aux:
                return tokens, presence, {"mock": True}
            return tokens, presence


@unittest.skipUnless(_HAVE_DEPS, "torch is not available in this runtime")
class ContourAwareHabitatSurvivalModelTest(unittest.TestCase):
    def _make_model(self, **kwargs):
        defaults = dict(
            backbone=_MockBackbone(out_dim=32, num_tokens=5),
            num_time_bins=5,
            time_bin_width_days=180.0,
            model_dim=16,
            num_heads=4,
            transformer_layers=1,
        )
        defaults.update(kwargs)
        return ContourAwareHabitatSurvivalModel(**defaults)

    def test_forward_image_only(self):
        model = self._make_model()
        x = torch.randn(2, 1, 8, 8, 8)
        logits = model(x)
        self.assertEqual(set(logits.keys()), {"OS", "DSS", "DFS"})
        for v in logits.values():
            self.assertEqual(v.shape, (2, 5))
            self.assertTrue(torch.isfinite(v).all().item())

    def test_forward_with_clinical_and_radiomics(self):
        model = self._make_model(clinical_token_dim=6, radiomics_token_dim=4)
        x = torch.randn(2, 1, 8, 8, 8)
        clin = torch.randn(2, 3, 6)
        rad = torch.randn(2, 4, 4)
        logits = model(x, clin, rad)
        self.assertEqual(set(logits.keys()), {"OS", "DSS", "DFS"})

    def test_forward_with_nodes(self):
        model = self._make_model(node_token_dim=8, topology_dim=9)
        x = torch.randn(2, 1, 8, 8, 8)
        logits = model(
            x,
            node_tokens=torch.randn(2, 3, 8),
            node_presence=torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32),
            topology_token=torch.randn(2, 9),
        )
        self.assertEqual(set(logits.keys()), {"OS", "DSS", "DFS"})

    def test_return_aux(self):
        model = self._make_model()
        x = torch.randn(2, 1, 8, 8, 8)
        logits, aux = model(x, return_aux=True)
        self.assertIn("latent", aux)
        self.assertIn("backbone_aux", aux)
        self.assertEqual(aux["backbone_aux"]["mock"], True)

    def test_image_habitat_dropout_keeps_global(self):
        model = self._make_model()
        model.train()
        x = torch.randn(2, 1, 8, 8, 8)
        logits, aux = model(
            x,
            return_aux=True,
            image_habitat_dropout_p=1.0,
            keep_global_image_habitat=True,
        )
        self.assertEqual(set(logits.keys()), {"OS", "DSS", "DFS"})
        presence = aux["habitat_presence"]
        self.assertTrue(presence[:, 0].all().item())
        self.assertFalse(presence[:, 1:].any().item())

    def test_optional_modality_dropout_preserves_2d_topology_shape(self):
        from trifusesurv2.multimodal_survival.train import _drop_optional_modality

        tokens, presence = _drop_optional_modality(
            torch.ones(3, 9),
            torch.ones(3),
            p=1.0,
        )
        self.assertEqual(tokens.shape, (3, 9))
        self.assertEqual(presence.shape, (3, 1))
        self.assertEqual(float(tokens.sum().item()), 0.0)

    def test_training_batch_uses_topology_key(self):
        from trifusesurv2.multimodal_survival.train import _unpack_surv_batch

        topology = torch.randn(2, 1, 9)
        presence = torch.ones(2, 1)
        payload = _unpack_surv_batch(
            {
                "x": torch.randn(2, 1, 8, 8, 8),
                "t": torch.ones(2),
                "e": torch.ones(2),
                "clinical_tokens": torch.randn(2, 3, 6),
                "radiomics_tokens": torch.randn(2, 4, 5),
                "topology_token": topology,
                "topology_presence": presence,
                "pid": ["A", "B"],
            }
        )
        self.assertIs(payload["topology_token"], topology)
        self.assertIs(payload["topology_presence"], presence)
        self.assertNotIn("topo_token", payload)

    def test_return_gate_raises(self):
        model = self._make_model()
        x = torch.randn(2, 1, 8, 8, 8)
        with self.assertRaises(NotImplementedError):
            model(x, return_gate=True)

    def test_hazards_to_risk(self):
        model = self._make_model()
        logits_tensor = torch.zeros(2, 5)
        risk = model.hazards_to_risk(logits_tensor, horizon_days=365.0)
        self.assertEqual(risk.shape, (2,))
        self.assertTrue((risk >= 0.0).all() and (risk <= 1.0).all())

    def test_mock_backbone_satisfies_protocol(self):
        bb = _MockBackbone()
        self.assertIsInstance(bb, ImageTokenBackbone)


if __name__ == "__main__":
    unittest.main()
