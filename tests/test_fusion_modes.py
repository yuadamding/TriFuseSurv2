"""Controlled comparisons isolate routing while preserving measured information."""

import unittest

import torch

from trifusesurv2.models.contour_habitat_survival import ContourAwareHabitatSurvivalModel
from trifusesurv2.models.habitat_survival import HabitatAlignedSurvivalModel
from trifusesurv2.schema import IMAGE_HABITATS, RADIOLOGY_HABITATS, REGION_STATUS_CODES


class FusionModesTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(173)

    def model(self, mode, **kwargs):
        return HabitatAlignedSurvivalModel(
            image_token_dim=8, radiomics_token_dim=5, clinical_token_dim=4,
            num_time_bins=3, model_dim=16, num_heads=4, transformer_layers=1,
            dropout=0.0, fusion_mode=mode, **kwargs,
        ).eval()

    def inputs(self):
        return dict(
            image_tokens=torch.randn(2, 6, 8), image_presence=torch.ones(2, 6, dtype=torch.bool),
            radiomics_tokens=torch.randn(2, 4, 5), radiomics_presence=torch.ones(2, 4, dtype=torch.bool),
            clinical_tokens=torch.randn(2, 3, 4), clinical_presence=torch.ones(2, 3, dtype=torch.bool),
            region_states=torch.zeros(2, 4, dtype=torch.long),
        )

    def test_every_mode_accepts_the_same_information_and_has_finite_backward(self):
        for mode in ("F0", "F1", "F2", "F3", "F4"):
            with self.subTest(mode=mode):
                model = self.model(mode)
                inputs = self.inputs()
                for name in ("image_tokens", "radiomics_tokens", "clinical_tokens"):
                    inputs[name].requires_grad_()
                outputs, aux = model(**inputs, return_aux=True)
                self.assertEqual(set(outputs), {"OS", "DSS", "DFS"})
                for value in outputs.values():
                    self.assertEqual(tuple(value.shape), (2, 3))
                    self.assertTrue(bool(torch.isfinite(value).all()))
                sum(v.square().mean() for v in outputs.values()).backward()
                for name in ("image_tokens", "radiomics_tokens", "clinical_tokens"):
                    self.assertTrue(bool(torch.isfinite(inputs[name].grad).all()))
                    self.assertGreater(float(inputs[name].grad.abs().sum()), 0.0)
                self.assertEqual(aux["fusion_mode"], mode)
                self.assertEqual(model.parameter_count(), sum(p.numel() for p in model.parameters()))

    def test_f2_f3_have_identical_parameters_and_fixed_context_shapes(self):
        unrestricted = self.model("F2")
        aligned = self.model("F3")
        self.assertEqual(unrestricted.parameter_count(), aligned.parameter_count())
        self.assertEqual(set(unrestricted.state_dict()), set(aligned.state_dict()))
        aligned.load_state_dict(unrestricted.state_dict(), strict=True)
        inputs = self.inputs()
        _, u = unrestricted(**inputs, return_aux=True, return_attention=True)
        _, a = aligned(**inputs, return_aux=True, return_attention=True)
        for name in IMAGE_HABITATS:
            uu, aa = u["habitat_attention"][name], a["habitat_attention"][name]
            self.assertEqual(uu["weights"].shape, aa["weights"].shape)
            self.assertEqual(uu["context_names"], aa["context_names"])
            self.assertEqual(uu["context_mask"].shape, (2, 8))
            if name in RADIOLOGY_HABITATS:
                self.assertEqual(int(uu["context_mask"][0, 1:5].sum()), 4)
                self.assertEqual(int(aa["context_mask"][0, 1:5].sum()), 1)
            else:
                self.assertFalse(bool(uu["context_mask"][:, 1:5].any()))
                self.assertFalse(bool(aa["context_mask"][:, 1:5].any()))

    def test_nonmatching_radiomics_cannot_affect_f3_local_pt_representation(self):
        unrestricted = self.model("F2")
        aligned = self.model("F3")
        aligned.load_state_dict(unrestricted.state_dict(), strict=True)
        inputs = self.inputs()
        inputs["radiomics_tokens"].requires_grad_()
        probe = torch.arange(1.0, 17.0)
        for model, expected_nonmatch in ((unrestricted, True), (aligned, False)):
            with self.subTest(mode=model.fusion_mode):
                _, aux = model(**inputs, return_aux=True)
                local = (aux["habitat_tokens"][:, 1] * probe).sum()
                gradient = torch.autograd.grad(local, inputs["radiomics_tokens"])[0]
                self.assertGreater(float(gradient[:, 0].abs().sum()), 0.0)
                if expected_nonmatch:
                    self.assertGreater(float(gradient[:, 1:].abs().sum()), 0.0)
                else:
                    torch.testing.assert_close(gradient[:, 1:], torch.zeros_like(gradient[:, 1:]), atol=0, rtol=0)

    def test_clinical_routing_is_only_restricted_in_f4(self):
        aligned = self.model("F3")
        routed = self.model("F4")
        routed.load_state_dict(aligned.state_dict(), strict=True)
        inputs = self.inputs()
        inputs["clinical_tokens"].requires_grad_()
        probe = torch.arange(1.0, 17.0)
        for model in (aligned, routed):
            _, aux = model(**inputs, return_aux=True)
            local = (aux["habitat_tokens"][:, 1] * probe).sum()
            gradient = torch.autograd.grad(local, inputs["clinical_tokens"])[0]
            if model.fusion_mode == "F4":
                torch.testing.assert_close(gradient[:, 2], torch.zeros_like(gradient[:, 2]), atol=0, rtol=0)
            else:
                self.assertGreater(float(gradient[:, 2].abs().sum()), 0.0)

    def test_all_missing_is_finite_and_region_states_remain_distinguishable(self):
        for mode in ("F0", "F1", "F2", "F3", "F4"):
            with self.subTest(mode=mode):
                model = self.model(mode)
                inputs = self.inputs()
                for name in ("image_presence", "radiomics_presence", "clinical_presence"):
                    inputs[name].zero_()
                # Invalid values in unavailable measurements must not leak into outputs.
                for name in ("image_tokens", "radiomics_tokens", "clinical_tokens"):
                    inputs[name].fill_(float("nan"))
                inputs["region_states"][:] = REGION_STATUS_CODES["missing"]
                missing = model(**inputs)["OS"]
                inputs["region_states"][:] = REGION_STATUS_CODES["absent"]
                absent = model(**inputs)["OS"]
                self.assertTrue(bool(torch.isfinite(missing).all()))
                self.assertTrue(bool(torch.isfinite(absent).all()))
                self.assertFalse(torch.allclose(missing, absent))

    def test_flat_transformer_retains_anatomical_identifiers(self):
        model = self.model("F1")
        inputs = self.inputs()
        original = model(**inputs)["OS"]
        inputs["image_tokens"] = inputs["image_tokens"][:, [0, 3, 2, 1, 4, 5]]
        self.assertFalse(torch.allclose(original, model(**inputs)["OS"]))

    def test_node_extension_requires_explicit_legacy_mode(self):
        with self.assertRaisesRegex(ValueError, "node/topology"):
            self.model("F3", node_token_dim=9)
        model = self.model("F3")
        with self.assertRaisesRegex(ValueError, "controlled"):
            model(**self.inputs(), node_tokens=torch.zeros(2, 1, 9))

    def test_qualified_region_payload_reaches_backbone_and_head(self):
        class RecordingBackbone(torch.nn.Module):
            out_dim = 8
            num_tokens = 6

            def forward(self, image, **kwargs):
                self.received = kwargs
                tokens = image.new_zeros(image.shape[0], 6, 8)
                presence = torch.ones(image.shape[0], 6, device=image.device, dtype=torch.bool)
                return (tokens, presence, {}) if kwargs["return_aux"] else (tokens, presence)

        backbone = RecordingBackbone()
        wrapper = ContourAwareHabitatSurvivalModel(
            backbone=backbone, num_time_bins=3, time_bin_width_days=120,
            model_dim=16, num_heads=4, transformer_layers=1, dropout=0.0,
        ).eval()
        states = torch.tensor([[0, 1, 2, 3]])
        payload = dict(
            region_masks=torch.zeros(1, 4, 4, 4, 4), region_validity=torch.ones(1, 4, dtype=torch.bool),
            region_states=states, physical_geometry=torch.zeros(1, 20),
        )
        _, aux = wrapper(torch.zeros(1, 1, 4, 4, 4), **payload, return_aux=True)
        for name, value in payload.items():
            self.assertIs(backbone.received[name], value)
        torch.testing.assert_close(aux["region_states"], states)


if __name__ == "__main__":
    unittest.main()
