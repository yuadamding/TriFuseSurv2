"""Actual-mask and local-output checks for the prespecified routing control."""

import hashlib
import itertools
from pathlib import Path
import unittest

import torch

from trifusesurv2.models.contour_habitat_survival import ContourAwareHabitatSurvivalModel
from trifusesurv2.models.habitat_survival import HabitatAlignedSurvivalModel
from trifusesurv2.models.routing_intervention import (
    REGIONAL_ORDER,
    ROUTING_DESIGN_SHA256,
    ROUTING_EXPORT_LABEL,
    ROUTING_INTERVENTION_ID,
    availability_cycle_assignment,
    routing_intervention_metadata,
)
from trifusesurv2.schema import IMAGE_HABITATS


def reference_assignment(q, r):
    """Literal four-class specification, independent of the tensor algorithm."""
    destination = list(range(4))
    for signature in itertools.product((False, True), repeat=2):
        group = [i for i in range(4) if (bool(q[i]), bool(r[i])) == signature]
        for i, source in enumerate(group):
            destination[source] = group[(i + 1) % len(group)]
    return destination


class MismatchedRoutingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def model(self, *, intervention=ROUTING_INTERVENTION_ID, **kwargs):
        return HabitatAlignedSurvivalModel(
            image_token_dim=8, radiomics_token_dim=5, clinical_token_dim=4,
            num_time_bins=3, model_dim=16, num_heads=4, transformer_layers=1,
            dropout=0.0, fusion_mode="F3", routing_intervention=intervention,
            **kwargs,
        ).eval()

    def inputs(self, q, r):
        batch = len(q)
        gen = torch.Generator().manual_seed(9137)
        image_presence = torch.ones(batch, 6, dtype=torch.bool)
        image_presence[:, 1:5] = q
        clinical_presence = torch.ones(batch, 3, dtype=torch.bool)
        clinical_presence[::3, 0] = False
        return {
            "image_tokens": torch.randn(batch, 6, 8, generator=gen),
            "image_presence": image_presence,
            "radiomics_tokens": torch.randn(batch, 4, 5, generator=gen),
            "radiomics_presence": r,
            "clinical_tokens": torch.randn(batch, 3, 4, generator=gen),
            "clinical_presence": clinical_presence,
        }

    @staticmethod
    def patterns():
        pairs = list(itertools.product(itertools.product((False, True), repeat=4), repeat=2))
        q = torch.tensor([pair[0] for pair in pairs], dtype=torch.bool)
        r = torch.tensor([pair[1] for pair in pairs], dtype=torch.bool)
        return q, r

    def test_design_identity_and_constructor_boundary(self):
        spec = Path(__file__).resolve().parents[1] / "configs/mismatched_routing_spec.json"
        self.assertEqual(hashlib.sha256(spec.read_bytes()).hexdigest(), ROUTING_DESIGN_SHA256)
        self.assertEqual(routing_intervention_metadata(None), {})
        self.assertEqual(routing_intervention_metadata(ROUTING_INTERVENTION_ID), {
            "routing_intervention": ROUTING_INTERVENTION_ID,
            "routing_intervention_design_sha256": ROUTING_DESIGN_SHA256,
            "routing_intervention_export_label": ROUTING_EXPORT_LABEL,
        })
        for mode in ("F0", "F1", "F2", "F4", "legacy"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "fusion_mode=F3"):
                HabitatAlignedSurvivalModel(
                    image_token_dim=8, radiomics_token_dim=5, clinical_token_dim=4,
                    num_time_bins=3, fusion_mode=mode,
                    routing_intervention=ROUTING_INTERVENTION_ID,
                )
        with self.assertRaisesRegex(ValueError, "Unknown routing_intervention"):
            self.model(intervention="unversioned_shuffle")
        with self.assertRaisesRegex(ValueError, "canonical"):
            self.model(image_habitats=tuple(reversed(IMAGE_HABITATS)))

    def test_all_256_actual_masks_attention_and_local_outputs(self):
        q, r = self.patterns()
        expected_assignment = torch.tensor([reference_assignment(qq, rr) for qq, rr in zip(q, r)])
        inputs = self.inputs(q, r)
        model = self.model()
        captured = {}
        handles = []
        for habitat, fuser in model.habitat_fusers.items():
            def capture(_module, args, name=habitat):
                captured[name] = tuple(arg.detach().clone() for arg in args)
            handles.append(fuser.register_forward_pre_hook(capture))
        with torch.no_grad():
            logits, aux = model(**inputs, return_aux=True, return_attention=True)
        for handle in handles:
            handle.remove()
        self.assertTrue(torch.equal(aux["routing_assignment"], expected_assignment))
        self.assertTrue(torch.equal(aux["routing_prelocal_image_presence"], q))
        self.assertTrue(torch.equal(aux["routing_prelocal_radiomics_presence"], r))
        self.assertEqual(aux["routing_intervention"], ROUTING_INTERVENTION_ID)
        self.assertEqual(aux["routing_intervention_design_sha256"], ROUTING_DESIGN_SHA256)
        self.assertEqual(aux["routing_intervention_export_label"], ROUTING_EXPORT_LABEL)
        nominal = aux["local_context_routing"]
        self.assertEqual(tuple(nominal.shape), (256, 6, 8))
        static = model.local_context_routing()
        for i, habitat in enumerate(IMAGE_HABITATS):
            with self.subTest(query=habitat):
                expected = static[i][None].expand(256, -1).clone()
                if habitat in REGIONAL_ORDER:
                    regional = REGIONAL_ORDER.index(habitat)
                    expected[:, 1:5] = torch.nn.functional.one_hot(
                        expected_assignment[:, regional], num_classes=4
                    ).bool()
                self.assertTrue(torch.equal(nominal[:, i], expected))
                availability = torch.cat([
                    inputs["image_presence"][:, i:i + 1], r, inputs["clinical_presence"],
                ], dim=1)
                effective = expected & availability
                self.assertTrue(torch.equal(aux["local_context_mask"][:, i], effective))
                recorded = aux["habitat_attention"][habitat]
                self.assertTrue(torch.equal(recorded["context_mask"], effective))
                self.assertEqual(recorded["context_names"], aux["routing_context_names"][i])
                query, context, actual_mask = captured[habitat]
                self.assertTrue(torch.equal(actual_mask, effective))
                with torch.no_grad():
                    expected_local, _ = model.habitat_fusers[habitat](
                        query, context, effective, return_attention=True
                    )
                    expected_local *= inputs["image_presence"][:, i:i + 1]
                torch.testing.assert_close(aux["habitat_tokens"][:, i], expected_local, atol=0, rtol=0)
                # Empty rows use the pre-existing zero dummy context internally;
                # absent image queries still have exactly zero local output.
                weights = recorded["weights"]
                nonempty = effective.any(dim=1)
                blocked_weights = weights[nonempty] * (~effective[nonempty])[:, None, None, :]
                self.assertEqual(float(blocked_weights.abs().sum()), 0.0)
        for tensor in logits.values():
            self.assertTrue(bool(torch.isfinite(tensor).all()))
        self.assertEqual(float(aux["habitat_tokens"][:, 1:5][~q].abs().sum()), 0.0)

    def test_all_patterns_preserve_available_access_key_exposure_and_fallback(self):
        q, r = self.patterns()
        assignment = availability_cycle_assignment(q, r)
        torch.testing.assert_close(assignment.sort(dim=1).values, torch.arange(4).expand(256, -1), atol=0, rtol=0)
        self.assertTrue(torch.equal(r.gather(1, assignment), r))
        self.assertTrue(torch.equal(q.gather(1, assignment), q))
        edges = torch.nn.functional.one_hot(assignment, num_classes=4).to(torch.int64)
        incoming = (edges * q[:, :, None]).sum(dim=1)
        self.assertTrue(torch.equal(incoming, q.to(torch.int64)))
        self.assertTrue(torch.equal(incoming * r, (q & r).to(torch.int64)))
        for row, (qq, rr) in enumerate(zip(q, r)):
            for i in range(4):
                group_size = sum((bool(qq[j]), bool(rr[j])) == (bool(qq[i]), bool(rr[i])) for j in range(4))
                self.assertEqual(int(assignment[row, i]) == i, group_size == 1)

    def test_singleton_effective_fallback_matches_f3(self):
        q = r = torch.tensor([[True, False, False, False]])
        inputs = self.inputs(q, r)
        control, intervention = self.model(intervention=None), self.model()
        intervention.load_state_dict(control.state_dict(), strict=True)
        with torch.no_grad():
            c_logits, c_aux = control(**inputs, return_aux=True, return_attention=True)
            i_logits, i_aux = intervention(**inputs, return_aux=True, return_attention=True)
        for habitat in IMAGE_HABITATS:
            self.assertTrue(torch.equal(c_aux["habitat_attention"][habitat]["context_mask"], i_aux["habitat_attention"][habitat]["context_mask"]))
        self.assertTrue(torch.equal(c_aux["habitat_tokens"], i_aux["habitat_tokens"]))
        for endpoint in c_logits:
            self.assertTrue(torch.equal(c_logits[endpoint], i_logits[endpoint]))

    def test_blocked_key_cannot_change_local_query_but_routed_key_can(self):
        inputs = self.inputs(torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 4, dtype=torch.bool))
        inputs["radiomics_tokens"].requires_grad_()
        model = self.model()
        _, aux = model(**inputs, return_aux=True)
        self.assertEqual(aux["routing_assignment"].tolist(), [[1, 2, 3, 0]])
        local = aux["habitat_tokens"][:, 1]
        probe = torch.arange(1.0, 17.0)
        gradient = torch.autograd.grad((local * probe).sum(), inputs["radiomics_tokens"])[0]
        self.assertEqual(float(gradient[:, [0, 2, 3]].abs().sum()), 0.0)
        self.assertGreater(float(gradient[:, 1].abs().sum()), 0.0)
        changed = dict(inputs)
        changed["radiomics_tokens"] = inputs["radiomics_tokens"].detach().clone()
        changed["radiomics_tokens"][:, 0] += torch.tensor([11.0, -7.0, 3.0, 17.0, -13.0])
        with torch.no_grad():
            _, perturbed = model(**changed, return_aux=True)
        self.assertTrue(torch.equal(local.detach(), perturbed["habitat_tokens"][:, 1]))

    def test_mixed_batch_matches_individual_and_train_eval_routing(self):
        q, r = self.patterns()
        chosen = [0, 1, 5, 17, 31, 59, 99, 128, 177, 210, 255]
        inputs = self.inputs(q[chosen], r[chosen])
        model = self.model()
        with torch.no_grad():
            batched, batched_aux = model(**inputs, return_aux=True)
            for row in range(len(chosen)):
                single = {name: value[row:row + 1] for name, value in inputs.items()}
                logits, aux = model(**single, return_aux=True)
                self.assertTrue(torch.equal(aux["local_context_mask"], batched_aux["local_context_mask"][row:row + 1]))
                torch.testing.assert_close(aux["habitat_tokens"], batched_aux["habitat_tokens"][row:row + 1], atol=2e-6, rtol=2e-6)
                for endpoint in logits:
                    torch.testing.assert_close(logits[endpoint], batched[endpoint][row:row + 1], atol=2e-6, rtol=2e-6)
            model.train()
            trained, train_aux = model(**inputs, return_aux=True)
        self.assertTrue(torch.equal(train_aux["routing_assignment"], batched_aux["routing_assignment"]))
        self.assertTrue(torch.equal(train_aux["local_context_mask"], batched_aux["local_context_mask"]))
        for endpoint in trained:
            torch.testing.assert_close(trained[endpoint], batched[endpoint], atol=2e-6, rtol=2e-6)

    def test_parameter_initialization_rng_and_inactive_f3_shape_unchanged(self):
        for seed in (42, 142, 242):
            torch.manual_seed(seed)
            control = self.model(intervention=None)
            control_rng = torch.get_rng_state().clone()
            torch.manual_seed(seed)
            intervention = self.model()
            self.assertTrue(torch.equal(torch.get_rng_state(), control_rng))
            self.assertEqual(control.parameter_count(), intervention.parameter_count())
            self.assertEqual(list(control.state_dict()), list(intervention.state_dict()))
            for name, value in control.state_dict().items():
                self.assertTrue(torch.equal(value, intervention.state_dict()[name]), name)
        inputs = self.inputs(torch.ones(2, 4, dtype=torch.bool), torch.ones(2, 4, dtype=torch.bool))
        with torch.no_grad():
            _, aux = control(**inputs, return_aux=True)
        self.assertNotIn("routing_intervention", aux)
        self.assertNotIn("routing_assignment", aux)
        for seed in (7, 91):
            torch.manual_seed(seed)
            before = torch.get_rng_state().clone()
            self.assertEqual(availability_cycle_assignment(inputs["image_presence"][:, 1:5], inputs["radiomics_presence"]).tolist(), [[1, 2, 3, 0]] * 2)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
        changed = dict(inputs, image_tokens=inputs["image_tokens"] * 7, radiomics_tokens=inputs["radiomics_tokens"] - 15)
        with torch.no_grad():
            _, a = intervention(**inputs, return_aux=True)
            _, b = intervention(**changed, return_aux=True)
        self.assertTrue(torch.equal(a["routing_assignment"], b["routing_assignment"]))

    def test_invalid_masks_fail_and_missing_measurements_use_actual_absence(self):
        q, r = self.patterns()
        for bad in (q.float(), q[:, :3], q[:, :, None]):
            with self.assertRaisesRegex(ValueError, "Boolean"):
                availability_cycle_assignment(bad, r)
        model = self.model()
        inputs = self.inputs(q[:4], r[:4])
        inputs["radiomics_tokens"] = None
        inputs["radiomics_presence"] = None
        with torch.no_grad():
            _, aux = model(**inputs, return_aux=True)
        self.assertFalse(bool(aux["routing_prelocal_radiomics_presence"].any()))
        self.assertTrue(torch.equal(aux["routing_assignment"], availability_cycle_assignment(q[:4], torch.zeros(4, 4, dtype=torch.bool))))

    def test_contour_wrapper_forwards_actual_backbone_presence(self):
        class FixedBackbone(torch.nn.Module):
            out_dim = 8
            num_tokens = 6

            def forward(self, image, **kwargs):
                tokens = image.new_zeros(image.shape[0], 6, 8)
                presence = torch.tensor([[True, True, True, False, True, True]], device=image.device).expand(image.shape[0], -1)
                return (tokens, presence, {}) if kwargs["return_aux"] else (tokens, presence)

        model = ContourAwareHabitatSurvivalModel(
            backbone=FixedBackbone(), num_time_bins=3, time_bin_width_days=120,
            radiomics_token_dim=5, clinical_token_dim=4, model_dim=16,
            num_heads=4, transformer_layers=1, dropout=0.0,
            routing_intervention=ROUTING_INTERVENTION_ID,
        ).eval()
        rad = torch.ones(1, 4, dtype=torch.bool)
        with torch.no_grad():
            _, aux = model(torch.zeros(1, 1, 4, 4, 4), torch.zeros(1, 3, 4),
                           torch.zeros(1, 4, 5), radiomics_presence=rad, return_aux=True)
        self.assertEqual(model.routing_intervention, ROUTING_INTERVENTION_ID)
        self.assertEqual(aux["routing_prelocal_image_presence"].tolist(), [[True, True, False, True]])
        self.assertEqual(aux["routing_assignment"].tolist(), [[1, 3, 2, 0]])


if __name__ == "__main__":
    unittest.main()
