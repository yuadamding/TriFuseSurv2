"""CPU regressions for persisted supports and annotation-aware supervision."""

import contextlib
import unittest

import torch

from trifusesurv2.models.contour_habitat_survival import ContourAwareHabitatSurvivalModel
from trifusesurv2.models.swinunetr_shared_roi_token_backbone import ContourAwareROITokenBackbone
from trifusesurv2.multimodal_survival.train import train_one_epoch
from trifusesurv2.schema import PHYSICAL_GEOMETRY_FEATURES, REGION_STATUS_CODES
from trifusesurv2.utils.roi_focus import roi_focus_metrics


class RecordingTrainingModel(ContourAwareHabitatSurvivalModel):
    """Keep per-patient localization parameters so target leakage is observable."""

    def __init__(self, batch_size):
        torch.nn.Module.__init__(self)
        self.habitat_model = torch.nn.Identity()
        self.localization = torch.nn.Parameter(torch.zeros(batch_size, 2, 1, 2, 2, 2))
        self.presence = torch.nn.Parameter(torch.zeros(batch_size, 2))
        self.survival = torch.nn.Parameter(torch.zeros(batch_size, 3))

    def forward(self, x, clinical, radiomics, **kwargs):
        self.received = kwargs
        pt, ln = self.localization[:, 0], self.localization[:, 1]
        regions = kwargs["region_masks"]
        aux = {
            "loc_pt_logits": pt, "loc_ln_logits": ln,
            "pt_prob": pt.sigmoid(), "ln_prob": ln.sigmoid(),
            "pt_used": regions[:, 0:1], "ln_used": regions[:, 2:3],
            "pt_soft_used": regions[:, 0:1], "ln_soft_used": regions[:, 2:3],
            "pt_shell": regions[:, 1:2], "ln_shell": regions[:, 3:4],
            "pt_presence_logits": self.presence[:, 0],
            "ln_presence_logits": self.presence[:, 1],
        }
        return {name: self.survival for name in ("OS", "DSS", "DFS")}, {"backbone_aux": aux}


class VerifiedPipelineTests(unittest.TestCase):
    @staticmethod
    def _backbone():
        return ContourAwareROITokenBackbone(
            img_size=(32, 32, 32), feature_size=12, depths=(1, 1, 1, 1),
            num_heads=(3, 6, 12, 24), token_dim=16, token_mlp_dropout=0.0,
            token_dropout=0.0, drop_rate=0.0, attn_drop_rate=0.0,
            dropout_path_rate=0.0, min_roi_voxels_deep=0,
            require_verified_regions=True,
        )

    @staticmethod
    def _payload(batch_size, side):
        regions = torch.zeros(batch_size, 4, side, side, side)
        return {
            "region_masks": regions,
            "region_validity": torch.zeros(batch_size, 4, dtype=torch.bool),
            "region_states": torch.full((batch_size, 4), REGION_STATUS_CODES["missing"], dtype=torch.long),
            "physical_geometry": torch.zeros(batch_size, 2 * len(PHYSICAL_GEOMETRY_FEATURES)),
        }

    def test_full_model_uses_persisted_supports_independent_of_localizer_and_teacher_forcing(self):
        torch.manual_seed(39)
        backbone = self._backbone()
        model = ContourAwareHabitatSurvivalModel(
            backbone=backbone, fusion_mode="F3", num_time_bins=3,
            time_bin_width_days=120.0, model_dim=16, num_heads=4,
            transformer_layers=1, dropout=0.0,
        ).eval()
        image = torch.rand(2, 1, 32, 32, 32)
        payload = self._payload(2, 32)
        regions = payload["region_masks"]
        regions[0, 0, 12:16, 12:16, 12:16] = 1
        regions[0, 1, 8:12, 12:16, 12:16] = 1
        payload["region_validity"][0] = True
        payload["region_states"][0] = torch.tensor([0, 0, 1, 1])
        # These contradict the persisted supports deliberately: they may be
        # auxiliary supervision, but cannot redefine the pooled measurements.
        raw = torch.ones_like(image)
        outputs, aux = model(image, mask_pt=raw, mask_ln=raw,
                             teacher_force_alpha=0.0, return_aux=True, **payload)
        with torch.no_grad():
            for parameter in backbone.loc_pt_head.parameters():
                parameter.fill_(2.0)
            for parameter in backbone.loc_ln_head.parameters():
                parameter.fill_(-2.0)
        other, other_aux = model(image, mask_pt=raw, mask_ln=raw,
                                teacher_force_alpha=1.0, return_aux=True, **payload)
        for endpoint in outputs:
            torch.testing.assert_close(outputs[endpoint], other[endpoint], rtol=0, atol=0)
            self.assertTrue(bool(torch.isfinite(outputs[endpoint]).all()))
        for name, column in (("pt_used", 0), ("pt_shell", 1), ("ln_used", 2), ("ln_shell", 3)):
            torch.testing.assert_close(other_aux["backbone_aux"][name], regions[:, column:column + 1])
        expected = torch.tensor([[1, 1, 1, 0, 0, 1], [1, 0, 0, 0, 0, 1]], dtype=torch.bool)
        torch.testing.assert_close(aux["habitat_presence"], expected)
        torch.testing.assert_close(other_aux["habitat_presence"], expected)
        sum(value.square().mean() for value in other.values()).backward()
        encoder_grads = [p.grad for p in backbone.backbone_shared.parameters() if p.grad is not None]
        self.assertTrue(encoder_grads)
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in encoder_grads))
        self.assertTrue(all(p.grad is None for p in backbone.loc_pt_head.parameters()))

    def test_verified_backbone_fails_closed_without_complete_physical_payload(self):
        backbone = self._backbone()
        image = torch.zeros(1, 1, 32, 32, 32)
        with self.assertRaisesRegex(ValueError, "required"):
            backbone(image)
        with self.assertRaisesRegex(ValueError, "validity"):
            backbone(image, region_masks=torch.zeros(1, 4, 32, 32, 32))
        payload = self._payload(1, 32)
        payload["region_masks"][0, 0, 0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "states must agree"):
            backbone(image, **payload)
        for field, value, message in (("region_validity", 0.5, "binary"),
                                      ("region_states", 0.5, "integer")):
            with self.subTest(field=field):
                payload = self._payload(1, 32)
                payload[field] = payload[field].float()
                payload[field][0, 0] = value
                with self.assertRaisesRegex(ValueError, message):
                    backbone(image, **payload)

    @staticmethod
    def _run_training_batch(payload):
        batch_size = len(payload["region_masks"])
        model = RecordingTrainingModel(batch_size)
        batch = dict(payload)
        batch.update(
            x=torch.zeros(batch_size, 1, 2, 2, 2),
            mask_pt=payload["region_masks"][:, 0:1],
            mask_ln=payload["region_masks"][:, 2:3],
            t=torch.full((batch_size,), 60.0), e=torch.ones(batch_size),
            t_all=torch.full((batch_size, 3), 60.0), e_all=torch.ones(batch_size, 3),
            pid=[f"synthetic-{i}" for i in range(batch_size)],
        )
        stats = train_one_epoch(
            model, [batch], torch.optim.SGD(model.parameters(), lr=0.1), None,
            torch.device("cpu"), fold=0, epoch=1, primary_endpoint="OS",
            num_time_bins=3, time_bin_width_days=120.0,
            primary_surv_loss_weight=1.0, aux_surv_loss_weight=1.0,
            hazard_smooth_lambda=0.0, logit_l2_lambda=0.0,
            gate_entropy_lambda=0.0, gate_loadbal_lambda=0.0,
            teacher_force_alpha=0.5, survival_uses_teacher_forced_masks=False,
            survival_loss_weight=0.0, loc_loss_pt_lambda=1.0, loc_loss_ln_lambda=1.0,
            loc_presence_lambda=1.0, mask_support_lambda=1.0,
            mask_support_bce_weight=1.0, mask_support_dice_weight=1.0,
            mask_support_balanced_bce=True, mask_support_pos_weight_cap=500.0,
            mask_focus_lambda=1.0, pt_shell_radius=3, pt_shell_thickness_mm=10.0,
            loc_bce_weight=1.0, loc_dice_weight=1.0, loc_balanced_bce=True,
            loc_pos_weight_cap=500.0, log_every_batches=0, ema=None,
            autocast_ctx=contextlib.nullcontext,
        )
        return model, stats

    def test_training_updates_known_absence_but_never_missing_or_invalid_annotations(self):
        payload = self._payload(4, 2)
        payload["region_validity"][:2] = True
        payload["region_states"][0] = torch.tensor([0, 1, 1, 1])
        payload["region_states"][1] = REGION_STATUS_CODES["absent"]
        payload["region_states"][3] = REGION_STATUS_CODES["invalid"]
        payload["region_masks"][0, 0, 0, 0, 0] = 1
        model, stats = self._run_training_batch(payload)
        self.assertGreater(float(model.localization[0].detach().abs().sum()), 0.0)
        self.assertGreater(float(model.localization[1].detach().abs().sum()), 0.0)
        self.assertGreater(float(model.presence[1].detach().abs().sum()), 0.0)
        torch.testing.assert_close(model.localization[2:], torch.zeros_like(model.localization[2:]), rtol=0, atol=0)
        torch.testing.assert_close(model.presence[2:], torch.zeros_like(model.presence[2:]), rtol=0, atol=0)
        for name, value in payload.items():
            torch.testing.assert_close(model.received[name], value)
        self.assertGreater(stats["loss_loc_pt"], 0.0)

    def test_all_unavailable_annotations_produce_finite_zero_localization_losses(self):
        model, stats = self._run_training_batch(self._payload(2, 2))
        for name in ("loss_loc_pt", "loss_loc_ln", "loss_loc_presence", "loss_mask_support", "loss_mask_focus", "loss_total"):
            self.assertEqual(stats[name], 0.0)
        torch.testing.assert_close(model.localization, torch.zeros_like(model.localization), rtol=0, atol=0)
        torch.testing.assert_close(model.presence, torch.zeros_like(model.presence), rtol=0, atol=0)

    def test_roi_diagnostics_exclude_unavailable_annotations(self):
        target = torch.zeros(3, 1, 2, 2, 2)
        target[0, :, 0, 0, 0] = 1
        # The second row is a known absence; the third has no annotation.
        metrics = roi_focus_metrics(
            pred_prob=target, target_mask=target,
            annotation_validity=torch.tensor([True, True, False]),
        )
        self.assertEqual(float(metrics["gt_present_frac"]), 0.5)
        self.assertEqual(float(metrics["support_dice"]), 1.0)
        unavailable = roi_focus_metrics(
            pred_prob=target, target_mask=target,
            annotation_validity=torch.zeros(3, dtype=torch.bool),
        )
        self.assertTrue(all(bool(torch.isnan(value)) for value in unavailable.values()))


if __name__ == "__main__":
    unittest.main()
