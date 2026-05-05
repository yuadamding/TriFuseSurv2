"""Unit tests for v2 Grad-CAM core utilities."""

from __future__ import annotations

import unittest
import importlib.util

try:
    _HAVE_DEPS = importlib.util.find_spec("torch") is not None
except Exception:
    _HAVE_DEPS = False

if _HAVE_DEPS:
    import pandas as pd
    import torch

    from trifusesurv2.explain.gradcam_v2_core import (
        assert_v208_v2_checkpoint,
        image_habitat_names,
        supports_from_backbone_aux,
    )
    from trifusesurv2.multimodal_survival.generate_oof_gradcam_v208 import _oof_lookup


@unittest.skipUnless(_HAVE_DEPS, "torch is not available in this runtime")
class GradcamCoreTest(unittest.TestCase):
    def test_checkpoint_audit_accepts_common_state_aliases_and_rejects_v1(self):
        state = {
            "backbone.x": torch.zeros(1),
            "habitat_model.x": torch.zeros(1),
            "habitat_model.habitat_fusers.global.x": torch.zeros(1),
            "habitat_model.sequence_encoder.x": torch.zeros(1),
            "habitat_model.survival_heads.x": torch.zeros(1),
        }
        for key in ("model_state", "model_state_dict", "state_dict"):
            assert_v208_v2_checkpoint({key: state, "args": {"model_version": "v2"}}, checkpoint_path="relative.pt")

        with self.assertRaises(RuntimeError):
            assert_v208_v2_checkpoint(
                {"model_state": {"gate_mlp.0.weight": torch.zeros(1)}, "args": {"model_version": "v1"}},
                checkpoint_path="legacy.pt",
            )

    def test_supports_include_disjoint_peri_masks(self):
        pt = torch.zeros(1, 1, 1, 3, 3)
        ln = torch.zeros_like(pt)
        pt_shell = torch.zeros_like(pt)
        ln_shell = torch.zeros_like(pt)
        body = torch.ones_like(pt)
        pt[..., 0, 0] = 1.0
        ln[..., 0, 1] = 1.0
        pt_shell[..., 0, 0] = 1.0
        pt_shell[..., 1, 1] = 1.0
        ln_shell[..., 0, 1] = 1.0
        ln_shell[..., 2, 2] = 1.0
        supports = supports_from_backbone_aux(
            {
                "pt_used": pt,
                "ln_used": ln,
                "pt_shell": pt_shell,
                "ln_shell": ln_shell,
                "body": body,
            }
        )
        self.assertEqual(float(supports["pt_peri_disjoint"].sum().item()), 1.0)
        self.assertEqual(float(supports["ln_peri_disjoint"].sum().item()), 1.0)
        self.assertEqual(float(supports["habitat_union_disjoint"].sum().item()), 4.0)

    def test_image_habitat_names_follow_model_when_available(self):
        class _Habitat:
            image_habitats = ("global", "pt_intra", "pt_peri_3mm")

        class _Model:
            habitat_model = _Habitat()

        self.assertEqual(image_habitat_names(_Model()), ("global", "pt_intra", "pt_peri_3mm"))

    def test_oof_lookup_enforces_available_provenance_columns(self):
        df = pd.DataFrame(
            [
                {
                    "case_id": "P1",
                    "risk_endpoint": "DSS",
                    "risk_horizon_days": 1095.0,
                    "risk_score": 0.25,
                    "fold": 0,
                    "checkpoint": "best",
                    "weights_type": "ema",
                    "model_version": "v2",
                    "software_version": "2.0.8",
                    "commit_sha": "a3de12d6fa7b426995b859cd9574f5a6355a01d2",
                }
            ]
        )
        risk, notes = _oof_lookup(
            df,
            "P1",
            id_col="case_id",
            endpoint="DSS",
            horizon_days=1095.0,
            fold=0,
            checkpoint="best",
            weights="ema",
        )
        self.assertEqual(risk, 0.25)
        self.assertEqual(notes, "")
        with self.assertRaises(RuntimeError):
            _oof_lookup(
                df,
                "P1",
                id_col="case_id",
                endpoint="DSS",
                horizon_days=1095.0,
                fold=1,
                checkpoint="best",
                weights="ema",
            )


if __name__ == "__main__":
    unittest.main()
