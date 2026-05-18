"""Regression tests for deployment-like evaluation mask handling."""

from __future__ import annotations

import contextlib
import importlib.util
import unittest

_REQUIRED = ("numpy", "pandas", "SimpleITK", "torch", "monai", "sklearn")
try:
    _HAVE_DEPS = all(importlib.util.find_spec(name) is not None for name in _REQUIRED)
except Exception:
    _HAVE_DEPS = False

if _HAVE_DEPS:
    import torch
    import torch.nn as nn

    from trifusesurv2.multimodal_survival.train import _model_forward_eval


@unittest.skipUnless(_HAVE_DEPS, "training/eval dependencies are not available in this runtime")
class EvalMaskInvarianceTest(unittest.TestCase):
    def test_eval_logits_are_invariant_to_masks_when_teacher_force_is_zero(self):
        class MaskSensitiveDummy(nn.Module):
            def forward(
                self,
                *,
                x_img,
                clinical,
                radiomics,
                mask_pt=None,
                mask_ln=None,
                teacher_force_alpha=0.0,
                return_gate=False,
            ):
                base = x_img.flatten(1).mean(dim=1, keepdim=True).repeat(1, 4)
                if float(teacher_force_alpha) > 0.0:
                    mask_term = mask_pt.flatten(1).mean(dim=1, keepdim=True)
                    mask_term = mask_term + mask_ln.flatten(1).mean(dim=1, keepdim=True)
                    base = base + float(teacher_force_alpha) * mask_term
                return {"OS": base}

        device = torch.device("cpu")
        model = MaskSensitiveDummy().eval()
        x = torch.arange(8, dtype=torch.float32).view(1, 1, 2, 2, 2)
        payload_gt = {
            "x": x,
            "clin": None,
            "rad": None,
            "mask_pt": torch.ones_like(x),
            "mask_ln": torch.zeros_like(x),
        }
        payload_zero = {
            **payload_gt,
            "mask_pt": torch.zeros_like(x),
            "mask_ln": torch.zeros_like(x),
        }
        payload_rand = {
            **payload_gt,
            "mask_pt": torch.linspace(0, 1, steps=8, dtype=torch.float32).view_as(x),
            "mask_ln": torch.linspace(1, 0, steps=8, dtype=torch.float32).view_as(x),
        }

        autocast_ctx = lambda: contextlib.nullcontext()
        with torch.no_grad():
            logits_gt = _model_forward_eval(model, payload_gt, device, autocast_ctx, teacher_force_alpha=0.0)["OS"]
            logits_zero = _model_forward_eval(model, payload_zero, device, autocast_ctx, teacher_force_alpha=0.0)["OS"]
            logits_rand = _model_forward_eval(model, payload_rand, device, autocast_ctx, teacher_force_alpha=0.0)["OS"]
            logits_leaky = _model_forward_eval(model, payload_gt, device, autocast_ctx, teacher_force_alpha=1.0)["OS"]

        self.assertTrue(torch.equal(logits_gt, logits_zero))
        self.assertTrue(torch.equal(logits_gt, logits_rand))
        self.assertFalse(torch.equal(logits_gt, logits_leaky))


if __name__ == "__main__":
    unittest.main()
