"""Explainability utilities for TriFuseSurv2."""

from trifusesurv2.explain.gradcam_v2_core import (
    SOFTWARE_VERSION,
    TARGET_COMMIT_SHA,
    assert_v207_v2_checkpoint,
    signed_gradcam_3d,
)

__all__ = [
    "SOFTWARE_VERSION",
    "TARGET_COMMIT_SHA",
    "assert_v207_v2_checkpoint",
    "signed_gradcam_3d",
]
