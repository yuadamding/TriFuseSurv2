"""Prospectively specified, training-reference survival evaluation."""

from .curves import hazards_to_survival, write_survival_curves

__all__ = ["hazards_to_survival", "write_survival_curves"]
