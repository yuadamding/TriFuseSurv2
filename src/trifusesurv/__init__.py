"""Backward-compatible import alias for older TriFuseSurv launchers.

TriFuseSurv2 is packaged under :mod:`trifusesurv2`. Some existing cluster
wrappers may still launch modules under the historical :mod:`trifusesurv`
namespace, so this package forwards those imports instead of failing before
training starts.
"""

from trifusesurv2 import *  # noqa: F401,F403

