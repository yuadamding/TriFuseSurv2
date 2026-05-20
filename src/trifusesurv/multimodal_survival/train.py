"""Backward-compatible training entrypoint alias."""

from trifusesurv2.multimodal_survival.train import *  # noqa: F401,F403
from trifusesurv2.multimodal_survival.train import main


if __name__ == "__main__":
    raise SystemExit(main())
