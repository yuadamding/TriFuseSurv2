"""Compatibility shim for the LoRA stage 2 survival trainer (now merged into train.py with --use_lora)."""

from trifusesurv.multimodal_survival.train import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
