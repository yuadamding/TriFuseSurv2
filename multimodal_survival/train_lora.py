"""LoRA-specific stage 2 entry point."""

from __future__ import annotations

import sys

from trifusesurv.multimodal_survival.train import main as train_main


def main() -> None:
    if "--use_lora" not in sys.argv:
        sys.argv.append("--use_lora")
    train_main()


if __name__ == "__main__":
    main()
