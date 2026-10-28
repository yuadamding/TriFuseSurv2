#!/usr/bin/env python3
"""Diagnose the immutable budget-A CUDA fixture; never launch training."""
from retained_cuda_diagnostic import main


if __name__ == "__main__":
    raise SystemExit(main())
