"""Compatibility shim for grouped OOF SHAP v2 export (now merged into export_grouped_shap.py)."""

from trifusesurv.multimodal_survival.export_grouped_shap import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
