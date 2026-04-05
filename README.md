# TriFuseSurv

TriFuseSurv is an installable `src`-layout Python package around a two-stage training workflow:

0. Preparation: preprocess DICOM/RTSTRUCT data and generate patient-level split files.
1. Stage 1: pretrain a SwinViT / SwinUNETR encoder on tumor segmentation.
2. Stage 2: fuse three modalities for survival prediction using the pretrained image backbone.

## Layout

- `src/trifusesurv/`: package source
- `src/trifusesurv/models/`: reusable model components (SwinUNETR backbones, LoRA, survival model)
- `src/trifusesurv/utils/`: shared utilities (survival metrics/losses, clinical encoder, radiomics encoder, dataset)
- `src/trifusesurv/preprocessing/`: cohort export and CV split generation
- `src/trifusesurv/segmentation/`: stage 1 segmentation training, evaluation, Grad-CAM
- `src/trifusesurv/multimodal_survival/`: stage 2 survival training (with optional LoRA), SHAP analysis
- `src/trifusesurv/cli/`: compatibility shims for historical CLI paths
- `docs/`: notes and command examples
- `scripts/`: runnable shell scripts for each pipeline step

## Install

```bash
./scripts/install_env.sh
source .venv/bin/activate
```

If you need a non-default PyTorch wheel source, set `TORCH_INDEX_URL` before running `scripts/install_env.sh`.

## CLI Entry Points

- `trifusesurv-preprocess-export-swinunetr` -- export CT and mask volumes from DICOM
- `trifusesurv-make-cv-splits` -- create patient-level fold files
- `trifusesurv-stage1-train-seg` -- stage 1 segmentation pretraining
- `trifusesurv-stage1-eval-seg` -- stage 1 evaluation (Dice)
- `trifusesurv-stage1-gradcam-seg` -- stage 1 Grad-CAM
- `trifusesurv-stage2-train-survival` -- stage 2 multimodal survival training (pass `--use_lora` for LoRA)
- `trifusesurv-stage2-train-survival-lora` -- alias for stage 2 training
- `trifusesurv-stage2-shap-tokens` -- token-level SHAP analysis
- `trifusesurv-stage2-shap-grouped` -- grouped permutation SHAP (survival probability)
- `trifusesurv-stage2-shap-grouped-v2` -- alias for grouped SHAP

Legacy entry points (`trifusesurv-moe-train`, `trifusesurv-seg-pretrain`, etc.) are preserved via CLI shims.

## Pipeline

- Preparation:
  run `trifusesurv-preprocess-export-swinunetr` to export CT and mask volumes, then `trifusesurv-make-cv-splits` to create patient-level fold files.
- Stage 1 segmentation:
  train with `trifusesurv-stage1-train-seg`, evaluate with `trifusesurv-stage1-eval-seg`, and inspect attention with `trifusesurv-stage1-gradcam-seg`.
- Stage 2 multimodal survival:
  train with `trifusesurv-stage2-train-survival` (add `--use_lora` for LoRA fine-tuning), then run SHAP analysis with `trifusesurv-stage2-shap-tokens` or `trifusesurv-stage2-shap-grouped`.
- Two-stage H100 test driver:
  source settings from `scripts/config/pipeline_2xh100_test.env`, then run `./scripts/run_two_stage_pipeline_2xh100.sh`.

## Zip Bundle

The distributable zip contains:

- `src/`, `scripts/`, `docs/`, `README.md`, and `pyproject.toml`
- `scripts/install_env.sh` to create `.venv` and install dependencies
- the stage wrappers under `scripts/` to run preprocessing, stage 1, and stage 2

Rebuild the zip after changes with:

```bash
./scripts/build_zip_package.sh
```
