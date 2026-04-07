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

- `trifusesurv-prepare-opscc-tabular` -- prepare OPSCC cohort from CSV files (no DICOM required)
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

### Full Pipeline (with DICOM imaging)

- Preparation:
  run `trifusesurv-preprocess-export-swinunetr` to export CT and mask volumes, then `trifusesurv-prepare-opscc-tabular` to merge `opscc_survival_time_event.csv`, `clinical_covariate.csv`, and `cohort_radiomics_patient_wide.csv` into the stage-2 metafile, then `trifusesurv-make-cv-splits` to create patient-level fold files.
- Stage 1 segmentation:
  train with `trifusesurv-stage1-train-seg`, evaluate with `trifusesurv-stage1-eval-seg`, and inspect attention with `trifusesurv-stage1-gradcam-seg`.
- Stage 2 multimodal survival:
  train with `trifusesurv-stage2-train-survival` (add `--use_lora` for LoRA fine-tuning), then run SHAP analysis with `trifusesurv-stage2-shap-tokens` or `trifusesurv-stage2-shap-grouped`.
- Two-stage H100 test driver:
  source settings from `scripts/config/pipeline_2xh100_test.env`, then run `./scripts/run_two_stage_pipeline_2xh100.sh`.
  The packaged defaults assume this relative layout from the workspace root that contains `TriFuseSurv_package`:
  - `OPSCC`
  - `opscc_survival_time_event.csv`
  - `clinical_covariate.csv`
  - `cohort_radiomics_patient_wide.csv`
  GPU assignment is automatic by default: the pipeline detects all available GPUs and distributes stage jobs across them unless you override the GPU fields in the settings file.

### Stage-2 Metafile Preparation

`trifusesurv-prepare-opscc-tabular` is the bridge between preprocessing/stage 1 and stage 2. It keeps the CT and mask paths from `cohort_preprocessed.csv`, refreshes the survival endpoints from `opscc_survival_time_event.csv`, merges the clinical covariates, and records which patients are covered by the patient-wide radiomics CSV.

Example:
```bash
trifusesurv-prepare-opscc-tabular \
  --base_meta_csv OPSCC_preprocessed_128/cohort_preprocessed.csv \
  --surv_csv opscc_survival_time_event.csv \
  --clin_csv clinical_covariate.csv \
  --radio_csv cohort_radiomics_patient_wide.csv \
  --out_dir OPSCC_preprocessed_128 \
  --out_csv cohort_preprocessed_stage2.csv

trifusesurv-make-cv-splits \
  --meta_csv OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv \
  --endpoint OS \
  --cv_folds 4 \
  --out_dir runs/opscc_splits

trifusesurv-stage2-train-survival \
  --meta_csv OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv \
  --splits_dir runs/opscc_splits \
  --use_radiomics \
  --radiomics_root cohort_radiomics_patient_wide.csv
```

## Zip Bundle

The distributable zip contains:

- `src/`, `scripts/`, `docs/`, `README.md`, and `pyproject.toml`
- `scripts/install_env.sh` to create `.venv` and install dependencies
- the stage wrappers under `scripts/` to run preprocessing, stage 1, and stage 2

Rebuild the zip after changes with:

```bash
./scripts/build_zip_package.sh
```
