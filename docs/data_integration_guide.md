# Data Integration Guide

This guide assumes you run direct `python -m ...` commands from the workspace root that contains `TriFuseSurv_package`:

```text
opscc_survival_time_event.csv
clinical_covariate.csv
cohort_radiomics_patient_wide.csv
OPSCC/
TriFuseSurv_package/
```

The recommended flow is:

1. Preprocess DICOM + RTSTRUCT into fixed-size NIfTI volumes.
2. Merge survival, clinical, and radiomics coverage into the preprocessed cohort CSV.
3. Generate patient-level CV splits.
4. Run stage 1 segmentation pretraining.
5. Run stage 2 multimodal survival training.

## Step 1: Preprocess Imaging

```bash
PYTHONPATH=TriFuseSurv_package/src python3 -m trifusesurv.preprocessing.export_swinunetr \
  --root OPSCC \
  --surv_csv opscc_survival_time_event.csv \
  --out_root OPSCC_preprocessed_128 \
  --spacing 0.5 0.5 1 \
  --size 128 256 256 \
  --margin_mm 30
```

Output:

- `OPSCC_preprocessed_128/cohort_preprocessed.csv`
- per-patient `ct.nii.gz`, `mask_primary.nii.gz`, `mask_nodal.nii.gz`

## Step 2: Prepare the Stage-2 Metafile

```bash
PYTHONPATH=TriFuseSurv_package/src python3 -m trifusesurv.preprocessing.prepare_opscc_tabular \
  --base_meta_csv OPSCC_preprocessed_128/cohort_preprocessed.csv \
  --surv_csv opscc_survival_time_event.csv \
  --clin_csv clinical_covariate.csv \
  --radio_csv cohort_radiomics_patient_wide.csv \
  --out_dir OPSCC_preprocessed_128 \
  --out_csv cohort_preprocessed_stage2.csv
```

Output:

- `OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv`
- `OPSCC_preprocessed_128/preparation_summary.json`

This step:

- preserves CT and mask paths from preprocessing
- refreshes survival endpoints from the survival CSV
- merges clinical covariates by normalized patient ID
- records radiomics coverage for the patient-wide radiomics table

## Step 3: Make CV Splits

```bash
PYTHONPATH=TriFuseSurv_package/src python3 -m trifusesurv.preprocessing.make_cv_splits \
  --meta_csv OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv \
  --endpoint OS \
  --cv_folds 4 \
  --val_frac 0.2 \
  --split_seed 1 \
  --out_dir runs/opscc_splits_os_seed1
```

## Step 4: Train

Stage 1:

```bash
./scripts/run_stage1_pretrain_pt.sh
./scripts/run_stage1_pretrain_ln.sh
```

Stage 2:

```bash
./scripts/run_stage2_survival.sh
```

The stage-2 wrapper now points at the current patient-wide radiomics CSV by default:

- `cohort_radiomics_patient_wide.csv`

## Integrated Driver

For the packaged end-to-end runner, use:

```bash
cd TriFuseSurv_package
./scripts/run_two_stage_pipeline_2xh100.sh
```

Its settings live in:

- `scripts/config/pipeline_2xh100_test.env`

Those defaults already match the current relative layout above.
