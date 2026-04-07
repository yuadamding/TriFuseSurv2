# TriFuseSurv - OPSCC Tabular Merge Quick Start

## What This Step Does

This package now uses `prepare_opscc_tabular.py` as the stage between preprocessing/stage 1 and stage 2.
It merges:

1. `opscc_survival_time_event.csv`
2. `clinical_covariate.csv`
3. `cohort_radiomics_patient_wide.csv`

into the preprocessed imaging metafile so stage 2 can train from one CSV plus the patient-wide radiomics table.

## Your Data

```
At the workspace root:
├── opscc_survival_time_event.csv      (519 patients, survival endpoints)
├── clinical_covariate.csv             (1,587 patients, clinical features)
├── cohort_radiomics_patient_wide.csv  (613 patients, radiomics features)
└── TriFuseSurv_package/               (Python package)
```

**Coverage**: 516/519 survival patients have clinical data (99.4%), ~550 have radiomics.

## Quick Start

### Step 1: Prepare Data
```bash
PYTHONPATH=TriFuseSurv_package/src python3 -m trifusesurv.preprocessing.prepare_opscc_tabular \
  --base_meta_csv OPSCC_preprocessed_128/cohort_preprocessed.csv \
  --surv_csv opscc_survival_time_event.csv \
  --clin_csv clinical_covariate.csv \
  --radio_csv cohort_radiomics_patient_wide.csv \
  --out_dir OPSCC_preprocessed_128 \
  --out_csv cohort_preprocessed_stage2.csv
```

**Outputs**:
- `OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv` - merged stage-2 metafile
- `OPSCC_preprocessed_128/preparation_summary.json` - data coverage summary

### Step 2: Create CV Splits
```bash
PYTHONPATH=TriFuseSurv_package/src python3 -m trifusesurv.preprocessing.make_cv_splits \
  --meta_csv OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv \
  --endpoint OS \
  --cv_folds 4 \
  --val_frac 0.2 \
  --split_seed 1 \
  --out_dir runs/opscc_splits_os
```

**Outputs**:
- `runs/opscc_splits_os/splits.csv` - fold assignments
- `runs/opscc_splits_os/fold_00/`, `fold_01/`, etc. - train/val/test IDs

### Step 3: Train Stage 2 (Multimodal Survival)
```bash
PYTHONPATH=TriFuseSurv_package/src python3 -m trifusesurv.multimodal_survival.train \
  --meta_csv OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv \
  --splits_dir runs/opscc_splits_os \
  --use_radiomics \
  --radiomics_root cohort_radiomics_patient_wide.csv \
  --endpoint OS \
  --out_dir runs/opscc_stage2_os \
  --exp_name opscc_tabular_os \
  --device cuda:0 \
  --batch_size 2 \
  --epochs 50
```

**Key Arguments**:
- `--use_radiomics` - enable radiomics features (clinical always included)
- `--radiomics_root` - path to the patient-wide radiomics CSV
- `--endpoint OS` - which survival endpoint (OS/DSS/DFS)
- `--device cuda:0` - GPU device (or cpu if no GPU)

### Step 4: (Optional) SHAP Analysis
```bash
PYTHONPATH=TriFuseSurv_package/src python3 -m trifusesurv.multimodal_survival.export_grouped_shap \
  --ckpt_base_dir runs/opscc_stage2_os \
  --meta_csv OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv \
  --splits_dir runs/opscc_splits_os \
  --radiomics_root cohort_radiomics_patient_wide.csv \
  --endpoint OS \
  --device cuda:0
```

Use this to understand which clinical/radiomics features drive predictions.

## Environment Requirements

The package requires these dependencies (listed in `pyproject.toml`):
- torch
- monai
- numpy
- pandas
- scikit-learn
- shap
- pydicom (not needed for tabular-only)
- SimpleITK (not needed for tabular-only)
- rt-utils (not needed for tabular-only)

To install:
```bash
bash TriFuseSurv_package/scripts/install_env.sh  # Creates TriFuseSurv_package/.venv
source TriFuseSurv_package/.venv/bin/activate
```

Or install manually:
```bash
pip install -r requirements.txt  # or below manually
pip install torch torchvision  # use latest/CUDA version
pip install monai numpy pandas scikit-learn shap pydicom SimpleITK rt-utils
pip install -e .  # Install package in editable mode
```

## What Training Does

**Input**:
- Merged metafile (patient_id, survival endpoints, clinical features, radiomics features, radiomics status)
- CV splits (fold assignments)

**Processing** (for each fold):
1. Load clinical features → ClinicalEncoder (one-hot categorical + z-scored numeric)
2. Load radiomics features (if `--use_radiomics`) → PCA reduction
3. Initialize SwinUNETRTokenMoEDiscrete model (CT/mask components skipped, using zeros)
4. Train with discrete-time survival loss + gates + SHAP validity checks
5. Save checkpoint per epoch

**Output**:
- Model checkpoints: `runs/opscc_stage2_os/fold_00/ckpt_best.pt`, etc.
- Metrics: C-index, IBS, time-dependent AUC per fold
- SHAP values (if analysis run)

## Expected Performance

With tabular data only (no imaging):
- **C-index** (concordance): 0.60-0.70 typically (depends on feature quality)
- **Events**: 215/519 for OS (41.4%), 120/519 for DSS (23.1%)
- **Training time**: 2-5 hours on V100/A100 GPU (50 epochs, batch size 16, 4 folds)

When DICOM imaging data becomes available:
- Add Stage 1 segmentation pretraining
- Feed image features to Stage 2 → typically +5-15% C-index improvement

## File Locations

| Component | Path |
|-----------|------|
| Source code | `TriFuseSurv_package/src/trifusesurv/` |
| Python package | `TriFuseSurv_package/` (installable) |
| Shell scripts | `TriFuseSurv_package/scripts/` |
| Documentation | `TriFuseSurv_package/docs/` |
| Input CSVs | `*.csv` at the workspace root |
| Outputs | `runs/` under the workspace root |

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
→ Install dependencies: `./scripts/install_env.sh && source .venv/bin/activate`

### "No radiomics features found"
→ The radiomics CSV was too large to parse? Check `preparation_summary.json` for actual counts.

### "Clinical features have NaN values"
→ Some patients are missing clinical data. The encoder will one-hot a special "missing" category.

### GPU out of memory
→ Reduce `--batch_size` (default 16 → try 8 or 4)

### Training not converging
→ Check learning rates, try `--lr_head 5e-5` or reduce `--surv_dropout_p` from 0.40

## Next: Adding DICOM Imaging

When you have DICOM files in `OPSCC/` at the workspace root:

1. Run preprocessing:
   ```bash
   PYTHONPATH=TriFuseSurv_package/src python3 -m trifusesurv.preprocessing.export_swinunetr \
     --root OPSCC \
     --surv_csv opscc_survival_time_event.csv \
     --out_root OPSCC_preprocessed_128 \
     --spacing 0.5 0.5 1 --size 128 128 128
   ```

2. Modify prepare_opscc_tabular to inject CT/mask paths

3. Run Stage 1 segmentation training

4. Full multimodal Stage 2 training

Until then, tabular-only pipeline is fully functional and ready to use!

## Documentation

Full details in:
- `TriFuseSurv_package/docs/data_integration_guide.md` - comprehensive guide
- `TriFuseSurv_package/docs/commands.md` - CLI reference
- `TriFuseSurv_package/README.md` - package overview
