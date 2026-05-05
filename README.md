# TriFuseSurv2 Package

Unified operational package for habitat-aligned, node-aware OPSCC survival modeling.

## What this package contains

This package merges two codebases:

1. **v2 foundations** (schema, encoders, habitat model, node utilities) — from `TriFuseSurv2/`
2. **Operational pipeline** (backbone, training, evaluation, preprocessing) — from `TriFuseSurv_package/`

All modules live under the `trifusesurv2` namespace.

Preprocessing outputs and stage-2 metafiles are written with relative path
fields so the package remains relocatable across machines and workspaces.

## Package structure

```
src/trifusesurv2/
  schema.py                          # Shared constants: habitats, endpoints, clinical groups
  data/
    batch.py                         # HabitatBatch, TokenBlock, SurvivalTargets
  encoders/
    clinical.py                      # SemanticClinicalTokenEncoder (v2 grouped tokens)
    radiomics.py                     # HabitatRadiomicsTokenEncoder (v2 per-habitat PCA)
  models/
    habitat_survival.py              # HabitatAlignedSurvivalModel, PTNodeCrossAttention
    contour_habitat_survival.py      # ContourAwareHabitatSurvivalModel (backbone wrapper)
    survival_model.py                # SwinUNETRTokenMoEDiscrete (v1 late-fusion model)
    swinunetr_shared_roi_token_backbone.py  # ContourAwareROITokenBackbone
    swinunetr_backbone_utils.py      # Pretrained weight loading
    lora.py                          # LoRA adaptation
  multimodal_survival/
    train.py                         # Training pipeline
    evaluate_oof_cindex.py           # Out-of-fold evaluation
  preprocessing/
    nodes.py                         # Node-instance extraction, topology, serialization
    export_swinunetr.py              # DICOM → NIfTI preprocessing
    make_cv_splits.py                # Cross-validation split generation
    prepare_opscc_tabular.py         # Tabular data preparation
  utils/
    clinical.py                      # ClinicalEncoder (v1 flat encoding)
    radiomics.py                     # RadiomicsEncoder (v1 flat encoding)
    data.py                          # PreprocessedContourAwareDataset
    survival.py                      # Loss functions, metrics (c-index, IBS, AUC, DCA)
```

## Installation

First-time setup:

```bash
bash scripts/install_env.sh
```

By default this now keeps the environment local to the package:
- virtualenv in `./.venv`
- pip cache in `./.install_env/pip-cache`
- temp/build files in `./.install_env/tmp`

Fast package refresh after code changes:

```bash
source .venv/bin/activate
python -m pip install -e . --no-deps
```

Avoid using `python -m pip install --upgrade -e .` for routine refreshes,
because pip may spend a long time re-resolving and upgrading heavyweight
dependencies like `torch` and `monai`.

If you need a manual full install inside an already-active environment:

```bash
pip install -e .
```

## CLI entry points

```bash
trifusesurv2-train                   # Joint contour-aware survival training
trifusesurv2-evaluate-oof            # Out-of-fold c-index evaluation
trifusesurv2-gradcam-v208            # v2 habitat Grad-CAM/attention/ablation export
trifusesurv2-preprocess-export       # DICOM to NIfTI preprocessing
trifusesurv2-make-cv-splits          # Cross-validation split generation
trifusesurv2-prepare-opscc-tabular   # Tabular data preparation
```

`trifusesurv2-gradcam-v208` defaults to `--checkpoint last --weights ema`,
matching the default `test_risks_ema.csv` export from training. Use
`--checkpoint best --weights best` when explaining `test_risks_best.csv`.

## Operational search scripts

This package is the runnable home for search wrappers and fixed-setting CV
launchers. The current scripts live in `scripts/`:

- `scripts/run_contour_aware_cindex_search_75gb_30ep.sh`
  - broad 75 GB / 30 epoch multitask search
- `scripts/run_contour_aware_cindex_search_75gb_tf24_followup.sh`
  - tighter search around the strongest `tf24` window
- `scripts/run_massive_testing_20hr_75gb_30ep.sh`
  - larger around-the-winner search, sized for roughly 20 hours on 4 GPUs
- `scripts/run_v75_tri_h1095_tf24_4fold.sh`
  - fixed 4-fold rerun of the strongest anchor setting
- `scripts/run_optimal_setting_search_75gb_30ep.sh`
  - v2 habitat-aligned two-phase launcher with `quick`, `balanced`, and `broad`
    profiles; use `DRY_RUN=1` to print the candidate settings without training
- `scripts/run_dual_h100_140gb_best_perf_20hr.sh`
  - dual-H100 best-performance search; resumable by default with full checkpoints
- `scripts/run_dual_h100_140gb_diverse_search_20hr.sh`
  - dual-H100 diverse search with deliberately different settings around the current winner

For the dual-H100 search, interrupted runs now restart from `last.pt` by default.
To force a clean rerun instead of resuming:

```bash
RESUME=0 SKIP_FINISHED=0 bash scripts/run_dual_h100_140gb_best_perf_20hr.sh
```

## Two model paths

### v2 (habitat-aligned fusion, default)
```bash
PYTHONPATH=src python3 -m trifusesurv2.multimodal_survival.train \
  --meta_csv ... --splits_csv ...
```

Uses `ContourAwareHabitatSurvivalModel` with `SemanticClinicalTokenEncoder`
and `HabitatRadiomicsTokenEncoder`. By default, v2 radiomics expects the
patient-wide habitat CSV `cohort_radiomics_patient_wide.csv`; pass
`--no_radiomics` to train without radiomics, or `--radiomics_root ...` to use a
different wide CSV. Node/topology projections are enabled by default and use
`--node_topology_dir` when JSON summaries are available. Training also applies a
ramped structured v2 dropout curriculum over image habitats, clinical groups,
radiomics habitats, node tokens, and topology tokens; tune it with
`--v2_*_dropout_p` and `--v2_dropout_ramp_epochs`.

### v1 (late-fusion legacy)
```bash
PYTHONPATH=src python3 -m trifusesurv2.multimodal_survival.train \
  --meta_csv ... --splits_csv ... --model_version v1
```

## Running tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
