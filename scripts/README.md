# Run Scripts

These scripts are adapted from the original workflow and run the packaged code from this checkout via `PYTHONPATH=src`.

Organized stage directories:

- `build_zip_package.sh`
- `install_env.sh`
- `config/pipeline_2xh100_test.env`
- `preprocessing/export_swinunetr.sh`
- `preprocessing/make_cv_splits.sh`
- `stage1/pretrain_pt.sh`
- `stage1/pretrain_ln.sh`
- `stage2/train_survival.sh`
- `stage2/train_survival_lora.sh`
- `stage2/export_shap_grouped.sh`
- `stage2/export_shap_grouped_v2.sh`

Top-level wrapper scripts are still present for backward compatibility:

- `run_two_stage_pipeline_2xh100.sh`
- `run_preprocess_export_swinunetr.sh`
- `run_make_cv_splits.sh`
- `run_stage1_pretrain_pt.sh`
- `run_stage1_pretrain_ln.sh`
- `run_stage2_survival.sh`
- `run_stage2_survival_lora.sh`
- `run_stage2_shap_grouped.sh`
- `run_stage2_shap_grouped_v2.sh`

Recommended setup:

- `./scripts/install_env.sh`
- `source .venv/bin/activate`
- `./scripts/run_two_stage_pipeline_2xh100.sh`
- `./scripts/build_zip_package.sh`

Each script can be customized by overriding environment variables such as `META_CSV`, `SPLITS_DIR`, `PT_CKPT`, `LN_CKPT`, `OUT_DIR`, `EXP_NAME`, `CUDA_DEVICE`, and `DEVICE`.
The preprocessing scripts support `DICOM_ROOT`, `SURV_CSV`, `OUT_ROOT`, `OUT_CSV`, `MARGIN_MM`, `HU_MIN`, `HU_MAX`, `QC_REPORT`, `QC_POLICY`, `QC_DROP_AIR_GT`, `CV_FOLDS`, `VAL_FRAC`, and `SPLIT_SEED`.
For the grouped SHAP scripts, you can also set `CKPT_BASE_DIR` or individual `CKPT_FOLD_0` through `CKPT_FOLD_3` paths.
