#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

META_CSV="${META_CSV:-OPSCC_preprocessed_128/cohort_preprocessed_with_clin.csv}"
SPLITS_DIR="${SPLITS_DIR:-runs/opscc_splits_os_seed1}"
OUT_DIR="${OUT_DIR:-runs/shap_oof_lora96_v2}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DEVICE="${DEVICE:-cuda:0}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-runs/moe_runs}"

CKPT_FOLD_0="${CKPT_FOLD_0:-$CKPT_BASE_DIR/cv4_fold00_lora96_perfTune_v4_rad70_gate30_swa45/fold_00/last.pt}"
CKPT_FOLD_1="${CKPT_FOLD_1:-$CKPT_BASE_DIR/cv4_fold01_lora96_perfTune_v4_rad70_gate30_swa45/fold_01/last.pt}"
CKPT_FOLD_2="${CKPT_FOLD_2:-$CKPT_BASE_DIR/cv4_fold02_lora96_perfTune_v4_rad70_gate30_swa45/fold_02/last.pt}"
CKPT_FOLD_3="${CKPT_FOLD_3:-$CKPT_BASE_DIR/cv4_fold03_lora96_perfTune_v4_rad70_gate30_swa45/fold_03/last.pt}"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
python3 -m trifusesurv.multimodal_survival.export_grouped_shap \
  --meta_csv "$META_CSV" \
  --splits_dir "$SPLITS_DIR" \
  --cv_folds 4 \
  --strict_splits \
  --endpoint OS \
  --ct_col ct_out_path \
  --mask_pt_col mask_primary_out_path \
  --mask_ln_col mask_nodal_out_path \
  --ckpt_fold "0:$CKPT_FOLD_0" \
  --ckpt_fold "1:$CKPT_FOLD_1" \
  --ckpt_fold "2:$CKPT_FOLD_2" \
  --ckpt_fold "3:$CKPT_FOLD_3" \
  --weights ema \
  --lora_alpha 32 \
  --lora_dropout 0.0 \
  --lora_scope both \
  --img_size 128 256 256 \
  --feature_size 96 \
  --depths 2 2 18 2 \
  --num_heads 3 6 12 24 \
  --drop_rate 0.0 \
  --attn_drop_rate 0.0 \
  --dropout_path_rate 0.0 \
  --use_checkpoint \
  --img_token_dim 768 \
  --token_mlp_hidden_dim 1536 \
  --pt_shell_radius 5 \
  --ln_shell_radius 5 \
  --shell_body_from_ct \
  --time_bin_width_days 180 \
  --risk_horizon_days 365 \
  --use_radiomics \
  --radiomics_pca_total_components 100 \
  --n_perm 32 \
  --bg_size 64 \
  --workers 8 \
  --device "$DEVICE" \
  --amp \
  --out_dir "$OUT_DIR"
