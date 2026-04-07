#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$PACKAGE_DIR/.." && pwd)"
cd "$WORKSPACE_ROOT"

export PYTHONPATH="$PACKAGE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
source "$PACKAGE_DIR/scripts/lib/gpu_utils.sh"

META_CSV="${META_CSV:-OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv}"
SPLITS_DIR="${SPLITS_DIR:-runs/opscc_splits_os_seed1}"
PT_CKPT="${PT_CKPT:-runs/seg_pt_h100_test/all/seg_best.pt}"
LN_CKPT="${LN_CKPT:-runs/seg_ln_h100_test/all/seg_best.pt}"
RADIOMICS_SOURCE="${RADIOMICS_SOURCE:-cohort_radiomics_patient_wide.csv}"
OUT_DIR="${OUT_DIR:-runs/moe_discrete_swinunetr}"
EXP_NAME="${EXP_NAME:-cv4_best_fold03}"
CUDA_DEVICE="${CUDA_DEVICE:-auto}"
DEVICE="${DEVICE:-cuda:0}"
DEBUG_FOLD="${DEBUG_FOLD:-3}"

if [[ "$CUDA_DEVICE" == "auto" || -z "$CUDA_DEVICE" ]]; then
  if ! CUDA_DEVICE="$(tf_first_gpu_id)"; then
    echo "[error] could not detect an available GPU for stage2 survival training." >&2
    exit 1
  fi
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
python3 -m trifusesurv.multimodal_survival.train \
  --meta_csv "$META_CSV" \
  --splits_dir "$SPLITS_DIR" \
  --cv_folds 4 \
  --debug_fold "$DEBUG_FOLD" \
  --strict_splits \
  --endpoint OS \
  --ct_col ct_out_path \
  --mask_pt_col mask_primary_out_path \
  --mask_ln_col mask_nodal_out_path \
  --seg_pretrain_pt_ckpt "$PT_CKPT" \
  --seg_pretrain_ln_ckpt "$LN_CKPT" \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --img_size 128 256 256 \
  --epochs 60 \
  --batch_size 1 \
  --workers 8 \
  --amp \
  --use_checkpoint \
  --device "$DEVICE" \
  --use_radiomics \
  --radiomics_root "$RADIOMICS_SOURCE" \
  --use_ema \
  --use_swa \
  --export_extra_risks \
  --lr_backbone 1e-3 \
  --wd_backbone 0 \
  --lr_head 1e-4 \
  --wd_rad 2e-3 \
  --modality_dropout_rad_p 0.20 \
  --ema_decay 0.9995 \
  --swa_start_epoch 10 \
  --swa_update_freq_epochs 1 \
  --pt_shell_radius 5 \
  --ln_shell_radius 5 \
  --radiomics_pca_total_components 100 \
  --img_token_dim 768 \
  --token_mlp_hidden_dim 1536 \
  --img_proj_hidden_dim 1024 \
  --img_tok_ffn_hidden_dim 1024 \
  --img_post_hidden_dim 1024 \
  --img_attn_heads 4 \
  --gate_hidden_dim 512 \
  --rad_hidden_dim 1024 \
  --rad_proj_dropout_p 0.30 \
  --proj_dropout_p 0.35 \
  --expert_dropout_p 0.15 \
  --token_mlp_dropout 0.55 \
  --token_dropout 0.10 \
  --attn_dropout_p 0.15 \
  --shell_body_from_ct \
  "$@"
