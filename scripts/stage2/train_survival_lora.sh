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
OUT_DIR="${OUT_DIR:-runs/moe_runs}"
EXP_NAME="${EXP_NAME:-cv4_fold03_lora96_presenceFix_ep20}"
CUDA_DEVICE="${CUDA_DEVICE:-auto}"
DEVICE="${DEVICE:-cuda:0}"
DEBUG_FOLD="${DEBUG_FOLD:-3}"

mkdir -p "$OUT_DIR"

if [[ "$CUDA_DEVICE" == "auto" || -z "$CUDA_DEVICE" ]]; then
  if ! CUDA_DEVICE="$(tf_first_gpu_id)"; then
    echo "[error] could not detect an available GPU for stage2 LoRA training." >&2
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
  --epochs 20 \
  --batch_size 1 \
  --workers 16 \
  --amp \
  --use_checkpoint \
  --device "$DEVICE" \
  --no_resume \
  --use_radiomics \
  --radiomics_root "$RADIOMICS_SOURCE" \
  --use_ema \
  --use_swa \
  --export_extra_risks \
  --clinical_cols NSTAGE AGE SEX T N M KFCF SMOKE ALCOHOL HPV \
  --lr_backbone 8e-4 \
  --wd_backbone 0 \
  --lr_lora 3e-4 \
  --wd_lora 0 \
  --lr_head 8e-5 \
  --wd_head 1e-3 \
  --wd_clin 5e-4 \
  --wd_rad 2e-3 \
  --pt_shell_radius 5 \
  --ln_shell_radius 5 \
  --radiomics_pca_total_components 100 \
  --shell_body_from_ct \
  --use_lora \
  --lora_scope both \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_targets qkv proj fc1 fc2 linear1 linear2 \
  --lora_min_replacements 1 \
  --train_mask_patch_embed \
  --nan_guard \
  --mask_interp trilinear \
  --min_roi_frac 0 \
  --min_roi_voxels_deep 0 \
  --token_dropout 0.0 \
  --expert_dropout_p 0.0 \
  --modality_dropout_clin_p 0.0 \
  --modality_dropout_rad_p 0.0 \
  --clinical_noise_std 0.0 \
  --radiomics_noise_std 0.0 \
  --gate_entropy_lambda 0.002 \
  --gate_loadbal_lambda 0.002 \
  --hazard_smooth_lambda 0.005 \
  --logit_l2_lambda 0.0 \
  "$@"
