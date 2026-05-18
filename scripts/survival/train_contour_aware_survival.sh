#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$PACKAGE_DIR/.." && pwd)"
cd "$WORKSPACE_ROOT"

export PYTHONPATH="$PACKAGE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
source "$PACKAGE_DIR/scripts/lib/gpu_utils.sh"
tf_require_python_modules numpy pandas SimpleITK torch monai sklearn pydicom rt_utils cv2

META_CSV="${META_CSV:-OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv}"
RADIOMICS_SOURCE="${RADIOMICS_SOURCE:-cohort_radiomics_patient_wide.csv}"
ENDPOINT="${ENDPOINT:-OS}"
ENDPOINT_LC="$(printf '%s' "$ENDPOINT" | tr '[:upper:]' '[:lower:]')"
SPLITS_DIR="${SPLITS_DIR:-runs/opscc_splits_${ENDPOINT_LC}_seed1}"
OUT_DIR="${OUT_DIR:-runs/contour_aware_survival_${ENDPOINT_LC}}"
EXP_NAME="${EXP_NAME:-cv4_contour_aware_${ENDPOINT_LC}_fold03}"
CUDA_DEVICE="${CUDA_DEVICE:-auto}"
REQUESTED_DEVICE="${DEVICE:-cuda:0}"
DEBUG_FOLD="${DEBUG_FOLD:-3}"
WORKERS="${WORKERS:-2}"
CONTOUR_WARMSTART_CKPT="${CONTOUR_WARMSTART_CKPT:-${SHARED_SEG_PRETRAIN_CKPT:-}}"
CONTOUR_WARMSTART_DIR="${CONTOUR_WARMSTART_DIR:-${SHARED_SEG_PRETRAIN_DIR:-}}"
CONTOUR_WARMSTART_NAME="${CONTOUR_WARMSTART_NAME:-${SHARED_SEG_PRETRAIN_NAME:-best.pt}}"

if [[ "$CUDA_DEVICE" == "auto" || -z "$CUDA_DEVICE" ]]; then
  if ! CUDA_DEVICE="$(tf_detect_gpu_ids_by_free_mem "${MIN_FREE_GPU_MB:-1}" | head -n 1)"; then
    CUDA_DEVICE=""
  fi
  if [[ -z "$CUDA_DEVICE" ]]; then
    if ! CUDA_DEVICE="$(tf_first_gpu_id)"; then
      echo "[error] could not detect an available GPU for contour-aware survival training." >&2
      exit 1
    fi
  fi
fi

if [[ -z "$CUDA_DEVICE" ]]; then
    echo "[error] could not detect an available GPU for contour-aware survival training." >&2
    exit 1
fi

if [[ -n "$REQUESTED_DEVICE" && "$REQUESTED_DEVICE" != "cuda" && "$REQUESTED_DEVICE" != "cuda:0" ]]; then
  echo "[warn] overriding DEVICE=$REQUESTED_DEVICE to cuda:0 inside CUDA_VISIBLE_DEVICES=$CUDA_DEVICE for single-GPU isolation"
fi
JOB_DEVICE="cuda:0"
LOG_EVERY_BATCHES="${LOG_EVERY_BATCHES:-50}"
RESUME="${RESUME:-0}"
EPOCHS="${EPOCHS:-60}"
ROI_FOCUS_WARMUP_EPOCHS="${ROI_FOCUS_WARMUP_EPOCHS:-30}"
ROI_FOCUS_WARMUP_SURVIVAL_WEIGHT="${ROI_FOCUS_WARMUP_SURVIVAL_WEIGHT:-0.0}"
SURVIVAL_USE_GT_MASKS="${SURVIVAL_USE_GT_MASKS:-1}"
MASK_GUIDANCE_ALPHA="${MASK_GUIDANCE_ALPHA:-1.0}"
TEACHER_FORCE_EPOCHS="${TEACHER_FORCE_EPOCHS:-0}"
TEACHER_FORCE_START="${TEACHER_FORCE_START:-1.0}"
TEACHER_FORCE_END="${TEACHER_FORCE_END:-1.0}"
LOC_LOSS_PT_LAMBDA="${LOC_LOSS_PT_LAMBDA:-4.0}"
LOC_LOSS_LN_LAMBDA="${LOC_LOSS_LN_LAMBDA:-4.0}"
LOC_PRESENCE_LAMBDA="${LOC_PRESENCE_LAMBDA:-0.20}"
LOC_BCE_WEIGHT="${LOC_BCE_WEIGHT:-1.0}"
LOC_DICE_WEIGHT="${LOC_DICE_WEIGHT:-1.0}"
LOC_POS_WEIGHT_CAP="${LOC_POS_WEIGHT_CAP:-1000.0}"
MASK_SUPPORT_LAMBDA="${MASK_SUPPORT_LAMBDA:-2.0}"
MASK_SUPPORT_BCE_WEIGHT="${MASK_SUPPORT_BCE_WEIGHT:-1.0}"
MASK_SUPPORT_DICE_WEIGHT="${MASK_SUPPORT_DICE_WEIGHT:-1.0}"
MASK_SUPPORT_POS_WEIGHT_CAP="${MASK_SUPPORT_POS_WEIGHT_CAP:-1000.0}"
MASK_FOCUS_LAMBDA="${MASK_FOCUS_LAMBDA:-8.0}"
LOC_FEATURE_FROM_END="${LOC_FEATURE_FROM_END:-4}"
ROI_SUPPORT_THRESHOLD="${ROI_SUPPORT_THRESHOLD:-0.50}"
ROI_SUPPORT_FALLBACK_THRESHOLD="${ROI_SUPPORT_FALLBACK_THRESHOLD:-0.05}"
ROI_SUPPORT_FALLBACK_RELMAX="${ROI_SUPPORT_FALLBACK_RELMAX:-0.50}"
SWA_START_EPOCH="${SWA_START_EPOCH:-$((10#$ROI_FOCUS_WARMUP_EPOCHS + 5))}"
LR_BACKBONE="${LR_BACKBONE:-3e-4}"
WD_BACKBONE="${WD_BACKBONE:-1e-4}"
LR_HEAD="${LR_HEAD:-1e-4}"
WD_RAD="${WD_RAD:-2e-3}"
PRIMARY_SURV_LOSS_WEIGHT="${PRIMARY_SURV_LOSS_WEIGHT:-1.0}"
AUX_SURV_LOSS_WEIGHT="${AUX_SURV_LOSS_WEIGHT:-0.35}"
EMA_DECAY="${EMA_DECAY:-0.9995}"
PT_SHELL_RADIUS="${PT_SHELL_RADIUS:-5}"
LN_SHELL_RADIUS="${LN_SHELL_RADIUS:-5}"
MODALITY_DROPOUT_CLIN_P="${MODALITY_DROPOUT_CLIN_P:-0.20}"
MODALITY_DROPOUT_RAD_P="${MODALITY_DROPOUT_RAD_P:-0.20}"
RAD_PROJ_DROPOUT_P="${RAD_PROJ_DROPOUT_P:-0.30}"
PROJ_DROPOUT_P="${PROJ_DROPOUT_P:-0.35}"
EXPERT_DROPOUT_P="${EXPERT_DROPOUT_P:-0.15}"
TOKEN_MLP_DROPOUT="${TOKEN_MLP_DROPOUT:-0.55}"
TOKEN_DROPOUT="${TOKEN_DROPOUT:-0.10}"
ATTN_DROPOUT_P="${ATTN_DROPOUT_P:-0.15}"
V2_DROPOUT="${V2_DROPOUT:-0.10}"
V2_IMAGE_HABITAT_DROPOUT_P="${V2_IMAGE_HABITAT_DROPOUT_P:-0.05}"
V2_NODE_DROPOUT_P="${V2_NODE_DROPOUT_P:-0.10}"
V2_TOPOLOGY_DROPOUT_P="${V2_TOPOLOGY_DROPOUT_P:-0.10}"
V2_DROPOUT_RAMP_EPOCHS="${V2_DROPOUT_RAMP_EPOCHS:-12}"

resume_args=(--no_resume)
if [[ "$RESUME" == "1" || "$RESUME" == "true" || "$RESUME" == "yes" ]]; then
  resume_args=(--resume)
fi

extra_args=()
if [[ -n "$CONTOUR_WARMSTART_CKPT" ]]; then
  extra_args+=(--contour_warmstart_ckpt "$CONTOUR_WARMSTART_CKPT")
elif [[ -n "$CONTOUR_WARMSTART_DIR" ]]; then
  extra_args+=(--contour_warmstart_dir "$CONTOUR_WARMSTART_DIR" --contour_warmstart_name "$CONTOUR_WARMSTART_NAME")
else
  extra_args+=(--no_align_swin_cfg_from_contour_warmstart)
fi

survival_tf_args=(--mask_guidance_alpha "$MASK_GUIDANCE_ALPHA")
if [[ "$SURVIVAL_USE_GT_MASKS" == "1" || "$SURVIVAL_USE_GT_MASKS" == "true" || "$SURVIVAL_USE_GT_MASKS" == "yes" ]]; then
  survival_tf_args=(--survival_uses_teacher_forced_masks --mask_guidance_alpha "$MASK_GUIDANCE_ALPHA")
fi

echo "[train] resume=$RESUME epochs=$EPOCHS roi_focus_warmup_epochs=$ROI_FOCUS_WARMUP_EPOCHS survival_warmup_weight=$ROI_FOCUS_WARMUP_SURVIVAL_WEIGHT swa_start=$SWA_START_EPOCH"
echo "[train] survival_use_gt_masks=$SURVIVAL_USE_GT_MASKS mask_guidance_alpha=$MASK_GUIDANCE_ALPHA teacher_force_epochs=$TEACHER_FORCE_EPOCHS"
echo "[train] loc_feature_from_end=$LOC_FEATURE_FROM_END loc_loss_pt=$LOC_LOSS_PT_LAMBDA loc_loss_ln=$LOC_LOSS_LN_LAMBDA mask_support=$MASK_SUPPORT_LAMBDA mask_focus=$MASK_FOCUS_LAMBDA balanced_bce=1"
echo "[train] lr_backbone=$LR_BACKBONE lr_head=$LR_HEAD aux_w=$AUX_SURV_LOSS_WEIGHT pt_shell=$PT_SHELL_RADIUS ln_shell=$LN_SHELL_RADIUS"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
PYTHONUNBUFFERED=1 \
python3 -u -m trifusesurv2.multimodal_survival.train \
  --meta_csv "$META_CSV" \
  --splits_dir "$SPLITS_DIR" \
  --cv_folds 4 \
  --debug_fold "$DEBUG_FOLD" \
  --strict_splits \
  --endpoint "$ENDPOINT" \
  --ct_col ct_out_path \
  --mask_pt_col mask_primary_out_path \
  --mask_ln_col mask_nodal_out_path \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --img_size 128 256 256 \
  --epochs "$EPOCHS" \
  --batch_size 1 \
  --workers "$WORKERS" \
  --log_every_batches "$LOG_EVERY_BATCHES" \
  "${resume_args[@]}" \
  --amp \
  --use_checkpoint \
  --device "$JOB_DEVICE" \
  --use_radiomics \
  --radiomics_root "$RADIOMICS_SOURCE" \
  --use_ema \
  --use_swa \
  --export_extra_risks \
  --lr_backbone "$LR_BACKBONE" \
  --wd_backbone "$WD_BACKBONE" \
  --lr_head "$LR_HEAD" \
  --wd_rad "$WD_RAD" \
  --modality_dropout_clin_p "$MODALITY_DROPOUT_CLIN_P" \
  --modality_dropout_rad_p "$MODALITY_DROPOUT_RAD_P" \
  --primary_surv_loss_weight "$PRIMARY_SURV_LOSS_WEIGHT" \
  --aux_surv_loss_weight "$AUX_SURV_LOSS_WEIGHT" \
  --ema_decay "$EMA_DECAY" \
  --swa_start_epoch "$SWA_START_EPOCH" \
  --swa_update_freq_epochs 1 \
  --pt_shell_radius "$PT_SHELL_RADIUS" \
  --ln_shell_radius "$LN_SHELL_RADIUS" \
  --radiomics_pca_total_components 100 \
  --img_token_dim 768 \
  --token_mlp_hidden_dim 1536 \
  --img_proj_hidden_dim 1024 \
  --img_tok_ffn_hidden_dim 1024 \
  --img_post_hidden_dim 1024 \
  --img_attn_heads 4 \
  --gate_hidden_dim 512 \
  --rad_hidden_dim 1024 \
  --rad_proj_dropout_p "$RAD_PROJ_DROPOUT_P" \
  --proj_dropout_p "$PROJ_DROPOUT_P" \
  --expert_dropout_p "$EXPERT_DROPOUT_P" \
  --token_mlp_dropout "$TOKEN_MLP_DROPOUT" \
  --token_dropout "$TOKEN_DROPOUT" \
  --attn_dropout_p "$ATTN_DROPOUT_P" \
  --v2_dropout "$V2_DROPOUT" \
  --v2_image_habitat_dropout_p "$V2_IMAGE_HABITAT_DROPOUT_P" \
  --v2_node_dropout_p "$V2_NODE_DROPOUT_P" \
  --v2_topology_dropout_p "$V2_TOPOLOGY_DROPOUT_P" \
  --v2_dropout_ramp_epochs "$V2_DROPOUT_RAMP_EPOCHS" \
  --use_multiscale \
  --mask_interp trilinear \
  --loc_feature_from_end "$LOC_FEATURE_FROM_END" \
  --min_roi_voxels_deep 0 \
  --teacher_force_epochs "$TEACHER_FORCE_EPOCHS" \
  --teacher_force_start "$TEACHER_FORCE_START" \
  --teacher_force_end "$TEACHER_FORCE_END" \
  "${survival_tf_args[@]}" \
  --roi_focus_warmup_epochs "$ROI_FOCUS_WARMUP_EPOCHS" \
  --roi_focus_warmup_survival_weight "$ROI_FOCUS_WARMUP_SURVIVAL_WEIGHT" \
  --loc_loss_pt_lambda "$LOC_LOSS_PT_LAMBDA" \
  --loc_loss_ln_lambda "$LOC_LOSS_LN_LAMBDA" \
  --loc_presence_lambda "$LOC_PRESENCE_LAMBDA" \
  --loc_bce_weight "$LOC_BCE_WEIGHT" \
  --loc_dice_weight "$LOC_DICE_WEIGHT" \
  --loc_pos_weight_cap "$LOC_POS_WEIGHT_CAP" \
  --mask_support_lambda "$MASK_SUPPORT_LAMBDA" \
  --mask_support_bce_weight "$MASK_SUPPORT_BCE_WEIGHT" \
  --mask_support_dice_weight "$MASK_SUPPORT_DICE_WEIGHT" \
  --mask_support_pos_weight_cap "$MASK_SUPPORT_POS_WEIGHT_CAP" \
  --mask_focus_lambda "$MASK_FOCUS_LAMBDA" \
  --roi_support_threshold "$ROI_SUPPORT_THRESHOLD" \
  --roi_support_fallback_threshold "$ROI_SUPPORT_FALLBACK_THRESHOLD" \
  --roi_support_fallback_relmax "$ROI_SUPPORT_FALLBACK_RELMAX" \
  --shell_body_from_ct \
  "${extra_args[@]}" \
  "$@"
