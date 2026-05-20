#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$PACKAGE_DIR/.." && pwd)"
cd "$WORKSPACE_ROOT"

export PYTHONPATH="$PACKAGE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"
source "$PACKAGE_DIR/scripts/lib/gpu_utils.sh"
tf_require_python_modules numpy pandas SimpleITK torch monai sklearn pydicom

META_CSV="${META_CSV:-OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv}"
RADIOMICS_SOURCE="${RADIOMICS_SOURCE:-cohort_radiomics_patient_wide.csv}"
IMG_SIZE="${IMG_SIZE:-128 256 256}"
ENDPOINT="${ENDPOINT:-OS}"
ENDPOINT_LC="$(printf '%s' "$ENDPOINT" | tr '[:upper:]' '[:lower:]')"
SPLITS_DIR="${SPLITS_DIR:-runs/opscc_splits_${ENDPOINT_LC}_seed1}"
OUT_DIR="${OUT_DIR:-runs/contour_aware_survival_${ENDPOINT_LC}}"
EXP_NAME="${EXP_NAME:-cv4_contour_aware_${ENDPOINT_LC}_fold03}"
CUDA_DEVICE="${CUDA_DEVICE:-auto}"
REQUESTED_DEVICE="${DEVICE:-cuda:0}"
DEBUG_FOLD="${DEBUG_FOLD:-3}"
WORKERS="${WORKERS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
CACHE_VOLUMES="${CACHE_VOLUMES:-1}"
VOLUME_CACHE_SIZE="${VOLUME_CACHE_SIZE:-12}"
ROI_FOCUS_EVERY_BATCHES="${ROI_FOCUS_EVERY_BATCHES:-10}"
CONTOUR_WARMSTART_CKPT="${CONTOUR_WARMSTART_CKPT:-${SHARED_SEG_PRETRAIN_CKPT:-}}"
CONTOUR_WARMSTART_DIR="${CONTOUR_WARMSTART_DIR:-${SHARED_SEG_PRETRAIN_DIR:-}}"
CONTOUR_WARMSTART_NAME="${CONTOUR_WARMSTART_NAME:-${SHARED_SEG_PRETRAIN_NAME:-best.pt}}"

if [[ "$REQUESTED_DEVICE" == "cpu" ]]; then
  JOB_DEVICE="cpu"
else
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
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
  tf_require_torch_cuda

  if [[ -n "$REQUESTED_DEVICE" && "$REQUESTED_DEVICE" != "cuda" && "$REQUESTED_DEVICE" != "cuda:0" ]]; then
    echo "[warn] overriding DEVICE=$REQUESTED_DEVICE to cuda:0 inside CUDA_VISIBLE_DEVICES=$CUDA_DEVICE for single-GPU isolation"
  fi
  JOB_DEVICE="cuda:0"
fi
LOG_EVERY_BATCHES="${LOG_EVERY_BATCHES:-50}"
RESUME="${RESUME:-0}"
ALLOW_PARTIAL_RESUME="${ALLOW_PARTIAL_RESUME:-0}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-0}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-8}"
USE_CHECKPOINT="${USE_CHECKPOINT:-1}"
NONFINITE_CHECK_EVERY_BATCHES="${NONFINITE_CHECK_EVERY_BATCHES:-10}"
ROI_FOCUS_WARMUP_EPOCHS="${ROI_FOCUS_WARMUP_EPOCHS:-10}"
ROI_FOCUS_WARMUP_SURVIVAL_WEIGHT="${ROI_FOCUS_WARMUP_SURVIVAL_WEIGHT:-0.2}"
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
SWA_START_EPOCH="${SWA_START_EPOCH:-60}"
LR_BACKBONE="${LR_BACKBONE:-3e-5}"
WD_BACKBONE="${WD_BACKBONE:-1e-4}"
LR_HEAD="${LR_HEAD:-3e-4}"
LR_CLIN="${LR_CLIN:-1e-4}"
LR_RAD="${LR_RAD:-1e-4}"
WD_RAD="${WD_RAD:-2e-3}"
TIME_BIN_WIDTH_DAYS="${TIME_BIN_WIDTH_DAYS:-120.0}"
REPORT_METRIC="${REPORT_METRIC:-composite}"
EXPORT_POLICY="${EXPORT_POLICY:-best_val}"
FEATURE_SIZE="${FEATURE_SIZE:-96}"
DEPTHS="${DEPTHS:-2 2 18 2}"
NUM_HEADS="${NUM_HEADS:-3 6 12 24}"
FUSED_DIM="${FUSED_DIM:-512}"
RADIOMICS_PCA_TOTAL_COMPONENTS="${RADIOMICS_PCA_TOTAL_COMPONENTS:-100}"
IMG_TOKEN_DIM="${IMG_TOKEN_DIM:-768}"
TOKEN_MLP_HIDDEN_DIM="${TOKEN_MLP_HIDDEN_DIM:-1536}"
IMG_PROJ_HIDDEN_DIM="${IMG_PROJ_HIDDEN_DIM:-1024}"
IMG_TOK_FFN_HIDDEN_DIM="${IMG_TOK_FFN_HIDDEN_DIM:-1024}"
IMG_POST_HIDDEN_DIM="${IMG_POST_HIDDEN_DIM:-1024}"
IMG_ATTN_HEADS="${IMG_ATTN_HEADS:-4}"
GATE_HIDDEN_DIM="${GATE_HIDDEN_DIM:-512}"
RAD_HIDDEN_DIM="${RAD_HIDDEN_DIM:-1024}"
PRIMARY_SURV_LOSS_WEIGHT="${PRIMARY_SURV_LOSS_WEIGHT:-1.0}"
AUX_SURV_LOSS_WEIGHT="${AUX_SURV_LOSS_WEIGHT:-0.2}"
HAZARD_SMOOTH_LAMBDA="${HAZARD_SMOOTH_LAMBDA:-0.003}"
EMA_DECAY="${EMA_DECAY:-0.9995}"
PT_SHELL_RADIUS="${PT_SHELL_RADIUS:-5}"
LN_SHELL_RADIUS="${LN_SHELL_RADIUS:-5}"
PT_SHELL_THICKNESS_MM="${PT_SHELL_THICKNESS_MM:-10.0}"
LN_SHELL_THICKNESS_MM="${LN_SHELL_THICKNESS_MM:-0.0}"
SHELL_BODY_FROM_CT="${SHELL_BODY_FROM_CT:-1}"
BODY_CT_THR="${BODY_CT_THR:-0.02}"
BODY_CT_THR_HU="${BODY_CT_THR_HU:--500.0}"
BODY_CLOSE_R="${BODY_CLOSE_R:-2}"
BODY_MAX_FRAC="${BODY_MAX_FRAC:-0.995}"
SYNC_SANITIZE_CHECKS="${SYNC_SANITIZE_CHECKS:-0}"
MODALITY_DROPOUT_CLIN_P="${MODALITY_DROPOUT_CLIN_P:-0.10}"
MODALITY_DROPOUT_RAD_P="${MODALITY_DROPOUT_RAD_P:-0.10}"
RAD_PROJ_DROPOUT_P="${RAD_PROJ_DROPOUT_P:-0.15}"
PROJ_DROPOUT_P="${PROJ_DROPOUT_P:-0.20}"
EXPERT_DROPOUT_P="${EXPERT_DROPOUT_P:-0.15}"
TOKEN_MLP_DROPOUT="${TOKEN_MLP_DROPOUT:-0.35}"
TOKEN_DROPOUT="${TOKEN_DROPOUT:-0.05}"
ATTN_DROPOUT_P="${ATTN_DROPOUT_P:-0.10}"
V2_MODEL_DIM="${V2_MODEL_DIM:-256}"
V2_NUM_HEADS="${V2_NUM_HEADS:-8}"
V2_TRANSFORMER_LAYERS="${V2_TRANSFORMER_LAYERS:-2}"
V2_RADIOMICS_PCS_PER_GROUP="${V2_RADIOMICS_PCS_PER_GROUP:-16}"
V2_DROPOUT="${V2_DROPOUT:-0.05}"
V2_IMAGE_HABITAT_DROPOUT_P="${V2_IMAGE_HABITAT_DROPOUT_P:-0.05}"
V2_NODE_DROPOUT_P="${V2_NODE_DROPOUT_P:-0.10}"
V2_TOPOLOGY_DROPOUT_P="${V2_TOPOLOGY_DROPOUT_P:-0.10}"
V2_DROPOUT_RAMP_EPOCHS="${V2_DROPOUT_RAMP_EPOCHS:-12}"

resume_args=(--no_resume)
if [[ "$RESUME" == "1" || "$RESUME" == "true" || "$RESUME" == "yes" ]]; then
  resume_args=(--resume)
fi

partial_resume_args=()
if [[ "$ALLOW_PARTIAL_RESUME" == "1" || "$ALLOW_PARTIAL_RESUME" == "true" || "$ALLOW_PARTIAL_RESUME" == "yes" ]]; then
  partial_resume_args=(--allow_partial_resume)
fi

checkpoint_args=(--use_checkpoint)
if [[ "$USE_CHECKPOINT" == "0" || "$USE_CHECKPOINT" == "false" || "$USE_CHECKPOINT" == "no" ]]; then
  checkpoint_args=(--no_use_checkpoint)
fi

shell_body_args=()
if [[ "$SHELL_BODY_FROM_CT" == "1" || "$SHELL_BODY_FROM_CT" == "true" || "$SHELL_BODY_FROM_CT" == "yes" ]]; then
  shell_body_args=(
    --shell_body_from_ct
    --body_ct_thr "$BODY_CT_THR"
    --body_ct_thr_hu "$BODY_CT_THR_HU"
    --body_close_r "$BODY_CLOSE_R"
    --body_max_frac "$BODY_MAX_FRAC"
  )
fi

sanitize_args=()
if [[ "$SYNC_SANITIZE_CHECKS" == "1" || "$SYNC_SANITIZE_CHECKS" == "true" || "$SYNC_SANITIZE_CHECKS" == "yes" ]]; then
  sanitize_args=(--sync_sanitize_checks)
fi

cache_args=(--no_cache_volumes)
if [[ "$CACHE_VOLUMES" == "1" || "$CACHE_VOLUMES" == "true" || "$CACHE_VOLUMES" == "yes" ]]; then
  cache_args=(--cache_volumes)
fi

TRAIN_ARG_SOURCE="${TRAIN_ARG_SOURCE:-$PACKAGE_DIR/src/trifusesurv2/multimodal_survival/train.py}"
train_supports_arg() {
  local arg_name="$1"
  [[ -f "$TRAIN_ARG_SOURCE" ]] && grep -q -- "$arg_name" "$TRAIN_ARG_SOURCE"
}

eval_batch_args=()
if train_supports_arg "--eval_batch_size"; then
  eval_batch_args=(--eval_batch_size "$EVAL_BATCH_SIZE")
else
  echo "[warn] train.py does not support --eval_batch_size; skipping that fast-path arg. Check package/source version if this was unexpected." >&2
fi

nonfinite_check_args=()
if train_supports_arg "--nonfinite_check_every_batches"; then
  nonfinite_check_args=(--nonfinite_check_every_batches "$NONFINITE_CHECK_EVERY_BATCHES")
else
  echo "[warn] train.py does not support --nonfinite_check_every_batches; using its default strict loss check." >&2
fi

if (( ${#sanitize_args[@]} > 0 )) && ! train_supports_arg "--sync_sanitize_checks"; then
  echo "[warn] train.py does not support --sync_sanitize_checks; skipping that debug arg." >&2
  sanitize_args=()
fi

read -r -a depths_args <<< "$DEPTHS"
read -r -a num_heads_args <<< "$NUM_HEADS"
read -r -a img_size_args <<< "$IMG_SIZE"
if (( ${#img_size_args[@]} != 3 )); then
  echo "[error] IMG_SIZE must contain exactly three integers in D H W order, got: $IMG_SIZE" >&2
  exit 1
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

echo "[train] resume=$RESUME allow_partial_resume=$ALLOW_PARTIAL_RESUME epochs=$EPOCHS roi_focus_warmup_epochs=$ROI_FOCUS_WARMUP_EPOCHS survival_warmup_weight=$ROI_FOCUS_WARMUP_SURVIVAL_WEIGHT swa_start=$SWA_START_EPOCH"
echo "[train] survival_use_gt_masks=$SURVIVAL_USE_GT_MASKS mask_guidance_alpha=$MASK_GUIDANCE_ALPHA teacher_force_epochs=$TEACHER_FORCE_EPOCHS"
echo "[train] loc_feature_from_end=$LOC_FEATURE_FROM_END loc_loss_pt=$LOC_LOSS_PT_LAMBDA loc_loss_ln=$LOC_LOSS_LN_LAMBDA mask_support=$MASK_SUPPORT_LAMBDA mask_focus=$MASK_FOCUS_LAMBDA balanced_bce=1"
echo "[train] batch_size=$BATCH_SIZE eval_batch_size=$EVAL_BATCH_SIZE grad_accum=$GRAD_ACCUMULATION_STEPS use_checkpoint=$USE_CHECKPOINT feature_size=$FEATURE_SIZE depths=($DEPTHS) heads=($NUM_HEADS)"
echo "[train] fused_dim=$FUSED_DIM img_token_dim=$IMG_TOKEN_DIM v2_dim=$V2_MODEL_DIM v2_layers=$V2_TRANSFORMER_LAYERS v2_heads=$V2_NUM_HEADS"
echo "[train] lr_backbone=$LR_BACKBONE lr_head=$LR_HEAD lr_clin=$LR_CLIN lr_rad=$LR_RAD aux_w=$AUX_SURV_LOSS_WEIGHT hazard_smooth=$HAZARD_SMOOTH_LAMBDA"
echo "[train] time_bin_width_days=$TIME_BIN_WIDTH_DAYS report_metric=$REPORT_METRIC export_policy=$EXPORT_POLICY pt_shell_mm=$PT_SHELL_THICKNESS_MM pt_shell_radius_fallback=$PT_SHELL_RADIUS ln_shell_mm=$LN_SHELL_THICKNESS_MM ln_shell_radius=$LN_SHELL_RADIUS img_size=($IMG_SIZE)"
echo "[train] python_bin=$PYTHON_BIN workers=$WORKERS eval_workers=$EVAL_WORKERS prefetch_factor=$PREFETCH_FACTOR cache_volumes=$CACHE_VOLUMES volume_cache_size=$VOLUME_CACHE_SIZE roi_focus_every_batches=$ROI_FOCUS_EVERY_BATCHES nonfinite_check_every_batches=$NONFINITE_CHECK_EVERY_BATCHES shell_body_from_ct=$SHELL_BODY_FROM_CT body_ct_thr=$BODY_CT_THR sync_sanitize_checks=$SYNC_SANITIZE_CHECKS omp=$OMP_NUM_THREADS mkl=$MKL_NUM_THREADS itk=$ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS tf32_override=${NVIDIA_TF32_OVERRIDE:-<unset>}"

PYTHONUNBUFFERED=1 \
"$PYTHON_BIN" -u -m trifusesurv2.multimodal_survival.train \
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
  --img_size "${img_size_args[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  "${eval_batch_args[@]}" \
  --grad_accumulation_steps "$GRAD_ACCUMULATION_STEPS" \
  --workers "$WORKERS" \
  --eval_workers "$EVAL_WORKERS" \
  --prefetch_factor "$PREFETCH_FACTOR" \
  "${cache_args[@]}" \
  --volume_cache_size "$VOLUME_CACHE_SIZE" \
  --roi_focus_every_batches "$ROI_FOCUS_EVERY_BATCHES" \
  "${nonfinite_check_args[@]}" \
  --log_every_batches "$LOG_EVERY_BATCHES" \
  "${resume_args[@]}" \
  "${partial_resume_args[@]}" \
  --amp \
  --allow_tf32 \
  --matmul_precision high \
  "${checkpoint_args[@]}" \
  --device "$JOB_DEVICE" \
  --use_radiomics \
  --radiomics_root "$RADIOMICS_SOURCE" \
  --use_ema \
  --use_swa \
  --export_extra_risks \
  --lr_backbone "$LR_BACKBONE" \
  --wd_backbone "$WD_BACKBONE" \
  --lr_head "$LR_HEAD" \
  --lr_clin "$LR_CLIN" \
  --lr_rad "$LR_RAD" \
  --wd_rad "$WD_RAD" \
  --time_bin_width_days "$TIME_BIN_WIDTH_DAYS" \
  --report_metric "$REPORT_METRIC" \
  --export_policy "$EXPORT_POLICY" \
  --feature_size "$FEATURE_SIZE" \
  --depths "${depths_args[@]}" \
  --num_heads "${num_heads_args[@]}" \
  --fused_dim "$FUSED_DIM" \
  --modality_dropout_clin_p "$MODALITY_DROPOUT_CLIN_P" \
  --modality_dropout_rad_p "$MODALITY_DROPOUT_RAD_P" \
  --primary_surv_loss_weight "$PRIMARY_SURV_LOSS_WEIGHT" \
  --aux_surv_loss_weight "$AUX_SURV_LOSS_WEIGHT" \
  --hazard_smooth_lambda "$HAZARD_SMOOTH_LAMBDA" \
  --ema_decay "$EMA_DECAY" \
  --swa_start_epoch "$SWA_START_EPOCH" \
  --swa_update_freq_epochs 1 \
  --pt_shell_radius "$PT_SHELL_RADIUS" \
  --ln_shell_radius "$LN_SHELL_RADIUS" \
  --pt_shell_thickness_mm "$PT_SHELL_THICKNESS_MM" \
  --ln_shell_thickness_mm "$LN_SHELL_THICKNESS_MM" \
  --radiomics_pca_total_components "$RADIOMICS_PCA_TOTAL_COMPONENTS" \
  --img_token_dim "$IMG_TOKEN_DIM" \
  --token_mlp_hidden_dim "$TOKEN_MLP_HIDDEN_DIM" \
  --img_proj_hidden_dim "$IMG_PROJ_HIDDEN_DIM" \
  --img_tok_ffn_hidden_dim "$IMG_TOK_FFN_HIDDEN_DIM" \
  --img_post_hidden_dim "$IMG_POST_HIDDEN_DIM" \
  --img_attn_heads "$IMG_ATTN_HEADS" \
  --gate_hidden_dim "$GATE_HIDDEN_DIM" \
  --rad_hidden_dim "$RAD_HIDDEN_DIM" \
  --rad_proj_dropout_p "$RAD_PROJ_DROPOUT_P" \
  --proj_dropout_p "$PROJ_DROPOUT_P" \
  --expert_dropout_p "$EXPERT_DROPOUT_P" \
  --token_mlp_dropout "$TOKEN_MLP_DROPOUT" \
  --token_dropout "$TOKEN_DROPOUT" \
  --attn_dropout_p "$ATTN_DROPOUT_P" \
  --v2_model_dim "$V2_MODEL_DIM" \
  --v2_num_heads "$V2_NUM_HEADS" \
  --v2_transformer_layers "$V2_TRANSFORMER_LAYERS" \
  --v2_radiomics_pcs_per_group "$V2_RADIOMICS_PCS_PER_GROUP" \
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
  "${shell_body_args[@]}" \
  "${sanitize_args[@]}" \
  "${extra_args[@]}" \
  "$@"
