#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$PACKAGE_DIR/.." && pwd)"
cd "$WORKSPACE_ROOT"

ENDPOINT="${ENDPOINT:-OS}"
ENDPOINT_LC="$(printf '%s' "$ENDPOINT" | tr '[:upper:]' '[:lower:]')"
DEBUG_FOLD="${DEBUG_FOLD:-3}"
fold_tag="$DEBUG_FOLD"
if [[ "$fold_tag" =~ ^[0-9]+$ ]]; then
  fold_tag="$(printf '%02d' "$((10#$fold_tag))")"
fi

GPU_IDS="${GPU_IDS:-0,1,2,3}"
gpu_ids_spaced="${GPU_IDS//,/ }"
read -r -a GPU_ARRAY <<< "$gpu_ids_spaced"
if (( ${#GPU_ARRAY[@]} == 0 )); then
  echo "[search][FAIL] GPU_IDS is empty. Example: GPU_IDS=0,1,2,3" >&2
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

OUT_ROOT="${OUT_ROOT:-runs/roi_constrained_h100_search_round3_${ENDPOINT_LC}_fold${fold_tag}}"
TRAIN_WRAPPER="${TRAIN_WRAPPER:-$PACKAGE_DIR/scripts/survival/train_with_roi_focus_watch.sh}"
EPOCHS="${EPOCHS:-80}"
IMG_SIZE="${IMG_SIZE:-128 256 256}"
WORKERS="${WORKERS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
CACHE_VOLUMES="${CACHE_VOLUMES:-1}"
VOLUME_CACHE_SIZE="${VOLUME_CACHE_SIZE:-12}"
ROI_FOCUS_EVERY_BATCHES="${ROI_FOCUS_EVERY_BATCHES:-10}"
LOG_EVERY_BATCHES="${LOG_EVERY_BATCHES:-50}"
WATCH_INTERVAL_SECONDS="${WATCH_INTERVAL_SECONDS:-60}"
TRIAL_LIMIT="${TRIAL_LIMIT:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_REQUIREMENTS_CHECK="${SKIP_REQUIREMENTS_CHECK:-0}"
ROI_FOCUS_WARMUP_EPOCHS="${ROI_FOCUS_WARMUP_EPOCHS:-10}"
CHECK_LATEST_N="${CHECK_LATEST_N:-8}"
LATEST_N="${LATEST_N:-8}"
VRAM_SAMPLE_SECONDS="${VRAM_SAMPLE_SECONDS:-20}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEARCH_SKIP_DONE="${SEARCH_SKIP_DONE:-1}"
SEARCH_FORCE_RERUN="${SEARCH_FORCE_RERUN:-0}"
SEARCH_RERUN_FAILED="${SEARCH_RERUN_FAILED:-1}"
SEARCH_RESUME_INTERRUPTED="${SEARCH_RESUME_INTERRUPTED:-1}"
SEARCH_ALLOW_PARTIAL_RESUME="${SEARCH_ALLOW_PARTIAL_RESUME:-0}"
SEARCH_ACCEPT_COMPLETE_OUTPUTS="${SEARCH_ACCEPT_COMPLETE_OUTPUTS:-1}"
SEARCH_PROFILE="${SEARCH_PROFILE:-auto}"
NONFINITE_CHECK_EVERY_BATCHES="${NONFINITE_CHECK_EVERY_BATCHES:-10}"
BODY_CT_THR="${BODY_CT_THR:-0.02}"

read -r -a IMG_SIZE_ARRAY <<< "$IMG_SIZE"
if (( ${#IMG_SIZE_ARRAY[@]} != 3 )); then
  echo "[search][FAIL] IMG_SIZE must contain exactly three integers in D H W order, got: $IMG_SIZE" >&2
  exit 1
fi
IMG_SIZE_VOXELS="$((10#${IMG_SIZE_ARRAY[0]} * 10#${IMG_SIZE_ARRAY[1]} * 10#${IMG_SIZE_ARRAY[2]}))"
FULL_IMG_SIZE_VOXELS="$((128 * 256 * 256))"
RESOLVED_SEARCH_PROFILE="$SEARCH_PROFILE"
if [[ "$RESOLVED_SEARCH_PROFILE" == "auto" ]]; then
  if (( IMG_SIZE_VOXELS < FULL_IMG_SIZE_VOXELS )); then
    RESOLVED_SEARCH_PROFILE="roi_crop"
  else
    RESOLVED_SEARCH_PROFILE="full"
  fi
fi
if [[ "$RESOLVED_SEARCH_PROFILE" != "full" && "$RESOLVED_SEARCH_PROFILE" != "roi_crop" ]]; then
  echo "[search][FAIL] SEARCH_PROFILE must be auto, full, or roi_crop. Got: $SEARCH_PROFILE" >&2
  exit 1
fi
if [[ -z "${EVAL_BATCH_SIZE+x}" ]]; then
  if [[ "$RESOLVED_SEARCH_PROFILE" == "roi_crop" ]]; then
    EVAL_BATCH_SIZE=8
  else
    EVAL_BATCH_SIZE=0
  fi
fi

if [[ ! -f "$TRAIN_WRAPPER" ]]; then
  echo "[search][FAIL] train wrapper not found: $TRAIN_WRAPPER" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT/logs"

# Round 3 keeps the ROI constraints pinned while moving the training recipe toward
# prediction performance: shorter localization-only warmup, nonzero survival loss
# during warmup, larger effective batch, lower backbone LR, faster heads, 120-day
# bins, lighter regularization, and validation-selected export.
# Previous round-2 settings are still visible in trial names for continuity.
# - low_dropout_focus12 had the best test c-index.
# - aux50/focus16 were close and had strong validation AUC.
# - all trials passed ROI constraints, so this round changes capacity/VRAM
#   aggressively while keeping GT-mask survival and strict ROI checks pinned.
COMMON_ENV=(
  "ENDPOINT=$ENDPOINT"
  "DEBUG_FOLD=$DEBUG_FOLD"
  "EPOCHS=$EPOCHS"
  "IMG_SIZE=$IMG_SIZE"
  "EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE"
  "WORKERS=$WORKERS"
  "EVAL_WORKERS=$EVAL_WORKERS"
  "PREFETCH_FACTOR=$PREFETCH_FACTOR"
  "CACHE_VOLUMES=$CACHE_VOLUMES"
  "VOLUME_CACHE_SIZE=$VOLUME_CACHE_SIZE"
  "ROI_FOCUS_EVERY_BATCHES=$ROI_FOCUS_EVERY_BATCHES"
  "LOG_EVERY_BATCHES=$LOG_EVERY_BATCHES"
  "WATCH_INTERVAL_SECONDS=$WATCH_INTERVAL_SECONDS"
  "LATEST_N=$LATEST_N"
  "CHECK_LATEST_N=$CHECK_LATEST_N"
  "PYTHON_BIN=$PYTHON_BIN"
  "STRICT=1"
  "SURVIVAL_USE_GT_MASKS=1"
  "MASK_GUIDANCE_ALPHA=1.0"
  "TEACHER_FORCE_EPOCHS=0"
  "TEACHER_FORCE_START=1.0"
  "TEACHER_FORCE_END=1.0"
  "ROI_FOCUS_WARMUP_EPOCHS=$ROI_FOCUS_WARMUP_EPOCHS"
  "ROI_FOCUS_WARMUP_SURVIVAL_WEIGHT=0.2"
  "WARMUP_EPOCHS=$ROI_FOCUS_WARMUP_EPOCHS"
  "GRAD_ACCUMULATION_STEPS=8"
  "NONFINITE_CHECK_EVERY_BATCHES=$NONFINITE_CHECK_EVERY_BATCHES"
  "BODY_CT_THR=$BODY_CT_THR"
  "PT_SHELL_THICKNESS_MM=10.0"
  "LN_SHELL_THICKNESS_MM=0.0"
  "TIME_BIN_WIDTH_DAYS=120.0"
  "REPORT_METRIC=composite"
  "EXPORT_POLICY=best_val"
  "HAZARD_SMOOTH_LAMBDA=0.003"
  "SWA_START_EPOCH=60"
  "LR_CLIN=1e-4"
  "LR_RAD=1e-4"
  "MIN_PROB_MASS_INSIDE_GT=0.95"
  "MIN_SUPPORT_RECALL=0.95"
  "MIN_SUPPORT_DICE=0.02"
  "MAX_EMPTY_WHEN_GT_PRESENT=0.25"
  "SKIP_REQUIREMENTS_CHECK=$SKIP_REQUIREMENTS_CHECK"
  "OMP_NUM_THREADS=$OMP_NUM_THREADS"
  "MKL_NUM_THREADS=$MKL_NUM_THREADS"
  "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
  "NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
  "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"
  "MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX"
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512"
)

if [[ -n "${NVIDIA_TF32_OVERRIDE:-}" ]]; then
  COMMON_ENV+=("NVIDIA_TF32_OVERRIDE=$NVIDIA_TF32_OVERRIDE")
fi

LOW_DROP="MODALITY_DROPOUT_CLIN_P=0.10 MODALITY_DROPOUT_RAD_P=0.10 V2_IMAGE_HABITAT_DROPOUT_P=0.025 V2_NODE_DROPOUT_P=0.05 V2_TOPOLOGY_DROPOUT_P=0.05 TOKEN_MLP_DROPOUT=0.40 PROJ_DROPOUT_P=0.25 RAD_PROJ_DROPOUT_P=0.20 EXPERT_DROPOUT_P=0.10 ATTN_DROPOUT_P=0.10 V2_DROPOUT=0.08"
MID_DROP="MODALITY_DROPOUT_CLIN_P=0.16 MODALITY_DROPOUT_RAD_P=0.16 V2_IMAGE_HABITAT_DROPOUT_P=0.04 V2_NODE_DROPOUT_P=0.08 V2_TOPOLOGY_DROPOUT_P=0.08 TOKEN_MLP_DROPOUT=0.45 PROJ_DROPOUT_P=0.30 RAD_PROJ_DROPOUT_P=0.25 EXPERT_DROPOUT_P=0.12 ATTN_DROPOUT_P=0.12 V2_DROPOUT=0.10"
CAP_384="FUSED_DIM=640 IMG_TOKEN_DIM=1024 TOKEN_MLP_HIDDEN_DIM=2048 IMG_PROJ_HIDDEN_DIM=1536 IMG_TOK_FFN_HIDDEN_DIM=1536 IMG_POST_HIDDEN_DIM=1536 IMG_ATTN_HEADS=8 GATE_HIDDEN_DIM=768 RAD_HIDDEN_DIM=1536 V2_MODEL_DIM=384 V2_NUM_HEADS=8 V2_TRANSFORMER_LAYERS=3 RADIOMICS_PCA_TOTAL_COMPONENTS=160 V2_RADIOMICS_PCS_PER_GROUP=24"
CAP_512="FUSED_DIM=768 IMG_TOKEN_DIM=1280 TOKEN_MLP_HIDDEN_DIM=3072 IMG_PROJ_HIDDEN_DIM=2048 IMG_TOK_FFN_HIDDEN_DIM=2048 IMG_POST_HIDDEN_DIM=2048 IMG_ATTN_HEADS=8 GATE_HIDDEN_DIM=1024 RAD_HIDDEN_DIM=2048 V2_MODEL_DIM=512 V2_NUM_HEADS=8 V2_TRANSFORMER_LAYERS=4 RADIOMICS_PCA_TOTAL_COMPONENTS=192 V2_RADIOMICS_PCS_PER_GROUP=32"
ROI_BASE="LOC_LOSS_PT_LAMBDA=4 LOC_LOSS_LN_LAMBDA=4 MASK_SUPPORT_LAMBDA=2 PT_SHELL_RADIUS=5 LN_SHELL_RADIUS=5"
ROI_HEAVY="LOC_LOSS_PT_LAMBDA=6 LOC_LOSS_LN_LAMBDA=6 MASK_SUPPORT_LAMBDA=3 PT_SHELL_RADIUS=7 LN_SHELL_RADIUS=5"
PRED_TRAIN="LR_BACKBONE=3e-5 LR_HEAD=3e-4 LR_CLIN=1e-4 LR_RAD=1e-4 GRAD_ACCUMULATION_STEPS=8 TIME_BIN_WIDTH_DAYS=120.0 REPORT_METRIC=composite EXPORT_POLICY=best_val HAZARD_SMOOTH_LAMBDA=0.003 SWA_START_EPOCH=60"
ROI_CROP_BS4="GRAD_ACCUMULATION_STEPS=4 EVAL_BATCH_SIZE=8 NONFINITE_CHECK_EVERY_BATCHES=$NONFINITE_CHECK_EVERY_BATCHES TIME_BIN_WIDTH_DAYS=120.0 REPORT_METRIC=composite EXPORT_POLICY=best_val HAZARD_SMOOTH_LAMBDA=0.003 SWA_START_EPOCH=60"
ROI_CROP_BS6="GRAD_ACCUMULATION_STEPS=3 EVAL_BATCH_SIZE=12 NONFINITE_CHECK_EVERY_BATCHES=$NONFINITE_CHECK_EVERY_BATCHES TIME_BIN_WIDTH_DAYS=120.0 REPORT_METRIC=composite EXPORT_POLICY=best_val HAZARD_SMOOTH_LAMBDA=0.003 SWA_START_EPOCH=60"
ROI_CROP_BS8="GRAD_ACCUMULATION_STEPS=2 EVAL_BATCH_SIZE=16 NONFINITE_CHECK_EVERY_BATCHES=$NONFINITE_CHECK_EVERY_BATCHES TIME_BIN_WIDTH_DAYS=120.0 REPORT_METRIC=composite EXPORT_POLICY=best_val HAZARD_SMOOTH_LAMBDA=0.003 SWA_START_EPOCH=60"

if [[ "$RESOLVED_SEARCH_PROFILE" == "roi_crop" ]]; then
  TRIALS=(
    "roi_bs4_nochk_lowdrop_aux20_big384|BATCH_SIZE=4 USE_CHECKPOINT=0 AUX_SURV_LOSS_WEIGHT=0.20 MASK_FOCUS_LAMBDA=12 $ROI_BASE $LOW_DROP $CAP_384 LR_BACKBONE=3e-5 LR_HEAD=3e-4 LR_CLIN=1e-4 LR_RAD=1e-4 $ROI_CROP_BS4|"
    "roi_bs4_nochk_lowdrop_aux35_big384|BATCH_SIZE=4 USE_CHECKPOINT=0 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=12 $ROI_BASE $LOW_DROP $CAP_384 LR_BACKBONE=3e-5 LR_HEAD=3e-4 LR_CLIN=1e-4 LR_RAD=1e-4 $ROI_CROP_BS4|"
    "roi_bs6_nochk_lowdrop_aux20_big384|BATCH_SIZE=6 USE_CHECKPOINT=0 AUX_SURV_LOSS_WEIGHT=0.20 MASK_FOCUS_LAMBDA=12 $ROI_BASE $LOW_DROP $CAP_384 LR_BACKBONE=3e-5 LR_HEAD=3e-4 LR_CLIN=1e-4 LR_RAD=1e-4 $ROI_CROP_BS6|"
    "roi_bs6_nochk_lowdrop_aux35_big384|BATCH_SIZE=6 USE_CHECKPOINT=0 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=12 $ROI_BASE $LOW_DROP $CAP_384 LR_BACKBONE=3e-5 LR_HEAD=3e-4 LR_CLIN=1e-4 LR_RAD=1e-4 $ROI_CROP_BS6|"
    "roi_bs8_ckpt_lowdrop_aux20_big384|BATCH_SIZE=8 USE_CHECKPOINT=1 AUX_SURV_LOSS_WEIGHT=0.20 MASK_FOCUS_LAMBDA=12 $ROI_BASE $LOW_DROP $CAP_384 LR_BACKBONE=3e-5 LR_HEAD=3e-4 LR_CLIN=1e-4 LR_RAD=1e-4 $ROI_CROP_BS8|"
    "roi_bs8_ckpt_lowdrop_aux35_big384|BATCH_SIZE=8 USE_CHECKPOINT=1 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=12 $ROI_BASE $LOW_DROP $CAP_384 LR_BACKBONE=3e-5 LR_HEAD=3e-4 LR_CLIN=1e-4 LR_RAD=1e-4 $ROI_CROP_BS8|"
    "roi_bs4_nochk_focus16_big384|BATCH_SIZE=4 USE_CHECKPOINT=0 AUX_SURV_LOSS_WEIGHT=0.20 MASK_FOCUS_LAMBDA=16 $ROI_BASE $LOW_DROP $CAP_384 LR_BACKBONE=3e-5 LR_HEAD=3e-4 LR_CLIN=1e-4 LR_RAD=1e-4 $ROI_CROP_BS4|"
    "roi_bs6_ckpt_focus16_big384|BATCH_SIZE=6 USE_CHECKPOINT=1 AUX_SURV_LOSS_WEIGHT=0.20 MASK_FOCUS_LAMBDA=16 $ROI_BASE $LOW_DROP $CAP_384 LR_BACKBONE=3e-5 LR_HEAD=3e-4 LR_CLIN=1e-4 LR_RAD=1e-4 $ROI_CROP_BS6|"
    "roi_bs4_nochk_lowdrop_aux20_big512|BATCH_SIZE=4 USE_CHECKPOINT=0 AUX_SURV_LOSS_WEIGHT=0.20 MASK_FOCUS_LAMBDA=12 $ROI_BASE $LOW_DROP $CAP_512 LR_BACKBONE=3e-5 LR_HEAD=3e-4 LR_CLIN=1e-4 LR_RAD=1e-4 $ROI_CROP_BS4|"
    "roi_bs6_ckpt_lowdrop_aux20_big512|BATCH_SIZE=6 USE_CHECKPOINT=1 AUX_SURV_LOSS_WEIGHT=0.20 MASK_FOCUS_LAMBDA=12 $ROI_BASE $LOW_DROP $CAP_512 LR_BACKBONE=3e-5 LR_HEAD=3e-4 LR_CLIN=1e-4 LR_RAD=1e-4 $ROI_CROP_BS6|"
    "roi_feature120_bs4_nochk_big384|BATCH_SIZE=4 USE_CHECKPOINT=0 FEATURE_SIZE=120 AUX_SURV_LOSS_WEIGHT=0.20 MASK_FOCUS_LAMBDA=12 $ROI_BASE $MID_DROP $CAP_384 LR_BACKBONE=3e-5 LR_HEAD=3e-4 LR_CLIN=1e-4 LR_RAD=1e-4 $ROI_CROP_BS4|"
    "roi_feature120_bs6_ckpt_big384|BATCH_SIZE=6 USE_CHECKPOINT=1 FEATURE_SIZE=120 AUX_SURV_LOSS_WEIGHT=0.20 MASK_FOCUS_LAMBDA=12 $ROI_BASE $MID_DROP $CAP_384 LR_BACKBONE=3e-5 LR_HEAD=3e-4 LR_CLIN=1e-4 LR_RAD=1e-4 $ROI_CROP_BS6|"
  )
else
  TRIALS=(
    "r2_bs2_lowdrop_aux35_big384|BATCH_SIZE=2 USE_CHECKPOINT=1 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=12 $ROI_BASE $LOW_DROP $CAP_384 $PRED_TRAIN|"
    "r2_bs2_lowdrop_aux50_big384|BATCH_SIZE=2 USE_CHECKPOINT=1 AUX_SURV_LOSS_WEIGHT=0.50 MASK_FOCUS_LAMBDA=12 $ROI_BASE $LOW_DROP $CAP_384 $PRED_TRAIN|"
    "r2_nochk_lowdrop_aux35_big384|BATCH_SIZE=1 USE_CHECKPOINT=0 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=12 $ROI_BASE $LOW_DROP $CAP_384 $PRED_TRAIN|"
    "r2_nochk_lowdrop_aux50_big384|BATCH_SIZE=1 USE_CHECKPOINT=0 AUX_SURV_LOSS_WEIGHT=0.50 MASK_FOCUS_LAMBDA=12 $ROI_BASE $LOW_DROP $CAP_384 $PRED_TRAIN|"
    "r2_bs2_focus16_big384|BATCH_SIZE=2 USE_CHECKPOINT=1 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=16 $ROI_BASE $LOW_DROP $CAP_384 $PRED_TRAIN|"
    "r2_nochk_focus16_big384|BATCH_SIZE=1 USE_CHECKPOINT=0 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=16 $ROI_BASE $LOW_DROP $CAP_384 $PRED_TRAIN|"
    "r2_bs2_shell7_focus16_big384|BATCH_SIZE=2 USE_CHECKPOINT=1 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=16 $ROI_HEAVY $LOW_DROP $CAP_384 $PRED_TRAIN|"
    "r2_nochk_shell7_focus16_big384|BATCH_SIZE=1 USE_CHECKPOINT=0 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=16 $ROI_HEAVY $LOW_DROP $CAP_384 $PRED_TRAIN|"
    "r2_bs2_lowdrop_aux35_big512|BATCH_SIZE=2 USE_CHECKPOINT=1 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=12 $ROI_BASE $LOW_DROP $CAP_512 $PRED_TRAIN|"
    "r2_nochk_lowdrop_aux35_big512|BATCH_SIZE=1 USE_CHECKPOINT=0 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=12 $ROI_BASE $LOW_DROP $CAP_512 $PRED_TRAIN|"
    "r2_feature120_ckpt_big384|BATCH_SIZE=1 USE_CHECKPOINT=1 FEATURE_SIZE=120 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=12 $ROI_BASE $MID_DROP $CAP_384 $PRED_TRAIN|"
    "r2_feature120_bs2_ckpt_big384|BATCH_SIZE=2 USE_CHECKPOINT=1 FEATURE_SIZE=120 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=12 $ROI_BASE $MID_DROP $CAP_384 $PRED_TRAIN|"
    "r2_near77_bs2_nochk_lowdrop_big384|BATCH_SIZE=2 USE_CHECKPOINT=0 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=12 $ROI_BASE $LOW_DROP $CAP_384 $PRED_TRAIN|"
    "r2_near77_bs2_nochk_focus16_big384|BATCH_SIZE=2 USE_CHECKPOINT=0 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=16 $ROI_BASE $LOW_DROP $CAP_384 $PRED_TRAIN|"
    "r2_near77_bs2_ckpt_feature120_big512|BATCH_SIZE=2 USE_CHECKPOINT=1 FEATURE_SIZE=120 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=12 $ROI_BASE $MID_DROP $CAP_512 $PRED_TRAIN|"
    "r2_near77_bs1_nochk_feature120_big512|BATCH_SIZE=1 USE_CHECKPOINT=0 FEATURE_SIZE=120 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=12 $ROI_BASE $MID_DROP $CAP_512 $PRED_TRAIN|"
  )
fi

trial_total="${#TRIALS[@]}"
if [[ "$TRIAL_LIMIT" =~ ^[0-9]+$ ]] && (( TRIAL_LIMIT > 0 && TRIAL_LIMIT < trial_total )); then
  trial_total="$TRIAL_LIMIT"
fi

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|Yes|y|Y|on|ON|On) return 0 ;;
    *) return 1 ;;
  esac
}

read_first_line() {
  local path="$1"
  local line=""
  if [[ -f "$path" ]]; then
    IFS= read -r line < "$path" || true
    printf '%s' "$line"
  fi
}

trial_has_checkpoint() {
  local trial_dir="$1"
  [[ -d "$trial_dir" ]] || return 1
  find "$trial_dir" -type f -name "last.pt" -print -quit 2>/dev/null | grep -q .
}

trial_has_complete_outputs() {
  local trial_dir="$1"
  local summary_path
  [[ -d "$trial_dir" ]] || return 1
  while IFS= read -r summary_path; do
    if "$PYTHON_BIN" - "$summary_path" >/dev/null 2>&1 <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file() or path.stat().st_size <= 0:
    raise SystemExit(1)
summary = json.loads(path.read_text())
folds = summary.get("folds_run")
if not isinstance(folds, list) or not folds:
    raise SystemExit(1)
required = ("mean_fold_test_c_index", "primary_endpoint", "fold_export_suffixes", "export_policy")
if any(key not in summary for key in required):
    raise SystemExit(1)
PY
    then
      return 0
    fi
  done < <(find "$trial_dir" -type f -name "cv_summary.json" -print 2>/dev/null)
  return 1
}

next_attempt_number() {
  local trial_dir="$1"
  local attempt_file="$trial_dir/attempt.txt"
  local previous=""
  if [[ -f "$attempt_file" ]]; then
    previous="$(tr -cd '0-9' < "$attempt_file" || true)"
  fi
  if [[ -z "$previous" ]]; then
    previous=0
  fi
  printf '%d' "$((10#$previous + 1))"
}

write_search_manifest() {
  {
    echo "out_root=$OUT_ROOT"
    echo "endpoint=$ENDPOINT"
    echo "debug_fold=$DEBUG_FOLD"
    echo "gpu_ids=${GPU_ARRAY[*]}"
    echo "epochs=$EPOCHS"
    echo "img_size=$IMG_SIZE"
    echo "img_size_voxels=$IMG_SIZE_VOXELS"
    echo "search_profile=$RESOLVED_SEARCH_PROFILE"
    echo "eval_batch_size_default=$EVAL_BATCH_SIZE"
    echo "workers=$WORKERS"
    echo "eval_workers=$EVAL_WORKERS"
    echo "prefetch_factor=$PREFETCH_FACTOR"
    echo "cache_volumes=$CACHE_VOLUMES"
    echo "volume_cache_size=$VOLUME_CACHE_SIZE"
    echo "roi_focus_every_batches=$ROI_FOCUS_EVERY_BATCHES"
    echo "nonfinite_check_every_batches=$NONFINITE_CHECK_EVERY_BATCHES"
    echo "body_ct_thr=$BODY_CT_THR"
    echo "pt_shell_thickness_mm=10.0"
    echo "ln_shell_thickness_mm=0.0"
    echo "roi_focus_warmup_epochs=$ROI_FOCUS_WARMUP_EPOCHS"
    echo "target_peak_vram_gb=77"
    echo "vram_sample_seconds=$VRAM_SAMPLE_SECONDS"
    echo "strict=1"
    echo "survival_use_gt_masks=1"
    echo "mask_guidance_alpha=1.0"
    echo "score=validation_only_composite_auc_c"
    echo "min_prob_mass_inside_gt=0.95"
    echo "min_support_recall=0.95"
    echo "min_support_dice=0.02"
    echo "search_skip_done=$SEARCH_SKIP_DONE"
    echo "search_force_rerun=$SEARCH_FORCE_RERUN"
    echo "search_rerun_failed=$SEARCH_RERUN_FAILED"
    echo "search_resume_interrupted=$SEARCH_RESUME_INTERRUPTED"
    echo "search_allow_partial_resume=$SEARCH_ALLOW_PARTIAL_RESUME"
    echo "search_accept_complete_outputs=$SEARCH_ACCEPT_COMPLETE_OUTPUTS"
    echo "trial_total=$trial_total"
    echo
    for ((i = 0; i < trial_total; i++)); do
      IFS='|' read -r trial_name trial_env trial_args <<< "${TRIALS[$i]}"
      echo "trial_$i=$trial_name"
      echo "trial_${i}_env=$trial_env"
      echo "trial_${i}_args=$trial_args"
    done
  } > "$OUT_ROOT/search_manifest.txt"
}

run_slot() {
  local slot="$1"
  local gpu="$2"
  local idx trial_name trial_env trial_args trial_dir exp_name log_file rc
  local vram_file vram_pid status_path rc_path previous_status previous_rc
  local resume_value allow_partial_value attempt attempt_tag checkpoint_state complete_state
  for ((idx = slot; idx < trial_total; idx += ${#GPU_ARRAY[@]})); do
    IFS='|' read -r trial_name trial_env trial_args <<< "${TRIALS[$idx]}"
    trial_dir="$OUT_ROOT/$(printf '%02d' "$idx")_${trial_name}"
    exp_name="$(printf 'cv4_contour_aware_%s_round3_%02d_%s_fold%s' "$ENDPOINT_LC" "$idx" "$trial_name" "$fold_tag")"
    mkdir -p "$trial_dir"

    status_path="$trial_dir/status.txt"
    rc_path="$trial_dir/status.rc"
    previous_status="$(read_first_line "$status_path")"
    previous_rc="$(read_first_line "$rc_path")"
    if [[ -z "$previous_status" ]]; then
      previous_status="missing"
    fi
    complete_state=0
    if trial_has_complete_outputs "$trial_dir"; then
      complete_state=1
    fi

    if ! is_truthy "$SEARCH_FORCE_RERUN"; then
      if is_truthy "$SEARCH_ACCEPT_COMPLETE_OUTPUTS" && [[ "$previous_status" != "failed" && "$complete_state" == "1" ]]; then
        echo "done" > "$status_path"
        echo "0" > "$rc_path"
        echo "[search][slot $slot gpu $gpu] skipping trial $idx with complete outputs: $trial_name"
        continue
      fi
      if is_truthy "$SEARCH_SKIP_DONE" && [[ "$previous_status" == "done" && "$previous_rc" == "0" ]]; then
        echo "[search][slot $slot gpu $gpu] skipping completed trial $idx: $trial_name"
        continue
      fi
      if ! is_truthy "$SEARCH_RERUN_FAILED" && [[ "$previous_status" == "failed" ]]; then
        echo "[search][slot $slot gpu $gpu] skipping failed trial $idx because SEARCH_RERUN_FAILED=0: $trial_name"
        continue
      fi
    fi

    checkpoint_state=0
    if trial_has_checkpoint "$trial_dir"; then
      checkpoint_state=1
    fi
    resume_value=0
    if ! is_truthy "$SEARCH_FORCE_RERUN" && is_truthy "$SEARCH_RESUME_INTERRUPTED" && [[ "$checkpoint_state" == "1" ]]; then
      resume_value=1
    fi
    allow_partial_value=0
    if is_truthy "$SEARCH_ALLOW_PARTIAL_RESUME"; then
      allow_partial_value=1
    fi
    attempt="$(next_attempt_number "$trial_dir")"
    printf '%s\n' "$attempt" > "$trial_dir/attempt.txt"
    attempt_tag="$(printf 'attempt%02d' "$attempt")"
    log_file="$OUT_ROOT/logs/$(printf '%02d' "$idx")_${trial_name}_${attempt_tag}.log"
    vram_file="$trial_dir/vram_gpu_${gpu}_${attempt_tag}.csv"

    read -r -a trial_env_array <<< "$trial_env"
    read -r -a trial_arg_array <<< "$trial_args"

    {
      echo "trial_index=$idx"
      echo "trial_name=$trial_name"
      echo "slot=$slot"
      echo "cuda_device=$gpu"
      echo "target_peak_vram_gb=77"
      echo "out_dir=$trial_dir"
      echo "exp_name=$exp_name"
      echo "attempt=$attempt"
      echo "previous_status=$previous_status"
      echo "previous_rc=$previous_rc"
      echo "has_last_checkpoint=$checkpoint_state"
      echo "resume=$resume_value"
      echo "allow_partial_resume=$allow_partial_value"
      echo "env=$trial_env"
      echo "args=$trial_args"
      echo "log=$log_file"
      echo "vram_csv=$vram_file"
      echo "started_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    } > "$trial_dir/trial.env"
    echo "running" > "$status_path"
    rm -f "$rc_path"

    echo "[search][slot $slot gpu $gpu] starting trial $idx/$((trial_total - 1)): $trial_name resume=$resume_value attempt=$attempt previous_status=$previous_status"
    vram_pid=""
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used \
        --format=csv \
        -i "$gpu" \
        -l "$VRAM_SAMPLE_SECONDS" > "$vram_file" 2>/dev/null &
      vram_pid="$!"
    fi

    set +e
    env \
      "${COMMON_ENV[@]}" \
      "${trial_env_array[@]}" \
      "RESUME=$resume_value" \
      "ALLOW_PARTIAL_RESUME=$allow_partial_value" \
      "CUDA_DEVICE=$gpu" \
      "OUT_DIR=$trial_dir" \
      "EXP_NAME=$exp_name" \
      bash "$TRAIN_WRAPPER" "${trial_arg_array[@]}" > "$log_file" 2>&1
    rc="$?"
    set -e

    if [[ -n "$vram_pid" ]]; then
      kill "$vram_pid" >/dev/null 2>&1 || true
      wait "$vram_pid" >/dev/null 2>&1 || true
    fi

    echo "$rc" > "$rc_path"
    date -u '+%Y-%m-%dT%H:%M:%SZ' > "$trial_dir/finished_utc.txt"
    if [[ "$rc" == "0" ]]; then
      echo "done" > "$status_path"
      echo "[search][slot $slot gpu $gpu] completed trial $idx: $trial_name"
    else
      echo "failed" > "$status_path"
      echo "[search][slot $slot gpu $gpu][FAIL] trial $idx failed rc=$rc: $trial_name (log: $log_file)" >&2
    fi
  done
}

aggregate_results() {
  "$PYTHON_BIN" - "$OUT_ROOT" <<'PY'
import csv
import json
import math
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])

def as_float(value):
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")

def max_finite(values):
    finite = [v for v in (as_float(value) for value in values) if math.isfinite(v)]
    return max(finite) if finite else float("nan")

def last_metrics_row(trial_dir):
    paths = sorted(trial_dir.rglob("metrics.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths:
        try:
            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                return rows[-1], rows, path
        except Exception:
            continue
    return {}, [], None

def summary_json(trial_dir):
    paths = sorted(trial_dir.rglob("cv_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths:
        try:
            return json.loads(path.read_text()), path
        except Exception:
            continue
    return {}, None

def peak_vram_mib(trial_dir):
    peaks = []
    for path in trial_dir.glob("vram_gpu_*.csv"):
        try:
            with path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    value = row.get(" memory.used [MiB]", row.get("memory.used [MiB]", ""))
                    m = re.search(r"([0-9.]+)", str(value))
                    if m:
                        peaks.append(float(m.group(1)))
        except Exception:
            pass
    return max(peaks) if peaks else float("nan")

def gpu_util_stats(trial_dir):
    gpu_utils = []
    mem_utils = []
    for path in trial_dir.glob("vram_gpu_*.csv"):
        try:
            with path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    gpu_value = row.get(" utilization.gpu [%]", row.get("utilization.gpu [%]", ""))
                    mem_value = row.get(" utilization.memory [%]", row.get("utilization.memory [%]", ""))
                    gpu_match = re.search(r"([0-9.]+)", str(gpu_value))
                    mem_match = re.search(r"([0-9.]+)", str(mem_value))
                    if gpu_match:
                        gpu_utils.append(float(gpu_match.group(1)))
                    if mem_match:
                        mem_utils.append(float(mem_match.group(1)))
        except Exception:
            pass
    mean_gpu = sum(gpu_utils) / len(gpu_utils) if gpu_utils else float("nan")
    peak_gpu = max(gpu_utils) if gpu_utils else float("nan")
    mean_mem = sum(mem_utils) / len(mem_utils) if mem_utils else float("nan")
    return mean_gpu, peak_gpu, mean_mem

rows = []
for trial_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "logs"):
    rc_path = trial_dir / "status.rc"
    status_path = trial_dir / "status.txt"
    env_path = trial_dir / "trial.env"
    status = status_path.read_text().strip() if status_path.exists() else "missing"
    rc = rc_path.read_text().strip() if rc_path.exists() else ""
    env_lines = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                env_lines[k] = v
    summary, summary_path = summary_json(trial_dir)
    last, metric_rows, metrics_path = last_metrics_row(trial_dir)
    best_val_composite = float("nan")
    best_val_auc = float("nan")
    best_val_c = float("nan")
    if metric_rows:
        best_val_composite = max_finite(
            r.get("val_composite_score", r.get("val_composite"))
            for r in metric_rows
        )
        best_val_auc = max_finite(r.get("val_auc_1095d") for r in metric_rows)
        best_val_c = max_finite(r.get("val_c_index", r.get("val_c")) for r in metric_rows)
    test_c = as_float(summary.get("mean_fold_test_c_index", summary.get("mean_test_c_index")))
    score = best_val_composite
    if not math.isfinite(score):
        score = best_val_auc
    if not math.isfinite(score):
        score = best_val_c
    peak_mib = peak_vram_mib(trial_dir)
    mean_gpu_util, peak_gpu_util, mean_mem_util = gpu_util_stats(trial_dir)
    rows.append({
        "trial": trial_dir.name,
        "status": status,
        "rc": rc,
        "score": score,
        "test_c_index": test_c,
        "best_val_composite": best_val_composite,
        "best_val_auc_1095d": best_val_auc,
        "best_val_c": best_val_c,
        "final_val_auc_1095d": as_float(last.get("val_auc_1095d")),
        "final_val_c": as_float(last.get("val_c_index", last.get("val_c"))),
        "peak_vram_mib": peak_mib,
        "peak_vram_gb": peak_mib / 1024.0 if math.isfinite(peak_mib) else float("nan"),
        "mean_gpu_util_pct": mean_gpu_util,
        "peak_gpu_util_pct": peak_gpu_util,
        "mean_mem_util_pct": mean_mem_util,
        "attempt": env_lines.get("attempt", ""),
        "resume": env_lines.get("resume", ""),
        "previous_status": env_lines.get("previous_status", ""),
        "pt_mass": as_float(last.get("train_roi_focus_pt_prob_mass_inside_gt")),
        "ln_mass": as_float(last.get("train_roi_focus_ln_prob_mass_inside_gt")),
        "pt_peri_mass": as_float(last.get("train_roi_focus_pt_peri_prob_mass_inside_gt")),
        "pt_rec": as_float(last.get("train_roi_focus_pt_support_recall")),
        "ln_rec": as_float(last.get("train_roi_focus_ln_support_recall")),
        "pt_peri_rec": as_float(last.get("train_roi_focus_pt_peri_support_recall")),
        "pt_dice": as_float(last.get("train_roi_focus_pt_support_dice")),
        "ln_dice": as_float(last.get("train_roi_focus_ln_support_dice")),
        "pt_peri_dice": as_float(last.get("train_roi_focus_pt_peri_support_dice")),
        "env": env_lines.get("env", ""),
        "metrics_csv": str(metrics_path) if metrics_path else "",
        "summary_json": str(summary_path) if summary_path else "",
    })

rows.sort(key=lambda r: (math.isfinite(r["score"]), r["score"]), reverse=True)
out_csv = root / "search_summary.csv"
fieldnames = [
    "trial", "status", "rc", "score", "test_c_index", "best_val_composite", "best_val_auc_1095d", "best_val_c",
    "final_val_auc_1095d", "final_val_c", "peak_vram_mib", "peak_vram_gb",
    "mean_gpu_util_pct", "peak_gpu_util_pct", "mean_mem_util_pct",
    "attempt", "resume", "previous_status",
    "pt_mass", "ln_mass", "pt_peri_mass",
    "pt_rec", "ln_rec", "pt_peri_rec",
    "pt_dice", "ln_dice", "pt_peri_dice",
    "env", "metrics_csv", "summary_json",
]
with out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

def fmt(x):
    return "" if not math.isfinite(x) else f"{x:.4f}"

print(f"[search] wrote {out_csv}")
print("[search] top trials:")
print("rank,trial,status,score,test_c,best_val_composite,best_val_auc,best_val_c,peak_vram_gb,mean_gpu_util,peak_gpu_util,pt_mass,ln_mass,ptp_mass,pt_rec,ln_rec,ptp_rec")
for rank, r in enumerate(rows[:10], 1):
    print(",".join([
        str(rank),
        r["trial"],
        r["status"],
        fmt(r["score"]),
        fmt(r["test_c_index"]),
        fmt(r["best_val_composite"]),
        fmt(r["best_val_auc_1095d"]),
        fmt(r["best_val_c"]),
        fmt(r["peak_vram_gb"]),
        fmt(r["mean_gpu_util_pct"]),
        fmt(r["peak_gpu_util_pct"]),
        fmt(r["pt_mass"]),
        fmt(r["ln_mass"]),
        fmt(r["pt_peri_mass"]),
        fmt(r["pt_rec"]),
        fmt(r["ln_rec"]),
        fmt(r["pt_peri_rec"]),
    ]))
PY
}

write_search_manifest
echo "[search] package_dir=$PACKAGE_DIR"
echo "[search] out_root=$OUT_ROOT"
echo "[search] gpu_ids=${GPU_ARRAY[*]}"
echo "[search] trials=$trial_total epochs=$EPOCHS target_peak_vram_gb=77"
echo "[search] enforcing GT survival masks and strict PT/LN/PT-peri ROI constraints"

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "yes" ]]; then
  echo "[search] dry run only; wrote manifest to $OUT_ROOT/search_manifest.txt"
  for ((i = 0; i < trial_total; i++)); do
    IFS='|' read -r trial_name trial_env trial_args <<< "${TRIALS[$i]}"
    gpu="${GPU_ARRAY[$((i % ${#GPU_ARRAY[@]}))]}"
    trial_dir="$OUT_ROOT/$(printf '%02d' "$i")_${trial_name}"
    status="$(read_first_line "$trial_dir/status.txt")"
    rc="$(read_first_line "$trial_dir/status.rc")"
    if [[ -z "$status" ]]; then
      status="missing"
    fi
    checkpoint_state=0
    if trial_has_checkpoint "$trial_dir"; then
      checkpoint_state=1
    fi
    complete_state=0
    if trial_has_complete_outputs "$trial_dir"; then
      complete_state=1
    fi
    action="run"
    resume_value=0
    if ! is_truthy "$SEARCH_FORCE_RERUN"; then
      if is_truthy "$SEARCH_ACCEPT_COMPLETE_OUTPUTS" && [[ "$status" != "failed" && "$complete_state" == "1" ]]; then
        action="skip_complete_outputs"
      elif is_truthy "$SEARCH_SKIP_DONE" && [[ "$status" == "done" && "$rc" == "0" ]]; then
        action="skip_done"
      elif ! is_truthy "$SEARCH_RERUN_FAILED" && [[ "$status" == "failed" ]]; then
        action="skip_failed"
      fi
    fi
    if [[ "$action" == "run" ]] && ! is_truthy "$SEARCH_FORCE_RERUN" && is_truthy "$SEARCH_RESUME_INTERRUPTED" && [[ "$checkpoint_state" == "1" ]]; then
      resume_value=1
      action="resume"
    fi
    echo "[search][dry-run] trial $i gpu=$gpu action=$action resume=$resume_value status=$status rc=$rc checkpoint=$checkpoint_state complete=$complete_state name=$trial_name env='$trial_env' args='$trial_args'"
  done
  exit 0
fi

pids=()
for slot in "${!GPU_ARRAY[@]}"; do
  run_slot "$slot" "${GPU_ARRAY[$slot]}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

aggregate_results
