#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SETTINGS_FILE="${1:-${PIPELINE_SETTINGS:-scripts/config/pipeline_2xh100_test.env}}"
if [[ ! -f "$SETTINGS_FILE" ]]; then
  echo "[error] settings file not found: $SETTINGS_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$SETTINGS_FILE"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

wait_all_or_fail() {
  local status=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

run_preprocess() {
  echo "[pipeline] preprocessing -> $PREPROCESS_OUT_ROOT"
  DICOM_ROOT="$DICOM_ROOT" \
  SURV_CSV="$SURV_CSV" \
  OUT_ROOT="$PREPROCESS_OUT_ROOT" \
  OUT_CSV="$PREPROCESS_OUT_CSV" \
  SPACING_X="$PREPROCESS_SPACING_X" \
  SPACING_Y="$PREPROCESS_SPACING_Y" \
  SPACING_Z="$PREPROCESS_SPACING_Z" \
  SIZE_D="$PREPROCESS_SIZE_D" \
  SIZE_H="$PREPROCESS_SIZE_H" \
  SIZE_W="$PREPROCESS_SIZE_W" \
  MARGIN_MM="$PREPROCESS_MARGIN_MM" \
  HU_MIN="$PREPROCESS_HU_MIN" \
  HU_MAX="$PREPROCESS_HU_MAX" \
  RECURSIVE="$PREPROCESS_RECURSIVE" \
  ./scripts/run_preprocess_export_swinunetr.sh "${PREPROCESS_EXTRA_ARGS[@]}"
}

run_splits() {
  echo "[pipeline] making splits -> $SPLITS_OUT_DIR"
  META_CSV="$SPLITS_META_CSV" \
  QC_REPORT="$SPLITS_QC_REPORT" \
  QC_POLICY="$SPLITS_QC_POLICY" \
  QC_DROP_AIR_GT="$SPLITS_QC_DROP_AIR_GT" \
  ENDPOINT="$SPLITS_ENDPOINT" \
  CV_FOLDS="$SPLITS_CV_FOLDS" \
  VAL_FRAC="$SPLITS_VAL_FRAC" \
  SPLIT_SEED="$SPLITS_SEED" \
  OUT_DIR="$SPLITS_OUT_DIR" \
  ./scripts/run_make_cv_splits.sh "${SPLITS_EXTRA_ARGS[@]}"
}

run_stage1() {
  echo "[pipeline] stage 1 PT/LN pretraining on GPUs $STAGE1_GPU_PT and $STAGE1_GPU_LN"

  META_CSV="$STAGE1_META_CSV" \
  OUT_DIR="$STAGE1_PT_OUT_DIR" \
  CUDA_DEVICE="$STAGE1_GPU_PT" \
  DEVICE="$STAGE1_DEVICE" \
  ./scripts/run_stage1_pretrain_pt.sh \
    --epochs "$STAGE1_EPOCHS" \
    --batch_size "$STAGE1_BATCH_SIZE" \
    --workers "$STAGE1_WORKERS" \
    "${STAGE1_PT_EXTRA_ARGS[@]}" &
  pt_pid=$!

  META_CSV="$STAGE1_META_CSV" \
  OUT_DIR="$STAGE1_LN_OUT_DIR" \
  CUDA_DEVICE="$STAGE1_GPU_LN" \
  DEVICE="$STAGE1_DEVICE" \
  ./scripts/run_stage1_pretrain_ln.sh \
    --epochs "$STAGE1_EPOCHS" \
    --batch_size "$STAGE1_BATCH_SIZE" \
    --workers "$STAGE1_WORKERS" \
    "${STAGE1_LN_EXTRA_ARGS[@]}" &
  ln_pid=$!

  wait_all_or_fail "$pt_pid" "$ln_pid"
}

resolve_stage1_ckpts() {
  STAGE2_PT_CKPT="${PT_CKPT:-${STAGE1_PT_OUT_DIR}/all/seg_best.pt}"
  STAGE2_LN_CKPT="${LN_CKPT:-${STAGE1_LN_OUT_DIR}/all/seg_best.pt}"

  if [[ ! -f "$STAGE2_PT_CKPT" ]]; then
    echo "[error] PT checkpoint not found: $STAGE2_PT_CKPT" >&2
    exit 1
  fi
  if [[ ! -f "$STAGE2_LN_CKPT" ]]; then
    echo "[error] LN checkpoint not found: $STAGE2_LN_CKPT" >&2
    exit 1
  fi
}

run_stage2_fold() {
  local fold="$1"
  local gpu="$2"
  local exp_name="${STAGE2_EXP_PREFIX}_fold$(printf '%02d' "$fold")"
  local -a extra_args=(
    --epochs "$STAGE2_EPOCHS"
    --batch_size "$STAGE2_BATCH_SIZE"
    --workers "$STAGE2_WORKERS"
  )

  if [[ "$STAGE2_USE_RESUME" != "1" ]]; then
    extra_args+=(--no_resume)
  fi
  if [[ "${STAGE2_DEBUG_MAX_TRAIN:-0}" -gt 0 ]]; then
    extra_args+=(--debug_max_train "$STAGE2_DEBUG_MAX_TRAIN")
  fi
  if [[ "${STAGE2_DEBUG_MAX_VAL:-0}" -gt 0 ]]; then
    extra_args+=(--debug_max_val "$STAGE2_DEBUG_MAX_VAL")
  fi
  if [[ "${STAGE2_DEBUG_MAX_TEST:-0}" -gt 0 ]]; then
    extra_args+=(--debug_max_test "$STAGE2_DEBUG_MAX_TEST")
  fi
  extra_args+=("${STAGE2_EXTRA_ARGS[@]}")

  echo "[pipeline] stage 2 fold $fold on GPU $gpu -> $exp_name"
  if [[ "$STAGE2_USE_LORA" == "1" ]]; then
    META_CSV="$STAGE2_META_CSV" \
    SPLITS_DIR="$STAGE2_SPLITS_DIR" \
    PT_CKPT="$STAGE2_PT_CKPT" \
    LN_CKPT="$STAGE2_LN_CKPT" \
    OUT_DIR="$STAGE2_OUT_DIR" \
    EXP_NAME="$exp_name" \
    DEBUG_FOLD="$fold" \
    CUDA_DEVICE="$gpu" \
    DEVICE="$STAGE2_DEVICE" \
    ./scripts/run_stage2_survival_lora.sh "${extra_args[@]}"
  else
    META_CSV="$STAGE2_META_CSV" \
    SPLITS_DIR="$STAGE2_SPLITS_DIR" \
    PT_CKPT="$STAGE2_PT_CKPT" \
    LN_CKPT="$STAGE2_LN_CKPT" \
    OUT_DIR="$STAGE2_OUT_DIR" \
    EXP_NAME="$exp_name" \
    DEBUG_FOLD="$fold" \
    CUDA_DEVICE="$gpu" \
    DEVICE="$STAGE2_DEVICE" \
    ./scripts/run_stage2_survival.sh "${extra_args[@]}"
  fi
}

run_stage2() {
  if [[ ! -f "$STAGE2_META_CSV" ]]; then
    echo "[error] stage 2 meta csv not found: $STAGE2_META_CSV" >&2
    echo "[error] provide a clinically augmented metafile in the settings file." >&2
    exit 1
  fi
  if [[ ! -d "$STAGE2_SPLITS_DIR" ]]; then
    echo "[error] stage 2 splits dir not found: $STAGE2_SPLITS_DIR" >&2
    exit 1
  fi

  resolve_stage1_ckpts

  local max_parallel="${#STAGE2_GPU_IDS[@]}"
  if (( max_parallel == 0 )); then
    echo "[error] STAGE2_GPU_IDS is empty." >&2
    exit 1
  fi

  local -a pids=()
  local launch_idx=0
  for fold in "${STAGE2_FOLDS[@]}"; do
    local gpu="${STAGE2_GPU_IDS[$(( launch_idx % max_parallel ))]}"
    run_stage2_fold "$fold" "$gpu" &
    pids+=("$!")
    launch_idx=$(( launch_idx + 1 ))

    if (( ${#pids[@]} == max_parallel )); then
      wait_all_or_fail "${pids[@]}"
      pids=()
    fi
  done

  if (( ${#pids[@]} > 0 )); then
    wait_all_or_fail "${pids[@]}"
  fi
}

if [[ "$RUN_PREPROCESS" == "1" ]]; then
  run_preprocess
fi

if [[ "$RUN_SPLITS" == "1" ]]; then
  run_splits
fi

if [[ "$RUN_STAGE1" == "1" ]]; then
  run_stage1
fi

if [[ "$RUN_STAGE2" == "1" ]]; then
  run_stage2
fi

echo "[done] two-stage pipeline finished"
