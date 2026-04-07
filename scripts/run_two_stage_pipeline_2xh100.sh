#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PACKAGE_DIR/.." && pwd)"

SETTINGS_FILE="${1:-${PIPELINE_SETTINGS:-$PACKAGE_DIR/scripts/config/pipeline_2xh100_test.env}}"
if [[ "$SETTINGS_FILE" != /* && ! -f "$SETTINGS_FILE" && -f "$PACKAGE_DIR/$SETTINGS_FILE" ]]; then
  SETTINGS_FILE="$PACKAGE_DIR/$SETTINGS_FILE"
fi
if [[ ! -f "$SETTINGS_FILE" ]]; then
  echo "[error] settings file not found: $SETTINGS_FILE" >&2
  exit 1
fi

ENV_RUN_PREPROCESS="${RUN_PREPROCESS-__TF_UNSET__}"
ENV_RUN_PREPARE_STAGE2="${RUN_PREPARE_STAGE2-__TF_UNSET__}"
ENV_RUN_SPLITS="${RUN_SPLITS-__TF_UNSET__}"
ENV_RUN_STAGE1="${RUN_STAGE1-__TF_UNSET__}"
ENV_RUN_STAGE2="${RUN_STAGE2-__TF_UNSET__}"

# shellcheck disable=SC1090
source "$SETTINGS_FILE"

if [[ "$ENV_RUN_PREPROCESS" != "__TF_UNSET__" ]]; then
  RUN_PREPROCESS="$ENV_RUN_PREPROCESS"
fi
if [[ "$ENV_RUN_PREPARE_STAGE2" != "__TF_UNSET__" ]]; then
  RUN_PREPARE_STAGE2="$ENV_RUN_PREPARE_STAGE2"
fi
if [[ "$ENV_RUN_SPLITS" != "__TF_UNSET__" ]]; then
  RUN_SPLITS="$ENV_RUN_SPLITS"
fi
if [[ "$ENV_RUN_STAGE1" != "__TF_UNSET__" ]]; then
  RUN_STAGE1="$ENV_RUN_STAGE1"
fi
if [[ "$ENV_RUN_STAGE2" != "__TF_UNSET__" ]]; then
  RUN_STAGE2="$ENV_RUN_STAGE2"
fi

: "${STAGE1_GPU_PT:=}"
: "${STAGE1_GPU_LN:=}"
if ! declare -p STAGE2_GPU_IDS >/dev/null 2>&1; then
  STAGE2_GPU_IDS=()
else
  case "$(declare -p STAGE2_GPU_IDS 2>/dev/null)" in
    "declare -a "*)
      ;;
    *)
      if [[ -n "${STAGE2_GPU_IDS:-}" ]]; then
        STAGE2_GPU_IDS=("$STAGE2_GPU_IDS")
      else
        STAGE2_GPU_IDS=()
      fi
      ;;
  esac
fi

cd "$WORKSPACE_ROOT"

export PYTHONPATH="$PACKAGE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
source "$PACKAGE_DIR/scripts/lib/gpu_utils.sh"

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

detect_available_gpus() {
  local -a ids=()
  mapfile -t ids < <(tf_detect_gpu_ids)
  if (( ${#ids[@]} == 0 )); then
    echo "[error] no GPUs detected. Set CUDA_VISIBLE_DEVICES or ensure nvidia-smi is available." >&2
    exit 1
  fi
  AVAILABLE_GPU_IDS=("${ids[@]}")
  echo "[pipeline] detected GPUs: ${AVAILABLE_GPU_IDS[*]}"
}

resolve_stage1_gpus() {
  RESOLVED_STAGE1_GPU_PT="${STAGE1_GPU_PT:-}"
  RESOLVED_STAGE1_GPU_LN="${STAGE1_GPU_LN:-}"

  if [[ -n "$RESOLVED_STAGE1_GPU_PT" && -n "$RESOLVED_STAGE1_GPU_LN" ]]; then
    return
  fi

  if [[ -n "$RESOLVED_STAGE1_GPU_PT" && -z "$RESOLVED_STAGE1_GPU_LN" ]]; then
    if mapfile -t AVAILABLE_GPU_IDS < <(tf_detect_gpu_ids); then
      if (( ${#AVAILABLE_GPU_IDS[@]} >= 2 )); then
        RESOLVED_STAGE1_GPU_LN="${AVAILABLE_GPU_IDS[1]}"
      else
        RESOLVED_STAGE1_GPU_LN="$RESOLVED_STAGE1_GPU_PT"
      fi
      return
    fi
    RESOLVED_STAGE1_GPU_LN="$RESOLVED_STAGE1_GPU_PT"
    return
  fi

  if [[ -z "$RESOLVED_STAGE1_GPU_PT" && -n "$RESOLVED_STAGE1_GPU_LN" ]]; then
    RESOLVED_STAGE1_GPU_PT="$RESOLVED_STAGE1_GPU_LN"
    return
  fi

  detect_available_gpus

  if [[ -z "$RESOLVED_STAGE1_GPU_PT" ]]; then
    RESOLVED_STAGE1_GPU_PT="${AVAILABLE_GPU_IDS[0]}"
  fi
  if [[ -z "$RESOLVED_STAGE1_GPU_LN" ]]; then
    if (( ${#AVAILABLE_GPU_IDS[@]} >= 2 )); then
      RESOLVED_STAGE1_GPU_LN="${AVAILABLE_GPU_IDS[1]}"
    else
      RESOLVED_STAGE1_GPU_LN="${AVAILABLE_GPU_IDS[0]}"
    fi
  fi
}

resolve_stage2_gpus() {
  local -a ids=()

  if (( ${#STAGE2_GPU_IDS[@]} > 0 )); then
    ids=()
    local gpu
    for gpu in "${STAGE2_GPU_IDS[@]}"; do
      if [[ -z "$gpu" ]]; then
        continue
      fi
      if [[ "$gpu" == *","* || "$gpu" == *" "* ]]; then
        local part
        for part in ${gpu//,/ }; do
          [[ -n "$part" ]] && ids+=("$part")
        done
      else
        ids+=("$gpu")
      fi
    done
  fi

  if (( ${#ids[@]} == 0 )); then
    detect_available_gpus
    ids=("${AVAILABLE_GPU_IDS[@]}")
  fi

  RESOLVED_STAGE2_GPU_IDS=("${ids[@]}")
}

maybe_enable_missing_prereqs() {
  local preprocess_meta="${PREPROCESS_OUT_ROOT}/${PREPROCESS_OUT_CSV}"
  local stage2_meta="${PREPARE_STAGE2_OUT_DIR}/${PREPARE_STAGE2_OUT_CSV}"

  if [[ "$RUN_PREPROCESS" != "1" ]]; then
    if [[ "$RUN_PREPARE_STAGE2" == "1" || "$RUN_STAGE1" == "1" ]]; then
      if [[ ! -f "$preprocess_meta" ]]; then
        echo "[pipeline] missing $preprocess_meta, enabling preprocessing"
        RUN_PREPROCESS=1
      fi
    fi
  fi

  if [[ "$RUN_PREPARE_STAGE2" != "1" ]]; then
    if [[ "$RUN_SPLITS" == "1" || "$RUN_STAGE2" == "1" ]]; then
      if [[ ! -f "$stage2_meta" ]]; then
        echo "[pipeline] missing $stage2_meta, enabling stage-2 metafile preparation"
        RUN_PREPARE_STAGE2=1
      fi
    fi
  fi

  if [[ "$RUN_SPLITS" != "1" && "$RUN_STAGE2" == "1" ]]; then
    if [[ ! -d "$SPLITS_OUT_DIR" ]]; then
      echo "[pipeline] missing $SPLITS_OUT_DIR, enabling split generation"
      RUN_SPLITS=1
    fi
  fi
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
  bash "$PACKAGE_DIR/scripts/run_preprocess_export_swinunetr.sh" "${PREPROCESS_EXTRA_ARGS[@]}"
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
  bash "$PACKAGE_DIR/scripts/run_make_cv_splits.sh" "${SPLITS_EXTRA_ARGS[@]}"
}

run_prepare_stage2() {
  echo "[pipeline] preparing stage-2 metafile -> $PREPARE_STAGE2_OUT_DIR/$PREPARE_STAGE2_OUT_CSV"
  BASE_META_CSV="$PREPARE_STAGE2_BASE_META_CSV" \
  SURV_CSV="$PREPARE_STAGE2_SURV_CSV" \
  CLIN_CSV="$PREPARE_STAGE2_CLIN_CSV" \
  RADIO_CSV="$PREPARE_STAGE2_RADIO_CSV" \
  OUT_DIR="$PREPARE_STAGE2_OUT_DIR" \
  OUT_CSV="$PREPARE_STAGE2_OUT_CSV" \
  bash "$PACKAGE_DIR/scripts/run_prepare_opscc_tabular.sh" "${PREPARE_STAGE2_EXTRA_ARGS[@]}"
}

run_stage1() {
  resolve_stage1_gpus
  echo "[pipeline] stage 1 PT/LN pretraining on GPUs $RESOLVED_STAGE1_GPU_PT and $RESOLVED_STAGE1_GPU_LN"

  META_CSV="$STAGE1_META_CSV" \
  OUT_DIR="$STAGE1_PT_OUT_DIR" \
  CUDA_DEVICE="$RESOLVED_STAGE1_GPU_PT" \
  DEVICE="$STAGE1_DEVICE" \
  bash "$PACKAGE_DIR/scripts/run_stage1_pretrain_pt.sh" \
    --epochs "$STAGE1_EPOCHS" \
    --batch_size "$STAGE1_BATCH_SIZE" \
    --workers "$STAGE1_WORKERS" \
    "${STAGE1_PT_EXTRA_ARGS[@]}" &
  pt_pid=$!

  if [[ "$RESOLVED_STAGE1_GPU_LN" == "$RESOLVED_STAGE1_GPU_PT" ]]; then
    wait_all_or_fail "$pt_pid"
    META_CSV="$STAGE1_META_CSV" \
    OUT_DIR="$STAGE1_LN_OUT_DIR" \
    CUDA_DEVICE="$RESOLVED_STAGE1_GPU_LN" \
    DEVICE="$STAGE1_DEVICE" \
    bash "$PACKAGE_DIR/scripts/run_stage1_pretrain_ln.sh" \
      --epochs "$STAGE1_EPOCHS" \
      --batch_size "$STAGE1_BATCH_SIZE" \
      --workers "$STAGE1_WORKERS" \
      "${STAGE1_LN_EXTRA_ARGS[@]}"
    return
  fi

  META_CSV="$STAGE1_META_CSV" \
  OUT_DIR="$STAGE1_LN_OUT_DIR" \
  CUDA_DEVICE="$RESOLVED_STAGE1_GPU_LN" \
  DEVICE="$STAGE1_DEVICE" \
  bash "$PACKAGE_DIR/scripts/run_stage1_pretrain_ln.sh" \
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
    RADIOMICS_SOURCE="$STAGE2_RADIOMICS_SOURCE" \
    OUT_DIR="$STAGE2_OUT_DIR" \
    EXP_NAME="$exp_name" \
    DEBUG_FOLD="$fold" \
    CUDA_DEVICE="$gpu" \
    DEVICE="$STAGE2_DEVICE" \
    bash "$PACKAGE_DIR/scripts/run_stage2_survival_lora.sh" "${extra_args[@]}"
  else
    META_CSV="$STAGE2_META_CSV" \
    SPLITS_DIR="$STAGE2_SPLITS_DIR" \
    PT_CKPT="$STAGE2_PT_CKPT" \
    LN_CKPT="$STAGE2_LN_CKPT" \
    RADIOMICS_SOURCE="$STAGE2_RADIOMICS_SOURCE" \
    OUT_DIR="$STAGE2_OUT_DIR" \
    EXP_NAME="$exp_name" \
    DEBUG_FOLD="$fold" \
    CUDA_DEVICE="$gpu" \
    DEVICE="$STAGE2_DEVICE" \
    bash "$PACKAGE_DIR/scripts/run_stage2_survival.sh" "${extra_args[@]}"
  fi
}

run_stage2() {
  if [[ ! -f "$STAGE2_META_CSV" ]]; then
    echo "[error] stage 2 meta csv not found: $STAGE2_META_CSV" >&2
    echo "[error] provide a prepared stage-2 metafile in the settings file." >&2
    exit 1
  fi
  if [[ ! -d "$STAGE2_SPLITS_DIR" ]]; then
    echo "[error] stage 2 splits dir not found: $STAGE2_SPLITS_DIR" >&2
    exit 1
  fi

  resolve_stage1_ckpts
  resolve_stage2_gpus

  local max_parallel="${#RESOLVED_STAGE2_GPU_IDS[@]}"

  local -a pids=()
  local launch_idx=0
  for fold in "${STAGE2_FOLDS[@]}"; do
    local gpu="${RESOLVED_STAGE2_GPU_IDS[$(( launch_idx % max_parallel ))]}"
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

maybe_enable_missing_prereqs

if [[ "$RUN_PREPROCESS" == "1" ]]; then
  run_preprocess
fi

if [[ "${RUN_PREPARE_STAGE2:-0}" == "1" ]]; then
  run_prepare_stage2
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
