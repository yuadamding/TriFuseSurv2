#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$PACKAGE_DIR/.." && pwd)"
cd "$WORKSPACE_ROOT"

ENDPOINT="${ENDPOINT:-OS}"
ENDPOINT_LC="$(printf '%s' "$ENDPOINT" | tr '[:upper:]' '[:lower:]')"
DEBUG_FOLD="${DEBUG_FOLD:-${FOLD:-3}}"
OUT_DIR="${OUT_DIR:-runs/contour_aware_survival_${ENDPOINT_LC}}"
if [[ -z "${EXP_NAME:-}" ]]; then
  if [[ "$DEBUG_FOLD" =~ ^[0-9]+$ ]]; then
    EXP_NAME="$(printf 'cv4_contour_aware_%s_fold%02d' "$ENDPOINT_LC" "$((10#$DEBUG_FOLD))")"
  else
    EXP_NAME="cv4_contour_aware_${ENDPOINT_LC}"
  fi
fi

TRAIN_SCRIPT="${TRAIN_SCRIPT:-$PACKAGE_DIR/scripts/survival/train_contour_aware_survival.sh}"
WATCH_SCRIPT="${WATCH_SCRIPT:-$PACKAGE_DIR/scripts/survival/watch_roi_focus_training.sh}"
SKIP_REQUIREMENTS_CHECK="${SKIP_REQUIREMENTS_CHECK:-0}"
WATCH_INTERVAL_SECONDS="${WATCH_INTERVAL_SECONDS:-${INTERVAL_SECONDS:-60}}"
LATEST_N="${LATEST_N:-8}"
CHECK_LATEST_N="${CHECK_LATEST_N:-$LATEST_N}"
STRICT="${STRICT:-0}"
ROI_FOCUS_WARMUP_EPOCHS="${ROI_FOCUS_WARMUP_EPOCHS:-10}"
ROI_FOCUS_WARMUP_SURVIVAL_WEIGHT="${ROI_FOCUS_WARMUP_SURVIVAL_WEIGHT:-0.2}"

for ((arg_i = 1; arg_i <= $#; arg_i++)); do
  arg="${!arg_i}"
  case "$arg" in
    --roi_focus_warmup_epochs)
      next_i=$((arg_i + 1))
      if (( next_i <= $# )); then
        ROI_FOCUS_WARMUP_EPOCHS="${!next_i}"
      fi
      ;;
    --roi_focus_warmup_epochs=*)
      ROI_FOCUS_WARMUP_EPOCHS="${arg#*=}"
      ;;
    --roi_focus_warmup_survival_weight)
      next_i=$((arg_i + 1))
      if (( next_i <= $# )); then
        ROI_FOCUS_WARMUP_SURVIVAL_WEIGHT="${!next_i}"
      fi
      ;;
    --roi_focus_warmup_survival_weight=*)
      ROI_FOCUS_WARMUP_SURVIVAL_WEIGHT="${arg#*=}"
      ;;
  esac
done
export ROI_FOCUS_WARMUP_EPOCHS ROI_FOCUS_WARMUP_SURVIVAL_WEIGHT
WARMUP_EPOCHS="${WARMUP_EPOCHS:-$ROI_FOCUS_WARMUP_EPOCHS}"
MIN_PROB_MASS_INSIDE_GT="${MIN_PROB_MASS_INSIDE_GT:-0.95}"
MIN_SUPPORT_RECALL="${MIN_SUPPORT_RECALL:-0.95}"
MIN_SUPPORT_DICE="${MIN_SUPPORT_DICE:-0.02}"
MAX_EMPTY_WHEN_GT_PRESENT="${MAX_EMPTY_WHEN_GT_PRESENT:-0.25}"

if [[ ! -x "$TRAIN_SCRIPT" && ! -f "$TRAIN_SCRIPT" ]]; then
  echo "[train+roi-focus][FAIL] training script not found: $TRAIN_SCRIPT" >&2
  exit 1
fi
if [[ ! -x "$WATCH_SCRIPT" && ! -f "$WATCH_SCRIPT" ]]; then
  echo "[train+roi-focus][FAIL] ROI watcher not found: $WATCH_SCRIPT" >&2
  exit 1
fi

export PYTHONPATH="$PACKAGE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$SKIP_REQUIREMENTS_CHECK" != "1" && "$SKIP_REQUIREMENTS_CHECK" != "true" && "$SKIP_REQUIREMENTS_CHECK" != "yes" ]]; then
  source "$PACKAGE_DIR/scripts/lib/gpu_utils.sh"
  tf_require_python_modules numpy pandas SimpleITK torch monai sklearn pydicom
fi

mkdir -p "$OUT_DIR"
run_start_epoch="$(($(date +%s) - 2))"
watch_fold="$DEBUG_FOLD"
if [[ "$watch_fold" == "-1" ]]; then
  watch_fold=""
fi

echo "[train+roi-focus] workspace=$WORKSPACE_ROOT"
echo "[train+roi-focus] train_script=$TRAIN_SCRIPT"
echo "[train+roi-focus] out_dir=$OUT_DIR"
echo "[train+roi-focus] exp_name=$EXP_NAME"
echo "[train+roi-focus] debug_fold=$DEBUG_FOLD"
echo "[train+roi-focus] roi_focus_warmup_epochs=$ROI_FOCUS_WARMUP_EPOCHS watcher_warmup_epochs=$WARMUP_EPOCHS"
echo "[train+roi-focus] watcher: OUT_DIR=$OUT_DIR FOLD=${watch_fold:-<any>} MIN_MTIME_EPOCH=$run_start_epoch"

watch_pid=""
train_pid=""
cleanup_watch() {
  if [[ -n "$watch_pid" ]]; then
    kill "$watch_pid" >/dev/null 2>&1 || true
    wait "$watch_pid" >/dev/null 2>&1 || true
    watch_pid=""
  fi
}
cleanup_train() {
  if [[ -n "$train_pid" ]]; then
    kill "$train_pid" >/dev/null 2>&1 || true
    wait "$train_pid" >/dev/null 2>&1 || true
    train_pid=""
  fi
}
is_running_job() {
  local target_pid="$1"
  local job_pid
  while IFS= read -r job_pid; do
    if [[ "$job_pid" == "$target_pid" ]]; then
      return 0
    fi
  done < <(jobs -r -p)
  return 1
}
cleanup_all() {
  cleanup_watch
  cleanup_train
}
trap cleanup_all EXIT INT TERM

OUT_DIR="$OUT_DIR" \
FOLD="$watch_fold" \
WATCH=1 \
INTERVAL_SECONDS="$WATCH_INTERVAL_SECONDS" \
LATEST_N="$LATEST_N" \
CHECK_LATEST_N="$CHECK_LATEST_N" \
STRICT="$STRICT" \
WARMUP_EPOCHS="$WARMUP_EPOCHS" \
MIN_MTIME_EPOCH="$run_start_epoch" \
MIN_PROB_MASS_INSIDE_GT="$MIN_PROB_MASS_INSIDE_GT" \
MIN_SUPPORT_RECALL="$MIN_SUPPORT_RECALL" \
MIN_SUPPORT_DICE="$MIN_SUPPORT_DICE" \
MAX_EMPTY_WHEN_GT_PRESENT="$MAX_EMPTY_WHEN_GT_PRESENT" \
bash "$WATCH_SCRIPT" &
watch_pid="$!"

set +e
ENDPOINT="$ENDPOINT" \
DEBUG_FOLD="$DEBUG_FOLD" \
OUT_DIR="$OUT_DIR" \
EXP_NAME="$EXP_NAME" \
bash "$TRAIN_SCRIPT" "$@" &
train_pid="$!"
train_rc=0
watch_rc=0
while true; do
  if [[ -n "$watch_pid" ]] && ! is_running_job "$watch_pid"; then
    wait "$watch_pid"
    watch_rc="$?"
    watch_pid=""
    if [[ "$watch_rc" != "0" ]]; then
      echo "[train+roi-focus][FAIL] ROI-focus watcher exited with rc=$watch_rc; stopping training." >&2
      cleanup_train
      train_rc="$watch_rc"
      break
    fi
  fi
  if [[ -n "$train_pid" ]] && ! is_running_job "$train_pid"; then
    wait "$train_pid"
    train_rc="$?"
    train_pid=""
    break
  fi
  sleep 5
done
set -e

cleanup_watch

echo "[train+roi-focus] final ROI-focus snapshot"
set +e
OUT_DIR="$OUT_DIR" \
FOLD="$watch_fold" \
WATCH=0 \
LATEST_N="$LATEST_N" \
CHECK_LATEST_N="$CHECK_LATEST_N" \
STRICT="$STRICT" \
WARMUP_EPOCHS="$WARMUP_EPOCHS" \
MIN_MTIME_EPOCH="$run_start_epoch" \
MIN_PROB_MASS_INSIDE_GT="$MIN_PROB_MASS_INSIDE_GT" \
MIN_SUPPORT_RECALL="$MIN_SUPPORT_RECALL" \
MIN_SUPPORT_DICE="$MIN_SUPPORT_DICE" \
MAX_EMPTY_WHEN_GT_PRESENT="$MAX_EMPTY_WHEN_GT_PRESENT" \
bash "$WATCH_SCRIPT"
focus_rc="$?"
set -e

if [[ "$train_rc" != "0" ]]; then
  exit "$train_rc"
fi
exit "$focus_rc"
