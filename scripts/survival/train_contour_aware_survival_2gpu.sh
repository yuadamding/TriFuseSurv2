#!/usr/bin/env bash
set -euo pipefail
# Thin 2-GPU launcher: sets up CUDA_VISIBLE_DEVICES + --data_parallel,
# then forwards ALL training args from the caller.  Does NOT hardcode
# hyperparameters — those come from the invoking script or CLI.

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
CUDA_DEVICE_PAIR="${CUDA_DEVICE_PAIR:-${CUDA_DEVICE:-auto}}"
DEBUG_FOLD="${DEBUG_FOLD:-3}"
WORKERS="${WORKERS:-16}"
LOG_EVERY_BATCHES="${LOG_EVERY_BATCHES:-50}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
CONTOUR_WARMSTART_CKPT="${CONTOUR_WARMSTART_CKPT:-${SHARED_SEG_PRETRAIN_CKPT:-}}"
CONTOUR_WARMSTART_DIR="${CONTOUR_WARMSTART_DIR:-${SHARED_SEG_PRETRAIN_DIR:-}}"
CONTOUR_WARMSTART_NAME="${CONTOUR_WARMSTART_NAME:-${SHARED_SEG_PRETRAIN_NAME:-best.pt}}"

pick_auto_pair() {
  local -a ids=()
  mapfile -t ids < <(tf_detect_gpu_ids_by_free_mem "${MIN_FREE_GPU_MB:-1}")
  if (( ${#ids[@]} < 2 )); then
    mapfile -t ids < <(tf_detect_gpu_ids)
  fi
  if (( ${#ids[@]} < 2 )); then
    return 1
  fi
  printf '%s,%s\n' "${ids[0]}" "${ids[1]}"
}

if [[ "$CUDA_DEVICE_PAIR" == "auto" || -z "$CUDA_DEVICE_PAIR" ]]; then
  if ! CUDA_DEVICE_PAIR="$(pick_auto_pair)"; then
    echo "[error] could not detect at least two available GPUs for 2-GPU training." >&2
    exit 1
  fi
fi

IFS=',' read -r -a _visible_gpu_ids <<<"$CUDA_DEVICE_PAIR"
if (( ${#_visible_gpu_ids[@]} < 2 )); then
  echo "[error] CUDA_DEVICE_PAIR must contain at least two GPU ids, got: $CUDA_DEVICE_PAIR" >&2
  exit 1
fi

JOB_DEVICE="cuda:0"

extra_args=()
if [[ -n "$CONTOUR_WARMSTART_CKPT" ]]; then
  extra_args+=(--contour_warmstart_ckpt "$CONTOUR_WARMSTART_CKPT")
elif [[ -n "$CONTOUR_WARMSTART_DIR" ]]; then
  extra_args+=(--contour_warmstart_dir "$CONTOUR_WARMSTART_DIR" --contour_warmstart_name "$CONTOUR_WARMSTART_NAME")
else
  extra_args+=(--no_align_swin_cfg_from_contour_warmstart)
fi

echo "[train-2gpu] visible_gpus=$CUDA_DEVICE_PAIR device=$JOB_DEVICE data_parallel=1 debug_fold=$DEBUG_FOLD"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_PAIR" \
PYTORCH_ALLOC_CONF="$PYTORCH_ALLOC_CONF" \
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
  --workers "$WORKERS" \
  --log_every_batches "$LOG_EVERY_BATCHES" \
  --amp \
  --device "$JOB_DEVICE" \
  --data_parallel \
  --use_radiomics \
  --radiomics_root "$RADIOMICS_SOURCE" \
  "${extra_args[@]}" \
  "$@"
