#!/usr/bin/env bash
set -euo pipefail
# DDP launcher: uses torchrun for DistributedDataParallel across N GPUs.
# Much faster than DataParallel (~95% scaling vs ~60%).
#
# The caller specifies which GPUs to use via CUDA_DEVICE_PAIR (e.g. "0,1")
# or CUDA_VISIBLE_DEVICES.  torchrun is launched with CUDA_VISIBLE_DEVICES
# restricted to those GPUs so it cannot collide with other parallel jobs.
#
# Usage:
#   CUDA_DEVICE_PAIR=0,1 bash scripts/survival/train_contour_aware_survival_ddp.sh --epochs 60 ...

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
DEBUG_FOLD="${DEBUG_FOLD:-3}"
WORKERS="${WORKERS:-2}"
LOG_EVERY_BATCHES="${LOG_EVERY_BATCHES:-50}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
CONTOUR_WARMSTART_CKPT="${CONTOUR_WARMSTART_CKPT:-${SHARED_SEG_PRETRAIN_CKPT:-}}"
CONTOUR_WARMSTART_DIR="${CONTOUR_WARMSTART_DIR:-${SHARED_SEG_PRETRAIN_DIR:-}}"
CONTOUR_WARMSTART_NAME="${CONTOUR_WARMSTART_NAME:-${SHARED_SEG_PRETRAIN_NAME:-best.pt}}"

# Determine which GPUs this job should use.
# CUDA_DEVICE_PAIR is set by the parent search script (e.g. "2,3").
# We set CUDA_VISIBLE_DEVICES to exactly those GPUs so torchrun only sees them.
CUDA_DEVICE_PAIR="${CUDA_DEVICE_PAIR:-${CUDA_VISIBLE_DEVICES:-auto}}"

if [[ "$CUDA_DEVICE_PAIR" == "auto" || -z "$CUDA_DEVICE_PAIR" ]]; then
  pick_auto_pair() {
    local -a ids=()
    mapfile -t ids < <(tf_detect_gpu_ids_by_free_mem "${MIN_FREE_GPU_MB:-1}")
    if (( ${#ids[@]} < 2 )); then
      mapfile -t ids < <(tf_detect_gpu_ids)
    fi
    if (( ${#ids[@]} < 2 )); then return 1; fi
    printf '%s,%s\n' "${ids[0]}" "${ids[1]}"
  }
  if ! CUDA_DEVICE_PAIR="$(pick_auto_pair)"; then
    echo "[error] could not detect at least two available GPUs for DDP training." >&2
    exit 1
  fi
fi

IFS=',' read -r -a _gpu_ids <<<"$CUDA_DEVICE_PAIR"
NPROC="${#_gpu_ids[@]}"

if (( NPROC < 2 )); then
  echo "[error] DDP requires at least 2 GPUs, got CUDA_DEVICE_PAIR=$CUDA_DEVICE_PAIR" >&2
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

echo "[train-ddp] visible_gpus=$CUDA_DEVICE_PAIR nproc=$NPROC debug_fold=$DEBUG_FOLD"

# CRITICAL: set CUDA_VISIBLE_DEVICES so torchrun only sees the assigned GPUs.
# torchrun assigns LOCAL_RANK 0..N-1 which map to these visible devices.
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_PAIR" \
PYTORCH_ALLOC_CONF="$PYTORCH_ALLOC_CONF" \
PYTHONUNBUFFERED=1 \
torchrun --standalone --nproc_per_node="$NPROC" \
  -m trifusesurv2.multimodal_survival.train \
  --ddp \
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
  --use_radiomics \
  --radiomics_root "$RADIOMICS_SOURCE" \
  "${extra_args[@]}" \
  "$@"
