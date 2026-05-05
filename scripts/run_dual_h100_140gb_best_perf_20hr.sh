#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-60}"
TOTAL_WORKERS_PER_FOLD="${TOTAL_WORKERS_PER_FOLD:-16}"
RESUME="${RESUME:-1}"
LIGHTWEIGHT_CHECKPOINTS="${LIGHTWEIGHT_CHECKPOINTS:-}"
# This script is specifically for dual-H100 (~140 GB total) runs.
# Default to 2-GPU DistributedDataParallel for better scaling/utilization.
# DataParallel and fold_per_gpu remain available as explicit overrides.
MULTI_GPU_MODE="${MULTI_GPU_MODE:-ddp}"
if [[ "$MULTI_GPU_MODE" != "dp" && "$MULTI_GPU_MODE" != "fold_per_gpu" && "$MULTI_GPU_MODE" != "ddp" ]]; then
  echo "[error] MULTI_GPU_MODE must be one of: dp, fold_per_gpu, ddp. Got: $MULTI_GPU_MODE" >&2
  exit 1
fi
if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
  echo "[error] RESUME must be 0 or 1. Got: $RESUME" >&2
  exit 1
fi
if [[ -z "$LIGHTWEIGHT_CHECKPOINTS" ]]; then
  if [[ "$RESUME" == "1" ]]; then
    LIGHTWEIGHT_CHECKPOINTS="0"
  else
    LIGHTWEIGHT_CHECKPOINTS="1"
  fi
fi
if [[ "$LIGHTWEIGHT_CHECKPOINTS" != "0" && "$LIGHTWEIGHT_CHECKPOINTS" != "1" ]]; then
  echo "[error] LIGHTWEIGHT_CHECKPOINTS must be 0 or 1. Got: $LIGHTWEIGHT_CHECKPOINTS" >&2
  exit 1
fi
if [[ "$RESUME" == "1" && "$LIGHTWEIGHT_CHECKPOINTS" == "1" ]]; then
  echo "[error] RESUME=1 requires LIGHTWEIGHT_CHECKPOINTS=0 so full optimizer/scheduler state is saved." >&2
  exit 1
fi
if [[ -z "${OUT_DIR:-}" ]]; then
  OUT_DIR="runs/dual_h100_140gb_best_perf_20hr_${ENDPOINT_LC}_${MULTI_GPU_MODE}"
fi

compute_workers_per_loader() {
  local total_budget="${1:-16}"
  local mode="$2"
  local loaders_per_rank=4
  local ranks_per_fold=1
  if [[ "$mode" == "ddp" ]]; then
    ranks_per_fold=2
  fi
  local denom=$(( loaders_per_rank * ranks_per_fold ))
  local workers=$(( total_budget / denom ))
  if (( workers < 1 )); then
    workers=1
  fi
  printf '%s\n' "$workers"
}

if [[ -z "${WORKERS:-}" ]]; then
  WORKERS="$(compute_workers_per_loader "$TOTAL_WORKERS_PER_FOLD" "$MULTI_GPU_MODE")"
fi

if [[ "$MULTI_GPU_MODE" == "dp" ]]; then
  # DataParallel: batch_size is global across the paired GPUs.
  BATCH_SIZE="${BATCH_SIZE:-2}"
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
elif [[ "$MULTI_GPU_MODE" == "ddp" ]]; then
  # DDP: batch_size is per-rank (per GPU). Keep 1 per rank by default so the
  # high-memory profile comes primarily from disabling activation checkpointing
  # rather than from doubling the Swin backbone activation volume and OOMing.
  BATCH_SIZE="${BATCH_SIZE:-1}"
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
else
  # fold_per_gpu: 1 GPU per fold for debugging/throughput only.
  BATCH_SIZE="${BATCH_SIZE:-1}"
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
fi
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-78000}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}"
FAIL_LOG_TAIL_LINES="${FAIL_LOG_TAIL_LINES:-120}"
SKIP_FINISHED="${SKIP_FINISHED:-1}"
MAX_PARALLEL_PAIRS="${MAX_PARALLEL_PAIRS:-0}"

compute_package_code_signature() {
  local root="$1"
  python3 - "$root" <<'PY'
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
scan_roots = [root / "src" / "trifusesurv2", root / "scripts"]

files = []
for scan_root in scan_roots:
    if not scan_root.exists():
        continue
    for path in scan_root.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        files.append(path)

hasher = hashlib.sha256()
for path in sorted(files):
    rel = path.relative_to(root).as_posix().encode("utf-8")
    hasher.update(rel + b"\0")
    hasher.update(path.read_bytes())
    hasher.update(b"\0")

print(hasher.hexdigest())
PY
}

PACKAGE_CODE_SIGNATURE="$(compute_package_code_signature "$PACKAGE_DIR")"

normalize_list_var() {
  local var_name="$1"
  local default_words="$2"
  local decl
  local -a values=()
  if decl="$(declare -p "$var_name" 2>/dev/null)" && [[ "$decl" == declare\ -a* ]]; then
    eval "values=(\"\${${var_name}[@]}\")"
  else
    local raw
    raw="$(eval "printf '%s' \"\${${var_name}:-${default_words}}\"")"
    read -r -a values <<< "$raw"
  fi
  eval "${var_name}=()"
  local item
  for item in "${values[@]}"; do
    eval "${var_name}+=(\"\$item\")"
  done
}

normalize_list_var FOLDS "0 1 2 3"
normalize_list_var WEIGHTS_TO_SCORE "best ema swa last"
normalize_list_var TRIALS "dual140_eqtf22_lrbb2e4_f2048 dual140_eqtf23_lrbb2e4_f2048 dual140_eqtf24_lrbb2e4_f2048 dual140_eqtf23_lrbb16e4_f2048 dual140_eqtf23_lrbb2e4_f2304"

SUMMARY_CSV="$OUT_DIR/tuning_summary.csv"
SUMMARY_RANKED_CSV="$OUT_DIR/tuning_summary_ranked.csv"
FAIL_CSV="$OUT_DIR/failed_trials.csv"
LOG_DIR="$OUT_DIR/logs"

ensure_required_inputs() {
  local path
  for path in "$META_CSV" "$RADIOMICS_SOURCE"; do
    if [[ ! -f "$path" ]]; then
      echo "[error] required file not found: $path" >&2
      exit 1
    fi
  done
  if [[ ! -d "$SPLITS_DIR" ]]; then
    echo "[error] splits dir not found: $SPLITS_DIR" >&2
    exit 1
  fi
  if (( ${#TRIALS[@]} == 0 )); then
    echo "[error] no trials configured" >&2
    exit 1
  fi
}

detect_available_gpus() {
  local -a ids=()
  mapfile -t ids < <(tf_detect_gpu_ids_by_free_mem "$MIN_FREE_GPU_MB")

  if [[ "$MULTI_GPU_MODE" == "fold_per_gpu" ]]; then
    # Each GPU runs one fold independently — maximum throughput.
    if (( ${#ids[@]} < 1 )); then
      echo "[error] need at least 1 GPU with >=${MIN_FREE_GPU_MB} MB free; found 0" >&2
      exit 1
    fi
    AVAILABLE_GPU_PAIRS=()
    for gid in "${ids[@]}"; do
      AVAILABLE_GPU_PAIRS+=("$gid")
    done
    local want="${MAX_PARALLEL_PAIRS}"
    if [[ ! "$want" =~ ^[0-9]+$ ]] || (( want <= 0 || want > ${#AVAILABLE_GPU_PAIRS[@]} )); then
      want="${#AVAILABLE_GPU_PAIRS[@]}"
    fi
    AVAILABLE_GPU_PAIRS=("${AVAILABLE_GPU_PAIRS[@]:0:$want}")
    echo "[dual140] fold-per-gpu mode: ${#AVAILABLE_GPU_PAIRS[@]} GPUs available (${AVAILABLE_GPU_PAIRS[*]})"
  else
    # Multi-GPU fold mode: pair GPUs for 2-GPU-per-fold (DP or DDP).
    if (( ${#ids[@]} < 2 )); then
      echo "[error] need at least 2 GPUs with >=${MIN_FREE_GPU_MB} MB free; found ${#ids[@]}" >&2
      exit 1
    fi
    local -a pairs=()
    local i=0
    while (( i + 1 < ${#ids[@]} )); do
      pairs+=("${ids[$i]},${ids[$((i + 1))]}")
      i=$(( i + 2 ))
    done
    if (( ${#pairs[@]} == 0 )); then
      echo "[error] could not form any 2-GPU pairs from detected GPUs: ${ids[*]}" >&2
      exit 1
    fi
    local want="${MAX_PARALLEL_PAIRS}"
    if [[ ! "$want" =~ ^[0-9]+$ ]] || (( want <= 0 || want > ${#pairs[@]} )); then
      want="${#pairs[@]}"
    fi
    AVAILABLE_GPU_PAIRS=("${pairs[@]:0:$want}")
    if [[ "$MULTI_GPU_MODE" == "ddp" ]]; then
      echo "[dual140] DDP mode: ${#AVAILABLE_GPU_PAIRS[@]} GPU pairs (${AVAILABLE_GPU_PAIRS[*]})"
    else
      echo "[dual140] DataParallel mode: ${#AVAILABLE_GPU_PAIRS[@]} GPU pairs (${AVAILABLE_GPU_PAIRS[*]})"
    fi
  fi
}

append_summary_row() {
  local summary_json="$1"
  local trial="$2"
  python3 - "$summary_json" "$trial" "$SUMMARY_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
trial = sys.argv[2]
summary_csv = Path(sys.argv[3])
data = json.loads(summary_path.read_text())
row = {
    "trial": trial,
    "weight": data["weights"],
    "c_index": data["c_index"],
    "n_predictions": data["n_predictions"],
    "n_evaluable": data["n_evaluable"],
    "n_risk_files": data["n_risk_files"],
}
write_header = not summary_csv.exists()
with summary_csv.open("a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
        w.writeheader()
    w.writerow(row)
print(f"[dual140] {trial} weight={row['weight']} OOF c-index={row['c_index']:.4f}")
PY
}

append_failure_row() {
  local trial="$1"
  local fold="$2"
  local log_file="$3"
  local status="$4"
  python3 - "$FAIL_CSV" "$trial" "$fold" "$log_file" "$status" <<'PY'
import csv
import sys
from pathlib import Path

fail_csv = Path(sys.argv[1])
row = {
    "trial": sys.argv[2],
    "fold": int(sys.argv[3]),
    "log_file": sys.argv[4],
    "exit_status": int(sys.argv[5]),
}
write_header = not fail_csv.exists()
with fail_csv.open("a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
        w.writeheader()
    w.writerow(row)
PY
}

write_ranked_summary() {
  python3 - "$SUMMARY_CSV" "$SUMMARY_RANKED_CSV" <<'PY'
import csv
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
ranked_csv = Path(sys.argv[2])
if not summary_csv.exists():
    raise SystemExit(0)

with summary_csv.open(newline="") as f:
    rows = list(csv.DictReader(f))

def sort_key(row):
    try:
        value = float(row["c_index"])
    except Exception:
        return float("-inf")
    return value if value == value else float("-inf")

rows.sort(key=sort_key, reverse=True)

with ranked_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
    if rows:
        w.writeheader()
        w.writerows(rows)
PY
}

print_failure_log_tail() {
  local log_file="$1"
  local lines="${2:-120}"
  if [[ -f "$log_file" ]]; then
    echo "[dual140] ---- tail -n ${lines} ${log_file} ----" >&2
    tail -n "$lines" "$log_file" >&2 || true
    echo "[dual140] ---- end log tail ----" >&2
  else
    echo "[dual140][warn] missing failure log: $log_file" >&2
  fi
}

fold_export_dir() {
  local trial="$1"
  local fold="$2"
  printf '%s/%s_fold%02d/fold_%02d' "$OUT_DIR/$trial" "$trial" "$fold" "$fold"
}

fold_last_checkpoint() {
  local trial="$1"
  local fold="$2"
  printf '%s/last.pt' "$(fold_export_dir "$trial" "$fold")"
}

fold_has_complete_risks() {
  local trial="$1"
  local fold="$2"
  local weight
  local fold_dir
  fold_dir="$(fold_export_dir "$trial" "$fold")"
  for weight in "${WEIGHTS_TO_SCORE[@]}"; do
    [[ -f "$fold_dir/test_risks_${weight}.csv" ]] || return 1
  done
  return 0
}

trial_signature_file() {
  local trial="$1"
  printf '%s/.trial_signature.txt' "$OUT_DIR/$trial"
}

trial_signature_matches() {
  local trial="$1"
  local sig_file
  sig_file="$(trial_signature_file "$trial")"
  [[ -f "$sig_file" ]] || return 1
  cmp -s <(printf '%s' "$TRIAL_SIGNATURE") "$sig_file"
}

write_trial_signature() {
  local trial="$1"
  local sig_file
  sig_file="$(trial_signature_file "$trial")"
  printf '%s' "$TRIAL_SIGNATURE" > "$sig_file"
}

configure_trial() {
  local trial="$1"

  local fused_dim="1536"
  local token_dim="2048"
  local token_mlp_hidden_dim="3584"
  local img_proj_hidden_dim="3584"
  local img_tok_ffn_hidden_dim="3584"
  local img_post_hidden_dim="3584"
  local img_attn_heads="8"
  local gate_hidden_dim="2048"
  local rad_hidden_dim="2560"
  local pca="32"
  local time_bin_width="180"
  local risk_horizon="1095"
  local pt_shell_radius="5"
  local ln_shell_radius="5"
  local teacher_force_epochs="44"
  local teacher_force_start="1.0"
  local teacher_force_end="0.0"
  local aux_surv_loss_weight="0.35"
  local loc_loss_pt_lambda="0.25"
  local loc_loss_ln_lambda="0.25"
  local loc_presence_lambda="0.05"
  local lr_backbone="2e-4"
  local lr_head="7e-5"
  local checkpoint_flag="--no_use_checkpoint"
  local trial_batch_size="$BATCH_SIZE"
  local use_multiscale="1"

  case "$trial" in
    dual140_eqtf22_lrbb2e4_f2048)
      fused_dim="2048"
      token_dim="2560"
      token_mlp_hidden_dim="4608"
      img_proj_hidden_dim="4608"
      img_tok_ffn_hidden_dim="4608"
      img_post_hidden_dim="4608"
      img_attn_heads="16"
      gate_hidden_dim="3584"
      rad_hidden_dim="3584"
      ;;
    dual140_eqtf23_lrbb2e4_f2048)
      teacher_force_epochs="46"
      fused_dim="2048"
      token_dim="2560"
      token_mlp_hidden_dim="4608"
      img_proj_hidden_dim="4608"
      img_tok_ffn_hidden_dim="4608"
      img_post_hidden_dim="4608"
      img_attn_heads="16"
      gate_hidden_dim="3584"
      rad_hidden_dim="3584"
      ;;
    dual140_eqtf24_lrbb2e4_f2048)
      teacher_force_epochs="48"
      fused_dim="2048"
      token_dim="2560"
      token_mlp_hidden_dim="4608"
      img_proj_hidden_dim="4608"
      img_tok_ffn_hidden_dim="4608"
      img_post_hidden_dim="4608"
      img_attn_heads="16"
      gate_hidden_dim="3584"
      rad_hidden_dim="3584"
      ;;
    dual140_eqtf23_lrbb16e4_f2048)
      teacher_force_epochs="46"
      fused_dim="2048"
      token_dim="2560"
      token_mlp_hidden_dim="4608"
      img_proj_hidden_dim="4608"
      img_tok_ffn_hidden_dim="4608"
      img_post_hidden_dim="4608"
      img_attn_heads="16"
      gate_hidden_dim="3584"
      rad_hidden_dim="3584"
      lr_backbone="1.6e-4"
      lr_head="6e-5"
      ;;
    dual140_eqtf23_lrbb2e4_f2304)
      teacher_force_epochs="46"
      fused_dim="2304"
      token_dim="2816"
      token_mlp_hidden_dim="5120"
      img_proj_hidden_dim="5120"
      img_tok_ffn_hidden_dim="5120"
      img_post_hidden_dim="5120"
      img_attn_heads="16"
      gate_hidden_dim="4096"
      rad_hidden_dim="4096"
      ;;
    dual140_div_anchor_tf23_lrbb2e4_f2048)
      teacher_force_epochs="46"
      fused_dim="2048"
      token_dim="2560"
      token_mlp_hidden_dim="4608"
      img_proj_hidden_dim="4608"
      img_tok_ffn_hidden_dim="4608"
      img_post_hidden_dim="4608"
      img_attn_heads="16"
      gate_hidden_dim="3584"
      rad_hidden_dim="3584"
      ;;
    dual140_div_wide_tf23_lrbb2e4_f2304)
      teacher_force_epochs="46"
      fused_dim="2304"
      token_dim="2816"
      token_mlp_hidden_dim="5120"
      img_proj_hidden_dim="5120"
      img_tok_ffn_hidden_dim="5120"
      img_post_hidden_dim="5120"
      img_attn_heads="16"
      gate_hidden_dim="4096"
      rad_hidden_dim="4096"
      ;;
    dual140_div_nomulti_tf23_lrbb2e4_f2048)
      teacher_force_epochs="46"
      fused_dim="2048"
      token_dim="2560"
      token_mlp_hidden_dim="4608"
      img_proj_hidden_dim="4608"
      img_tok_ffn_hidden_dim="4608"
      img_post_hidden_dim="4608"
      img_attn_heads="16"
      gate_hidden_dim="3584"
      rad_hidden_dim="3584"
      use_multiscale="0"
      ;;
    dual140_div_shell3_tf23_lrbb2e4_f2048)
      teacher_force_epochs="46"
      fused_dim="2048"
      token_dim="2560"
      token_mlp_hidden_dim="4608"
      img_proj_hidden_dim="4608"
      img_tok_ffn_hidden_dim="4608"
      img_post_hidden_dim="4608"
      img_attn_heads="16"
      gate_hidden_dim="3584"
      rad_hidden_dim="3584"
      pt_shell_radius="3"
      ln_shell_radius="3"
      ;;
    dual140_div_aux020_tf23_lrbb2e4_f2048)
      teacher_force_epochs="46"
      fused_dim="2048"
      token_dim="2560"
      token_mlp_hidden_dim="4608"
      img_proj_hidden_dim="4608"
      img_tok_ffn_hidden_dim="4608"
      img_post_hidden_dim="4608"
      img_attn_heads="16"
      gate_hidden_dim="3584"
      rad_hidden_dim="3584"
      aux_surv_loss_weight="0.20"
      ;;
    dual140_div_time120_tf23_lrbb2e4_f2048)
      teacher_force_epochs="46"
      fused_dim="2048"
      token_dim="2560"
      token_mlp_hidden_dim="4608"
      img_proj_hidden_dim="4608"
      img_tok_ffn_hidden_dim="4608"
      img_post_hidden_dim="4608"
      img_attn_heads="16"
      gate_hidden_dim="3584"
      rad_hidden_dim="3584"
      time_bin_width="120"
      ;;
    *)
      echo "[error] unknown dual140 trial: $trial" >&2
      exit 1
      ;;
  esac

  if [[ "$MULTI_GPU_MODE" == "fold_per_gpu" ]]; then
    TRIAL_WRAPPER="$PACKAGE_DIR/scripts/survival/train_contour_aware_survival.sh"
  elif [[ "$MULTI_GPU_MODE" == "dp" ]]; then
    TRIAL_WRAPPER="$PACKAGE_DIR/scripts/survival/train_contour_aware_survival_2gpu.sh"
  else
    TRIAL_WRAPPER="$PACKAGE_DIR/scripts/survival/train_contour_aware_survival_ddp.sh"
  fi

  local resume_flag="--no_resume"
  local checkpoint_state_flag="--lightweight_checkpoints"
  if [[ "$RESUME" == "1" ]]; then
    resume_flag="--resume"
  fi
  if [[ "$LIGHTWEIGHT_CHECKPOINTS" == "0" ]]; then
    checkpoint_state_flag="--no_lightweight_checkpoints"
  fi

TRIAL_SIGNATURE="$(cat <<EOF
trial=$trial
mode=$MULTI_GPU_MODE
code_signature=$PACKAGE_CODE_SIGNATURE
epochs=$EPOCHS
resume=$RESUME
lightweight_checkpoints=$LIGHTWEIGHT_CHECKPOINTS
total_workers_per_fold=$TOTAL_WORKERS_PER_FOLD
batch_size=$BATCH_SIZE
trial_batch_size=$trial_batch_size
grad_accumulation_steps=$GRAD_ACCUM_STEPS
workers=$WORKERS
checkpoint_flag=$checkpoint_flag
fused_dim=$fused_dim
img_token_dim=$token_dim
token_mlp_hidden_dim=$token_mlp_hidden_dim
img_proj_hidden_dim=$img_proj_hidden_dim
img_tok_ffn_hidden_dim=$img_tok_ffn_hidden_dim
img_post_hidden_dim=$img_post_hidden_dim
img_attn_heads=$img_attn_heads
gate_hidden_dim=$gate_hidden_dim
rad_hidden_dim=$rad_hidden_dim
radiomics_pca_total_components=$pca
time_bin_width_days=$time_bin_width
risk_horizon_days=$risk_horizon
pt_shell_radius=$pt_shell_radius
ln_shell_radius=$ln_shell_radius
use_multiscale=$use_multiscale
teacher_force_epochs=$teacher_force_epochs
aux_surv_loss_weight=$aux_surv_loss_weight
loc_loss_pt_lambda=$loc_loss_pt_lambda
loc_loss_ln_lambda=$loc_loss_ln_lambda
loc_presence_lambda=$loc_presence_lambda
lr_backbone=$lr_backbone
lr_head=$lr_head
EOF
)"

  TRIAL_ARGS=(
    --epochs "$EPOCHS"
    --batch_size "$trial_batch_size"
    --grad_accumulation_steps "$GRAD_ACCUM_STEPS"
    --workers "$WORKERS"
    "$resume_flag"
    "$checkpoint_state_flag"
    --report_metric c_index
    --use_ema
    --use_swa
    --export_extra_risks
    "$checkpoint_flag"
    --radiomics_pca_total_components "$pca"
    --time_bin_width_days "$time_bin_width"
    --risk_horizon_days "$risk_horizon"
    --primary_surv_loss_weight 1.0
    --aux_surv_loss_weight "$aux_surv_loss_weight"
    --ema_decay 0.9995
    --swa_start_epoch 8
    --swa_update_freq_epochs 1
    --pt_shell_radius "$pt_shell_radius"
    --ln_shell_radius "$ln_shell_radius"
    --fused_dim "$fused_dim"
    --img_token_dim "$token_dim"
    --token_mlp_hidden_dim "$token_mlp_hidden_dim"
    --img_proj_hidden_dim "$img_proj_hidden_dim"
    --img_tok_ffn_hidden_dim "$img_tok_ffn_hidden_dim"
    --img_post_hidden_dim "$img_post_hidden_dim"
    --img_attn_heads "$img_attn_heads"
    --gate_hidden_dim "$gate_hidden_dim"
    --rad_hidden_dim "$rad_hidden_dim"
    --lr_backbone "$lr_backbone"
    --lr_head "$lr_head"
    --wd_rad 1e-3
    --modality_dropout_clin_p 0.00
    --modality_dropout_rad_p 0.05
    --clinical_noise_std 0.0
    --radiomics_noise_std 0.0
    --gate_dropout_p 0.05
    --surv_dropout_p 0.10
    --rad_proj_dropout_p 0.05
    --proj_dropout_p 0.10
    --expert_dropout_p 0.00
    --token_mlp_dropout 0.10
    --token_dropout 0.02
    --attn_dropout_p 0.02
    --gate_entropy_lambda 0.001
    --gate_loadbal_lambda 0.001
    --hazard_smooth_lambda 0.001
    --teacher_force_epochs "$teacher_force_epochs"
    --teacher_force_start "$teacher_force_start"
    --teacher_force_end "$teacher_force_end"
    --loc_loss_pt_lambda "$loc_loss_pt_lambda"
    --loc_loss_ln_lambda "$loc_loss_ln_lambda"
    --loc_presence_lambda "$loc_presence_lambda"
    --shell_body_from_ct
  )

  if [[ "$use_multiscale" == "1" ]]; then
    TRIAL_ARGS+=(--use_multiscale)
  fi
}

score_trial_weights() {
  local trial="$1"
  local trial_dir="$OUT_DIR/$trial"
  local weight

  for weight in "${WEIGHTS_TO_SCORE[@]}"; do
    local -a matches=()
    while IFS= read -r path; do
      matches+=("$path")
    done < <(find "$trial_dir" -path "*/test_risks_${weight}.csv" -type f | sort)

    if (( ${#matches[@]} == 0 )); then
      echo "[dual140] trial=$trial weight=$weight -> no matching risk files, skipping"
      continue
    fi
    if (( ${#matches[@]} != ${#FOLDS[@]} )); then
      echo "[dual140] trial=$trial weight=$weight -> found ${#matches[@]} risk files for ${#FOLDS[@]} folds, skipping incomplete OOF score"
      continue
    fi

    local summary_json="$trial_dir/oof_${weight}_summary.json"
    local pred_csv="$trial_dir/oof_${weight}_predictions.csv"
    META_CSV="$META_CSV" \
    ENDPOINT="$ENDPOINT" \
    WEIGHTS="$weight" \
    TRIAL_ROOT="$trial_dir" \
    EXP_PREFIX="$trial" \
    OUT_JSON="$summary_json" \
    OUT_CSV="$pred_csv" \
    bash "$PACKAGE_DIR/scripts/survival/evaluate_oof_cindex.sh"
    append_summary_row "$summary_json" "$trial"
  done
}

run_trial() {
  local trial="$1"
  configure_trial "$trial"

  local trial_dir="$OUT_DIR/$trial"
  if [[ "$SKIP_FINISHED" != "1" ]]; then
    rm -rf "$trial_dir"
  fi
  mkdir -p "$trial_dir"

  if [[ "$SKIP_FINISHED" == "1" ]]; then
    if ! trial_signature_matches "$trial"; then
      if [[ -e "$(trial_signature_file "$trial")" ]] || find "$trial_dir" -mindepth 1 -print -quit | grep -q .; then
        echo "[dual140][warn] trial=$trial has existing outputs from an incompatible run signature; clearing stale outputs before rerun"
        rm -rf "$trial_dir"
        mkdir -p "$trial_dir"
      fi
    fi
  fi
  write_trial_signature "$trial"

  local -a pids=()
  local -a running_pairs=()
  local -a running_folds=()
  local -a running_logs=()
  local -a free_pairs=("${AVAILABLE_GPU_PAIRS[@]}")
  local -a pending_folds=()
  local fold
  for fold in "${FOLDS[@]}"; do
    if [[ "$SKIP_FINISHED" == "1" ]] && fold_has_complete_risks "$trial" "$fold"; then
      echo "[dual140] trial=$trial fold=$fold -> existing completed risk exports found, skipping training"
      continue
    fi
    pending_folds+=("$fold")
  done

  if (( ${#pending_folds[@]} == 0 )); then
    echo "[dual140] trial=$trial -> all requested folds already finished; scoring existing outputs"
    score_trial_weights "$trial"
    return 0
  fi

  local total_folds="${#pending_folds[@]}"
  local next_fold_idx=0
  local trial_failed=0

  while (( next_fold_idx < total_folds || ${#pids[@]} > 0 )); do
    while (( trial_failed == 0 && next_fold_idx < total_folds && ${#free_pairs[@]} > 0 )); do
      local fold="${pending_folds[$next_fold_idx]}"
      local gpu_pair="${free_pairs[0]}"
      local exp_name="${trial}_fold$(printf '%02d' "$fold")"
      local exp_dir="$trial_dir/$exp_name"
      local last_ckpt
      last_ckpt="$(fold_last_checkpoint "$trial" "$fold")"
      local log_file="$LOG_DIR/${trial}_fold$(printf '%02d' "$fold").log"
      if [[ "$RESUME" == "1" && -f "$last_ckpt" ]]; then
        echo "[dual140] trial=$trial fold=$fold -> resuming from $last_ckpt"
      else
        rm -rf "$exp_dir"
      fi
      free_pairs=("${free_pairs[@]:1}")

      echo "[dual140] trial=$trial fold=$fold gpus=$gpu_pair mode=$MULTI_GPU_MODE log=$log_file"
      META_CSV="$META_CSV" \
      SPLITS_DIR="$SPLITS_DIR" \
      RADIOMICS_SOURCE="$RADIOMICS_SOURCE" \
      ENDPOINT="$ENDPOINT" \
      OUT_DIR="$trial_dir" \
      EXP_NAME="$exp_name" \
      DEBUG_FOLD="$fold" \
      CUDA_VISIBLE_DEVICES="$gpu_pair" \
      CUDA_DEVICE_PAIR="$gpu_pair" \
      DEVICE="cuda:0" \
      PYTORCH_ALLOC_CONF="$PYTORCH_ALLOC_CONF" \
      bash "$TRIAL_WRAPPER" "${TRIAL_ARGS[@]}" >"$log_file" 2>&1 &

      pids+=("$!")
      running_pairs+=("$gpu_pair")
      running_folds+=("$fold")
      running_logs+=("$log_file")
      next_fold_idx=$(( next_fold_idx + 1 ))
    done

    if (( ${#pids[@]} == 0 )); then
      break
    fi

    tf_wait_for_any_tracked_pid pids running_pairs running_folds running_logs
    free_pairs+=("$TF_WAIT_META1")

    if (( TF_WAIT_STATUS != 0 )); then
      trial_failed=1
      echo "[dual140][warn] trial=$trial fold=$TF_WAIT_META2 failed with status=$TF_WAIT_STATUS; see $TF_WAIT_META3" >&2
      append_failure_row "$trial" "$TF_WAIT_META2" "$TF_WAIT_META3" "$TF_WAIT_STATUS"
      print_failure_log_tail "$TF_WAIT_META3" "$FAIL_LOG_TAIL_LINES"
    fi
  done

  if (( trial_failed != 0 )); then
    return 1
  fi

  score_trial_weights "$trial"
}

mkdir -p "$OUT_DIR" "$LOG_DIR"
rm -f "$SUMMARY_CSV" "$SUMMARY_RANKED_CSV" "$FAIL_CSV"
ensure_required_inputs
detect_available_gpus
any_failed=0

for trial in "${TRIALS[@]}"; do
  if ! run_trial "$trial"; then
    any_failed=1
    echo "[dual140][warn] skipping OOF scoring for failed trial: $trial" >&2
    if [[ "$STOP_ON_FAILURE" == "1" ]]; then
      echo "[dual140][warn] stopping search after first failed trial (STOP_ON_FAILURE=1)." >&2
      break
    fi
  fi
done

write_ranked_summary
echo "[done] dual-H100 ~140GB best-performance search summary -> $SUMMARY_CSV"
if [[ -f "$FAIL_CSV" ]]; then
  echo "[done] failed trials -> $FAIL_CSV"
fi
if (( any_failed != 0 )); then
  exit 1
fi
