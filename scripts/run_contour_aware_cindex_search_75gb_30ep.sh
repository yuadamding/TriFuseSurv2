#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PACKAGE_DIR/.." && pwd)"
cd "$WORKSPACE_ROOT"

export PYTHONPATH="$PACKAGE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
source "$PACKAGE_DIR/scripts/lib/gpu_utils.sh"

META_CSV="${META_CSV:-OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv}"
RADIOMICS_SOURCE="${RADIOMICS_SOURCE:-cohort_radiomics_patient_wide.csv}"
NODE_TOPOLOGY_DIR="${NODE_TOPOLOGY_DIR:-}"
MODEL_VERSION="${MODEL_VERSION:-v2}"
ENDPOINT="${ENDPOINT:-OS}"
ENDPOINT_LC="$(printf '%s' "$ENDPOINT" | tr '[:upper:]' '[:lower:]')"
SPLITS_DIR="${SPLITS_DIR:-runs/opscc_splits_${ENDPOINT_LC}_seed1}"
OUT_DIR="${OUT_DIR:-runs/contour_aware_cindex_search_75gb_30ep_multitask_${ENDPOINT_LC}}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="30"
MAX_PARALLEL="${MAX_PARALLEL:-0}"
WORKERS="${WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-75000}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}"
FAIL_LOG_TAIL_LINES="${FAIL_LOG_TAIL_LINES:-120}"
SKIP_FINISHED="${SKIP_FINISHED:-1}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_OUTER_TEST_SCORING="${ALLOW_OUTER_TEST_SCORING:-0}"

if [[ "$DRY_RUN" != "1" ]]; then
  tf_require_python_modules numpy pandas SimpleITK torch monai sklearn pydicom rt_utils cv2
fi

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
normalize_list_var TRIALS "v75_tri_base_h1095 v75_tri_aux020_h1095 v75_tri_aux055_h1095 v75_tri_h1095_bw120 v75_tri_h1095_nomultiscale v75_tri_h1095_tf24 v75_tri_h1095_locweak v75_tri_h1095_nomultiscale_locweak"

SUMMARY_CSV="$OUT_DIR/tuning_summary.csv"
SUMMARY_RANKED_CSV="$OUT_DIR/tuning_summary_ranked.csv"
FAIL_CSV="$OUT_DIR/failed_trials.csv"
LOG_DIR="$OUT_DIR/logs"

ensure_required_inputs() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[search75][dry-run] skipping input file checks"
    return 0
  fi
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
  if [[ ! "$EPOCHS" =~ ^[0-9]+$ ]] || (( EPOCHS <= 0 )); then
    echo "[error] EPOCHS must be a positive integer, got: $EPOCHS" >&2
    exit 1
  fi
}

detect_available_gpus() {
  if [[ "$DRY_RUN" == "1" ]]; then
    AVAILABLE_GPU_IDS=("0")
    echo "[search75][dry-run] using synthetic gpu list: ${AVAILABLE_GPU_IDS[*]}"
    return 0
  fi
  local -a ids=()
  if declare -F tf_detect_gpu_ids_by_free_mem >/dev/null 2>&1; then
    mapfile -t ids < <(tf_detect_gpu_ids_by_free_mem "$MIN_FREE_GPU_MB")
  elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t ids < <(
      nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' -v min_free="$MIN_FREE_GPU_MB" '{gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); if ($2+0 >= min_free+0) print $1}'
    )
  else
    mapfile -t ids < <(tf_detect_gpu_ids)
  fi
  if (( ${#ids[@]} == 0 )); then
    echo "[error] no GPUs with at least ${MIN_FREE_GPU_MB} MB free were detected." >&2
    exit 1
  fi

  local want="${MAX_PARALLEL}"
  if [[ ! "$want" =~ ^[0-9]+$ ]] || (( want <= 0 || want > ${#ids[@]} )); then
    want="${#ids[@]}"
  fi
  AVAILABLE_GPU_IDS=("${ids[@]:0:$want}")
  echo "[search75] detected GPUs with >=${MIN_FREE_GPU_MB} MB free: ${AVAILABLE_GPU_IDS[*]} | max_parallel=${#AVAILABLE_GPU_IDS[@]}"
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
    "score_source": "outer_test_oof",
    "weight": data["weights"],
    "c_index": data["c_index"],
    "n_predictions": data["n_predictions"],
    "n_evaluable": data["n_evaluable"],
    "n_comparable_pairs": data.get("n_comparable_pairs", ""),
    "n_risk_files": data["n_risk_files"],
    "n_metric_files": "",
}
write_header = not summary_csv.exists()
with summary_csv.open("a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
        w.writeheader()
    w.writerow(row)
print(f"[search75] {trial} weight={row['weight']} OOF c-index={row['c_index']:.4f}")
PY
}

append_validation_summary_row() {
  local trial="$1"
  local trial_dir="$OUT_DIR/$trial"
  local -a matches=()
  while IFS= read -r path; do
    matches+=("$path")
  done < <(find "$trial_dir" -path "*/metrics.csv" -type f | sort)

  if (( ${#matches[@]} == 0 )); then
    echo "[search75] trial=$trial -> no metrics.csv files, skipping validation score"
    return 0
  fi
  if (( ${#matches[@]} != ${#FOLDS[@]} )); then
    echo "[search75] trial=$trial -> found ${#matches[@]} metrics files for ${#FOLDS[@]} folds, skipping incomplete validation score"
    return 0
  fi

  python3 - "$trial" "$SUMMARY_CSV" "${matches[@]}" <<'PY'
import csv
import math
import sys
from pathlib import Path

trial = sys.argv[1]
summary_csv = Path(sys.argv[2])
metric_paths = [Path(p) for p in sys.argv[3:]]
fold_best = []
for path in metric_paths:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    values = []
    for row in rows:
        try:
            value = float(row.get("val_c_index", "nan"))
        except Exception:
            value = float("nan")
        if math.isfinite(value):
            values.append(value)
    if values:
        fold_best.append(max(values))

c_index = sum(fold_best) / len(fold_best) if fold_best else float("nan")
row = {
    "trial": trial,
    "score_source": "validation_metrics",
    "weight": "validation",
    "c_index": c_index,
    "n_predictions": "",
    "n_evaluable": len(fold_best),
    "n_comparable_pairs": "",
    "n_risk_files": "",
    "n_metric_files": len(metric_paths),
}
write_header = not summary_csv.exists()
with summary_csv.open("a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
        w.writeheader()
    w.writerow(row)
print(f"[search75] {trial} validation mean best val c-index={c_index:.4f} folds={len(fold_best)}")
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
    echo "[search75] ---- tail -n ${lines} ${log_file} ----" >&2
    tail -n "$lines" "$log_file" >&2 || true
    echo "[search75] ---- end log tail ----" >&2
  else
    echo "[search75][warn] missing failure log: $log_file" >&2
  fi
}

fold_export_dir() {
  local trial="$1"
  local fold="$2"
  printf '%s/%s_fold%02d/fold_%02d' "$OUT_DIR/$trial" "$trial" "$fold" "$fold"
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

fold_has_complete_metrics() {
  local trial="$1"
  local fold="$2"
  local fold_dir
  fold_dir="$(fold_export_dir "$trial" "$fold")"
  [[ -f "$fold_dir/metrics.csv" ]]
}

fold_has_complete_outputs() {
  if [[ "$ALLOW_OUTER_TEST_SCORING" == "1" ]]; then
    fold_has_complete_risks "$@"
  else
    fold_has_complete_metrics "$@"
  fi
}

configure_trial() {
  local trial="$1"

  local model_version="$MODEL_VERSION"
  local fused_dim="1408"
  local token_dim="1920"
  local token_mlp_hidden_dim="3328"
  local img_proj_hidden_dim="3328"
  local img_tok_ffn_hidden_dim="3328"
  local img_post_hidden_dim="3328"
  local img_attn_heads="8"
  local gate_hidden_dim="1792"
  local rad_hidden_dim="2304"
  local pca="32"
  local v2_model_dim="256"
  local v2_num_heads="8"
  local v2_transformer_layers="2"
  local v2_dropout="0.10"
  local v2_radiomics_pcs_per_group="16"
  local v2_max_nodes="16"
  local v2_node_token_dim="9"
  local v2_topology_dim="9"
  local v2_clinical_group_dropout_p="0.05"
  local v2_radiomics_group_dropout_p="0.10"
  local v2_node_dropout_p="0.10"
  local v2_topology_dropout_p="0.10"
  local v2_image_habitat_dropout_p="0.05"
  local v2_dropout_ramp_epochs="12"
  local time_bin_width="180"
  local risk_horizon="1095"
  local pt_shell_radius="5"
  local ln_shell_radius="5"
  local teacher_force_epochs="16"
  local teacher_force_start="1.0"
  local teacher_force_end="0.0"
  local loc_loss_pt_lambda="0.25"
  local loc_loss_ln_lambda="0.25"
  local loc_presence_lambda="0.05"
  local primary_surv_loss_weight="1.0"
  local aux_surv_loss_weight="0.35"
  local use_multiscale="1"
  local use_radiomics="1"

  case "$trial" in
    v2_anchor_m256_rad16_tf16)
      token_dim="768"
      token_mlp_hidden_dim="1536"
      img_proj_hidden_dim="1536"
      img_tok_ffn_hidden_dim="1536"
      img_post_hidden_dim="1536"
      rad_hidden_dim="1024"
      pca="64"
      v2_model_dim="256"
      v2_radiomics_pcs_per_group="16"
      teacher_force_epochs="16"
      v2_dropout_ramp_epochs="12"
      ;;
    v2_m256_rad24_tf20)
      token_dim="768"
      token_mlp_hidden_dim="1536"
      img_proj_hidden_dim="1536"
      img_tok_ffn_hidden_dim="1536"
      img_post_hidden_dim="1536"
      rad_hidden_dim="1024"
      pca="96"
      v2_model_dim="256"
      v2_radiomics_pcs_per_group="24"
      teacher_force_epochs="20"
      v2_radiomics_group_dropout_p="0.12"
      ;;
    v2_m256_rad32_tf24)
      token_dim="768"
      token_mlp_hidden_dim="1536"
      img_proj_hidden_dim="1536"
      img_tok_ffn_hidden_dim="1536"
      img_post_hidden_dim="1536"
      rad_hidden_dim="1024"
      pca="128"
      v2_model_dim="256"
      v2_radiomics_pcs_per_group="32"
      teacher_force_epochs="24"
      v2_radiomics_group_dropout_p="0.15"
      ;;
    v2_m384_rad16_tf16)
      token_dim="1024"
      token_mlp_hidden_dim="2048"
      img_proj_hidden_dim="2048"
      img_tok_ffn_hidden_dim="2048"
      img_post_hidden_dim="2048"
      rad_hidden_dim="1536"
      pca="64"
      v2_model_dim="384"
      v2_num_heads="8"
      v2_radiomics_pcs_per_group="16"
      teacher_force_epochs="16"
      v2_image_habitat_dropout_p="0.04"
      ;;
    v2_m384_rad24_tf20)
      token_dim="1024"
      token_mlp_hidden_dim="2048"
      img_proj_hidden_dim="2048"
      img_tok_ffn_hidden_dim="2048"
      img_post_hidden_dim="2048"
      rad_hidden_dim="1536"
      pca="96"
      v2_model_dim="384"
      v2_num_heads="8"
      v2_radiomics_pcs_per_group="24"
      v2_dropout="0.12"
      teacher_force_epochs="20"
      v2_radiomics_group_dropout_p="0.12"
      ;;
    v2_m384_rad32_tf24)
      token_dim="1024"
      token_mlp_hidden_dim="2048"
      img_proj_hidden_dim="2048"
      img_tok_ffn_hidden_dim="2048"
      img_post_hidden_dim="2048"
      rad_hidden_dim="1536"
      pca="128"
      v2_model_dim="384"
      v2_num_heads="8"
      v2_radiomics_pcs_per_group="32"
      v2_dropout="0.12"
      teacher_force_epochs="24"
      v2_radiomics_group_dropout_p="0.15"
      ;;
    v2_m384_layers3_rad24_tf20)
      token_dim="1024"
      token_mlp_hidden_dim="2048"
      img_proj_hidden_dim="2048"
      img_tok_ffn_hidden_dim="2048"
      img_post_hidden_dim="2048"
      rad_hidden_dim="1536"
      pca="96"
      v2_model_dim="384"
      v2_num_heads="8"
      v2_transformer_layers="3"
      v2_radiomics_pcs_per_group="24"
      teacher_force_epochs="20"
      v2_node_dropout_p="0.12"
      v2_topology_dropout_p="0.12"
      ;;
    v2_m512_rad16_tf16)
      token_dim="1024"
      token_mlp_hidden_dim="2048"
      img_proj_hidden_dim="2048"
      img_tok_ffn_hidden_dim="2048"
      img_post_hidden_dim="2048"
      rad_hidden_dim="1536"
      pca="64"
      v2_model_dim="512"
      v2_num_heads="8"
      v2_radiomics_pcs_per_group="16"
      v2_dropout="0.15"
      teacher_force_epochs="16"
      v2_image_habitat_dropout_p="0.04"
      ;;
    v2_m384_rad24_tf24_locweak)
      token_dim="1024"
      token_mlp_hidden_dim="2048"
      img_proj_hidden_dim="2048"
      img_tok_ffn_hidden_dim="2048"
      img_post_hidden_dim="2048"
      rad_hidden_dim="1536"
      pca="96"
      v2_model_dim="384"
      v2_radiomics_pcs_per_group="24"
      teacher_force_epochs="24"
      loc_loss_pt_lambda="0.12"
      loc_loss_ln_lambda="0.12"
      loc_presence_lambda="0.02"
      v2_dropout_ramp_epochs="16"
      ;;
    v2_m384_nomultiscale_rad24_tf20)
      token_dim="1024"
      token_mlp_hidden_dim="2048"
      img_proj_hidden_dim="2048"
      img_tok_ffn_hidden_dim="2048"
      img_post_hidden_dim="2048"
      rad_hidden_dim="1536"
      pca="96"
      v2_model_dim="384"
      v2_radiomics_pcs_per_group="24"
      teacher_force_epochs="20"
      use_multiscale="0"
      v2_image_habitat_dropout_p="0.03"
      ;;
    v75_tri_base_h1095)
      ;;
    v75_tri_aux020_h1095)
      aux_surv_loss_weight="0.20"
      ;;
    v75_tri_aux055_h1095)
      aux_surv_loss_weight="0.55"
      ;;
    v75_tri_h1095_bw120)
      time_bin_width="120"
      ;;
    v75_tri_h1095_nomultiscale)
      use_multiscale="0"
      ;;
    v75_tri_h1095_tf24)
      teacher_force_epochs="24"
      ;;
    v75_tri_h1095_locweak)
      loc_loss_pt_lambda="0.12"
      loc_loss_ln_lambda="0.12"
      loc_presence_lambda="0.02"
      ;;
    v75_tri_h1095_nomultiscale_locweak)
      use_multiscale="0"
      loc_loss_pt_lambda="0.12"
      loc_loss_ln_lambda="0.12"
      loc_presence_lambda="0.02"
      ;;
    *)
      echo "[error] unknown 75gb trial: $trial" >&2
      exit 1
      ;;
  esac

  TRIAL_WRAPPER="$PACKAGE_DIR/scripts/survival/train_contour_aware_survival.sh"
  TRIAL_ARGS=(
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --workers "$WORKERS"
    --no_resume
    --lightweight_checkpoints
    --report_metric c_index
    --use_ema
    --use_swa
    --export_extra_risks
    --no_use_checkpoint
    --model_version "$model_version"
    --time_bin_width_days "$time_bin_width"
    --risk_horizon_days "$risk_horizon"
    --primary_surv_loss_weight "$primary_surv_loss_weight"
    --aux_surv_loss_weight "$aux_surv_loss_weight"
    --ema_decay 0.9995
    --swa_start_epoch 10
    --swa_update_freq_epochs 1
    --pt_shell_radius "$pt_shell_radius"
    --ln_shell_radius "$ln_shell_radius"
    --img_token_dim "$token_dim"
    --token_mlp_hidden_dim "$token_mlp_hidden_dim"
    --lr_backbone 3e-4
    --lr_head 8e-5
    --wd_rad 1e-3
    --token_mlp_dropout 0.10
    --token_dropout 0.02
    --hazard_smooth_lambda 0.001
    --teacher_force_epochs "$teacher_force_epochs"
    --teacher_force_start "$teacher_force_start"
    --teacher_force_end "$teacher_force_end"
    --loc_loss_pt_lambda "$loc_loss_pt_lambda"
    --loc_loss_ln_lambda "$loc_loss_ln_lambda"
    --loc_presence_lambda "$loc_presence_lambda"
    --shell_body_from_ct
  )

  if [[ "$model_version" == "v2" ]]; then
    TRIAL_ARGS+=(
      --v2_model_dim "$v2_model_dim"
      --v2_num_heads "$v2_num_heads"
      --v2_transformer_layers "$v2_transformer_layers"
      --v2_dropout "$v2_dropout"
      --v2_radiomics_pcs_per_group "$v2_radiomics_pcs_per_group"
      --v2_max_nodes "$v2_max_nodes"
      --v2_node_token_dim "$v2_node_token_dim"
      --v2_topology_dim "$v2_topology_dim"
      --v2_clinical_group_dropout_p "$v2_clinical_group_dropout_p"
      --v2_radiomics_group_dropout_p "$v2_radiomics_group_dropout_p"
      --v2_node_dropout_p "$v2_node_dropout_p"
      --v2_topology_dropout_p "$v2_topology_dropout_p"
      --v2_image_habitat_dropout_p "$v2_image_habitat_dropout_p"
      --v2_dropout_ramp_epochs "$v2_dropout_ramp_epochs"
    )
  else
    TRIAL_ARGS+=(
      --radiomics_pca_total_components "$pca"
      --fused_dim "$fused_dim"
      --img_proj_hidden_dim "$img_proj_hidden_dim"
      --img_tok_ffn_hidden_dim "$img_tok_ffn_hidden_dim"
      --img_post_hidden_dim "$img_post_hidden_dim"
      --img_attn_heads "$img_attn_heads"
      --gate_hidden_dim "$gate_hidden_dim"
      --rad_hidden_dim "$rad_hidden_dim"
      --modality_dropout_clin_p 0.00
      --modality_dropout_rad_p 0.05
      --clinical_noise_std 0.0
      --radiomics_noise_std 0.0
      --gate_dropout_p 0.05
      --surv_dropout_p 0.10
      --rad_proj_dropout_p 0.05
      --proj_dropout_p 0.10
      --expert_dropout_p 0.00
      --attn_dropout_p 0.02
      --gate_entropy_lambda 0.001
      --gate_loadbal_lambda 0.001
    )
  fi

  if [[ "$use_multiscale" == "1" ]]; then
    TRIAL_ARGS+=(--use_multiscale)
  fi
  if [[ "$use_radiomics" == "0" ]]; then
    TRIAL_ARGS+=(--no_radiomics)
  fi
  if [[ -n "$NODE_TOPOLOGY_DIR" ]]; then
    TRIAL_ARGS+=(--node_topology_dir "$NODE_TOPOLOGY_DIR")
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
      echo "[search75] trial=$trial weight=$weight -> no matching risk files, skipping"
      continue
    fi
    if (( ${#matches[@]} != ${#FOLDS[@]} )); then
      echo "[search75] trial=$trial weight=$weight -> found ${#matches[@]} risk files for ${#FOLDS[@]} folds, skipping incomplete OOF score"
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

score_trial() {
  local trial="$1"
  append_validation_summary_row "$trial"
  if [[ "$ALLOW_OUTER_TEST_SCORING" == "1" ]]; then
    score_trial_weights "$trial"
  else
    echo "[search75] trial=$trial -> outer-test OOF scoring disabled; set ALLOW_OUTER_TEST_SCORING=1 only for final audit reporting"
  fi
}

run_trial() {
  local trial="$1"
  configure_trial "$trial"
  if [[ "$DRY_RUN" == "1" ]]; then
    local display_wrapper="${TRIAL_WRAPPER#$WORKSPACE_ROOT/}"
    echo "[search75][dry-run] trial=$trial wrapper=$display_wrapper"
    printf '[search75][dry-run] trial_args='
    printf '%q ' "${TRIAL_ARGS[@]}"
    printf '\n'
    return 0
  fi

  local trial_dir="$OUT_DIR/$trial"
  if [[ "$SKIP_FINISHED" != "1" ]]; then
    rm -rf "$trial_dir"
  fi
  mkdir -p "$trial_dir"

  local -a pids=()
  local -a running_gpus=()
  local -a running_folds=()
  local -a running_logs=()
  local -a free_gpus=("${AVAILABLE_GPU_IDS[@]}")
  local -a pending_folds=()
  local fold
  for fold in "${FOLDS[@]}"; do
    if [[ "$SKIP_FINISHED" == "1" ]] && fold_has_complete_outputs "$trial" "$fold"; then
      echo "[search75] trial=$trial fold=$fold -> existing completed outputs found, skipping training"
      continue
    fi
    pending_folds+=("$fold")
  done

  if (( ${#pending_folds[@]} == 0 )); then
    echo "[search75] trial=$trial -> all requested folds already finished; scoring existing outputs"
    score_trial "$trial"
    return 0
  fi

  local total_folds="${#pending_folds[@]}"
  local next_fold_idx=0
  local trial_failed=0

  while (( next_fold_idx < total_folds || ${#pids[@]} > 0 )); do
    while (( trial_failed == 0 && next_fold_idx < total_folds && ${#free_gpus[@]} > 0 )); do
      local fold="${pending_folds[$next_fold_idx]}"
      local gpu="${free_gpus[0]}"
      local exp_name="${trial}_fold$(printf '%02d' "$fold")"
      local log_file="$LOG_DIR/${trial}_fold$(printf '%02d' "$fold").log"
      rm -rf "$trial_dir/$exp_name"
      free_gpus=("${free_gpus[@]:1}")

      echo "[search75] trial=$trial fold=$fold gpu=$gpu log=$log_file"
      META_CSV="$META_CSV" \
      SPLITS_DIR="$SPLITS_DIR" \
      RADIOMICS_SOURCE="$RADIOMICS_SOURCE" \
      NODE_TOPOLOGY_DIR="$NODE_TOPOLOGY_DIR" \
      MODEL_VERSION="$MODEL_VERSION" \
      DRY_RUN="$DRY_RUN" \
      ENDPOINT="$ENDPOINT" \
      OUT_DIR="$trial_dir" \
      EXP_NAME="$exp_name" \
      DEBUG_FOLD="$fold" \
      CUDA_DEVICE="$gpu" \
      DEVICE="$DEVICE" \
      PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
      bash "$TRIAL_WRAPPER" "${TRIAL_ARGS[@]}" >"$log_file" 2>&1 &

      pids+=("$!")
      running_gpus+=("$gpu")
      running_folds+=("$fold")
      running_logs+=("$log_file")
      next_fold_idx=$(( next_fold_idx + 1 ))
    done

    if (( ${#pids[@]} == 0 )); then
      break
    fi

    tf_wait_for_any_tracked_pid pids running_gpus running_folds running_logs
    free_gpus+=("$TF_WAIT_META1")

    if (( TF_WAIT_STATUS != 0 )); then
      trial_failed=1
      echo "[search75][warn] trial=$trial fold=$TF_WAIT_META2 failed with status=$TF_WAIT_STATUS; see $TF_WAIT_META3" >&2
      append_failure_row "$trial" "$TF_WAIT_META2" "$TF_WAIT_META3" "$TF_WAIT_STATUS"
      print_failure_log_tail "$TF_WAIT_META3" "$FAIL_LOG_TAIL_LINES"
    fi
  done

  if (( trial_failed != 0 )); then
    return 1
  fi

  score_trial "$trial"
}

mkdir -p "$OUT_DIR" "$LOG_DIR"
rm -f "$SUMMARY_CSV" "$SUMMARY_RANKED_CSV" "$FAIL_CSV"
ensure_required_inputs
detect_available_gpus
any_failed=0

for trial in "${TRIALS[@]}"; do
  if ! run_trial "$trial"; then
    any_failed=1
    echo "[search75][warn] skipping OOF scoring for failed trial: $trial" >&2
    if [[ "$STOP_ON_FAILURE" == "1" ]]; then
      echo "[search75][warn] stopping search after first failed trial (STOP_ON_FAILURE=1)." >&2
      break
    fi
  fi
done

write_ranked_summary
echo "[done] contour-aware 75GB 30-epoch search summary -> $SUMMARY_CSV"
if [[ -f "$FAIL_CSV" ]]; then
  echo "[done] failed trials -> $FAIL_CSV"
fi
if (( any_failed != 0 )); then
  exit 1
fi
