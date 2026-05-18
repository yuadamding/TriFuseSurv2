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

OUT_ROOT="${OUT_ROOT:-runs/roi_constrained_h100_search_${ENDPOINT_LC}_fold${fold_tag}}"
TRAIN_WRAPPER="${TRAIN_WRAPPER:-$PACKAGE_DIR/scripts/survival/train_with_roi_focus_watch.sh}"
EPOCHS="${EPOCHS:-60}"
WORKERS="${WORKERS:-2}"
LOG_EVERY_BATCHES="${LOG_EVERY_BATCHES:-50}"
WATCH_INTERVAL_SECONDS="${WATCH_INTERVAL_SECONDS:-60}"
TRIAL_LIMIT="${TRIAL_LIMIT:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_REQUIREMENTS_CHECK="${SKIP_REQUIREMENTS_CHECK:-0}"
ROI_FOCUS_WARMUP_EPOCHS="${ROI_FOCUS_WARMUP_EPOCHS:-30}"
CHECK_LATEST_N="${CHECK_LATEST_N:-8}"
LATEST_N="${LATEST_N:-8}"

if [[ ! -f "$TRAIN_WRAPPER" ]]; then
  echo "[search][FAIL] train wrapper not found: $TRAIN_WRAPPER" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT/logs"

# These are deliberately forced: every search trial uses GT masks for survival
# and the strict ROI watcher checks PT, LN, and PT peritumoral support.
COMMON_ENV=(
  "ENDPOINT=$ENDPOINT"
  "DEBUG_FOLD=$DEBUG_FOLD"
  "EPOCHS=$EPOCHS"
  "WORKERS=$WORKERS"
  "LOG_EVERY_BATCHES=$LOG_EVERY_BATCHES"
  "WATCH_INTERVAL_SECONDS=$WATCH_INTERVAL_SECONDS"
  "LATEST_N=$LATEST_N"
  "CHECK_LATEST_N=$CHECK_LATEST_N"
  "STRICT=1"
  "RESUME=0"
  "SURVIVAL_USE_GT_MASKS=1"
  "MASK_GUIDANCE_ALPHA=1.0"
  "TEACHER_FORCE_EPOCHS=0"
  "TEACHER_FORCE_START=1.0"
  "TEACHER_FORCE_END=1.0"
  "ROI_FOCUS_WARMUP_EPOCHS=$ROI_FOCUS_WARMUP_EPOCHS"
  "ROI_FOCUS_WARMUP_SURVIVAL_WEIGHT=0.0"
  "WARMUP_EPOCHS=$ROI_FOCUS_WARMUP_EPOCHS"
  "MIN_PROB_MASS_INSIDE_GT=0.95"
  "MIN_SUPPORT_RECALL=0.95"
  "MIN_SUPPORT_DICE=0.02"
  "MAX_EMPTY_WHEN_GT_PRESENT=0.25"
  "SKIP_REQUIREMENTS_CHECK=$SKIP_REQUIREMENTS_CHECK"
)

TRIALS=(
  "base_focus8|LR_BACKBONE=3e-4 LR_HEAD=1e-4 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=8 LOC_LOSS_PT_LAMBDA=4 LOC_LOSS_LN_LAMBDA=4 MASK_SUPPORT_LAMBDA=2 PT_SHELL_RADIUS=5 LN_SHELL_RADIUS=5 MODALITY_DROPOUT_CLIN_P=0.20 MODALITY_DROPOUT_RAD_P=0.20 V2_IMAGE_HABITAT_DROPOUT_P=0.05 V2_NODE_DROPOUT_P=0.10 V2_TOPOLOGY_DROPOUT_P=0.10 TOKEN_MLP_DROPOUT=0.55 PROJ_DROPOUT_P=0.35 RAD_PROJ_DROPOUT_P=0.30 EXPERT_DROPOUT_P=0.15 ATTN_DROPOUT_P=0.15|"
  "focus12|LR_BACKBONE=3e-4 LR_HEAD=1e-4 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=12 LOC_LOSS_PT_LAMBDA=4 LOC_LOSS_LN_LAMBDA=4 MASK_SUPPORT_LAMBDA=2 PT_SHELL_RADIUS=5 LN_SHELL_RADIUS=5 MODALITY_DROPOUT_CLIN_P=0.20 MODALITY_DROPOUT_RAD_P=0.20 V2_IMAGE_HABITAT_DROPOUT_P=0.05 V2_NODE_DROPOUT_P=0.10 V2_TOPOLOGY_DROPOUT_P=0.10 TOKEN_MLP_DROPOUT=0.55 PROJ_DROPOUT_P=0.35 RAD_PROJ_DROPOUT_P=0.30 EXPERT_DROPOUT_P=0.15 ATTN_DROPOUT_P=0.15|"
  "focus16|LR_BACKBONE=3e-4 LR_HEAD=1e-4 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=16 LOC_LOSS_PT_LAMBDA=4 LOC_LOSS_LN_LAMBDA=4 MASK_SUPPORT_LAMBDA=2 PT_SHELL_RADIUS=5 LN_SHELL_RADIUS=5 MODALITY_DROPOUT_CLIN_P=0.20 MODALITY_DROPOUT_RAD_P=0.20 V2_IMAGE_HABITAT_DROPOUT_P=0.05 V2_NODE_DROPOUT_P=0.10 V2_TOPOLOGY_DROPOUT_P=0.10 TOKEN_MLP_DROPOUT=0.55 PROJ_DROPOUT_P=0.35 RAD_PROJ_DROPOUT_P=0.30 EXPERT_DROPOUT_P=0.15 ATTN_DROPOUT_P=0.15|"
  "shell7_focus12|LR_BACKBONE=3e-4 LR_HEAD=1e-4 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=12 LOC_LOSS_PT_LAMBDA=4 LOC_LOSS_LN_LAMBDA=4 MASK_SUPPORT_LAMBDA=2 PT_SHELL_RADIUS=7 LN_SHELL_RADIUS=5 MODALITY_DROPOUT_CLIN_P=0.20 MODALITY_DROPOUT_RAD_P=0.20 V2_IMAGE_HABITAT_DROPOUT_P=0.05 V2_NODE_DROPOUT_P=0.10 V2_TOPOLOGY_DROPOUT_P=0.10 TOKEN_MLP_DROPOUT=0.55 PROJ_DROPOUT_P=0.35 RAD_PROJ_DROPOUT_P=0.30 EXPERT_DROPOUT_P=0.15 ATTN_DROPOUT_P=0.15|"
  "shell7_focus16|LR_BACKBONE=3e-4 LR_HEAD=1e-4 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=16 LOC_LOSS_PT_LAMBDA=4 LOC_LOSS_LN_LAMBDA=4 MASK_SUPPORT_LAMBDA=2 PT_SHELL_RADIUS=7 LN_SHELL_RADIUS=5 MODALITY_DROPOUT_CLIN_P=0.20 MODALITY_DROPOUT_RAD_P=0.20 V2_IMAGE_HABITAT_DROPOUT_P=0.05 V2_NODE_DROPOUT_P=0.10 V2_TOPOLOGY_DROPOUT_P=0.10 TOKEN_MLP_DROPOUT=0.55 PROJ_DROPOUT_P=0.35 RAD_PROJ_DROPOUT_P=0.30 EXPERT_DROPOUT_P=0.15 ATTN_DROPOUT_P=0.15|"
  "low_lr_focus12|LR_BACKBONE=1.5e-4 LR_HEAD=7e-5 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=12 LOC_LOSS_PT_LAMBDA=4 LOC_LOSS_LN_LAMBDA=4 MASK_SUPPORT_LAMBDA=2 PT_SHELL_RADIUS=5 LN_SHELL_RADIUS=5 MODALITY_DROPOUT_CLIN_P=0.20 MODALITY_DROPOUT_RAD_P=0.20 V2_IMAGE_HABITAT_DROPOUT_P=0.05 V2_NODE_DROPOUT_P=0.10 V2_TOPOLOGY_DROPOUT_P=0.10 TOKEN_MLP_DROPOUT=0.55 PROJ_DROPOUT_P=0.35 RAD_PROJ_DROPOUT_P=0.30 EXPERT_DROPOUT_P=0.15 ATTN_DROPOUT_P=0.15|"
  "head_lr_focus12|LR_BACKBONE=2e-4 LR_HEAD=1.5e-4 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=12 LOC_LOSS_PT_LAMBDA=4 LOC_LOSS_LN_LAMBDA=4 MASK_SUPPORT_LAMBDA=2 PT_SHELL_RADIUS=5 LN_SHELL_RADIUS=5 MODALITY_DROPOUT_CLIN_P=0.20 MODALITY_DROPOUT_RAD_P=0.20 V2_IMAGE_HABITAT_DROPOUT_P=0.05 V2_NODE_DROPOUT_P=0.10 V2_TOPOLOGY_DROPOUT_P=0.10 TOKEN_MLP_DROPOUT=0.55 PROJ_DROPOUT_P=0.35 RAD_PROJ_DROPOUT_P=0.30 EXPERT_DROPOUT_P=0.15 ATTN_DROPOUT_P=0.15|"
  "aux50_focus12|LR_BACKBONE=3e-4 LR_HEAD=1e-4 AUX_SURV_LOSS_WEIGHT=0.50 MASK_FOCUS_LAMBDA=12 LOC_LOSS_PT_LAMBDA=4 LOC_LOSS_LN_LAMBDA=4 MASK_SUPPORT_LAMBDA=2 PT_SHELL_RADIUS=5 LN_SHELL_RADIUS=5 MODALITY_DROPOUT_CLIN_P=0.20 MODALITY_DROPOUT_RAD_P=0.20 V2_IMAGE_HABITAT_DROPOUT_P=0.05 V2_NODE_DROPOUT_P=0.10 V2_TOPOLOGY_DROPOUT_P=0.10 TOKEN_MLP_DROPOUT=0.55 PROJ_DROPOUT_P=0.35 RAD_PROJ_DROPOUT_P=0.30 EXPERT_DROPOUT_P=0.15 ATTN_DROPOUT_P=0.15|"
  "aux20_focus12|LR_BACKBONE=3e-4 LR_HEAD=1e-4 AUX_SURV_LOSS_WEIGHT=0.20 MASK_FOCUS_LAMBDA=12 LOC_LOSS_PT_LAMBDA=4 LOC_LOSS_LN_LAMBDA=4 MASK_SUPPORT_LAMBDA=2 PT_SHELL_RADIUS=5 LN_SHELL_RADIUS=5 MODALITY_DROPOUT_CLIN_P=0.20 MODALITY_DROPOUT_RAD_P=0.20 V2_IMAGE_HABITAT_DROPOUT_P=0.05 V2_NODE_DROPOUT_P=0.10 V2_TOPOLOGY_DROPOUT_P=0.10 TOKEN_MLP_DROPOUT=0.55 PROJ_DROPOUT_P=0.35 RAD_PROJ_DROPOUT_P=0.30 EXPERT_DROPOUT_P=0.15 ATTN_DROPOUT_P=0.15|"
  "low_dropout_focus12|LR_BACKBONE=3e-4 LR_HEAD=1e-4 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=12 LOC_LOSS_PT_LAMBDA=4 LOC_LOSS_LN_LAMBDA=4 MASK_SUPPORT_LAMBDA=2 PT_SHELL_RADIUS=5 LN_SHELL_RADIUS=5 MODALITY_DROPOUT_CLIN_P=0.10 MODALITY_DROPOUT_RAD_P=0.10 V2_IMAGE_HABITAT_DROPOUT_P=0.025 V2_NODE_DROPOUT_P=0.05 V2_TOPOLOGY_DROPOUT_P=0.05 TOKEN_MLP_DROPOUT=0.40 PROJ_DROPOUT_P=0.25 RAD_PROJ_DROPOUT_P=0.20 EXPERT_DROPOUT_P=0.10 ATTN_DROPOUT_P=0.10|"
  "high_dropout_focus12|LR_BACKBONE=3e-4 LR_HEAD=1e-4 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=12 LOC_LOSS_PT_LAMBDA=4 LOC_LOSS_LN_LAMBDA=4 MASK_SUPPORT_LAMBDA=2 PT_SHELL_RADIUS=5 LN_SHELL_RADIUS=5 MODALITY_DROPOUT_CLIN_P=0.30 MODALITY_DROPOUT_RAD_P=0.30 V2_IMAGE_HABITAT_DROPOUT_P=0.08 V2_NODE_DROPOUT_P=0.15 V2_TOPOLOGY_DROPOUT_P=0.15 TOKEN_MLP_DROPOUT=0.60 PROJ_DROPOUT_P=0.45 RAD_PROJ_DROPOUT_P=0.40 EXPERT_DROPOUT_P=0.20 ATTN_DROPOUT_P=0.20|"
  "roi_heavy_focus16|LR_BACKBONE=2e-4 LR_HEAD=1e-4 AUX_SURV_LOSS_WEIGHT=0.35 MASK_FOCUS_LAMBDA=16 LOC_LOSS_PT_LAMBDA=6 LOC_LOSS_LN_LAMBDA=6 MASK_SUPPORT_LAMBDA=3 PT_SHELL_RADIUS=7 LN_SHELL_RADIUS=5 MODALITY_DROPOUT_CLIN_P=0.20 MODALITY_DROPOUT_RAD_P=0.20 V2_IMAGE_HABITAT_DROPOUT_P=0.05 V2_NODE_DROPOUT_P=0.10 V2_TOPOLOGY_DROPOUT_P=0.10 TOKEN_MLP_DROPOUT=0.55 PROJ_DROPOUT_P=0.35 RAD_PROJ_DROPOUT_P=0.30 EXPERT_DROPOUT_P=0.15 ATTN_DROPOUT_P=0.15|"
)

trial_total="${#TRIALS[@]}"
if [[ "$TRIAL_LIMIT" =~ ^[0-9]+$ ]] && (( TRIAL_LIMIT > 0 && TRIAL_LIMIT < trial_total )); then
  trial_total="$TRIAL_LIMIT"
fi

write_search_manifest() {
  {
    echo "out_root=$OUT_ROOT"
    echo "endpoint=$ENDPOINT"
    echo "debug_fold=$DEBUG_FOLD"
    echo "gpu_ids=${GPU_ARRAY[*]}"
    echo "epochs=$EPOCHS"
    echo "roi_focus_warmup_epochs=$ROI_FOCUS_WARMUP_EPOCHS"
    echo "strict=1"
    echo "survival_use_gt_masks=1"
    echo "mask_guidance_alpha=1.0"
    echo "min_prob_mass_inside_gt=0.95"
    echo "min_support_recall=0.95"
    echo "min_support_dice=0.02"
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
  for ((idx = slot; idx < trial_total; idx += ${#GPU_ARRAY[@]})); do
    IFS='|' read -r trial_name trial_env trial_args <<< "${TRIALS[$idx]}"
    trial_dir="$OUT_ROOT/$(printf '%02d' "$idx")_${trial_name}"
    exp_name="$(printf 'cv4_contour_aware_%s_search%02d_%s_fold%s' "$ENDPOINT_LC" "$idx" "$trial_name" "$fold_tag")"
    log_file="$OUT_ROOT/logs/$(printf '%02d' "$idx")_${trial_name}.log"
    mkdir -p "$trial_dir"

    read -r -a trial_env_array <<< "$trial_env"
    read -r -a trial_arg_array <<< "$trial_args"

    {
      echo "trial_index=$idx"
      echo "trial_name=$trial_name"
      echo "slot=$slot"
      echo "cuda_device=$gpu"
      echo "out_dir=$trial_dir"
      echo "exp_name=$exp_name"
      echo "env=$trial_env"
      echo "args=$trial_args"
      echo "log=$log_file"
      echo "started_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    } > "$trial_dir/trial.env"

    echo "[search][slot $slot gpu $gpu] starting trial $idx/$((trial_total - 1)): $trial_name"
    set +e
    env \
      "${COMMON_ENV[@]}" \
      "${trial_env_array[@]}" \
      "CUDA_DEVICE=$gpu" \
      "OUT_DIR=$trial_dir" \
      "EXP_NAME=$exp_name" \
      bash "$TRAIN_WRAPPER" "${trial_arg_array[@]}" > "$log_file" 2>&1
    rc="$?"
    set -e

    echo "$rc" > "$trial_dir/status.rc"
    date -u '+%Y-%m-%dT%H:%M:%SZ' > "$trial_dir/finished_utc.txt"
    if [[ "$rc" == "0" ]]; then
      echo "done" > "$trial_dir/status.txt"
      echo "[search][slot $slot gpu $gpu] completed trial $idx: $trial_name"
    else
      echo "failed" > "$trial_dir/status.txt"
      echo "[search][slot $slot gpu $gpu][FAIL] trial $idx failed rc=$rc: $trial_name (log: $log_file)" >&2
    fi
  done
}

aggregate_results() {
  python3 - "$OUT_ROOT" <<'PY'
import csv
import json
import math
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
    best_val_auc = float("nan")
    if metric_rows:
        best_val_auc = max((as_float(r.get("val_auc_1095d")) for r in metric_rows), default=float("nan"))
    test_c = as_float(summary.get("mean_fold_test_c_index", summary.get("mean_test_c_index")))
    score = test_c if math.isfinite(test_c) else best_val_auc
    rows.append({
        "trial": trial_dir.name,
        "status": status,
        "rc": rc,
        "score": score,
        "test_c_index": test_c,
        "best_val_auc_1095d": best_val_auc,
        "final_val_auc_1095d": as_float(last.get("val_auc_1095d")),
        "final_val_c": as_float(last.get("val_c")),
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
    "trial", "status", "rc", "score", "test_c_index", "best_val_auc_1095d",
    "final_val_auc_1095d", "final_val_c",
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
print("rank,trial,status,score,test_c,best_val_auc,pt_mass,ln_mass,pt_peri_mass,pt_rec,ln_rec,pt_peri_rec")
for rank, r in enumerate(rows[:10], 1):
    print(",".join([
        str(rank),
        r["trial"],
        r["status"],
        fmt(r["score"]),
        fmt(r["test_c_index"]),
        fmt(r["best_val_auc_1095d"]),
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
echo "[search] trials=$trial_total epochs=$EPOCHS"
echo "[search] enforcing GT survival masks and strict PT/LN/PT-peri ROI constraints"

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "yes" ]]; then
  echo "[search] dry run only; wrote manifest to $OUT_ROOT/search_manifest.txt"
  for ((i = 0; i < trial_total; i++)); do
    IFS='|' read -r trial_name trial_env trial_args <<< "${TRIALS[$i]}"
    gpu="${GPU_ARRAY[$((i % ${#GPU_ARRAY[@]}))]}"
    echo "[search][dry-run] trial $i gpu=$gpu name=$trial_name env='$trial_env' args='$trial_args'"
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
