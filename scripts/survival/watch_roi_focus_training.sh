#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$PACKAGE_DIR/.." && pwd)"
cd "$WORKSPACE_ROOT"

METRICS_CSV="${METRICS_CSV:-}"
OUT_DIR="${OUT_DIR:-runs}"
FOLD="${FOLD:-}"
DISCOVER="${DISCOVER:-1}"
LATEST_N="${LATEST_N:-8}"
CHECK_LATEST_N="${CHECK_LATEST_N:-$LATEST_N}"
WATCH="${WATCH:-0}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
STRICT="${STRICT:-0}"
MIN_PROB_MASS_INSIDE_GT="${MIN_PROB_MASS_INSIDE_GT:-0.95}"
MIN_SUPPORT_RECALL="${MIN_SUPPORT_RECALL:-0.95}"
MIN_SUPPORT_DICE="${MIN_SUPPORT_DICE:-0.02}"
MAX_EMPTY_WHEN_GT_PRESENT="${MAX_EMPTY_WHEN_GT_PRESENT:-0.25}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
MIN_MTIME_EPOCH="${MIN_MTIME_EPOCH:-0}"

fold_dir_name() {
  if [[ -z "$FOLD" ]]; then
    return 1
  fi
  if [[ ! "$FOLD" =~ ^[0-9]+$ ]]; then
    echo "[roi-focus][FAIL] FOLD must be a non-negative integer when set; got: $FOLD" >&2
    return 1
  fi
  printf 'fold_%02d' "$((10#$FOLD))"
}

explicit_metrics_csv() {
  if [[ -n "$METRICS_CSV" ]]; then
    printf '%s\n' "$METRICS_CSV"
    return 0
  fi
  return 1
}

discover_metrics_csv() {
  if [[ "$DISCOVER" != "1" && "$DISCOVER" != "true" && "$DISCOVER" != "yes" ]]; then
    return 1
  fi
  if [[ ! -d "$OUT_DIR" ]]; then
    return 1
  fi
  local awk_filter
  awk_filter='{ if ($1 >= min_mtime) { sub(/^[^ ]+ /, ""); print } }'
  if [[ -n "$FOLD" ]]; then
    local fold_dir
    fold_dir="$(fold_dir_name)"
    find "$OUT_DIR" \( -path "*/${fold_dir}/roi_focus_live.csv" -o -path "*/${fold_dir}/metrics.csv" \) -type f -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | awk -v min_mtime="$MIN_MTIME_EPOCH" "$awk_filter" \
      | head -n 1
    return 0
  fi
  find "$OUT_DIR" \( -name 'roi_focus_live.csv' -o -name 'metrics.csv' \) -type f -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk -v min_mtime="$MIN_MTIME_EPOCH" "$awk_filter" \
    | head -n 1
}

print_wait_message() {
  echo "[roi-focus][WAIT] ROI-focus CSV not found yet." >&2
  echo "  workspace: $WORKSPACE_ROOT" >&2
  echo "  watching: $OUT_DIR/**/{roi_focus_live.csv,metrics.csv}" >&2
  echo "  FOLD=${FOLD:-<unset>}" >&2
  echo "  MIN_MTIME_EPOCH=$MIN_MTIME_EPOCH" >&2
  echo "  Set METRICS_CSV=... only if you want to pin one file exactly." >&2
}

run_once() {
  local metrics
  metrics="$(explicit_metrics_csv || true)"
  if [[ -z "$metrics" || ! -f "$metrics" ]]; then
    local discovered
    discovered="$(discover_metrics_csv || true)"
    if [[ -n "$discovered" && -f "$discovered" ]]; then
      metrics="$discovered"
    fi
  fi
  if [[ -z "$metrics" || ! -f "$metrics" ]]; then
    print_wait_message
    return 2
  fi

  python3 - "$metrics" <<'PY'
from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
latest_n = int(os.environ.get("LATEST_N", "8"))
check_latest_n = int(os.environ.get("CHECK_LATEST_N", str(latest_n)))
strict = os.environ.get("STRICT", "0").strip().lower() in {"1", "true", "yes"}
warmup_epochs = int(os.environ.get("WARMUP_EPOCHS", "0"))

thresholds = {
    "prob_mass_inside_gt": float(os.environ.get("MIN_PROB_MASS_INSIDE_GT", "0.95")),
    "support_recall": float(os.environ.get("MIN_SUPPORT_RECALL", "0.95")),
    "support_dice": float(os.environ.get("MIN_SUPPORT_DICE", "0.02")),
    "support_empty_when_gt_present": float(os.environ.get("MAX_EMPTY_WHEN_GT_PRESENT", "0.25")),
}

def fnum(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")

def fmt(value: str, width: int = 7) -> str:
    val = fnum(value)
    if math.isfinite(val):
        return f"{val:{width}.3f}"
    return f"{'nan':>{width}}"

with metrics_path.open(newline="") as f:
    rows = list(csv.DictReader(f))

required = [
    "epoch",
    "train_roi_focus_pt_prob_mass_inside_gt",
    "train_roi_focus_ln_prob_mass_inside_gt",
    "train_roi_focus_pt_peri_prob_mass_inside_gt",
    "train_roi_focus_pt_support_dice",
    "train_roi_focus_ln_support_dice",
    "train_roi_focus_pt_peri_support_dice",
    "train_roi_focus_pt_support_recall",
    "train_roi_focus_ln_support_recall",
    "train_roi_focus_pt_peri_support_recall",
    "train_roi_focus_pt_support_empty_when_gt_present",
    "train_roi_focus_ln_support_empty_when_gt_present",
    "train_roi_focus_pt_peri_support_empty_when_gt_present",
]
missing = [name for name in required if rows and name not in rows[0]]
if not rows:
    print(f"[roi-focus][WAIT] no rows yet in {metrics_path}")
    raise SystemExit(2)
if missing:
    print("[roi-focus][FAIL] ROI-focus CSV does not contain ROI-focus columns yet.")
    print(f"  source_csv={metrics_path}")
    print("  rerun/resume training with the updated package so ROI-focus columns are written.")
    print("  missing=" + ", ".join(missing))
    raise SystemExit(1)

print(f"[roi-focus] source_csv={metrics_path}")
print(
    "epoch  tf_surv  surv_w  pt_mass  ln_mass  ptp_mass  "
    "pt_dice  ln_dice  ptp_dice  pt_rec  ln_rec  ptp_rec  "
    "pt_empty  ln_empty  ptp_empty"
)
for row in rows[-max(1, latest_n):]:
    print(
        f"{str(row.get('epoch', '')):>5}  "
        f"{fmt(row.get('survival_teacher_force_alpha', ''), 7)}  "
        f"{fmt(row.get('survival_loss_weight', '1.0'), 6)}  "
        f"{fmt(row.get('train_roi_focus_pt_prob_mass_inside_gt', ''))}  "
        f"{fmt(row.get('train_roi_focus_ln_prob_mass_inside_gt', ''))}  "
        f"{fmt(row.get('train_roi_focus_pt_peri_prob_mass_inside_gt', ''))}  "
        f"{fmt(row.get('train_roi_focus_pt_support_dice', ''))}  "
        f"{fmt(row.get('train_roi_focus_ln_support_dice', ''))}  "
        f"{fmt(row.get('train_roi_focus_pt_peri_support_dice', ''))}  "
        f"{fmt(row.get('train_roi_focus_pt_support_recall', ''))}  "
        f"{fmt(row.get('train_roi_focus_ln_support_recall', ''))}  "
        f"{fmt(row.get('train_roi_focus_pt_peri_support_recall', ''))}  "
        f"{fmt(row.get('train_roi_focus_pt_support_empty_when_gt_present', ''))}  "
        f"{fmt(row.get('train_roi_focus_ln_support_empty_when_gt_present', ''))}  "
        f"{fmt(row.get('train_roi_focus_pt_peri_support_empty_when_gt_present', ''))}"
    )

alerts: list[str] = []
checked = 0
check_rows = rows[-max(1, check_latest_n):] if check_latest_n > 0 else rows
for row in check_rows:
    epoch = int(fnum(row.get("epoch", "nan"))) if math.isfinite(fnum(row.get("epoch", "nan"))) else -1
    if epoch <= warmup_epochs:
        continue
    checked += 1
    for roi in ("pt", "ln", "pt_peri"):
        roi_label = {"pt": "PT", "ln": "LN", "pt_peri": "PT-PERI"}.get(roi, roi.upper())
        mass = fnum(row.get(f"train_roi_focus_{roi}_prob_mass_inside_gt", "nan"))
        recall = fnum(row.get(f"train_roi_focus_{roi}_support_recall", "nan"))
        dice = fnum(row.get(f"train_roi_focus_{roi}_support_dice", "nan"))
        empty = fnum(row.get(f"train_roi_focus_{roi}_support_empty_when_gt_present", "nan"))
        if math.isfinite(mass) and mass < thresholds["prob_mass_inside_gt"]:
            alerts.append(f"epoch {epoch} {roi_label} prob_mass_inside_gt={mass:.3f} < {thresholds['prob_mass_inside_gt']:.3f}")
        if math.isfinite(recall) and recall < thresholds["support_recall"]:
            alerts.append(f"epoch {epoch} {roi_label} support_recall={recall:.3f} < {thresholds['support_recall']:.3f}")
        if math.isfinite(dice) and dice < thresholds["support_dice"]:
            alerts.append(f"epoch {epoch} {roi_label} support_dice={dice:.3f} < {thresholds['support_dice']:.3f}")
        if math.isfinite(empty) and empty > thresholds["support_empty_when_gt_present"]:
            alerts.append(f"epoch {epoch} {roi_label} support_empty_when_gt_present={empty:.3f} > {thresholds['support_empty_when_gt_present']:.3f}")

if checked == 0:
    print(f"[roi-focus][INFO] no epochs beyond WARMUP_EPOCHS={warmup_epochs} yet")
elif alerts:
    scope = f"latest {len(check_rows)} row(s)" if check_latest_n > 0 else "all rows"
    print(f"[roi-focus][WARN] {len(alerts)} focus warning(s) beyond warmup in {scope}. First warnings:")
    for item in alerts[:20]:
        print(f"  {item}")
    if strict:
        raise SystemExit(1)
else:
    print(f"[roi-focus][OK] all checked epochs passed ROI-focus thresholds beyond warmup={warmup_epochs}")
PY
}

if [[ "$WATCH" == "1" || "$WATCH" == "true" || "$WATCH" == "yes" ]]; then
  while true; do
    set +e
    run_once
    rc="$?"
    set -e
    if [[ "$rc" != "0" && "$rc" != "2" ]]; then
      exit "$rc"
    fi
    sleep "$INTERVAL_SECONDS"
  done
else
  run_once
fi
