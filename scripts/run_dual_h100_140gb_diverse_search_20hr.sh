#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACKAGE_DIR"

TRIALS=(
  dual140_div_anchor_tf23_lrbb2e4_f2048
  dual140_div_wide_tf23_lrbb2e4_f2304
  dual140_div_nomulti_tf23_lrbb2e4_f2048
  dual140_div_shell3_tf23_lrbb2e4_f2048
  dual140_div_aux020_tf23_lrbb2e4_f2048
  dual140_div_time120_tf23_lrbb2e4_f2048
)

OUT_DIR="${OUT_DIR:-runs/dual_h100_140gb_diverse_search_20hr_os_ddp}"
EPOCHS="${EPOCHS:-40}"

echo "[diverse20] launching dual-H100 diverse search"
echo "[diverse20] epochs=$EPOCHS"
echo "[diverse20] trials=${TRIALS[*]}"
echo "[diverse20] out_dir=$OUT_DIR"

TRIALS="${TRIALS[*]}" \
OUT_DIR="$OUT_DIR" \
EPOCHS="$EPOCHS" \
bash "$PACKAGE_DIR/scripts/run_dual_h100_140gb_best_perf_20hr.sh"
