#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENDPOINT="${ENDPOINT:-OS}"
ENDPOINT_LC="$(printf '%s' "$ENDPOINT" | tr '[:upper:]' '[:lower:]')"
OUT_DIR="${OUT_DIR:-runs/best_performance_20hr_75gb_30ep_${ENDPOINT_LC}}"
FOLDS_VALUE="${FOLDS:-0 1 2 3}"

# 12 settings x 4 folds = 48 fold jobs.
# This bundle concentrates on the strongest current region:
# teacher forcing around 22-24, lower backbone LR, and cap-up capacity.
TRIALS_VALUE="${TRIALS:-tf22_h1095 tf23_h1095 v75_tri_h1095_tf24 tf22_lrbb2e4_h1095 tf23_lrbb2e4_h1095 tf24_lrbb2e4_h1095 tf22_capup_h1095 tf23_capup_h1095 tf24_capup_h1095 tf22_capup_lrbb2e4_h1095 tf23_capup_lrbb2e4_h1095 tf24_capup_lrbb2e4_h1095}"

exec env \
  ENDPOINT="$ENDPOINT" \
  FOLDS="$FOLDS_VALUE" \
  TRIALS="$TRIALS_VALUE" \
  OUT_DIR="$OUT_DIR" \
  bash "$PACKAGE_DIR/scripts/run_massive_testing_20hr_75gb_30ep.sh" "$@"
