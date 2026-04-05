#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

META_CSV="${META_CSV:-OPSCC_preprocessed_128/cohort_preprocessed.csv}"
QC_REPORT="${QC_REPORT:-}"
QC_POLICY="${QC_POLICY:-none}"
QC_DROP_AIR_GT="${QC_DROP_AIR_GT:-0}"
ENDPOINT="${ENDPOINT:-OS}"
CV_FOLDS="${CV_FOLDS:-4}"
VAL_FRAC="${VAL_FRAC:-0.2}"
SPLIT_SEED="${SPLIT_SEED:-1}"
OUT_DIR="${OUT_DIR:-runs/opscc_splits_os_seed1}"

python3 -m trifusesurv.preprocessing.make_cv_splits \
  --meta_csv "$META_CSV" \
  --qc_report "$QC_REPORT" \
  --qc_policy "$QC_POLICY" \
  --qc_drop_air_gt "$QC_DROP_AIR_GT" \
  --endpoint "$ENDPOINT" \
  --cv_folds "$CV_FOLDS" \
  --val_frac "$VAL_FRAC" \
  --split_seed "$SPLIT_SEED" \
  --out_dir "$OUT_DIR" \
  "$@"
