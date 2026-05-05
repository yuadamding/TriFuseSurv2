#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PACKAGE_DIR/.." && pwd)"
cd "$WORKSPACE_ROOT"

ENDPOINT="${ENDPOINT:-OS}"
ENDPOINT_LC="$(printf '%s' "$ENDPOINT" | tr '[:upper:]' '[:lower:]')"
OUT_DIR_BASE="${OUT_DIR_BASE:-runs/optimal_setting_search_75gb_30ep_v2_${ENDPOINT_LC}}"
FOLDS_VALUE="${FOLDS:-0 1 2 3}"
SEARCH_PROFILE="${SEARCH_PROFILE:-balanced}"
RADIOMICS_SOURCE="${RADIOMICS_SOURCE:-cohort_radiomics_patient_wide.csv}"
NODE_TOPOLOGY_DIR="${NODE_TOPOLOGY_DIR:-}"
MODEL_VERSION="${MODEL_VERSION:-v2}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_OUTER_TEST_SCORING="${ALLOW_OUTER_TEST_SCORING:-0}"
LOCKED_CONFIG_HASH="${LOCKED_CONFIG_HASH:-}"

# Phase 1 is deliberately small: cover model width, radiomics-token PCA width,
# and teacher-forcing duration without spending the whole allocation.
case "$SEARCH_PROFILE" in
  quick)
    PHASE1_TRIALS_DEFAULT="v2_anchor_m256_rad16_tf16 v2_m384_rad24_tf20"
    PHASE2_TRIALS_DEFAULT="v2_m256_rad24_tf20 v2_m384_rad24_tf24_locweak"
    ;;
  balanced)
    PHASE1_TRIALS_DEFAULT="v2_anchor_m256_rad16_tf16 v2_m256_rad24_tf20 v2_m384_rad16_tf16 v2_m384_rad24_tf20"
    PHASE2_TRIALS_DEFAULT="v2_m256_rad32_tf24 v2_m384_rad32_tf24 v2_m384_layers3_rad24_tf20 v2_m384_rad24_tf24_locweak v2_m384_nomultiscale_rad24_tf20"
    ;;
  broad)
    PHASE1_TRIALS_DEFAULT="v2_anchor_m256_rad16_tf16 v2_m256_rad24_tf20 v2_m256_rad32_tf24 v2_m384_rad16_tf16 v2_m384_rad24_tf20 v2_m384_rad32_tf24"
    PHASE2_TRIALS_DEFAULT="v2_m384_layers3_rad24_tf20 v2_m512_rad16_tf16 v2_m384_rad24_tf24_locweak v2_m384_nomultiscale_rad24_tf20"
    ;;
  *)
    echo "[optimal-search][error] SEARCH_PROFILE must be quick, balanced, or broad; got: $SEARCH_PROFILE" >&2
    exit 1
    ;;
esac

PHASE1_TRIALS="${PHASE1_TRIALS:-$PHASE1_TRIALS_DEFAULT}"
PHASE2_TRIALS="${PHASE2_TRIALS:-$PHASE2_TRIALS_DEFAULT}"
RUN_PHASE2="${RUN_PHASE2:-1}"
WEIGHTS_TO_SCORE_VALUE="${WEIGHTS_TO_SCORE:-ema best swa last}"

if [[ "$MODEL_VERSION" != "v2" ]]; then
  echo "[optimal-search][warn] MODEL_VERSION=$MODEL_VERSION; this script is tuned for v2 habitat-aligned searches." >&2
fi

echo "[optimal-search] profile=$SEARCH_PROFILE endpoint=$ENDPOINT folds=$FOLDS_VALUE"
echo "[optimal-search] out_dir_base=$OUT_DIR_BASE"
echo "[optimal-search] radiomics=$RADIOMICS_SOURCE"
if [[ -n "$NODE_TOPOLOGY_DIR" ]]; then
  echo "[optimal-search] node_topology_dir=$NODE_TOPOLOGY_DIR"
else
  echo "[optimal-search] node_topology_dir=<none; node/topology tokens will be absent>"
fi

echo "[optimal-search] phase 1/2: v2 coarse optimal-setting search"
env \
  MODEL_VERSION="$MODEL_VERSION" \
  ENDPOINT="$ENDPOINT" \
  FOLDS="$FOLDS_VALUE" \
  TRIALS="$PHASE1_TRIALS" \
  WEIGHTS_TO_SCORE="$WEIGHTS_TO_SCORE_VALUE" \
  RADIOMICS_SOURCE="$RADIOMICS_SOURCE" \
  NODE_TOPOLOGY_DIR="$NODE_TOPOLOGY_DIR" \
  ALLOW_OUTER_TEST_SCORING="$ALLOW_OUTER_TEST_SCORING" \
  LOCKED_CONFIG_HASH="$LOCKED_CONFIG_HASH" \
  DRY_RUN="$DRY_RUN" \
  OUT_DIR="$OUT_DIR_BASE/phase1_coarse" \
  bash "$PACKAGE_DIR/scripts/run_contour_aware_cindex_search_75gb_30ep.sh" "$@"

if [[ "$RUN_PHASE2" != "1" ]]; then
  echo "[optimal-search] RUN_PHASE2=$RUN_PHASE2; stopping after phase 1"
  exit 0
fi

echo "[optimal-search] phase 2/2: v2 high-value follow-up search"
exec env \
  MODEL_VERSION="$MODEL_VERSION" \
  ENDPOINT="$ENDPOINT" \
  FOLDS="$FOLDS_VALUE" \
  TRIALS="$PHASE2_TRIALS" \
  WEIGHTS_TO_SCORE="$WEIGHTS_TO_SCORE_VALUE" \
  RADIOMICS_SOURCE="$RADIOMICS_SOURCE" \
  NODE_TOPOLOGY_DIR="$NODE_TOPOLOGY_DIR" \
  ALLOW_OUTER_TEST_SCORING="$ALLOW_OUTER_TEST_SCORING" \
  LOCKED_CONFIG_HASH="$LOCKED_CONFIG_HASH" \
  DRY_RUN="$DRY_RUN" \
  OUT_DIR="$OUT_DIR_BASE/phase2_followup" \
  bash "$PACKAGE_DIR/scripts/run_contour_aware_cindex_search_75gb_30ep.sh" "$@"
