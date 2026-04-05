#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-}"

"$PYTHON_BIN" -m venv "$VENV_DIR"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ -n "$TORCH_INDEX_URL" ]]; then
  python -m pip install --index-url "$TORCH_INDEX_URL" torch
fi

python -m pip install -e .
python -m pip install opencv-python-headless

if [[ -n "$EXTRA_PIP_PACKAGES" ]]; then
  python -m pip install $EXTRA_PIP_PACKAGES
fi

cat <<EOF
[done] environment installed in $VENV_DIR

Activate it with:
  source $VENV_DIR/bin/activate

Then run, for example:
  ./scripts/run_preprocess_export_swinunetr.sh
  ./scripts/run_make_cv_splits.sh
  ./scripts/run_stage1_pretrain_pt.sh
  ./scripts/run_stage2_survival.sh
EOF
