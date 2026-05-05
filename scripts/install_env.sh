#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
LOCAL_ENV_ROOT="${LOCAL_ENV_ROOT:-$ROOT_DIR/.install_env}"
LOCAL_PIP_CACHE_DIR="${LOCAL_PIP_CACHE_DIR:-$LOCAL_ENV_ROOT/pip-cache}"
LOCAL_PIP_TMP_DIR="${LOCAL_PIP_TMP_DIR:-$LOCAL_ENV_ROOT/tmp}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-}"
OPENCV_HEADLESS_SPEC="${OPENCV_HEADLESS_SPEC:-opencv-python-headless==4.10.0.84}"
UPGRADE_PIP_TOOLS="${UPGRADE_PIP_TOOLS:-0}"
FORCE_RECREATE_VENV="${FORCE_RECREATE_VENV:-0}"
FORCE_FULL_INSTALL="${FORCE_FULL_INSTALL:-0}"
AUTO_REPAIR_OPENCV="${AUTO_REPAIR_OPENCV:-1}"

prepare_local_install_dirs() {
  mkdir -p "$LOCAL_ENV_ROOT" "$LOCAL_PIP_CACHE_DIR" "$LOCAL_PIP_TMP_DIR"
  export TMPDIR="$LOCAL_PIP_TMP_DIR"
  export PIP_CACHE_DIR="$LOCAL_PIP_CACHE_DIR"
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  export PIP_NO_INPUT=1
}

create_or_reuse_venv() {
  if [[ "$FORCE_RECREATE_VENV" == "1" && -d "$VENV_DIR" ]]; then
    rm -rf "$VENV_DIR"
  fi

  if [[ ! -d "$VENV_DIR" ]]; then
    echo "[install] creating virtualenv at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  else
    echo "[install] reusing existing virtualenv at $VENV_DIR"
  fi
}

missing_required_packages() {
  python - <<'PY'
import importlib

required = {
    "einops": "einops",
    "numpy": "numpy",
    "pandas": "pandas",
    "pydicom": "pydicom",
    "SimpleITK": "SimpleITK",
    "rt_utils": "rt-utils",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "monai": "monai",
    "cv2": "opencv-python-headless",
}
missing = []
for mod_name, pkg_name in required.items():
    try:
        importlib.import_module(mod_name)
    except Exception:
        missing.append(pkg_name)
print(" ".join(missing))
PY
}

verify_runtime() {
  python - <<'PY'
import importlib
import sys

required = {
    "einops": "einops",
    "numpy": "numpy",
    "pandas": "pandas",
    "pydicom": "pydicom",
    "SimpleITK": "SimpleITK",
    "rt_utils": "rt-utils",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "monai": "monai",
    "cv2": "opencv-python-headless",
    "trifusesurv2": "trifusesurv2",
}
missing = []
for mod_name, pkg_name in required.items():
    try:
        importlib.import_module(mod_name)
    except Exception:
        missing.append(pkg_name)

if missing:
    sys.stderr.write(
        "[error] environment is still missing required packages: "
        + ", ".join(missing)
        + "\n"
    )
    sys.stderr.write(
        "[error] try: python -m pip install --upgrade " + " ".join(missing) + "\n"
    )
    raise SystemExit(1)
PY
}

repair_opencv_headless() {
  echo "[install] repairing OpenCV headless runtime"
  python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless || true
  python -m pip install --upgrade "$OPENCV_HEADLESS_SPEC"
}

maybe_install_torch_from_custom_index() {
  if [[ -z "$TORCH_INDEX_URL" ]]; then
    return 0
  fi

  if python - <<'PY'
import importlib
import sys
try:
    importlib.import_module("torch")
except Exception:
    raise SystemExit(1)
PY
  then
    return 0
  fi

  echo "[install] torch missing; installing from TORCH_INDEX_URL"
  python -m pip install --index-url "$TORCH_INDEX_URL" torch
}

prepare_local_install_dirs
create_or_reuse_venv

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

if [[ "$UPGRADE_PIP_TOOLS" == "1" ]]; then
  echo "[install] upgrading pip/setuptools/wheel"
  python -m pip install --upgrade pip setuptools wheel
fi

maybe_install_torch_from_custom_index

missing_before="$(missing_required_packages)"
if [[ "$FORCE_FULL_INSTALL" == "1" || -n "$missing_before" ]]; then
  echo "[install] installing package and dependencies"
  python -m pip install -e .
else
  echo "[install] dependencies already present; refreshing editable package only"
  python -m pip install -e . --no-deps
fi

if [[ "$AUTO_REPAIR_OPENCV" == "1" ]]; then
  if ! python - <<'PY'
import importlib
import sys
try:
    importlib.import_module("cv2")
except Exception:
    raise SystemExit(1)
PY
  then
    repair_opencv_headless
  fi
fi

verify_runtime

if [[ -n "$EXTRA_PIP_PACKAGES" ]]; then
  python -m pip install $EXTRA_PIP_PACKAGES
fi

cat <<EOF
[done] environment installed in $VENV_DIR

Local installer state:
  PIP cache: $LOCAL_PIP_CACHE_DIR
  temp/build: $LOCAL_PIP_TMP_DIR

Activate it with:
  source $VENV_DIR/bin/activate

Fast refresh next time:
  source $VENV_DIR/bin/activate
  python -m pip install -e . --no-deps

Then run, for example:
  ./scripts/run_contour_aware_cindex_search_75gb_30ep.sh
  ./scripts/run_v75_tri_h1095_tf24_4fold.sh
EOF
