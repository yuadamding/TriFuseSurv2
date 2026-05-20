#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_PREFIX="${CONDA_ENV_PREFIX:-$ROOT_DIR/.conda_env}"
MINIFORGE_HOME="${MINIFORGE_HOME:-}"
INSTALL_MINIFORGE="${INSTALL_MINIFORGE:-1}"
MINIFORGE_INSTALLER_URL="${MINIFORGE_INSTALLER_URL:-https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh}"
CONDA_SOLVER="${CONDA_SOLVER:-auto}"
CONDA_CHANNEL_ARGS="${CONDA_CHANNEL_ARGS:---override-channels -c pytorch -c nvidia -c conda-forge}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PYTORCH_SPEC="${PYTORCH_SPEC:-pytorch=2.5.1=*cuda12.4*}"
PYTORCH_CUDA_SPEC="${PYTORCH_CUDA_SPEC:-pytorch-cuda=12.4}"
LOCAL_EDITABLE_FLAGS="${LOCAL_EDITABLE_FLAGS:---no-build-isolation --no-deps}"
EXTRA_CONDA_PACKAGES="${EXTRA_CONDA_PACKAGES:-}"
EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-}"

is_truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

find_miniforge_conda() {
  local candidate
  for candidate in \
    "${MINIFORGE_HOME:+$MINIFORGE_HOME/bin/conda}" \
    "$HOME/miniforge3/bin/conda" \
    "$HOME/mambaforge/bin/conda" \
    "${CONDA_EXE:-}"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi

  return 1
}

install_miniforge() {
  local target="${MINIFORGE_HOME:-$HOME/miniforge3}"
  local installer
  if [[ -x "$target/bin/conda" ]]; then
    echo "[env] using existing Miniforge at: $target"
    MINIFORGE_HOME="$target"
    CONDA_BIN="$target/bin/conda"
    return 0
  fi

  installer="$(mktemp -t miniforge-installer.XXXXXX.sh)"
  echo "[env] installing Miniforge to: $target"
  if command -v curl >/dev/null 2>&1; then
    curl -L --retry 5 --retry-delay 3 -o "$installer" "$MINIFORGE_INSTALLER_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=5 --waitretry=3 -O "$installer" "$MINIFORGE_INSTALLER_URL"
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "$MINIFORGE_INSTALLER_URL" "$installer" <<'PY'
from __future__ import annotations

import shutil
import sys
import urllib.request

url, out_path = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(url, timeout=120) as response, open(out_path, "wb") as out_file:
    shutil.copyfileobj(response, out_file)
PY
  elif command -v python >/dev/null 2>&1; then
    python - "$MINIFORGE_INSTALLER_URL" "$installer" <<'PY'
from __future__ import annotations

import shutil
import sys
import urllib.request

url, out_path = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(url, timeout=120) as response, open(out_path, "wb") as out_file:
    shutil.copyfileobj(response, out_file)
PY
  else
    echo "[error] installing Miniforge requires curl, wget, python3, or python." >&2
    rm -f "$installer"
    exit 1
  fi
  bash "$installer" -b -p "$target"
  rm -f "$installer"
  MINIFORGE_HOME="$target"
  CONDA_BIN="$target/bin/conda"
}

CONDA_BIN="$(find_miniforge_conda || true)"
if [[ -z "$CONDA_BIN" ]]; then
  if is_truthy "$INSTALL_MINIFORGE"; then
    install_miniforge
  else
  cat >&2 <<'EOF'
[error] Miniforge conda was not found.

Install Miniforge, then rerun:
  wget -O Miniforge3-Linux-x86_64.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
  export PATH="$HOME/miniforge3/bin:$PATH"
  bash TriFuseSurv2_package/scripts/install_env.sh

This installer bootstraps Miniforge by default. To disable bootstrap:
  INSTALL_MINIFORGE=0 bash TriFuseSurv2_package/scripts/install_env.sh
EOF
  exit 1
  fi
fi

CONDA_BASE="$("$CONDA_BIN" info --base)"
CONDA_BASE_LC="$(printf '%s' "$CONDA_BASE" | tr '[:upper:]' '[:lower:]')"
if [[ "$CONDA_BASE_LC" != *miniforge* && "$CONDA_BASE_LC" != *mambaforge* ]]; then
  if is_truthy "$INSTALL_MINIFORGE"; then
    install_miniforge
    CONDA_BASE="$("$CONDA_BIN" info --base)"
    CONDA_BASE_LC="$(printf '%s' "$CONDA_BASE" | tr '[:upper:]' '[:lower:]')"
  fi
fi

if [[ "$CONDA_BASE_LC" != *miniforge* && "$CONDA_BASE_LC" != *mambaforge* ]]; then
  cat >&2 <<EOF
[error] Found conda, but it does not look like Miniforge/Mambaforge:
  $CONDA_BASE

Set MINIFORGE_HOME to your Miniforge install, or put Miniforge first on PATH.
Or bootstrap Miniforge into \$HOME/miniforge3:
  bash TriFuseSurv2_package/scripts/install_env.sh

To disable bootstrap:
  INSTALL_MINIFORGE=0 bash TriFuseSurv2_package/scripts/install_env.sh
EOF
  exit 1
fi

SOLVER_BIN="$CONDA_BIN"
if [[ "$CONDA_SOLVER" == "mamba" || "$CONDA_SOLVER" == "auto" ]]; then
  if [[ -x "$CONDA_BASE/bin/mamba" ]]; then
    SOLVER_BIN="$CONDA_BASE/bin/mamba"
  elif [[ "$CONDA_SOLVER" == "mamba" ]]; then
    echo "[error] CONDA_SOLVER=mamba was requested, but mamba is not installed in $CONDA_BASE" >&2
    exit 1
  fi
fi

read -r -a channel_args <<< "$CONDA_CHANNEL_ARGS"

conda_packages=(
  "python=$PYTHON_VERSION"
  pip
  "setuptools>=68"
  wheel
  einops
  monai
  numpy
  pandas
  pydicom
  simpleitk
  scikit-learn
  matplotlib
)
if [[ -n "$PYTORCH_SPEC" && "$PYTORCH_SPEC" != "none" ]]; then
  conda_packages+=("$PYTORCH_SPEC")
fi
if [[ -n "$PYTORCH_CUDA_SPEC" && "$PYTORCH_CUDA_SPEC" != "none" ]]; then
  conda_packages+=("$PYTORCH_CUDA_SPEC")
fi
if [[ -n "$EXTRA_CONDA_PACKAGES" ]]; then
  read -r -a extra_conda_packages <<< "$EXTRA_CONDA_PACKAGES"
  conda_packages+=("${extra_conda_packages[@]}")
fi

mkdir -p "$(dirname "$ENV_PREFIX")"

if [[ -d "$ENV_PREFIX/conda-meta" ]]; then
  echo "[env] updating Miniforge environment: $ENV_PREFIX"
  "$SOLVER_BIN" install -y -p "$ENV_PREFIX" "${channel_args[@]}" "${conda_packages[@]}"
else
  echo "[env] creating Miniforge environment: $ENV_PREFIX"
  "$SOLVER_BIN" create -y -p "$ENV_PREFIX" "${channel_args[@]}" "${conda_packages[@]}"
fi

read -r -a local_editable_flags <<< "$LOCAL_EDITABLE_FLAGS"
"$CONDA_BIN" run -p "$ENV_PREFIX" python -m pip install --upgrade "${local_editable_flags[@]}" -e .

if [[ -n "$EXTRA_PIP_PACKAGES" ]]; then
  # Last-resort hook for local/private wheels only; runtime deps above are conda-managed.
  # shellcheck disable=SC2086
  "$CONDA_BIN" run -p "$ENV_PREFIX" python -m pip install --no-deps $EXTRA_PIP_PACKAGES
fi

"$CONDA_BIN" run -p "$ENV_PREFIX" python - <<'PY'
import importlib
import sys

required = {
    "einops": "einops",
    "numpy": "numpy",
    "pandas": "pandas",
    "pydicom": "pydicom",
    "SimpleITK": "SimpleITK",
    "sklearn": "scikit-learn",
    "torch": "pytorch",
    "monai": "monai",
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
    raise SystemExit(1)

import torch

if torch.version.cuda is None:
    sys.stderr.write(
        "[error] installed PyTorch is CPU-only. Recreate the environment with the "
        "pytorch and nvidia channels, for example:\n"
        "  rm -rf .conda_env\n"
        "  PYTORCH_SPEC='pytorch=2.5.1=*cuda12.4*' PYTORCH_CUDA_SPEC=pytorch-cuda=12.4 "
        "bash scripts/install_env.sh\n"
    )
    raise SystemExit(1)
PY

cat <<EOF
[done] Miniforge environment installed at:
  $ENV_PREFIX

Activate it with:
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$ENV_PREFIX"

Then run, for example:
  ./scripts/survival/train_contour_aware_survival.sh
  ./scripts/survival/search_roi_constrained_h100.sh

Without activation, pass the environment Python explicitly:
  PYTHON_BIN="$ENV_PREFIX/bin/python" bash scripts/survival/search_roi_constrained_h100.sh
EOF
