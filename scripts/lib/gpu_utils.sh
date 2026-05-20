#!/usr/bin/env bash

# Shared GPU detection helpers for TriFuseSurv shell scripts.

tf_detect_gpu_ids() {
  local -a ids=()
  local visible="${CUDA_VISIBLE_DEVICES:-}"
  local python_bin="${PYTHON_BIN:-python3}"
  local item

  if [[ -n "$visible" && "$visible" != "NoDevFiles" ]]; then
    IFS=',' read -r -a ids <<<"$visible"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    while IFS= read -r item; do
      item="$(printf '%s' "$item" | tr -d '[:space:]')"
      [[ -n "$item" ]] && ids+=("$item")
    done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
  elif command -v "$python_bin" >/dev/null 2>&1; then
    while IFS= read -r item; do
      item="$(printf '%s' "$item" | tr -d '[:space:]')"
      [[ -n "$item" ]] && ids+=("$item")
    done < <("$python_bin" - <<'PY'
try:
    import torch
    for i in range(int(torch.cuda.device_count())):
        print(i)
except Exception:
    pass
PY
)
  fi

  for item in "${ids[@]}"; do
    item="$(printf '%s' "$item" | tr -d '[:space:]')"
    [[ -n "$item" ]] && printf '%s\n' "$item"
  done
}

tf_first_gpu_id() {
  local -a ids=()
  mapfile -t ids < <(tf_detect_gpu_ids)
  if (( ${#ids[@]} == 0 )); then
    return 1
  fi
  printf '%s\n' "${ids[0]}"
}

tf_detect_gpu_ids_by_free_mem() {
  local min_free_mb="${1:-0}"
  local -a visible_ids=()
  local -A allowed=()
  local line idx total used free

  mapfile -t visible_ids < <(tf_detect_gpu_ids)
  if (( ${#visible_ids[@]} == 0 )); then
    return 0
  fi
  for idx in "${visible_ids[@]}"; do
    allowed["$idx"]=1
  done

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' "${visible_ids[@]}"
    return 0
  fi

  while IFS=',' read -r idx total used free; do
    idx="$(printf '%s' "$idx" | tr -d '[:space:]')"
    free="$(printf '%s' "$free" | tr -d '[:space:]')"
    [[ -n "$idx" && -n "$free" ]] || continue
    [[ -n "${allowed[$idx]:-}" ]] || continue
    if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= min_free_mb )); then
      printf '%s,%s\n' "$free" "$idx"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null || true) \
    | sort -t, -k1,1nr \
    | cut -d, -f2
}

tf_wait_for_any_tracked_pid() {
  local -n _pids_ref="$1"
  local -n _meta1_ref="$2"
  local -n _meta2_ref="$3"
  local -n _meta3_ref="$4"
  local _sleep_secs="${5:-1}"
  local _idx _pid _status

  TF_WAIT_PID=""
  TF_WAIT_STATUS=""
  TF_WAIT_META1=""
  TF_WAIT_META2=""
  TF_WAIT_META3=""

  while (( ${#_pids_ref[@]} > 0 )); do
    for _idx in "${!_pids_ref[@]}"; do
      _pid="${_pids_ref[$_idx]}"
      if kill -0 "$_pid" 2>/dev/null; then
        continue
      fi

      if wait "$_pid"; then
        _status=0
      else
        _status=$?
      fi

      TF_WAIT_PID="$_pid"
      TF_WAIT_STATUS="$_status"
      TF_WAIT_META1="${_meta1_ref[$_idx]:-}"
      TF_WAIT_META2="${_meta2_ref[$_idx]:-}"
      TF_WAIT_META3="${_meta3_ref[$_idx]:-}"

      unset '_pids_ref[_idx]' '_meta1_ref[_idx]' '_meta2_ref[_idx]' '_meta3_ref[_idx]'
      _pids_ref=("${_pids_ref[@]}")
      _meta1_ref=("${_meta1_ref[@]}")
      _meta2_ref=("${_meta2_ref[@]}")
      _meta3_ref=("${_meta3_ref[@]}")
      return 0
    done
    sleep "$_sleep_secs"
  done

  return 1
}

tf_require_python_modules() {
  local python_bin="${PYTHON_BIN:-python3}"
  if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "[error] required python executable not found: $python_bin" >&2
    return 1
  fi

  "$python_bin" - "$@" <<'PY'
import importlib
import sys

modules = sys.argv[1:]
missing = []
for mod in modules:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)

if missing:
    sys.stderr.write(
        "[error] active Python environment is missing required modules: "
        + ", ".join(missing)
        + "\n"
    )
    sys.stderr.write(
        "[error] activate the package environment first, or install runtime dependencies with:\n"
        "  cd TriFuseSurv2_package\n"
        "  bash scripts/install_env.sh\n"
        "  source \"$(conda info --base)/etc/profile.d/conda.sh\"\n"
        "  conda activate \"$PWD/.conda_env\"\n"
        "\n"
        "Without activation, set PYTHON_BIN to the Miniforge env Python.\n"
    )
    raise SystemExit(1)
PY
}

tf_require_torch_cuda() {
  local python_bin="${PYTHON_BIN:-python3}"
  if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "[error] required python executable not found: $python_bin" >&2
    return 1
  fi

  "$python_bin" - <<'PY'
import os
import sys

try:
    import torch
except Exception as exc:
    sys.stderr.write(f"[error] failed to import torch: {exc}\n")
    raise SystemExit(1)

print(
    "[cuda-check] "
    f"torch={getattr(torch, '__version__', '<unknown>')} "
    f"torch_cuda={torch.version.cuda} "
    f"cuda_available={torch.cuda.is_available()} "
    f"device_count={torch.cuda.device_count()} "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
    flush=True,
)

if torch.version.cuda is None:
    sys.stderr.write(
        "[error] PyTorch is CPU-only in the active environment. Recreate the "
        "Miniforge env with CUDA PyTorch:\n"
        "  cd TriFuseSurv2_package\n"
        "  rm -rf .conda_env\n"
        "  PYTORCH_SPEC='pytorch=2.5.1=*cuda12.4*' PYTORCH_CUDA_SPEC=pytorch-cuda=12.4 "
        "bash scripts/install_env.sh\n"
    )
    raise SystemExit(1)

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    sys.stderr.write(
        "[error] CUDA PyTorch is installed, but no CUDA device is visible. "
        "Run this from a GPU node/session and verify nvidia-smi works.\n"
    )
    raise SystemExit(1)
PY
}
