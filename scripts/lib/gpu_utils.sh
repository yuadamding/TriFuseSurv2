#!/usr/bin/env bash

# Shared GPU detection helpers for TriFuseSurv shell scripts.

tf_detect_gpu_ids() {
  local -a ids=()
  local visible="${CUDA_VISIBLE_DEVICES:-}"
  local item

  if [[ -n "$visible" && "$visible" != "NoDevFiles" ]]; then
    IFS=',' read -r -a ids <<<"$visible"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    while IFS= read -r item; do
      item="$(printf '%s' "$item" | tr -d '[:space:]')"
      [[ -n "$item" ]] && ids+=("$item")
    done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
  elif command -v python3 >/dev/null 2>&1; then
    while IFS= read -r item; do
      item="$(printf '%s' "$item" | tr -d '[:space:]')"
      [[ -n "$item" ]] && ids+=("$item")
    done < <(python3 - <<'PY'
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
