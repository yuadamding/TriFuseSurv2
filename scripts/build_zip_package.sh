#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PACKAGE_NAME="${PACKAGE_NAME:-TriFuseSurv2_package}"
TOP_LEVEL_ZIP_PATH="${TOP_LEVEL_ZIP_PATH:-../${PACKAGE_NAME}.zip}"
PACKAGE_PROFILE="${PACKAGE_PROFILE:-compact}"
TMP_DIR="$(mktemp -d)"
STAGE_DIR="$TMP_DIR/$PACKAGE_NAME"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$STAGE_DIR"

case "$PACKAGE_PROFILE" in
  compact)
    copy_items=(
      README.md
      pyproject.toml
      src
    )
    ;;
  full)
    copy_items=(
      README.md
      pyproject.toml
      scripts
      src
      tests
    )
    if [[ -d docs ]]; then
      copy_items+=(docs)
    fi
    ;;
  *)
    echo "[error] PACKAGE_PROFILE must be compact or full, got: $PACKAGE_PROFILE" >&2
    exit 1
    ;;
esac

mkdir -p "$(dirname "$TOP_LEVEL_ZIP_PATH")"
TOP_LEVEL_ZIP_PATH="$(cd "$(dirname "$TOP_LEVEL_ZIP_PATH")" && pwd)/$(basename "$TOP_LEVEL_ZIP_PATH")"

rsync -a \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  "${copy_items[@]}" \
  "$STAGE_DIR"/

rm -f "$TOP_LEVEL_ZIP_PATH"

(
  cd "$TMP_DIR"
  zip -qr "$TOP_LEVEL_ZIP_PATH" "$PACKAGE_NAME"
)

echo "[done] wrote $TOP_LEVEL_ZIP_PATH (profile=$PACKAGE_PROFILE)"
