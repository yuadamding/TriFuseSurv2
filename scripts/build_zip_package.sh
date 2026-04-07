#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PACKAGE_NAME="${PACKAGE_NAME:-TriFuseSurv_package}"
DIST_DIR="${DIST_DIR:-dist}"
STAGE_DIR="$DIST_DIR/$PACKAGE_NAME"
ZIP_PATH="$DIST_DIR/${PACKAGE_NAME}.zip"
TOP_LEVEL_ZIP_PATH="${TOP_LEVEL_ZIP_PATH:-../${PACKAGE_NAME}.zip}"

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

rsync -a \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  README.md \
  pyproject.toml \
  docs \
  scripts \
  src \
  "$STAGE_DIR"/

rm -f "$ZIP_PATH"
(
  cd "$DIST_DIR"
  zip -qr "${PACKAGE_NAME}.zip" "$PACKAGE_NAME"
)

cp "$ZIP_PATH" "$TOP_LEVEL_ZIP_PATH"

echo "[done] wrote $ZIP_PATH"
