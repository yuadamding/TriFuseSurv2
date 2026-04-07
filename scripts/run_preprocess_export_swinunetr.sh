#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$PACKAGE_DIR/scripts/preprocessing/export_swinunetr.sh" "$@"
