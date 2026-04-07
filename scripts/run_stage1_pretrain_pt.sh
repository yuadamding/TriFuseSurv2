#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$PACKAGE_DIR/scripts/stage1/pretrain_pt.sh" "$@"
