#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

META_CSV="${META_CSV:-OPSCC_preprocessed_128/cohort_preprocessed.csv}"
OUT_DIR="${OUT_DIR:-runs/seg_overfit_ln_big_stable}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
DEVICE="${DEVICE:-cuda:0}"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
python3 -m trifusesurv.segmentation.train \
  --meta_csv "$META_CSV" \
  --train_mode all \
  --out_dir "$OUT_DIR" \
  --mask_col mask_nodal_out_path \
  --img_size 128 256 256 \
  --feature_size 96 \
  --depths 2 2 18 2 \
  --num_heads 3 6 12 24 \
  --drop_rate 0 \
  --attn_drop_rate 0 \
  --dropout_path_rate 0 \
  --epochs 100 \
  --batch_size 1 \
  --workers 16 \
  --amp \
  --use_checkpoint \
  --device "$DEVICE" \
  --lr 5e-5 \
  --wd 0 \
  --grad_clip 1.0 \
  --loss bce_dice \
  --max_pos_weight 200 \
  --pos_oversample 1 \
  "$@"
