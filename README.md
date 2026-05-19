# TriFuseSurv2 Runtime Package

Minimal runtime package for contour-aware OPSCC survival training with strict
ROI-focus monitoring.

## Contents

```text
src/trifusesurv2/
  multimodal_survival/train.py       # training entrypoint
  models/                            # survival and ROI-token models
  encoders/                          # clinical and radiomics token encoders
  utils/                             # data loading, survival metrics, ROI focus metrics
  data/                              # lightweight token batch containers
  explain/gradcam_v2_core.py         # optional epoch Grad-CAM core
  schema.py

scripts/
  lib/gpu_utils.sh
  survival/search_roi_constrained_h100.sh
  survival/train_contour_aware_survival.sh
  survival/train_with_roi_focus_watch.sh
  survival/watch_roi_focus_training.sh
```

Everything else from the earlier developer bundle has been removed from this
runtime package: tests, preprocessing commands, legacy search wrappers, DDP/LoRA
alternate launchers, OOF evaluation scripts, and developer build helpers.

## Install

From the package directory:

```bash
python -m pip install -e .
```

If the environment already has the heavy dependencies installed, refresh code
without dependency resolution:

```bash
python -m pip install -e . --no-deps
```

## ROI-Constrained 4-H100 Round-2 Search

```bash
GPU_IDS=0,1,2,3 \
OUT_ROOT=runs/roi_constrained_h100_search_round2_os_fold03 \
DEBUG_FOLD=3 \
WATCH_INTERVAL_SECONDS=60 \
bash scripts/survival/search_roi_constrained_h100.sh
```

The search launcher is centered on the strongest `search_summary.csv` rows from
round 1 (`low_dropout_focus12`, `aux50`, and `focus16`), then pushes memory with
batch-size 2, checkpoint-off, larger token/transformer dimensions, and
`feature_size=120` probes. It runs one trial per GPU slot and enforces the same
ROI constraints on every trial:

- `SURVIVAL_USE_GT_MASKS=1`
- `MASK_GUIDANCE_ALPHA=1.0`
- `STRICT=1`
- `MIN_PROB_MASS_INSIDE_GT=0.95`
- `MIN_SUPPORT_RECALL=0.95`
- `MIN_SUPPORT_DICE=0.02`

It monitors PT, LN, and PT peritumoral ROI support. Results are aggregated into:

```text
<OUT_ROOT>/search_summary.csv
```

The summary includes `peak_vram_mib` and `peak_vram_gb`, sampled with
`nvidia-smi`, so trials can be ranked by both survival score and actual H100
memory use. The target peak is about 77 GB; aggressive trials may fail with OOM,
and the launcher records that failure while the other slots continue.

## Single Trial

```bash
OUT_DIR=runs/contour_aware_survival_os_roi_focus \
DEBUG_FOLD=3 \
WATCH_INTERVAL_SECONDS=60 \
STRICT=1 \
bash scripts/survival/train_with_roi_focus_watch.sh
```

The default training launcher starts fresh (`RESUME=0`), runs a localization-only
warmup, then enables survival loss. Survival train/eval/export passes use
GT-mask ROI teacher forcing by default.

## ROI Metrics

- `pt_mass`, `ln_mass`, `pt_peri_mass`: fraction of effective ROI probability mass
  inside the corresponding GT region.
- `pt_rec`, `ln_rec`, `pt_peri_rec`: fraction of GT support covered by the support
  map.
- `pt_dice`, `ln_dice`, `pt_peri_dice`: support Dice against GT.

`pt_peri_*` refers to the primary-tumor peritumoral shell.
