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
  environment.yml
  install_env.sh
  preprocess_roi_inputs.py
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
bash scripts/install_env.sh
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$PWD/.conda_env"
```

The installer uses Miniforge or Mambaforge only. If Miniforge is not found, or
if the active conda is a system install such as `/opt/conda`, it bootstraps
Miniforge into `$HOME/miniforge3` using `curl`, `wget`, `python3`, or `python`,
then creates a local conda environment at `.conda_env`. Runtime dependencies are
installed through conda/mamba, and TriFuseSurv2 itself is installed as a local
editable package with `pip --no-deps --no-build-isolation` inside that
environment.

The default conda solve pins `pytorch=2.5.1=*cuda12.4*` plus
`pytorch-cuda=12.4` from the `pytorch` and `nvidia` channels. This avoids
CPU-only PyTorch builds that can satisfy looser version pins.

Training launchers default to throughput-oriented settings for one process per
GPU: CUDA TF32 enabled, cuDNN benchmark enabled unless deterministic mode is
requested, eight train DataLoader workers, two eval DataLoader workers,
prefetch factor 4, per-worker decoded-volume caching, sampled ROI-focus
diagnostics, and one BLAS/ITK thread per process to avoid CPU oversubscription
during 4-GPU searches.

Without activating the conda environment, pass its Python explicitly:

```bash
PYTHON_BIN="$PWD/.conda_env/bin/python" bash scripts/survival/search_roi_constrained_h100.sh
```

## ROI-Crop Preprocessing

The survival model can train on ROI-focused crops instead of the original
`128x256x256` volumes. Use crops with margin, not exact outside-mask zeroing,
because the model's `pt_peri` and `ln_peri` tokens need real peritumoral CT
context.

From the workspace root:

```bash
cd /rsrch8/home/bcb/yding4/opscc

PYTHONPATH="$PWD/TriFuseSurv2_package/src" \
"$PWD/TriFuseSurv2_package/.conda_env/bin/python" \
  TriFuseSurv2_package/scripts/preprocess_roi_inputs.py \
  --meta_csv OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv \
  --out_root OPSCC_preprocessed_roi_96x128x128 \
  --output_size 96 128 128 \
  --margin_voxels 24 \
  --workers 4
```

Then launch search against the cropped metadata and matching image size:

```bash
GPU_IDS=0,1,2,3 \
ENDPOINT=OS \
DEBUG_FOLD=3 \
META_CSV=OPSCC_preprocessed_roi_96x128x128/cohort_preprocessed_stage2_roi.csv \
IMG_SIZE="96 128 128" \
SEARCH_PROFILE=roi_crop \
OUT_ROOT=runs/roi_constrained_h100_search_round3_os_fold03_roi96 \
WATCH_INTERVAL_SECONDS=60 \
bash TriFuseSurv2_package/scripts/survival/search_roi_constrained_h100.sh
```

`96x128x128` is about 5.3x fewer voxels than `128x256x256`, so it should
improve GPU step time and may allow more aggressive batch/checkpoint settings.
If nodal disease is far from the primary tumor, use a larger crop such as
`128 160 160` or a larger margin.

When `SEARCH_PROFILE=auto`, any `IMG_SIZE` smaller than `128 256 256` selects
the ROI-crop search profile. That profile uses larger train batches
(`BATCH_SIZE=4/6/8`), larger eval batches, lower gradient accumulation, fixed
normalized-CT body threshold `BODY_CT_THR=0.02`, and less frequent explicit
nonfinite-loss synchronization. Set `NONFINITE_CHECK_EVERY_BATCHES=1` for the
strictest debugging run, or `BODY_CT_THR=auto` only if the CT files are raw HU
instead of normalized preprocessed volumes.

## ROI-Constrained 4-H100 Round-3 Search

```bash
GPU_IDS=0,1,2,3 \
OUT_ROOT=runs/roi_constrained_h100_search_round3_os_fold03 \
DEBUG_FOLD=3 \
WATCH_INTERVAL_SECONDS=60 \
bash scripts/survival/search_roi_constrained_h100.sh
```

The search launcher keeps strict ROI constraints pinned while moving the recipe
toward prediction performance: shorter ROI-focus warmup, nonzero survival loss
during warmup, gradient accumulation, lower backbone learning rate, faster
survival heads, 120-day time bins, later SWA, lighter regularization, and
validation-selected export. It runs one trial per GPU slot and enforces the same
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

GPU-throughput defaults can be overridden when the node is CPU/RAM constrained:

```bash
WORKERS=4
EVAL_WORKERS=1
CACHE_VOLUMES=0
VOLUME_CACHE_SIZE=6
ROI_FOCUS_EVERY_BATCHES=1
```

The search is restart-aware. By default, rerunning the same command with the
same `OUT_ROOT` skips trials marked `done` with return code `0`, retries failed
or missing trials, and resumes interrupted trials that contain a `last.pt`
checkpoint. If a completed trial wrote `cv_summary.json` but the launcher died
before marking it done, the rerun accepts that completed output and records the
trial as done. New attempts write `attemptNN` logs instead of overwriting
earlier logs. Override behavior with:

```bash
SEARCH_FORCE_RERUN=1        # fresh-rerun every trial instead of skipping/resuming
SEARCH_RERUN_FAILED=0       # leave failed trials untouched
SEARCH_RESUME_INTERRUPTED=0 # restart incomplete trials from scratch
SEARCH_ACCEPT_COMPLETE_OUTPUTS=0 # require status.txt/status.rc instead
SEARCH_ALLOW_PARTIAL_RESUME=1 # allow partial checkpoint restores if needed
```

The summary includes `peak_vram_mib` and `peak_vram_gb`, sampled with
`nvidia-smi`, plus mean/peak GPU utilization, so trials can be ranked by both
survival score and actual H100 use. The target peak is about 77 GB; aggressive
trials may fail with OOM, and the launcher records that failure while the other
slots continue.

## Single Trial

```bash
OUT_DIR=runs/contour_aware_survival_os_roi_focus \
DEBUG_FOLD=3 \
WATCH_INTERVAL_SECONDS=60 \
STRICT=1 \
bash scripts/survival/train_with_roi_focus_watch.sh
```

The default training launcher starts fresh (`RESUME=0`), runs an ROI-focus
warmup with nonzero survival loss, and uses GT-mask ROI teacher forcing for
survival train/eval/export by default.

## ROI Metrics

- `pt_mass`, `ln_mass`, `pt_peri_mass`: fraction of effective ROI probability mass
  inside the corresponding GT region.
- `pt_rec`, `ln_rec`, `pt_peri_rec`: fraction of GT support covered by the support
  map.
- `pt_dice`, `ln_dice`, `pt_peri_dice`: support Dice against GT.

`pt_peri_*` refers to the primary-tumor peritumoral shell.
