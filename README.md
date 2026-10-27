# TriFuseSurv2 Runtime Package

OPSCC survival training with shared physical CT/radiomics compartments and
controlled F0–F4 fusion alternatives. The default is **F3, verified regions,
expert contours**. These are implementation capabilities, not evidence of
improved prognosis or clinical validity.

Start with the [anatomical correspondence workflow](docs/anatomical_correspondence.md)
for input requirements, migration, comparison settings, and remaining study gates.
The principal comparison recipe disables both image-token dropout paths.
`Patient_ID` is the default patient key, with an unambiguous `patient_id` alias
accepted. Acquisition and provenance identifiers retain their original strings.

## Contents

```text
src/trifusesurv2/
  multimodal_survival/train.py       # training entrypoint
  models/                            # survival and ROI-token models
  encoders/                          # clinical and radiomics token encoders
  utils/                             # data loading, survival metrics, ROI focus metrics
  data/                              # lightweight token batch containers
  preprocessing/                     # physical compartments, provenance and loading
  explain/gradcam_v2_core.py         # optional epoch Grad-CAM core
  schema.py

environment.yml
tests/                               # CPU regression checks
scripts/
  install_env.sh
  preprocess_roi_inputs.py
  extract_region_radiomics.py         # optional PyRadiomics adapter
  visualize_region_qc.py              # manifest-validated QC overlays
  audit_region_sampling.py            # declared spacing/volume/validity audit
  lib/gpu_utils.sh
  survival/search_roi_constrained_h100.sh
  survival/train_contour_aware_survival.sh
  survival/train_with_roi_focus_watch.sh
  survival/watch_roi_focus_training.sh
```

Legacy search wrappers, DDP/LoRA alternate launchers, OOF evaluation scripts,
and developer build helpers from the earlier bundle are not included.

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
The installer checks dependency imports and the CUDA build after installation.
The monitored training launcher requires `setsid` from Linux `util-linux` so
it can terminate the trainer and its workers together on failure or interruption.

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

CT inputs must already be finite and normalized to `[0,1]` with the same
preprocessing for training and evaluation. Raw HU inputs are rejected by both
the crop preprocessor and the dataset loader; choosing `BODY_CT_THR=auto` does
not normalize an image.

Patient IDs must be unique after directory-name sanitization. The preprocessor
checks all IDs before writing output. Each generated case includes
`roi_crop_provenance.json` with crop settings, source hashes, and output hashes
and geometry. Existing crops are reused only when these checks match. For a
different configuration, changed sources, partial outputs, or older crops
without provenance, choose a new output root or explicitly use `--overwrite`
to regenerate them. Existing files are retained when reuse validation fails.

Verified preprocessing constructs both exterior shells on the reference grid
using the same 10 mm Euclidean definition, excludes both tumor interiors, and
intersects an explicitly supplied valid-tissue mask. Shell overlap is allowed
and recorded. Regions are constructed before cropping, persisted, and used by
both CT pooling and radiomics. Empty shells remain empty. Missing annotations
remain distinct from known absence.

Metadata must include acquisition, contour and intensity provenance, plus the
valid-tissue definition; see the [required fields](docs/anatomical_correspondence.md#input-contract).
Existing crops and old radiomics CSVs are **not** automatically qualified.

From the workspace root:

```bash
cd /storage/TriFuseSurv

PYTHONPATH="$PWD/TriFuseSurv2/src" \
"$PWD/TriFuseSurv2/.conda_env/bin/python" \
  TriFuseSurv2/scripts/preprocess_roi_inputs.py \
  --meta_csv OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv \
  --out_root OPSCC_preprocessed_roi_96x128x128 \
  --output_size 96 128 128 \
  --region_mode verified --margin_mm 10 \
  --workers 4
```

Extract matching radiomics with a reviewed explicit configuration, then launch
one selected fusion mode against the generated metadata:

```bash
FUSION_MODE=F3 REGION_MODE=verified DEPLOYMENT_MODE=expert \
COMPARISON_RECIPE=principal ID_COL=Patient_ID \
ENDPOINT=OS DEBUG_FOLD=3 \
META_CSV=OPSCC_preprocessed_roi_96x128x128/cohort_preprocessed_stage2_roi.csv \
RADIOMICS_SOURCE=regions_radiomics.csv \
IMG_SIZE="96 128 128" \
OUT_DIR=runs/fusion_f3_os_fold03 \
bash TriFuseSurv2/scripts/survival/train_contour_aware_survival.sh
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
strictest debugging run. The normalized input requirement applies to every
search profile.

## ROI-Constrained 4-H100 Round-3 Search

This is a historical tuning workflow, not the controlled architecture study.
It now requires verified inputs by default. F0–F4 comparisons must use the same
patients, splits, seeds, backbone initialization, fitting rules and tuning
opportunities; save each mode in its own output directory.

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
- `PT_SHELL_THICKNESS_MM=10.0`

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
SEARCH_ALLOW_PARTIAL_RESUME=1 # must be prespecified in the original experiment contract
```

Strict recovery checks all model keys and required optimizer, scheduler,
scaler, EMA and SWA state. EMA may be absent during localization-only warmup;
SWA may have no averages before its first update. Other missing state requires
a prespecified partial-resume policy or a fresh run. Partial-resume permission
cannot be changed within an existing experiment. A resume directory containing
only `best.pt` requires restoration of the original `last.pt` continuation
state or an explicit fresh run.

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

The default launcher starts fresh (`RESUME=0`). In verified mode, persisted
compartments determine survival pooling during train/eval/export independently
of the localizer or teacher-forcing settings. Localization remains an auxiliary
task, with missing/invalid annotations excluded from supervision.

Each fold records parameter counts, an input/model contract and a canonical
experiment specification. Resume rejects changed attention-head counts,
objectives, dropout, checkpoint-selection metrics, horizons, optimizer/scheduler
settings and data identities, even when parameter shapes are compatible.
Only explicitly listed reporting/loader settings may change. Older contracts
cannot establish the expanded continuation requirements.

The Python entrypoint accepts `--time_col` and `--event_col` overrides for the
selected primary endpoint. These columns are used consistently for filtering,
multitask training labels, time bins and evaluation; auxiliary endpoints keep
their standard columns. Radiomics CSV identifiers are read as strings to
preserve leading zeros.

`--report_metric nll` and `--report_metric ibs` select lower values for both best
checkpoints and final exports. AUC, C-index and composite scores select higher
values. Logs and checkpoint metadata retain the original metric values.

## ROI Metrics

- `pt_mass`, `ln_mass`, `pt_peri_mass`: fraction of effective ROI probability mass
  inside the corresponding GT region.
- `pt_rec`, `ln_rec`, `pt_peri_rec`: fraction of GT support covered by the support
  map.
- `pt_dice`, `ln_dice`, `pt_peri_dice`: support Dice against GT.

`pt_peri_*` refers to the 10 mm primary-tumor peritumoral shell.

In verified mode these support metrics check the supplied regions and may be
perfect by construction. They do not measure automated segmentation accuracy.
Separate `pt_localizer_*` and `ln_localizer_*` metrics score predicted contours
against available annotation; missing/invalid annotations are excluded.

ROI masks use support-preserving reduction on coarser feature grids. Masked
and weighted attention return zero for empty support instead of pooling the
whole feature grid. The revised fusion and geometry inputs require new training
runs; older checkpoints/results do not qualify the verified model. Coarse feature
cells and shared encoder receptive fields can contain surrounding context even
when pooling uses the declared region.

## Regression checks

From this repository, with runtime dependencies available:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m unittest discover -s tests -v
```

These checks use synthetic inputs and temporary mocked launchers, including a
small real CPU model forward/backward pass. They cover preprocessing reuse,
patient identity, physical shells, image/radiomics identity, missing annotations,
F0–F4 routing, ROI pooling, checkpoint recovery, survival column overrides,
selection direction, and process cleanup. The extractor adapter is tested with
a synthetic implementation; this does not qualify a PyRadiomics installation,
full-cohort performance, or CUDA execution.
