# TriFuseSurv2 Runtime Package

OPSCC survival training with shared physical CT/radiomics compartments and
controlled F0–F4 fusion alternatives. The default is **F3, verified regions,
expert contours**. These are implementation capabilities, not evidence of
improved prognosis or clinical validity.

Start with the [implementation and validation guide](docs/implementation_and_validation.md)
for design decisions, lessons from revisions, evidence levels, and remaining work.
The [anatomical correspondence workflow](docs/anatomical_correspondence.md)
provides input requirements, migration, and comparison settings.
The principal comparison recipe disables both image-token dropout paths.
It also requires every patient to use the same explicit common-grid policy.
The preprocessor maps the full available source onto that physical lattice,
then crops/pads by integer offsets; it never resizes an overflowing canvas.
`Patient_ID` is the default patient key, with an unambiguous `patient_id` alias
accepted. Acquisition and provenance identifiers retain their original strings.

The [development sampling proposal](docs/common_grid_qualification.md) records
the local input audit, candidate grid and extraction configuration. The local
613-case inventory contains previously prepared inputs with missing original
CT/RTSTRUCT paths and no qualified HU provenance. It is not a ready verified
cohort. On October 27, 2026, executable source `614d7fb` passed synthetic
common-grid checks, actual PyRadiomics invariance checks, tiny F2/F3 CUDA
forward/backward checks, and six representative-case larger-canvas comparisons
inside a worker with one verified H100. Sampling and PyRadiomics ran on CPU;
the tiny model checks ran on CUDA. Overlay review accepted only development geometry, retaining
source-FOV truncation and small-region sampling limitations.

The later [paired pilot recovery](docs/pilot_runtime_recovery.md) records completed
613-case preparation, the supervisor/DataLoader fixes, and the corrected run
queued for one H100 as of October 27, 2026, 11:38 PM CDT. The later
[pilot completion report](docs/pilot_completion.md) records both modes finishing
two epochs on October 28, 2026, 12:50 AM CDT, followed by verified cleanup.
Its 18 required checkpoint acceptance criteria passed, and the separate
metadata reconciliation completed on October 28, 2026, 8:54 AM CDT.

The next [OS study protocol](docs/study_protocol.md) and
[specification](configs/study_spec.json) preserve the architecture and split.
The user confirmed OS from diagnosis in days, event 1 as death from any cause,
and fixed delivered censoring durations. Clinical encoding is approved;
prospective neural models and baselines omit NSTAGE because the user confirmed
it is derived from retained T/N/M.
Computational cohort review is complete for development use. The principal
specification remains draft until the common neural training budget is justified;
C0/C1 has a separate frozen recipe, with execution deferred behind training-
budget determination and final study gates. New
[pilot/cohort auditors](docs/acceptance_reports.md),
[paired evaluator and survival exports](docs/paired_evaluation.md), and
[C0/C1 baselines](docs/tabular_baselines.md) provide the prospective checks.
Their [separate frozen protocol](configs/baseline_study_spec.json) is prepared
for execution after the development-budget experiment and final study freeze.
Synthetic software tests and metadata accounting are separate from completed
H100 training and scientific results.
The [development-budget protocol](docs/budget_development.md) starts with a
separate 12-epoch pair on training/validation data only. The new
`--development-only` trainer path preserves reserved test IDs while excluding
their records, datasets, evaluations and exports. The primary scientific metric
is mean paired deltaIBS over 210–730 days, with the user-defined worthwhile
reduction of 0.005. Development training source `30dfdec` passed the full
synthetic suite of 260 tests and 176 subtests. The later supporting-horizon
evaluator update passed 30 focused evaluator/baseline tests and 16 subtests.
Its [dated launch record](docs/budget_development_run.json) shows the separate
source `30dfdec` Job submitted on October 28, 2026, at 11:10 AM CDT. At the
11:11 AM CDT snapshot it was Pending cluster resources with zero Running/Ready
H100s and a matched detached supervisor. The 53-test host controller gate passed;
the new source's same-image CUDA gate and both fits await worker startup.
This snapshot does not establish a common training cap or a scientific result.
The [fresh October 28 report](docs/explicit_report_20261028.md) supersedes that
Pending state: the worker verified one H100 and passed 37 source tests, then
failed its tiny CUDA gradient qualification before either real-data fit began.
Exact cleanup completed; the 11:23 AM CDT snapshot found no retained Job or Pod.
The subsequent [clinical typing and numerical qualification work](docs/numerical_qualification.md)
adds an explicit prospective categorical policy and a matched diagnostic of the
exact retained CUDA fixture. The new study configuration requires
`--clinical_type_policy approved_categorical_v1`; legacy encoding remains
available for historical contracts. Numerical correction and a fresh budget
attempt require separately bound H100 evidence.
The [dated follow-up handoff](docs/clinical_cuda_followup_run.json) records
source `21d5e03` submitted for the six-cell diagnostic. At October 28, 2026,
12:35 PM CDT it was Running/Ready with one GPU request and a matched supervisor;
the separate 12:36 PM CDT startup receipt verified one H100 80GB. Matrix results
and the intended CT/Swin qualification remain pending at that snapshot.
The [completed diagnostic report](docs/cuda_diagnostic_20261028.md) supersedes
that snapshot: all six cells completed, cleanup was verified, and the unchanged
production AMP policy was selected. Full CT/Swin survival-NLL qualification
remains required before the fresh 12-epoch budget pair.
The [explanatory study plan](docs/explanatory_study_plan.md) records the later
fusion ladder and distinguishes it from routing and contour designs that still
need a complete specification before held-out review.

The [whole-cohort report](docs/cohort_qualification.md) reconciles all 613
prepared patients and records computational dispositions for all 525 queued
cases, while preserving their original pending anatomical-review fields. The unchanged pilot
completed both modes and passed the separate acceptance audit. Its original
controller receipt mismatch was resolved by separately pinned validation;
[optimizer identity reconstruction](docs/checkpoint_optimizer_mapping.md)
provides the separately pinned postprocessing path.

The [dated evidence summary](docs/development_qualification_summary.json) binds
those observations to source and receipt hashes. At its latest snapshot,
**October 27, 2026, 9:03 PM CDT**, the separate single-H100 cohort worker was
Running/Ready in input preparation; F2/F3 pilot training had not started.
This is a historical snapshot, not a live status display. No completed cohort
comparison or clinical result is claimed. Documentation-only updates do not
change the frozen source used by that run.

## Contents

```text
src/trifusesurv2/
  multimodal_survival/train.py       # training entrypoint
  models/                            # survival and ROI-token models
  encoders/                          # clinical and radiomics token encoders
  utils/                             # data loading, survival metrics, ROI focus metrics
  data/                              # lightweight token batch containers
  preprocessing/                     # physical compartments, common lattice, provenance and loading
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
This generic installation recipe is separate from the recorded October 27 H100
environment, which used a qualified container overlay with PyTorch 2.9.1+cu128;
see the [validation guide](docs/implementation_and_validation.md).
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

The survival model can train on ROI-focused canvases containing the complete
declared compartments. Common-grid preprocessing is the verified CLI default
and requires an explicit JSON configuration for physical spacing, direction
and lattice origin. It has no automatic resolution choice. Image context is
retained inside the available source FOV; exterior padding contributes no
compartment support.

Learned-image CT inputs must already be finite and normalized to `[0,1]` with
the same preprocessing for training and evaluation. Raw HU cannot replace that
input; a separately bound HU image from the same acquisition may serve the
radiomics branch. Choosing `BODY_CT_THR=auto` does not normalize an image.

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
remain distinct from known absence. Common-grid images and masks are compressed
`.mha` files, preserving the double-precision geometry needed for arbitrary
lattice phase. Geometry is checked after writing and before publication. The
diagnostic legacy resize path retains `.nii.gz` output.

Metadata must include acquisition, contour and intensity provenance, plus the
valid-tissue definition; see the [required fields](docs/anatomical_correspondence.md#input-contract).
Existing crops and old radiomics CSVs are **not** automatically qualified.

Example after reference metadata and a sampling policy have been qualified:

```bash
PYTHONPATH=src python scripts/preprocess_roi_inputs.py \
  --meta_csv /path/to/qualified_reference_metadata.csv \
  --id_col Patient_ID \
  --sampling_policy common_grid \
  --grid_config_json /path/to/qualified_grid.json \
  --out_root /path/to/new_verified_inputs \
  --output_size 96 160 96 \
  --region_mode verified --margin_mm 10 \
  --workers 4
```

Canvas size is D/H/W; physical spacing is x/y/z. The proposed development
configuration is [1 × 1 × 3 mm](configs/common_grid_candidate.json), which gives
this canvas a 96 × 160 × 288 mm physical FOV in x/y/z. That candidate is not a
frozen clinical policy. A larger spatial extent requires a larger canvas or
bucket with the same lattice. Overflow fails explicitly. The former
patient-dependent resize remains available only as the explicit
`--sampling_policy legacy_resize` diagnostic and cannot enter a principal
F0–F4 comparison.

Extract matching radiomics with a reviewed explicit configuration, then launch
one selected fusion mode against the generated metadata:

```bash
FUSION_MODE=F3 REGION_MODE=verified DEPLOYMENT_MODE=expert \
COMPARISON_RECIPE=principal ID_COL=Patient_ID \
ENDPOINT=OS DEBUG_FOLD=3 \
META_CSV=/path/to/new_verified_inputs/verified_metadata.csv \
RADIOMICS_SOURCE=/path/to/qualified_regions_radiomics.csv \
IMG_SIZE="96 160 96" \
OUT_DIR=runs/fusion_f3_os_fold03 \
bash scripts/survival/train_contour_aware_survival.sh
```

Use the actual generated metadata filename. Neither local preprocessed CSV
already satisfies these provenance and qualification prerequisites. The first
controlled contrast remains F2 versus F3; matched splits, initialization,
measurements and training settings are required.

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

The search contract fingerprints the checkpoint actually selected for each
relevant fold and for backbone configuration alignment, using the trainer's
explicit-file, `all/`, and fold-directory resolution rules. Resolved paths,
existence and SHA256 are checked before skipping completed trials or changing
receipts. Replacing checkpoint bytes at the same path, making a previously
missing initialization available, or changing directory selection rejects reuse.

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
common-grid invariance under distant nodes, integer crop/padding and canvas
changes, F0–F4 routing, ROI pooling, checkpoint recovery, survival column overrides,
selection direction, and process cleanup. The extractor adapter is tested with
a synthetic implementation; this does not qualify a PyRadiomics installation,
full-cohort performance, or CUDA execution.
