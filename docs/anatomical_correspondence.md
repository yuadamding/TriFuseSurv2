# Anatomical correspondence workflow

The implemented question is whether matching learned CT and handcrafted
radiomics measurements within the same anatomical compartment helps survival
prediction. F2 versus F3 isolates that routing constraint. No performance or
novelty claim follows from the software tests.

This revision implements the immediate target: common physical compartments,
explicit missingness without semantic fallback, and F0–F4 in one trainer.
Contour-consistency training, individual-node phenotypes, clinical-only baseline
models, F5, learning curves, paired evaluation and external validation remain
separate study stages. No cohort has been regenerated or study fits run by this
revision.

The subsequent targeted corrections lock semantic settings on resume, preserve
literal acquisition IDs, disable image-token dropout for principal comparisons,
and cache validated region payloads. Physical sampling qualification is still a
separate requirement; exact shared masks do not establish crop-independent
radiomics measurements.

## Physical definition

The reference grid is a SimpleITK image with positive spacing, finite origin,
orthonormal direction and nonempty size. The caller must qualify the metadata
as millimeters and LPS coordinates. Rotations and reflections are supported;
sheared/nonorthogonal grids are rejected. For a lesion support M, the shell is
the reference voxel centers with Euclidean distance `0 < d(x, M) <= 10 mm`,
intersected with the declared valid-tissue mask and excluding **both** PT and
LN interiors. Empty M gives an empty shell. PT/LN shells may overlap; the
manifest records overlap volume.

The implementation uses axis spacing in SciPy's exact Euclidean distance
transform. Array order is D/H/W (z/y/x); SimpleITK metadata is x/y/z. Distances
are discrete voxel-center measurements, not distances to reconstructed
subvoxel contour surfaces. See [SciPy EDT](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html)
and [SimpleITK physical geometry](https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html).

Regions are generated **before** the crop. A physical margin of at least 10 mm
is required. The preprocessor verifies that all reference support is retained
and records this fraction separately from source-field-of-view truncation.
It resamples intensities linearly and masks with nearest neighbor onto one
model grid, recording new support identities, volumes and volume ratios.
Resampling does not imply exact volume preservation; a small region lost during
resampling becomes unavailable for image features. Field-of-view truncation
flags require cohort QC and do not constitute a complete biological shell.

CT pooling and radiomics use these same **persisted model-grid masks**. The
reference region remains their recorded parent. This is exact agreement of
declared supports; pooling at coarser feature grids uses support-preserving
mask reduction, and shared CNN/transformer receptive fields retain context.

## Input contract

The preprocessing metadata CSV uses these fields (paths may be relative to
the CSV). Identifiers must be unique, nonempty strings; keep leading zeros.
The user-selected patient key is `Patient_ID` (`--id_col Patient_ID`, launcher
`ID_COL=Patient_ID`). An unambiguous case/punctuation spelling such as
`patient_id` resolves to the original column; multiple aliases are rejected.
The manifest's canonical field remains `patient_id`. Patient and acquisition
identifiers are separate: a patient ID never silently supplies `scan_id`.
Shared CSV readers preserve numeric-looking patient/case/scan/radiomics-scan
IDs and provenance IDs, including leading zeros. Blank identifier cells are
explicitly missing; literal strings such as `NA` remain literal identifiers.

| Field | Meaning |
|---|---|
| `patient_id`, `scan_id` | Patient and selected acquisition identity |
| `ct_out_path` | Finite CT normalized to [0,1] for the learned image branch |
| `ct_intensity_transform` | Exact source-to-model intensity recipe |
| `mask_primary_out_path`, `mask_nodal_out_path` | Binary reference-grid contours |
| `pt_annotation_available`, `ln_annotation_available` | Optional boolean; false explicitly means missing annotation, not negative disease |
| `contour_source`, `contour_version` | Expert/predicted provenance and version/reader identity |
| `valid_tissue_out_path`, `valid_tissue_definition` | Qualified reference-grid tissue support and its definition |
| `radiomics_ct_path`, `radiomics_scan_id`, `radiomics_intensity_transform` | Optional separate intensity image from the **same acquisition and geometry**, e.g. HU |
| `midline_lps_x_mm` | Optional qualified LPS midline; required for laterality |
| `segmentation_provenance` | JSON provenance required for automated mode, described below |

A supplied, empty LN contour is known absence. An unavailable contour must be
marked unavailable explicitly; a missing file is an error. Because a shell
must exclude both interiors, missing either annotation can invalidate an
otherwise definable shell. States `present`, `absent`, `missing`, `invalid`
and feature availability are distinct. Failed radiomics are unavailable, never
replaced by interior measurements or silently labeled absent anatomy.

The default valid-tissue policy requires a mask. `--valid_tissue_policy full_fov`
is an explicit engineering opt-in that includes all imaged voxels, including
air/table if present; it is recorded as such and is not tissue segmentation.
Do not use it to claim tissue-restricted measurements.

If a separate radiomics image is omitted, extraction uses the normalized CT
intensities with their recorded transform. That is not HU radiomics. Choose
and lock the intensity/discretization recipe before fitting models.

The [cohort dictionary template](cohort_dictionary_template.csv) lists additional
study-audit fields; it contains no inferred cohort facts. OS is the software's
default development endpoint. Confirm endpoint ascertainment, time origin and
follow-up before freezing OS/DSS/DFS roles. The survival objective is unchanged.

## Prepare shared inputs

From the repository, with dependencies available:

```bash
PYTHONPATH=src python scripts/preprocess_roi_inputs.py \
  --meta_csv /path/to/qualified_reference_metadata.csv \
  --id_col Patient_ID \
  --out_root /path/to/new_verified_crops \
  --output_size 96 128 128 --margin_mm 10 --region_mode verified
```

Each case contains normalized CT, a radiomics intensity image, four compartment
masks, supervision-mask copies, `region_manifest.json`, and a crop provenance
receipt. The output CSV adds `region_manifest_path`. Regeneration checks source
and output hashes; incompatible or partial outputs require a new root or an
explicit overwrite. Existing source data is not modified.

Each region identifies patient, acquisition, contour source/version, anatomy,
reference geometry and support checksum. The model-grid representation adds
its own geometry, checksum and region ID. Loading checks those identities
against the actual CT and masks, rejects truncated crops and enforces exclusion
rules. Provenance records establish traceability; they do not independently
prove the truth of supplied clinical/segmentation declarations.

Render a standalone QC image for one manifest with Matplotlib (included in the
conda environment, or available through the `qc` extra):

```bash
PYTHONPATH=src python scripts/visualize_region_qc.py \
  /path/to/new_verified_crops/001/region_manifest.json --output region_qc.png
```

The plot validates bound CT/mask files and shows both interiors, both shells,
overlap, explicit states, source-field-of-view flags and crop/resampling volume
accounting. Its axes describe grid directions, not patient laterality. The
[synthetic QC example](figures/synthetic_region_qc.png) illustrates full retention
of available reference support alongside a source-field-of-view truncation
flag; it is an engineering phantom, not clinical qualification.

## Qualify and freeze physical sampling

The current cropper resizes each patient-specific PT/LN union crop to a fixed
tensor shape. Its model spacing therefore depends on crop extent. Adding a
distant node can change the sampling, measured volume and intensity statistics
of an otherwise unchanged primary region. This mechanism is covered by a
synthetic regression; it is not an estimate of the effect in this cohort.
The existing cropper remains available, but its variable sampling is not a
qualified principal-study policy.

Before complete-cohort extraction, inspect declared spacing distributions,
model/reference volume ratios, small-support disappearance and crop/FOV flags:

```bash
PYTHONPATH=src python scripts/audit_region_sampling.py \
  --metadata-csv /path/to/verified_metadata.csv --id_col Patient_ID \
  --institution-col SITE --protocol-col PROTOCOL \
  --nodal-burden-col BURDEN --out-dir /path/to/new_sampling_audit
```

The optional institution/protocol/burden flags refer to actual supplied
metadata columns. The audit writes patient and region CSVs, failures and a
JSON summary with distributions, strata and descriptive burden associations.
It reads manifests only: it does not verify image bytes, inspect outcomes,
extract features, select a resolution, or grant a qualification pass. Use the
QC renderer and the chosen PyRadiomics environment for the image/feature checks.

Select representative cases with no nodes, separated nodal lesions, small
regions, source-boundary lesions, varied acquisition spacing/protocols, and
empty/invalid shells. Compare representative features under crop changes while
holding source CT and the compartment fixed. Include source-grid history:
resampling already performed before this cropper also affects measurement.

The recommended next policy is a common physical lattice: choose spacing and
orientation from acquisition quality and compute constraints, bind the lattice
origin/phase independently of tumor extent, then crop by integer grid offsets.
Padding or size buckets should accommodate extent without resizing. Fixed
spacing alone is insufficient if crop-dependent origin changes subvoxel phase.
Validate invariance under a distant node and integer-offset padding/crop changes,
plus small-region and rotated/anisotropic cases. This common-grid implementation,
the numeric spacing and actual extraction configuration remain to be qualified
and frozen before the expensive F2/F3 study. No setting is selected from outcomes.

## Extract and bind radiomics

`scripts/extract_region_radiomics.py` is an optional PyRadiomics adapter. Install
PyRadiomics separately in a qualified extraction environment (the package
declares a `radiomics` extra), and supply a nonempty explicit JSON configuration.
For example, this tiny configuration is useful for engineering checks only:

```json
{"imageType":{"Original":{}},"featureClass":{"firstorder":["Mean"]},"setting":{"binWidth":25}}
```

```bash
PYTHONPATH=src python scripts/extract_region_radiomics.py \
  --meta_csv /path/to/new_verified_crops/cohort_preprocessed_stage2_roi.csv \
  --id_col Patient_ID \
  --config_json /path/to/reviewed_radiomics_config.json \
  --out_csv /path/to/regions_radiomics.csv
```

The adapter disallows extraction-time support changes such as resampling,
resegmentation or mask correction. It records extraction errors as invalid
features and writes a separate extraction receipt. The reference manifest is
preserved. Extracted rows bind acquisition, intensity checksum/transform,
configuration/hash, software/version and each exact image-region ID and mask checksum.

External CSVs must provide the same contract. Habitat column names alone are
insufficient. Valid feature rows must contain finite values and agree with
feature-presence flags. All compared patients use one extraction configuration.
Feature-column selection, standardization and PCA use training patients only;
validation/test rows are transformed with those fitted objects.

The extraction adapter has synthetic tests, not a completed numerical
PyRadiomics qualification. Review extraction settings, feature failures and
representative overlays before cohort use. Do not add provenance columns to old
features unless their actual extraction inputs and recipe have been verified.

## Geometry and deployment

The geometry token retains 19 explicitly named crop-dependent descriptors in
`schema.CROP_GEOMETRY_FEATURES`: occupancy fractions, normalized centroids,
displacements, centroid distance, burden ratio and presence. They are separate
from `PHYSICAL_GEOMETRY_FEATURES`: PT/nodal volume, PT–LN centroid and boundary
voxel-center distances, nodal voxel RMS spread, qualified left/right/bilateral
flags, plus unavailable node-count/largest-node placeholders. No connected
components are interpreted as anatomical nodes. Physical values use a fixed
signed `log1p(abs(value))` transform, validity bits, and region-state encodings.

Expert mode uses supplied expert contours for all dependent inputs, including
crop, radiomics and geometry. Persisted supports determine pooling regardless
of teacher-forcing alpha or the internal localizer's prediction. Missing
annotations supply no localization loss; known absence remains a valid
negative target. `pt`/`ln`/`pt_peri` ROI metrics check applied supports and can
be perfect by construction. Separate `pt_localizer`/`ln_localizer` metrics score
the learned localizers against available annotation.

Automated mode requires predicted/automated contour sources for every region
and `segmentation_provenance` with `method` (`out_of_fold` or `independent`),
`segmenter_id`, and explicit `training_patient_ids` excluding the evaluated
patient. All dependent inputs must be rebuilt from those contours. This check
does not train/qualify a segmenter or establish independence across duplicate
patients. Manual-derived inputs cannot pass as automated merely by disabling
teacher forcing. Contour-variation provenance is supported by preprocessing;
synchronized augmentation/consistency training is not implemented here.

Spatial flips are disabled for verified datasets because stored radiomics and
laterality cannot be silently transformed with the image. Intensity augmentation
can still be applied to the image branch; keep that recipe identical across
comparisons.

With `--cache_volumes --volume_cache_size N`, verified datasets retain at most
N complete patient payloads per worker: CT, supervision masks, shared regions,
states and geometry. Warm hits avoid image decoding and full CT hashing while
checking every bound file's resolved path, device, inode, size and nanosecond
mtime/ctime. Any change invalidates the entry and triggers full validation;
changed manifest content is rejected against the dataset's original contract.
Outputs are copies, and forked workers start their own caches. The cache uses
filesystem metadata to detect change; inputs must remain immutable during use.
It is not a cryptographic filesystem snapshot. No training-speed factor is
claimed from the I/O call-count tests.

## Controlled fusion alternatives

| Mode | Fusion | Clinical routing |
|---|---|---|
| F0 | Concatenate the common projected measurements and availability into an MLP | All groups |
| F1 | Flat transformer over identified image, radiomics and clinical tokens | All groups |
| F2 | Local compartment fusion, then global integration; every regional query can access all regional radiomics | All groups |
| F3 | Same hierarchy, matching region-to-radiomics connections only | All groups |
| F4 | F3 with the prespecified habitat-specific clinical routing | Restricted locally |

F2/F3/F4 have identical parameter names/shapes and fixed local context tensor
sizes. They differ only in allowed attention edges. Global and geometry image
queries have no local regional-radiomics edges in either F2 or F3. F4 limits
local clinical routing, not patient-level access to clinical information.
All modes share anatomical/source identifiers, region-state and availability
conditioning. Node/topology inputs are disabled and rejected for F0–F4.

`COMPARISON_RECIPE=principal` (CLI `--comparison_recipe principal`) is the default.
It requires `TOKEN_DROPOUT=0` and `V2_IMAGE_HABITAT_DROPOUT_P=0`, including
historical search-profile overrides. A dropped image token otherwise removes
its local radiomics route in F3/F4 while F2 can retain other routes, confounding
the principal interpretation. Nonzero rates require the explicitly separate
`robustness` recipe. The launcher supplies its historical nonzero defaults for
that recipe; direct Python invocations set desired rates explicitly. No change
to the fusion architecture or claim of improved accuracy follows from this
regularization choice.

Use the existing launcher with `FUSION_MODE=F0` through `F4`; keep `META_CSV`,
`RADIOMICS_SOURCE`, `SPLITS_DIR`, `SEED`, backbone weights, optimizer/objective,
dropout, batch size and training budget fixed. Give each mode/fold/seed a
different `OUT_DIR`. The trainer records counts and an input/model contract
per fold and in checkpoints, including the observed Git revision, dirty-checkout
flag and Python/script source digest. An environment-provided revision is
recorded separately as a declaration. Only the observed commit and executable
source digest bind continuation: a new log/output file can change the dirty
diagnostic without changing executable source. Resuming under changed source, fusion mode, split,
metadata, manifest or radiomics file is rejected. The canonical experiment
specification records every resolved CLI setting except the explicit operational
allowlist, so newly introduced semantic options are locked by default. This
includes backbone/fusion attention-head counts, dropout, augmentation-related
settings, loss weights, selection/export metrics, evaluation horizons, optimizer
settings, epoch budget and scheduler continuation. Runtime records include
world size, precision-related arguments, PyTorch version and warm-start hash.

Only `resume`, `workers`, `eval_workers`, `prefetch_factor`, `cache_volumes`,
`volume_cache_size`, `log_every_batches`, `roi_focus_every_batches`, and
`print_trainable_backbone_params` may change without a new experiment. This
permits operational changes; it does not promise bitwise RNG replay across
worker changes. Checkpoint-embedded contracts are checked even if their JSON
sidecar is absent. Older incomplete contracts require a new experiment rather
than reconstruction of unspecified settings. Changing the epoch budget is an
experiment change, not an implicit scheduler extension.
`best.pt` alone is not continuation state: resume refuses that case without
`last.pt`. A partial-resume policy must also have been chosen in the original
experiment; it cannot be enabled later to reinterpret an existing run.

Search output roots also bind their recipe before skipping completed trials or
rewriting search manifests. Use separate `OUT_ROOT` values across recipes/modes;
historical roots lacking the recipe contract cannot be silently reused.
F0/F1 capacities differ;
report counts and use comparable tuning opportunities instead of claiming
identical capacity for all modes.

Historical behavior requires explicit `--fusion_mode legacy --region_mode legacy`
(launcher: `FUSION_MODE=legacy REGION_MODE=legacy`). Legacy inputs have no
verified correspondence claim. Semantic shell substitution is removed in
every mode; explicit physical-shell mode requires spacing and never falls back
silently to voxels. New verified geometry/fusion checkpoints are not compatible
with historical checkpoints. Start new experiments.

## Evidence still required

Qualify the cohort inventory, endpoint dictionary, masks, field-of-view flags,
resampling volumes, extraction failures and visual QC on representative patients.
Freeze direct comparisons on the same patients with the same splits/seeds.
Previously searched folds are development data. Fit all preprocessing within
training data and keep segmenter/backbone training independent of evaluation.

Then screen and confirm F2 versus F3, F1 and a strong simple comparator, reporting
parameter counts and paired uncertainty on supported horizons. A separate
clinical/burden baseline ladder is still needed. Choose discrimination/calibration
metrics and censoring conventions before final results. No new evaluation
method or confirmatory study is implemented by this change.

Only after establishing the core comparison should contour sensitivity guide
the next extension: synchronized alternate contours with recomputed radiomics,
geometry and validity, followed by consistency if justified. Individual-node
phenotypes remain conditional on reliable instance annotations. Synthetic CPU
tests establish software properties, not clinical robustness, survival benefit,
external validation or GPU qualification.
