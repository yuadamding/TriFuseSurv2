# Implementation and validation knowledge

TriFuseSurv2 asks whether anatomically matching learned CT features and
handcrafted radiomics improves survival prediction. The central controlled
contrast is **F3 versus F2**: the same hierarchical model with matched versus
unrestricted regional radiomics connections. Implementation checks and the
development qualification described here do not establish a prognostic benefit.

This guide records the executable baseline
`614d7fbdb3551f58c2e8b5da403f7b6f5418f23e` and evidence available on
**October 27, 2026, through 9:03 PM CDT**. Later documentation changes do not
change the source bound to an existing run. This is a dated knowledge record,
not a live scheduler status page.

Use the [anatomical correspondence workflow](anatomical_correspondence.md) for
the detailed input schema and commands, and the
[common-grid qualification record](common_grid_qualification.md) for the
candidate settings and their acquisition limitations. Operational receipts,
patient-level outputs and controller state remain in the separately maintained
run workspace; they are not repository documentation.
The [portable evidence ledger](development_qualification_summary.json) records
aggregate results, source/configuration identities, dated status and hashes of
the underlying receipts without bundling patient-level artifacts.

## Architecture and the controlled question

The [contour-aware wrapper](../src/trifusesurv2/models/contour_habitat_survival.py)
connects the [shared SwinUNETR backbone](../src/trifusesurv2/models/swinunetr_shared_roi_token_backbone.py)
to the [fusion model](../src/trifusesurv2/models/habitat_survival.py).
The [shared schema](../src/trifusesurv2/schema.py) defines six image tokens:
`global`, `pt_intra`, `pt_peri`, `ln_intra`, `ln_peri`, and `shape_spatial`.
Radiomics provides four corresponding regional tokens, without global or
geometry radiomics tokens. Default clinical groups are biology, burden and host;
treatment is an explicit alternative configuration.

| Mode | Fusion | Local clinical access |
|---|---|---|
| F0 | Concatenate all projected measurements and availability/state information into an MLP | All clinical groups enter the concatenation |
| F1 | Flat transformer over image, radiomics and clinical tokens, with explicit source/anatomical identities | All clinical groups enter the transformer |
| F2 | Regional cross-attention followed by a global transformer; each regional image query can access all four regional radiomics tokens | All groups for every image query |
| F3 | Same hierarchy as F2; each regional query can access only its matching radiomics token | All groups for every image query |
| F4 | F3 with the schema's prespecified habitat-specific clinical connections | Restricted locally; the global image query still has every group |

F2/F3/F4 instantiate identical parameter names, shapes and local context sizes.
Their permitted attention edges differ. In every hierarchical mode, global and
geometry image queries have no regional-radiomics edges. F4 therefore changes
local routing without removing patient-level access to a clinical group.
Equal capacity is not asserted for F0/F1; report actual parameter counts for
each fitted model. Node/topology inputs are rejected for F0–F4 and exist only in
the explicit legacy route.

All modes receive region states and measurement availability separately.
An unavailable image token suppresses its local fused token, even if a
radiomics measurement is available. This is why the principal recipe requires
both `TOKEN_DROPOUT=0` and `V2_IMAGE_HABITAT_DROPOUT_P=0`: otherwise a dropped
query can remove a matched radiomics route in F3 while F2 still accesses that
measurement through another region. Nonzero token dropout belongs in the
separate robustness recipe. See the
[routing tests](../tests/test_fusion_modes.py) and
[recipe tests](../tests/test_comparison_recipe.py).

The survival heads use a shared prognostic trunk plus endpoint-specific
residual logits for OS, DSS and DFS over discrete time bins. This is multitask
survival prediction; it is not a competing-risk or multistate event model.

## Input identities and anatomical meaning

`Patient_ID` is the patient key. An unambiguous spelling alias such as
`patient_id` is accepted without renaming the source column. Patient identity
does not establish acquisition identity: `scan_id` must identify the selected
acquisition, and a separate radiomics image must identify that same acquisition.
The development adapter uses the supplied `ct_series_uid` for this purpose.
Numeric-looking strings, leading zeros, `NA` and `NULL` remain literal IDs;
only empty/whitespace-only cells are missing. Ambiguous aliases fail. See the
[identifier reader](../src/trifusesurv2/preprocessing/identifiers.py).

Verified inputs bind the actual images and masks to patient/acquisition IDs,
contour source/version, declared physical geometry, intensity transform,
valid-tissue definition and extraction configuration. Region IDs include the
reference support identity; model-grid masks have their own identities and
checksums. Feature column names alone cannot establish anatomical correspondence.
Old radiomics cannot be promoted by adding guessed provenance columns.

| State or flag | Meaning and consequence |
|---|---|
| `present` | Nonempty declared reference support; feature availability still needs a valid model-grid measurement |
| `absent` | Explicitly available annotation establishes empty support; absence of a shell is not absence of its lesion |
| `missing` | Annotation unavailable; do not reinterpret it as negative disease |
| `invalid` | Required definition or measurement cannot be accepted, such as a shell with unknown exclusion anatomy |
| Image/radiomics availability | Whether that modality has a usable measurement; neither zero padding nor an extraction failure proves anatomical absence |

The [compartment implementation](../src/trifusesurv2/preprocessing/compartments.py)
constructs PT and LN exterior shells on the reference grid before any crop.
Both use voxel-center Euclidean distance `0 < d <= 10 mm`, intersect the
declared valid-tissue mask, and exclude both tumor interiors. PT/LN shell overlap
is allowed and measured. Missing exclusion anatomy can invalidate a shell;
empty shells stay empty and are never replaced with tumor interior.
An available empty contour is a negative annotation under the generic contract;
the legacy development adapter deliberately treats an `empty` QC string as
unavailable annotation because that string does not qualify a negative contour.

The learned-image branch requires finite CT values already normalized to
`[0,1]`. A body threshold does not normalize CT. Radiomics may use either the
same stored normalized image or a separately bound same-acquisition intensity
image. Only the latter can support HU-based features if its HU provenance is
qualified. The current development inputs have no qualified HU provenance.

## Physical sampling and geometry lessons

The [sampling implementation](../src/trifusesurv2/preprocessing/sampling.py)
uses an explicit common lattice: spacing, direction and origin/phase, with mm
units and LPS coordinates. Fixed spacing alone is insufficient. The full
source-FOV grid depends on source geometry and the policy, independently of
lesion extent. Each intensity image is interpolated once with linear float32
sampling; each reference compartment is mapped once with nearest neighbor.
Only integer crop/pad operations follow. Overflow fails and requires a larger
canvas or bucket at the same lattice; it never triggers a hidden resize.

CT pooling and radiomics use the same persisted model-grid masks. A distant
node outside an unchanged primary shell must not alter that primary region's
sampled CT/support/features merely by moving the crop envelope. This is a
regional measurement requirement. It is not a requirement that a patient's
survival prediction remain unchanged when nodal disease changes.

Several different QC facts must remain separate:

- **Crop retention:** whether the declared source support centers remain
  inside the output canvas.
- **Resampling loss:** a small support may disappear on the coarser lattice
  despite 100% source-center retention; its image measurement is unavailable.
- **Volume change:** nearest-neighbor resampling does not conserve small-region
  volume. Retain both reference and model-grid volumes and their ratio.
- **Source-FOV truncation:** the supplied source may already exclude part of a
  biological shell. Padding cannot recover that anatomy.

The required crop envelope includes projected source support centers and mapped
support voxels, including centers of a region that disappears. Masks remain
empty outside available source support. Compressed `.mha` files preserve the
double-precision physical geometry needed for arbitrary lattice phase; written
geometry is read back before publication. The historical NIfTI resize route
remains an explicit diagnostic and is rejected by principal comparisons.

Verified pooling uses support-preserving mask reduction on coarse feature grids
and returns zero for empty support. Shared encoder receptive fields can still
include surrounding context; regional pooling does not imply that every feature
depends exclusively on interior voxels. Applied-support ROI scores can be
perfect by construction. Separate localizer scores assess predictions against
available annotations; neither is expert anatomical contour certification.

The geometry token retains 19 named crop-dependent descriptors and adds
reference-grid physical measurements with validity bits and region states.
Physical measurements include PT/nodal volume, centroid distance, boundary
voxel-center distance and nodal voxel RMS spread. Laterality requires a supplied
qualified midline. Node count and largest-node volume remain unavailable
placeholders: connected components are not treated as anatomical node instances.
Physical values use fixed signed `log1p(abs(value))`, distinct from the existing
crop-relative descriptors. See [schema](../src/trifusesurv2/schema.py) and
[geometry construction](../src/trifusesurv2/preprocessing/compartments.py).

Expert deployment consumes supplied contours for every dependent input.
Automated deployment requires predicted contours with out-of-fold or independent
segmenter provenance, and rebuilding the crop, radiomics and geometry from those
contours. Turning off teacher forcing does not convert manual-derived inputs
into an automated pipeline. Spatial flips are disabled for verified datasets
until every dependent input can be transformed consistently.

## Continuation and data integrity

The [trainer](../src/trifusesurv2/multimodal_survival/train.py) records a semantic
experiment specification and a model/input contract. These bind source identity,
patient splits, metadata, region manifests, radiomics, selected initialization,
runtime topology and resolved scientific settings. Resume must reject semantic
changes even when tensor shapes match: attention heads, objectives, dropout,
horizons, selection metrics, epochs and optimizer/scheduler settings matter.
Only the explicit reporting/loader allowlist in
[`utils/experiment.py`](../src/trifusesurv2/utils/experiment.py) is mutable.

Warm-start selection is shared between trainer and search through
[`utils/warmstart.py`](../src/trifusesurv2/utils/warmstart.py). The search's
schema-3 contract resolves explicit files, `all/` and relevant fold directories,
and hashes the selected checkpoint bytes before skipping or reusing a trial.
Replacing weights at the same pathname, making missing initialization available
or changing which directory wins resolution changes the contract. A pathname
or environment-variable fingerprint alone was insufficient.

Strict resume restores model and required optimizer, scheduler, scaler, EMA and
SWA state. Allowed warmup/pre-update exceptions are explicit. Partial resume
must have been declared in the original contract; it cannot be introduced to
rescue an incompatible attempt. A directory containing only `best.pt` does not
provide the original `last.pt` continuation state. Checkpoint selection respects
metric direction: NLL/IBS are minimized, while AUC/C-index/composites are maximized.

The [verified dataset cache](../src/trifusesurv2/utils/data.py) bounds complete
validated patient payloads per worker, including CT, supervision masks, shared
regions, state and geometry. Warm hits avoid repeated decoding/full CT hashing.
Bound-file signatures include resolved path, device, inode, size and nanosecond
mtime/ctime; changes trigger validation, changed manifest content is rejected,
and returned arrays are copies. Inputs must remain immutable: filesystem
signatures are a practical change detector, not a cryptographic snapshot.
I/O call-count tests do not establish a training-speed multiplier.

## What the October 27 evidence establishes

The frozen source passed 162 CPU engineering tests. Qualification then ran in a
worker with exactly one NVIDIA H100 80GB HBM3, compute capability 9.0, using
PyTorch 2.9.1+cu128, MONAI 1.5.2, NumPy 1.26.4, SimpleITK 2.5.2, SciPy 1.15.3
and PyRadiomics 3.0.1. Runtime receipts, rather than the installer defaults,
establish those versions. Qualification completed at 7:33 PM CDT; subsequent
overlay review accepted only development geometry.

| Evidence | Observed result | Scope limit |
|---|---|---|
| CPU regression suite | 162 tests passed before source publication | Engineering contracts and synthetic behavior |
| Common-grid tests inside the H100 worker | Seven CPU/SimpleITK checks passed | Their host had an H100; these tests themselves were CPU work |
| Actual PyRadiomics on synthetic supports | Invariance checks passed under the candidate recipe | Synthetic measurement checks, not clinical reproducibility |
| Tiny F2/F3 CUDA models | Finite forward/backward; equal 3,907,163 parameters and identical initialization hash | CUDA/model plumbing, not full-size training or convergence |
| Six outcome-free real representatives | Preparation passed; all six larger-canvas feature comparisons had zero mismatches at `rtol=1e-5`, `atol=1e-7` | Available prepared images and candidate normalized-intensity recipe only |
| Review of all six overlays | No obvious gross computational displacement; limitations retained | Development geometry review without expert contour-accuracy certification |
| Paired full-size development pilot | At 9:03 PM CDT, materialization was running; F2/F3 had not started | No completed epochs or performance result existed in that snapshot |

The representative comparisons assessed 2,140 features in total, ranging from
107 to 428 assessable features per case.
All six retained 100% of declared reference crop support. Five had both PT and
LN shell source-FOV flags. One nodal interior disappeared while its independently
constructed shell remained measurable; one case retained unavailable LN
annotation and an invalid empty primary shell. The largest primary
model/reference volume ratio was 1.219672. These observations support
retaining explicit QC states; they do not justify claiming negligible sampling
error or complete biological shells.

The development candidate is x/y/z spacing 1 × 1 × 3 mm, identity LPS direction
and zero global phase, with a D/H/W canvas of 96 × 160 × 96. The
[radiomics recipe](../configs/radiomics_normalized_candidate.json) uses only
`Original` features, bin width 0.0125 in normalized-intensity units, and no
extraction-time normalization, resampling, resegmentation or mask correction.
Its semantic configuration hash differs from a hash of the JSON file's bytes.
The qualification receipts bind these identities:

| Identity | SHA-256 |
|---|---|
| Common-grid policy | `40477d46fc753d18d3eae9858203a806af13e1c2c442764b4d0b4c316b617613` |
| Radiomics semantic configuration | `d5c7b075b394dca62244360d289b5eb9cd1e745fbb1878c169d7874956e88099` |

The working inventory has 613 unique patient records with already prepared CTs
and derived contours. Original CT/RTSTRUCT paths are unavailable, upstream
normalization/cropping history is unresolved, contours are unreviewed, and
normalized CT `>0.02` is only a development tissue candidate. None of the
checks above resolves those limitations. The
[development adapter](../scripts/prepare_development_inputs.py) records them
explicitly and does not create evaluation splits or start training.

## Development pilot versus the remaining study

The frozen pilot prepares all imaging inputs without consulting outcomes, then
records explicit outcome eligibility. The CSV audit found 519 valid OS pairs
(288 events, 231 censored) and 94 paired-missing outcomes; DSS and DFS each also
had 519 valid pairs. No malformed or incomplete pairs were found in that audit.
Paired-missing OS rows are excluded from the training delivery with reasons;
malformed/incomplete pairs fail instead of being silently dropped. This audit
does not establish endpoint ascertainment, follow-up adequacy or final imaging
eligibility. Both modes must use the same final eligible patients.

The pilot plan runs F2 then F3 on one H100, two epochs per mode, one fixed
patient-disjoint OS-stratified split and seed 42. It uses the full configured
backbone (feature size 96; depths 2/2/18/2; heads 3/6/12/24), batch size one,
eight-step gradient accumulation, AMP and gradient checkpointing. Dropout/noise
are zero; initialization and all shared scientific settings are matched.
Clinical/radiomics transformations fit only on training patients. Selection
uses validation NLL and exports the selected validation checkpoint.

Acceptance requires completed expected epochs, finite losses/risks/model tensors,
correct held-out membership and semantic contracts, matching parameter counts,
and hash-bound best/last checkpoints with positive optimizer-step evidence.
Finite AMP losses without optimizer updates would not establish training.
The pilot's `expert` selector denotes the contour-conditioned route; it does not
certify the supplied legacy contours. Two epochs and an explored development
split cannot establish convergence or a clinical F3 advantage.

The remaining scientific sequence is:

1. Resolve or explicitly bound acquisition/intensity/contour/tissue limitations;
   finish cohort materialization and review exclusions, failures, small-support
   loss and source-FOV truncation before fixing the study cohort.
2. Complete and inspect the paired capacity pilot, preserving failed criteria
   if any. Freeze the adequate training budget and measurement policy before the
   substantive F2/F3 comparison; a development pass is not that comparison.
3. Prespecify endpoint/time origin, ascertainment, horizons, censoring handling,
   checkpoint selection, splits/seeds and tuning opportunities. Include
   clinical-only and clinical plus physical burden/geometry baselines.
4. Report paired F3–F2 differences with uncertainty, discrimination and
   calibration. Separate variability across training seeds from patient-sampling
   uncertainty; retain explored folds as development data and seek independent
   validation before claims of clinical utility.
5. After the core contrast, study synchronized contour variation across CT
   pooling, radiomics and geometry, with an explicit robustness protocol.
   Contour-provenance support does not itself implement consistency training.

Individual-node phenotypes, multiscale shells, F5, learning curves and external
validation remain extensions or separate evaluation stages. They must not be
inferred from schema placeholders, legacy node code, software test success,
GPU allocation or representative-case measurement qualification.
