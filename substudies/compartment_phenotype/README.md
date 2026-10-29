# Prespecified compartment phenotype substudy

This is the user's October 29, 2026 cohort-only plan: measure primary and nodal
CT intensity IQR on the same delivered supports, qualify perturbation
robustness, assess the fixed model's precision, and only then consider OS
association. It is separate from neural training and requires no GPU job.

The [completed October 29 milestone](RESULTS_20261029.md) records a passed
paired measurement gate, failed paired design gate and failed fallback
measurement gate. No real OS association or held-out model assessment was run.

The protocol is prespecified relative to these new phenotype–OS associations.
It is informed by the prior local data-characteristics audit; it is not an
external registration or a claim that the underlying cohort is independently
validated. Raw source files and original split labels are preserved.

## Bound execution stages

1. Freeze `protocol.yaml`, code hashes, interpreter/package identities and
   input source hashes in a new run directory. Keep restricted identifiers
   outside the repository. Source locks must exist before real execution.
2. `prepare.py --stage sources`, then `screen`, then `cohort`: rehash the
   delivered files, screen all 613 images for physical-coordinate overlap
   across UIDs, audit authentic timing, and prepare all applicable eligibility
   flags. The known shifted-crop pair must be recovered by the fingerprint
   screen itself. Unresolved conflicts are analytical exclusions; no image or
   source record is moved or deleted.
3. `measure.py --stage extract`, then `prepare.py --stage finalcohort`:
   independently validate IQR, volumes, geometry, support identity and M/D
   reconstruction. Publish exact patients, deaths, censoring counts and
   fixed-order exclusions by original split in CSV and Parquet.
4. `measure.py --stage qualify`: use eligible **original training records
   only**, without endpoint columns. Erosion/dilation use physical spacing;
   dilation uses padding to detect loss outside the delivered support.
   Coarsening preserves outer voxel edges, direction and physical FOV.
5. Only if paired qualification passes, `analyze.R --stage design` runs the
   fixed 1,000 simulations per scenario; `lock` freezes the resulting analysis
   contract. `primary` and `supporting` are separate explicit stages. Test
   application is another explicit stage, never a default side effect.
6. A failed paired measurement/design gate permits the independently
   prespecified primary-only sphericity fallback **before any real paired
   association is inspected**. `fallback-extract` and `fallback-qualify` use
   the pinned PyRadiomics 3.0.1 mesh definition and a separate feature table;
   `fallback-design` has two 1,000-simulation scenarios. Its own gates are
   mandatory before `fallback-fit` or `fallback-test`.
7. `report.py` reads existing artifacts and produces Markdown, self-contained
   HTML and optionally PDF. It never invokes an association or test-set fit.

Python CLIs require `--protocol PATH --output-root RUNROOT` (preparation and
measurement); reporting uses `--protocol PATH --run-root RUNROOT [--pdf]`.
R uses `--protocol PATH --run-root RUNROOT --stage STAGE`.

## Prespecified operational choices

- IQR uses linear quantiles on delivered normalized intensities; zero IQR is
  valid. Volumes use voxel count × spacing product / 1,000 cm³, not mesh volume.
- Minimum support is 100 retained voxels per required region, equivalent to
  0.025 cm³ on this delivered grid. This is a project rule, not clinical truth.
- The three gating perturbations are 1-mm erosion, 1-mm dilation and 1-mm
  isotropic coarsening. The 2-mm grid is a separate stress test.
- Per condition: evaluability ≥90%, absolute-agreement ICC(A,1) ≥0.90,
  percentile paired-bootstrap lower 95% bound ≥0.75, and median absolute
  change ≤0.20 unperturbed original-training SD. Bootstrap uses 1,000 patient
  pairs. Undefined variation or inadequate finite bootstrap support fails.
- Design criteria: usable fits ≥95%, null rejection ≤7%, confidence-interval
  coverage 93–97%, and primary log-HR interval width ≤2 log(1.5). The precise
  coverage band is a frozen operational interpretation of the supplied plan's
  “approximately calibrated” requirement.
- The width rule concerns the primary D estimand in null/alternative scenarios.
  The correctly specified seven/eight-slope temporal stress guard retains its
  usable-fit/null/coverage checks; its D-late width is reported separately and
  does not veto primary precision. Misspecified six-slope PH behavior under a
  changing M effect is reported explicitly, not validated by that guard.
- Timing unknown stays unknown in manifests. Diagnosis-origin associations
  under that limitation are exploratory retrospective, not verified baseline
  or pretreatment predictions. No processing time is a clinical date.

## Validation and evidence boundaries

Focused fixtures test shifted-crop detection, constants, flag/flow ordering,
Parquet literal-ID/bool round trips, padded morphology, edge-preserving oblique
resampling, independent linear quantiles, zero/empty supports, absolute ICC,
the actual sphericity API, six/fifteen/seven/eight/four-slope model dimensions,
coefficient covariance transformations, same-patient LRTs, temporal death
accounting, patient bootstrap pairing, source bindings and fail-closed gates.
These fixtures are not real-cohort qualification or a scientific result.

`protocol.yaml` contains exact roots and seeds for the initial local attempt.
Outputs are under `/storage/TriFuseSurv/runs/phenotype-substudy-20261029`.
Source, feature and cohort SHA-256 bindings must match the immutable run lock;
do not edit a bound generation or relax a failed criterion after its results.
