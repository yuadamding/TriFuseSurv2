# Whole-cohort accounting — October 28, 2026

**Accounting and computational development review pass.** The frozen inventory
contains 519 OS-valid patients with unchanged 365/77/77 assignments. A separate
review records a disposition for all 613 prepared cases, 2,452 compartments and
525 queued cases. It joins retained metadata, manifests, radiomics, sampling
audits and input receipts; it does not re-extract features, verify current
CT/mask bytes, or establish anatomical or clinical qualification.

There are **613 prepared patients, 519 OS-valid patients, and 94 exclusions for
paired-missing OS outcomes**. The unchanged development split is 365/77/77.
All 519 supplied endpoint records match the eligible delivery numerically and
by literal patient ID. Outcome exclusions do not imply missing primary tumors.
The tables contain 613 patient rows and 2,452 compartment rows.

## Compartment accounting

All counts below use the full 613-patient prepared inventory. Anatomical states
refer to the reference annotation; image availability additionally reflects
model-grid sampling. No unavailable annotation is relabeled disease absence.

| Compartment | Reference states | Image available | Radiomics available | Disappeared on model grid | Extraction failures | Shell/source-FOV flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PT interior | {"missing": 1, "present": 612} | 612 | 612 | 0 | 0 | 0 |
| PT shell | {"invalid": 116, "missing": 1, "present": 496} | 496 | 496 | 0 | 0 | 481 |
| LN interior | {"missing": 117, "present": 496} | 493 | 490 | 3 | 3 | 0 |
| LN shell | {"missing": 117, "present": 496} | 496 | 496 | 0 | 0 | 473 |

Unknown nodal exclusion anatomy invalidates the PT shell in **116** cases.
One additional patient is missing PT itself and is excluded from that causal
intersection. Three known-present LN interiors disappear after resampling
while their independently constructed shells remain measurable. Three other
LN interiors have extraction failures. These are different failure mechanisms.

## Sampling distributions

Volume distributions include only present reference anatomy with valid support
and passed declaration checks. Missing/invalid placeholders are excluded;
true model-volume zeros caused by disappearance remain included. The JSON
records each population, count, exclusions and missing measurements.

| Compartment | Qualified reference population | Median model/reference ratio | 5th–95th percentile | Minimum–maximum ratio |
| --- | ---: | ---: | --- | --- |
| PT interior | 612 | 1.0001 | 0.9580–1.0525 | 0.5924–1.7363 |
| PT shell | 496 | 0.9991 | 0.9893–1.0088 | 0.9516–1.0501 |
| LN interior | 496 | 0.9999 | 0.9266–1.0897 | 0.0000–2.6667 |
| LN shell | 496 | 1.0006 | 0.9810–1.0198 | 0.9557–1.0543 |

Source-FOV flags affect 481 PT shells and 473 LN shells. Crop retention cannot
restore tissue missing from the previously cropped source. Typical near-unit
volume ratios do not erase the extreme small-region changes or disappearing
interiors. The distribution tails require review, not an automatic exclusion.

## Completed computational review and frozen development policy

The deterministic queue contains **525 cases**: 520 with one or more recorded
flags and five ordinary examples. Its rules include failed geometry/binding,
disappearance, extraction errors, boundary/crop flags and ratios outside
[0.8,1.2]. There are no actual geometry/binding failures or declared crop losses.
All 2,452 manifest geometry/binding checks pass, and each declared reference
retention fraction is 1.0. These are checks of recorded geometry and sampling
counts, not a new verification of image or mask bytes.

The separate computational ledger completes the following outcome-blind
dispositions without changing the original visual-review fields:

| Finding | Computational disposition |
| --- | --- |
| Missing or invalid anatomy | Preserve the unavailable image/radiomics masks; do not substitute disease absence. |
| Unknown nodal exclusion anatomy | Preserve unavailable PT shells and their recorded cause. |
| Source-FOV limits in 518 unique cases | Retain the source-limited supports and flags; do not infer tissue outside the supplied source. |
| Three disappeared LN interiors | Mask their model-grid image and radiomics features; preserve valid reference geometry and independently valid shells. |
| Three other LN extraction failures | Preserve image availability and mask the failed radiomics features. |
| Twenty model/reference ratio outliers | Record each exact reference/model volume, voxel count and scale; retain both measurements and the sampling limitation. |
| Five ordinary queue examples | Confirm metadata declarations/bindings; mark unavailable image review explicitly. |

The 20 outliers comprise six PT and 14 LN interiors, including the three
disappearances. Their reference voxels measure 0.5×0.5×1 mm (0.25 mm³), while
model voxels measure 1×1×3 mm (3 mm³). The case ledger records actual support
counts; it does not assign every change to small-region rounding. For example,
one PT increases from 3,797.75 to 6,594 mm³ and one LN from 2,770.25 to
4,275 mm³. These substantial differences remain explicit measurement
limitations. The declared full-grid counts equal the retained crop counts, so
the recorded changes occur before final cropping; no new spatial validation
or anatomical explanation is inferred.

The six retained representative PNGs match their historical bytes, but their
full-cohort manifests differ from the representative manifests. The reporter
therefore does not transfer historical approval to the current manifest set.
A separate fresh inspection of the six available panels records visible boundary
truncation, the invalid shell with unknown LN anatomy, small-primary volume
change, and a disappeared LN interior with a retained shell. It is limited to
displayed planes and tables; it neither verifies all 3D supports nor closes
the visual/clinical review queue or establishes expert anatomical accuracy.
All 525 original `review_status: pending` values are preserved; the new
`computational_review_status: completed` describes the narrower review actually
performed.

Frozen computational development policy: retain the common grid, measurement
recipe and exact 519-patient OS inventory while representing unavailable
compartments explicitly. There are no new exclusions, feature-availability
changes or split changes. The 94 paired-missing OS exclusions remain outcome
exclusions. Measurement dispositions use no outcomes, split labels or model
predictions; the existing eligibility decisions are joined afterward. Any later
measurement change or new exclusion must version the policy/inventory and
preserve the original assignments of remaining patients.

The baseline geometry inventory is **qualified for development** under these
limitations and binds all 519 literal patient IDs to the exact manifest paths
and hashes in the frozen eligible metadata. Reference PT volume is valid for
519 patients; LN volume, PT–LN centroid distance, minimum surface distance and
LN spread are valid for 414. Laterality, anatomical node counts and largest-node
volume are unavailable throughout. Unavailable measurements retain false
validity flags and are exported as null, not fabricated observed zeros. This
qualification does not depend on choosing a neural training epoch budget.

Original CT/RTSTRUCT inputs and normalization history remain unavailable.
Contours are derived and lack expert qualification; the tissue threshold is a
development candidate with no restored HU interpretation. Neither this cohort
nor the old evaluation history establishes an untouched external validation set.

## Evidence locations

The immutable current report is under
`/storage/TriFuseSurv/runs/study-20261028/cohort-quality-v3/`:
`cohort_qc_patients.csv`, `cohort_qc_regions.csv`, `cohort_eligibility.csv`,
`cohort_review_queue.csv`, `cohort_qc_errors.csv`, and `cohort_qualification.json`.
The initial and v2 reports are preserved; v3 fixes population denominators so
unknown-anatomy zero placeholders do not distort volume distributions.

The new computational review is separately recorded under
`/storage/TriFuseSurv/runs/study-20261028/cohort-development-review-a/`:
`patient_dispositions.csv`, `compartment_dispositions.csv`,
`queue_dispositions.csv`, `volume_ratio_case_review.csv`,
`geometry_measurements.json`, `manifest_inventory.json`, and
`geometry_qualification.json`. The case report and source bindings preserve
all original review fields, input identities and the six limited panel
inspections. Earlier reports and their pending review states remain unchanged.

The exact eligible manifest inventory SHA-256 is
`4205ccd31e36905c5a04d5d973451a4201b941fd7428db6c1a34a2b1e9da092f`.
The geometry qualification SHA-256 is
`55a539798df389f513d726639374c560d84cb33f4d0c80b84e95ed1d152d8013`.

Portable aggregate identities and receipt hashes are in
[study_preparation_evidence.json](study_preparation_evidence.json).
See [the report interfaces](acceptance_reports.md) and
[prospective study protocol](study_protocol.md) for remaining acceptance gates.
