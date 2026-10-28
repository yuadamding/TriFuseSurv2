# Whole-cohort accounting — October 28, 2026

**Accounting passes; measurement review remains pending.** This report joins
retained metadata, manifests, radiomics, sampling audits and input receipts.
It does not re-extract features, verify current CT/mask bytes, or qualify
original anatomy and clinical provenance.

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

## Review and policy

The deterministic queue contains **525 cases**, triggered by geometry/binding
failures, disappeared regions, extraction errors, boundary/crop flags or ratios
outside [0.8,1.2], plus five ordinary examples. These are review triggers,
not new patient eligibility rules. Every queue adjudication remains pending.

The six retained representative PNGs match their historical bytes, but their
full-cohort manifests differ from the representative manifests. The reporter
therefore does not transfer historical approval to the current manifest set.
A separate fresh inspection of the six available panels records visible boundary
truncation, the invalid shell with unknown LN anatomy, small-primary volume
change, and a disappeared LN interior with a retained shell. It is limited to
displayed planes and tables; it neither verifies all 3D supports nor closes
the full review queue or establishes expert anatomical accuracy.

Proposed development policy: retain the frozen common grid, measurement recipe
and OS inventory while representing unavailable compartments explicitly. Do not
exclude a patient solely because nodal radiomics are unavailable. If adjudication
requires exclusions or measurement changes, version the cohort/policy and retain
original split assignments for remaining patients before substantive fitting.

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

Portable aggregate identities and receipt hashes are in
[study_preparation_evidence.json](study_preparation_evidence.json).
See [the report interfaces](acceptance_reports.md) and
[prospective study protocol](study_protocol.md) for remaining acceptance gates.
