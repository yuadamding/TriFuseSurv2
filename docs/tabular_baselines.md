# C0/C1 baseline implementation

The baseline runner fits one deterministic penalized Cox model per input set,
using the same eligible patients and literal split inventories as the neural
comparison. It does not qualify the current cohort, confirm clinical coding,
or supply missing endpoint definitions. A draft study specification produces a
`baseline_receipt.json` with `status: blocked_or_failed` before fitting.

No real-cohort baseline fit has been established by the synthetic software
tests. Execute real-data fitting within the approved study run after endpoint,
clinical, measurement, split and evaluation-support gates pass.

## Inputs and clinical coding

C0 uses the pretreatment clinical information supplied to the fusion models.
The runner requires an explicit numeric/categorical policy; it does not infer
that a numeric code represents a continuous measurement, or guess HPV, smoking,
alcohol, pathology or stage coding. Treatment (`TX`) is excluded.

Every column in `baselines.fusion_clinical_columns` must be selected or accounted
for by confirmed staging redundancy. Use either T/N/M components or a confirmed
composite staging representation. For example, dropping `NSTAGE` requires a
documented, confirmed relationship to the retained stage components. The local
dictionary template does not establish that relationship. A string indicating
an unconfirmed source is not adequate scientific evidence for freezing the
policy.

Numeric features use training medians, followed by training means and standard
deviations. Numeric missingness has a separate indicator. Categorical levels
are learned only from training patients, with separate missing/unseen slots.
Constant training columns are dropped, including unavailable-only features; a
new held-out level cannot acquire an untrained coefficient. No validation or
test distribution is used to fit preprocessing.

## Physical geometry for C1

C1 adds these manifest-defined reference measurements:

| Feature | Representation |
|---|---|
| PT and LN reference volume | `log1p(volume_mm3)` |
| PT–LN centroid distance | Millimeters |
| PT–LN minimum surface distance | Millimeters, boundary voxel-center discretization |
| LN aggregate voxel RMS spread | Millimeters, not spread of identified nodes |
| Left/right/bilateral indicators | Only where manifest validity and anatomical midline support them |
| Measurement availability | Separate numeric missingness indicators |
| PT/LN annotation state | Present, absent, missing or invalid categorical values |

The manifest's known-absent lesion has a valid zero volume. A missing or invalid
annotation has an unavailable volume, which is imputed with its missingness
indicator. Distances involving absent anatomy remain unavailable. Reference
measurements are not replaced by model-grid crop fractions or inferred from
whether resampled image features survived.

`node_count` and `largest_node_volume_mm3` are excluded. The present geometry
version does not identify anatomical node instances; connected components are
not a substitute. Each manifest must pass the existing schema, unit, geometry,
region identity and physical-provenance checks. The frozen manifest inventory
binds the exact path and bytes for every eligible patient, and must match the
metadata's manifest path. Clinical measurement qualification and accepted source
limitations must also be declared prospectively; internal checksum consistency
does not establish contour review or acquisition provenance.

## Estimator and selection

Both modes use `sksurv.linear_model.CoxPHSurvivalAnalysis` with positive L2
penalties from the same frozen grid and Efron handling of tied event times.
The baseline cumulative hazard used for prediction is the library's Breslow
estimator. See the [official Cox model API](https://scikit-survival.readthedocs.io/en/v0.25.0/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html).

Select the penalty by minimum **validation IBS**, using the training outcome
distribution as censoring reference and the study's predeclared supported IBS
grid. This baseline selection criterion is distinct from the neural models'
validation NLL. Exact IBS ties choose the larger penalty. Every grid candidate
must converge without warnings and produce finite outputs; a failed candidate
stops the attempt instead of silently changing the search inventory.

The selected model is already fitted on training patients; it is not refitted
on validation or test patients. A pre-fit outcome inventory hash binds the
fixed cohort and training censoring reference; test labels do not enter fitting
or penalty selection. The implementation uses the reference package's
[integrated Brier score](https://scikit-survival.readthedocs.io/en/v0.25.0/api/generated/sksurv.metrics.integrated_brier_score.html),
with no hidden truncation of unsupported outcomes or extrapolation beyond
training follow-up. Support must be established before freezing the grid.

## Frozen configuration and execution

The runner enforces the shared evaluator's frozen specification and pinned
reference-library version before fitting. The study specification must have a confirmed
`endpoint_definition` containing `time_origin`, `time_unit`, `event_definition`
and `censoring_rule`. Optional `time_column` and `event_column` default to
`OS.TIME` and `OS.EVENT`. Values are not recoded: event 1 must already denote
the declared OS event and event 0 censoring.

`evaluation.outcome_inventory_sha256` must match the canonical complete outcome
table derived from the bound metadata and split inventories. This connects the
baseline cohort to the exact evaluator cohort and training censoring reference;
jointly dropping a patient from metadata and predictions cannot preserve the
same frozen contract.

`baselines` requires these keys:

```json
{
  "status": "frozen",
  "metadata_sha256": "<actual eligible metadata SHA-256>",
  "split_files_sha256": {"train": "<sha>", "val": "<sha>", "test": "<sha>"},
  "fusion_clinical_columns": ["<same clinical inventory as the fusion model>"],
  "clinical": {
    "status": "confirmed",
    "prediction_time": "<confirmed pretreatment availability>",
    "coding_source": "<qualified clinical dictionary identity>",
    "numeric": ["<confirmed numeric columns>"],
    "categorical": ["<confirmed categorical columns>"],
    "redundant_columns": {
      "NSTAGE": {
        "confirmed": true,
        "represented_by": ["T", "N", "M"],
        "justification": "<confirmed coding relationship; not an assumption>"
      }
    }
  },
  "geometry": {
    "status": "qualified_for_development",
    "manifest_inventory_sha256": "<actual inventory SHA-256>",
    "manifest_column": "region_manifest_path",
    "accepted_limitations": ["<qualified source limitations>"]
  },
  "alpha_grid": [0.01, 0.1, 1.0, 10.0, 100.0],
  "selection": "validation_ibs",
  "ties": "efron",
  "n_iter": 100,
  "tol": 1e-9
}
```

This is a schema illustration, not a qualified current-cohort policy. Replace
placeholders using verified evidence; do not change a status to bypass a gate.
The example penalty grid is prospective and requires freezing before fitting.

The manifest inventory is a JSON object keyed by literal patient ID, with
`{"path": "...", "sha256": "..."}` entries. Relative paths resolve against the
metadata CSV directory. An optional `geometry.scan_column` additionally checks
the metadata scan ID against the manifest. Split files are `train_ids.txt`,
`val_ids.txt` and `test_ids.txt`; duplicate, missing, overlapping or extra
patients are errors. Metadata must contain exactly their eligible union.

```bash
PYTHONPATH=src python scripts/fit_tabular_baselines.py \
  --study-spec /approved/run/study_spec.json \
  --metadata-csv /approved/run/eligible_metadata.csv \
  --split-dir /approved/run/splits/fold_00 \
  --manifest-inventory /approved/run/manifest_inventory.json \
  --out-dir /approved/run/new-baseline-attempt
```

The output directory must be new. `--modes C0` permits an explicitly requested
clinical-only fit without a geometry inventory; the default requires both C0
and C1. The command does not launch a scheduler or select hardware. Install the
repository's pinned `study` extra in the source-bound execution environment.

## Artifacts and common evaluation

Each mode writes a JSON model checkpoint containing coefficients, preprocessing
parameters, the baseline cumulative hazard and contract hash. The contract
records source/dependency identities, input/split hashes, clinical policy,
geometry inventory and every candidate's validation score. No executable pickle
is needed to inspect the fitted model. Train, validation and test survival
curves use the shared long-form export and carry checkpoint/contract hashes.

Report `C0` and `C1` with `seed: deterministic`. The shared evaluator supports
`evaluation.mode_seeds` so their single fit can appear in the same results table
as F2/F3 seeds 42, 142 and 242. Use paired patient resamples across modes, and
declare F3–F2 as the principal contrast. Do not duplicate the deterministic
baselines three times and label those copies as training-seed variation.

The synthetic tests cover held-out preprocessing leakage, test-outcome changes
leaving selection unchanged, literal IDs/split overlap, confirmed clinical
policies, absent versus unavailable geometry, reference-library scoring,
checkpoint prediction reconstruction and unsupported horizons. These are
engineering checks, not clinical model performance or scientific acceptance.
