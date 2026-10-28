# Paired survival evaluation and curve exports

This implementation is prospective software. Synthetic reference checks establish
engineering behavior; they do not qualify the local cohort, demonstrate H100
training, establish convergence, or show that F3 improves prediction. The
completed two-epoch pilot remains bound to source
`614d7fbdb3551f58c2e8b5da403f7b6f5418f23e` and does not gain new export artifacts
retroactively by changing this repository.

## Input contract

`scripts/evaluate_paired_survival.py` accepts a frozen study JSON, an outcome CSV,
and one or more long-form prediction CSVs. Patient IDs are literal strings:
`001`, `1`, and `NA` remain distinct. Outcomes use
`Patient_ID,split,endpoint,observed_time,event`, optionally `fold`. Every patient
occurs once, and train/validation/test inventories must be disjoint. Times have
the protocol's confirmed unit; events are binary.

The frozen `evaluation.outcome_inventory_sha256` binds the complete normalized
train/validation/test table, including literal IDs, splits and labels. Use
`evaluation.paired.outcome_inventory_sha256(frame, endpoint)` to compute it.
Rows are sorted by literal ID, times normalized with Python's round-trip float
parser, events to integers, and records serialized as sorted-key compact JSON.
Jointly removing a patient from outcomes and every prediction file still fails
this binding, as does changing the training censoring reference. Optional
`outcomes_file_sha256` additionally binds the exact CSV bytes.

Predictions use these exact columns:

```text
Patient_ID,split,fold,seed,mode,endpoint,observed_time,event,evaluation_time,survival_probability,checkpoint_sha256,model_contract_sha256,risk_horizon,risk_score
```

Each fit must contain exactly the outcome inventory for the specified split, with
one row per patient and evaluation time. Each required evaluation time must be
present explicitly. Additional native bin endpoints are allowed and checked for
monotonicity. Missing cases, duplicate rows, incomplete seeds, mixed checkpoint
hashes, outcome mismatches, NaN/Inf, probabilities outside [0,1], increasing
survival, and incorrect risk direction are errors. Scalar risk must equal
`1-S(primary_horizon)`; no patient dropping or interpolation repairs these errors.

Every row carries the selected checkpoint and model-contract SHA-256. The CLI's
optional `--artifacts` manifest maps `MODE/SEED` to `checkpoint_path` and
`model_contract_path` and verifies the actual file bytes. Without it, the result
explicitly says `declared_export_hashes_only`; declaration is not file verification.
Its status is `engineering_evaluation_unverified_artifacts`, not completed
scientific artifact qualification.
Relative paths in this manifest resolve against the manifest directory.

## Protocol interface and gates

Scientific scoring requires `status: frozen` and an `endpoint_definition` with
`status: confirmed` plus nonempty `time_origin`, `time_unit`, `event_definition`,
and `censoring_rule`. Column names and file integrity do not confirm the clinical
endpoint definition. The evaluator never freezes the protocol.

The study's `evaluation` block contains:

| Key | Meaning |
| --- | --- |
| `endpoint`, `fold`, `split` | One endpoint, one held-out fold and `val` or `test` per invocation |
| `modes`, `seeds` | Complete prespecified fit inventory |
| `mode_seeds` | Optional exact mode-to-seed mapping; C0/C1 can use `deterministic` |
| `contrasts` | Ordered `[comparator,candidate]` pairs; both require the same ordered seeds |
| `primary_horizon`, `secondary_horizons` | Explicit prediction/calibration horizons |
| `concordance_tau` | Explicit IPCW concordance event-time truncation |
| `ibs_times` | At least two positive, strictly increasing times defining the entire integration grid |
| `calibration_bins` | Fixed probability edges, e.g. `[0,.2,.4,.6,.8,1]`; never test-fitted quantiles |
| `tied_tol` | Risk tie tolerance, normally `1e-8` |
| `bootstrap_replicates`, `bootstrap_seed` | Prospectively fixed draws; the requested study uses 2,000 |
| `bootstrap_min_valid_fraction` | Prespecified minimum valid fraction for a reported interval |

With `mode_seeds`, the core table can include deterministic C0/C1 and three-seed
F2/F3. A deterministic baseline is not copied into three fictitious independent
fits. The default paired contrast is between the first two modes; explicitly set
`contrasts` when adding baselines. A C0/C1 contrast is permitted when both have
the deterministic seed. A deterministic/deep contrast requires a separately
specified estimand; the current matched-seed contrast rejects it.

`--support-only` accepts a draft spec, reads development outcome rows only, and
rejects prediction/artifact arguments. It reports patients at risk, events and
censorings by each candidate horizon, observations beyond it, and training
censoring survival. Optional `candidate_horizons` overrides the primary/secondary
candidate list. Thresholds and horizon selection still belong in the prospective
protocol; this command does not automatically choose the most favorable horizon.

```bash
python scripts/evaluate_paired_survival.py \
  --spec study_spec.json --outcomes outcomes.csv \
  --support-only --out-dir support-report

python scripts/evaluate_paired_survival.py \
  --spec study_spec.json --outcomes outcomes.csv \
  --predictions F2_42_test.csv F2_142_test.csv F2_242_test.csv \
                F3_42_test.csv F3_142_test.csv F3_242_test.csv \
  --artifacts selected-artifacts.json --out-dir paired-report
```

Use separate files/invocations for validation and test; extra split rows are
rejected. Existing report JSONs are preserved rather than silently overwritten.

## Estimators and uncertainty

The implementation requires `scikit-survival==0.25.0` and delegates IPCW
concordance, pointwise Brier and integrated Brier to that reference. It fits the
censoring distribution using the training outcomes, with higher scalar risk
meaning earlier death. The explicit `concordance_tau` is distinct from the risk
horizon. Risk ties follow the declared tolerance. See the official
[IPCW concordance API](https://scikit-survival.readthedocs.io/en/v0.25.0/api/generated/sksurv.metrics.concordance_index_ipcw.html).

IBS uses survival probabilities on the complete frozen common grid, trapezoidal
integration divided by the interval length. The interval is exactly the first
and last `ibs_times` entries. This is a new reference-backed analysis; the
trainer's historical custom bin-endpoint `ibs` remains a diagnostic. See the
[IBS API](https://scikit-survival.readthedocs.io/en/v0.25.0/api/generated/sksurv.metrics.integrated_brier_score.html)
and [reference implementation](https://github.com/sebp/scikit-survival/blob/v0.25.0/sksurv/metrics.py).

The additive `supporting_horizon_brier` report scores every distinct
`primary_horizon` and `secondary_horizons` entry, including horizons outside the
IBS integration grid. It uses the explicit survival probability at that time and
the pinned [Brier-score reference](https://scikit-survival.readthedocs.io/en/v0.25.0/api/generated/sksurv.metrics.brier_score.html),
with censoring estimated from training patients. Lower is better. The existing
per-fit `brier` array still describes only `ibs_times`; neither the primary IBS
interval nor its meaningful-improvement threshold changes.

Each supporting horizon reports per-fit scores, per-mode seed means and sample
SDs, and matched-seed candidate-minus-comparator differences. Its percentile
intervals regenerate exactly the primary patient draws, preserving pairing
across all fits, but count valid draws independently by horizon. A secondary
horizon outside a bootstrap sample's follow-up therefore cannot change the
primary IBS/concordance intervals. These supporting intervals are pointwise;
they are not simultaneous confidence bands or adjusted for multiple comparisons.

Known, independently checked reference follow-up failures mark that supporting
horizon `unassessable`, with the exact error and follow-up diagnostics; primary
outputs and other assessable horizons remain available. No patient or time is
removed, no score is imputed, and no alternative horizon is chosen. Bootstraps
are not run for a horizon whose full-cohort score is unassessable. For assessable
horizons, invalid draws and their reasons are retained; insufficient valid draws
yield null intervals while retaining the point scores. The existing protocol,
all-horizon training censoring support, prediction and calibration gates still
apply. Unexpected reference errors propagate instead of becoming scientific
unassessability.

Unknown/zero training censoring support, a grid outside held-out follow-up range,
or otherwise undefined reference scores fail explicitly. The reference Brier
implementation also checks observed follow-up support. The evaluator does not
silently administratively censor late observations, alter the grid after seeing
test outcomes, or extrapolate the censoring distribution. If administrative
truncation is desired, it needs a separately reviewed prospective policy and
implementation; it is not an automatic recovery option.

Calibration uses fixed probability bins and training-reference IPCW event
incidence: within each bin, average `I(event and T<=h)/G_train(T)` over **all**
patients in the bin. Censored-before-horizon people remain in the denominator
without being assigned known event-free outcomes. Empty bins report nulls.
This Horvitz--Thompson estimate can exceed one in small bins; it is reported
without clipping, alongside event, early-censoring and beyond-horizon counts.
Calibration tables are descriptive and currently have no confidence intervals.
Training censoring estimation uses the reference's reverse Kaplan--Meier tie
convention, described in its
[Kaplan--Meier API](https://scikit-survival.readthedocs.io/en/v0.25.0/api/generated/sksurv.nonparametric.kaplan_meier_estimator.html).

Each bootstrap draw samples held-out patient indices once and applies those
indices to every fit. A paired statistic is the average of the per-seed
candidate-minus-comparator differences; models/seeds are not independent patient
samples. Reports include point estimates, per-seed differences, seed sample SD,
and percentile 95% intervals. Absolute per-mode metrics also receive intervals
from the same draws, averaging scores across that mode's seeds.

Undefined draws are counted with reasons and never replaced. If the valid
fraction is below the prespecified threshold, the interval is null. The report
records requested/valid counts and a hash of all draws. This quantifies held-out
patient sampling conditional on the fixed trained models and training censoring
reference; it does not include development-set sampling or establish independent
external validation.

## New-run curve export

Future training can enable `--export_survival_curves` and pass the frozen
evaluation grid and secondary horizons through `--curve_export_times_days`.
After the existing checkpoint selection, the trainer restores the selected
checkpoint/EMA/SWA state and exports validation and test curves. The training
architecture, objectives, optimizer, schedule and selection rule are unchanged.
For EMA/SWA, it first persists the actual selected inference state in a separate
checkpoint and binds that file's hash. This also supports lightweight source
checkpoints that intentionally omitted averaging state. The inference artifact
is not an optimizer-resume checkpoint and existing artifacts are not overwritten.

Exports include zero, every native end-of-bin time, the primary risk horizon and
all requested extra times. They use the v2 model's existing piecewise exponential
within-bin convention and hazard bounds `[1e-7,1-1e-7]`, evaluated in NumPy double
precision. Requested times beyond the trained hazard bins fail before training;
there is no constant-tail extrapolation. Row outcomes come from the original
split metadata, preserving precision rather than copying float32 batch labels.
The adjacent `.csv.json` receipt records file/hash identity, weight type, source,
bin width/count and interpolation convention.

The frozen pilot has no retroactive curve export. A separate, source-pinned
checkpoint postprocessor remains necessary for its historical checkpoints.
New source-bound substantive fits must explicitly enable this export and bind
their complete prediction grid before launch.

Future epoch CSVs also retain full-precision `val_nll` and
`epoch_train_and_evaluation_seconds`. The latter is monotonic wall time from
epoch entry through training, training-split evaluation and validation-split
evaluation, including the epoch's initial setup. It excludes checkpoint writes,
SWA bookkeeping, Grad-CAM and final selected-checkpoint export. It is not pure
GPU kernel time. Existing CSV headers must match exactly before appending;
historical files are not silently upgraded. The frozen pilot lacks these fields,
and its rounded console NLL cannot be substituted for a full-precision trace.
No AMP attempted/skipped update counters are inferred from timing or gradient
scale changes; successful survival-path updates require their own evidence.

## Synthetic verification

`tests/test_paired_survival_evaluation.py` compares actual reference calls with
independent IPCW pair/Brier calculations, tests risk direction and censoring/risk
ties, identical-model zero contrasts, literal IDs, complete fit inventories,
curve support/monotonicity, missing values and unsupported horizons. The export
integration fixture checks the selected endpoint, patient ordering, original
outcome precision, bin grid, checkpoint/contract hashes and sidecar binding.
`tests/test_supporting_horizon_brier.py` additionally checks horizons outside the
IBS grid, manual training-reference IPCW scores and bootstrap intervals,
independent support failures, paired seed contrasts and unchanged primary output.
Run these as engineering fixtures in the pinned study environment; clinical
evaluation remains subject to the study and single-H100 execution gates.

Version 0.25 supports Python 3.10 and scikit-learn 1.6/1.7, matching the intended
worker interpreter; see its
[installation requirements](https://scikit-survival.readthedocs.io/en/v0.25.0/install.html).
An approved worker image alone does not prove that these study dependencies or
reference fixtures have been qualified inside a new execution environment.
