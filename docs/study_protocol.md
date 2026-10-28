# Prospective OS development study

The [machine-readable specification](../configs/study_spec.json) is **draft**.
The fusion architecture remains fixed. This document defines the proposed next
comparison and the evidence still required before substantive fitting; it is
not a launch receipt or a claim of prognostic improvement.

The endpoint CSV supplied by the user contains 519 records. Every literal
patient ID and numerical OS time/event pair matches the retained eligible
delivery. The original 365/77/77 training/validation/test assignment is unchanged.
The user confirmed **days** and **1 = death**. The CSV has no time-origin or
censoring definition, so those fields remain unconfirmed. Zero is the observed
non-event code; its clinical ascertainment rule is not inferred from the file.
Source and split hashes are in the specification and outcome reconciliation
receipt. The separate full-cohort accounting includes 613 prepared patients,
of whom 94 have both OS fields missing.

## Development decisions and remaining gates

| Item | Prospective decision or unresolved gate |
| --- | --- |
| Primary endpoint/comparison | OS; F3 minus F2 |
| Patient membership | Retain the existing eligible delivery and split; version any justified exclusions before fitting |
| Seeds | 42, 142, 242, paired across F2/F3; all six fits required |
| Architecture | Fixed existing F2/F3 implementation; no new fusion component |
| Selection | Minimum validation NLL; best.pt retains first strict minimum; final export ties use existing last/EMA/SWA/best priority among enabled candidates |
| Image-token dropout | Both principal comparison paths remain zero |
| Other objectives/settings | Same bound scientific configuration in both modes; the draft records pilot settings and the full original plan hash |
| Training duration | Unset until pilot optimization/runtime and development learning curves justify a common budget |
| Baselines | C0 clinical; C1 the same clinical information plus qualified reference geometry |
| Execution | Sequentially within the user-authorized one-Kubernetes-H100 cap; Pending jobs count |
| Reporting | Reference IPCW concordance, fixed-grid IBS and censoring-aware calibration; scalar risk is `1-S(primary_horizon)` |

The two-epoch pilot offers 46 optimizer-step opportunities per epoch and 92 per
mode. It can qualify execution and capacity, not adequate training. The new
auditor requires evidence of successful updates in the shared and OS residual
survival heads, finite required losses/weights/risks, exact patient membership,
checkpoint selection and paired contracts. Anonymous Adam state counters are
insufficient to identify those trained parameters. See
[acceptance reports](acceptance_reports.md).

Before a longer experiment, compare development loss curves and actual epoch
runtime, declare one common maximum epoch/update budget and selection rule,
and bind the new source, inputs, environment and complete export grid. Do not
choose an arbitrary 80/100 epochs, stop the panel after an unfavorable first
pair, or tune against held-out results. The pilot's 1,095-day risk setting is
preserved in its frozen execution; the proposed substantive 730-day setting
requires a separately identified attempt.

## Proposed horizons and integration interval

The following metadata-only planning calculations use training censoring
Kaplan–Meier estimates and training/validation outcomes. No held-out model
predictions were available or read. They provisionally interpret code 0 as
censoring while the endpoint definition is completed.

| Horizon (days) | Training observed beyond | Validation observed beyond | Training events by horizon | Validation events by horizon | Training censoring survival |
| --- | ---: | ---: | ---: | ---: | ---: |
| 365 | 335 | 70 | 29 | 5 | 0.9971 |
| 730 | 313 | 61 | 49 | 12 | 0.9909 |
| 1,095 | 287 | 55 | 69 | 16 | 0.9712 |

These support the proposed 730-day primary horizon and 365/1,095-day secondary
horizons. They do not prove that the final held-out reference metrics are
assessable. Concordance truncation is proposed at 730 days, separately declared
from the risk horizon even though their values coincide.

The proposed IBS interval is **210–730 days**, at 30-day steps plus explicit
365- and 730-day points. Training and validation minimum observed follow-up are
10 and 194 days; 210 is the first 30-day grid point strictly above both.
This satisfies the reference metric's lower evaluation-time requirement on
development data. It is not IBS from time zero. Fix this complete grid before
held-out comparison; unsupported test support produces an unassessable result,
not patient deletion or an after-the-fact interval change.

The prospective fallback rule is recorded in the specification. Horizon choice
requires positive training censoring survival, observations beyond the horizon
and observed events in both training and validation. If 730 days becomes
unsupported because a justified cohort correction changes development support,
propose 365 in a new specification **before** fitting/scoring. The software never
silently shortens horizons or extrapolates follow-up/censoring support.

## Estimation, exports and uncertainty

Use the pinned `scikit-survival==0.25.0` reference implementation in a separately
qualified study environment. The `study` optional dependency set records the
Python-3.10-compatible numerical versions used by synthetic fixtures. Installing
them locally does not qualify the approved worker environment; the active pilot
overlay remains unchanged.

The evaluator binds the entire outcome/split inventory, requires complete
per-patient survival curves and verifies actual checkpoint/contract files for
scientific artifact qualification. The trainer's historical custom IBS remains
a diagnostic, separate from the reference estimate. Future exports include native
bin endpoints and the exact frozen evaluation times; averaged inference weights
are separately serialized and hashed. See [the evaluator contract](paired_evaluation.md).

Use 2,000 paired patient-bootstrap draws, seeded at 20261028, with the same
resampled indices for every fitted mode and seed. The primary statistic averages
the three per-seed F3-minus-F2 differences. Report per-seed estimates and seed
spread separately. Percentile 95% intervals quantify held-out patient sampling
conditional on the fitted models and training censoring reference; they do not
include retraining uncertainty. Record invalid draws and reasons; require at
least 95% valid draws to report an interval, without replacing failed draws.

Calibration at each supported reporting horizon uses fixed risk-bin edges
0, .2, .4, .6, .8, 1 and training-reference IPCW observed event incidence.
Report bin counts, events, early censoring and observations beyond the horizon.
Sparse/empty bins remain visible; no test-fitted bins or uncensored binary-label
regression. Calibration is descriptive, without a claimed calibration interval.

## Clinical and geometry baselines

The proposed C0 encoding keeps AGE continuous and preserves categories for
coded fields, including N2A/N2B/N2C and T4B. Clinical code meanings, prediction-time
availability, staging edition and whether NSTAGE is a redundant overall stage
group remain unresolved. Do not fit both an unqualified composite stage and
its purported components, silently drop information, or relabel unknown HPV
codes as negative. The final policy must retain the same prognostic information
as the fusion models and explicitly account for any staging redundancy. TX is
excluded. The current draft deliberately cannot pass the baseline fit gate.

C1 uses reference-grid PT/LN volumes (`log1p` in mm³), distances, voxel spread,
and laterality only where its anatomical midline and measurement validity are
qualified. It keeps known absence distinct from unavailable anatomy, with
missingness indicators and annotation states. Unqualified node counts and
largest-node placeholders are excluded. Source-FOV and derived-contour
limitations require explicit acceptance for development; they do not become
clinical measurement qualification.

Fit imputation, scaling and categorical encoding on training patients only.
Select ridge Cox penalties from [.01, .1, 1, 10, 100] using validation IBS on the
same frozen grid and training censoring reference. Use Efron ties; an exact
validation-score tie chooses the larger penalty. Do not refit on validation.
Held-out outcomes enter prediction export only after model selection and
checkpoint serialization. C0/C1 are deterministic fits, not three duplicated
seed experiments. See [baseline implementation](tabular_baselines.md).

## Subsequent stages

Only after an interpretable, adequately trained F2/F3 core should F0/F1/F4
explanatory comparisons identify the source of differences. F5 is conditional
on the final attention claim. A five-fold, 30-fit extension needs its own
resource-bound specification; it is not implied by three paired seeds.

Contour robustness follows the core study. Each perturbed contour must change
image-region masks, radiomics and geometry together. Compare clean training,
augmentation, and consistency conditions separately, with paired seeds and
matched evaluation perturbations. No disconnected mask-only intervention or
early consistency objective is included in this implementation.

The current blockers are endpoint semantics, clinical coding/staging, flagged
cohort review, pilot completion and a justified training budget. No baseline
fit, substantive F2/F3 experiment or clinical result is claimed here.
