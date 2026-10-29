# TriFuseSurv2 explicit report — October 29, 2026

**Both budget-C development fits completed, but the paired experiment failed
numerical acceptance.** F2 completed 551 of 552 attempted optimizer updates;
F3 completed all 552. F2's one nonfinite-gradient update and AMP skip violate
the bound zero-skip requirement. The common training cap remains unaccepted.
The H100 Job and its owned Pod have been cleaned up. No baseline, principal
six-fit comparison, mismatch cohort fit or held-out scientific evaluation has
been completed.

This report uses the exact **October 29, 2026, 8:58:12 AM CDT** snapshot and
hash-verified terminal metadata. The worker finished on **October 28, 2026,
at 11:47:05 PM CDT**. It supersedes the current-state descriptions in the
[October 28 explicit report](explicit_report_20261028.md),
[partial C report](budget_c_qualification_20261028.md),
[mismatch implementation report](mismatched_routing_implementation.md) and
[release preparation report](study_release_preparation_20261028.md). Those
dated records retain their historical observations and original bindings.
The [machine-readable report](explicit_report_20261029.json) contains exact
metrics, source identities and evidence hashes.

## Cohort and approved prediction inputs

| Item | Current definition or count |
| --- | --- |
| Prepared cases | 613 |
| OS-eligible patients | 519 |
| Excluded for both OS fields missing | 94 |
| Frozen training / validation / test assignment | 365 / 77 / 77 |
| OS origin and unit | Diagnosis; days |
| Event 1 | Death from any cause |
| Event 0 | Right-censored at the supplied, unchanged OS.TIME |
| Numeric clinical predictor | AGE |
| Categorical clinical predictors | PATHOLOGY, SEX, RACE, T, N, M, KFCF, SMOKE, ALCOHOL, HPV |
| Excluded clinical columns | NSTAGE, because derived from T/N/M; TX |

The treatment table previously supplied describes the 519 eligible patients.
TX does not enter the current neural models or configured C0/C1 baselines.
Clinical categories retain their literal source values; unverified category
meanings or a staging edition are not inferred. The user's endpoint and
encoding decisions remain sufficient for the approved development scope;
no additional codebook or ascertainment approval gate is introduced here.
Delivered censoring durations do not increase with the system clock.

Computational cohort review covers all 613 prepared cases and 2,452
compartments. Its accepted use is computational development. Original
CT/RTSTRUCT paths and qualified HU provenance remain unavailable. Delivered
CT is already normalized/cropped and contours are derived. Source-field-of-view
truncation, unavailable compartments and sampling/extraction limitations remain
explicit. Computational checks do not certify anatomical or clinical validity.

## What has executed

| Stage | Verified result |
| --- | --- |
| Input preparation and computational review | Complete for the retained development inventory |
| Earlier two-epoch pilot | F2 and F3 completed on one verified H100; 18 required engineering criteria accepted after separate metadata reconciliation; cleanup complete |
| Budget A | Tiny fabricated CUDA gradient gate failed before any real-data fit; 37 source tests passed; cleanup complete |
| Six-cell numerical diagnostic | Completed and cleaned; retained token fixture showed scale-dependent backward instability |
| Budget B | F2 completed 12 epochs with one skip; F3 interrupted after one complete epoch; cleanup subsequently verified after administrator fencing confirmation |
| Budget C production fixture | Both full production Swin/survival-NLL modes passed two actual updates, including final partial accumulation, with zero skips |
| Budget C real-data pair | Both 12-epoch fits and both checkpoint-audit stages completed; paired assessment failed its numerical gate; cleanup complete |
| C0/C1 selected baseline models | 0 of 2 completed |
| Fresh principal F2/F3 fits | 0 of 6 completed |
| Mismatch cohort fits | 0 of 3 completed; prospective implementation is CPU engineering qualified |
| Held-out primary/secondary evaluation | Not run |

The diagnostic's FP32 and scale-1 FP16 cells passed; the original scale-65,536
FP16 token fixture failed. That result concerns the retained fabricated fixture,
not the production CT/Swin survival objective. Budget C subsequently passed the
actual production fixture at initial scale 32,768. Its real-cohort F2 skip shows
that a short fabricated pass does not establish sustained numerical acceptance.

Budget B's F3 subprocess terminated with signal 15. The retained evidence does
not establish the initiating cause or completed epoch-2 finalization. The later
node taint must not be presented as a proven cause of that termination.

## Budget-C configuration and results

The exact run is `trifusesurv2-budget-development-20261028c`, Job UID
`e8c1468d-1b19-4f60-b922-7bf2a1541587`. It used archived scientific source
`1fed762a2457c49bbd334c5ac5f83c65dbdbcc16`, one verified H100 80GB, the approved
pinned radiomics image and institutional home PVC. Requested resources were
24 CPUs, 200 GiB RAM and 8 GiB shared memory. The existing worker environment
was reused. There was no automatic retry, container restart or active deadline.

Both fresh fits used seed 42, 12 epochs, batch size 1, accumulation 8 and
`CosineAnnealingLR(T_max=12, eta_min=1e-6)`. Both had 307,566,600 trainable
parameters. The intended numerical policy was FP16 AMP with initial scale
32,768. The full production Swin configuration used input size 96×160×96,
feature size 96 and depths 2/2/18/2. The principal zero-dropout policy, approved
clinical encoding and paired inputs were retained.

| Quantity | F2 | F3 |
| --- | ---: | ---: |
| Completed epochs | 12 | 12 |
| Attempted optimizer updates | 552 | 552 |
| Successful optimizer updates | 551 | 552 |
| AMP-skipped updates | 1 | 0 |
| Nonfinite-gradient updates | 1, in epoch 1 | 0 |
| Best validation NLL | 2.520520329620544 | 2.5002673904278447 |
| Selected best epoch | 6 | 6 |
| Final validation NLL, epoch 12 | 2.7504580977868724 | 2.6750605452515477 |
| Final epoch-end training NLL | 0.7764404119821648 | 0.925733907603697 |
| Final AMP scale | 16,384 | 32,768 |
| Trainer subprocess wall time | 2,900.628 seconds | 2,845.401 seconds |
| Median measured train/evaluation epoch time | 199.922 seconds | 195.828 seconds |

The subprocess times include setup, checkpoints and development finalization.
Measured epoch times have the narrower scope of training plus epoch-end
training/validation evaluation. They are not interchangeable GPU-runtime
measurements or a completion estimate for the remaining study.

F3's lower seed-42 development validation NLL is descriptive. It is not the
primary scientific metric, is not held-out performance and does not establish
F3 superiority. The pair also has unequal realized optimization exposure.

## Why the assessment failed

The frozen curve rule compares median validation NLL in epochs 3–7 and 8–12.
It requires an extension if previous minus recent median exceeds
`max(0.002, 0.001 × abs(previous median))`, or if the earliest global minimum
lies in epochs 8–12.

| Curve-rule quantity | F2 | F3 |
| --- | ---: | ---: |
| Previous median, epochs 3–7 | 2.5680259200562903 | 2.562114275459732 |
| Recent median, epochs 8–12 | 2.740267673616189 | 2.6736717104621524 |
| Previous minus recent | -0.1722417535598987 | -0.1115574350024204 |
| Required improvement threshold | 0.0025680259200562903 | 0.002562114275459732 |
| Minimum in final five epochs | No | No |
| Curve-based extension trigger | No | No |

Validation deteriorated over these windows while training NLL decreased.
Neither curve triggers the declared extension, but acceptance also requires
complete, numerically qualified exposure. F2's skipped update violates that
separate requirement. The assessment therefore returned
`numerical_review_required`; `assess-budget` exited 2 and the worker recorded
`failed_budget_development`. Training itself and the checkpoint-audit stages
returned zero. These are different stage outcomes.

The established numerical observations are one nonfinite-gradient update,
one AMP skip and the recorded scale backoff in F2. The collected metadata does
not identify the offending parameters or establish the upstream numerical
mechanism. Finite saved tensors do not erase the lost update. This result
accepts neither the 12-epoch cap nor an automatic 24-epoch extension, and it
does not authorize treating one skip as acceptable after observing it.

## Checkpoint integrity and its limits

The worker audited `best.pt` and `last.pt` for each mode. All four audits found
finite values in 585 model tensors and 1,584 optimizer tensors over 528
optimizer states. Parameter identity and optimizer-group mappings matched the
reconstructed 307,566,600-parameter model. The audits observed positive named
Adam states for 326 shared-backbone parameters, six OS residual-head parameters
and six shared survival-head parameters. Best-checkpoint steps were 275 for
F2 and 276 for F3; final steps were 551 and 552 respectively.

The checkpoint auditor reconstructed without fitting encoders or performing a
model forward. This is saved-state and named-update evidence, not an independent
predictive replay. The report preparation verified metadata hashes and their
references; it did not load checkpoint tensors locally or on a login host.

The supervisor captured the terminal result and completed cleanup at
11:47:10 PM CDT on October 28. Its receipt records two exact Job/owned-Pod
absence reads. The October 29 snapshot independently found the exact Job
absent and zero owned Pods. Budget C retains **zero H100 reservations**. The
snapshot landed on a different login backend, so it did not freshly establish
absence of the old supervisor process on its bound backend. This does not
change the verified Kubernetes cleanup result.

## Implemented software and outstanding qualification

Implementation revision `989cf455861b5336075ff27522c82c85ad45ce61` adds the
availability-cycle mismatch intervention, immutable routing identities,
checkpoint/inference checks and actual per-patient routing-coverage exports.
Its full fabricated CPU suite passed **358 tests and 337 subtests**. Historical
inactive F3 behavior matched bit for bit in 18 scenarios covering both model
classes, three seeds, train/eval and dropout conditions, and all 256 availability
patterns. These counts describe their stated scopes and are not added to
overlapping earlier test totals.

The intervention's tiny fabricated Swin/survival-NLL comparison executed both
actual branches with two successful CPU updates each and zero skips. Full
production H100 qualification of matched F3 versus the intervention remains
pending. Budget C's older source-bound qualification does not qualify this new
source or its new branch. C was never patched with these changes.

The corrected baseline preparation preserves 4,862 development clinical cells
and a 16-wheel dependency lock. Its execution guard remains in place. The
[principal candidate](../configs/principal_release_candidate.json) and
[baseline candidate](../configs/baseline_execution_release_candidate.json)
remain prepared, unfrozen and unreleased. C supplies terminal and cleanup
evidence, but its numerical failure cannot fill their accepted-policy, cap or
qualified paired-assessment fields. Frozen candidate bytes are not rebound by
this report.

## Scientific result still required

The principal analysis remains the mean paired F3-minus-F2 IBS over **210–730
days**, across fresh fits at seeds **42, 142 and 242**. Negative values favor
F3. The minimum worthwhile absolute reduction is **0.005**, corresponding to
deltaIBS at or below -0.005. The planned interval uses 2,000 shared paired
patient-bootstrap draws, seed 20261028, requiring at least 95% valid draws.
It is conditional on the fitted models and training censoring reference;
retraining uncertainty is not included.

Report the point estimate and interval separately: an interval wholly below
zero supports improvement, while an upper bound at or below -0.005 supports
at least the declared worthwhile reduction. IPCW concordance at 730 days is
key secondary; supported Brier scores and descriptive calibration at 365, 730
and 1,095 days remain supporting analyses.

No held-out IBS, concordance, calibration, paired interval, clinical benefit
or external-validation result is available. The 77 test patients were excluded
from budget fitting, evaluation and exports. Their reserved IDs were retained
only for membership/disjointness checks. Completed development weights do not
replace any of the six fresh principal fits.

## Remaining execution sequence

1. Resolve the numerical exposure finding and bind any justified prospective
   numerical correction to exact source, settings and qualification evidence.
   Preserve C's failed acceptance and original policy. A changed policy needs
   its own qualification and fresh paired budget evidence under that policy.
2. Accept a common cap only after a numerically qualified pair satisfies the
   declared curve rule. Bind the complete cosine schedule and final principal
   protocol; preserve all failed attempts.
3. Generate fresh baseline/principal execution releases with accepted artifact
   bindings, exact predecessor cleanup and new admission/controller/Job
   identities. Qualify the locked baseline environment and literal reader with
   the actual worker interpreter; a status-field edit cannot release the
   guarded preparation.
4. Run C0 clinical and C1 clinical-plus-reference-geometry baselines, with five
   penalty candidates per mode and validation-only selection. Run all six fresh
   paired neural fits. Preserve selected checkpoint, contract and complete
   survival-curve exports.
5. Execute the frozen paired evaluation and report every seed, uncertainty,
   support failures and calibration. Then qualify the new mismatch source on
   production H100 and run its three fits under the specified later design.
   Further fusion and contour-robustness stages remain subsequent work.

All execution stays within the existing one-Kubernetes-H100 cap, including
Pending requests. No new Job, numerical retry, policy change or held-out
evaluation was performed during this report preparation. An overall completion
ETA remains unsupported while numerical disposition and the accepted training
budget are unresolved.
