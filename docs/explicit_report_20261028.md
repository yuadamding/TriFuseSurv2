# TriFuseSurv2 report — October 28, 2026

The current development-budget attempt failed during synthetic CUDA
qualification before either real-data fit began. At the fresh **October 28,
2026, 11:23:49 AM CDT** snapshot, its Job and Pods were absent after verified
cleanup. Zero project GPUs remain allocated by this attempt. The earlier
11:11 AM Pending snapshot is historical and has been superseded.

## Current attempt and exact failure

Job `trifusesurv2-budget-development-20261028a`, UID
`d5ca7fee-5555-403a-9850-41fed98a82b9`, used frozen source `30dfdec` and the
approved pinned radiomics image. It requested one H100, 24 CPUs, 200 GiB RAM
and 8 GiB shared memory. The institutional home PVC and existing radiomics
environment were readable; no environment installation was performed.

The worker started at 11:18:55 AM CDT and verified exactly one NVIDIA H100
80GB HBM3 with compute capability 9.0. All 37 focused source tests passed in
the approved container: 8 development-training tests, 17 training-contract
tests, 4 survival-export tests and 8 clinical-whitelist tests. These suites
used fabricated CPU fixtures inside the GPU worker.

The subsequent tiny CUDA AMP fixture stopped with
`ValueError: Nonfinite tiny CUDA gradient`. Its qualification subprocess ran
92.57 seconds, including those source suites. The worker failure receipt was
written at 11:20:36 AM CDT. This fixture uses fabricated token inputs and a
squared-logit loss; it does not exercise real CT/Swin training or the clinical
survival NLL objective.

The failure occurred after a finite loss, scaled backward and optimizer
unscale. The guard rejects either an empty gradient list or any nonfinite
gradient. The log does not identify the mode, affected parameters or scaler
state, so the exact mechanism and whether an earlier synthetic mode completed
are unresolved. An AMP scaling issue is a diagnostic possibility, not an
established cause. No real-data training command was invoked.

There was one failed attempt, zero container restarts and no automatic retry.
The supervisor captured terminal evidence and completed exact UID-bound
cleanup with two Job/Pod absence reads. Its terminal receipt was written at
11:20:41 AM CDT. Original outputs, logs, bindings and failure receipts remain
intact. The [terminal evidence summary](budget_development_failure.json) binds
the fresh snapshot and preserved logs.

## Completed work and remaining execution

| Stage | Verified result |
| --- | --- |
| Cohort preparation | 613 prepared; 519 OS-valid; 94 paired-missing outcome exclusions |
| Frozen assignment | 365 training, 77 validation, 77 test patients |
| Computational cohort review | All 613 patients and 2,452 compartments reviewed; all 525 queued cases have computational dispositions |
| Earlier two-epoch pilot | F2 and F3 completed on one verified H100; all 18 required engineering criteria passed after separate metadata reconciliation |
| New source qualification | 37 source tests passed in the container; tiny CUDA gradient gate failed |
| New 12-epoch development pair | 0 of 2 real-data fits started or completed |
| C0/C1 baseline results | 0 of 2 selected baseline models completed; package is published and host-qualified, execution deferred |
| Substantive neural comparison | 0 of 6 fresh F2/F3 fits completed |
| Scientific paired evaluation | No substantive IBS, IPCW concordance, calibration or paired confidence interval result available |

The earlier pilot's exact best validation OS NLL was:

| Pilot mode | Best epoch | Validation NLL |
| --- | ---: | ---: |
| F2 | 1 | 2.5924600159580056 |
| F3 | 2 | 2.584356857536288 |

Both pilot modes had 307,566,600 trainable parameters. Each of 24 audited
survival-head parameters received 92 successful Adam updates per mode.
These establish execution and checkpoint integrity. Two epochs and these
validation losses do not establish an adequate budget or F3 superiority.
The original postprocess controller's source-identity comparison failure was
preserved; a separate metadata reconciliation accepted all 18 required
criteria without rerunning training.

Source `30dfdec` passed 260 synthetic tests and 176 subtests locally. The new
controller passed 53 tests on actual Seadragon host Python. The later
supporting-horizon Brier update passed 30 focused evaluator/baseline tests and
16 subtests, including unchanged primary output. These counts overlap and
must not be summed into a unique-test total. Repository `main` includes the
evaluator and documentation updates; the failed run remains bound to `30dfdec`.

## Approved study definitions

OS starts at diagnosis, is measured in days, and event 1 means death from any
cause. Event 0 uses the delivered OS.TIME as fixed censoring follow-up; the
system clock does not extend it. AGE is continuous, approved coded clinical
fields are categorical, T/N/M remain included, and their redundant NSTAGE plus
TX are omitted. The implementation enforces this source-column policy.

The primary scientific comparison is the mean paired F3-minus-F2 IBS over
210–730 days across seeds 42, 142 and 242. The minimum worthwhile absolute IBS
reduction is **0.005**, corresponding to delta IBS at or below **-0.005**.
A paired interval wholly below zero supports improvement; an upper bound at
or below -0.005 supports at least the declared worthwhile reduction. The
planned uncertainty procedure uses 2,000 paired patient bootstrap draws.
The key secondary concordance uses tau 730 days and risk 1-S(730); supporting
Brier and calibration horizons are 365, 730 and 1,095 days.

The separate budget experiment uses only 365 training and 77 validation
patients, seed 42 and an initial 12 epochs for both modes. Test IDs are retained
only for reserved-membership checks. No test outcomes, image/radiomics payloads,
model evaluation or exports inform budget selection.

The cap is not established. The prespecified assessment compares previous
and recent five-epoch validation-NLL medians, with threshold
max(0.002, 0.001 times the previous median). Continued improvement above this
threshold or a global validation minimum in the final five epochs of either
mode requires a fresh common extension for both modes. This NLL rule is
separate from the 0.005 scientific IBS effect. The full principal study remains
draft until the common cap and corresponding schedule are qualified.

## Remaining work and limits

First diagnose the qualification fixture with explicit mode, gradient
names/counts, dtypes, scaler state and actual step/skip evidence. Matched
fabricated FP32 and FP16 checks can distinguish missing gradients, backward
instability and scaling overflow. Preserve this failed attempt and use a fresh
bound attempt after a justified correction; do not relax finite-gradient
requirements or infer a clinical-training fix from this log.

Then complete the paired budget assessment, extend both modes if its rule
requires it, freeze the full study, run C0/C1 and all six fresh principal neural
fits, and execute paired evaluation with uncertainty and calibration. Later
fusion/routing ablations and contour-robustness experiments remain unstarted.
No defensible completion ETA is available before numerical qualification and
measured training throughput.

The local inputs remain development-only. Original CT/RTSTRUCT sources and HU
provenance are unavailable; delivered CT is already normalized/cropped and
contours are derived. Computational review does not certify anatomical or
clinical validity. Source-boundary truncation, unavailable compartments and
sampling/extraction failures remain explicit. No clinical performance,
external validation or prognostic superiority claim is supported yet.
