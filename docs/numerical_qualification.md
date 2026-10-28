# Clinical typing correction and retained CUDA diagnosis

The review of source `48eeb47` identified an independent prospective clinical
encoding mismatch. The source-column whitelist excluded NSTAGE/TX, while the
legacy schema still treated several approved coded fields as numeric/ordinal.
The explicit [clinical type policy](clinical_type_policy.md) corrects this for
new study runs. AGE is numeric; the other ten approved fields are categorical.
T/N/M have no additional ordinal or derived raw-stage channels. The study and
budget configurations select `--clinical_type_policy approved_categorical_v1`.
The default legacy API, completed pilot and saved encoders retain their own
encoding. Corrected token dimensions require a fresh source-bound attempt.

The numerical failure is separate. Budget attempt A used frozen source
`30dfdec` and failed inside its operational `qualify_worker.py`, before any
real-data training command. The exact retained fixture SHA-256 is
`61cc6794a50ccb330427746d4c751b8535abe82772168d73a578f0c757b57e8a`;
its plan SHA-256 is
`11ed467e1aae357f5c7491df54473c34916f0d19f89a01c74ba00f860315505d`.
The different pre-existing `scripts/qualify_h100.py` is not substituted for it.
Original failure, source, controller and cleanup evidence remain unchanged.

## Matched diagnostic matrix

`scripts/diagnose_retained_cuda_fixture.py` and its standalone helper extract
the setup, optimizer, scaler and forward/loss statements from those exact
fixture bytes. Scientific imports resolve to the retained original source;
the prospective clinical correction does not enter this reproduction.
The new instrumentation has a separate source/file identity.

Each mode, F2 and F3, has these three cells:

| Cell | Forward | Scaling |
| --- | --- | --- |
| original_fp16 | Exact original FP16 autocast | Exact original GradScaler defaults |
| fp32 | Autocast disabled | No scaler |
| fp16_scale1 | FP16 autocast | Initial scale one |

Each cell starts from the same retained model weights, inputs, masks and RNG
state within its mode. INITIAL.json records their independent hashes and
INITIAL.pt retains the fabricated tensors/state. The original token model has
no lazy parameters; an unexpected lazy state is captured as an error rather
than silently changing the historical initialization. Backend and optimizer
settings remain fixed and recorded. Each cell has at most one optimizer
opportunity; there is no automatic scale retry or clinical training.

Durable phase records precede each numerical guard. They retain output/loss
dtypes, shapes, finite counts/ranges; scaled loss and scaler settings; named
gradients before and after unscale; missing, NaN and infinite gradients;
finite-only magnitudes/norms; declared shared/OS paths; actual optimizer
invocations, Adam counters/moments, finite updated state and post-update loss.
Backward-node observations retain early nonfinite observations without changing
gradients; observation order alone does not establish a root cause. Unscale
runs once after the intended backward and before any gradient modification.

The original guard remains enforced. A cell stopped by it has no optimizer
attempt or scaler-backoff claim. Expected numerical failures remain results
of the diagnostic; unexpected exceptions or incomplete/changed evidence fail
the diagnostic lifecycle. A complete six-cell matrix explicitly records
`qualification_passed: false`. It cannot promote the failed attempt to success
or establish a clinical training budget.

The PyTorch documentation describes how gradient scaling and update skipping
operate. These provide hypotheses for the comparison, not a diagnosis of this
failure: [AMP reference](https://docs.pytorch.org/docs/2.9/amp.html) and
[AMP examples](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html).

## Intended training-path qualification

After reviewing the matrix, any numerical correction must be explicit and
justified. `scripts/qualify_swin_survival.py` provides a separate fabricated
engineering check through the actual trainer construction, CT/Swin backbone,
survival NLL, optimizer groups and update helper for both modes. Its production
profile requests the full 96×160×96 input, feature size 96, depths 2/2/18/2,
and nine fabricated batches with accumulation eight. This exercises one full
boundary and the final partial boundary. It uses the prospective categorical
policy and requires actual successful updates, finite active shared/OS/Swin
gradients and optimizer/model state, and finite following outputs/NLL.

The script's explicit precision/scaler alternatives are diagnostics. Their
existence does not change the production training policy. The H100 production
profile must run under the adopted, source-bound settings before budget
training resumes. Its smaller CPU profile verifies script plumbing only.

## Execution boundary

The fresh six-cell diagnostic uses one approved Kubernetes H100, the pinned
radiomics image, existing interpreter and institutional project PVC. New
bundle/output/lifecycle identities preserve the failed attempt. It reads no
clinical cohort inputs and starts no baseline or real-data neural fit.
Actual hardware, source, cell identities and terminal cleanup are separately
bound. Local fabricated tests and mocked receipt producers do not substitute
for the H100 matrix.

The accepted pilot audit, OS definitions, computational cohort review,
210–730-day primary IBS interval and 0.005 worthwhile effect remain in force.
The NLL budget rule remains unchanged: start with a common 12-epoch pair;
apply its median-window/boundary criterion; use a fresh common 24-epoch pair
with the corresponding cosine schedule if required. The numerical failure
provides no evidence to change the epoch cap. Freeze the main study only after
qualification and budget determination, then execute C0/C1 and all six fresh
principal neural fits and the paired scientific evaluation.

## Dated execution handoff — October 28, 2026

Instrumentation source `21d5e03` is published separately from the retained
scientific source `30dfdec`. The full synthetic CPU suite passed 302 tests and
211 subtests. Nine affected qualifier checks passed after the final
type-descriptor/settings-guard review. The controller passed 32 tests locally
against the rendered manifest and 32 on actual Seadragon host Python 3.6.
These are engineering results, separate from the numerical matrix.

Job `trifusesurv2-amp-diagnostic-20261028a`, UID
`31b9958f-7ab9-4510-a86b-f2ded8eac51a`, was submitted once after immutable
publication and admission review. At **12:35:43 PM CDT**, the Pod was
Running/Ready with one GPU request, no restarts and a matched detached owner.
A separate **12:36:39 PM CDT** guarded startup read verified one H100 80GB and
the exact current Pod UID. Source tests and the six-cell numerical result
remain unobserved at this handoff. The controller continues independently and
owns terminal capture, metadata validation and exact cleanup.

The [dated JSON](clinical_cuda_followup_run.json) binds the plan, manifest,
controller, admitted Pod spec, publication, host gate, launch and startup
receipts. Existing radiomics-v1 is reused without installation; verified staging
archives are removed and no temporary work-mac relay files were created.
The next scientific steps remain the numerical disposition, actual CT/Swin
survival-NLL qualification and a fresh common-budget pair. No numerical-policy
change or budget adequacy is claimed from this handoff.
