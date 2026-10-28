# Prospective training-budget experiment — October 28, 2026

**Determine a common F2/F3 budget before launching C0/C1 or the six substantive
neural fits.** Pilot acceptance and computational cohort qualification remain
accepted. This proposal has performed no training and submits no job.

Start both modes at **12 epochs with seed 42**, sequentially on one H100. With
365 training patients, batch size 1 and gradient accumulation of 8, this gives
46 planned optimizer opportunities per epoch and **552 per mode**. Record
successful updates and AMP skips separately.

Twelve provides two initial epochs followed by two five-epoch observation
windows. It is an initial assessment budget, not a claim of adequate training.

## Prespecified adequacy rule

For each mode at common cap E, use full-precision validation OS NLL:

1. Previous window: epochs E−9 through E−5. Recent window: E−4 through E.
2. Improvement: median(previous) − median(recent).
3. Threshold: δ = max(0.002, 0.001 × abs(median(previous))).
4. Require an extension if improvement exceeds δ **or** the earliest global
   minimum among epochs 1..E lies in the final five epochs. Exact NLL ties use
   the earliest epoch; equality to δ alone does not require an extension.
5. Accept the common cap only when both modes finish with complete, finite,
   numerically qualified traces and measured epoch timing, and neither mode
   triggers an extension.

For E=12, compare epochs 3–7 with 8–12. All epochs, including the first two,
remain eligible for checkpoint selection. Report online training objectives and
epoch-end training NLL alongside validation NLL; diagnose optimization failure
or validation deterioration explicitly. This is operational validation-based
budget adequacy, not proof of optimizer convergence or clinical benefit.

If either mode requires more training, use a new paired attempt at twice the
common cap: initially **12→24 epochs**, or 1,104 opportunities per mode. Restart
both modes from the same fresh seed-42 initialization policy and preserve the
earlier attempt. The existing `CosineAnnealingLR` uses `T_max=epochs`, steps
after each survival-active epoch and has `eta_min=1e-6`; changing the cap changes
the whole trajectory. Do not resume an old attempt with a different T_max.

Each extension requires a prospective source/input/schedule record within the
existing authorized scope. There is no invented elapsed-time deadline or
overall epoch ceiling. Label the experiment compute-limited only when an actual
resource/request limit or explicit user budget prevents needed further work.

## Inputs and paired configuration

Use the existing 365 training and 77 validation assignments and the qualified
common-grid measurement policy. Prepare bound 442-row development views of the
metadata and radiomics. The 77 test patients remain withheld; their IDs establish
reserved membership and disjointness only. Do not use test outcomes, features,
metrics, model forwards or exports in budget selection. Fit encoders and time
bins using training patients only.

Use a new source and run, preserving the completed `614d7fb` pilot. Set the
prospective risk horizon to 730 days. Keep the same random-initialization,
architecture, objectives, optimizer settings and principal zero-dropout policy
for both modes. Starting learning rates remain backbone 0.001 and
head/clinical/radiomics 0.0001. Localization-only warmup, EMA and SWA remain off.

Select checkpoints by minimum validation OS NLL. `best.pt` retains the first
epoch on a tie; final export ties retain the existing last/EMA/SWA/best insertion
priority among enabled candidates. No held-out metric selects a checkpoint.

Omit the user-confirmed TNM-derived `NSTAGE` from both neural models and
baselines; also exclude TX. The v2 semantic encoder must enforce the actual
source-column whitelist, including derived aliases. The pre-change v2 branch
ignored `clinical_cols`, and the pre-change trainer unconditionally processed
test rows. The new whitelist and opt-in development-only path therefore need
source-bound qualification before launch.

## Pilot evidence and limits

| Evidence | F2 | F3 |
| --- | ---: | ---: |
| Online training OS NLL, epoch 1→2 | 3.744545→2.560448 | 3.557917→2.548273 |
| Validation OS NLL, epoch 1 | 2.5924600159580056 | rounded 2.5965; interval [2.59645,2.59655] |
| Validation OS NLL, epoch 2 | 2.5941849681367346 | 2.584356857536288 |
| Successful Adam updates per audited survival-head parameter | 92 | 92 |
| Whole-stage seconds including setup and evaluation | 617.795080 | 681.221964 |

F2 validation NLL worsened slightly; F3 improved and reached its minimum at the
boundary. Both online training losses decreased. Two observations cannot
establish an adequate cap. Online training loss averages changing batch models
and is not epoch-end training evaluation NLL. The 24 audited survival-head
parameters per mode demonstrate optimization; they do not demonstrate benefit.

Historical epoch timing, peak memory and explicit AMP-attempt/skip counters were
unmeasured. Whole-stage times are not measured epoch costs or an ETA. No held-out
metric is used to choose this proposal.

Prospective traces must include full-precision epoch-end training and validation
NLL, online losses, scoped epoch time, group learning rates and planned/attempted/
successful/skipped updates, including the last partial accumulation. Bind named
shared/OS head updates and finite losses/checkpoint/optimizer tensors. Any lost
optimization exposure needs an explicit numerical disposition before adequacy
is accepted. Report capacity only when actually measured with a stated scope.

## Result and evidence contract

Once the common rule passes, freeze the cap, corresponding cosine schedule and
complete study specification. Then run C0/C1 once each and all six fresh F2/F3
fits with paired seeds 42, 142 and 242. Development fits do not replace any
substantive fit.

The primary scientific metric is paired F3-minus-F2 IBS over **210–730 days**.
The latest user-defined minimum worthwhile effect is an absolute reduction of
**0.005** (ΔIBS≤−0.005). This is separate from the NLL stability threshold and
is not used to determine the training cap.

Local prospective artifacts are under
`/storage/TriFuseSurv/runs/study-20261028/budget-development-a/`:
`budget_development.proposed.json`, `budget_development.md`,
`pilot_development_evidence.json`, and `FILES.json`.
The proposal SHA-256 is
`89306388b7fb5d4d594c7e4202a60ff0258ca282627ad92dded6a528542dd7d5`.
Executable source `30dfdec`, the 442-row development views and the execution
manifest are now bound in the [dated launch record](budget_development_run.json).
See [the cohort qualification](cohort_qualification.md) and
[study protocol](study_protocol.md) for the retained scientific scope.

## First assessment attempt — October 28, 2026

Job `trifusesurv2-budget-development-20261028a`, UID
`d5ca7fee-5555-403a-9850-41fed98a82b9`, was submitted once and activated at
11:10 AM CDT after two suspended, zero-Pod binding reads. The bound resources
are one H100, 24 CPUs, 200 GiB RAM and 8 GiB memory-backed shared memory.
It reuses the pinned radiomics image, institutional PVC/subpath and existing
`radiomics-v1` interpreter without installation. The new output and lifecycle
roots are separate from the completed pilot and baseline candidate.

The controller passed 53 tests on actual Seadragon Python 3.6. These include
metadata from tiny fabricated F2/F3 producer runs whose full checkpoints were
accepted by the unchanged numerical auditor. Portable fixture tests deny reads
through original workstation paths. These checks do not establish actual H100
execution of the new source: focused same-image source tests, a tiny CUDA AMP
update for each fusion mode and shared-memory transport still run in the worker
before either real-data fit.

At **October 28, 2026, 11:11:06 AM CDT**, the Pod was Pending because the
cluster could not place the bound GPU/CPU/memory request. Zero H100s were
Running/Ready; the detached supervisor's exact process identity matched.
No worker or fit had started, and no queue ETA was established. Pending counts
against the one-H100 cap. The bound owner continues independently and captures
terminal evidence and exact cleanup. Do not duplicate the Pending attempt or
replay publication/launch commands to inspect its status.

The common substantive cap remains unset. Completion of this pair must produce
the frozen NLL-window assessment, finite/update exposure evidence and scoped
timings before the rule can accept the cap or require a fresh common extension.
The separate frozen C0/C1 candidate stays deferred; none of the six substantive
neural fits has run. The dated JSON binds source, plan, manifest, admission,
controller, host gate, launch and snapshot hashes. Original pilot receipts and
their separately reconciled controller failure remain intact.
