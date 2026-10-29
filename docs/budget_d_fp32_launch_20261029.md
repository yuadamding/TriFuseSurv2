# FP32 production qualification and budget-D launch — October 29, 2026

**At October 29, 2026, 11:38:25 AM CDT, budget D had one Running/Ready H100
with zero container restarts and its exact supervisor alive on ldragon3.**
The source-qualification stage had started. Production qualification had no
completion receipt; startup reported zero patient-fit invocations and the
snapshot contained no fit progress. No common cap, numerical policy or
scientific result is accepted at this boundary.

The [machine-readable record](budget_d_fp32_launch_20261029.json) binds the
exact publication, launch and read-only status receipts. The snapshot was
non-atomic and is a dated observation. It is not a fresh query when this file
is read. All times here use `America/Chicago`; original UTC evidence is retained.

## Exact attempt and launch evidence

| Item | Recorded value |
| --- | --- |
| Job | `trifusesurv2-budget-development-20261029d` |
| Job UID | `3e44b2b8-6e59-4006-a3f1-6ecf7e2d38c6` |
| Pod | `trifusesurv2-budget-development-20261029d-6gmbs` |
| Pod UID | `25a5ef84-6321-464d-a260-0540027bb436` |
| Launch handoff | October 29, 2026, 11:37:37 AM CDT |
| Source qualification start | October 29, 2026, 11:37:50 AM CDT |
| Snapshot | October 29, 2026, 11:38:25 AM CDT |
| Actual device | One NVIDIA H100 80GB HBM3; compute capability 9.0; 85,028,372,480 bytes |
| Worker backend | PyTorch `2.9.1+cu128`, CUDA build `12.8` |
| Supervisor | Exact bound identity alive on ldragon3; PID 2573568 |
| Execution source | `cdf7bd25ef82179f6c491286f1eaae7e8c2d0160` |
| Executable source tree SHA-256 | `2d4715f019e6548e75ef517270fe8f0491e8a0a60b6715210edf7c1d25d164c4` |
| Plan SHA-256 | `dd3b82a4ddfd5132e1913198c1813f43079b8da5abf60a96be30e299f322e55b` |

The Job was bound while suspended, with two zero-Pod reads before activation.
The publication receipt verifies both immutable bundle readbacks and removal
of their staging archives. The existing radiomics environment and approved
PVC were reused; no environment installation or work-mac relay file was added.
The development-input verification covered 4,420 files for 442 patients and
709,056,957 bytes, excluding test payloads.

## Selected policy and required sequence

The [full diagnostic disposition](production_numerical_disposition_20261029.md)
selects [policy `production_fp32_zero_skip_stop_v3`](../configs/production_numerical_policy_v3_candidate.json).
The matched real-data replay reproduced update 27's scale-32,768 overflow.
Scale 16,384 was finite but retained material gradient-magnitude and clipping
differences. The new attempt therefore disables autocast and GradScaler.
It retains the recorded backend configuration: matmul TF32 enabled, cuDNN
TF32 disabled and float32 matmul precision `high`, under the same configured
deterministic path. The new production qualifier must verify those effective
settings. This is not a strict IEEE-FP32 claim or proof of universal FP16 failure.

One H100 Job performs the following sequence:

1. Run the three current affected source suites and shared-memory checks in
   the existing worker interpreter, then qualify both full production
   CT/Swin/survival-NLL modes. Each mode must demonstrate two actual successful
   updates from nine fabricated batches, including the final partial
   accumulation, finite gradients/state and following survival NLL.
2. Only after both modes pass qualification, start fresh F2 and F3 fits with
   seed 42, 12 epochs each and cosine `T_max=12`. Neither historical fit is
   reused. Each mode must complete all **552 actual successful updates**, with
   zero skips or nonfinite events, before numerical acceptance.
3. Audit the actual saved checkpoints and apply the unchanged paired
   validation-NLL budget rule. No cap is accepted from the launch or fixture
   evidence alone.

The active `zero_skip_stop_v1` policy writes an exclusive failure receipt and
terminates at the first disqualifying forward, required-gradient, gradient
element, clipping-norm or actual-update event. An F2 numerical failure prevents
F3 from starting. There is no automatic retry or precision fallback. The bound
one-H100 cap includes Pending requests. No whole-job elapsed-time ceiling was
added; the declared work and acceptance gates define the attempt.

## Preparation evidence and its limits

| Check | Passed evidence | Scope |
| --- | --- | --- |
| Full source regression | 382 tests and 351 subtests | Fabricated local CPU engineering |
| Actual tiny producer and checkpoint-auditor CLI | Both modes: 12 epochs, 24 successful updates, zero skips, no scaler | Tiny fabricated CPU inputs; these are not 552 production updates |
| Local controller qualification | 88 tests | Fabricated metadata/lifecycle checks |
| Actual Seadragon host controller gate | 88 tests, zero failures | Python 3.6 host checks before launch; no model/GPU work by those tests |

Failed producer-fixture attempts a/b/c and the earlier failed local final-gate
log remain preserved separately from the passing preparation. Their failures
were not rewritten into passing evidence. Historical budget B, numerically
ineligible budget C and the completed production-gradient diagnostic retain
their original sources, policies, results and cleanup receipts.

The [budget-C report](explicit_report_20261029.md) remains historical evidence,
including F2's 551 successful updates plus one skip. D's policy decision and
launch do not retroactively qualify C. At the D snapshot, source qualification
was incomplete, full H100 production qualification was pending and both
patient fits were unstarted. No ETA is supported yet.

C0/C1 baseline fits, six principal neural fits, mismatch H100 qualification,
held-out scoring and the primary scientific comparison remain deferred. The
primary paired F3-minus-F2 IBS over 210–730 days and the user-selected minimum
worthwhile absolute reduction of **0.005** are unchanged; that threshold does
not select numerical policy or training budget.
