# Production-gradient diagnostic and prospective numerical recovery

The completed [budget-C report](explicit_report_20261029.md) retains its
`numerical_review_required` disposition. The next GPU submission diagnoses its
real-data F2 update path. A second, conditional submission combines production
qualification and a fresh F2/F3 budget pair under a justified prospective
policy. The candidate cap stays 12 epochs with the complete `T_max=12` schedule.
No retrospective skip tolerance or six-epoch cap is introduced.

## Retained C traces and inferred event location

The complete, unmodified aggregate traces are now available:

| Mode | CSV | SHA-256 |
| --- | --- | --- |
| F2 | [12-epoch trace](results/budget_c_F2_complete_metrics.csv) | `f674327dec0bab9942b22d7ca040856372c85b7b9d83d55be5a97cc7d4e701c8` |
| F3 | [12-epoch trace](results/budget_c_F3_complete_metrics.csv) | `e9769718f74a9be9500c53977264564c7a9933965f0037d54025644742a4d465` |

Their validation medians and attempted/successful/skipped/nonfinite-gradient
totals independently match the terminal assessment. They contain aggregate
epoch metrics, without individual patient identifiers or prediction payloads.
The earlier two-row F2 CSV remains historical partial evidence.

C's F2 best checkpoint records 276 cumulative attempts and scaler growth
tracker 249; its last records 552 attempts and tracker 525. Both differences
are 27. The [PyTorch 2.9.1 scaler kernel](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/cuda/AmpKernels.cu)
resets the tracker after a nonfinite event and increments it after success.
With growth interval 2,000, no reset/reload and one update per optimizer
opportunity, this points to attempted update 27, or batches 209–216 at
accumulation eight. It is an inference to verify, not a recorded batch-level
failure or an identified patient/parameter.

The archived trainer uses one scaler for the fresh fit, with resume disabled.
The diagnostic additionally records the installed scaler implementation and
actual update/unscale behavior. Its observation order follows
[PyTorch's accumulation and clipping guidance](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html):
retain the scale during accumulation, unscale once at the completed boundary,
observe before clipping and distinguish actual optimizer execution from an
attempted step.

## First submission: archived production computation plus observations

The entry point is `scripts/diagnose_production_gradients.py`. It loads C's
archived source and exact plan rather than replacing the archived trainer with
the prospective source. It retains F2, seed 42, initialization, original
training order and transformed inputs, model/objective, optimizer, initial
FP16 scale 32,768, batch size one, accumulation eight, clipping, backend flags
and complete 12-epoch scheduler configuration.

The implementation passed 379 fabricated CPU tests and 340 subtests with
unchanged source hashes throughout the final run. The 11 focused diagnostic
tests also passed when imported against archived C source. They cover ordinary
training parity, full/partial divisors, matched-state restoration, preclip
observations, norm overflow, expected failures and exact PyTorch-build checks.
The portable lifecycle passed 49 tests. These are engineering checks; the
production diagnostic and any candidate-policy acceptance require separate
H100 evidence.

The observed segment is bounded by work: at most the original first training
epoch, 365 microbatches and 46 update attempts. It does not change the configured
cap to one epoch. It exits before validation and later epochs. Test IDs remain
reserved membership checks; no held-out payload, outcome scoring or export is
part of the diagnostic.

Observations include mode/seed/epoch/batch/update location; actual accumulated
batch count and original loss divisor; forward loss/logit dtype and finiteness;
scaled gradients during accumulation; unscaled gradients before clipping;
missing required gradients separately from inactive parameters; NaN/Inf names
and counts; finite magnitudes and an independent float64 norm; scaler state;
and actual optimizer calls. Instrumentation observes detached values and must
leave the ordinary computation and RNG progression unchanged.

Before each update window, keep a rolling CPU model/optimizer/scaler/RNG state
and capture its actual transformed batches. Retain the inferred window 27 and,
if different, the first observed failing window. Select the first actual failing
window for replay, or window 27 if no failure appears. The diagnostic performs
exactly three cells for that one selected window:

| Cell | Precision / initial replay scale |
| --- | --- |
| Historical-policy replay | FP16 / 32,768 |
| Evidence-motivated candidate | FP16 / 16,384 |
| Reference comparison | Autocast disabled, FP32 |

Each cell restores identical pre-window model/optimizer state, RNG and saved
transformed inputs. Only the declared precision/scaler factor changes. Report
preclip gradients, actual execution, resulting state, post-update survival NLL
and gradient differences against FP32. Record actual TF32/backend settings;
autocast-disabled FP32 with TF32 enabled is not a strict IEEE-FP32 reference.
Raw replay payloads remain inside the approved project; portable evidence
contains hashes and numerical summaries.

A failed historical-policy cell does not abort the other cells. Diagnostic
completion, policy-cell success and production qualification are separate
statuses. If the location or failure does not reproduce, record that bounded
negative result and its uncertainty. Never turn the inferred location into an
observed fact. A cell that cannot execute is not a passing numerical comparison.

## Numerical decision and prospective stopping behavior

The following table records the original conditional diagnostic plan. The later
[full numerical disposition](production_numerical_disposition_20261029.md)
supersedes its recommendation to advance scale 16,384: that cell was finite but
had substantial gradient-magnitude and global-clipping differences from FP32.
Autocast-disabled FP32, with no GradScaler and the recorded TF32 settings, was
selected for the [fresh budget-D attempt](budget_d_fp32_launch_20261029.md).
Production qualification and paired numerical acceptance remain pending at
that report's dated launch snapshot. The original table and historical
diagnostic receipts remain preserved below.

| Matched result | Disposition |
| --- | --- |
| Historical FP16 fails; lower-scale FP16 and FP32 pass | Supports a scale-sensitive finding for the captured state; advance 16,384 to prospective qualification |
| Both FP16 scales fail; FP32 passes | Investigate the observed backward region before binding a specified precision correction |
| FP32 fails | Investigate objective, operations, inputs and optimizer state before choosing a correction |
| Historical failure is not reproduced | Preserve non-reproduction; any candidate decision must state the remaining uncertainty and receive full prospective qualification |

The numerical decision does not use validation NLL. Scale 16,384 is initially
a candidate: starting at it executes an update C skipped and changes the later
trajectory. BF16 requires deliberate trainer, contract and qualifier changes;
it is not an implicit fallback or settings-only replacement.

Prospective budget/principal execution uses the opt-in immutable
`--numerical_failure_policy zero_skip_stop_v1`. The default retains historical
behavior. On the first nonfinite forward, missing required gradient,
nonfinite gradient element, nonfinite clipping norm or actual optimizer skip,
the active policy publishes an exclusive numerical-failure receipt and raises
to terminate the attempt. A preclip stop records that no optimizer opportunity
was executed; it does not fabricate a skip count. An observed scaler skip
records the actual attempt/call/skip evidence. The sequential worker must not
start F3 after an F2 numerical failure.

## Second submission and study release

After a justified policy decision, publish new exact source/settings/image and
numerical-policy bindings. Preserve C's policy and receipts. The prospective
source includes the implemented mismatch code with its intervention disabled
for ordinary F2/F3.

One new H100 submission runs full production F2/F3 qualification first. Require
the full canvas/backbone, actual trainer/objective/parameter grouping, nine
batches with accumulation eight, two successful updates per mode including the
last partial accumulation, zero skips, finite required gradients/state and a
finite following NLL. Only a passing gate may start either patient fit.

Then run fresh F2 and F3 at seed 42 on the same 365/77 development assignment,
12 epochs and `T_max=12`, with one numerical policy. The first epoch is the early
real-data observation within each complete run. Acceptance requires 552
attempts, 552 successes and zero disqualifying events per mode, complete
full-precision validation traces/timing, selection and best/final audits.

Apply the unchanged curve rule only after numerical acceptance: neither
extension trigger accepts 12; either trigger requires a fresh common 24-epoch
pair with `T_max=24`. Numerical failure retains its own disposition.

The existing principal and baseline candidates retain their fixed cohort,
clinical policy, reference measurements, 0.005 IBS effect and bootstrap design.
Accepted source/policy/qualification/budget/cap/schedule fields remain unresolved
until actual evidence exists. The corrected baseline reader, 4,862 verified
development cells and locked dependencies remain reusable. Execution requires
a fresh guarded-worker release and actual-worker environment/reader checks.

After acceptance and final freeze, execute C0/C1 and all six principal fits,
then the frozen paired evaluation. The mismatch production qualifier gates its
three subsequent fits and does not delay ordinary principal release. All Jobs
share the existing one-H100 allowance, including Pending reservations. No
elapsed-time execution window is imposed; the diagnostic bound is the declared
epoch/update work inventory.
