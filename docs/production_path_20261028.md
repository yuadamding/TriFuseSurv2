# Production path and budget-B handoff — October 28, 2026

**Full production-path qualification passed for both modes. Fresh F2 training
was active on one H100 at October 28, 2026, 1:44:29 PM CDT.** F3 follows in the
same sequential Job. The paired development assessment remains pending.

| Immediate milestone | Evidence and disposition |
| --- | --- |
| Retained CUDA diagnostic | All six cells collected; terminal cleanup verified. Original-scale FP16 failed with present but nonfinite gradients; FP32 and scale-one FP16 passed in each mode. |
| Baseline CSV ingestion | Fixed at the raw CSV boundary. Literal categories and nonblank whitespace survive; AGE reaches strict numeric validation. Eleven fabricated CPU tests and sixteen subtests passed, including an actual Cox fit. |
| Full CT/Swin H100 qualification | Both F2 and F3 passed under unchanged production FP16/default dynamic scaling. Two actual updates per mode, zero skips, full and final partial accumulation, finite required gradients/state/following NLL. |
| Fresh paired budget experiment | F2 started; F3 pending. Twelve epochs per mode, seed 42, 365 training and 77 validation patients. No completed epoch row was available in the handoff snapshot. |

## Qualified execution

The worker is pinned to source
`3298bf017c667ea44ae8ad621dc8cfaa95961aa6`, executable-tree SHA-256
`c40782445575be0e3338971c03f31764080d3ad77db60e5b9b6ec60ed7603806`.
Job `trifusesurv2-budget-development-20261028b` has UID
`8cc49447-6372-4d6b-8edf-199db3904cfd` in context
`yding4_yn-gpu-workload@kubernetes-admin@kubernetes`, namespace
`yn-gpu-workload`. The snapshot verified one Running/Ready H100 80GB and the
exact detached controller identity. There were no container restarts or
reported worker errors in the available handoff evidence.

Both immutable bundles passed readback. All 70 controller tests passed locally
and on Seadragon's actual host Python 3.6 interpreter. The exact server dry run,
real suspended Job, two zero-Pod reads and activation were bound separately.
The Job reserves 24 CPUs, 200 GiB RAM and one H100; Pending reservations count
against the one-H100 limit. The existing approved image, PVC/subpath and
radiomics interpreter were reused. There is no automatic workload retry or
whole-job deadline. Verified temporary staging archives were removed; streaming
transfer created no work-Mac relay files.

The qualifier exercised the full `96×160×96` canvas, Swin feature size 96 and
depths `[2,2,18,2]`, actual trainer construction, survival NLL, parameter groups
and update helper. Each mode processed nine fabricated batches with
accumulation eight, producing updates after batches eight and nine. The raw
criterion-level report SHA-256 is
`115ee00881cccdba83735fcb27a96f015f8dacbd935a68fe42a5bea744491665`.
Its collected bytes also passed an independent replay of the strict metadata
consumer, without tensor loading or another scheduler query.

The initial scaler was 65,536, with growth factor 2, backoff 0.5 and growth
interval 2,000. Both modes retained the actual configured backend observations:
deterministic algorithms enabled, cuDNN deterministic enabled, cuDNN TF32 and
benchmark disabled, float32 matmul precision `high`, and CUDA matmul TF32
enabled. This last value was measured after trainer configuration; it was not
inferred from the deterministic flag. The original diagnostic's token loss and
backend defaults remain a separate experiment.

The [portable qualification result](production_qualification_result.json) and
[bound run snapshot](budget_development_b_run.json) retain source, policy,
admission and artifact hashes. The bound policy file retains its prospective
prequalification status; the separate result records acceptance without changing
the active Job's source or policy bindings. The full qualifier uses fabricated inputs, an
invented small clinical vocabulary and six time bins. Its 307,518,048 parameters
describe that fixture, not the real cohort models. Real-cohort numerical health,
checkpoint dimensions and update exposure remain subject to the training audits.

## Completion and scientific interpretation

The Job trains F2, audits its checkpoints, then trains and audits F3. It applies
the unchanged [common-budget rule](budget_development.md) only after both traces
are complete. At twelve epochs, compare validation-NLL medians from epochs 3–7
and 8–12, with `delta = max(0.002, 0.001 * abs(previous_median))`. Either a larger
improvement or an earliest global minimum in epochs 8–12 requires a fresh common
24-epoch pair with its complete `T_max=24` schedule. Skipped or lost optimization
exposure needs an explicit numerical disposition. The current snapshot provides
no measured epoch throughput or defensible completion ETA.

After a common cap is accepted, freeze the principal study, replace the old
baseline package with one bound to the corrected reader, fit C0/C1 and complete
all six fresh substantive neural fits. Development runs do not substitute for
those fits. The 77 test patients remain reserved; no test scoring is part of
this budget attempt. Primary mean paired F3-minus-F2 IBS over 210–730 days and
the 0.005 worthwhile-reduction threshold are unchanged. No F3 benefit or adequate
training cap is established by this handoff.

The [explanatory plan](explanatory_study_plan.md) records the later fusion ladder
before principal test review. Mismatched routing and contour robustness still
require complete versioned designs; later choices must be labeled exploratory
if made after that review. Original CT/RTSTRUCT/HU provenance remains unavailable,
so engineering qualification does not establish anatomical or clinical validity.

Local preparation also caught three operational issues before submission: an
empty-list descriptor constructor, a stale cleanup container selector, and a
missing AST-only fixture source. Each was corrected and its failed evidence
retained before the final 70-test gates. Earlier pilot, budget-A and diagnostic-A
sources, receipts and outputs remain preserved.
