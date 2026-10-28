# Two-epoch F2/F3 pilot completion

The unchanged recovery pilot completed both modes on **October 28, 2026,
12:50 AM CDT**, using one verified NVIDIA H100 80GB HBM3. The retained
controller captured successful terminal Job/Pod evidence and removed both;
two absence reads confirm cleanup. The separate checkpoint worker completed
on **October 28, 2026, 1:32 AM CDT**, and its audit reports **all 18 required
criteria passed**. The controller subsequently rejected a source-identity
schema comparison. A separate corrected metadata validator passed against
the original files on **October 28, 2026, 8:54 AM CDT**. The pilot now has a
passed engineering acceptance report and reconciled receipts. The original
failed controller receipts remain unchanged as historical evidence.

The [machine-readable evidence](pilot_completion_evidence.json) records exact
source, identity and receipt hashes. Earlier [recovery observations](pilot_runtime_recovery.md)
and [study preparation evidence](study_preparation_evidence.json) remain dated
history. No old blocked acceptance report has been relabeled.

## Execution and retained evidence

| Item | F2 | F3 |
| --- | --- | --- |
| Completed epochs | 2 | 2 |
| Training process exit | 0 | 0 |
| Whole training-stage duration | 617.795 seconds | 681.222 seconds |
| Final export candidate | best | last |
| Selected checkpoint epoch | 1 | 2 |
| Exact selected validation NLL | 2.5924600159580056 | 2.584356857536288 |
| Audited survival-head parameters | 24 | 24 |
| Successful Adam updates per audited parameter | 92 | 92 |
| Rounded validation NLL, epoch 1 | 2.5925 | 2.5965 |
| Rounded validation NLL, epoch 2 | 2.5942 | 2.5844 |

Stage durations include initialization, final evaluation and artifact checking
after process exit. Dividing them by
two does not yield measured epoch throughput. The frozen trainer did not save
individual epoch timing or peak RAM/VRAM measurements. Prospective code now
records scoped epoch time and full-precision validation NLL.

Each mode retained `best.pt` and `last.pt`; every file is 1,986,454,173 bytes.
Full checkpoints remain in the approved remote project. Local metadata copies
contain hashes and receipts, not checkpoint payloads. The shared-memory mount
was 8 GiB, and the pretraining canary verified 377,487,360 bytes of shared
tensors, then removed them. This is capacity evidence, not a peak-use estimate.

Scientific execution remained bound to `614d7fbdb3551f58c2e8b5da403f7b6f5418f23e`.
Source `71d7c16112744cbf092d56c98c3ad7838ff995e9` adds the separately pinned auditor
and optimizer identity reconstruction; it does not change the completed run.

## Stronger acceptance and scheduling

The postprocessor reconstructed the exact frozen model and optimizer grouping
without a forward pass or encoder fitting. Both modes have 307,566,600
trainable parameters; each reconstruction mapped 561 named parameters across
eight optimizer groups, with 528 serialized optimizer states. Every one of the
24 audited survival-head parameters, including the shared and OS residual
paths, has 92 successful Adam updates and nonzero gradient moments. This does
not assert that every registered model parameter was updated.

The planned 365 training patients, batch size one and eight-batch accumulation
provide 46 update opportunities per epoch and 92 in total. Successful updates
are measured from optimizer state. Attempted/skipped AMP-update counters were
not retained separately.

All 18 required worker criteria passed: completed epochs, named survival-path
optimization, finite required losses/model tensors/risks, exact 77-patient test
inventories, shared scientific settings and source/input identities, equal
parameter counts, artifact hashes and validation-NLL checkpoint selection.
The optional runtime-capacity criterion remains unassessable for missing
epoch/peak-memory telemetry. No peak memory or longer-training cost estimate
is inferred from the passed engineering criteria.

The historical epoch CSV omits validation NLL, but the completed per-mode logs
retain four-decimal NLL values. Source-bound log parsing represents each value
as a closed rounding interval of plus/minus 0.00005 and requires the exact
checkpoint NLL inside its corresponding interval. For F2, the exact epoch-one
NLL is 2.5924600159580056, below epoch two's 2.5941849681367346, so `best` is
selected. For F3, the exact epoch-two NLL is 2.584356857536288, below the
epoch-one interval's lower bound of 2.59645. Both F3 checkpoint candidates
represent epoch two with the same NLL, so the prescribed final-export tie rule
selects `last`. F3's exact epoch-one NLL remains unknown; the interval proof
does not reconstruct it.

The first postprocessing preparation requested four CPUs, 32 GiB and zero GPUs.
The server dry run on October 28, 2026, 1:15 AM CDT rewrote both CPU and memory
requests/limits to zero. That attempt was rejected before publication or real
Job creation. Its local bundle and rejection evidence are preserved.

The replacement uses the existing institutional single-H100 template:
24 CPUs, 200 GiB and one H100 reservation. Its October 28, 2026, 1:19 AM CDT
server dry run preserved those resources. The completed worker performed
checkpoint calculations on CPU (`CUDA_VISIBLE_DEVICES=-1`, zero Torch CUDA
devices) and independently observed one NVIDIA H100 80GB HBM3 through
`nvidia-smi`. It performed zero model forward passes and zero training
invocations. GPU reservation is distinct from GPU computation. The prior pilot
was cleaned before admission, retaining the one-H100 concurrent cap.

The separate Job is `trifusesurv2-pilot-postprocess-20261028b`, UID
`04ba9835-e257-43a7-8a55-50d8be5f32c4`. The actual suspended Job matched the
reviewed admission profile and was bound before activation. At the retained
October 28, 2026, 1:27 AM CDT snapshot it was Pending eligible resources; that
observation is preserved as history. It later ran once, and the terminal Job
and Pod record successful worker exit. The controller completed exact cleanup
with two absence reads despite its application-validation failure. The
October 28, 2026, **8:38:45 AM CDT** status snapshot confirms the Job absent and
zero owned Pods.

The controller error was `Optimizer mapping does not bind the frozen
checkpoint/source.` Its validator compared the complete checkpoint source
identity against the plan's two-field identity. Actual map/proof identities
agree on the frozen commit and source-tree hash; they additionally retain
`source_dirty: false` and `declared_commit_sha: null`. Independent local
readback verified all 17 collected artifact hashes and the FINAL/report,
map/proof, source, parent and cleanup links. The separate validator requires
exact commit/tree agreement, the known clean four-field checkpoint identity,
and exact equality between map and reconstruction-proof identities. Six
regression checks cover actual artifacts, dirty or inconsistent identities,
tampered bytes, unsafe paths and weakened cleanup evidence.

The corrected validator passed against the original remote metadata under the
qualified host Python. Its separate receipt is
`pilot-postprocess-b-reconciliation-a/REMOTE_RECONCILIATION.json`, SHA-256
`5427b7a2dd71b5240fea7af71e3de2d9d831e2e117d6806f59678934df060a11`.
All 18 required criteria are reconciled. This correction performed no
checkpoint loading, scheduler action, training or model forward pass. The
original controller's failed receipts retain their original statuses.

Operational preparation and lifecycle receipts are under the approved
`TriFuseSurv2/common-grid-h100-20261027` root and the local
`runs/study-20261028` evidence directory. Terminal Job/Pod, stage log references,
the worker report, the failed controller check and cleanup receipts are retained.
There is no whole-job deadline or automatic retry. A process exit code cannot
replace the criterion-level acceptance report.

Publication readback and 30 fabricated controller tests passed on the actual
host Python 3.6.8. Seven fabricated worker tests passed locally. These establish
software/lifecycle checks. The subsequent real-checkpoint worker audit is
recorded separately above; the corrected metadata reconciliation closes the
receipt-validation issue. The
unmodified scientific worker source, checkpoints and outputs remain in place.

## Scientific interpretation

Training loss decreased in both modes across two epochs. This does not show
convergence, an adequate epoch budget, or F3 superiority. The current split
remains development data. All six prespecified paired-seed substantive fits
are still required after protocol, cohort and budget gates pass.

The cohort report accounts for all 613 prepared patients, including 519 with
valid OS pairs and 94 with both primary outcomes missing. The user confirmed
diagnosis as OS origin, days, death coded as event 1, and retaining the delivered
CSV censoring described as “till now”; no rolling-clock recomputation is implied.
The user approved AGE continuous and coded fields categorical with TX
excluded, then confirmed NSTAGE is derived from T/N/M. The prospective
baseline retains T/N/M and omits only redundant NSTAGE. The computational
development review is complete with all 525 queued cases accounted for. A
justified common neural training budget remains to be established. See the [study protocol](study_protocol.md) and
[cohort qualification](cohort_qualification.md).
