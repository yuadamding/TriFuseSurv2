# Two-epoch F2/F3 pilot completion

The unchanged recovery pilot completed both modes on **October 28, 2026,
12:50 AM CDT**, using one verified NVIDIA H100 80GB HBM3. The retained
controller captured successful terminal Job/Pod evidence and removed both;
two absence reads confirm cleanup. The separate stronger checkpoint audit
was submitted and is **Pending cluster resources** as of October 28, 2026,
1:27 AM CDT. Its supervisor is healthy; its worker has not started.

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

The postprocessor reconstructs the exact frozen model and optimizer grouping
without a forward pass or encoder fitting. It checks named shared and OS
survival-head updates, required finite losses/weights/risks, exact test IDs,
paired contracts, checkpoint hashes and validation-NLL selection.

The historical epoch CSV omits validation NLL, but the completed per-mode logs
retain four-decimal NLL values. Source-bound log parsing represents each value
as a closed rounding interval of plus/minus 0.00005 and requires the exact
checkpoint NLL inside its corresponding interval. Strictly separated
intervals can prove ordering; overlapping intervals remain unresolved. The
postprocessor still has to verify the actual checkpoint values.

The first postprocessing preparation requested four CPUs, 32 GiB and zero GPUs.
The server dry run on October 28, 2026, 1:15 AM CDT rewrote both CPU and memory
requests/limits to zero. That attempt was rejected before publication or real
Job creation. Its local bundle and rejection evidence are preserved.

The replacement uses the existing institutional single-H100 template:
24 CPUs, 200 GiB and one H100 reservation. Its October 28, 2026, 1:19 AM CDT
server dry run preserved those resources. Actual checkpoint calculations are
explicitly CPU-only (`CUDA_VISIBLE_DEVICES=-1`); the worker will verify the
reserved hardware separately with `nvidia-smi`. This adds no model training, inference or
scientific GPU-compute claim. The prior pilot is cleaned before admission,
so the one-H100 concurrent cap is retained.

The separate Job is `trifusesurv2-pilot-postprocess-20261028b`, UID
`04ba9835-e257-43a7-8a55-50d8be5f32c4`. The actual suspended Job matched the
reviewed admission profile and was bound before activation. The detached
supervisor's process identity matched on its recorded backend at the snapshot.
One H100 is requested and counts against the cap; zero are Running/Ready.
The exact Pod reports insufficient eligible GPU/CPU/memory capacity. There is
no defensible queue ETA and no duplicate attempt or scheduler substitution.

Operational preparation and lifecycle receipts are under the approved
`TriFuseSurv2/common-grid-h100-20261027` root and the local
`runs/study-20261028` evidence directory. The controller will preserve terminal
Job/Pod/log evidence, validate the report and exact-clean the owned objects.
There is no whole-job deadline or automatic retry. A process exit code cannot
replace the criterion-level acceptance report.

Publication readback and 30 fabricated controller tests passed on the actual
host Python 3.6.8. Seven fabricated worker tests passed locally. These establish
software/lifecycle checks, not completed real-checkpoint acceptance. The
unmodified scientific worker source, checkpoints and outputs remain in place.

## Scientific interpretation

Training loss decreased in both modes across two epochs. This does not show
convergence, an adequate epoch budget, or F3 superiority. The current split
remains development data. All six prespecified paired-seed substantive fits
are still required after protocol, cohort and budget gates pass.

The cohort report accounts for all 613 prepared patients, including 519 with
valid OS pairs and 94 with both primary outcomes missing. The user confirmed
days and event 1 as death. OS origin/censoring, clinical coding and the flagged
cohort review remain unresolved; the prospective study specification remains
draft. See the [study protocol](study_protocol.md) and
[cohort qualification](cohort_qualification.md).
