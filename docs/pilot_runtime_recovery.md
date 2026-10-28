# Paired pilot runtime recovery

At **October 27, 2026, 11:38 PM CDT**, corrected attempt B was Pending for
cluster placement resources. Its detached supervisor was healthy; its H100
worker and training processes had not started. This is a dated observation,
not live status. The [portable recovery ledger](pilot_runtime_recovery.json)
binds the evidence and preserves the remaining limitations.

The original attempt completed preparation of **613 cases**. The builder
excluded 94 paired-missing primary outcomes and fixed the eligible cohort at
**519 patients: 365 training, 77 validation, and 77 test**. F2 then failed while
reading its first DataLoader batch, before any optimizer update; F3 never
started. Its terminal evidence and exact cleanup were retained.

Three operational corrections were qualified:

- The supervisor's expired-watch recovery now uses fresh raw API collection
  resource versions. Its queue and JSON decoder are bounded, retired streams
  release subprocess/thread/file resources, and consecutive transport errors
  stop recovery without retrying the workload. This addresses the repeated
  HTTP 410 loop that preceded the original `MemoryError`.
- The DataLoader bus error indicated insufficient shared memory. The new Pod
  supplies an **8 GiB memory-backed `/dev/shm`**, within the existing 200 GiB
  memory contract. A zero-GPU check in the approved image measured that mount
  and retained 360 MiB of tensors transferred through two DataLoader workers.
  The old Pod's exact shared-memory capacity was not measured before cleanup.
- The status reader accepts bounded receipts up to 8 MiB and reports oversized
  receipts without aborting the complete status snapshot. Successful preparation
  receipts had exceeded its previous 2 MiB limit.

B reuses the same prepared files, split, seed, model capacity, image, interpreter
and training settings. CPU preparation verified **8,623 distinct bound files**.
Only the F2/F3 output directories changed in the retained training plan; no image
extraction, split search or checkpoint resume was repeated. Actual training will
rehash its inputs before each mode and repeat the shared-memory guard on its
assigned H100. The active source remains
`614d7fbdb3551f58c2e8b5da403f7b6f5418f23e`; later repository changes do not alter it.

Validation passed 34 local controller/continuation checks, 31 applicable checks
under the actual host Python, and the same-image CPU preparation/transport gate.
The current status adapter passed 12 checks. Zero-GPU qualification does not prove
H100 execution, completed epochs, optimizer updates or scientific performance.
Acceptance still requires both two-epoch runs, finite losses/risks/checkpoints,
positive optimizer-step evidence, matching parameter counts and exact cleanup.

## Render plan identities together

Review also found a retained metadata discrepancy: B's Job and Pod-template
`trifusesurv2_plan_sha256` annotation still names A's plan. The mounted B plan,
worker `TRIFUSE_PLAN_SHA256`, immutable live binding and static gate correctly
agree on B's plan. Treat those as the authority for this frozen attempt. The
annotation and active execution contract were left unchanged; the discrepancy
is recorded explicitly in the recovery ledger.

For a **new, unpublished** bundle, use
[bind_h100_job_plan.py](../scripts/bind_h100_job_plan.py) to bind both annotation
locations and worker environment together. It checks the run identity, image,
worker executable and effective argv, supporting Kubernetes' split `command`
and `args`. It never contacts Kubernetes or overwrites an existing output.

```bash
python scripts/bind_h100_job_plan.py \
  --plan /path/to/new/PLAN.json --job /path/to/draft/JOB.json \
  --output /path/to/new/JOB.json
python scripts/bind_h100_job_plan.py \
  --plan /path/to/new/PLAN.json --job /path/to/new/JOB.json
```

Five focused regressions cover stale annotations despite a correct environment,
input preservation, ambiguous environment variables, split argv and overwrite
refusal. This helper is an identity preflight; it does not replace immutable
publication, image/runtime qualification, admission audit or launch authorization.
Do not apply a newly rendered manifest to an active attempt.

The institution-specific controller sources, exact process identities, original
failures, cleanup receipts and current pointer remain in the separate run archive
`runs/common-grid-h100-20261027/PAIRED_RECOVERY_REPORT.md`. The earlier
[implementation guide](implementation_and_validation.md) and
[qualification ledger](development_qualification_summary.json) retain their
original observation boundaries. Neither this recovery nor a future two-epoch
pilot pass establishes convergence, clinical qualification or F3 benefit.
