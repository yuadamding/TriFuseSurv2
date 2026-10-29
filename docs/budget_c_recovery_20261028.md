# Budget-C recovery — October 28, 2026

**Budget B is now cleaned up; the common training cap remains unassessed.**
After the user confirmed that the worker had stopped or its node was safely
fenced, the exact read-only snapshot at **October 28, 2026, 7:35:33 PM CDT**
found no old Job and no UID-owned Pod in two API reads. It also verified that
the old supervisor was absent on its bound backend. An exclusive reconciliation
receipt was published at **7:36:32 PM CDT**, SHA-256
`fa5ede8cc999747d4edc674420bddfd244f3bb90a79de2c5e9c3d29f72e1816b`.
The earlier retained-resource receipt and `delete pods = no` authorization
answer remain historical evidence. This reconciliation performed no Kubernetes
mutation, force deletion or finalizer bypass. Project outputs and checkpoints
remain preserved.

The [budget-B assessment](budget_b_assessment_20261028.md) still describes the
scientific outcome accurately: F2 finished twelve epochs with **551 of 552**
successful updates and one AMP skip; its best and last checkpoint states passed
the finite-state audit. F3 received signal 15 after one completed epoch. Its
second-epoch log reaches the final batch, but the complete summary, save and
final successful-update count are unproven. Cleanup does not resolve that
exposure deficit or supply the missing paired trace. Twelve epochs are not
accepted, and no automatic extension to 24 epochs follows from this failure.
Neither budget B nor C has supplied a held-out test evaluation or
a scientific comparison.

The replacement **`trifusesurv2-budget-development-20261028c`** was submitted
against source **`1fed762a2457c49bbd334c5ac5f83c65dbdbcc16`** and plan SHA-256
`3fadfbda88ede8abf61063084d9d531d7344047d4786858205ba1643a17f90be`.
The detached activation handoff completed at **October 28, 2026, 7:39:19 PM CDT**.
The first worker snapshot at **7:40:08 PM CDT** found **one unscheduled Pending
Pod, zero Running/Ready H100s, zero restarts and zero real training invocations**.
Its `PodScheduled=False` reason was `Unschedulable`; the compact snapshot did not
capture a more specific scheduler message. The supervisor was alive with its
exact recorded identity. Pending occupies the one-H100 allowance. The worker's
hardware proof and numerical qualification await startup; there is insufficient
evidence for a completion ETA. The [machine-readable report](budget_c_recovery_20261028.json)
records the actual Job UID, package bindings and snapshot evidence.

Server admission preserved one H100, 24 CPUs, 200 GiB RAM, the approved pinned
image and PVC/subpath, nonroot execution, bounded temporary/shared memory and
zero retries. Both immutable bundles passed complete readback verification;
their temporary staging archives were removed. The actual Seadragon host
interpreter passed all **85 fabricated controller tests** with no failures.
No new environment or temporary work-mac relay file was created.

C binds the [candidate FP16 initial-scale policy of 32,768](../configs/production_numerical_policy_v2_candidate.json)
and the [new budget specification](../configs/budget_development_v2_candidate_spec.json).
The changed neural source passed **59 fabricated CPU tests and 91 subtests**;
this is separate from H100 qualification. Before either real-data fit, the
worker must pass the three affected CPU source suites, its shared-memory check,
and full production Swin/survival-NLL qualification with **two successful
updates and zero skips per mode**. Only then may it start fresh F2 followed by
fresh F3, each with **12 epochs, `T_max=12`, seed 42 and 552 successful updates**.
Neither historical fit is imported. The new scale remains a candidate until
these gates pass; F2's historical backoff alone does not prove a fix.

The attempt retains the one-H100 cap, including Pending allocations, zero
automatic retries and no agent-imposed execution deadline. Reserved test records
remain excluded from development training, evaluation and exports. Budget
acceptance requires complete numerical and update-exposure audits for both
modes, followed by the unchanged prespecified curve rule.

The [corrected baseline candidate](baseline_candidate_b_20261028.md) remains
prepared with 4,862 byte-verified development clinical cells and no C0/C1 fits.
The [mismatched-routing design](mismatched_routing_study.md) and development
feasibility checks are complete; runtime parity qualification and its three
fits remain pending. The [principal freeze](principal_freeze_checklist.md),
C0/C1 and all six substantive F2/F3 fits follow acceptance of the common cap.
No F3 benefit or clinically qualified result is established by this recovery.
