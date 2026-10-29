# Budget-C qualification and numerical observation — October 28, 2026

**The new 32,768 policy passed the fabricated H100 qualification, but real F2
training recorded an AMP skip. The common cap remains unresolved.**

The exact scheduler snapshot at **October 28, 2026, 10:16:53 PM CDT** verified
one Running/Ready H100, zero restarts and the exact live detached supervisor.
The worker started at 10:08:30 PM CDT and began F2 at 10:09:59 PM CDT. This
supersedes the Pending snapshot in the [recovery report](budget_c_recovery_20261028.md).
The bound scientific source remains `1fed762`; prospective mismatch changes
are separate from the published worker archive.

Both F2/F3 full Swin survival-NLL checks passed with the actual trainer-default
FP16 scale **32,768**, nine fabricated batches and accumulation eight. Each
made two actual successful optimizer updates, including the final one-batch
accumulation, with zero skips. Required gradients, model/optimizer state and
following survival NLL were finite. The worker's three affected source suites
and shared-memory transport checks also passed.

Eight completed qualification artifacts/logs were collected with exact hash
verification. An independent metadata consumer accepted the original report;
it loaded no checkpoint/image tensor or patient outcome. Effective CUDA matmul
TF32 was true, cuDNN TF32 false, float32 matmul precision `high`, deterministic
algorithms and cuDNN deterministic true, and cuDNN benchmark false. These
fabricated updates supply no real-cohort optimization exposure.

The later read-only numerical packet at **10:20:33 PM CDT** contains two
complete F2 epoch rows:

| Epoch | Epoch-end training NLL | Validation NLL | Attempts | Successful | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.5503730687172443 | 2.644569021056999 | 46 | 45 | 1 |
| 2 | 2.4779909956332755 | 2.6174644984595186 | 46 | 46 | 0 |

The bound real-model contract explicitly records initial scale 32,768. The
[partial trace](results/budget_c_F2_partial_metrics.csv) has **91 successes,
92 attempts and one skip/nonfinite-gradient update**. The declared C acceptance
requires 552 successes and zero skips per mode. This F2 trajectory cannot meet
that zero-skip requirement by merely finishing its remaining planned epochs.
The observation does not establish corrupt checkpoint state or the cause of
the nonfinite gradient; best/last audits are still pending. F3 had not been
observed starting at the scheduler snapshot. Counts are observations of complete
epoch rows, not a claim about unobserved later or partial work.

The paired disposition is therefore **unresolved**. The complete paired curve
rule has not been applied, twelve epochs are not accepted, and a numerical skip
does not authorize an automatic 24-epoch extension or another retry. No skip
tolerance or active numerical setting has been changed. Preserve the existing
attempt and its sequential workflow so final traces/audits retain their exact
source and policy identities; any later numerical recovery needs its own
cause-specific disposition and qualification.

The [machine-readable report](budget_c_qualification_20261028.json) binds the
qualification, worker gate, independent acceptance and partial observation.
These observations are not an atomic snapshot. Held-out test scoring and the
principal/baseline release remain deferred. No F3 benefit or clinical result
is established.
