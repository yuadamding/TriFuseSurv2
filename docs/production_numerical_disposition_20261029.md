# Numerical disposition after the real-data diagnostic — October 29, 2026

**Select autocast-disabled FP32 for prospective production qualification and a
fresh paired budget experiment.** Preserve the existing TF32/backend settings.
Scale 16,384 was finite in the captured window, but the full gradient comparison
does not support treating it as a faithful correction. No production policy,
training cap or principal study is accepted by this decision alone.

The [machine-readable disposition](production_numerical_disposition_20261029.json)
and [new candidate policy](../configs/production_numerical_policy_v3_candidate.json)
bind the exact diagnostic, independent metadata audit and new source files.
The completed diagnostic report has SHA-256
`2d024ee5eba9aeda095a8fc11ebc512ef52618095ba54e57597abee1abb7db3d`.
No checkpoint or replay tensor was deserialized for this review.

## Confirmed event and matched replay

The actual first failure was update **27**, covering batches **209–216**.
The first nonfinite accumulated gradient appeared after batch **215** and
persisted through 216. Before clipping, the only nonfinite parameter was
`backbone.backbone_shared.swinViT.patch_embed.proj.bias`: one negative infinity
among 96 elements, with no NaN. Forward losses/logits remained finite.

The original epoch had 46 attempts, 45 successful optimizer calls and one skip.
There were 46 unscales, 46 scaler updates and zero scaler reloads. The event
reduced scale 32,768 to 16,384 and reset its growth tracker from 26 to zero.
The historical replay reproduced every recorded original-window comparison,
including input/state/RNG identities, loss/gradient summaries, scaler transitions
and actual optimizer behavior.

| Captured-window cell | Successful / attempted updates | Nonfinite preclip gradients | Result |
| --- | ---: | --- | --- |
| FP16, scale 32,768 | 0 / 1 | One negative infinity | Failed |
| FP16, scale 16,384 | 1 / 1 | None | Finite candidate |
| Autocast-disabled FP32 | 1 / 1 | None | Selected path to qualify |

Both successful cells had required gradients present, finite following survival
NLL and all 2,169 audited model/optimizer tensors finite. This is one matched
window, not a complete fresh trajectory.

## Why the lower scale is insufficient as the correction

| Gradient group | Scale-16k norm | FP32 norm | Relative L2 difference | Derived cosine |
| --- | ---: | ---: | ---: | ---: |
| All 528 observed tensors | 60.918257 | 24.539420 | 1.570077 | 0.946135 |
| Swin backbone, 326 tensors | 59.617406 | 21.098477 | 1.826129 | 0.999706 |
| Other backbone/localizer, 46 tensors | 4.131038 | 4.164018 | 0.031206 | 0.999541 |
| Fusion/projection, 132 tensors | 8.315417 | 8.312923 | 0.003984 | 0.999992 |
| Survival heads, 24 tensors | 8.401626 | 8.402005 | 0.001431 | 0.999999 |

Large early Swin weights contribute most of the discrepancy; it is not limited
to the overflowing bias or tiny gradients. The amplified backbone norm changes
global clipping. Dividing the recorded gradients by their respective recorded
global norms gives approximately 60% relative differences for survival/fusion
groups. This is a derived unit-normalization approximation, omitting the clipping
epsilon, rather than a second raw-vector replay.

Only 365 of 91,070,662 elements were zero in FP16 while nonzero in FP32, with
six reverse mismatches. Those sparse zero differences do not explain the norm
amplification. Close forward losses also do not establish close backward
gradients. The decision uses numerical evidence, without selecting on validation
NLL or adding a retrospective acceptance tolerance.

The FP32 comparison retained matmul TF32 enabled, cuDNN TF32 disabled and
float32 matmul precision `high`. The new policy preserves those settings. It
does **not** claim a strict IEEE-FP32 reference or identify the first faulty
autograd operation. It also makes no universal claim that every FP16 run is
invalid.

## Prospective correction and execution gates

The new policy explicitly uses `amp=False`, no initial scale and no GradScaler,
with `numerical_failure_policy=zero_skip_stop_v1`. Accumulation eight, max-norm
one clipping, architecture, objectives, AdamW groups, seed, input assignment and
the complete 12-epoch cosine schedule remain fixed. Historical C policy bytes,
failed acceptance and all diagnostic cells remain preserved.

The full production qualifier supports explicitly bound FP32 for ordinary
F2/F3 and the later routing comparison. It retains strict historical FP16
checks and rejects diagnostic precision overrides. Its focused checks and the
full source-stable fabricated CPU suite passed: **382 tests and 351 subtests**.
These checks do not substitute for the upcoming H100 gate.

The new paired Job must first qualify both full-production F2/F3 paths under
the exact FP32 policy: nine batches, accumulation eight, two successful updates
each including the final partial accumulation, zero numerical failures and
finite required gradients/state/following NLL. A failed gate starts no patient
fit. Then both fresh seed-42 fits must complete 12 epochs, `T_max=12` and 552
successful updates each, with zero disqualifying events and passing best/final
checkpoint audits. An F2 failure prevents F3 from starting.

Apply the unchanged validation-NLL curve rule only after numerical acceptance.
No extension trigger accepts the common 12-epoch cap; either trigger requires
a fresh common 24-epoch pair. Numerical failure does not select an extension.
The one-H100 cap includes Pending requests. Principal/baseline releases and
held-out scientific evaluation remain gated on the qualified paired assessment.
