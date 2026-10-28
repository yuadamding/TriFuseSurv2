# Retained CUDA diagnostic and production policy — October 28, 2026

The six-cell diagnostic completed and its Job was cleaned up on **October 28,
2026, at 12:36:58 PM CDT**. Both original FP16 cells produced nonfinite backward
gradients; FP32 and FP16 with initial scale 1 each completed an actual update in
both fusion modes. No real-data neural fit was invoked. This is completed
diagnostic evidence, not production numerical qualification or a scientific
result. The [machine-readable result](retained_cuda_diagnostic_result.json)
preserves the individual cell hashes and observations.

## Exact execution and retained evidence

Job `trifusesurv2-amp-diagnostic-20261028a` had UID
`31b9958f-7ab9-4510-a86b-f2ded8eac51a`. It used diagnostic software commit
`21d5e037aa305cefe4c6f1767c61478330df169d` against the original model source
`30dfdec6c285ec5d813f0e6dc1ebc47cbf4ebf1d`. The retained qualifier SHA-256 is
`61cc6794a50ccb330427746d4c751b8535abe82772168d73a578f0c757b57e8a`.
Original initialization, inputs, masks and RNG state were preserved and their
digests matched across the prescribed cells. Tensor snapshots remain retained;
the collected evidence consists of metadata and verified inventories.

The raw `DIAGNOSTIC.json` SHA-256 is
`9181159f05e3d78952b67d319d0889e796f19eb23359f60a0930c2389f06166d`.
The metadata collection receipt SHA-256 is
`eb587680d029d1e872e530a2738af04ff5ea8f9e8e3b9559497e5fc8695240a1`.
The controller completed, and cleanup verified absence twice; cleanup receipt
SHA-256 is
`2b17f0c9ab067213c73d399b90edce34e15e081ffdd1d1b6e6a87a39b569e05b`.
These records preserve the earlier budget-A failure without rewriting it.

## What the six cells establish

Each cell used the retained token fixture's squared-logit objective. Every
forward loss was finite and all 144 parameter gradients were present.

| Mode | Precision / initial scale | Nonfinite gradient tensors | Successful updates | Result |
| --- | --- | ---: | ---: | --- |
| F2 | FP16 / 65,536 | 131 | 0 | Numerical failure |
| F2 | FP32 / no scaler | 0 | 1 | Numerical pass |
| F2 | FP16 / 1 | 0 | 1 | Numerical pass |
| F3 | FP16 / 65,536 | 129 | 0 | Numerical failure |
| F3 | FP32 / no scaler | 0 | 1 | Numerical pass |
| F3 | FP16 / 1 | 0 | 1 | Numerical pass |

Within each mode the two FP16 cells had identical forward losses. The original
scale cells failed before an optimizer call: there were zero attempted updates,
zero AMP skips and no observed scaler backoff or retry. All four passing cells
had finite updated parameters, optimizer state and post-update loss, with named
shared/OS head Adam counters corroborating the update. There were no unexpected
script errors.

This isolates scale-dependent nonfinite backward gradients in this fixture.
The first observed nonfinite autograd nodes were `AddmmBackward0` for F2 and
`ToCopyBackward0` for F3; observation order does not establish the causal origin
or identify a faulty production operation. The fixture does not establish that
the full CT/Swin survival-NLL training path fails at its default scale.

## Selected production policy and pending qualification

At **October 28, 2026, 1:17:32 PM CDT**, the
[production numerical policy](../configs/production_numerical_policy.json)
selected `production_fp16_dynamic_default_v1`: retain CUDA FP16 and the existing
dynamic GradScaler defaults (initial scale 65,536, growth factor 2, backoff 0.5,
growth interval 2,000). There is no scale override and no change to production
numerical behavior. The policy SHA-256 is
`e0d01361ddcaa14b4176dae9ab07be8c67d6d2c446e1e5220dd01887802b9033`.

**Full production-path qualification remains pending at this report's snapshot.**
The fresh budget-B worker must first exercise the actual CT/Swin survival NLL
with the approved categorical clinical policy, full production shape and
unchanged trainer AMP/update helpers. Each mode must process nine fabricated
batches with accumulation eight, completing both the full accumulation and the
final partial accumulation: two actual successful updates, zero skips, finite
required gradients and state, and finite post-update survival NLL. Effective
backend flags must be observed after the actual trainer configuration; they
must not be inferred from the retained diagnostic's different backend defaults.
A failure preserves evidence and prevents real fits, without automatic retries
or an unrecorded precision change.

## Budget B and scientific limits

Only after both full-path checks pass may the separate budget-B attempt train
F2 and F3 sequentially on one H100, with seed 42 and an initial common cap of
12 epochs. The bound development views contain 365 training and 77 validation
patients. The remaining 77 test IDs establish reservation/disjointness only;
their features, outcomes, forwards and metrics are excluded. Batch size 1 and
accumulation 8 imply 46 planned optimizer opportunities per epoch, or 552 per
mode, with actual successes and skips audited separately.

The [budget protocol](budget_development.md) and
[amended specification](../configs/budget_development_spec.json) retain the
prespecified validation-NLL adequacy rule and paired fresh extension to 24
epochs when required. The amended specification SHA-256 is
`05540501623ee0df695138a01fde9573fa23953c98f5583ab67e943715c60a28`.
No adequate cap, completed budget-B training or F3 benefit is claimed here.
C0/C1 and the six substantive neural fits remain behind budget determination
and final study gates. Primary paired F3-minus-F2 IBS over 210–730 days retains
the user-defined worthwhile reduction of 0.005.

The accompanying baseline ingestion correction preserves declared clinical
fields before CSV inference: coded values retain their literal categories,
blank AGE remains eligible for training-median imputation, and nonblank invalid
AGE values such as `NA` and `NaN` are rejected. Endpoint parsing is unchanged.
This software correction is separate from real-data baseline qualification;
see [baseline documentation](tabular_baselines.md).

Any additional novelty or perturbation study remains deferred planning. This
diagnostic adds no perturbation recipe, hyperparameter search or scientific
acceptance claim.
