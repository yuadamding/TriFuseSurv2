# Production-gradient diagnostic activation — October 29, 2026

The bounded real-data diagnostic was activated at **10:26:43 AM CDT on
October 29, 2026**. At the **10:29:21 AM CDT** snapshot it was Running/Ready on
one verified NVIDIA H100 80GB HBM3, with zero restarts and the exact detached
supervisor identity alive. PyTorch is the pinned `2.9.1+cu128`, CUDA build 12.8,
and compute capability 9.0. No numerical result or new policy decision was
available at this snapshot.

The [machine-readable record](production_gradient_diagnostic_20261029.json)
binds source, plan, admission, controller, startup and snapshot evidence. The
[diagnostic protocol](production_gradient_diagnostic.md) contains the complete
work and decision rules; its linked full C metric CSVs remain unchanged.

| Item | Bound value or observed result |
| --- | --- |
| Job | `trifusesurv2-production-gradient-diagnostic-20261029a` |
| Job UID | `7ab92dcd-d4bc-4458-8b6b-393e4ad859c2` |
| Scientific model/training source | Archived C, `1fed762` |
| Diagnostic sidecar source | `4f1d220`; three scripts and their focused test |
| Real-data configuration | F2, seed 42, original FP16 scale 32,768, accumulation eight |
| Original candidate schedule | 12 epochs and complete `T_max=12`, retained during the bounded observation |
| Diagnostic work ceiling | One original training epoch, 46 optimizer opportunities; then three matched replays of one selected window |
| Target location | Update 27 / batches 209–216; inference awaiting observation |
| Container source-test stage | Completed with exit zero; fixed inventory of 11 fabricated CPU tests |
| Real-data diagnostic stage | Started at 10:28:10 AM CDT; incomplete at the snapshot |
| Full budget fits / test scoring | None |
| Allocated / maximum H100s including Pending | 1 / 1 |
| Automatic retries / active execution deadline | None / none |

Both immutable bundles were published and readback verified: 10 workload files
and 47 controller files. The temporary staging archives were removed, and no
Mac relay files were created. The existing environment was reused without
installation. The publisher verified all 4,420 source-bound development files
for 442 training/validation cases, totaling 709,056,957 bytes. The diagnostic
forwards remain restricted to the 365 training cases. Actual-worker input byte
checks precede the diagnostic epoch.

The final source-stable local fabricated CPU suite passed **379 tests and 340
subtests**. The portable lifecycle passed **49 tests locally and 49 on the
actual Seadragon host interpreter**. An exact sidecar-only layout, importing
archived C source, passed the 11 diagnostic tests. These checks establish their
engineering scopes; they do not qualify a numerical correction or training cap.

C's owner absence was freshly verified on its bound backend, and its exact
Job/Pod absence was verified twice before this release. C's rejected numerical
result remains intact. The original source is read without modification;
the new stop-on-failure policy is prospective and is not inserted into the
historical reproduction. A failing historical replay can be a completed
diagnostic, while runtime errors or original/replay disagreement remain
unresolved and cannot justify candidate advancement.

The successor preparation retains a single combined production-qualification
and fresh F2/F3 12-epoch Job, with the new immediate numerical-stop policy.
Its selected numerical policy, accepted source and release bindings remain
unresolved. No successor Job has been submitted. Principal/baseline release
candidates remain unfrozen and unreleased, and the six principal fits,
mismatch H100 qualification, three mismatch fits and held-out evaluation remain
pending. Their existing scientific definitions are unchanged.

The detached supervisor owns terminal capture and exact cleanup. This is one
dated activation snapshot; it does not establish diagnostic completion, policy
acceptance or an overall completion ETA.
