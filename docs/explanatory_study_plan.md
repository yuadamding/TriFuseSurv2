# Explanatory comparisons after the principal study

This plan records the explanatory contrasts before review of the principal
held-out results. The single primary comparison remains **F3 minus F2**, with
mean paired IBS over 210–730 days and the unchanged 0.005 worthwhile-reduction
threshold. These explanatory analyses cannot replace that comparison or rescue
an unfavorable primary result. The qualified production path, accepted common
training budget and complete F2/F3 core take priority.

The ladder adds **nine fits**: F0, F1 and F4, each with seeds **42, 142 and 242**.
Reuse the corresponding substantive F2/F3 fits. Run and report the complete
ladder regardless of the primary effect's sign, subject to execution approval
and the same qualification gates; do not select modes or seeds after seeing
held-out performance. This document does not authorize additional Jobs, GPUs,
retries or training. Any later execution remains within the separately approved
resource contract.

| Contrast | Fixed implementation and explanatory hypothesis |
| --- | --- |
| F1 minus F0 | F0 concatenates projected measurements and availability into an MLP; F1 uses a flat transformer with explicit token identities. A negative IBS difference would favor the flat transformer representation over concatenation. |
| F2 minus F1 | F2 adds the existing local/global hierarchy with unrestricted regional radiomics edges. A negative difference would favor that hierarchy over flat fusion. |
| F4 minus F3 | Both use matched regional radiomics routing; F4 additionally restricts clinical-context edges. A negative difference would favor the existing clinical routing restriction. |

F4 uses the existing prognostic routing in
[`schema.py`](../src/trifusesurv2/schema.py): global receives biology, burden and
host; PT intra receives biology and burden; PT peri receives biology and host;
LN intra, LN peri and shape/spatial receive burden and host. Treatment is
excluded. Bind this exact routing table and its source identity before execution;
do not choose a different table from held-out results. F0/F1/F2 have different
fusion structures and may have different parameter counts. Report those counts;
their contrasts do not isolate a parameter-count-controlled causal mechanism.

All nine fits inherit the **same future accepted common epoch cap E** and full
scientific settings as the substantive F2/F3 core. E is currently unresolved;
it must be accepted using development evidence and frozen before any principal
held-out review. Use the same cohort and literal 365/77/77 split, source-bound
measurements and missingness rules, train-fitted encoders, approved clinical
types, backbone initialization, optimizer, scheduler, numerical policy,
effective batch size, objectives and zero image-token dropout. Only the named
fusion-mode factor changes. No mode-specific tuning or extra training exposure
is permitted. Select checkpoints by the same minimum validation-NLL rule and
fixed tie/candidate priority recorded in the final core protocol. Bind all
settings and input/source hashes before fitting or exporting test predictions.

For each contrast, report every seed and the arithmetic mean of the three
paired IBS differences on the exact core 210–730-day grid. Use the same
training-reference IPCW evaluator, 730-day concordance truncation and
`1-S(730)` risk as the core. Report concordance as key secondary, supported
Brier scores at 365/730/1,095 days and descriptive censoring-aware calibration.
Negative IBS/Brier differences and positive concordance differences favor the
first-named mode. Preserve unsupported metrics and their reasons; do not drop
patients, shorten grids or retune horizons after review.

Use the core's 2,000 paired patient-bootstrap draws, seed 20261028, shared across
modes and training seeds. Report **descriptive, pointwise 95% percentile
intervals**, with the same minimum 95% valid-draw rule and invalid-draw accounting.
These intervals condition on the fitted models and training censoring reference;
they do not include retraining uncertainty. Show training-seed spread separately.
There is no multiplicity-controlled confirmatory claim for this three-contrast
ladder, its secondary metrics or horizons. The primary F3-minus-F2 analysis and
its threshold remain unchanged.

The [availability-matched mismatch design](mismatched_routing_study.md) now
specifies one deterministic incorrect-routing rule in a separate
[versioned specification](../configs/mismatched_routing_spec.json). It adds
**three fits beyond the nine-fit ladder**, using seeds 42, 142 and 242 and the
corresponding substantive F3 comparators. The rule cycles edges within joint
image/radiomics availability classes, retains singleton identity fallbacks and
preserves token identities, values and availability. Its
[runtime implementation and CPU actual-mask checks](mismatched_routing_implementation.md)
are complete. Production H100 qualification and the three cohort fits remain
pending under the accepted-policy and resource gates; the design does not
displace the core study. Equal
sparsity and available access do not imply equal biological information.

**Contour robustness remains planned, but not yet prespecified.** Before it can
run, a separate versioned design must freeze perturbation operators, physical
magnitudes and distributions; targeted compartments and anatomical constraints;
empty/invalid-contour and unavailable-measurement handling; synchronized image
mask, radiomics and geometry regeneration; clean-training, augmentation and
consistency comparisons; any new loss weights; paired perturbation seeds,
training seeds and fit counts; and evaluation contrasts and uncertainty rules.
Choose none of these from held-out performance.

Until those contour decisions are frozen, the heading does not define an
executable test. Any design chosen or revised after principal held-out review
must be labeled exploratory, even if documented before its own subsequent
execution; retain the earlier version and its prespecified scope.
