# Availability-matched anatomical mismatch — design v1, October 28, 2026

**Specify one deterministic incorrect-routing rule and three additional fits.**
The development audit supports this design for review and implementation before
principal held-out results are examined. It does not implement the intervention,
qualify its runtime, or authorize execution. Completing the common-budget
assessment and principal F2/F3 panel remains the priority. The common cap is
unresolved; this document does not infer a cap from partial training or a
numerical qualification.

The versioned [machine-readable specification](../configs/mismatched_routing_spec.json)
defines `anatomical_mismatch_availability_cycle_v1`. Its question is whether
correct anatomical pairing adds predictive value beyond restricting local
radiomics connectivity, under the declared available-access constraints.

## Match the implemented F3 boundary

In the current [local fusion code](../src/trifusesurv2/models/habitat_survival.py),
each of `pt_intra`, `pt_peri`, `ln_intra`, and `ln_peri` has one local radiomics
edge to the same-named key. Global and shape/spatial queries have no regional
radiomics edge. Each query also retains its self-image and all approved clinical
contexts. The actual context mask is the logical AND of routing and availability;
the image-presence mask also gates that query's output before global fusion.

The intervention changes only those four query-to-radiomics routing entries.
It leaves true region identifiers, feature values, token order, source and
anatomical embeddings, missingness/state codes, and each key's own projection
unchanged. It does not shuffle radiomics vectors, rename compartments, restrict
clinical access, or modify global fusion. Every regional query still has one
nominal radiomics edge and every radiomics key still has one incoming edge.

## One fixed conditional mapping

Use the canonical order **PT-intra, PT-peri, LN-intra, LN-peri**. For each input,
let `q[i]` be the regional image-query presence and `r[i]` the radiomics-key
presence entering local fusion. Partition the four indices by `(q[i], r[i])`.
Within each class of at least two indices, ordered canonically, route every
query to the next key, wrapping the last to the first. A singleton retains its
identity edge. Apply the same rule during training, validation and testing.

```text
assignment = identity(4)
for signature in [(0,0), (0,1), (1,0), (1,1)]:
    group = canonical indices with (q[i], r[i]) == signature
    if len(group) >= 2:
        assignment[group[k]] = group[(k + 1) mod len(group)]
```

This is a fixed algorithm, not a fixed global permutation or a resampled
augmentation. It uses no patient identifier, outcome, prediction, feature value,
attention weight, split label or RNG. Routing has no random seed; model seeds
remain 42, 142 and 242. There is one rule replicate and no permutation search.

For every query, `r[assignment[i]] == r[i]`, so the number of available local
radiomics keys is unchanged, even for an unavailable image query. Also,
`q[assignment[i]] == q[i]`: each key receives the same number of incoming edges
from available image queries as in F3. The within-class cycles are bijections
and change every nonsingleton assignment. A singleton has no incorrect target
under these constraints and retains the identity edge. No patient is excluded.
Equal access counts do not imply equal biological information in different keys.

The algorithm covers all 256 binary `(q,r)` pattern pairs, including patterns
not observed in development. Future held-out patterns must use this rule without
retuning. Presence must come from the actual model masks; the metadata audit
below is not a replacement for that implementation check.

## Development-only feasibility

The audit reconciled the exact 365 training and 77 validation IDs against the
442-entry manifest inventory. It verified every referenced manifest hash, the
common-grid image-voxel/validity metadata, and radiomics availability/status and
region/support identities. It examined no held-out records, outcomes or
predictions, loaded no image or checkpoint tensor, fitted no encoder or model,
and performed no remote or GPU work. Radiomics feature values were not examined
or converted; only availability and binding fields were projected from the
development CSV. Masks were inferred from previously qualified metadata rather
than loaded again, so future encoder/mask parity remains a required gate.

In the table, binary patterns and destinations follow the canonical order above.
`I`, `P`, `L`, and `Q` denote PT-intra, PT-peri, LN-intra and LN-peri respectively.
The destination list gives the key for each query in that same order.

| Image `q` | Radiomics `r` | Train | Validation | Destinations | Effective wrong edges per patient |
| --- | --- | ---: | ---: | --- | ---: |
| 1000 | 1000 | 73 | 16 | I, L, Q, P | 0 |
| 1101 | 1101 | 1 | 0 | P, Q, L, I | 3 |
| 1111 | 1101 | 3 | 0 | P, Q, L, I | 3 |
| 1111 | 1111 | 288 | 61 | P, L, Q, I | 4 |

An effective mismatch requires an available image query, an available routed
key and a nonidentity assignment. Permuting masked unavailable keys is not
counted as an effective intervention.

| Coverage | Train | Validation | Combined development |
| --- | ---: | ---: | ---: |
| Patients receiving an effective mismatch | 292/365 | 61/77 | 353/442 (79.86%) |
| Patients with no effective mismatch | 73 | 16 | 89 |
| Effective mismatched / original active regional edges | 1,164/1,237 | 244/260 | 1,408/1,497 (94.05%) |
| Active identity-fallback edges | 73 | 16 | 89 |

All 24 fixed global permutations were enumerated. None is a full derangement
preserving every observed query's available access. Only one nonidentity partial
permutation does so: swap PT-peri with LN-peri, keeping both intra regions matched.
The singleton PT-intra pattern already prevents a universal full derangement.
The conditional rule therefore changes all feasible within-class assignments
while retaining access and explicit fallbacks; it was selected without outcomes
or model performance. The four observed strata are not a restriction on future
eligible cases.

The immutable local aggregate receipt is
`study-20261028/mismatched-routing-design-a/FEASIBILITY.json`, SHA-256
`7ee7ccab32efaa3940e5add0a87382cd52aaa689435fb98307464166e0fb806c`.
Its companion `assess_feasibility.py` records exact input/source hashes and
exhaustively checks all 256 fabricated binary-mask pairs. This is a discrete
mask/metadata feasibility check, not an actual attention or training test.
No patient IDs are included in the public report or aggregate receipt.

## Experiment matrix and analysis

After implementation, qualification and the accepted principal budget, fit
`F3_mismatch_availability_cycle_v1` once for each seed **42, 142 and 242**:
**three additional fits**, reusing the corresponding fresh substantive F3 fits.
This is a training-and-evaluation intervention, not an inference-only ablation.
Use the same future frozen common cap and complete cosine schedule, cohort and
split, train-fitted preprocessing, numerical policy, initialization, parameter
shapes/counts, optimizer, objectives, effective batch size, zero-dropout policy,
data order and checkpoint-selection/tie rules as F3. No separate tuning or
extra exposure is permitted. A source change affecting the reused F3 behavior
invalidates that reuse and requires a separately reviewed plan revision.

The main explanatory contrast is **mismatch minus matched F3 IBS** on the exact
core 210–730-day grid. A positive difference favors correct pairing. Compute
each paired seed difference, then their arithmetic mean; report all three and
seed spread. Do not average survival curves before scoring. Keep training-only
censoring estimation, the fixed 730-day IPCW concordance definition and
`1-S(730)` risk. Negative concordance differences favor correct pairing.
Report supported Brier scores at 365/730/1,095 days and the core descriptive
censoring-aware calibration results.

Reuse the principal sequence of **2,000 patient-bootstrap draws**, seed
20261028, with identical sampled indices across modes and training seeds. Each
draw recomputes the mean of the paired per-seed differences. Report pointwise
95% percentile intervals only when at least 95% of draws are valid; retain
invalid-draw counts and reasons. These intervals condition on the fitted models
and training censoring reference and omit retraining uncertainty. Unsupported
metrics remain unassessable without dropping patients or changing horizons.

The outcome estimand includes the entire original eligible test set, including
identity-fallback cases. Report availability coverage by split and stratum;
this version does not add availability-subgroup outcome tests. Fallback cases
have no direct local routing change, but their predicted risks may still differ
because the intervention changes the shared weights learned during training.
There is no multiplicity-controlled confirmatory claim, independently specified
worthwhile-effect threshold, or proof of anatomical causality from this contrast.
The primary F3-minus-F2 mean paired IBS and its 0.005 worthwhile-reduction
criterion remain unchanged; this control cannot rescue an unfavorable core.

## Required implementation qualification and limits

Before fitting, use a new explicit intervention identifier and bind the design
hash in saved contracts and exports. At design freeze the runtime had no implementation
of this new rule. The later [implementation record](mismatched_routing_implementation.md)
provides the CPU runtime, contract/export and compatibility evidence; production
H100 qualification remains required. Required tests must compare actual local attention masks,
context names and outputs, verify all 256 availability pairs and singleton
fallback, and establish same-seed parameter/RNG identity and unchanged F3
execution. Check batch independence and identical train/eval routing behavior.

Inspect local output **before global fusion** when testing blocked-key effects.
A key blocked from one query may reach another query whose fused output reaches
the global transformer; final-risk invariance is not the correct local-routing
test. Keep global/geometry/clinical access unchanged. Full source-bound numerical
qualification on the approved H100 workflow must precede any intervention fit;
this design does not authorize another Job or bypass the one-H100 limit.

This single deterministic conditional rule is not representative of all
incorrect anatomies. Its access constraints and singleton fallbacks limit the
interpretation, and engineering success cannot establish clinical validity or
resolve original image/contour provenance. A contour-robustness recipe remains
separate, deferred work. If the design changes after principal held-out review,
label that changed analysis exploratory and retain this version.
