# Availability-cycle mismatch implementation

The [specified control](mismatched_routing_study.md) is implemented in a
prospective revision. Its production H100 qualification and three cohort fits
remain pending. Active budget C continues using its archived `1fed762` source;
these local changes do not alter that attempt.

Use `fusion_mode=F3` and the optional argument
`--routing_intervention anatomical_mismatch_availability_cycle_v1` on the
verified v2 principal configuration. The inactive default is `None`. Other
fusion modes, altered compartment inventories and unavailable radiomics
construction are rejected for the intervention. No parameters are added.

The saved intervention identity is:

| Field | Value |
| --- | --- |
| Identifier | `anatomical_mismatch_availability_cycle_v1` |
| Frozen design SHA-256 | `c9c30676ca2059ead03f4ec7584fe0a688e7b7c9e916bd02ccc6956072ea66df` |
| Prediction mode label | `F3_mismatch_availability_cycle_v1` |
| Model fusion mode | `F3` |

The regional order is PT-intra, PT-peri, LN-intra, LN-peri. The runtime reads
actual Boolean image-query and radiomics-key presence after preprocessing and
model-side availability handling. It computes the next canonical member of
each joint-availability class, retaining singleton identity fallback. It expands
ordinary F3 routing to a patient-specific mask and replaces only the four
regional radiomics assignments. The original availability mask is then applied.
Source projections, anatomical identities, embeddings, image-query gating,
clinical/image routes, global fusion and region-status values remain unchanged.
The assignment uses no feature values, outcomes, patient IDs, seed or RNG draw.

Nine focused runtime tests check all 256 mask pairs against actual attention
masks and preglobal local outputs, the usable-access and incoming-exposure
invariants, singleton fallback, mixed batches, train/eval behavior and blocked-key
local isolation. Eight existing fusion tests also pass. The blocked-key test
examines local habitat tokens; it does not require final-risk invariance.

An independent comparison with the original model sources passed bit for bit
in 18 scenarios: both model classes, seeds 42/142/242 and evaluation/training
conditions, including nonzero dropout. Each scenario covers all 256 availability
patterns, initial parameters, construction/forward RNG, outputs/auxiliary values
and named gradients. The historical source files are identical to budget C's
corresponding model files. This establishes the tested CPU model compatibility;
it does not replace the new branch's production numerical qualification.

The trainer omits inactive `None` from historical canonical settings and saved
arguments. An active identifier, design hash and export label are immutable
experiment/checkpoint fields. Restoration, selected/averaged inference weights
and exports reject inconsistent same-shape routing identities. The construction
mapper retains its strict historical adapter and adds the exact new constructor
plus reviewed helper AST/source bindings; unknown drift is rejected. Its helper
AST normalization handles the empty `type_params` difference between CPython
3.10 and 3.12 without ignoring declared generic parameters.

Active prediction exports retain the true F3 model configuration while using
the distinct prediction label. Each split also receives a routing-coverage CSV
in literal prediction-patient order. It records actual query/key presence,
assigned key for each query, active edges, effective wrong edges and active
identity-fallback edges. The curve sidecar binds that CSV hash together with
the selected checkpoint and model-contract hashes, plus complete split/stratum
aggregates. Missing, duplicate or malformed actual coverage is rejected. The
contract binds the coverage schema. Training-split coverage explicitly describes
selected-weight `train_eval` forwards, not every training step. Development-only
fixtures export train/validation coverage without loading reserved test data.
Inactive F3 exports add no coverage artifact or new metadata.

The qualifier now offers the explicit `--comparison routing-intervention`.
Its settings contain matched `F3` and `F3_mismatch_availability_cycle_v1`, both
with actual fusion mode F3. The intervention settings bind all three identity
fields. Schema 2 records actual training/evaluation prelocal masks, local fuser
masks and finite preglobal outputs, shared initial parameter/RNG state, and
optimizer exposure. Ordinary paired F2/F3 qualification retains schema 1.
A historical paired pass, diagnostic precision override or tiny CPU report
cannot establish the production intervention pass.

The tiny fabricated Swin/NLL comparison made two successful CPU updates per
mode, including the final partial accumulation, with zero skips. The actual
intervention branch executed: its three training forwards had 12 effective
wrong edges, while the matched control had zero. Contract/export tests include
a full small synthetic training fold and actual mixed-mask prediction export.
These are engineering checks, with no real cohort or H100 intervention fit.
The [machine-readable implementation record](mismatched_routing_implementation.json)
binds their source files and receipts.

Before the later three fits, publish a new exact source/settings/image/numerical-
policy addendum, qualify both branches on the full production H100 profile
(nine batches, accumulation eight, two actual successes and zero skips), and
bind that report to the release. Preserve historical candidate policy bytes;
their source hashes describe C, not this prospective revision. Follow the
existing one-H100 sequence, accepted common cap and unchanged mismatch analysis.
The primary F3-minus-F2 IBS criterion and 0.005 worthwhile reduction remain fixed.
