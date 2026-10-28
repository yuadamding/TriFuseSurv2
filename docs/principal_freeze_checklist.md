# Principal-study freeze preparation

This is preparation for a new principal protocol version, **not a frozen
protocol or an execution release**. The common epoch cap remains `null` until
both qualified development traces support the prespecified decision. No cap is
inferred from a completed F2 fit or an interrupted F3 fit. Retain every existing
attempt, source, numerical-policy record and failure disposition.

The local draft manifest is
`/storage/TriFuseSurv/runs/study-20261028/principal-freeze-preparation-b/principal_freeze.draft.json`,
SHA-256 `73e445601ee2cd5548f58382e0873fa0bb123ab2d9c0461ff89a162951c2a359`.
It binds source files, existing configurations, accepted qualification evidence,
measurements, the replacement baseline candidate and the field mapping below.
Its preparation used local source reads and metadata hashes; no new tests,
model runs, checkpoint/image payload reads or remote operations were performed.

| Component | Existing fields/evidence | Content to carry into the new freeze |
| --- | --- | --- |
| Endpoint | `endpoint_definition` | OS from diagnosis, days, event 1 = death from any cause; event 0 uses the supplied fixed OS.TIME. No clock-based follow-up extension. |
| Membership | `cohort`, split hashes | 519 eligible patients, unchanged 365/77/77 assignment; 94 paired-missing-OS exclusions from 613 prepared cases. |
| Measurements | `baselines.geometry`, exact manifest/radiomics inventories | Existing common-grid policy and reference geometry, original validity/availability states, accepted computational-development limitations. |
| Clinical inputs | `training.clinical_type_policy`, `paired_shared_settings.clinical_cols`, `baselines.clinical` | AGE numeric; PATHOLOGY, SEX, RACE, T, N, M, KFCF, SMOKE, ALCOHOL, HPV categorical. NSTAGE/TX excluded. Fit preprocessing on training patients only. |
| Numerical behavior | Bound numerical policy plus accepted production result | CUDA FP16 autocast, default dynamic scaler; effective CUDA matmul TF32 **true**, cuDNN TF32 **false**, float32 matmul precision `high`. |
| Common budget | `training.common_epochs`, budget specification | Still unresolved. Bind the paired assessment, accepted common cap and full cosine schedule together. |
| Full training configuration | Budget-B parsed settings and final principal settings | Preserve architecture, objectives and optimizer; explicitly bind principal seeds, full eligible input scope, exports and fresh output roots. |
| Selection | `training.selection` | Validation NLL only; strict best-history minimum and existing final-candidate tie priority. |
| Fit inventory | `training.paired_seeds`, `evaluation.mode_seeds` | Six fresh F2/F3 fits; two selected deterministic baseline models from ten candidate fits. |
| Evaluation | `evaluation`, `analysis` | Fixed patient inventory/grid, mean paired deltaIBS, 0.005 worthwhile reduction, paired bootstrap and supporting analyses. |

## Source and accepted qualification

Scientific source `3298bf017c667ea44ae8ad621dc8cfaa95961aa6` has executable-tree
SHA-256 `c40782445575be0e3338971c03f31764080d3ad77db60e5b9b6ec60ed7603806`.
Documentation commit `038296466f5ac0baa9e5d84c9dc0e078049d65a5` records that
qualification without changing these executable bytes. The final principal
release must identify its own exact source; the draft does not invent a future
commit or inherit an unverified source identity.

| Implemented component | SHA-256 |
| --- | --- |
| Trainer | `d6e9d4caaf959f33942c0c4856e7ff2769e0536c72be468a8de2c0384d3570c2` |
| Corrected baseline reader/runner | `e21f961c35dd55ca9988fca26173413048c90e8ae21382f73423c46303e4e361` |
| Baseline preprocessing | `27c4214038f7e757742de23d50c41fe8ee4661b45ab8a067caa6dd2888adfcfd` |
| Paired evaluator | `738e218a25366172b80a333d70c72beff68a652bdf9a89b6894f735d251b34c9` |
| Curve writer | `8a18e9654f32b237f166d463b29a45f10ae6212a2761317e4e48848ad9f777e5` |
| Production qualifier | `0d1648e16b6c25a46ea1486258b9505ba7670492e48a67f163d4bfb9bbbd6445` |

The [accepted production result](production_qualification_result.json) binds raw
report `115ee00881cccdba83735fcb27a96f015f8dacbd935a68fe42a5bea744491665`, worker
gate `2b02c9780f170d53afaca210194e06c3259b47032a2f3e21926bf6f0a77cd04f` and
independent metadata acceptance
`a2c6e7c4517ce1e4c307b47ed5e8186ded858ed85f9594f42e8c56345693c32b`.
Both modes passed the fabricated full CT/Swin survival-NLL path: two actual
successful updates, zero skips, full eight-batch and final one-batch accumulation
boundaries, finite required gradients/state and following NLL. These updates
do not contribute to real-cohort exposure. Fixture dimensions and parameter
counts do not describe the cohort models.

The unchanged precision policy is
`e0d01361ddcaa14b4176dae9ab07be8c67d6d2c446e1e5220dd01887802b9033`:
initial scale 65,536, growth factor 2, backoff 0.5, growth interval 2,000.
Effective cuDNN deterministic is true, benchmark false, and deterministic
algorithms are enabled with the trainer's existing warning behavior.

An **unapplied** future logging patch is retained at
`principal-freeze-preparation-b/effective_backend_logging.proposed.patch`
under the local study root, SHA-256
`f3d32e4a44f8c5107514c44a5915d8633af2fade5cd179b15513c57d937741fe`.
It replaces the stale local-variable display with the effective getters for
CUDA matmul TF32, cuDNN TF32 and float32 matmul precision. It changes no backend
assignment, precision, gradient scaler, RNG, optimizer or schedule. It is a
review-only proposal for a future source revision, not a patch to budget B.

## Inputs and corrected baseline candidate

The exact eligible metadata, outcomes and 519-case manifest inventory remain
`a7121f3b4258f33bd1189931c2a70f7e13b036017c1fe5235714105ff362292e`,
`0626de5668aa88564ed2837acbf53c4fd276fc591ae9a8e26279da527299d5fd`, and
`4205ccd31e36905c5a04d5d973451a4201b941fd7428db6c1a34a2b1e9da092f`.
The full eligible radiomics CSV is
`3a38702a4ac0248ea46829c592bff00e00d4804016697fb27fea8fca80eb7e82`.
The manifest also binds individual split-file hashes, the geometry export and
source receipts. The 442-patient development views are separately identified;
they do not replace the full principal inventory.

Keep missing/invalid anatomy distinct from known absence. Reference measurements
are used only where their validity flag is true; invalid values remain null or
masked. Computational qualification retains source-FOV limits and unavailable
original CT/RTSTRUCT provenance. It does not establish anatomical accuracy,
clinical validity, or a new whole-cohort visual review.

The corrected baseline candidate is local at
`baseline-development-b/sealed/baseline-development-b`. Its PLAN SHA-256 is
`c32c0bd4f2eeb55ea0d2ec90674a26054499a836baceef364e9de279460f1925`.
`CLINICAL_CELL_IDENTITY.json`, SHA-256
`f831982d1c1eb55750726ed1f39f28d05e2af2cacd500d41f1f0f5a66d107ef8`, records
identical 4,862 approved clinical cells across 442 train/validation patients and
whole-CSV byte identity. It analyzes no held-out clinical cells or outcomes.
Literal nonblank categories remain verbatim, including `NA`, case, numeric-looking
strings and surrounding whitespace; blank/whitespace-only cells are missing.
AGE remains available for strict numeric validation. Training-only vocabulary
fitting and distinct missing/unseen slots remain required.

This is a **candidate**, not a deployment release. Its retained baseline recipe
SHA-256 is `a6d129c091666955ef997fc950c218fcabc64a5566ebfdb9183357dee850bf7b`.
The bound 16-wheel dependency lock is
`248537fa00118cee03d61aa2bda6fb1a3074704341df6d43c4eb5e694a23ba31`.
Source-reader regression acceptance and verified input bytes do not claim a
new worker environment or cohort fit. Release follows the principal freeze
under the user's execution order.

## Training, selection and complete panel

For every fresh principal fit, use the same accepted cap and
`CosineAnnealingLR(T_max=cap, eta_min=1e-6)`. AdamW uses betas (.9,.999), epsilon
1e-8, batch size 1 and accumulation 8, including the final partial accumulation:
46 planned opportunities per epoch. Actual attempts, successes and skips remain
separate measured counts. No acceptable-skip threshold is added after a run.

| Parameter group | Learning rate | Weight decay |
| --- | ---: | ---: |
| Backbone | 0.001 | 0 |
| Head | 0.0001 | 0.001 |
| Clinical | 0.0001 | 0.0005 |
| Radiomics | 0.0001 | 0.001 |

The complete parsed settings are retained in the draft's companion
`budget_B_settings_by_mode.json`; they include inactive LoRA settings without
claiming an active LoRA group. Architecture, losses and zero-dropout policy
remain fixed. Each principal mode starts fresh from its paired seed, with no
contour warm-start, EMA or SWA. Budget and pilot weights are not principal
initializations.

| Sequential neural run | Mode | Seed |
| ---: | --- | ---: |
| 1 | F2 | 42 |
| 2 | F3 | 42 |
| 3 | F2 | 142 |
| 4 | F3 | 142 |
| 5 | F2 | 242 |
| 6 | F3 | 242 |

C0/C1 each evaluate the five fixed penalties [.01,.1,1,10,100] using validation
reference IBS, Efron ties, and the larger penalty on an exact selection tie.
No validation refit is performed. Thus the panel contains eight selected final
models after ten baseline candidates and six fresh neural fits.

Neural `best.pt` keeps the first strict minimum validation NLL. Final export
selection uses the existing insertion priority last, EMA, SWA, best among
enabled candidates; with EMA/SWA disabled, an exact final tie selects last.
Held-out outcomes do not select checkpoints. Preserve complete unfavourable
fits as well as favourable fits.

## Exports, analysis and finalization

Export complete per-patient survival curves from the actual selected serialized
checkpoint for validation and test. Bind checkpoint, encoder/model contract,
source, literal IDs, split and seed. Include native bin endpoints and all frozen
times: 210,240,270,300,330,360,365,390,420,450,480,510,540,570,600,630,660,690,720,
730,1095 days. Scalar risk is `1-S(730)`; survival must be finite, within [0,1]
and nonincreasing. The evaluator requires exact inventory coverage and verifies
artifact files, rather than accepting export-declared hashes alone.

The primary estimate is the mean of three paired `IBS(F3)-IBS(F2)` differences
over the fixed 210–730-day grid. The worthwhile threshold remains **−0.005**.
Do not replace this with a best-seed result or IBS of an averaged prediction.
Use scikit-survival 0.25.0 and 2,000 shared patient-bootstrap draws seeded
20261028, pairing the same resampled patient indices across modes and seeds.
Retain invalid draws; require at least 95% valid for a percentile 95% interval.
Report seed spread separately from this interval conditional on fixed models
and the training censoring reference.

Supporting analyses remain IPCW concordance at tau 730 using `1-S(730)`, Brier
scores at 365/730/1095 where supported, and descriptive censoring-aware
calibration using fixed edges 0,.2,.4,.6,.8,1. Unsupported metrics retain an
explicit unassessable disposition; no horizon retuning, patient deletion,
replacement bootstrap draws or test-driven refitting. Malformed artifacts are
errors, not scientific unassessability.

To finalize this preparation, supply the qualified paired budget decision,
bind its accepted cap and complete schedule, publish a new frozen principal
specification with exact per-fit settings/source/input identities, and release
the replacement baseline package under that version. The principal matrix and
analysis above already specify the remaining scientific choices. No new
codebook, anatomical-review or elapsed-time approval requirement is added.

The current study JSON still contains historical new-source-qualification and
prequalification numerical status strings. The separate accepted production
result supplies that evidence without rewriting bound history. Likewise,
`baselines.fusion_clinical_columns` retains a pre-redundancy accounting inventory;
the actual selected numeric/categorical fields exclude NSTAGE. These distinctions
must remain explicit in the new principal version.

## Subsequent budget-B recovery disposition

The collected B evidence changes what can be carried forward as adequate
training. F2 completed 12 epochs with finite checkpoint audits, but recorded
**551 successful updates out of 552 opportunities**, one AMP skip and one
nonfinite-gradient update in epoch 1. F3 was interrupted after one complete
epoch. Neither a finite checkpoint audit nor F2's lack of a curve-extension
trigger accepts the paired cap with this exposure unresolved.

The local `budget-b-recovery-a/RECOVERY_DECISION.json` preserves these results.
The prospective recovery now proposes **fresh fits of both modes** under a
separately bound FP16 initial-scale-32,768 candidate, after fresh production-path
qualification of that source and policy. This candidate follows F2's observed
backoff to 32,768 and absence of later skips; it is not itself an H100 pass.
Original default-scale F2 remains historical evidence and is not pooled as a
new-policy fit. The original draft manifest above intentionally retains the
already qualified default-scale implementation and records no new policy pass.

The trainer's optional `--amp_init_scale` defaults to `None`, preserving its
existing default factory call and legacy contract shape. An explicit finite
positive override is immutable in the experiment/checkpoint contract. The
effective-backend logging correction is now future source work, while the
retained B archive remains unchanged. Release requires validators bound to the
new policy; the old helpers' fixed default-scale expectation cannot be silently
reused for 32,768. No additional GPU may be allocated while the old Job, its
owned Pod or execution owner remains unreconciled.
