# Pilot acceptance and cohort accounting

The standalone reports inspect retained evidence without changing the frozen
`614d7fb` pilot bundle, model architecture, or its scientific settings. Their
source SHA-256 and all supplied evidence hashes are retained in each report.
Output directories must be new. Report creation is not a training launch.

## Pilot

For a pending run, no checkpoint access or PyTorch import is needed:

```bash
python scripts/audit_pilot_completion.py \
  --execution-plan /approved/bundle/PLAN.json \
  --training-plan /approved/preparation/training_plan.json \
  --out-dir /approved/new-pilot-report
```

After completion, additionally supply `--runtime-final /approved/output/runtime/FINAL.json`
and `--trusted-root /approved/project-root`. These explicitly identify the
trusted checkpoint location. The auditor verifies the runtime → training receipt
→ artifact hash chain before loading full PyTorch checkpoints on CPU. The
frozen checkpoint stores fitted encoder objects and requires `weights_only=False`;
never point this option at an untrusted artifact. Symlinks, paths outside the
trusted root, changed bytes and duplicate artifact bindings are rejected.

Outputs are `pilot_acceptance.json`, `pilot_acceptance.md` and
`pilot_runtime_summary.json`. Exit codes are 0 passed, 1 failed, 2 blocked.
Missing evidence is blocked; a failed criterion cannot be replaced by a passing
receipt label. Optional metrics and missing runtime telemetry are separately
unassessable. A passing two-epoch audit is engineering acceptance, not evidence
of convergence, predictive benefit, or clinical qualification.

Required checks cover exactly two epochs, all seven required loss fields,
finite primary risks and checkpoint tensors, literal held-out IDs with no
duplicates, source/input identities, identical shared settings and parameter
counts, and validation-NLL selection. The execution plan's source/image/run
identity and fresh worker/trainer observations must establish one H100 with
compute capability 9.x and at least 75 GiB; optional peak-memory telemetry is
separate from this required device-identity gate. The actual frozen split and accumulation
settings determine update opportunities: 365 train patients, batch 1 and
accumulation 8 give 46 per epoch and 92 in total. Adam state steps count successful
optimizer executions. They do not expose how many AMP attempts were skipped.
No skipped-update count is invented.

The frozen optimizer does not store parameter names. Generic positive Adam
steps therefore cannot establish training in the survival path. Each optional
`--parameter-map F2=/approved/map.json` or `F3=...` must contain:

```json
{
  "mapping_method": "actual_model_parameter_groups",
  "checkpoint_sha256": "last.pt SHA-256",
  "source_identity": {"same complete object as the checkpoint": "..."},
  "parameter_names_by_group": [["actual model parameter names in optimizer order"]],
  "reconstruction_evidence": {"path": "/approved/reconstruction.json", "sha256": "..."}
}
```

The map producer is `scripts/checkpoint_optimizer_mapping.py`. It reconstructs
the actual model from the frozen source and checkpoint encoders, and calls the
frozen `make_param_groups` function, with group lengths and tensor shapes
verified. State-dictionary order and shape matching are insufficient.
The auditor validates the map binding, all named survival Adam states, and
nonzero moments in both the shared and primary OS residual path. Missing maps
leave this criterion blocked. The reconstruction proof must bind schema/status,
checkpoint/source identity, the canonical parameter-group digest, actual named
parameter inventory and parameter counts. Arbitrary JSON with a matching file
hash is insufficient. Real frozen-checkpoint reconstruction still requires
execution in the qualified environment; fabricated fixtures prove software
behavior only.

Frozen `metrics.csv` does not store validation NLL. If `best.pt` is epoch one,
its full-precision metric and epoch-two `last.pt` can prove the two-candidate
selection. If `best.pt` is epoch two, the missing epoch-one full-precision NLL
leaves selection blocked. A `best` filename is not proof. The prescribed export
tie rule selects `last`. A future separately pinned validation trace or
qualified checkpoint reevaluation is needed to close that evidence gap.

New trainer runs record full-precision `val_nll`. The auditor consumes that
hash-bound epoch trace when present, including the distinction between
`best.pt` retaining the first epoch on a tie and final exports preferring
`last` on a candidate tie. New telemetry does not backfill frozen-run history.

Stage elapsed times include initialization and final evaluation; they are
reported with that scope and never relabeled seconds per epoch. Frozen runs
lack per-epoch timing and GPU/RAM peak telemetry. Available hardware and
shared-memory preflight measurements are retained separately.
New runs also record `epoch_train_and_evaluation_seconds`: epoch entry through
training and train/validation evaluation, excluding subsequent SWA,
checkpointing, Grad-CAM and final export. The report retains that named scope.

## Full cohort

```bash
python scripts/summarize_cohort_quality.py \
  --metadata-csv /mirror/development/cohort_development_common_grid.csv \
  --source-metadata-csv /mirror/local-inputs/cohort_preprocessed_stage2.csv \
  --radiomics-csv /mirror/development/radiomics_development.csv \
  --development-receipt /mirror/development/development_receipt.json \
  --builder-receipt /mirror/plan/builder_receipt.json \
  --sampling-audit-dir /mirror/development/audit \
  --visual-review-receipt /evidence/prior-visual_review.json \
  --representative-receipt /evidence/prior-representatives.json \
  --path-map /remote/project=/mirror \
  --out-dir /reports/new-cohort-report
```

The mirror also needs the builder-bound `outcome_eligibility.json` and every
development-receipt-bound `cases/<literal-ID>/region_manifest.json`. Prefix maps
resolve retained absolute paths for reads; source bytes are never rewritten.
The same report can be run where the original files are visible without a map.

Outputs are `cohort_qc_patients.csv`, `cohort_qc_regions.csv` (four observations
per patient), `cohort_eligibility.csv`, `cohort_review_queue.csv`,
`cohort_qc_errors.csv`, `cohort_qualification.json` and `cohort_qualification.md`.
IDs such as `001`, `1` and `NA` remain distinct strings. Counts default to
613 prepared / 519 OS-valid / 94 paired-missing OS outcomes. Malformed outcome
pairs fail the accounting gate; they are not silently treated as missing.
The frozen patient-level exclusion list and train/val/test assignments are
reconciled exactly. No anatomical availability rule is added to eligibility.

The report revalidates manifest geometry, common-grid declarations and
radiomics/image support/configuration bindings without opening images or
extracting new features. It distinguishes reference anatomical states from
model-grid disappearance, unknown nodal exclusion anatomy from known absence,
source-FOV flags from crop retention, and missing anatomy from extraction
failures. Invalid manifest evidence yields four unassessable observations,
preserving patient accounting without inventing absent anatomy.

Volume and model/reference-ratio distributions use only present reference
anatomy with valid support and passed geometry declarations. Zero placeholders
for missing, invalid or absent compartments do not enter these distributions.
A true model-grid zero for known-present anatomy that disappeared on resampling
remains included. Each distribution names its population, its total and
included/excluded observation counts, excluded anatomical states, and missing
measurement count. Raw declared volumes remain available in the compartment
table for audit, without being interpreted as measured absent anatomy.

Review triggers include every geometry/binding failure, disappearance,
extraction failure, crop/source-boundary flag and volume ratio outside
`[0.8, 1.2]`, plus five ordinary cases selected by a deterministic ID hash.
`--review-volume-ratio` and `--ordinary-sample-size` record any alternate review
policy. These thresholds prioritize review and never exclude a patient.
Historical representative observations are imported only when both copied panel
bytes and manifest hashes match the hash-bound representative receipt. A panel
match alone is insufficient. Applicability results remain explicit; historical
evidence is not expanded into a new whole-cohort review claim. New review
decisions remain pending.

Cohort exit codes are 0 accounted with no requested review outstanding,
1 failed accounting, and 2 accounted with review pending. The JSON exposes
accounting and review gates independently. Source limitations, provisional
tissue/intensity definitions and lack of expert contour qualification remain
explicit even if accounting passes.

The focused tests use fabricated CPU fixtures only:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python -m pytest -q \
  tests/test_pilot_acceptance.py tests/test_cohort_quality.py
```
