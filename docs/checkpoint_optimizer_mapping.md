# Reconstructing a frozen pilot's optimizer parameter map

`scripts/map_checkpoint_optimizer.py` produces the parameter-name mapping needed
by the pilot auditor. It reconstructs the actual model and calls the frozen
trainer's `make_param_groups`; it does not assign names by tensor shape or by
the order of checkpoint `model_state` entries.

This tool is synthetically tested. It has not qualified an actual completed
paired-development pilot checkpoint. A pending Job, an allocated GPU or this
tool's existence does not establish successful survival updates.

## Supported reconstruction

The current adapter supports verified-region v2 F2/F3 with full fine tuning,
without LoRA or an explicitly permitted partial resume. A complete checkpoint
must contain its full model state, optimizer state, serialized arguments,
fitted encoder objects and dimensions, and the source-bound model contract.
Lightweight checkpoints cannot supply optimizer evidence.

The command must run in a fresh Python process. Its helper lives beside the
script, allowing it to import the older selected package without copying any
file into that package. Bytecode-cache writes are disabled before importing the
frozen package. It checks the checkpoint file's expected SHA-256 and
requires the selected package's executable source-tree digest to match both
the checkpoint and its model contract. It uses the source identity's existing
algorithm over `src` and `scripts` Python/shell files. It rechecks both source
and checkpoint identities before publishing evidence.

Construction uses the three original backbone/model construction statements
extracted from the checksum-bound `run_one_fold` AST. This retains the actual
constructor arguments, source defaults and class implementations. Unknown
construction layouts fail rather than receiving a guessed adapter. Clinical
and radiomics dimensions come from the checkpoint's stored encoders and must
agree with their serialized `output_dim`; no metadata, CT, contour, radiomics
CSV, training batch or encoder fit is needed.

The reviewed construction AST digest is
`fa777fcf8671ab33730c92baf4b7f3356557ba85c2423a11e59f876c19099f57`.
A read-only comparison found the same three-statement block at frozen source
`614d7fb` and in the prospective implementation. The adapter requires that
digest; this source compatibility check does not replace actual-checkpoint
reconstruction and acceptance.

PyTorch's strict named state-dictionary loading initializes registered Lazy
modules from their corresponding checkpoint entries. Parameter registration
order must remain unchanged. Missing/extra keys, unresolved lazy tensors,
nonfinite state, shape/dtype mismatches, unexpected frozen parameters, or total
and trainable parameter counts differing from the original contract stop the
operation. Forward-created parameters would introduce unmatched state keys and
are not supported. Original warm-start files are unnecessary: their initial
values are superseded by the complete checkpoint state. A changed trainability
policy needs a separately reviewed adapter.

The selected source's `make_param_groups` returns live parameter objects. The
tool associates those objects with `model.named_parameters()` by object
identity, then creates an unstepped AdamW instance using the original settings.
Serialized parameter IDs and their group order must match its state-dictionary
layout exactly. Weight decay, initial learning rate where stored, Adam options,
group coverage, and existing moment shapes/finiteness are checked. It runs no
forward pass, loss computation, optimizer step or model fit.

The PyTorch version must exactly match the checkpoint's execution contract.
Supply the MONAI version from the qualified execution environment explicitly;
the tool checks the installed version and records both runtime versions. The
historic paired pilot environment reports MONAI 1.5.2, but that dated observation
does not qualify a new environment by itself. Use only trusted project-owned
checkpoints: serialized fitted encoder objects require ordinary PyTorch
deserialization rather than `weights_only` loading.

## Invocation and auditor integration

Run in the approved project execution environment after a real `last.pt`
exists and its source/runtime/input receipts have been reconciled:

```bash
python /new-postprocessor/scripts/map_checkpoint_optimizer.py \
  --checkpoint /approved/pilot/F2/last.pt \
  --checkpoint-sha256 <receipt-bound-checkpoint-sha256> \
  --source-root /approved/frozen-training-package \
  --expected-monai-version 1.5.2 \
  --out-dir /approved/new-postprocessing/F2-optimizer-map
```

The output directory must be new. Run separately for F2 and F3, using each
mode's own checkpoint identity. Do not edit or relaunch a healthy bound pilot
to add this postprocessor. Real checkpoint inspection remains subject to the
approved single-H100 study's execution and storage rules.

The output is `optimizer_parameter_map.json`, with:

- `mapping_method: actual_model_parameter_groups`;
- the exact checkpoint SHA-256 and original `source_identity`;
- `parameter_names_by_group`, in actual serialized optimizer group order;
- the path and SHA-256 of `reconstruction_evidence.json`.

The evidence includes its reconstruction status/schema, the checkpoint/source
bindings, the canonical group-name digest, source construction AST/file hashes,
dimensions and parameter counts, observed runtime, postprocessor hash, and
actual-group/serialized-ID checks. The auditor validates these semantic bindings
as well as the evidence file hash. The postprocessor identity is distinct from
the older checkpoint source identity; neither is relabeled as the other.

Pass each map through the auditor's `--parameter-map F2=...` and
`--parameter-map F3=...` arguments, with the map and evidence under its trusted
project root. A successfully reconstructed map establishes parameter identity
only. The auditor must still verify positive survival-path update counters,
nonzero Adam moments in the shared survival and OS residual branches, finite
weights/losses, checkpoint selection, exact predictions and all other pilot
acceptance conditions.

Synthetic tests exercise actual model construction and parameter groups,
strict lazy loading, serialized-ID permutation rejection, unsupported policies,
missing state, and clean-process CLI publication. Their fabricated Lazy feature
widths and Adam moments are engineering fixtures, not image-forward or
scientific training qualification.
