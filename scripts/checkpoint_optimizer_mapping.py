"""Read-only reconstruction support for trusted, source-bound v2 checkpoints.

This module lives beside the command so it can import an older frozen package
without adding files to, or changing the identity of, that package.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys


# Reviewed construction block shared by the frozen 614d7fb pilot and the
# prospective source. A changed constructor requires reviewing this adapter.
SUPPORTED_CONSTRUCTION_AST_SHA256 = "fa777fcf8671ab33730c92baf4b7f3356557ba85c2423a11e59f876c19099f57"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root):
    """Match the frozen package's source_identity source-tree algorithm."""
    root = Path(root).resolve()
    files = sorted(path for directory in (root / "src", root / "scripts") if directory.is_dir()
                   for path in directory.rglob("*") if path.is_file() and path.suffix in {".py", ".sh"})
    if not files:
        raise ValueError("No executable source files found in the selected source root.")
    digest = hashlib.sha256()
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError("Source tree contains an executable file outside its bound root.")
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def import_frozen_trainer(root):
    root = Path(root).resolve()
    for name in list(sys.modules):
        if name == "trifusesurv2" or name.startswith("trifusesurv2."):
            raise ValueError("TriFuseSurv2 is already imported; use a fresh interpreter for source-bound reconstruction.")
    # Imports must not create __pycache__ entries inside the frozen package.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    trainer = importlib.import_module("trifusesurv2.multimodal_survival.train")
    if not Path(trainer.__file__).resolve().is_relative_to(root / "src"):
        raise ValueError("Trainer import escaped the selected source root.")
    return trainer


def construction_statements(trainer):
    """Extract the three construction statements from the actual bound trainer.

    No copied architecture recipe, encoder fitting, dataset access, warm-start
    loading, training loop or forward pass is executed. Unknown source layouts
    require a new reviewed adapter rather than a guessed reconstruction.
    """
    path = Path(trainer.__file__)
    source = path.read_text()
    module = ast.parse(source, filename=str(path))
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "run_one_fold"]
    if len(functions) != 1:
        raise ValueError("Unsupported trainer: no unique run_one_fold implementation.")
    nodes = functions[0].body
    def assigned(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    starts = [i for i, node in enumerate(nodes) if assigned(node, "backbone_cfg")]
    ends = [i for i, node in enumerate(nodes) if assigned(node, "injected")]
    if len(starts) != 1 or len(ends) != 1 or ends[0] - starts[0] != 3:
        raise ValueError("Unsupported model construction layout; refusing an inferred recipe.")
    selected = nodes[starts[0]:ends[0]]
    update, choice = selected[1:]
    if (not isinstance(update, ast.Expr) or not isinstance(update.value, ast.Call)
            or ast.unparse(update.value.func) != "backbone_cfg.update" or not isinstance(choice, ast.If)
            or "model_version" not in ast.unparse(choice.test)
            or not any(assigned(node, "base_model") for node in choice.body)):
        raise ValueError("Unsupported model constructor statements.")
    # In the supported path only LoRA changes trainability before parameter
    # grouping. Checkpoint weight loading is superseded by strict full-state
    # restoration below. Reject newly introduced freeze policies explicitly.
    function_source = ast.get_source_segment(source, functions[0])
    if function_source.count("freeze_all_params(") != 1 or "requires_grad_(" in function_source or "requires_grad =" in function_source:
        raise ValueError("Unrecognized trainability policy in source; review reconstruction before use.")
    block = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(block)
    canonical = ast.dump(block, include_attributes=False)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    if digest != SUPPORTED_CONSTRUCTION_AST_SHA256:
        raise ValueError("Construction AST differs from the reviewed frozen-pilot adapter.")
    return compile(block, str(path), "exec"), {
        "trainer_path": str(path.resolve()), "trainer_sha256": file_sha256(path),
        "construction_ast_sha256": digest,
        "construction_first_line": selected[0].lineno, "construction_last_line": selected[-1].end_lineno,
    }


def reconstruct_model(checkpoint, trainer):
    import torch
    from torch.nn.parameter import UninitializedParameter, UninitializedBuffer

    arguments = checkpoint.get("args")
    encoders = checkpoint.get("encoders")
    if not isinstance(arguments, dict) or not isinstance(encoders, dict):
        raise ValueError("Checkpoint needs serialized args and encoder configuration.")
    if (arguments.get("model_version") != "v2" or arguments.get("fusion_mode") not in {"F2", "F3"}
            or arguments.get("region_mode") != "verified" or arguments.get("use_lora") is not False):
        raise ValueError("Supported reconstruction is verified F2/F3 v2 without LoRA; other policies need qualification.")
    if arguments.get("allow_partial_resume", False):
        raise ValueError("Partially resumed checkpoints require a separately qualified reconstruction history.")
    if encoders.get("model_version") != "v2" or encoders.get("fusion_mode") != arguments["fusion_mode"]:
        raise ValueError("Checkpoint encoder/model configuration disagrees with args.")
    dimensions = {}
    for kind, enabled in (("clinical", True), ("radiomics", bool(arguments.get("use_radiomics")))):
        dimension = encoders.get(f"{kind}_token_dim")
        fitted = encoders.get(kind)
        expected_class = trainer.SemanticClinicalTokenEncoder if kind == "clinical" else trainer.HabitatRadiomicsTokenEncoder
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
            raise ValueError(f"Missing exact serialized {kind} token dimension.")
        if enabled and (not isinstance(fitted, expected_class) or getattr(fitted, "output_dim", None) != dimension):
            raise ValueError(f"Serialized {kind} encoder does not confirm its token dimension.")
        if not enabled and (dimension != 0 or fitted is not None):
            raise ValueError("Disabled radiomics must have zero dimension and no fitted encoder.")
        dimensions[kind] = dimension
    contract = encoders.get("model_contract", {})
    spec = contract.get("experiment_spec", {})
    if trainer.experiment_spec_sha256(spec) != contract.get("experiment_spec_sha256"):
        raise ValueError("Checkpoint experiment specification checksum mismatch.")
    if spec.get("runtime", {}).get("torch_version") != str(torch.__version__):
        raise ValueError("PyTorch runtime differs from the source-bound checkpoint runtime.")
    frozen_settings = spec.get("settings", {})
    for key, value in frozen_settings.items():
        if key not in arguments or json.dumps(arguments[key], sort_keys=True) != json.dumps(value, sort_keys=True):
            raise ValueError(f"Serialized checkpoint arg differs from its model contract: {key}.")
    if (contract.get("num_time_bins") != checkpoint.get("num_time_bins")
            or contract.get("fusion_mode") != arguments["fusion_mode"]
            or contract.get("time_bin_width_days") != arguments.get("time_bin_width_days")):
        raise ValueError("Time bins/fusion configuration disagrees with the serialized model contract.")
    code, evidence = construction_statements(trainer)
    args = argparse.Namespace(**arguments)
    namespace = dict(trainer.__dict__)
    namespace.update(args=args, fold=0, fold_num_time_bins=int(checkpoint["num_time_bins"]),
                     clin_dim=dimensions["clinical"], rad_dim=dimensions["radiomics"],
                     mode=trainer._image_encoder_mode(args), device=torch.device("cpu"))
    exec(code, namespace)
    model = namespace["base_model"]
    registered_names = list(dict(model.named_parameters()))
    state = checkpoint.get("model_state")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint has no complete model state.")
    for name, value in state.items():
        if isinstance(value, (UninitializedParameter, UninitializedBuffer)) or not torch.is_tensor(value):
            raise ValueError(f"Unmaterialized or unsupported model state: {name}.")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"Nonfinite model state: {name}.")
    # PyTorch's registered Lazy-module load hooks initialize by exact named
    # state keys. This never searches for parameters by coincident tensor shape.
    model.load_state_dict(state, strict=True)
    if registered_names != list(dict(model.named_parameters())):
        raise ValueError("Checkpoint loading changed parameter registration/order.")
    for name, parameter in model.named_parameters():
        if isinstance(parameter, UninitializedParameter):
            raise ValueError(f"Lazy parameter remains unmaterialized: {name}.")
        if not parameter.requires_grad:
            raise ValueError("Unexpected frozen parameter in supported full-finetuning policy.")
    actual = model.state_dict()
    if set(actual) != set(state) or any(actual[name].shape != state[name].shape or actual[name].dtype != state[name].dtype
                                       for name in state):
        raise ValueError("Reconstructed model state inventory/shape/dtype mismatch.")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameters != contract.get("parameters") or trainable != contract.get("trainable_parameters"):
        raise ValueError("Reconstructed total/trainable parameter counts differ from the checkpoint contract.")
    evidence.update(encoder_dimensions=dimensions, parameters=parameters, trainable_parameters=trainable,
                    named_parameters=registered_names, lazy_materialization="strict_named_state_dict_load",
                    forward_passes=0, encoder_fits=0, warmstart_policy="full checkpoint state supersedes initial values",
                    trainability_policy="bound-source full fine tuning; LoRA and partial resume rejected")
    return model, evidence


def parameter_mapping(model, checkpoint, trainer):
    import torch

    args = checkpoint["args"]
    controls = {name: float(args[name]) for prefix in ("lr", "wd")
                for suffix in ("backbone", "lora", "head", "clin", "rad") for name in [f"{prefix}_{suffix}"]}
    groups = trainer.make_param_groups(model, **controls)
    names_by_identity = {id(parameter): name for name, parameter in model.named_parameters()}
    names = [[names_by_identity[id(parameter)] for parameter in group["params"]] for group in groups]
    flattened = [name for group in names for name in group]
    if len(flattened) != len(set(flattened)) or set(flattened) != {name for name, parameter in model.named_parameters() if parameter.requires_grad}:
        raise ValueError("Actual source optimizer grouping omitted or duplicated trainable parameters.")
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.0)
    fresh = optimizer.state_dict()
    saved = checkpoint.get("optimizer_state")
    if not isinstance(saved, dict) or not isinstance(saved.get("param_groups"), list):
        raise ValueError("Full optimizer state is required; lightweight checkpoints cannot prove this mapping.")
    if len(saved["param_groups"]) != len(fresh["param_groups"]):
        raise ValueError("Serialized and reconstructed optimizer group counts differ.")
    id_to_name = {}
    for index, (expected, actual) in enumerate(zip(fresh["param_groups"], saved["param_groups"])):
        if expected["params"] != actual.get("params"):
            raise ValueError(f"Optimizer parameter IDs/order disagree in group {index}.")
        for key in ("weight_decay", "betas", "eps", "amsgrad", "maximize"):
            if actual.get(key) != expected.get(key):
                raise ValueError(f"Optimizer policy {key} differs in group {index}.")
        if "initial_lr" in actual and actual["initial_lr"] != expected["lr"]:
            raise ValueError(f"Optimizer initial learning rate differs in group {index}.")
        id_to_name.update(zip(actual["params"], names[index]))
    if not set(saved.get("state", {})) <= set(id_to_name):
        raise ValueError("Optimizer state refers to an unknown parameter ID.")
    by_name = dict(model.named_parameters())
    for identifier, values in saved.get("state", {}).items():
        name = id_to_name[identifier]
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = values.get(key)
            if not torch.is_tensor(tensor) or tensor.shape != by_name[name].shape or not torch.isfinite(tensor).all().item():
                raise ValueError(f"Invalid {key} for reconstructed parameter {name}.")
    return names, {"group_count": len(names), "parameter_count": len(flattened),
                   "optimizer_states_present": len(saved.get("state", {})),
                   "grouping_method": "actual source make_param_groups + live parameter object identity",
                   "serialized_ids_match_reconstructed_optimizer": True}


def run(checkpoint_path, *, checkpoint_sha256, source_root, expected_monai_version, out_dir):
    import torch

    checkpoint_path, source_root, out_dir = Path(checkpoint_path), Path(source_root).resolve(), Path(out_dir)
    if file_sha256(checkpoint_path) != checkpoint_sha256:
        raise ValueError("Checkpoint SHA-256 does not match the expected immutable identity.")
    if importlib.metadata.version("monai") != expected_monai_version:
        raise ValueError("MONAI runtime differs from the explicitly qualified reconstruction version.")
    observed_source = tree_sha256(source_root)
    trainer = import_frozen_trainer(source_root)
    # Caller supplies a trusted project-owned checkpoint and its bound digest.
    # Encoder objects require ordinary torch deserialization, not weights_only.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    identity = checkpoint.get("source_identity", {})
    if identity.get("source_tree_sha256") != observed_source:
        raise ValueError("Selected package bytes do not match checkpoint source identity.")
    contract_identity = checkpoint.get("encoders", {}).get("model_contract", {}).get("source_identity", {})
    if contract_identity.get("source_tree_sha256") != observed_source:
        raise ValueError("Model contract and checkpoint source identities disagree.")
    model, reconstruction = reconstruct_model(checkpoint, trainer)
    names, group_evidence = parameter_mapping(model, checkpoint, trainer)
    if file_sha256(checkpoint_path) != checkpoint_sha256 or tree_sha256(source_root) != observed_source:
        raise ValueError("Checkpoint or frozen source changed during reconstruction.")
    evidence = {"schema_version": 1, "status": "reconstructed_parameter_mapping",
                "mapping_method": "actual_model_parameter_groups",
                "parameter_names_by_group_sha256": hashlib.sha256(json.dumps(names, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "checkpoint_sha256": checkpoint_sha256, "checkpoint_source_identity": identity,
                "source_root": str(source_root), "source_tree_sha256": observed_source,
                "runtime": {"torch": str(torch.__version__), "monai": importlib.metadata.version("monai")},
                "postprocessor_sha256": file_sha256(__file__), "reconstruction": reconstruction,
                "optimizer": group_evidence,
                "limitations": ["Parameter identity reconstruction is not proof of successful training updates.",
                                "The pilot auditor must independently inspect survival-path optimizer moments and steps."]}
    out_dir.mkdir(parents=True, exist_ok=False)
    evidence_path = out_dir / "reconstruction_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    mapping = {"schema_version": 1, "mapping_method": "actual_model_parameter_groups",
               "checkpoint_sha256": checkpoint_sha256, "source_identity": identity,
               "reconstruction_evidence": {"path": str(evidence_path.resolve()), "sha256": file_sha256(evidence_path)},
               "parameter_names_by_group": names}
    mapping_path = out_dir / "optimizer_parameter_map.json"
    mapping_path.write_text(json.dumps(mapping, indent=2, allow_nan=False) + "\n")
    return {"status": "reconstructed_parameter_mapping", "mapping_path": str(mapping_path.resolve()),
            "mapping_sha256": file_sha256(mapping_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--expected-monai-version", required=True,
                        help="Version from the qualified execution environment, not an inferred substitute")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.checkpoint, checkpoint_sha256=args.checkpoint_sha256,
                         source_root=args.source_root, expected_monai_version=args.expected_monai_version,
                         out_dir=args.out_dir)))
    return 0
