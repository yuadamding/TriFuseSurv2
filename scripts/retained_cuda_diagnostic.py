"""Six matched fabricated CUDA cells, with durable evidence before each guard.

This deliberately imports the *original* scientific source, not the checkout
containing this diagnostic. No model, optimizer, loss or production policy is
corrected here. Only autocast/scaling differ among the three declared cells.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback


ORIGINAL_FIXTURE_SHA256 = "61cc6794a50ccb330427746d4c751b8535abe82772168d73a578f0c757b57e8a"
ORIGINAL_PLAN_SHA256 = "11ed467e1aae357f5c7491df54473c34916f0d19f89a01c74ba00f860315505d"
ORIGINAL_SOURCE_SHA256 = "c75dcaeb557791e22734c4eceedb5fb07b20b1a10f0a1ea2a9015f2f69244c13"
CELLS = ("original_fp16", "fp32", "fp16_scale1")
REQUIRED_PREFIXES = {
    "shared_survival_head": ("survival_heads.shared.", "survival_heads.shared_logits."),
    "os_survival_head": ("survival_heads.endpoint_residuals.OS.",),
}


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def write_json(path, value):
    path = Path(path)
    with path.open("xb") as handle:
        handle.write(json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    return reference(path)


def checked_file(path, expected):
    path = Path(path)
    if ".." in path.parts or path.is_symlink() or not path.is_file() or sha(path) != expected:
        raise ValueError("File identity mismatch: " + str(path))
    return path.resolve()


def source_tree_sha256(root):
    root = Path(root)
    digest = hashlib.sha256()
    files = sorted(p for directory in (root / "src", root / "scripts") if directory.is_dir()
                   for p in directory.rglob("*") if p.is_file() and p.suffix in {".py", ".sh"})
    if not files or any(p.is_symlink() for p in files):
        raise ValueError("Scientific source inventory missing or symlinked")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(sha(path)))
    return digest.hexdigest()


def tensor_summary(value):
    """JSON-safe observations; finite-only norm uses float64 and scale reduction."""
    import torch
    value = value.detach()
    dense = value.to_dense() if value.is_sparse else value
    x = dense.to(dtype=torch.float64).reshape(-1)
    finite = torch.isfinite(x)
    clean = x[finite]
    maximum = float(clean.abs().max().item()) if clean.numel() else None
    norm = (maximum * math.sqrt(float(((clean / maximum) ** 2).sum().item()))
            if maximum else (0.0 if clean.numel() else None))
    norm_overflow = norm is not None and not math.isfinite(norm)
    return {"dtype": str(value.dtype), "shape": list(value.shape), "device": str(value.device),
            "scalar_value": float(x.item()) if x.numel() == 1 and bool(finite.all()) else None,
            "elements": value.numel(), "finite_elements": int(finite.sum().item()),
            "nan_elements": int(torch.isnan(x).sum().item()),
            "positive_inf_elements": int(torch.isposinf(x).sum().item()),
            "negative_inf_elements": int(torch.isneginf(x).sum().item()),
            "all_finite": bool(finite.all().item()),
            "nonzero_finite_elements": int((clean != 0).sum().item()),
            "finite_max_abs": maximum,
            "finite_l2_norm": None if norm_overflow else norm,
            "finite_l2_norm_overflow": norm_overflow}


def tensor_identity(value):
    import torch
    if value.is_sparse:
        raise ValueError("Sparse tensors are outside the retained fixture identity contract")
    value = value.detach().cpu().contiguous()
    payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {"dtype": str(value.dtype), "shape": list(value.shape),
            "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def structured_identity(value):
    """Deterministic nested tensor/RNG identity, independent of torch.save bytes."""
    import numpy as np
    import torch
    if torch.is_tensor(value):
        return {"tensor": tensor_identity(value)}
    if isinstance(value, np.ndarray):
        return {"ndarray": {"dtype": str(value.dtype), "shape": list(value.shape),
                            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}}
    if isinstance(value, dict):
        return {str(k): structured_identity(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return {"tuple" if isinstance(value, tuple) else "list": [structured_identity(v) for v in value]}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("Unsupported fixture identity value: " + type(value).__name__)


def identity_digest(value):
    return hashlib.sha256(json_bytes(structured_identity(value))).hexdigest()


def capture_rng(cuda=True):
    import numpy as np
    import torch
    return {"torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [s.clone().cpu() for s in torch.cuda.get_rng_state_all()] if cuda else [],
            "numpy": copy.deepcopy(np.random.get_state()), "python": random.getstate()}


def restore_rng(state):
    import numpy as np
    import torch
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def cpu_copy(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def initial_identity(model, inputs, rng):
    return {"model_state_sha256": identity_digest(model.state_dict()),
            "inputs_sha256": identity_digest(inputs), "rng_state_sha256": identity_digest(rng)}


def verify_initial_match(model, inputs, expected, cuda=True):
    actual = initial_identity(model, inputs, capture_rng(cuda=cuda))
    if actual != expected:
        raise ValueError("Cell initial state/input/RNG differs from retained materialized fixture")
    return {"status": "matched", **actual}


def required_parameter_groups(model):
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    groups = {key: [name for name in names if name.startswith(prefixes)]
              for key, prefixes in REQUIRED_PREFIXES.items()}
    if any(not group for group in groups.values()):
        raise ValueError("Retained fixture is missing a declared required survival path")
    return names, groups


def gradient_observations(model, groups):
    parameters, missing, nonfinite = {}, [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            missing.append(name)
            parameters[name] = {"status": "missing"}
            continue
        record = tensor_summary(p.grad)
        record["status"] = "finite" if record["all_finite"] else "nonfinite"
        parameters[name] = record
        if not record["all_finite"]:
            nonfinite.append(name)
    required = {}
    for key, names in groups.items():
        required[key] = {"parameter_names": names,
                         "missing": [n for n in names if n in missing],
                         "nonfinite": [n for n in names if n in nonfinite],
                         "nonzero_finite_parameter_names": [n for n in names
                             if parameters[n].get("nonzero_finite_elements", 0) > 0]}
    return {"parameters": parameters, "gradient_tensor_count": len(parameters) - len(missing),
            "missing_names": missing, "nonfinite_names": nonfinite, "required_paths": required}


def gradient_failures(observed):
    failures = []
    if not observed["gradient_tensor_count"]:
        failures.append("empty_gradient_collection")
    if observed["nonfinite_names"]:
        failures.append("nonfinite_gradients")
    for name, record in observed["required_paths"].items():
        if record["missing"]:
            failures.append("missing_required_gradients:" + name)
        if not record["nonzero_finite_parameter_names"]:
            failures.append("no_nonzero_finite_required_gradient:" + name)
    return failures


def parameter_and_optimizer_observations(model, optimizer, groups):
    import torch
    parameters = {name: tensor_summary(p) for name, p in model.named_parameters()}
    states = {}
    for name, p in model.named_parameters():
        state = optimizer.state.get(p, {})
        states[name] = {str(key): tensor_summary(value) if torch.is_tensor(value) else value
                        for key, value in state.items()}
    required = {}
    named = dict(model.named_parameters())
    for group, names in groups.items():
        evidence = []
        for name in names:
            state = optimizer.state.get(named[name], {})
            step = state.get("step")
            step = float(step.item()) if torch.is_tensor(step) else step
            moments = {key: states[name].get(key) for key in ("exp_avg", "exp_avg_sq")}
            evidence.append({"parameter_name": name, "step": step if step is None or math.isfinite(step) else None,
                             "step_is_finite": step is not None and math.isfinite(step), "moments": moments})
        required[group] = evidence
    return {"parameters": parameters, "optimizer_states": states, "required_paths": required,
            "model_parameters_finite": all(p["all_finite"] for p in parameters.values()),
            "optimizer_tensors_finite": all(v["all_finite"] for state in states.values()
                 for v in state.values() if isinstance(v, dict) and "all_finite" in v)}


def update_failures(observed, successful):
    failures = []
    if successful != 1:
        failures.append("actual_optimizer_step_count_not_one")
    if not observed["model_parameters_finite"]:
        failures.append("nonfinite_updated_parameters")
    if not observed["optimizer_tensors_finite"]:
        failures.append("nonfinite_optimizer_state")
    for group, states in observed["required_paths"].items():
        if any(not row["step_is_finite"] or row["step"] != 1 for row in states):
            failures.append("required_adam_step_not_one:" + group)
        if not any(moment and moment["all_finite"] and moment["nonzero_finite_elements"] > 0
                   for row in states for moment in row["moments"].values()):
            failures.append("no_nonzero_finite_required_moments:" + group)
    return failures


def compile_original_program(path):
    """Extract setup/optimizer/scaler/loss statements from the exact pinned bytes.

    The original main and its I/O are never imported/executed. AST boundaries
    are validated, and source-segment digests accompany the diagnostic.
    """
    path = checked_file(path, ORIGINAL_FIXTURE_SHA256)
    source = path.read_text()
    module = ast.parse(source, filename=str(path))
    gate = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "cuda_gate")
    loop = next(n for n in gate.body if isinstance(n, ast.For))
    if ast.literal_eval(loop.iter) != ("F2", "F3"):
        raise ValueError("Original mode loop differs")
    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    optimizer_node = next(n for n in loop.body if assigns(n, "optimizer"))
    scaler_node = next(n for n in loop.body if assigns(n, "scaler"))
    forward = next(n for n in loop.body if isinstance(n, ast.With))
    prefix = gate.body[:gate.body.index(loop)]
    setup = prefix + loop.body[:loop.body.index(optimizer_node)]
    namespace = {"expected_observed_source": lambda plan: plan["source_identity"]}
    import torch
    namespace["torch"] = torch
    def function(name, args, body, returned):
        node = ast.FunctionDef(name=name, args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=a) for a in args],
            kwonlyargs=[], kw_defaults=[], defaults=[]), body=copy.deepcopy(body) + [ast.Return(ast.parse(returned, mode="eval").body)], decorator_list=[])
        code = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(code, str(path) + ":extracted:" + name, "exec"), namespace)
        return namespace[name]
    prepare = function("prepare", ["plan", "mode"], setup, "(model, inputs, encoder.column_policy)")
    optimizer = function("optimizer_factory", ["model"], [optimizer_node], "optimizer")
    original_scaler = function("original_scaler", [], [scaler_node], "scaler")
    scale_one_node = copy.deepcopy(scaler_node)
    scale_one_node.value.keywords.append(ast.keyword(arg="init_scale", value=ast.Constant(value=1.0)))
    scale_one = function("scale_one", [], [scale_one_node], "scaler")
    forward_fp16 = function("forward_fp16", ["model", "inputs"], [forward], "(outputs, loss)")
    forward_fp32 = copy.deepcopy(forward)
    forward_fp32.items[0].context_expr.keywords.append(ast.keyword(arg="enabled", value=ast.Constant(value=False)))
    forward_fp32 = function("forward_fp32", ["model", "inputs"], [forward_fp32], "(outputs, loss)")
    extracted = {name: {"source": ast.get_source_segment(source, node),
                        "ast_sha256": hashlib.sha256(ast.dump(node).encode()).hexdigest()}
                 for name, node in (("optimizer", optimizer_node), ("scaler", scaler_node), ("forward_loss", forward))}
    return {"prepare": prepare, "optimizer_factory": optimizer, "original_scaler": original_scaler,
            "scale_one": scale_one, "forward_fp16": forward_fp16, "forward_fp32": forward_fp32,
            "extracted": extracted}


def loss_observations(outputs, loss):
    result = {"outputs": {name: tensor_summary(value) for name, value in outputs.items()},
              "loss": tensor_summary(loss), "loss_requires_grad": bool(loss.requires_grad)}
    result["loss"]["value"] = float(loss.detach().item()) if result["loss"]["all_finite"] else None
    return result


def attach_backward_observer(loss):
    """Observe graph-node gradients without replacing gradients or operations."""
    import torch
    observations = {"policy": "nonmutating autograd node post-hooks; first observed node is not proof of root cause",
                    "nodes_observed": 0, "nodes_with_nonfinite": 0, "first_nonfinite_nodes": []}
    handles, visited, stack = [], set(), [loss.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        node_name = node.name()
        def hook(grad_inputs, grad_outputs, name=node_name):
            observations["nodes_observed"] += 1
            bad = []
            for direction, values in (("grad_inputs", grad_inputs), ("grad_outputs", grad_outputs)):
                for index, value in enumerate(values):
                    if value is not None and not bool(torch.isfinite(value).all()):
                        bad.append({"direction": direction, "index": index, "tensor": tensor_summary(value)})
            if bad:
                observations["nodes_with_nonfinite"] += 1
                if len(observations["first_nonfinite_nodes"]) < 16:
                    observations["first_nonfinite_nodes"].append({"node": name, "observed_order": observations["nodes_observed"], "gradients": bad})
            return None
        handles.append(node.register_hook(hook))
        stack.extend(next_node for next_node, _ in node.next_functions)
    observations["registered_nodes"] = len(visited)
    observations["maximum_recorded_nonfinite_nodes"] = 16
    return observations, handles


def run_cell(directory, mode, cell, base_model, base_inputs, rng, expected, program, source_bindings, *, cuda=True):
    """One forward/backward and at most one update, with no recovery retry."""
    import torch
    directory = Path(directory); directory.mkdir()
    phases, failures, hook_handles = [], [], []
    report = {"schema_version": 1, "mode": mode, "cell": cell, "iteration": 1, "seed": 42,
              "status": "unexpected_error", "numerical_passed": False,
              "source_bindings": source_bindings, "identity": expected,
              "phase_receipts": phases, "numerical_failures": failures, "exception": None,
              "update_evidence": {"attempted": 0, "successful": 0, "skipped": 0, "not_attempted": 1}}
    phase = "initialize"
    def retain(name, observed):
        return phases.append({"phase": name, **write_json(directory / ("%02d-%s.json" % (len(phases), name)),
            {"schema_version": 1, "mode": mode, "cell": cell, "phase": name, "observation": observed})})
    try:
        model, inputs = copy.deepcopy(base_model), {k: v.detach().clone() for k, v in base_inputs.items()}
        optimizer = program["optimizer_factory"](model)
        scaler = None if cell == "fp32" else program["original_scaler" if cell == "original_fp16" else "scale_one"]()
        report["settings"] = {"autocast_enabled": cell != "fp32", "autocast_dtype": "torch.float16" if cell != "fp32" else None,
            "scaler": scaler.state_dict() if scaler else None, "optimizer_class": type(optimizer).__name__,
            "optimizer_defaults": optimizer.defaults, "accumulation_batches": 1, "clipping": None,
            "loss": "sum(endpoint_logits.float().square().mean())", "backend": backend_settings()}
        report["settings"]["optimizer_initial_state_sha256"] = identity_digest(optimizer.state_dict())
        restore_rng(rng)
        report["initial_match"] = verify_initial_match(model, inputs, expected, cuda=cuda)
        names, groups = required_parameter_groups(model)
        report["expected_active_parameter_names"] = names
        report["required_parameter_paths"] = groups
        retain(phase, {"initial_match": report["initial_match"], "settings": report["settings"],
                       "expected_active_parameter_names": names, "required_parameter_paths": groups})
        successful = []
        hook_handles.append(optimizer.register_step_post_hook(lambda *unused: successful.append(True)))
        forward = program["forward_fp32" if cell == "fp32" else "forward_fp16"]
        phase = "forward"
        outputs, loss = forward(model, inputs)
        observed = loss_observations(outputs, loss)
        retain(phase, observed)
        if not observed["loss"]["all_finite"] or not all(x["all_finite"] for x in observed["outputs"].values()):
            failures.append("nonfinite_forward_loss_or_outputs")
        if not observed["loss_requires_grad"]:
            failures.append("loss_requires_grad_false")
        if not failures:
            phase = "scaled_loss"
            differentiated = scaler.scale(loss) if scaler else loss
            scaled_observation = tensor_summary(differentiated)
            retain(phase, {"loss": scaled_observation, "scaler_scale": scaler.get_scale() if scaler else None,
                           "requires_grad": bool(differentiated.requires_grad)})
            if not scaled_observation["all_finite"]:
                failures.append("nonfinite_scaled_loss")
            observer, handles = attach_backward_observer(differentiated)
            hook_handles.extend(handles)
            phase = "backward"
            try:
                differentiated.backward()
            finally:
                retain("backward_observer", observer)
            phase = "gradients_before_unscale"
            before = gradient_observations(model, groups)
            retain(phase, before)
            phase = "unscale"
            if scaler:
                scaler.unscale_(optimizer)
            retain(phase, {"calls": 1 if scaler else 0, "reason": "exactly once after the sole backward" if scaler else "no scaler in FP32 cell"})
            phase = "gradients_after_unscale"
            after = gradient_observations(model, groups)
            retain(phase, after)
            failures.extend(gradient_failures(after))
            phase = "original_guard"
            original_guard_passed = bool(after["gradient_tensor_count"] and not after["nonfinite_names"])
            retain(phase, {"passed": original_guard_passed, "condition": "nonempty gradient collection and every observed gradient finite",
                           "required_paths_are_additional_diagnostic_checks": True, "failure_reasons": list(failures)})
            # Preserve the original failing guard: no skip/backoff/update is
            # executed in a cell whose gradients failed. There is no retry.
            if not failures:
                phase = "update"
                report["update_evidence"].update(attempted=1, not_attempted=0)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if cuda:
                    torch.cuda.synchronize()
                report["update_evidence"].update(successful=len(successful), skipped=1 - len(successful))
                observed = parameter_and_optimizer_observations(model, optimizer, groups)
                retain(phase, {**observed, "actual_updates": report["update_evidence"],
                               "scaler_after": scaler.state_dict() if scaler else None})
                failures.extend(update_failures(observed, len(successful)))
                phase = "post_update_forward"
                with torch.no_grad():
                    post_outputs, post_loss = forward(model, inputs)
                post = loss_observations(post_outputs, post_loss)
                retain(phase, post)
                if not post["loss"]["all_finite"] or not all(x["all_finite"] for x in post["outputs"].values()):
                    failures.append("nonfinite_post_update_outputs_or_loss")
        report["status"] = "numerical_failure" if failures else "numerically_passed"
        report["numerical_passed"] = not failures
    except Exception as error:
        report["exception"] = {"phase": phase, "type": type(error).__name__, "message": str(error),
                               "traceback": traceback.format_exc()}
        retain("unexpected_error", report["exception"])
    finally:
        for handle in hook_handles:
            handle.remove()
    return report, write_json(directory / "CELL.json", report)


def backend_settings():
    import torch
    result = {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
              "float32_matmul_precision": torch.get_float32_matmul_precision(),
              "cudnn_enabled": torch.backends.cudnn.enabled, "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    for name in ("allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"):
        result[name] = getattr(torch.backends.cuda.matmul, name)
    for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled"):
        result[name] = getattr(torch.backends.cuda, name)()
    return result


def materialize_fixture(model, inputs, forward, *, cuda=True):
    """Materialize lazy parameters once, preserving the RNG for matched cells."""
    import torch
    lazy = [name for name, p in list(model.named_parameters()) + list(model.named_buffers())
            if isinstance(p, (torch.nn.parameter.UninitializedParameter, torch.nn.parameter.UninitializedBuffer))]
    if lazy:
        with torch.no_grad():
            forward(model, inputs)
    remaining = [name for name, p in list(model.named_parameters()) + list(model.named_buffers())
                 if isinstance(p, (torch.nn.parameter.UninitializedParameter, torch.nn.parameter.UninitializedBuffer))]
    if remaining:
        raise ValueError("Unmaterialized parameters remain: " + repr(remaining))
    return {"lazy_names_before": lazy, "lazy_names_after": remaining,
            "materialization_forward_count": int(bool(lazy)), "materialization_precision": "fp32" if lazy else None,
            "exact_original_initialization": not bool(lazy)}


def interpret(cells):
    by_mode = {}
    for mode in ("F2", "F3"):
        records = {row["cell"]: row for row in cells if row["mode"] == mode}
        if set(records) != set(CELLS) or any(row["status"] == "unexpected_error" for row in records.values()):
            result = "incomplete_or_unexpected_error"
        elif all(row["numerical_passed"] for row in records.values()):
            result = "original_failure_not_reproduced"
        elif not records["fp32"]["numerical_passed"]:
            result = "failure_not_specific_to_fp16_scaling"
        elif not records["fp16_scale1"]["numerical_passed"]:
            result = "fp16_path_requires_investigation"
        elif not records["original_fp16"]["numerical_passed"]:
            result = "pattern_consistent_with_scale_dependent_failure_not_root_cause_proof"
        else:
            result = "mixed_pattern_requires_review"
        by_mode[mode] = result
    return by_mode


def retain_initial_fixture(directory, mode, model, inputs, rng, materialization, clinical_policy, optimizer, scaler):
    """Write the same initial-state artifacts for real and mocked producer tests."""
    import torch
    directory = Path(directory)
    expected = initial_identity(model, inputs, rng)
    names, groups = required_parameter_groups(model)
    snapshot = directory / "INITIAL.pt"
    with snapshot.open("xb") as handle:
        torch.save({"model_state": cpu_copy(model.state_dict()), "inputs": cpu_copy(inputs), "rng_state": cpu_copy(rng)}, handle)
        handle.flush(); os.fsync(handle.fileno())
    record = {"mode": mode, "seed": 42, "identity": expected, "serialized_fixture": reference(snapshot),
              "model_state_inventory": structured_identity(model.state_dict()), "input_inventory": structured_identity(inputs),
              "rng_inventory": structured_identity(rng), "materialization": materialization,
              "original_scaler": scaler.state_dict(), "original_optimizer_defaults": optimizer.defaults,
              "optimizer_initial_state_sha256": identity_digest(optimizer.state_dict()),
              "clinical_policy": clinical_policy, "expected_active_parameter_names": names, "required_parameter_paths": groups}
    return {"identity": expected, "identity_file": write_json(directory / "INITIAL.json", record),
            "serialized_fixture": reference(snapshot)}


def make_summary(source_bindings, runtime, extracted):
    return {"schema_version": 1, "status": "failed_diagnostic", "purpose": "retained_cuda_fixture_diagnosis",
            "qualification_passed": False, "clinical_training_invocations": 0,
            "expected_cells": [mode + "/" + cell for mode in ("F2", "F3") for cell in CELLS],
            "source_bindings": source_bindings, "runtime": runtime, "mode_fixtures": {}, "cells": [],
            "extracted_original_statements": extracted, "automatic_retries": 0,
            "scope": "six fabricated token cells, each at most one optimizer update; no full CT/Swin or clinical fit"}


def finalize_summary(summary, output_dir, bindings_unchanged):
    summary["cell_counts"] = {status: sum(c["status"] == status for c in summary["cells"])
                              for status in ("numerically_passed", "numerical_failure", "unexpected_error")}
    summary["interpretation"] = interpret(summary["cells"])
    summary["all_cells_numerically_passed"] = all(c["numerical_passed"] for c in summary["cells"])
    summary["bindings_unchanged_after"] = bool(bindings_unchanged)
    actual_cells = [row["mode"] + "/" + row["cell"] for row in summary["cells"]]
    if actual_cells == summary["expected_cells"] and not summary["cell_counts"]["unexpected_error"] and bindings_unchanged:
        summary["status"] = "completed_diagnostic"
    write_json(Path(output_dir) / "DIAGNOSTIC.json", summary)
    return 0 if summary["status"] == "completed_diagnostic" else 2


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-source-dir", type=Path, required=True)
    p.add_argument("--original-fixture-path", type=Path, required=True)
    p.add_argument("--original-fixture-sha256", required=True)
    p.add_argument("--original-plan", type=Path, required=True)
    p.add_argument("--original-plan-sha256", required=True)
    p.add_argument("--expected-source-tree-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--container-image", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if (args.original_fixture_sha256, args.original_plan_sha256, args.expected_source_tree_sha256) != (
            ORIGINAL_FIXTURE_SHA256, ORIGINAL_PLAN_SHA256, ORIGINAL_SOURCE_SHA256):
        raise ValueError("Diagnostic is restricted to the exact retained budget-A fixture/source/plan")
    fixture = checked_file(args.original_fixture_path, args.original_fixture_sha256)
    plan_path = checked_file(args.original_plan, args.original_plan_sha256)
    source = args.model_source_dir.resolve()
    if ".." in args.model_source_dir.parts or args.model_source_dir.is_symlink() or source_tree_sha256(source) != args.expected_source_tree_sha256:
        raise ValueError("Original scientific source identity differs")
    if any(name == "trifusesurv2" or name.startswith("trifusesurv2.") for name in sys.modules):
        raise ValueError("Scientific package was imported before original source isolation")
    plan = json.loads(plan_path.read_text())
    if plan["source"]["source_tree_sha256"] != args.expected_source_tree_sha256:
        raise ValueError("Original plan source differs")
    previous_declaration = os.environ.get("TRIFUSESURV2_COMMIT_SHA")
    os.environ["TRIFUSESURV2_COMMIT_SHA"] = plan["source"]["commit_sha"]
    sys.path.insert(0, str(source / "src"))
    import torch
    from trifusesurv2.utils.provenance import source_identity
    import trifusesurv2.models.habitat_survival as original_model
    if not Path(original_model.__file__).resolve().is_relative_to(source) or source_identity() != plan["source_identity"]:
        raise ValueError("Observed imported source differs from original archive identity")
    if sys.executable != plan["environment"]["python"]:
        raise ValueError("Interpreter differs from retained worker environment")
    if args.container_image != plan["pinned_image"]:
        raise ValueError("Declared image differs from original pinned image")
    if torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name(0):
        raise ValueError("Exact one-H100 diagnostic allocation required")
    if ".." in args.output_dir.parts or args.output_dir.is_symlink():
        raise ValueError("Output path must be a new non-symlink directory")
    args.output_dir.mkdir(parents=False, exist_ok=False)
    program = compile_original_program(fixture)
    source_bindings = {"model_source": {"path": str(source), "identity": source_identity()},
                       "original_fixture": reference(fixture), "original_plan": reference(plan_path),
                       "diagnostic_files": [reference(Path(__file__)), reference(Path(__file__).with_name("diagnose_retained_cuda_fixture.py"))]}
    gpu = torch.cuda.get_device_properties(0)
    probe = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader,nounits"],
                           check=True, capture_output=True, text=True, timeout=30)
    runtime = {"python_executable": sys.executable, "python_version": sys.version,
               "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
               "gpu": {"name": gpu.name, "total_memory": gpu.total_memory, "uuid": str(gpu.uuid),
                       "visible_devices": torch.cuda.device_count(), "nvidia_smi": probe.stdout.strip()},
               "container_image_declaration": args.container_image, "backend": backend_settings(),
               "incoming_commit_declaration": previous_declaration,
               "original_source_commit_declaration": os.environ["TRIFUSESURV2_COMMIT_SHA"]}
    summary = make_summary(source_bindings, runtime, program["extracted"])
    write_json(args.output_dir / "STARTUP.json", summary)
    for mode in ("F2", "F3"):
        directory = args.output_dir / mode; directory.mkdir()
        try:
            model, inputs, clinical_policy = program["prepare"](plan, mode)
            # Match original construction order through optimizer and scaler.
            original_optimizer = program["optimizer_factory"](model)
            original_scaler = program["original_scaler"]()
            materialization = materialize_fixture(model, inputs, program["forward_fp32"])
            if not materialization["exact_original_initialization"]:
                raise ValueError("Exact original model unexpectedly contains lazy state; retain evidence and review before reproduction")
            rng = capture_rng()
            summary["mode_fixtures"][mode] = retain_initial_fixture(directory, mode, model, inputs, rng, materialization,
                                                                    clinical_policy, original_optimizer, original_scaler)
            expected = summary["mode_fixtures"][mode]["identity"]
            del original_optimizer, original_scaler
            for cell in CELLS:
                observed, ref = run_cell(directory / cell, mode, cell, model, inputs, rng, expected, program, source_bindings)
                summary["cells"].append({"mode": mode, "cell": cell, "status": observed["status"],
                    "numerical_passed": observed["numerical_passed"], "report": ref})
            del model, inputs, rng
            torch.cuda.empty_cache()
        except Exception as error:
            ref = write_json(directory / "PREPARATION_ERROR.json", {"mode": mode, "type": type(error).__name__,
                "message": str(error), "traceback": traceback.format_exc()})
            for cell in CELLS:
                if not any(row["mode"] == mode and row["cell"] == cell for row in summary["cells"]):
                    summary["cells"].append({"mode": mode, "cell": cell, "status": "unexpected_error",
                                             "numerical_passed": False, "report": ref})
    unchanged = (sha(fixture) == args.original_fixture_sha256 and sha(plan_path) == args.original_plan_sha256
        and source_tree_sha256(source) == args.expected_source_tree_sha256)
    return finalize_summary(summary, args.output_dir, unchanged)
