"""Fail-closed postprocessing of the frozen, two-epoch F2/F3 pilot.

This module imports torch only after the receipt chain and file hashes pass.
An optimizer's anonymous state IDs do not identify the survival path. A map
produced by a separately qualified reconstruction of the actual model is
required; tensor shape or state-dictionary order is deliberately insufficient.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any


class Unavailable(ValueError):
    """Evidence is missing; no pass or failure may be inferred."""


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path, *, optional_nonfinite_fields=()) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=float if optional_nonfinite_fields else
                       lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {value}")))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    def scrub(item, location=()):
        if isinstance(item, float) and not math.isfinite(item):
            if len(location) == 1 and location[0] in optional_nonfinite_fields:
                return None
            raise ValueError(f"Nonfinite required JSON field: {'.'.join(map(str, location))}")
        if isinstance(item, dict):
            return {key: scrub(child, (*location, key)) for key, child in item.items()}
        if isinstance(item, list):
            return [scrub(child, (*location, index)) for index, child in enumerate(item)]
        return item
    value = scrub(value)
    return value


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"Missing or duplicate CSV columns: {path}")
        rows = list(reader)
        if any(None in row or None in row.values() for row in rows):
            raise ValueError(f"Malformed CSV row: {path}")
        return rows


def literal(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("IDs must be nonempty literal strings without surrounding whitespace")
    return value


def exact_ids(values, expected=None) -> list[str]:
    ids = [literal(value) for value in values]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty or duplicate patient inventory")
    if expected is not None:
        expected = exact_ids(expected)
        if set(ids) != set(expected):
            raise ValueError(f"Patient membership mismatch: missing={sorted(set(expected)-set(ids))}, extra={sorted(set(ids)-set(expected))}")
    return ids


def finite(value, label="value") -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite {label}")
    return number


def checked_file(path: Path, *, root: Path | None = None, digest: str | None = None) -> Path:
    path = Path(path).absolute()
    if ".." in path.parts or (root is not None and ".." in root.absolute().parts):
        raise ValueError("Parent traversal is not permitted in evidence or trusted-root paths")
    if not path.is_file():
        raise Unavailable(f"Missing evidence: {path}")
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError(f"Symlinks are not accepted: {path}")
    if root is not None and not path.is_relative_to(root.absolute()):
        raise ValueError(f"Evidence escapes trusted root: {path}")
    if digest is not None and sha256(path) != digest:
        raise ValueError(f"SHA256 mismatch: {path}")
    return path


def write_json(path: Path, value: dict):
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def source_record(path: Path) -> dict:
    return {"path": str(path.absolute()), "sha256": sha256(path), "bytes": path.stat().st_size}


def report_sources(*relative_paths) -> list[dict]:
    root = Path(__file__).resolve().parents[3]
    return [source_record(root / relative) for relative in relative_paths]


def optimizer_evidence(checkpoint: dict, mapping: dict, checkpoint_sha256: str, opportunities: int, *, reconstruction: dict) -> dict:
    """Inspect actual named survival states; no inference from anonymous IDs."""
    if (mapping.get("mapping_method") != "actual_model_parameter_groups"
            or mapping.get("checkpoint_sha256") != checkpoint_sha256
            or mapping.get("source_identity") != checkpoint.get("source_identity")):
        raise ValueError("Optimizer map must bind this checkpoint and actual model reconstruction/source")
    provenance = mapping.get("reconstruction_evidence")
    if not isinstance(provenance, dict) or not provenance.get("path") or not provenance.get("sha256"):
        raise Unavailable("Actual-model reconstruction evidence is not bound by the optimizer map")
    groups = checkpoint.get("optimizer_state", {}).get("param_groups", [])
    names = mapping.get("parameter_names_by_group", [])
    if len(groups) != len(names) or not groups:
        raise ValueError("Optimizer map group count differs")
    state = checkpoint["optimizer_state"].get("state", {})
    model_state = checkpoint.get("model_state", {})
    group_digest = hashlib.sha256(json.dumps(names, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (reconstruction.get("schema_version") != 1 or reconstruction.get("status") != "reconstructed_parameter_mapping"
            or reconstruction.get("checkpoint_sha256") != checkpoint_sha256
            or reconstruction.get("checkpoint_source_identity") != checkpoint.get("source_identity")
            or reconstruction.get("source_tree_sha256") != checkpoint.get("source_identity", {}).get("source_tree_sha256")
            or not reconstruction.get("source_tree_sha256")
            or reconstruction.get("mapping_method") != "actual_model_parameter_groups"
            or reconstruction.get("parameter_names_by_group_sha256") != group_digest):
        raise ValueError("Reconstruction evidence does not bind the actual checkpoint/source/parameter groups")
    grouping = reconstruction.get("optimizer", {})
    flat_names = [name for group in names for name in group]
    reconstructed = reconstruction.get("reconstruction", {})
    registered = reconstructed.get("named_parameters", [])
    if (grouping.get("grouping_method") != "actual source make_param_groups + live parameter object identity"
            or grouping.get("serialized_ids_match_reconstructed_optimizer") is not True
            or grouping.get("group_count") != len(groups) or grouping.get("parameter_count") != len(flat_names)
            or grouping.get("optimizer_states_present") != len(state)
            or not isinstance(registered, list) or len(registered) != len(set(registered))
            or set(registered) != set(flat_names) or reconstructed.get("lazy_materialization") != "strict_named_state_dict_load"):
        raise ValueError("Reconstruction evidence parameter inventory/grouping differs")
    all_names, all_ids, survival = [], [], []
    for group, group_names in zip(groups, names):
        if len(group["params"]) != len(group_names):
            raise ValueError("Optimizer map group length differs")
        for param_id, name in zip(group["params"], group_names):
            if name not in model_state:
                raise ValueError(f"Mapped parameter absent from model: {name}")
            all_names.append(name)
            all_ids.append(param_id)
            # The qualified v2 structured survival head includes shared and
            # endpoint residual paths. Other model families need a new audit.
            if name.startswith("habitat_model.survival_heads."):
                item = state.get(param_id)
                if not isinstance(item, dict):
                    raise ValueError(f"Missing optimizer state in trained survival head: {name}")
                step = finite(item.get("step"), "survival optimizer step")
                if step != int(step) or not 0 < step <= opportunities:
                    raise ValueError(f"Invalid survival optimizer update count: {name}")
                moments = item.get("exp_avg"), item.get("exp_avg_sq")
                if any(moment is None or tuple(moment.shape) != tuple(model_state[name].shape) for moment in moments):
                    raise ValueError(f"Survival optimizer moment shape differs: {name}")
                survival.append({"parameter": name, "successful_adam_updates": int(step),
                                 "nonzero_moment": any(bool(moment.ne(0).any()) for moment in moments)})
    if len(set(all_names)) != len(all_names) or len(set(all_ids)) != len(all_ids):
        raise ValueError("Duplicate parameter in optimizer map")
    parameter_count = sum(model_state[name].numel() for name in all_names)
    if reconstructed.get("parameters") != parameter_count or reconstructed.get("trainable_parameters") != parameter_count:
        raise ValueError("Reconstructed full-finetuning parameter counts differ")
    if not survival or not any(item["nonzero_moment"] for item in survival):
        raise ValueError("No nonzero gradient moments and successful updates in survival head")
    for path in ("habitat_model.survival_heads.shared.", "habitat_model.survival_heads.endpoint_residuals.OS."):
        if not any(item["parameter"].startswith(path) and item["nonzero_moment"] for item in survival):
            raise ValueError("No demonstrated updates in the primary OS shared/residual survival path")
    steps = [item["successful_adam_updates"] for item in survival]
    return {"survival_parameters": survival, "successful_updates_min": min(steps),
            "successful_updates_max": max(steps), "optimizer_step_opportunities": opportunities,
            "attempted_updates": None, "attempted_update_telemetry": "unavailable_in_frozen_trainer",
            "interpretation": "Adam state steps count successful optimizer updates, not AMP attempts; nonzero moments prove survival-path optimization, not clinical benefit."}


def selection_evidence(best: dict, last: dict, rows: list[dict], selected: str, *, epoch_metrics=None) -> dict:
    """The pilot has exactly two epochs; last wins export ties by source order."""
    if last.get("epoch") != 2 or best.get("epoch") not in (1, 2):
        raise ValueError("Unexpected two-epoch checkpoint epochs")
    b, end = finite(best.get("report_metric"), "best validation NLL"), finite(last.get("report_metric"), "last validation NLL")
    if b > end or (best["epoch"] == 2 and b != end):
        raise ValueError("Best checkpoint is inconsistent with minimum validation NLL")
    expected = "best" if b < end else "last"
    if selected != expected:
        raise ValueError(f"Exported checkpoint is {selected}; validation NLL requires {expected}")
    value = b if selected == "best" else end
    if not rows or any(row.get("validation_report_metric") != "nll"
                       or finite(row.get("validation_report_metric_value")) != value
                       or row.get("checkpoint") != selected for row in rows):
        raise ValueError("Selected risk metadata disagrees with checkpoint validation NLL")
    # If best was epoch two, the frozen files do not preserve epoch-one NLL.
    # A self-labelled 'best' checkpoint is insufficient to prove that comparison.
    trace = None
    if epoch_metrics and all("val_nll" in row for row in epoch_metrics):
        if [int(row["epoch"]) for row in epoch_metrics] != [1, 2]:
            raise ValueError("Validation trace must cover exactly epochs one and two")
        trace = [finite(row["val_nll"], "epoch validation NLL") for row in epoch_metrics]
        expected_best_epoch = 1 if trace[0] <= trace[1] else 2
        if best["epoch"] != expected_best_epoch or b != trace[expected_best_epoch - 1] or end != trace[1]:
            raise ValueError("Checkpoint NLL/epoch disagrees with full-precision epoch validation trace")
    if best["epoch"] == 2 and trace is None:
        raise Unavailable("Epoch-one validation NLL is not retained in metrics.csv; need an independently bound full-precision validation trace")
    return {"selected": selected, "best_validation_nll": b, "last_validation_nll": end,
            "rule": "minimum validation NLL; last wins export ties; best.pt retains first epoch on an epoch-NLL tie",
            "candidate_epochs": [best["epoch"], last["epoch"]], "full_precision_epoch_nll": trace}


def audit_pilot(*, training_plan: Path | None = None, execution_plan: Path | None = None,
                runtime_final: Path | None = None, trusted_root: Path | None = None,
                parameter_maps: dict[str, Path] | None = None) -> dict:
    criteria, inputs, contracts, stages, mode_data = [], {}, {}, [], {}

    def check(name, callback, *, required=True, mode=None):
        record = {"criterion": name, "required": required, "mode": mode}
        try:
            record.update(status="passed", evidence=callback())
        except Unavailable as error:
            record.update(status="blocked" if required else "unassessable", reason=str(error))
        except (ValueError, KeyError, TypeError, OSError, ImportError) as error:
            record.update(status="failed", reason=f"{type(error).__name__}: {error}")
        criteria.append(record)
        return record["status"] == "passed"

    def capture(path, label):
        path = checked_file(path)
        inputs[label] = source_record(path)
        return read_json(path)

    plan = capture(training_plan, "training_plan") if training_plan else None
    execution = capture(execution_plan, "execution_plan") if execution_plan else None
    if plan is None and execution is None:
        raise ValueError("At least one execution or training plan is required")
    settings = plan.get("expected_settings", {}) if plan else {}
    runtime = capture(runtime_final, "runtime_final") if runtime_final and runtime_final.exists() else None
    binding = {}

    def receipt_chain():
        nonlocal stages, binding
        if plan is None or runtime is None or trusted_root is None:
            raise Unavailable("Complete training plan, runtime FINAL, and trusted root are required for checkpoint inspection")
        if runtime.get("status") != "passed_paired_development":
            if runtime.get("status") == "failed":
                raise ValueError("Frozen execution failed: " + str(runtime.get("error")))
            raise Unavailable("Frozen execution has not completed successfully")
        if execution and runtime.get("plan_sha256") != inputs["execution_plan"]["sha256"]:
            raise ValueError("Runtime execution-plan hash differs")
        if runtime.get("training_plan_sha256") != inputs["training_plan"]["sha256"]:
            raise ValueError("Runtime training-plan hash differs")
        ref = runtime["training_receipt"]
        receipt_path = checked_file(Path(ref["path"]), root=trusted_root, digest=ref["sha256"])
        receipt = capture(receipt_path, "training_receipt")
        if receipt.get("status") != "passed_paired_development_training" or receipt.get("training_plan_sha256") != inputs["training_plan"]["sha256"]:
            raise ValueError("Training receipt does not bind successful execution of this plan")
        if receipt.get("modes") != ["F2", "F3"] or receipt.get("epochs_per_mode") != 2:
            raise ValueError("Paired training budget differs")
        stages = receipt["stage_receipts"]
        if [stage.get("fusion_mode") for stage in stages] != ["F2", "F3"]:
            raise ValueError("Expected one sequential F2 then F3 invocation")
        for stage in stages:
            if stage.get("invocations") != 1 or stage.get("returncode") != 0:
                raise ValueError("Training invocation failed or repeated")
            for item in stage["artifact_validation"]["artifacts"]:
                path = checked_file(Path(item["path"]), root=trusted_root, digest=item["sha256"])
                if path.stat().st_size != item["bytes"] or str(path) in binding:
                    raise ValueError("Artifact size mismatch or duplicate binding")
                binding[str(path)] = item
        if stages[0]["finished_at_utc"] > stages[1]["started_at_utc"]:
            raise ValueError("Paired execution intervals overlap")
        return {"training_receipt": inputs["training_receipt"], "artifacts": list(binding.values())}

    chain_ok = check("receipt_chain_and_artifact_integrity", receipt_chain)

    def execution_identity():
        if not chain_ok or runtime is None or execution is None or plan is None:
            raise Unavailable("Successful runtime receipt and both frozen execution/training plans are required")
        for key in ("source_commit", "run_id", "pinned_image"):
            if not execution.get(key) or runtime.get(key) != execution[key]:
                raise ValueError(f"Runtime {key} differs from frozen execution plan")
        if execution["source_commit"] != plan["source_identity"].get("commit_sha"):
            raise ValueError("Execution and scientific training source identities differ")
        if not runtime.get("pod_uid") or not runtime.get("pod_name"):
            raise Unavailable("Worker pod identity is unavailable")
        hardware = runtime.get("hardware", {})
        if (hardware.get("visible_device_count") != 1 or "H100" not in str(hardware.get("name", "")).upper()
                or hardware.get("compute_capability", [None])[0] != 9
                or hardware.get("total_memory_bytes", 0) < 75 * 1024 ** 3):
            raise ValueError("Worker hardware does not establish exactly one qualified H100")
        training = read_json(Path(inputs["training_receipt"]["path"]))
        if any(training.get("hardware", {}).get(key) != hardware.get(key) for key in ("name", "visible_device_count", "compute_capability", "total_memory_bytes")):
            raise ValueError("Training and worker hardware observations differ")
        return {"hardware": hardware, "pod_uid": runtime["pod_uid"], "pod_name": runtime["pod_name"],
                "pinned_image": runtime["pinned_image"], "source_commit": runtime["source_commit"]}
    check("single_h100_execution_identity", execution_identity)

    def bound(path):
        item = binding.get(str(path.absolute()))
        if item is None:
            raise Unavailable(f"Artifact is not hash-bound by successful training receipt: {path}")
        return checked_file(path, root=trusted_root, digest=item["sha256"])

    for mode in ("F2", "F3"):
        data = mode_data[mode] = {}

        def load_mode():
            if not chain_ok or plan is None:
                raise Unavailable("Successful receipt chain is unavailable")
            runs = [run for run in plan["runs"] if run["fusion_mode"] == mode]
            if len(runs) != 1:
                raise ValueError("Missing or duplicate mode in plan")
            experiment = Path(runs[0]["out_dir"]) / settings["exp_name"]
            fold = experiment / "fold_00"
            data.update(experiment=experiment, fold=fold)
            data["metrics"] = read_csv(bound(fold / "metrics.csv"))
            data["contract"] = read_json(bound(fold / "model_contract.json"))
            contracts[mode] = data["contract"]
            data["risks"] = read_csv(bound(experiment / "cv_test_risks_export.csv"))
            data["summary"] = read_json(bound(experiment / "cv_summary.json"), optional_nonfinite_fields=("mean_fold_test_c_index",))
            summary = data["summary"]
            if (summary.get("fusion_mode") != mode or summary.get("folds_run") != [0]
                    or summary.get("use_lora") is not False or summary.get("report_metric") != "nll"):
                raise ValueError("Final summary differs from frozen paired pilot recipe")
            suffix = data["summary"]["fold_export_suffixes"]
            if len(suffix) != 1 or suffix[0] not in ("best", "last"):
                raise ValueError("Invalid selected checkpoint suffix")
            data["selected"] = suffix[0]
            data["selected_risks"] = read_csv(bound(fold / f"test_risks_{suffix[0]}.csv"))
            return {"fold": str(fold), "selected": suffix[0]}

        loaded = check("completed_artifacts", load_mode, mode=mode)

        def need():
            if not loaded:
                raise Unavailable("Completed artifact inventory is unavailable")

        def epochs():
            need()
            if settings.get("epochs") != 2 or settings.get("resume"):
                raise ValueError("Frozen two-epoch, non-resumed training budget differs")
            if [int(row["epoch"]) for row in data["metrics"]] != [1, 2]:
                raise ValueError("Expected exactly two distinct completed epochs")
            return {"completed_epochs": [1, 2]}
        check("epoch_completion", epochs, mode=mode)

        def losses_risks():
            need()
            required = ("train_loss", "train_surv_loss", "train_surv_primary_loss", "train_surv_aux_loss",
                        "train_surv_os_nll", "train_surv_dss_nll", "train_surv_dfs_nll")
            losses = [{name: finite(row[name], name) for name in required} for row in data["metrics"]]
            for row in data["risks"] + data["selected_risks"]:
                if not 0 <= finite(row["risk_score"], "primary prediction") <= 1:
                    raise ValueError("Exported cumulative risk is outside [0,1]")
                horizon = finite(row["risk_horizon_days"], "risk horizon")
                if horizon <= 0 or horizon != finite(settings["risk_horizon_days"]):
                    raise ValueError("Risk horizon differs from frozen plan")
                if row.get("risk_endpoint") != settings["endpoint"]:
                    raise ValueError("Risk endpoint differs from frozen plan")
            return {"required_losses": losses, "primary_predictions": len(data["risks"]),
                    "optional_metrics": {"test_c_index": data["summary"].get("mean_fold_test_c_index"),
                                         "test_c_index_assessability": "finite" if data["summary"].get("mean_fold_test_c_index") is not None else "unassessable"}}
        check("finite_required_losses_and_predictions", losses_risks, mode=mode)

        def identities():
            need()
            contract = data["contract"]
            if contract["fusion_mode"] != mode or contract["source_identity"] != plan["source_identity"]:
                raise ValueError("Mode or source identity differs")
            plan_binding = {str(Path(item["path"]).absolute()): item["sha256"] for item in plan["bound_files"]}
            for split in ("train", "val", "test"):
                path = Path(settings["splits_dir"]) / "fold_00" / f"{split}_ids.txt"
                if str(path.absolute()) not in plan_binding:
                    raise ValueError("Split inventory absent from bound plan")
                declared = checked_file(path, root=trusted_root, digest=plan_binding[str(path.absolute())]).read_text().splitlines()
                exact_ids(contract[f"{split}_ids"], declared)
            inventories = [set(contract[f"{split}_ids"]) for split in ("train", "val", "test")]
            if any(inventories[a] & inventories[b] for a, b in ((0, 1), (0, 2), (1, 2))):
                raise ValueError("Patient overlap between splits")
            id_col = settings["id_col"]
            exact_ids([row[id_col] for row in data["risks"]], contract["test_ids"])
            exact_ids([row[id_col] for row in data["selected_risks"]], contract["test_ids"])
            aggregate = {row[id_col]: float(row["risk_score"]) for row in data["risks"]}
            if any(float(row["risk_score"]) != aggregate[row[id_col]] for row in data["selected_risks"]):
                raise ValueError("Aggregate and selected checkpoint risks differ")
            for setting, contract_key in (("meta_csv", "metadata_sha256"), ("radiomics_root", "radiomics_csv_sha256")):
                path = Path(settings[setting])
                if plan_binding.get(str(path.absolute())) != contract[contract_key]:
                    raise ValueError("Model input identity differs from plan")
                checked_file(path, root=trusted_root, digest=contract[contract_key])
            if (set(contract.get("region_manifest_sha256", {})) != set.union(*inventories)
                    or not set(contract["region_manifest_sha256"].values()) <= set(plan_binding.values())):
                raise ValueError("Region manifest identities are not plan-bound")
            metadata_path = Path(settings["meta_csv"])
            metadata = read_csv(metadata_path)
            exact_ids([row[id_col] for row in metadata], list(set.union(*inventories)))
            for row in metadata:
                manifest_path = Path(row["region_manifest_path"])
                if not manifest_path.is_absolute():
                    manifest_path = metadata_path.parent / manifest_path
                digest = contract["region_manifest_sha256"][row[id_col]]
                if plan_binding.get(str(manifest_path.absolute())) != digest:
                    raise ValueError("Per-patient manifest path/hash differs from frozen plan")
                manifest = read_json(checked_file(manifest_path, root=trusted_root, digest=digest))
                if manifest.get("patient_id") != row[id_col] or manifest.get("scan_id") != literal(row["scan_id"]):
                    raise ValueError("Metadata/manifest patient or acquisition identity differs")
            return {"split_counts": dict(zip(("train", "val", "test"), map(len, inventories))),
                    "input_content_scope": "metadata/radiomics/splits/manifests rehashed; each patient/scan and manifest path bound to frozen plan"}
        check("exact_patient_and_input_identities", identities, mode=mode)

        def checkpoint_numerics():
            need()
            import torch
            checkpoints = {}
            for suffix in ("best", "last"):
                path = bound(data["fold"] / f"{suffix}.pt")
                # Only explicitly trusted, successful-worker hash-bound files
                # reach this full-pickle operation (fitted encoders are stored).
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                if checkpoint.get("args", {}).get("fusion_mode") != mode:
                    raise ValueError("Checkpoint fusion mode differs")
                if checkpoint.get("args", {}).get("report_metric") != "nll":
                    raise ValueError("Checkpoint was not selected with validation NLL")
                checkpoint_settings = json.loads(json.dumps({key: checkpoint.get("args", {}).get(key) for key in settings}, allow_nan=False))
                if checkpoint_settings != settings:
                    raise ValueError("Checkpoint scientific settings differ from frozen plan")
                if any(checkpoint.get("source_identity", {}).get(key) != value for key, value in plan["source_identity"].items()):
                    raise ValueError("Checkpoint executable source identity differs from plan")
                model = checkpoint.get("model_state", {})
                if not model or any(not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all()) for tensor in model.values()):
                    raise ValueError("Missing or nonfinite model tensor")
                for state in checkpoint.get("optimizer_state", {}).get("state", {}).values():
                    for value in state.values():
                        if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                            raise ValueError("Nonfinite optimizer state")
                checkpoints[suffix] = checkpoint
            data["checkpoints"] = checkpoints
            return {suffix: {"sha256": binding[str((data["fold"] / f"{suffix}.pt").absolute())]["sha256"],
                             "epoch": checkpoint["epoch"]} for suffix, checkpoint in checkpoints.items()}
        check("finite_checkpoint_tensors", checkpoint_numerics, mode=mode)

        def optimization():
            if "checkpoints" not in data:
                raise Unavailable("Trusted checkpoint tensors have not been inspected")
            path = (parameter_maps or {}).get(mode)
            if path is None:
                raise Unavailable("Anonymous optimizer IDs cannot prove survival-path updates: actual-model parameter map required")
            mapping = capture(path, f"{mode}_parameter_map")
            ref = mapping.get("reconstruction_evidence", {})
            reconstruction = read_json(checked_file(Path(ref["path"]), root=trusted_root, digest=ref["sha256"]))
            for name in ("parameters", "trainable_parameters"):
                if reconstruction.get("reconstruction", {}).get(name) != data["contract"][name]:
                    raise ValueError("Reconstruction parameter count differs from model contract")
            count = len(data["contract"]["train_ids"])
            batch, accumulation = int(settings["batch_size"]), int(settings["grad_accumulation_steps"])
            opportunities = math.ceil(math.ceil(count / batch) / accumulation) * 2
            result = optimizer_evidence(data["checkpoints"]["last"], mapping,
                                        binding[str((data["fold"] / "last.pt").absolute())]["sha256"], opportunities,
                                        reconstruction=reconstruction)
            result["opportunities_per_epoch"] = opportunities // 2
            return result
        check("survival_path_optimizer_updates", optimization, mode=mode)

        def selection():
            if "checkpoints" not in data:
                raise Unavailable("Checkpoint validation-NLL evidence is unavailable")
            if settings.get("report_metric") != "nll" or settings.get("export_policy") != "best_val":
                raise ValueError("Frozen validation-NLL export rule differs")
            return selection_evidence(data["checkpoints"]["best"], data["checkpoints"]["last"], data["selected_risks"], data["selected"], epoch_metrics=data["metrics"])
        check("validation_nll_checkpoint_selection", selection, mode=mode)

    def comparability():
        if set(contracts) != {"F2", "F3"}:
            raise Unavailable("Both model contracts are required")
        normalized = []
        for contract in contracts.values():
            spec = contract["experiment_spec"]
            encoded = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            if hashlib.sha256(encoded).hexdigest() != contract["experiment_spec_sha256"]:
                raise ValueError("Experiment specification hash differs")
            shared = {key: value for key, value in spec["settings"].items() if key not in {"fusion_mode", "out_dir"}}
            if shared != settings:
                raise ValueError("Scientific settings differ from frozen plan")
            normalized.append({"settings": shared, "runtime": spec["runtime"],
                               **{key: contract[key] for key in ("parameters", "trainable_parameters", "metadata_sha256",
                                  "radiomics_csv_sha256", "region_manifest_sha256", "train_ids", "val_ids", "test_ids", "source_identity")}})
        if normalized[0] != normalized[1] or normalized[0]["trainable_parameters"] <= 0:
            raise ValueError("Shared scientific settings, input identities or parameter counts differ")
        return {"parameters": normalized[0]["parameters"], "trainable_parameters": normalized[0]["trainable_parameters"],
                "source_identity": plan["source_identity"]}
    check("paired_comparability", comparability)
    runtime_summary = {"epoch_seconds": {"status": "unassessable", "reason": "Frozen trainer has no full-precision epoch timing telemetry"},
                       "gpu_peak_memory": {"status": "unassessable"}, "ram_peak_memory": {"status": "unassessable"},
                       "mode_elapsed_seconds_including_setup_and_evaluation": {},
                       "reported_hardware": (runtime or {}).get("hardware"),
                       "shared_memory_preflight": (runtime or {}).get("shared_memory"),
                       "training_cost_estimate": "unassessable until completed stage wall times are available"}
    runtime_summary["epoch_train_and_evaluation_seconds_by_mode"] = {}
    for mode, data in mode_data.items():
        rows = data.get("metrics", [])
        if rows and all("epoch_train_and_evaluation_seconds" in row for row in rows):
            try:
                seconds = [finite(row["epoch_train_and_evaluation_seconds"]) for row in rows]
                if all(value > 0 for value in seconds):
                    runtime_summary["epoch_train_and_evaluation_seconds_by_mode"][mode] = seconds
            except (TypeError, ValueError):
                pass
    runtime_summary["epoch_timing_scope"] = "Future telemetry: epoch entry through training and train/validation evaluation; excludes SWA/checkpoint/GradCAM/final export. Historical missing telemetry stays unassessable."
    for stage in stages:
        try:
            elapsed = (datetime.fromisoformat(stage["finished_at_utc"].replace("Z", "+00:00")) - datetime.fromisoformat(stage["started_at_utc"].replace("Z", "+00:00"))).total_seconds()
            if elapsed >= 0:
                runtime_summary["mode_elapsed_seconds_including_setup_and_evaluation"][stage["fusion_mode"]] = elapsed
        except (KeyError, TypeError, ValueError):
            pass
    criteria.append({"criterion": "runtime_capacity", "required": False, "mode": None,
                     "status": "unassessable", "evidence": runtime_summary})
    required = [item["status"] for item in criteria if item["required"]]
    status = "failed" if "failed" in required else "blocked" if "blocked" in required else "passed"
    return {"schema_version": 1, "status": status, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "two-epoch development engineering acceptance; no convergence or clinical effectiveness claim",
            "inputs": inputs, "criteria": criteria, "runtime_summary": runtime_summary,
            "postprocessor": source_record(Path(__file__)),
            "postprocessor_sources": report_sources("src/trifusesurv2/validation/pilot.py", "scripts/audit_pilot_completion.py")}


def write_pilot_report(report: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / "pilot_acceptance.json", report)
    write_json(out_dir / "pilot_runtime_summary.json", report["runtime_summary"])
    lines = [f"# Pilot acceptance: {report['status'].upper()}", "", report["scope"], "",
             "Missing telemetry is not evidence of successful training. The execution bundle was not modified.", "",
             "| Mode | Criterion | Result | Evidence / limitation |", "| --- | --- | --- | --- |"]
    for item in report["criteria"]:
        lines.append(f"| {item['mode'] or 'paired'} | {item['criterion']} | {item['status']} | {str(item.get('reason', 'See JSON evidence')).replace('|', '/')} |")
    (out_dir / "pilot_acceptance.md").write_text("\n".join(lines) + "\n")
