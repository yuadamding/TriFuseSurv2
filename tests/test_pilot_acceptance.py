"""Fabricated CPU artifacts only: these tests do not qualify the real pilot."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from trifusesurv2.validation.pilot import (
    Unavailable, audit_pilot, checked_file, exact_ids, optimizer_evidence,
    selection_evidence, sha256, write_json, write_pilot_report, read_json,
)
from trifusesurv2.validation.cohort_quality import write_csv


def checkpoint_fixture():
    model = torch.nn.Module()
    model.shared = torch.nn.Linear(2, 2)
    model.endpoint_residuals = torch.nn.ModuleDict({"OS": torch.nn.Linear(2, 1)})
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        model.endpoint_residuals["OS"](model.shared(torch.ones(1, 2))).sum().backward()
        optimizer.step()
    names = ["habitat_model.survival_heads." + name for name, _ in model.named_parameters()]
    checkpoint = {"source_identity": {"commit_sha": "fabricated", "source_tree_sha256": "fabricated_tree"},
                  "model_state": dict(zip(names, model.state_dict().values())),
                  "optimizer_state": optimizer.state_dict()}
    mapping = {"mapping_method": "actual_model_parameter_groups", "checkpoint_sha256": "fixture_hash",
               "source_identity": checkpoint["source_identity"], "parameter_names_by_group": [names],
               "reconstruction_evidence": {"path": "fabricated.json", "sha256": "fixture_reconstruction"}}
    return checkpoint, mapping


def proof_fixture(checkpoint, mapping):
    names = mapping["parameter_names_by_group"]
    parameters = sum(value.numel() for value in checkpoint["model_state"].values())
    return {"schema_version": 1, "status": "reconstructed_parameter_mapping", "checkpoint_sha256": mapping["checkpoint_sha256"],
            "checkpoint_source_identity": checkpoint["source_identity"], "source_tree_sha256": checkpoint["source_identity"]["source_tree_sha256"],
            "mapping_method": "actual_model_parameter_groups",
            "parameter_names_by_group_sha256": hashlib.sha256(json.dumps(names, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "reconstruction": {"named_parameters": list(checkpoint["model_state"]), "lazy_materialization": "strict_named_state_dict_load",
                               "parameters": parameters, "trainable_parameters": parameters},
            "optimizer": {"group_count": len(names), "parameter_count": sum(map(len, names)),
                          "optimizer_states_present": len(checkpoint["optimizer_state"]["state"]),
                          "grouping_method": "actual source make_param_groups + live parameter object identity",
                          "serialized_ids_match_reconstructed_optimizer": True}, "fixture_only": True}


def test_optimizer_proves_successful_survival_updates_without_equating_amp_opportunities():
    checkpoint, mapping = checkpoint_fixture()
    result = optimizer_evidence(checkpoint, mapping, "fixture_hash", 92, reconstruction=proof_fixture(checkpoint, mapping))
    assert result["successful_updates_min"] == 2
    assert result["optimizer_step_opportunities"] == 92
    assert result["attempted_updates"] is None


def test_positive_adam_steps_in_other_parameters_are_insufficient():
    checkpoint, mapping = checkpoint_fixture()
    checkpoint["model_state"] = {name.replace("survival_heads", "localization_heads"): value for name, value in checkpoint["model_state"].items()}
    mapping["parameter_names_by_group"] = [[name.replace("survival_heads", "localization_heads") for name in mapping["parameter_names_by_group"][0]]]
    with pytest.raises(ValueError, match="survival head"):
        optimizer_evidence(checkpoint, mapping, "fixture_hash", 92, reconstruction=proof_fixture(checkpoint, mapping))


@pytest.mark.parametrize("fault", ["missing_state", "zero_moments", "wrong_hash", "duplicate_name", "wrong_count", "no_reconstruction"])
def test_optimizer_mapping_and_nonzero_moments_fail_closed(fault):
    checkpoint, mapping = checkpoint_fixture()
    if fault == "missing_state":
        checkpoint["optimizer_state"]["state"].clear()
    elif fault == "zero_moments":
        for item in checkpoint["optimizer_state"]["state"].values():
            item["exp_avg"].zero_(); item["exp_avg_sq"].zero_()
    elif fault == "wrong_hash":
        mapping["checkpoint_sha256"] = "different"
    elif fault == "duplicate_name":
        mapping["parameter_names_by_group"][0][1] = mapping["parameter_names_by_group"][0][0]
    elif fault == "wrong_count":
        mapping["parameter_names_by_group"][0].pop()
    else:
        mapping.pop("reconstruction_evidence")
    with pytest.raises(ValueError):
        optimizer_evidence(checkpoint, mapping, "fixture_hash", 92, reconstruction=proof_fixture(checkpoint, mapping))


def test_arbitrary_reconstruction_receipt_cannot_establish_parameter_identity():
    checkpoint, mapping = checkpoint_fixture()
    with pytest.raises(ValueError, match="Reconstruction evidence"):
        optimizer_evidence(checkpoint, mapping, "fixture_hash", 92, reconstruction={"fixture_only": True})


def selected_rows(value, suffix):
    return [{"validation_report_metric": "nll", "validation_report_metric_value": str(value), "checkpoint": suffix}]


def test_nll_selection_obeys_lower_is_better_and_last_tie_order():
    best, last = {"epoch": 1, "report_metric": .3}, {"epoch": 2, "report_metric": .5}
    assert selection_evidence(best, last, selected_rows(.3, "best"), "best")["selected"] == "best"
    with pytest.raises(ValueError, match="requires best"):
        selection_evidence(best, last, selected_rows(.5, "last"), "last")
    last["report_metric"] = .3
    assert selection_evidence(best, last, selected_rows(.3, "last"), "last")["selected"] == "last"


def test_self_labelled_best_epoch_two_cannot_prove_missing_epoch_one_nll():
    checkpoint = {"epoch": 2, "report_metric": .3}
    with pytest.raises(Unavailable, match="Epoch-one"):
        selection_evidence(checkpoint, checkpoint, selected_rows(.3, "last"), "last")


def test_future_full_precision_epoch_trace_closes_selection_history_gap():
    checkpoint = {"epoch": 2, "report_metric": .3}
    trace = [{"epoch": 1, "val_nll": .4}, {"epoch": 2, "val_nll": .3}]
    assert selection_evidence(checkpoint, checkpoint, selected_rows(.3, "last"), "last", epoch_metrics=trace)["full_precision_epoch_nll"] == [.4, .3]
    trace[0]["val_nll"] = .3
    with pytest.raises(ValueError, match="validation trace"):
        selection_evidence(checkpoint, checkpoint, selected_rows(.3, "last"), "last", epoch_metrics=trace)


def test_literal_ids_and_duplicate_rejection():
    assert exact_ids(["NA", "001", "1"], ["1", "NA", "001"]) == ["NA", "001", "1"]
    for values in (["001", "001"], ["1", "NA"], [" 001", "NA", "1"]):
        with pytest.raises(ValueError):
            exact_ids(values, ["NA", "001", "1"])


def test_optional_cindex_nan_is_unassessable_but_required_fields_reject_nonfinite(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text('{"mean_fold_test_c_index": NaN, "risk_horizon_days": 730}')
    assert read_json(path, optional_nonfinite_fields=("mean_fold_test_c_index",))["mean_fold_test_c_index"] is None
    with pytest.raises(ValueError, match="Nonfinite"):
        read_json(path)
    path.write_text('{"mean_fold_test_c_index": NaN, "risk_horizon_days": NaN}')
    with pytest.raises(ValueError, match="required JSON"):
        read_json(path, optional_nonfinite_fields=("mean_fold_test_c_index",))


def test_pending_plan_writes_blocked_report_without_importing_or_loading_checkpoint(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.json"
    write_json(plan, {"run_id": "synthetic-pending", "source_commit": "fabricated"})
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Pending audit must not read checkpoints"))
    report = audit_pilot(execution_plan=plan)
    assert report["status"] == "blocked"
    assert not any(item["status"] == "passed" for item in report["criteria"] if item["required"])
    output = tmp_path / "report"
    write_pilot_report(report, output)
    assert json.loads((output / "pilot_acceptance.json").read_text())["status"] == "blocked"
    with pytest.raises(FileExistsError):
        write_pilot_report(report, output)


def test_untrusted_paths_and_changed_bytes_are_rejected(tmp_path):
    root = tmp_path / "approved"; root.mkdir()
    file = root / "file"; file.write_text("original")
    digest = sha256(file)
    assert checked_file(file, root=root, digest=digest) == file
    file.write_text("changed")
    with pytest.raises(ValueError, match="SHA256"):
        checked_file(file, root=root, digest=digest)
    link = root / "link"; link.symlink_to(file)
    with pytest.raises(ValueError, match="Symlinks"):
        checked_file(link, root=root)
    external = tmp_path / "outside"; external.write_text("data")
    with pytest.raises(ValueError, match="escapes"):
        checked_file(external, root=root)
    with pytest.raises(ValueError, match="traversal"):
        checked_file(root / ".." / "outside", root=root)


def complete_fixture(root):
    checkpoint, mapping = checkpoint_fixture()
    source = checkpoint["source_identity"]
    ids = {"train": ["train-001"], "val": ["val-NA"], "test": ["001", "NA"]}
    split_dir = root / "splits" / "fold_00"; split_dir.mkdir(parents=True)
    paths = []
    for split, members in ids.items():
        path = split_dir / f"{split}_ids.txt"; path.write_text("\n".join(members) + "\n"); paths.append(path)
    meta, rad = root / "meta.csv", root / "rad.csv"
    write_csv(rad, [{"patient_id": pid, "fixture_feature": 1} for part in ids.values() for pid in part])
    paths.extend((meta, rad))
    manifests = {}
    for part in ids.values():
        for pid in part:
            path = root / f"manifest-{pid}.json"; write_json(path, {"patient_id": pid, "scan_id": "scan-" + pid, "fixture_only": True})
            paths.append(path); manifests[pid] = sha256(path)
    write_csv(meta, [{"patient_id": pid, "scan_id": "scan-" + pid, "region_manifest_path": str(root / f"manifest-{pid}.json")} for part in ids.values() for pid in part])
    settings = {"exp_name": "fixture", "epochs": 2, "resume": False, "batch_size": 1, "grad_accumulation_steps": 8,
                "splits_dir": str(split_dir.parent), "meta_csv": str(meta), "radiomics_root": str(rad), "id_col": "patient_id",
                "report_metric": "nll", "export_policy": "best_val", "risk_horizon_days": 730, "endpoint": "OS"}
    plan_path = root / "training_plan.json"
    plan = {"expected_settings": settings, "source_identity": source,
            "bound_files": [{"path": str(path), "sha256": sha256(path)} for path in paths],
            "runs": [{"fusion_mode": mode, "out_dir": str(root / mode)} for mode in ("F2", "F3")]}
    write_json(plan_path, plan)
    stages, maps = [], {}
    for index, mode in enumerate(("F2", "F3")):
        experiment = root / mode / "fixture"; fold = experiment / "fold_00"; fold.mkdir(parents=True)
        spec = {"settings": {**settings, "fusion_mode": mode, "out_dir": str(root / mode)}, "runtime": {"fixture_cpu": True}}
        contract = {"fusion_mode": mode, "source_identity": source, "trainable_parameters": 9, "parameters": 9,
                    "experiment_spec": spec,
                    "experiment_spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    "metadata_sha256": sha256(meta), "radiomics_csv_sha256": sha256(rad), "region_manifest_sha256": manifests,
                    **{f"{split}_ids": values for split, values in ids.items()}}
        write_json(fold / "model_contract.json", contract)
        losses = {key: 1 for key in ("train_loss", "train_surv_loss", "train_surv_primary_loss", "train_surv_aux_loss",
                                     "train_surv_os_nll", "train_surv_dss_nll", "train_surv_dfs_nll")}
        write_csv(fold / "metrics.csv", [{"epoch": epoch, **losses} for epoch in (1, 2)])
        for suffix, epoch, value in (("best", 1, .3), ("last", 2, .5)):
            ck = copy.deepcopy(checkpoint)
            ck.update(epoch=epoch, report_metric=value, args=spec["settings"])
            torch.save(ck, fold / f"{suffix}.pt")
        risks = [{"patient_id": pid, "risk_score": .2, "risk_horizon_days": 730, "risk_endpoint": "OS"} for pid in ids["test"]]
        write_csv(experiment / "cv_test_risks_export.csv", risks)
        write_csv(fold / "test_risks_best.csv", [{**row, **selected_rows(.3, "best")[0]} for row in risks])
        write_json(experiment / "cv_summary.json", {"fold_export_suffixes": ["best"], "fusion_mode": mode,
                                                   "folds_run": [0], "use_lora": False, "report_metric": "nll"})
        artifacts = [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in experiment.rglob("*") if path.is_file()]
        stages.append({"fusion_mode": mode, "invocations": 1, "returncode": 0,
                       "started_at_utc": f"2026-01-01T00:0{index}:00+00:00", "finished_at_utc": f"2026-01-01T00:0{index+1}:00+00:00",
                       "artifact_validation": {"artifacts": artifacts}})
        bound_map = copy.deepcopy(mapping)
        bound_map["checkpoint_sha256"] = sha256(fold / "last.pt")
        proof = root / f"{mode}-reconstruction.json"; write_json(proof, proof_fixture(checkpoint, bound_map))
        bound_map["reconstruction_evidence"] = {"path": str(proof), "sha256": sha256(proof)}
        maps[mode] = root / f"{mode}-map.json"; write_json(maps[mode], bound_map)
    training_final = root / "training_FINAL.json"
    # Fabricated hardware metadata tests report parsing; all actual compute is CPU.
    hardware = {"name": "FABRICATED H100", "visible_device_count": 1, "compute_capability": [9, 0], "total_memory_bytes": 80 * 1024**3}
    write_json(training_final, {"status": "passed_paired_development_training", "training_plan_sha256": sha256(plan_path),
                                "modes": ["F2", "F3"], "epochs_per_mode": 2, "stage_receipts": stages, "hardware": hardware})
    execution = root / "execution_plan.json"
    execution_fields = {"run_id": "fabricated_cpu_fixture", "source_commit": "fabricated", "pinned_image": "fabricated_fixture"}
    write_json(execution, execution_fields)
    runtime = root / "runtime_FINAL.json"
    write_json(runtime, {"status": "passed_paired_development", "training_plan_sha256": sha256(plan_path),
                         "training_receipt": {"path": str(training_final), "sha256": sha256(training_final)},
                         "plan_sha256": sha256(execution), **execution_fields, "pod_uid": "fabricated", "pod_name": "fabricated", "hardware": hardware})
    return dict(training_plan=plan_path, execution_plan=execution, runtime_final=runtime, trusted_root=root, parameter_maps=maps)


def test_complete_synthetic_receipt_chain_exercises_all_required_gates(tmp_path):
    report = audit_pilot(**complete_fixture(tmp_path))
    assert report["status"] == "passed", [(item["criterion"], item.get("reason")) for item in report["criteria"] if item["status"] != "passed"]
    assert report["runtime_summary"]["epoch_seconds"]["status"] == "unassessable"
    assert report["runtime_summary"]["mode_elapsed_seconds_including_setup_and_evaluation"] == {"F2": 60, "F3": 60}


def test_changed_checkpoint_rejected_before_any_pickle_load(tmp_path, monkeypatch):
    kwargs = complete_fixture(tmp_path)
    with (tmp_path / "F2/fixture/fold_00/last.pt").open("ab") as handle:
        handle.write(b"changed")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Must not unpickle after integrity failure"))
    assert audit_pilot(**kwargs)["status"] == "failed"


def test_complete_training_without_actual_optimizer_map_remains_blocked(tmp_path):
    kwargs = complete_fixture(tmp_path)
    kwargs.pop("parameter_maps")
    report = audit_pilot(**kwargs)
    assert report["status"] == "blocked"
    assert all(item["status"] == "blocked" for item in report["criteria"] if item["criterion"] == "survival_path_optimizer_updates")
