"""Study and input binding for baseline execution; drafts fail before fitting."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from trifusesurv2.preprocessing.identifiers import read_identifier_csv, resolve_identifier_column
from trifusesurv2.utils.provenance import source_identity
from .cox import fit_cox_baseline, serializable_model, survival_probabilities, survival_target
from .features import (GEOMETRY_COLUMNS, GEOMETRY_STATES, attach_geometry, clinical_columns,
                       object_sha256, sha256_file)


def write_json(path: Path, record) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_splits(directory: Path) -> tuple[dict, dict]:
    splits, hashes, seen = {}, {}, set()
    for split in ("train", "val", "test"):
        path = directory / f"{split}_ids.txt"
        identifiers = path.read_text().splitlines()
        if (not identifiers or any(not pid.strip() or pid != pid.strip() for pid in identifiers)
                or len(identifiers) != len(set(identifiers))):
            raise ValueError(f"Empty or duplicate literal IDs in {split} inventory.")
        if set(identifiers) & seen:
            raise ValueError("Patient overlap between training, validation and test splits.")
        seen.update(identifiers)
        splits[split], hashes[split] = identifiers, sha256_file(path)
    return splits, hashes


def validate_protocol(spec: dict, modes: list[str]) -> tuple[dict, dict]:
    from trifusesurv2.evaluation.paired import require_reference, validate_spec

    if spec.get("status") != "frozen":
        raise ValueError("Study specification is not frozen; no baseline fitting is allowed.")
    endpoint = spec.get("endpoint_definition", {})
    if endpoint.get("status") != "confirmed":
        raise ValueError("Endpoint origin, units, event coding and censoring must be confirmed.")
    # The baseline uses the shared evaluator's endpoint mapping and declaration.
    for key in ("time_origin", "time_unit", "event_definition", "censoring_rule"):
        if not endpoint.get(key):
            raise ValueError(f"Missing confirmed endpoint definition: {key}.")
    evaluation, baseline = validate_spec(spec), spec.get("baselines", {})
    require_reference()
    if evaluation.get("endpoint") != "OS":
        raise ValueError("These baseline contracts currently support the primary OS endpoint.")
    if baseline.get("status") != "frozen" or baseline.get("selection") != "validation_ibs":
        raise ValueError("Baseline recipe and validation-IBS selection must be frozen.")
    if "C1" in modes:
        geometry = baseline.get("geometry", {})
        if geometry.get("status") != "qualified_for_development" or not geometry.get("accepted_limitations"):
            raise ValueError("C1 requires an explicit geometry qualification and accepted source limitations.")
    clinical_columns(baseline.get("clinical", {}), baseline.get("fusion_clinical_columns", []))
    return evaluation, baseline


def run(args) -> dict:
    specification_path = Path(args.study_spec)
    spec = json.loads(specification_path.read_text())
    evaluation, baseline = validate_protocol(spec, args.modes)
    # Resolve confirmed types before CSV inference can merge literal codes or
    # turn nonblank categories/invalid numeric literals such as "NA" into missing.
    numeric, categorical = clinical_columns(baseline["clinical"], baseline["fusion_clinical_columns"])
    from trifusesurv2.evaluation.curves import write_survival_curves
    from trifusesurv2.evaluation.paired import outcome_inventory_sha256
    metadata_path = Path(args.metadata_csv)
    if sha256_file(metadata_path) != baseline["metadata_sha256"]:
        raise ValueError("Metadata checksum differs from the frozen baseline/neural cohort.")
    splits, split_hashes = load_splits(Path(args.split_dir))
    if split_hashes != baseline["split_files_sha256"]:
        raise ValueError("Split checksums differ from the frozen baseline/neural cohort.")
    frame = read_identifier_csv(
        metadata_path, identifier_columns=(args.id_column,), float_precision="round_trip",
        converters={name: lambda value: value for name in numeric + categorical},
    )
    identifier = resolve_identifier_column(frame.columns, args.id_column)
    if frame[identifier].isna().any() or frame[identifier].duplicated().any():
        raise ValueError("Baseline metadata contains missing or duplicate patient IDs.")
    frame = frame.set_index(identifier)
    eligible = set().union(*(set(values) for values in splits.values()))
    if set(frame.index) != eligible:
        raise ValueError("Baseline metadata must exactly equal the neural eligible split union.")
    time_column = spec["endpoint_definition"].get("time_column", "OS.TIME")
    event_column = spec["endpoint_definition"].get("event_column", "OS.EVENT")
    split_for_patient = {pid: split for split, ids in splits.items() for pid in ids}
    outcome_inventory = frame[[time_column, event_column]].rename(
        columns={time_column: "observed_time", event_column: "event"}).reset_index(drop=True)
    outcome_inventory["Patient_ID"] = frame.index.tolist()
    outcome_inventory["split"] = [split_for_patient[pid] for pid in frame.index]
    outcome_inventory["endpoint"] = evaluation["endpoint"]
    if outcome_inventory_sha256(outcome_inventory, evaluation["endpoint"]) != evaluation["outcome_inventory_sha256"]:
        raise ValueError("Baseline outcome inventory differs from the frozen evaluator cohort and censoring reference.")
    geometry_identity = None
    if "C1" in args.modes:
        if not args.manifest_inventory:
            raise ValueError("C1 requires --manifest-inventory with bound paths and SHA-256 values.")
        geometry = baseline["geometry"]
        geometry_identity = sha256_file(args.manifest_inventory)
        if geometry_identity != geometry["manifest_inventory_sha256"]:
            raise ValueError("Geometry inventory differs from the frozen qualified inventory.")
        frame = attach_geometry(
            frame, inventory=json.loads(Path(args.manifest_inventory).read_text()),
            manifest_column=geometry["manifest_column"], metadata_directory=metadata_path.parent,
            scan_column=geometry.get("scan_column"),
        )
    grid = np.asarray(sorted(set(evaluation["ibs_times"] + [evaluation["primary_horizon"]]
                                 + evaluation.get("secondary_horizons", []))), dtype=float)
    if (not np.isfinite(grid).all() or np.any(grid < 0)
            or evaluation.get("risk_definition", "1-S(primary_horizon)") != "1-S(primary_horizon)"):
        raise ValueError("Invalid evaluation grid or unsupported scalar-risk definition.")
    dependencies = {name: version(name) for name in ("scikit-survival", "scikit-learn", "numpy", "scipy", "pandas")}
    root = Path(args.out_dir)
    outputs = {}
    for mode in args.modes:
        model_numeric = numeric + (list(GEOMETRY_COLUMNS) if mode == "C1" else [])
        model_categorical = categorical + (list(GEOMETRY_STATES) if mode == "C1" else [])
        model, preprocessor, selection = fit_cox_baseline(
            frame.loc[splits["train"]], frame.loc[splits["val"]], numeric=model_numeric,
            categorical=model_categorical, time_column=time_column, event_column=event_column,
            alpha_grid=baseline["alpha_grid"], validation_times=evaluation["ibs_times"],
            ties=baseline["ties"], n_iter=baseline["n_iter"], tol=baseline["tol"],
        )
        mode_directory = root / mode
        mode_directory.mkdir()
        contract = {
            "schema_version": 1, "mode": mode, "seed": "deterministic", "endpoint": "OS",
            "study_spec_sha256": sha256_file(specification_path), "source": source_identity(),
            "dependencies": dependencies, "metadata_sha256": sha256_file(metadata_path),
            "split_files_sha256": split_hashes, "manifest_inventory_sha256": geometry_identity,
            "selection": selection, "clinical_policy": baseline["clinical"],
            "clinical_csv_ingestion": {
                "categorical_columns": categorical,
                "categorical_conversion": "verbatim strings before pandas numeric/NA inference",
                "categorical_value_policy": preprocessor.to_dict()["categorical_value_policy"],
                "numeric_columns": numeric,
                "numeric_conversion": "verbatim strings before existing finite numeric validation",
                "float_precision": "round_trip",
            },
            "geometry_features": list(GEOMETRY_COLUMNS) if mode == "C1" else [],
            "volume_transform": "log1p(volume_mm3)" if mode == "C1" else None,
            "node_instances": "excluded; node_count and largest_node_volume are unqualified",
            "training_ids_sha256": object_sha256(splits["train"]),
            "validation_ids_sha256": object_sha256(splits["val"]),
            "test_ids_sha256": object_sha256(splits["test"]),
        }
        contract_path = mode_directory / "model_contract.json"
        write_json(contract_path, contract)
        model_path = mode_directory / "selected_model.json"
        checkpoint = serializable_model(model, preprocessor)
        checkpoint["model_contract_sha256"] = sha256_file(contract_path)
        write_json(model_path, checkpoint)
        predictions = {}
        # Held-out labels are not supplied to fitting or alpha selection. Their
        # pre-fit inventory hash binds the fixed cohort, not a tuning criterion.
        for split, ids in splits.items():
            subset = frame.loc[ids]
            target = survival_target(subset, time_column, event_column)
            probabilities = survival_probabilities(model, preprocessor.transform(subset), grid)
            path = mode_directory / f"{split}_survival_curves.csv"
            write_survival_curves(
                path, patient_ids=ids, observed_times=target["time"], events=target["event"],
                evaluation_times=grid, survival_probabilities=probabilities, split=split,
                fold=evaluation["fold"], seed="deterministic", mode=mode, endpoint="OS",
                checkpoint_sha256=sha256_file(model_path), model_contract_sha256=sha256_file(contract_path),
                risk_horizon=evaluation["primary_horizon"], metadata={"study_spec_sha256": sha256_file(specification_path)},
            )
            predictions[split] = {"path": str(path.resolve()), "sha256": sha256_file(path), "patients": len(ids)}
        outputs[mode] = {"selected_alpha": selection["selected_alpha"],
                         "model_sha256": sha256_file(model_path),
                         "model_contract_sha256": sha256_file(contract_path), "predictions": predictions}
    return {"status": "complete", "schema_version": 1, "outputs": outputs,
            "study_spec_sha256": sha256_file(specification_path),
            "interpretation": "Deterministic baseline fits; scientific comparison requires the shared paired evaluator."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-spec", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--manifest-inventory")
    parser.add_argument("--id-column", default="Patient_ID")
    parser.add_argument("--modes", nargs="+", choices=("C0", "C1"), default=["C0", "C1"])
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    try:
        if len(set(args.modes)) != len(args.modes):
            raise ValueError("Baseline modes must be unique.")
        receipt = run(args)
    except Exception as error:
        write_json(output / "baseline_receipt.json", {"status": "blocked_or_failed", "error": str(error),
                   "error_type": type(error).__name__, "scientific_acceptance": False})
        raise
    write_json(output / "baseline_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "out_dir": str(output.resolve())}))
    return 0
