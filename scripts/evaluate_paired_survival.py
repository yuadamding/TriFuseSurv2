#!/usr/bin/env python3
"""Evaluate frozen, paired survival predictions, or prepare development support.

No training, data acquisition, source mutation, or automatic protocol freezing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True,
                        help="Literal Patient_ID,split,endpoint,observed_time,event CSV")
    parser.add_argument("--predictions", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--support-only", action="store_true",
                        help="Read only train/val outcome rows; never read predictions")
    parser.add_argument("--artifacts", type=Path,
                        help="Optional fit-key manifest mapping MODE/SEED to checkpoint_path and model_contract_path")
    args = parser.parse_args()
    import pandas as pd
    from trifusesurv2.evaluation.curves import file_sha256
    from trifusesurv2.evaluation.paired import (
        censoring_support_table, evaluate_predictions, read_literal_csv, validate_spec,
    )

    spec = json.loads(args.spec.read_text())
    outcomes = read_literal_csv(args.outcomes)
    if args.support_only:
        if args.predictions or args.artifacts:
            parser.error("--support-only cannot receive prediction/artifact files")
        ev = spec["evaluation"]
        horizons = ev.get("candidate_horizons", [ev["primary_horizon"], *ev["secondary_horizons"]])
        rows = censoring_support_table(outcomes, ev["endpoint"], horizons)
        report = {"schema_version": 1, "status": "development_support_only",
                  "scope": "train/validation outcomes; no held-out prediction comparison", "support": rows}
    else:
        validate_spec(spec)
        if spec["evaluation"].get("outcomes_file_sha256"):
            if file_sha256(args.outcomes) != spec["evaluation"]["outcomes_file_sha256"]:
                raise ValueError("outcome file hash differs from frozen protocol")
        if not args.predictions:
            parser.error("evaluation requires --predictions")
        predictions = pd.concat([read_literal_csv(path) for path in args.predictions], ignore_index=True)
        # A CLI invocation is a single specified endpoint/fold/split. Extra rows
        # are errors, not silently filtered into a more favorable cohort.
        if args.artifacts:
            artifacts = json.loads(args.artifacts.read_text())
            declared = predictions.groupby(["mode", "seed"])
            if set(artifacts) != {f"{a}/{b}" for a, b in declared.groups}:
                raise ValueError("artifact manifest does not exactly cover prediction fits")
            for (mode, seed), frame in declared:
                binding = artifacts[f"{mode}/{seed}"]
                for field, path_field in (("checkpoint_sha256", "checkpoint_path"),
                                           ("model_contract_sha256", "model_contract_path")):
                    path = Path(binding[path_field])
                    if not path.is_absolute():
                        path = args.artifacts.parent / path
                    if not frame[field].eq(file_sha256(path)).all():
                        raise ValueError(f"artifact hash mismatch for {mode}/{seed}/{field}")
        report = evaluate_predictions(predictions, outcomes, spec)
        report["artifact_identity_check"] = "verified_from_files" if args.artifacts else "declared_export_hashes_only"
        if not args.artifacts:
            report["status"] = "engineering_evaluation_unverified_artifacts"
        report["prediction_inputs"] = [{"path": str(path), "sha256": file_sha256(path)} for path in args.predictions]
        if args.artifacts:
            report["artifact_manifest_sha256"] = file_sha256(args.artifacts)
    report["study_spec_sha256"] = file_sha256(args.spec)
    report["outcomes_sha256"] = file_sha256(args.outcomes)
    from trifusesurv2.utils.provenance import source_identity
    import importlib.metadata
    report["source_identity"] = source_identity()
    report["dependencies"] = {name: importlib.metadata.version(name)
                              for name in ("numpy", "pandas", "scipy", "scikit-learn", "scikit-survival")}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / ("development_support.json" if args.support_only else "paired_evaluation.json")
    if output.exists():
        raise FileExistsError(f"preserve existing report: {output}")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.support_only:
        pd.DataFrame(report["support"]).to_csv(args.out_dir / "development_support.csv", index=False)
    print(json.dumps({"status": report["status"], "report": str(output), "sha256": file_sha256(output)}))


if __name__ == "__main__":
    main()
