#!/usr/bin/env python3
"""Qualify one H100 with synthetic measurements; optionally run a bound F2/F3 plan.

This does not qualify real patient inputs or establish prognostic performance.
It never installs software, downloads data, retries training, or selects a grid.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))


def require_one_h100(torch_module=None):
    if torch_module is None:
        import torch as torch_module
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() != 1:
        raise RuntimeError("Qualification requires exactly one visible CUDA GPU.")
    gpu = torch_module.cuda.get_device_properties(0)
    if "H100" not in gpu.name.upper() or gpu.major != 9 or gpu.total_memory < 75 * 1024**3:
        raise RuntimeError("Qualification requires one H100, compute capability 9.x, with at least 75 GiB.")
    return {"name": gpu.name, "total_memory_bytes": gpu.total_memory,
            "compute_capability": [gpu.major, gpu.minor], "visible_device_count": 1,
            "torch_version": str(torch_module.__version__), "cuda_build": torch_module.version.cuda}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scientific_phantom_checks(grid_config, radiomics_config, *, rtol, atol):
    import numpy as np
    import SimpleITK as sitk
    import radiomics
    from radiomics.featureextractor import RadiomicsFeatureExtractor
    from trifusesurv2.preprocessing.compartments import ReferenceGeometry, build_compartments
    from trifusesurv2.preprocessing.region_io import canonical_json_sha256, validate_radiomics_config
    from trifusesurv2.preprocessing.sampling import CommonGridPolicy, full_grid_geometry, sample_common_grid

    validate_radiomics_config(radiomics_config)
    if set(radiomics_config.get("imageType", {})) != {"Original"}:
        raise ValueError("Qualification requires an explicit Original-only PyRadiomics configuration.")
    if (radiomics_config.get("setting", {}).get("normalize", False)
            or radiomics_config["imageType"]["Original"].get("normalize", False)):
        raise ValueError("Crop-global intensity normalization is incompatible with this invariance qualification.")
    policy = CommonGridPolicy.from_dict(grid_config)
    geometry = ReferenceGeometry((64, 64, 64), (1., 1., 1.))
    z, y, x = np.indices(geometry.shape_dhw)
    pt = (z - 24)**2 + (y - 24)**2 + (x - 24)**2 <= 9
    nearby = (z - 24)**2 + (y - 24)**2 + (x - 34)**2 <= 4
    remote = (z - 50)**2 + (y - 48)**2 + (x - 52)**2 <= 4
    normalized = (.5 + .2 * np.sin(x * .7) + .15 * np.cos(y * .9)).astype(np.float32)
    image = sitk.GetImageFromArray(normalized)
    image.SetSpacing(geometry.spacing_xyz)
    extractor = RadiomicsFeatureExtractor(radiomics_config)
    shape = tuple(size + 8 for size in full_grid_geometry(geometry, policy)[0].shape_dhw)

    def sample(ln, output_shape=shape, start=None):
        masks, _ = build_compartments(pt, ln, np.ones_like(pt), geometry,
            patient_id="synthetic", scan_id="synthetic", contour_source="synthetic",
            contour_version="h100-qualification-v1", valid_tissue_source="synthetic full FOV")
        return sample_common_grid({"radiomics_image": image}, masks, geometry, policy, output_shape,
                                  crop_start_dhw=start)

    baseline = sample(nearby)
    cases = {"baseline": baseline, "distant_ln": sample(nearby | remote),
             "integer_crop": sample(nearby, start=np.asarray(baseline.metadata["crop_start_dhw"]) + (1, -1, 2)),
             "padding": sample(nearby, output_shape=tuple(size + 12 for size in shape))}
    report = {}
    for region in ("pt_intra", "pt_peri"):
        measurements, features = {}, {}
        for name, sampled in cases.items():
            mask = sampled.masks[region]
            if not mask.any():
                raise ValueError(f"Synthetic {region} disappeared under the supplied policy.")
            indices = np.argwhere(mask) + np.asarray(sampled.metadata["crop_start_dhw"])
            values = sitk.GetArrayFromImage(sampled.images["radiomics_image"])[mask]
            measurements[name] = (indices, values)
            extracted = extractor.execute(sampled.images["radiomics_image"], sampled.mask_images[region], label=1)
            vector = {key: float(value) for key, value in extracted.items() if key.startswith("original_")}
            if not vector or not np.isfinite(list(vector.values())).all():
                raise ValueError(f"PyRadiomics did not return finite Original features for {name}/{region}.")
            features[name] = vector
        reference = features["baseline"]
        for name in cases:
            np.testing.assert_array_equal(measurements[name][0], measurements["baseline"][0])
            np.testing.assert_array_equal(measurements[name][1], measurements["baseline"][1])
            if features[name].keys() != reference.keys():
                raise ValueError("PyRadiomics feature inventories differ across phantom cases.")
            keys = sorted(reference)
            np.testing.assert_allclose([features[name][key] for key in keys], [reference[key] for key in keys], rtol=rtol, atol=atol)
        report[region] = {"feature_count": len(reference), "features": features,
                          "sampled_voxels": len(measurements["baseline"][1])}
    return {"scope": "synthetic measurement invariance, not real-input qualification",
            "radiomics_config_sha256": canonical_json_sha256(radiomics_config),
            "policy_id": policy.policy_id, "intensities": "synthetic normalized CT in [0,1]", "pyradiomics_version": radiomics.__version__,
            "rtol": rtol, "atol": atol, "regions": report}


def _model_check(device):
    """Actual model contract; CPU use is engineering verification, not GPU qualification."""
    import torch
    from trifusesurv2.models.contour_habitat_survival import ContourAwareHabitatSurvivalModel
    from trifusesurv2.models.swinunetr_shared_roi_token_backbone import ContourAwareROITokenBackbone
    from trifusesurv2.schema import PHYSICAL_GEOMETRY_FEATURES

    torch.manual_seed(1701)
    image = torch.rand(2, 1, 32, 32, 32, device=device)
    clinical = torch.randn(2, 3, 4, device=device)
    radiomics = torch.randn(2, 4, 5, device=device)
    regions = torch.zeros(2, 4, 32, 32, 32, device=device)
    regions[:, 0, 12:16, 12:16, 12:16] = 1
    regions[:, 1, 8:12, 12:16, 12:16] = 1
    results = {}
    for mode in ("F2", "F3"):
        torch.manual_seed(1701)
        backbone = ContourAwareROITokenBackbone(
            img_size=(32, 32, 32), feature_size=12, depths=(1, 1, 1, 1), num_heads=(3, 6, 12, 24),
            token_dim=16, token_mlp_dropout=0, token_dropout=0, drop_rate=0, attn_drop_rate=0,
            dropout_path_rate=0, require_verified_regions=True,
        )
        model = ContourAwareHabitatSurvivalModel(
            backbone=backbone, fusion_mode=mode, num_time_bins=3, time_bin_width_days=120,
            model_dim=16, num_heads=4, transformer_layers=1, dropout=0,
            clinical_token_dim=4, radiomics_token_dim=5,
        ).to(device).train()
        output = model(image, clinical, radiomics,
                       radiomics_presence=torch.tensor([[1, 1, 0, 0]] * 2, device=device),
                       region_masks=regions, region_validity=torch.ones(2, 4, dtype=torch.bool, device=device),
                       region_states=torch.tensor([[0, 0, 1, 1]] * 2, device=device),
                       physical_geometry=torch.zeros(2, 2 * len(PHYSICAL_GEOMETRY_FEATURES), device=device))
        if not isinstance(output, dict) or set(output) != {"OS", "DSS", "DFS"}:
            raise RuntimeError("Unexpected multitask model output contract.")
        if not all(value.device == image.device and tuple(value.shape) == (2, 3) and torch.isfinite(value).all()
                   for value in output.values()):
            raise RuntimeError("Model outputs are on the wrong device, incorrectly shaped or nonfinite.")
        sum(value.square().mean() for value in output.values()).backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("Model backward produced missing/nonfinite gradients.")
        state = hashlib.sha256()
        for name, value in model.state_dict().items():
            state.update(name.encode() + str(tuple(value.shape)).encode() + value.detach().cpu().numpy().tobytes())
        results[mode] = {"parameters": sum(parameter.numel() for parameter in model.parameters()),
                         "initialized_state_sha256": state.hexdigest(), "finite_backward": True,
                         "device": str(image.device), "endpoint_shapes": {key: list(value.shape) for key, value in output.items()}}
        del model, backbone, output, gradients
    if image.is_cuda:
        torch.cuda.synchronize()
    if results["F2"] != results["F3"]:
        raise RuntimeError("F2/F3 initialization or parameter counts differ under identical seeds.")
    return results


def gpu_model_check():
    return _model_check("cuda:0")


def verify_bound_files(records):
    bound = {}
    for record in records:
        path = Path(record["path"])
        if not path.is_absolute() or str(path.resolve()) in bound or sha256(path) != record["sha256"]:
            raise ValueError("Training plan contains a duplicate, relative or changed bound file.")
        bound[str(path.resolve())] = record["sha256"]
    return bound


def normalized_settings(args):
    from trifusesurv2.utils.experiment import canonical_experiment_spec
    settings = canonical_experiment_spec(args)["settings"]
    return {key: value for key, value in settings.items() if key not in {"fusion_mode", "out_dir"}}


def validate_training_plan(plan, *, qualification):
    from trifusesurv2.multimodal_survival.train import parse_args
    from trifusesurv2.utils.provenance import source_contract_identity
    from trifusesurv2.utils.warmstart import launcher_warmstart_settings, warmstart_fingerprints
    from trifusesurv2.preprocessing.identifiers import resolve_identifier_column

    if set(plan) != {"schema_version", "source_identity", "common_args", "expected_settings", "expected_warmstart", "bound_files", "runs"} or plan["schema_version"] != 1:
        raise ValueError("Training requires a complete schema_version=1 bound plan.")
    if plan["source_identity"] != source_contract_identity():
        raise ValueError("Training plan source identity differs from the worker.")
    common = plan["common_args"]
    if not isinstance(common, list) or not all(isinstance(value, str) for value in common):
        raise ValueError("common_args must be an explicit argument list.")
    flags = {value.split("=", 1)[0] for value in common if value.startswith("--")}
    required = {"--meta_csv", "--id_col", "--epochs", "--seed", "--img_size", "--cv_folds", "--debug_fold", "--device", "--no_resume"}
    if not required <= flags or flags & {"--fusion_mode", "--out_dir", "--resume"}:
        raise ValueError("Plan must explicitly bind data, shape, seed, fold and budget; mode/output belong only in runs.")
    if [run.get("fusion_mode") for run in plan["runs"]] != ["F2", "F3"]:
        raise ValueError("Plan must contain exactly F2 then F3, sequentially.")
    bound = verify_bound_files(plan["bound_files"])
    if not qualification.get("policy_id") or not qualification.get("radiomics_config_sha256"):
        raise ValueError("Training requires the successfully qualified grid and radiomics configuration identities.")
    # parse_args can deserialize a configuration checkpoint. Resolve/hash it
    # using the tensor-free helper before allowing that parser to run.
    warm_args = launcher_warmstart_settings({}, ["--align_swin_cfg_from_contour_warmstart", *common])
    preparse_warm = warmstart_fingerprints(warm_args)
    if preparse_warm != plan["expected_warmstart"]:
        raise ValueError("Warm-start selection differs from the bound training plan.")
    for item in [*preparse_warm["folds"].values(), preparse_warm["config_checkpoint"]]:
        if item and item["exists"] and item["path"] not in bound:
            raise ValueError("Warm-start files must be hash-bound before checkpoint deserialization.")
    parsed = []
    for run in plan["runs"]:
        if set(run) != {"fusion_mode", "out_dir"} or not Path(run["out_dir"]).is_absolute() or Path(run["out_dir"]).exists():
            raise ValueError("Each run needs a new absolute output directory.")
        argv = [*common, "--fusion_mode", run["fusion_mode"], "--out_dir", run["out_dir"]]
        old_argv = sys.argv
        try:
            sys.argv = ["train", *argv]
            args = parse_args()
        finally:
            sys.argv = old_argv
        if args.exp_name in {"", ".", ".."} or Path(args.exp_name).name != args.exp_name or "/" in args.exp_name or "\\" in args.exp_name:
            raise ValueError("exp_name must be one nonempty relative path component inside its output root.")
        if any(value and not Path(value).is_absolute() for value in (
                args.meta_csv, args.radiomics_root, args.splits_dir, args.splits_csv,
                args.contour_warmstart_ckpt, args.contour_warmstart_dir)):
            raise ValueError("Training input paths must be absolute to avoid worker-directory ambiguity.")
        if str(Path(args.meta_csv).resolve()) not in bound:
            raise ValueError("Training metadata must be hash-bound before inspection.")
        with Path(args.meta_csv).open(newline="") as handle:
            metadata = csv.DictReader(handle)
            args.id_col = resolve_identifier_column(metadata.fieldnames, args.id_col)
            manifest_paths = []
            for row in metadata:
                declared = row.get(args.region_manifest_col)
                if not declared:
                    raise ValueError("Verified metadata requires a region manifest for every row.")
                path = Path(declared)
                manifest_paths.append(path if path.is_absolute() else Path(args.meta_csv).parent / path)
        if normalized_settings(args) != plan["expected_settings"]:
            raise ValueError("Resolved semantic settings differ from the approved training plan.")
        if args.device != "cuda:0" or args.ddp or args.data_parallel or args.comparison_recipe != "principal" or args.region_mode != "verified" or not args.use_radiomics:
            raise ValueError("Training plan must use one CUDA device and the verified principal recipe.")
        if args.epochs < 1 or args.cv_folds < 1 or not (-1 <= args.debug_fold < args.cv_folds):
            raise ValueError("Training requires a positive epoch/fold budget and a valid selected fold.")
        inputs = [Path(args.meta_csv), *manifest_paths]
        if args.use_radiomics:
            inputs.append(Path(args.radiomics_root))
        if args.splits_csv:
            inputs.append(Path(args.splits_csv))
        else:
            inputs.extend(Path(args.splits_dir) / f"fold_{fold:02d}" / f"{part}_ids.txt"
                          for fold in range(args.cv_folds) for part in ("train", "val", "test"))
        warm = warmstart_fingerprints(args)
        if warm != plan["expected_warmstart"]:
            raise ValueError("Warm-start selection differs from the bound training plan.")
        inputs.extend(Path(item["path"]) for item in [*warm["folds"].values(), warm["config_checkpoint"]]
                      if item and item["exists"])
        if any(str(path.resolve()) not in bound for path in inputs):
            raise ValueError("Training metadata, radiomics, split and selected warm-start files must be hash-bound.")
        # Full support/payload validation remains the trainer's real-input gate.
        # Bind that input's sampling/configuration to what this worker measured.
        for path in manifest_paths:
            manifest = json.loads(path.read_text())
            sampling = manifest.get("sampling", {})
            if sampling.get("policy") != "common_grid" or sampling.get("policy_id") != qualification["policy_id"]:
                raise ValueError("Training region sampling differs from the qualified common-grid policy.")
            if manifest.get("image", {}).get("geometry", {}).get("shape_dhw") != list(args.img_size):
                raise ValueError("Bound image shape differs from the explicit training image size.")
        with Path(args.radiomics_root).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or any(row.get("radiomics_config_sha256") != qualification["radiomics_config_sha256"] for row in rows):
            raise ValueError("Training radiomics configuration differs from the qualified extraction configuration.")
        parsed.append((args, argv))
    roots = [Path(item[0].out_dir).resolve() for item in parsed]
    if roots[0] == roots[1] or roots[0] in roots[1].parents or roots[1] in roots[0].parents:
        raise ValueError("F2/F3 outputs must be separate.")
    return parsed


def execute_training_plan(plan, output, *, qualification):
    parsed = validate_training_plan(plan, qualification=qualification)
    results, reference_counts = [], None
    for args, argv in parsed:
        verify_bound_files(plan["bound_files"])
        from trifusesurv2.utils.warmstart import warmstart_fingerprints
        if warmstart_fingerprints(args) != plan["expected_warmstart"]:
            raise ValueError("Warm-start selection changed before the next paired run.")
        command = [sys.executable, "-m", "trifusesurv2.multimodal_survival.train", *argv]
        env = dict(os.environ, PYTHONPATH=str(PACKAGE / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
        with (output / f"training_{args.fusion_mode}.log").open("x") as log:
            subprocess.run(command, cwd=PACKAGE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        folds = [args.debug_fold] if args.debug_fold >= 0 else list(range(args.cv_folds))
        counts = {}
        for fold in folds:
            contract_path = Path(args.out_dir) / args.exp_name / f"fold_{fold:02d}" / "model_contract.json"
            contract = json.loads(contract_path.read_text())
            settings = {key: value for key, value in contract["experiment_spec"]["settings"].items()
                        if key not in {"fusion_mode", "out_dir"}}
            if settings != plan["expected_settings"]:
                raise RuntimeError("Training artifact semantic settings differ from the bound plan.")
            counts[str(fold)] = [contract["parameters"], contract["trainable_parameters"]]
        if reference_counts is not None and counts != reference_counts:
            raise RuntimeError("Paired F2/F3 parameter counts differ.")
        reference_counts = counts
        results.append({"fusion_mode": args.fusion_mode, "command": command, "fold_parameter_counts": counts})
    return results


def run_sampling_tests(output):
    test_file = PACKAGE / "tests/test_common_grid.py"
    if not test_file.is_file():
        raise RuntimeError("The source-bound common-grid regression inventory is missing.")
    env = dict(os.environ, PYTHONPATH=str(PACKAGE / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    log_path = output / "common_grid_tests.log"
    with log_path.open("x") as log:
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", test_file.name, "-v"],
                       cwd=PACKAGE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    count = re.search(r"Ran (\d+) tests? in", log_path.read_text())
    if count is None or int(count.group(1)) < 1:
        raise RuntimeError("Common-grid qualification did not execute any regressions.")
    return {"test_file": str(test_file), "sha256": sha256(test_file), "passed": True,
            "test_count": int(count.group(1)),
            "scope": "synthetic common-grid CPU/SimpleITK tests inside the H100 worker"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-config-json", required=True)
    parser.add_argument("--radiomics-config-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--training-plan-json")
    parser.add_argument("--declared-container-image", default="")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-7)
    args = parser.parse_args(argv)
    gpu = require_one_h100()  # No input/config/output access before this gate.
    if not all(math.isfinite(value) and value >= 0 for value in (args.rtol, args.atol)):
        raise ValueError("Feature tolerances must be finite and nonnegative.")
    from trifusesurv2.utils.provenance import source_identity
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "running", "gpu": gpu, "source": source_identity(),
               "python": sys.version, "platform": platform.platform(),
               "python_executable": sys.executable, "uid": os.getuid(), "gid": os.getgid(),
               "output_uid": output.stat().st_uid, "output_gid": output.stat().st_gid,
               "declared_container_image": args.declared_container_image,
               "scope": "synthetic CUDA and measurement qualification; real-input qualification remains separate"}
    receipt_path = output / "qualification.json"
    try:
        receipt["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "monai", "SimpleITK", "numpy", "scipy", "pyradiomics")}
        grid = json.loads(Path(args.grid_config_json).read_text())
        config = json.loads(Path(args.radiomics_config_json).read_text())
        receipt["config_files"] = {str(Path(path).resolve()): sha256(path) for path in (args.grid_config_json, args.radiomics_config_json)}
        receipt["sampling_regressions"] = run_sampling_tests(output)
        receipt["scientific_checks"] = scientific_phantom_checks(grid, config, rtol=args.rtol, atol=args.atol)
        receipt["cuda_model_checks"] = gpu_model_check()
        if args.training_plan_json:
            receipt["training_plan_sha256"] = sha256(args.training_plan_json)
            plan = json.loads(Path(args.training_plan_json).read_text())
            receipt["training"] = execute_training_plan(plan, output, qualification=receipt["scientific_checks"])
        receipt["status"] = "passed"
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
