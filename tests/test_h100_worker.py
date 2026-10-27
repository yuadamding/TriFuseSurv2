"""Local control-flow tests; actual CUDA/PyRadiomics qualification runs on the H100."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


PACKAGE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("qualify_h100", PACKAGE / "scripts/qualify_h100.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class H100WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="trifusesurv2-h100-worker-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.qualification = {"policy_id": "qualified-policy", "radiomics_config_sha256": "qualified-radiomics"}

    def fake_torch(self, *, available=True, count=1, name="NVIDIA H100 80GB HBM3", major=9, memory=80 * 1024**3):
        properties = types.SimpleNamespace(name=name, major=major, minor=0, total_memory=memory)
        return types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: available, device_count=lambda: count,
                                       get_device_properties=lambda index: properties),
            __version__="test-only", version=types.SimpleNamespace(cuda="test-only"),
        )

    def test_exactly_one_sufficient_h100_is_required(self):
        identity = worker.require_one_h100(self.fake_torch())
        self.assertEqual(identity["visible_device_count"], 1)
        for change in ({"available": False}, {"count": 0}, {"count": 2},
                       {"name": "NVIDIA A100"}, {"major": 8}, {"memory": 74 * 1024**3}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                worker.require_one_h100(self.fake_torch(**change))

    def test_actual_model_contract_on_cpu_is_only_engineering_verification(self):
        result = worker._model_check("cpu")
        self.assertEqual(result["F2"], result["F3"])
        self.assertEqual(result["F2"]["device"], "cpu")
        self.assertEqual(result["F2"]["endpoint_shapes"], {"OS": [2, 3], "DSS": [2, 3], "DFS": [2, 3]})
        self.assertTrue(result["F2"]["finite_backward"])

    def test_rejected_hardware_does_not_read_inputs_or_create_output(self):
        output = self.root / "no-output"
        with mock.patch.object(worker, "require_one_h100", side_effect=RuntimeError("GPU rejected")), \
                mock.patch.object(Path, "read_text", side_effect=AssertionError("input read")):
            with self.assertRaisesRegex(RuntimeError, "GPU rejected"):
                worker.main(["--grid-config-json", "/missing/grid.json", "--radiomics-config-json", "/missing/rad.json",
                             "--out-dir", str(output)])
        self.assertFalse(output.exists())

    def make_plan(self):
        from trifusesurv2.multimodal_survival.train import parse_args
        from trifusesurv2.utils.provenance import source_contract_identity
        from trifusesurv2.utils.warmstart import warmstart_fingerprints

        manifest = self.root / "regions.json"
        manifest.write_text(json.dumps({"sampling": {"policy": "common_grid", "policy_id": self.qualification["policy_id"]}, "image": {"geometry": {"shape_dhw": [32, 32, 32]}}}))
        metadata = self.root / "metadata.csv"
        metadata.write_text(f"Patient_ID,region_manifest_path\n001,{manifest}\n")
        radiomics = self.root / "radiomics.csv"
        radiomics.write_text("Patient_ID,pt_intra__original_firstorder_Mean,radiomics_config_sha256\n001,0.5,qualified-radiomics\n")
        split = self.root / "split.csv"
        split.write_text("Patient_ID,fold,split\n001,0,train\n002,0,val\n003,0,test\n")
        common = ["--meta_csv", str(metadata), "--id_col", "Patient_ID", "--radiomics_root", str(radiomics),
                  "--splits_csv", str(split), "--epochs", "1", "--seed", "42", "--img_size", "32", "32", "32",
                  "--cv_folds", "1", "--debug_fold", "0", "--device", "cuda:0", "--no_resume",
                  "--no_align_swin_cfg_from_contour_warmstart", "--exp_name", "paired"]
        runs = [{"fusion_mode": mode, "out_dir": str(self.root / mode)} for mode in ("F2", "F3")]
        with mock.patch.object(sys, "argv", ["train", *common, "--fusion_mode", "F2", "--out_dir", runs[0]["out_dir"]]):
            args = parse_args()
        return {"schema_version": 1, "source_identity": source_contract_identity(), "common_args": common,
                "expected_settings": worker.normalized_settings(args), "expected_warmstart": warmstart_fingerprints(args),
                "bound_files": [{"path": str(path), "sha256": worker.sha256(path)}
                                for path in (metadata, radiomics, split, manifest)], "runs": runs}

    def test_bound_plan_keeps_identical_settings_and_separate_fresh_outputs(self):
        plan = self.make_plan()
        parsed = worker.validate_training_plan(plan, qualification=self.qualification)
        self.assertEqual([args.fusion_mode for args, _ in parsed], ["F2", "F3"])
        self.assertEqual(worker.normalized_settings(parsed[0][0]), worker.normalized_settings(parsed[1][0]))
        for run in plan["runs"]:
            self.assertFalse(Path(run["out_dir"]).exists())

    def test_plan_rejects_changed_semantics_source_inputs_and_output_reuse(self):
        plan = self.make_plan()
        for change in ("settings", "source", "order", "budget", "overlap"):
            changed = copy.deepcopy(plan)
            if change == "settings":
                changed["common_args"] += ["--v2_num_heads", "4"]
            elif change == "source":
                changed["source_identity"]["source_tree_sha256"] = "different"
            elif change == "order":
                changed["runs"].reverse()
            elif change == "budget":
                index = changed["common_args"].index("--epochs")
                del changed["common_args"][index:index + 2]
            else:
                changed["runs"][1]["out_dir"] = str(Path(changed["runs"][0]["out_dir"]) / "nested")
            with self.subTest(change=change), self.assertRaises(ValueError):
                worker.validate_training_plan(changed, qualification=self.qualification)
        Path(plan["runs"][0]["out_dir"]).mkdir()
        with self.assertRaisesRegex(ValueError, "new absolute"):
            worker.validate_training_plan(plan, qualification=self.qualification)
        Path(plan["runs"][0]["out_dir"]).rmdir()
        Path(plan["bound_files"][0]["path"]).write_text("changed metadata\n")
        with self.assertRaisesRegex(ValueError, "changed bound file"):
            worker.validate_training_plan(plan, qualification=self.qualification)

    def test_manifest_and_warmstart_resolution_are_bound_before_training(self):
        plan = self.make_plan()
        changed = copy.deepcopy(plan)
        changed["bound_files"].pop()
        with self.assertRaisesRegex(ValueError, "must be hash-bound"):
            worker.validate_training_plan(changed, qualification=self.qualification)
        plan["expected_warmstart"]["folds"]["0"]["path"] = "/different/initialization.pt"
        with self.assertRaisesRegex(ValueError, "Warm-start selection"):
            worker.validate_training_plan(plan, qualification=self.qualification)

    def test_exp_name_cannot_escape_or_alias_declared_output_roots(self):
        plan = self.make_plan()
        for name in (str(self.root / "shared-output"), "../shared-output", ".", "..", "", "nested/experiment"):
            changed = copy.deepcopy(plan)
            changed["common_args"][changed["common_args"].index("--exp_name") + 1] = name
            changed["expected_settings"]["exp_name"] = name
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "exp_name"):
                worker.validate_training_plan(changed, qualification=self.qualification)

    def test_training_must_use_the_qualified_grid_and_radiomics_configuration(self):
        plan = self.make_plan()
        for field in ("policy_id", "radiomics_config_sha256"):
            changed = {**self.qualification, field: "different-qualification"}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "qualified"):
                worker.validate_training_plan(plan, qualification=changed)
        index = plan["common_args"].index("--img_size")
        plan["common_args"][index + 1:index + 4] = ["64", "64", "64"]
        plan["expected_settings"]["img_size"] = [64, 64, 64]
        with self.assertRaisesRegex(ValueError, "Bound image shape"):
            worker.validate_training_plan(plan, qualification=self.qualification)

    def test_warmstart_hash_binding_precedes_checkpoint_deserialization(self):
        from trifusesurv2.utils.warmstart import launcher_warmstart_settings, warmstart_fingerprints
        plan = self.make_plan()
        checkpoint = self.root / "unbound.pt"
        checkpoint.write_bytes(b"not a deserialized checkpoint")
        plan["common_args"] += ["--contour_warmstart_ckpt", str(checkpoint), "--align_swin_cfg_from_contour_warmstart"]
        settings = launcher_warmstart_settings({}, ["--align_swin_cfg_from_contour_warmstart", *plan["common_args"]])
        plan["expected_warmstart"] = warmstart_fingerprints(settings)
        with mock.patch("trifusesurv2.multimodal_survival.train.align_backbone_cfg_to_contour_warmstart") as deserialize:
            with self.assertRaisesRegex(ValueError, "before checkpoint deserialization"):
                worker.validate_training_plan(plan, qualification=self.qualification)
        deserialize.assert_not_called()

    def fake_train(self, plan, *, wrong_counts=False):
        def run(command, **kwargs):
            mode = command[command.index("--fusion_mode") + 1]
            output = Path(command[command.index("--out_dir") + 1]) / "paired/fold_00"
            output.mkdir(parents=True)
            settings = dict(plan["expected_settings"], fusion_mode=mode, out_dir=str(output.parents[1]))
            (output / "model_contract.json").write_text(json.dumps({
                "experiment_spec": {"settings": settings}, "parameters": 101 if wrong_counts and mode == "F3" else 100,
                "trainable_parameters": 80,
            }))
            return subprocess.CompletedProcess(command, 0)
        return run

    def test_training_is_sequential_and_count_mismatch_is_not_accepted(self):
        plan = self.make_plan()
        output = self.root / "receipts"
        output.mkdir()
        with mock.patch.object(worker.subprocess, "run", side_effect=self.fake_train(plan, wrong_counts=True)) as run:
            with self.assertRaisesRegex(RuntimeError, "parameter counts differ"):
                worker.execute_training_plan(plan, output, qualification=self.qualification)
        self.assertEqual(run.call_count, 2)
        self.assertEqual([call.args[0][call.args[0].index("--fusion_mode") + 1] for call in run.call_args_list], ["F2", "F3"])
        for call in run.call_args_list:
            self.assertTrue(call.kwargs["check"])
            self.assertEqual(call.args[0][3:-4], plan["common_args"])

    def test_failed_first_training_is_not_retried_or_followed_by_f3(self):
        plan = self.make_plan()
        output = self.root / "receipts"
        output.mkdir()
        with mock.patch.object(worker.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "train")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                worker.execute_training_plan(plan, output, qualification=self.qualification)
        self.assertEqual(run.call_count, 1)
        self.assertFalse((output / "training_F3.log").exists())

    def test_unchanged_paired_plan_can_finish_both_runs(self):
        plan = self.make_plan()
        output = self.root / "receipts"
        output.mkdir()
        with mock.patch.object(worker.subprocess, "run", side_effect=self.fake_train(plan)):
            result = worker.execute_training_plan(plan, output, qualification=self.qualification)
        self.assertEqual([run["fusion_mode"] for run in result], ["F2", "F3"])
        self.assertEqual(result[0]["fold_parameter_counts"], result[1]["fold_parameter_counts"])

    def test_qualification_only_writes_failure_receipt_without_starting_training(self):
        output = self.root / "qualification"
        grid, config = self.root / "grid.json", self.root / "radiomics.json"
        grid.write_text("{}")
        config.write_text("{}")
        with mock.patch.object(worker, "require_one_h100", return_value={"name": "fake gate"}), \
                mock.patch.object(worker.importlib.metadata, "version", return_value="test-only"), \
                mock.patch.object(worker, "run_sampling_tests", return_value={"passed": True}), \
                mock.patch.object(worker, "scientific_phantom_checks", side_effect=ValueError("qualification failed")), \
                mock.patch.object(worker, "gpu_model_check") as gpu, \
                mock.patch.object(worker, "execute_training_plan") as train:
            with self.assertRaisesRegex(ValueError, "qualification failed"):
                worker.main(["--grid-config-json", str(grid), "--radiomics-config-json", str(config), "--out-dir", str(output)])
        receipt = json.loads((output / "qualification.json").read_text())
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("qualification failed", receipt["error"])
        gpu.assert_not_called()
        train.assert_not_called()

    def test_fake_extractor_exercises_sampling_without_claiming_pyradiomics(self):
        import numpy as np
        import SimpleITK as sitk
        captured = []

        class FakeExtractor:
            def __init__(self, config):
                pass

            def execute(self, image, mask, label):
                values = sitk.GetArrayFromImage(image)[sitk.GetArrayFromImage(mask).astype(bool)]
                captured.append(values)
                return {"original_firstorder_Mean": np.mean(values), "original_firstorder_Variance": np.var(values)}

        fake = types.ModuleType("radiomics")
        fake.__version__ = "fake-test-only"
        feature = types.ModuleType("radiomics.featureextractor")
        feature.RadiomicsFeatureExtractor = FakeExtractor
        grid = json.loads((PACKAGE / "configs/common_grid_candidate.json").read_text())
        config = json.loads((PACKAGE / "configs/radiomics_normalized_candidate.json").read_text())
        with mock.patch.dict(sys.modules, {"radiomics": fake, "radiomics.featureextractor": feature}):
            result = worker.scientific_phantom_checks(grid, config, rtol=1e-5, atol=1e-7)
            self.assertEqual(result["pyradiomics_version"], "fake-test-only")
            self.assertEqual(len(captured), 8)
            self.assertTrue(all(np.all((values >= 0) & (values <= 1)) for values in captured))
            config["setting"]["normalize"] = True
            with self.assertRaisesRegex(ValueError, "normalization"):
                worker.scientific_phantom_checks(grid, config, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
