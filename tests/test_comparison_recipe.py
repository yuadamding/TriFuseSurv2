"""Principal recipes preserve measured inputs across the fusion comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


PACKAGE = Path(__file__).resolve().parents[1]
LAUNCHER = PACKAGE / "scripts/survival/train_contour_aware_survival.sh"
SEARCH = PACKAGE / "scripts/survival/search_roi_constrained_h100.sh"


class ComparisonRecipeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="trifusesurv2-recipe-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.capture = self.root / "arguments.json"
        self.python_mock = self.root / "python-mock"
        self.python_mock.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "if sys.argv[1:2] == ['-']:\n"
            "    sys.stdin.read()\n"
            "else:\n"
            "    Path(os.environ['MOCK_ARGUMENTS']).write_text(json.dumps(sys.argv[1:]))\n"
        )
        self.python_mock.chmod(0o755)
        self.env = dict(os.environ)
        for name in ("COMPARISON_RECIPE", "TOKEN_DROPOUT", "V2_IMAGE_HABITAT_DROPOUT_P", "FUSION_MODE", "ID_COL", "SEED", "REGION_MODE", "V2_NUM_HEADS"):
            self.env.pop(name, None)
        self.env.update(
            PYTHON_BIN=str(self.python_mock), DEVICE="cpu", MOCK_ARGUMENTS=str(self.capture),
            OUT_DIR=str(self.root / "outputs"),
        )

    def launch(self, **overrides):
        result = subprocess.run(
            ["bash", str(LAUNCHER)], env={**self.env, **overrides},
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = json.loads(self.capture.read_text())
        self.assertEqual(arguments[:3], ["-u", "-m", "trifusesurv2.multimodal_survival.train"])
        return {name: arguments[index + 1] for index, name in enumerate(arguments[:-1]) if name.startswith("--")}

    def run_search(self, profile, recipe=None, out_root=None, **overrides):
        out_root = out_root or self.root / f"search-{profile}-{recipe or 'default'}"
        env = {
            **self.env, "OUT_ROOT": str(out_root), "DRY_RUN": "1",
            "SEARCH_PROFILE": profile, "TRIAL_LIMIT": "0", "GPU_IDS": "0",
            "PYTHON_BIN": sys.executable,
        }
        if recipe is not None:
            env["COMPARISON_RECIPE"] = recipe
        env.update(overrides)
        result = subprocess.run(["bash", str(SEARCH)], env=env, capture_output=True, text=True, timeout=15)
        return result, out_root

    def search_plan(self, profile, recipe=None):
        result, out_root = self.run_search(profile, recipe)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = dict(line.split("=", 1) for line in (out_root / "search_manifest.txt").read_text().splitlines() if "=" in line)
        trials = [dict(value.split("=", 1) for value in shlex.split(contents))
                  for key, contents in manifest.items() if re.fullmatch(r"trial_\d+_env", key)]
        self.assertTrue(trials)
        return manifest, trials

    def test_default_principal_arguments_disable_both_paths_for_all_fusion_modes(self):
        for mode in ("F0", "F1", "F2", "F3", "F4"):
            with self.subTest(mode=mode):
                arguments = self.launch(FUSION_MODE=mode)
                self.assertEqual(arguments["--comparison_recipe"], "principal")
                self.assertEqual(arguments["--fusion_mode"], mode)
                self.assertEqual(arguments["--id_col"], "Patient_ID")
                self.assertEqual(float(arguments["--token_dropout"]), 0.0)
                self.assertEqual(float(arguments["--v2_image_habitat_dropout_p"]), 0.0)

    def test_patient_identifier_column_is_forwarded_explicitly(self):
        arguments = self.launch(ID_COL="patient_key")
        self.assertEqual(arguments["--id_col"], "patient_key")

    def test_robustness_explicitly_preserves_optional_dropout(self):
        arguments = self.launch(COMPARISON_RECIPE="robustness")
        self.assertEqual(arguments["--comparison_recipe"], "robustness")
        self.assertEqual(float(arguments["--token_dropout"]), 0.05)
        self.assertEqual(float(arguments["--v2_image_habitat_dropout_p"]), 0.05)
        arguments = self.launch(COMPARISON_RECIPE="robustness", TOKEN_DROPOUT="0.2", V2_IMAGE_HABITAT_DROPOUT_P="0.3")
        self.assertEqual(float(arguments["--token_dropout"]), 0.2)
        self.assertEqual(float(arguments["--v2_image_habitat_dropout_p"]), 0.3)

    def test_all_search_profiles_preserve_principal_zeros_in_forwarded_arguments(self):
        for profile in ("full", "roi_crop"):
            with self.subTest(profile=profile):
                manifest, trials = self.search_plan(profile)
                self.assertEqual(manifest["comparison_recipe"], "principal")
                self.assertEqual(float(manifest["token_dropout"]), 0.0)
                for trial in trials:
                    self.assertEqual(float(trial["V2_IMAGE_HABITAT_DROPOUT_P"]), 0.0)
                # Replay planned trial environments through the actual launcher,
                # substituting only the interpreter so no training can begin.
                for trial in (trials[0], trials[-1]):
                    arguments = self.launch(COMPARISON_RECIPE=manifest["comparison_recipe"],
                                            TOKEN_DROPOUT=manifest["token_dropout"], **trial)
                    self.assertEqual(float(arguments["--token_dropout"]), 0.0)
                    self.assertEqual(float(arguments["--v2_image_habitat_dropout_p"]), 0.0)

    def test_search_dropout_profiles_require_explicit_robustness_recipe(self):
        manifest, trials = self.search_plan("full", "robustness")
        self.assertEqual(manifest["comparison_recipe"], "robustness")
        self.assertEqual(float(manifest["token_dropout"]), 0.05)
        self.assertEqual({float(trial["V2_IMAGE_HABITAT_DROPOUT_P"]) for trial in trials}, {0.025, 0.04})

    def test_search_rejects_changed_recipe_before_overwriting_prior_receipts(self):
        result, out_root = self.run_search("full")
        self.assertEqual(result.returncode, 0, result.stderr)
        prior = {path.name: path.read_bytes() for path in out_root.iterdir() if path.is_file()}
        trial = out_root / "00_previous_trial"
        trial.mkdir()
        (trial / "status.txt").write_text("done\n")
        (trial / "status.rc").write_text("0\n")
        for changes in ({"COMPARISON_RECIPE": "robustness"}, {"FUSION_MODE": "F2"},
                        {"REGION_MODE": "legacy"}, {"SEED": "123"}, {"ID_COL": "patient_key"},
                        {"TOKEN_DROPOUT": "0.1"}, {"V2_IMAGE_HABITAT_DROPOUT_P": "0.1"},
                        {"V2_NUM_HEADS": "4"}):
            with self.subTest(changes=changes):
                result, _ = self.run_search("full", out_root=out_root, **changes)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("Search recipe contract changed", result.stderr)
                self.assertIn("new OUT_ROOT", result.stderr)
                for name, content in prior.items():
                    self.assertEqual((out_root / name).read_bytes(), content)
                self.assertEqual((trial / "status.txt").read_text(), "done\n")
                self.assertEqual((trial / "status.rc").read_text(), "0\n")

    def test_search_binds_input_bytes_and_excludes_unrelated_environment(self):
        metadata = self.root / "metadata.csv"
        manifest = self.root / "region_manifest.json"
        radiomics = self.root / "radiomics.csv"
        split = self.root / "splits/fold_00/train_ids.txt"
        split.parent.mkdir(parents=True)
        metadata.write_text("Patient_ID,region_manifest_path\n001,region_manifest.json\n")
        manifest.write_text('{"patient_id": "001"}\n')
        radiomics.write_text("Patient_ID,feature\n001,1\n")
        split.write_text("001\n")
        overrides = {
            "META_CSV": str(metadata), "RADIOMICS_SOURCE": str(radiomics),
            "SPLITS_DIR": str(split.parent.parent), "UNRELATED_SECRET_TOKEN": "do-not-record-this",
        }
        result, out_root = self.run_search("full", **overrides)
        self.assertEqual(result.returncode, 0, result.stderr)
        contract = (out_root / "search_recipe_contract.json").read_text()
        self.assertNotIn("UNRELATED_SECRET_TOKEN", contract)
        self.assertNotIn("do-not-record-this", contract)
        self.assertEqual(len(json.loads(contract)["source_identity"]["source_tree_sha256"]), 64)
        receipts = {path.name: path.read_bytes() for path in out_root.iterdir() if path.is_file()}
        for path in (metadata, radiomics, split, manifest):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                try:
                    result, _ = self.run_search("full", out_root=out_root, **overrides)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn("Search recipe contract changed", result.stderr)
                    for name, content in receipts.items():
                        self.assertEqual((out_root / name).read_bytes(), content)
                finally:
                    path.write_bytes(original)

    def test_search_source_digest_rejects_changed_program_defaults(self):
        copied = self.root / "copied-package"
        shutil.copytree(PACKAGE / "src", copied / "src", ignore=shutil.ignore_patterns("__pycache__"))
        scripts = copied / "scripts/survival"
        scripts.mkdir(parents=True)
        for name in (SEARCH.name, LAUNCHER.name, "train_with_roi_focus_watch.sh"):
            shutil.copy2(PACKAGE / "scripts/survival" / name, scripts / name)
        with mock.patch(f"{__name__}.SEARCH", scripts / SEARCH.name):
            result, out_root = self.run_search("full")
            self.assertEqual(result.returncode, 0, result.stderr)
            receipts = {path.name: path.read_bytes() for path in out_root.iterdir() if path.is_file()}
            launcher = scripts / LAUNCHER.name
            launcher.write_text(launcher.read_text().replace('V2_NUM_HEADS:-8', 'V2_NUM_HEADS:-4'))
            result, _ = self.run_search("full", out_root=out_root)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("Search recipe contract changed", result.stderr)
            for name, content in receipts.items():
                self.assertEqual((out_root / name).read_bytes(), content)

    def test_search_refuses_unbound_historical_outputs_without_relabeling_them(self):
        out_root = self.root / "old-search"
        out_root.mkdir()
        previous = out_root / "search_manifest.txt"
        previous.write_text("historical receipt\n")
        result, _ = self.run_search("full", out_root=out_root)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("outputs without a search recipe contract", result.stderr)
        self.assertEqual(previous.read_text(), "historical receipt\n")
        self.assertFalse((out_root / "search_recipe_contract.json").exists())

    def test_search_accepts_operational_changes_without_changing_contract(self):
        result, out_root = self.run_search("full")
        self.assertEqual(result.returncode, 0, result.stderr)
        contract = (out_root / "search_recipe_contract.json").read_bytes()
        result, _ = self.run_search("full", out_root=out_root, WORKERS="3", LOG_EVERY_BATCHES="1", TRIAL_LIMIT="2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((out_root / "search_recipe_contract.json").read_bytes(), contract)

    def test_cli_principal_defaults_and_robustness_validation(self):
        from trifusesurv2.multimodal_survival.train import parse_args

        def parse(*extra):
            with mock.patch.object(sys, "argv", ["train", "--meta_csv", "synthetic.csv", *extra]):
                return parse_args()

        principal = parse()
        self.assertEqual(principal.comparison_recipe, "principal")
        self.assertEqual(principal.token_dropout, 0.0)
        self.assertEqual(principal.v2_image_habitat_dropout_p, 0.0)
        for option in ("--token_dropout", "--v2_image_habitat_dropout_p"):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "principal F0.*F4 comparison requires"):
                parse(option, "0.1")
        robustness = parse("--comparison_recipe", "robustness", "--token_dropout", "0.2", "--v2_image_habitat_dropout_p", "0.3")
        self.assertEqual(robustness.token_dropout, 0.2)
        self.assertEqual(robustness.v2_image_habitat_dropout_p, 0.3)

    def test_dropped_image_changes_radiomics_availability_differently_by_mode(self):
        import torch
        from trifusesurv2.models.habitat_survival import HabitatAlignedSurvivalModel

        for mode in ("F0", "F1", "F2", "F3", "F4"):
            with self.subTest(mode=mode):
                torch.manual_seed(143)
                model = HabitatAlignedSurvivalModel(
                    image_token_dim=8, radiomics_token_dim=5, clinical_token_dim=4,
                    num_time_bins=3, model_dim=16, num_heads=4, transformer_layers=1,
                    dropout=0.0, fusion_mode=mode,
                ).eval()
                image_presence = torch.ones(2, 6, dtype=torch.bool)
                radiomics = torch.randn(2, 4, 5, requires_grad=True)
                inputs = dict(
                    image_tokens=torch.randn(2, 6, 8), image_presence=image_presence,
                    radiomics_tokens=radiomics, radiomics_presence=torch.ones(2, 4, dtype=torch.bool),
                    clinical_tokens=torch.randn(2, 3, 4), clinical_presence=torch.ones(2, 3, dtype=torch.bool),
                    region_states=torch.zeros(2, 4, dtype=torch.long),
                )
                measured = model(**inputs)["OS"].square().sum()
                gradient = torch.autograd.grad(measured, radiomics)[0]
                self.assertGreater(float(gradient[:, 0].abs().sum()), 0.0)
                image_presence[:, 1] = False
                dropped = model(**inputs)["OS"].square().sum()
                gradient = torch.autograd.grad(dropped, radiomics)[0]
                if mode in {"F3", "F4"}:
                    torch.testing.assert_close(gradient[:, 0], torch.zeros_like(gradient[:, 0]), rtol=0, atol=0)
                else:
                    self.assertGreater(float(gradient[:, 0].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
