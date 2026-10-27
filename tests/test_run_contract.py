"""Continuations must not silently switch data or shape-compatible fusion modes."""
import copy
import argparse
import unittest
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch
import torch

from trifusesurv2.multimodal_survival.train import _validate_resume_model_contract
from trifusesurv2.utils.provenance import source_identity, source_contract_identity
from trifusesurv2.utils.experiment import canonical_experiment_spec, experiment_spec_sha256, execution_topology


class RunContractTests(unittest.TestCase):
    def test_unchanged_contract_resumes(self):
        contract = {"fusion_mode": "F3", "train_ids": ["001"],
                    "region_manifest_sha256": {"001": "source"}}
        _validate_resume_model_contract(copy.deepcopy(contract), contract)

    def test_same_shape_fusion_or_input_changes_reject_resume(self):
        contract = {"fusion_mode": "F3", "train_ids": ["001"],
                    "region_manifest_sha256": {"001": "source"}, "metadata_sha256": "labels"}
        for key, value in (("fusion_mode", "F2"), ("train_ids", ["002"]),
                           ("region_manifest_sha256", {"001": "changed"}),
                           ("metadata_sha256", "changed")):
            with self.subTest(key=key):
                changed = copy.deepcopy(contract)
                changed[key] = value
                with self.assertRaisesRegex(RuntimeError, key):
                    _validate_resume_model_contract(contract, changed)

    def test_verified_contract_cannot_be_inferred_from_older_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError, "start a new experiment"):
            _validate_resume_model_contract({}, {"region_mode": "verified"})

    def test_semantic_architecture_objective_and_selection_settings_are_locked(self):
        args = argparse.Namespace(v2_num_heads=4, num_heads=[3, 6, 12, 24], report_metric="c_index",
                                  aux_surv_loss_weight=0.35, lr_head=0.001, epochs=20,
                                  auc_times_days=[365.0], v2_dropout=0.1,
                                  future_semantic_option="original", workers=0, resume=False)
        original = {"experiment_spec": canonical_experiment_spec(args)}
        for key, value in (("v2_num_heads", 8), ("num_heads", [6, 6, 12, 24]),
                           ("report_metric", "ibs"), ("aux_surv_loss_weight", 0),
                           ("lr_head", 0.002), ("epochs", 21), ("auc_times_days", [730.0]),
                           ("v2_dropout", 0.2), ("future_semantic_option", "changed")):
            with self.subTest(key=key):
                changed = copy.deepcopy(args)
                setattr(changed, key, value)
                with self.assertRaisesRegex(RuntimeError, key):
                    _validate_resume_model_contract(original, {"experiment_spec": canonical_experiment_spec(changed)})

    def test_only_named_operational_changes_are_allowed_on_resume(self):
        args = argparse.Namespace(v2_num_heads=4, workers=0, eval_workers=0, resume=False,
                                  log_every_batches=50, cache_volumes=False, volume_cache_size=0)
        original = canonical_experiment_spec(args)
        args.workers, args.eval_workers, args.resume = 2, 1, True
        args.log_every_batches, args.cache_volumes, args.volume_cache_size = 1, True, 8
        changed = canonical_experiment_spec(args)
        self.assertEqual(original, changed)
        self.assertEqual(experiment_spec_sha256(original), experiment_spec_sha256(changed))

    def test_experiment_spec_is_stable_json_and_rejects_nonfinite_settings(self):
        args = argparse.Namespace(num_heads=(3, 6), loss_weight=1.0)
        original = canonical_experiment_spec(args)
        args.num_heads = [3, 6]
        self.assertEqual(original, canonical_experiment_spec(args))
        args.loss_weight = float("nan")
        with self.assertRaises(ValueError):
            canonical_experiment_spec(args)

    def test_ddp_ranks_share_the_contract_but_changed_world_size_is_rejected(self):
        args = argparse.Namespace(ddp=True, v2_num_heads=4)
        rank0 = canonical_experiment_spec(args, runtime=execution_topology(torch.device("cuda:0"), ddp_world_size=2))
        rank1 = canonical_experiment_spec(args, runtime=execution_topology(torch.device("cuda:1"), ddp_world_size=2))
        self.assertEqual(rank0, rank1)
        different_world = canonical_experiment_spec(args, runtime=execution_topology(torch.device("cuda:0"), ddp_world_size=4))
        with self.assertRaisesRegex(RuntimeError, "effective_world_size"):
            _validate_resume_model_contract(rank0, different_world)

    def test_source_identity_never_promotes_an_environment_declaration_to_observed_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "src" / "model.py").write_text("VALUE = 1\n")
            with patch.dict("os.environ", {"TRIFUSESURV2_COMMIT_SHA": "historical-declaration"}):
                identity = source_identity(root)
            self.assertIsNone(identity["commit_sha"])
            self.assertEqual(identity["declared_commit_sha"], "historical-declaration")
            self.assertEqual(len(identity["source_tree_sha256"]), 64)
            (root / "src" / "model.py").write_text("VALUE = 2\n")
            source_identity.cache_clear()
            self.assertNotEqual(identity["source_tree_sha256"], source_identity(root)["source_tree_sha256"])

    def test_run_outputs_change_dirty_diagnostic_without_changing_source_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            code = root / "src" / "model.py"
            code.write_text("VALUE = 1\n")
            def git(*arguments):
                subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True)
            git("init", "-q")
            git("add", "src/model.py")
            git("-c", "user.name=Regression", "-c", "user.email=regression@example.invalid", "commit", "-qm", "fixture")
            original = source_contract_identity(root)
            self.assertFalse(source_identity(root)["source_dirty"])
            (root / "runs").mkdir()
            (root / "runs" / "model_contract.json").write_text("{}\n")
            source_identity.cache_clear()
            self.assertTrue(source_identity(root)["source_dirty"])
            self.assertEqual(original, source_contract_identity(root))
            code.write_text("VALUE = 2\n")
            source_identity.cache_clear()
            self.assertNotEqual(original, source_contract_identity(root))


if __name__ == "__main__":
    unittest.main()
