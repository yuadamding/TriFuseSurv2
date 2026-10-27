"""Continuations must not silently switch data or shape-compatible fusion modes."""
import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from trifusesurv2.multimodal_survival.train import _validate_resume_model_contract
from trifusesurv2.utils.provenance import source_identity


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


if __name__ == "__main__":
    unittest.main()
