"""Synthetic topology/optimizer fixtures; no real patient forward pass or fit."""
from collections import OrderedDict
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn

from trifusesurv2.encoders.clinical import SemanticClinicalTokenEncoder
from trifusesurv2.multimodal_survival import train


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("optimizer_mapping_tool", ROOT / "scripts" / "checkpoint_optimizer_mapping.py")
mapping = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mapping)


def synthetic_checkpoint():
    with patch.object(sys, "argv", ["fixture", "--meta_csv", "unused.csv", "--no_radiomics",
                                    "--img_size", "32", "32", "32", "--feature_size", "12",
                                    "--depths", "1", "1", "1", "1", "--img_token_dim", "8",
                                    "--v2_model_dim", "16", "--v2_num_heads", "4",
                                    "--v2_transformer_layers", "1", "--no_ema", "--no_swa"]):
        args = train.parse_args()
    code, _ = mapping.construction_statements(train)
    namespace = dict(train.__dict__)
    namespace.update(args=args, fold=0, fold_num_time_bins=3, clin_dim=0, rad_dim=0,
                     mode=train._image_encoder_mode(args), device=torch.device("cpu"))
    exec(code, namespace)
    model = namespace["base_model"]
    # These deliberately fabricated feature widths test Lazy load hooks and
    # registration order; they do not qualify a clinical image forward path.
    for module in model.modules():
        if isinstance(module, nn.LazyLinear):
            module.initialize_parameters(torch.zeros(1, 5))
        elif isinstance(module, nn.LazyConv3d):
            module.initialize_parameters(torch.zeros(1, 3, 1, 1, 1))
    source = {"commit_sha": None, "source_tree_sha256": mapping.tree_sha256(ROOT),
              "source_dirty": True, "declared_commit_sha": None}
    spec = train.canonical_experiment_spec(args, runtime={"torch_version": str(torch.__version__)})
    controls = {f"{prefix}_{suffix}": getattr(args, f"{prefix}_{suffix}")
                for prefix in ("lr", "wd") for suffix in ("backbone", "lora", "head", "clin", "rad")}
    groups = train.make_param_groups(model, **controls)
    optimizer = torch.optim.AdamW(groups, betas=(.9, .999), weight_decay=0.)
    # Synthetic Adam state exercises map/moment checks without optimizer steps.
    for name, parameter in model.named_parameters():
        if name.startswith("habitat_model.survival_heads."):
            optimizer.state[parameter] = {"step": torch.tensor(2.),
                                          "exp_avg": torch.full_like(parameter, .01),
                                          "exp_avg_sq": torch.full_like(parameter, .001)}
    contract = {"source_identity": {"commit_sha": None, "source_tree_sha256": source["source_tree_sha256"]},
                "experiment_spec": spec, "experiment_spec_sha256": train.experiment_spec_sha256(spec),
                "fusion_mode": args.fusion_mode, "num_time_bins": 3,
                "time_bin_width_days": args.time_bin_width_days,
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    checkpoint = {"args": vars(args), "source_identity": source, "num_time_bins": 3,
                  "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                  "encoders": {"model_version": "v2", "fusion_mode": args.fusion_mode,
                               "clinical": SemanticClinicalTokenEncoder(OrderedDict()),
                               "radiomics": None, "clinical_token_dim": 0, "radiomics_token_dim": 0,
                               "model_contract": contract}}
    return checkpoint, model, groups


class CheckpointOptimizerMappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.checkpoint, cls.model, cls.groups = synthetic_checkpoint()

    def test_actual_parameter_groups_survive_strict_lazy_reconstruction(self):
        model, evidence = mapping.reconstruct_model(self.checkpoint, train)
        names, proof = mapping.parameter_mapping(model, self.checkpoint, train)
        identities = {id(parameter): name for name, parameter in self.model.named_parameters()}
        expected = [[identities[id(parameter)] for parameter in group["params"]] for group in self.groups]
        self.assertEqual(names, expected)
        self.assertEqual(evidence["forward_passes"], 0)
        self.assertTrue(proof["serialized_ids_match_reconstructed_optimizer"])

    def test_unsupported_policy_and_incomplete_state_fail(self):
        for key, value in (("use_lora", True), ("model_version", "v1"), ("allow_partial_resume", True)):
            bad = {**self.checkpoint, "args": {**self.checkpoint["args"], key: value}}
            with self.subTest(key=key), self.assertRaises(ValueError):
                mapping.reconstruct_model(bad, train)
        bad = {**self.checkpoint, "model_state": dict(self.checkpoint["model_state"])}
        bad["model_state"].pop(next(iter(bad["model_state"])))
        with self.assertRaises(RuntimeError):
            mapping.reconstruct_model(bad, train)

    def test_permuted_optimizer_ids_and_lightweight_checkpoint_fail(self):
        state = deepcopy(self.checkpoint["optimizer_state"])
        state["param_groups"][0]["params"][:2] = state["param_groups"][0]["params"][:2][::-1]
        with self.assertRaisesRegex(ValueError, "IDs/order"):
            mapping.parameter_mapping(self.model, {**self.checkpoint, "optimizer_state": state}, train)
        with self.assertRaisesRegex(ValueError, "Full optimizer"):
            mapping.parameter_mapping(self.model, {**self.checkpoint, "optimizer_state": None}, train)

    def test_clean_process_cli_writes_auditor_compatible_bound_evidence(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "fixture.pt"
            torch.save(self.checkpoint, path)
            command = [sys.executable, str(ROOT / "scripts" / "map_checkpoint_optimizer.py"),
                       "--checkpoint", str(path), "--checkpoint-sha256", mapping.file_sha256(path),
                       "--source-root", str(ROOT), "--expected-monai-version", "1.5.2",
                       "--out-dir", str(root / "map")]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads((root / "map" / "optimizer_parameter_map.json").read_text())
            self.assertEqual(record["mapping_method"], "actual_model_parameter_groups")
            proof_path = Path(record["reconstruction_evidence"]["path"])
            self.assertEqual(record["reconstruction_evidence"]["sha256"], mapping.file_sha256(proof_path))
            proof = json.loads(proof_path.read_text())
            self.assertEqual(proof["checkpoint_sha256"], mapping.file_sha256(path))
            self.assertEqual(proof["checkpoint_source_identity"], record["source_identity"])
            from trifusesurv2.validation.pilot import optimizer_evidence
            accepted = optimizer_evidence(self.checkpoint, record, mapping.file_sha256(path),
                                          opportunities=4, reconstruction=proof)
            self.assertEqual(accepted["successful_updates_min"], 2)
            self.assertEqual(accepted["successful_updates_max"], 2)
            # A wrong expected checkpoint identity fails before reconstruction.
            command[command.index("--checkpoint-sha256") + 1] = "0" * 64
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SHA-256", result.stderr)


if __name__ == "__main__":
    unittest.main()
