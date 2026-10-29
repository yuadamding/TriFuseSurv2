"""Synthetic topology/optimizer fixtures; no real patient forward pass or fit."""
from collections import OrderedDict
from copy import deepcopy
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import numpy as np
from torch import nn

from trifusesurv2.encoders.clinical import SemanticClinicalTokenEncoder
from trifusesurv2.multimodal_survival import train


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("optimizer_mapping_tool", ROOT / "scripts" / "checkpoint_optimizer_mapping.py")
mapping = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mapping)


def synthetic_checkpoint(*, routing_intervention=False):
    arguments = ["fixture", "--meta_csv", "unused.csv", "--no_radiomics",
                                    "--img_size", "32", "32", "32", "--feature_size", "12",
                                    "--depths", "1", "1", "1", "1", "--img_token_dim", "8",
                                    "--v2_model_dim", "16", "--v2_num_heads", "4",
                                    "--v2_transformer_layers", "1", "--no_ema", "--no_swa"]
    if routing_intervention:
        from trifusesurv2.models.routing_intervention import ROUTING_INTERVENTION_ID
        arguments.remove('--no_radiomics')
        arguments += ["--routing_intervention", ROUTING_INTERVENTION_ID]
    with patch.object(sys, "argv", arguments):
        args = train.parse_args()
    code, _ = mapping.construction_statements(train)
    radiomics, rad_dim = None, 0
    if routing_intervention:
        from trifusesurv2.encoders.radiomics import HabitatRadiomicsTokenEncoder, GroupPCASpec
        from trifusesurv2.models.routing_intervention import REGIONAL_ORDER
        rad_dim = 4
        radiomics = HabitatRadiomicsTokenEncoder(group_names=REGIONAL_ORDER,
            pca_specs={name: GroupPCASpec(name=name, input_dim=4, output_dim=4,
                lower=np.full(4, -1.), upper=np.ones(4), mean=np.zeros(4), scale=np.ones(4),
                pca_mean=np.zeros(4), components=np.eye(4), explained_variance_ratio=np.full(4, .25))
                for name in REGIONAL_ORDER}, patient_vectors={}, presence_map={})
    namespace = dict(train.__dict__)
    namespace.update(args=args, fold=0, fold_num_time_bins=3, clin_dim=0, rad_dim=rad_dim,
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
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                **train._routing_intervention_contract(args)}
    saved_arguments = dict(vars(args))
    if saved_arguments.get('routing_intervention') is None:
        saved_arguments.pop('routing_intervention', None)
    checkpoint = {"args": saved_arguments, "source_identity": source, "num_time_bins": 3,
                  "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                  "encoders": {"model_version": "v2", "fusion_mode": args.fusion_mode,
                               "clinical": SemanticClinicalTokenEncoder(OrderedDict()),
                               "radiomics": radiomics, "clinical_token_dim": 0, "radiomics_token_dim": rad_dim,
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

    def test_new_constructor_and_exact_helper_bindings_are_recorded(self):
        _, proof = mapping.construction_statements(train)
        self.assertEqual(proof['construction_ast_sha256'], mapping.ROUTING_CONSTRUCTION_AST_SHA256)
        dependencies = proof['routing_construction_dependencies']
        self.assertEqual(dependencies['trainer_helper_ast_sha256'], mapping.ROUTING_TRAINER_HELPER_AST_SHA256)
        self.assertEqual(dependencies['metadata_helper_ast_sha256'], mapping.ROUTING_METADATA_HELPER_AST_SHA256)
        self.assertEqual(dependencies['metadata_source_sha256'], mapping.file_sha256(dependencies['metadata_source_path']))

    def test_helper_digest_is_stable_across_only_empty_type_parameter_fields(self):
        function = ast.parse('def reviewed(value):\n    return value\n').body[0]
        digest = mapping.function_ast_digest(function)
        function._fields = tuple(key for key in function._fields if key != 'type_params') + ('type_params',)
        function.type_params = []
        self.assertEqual(mapping.function_ast_digest(function), digest)
        function.type_params = [ast.Name(id='DifferentGenericType', ctx=ast.Load())]
        self.assertNotEqual(mapping.function_ast_digest(function), digest)

    def test_legacy_constructor_remains_supported_without_routing_helpers(self):
        source = Path(train.__file__).read_text()
        self.assertEqual(source.count('            **_routing_model_kwargs(args),'), 1)
        legacy = source.replace('            **_routing_model_kwargs(args),\n', '')
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'legacy_train.py'; path.write_text(legacy)
            _, proof = mapping.construction_statements(SimpleNamespace(__file__=str(path)))
        self.assertEqual(proof['construction_ast_sha256'], mapping.SUPPORTED_CONSTRUCTION_AST_SHA256)
        self.assertNotIn('routing_construction_dependencies', proof)

    def test_changed_constructor_or_dependency_does_not_bypass_the_reviewed_adapter(self):
        source = Path(train.__file__).read_text()
        for change in ('constructor', 'helper'):
            if change == 'constructor':
                changed = source.replace('            **_routing_model_kwargs(args),',
                                         '            **_routing_model_kwargs(args), arbitrary_flag=True,')
                expected = 'Construction AST differs'
            else:
                changed = source.replace('return {"routing_intervention": metadata["routing_intervention"]} if metadata else {}',
                                         'return {"routing_intervention": None}')
                expected = 'helper AST differs'
            self.assertNotEqual(source, changed)
            with self.subTest(change=change), TemporaryDirectory() as temporary:
                path = Path(temporary) / 'changed_train.py'; path.write_text(changed)
                with self.assertRaisesRegex(ValueError, expected):
                    mapping.construction_statements(SimpleNamespace(__file__=str(path)))

    def test_intervention_checkpoint_reconstruction_preserves_and_checks_its_identity(self):
        from trifusesurv2.models.routing_intervention import ROUTING_INTERVENTION_ID
        checkpoint, _, _ = synthetic_checkpoint(routing_intervention=True)
        model, proof = mapping.reconstruct_model(checkpoint, train)
        self.assertEqual(model.routing_intervention, ROUTING_INTERVENTION_ID)
        self.assertEqual(proof['checkpoint_routing_metadata']['routing_intervention'], ROUTING_INTERVENTION_ID)
        for location in ('args', 'contract'):
            bad = deepcopy(checkpoint)
            fields = bad['args'] if location == 'args' else bad['encoders']['model_contract']
            fields['routing_intervention_design_sha256'] = 'wrong'
            with self.subTest(location=location), self.assertRaises(ValueError):
                mapping.reconstruct_model(bad, train)

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
