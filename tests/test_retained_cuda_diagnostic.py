"""CPU-only identity/report/flow fixtures; no numerical matrix or neural fits."""
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("retained_cuda_diagnostic", ROOT / "scripts/retained_cuda_diagnostic.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)
ORIGINAL = Path(__file__).parent / "fixtures/retained_budget_a_qualify_worker.py"


def tiny_metadata_model():
    model = torch.nn.Module()
    model.survival_heads = torch.nn.Module()
    model.survival_heads.shared = torch.nn.Linear(1, 1)
    model.survival_heads.shared_logits = torch.nn.Linear(1, 1)
    model.survival_heads.endpoint_residuals = torch.nn.ModuleDict({"OS": torch.nn.Linear(1, 1)})
    model.unused = torch.nn.Parameter(torch.ones(1))
    return model


class IdentityTests(unittest.TestCase):
    def test_exact_original_fixture_extraction_and_tamper_rejection(self):
        self.assertEqual(diagnostic.sha(ORIGINAL), diagnostic.ORIGINAL_FIXTURE_SHA256)
        program = diagnostic.compile_original_program(ORIGINAL)
        self.assertEqual(program["extracted"]["optimizer"]["source"],
                         "optimizer = torch.optim.AdamW(model.parameters(), lr=.001)")
        self.assertEqual(program["extracted"]["scaler"]["source"], "scaler = torch.amp.GradScaler('cuda')")
        self.assertIn("sum(value.float().square().mean() for value in outputs.values())",
                      program["extracted"]["forward_loss"]["source"])
        self.assertIn(str(ORIGINAL.resolve()), program["prepare"].__code__.co_filename)
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "altered.py"
            altered.write_bytes(ORIGINAL.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                diagnostic.compile_original_program(altered)

    def test_tensor_hash_binds_dtype_shape_and_values(self):
        tensor = torch.tensor([[1., 2.], [3., 4.]])
        expected = diagnostic.identity_digest({"x": tensor})
        self.assertEqual(expected, diagnostic.identity_digest({"x": tensor.clone()}))
        for modified in (tensor.reshape(-1), tensor.double(), tensor + .25):
            with self.subTest(modified=modified):
                self.assertNotEqual(expected, diagnostic.identity_digest({"x": modified}))
        self.assertEqual(diagnostic.tensor_identity(tensor.t()), diagnostic.tensor_identity(tensor.t().contiguous()))
        self.assertEqual(diagnostic.tensor_identity(torch.tensor(1.))["bytes"], 4)

    def test_rng_restore_covers_python_numpy_and_torch(self):
        state = diagnostic.capture_rng(cuda=False)
        expected = (random.random(), np.random.random(), torch.rand(3))
        diagnostic.restore_rng(state)
        actual = (random.random(), np.random.random(), torch.rand(3))
        self.assertEqual(expected[:2], actual[:2])
        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)

    def test_cell_match_rejects_independently_changed_state_input_or_rng(self):
        model, inputs = tiny_metadata_model(), {"mask": torch.tensor([True, False])}
        rng = diagnostic.capture_rng(cuda=False)
        expected = diagnostic.initial_identity(model, inputs, rng)
        self.assertEqual(diagnostic.verify_initial_match(model, inputs, expected, cuda=False)["status"], "matched")
        inputs["mask"][0] = False
        with self.assertRaisesRegex(ValueError, "differs"):
            diagnostic.verify_initial_match(model, inputs, expected, cuda=False)
        inputs["mask"][0] = True
        with torch.no_grad():
            model.unused.add_(1)
        with self.assertRaisesRegex(ValueError, "differs"):
            diagnostic.verify_initial_match(model, inputs, expected, cuda=False)
        with torch.no_grad():
            model.unused.sub_(1)
        torch.rand(1)
        with self.assertRaisesRegex(ValueError, "differs"):
            diagnostic.verify_initial_match(model, inputs, expected, cuda=False)

    def test_lazy_materialization_is_once_before_snapshot(self):
        model = torch.nn.LazyLinear(2)
        inputs = {"x": torch.ones(1, 3)}
        # Only tensor allocation/initialization is exercised; no backward/fit.
        calls = []
        def materialize(m, x):
            calls.append(True)
            return m(x["x"])
        result = diagnostic.materialize_fixture(model, inputs, materialize, cuda=False)
        self.assertTrue(result["lazy_names_before"])
        self.assertEqual(result["lazy_names_after"], [])
        self.assertEqual(result["materialization_forward_count"], 1)
        diagnostic.materialize_fixture(model, inputs, materialize, cuda=False)
        self.assertEqual(len(calls), 1)

    def test_source_digest_binds_exact_python_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src/a.py").write_text("x = 1\n")
            first = diagnostic.source_tree_sha256(root)
            (root / "src/a.py").write_text("x = 2\n")
            self.assertNotEqual(first, diagnostic.source_tree_sha256(root))


class NumericalMetadataTests(unittest.TestCase):
    def test_scalar_value_preserves_signed_finite_values_and_rejects_nonfinite(self):
        for value in (-2.5, -0.0, 0.0, 1.0):
            with self.subTest(value=value):
                observed = diagnostic.tensor_summary(torch.tensor(value))
                self.assertEqual(observed["scalar_value"], value)
                self.assertEqual(observed["shape"], [])
                json.dumps(observed, allow_nan=False)
        singleton = diagnostic.tensor_summary(torch.tensor([-3.]))
        self.assertEqual(singleton["scalar_value"], -3.)
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                observed = diagnostic.tensor_summary(torch.tensor(value))
                self.assertIsNone(observed["scalar_value"])
                self.assertFalse(observed["all_finite"])
                json.dumps(observed, allow_nan=False)
        self.assertIsNone(diagnostic.tensor_summary(torch.tensor([1., 2.]))["scalar_value"])

    def test_real_cpu_adamw_scalar_step_corresponds_to_named_counter(self):
        # One actual optimizer operation with prescribed fabricated gradients.
        # No neural forward/backward, clinical data or model fit is performed.
        model = tiny_metadata_model()
        _, groups = diagnostic.required_parameter_groups(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        successful = []
        handle = optimizer.register_step_post_hook(lambda *_: successful.append(True))
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        handle.remove()
        observed = diagnostic.parameter_and_optimizer_observations(model, optimizer, groups)
        self.assertEqual(len(successful), 1)
        self.assertEqual(diagnostic.update_failures(observed, len(successful)), [])
        for rows in observed["required_paths"].values():
            for row in rows:
                raw = observed["optimizer_states"][row["parameter_name"]]["step"]
                self.assertEqual(raw["shape"], [])
                self.assertEqual(raw["elements"], 1)
                self.assertEqual(raw["scalar_value"], 1.0)
                self.assertEqual(raw["scalar_value"], row["step"])
                self.assertTrue(raw["all_finite"])
        json.dumps(observed, allow_nan=False)

    def test_nonfinite_counts_norm_and_json_are_safe(self):
        result = diagnostic.tensor_summary(torch.tensor([float("nan"), float("inf"), -float("inf"), 3., 4.]))
        self.assertEqual((result["nan_elements"], result["positive_inf_elements"], result["negative_inf_elements"]), (1, 1, 1))
        self.assertEqual(result["finite_l2_norm"], 5.)
        self.assertEqual(result["finite_max_abs"], 4.)
        self.assertEqual(result["nonzero_finite_elements"], 2)
        json.dumps(result, allow_nan=False)
        huge = diagnostic.tensor_summary(torch.tensor([3e38, 3e38]))
        self.assertGreater(huge["finite_l2_norm"], 3e38)
        self.assertTrue(huge["all_finite"])
        all_bad = diagnostic.tensor_summary(torch.tensor([float("nan"), float("inf")]))
        self.assertIsNone(all_bad["finite_l2_norm"])
        json.dumps(all_bad, allow_nan=False)

    def test_missing_empty_and_nonfinite_gradients_are_distinct(self):
        model = tiny_metadata_model()
        _, groups = diagnostic.required_parameter_groups(model)
        empty = diagnostic.gradient_observations(model, groups)
        self.assertIn("empty_gradient_collection", diagnostic.gradient_failures(empty))
        self.assertEqual(empty["nonfinite_names"], [])
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        model.unused.grad = None
        finite = diagnostic.gradient_observations(model, groups)
        self.assertEqual(diagnostic.gradient_failures(finite), [])
        self.assertEqual(finite["missing_names"], ["unused"])
        model.survival_heads.shared.weight.grad.fill_(float("nan"))
        nonfinite = diagnostic.gradient_observations(model, groups)
        self.assertIn("nonfinite_gradients", diagnostic.gradient_failures(nonfinite))
        self.assertEqual(nonfinite["nonfinite_names"], ["survival_heads.shared.weight"])
        model.survival_heads.shared.weight.grad = None
        missing = diagnostic.gradient_observations(model, groups)
        self.assertIn("missing_required_gradients:shared_survival_head", diagnostic.gradient_failures(missing))

    def test_required_optimizer_states_are_named_and_steps_exact(self):
        model = tiny_metadata_model()
        _, groups = diagnostic.required_parameter_groups(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        # Fabricated states; no optimizer operation is executed in this test.
        for p in model.parameters():
            optimizer.state[p] = {"step": torch.tensor(1.), "exp_avg": torch.ones_like(p), "exp_avg_sq": torch.ones_like(p)}
        observed = diagnostic.parameter_and_optimizer_observations(model, optimizer, groups)
        self.assertEqual(diagnostic.update_failures(observed, 1), [])
        self.assertEqual(observed["required_paths"]["os_survival_head"][0]["parameter_name"],
                         "survival_heads.endpoint_residuals.OS.weight")
        optimizer.state[model.survival_heads.shared.weight]["step"] = torch.tensor(2.)
        observed = diagnostic.parameter_and_optimizer_observations(model, optimizer, groups)
        self.assertIn("required_adam_step_not_one:shared_survival_head", diagnostic.update_failures(observed, 1))
        self.assertIn("actual_optimizer_step_count_not_one", diagnostic.update_failures(observed, 0))

    def test_no_qualification_or_causal_claim_from_pattern(self):
        cells = [{"mode": mode, "cell": cell, "status": "numerically_passed", "numerical_passed": True}
                 for mode in ("F2", "F3") for cell in diagnostic.CELLS]
        self.assertEqual(diagnostic.interpret(cells)["F2"], "original_failure_not_reproduced")
        cells[0].update(status="numerical_failure", numerical_passed=False)
        self.assertIn("not_root_cause_proof", diagnostic.interpret(cells)["F2"])
        cells[1].update(status="unexpected_error", numerical_passed=False)
        self.assertEqual(diagnostic.interpret(cells)["F2"], "incomplete_or_unexpected_error")

    def test_writes_are_exclusive_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            ref = diagnostic.write_json(path, {"value": 2})
            self.assertEqual(ref["sha256"], diagnostic.sha(path))
            self.assertEqual(ref["bytes"], path.stat().st_size)
            with self.assertRaises(FileExistsError):
                diagnostic.write_json(path, {"value": 3})


class CapturedFailureFlowTests(unittest.TestCase):
    def test_six_cell_mocked_producer_uses_actual_writers_and_is_not_qualification(self):
        from retained_cuda_fixture_producer import produce
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "producer"
            report = produce(root)
            self.assertEqual(report["status"], "completed_diagnostic")
            self.assertFalse(report["qualification_passed"])
            self.assertFalse(report["all_cells_numerically_passed"])
            self.assertEqual(len(report["cells"]), 6)
            for mode in ("F2", "F3"):
                expected = report["mode_fixtures"][mode]["identity"]
                for row in [row for row in report["cells"] if row["mode"] == mode]:
                    cell = json.loads(Path(row["report"]["path"]).read_text())
                    self.assertEqual(cell["identity"], expected)
                    self.assertEqual({k: cell["initial_match"][k] for k in expected}, expected)
                    self.assertEqual(cell["exception"], None)
            self.assertEqual(json.loads((root / "FIXTURE.json").read_text())["CUDA_calls"], 0)

    def test_missing_or_duplicate_cell_inventory_cannot_complete(self):
        for cells in ([], [{"mode": "F2", "cell": "fp32", "status": "numerically_passed", "numerical_passed": True}] * 6):
            with self.subTest(cells=len(cells)), tempfile.TemporaryDirectory() as directory:
                report = diagnostic.make_summary({}, {}, {})
                report["cells"] = cells
                self.assertEqual(diagnostic.finalize_summary(report, directory, True), 2)
                self.assertEqual(report["status"], "failed_diagnostic")

    def test_failed_gradient_evidence_precedes_guard_and_update_is_not_called(self):
        model = tiny_metadata_model()
        inputs = {"fabricated": torch.ones(1)}
        rng = diagnostic.capture_rng(cuda=False)
        expected = diagnostic.initial_identity(model, inputs, rng)
        created = []
        def forward(current, unused):
            created.append(current)
            loss = torch.tensor(1., requires_grad=True)
            return {"OS": loss}, loss
        def fake_backward(unused):
            for p in created[-1].parameters():
                p.grad = torch.full_like(p, float("inf"))
        program = {"optimizer_factory": lambda current: torch.optim.AdamW(current.parameters(), lr=.001),
                   "forward_fp32": forward}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(torch.Tensor, "backward", fake_backward), \
                patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("update must not run")), \
                patch.object(diagnostic, "attach_backward_observer", return_value=({"fixture": "no numerical backward"}, [])):
            report, ref = diagnostic.run_cell(Path(directory) / "cell", "F2", "fp32", model, inputs, rng,
                                              expected, program, {"fixture": True}, cuda=False)
            self.assertEqual(report["status"], "numerical_failure")
            self.assertEqual(report["update_evidence"], {"attempted": 0, "successful": 0, "skipped": 0, "not_attempted": 1})
            phases = [row["phase"] for row in report["phase_receipts"]]
            self.assertLess(phases.index("gradients_before_unscale"), phases.index("original_guard"))
            self.assertLess(phases.index("gradients_after_unscale"), phases.index("original_guard"))
            self.assertIn("nonfinite_gradients", report["numerical_failures"])
            self.assertEqual(json.loads(Path(ref["path"]).read_text())["status"], "numerical_failure")
            for record in report["phase_receipts"]:
                self.assertEqual(record["sha256"], diagnostic.sha(record["path"]))

    def test_unknown_error_is_retained_without_numerical_relabeling(self):
        model, inputs = tiny_metadata_model(), {"fabricated": torch.ones(1)}
        rng = diagnostic.capture_rng(cuda=False)
        expected = diagnostic.initial_identity(model, inputs, rng)
        def broken(*unused):
            raise RuntimeError("fabricated unexpected kernel error")
        program = {"optimizer_factory": lambda current: torch.optim.AdamW(current.parameters(), lr=.001), "forward_fp32": broken}
        with tempfile.TemporaryDirectory() as directory:
            report, _ = diagnostic.run_cell(Path(directory) / "cell", "F2", "fp32", model, inputs, rng,
                                             expected, program, {}, cuda=False)
            self.assertEqual(report["status"], "unexpected_error")
            self.assertEqual(report["exception"]["phase"], "forward")
            self.assertEqual(report["numerical_failures"], [])


if __name__ == "__main__":
    unittest.main()
