"""Mocked CPU orchestration artifact producer, never a CUDA qualification.

All forwards, backwards, optimizer calls and scaler behavior are simulated.
Only identity hashing, structured observation and receipt writers are real.
"""
from pathlib import Path
from collections import OrderedDict
from unittest.mock import patch

import torch

from test_retained_cuda_diagnostic import diagnostic, tiny_metadata_model, ORIGINAL


class MockAdamW:
    def __init__(self, model):
        self.parameters = list(model.parameters())
        self.defaults = {"lr": .001, "betas": (.9, .999), "eps": 1e-8, "weight_decay": .01,
                         "amsgrad": False, "foreach": None, "maximize": False, "capturable": False,
                         "differentiable": False, "fused": None, "decoupled_weight_decay": True}
        self.state, self.hooks = {}, OrderedDict()

    def state_dict(self):
        return {"state": {i: self.state[p] for i, p in enumerate(self.parameters) if p in self.state},
                "param_groups": [{**self.defaults, "params": list(range(len(self.parameters)))}]}

    def register_step_post_hook(self, hook):
        handle = torch.utils.hooks.RemovableHandle(self.hooks)
        self.hooks[handle.id] = hook
        return handle

    def step(self):
        for p in self.parameters:
            if p.grad is not None:
                self.state[p] = {"step": torch.tensor(1.), "exp_avg": torch.ones_like(p), "exp_avg_sq": torch.ones_like(p)}
        for hook in self.hooks.values():
            hook(self, (), {})


class MockGradScaler:
    def __init__(self, scale):
        self.value = scale

    def state_dict(self):
        return {"scale": self.value, "growth_factor": 2., "backoff_factor": .5, "growth_interval": 2000, "_growth_tracker": 0}

    def get_scale(self):
        return self.value

    def scale(self, loss):
        return loss  # Intentional simulated scaling, declared in FIXTURE.json.

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


def produce(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    marker = {"purpose": "CPU metadata orchestration fixture only", "simulated": ["forward", "backward", "optimizer", "scaler", "GPU", "source_environment"],
              "clinical_inputs": 0, "clinical_fits": 0, "CUDA_calls": 0, "numerical_matrix_executed": False}
    diagnostic.write_json(directory / "FIXTURE.json", marker)
    plan = directory / "SYNTHETIC_PLAN.json"
    diagnostic.write_json(plan, {"fixture": marker})
    bindings = {"model_source": {"path": "SYNTHETIC-NO-MODEL-SOURCE", "identity": {
                    "commit_sha": None, "source_dirty": None, "source_tree_sha256": "0" * 64, "declared_commit_sha": "0" * 40}},
                "original_fixture": diagnostic.reference(ORIGINAL), "original_plan": diagnostic.reference(plan),
                "diagnostic_files": [diagnostic.reference(Path(diagnostic.__file__)),
                                     diagnostic.reference(Path(diagnostic.__file__).with_name("diagnose_retained_cuda_fixture.py"))]}
    runtime = {"fixture": marker, "python_executable": "SIMULATED", "python_version": "SIMULATED", "torch": "SIMULATED",
               "cuda": "SIMULATED", "cudnn": "SIMULATED", "gpu": {"name": "SIMULATED-H100", "total_memory": 1,
                    "uuid": "SIMULATED", "visible_devices": 1, "nvidia_smi": "SIMULATED"},
               "container_image_declaration": "SIMULATED", "backend": diagnostic.backend_settings(),
               "incoming_commit_declaration": None, "original_source_commit_declaration": "0" * 40}
    original = diagnostic.compile_original_program(ORIGINAL)
    summary = diagnostic.make_summary(bindings, runtime, original["extracted"])
    diagnostic.write_json(directory / "STARTUP.json", summary)
    for mode in ("F2", "F3"):
        mode_dir = directory / mode; mode_dir.mkdir()
        model, inputs = tiny_metadata_model(), {"fabricated": torch.ones(1), "mask": torch.tensor([True])}
        rng = diagnostic.capture_rng(cuda=False)
        materialization = {"lazy_names_before": [], "lazy_names_after": [], "materialization_forward_count": 0,
                           "materialization_precision": None, "exact_original_initialization": True}
        summary["mode_fixtures"][mode] = diagnostic.retain_initial_fixture(mode_dir, mode, model, inputs, rng,
             materialization, {"mode": "explicit_source_columns", "columns": ["AGE"]}, MockAdamW(model), MockGradScaler(65536.))
        expected = summary["mode_fixtures"][mode]["identity"]
        for cell in diagnostic.CELLS:
            models = []
            def forward(current, unused):
                models.append(current)
                loss = torch.tensor(1., requires_grad=True)
                return {"OS": loss, "DSS": loss, "DFS": loss}, loss
            def backward(unused):
                for name, p in models[-1].named_parameters():
                    if name != "unused":
                        p.grad = torch.full_like(p, float("inf") if cell == "original_fp16" else 1.)
            program = {"optimizer_factory": MockAdamW, "original_scaler": lambda: MockGradScaler(65536.),
                       "scale_one": lambda: MockGradScaler(1.), "forward_fp32": forward, "forward_fp16": forward}
            with patch.object(torch.Tensor, "backward", backward), \
                    patch.object(diagnostic, "attach_backward_observer", return_value=({"fixture": marker}, [])):
                report, ref = diagnostic.run_cell(mode_dir / cell, mode, cell, model, inputs, rng, expected,
                                                  program, bindings, cuda=False)
            summary["cells"].append({"mode": mode, "cell": cell, "status": report["status"],
                                     "numerical_passed": report["numerical_passed"], "report": ref})
    code = diagnostic.finalize_summary(summary, directory, True)
    if code or summary["cell_counts"] != {"numerically_passed": 4, "numerical_failure": 2, "unexpected_error": 0}:
        raise AssertionError("Mocked producer flow failed: " + repr(summary["cell_counts"]))
    return summary
