"""Plan metadata regressions on fabricated manifests; no cluster dependencies."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("bind_h100_job_plan", Path(__file__).parents[1] / "scripts/bind_h100_job_plan.py")
binder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(binder)


class JobBindingTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"run_id": "synthetic-run", "python_executable": "/synthetic/python", "bundle_root": "/synthetic/bundle", "pinned_image": "example@sha256:synthetic"}
        self.job = {"apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": "synthetic-run", "annotations": {}}, "spec": {"template": {"metadata": {"annotations": {}}, "spec": {"containers": [{"image": self.plan["pinned_image"], "command": ["/synthetic/python", "-B", "-u", "/synthetic/bundle/worker.py", "--plan", "/synthetic/bundle/PLAN.json"], "env": [{"name": "OMP_NUM_THREADS", "value": "8"}]}]}}}}

    def test_stale_annotation_is_rejected_even_with_correct_worker_environment(self):
        rendered = binder.bind(self.job, self.plan, "new-plan")
        for metadata in (rendered["metadata"], rendered["spec"]["template"]["metadata"]):
            metadata["annotations"][binder.HASH_KEY] = "old-plan"
            with self.assertRaisesRegex(ValueError, "annotation"):
                binder.check(rendered, self.plan, "new-plan")
            metadata["annotations"][binder.HASH_KEY] = "new-plan"

    def test_binding_preserves_input_and_execution_settings(self):
        original = copy.deepcopy(self.job)
        rendered = binder.bind(self.job, self.plan, "new-plan")
        binder.check(rendered, self.plan, "new-plan")
        self.assertEqual(self.job, original)
        before = self.job["spec"]["template"]["spec"]["containers"][0]
        after = rendered["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(before["command"], after["command"])
        self.assertIn(before["env"][0], after["env"])

    def test_separate_kubernetes_command_and_args_preserve_the_same_argv(self):
        container = self.job["spec"]["template"]["spec"]["containers"][0]
        container["args"] = container["command"][-2:]
        container["command"] = container["command"][:-2]
        rendered = binder.bind(self.job, self.plan, "new-plan")
        binder.check(rendered, self.plan, "new-plan")
        self.assertEqual(rendered["spec"]["template"]["spec"]["containers"][0]["args"], container["args"])

    def test_ambiguous_environment_and_wrong_executable_fail(self):
        for changed in ("duplicate", "command"):
            job = copy.deepcopy(self.job)
            container = job["spec"]["template"]["spec"]["containers"][0]
            if changed == "duplicate":
                container["env"] *= 2
            else:
                container["command"][0] = "/different/python"
            with self.assertRaises(ValueError):
                binder.bind(job, self.plan, "new-plan")

    def test_cli_never_overwrites_a_bound_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, job, output = root / "plan.json", root / "job.json", root / "output.json"
            plan.write_text(json.dumps(self.plan)); job.write_text(json.dumps(self.job))
            args = ["--plan", str(plan), "--job", str(job), "--output", str(output)]
            binder.main(args)
            previous = output.read_bytes()
            with self.assertRaises(FileExistsError):
                binder.main(args)
            self.assertEqual(previous, output.read_bytes())
            binder.main(["--plan", str(plan), "--job", str(output)])


if __name__ == "__main__":
    unittest.main()
