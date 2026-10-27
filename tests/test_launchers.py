"""Launcher regressions using temporary processes, without training or installs."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest


PACKAGE = Path(__file__).resolve().parents[1]


def write_executable(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).lstrip())
    path.chmod(0o755)
    return path


def process_running(pid: int) -> bool:
    try:
        # A zombie no longer executes or owns GPU work, but may await init's reap.
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


@unittest.skipUnless(sys.platform == "linux" and shutil.which("setsid"), "Linux setsid required")
class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="trifusesurv2-watchdog-test-")
        self.root = Path(self.temporary.name)
        self.process = None
        self.log = (self.root / "wrapper.log").open("w")
        trainer = write_executable(self.root / "trainer.py", r'''
            import os
            from pathlib import Path
            import signal
            import subprocess
            import sys
            import time

            root = Path(os.environ["MOCK_ROOT"])
            if len(sys.argv) > 1:
                (root / "worker.pid").write_text(str(os.getpid()))
                while True:
                    time.sleep(1)

            (root / "trainer.pid").write_text(str(os.getpid()))
            child = subprocess.Popen([sys.executable, __file__, "worker"])

            def stop(signum, frame):
                child.terminate()
                child.wait()
                raise SystemExit(128 + signum)

            if os.environ.get("MOCK_IGNORE_TERM") == "1":
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
            else:
                signal.signal(signal.SIGTERM, stop)
            while not (root / "worker.pid").exists():
                time.sleep(0.01)
            (root / "ready").touch()
            while True:
                child.poll()
                time.sleep(0.05)
        ''')
        python_mock = write_executable(self.root / "python-mock", r'''
            #!/usr/bin/env bash
            if [[ "${1:-}" == "-" ]]; then
              cat >/dev/null
              exit 0
            fi
            exec "$TEST_PYTHON" "$MOCK_TRAINER"
        ''')
        watcher = write_executable(self.root / "watcher.sh", r'''
            #!/usr/bin/env bash
            if [[ "$WATCH" == "0" ]]; then
              exit 0
            fi
            printf '%s\n' "$$" > "$MOCK_ROOT/watcher.pid"
            while [[ ! -f "$MOCK_ROOT/ready" ]]; do
              sleep 0.01
            done
            if [[ "$MOCK_WATCH_ACTION" == "fail" ]]; then
              exit 7
            fi
            exec sleep 300
        ''')
        self.env = {
            **os.environ,
            "PYTHON_BIN": str(python_mock),
            "TEST_PYTHON": sys.executable,
            "MOCK_ROOT": str(self.root),
            "MOCK_TRAINER": str(trainer),
            "MOCK_WATCH_ACTION": "hold",
            "WATCH_SCRIPT": str(watcher),
            "TRAIN_SCRIPT": str(PACKAGE / "scripts/survival/train_contour_aware_survival.sh"),
            "SKIP_REQUIREMENTS_CHECK": "1",
            "DEVICE": "cpu",
            "OUT_DIR": str(self.root / "outputs"),
        }

    def tearDown(self):
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        for pid_file in self.root.glob("*.pid"):
            pid = int(pid_file.read_text())
            if process_running(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.log.close()
        self.temporary.cleanup()

    def launch(self):
        self.process = subprocess.Popen(
            ["bash", str(PACKAGE / "scripts/survival/train_with_roi_focus_watch.sh")],
            env=self.env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )

    def wait_ready(self):
        deadline = time.monotonic() + 10
        while not (self.root / "ready").exists():
            if self.process.poll() is not None or time.monotonic() >= deadline:
                self.fail((self.root / "wrapper.log").read_text())
            time.sleep(0.02)

    def assert_children_stopped(self):
        for name in ("trainer", "worker", "watcher"):
            pid = int((self.root / f"{name}.pid").read_text())
            self.assertFalse(process_running(pid), f"{name} process {pid} survived shutdown")

    def test_watcher_failure_stops_worker_and_escalates_unresponsive_trainer(self):
        self.env.update(MOCK_WATCH_ACTION="fail", MOCK_IGNORE_TERM="1")
        self.launch()
        self.wait_ready()
        self.assertEqual(self.process.wait(timeout=25), 7)
        self.assert_children_stopped()

    def test_termination_exits_wrapper_and_stops_both_process_groups(self):
        self.launch()
        self.wait_ready()
        self.process.terminate()
        self.assertEqual(self.process.wait(timeout=25), 143)
        self.assert_children_stopped()


class InstallerTests(unittest.TestCase):
    def run_installer(self, cuda_version: str | None):
        with tempfile.TemporaryDirectory(prefix="trifusesurv2-installer-test-") as td:
            root = Path(td)
            base = root / "miniforge-mock"
            modules = root / "modules"
            modules.mkdir()
            for name in ("einops", "numpy", "pandas", "pydicom", "SimpleITK", "sklearn", "monai"):
                (modules / f"{name}.py").touch()
            (modules / "torch.py").write_text(
                "from types import SimpleNamespace\n"
                f"version = SimpleNamespace(cuda={cuda_version!r})\n"
            )
            write_executable(base / "bin/conda", f"#!{sys.executable}\n" + textwrap.dedent(r'''
                import os
                import subprocess
                import sys

                args = sys.argv[1:]
                if args == ["info", "--base"]:
                    print(os.environ["MOCK_CONDA_BASE"])
                elif args[0] in {"create", "install"}:
                    pass  # No dependency installation in this test.
                elif args[0] == "run":
                    forwarded_stdin = "--no-capture-output" in args
                    command = args[args.index("python") + 1:]
                    if command != ["-"]:
                        raise SystemExit(0)  # Skip pip's editable install.
                    stdin = sys.stdin if forwarded_stdin else subprocess.DEVNULL
                    raise SystemExit(subprocess.call([sys.executable, "-"], stdin=stdin))
                else:
                    raise SystemExit("unexpected mock conda arguments")
            '''))
            env = {
                **os.environ,
                "MINIFORGE_HOME": str(base),
                "MOCK_CONDA_BASE": str(base),
                "CONDA_ENV_PREFIX": str(root / "environment"),
                "INSTALL_MINIFORGE": "0",
                "CONDA_SOLVER": "conda",
                "PYTHONPATH": str(modules),
                "PYTHONDONTWRITEBYTECODE": "1",
                "EXTRA_CONDA_PACKAGES": "",
                "EXTRA_PIP_PACKAGES": "",
            }
            return subprocess.run(
                ["bash", str(PACKAGE / "scripts/install_env.sh")],
                env=env, capture_output=True, text=True, timeout=15,
            )

    def test_validation_rejects_cpu_only_torch(self):
        result = self.run_installer(None)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("installed PyTorch is CPU-only", result.stderr)
        self.assertNotIn("[done]", result.stdout)

    def test_validation_accepts_cuda_build_without_using_a_gpu(self):
        result = self.run_installer("12.4")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[done]", result.stdout)


if __name__ == "__main__":
    unittest.main()
