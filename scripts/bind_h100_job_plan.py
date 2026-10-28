#!/usr/bin/env python3
"""Check or bind a rendered Job's plan metadata; never contact Kubernetes.

Use before sealing a new workload. Existing execution manifests and receipts
must be retained unchanged. This checks plan identity, not cluster admission,
resource capacity, scientific qualification, or permission to launch.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path


HASH_KEY = "trifusesurv2_plan_sha256"
RUN_KEY = "seadragon_run_id"


def check(job, plan, plan_sha256):
    containers = job["spec"]["template"]["spec"].get("containers", [])
    if job.get("apiVersion") != "batch/v1" or job.get("kind") != "Job" or len(containers) != 1:
        raise ValueError("Expected a batch/v1 Job with exactly one worker.")
    if job["metadata"]["name"] != plan["run_id"]:
        raise ValueError("Job name differs from the bound run identity.")
    for metadata in (job["metadata"], job["spec"]["template"]["metadata"]):
        annotations = metadata.get("annotations", {})
        if annotations.get(HASH_KEY) != plan_sha256 or annotations.get(RUN_KEY) != plan["run_id"]:
            raise ValueError("Job or Pod-template annotation has a stale plan/run identity.")
    container = containers[0]
    env = container.get("env", [])
    if len({item["name"] for item in env}) != len(env):
        raise ValueError("Duplicate environment variable names are ambiguous.")
    values = {item["name"]: item for item in env}
    for key, value in (("TRIFUSE_PLAN_SHA256", plan_sha256), ("TRIFUSE_RUN_ID", plan["run_id"])):
        if values.get(key) != {"name": key, "value": value}:
            raise ValueError("Worker environment has a stale or indirect plan/run identity.")
    expected = [plan["python_executable"], "-B", "-u", str(Path(plan["bundle_root"]) / "worker.py"),
                "--plan", str(Path(plan["bundle_root"]) / "PLAN.json")]
    command, arguments = container.get("command", []), container.get("args", [])
    if not isinstance(command, list) or not isinstance(arguments, list) or command + arguments != expected:
        raise ValueError("Worker command does not select the exact plan and executable.")
    if container.get("image") != plan["pinned_image"]:
        raise ValueError("Worker image differs from the plan.")


def bind(job, plan, plan_sha256):
    """Return a fresh rendering; preserve all execution settings and the input."""
    rendered = copy.deepcopy(job)
    containers = rendered["spec"]["template"]["spec"].get("containers", [])
    if len(containers) != 1:
        raise ValueError("Expected exactly one worker.")
    env = containers[0].get("env", [])
    if len({item["name"] for item in env}) != len(env):
        raise ValueError("Duplicate environment variable names are ambiguous.")
    for metadata in (rendered["metadata"], rendered["spec"]["template"]["metadata"]):
        metadata.setdefault("annotations", {}).update({HASH_KEY: plan_sha256, RUN_KEY: plan["run_id"]})
    env = [item for item in env if item["name"] not in ("TRIFUSE_PLAN_SHA256", "TRIFUSE_RUN_ID")]
    env.extend([{"name": "TRIFUSE_PLAN_SHA256", "value": plan_sha256},
                {"name": "TRIFUSE_RUN_ID", "value": plan["run_id"]}])
    containers[0]["env"] = env
    check(rendered, plan, plan_sha256)
    return rendered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Write a fresh corrected rendering; omit for check only.")
    args = parser.parse_args(argv)
    raw = args.plan.read_bytes()
    plan, job = json.loads(raw), json.loads(args.job.read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    if args.output is None:
        check(job, plan, digest)
    else:
        rendered = bind(job, plan, digest)
        data = (json.dumps(rendered, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        fd = os.open(str(args.output), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"status": "passed", "plan_sha256": digest, "cluster_contacted": False}))


if __name__ == "__main__":
    main()
