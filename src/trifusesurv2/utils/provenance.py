"""Report observed source identity without assigning a historical commit by default."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess


@lru_cache(maxsize=None)
def source_identity(root: Path | None = None) -> dict:
    """Snapshot installed Python/scripts; report Git identity only when observable.

    A dirty checkout is identified by its actual source digest as well as HEAD.
    An environment-provided SHA is retained as a declaration, never as evidence.
    The snapshot is cached per process; runs must not hot-edit loaded source.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    files = sorted(
        p for directory in (root / "src", root / "scripts") if directory.is_dir()
        for p in directory.rglob("*") if p.is_file() and p.suffix in {".py", ".sh"}
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    identity = {"commit_sha": None, "source_dirty": None,
                "source_tree_sha256": digest.hexdigest() if files else None,
                "declared_commit_sha": os.environ.get("TRIFUSESURV2_COMMIT_SHA") or None}
    try:
        def git(*args):
            return subprocess.run(["git", "-C", str(root), *args], check=True,
                                  capture_output=True, text=True, timeout=5).stdout.strip()
        if Path(git("rev-parse", "--show-toplevel")).resolve() == root.resolve():
            identity["commit_sha"] = git("rev-parse", "HEAD")
            identity["source_dirty"] = bool(git("status", "--porcelain", "--untracked-files=normal"))
    except (OSError, subprocess.SubprocessError):
        pass
    return identity


def source_contract_identity(root: Path | None = None) -> dict:
    """Bind executable source, excluding declarations and incidental run outputs.

    The full identity retains the observed Git dirty flag for reporting. Creating
    logs/results inside a checkout can change that flag without changing code.
    """
    observed = source_identity(root)
    return {name: observed[name] for name in ("commit_sha", "source_tree_sha256")}
