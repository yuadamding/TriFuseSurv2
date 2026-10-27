"""Shared warm-start selection and search fingerprints, without tensor imports."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Mapping, Sequence


def candidate_checkpoint_names(args) -> list[str]:
    names = []
    for value in (getattr(args, "contour_warmstart_name", "best.pt"),
                  getattr(args, "shared_seg_pretrain_name", "best.pt"), "best.pt", "seg_best.pt"):
        name = str(value or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _checkpoint_and_directory(args):
    checkpoint = getattr(args, "contour_warmstart_ckpt", "") or getattr(args, "shared_seg_pretrain_ckpt", "") or ""
    directory = getattr(args, "contour_warmstart_dir", "") or getattr(args, "shared_seg_pretrain_dir", "") or ""
    return checkpoint, directory


def resolve_checkpoint_for_config(args) -> str:
    checkpoint, directory = _checkpoint_and_directory(args)
    if checkpoint and os.path.isfile(checkpoint):
        return checkpoint
    if directory:
        for name in candidate_checkpoint_names(args):
            all_path = os.path.join(directory, "all", name)
            if os.path.isfile(all_path):
                return all_path
            for fold in range(int(getattr(args, "cv_folds", 4))):
                path = os.path.join(directory, f"fold_{fold:02d}", name)
                if os.path.isfile(path):
                    return path
    return ""


def resolve_checkpoint_for_fold(args, fold: int) -> str:
    checkpoint, directory = _checkpoint_and_directory(args)
    if checkpoint and os.path.isfile(checkpoint):
        return checkpoint
    if directory:
        for name in candidate_checkpoint_names(args):
            all_path = os.path.join(directory, "all", name)
            if os.path.isfile(all_path):
                return all_path
            fold_path = os.path.join(directory, f"fold_{int(fold):02d}", name)
            if os.path.isfile(fold_path):
                return fold_path
    return checkpoint


def launcher_warmstart_settings(environment: Mapping[str, str | None], arguments: Sequence[str] = ()):
    """Mirror launcher defaults and its trailing CLI overrides for initialization.

    Callers supply the merged inherited/common/per-trial environment. The
    launcher passes only an explicit checkpoint OR a directory, in that order.
    Trailing CLI arguments can subsequently override/add either option.
    """
    checkpoint = environment.get("CONTOUR_WARMSTART_CKPT") or environment.get("SHARED_SEG_PRETRAIN_CKPT") or ""
    directory = environment.get("CONTOUR_WARMSTART_DIR") or environment.get("SHARED_SEG_PRETRAIN_DIR") or ""
    name = environment.get("CONTOUR_WARMSTART_NAME") or environment.get("SHARED_SEG_PRETRAIN_NAME") or "best.pt"
    settings = argparse.Namespace(
        contour_warmstart_ckpt=checkpoint,
        contour_warmstart_dir=directory if not checkpoint else "",
        contour_warmstart_name=name if directory and not checkpoint else "best.pt",
        align_swin_cfg_from_contour_warmstart=bool(checkpoint or directory),
        cv_folds=4, debug_fold=int(environment.get("DEBUG_FOLD") or 3),
    )
    parser = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    for suffix in ("ckpt", "dir", "name"):
        parser.add_argument(f"--contour_warmstart_{suffix}", f"--shared_seg_pretrain_{suffix}",
                            dest=f"contour_warmstart_{suffix}")
    parser.add_argument("--cv_folds", type=int)
    parser.add_argument("--debug_fold", type=int)
    parser.add_argument("--align_swin_cfg_from_contour_warmstart", "--align_swin_cfg_from_seg_ckpt",
                        dest="align_swin_cfg_from_contour_warmstart", action="store_true")
    parser.add_argument("--no_align_swin_cfg_from_contour_warmstart", "--no_align_swin_cfg_from_seg_ckpt",
                        dest="align_swin_cfg_from_contour_warmstart", action="store_false")
    return parser.parse_known_args(list(arguments), namespace=settings)[0]


def warmstart_fingerprints(args, *, cache: dict | None = None) -> dict:
    """Record selected files without deserializing weights; cache is per invocation."""
    cache = {} if cache is None else cache

    def fingerprint(selected):
        if not selected:
            return {"path": None, "exists": False, "sha256": None}
        # Match os.path.isfile in the trainer: quoted '~' is a literal path,
        # not shell expansion. Resolve only after choosing the actual file.
        path = Path(selected).resolve()
        key = str(path)
        if key not in cache:
            exists = path.is_file()
            digest = None
            if exists:
                sha = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        sha.update(block)
                digest = sha.hexdigest()
            cache[key] = {"path": key, "exists": exists, "sha256": digest}
        return dict(cache[key])

    folds = [int(args.debug_fold)] if int(args.debug_fold) >= 0 else list(range(int(args.cv_folds)))
    align = bool(args.align_swin_cfg_from_contour_warmstart)
    return {
        "folds": {str(fold): fingerprint(resolve_checkpoint_for_fold(args, fold)) for fold in folds},
        "config_alignment_enabled": align,
        "config_checkpoint": fingerprint(resolve_checkpoint_for_config(args)) if align else None,
    }
