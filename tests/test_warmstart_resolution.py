"""Warm-start selection matches trainer and launcher priority without loading weights."""
import argparse
from pathlib import Path
import tempfile
import unittest

from trifusesurv2.utils.warmstart import (
    candidate_checkpoint_names, launcher_warmstart_settings,
    resolve_checkpoint_for_config, resolve_checkpoint_for_fold, warmstart_fingerprints,
)


class WarmstartResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="trifusesurv2-warmstart-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def checkpoint(self, path, contents=b"synthetic weights"):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return str(path)

    def test_filename_priority_and_all_fold_selection_are_preserved(self):
        fold_custom = self.checkpoint("weights/fold_03/custom.pt")
        all_best = self.checkpoint("weights/all/best.pt")
        args = argparse.Namespace(contour_warmstart_dir=str(self.root / "weights"),
                                  contour_warmstart_name="custom.pt", cv_folds=4)
        self.assertEqual(candidate_checkpoint_names(args), ["custom.pt", "best.pt", "seg_best.pt"])
        self.assertEqual(resolve_checkpoint_for_fold(args, 3), fold_custom)
        self.assertEqual(resolve_checkpoint_for_fold(args, 2), all_best)
        all_custom = self.checkpoint("weights/all/custom.pt")
        self.assertEqual(resolve_checkpoint_for_fold(args, 3), all_custom)
        explicit = self.checkpoint("explicit.pt")
        args.contour_warmstart_ckpt = explicit
        self.assertEqual(resolve_checkpoint_for_fold(args, 3), explicit)
        Path(explicit).unlink()
        self.assertEqual(resolve_checkpoint_for_fold(args, 3), all_custom)

    def test_launcher_legacy_aliases_and_trailing_cli_overrides(self):
        primary = str(self.root / "missing-explicit.pt")
        chosen = self.checkpoint("weights/fold_03/chosen.pt")
        environment = {"CONTOUR_WARMSTART_CKPT": primary,
                       "SHARED_SEG_PRETRAIN_DIR": str(self.root / "weights"),
                       "SHARED_SEG_PRETRAIN_NAME": "chosen.pt"}
        settings = launcher_warmstart_settings(environment)
        # An explicit env checkpoint suppresses directory/name env arguments.
        self.assertEqual(settings.contour_warmstart_dir, "")
        self.assertEqual(resolve_checkpoint_for_fold(settings, 3), primary)
        settings = launcher_warmstart_settings(environment, [
            "--shared_seg_pretrain_dir", str(self.root / "weights"),
            "--shared_seg_pretrain_name=chosen.pt",
        ])
        self.assertEqual(resolve_checkpoint_for_fold(settings, 3), chosen)
        explicit = self.checkpoint("replacement.pt")
        settings = launcher_warmstart_settings(environment, ["--shared_seg_pretrain_ckpt", explicit])
        self.assertEqual(resolve_checkpoint_for_fold(settings, 3), explicit)
        abbreviated = launcher_warmstart_settings(environment, ["--shared_seg_pretrain_c", explicit])
        self.assertEqual(resolve_checkpoint_for_fold(abbreviated, 3), explicit)
        alias = launcher_warmstart_settings({"SHARED_SEG_PRETRAIN_DIR": str(self.root / "weights"),
                                            "SHARED_SEG_PRETRAIN_NAME": "chosen.pt"})
        self.assertEqual(resolve_checkpoint_for_fold(alias, 3), chosen)

    def test_config_source_and_every_requested_fold_are_fingerprinted(self):
        first = self.checkpoint("weights/fold_00/best.pt", b"config source")
        last = self.checkpoint("weights/fold_03/best.pt", b"fold3 source")
        settings = launcher_warmstart_settings({"CONTOUR_WARMSTART_DIR": str(self.root / "weights")})
        selected = warmstart_fingerprints(settings)
        self.assertEqual(selected["config_checkpoint"]["path"], first)
        self.assertEqual(selected["folds"]["3"]["path"], last)
        self.assertEqual(resolve_checkpoint_for_config(settings), first)
        Path(first).write_bytes(b"changed configuration source")
        self.assertNotEqual(selected, warmstart_fingerprints(settings))
        settings = launcher_warmstart_settings(
            {"CONTOUR_WARMSTART_DIR": str(self.root / "weights")},
            ["--debug_fold=-1", "--cv_folds=4", "--no_align_swin_cfg_from_seg_ckpt"],
        )
        selected = warmstart_fingerprints(settings)
        self.assertEqual(set(selected["folds"]), {"0", "1", "2", "3"})
        self.assertFalse(selected["config_alignment_enabled"])
        self.assertIsNone(selected["config_checkpoint"])
        self.assertEqual(selected["folds"]["1"], {"path": None, "exists": False, "sha256": None})


if __name__ == "__main__":
    unittest.main()
