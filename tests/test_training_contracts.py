"""Regression checks for survival labels, checkpoint recovery, and selection."""

import argparse
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from trifusesurv2.multimodal_survival import train
from trifusesurv2.utils import survival
from trifusesurv2.utils.data import PreprocessedContourAwareDataset


class _ScalerState:
    """Exercise scaler serialization without requiring CUDA."""

    def state_dict(self):
        return {"scale": 128.0}

    def load_state_dict(self, state):
        self.scale = state["scale"]


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "last.pt"

    def components(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        ema = survival.EMAWeights(model)
        swa = survival.SWAWeights(model)
        return model, optimizer, _ScalerState(), scheduler, ema, swa

    def checkpoint(self, *, warmup=False):
        model, optimizer, scaler, scheduler, ema, swa = self.components()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        ema.update(model)
        swa.update(model)
        args = argparse.Namespace(
            use_ema=True,
            roi_focus_warmup_epochs=6 if warmup else 0,
            roi_focus_warmup_survival_weight=0.0 if warmup else 1.0,
        )
        train.save_checkpoint(
            self.path, 5, 3, model, args, optimizer, scaler, scheduler, 0.5,
            None if warmup else ema, swa,
        )
        return torch.load(self.path, weights_only=False)

    def restore(self, state):
        torch.save(state, self.path)
        components = self.components()
        with contextlib.redirect_stdout(io.StringIO()):
            result = train.load_checkpoint(self.path, 3, *components)
        return result, components

    def test_complete_roundtrip_restores_parameters_and_training_state(self):
        state = self.checkpoint()
        result, (model, optimizer, scaler, scheduler, ema, swa) = self.restore(state)
        self.assertEqual(result, (6, True))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state["model_state"][name])
        self.assertEqual(scheduler.state_dict(), state["scheduler_state"])
        self.assertEqual(scaler.scale, 128.0)
        self.assertEqual(ema.num_updates, 1)
        self.assertEqual(swa.n_averaged, 1)
        self.assertTrue(optimizer.state)

    def test_missing_model_parameter_is_partial(self):
        state = self.checkpoint()
        del state["model_state"]["bias"]
        self.assertEqual(self.restore(state)[0], (6, False))

    def test_each_required_training_state_is_checked(self):
        original = self.checkpoint()
        for name in ("optimizer_state", "scaler_state", "scheduler_state", "ema", "swa"):
            with self.subTest(name=name):
                state = copy.deepcopy(original)
                state[name] = None
                self.assertEqual(self.restore(state)[0], (6, False))

    def test_incomplete_nested_state_is_partial(self):
        original = self.checkpoint()
        for name in ("ema", "swa"):
            with self.subTest(name=name):
                state = copy.deepcopy(original)
                del state[name]["shadow"]["bias"]
                self.assertEqual(self.restore(state)[0], (6, False))
        state = copy.deepcopy(original)
        del state["scheduler_state"]["last_epoch"]
        self.assertEqual(self.restore(state)[0], (6, False))

    def test_localization_only_warmup_allows_deferred_ema(self):
        self.assertEqual(self.restore(self.checkpoint(warmup=True))[0], (6, True))

    def test_swa_before_first_update_is_valid(self):
        state = self.checkpoint()
        state["swa"]["n_averaged"] = 0
        state["swa"]["shadow"] = {}
        self.assertEqual(self.restore(state)[0], (6, True))

    def test_incomplete_export_restore_does_not_mutate_model(self):
        state = self.checkpoint()
        del state["model_state"]["bias"]
        torch.save(state, self.path)
        model = torch.nn.Linear(2, 1)
        before = copy.deepcopy(model.state_dict())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(train.load_model_state_only(self.path, model))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])


class SelectionTests(unittest.TestCase):
    def test_loss_metrics_select_lower_values_and_keep_raw_reporting(self):
        for metric in ("nll", "ibs"):
            with self.subTest(metric=metric):
                candidates = {"last": {"val": {metric: 1.5}}, "best": {"val": {metric: 0.2}}}
                self.assertEqual(train.select_export_candidate(candidates, metric, "best_val"), "best")
                self.assertEqual(train.report_metric_value(candidates["best"]["val"], metric), 0.2)
                self.assertGreater(train.metric_selection_score(0.2, metric), train.metric_selection_score(1.5, metric))

    def test_discrimination_metrics_still_maximize(self):
        candidates = {"last": {"val": {"c_index": 0.6}}, "ema": {"val": {"c_index": 0.8}}}
        self.assertEqual(train.select_export_candidate(candidates, "c_index", "best_val"), "ema")

    def test_missing_metric_is_never_preferred_to_finite_loss(self):
        candidates = {"last": {"val": {}}, "best": {"val": {"nll": 2.0}}}
        self.assertEqual(train.select_export_candidate(candidates, "nll", "best_val"), "best")
        self.assertEqual(train.metric_selection_score(float("-inf"), "nll"), float("-inf"))

    def test_legacy_ema_priority_is_preserved(self):
        candidates = {"last": {"val": {"nll": 0.1}}, "ema": {"val": {"nll": 2.0}}}
        self.assertEqual(train.select_export_candidate(candidates, "nll", "ema_priority"), "ema")


class SurvivalColumnTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def args(self, endpoint="OS"):
        return argparse.Namespace(
            endpoint=endpoint, time_col="custom_days", event_col="custom_event",
            time_bin_width_days=100.0, max_time_bins=100, risk_horizon_days=100.0,
            auc_times_days=[100.0],
        )

    def test_split_csv_preserves_patient_id_alias_and_literal_identifiers(self):
        csv = self.root / "splits.csv"
        pd.DataFrame({"Patient_ID": ["001", "NA", "003"], "fold": [0, 0, 0],
                      "split": ["train", "val", "test"]}).to_csv(csv, index=False)
        split = train.load_precomputed_splits(1, splits_csv=str(csv))[0]
        self.assertEqual(split, {"train": ["001"], "val": ["NA"], "test": ["003"]})
        pd.DataFrame({"Patient_ID": [None], "fold": [0], "split": ["train"]}).to_csv(csv, index=False)
        with self.assertRaisesRegex(ValueError, "nonmissing"):
            train.load_precomputed_splits(1, splits_csv=str(csv))

    def test_main_rejects_blank_patient_id_before_string_conversion(self):
        csv = self.root / "meta.csv"
        pd.DataFrame({"Patient_ID": [None], "custom_days": [350], "custom_event": [1]}).to_csv(csv, index=False)
        argv = ["train", "--meta_csv", str(csv), "--time_col", "custom_days", "--event_col", "custom_event",
                "--device", "cpu", "--out_dir", str(self.root / "out"), "--no_resume"]
        with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "nonmissing"):
            train.main()

    def test_custom_columns_override_only_primary_head(self):
        for endpoint in train.SURVIVAL_ENDPOINTS:
            with self.subTest(endpoint=endpoint):
                time_cols, event_cols = train.multitask_survival_columns(self.args(endpoint))
                for i, name in enumerate(train.SURVIVAL_ENDPOINTS):
                    expected = ("custom_days", "custom_event") if name == endpoint else train.ENDPOINT_MAP[name]
                    self.assertEqual((time_cols[i], event_cols[i]), expected)

    def test_custom_times_determine_time_bins(self):
        data = pd.DataFrame({"custom_days": [350.0], train.MULTITASK_TIME_COLS[0]: [9000.0]})
        self.assertEqual(train.compute_multitask_num_time_bins(data, self.args()), (4, 350.0))

    def test_dataset_training_and_evaluation_labels_agree(self):
        ct_path = self.root / "ct.nii.gz"
        mask_path = self.root / "mask.nii.gz"
        sitk.WriteImage(sitk.GetImageFromArray(np.full((4, 4, 4), 0.5, dtype=np.float32)), str(ct_path))
        sitk.WriteImage(sitk.GetImageFromArray(np.ones((4, 4, 4), dtype=np.uint8)), str(mask_path))
        meta = pd.DataFrame({
            "patient_id": ["001"], "custom_days": [350.0], "custom_event": [1.0],
            train.MULTITASK_TIME_COLS[0]: [9000.0], train.MULTITASK_EVENT_COLS[0]: [0.0],
            "ct": [str(ct_path)], "pt": [str(mask_path)], "ln": [str(mask_path)],
        })
        time_cols, event_cols = train.multitask_survival_columns(self.args())
        ds = PreprocessedContourAwareDataset(
            meta, id_col="patient_id", time_col="custom_days", event_col="custom_event",
            multi_time_cols=time_cols, multi_event_cols=event_cols,
            ct_col="ct", mask_pt_col="pt", mask_ln_col="ln", clinical_encoder=None,
            radiomics_encoder=None, use_radiomics=False, mode="eval", expected_dhw=(4, 4, 4),
        )
        sample = ds[0]
        self.assertEqual(float(sample[3]), 350.0)
        self.assertEqual(float(sample[5][0]), 350.0)
        self.assertEqual(float(sample[4]), 1.0)
        self.assertEqual(float(sample[6][0]), 1.0)

    def test_main_accepts_custom_only_labels(self):
        csv = self.root / "meta.csv"
        pd.DataFrame({"patient_id": ["001"], "custom_days": [350], "custom_event": [1]}).to_csv(csv, index=False)
        argv = [
            "train", "--meta_csv", str(csv), "--time_col", "custom_days", "--event_col", "custom_event",
            "--device", "cpu", "--out_dir", str(self.root / "out"), "--no_resume", "--cv_folds", "1",
        ]
        result = {"export_risks": {}, "test_metrics_export": {"c_index": 0.5}, "num_time_bins": 4, "export_suffix": "last"}
        with patch("sys.argv", argv), patch.object(train, "resolve_img_size_against_data", return_value=((32, 32, 32), (32, 32, 32))), patch.object(train, "load_precomputed_splits", return_value={0: {}}), patch.object(train, "run_one_fold", return_value=result) as run, contextlib.redirect_stdout(io.StringIO()):
            train.main()
        retained = run.call_args.kwargs["meta"]
        self.assertEqual(retained["patient_id"].tolist(), ["001"])
        self.assertEqual(retained["custom_days"].tolist(), [350])


if __name__ == "__main__":
    unittest.main()
