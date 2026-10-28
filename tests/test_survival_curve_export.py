"""Synthetic export integration; no clinical inputs or model fitting."""
import contextlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from trifusesurv2.evaluation.curves import file_sha256, hazards_to_survival
from trifusesurv2.evaluation.paired import read_literal_csv
from trifusesurv2.multimodal_survival import train


class CurveExportTests(unittest.TestCase):
    def test_curve_checkpoint_restore_rejects_partial_model_states(self):
        model = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial.pt"
            torch.save({"model_state": {"weight": torch.ones(1, 1)}}, path)
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                train.restore_selected_curve_checkpoint(model, path)

    def test_averaged_export_binds_actual_weights_when_source_is_lightweight(self):
        for kind in ("ema", "swa"):
            with self.subTest(weights=kind), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                model = torch.nn.Linear(1, 1, bias=False)
                model.weight.data.fill_(1.)
                average = train.H.EMAWeights(model) if kind == "ema" else train.H.SWAWeights(model)
                if kind == "swa":
                    average.update(model)
                average.shadow["weight"].fill_(4.)
                source, contract, selected = tmp / "last.pt", tmp / "contract.json", tmp / "selected.pt"
                torch.save({"model_state": model.state_dict(), "ema": None, "swa": None}, source)
                contract.write_text('{}')
                with average.apply_to(model):
                    train.persist_selected_inference_weights(model, selected, source_checkpoint=source,
                                                              contract_path=contract, weights=kind)
                persisted = torch.load(selected, weights_only=False)
                self.assertEqual(persisted["model_state"]["weight"].item(), 4.)
                self.assertEqual(model.weight.item(), 1.)
                self.assertEqual(persisted["source_checkpoint_sha256"], file_sha256(source))
                self.assertEqual(persisted["weights"], kind)
                with self.assertRaises(FileExistsError):
                    train.persist_selected_inference_weights(model, selected, source_checkpoint=source,
                                                              contract_path=contract, weights=kind)

    def test_selected_endpoint_ids_outcomes_and_weights_are_traceable(self):
        model = torch.nn.Linear(1, 1)
        payload = {"x": torch.zeros(2, 1), "t": torch.tensor([5., 8.]),
                   "e": torch.tensor([1., 0.]), "pid": ["001", "NA"]}
        outcomes = pd.DataFrame({"Patient_ID": ["NA", "001"], "OS.TIME": [8.25, 5.25], "OS.EVENT": [0, 1]})
        args = SimpleNamespace(id_col="Patient_ID", time_col="OS.TIME", event_col="OS.EVENT",
                               endpoint="OS", fusion_mode="F3", seed=42, time_bin_width_days=10.,
                               risk_horizon_days=15., curve_export_times_days=[5.])
        logits = torch.logit(torch.tensor([[.2, .5], [.1, .4]]))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            checkpoint, contract, output = tmp / "best.pt", tmp / "model_contract.json", tmp / "curves.csv"
            checkpoint.write_bytes(b"synthetic selected checkpoint")
            contract.write_text('{"synthetic":true}')
            with mock.patch.object(train, "_model_forward_eval", return_value={"OS": logits, "DSS": logits + 2}):
                train.export_survival_curves(model, [payload], torch.device("cpu"), outcome_frame=outcomes,
                    args=args, fold=0, split="test", checkpoint_path=checkpoint, contract_path=contract,
                    weights="best", autocast_ctx=contextlib.nullcontext, teacher_force_alpha=0., output_path=output)
            rows = read_literal_csv(output)
            self.assertEqual(rows["Patient_ID"].drop_duplicates().tolist(), ["NA", "001"])
            self.assertEqual(rows.loc[rows["Patient_ID"].eq("NA"), "observed_time"].unique().tolist(), ["8.25"])
            self.assertEqual(rows["checkpoint_sha256"].unique().tolist(), [file_sha256(checkpoint)])
            self.assertEqual(rows["model_contract_sha256"].unique().tolist(), [file_sha256(contract)])
            self.assertEqual(rows["evaluation_time"].astype(float).unique().tolist(), [0, 5, 10, 15, 20])
            expected = hazards_to_survival(torch.sigmoid(logits).numpy()[[1, 0]], [0, 5, 10, 15, 20], 10)
            np.testing.assert_allclose(rows["survival_probability"].astype(float).to_numpy().reshape(2, -1), expected)
            receipt = json.loads(output.with_suffix(".csv.json").read_text())
            self.assertEqual(receipt["metadata"]["weights"], "best")
            self.assertEqual(receipt["csv_sha256"], file_sha256(output))

    def test_duplicate_loader_ids_fail(self):
        payload = {"x": torch.zeros(2, 1), "t": torch.ones(2), "e": torch.ones(2), "pid": ["001", "001"]}
        with mock.patch.object(train, "_model_forward_eval", return_value=torch.zeros(2, 2)):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                train.export_survival_curves(torch.nn.Linear(1, 1), [payload], torch.device("cpu"),
                    outcome_frame=pd.DataFrame(), args=SimpleNamespace(endpoint="OS"), fold=0, split="test",
                    checkpoint_path=None, contract_path=None, weights="best", autocast_ctx=contextlib.nullcontext,
                    teacher_force_alpha=0., output_path=None)


if __name__ == "__main__":
    unittest.main()
