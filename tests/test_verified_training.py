"""One synthetic CPU fold through fitting, training, export and strict resume."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

import pandas as pd
import torch

import test_region_data as fixtures
from trifusesurv2.multimodal_survival import train


class VerifiedTrainingTests(unittest.TestCase):
    def test_verified_fold_and_resume_bind_the_same_fusion_and_inputs(self):
        fixture = fixtures.RegionDataTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        rows, features = [], []
        for index in range(6):
            row, manifest, _job = fixture.case(f"{index + 1:03d}")
            row.update(time=120.0 + index * 120.0, event=float(index % 2 == 0), AGE=50 + index,
                       HPV=index % 2, T=2, N=1)
            rows.append(row)
            features.append(fixtures.extract.extract_case(manifest, fixtures.FeatureExtractor(), fixture.config, software_version="test-adapter"))
        meta = pd.DataFrame(rows)
        metadata_path, radiomics_path = fixture.output / "meta.csv", fixture.output / "radiomics.csv"
        meta.to_csv(metadata_path, index=False)
        pd.DataFrame(features).to_csv(radiomics_path, index=False)
        argv = [
            "train", "--meta_csv", str(metadata_path), "--radiomics_root", str(radiomics_path),
            "--ct_col", "ct", "--mask_pt_col", "pt", "--mask_ln_col", "ln",
            "--time_col", "time", "--event_col", "event", "--device", "cpu",
            "--img_size", "32", "32", "32", "--feature_size", "12", "--depths", "1", "1", "1", "1",
            "--img_token_dim", "16", "--v2_model_dim", "16", "--v2_num_heads", "4",
            "--v2_transformer_layers", "1", "--v2_radiomics_pcs_per_group", "2",
            "--epochs", "1", "--batch_size", "2", "--eval_batch_size", "2",
            "--workers", "0", "--eval_workers", "0", "--no_ema", "--no_swa",
            "--no_resume", "--no_export_extra_risks", "--risk_horizon_days", "240",
            "--auc_times_days", "120", "240", "--time_bin_width_days", "120",
        ]
        with patch("sys.argv", argv):
            args = train.parse_args()
        split = {"train": ["001", "002"], "val": ["003", "004"], "test": ["005", "006"]}
        out_root = fixture.root / "run"
        kwargs = dict(fold=0, args=args, meta=meta, split=split, device=torch.device("cpu"),
                      scaler=None, autocast_ctx=contextlib.nullcontext, out_root=out_root)
        with contextlib.redirect_stdout(io.StringIO()):
            result = train.run_one_fold(**kwargs)
        self.assertEqual(set(result["export_risks"]), {"005", "006"})
        contract_path = out_root / "fold_00" / "model_contract.json"
        before = contract_path.read_bytes()
        contract = json.loads(before)
        self.assertEqual(contract["fusion_mode"], "F3")
        self.assertEqual(len(contract["region_manifest_sha256"]), 6)
        self.assertGreater(contract["parameters"], 0)
        args.resume = True
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = train.run_one_fold(**kwargs)
        self.assertEqual(set(resumed["export_risks"]), {"005", "006"})
        args.fusion_mode = "F2"
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "fusion_mode"):
            train.run_one_fold(**kwargs)
        self.assertEqual(contract_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
