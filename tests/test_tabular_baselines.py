"""Fabricated engineering cases; these do not qualify any clinical cohort."""
from copy import deepcopy
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sksurv.metrics import integrated_brier_score

from trifusesurv2.baselines.cox import (fit_cox_baseline, serializable_model,
                                      survival_probabilities, survival_target)
from trifusesurv2.baselines.features import (TabularPreprocessor, clinical_columns,
                                           geometry_features, sha256_file)
from trifusesurv2.baselines.runner import load_splits, main, validate_protocol
from trifusesurv2.evaluation.paired import outcome_inventory_sha256
from trifusesurv2.preprocessing.compartments import ReferenceGeometry, build_compartments


def clinical_policy():
    return {"status": "confirmed", "prediction_time": "fabricated pretreatment fixture",
            "coding_source": "fabricated fixture, not cohort coding", "numeric": ["AGE"],
            "categorical": ["T", "N", "M"], "redundant_columns": {
                "NSTAGE": {"confirmed": True, "represented_by": ["T", "N", "M"],
                           "justification": "synthetic composite generated from TNM"}}}


def synthetic_frames():
    rng = np.random.default_rng(812)
    def frame(n, start):
        age = rng.normal(60, 8, n)
        return pd.DataFrame({"AGE": age, "T": [f"T{x % 3 + 1}" for x in range(n)],
                             "N": [f"N{x % 2}" for x in range(n)], "M": ["M0"] * n,
                             "NSTAGE": ["fixture"] * n,
                             "OS.TIME": np.linspace(2, 70 if start else 100, n),
                             "OS.EVENT": [int(x % 4 != 0) for x in range(n)]},
                            index=[f"{start + x:04d}" for x in range(n)])
    return frame(32, 0), frame(16, 100), frame(12, 200)


def confirmed_spec():
    return {"status": "frozen", "endpoint_definition": {
                "status": "confirmed", "time_origin": "synthetic zero", "time_unit": "days",
                "event_definition": "fabricated event=1", "censoring_rule": "synthetic censoring"},
            "evaluation": {"endpoint": "OS", "fold": 0, "split": "test", "modes": ["C0", "C1"],
                           "seeds": ["deterministic"], "primary_horizon": 20.,
                           "secondary_horizons": [10., 40.], "ibs_times": [10., 20., 40.],
                           "concordance_tau": 40., "calibration_bins": [0., .5, 1.],
                           "bootstrap_replicates": 20, "bootstrap_seed": 15,
                           "bootstrap_min_valid_fraction": .5, "tied_tol": 1e-8,
                           "outcome_inventory_sha256": "a" * 64},
            "baselines": {"status": "frozen", "clinical": clinical_policy(),
                          "fusion_clinical_columns": ["AGE", "T", "N", "M", "NSTAGE"],
                          "selection": "validation_ibs", "alpha_grid": [.1, 1., 10.],
                          "ties": "efron", "n_iter": 100, "tol": 1e-9}}


class TabularBaselineTests(unittest.TestCase):
    def test_raw_csv_runner_preserves_categories_and_fits_only_training_vocabulary(self):
        train, val, test = synthetic_frames()
        categorical = ["PATHOLOGY", "SEX", "RACE", "T", "N", "M", "KFCF", "SMOKE", "ALCOHOL", "HPV"]
        literals = ["NA", "nd", "ND", "01", "1", "1.0", "", "   ", " nd ", "__missing__", "__unseen__"]
        rows = []
        for split, frame in (("train", train), ("val", val), ("test", test)):
            for index, (pid, row) in enumerate(frame.iterrows()):
                cells = {"Patient_ID": pid, "AGE": row["AGE"], "OS.TIME": row["OS.TIME"],
                         "OS.EVENT": row["OS.EVENT"], "NSTAGE": "excluded-stage", "TX": "excluded-treatment"}
                for offset, name in enumerate(categorical):
                    cells[name] = literals[(index + offset) % len(literals)]
                if split != "train":
                    cells.update({name: split + "-only-category" for name in categorical})
                    cells["AGE"] += 20
                # Keep this entire source column numeric-looking, including
                # held-out rows, so pandas would otherwise merge 01/1/1.0.
                cells["KFCF"] = (["01", "1", "1.0", "2", ""][index % 5] if split == "train"
                                  else "3" if split == "val" else "4")
                rows.append(cells)
        rows[0]["AGE"], rows[1]["AGE"] = "", " \t"
        with TemporaryDirectory() as temporary:
            root = Path(temporary); splitdir = root / "splits"; splitdir.mkdir()
            for split, frame in (("train", train), ("val", val), ("test", test)):
                (splitdir / f"{split}_ids.txt").write_text("\n".join(frame.index) + "\n")
            metadata = root / "literal.csv"; spec_path = root / "spec.json"
            def write_csv():
                with metadata.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
            write_csv()
            spec = confirmed_spec()
            spec["baselines"]["clinical"]["categorical"] = categorical
            spec["baselines"]["fusion_clinical_columns"] = ["AGE", *categorical, "NSTAGE"]
            spec["baselines"]["split_files_sha256"] = load_splits(splitdir)[1]
            spec["baselines"]["metadata_sha256"] = sha256_file(metadata)
            outcomes = pd.DataFrame(rows)[["Patient_ID", "OS.TIME", "OS.EVENT"]].rename(
                columns={"OS.TIME": "observed_time", "OS.EVENT": "event"})
            outcomes["endpoint"] = "OS"
            outcomes["split"] = ["train"] * len(train) + ["val"] * len(val) + ["test"] * len(test)
            spec["evaluation"]["outcome_inventory_sha256"] = outcome_inventory_sha256(outcomes, "OS")
            spec_path.write_text(json.dumps(spec))
            fitted, read_frames = [], []
            def capture(training, validation, **kwargs):
                read_frames.extend([training.copy(), validation.copy()])
                result = fit_cox_baseline(training, validation, **kwargs)
                fitted.append(result[1])
                return result
            argv = ["--study-spec", str(spec_path), "--metadata-csv", str(metadata),
                    "--split-dir", str(splitdir), "--modes", "C0", "--out-dir", str(root / "out")]
            with patch("trifusesurv2.baselines.runner.fit_cox_baseline", side_effect=capture):
                main(argv)
            encoder = fitted[0]
            for name in categorical:
                with self.subTest(column=name):
                    expected = sorted({row[name] for row in rows[:len(train)] if row[name].strip()})
                    self.assertEqual(encoder.categories[name], expected)
                    self.assertEqual(read_frames[0][name].tolist(), [row[name] for row in rows[:len(train)]])
            self.assertEqual(encoder.categories["KFCF"], ["01", "1", "1.0", "2"])
            self.assertEqual(encoder.numeric, ["AGE"])
            self.assertEqual(encoder.categorical, categorical)
            self.assertFalse(set(encoder.categories) & {"NSTAGE", "TX", "T_RAW", "N_RAW", "M_RAW"})
            expected_age = train.AGE.copy(); expected_age.iloc[:2] = np.nan
            self.assertAlmostEqual(encoder.parameters["AGE"]["median"], expected_age.median())
            self.assertAlmostEqual(encoder.parameters["AGE"]["mean"], expected_age.fillna(expected_age.median()).mean())
            self.assertEqual(encoder.parameters["AGE"]["observed_training_count"], len(train) - 2)
            self.assertEqual(read_frames[0]["AGE"].tolist(), [str(row["AGE"]) for row in rows[:len(train)]])
            for name in ("OS.TIME", "OS.EVENT"):
                self.assertTrue(pd.api.types.is_numeric_dtype(read_frames[0][name]))
            before = deepcopy(encoder.to_dict())
            encoder.transform(read_frames[1])
            raw, names = encoder._raw_transform(read_frames[1])
            self.assertTrue((raw[:, names.index("HPV::unseen")] == 1).all())
            self.assertTrue((raw[:, names.index("HPV::missing")] == 0).all())
            self.assertTrue((raw[:, names.index("KFCF::unseen")] == 1).all())
            self.assertEqual(encoder.to_dict(), before)
            checkpoint = json.loads((root / "out/C0/selected_model.json").read_text())
            contract = json.loads((root / "out/C0/model_contract.json").read_text())
            self.assertEqual(checkpoint["preprocessor"], before)
            self.assertEqual(contract["clinical_csv_ingestion"]["categorical_columns"], categorical)
            self.assertEqual(contract["clinical_csv_ingestion"]["categorical_value_policy"],
                             before["categorical_value_policy"])
            # Preserve invalid numeric literals until the existing strict guard,
            # rather than allowing pandas' NA inference to hide them as blanks.
            for index, value in enumerate(("not-a-number", "NA", "NaN")):
                with self.subTest(invalid_raw_age=value):
                    rows[0]["AGE"] = value
                    write_csv(); spec["baselines"]["metadata_sha256"] = sha256_file(metadata)
                    spec_path.write_text(json.dumps(spec)); argv[-1] = str(root / f"bad-age-{index}")
                    with self.assertRaisesRegex(ValueError, "numeric coding in AGE"):
                        main(argv)

    def test_actual_missing_values_and_marker_like_categories_remain_distinct(self):
        frame = pd.DataFrame({"HPV": [None, np.nan, pd.NA, "", " \t", "NA", "__missing__", "__unseen__", " NA "]})
        encoder = TabularPreprocessor([], ["HPV"]).fit(frame)
        raw, names = encoder._raw_transform(frame)
        self.assertEqual(encoder.categories["HPV"], [" NA ", "NA", "__missing__", "__unseen__"])
        np.testing.assert_array_equal(raw[:, names.index("HPV::missing")], [1, 1, 1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(raw[7, names.index('HPV::level="__unseen__"')], 1)
        self.assertEqual(raw[7, names.index("HPV::unseen")], 0)
        self.assertEqual(len(names), len(set(names)))

    def test_train_only_preprocessing_and_literal_categories(self):
        train = pd.DataFrame({"AGE": [10., 30., np.nan], "SEX": ["M", "F", None]})
        encoder = TabularPreprocessor(["AGE"], ["SEX"]).fit(train)
        before = deepcopy(encoder.to_dict())
        heldout = pd.DataFrame({"AGE": [10000., np.nan], "SEX": ["new-level", None]})
        result = encoder.transform(heldout)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(encoder.parameters["AGE"]["median"], 20.)
        self.assertEqual(encoder.categories["SEX"], ["F", "M"])
        self.assertEqual(before, encoder.to_dict())
        self.assertNotEqual(result[0].tolist(), result[1].tolist())
        with self.assertRaisesRegex(ValueError, "coding"):
            encoder.transform(pd.DataFrame({"AGE": ["over-sixty"], "SEX": ["M"]}))

    def test_clinical_coding_cannot_assume_stage_redundancy_or_treatment(self):
        inventory = ["AGE", "T", "N", "M", "NSTAGE"]
        self.assertEqual(clinical_columns(clinical_policy(), inventory), (["AGE"], ["T", "N", "M"]))
        for update in ({"status": "unconfirmed"}, {"redundant_columns": {}},
                       {"categorical": ["T", "N", "M", "NSTAGE"], "redundant_columns": {}}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                clinical_columns({**clinical_policy(), **update}, inventory)
        with self.assertRaisesRegex(ValueError, "TX"):
            clinical_columns(clinical_policy(), inventory + ["TX"])

    def test_geometry_absence_is_not_missing_or_node_instance(self):
        geometry = ReferenceGeometry((9, 9, 9), (1., 1., 2.))
        pt = np.zeros(geometry.shape_dhw, dtype=bool)
        pt[3:5, 3:5, 3:5] = True
        zero = np.zeros_like(pt)
        records = []
        for ln in (zero, None):
            _, manifest = build_compartments(pt, ln, np.ones_like(pt), geometry,
                                            patient_id="001", scan_id="scan-A",
                                            contour_source="synthetic", contour_version="fixture-v1",
                                            valid_tissue_source="fabricated full canvas")
            records.append(geometry_features(manifest, patient_id="001", scan_id="scan-A"))
            with self.assertRaisesRegex(ValueError, "identity"):
                geometry_features(manifest, patient_id="1")
        absent, missing = records
        self.assertEqual(absent["ln_volume_mm3"], 0.)
        self.assertEqual(absent["ln_annotation_state"], "absent")
        self.assertTrue(np.isnan(missing["ln_volume_mm3"]))
        self.assertEqual(missing["ln_annotation_state"], "missing")
        self.assertTrue(np.isnan(absent["pt_ln_centroid_distance_mm"]))
        self.assertNotIn("node_count", absent)
        self.assertNotIn("largest_node_volume_mm3", absent)

    def test_cox_selection_matches_reference_and_checkpoint_predictions(self):
        train, val, test = synthetic_frames()
        grid = [10., 20., 40.]
        model, encoder, selection = fit_cox_baseline(
            train, val, numeric=["AGE"], categorical=["T", "N", "M"],
            time_column="OS.TIME", event_column="OS.EVENT", alpha_grid=[.1, 1., 10.],
            validation_times=grid,
        )
        curves = survival_probabilities(model, encoder.transform(val), np.asarray(grid))
        reference = integrated_brier_score(survival_target(train, "OS.TIME", "OS.EVENT"),
                                           survival_target(val, "OS.TIME", "OS.EVENT"), curves, grid)
        self.assertAlmostEqual(reference, min(row["validation_ibs"] for row in selection["candidates"]))
        checkpoint = serializable_model(model, encoder)
        times, baseline = checkpoint["baseline_times"], checkpoint["baseline_cumulative_hazard"]
        indices = np.searchsorted(times, grid, side="right") - 1
        cumulative = np.asarray([baseline[i] if i >= 0 else 0 for i in indices])
        restored = np.exp(-np.exp(encoder.transform(test) @ checkpoint["coefficients"])[:, None] * cumulative)
        np.testing.assert_allclose(restored, survival_probabilities(model, encoder.transform(test), np.asarray(grid)))
        with self.assertRaisesRegex(ValueError, "exceeds training"):
            survival_probabilities(model, encoder.transform(test), np.asarray([10., 101.]))

    def test_tied_penalty_selection_prefers_stronger_regularization(self):
        train, val, _ = synthetic_frames()
        with patch("sksurv.metrics.integrated_brier_score", side_effect=[.2, .1, .1]):
            model, _, selected = fit_cox_baseline(
                train, val, numeric=["AGE"], categorical=["T"], time_column="OS.TIME",
                event_column="OS.EVENT", alpha_grid=[.1, 1., 10.], validation_times=[10., 20.])
        self.assertEqual(model.alpha, 10.)
        self.assertEqual(selected["selected_alpha"], 10.)

    def test_draft_protocol_writes_blocked_receipt_without_fit(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = root / "spec.json"
            spec.write_text(json.dumps({"status": "draft"}))
            with patch("trifusesurv2.baselines.runner.fit_cox_baseline") as fit:
                with self.assertRaisesRegex(ValueError, "not frozen"):
                    main(["--study-spec", str(spec), "--metadata-csv", "unused.csv",
                          "--split-dir", "unused", "--out-dir", str(root / "out")])
                fit.assert_not_called()
            receipt = json.loads((root / "out" / "baseline_receipt.json").read_text())
            self.assertEqual(receipt["status"], "blocked_or_failed")

    def test_shared_evaluator_contract_and_reference_version_gate_fitting(self):
        spec = confirmed_spec()
        bad = deepcopy(spec)
        bad["evaluation"]["ibs_times"] = [10., 10.]
        with self.assertRaisesRegex(ValueError, "IBS grid"):
            validate_protocol(bad, ["C0"])
        with patch("trifusesurv2.evaluation.paired.importlib.metadata.version", return_value="0.99.0"):
            with self.assertRaisesRegex(ValueError, "reference version mismatch"):
                validate_protocol(spec, ["C0"])

    def test_literal_split_ids_and_overlap(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, ids in {"train": "001\n1\n", "val": "NA\n", "test": "A\n"}.items():
                (root / f"{name}_ids.txt").write_text(ids)
            splits, _ = load_splits(root)
            self.assertEqual(splits["train"], ["001", "1"])
            self.assertEqual(splits["val"], ["NA"])
            (root / "test_ids.txt").write_text("001\n")
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_splits(root)

    def test_test_outcomes_do_not_change_selected_fit(self):
        train, val, test = synthetic_frames()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            splitdir = root / "splits"
            splitdir.mkdir()
            for name, frame in (("train", train), ("val", val), ("test", test)):
                (splitdir / f"{name}_ids.txt").write_text("\n".join(frame.index) + "\n")
            _, hashes = load_splits(splitdir)
            all_rows = pd.concat([train, val, test]).rename_axis("Patient_ID").reset_index()
            geometry = ReferenceGeometry((9, 9, 9), (1., 1., 2.))
            inventory = {}
            manifest_paths = []
            for index, patient_id in enumerate(all_rows["Patient_ID"]):
                pt = np.zeros(geometry.shape_dhw, dtype=bool)
                pt[2:4, 2:4, 2:4 + index % 2] = True
                ln = np.zeros_like(pt)
                if index % 3 == 0:
                    ln[6:8, 6:8, 6:8] = True
                elif index % 3 == 1:
                    ln = None
                _, manifest = build_compartments(pt, ln, np.ones_like(pt), geometry,
                                                patient_id=patient_id, scan_id=f"scan-{patient_id}",
                                                contour_source="synthetic", contour_version="fixture-v1",
                                                valid_tissue_source="fabricated full canvas")
                path = root / f"{patient_id}-manifest.json"
                path.write_text(json.dumps(manifest))
                inventory[patient_id] = {"path": str(path), "sha256": sha256_file(path)}
                manifest_paths.append(str(path))
            all_rows["region_manifest_path"] = manifest_paths
            inventory_path = root / "manifest_inventory.json"
            inventory_path.write_text(json.dumps(inventory))
            metadata = root / "metadata.csv"
            spec_path = root / "study_spec.json"
            spec = confirmed_spec()
            spec["baselines"].update({"geometry": {"status": "qualified_for_development",
                                                    "accepted_limitations": ["fabricated geometric masks"],
                                                    "manifest_column": "region_manifest_path",
                                                    "manifest_inventory_sha256": sha256_file(inventory_path)},
                                       "split_files_sha256": hashes})
            results = []
            for attempt in range(2):
                if attempt:
                    heldout = all_rows["Patient_ID"].isin(test.index)
                    all_rows.loc[heldout, "OS.EVENT"] = 1 - all_rows.loc[heldout, "OS.EVENT"]
                    all_rows.loc[heldout, "OS.TIME"] *= 0.8
                all_rows.to_csv(metadata, index=False)
                spec["baselines"]["metadata_sha256"] = sha256_file(metadata)
                if attempt:
                    spec_path.write_text(json.dumps(spec))
                    with patch("trifusesurv2.baselines.runner.fit_cox_baseline") as fit:
                        with self.assertRaisesRegex(ValueError, "outcome inventory"):
                            main(["--study-spec", str(spec_path), "--metadata-csv", str(metadata),
                                  "--split-dir", str(splitdir), "--out-dir", str(root / "mismatched-outcomes"),
                                  "--manifest-inventory", str(inventory_path)])
                        fit.assert_not_called()
                bound_outcomes = all_rows[["Patient_ID", "OS.TIME", "OS.EVENT"]].rename(
                    columns={"OS.TIME": "observed_time", "OS.EVENT": "event"})
                bound_outcomes["endpoint"] = "OS"
                bound_outcomes["split"] = ["train" if pid in train.index else "val" if pid in val.index else "test"
                                           for pid in bound_outcomes["Patient_ID"]]
                spec["evaluation"]["outcome_inventory_sha256"] = outcome_inventory_sha256(bound_outcomes, "OS")
                spec_path.write_text(json.dumps(spec))
                output = root / f"out{attempt}"
                main(["--study-spec", str(spec_path), "--metadata-csv", str(metadata),
                      "--split-dir", str(splitdir), "--out-dir", str(output),
                      "--manifest-inventory", str(inventory_path)])
                results.append(json.loads((output / "C0" / "selected_model.json").read_text()))
                predictions = pd.read_csv(output / "C0" / "test_survival_curves.csv", dtype={"Patient_ID": str})
                self.assertEqual(set(predictions["Patient_ID"]), set(test.index))
                self.assertEqual(set(predictions["seed"]), {"deterministic"})
                geometry_model = json.loads((output / "C1" / "selected_model.json").read_text())
                self.assertIn("ln_volume_mm3::unavailable", geometry_model["preprocessor"]["feature_names"])
                self.assertNotIn("node_count", geometry_model["preprocessor"]["numeric"])
            for key in ("coefficients", "alpha", "preprocessor", "baseline_times", "baseline_cumulative_hazard"):
                self.assertEqual(results[0][key], results[1][key])


if __name__ == "__main__":
    unittest.main()
