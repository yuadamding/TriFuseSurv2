"""Regression checks for patient identity, ROI reuse, and CT input contracts."""

from collections import OrderedDict
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import SimpleITK as sitk

from trifusesurv2.encoders.radiomics import HabitatRadiomicsTokenEncoder
from trifusesurv2.utils.data import PreprocessedContourAwareDataset
from trifusesurv2.utils.radiomics import RadiomicsEncoder


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocess_roi_inputs.py"
_SPEC = importlib.util.spec_from_file_location("roi_preprocess_data_contracts", _SCRIPT)
roi = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(roi)


class DataContractsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="trifusesurv2-data-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def source(self, pid="001", value=0.25, extension="nii.gz"):
        case = self.root / "source" / pid
        case.mkdir(parents=True, exist_ok=True)
        ct = np.full((8, 8, 8), value, dtype=np.float32)
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        mask[3:5, 3:5, 3:5] = 1
        paths = {}
        for key, array in (("ct", ct), ("pt", mask), ("ln", mask)):
            path = case / f"{key}.{extension}"
            sitk.WriteImage(sitk.GetImageFromArray(array), str(path))
            paths[key] = str(path)
        return {"patient_id": pid, "time": 10.0, "event": 1.0, **paths}

    def job(self, row, **changes):
        output = self.root / "output"
        job = dict(
            row=row, index=0, id_col="patient_id", meta_dir=str(self.root / "source"),
            out_root=str(output), out_csv_dir=str(output), output_size_dhw=(4, 4, 4),
            margin_dhw=(1, 1, 1), overwrite=False, expand_to_aspect=True,
            fail_empty_roi=False, ct_col="ct", mask_pt_col="pt", mask_ln_col="ln",
        )
        return {**job, **changes}

    def dataset(self, row, mode):
        return PreprocessedContourAwareDataset(
            pd.DataFrame([row]), id_col="patient_id", time_col="time", event_col="event",
            ct_col="ct", mask_pt_col="pt", mask_ln_col="ln", clinical_encoder=None,
            radiomics_encoder=None, use_radiomics=False, expected_dhw=(8, 8, 8),
            mode=mode, spatial_augment=False,
        )

    def test_batch_collision_fails_before_any_outputs(self):
        metadata = self.root / "input.csv"
        pd.DataFrame({"patient_id": ["A B", "A_B"]}).to_csv(metadata, index=False)
        output = self.root / "collision-output"
        with self.assertRaisesRegex(ValueError, "same output directory"):
            roi.main(["--meta_csv", str(metadata), "--out_root", str(output)])
        self.assertFalse(output.exists())

    def test_invalid_or_duplicate_identifiers_are_rejected(self):
        for ids in (["."], [".."], [""], [np.nan], ["same", "same"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                roi._validate_output_ids(ids)

    def test_unchanged_crops_reuse_verified_files(self):
        job = self.job(self.source())
        first = roi._process_one(job)
        output = Path(job["out_root"]) / first["ct_out_path"]
        before = output.stat().st_mtime_ns
        second = roi._process_one(job)
        self.assertEqual(first, second)
        self.assertEqual(before, output.stat().st_mtime_ns)
        provenance = json.loads((output.parent / "roi_crop_provenance.json").read_text())
        self.assertEqual(provenance["inputs"]["patient_id"], "001")
        self.assertEqual(set(provenance["inputs"]["sources"]), {"ct", "pt", "ln"})

    def test_changed_settings_fail_until_explicit_overwrite(self):
        job = self.job(self.source())
        first = roi._process_one(job)
        for changes in ({"output_size_dhw": (6, 6, 6)}, {"margin_dhw": (2, 2, 2)}):
            with self.subTest(changes=changes), self.assertRaisesRegex(RuntimeError, "--overwrite"):
                roi._process_one({**job, **changes})
        output = Path(job["out_root"]) / first["ct_out_path"]
        self.assertEqual(sitk.ReadImage(str(output)).GetSize(), (4, 4, 4))
        changed = roi._process_one({**job, "output_size_dhw": (6, 6, 6), "overwrite": True})
        self.assertEqual(json.loads(changed["roi_crop_output_size_dhw"]), [6, 6, 6])
        self.assertEqual(sitk.ReadImage(str(output)).GetSize(), (6, 6, 6))

    def test_changed_source_and_output_bytes_prevent_reuse(self):
        row = self.source()
        job = self.job(row)
        result = roi._process_one(job)
        self.source(value=0.75)
        with self.assertRaisesRegex(RuntimeError, "source images or crop settings"):
            roi._process_one(job)
        roi._process_one({**job, "overwrite": True})
        output = Path(job["out_root"]) / result["ct_out_path"]
        self.assertAlmostEqual(float(sitk.GetArrayFromImage(sitk.ReadImage(str(output))).mean()), 0.75)
        # Preserve a readable NIfTI header while altering the serialized output.
        with output.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(RuntimeError, "recorded hashes or geometry"):
            roi._process_one(job)

    def test_missing_provenance_and_partial_outputs_prevent_reuse(self):
        job = self.job(self.source())
        result = roi._process_one(job)
        case = (Path(job["out_root"]) / result["ct_out_path"]).parent
        (case / "roi_crop_provenance.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "--overwrite"):
            roi._process_one(job)
        roi._process_one({**job, "overwrite": True})
        (case / "mask_nodal_roi.nii.gz").unlink()
        with self.assertRaisesRegex(RuntimeError, "--overwrite"):
            roi._process_one(job)

    def test_raw_hu_rejected_in_train_eval_and_preprocessing(self):
        row = self.source(value=50.0)
        for mode in ("train", "eval"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "Raw HU input is unsupported"):
                self.dataset(row, mode)[0]
        with self.assertRaisesRegex(ValueError, "Raw HU input is unsupported"):
            roi._process_one(self.job(row))
        self.assertFalse((self.root / "output").exists())

    def test_nonfinite_ct_rejected_and_normalized_ct_accepted(self):
        row = self.source()
        self.assertTrue(np.allclose(self.dataset(row, "eval")[0][0].numpy(), 0.25))
        augmented = self.dataset(row, "train")[0][0].numpy()
        self.assertTrue(np.isfinite(augmented).all())
        self.assertTrue(((augmented >= 0.0) & (augmented <= 1.0)).all())
        for value in (np.nan, np.inf):
            # ITK's NIfTI reader replaces nonfinite voxels with zero; MHA retains them.
            row = self.source(value=value, extension="mha")
            for mode in ("train", "eval"):
                with self.subTest(value=value, mode=mode), self.assertRaisesRegex(ValueError, "finite intensities"):
                    self.dataset(row, mode)[0]

    def test_v2_radiomics_preserves_numeric_string_identifiers(self):
        ids = ["001", "1", "102"]
        for id_col in ("case_id", "patient_id"):
            with self.subTest(id_col=id_col):
                csv = self.root / f"radiomics-{id_col}.csv"
                pd.DataFrame({id_col: ids, "PT_intratumor__feature": [1., 2., 3.],
                              "present__PT_intratumor": [1, 1, 1]}).to_csv(csv, index=False)
                encoder = HabitatRadiomicsTokenEncoder.fit_from_wide_csv(
                    radiomics_csv=csv, train_ids=ids, all_ids=ids, total_pcs_per_group=1,
                    group_prefixes=OrderedDict([("pt_intra", "PT_intratumor__")]),
                )
                tokens, presence = encoder.encode_patient_token_matrix("001")
                other, other_presence = encoder.encode_patient_token_matrix("1")
                np.testing.assert_array_equal(presence, [1.0])
                np.testing.assert_array_equal(other_presence, [1.0])
                self.assertFalse(np.allclose(tokens, other))

    def test_legacy_radiomics_preserves_numeric_string_identifiers(self):
        ids = ["001", "1", "102"]
        csv = self.root / "radiomics.csv"
        pd.DataFrame({"case_id": ids, "PT_intratumor__feature": [1., 2., 3.],
                      "present__PT_intratumor": [1, 1, 1]}).to_csv(csv, index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            encoder = RadiomicsEncoder.fit_from_wide_csv(ids, ids, csv, total_pcs=4, seed=1)
        first, second = encoder.encode_patient("001"), encoder.encode_patient("1")
        self.assertEqual(float(first[encoder.total_pc_dim]), 1.0)
        self.assertEqual(float(second[encoder.total_pc_dim]), 1.0)
        self.assertFalse(np.allclose(first, second))


if __name__ == "__main__":
    unittest.main()
