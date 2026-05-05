"""Unit tests for the habitat radiomics encoder."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest

try:
    _HAVE_DEPS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "pandas", "sklearn"))
except Exception:
    _HAVE_DEPS = False

if _HAVE_DEPS:
    import numpy as np
    import pandas as pd

    from trifusesurv2.encoders.radiomics import HabitatRadiomicsTokenEncoder


@unittest.skipUnless(_HAVE_DEPS, "numpy/pandas/sklearn are not available in this runtime")
class HabitatRadiomicsTokenEncoderTest(unittest.TestCase):
    def _write_csv(self, rows):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        try:
            pd.DataFrame(rows).to_csv(tmp.name, index=False)
            return tmp.name
        finally:
            tmp.close()

    def test_fit_from_wide_csv_requires_presence_columns_by_default(self):
        path = self._write_csv(
            [
                {"patient_id": "A1", "PT_intratumor__f1": 1.0, "PT_peritumor_10mm__f1": 2.0},
                {"patient_id": "A2", "PT_intratumor__f1": 1.5, "PT_peritumor_10mm__f1": 2.5},
            ]
        )
        try:
            with self.assertRaises(ValueError):
                HabitatRadiomicsTokenEncoder.fit_from_wide_csv(
                    radiomics_csv=path,
                    train_ids=["A1", "A2"],
                    all_ids=["A1", "A2"],
                    total_pcs_per_group=1,
                )
        finally:
            os.unlink(path)

    def test_fit_from_wide_csv_uses_explicit_presence_columns(self):
        path = self._write_csv(
            [
                {
                    "patient_id": "A1",
                    "PT_intratumor__f1": 1.0,
                    "PT_peritumor_10mm__f1": 2.0,
                    "LN_intratumor__f1": 3.0,
                    "LN_peritumor_10mm__f1": 4.0,
                    "present__PT_intratumor": 1,
                    "present__PT_peritumor_10mm": 1,
                    "present__LN_intratumor": 0,
                    "present__LN_peritumor_10mm": 0,
                },
                {
                    "patient_id": "A2",
                    "PT_intratumor__f1": 1.5,
                    "PT_peritumor_10mm__f1": 2.5,
                    "LN_intratumor__f1": 3.5,
                    "LN_peritumor_10mm__f1": 4.5,
                    "present__PT_intratumor": 1,
                    "present__PT_peritumor_10mm": 1,
                    "present__LN_intratumor": 1,
                    "present__LN_peritumor_10mm": 1,
                },
                {
                    "patient_id": "A3",
                    "PT_intratumor__f1": 1.8,
                    "PT_peritumor_10mm__f1": 2.8,
                    "LN_intratumor__f1": 3.8,
                    "LN_peritumor_10mm__f1": 4.8,
                    "present__PT_intratumor": 1,
                    "present__PT_peritumor_10mm": 1,
                    "present__LN_intratumor": 1,
                    "present__LN_peritumor_10mm": 1,
                },
            ]
        )
        try:
            enc = HabitatRadiomicsTokenEncoder.fit_from_wide_csv(
                radiomics_csv=path,
                train_ids=["A1", "A2", "A3"],
                all_ids=["A1", "A2", "A3"],
                total_pcs_per_group=1,
            )
            mat, pres = enc.encode_patient_token_matrix("A1")
            self.assertEqual(mat.shape, (4, 1))
            self.assertTrue(np.array_equal(pres, np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32)))
            for spec in enc.pca_specs.values():
                self.assertEqual(spec.scale.shape, (spec.input_dim,))
                self.assertTrue(np.all(np.isfinite(spec.scale)))
                self.assertTrue(np.all(spec.scale > 0.0))
                self.assertEqual(spec.pca_mean.shape, (spec.input_dim,))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
