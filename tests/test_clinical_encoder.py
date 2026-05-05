"""Unit tests for the semantic clinical encoder."""

from __future__ import annotations

import unittest

import importlib.util

try:
    _HAVE_DEPS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "pandas"))
except Exception:
    _HAVE_DEPS = False

if _HAVE_DEPS:
    import numpy as np
    import pandas as pd
    from trifusesurv2.encoders.clinical import SemanticClinicalTokenEncoder, parse_ordinal_value
    from trifusesurv2.schema import TREATMENT_AWARE_CLINICAL_TOKEN_GROUPS


@unittest.skipUnless(_HAVE_DEPS, "numpy/pandas are not available in this runtime")
class SemanticClinicalTokenEncoderTest(unittest.TestCase):
    def test_stage_roman_suffix_is_parsed(self):
        self.assertEqual(parse_ordinal_value("NSTAGE", "Stage IVA"), 4.0)
        self.assertEqual(parse_ordinal_value("NSTAGE", "IVB"), 4.0)

    def test_grouped_token_matrix_shape(self):
        df = pd.DataFrame(
            [
                {
                    "HPV": 1,
                    "PATHOLOGY": "SCC",
                    "T": "T2",
                    "N": "N1",
                    "M": "M0",
                    "NSTAGE": "III",
                    "AGE": 60,
                    "SEX": "M",
                    "RACE": "W",
                    "KFCF": 90,
                    "SMOKE": 1,
                    "ALCOHOL": 0,
                    "TX": "CRT",
                },
                {
                    "HPV": 0,
                    "PATHOLOGY": "SCC",
                    "T": "T3",
                    "N": "N2",
                    "M": "M0",
                    "NSTAGE": "IV",
                    "AGE": 55,
                    "SEX": "F",
                    "RACE": "B",
                    "KFCF": 80,
                    "SMOKE": 0,
                    "ALCOHOL": 1,
                    "TX": "Surgery",
                },
            ]
        )
        enc = SemanticClinicalTokenEncoder.fit(df)
        mat, pres = enc.encode_frame_token_matrix(df)

        self.assertEqual(mat.shape[0], 2)
        self.assertEqual(mat.shape[1], enc.token_count)
        self.assertEqual(mat.shape[2], enc.max_token_dim)
        self.assertEqual(pres.shape, (2, enc.token_count))
        self.assertTrue(np.all(pres == 1.0))

    def test_treatment_aware_schema_keeps_tx_as_categorical(self):
        df = pd.DataFrame(
            [
                {"TX": 1, "HPV": 1, "PATHOLOGY": "SCC", "T": "T2", "N": "N1", "M": "M0", "NSTAGE": "IVA"},
                {"TX": 2, "HPV": 0, "PATHOLOGY": "SCC", "T": "T3", "N": "N2", "M": "M0", "NSTAGE": "III"},
            ]
        )
        enc = SemanticClinicalTokenEncoder.fit(df, token_groups=TREATMENT_AWARE_CLINICAL_TOKEN_GROUPS)
        treatment_spec = enc.group_specs["treatment"]
        self.assertEqual(len(treatment_spec.numeric), 0)
        self.assertEqual(len(treatment_spec.categorical), 1)

    def test_empty_clinical_fit_raises(self):
        with self.assertRaises(ValueError):
            SemanticClinicalTokenEncoder.fit(pd.DataFrame([{}]))

    def test_group_presence_reflects_actual_missingness(self):
        df = pd.DataFrame(
            [
                {
                    "HPV": 1,
                    "PATHOLOGY": "SCC",
                    "T": "T2",
                    "N": "N1",
                    "M": "M0",
                    "NSTAGE": "III",
                    "AGE": 60,
                    "SEX": "M",
                    "RACE": "W",
                    "KFCF": 90,
                    "SMOKE": 1,
                    "ALCOHOL": 0,
                },
                {
                    "HPV": np.nan,
                    "PATHOLOGY": "",
                    "T": np.nan,
                    "N": np.nan,
                    "M": np.nan,
                    "NSTAGE": np.nan,
                    "AGE": np.nan,
                    "SEX": None,
                    "RACE": None,
                    "KFCF": np.nan,
                    "SMOKE": np.nan,
                    "ALCOHOL": np.nan,
                },
            ]
        )
        enc = SemanticClinicalTokenEncoder.fit(df)
        _, pres = enc.encode_frame_token_matrix(df)
        self.assertTrue(np.all(pres[0] == 1.0))
        self.assertTrue(np.all(pres[1] == 0.0))


if __name__ == "__main__":
    unittest.main()
