"""Synthetic source-whitelist checks; no patient data or neural fitting."""

from collections import OrderedDict
import pickle
import unittest

import numpy as np
import pandas as pd

from trifusesurv2.encoders.clinical import SemanticClinicalTokenEncoder
from trifusesurv2.schema import CLINICAL_TOKEN_GROUPS, TREATMENT_AWARE_CLINICAL_TOKEN_GROUPS
from trifusesurv2.utils.clinical import DEFAULT_CLINICAL_COLS


def clinical_frame():
    return pd.DataFrame({
        "AGE": [40, 55, 70], "SEX": ["F", "M", "F"], "RACE": ["A", "B", "A"],
        "KFCF": [80, 90, 100], "SMOKE": [0, 1, 2], "ALCOHOL": [0, 1, 0],
        "HPV": [0, 1, 0], "PATHOLOGY": ["SCC", "OTHER", "SCC"],
        "T": ["T2", "T2B", "T4"], "N": ["N0", "N1", "N2"], "M": ["M0", "M0", "M1"],
        "NSTAGE": ["II", "III", "IVA"], "NSTAGE_RAW": ["II", "III", "IVA"],
        "TX": ["CHEMO", "SURGERY", "RADIO"],
    })


class SemanticClinicalColumnsTest(unittest.TestCase):
    def assert_encoded_equal(self, encoder, first, second):
        for left, right in zip(encoder.encode_frame_token_matrix(first), encoder.encode_frame_token_matrix(second)):
            np.testing.assert_array_equal(left, right)

    def test_excluded_stage_and_treatment_cannot_change_fit_or_transform(self):
        frame = clinical_frame()
        columns = [col for col in DEFAULT_CLINICAL_COLS if col != "NSTAGE"]
        changed = frame.assign(NSTAGE=["IV", None, "I"], NSTAGE_RAW=["SECRET", "IVA", "V"],
                               TX=["NEW", None, "OTHER"])
        encoder = SemanticClinicalTokenEncoder.fit(frame, columns=columns)
        refitted = SemanticClinicalTokenEncoder.fit(changed, columns=columns)
        self.assertEqual(encoder.group_specs, refitted.group_specs)
        self.assertEqual(encoder.group_names, tuple(CLINICAL_TOKEN_GROUPS))
        self.assertFalse({"NSTAGE", "NSTAGE_RAW", "TX"} & set(encoder.available_columns()))
        self.assertTrue({"T", "T_RAW", "N", "N_RAW", "M", "M_RAW"} <= set(encoder.available_columns()))
        self.assert_encoded_equal(encoder, frame, changed)
        self.assertEqual(encoder.column_policy, {"mode": "explicit_source_columns", "columns": columns})

    def test_excluded_values_cannot_make_a_missing_group_present(self):
        frame = clinical_frame()
        encoder = SemanticClinicalTokenEncoder.fit(frame, columns=["AGE", "T", "N", "M"])
        missing = frame.assign(T=None, N=None, M=None)
        changed = missing.assign(NSTAGE="IV", NSTAGE_RAW="IV", TX="NEW")
        matrix, presence = encoder.encode_frame_token_matrix(changed)
        self.assertTrue(np.isfinite(matrix).all())
        np.testing.assert_array_equal(presence[:, encoder.group_names.index("burden")], 0)
        np.testing.assert_array_equal(presence[:, encoder.group_names.index("biology")], 0)
        self.assert_encoded_equal(encoder, missing, changed)

    def test_included_columns_and_stage_subclasses_still_change_their_tokens(self):
        frame = clinical_frame()
        encoder = SemanticClinicalTokenEncoder.fit(frame, columns=["AGE", "HPV", "T"])
        baseline, _ = encoder.encode_frame_token_matrix(frame)
        for column, value, group in (("AGE", 60, "host"), ("HPV", 1, "biology"), ("T", "T2B", "burden")):
            with self.subTest(column=column):
                changed = frame.copy()
                changed.loc[0, column] = value
                observed, _ = encoder.encode_frame_token_matrix(changed)
                group_idx = encoder.group_names.index(group)
                self.assertFalse(np.array_equal(baseline[0, group_idx], observed[0, group_idx]))
                for idx in range(encoder.token_count):
                    if idx != group_idx:
                        np.testing.assert_array_equal(baseline[0, idx], observed[0, idx])

    def test_explicit_raw_column_never_reads_an_excluded_ordinal_source(self):
        frame = clinical_frame().assign(T_RAW=["T1", "T2", "T3"])
        encoder = SemanticClinicalTokenEncoder.fit(frame, columns=["T_RAW"])
        self.assertEqual(encoder.available_columns(), ["T_RAW"])
        self.assert_encoded_equal(encoder, frame, frame.assign(T=["T9", None, "TIS"]))
        categorical = encoder.group_specs["burden"].categorical[0]
        self.assertEqual(set(categorical.mapping), {"T1", "T2", "T3"})

    def test_checkpoint_round_trip_preserves_whitelist_and_legacy_state(self):
        frame = clinical_frame()
        encoder = SemanticClinicalTokenEncoder.fit(frame, columns=["AGE", "T"])
        restored = pickle.loads(pickle.dumps(encoder))
        self.assertEqual(restored.allowed_columns, ("AGE", "T"))
        self.assertEqual(restored.column_policy, encoder.column_policy)
        self.assert_encoded_equal(restored, frame, frame.assign(NSTAGE="V", NSTAGE_RAW="V", TX="NEW"))
        legacy = SemanticClinicalTokenEncoder.fit(frame)
        self.assertEqual(set(vars(legacy)), {"group_specs", "group_names", "token_dims", "max_token_dim", "output_dim"})
        legacy_restored = pickle.loads(pickle.dumps(legacy))
        self.assertIsNone(legacy_restored.allowed_columns)
        self.assertEqual(legacy_restored.column_policy, {"mode": "all_token_group_sources"})
        for left, right in zip(legacy.encode_frame_token_matrix(frame), legacy_restored.encode_frame_token_matrix(frame)):
            np.testing.assert_array_equal(left, right)

    def test_default_groups_and_encoding_remain_compatible(self):
        frame = clinical_frame()
        legacy = SemanticClinicalTokenEncoder.fit(frame)
        explicit = SemanticClinicalTokenEncoder.fit(frame, columns=DEFAULT_CLINICAL_COLS)
        self.assertEqual(legacy.group_specs, explicit.group_specs)
        self.assertTrue({"NSTAGE", "NSTAGE_RAW"} <= set(legacy.available_columns()))
        self.assertNotIn("TX", legacy.available_columns())
        for left, right in zip(legacy.encode_frame_token_matrix(frame), explicit.encode_frame_token_matrix(frame)):
            np.testing.assert_array_equal(left, right)

    def test_custom_treatment_group_retains_identity_but_obeys_whitelist(self):
        frame = clinical_frame()
        restricted = SemanticClinicalTokenEncoder.fit(frame, TREATMENT_AWARE_CLINICAL_TOKEN_GROUPS, columns=["AGE"])
        self.assertEqual(restricted.group_names, tuple(TREATMENT_AWARE_CLINICAL_TOKEN_GROUPS))
        self.assertEqual(restricted.token_dims["treatment"], 0)
        self.assert_encoded_equal(restricted, frame, frame.assign(TX="DIFFERENT"))
        legacy = SemanticClinicalTokenEncoder.fit(frame, TREATMENT_AWARE_CLINICAL_TOKEN_GROUPS)
        self.assertIn("TX", legacy.available_columns())

    def test_invalid_or_empty_explicit_whitelists_fail_instead_of_falling_back(self):
        for columns in ("AGE", ["AGE", "AGE"], [" AGE"], [""], [None], []):
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                SemanticClinicalTokenEncoder.fit(clinical_frame(), columns=columns)
        # Existing direct construction of an empty encoder remains supported.
        self.assertEqual(SemanticClinicalTokenEncoder(OrderedDict()).output_dim, 0)


if __name__ == "__main__":
    unittest.main()
