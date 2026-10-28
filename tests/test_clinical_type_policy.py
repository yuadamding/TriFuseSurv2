"""Prospective categorical policy on invented rows; no clinical data fitting."""
import contextlib
import io
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from trifusesurv2.encoders.clinical import (
    APPROVED_CLINICAL_TYPES, APPROVED_TYPE_POLICY, SemanticClinicalTokenEncoder,
)
from trifusesurv2.multimodal_survival import train


def invented_frame():
    return pd.DataFrame({
        'AGE': [40, 50, 60, 70], 'HPV': ['0', '1', 'nd', 'ND'],
        'KFCF': ['080', '80', '90', 'ND'], 'SMOKE': ['0', '1', 'nd', 'ND'],
        'ALCOHOL': ['0', '1', '2', 'unknown'], 'T': ['T2', 'T4A', 'T4B', '2'],
        'N': ['N2A', 'N2B', 'N2C', 'N0'], 'M': ['M0', 'M1', 'MX', '0'],
        'PATHOLOGY': ['SCC', 'OTHER', 'NA', 'SCC'], 'SEX': ['F', 'M', 'F', 'M'],
        'RACE': ['A', 'B', 'A', 'NA'], 'NSTAGE': ['II', 'III', 'IVA', 'IVB'],
        'NSTAGE_RAW': ['IV'] * 4, 'TX': ['surgery', 'radiation', 'chemo', 'none'],
        'T_RAW': ['wrong'] * 4, 'N_RAW': ['wrong'] * 4, 'M_RAW': ['wrong'] * 4,
    })


def fitted(frame=None, columns=None):
    return SemanticClinicalTokenEncoder.fit(invented_frame() if frame is None else frame,
        columns=list(APPROVED_CLINICAL_TYPES) if columns is None else columns, type_policy=APPROVED_TYPE_POLICY)


class ProspectiveClinicalTypesTests(unittest.TestCase):
    def test_exact_declared_types_no_ordinal_or_derived_stage_channels(self):
        encoder = fitted()
        numeric = [feature for group in encoder.group_specs.values() for feature in group.numeric]
        categorical = [feature for group in encoder.group_specs.values() for feature in group.categorical]
        self.assertEqual([feature.name for feature in numeric], ['AGE'])
        self.assertFalse(numeric[0].ordinal)
        self.assertEqual({feature.name for feature in categorical}, set(APPROVED_CLINICAL_TYPES) - {'AGE'})
        self.assertEqual(len(encoder.available_columns()), len(APPROVED_CLINICAL_TYPES))
        hpv = encoder.group_specs['biology'].categorical[0]
        self.assertEqual(set(hpv.mapping), {'0', '1', 'nd', 'ND'})
        matrix, presence = encoder.encode_frame_token_matrix(invented_frame())
        self.assertTrue(np.isfinite(matrix).all())
        self.assertTrue(presence.all())
        self.assertFalse(np.array_equal(matrix[2, 0], matrix[3, 0]))
        self.assertEqual(encoder.type_policy['columns'], APPROVED_CLINICAL_TYPES)
        json.dumps(encoder.type_policy, allow_nan=False)

    def test_missing_and_unseen_have_distinct_slots_and_presence(self):
        encoder = fitted(pd.DataFrame({'HPV': ['nd', 'ND', 'NA']}), ['HPV'])
        spec = encoder.group_specs['biology'].categorical[0]
        for value in (None, np.nan, pd.NA, '', '   '):
            with self.subTest(missing=value):
                matrix, presence = encoder.encode_row_token_matrix(pd.Series({'HPV': value}))
                self.assertEqual(matrix[0, spec.missing_index], 1)
                self.assertEqual(matrix[0, spec.unk_index], 0)
                self.assertEqual(presence[0], 0)
        for value in ('unseen', ' ND ', 'NaN'):
            with self.subTest(unseen=value):
                matrix, presence = encoder.encode_row_token_matrix(pd.Series({'HPV': value}))
                self.assertEqual(matrix[0, spec.unk_index], 1)
                self.assertEqual(matrix[0, spec.missing_index], 0)
                self.assertEqual(presence[0], 1)
        matrix, presence = encoder.encode_row_token_matrix(pd.Series({'HPV': 'NA'}))
        self.assertEqual(matrix[0, spec.mapping['NA']], 1)
        self.assertEqual(presence[0], 1)

    def test_entirely_missing_or_absent_training_column_retains_missing_unseen_policy(self):
        for frame in (pd.DataFrame({'HPV': [None, '', np.nan]}), pd.DataFrame({'AGE': [40, 50, 60]})):
            with self.subTest(columns=list(frame)):
                encoder = fitted(frame, ['HPV'])
                feature = encoder.group_specs['biology'].categorical[0]
                self.assertEqual(feature.mapping, {})
                self.assertEqual(encoder.output_dim, 2)
                before = pickle.dumps(encoder)
                values, presence = encoder.encode_frame_token_matrix(pd.DataFrame({'HPV': [None, 'new-validation-code']}))
                np.testing.assert_array_equal(values[:, 0], [[1, 0], [0, 1]])
                np.testing.assert_array_equal(presence[:, 0], [0, 1])
                self.assertEqual(pickle.dumps(encoder), before)

    def test_validation_values_cannot_expand_vocabulary_or_change_training_statistics(self):
        training = pd.DataFrame({'AGE': [40, 50, 60], 'HPV': ['nd', 'ND', '0']})
        encoder = fitted(training, ['AGE', 'HPV'])
        before = pickle.dumps(encoder)
        validation = pd.DataFrame({'AGE': [999], 'HPV': ['validation-only']})
        encoder.encode_frame_token_matrix(validation)
        self.assertEqual(pickle.dumps(encoder), before)
        self.assertNotIn('validation-only', encoder.group_specs['biology'].categorical[0].mapping)
        self.assertEqual(encoder.group_specs['host'].numeric[0].mean, 50)
        restored = pickle.loads(before)
        self.assertEqual(restored.type_policy, encoder.type_policy)
        for expected, actual in zip(encoder.encode_frame_token_matrix(validation), restored.encode_frame_token_matrix(validation)):
            np.testing.assert_array_equal(expected, actual)

    def test_fractional_age_does_not_promote_integer_categories_during_frame_encoding(self):
        frame = pd.DataFrame({'AGE': [40.5, 50.5], 'HPV': [0, 1]})
        encoder = fitted(frame, ['AGE', 'HPV'])
        hpv = encoder.group_specs['biology'].categorical[0]
        values, _ = encoder.encode_frame_token_matrix(frame)
        self.assertEqual(set(hpv.mapping), {'0', '1'})
        self.assertEqual(values[0, 0, hpv.mapping['0']], 1)
        self.assertEqual(values[1, 0, hpv.mapping['1']], 1)
        self.assertEqual(values[:, 0, hpv.unk_index].sum(), 0)

    def test_excluded_columns_and_derived_aliases_never_affect_fit_tokens_or_presence(self):
        original = invented_frame()
        excluded = ['NSTAGE', 'NSTAGE_RAW', 'TX', 'T_RAW', 'N_RAW', 'M_RAW']
        changed = original.assign(**{name: ['new', None, 'different', ''] for name in excluded})
        encoder, other = fitted(original), fitted(changed)
        self.assertEqual(encoder.group_specs, other.group_specs)
        for expected, actual in zip(encoder.encode_frame_token_matrix(original), encoder.encode_frame_token_matrix(changed)):
            np.testing.assert_array_equal(expected, actual)
        for field in excluded:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'excluded'):
                fitted(original, ['AGE', field])
        restricted = fitted(original, ['HPV'])
        _, presence = restricted.encode_row_token_matrix(pd.Series({'HPV': None, 'NSTAGE': 'IV', 'TX': 'chemo'}))
        self.assertFalse(presence.any())

    def test_numeric_age_scaling_missing_and_invalid_values_are_declared(self):
        encoder = fitted(pd.DataFrame({'AGE': [40, 50, 60, np.nan]}), ['AGE'])
        spec = encoder.group_specs['host'].numeric[0]
        self.assertEqual((spec.mean, spec.std), (50, 10))
        values, presence = encoder.encode_frame_token_matrix(pd.DataFrame({'AGE': [None, 50, 60]}))
        np.testing.assert_array_equal(values[:, 2], [[0, 1], [0, 0], [1, 0]])
        np.testing.assert_array_equal(presence[:, 2], [0, 1, 1])
        for invalid in ('unknown', 'inf', float('inf')):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, 'finite number'):
                    fitted(pd.DataFrame({'AGE': [40, invalid]}), ['AGE'])
                with self.assertRaisesRegex(ValueError, 'finite number'):
                    encoder.encode_row_token_matrix(pd.Series({'AGE': invalid}))
        missing = fitted(pd.DataFrame({'AGE': [None, None]}), ['AGE'])
        self.assertEqual(missing.group_specs['host'].numeric[0].mean, 0)
        self.assertEqual(missing.group_specs['host'].numeric[0].std, 1)

    def test_legacy_defaults_and_old_pickle_state_are_unchanged(self):
        frame = invented_frame()
        legacy = SemanticClinicalTokenEncoder.fit(frame, columns=['AGE', 'HPV', 'T'])
        self.assertNotIn('_type_policy', vars(legacy))
        self.assertEqual(legacy.type_policy, {'name': 'legacy'})
        self.assertIn('T_RAW', legacy.available_columns())
        self.assertEqual(legacy.group_specs['biology'].categorical, ())
        _, presence = legacy.encode_frame_token_matrix(frame)
        np.testing.assert_array_equal(presence[2:, 0], [0, 0])
        restored = pickle.loads(pickle.dumps(legacy))
        self.assertNotIn('_type_policy', vars(restored))
        for expected, actual in zip(legacy.encode_frame_token_matrix(frame), restored.encode_frame_token_matrix(frame)):
            np.testing.assert_array_equal(expected, actual)
        for policy, columns in (('unknown', ['AGE']), (APPROVED_TYPE_POLICY, None), (APPROVED_TYPE_POLICY, [])):
            with self.subTest(policy=policy, columns=columns), self.assertRaises(ValueError):
                SemanticClinicalTokenEncoder.fit(frame, columns=columns, type_policy=policy)


class ProspectiveClinicalIngestionTests(unittest.TestCase):
    def test_csv_literal_categories_survive_full_and_development_readers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'clinical.csv'
            path.write_text('Patient_ID,AGE,HPV,KFCF,OS.TIME,OS.EVENT\n'
                            '001,40,nd,01,100,1\n002,50,ND,1,200,0\nNA,60,NA,1.0,300,1\n004,,, ,400,0\n')
            columns = ['AGE', 'HPV', 'KFCF']
            prospective = train.read_clinical_metadata(path, 'Patient_ID', columns, APPROVED_TYPE_POLICY)
            legacy = train.read_clinical_metadata(path, 'Patient_ID', columns)
            self.assertEqual(prospective.HPV.tolist(), ['nd', 'ND', 'NA', ''])
            self.assertEqual(prospective.KFCF.tolist(), ['01', '1', '1.0', ' '])
            self.assertTrue(pd.isna(legacy.HPV.iloc[2]))
            self.assertEqual(prospective.Patient_ID.tolist(), ['001', '002', 'NA', '004'])
            self.assertEqual(prospective['OS.EVENT'].tolist(), legacy['OS.EVENT'].tolist())
            self.assertEqual(prospective['OS.TIME'].tolist(), [100, 200, 300, 400])
            split = {'train': ['001', '002'], 'val': ['NA'], 'test': ['004']}
            development = train.read_development_metadata(str(path), 'Patient_ID', split)
            self.assertEqual(development[columns].to_dict('list'), prospective.iloc[:3][columns].to_dict('list'))
            encoder = fitted(prospective.iloc[:3], columns)
            self.assertEqual(set(encoder.group_specs['biology'].categorical[0].mapping), {'nd', 'ND', 'NA'})

    def test_main_passes_literal_clinical_cells_and_numeric_endpoints_to_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / 'metadata.csv'
            path.write_text('Patient_ID,AGE,HPV,OS.TIME,OS.EVENT\n001,40,NA,100,1\n002,50,nd,200,0\n003,60,ND,300,1\n')
            argv = ['train', '--meta_csv', str(path), '--device', 'cpu', '--out_dir', str(root / 'run'),
                    '--clinical_cols', 'AGE', 'HPV', '--clinical_type_policy', APPROVED_TYPE_POLICY, '--cv_folds', '1']
            result = {'export_risks': {}, 'test_metrics_export': {'c_index': .5}, 'num_time_bins': 3,
                      'train_max_time_days': 300, 'export_suffix': 'last'}
            split = {'train': ['001'], 'val': ['002'], 'test': ['003']}
            with patch('sys.argv', argv), patch.object(train, 'load_precomputed_splits', return_value={0: split}), \
                 patch.object(train, 'resolve_img_size_against_data', return_value=((32, 32, 32), (32, 32, 32))), \
                 patch.object(train, 'run_one_fold', return_value=result) as run, contextlib.redirect_stdout(io.StringIO()):
                train.main()
            passed = run.call_args.kwargs['meta']
            self.assertEqual(passed.HPV.tolist(), ['NA', 'nd', 'ND'])
            self.assertEqual(passed['OS.EVENT'].tolist(), [1, 0, 1])
            self.assertEqual(run.call_args.kwargs['args'].clinical_type_policy, APPROVED_TYPE_POLICY)

    def test_cli_policy_is_opt_in_and_rejects_incompatible_legacy_model(self):
        base = ['train', '--meta_csv', 'unused.csv']
        with patch('sys.argv', base), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(train.parse_args().clinical_type_policy, 'legacy')
        with patch('sys.argv', [*base, '--clinical_type_policy', APPROVED_TYPE_POLICY]), self.assertRaisesRegex(ValueError, 'excluded'):
            train.parse_args()
        with patch('sys.argv', [*base, '--clinical_type_policy', APPROVED_TYPE_POLICY,
                               '--clinical_cols', 'AGE', 'HPV', '--model_version', 'v1']), self.assertRaisesRegex(ValueError, 'v2'):
            train.parse_args()


if __name__ == '__main__':
    unittest.main()
