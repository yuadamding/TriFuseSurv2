"""CSV identifiers must survive numeric-looking values and pandas NA aliases."""
import io
import unittest

import pandas as pd

from trifusesurv2.preprocessing.identifiers import read_identifier_csv, resolve_identifier_column, require_identifier


class IdentifierTests(unittest.TestCase):
    def test_literal_identifiers_and_explicit_missing_values(self):
        source = io.StringIO(
            'Patient_ID,case_id,scan_id,radiomics_scan_id,ct_series_uid,rtstruct_referenced_series_uid,contour_version,region_id__pt_intra,support_sha256__pt_intra,age\n'
            '001,0002,0003,0003,0004,0004,0005,0006,0007,50\n'
            'NA,nan,NULL,NA,NA,nan,NA,NULL,nan,NA\n'
            ',,,,,,,,,\n')
        frame = read_identifier_csv(source, identifier_columns=('patient_id',))
        self.assertEqual(resolve_identifier_column(frame.columns, 'patient_id'), 'Patient_ID')
        self.assertEqual(frame.loc[0, 'Patient_ID'], '001')
        self.assertEqual(frame.loc[0, 'scan_id'], '0003')
        self.assertEqual(frame.loc[0, 'ct_series_uid'], '0004')
        self.assertEqual(frame.loc[0, 'contour_version'], '0005')
        self.assertEqual(frame.loc[0, 'support_sha256__pt_intra'], '0007')
        self.assertEqual(frame.loc[1, 'Patient_ID'], 'NA')
        self.assertEqual(frame.loc[1, 'scan_id'], 'NULL')
        self.assertTrue(pd.isna(frame.loc[2, 'Patient_ID']))
        self.assertTrue(pd.isna(frame.loc[1, 'age']))
        with self.assertRaisesRegex(ValueError, 'nonmissing literal'):
            require_identifier(frame.loc[2, 'Patient_ID'], name='Patient_ID')
        output = io.StringIO(); frame.to_csv(output, index=False); output.seek(0)
        pd.testing.assert_frame_equal(read_identifier_csv(output), frame)

    def test_alias_resolution_is_unambiguous_and_preserves_columns(self):
        self.assertEqual(resolve_identifier_column(['patient_id', 'scan_id'], 'Patient_ID'), 'patient_id')
        self.assertIsNone(resolve_identifier_column(['patient_id'], 'scan_id', required=False))
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            resolve_identifier_column(['patient_id', 'PatientID'], 'patient_id')
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            resolve_identifier_column(['patient_id', 'PatientID'], 'PATIENT-ID')
        with self.assertRaisesRegex(ValueError, 'no matching'):
            resolve_identifier_column(['Patient_ID'], 'scan_id')


if __name__ == '__main__':
    unittest.main()
