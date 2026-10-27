"""Metadata and synthetic-input checks; no cohort processing or actual extraction."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import SimpleITK as sitk

from trifusesurv2.preprocessing.identifiers import read_identifier_csv

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_development_inputs', ROOT / 'scripts' / 'prepare_development_inputs.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class DevelopmentInputTests(unittest.TestCase):
    def row(self, pid='001', *, pt=100., ln=50., z=2.):
        return dict(Patient_ID=pid, ct_series_uid='0007', rtstruct_referenced_series_uid='0007', status='ok',
                    ct_out_path=f'{pid}/ct.nii.gz', mask_primary_out_path=f'{pid}/pt.nii.gz', mask_nodal_out_path=f'{pid}/ln.nii.gz',
                    mask_primary_voxels=10, mask_nodal_voxels=5, mask_primary_mm3=pt, mask_nodal_mm3=ln,
                    mask_primary_qc='PT:aligned', mask_nodal_qc='MetaLymphnode:aligned', ct_spacing_xyz=f'1|1|{z}',
                    **{'OS.TIME': 400., 'OS.EVENT': 1})

    def test_literal_acquisition_and_unreviewed_provenance_are_retained(self):
        row, qc = adapter.development_row(self.row(), id_col='Patient_ID', source_csv_sha256='abc')
        self.assertEqual((row['Patient_ID'], row['scan_id']), ('001', '0007'))
        self.assertEqual(row['contour_source'], 'legacy_rtstruct_derived_unreviewed')
        self.assertEqual(row['contour_version'], 'source_csv_sha256:abc')
        self.assertIn('implementation unavailable', row['ct_intensity_transform'])
        self.assertTrue(qc['pt_annotation_available'])
        self.assertTrue(qc['ln_annotation_available'])
        self.assertTrue(row['development_only'])

    def test_empty_failed_unknown_and_zero_support_are_not_known_absence(self):
        for value, count in (('empty', 0), ('MetaLymphnode:failed(RuntimeError)', 5), ('unknown', 5), ('MetaLymphnode:aligned', 0)):
            row = self.row(); row.update(mask_nodal_qc=value, mask_nodal_voxels=count)
            with self.subTest(qc=value, count=count):
                prepared, qc = adapter.development_row(row, id_col='Patient_ID', source_csv_sha256='abc')
                self.assertFalse(prepared['ln_annotation_available'])
                self.assertFalse(qc['ln_annotation_available'])
        row = self.row(); row['mask_primary_qc'] = 'PT:failed(RuntimeError)'
        self.assertFalse(adapter.development_row(row, id_col='Patient_ID', source_csv_sha256='abc')[0]['pt_annotation_available'])

    def test_missing_or_contradictory_acquisition_identity_is_rejected(self):
        for update in ({'ct_series_uid': pd.NA}, {'rtstruct_referenced_series_uid': 'different-scan'}, {'status': 'failed'}):
            row = self.row(); row.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                adapter.development_row(row, id_col='Patient_ID', source_csv_sha256='abc')

    def test_representative_selection_is_deterministic_and_outcome_free(self):
        rows = [self.row('001', pt=1., ln=8., z=3.), self.row('002', pt=100., ln=1., z=1.),
                self.row('003', pt=5., ln=100., z=2.), self.row('004', pt=10., ln=0., z=4.),
                self.row('005', pt=50., ln=10., z=.5), self.row('006', pt=1., ln=8., z=3.)]
        rows[3].update(mask_nodal_qc='empty', mask_nodal_voxels=0)
        chosen, reasons = adapter.select_representatives(rows, id_col='Patient_ID')
        self.assertEqual([row['Patient_ID'] for row in chosen], ['001', '002', '003', '004', '005'])
        self.assertIn('no_nodal_support_annotation_unknown', reasons['004'])
        changed = copy.deepcopy(rows[::-1])
        for row in changed:
            row['OS.TIME'], row['OS.EVENT'] = 1e9, 0
        second, second_reasons = adapter.select_representatives(changed, id_col='Patient_ID')
        self.assertEqual([row['Patient_ID'] for row in second], [row['Patient_ID'] for row in chosen])
        self.assertEqual(second_reasons, reasons)

    def test_feature_csv_uses_union_of_available_compartments(self):
        rows = [{'patient_id': '001', 'scan_id': '0007', 'feature_status__ln_intra': 'missing'},
                {'patient_id': '002', 'scan_id': '0008', 'feature_status__ln_intra': 'valid',
                 'LN_intratumor__original_firstorder_Mean': .25}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'features.csv'
            adapter.write_feature_table(rows, path)
            frame = read_identifier_csv(path)
        self.assertTrue(pd.isna(frame.loc[0, 'LN_intratumor__original_firstorder_Mean']))
        self.assertEqual(frame.loc[1, 'LN_intratumor__original_firstorder_Mean'], .25)
        self.assertEqual(frame.loc[0, 'scan_id'], '0007')

    def test_declared_available_annotation_requires_actual_positive_binary_support(self):
        ct = sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.float32))
        empty = sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, 'absence is not established'):
            adapter.validate_available_mask(empty, ct_image=ct, source=Path('synthetic-empty'))
        values = np.zeros((4, 4, 4), dtype=np.float32); values[1, 1, 1] = .25
        with self.assertRaisesRegex(ValueError, 'finite binary'):
            adapter.validate_available_mask(sitk.GetImageFromArray(values), ct_image=ct, source=Path('synthetic-soft'))
        values[1, 1, 1] = 1
        self.assertEqual(adapter.validate_available_mask(sitk.GetImageFromArray(values), ct_image=ct, source=Path('synthetic-mask')), 1)

    def test_scientific_cli_requires_h100_before_any_input_read(self):
        with patch.object(adapter, 'require_one_h100', side_effect=RuntimeError('no approved H100')), \
                patch.object(adapter, 'read_identifier_csv', side_effect=AssertionError('must not read sources')):
            with self.assertRaisesRegex(RuntimeError, 'no approved H100'):
                adapter.main(['--source-csv', 'missing.csv', '--data-root', 'missing', '--out-dir', 'missing-out',
                              '--grid-config-json', 'missing.json', '--radiomics-config-json', 'missing-rads.json',
                              '--output-shape', '96', '160', '96'])

    def test_all_case_overlays_use_representatives_only(self):
        rows = [self.row(str(index).zfill(3), pt=10., ln=5., z=2.) for index in range(30)]
        self.assertEqual(adapter.overlay_patient_ids(rows, id_col='Patient_ID', selection_mode='all_eligible'), ['000'])
        explicit = adapter.overlay_patient_ids(rows[:3], id_col='Patient_ID', selection_mode='explicit_identifiers')
        self.assertEqual(explicit, ['000', '001', '002'])

    def test_overlay_receipt_binds_mock_artifact_without_rendering_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows = [dict(self.row(), region_manifest_path='cases/001/region_manifest.json')]

            def fake_render(manifest, path, *, synthetic):
                self.assertFalse(synthetic)
                self.assertEqual(manifest, output / rows[0]['region_manifest_path'])
                path.parent.mkdir(parents=True)
                path.write_bytes(b'engineering-mock-not-an-image')

            result = adapter.render_qc_overlays(rows, ['001', '002'], id_col='Patient_ID', output=output, render=fake_render)
            self.assertEqual(result['001']['sha256'], adapter.file_sha256(output / result['001']['path']))
            self.assertEqual(result['001']['status'], 'rendered_pending_visual_review')
            self.assertEqual(result['002']['status'], 'not_prepared')


if __name__ == '__main__':
    unittest.main()
