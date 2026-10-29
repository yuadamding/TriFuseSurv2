"""Critical synthetic guards for source screening and fixed-order eligibility."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml

import prepare as p


SETTINGS = {'global_lattice_spacing_mm': 4., 'interpolation': 'linear',
    'intensity_range': [0., 1.], 'excluded_constant_values': [0., .5, 1.],
    'constant_exclusion_tolerance': 1e-6, 'fingerprint_quantization': 1e-5,
    'candidate_min_matched_points': 64,
    'candidate_min_fraction_smaller_informative_support': .1,
    'review_min_common_informative_points': 1000, 'review_equal_tolerance': 1e-5,
    'review_min_nearly_equal_fraction': .5, 'review_min_correlation': .995,
    'review_max_mean_absolute_difference': .005}


def image(array, origin=(0., 0., 0.)):
    result = sitk.GetImageFromArray(array.astype(np.float32))
    result.SetSpacing((1., 1., 1.)); result.SetOrigin(origin)
    return result


class PhysicalScreenTests(unittest.TestCase):
    def test_shifted_crops_match_global_physical_coordinates_without_UID(self):
        rng = np.random.default_rng(34)
        original = rng.uniform(.05, .95, (36, 36, 36)).astype(np.float32)
        a = image(original[:28, :28, :28])
        b = image(original[2:30, 3:31, 1:29], origin=(1., 3., 2.))
        unrelated = image(rng.uniform(.05, .95, (28, 28, 28)))
        keys = [p.fingerprint(i, SETTINGS)[0] for i in (a, b, unrelated)]
        counts = p.matching_counts(keys)
        self.assertGreaterEqual(int(counts[0, 1]), 64)
        self.assertGreater(int(counts[0, 1]) / min(len(keys[0]), len(keys[1])), .1)
        self.assertLess(int(counts[0, 2]), 64)
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp) / 'left.nii.gz', Path(temp) / 'right.nii.gz'
            sitk.WriteImage(a, str(left)); sitk.WriteImage(b, str(right))
            review = p.review_pair(left, right, SETTINGS)
        self.assertTrue(review['shared_source_content_threshold_met'])
        self.assertTrue(review['analytically_exclude_both'])
        self.assertGreater(review['fraction_absolute_difference_within_tolerance'], .999)

    def test_constant_padding_does_not_make_an_independence_decision(self):
        constant = image(np.zeros((24, 24, 24), dtype=np.float32))
        keys, _ = p.fingerprint(constant, SETTINGS)
        self.assertEqual(len(keys), 0)
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp) / 'left.nii.gz', Path(temp) / 'right.nii.gz'
            sitk.WriteImage(constant, str(left)); sitk.WriteImage(constant, str(right))
            review = p.review_pair(left, right, SETTINGS)
        self.assertFalse(review['sufficient_support'])
        self.assertTrue(review['analytically_exclude_both'])
        self.assertEqual(review['disposition'], 'unresolved_insufficient_review_support')

    def test_real_values_must_be_finite_and_nonconstant(self):
        observed = p.informative(np.array([0, .5, 1, np.nan, np.inf, -.1, 1.1, .2, .7]), SETTINGS)
        self.assertEqual(observed.tolist(), [False] * 7 + [True, True])


class EligibilityTests(unittest.TestCase):
    def test_primary_only_does_not_require_nodal_annotation(self):
        row = {'ct_readable': 'True', 'ct_nonfinite_voxels': '0',
               'primary_readable': 'True', 'primary_geometry_matches_ct': 'True',
               'primary_nonfinite_voxels': '0', 'primary_labels': '[0,1]',
               'primary_voxels': '100', 'primary_touches_array_boundary': 'False',
               'nodal_readable': 'True', 'nodal_geometry_matches_ct': 'True',
               'nodal_nonfinite_voxels': '0', 'nodal_labels': '[0]', 'nodal_voxels': '0',
               'nodal_touches_array_boundary': 'False', 'roi_primary_names': 'PT',
               'meta_patient_id_matches_directory': 'True', 'mask_primary_qc_metadata': 'PT:aligned',
               'primary_nodal_overlap_voxels': '0'}
        self.assertTrue(all(p.rule_flags(row, False, 100).values()))
        self.assertFalse(p.rule_flags(row, True, 100)['nonempty_required_regions'])
        row['roi_primary_names'] = 'PT|PeriTumor'
        self.assertFalse(p.rule_flags(row, False, 100)['primary_is_PT_only'])

    def test_no_date_inference_from_UID_paths_or_processing_time(self):
        record = {'ct_series_uid': '1.20250916123116685',
                  'rtstruct_path': '/RS202509161231.dcm', 'created_at_utc': '2026-10-29'}
        self.assertEqual(p.timing_fields(record), [])
        record['AcquisitionDate'] = '20250101'
        self.assertEqual(p.timing_fields(record)[0]['role'], 'CT_acquisition')

    def test_final_measurement_rule_precedes_clinical_rule_and_no_double_count(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__('shutil').rmtree(root))
        (root / 'preparation').mkdir(); (root / 'measurements').mkdir()
        (root / 'preparation/source_manifest.json').write_text(json.dumps({'fixed_inputs': {}}))
        order = ['valid_OS', 'no_unresolved_source_conflict', 'readable_finite_aligned_binary',
            'nonempty_required_regions', 'primary_is_PT_only', 'no_required_mask_boundary_contact',
            'no_PT_LN_overlap_for_paired', 'minimum_100_voxels_each_required_region',
            'valid_same_support_IQR_and_voxel_volume', 'recorded_AGE_and_MF_SEX']
        rows = []
        for patient in ('A', 'B', 'C', 'D', 'E'):
            row = {'patient_id': patient, 'split': 'train', 'OS_TIME': 100., 'OS_EVENT': 1.,
                'AGE': 50., 'entry_days': np.nan, 'primary_voxels': 150, 'nodal_voxels': 150,
                'primary_volume_cm3_audited': .0375, 'nodal_volume_cm3_audited': .0375,
                'original_z_spacing_mm': 1., 'valid_OS': True, 'no_unresolved_source_conflict': True,
                'recorded_AGE_and_MF_SEX': True,
                'paired_measurement_candidate': patient in ('A', 'C'),
                'primary_only_measurement_candidate': patient in ('A', 'B', 'C')}
            for kind in ('paired', 'primary_only'):
                for rule in order[2:8]: row[kind + '__' + rule] = True
            if patient == 'B':
                row['paired__minimum_100_voxels_each_required_region'] = False
                row['recorded_AGE_and_MF_SEX'] = False
            if patient == 'C': row['recorded_AGE_and_MF_SEX'] = False
            if patient == 'D': row['valid_OS'] = False
            if patient == 'E': row['no_unresolved_source_conflict'] = False
            rows.append(row)
        pd.DataFrame(rows).to_csv(root / 'preparation/cohort_manifest_pre_measurement.csv', index=False)
        pd.DataFrame([{'patient_id': patient, 'paired_measurement_valid': patient != 'C',
                       'primary_only_measurement_valid': patient != 'C'} for patient in ('A', 'B', 'C')]).to_csv(root / 'measurements/measurement_validation.csv', index=False)
        frame = pd.read_csv(root / 'measurements/measurement_validation.csv')
        frame.to_csv(root / 'measurements/phenotype_features.csv', index=False)
        (root / 'protocol').mkdir()
        (root / 'protocol/PROTOCOL_LOCK.json').write_text(json.dumps({'protocol_sha256': 'p' * 64,
            'code_sha256': {'measure.py': 'm' * 64}}))
        p.json_write(root / 'measurements/EXTRACTION.json', {
            'status': 'completed', 'protocol_sha256': 'p' * 64, 'code_sha256': 'm' * 64,
            'premeasurement_manifest_sha256': p.sha256(root / 'preparation/cohort_manifest_pre_measurement.csv'),
            'validation_sha256': p.sha256(root / 'measurements/measurement_validation.csv'),
            'features_sha256': p.sha256(root / 'measurements/phenotype_features.csv'), 'candidate_n': 3})
        p.cohort_final({'paths': {'protocol_lock': str(root / 'protocol/PROTOCOL_LOCK.json')},
                        'eligibility': {'flow_order': order}}, root)
        observed = pd.read_csv(root / 'preparation/cohort_manifest.csv').set_index('patient_id')
        self.assertEqual(observed.loc['C', 'paired_first_exclusion'], 'valid_same_support_IQR_and_voxel_volume')
        self.assertEqual(observed.loc['B', 'paired_first_exclusion'], 'minimum_100_voxels_each_required_region')
        self.assertEqual(observed.loc['E', 'paired_first_exclusion'], 'no_unresolved_source_conflict')
        self.assertEqual(observed.loc['D', 'paired_measurement_assessment_status'], 'not_assessed_structural_exclusion')
        self.assertNotIn('valid_same_support_IQR_and_voxel_volume', observed.loc['D', 'paired_all_applicable_exclusions'])
        self.assertTrue(observed.loc['A', 'paired_eligible'])
        self.assertEqual(int(observed.paired_eligible.sum()), 1)
        self.assertTrue(pd.read_parquet(root / 'preparation/cohort_manifest.parquet').paired_eligible.dtype == bool)
        flow = pd.read_csv(root / 'preparation/eligibility_flow.csv')
        all_flow = flow[(flow.analysis == 'paired') & (flow.split == 'all')]
        self.assertEqual(int(all_flow.newly_excluded.sum()) + int(observed.paired_eligible.sum()), 5)
        changed = pd.read_csv(root / 'measurements/measurement_validation.csv')
        changed.loc[0, 'paired_measurement_valid'] = False
        changed.to_csv(root / 'measurements/measurement_validation.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'not bound'):
            p.cohort_final({'paths': {'protocol_lock': str(root / 'protocol/PROTOCOL_LOCK.json')},
                            'eligibility': {'flow_order': order}}, root)


if __name__ == '__main__': unittest.main()
