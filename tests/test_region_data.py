"""Synthetic end-to-end checks for physical region identity across both streams."""
import copy
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from trifusesurv2.encoders.radiomics import HabitatRadiomicsTokenEncoder
from trifusesurv2.preprocessing.region_io import (
    canonical_json_sha256, load_region_manifest, load_region_arrays, validate_radiomics_binding,
)
from trifusesurv2.utils.data import PreprocessedHabitatSurvivalDataset
from trifusesurv2.schema import RADIOLOGY_HABITATS, REGION_STATUS_CODES, PHYSICAL_GEOMETRY_FEATURES
from trifusesurv2.data.batch import HabitatBatch, TokenBlock
from trifusesurv2.preprocessing import region_io
from trifusesurv2.preprocessing.identifiers import read_identifier_csv

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


crop = script('preprocess_roi_inputs')
extract = script('extract_region_radiomics')


class Clinical:
    def encode_row_token_matrix(self, row):
        return np.zeros((3, 2), dtype=np.float32), np.ones(3, dtype=np.float32)


class FeatureExtractor:
    """A test adapter, not a replacement or qualification of PyRadiomics."""
    def execute(self, image, mask, label=1):
        values = sitk.GetArrayFromImage(sitk.ReadImage(image))
        support = sitk.GetArrayFromImage(sitk.ReadImage(mask)) > 0
        return {'original_firstorder_Mean': values[support].mean(),
                'original_shape_VoxelVolume': float(support.sum()), 'diagnostics_test': 'ignored'}


class RegionDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'output'
        self.config = {'setting': {'binWidth': 25}, 'featureClass': {'firstorder': ['Mean']}}
        self.grid_config = {'schema_version': 1, 'spacing_xyz': [1.0, 1.5, 2.0], 'origin_xyz': [0.0, 0.0, 0.0],
                            'direction': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                            'units': 'mm', 'coordinate_system': 'LPS'}

    def case(self, pid='001', *, ln_available=True, value=0.5, scan_id=None):
        source = self.root / 'sources' / pid
        source.mkdir(parents=True)
        shape = (40, 40, 40)
        pt = np.zeros(shape, dtype=np.uint8); pt[16:19, 16:19, 16:19] = 1
        ln = np.zeros(shape, dtype=np.uint8); ln[22:25, 22:25, 22:25] = 1
        arrays = {'ct': np.full(shape, value, dtype=np.float32), 'pt': pt, 'ln': ln,
                  'hu': np.indices(shape).sum(axis=0).astype(np.float32) - 800 + 10 * value}
        scan_id = f'scan-{pid}' if scan_id is None else scan_id
        row = {'patient_id': pid, 'scan_id': scan_id, 'contour_source': 'expert',
               'contour_version': 'reader1-v1', 'ct_intensity_transform': 'synthetic normalized [0,1]',
               'radiomics_scan_id': scan_id, 'radiomics_intensity_transform': 'synthetic HU; no intensity normalization',
               'ln_annotation_available': ln_available, 'time': 350.0, 'event': 1.0}
        for name, array in arrays.items():
            image = sitk.GetImageFromArray(array); image.SetSpacing((1.0, 1.5, 2.0))
            path = source / f'{name}.nii.gz'; sitk.WriteImage(image, str(path))
            row[name] = str(path)
        row['radiomics_ct_path'] = row['hu']
        job = dict(row=copy.deepcopy(row), index=0, id_col='patient_id', meta_dir=str(source), out_root=str(self.output),
                   out_csv_dir=str(self.output), output_size_dhw=(32, 32, 32), region_mode='verified',
                   sampling_policy='common_grid', grid_config=copy.deepcopy(self.grid_config),
                   margin_mm=10.0, valid_tissue_policy='full_fov', overwrite=False,
                   expand_to_aspect=True, fail_empty_roi=True, ct_col='ct', mask_pt_col='pt', mask_ln_col='ln')
        result = crop._process_one(job)
        row.update(ct=result['ct_out_path'], pt=result['mask_primary_out_path'], ln=result['mask_nodal_out_path'],
                   region_manifest_path=result['region_manifest_path'])
        manifest = load_region_manifest(self.output / result['region_manifest_path'], pid)
        return row, manifest, job

    def dataset(self, row, *, rad=None, mode='eval', **kwargs):
        return PreprocessedHabitatSurvivalDataset(
            pd.DataFrame(row if isinstance(row, list) else [row]), id_col='patient_id', time_col='time', event_col='event',
            ct_col='ct', mask_pt_col='pt', mask_ln_col='ln', clinical_token_encoder=Clinical(),
            clinical_token_dim=2, radiomics_token_encoder=rad, radiomics_token_dim=rad.output_dim if rad else 0,
            use_radiomics=rad is not None, expected_dhw=(32, 32, 32), data_root=str(self.output),
            region_mode='verified', mode=mode, **kwargs,
        )

    def features(self, manifest):
        return extract.extract_case(manifest, FeatureExtractor(), self.config, software_version='test-adapter')

    def test_shared_reference_regions_survive_physical_crop_and_bind_dataset(self):
        row, manifest, job = self.case()
        sample = self.dataset(row)[0]
        self.assertEqual(tuple(sample['region_masks'].shape), (4, 32, 32, 32))
        self.assertEqual(sample['physical_geometry'].numel(), 2 * len(PHYSICAL_GEOMETRY_FEATURES))
        self.assertTrue(bool(sample['region_validity'].all()))
        tumor = sample['region_masks'][0].bool() | sample['region_masks'][2].bool()
        self.assertFalse(bool((tumor & (sample['region_masks'][1].bool() | sample['region_masks'][3].bool())).any()))
        for record in manifest['regions'].values():
            self.assertEqual(record['crop_completeness']['fraction'], 1.0)
            self.assertIn('image_to_reference_volume_ratio', record['crop_completeness'])
            self.assertIn('reference_boundary_truncated', record)
            self.assertNotEqual(record['image_region_id'], record['region_id'])
        before = (self.output / row['region_manifest_path']).stat().st_mtime_ns
        crop._process_one(job)
        self.assertEqual((self.output / row['region_manifest_path']).stat().st_mtime_ns, before)

    def test_distinct_hu_transform_keeps_matching_exact_support(self):
        row, manifest, _ = self.case()
        features = self.features(manifest)
        validate_radiomics_binding(features, manifest)
        self.assertLess(features['PT_intratumor__original_firstorder_Mean'], 0)
        for name in RADIOLOGY_HABITATS:
            self.assertEqual(features[f'region_id__{name}'], manifest['regions'][name]['image_region_id'])
            self.assertEqual(features[f'feature_status__{name}'], 'valid')

    def test_fractional_lattice_phase_survives_disk_at_large_lps_origins(self):
        _, _, job = self.case()
        for key in ('ct', 'pt', 'ln', 'hu'):
            path = job['row'][key]
            image = sitk.ReadImage(path)
            image.SetOrigin((-1000.3, -400.7, -200.2))
            sitk.WriteImage(image, path)
        job['grid_config']['spacing_xyz'] = [1.3, 1.5, 2.0]
        job['grid_config']['origin_xyz'] = [.123456789, .234567891, .345678912]
        job['out_root'] = job['out_csv_dir'] = str(self.root / 'precise')
        result = crop._process_one(job)
        manifest = load_region_manifest(Path(job['out_root']) / result['region_manifest_path'])
        image_path = Path(job['out_root']) / result['ct_out_path']
        self.assertEqual(image_path.suffix, '.mha')
        actual = sitk.ReadImage(str(image_path))
        self.assertEqual(actual.GetOrigin(), tuple(manifest['image']['geometry']['origin_xyz']))
        self.assertEqual(actual.GetSpacing(), tuple(job['grid_config']['spacing_xyz']))
        load_region_arrays(manifest, ct_path=image_path)

    def test_encoder_and_dataset_reject_mismatched_radiomics(self):
        cases = [self.case('001'), self.case('002', value=0.7)]
        rows = [self.features(manifest) for _, manifest, _ in cases]
        path = self.root / 'features.csv'; pd.DataFrame(rows).to_csv(path, index=False)
        manifests = {row['patient_id']: manifest for row, manifest, _ in cases}
        encoder = HabitatRadiomicsTokenEncoder.fit_from_wide_csv(
            radiomics_csv=path, train_ids=['001', '002'], all_ids=['001', '002'], total_pcs_per_group=1,
            require_region_provenance=True, region_manifests=manifests)
        self.assertEqual(tuple(self.dataset(cases[0][0], rad=encoder)[0]['radiomics_presence'].shape), (4,))
        changed = copy.deepcopy(rows[0]); changed['region_id__pt_peri'] = rows[1]['region_id__pt_peri']
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            validate_radiomics_binding(changed, cases[0][1])
        encoder.region_bindings['001']['support_sha256__ln_intra'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.dataset(cases[0][0], rad=encoder)[0]

    def test_numeric_acquisition_ids_survive_extraction_csv_encoder_roundtrip(self):
        cases = [self.case('001', scan_id='00001'), self.case('002', scan_id='00002', value=0.7)]
        rows = [self.features(manifest) for _, manifest, _ in cases]
        # Exercise the user-specified patient header without renaming acquisition IDs.
        for row in rows:
            row['Patient_ID'] = row.pop('patient_id')
        path = self.root / 'numeric_identifiers.csv'
        pd.DataFrame(rows).to_csv(path, index=False)
        encoder = HabitatRadiomicsTokenEncoder.fit_from_wide_csv(
            radiomics_csv=path, train_ids=['001', '002'], all_ids=['001', '002'], total_pcs_per_group=1,
            require_region_provenance=True,
            region_manifests={row['patient_id']: manifest for row, manifest, _ in cases})
        self.assertEqual(encoder.region_bindings['001']['scan_id'], '00001')
        self.assertEqual(encoder.region_bindings['001']['Patient_ID'], '001')
        self.assertTrue(bool(self.dataset(cases[0][0], rad=encoder)[0]['radiomics_presence'].all()))

    def test_preprocessing_cli_preserves_patient_and_acquisition_id_columns(self):
        _, _, job = self.case(scan_id='00001')
        source = self.root / 'input.csv'
        pd.DataFrame([job['row']]).rename(columns={'patient_id': 'Patient_ID'}).to_csv(source, index=False)
        output = self.root / 'cli_output'
        config = self.root / 'grid.json'; config.write_text(json.dumps(self.grid_config))
        with contextlib.redirect_stdout(io.StringIO()):
            crop.main(['--meta_csv', str(source), '--out_root', str(output), '--ct_col', 'ct',
                       '--mask_pt_col', 'pt', '--mask_ln_col', 'ln', '--valid_tissue_policy', 'full_fov',
                       '--grid_config_json', str(config),
                       '--output_size', '32', '32', '32'])
        meta = read_identifier_csv(output / 'cohort_preprocessed_stage2_roi.csv')
        self.assertIn('Patient_ID', meta)
        self.assertNotIn('patient_id', meta)
        self.assertEqual(meta.loc[0, 'Patient_ID'], '001')
        self.assertEqual(meta.loc[0, 'scan_id'], '00001')
        self.assertEqual(meta.loc[0, 'radiomics_scan_id'], '00001')
        manifest = load_region_manifest(output / meta.loc[0, 'region_manifest_path'], '001')
        self.assertEqual(manifest['scan_id'], '00001')
        self.assertEqual(manifest['sampling']['policy'], 'common_grid')
        self.assertEqual(manifest['image']['geometry']['spacing_xyz'], [1.0, 1.5, 2.0])

    def test_verified_cli_requires_an_explicit_common_grid(self):
        _, _, job = self.case()
        source = self.root / 'input.csv'; pd.DataFrame([job['row']]).to_csv(source, index=False)
        output = self.root / 'not_created'
        with self.assertRaisesRegex(ValueError, 'requires --grid_config_json'):
            crop.main(['--meta_csv', str(source), '--out_root', str(output), '--ct_col', 'ct',
                       '--mask_pt_col', 'pt', '--mask_ln_col', 'ln', '--valid_tissue_policy', 'full_fov'])
        self.assertFalse(output.exists())

    def test_common_grid_overflow_fails_before_output_publication(self):
        _, _, job = self.case()
        output = self.root / 'too_small'
        job.update(out_root=str(output), out_csv_dir=str(output), output_size_dhw=(8, 8, 8))
        with self.assertRaisesRegex(ValueError, 'canvas overflow'):
            crop._process_one(job)
        self.assertFalse(output.exists())

    def test_legacy_resize_remains_an_explicit_diagnostic_path(self):
        _, _, job = self.case()
        output = self.root / 'legacy_resize'
        job.update(out_root=str(output), out_csv_dir=str(output), sampling_policy='legacy_resize')
        result = crop._process_one(job)
        manifest = load_region_manifest(output / result['region_manifest_path'], '001')
        self.assertEqual(manifest['sampling']['policy'], 'legacy_resize')
        self.assertTrue(manifest['sampling']['diagnostic_only'])
        self.assertNotEqual(manifest['image']['geometry']['spacing_xyz'], self.grid_config['spacing_xyz'])

    def test_remote_node_changes_crop_but_not_primary_common_grid_measurements(self):
        cases = [self.case('001'), self.case('002')]
        distant_job = cases[1][2]
        path = Path(distant_job['row']['ln'])
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image)
        array[22:25, 22:25, 34:37] = 1
        changed = sitk.GetImageFromArray(array); changed.CopyInformation(image)
        sitk.WriteImage(changed, str(path))
        manifests = []
        for _, _, job in cases:
            output = self.root / 'common_phantom'
            job = copy.deepcopy(job)
            job.update(out_root=str(output), out_csv_dir=str(output), output_size_dhw=(64, 64, 64))
            result = crop._process_one(job)
            manifests.append(load_region_manifest(output / result['region_manifest_path'], job['row']['patient_id']))
        first, second = manifests
        self.assertEqual(first['sampling']['policy_id'], second['sampling']['policy_id'])
        self.assertEqual(first['sampling']['full_grid_geometry'], second['sampling']['full_grid_geometry'])
        self.assertNotEqual(first['sampling']['crop_start_dhw'], second['sampling']['crop_start_dhw'])
        left, right = [self.features(manifest) for manifest in manifests]
        for prefix in ('PT_intratumor__', 'PT_peritumor_10mm__'):
            for feature in ('original_firstorder_Mean', 'original_shape_VoxelVolume'):
                self.assertEqual(left[prefix + feature], right[prefix + feature])
        for name in ('pt_intra', 'pt_peri'):
            self.assertEqual(first['regions'][name]['physical_volume_mm3'], second['regions'][name]['physical_volume_mm3'])
            self.assertEqual(first['sampling']['regions'][name]['image_volume_mm3'], second['sampling']['regions'][name]['image_volume_mm3'])
        self.assertEqual(first['image']['geometry']['spacing_xyz'], self.grid_config['spacing_xyz'])

    def test_radiomics_sampling_policy_binding_cannot_be_changed(self):
        _, manifest, _ = self.case()
        row = self.features(manifest)
        row['sampling_policy_id'] = 'different-grid'
        with self.assertRaisesRegex(ValueError, 'sampling policy'):
            validate_radiomics_binding(row, manifest)

    def test_verified_warm_cache_avoids_hashes_and_decodes_and_returns_copies(self):
        row, _, _ = self.case()
        dataset = self.dataset(row, cache_volumes=True, volume_cache_size=1)
        with patch.object(sitk, 'ReadImage', wraps=sitk.ReadImage) as reads, patch.object(
                region_io, 'file_sha256', wraps=region_io.file_sha256) as hashes:
            first = dataset[0]
            self.assertGreater(reads.call_count, 0)
            self.assertEqual(hashes.call_count, 1)
            expected = {name: first[name].clone() for name in ('x', 'mask_pt', 'region_masks', 'region_states', 'physical_geometry')}
            for name in expected:
                first[name].fill_(0)
            reads.reset_mock(); hashes.reset_mock()
            second = dataset[0]
            self.assertEqual(reads.call_count, 0)
            self.assertEqual(hashes.call_count, 0)
            for name, value in expected.items():
                torch.testing.assert_close(second[name], value)

    def test_verified_cache_revalidates_touch_and_rejects_changed_ct(self):
        row, _, _ = self.case()
        dataset = self.dataset(row, cache_volumes=True, volume_cache_size=1)
        dataset[0]
        path = self.output / row['ct']
        previous = path.stat()
        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1))
        with patch.object(region_io, 'file_sha256', wraps=region_io.file_sha256) as hashes:
            dataset[0]
            self.assertEqual(hashes.call_count, 1)
        with path.open('ab') as handle:
            handle.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'CT input does not match'):
            dataset[0]

    def test_verified_cache_detects_mask_edit_with_restored_mtime(self):
        row, manifest, _ = self.case()
        dataset = self.dataset(row, cache_volumes=True, volume_cache_size=1)
        dataset[0]
        path = Path(manifest['_manifest_path']).parent / manifest['regions']['pt_intra']['image_mask_path']
        previous = path.stat()
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image)
        array[0, 0, 0] = 1
        changed = sitk.GetImageFromArray(array); changed.CopyInformation(image)
        sitk.WriteImage(changed, str(path))
        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'mask checksum differs'):
            dataset[0]

    def test_verified_cache_rejects_nonbinary_changed_supervision(self):
        row, _, _ = self.case()
        dataset = self.dataset(row, cache_volumes=True, volume_cache_size=1)
        dataset[0]
        path = self.output / row['pt']
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image).astype(np.float32)
        array[0, 0, 0] = 0.25
        changed = sitk.GetImageFromArray(array); changed.CopyInformation(image)
        sitk.WriteImage(changed, str(path))
        with self.assertRaisesRegex(ValueError, 'supervision mask must contain finite binary'):
            dataset[0]

    def test_verified_cache_rejects_manifest_change_after_initialization(self):
        row, manifest, _ = self.case()
        dataset = self.dataset(row, cache_volumes=True, volume_cache_size=1)
        dataset[0]
        path = Path(manifest['_manifest_path'])
        changed = json.loads(path.read_text()); changed['new_note'] = 'changed'
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'manifest changed after dataset initialization'):
            dataset[0]

    def test_verified_cache_is_bounded_and_reset_for_new_worker(self):
        first, _, _ = self.case('001')
        second, _, _ = self.case('002')
        dataset = self.dataset([first, second], cache_volumes=True, volume_cache_size=1)
        dataset[0]; dataset[1]
        self.assertEqual(list(dataset._verified_region_cache), ['002'])
        with patch.object(region_io, 'file_sha256', wraps=region_io.file_sha256) as hashes:
            dataset[0]
            self.assertEqual(hashes.call_count, 1)
            hashes.reset_mock()
            with patch('trifusesurv2.utils.data.os.getpid', return_value=os.getpid() + 1):
                dataset[0]
            self.assertEqual(hashes.call_count, 1)
        self.assertEqual(len(dataset._verified_region_cache), 1)

    def test_dataset_batches_reach_actual_verified_model_with_optional_modalities(self):
        from trifusesurv2.models.contour_habitat_survival import ContourAwareHabitatSurvivalModel
        from trifusesurv2.models.swinunetr_shared_roi_token_backbone import ContourAwareROITokenBackbone
        from trifusesurv2.multimodal_survival import train

        cases = [self.case('001'), self.case('002', value=0.7)]
        path = self.root / 'features.csv'
        pd.DataFrame([self.features(manifest) for _, manifest, _ in cases]).to_csv(path, index=False)
        encoder = HabitatRadiomicsTokenEncoder.fit_from_wide_csv(
            radiomics_csv=path, train_ids=['001', '002'], all_ids=['001', '002'], total_pcs_per_group=1,
            require_region_provenance=True,
            region_manifests={row['patient_id']: manifest for row, manifest, _ in cases})
        for radiomics in (None, encoder):
            with self.subTest(radiomics=radiomics is not None):
                dataset = self.dataset(cases[0][0], rad=radiomics)
                batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
                self.assertNotIn('node_tokens', batch)
                self.assertNotIn('topology_token', batch)
                if radiomics is None:
                    self.assertNotIn('radiomics_tokens', batch)
                backbone = ContourAwareROITokenBackbone(
                    img_size=(32, 32, 32), feature_size=12, depths=(1, 1, 1, 1),
                    num_heads=(3, 6, 12, 24), token_dim=16, token_mlp_dropout=0.0,
                    token_dropout=0.0, drop_rate=0.0, attn_drop_rate=0.0,
                    dropout_path_rate=0.0, require_verified_regions=True)
                model = ContourAwareHabitatSurvivalModel(
                    backbone=backbone, num_time_bins=4, time_bin_width_days=120,
                    radiomics_token_dim=encoder.output_dim if radiomics else 0,
                    clinical_token_dim=2, model_dim=16, num_heads=4,
                    transformer_layers=1, dropout=0.0, fusion_mode='F3').eval()
                with torch.no_grad():
                    logits = train._model_forward_eval(
                        model, train._unpack_surv_batch(batch), torch.device('cpu'), contextlib.nullcontext)
                self.assertEqual(set(logits), {'OS', 'DSS', 'DFS'})
                for value in logits.values():
                    self.assertEqual(tuple(value.shape), (1, 4))
                    self.assertTrue(bool(torch.isfinite(value).all()))

    def test_missing_nodal_annotation_remains_missing_and_not_absent(self):
        row, manifest, _ = self.case(ln_available=False)
        sample = self.dataset(row)[0]
        self.assertEqual(int(sample['region_states'][2]), REGION_STATUS_CODES['missing'])
        self.assertFalse(bool(sample['region_validity'][2]))
        self.assertEqual(int(sample['region_states'][1]), REGION_STATUS_CODES['invalid'])
        self.assertFalse(bool(sample['region_masks'][2].any()))
        features = self.features(manifest)
        self.assertEqual(features['feature_status__ln_intra'], 'missing')

    def test_automated_evaluation_rejects_expert_provenance(self):
        row, _, _ = self.case()
        with self.assertRaisesRegex(ValueError, 'predicted contours'):
            self.dataset(row, deployment_mode='automated')

    def test_verified_spatial_augmentation_is_not_silently_unsynchronized(self):
        row, _, _ = self.case()
        self.assertFalse(self.dataset(row, mode='train').spatial_augment)
        with self.assertRaisesRegex(ValueError, 'transforming all supports'):
            self.dataset(row, mode='train', spatial_augment=True)

    def test_support_altering_extraction_settings_are_rejected(self):
        for settings in ({'resampledPixelSpacing': [1, 1, 1]}, {'resegmentRange': [-100, 100]}, {'correctMask': True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                extract.validate_extraction_config({'setting': settings})

    def test_external_features_cannot_resegment_the_shared_support(self):
        _, manifest, _ = self.case()
        row = self.features(manifest)
        config = {'setting': {'resegmentRange': [-100, 100]}}
        row['radiomics_config_json'] = json.dumps(config)
        row['radiomics_config_sha256'] = canonical_json_sha256(config)
        with self.assertRaisesRegex(ValueError, 'changes declared region support'):
            validate_radiomics_binding(row, manifest)

    def test_valid_features_require_nonempty_image_support(self):
        _, manifest, _ = self.case()
        row = self.features(manifest)
        manifest['regions']['pt_intra']['feature_validity']['image'] = False
        with self.assertRaisesRegex(ValueError, 'absent or unavailable tissue'):
            validate_radiomics_binding(row, manifest)

    def test_integer_hu_interpolation_preserves_fractional_values(self):
        array = np.zeros((4, 4, 4), dtype=np.int16)
        array[:, :, 1:] = 1
        image = sitk.GetImageFromArray(array)
        resampled = crop._resample_to_size(image, (8, 8, 8), is_mask=False)
        self.assertEqual(resampled.GetPixelID(), sitk.sitkFloat32)
        self.assertAlmostEqual(float(sitk.GetArrayFromImage(resampled)[0, 0, 1]), 0.5)

    def test_verified_preprocessing_rejects_nonbinary_and_nonfinite_source_masks(self):
        _, _, original_job = self.case()
        reference = sitk.ReadImage(original_job['row']['ct'])
        for name in ('pt', 'ln', 'valid_tissue'):
            for bad_value in (float('nan'), float('inf'), 0.25, 2.0):
                with self.subTest(mask=name, value=bad_value):
                    job = copy.deepcopy(original_job)
                    array = np.zeros((40, 40, 40), dtype=np.float32)
                    array[16, 16, 16] = bad_value
                    image = sitk.GetImageFromArray(array)
                    image.CopyInformation(reference)
                    path = self.root / 'invalid_source_mask.mha'
                    sitk.WriteImage(image, str(path))
                    if name == 'valid_tissue':
                        job['valid_tissue_policy'] = 'mask'
                        job['row']['valid_tissue_out_path'] = str(path)
                        job['row']['valid_tissue_definition'] = 'synthetic explicit tissue'
                    else:
                        job['row'][name] = str(path)
                    with self.assertRaisesRegex(ValueError, 'finite binary 0/1'):
                        crop._process_one(job)

    def test_persisted_support_does_not_hide_nonbinary_values_by_thresholding(self):
        row, manifest, _ = self.case()
        base = Path(manifest['_manifest_path']).parent
        path = base / manifest['regions']['pt_intra']['image_mask_path']
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image).astype(np.float32)
        array[0, 0, 0] = 0.25  # The old threshold would retain the same support hash.
        changed = sitk.GetImageFromArray(array)
        changed.CopyInformation(image)
        sitk.WriteImage(changed, str(path))
        with self.assertRaisesRegex(ValueError, 'finite binary'):
            load_region_arrays(manifest, ct_path=self.output / row['ct'])

    def test_manifest_checksum_detects_changed_image(self):
        row, manifest, _ = self.case()
        path = self.output / row['ct']
        with path.open('ab') as file:
            file.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'CT input does not match'):
            load_region_arrays(manifest, ct_path=path)

    def test_structured_batch_checks_correspondence(self):
        image_ids = (('global', 'pt', 'peri', 'ln', 'lnperi', 'geometry'),)
        radiomics_ids = (('pt', 'peri', 'ln', 'wrong'),)
        batch = HabitatBatch(image=TokenBlock(torch.zeros(1, 6, 2), region_ids=image_ids),
                             radiomics=TokenBlock(torch.zeros(1, 4, 2), region_ids=radiomics_ids), verified_regions=True)
        with self.assertRaisesRegex(ValueError, 'correspondence mismatch'):
            batch.validate()


if __name__ == '__main__':
    unittest.main()
