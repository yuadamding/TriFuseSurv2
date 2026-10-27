"""Synthetic end-to-end checks for physical region identity across both streams."""
import copy
import contextlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

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

    def case(self, pid='001', *, ln_available=True, value=0.5):
        source = self.root / 'sources' / pid
        source.mkdir(parents=True)
        shape = (40, 40, 40)
        pt = np.zeros(shape, dtype=np.uint8); pt[16:19, 16:19, 16:19] = 1
        ln = np.zeros(shape, dtype=np.uint8); ln[22:25, 22:25, 22:25] = 1
        arrays = {'ct': np.full(shape, value, dtype=np.float32), 'pt': pt, 'ln': ln,
                  'hu': np.indices(shape).sum(axis=0).astype(np.float32) - 800 + 10 * value}
        row = {'patient_id': pid, 'scan_id': f'scan-{pid}', 'contour_source': 'expert',
               'contour_version': 'reader1-v1', 'ct_intensity_transform': 'synthetic normalized [0,1]',
               'radiomics_scan_id': f'scan-{pid}', 'radiomics_intensity_transform': 'synthetic HU; no intensity normalization',
               'ln_annotation_available': ln_available, 'time': 350.0, 'event': 1.0}
        for name, array in arrays.items():
            image = sitk.GetImageFromArray(array); image.SetSpacing((1.0, 1.5, 2.0))
            path = source / f'{name}.nii.gz'; sitk.WriteImage(image, str(path))
            row[name] = str(path)
        row['radiomics_ct_path'] = row['hu']
        job = dict(row=copy.deepcopy(row), index=0, id_col='patient_id', meta_dir=str(source), out_root=str(self.output),
                   out_csv_dir=str(self.output), output_size_dhw=(32, 32, 32), region_mode='verified',
                   margin_mm=10.0, valid_tissue_policy='full_fov', overwrite=False,
                   expand_to_aspect=True, fail_empty_roi=True, ct_col='ct', mask_pt_col='pt', mask_ln_col='ln')
        result = crop._process_one(job)
        row.update(ct=result['ct_out_path'], pt=result['mask_primary_out_path'], ln=result['mask_nodal_out_path'],
                   region_manifest_path=result['region_manifest_path'])
        manifest = load_region_manifest(self.output / result['region_manifest_path'], pid)
        return row, manifest, job

    def dataset(self, row, *, rad=None, mode='eval', **kwargs):
        return PreprocessedHabitatSurvivalDataset(
            pd.DataFrame([row]), id_col='patient_id', time_col='time', event_col='event',
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
