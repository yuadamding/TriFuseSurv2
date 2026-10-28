"""Fabricated-data checks for development isolation and actual optimizer updates."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import torch

import test_region_data as fixtures
from trifusesurv2.multimodal_survival import train


class DevelopmentIsolationTests(unittest.TestCase):
    def test_main_does_not_interpret_test_outcomes_or_inspect_test_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'metadata.csv'
            pd.DataFrame({'Patient_ID': ['001', 'NA', 'test-only'],
                          'OS.TIME': [120, 240, 'not a time'], 'OS.EVENT': [1, 0, 7],
                          'ct': ['train-image', 'val-image', '/must/not/read/test-image']}).to_csv(path, index=False)
            split = {'train': ['001'], 'val': ['NA'], 'test': ['test-only']}
            argv = ['train', '--meta_csv', str(path), '--device', 'cpu', '--out_dir', str(root),
                    '--exp_name', 'development', '--cv_folds', '1', '--development-only', '--no_resume']
            result = {'export_risks': {}, 'test_metrics_export': {}, 'num_time_bins': 2,
                      'train_max_time_days': 240, 'export_suffix': 'last'}
            with patch('sys.argv', argv), patch.object(train, 'load_precomputed_splits', return_value={0: split}), \
                 patch.object(train, 'resolve_img_size_against_data', return_value=((32, 32, 32), (32, 32, 32))) as inspect, \
                 patch.object(train, 'run_one_fold', return_value=result) as run, contextlib.redirect_stdout(io.StringIO()):
                train.main()
            self.assertEqual(inspect.call_args.args[0]['Patient_ID'].tolist(), ['001', 'NA'])
            self.assertEqual(run.call_args.kwargs['meta']['Patient_ID'].tolist(), ['001', 'NA'])
            self.assertEqual(run.call_args.kwargs['split'], split)
            summary = json.loads((root / 'development/cv_summary.json').read_text())
            self.assertNotIn('mean_fold_test_c_index', summary)
            self.assertEqual(summary['reserved_test_ids_by_fold'], {'0': ['test-only']})
            self.assertEqual(summary['test_evaluation'], 'not_run')
            self.assertFalse((root / 'development/cv_test_risks_export.csv').exists())

    def test_reserved_test_ids_need_not_have_metadata_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'meta.csv'
            path.write_text('Patient_ID,OS.EVENT\n001,1\nNA,0\n')
            frame = train.read_development_metadata(str(path), 'patient_id',
                        {'train': ['001'], 'val': ['NA'], 'test': ['absent']})
            self.assertEqual(frame['Patient_ID'].tolist(), ['001', 'NA'])

    def test_test_curve_export_is_rejected_before_any_forward_or_file_read(self):
        with self.assertRaisesRegex(ValueError, 'forbids held-out'):
            train.export_survival_curves(None, None, None, outcome_frame=None,
                args=SimpleNamespace(development_only=True), fold=0, split='test', checkpoint_path=None,
                contract_path=None, weights='best', autocast_ctx=None, teacher_force_alpha=0, output_path=None)

    def test_cli_default_and_explicit_guard(self):
        with patch('sys.argv', ['train', '--meta_csv', 'unused.csv']):
            self.assertFalse(train.parse_args().development_only)
        argv = ['train', '--meta_csv', 'unused.csv', '--development-only', '--cv_folds', '1',
                '--epoch_gradcam', '--epoch_gradcam_split', 'test']
        with patch('sys.argv', argv), self.assertRaisesRegex(ValueError, 'held-out test'):
            train.parse_args()


class OptimizerHealthTests(unittest.TestCase):
    def test_real_grad_scaler_overflow_skips_optimizer_state_update(self):
        for overflow in (False, True):
            with self.subTest(overflow=overflow):
                model = torch.nn.Linear(1, 1, bias=False)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
                scaler = torch.amp.GradScaler('cpu')
                scaler.scale(model(torch.ones(1, 1)).sum()).backward()
                if overflow:
                    model.weight.grad.fill_(float('inf'))
                observed = train.perform_optimizer_update(model, optimizer, scaler)
                self.assertEqual(observed['optimizer_steps_succeeded'], int(not overflow))
                self.assertEqual(observed['optimizer_steps_skipped'], int(overflow))
                self.assertEqual(bool(optimizer.state), not overflow)

    def test_success_and_scaler_skip_are_distinguished_even_with_none_step_return(self):
        class Scaler:
            def __init__(self, skip):
                self.skip, self.scale = skip, 128.
            def get_scale(self):
                return self.scale
            def unscale_(self, optimizer):
                pass
            def step(self, optimizer):
                if not self.skip:
                    return optimizer.step()
            def update(self):
                if self.skip:
                    self.scale /= 2
        for skip in (False, True):
            with self.subTest(skip=skip):
                model = torch.nn.Linear(2, 1, bias=False)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
                before = model.weight.detach().clone()
                model(torch.ones(1, 2)).sum().backward()
                observed = train.perform_optimizer_update(model, optimizer, Scaler(skip))
                self.assertEqual(observed['optimizer_step_attempts'], 1)
                self.assertEqual(observed['optimizer_steps_succeeded'], int(not skip))
                self.assertEqual(observed['optimizer_steps_skipped'], int(skip))
                self.assertGreater(observed['gradient_norm'], 1)
                self.assertEqual(torch.equal(before, model.weight), skip)
                self.assertFalse(optimizer._optimizer_step_post_hooks)

    def test_cpu_update_counts_actual_step_and_reports_nonfinite_gradient(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
        model.weight.grad = torch.full_like(model.weight, float('inf'))
        observed = train.perform_optimizer_update(model, optimizer)
        self.assertEqual(observed['optimizer_steps_succeeded'], 1)
        self.assertEqual(observed['gradient_norm'], float('inf'))


class DevelopmentFoldTests(unittest.TestCase):
    def test_actual_fold_uses_no_test_manifests_loader_or_predictions_and_records_updates(self):
        fixture = fixtures.RegionDataTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        rows, features = [], []
        for index in range(5):
            row, manifest, _ = fixture.case(f'{index + 1:03d}')
            row.update(time=120. + index * 120., event=float(index % 2 == 0), AGE=50 + index,
                       HPV=index % 2, T=2, N=1, NSTAGE='must-be-excluded')
            rows.append(row)
            features.append(fixtures.extract.extract_case(manifest, fixtures.FeatureExtractor(), fixture.config,
                                                         software_version='test-adapter'))
        meta = pd.DataFrame(rows)
        metadata, radiomics = fixture.output / 'meta.csv', fixture.output / 'radiomics.csv'
        meta.to_csv(metadata, index=False)
        pd.DataFrame(features).to_csv(radiomics, index=False)
        argv = ['train', '--meta_csv', str(metadata), '--radiomics_root', str(radiomics),
                '--ct_col', 'ct', '--mask_pt_col', 'pt', '--mask_ln_col', 'ln',
                '--time_col', 'time', '--event_col', 'event', '--device', 'cpu',
                '--development-only', '--cv_folds', '1', '--report_metric', 'nll',
                '--clinical_type_policy', 'approved_categorical_v1',
                '--clinical_cols', 'AGE', 'HPV', 'T', 'N',
                '--img_size', '32', '32', '32', '--feature_size', '12', '--depths', '1', '1', '1', '1',
                '--img_token_dim', '16', '--v2_model_dim', '16', '--v2_num_heads', '4',
                '--v2_transformer_layers', '1', '--v2_radiomics_pcs_per_group', '2',
                '--epochs', '1', '--batch_size', '1', '--eval_batch_size', '2', '--grad_accumulation_steps', '2',
                '--workers', '0', '--eval_workers', '0', '--no_ema', '--no_swa', '--no_resume',
                '--risk_horizon_days', '240', '--auc_times_days', '120', '240', '--time_bin_width_days', '120']
        with patch('sys.argv', argv):
            args = train.parse_args()
        split = {'train': ['001', '002', '003'], 'val': ['004', '005'], 'test': ['missing-test']}
        root = fixture.root / 'run'
        with patch.object(train, 'predict_risk_scores', side_effect=AssertionError('test risk path entered')), \
             contextlib.redirect_stdout(io.StringIO()):
            result = train.run_one_fold(fold=0, args=args, meta=meta, split=split, device=torch.device('cpu'),
                scaler=None, autocast_ctx=contextlib.nullcontext, out_root=root)
        fold = root / 'fold_00'
        contract = json.loads((fold / 'model_contract.json').read_text())
        selection = json.loads((fold / 'development_selection.json').read_text())
        metrics = pd.read_csv(fold / 'metrics.csv').iloc[0]
        self.assertEqual(contract['evaluation_scope'], 'development_only')
        self.assertEqual(contract['test_ids'], ['missing-test'])
        self.assertEqual(len(contract['region_manifest_sha256']), 5)
        self.assertEqual(contract['clinical_column_policy']['columns'], ['AGE', 'HPV', 'T', 'N'])
        expected_types = {'AGE': 'numeric', 'HPV': 'categorical', 'T': 'categorical', 'N': 'categorical'}
        self.assertEqual(contract['clinical_type_policy']['columns'], expected_types)
        self.assertEqual(contract['experiment_spec']['settings']['clinical_type_policy'], 'approved_categorical_v1')
        self.assertEqual(contract['experiment_spec']['runtime']['clinical_type_policy'], contract['clinical_type_policy'])
        self.assertNotIn('amp_init_scale', contract['experiment_spec']['settings'])
        self.assertEqual(selection['test_exports'], [])
        self.assertEqual(selection['selection_metric'], 'nll')
        self.assertEqual(result['export_risks'], {})
        self.assertEqual(result['n_test'], 0)
        self.assertFalse(list(fold.glob('*test*csv')))
        self.assertEqual(int(metrics['training_batches']), 3)
        self.assertEqual(int(metrics['optimizer_step_attempts']), 2)
        self.assertEqual(int(metrics['optimizer_steps_succeeded']), 2)
        self.assertEqual(int(metrics['optimizer_steps_skipped']), 0)
        self.assertEqual(int(metrics['nonfinite_gradient_updates']), 0)
        self.assertEqual(int(metrics['loss_finite_checks']), 3)
        self.assertTrue(pd.notna(metrics['train_eval_nll']))
        for phase in ('train', 'train_evaluation', 'validation'):
            self.assertGreater(metrics[f'epoch_{phase}_seconds'], 0)
        self.assertIn('optimizer_lr_group_0', metrics.index)
        checkpoint = torch.load(fold / 'last.pt', weights_only=False)
        self.assertIsNotNone(checkpoint['optimizer_state'])
        self.assertNotIn('amp_init_scale', checkpoint['args'])
        self.assertEqual(checkpoint['encoders']['clinical'].type_policy, contract['clinical_type_policy'])
        self.assertNotIn('T_RAW', checkpoint['encoders']['clinical'].available_columns())


if __name__ == '__main__':
    unittest.main()
