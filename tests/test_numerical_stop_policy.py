"""Fabricated numerical-policy checks; no cohort or GPU execution."""
import argparse
import contextlib
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from trifusesurv2.multimodal_survival import train
from trifusesurv2.utils.experiment import canonical_experiment_spec, experiment_spec_sha256


class NumericalStopPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'fold_00/numerical_failure.json'

    def context(self, **extra):
        return {'receipt_path': str(self.path), 'required_parameter_names': ['weight'],
                'mode': 'F2', 'seed': 42, 'fold': 0, 'epoch': 1,
                'microbatch_index': 216, 'attempted_update_in_epoch': 27,
                'actual_accumulation_batches': 8, 'loss_divisor': 8, **extra}

    def model_optimizer(self):
        model = torch.nn.Linear(2, 1, bias=False)
        return model, torch.optim.AdamW(model.parameters(), lr=.01)

    def receipt(self):
        return json.loads(self.path.read_text())

    def test_default_policy_and_explicit_none_preserve_canonical_identity(self):
        plain = argparse.Namespace(seed=42)
        inactive = argparse.Namespace(seed=42, numerical_failure_policy=None)
        self.assertEqual(canonical_experiment_spec(plain), canonical_experiment_spec(inactive))
        active = argparse.Namespace(seed=42, numerical_failure_policy=train.NUMERICAL_FAILURE_POLICY)
        self.assertNotEqual(experiment_spec_sha256(canonical_experiment_spec(plain)),
                            experiment_spec_sha256(canonical_experiment_spec(active)))
        self.assertEqual(train._numerical_policy_contract(inactive), {})
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            train._numerical_policy_contract(argparse.Namespace(numerical_failure_policy='tolerate_one_skip'))

    def test_finite_monitored_update_is_bitwise_equal_to_ordinary_update(self):
        model, optimizer = self.model_optimizer()
        state = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state().clone()
        outcomes = []
        for active in [False, True]:
            model, optimizer = self.model_optimizer()
            model.load_state_dict(state)
            torch.set_rng_state(rng)
            model(torch.ones(1, 2)).sum().backward()
            observed = train.perform_optimizer_update(model, optimizer,
                **({'numerical_failure_context': self.context()} if active else {}))
            outcomes.append((copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), observed,
                             torch.get_rng_state().clone()))
        for key in outcomes[0][0]:
            self.assertTrue(torch.equal(outcomes[0][0][key], outcomes[1][0][key]))
        for index, values in outcomes[0][1]['state'].items():
            for key, value in values.items():
                self.assertTrue(torch.equal(value, outcomes[1][1]['state'][index][key]))
        self.assertEqual(outcomes[0][2], outcomes[1][2])
        self.assertTrue(torch.equal(outcomes[0][3], outcomes[1][3]))
        self.assertFalse(self.path.exists())

    def test_real_scaler_nonfinite_elements_are_recorded_before_clipping(self):
        model, optimizer = self.model_optimizer()
        scaler = torch.amp.GradScaler('cpu', init_scale=32768)
        scaler.scale(model(torch.ones(1, 2)).sum()).backward()
        model.weight.grad[0, 0] = float('inf')
        before = model.weight.detach().clone()
        with patch.object(train, 'clip_grad_norm_', side_effect=AssertionError('clip must not run')), \
             self.assertRaises(train.NumericalTrainingFailure):
            train.perform_optimizer_update(model, optimizer, scaler,
                numerical_failure_context=self.context(scaler_state_before=scaler.state_dict()))
        receipt = self.receipt()
        self.assertEqual(receipt['reasons'], ['nonfinite_gradient_elements'])
        evidence = receipt['evidence']
        self.assertEqual(evidence['disposition_phase'], 'before_clip_before_optimizer_step')
        self.assertEqual(evidence['actual_optimizer_calls'], 0)
        self.assertIsNone(evidence['optimizer_steps_skipped'])
        self.assertEqual(evidence['preclip_gradients']['parameters']['weight']['infinite_elements'], 1)
        self.assertEqual(evidence['preclip_gradients']['parameters']['weight']['nan_elements'], 0)
        self.assertTrue(torch.equal(before, model.weight))
        self.assertFalse(optimizer.state)
        self.assertFalse(optimizer._optimizer_step_post_hooks)
        self.assertEqual(receipt['location']['attempted_update_in_epoch'], 27)

    def test_required_missing_gradient_is_distinct_from_inactive_parameters(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        model.weight.grad = torch.ones_like(model.weight)
        observed = train.observe_preclip_gradients(model, ['weight'])
        self.assertEqual(observed['grad_none'], ['bias'])
        self.assertEqual(observed['missing_required_parameters'], [])
        with self.assertRaises(train.NumericalTrainingFailure):
            train.perform_optimizer_update(model, optimizer,
                numerical_failure_context=self.context(required_parameter_names=['weight', 'bias']))
        self.assertEqual(self.receipt()['reasons'], ['missing_required_gradients'])

    def test_finite_elements_and_nonfinite_aggregate_norm_are_distinguished(self):
        model, optimizer = self.model_optimizer()
        model.weight.grad = torch.full_like(model.weight, 3e38)
        with self.assertRaises(train.NumericalTrainingFailure):
            train.perform_optimizer_update(model, optimizer, numerical_failure_context=self.context())
        receipt = self.receipt()
        self.assertEqual(receipt['reasons'], ['nonfinite_aggregate_clipping_norm'])
        preclip = receipt['evidence']['preclip_gradients']
        self.assertEqual(preclip['nonfinite_parameters'], [])
        self.assertGreater(preclip['finite_elements_global_l2_norm_float64'], 3e38)
        self.assertEqual(receipt['evidence']['actual_optimizer_calls'], 0)
        self.assertFalse(optimizer.state)

    def test_finite_gradient_skip_is_observed_and_receipt_cannot_be_overwritten(self):
        class SkipScaler:
            def get_scale(self): return 32768.
            def state_dict(self): return {'scale': 32768.}
            def unscale_(self, optimizer): pass
            def step(self, optimizer): pass
            def update(self): pass
        model, optimizer = self.model_optimizer()
        model.weight.grad = torch.ones_like(model.weight)
        with self.assertRaises(train.NumericalTrainingFailure):
            train.perform_optimizer_update(model, optimizer, SkipScaler(), numerical_failure_context=self.context())
        raw = self.path.read_bytes()
        receipt = self.receipt()
        self.assertEqual(receipt['reasons'], ['optimizer_step_skipped'])
        self.assertEqual(receipt['evidence']['actual_optimizer_calls'], 0)
        self.assertEqual(receipt['evidence']['optimizer_steps_skipped'], 1)
        model.weight.grad = torch.ones_like(model.weight)
        with self.assertRaises(FileExistsError):
            train.perform_optimizer_update(model, optimizer, SkipScaler(), numerical_failure_context=self.context())
        self.assertEqual(self.path.read_bytes(), raw)

    def test_checkpoint_policy_is_bound_and_tampering_is_rejected(self):
        args = argparse.Namespace(numerical_failure_policy=train.NUMERICAL_FAILURE_POLICY)
        policy = train._numerical_policy_contract(args)
        contract = {**policy, 'experiment_spec': canonical_experiment_spec(args, runtime=policy)}
        ck = {'args': vars(args), 'encoders': {'model_contract': contract}}
        self.assertEqual(train._checkpoint_numerical_metadata(ck), policy)
        changed = copy.deepcopy(ck)
        changed['args']['numerical_failure_policy'] = None
        with self.assertRaisesRegex(ValueError, 'disagree'):
            train._checkpoint_numerical_metadata(changed)
        changed = copy.deepcopy(ck)
        changed['encoders']['model_contract']['experiment_spec']['runtime']['numerical_failure_action'] = 'continue'
        with self.assertRaisesRegex(ValueError, 'disagree'):
            train._checkpoint_numerical_metadata(changed)
        self.assertEqual(train._checkpoint_numerical_metadata({'args': {}}), {})

    def test_failure_escapes_sequential_runner_before_the_second_mode(self):
        second = Path(self.tmp.name) / 'F3-START.json'
        code = '''import subprocess,sys\nfrom pathlib import Path\nsubprocess.run([sys.executable,'-c',sys.argv[1]],check=True)\nPath(sys.argv[2]).write_text('must not run')\n'''
        failure = f'''import torch\nfrom trifusesurv2.multimodal_survival import train\nm=torch.nn.Linear(1,1,bias=False)\no=torch.optim.AdamW(m.parameters())\nm.weight.grad=torch.full_like(m.weight,float('inf'))\ntrain.perform_optimizer_update(m,o,numerical_failure_context={{'receipt_path':{str(self.path)!r},'required_parameter_names':['weight']}})\n'''
        completed = subprocess.run([sys.executable, '-c', code, failure, str(second)],
                                   capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(second.exists())
        self.assertTrue(self.path.exists())
        self.assertIn('NUMERICAL_STOP', completed.stderr)


class ActualNumericalPolicyFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_actual_verified_fold_binds_policy_and_keeps_reserved_test_unread(self):
        import io
        import pandas as pd
        import test_region_data as fixtures
        fixture = fixtures.RegionDataTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        rows, features = [], []
        for index in range(5):
            row, manifest, _ = fixture.case(f'{index + 1:03d}')
            row.update(time=120. + index * 120., event=float(index % 2 == 0), AGE=50 + index,
                       HPV=index % 2, T=2, N=1)
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
                '--clinical_type_policy', 'approved_categorical_v1', '--clinical_cols', 'AGE', 'HPV', 'T', 'N',
                '--numerical_failure_policy', train.NUMERICAL_FAILURE_POLICY,
                '--img_size', '32', '32', '32', '--feature_size', '12', '--depths', '1', '1', '1', '1',
                '--img_token_dim', '16', '--v2_model_dim', '16', '--v2_num_heads', '4',
                '--v2_transformer_layers', '1', '--v2_radiomics_pcs_per_group', '2',
                '--epochs', '1', '--batch_size', '1', '--eval_batch_size', '2', '--grad_accumulation_steps', '2',
                '--workers', '0', '--eval_workers', '0', '--no_ema', '--no_swa', '--no_resume',
                '--risk_horizon_days', '240', '--auc_times_days', '120', '240', '--time_bin_width_days', '120']
        with patch('sys.argv', argv):
            args = train.parse_args()
        root = fixture.root / 'run'
        split = {'train': ['001', '002', '003'], 'val': ['004', '005'], 'test': ['reserved-unavailable']}
        with patch.object(train, 'predict_risk_scores', side_effect=AssertionError('held-out prediction entered')), \
             contextlib.redirect_stdout(io.StringIO()):
            train.run_one_fold(fold=0, args=args, meta=meta, split=split, device=torch.device('cpu'),
                scaler=None, autocast_ctx=contextlib.nullcontext, out_root=root)
        fold = root / 'fold_00'
        contract = json.loads((fold / 'model_contract.json').read_text())
        self.assertEqual(contract['numerical_failure_policy'], train.NUMERICAL_FAILURE_POLICY)
        self.assertEqual(contract['experiment_spec']['settings']['numerical_failure_policy'], train.NUMERICAL_FAILURE_POLICY)
        self.assertEqual(contract['experiment_spec']['runtime']['numerical_failure_receipt_schema'], 1)
        self.assertEqual(contract['test_ids'], ['reserved-unavailable'])
        metrics = pd.read_csv(fold / 'metrics.csv').iloc[0]
        self.assertEqual(int(metrics['optimizer_steps_succeeded']), 2)
        self.assertEqual(int(metrics['optimizer_steps_skipped']), 0)
        ck = torch.load(fold / 'best.pt', map_location='cpu', weights_only=False)
        self.assertEqual(train._checkpoint_numerical_metadata(ck), train._numerical_policy_contract(args))
        self.assertFalse((fold / 'numerical_failure.json').exists())
        self.assertFalse(list(fold.glob('*test*csv')))

    def test_actual_tiny_qualifier_executes_active_policy_for_both_modes(self):
        import io
        from test_swin_survival_qualification import qualification, tiny_settings
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = tiny_settings()
            for row in settings.values():
                row['numerical_failure_policy'] = train.NUMERICAL_FAILURE_POLICY
            path = root / 'settings.json'; output = root / 'qualification.json'
            path.write_text(json.dumps({'settings_by_mode': settings}))
            with contextlib.redirect_stdout(io.StringIO()):
                code = qualification.main(['--training-settings', str(path), '--settings-sha256', qualification.file_sha256(path),
                    '--output', str(output), '--profile', 'tiny', '--device', 'cpu', '--precision', 'fp32', '--batches', '3'])
            report = json.loads(output.read_text())
            self.assertEqual(code, 0, report.get('error'))
            self.assertEqual(report['schema_version'], 1)
            for row in report['modes'].values():
                self.assertEqual(row['numerical_failure_contract']['numerical_failure_policy'], train.NUMERICAL_FAILURE_POLICY)
                self.assertEqual(row['actual_successful_updates'], 2)
                self.assertEqual(row['training_statistics']['optimizer_steps_skipped'], 0)
                self.assertFalse(Path(row['numerical_failure_receipt_path']).exists())


if __name__ == '__main__':
    unittest.main()
