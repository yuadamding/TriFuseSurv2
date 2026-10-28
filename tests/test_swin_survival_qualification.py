"""Fabricated CPU checks for the actual trainer/Swin qualification entrypoint."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import Mock, patch

import torch
from trifusesurv2.multimodal_survival import train

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qualify_swin_survival', ROOT / 'scripts/qualify_swin_survival.py')
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)


def tiny_settings():
    with patch('sys.argv', ['train', '--meta_csv', '/fabricated/never-read.csv']):
        settings = vars(train.parse_args())
    settings.update(img_size=[32, 32, 32], feature_size=12, depths=[1, 1, 1, 1],
        img_token_dim=16, token_mlp_hidden_dim=32, v2_model_dim=16, v2_num_heads=4,
        v2_transformer_layers=1, v2_dropout=0., use_checkpoint=True,
        token_mlp_dropout=0., token_dropout=0., drop_rate=0., attn_drop_rate=0., dropout_path_rate=0.,
        clinical_cols=qualification.CLINICAL_COLUMNS, clinical_type_policy='approved_categorical_v1',
        use_lora=False, use_ema=False, use_swa=False, resume=False, allow_partial_resume=False,
        contour_warmstart_ckpt='', contour_warmstart_dir='', align_swin_cfg_from_contour_warmstart=False,
        batch_size=1, grad_accumulation_steps=2, roi_focus_warmup_epochs=0,
        primary_surv_loss_weight=1., aux_surv_loss_weight=.35, deterministic=True,
        modality_dropout_clin_p=0., modality_dropout_rad_p=0., v2_clinical_group_dropout_p=0.,
        v2_radiomics_group_dropout_p=0., v2_node_dropout_p=0., v2_topology_dropout_p=0.,
        v2_image_habitat_dropout_p=0., v2_dropout_ramp_epochs=0, survival_uses_teacher_forced_masks=False,
        device='cpu', amp=False, ddp=False, data_parallel=False, log_every_batches=0)
    return {mode: dict(settings, fusion_mode=mode, out_dir='/fabricated/' + mode) for mode in ('F2', 'F3')}


class QualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_actual_tiny_swin_nll_training_updates_and_final_partial_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings_path = Path(temporary) / 'settings.json'; output = Path(temporary) / 'qualification.json'
            settings_path.write_text(json.dumps({'settings_by_mode': tiny_settings()}))
            with contextlib.redirect_stdout(io.StringIO()):
                code = qualification.main(['--training-settings', str(settings_path),
                    '--settings-sha256', qualification.file_sha256(settings_path), '--output', str(output),
                    '--profile', 'tiny', '--device', 'cpu', '--precision', 'fp32', '--batches', '3',
                    '--container-image', 'local-fabricated-CPU-only'])
            result = json.loads(output.read_text())
        self.assertEqual(code, 0, result.get('error'))
        self.assertEqual(result['status'], 'passed', result.get('error'))
        self.assertEqual(result['patient_input_reads'], 0)
        self.assertEqual(result['checkpoint_payload_reads'], 0)
        self.assertEqual(result['modes']['F2']['initial_state_sha256'], result['modes']['F3']['initial_state_sha256'])
        for mode in ('F2', 'F3'):
            record = result['modes'][mode]
            self.assertEqual(record['clinical_type_policy']['name'], 'approved_categorical_v1')
            self.assertEqual(record['clinical_policy']['columns'], qualification.CLINICAL_COLUMNS)
            self.assertEqual(record['actual_successful_updates'], 2)
            self.assertEqual([item['after_training_batch'] for item in record['updates']], [2, 3])
            self.assertEqual(record['training_statistics']['optimizer_steps_skipped'], 0)
            self.assertEqual(record['final_partial_accumulation_batches'], 1)
            for boundary in record['updates']:
                self.assertFalse(boundary['after_unscale_before_clip']['nonfinite_parameters'])
                self.assertFalse(boundary['after_unscale_before_clip']['missing_required_parameters'])
            for item in record['post_update_forward'].values():
                self.assertEqual(item['loss_dtype'], 'torch.float32')
            self.assertNotEqual(record['initial_state_sha256'], record['final_state_sha256'])
        json.dumps(result, allow_nan=False)

    def test_production_profile_cannot_be_promoted_from_tiny_shape(self):
        with self.assertRaisesRegex(ValueError, 'full Swin'):
            qualification.validate_settings(tiny_settings(), 'production')

    def test_wrong_policy_or_mismatched_pair_rejected(self):
        for change in ('legacy_policy', 'staging_column', 'pair_lr'):
            values = tiny_settings()
            if change == 'legacy_policy': values['F2']['clinical_type_policy'] = 'legacy'
            elif change == 'staging_column': values['F2']['clinical_cols'] = qualification.CLINICAL_COLUMNS + ['NSTAGE']
            else: values['F3']['lr_head'] = .5
            with self.subTest(change=change), self.assertRaises(ValueError):
                qualification.validate_settings(values, 'tiny')

    def test_unsupported_epoch_policies_are_rejected_before_construction(self):
        changes = {'endpoint': 'DSS', 'survival_uses_teacher_forced_masks': True,
            'teacher_force_epochs': 1, 'teacher_force_start': .5, 'teacher_force_end': .2,
            'mask_guidance_alpha': .1, 'v2_clinical_group_dropout_p': -1.,
            'v2_radiomics_group_dropout_p': -1., 'v2_node_dropout_p': float('nan')}
        for key, value in changes.items():
            values = tiny_settings()
            for mode in values: values[mode][key] = value
            with self.subTest(setting=key), self.assertRaises(ValueError):
                qualification.validate_settings(values, 'tiny')

    def test_gradient_failure_classes_remain_distinct(self):
        model = torch.nn.Linear(2, 1)
        groups = {'required': ['weight']}; required = ['weight']
        observation = qualification.observe_gradients(model, groups, required)
        with self.assertRaisesRegex(ValueError, 'empty'):
            qualification.require_gradients(observation)
        model.bias.grad = torch.ones_like(model.bias)
        observation = qualification.observe_gradients(model, groups, required)
        with self.assertRaisesRegex(ValueError, 'missing'):
            qualification.require_gradients(observation)
        for value in (float('nan'), float('inf')):
            model.weight.grad = torch.full_like(model.weight, value)
            observation = qualification.observe_gradients(model, groups, required)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'NaN or infinite'):
                qualification.require_gradients(observation)
            self.assertEqual(observation['nonfinite_parameters'], ['weight'])
            json.dumps(observation, allow_nan=False)

    def test_cpu_precision_and_scale_are_explicit(self):
        with self.assertRaisesRegex(ValueError, 'CPU fixtures'):
            qualification.make_precision(train, torch.device('cpu'), 'trainer-default', None, True)
        with self.assertRaisesRegex(ValueError, 'finite and positive'):
            qualification.make_precision(train, torch.device('cpu'), 'fp32', float('nan'), False)

    def test_trainer_precision_propagates_bound_scale_and_preserves_legacy_default(self):
        device = torch.device('cuda:0')  # An identity only; no CUDA allocation or call.
        for setting in (None, 32768.0):
            scale = 65536.0 if setting is None else setting
            state = {'scale': scale, 'growth_factor': 2., 'backoff_factor': .5,
                     'growth_interval': 2000, '_growth_tracker': 0}
            scaler = SimpleNamespace(state_dict=lambda: state)
            autocast = lambda: contextlib.nullcontext()
            trainer = SimpleNamespace(make_amp=Mock(return_value=(scaler, autocast)))
            with self.subTest(setting=setting), patch.object(torch, 'get_autocast_dtype', return_value=torch.float16):
                observed, context, report = qualification.make_precision(
                    trainer, device, 'trainer-default', None, True, training_init_scale=setting)
            expected = {'enabled': True}
            if setting is not None: expected['init_scale'] = setting
            trainer.make_amp.assert_called_once_with(device, **expected)
            self.assertIs(observed, scaler); self.assertIs(context, autocast)
            self.assertEqual(report, {'autocast_enabled': True, 'dtype': 'torch.float16',
                'scaler': state, 'selection': 'trainer-default', 'production_policy_adopted_by_this_script': False})

    def test_trainer_precision_rejects_cli_or_observed_source_scale_mismatch(self):
        device = torch.device('cuda:0')
        for initial, observed, enabled in ((1., 32768., True), (None, 65536., True), (None, 32768., False)):
            scaler = SimpleNamespace(state_dict=lambda: {'scale': observed})
            trainer = SimpleNamespace(make_amp=Mock(return_value=(scaler, contextlib.nullcontext)))
            with self.subTest(initial=initial, observed=observed, enabled=enabled), \
                    patch.object(torch, 'get_autocast_dtype', return_value=torch.float16), self.assertRaises(ValueError):
                qualification.make_precision(trainer, device, 'trainer-default', initial, enabled,
                                             training_init_scale=32768.)
            if initial is not None or not enabled: trainer.make_amp.assert_not_called()

    def test_invalid_bound_scale_rejected_before_hardware_or_model_work(self):
        for invalid in (0, -1, float('nan'), float('inf'), -float('inf'), True, '32768'):
            values = tiny_settings()
            for setting in values.values(): setting['amp_init_scale'] = invalid
            with self.subTest(scale=invalid), self.assertRaisesRegex(ValueError, 'finite and positive'):
                qualification.validate_settings(values, 'tiny')
            trainer = SimpleNamespace(make_amp=Mock())
            with self.subTest(direct_scale=invalid), self.assertRaisesRegex(ValueError, 'finite and positive'):
                qualification.make_precision(trainer, torch.device('cuda:0'), 'trainer-default', None, True,
                                             training_init_scale=invalid)
            trainer.make_amp.assert_not_called()
        values = tiny_settings()
        values['F2']['amp_init_scale'] = 32768.
        values['F3']['amp_init_scale'] = 65536.
        with self.assertRaisesRegex(ValueError, 'differ beyond'):
            qualification.validate_settings(values, 'tiny')

    def test_mode_uses_training_setting_for_precision_before_model_construction(self):
        settings = tiny_settings()['F2']; settings.update(amp=True, amp_init_scale=32768.)
        trainer = SimpleNamespace(set_seed=Mock(), configure_cuda_performance=Mock())
        encoder = SimpleNamespace(column_policy={}, type_policy={}, output_dim=5)
        options = SimpleNamespace(device='cpu', batches=3, precision='trainer-default', amp_init_scale=None)
        with patch.object(qualification, 'fabricated_batches', return_value=([], encoder, {})), \
                patch.object(qualification, 'make_precision', return_value=(None, contextlib.nullcontext, {})) as precision, \
                patch.object(qualification, 'build_model', side_effect=RuntimeError('stop before model')):
            with self.assertRaisesRegex(RuntimeError, 'stop before model'):
                qualification.qualify_mode(trainer, settings, options, {})
        precision.assert_called_once_with(trainer, torch.device('cpu'), 'trainer-default', None, True,
                                          training_init_scale=32768.)

    def test_explicit_fp16_scale_remains_separate_diagnostic(self):
        trainer = SimpleNamespace(make_amp=Mock())
        scaler = SimpleNamespace(state_dict=lambda: {'scale': 1.})
        with patch.object(torch.amp, 'GradScaler', return_value=scaler) as constructor:
            _, _, report = qualification.make_precision(trainer, torch.device('cuda:0'), 'fp16', 1., True,
                                                        training_init_scale=32768.)
        constructor.assert_called_once_with('cuda', init_scale=1.)
        trainer.make_amp.assert_not_called()
        self.assertEqual(report['selection'], 'fp16')
        self.assertEqual(report['scaler']['scale'], 1.)
        self.assertFalse(report['production_policy_adopted_by_this_script'])

    def test_failure_record_survives_before_any_model_construction(self):
        options = SimpleNamespace(profile='tiny', device='cpu', precision='fp32', amp_init_scale=None,
                                  batches=2, container_image='fabricated')
        result = qualification.run(tiny_settings(), options)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('partial batch', result['error'])
        self.assertEqual(result['modes'], {})
        json.dumps(result, allow_nan=False)

    def test_nonfinite_observations_are_preserved_in_legal_json(self):
        safe = qualification.json_safe_evidence({'gradient_norm': float('inf'), 'loss': float('nan')})
        self.assertEqual(safe['gradient_norm'], {'nonfinite_numeric_value': 'inf'})
        self.assertEqual(safe['loss'], {'nonfinite_numeric_value': 'nan'})
        json.dumps(safe, allow_nan=False)

    def test_both_prescribed_modes_retain_failure_records_without_retry(self):
        options = SimpleNamespace(profile='tiny', device='cpu', precision='fp32', amp_init_scale=None,
                                  batches=3, container_image='fabricated')
        with patch.object(qualification, 'qualify_mode', side_effect=RuntimeError('synthetic phase failure')) as attempt:
            result = qualification.run(tiny_settings(), options)
        self.assertEqual(attempt.call_count, 2)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(set(result['modes']), {'F2', 'F3'})
        self.assertTrue(all(item['error'] == 'synthetic phase failure' for item in result['modes'].values()))


if __name__ == '__main__':
    unittest.main()
