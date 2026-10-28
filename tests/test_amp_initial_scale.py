"""CPU configuration/serialization checks; no CUDA qualification is claimed."""
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from trifusesurv2.multimodal_survival import train
from trifusesurv2.utils.experiment import canonical_experiment_spec


class AmpInitialScaleTests(unittest.TestCase):
    def parse(self, *extra):
        with patch('sys.argv', ['train', '--meta_csv', 'unused.csv', *extra]):
            return train.parse_args()

    def test_cli_default_and_explicit_scale(self):
        self.assertIsNone(self.parse().amp_init_scale)
        self.assertEqual(self.parse('--amp', '--amp_init_scale', '32768').amp_init_scale, 32768.0)

    def test_bad_cli_scale_fails_before_seed_or_device_initialization(self):
        for value in ('0', '-1', 'nan', 'inf', '-inf'):
            with self.subTest(value=value), patch('sys.argv',
                    ['train', '--meta_csv', 'unused.csv', '--amp', '--amp_init_scale=' + value]), \
                    patch.object(train, 'set_seed') as seed, patch.object(train, 'parse_device') as device:
                with self.assertRaisesRegex(ValueError, 'finite positive'):
                    train.main()
                seed.assert_not_called()
                device.assert_not_called()

    def test_nondefault_scale_requires_cuda_amp(self):
        for options in (('--amp_init_scale', '32768'), ('--amp', '--device', 'cpu', '--amp_init_scale', '32768')):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'enabled CUDA AMP'):
                self.parse(*options)
        for device, enabled in ((SimpleNamespace(type='cpu'), True), (SimpleNamespace(type='cuda'), False)):
            with self.subTest(device=device.type, enabled=enabled), self.assertRaisesRegex(ValueError, 'enabled CUDA AMP'):
                train.make_amp(device, enabled=enabled, init_scale=32768.0)

    def test_none_preserves_actual_factory_defaults_and_override_is_explicit(self):
        for scale, expected in ((None, {}), (32768.0, {'init_scale': 32768.0})):
            with self.subTest(scale=scale), patch.object(train.torch.amp, 'GradScaler') as factory:
                scaler, _ = train.make_amp(SimpleNamespace(type='cuda'), init_scale=scale)
                factory.assert_called_once_with('cuda', enabled=True, **expected)
                self.assertIs(scaler, factory.return_value)

    def test_legacy_factory_fallback_preserves_the_same_policy(self):
        for scale, expected in ((None, {}), (32768.0, {'init_scale': 32768.0})):
            with self.subTest(scale=scale), patch.object(train.torch.amp, 'GradScaler', side_effect=AttributeError('old API')), \
                    patch.object(train.torch.cuda.amp, 'GradScaler') as factory:
                scaler, _ = train.make_amp(SimpleNamespace(type='cuda'), enabled=True, init_scale=scale)
                factory.assert_called_once_with(enabled=True, **expected)
                self.assertIs(scaler, factory.return_value)

    def test_invalid_direct_scale_never_reaches_any_cuda_factory(self):
        for value in (0, -1, float('nan'), float('inf'), -float('inf'), True, 'invalid'):
            with self.subTest(value=value), patch.object(train.torch.amp, 'GradScaler') as modern, \
                    patch.object(train.torch.cuda.amp, 'GradScaler') as legacy:
                with self.assertRaisesRegex(ValueError, 'finite positive'):
                    train.make_amp(SimpleNamespace(type='cuda'), init_scale=value)
                modern.assert_not_called()
                legacy.assert_not_called()

    def test_cpu_default_remains_disabled_without_scaler(self):
        with patch.object(train.torch.amp, 'GradScaler') as factory:
            scaler, autocast = train.make_amp(SimpleNamespace(type='cpu'), enabled=True)
        self.assertIsNone(scaler)
        factory.assert_not_called()
        with autocast():
            pass

    def test_main_passes_bound_scale_and_supports_old_namespace_without_field(self):
        class StopAtAmp(Exception):
            pass
        for value in (None, 32768.0, 'absent'):
            args = self.parse('--amp')
            if value == 'absent':
                del args.amp_init_scale
            else:
                args.amp_init_scale = value
            fake_device = SimpleNamespace(type='cuda')
            with self.subTest(value=value), patch.object(train, 'parse_args', return_value=args), \
                    patch.object(train, 'set_seed'), patch.object(train, 'parse_device', return_value=fake_device), \
                    patch.object(train, 'bind_cuda_device'), patch.object(train, 'configure_cuda_performance'), \
                    patch.object(train, 'make_amp', side_effect=StopAtAmp) as factory:
                with self.assertRaises(StopAtAmp):
                    train.main()
                factory.assert_called_once_with(fake_device, enabled=True,
                                                init_scale=None if value == 'absent' else value)

    def test_default_contract_shape_is_legacy_and_override_is_immutable(self):
        old_args = self.parse('--amp')
        del old_args.amp_init_scale
        old = canonical_experiment_spec(old_args)
        current = copy.deepcopy(old_args)
        current.amp_init_scale = None
        self.assertEqual(canonical_experiment_spec(current), old)
        self.assertIn('amp_init_scale', vars(current))
        current.amp_init_scale = 32768.0
        custom = canonical_experiment_spec(current)
        self.assertEqual(custom['settings']['amp_init_scale'], 32768.0)
        with self.assertRaisesRegex(RuntimeError, 'amp_init_scale'):
            train._validate_resume_model_contract({'experiment_spec': old}, {'experiment_spec': custom})
        current.amp_init_scale = 16384.0
        with self.assertRaisesRegex(RuntimeError, 'amp_init_scale'):
            train._validate_resume_model_contract({'experiment_spec': custom},
                                                  {'experiment_spec': canonical_experiment_spec(current)})

    def test_checkpoint_and_embedded_model_contract_retain_only_explicit_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            for value in ('absent', None, 32768.0):
                args = argparse.Namespace(amp=True)
                if value != 'absent':
                    args.amp_init_scale = value
                contract = {'experiment_spec': canonical_experiment_spec(args)}
                model = torch.nn.Linear(2, 1)
                optimizer = torch.optim.AdamW(model.parameters())
                path = Path(temporary) / ('checkpoint_' + str(value) + '.pt')
                train.save_checkpoint(path, 1, 2, model, args, optimizer, None, None, 1.0,
                                      None, None, encoders={'model_contract': contract})
                saved = torch.load(path, weights_only=False, map_location='cpu')
                for settings in (saved['args'], saved['encoders']['model_contract']['experiment_spec']['settings']):
                    with self.subTest(value=value, location='checkpoint_or_embedded_contract'):
                        if value == 32768.0:
                            self.assertEqual(settings['amp_init_scale'], 32768.0)
                        else:
                            self.assertNotIn('amp_init_scale', settings)

    def test_performance_log_reads_effective_separate_backend_values(self):
        matmul = SimpleNamespace(allow_tf32=True)
        cudnn = SimpleNamespace(allow_tf32=True, benchmark=False, deterministic=True)
        calls = []
        def set_precision(value):
            calls.append(value)
            matmul.allow_tf32 = True  # Simulate the qualified high-precision setter behavior.
        fake_torch = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul), cudnn=cudnn),
                                     set_float32_matmul_precision=set_precision,
                                     get_float32_matmul_precision=lambda: 'high')
        with patch.object(train, 'torch', fake_torch), patch.object(train, '_log') as log:
            train.configure_cuda_performance(SimpleNamespace(deterministic=True, allow_tf32=True,
                                                             matmul_precision='high'), SimpleNamespace(type='cuda'))
        self.assertEqual(calls, ['high'])
        self.assertTrue(matmul.allow_tf32)
        self.assertFalse(cudnn.allow_tf32)
        message = log.call_args.args[0]
        self.assertIn('cuda_matmul_allow_tf32=True', message)
        self.assertIn('cudnn_allow_tf32=False', message)
        self.assertIn('float32_matmul_precision=high', message)


if __name__ == '__main__':
    unittest.main()
