"""CPU fabricated evidence for diagnostic isolation, ordering and matched replay."""
import contextlib
import copy
import importlib.util
import io
import inspect
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from trifusesurv2.multimodal_survival import train

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('diagnose_production_gradients', ROOT / 'scripts/diagnose_production_gradients.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)
retained, qualification = diagnostic.utilities()


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[.1]]))
        self.intentionally_inactive = torch.nn.Parameter(torch.tensor([1.]))

    def forward(self, x):
        return {'OS': x @ self.weight}


def tiny_trainer(*, inject_failure=True):
    """Use the actual update helper and actual CPU GradScaler for replay tests."""
    trainer = SimpleNamespace()
    # The actual helper resolves its clip global on the real trainer module;
    # this adapter forwards that one observation surface without changing it.
    trainer.clip_grad_norm_ = train.clip_grad_norm_

    def update(model, optimizer, scaler=None):
        with patch.object(train, 'clip_grad_norm_', trainer.clip_grad_norm_):
            return train.perform_optimizer_update(model, optimizer, scaler)
    trainer.perform_optimizer_update = update
    trainer._unpack_surv_batch = lambda batch: batch
    trainer._endpoint_survival_losses = lambda logits, **kw: ({'OS': (logits['OS'] - kw['t_all']).square().mean()}, {'OS': 1})
    trainer._model_forward_eval = lambda model, payload, device, autocast: model(payload['x'])
    trainer.make_amp = lambda device, enabled, init_scale: (torch.amp.GradScaler('cpu', init_scale=init_scale), contextlib.nullcontext)

    def one_epoch(model, loader, optimizer, scaler, device, **kwargs):
        optimizer.zero_grad(set_to_none=True)
        rows = []
        model.train()
        for index, batch in enumerate(loader, 1):
            payload = trainer._unpack_surv_batch(batch)
            with kwargs['autocast_ctx']():
                losses, _ = trainer._endpoint_survival_losses(model(payload['x']), t_all=payload['t_all'], e_all=payload['e_all'])
                loss = losses['OS'] / kwargs['grad_accumulation_steps']
            if inject_failure and payload.get('overflow', False) and scaler is not None and scaler.get_scale() >= 32768:
                loss.register_hook(lambda value: value * float('inf'))
            (scaler.scale(loss) if scaler else loss).backward()
            if index % kwargs['grad_accumulation_steps'] == 0 or index == len(loader):
                rows.append(trainer.perform_optimizer_update(model, optimizer, scaler))
                optimizer.zero_grad(set_to_none=True)
        return {'training_batches': len(loader), 'optimizer_step_attempts': len(rows),
                'optimizer_steps_succeeded': sum(x['optimizer_steps_succeeded'] for x in rows),
                'optimizer_steps_skipped': sum(x['optimizer_steps_skipped'] for x in rows)}
    trainer.train_one_epoch = one_epoch
    return trainer


def tiny_qualifier():
    return SimpleNamespace(active_names=lambda model: ({'shared_backbone': ['weight']}, ['weight']))


def write_cpu_fixture(directory, *, inject_failure=True):
    """Persist an actual tiny CPU producer report for portable lifecycle tests."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scaler = torch.amp.GradScaler('cpu', init_scale=32768)
    batches = [{'x': torch.ones(1, 1), 't_all': torch.ones(1, 1), 'e_all': torch.ones(1, 1),
                'pid': ['fabricated-not-exported'], 'overflow': i == 8} for i in range(17)]
    kwargs = dict(epoch=1, grad_accumulation_steps=8, autocast_ctx=contextlib.nullcontext,
                  time_bin_width_days=180., num_time_bins=2)
    with contextlib.redirect_stdout(io.StringIO()):
        result = diagnostic.run_first_epoch(tiny_trainer(inject_failure=inject_failure), model, batches, optimizer, scaler,
                                            torch.device('cpu'), kwargs, directory, 2, retained, tiny_qualifier())
    result.update(schema_version=1, status='completed_diagnostic', source={'commit_sha': 'fabricated_CPU_fixture'},
                  plan_sha256='fabricated_CPU_fixture', runtime={'device': 'cpu', 'torch_version': torch.__version__},
                  scope='Fabricated CPU protocol fixture; no H100/FP16-production or real cohort claim',
                  heldout_payload_reads=0, test_evaluation='not_run')
    diagnostic.validate_report(result, require_production=False)
    diagnostic.write_json(directory / 'DIAGNOSTIC.json', result)
    return result


class DiagnosticTests(unittest.TestCase):
    def test_production_runtime_requires_exact_historical_torch_build(self):
        diagnostic.validate_production_runtime({'device': 'cuda:0', 'torch_version': '2.9.1+cu128'})
        for version in ('2.9.10+cu128', '2.9.1+cu126', '2.9.1'):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'PyTorch version'):
                diagnostic.validate_production_runtime({'device': 'cuda:0', 'torch_version': version})

    def test_actual_train_one_epoch_parity_and_final_partial_divisor(self):
        class LegacyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[.1, -.1]]))

            def forward(self, x, clin, rad, **kwargs):
                return {'OS': x @ self.weight}, None, None

        with patch('sys.argv', ['train', '--meta_csv', '/fabricated/never-read.csv']):
            settings = vars(train.parse_args())
        kwargs = {key: settings[key] for key, parameter in inspect.signature(train.train_one_epoch).parameters.items()
                  if key in settings and parameter.kind == inspect.Parameter.KEYWORD_ONLY}
        kwargs.update(fold=0, epoch=1, primary_endpoint='OS', num_time_bins=2,
                      time_bin_width_days=180., teacher_force_alpha=0., survival_loss_weight=1., ema=None,
                      autocast_ctx=contextlib.nullcontext, grad_accumulation_steps=2, nonfinite_check_every_batches=1,
                      loc_loss_pt_lambda=0., loc_loss_ln_lambda=0., loc_presence_lambda=0., mask_support_lambda=0.,
                      mask_focus_lambda=0., log_every_batches=0)
        batches = [{'x': torch.tensor([[.5 + index / 10]]), 't': torch.tensor([120.]),
                    'e': torch.tensor([1.]), 'pid': ['fabricated-train-only']} for index in range(3)]
        plain = LegacyModel(); observed = copy.deepcopy(plain)
        first = torch.optim.AdamW(plain.parameters(), lr=.01)
        second = torch.optim.AdamW(observed.parameters(), lr=.01)
        with contextlib.redirect_stdout(io.StringIO()):
            baseline = train.train_one_epoch(plain, batches, first, None, torch.device('cpu'), **kwargs)
            record = {}
            with diagnostic.Observer(train, observed, second, None, torch.device('cpu'), retained, tiny_qualifier(), record) as observer:
                observer.divisor = 2
                result = train.train_one_epoch(observed, batches, second, None, torch.device('cpu'), **kwargs)
        self.assertTrue(torch.equal(plain.weight, observed.weight))
        self.assertEqual(baseline['optimizer_steps_succeeded'], result['optimizer_steps_succeeded'])
        for key, value in baseline.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                self.assertEqual(value, result[key], key)
        self.assertEqual([row['actual_accumulation_count'] for row in record['updates']], [2, 1])
        self.assertEqual([row['actual_loss_divisor'] for row in record['updates']], [2, 2])
        self.assertEqual([row['actual_loss_divisor'] for row in record['batches']], [2, 2, 2])

    def test_preclip_nonfinite_is_retained_and_three_cells_finish(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = write_cpu_fixture(temporary)
            self.assertEqual(report['inference']['observed_failure_update'], 2)
            self.assertEqual(report['selected_window']['first_batch'], 9)
            self.assertEqual(report['selected_window']['end_batch'], 16)
            self.assertTrue(report['selected_window']['following_payload_present'])
            self.assertEqual(report['cells']['fp16_scale32768']['status'], 'failed_numerical')
            self.assertEqual(report['cells']['fp16_scale16384']['status'], 'passed_numerical')
            self.assertEqual(report['cells']['fp32']['status'], 'passed_numerical')
            self.assertTrue(report['original_window_replay_comparison']['all_checks_agree'])
            self.assertTrue(report['candidate_scale16384_supported_on_captured_window'])
            bad = report['cells']['fp16_scale32768']['updates'][0]
            self.assertEqual(bad['after_unscale_before_clip']['nonfinite_parameters'], ['weight'])
            gradient = bad['after_unscale_before_clip']['parameters']['weight']
            self.assertGreater(gradient['positive_inf_elements'] + gradient['negative_inf_elements'], 0)
            self.assertEqual(bad['actual_optimizer_calls_after'] - bad['actual_optimizer_calls_before'], 0)
            self.assertTrue(report['cells']['fp16_scale16384']['post_update_state']['all_tensors_finite'])
            text = (Path(temporary) / 'DIAGNOSTIC.json').read_text()
            self.assertNotIn('fabricated-not-exported', text)
            self.assertIn('gradient_comparison_to_FP32', report['cells']['fp16_scale16384'])
            with self.assertRaises(ValueError):
                diagnostic.validate_report(report, require_production=True)

    def test_nonreproduction_and_identity_tamper_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = write_cpu_fixture(temporary, inject_failure=False)
            self.assertIsNone(report['inference']['observed_failure_update'])
            self.assertFalse(report['inference']['reproduced_at_inferred_location'])
            self.assertEqual(report['inference']['selection_reason'], 'no_event_reproduced_in_first_epoch_use_inferred_window')
            broken = copy.deepcopy(report)
            broken['cells']['fp32']['initial_identity']['model_state_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'identities'):
                diagnostic.validate_report(broken, require_production=False)

    def test_required_missing_is_distinct_from_inactive_and_nonfinite_norm(self):
        model = Model()
        model.weight.grad = torch.full_like(model.weight, 1e30)
        observation = diagnostic.gradient_summary(model, {'active': ['weight']}, ['weight'], retained, detailed=True)
        self.assertEqual(observation['missing_required_parameters'], [])
        self.assertEqual(observation['missing_other_parameters'], ['intentionally_inactive'])
        self.assertTrue(observation['all_observed_elements_finite'])
        self.assertTrue(math.isfinite(observation['robust_finite_elements_l2_norm_float64']))
        model.weight.grad = None
        observation = diagnostic.gradient_summary(model, {}, ['weight'], retained)
        self.assertEqual(observation['missing_required_parameters'], ['weight'])

    def test_finite_elements_with_overflowing_native_norm_are_distinct(self):
        model = Model()
        model.weight = torch.nn.Parameter(torch.ones(1, 2))
        model.weight.grad = torch.full_like(model.weight, 1e30)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        record = {}
        with diagnostic.Observer(train, model, optimizer, None, torch.device('cpu'), retained, tiny_qualifier(), record) as observer:
            observer.divisor = 1
            observer.microbatch = 1
            train.perform_optimizer_update(model, optimizer)
        update = record['updates'][0]
        self.assertEqual(update['after_unscale_before_clip']['nonfinite_parameters'], [])
        self.assertTrue(math.isfinite(update['after_unscale_before_clip']['robust_finite_elements_l2_norm_float64']))
        self.assertTrue(update['aggregate_norm_nonfinite'])
        self.assertTrue(update['disqualifying_numerical_event'])

    def test_failed_cell_runtime_exception_does_not_abort_remaining_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = tiny_trainer(inject_failure=False)
            model = Model(); optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            scaler = torch.amp.GradScaler('cpu', init_scale=32768)
            state = diagnostic.snapshot_state(model, optimizer, scaler, retained, torch.device('cpu'))
            saved = {'state': state, 'batches': [{'x': torch.ones(1, 1), 't_all': torch.ones(1, 1), 'e_all': torch.ones(1, 1)}] * 8,
                     'following_batch': None}
            original = trainer.make_amp
            trainer.make_amp = lambda device, enabled, init_scale: (_ for _ in ()).throw(RuntimeError('cell failed')) if init_scale == 32768 else original(device, enabled, init_scale)
            records = diagnostic.replay_cells(trainer, model, optimizer, saved,
                                               dict(epoch=1, grad_accumulation_steps=8, autocast_ctx=contextlib.nullcontext,
                                                    time_bin_width_days=180., num_time_bins=2),
                                               torch.device('cpu'), Path(temporary), retained, tiny_qualifier())
            self.assertEqual(records['fp16_scale32768']['status'], 'runtime_error')
            self.assertEqual(records['fp16_scale16384']['status'], 'passed_numerical')
            self.assertEqual(records['fp32']['status'], 'passed_numerical')

    def test_late_original_cell_runtime_error_cannot_support_candidate(self):
        following = diagnostic.following_nll
        calls = [0]
        def fail_first(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 1:
                raise RuntimeError('fabricated following-forward error')
            return following(*args, **kwargs)
        with tempfile.TemporaryDirectory() as temporary, patch.object(diagnostic, 'following_nll', fail_first):
            report = write_cpu_fixture(temporary)
            self.assertTrue(report['original_window_replay_comparison']['original_event_reproduced_by_replay_A'])
            self.assertEqual(report['cells']['fp16_scale32768']['status'], 'runtime_error')
            self.assertEqual(report['cells']['fp16_scale16384']['status'], 'passed_numerical')
            self.assertFalse(report['candidate_scale16384_supported_on_captured_window'])

    def test_observer_restores_globals_and_does_not_change_actual_trainer_update(self):
        torch.manual_seed(42)
        baseline = Model(); observed = copy.deepcopy(baseline)
        before_rng = torch.get_rng_state().clone()
        optimizer = torch.optim.AdamW(baseline.parameters(), lr=.01)
        baseline(torch.ones(1, 1))['OS'].square().sum().backward()
        plain = train.perform_optimizer_update(baseline, optimizer)
        torch.set_rng_state(before_rng)
        observed_optimizer = torch.optim.AdamW(observed.parameters(), lr=.01)
        original_backward, original_clip, original_update = torch.Tensor.backward, train.clip_grad_norm_, train.perform_optimizer_update
        record = {}
        with diagnostic.Observer(train, observed, observed_optimizer, None, torch.device('cpu'), retained, tiny_qualifier(), record) as observer:
            observer.divisor = 1
            observer.microbatch = 1
            record['batches'].append({'microbatch_index': 1})
            observed(torch.ones(1, 1))['OS'].square().sum().backward()
            result = train.perform_optimizer_update(observed, observed_optimizer)
        self.assertEqual(plain, result)
        self.assertTrue(torch.equal(baseline.weight, observed.weight))
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertIs(torch.Tensor.backward, original_backward)
        self.assertIs(train.clip_grad_norm_, original_clip)
        self.assertIs(train.perform_optimizer_update, original_update)

    def test_observer_cleanup_on_exception(self):
        trainer = tiny_trainer()
        model = Model(); optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        original = trainer.perform_optimizer_update
        backward = torch.Tensor.backward
        with self.assertRaises(RuntimeError):
            with diagnostic.Observer(trainer, model, optimizer, None, torch.device('cpu'), retained, tiny_qualifier(), {}):
                raise RuntimeError('fabricated')
        self.assertIs(trainer.perform_optimizer_update, original)
        self.assertIs(torch.Tensor.backward, backward)
        self.assertFalse(optimizer._optimizer_step_post_hooks)

    def test_nonfinite_json_is_explicit_and_no_patient_payload_is_serialized_to_report(self):
        value = diagnostic.safe({'loss': float('nan'), 'norm': float('inf')})
        self.assertEqual(value, {'loss': {'nonfinite': 'nan'}, 'norm': {'nonfinite': '+inf'}})
        json.dumps(value, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
