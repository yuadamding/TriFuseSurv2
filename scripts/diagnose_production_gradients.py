#!/usr/bin/env python3
"""Observe archived budget-C training and replay one real accumulation window.

This is a diagnostic, not a trainer or a policy acceptance gate. The archived
trainer's construction, loader, forward/loss/backward and optimizer helper run
unchanged. Temporary observations are removed on every exit. Tensor payloads
remain under the supplied private output directory; DIAGNOSTIC.json contains
hashes and numerical summaries only, never patient identifiers.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys

C_COMMIT = '1fed762a2457c49bbd334c5ac5f83c65dbdbcc16'
C_PLAN = '3fadfbda88ede8abf61063084d9d531d7344047d4786858205ba1643a17f90be'
CELL_NAMES = ('fp16_scale32768', 'fp16_scale16384', 'fp32')
PINNED_TORCH_BUILD = '2.9.1+cu128'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def reference(path):
    return {'path': str(path), 'sha256': sha(path), 'bytes': Path(path).stat().st_size}


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(safe(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': 'nan' if math.isnan(value) else ('+inf' if value > 0 else '-inf')}
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    return value


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def utilities():
    directory = Path(__file__).resolve().parent
    retained = module_from_path('_production_diagnostic_retained', directory / 'retained_cuda_diagnostic.py')
    qualifier = module_from_path('_production_diagnostic_qualifier', directory / 'qualify_swin_survival.py')
    return retained, qualifier


def gradient_summary(model, groups, required, retained, *, detailed=False):
    """Elementwise health and a scaled float64 norm, independent of clipping."""
    import torch
    missing, bad, rows, norms = [], [], {}, []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        grad = parameter.grad.detach()
        finite = bool(torch.isfinite(grad).all().item())
        if detailed or not finite or name in required:
            row = retained.tensor_summary(grad)
            rows[name] = row
            if detailed:
                norms.append(row['finite_l2_norm'])
        if not finite:
            bad.append(name)
    robust = math.hypot(*(n for n in norms if n is not None)) if detailed else None
    return {'gradient_tensor_count': sum(p.requires_grad for p in model.parameters()) - len(missing),
            'missing_required_parameters': sorted(set(missing) & set(required)),
            'missing_other_parameters': sorted(set(missing) - set(required)),
            'required_parameter_names': list(required), 'nonfinite_parameters': bad,
            'parameters': rows, 'all_observed_elements_finite': not bad,
            'robust_finite_elements_l2_norm_float64': robust if robust is None or math.isfinite(robust) else None,
            'robust_norm_overflow': robust is not None and not math.isfinite(robust),
            'required_groups': {key: [name for name in names if name in required] for key, names in groups.items()}}


def state_summary(model, optimizer):
    import torch
    tensors, bad = 0, []
    for name, value in model.state_dict().items():
        if torch.is_tensor(value):
            tensors += 1
            if not bool(torch.isfinite(value).all().item()):
                bad.append('model:' + name)
    for index, values in enumerate(optimizer.state.values()):
        for key, value in values.items():
            if torch.is_tensor(value):
                tensors += 1
                if not bool(torch.isfinite(value).all().item()):
                    bad.append('optimizer:{}:{}'.format(index, key))
    return {'tensors_checked': tensors, 'all_tensors_finite': not bad, 'nonfinite_names': bad}


def snapshot_state(model, optimizer, scaler, retained, device):
    require(all(p.grad is None for p in model.parameters()), 'Pre-window gradients must be None after zero_grad')
    return {'model': retained.cpu_copy(model.state_dict()),
            'optimizer': retained.cpu_copy(optimizer.state_dict()),
            'scaler': retained.cpu_copy(scaler.state_dict()) if scaler is not None else None,
            'rng': retained.capture_rng(cuda=device.type == 'cuda')}


def snapshot_identity(state, batches, retained):
    return {'model_state_sha256': retained.identity_digest(state['model']),
            'optimizer_state_sha256': retained.identity_digest(state['optimizer']),
            'rng_state_sha256': retained.identity_digest(state['rng']),
            'transformed_batches_sha256': retained.identity_digest(batches),
            'pre_window_scaler_state': state['scaler']}


class EpochCaptured(BaseException):
    """Return before the archived trainer's validation/checkpoint path."""


def record_loss_context(row, value, caller, divisor, retained):
    """Observe the actual archived train frame rather than infer a partial divisor."""
    try:
        values = caller.f_locals
        actual = values.get('accum_steps')
        row['actual_loss_divisor'] = int(actual) if isinstance(actual, int) else int(divisor)
        row['loss_divisor_source'] = 'archived_training_frame_accum_steps' if isinstance(actual, int) else 'fabricated_fixture_argument'
        row['actual_accumulation_count_before_backward'] = values.get('batches_since_step')
        row['accumulation_normalized_total_loss'] = retained.tensor_summary(value)
        original = values.get('loss')
        if original is not None and hasattr(original, 'detach') and original is not value:
            row['unscaled_total_loss_before_accumulation_divisor'] = retained.tensor_summary(original)
    finally:
        del caller


class Observer:
    def __init__(self, trainer, model, optimizer, scaler, device, retained, qualifier,
                 record, *, capture=False, target_update=27, out_dir=None, gradient_path=None):
        self.trainer, self.model, self.optimizer, self.scaler = trainer, model, optimizer, scaler
        self.device, self.retained, self.qualifier, self.record = device, retained, qualifier, record
        self.groups, self.required = qualifier.active_names(model)
        self.capture, self.target_update, self.out_dir = capture, target_update, out_dir
        self.gradient_path = gradient_path
        self.microbatch, self.attempts, self.window_batches = 0, 0, []
        self.rolling = None
        self.captures = {}
        self.first_failure = None
        self.active = None
        self.originals = {}
        self.scaler_originals = {}
        self.optimizer_calls = 0
        self.backward_calls = 0
        self.record.update(batches=[], updates=[], scaler_method_calls={'unscale_': 0, 'update': 0, 'load_state_dict': 0})

    def __enter__(self):
        import torch
        t = self.trainer
        self.originals = {key: getattr(t, key) for key in
                          ('_unpack_surv_batch', '_endpoint_survival_losses', 'clip_grad_norm_', 'perform_optimizer_update')}
        self.original_backward = torch.Tensor.backward
        self.handle = self.optimizer.register_step_post_hook(self._stepped)
        t._unpack_surv_batch = self.unpack
        t._endpoint_survival_losses = self.endpoint_losses
        t.clip_grad_norm_ = self.clip
        t.perform_optimizer_update = self.update
        observer = self

        def backward(tensor, *args, **kwargs):
            row = observer.record['batches'][-1]
            if 'accumulation_normalized_total_loss' not in row:
                record_loss_context(row, tensor, inspect.currentframe().f_back, observer.divisor, observer.retained)
            result = observer.original_backward(tensor, *args, **kwargs)
            observer.backward_calls += 1
            observation = gradient_summary(observer.model, observer.groups, observer.required, observer.retained)
            observer.record['batches'][-1]['scaled_gradients_after_backward'] = observation
            return result
        torch.Tensor.backward = backward
        if self.scaler is not None:
            for key in ('unscale_', 'update', 'load_state_dict'):
                original = getattr(self.scaler, key)
                self.scaler_originals[key] = original

                def counted(*args, _key=key, _original=original, **kwargs):
                    self.record['scaler_method_calls'][_key] += 1
                    return _original(*args, **kwargs)
                setattr(self.scaler, key, counted)
            self.scaler_originals['scale'] = self.scaler.scale

            def scale(loss, *args, **kwargs):
                row = self.record['batches'][-1]
                record_loss_context(row, loss, inspect.currentframe().f_back, self.divisor, self.retained)
                scaled = self.scaler_originals['scale'](loss, *args, **kwargs)
                row['scaled_loss'] = self.retained.tensor_summary(scaled)
                return scaled
            self.scaler.scale = scale
        return self

    def __exit__(self, *unused):
        import torch
        for name, value in self.originals.items():
            setattr(self.trainer, name, value)
        torch.Tensor.backward = self.original_backward
        for name, value in self.scaler_originals.items():
            setattr(self.scaler, name, value)
        self.handle.remove()

    def _stepped(self, *unused):
        self.optimizer_calls += 1

    def unpack(self, batch):
        if self.capture:
            for saved in self.captures.values():
                if saved.get('following_batch') is None and self.microbatch == saved['end_batch']:
                    saved['following_batch'] = self.retained.cpu_copy(batch)
            if self.microbatch % self.divisor == 0:
                self.rolling = snapshot_state(self.model, self.optimizer, self.scaler, self.retained, self.device)
                self.window_batches = []
            self.window_batches.append(self.retained.cpu_copy(batch))
        self.microbatch += 1
        self.record['batches'].append({'microbatch_index': self.microbatch, 'mode': 'F2', 'seed': 42, 'epoch': 1})
        return self.originals['_unpack_surv_batch'](batch)

    def endpoint_losses(self, logits, **kwargs):
        losses, counts = self.originals['_endpoint_survival_losses'](logits, **kwargs)
        row = self.record['batches'][-1]
        row['endpoints'] = {name: {'logits': self.retained.tensor_summary(logits[name]),
                                  'survival_nll': self.retained.tensor_summary(loss),
                                  'valid_outcome_count': int(counts[name])}
                            for name, loss in losses.items()}
        return losses, counts

    def clip(self, parameters, *args, **kwargs):
        import torch
        self.active['after_unscale_before_clip'] = gradient_summary(
            self.model, self.groups, self.required, self.retained, detailed=True)
        if self.gradient_path is not None:
            gradients = {name: p.grad.detach().cpu().clone() for name, p in self.model.named_parameters() if p.grad is not None}
            with self.gradient_path.open('xb') as stream:
                torch.save(gradients, stream)
            self.active['unscaled_preclip_gradients'] = reference(self.gradient_path)
            del gradients
        result = self.originals['clip_grad_norm_'](parameters, *args, **kwargs)
        norm = float(result.item()) if torch.is_tensor(result) else float(result)
        self.active['actual_clipping_norm'] = norm
        self.active['aggregate_norm_nonfinite'] = not math.isfinite(norm)
        return result

    def update(self, *args, **kwargs):
        self.attempts += 1
        start = self.microbatch - ((self.microbatch - 1) % self.divisor)
        row = {'attempted_update_index': self.attempts, 'epoch': 1, 'mode': 'F2', 'seed': 42,
               'first_microbatch': start, 'last_microbatch': self.microbatch,
               'actual_accumulation_count': self.microbatch - start + 1, 'actual_loss_divisor': self.divisor,
               'scaler_before': copy.deepcopy(self.scaler.state_dict()) if self.scaler else None,
               'actual_optimizer_calls_before': self.optimizer_calls,
               'scaled_gradients_before_unscale': gradient_summary(self.model, self.groups, self.required, self.retained)}
        self.active = row
        self.record['updates'].append(row)
        result = self.originals['perform_optimizer_update'](*args, **kwargs)
        row.update(result)
        row.update(actual_optimizer_calls_after=self.optimizer_calls,
                   scaler_after=copy.deepcopy(self.scaler.state_dict()) if self.scaler else None)
        preclip = row['after_unscale_before_clip']
        failed = (result['optimizer_steps_skipped'] != 0 or bool(preclip['nonfinite_parameters'])
                  or bool(preclip['missing_required_parameters']) or row['aggregate_norm_nonfinite'])
        row['disqualifying_numerical_event'] = failed
        if self.capture and (self.attempts == self.target_update or failed and self.first_failure is None):
            saved = {'update_index': self.attempts, 'state': self.rolling, 'batches': self.window_batches,
                     'first_batch': start, 'end_batch': self.microbatch, 'following_batch': None}
            self.captures[self.attempts] = saved
        if failed and self.first_failure is None:
            self.first_failure = self.attempts
        self.rolling = None
        self.window_batches = []
        return result


def restore_window(model, optimizer, state, retained, device):
    model.load_state_dict(state['model'], strict=True)
    # CPU scalar Adam steps may otherwise alias retained snapshot tensors even
    # when the parameter moments are moved to CUDA by load_state_dict.
    optimizer.load_state_dict(retained.cpu_copy(state['optimizer']))
    optimizer.zero_grad(set_to_none=True)
    model.train()
    retained.restore_rng(state['rng'])
    actual = {'model_state_sha256': retained.identity_digest(model.state_dict()),
              'optimizer_state_sha256': retained.identity_digest(optimizer.state_dict()),
              'rng_state_sha256': retained.identity_digest(retained.capture_rng(cuda=device.type == 'cuda'))}
    for key, obj in [('model_state_sha256', state['model']), ('optimizer_state_sha256', state['optimizer']),
                     ('rng_state_sha256', state['rng'])]:
        require(actual[key] == retained.identity_digest(obj), 'Replay state differs: ' + key)
    return actual


def following_nll(trainer, model, batch, device, autocast, kwargs, retained):
    import torch
    model.eval()
    with torch.no_grad():
        payload = trainer._unpack_surv_batch(batch)
        output = trainer._model_forward_eval(model, payload, device, autocast)
        losses, counts = trainer._endpoint_survival_losses(
            output, t_all=payload['t_all'].to(device), e_all=payload['e_all'].to(device),
            time_bin_width_days=kwargs['time_bin_width_days'], num_time_bins=kwargs['num_time_bins'])
    return {name: {'nll': retained.tensor_summary(loss), 'valid_outcome_count': int(counts[name])}
            for name, loss in losses.items()}


def compare_gradients(candidate_path, reference_path, retained):
    import torch
    candidate = torch.load(candidate_path, map_location='cpu', weights_only=False)
    fp32 = torch.load(reference_path, map_location='cpu', weights_only=False)
    rows = {}
    for name in sorted(set(candidate) | set(fp32)):
        if name not in candidate or name not in fp32:
            rows[name] = {'missing_in': 'candidate' if name not in candidate else 'fp32'}
            continue
        a, b = candidate[name].double(), fp32[name].double()
        finite = torch.isfinite(a) & torch.isfinite(b)
        af, bf = a[finite], b[finite]
        diff, ref = retained.tensor_summary(af - bf), retained.tensor_summary(bf)
        denominator = ref['finite_l2_norm']
        rows[name] = {'joint_finite_elements': int(finite.sum().item()), 'elements': a.numel(),
                      'candidate_zero_where_fp32_nonzero': int(((af == 0) & (bf != 0)).sum().item()),
                      'fp32_zero_where_candidate_nonzero': int(((bf == 0) & (af != 0)).sum().item()),
                      'difference_max_abs': diff['finite_max_abs'],
                      'relative_l2_on_joint_finite': diff['finite_l2_norm'] / denominator if denominator else None}
    return {'scope': 'unscaled gradients before clipping; no automatic material-error tolerance', 'parameters': rows}


def compare_original_window(epoch, selected, cell):
    """Exact equality of observed arithmetic; NaN/Inf are explicit categories."""
    original = epoch['updates'][selected['update_index'] - 1]
    expected_batches = epoch['batches'][selected['first_batch'] - 1:selected['end_batch']]
    observed_batches = cell.get('batches', [])
    fields = ('endpoints', 'accumulation_normalized_total_loss', 'scaled_loss',
              'unscaled_total_loss_before_accumulation_divisor', 'actual_loss_divisor',
              'scaled_gradients_after_backward')
    checks = {'same_microbatch_count': len(expected_batches) == len(observed_batches)}
    for key in fields:
        checks[key] = (checks['same_microbatch_count'] and
                       safe([row.get(key) for row in expected_batches]) == safe([row.get(key) for row in observed_batches]))
    updates = cell.get('updates', [])
    checks['one_replay_update'] = len(updates) == 1
    for key in ('after_unscale_before_clip', 'actual_clipping_norm', 'aggregate_norm_nonfinite',
                'optimizer_steps_succeeded', 'optimizer_steps_skipped', 'scaler_before', 'scaler_after'):
        checks[key] = bool(updates) and safe(original.get(key)) == safe(updates[0].get(key))
    original_delta = original['actual_optimizer_calls_after'] - original['actual_optimizer_calls_before']
    checks['actual_optimizer_call_delta'] = bool(updates) and original_delta == updates[0]['actual_optimizer_calls_after'] - updates[0]['actual_optimizer_calls_before']
    agreement = all(checks.values())
    return {'comparison': 'exact recorded tensor/loss/gradient summaries and scaler/optimizer observations; no added tolerance',
            'checks': checks, 'all_checks_agree': agreement,
            'original_disqualifying_event': original['disqualifying_numerical_event'],
            'original_event_reproduced_by_replay_A': agreement and original['disqualifying_numerical_event'],
            'only_expected_control_differences': ['local microbatch numbering', 'diagnostic logging', 'loader length equals captured window'],
            'mismatch_disposition': None if agreement else 'unresolved_replay_difference_candidate_advancement_blocked'}


def replay_cells(trainer, model, optimizer, saved, kwargs, device, out_dir, retained, qualifier):
    import torch
    records = {}
    expected = snapshot_identity(saved['state'], saved['batches'], retained)
    for name, init_scale in zip(CELL_NAMES, [32768., 16384., None]):
        record = {'cell': name, 'status': 'runtime_error', 'initial_identity': None,
                  'precision': 'autocast_disabled_FP32' if init_scale is None else 'FP16',
                  'initial_scale': init_scale, 'backend': retained.backend_settings(),
                  'strict_IEEE_FP32_reference': False, 'updates': [], 'attempted': True}
        records[name] = record
        try:
            restored = restore_window(model, optimizer, saved['state'], retained, device)
            restored['transformed_batches_sha256'] = retained.identity_digest(saved['batches'])
            require(all(restored[k] == expected[k] for k in restored), 'Cell input/state identity differs')
            record['initial_identity'] = restored
            if init_scale is None:
                scaler, autocast = None, lambda: contextlib.nullcontext()
            else:
                scaler, autocast = trainer.make_amp(device, enabled=True, init_scale=init_scale)
                state = copy.deepcopy(saved['state']['scaler'])
                require(state is not None, 'Original production window must have a scaler')
                state['scale'] = init_scale
                scaler.load_state_dict(state)
            record['scaler_initial_state'] = scaler.state_dict() if scaler is not None else None
            gradient_path = out_dir / (name + '-unscaled-preclip-gradients.pt')
            with Observer(trainer, model, optimizer, scaler, device, retained, qualifier, record,
                          gradient_path=gradient_path) as observer:
                observer.divisor = int(kwargs['grad_accumulation_steps'])
                replay_kwargs = dict(kwargs, autocast_ctx=autocast)
                if 'roi_focus_live_csv' in replay_kwargs:
                    replay_kwargs['roi_focus_live_csv'] = None
                record['stats'] = trainer.train_one_epoch(model, saved['batches'], optimizer, scaler, device, **replay_kwargs)
            require(len(record['updates']) == 1, 'Replay must attempt exactly one optimizer update')
            record['post_update_state'] = state_summary(model, optimizer)
            follow = saved.get('following_batch')
            record['following_NLL_scope'] = ('next_actual_transformed_training_microbatch' if follow is not None
                                             else 'saved_first_window_training_microbatch_no_next_batch_in_epoch')
            record['following_survival_NLL'] = following_nll(trainer, model, follow if follow is not None else saved['batches'][0],
                                                           device, autocast, kwargs, retained)
            updates_ok = all(not row['disqualifying_numerical_event'] and
                             row['actual_optimizer_calls_after'] - row['actual_optimizer_calls_before'] == 1
                             for row in record['updates'])
            following_ok = all(row['nll']['all_finite'] for row in record['following_survival_NLL'].values())
            record['status'] = 'passed_numerical' if updates_ok and following_ok and record['post_update_state']['all_tensors_finite'] else 'failed_numerical'
        except Exception as error:
            record['error_type'] = type(error).__name__
            record['failure_phase'] = 'replay_or_following_NLL'
            # Error strings may contain patient previews from the archived trainer.
            record['error_detail'] = 'See project-private worker log; patient-bearing exception text is omitted.'
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        write_json(out_dir / (name + '.json'), record)
        gc.collect()
    fp32_path = out_dir / 'fp32-unscaled-preclip-gradients.pt'
    if fp32_path.is_file():
        for name in CELL_NAMES[:2]:
            path = out_dir / (name + '-unscaled-preclip-gradients.pt')
            if path.is_file():
                records[name]['gradient_comparison_to_FP32'] = compare_gradients(path, fp32_path, retained)
    return records


def validate_production_runtime(runtime):
    require(runtime['device'].startswith('cuda') and runtime['torch_version'] == PINNED_TORCH_BUILD,
            'Production device/PyTorch version differs')


def validate_report(report, *, require_production=True):
    require(report.get('schema_version') == 1 and report.get('status') == 'completed_diagnostic', 'Diagnostic incomplete')
    epoch = report['original_epoch']
    require(epoch['planned_batches'] == epoch['stats']['training_batches'], 'Incomplete original epoch')
    require(len(epoch['updates']) == math.ceil(epoch['planned_batches'] / epoch['loss_divisor']), 'Update count differs')
    require(epoch['scaler_method_calls']['unscale_'] == epoch['scaler_method_calls']['update'] == len(epoch['updates']),
            'Scaler update/unscale cadence differs')
    require(epoch['scaler_method_calls']['load_state_dict'] == 0, 'Original scaler was reloaded')
    require(set(report['cells']) == set(CELL_NAMES), 'Exactly three diagnostic cells are required')
    expected = report['selected_window']['identity']
    for name, cell in report['cells'].items():
        require(cell['attempted'] is True and cell['status'] in ['passed_numerical', 'failed_numerical', 'runtime_error'],
                'Replay cell lacks a disposition')
        if cell['initial_identity'] is not None:
            require(all(cell['initial_identity'][key] == expected[key] for key in cell['initial_identity']),
                    'Replay identities differ')
        if cell['status'] != 'runtime_error':
            require(len(cell['updates']) == 1 and cell['post_update_state']['tensors_checked'] > 0,
                    'Replay execution evidence incomplete')
    comparison = compare_original_window(epoch, report['selected_window'], report['cells']['fp16_scale32768'])
    require(comparison == report['original_window_replay_comparison'], 'Original/replay comparison differs')
    supported = (comparison['original_event_reproduced_by_replay_A']
                 and report['cells']['fp16_scale32768']['status'] == 'failed_numerical'
                 and report['cells']['fp16_scale16384']['status'] == report['cells']['fp32']['status'] == 'passed_numerical')
    require(report['candidate_scale16384_supported_on_captured_window'] == supported, 'Candidate numerical support differs')
    if require_production:
        require(epoch['planned_batches'] == 365 and epoch['loss_divisor'] == 8 and len(epoch['updates']) == 46,
                'Production diagnostic requires one complete 365-batch epoch with divisor eight')
        require(report['source']['commit_sha'] == C_COMMIT and report['plan_sha256'] == C_PLAN,
                'Historical source/plan binding differs')
        validate_production_runtime(report['runtime'])
        require(all(row['actual_loss_divisor'] == 8 and row['loss_divisor_source'] == 'archived_training_frame_accum_steps'
                    for row in epoch['batches']), 'Actual accumulation divisor was not observed from archived computation')
        require([row['actual_accumulation_count'] for row in epoch['updates']] == [8] * 45 + [5],
                'Original full/final partial accumulation differs')
    return {'status': 'completed_diagnostic_metadata_valid', 'numerical_policy_accepted': False,
            'original_optimizer_opportunities': len(epoch['updates']), 'replay_cells': 3}


def run_first_epoch(trainer, model, loader, optimizer, scaler, device, kwargs, out_dir, target_update, retained, qualifier):
    require(kwargs['epoch'] == 1 and kwargs['grad_accumulation_steps'] == 8, 'Only the original first epoch is allowed')
    require(scaler is not None and float(scaler.get_scale()) == 32768., 'Original scale must be 32768')
    record = {'planned_batches': len(loader), 'loss_divisor': kwargs['grad_accumulation_steps']}
    original_epoch = trainer.train_one_epoch
    try:
        with Observer(trainer, model, optimizer, scaler, device, retained, qualifier, record,
                      capture=True, target_update=target_update, out_dir=out_dir) as observer:
            observer.divisor = kwargs['grad_accumulation_steps']
            record['stats'] = original_epoch(model, loader, optimizer, scaler, device, **kwargs)
    finally:
        write_json(out_dir / 'original-epoch.json', record)
    require(target_update in observer.captures, 'Inferred target window was not reached')
    selected = observer.first_failure or target_update
    selected_capture = observer.captures[selected]
    captures = []
    for index, saved in sorted(observer.captures.items()):
        identity = snapshot_identity(saved['state'], saved['batches'], retained)
        path = out_dir / ('window-{:03d}.pt'.format(index))
        import torch
        with path.open('xb') as stream:
            torch.save(saved, stream)
        captures.append({'update_index': index, 'first_batch': saved['first_batch'], 'end_batch': saved['end_batch'],
                         'identity': identity, 'payload': reference(path), 'selected_for_replay': index == selected,
                         'following_payload_present': saved['following_batch'] is not None})
        if index != selected:
            saved.clear()
    selected_record = next(x for x in captures if x['update_index'] == selected)
    cells = replay_cells(trainer, model, optimizer, selected_capture, kwargs, device, out_dir, retained, qualifier)
    comparison = compare_original_window(record, selected_record, cells['fp16_scale32768'])
    scale_supported = (comparison['original_event_reproduced_by_replay_A']
                       and cells['fp16_scale32768']['status'] == 'failed_numerical'
                       and cells['fp16_scale16384']['status'] == cells['fp32']['status'] == 'passed_numerical')
    return {'original_epoch': record, 'captures': captures, 'selected_window': selected_record, 'cells': cells,
            'original_window_replay_comparison': comparison,
            'candidate_scale16384_supported_on_captured_window': scale_supported,
            'candidate_support_scope': 'Captured-window numerical evidence only; prospective full production and real paired-budget qualification still required.',
            'standalone_cell_report_scope': 'Per-cell JSON precedes cross-cell gradient-comparison enrichment; DIAGNOSTIC.json contains the authoritative enriched records.',
            'inference': {'target_update': target_update, 'inferred_batches': [(target_update - 1) * 8 + 1, target_update * 8],
                          'observed_failure_update': observer.first_failure,
                          'reproduced_at_inferred_location': observer.first_failure == target_update,
                          'selection_reason': 'first_observed_disqualifying_update' if observer.first_failure else 'no_event_reproduced_in_first_epoch_use_inferred_window'}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--target-update', type=int, default=27)
    parser.add_argument('--device', default='cuda:0')
    options = parser.parse_args(argv)
    require(options.plan_sha256 == C_PLAN and sha(options.plan) == options.plan_sha256, 'Archived C plan binding differs')
    require(options.plan.is_file() and not options.plan.is_symlink(), 'Plan must be a regular file')
    require(options.target_update == 27 and options.device == 'cuda:0', 'Production diagnostic target/device differs')
    require(not options.out_dir.exists() and '..' not in options.out_dir.parts, 'Output must be a fresh explicit directory')
    options.out_dir.mkdir(parents=True)
    report = {'schema_version': 1, 'status': 'incomplete_diagnostic', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'plan_sha256': options.plan_sha256, 'test_evaluation': 'not_run', 'heldout_payload_reads': 0}
    retained, qualifier = utilities()
    plan = json.loads(options.plan.read_text())
    report['source'] = plan['source']
    source = Path(plan['new_repo'])
    try:
        require(options.plan.parent == Path(plan['bundle_root']) and source == options.plan.parent / 'source',
                'Plan/source paths differ from archived bundle')
        require(plan['source']['commit_sha'] == C_COMMIT and retained.source_tree_sha256(source) == plan['source']['source_tree_sha256'],
                'Archived source bytes differ')
        settings = plan['settings_by_mode']['F2']
        report['archived_settings_sha256'] = retained.identity_digest(settings)
        require(settings['epochs'] == 12 and settings['seed'] == 42 and settings['development_only'] is True
                and settings['fusion_mode'] == 'F2' and settings['amp_init_scale'] == 32768. and settings['grad_accumulation_steps'] == 8,
                'Archived numerical/training settings differ')
        for key in ('metadata', 'radiomics', 'train_ids', 'val_ids', 'test_ids'):
            ref = plan['inputs'][key]
            path = Path(ref['path'])
            require(path.is_file() and not path.is_symlink() and sha(path) == ref['sha256'], 'Input binding differs: ' + key)
        # No current trifusesurv2 module may substitute for the archived package.
        require(not any(k == 'trifusesurv2' or k.startswith('trifusesurv2.') for k in sys.modules), 'Trainer package imported before archive selection')
        sys.path.insert(0, str(source / 'src'))
        from trifusesurv2.multimodal_survival import train as trainer
        import torch
        require(Path(trainer.__file__).resolve() == (source / 'src/trifusesurv2/multimodal_survival/train.py').resolve(), 'Wrong trainer module')
        for name, imported in list(sys.modules.items()):
            if name == 'trifusesurv2' or name.startswith('trifusesurv2.'):
                path = getattr(imported, '__file__', None)
                require(path is not None and source.resolve() in Path(path).resolve().parents,
                        'Mixed current/archive module origin: ' + name)
        scaler_file = Path(inspect.getfile(torch.amp.GradScaler))
        report['runtime'] = {'device': options.device, 'torch_version': torch.__version__,
                             'GradScaler_implementation': reference(scaler_file),
                             'training_resume': False, 'scheduler_T_max': 12}
        report['diagnostic_source_files'] = {p.name: reference(p) for p in
                                             [Path(__file__), Path(retained.__file__), Path(qualifier.__file__)]}
        original_train = trainer.train_one_epoch

        def intercepted(model, loader, optimizer, scaler, device, **kwargs):
            trainer.train_one_epoch = original_train
            require(len(loader) == 365, 'Original training loader must contain 365 batches')
            report['runtime']['backend'] = retained.backend_settings()
            report['runtime']['GradScaler_initial_state'] = scaler.state_dict()
            report.update(run_first_epoch(trainer, model, loader, optimizer, scaler, device, kwargs,
                                          options.out_dir, options.target_update, retained, qualifier))
            report['status'] = 'completed_diagnostic'
            report['validation'] = validate_report(report)
            raise EpochCaptured()

        trainer.train_one_epoch = intercepted
        original_argv = sys.argv
        args = list(plan['argv_by_mode']['F2'])
        out_index = args.index('--out_dir') + 1
        args[out_index] = str(options.out_dir / 'original-trainer-setup')
        report['archived_argv_sha256'] = retained.identity_digest(plan['argv_by_mode']['F2'])
        report['only_argv_change'] = {'flag': '--out_dir', 'diagnostic_output': args[out_index]}
        try:
            sys.argv = [str(source / 'src/trifusesurv2/multimodal_survival/train.py'), *args]
            trainer.main()
            raise RuntimeError('Archived trainer returned without diagnostic interception')
        except EpochCaptured:
            pass
        finally:
            trainer.train_one_epoch = original_train
            sys.argv = original_argv
    except BaseException as error:
        report['error_type'] = type(error).__name__
        report['error_detail'] = 'Project-private worker log contains details; portable report omits possible patient-bearing exception text.'
        raise
    finally:
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        partial = options.out_dir / 'original-epoch.json'
        if partial.is_file():
            report['original_epoch_metadata'] = reference(partial)
        write_json(options.out_dir / 'DIAGNOSTIC.json', report)
    print(json.dumps({'status': report['status'], 'report': str(options.out_dir / 'DIAGNOSTIC.json'),
                      'cell_statuses': {key: value['status'] for key, value in report['cells'].items()}}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
