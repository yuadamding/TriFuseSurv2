#!/usr/bin/env python3
"""Fabricated actual CT/Swin survival-NLL qualification; never read cohort inputs.

The production profile must run on one H100. The tiny CPU profile is only a
software fixture and cannot establish GPU or production-shape qualification.
This script neither selects a precision policy nor retries a failing update.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
CLINICAL_COLUMNS = ['PATHOLOGY', 'AGE', 'SEX', 'RACE', 'T', 'N', 'M', 'KFCF', 'SMOKE', 'ALCOHOL', 'HPV']
GROUP_PREFIXES = {
    'os_survival_head': ('habitat_model.survival_heads.endpoint_residuals.OS.',),
    'shared_survival_head': ('habitat_model.survival_heads.shared.', 'habitat_model.survival_heads.shared_logits.'),
    'shared_backbone': ('backbone.backbone_shared.swinViT.',),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_initial_scale(value):
    require(value is None or type(value) in (int, float) and math.isfinite(value) and value > 0,
            'Initial scaler scale must be finite and positive')


def comparison_modes(comparison):
    if comparison == 'paired-fusion':
        return ('F2', 'F3')
    require(comparison == 'routing-intervention', 'Unknown qualification comparison')
    from trifusesurv2.models.routing_intervention import ROUTING_EXPORT_LABEL
    return ('F3', ROUTING_EXPORT_LABEL)


def validate_settings(settings_by_mode, profile, comparison='paired-fusion'):
    modes = comparison_modes(comparison)
    require(set(settings_by_mode) == set(modes),
            'Exactly paired F2/F3 settings are required' if comparison == 'paired-fusion'
            else 'Exactly matched F3 and the distinct routing-intervention settings are required')
    routing_fields = ('routing_intervention', 'routing_intervention_design_sha256',
                      'routing_intervention_export_label')
    if comparison == 'routing-intervention':
        from trifusesurv2.models.routing_intervention import ROUTING_INTERVENTION_ID, routing_intervention_metadata
        expected_intervention = routing_intervention_metadata(ROUTING_INTERVENTION_ID)
    shared = []
    for mode in modes:
        settings = settings_by_mode[mode]
        validate_initial_scale(settings.get('amp_init_scale'))
        require(settings.get('numerical_failure_policy') in (None, 'zero_skip_stop_v1'),
                'Unknown prospective numerical failure policy')
        fusion_mode = mode if comparison == 'paired-fusion' else 'F3'
        require(settings['fusion_mode'] == fusion_mode and settings['model_version'] == 'v2'
                and settings['region_mode'] == 'verified', 'Only verified F2/F3 v2 is qualified')
        if comparison == 'routing-intervention' and mode != 'F3':
            require(all(settings.get(key) == value for key, value in expected_intervention.items()),
                    'The intervention settings must bind the exact identifier, design hash and export label')
        else:
            require(all(settings.get(key) is None for key in routing_fields),
                    'The matched control and ordinary paired qualification require inactive routing')
        if comparison == 'routing-intervention':
            require(settings.get('comparison_recipe') == 'principal',
                    'Routing qualification requires the principal controlled comparison recipe')
        require(settings['clinical_cols'] == CLINICAL_COLUMNS
                and settings.get('clinical_type_policy') == 'approved_categorical_v1',
                'The approved prospective clinical column/type policy is required')
        require(not any(settings.get(key) for key in ('resume', 'use_lora', 'use_ema', 'use_swa',
            'allow_partial_resume', 'contour_warmstart_ckpt', 'contour_warmstart_dir',
            'align_swin_cfg_from_contour_warmstart', 'ddp', 'data_parallel')),
            'Fresh full-parameter single-rank construction without checkpoint inputs is required')
        require(settings['batch_size'] == 1 and settings['grad_accumulation_steps'] >= 2,
                'Qualification must exercise batch one and an actual accumulation boundary')
        require(settings['roi_focus_warmup_epochs'] == 0 and settings['primary_surv_loss_weight'] > 0
                and settings['aux_surv_loss_weight'] > 0, 'All fabricated endpoints must participate without a survival warmup')
        require(settings['endpoint'] == 'OS' and settings['survival_uses_teacher_forced_masks'] is False
                and all(settings[key] == 0 for key in ('teacher_force_epochs', 'teacher_force_start',
                    'teacher_force_end', 'mask_guidance_alpha')),
                'This qualification supports OS with disabled, zero teacher forcing only')
        require(all(isinstance(settings[key], (int, float)) and math.isfinite(settings[key])
                    and 0 <= settings[key] <= 1 for key in ('v2_clinical_group_dropout_p',
                    'v2_radiomics_group_dropout_p', 'v2_node_dropout_p', 'v2_topology_dropout_p',
                    'v2_image_habitat_dropout_p')),
                'Explicit finite v2 dropout probabilities are required; fallback sentinels are unsupported')
        require(settings['token_dropout'] == 0 and settings['v2_image_habitat_dropout_p'] == 0,
                'Principal image-token dropout must remain disabled')
        if profile == 'production':
            require(settings['img_size'] == [96, 160, 96] and settings['feature_size'] == 96
                    and settings['depths'] == [2, 2, 18, 2] and settings['num_heads'] == [3, 6, 12, 24]
                    and settings['use_checkpoint'] is True and settings['grad_accumulation_steps'] == 8,
                    'Production qualification requires the declared full Swin/canvas and eight-batch accumulation')
            if comparison == 'routing-intervention':
                require(settings.get('amp') is True and settings.get('amp_init_scale') is not None,
                        'Production routing qualification requires an explicit bound CUDA AMP scale')
        elif profile == 'tiny':
            require(settings['img_size'] == [32, 32, 32] and settings['feature_size'] == 12
                    and settings['depths'] == [1, 1, 1, 1], 'Tiny engineering fixture shape differs')
        else:
            raise ValueError('Unknown qualification profile')
        ignored = ('fusion_mode', 'out_dir') + (routing_fields if comparison == 'routing-intervention' else ())
        shared.append({key: value for key, value in settings.items() if key not in ignored})
    require(shared[0] == shared[1], 'F2/F3 settings differ beyond fusion mode/output directory'
            if comparison == 'paired-fusion' else 'Matched/intervention settings differ beyond routing/output directory')
    return canonical_sha256(shared[0])


def make_precision(trainer, device, precision, init_scale, amp_enabled, *, training_init_scale=None):
    import torch
    validate_initial_scale(init_scale)
    validate_initial_scale(training_init_scale)
    if device.type != 'cuda':
        require(precision == 'fp32' and init_scale is None, 'CPU fixtures only support explicit FP32 without a scaler')
        return None, lambda: contextlib.nullcontext(), {'autocast_enabled': False, 'dtype': 'float32', 'scaler': None}
    if precision == 'trainer-default':
        require(init_scale is None, 'Trainer-default scale must come from bound training settings; use explicit fp16 for a diagnostic override')
        if training_init_scale is None:
            # Preserve the historical call and Torch default construction.
            scaler, autocast = trainer.make_amp(device, enabled=amp_enabled)
        else:
            require(amp_enabled, 'A bound initial scale requires enabled training AMP')
            scaler, autocast = trainer.make_amp(device, enabled=amp_enabled, init_scale=training_init_scale)
        dtype = str(torch.get_autocast_dtype('cuda')) if amp_enabled else 'torch.float32'
    elif precision == 'fp32':
        require(init_scale is None, 'FP32 does not use a scaler')
        scaler, autocast, dtype = None, lambda: contextlib.nullcontext(), 'torch.float32'
    else:
        require(precision in ('fp16', 'bf16'), 'Unsupported precision')
        dtype_obj = torch.float16 if precision == 'fp16' else torch.bfloat16
        dtype = str(dtype_obj)
        require(precision != 'bf16' or init_scale is None, 'BF16 diagnostic uses no scaler; this is not a policy adoption')
        scaler = torch.amp.GradScaler('cuda', **({'init_scale': init_scale} if init_scale is not None else {})) if precision == 'fp16' else None
        autocast = lambda: torch.amp.autocast('cuda', dtype=dtype_obj)
    state = scaler.state_dict() if scaler is not None else None
    if precision == 'trainer-default' and training_init_scale is not None:
        require(isinstance(state, dict) and state.get('scale') == training_init_scale,
                'Observed scaler initial scale differs from bound training settings')
    return scaler, autocast, {'autocast_enabled': precision != 'fp32' and (precision != 'trainer-default' or amp_enabled),
        'dtype': dtype, 'scaler': state,
        'selection': precision, 'production_policy_adopted_by_this_script': False}


def tensor_sha256(items):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(items):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode() + b'\0' + str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
        # NumPy preserves logical values even for size-one contiguous views
        # whose otherwise irrelevant stride prevents a dtype reinterpretation.
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def fabricated_batches(settings, count):
    import numpy as np
    import pandas as pd
    import torch
    from trifusesurv2.encoders.clinical import SemanticClinicalTokenEncoder
    from trifusesurv2.schema import PHYSICAL_GEOMETRY_FEATURES
    frame = pd.DataFrame({
        'AGE': [40., 70.], 'PATHOLOGY': ['type_a', 'type_b'], 'SEX': ['1', '2'], 'RACE': ['1', '2'],
        'T': ['T2', 'T4B'], 'N': ['N2A', 'N2B'], 'M': ['M0', 'M1'], 'KFCF': ['80', '90'],
        'SMOKE': ['1', '2'], 'ALCOHOL': ['1', '2'], 'HPV': ['nd', 'ND'], 'NSTAGE': ['excluded_a', 'excluded_b'],
        'TX': ['excluded_x', 'excluded_y']})
    encoder = SemanticClinicalTokenEncoder.fit(frame, columns=CLINICAL_COLUMNS, type_policy='approved_categorical_v1')
    tokens, presence = encoder.encode_frame_token_matrix(frame)
    require(np.isfinite(tokens).all() and np.isfinite(presence).all(), 'Fabricated clinical encoder produced nonfinite inputs')
    shape = tuple(settings['img_size'])
    axes = [torch.linspace(-1, 1, size) for size in shape]
    z, y, x = torch.meshgrid(*axes, indexing='ij')
    image = (.5 + .12 * torch.sin(3 * x) + .09 * torch.cos(4 * y) + .08 * z).unsqueeze(0).unsqueeze(0)
    pt_radius = ((z + .2) / .22)**2 + (y / .22)**2 + (x / .22)**2
    ln_radius = ((z - .3) / .15)**2 + ((y - .3) / .15)**2 + (x / .15)**2
    pt, ln = pt_radius <= 1, ln_radius <= 1
    regions = torch.stack((pt, (pt_radius > 1) & (pt_radius <= 2.5) & ~ln,
                           ln, (ln_radius > 1) & (ln_radius <= 2.5) & ~pt)).float().unsqueeze(0)
    require(bool((regions.sum(dim=(2, 3, 4)) > 0).all()), 'Every fabricated region must be present')
    generator = torch.Generator().manual_seed(19281)
    radiomics = torch.randn(2, 4, 16, generator=generator) * .2
    prototypes = []
    for index in range(2):
        times = torch.tensor([[270., 450., 630.]]) + 20 * index
        events = torch.tensor([[1., 1., 0.]]) if index == 0 else torch.tensor([[0., 0., 1.]])
        prototypes.append({'x': image + .01 * index, 'mask_pt': regions[:, 0:1], 'mask_ln': regions[:, 2:3],
            't': times[:, 0], 'e': events[:, 0], 't_all': times, 'e_all': events,
            'clinical_tokens': torch.from_numpy(tokens[index:index + 1]),
            'clinical_presence': torch.from_numpy(presence[index:index + 1]),
            'radiomics_tokens': radiomics[index:index + 1], 'radiomics_presence': torch.ones(1, 4),
            'voxel_spacing_dhw': torch.tensor([[3., 1., 1.]]), 'region_masks': regions,
            'region_validity': torch.ones(1, 4, dtype=torch.bool), 'region_states': torch.zeros(1, 4, dtype=torch.long),
            'physical_geometry': torch.zeros(1, 2 * len(PHYSICAL_GEOMETRY_FEATURES)), 'pid': ['fabricated_' + str(index)]})
    batches = [prototypes[index % 2] for index in range(count)]
    digest = tensor_sha256((str(index) + '/' + key, value) for index, batch in enumerate(batches)
                          for key, value in batch.items() if torch.is_tensor(value))
    return batches, encoder, {'sha256': digest, 'batches': count, 'distinct_fabricated_patients': 2,
        'shape_bcdhw': [1, 1, *shape], 'radiomics': 'fabricated tokens; no extraction or patient feature fitting',
        'clinical_columns': CLINICAL_COLUMNS, 'clinical_type_policy': 'approved_categorical_v1',
        'clinical_token_dimension': encoder.output_dim, 'radiomics_token_dimension': 16,
        'physical_geometry': 'all measurement-validity flags false; no anatomical measurements fabricated as observed'}


def build_model(trainer, settings, clinical_dim, device):
    """Use the already qualified, source-hashed constructor extractor."""
    path = ROOT / 'scripts/checkpoint_optimizer_mapping.py'
    spec = importlib.util.spec_from_file_location('qualification_constructor', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    code, evidence = module.construction_statements(trainer)
    namespace = dict(trainer.__dict__)
    args = argparse.Namespace(**settings)
    namespace.update(args=args, fold=0, fold_num_time_bins=6, clin_dim=clinical_dim,
                     rad_dim=16, mode=trainer._image_encoder_mode(args), device=device)
    exec(code, namespace)
    return namespace['base_model'], dict(evidence, helper_sha256=file_sha256(path), time_bins=6)


def tensor_observation(tensor):
    import torch
    value = tensor.detach()
    finite = torch.isfinite(value)
    finite_values = value[finite].double()
    return {'dtype': str(value.dtype), 'shape': list(value.shape), 'elements': value.numel(),
        'finite_elements': int(finite.sum().item()), 'nan_elements': int(torch.isnan(value).sum().item()),
        'infinite_elements': int(torch.isinf(value).sum().item()),
        'maximum_finite_absolute_value': float(finite_values.abs().max().item()) if finite_values.numel() else None,
        'finite_elements_l2_norm_float64': float(torch.linalg.vector_norm(finite_values).item()) if finite_values.numel() else None}


def json_safe_evidence(value):
    """Retain failed numerical observations without illegal JSON NaN/Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_numeric_value': repr(value)}
    if isinstance(value, dict):
        return {key: json_safe_evidence(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_evidence(item) for item in value]
    return value


def active_names(model):
    names = dict(model.named_parameters())
    groups = {key: [name for name in names if name.startswith(prefix)] for key, prefix in GROUP_PREFIXES.items()}
    require(all(groups.values()), 'Required named survival/backbone paths are absent')
    backbone_required = [name for name in groups['shared_backbone'] if name.endswith('patch_embed.proj.weight')
                         or name.endswith('layers1.0.blocks.0.attn.qkv.weight')]
    require(len(backbone_required) == 2, 'Expected concrete active Swin input/attention weights are absent')
    required = groups['os_survival_head'] + groups['shared_survival_head'] + backbone_required
    return groups, required


def observe_gradients(model, groups, required):
    rows, missing, invalid = {}, [], []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
            continue
        rows[name] = tensor_observation(parameter.grad)
        if rows[name]['finite_elements'] != rows[name]['elements']:
            invalid.append(name)
    return {'parameters': rows, 'grad_none': missing, 'nonfinite_parameters': invalid,
        'missing_required_parameters': sorted(set(required) & set(missing)),
        'required_parameter_names': required,
        'group_nonzero_gradient_parameters': {key: [name for name in names if name in rows
            and (rows[name]['maximum_finite_absolute_value'] or 0) > 0] for key, names in groups.items()}}


def require_gradients(observation):
    require(observation['parameters'], 'Gradient collection is empty')
    require(not observation['missing_required_parameters'], 'Required active-path gradients are missing')
    require(not observation['nonfinite_parameters'], 'NaN or infinite gradients were observed')
    require(all(observation['group_nonzero_gradient_parameters'].values()), 'Required path has no nonzero gradient')


def state_evidence(model, optimizer, groups, successful):
    import torch
    checked_model, checked_optimizer = 0, 0
    for name, value in model.state_dict().items():
        require(bool(torch.isfinite(value).all()), 'Nonfinite model state: ' + name)
        checked_model += 1
    for values in optimizer.state.values():
        for value in values.values():
            if torch.is_tensor(value):
                require(bool(torch.isfinite(value).all()), 'Nonfinite optimizer tensor')
                checked_optimizer += 1
    named = dict(model.named_parameters())
    evidence = {}
    for group, names in groups.items():
        steps, missing, nonzero_moments = {}, [], []
        for name in names:
            state = optimizer.state.get(named[name])
            if not state:
                missing.append(name); continue
            step = float(state['step'].item())
            require(step.is_integer() and step == successful, 'Named Adam step count differs from actual successful invocations')
            steps[name] = int(step)
            if bool(state['exp_avg'].abs().max() > 0) and bool(state['exp_avg_sq'].abs().max() > 0):
                nonzero_moments.append(name)
        require(steps and nonzero_moments, 'Required path lacks actual positive steps/nonzero moments')
        require(group == 'shared_backbone' or not missing, 'Survival-head optimizer states are missing')
        evidence[group] = {'parameter_steps': steps, 'parameters_without_state': missing,
                           'nonzero_moment_parameters': nonzero_moments}
    return {'model_tensors_finite_checked': checked_model, 'optimizer_tensors_finite_checked': checked_optimizer,
            'named_update_evidence': evidence}


@contextlib.contextmanager
def instrument_trainer(trainer, model, optimizer, scaler, record, groups, required):
    """Observe unchanged trainer helpers, including pre-clip gradients after unscale."""
    originals = {name: getattr(trainer, name) for name in ('perform_optimizer_update', 'clip_grad_norm_', '_endpoint_survival_losses')}
    actual_calls = [0]
    handle = optimizer.register_step_post_hook(lambda *_: actual_calls.__setitem__(0, actual_calls[0] + 1))
    active = [None]
    original_scale = scaler.scale if scaler is not None else None
    record['loss_scaling_observations'] = []

    def scale(loss, *args, **kwargs):
        observation = {'batch_index': len(record['batch_survival_losses']),
            'accumulation_normalized_total_loss': tensor_observation(loss),
            'requires_grad': bool(loss.requires_grad), 'scale': float(scaler.get_scale())}
        record['loss_scaling_observations'].append(observation)
        result = original_scale(loss, *args, **kwargs)
        observation['scaled_loss'] = tensor_observation(result)
        return result

    def clip(parameters, *args, **kwargs):
        observation = observe_gradients(model, groups, required)
        active[0]['after_unscale_before_clip'] = observation
        require_gradients(observation)
        return originals['clip_grad_norm_'](parameters, *args, **kwargs)

    def update(*args, **kwargs):
        boundary = {'update_index': len(record['updates']) + 1, 'after_training_batch': len(record['batch_survival_losses']),
            'scale_before': float(scaler.get_scale()) if scaler else None,
            'before_unscale': observe_gradients(model, groups, required), 'actual_optimizer_calls_before': actual_calls[0]}
        record['updates'].append(boundary); active[0] = boundary
        result = originals['perform_optimizer_update'](*args, **kwargs)
        boundary.update(result, actual_optimizer_calls_after=actual_calls[0],
                        scale_after=float(scaler.get_scale()) if scaler else None)
        require(result['optimizer_steps_succeeded'] == 1 and result['optimizer_steps_skipped'] == 0
                and actual_calls[0] == boundary['update_index'], 'Qualification requires successful actual updates without skipped exposure')
        require(math.isfinite(result['gradient_norm']), 'Actual trainer clipping norm is nonfinite')
        boundary['post_update_state'] = state_evidence(model, optimizer, groups, actual_calls[0])
        return result

    def endpoint_losses(logits, **kwargs):
        losses, counts = originals['_endpoint_survival_losses'](logits, **kwargs)
        item = {'batch_index': len(record['batch_survival_losses']) + 1, 'endpoints': {}}
        record['batch_survival_losses'].append(item)
        for endpoint, loss in losses.items():
            output = logits[endpoint]
            item['endpoints'][endpoint] = {'output': tensor_observation(output), 'loss': tensor_observation(loss),
                'loss_requires_grad': bool(loss.requires_grad), 'valid_outcome_count': int(counts[endpoint])}
            require(bool(__import__('torch').isfinite(output).all()) and bool(__import__('torch').isfinite(loss).all()),
                    'Nonfinite endpoint output/NLL')
            require(loss.dtype == __import__('torch').float32 and loss.requires_grad, 'Actual survival NLL must be differentiable FP32')
        require(set(losses) == {'OS', 'DSS', 'DFS'}, 'All three fabricated survival endpoints must be exercised')
        return losses, counts

    trainer.clip_grad_norm_ = clip
    trainer.perform_optimizer_update = update
    trainer._endpoint_survival_losses = endpoint_losses
    if scaler is not None: scaler.scale = scale
    try:
        yield
    finally:
        for name, value in originals.items(): setattr(trainer, name, value)
        if scaler is not None: scaler.scale = original_scale
        handle.remove()


def reference_routing_assignment(image_presence, radiomics_presence):
    """Independent, fabricated-mask reference; never reads IDs or measurements."""
    assignments = []
    for query_mask, key_mask in zip(image_presence.tolist(), radiomics_presence.tolist()):
        assignment = list(range(4))
        classes = {}
        for index, signature in enumerate(zip(query_mask, key_mask)):
            classes.setdefault(signature, []).append(index)
        for members in classes.values():
            for offset, index in enumerate(members):
                assignment[index] = members[(offset + 1) % len(members)]
        assignments.append(assignment)
    return assignments


@contextlib.contextmanager
def instrument_routing(model, settings, record, phase):
    """Observe actual fuser masks and local outputs without changing its graph.

    Hooks receive the prelocal presence masks after trainer dropout. The
    reference reconstructs the prescribed Boolean edges independently of the
    runtime assignment helper, then compares every actual local-fuser input.
    This instrumentation is exclusive to the explicit intervention comparison.
    """
    import torch
    from trifusesurv2.models.routing_intervention import (
        REGIONAL_ORDER, ROUTING_INTERVENTION_ID, routing_intervention_metadata,
    )
    core = model.habitat_model
    intervention = settings.get('routing_intervention')
    require(core.fusion_mode == 'F3' and core.routing_intervention == intervention,
            'Constructed routing branch differs from bound F3 settings')
    require(tuple(core.radiomics_habitats) == tuple(REGIONAL_ORDER),
            'Qualification requires the canonical four radiomics keys')
    expected_metadata = routing_intervention_metadata(intervention)
    evidence = record.setdefault('routing_execution', {
        'fusion_mode': 'F3', 'routing_intervention': intervention,
        'design': routing_intervention_metadata(ROUTING_INTERVENTION_ID),
        'observation': 'actual_prelocal_presence_and_local_fuser_input_masks_and_outputs',
        'phases': {},
    })
    frames = evidence['phases'].setdefault(phase, [])
    pending = [None]
    handles = []

    def begin(module, args, kwargs):
        require(pending[0] is None, 'Overlapping routing forwards are unsupported')
        image = kwargs['image_tokens']
        query = kwargs.get('image_presence')
        query = (torch.ones(image.shape[:2], device=image.device, dtype=torch.bool)
                 if query is None else query.to(device=image.device, dtype=torch.bool))
        keys = kwargs.get('radiomics_presence')
        if keys is None:
            keys = torch.full((image.shape[0], 4), kwargs.get('radiomics_tokens') is not None,
                              device=image.device, dtype=torch.bool)
        else:
            keys = keys.to(device=image.device, dtype=torch.bool)
        clinical = kwargs.get('clinical_presence')
        if clinical is None:
            clinical = torch.full((image.shape[0], len(core.clinical_groups)),
                                  kwargs.get('clinical_tokens') is not None,
                                  device=image.device, dtype=torch.bool)
        else:
            clinical = clinical.to(device=image.device, dtype=torch.bool)
        regional_queries = [core.image_index[name] for name in REGIONAL_ORDER]
        regional_presence = query[:, regional_queries]
        assignment = (reference_routing_assignment(regional_presence, keys) if intervention is not None
                      else [list(range(4)) for _ in range(image.shape[0])])
        assignment_tensor = torch.tensor(assignment, device=image.device, dtype=torch.long)
        nominal = torch.zeros((image.shape[0], len(core.image_habitats),
                               1 + 4 + len(core.clinical_groups)), device=image.device, dtype=torch.bool)
        nominal[:, :, 0] = True
        nominal[:, :, 5:] = True
        for region, index in enumerate(regional_queries):
            nominal[:, index, 1:5] = torch.nn.functional.one_hot(assignment_tensor[:, region], 4).bool()
        availability = torch.cat((query[:, :, None],
            keys[:, None].expand(-1, len(core.image_habitats), -1),
            clinical[:, None].expand(-1, len(core.image_habitats), -1)), dim=2)
        actual_expected = nominal & availability
        changed = (assignment_tensor != torch.arange(4, device=image.device)[None])
        effective = changed & regional_presence & keys.gather(1, assignment_tensor)
        frame = {'training': bool(module.training), 'query_names': list(core.image_habitats),
                 'prelocal_image_presence': regional_presence.tolist(),
                 'prelocal_radiomics_presence': keys.tolist(), 'assignment': assignment,
                 'effective_mismatched_edges': int(effective.sum().item()),
                 'expected_context_mask': actual_expected.tolist(), 'actual_local_fusers': {}}
        frames.append(frame)
        pending[0] = (frame, actual_expected)

    def observe(index, habitat):
        def hook(module, args):
            require(pending[0] is not None, 'Local fuser was called outside its recorded forward')
            frame, expected = pending[0]
            require(habitat not in frame['actual_local_fusers'], 'A local query was repeated')
            observed = args[2].detach()
            require(torch.equal(observed, expected[:, index]), 'Actual local attention mask differs from prescribed routing')
            frame['actual_local_fusers'][habitat] = {'context_mask': observed.tolist()}
        return hook

    def observe_output(habitat):
        def hook(module, args, output):
            frame, _ = pending[0]
            value = output[0] if isinstance(output, tuple) else output
            require(bool(torch.isfinite(value).all()), 'Nonfinite local output before global fusion')
            frame['actual_local_fusers'][habitat]['local_output'] = tensor_observation(value)
        return hook

    def finish(module, args, kwargs, output):
        require(pending[0] is not None, 'Routing forward completion has no input record')
        frame, _ = pending[0]
        require(set(frame['actual_local_fusers']) == set(core.image_habitats),
                'Actual local query coverage is incomplete')
        frame['status'] = 'passed_actual_masks_and_local_outputs'
        pending[0] = None

    handles.append(core.register_forward_pre_hook(begin, with_kwargs=True))
    handles.append(core.register_forward_hook(finish, with_kwargs=True))
    for index, habitat in enumerate(core.image_habitats):
        handles.append(core.habitat_fusers[habitat].register_forward_pre_hook(observe(index, habitat)))
        handles.append(core.habitat_fusers[habitat].register_forward_hook(observe_output(habitat)))
    try:
        yield
        require(pending[0] is None and frames and all(frame.get('status') == 'passed_actual_masks_and_local_outputs'
                for frame in frames), 'Actual routing execution evidence is incomplete')
        if intervention is not None and phase == 'training':
            require(any(frame['effective_mismatched_edges'] > 0 for frame in frames),
                    'Training did not exercise an effective mismatch')
        evidence.update(expected_metadata)
    finally:
        for handle in handles: handle.remove()


def qualify_mode(trainer, settings, options, record):
    import torch
    device = torch.device(options.device)
    trainer.set_seed(settings['seed'], deterministic=bool(settings['deterministic']))
    trainer.configure_cuda_performance(argparse.Namespace(**settings), device)
    record['backend_settings'] = {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'float32_matmul_precision': torch.get_float32_matmul_precision()}
    batches, encoder, inputs = fabricated_batches(settings, options.batches)
    record.update(input_identity=inputs, clinical_policy=encoder.column_policy,
                  clinical_type_policy=encoder.type_policy, updates=[], batch_survival_losses=[])
    scaler, autocast, precision = make_precision(trainer, device, options.precision, options.amp_init_scale,
                                               settings['amp'], training_init_scale=settings.get('amp_init_scale'))
    record['precision'] = precision
    model, construction = build_model(trainer, settings, encoder.output_dim, device)
    record['construction'] = construction
    trainer.materialize_lazy_modules(model, batches, device, autocast)
    record['initial_state_sha256'] = tensor_sha256(model.state_dict().items())
    record['parameters'] = sum(parameter.numel() for parameter in model.parameters())
    record['trainable_parameters'] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if getattr(options, 'comparison', 'paired-fusion') == 'routing-intervention':
        rng_states = [('cpu', torch.get_rng_state())]
        if device.type == 'cuda': rng_states.append(('cuda', torch.cuda.get_rng_state(device)))
        record['initial_rng_state_sha256'] = tensor_sha256(rng_states)
        record['bound_training_amp_init_scale'] = settings.get('amp_init_scale')
    groups, required = active_names(model)
    record['expected_active_parameter_names'] = required
    controls = {key: settings[key] for key in inspect.signature(trainer.make_param_groups).parameters if key != 'mm'}
    optimizer = torch.optim.AdamW(trainer.make_param_groups(model, **controls), betas=(.9, .999), weight_decay=0)
    kwargs = {key: settings[key] for key, parameter in inspect.signature(trainer.train_one_epoch).parameters.items()
              if key in settings and parameter.kind == inspect.Parameter.KEYWORD_ONLY}
    kwargs.update(fold=0, epoch=1, primary_endpoint='OS', num_time_bins=6, time_bin_width_days=settings['time_bin_width_days'],
        teacher_force_alpha=0., survival_loss_weight=1., ema=None, autocast_ctx=autocast,
        grad_accumulation_steps=settings['grad_accumulation_steps'], nonfinite_check_every_batches=1)
    numerical_contract = trainer._numerical_policy_contract(argparse.Namespace(**settings))
    if numerical_contract:
        require(getattr(options, 'output', None) is not None,
                'Active numerical-policy qualification requires an exact output evidence path')
        mode_label = str(record['mode'])
        failure_path = Path(options.output).with_name(Path(options.output).stem + '.' + mode_label + '.numerical_failure.json')
        require(not failure_path.exists() and not failure_path.is_symlink(),
                'Numerical failure evidence path must be new')
        kwargs.update(numerical_failure_policy=settings['numerical_failure_policy'],
                      numerical_failure_receipt=failure_path,
                      numerical_run_context={'seed': int(settings['seed']), 'mode': settings['fusion_mode'],
                                             'qualification_mode_label': mode_label})
        record['numerical_failure_contract'] = numerical_contract
        record['numerical_failure_receipt_path'] = str(failure_path)
    routing_comparison = getattr(options, 'comparison', 'paired-fusion') == 'routing-intervention'
    routing_context = (instrument_routing(model, settings, record, 'training') if routing_comparison
                       else contextlib.nullcontext())
    with routing_context, instrument_trainer(trainer, model, optimizer, scaler, record, groups, required):
        stats = trainer.train_one_epoch(model, batches, optimizer, scaler, device, **kwargs)
    record['training_statistics'] = {key: value for key, value in stats.items() if isinstance(value, (int, float)) and math.isfinite(value)}
    record['unassessable_optional_statistics'] = [key for key, value in stats.items() if isinstance(value, (int, float)) and not math.isfinite(value)]
    expected = math.ceil(options.batches / settings['grad_accumulation_steps'])
    require(stats['training_batches'] == options.batches and stats['loss_finite_checks'] == options.batches
            and stats['optimizer_step_attempts'] == expected and stats['optimizer_steps_succeeded'] == expected
            and stats['optimizer_steps_skipped'] == 0 and stats['nonfinite_gradient_updates'] == 0,
            'Actual trainer accumulation/update telemetry differs')
    require([row['after_training_batch'] for row in record['updates']] == [settings['grad_accumulation_steps'], options.batches],
            'Full accumulation and final partial accumulation were not both exercised')
    for key in ('loss_total', 'loss_surv_total', 'loss_surv_primary', 'surv_os_nll', 'surv_dss_nll', 'surv_dfs_nll'):
        require(math.isfinite(stats[key]), 'Required aggregate training loss is nonfinite: ' + key)
    model.eval()
    routing_context = (instrument_routing(model, settings, record, 'post_update_eval') if routing_comparison
                       else contextlib.nullcontext())
    with routing_context, torch.no_grad():
        payload = trainer._unpack_surv_batch(batches[-1])
        output = trainer._model_forward_eval(model, payload, device, autocast, teacher_force_alpha=0.)
        losses, counts = trainer._endpoint_survival_losses(output, t_all=payload['t_all'].to(device),
            e_all=payload['e_all'].to(device), time_bin_width_days=settings['time_bin_width_days'], num_time_bins=6)
    record['post_update_forward'] = {endpoint: {'output': tensor_observation(output[endpoint]),
        'nll': float(loss.item()), 'loss_dtype': str(loss.dtype)} for endpoint, loss in losses.items()}
    require(all(bool(torch.isfinite(output[name]).all()) and math.isfinite(float(loss)) for name, loss in losses.items()),
            'Post-update outputs or survival NLL are nonfinite')
    record['final_state'] = state_evidence(model, optimizer, groups, expected)
    record['final_state_sha256'] = tensor_sha256(model.state_dict().items())
    require(record['final_state_sha256'] != record['initial_state_sha256'], 'Actual model state did not change')
    record.update(status='passed', actual_successful_updates=expected, final_partial_accumulation_batches=options.batches % settings['grad_accumulation_steps'])
    del model, optimizer, batches, output, losses
    gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()


def validate_routing_comparison_report(report, *, profile='production'):
    """A legacy F2/F3 pass cannot release the newly introduced branch.

    This numerical-report check supplements the caller's source/settings/image
    hash bindings and the separate exhaustive runtime-mask compatibility gate.
    Tiny reports remain explicit CPU fixtures and cannot satisfy production.
    """
    from trifusesurv2.models.routing_intervention import (
        ROUTING_EXPORT_LABEL, ROUTING_INTERVENTION_ID, routing_intervention_metadata,
    )
    require(report.get('schema_version') == 2 and report.get('comparison') == 'routing-intervention'
            and report.get('profile') == profile and report.get('status') == 'passed',
            'An explicit successful routing-intervention comparison report with the required profile is required')
    require(profile in ('production', 'tiny'), 'Unknown routing report profile')
    require(report.get('routing_design') == routing_intervention_metadata(ROUTING_INTERVENTION_ID),
            'Routing report design binding differs')
    modes = report.get('modes', {})
    require(set(modes) == {'F3', ROUTING_EXPORT_LABEL}, 'Routing report mode labels differ')
    batches, accumulation = report.get('comparison_batches'), report.get('comparison_accumulation')
    require(type(accumulation) is int and accumulation >= 2 and batches == accumulation + 1,
            'Routing comparison must exercise one full and one final partial accumulation')
    if profile == 'production':
        hardware = report.get('hardware', {})
        require(report.get('device') == 'cuda:0' and batches == 9 and accumulation == 8
                and hardware.get('visible_device_count') == 1 and 'H100' in str(hardware.get('name', '')).upper()
                and hardware.get('compute_capability', [0])[0] == 9
                and hardware.get('total_memory_bytes', 0) >= 75 * 1024**3,
                'Routing production evidence requires the actual full one-H100 profile')
    for label in ('F3', ROUTING_EXPORT_LABEL):
        record = modes[label]
        require(record.get('status') == 'passed' and record.get('fusion_mode') == 'F3'
                and record.get('actual_successful_updates') == 2
                and record.get('final_partial_accumulation_batches') == 1,
                'Routing mode numerical qualification is incomplete')
        stats = record.get('training_statistics', {})
        require(stats.get('training_batches') == batches and stats.get('optimizer_step_attempts') == 2
                and stats.get('optimizer_steps_succeeded') == 2 and stats.get('optimizer_steps_skipped') == 0
                and stats.get('nonfinite_gradient_updates') == 0,
                'Routing mode has incomplete or skipped numerical exposure')
        execution = record.get('routing_execution', {})
        intervention = None if label == 'F3' else ROUTING_INTERVENTION_ID
        require(execution.get('fusion_mode') == 'F3' and execution.get('routing_intervention') == intervention
                and execution.get('design') == report['routing_design'], 'Observed routing branch differs')
        for phase, expected_count, training in (('training', batches, True), ('post_update_eval', 1, False)):
            frames = execution.get('phases', {}).get(phase, [])
            require(len(frames) == expected_count and all(frame.get('training') is training
                    and frame.get('status') == 'passed_actual_masks_and_local_outputs' for frame in frames),
                    'Actual training/evaluation local-mask evidence is incomplete')
            for frame in frames:
                names, actual = frame.get('query_names', []), frame.get('actual_local_fusers', {})
                expected_masks = frame.get('expected_context_mask', [])
                require(names and len(names) == len(set(names)) and set(names) == set(actual)
                        and expected_masks and all(len(row) == len(names) for row in expected_masks),
                        'Observed local-fuser query coverage is incomplete')
                for index, habitat in enumerate(names):
                    observed = actual[habitat]
                    output = observed.get('local_output', {})
                    require(observed.get('context_mask') == [row[index] for row in expected_masks]
                            and output.get('elements', 0) > 0
                            and output.get('finite_elements') == output.get('elements'),
                            'Observed local masks or finite local outputs differ from the execution evidence')
        wrong_edges = sum(frame['effective_mismatched_edges'] for frame in execution['phases']['training'])
        require(wrong_edges > 0 if intervention is not None else wrong_edges == 0,
                'The intended matched/intervention local branch did not execute')
        if profile == 'production':
            precision = record.get('precision', {})
            scale = record.get('bound_training_amp_init_scale')
            state = precision.get('scaler')
            require(precision.get('selection') == 'trainer-default'
                    and type(scale) in (int, float) and math.isfinite(scale) and scale > 0
                    and isinstance(state, dict) and state.get('scale') == scale,
                    'Production routing qualification must use the bound trainer scale, not a diagnostic override')
    left, right = modes['F3'], modes[ROUTING_EXPORT_LABEL]
    identity_keys = ('initial_state_sha256', 'initial_rng_state_sha256', 'input_identity',
                     'parameters', 'trainable_parameters')
    require(all(left.get(key) is not None and left[key] == right.get(key) for key in identity_keys)
            and left.get('bound_training_amp_init_scale') == right.get('bound_training_amp_init_scale'),
            'Matched/intervention initial states, RNG state, inputs, parameters or numerical policy differ')
    return {'status': 'passed_source_bound_routing_numerical_comparison', 'profile': profile,
            'routing_design': report['routing_design'], 'modes': ['F3', ROUTING_EXPORT_LABEL],
            'successful_updates_per_mode': 2, 'skipped_updates_per_mode': 0,
            'production_H100_qualified': profile == 'production'}


def run(settings_by_mode, options):
    import torch
    from trifusesurv2.multimodal_survival import train as trainer
    from trifusesurv2.utils.provenance import source_identity
    report = {'schema_version': 1, 'status': 'failed', 'scope': 'fabricated_actual_CT_Swin_survival_NLL_engineering',
        'profile': options.profile, 'device': options.device, 'source_identity': source_identity(),
        'fixture_sha256': file_sha256(__file__), 'patient_input_reads': 0, 'patient_fits': 0,
        'checkpoint_payload_reads': 0, 'production_precision_policy_changed': False, 'modes': {},
        'container_image': options.container_image, 'started_at_utc': datetime.now(timezone.utc).isoformat()}
    try:
        comparison = getattr(options, 'comparison', 'paired-fusion')
        modes = comparison_modes(comparison)
        if comparison == 'routing-intervention':
            from trifusesurv2.models.routing_intervention import ROUTING_INTERVENTION_ID, routing_intervention_metadata
            report.update(schema_version=2, comparison=comparison,
                          routing_design=routing_intervention_metadata(ROUTING_INTERVENTION_ID))
            if options.profile == 'production':
                require(options.precision == 'trainer-default' and options.amp_init_scale is None,
                        'Routing production qualification must use the bound trainer policy without diagnostic override')
        report['shared_settings_sha256'] = validate_settings(settings_by_mode, options.profile, comparison)
        accumulation = settings_by_mode[modes[0]]['grad_accumulation_steps']
        require(options.batches == accumulation + 1, 'Exactly one full accumulation and one final partial batch are required')
        if comparison == 'routing-intervention':
            report.update(comparison_batches=options.batches, comparison_accumulation=accumulation)
        if options.profile == 'production':
            require(options.device == 'cuda:0', 'Production profile requires one H100 CUDA device')
        if torch.device(options.device).type == 'cuda':
            from qualify_h100 import require_one_h100
            report['hardware'] = require_one_h100(torch)
        else:
            report['hardware'] = {'qualification': 'CPU engineering only; not H100 execution'}
        report['backend'] = {'torch_version': str(torch.__version__), 'cuda_build': torch.version.cuda}
        for mode in modes:
            record = {'status': 'failed', 'mode': mode, 'seed': settings_by_mode[mode]['seed']}
            if comparison == 'routing-intervention': record['fusion_mode'] = 'F3'
            report['modes'][mode] = record
            try:
                qualify_mode(trainer, settings_by_mode[mode], options, record)
            except Exception as error:
                record.update(error_type=type(error).__name__, error=str(error))
                gc.collect()
                if torch.device(options.device).type == 'cuda': torch.cuda.empty_cache()
        require(all(item['status'] == 'passed' for item in report['modes'].values()),
                'At least one prescribed fusion-mode qualification failed; inspect retained mode/phase evidence')
        left, right = (report['modes'][mode] for mode in modes)
        require(left['initial_state_sha256'] == right['initial_state_sha256']
                and left['input_identity'] == right['input_identity'] and left['parameters'] == right['parameters'],
                'Materialized paired initial states, fabricated inputs or parameter counts differ')
        report['status'] = 'passed'
        if comparison == 'routing-intervention':
            report['routing_comparison_validation'] = validate_routing_comparison_report(report, profile=options.profile)
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
    finally:
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    return json_safe_evidence(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-settings', type=Path, required=True)
    parser.add_argument('--settings-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=('production', 'tiny'), default='production')
    parser.add_argument('--comparison', choices=('paired-fusion', 'routing-intervention'), default='paired-fusion',
                        help='Explicit matched-F3/intervention qualification; legacy paired-fusion stays unchanged')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=('trainer-default', 'fp32', 'fp16', 'bf16'), default='trainer-default')
    parser.add_argument('--amp-init-scale', type=float,
                        help='Diagnostic explicit-FP16 override only; trainer-default uses bound training amp_init_scale')
    parser.add_argument('--batches', type=int, default=9)
    parser.add_argument('--container-image', default='unrecorded')
    args = parser.parse_args(argv)
    require(not args.output.exists() and not args.output.is_symlink(), 'Output must be a new evidence file')
    require(args.training_settings.is_file() and not args.training_settings.is_symlink()
            and args.training_settings.stat().st_size < 1024 * 1024
            and file_sha256(args.training_settings) == args.settings_sha256, 'Exact bounded settings JSON is required')
    settings = json.loads(args.training_settings.read_text())['settings_by_mode']
    report = run(settings, args)
    report['settings_file'] = {'path': str(args.training_settings.resolve()), 'sha256': args.settings_sha256}
    with args.output.open('x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'output': str(args.output), 'sha256': file_sha256(args.output)}))
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
