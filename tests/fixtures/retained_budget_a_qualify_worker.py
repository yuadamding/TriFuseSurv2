#!/usr/bin/env python3
"""Same-interpreter source tests and fabricated CUDA routing/clinical/AMP gate."""
import argparse
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from budget_contract import BUNDLE, OUTPUT, bound, load, sha, write, expected_observed_source


def cuda_gate(plan):
    import numpy as np
    import pandas as pd
    import torch
    from trifusesurv2.encoders.clinical import SemanticClinicalTokenEncoder
    from trifusesurv2.models.habitat_survival import HabitatAlignedSurvivalModel
    from trifusesurv2.utils.provenance import source_identity
    if source_identity() != expected_observed_source(plan):
        raise ValueError('Actual imported source identity differs')
    frame = pd.DataFrame({'AGE': [40, 60], 'HPV': [0, 1], 'T': ['T2', 'T3'], 'N': ['N0', 'N1'],
                          'M': ['M0', 'M0'], 'NSTAGE': ['II', 'IV'], 'TX': ['A', 'B']})
    encoder = SemanticClinicalTokenEncoder.fit(frame, columns=plan['settings_by_mode']['F2']['clinical_cols'])
    tokens, present = encoder.encode_frame_token_matrix(frame)
    changed = encoder.encode_frame_token_matrix(frame.assign(NSTAGE=['SECRET', None], TX=['SECRET', None]))
    if not all(np.array_equal(x, y) for x, y in zip((tokens, present), changed)):
        raise ValueError('Excluded clinical values reached prospective neural tokens')
    results = []
    for mode in ('F2', 'F3'):
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        model = HabitatAlignedSurvivalModel(image_token_dim=8, radiomics_token_dim=5,
            clinical_token_dim=encoder.max_token_dim, num_time_bins=3, model_dim=16,
            num_heads=4, transformer_layers=1, dropout=0., fusion_mode=mode).cuda().train()
        inputs = dict(image_tokens=torch.randn(2, 6, 8, device='cuda'),
            image_presence=torch.ones(2, 6, device='cuda', dtype=torch.bool),
            radiomics_tokens=torch.randn(2, 4, 5, device='cuda'),
            radiomics_presence=torch.ones(2, 4, device='cuda', dtype=torch.bool),
            clinical_tokens=torch.tensor(tokens, device='cuda'),
            clinical_presence=torch.tensor(present, device='cuda', dtype=torch.bool),
            region_states=torch.zeros(2, 4, device='cuda', dtype=torch.long))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        scaler = torch.amp.GradScaler('cuda'); successful = []
        handle = optimizer.register_step_post_hook(lambda *unused: successful.append(True))
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(**inputs)
            loss = sum(value.float().square().mean() for value in outputs.values())
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite tiny CUDA loss')
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
            raise ValueError('Nonfinite tiny CUDA gradient')
        scaler.step(optimizer); scaler.update(); handle.remove(); torch.cuda.synchronize()
        if len(successful) != 1 or not all(bool(torch.isfinite(value).all()) for value in model.parameters()):
            raise ValueError('Tiny actual CUDA AMP update did not complete')
        results.append({'mode': mode, 'loss': float(loss), 'actual_optimizer_steps': 1,
                        'gradients_finite': True, 'model_parameters_finite': True})
        del outputs, loss, gradients, inputs, optimizer, model, scaler
        torch.cuda.empty_cache()
    return {'status': 'passed', 'source_identity': source_identity(), 'modes': results,
        'clinical_whitelist': encoder.column_policy, 'excluded_clinical_mutation_invariance': True,
        'fixture_patients': 2, 'clinical_input_reads': 0, 'full_backbone_forward_performed': False,
        'scope': 'Tiny actual CUDA fusion/clinical routing forward-backward and GradScaler update; not full-backbone capacity or clinical qualification'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); plan = load(args.plan)
    if sys.executable != plan['environment']['python']:
        raise ValueError('Qualification interpreter differs')
    source = Path(plan['new_repo']); results = []
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONPATH=str(source / 'src'))
    for item in plan['qualification_tests']:
        test = bound({'path': str(source / item['path']), 'sha256': item['sha256']}, BUNDLE)
        command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(source / 'tests'), '-p', test.name, '-v']
        log = OUTPUT / (test.stem + '.log')
        with log.open('x') as handle:
            result = subprocess.run(command, cwd=source, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        results.append({'test': item, 'argv': command, 'returncode': result.returncode,
                        'log': {'path': str(log), 'sha256': sha(log)}})
        if result.returncode:
            raise ValueError('Source qualification test failed: ' + test.name)
    from shm_check import check_shared_memory
    shared_memory = check_shared_memory()
    result = {'schema_version': 1, 'status': 'passed', 'plan_sha256': sha(args.plan),
        'python_executable': sys.executable, 'python_prefix': sys.prefix, 'python_base_prefix': sys.base_prefix,
        'package_versions': {name: importlib.metadata.version(name) for name in ('torch', 'monai', 'numpy', 'pandas', 'scipy', 'scikit-learn')},
        'source_tests': results, 'cuda_gate': cuda_gate(plan), 'shared_memory': shared_memory,
        'clinical_qualification': 'not_established', 'dependency_installations': 0}
    write(OUTPUT / 'QUALIFICATION.json', result)


if __name__ == '__main__':
    main()
