#!/usr/bin/env python3
"""Small, source-bound phenotype extraction and training-only robustness gates.

No endpoint column is read by this module. The optional fallback is separately
qualified and requires a documented preceding paired feasibility failure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from scipy import ndimage
import scipy
import SimpleITK as sitk
import yaml

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


class SourceBindingError(RuntimeError):
    pass


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def finite_json(value):
    if isinstance(value, dict): return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [finite_json(v) for v in value]
    if isinstance(value, np.generic): return finite_json(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def exclusive_json(path, value):
    raw = (json.dumps(finite_json(value), sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'wb') as stream: stream.write(raw)


def exclusive_frame(path, frame, protocol_sha):
    path = Path(path)
    if path.suffix == '.parquet':
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(frame, preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        metadata[b'protocol_sha256'] = protocol_sha.encode('ascii')
        metadata[b'measure_script_sha256'] = sha256(__file__).encode('ascii')
        table = table.replace_schema_metadata(metadata)
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'wb') as stream: pq.write_table(table, stream, compression='zstd')
    else:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'w') as stream: frame.to_csv(stream, index=False)


def truth(values):
    values = pd.Series(values).fillna(False)
    if values.dtype == bool: return values
    parsed = values.astype(str).str.lower().str.strip()
    if not parsed.isin(['true', 'false', '1', '0']).all():
        raise ValueError('Boolean manifest field contains an unrecognized token')
    return parsed.isin(['true', '1'])


def protocol_and_lock(path, output_root):
    path = Path(path)
    protocol = yaml.safe_load(path.read_text())
    if protocol.get('schema_version') != 1:
        raise ValueError('Unsupported protocol schema')
    if Path(output_root).resolve() != Path(protocol['paths']['run_root']).resolve():
        raise ValueError('Output root differs from frozen protocol')
    lock_path = Path(protocol['paths']['protocol_lock'])
    lock = json.loads(lock_path.read_text())
    protocol_sha = sha256(path)
    if lock['protocol_sha256'] != protocol_sha:
        raise SourceBindingError('Protocol differs from prespecified lock')
    if lock['code_sha256']['measure.py'] != sha256(__file__):
        raise SourceBindingError('Measurement implementation differs from locked code')
    if not lock.get('source_commit'):
        raise SourceBindingError('Source commit is not bound')
    return protocol, protocol_sha, lock


def inventory_index(path):
    table = pd.read_csv(path, dtype={'patient_id': str, 'file': str})
    if table.duplicated(['patient_id', 'file']).any():
        raise SourceBindingError('Image source inventory contains duplicate identities')
    return {(r.patient_id, r.file): {'sha256': r.sha256, 'bytes': int(r.bytes)}
            for r in table.itertuples() if isinstance(r.patient_id, str)}


def geometry(image):
    return {'size_xyz': list(image.GetSize()), 'spacing_xyz': list(image.GetSpacing()),
            'origin_lps': list(image.GetOrigin()), 'direction_lps': list(image.GetDirection())}


def same_geometry(left, right, atol=1e-6):
    return (left.GetDimension() == right.GetDimension() == 3
            and left.GetSize() == right.GetSize()
            and all(np.allclose(a, b, rtol=0, atol=atol) for a, b in [
                (left.GetSpacing(), right.GetSpacing()),
                (left.GetOrigin(), right.GetOrigin()),
                (left.GetDirection(), right.GetDirection())]))


def load_case(imaging_root, patient_id, inventory):
    if patient_id not in {p for p, _ in inventory}:
        raise SourceBindingError('Patient is not present in frozen image inventory')
    root = Path(imaging_root) / patient_id
    bindings = {}
    for name in ['ct.nii.gz', 'mask_primary.nii.gz', 'mask_nodal.nii.gz', 'meta.json']:
        path = root / name
        if (patient_id, name) not in inventory:
            raise SourceBindingError('Frozen source inventory omits required case file: ' + name)
        expected = inventory[(patient_id, name)]
        if not path.is_file() or path.is_symlink() or path.stat().st_size != expected['bytes']:
            raise SourceBindingError('Source path/type/size differs: ' + str(path))
        observed = sha256(path)
        if observed != expected['sha256']:
            raise SourceBindingError('Source SHA-256 differs: ' + str(path))
        bindings[name] = observed
    images = [sitk.ReadImage(str(root / name)) for name in
              ['ct.nii.gz', 'mask_primary.nii.gz', 'mask_nodal.nii.gz']]
    ct, primary, nodal = images
    if ct.GetDimension() != 3 or not same_geometry(ct, primary) or not same_geometry(ct, nodal):
        raise ValueError('CT and required masks do not have matching 3D geometry')
    arrays = [sitk.GetArrayFromImage(image) for image in images]
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError('Nonfinite delivered CT/mask array')
    if arrays[0].min() < 0 or arrays[0].max() > 1:
        raise ValueError('Delivered CT does not use frozen normalized-intensity units')
    for array in arrays[1:]:
        if not np.isin(array, [0, 1]).all(): raise ValueError('Mask labels are not binary 0/1')
    metadata = json.loads((root / 'meta.json').read_text())
    return ct, primary, nodal, arrays[0], arrays[1].astype(bool), arrays[2].astype(bool), metadata, bindings


def support_hash(mask, image):
    header = json.dumps(geometry(image), sort_keys=True, separators=(',', ':')).encode('ascii')
    return hashlib.sha256(header + b'\n' + np.packbits(np.asarray(mask, dtype=np.uint8)).tobytes()).hexdigest()


def independent_linear_quantile(values, probability):
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size == 0: return float('nan')
    index = (ordered.size - 1) * probability
    lower = int(math.floor(index)); upper = int(math.ceil(index))
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower))


def region_values(ct_array, mask, spacing, minimum_voxels=100):
    values = np.asarray(ct_array[mask], dtype=np.float64)
    n = int(values.size)
    volume = n * float(np.prod(spacing)) / 1000.0
    if n == 0:
        return {'n': 0, 'V_cm3': 0.0, 'H': float('nan'), 'median': float('nan'),
                'f0': float('nan'), 'f1': float('nan'), 'iqr_independent_delta': float('nan'),
                'finite': True, 'support_valid': False, 'status': 'empty_region'}
    quantiles = np.quantile(values, [.25, .5, .75], method='linear')
    iqr = float(quantiles[2] - quantiles[0])
    independent = independent_linear_quantile(values, .75) - independent_linear_quantile(values, .25)
    finite = bool(np.isfinite(values).all() and np.isfinite(quantiles).all())
    return {'n': n, 'V_cm3': volume, 'H': iqr, 'median': float(quantiles[1]),
            'f0': float(np.count_nonzero(values == 0) / n),
            'f1': float(np.count_nonzero(values == 1) / n),
            'iqr_independent_delta': float(abs(iqr - independent)), 'finite': finite,
            'support_valid': n >= minimum_voxels and finite and iqr >= 0,
            'status': 'valid' if n >= minimum_voxels and finite and iqr >= 0 else 'fewer_than_minimum_voxels'}


def original_spacing(metadata):
    value = metadata.get('ct_spacing_xyz', '')
    try:
        values = [float(v) for v in value.split('|')] if isinstance(value, str) else list(map(float, value))
        if len(values) != 3 or not all(math.isfinite(v) and v > 0 for v in values): raise ValueError()
        return values
    except (ValueError, TypeError): return [float('nan')] * 3


def independent_volume(mask_image):
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(mask_image)
    return float(statistics.GetPhysicalSize(1) / 1000.0) if statistics.HasLabel(1) else 0.0


def extraction_case(task):
    patient_id, imaging_root, inventory, settings = task
    try:
        ct, primary, nodal, array, p, n, metadata, bindings = load_case(imaging_root, patient_id, inventory)
        regions = [region_values(array, mask, ct.GetSpacing(), settings['minimum_voxels']) for mask in [p, n]]
        rp, rn = regions
        row = {'patient_id': patient_id, 'HP': rp['H'], 'HN': rn['H'],
            'M': (rp['H'] + rn['H']) / 2, 'D': (rn['H'] - rp['H']) / 2,
            'VP_cm3': rp['V_cm3'], 'VN_cm3': rn['V_cm3'], 'nP': rp['n'], 'nN': rn['n'],
            'medianP': rp['median'], 'medianN': rn['median'], 'f0P': rp['f0'], 'f1P': rp['f1'],
            'f0N': rn['f0'], 'f1N': rn['f1'], 'PT_status': rp['status'], 'LN_status': rn['status'],
            'original_x_spacing_mm': original_spacing(metadata)[0],
            'original_y_spacing_mm': original_spacing(metadata)[1],
            'original_z_spacing_mm': original_spacing(metadata)[2],
            'voxel_volume_mm3': float(np.prod(ct.GetSpacing())),
            'ct_file_sha256': bindings['ct.nii.gz'], 'PT_mask_file_sha256': bindings['mask_primary.nii.gz'],
            'LN_mask_file_sha256': bindings['mask_nodal.nii.gz'], 'metadata_file_sha256': bindings['meta.json'],
            'PT_support_sha256': support_hash(p, ct), 'LN_support_sha256': support_hash(n, ct),
            'measurement_error': ''}
        pvol_delta = abs(independent_volume(primary) - row['VP_cm3'])
        nvol_delta = abs(independent_volume(nodal) - row['VN_cm3'])
        p_valid = bool(rp['support_valid'] and rp['iqr_independent_delta'] <= settings['iqr_validation_tolerance']
                       and pvol_delta <= 1e-9)
        n_valid = bool(rn['support_valid'] and rn['iqr_independent_delta'] <= settings['iqr_validation_tolerance']
                       and nvol_delta <= 1e-9)
        reconstruct = max(abs(row['HP'] - (row['M'] - row['D'])), abs(row['HN'] - (row['M'] + row['D']))) if p_valid and n_valid else float('nan')
        paired_valid = bool(p_valid and n_valid and reconstruct <= settings['iqr_validation_tolerance'])
        row.update(paired_measurement_valid=paired_valid, primary_only_measurement_valid=p_valid)
        validation = {'patient_id': patient_id, 'paired_measurement_valid': paired_valid,
            'primary_only_measurement_valid': p_valid, 'PT_independent_IQR_delta': rp['iqr_independent_delta'],
            'LN_independent_IQR_delta': rn['iqr_independent_delta'], 'PT_independent_volume_delta_cm3': pvol_delta,
            'LN_independent_volume_delta_cm3': nvol_delta, 'M_D_reconstruction_delta': reconstruct,
            'CT_PT_LN_geometry_equal': True, 'binary_masks': True, 'finite_CT': True,
            'source_hashes_verified': True, 'measurement_error': ''}
        return row, validation
    except SourceBindingError:
        raise
    except Exception as error:
        message = type(error).__name__ + ': ' + str(error)
        return ({'patient_id': patient_id, 'paired_measurement_valid': False,
                 'primary_only_measurement_valid': False, 'measurement_error': message},
                {'patient_id': patient_id, 'paired_measurement_valid': False,
                 'primary_only_measurement_valid': False, 'measurement_error': message})


def edge_contact(mask):
    return bool(any(np.any(face) for face in [mask[0], mask[-1], mask[:, 0], mask[:, -1], mask[:, :, 0], mask[:, :, -1]]))


def physical_morphology(mask, spacing_xyz, radius_mm, mode):
    """Discrete Euclidean morphology on padded physical grid, never clipped silently."""
    sampling = np.asarray(spacing_xyz, dtype=float)[::-1]
    padding = np.ceil(radius_mm / sampling).astype(int) + 2
    if not np.any(mask):
        return mask.copy(), {'outside_delivered_voxels': 0, 'padding_zyx': padding.tolist(),
                             'touches_array_boundary': False}
    padded = np.pad(mask, tuple((int(n), int(n)) for n in padding), mode='constant', constant_values=False)
    if mode == 'dilation':
        transformed = ndimage.distance_transform_edt(~padded, sampling=sampling) <= radius_mm + 1e-12
    elif mode == 'erosion':
        transformed = padded & (ndimage.distance_transform_edt(padded, sampling=sampling) > radius_mm + 1e-12)
    else: raise ValueError('Unknown morphology mode')
    core_slice = tuple(slice(int(pad), int(pad + size)) for pad, size in zip(padding, mask.shape))
    core = transformed[core_slice].copy()
    outside = int(np.count_nonzero(transformed) - np.count_nonzero(core))
    return core, {'outside_delivered_voxels': outside, 'padding_zyx': padding.tolist(),
                  'touches_array_boundary': edge_contact(core)}


def coarsen_images(ct, primary, nodal, target_mm):
    """Resample voxel-edge-aligned fields without changing physical outer edges."""
    old_spacing = np.asarray(ct.GetSpacing(), dtype=float)
    target_spacing = np.repeat(float(target_mm), 3)
    new_size_float = np.asarray(ct.GetSize()) * old_spacing / target_spacing
    if not np.allclose(new_size_float, np.rint(new_size_float), atol=1e-6, rtol=0):
        raise ValueError('Requested grid cannot exactly preserve delivered voxel-edge FOV')
    new_size = np.rint(new_size_float).astype(int)
    if (new_size <= 0).any(): raise ValueError('Empty target grid')
    direction = np.asarray(ct.GetDirection()).reshape(3, 3)
    new_origin = np.asarray(ct.GetOrigin()) + direction @ ((target_spacing - old_spacing) / 2)
    result = []
    for i, image in enumerate([ct, primary, nodal]):
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize([int(n) for n in new_size]); resampler.SetOutputSpacing(target_spacing.tolist())
        resampler.SetOutputOrigin(new_origin.tolist()); resampler.SetOutputDirection(ct.GetDirection())
        resampler.SetTransform(sitk.Transform()); resampler.SetDefaultPixelValue(0)
        resampler.SetInterpolator(sitk.sitkLinear if i == 0 else sitk.sitkNearestNeighbor)
        resampler.SetOutputPixelType(sitk.sitkFloat32 if i == 0 else sitk.sitkUInt8)
        result.append(resampler.Execute(image))
    return result


def sphericity(ct, mask_image):
    import radiomics
    from radiomics.shape import RadiomicsShape
    if radiomics.__version__.lstrip('v') != '3.0.1':
        raise SourceBindingError('Fallback sphericity requires exact PyRadiomics 3.0.1')
    shape = RadiomicsShape(ct, mask_image, label=1)
    value = float(shape.getSphericityFeatureValue())
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid mesh-based sphericity')
    return value


def array_mask_image(mask, reference):
    image = sitk.GetImageFromArray(np.asarray(mask, dtype=np.uint8)); image.CopyInformation(reference)
    return image


def perturbation_case(task):
    patient_id, imaging_root, inventory, settings, fallback = task
    ct, primary, nodal, ct_array, p, n, metadata, bindings = load_case(imaging_root, patient_id, inventory)
    rows = []
    conditions = settings['primary_conditions'] + settings['stress_conditions']
    for condition in conditions:
        record = {'patient_id': patient_id, 'condition': condition,
                  'gating': condition in settings['primary_conditions'], 'available': False,
                  'reason': '', 'CT_file_sha256': bindings['ct.nii.gz']}
        try:
            reference = ct; image_array = ct_array
            outside_p = 0; outside_n = 0
            if condition in ['erosion_1mm', 'dilation_1mm']:
                mode = 'erosion' if condition.startswith('erosion') else 'dilation'
                pp, pinfo = physical_morphology(p, ct.GetSpacing(), settings['morphology_mm'], mode)
                nn, ninfo = physical_morphology(n, ct.GetSpacing(), settings['morphology_mm'], mode) if not fallback else (n, {'outside_delivered_voxels': 0})
                outside_p = pinfo['outside_delivered_voxels']; outside_n = ninfo['outside_delivered_voxels']
            elif condition in ['isotropic_1mm', 'isotropic_2mm']:
                target = 1.0 if condition == 'isotropic_1mm' else 2.0
                reference, pi, ni = coarsen_images(ct, primary, nodal, target)
                image_array = sitk.GetArrayFromImage(reference)
                pp, nn = sitk.GetArrayFromImage(pi).astype(bool), sitk.GetArrayFromImage(ni).astype(bool)
            else: raise ValueError('Unknown prespecified perturbation condition')
            n_p = int(np.count_nonzero(pp)); n_n = int(np.count_nonzero(nn))
            overlap = int(np.count_nonzero(pp & nn))
            record.update(nP=n_p, nN=n_n, outside_PT_voxels=outside_p, outside_LN_voxels=outside_n,
                          PT_LN_overlap_voxels=overlap, PT_touches_boundary=edge_contact(pp),
                          LN_touches_boundary=edge_contact(nn), grid_spacing_xyz=list(reference.GetSpacing()))
            reasons = []
            if n_p == 0 or (not fallback and n_n == 0): reasons.append('empty_region')
            if n_p < settings['minimum_voxels'] or (not fallback and n_n < settings['minimum_voxels']): reasons.append('fewer_than_100_voxels')
            if outside_p > 0 or (not fallback and outside_n > 0): reasons.append('support_crosses_delivered_boundary')
            if not fallback and overlap > 0: reasons.append('PT_LN_overlap')
            if reasons:
                record['reason'] = ';'.join(reasons)
            elif fallback:
                record.update(available=True, phiP=sphericity(reference, array_mask_image(pp, reference)),
                              PT_support_sha256=support_hash(pp, reference))
            else:
                rp = region_values(image_array, pp, reference.GetSpacing(), settings['minimum_voxels'])
                rn = region_values(image_array, nn, reference.GetSpacing(), settings['minimum_voxels'])
                record.update(available=True, HP=rp['H'], HN=rn['H'], M=(rp['H']+rn['H'])/2,
                              D=(rn['H']-rp['H'])/2, PT_support_sha256=support_hash(pp, reference),
                              LN_support_sha256=support_hash(nn, reference))
        except SourceBindingError:
            raise
        except Exception as error:
            record['reason'] = type(error).__name__ + ': ' + str(error)
        rows.append(record)
    return rows


def icc_absolute_single(left, right):
    """Two-way absolute-agreement single-measure ICC(A,1), including bias."""
    values = np.column_stack([left, right]).astype(float)
    n, k = values.shape
    if n < 2 or not np.isfinite(values).all(): return float('nan')
    row_mean = values.mean(axis=1); column_mean = values.mean(axis=0); grand = values.mean()
    ms_rows = k * np.sum((row_mean-grand)**2) / (n-1)
    ms_columns = n * np.sum((column_mean-grand)**2) / (k-1)
    ms_error = np.sum((values-row_mean[:, None]-column_mean[None, :]+grand)**2) / ((n-1)*(k-1))
    denominator = ms_rows + (k-1)*ms_error + k*(ms_columns-ms_error)/n
    if denominator <= 0 or not math.isfinite(float(denominator)): return float('nan')
    return float((ms_rows-ms_error)/denominator)


def agreement_summary(left, right, baseline_sd, settings, seed):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    estimate = icc_absolute_single(left, right)
    rng = np.random.default_rng(seed); samples = []
    for _ in range(settings['bootstrap_resamples']):
        indices = rng.integers(0, len(left), size=len(left)) if len(left) else np.array([], dtype=int)
        value = icc_absolute_single(left[indices], right[indices])
        if math.isfinite(value): samples.append(value)
    valid_fraction = len(samples) / settings['bootstrap_resamples']
    interval = np.quantile(samples, [.025, .975], method='linear').tolist() if samples else [float('nan')] * 2
    sd_valid = math.isfinite(baseline_sd) and baseline_sd > settings['zero_sd_tolerance']
    change = float(np.median(np.abs(right-left))) if len(left) else float('nan')
    standardized = change / baseline_sd if sd_valid else float('nan')
    checks = {'nonzero_baseline_training_SD': sd_valid,
        'finite_ICC': math.isfinite(estimate), 'ICC_at_least_threshold': math.isfinite(estimate) and estimate >= settings['icc_min'],
        'lower95_at_least_threshold': math.isfinite(interval[0]) and interval[0] >= settings['icc_lower95_min'],
        'bootstrap_valid_fraction': valid_fraction >= settings['minimum_bootstrap_valid_fraction'],
        'median_absolute_change_within_training_SD_threshold': math.isfinite(standardized) and standardized <= settings['median_change_sd_max']}
    return {'ICC_A_1': estimate, 'ICC_lower95': interval[0], 'ICC_upper95': interval[1],
            'bootstrap_requested': settings['bootstrap_resamples'], 'bootstrap_finite': len(samples),
            'bootstrap_valid_fraction': valid_fraction, 'bootstrap_seed': seed,
            'baseline_training_SD': baseline_sd, 'median_absolute_change': change,
            'median_absolute_change_in_baseline_training_SD': standardized,
            'checks': checks, 'passed': all(checks.values())}


def run_extraction(protocol, protocol_sha, output_root, workers):
    paths = protocol['paths']; settings = protocol['measurement']
    manifest_path = output_root / 'preparation/cohort_manifest_pre_measurement.csv'
    manifest = pd.read_csv(manifest_path, usecols=['patient_id', 'split', 'paired_measurement_candidate', 'primary_only_measurement_candidate'], dtype={'patient_id': str})
    if manifest.patient_id.duplicated().any(): raise ValueError('Duplicate cohort IDs')
    candidates = manifest[truth(manifest.paired_measurement_candidate) | truth(manifest.primary_only_measurement_candidate)]
    inventory = inventory_index(paths['image_source_inventory_csv'])
    destination = output_root / 'measurements'; destination.mkdir(mode=0o700, exist_ok=False)
    tasks = [(pid, paths['imaging_root'], inventory, settings) for pid in candidates.patient_id]
    results = []
    with (destination / 'extraction_progress.jsonl').open('x') as progress:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for row, validation in pool.map(extraction_case, tasks):
                    results.append((row, validation)); progress.write(json.dumps(finite_json({'features': row, 'validation': validation}), allow_nan=False)+'\n'); progress.flush()
        else:
            for task in tasks:
                row, validation = extraction_case(task); results.append((row, validation))
                progress.write(json.dumps(finite_json({'features': row, 'validation': validation}), allow_nan=False)+'\n'); progress.flush()
    features = pd.DataFrame([r[0] for r in results]); validation = pd.DataFrame([r[1] for r in results])
    if not features.empty:
        features = features.merge(candidates[['patient_id', 'split']], on='patient_id', validate='one_to_one')
    for suffix in ['csv', 'parquet']: exclusive_frame(destination / ('phenotype_features.'+suffix), features, protocol_sha)
    exclusive_frame(destination / 'measurement_validation.csv', validation, protocol_sha)
    exclusive_json(destination / 'EXTRACTION.json', {'at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'completed', 'protocol_sha256': protocol_sha, 'code_sha256': sha256(__file__),
        'premeasurement_manifest_sha256': sha256(manifest_path), 'inventory_sha256': sha256(paths['image_source_inventory_csv']),
        'candidate_n': len(candidates), 'features_sha256': sha256(destination/'phenotype_features.csv'),
        'validation_sha256': sha256(destination/'measurement_validation.csv'), 'endpoint_columns_read': [],
        'quantile_method': 'linear', 'IQR_independent_method': 'explicit_sorted_order_statistic_linear_interpolation',
        'volume_independent_method': 'SimpleITK_LabelShapeStatistics_GetPhysicalSize',
        'support_hash_definition': 'geometry_JSON_newline_numpy_packbits_binary_mask',
        'software': software_versions(), 'new_model_fits': 0, 'remote_calls': 0})
    return {'stage': 'extract', 'candidate_n': len(candidates), 'output': str(destination)}


def software_versions():
    import pyarrow
    return {'Python': platform.python_version(), 'NumPy': np.__version__, 'SciPy': scipy.__version__,
            'pandas': pd.__version__, 'SimpleITK': sitk.Version_VersionString(),
            'pyarrow': pyarrow.__version__, 'PyYAML': yaml.__version__}


def fallback_allowed(output_root, protocol_sha):
    for name in ['PRIMARY_ASSOCIATION_STARTED.json', 'primary_result.json', 'FALLBACK_ASSOCIATION_STARTED.json']:
        if (output_root/'statistics'/name).exists():
            raise ValueError('Fallback activation is forbidden after an association stage has started')
    paired = output_root / 'qualification/paired_decision.json'
    if paired.exists():
        decision = json.loads(paired.read_text())
        if decision['protocol_sha256'] != protocol_sha: raise SourceBindingError('Prior paired decision protocol mismatch')
        if decision['passed'] is False: return {'kind': 'paired_measurement_failed', 'path': str(paired), 'sha256': sha256(paired)}
    design = output_root / 'statistics/design_assessment.json'
    if design.exists():
        decision = json.loads(design.read_text())
        if decision.get('protocol_sha256') != protocol_sha: raise SourceBindingError('Prior design protocol mismatch')
        if decision.get('passed') is False or decision.get('adequate') is False:
            return {'kind': 'paired_design_failed', 'path': str(design), 'sha256': sha256(design)}
    raise ValueError('Fallback requires a bound paired measurement/design failure before association')


def run_fallback_extraction(protocol, protocol_sha, output_root):
    reason = fallback_allowed(output_root, protocol_sha)
    manifest = pd.read_csv(output_root/'preparation/cohort_manifest.csv', usecols=['patient_id', 'split', 'primary_only_eligible'], dtype={'patient_id': str})
    candidates = manifest[truth(manifest.primary_only_eligible)]
    inventory = inventory_index(protocol['paths']['image_source_inventory_csv']); rows = []
    for pid in candidates.patient_id:
        ct, primary, nodal, array, p, n, metadata, bindings = load_case(protocol['paths']['imaging_root'], pid, inventory)
        row = {'patient_id': pid, 'phiP': float('nan'), 'valid': False, 'error': ''}
        try: row.update(phiP=sphericity(ct, primary), valid=True)
        except SourceBindingError: raise
        except Exception as error: row['error'] = type(error).__name__+': '+str(error)
        row.update(PT_support_sha256=support_hash(p, ct), ct_file_sha256=bindings['ct.nii.gz'], PT_mask_file_sha256=bindings['mask_primary.nii.gz'])
        rows.append(row)
    frame = pd.DataFrame(rows)
    for suffix in ['csv', 'parquet']: exclusive_frame(output_root/'measurements'/('primary_only_features.'+suffix), frame, protocol_sha)
    exclusive_json(output_root/'measurements/FALLBACK_EXTRACTION.json', {'protocol_sha256': protocol_sha,
        'activation': reason, 'definition': protocol['measurement']['fallback_definition'],
        'candidate_n': len(candidates), 'valid_n': int(frame.valid.sum()), 'feature_file_sha256': sha256(output_root/'measurements/primary_only_features.csv'),
        'endpoint_columns_read': [], 'no_primary_association_inspected': True, 'software': software_versions()})
    return {'stage': 'fallback-extract', 'candidate_n': len(candidates)}


def run_qualification(protocol, protocol_sha, output_root, workers, fallback=False):
    settings = protocol['measurement']; paths = protocol['paths']
    kind = 'primary_only' if fallback else 'paired'
    activation = fallback_allowed(output_root, protocol_sha) if fallback else None
    manifest_path = output_root/'preparation/cohort_manifest.csv'
    manifest_sha = sha256(manifest_path)
    flag = kind+'_eligible'
    manifest = pd.read_csv(manifest_path, usecols=['patient_id', 'split', flag], dtype={'patient_id': str})
    selected = manifest[(manifest.split=='train') & truth(manifest[flag])]
    if selected.patient_id.duplicated().any(): raise ValueError('Duplicate training patient IDs')
    baseline_file = output_root/'measurements/phenotype_features.csv'
    base_features_sha = sha256(baseline_file)
    shape_path = output_root/'measurements/primary_only_features.csv'
    shape_sha = sha256(shape_path) if fallback else None
    columns = ['patient_id', 'M', 'D', 'paired_measurement_valid'] if not fallback else ['patient_id', 'HP', 'VP_cm3']
    baseline = pd.read_csv(baseline_file, usecols=columns, dtype={'patient_id': str})
    if fallback:
        phi = pd.read_csv(shape_path, usecols=['patient_id', 'phiP', 'valid'], dtype={'patient_id': str})
        baseline = baseline.merge(phi, on='patient_id', validate='one_to_one')
    baseline = selected[['patient_id']].merge(baseline, on='patient_id', how='left', validate='one_to_one')
    features = ['phiP'] if fallback else ['M', 'D']
    baseline_valid = len(baseline) >= 2 and all(np.isfinite(baseline[f]).all() for f in features)
    if not fallback: baseline_valid = bool(baseline_valid and truth(baseline.paired_measurement_valid).all())
    else: baseline_valid = bool(baseline_valid and truth(baseline.valid).all())
    destination = output_root/'qualification'; destination.mkdir(mode=0o700, exist_ok=True)
    decision_path = destination/(kind+'_decision.json')
    if decision_path.exists(): raise FileExistsError(str(decision_path))
    inventory = inventory_index(paths['image_source_inventory_csv'])
    tasks = [(pid, paths['imaging_root'], inventory, settings, fallback) for pid in selected.patient_id]
    rows = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(perturbation_case, tasks): rows.extend(result)
    else:
        for task in tasks: rows.extend(perturbation_case(task))
    results = pd.DataFrame(rows)
    output_file = destination/(kind+'_perturbation_records.csv')
    exclusive_frame(output_file, results, protocol_sha)
    conditions = []
    for condition_index, condition in enumerate(settings['primary_conditions']+settings['stress_conditions']):
        subset = results[results.condition==condition] if not results.empty else pd.DataFrame(columns=['available','patient_id'])
        valid = subset[truth(subset.available)]
        matched = baseline.merge(valid, on='patient_id', validate='one_to_one', suffixes=('_baseline','_perturbed'))
        coverage = len(valid)/len(selected) if len(selected) else 0.0
        measures = {}
        for feature_index, feature in enumerate(features):
            sd = float(baseline[feature].std(ddof=1)) if len(baseline)>1 else float('nan')
            seed = (settings['fallback_bootstrap_seed'] if fallback else settings['bootstrap_seed']) + condition_index*100 + feature_index
            measures[feature] = agreement_summary(matched[feature+'_baseline'].to_numpy() if len(matched) else [],
                matched[feature+'_perturbed'].to_numpy() if len(matched) else [], sd, settings, seed)
        checks = {'baseline_training_numerical_validation': baseline_valid,
                  'coverage_at_least_threshold': coverage >= settings['coverage_min']}
        checks.update({feature+'_agreement': value['passed'] for feature,value in measures.items()})
        conditions.append({'condition': condition, 'gating': condition in settings['primary_conditions'],
            'baseline_training_n': len(selected), 'evaluable_n': len(valid), 'coverage': coverage,
            'unavailable_reasons': subset.loc[~truth(subset.available),'reason'].value_counts().to_dict() if len(subset) else {},
            'measurements': measures, 'checks': checks, 'passed': all(checks.values())})
    gating_conditions = [condition for condition in conditions if condition['gating']]
    if sha256(manifest_path) != manifest_sha or sha256(baseline_file) != base_features_sha:
        raise SourceBindingError('Final cohort/base measurements changed during qualification')
    if fallback and sha256(shape_path) != shape_sha:
        raise SourceBindingError('Sphericity measurements changed during qualification')
    decision = {'schema_version': 1, 'at_utc': datetime.now(timezone.utc).isoformat(),
        'analysis': kind, 'passed': bool(baseline_valid and all(c['passed'] for c in gating_conditions)),
        'protocol_sha256': protocol_sha, 'code_sha256': sha256(__file__),
        'cohort_manifest_sha256': manifest_sha,
        'baseline_feature_file_sha256': shape_sha if fallback else base_features_sha,
        'supporting_measurement_file_sha256': base_features_sha if fallback else None,
        'perturbation_records_sha256': sha256(output_file), 'baseline_train_n': len(selected),
        'baseline_training_numerical_validation': baseline_valid, 'qualification_partition': 'original_train_only',
        'baseline_train_ids_sha256': hashlib.sha256(('\n'.join(sorted(selected.patient_id))+'\n').encode()).hexdigest(),
        'primary_conditions': settings['primary_conditions'], 'stress_conditions': settings['stress_conditions'],
        'stress_2mm_is_gating': False, 'conditions': conditions,
        'endpoint_columns_read': [], 'new_OS_associations_fitted': 0,
        'no_feature_screening': True, 'activation': activation,
        'interpretation': 'Perturbation robustness only; no independent reader/test-retest/biological validity claim',
        'software': software_versions()}
    for feature in features: decision['baseline_train_SD_'+feature] = float(baseline[feature].std(ddof=1)) if len(baseline)>1 else float('nan')
    if fallback: decision['feature_name'] = 'phiP'
    exclusive_json(decision_path, decision)
    return {'stage': 'fallback-qualify' if fallback else 'qualify', 'passed': decision['passed'], 'baseline_train_n': len(selected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--stage', choices=['extract', 'qualify', 'fallback-extract', 'fallback-qualify'], required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers not in [1,2]: raise ValueError('CPU measurement stages admit one or two workers')
    protocol, protocol_sha, lock = protocol_and_lock(args.protocol, args.output_root)
    if args.stage == 'extract': result = run_extraction(protocol, protocol_sha, args.output_root, args.workers)
    elif args.stage == 'qualify': result = run_qualification(protocol, protocol_sha, args.output_root, args.workers)
    elif args.stage == 'fallback-extract': result = run_fallback_extraction(protocol, protocol_sha, args.output_root)
    else: result = run_qualification(protocol, protocol_sha, args.output_root, args.workers, fallback=True)
    print(json.dumps(finite_json(result), sort_keys=True, allow_nan=False))


if __name__ == '__main__': main()
