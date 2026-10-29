#!/usr/bin/env python3
"""Prepare a separate, source-bound phenotype cohort without fitting OS models.

Stages are explicit and never overwrite a completed artifact. Screening uses
only CT/geometry and retained series identifiers. Outcome values are used only
for eligibility and aggregate flow counts, never to select source records.
"""
from __future__ import annotations

import argparse
import collections
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
import SimpleITK as sitk
import yaml

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def plain(value):
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, np.generic): return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def json_write(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(plain(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    path.chmod(0o600)


def read_table(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def table_write(frame, path):
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    frame.to_csv(path, index=False)
    path.chmod(0o600)


def parquet_write(frame, path):
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    frame.to_parquet(path, engine='pyarrow', index=False)
    assert pd.read_parquet(path).shape == frame.shape
    path.chmod(0o600)


def truth(value):
    return str(value).strip().lower() in ('true', '1', '1.0')


def numeric(value):
    try: return float(value)
    except (ValueError, TypeError): return float('nan')


def unique_table(path, identifier):
    frame = read_table(path)
    if frame[identifier].eq('').any() or frame[identifier].duplicated().any():
        raise ValueError('Nonunique or empty literal identifiers: ' + str(path))
    return frame.set_index(identifier, drop=False)


def load_protocol(path, output_root=None, require_lock=True):
    path = Path(path).resolve()
    protocol = yaml.safe_load(path.read_text())
    if protocol.get('schema_version') != 1: raise ValueError('Unsupported protocol')
    root = Path(output_root or protocol['paths']['run_root']).resolve()
    if root != Path(protocol['paths']['run_root']).resolve():
        raise ValueError('Output root differs from the fixed protocol')
    if require_lock:
        lock = json.loads(Path(protocol['paths']['protocol_lock']).read_text())
        if lock['protocol_sha256'] != sha256(path): raise ValueError('Protocol lock differs')
        if lock['code_sha256']['prepare.py'] != sha256(__file__):
            raise ValueError('Preparation code differs from its prospective lock')
    return protocol, root


def source_ref(path, kind=None):
    path = Path(path)
    record = {'path': str(path.resolve()), 'bytes': path.stat().st_size,
              'sha256': sha256(path), 'kind': kind or path.suffix.lstrip('.')}
    if path.suffix == '.csv':
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.reader(stream); columns = next(reader, [])
            record.update(rows=sum(1 for _ in reader), columns=len(columns), schema=columns)
    return record


def source_manifest(protocol, root):
    output = root / 'preparation'; output.mkdir(mode=0o700, exist_ok=True)
    destination = output / 'source_manifest.json'
    if destination.exists(): raise FileExistsError(destination)
    paths = protocol['paths']; image_root = Path(paths['imaging_root'])
    inventory = read_table(paths['image_source_inventory_csv'])
    qc = unique_table(paths['imaging_qc'], 'patient_id')
    image_records = []
    for row in inventory.to_dict('records'):
        patient = row['patient_id']
        path = image_root / patient / row['file'] if patient else image_root / row['file']
        observed = source_ref(path, 'delivered_image_or_metadata')
        if observed['sha256'] != row['sha256'] or observed['bytes'] != int(row['bytes']):
            raise ValueError('Source changed since the all-pixel audit: ' + str(path))
        observed.update(patient_id=patient, file=row['file'])
        if patient and row['file'].endswith('.nii.gz'):
            prefix = {'ct.nii.gz': 'ct', 'mask_primary.nii.gz': 'primary',
                      'mask_nodal.nii.gz': 'nodal', 'mask_union.nii.gz': 'union'}[row['file']]
            source = qc.loc[patient]
            observed['audited_header_schema'] = {key: source[prefix + '_' + key]
                for key in ('size_xyz', 'spacing_xyz', 'origin_lps_xyz', 'direction_lps', 'dtype')}
            observed['decoded_voxel_sha256_from_full_pixel_audit'] = source[prefix + '_voxel_sha256']
        image_records.append(observed)
    fixed = {}
    for name, path in paths.items():
        source = Path(str(path))
        if source.is_file() and name not in ('cohort_manifest_csv', 'phenotype_features_csv', 'protocol_lock'):
            fixed[name] = source_ref(source)
    retained_timing_files = []
    if paths.get('timing_retained_metadata_root'):
        for source in sorted(Path(paths['timing_retained_metadata_root']).rglob('*.json')):
            retained_timing_files.append(source_ref(source, 'retained_metadata_clinical_date_key_search_only'))
    result = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'study_id': protocol['study_id'], 'protocol_sha256': json.loads(Path(paths['protocol_lock']).read_text())['protocol_sha256'],
        'fixed_inputs': fixed, 'delivered_files': image_records,
        'retained_timing_metadata_files': retained_timing_files,
        'delivered_case_count': len(qc), 'delivered_file_count': len(image_records),
        'delivered_bytes': sum(r['bytes'] for r in image_records),
        'all_delivered_file_hashes_freshly_verified_against_pixel_audit': True,
        'header_and_decoded_pixel_properties_reused_from_matching_hash_bound_full_audit': True,
        'software': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'SimpleITK': sitk.Version_VersionString()},
        'source_mutation': False, 'split_mutation': False, 'model_fits': 0, 'remote_calls': 0}
    json_write(destination, result)
    print(json.dumps({'stage': 'sources', 'files_verified': len(image_records), 'cases': len(qc)}, sort_keys=True), flush=True)


def verify_fixed_inputs(protocol, root, verify_meta=False):
    manifest = json.loads((root / 'preparation/source_manifest.json').read_text())
    for name, record in manifest['fixed_inputs'].items():
        if sha256(record['path']) != record['sha256']:
            raise ValueError('Fixed source changed after source freeze: ' + name)
    if verify_meta:
        for record in manifest['delivered_files']:
            if record['file'] == 'meta.json' and sha256(record['path']) != record['sha256']:
                raise ValueError('Per-case source metadata changed: ' + record['patient_id'])
        for record in manifest.get('retained_timing_metadata_files', []):
            if sha256(record['path']) != record['sha256']:
                raise ValueError('Retained timing metadata changed: ' + record['path'])


def image_bounds(image):
    size = np.asarray(image.GetSize(), dtype=float)
    corners = np.asarray(list(itertools.product(*[(0., n - 1.) for n in size])))
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    world = np.asarray(image.GetOrigin()) + (corners * image.GetSpacing()) @ direction.T
    return world.min(axis=0), world.max(axis=0)


def lattice_points(lower, upper, spacing):
    lower = np.ceil(np.asarray(lower) / spacing).astype(np.int32)
    upper = np.floor(np.asarray(upper) / spacing).astype(np.int32)
    if (upper < lower).any(): return np.empty((0, 3), dtype=np.int32)
    xyz = np.meshgrid(*(np.arange(a, b + 1, dtype=np.int32) for a, b in zip(lower, upper)), indexing='ij')
    return np.column_stack([value.ravel() for value in xyz])


def sample_image(image, array, world):
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    xyz = ((world - image.GetOrigin()) @ np.linalg.inv(direction).T) / image.GetSpacing()
    coords = xyz[:, ::-1].T
    return map_coordinates(array, coords, order=1, mode='constant', cval=np.nan, prefilter=False)


def informative(values, settings):
    values = np.asarray(values)
    good = np.isfinite(values)
    low, high = settings.get('intensity_range', [0., 1.])
    good &= (values >= low) & (values <= high)
    tolerance = float(settings['constant_exclusion_tolerance'])
    for value in settings['excluded_constant_values']:
        good &= np.abs(values - value) > tolerance
    return good


def fingerprint(image, settings):
    spacing = float(settings['global_lattice_spacing_mm'])
    points = lattice_points(*image_bounds(image), spacing)
    array = sitk.GetArrayFromImage(image)
    values = sample_image(image, array, points.astype(float) * spacing)
    keep = informative(values, settings)
    points, values = points[keep], values[keep]
    quantized = np.rint(values / float(settings['fingerprint_quantization'])).astype(np.int32)
    keys = np.column_stack((points, quantized)).astype(np.int32)
    return keys, values


def matching_counts(keys_by_case):
    n = len(keys_by_case)
    counts = np.zeros((n, n), dtype=np.uint32)
    if not any(len(k) for k in keys_by_case): return counts
    keys = np.concatenate(keys_by_case)
    identifiers = np.concatenate([np.full(len(k), i, dtype=np.uint16) for i, k in enumerate(keys_by_case)])
    order = np.lexsort(tuple(keys[:, col] for col in (3, 2, 1, 0)))
    keys, identifiers = keys[order], identifiers[order]
    boundaries = np.r_[0, np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1, len(keys)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 2: continue
        members = np.unique(identifiers[start:end])
        for position, left in enumerate(members[:-1]):
            right = members[position + 1:]
            counts[left, right] += 1
    return counts + counts.T


def review_pair(left_path, right_path, settings):
    left = sitk.ReadImage(str(left_path)); right = sitk.ReadImage(str(right_path))
    low_a, high_a = image_bounds(left); low_b, high_b = image_bounds(right)
    # Review samples are denser than candidate generation and are not quantized.
    spacing = float(settings.get('review_lattice_spacing_mm', 2.0))
    points = lattice_points(np.maximum(low_a, low_b), np.minimum(high_a, high_b), spacing)
    world = points.astype(float) * spacing
    a = sample_image(left, sitk.GetArrayFromImage(left), world)
    b = sample_image(right, sitk.GetArrayFromImage(right), world)
    keep = informative(a, settings) & informative(b, settings)
    a, b = a[keep].astype(float), b[keep].astype(float)
    n = len(a)
    sufficient = n >= int(settings['review_min_common_informative_points'])
    delta = np.abs(a - b)
    fraction = float(np.mean(delta <= float(settings['review_equal_tolerance']))) if n else None
    mae = float(np.mean(delta)) if n else None
    correlation = float(np.corrcoef(a, b)[0, 1]) if n > 1 and np.std(a) > 0 and np.std(b) > 0 else None
    near_equal = fraction is not None and fraction >= float(settings['review_min_nearly_equal_fraction'])
    correlated = correlation is not None and correlation >= float(settings['review_min_correlation']) and mae <= float(settings['review_max_mean_absolute_difference'])
    shared = bool(sufficient and (near_equal or correlated))
    return {'review_lattice_spacing_mm': spacing, 'common_informative_points': n,
        'sufficient_support': sufficient, 'fraction_absolute_difference_within_tolerance': fraction,
        'mean_absolute_difference': mae, 'median_absolute_difference': float(np.median(delta)) if n else None,
        'maximum_absolute_difference': float(np.max(delta)) if n else None,
        'pearson_correlation': correlation, 'shared_source_content_threshold_met': shared,
        'disposition': 'unresolved_shared_content' if shared else
            'unresolved_insufficient_review_support' if not sufficient else 'review_below_prespecified_similarity_threshold',
        'analytically_exclude_both': shared or not sufficient,
        'outcome_values_used': False}


def run_screen(protocol, root):
    verify_fixed_inputs(protocol, root)
    destination = root / 'screen'; destination.mkdir(mode=0o700, exist_ok=False)
    paths = protocol['paths']; settings = protocol['source_integrity_screen']
    qc = unique_table(paths['imaging_qc'], 'patient_id')
    ids = sorted(qc.index.tolist()); keys = []; supports = []; uid = {}
    for number, patient in enumerate(ids, 1):
        path = Path(paths['imaging_root']) / patient / 'ct.nii.gz'
        if sha256(path) != qc.loc[patient, 'ct_file_sha256']:
            raise ValueError('Screen CT hash differs from the full-pixel audit: ' + patient)
        image = sitk.ReadImage(str(path)); key, values = fingerprint(image, settings)
        keys.append(key); supports.append(len(key)); uid[patient] = qc.loc[patient, 'ct_series_uid']
        if number % 50 == 0: print('screen fingerprints', number, '/', len(ids), flush=True)
    counts = matching_counts(keys)
    candidates = []; score_rows = []
    for i, left in enumerate(ids):
        for j in range(i + 1, len(ids)):
            right = ids[j]; number = int(counts[i, j]); denominator = min(supports[i], supports[j])
            fraction = number / denominator if denominator else 0.
            candidate = number >= int(settings['candidate_min_matched_points']) and fraction >= float(settings['candidate_min_fraction_smaller_informative_support'])
            row = {'left_id': left, 'right_id': right, 'matched_fingerprints': number,
                'smaller_informative_support': denominator, 'fraction_smaller_support': fraction,
                'same_recorded_series_UID': bool(uid[left] and uid[left] == uid[right]),
                'fingerprint_candidate': candidate}
            score_rows.append(row)
            if candidate: candidates.append(row.copy())
    table_write(pd.DataFrame(score_rows), destination / 'all_pair_fingerprint_scores.csv')
    table_write(pd.DataFrame({'patient_id': ids, 'informative_fingerprint_points': supports}), destination / 'fingerprint_support.csv')
    known_groups = json.loads(Path(paths['known_source_conflicts']).read_text())
    known_pairs = {tuple(sorted(pair)) for group in known_groups for pair in itertools.combinations(group, 2)}
    detected = {tuple((row['left_id'], row['right_id'])) for row in candidates}
    positive_control_ok = known_pairs.issubset(detected)
    if not positive_control_ok:
        json_write(destination / 'SCREEN_FAILURE.json', {'status': 'failed_known_positive_control',
            'known_pairs': sorted(known_pairs), 'detected_candidates': sorted(detected),
            'outcomes_used': False, 'candidate_settings_unchanged': True})
        raise RuntimeError('Prespecified screen missed an established shifted-crop pair')
    reviews = []; excluded = set(itertools.chain.from_iterable(known_groups))
    for candidate in candidates:
        left, right = candidate['left_id'], candidate['right_id']
        try:
            review = review_pair(Path(paths['imaging_root']) / left / 'ct.nii.gz',
                                 Path(paths['imaging_root']) / right / 'ct.nii.gz', settings)
        except Exception as error:
            review = {'disposition': 'unresolved_review_error', 'analytically_exclude_both': True,
                      'error_type': type(error).__name__, 'error': str(error), 'outcome_values_used': False}
        review.update(candidate); reviews.append(review)
        if review['analytically_exclude_both']: excluded.update((left, right))
    result = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'completed_outcome_blind_source_screen', 'patients_screened': len(ids),
        'pairs_screened': len(score_rows), 'settings': settings,
        'fingerprint_positive_control_detected': positive_control_ok,
        'known_positive_pair_count': len(known_pairs), 'candidate_count': len(candidates),
        'different_UID_candidates': sum(not row['same_recorded_series_UID'] for row in candidates),
        'candidate_reviews': reviews, 'analytically_excluded_ids_restricted': sorted(excluded),
        'patient_identity_adjudicated': False, 'endpoint_values_used_for_screen_or_record_choice': False,
        'original_data_or_splits_changed': False, 'arbitrary_registration_performed': False,
        'scope_limit': settings['limitations']}
    json_write(destination / 'source_similarity_screen.json', result)
    table_write(pd.DataFrame(reviews), destination / 'candidate_reviews_restricted.csv')
    json_write(destination / 'source_conflicts_restricted.json', {
        'analytical_exclusion_only': True, 'source_files_moved_or_copied': False,
        'unresolved_ids': sorted(excluded), 'known_groups': known_groups,
        'reviewed_candidates': reviews, 'no_outcome_based_record_choice': True})
    print(json.dumps({'stage': 'screen', 'patients': len(ids), 'pairs': len(score_rows),
                      'candidates': len(candidates), 'excluded_records': len(excluded)}, sort_keys=True), flush=True)


DATE_KEYS = {'acquisitiondate': 'CT_acquisition', 'ctacquisitiondate': 'CT_acquisition',
    'ctacquisitiondatetime': 'CT_acquisition', 'acquisitiondatetime': 'CT_acquisition',
    'scandate': 'CT_acquisition', 'studydate': 'CT_acquisition_candidate',
    'seriesdate': 'CT_acquisition_candidate',
    'diagnosisdate': 'diagnosis', 'dateofdiagnosis': 'diagnosis',
    'treatmentdate': 'treatment', 'treatmentstartdate': 'treatment',
    'pretreatment': 'pretreatment_status', 'pretreatmentstatus': 'pretreatment_status',
    'treatmentstatus': 'treatment_status'}


def timing_fields(value, prefix=''):
    findings = []
    if isinstance(value, dict):
        for key, item in value.items():
            normal = ''.join(c for c in key.lower() if c.isalnum())
            name = prefix + '.' + key if prefix else key
            if normal in DATE_KEYS and item not in (None, ''):
                findings.append({'field': name, 'role': DATE_KEYS[normal], 'value': item})
            findings.extend(timing_fields(item, name))
    elif isinstance(value, list):
        for i, item in enumerate(value): findings.extend(timing_fields(item, prefix + '[' + str(i) + ']'))
    return findings


def timing_audit(protocol, ids, root):
    paths = protocol['paths']; rows = []; found = []
    retained = []
    frozen_sources = json.loads((root / 'preparation/source_manifest.json').read_text())
    for record in frozen_sources.get('retained_timing_metadata_files', []):
        path = Path(record['path'])
        value = json.loads(path.read_text())
        fields = timing_fields(value)
        if fields:
            retained.append({'source': str(path), 'candidate_fields': fields,
                             'patient_id_if_recorded': value.get('patient_id', value.get('case_id', ''))
                                 if isinstance(value, dict) else ''})
    for patient in ids:
        meta_path = Path(paths['imaging_root']) / patient / 'meta.json'
        meta = json.loads(meta_path.read_text()); fields = timing_fields(meta)
        if fields: found.append({'patient_id': patient, 'source': str(meta_path), 'candidate_fields': fields})
        # A genuine candidate field requires review, not inference from a path,
        # UID, file time, processing timestamp or the audit's calendar date.
        rows.append({'patient_id': patient, 'timing_status':
            'candidate_authentic_timing_field_requires_adjudication' if fields else 'unknown_no_authentic_clinical_timing_fields',
            'entry_days': np.nan, 'verified_baseline_at_diagnosis': False,
            'verified_delayed_entry': False, 'source_date_candidate_count': len(fields)})
    table_headers = {name: read_table(paths[name]).columns.tolist() for name in ('clinical', 'outcomes', 'current_radiomics')}
    source_schema_date_keys = {name: [column for column in columns
        if ''.join(c for c in column.lower() if c.isalnum()) in DATE_KEYS]
        for name, columns in table_headers.items()}
    result = {'schema_version': 1, 'case_count': len(ids), 'per_case_metadata_files_searched': len(ids),
        'authentic_candidate_fields': found, 'clinical_outcome_schema_timing_columns': source_schema_date_keys,
        'retained_JSON_metadata_files_searched': len(frozen_sources.get('retained_timing_metadata_files', [])),
        'retained_candidate_clinical_timing_fields_requiring_adjudication': retained,
        'retained_search_scope': 'Whitelisted clinical date/status field names only; no model metric/risk values inspected or summarized',
        'metadata_schema_keys': sorted({key for p in ids for key in json.loads((Path(paths['imaging_root']) / p / 'meta.json').read_text()).keys()}),
        'status_counts': dict(collections.Counter(r['timing_status'] for r in rows)),
        'interpretation': 'exploratory_retrospective_OS_association_not_verified_pretreatment_prediction',
        'mtime_or_processing_time_used_as_clinical_date': False,
        'date_like_tokens_in_UIDs_or_paths_used_as_clinical_date': False,
        'unknown_entry_time_encoded_as_verified_zero': False,
        'known_later_scan_with_valid_delayed_entry': 0,
        'original_calendar_event_dates_present': False}
    json_write(root / 'preparation/timing_audit.json', result)
    table_write(pd.DataFrame(rows), root / 'preparation/timing_status.csv')
    return pd.DataFrame(rows).set_index('patient_id', drop=False)


def rule_flags(row, paired, min_voxels):
    regions = ('primary', 'nodal') if paired else ('primary',)
    primary_names = [name.strip().lower() for name in row['roi_primary_names'].split('|') if name.strip()]
    ready = (truth(row['ct_readable']) and numeric(row['ct_nonfinite_voxels']) == 0
        and all(truth(row[r + '_readable']) and truth(row[r + '_geometry_matches_ct'])
                and numeric(row[r + '_nonfinite_voxels']) == 0
                and set(json.loads(row[r + '_labels'])).issubset({0, 1}) for r in regions))
    return {
        'readable_finite_aligned_binary': ready,
        'nonempty_required_regions': all(numeric(row[r + '_voxels']) > 0 for r in regions),
        'primary_is_PT_only': primary_names == ['pt']
            and truth(row['meta_patient_id_matches_directory'])
            and bool(row['roi_primary_names']) and 'failed' not in row['mask_primary_qc_metadata'].lower(),
        'no_required_mask_boundary_contact': all(not truth(row[r + '_touches_array_boundary']) for r in regions),
        'no_PT_LN_overlap_for_paired': numeric(row['primary_nodal_overlap_voxels']) == 0 if paired else True,
        'minimum_100_voxels_each_required_region': all(numeric(row[r + '_voxels']) >= min_voxels for r in regions)}


def split_lookup(paths):
    result = {}
    for split in ('train', 'val', 'test'):
        ids = Path(paths[split + '_ids']).read_text().splitlines()
        if len(ids) != len(set(ids)): raise ValueError('Duplicated split ID')
        for patient in ids:
            if patient in result: raise ValueError('Split memberships overlap')
            result[patient] = split
    return result


def cohort_pre_measurement(protocol, root):
    verify_fixed_inputs(protocol, root, verify_meta=True)
    destination = root / 'preparation'; destination.mkdir(mode=0o700, exist_ok=True)
    target = destination / 'cohort_manifest_pre_measurement.csv'
    if target.exists(): raise FileExistsError(target)
    paths = protocol['paths']; qc = unique_table(paths['imaging_qc'], 'patient_id')
    clinical = unique_table(paths['clinical'], 'L_ID'); outcomes = unique_table(paths['outcomes'], 'Patient_ID')
    screen = json.loads((root / 'screen/source_similarity_screen.json').read_text())
    if screen['status'] != 'completed_outcome_blind_source_screen' or not screen['fingerprint_positive_control_detected']:
        raise ValueError('Source screen did not pass its positive-control gate')
    conflicts = set(screen['analytically_excluded_ids_restricted'])
    splits = split_lookup(paths); timing = timing_audit(protocol, sorted(qc.index), root)
    minimum = int(protocol['eligibility']['minimum_voxels_each_region']); rows = []
    for patient, image in qc.sort_index().iterrows():
        clin = clinical.loc[patient].to_dict() if patient in clinical.index else {}
        endpoint = outcomes.loc[patient].to_dict() if patient in outcomes.index else {}
        duration, event = numeric(endpoint.get('OS.TIME')), numeric(endpoint.get('OS.EVENT'))
        valid_os = math.isfinite(duration) and duration > 0 and event in (0., 1.)
        age = numeric(clin.get('AGE')); sex = clin.get('SEX', '')
        good_clinical = math.isfinite(age) and sex in ('M', 'F')
        record = {'patient_id': patient, 'split': splits.get(patient, 'excluded_missing_OS'),
            'OS_TIME': duration, 'OS_EVENT': event, 'AGE': age, 'SEX': sex,
            'T': clin.get('T', ''), 'N': clin.get('N', ''), 'HPV': clin.get('HPV', ''),
            'timing_status': timing.loc[patient, 'timing_status'], 'entry_days': np.nan,
            'verified_baseline_at_diagnosis': False, 'valid_OS': valid_os,
            'no_unresolved_source_conflict': patient not in conflicts,
            'analytical_source_conflict_exclusion': patient in conflicts,
            'recorded_AGE_and_MF_SEX': good_clinical,
            'primary_voxels': numeric(image['primary_voxels']), 'nodal_voxels': numeric(image['nodal_voxels']),
            'primary_volume_cm3_audited': numeric(image['primary_volume_mm3']) / 1000.,
            'nodal_volume_cm3_audited': numeric(image['nodal_volume_mm3']) / 1000.,
            'original_z_spacing_mm': numeric(image['original_spacing_z_mm']),
            'eligibility_stage': 'pre_measurement_rule_9_not_yet_observed'}
        for kind, paired in (('paired', True), ('primary_only', False)):
            flags = rule_flags(image, paired, minimum)
            for key, value in flags.items(): record[kind + '__' + key] = value
            structural = valid_os and patient not in conflicts and all(flags.values())
            record[kind + '_measurement_candidate'] = structural
            record[kind + '_eligible_pre_measurement'] = structural and good_clinical
        rows.append(record)
    frame = pd.DataFrame(rows)
    assert len(frame) == len(qc) and frame.patient_id.is_unique
    assert frame.loc[frame.valid_OS, 'patient_id'].isin(splits).all()
    table_write(frame, target)
    parquet_write(frame, destination / 'cohort_manifest_pre_measurement.parquet')
    counts = []
    for kind in ('paired', 'primary_only'):
        for split in ('all', 'train', 'val', 'test', 'excluded_missing_OS'):
            subset = frame if split == 'all' else frame[frame.split == split]
            selected = subset[subset[kind + '_eligible_pre_measurement']]
            counts.append({'analysis': kind, 'split': split, 'patients': len(selected),
                           'deaths': int(selected.OS_EVENT.sum()), 'censored': int((selected.OS_EVENT == 0).sum()),
                           'stage': 'pre_measurement_provisional'})
    table_write(pd.DataFrame(counts), destination / 'eligibility_counts_pre_measurement.csv')
    json_write(destination / 'PREPARATION_PRE_MEASUREMENT.json', {
        'status': 'structural_preparation_complete_measurement_rule_9_pending',
        'case_count': len(frame), 'counts': counts, 'known_timing_verified': False,
        'source_mutation': False, 'split_mutation': False, 'outcome_association_fits': 0,
        'cohort_manifest_csv_sha256': sha256(target)})
    print(json.dumps({'stage': 'cohort', 'provisional_counts': counts}, sort_keys=True), flush=True)


def cohort_final(protocol, root):
    verify_fixed_inputs(protocol, root)
    destination = root / 'preparation'
    source = destination / 'cohort_manifest_pre_measurement.csv'
    pre = read_table(source); validation_path = root / 'measurements/measurement_validation.csv'
    validation = unique_table(validation_path, 'patient_id')
    extraction_path = root / 'measurements/EXTRACTION.json'
    extraction = json.loads(extraction_path.read_text())
    lock = json.loads(Path(protocol['paths']['protocol_lock']).read_text())
    features_path = root / 'measurements/phenotype_features.csv'
    if (extraction.get('status') != 'completed'
            or extraction.get('protocol_sha256') != lock['protocol_sha256']
            or extraction.get('code_sha256') != lock['code_sha256']['measure.py']
            or extraction.get('premeasurement_manifest_sha256') != sha256(source)
            or extraction.get('validation_sha256') != sha256(validation_path)
            or extraction.get('features_sha256') != sha256(features_path)):
        raise ValueError('Actual measurement extraction is not bound to this protocol/source manifest')
    candidates = set(pre.loc[pre.paired_measurement_candidate.map(truth)
                             | pre.primary_only_measurement_candidate.map(truth), 'patient_id'])
    if candidates != set(validation.index) or extraction.get('candidate_n') != len(candidates):
        raise ValueError('Measurement validation is incomplete for structural candidates')
    features = unique_table(features_path, 'patient_id')
    if set(features.index) != candidates:
        raise ValueError('Actual phenotype features and structural candidates differ')
    for analysis in ('paired', 'primary_only'):
        name = analysis + '_measurement_valid'
        if not validation[name].str.lower().isin(['true', 'false']).all():
            raise ValueError('Measurement validity contains an unrecognized token')
        if not all(truth(validation.loc[patient, name]) == truth(features.loc[patient, name]) for patient in candidates):
            raise ValueError('Feature and independent measurement-validation flags disagree')
    frame = pre.copy(); flow_rows = []; counts = []
    for analysis in ('paired', 'primary_only'):
        actual = []; assessment = []
        for patient in frame.patient_id:
            observed = patient in validation.index
            valid = observed and truth(validation.loc[patient, analysis + '_measurement_valid'])
            actual.append(valid)
            assessment.append('valid' if valid else 'invalid' if observed else 'not_assessed_structural_exclusion')
        frame[analysis + '__valid_same_support_IQR_and_voxel_volume'] = actual
        frame[analysis + '_measurement_assessment_status'] = assessment
        flags = []
        for name in protocol['eligibility']['flow_order']:
            column = name if name in ('valid_OS', 'no_unresolved_source_conflict', 'recorded_AGE_and_MF_SEX') else analysis + '__' + name
            flags.append(frame[column].map(truth).to_numpy())
        cumulative = np.ones(len(frame), dtype=bool); first = np.full(len(frame), '', dtype=object)
        for step, (name, passing) in enumerate(zip(protocol['eligibility']['flow_order'], flags), 1):
            newly_excluded = cumulative & ~passing; first[newly_excluded] = name; cumulative &= passing
            for split in ('all', 'train', 'val', 'test', 'excluded_missing_OS'):
                in_split = np.ones(len(frame), dtype=bool) if split == 'all' else frame.split.eq(split).to_numpy()
                subset = frame.loc[cumulative & in_split]
                events = pd.to_numeric(subset.OS_EVENT, errors='coerce')
                assessed = np.ones(len(frame), dtype=bool) if name != 'valid_same_support_IQR_and_voxel_volume' else frame[analysis + '_measurement_assessment_status'].ne('not_assessed_structural_exclusion').to_numpy()
                flow_rows.append({'analysis': analysis, 'step': step, 'rule': name, 'split': split,
                    'patients_remaining': len(subset), 'deaths_remaining': int(events.sum()),
                    'censored_remaining': int(events.eq(0).sum()),
                    'newly_excluded': int((newly_excluded & in_split).sum()),
                    'all_applicable_failures': int((~passing & in_split & assessed).sum()),
                    'not_assessed': int((~assessed & in_split).sum())})
        frame[analysis + '_eligible'] = cumulative
        frame[analysis + '_first_exclusion'] = first
        frame[analysis + '_all_applicable_exclusions'] = [';'.join(name for name, passing in zip(protocol['eligibility']['flow_order'], flags) if not passing[i] and not
            (name == 'valid_same_support_IQR_and_voxel_volume' and assessment[i] == 'not_assessed_structural_exclusion')) for i in range(len(frame))]
        frame[analysis + '_unassessed_rules'] = ['valid_same_support_IQR_and_voxel_volume' if value == 'not_assessed_structural_exclusion' else '' for value in assessment]
        for split in ('all', 'train', 'val', 'test', 'excluded_missing_OS'):
            subset = frame.loc[cumulative & (np.ones(len(frame), dtype=bool) if split == 'all' else frame.split.eq(split).to_numpy())]
            events = pd.to_numeric(subset.OS_EVENT, errors='coerce')
            counts.append({'analysis': analysis, 'split': split, 'patients': len(subset),
                           'deaths': int(events.sum()), 'censored': int(events.eq(0).sum())})
    for name in ('OS_TIME', 'OS_EVENT', 'AGE', 'entry_days', 'primary_voxels', 'nodal_voxels',
                 'primary_volume_cm3_audited', 'nodal_volume_cm3_audited', 'original_z_spacing_mm'):
        frame[name] = pd.to_numeric(frame[name], errors='coerce')
    for name in frame.columns:
        if '__' in name or name in ('valid_OS', 'no_unresolved_source_conflict',
                'analytical_source_conflict_exclusion', 'recorded_AGE_and_MF_SEX',
                'verified_baseline_at_diagnosis') or name.endswith(('_measurement_candidate', '_eligible_pre_measurement')):
            frame[name] = frame[name].map(truth)
    frame['eligibility_stage'] = 'final_including_actual_same_support_measurement_validation'
    table_write(frame, destination / 'cohort_manifest.csv')
    parquet_write(frame, destination / 'cohort_manifest.parquet')
    table_write(pd.DataFrame(flow_rows), destination / 'eligibility_flow.csv')
    table_write(pd.DataFrame(counts), destination / 'eligibility_counts.csv')
    json_write(destination / 'FINAL_COHORT.json', {'schema_version': 1,
        'status': 'eligibility_complete_before_phenotype_OS_association',
        'pre_measurement_manifest_sha256': sha256(source),
        'actual_extraction_receipt_sha256': sha256(extraction_path),
        'measurement_validation_sha256': sha256(validation_path),
        'cohort_manifest_sha256': sha256(destination / 'cohort_manifest.csv'),
        'counts': counts, 'source_records_and_original_split_labels_unchanged': True,
        'all_applicable_and_fixed_order_exclusions_retained': True,
        'timing_interpretation': 'exploratory_retrospective_OS_association_not_verified_pretreatment_prediction',
        'outcome_association_fits': 0})
    print(json.dumps({'stage': 'finalcohort', 'counts': counts}, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--stage', required=True, choices=('sources', 'screen', 'cohort', 'finalcohort'))
    args = parser.parse_args(); protocol, root = load_protocol(args.protocol, args.output_root)
    {'sources': source_manifest, 'screen': run_screen, 'cohort': cohort_pre_measurement,
     'finalcohort': cohort_final}[args.stage](protocol, root)


if __name__ == '__main__': main()
