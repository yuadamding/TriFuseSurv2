#!/usr/bin/env python3
"""Prepare development-only shared regions from stored normalized legacy inputs.

Scientific execution requires the approved single H100 worker. Metadata helper
functions remain independently testable without opening image files. This
adapter neither establishes contour accuracy nor creates evaluation splits.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
sys.path.insert(0, str(PACKAGE / 'scripts'))

import numpy as np
import pandas as pd
import SimpleITK as sitk

import audit_region_sampling as audit
import preprocess_roi_inputs as preprocess
from extract_region_radiomics import extract_case, validate_extraction_config
from qualify_h100 import require_one_h100
from trifusesurv2.encoders.radiomics import DEFAULT_GROUP_PREFIXES
from trifusesurv2.preprocessing.identifiers import read_identifier_csv, require_identifier, resolve_identifier_column
from trifusesurv2.preprocessing.compartments import assert_same_geometry
from trifusesurv2.preprocessing.region_io import canonical_json_sha256, file_sha256, load_region_manifest
from trifusesurv2.preprocessing.sampling import CommonGridPolicy
from trifusesurv2.utils.data import resolve_preprocessed_case_path, validate_normalized_ct


CONTOUR_SOURCE = 'legacy_rtstruct_derived_unreviewed'
INTENSITY_DESCRIPTION = 'stored normalized [0,1]; original normalization implementation unavailable'
TISSUE_DESCRIPTION = ('development candidate: stored normalized CT > 0.02; normalization provenance unavailable; '
                      'not a validated anatomical tissue classification')
LIMITATIONS = [
    'Original CT and RTSTRUCT sources are unavailable in this input delivery.',
    'Stored CT was already normalized/cropped; original preprocessing history is unresolved.',
    'Derived contours are unreviewed; aligned QC does not establish anatomical or clinical accuracy.',
    'Empty or failed legacy contour QC means unavailable annotation, not known disease absence.',
    'The intensity threshold is a development tissue candidate and makes no HU interpretation.',
    'These inputs are development-only and do not establish clinical performance or an untouched evaluation cohort.',
]


def _text(value):
    return '' if value is None or pd.isna(value) else str(value)


def _nonnegative(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def annotation_availability(row, lesion):
    """Legacy empty/failed masks cannot establish absence of annotated disease."""
    qc = _text(row.get(f'mask_{lesion}_qc')).strip().casefold()
    tokens = [value.strip() for value in qc.split(';') if value.strip()]
    if not tokens or qc == 'empty':
        return False, 'empty_or_missing_qc_is_unknown_annotation'
    if any('failed' in value for value in tokens):
        return False, 'failed_source_mask_qc'
    if not all(value == 'aligned' or value.endswith(':aligned') for value in tokens):
        return False, 'unrecognized_source_mask_qc'
    count = _nonnegative(row.get(f'mask_{lesion}_voxels'))
    if count is None or count <= 0:
        return False, 'positive_mask_support_not_established'
    return True, 'legacy_aligned_nonempty_unreviewed'


def development_row(row, *, id_col, source_csv_sha256):
    """Attach conservative source declarations; do not inspect outcomes or images."""
    prepared = dict(row)
    pid = require_identifier(row.get(id_col), name=id_col)
    scan = require_identifier(row.get('ct_series_uid'), name='ct_series_uid')
    referenced_scan = _text(row.get('rtstruct_referenced_series_uid'))
    if referenced_scan and referenced_scan != scan:
        raise ValueError(f'{pid}: RTSTRUCT referenced-series metadata contradicts the CT acquisition identity.')
    if _text(row.get('status')).strip().casefold() != 'ok':
        raise ValueError(f'{pid}: source preprocessing status is not ok.')
    if not _text(row.get('ct_out_path')).strip():
        raise ValueError(f'{pid}: stored normalized CT path is missing.')
    pt_available, pt_reason = annotation_availability(row, 'primary')
    ln_available, ln_reason = annotation_availability(row, 'nodal')
    prepared.update(
        scan_id=scan, radiomics_scan_id=scan, contour_source=CONTOUR_SOURCE,
        contour_version=f'source_csv_sha256:{source_csv_sha256}',
        ct_intensity_transform=INTENSITY_DESCRIPTION, radiomics_intensity_transform=INTENSITY_DESCRIPTION,
        pt_annotation_available=pt_available, ln_annotation_available=ln_available,
        valid_tissue_definition=TISSUE_DESCRIPTION, development_only=True,
        source_history_status='original_CT_RTSTRUCT_and_normalization_history_unavailable',
    )
    qc = dict(patient_id=pid, scan_id=scan, pt_annotation_available=pt_available,
              ln_annotation_available=ln_available, pt_annotation_reason=pt_reason,
              ln_annotation_reason=ln_reason, source_primary_qc=_text(row.get('mask_primary_qc')),
              source_nodal_qc=_text(row.get('mask_nodal_qc')), development_only=True)
    return prepared, qc


def declared_z_spacing(row):
    try:
        spacing = [float(value) for value in _text(row.get('ct_spacing_xyz')).split('|')]
    except ValueError:
        return None
    return spacing[2] if len(spacing) == 3 and all(math.isfinite(x) and x > 0 for x in spacing) else None


def select_representatives(rows, *, id_col):
    """Deterministic extrema of declared anatomy/acquisition metadata, never outcomes."""
    rows = sorted(rows, key=lambda row: require_identifier(row[id_col], name=id_col))
    reasons = {}

    def choose(label, candidates, measure, maximum=False):
        candidates = [(measure(row), row) for row in candidates]
        candidates = [(value, row) for value, row in candidates if value is not None]
        if candidates:
            _, selected = min(candidates, key=lambda pair: (-pair[0] if maximum else pair[0], pair[1][id_col]))
            reasons.setdefault(selected[id_col], []).append(label)

    pt = [row for row in rows if annotation_availability(row, 'primary')[0]
          and (_nonnegative(row.get('mask_primary_mm3')) or 0) > 0]
    ln = [row for row in rows if annotation_availability(row, 'nodal')[0]
          and (_nonnegative(row.get('mask_nodal_mm3')) or 0) > 0]
    choose('small_positive_primary', pt, lambda row: _nonnegative(row.get('mask_primary_mm3')))
    choose('large_primary', pt, lambda row: _nonnegative(row.get('mask_primary_mm3')), maximum=True)
    choose('small_positive_nodal', ln, lambda row: _nonnegative(row.get('mask_nodal_mm3')))
    choose('large_nodal', ln, lambda row: _nonnegative(row.get('mask_nodal_mm3')), maximum=True)
    empty_ln = [row for row in rows if _text(row.get('mask_nodal_qc')).strip().casefold() == 'empty']
    if empty_ln:
        reasons.setdefault(empty_ln[0][id_col], []).append('no_nodal_support_annotation_unknown')
    choose('finest_declared_acquisition_z', rows, declared_z_spacing)
    choose('coarsest_declared_acquisition_z', rows, declared_z_spacing, maximum=True)
    return [row for row in rows if row[id_col] in reasons], reasons


def resolve_source(value, *, patient_id, data_root):
    path = Path(resolve_preprocessed_case_path(str(value), data_root=str(data_root), patient_id=patient_id)).resolve(strict=True)
    if not path.is_relative_to(data_root.resolve()):
        raise ValueError(f'{patient_id}: source must resolve inside the supplied data root: {path}')
    return path


def validate_available_mask(image, *, ct_image, source):
    assert_same_geometry(ct_image, image)
    support = preprocess._verified_binary_mask(image, source=source)
    if not support.any():
        raise ValueError(f'Available legacy annotation has empty stored support; absence is not established: {source}')
    return int(support.sum())


def write_feature_table(rows, path):
    # Missing compartments in the first case must not discard later measurements.
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['patient_id', 'scan_id'])
    frame.to_csv(path, index=False)


def overlay_patient_ids(rows, *, id_col, selection_mode):
    if selection_mode == 'all_eligible':
        rows, _ = select_representatives(rows, id_col=id_col)
    return sorted(row[id_col] for row in rows)


def render_qc_overlays(rows, patient_ids, *, id_col, output, render=None):
    if render is None:
        from visualize_region_qc import render
    prepared = {row[id_col]: row for row in rows}
    result = {}
    for pid in patient_ids:
        if pid not in prepared:
            result[pid] = {'status': 'not_prepared'}
            continue
        path = output / 'qc_overlays' / f'{preprocess._safe_id(pid)}.png'
        manifest = output / prepared[pid]['region_manifest_path']
        try:
            render(manifest, path, synthetic=False)
            result[pid] = {'status': 'rendered_pending_visual_review', 'development_only': True,
                           'path': str(path.relative_to(output)), 'sha256': file_sha256(path),
                           'source_region_manifest': str(manifest.relative_to(output))}
        except Exception as exc:
            result[pid] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
    return result


def compare_region_features(first, second, *, rtol, atol):
    mismatches, compared = [], 0
    for habitat, prefix in DEFAULT_GROUP_PREFIXES.items():
        status = first[f'feature_status__{habitat}']
        if status != second[f'feature_status__{habitat}']:
            mismatches.append(f'{habitat}: feature availability differs')
            continue
        if status != 'valid':
            continue
        left = {name for name in first if name.startswith(prefix)}
        right = {name for name in second if name.startswith(prefix)}
        if left != right:
            mismatches.append(f'{habitat}: feature names differ')
            continue
        for name in sorted(left):
            compared += 1
            if not math.isclose(float(first[name]), float(second[name]), rel_tol=rtol, abs_tol=atol):
                mismatches.append(name)
    return {'status': 'failed' if mismatches else ('passed' if compared else 'not_assessable'),
            'features_compared': compared, 'mismatches': mismatches, 'rtol': rtol, 'atol': atol}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-csv', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--grid-config-json', required=True, type=Path)
    parser.add_argument('--radiomics-config-json', required=True, type=Path)
    parser.add_argument('--output-shape', required=True, type=int, nargs=3, metavar=('D', 'H', 'W'))
    parser.add_argument('--id-col', default='Patient_ID')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--ids-file', type=Path, help='One literal patient identifier per line.')
    selection.add_argument('--all-cases', action='store_true')
    parser.add_argument('--check-larger-canvas', action=argparse.BooleanOptionalAction, default=None,
                        help='Compare real features on a canvas 16 voxels larger per axis; defaults on for representative selection.')
    parser.add_argument('--feature-rtol', type=float, default=1e-5)
    parser.add_argument('--feature-atol', type=float, default=1e-7)
    args = parser.parse_args(argv)
    runtime = require_one_h100()  # No input reads or scientific processing before worker qualification.
    if any(value <= 0 for value in args.output_shape):
        raise ValueError('Output shape dimensions must be positive.')
    if not all(math.isfinite(value) and value >= 0 for value in (args.feature_rtol, args.feature_atol)):
        raise ValueError('Feature tolerances must be finite and nonnegative.')
    policy = CommonGridPolicy.from_dict(json.loads(args.grid_config_json.read_text()))
    config = json.loads(args.radiomics_config_json.read_text())
    validate_extraction_config(config)
    import radiomics
    from radiomics.featureextractor import RadiomicsFeatureExtractor
    extractor = RadiomicsFeatureExtractor(config)
    source_hash = file_sha256(args.source_csv)
    meta = read_identifier_csv(args.source_csv, identifier_columns=(args.id_col,))
    id_col = resolve_identifier_column(meta.columns, args.id_col)
    preprocess._validate_output_ids(meta[id_col].tolist())
    eligible, excluded, quality = [], [], {}
    for _, source in meta.iterrows():
        try:
            row, qc = development_row(source, id_col=id_col, source_csv_sha256=source_hash)
            eligible.append(row)
            quality[row[id_col]] = qc
        except ValueError as exc:
            excluded.append({'patient_id': _text(source.get(id_col)), 'reason': str(exc)})
    reasons = {}
    if args.ids_file:
        requested = [value for value in args.ids_file.read_text().splitlines() if value.strip()]
        if len(requested) != len(set(requested)):
            raise ValueError('Explicit patient identifiers must be unique.')
        available = {row[id_col]: row for row in eligible}
        if set(requested) - set(available):
            raise ValueError(f'Explicit IDs are missing or ineligible: {sorted(set(requested) - set(available))}')
        selected = [available[pid] for pid in sorted(requested)]
        reasons = {pid: ['explicit_identifier'] for pid in requested}
        selection_mode = 'explicit_identifiers'
    elif args.all_cases:
        selected = sorted(eligible, key=lambda row: row[id_col])
        reasons = {row[id_col]: ['all_eligible_development_cases'] for row in selected}
        selection_mode = 'all_eligible'
    else:
        selected, reasons = select_representatives(eligible, id_col=id_col)
        selection_mode = 'outcome_free_representatives'
    if not selected:
        raise ValueError('No eligible development cases were selected.')
    overlay_ids = overlay_patient_ids(selected, id_col=id_col, selection_mode=selection_mode)
    check_larger = args.check_larger_canvas if args.check_larger_canvas is not None else selection_mode == 'outcome_free_representatives'
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata_rows, features, errors, invariance, manifests = [], [], [], {}, {}
    for index, source in enumerate(selected):
        row = dict(source)
        pid = row[id_col]
        try:
            ct_path = resolve_source(row['ct_out_path'], patient_id=pid, data_root=args.data_root)
            ct = sitk.ReadImage(str(ct_path))
            values = sitk.GetArrayFromImage(ct)
            validate_normalized_ct(values, source=str(ct_path))
            tissue = sitk.GetImageFromArray((values > .02).astype(np.uint8)); tissue.CopyInformation(ct)
            candidate_dir = output / 'source_candidates' / preprocess._safe_id(pid)
            candidate_dir.mkdir(parents=True)
            tissue_path = candidate_dir / 'candidate_tissue.mha'; sitk.WriteImage(tissue, str(tissue_path))
            row.update(ct_out_path=str(ct_path), radiomics_ct_path=str(ct_path), valid_tissue_out_path=str(tissue_path))
            for available, column in (('pt_annotation_available', 'mask_primary_out_path'), ('ln_annotation_available', 'mask_nodal_out_path')):
                if row[available]:
                    row[column] = str(resolve_source(row[column], patient_id=pid, data_root=args.data_root))
                    actual_count = validate_available_mask(sitk.ReadImage(row[column]), ct_image=ct, source=Path(row[column]))
                    quality[pid][f'{column}_actual_voxels'] = actual_count
            job = dict(row=row, index=index, id_col=id_col, ct_col='ct_out_path', mask_pt_col='mask_primary_out_path',
                       mask_ln_col='mask_nodal_out_path', meta_dir=str(args.data_root.resolve()), out_root=str(output / 'cases'),
                       out_csv_dir=str(output), output_size_dhw=tuple(args.output_shape), region_mode='verified',
                       sampling_policy='common_grid', grid_config=policy.to_dict(), margin_mm=10.,
                       valid_tissue_policy='mask', valid_tissue_col='valid_tissue_out_path', deployment_mode='expert',
                       overwrite=False, fail_empty_roi=False, expand_to_aspect=False)
            result = preprocess._process_one(job)
            manifest = load_region_manifest(output / result['region_manifest_path'], pid)
            feature_row = extract_case(manifest, extractor, config, software_version=radiomics.__version__)
            if check_larger:
                larger = dict(job, out_root=str(output / 'larger_canvas'), output_size_dhw=tuple(value + 16 for value in args.output_shape))
                expanded = preprocess._process_one(larger)
                expanded_manifest = load_region_manifest(output / expanded['region_manifest_path'], pid)
                expanded_features = extract_case(expanded_manifest, extractor, config, software_version=radiomics.__version__)
                invariance[pid] = compare_region_features(feature_row, expanded_features, rtol=args.feature_rtol, atol=args.feature_atol)
            row.update(ct_out_path=result['ct_out_path'], mask_primary_out_path=result['mask_primary_out_path'],
                       mask_nodal_out_path=result['mask_nodal_out_path'], region_manifest_path=result['region_manifest_path'],
                       sampling_policy='common_grid', sampling_policy_id=policy.policy_id)
            metadata_rows.append(row); features.append(feature_row)
            manifests[pid] = {'path': result['region_manifest_path'], 'sha256': file_sha256(output / result['region_manifest_path'])}
            quality[pid].update(processing_status='prepared_development_only',
                                feature_statuses={name: feature_row[f'feature_status__{name}'] for name in DEFAULT_GROUP_PREFIXES},
                                feature_errors={name: feature_row.get(f'feature_error__{name}') for name in DEFAULT_GROUP_PREFIXES})
        except Exception as exc:
            errors.append({'patient_id': pid, 'error': f'{type(exc).__name__}: {exc}'})
            quality[pid]['processing_status'] = 'failed'
    metadata_path = output / 'cohort_development_common_grid.csv'
    radiomics_path = output / 'radiomics_development.csv'
    pd.DataFrame(metadata_rows, columns=list(metadata_rows[0]) if metadata_rows else list(meta.columns) + ['region_manifest_path']).to_csv(metadata_path, index=False)
    write_feature_table(features, radiomics_path)
    audit_status = audit.main(['--metadata-csv', str(metadata_path), '--id_col', id_col, '--out-dir', str(output / 'audit')]) if metadata_rows else None
    overlays = render_qc_overlays(metadata_rows, overlay_ids, id_col=id_col, output=output)
    receipt = {
        'schema_version': 1, 'development_only': True, 'scientific_qualification': 'not_established',
        'status': 'partial' if errors else 'prepared_development_only', 'limitations': LIMITATIONS,
        'runtime': runtime, 'radiomics_software': f'PyRadiomics {radiomics.__version__}',
        'source_csv': str(args.source_csv.resolve()), 'source_csv_sha256': source_hash,
        'grid_configuration': policy.to_dict(), 'sampling_policy_id': policy.policy_id,
        'radiomics_config_file_sha256': file_sha256(args.radiomics_config_json),
        'radiomics_config_sha256': canonical_json_sha256(config), 'radiomics_configuration': config,
        'output_shape_dhw': args.output_shape, 'contour_source': CONTOUR_SOURCE,
        'intensity_description': INTENSITY_DESCRIPTION, 'tissue_candidate_definition': TISSUE_DESCRIPTION,
        'selection_mode': selection_mode, 'selection_uses_outcomes': False,
        'selected_patient_ids': [row[id_col] for row in selected], 'selection_reasons': reasons,
        'excluded_metadata_rows': excluded, 'prepared_cases': len(metadata_rows), 'errors': errors,
        'quality': {row[id_col]: quality[row[id_col]] for row in selected},
        'source_manifests': manifests, 'larger_canvas_invariance': invariance, 'audit_exit_code': audit_status,
        'qc_overlay_requested_patient_ids': overlay_ids, 'qc_overlays': overlays,
        'visual_review_status': 'pending_worker_artifact_retrieval_and_inspection',
        'metadata_csv': metadata_path.name, 'metadata_csv_sha256': file_sha256(metadata_path),
        'radiomics_csv': radiomics_path.name, 'radiomics_csv_sha256': file_sha256(radiomics_path),
        'training_started': False, 'evaluation_splits_created': False,
    }
    (output / 'development_receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(f'Prepared {len(metadata_rows)}/{len(selected)} development cases; receipt: {output / "development_receipt.json"}')
    return int(bool(errors) or audit_status not in (None, 0)
               or any(value['status'] == 'failed' for value in invariance.values())
               or any(value['status'] == 'failed' for value in overlays.values()))


if __name__ == '__main__':
    raise SystemExit(main())
