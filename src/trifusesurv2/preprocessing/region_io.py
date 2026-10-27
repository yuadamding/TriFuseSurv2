"""Read verified model-grid regions and bind external radiomics to their support.

A manifest records both reference-grid anatomy and its explicit resampled image
representation. Radiomics must identify the latter, not merely a habitat name.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import SimpleITK as sitk

from trifusesurv2.schema import RADIOLOGY_HABITATS, REGION_STATUS_CODES, PHYSICAL_GEOMETRY_FEATURES
from trifusesurv2.preprocessing.compartments import (
    ReferenceGeometry, assert_same_geometry, mask_checksum, validate_region_manifest,
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_radiomics_config(config: Mapping[str, Any]) -> None:
    """Require extraction to retain the persisted binary support on its grid."""
    if not isinstance(config, dict) or not config:
        raise ValueError('A nonempty explicit radiomics configuration is required.')
    settings = config.get('setting', {})
    image_types = config.get('imageType', {})
    if not isinstance(settings, dict) or not isinstance(image_types, dict):
        raise ValueError('Radiomics setting and imageType must be configuration objects.')
    for options in [settings, *image_types.values()]:
        if options is None:
            continue
        if not isinstance(options, dict):
            raise ValueError('Each radiomics imageType must declare a settings object.')
        for key in ('resampledPixelSpacing', 'resegmentRange'):
            if options.get(key) is not None:
                raise ValueError(f'{key} changes declared region support; preprocess a shared region instead.')
        if options.get('correctMask', False):
            raise ValueError('correctMask cannot alter a verified region during feature extraction.')
        if options.get('label', 1) != 1:
            raise ValueError('Persisted binary compartment masks use label=1.')


def resolve_manifest_path(value: str, data_root: str | Path | None) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(data_root or '.') / path
    return path.resolve(strict=True)


def validate_deployment(manifest: Mapping[str, Any], deployment_mode: str) -> None:
    if deployment_mode not in {'expert', 'automated', 'contour_variation'}:
        raise ValueError(f'Unknown deployment mode: {deployment_mode}')
    if deployment_mode != 'automated':
        return
    sources = {str(r['contour_source']).lower() for r in manifest['regions'].values()}
    if not sources or not sources.issubset({'automated', 'predicted'}):
        raise ValueError('Automated evaluation requires predicted contours for every region and the crop.')
    provenance = manifest.get('segmentation_provenance', {})
    if provenance.get('method') not in {'out_of_fold', 'independent'} or not provenance.get('segmenter_id'):
        raise ValueError('Automated evaluation requires an identified independent or out-of-fold segmenter.')
    trained = provenance.get('training_patient_ids')
    if not isinstance(trained, list) or str(manifest['patient_id']) in {str(x) for x in trained}:
        raise ValueError('Segmenter training identities must be declared and exclude this evaluation patient.')


def load_region_manifest(path: str | Path, patient_id: str | None = None, *, deployment_mode: str = 'expert') -> dict:
    path = Path(path).resolve(strict=True)
    manifest = json.loads(path.read_text())
    validate_region_manifest(manifest)
    if patient_id is not None and str(manifest['patient_id']) != str(patient_id):
        raise ValueError(f'Region manifest patient mismatch: expected {patient_id}, got {manifest["patient_id"]}')
    validate_deployment(manifest, deployment_mode)
    image = manifest.get('image', {})
    if not image.get('path') or not image.get('sha256') or not image.get('intensity_transform'):
        raise ValueError('Verified regions require a bound image and documented intensity transform.')
    if str(image.get('scan_id', '')) != str(manifest['scan_id']):
        raise ValueError('Bound image acquisition differs from the region manifest.')
    image_geometry = ReferenceGeometry.from_dict(image['geometry'])
    for habitat in RADIOLOGY_HABITATS:
        record = manifest['regions'][habitat]
        if not all(record.get(k) for k in ('image_region_id', 'image_mask_path', 'image_mask_sha256', 'image_geometry')):
            raise ValueError(f'{habitat} has no verified model-grid representation.')
        completeness = record.get('crop_completeness') or {}
        fraction = float(completeness.get('fraction', 0.0))
        if not np.isfinite(fraction) or not np.isclose(fraction, 1.0, rtol=0, atol=1e-9):
            raise ValueError(f'{habitat}: crop truncates the declared reference region.')
        assert_same_geometry(image_geometry, record['image_geometry'])
        if completeness.get('retained_reference_voxels') != record['voxel_count'] or completeness.get('reference_voxels') != record['voxel_count']:
            raise ValueError(f'{habitat}: crop retained-support accounting is inconsistent.')
    manifest['_manifest_path'] = str(path)
    return manifest


def load_region_manifests(meta, *, id_col: str, manifest_col: str = 'region_manifest_path',
                          data_root: str | Path | None = None, deployment_mode: str = 'expert') -> dict[str, dict]:
    if manifest_col not in meta.columns:
        raise ValueError(f'Verified mode requires metadata column {manifest_col}. Regenerate shared physical regions.')
    result = {}
    for _, row in meta.iterrows():
        pid = str(row[id_col])
        path = resolve_manifest_path(str(row[manifest_col]), data_root)
        value = load_region_manifest(path, pid, deployment_mode=deployment_mode)
        if pid in result and result[pid]['_manifest_path'] != str(path):
            raise ValueError(f'Multiple region manifests for patient {pid}')
        result[pid] = value
    return result


def load_region_arrays(manifest: Mapping[str, Any], *, ct_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = Path(manifest['_manifest_path']).parent
    image = manifest['image']
    declared_ct = (base / image['path']).resolve()
    if Path(ct_path).resolve() != declared_ct or file_sha256(declared_ct) != image['sha256']:
        raise ValueError('CT input does not match the image bound to the region manifest.')
    ct_geometry = ReferenceGeometry.from_image(sitk.ReadImage(str(declared_ct)))
    assert_same_geometry(ct_geometry, ReferenceGeometry.from_dict(image['geometry']))
    masks, valid, states = [], [], []
    for habitat in RADIOLOGY_HABITATS:
        record = manifest['regions'][habitat]
        img = sitk.ReadImage(str(base / record['image_mask_path']))
        assert_same_geometry(ct_geometry, img)
        arr = sitk.GetArrayFromImage(img)
        if mask_checksum(arr) != record['image_mask_sha256']:
            raise ValueError(f'{habitat}: model mask checksum differs from its region manifest.')
        arr = arr.astype(bool)
        if bool(record['feature_validity']['image']) != bool(record['support_valid'] and arr.any()):
            raise ValueError(f'{habitat}: image feature validity contradicts the loaded support.')
        masks.append(arr.astype(np.float32))
        image_valid = bool(record['support_valid']) and (not record['presence'] or bool(arr.any()))
        valid.append(image_valid)
        state = record['status']
        if record['presence'] and not bool(arr.any()):
            state = 'invalid'
        states.append(REGION_STATUS_CODES[state])
    stacked = np.stack(masks)
    validate_region_manifest(manifest, masks=dict(zip(RADIOLOGY_HABITATS, stacked)), geometry=ct_geometry)
    geometry = manifest['physical_geometry']
    if tuple(geometry['names']) != tuple(PHYSICAL_GEOMETRY_FEATURES):
        raise ValueError('Physical geometry feature names/order do not match this model contract.')
    values = np.asarray(geometry['values'], dtype=np.float32)
    validity = np.asarray(geometry['valid'], dtype=np.float32)
    if not np.isfinite(values).all() or not np.isin(validity, [0, 1]).all():
        raise ValueError('Invalid physical geometry values/validity.')
    return stacked, np.asarray(valid, dtype=bool), np.asarray(states, dtype=np.int64), np.concatenate((values, validity))


def validate_radiomics_binding(row: Mapping[str, Any], manifest: Mapping[str, Any], *, group_names=RADIOLOGY_HABITATS) -> None:
    """Reject a radiomics row unless it declares the exact persisted image supports."""
    if str(row.get('scan_id', '')) != str(manifest['scan_id']):
        raise ValueError('Radiomics acquisition scan_id does not match the image region manifest.')
    image = manifest.get('radiomics_image')
    if not image:
        raise ValueError('Region manifest has no qualified radiomics intensity image.')
    if str(image.get('scan_id', '')) != str(manifest['scan_id']):
        raise ValueError('Radiomics intensity image acquisition differs from the region manifest.')
    assert_same_geometry(ReferenceGeometry.from_dict(manifest['image']['geometry']), image['geometry'])
    if str(row.get('radiomics_image_sha256', '')) != image.get('sha256'):
        raise ValueError('Radiomics intensity image checksum does not match the declared acquisition.')
    if str(row.get('radiomics_intensity_transform', '')) != str(image.get('intensity_transform', '')):
        raise ValueError('Radiomics intensity transform does not match the bound image.')
    raw_config = row.get('radiomics_config_json', '')
    try:
        config = json.loads(str(raw_config))
    except (TypeError, ValueError) as exc:
        raise ValueError('Radiomics must declare its extraction configuration JSON.') from exc
    validate_radiomics_config(config)
    if str(row.get('radiomics_config_sha256', '')) != canonical_json_sha256(config):
        raise ValueError('Radiomics configuration checksum mismatch.')
    software = row.get('radiomics_software')
    if not isinstance(software, str) or not software.strip():
        raise ValueError('Radiomics must identify the extraction software and version.')
    for habitat in group_names:
        record = manifest['regions'][habitat]
        if str(row.get(f'region_id__{habitat}', '')) != record['image_region_id']:
            raise ValueError(f'{habitat}: radiomics/image region identity mismatch.')
        if str(row.get(f'support_sha256__{habitat}', '')) != record['image_mask_sha256']:
            raise ValueError(f'{habitat}: radiomics/image support checksum mismatch.')
        status = str(row.get(f'feature_status__{habitat}', ''))
        if status not in {'valid', 'absent', 'missing', 'invalid'}:
            raise ValueError(f'{habitat}: explicit radiomics feature status is required.')
        if status == 'valid' and (not record['support_valid'] or not record['presence'] or not record['feature_validity']['image']):
            raise ValueError(f'{habitat}: features declared valid for absent or unavailable tissue.')
        if status == 'absent' and record['status'] != 'absent':
            raise ValueError(f'{habitat}: absence must not substitute for unavailable/invalid input.')
