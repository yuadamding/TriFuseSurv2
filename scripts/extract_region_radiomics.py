#!/usr/bin/env python3
"""Extract optional PyRadiomics features from the exact persisted image supports.

The extractor rejects internal mask resampling/resegmentation. Intensity filters
may differ from the learned CT pipeline and are recorded in the supplied config.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))

import pandas as pd
import SimpleITK as sitk

from trifusesurv2.encoders.radiomics import DEFAULT_GROUP_PREFIXES, DEFAULT_PRESENCE_COLUMNS
from trifusesurv2.preprocessing.compartments import ReferenceGeometry, assert_same_geometry
from trifusesurv2.preprocessing.region_io import (
    canonical_json_sha256, file_sha256, load_region_arrays, load_region_manifests,
    validate_radiomics_binding, validate_radiomics_config,
)


def validate_extraction_config(config: dict[str, Any]) -> None:
    validate_radiomics_config(config)


def extract_case(manifest, extractor, config, *, software_version: str) -> dict[str, Any]:
    validate_extraction_config(config)
    base = Path(manifest['_manifest_path']).parent
    image = manifest.get('radiomics_image')
    if not image:
        raise ValueError('The region manifest does not bind a radiomics intensity image.')
    raw_path = base / image['path']
    if file_sha256(raw_path) != image['sha256']:
        raise ValueError('Radiomics intensity file differs from the bound acquisition.')
    raw_geometry = ReferenceGeometry.from_image(sitk.ReadImage(str(raw_path)))
    assert_same_geometry(raw_geometry, ReferenceGeometry.from_dict(image['geometry']))
    assert_same_geometry(raw_geometry, ReferenceGeometry.from_dict(manifest['image']['geometry']))
    load_region_arrays(manifest, ct_path=base / manifest['image']['path'])
    row = {
        'patient_id': str(manifest['patient_id']), 'scan_id': str(manifest['scan_id']),
        'radiomics_image_sha256': image['sha256'],
        'radiomics_intensity_transform': image['intensity_transform'],
        'radiomics_config_json': json.dumps(config, sort_keys=True, separators=(',', ':')),
        'radiomics_config_sha256': canonical_json_sha256(config),
        'radiomics_software': f'PyRadiomics {software_version}',
    }
    for habitat, prefix in DEFAULT_GROUP_PREFIXES.items():
        record = manifest['regions'][habitat]
        row[f'region_id__{habitat}'] = record['image_region_id']
        row[f'support_sha256__{habitat}'] = record['image_mask_sha256']
        row[DEFAULT_PRESENCE_COLUMNS[habitat]] = 0
        status = record['status']
        if status == 'present' and not record['feature_validity']['image']:
            status = 'invalid'
        if status != 'present':
            row[f'feature_status__{habitat}'] = status
            continue
        try:
            result = extractor.execute(str(raw_path), str(base / record['image_mask_path']), label=1)
            features = {name: float(value) for name, value in result.items() if not name.startswith('diagnostics_')}
            if not features or not all(math.isfinite(v) for v in features.values()):
                raise ValueError('No finite feature vector was produced.')
        except Exception as exc:
            row[f'feature_status__{habitat}'] = 'invalid'
            row[f'feature_error__{habitat}'] = f'{type(exc).__name__}: {exc}'
            continue
        row.update({prefix + name: value for name, value in features.items()})
        row[DEFAULT_PRESENCE_COLUMNS[habitat]] = 1
        row[f'feature_status__{habitat}'] = 'valid'
    validate_radiomics_binding(row, manifest)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meta_csv', required=True)
    parser.add_argument('--config_json', required=True)
    parser.add_argument('--out_csv', required=True)
    parser.add_argument('--id_col', default='patient_id')
    parser.add_argument('--region_manifest_col', default='region_manifest_path')
    parser.add_argument('--deployment_mode', choices=('expert', 'automated', 'contour_variation'), default='expert')
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config_json).read_text())
    validate_extraction_config(config)
    try:
        import radiomics
        from radiomics.featureextractor import RadiomicsFeatureExtractor
    except ImportError as exc:
        raise RuntimeError('This optional extraction command requires PyRadiomics in the selected environment.') from exc
    extractor = RadiomicsFeatureExtractor(config)
    meta_path = Path(args.meta_csv).resolve()
    meta = pd.read_csv(meta_path, dtype={args.id_col: str})
    manifests = load_region_manifests(meta, id_col=args.id_col, manifest_col=args.region_manifest_col,
                                     data_root=meta_path.parent, deployment_mode=args.deployment_mode)
    rows = [extract_case(manifest, extractor, config, software_version=radiomics.__version__) for manifest in manifests.values()]
    output = Path(args.out_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name('.' + output.name + '.tmp')
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(output)
    receipt = {
        'radiomics_csv_sha256': file_sha256(output), 'radiomics_config': config,
        'radiomics_config_sha256': canonical_json_sha256(config),
        'software': f'PyRadiomics {radiomics.__version__}',
        'patients': {
            row['patient_id']: {
                'source_region_manifest': manifests[row['patient_id']]['_manifest_path'],
                'source_region_manifest_sha256': file_sha256(manifests[row['patient_id']]['_manifest_path']),
                'regions': {habitat: {'image_region_id': row[f'region_id__{habitat}'],
                                     'support_sha256': row[f'support_sha256__{habitat}'],
                                     'feature_status': row[f'feature_status__{habitat}'],
                                     'radiomics_config_sha256': row['radiomics_config_sha256']}
                            for habitat in DEFAULT_GROUP_PREFIXES},
            } for row in rows
        },
    }
    output.with_suffix(output.suffix + '.regions.json').write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    print(f'Wrote {len(rows)} region-bound radiomics rows to {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
