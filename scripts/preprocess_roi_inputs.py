#!/usr/bin/env python3
"""Create fixed-size ROI-crop inputs for TriFuseSurv2 survival training.

The output metadata keeps the same schema as the source CSV, but rewrites the
CT/PT/LN path columns to point at cropped NIfTI files. Crops are centered on the
union of the primary-tumor and nodal masks plus a configurable margin, then
resampled to a model-friendly fixed size.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PACKAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import SimpleITK as sitk

from trifusesurv2.utils.data import resolve_preprocessed_case_path


def _parse_margin(values: Sequence[int]) -> tuple[int, int, int]:
    vals = [int(v) for v in values]
    if len(vals) == 1:
        return vals[0], vals[0], vals[0]
    if len(vals) == 3:
        return vals[0], vals[1], vals[2]
    raise ValueError("--margin_voxels expects either one integer or three integers: D H W")


def _safe_id(value: Any) -> str:
    raw = str(value).strip()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw) or "case"


def _relpath(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return str(path)


def _bbox_from_mask(
    mask: np.ndarray,
    *,
    margin_dhw: tuple[int, int, int],
    output_size_dhw: tuple[int, int, int],
    expand_to_aspect: bool,
    fail_empty_roi: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    shape = np.asarray(mask.shape, dtype=np.int64)
    coords = np.argwhere(mask)
    if coords.size == 0:
        if fail_empty_roi:
            raise ValueError("empty PT/LN ROI mask")
        return np.zeros(3, dtype=np.int64), shape.copy(), "empty_roi_full_volume"

    margin = np.asarray(margin_dhw, dtype=np.int64)
    start = np.maximum(coords.min(axis=0) - margin, 0)
    end = np.minimum(coords.max(axis=0) + 1 + margin, shape)
    if not expand_to_aspect:
        return start.astype(np.int64), end.astype(np.int64), "ok"

    target = np.asarray(output_size_dhw, dtype=np.float64)
    size = np.maximum(end - start, 1).astype(np.float64)
    scale = float(np.max(size / target))
    desired = np.ceil(scale * target).astype(np.int64)
    desired = np.maximum(desired, (end - start).astype(np.int64))
    desired = np.minimum(desired, shape)

    center = (start.astype(np.float64) + end.astype(np.float64)) / 2.0
    start2 = np.floor(center - desired.astype(np.float64) / 2.0).astype(np.int64)
    start2 = np.maximum(start2, 0)
    start2 = np.minimum(start2, shape - desired)
    end2 = start2 + desired
    return start2.astype(np.int64), end2.astype(np.int64), "ok"


def _crop_image(img: sitk.Image, start_dhw: np.ndarray, end_dhw: np.ndarray) -> sitk.Image:
    size_dhw = end_dhw - start_dhw
    index_xyz = [int(start_dhw[2]), int(start_dhw[1]), int(start_dhw[0])]
    size_xyz = [int(size_dhw[2]), int(size_dhw[1]), int(size_dhw[0])]
    return sitk.RegionOfInterest(img, size_xyz, index_xyz)


def _copy_geometry(src: sitk.Image, dst: sitk.Image) -> sitk.Image:
    dst.SetOrigin(src.GetOrigin())
    dst.SetSpacing(src.GetSpacing())
    dst.SetDirection(src.GetDirection())
    return dst


def _resample_to_size(img: sitk.Image, output_size_dhw: tuple[int, int, int], *, is_mask: bool) -> sitk.Image:
    output_size_xyz = [int(output_size_dhw[2]), int(output_size_dhw[1]), int(output_size_dhw[0])]
    input_size = np.asarray(img.GetSize(), dtype=np.float64)
    output_size = np.asarray(output_size_xyz, dtype=np.float64)
    input_spacing = np.asarray(img.GetSpacing(), dtype=np.float64)
    output_spacing = tuple((input_spacing * input_size / output_size).tolist())

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(output_size_xyz)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    out = resampler.Execute(img)

    if not is_mask:
        return sitk.Cast(out, sitk.sitkFloat32)

    arr = (sitk.GetArrayFromImage(out) > 0.5).astype(np.uint8)
    return _copy_geometry(out, sitk.GetImageFromArray(arr))


def _resolve_required_path(row: Mapping[str, Any], col: str, *, meta_dir: Path, id_col: str) -> Path:
    pid = str(row.get(id_col, ""))
    raw = str(row.get(col, "")).strip()
    resolved = Path(resolve_preprocessed_case_path(raw, data_root=str(meta_dir), patient_id=pid))
    if not resolved.is_file():
        raise FileNotFoundError(f"{col} not found for patient_id={pid}: {resolved} (raw={raw})")
    return resolved


def _process_one(job: Mapping[str, Any]) -> dict[str, Any]:
    row = job["row"]
    index = int(job["index"])
    id_col = str(job["id_col"])
    pid = str(row[id_col])
    pid_safe = _safe_id(pid)
    meta_dir = Path(job["meta_dir"])
    out_root = Path(job["out_root"])
    out_csv_dir = Path(job["out_csv_dir"])
    output_size_dhw = tuple(int(x) for x in job["output_size_dhw"])
    margin_dhw = tuple(int(x) for x in job["margin_dhw"])
    overwrite = bool(job["overwrite"])
    expand_to_aspect = bool(job["expand_to_aspect"])
    fail_empty_roi = bool(job["fail_empty_roi"])

    ct_path = _resolve_required_path(row, str(job["ct_col"]), meta_dir=meta_dir, id_col=id_col)
    pt_path = _resolve_required_path(row, str(job["mask_pt_col"]), meta_dir=meta_dir, id_col=id_col)
    ln_path = _resolve_required_path(row, str(job["mask_ln_col"]), meta_dir=meta_dir, id_col=id_col)

    case_dir = out_root / pid_safe
    case_dir.mkdir(parents=True, exist_ok=True)
    ct_out = case_dir / "ct_roi.nii.gz"
    pt_out = case_dir / "mask_primary_roi.nii.gz"
    ln_out = case_dir / "mask_nodal_roi.nii.gz"

    ct_img = sitk.ReadImage(str(ct_path))
    pt_img = sitk.ReadImage(str(pt_path))
    ln_img = sitk.ReadImage(str(ln_path))
    if ct_img.GetSize() != pt_img.GetSize() or ct_img.GetSize() != ln_img.GetSize():
        raise ValueError(
            f"patient_id={pid} CT/PT/LN image sizes differ: "
            f"ct={ct_img.GetSize()} pt={pt_img.GetSize()} ln={ln_img.GetSize()}"
        )

    pt = sitk.GetArrayFromImage(pt_img) > 0.5
    ln = sitk.GetArrayFromImage(ln_img) > 0.5
    union = np.logical_or(pt, ln)
    start, end, status = _bbox_from_mask(
        union,
        margin_dhw=margin_dhw,
        output_size_dhw=output_size_dhw,
        expand_to_aspect=expand_to_aspect,
        fail_empty_roi=fail_empty_roi,
    )

    if overwrite or not (ct_out.is_file() and pt_out.is_file() and ln_out.is_file()):
        ct_crop = _crop_image(ct_img, start, end)
        pt_crop = _crop_image(pt_img, start, end)
        ln_crop = _crop_image(ln_img, start, end)
        sitk.WriteImage(_resample_to_size(ct_crop, output_size_dhw, is_mask=False), str(ct_out))
        sitk.WriteImage(_resample_to_size(pt_crop, output_size_dhw, is_mask=True), str(pt_out))
        sitk.WriteImage(_resample_to_size(ln_crop, output_size_dhw, is_mask=True), str(ln_out))

    return {
        "index": index,
        "pid": pid,
        "ct_out_path": _relpath(ct_out, out_csv_dir),
        "mask_primary_out_path": _relpath(pt_out, out_csv_dir),
        "mask_nodal_out_path": _relpath(ln_out, out_csv_dir),
        "roi_crop_status": status,
        "roi_crop_bbox_zyx": json.dumps([[int(x) for x in start], [int(x) for x in end]]),
        "roi_crop_margin_voxels": json.dumps([int(x) for x in margin_dhw]),
        "roi_crop_output_size_dhw": json.dumps([int(x) for x in output_size_dhw]),
        "roi_crop_source_ct": str(ct_path),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta_csv", default="OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv")
    p.add_argument("--out_root", default="OPSCC_preprocessed_roi_96x128x128")
    p.add_argument("--out_csv", default="")
    p.add_argument("--id_col", default="patient_id")
    p.add_argument("--ct_col", default="ct_out_path")
    p.add_argument("--mask_pt_col", default="mask_primary_out_path")
    p.add_argument("--mask_ln_col", default="mask_nodal_out_path")
    p.add_argument("--output_size", type=int, nargs=3, default=[96, 128, 128], metavar=("D", "H", "W"))
    p.add_argument(
        "--margin_voxels",
        type=int,
        nargs="+",
        default=[24],
        help="One isotropic margin or three margins in D H W voxel order.",
    )
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail_empty_roi", action="store_true")
    p.add_argument(
        "--no_expand_to_aspect",
        action="store_true",
        help="Do not expand the ROI bounding box toward the output-size aspect ratio before resampling.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    meta_csv = Path(args.meta_csv).resolve()
    out_root = Path(args.out_root).resolve()
    out_csv = Path(args.out_csv).resolve() if str(args.out_csv).strip() else out_root / "cohort_preprocessed_stage2_roi.csv"
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    output_size_dhw = tuple(int(x) for x in args.output_size)
    margin_dhw = _parse_margin(args.margin_voxels)
    if any(x <= 0 for x in output_size_dhw):
        raise ValueError(f"--output_size must be positive, got {output_size_dhw}")
    if any(x < 0 for x in margin_dhw):
        raise ValueError(f"--margin_voxels must be nonnegative, got {margin_dhw}")

    df = pd.read_csv(meta_csv, dtype={args.id_col: str})
    df[args.id_col] = df[args.id_col].astype(str)
    for col in (
        args.ct_col,
        args.mask_pt_col,
        args.mask_ln_col,
        "roi_crop_status",
        "roi_crop_bbox_zyx",
        "roi_crop_margin_voxels",
        "roi_crop_output_size_dhw",
        "roi_crop_source_ct",
    ):
        if col not in df.columns:
            df[col] = ""

    runnable_indices: list[int] = []
    if "status" in df.columns:
        status_ok = df["status"].astype(str).str.lower().eq("ok")
        skipped = int((~status_ok).sum())
        if skipped:
            df.loc[~status_ok, "roi_crop_status"] = "skipped_non_ok_status"
        runnable_indices = [int(i) for i in df.index[status_ok]]
    else:
        runnable_indices = [int(i) for i in df.index]

    jobs = [
        {
            "index": int(i),
            "row": df.loc[i].to_dict(),
            "id_col": args.id_col,
            "ct_col": args.ct_col,
            "mask_pt_col": args.mask_pt_col,
            "mask_ln_col": args.mask_ln_col,
            "meta_dir": str(meta_csv.parent),
            "out_root": str(out_root),
            "out_csv_dir": str(out_csv.parent),
            "output_size_dhw": output_size_dhw,
            "margin_dhw": margin_dhw,
            "overwrite": bool(args.overwrite),
            "expand_to_aspect": not bool(args.no_expand_to_aspect),
            "fail_empty_roi": bool(args.fail_empty_roi),
        }
        for i in runnable_indices
    ]

    print(
        f"[roi-preprocess] meta_csv={meta_csv} rows={len(df)} runnable={len(jobs)} "
        f"out_root={out_root} output_size_dhw={output_size_dhw} margin_dhw={margin_dhw}"
    )

    results: list[dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for n, job in enumerate(jobs, start=1):
            result = _process_one(job)
            results.append(result)
            if n == 1 or n == len(jobs) or n % 25 == 0:
                print(f"[roi-preprocess] processed {n}/{len(jobs)} patient_id={result['pid']}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_one, job) for job in jobs]
            for n, fut in enumerate(as_completed(futures), start=1):
                result = fut.result()
                results.append(result)
                if n == 1 or n == len(jobs) or n % 25 == 0:
                    print(f"[roi-preprocess] processed {n}/{len(jobs)} patient_id={result['pid']}")

    for result in results:
        idx = int(result["index"])
        df.at[idx, args.ct_col] = result["ct_out_path"]
        df.at[idx, args.mask_pt_col] = result["mask_primary_out_path"]
        df.at[idx, args.mask_ln_col] = result["mask_nodal_out_path"]
        for key in (
            "roi_crop_status",
            "roi_crop_bbox_zyx",
            "roi_crop_margin_voxels",
            "roi_crop_output_size_dhw",
            "roi_crop_source_ct",
        ):
            df.at[idx, key] = result[key]

    df.to_csv(out_csv, index=False)
    print(f"[roi-preprocess] wrote {out_csv}")
    print(f"[roi-preprocess] train with: META_CSV={_relpath(out_csv, Path.cwd())} IMG_SIZE=\"{' '.join(map(str, output_size_dhw))}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
