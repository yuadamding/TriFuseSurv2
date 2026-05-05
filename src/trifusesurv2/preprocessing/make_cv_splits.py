#!/usr/bin/env python3
"""
trifusesurv2.preprocessing.make_cv_splits

Reproduces the packaged train/val/test splits used by the survival trainer:
- same status filtering
- same QC filtering
- same stratified K-fold construction
- same per-fold train/val split seeding

Outputs:
  out_dir/
    splits.csv                      (patient_id, fold, split)
    fold_00/train_ids.txt, val_ids.txt, test_ids.txt
    ...

PYTHONPATH=src python3 -m trifusesurv2.preprocessing.make_cv_splits \
  --meta_csv OPSCC_preprocessed_128/cohort_preprocessed.csv \
  --qc_report OPSCC_preprocessed_128/qc/qc_report.csv \
  --qc_policy none --qc_drop_air_gt 0 \
  --endpoint OS \
  --cv_folds 4 --val_frac 0.2 --split_seed 1 \
  --out_dir runs/opscc_splits_<endpoint>_seed1

PYTHONPATH=src python3 -m trifusesurv2.preprocessing.make_cv_splits \
  --meta_csv OPSCC_preprocessed_128/cohort_preprocessed.csv \
  --qc_report OPSCC_preprocessed_128/qc/qc_report.csv \
  --qc_policy none --qc_drop_air_gt 0 \
  --endpoint OS \
  --cv_folds 1 --val_frac 0.2 --split_seed 1 \
  --out_dir runs/opscc_splits_<endpoint>_seed3

When `--cv_folds 1`, this writes a single `fold_00/` with train/val IDs and an empty
`test_ids.txt`, which keeps the downstream packaged trainer interface consistent.
"""

import os
import argparse
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


from trifusesurv2.schema import ENDPOINT_MAP


def _validate_patient_ids(patient_ids: pd.Series, *, context: str) -> pd.Series:
    patient_ids = patient_ids.astype(str).str.strip()
    blank = patient_ids == ""
    if blank.any():
        raise ValueError(f"{context} contains blank patient_id values.")

    dup_ids = patient_ids[patient_ids.duplicated(keep=False)].unique().tolist()
    if dup_ids:
        raise ValueError(f"{context} contains duplicate patient_id values: {dup_ids[:10]}")
    return patient_ids


def _validate_binary_events(events, *, context: str) -> np.ndarray:
    arr = np.asarray(events, dtype=int)
    bad = sorted(int(x) for x in np.unique(arr) if int(x) not in (0, 1))
    if bad:
        raise ValueError(f"{context} must be binary with values in {{0,1}}, got {bad}")
    return arr


def _endpoint_valid_mask(df: pd.DataFrame, time_col: str, event_col: str) -> pd.Series:
    times = pd.to_numeric(df[time_col], errors="coerce")
    events = pd.to_numeric(df[event_col], errors="coerce")
    return times.notna() & events.notna() & (times > 0) & events.isin([0, 1])


def load_items_for_splits(
    meta_csv: str,
    endpoint: str,
    require_status_ok: bool = True,
    require_survival_matched: bool = True,
) -> List[Dict[str, Any]]:
    df = pd.read_csv(meta_csv)

    if endpoint not in ENDPOINT_MAP:
        raise ValueError(f"--endpoint must be one of {list(ENDPOINT_MAP.keys())}, got {endpoint}")

    tcol, ecol = ENDPOINT_MAP[endpoint]
    needed = ["patient_id", tcol, ecol]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Metafile missing required columns: {missing}")

    df = df.copy()

    if require_status_ok and "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"]

    df[tcol] = pd.to_numeric(df[tcol], errors="coerce")
    df[ecol] = pd.to_numeric(df[ecol], errors="coerce")
    if require_survival_matched:
        df = df[df[tcol].notna() & df[ecol].notna()]
    df = df.dropna(subset=[tcol, ecol])
    df["patient_id"] = _validate_patient_ids(df["patient_id"], context=f"{meta_csv} after filtering")
    df[ecol] = df[ecol].astype(int)
    _validate_binary_events(df[ecol].to_numpy(), context=f"{meta_csv}:{ecol}")

    items: List[Dict[str, Any]] = []
    for _, r in df.iterrows():  # preserves file row order (matches your training script)
        items.append(
            {
                "patient_id": str(r["patient_id"]),
                "time": float(r[tcol]),
                "event": int(r[ecol]),
            }
        )
    return items


def load_primary_and_aux_train_ids(
    meta_csv: str,
    endpoint: str,
    require_status_ok: bool = True,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    df = pd.read_csv(meta_csv)

    if endpoint not in ENDPOINT_MAP:
        raise ValueError(f"--endpoint must be one of {list(ENDPOINT_MAP.keys())}, got {endpoint}")

    tcol, ecol = ENDPOINT_MAP[endpoint]
    needed = ["patient_id", *[c for pair in ENDPOINT_MAP.values() for c in pair]]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Metafile missing required columns: {missing}")

    df = df.copy()
    if require_status_ok and "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"]

    df["patient_id"] = _validate_patient_ids(df["patient_id"], context=f"{meta_csv} after filtering")

    valid_any = pd.Series(False, index=df.index)
    for time_col, event_col in ENDPOINT_MAP.values():
        valid_ep = _endpoint_valid_mask(df, time_col, event_col)
        if bool(valid_ep.any()):
            _validate_binary_events(
                pd.to_numeric(df.loc[valid_ep, event_col], errors="raise").astype(int).to_numpy(),
                context=f"{meta_csv}:{event_col}",
            )
        valid_any = valid_any | valid_ep

    primary_valid = _endpoint_valid_mask(df, tcol, ecol)
    primary_items = []
    for _, r in df.loc[primary_valid].iterrows():
        primary_items.append(
            {
                "patient_id": str(r["patient_id"]),
                "time": float(r[tcol]),
                "event": int(pd.to_numeric(r[ecol], errors="raise")),
            }
        )

    aux_only_ids = (
        df.loc[valid_any & (~primary_valid), "patient_id"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    return primary_items, aux_only_ids


def qc_filter_items(
    items: List[Dict[str, Any]],
    qc_report: str,
    qc_policy: str,
    qc_id_col: str,
    qc_severity_col: str,
    qc_drop_if_contains: List[str],
    qc_drop_air_gt: float,
) -> List[Dict[str, Any]]:
    if qc_policy == "none" or not qc_report:
        return items

    keep_ids = load_qc_keep_ids(
        qc_report=qc_report,
        qc_policy=qc_policy,
        qc_id_col=qc_id_col,
        qc_severity_col=qc_severity_col,
        qc_drop_if_contains=qc_drop_if_contains,
        qc_drop_air_gt=qc_drop_air_gt,
    )

    item_ids = set(str(it["patient_id"]) for it in items)
    qc_ids_all = load_qc_all_ids(qc_report, qc_id_col)
    missing_in_qc = item_ids - qc_ids_all
    if missing_in_qc:
        print(f"[qc] warning: {len(missing_in_qc)} item(s) not found in QC report; they will be dropped.")

    before = len(items)
    items2 = [it for it in items if str(it["patient_id"]) in keep_ids]
    print(f"[qc] applied {qc_policy}: kept {len(items2)}/{before} (dropped {before-len(items2)})")
    return items2


def load_qc_all_ids(qc_report: str, qc_id_col: str) -> set[str]:
    qc = pd.read_csv(qc_report)
    if qc_id_col not in qc.columns:
        raise ValueError(f"QC report missing '{qc_id_col}'")
    return set(qc[qc_id_col].astype(str).tolist())


def load_qc_keep_ids(
    *,
    qc_report: str,
    qc_policy: str,
    qc_id_col: str,
    qc_severity_col: str,
    qc_drop_if_contains: List[str],
    qc_drop_air_gt: float,
) -> set[str]:
    if qc_policy == "none" or not qc_report:
        return set()

    qc = pd.read_csv(qc_report)
    if qc_id_col not in qc.columns:
        raise ValueError(f"QC report missing '{qc_id_col}'")

    qc = qc.copy()
    qc[qc_id_col] = qc[qc_id_col].astype(str)

    if qc_severity_col not in qc.columns:
        raise ValueError(f"QC report must include '{qc_severity_col}' column.")
    sev = qc[qc_severity_col].astype(str).str.lower()

    if qc_policy == "drop_fail":
        keep = sev != "fail"
    elif qc_policy == "drop_fail_warn":
        keep = sev == "pass"
    else:
        raise ValueError(f"Unknown qc_policy: {qc_policy}")

    flag_col = None
    for c in ["all_flags", "flags", "fail_flags", "warn_flags", "outlier_flags"]:
        if c in qc.columns:
            flag_col = c
            break
    if qc_drop_if_contains and flag_col is not None:
        txt = qc[flag_col].fillna("").astype(str)
        for sub in qc_drop_if_contains:
            keep = keep & (~txt.str.contains(str(sub), regex=False))

    if qc_drop_air_gt > 0 and "union_in_air_frac" in qc.columns:
        air = pd.to_numeric(qc["union_in_air_frac"], errors="coerce")
        keep = keep & ~(air > float(qc_drop_air_gt))

    return set(qc.loc[keep, qc_id_col].astype(str).tolist())


def _build_strata(events: np.ndarray, times: Optional[np.ndarray], n_time_bins: int) -> np.ndarray:
    """Build composite strata from (event, coarse time bin).

    When ``times`` is None or ``n_time_bins <= 0``, falls back to event-only
    stratification for backward compatibility.
    """
    events = np.asarray(events, dtype=int)
    if times is None or int(n_time_bins) <= 0:
        return events

    times = np.asarray(times, dtype=float)
    event_mask = events == 1
    if event_mask.sum() < max(2, int(n_time_bins)):
        return events

    event_times = times[event_mask]
    try:
        quantiles = np.quantile(event_times, np.linspace(0, 1, int(n_time_bins) + 1)[1:-1])
    except Exception:
        return events
    quantiles = np.unique(quantiles)
    if quantiles.size == 0:
        return events

    time_bin = np.digitize(times, quantiles).astype(int)
    return events * (int(n_time_bins) + 1) + time_bin


def _split_words(value: str) -> List[str]:
    out: List[str] = []
    for part in str(value or "").replace(",", " ").split():
        part = part.strip()
        if part:
            out.append(part)
    return out


def _extra_strata_from_meta(meta: pd.DataFrame, patient_ids: List[str], cols: List[str]) -> Optional[np.ndarray]:
    cols = [c for c in cols if c in meta.columns]
    if not cols:
        return None
    m = meta.copy()
    m["patient_id"] = m["patient_id"].astype(str)
    m = m.drop_duplicates("patient_id").set_index("patient_id")
    labels = []
    for pid in patient_ids:
        if pid not in m.index:
            labels.append("missing")
            continue
        parts = []
        for col in cols:
            value = m.at[pid, col]
            if pd.isna(value) or str(value).strip() == "":
                value = "unknown"
            parts.append(f"{col}={str(value).strip().lower()}")
        labels.append("|".join(parts))
    return np.asarray(labels, dtype=object)


def _combine_strata(base: np.ndarray, extra: Optional[np.ndarray]) -> np.ndarray:
    if extra is None:
        return np.asarray(base)
    labels = np.asarray([f"{int(b)}|{str(e)}" for b, e in zip(base, extra)], dtype=object)
    _, codes = np.unique(labels, return_inverse=True)
    return codes.astype(int)


def stratified_kfold_indices(
    events: np.ndarray,
    k: int,
    seed: int,
    *,
    times: Optional[np.ndarray] = None,
    n_time_bins: int = 0,
    extra_strata: Optional[np.ndarray] = None,
) -> List[List[int]]:
    if int(k) < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    rng = np.random.default_rng(seed)
    events = _validate_binary_events(events, context="events")

    strata = _combine_strata(_build_strata(events, times, n_time_bins), extra_strata)
    unique_strata = np.unique(strata)
    rng.shuffle(unique_strata)

    folds: List[List[int]] = [[] for _ in range(k)]
    cursor = 0
    for s in unique_strata:
        members = np.where(strata == s)[0]
        rng.shuffle(members)
        for i, idx in enumerate(members):
            folds[(cursor + i) % k].append(int(idx))
        cursor = (cursor + len(members)) % k

    for f in folds:
        rng.shuffle(f)
    return folds


def stratified_train_val_split(
    indices: List[int],
    events: np.ndarray,
    val_frac: float,
    seed: int,
    *,
    extra_strata: Optional[np.ndarray] = None,
) -> Tuple[List[int], List[int]]:
    if not 0 <= float(val_frac) < 1:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")

    rng = np.random.default_rng(seed)
    idx = np.array(indices, dtype=int)
    if len(idx) <= 1 or val_frac <= 0:
        return [int(x) for x in idx.tolist()], []

    n_val = int(round(len(idx) * val_frac))
    n_val = min(n_val, len(idx) - 1)
    if n_val <= 0:
        return [int(x) for x in idx.tolist()], []

    ev = _validate_binary_events(events, context="events")[idx]
    if extra_strata is None:
        strata = ev
    else:
        strata = _combine_strata(ev, np.asarray(extra_strata, dtype=object)[idx])

    groups = []
    for s in np.unique(strata):
        members = idx[strata == s]
        rng.shuffle(members)
        raw = float(len(members)) * float(val_frac)
        take = min(int(np.floor(raw)), max(0, len(members) - 1))
        groups.append({"members": members, "take": take, "remainder": raw - float(np.floor(raw))})

    current = sum(int(g["take"]) for g in groups)
    for g in sorted(groups, key=lambda item: item["remainder"], reverse=True):
        if current >= n_val:
            break
        if int(g["take"]) < len(g["members"]):
            g["take"] = int(g["take"]) + 1
            current += 1

    val_parts = [g["members"][: int(g["take"])] for g in groups if int(g["take"]) > 0]
    if val_parts:
        val_idx = np.concatenate(val_parts)
    else:
        val_idx = idx[:n_val]
    rng.shuffle(val_idx)

    val_set = set(int(x) for x in val_idx.tolist())
    tr_idx = [int(x) for x in idx.tolist() if int(x) not in val_set]
    va_idx = [int(x) for x in idx.tolist() if int(x) in val_set]
    return tr_idx, va_idx


def make_fold_splits(
    events: np.ndarray,
    cv_folds: int,
    val_frac: float,
    split_seed: int,
    *,
    times: Optional[np.ndarray] = None,
    n_time_bins: int = 0,
    extra_strata: Optional[np.ndarray] = None,
) -> List[Dict[str, List[int]]]:
    events = _validate_binary_events(events, context="events")
    cv_folds = int(cv_folds)
    if cv_folds < 1:
        raise ValueError(f"cv_folds must be >= 1, got {cv_folds}")

    if cv_folds == 1:
        tr_idx, va_idx = stratified_train_val_split(
            list(range(len(events))),
            events,
            val_frac,
            split_seed + 1000,
            extra_strata=extra_strata,
        )
        return [{"train": tr_idx, "val": va_idx, "test": []}]

    if len(events) < cv_folds:
        raise ValueError(f"Not enough samples ({len(events)}) for {cv_folds}-fold CV after QC.")

    folds = stratified_kfold_indices(events, cv_folds, split_seed, times=times, n_time_bins=n_time_bins, extra_strata=extra_strata)
    split_defs: List[Dict[str, List[int]]] = []
    for fold_idx in range(cv_folds):
        test_idx = folds[fold_idx]
        trainval_idx = [i for other_idx, fold in enumerate(folds) if other_idx != fold_idx for i in fold]
        tr_idx, va_idx = stratified_train_val_split(
            trainval_idx,
            events,
            val_frac,
            split_seed + 1000 + fold_idx,
            extra_strata=extra_strata,
        )
        split_defs.append({"train": tr_idx, "val": va_idx, "test": test_idx})
    return split_defs


def write_fold_balance(
    path: str,
    *,
    split_rows: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    meta_csv: str,
    balance_cols: List[str],
) -> None:
    meta = pd.read_csv(meta_csv, dtype={"patient_id": str})
    meta["patient_id"] = meta["patient_id"].astype(str)
    item_df = pd.DataFrame(items)
    item_df["patient_id"] = item_df["patient_id"].astype(str)
    split_df = pd.DataFrame(split_rows)
    if split_df.empty:
        pd.DataFrame().to_csv(path, index=False)
        return
    df = split_df.merge(item_df[["patient_id", "time", "event"]], on="patient_id", how="left")
    extra_cols = [c for c in balance_cols if c in meta.columns]
    if extra_cols:
        df = df.merge(meta[["patient_id", *extra_cols]].drop_duplicates("patient_id"), on="patient_id", how="left")

    rows: List[Dict[str, Any]] = []
    for (fold, split), sub in df.groupby(["fold", "split"], dropna=False):
        event = pd.to_numeric(sub["event"], errors="coerce")
        time = pd.to_numeric(sub["time"], errors="coerce")
        rows.extend(
            [
                {"fold": fold, "split": split, "metric": "n", "value": "all", "count": int(len(sub)), "fraction": 1.0},
                {"fold": fold, "split": split, "metric": "event_count", "value": "1", "count": int((event == 1).sum()), "fraction": float((event == 1).mean()) if event.notna().any() else float("nan")},
                {"fold": fold, "split": split, "metric": "median_time_days", "value": "median", "count": float(time.median()) if time.notna().any() else float("nan"), "fraction": ""},
            ]
        )
        for col in extra_cols:
            values = sub[col].fillna("unknown").astype(str)
            denom = max(1, len(values))
            for value, count in values.value_counts(dropna=False).sort_index().items():
                rows.append(
                    {
                        "fold": fold,
                        "split": split,
                        "metric": col,
                        "value": value,
                        "count": int(count),
                        "fraction": float(count) / float(denom),
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_ids(path: str, ids: List[str]):
    with open(path, "w") as f:
        for x in ids:
            f.write(f"{x}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", type=str, required=True)
    ap.add_argument("--endpoint", type=str, default="OS", choices=["OS", "DSS", "DFS"])

    # Match your training script flags
    ap.add_argument("--keep_bad_status", action="store_true")
    ap.add_argument("--keep_unmatched_survival", action="store_true")
    ap.add_argument("--include_aux_only_train", dest="include_aux_only_train", action="store_true")
    ap.add_argument("--no_include_aux_only_train", dest="include_aux_only_train", action="store_false")
    ap.set_defaults(include_aux_only_train=True)

    # QC
    ap.add_argument("--qc_report", type=str, default="")
    ap.add_argument("--qc_policy", type=str, default="none", choices=["none", "drop_fail", "drop_fail_warn"])
    ap.add_argument("--qc_id_col", type=str, default="patient_id")
    ap.add_argument("--qc_severity_col", type=str, default="severity")
    ap.add_argument("--qc_drop_if_contains", type=str, action="append", default=[])
    ap.add_argument("--qc_drop_air_gt", type=float, default=-1.0)

    # CV
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=1)
    ap.add_argument("--stratify_time_bins", type=int, default=4,
                    help="Stratify on (event, coarse time bin). 0 = event-only (legacy).")
    ap.add_argument(
        "--stratify_cols",
        type=str,
        default="HPV p16 T N NSTAGE TX scanner site smoke_group crop_mode",
        help="Optional metadata/QC columns to include in split strata when present and report in fold_balance.csv.",
    )
    ap.add_argument(
        "--balance_cols",
        type=str,
        default="",
        help="Optional metadata columns for fold_balance.csv. Defaults to --stratify_cols.",
    )

    ap.add_argument("--out_dir", type=str, default="cv_splits")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    items = load_items_for_splits(
        args.meta_csv,
        args.endpoint,
        require_status_ok=not args.keep_bad_status,
        require_survival_matched=not args.keep_unmatched_survival,
    )

    items = qc_filter_items(
        items,
        qc_report=args.qc_report,
        qc_policy=args.qc_policy,
        qc_id_col=args.qc_id_col,
        qc_severity_col=args.qc_severity_col,
        qc_drop_if_contains=args.qc_drop_if_contains,
        qc_drop_air_gt=args.qc_drop_air_gt,
    )

    if not items:
        raise ValueError("No samples remain after filtering and QC.")

    aux_only_train_ids: List[str] = []
    if args.include_aux_only_train:
        _, aux_only_train_ids = load_primary_and_aux_train_ids(
            args.meta_csv,
            args.endpoint,
            require_status_ok=not args.keep_bad_status,
        )
        if args.qc_policy != "none" and args.qc_report:
            keep_ids = load_qc_keep_ids(
                qc_report=args.qc_report,
                qc_policy=args.qc_policy,
                qc_id_col=args.qc_id_col,
                qc_severity_col=args.qc_severity_col,
                qc_drop_if_contains=args.qc_drop_if_contains,
                qc_drop_air_gt=args.qc_drop_air_gt,
            )
            aux_only_train_ids = [pid for pid in aux_only_train_ids if pid in keep_ids]
        if aux_only_train_ids:
            print(f"[mtl] adding {len(aux_only_train_ids)} aux-only training case(s) with non-primary survival labels")

    events = np.array([it["event"] for it in items], dtype=int)
    times = np.array([it["time"] for it in items], dtype=float)
    item_ids = [str(it["patient_id"]) for it in items]
    meta_for_balance = pd.read_csv(args.meta_csv, dtype={"patient_id": str})
    stratify_cols = _split_words(args.stratify_cols)
    balance_cols = _split_words(args.balance_cols) or stratify_cols
    extra_strata = _extra_strata_from_meta(meta_for_balance, item_ids, stratify_cols)
    split_defs = make_fold_splits(
        events, args.cv_folds, args.val_frac, args.split_seed,
        times=times, n_time_bins=args.stratify_time_bins, extra_strata=extra_strata,
    )

    rows = []
    for f, split_def in enumerate(split_defs):
        train_ids = [items[i]["patient_id"] for i in split_def["train"]]
        val_ids = [items[i]["patient_id"] for i in split_def["val"]]
        test_ids = [items[i]["patient_id"] for i in split_def["test"]]
        if aux_only_train_ids:
            train_ids = train_ids + [pid for pid in aux_only_train_ids if pid not in set(train_ids)]

        print(f"[fold {f:02d}] train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

        fold_dir = os.path.join(args.out_dir, f"fold_{f:02d}")
        os.makedirs(fold_dir, exist_ok=True)
        write_ids(os.path.join(fold_dir, "train_ids.txt"), train_ids)
        write_ids(os.path.join(fold_dir, "val_ids.txt"), val_ids)
        write_ids(os.path.join(fold_dir, "test_ids.txt"), test_ids)

        for pid in train_ids:
            rows.append(
                {
                    "patient_id": pid,
                    "fold": f,
                    "split": "train",
                    "cohort": "aux_only_train" if pid in set(aux_only_train_ids) else "primary",
                }
            )
        for pid in val_ids:
            rows.append({"patient_id": pid, "fold": f, "split": "val", "cohort": "primary"})
        for pid in test_ids:
            rows.append({"patient_id": pid, "fold": f, "split": "test", "cohort": "primary"})

    out_csv = os.path.join(args.out_dir, "splits.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    balance_csv = os.path.join(args.out_dir, "fold_balance.csv")
    write_fold_balance(balance_csv, split_rows=rows, items=items, meta_csv=args.meta_csv, balance_cols=balance_cols)
    print(f"[done] wrote {out_csv}")
    print(f"[done] wrote {balance_csv}")


if __name__ == "__main__":
    main()
