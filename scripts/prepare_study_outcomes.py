#!/usr/bin/env python3
"""Reconcile an endpoint source with retained eligible metadata and splits.

This metadata-only operation does not infer time origin/censoring, fit a model,
change patient assignments, or qualify clinical outcomes. Literal IDs and
numerically equal source/delivery outcomes are required before publication.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows_by_id(path, requested):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if not columns or len(columns) != len(set(columns)):
            raise ValueError("CSV columns must be present and unique.")
        if requested not in columns:
            aliases = [name for name in ("Patient_ID", "patient_id") if name in columns]
            if len(aliases) != 1:
                raise ValueError("A single unambiguous patient identifier is required.")
            requested = aliases[0]
        rows = list(reader)
    result = {}
    for row in rows:
        if None in row or None in row.values():
            raise ValueError("Malformed CSV row.")
        if all(name in row for name in ("Patient_ID", "patient_id")) and row["Patient_ID"] != row["patient_id"]:
            raise ValueError("Patient identifier aliases disagree.")
        pid = row[requested]
        if not pid or pid.strip() != pid or pid in result:
            raise ValueError("Patient identifiers must be unique, literal, and nonempty.")
        result[pid] = row
    return result


def endpoint_values(row, time_column, event_column):
    try:
        time, event = Decimal(row[time_column]), Decimal(row[event_column])
    except (InvalidOperation, KeyError, TypeError) as error:
        raise ValueError("Required endpoint value is missing or unparseable.") from error
    if not time.is_finite() or time < 0 or not event.is_finite() or event not in (0, 1):
        raise ValueError("Outcomes require finite nonnegative times and binary events.")
    return time, event


def prepare(source_path, eligible_path, split_dir, *, endpoint="OS", id_column="Patient_ID"):
    source = rows_by_id(source_path, id_column)
    delivery = rows_by_id(eligible_path, id_column)
    if not delivery or set(source) != set(delivery):
        raise ValueError("Endpoint source and eligible delivery must contain exactly the same patients.")
    time_column, event_column = endpoint + ".TIME", endpoint + ".EVENT"
    for pid, row in delivery.items():
        if endpoint_values(row, time_column, event_column) != endpoint_values(source[pid], time_column, event_column):
            raise ValueError("Source/delivery endpoint mismatch; no outcomes published.")
    seen, output, hashes, counts = set(), [], {}, {}
    for split in ("train", "val", "test"):
        path = Path(split_dir) / f"{split}_ids.txt"
        identifiers = path.read_text().splitlines()
        if (not identifiers or any(not pid or pid.strip() != pid for pid in identifiers)
                or len(set(identifiers)) != len(identifiers) or seen.intersection(identifiers)):
            raise ValueError("Split identifiers must be nonempty, unique and mutually disjoint.")
        if not set(identifiers) <= set(delivery):
            raise ValueError("Split contains patients absent from eligible metadata.")
        seen.update(identifiers)
        counts[split], hashes[split] = len(identifiers), sha256(path)
        for pid in identifiers:
            time, event = endpoint_values(source[pid], time_column, event_column)
            output.append({"Patient_ID": pid, "split": split, "endpoint": endpoint,
                           "observed_time": str(time), "event": int(event)})
    if seen != set(delivery):
        raise ValueError("Split union does not equal the eligible patient inventory.")
    receipt = {"schema_version": 1, "status": "exact_source_delivery_split_match",
               "endpoint": endpoint, "patients": len(delivery), "split_counts": counts,
               "source_csv_sha256": sha256(source_path),
               "eligible_metadata_sha256": sha256(eligible_path), "split_files_sha256": hashes,
               "time_origin_inferred": False, "censoring_rule_inferred": False,
               "scientific_qualification": False, "split_assignments_changed": False}
    return output, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--eligible-metadata", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--endpoint", choices=("OS", "DSS", "DFS"), default="OS")
    parser.add_argument("--id-column", default="Patient_ID")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, receipt = prepare(args.source_csv, args.eligible_metadata, args.split_dir,
                            endpoint=args.endpoint, id_column=args.id_column)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    path = args.out_dir / "outcomes.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt["outcomes_sha256"] = sha256(path)
    (args.out_dir / "outcome_reconciliation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
