#!/usr/bin/env python3
"""Join retained full-cohort metadata and evidence; never load images or re-extract radiomics."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trifusesurv2.validation.cohort_quality import summarize_cohort, write_cohort_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("metadata-csv", "source-metadata-csv", "radiomics-csv", "development-receipt", "builder-receipt", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sampling-audit-dir", type=Path)
    parser.add_argument("--visual-review-receipt", type=Path)
    parser.add_argument("--representative-receipt", type=Path, help="Receipt hash-bound by the historical visual review, for exact manifest applicability checks")
    parser.add_argument("--path-map", action="append", default=[], metavar="REMOTE=LOCAL")
    parser.add_argument("--expected-counts", type=int, nargs=3, default=(613, 519, 94), metavar=("PREPARED", "ELIGIBLE", "MISSING"))
    parser.add_argument("--review-volume-ratio", type=float, nargs=2, default=(.8, 1.2), metavar=("LOW", "HIGH"))
    parser.add_argument("--ordinary-sample-size", type=int, default=5)
    args = parser.parse_args(argv)
    report = summarize_cohort(metadata_csv=args.metadata_csv, source_metadata_csv=args.source_metadata_csv,
                              radiomics_csv=args.radiomics_csv, development_receipt=args.development_receipt,
                              builder_receipt=args.builder_receipt, path_mappings=[value.split("=", 1) for value in args.path_map],
                              sampling_audit_dir=args.sampling_audit_dir, visual_review_receipt=args.visual_review_receipt,
                              representative_receipt=args.representative_receipt,
                              expected_counts=args.expected_counts, ratio_bounds=args.review_volume_ratio,
                              ordinary_sample_size=args.ordinary_sample_size)
    write_cohort_report(report, args.out_dir)
    print(f"Cohort accounting: {report['accounting_gate']}; review: {report['review_gate']}; {args.out_dir}")
    return 1 if report["accounting_gate"] != "passed" else 2 if report["review_gate"] == "pending" else 0


if __name__ == "__main__":
    raise SystemExit(main())
