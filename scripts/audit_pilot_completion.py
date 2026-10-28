#!/usr/bin/env python3
"""Write a separate read-only two-epoch pilot acceptance report (exit 0/1/2 = passed/failed/blocked)."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trifusesurv2.validation.pilot import audit_pilot, write_pilot_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-plan", type=Path)
    parser.add_argument("--execution-plan", type=Path)
    parser.add_argument("--runtime-final", type=Path)
    parser.add_argument("--trusted-root", type=Path, help="Approved root for hash-bound trusted full-pickle checkpoint reads")
    parser.add_argument("--parameter-map", action="append", default=[], metavar="MODE=JSON")
    parser.add_argument("--training-log-receipt", type=Path, help="Separately pinned completed per-mode logs for rigorous four-decimal NLL interval proof")
    parser.add_argument("--out-dir", type=Path, required=True, help="New report directory; existing outputs are never overwritten")
    args = parser.parse_args(argv)
    mappings = {}
    for value in args.parameter_map:
        mode, path = value.split("=", 1)
        if mode not in {"F2", "F3"} or mode in mappings:
            parser.error("Parameter maps must name each of F2/F3 at most once")
        mappings[mode] = Path(path)
    if not args.training_plan and not args.execution_plan:
        parser.error("At least one --training-plan or --execution-plan is required")
    report = audit_pilot(training_plan=args.training_plan, execution_plan=args.execution_plan,
                         runtime_final=args.runtime_final, trusted_root=args.trusted_root, parameter_maps=mappings,
                         training_log_receipt=args.training_log_receipt)
    write_pilot_report(report, args.out_dir)
    print(f"Pilot acceptance: {report['status']}; {args.out_dir / 'pilot_acceptance.json'}")
    return {"passed": 0, "failed": 1, "blocked": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
