"""CLI for data quality reports."""

import argparse
import json
from pathlib import Path

import pandas as pd

from health_dq.checks import run_quality_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tabular data quality checks")
    parser.add_argument("--input", required=True, help="CSV input path")
    parser.add_argument("--reference", help="Optional reference CSV for drift check")
    parser.add_argument("--report", default="dq_report.json", help="Output JSON path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    ref = pd.read_csv(args.reference) if args.reference else None
    report = run_quality_report(df, reference=ref)

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Report saved to {out}")
    print(f"Rows: {report['summary']['rows']}, duplicates: {report['duplicates']['duplicate_rows']}")


if __name__ == "__main__":
    main()
