#!/usr/bin/env python3
"""Catalogue and aggregate locally downloaded Telecom Italia Milan files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.telecom_italia_migration.telecom_data import (  # noqa: E402
    aggregate_files,
    fetch_catalog,
    write_catalog,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "telecom_italia" / "raw")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "telecom_italia" / "processed" / "milan_internet_activity.csv",
    )
    parser.add_argument("--catalog-only", action="store_true")
    parser.add_argument("--grid-start", type=int, default=5050)
    parser.add_argument("--grid-count", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = fetch_catalog()
    catalog_path = ROOT / "data" / "telecom_italia" / "official_catalog.json"
    write_catalog(catalog_path, catalog)
    print(f"catalog={catalog_path} files={len(catalog)} bytes={sum(row.size_bytes for row in catalog)}")
    if args.catalog_only:
        return 0
    raw_files = sorted(args.raw_dir.glob("sms-call-internet-mi-*.txt"))
    if not raw_files:
        raise FileNotFoundError(
            f"no official raw files found in {args.raw_dir}; download them from the DOI page first"
        )
    expected = {row.name: row.md5 for row in catalog}
    report = aggregate_files(
        raw_files,
        args.output,
        grid_ids=range(args.grid_start, args.grid_start + args.grid_count),
        expected_md5=expected,
    )
    print(
        f"output={args.output} timestamps={report['timestamps']} "
        f"rows={report['rows_written']} malformed={report['malformed_source_rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
