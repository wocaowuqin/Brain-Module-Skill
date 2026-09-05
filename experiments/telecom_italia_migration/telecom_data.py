"""Acquire metadata and aggregate Telecom Italia Milan activity files."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import quote
from urllib.request import Request, urlopen


DATASET_DOI = "doi:10.7910/DVN/EGZHFV"
DATAVERSE_BASE = "https://dataverse.harvard.edu"
METADATA_URL = (
    f"{DATAVERSE_BASE}/api/datasets/:persistentId/"
    f"?persistentId={quote(DATASET_DOI, safe=':/')}"
)


@dataclass(frozen=True)
class SourceFile:
    file_id: int
    name: str
    size_bytes: int
    persistent_id: str
    md5: str

    @property
    def access_url(self) -> str:
        return f"{DATAVERSE_BASE}/api/access/datafile/{self.file_id}"


def fetch_catalog(timeout_seconds: float = 30.0) -> list[SourceFile]:
    """Read the public Dataverse catalogue without downloading data files."""

    request = Request(
        METADATA_URL,
        headers={
            "Accept": "application/json",
            "User-Agent": "hrl-marl-reconfig-starter/1.0 (academic reproduction)",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.load(response)
    if payload.get("status") != "OK":
        raise RuntimeError(f"Dataverse metadata request failed: {payload}")
    rows: list[SourceFile] = []
    for item in payload["data"]["latestVersion"]["files"]:
        data = item["dataFile"]
        name = str(item["label"])
        if not name.startswith("sms-call-internet-mi-") or not name.endswith(".txt"):
            continue
        checksum = data.get("checksum") or {}
        rows.append(
            SourceFile(
                file_id=int(data["id"]),
                name=name,
                size_bytes=int(data["filesize"]),
                persistent_id=str(data["persistentId"]),
                md5=str(checksum.get("value", data.get("md5", ""))).lower(),
            )
        )
    return sorted(rows, key=lambda row: row.name)


def write_catalog(path: Path, rows: Sequence[SourceFile]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_doi": DATASET_DOI,
        "metadata_url": METADATA_URL,
        "license": "ODbL-1.0 (see dataset terms)",
        "guestbook_required": True,
        "file_count": len(rows),
        "total_size_bytes": sum(row.size_bytes for row in rows),
        "files": [asdict(row) | {"access_url": row.access_url} for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _parse_number(value: bytes) -> float:
    if not value:
        return 0.0
    return float(value)


def aggregate_files(
    raw_files: Sequence[Path],
    output_csv: Path,
    *,
    grid_ids: Iterable[int],
    expected_md5: dict[str, str] | None = None,
) -> dict[str, object]:
    """Aggregate Internet activity over country codes for selected Milan grids.

    Source columns are grid id, Unix epoch milliseconds, country code, SMS-in,
    SMS-out, call-in, call-out, and Internet activity. Empty activity fields
    are interpreted as zero, matching the dataset representation.
    """

    selected = {int(value) for value in grid_ids}
    if not selected:
        raise ValueError("at least one grid id is required")
    expected_md5 = expected_md5 or {}
    values: dict[tuple[int, int], float] = {}
    source_rows: list[dict[str, object]] = []
    malformed = 0
    for path in raw_files:
        digest = hashlib.md5()
        row_count = 0
        selected_count = 0
        with path.open("rb") as handle:
            for raw in handle:
                digest.update(raw)
                row_count += 1
                columns = raw.rstrip(b"\r\n").split(b"\t")
                if len(columns) != 8:
                    malformed += 1
                    continue
                try:
                    grid_id = int(columns[0])
                    timestamp_ms = int(columns[1])
                except ValueError:
                    malformed += 1
                    continue
                if grid_id not in selected:
                    continue
                try:
                    activity = _parse_number(columns[7])
                except ValueError:
                    malformed += 1
                    continue
                key = (timestamp_ms, grid_id)
                values[key] = values.get(key, 0.0) + activity
                selected_count += 1
        actual = digest.hexdigest()
        expected = expected_md5.get(path.name, "").lower()
        if expected and actual != expected:
            raise ValueError(f"MD5 mismatch for {path.name}: {actual} != {expected}")
        source_rows.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "md5": actual,
                "md5_verified": bool(expected),
                "rows": row_count,
                "selected_rows": selected_count,
            }
        )

    timestamps = sorted({timestamp for timestamp, _ in values})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_ms", "timestamp_utc", "grid_id", "internet_activity"])
        for timestamp in timestamps:
            utc = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).isoformat()
            for grid_id in sorted(selected):
                writer.writerow([timestamp, utc, grid_id, values.get((timestamp, grid_id), 0.0)])

    report: dict[str, object] = {
        "dataset_doi": DATASET_DOI,
        "source_files": source_rows,
        "grid_ids": sorted(selected),
        "timestamps": len(timestamps),
        "rows_written": len(timestamps) * len(selected),
        "malformed_source_rows": malformed,
        "output_csv": str(output_csv.resolve()),
        "zero_filled_cells": sum(
            int((timestamp, grid_id) not in values)
            for timestamp in timestamps
            for grid_id in selected
        ),
    }
    output_csv.with_suffix(".metadata.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def load_activity_csv(path: Path) -> tuple[list[int], dict[int, list[float]]]:
    """Load long-form aggregate CSV into aligned per-grid time series."""

    rows: list[tuple[int, int, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (int(row["timestamp_ms"]), int(row["grid_id"]), float(row["internet_activity"]))
            )
    timestamps = sorted({row[0] for row in rows})
    grid_ids = sorted({row[1] for row in rows})
    lookup = {(timestamp, grid): value for timestamp, grid, value in rows}
    series = {
        grid: [lookup.get((timestamp, grid), 0.0) for timestamp in timestamps]
        for grid in grid_ids
    }
    return timestamps, series
