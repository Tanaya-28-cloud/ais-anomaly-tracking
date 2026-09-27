"""Inspect AIS CSV data before feature extraction or model training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = (
    "MMSI",
    "LAT",
    "LON",
    "SOG",
    "COG",
    "BaseDateTime",
)

NUMERIC_COLUMNS = ("MMSI", "LAT", "LON", "SOG", "COG")
SOURCE_DIRECTORIES = ("terrestrial", "satellite")


@dataclass(frozen=True)
class InspectionReport:
    """Summary of one AIS dataframe's readiness for feature extraction."""

    row_count: int
    columns: tuple[str, ...]
    missing_columns: tuple[str, ...]
    invalid_counts: dict[str, int]
    missing_value_counts: dict[str, int]

    @property
    def is_ready(self) -> bool:
        return not self.missing_columns and not any(self.invalid_counts.values())


def inspect_dataframe(data: pd.DataFrame) -> InspectionReport:
    """Validate the source fields used by the initial ML feature set."""
    columns = tuple(str(column) for column in data.columns)
    missing_columns = tuple(
        column for column in REQUIRED_COLUMNS if column not in data.columns
    )

    invalid_counts: dict[str, int] = {}
    for column in NUMERIC_COLUMNS:
        if column not in data.columns:
            continue
        invalid_counts[column] = int(
            pd.to_numeric(data[column], errors="coerce").isna().sum()
        )

    missing_value_counts = {
        column: int(data[column].isna().sum())
        for column in REQUIRED_COLUMNS
        if column in data.columns
    }

    return InspectionReport(
        row_count=len(data),
        columns=columns,
        missing_columns=missing_columns,
        invalid_counts=invalid_counts,
        missing_value_counts=missing_value_counts,
    )


def inspect_csv(path: Path) -> InspectionReport:
    """Load and inspect one CSV file."""
    return inspect_dataframe(pd.read_csv(path))


def find_csv_files(data_directory: Path) -> list[Path]:
    """Return AIS source CSV files, excluding label metadata."""
    source_files = [
        path
        for directory in SOURCE_DIRECTORIES
        for path in (data_directory / directory).rglob("*.csv")
    ]
    return sorted(source_files)


def format_report(path: Path, report: InspectionReport) -> str:
    """Format a report for command-line use."""
    lines = [f"{path}: {report.row_count} rows"]
    lines.append(f"  columns: {', '.join(report.columns) or '(none)'}")
    lines.append(
        "  missing required columns: "
        f"{', '.join(report.missing_columns) or '(none)'}"
    )
    invalid = ", ".join(
        f"{column}={count}"
        for column, count in report.invalid_counts.items()
        if count
    )
    lines.append(f"  invalid numeric values: {invalid or '(none)'}")
    missing = ", ".join(
        f"{column}={count}"
        for column, count in report.missing_value_counts.items()
        if count
    )
    lines.append(f"  missing values: {missing or '(none)'}")
    lines.append(f"  ready for feature extraction: {'yes' if report.is_ready else 'no'}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory", type=Path)
    args = parser.parse_args()

    csv_files = find_csv_files(args.data_directory)
    if not csv_files:
        print(f"No CSV files found under {args.data_directory}")
        return 1

    exit_code = 0
    for path in csv_files:
        report = inspect_csv(path)
        print(format_report(path, report))
        if not report.is_ready:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
