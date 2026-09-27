"""
Populates PostgreSQL directly from a cleaned CSV, bypassing MQTT
entirely — same rule-engine + db_writer logic receiver.py will use
live, so this is a genuine test of steps 2 and 3, not a shortcut
around them.

Usage:
    python scripts/populate_db.py --csv data/cleaned/cleaned_ais_data.csv --limit 5000

--limit keeps this fast for local testing; drop it to load everything.
"""

import argparse
import sys
import os

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from receiver.rules import RuleEngine
from receiver import db_writer


def load_records(csv_path, limit=None):
    df = pd.read_csv(csv_path)
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
    df = df.sort_values("BaseDateTime")
    if limit:
        df = df.head(limit)
    records = []
    for _, row in df.iterrows():
        records.append({
            "mmsi": int(row["MMSI"]),
            "lat": float(row["LAT"]),
            "lon": float(row["LON"]),
            "sog": float(row["SOG"]) if pd.notna(row["SOG"]) else None,
            "cog": float(row["COG"]) if pd.notna(row["COG"]) else None,
            "heading": float(row["Heading"]) if "Heading" in row and pd.notna(row["Heading"]) else None,
            "vessel_type": row.get("VesselType"),
            "timestamp": row["BaseDateTime"],
            "channel": row.get("channel", "terrestrial"),
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only load the first N records (recommended for quick local testing)")
    parser.add_argument("--dark-threshold-min", type=float, default=10.0)
    args = parser.parse_args()

    print(f"Loading {args.csv} ...")
    records = load_records(args.csv, args.limit)
    print(f"Loaded {len(records)} records — writing to PostgreSQL...")

    engine = RuleEngine(dark_period_threshold_min=args.dark_threshold_min)
    flagged_count = 0

    for i, record in enumerate(records):
        verdict = engine.process(record)
        db_writer.insert_record(record)
        if verdict["flagged"]:
            db_writer.update_flag(
                mmsi=record["mmsi"],
                record_timestamp=record["timestamp"],
                flagged=True,
                anomaly_type=",".join(verdict["anomaly_types"]),
                risk_score=verdict["risk_score"],
            )
            flagged_count += 1

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(records)} written ({flagged_count} flagged so far)")

    print(f"\nDone. {len(records)} records written, {flagged_count} flagged.")


if __name__ == "__main__":
    main()