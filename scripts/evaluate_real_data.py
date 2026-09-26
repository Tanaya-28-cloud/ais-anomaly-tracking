import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from receiver.rules import RuleEngine


def load_records(csv_path):
    df = pd.read_csv(csv_path)
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
    df = df.sort_values("BaseDateTime")

    records = []

    for _, row in df.iterrows():
        records.append({
            "mmsi": int(row["MMSI"]),
            "lat": float(row["LAT"]),
            "lon": float(row["LON"]),
            "sog": float(row["SOG"]) if pd.notna(row["SOG"]) else None,
            "cog": float(row["COG"]) if pd.notna(row["COG"]) else None,
            "vessel_type": row.get("VesselType"),
            "timestamp": row["BaseDateTime"],
            "channel": row.get("channel", "terrestrial"),
        })

    return records


def main():
    csv_path = r"D:\tanaya\.vscode\BE 1\cleaned_ais_data.csv"
    labels_path = r"D:\tanaya\.vscode\BE 1\injected_anomalies.csv"

    print("Loading AIS data...")
    records = load_records(csv_path)
    print(f"Loaded {len(records)} records")

    engine = RuleEngine()
    flagged_mmsis = set()

    for i, record in enumerate(records):
        verdict = engine.process(record)

        if verdict["flagged"]:
            flagged_mmsis.add(record["mmsi"])

        if i % 500 == 0 and i > 0:
            engine.check_dark_vessels()

    labels = pd.read_csv(labels_path)

    label_mmsis = set(labels["mmsi"])

    missed_mmsis = label_mmsis - flagged_mmsis
    caught_mmsis = label_mmsis & flagged_mmsis

    print("\n========== RESULTS ==========")
    print(f"Injected anomalous vessels : {len(label_mmsis)}")
    print(f"Caught                     : {len(caught_mmsis)}")
    print(f"Missed                     : {len(missed_mmsis)}")

    print("\n========== MISSED ANOMALIES ==========")

    missed_labels = labels[labels["mmsi"].isin(missed_mmsis)]

    print(
        missed_labels[
            ["mmsi", "anomaly_type", "start_time", "end_time", "notes"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()