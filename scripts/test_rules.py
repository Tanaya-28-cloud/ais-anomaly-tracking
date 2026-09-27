"""
Test harness for the rule engine — replays a CSV through RuleEngine
in timestamp order, exactly like the live receiver eventually will,
but with no MQTT/networking involved.

Usage:
    python scripts/test_rules.py --csv data/cleaned/cleaned_ais_data.csv --labels data/labels/injected_anomalies.csv

If --labels is omitted, it just prints what got flagged without
scoring against ground truth.

Requires: pip install pandas
"""

import argparse
import sys
import os

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from receiver.rules import RuleEngine

# Ground truth uses different terminology than the engine's internal
# rule names. Mapped here, in the evaluation layer only — RuleEngine's
# own vocabulary is intentionally left unchanged (per project instructions
# not to rename the existing API).
LABEL_TO_ENGINE_TYPE = {
    "dark_period": "dark_event",
    "type_mismatch": "identity_mismatch",
    "mmsi_duplication": "mmsi_duplication",
    "position_jump": "position_jump",
}


def load_records(csv_path):
    df = pd.read_csv(csv_path)
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
    df = df.sort_values("BaseDateTime")  # critical — engine assumes chronological order
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


def run_engine(records, dark_check_interval_min, dark_threshold_min):
    """
    Replays records through the engine, scheduling dark-vessel sweeps
    based on SIMULATION time (the timestamp of the data itself), not
    record count and not wall-clock time. A sweep runs whenever at
    least `dark_check_interval_min` of simulated time has elapsed
    since the last sweep.
    """
    engine = RuleEngine(dark_period_threshold_min=dark_threshold_min)
    flagged_events = []
    last_dark_check_time = None

    for record in records:
        current_time = record["timestamp"]

        # Check whether a dark sweep is due BEFORE processing this
        # record, using state as of just before this message arrives.
        # This matters: if this very record is what ends a vessel's
        # long silence, process() is about to refresh its last_seen to
        # "now," which would hide that just-ended gap from the sweep
        # if we checked afterward instead.
        if last_dark_check_time is None:
            last_dark_check_time = current_time
        elapsed_min = (current_time - last_dark_check_time).total_seconds() / 60
        if elapsed_min >= dark_check_interval_min:
            for hit in engine.check_dark_vessels(now=current_time):
                flagged_events.append({
                    "mmsi": hit["mmsi"],
                    "timestamp": hit["timestamp"],
                    "risk_score": hit["severity"],
                    "anomaly_types": [hit["type"]],
                    "channel": "n/a",
                })
            last_dark_check_time = current_time

        verdict = engine.process(record)
        if verdict["flagged"]:
            flagged_events.append({
                "mmsi": record["mmsi"],
                "timestamp": record["timestamp"],
                "risk_score": verdict["risk_score"],
                "anomaly_types": verdict["anomaly_types"],
                "channel": record.get("channel", "n/a"),
            })

    # Final sweep — catches a vessel that's still dark when the
    # dataset ends and never sent another message to trigger a reset.
    for hit in engine.check_dark_vessels(now=engine.latest_sim_time):
        flagged_events.append({
            "mmsi": hit["mmsi"],
            "timestamp": hit["timestamp"],
            "risk_score": hit["severity"],
            "anomaly_types": [hit["type"]],
            "channel": "n/a",
        })

    return flagged_events


def rough_vessel_level_score(flagged_events, labels):
    """The original coarse check: was this MMSI ever flagged at all."""
    flagged_mmsis = {e["mmsi"] for e in flagged_events}
    label_mmsis = set(labels["mmsi"])
    caught = flagged_mmsis & label_mmsis
    missed = label_mmsis - flagged_mmsis
    false_positives = flagged_mmsis - label_mmsis
    print(f"\n--- Rough vessel-level scoring (baseline, kept for comparison) ---")
    print(f"Injected anomalous vessels: {len(label_mmsis)}")
    print(f"Caught: {len(caught)} | Missed: {len(missed)} | Flagged but not labeled: {len(false_positives)}")


def event_level_score(flagged_events, labels, tolerance_min):
    """
    Matches on MMSI + anomaly type + time-window overlap, not just
    'was this MMSI flagged for anything, ever.' A label counts as
    caught if a flagged event exists for the same MMSI, with the
    engine-equivalent type, whose timestamp falls within
    [start_time - tolerance, end_time + tolerance].
    """
    labels = labels.copy()
    labels["start_time"] = pd.to_datetime(labels["start_time"])
    labels["end_time"] = pd.to_datetime(labels["end_time"])
    tol = pd.Timedelta(minutes=tolerance_min)

    caught_labels = []
    missed_labels = []

    for _, label in labels.iterrows():
        expected_engine_type = LABEL_TO_ENGINE_TYPE.get(label["anomaly_type"])
        window_start = label["start_time"] - tol
        window_end = label["end_time"] + tol

        match = any(
            e["mmsi"] == label["mmsi"]
            and expected_engine_type in e["anomaly_types"]
            and window_start <= e["timestamp"] <= window_end
            for e in flagged_events
        )
        if match:
            caught_labels.append(label)
        else:
            missed_labels.append(label)

    matched_label_keys = {
        (l["mmsi"], LABEL_TO_ENGINE_TYPE.get(l["anomaly_type"]))
        for l in caught_labels
    }
    unexplained_events = [
        e for e in flagged_events
        if not any((e["mmsi"], t) in matched_label_keys for t in e["anomaly_types"])
    ]

    print(f"\n--- Event-level scoring (MMSI + type + time window, tolerance +/-{tolerance_min} min) ---")
    print(f"Total labeled anomalies: {len(labels)}")
    print(f"Caught: {len(caught_labels)} | Missed: {len(missed_labels)}")
    print(f"Flagged events not matching any label: {len(unexplained_events)}")

    unexplained_by_type = {}
    for e in unexplained_events:
        for t in e["anomaly_types"]:
            unexplained_by_type[t] = unexplained_by_type.get(t, 0) + 1
    if unexplained_by_type:
        print(f"  Breakdown by type: {unexplained_by_type}")

    if missed_labels:
        print(f"\nMissed anomalies:")
        for l in missed_labels:
            print(f"  MMSI {l['mmsi']} - {l['anomaly_type']} ({l['start_time']} to {l['end_time']})")

    by_type = {}
    for l in labels.itertuples():
        by_type.setdefault(l.anomaly_type, [0, 0])[1] += 1
    for l in caught_labels:
        by_type[l["anomaly_type"]][0] += 1
    print(f"\nBy anomaly type (caught/total):")
    for t, (c, total) in by_type.items():
        print(f"  {t}: {c}/{total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--labels", default=None, help="Optional ground-truth CSV from Sudan's injection script")
    parser.add_argument("--dark-check-interval-min", type=float, default=5.0,
                         help="Run the dark-vessel sweep every N minutes of SIMULATED time (not record count)")
    parser.add_argument("--dark-threshold-min", type=float, default=15.0,
                         help="Minutes of silence before a vessel is flagged dark — tune this against real data")
    parser.add_argument("--match-tolerance-min", type=float, default=5.0,
                         help="Time-window tolerance (minutes) for event-level scoring")
    args = parser.parse_args()

    print(f"Loading {args.csv} ...")
    records = load_records(args.csv)
    print(f"Loaded {len(records)} records\n")

    flagged_events = run_engine(records, args.dark_check_interval_min, args.dark_threshold_min)

    print(f"Total flagged events: {len(flagged_events)}\n")
    for e in flagged_events[:20]:
        print(f"  MMSI {e['mmsi']} @ {e['timestamp']} [{e['channel']}] - {e['anomaly_types']} (score {e['risk_score']})")
    if len(flagged_events) > 20:
        print(f"  ... and {len(flagged_events) - 20} more")

    if args.labels:
        labels = pd.read_csv(args.labels)
        rough_vessel_level_score(flagged_events, labels)
        event_level_score(flagged_events, labels, args.match_tolerance_min)


if __name__ == "__main__":
    main()