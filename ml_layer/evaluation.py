"""Evaluate ML flags against injected anomaly windows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from receiver.rules import RuleEngine


LABEL_TO_RULE_TYPE = {
    "dark_period": "dark_event",
    "type_mismatch": "identity_mismatch",
    "mmsi_duplication": "mmsi_duplication",
    "position_jump": "position_jump",
}

def load_labels(path: Path) -> pd.DataFrame:
    """Load and normalize injected anomaly windows."""
    labels = pd.read_csv(path)
    required = {"mmsi", "anomaly_type", "start_time", "end_time"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Missing label columns: {missing}")
    labels["mmsi"] = pd.to_numeric(labels["mmsi"], errors="raise").astype("int64")
    labels["start_time"] = pd.to_datetime(
        labels["start_time"], errors="raise", format="mixed", utc=True
    )
    labels["end_time"] = pd.to_datetime(
        labels["end_time"], errors="raise", format="mixed", utc=True
    )
    return labels


def annotate_ground_truth(
    scored: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Mark rows inside anomaly windows and rows immediately after dark gaps."""
    annotated = scored.copy()
    timestamps = pd.to_datetime(
        annotated["BaseDateTime"], errors="raise", format="mixed", utc=True
    )
    annotated["ground_truth_anomaly"] = False
    annotated["ground_truth_types"] = ""

    for label in labels.itertuples(index=False):
        mask = (
            (annotated["MMSI"] == label.mmsi)
            & (timestamps >= label.start_time)
            & (timestamps <= label.end_time)
        )
        if label.anomaly_type == "dark_period":
            mask |= (
                (annotated["MMSI"] == label.mmsi)
                & (annotated["previous_timestamp"] < label.start_time)
                & (timestamps >= label.end_time)
            )
        annotated.loc[mask, "ground_truth_anomaly"] = True
        existing = annotated.loc[mask, "ground_truth_types"]
        annotated.loc[mask, "ground_truth_types"] = existing.where(
            existing.eq(""), existing + ";"
        ) + label.anomaly_type
    return annotated


def build_evaluation_report(
    scored: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    dark_period_threshold_min: float = 10.0,
    dark_check_interval_min: float = 5.0,
    match_tolerance_min: float = 5.0,
) -> dict[str, object]:
    """Replay rules and compare both detectors to injected anomaly labels."""
    annotated = annotate_ground_truth(scored, labels)
    rule_events, rule_record_flags = replay_rule_engine(
        annotated,
        dark_period_threshold_min=dark_period_threshold_min,
        dark_check_interval_min=dark_check_interval_min,
    )
    annotated["rule_engine_flag"] = rule_record_flags

    ml_rule_agreement = annotated["ml_anomaly_flag"] == annotated["rule_engine_flag"]
    report: dict[str, object] = {
        "records": int(len(annotated)),
        "model_flagged_records": int(annotated["ml_anomaly_flag"].sum()),
        "ground_truth_records": int(annotated["ground_truth_anomaly"].sum()),
        "rule_engine": {
            "dark_period_threshold_min": dark_period_threshold_min,
            "dark_check_interval_min": dark_check_interval_min,
            "event_count": len(rule_events),
            "per_message_flag_count": int(rule_record_flags.sum()),
            "periodic_dark_sweep_event_count": sum(
                event.get("detection_source") == "periodic_dark_sweep"
                for event in rule_events
            ),
        },
        "model_rule_agreement": {
            "records_agree": int(ml_rule_agreement.sum()),
            "agreement_rate": float(ml_rule_agreement.mean()),
            "both_flagged": int(
                (annotated["ml_anomaly_flag"] & annotated["rule_engine_flag"]).sum()
            ),
            "ml_only": int(
                (annotated["ml_anomaly_flag"] & ~annotated["rule_engine_flag"]).sum()
            ),
            "rule_only": int(
                (~annotated["ml_anomaly_flag"] & annotated["rule_engine_flag"]).sum()
            ),
            "neither_flagged": int(
                (~annotated["ml_anomaly_flag"] & ~annotated["rule_engine_flag"]).sum()
            ),
        },
        "label_matching_tolerance_min": match_tolerance_min,
        "by_anomaly_type": {},
    }
    unlabelled = ~annotated["ground_truth_anomaly"]
    report["false_positive_rate"] = float(
        annotated.loc[unlabelled, "ml_anomaly_flag"].mean()
    )

    by_type: dict[str, object] = {}
    for anomaly_type in labels["anomaly_type"].unique():
        type_labels = labels[labels["anomaly_type"] == anomaly_type]
        scenario_matches = _label_scenario_matches(
            annotated, type_labels, rule_events, match_tolerance_min
        )
        rows = annotated[annotated["ground_truth_types"].str.contains(anomaly_type)]
        by_type[anomaly_type] = {
            "label_scenarios": int(len(type_labels)),
            "matched_records": int(len(rows)),
            "flagged_records": int(rows["ml_anomaly_flag"].sum()),
            "detection_rate": float(rows["ml_anomaly_flag"].mean())
            if len(rows)
            else None,
            "ml_caught_label_scenarios": sum(
                ml_hit for ml_hit, _ in scenario_matches
            ),
            "rule_caught_label_scenarios": sum(
                rule_hit for _, rule_hit in scenario_matches
            ),
            "both_caught_label_scenarios": sum(
                ml_hit and rule_hit for ml_hit, rule_hit in scenario_matches
            ),
            "ml_only_label_scenarios": sum(
                ml_hit and not rule_hit for ml_hit, rule_hit in scenario_matches
            ),
            "rule_only_label_scenarios": sum(
                rule_hit and not ml_hit for ml_hit, rule_hit in scenario_matches
            ),
            "neither_caught_label_scenarios": sum(
                not ml_hit and not rule_hit for ml_hit, rule_hit in scenario_matches
            ),
        }
    report["by_anomaly_type"] = by_type
    return report


def replay_rule_engine(
    records: pd.DataFrame,
    *,
    dark_period_threshold_min: float,
    dark_check_interval_min: float,
) -> tuple[list[dict[str, object]], pd.Series]:
    """Replay the production RuleEngine in timestamp order, matching its harness."""
    ordered = records.sort_values(["BaseDateTime", "MMSI", "channel"])
    engine = RuleEngine(dark_period_threshold_min=dark_period_threshold_min)
    events: list[dict[str, object]] = []
    process_flags: list[bool] = []
    last_dark_check_time = None

    for row in ordered.itertuples(index=False):
        timestamp = row.BaseDateTime
        if last_dark_check_time is None:
            last_dark_check_time = timestamp
        elapsed_min = (timestamp - last_dark_check_time).total_seconds() / 60
        if elapsed_min >= dark_check_interval_min:
            events.extend(
                _event_from_sweep(hit, "periodic_dark_sweep")
                for hit in engine.check_dark_vessels(now=timestamp)
            )
            last_dark_check_time = timestamp

        record = {
            "mmsi": int(row.MMSI),
            "lat": float(row.LAT),
            "lon": float(row.LON),
            "sog": _optional_float(row.SOG),
            "cog": _optional_float(row.COG),
            "heading": _optional_float(getattr(row, "Heading", None)),
            "vessel_type": _optional_float(getattr(row, "VesselType", None)),
            "timestamp": timestamp,
            "channel": row.channel,
        }
        verdict = engine.process(record)
        process_flags.append(bool(verdict["flagged"]))
        if verdict["flagged"]:
            events.append(
                {
                    "mmsi": record["mmsi"],
                    "timestamp": timestamp,
                    "anomaly_types": verdict["anomaly_types"],
                    "channel": record["channel"],
                    "detection_source": "message",
                }
            )

    events.extend(
        _event_from_sweep(hit, "final_dark_sweep")
        for hit in engine.check_dark_vessels(now=engine.latest_sim_time)
    )

    flags = pd.Series(False, index=records.index, dtype=bool)
    flags.loc[ordered.index] = process_flags
    return events, flags


def _event_from_sweep(
    hit: dict[str, object], detection_source: str
) -> dict[str, object]:
    return {
        "mmsi": hit["mmsi"],
        "timestamp": hit["timestamp"],
        "anomaly_types": [hit["type"]],
        "channel": "n/a",
        "detection_source": detection_source,
    }


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _label_scenario_matches(
    records: pd.DataFrame,
    labels: pd.DataFrame,
    events: list[dict[str, object]],
    tolerance_min: float,
) -> list[tuple[bool, bool]]:
    tolerance = pd.Timedelta(minutes=tolerance_min)
    matches: list[tuple[bool, bool]] = []
    for label in labels.itertuples(index=False):
        start = label.start_time - tolerance
        end = label.end_time + tolerance
        ml_hit = bool(
            (
                (records["MMSI"] == label.mmsi)
                & (records["BaseDateTime"] >= start)
                & (records["BaseDateTime"] <= end)
                & records["ml_anomaly_flag"]
            ).any()
        )
        rule_type = LABEL_TO_RULE_TYPE.get(label.anomaly_type)
        rule_hit = any(
            event["mmsi"] == label.mmsi
            and rule_type in event["anomaly_types"]
            and start <= event["timestamp"] <= end
            for event in events
        )
        matches.append((ml_hit, rule_hit))
    return matches
