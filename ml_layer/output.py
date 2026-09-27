"""Stable anomaly result format for logging, UI, and MQTT consumers."""

from __future__ import annotations

from typing import Any


OUTPUT_SCHEMA_VERSION = "1.0"
FEATURE_VERSION = "trajectory-v1"


def build_anomaly_output(
    record: dict[str, Any],
    *,
    anomaly_score: float,
    anomaly_flag: bool,
    feature_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the original record plus standardized ML result fields."""
    output = dict(record)
    output.update(
        {
            "mmsi": record.get("MMSI", record.get("mmsi")),
            "timestamp": record.get("BaseDateTime", record.get("timestamp")),
            "channel": record.get("channel"),
            "anomaly_score": float(anomaly_score),
            "anomaly_flag": bool(anomaly_flag),
            "anomaly_source": "isolation_forest",
            "ml_output_schema": OUTPUT_SCHEMA_VERSION,
            "feature_version": FEATURE_VERSION,
        }
    )
    if feature_values is not None:
        output["ml_features"] = {
            key: _json_value(value) for key, value in feature_values.items()
        }
    return {key: _json_value(value) for key, value in output.items()}


def _json_value(value: Any) -> Any:
    """Convert pandas/numpy scalars and missing values to JSON-safe values."""
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if value != value:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
