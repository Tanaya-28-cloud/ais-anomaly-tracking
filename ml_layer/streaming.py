"""Stateful per-record feature extraction and ML scoring."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .features import build_features
from .model import MODEL_FEATURES, ScoringDetector
from .output import build_anomaly_output


class StreamingScorer:
    """Score MQTT records using the previous observation for each MMSI."""

    def __init__(self, detector: ScoringDetector) -> None:
        self.detector = detector
        self._previous: dict[int, dict[str, Any]] = {}

    def score_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Score one record and update state only after successful scoring."""
        mmsi = int(record["MMSI"])
        current = dict(record)
        current["_stream_current"] = True
        previous = self._previous.get(mmsi)
        if previous is not None:
            previous = dict(previous)
            previous["_stream_current"] = False
            rows = [previous, current]
        else:
            rows = [current]
        feature_rows = build_features(pd.DataFrame(rows))
        current_mask = feature_rows["_stream_current"].astype(bool)
        current_features = feature_rows[current_mask].copy()
        scored = self.detector.score(current_features)
        row = scored.iloc[0]
        self._previous[mmsi] = dict(record)
        feature_values = {
            name: row[name] for name in MODEL_FEATURES if name in row
        }
        explanation = None
        if bool(row["ml_anomaly_flag"]):
            explain = getattr(self.detector, "explain", None)
            if callable(explain):
                try:
                    explanation = explain(current_features)
                except Exception as exc:
                    explanation = {
                        "status": "error",
                        "method": "SHAP PermutationExplainer",
                        "message": f"Could not explain this score: {exc}",
                    }
        return build_anomaly_output(
            record,
            anomaly_score=row["anomaly_score"],
            anomaly_flag=row["ml_anomaly_flag"],
            score_threshold=self.detector.score_threshold,
            anomaly_source=self.detector.model_name,
            feature_values=feature_values,
            shap_explanation=explanation,
        )
