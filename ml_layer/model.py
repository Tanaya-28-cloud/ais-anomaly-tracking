"""Initial unsupervised AIS anomaly detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest


MODEL_FEATURES = (
    "time_since_last_transmission_seconds",
    "distance_from_last_km",
    "implied_speed_knots",
    "sog_delta",
    "cog_delta",
    "speed_ratio_to_reported_sog",
    "non_positive_time_delta",
)


@dataclass
class IsolationForestDetector:
    """Fitted detector and the medians used to make scoring deterministic."""

    model: IsolationForest
    feature_medians: pd.Series

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        *,
        contamination: float | str = "auto",
        random_state: int = 42,
    ) -> "IsolationForestDetector":
        matrix = _prepare_matrix(features)
        model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(matrix)
        medians = matrix.median()
        return cls(model=model, feature_medians=medians)

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return input records with anomaly scores and binary flags."""
        matrix = _prepare_matrix(features, self.feature_medians)
        scored = features.copy()
        scored["anomaly_score"] = -self.model.decision_function(matrix)
        scored["ml_anomaly_flag"] = self.model.predict(matrix) == -1
        return scored

    def save(self, path: Path) -> None:
        """Persist the fitted detector for later batch or streaming scoring."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "IsolationForestDetector":
        """Load a detector previously saved with :meth:`save`."""
        detector = joblib.load(path)
        if not isinstance(detector, cls):
            raise TypeError(f"Expected {cls.__name__} artifact, got {type(detector).__name__}")
        return detector


def _prepare_matrix(
    features: pd.DataFrame,
    medians: pd.Series | None = None,
) -> pd.DataFrame:
    missing = sorted(set(MODEL_FEATURES) - set(features.columns))
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    matrix = features.loc[:, MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([float("inf"), float("-inf")], pd.NA)
    if medians is None:
        medians = matrix.median()
    return matrix.fillna(medians).fillna(0.0)
