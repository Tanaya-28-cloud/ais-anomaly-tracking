"""Initial unsupervised AIS anomaly detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM


MODEL_FEATURES = (
    "time_since_last_transmission_seconds",
    "distance_from_last_km",
    "implied_speed_knots",
    "sog_delta",
    "cog_delta",
    "speed_ratio_to_reported_sog",
    "non_positive_time_delta",
)
FEATURE_LABELS = {
    "time_since_last_transmission_seconds": "Time since previous AIS report (seconds)",
    "distance_from_last_km": "Distance from previous position (km)",
    "implied_speed_knots": "Speed implied by positions (knots)",
    "sog_delta": "Change in reported speed (knots)",
    "cog_delta": "Change in course (degrees)",
    "speed_ratio_to_reported_sog": "Implied speed / previous reported speed",
    "non_positive_time_delta": "Timestamp is duplicate or out of order",
}


class ScoringDetector(Protocol):
    """Common interface used by batch and streaming scorers."""

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return the feature rows with anomaly score and flag columns."""


@dataclass
class RBFOneClassSVMDetector:
    """RBF One-Class SVM with persisted preprocessing and score threshold."""

    model: OneClassSVM
    feature_medians: pd.Series
    scaler: RobustScaler
    score_threshold: float
    training_rows: int
    alert_fraction: float
    explain_background: np.ndarray | None = None

    @property
    def model_name(self) -> str:
        return "rbf_one_class_svm"

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        *,
        max_training_rows: int = 10_000,
        alert_fraction: float = 0.05,
        random_state: int = 42,
    ) -> "RBFOneClassSVMDetector":
        if not 0 < alert_fraction < 0.5:
            raise ValueError("alert_fraction must be greater than 0 and less than 0.5")
        if max_training_rows < 2:
            raise ValueError("max_training_rows must be at least 2")

        matrix = features.loc[:, MODEL_FEATURES].apply(
            pd.to_numeric, errors="coerce"
        )
        matrix = matrix.replace([float("inf"), float("-inf")], pd.NA)
        if len(matrix) > max_training_rows:
            matrix = matrix.sample(n=max_training_rows, random_state=random_state)
        medians = matrix.median().fillna(0.0)
        matrix = matrix.fillna(medians).fillna(0.0)
        scaler = RobustScaler()
        scaled_matrix = scaler.fit_transform(matrix)
        model = OneClassSVM(kernel="rbf", nu=alert_fraction, gamma="scale")
        model.fit(scaled_matrix)
        training_scores = -model.decision_function(scaled_matrix).reshape(-1)
        threshold = float(np.quantile(training_scores, 1 - alert_fraction))
        background_count = min(20, len(matrix))
        explain_background = matrix.sample(
            n=background_count, random_state=random_state + 1
        ).to_numpy(dtype=float)
        return cls(
            model=model,
            feature_medians=medians,
            scaler=scaler,
            score_threshold=threshold,
            training_rows=len(matrix),
            alert_fraction=alert_fraction,
            explain_background=explain_background,
        )

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """Score records using the persisted training transform and cutoff."""
        matrix = _prepare_matrix(features, self.feature_medians)
        scaled_matrix = self.scaler.transform(matrix)
        scores = -self.model.decision_function(scaled_matrix).reshape(-1)
        scored = features.copy()
        scored["anomaly_score"] = scores
        scored["ml_anomaly_flag"] = scores >= self.score_threshold
        return scored

    def explain(self, features: pd.DataFrame, *, permutations: int = 8) -> dict[str, object]:
        """Explain one score with model-agnostic SHAP permutation values."""
        import shap

        row = _prepare_matrix(features, self.feature_medians).iloc[[0]]
        background = self.explain_background
        if background is None or len(background) == 0:
            raise RuntimeError(
                "This artifact has no SHAP background data. Retrain the RBF artifact."
            )

        if not hasattr(self, "_shap_explainer"):
            self._shap_explainer = shap.PermutationExplainer(
                self._score_raw_rows,
                shap.maskers.Independent(background, max_samples=len(background)),
                feature_names=list(MODEL_FEATURES),
                seed=42,
            )

        feature_count = len(MODEL_FEATURES)
        explanation = self._shap_explainer(
            row.to_numpy(dtype=float),
            max_evals=(2 * feature_count + 1) * permutations,
            silent=True,
        )
        values = explanation.values[0]
        base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
        score = float(self.score(row)["anomaly_score"].iloc[0])
        shap_sum = base_value + float(np.sum(values))
        return {
            "method": "SHAP PermutationExplainer",
            "output": "anomaly_score",
            "base_value": base_value,
            "score": score,
            "score_threshold": float(self.score_threshold),
            "additivity_residual": score - shap_sum,
            "permutations": permutations,
            "features": [
                {
                    "name": name,
                    "label": FEATURE_LABELS[name],
                    "value": float(row.iloc[0][name]),
                    "shap_value": float(value),
                    "effect": "increases anomaly score" if value > 0 else "decreases anomaly score",
                }
                for name, value in zip(MODEL_FEATURES, values)
            ],
        }

    def _score_raw_rows(self, values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=MODEL_FEATURES)
        prepared = _prepare_matrix(frame, self.feature_medians)
        scaled = self.scaler.transform(prepared)
        return -self.model.decision_function(scaled).reshape(-1)

    def save(self, path: Path) -> None:
        """Persist the fitted SVM and its required preprocessing state."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "RBFOneClassSVMDetector":
        """Load a saved RBF detector artifact."""
        detector = joblib.load(path)
        if not isinstance(detector, cls):
            raise TypeError(f"Expected {cls.__name__} artifact, got {type(detector).__name__}")
        return detector


def load_detector(path: Path) -> ScoringDetector:
    """Load the selected RBF One-Class SVM detector artifact."""
    detector = joblib.load(path)
    if isinstance(detector, RBFOneClassSVMDetector):
        return detector
    raise TypeError(f"Unsupported detector artifact: {type(detector).__name__}")


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
