"""Compare unsupervised detectors with leave-one-day-out validation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from .evaluation import annotate_ground_truth, load_labels
from .features import build_features, load_source_data
from .model import MODEL_FEATURES


def _make_model(name: str, random_state: int) -> Any:
    if name == "isolation_forest":
        return IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "local_outlier_factor":
        return LocalOutlierFactor(
            n_neighbors=35,
            novelty=True,
            contamination="auto",
            n_jobs=-1,
        )
    if name == "one_class_svm":
        return OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    if name == "sgd_one_class_svm":
        return SGDOneClassSVM(
            nu=0.05,
            random_state=random_state,
            max_iter=1000,
            tol=1e-3,
        )
    if name == "elliptic_envelope":
        return EllipticEnvelope(
            contamination=0.05,
            random_state=random_state,
        )
    if name == "minibatch_kmeans":
        return MiniBatchKMeans(
            n_clusters=16,
            batch_size=1024,
            n_init=3,
            max_iter=100,
            random_state=random_state,
        )
    raise ValueError(f"Unknown detector: {name}")


def _anomaly_scores(model: Any, matrix: np.ndarray, name: str) -> np.ndarray:
    if name == "local_outlier_factor":
        return -model.score_samples(matrix)
    if name == "elliptic_envelope":
        return -model.score_samples(matrix)
    if name == "minibatch_kmeans":
        return model.transform(matrix).min(axis=1)
    return -model.decision_function(matrix).reshape(-1)


def _fit_and_score_fold(
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    max_training_rows: int,
    random_state: int,
    alert_fraction: float,
) -> tuple[np.ndarray, float, int, float, float, dict[str, float | int]]:
    if len(train) > max_training_rows:
        train = train.sample(n=max_training_rows, random_state=random_state)

    train_values = train.loc[:, MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    test_values = test.loc[:, MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    train_values = train_values.replace([np.inf, -np.inf], np.nan)
    test_values = test_values.replace([np.inf, -np.inf], np.nan)
    medians = train_values.median().fillna(0.0)
    train_values = train_values.fillna(medians).fillna(0.0)
    test_values = test_values.fillna(medians).fillna(0.0)

    if name == "isolation_forest":
        train_matrix = train_values.to_numpy()
        test_matrix = test_values.to_numpy()
    else:
        scaler = RobustScaler()
        train_matrix = scaler.fit_transform(train_values)
        test_matrix = scaler.transform(test_values)

    model = _make_model(name, random_state)
    started = time.perf_counter()
    model.fit(train_matrix)
    fit_seconds = time.perf_counter() - started

    if name == "local_outlier_factor":
        train_scores = -model.negative_outlier_factor_
    else:
        train_scores = _anomaly_scores(model, train_matrix, name)
    threshold = float(np.quantile(train_scores, 1 - alert_fraction))
    scoring_started = time.perf_counter()
    test_scores = _anomaly_scores(model, test_matrix, name)
    scoring_seconds = time.perf_counter() - scoring_started
    latency_size = min(200, len(test_matrix))
    latency_indices = np.linspace(
        0, len(test_matrix) - 1, num=latency_size, dtype=int
    )
    per_record_latencies_ms = []
    for row_index in latency_indices:
        row_started = time.perf_counter()
        _anomaly_scores(model, test_matrix[row_index : row_index + 1], name)
        per_record_latencies_ms.append(
            (time.perf_counter() - row_started) * 1000
        )
    latency = {
        "sample_count": latency_size,
        "p50_ms": float(np.percentile(per_record_latencies_ms, 50)),
        "p95_ms": float(np.percentile(per_record_latencies_ms, 95)),
    }
    return test_scores, threshold, len(train), fit_seconds, scoring_seconds, latency


def _scenario_hits(
    scored: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    tolerance_min: float,
) -> dict[str, Any]:
    tolerance = pd.Timedelta(minutes=tolerance_min)
    matches: list[dict[str, Any]] = []
    for label in labels.itertuples(index=False):
        start = label.start_time - tolerance
        end = label.end_time + tolerance
        candidates = scored[
            (scored["MMSI"] == label.mmsi)
            & (scored["BaseDateTime"] >= start)
            & (scored["BaseDateTime"] <= end)
        ]
        caught = bool(candidates["model_flag"].any())
        matches.append(
            {
                "mmsi": int(label.mmsi),
                "anomaly_type": label.anomaly_type,
                "caught": caught,
            }
        )

    per_type: dict[str, dict[str, int]] = {}
    for anomaly_type in labels["anomaly_type"].unique():
        group = [item for item in matches if item["anomaly_type"] == anomaly_type]
        per_type[anomaly_type] = {
            "caught": sum(item["caught"] for item in group),
            "total": len(group),
        }

    return {
        "caught": sum(item["caught"] for item in matches),
        "total": len(matches),
        "by_type": per_type,
        "scenarios": matches,
    }


def compare_models(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    max_training_rows: int = 10000,
    alert_fraction: float = 0.05,
    tolerance_min: float = 5.0,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run leave-one-calendar-day-out scoring for each candidate detector."""
    features = features.copy()
    features["BaseDateTime"] = pd.to_datetime(
        features["BaseDateTime"], format="mixed", errors="raise", utc=True
    )
    features["validation_day"] = features["BaseDateTime"].dt.strftime("%Y-%m-%d")
    labels = labels.copy()
    labels["validation_day"] = labels["start_time"].dt.strftime("%Y-%m-%d")
    days = sorted(features["validation_day"].unique())
    ground_truth = annotate_ground_truth(features, labels)

    report: dict[str, Any] = {
        "validation": "leave-one-calendar-day-out",
        "days": days,
        "max_training_rows_per_fold": max_training_rows,
        "training_alert_fraction": alert_fraction,
        "scenario_match_tolerance_min": tolerance_min,
        "random_state": random_state,
        "training_sample_strategy": "same deterministic sample for all detectors within each fold",
        "detector_settings": {
            "isolation_forest": {"n_estimators": 200, "contamination": "auto"},
            "local_outlier_factor": {"n_neighbors": 35, "novelty": True, "contamination": "auto"},
            "one_class_svm": {"kernel": "rbf", "nu": 0.05, "gamma": "scale"},
            "sgd_one_class_svm": {"nu": 0.05, "max_iter": 1000, "tol": 0.001},
            "elliptic_envelope": {"contamination": 0.05},
            "minibatch_kmeans": {"n_clusters": 16, "batch_size": 1024, "n_init": 3},
        },
        "scoring_note": "Threshold calibrated from training-fold score quantile; test labels are not used for fitting or threshold selection.",
        "models": {},
    }

    for model_index, name in enumerate(
        (
            "isolation_forest",
            "local_outlier_factor",
            "one_class_svm",
            "sgd_one_class_svm",
            "elliptic_envelope",
            "minibatch_kmeans",
        )
    ):
        fold_frames: list[pd.DataFrame] = []
        fold_info: list[dict[str, Any]] = []
        total_fit_seconds = 0.0
        total_scoring_seconds = 0.0

        for fold_index, day in enumerate(days):
            train_mask = features["validation_day"] != day
            test_mask = features["validation_day"] == day
            train = features.loc[train_mask]
            test = features.loc[test_mask].copy()
            if len(train) > max_training_rows:
                train = train.sample(
                    n=max_training_rows,
                    random_state=random_state + fold_index,
                )
            (
                scores,
                threshold,
                fitted_rows,
                fit_seconds,
                scoring_seconds,
                latency,
            ) = _fit_and_score_fold(
                name,
                train,
                test,
                max_training_rows=max_training_rows,
                random_state=random_state + model_index * 100 + fold_index,
                alert_fraction=alert_fraction,
            )
            test["anomaly_score"] = scores
            test["model_flag"] = scores >= threshold
            fold_frames.append(test)
            total_fit_seconds += fit_seconds
            total_scoring_seconds += scoring_seconds
            fold_info.append(
                {
                    "held_out_day": day,
                    "test_rows": len(test),
                    "training_rows_used": fitted_rows,
                    "score_threshold": threshold,
                    "fit_seconds": round(fit_seconds, 3),
                    "score_seconds": round(scoring_seconds, 3),
                    "single_record_model_score_latency": latency,
                }
            )

        oof = pd.concat(fold_frames, ignore_index=True)
        oof = annotate_ground_truth(oof, labels)
        test_labels = labels[labels["validation_day"].isin(days)]
        scenario = _scenario_hits(oof, test_labels, tolerance_min=tolerance_min)
        unlabelled = ~oof["ground_truth_anomaly"]
        model_result = {
            "folds": fold_info,
            "out_of_fold_rows": len(oof),
            "flagged_rows": int(oof["model_flag"].sum()),
            "flag_rate": float(oof["model_flag"].mean()),
            "provisional_unlabelled_flag_rate": float(
                oof.loc[unlabelled, "model_flag"].mean()
            ),
            "scenario_detection": scenario,
            "total_fit_seconds": round(total_fit_seconds, 3),
            "total_batch_score_seconds": round(total_scoring_seconds, 3),
        }
        report["models"][name] = model_result

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-training-rows", type=int, default=10000)
    parser.add_argument("--alert-fraction", type=float, default=0.05)
    parser.add_argument("--match-tolerance-min", type=float, default=5.0)
    args = parser.parse_args()

    features = build_features(load_source_data(args.data_directory))
    labels = load_labels(args.labels)
    report = compare_models(
        features,
        labels,
        max_training_rows=args.max_training_rows,
        alert_fraction=args.alert_fraction,
        tolerance_min=args.match_tolerance_min,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
