"""Evaluate a saved AIS detector against injected anomaly labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import build_evaluation_report, load_labels
from .features import build_features, load_source_data
from .model import load_detector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dark-period-threshold-min", type=float, default=10.0)
    parser.add_argument("--dark-check-interval-min", type=float, default=5.0)
    parser.add_argument("--match-tolerance-min", type=float, default=5.0)
    args = parser.parse_args()

    features = build_features(load_source_data(args.data_directory))
    detector = load_detector(args.artifact)
    scored = detector.score(features)
    report = build_evaluation_report(
        scored,
        load_labels(args.labels),
        dark_period_threshold_min=args.dark_period_threshold_min,
        dark_check_interval_min=args.dark_check_interval_min,
        match_tolerance_min=args.match_tolerance_min,
    )
    report["detector"] = {
        "name": detector.model_name,
        "class": type(detector).__name__,
        "training_rows": detector.training_rows,
        "alert_fraction": detector.alert_fraction,
        "nu": detector.model.nu,
        "kernel": detector.model.kernel,
        "gamma": detector.model.gamma,
        "score_threshold": detector.score_threshold,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
