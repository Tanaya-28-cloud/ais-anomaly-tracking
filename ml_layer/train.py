"""Train and persist the RBF One-Class SVM AIS detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from .features import build_features, load_source_data
from .model import RBFOneClassSVMDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--max-training-rows", type=int, default=10_000)
    parser.add_argument("--alert-fraction", type=float, default=0.05)
    args = parser.parse_args()

    features = build_features(load_source_data(args.data_directory))
    detector = RBFOneClassSVMDetector.fit(
        features,
        max_training_rows=args.max_training_rows,
        alert_fraction=args.alert_fraction,
    )
    detector.save(args.artifact)
    print(f"model type: {detector.model_name}")
    print(f"saved detector artifact: {args.artifact}")
    print(f"source records: {len(features)}")
    print(f"model training rows: {detector.training_rows}")
    print(f"training alert fraction: {detector.alert_fraction}")
    print(f"training-derived score threshold: {detector.score_threshold:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
