"""Train and persist the initial AIS anomaly detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from .features import build_features, load_source_data
from .model import IsolationForestDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--contamination", default="auto")
    args = parser.parse_args()

    contamination: float | str = args.contamination
    if contamination != "auto":
        contamination = float(contamination)
    features = build_features(load_source_data(args.data_directory))
    detector = IsolationForestDetector.fit(
        features, contamination=contamination
    )
    detector.save(args.artifact)
    print(f"saved detector artifact: {args.artifact}")
    print(f"training records: {len(features)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
