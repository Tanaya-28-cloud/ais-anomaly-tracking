"""Feature extraction for per-vessel AIS trajectory records."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_inspection import REQUIRED_COLUMNS, find_csv_files, inspect_dataframe


EARTH_RADIUS_KM = 6371.0088
KILOMETERS_PER_NAUTICAL_MILE = 1.852


def load_source_data(data_directory: Path) -> pd.DataFrame:
    """Load terrestrial and satellite AIS records into one ordered dataframe."""
    paths = find_csv_files(data_directory)
    if not paths:
        raise FileNotFoundError(f"No AIS source CSV files found under {data_directory}")

    frames = [pd.read_csv(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    report = inspect_dataframe(data)
    if not report.is_ready:
        raise ValueError(
            "AIS source data is not ready: "
            f"missing={report.missing_columns}, invalid={report.invalid_counts}"
        )

    data["MMSI"] = pd.to_numeric(data["MMSI"], errors="raise").astype("int64")
    data["BaseDateTime"] = pd.to_datetime(
        data["BaseDateTime"], errors="raise", format="mixed", utc=True
    )
    for column in ("LAT", "LON", "SOG", "COG"):
        data[column] = pd.to_numeric(data[column], errors="raise")

    return data.sort_values(["MMSI", "BaseDateTime", "channel"]).reset_index(drop=True)


def _haversine_km(
    latitude: pd.Series,
    longitude: pd.Series,
    previous_latitude: pd.Series,
    previous_longitude: pd.Series,
) -> pd.Series:
    """Calculate great-circle distance between consecutive positions."""
    latitude_radians = np.radians(latitude)
    previous_latitude_radians = np.radians(previous_latitude)
    delta_latitude = previous_latitude_radians - latitude_radians
    delta_longitude = np.radians(previous_longitude - longitude)
    haversine = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(latitude_radians)
        * np.cos(previous_latitude_radians)
        * np.sin(delta_longitude / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(haversine.clip(0, 1)))


def _circular_difference(current: pd.Series, previous: pd.Series) -> pd.Series:
    """Return the smallest absolute difference between two headings."""
    return ((current - previous + 180) % 360 - 180).abs()


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add trajectory features while preserving the original AIS columns."""
    required = set(REQUIRED_COLUMNS) | {"channel"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns required for feature extraction: {missing}")

    features = data.copy()
    features["BaseDateTime"] = pd.to_datetime(
        features["BaseDateTime"], errors="raise", format="mixed", utc=True
    )
    features = features.sort_values(
        ["MMSI", "BaseDateTime", "channel"]
    ).reset_index(drop=True)
    grouped = features.groupby("MMSI", sort=False)

    features["previous_timestamp"] = grouped["BaseDateTime"].shift()
    features["previous_latitude"] = grouped["LAT"].shift()
    features["previous_longitude"] = grouped["LON"].shift()
    features["previous_sog"] = grouped["SOG"].shift()
    features["previous_cog"] = grouped["COG"].shift()
    features["time_since_last_transmission_seconds"] = (
        features["BaseDateTime"] - features["previous_timestamp"]
    ).dt.total_seconds()
    features["non_positive_time_delta"] = (
        features["time_since_last_transmission_seconds"] <= 0
    ).astype("int8")
    features["distance_from_last_km"] = _haversine_km(
        features["LAT"],
        features["LON"],
        features["previous_latitude"],
        features["previous_longitude"],
    )
    valid_time_delta = features["time_since_last_transmission_seconds"] > 0
    features["implied_speed_knots"] = np.where(
        valid_time_delta,
        features["distance_from_last_km"]
        / features["time_since_last_transmission_seconds"]
        * 3600
        / KILOMETERS_PER_NAUTICAL_MILE,
        np.nan,
    )
    features["sog_delta"] = (features["SOG"] - features["previous_sog"]).abs()
    features["cog_delta"] = _circular_difference(
        features["COG"], features["previous_cog"]
    )
    features["speed_ratio_to_reported_sog"] = features["implied_speed_knots"] / (
        features["previous_sog"].clip(lower=1)
    )

    return features
