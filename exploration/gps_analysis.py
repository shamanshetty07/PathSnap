"""GPS trajectory analysis for PathSnap EDA.

Computes per-trip statistics for the ``.track`` GPS observations:

* observation counts
* timestamp interval distribution and 1-second regularity
* gaps greater than 1, 5 and 10 seconds
* duplicate coordinates
* invalid longitude/latitude values
* non-increasing timestamps
* Haversine distances and speeds between consecutive observations
* suspected outliers / unusually high speeds

The original dataset is never modified. Malformed records are counted and
flagged, not silently removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .data_loader import EXTENSIONS, _read_table, _resolve_file

# Speed threshold (m/s) above which a GPS segment is flagged as suspicious.
# 30 m/s ~= 108 km/h, a conservative upper bound for most road vehicles.
SPEED_OUTLIER_THRESHOLD_MPS = 30.0

EARTH_RADIUS_M = 6371000.0


@dataclass
class GpsRawArrays:
    """Raw per-observation arrays collected for pooled plotting."""

    timestamps: np.ndarray
    intervals: np.ndarray
    distances_m: np.ndarray
    speeds_mps: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray


@dataclass
class GpsStats:
    """Result of analysing a single trip's GPS track."""

    trip_id: str
    split: str
    num_observations: int
    num_raw_rows: int
    invalid_coordinate_rows: int
    invalid_longitude: int
    invalid_latitude: int
    duplicate_coordinates: int
    non_increasing_timestamps: int
    pct_one_second_apart: float
    mean_interval_s: float
    median_interval_s: float
    min_interval_s: float
    max_interval_s: float
    gaps_gt_1s: int
    gaps_gt_5s: int
    gaps_gt_10s: int
    num_consecutive_distances: int
    mean_distance_m: float
    median_distance_m: float
    max_distance_m: float
    num_speed_estimates: int
    mean_speed_mps: float
    median_speed_mps: float
    max_speed_mps: float
    num_high_speed_segments: int
    high_speed_threshold_mps: float
    suspicious_observations: int
    load_warnings: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "split": self.split,
            "trip_id": self.trip_id,
            "num_observations": self.num_observations,
            "num_raw_rows": self.num_raw_rows,
            "invalid_coordinate_rows": self.invalid_coordinate_rows,
            "invalid_longitude": self.invalid_longitude,
            "invalid_latitude": self.invalid_latitude,
            "duplicate_coordinates": self.duplicate_coordinates,
            "non_increasing_timestamps": self.non_increasing_timestamps,
            "pct_one_second_apart": self.pct_one_second_apart,
            "mean_interval_s": self.mean_interval_s,
            "median_interval_s": self.median_interval_s,
            "min_interval_s": self.min_interval_s,
            "max_interval_s": self.max_interval_s,
            "gaps_gt_1s": self.gaps_gt_1s,
            "gaps_gt_5s": self.gaps_gt_5s,
            "gaps_gt_10s": self.gaps_gt_10s,
            "num_consecutive_distances": self.num_consecutive_distances,
            "mean_distance_m": self.mean_distance_m,
            "median_distance_m": self.median_distance_m,
            "max_distance_m": self.max_distance_m,
            "num_speed_estimates": self.num_speed_estimates,
            "mean_speed_mps": self.mean_speed_mps,
            "median_speed_mps": self.median_speed_mps,
            "max_speed_mps": self.max_speed_mps,
            "num_high_speed_segments": self.num_high_speed_segments,
            "high_speed_threshold_mps": self.high_speed_threshold_mps,
            "suspicious_observations": self.suspicious_observations,
        }


def haversine_distance_m(lat1, lon1, lat2, lon2):
    """Haversine distance in metres between (lat, lon) pairs.

    Inputs may be scalars or NumPy arrays of equal shape.
    """
    lat1 = np.radians(np.asarray(lat1, dtype=np.float64))
    lon1 = np.radians(np.asarray(lon1, dtype=np.float64))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float64))
    lon2 = np.radians(np.asarray(lon2, dtype=np.float64))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def _read_raw_track(path: Path) -> pd.DataFrame:
    """Read the raw .track file without dropping any rows.

    Returns a DataFrame with numeric ``longitude``/``latitude``/``timestamp``
    columns (non-numeric values become NaN). No rows are removed.
    """
    frame = _read_table(path, ["longitude", "latitude", "timestamp"], 3)
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    return frame


def _valid_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["longitude"].notna() & frame["latitude"].notna() & frame["timestamp"].notna()


def analyze_gps(trip_data, data_dir: Path | str) -> GpsStats:
    """Analyse the GPS track of a single :class:`TripData`.

    ``trip_data.track`` holds the cleaned observations; the raw file is re-read
    solely to count malformed / invalid records that the loader discarded.
    """
    data_dir = Path(data_dir)
    track_path = _resolve_file(data_dir, trip_data.trip_id, "track")
    raw_frame = _read_raw_track(track_path) if track_path is not None else pd.DataFrame(
        columns=["longitude", "latitude", "timestamp"]
    )

    num_raw_rows = len(raw_frame)
    valid_mask = _valid_mask(raw_frame)
    invalid_coordinate_rows = int((~valid_mask).sum())
    invalid_longitude = int(raw_frame["longitude"].isna().sum())
    invalid_latitude = int(raw_frame["latitude"].isna().sum())

    clean = trip_data.track.reset_index(drop=True)
    num_obs = int(len(clean))

    # Duplicate coordinates among valid observations.
    if num_obs > 0:
        dup = clean.duplicated(subset=["longitude", "latitude"], keep=False)
        duplicate_coordinates = int(dup.sum())
    else:
        duplicate_coordinates = 0

    # Timestamp analysis.
    if num_obs >= 2:
        timestamps = clean["timestamp"].to_numpy(dtype=np.float64)
        intervals = np.diff(timestamps)
        non_increasing = int((intervals <= 0).sum())
        pct_one_second = float((np.isclose(intervals, 1.0)).mean() * 100.0)
        mean_interval = float(np.mean(intervals))
        median_interval = float(np.median(intervals))
        min_interval = float(np.min(intervals))
        max_interval = float(np.max(intervals))
        gaps_gt_1 = int((intervals > 1.0).sum())
        gaps_gt_5 = int((intervals > 5.0).sum())
        gaps_gt_10 = int((intervals > 10.0).sum())
    else:
        intervals = np.array([], dtype=np.float64)
        non_increasing = 0
        pct_one_second = 0.0 if num_obs == 0 else 100.0
        mean_interval = median_interval = min_interval = max_interval = 0.0
        gaps_gt_1 = gaps_gt_5 = gaps_gt_10 = 0

    # Haversine distances and speeds between consecutive valid observations.
    if num_obs >= 2:
        lat = clean["latitude"].to_numpy(dtype=np.float64)
        lon = clean["longitude"].to_numpy(dtype=np.float64)
        distances = haversine_distance_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
        valid_intervals = intervals[intervals > 0]
        speeds = distances[intervals > 0] / intervals[intervals > 0]

        num_high_speed = int((speeds > SPEED_OUTLIER_THRESHOLD_MPS).sum())
        # A suspicious observation is the endpoint of a high-speed segment or a
        # non-increasing timestamp.
        suspicious = num_high_speed + non_increasing
    else:
        distances = np.array([], dtype=np.float64)
        speeds = np.array([], dtype=np.float64)
        num_high_speed = 0
        suspicious = non_increasing

    warnings = list(trip_data.load_warnings)
    if invalid_coordinate_rows:
        warnings.append(
            f"{track_path.name}: {invalid_coordinate_rows} row(s) with invalid "
            "coordinates were excluded"
        )

    stats = GpsStats(
        trip_id=trip_data.trip_id,
        split=trip_data.split,
        num_observations=num_obs,
        num_raw_rows=num_raw_rows,
        invalid_coordinate_rows=invalid_coordinate_rows,
        invalid_longitude=invalid_longitude,
        invalid_latitude=invalid_latitude,
        duplicate_coordinates=duplicate_coordinates,
        non_increasing_timestamps=non_increasing,
        pct_one_second_apart=pct_one_second,
        mean_interval_s=mean_interval,
        median_interval_s=median_interval,
        min_interval_s=min_interval,
        max_interval_s=max_interval,
        gaps_gt_1s=gaps_gt_1,
        gaps_gt_5s=gaps_gt_5,
        gaps_gt_10s=gaps_gt_10,
        num_consecutive_distances=int(len(distances)),
        mean_distance_m=float(np.mean(distances)) if len(distances) else 0.0,
        median_distance_m=float(np.median(distances)) if len(distances) else 0.0,
        max_distance_m=float(np.max(distances)) if len(distances) else 0.0,
        num_speed_estimates=int(len(speeds)),
        mean_speed_mps=float(np.mean(speeds)) if len(speeds) else 0.0,
        median_speed_mps=float(np.median(speeds)) if len(speeds) else 0.0,
        max_speed_mps=float(np.max(speeds)) if len(speeds) else 0.0,
        num_high_speed_segments=num_high_speed,
        high_speed_threshold_mps=SPEED_OUTLIER_THRESHOLD_MPS,
        suspicious_observations=int(suspicious),
        load_warnings=warnings,
    )

    raw = GpsRawArrays(
        timestamps=clean["timestamp"].to_numpy(dtype=np.float64),
        intervals=intervals,
        distances_m=distances,
        speeds_mps=speeds,
        latitude=clean["latitude"].to_numpy(dtype=np.float64),
        longitude=clean["longitude"].to_numpy(dtype=np.float64),
    )
    return stats, raw
