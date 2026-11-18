"""Agent 2: Temporal and Movement Feature Engineering.

Features derived from GPS movement over time for every observation-candidate pair.

Handles:
- First observation separately (no previous observation).
- Missing observations.
- Zero time differences.
- Large timestamp gaps.
- Misleading acceleration across large GPS gaps.
- Circular angle differences for bearings.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from exploration.candidate_generation.geometry import (
    gps_bearing_degrees,
    haversine_distance_m,
)


def _circular_bearing_diff(b1: float, b2: float) -> float:
    """Circular difference between two bearings in degrees, in [-180, 180]."""
    if math.isnan(b1) or math.isnan(b2):
        return float("nan")
    diff = (b2 - b1 + 180.0) % 360.0 - 180.0
    return float(diff)


def compute_temporal_features(
    df: pd.DataFrame,
    haversine_fn=haversine_distance_m,
) -> pd.DataFrame:
    """Compute temporal features for every observation-candidate pair.

    Parameters
    ----------
    df : pd.DataFrame
        Candidate-level DataFrame with candidate generator columns plus
        spatial features. Must contain:
        observation_index, timestamp, gps_longitude, gps_latitude,
        distance_along_segment_m, segment_bearing, gps_bearing.
    haversine_fn : callable
        Function (lat1, lon1, lat2, lon2) -> distance in metres.

    Returns
    -------
    pd.DataFrame
        Copy of df with temporal_* feature columns appended.
    """
    result = df.copy()
    n = len(result)

    # Initialize feature columns with NaN.
    feature_cols = {
        "temporal_time_diff_s": np.full(n, np.nan, dtype=np.float64),
        "temporal_distance_travelled_m": np.full(n, np.nan, dtype=np.float64),
        "temporal_gps_speed_mps": np.full(n, np.nan, dtype=np.float64),
        "temporal_gps_bearing_change_deg": np.full(n, np.nan, dtype=np.float64),
        "temporal_acceleration_mps2": np.full(n, np.nan, dtype=np.float64),
        "temporal_time_since_first_s": np.full(n, np.nan, dtype=np.float64),
        "temporal_is_large_time_gap": np.zeros(n, dtype=np.float64),
        "temporal_movement_consistency": np.full(n, np.nan, dtype=np.float64),
        "temporal_speed_bearing_alignment_deg": np.full(n, np.nan, dtype=np.float64),
    }

    # Required columns.
    required = [
        "observation_index", "timestamp", "gps_longitude", "gps_latitude",
        "gps_bearing",
    ]
    missing = [c for c in required if c not in result.columns]
    if missing:
        raise ValueError(f"Missing required columns for temporal features: {missing}")

    timestamps = result["timestamp"].to_numpy(dtype=np.float64)
    lons = result["gps_longitude"].to_numpy(dtype=np.float64)
    lats = result["gps_latitude"].to_numpy(dtype=np.float64)
    bearings = result["gps_bearing"].to_numpy(dtype=np.float64)
    obs_indices = result["observation_index"].to_numpy(dtype=np.int64)

    # Group by observation_index to get unique observation rows (one per obs).
    # Since each observation can have multiple candidates, we compute temporal
    # features per observation and broadcast to all candidates of that obs.
    obs_unique = np.unique(obs_indices)

    # Map observation index -> position in arrays.
    obs_to_positions: dict[int, list[int]] = {}
    for i, oi in enumerate(obs_indices):
        obs_to_positions.setdefault(int(oi), []).append(i)

    # Compute per-observation temporal values.
    obs_time_diff: dict[int, float] = {}
    obs_distance_travelled: dict[int, float] = {}
    obs_gps_speed: dict[int, float] = {}
    obs_bearing_change: dict[int, float] = {}
    obs_acceleration: dict[int, float] = {}
    obs_time_since_first: dict[int, float] = {}
    obs_is_large_gap: dict[int, bool] = {}
    obs_movement_consistency: dict[int, float] = {}
    obs_speed_bearing_alignment: dict[int, float] = {}

    # Large gap threshold: 10 seconds (from EDA analysis).
    LARGE_GAP_S = 10.0

    # First observation (sorted by observation_index).
    first_obs = int(obs_unique[0])
    first_ts = float(timestamps[obs_indices == first_obs][0])
    first_lat = float(lats[obs_indices == first_obs][0])
    first_lon = float(lons[obs_indices == first_obs][0])

    obs_time_since_first[first_obs] = 0.0
    obs_time_diff[first_obs] = float("nan")
    obs_distance_travelled[first_obs] = float("nan")
    obs_gps_speed[first_obs] = float("nan")
    obs_bearing_change[first_obs] = float("nan")
    obs_acceleration[first_obs] = float("nan")
    obs_is_large_gap[first_obs] = False
    obs_movement_consistency[first_obs] = float("nan")
    obs_speed_bearing_alignment[first_obs] = float("nan")

    prev_obs = first_obs
    prev_ts = first_ts
    prev_lat = first_lat
    prev_lon = first_lon
    prev_bearing = float(bearings[obs_indices == first_obs][0])
    prev_speed = float("nan")

    for oi in obs_unique[1:]:
        positions = obs_to_positions[int(oi)]
        ts = float(timestamps[positions[0]])
        lat = float(lats[positions[0]])
        lon = float(lons[positions[0]])
        bearing = float(bearings[positions[0]])

        time_diff = ts - prev_ts
        obs_time_diff[int(oi)] = time_diff

        if time_diff > 0:
            dist_travelled = haversine_fn(prev_lat, prev_lon, lat, lon)
            obs_distance_travelled[int(oi)] = dist_travelled
            speed = dist_travelled / time_diff
            obs_gps_speed[int(oi)] = speed
        else:
            # Zero or negative time difference.
            obs_distance_travelled[int(oi)] = 0.0
            obs_gps_speed[int(oi)] = float("nan")

        # Time since first observation.
        obs_time_since_first[int(oi)] = ts - first_ts

        # Large time gap indicator.
        is_large_gap = time_diff > LARGE_GAP_S
        obs_is_large_gap[int(oi)] = is_large_gap

        # Bearing change (circular).
        brg_change = _circular_bearing_diff(prev_bearing, bearing)
        obs_bearing_change[int(oi)] = brg_change

        # Acceleration estimate (only if not a large gap and we have prev speed).
        if not is_large_gap and not math.isnan(prev_speed) and time_diff > 0:
            accel = (obs_gps_speed[int(oi)] - prev_speed) / time_diff
            obs_acceleration[int(oi)] = accel
        else:
            obs_acceleration[int(oi)] = float("nan")

        # Movement consistency: low bearing change and reasonable speed.
        if not math.isnan(brg_change) and not math.isnan(obs_gps_speed[int(oi)]):
            # Consistency = 1 / (1 + |bearing_change|) capped by speed sanity.
            consistency = 1.0 / (1.0 + abs(brg_change))
            if obs_gps_speed[int(oi)] > 50.0:  # Unrealistically fast (>180 km/h)
                consistency *= 0.5
            obs_movement_consistency[int(oi)] = consistency
        else:
            obs_movement_consistency[int(oi)] = float("nan")

        # Speed-bearing alignment: difference between GPS movement bearing
        # and candidate segment bearing, per candidate.
        # This is computed per candidate below since segment_bearing varies.

        prev_obs = int(oi)
        prev_ts = ts
        prev_lat = lat
        prev_lon = lon
        prev_bearing = bearing
        prev_speed = obs_gps_speed[int(oi)]

    # Broadcast per-observation values to all candidate rows.
    for oi in obs_unique:
        positions = obs_to_positions[int(oi)]
        for pos in positions:
            feature_cols["temporal_time_diff_s"][pos] = obs_time_diff[int(oi)]
            feature_cols["temporal_distance_travelled_m"][pos] = obs_distance_travelled[int(oi)]
            feature_cols["temporal_gps_speed_mps"][pos] = obs_gps_speed[int(oi)]
            feature_cols["temporal_gps_bearing_change_deg"][pos] = obs_bearing_change[int(oi)]
            feature_cols["temporal_acceleration_mps2"][pos] = obs_acceleration[int(oi)]
            feature_cols["temporal_time_since_first_s"][pos] = obs_time_since_first[int(oi)]
            feature_cols["temporal_is_large_time_gap"][pos] = 1.0 if obs_is_large_gap[int(oi)] else 0.0
            feature_cols["temporal_movement_consistency"][pos] = obs_movement_consistency[int(oi)]

    # Speed-bearing alignment: per candidate, difference between GPS movement
    # bearing and segment bearing.
    if "segment_bearing" in result.columns:
        seg_bearings = result["segment_bearing"].to_numpy(dtype=np.float64)
        for i in range(n):
            gps_brg = bearings[i]
            seg_brg = seg_bearings[i]
            feature_cols["temporal_speed_bearing_alignment_deg"][i] = _circular_bearing_diff(
                gps_brg, seg_brg
            )

    # Append all feature columns.
    for col, values in feature_cols.items():
        result[col] = values

    return result


def get_temporal_feature_names() -> list[str]:
    """Return the list of temporal feature column names."""
    return [
        "temporal_time_diff_s",
        "temporal_distance_travelled_m",
        "temporal_gps_speed_mps",
        "temporal_gps_bearing_change_deg",
        "temporal_acceleration_mps2",
        "temporal_time_since_first_s",
        "temporal_is_large_time_gap",
        "temporal_movement_consistency",
        "temporal_speed_bearing_alignment_deg",
    ]
