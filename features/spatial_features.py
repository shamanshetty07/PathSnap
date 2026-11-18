"""Agent 1: Spatial Feature Engineering.

Spatial and geometric features for every GPS observation-candidate pair.

All distances are in metres. Zero-length segments and missing/invalid
coordinates are handled gracefully.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from exploration.candidate_generation.geometry import (
    EARTH_RADIUS_M,
    haversine_distance_m,
    project_point_on_segment,
)


def _safe_divide(numerator: float, denominator: float) -> float:
    """Safe division returning NaN when denominator is zero or invalid."""
    if denominator is None or denominator == 0.0 or math.isnan(denominator):
        return float("nan")
    return numerator / denominator


def compute_spatial_features(
    df: pd.DataFrame,
    projection_fn=None,
    nodes_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute spatial features for every observation-candidate pair.

    Parameters
    ----------
    df : pd.DataFrame
        Candidate-level DataFrame with columns from the candidate generator:
        observation_index, timestamp, gps_longitude, gps_latitude, arc_id,
        from_node_id, to_node_id, distance_m, projected_longitude,
        projected_latitude, segment_length_m, distance_along_segment_m,
        segment_bearing, gps_bearing.
    projection_fn : callable, optional
        Function (longitude, latitude) -> (x, y) in metres. If provided,
        projected coordinates are used for endpoint distances. Otherwise
        Haversine is used.
    nodes_df : pd.DataFrame, optional
        Road-network nodes with index = node_id and columns longitude, latitude.
        Used for endpoint distances when projection_fn is not available.

    Returns
    -------
    pd.DataFrame
        Copy of df with spatial_* feature columns appended.
    """
    result = df.copy()

    n = len(result)

    # Initialize all feature columns with NaN.
    feature_cols = {
        "spatial_distance_to_segment_m": np.full(n, np.nan, dtype=np.float64),
        "spatial_normalized_distance": np.full(n, np.nan, dtype=np.float64),
        "spatial_distance_along_segment_m": np.full(n, np.nan, dtype=np.float64),
        "spatial_remaining_distance_m": np.full(n, np.nan, dtype=np.float64),
        "spatial_relative_position": np.full(n, np.nan, dtype=np.float64),
        "spatial_distance_to_start_node_m": np.full(n, np.nan, dtype=np.float64),
        "spatial_distance_to_end_node_m": np.full(n, np.nan, dtype=np.float64),
        "spatial_segment_length_m": np.full(n, np.nan, dtype=np.float64),
        "spatial_is_zero_length_segment": np.zeros(n, dtype=np.float64),
        "spatial_gps_to_projected_offset_m": np.full(n, np.nan, dtype=np.float64),
    }

    # Validate required columns exist.
    required = [
        "distance_m", "segment_length_m", "distance_along_segment_m",
        "gps_longitude", "gps_latitude", "projected_longitude",
        "projected_latitude",
    ]
    missing = [c for c in required if c not in result.columns]
    if missing:
        raise ValueError(f"Missing required columns for spatial features: {missing}")

    distance_to_segment = result["distance_m"].to_numpy(dtype=np.float64)
    segment_length = result["segment_length_m"].to_numpy(dtype=np.float64)
    distance_along = result["distance_along_segment_m"].to_numpy(dtype=np.float64)

    feature_cols["spatial_distance_to_segment_m"] = distance_to_segment

    # Normalized distance relative to segment length.
    normalized = np.array([
        _safe_divide(d, s) for d, s in zip(distance_to_segment, segment_length)
    ], dtype=np.float64)
    feature_cols["spatial_normalized_distance"] = normalized

    # Distance along segment (already computed by candidate generator).
    feature_cols["spatial_distance_along_segment_m"] = distance_along

    # Remaining distance along segment.
    remaining = np.array([
        _safe_divide(s - d, 1.0) if not math.isnan(s) and not math.isnan(d)
        else float("nan")
        for s, d in zip(segment_length, distance_along)
    ], dtype=np.float64)
    feature_cols["spatial_remaining_distance_m"] = remaining

    # Relative position (t value) along segment.
    relative_pos = np.array([
        _safe_divide(d, s) for d, s in zip(distance_along, segment_length)
    ], dtype=np.float64)
    feature_cols["spatial_relative_position"] = relative_pos

    # Zero-length segment indicator.
    zero_length = segment_length == 0.0
    feature_cols["spatial_is_zero_length_segment"] = zero_length.astype(np.float64)

    # GPS to projected point offset (same as distance to segment).
    feature_cols["spatial_gps_to_projected_offset_m"] = distance_to_segment

    # Endpoint distances (Haversine from GPS to segment endpoints).
    if nodes_df is not None and not nodes_df.empty:
        node_lon = nodes_df["longitude"].to_dict()
        node_lat = nodes_df["latitude"].to_dict()

        dist_to_start = []
        dist_to_end = []
        for _, row in result.iterrows():
            from_node = int(row["from_node_id"])
            to_node = int(row["to_node_id"])
            gps_lat = float(row["gps_latitude"])
            gps_lon = float(row["gps_longitude"])

            if from_node in node_lon and to_node in node_lat:
                d_start = haversine_distance_m(
                    gps_lat, gps_lon,
                    float(node_lat[from_node]), float(node_lon[from_node]),
                )
            else:
                d_start = float("nan")

            if to_node in node_lon and to_node in node_lat:
                d_end = haversine_distance_m(
                    gps_lat, gps_lon,
                    float(node_lat[to_node]), float(node_lon[to_node]),
                )
            else:
                d_end = float("nan")

            dist_to_start.append(d_start)
            dist_to_end.append(d_end)
    else:
        # Fallback: use projected coordinates if projection available,
        # otherwise leave NaN.
        dist_to_start = [float("nan")] * n
        dist_to_end = [float("nan")] * n

        if projection_fn is not None:
            gps_xys = np.array([
                projection_fn(float(row["gps_longitude"]), float(row["gps_latitude"]))
                for _, row in result.iterrows()
            ])
            start_xys = np.array([
                (float(row.get("ax", np.nan)), float(row.get("ay", np.nan)))
                if "ax" in result.columns else (np.nan, np.nan)
                for _, row in result.iterrows()
            ])
            end_xys = np.array([
                (float(row.get("bx", np.nan)), float(row.get("by", np.nan)))
                if "bx" in result.columns else (np.nan, np.nan)
                for _, row in result.iterrows()
            ])
            for i in range(n):
                if not np.any(np.isnan(gps_xys[i])) and not np.any(np.isnan(start_xys[i])):
                    dist_to_start[i] = float(np.hypot(
                        gps_xys[i, 0] - start_xys[i, 0],
                        gps_xys[i, 1] - start_xys[i, 1],
                    ))
                if not np.any(np.isnan(gps_xys[i])) and not np.any(np.isnan(end_xys[i])):
                    dist_to_end[i] = float(np.hypot(
                        gps_xys[i, 0] - end_xys[i, 0],
                        gps_xys[i, 1] - end_xys[i, 1],
                    ))

    feature_cols["spatial_distance_to_start_node_m"] = np.array(
        dist_to_start, dtype=np.float64
    )
    feature_cols["spatial_distance_to_end_node_m"] = np.array(
        dist_to_end, dtype=np.float64
    )

    # Segment length feature.
    feature_cols["spatial_segment_length_m"] = segment_length

    # Spatial displacement: horizontal/vertical components in projected space.
    if projection_fn is not None:
        disp_x = np.full(n, np.nan, dtype=np.float64)
        disp_y = np.full(n, np.nan, dtype=np.float64)
        for i, (_, row) in enumerate(result.iterrows()):
            try:
                gps_x, gps_y = projection_fn(
                    float(row["gps_longitude"]), float(row["gps_latitude"])
                )
                proj_x, proj_y = projection_fn(
                    float(row["projected_longitude"]),
                    float(row["projected_latitude"]),
                )
                disp_x[i] = proj_x - gps_x
                disp_y[i] = proj_y - gps_y
            except Exception:
                pass
        feature_cols["spatial_displacement_x_m"] = disp_x
        feature_cols["spatial_displacement_y_m"] = disp_y

    # Append all feature columns.
    for col, values in feature_cols.items():
        result[col] = values

    return result


def get_spatial_feature_names() -> list[str]:
    """Return the list of spatial feature column names."""
    return [
        "spatial_distance_to_segment_m",
        "spatial_normalized_distance",
        "spatial_distance_along_segment_m",
        "spatial_remaining_distance_m",
        "spatial_relative_position",
        "spatial_distance_to_start_node_m",
        "spatial_distance_to_end_node_m",
        "spatial_segment_length_m",
        "spatial_is_zero_length_segment",
        "spatial_gps_to_projected_offset_m",
        "spatial_displacement_x_m",
        "spatial_displacement_y_m",
    ]
