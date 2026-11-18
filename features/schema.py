"""Common feature schema for PathSnap.

One row = one GPS observation x one candidate road arc.
Keys: (trip_id, observation_index, arc_id).

Column roles
------------
KEY_COLUMNS      identifiers, never model inputs.
BASE_COLUMNS     raw candidate-generator output passed through for traceability
                 (raw coordinates / timestamp / node ids); never model inputs.
feature columns  anything starting with FEATURE_PREFIXES (model inputs).
LABEL_COLUMNS    produced by features.label_alignment from the reference route.
                 NEVER model inputs (leakage).
"""

from __future__ import annotations

import pandas as pd

KEY_COLUMNS = ["trip_id", "observation_index", "arc_id"]

BASE_COLUMNS = [
    "timestamp", "gps_longitude", "gps_latitude", "from_node_id", "to_node_id",
    "distance_m", "projected_longitude", "projected_latitude",
    "segment_length_m", "distance_along_segment_m", "segment_bearing",
    "gps_bearing",
]

FEATURE_PREFIXES = ("spatial_", "temporal_", "network_", "transition_")

LABEL_COLUMNS = ["label_aligned", "label_obs_has_label", "label_on_route_anywhere",
                 "label_route_position"]

# Columns the candidate generator must supply to every feature group.
REQUIRED_CANDIDATE_COLUMNS = [
    "observation_index", "arc_id", "timestamp", "gps_longitude", "gps_latitude",
    "from_node_id", "to_node_id", "distance_m", "projected_longitude",
    "projected_latitude", "segment_length_m", "distance_along_segment_m",
    "segment_bearing",
]

# Why a feature can legitimately be NaN (never silently zero-filled).
MISSING_REASONS = {
    "spatial_relative_position": "zero-length segment (division by zero)",
    "spatial_normalized_distance": "zero-length segment (division by zero)",
    "temporal_time_diff_s": "first observation has no predecessor",
    "temporal_gps_displacement_m": "first observation has no predecessor",
    "temporal_gps_speed_mps": "first observation, or non-positive time difference",
    "temporal_gps_bearing_deg": "first observation or vehicle stationary (<2 m): bearing undefined",
    "temporal_fwd_bearing_deg": "last observation or vehicle stationary (<2 m): bearing undefined",
    "temporal_gps_bearing_change_deg": "needs two defined consecutive movement bearings",
    "temporal_acceleration_mps2": "needs two valid consecutive speeds and no large time gap",
    "temporal_movement_consistency": "bearing change undefined",
    "network_bearing_diff_back_deg": "GPS movement bearing undefined (first obs / stationary)",
    "network_bearing_diff_fwd_deg": "GPS forward bearing undefined (last obs / stationary)",
    "network_directional_compat": "both GPS bearings undefined",
    "network_is_reverse_direction": "both GPS bearings undefined",
    "transition_*": "no previous observation with candidates (first obs / preceding candidate miss), "
                    "or no previous candidate reachable within 2 hops (network-distance features only)",
}


def validate_candidate_frame(df: pd.DataFrame, extra=()) -> None:
    missing = [c for c in list(REQUIRED_CANDIDATE_COLUMNS) + list(extra) if c not in df.columns]
    if missing:
        raise ValueError(f"Candidate frame is missing required columns: {missing}")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model-input feature columns present in *df* (prefix-based, labels excluded)."""
    return [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]


def all_feature_names() -> list[str]:
    from .spatial_features import get_spatial_feature_names
    from .temporal_features import get_temporal_feature_names
    from .network_features import get_network_feature_names
    from .transition_features import get_transition_feature_names
    return (get_spatial_feature_names() + get_temporal_feature_names()
            + get_network_feature_names() + get_transition_feature_names())
