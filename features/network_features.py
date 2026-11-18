"""Agent 3: Road Network and Direction Feature Engineering.

Features describing the candidate road segment and its relationship to vehicle
movement.

Key design decisions:
- Directed arc IDs are preserved (from_node -> to_node).
- Geometric proximity is NOT treated as graph connectivity.
- No expensive all-pairs graph calculations.
- Connectivity is determined by matching node IDs (to_node of one arc ==
  from_node of another).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from exploration.candidate_generation.geometry import haversine_distance_m  # noqa: F401


def _circular_bearing_diff(b1: float, b2: float) -> float:
    """Circular difference between two bearings in degrees, in [-180, 180]."""
    if math.isnan(b1) or math.isnan(b2):
        return float("nan")
    diff = (b2 - b1 + 180.0) % 360.0 - 180.0
    return float(diff)


def compute_network_features(
    df: pd.DataFrame,
    arcs_df: pd.DataFrame | None = None,
    nodes_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute network features for every observation-candidate pair.

    Parameters
    ----------
    df : pd.DataFrame
        Candidate-level DataFrame with candidate generator columns plus
        spatial/temporal features. Must contain:
        observation_index, arc_id, from_node_id, to_node_id,
        segment_bearing, gps_bearing, segment_length_m,
        distance_along_segment_m.
    arcs_df : pd.DataFrame, optional
        Road-network arcs with index = arc_id and columns from_node, to_node.
        Used for connectivity checks.
    nodes_df : pd.DataFrame, optional
        Road-network nodes with index = node_id. Used for node degree info.

    Returns
    -------
    pd.DataFrame
        Copy of df with network_* feature columns appended.
    """
    result = df.copy()
    n = len(result)

    # Initialize feature columns with NaN.
    feature_cols = {
        "network_bearing_diff_deg": np.full(n, np.nan, dtype=np.float64),
        "network_directional_compatibility": np.full(n, np.nan, dtype=np.float64),
        "network_segment_orientation_deg": np.full(n, np.nan, dtype=np.float64),
        "network_segment_length_m": np.full(n, np.nan, dtype=np.float64),
        "network_distance_along_segment_m": np.full(n, np.nan, dtype=np.float64),
        "network_candidate_direction_consistency": np.full(n, np.nan, dtype=np.float64),
        "network_arc_out_degree": np.full(n, np.nan, dtype=np.float64),
        "network_arc_in_degree": np.full(n, np.nan, dtype=np.float64),
        "network_from_node_degree": np.full(n, np.nan, dtype=np.float64),
        "network_to_node_degree": np.full(n, np.nan, dtype=np.float64),
        "network_is_self_loop": np.zeros(n, dtype=np.float64),
        "network_has_outgoing_arc": np.zeros(n, dtype=np.float64),
        "network_has_incoming_arc": np.zeros(n, dtype=np.float64),
    }

    required = [
        "observation_index", "arc_id", "from_node_id", "to_node_id",
        "segment_bearing", "gps_bearing", "segment_length_m",
        "distance_along_segment_m",
    ]
    missing = [c for c in required if c not in result.columns]
    if missing:
        raise ValueError(f"Missing required columns for network features: {missing}")

    arc_ids = result["arc_id"].to_numpy(dtype=np.int64)
    from_nodes = result["from_node_id"].to_numpy(dtype=np.int64)
    to_nodes = result["to_node_id"].to_numpy(dtype=np.int64)
    seg_bearings = result["segment_bearing"].to_numpy(dtype=np.float64)
    gps_bearings = result["gps_bearing"].to_numpy(dtype=np.float64)
    seg_lengths = result["segment_length_m"].to_numpy(dtype=np.float64)
    dist_along = result["distance_along_segment_m"].to_numpy(dtype=np.float64)

    # Bearing difference (GPS movement vs segment).
    for i in range(n):
        feature_cols["network_bearing_diff_deg"][i] = _circular_bearing_diff(
            gps_bearings[i], seg_bearings[i]
        )

    # Directional compatibility: cosine of bearing difference, in [-1, 1].
    bearing_diff = feature_cols["network_bearing_diff_deg"]
    compatibility = np.array([
        math.cos(math.radians(d)) if not math.isnan(d) else float("nan")
        for d in bearing_diff
    ], dtype=np.float64)
    feature_cols["network_directional_compatibility"] = compatibility

    # Segment orientation (same as segment_bearing but renamed for clarity).
    feature_cols["network_segment_orientation_deg"] = seg_bearings

    # Segment length.
    feature_cols["network_segment_length_m"] = seg_lengths

    # Distance along segment.
    feature_cols["network_distance_along_segment_m"] = dist_along

    # Candidate direction consistency: how aligned GPS movement is with arc.
    # Combines directional compatibility with relative position along segment.
    relative_pos = np.array([
        _safe_divide(d, s) for d, s in zip(dist_along, seg_lengths)
    ], dtype=np.float64)
    direction_consistency = np.array([
        c * (1.0 - abs(rp - 0.5) * 2.0) if not (math.isnan(c) or math.isnan(rp))
        else float("nan")
        for c, rp in zip(compatibility, relative_pos)
    ], dtype=np.float64)
    feature_cols["network_candidate_direction_consistency"] = direction_consistency

    # Build arc connectivity maps from arcs_df if available.
    arc_outgoing: dict[int, set[int]] = {}
    arc_incoming: dict[int, set[int]] = {}
    node_out_degree: dict[int, int] = {}
    node_in_degree: dict[int, int] = {}

    if arcs_df is not None and not arcs_df.empty:
        for aid, row in arcs_df.iterrows():
            aid_int = int(aid)
            fn = int(row["from_node"])
            tn = int(row["to_node"])
            arc_outgoing.setdefault(aid_int, set()).add(tn)
            arc_incoming.setdefault(aid_int, set()).add(fn)
            node_out_degree[fn] = node_out_degree.get(fn, 0) + 1
            node_in_degree[tn] = node_in_degree.get(tn, 0) + 1

    # Per-candidate connectivity features.
    for i in range(n):
        aid = int(arc_ids[i])
        fn = int(from_nodes[i])
        tn = int(to_nodes[i])

        # Self-loop.
        feature_cols["network_is_self_loop"][i] = 1.0 if fn == tn else 0.0

        # Node degrees.
        feature_cols["network_from_node_degree"][i] = float(
            node_out_degree.get(fn, 0) + node_in_degree.get(fn, 0)
        )
        feature_cols["network_to_node_degree"][i] = float(
            node_out_degree.get(tn, 0) + node_in_degree.get(tn, 0)
        )

        # Whether this arc has outgoing connections (from its to_node).
        has_outgoing = 0
        for other_aid, targets in arc_outgoing.items():
            if other_aid != aid and tn in targets:
                has_outgoing = 1
                break
        feature_cols["network_has_outgoing_arc"][i] = float(has_outgoing)

        # Whether this arc has incoming connections (to its from_node).
        has_incoming = 0
        for other_aid, sources in arc_incoming.items():
            if other_aid != aid and fn in sources:
                has_incoming = 1
                break
        feature_cols["network_has_incoming_arc"][i] = float(has_incoming)

    # Append all feature columns.
    for col, values in feature_cols.items():
        result[col] = values

    return result


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator is None or denominator == 0.0 or math.isnan(denominator):
        return float("nan")
    return numerator / denominator


def get_network_feature_names() -> list[str]:
    """Return the list of network feature column names."""
    return [
        "network_bearing_diff_deg",
        "network_directional_compatibility",
        "network_segment_orientation_deg",
        "network_segment_length_m",
        "network_distance_along_segment_m",
        "network_candidate_direction_consistency",
        "network_arc_out_degree",
        "network_arc_in_degree",
        "network_from_node_degree",
        "network_to_node_degree",
        "network_is_self_loop",
        "network_has_outgoing_arc",
        "network_has_incoming_arc",
    ]
