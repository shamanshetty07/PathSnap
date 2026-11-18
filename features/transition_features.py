"""Agent 4: Candidate Transition and Sequence Feature Engineering.

Features describing relationships between candidates across consecutive GPS
observations.

Key design decisions:
- Uses actual road-network topology for connectivity checks.
- Respects directed arcs (from_node -> to_node).
- Geometric proximity is NOT treated as proof of connectivity.
- Handles missing candidates and GPS timestamp gaps.
- Avoids creating artificial transitions.
- Does not use reference-route labels as transition features.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from exploration.candidate_generation.geometry import haversine_distance_m


def _circular_bearing_diff(b1: float, b2: float) -> float:
    """Circular difference between two bearings in degrees, in [-180, 180]."""
    if math.isnan(b1) or math.isnan(b2):
        return float("nan")
    diff = (b2 - b1 + 180.0) % 360.0 - 180.0
    return float(diff)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator is None or denominator == 0.0 or math.isnan(denominator):
        return float("nan")
    return numerator / denominator


def compute_transition_features(
    df: pd.DataFrame,
    arcs_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute transition features for every observation-candidate pair.

    For each candidate at observation i, computes features describing the
    transition from candidates at observation i-1. The transition features
    are computed against the closest candidate from the previous observation
    (by projected point distance), with aggregate statistics across all
    previous-observation candidates also recorded.

    Parameters
    ----------
    df : pd.DataFrame
        Candidate-level DataFrame sorted by observation_index. Must contain:
        observation_index, timestamp, arc_id, from_node_id, to_node_id,
        projected_longitude, projected_latitude, distance_along_segment_m,
        segment_length_m, segment_bearing, gps_bearing, distance_m.
    arcs_df : pd.DataFrame, optional
        Road-network arcs with index = arc_id and columns from_node, to_node.
        Used for connectivity checks between consecutive candidate arcs.

    Returns
    -------
    pd.DataFrame
        Copy of df with transition_* feature columns appended.
    """
    result = df.copy()
    n = len(result)

    # Initialize feature columns with NaN.
    feature_cols = {
        "transition_dist_to_prev_projection_m": np.full(n, np.nan, dtype=np.float64),
        "transition_time_diff_s": np.full(n, np.nan, dtype=np.float64),
        "transition_implied_speed_mps": np.full(n, np.nan, dtype=np.float64),
        "transition_candidate_bearing_diff_deg": np.full(n, np.nan, dtype=np.float64),
        "transition_gps_bearing_diff_deg": np.full(n, np.nan, dtype=np.float64),
        "transition_arcs_directly_connected": np.zeros(n, dtype=np.float64),
        "transition_topologically_possible": np.zeros(n, dtype=np.float64),
        "transition_network_distance_m": np.full(n, np.nan, dtype=np.float64),
        "transition_network_vs_gps_diff_m": np.full(n, np.nan, dtype=np.float64),
        "transition_feasibility_score": np.full(n, np.nan, dtype=np.float64),
        "transition_candidate_continuity": np.full(n, np.nan, dtype=np.float64),
        "transition_num_prev_candidates": np.full(n, np.nan, dtype=np.float64),
        "transition_min_prev_dist_m": np.full(n, np.nan, dtype=np.float64),
    }

    required = [
        "observation_index", "timestamp", "arc_id", "from_node_id", "to_node_id",
        "projected_longitude", "projected_latitude", "distance_along_segment_m",
        "segment_length_m", "segment_bearing", "gps_bearing", "distance_m",
    ]
    missing = [c for c in required if c not in result.columns]
    if missing:
        raise ValueError(f"Missing required columns for transition features: {missing}")

    # Build arc connectivity map from arcs_df.
    # arc_id -> set of to_node_ids of outgoing arcs from its to_node.
    # Two arcs A (from->to) and B (from2->to2) are directly connected
    # if A.to_node == B.from_node.
    arc_to_node: dict[int, int] = {}
    node_to_outgoing_arcs: dict[int, set[int]] = {}
    if arcs_df is not None and not arcs_df.empty:
        for aid, row in arcs_df.iterrows():
            aid_int = int(aid)
            fn = int(row["from_node"])
            tn = int(row["to_node"])
            arc_to_node[aid_int] = tn
            node_to_outgoing_arcs.setdefault(fn, set()).add(aid_int)

    # Group candidates by observation_index.
    obs_to_rows: dict[int, list[int]] = {}
    for i in range(n):
        oi = int(result.iloc[i]["observation_index"])
        obs_to_rows.setdefault(oi, []).append(i)

    sorted_obs = sorted(obs_to_rows.keys())

    # Precompute per-observation data.
    obs_data: dict[int, dict] = {}
    for oi in sorted_obs:
        rows = obs_to_rows[oi]
        obs_data[oi] = {
            "row_indices": rows,
            "timestamps": [float(result.iloc[r]["timestamp"]) for r in rows],
            "projected_lons": [float(result.iloc[r]["projected_longitude"]) for r in rows],
            "projected_lats": [float(result.iloc[r]["projected_latitude"]) for r in rows],
            "arc_ids": [int(result.iloc[r]["arc_id"]) for r in rows],
            "from_nodes": [int(result.iloc[r]["from_node_id"]) for r in rows],
            "to_nodes": [int(result.iloc[r]["to_node_id"]) for r in rows],
            "segment_bearings": [float(result.iloc[r]["segment_bearing"]) for r in rows],
            "gps_bearings": [float(result.iloc[r]["gps_bearing"]) for r in rows],
        }

    # For each observation i > 0, compute transition features for each candidate
    # based on the previous observation's candidates.
    for obs_idx in sorted_obs[1:]:
        prev_obs = sorted_obs[sorted_obs.index(obs_idx) - 1]

        curr_data = obs_data[obs_idx]
        prev_data = obs_data[prev_obs]

        curr_ts = curr_data["timestamps"][0]
        prev_ts = prev_data["timestamps"][0]
        time_diff = curr_ts - prev_ts

        num_prev = len(prev_data["row_indices"])

        for j, curr_row in enumerate(curr_data["row_indices"]):
            # Distance to closest previous projection.
            min_prev_dist = float("inf")
            best_prev_j = -1

            curr_proj_lat = curr_data["projected_lats"][j]
            curr_proj_lon = curr_data["projected_lons"][j]
            curr_arc_id = curr_data["arc_ids"][j]
            curr_from_node = curr_data["from_nodes"][j]
            curr_to_node = curr_data["to_nodes"][j]
            curr_seg_brg = curr_data["segment_bearings"][j]
            curr_gps_brg = curr_data["gps_bearings"][j]
            curr_dist_along = float(result.iloc[curr_row]["distance_along_segment_m"])
            curr_seg_len = float(result.iloc[curr_row]["segment_length_m"])

            # Transition features aggregated across all prev candidates.
            all_prev_dists = []
            all_prev_connected = []
            all_prev_topological = []
            all_prev_implied_speeds = []

            for k in range(num_prev):
                prev_proj_lat = prev_data["projected_lats"][k]
                prev_proj_lon = prev_data["projected_lons"][k]
                prev_arc_id = prev_data["arc_ids"][k]
                prev_to_node = prev_data["to_nodes"][k]
                prev_from_node = prev_data["from_nodes"][k]
                prev_seg_brg = prev_data["segment_bearings"][k]
                prev_gps_brg = prev_data["gps_bearings"][k]

                # Projected point distance.
                proj_dist = haversine_distance_m(
                    prev_proj_lat, prev_proj_lon, curr_proj_lat, curr_proj_lon
                )
                all_prev_dists.append(proj_dist)

                if proj_dist < min_prev_dist:
                    min_prev_dist = proj_dist
                    best_prev_j = k

                # Direct connectivity: prev.to_node == curr.from_node.
                directly_connected = 1.0 if prev_to_node == curr_from_node else 0.0
                all_prev_connected.append(directly_connected)

                # Topologically possible: there's an arc from prev.to_node
                # to curr.from_node in the road network.
                topological = 0.0
                if prev_to_node in node_to_outgoing_arcs:
                    # Check if any outgoing arc from prev.to_node leads to curr.from_node
                    # OR if curr.from_node == prev.to_node (direct connection).
                    if prev_to_node == curr_from_node:
                        topological = 1.0
                    else:
                        # Check if there's an arc from prev.to_node that starts a path
                        # to curr.from_node. For efficiency, just check one-hop.
                        for next_aid in node_to_outgoing_arcs[prev_to_node]:
                            if next_aid in arc_to_node and arc_to_node[next_aid] == curr_from_node:
                                topological = 1.0
                                break
                all_prev_topological.append(topological)

                # Implied speed between projections.
                if time_diff > 0:
                    implied_speed = proj_dist / time_diff
                else:
                    implied_speed = float("nan")
                all_prev_implied_speeds.append(implied_speed)

            # Best (closest) previous candidate features.
            if best_prev_j >= 0:
                feature_cols["transition_dist_to_prev_projection_m"][curr_row] = min_prev_dist
                feature_cols["transition_num_prev_candidates"][curr_row] = float(num_prev)
                feature_cols["transition_min_prev_dist_m"][curr_row] = min_prev_dist

                # Time difference.
                feature_cols["transition_time_diff_s"][curr_row] = time_diff

                # Implied speed using closest projection.
                if time_diff > 0 and not math.isnan(min_prev_dist):
                    feature_cols["transition_implied_speed_mps"][curr_row] = min_prev_dist / time_diff

                # Bearing differences.
                prev_seg_brg = prev_data["segment_bearings"][best_prev_j]
                prev_gps_brg = prev_data["gps_bearings"][best_prev_j]
                feature_cols["transition_candidate_bearing_diff_deg"][curr_row] = _circular_bearing_diff(
                    prev_seg_brg, curr_seg_brg
                )
                feature_cols["transition_gps_bearing_diff_deg"][curr_row] = _circular_bearing_diff(
                    prev_gps_brg, curr_gps_brg
                )

                # Connectivity.
                prev_to_node = prev_data["to_nodes"][best_prev_j]
                feature_cols["transition_arcs_directly_connected"][curr_row] = (
                    1.0 if prev_to_node == curr_from_node else 0.0
                )

                # Topological possibility.
                topological = 0.0
                if prev_to_node in node_to_outgoing_arcs:
                    if prev_to_node == curr_from_node:
                        topological = 1.0
                    else:
                        for next_aid in node_to_outgoing_arcs[prev_to_node]:
                            if next_aid in arc_to_node and arc_to_node[next_aid] == curr_from_node:
                                topological = 1.0
                                break
                feature_cols["transition_topologically_possible"][curr_row] = topological

                # Network distance vs GPS displacement.
                # Network distance = distance along prev segment to its end +
                # distance along curr segment from its start (if directly connected).
                if prev_to_node == curr_from_node:
                    prev_dist_along = float(result.iloc[prev_data["row_indices"][best_prev_j]]["distance_along_segment_m"])
                    prev_seg_len = float(result.iloc[prev_data["row_indices"][best_prev_j]]["segment_length_m"])
                    network_dist = (prev_seg_len - prev_dist_along) + curr_dist_along
                else:
                    network_dist = float("nan")
                feature_cols["transition_network_distance_m"][curr_row] = network_dist

                if not math.isnan(network_dist) and not math.isnan(min_prev_dist):
                    feature_cols["transition_network_vs_gps_diff_m"][curr_row] = abs(network_dist - min_prev_dist)

                # Transition feasibility: high if directly connected, low implied speed,
                # small bearing change.
                bearing_change = abs(_circular_bearing_diff(prev_seg_brg, curr_seg_brg))
                speed_ok = 1.0
                if not math.isnan(min_prev_dist) and time_diff > 0:
                    speed = min_prev_dist / time_diff
                    if speed > 30.0:  # >108 km/h unrealistic
                        speed_ok = 0.5
                feat = 0.0
                if prev_to_node == curr_from_node:
                    feat += 0.5
                feat += 0.3 * (1.0 - min(bearing_change / 180.0, 1.0))
                feat += 0.2 * speed_ok
                feature_cols["transition_feasibility_score"][curr_row] = feat

                # Candidate continuity: relative positions along segments.
                prev_rel_pos = _safe_divide(
                    float(result.iloc[prev_data["row_indices"][best_prev_j]]["distance_along_segment_m"]),
                    float(result.iloc[prev_data["row_indices"][best_prev_j]]["segment_length_m"]),
                )
                curr_rel_pos = _safe_divide(curr_dist_along, curr_seg_len)
                if not (math.isnan(prev_rel_pos) or math.isnan(curr_rel_pos)):
                    continuity = 1.0 - abs(curr_rel_pos - prev_rel_pos)
                    feature_cols["transition_candidate_continuity"][curr_row] = continuity

    # Append all feature columns.
    for col, values in feature_cols.items():
        result[col] = values

    return result


def get_transition_feature_names() -> list[str]:
    """Return the list of transition feature column names."""
    return [
        "transition_dist_to_prev_projection_m",
        "transition_time_diff_s",
        "transition_implied_speed_mps",
        "transition_candidate_bearing_diff_deg",
        "transition_gps_bearing_diff_deg",
        "transition_arcs_directly_connected",
        "transition_topologically_possible",
        "transition_network_distance_m",
        "transition_network_vs_gps_diff_m",
        "transition_feasibility_score",
        "transition_candidate_continuity",
        "transition_num_prev_candidates",
        "transition_min_prev_dist_m",
    ]
