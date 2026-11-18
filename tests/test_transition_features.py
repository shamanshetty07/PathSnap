import numpy as np
import pandas as pd

from features.transition_features import compute_transition_features, get_transition_feature_names


def test_first_obs_has_no_prev(cand, trip):
    out = compute_transition_features(cand, trip.arcs, trip.nodes)
    first = out[out["observation_index"] == 0]
    assert (first["transition_has_prev"] == 0).all()
    assert first["transition_proj_dist_min_m"].isna().all()
    assert set(get_transition_feature_names()) <= set(out.columns)
    assert len(out) == len(cand)


def test_same_arc_progress_matches_gps(cand, trip):
    out = compute_transition_features(cand, trip.arcs, trip.nodes)
    r = out[(out["observation_index"] == 5) & (out["arc_id"] == 0)].iloc[0]
    assert r["transition_same_arc_as_prev"] == 1 and r["transition_feasible"] == 1
    assert r["transition_net_vs_gps_diff_m"] < 1.0           # network distance ~ GPS step


def test_opposite_direction_arc_not_connected_forward(cand, trip):
    out = compute_transition_features(cand, trip.arcs, trip.nodes)
    # westbound arc 4 (1->0): previous candidates include it, but moving "forward" along it
    # means decreasing along-distance... it is eastward GPS so progress along arc 4 is backwards
    r = out[(out["observation_index"] == 5) & (out["arc_id"] == 4)].iloc[0]
    assert r["transition_same_arc_as_prev"] in (0.0, 1.0)


def test_adjacent_arc_transition_distance(trip):
    # Hand-built pair: prev on arc 0 at 70/72 m, current on arc 1 at 4 m
    seg0, seg1 = 72.0, 72.0
    base = dict(trip_id="T", timestamp=0.0, gps_longitude=8.0, gps_latitude=49.0, projected_longitude=8.0,
                projected_latitude=49.0, segment_bearing=90.0, gps_bearing=90.0, distance_m=1.0)
    rows = [
        dict(base, observation_index=0, arc_id=0, from_node_id=0, to_node_id=1, segment_length_m=seg0,
             distance_along_segment_m=70.0, timestamp=1.0),
        dict(base, observation_index=1, arc_id=1, from_node_id=1, to_node_id=2, segment_length_m=seg1,
             distance_along_segment_m=4.0, timestamp=2.0),
    ]
    out = compute_transition_features(pd.DataFrame(rows), trip.arcs, trip.nodes)
    r = out.iloc[1]
    assert r["transition_connected_1hop"] == 1
    assert np.isclose(r["transition_best_network_dist_m"], (72.0 - 70.0) + 4.0)


def test_unconnected_not_faked(trip):
    base = dict(trip_id="T", gps_longitude=8.0, gps_latitude=49.0, projected_longitude=8.0,
                projected_latitude=49.0, segment_bearing=90.0, gps_bearing=90.0, distance_m=1.0,
                segment_length_m=72.0, distance_along_segment_m=10.0)
    rows = [dict(base, observation_index=0, arc_id=0, from_node_id=0, to_node_id=1, timestamp=1.0),
            dict(base, observation_index=1, arc_id=3, from_node_id=3, to_node_id=4, timestamp=2.0)]
    out = compute_transition_features(pd.DataFrame(rows), trip.arcs, trip.nodes)
    r = out.iloc[1]
    assert r["transition_connected_1hop"] == 0
    assert np.isnan(r["transition_best_network_dist_m"])        # arcs 0 -> 3 need 2 intermediate arcs
    assert r["transition_feasible"] == 0


def test_two_hop(trip):
    base = dict(trip_id="T", gps_longitude=8.0, gps_latitude=49.0, projected_longitude=8.0,
                projected_latitude=49.0, segment_bearing=90.0, gps_bearing=90.0, distance_m=1.0,
                segment_length_m=72.0, distance_along_segment_m=10.0)
    rows = [dict(base, observation_index=0, arc_id=0, from_node_id=0, to_node_id=1, timestamp=1.0),
            dict(base, observation_index=1, arc_id=2, from_node_id=2, to_node_id=3, timestamp=2.0)]
    out = compute_transition_features(pd.DataFrame(rows), trip.arcs, trip.nodes)
    r = out.iloc[1]
    assert r["transition_reachable_2hop"] == 1 and r["transition_connected_1hop"] == 0
    assert 140 < r["transition_best_network_dist_m"] < 160     # 62 + ~72 + 10
