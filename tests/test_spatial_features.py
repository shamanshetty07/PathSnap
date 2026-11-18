import numpy as np

from features.spatial_features import compute_spatial_features, get_spatial_feature_names


def test_columns_and_row_preservation(cand, trip):
    out = compute_spatial_features(cand, trip.nodes)
    assert len(out) == len(cand)
    assert set(get_spatial_feature_names()) <= set(out.columns)
    assert (out[["trip_id", "observation_index", "arc_id"]] == cand[["trip_id", "observation_index", "arc_id"]]).all().all()


def test_geometry_identities(cand, trip):
    out = compute_spatial_features(cand, trip.nodes)
    np.testing.assert_allclose(out["spatial_along_m"] + out["spatial_remaining_m"], out["spatial_segment_length_m"], atol=1e-6)
    assert out["spatial_rel_position"].between(0, 1).all()
    assert (out["spatial_dist_excess_m"] >= 0).all()
    # nearest candidate has rank 0 and excess 0
    best = out[out["spatial_dist_rank"] == 0]
    assert (best["spatial_dist_excess_m"] == 0).all()
    assert best["observation_index"].is_unique
    # GPS is ~2.2 m north of the road
    assert out["spatial_dist_to_segment_m"].min() < 3.0
    assert (out["spatial_offset_north_m"].abs() < 3.0).all()


def test_endpoint_distance_matches_projection(cand, trip):
    out = compute_spatial_features(cand, trip.nodes)
    # GPS-to-node distance >= along-distance minus tiny tolerance? use triangle ineq: node dist >= |along - offset|
    assert (out["spatial_dist_to_start_node_m"] + 1e-6 >= (out["spatial_along_m"] - 3.0).clip(lower=0)).all()


def test_missing_nodes_give_nan_not_zero(cand):
    out = compute_spatial_features(cand, None)
    assert out["spatial_dist_to_start_node_m"].isna().all()


def test_deterministic(cand, trip):
    a = compute_spatial_features(cand, trip.nodes)
    b = compute_spatial_features(cand, trip.nodes)
    assert a.equals(b)
