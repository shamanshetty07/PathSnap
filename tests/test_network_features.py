import numpy as np

from features.network_features import compute_network_features, get_network_feature_names
from features.temporal_features import compute_temporal_features


def _prep(cand, trip):
    return compute_network_features(compute_temporal_features(cand, track=trip.track), trip.arcs, trip.nodes)


def test_direction_distinguishes_two_way_arcs(cand, trip):
    out = _prep(cand, trip)
    mid = out[out["observation_index"] == 10]
    east = mid[mid["arc_id"] <= 3]      # eastbound arcs match movement
    west = mid[mid["arc_id"] >= 4]
    assert (east["network_directional_compat"] > 0.99).all()
    assert (west["network_directional_compat"] < -0.99).all()
    assert (west["network_is_reverse_direction"] == 1).all()
    assert (east["network_is_reverse_direction"] == 0).all()


def test_topology_features(cand, trip):
    out = _prep(cand, trip)
    assert set(get_network_feature_names()) <= set(out.columns)
    a0 = out[out["arc_id"] == 0].iloc[0]
    assert a0["network_has_reverse_arc"] == 1
    assert a0["network_from_out_degree"] == 1 and a0["network_to_out_degree"] == 2
    assert a0["network_is_junction_end"] == 1 and a0["network_is_dead_end"] == 0
    assert np.isclose(a0["network_segment_sin"] ** 2 + a0["network_segment_cos"] ** 2, 1.0)


def test_no_arcs_gives_nan(cand):
    out = compute_network_features(cand, None)
    assert out["network_to_out_degree"].isna().all()
    assert len(out) == len(cand)


def test_undefined_bearing_gives_nan(cand, trip):
    out = _prep(cand, trip)
    first = out[out["observation_index"] == 0]
    assert first["network_bearing_diff_back_deg"].isna().all()
