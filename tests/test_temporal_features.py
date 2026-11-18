import numpy as np
import pandas as pd

from features.temporal_features import compute_temporal_features, get_temporal_feature_names


def test_first_observation_is_nan_not_zero(cand, trip):
    out = compute_temporal_features(cand, track=trip.track)
    first = out[out["observation_index"] == 0]
    assert first["temporal_time_diff_s"].isna().all()
    assert first["temporal_gps_speed_mps"].isna().all()
    assert (first["temporal_time_since_first_s"] == 0).all()


def test_speed_bearing_eastward(cand, trip):
    out = compute_temporal_features(cand, track=trip.track)
    mid = out[out["observation_index"] == 10]
    assert np.allclose(mid["temporal_time_diff_s"], 1.0)
    assert np.allclose(mid["temporal_gps_speed_mps"], 0.0001 * 111195 * np.cos(np.radians(49)), rtol=0.01)
    assert np.allclose(mid["temporal_gps_bearing_deg"], 90.0, atol=0.5)
    assert abs(mid["temporal_gps_bearing_change_deg"].iloc[0]) < 0.5
    assert mid["temporal_movement_consistency"].iloc[0] > 0.99


def test_circular_bearing_change():
    from features._geo import circ_diff
    assert circ_diff(350.0, 10.0) == 20.0
    assert circ_diff(10.0, 350.0) == -20.0


def test_large_gap_and_acceleration_suppressed(cand, trip):
    t = trip.track.copy()
    t.loc[20:, "timestamp"] += 30.0
    out = compute_temporal_features(cand, track=t)
    gap = out[out["observation_index"] == 20]
    assert (gap["temporal_is_large_time_gap"] == 1).all()
    assert gap["temporal_acceleration_mps2"].isna().all()


def test_stationary_bearing_undefined():
    from tests._synth import make_trip
    from exploration.candidate_generation.candidate_generator import generate_candidates
    tr = make_trip(n_obs=10, step_deg=0.0)
    res = generate_candidates(tr)
    df = pd.DataFrame([c.to_dict() for c in res.candidates])
    out = compute_temporal_features(df, track=tr.track)
    later = out[out["observation_index"] > 1]
    assert later["temporal_gps_bearing_deg"].isna().all()
    assert (later["temporal_is_stationary"] == 1).all()


def test_works_without_track_and_names(cand):
    out = compute_temporal_features(cand)
    assert set(get_temporal_feature_names()) <= set(out.columns)
    assert len(out) == len(cand)
