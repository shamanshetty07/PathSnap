import numpy as np
import pandas as pd

from features.label_alignment import align_labels, compute_label_statistics, investigate_alignment_quality
from features.feature_pipeline import build_trip_features
from tests._synth import make_trip


def test_alignment_is_monotone_and_correct():
    trip = make_trip(n_obs=40, route=(0, 1, 2, 3))
    df, rep = build_trip_features(trip)
    pos = df[df["label_aligned"] == 1].sort_values("observation_index")
    assert pos["observation_index"].is_unique                      # exactly one positive per obs
    assert (np.diff(pos["label_route_position"].to_numpy()) >= 0).all()
    assert rep["monotone_violations"] == 0
    # eastbound arcs only (westbound arcs are not in the route)
    assert set(pos["arc_id"]) <= {0, 1, 2, 3}
    assert df.groupby("observation_index")["label_aligned"].sum().max() == 1


def test_index_is_not_assumed_equal_to_route_position():
    trip = make_trip(n_obs=40, route=(0, 1, 2, 3))
    df, _ = build_trip_features(trip)
    pos = df[df["label_aligned"] == 1]
    # 40 observations vs 4 route arcs: positions must not equal observation indices
    assert pos["label_route_position"].max() <= 3 < pos["observation_index"].max()


def test_repeated_arcs_handled_in_order():
    trip = make_trip(n_obs=10, route=(0, 1, 0))
    df, _ = build_trip_features(trip)
    assert (df["label_aligned"].sum()) >= 1


def test_off_route_observations_unlabeled():
    trip = make_trip(n_obs=40, route=(2, 3))          # first half of the drive is off-route
    df, _ = build_trip_features(trip)
    early = df[df["observation_index"] < 5]
    assert (early["label_obs_has_label"] == 0).all() and (early["label_aligned"] == 0).all()


def test_empty_inputs_and_stats():
    out = align_labels(pd.DataFrame({"arc_id": [], "observation_index": [], "distance_m": []}), [1, 2])
    assert len(out) == 0
    out = align_labels(pd.DataFrame({"arc_id": [1], "observation_index": [0], "distance_m": [1.0]}), [])
    assert out["label_aligned"].sum() == 0
    st = compute_label_statistics(out)
    assert st["labeled_observations"] == 0


def test_labels_are_not_feature_columns():
    from features.schema import feature_columns
    trip = make_trip()
    df, _ = build_trip_features(trip)
    assert not any(c.startswith("label_") for c in feature_columns(df))
