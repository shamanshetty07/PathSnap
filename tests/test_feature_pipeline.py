import numpy as np
import pandas as pd
import pytest

from features.feature_pipeline import build_trip_features, run_feature_pipeline
from features.feature_validation import (
    assert_no_group_leakage, check_keys, check_reference_route_leakage, count_non_finite,
    find_redundant_features, make_group_folds,
)
from features.schema import KEY_COLUMNS, all_feature_names, feature_columns
from tests._synth import make_trip


def test_schema_keys_features_unique_and_present():
    trip = make_trip()
    df, _ = build_trip_features(trip)
    names = all_feature_names()
    assert len(names) == len(set(names))
    assert set(names) == set(feature_columns(df))
    assert list(df.columns[:3]) == KEY_COLUMNS
    assert check_keys(df)["ok"]


def test_no_infinite_values():
    df, _ = build_trip_features(make_trip())
    assert count_non_finite(df) == {}


def test_features_independent_of_reference_route():
    """Leakage: changing the route must not change any feature value."""
    a, _ = build_trip_features(make_trip(route=(0, 1, 2, 3)))
    b, _ = build_trip_features(make_trip(route=(7, 6, 5)))
    cols = feature_columns(a)
    pd.testing.assert_frame_equal(a[cols], b[cols])
    assert not a["label_aligned"].equals(b["label_aligned"])


def test_leakage_checker_flags_bad_columns():
    df, _ = build_trip_features(make_trip())
    assert check_reference_route_leakage(df)["forbidden_feature_names"] == []
    df["spatial_route_pos"] = df["label_aligned"]
    assert not check_reference_route_leakage(df)["ok"]


def test_deterministic():
    a, _ = build_trip_features(make_trip())
    b, _ = build_trip_features(make_trip())
    assert a.equals(b)


def test_radius_changes_candidate_count():
    small, _ = build_trip_features(make_trip(), search_radius_m=3.0)
    large, _ = build_trip_features(make_trip(), search_radius_m=50.0)
    assert len(small) < len(large)


def test_group_folds_no_trip_leak():
    trips = np.repeat([f"{i:08d}" for i in range(10)], 7)
    folds = make_group_folds(trips, 5)
    assert len(folds) == 5
    assert_no_group_leakage(trips, folds)
    with pytest.raises(AssertionError):
        assert_no_group_leakage(trips, [(np.arange(len(trips)), np.arange(5))])


def test_redundancy_detection():
    corr = pd.DataFrame([[1, .99], [.99, 1]], columns=["a", "b"], index=["a", "b"])
    assert find_redundant_features(corr)[0]["feature_1"] == "a"


def test_cli_single_trip_real_data(tmp_path):
    from pathlib import Path
    if not Path("Data/student_test/00000052").exists():
        pytest.skip("dataset not available")
    df = run_feature_pipeline("student_test", "00000052", output_dir=tmp_path)
    assert (tmp_path / "student_test/00000052/features.csv").exists()
    assert check_keys(df)["ok"] and count_non_finite(df) == {}
