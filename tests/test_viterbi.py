import numpy as np
import pandas as pd
import pytest

from models.hmm import RoadNetwork, HiddenMarkovMapMatcher


@pytest.fixture
def matcher():
    nodes = pd.DataFrame({
        "node_id": [0, 1, 2, 3],
        "longitude": [77.0, 77.001, 77.002, 77.003],
        "latitude": [12.0, 12.0, 12.0, 12.0],
    })

    arcs = pd.DataFrame({
        "from_node_id": [0, 1, 2],
        "to_node_id": [1, 2, 3],
    })

    network = RoadNetwork(arcs, nodes)

    return HiddenMarkovMapMatcher(network)


def make_candidates(observation_index, candidates):
    rows = []

    for arc_id, probability in candidates:
        rows.append({
            "observation_index": observation_index,
            "arc_id": arc_id,
            "from_node_id": arc_id,
            "to_node_id": arc_id + 1,
            "distance_along_segment_m": 10.0,
            "segment_length_m": 100.0,
            "gps_lon": 77.0005 + observation_index * 0.001,
            "gps_lat": 12.0,
            "timestamp": float(observation_index),
            "emission_probability": probability,
        })

    return pd.DataFrame(rows)


def test_viterbi_returns_one_candidate_per_observation(matcher):

    groups = [
        make_candidates(0, [(0, 0.9), (1, 0.1)]),
        make_candidates(1, [(1, 0.9), (2, 0.1)]),
        make_candidates(2, [(2, 0.9), (1, 0.1)]),
    ]

    result = matcher.viterbi_decode(groups)

    assert len(result) == 3
    assert result["decoded"].all()
    assert result["observation_index"].tolist() == [0, 1, 2]


def test_viterbi_selects_high_probability_path(matcher):

    groups = [
        make_candidates(0, [(0, 0.95), (1, 0.05)]),
        make_candidates(1, [(1, 0.95), (2, 0.05)]),
    ]

    result = matcher.viterbi_decode(groups)

    assert result.iloc[0]["arc_id"] == 0
    assert result.iloc[1]["arc_id"] == 1


def test_viterbi_rejects_disconnected_transition(matcher):

    groups = [
        make_candidates(0, [(0, 0.9)]),
        make_candidates(1, [(2, 0.9)]),
    ]

    with pytest.raises(ValueError):
        matcher.viterbi_decode(groups)


def test_viterbi_handles_empty_input(matcher):

    result = matcher.viterbi_decode([])

    assert result.empty