import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _synth import make_trip  # noqa: E402

from exploration.candidate_generation.candidate_generator import generate_candidates  # noqa: E402


@pytest.fixture(scope="module")
def trip():
    return make_trip()


@pytest.fixture(scope="module")
def cand(trip):
    res = generate_candidates(trip, search_radius_m=50.0, max_candidates=5)
    df = pd.DataFrame([c.to_dict() for c in res.candidates])
    df["trip_id"] = trip.trip_id
    return df
