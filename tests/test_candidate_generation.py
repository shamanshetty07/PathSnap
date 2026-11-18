"""Tests for candidate generation: geometry, spatial index, and generator."""

import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from exploration.data_loader import TripData
from exploration.candidate_generation.candidate_generator import (
    Candidate,
    generate_candidates,
)
from exploration.candidate_generation.candidate_validation import (
    evaluate_reference_coverage,
)
from exploration.candidate_generation.geometry import (
    LocalProjection,
    gps_bearing_degrees,
    haversine_distance_m,
    project_point_on_segment,
    segment_bearing_meters,
)
from exploration.candidate_generation.spatial_index import RoadNetworkIndex


def _make_trip(track_rows, node_coords, arcs, route, trip_id="SYN", split="test"):
    """Build a TripData from synthetic data.

    track_rows: list of (lon, lat, ts)
    node_coords: dict {node_id: (lon, lat)}
    arcs: list of (from_node, to_node)
    route: list of arc ids
    """
    track = pd.DataFrame(track_rows, columns=["longitude", "latitude", "timestamp"])
    nodes = pd.DataFrame(
        [(lon, lat) for lon, lat in node_coords.values()],
        index=list(node_coords.keys()),
        columns=["longitude", "latitude"],
    )
    nodes.index.name = "node_id"
    arcs_df = pd.DataFrame(arcs, columns=["from_node", "to_node"])
    arcs_df.index.name = "arc_id"
    return TripData(
        trip_id=trip_id,
        split=split,
        track=track,
        nodes=nodes,
        arcs=arcs_df,
        route=list(route),
        base_dir=".",
    )


class TestGeometry(unittest.TestCase):
    def test_haversine_same_point(self):
        self.assertAlmostEqual(haversine_distance_m(0, 0, 0, 0), 0.0, places=6)

    def test_haversine_one_degree_lat(self):
        d = haversine_distance_m(0.0, 0.0, 1.0, 0.0)
        self.assertAlmostEqual(d, 111195.0, delta=500.0)

    def test_point_on_segment(self):
        # Segment from (0,0) to (100,0) in projected space; point at (50, 0).
        t, dist, proj = project_point_on_segment(50.0, 0.0, 0.0, 0.0, 100.0, 0.0)
        self.assertAlmostEqual(t, 0.5)
        self.assertAlmostEqual(dist, 0.0)
        self.assertAlmostEqual(proj[0], 50.0)
        self.assertAlmostEqual(proj[1], 0.0)

    def test_point_perpendicular_to_segment(self):
        # Segment from (0,0) to (100,0); point at (50, 10).
        t, dist, proj = project_point_on_segment(50.0, 10.0, 0.0, 0.0, 100.0, 0.0)
        self.assertAlmostEqual(t, 0.5)
        self.assertAlmostEqual(dist, 10.0)
        self.assertAlmostEqual(proj[0], 50.0)
        self.assertAlmostEqual(proj[1], 0.0)

    def test_point_before_segment_start(self):
        # Segment from (0,0) to (100,0); point at (-10, 0) -> clamped to start.
        t, dist, proj = project_point_on_segment(-10.0, 0.0, 0.0, 0.0, 100.0, 0.0)
        self.assertAlmostEqual(t, 0.0)
        self.assertAlmostEqual(dist, 10.0)
        self.assertAlmostEqual(proj[0], 0.0)

    def test_point_after_segment_end(self):
        # Segment from (0,0) to (100,0); point at (110, 0) -> clamped to end.
        t, dist, proj = project_point_on_segment(110.0, 0.0, 0.0, 0.0, 100.0, 0.0)
        self.assertAlmostEqual(t, 1.0)
        self.assertAlmostEqual(dist, 10.0)
        self.assertAlmostEqual(proj[0], 100.0)

    def test_zero_length_segment(self):
        t, dist, proj = project_point_on_segment(5.0, 5.0, 0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(t, 0.0)
        self.assertAlmostEqual(dist, math.hypot(5.0, 5.0))
        self.assertAlmostEqual(proj[0], 0.0)
        self.assertAlmostEqual(proj[1], 0.0)

    def test_segment_bearing_north(self):
        # Segment going north in projected space (x=0, y increases).
        b = segment_bearing_meters(0.0, 0.0, 0.0, 100.0)
        self.assertAlmostEqual(b, 0.0)

    def test_segment_bearing_east(self):
        # Segment going east.
        b = segment_bearing_meters(0.0, 0.0, 100.0, 0.0)
        self.assertAlmostEqual(b, 90.0)

    def test_local_projection_roundtrip(self):
        proj = LocalProjection(lon0=8.5, lat0=49.8)
        lon, lat = 8.6, 49.9
        x, y = proj.to_meters(lon, lat)
        lon2, lat2 = proj.from_meters(x, y)
        self.assertAlmostEqual(lon, lon2, places=6)
        self.assertAlmostEqual(lat, lat2, places=6)

    def test_gps_bearing_north(self):
        b = gps_bearing_degrees(0.0, 0.0, 1.0, 0.0)
        self.assertAlmostEqual(b, 0.0, places=1)

    def test_gps_bearing_east(self):
        b = gps_bearing_degrees(0.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(b, 90.0, places=1)


class TestSpatialIndex(unittest.TestCase):
    def test_build_and_query(self):
        trip = _make_trip(
            [],
            {0: (0.0, 0.0), 1: (0.001, 0.0), 2: (0.002, 0.0)},
            [(0, 1), (1, 2)],
            [],
        )
        index = RoadNetworkIndex(trip)
        index.build()
        self.assertEqual(len(index), 2)

        # Query near the first segment.
        results = index.query(0.0005, 0.0, radius_m=100.0)
        self.assertGreaterEqual(len(results), 1)

    def test_empty_network(self):
        trip = _make_trip([], {}, [], [])
        index = RoadNetworkIndex(trip)
        index.build()
        self.assertEqual(len(index), 0)
        results = index.query(0.0, 0.0, radius_m=100.0)
        self.assertEqual(results, [])

    def test_invalid_node_reference_skipped(self):
        # Arc references node 99 which doesn't exist.
        trip = _make_trip(
            [],
            {0: (0.0, 0.0), 1: (0.001, 0.0)},
            [(0, 1), (0, 99)],
            [],
        )
        index = RoadNetworkIndex(trip)
        index.build()
        # Only the valid arc should be indexed.
        self.assertEqual(len(index), 1)
        self.assertEqual(index.segments[0].arc_id, 0)


class TestCandidateGenerator(unittest.TestCase):
    def test_gps_point_on_segment(self):
        # One arc from node 0 to node 1, GPS point at the midpoint.
        trip = _make_trip(
            [(0.0005, 0.0, 1.0)],  # midpoint of arc 0
            {0: (0.0, 0.0), 1: (0.001, 0.0)},
            [(0, 1)],
            [0],
        )
        result = generate_candidates(trip, search_radius_m=100.0, max_candidates=5)
        self.assertGreater(result.num_candidates, 0)
        c = result.candidates[0]
        self.assertEqual(c.arc_id, 0)
        self.assertAlmostEqual(c.distance_m, 0.0, delta=5.0)  # approx on segment

    def test_gps_point_outside_radius(self):
        # GPS point far from any segment.
        trip = _make_trip(
            [(0.0, 0.0, 1.0)],
            {0: (1.0, 1.0), 1: (1.001, 1.0)},  # ~100 km away
            [(0, 1)],
            [],
        )
        result = generate_candidates(trip, search_radius_m=50.0, max_candidates=5)
        self.assertEqual(result.num_candidates, 0)
        self.assertEqual(result.miss_rate, 1.0)

    def test_multiple_candidates_ordered_by_distance(self):
        # Two segments near the GPS point.
        trip = _make_trip(
            [(0.0005, 0.0, 1.0)],
            {0: (0.0, 0.0), 1: (0.001, 0.0), 2: (0.0005, 0.0005)},
            [(0, 1), (0, 2)],
            [],
        )
        result = generate_candidates(trip, search_radius_m=200.0, max_candidates=5)
        self.assertEqual(result.num_candidates, 2)
        self.assertLessEqual(
            result.candidates[0].distance_m, result.candidates[1].distance_m
        )

    def test_max_candidates_cap(self):
        # Many segments near one point, but capped at max_candidates.
        node_coords = {0: (0.0, 0.0)}
        for i in range(1, 11):
            node_coords[i] = (0.0001 * i, 0.0)
        arcs = [(0, i) for i in range(1, 11)]
        trip = _make_trip(
            [(0.00005, 0.0, 1.0)],
            node_coords,
            arcs,
            [],
        )
        result = generate_candidates(trip, search_radius_m=500.0, max_candidates=3)
        counts: dict[int, int] = {}
        for c in result.candidates:
            counts[c.observation_index] = counts.get(c.observation_index, 0) + 1
        self.assertEqual(counts.get(0, 0), 3)

    def test_empty_track(self):
        trip = _make_trip(
            [],
            {0: (0.0, 0.0), 1: (0.001, 0.0)},
            [(0, 1)],
            [],
        )
        result = generate_candidates(trip, search_radius_m=100.0, max_candidates=5)
        self.assertEqual(result.num_observations, 0)
        self.assertEqual(result.num_candidates, 0)

    def test_candidate_fields(self):
        trip = _make_trip(
            [(0.0005, 0.0, 1.0)],
            {0: (0.0, 0.0), 1: (0.001, 0.0)},
            [(0, 1)],
            [0],
        )
        result = generate_candidates(trip, search_radius_m=100.0, max_candidates=5)
        self.assertGreater(result.num_candidates, 0)
        c = result.candidates[0]
        # All required fields present and typed.
        self.assertIsInstance(c.observation_index, int)
        self.assertIsInstance(c.timestamp, float)
        self.assertIsInstance(c.gps_longitude, float)
        self.assertIsInstance(c.gps_latitude, float)
        self.assertIsInstance(c.arc_id, int)
        self.assertIsInstance(c.from_node_id, int)
        self.assertIsInstance(c.to_node_id, int)
        self.assertIsInstance(c.distance_m, float)
        self.assertIsInstance(c.projected_longitude, float)
        self.assertIsInstance(c.projected_latitude, float)
        self.assertIsInstance(c.segment_length_m, float)
        self.assertIsInstance(c.distance_along_segment_m, float)
        self.assertIsInstance(c.segment_bearing, float)
        self.assertIsInstance(c.gps_bearing, float)

    def test_gps_bearing_first_observation_zero(self):
        trip = _make_trip(
            [(0.0, 0.0, 1.0), (0.0001, 0.0, 2.0)],
            {0: (0.0, 0.0), 1: (0.001, 0.0)},
            [(0, 1)],
            [],
        )
        result = generate_candidates(trip, search_radius_m=500.0, max_candidates=5)
        # First observation bearing should be 0 (no previous observation).
        first_cands = [c for c in result.candidates if c.observation_index == 0]
        if first_cands:
            self.assertEqual(first_cands[0].gps_bearing, 0.0)


class TestCandidateValidation(unittest.TestCase):
    def test_evaluate_reference_coverage(self):
        trip = _make_trip(
            [(0.0005, 0.0, 1.0)],
            {0: (0.0, 0.0), 1: (0.001, 0.0)},
            [(0, 1)],
            [0],  # reference route uses arc 0
        )
        result = generate_candidates(trip, search_radius_m=100.0, max_candidates=5)
        report = evaluate_reference_coverage(result, result.trip_id and [0])
        self.assertIsInstance(report.reference_arc_coverage, float)
        self.assertIsInstance(report.observation_reference_coverage, float)

    def test_evaluate_no_reference_overlap(self):
        trip = _make_trip(
            [(0.0, 0.0, 1.0)],
            {0: (1.0, 1.0), 1: (1.001, 1.0)},
            [(0, 1)],
            [0],  # reference arc is far away
        )
        result = generate_candidates(trip, search_radius_m=50.0, max_candidates=5)
        report = evaluate_reference_coverage(result, [0])
        self.assertEqual(report.reference_arc_coverage, 0.0)
        self.assertEqual(report.observation_reference_coverage, 0.0)


if __name__ == "__main__":
    unittest.main()
