"""Tests for EDA statistical calculations using synthetic in-memory data."""

from pathlib import Path

import numpy as np
import pandas as pd
import unittest

from exploration.data_loader import TripData
from exploration.gps_analysis import analyze_gps, haversine_distance_m
from exploration.network_analysis import analyze_network
from exploration.reference_route_analysis import analyze_reference_route
from exploration.spatial_analysis import analyze_spatial


def _build_trip(track_rows, node_coords, arcs, route):
    """Build a TripData.

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
        trip_id="SYN", split="test", track=track, nodes=nodes, arcs=arcs_df,
        route=list(route), base_dir=".",
    )


class TestHaversine(unittest.TestCase):
    def test_same_point_zero(self):
        self.assertAlmostEqual(haversine_distance_m(0, 0, 0, 0), 0.0, places=6)

    def test_known_distance(self):
        # ~111 km for 1 degree of latitude at the equator.
        d = haversine_distance_m(0.0, 0.0, 1.0, 0.0)
        self.assertAlmostEqual(d, 111195.0, delta=500.0)


class TestGpsAnalysis(unittest.TestCase):
    def test_regular_track(self):
        # Two points 1 degree apart in latitude, 1 second apart.
        trip = _build_trip(
            [(0.0, 0.0, 1.0), (0.0, 1.0, 2.0)],
            {0: (0.0, 0.0), 1: (0.0, 1.0)},
            [(0, 1)],
            [0],
        )
        stats, _ = analyze_gps(trip, Path("."))
        self.assertEqual(stats.num_observations, 2)
        self.assertAlmostEqual(stats.pct_one_second_apart, 100.0)
        self.assertAlmostEqual(stats.mean_interval_s, 1.0)
        self.assertEqual(stats.gaps_gt_1s, 0)
        self.assertAlmostEqual(stats.mean_distance_m, 111195.0, delta=500.0)
        self.assertEqual(stats.non_increasing_timestamps, 0)

    def test_non_increasing_and_duplicates(self):
        trip = _build_trip(
            [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.5)],
            {0: (0.0, 0.0)},
            [],
            [],
        )
        stats, _ = analyze_gps(trip, Path("."))
        self.assertEqual(stats.num_observations, 3)
        self.assertEqual(stats.duplicate_coordinates, 3)
        # intervals: 0, -0.5 -> both non-increasing
        self.assertEqual(stats.non_increasing_timestamps, 2)


class TestNetworkAnalysis(unittest.TestCase):
    def test_simple_network(self):
        # Nodes 0,1,2. Arcs: 0->1, 1->2, 2->0 (cycle). Node 3 isolated.
        trip = _build_trip(
            [],
            {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0), 3: (3.0, 0.0)},
            [(0, 1), (1, 2), (2, 0)],
            [],
        )
        stats, degrees = analyze_network(trip)
        self.assertEqual(stats.num_nodes, 4)
        self.assertEqual(stats.num_arcs, 3)
        self.assertEqual(stats.unique_nodes_referenced, 3)
        self.assertEqual(stats.isolated_nodes, 1)
        self.assertEqual(stats.connected_components, 1)  # single cycle component
        self.assertEqual(stats.num_self_loops, 0)
        # Reciprocity: none of (1,0),(2,1),(0,2) present -> directed.
        self.assertEqual(stats.network_orientation, "directed")

    def test_self_loop_and_duplicate(self):
        trip = _build_trip(
            [],
            {0: (0.0, 0.0), 1: (1.0, 0.0)},
            [(0, 0), (0, 1), (0, 1)],
            [],
        )
        stats, _ = analyze_network(trip)
        self.assertEqual(stats.num_self_loops, 1)
        self.assertEqual(stats.duplicate_directed_arcs, 1)
        self.assertEqual(stats.num_duplicate_arc_pairs, 1)


class TestReferenceRouteAnalysis(unittest.TestCase):
    def test_connected_route(self):
        # Nodes 0-1-2-3, arcs 0:0->1, 1:1->2, 2:2->3. Route [0,1,2].
        trip = _build_trip(
            [],
            {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0), 3: (3.0, 0.0)},
            [(0, 1), (1, 2), (2, 3)],
            [0, 1, 2],
        )
        stats = analyze_reference_route(trip)
        self.assertEqual(stats.route_length, 3)
        self.assertEqual(stats.unique_arcs, 3)
        self.assertEqual(stats.repeated_arc_occurrences, 0)
        self.assertTrue(stats.is_fully_connected)
        self.assertEqual(stats.disconnected_transitions, 0)
        self.assertEqual(stats.num_reconstructed_coordinates, 4)

    def test_disconnected_route(self):
        # Arcs 0:0->1, 1:1->2, 2:5->6 (starts at 5, disconnected from node 2).
        trip = _build_trip(
            [],
            {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0), 5: (5.0, 0.0), 6: (6.0, 0.0)},
            [(0, 1), (1, 2), (5, 6)],
            [0, 1, 2],
        )
        stats = analyze_reference_route(trip)
        self.assertFalse(stats.is_fully_connected)
        self.assertEqual(stats.disconnected_transitions, 1)
        self.assertEqual(stats.num_segments, 2)
        self.assertEqual(stats.num_reconstructed_coordinates, 5)

    def test_repeated_arc(self):
        # Arc 0:0->1 twice. Route [0,0] disconnected (1->0 mismatch).
        trip = _build_trip(
            [],
            {0: (0.0, 0.0), 1: (1.0, 0.0)},
            [(0, 1)],
            [0, 0],
        )
        stats = analyze_reference_route(trip)
        self.assertEqual(stats.route_length, 2)
        self.assertEqual(stats.unique_arcs, 1)
        self.assertEqual(stats.repeated_arc_occurrences, 1)
        self.assertEqual(stats.repeated_arc_ids, [0])
        self.assertFalse(stats.is_fully_connected)
        self.assertEqual(stats.disconnected_transitions, 1)


class TestSpatialAnalysis(unittest.TestCase):
    def test_nearest_node(self):
        trip = _build_trip(
            [(0.0, 0.0, 1.0)],
            {0: (0.0, 0.0), 1: (10.0, 0.0)},
            [],
            [],
        )
        result = analyze_spatial(trip)
        self.assertEqual(result.stats.num_gps_points, 1)
        self.assertEqual(result.stats.num_network_nodes, 2)
        self.assertAlmostEqual(result.stats.mean_nearest_node_distance_m, 0.0, places=6)
        self.assertEqual(result.stats.pct_within_5m, 100.0)


if __name__ == "__main__":
    unittest.main()
