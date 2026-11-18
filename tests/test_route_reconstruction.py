"""Basic tests for reference route reconstruction and disconnected transitions.

These tests use small, synthetic in-memory road networks so they do not depend
on the (git-ignored) dataset and run without downloading anything.
"""

import unittest

import pandas as pd

from exploration.data_loader import TripData
from exploration.route_reconstruction import reconstruct_reference_route


def _make_trip_data(route_arc_ids, track_rows=None):
    """Build a minimal TripData from a list of arc IDs."""
    nodes = pd.DataFrame(
        {
            "longitude": [0.0, 1.0, 2.0, 3.0, 5.0, 6.0],
            "latitude": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=[0, 1, 2, 3, 5, 6],
    )
    nodes.index.name = "node_id"
    arcs = pd.DataFrame(
        {
            "from_node": [0, 1, 2, 5],
            "to_node": [1, 2, 3, 6],
        },
        index=[0, 1, 2, 3],
    )
    arcs.index.name = "arc_id"
    track = pd.DataFrame(
        track_rows or [
            {"longitude": 0.0, "latitude": 0.0, "timestamp": 1.0},
            {"longitude": 3.0, "latitude": 0.0, "timestamp": 2.0},
        ]
    )
    return TripData(
        trip_id="TEST",
        split="test",
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=list(route_arc_ids),
        base_dir=".",
    )


class TestRouteReconstruction(unittest.TestCase):
    def test_fully_connected_route(self):
        route = reconstruct_reference_route(_make_trip_data([0, 1, 2]))
        self.assertTrue(route.is_fully_connected)
        self.assertEqual(route.arc_ids, [0, 1, 2])
        self.assertEqual(route.node_sequence, [0, 1, 2, 3])
        self.assertEqual(route.num_coordinates, 4)
        self.assertEqual(route.num_segments, 1)
        self.assertEqual(route.disconnected_transitions, [])

    def test_disconnected_transition(self):
        # Arc 0: 0->1, arc 1: 1->2, arc 3: 5->6 (starts at node 5, not node 2).
        route = reconstruct_reference_route(_make_trip_data([0, 1, 3]))
        self.assertFalse(route.is_fully_connected)
        self.assertEqual(len(route.disconnected_transitions), 1)
        t = route.disconnected_transitions[0]
        self.assertEqual(t.prev_arc_id, 1)
        self.assertEqual(t.next_arc_id, 3)
        self.assertEqual(t.prev_to_node, 2)
        self.assertEqual(t.next_from_node, 5)
        # Two connected segments: [0,1,2] and [5,6].
        self.assertEqual(route.num_segments, 2)
        self.assertEqual(len(route.segments[0]), 3)
        self.assertEqual(len(route.segments[1]), 2)
        self.assertEqual(route.num_coordinates, 5)

    def test_repeated_arc_ids_preserved(self):
        # Arc 0: 0->1 appears twice.
        route = reconstruct_reference_route(_make_trip_data([0, 0]))
        self.assertEqual(route.arc_ids, [0, 0])
        self.assertEqual(route.node_sequence, [0, 1, 0, 1])
        self.assertFalse(route.is_fully_connected)
        self.assertEqual(len(route.disconnected_transitions), 1)

    def test_empty_route(self):
        route = reconstruct_reference_route(_make_trip_data([]))
        self.assertTrue(route.is_fully_connected)
        self.assertEqual(route.arc_ids, [])
        self.assertEqual(route.num_coordinates, 0)
        self.assertEqual(route.num_segments, 0)


if __name__ == "__main__":
    unittest.main()
