"""Reference route reconstruction for PathSnap.

Reconstructs the geographical reference route from an ordered list of arc IDs
using the road-network nodes and arcs. Repeated arc IDs are preserved and
consecutive arcs are checked for connectivity; disconnected transitions are
reported but never silently repaired with artificial straight lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .route_validation import (
    DisconnectedTransition,
    RouteValidationReport,
    validate_continuity,
    validate_route_references,
)


@dataclass
class ReconstructedRoute:
    """The reconstructed geographical reference route for a trip.

    All coordinates are ``(latitude, longitude)`` tuples suitable for Folium,
    converted from the dataset's ``(longitude, latitude)`` storage order.
    """

    trip_id: str
    arc_ids: list[int]
    node_sequence: list[int]
    segments: list[list[tuple[float, float]]]
    disconnected_transitions: list[DisconnectedTransition]
    is_fully_connected: bool
    validation: RouteValidationReport
    load_warnings: list[str] = field(default_factory=list)

    @property
    def coordinates(self) -> list[tuple[float, float]]:
        """Flattened list of all reconstructed coordinates in route order."""
        return [coord for segment in self.segments for coord in segment]

    @property
    def num_coordinates(self) -> int:
        return len(self.coordinates)

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    @property
    def start_coordinate(self) -> Optional[tuple[float, float]]:
        if not self.segments or not self.segments[0]:
            return None
        return self.segments[0][0]

    @property
    def end_coordinate(self) -> Optional[tuple[float, float]]:
        if not self.segments or not self.segments[-1]:
            return None
        return self.segments[-1][-1]


def _arc_endpoints(arcs: pd.DataFrame, arc_id: int) -> Optional[tuple[int, int]]:
    """Return ``(from_node, to_node)`` for an arc, or ``None`` if unknown."""
    if arc_id not in arcs.index:
        return None
    return int(arcs.loc[arc_id, "from_node"]), int(arcs.loc[arc_id, "to_node"])


def _node_coord(nodes: pd.DataFrame, node_id: int) -> Optional[tuple[float, float]]:
    """Return ``(latitude, longitude)`` for a node, or ``None`` if unknown."""
    if node_id not in nodes.index:
        return None
    row = nodes.loc[node_id]
    return (float(row["latitude"]), float(row["longitude"]))


def reconstruct_reference_route(trip_data) -> ReconstructedRoute:
    """Build the geographical reference route from a loaded :class:`TripData`.

    The route is constructed in the exact order given by ``.route``. Each arc
    contributes its ``from_node`` (for the first arc) and ``to_node``
    coordinates. Consecutive connected arcs share a node, so a fully connected
    route with ``N`` arcs yields ``N + 1`` coordinates.

    Repeated arc IDs are preserved. Disconnected transitions are detected and
    reported, and the route is split into connected segments so that no
    artificial straight line is drawn across a disconnection.
    """
    nodes = trip_data.nodes
    arcs = trip_data.arcs
    arc_ids = list(trip_data.route)

    validation = validate_route_references(arc_ids, arcs, nodes)
    disconnected_transitions, is_fully_connected = validate_continuity(
        arc_ids, arcs, nodes
    )

    segments: list[list[tuple[float, float]]] = []
    node_sequence: list[int] = []

    if not arc_ids:
        return ReconstructedRoute(
            trip_id=trip_data.trip_id,
            arc_ids=[],
            node_sequence=[],
            segments=[],
            disconnected_transitions=disconnected_transitions,
            is_fully_connected=True,
            validation=validation,
            load_warnings=list(trip_data.load_warnings),
        )

    # Build connected segments. A new segment starts whenever a disconnection
    # is detected, so segments are never joined by artificial lines.
    current_segment: list[tuple[float, float]] = []

    first_endpoints = _arc_endpoints(arcs, arc_ids[0])
    if first_endpoints is None:
        raise ValueError(f"First arc ID {arc_ids[0]} is not present in .arcs")
    first_from_node = first_endpoints[0]
    first_from_coord = _node_coord(nodes, first_from_node)
    if first_from_coord is None:
        raise ValueError(
            f"Start node {first_from_node} of first arc {arc_ids[0]} not in .nodes"
        )
    current_segment.append(first_from_coord)
    node_sequence.append(first_from_node)

    for index, arc_id in enumerate(arc_ids):
        endpoints = _arc_endpoints(arcs, arc_id)
        if endpoints is None:
            # Arc reference is invalid: stop the current segment here and do
            # not fabricate geometry for a missing arc.
            if current_segment:
                segments.append(current_segment)
                current_segment = []
            continue

        from_node, to_node = endpoints
        to_coord = _node_coord(nodes, to_node)
        if to_coord is None:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
            continue

        current_segment.append(to_coord)
        node_sequence.append(to_node)

        # If the next arc does not start where this one ends, close the
        # segment and begin a new one at the next arc's start node.
        if index < len(arc_ids) - 1:
            next_arc_id = arc_ids[index + 1]
            next_endpoints = _arc_endpoints(arcs, next_arc_id)
            if next_endpoints is None:
                segments.append(current_segment)
                current_segment = []
                continue
            next_from_node = next_endpoints[0]
            if to_node != next_from_node:
                segments.append(current_segment)
                next_from_coord = _node_coord(nodes, next_from_node)
                current_segment = (
                    [next_from_coord] if next_from_coord is not None else []
                )
                node_sequence.append(next_from_node)

    if current_segment:
        segments.append(current_segment)

    return ReconstructedRoute(
        trip_id=trip_data.trip_id,
        arc_ids=arc_ids,
        node_sequence=node_sequence,
        segments=segments,
            disconnected_transitions=disconnected_transitions,
            is_fully_connected=is_fully_connected and validation.is_fully_connected,
        validation=validation,
        load_warnings=list(trip_data.load_warnings),
    )
