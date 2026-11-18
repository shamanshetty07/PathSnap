"""Reference route analysis for PathSnap EDA.

Reuses the existing route reconstruction and validation modules to explore
the ``.route`` files:

* reference-route length per trip
* unique vs repeated arc occurrences
* route continuity and disconnected transitions
* invalid arc references and missing node references
* reconstructed route distance (Haversine, within connected segments only)

The GPS observations and reference-route arcs are treated as independent
sequences; no correspondence between their indices is assumed. Disconnected
routes are reported, never silently repaired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .gps_analysis import haversine_distance_m
from .route_reconstruction import reconstruct_reference_route


@dataclass
class RouteStats:
    """Result of analysing a single trip's reference route."""

    trip_id: str
    split: str
    route_length: int
    unique_arcs: int
    repeated_arc_occurrences: int
    repeated_arc_ids: list[int]
    is_fully_connected: bool
    disconnected_transitions: int
    invalid_arc_references: int
    invalid_arc_ids: list[int]
    missing_node_references: int
    missing_node_ids: list[int]
    route_distance_m: Optional[float]
    num_reconstructed_coordinates: int
    num_segments: int
    load_warnings: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "split": self.split,
            "trip_id": self.trip_id,
            "route_length": self.route_length,
            "unique_arcs": self.unique_arcs,
            "repeated_arc_occurrences": self.repeated_arc_occurrences,
            "is_fully_connected": self.is_fully_connected,
            "disconnected_transitions": self.disconnected_transitions,
            "invalid_arc_references": self.invalid_arc_references,
            "missing_node_references": self.missing_node_references,
            "route_distance_m": self.route_distance_m,
            "num_reconstructed_coordinates": self.num_reconstructed_coordinates,
            "num_connected_segments": self.num_segments,
        }


def analyze_reference_route(trip_data) -> RouteStats:
    """Analyse the reference route of a single :class:`TripData`."""
    route = reconstruct_reference_route(trip_data)
    arc_ids = list(route.arc_ids)

    route_length = len(arc_ids)
    unique_arc_ids = set(arc_ids)
    unique_arcs = len(unique_arc_ids)
    repeated_arc_occurrences = route_length - unique_arcs

    # Arc IDs appearing more than once.
    if route_length > 0:
        counts = {}
        for a in arc_ids:
            counts[a] = counts.get(a, 0) + 1
        repeated_arc_ids = sorted([a for a, c in counts.items() if c > 1])
    else:
        repeated_arc_ids = []

    disconnected_transitions = len(route.disconnected_transitions)
    invalid_arc_references = len(route.validation.invalid_arc_ids)
    invalid_arc_ids = list(route.validation.invalid_arc_ids)
    missing_node_references = len(set(route.validation.missing_node_ids))
    missing_node_ids = sorted(set(route.validation.missing_node_ids))

    # Reconstructed route distance: Haversine along each connected segment,
    # never across a disconnected transition.
    route_distance_m = None
    if route.num_segments > 0:
        total = 0.0
        for segment in route.segments:
            if len(segment) >= 2:
                seg_lat = np.array([c[0] for c in segment], dtype=np.float64)
                seg_lon = np.array([c[1] for c in segment], dtype=np.float64)
                total += float(
                    np.sum(
                        haversine_distance_m(
                            seg_lat[:-1], seg_lon[:-1], seg_lat[1:], seg_lon[1:]
                        )
                    )
                )
        route_distance_m = total

    warnings = list(route.load_warnings)
    if invalid_arc_references:
        warnings.append(
            f"{invalid_arc_references} route arc ID(s) not present in .arcs"
        )
    if missing_node_references:
        warnings.append(
            f"{missing_node_references} unique node ID(s) referenced by route "
            "arcs absent from .nodes"
        )
    if disconnected_transitions:
        warnings.append(
            f"{disconnected_transitions} disconnected route transition(s) detected"
        )

    return RouteStats(
        trip_id=trip_data.trip_id,
        split=trip_data.split,
        route_length=route_length,
        unique_arcs=unique_arcs,
        repeated_arc_occurrences=repeated_arc_occurrences,
        repeated_arc_ids=repeated_arc_ids,
        is_fully_connected=bool(route.is_fully_connected),
        disconnected_transitions=disconnected_transitions,
        invalid_arc_references=invalid_arc_references,
        invalid_arc_ids=invalid_arc_ids,
        missing_node_references=missing_node_references,
        missing_node_ids=missing_node_ids,
        route_distance_m=route_distance_m,
        num_reconstructed_coordinates=route.num_coordinates,
        num_segments=route.num_segments,
        load_warnings=warnings,
    )
