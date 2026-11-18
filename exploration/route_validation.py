"""Route continuity and reference validation for PathSnap.

This module validates that an ordered list of reference arc IDs forms a
continuous path through the road network and reports any disconnected
transitions. It does not attempt to repair invalid routes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class DisconnectedTransition:
    """A single break in route continuity between two consecutive arcs."""

    transition_index: int  # index of the second arc in the route (arc i -> arc i+1)
    prev_arc_id: int
    next_arc_id: int
    prev_to_node: int  # end node of the previous arc
    next_from_node: int  # start node of the next arc
    prev_to_coord: tuple[float, float]  # (latitude, longitude)
    next_from_coord: tuple[float, float]  # (latitude, longitude)


@dataclass
class RouteValidationReport:
    """Outcome of validating a reconstructed reference route."""

    is_fully_connected: bool = True
    disconnected_transitions: list[DisconnectedTransition] = field(
        default_factory=list
    )
    total_arcs: int = 0
    valid_arc_references: bool = True
    invalid_arc_ids: list[int] = field(default_factory=list)
    missing_node_ids: list[int] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


def _node_coord(
    nodes: pd.DataFrame, node_id: int
) -> Optional[tuple[float, float]]:
    """Return ``(latitude, longitude)`` for a node, or ``None`` if unknown."""
    if node_id not in nodes.index:
        return None
    row = nodes.loc[node_id]
    # Dataset stores (longitude, latitude); Folium needs (latitude, longitude).
    return (float(row["latitude"]), float(row["longitude"]))


def validate_route_references(
    route_arc_ids: list[int], arcs: pd.DataFrame, nodes: pd.DataFrame
) -> RouteValidationReport:
    """Check that every arc ID and referenced node ID exists in the network."""
    report = RouteValidationReport(total_arcs=len(route_arc_ids))

    for arc_id in route_arc_ids:
        if arc_id not in arcs.index:
            report.invalid_arc_ids.append(int(arc_id))
            report.valid_arc_references = False
            continue
        from_node = int(arcs.loc[arc_id, "from_node"])
        to_node = int(arcs.loc[arc_id, "to_node"])
        if from_node not in nodes.index:
            report.missing_node_ids.append(from_node)
        if to_node not in nodes.index:
            report.missing_node_ids.append(to_node)

    report.is_fully_connected = not bool(
        report.invalid_arc_ids or report.missing_node_ids
    )

    if report.invalid_arc_ids:
        report.messages.append(
            f"{len(report.invalid_arc_ids)} arc ID(s) not found in .arcs: "
            f"{report.invalid_arc_ids[:10]}{'...' if len(report.invalid_arc_ids) > 10 else ''}"
        )
    if report.missing_node_ids:
        report.messages.append(
            f"{len(set(report.missing_node_ids))} node ID(s) referenced by arcs "
            f"not found in .nodes"
        )
    return report


def validate_continuity(
    route_arc_ids: list[int],
    arcs: pd.DataFrame,
    nodes: pd.DataFrame,
) -> tuple[list[DisconnectedTransition], bool]:
    """Detect disconnected transitions between consecutive reference arcs.

    Two consecutive arcs connect when the ``to_node`` of the first equals the
    ``from_node`` of the second. Any mismatch is reported as a disconnected
    transition and is *not* repaired.
    """
    transitions: list[DisconnectedTransition] = []

    for index in range(len(route_arc_ids) - 1):
        prev_arc_id = route_arc_ids[index]
        next_arc_id = route_arc_ids[index + 1]

        if prev_arc_id not in arcs.index or next_arc_id not in arcs.index:
            continue

        prev_to_node = int(arcs.loc[prev_arc_id, "to_node"])
        next_from_node = int(arcs.loc[next_arc_id, "from_node"])

        if prev_to_node == next_from_node:
            continue

        prev_to_coord = _node_coord(nodes, prev_to_node)
        next_from_coord = _node_coord(nodes, next_from_node)
        transitions.append(
            DisconnectedTransition(
                transition_index=index + 1,
                prev_arc_id=int(prev_arc_id),
                next_arc_id=int(next_arc_id),
                prev_to_node=prev_to_node,
                next_from_node=next_from_node,
                prev_to_coord=prev_to_coord,
                next_from_coord=next_from_coord,
            )
        )

    return transitions, len(transitions) == 0
