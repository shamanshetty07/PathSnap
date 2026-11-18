"""Spatial index for road-network segments.

Builds a KD-tree on segment midpoints in a local projected (metres) coordinate
system so that GPS observations can be queried for nearby road segments without
comparing every observation against every segment.

The index stores, for each arc:

* arc ID
* from_node_id, to_node_id
* projected endpoints (ax, ay, bx, by) in metres
* projected midpoint (mx, my) in metres
* segment length in metres

Querying returns arc IDs whose midpoint lies within an expanded search radius;
the exact point-to-segment distance is then computed for each candidate and
filtered against the true search radius.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from ..data_loader import TripData
from .geometry import LocalProjection, haversine_distance_m


@dataclass
class IndexedSegment:
    """A road segment indexed for spatial search."""

    arc_id: int
    from_node_id: int
    to_node_id: int
    ax: float  # projected start x (metres)
    ay: float  # projected start y (metres)
    bx: float  # projected end x (metres)
    by: float  # projected end y (metres)
    mx: float  # projected midpoint x (metres)
    my: float  # projected midpoint y (metres)
    length_m: float


class RoadNetworkIndex:
    """Spatial index over road-network arcs for a single trip."""

    def __init__(self, trip_data: TripData):
        self.trip_data = trip_data
        self.nodes = trip_data.nodes
        self.arcs = trip_data.arcs
        self.projection: Optional[LocalProjection] = None
        self.segments: list[IndexedSegment] = []
        self._tree: Optional[cKDTree] = None
        self._midpoints: Optional[np.ndarray] = None
        self._max_half_length_m: float = 0.0
        self._built = False

    def build(self) -> None:
        """Build the KD-tree from the trip's nodes and arcs."""
        nodes = self.nodes
        arcs = self.arcs

        if nodes.empty or arcs.empty:
            self._built = True
            return

        # Centre the projection on the mean node location for best local accuracy.
        lon0 = float(nodes["longitude"].mean())
        lat0 = float(nodes["latitude"].mean())
        self.projection = LocalProjection(lon0, lat0)

        # Project all node coordinates once.
        node_coords: dict[int, tuple[float, float]] = {}
        for node_id, row in nodes.iterrows():
            nid = int(node_id)
            x, y = self.projection.to_meters(
                float(row["longitude"]), float(row["latitude"])
            )
            node_coords[nid] = (x, y)

        segments: list[IndexedSegment] = []
        max_half_length = 0.0

        for arc_id, row in arcs.iterrows():
            aid = int(arc_id)
            from_node = int(row["from_node"])
            to_node = int(row["to_node"])

            # Skip arcs referencing nodes not present in the node table.
            if from_node not in node_coords or to_node not in node_coords:
                continue

            ax, ay = node_coords[from_node]
            bx, by = node_coords[to_node]
            mx = (ax + bx) / 2.0
            my = (ay + by) / 2.0
            length_m = float(np.hypot(bx - ax, by - ay))

            segments.append(
                IndexedSegment(
                    arc_id=aid,
                    from_node_id=from_node,
                    to_node_id=to_node,
                    ax=ax,
                    ay=ay,
                    bx=bx,
                    by=by,
                    mx=mx,
                    my=my,
                    length_m=length_m,
                )
            )
            max_half_length = max(max_half_length, length_m / 2.0)

        self.segments = segments
        if segments:
            self._midpoints = np.array(
                [[s.mx, s.my] for s in segments], dtype=np.float64
            )
            self._tree = cKDTree(self._midpoints)
        self._max_half_length_m = max_half_length
        self._built = True

    def query(
        self, longitude: float, latitude: float, radius_m: float
    ) -> list[IndexedSegment]:
        """Return indexed segments whose midpoint is within the expanded radius.

        The search radius is expanded by the maximum segment half-length so
        that a segment whose midpoint is slightly outside ``radius_m`` but
        whose endpoint is within ``radius_m`` of the query point is not missed.
        Exact point-to-segment filtering is performed by the caller.
        """
        if not self._built:
            raise RuntimeError("RoadNetworkIndex.build() must be called before query()")
        if self._tree is None or len(self.segments) == 0:
            return []

        if self.projection is None:
            return []

        qx, qy = self.projection.to_meters(longitude, latitude)

        # Expand radius to account for segment length.
        expanded_radius = radius_m + self._max_half_length_m
        indices = self._tree.query_ball_point([qx, qy], r=expanded_radius)

        return [self.segments[i] for i in indices]

    def __len__(self) -> int:
        return len(self.segments)
