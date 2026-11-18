"""Spatial analysis for PathSnap EDA.

Relates GPS observations to the road network using a KD-tree spatial index:

* nearest road-network node for every GPS observation
* nearest-node distance distribution (Haversine metres)
* fraction of observations within 5 / 10 / 20 / 50 m of a node

Nearest-node distance is only a preliminary diagnostic: it measures proximity
to a *node*, not to a road segment, and is not equivalent to map matching.

Dataset coordinates are longitude, latitude; distances are computed with the
Haversine formula. Because trips span different regions, no single projected
coordinate system is assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .gps_analysis import haversine_distance_m

# Nearest-node distance thresholds (metres) for "close to network" buckets.
DISTANCE_BUCKETS_M = (5.0, 10.0, 20.0, 50.0)
# Number of KD-tree candidates examined per GPS point before taking the
# minimum Haversine distance. Keeps the search local and fast.
KD_K_CANDIDATES = 10


@dataclass
class SpatialStats:
    """Result of the spatial analysis for a single trip."""

    trip_id: str
    split: str
    num_gps_points: int
    num_network_nodes: int
    mean_nearest_node_distance_m: float
    median_nearest_node_distance_m: float
    min_nearest_node_distance_m: float
    max_nearest_node_distance_m: float
    pct_within_5m: float
    pct_within_10m: float
    pct_within_20m: float
    pct_within_50m: float
    load_warnings: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "split": self.split,
            "trip_id": self.trip_id,
            "num_gps_points": self.num_gps_points,
            "num_network_nodes": self.num_network_nodes,
            "mean_nearest_node_distance_m": self.mean_nearest_node_distance_m,
            "median_nearest_node_distance_m": self.median_nearest_node_distance_m,
            "min_nearest_node_distance_m": self.min_nearest_node_distance_m,
            "max_nearest_node_distance_m": self.max_nearest_node_distance_m,
            "pct_within_5m": self.pct_within_5m,
            "pct_within_10m": self.pct_within_10m,
            "pct_within_20m": self.pct_within_20m,
            "pct_within_50m": self.pct_within_50m,
        }


@dataclass
class SpatialResult:
    stats: SpatialStats
    nearest_distances_m: np.ndarray


def _haversine_to_candidates(
    gps_lat: np.ndarray,
    gps_lon: np.ndarray,
    cand_lat: np.ndarray,
    cand_lon: np.ndarray,
) -> np.ndarray:
    """Haversine distance from each GPS point to its candidate nodes.

    ``gps_*`` have shape ``(n,)``; ``cand_*`` have shape ``(n, k)``. Returns
    shape ``(n, k)``.
    """
    n, k = cand_lat.shape
    gps_lat_r = np.radians(np.repeat(gps_lat, k)).reshape(n, k)
    gps_lon_r = np.radians(np.repeat(gps_lon, k)).reshape(n, k)
    cand_lat_r = np.radians(cand_lat)
    cand_lon_r = np.radians(cand_lon)

    dlat = cand_lat_r - gps_lat_r
    dlon = cand_lon_r - gps_lon_r
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(gps_lat_r) * np.cos(cand_lat_r) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return 6371000.0 * c


def analyze_spatial(trip_data, max_distances_for_pool: int = 200000) -> SpatialResult:
    """Analyse GPS-to-node proximity for a single :class:`TripData`.

    ``max_distances_for_pool`` caps how many nearest-node distances are
    retained for the pooled global distribution, to bound memory.
    """
    nodes = trip_data.nodes
    track = trip_data.track
    warnings = list(trip_data.load_warnings)

    num_gps = int(len(track))
    num_nodes = int(len(nodes))

    if num_gps == 0 or num_nodes == 0:
        empty = np.array([], dtype=np.float64)
        stats = SpatialStats(
            trip_id=trip_data.trip_id,
            split=trip_data.split,
            num_gps_points=num_gps,
            num_network_nodes=num_nodes,
            mean_nearest_node_distance_m=0.0,
            median_nearest_node_distance_m=0.0,
            min_nearest_node_distance_m=0.0,
            max_nearest_node_distance_m=0.0,
            pct_within_5m=0.0,
            pct_within_10m=0.0,
            pct_within_20m=0.0,
            pct_within_50m=0.0,
            load_warnings=warnings,
        )
        return SpatialResult(stats=stats, nearest_distances_m=empty)

    node_lat = nodes["latitude"].to_numpy(dtype=np.float64)
    node_lon = nodes["longitude"].to_numpy(dtype=np.float64)
    gps_lat = track["latitude"].to_numpy(dtype=np.float64)
    gps_lon = track["longitude"].to_numpy(dtype=np.float64)

    # KD-tree on (lat, lon) degrees to find candidate nearest nodes quickly.
    tree = cKDTree(np.column_stack([node_lat, node_lon]))
    k = min(KD_K_CANDIDATES, num_nodes)
    _, cand_idx = tree.query(np.column_stack([gps_lat, gps_lon]), k=k)
    cand_idx = np.atleast_2d(cand_idx)  # shape (n, k)

    cand_lat = node_lat[cand_idx]
    cand_lon = node_lon[cand_idx]

    dist_matrix = _haversine_to_candidates(gps_lat, gps_lon, cand_lat, cand_lon)
    nearest = np.min(dist_matrix, axis=1)

    # Cap the pooled distances to bound memory for the global distribution.
    if nearest.size > max_distances_for_pool:
        nearest = nearest[np.random.default_rng(0).choice(
            nearest.size, max_distances_for_pool, replace=False
        )]

    def pct_within(threshold: float) -> float:
        return float((nearest <= threshold).mean() * 100.0)

    stats = SpatialStats(
        trip_id=trip_data.trip_id,
        split=trip_data.split,
        num_gps_points=num_gps,
        num_network_nodes=num_nodes,
        mean_nearest_node_distance_m=float(np.mean(nearest)),
        median_nearest_node_distance_m=float(np.median(nearest)),
        min_nearest_node_distance_m=float(np.min(nearest)),
        max_nearest_node_distance_m=float(np.max(nearest)),
        pct_within_5m=pct_within(5.0),
        pct_within_10m=pct_within(10.0),
        pct_within_20m=pct_within(20.0),
        pct_within_50m=pct_within(50.0),
        load_warnings=warnings,
    )
    return SpatialResult(stats=stats, nearest_distances_m=nearest)
