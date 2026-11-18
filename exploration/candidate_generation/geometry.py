"""Geometric utilities for candidate generation.

Provides:

* Local equirectangular projection from (longitude, latitude) to metres,
  centred per trip so that Euclidean geometry in the projected plane is a good
  local approximation.
* Haversine distance for physical (great-circle) distance calculations.
* Point-to-segment distance with projection of the GPS point onto the road
  segment.
* Bearing calculations (segment bearing and GPS movement bearing).

Dataset coordinates are stored as (longitude, latitude). All internal
projected coordinates are (x, y) in metres.
"""

from __future__ import annotations

import math

EARTH_RADIUS_M = 6371000.0
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


class LocalProjection:
    """Local equirectangular projection centred on a reference point.

    Converts (longitude, latitude) to (x, y) metres and back. Accurate for
    the small geographical extents of a single trip. The projection is built
    once per trip from the mean node location.
    """

    def __init__(self, lon0: float, lat0: float):
        self.lon0 = lon0
        self.lat0 = lat0
        self.lat0_r = math.radians(lat0)
        self._cos_lat0 = math.cos(self.lat0_r)
        self._scale_x = EARTH_RADIUS_M * DEG_TO_RAD * self._cos_lat0
        self._scale_y = EARTH_RADIUS_M * DEG_TO_RAD

    def to_meters(self, longitude: float, latitude: float) -> tuple[float, float]:
        x = (longitude - self.lon0) * self._scale_x
        y = (latitude - self.lat0) * self._scale_y
        return x, y

    def from_meters(self, x: float, y: float) -> tuple[float, float]:
        longitude = self.lon0 + x / self._scale_x
        latitude = self.lat0 + y / self._scale_y
        return longitude, latitude


def project_point_on_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> tuple[float, float, tuple[float, float]]:
    """Project point P onto segment AB in projected (metres) space.

    Returns ``(t, distance_m, (proj_x, proj_y))`` where ``t`` is the
    fractional position along the segment (clamped to [0, 1]), ``distance_m``
    is the perpendicular/clamped distance from P to the segment, and
    ``(proj_x, proj_y)`` is the projected point.

    Zero-length segments (A == B) are handled: the projection is A, t is 0,
    and the distance is the distance from P to A.
    """
    abx = bx - ax
    aby = by - ay
    ab_len_sq = abx * abx + aby * aby

    if ab_len_sq == 0.0:
        # Zero-length segment: projection is the start point.
        proj_x, proj_y = ax, ay
        dist = math.hypot(px - ax, py - ay)
        return 0.0, dist, (proj_x, proj_y)

    apx = px - ax
    apy = py - ay
    t = (apx * abx + apy * aby) / ab_len_sq
    t_clamped = max(0.0, min(1.0, t))

    proj_x = ax + t_clamped * abx
    proj_y = ay + t_clamped * aby
    dist = math.hypot(px - proj_x, py - proj_y)
    return t_clamped, dist, (proj_x, proj_y)


def segment_bearing_meters(ax: float, ay: float, bx: float, by: float) -> float:
    """Bearing of segment A->B in degrees clockwise from north (projected space)."""
    dx = bx - ax
    dy = by - ay
    # In projected space x=east, y=north. Bearing from north = atan2(east, north).
    bearing_rad = math.atan2(dx, dy)
    bearing_deg = (bearing_rad * RAD_TO_DEG + 360.0) % 360.0
    return bearing_deg


def gps_bearing_degrees(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Initial bearing from point 1 to point 2 in degrees clockwise from north."""
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlon_r = math.radians(lon2 - lon1)

    y = math.sin(dlon_r) * math.cos(lat2_r)
    x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(
        lat2_r
    ) * math.cos(dlon_r)
    bearing_rad = math.atan2(y, x)
    return (bearing_rad * RAD_TO_DEG + 360.0) % 360.0
