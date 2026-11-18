"""Vectorised geometry helpers shared by the feature modules (NumPy only)."""

from __future__ import annotations

import numpy as np

from exploration.candidate_generation.geometry import EARTH_RADIUS_M


def haversine_np(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (vectorised, NaN-propagating)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2, dtype=np.float64) - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_np(lat1, lon1, lat2, lon2):
    """Initial compass bearing (deg clockwise from north, [0, 360))."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2, dtype=np.float64) - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def circ_diff(b_from, b_to):
    """Signed circular difference b_to - b_from in [-180, 180)."""
    return (np.asarray(b_to, dtype=np.float64) - b_from + 180.0) % 360.0 - 180.0


def safe_div(num, den):
    """Elementwise division returning NaN where the denominator is 0/NaN."""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.full(np.broadcast(num, den).shape, np.nan)
    ok = np.isfinite(den) & (den != 0.0)
    np.divide(num, den, out=out, where=ok)
    return out
