"""Agent 2: temporal / movement features, computed per GPS observation and
broadcast to every candidate of that observation.

If the full ``track`` is given, movement is computed on every GPS fix (so an
observation without candidates still counts as the predecessor). Otherwise the
observations present in *df* are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._geo import bearing_np, circ_diff, haversine_np
from .schema import validate_candidate_frame

LARGE_GAP_S = 10.0       # EDA: >10 s gaps are rare and break kinematics
STATIONARY_M = 2.0       # below this displacement a bearing is noise
STATIONARY_SPEED = 0.5   # m/s

TEMPORAL_FEATURES = [
    "temporal_time_diff_s",
    "temporal_gps_displacement_m",
    "temporal_gps_speed_mps",
    "temporal_gps_bearing_deg",         # backward movement bearing (prev -> this)
    "temporal_fwd_bearing_deg",         # forward movement bearing (this -> next)
    "temporal_gps_bearing_change_deg",  # signed circular change vs previous bearing
    "temporal_acceleration_mps2",
    "temporal_time_since_first_s",
    "temporal_is_large_time_gap",
    "temporal_is_stationary",
    "temporal_movement_consistency",    # (1+cos(bearing change))/2
]


def _observation_table(obs: pd.DataFrame) -> pd.DataFrame:
    """obs: columns observation_index, timestamp, lat, lon sorted by index."""
    t = obs["timestamp"].to_numpy(np.float64)
    lat = obs["lat"].to_numpy(np.float64)
    lon = obs["lon"].to_numpy(np.float64)
    n = len(obs)
    nan = np.full(n, np.nan)
    dt, disp, speed, brg, fbrg = nan.copy(), nan.copy(), nan.copy(), nan.copy(), nan.copy()
    if n > 1:
        dt[1:] = t[1:] - t[:-1]
        disp[1:] = haversine_np(lat[:-1], lon[:-1], lat[1:], lon[1:])
        b = bearing_np(lat[:-1], lon[:-1], lat[1:], lon[1:])
        moving = disp[1:] >= STATIONARY_M
        brg[1:] = np.where(moving, b, np.nan)
        fbrg[:-1] = brg[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            speed[1:] = np.where(dt[1:] > 0, disp[1:] / dt[1:], np.nan)
    change = np.full(n, np.nan)
    if n > 1:
        change[1:] = circ_diff(brg[:-1], brg[1:])
    large = np.where(np.isfinite(dt), dt > LARGE_GAP_S, False)
    accel = np.full(n, np.nan)
    if n > 2:
        ok = np.isfinite(speed[1:]) & np.isfinite(speed[:-1]) & (dt[1:] > 0) & ~large[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            accel[1:] = np.where(ok, (speed[1:] - speed[:-1]) / dt[1:], np.nan)
    return pd.DataFrame({
        "observation_index": obs["observation_index"].to_numpy(),
        "temporal_time_diff_s": dt,
        "temporal_gps_displacement_m": disp,
        "temporal_gps_speed_mps": speed,
        "temporal_gps_bearing_deg": brg,
        "temporal_fwd_bearing_deg": fbrg,
        "temporal_gps_bearing_change_deg": change,
        "temporal_acceleration_mps2": accel,
        "temporal_time_since_first_s": t - t[0],
        "temporal_is_large_time_gap": large.astype(np.float64),
        "temporal_is_stationary": np.where(np.isfinite(speed), speed < STATIONARY_SPEED, False).astype(np.float64),
        "temporal_movement_consistency": (1.0 + np.cos(np.radians(change))) / 2.0,
    })


def compute_temporal_features(df: pd.DataFrame, track: pd.DataFrame | None = None) -> pd.DataFrame:
    validate_candidate_frame(df)
    if df.empty:
        out = df.copy()
        for c in TEMPORAL_FEATURES:
            out[c] = np.array([], dtype=np.float64)
        return out
    if track is not None:
        obs = pd.DataFrame({
            "observation_index": np.arange(len(track)),
            "timestamp": track["timestamp"].to_numpy(np.float64),
            "lat": track["latitude"].to_numpy(np.float64),
            "lon": track["longitude"].to_numpy(np.float64),
        })
    else:
        obs = (df.drop_duplicates("observation_index").sort_values("observation_index")
                 .rename(columns={"gps_latitude": "lat", "gps_longitude": "lon"})
                 [["observation_index", "timestamp", "lat", "lon"]])
    table = _observation_table(obs.reset_index(drop=True))
    out = df.drop(columns=[c for c in TEMPORAL_FEATURES if c in df.columns]).merge(
        table, on="observation_index", how="left", sort=False)
    out.index = df.index
    return out


def get_temporal_feature_names() -> list[str]:
    return list(TEMPORAL_FEATURES)
