"""Agent 1: spatial features for each (GPS observation, candidate arc) pair.

Fully vectorised. All distances in metres. Uses only GPS + road network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._geo import haversine_np, safe_div
from .schema import validate_candidate_frame

SPATIAL_FEATURES = [
    "spatial_dist_to_segment_m",      # GPS -> segment (point-to-segment) distance
    "spatial_dist_rank",              # rank within observation (0 = nearest)
    "spatial_dist_excess_m",          # distance minus nearest candidate's distance
    "spatial_num_candidates",         # candidates available at this observation
    "spatial_along_m",                # distance along segment from start node
    "spatial_remaining_m",            # remaining distance to end node
    "spatial_rel_position",           # along / length in [0,1]
    "spatial_dist_to_start_node_m",   # GPS -> from-node
    "spatial_dist_to_end_node_m",     # GPS -> to-node
    "spatial_segment_length_m",
    "spatial_proj_at_endpoint",       # projection clamped to a segment end
    "spatial_offset_east_m",          # GPS -> projected point (east component)
    "spatial_offset_north_m",         # GPS -> projected point (north component)
]


def compute_spatial_features(df: pd.DataFrame, nodes_df: pd.DataFrame | None = None) -> pd.DataFrame:
    validate_candidate_frame(df)
    out = df.copy()
    d = out["distance_m"].to_numpy(np.float64)
    seg = out["segment_length_m"].to_numpy(np.float64)
    along = out["distance_along_segment_m"].to_numpy(np.float64)
    glat = out["gps_latitude"].to_numpy(np.float64)
    glon = out["gps_longitude"].to_numpy(np.float64)
    plat = out["projected_latitude"].to_numpy(np.float64)
    plon = out["projected_longitude"].to_numpy(np.float64)
    obs = out["observation_index"]

    out["spatial_dist_to_segment_m"] = d
    out["spatial_dist_rank"] = (
        pd.Series(d, index=out.index).groupby(obs).rank(method="first").to_numpy() - 1.0
    )
    out["spatial_dist_excess_m"] = d - pd.Series(d, index=out.index).groupby(obs).transform("min").to_numpy()
    out["spatial_num_candidates"] = obs.groupby(obs).transform("size").to_numpy(np.float64)
    out["spatial_along_m"] = along
    out["spatial_remaining_m"] = seg - along
    out["spatial_rel_position"] = safe_div(along, seg)
    out["spatial_segment_length_m"] = seg
    out["spatial_proj_at_endpoint"] = ((along <= 1e-6) | (seg - along <= 1e-6)).astype(np.float64)

    n = len(out)
    d_start = np.full(n, np.nan)
    d_end = np.full(n, np.nan)
    if nodes_df is not None and not nodes_df.empty:
        nlon = nodes_df["longitude"].reindex(range(int(nodes_df.index.max()) + 1)).to_numpy(np.float64)
        nlat = nodes_df["latitude"].reindex(range(int(nodes_df.index.max()) + 1)).to_numpy(np.float64)
        fn = out["from_node_id"].to_numpy(np.int64)
        tn = out["to_node_id"].to_numpy(np.int64)
        ok_f, ok_t = fn < len(nlon), tn < len(nlon)
        d_start[ok_f] = haversine_np(glat[ok_f], glon[ok_f], nlat[fn[ok_f]], nlon[fn[ok_f]])
        d_end[ok_t] = haversine_np(glat[ok_t], glon[ok_t], nlat[tn[ok_t]], nlon[tn[ok_t]])
    out["spatial_dist_to_start_node_m"] = d_start
    out["spatial_dist_to_end_node_m"] = d_end

    kx = np.pi / 180.0 * 6371000.0
    out["spatial_offset_east_m"] = (plon - glon) * kx * np.cos(np.radians(glat))
    out["spatial_offset_north_m"] = (plat - glat) * kx
    return out


def get_spatial_feature_names() -> list[str]:
    return list(SPATIAL_FEATURES)
