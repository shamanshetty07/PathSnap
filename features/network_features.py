"""Agent 3: road-network and direction features (vectorised).

Directed arcs are respected: connectivity is by node-id matching only, never
by geometric proximity. Uses GPS movement bearings from the temporal group, so
``compute_temporal_features`` must run first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._geo import circ_diff
from .schema import validate_candidate_frame

NETWORK_FEATURES = [
    "network_segment_sin",              # orientation encoded circularly
    "network_segment_cos",
    "network_bearing_diff_back_deg",    # |segment bearing - backward GPS bearing|, [0,180]
    "network_bearing_diff_fwd_deg",     # |segment bearing - forward GPS bearing|, [0,180]
    "network_directional_compat",       # mean cos(diff) over defined GPS bearings, [-1,1]
    "network_is_reverse_direction",     # 1 if every defined GPS bearing is >90 deg off
    "network_from_in_degree",
    "network_from_out_degree",
    "network_to_in_degree",
    "network_to_out_degree",
    "network_has_reverse_arc",          # an arc to->from exists (two-way road)
    "network_is_dead_end",              # to-node has no outgoing arc
    "network_is_junction_end",          # to-node has >=2 outgoing arcs (a real choice)
]


def compute_network_features(
    df: pd.DataFrame,
    arcs_df: pd.DataFrame | None = None,
    nodes_df: pd.DataFrame | None = None,  # kept for API symmetry
) -> pd.DataFrame:
    validate_candidate_frame(df)
    out = df.copy()
    n = len(out)
    seg_b = out["segment_bearing"].to_numpy(np.float64)
    out["network_segment_sin"] = np.sin(np.radians(seg_b))
    out["network_segment_cos"] = np.cos(np.radians(seg_b))

    back = out["temporal_gps_bearing_deg"].to_numpy(np.float64) if "temporal_gps_bearing_deg" in out else np.full(n, np.nan)
    fwd = out["temporal_fwd_bearing_deg"].to_numpy(np.float64) if "temporal_fwd_bearing_deg" in out else np.full(n, np.nan)
    d_back = np.abs(circ_diff(seg_b, back))
    d_fwd = np.abs(circ_diff(seg_b, fwd))
    out["network_bearing_diff_back_deg"] = d_back
    out["network_bearing_diff_fwd_deg"] = d_fwd
    cos_stack = np.vstack([np.cos(np.radians(d_back)), np.cos(np.radians(d_fwd))])
    both_nan = np.isnan(cos_stack).all(axis=0)
    cnt = (~np.isnan(cos_stack)).sum(axis=0)
    compat = np.where(cnt > 0, np.nansum(cos_stack, axis=0) / np.maximum(cnt, 1), np.nan)
    out["network_directional_compat"] = compat
    big = np.vstack([np.where(np.isnan(d_back), True, d_back > 90.0),
                     np.where(np.isnan(d_fwd), True, d_fwd > 90.0)]).all(axis=0)
    out["network_is_reverse_direction"] = np.where(both_nan, np.nan, big.astype(np.float64))

    fn = out["from_node_id"].to_numpy(np.int64)
    tn = out["to_node_id"].to_numpy(np.int64)
    nan = np.full(n, np.nan)
    cols = {k: nan.copy() for k in ("fi", "fo", "ti", "to", "rev")}
    if arcs_df is not None and not arcs_df.empty:
        a_f = arcs_df["from_node"].to_numpy(np.int64)
        a_t = arcs_df["to_node"].to_numpy(np.int64)
        size = int(max(a_f.max(), a_t.max(), fn.max(), tn.max())) + 1
        out_deg = np.bincount(a_f, minlength=size).astype(np.float64)
        in_deg = np.bincount(a_t, minlength=size).astype(np.float64)
        cols["fi"], cols["fo"] = in_deg[fn], out_deg[fn]
        cols["ti"], cols["to"] = in_deg[tn], out_deg[tn]
        edge_codes = np.unique(a_f * size + a_t)
        cols["rev"] = np.isin(tn * size + fn, edge_codes).astype(np.float64)
    out["network_from_in_degree"] = cols["fi"]
    out["network_from_out_degree"] = cols["fo"]
    out["network_to_in_degree"] = cols["ti"]
    out["network_to_out_degree"] = cols["to"]
    out["network_has_reverse_arc"] = cols["rev"]
    out["network_is_dead_end"] = np.where(np.isnan(cols["to"]), np.nan, (cols["to"] == 0).astype(np.float64))
    out["network_is_junction_end"] = np.where(np.isnan(cols["to"]), np.nan, (cols["to"] >= 2).astype(np.float64))
    return out


def get_network_feature_names() -> list[str]:
    return list(NETWORK_FEATURES)
