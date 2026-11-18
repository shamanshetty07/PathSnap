"""Agent 4: candidate-transition features (vectorised, pairwise).

For every candidate c at observation i, every candidate p at the previous
observation (the closest earlier observation that has candidates) forms a
pair. Pair quantities are computed, then aggregated back to c.

Network distance from p to c is the shortest of:
  * same arc            : along_c - along_p            (must be >= 0)
  * directly connected  : (len_p - along_p) + along_c  (p.to == c.from)
  * one intermediate arc: ... + len(m), m: p.to -> c.from
Anything farther is left NaN ("not reachable within 2 hops"), never faked.
The reference route is never used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._geo import circ_diff, haversine_np
from .schema import validate_candidate_frame

MAX_VEHICLE_SPEED_MPS = 80.0   # ~290 km/h; EDA shows legit 1 Hz steps up to ~72 m; implied speeds above are infeasible
BACKWARD_TOLERANCE_M = 5.0     # GPS noise allowance for "moving backwards" on one arc

TRANSITION_FEATURES = [
    "transition_has_prev",
    "transition_obs_gap",
    "transition_num_prev_candidates",
    "transition_proj_dist_min_m",
    "transition_proj_speed_mps",
    "transition_same_arc_as_prev",
    "transition_connected_1hop",
    "transition_reachable_2hop",
    "transition_best_network_dist_m",
    "transition_net_vs_gps_diff_m",
    "transition_feasible",
    "transition_num_feasible_prev",
    "transition_seg_bearing_change_deg",
]


def _arc_lengths(arcs_df: pd.DataFrame, nodes_df: pd.DataFrame) -> pd.DataFrame:
    size = int(nodes_df.index.max()) + 1
    lon = nodes_df["longitude"].reindex(range(size)).to_numpy(np.float64)
    lat = nodes_df["latitude"].reindex(range(size)).to_numpy(np.float64)
    f = arcs_df["from_node"].to_numpy(np.int64)
    t = arcs_df["to_node"].to_numpy(np.int64)
    ok = (f < size) & (t < size)
    ln = np.full(len(f), np.nan)
    ln[ok] = haversine_np(lat[f[ok]], lon[f[ok]], lat[t[ok]], lon[t[ok]])
    tab = pd.DataFrame({"mid_from": f, "mid_to": t, "mid_len": ln}).dropna()
    return tab.groupby(["mid_from", "mid_to"], as_index=False)["mid_len"].min()


def compute_transition_features(
    df: pd.DataFrame,
    arcs_df: pd.DataFrame | None = None,
    nodes_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    validate_candidate_frame(df)
    out = df.copy()
    n = len(out)
    res = {c: np.full(n, np.nan) for c in TRANSITION_FEATURES}
    res["transition_has_prev"] = np.zeros(n)
    res["transition_num_prev_candidates"] = np.zeros(n)
    res["transition_num_feasible_prev"] = np.zeros(n)
    if n == 0:
        for c, v in res.items():
            out[c] = v
        return out

    cur = pd.DataFrame({
        "row": np.arange(n),
        "obs": out["observation_index"].to_numpy(np.int64),
        "arc": out["arc_id"].to_numpy(np.int64),
        "f": out["from_node_id"].to_numpy(np.int64),
        "t": out["to_node_id"].to_numpy(np.int64),
        "along": out["distance_along_segment_m"].to_numpy(np.float64),
        "len": out["segment_length_m"].to_numpy(np.float64),
        "plat": out["projected_latitude"].to_numpy(np.float64),
        "plon": out["projected_longitude"].to_numpy(np.float64),
        "glat": out["gps_latitude"].to_numpy(np.float64),
        "glon": out["gps_longitude"].to_numpy(np.float64),
        "ts": out["timestamp"].to_numpy(np.float64),
        "sb": out["segment_bearing"].to_numpy(np.float64),
    })
    obs_sorted = np.sort(cur["obs"].unique())
    prev_map = pd.Series(np.r_[np.nan, obs_sorted[:-1]], index=obs_sorted)
    cur["prev_obs"] = cur["obs"].map(prev_map)
    has_prev = cur["prev_obs"].notna()
    res["transition_has_prev"] = has_prev.to_numpy(np.float64)
    res["transition_obs_gap"] = (cur["obs"] - cur["prev_obs"]).to_numpy(np.float64)

    prv = cur.drop(columns=["prev_obs", "row"]).add_prefix("p_").rename(columns={"p_obs": "prev_obs"})
    prv["prev_obs"] = prv["prev_obs"].astype(np.float64)
    pairs = cur[has_prev].merge(prv, on="prev_obs", how="inner")
    if not pairs.empty:
        dt = pairs["ts"] - pairs["p_ts"]
        proj = haversine_np(pairs["p_plat"], pairs["p_plon"], pairs["plat"], pairs["plon"])
        gps = haversine_np(pairs["p_glat"], pairs["p_glon"], pairs["glat"], pairs["glon"])
        rem_p = pairs["p_len"] - pairs["p_along"]

        same = pairs["arc"] == pairs["p_arc"]
        d_same = pairs["along"] - pairs["p_along"]
        d_same = d_same.where(same & (d_same >= -BACKWARD_TOLERANCE_M), np.nan).clip(lower=0.0)
        adj = (pairs["p_t"] == pairs["f"]) & ~same
        d_adj = (rem_p + pairs["along"]).where(adj, np.nan)
        d_two = pd.Series(np.nan, index=pairs.index)
        if arcs_df is not None and nodes_df is not None and not arcs_df.empty and not nodes_df.empty:
            tab = _arc_lengths(arcs_df, nodes_df)
            key = pairs[["p_t", "f"]].rename(columns={"p_t": "mid_from", "f": "mid_to"})
            mid = key.merge(tab, on=["mid_from", "mid_to"], how="left")["mid_len"].to_numpy()
            d_two = pd.Series(rem_p.to_numpy() + mid + pairs["along"].to_numpy(), index=pairs.index)
            d_two = d_two.where(~same & ~adj, np.nan)
        net = pd.concat([d_same, d_adj, d_two], axis=1).min(axis=1, skipna=True)
        pairs["net"] = net
        pairs["proj"] = proj
        pairs["gps"] = gps
        pairs["diff"] = (net - gps).abs()
        with np.errstate(divide="ignore", invalid="ignore"):
            pairs["speed"] = np.where(dt > 0, net / dt, np.nan)
            pairs["pspeed"] = np.where(dt > 0, proj / dt, np.nan)
        pairs["feasible"] = (net.notna() & ((pairs["speed"] <= MAX_VEHICLE_SPEED_MPS) | (dt <= 0))).astype(float)
        pairs["same"] = same.astype(float)
        pairs["conn1"] = (same & d_same.notna() | adj).astype(float)
        pairs["reach"] = net.notna().astype(float)
        pairs["bchg"] = np.abs(circ_diff(pairs["p_sb"], pairs["sb"]))

        g = pairs.groupby("row")
        agg = g.agg(
            n_prev=("proj", "size"), proj_min=("proj", "min"), same=("same", "max"),
            conn1=("conn1", "max"), reach=("reach", "max"), feas=("feasible", "max"),
            n_feas=("feasible", "sum"),
        )
        rows = agg.index.to_numpy()
        res["transition_num_prev_candidates"][rows] = agg["n_prev"].to_numpy()
        res["transition_proj_dist_min_m"][rows] = agg["proj_min"].to_numpy()
        res["transition_same_arc_as_prev"][rows] = agg["same"].to_numpy()
        res["transition_connected_1hop"][rows] = agg["conn1"].to_numpy()
        res["transition_reachable_2hop"][rows] = agg["reach"].to_numpy()
        res["transition_feasible"][rows] = agg["feas"].to_numpy()
        res["transition_num_feasible_prev"][rows] = agg["n_feas"].to_numpy()
        # speed of the projection-nearest previous candidate
        near = pairs.sort_values(["row", "proj"], kind="stable").drop_duplicates("row")
        res["transition_proj_speed_mps"][near["row"].to_numpy()] = near["pspeed"].to_numpy()
        # best reachable pair: smallest |network - GPS displacement|
        best = (pairs[pairs["net"].notna()].sort_values(["row", "diff"], kind="stable")
                .drop_duplicates("row"))
        r = best["row"].to_numpy()
        res["transition_best_network_dist_m"][r] = best["net"].to_numpy()
        res["transition_net_vs_gps_diff_m"][r] = best["diff"].to_numpy()
        res["transition_seg_bearing_change_deg"][r] = best["bchg"].to_numpy()
    for c, v in res.items():
        out[c] = v
    return out


def get_transition_feature_names() -> list[str]:
    return list(TRANSITION_FEATURES)
