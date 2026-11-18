"""GPS-observation -> reference-route label alignment.

The reference route is used ONLY here, to build training/evaluation labels.
It is never an input feature.

Why not index matching?  The route is a list of arcs; the track is a list of
1 Hz GPS fixes.  Their lengths differ (e.g. 1315 fixes vs. a few hundred arcs)
and arcs repeat, so observation i does not correspond to route[i].

Method (monotone Viterbi alignment)
-----------------------------------
States at observation i: pairs (candidate c, route position j) with
route[j] == c.arc_id.  Observations with no candidate on the route get no
state and no label (they are reported, not guessed).  The vehicle follows the
route in order, so the route position may not decrease:

    cost(state) = (distance_m / SIGMA_M)^2
    cost(j_prev -> j) = INF                         if j < j_prev
                      = JUMP_PENALTY * max(0, j - j_prev - free)   otherwise
        free = FREE_STEPS * (1 + number of unlabeled observations skipped)
    start cost = JUMP_PENALTY * max(0, j - FREE_STEPS)
    end   cost = JUMP_PENALTY * max(0, (R-1-j) - FREE_STEPS)

Labels:
    label_aligned            1 if the candidate is the aligned (arc, position)
    label_obs_has_label      1 if the observation received an aligned state
    label_on_route_anywhere  naive baseline: arc appears anywhere in the route
                             (kept only to quantify how weak that label is)
Rows of observations with label_obs_has_label == 0 must be excluded from
supervised training and metrics (their label_aligned is 0 by construction).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SIGMA_M = 10.0
JUMP_PENALTY = 0.5
FREE_STEPS = 3


def _positions(route: list[int]) -> dict[int, list[int]]:
    pos: dict[int, list[int]] = {}
    for j, a in enumerate(route):
        pos.setdefault(int(a), []).append(j)
    return pos


def align_labels(feature_df: pd.DataFrame, reference_route: list[int]) -> pd.DataFrame:
    """Return a copy of *feature_df* with the three label_* columns."""
    out = feature_df.copy()
    n = len(out)
    route = [int(a) for a in reference_route]
    R = len(route)
    out["label_aligned"] = 0
    out["label_obs_has_label"] = 0
    out["label_on_route_anywhere"] = out["arc_id"].isin(set(route)).astype(int) if n else 0
    out["label_route_position"] = -1  # diagnostic only; excluded from features by prefix
    if n == 0 or R == 0:
        return out

    pos = _positions(route)
    obs = out["observation_index"].to_numpy(np.int64)
    arc = out["arc_id"].to_numpy(np.int64)
    dist = out["distance_m"].to_numpy(np.float64)
    order = np.argsort(obs, kind="stable")

    # Build per-observation state lists: (row, route_pos, cost)
    layers: list[tuple[int, list[tuple[int, int, float]]]] = []
    start = 0
    sorted_obs = obs[order]
    bounds = np.flatnonzero(np.r_[True, sorted_obs[1:] != sorted_obs[:-1], True])
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        states = []
        for k in order[b0:b1]:
            for j in pos.get(int(arc[k]), ()):
                states.append((int(k), j, (dist[k] / SIGMA_M) ** 2))
        if states:
            layers.append((int(sorted_obs[b0]), states))
    if not layers:
        return out

    # Viterbi
    cost_prev = np.array([s[2] + JUMP_PENALTY * max(0, s[1] - FREE_STEPS) for s in layers[0][1]])
    back: list[np.ndarray] = [np.full(len(layers[0][1]), -1)]
    for li in range(1, len(layers)):
        skipped = layers[li][0] - layers[li - 1][0] - 1  # obs without any on-route state in between
        free = FREE_STEPS * (1 + max(skipped, 0))
        prev_states, cur_states = layers[li - 1][1], layers[li][1]
        pj = np.array([s[1] for s in prev_states])
        cj = np.array([s[1] for s in cur_states])
        step = cj[None, :] - pj[:, None]
        trans = np.where(step < 0, np.inf, JUMP_PENALTY * np.maximum(0, step - free))
        total = cost_prev[:, None] + trans
        arg = total.argmin(axis=0)
        best = total[arg, np.arange(len(cur_states))]
        if not np.isfinite(best).any():          # monotonicity impossible: restart softly
            total = cost_prev[:, None] + JUMP_PENALTY * np.abs(step)
            arg = total.argmin(axis=0)
            best = total[arg, np.arange(len(cur_states))]
        cost_prev = best + np.array([s[2] for s in cur_states])
        back.append(arg)
    end_cost = cost_prev + JUMP_PENALTY * np.maximum(
        0, (R - 1 - np.array([s[1] for s in layers[-1][1]])) - FREE_STEPS)
    k = int(np.argmin(end_cost))
    chosen: list[tuple[int, int]] = []
    for li in range(len(layers) - 1, -1, -1):
        row, j, _ = layers[li][1][k]
        chosen.append((row, j))
        k = int(back[li][k]) if li > 0 else -1
    lab = out["label_aligned"].to_numpy().copy()
    has = out["label_obs_has_label"].to_numpy().copy()
    rp = out["label_route_position"].to_numpy().copy()
    for row, j in chosen:
        lab[row] = 1
        rp[row] = j
        has[obs == obs[row]] = 1
    out["label_aligned"], out["label_obs_has_label"], out["label_route_position"] = lab, has, rp
    return out


def compute_label_statistics(labeled_df: pd.DataFrame) -> dict[str, Any]:
    n = len(labeled_df)
    if n == 0:
        return {"total_rows": 0}
    g = labeled_df.groupby("observation_index")
    n_obs = g.ngroups
    has = g["label_obs_has_label"].max()
    n_on = g["label_on_route_anywhere"].max()
    return {
        "total_rows": n,
        "total_observations": n_obs,
        "labeled_observations": int(has.sum()),
        "labeled_observation_rate": float(has.mean()),
        "observations_with_on_route_candidate": int(n_on.sum()),
        "positive_rows": int(labeled_df["label_aligned"].sum()),
        "naive_on_route_rows": int(labeled_df["label_on_route_anywhere"].sum()),
        "naive_on_route_row_rate": float(labeled_df["label_on_route_anywhere"].mean()),
        "aligned_positive_row_rate": float(labeled_df["label_aligned"].mean()),
    }


def investigate_alignment_quality(
    labeled_df: pd.DataFrame, trip_id: str, num_observations: int | None = None,
    route_length: int | None = None,
) -> dict[str, Any]:
    """Per-trip diagnostics of the alignment (monotonicity, jumps, nearest-rank agreement,
    deviation from naive proportional index matching)."""
    rep: dict[str, Any] = {"trip_id": trip_id, **compute_label_statistics(labeled_df)}
    if labeled_df.empty:
        rep["status"] = "empty"
        return rep
    pos = labeled_df[labeled_df["label_aligned"] == 1].sort_values("observation_index")
    rep["status"] = "ok" if len(pos) else "no_labels"
    if num_observations is not None:
        rep["observation_candidate_miss_rate"] = 1.0 - rep["total_observations"] / num_observations
    if route_length is not None:
        rep["route_length"] = route_length
    if len(pos):
        j = pos["label_route_position"].to_numpy()
        step = np.diff(j)
        rep["monotone_violations"] = int((step < 0).sum())
        rep["large_jumps_gt10"] = int((step > 10).sum())
        rep["max_jump"] = int(step.max()) if len(step) else 0
        rep["aligned_is_nearest_rate"] = float((pos["spatial_dist_rank"] == 0).mean()) \
            if "spatial_dist_rank" in pos else float("nan")
        rep["aligned_mean_dist_m"] = float(pos["distance_m"].mean())
        rep["route_positions_covered_rate"] = (
            float(len(np.unique(j)) / route_length) if route_length else float("nan"))
        if num_observations and route_length:
            expected = pos["observation_index"].to_numpy() / max(num_observations - 1, 1) * (route_length - 1)
            rep["mean_abs_dev_from_index_scaling"] = float(np.abs(j - expected).mean())
    return rep
