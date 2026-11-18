"""Label alignment for PathSnap map matching.

Aligns GPS observation candidates with the reference route to produce training
labels.  The reference route is used **only** for label construction and is
never exposed as an input feature.

Labelling strategy (spatial proximity):
    For each GPS observation's candidates, check whether the candidate's
    ``arc_id`` appears anywhere in the reference route.  If it does the
    candidate is labelled ``is_on_reference_route = 1``, otherwise ``0``.

This is a *necessary-but-not-sufficient* label: being on the reference route
does not guarantee it is the correct match for *this* observation (the vehicle
might have been on the same arc at a different time), but it is a solid,
deterministic, and defensible starting point.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Core labelling
# ---------------------------------------------------------------------------

def align_labels(
    feature_df: pd.DataFrame,
    reference_route: list[int],
) -> pd.DataFrame:
    """Add ``is_on_reference_route`` label column to a feature DataFrame.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature DataFrame with an ``arc_id`` column.
    reference_route : list[int]
        Ordered reference arc IDs from the ``.route`` file.

    Returns
    -------
    pd.DataFrame
        Copy of *feature_df* with an ``is_on_reference_route`` column
        (int, 0 or 1).
    """
    result = feature_df.copy()
    route_set = set(reference_route)
    result["is_on_reference_route"] = result["arc_id"].apply(
        lambda aid: 1 if int(aid) in route_set else 0
    )
    return result


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_label_statistics(labeled_df: pd.DataFrame) -> dict[str, Any]:
    """Compute label-distribution statistics.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        DataFrame with ``observation_index`` and ``is_on_reference_route``
        columns.

    Returns
    -------
    dict
        Keys: total_rows, positive_count, negative_count, positive_rate,
        observations_with_positive_label,
        observations_without_positive_label, miss_rate.
    """
    total_rows = len(labeled_df)
    positive_count = int((labeled_df["is_on_reference_route"] == 1).sum())
    negative_count = total_rows - positive_count
    positive_rate = positive_count / total_rows if total_rows > 0 else 0.0

    # Per-observation statistics
    obs_max_label = labeled_df.groupby("observation_index")[
        "is_on_reference_route"
    ].max()
    observations_with_positive = int((obs_max_label == 1).sum())
    observations_without_positive = int((obs_max_label == 0).sum())
    total_observations = len(obs_max_label)
    miss_rate = (
        observations_without_positive / total_observations
        if total_observations > 0
        else 0.0
    )

    return {
        "total_rows": total_rows,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": positive_rate,
        "observations_with_positive_label": observations_with_positive,
        "observations_without_positive_label": observations_without_positive,
        "miss_rate": miss_rate,
    }


# ---------------------------------------------------------------------------
# Quality investigation
# ---------------------------------------------------------------------------

def investigate_alignment_quality(
    labeled_df: pd.DataFrame,
    trip_id: str,
) -> dict[str, Any]:
    """Investigate alignment quality for a single trip.

    Checks:
    - How many observations have at least one positive label.
    - Consecutive positive labels on the same arc (consistency).
    - Gaps where no positive label exists among candidates.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        Labeled feature DataFrame for a single trip.
    trip_id : str
        Trip identifier (for reporting).

    Returns
    -------
    dict
        Detailed alignment-quality report.
    """
    report: dict[str, Any] = {"trip_id": trip_id}

    if labeled_df.empty:
        report["status"] = "empty"
        return report

    obs_groups = labeled_df.groupby("observation_index")

    # 1. Observations with at least one positive candidate
    obs_has_positive = obs_groups["is_on_reference_route"].max()
    n_obs = len(obs_has_positive)
    n_with_positive = int((obs_has_positive == 1).sum())
    n_without_positive = n_obs - n_with_positive

    report["total_observations"] = n_obs
    report["observations_with_positive"] = n_with_positive
    report["observations_without_positive"] = n_without_positive
    report["positive_observation_rate"] = (
        n_with_positive / n_obs if n_obs > 0 else 0.0
    )

    # 2. Consecutive positive labels on the same arc (consistency)
    # For each observation pick the positive-label arc(s) and check if they
    # match the previous observation's positive arc(s).
    sorted_obs = sorted(obs_has_positive.index)
    positive_arcs_per_obs: dict[int, set[int]] = {}
    for oi in sorted_obs:
        grp = labeled_df[labeled_df["observation_index"] == oi]
        pos_arcs = set(
            grp.loc[grp["is_on_reference_route"] == 1, "arc_id"].astype(int)
        )
        positive_arcs_per_obs[oi] = pos_arcs

    consecutive_same_arc = 0
    consecutive_different_arc = 0
    consecutive_total = 0
    for i in range(1, len(sorted_obs)):
        prev_oi = sorted_obs[i - 1]
        curr_oi = sorted_obs[i]
        prev_arcs = positive_arcs_per_obs[prev_oi]
        curr_arcs = positive_arcs_per_obs[curr_oi]
        if prev_arcs and curr_arcs:
            consecutive_total += 1
            if prev_arcs & curr_arcs:
                consecutive_same_arc += 1
            else:
                consecutive_different_arc += 1

    report["consecutive_pairs_total"] = consecutive_total
    report["consecutive_same_arc"] = consecutive_same_arc
    report["consecutive_different_arc"] = consecutive_different_arc

    # 3. Gaps: runs of consecutive observations with no positive label
    gap_lengths: list[int] = []
    current_gap = 0
    for oi in sorted_obs:
        if not positive_arcs_per_obs[oi]:
            current_gap += 1
        else:
            if current_gap > 0:
                gap_lengths.append(current_gap)
            current_gap = 0
    if current_gap > 0:
        gap_lengths.append(current_gap)

    report["num_gaps"] = len(gap_lengths)
    report["max_gap_length"] = max(gap_lengths) if gap_lengths else 0
    report["mean_gap_length"] = (
        sum(gap_lengths) / len(gap_lengths) if gap_lengths else 0.0
    )
    report["total_gap_observations"] = sum(gap_lengths)
    report["status"] = "ok"

    return report
