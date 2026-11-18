"""Feature validation: missing/inf checks, statistics, correlations,
redundancy, leakage checks, group-aware folds and the markdown report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import KEY_COLUMNS, LABEL_COLUMNS, MISSING_REASONS, feature_columns


def _numeric_features(df: pd.DataFrame) -> list[str]:
    return [c for c in feature_columns(df) if pd.api.types.is_numeric_dtype(df[c])]


def check_keys(df: pd.DataFrame) -> dict[str, Any]:
    """Keys present, no nulls, (trip_id, observation_index, arc_id) unique."""
    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        return {"ok": False, "missing_key_columns": missing}
    nulls = int(df[KEY_COLUMNS].isna().sum().sum())
    dups = int(df.duplicated(KEY_COLUMNS).sum())
    return {"ok": nulls == 0 and dups == 0, "null_keys": nulls, "duplicate_keys": dups}


def count_non_finite(df: pd.DataFrame) -> dict[str, int]:
    """Number of +/-inf values per feature (must be zero)."""
    cols = _numeric_features(df)
    arr = df[cols].to_numpy(dtype=np.float64, na_value=np.nan)
    inf = np.isinf(arr).sum(axis=0)
    return {c: int(k) for c, k in zip(cols, inf) if k}


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    cols = feature_columns(df)
    rows = []
    for c in cols:
        na = int(df[c].isna().sum())
        reason = MISSING_REASONS.get(c) or MISSING_REASONS["transition_*"] if c.startswith("transition_") \
            else MISSING_REASONS.get(c, "")
        rows.append({"feature": c, "missing_count": na,
                     "missing_pct": round(100.0 * na / max(len(df), 1), 3),
                     "expected_reason": reason if na else ""})
    return pd.DataFrame(rows)


def compute_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    cols = _numeric_features(df)
    d = df[cols].astype(np.float64)
    st = d.describe(percentiles=[0.01, 0.5, 0.99]).T
    st["missing_pct"] = 100.0 * d.isna().mean()
    st["n_unique"] = d.nunique()
    st["is_constant"] = st["n_unique"] <= 1
    st.index.name = "feature"
    return st.reset_index()


def compute_feature_correlations(df: pd.DataFrame, max_rows: int = 300_000) -> pd.DataFrame:
    cols = _numeric_features(df)
    d = df[cols].astype(np.float32)
    if len(d) > max_rows:
        d = d.sample(max_rows, random_state=0)
    return d.corr(method="pearson")


def find_redundant_features(corr: pd.DataFrame, threshold: float = 0.95) -> list[dict[str, Any]]:
    c = corr.abs().to_numpy()
    names = list(corr.columns)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.isfinite(c[i, j]) and c[i, j] >= threshold:
                out.append({"feature_1": names[i], "feature_2": names[j],
                            "correlation": round(float(corr.iloc[i, j]), 4)})
    return out


def find_duplicate_columns(df: pd.DataFrame, sample: int = 200_000) -> list[tuple[str, str]]:
    cols = _numeric_features(df)
    d = df[cols].head(sample)
    dup = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if np.array_equal(d[a].to_numpy(), d[b].to_numpy(), equal_nan=True):
                dup.append((a, b))
    return dup


def check_reference_route_leakage(df: pd.DataFrame, corr_threshold: float = 0.95) -> dict[str, Any]:
    """Structural + empirical leakage checks.

    * No feature column may be (or be named after) a label / route column.
    * Raw identifiers (arc_id) must not be feature columns.
    * No single feature may be near-perfectly correlated with a label.
    """
    feats = feature_columns(df)
    bad_names = [c for c in feats if c.startswith(("label_", "route_", "reference_")) or c in KEY_COLUMNS
                 or "route" in c]
    suspicious = {}
    if "label_aligned" in df.columns:
        lab = df["label_aligned"].astype(np.float32)
        for c in _numeric_features(df):
            r = df[c].astype(np.float32).corr(lab)
            if np.isfinite(r) and abs(r) >= corr_threshold:
                suspicious[c] = round(float(r), 4)
    return {"forbidden_feature_names": bad_names, "high_label_correlation": suspicious,
            "ok": not bad_names and not suspicious}


def make_group_folds(trip_ids, n_splits: int = 5):
    """Deterministic trip-level (group) folds: no trip appears in both train and validation."""
    from sklearn.model_selection import GroupKFold
    trip_ids = np.asarray(trip_ids)
    folds = list(GroupKFold(n_splits=n_splits).split(np.zeros(len(trip_ids)), groups=trip_ids))
    assert_no_group_leakage(trip_ids, folds)
    return folds


def assert_no_group_leakage(trip_ids, folds) -> None:
    trip_ids = np.asarray(trip_ids)
    for tr, va in folds:
        both = set(trip_ids[tr]) & set(trip_ids[va])
        if both:
            raise AssertionError(f"Trips appear in both train and validation: {sorted(both)[:5]}")


def build_report(df: pd.DataFrame, out_dir: Path, alignment: pd.DataFrame | None = None,
                 title_note: str = "") -> dict[str, Any]:
    """Write feature_statistics.csv, feature_correlations.csv, feature_report.md."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_feature_statistics(df)
    corr = compute_feature_correlations(df)
    stats.to_csv(out_dir / "feature_statistics.csv", index=False)
    corr.to_csv(out_dir / "feature_correlations.csv")
    miss = analyze_missing_values(df)
    red = find_redundant_features(corr)
    dup = find_duplicate_columns(df)
    inf = count_non_finite(df)
    keys = check_keys(df)
    leak = check_reference_route_leakage(df)
    const = stats.loc[stats["is_constant"], "feature"].tolist()

    L = ["# PathSnap feature report", "", title_note, "",
         f"- Rows (observation x candidate): {len(df):,}",
         f"- Trips: {df['trip_id'].nunique() if 'trip_id' in df else 'n/a'}",
         f"- Model-input features: {len(feature_columns(df))}",
         f"- Key check: {keys}",
         f"- Infinite values: {inf or 'none'}",
         f"- Constant features: {const or 'none'}",
         f"- Exact duplicate features: {dup or 'none'}",
         f"- Redundant pairs (|r| >= 0.95, sample): {len(red)}"]
    L += [f"  - {r['feature_1']} ~ {r['feature_2']} (r={r['correlation']})" for r in red]
    L += ["", "## Leakage check", f"- {leak}", "",
          "## Missing values (NaN kept; never zero-filled)", "",
          miss[miss.missing_count > 0].to_markdown(index=False) if (miss.missing_count > 0).any()
          else "None.", ""]
    if alignment is not None and len(alignment):
        L += ["## Label alignment", "", f"Trips aligned: {len(alignment)}"]
        for k in ("labeled_observation_rate", "aligned_is_nearest_rate", "aligned_mean_dist_m",
                  "monotone_violations", "large_jumps_gt10"):
            if k in alignment:
                L.append(f"- {k}: mean={alignment[k].mean():.4f}, min={alignment[k].min():.4f}")
        low = alignment.sort_values("labeled_observation_rate").head(5)[
            ["trip_id", "labeled_observation_rate", "observation_candidate_miss_rate"]]
        L += ["", "Lowest labeled-observation rates (investigate):", "", low.to_markdown(index=False), ""]
    (out_dir / "feature_report.md").write_text("\n".join(L))
    return {"keys": keys, "inf": inf, "leak": leak, "redundant": red, "duplicates": dup, "constant": const}
