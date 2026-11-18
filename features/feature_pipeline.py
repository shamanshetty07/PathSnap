"""Feature pipeline for PathSnap map matching.

Orchestrates the full feature-engineering process for a single trip or all
trips in a split:

1.  Load trip data  (data_loader)
2.  Generate candidates  (candidate_generator)
3.  Convert candidates to a DataFrame
4.  Apply spatial → temporal → network → transition feature groups
5.  Save the resulting DataFrame to ``outputs/features/{split}/{trip_id}/features.csv``

The reference route (``.route``) is **never** used as an input feature
(data-leakage prevention).  It is consumed only by
:mod:`features.label_alignment` for label construction.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from exploration.data_loader import TripDataError, load_trip
from exploration.candidate_generation.candidate_generator import (
    generate_candidates,
    CandidateGenerationResult,
)
from features.spatial_features import compute_spatial_features
from features.temporal_features import compute_temporal_features
from features.network_features import compute_network_features
from features.transition_features import compute_transition_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_trips(data_dir: Path, split: str) -> list[Path]:
    """Return sorted list of trip directories within a split."""
    split_dir = Path(data_dir) / split
    if not split_dir.is_dir():
        return []
    return sorted(
        [p for p in split_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: p.name,
    )


def _candidates_to_dataframe(
    result: CandidateGenerationResult,
) -> pd.DataFrame:
    """Convert a CandidateGenerationResult to a flat DataFrame."""
    rows = [c.to_dict() for c in result.candidates]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["trip_id"] = result.trip_id
    return df


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_feature_pipeline(
    split: str,
    trip_id: str,
    search_radius_m: float = 50.0,
    max_candidates: int = 5,
    output_dir: str | Path = "outputs/features",
    base_dir: str | Path = "Data",
) -> pd.DataFrame:
    """Run the full feature pipeline for a single trip.

    Parameters
    ----------
    split : str
        Dataset split (``train`` or ``student_test``).
    trip_id : str
        Trip identifier (e.g. ``00000001``).
    search_radius_m : float
        Search radius passed to candidate generation.
    max_candidates : int
        Maximum candidates per GPS observation.
    output_dir : str | Path
        Root directory for feature CSV outputs.
    base_dir : str | Path
        Root directory containing the Data folders.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame with one row per GPS-observation × candidate-arc pair.
    """
    output_dir = Path(output_dir)
    base_dir = Path(base_dir)

    # 1. Load trip data
    trip_data = load_trip(split=split, trip_id=trip_id, base_dir=base_dir)

    # 2. Generate candidates
    cand_result = generate_candidates(
        trip_data,
        search_radius_m=search_radius_m,
        max_candidates=max_candidates,
    )

    # 3. Convert to DataFrame
    df = _candidates_to_dataframe(cand_result)
    if df.empty:
        print(f"  [WARN] No candidates generated for {split}/{trip_id}")
        return df

    # 4. Apply feature groups sequentially
    df = compute_spatial_features(df, nodes_df=trip_data.nodes)
    df = compute_temporal_features(df)
    df = compute_network_features(df, arcs_df=trip_data.arcs, nodes_df=trip_data.nodes)
    df = compute_transition_features(df, arcs_df=trip_data.arcs)

    # 5. Save to CSV
    trip_out = output_dir / split / trip_id
    trip_out.mkdir(parents=True, exist_ok=True)
    csv_path = trip_out / "features.csv"
    df.to_csv(csv_path, index=False)

    return df


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def run_all_trips(
    split: str,
    search_radius_m: float = 50.0,
    max_candidates: int = 5,
    output_dir: str | Path = "outputs/features",
    base_dir: str | Path = "Data",
) -> list[pd.DataFrame]:
    """Run the feature pipeline for every trip in a split.

    Returns the list of per-trip feature DataFrames.
    """
    output_dir = Path(output_dir)
    base_dir = Path(base_dir)

    trips = _list_trips(base_dir, split)
    print(f"Found {len(trips)} trips in {split}")

    results: list[pd.DataFrame] = []
    total_rows = 0
    total_cols = 0
    errors = 0

    for idx, trip_dir in enumerate(trips, 1):
        tid = trip_dir.name
        print(f"  [{idx}/{len(trips)}] Processing {split}/{tid} ...", end=" ")
        t0 = time.time()
        try:
            df = run_feature_pipeline(
                split=split,
                trip_id=tid,
                search_radius_m=search_radius_m,
                max_candidates=max_candidates,
                output_dir=output_dir,
                base_dir=base_dir,
            )
            elapsed = time.time() - t0
            print(f"{len(df)} rows, {len(df.columns)} cols  ({elapsed:.1f}s)")
            results.append(df)
            total_rows += len(df)
            if len(df.columns) > total_cols:
                total_cols = len(df.columns)
        except TripDataError as exc:
            elapsed = time.time() - t0
            print(f"SKIP ({exc})  ({elapsed:.1f}s)")
            errors += 1
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"ERROR ({exc})  ({elapsed:.1f}s)")
            errors += 1

    print(f"\n--- Summary for split={split} ---")
    print(f"  Trips processed: {len(results)}")
    print(f"  Trips skipped/errored: {errors}")
    print(f"  Total rows: {total_rows}")
    print(f"  Max columns: {total_cols}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="PathSnap feature pipeline – generate ML features for map matching.",
    )
    parser.add_argument(
        "--split",
        required=True,
        help="Dataset split (train or student_test).",
    )
    parser.add_argument(
        "--trip",
        default=None,
        help="Single trip identifier (e.g. 00000001).",
    )
    parser.add_argument(
        "--all-trips",
        action="store_true",
        help="Process all trips in the split.",
    )
    parser.add_argument(
        "--search-radius",
        type=float,
        default=50.0,
        help="Search radius in metres (default: 50).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=5,
        help="Maximum candidates per observation (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/features",
        help="Output directory (default: outputs/features).",
    )
    parser.add_argument(
        "--base-dir",
        default="Data",
        help="Base data directory (default: Data).",
    )
    args = parser.parse_args(argv)

    if not args.all_trips and args.trip is None:
        parser.error("--trip is required unless --all-trips is specified.")

    try:
        if args.all_trips:
            dfs = run_all_trips(
                split=args.split,
                search_radius_m=args.search_radius,
                max_candidates=args.max_candidates,
                output_dir=Path(args.output_dir),
                base_dir=Path(args.base_dir),
            )
        else:
            t0 = time.time()
            df = run_feature_pipeline(
                split=args.split,
                trip_id=args.trip,
                search_radius_m=args.search_radius,
                max_candidates=args.max_candidates,
                output_dir=Path(args.output_dir),
                base_dir=Path(args.base_dir),
            )
            elapsed = time.time() - t0
            print(f"Trip: {args.split}/{args.trip}")
            print(f"Rows: {len(df)}")
            print(f"Columns: {len(df.columns)}")
            feature_cols = [
                c for c in df.columns
                if c.startswith(("spatial_", "temporal_", "network_", "transition_"))
            ]
            print(f"Feature columns: {len(feature_cols)}")
            print(f"Elapsed: {elapsed:.2f}s")
            print(f"Output: outputs/features/{args.split}/{args.trip}/features.csv")
    except TripDataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
