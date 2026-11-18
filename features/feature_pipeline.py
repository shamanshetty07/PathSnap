"""PathSnap feature pipeline.

    python -m features.feature_pipeline --split train --all-trips
    python -m features.feature_pipeline --split student_test --all-trips
    python -m features.feature_pipeline --split train --trip 00000001
    # tune candidate generation:   --search-radius 30 --max-candidates 3

One output row = one GPS observation x one candidate arc.
Output: outputs/features/<split>/<trip_id>/features.csv (keys + base columns +
features + label_* columns).  The reference route is used only by
``features.label_alignment`` to build label_* columns; feature groups never
receive it.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import pandas as pd

from exploration.candidate_generation.candidate_generator import generate_candidates
from exploration.data_loader import TripDataError, load_trip
from features.label_alignment import align_labels, investigate_alignment_quality
from features.network_features import compute_network_features
from features.schema import BASE_COLUMNS, KEY_COLUMNS, LABEL_COLUMNS, feature_columns
from features.spatial_features import compute_spatial_features
from features.temporal_features import compute_temporal_features
from features.transition_features import compute_transition_features

SPLITS = ("train", "student_test")


def _list_trips(data_dir: Path, split: str) -> list[Path]:
    d = Path(data_dir) / split
    if not d.is_dir():
        return []
    return sorted((p for p in d.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: p.name)


def build_trip_features(trip, search_radius_m: float = 50.0, max_candidates: int = 5):
    """Features + labels for a loaded TripData. Returns (df, alignment_report)."""
    res = generate_candidates(trip, search_radius_m=search_radius_m, max_candidates=max_candidates)
    df = pd.DataFrame([c.to_dict() for c in res.candidates])
    if df.empty:
        return df, {"trip_id": trip.trip_id, "status": "no_candidates",
                    "total_observations": 0, "labeled_observation_rate": 0.0,
                    "observation_candidate_miss_rate": 1.0}
    df["trip_id"] = trip.trip_id
    # Feature groups see only GPS, candidates and the road network (never trip.route).
    df = compute_spatial_features(df, nodes_df=trip.nodes)
    df = compute_temporal_features(df, track=trip.track)
    df = compute_network_features(df, arcs_df=trip.arcs, nodes_df=trip.nodes)
    df = compute_transition_features(df, arcs_df=trip.arcs, nodes_df=trip.nodes)
    # Labels last, from the reference route.
    df = align_labels(df, trip.route)
    cols = KEY_COLUMNS + [c for c in BASE_COLUMNS if c in df.columns] + feature_columns(df) + LABEL_COLUMNS
    df = df[cols]
    rep = investigate_alignment_quality(df, trip.trip_id, len(trip.track), len(trip.route))
    return df, rep


def run_feature_pipeline(split: str, trip_id: str, search_radius_m: float = 50.0,
                         max_candidates: int = 5, output_dir="outputs/features",
                         base_dir="Data", return_report: bool = False):
    trip = load_trip(split=split, trip_id=trip_id, base_dir=Path(base_dir))
    df, rep = build_trip_features(trip, search_radius_m, max_candidates)
    out = Path(output_dir) / split / trip_id
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "features.csv", index=False, float_format="%.6g")
    return (df, rep) if return_report else df


def _worker(args):
    split, tid, radius, maxc, out, base = args
    t0 = time.time()
    try:
        df, rep = run_feature_pipeline(split, tid, radius, maxc, out, base, return_report=True)
        rep.update(split=split, rows=len(df), seconds=round(time.time() - t0, 2))
        return tid, rep, None
    except TripDataError as exc:
        return tid, None, f"SKIP {exc}"
    except Exception as exc:  # noqa: BLE001
        return tid, None, f"ERROR {exc!r}"


def run_all_trips(split: str, search_radius_m: float = 50.0, max_candidates: int = 5,
                  output_dir="outputs/features", base_dir="Data", jobs: int = 1) -> pd.DataFrame:
    trips = _list_trips(Path(base_dir), split)
    print(f"Found {len(trips)} trips in {split} (radius={search_radius_m} m, max_cand={max_candidates}, jobs={jobs})")
    tasks = [(split, p.name, search_radius_m, max_candidates, str(output_dir), str(base_dir)) for p in trips]
    reports, errors = [], 0
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(_worker, tasks))
    else:
        results = map(_worker, tasks)
    for i, (tid, rep, err) in enumerate(results, 1):
        if err:
            errors += 1
            print(f"  [{i}/{len(trips)}] {tid}: {err}")
        else:
            reports.append(rep)
            print(f"  [{i}/{len(trips)}] {tid}: {rep['rows']} rows ({rep['seconds']}s)")
    rep_df = pd.DataFrame(reports)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep_df.to_csv(out / f"alignment_{split}.csv", index=False)
    print(f"Done: {len(reports)} ok, {errors} failed, {int(rep_df['rows'].sum()) if len(rep_df) else 0} rows")
    return rep_df


def load_split_features(output_dir="outputs/features", split: str | None = None) -> pd.DataFrame:
    """Concatenate all per-trip feature CSVs (float32 for numeric columns)."""
    parts = []
    for sp in ([split] if split else SPLITS):
        for f in sorted((Path(output_dir) / sp).glob("*/features.csv")):
            d = pd.read_csv(f, dtype={"trip_id": str})
            d["split"] = sp
            fl = d.select_dtypes("float64").columns.difference(["timestamp", "gps_longitude", "gps_latitude",
                                                                "projected_longitude", "projected_latitude"])
            d[fl] = d[fl].astype("float32")
            parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def write_reports(output_dir="outputs/features") -> None:
    from features.feature_validation import build_report
    df = load_split_features(output_dir)
    if df.empty:
        return
    al = [pd.read_csv(Path(output_dir) / f"alignment_{s}.csv", dtype={"trip_id": str})
          for s in SPLITS if (Path(output_dir) / f"alignment_{s}.csv").exists()]
    splits = ", ".join(sorted(df["split"].unique()))
    build_report(df.drop(columns="split"), Path(output_dir),
                 pd.concat(al, ignore_index=True) if al else None, f"Splits included: {splits}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="PathSnap feature pipeline.")
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--trip")
    ap.add_argument("--all-trips", action="store_true")
    ap.add_argument("--search-radius", type=float, default=50.0, help="metres (default 50)")
    ap.add_argument("--max-candidates", type=int, default=5)
    ap.add_argument("--output-dir", default="outputs/features")
    ap.add_argument("--base-dir", default="Data")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes for --all-trips")
    ap.add_argument("--no-report", action="store_true", help="skip statistics/correlation/report generation")
    a = ap.parse_args(argv)
    if not a.all_trips and not a.trip:
        ap.error("--trip is required unless --all-trips is specified.")
    try:
        if a.all_trips:
            run_all_trips(a.split, a.search_radius, a.max_candidates, a.output_dir, a.base_dir, a.jobs)
        else:
            t0 = time.time()
            df, rep = run_feature_pipeline(a.split, a.trip, a.search_radius, a.max_candidates,
                                           a.output_dir, a.base_dir, return_report=True)
            print(f"Trip {a.split}/{a.trip}: {len(df)} rows, {len(feature_columns(df))} features, "
                  f"labeled obs rate {rep.get('labeled_observation_rate', 0):.3f}, {time.time()-t0:.1f}s")
        if not a.no_report:
            write_reports(a.output_dir)
    except (TripDataError, Exception) as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
