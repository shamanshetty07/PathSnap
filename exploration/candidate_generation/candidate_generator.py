"""Candidate generation for PathSnap map matching.

For every GPS observation, identifies nearby plausible road segments (arcs)
from the road network using a KD-tree spatial index and exact point-to-segment
distance calculations.

Candidate generation depends *only* on GPS observations and the road network.
Reference-route information is never used to generate candidates; it is used
solely for evaluation (see :mod:`exploration.candidate_generation.candidate_validation`).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..data_loader import TripDataError, load_trip
from .geometry import (
    EARTH_RADIUS_M,
    LocalProjection,
    gps_bearing_degrees,
    haversine_distance_m,
    project_point_on_segment,
    segment_bearing_meters,
)
from .spatial_index import RoadNetworkIndex


@dataclass
class Candidate:
    """A single candidate road segment for a GPS observation."""

    observation_index: int
    timestamp: float
    gps_longitude: float
    gps_latitude: float
    arc_id: int
    from_node_id: int
    to_node_id: int
    distance_m: float
    projected_longitude: float
    projected_latitude: float
    segment_length_m: float
    distance_along_segment_m: float
    segment_bearing: float
    gps_bearing: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CandidateGenerationResult:
    """Result of generating candidates for a single trip."""

    trip_id: str
    split: str
    num_observations: int
    num_arcs_indexed: int
    search_radius_m: float
    max_candidates: int
    candidates: list[Candidate] = field(default_factory=list)
    load_warnings: list[str] = field(default_factory=list)

    @property
    def num_candidates(self) -> int:
        return len(self.candidates)

    @property
    def observations_with_candidates(self) -> int:
        return len({c.observation_index for c in self.candidates})

    @property
    def miss_rate(self) -> float:
        if self.num_observations == 0:
            return 0.0
        return 1.0 - self.observations_with_candidates / self.num_observations

    @property
    def avg_candidates_per_observation(self) -> float:
        if self.num_observations == 0:
            return 0.0
        return self.num_candidates / self.num_observations

    def to_summary_dict(self) -> dict:
        return {
            "split": self.split,
            "trip_id": self.trip_id,
            "num_observations": self.num_observations,
            "num_arcs_indexed": self.num_arcs_indexed,
            "search_radius_m": self.search_radius_m,
            "max_candidates": self.max_candidates,
            "num_candidates": self.num_candidates,
            "observations_with_candidates": self.observations_with_candidates,
            "observations_without_candidates": self.num_observations
            - self.observations_with_candidates,
            "miss_rate": self.miss_rate,
            "avg_candidates_per_observation": self.avg_candidates_per_observation,
            "max_candidates_per_observation": self._max_candidates_per_obs(),
            "min_candidates_per_observation": self._min_candidates_per_obs(),
        }

    def _counts_per_obs(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for c in self.candidates:
            counts[c.observation_index] = counts.get(c.observation_index, 0) + 1
        return counts

    def _max_candidates_per_obs(self) -> int:
        counts = self._counts_per_obs()
        return max(counts.values()) if counts else 0

    def _min_candidates_per_obs(self) -> int:
        counts = self._counts_per_obs()
        return min(counts.values()) if counts else 0


def _compute_gps_bearings(track: pd.DataFrame) -> np.ndarray:
    """Compute bearing from previous observation to current for each row.

    Row 0 has no previous observation; its bearing is 0.0.
    """
    n = len(track)
    bearings = np.zeros(n, dtype=np.float64)
    if n < 2:
        return bearings
    lats = track["latitude"].to_numpy(dtype=np.float64)
    lons = track["longitude"].to_numpy(dtype=np.float64)
    for i in range(1, n):
        bearings[i] = gps_bearing_degrees(
            lats[i - 1], lons[i - 1], lats[i], lons[i]
        )
    return bearings


def generate_candidates(
    trip_data: TripData,
    search_radius_m: float = 50.0,
    max_candidates: int = 5,
) -> CandidateGenerationResult:
    """Generate candidate road segments for every GPS observation in a trip.

    Parameters
    ----------
    trip_data : TripData
        Loaded trip data (track, nodes, arcs).
    search_radius_m : float
        Maximum distance (metres) from a GPS observation to a road segment
        for that segment to be considered a candidate.
    max_candidates : int
        Maximum number of candidates returned per GPS observation, sorted by
        distance.

    Returns
    -------
    CandidateGenerationResult
        All candidates plus summary statistics.
    """
    track = trip_data.track
    nodes = trip_data.nodes
    arcs = trip_data.arcs
    warnings = list(trip_data.load_warnings)

    num_observations = int(len(track))
    num_arcs_indexed = int(len(arcs))

    result = CandidateGenerationResult(
        trip_id=trip_data.trip_id,
        split=trip_data.split,
        num_observations=num_observations,
        num_arcs_indexed=num_arcs_indexed,
        search_radius_m=search_radius_m,
        max_candidates=max_candidates,
        load_warnings=warnings,
    )

    if num_observations == 0:
        warnings.append("No GPS observations to generate candidates for.")
        return result
    if arcs.empty or nodes.empty:
        warnings.append("Road network is empty; no candidates can be generated.")
        return result

    # Build spatial index on road segments.
    index = RoadNetworkIndex(trip_data)
    index.build()
    result.num_arcs_indexed = len(index)
    if len(index) == 0:
        warnings.append("No valid road segments to index.")
        return result

    # Precompute GPS bearings.
    bearings = _compute_gps_bearings(track)

    # Generate candidates per observation.
    lats = track["latitude"].to_numpy(dtype=np.float64)
    lons = track["longitude"].to_numpy(dtype=np.float64)
    timestamps = track["timestamp"].to_numpy(dtype=np.float64)

    for i in range(num_observations):
        lon = float(lons[i])
        lat = float(lats[i])
        ts = float(timestamps[i])
        gps_bearing = float(bearings[i])

        nearby_segments = index.query(lon, lat, search_radius_m)

        obs_candidates: list[Candidate] = []
        for seg in nearby_segments:
            # Project GPS point onto the segment in projected space.
            proj = index.projection
            assert proj is not None
            qx, qy = proj.to_meters(lon, lat)
            t, dist_m, (proj_x, proj_y) = project_point_on_segment(
                qx, qy, seg.ax, seg.ay, seg.bx, seg.by
            )

            # Filter by exact search radius (point-to-segment distance).
            if dist_m > search_radius_m:
                continue

            proj_lon, proj_lat = proj.from_meters(proj_x, proj_y)
            distance_along = t * seg.length_m

            obs_candidates.append(
                Candidate(
                    observation_index=i,
                    timestamp=ts,
                    gps_longitude=lon,
                    gps_latitude=lat,
                    arc_id=seg.arc_id,
                    from_node_id=seg.from_node_id,
                    to_node_id=seg.to_node_id,
                    distance_m=float(dist_m),
                    projected_longitude=float(proj_lon),
                    projected_latitude=float(proj_lat),
                    segment_length_m=float(seg.length_m),
                    distance_along_segment_m=float(distance_along),
                    segment_bearing=float(
                        segment_bearing_meters(seg.ax, seg.ay, seg.bx, seg.by)
                    ),
                    gps_bearing=gps_bearing,
                )
            )

        # Sort by distance and cap the number of candidates.
        obs_candidates.sort(key=lambda c: c.distance_m)
        result.candidates.extend(obs_candidates[:max_candidates])

    return result


def save_candidates_csv(result: CandidateGenerationResult, path: Path) -> None:
    """Save candidates to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "observation_index",
                "timestamp",
                "gps_longitude",
                "gps_latitude",
                "arc_id",
                "from_node_id",
                "to_node_id",
                "distance_m",
                "projected_longitude",
                "projected_latitude",
                "segment_length_m",
                "distance_along_segment_m",
                "segment_bearing",
                "gps_bearing",
            ],
        )
        writer.writeheader()
        for c in result.candidates:
            writer.writerow(c.to_dict())


def save_candidates_jsonl(result: CandidateGenerationResult, path: Path) -> None:
    """Save candidates to a JSON Lines file (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in result.candidates:
            f.write(json.dumps(c.to_dict()) + "\n")


def run_trip(
    split: str,
    trip_id: str,
    search_radius_m: float = 50.0,
    max_candidates: int = 5,
    output_dir: Path | str = "outputs/candidates",
    generate_visualisation: bool = False,
    base_dir: Path | str = "Data",
) -> CandidateGenerationResult:
    """Load a trip, generate candidates, save outputs, and return the result."""
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)

    trip_data = load_trip(split=split, trip_id=trip_id, base_dir=base_dir)
    result = generate_candidates(
        trip_data,
        search_radius_m=search_radius_m,
        max_candidates=max_candidates,
    )

    # Save candidates per trip.
    trip_dir = output_dir / split / trip_id
    trip_dir.mkdir(parents=True, exist_ok=True)
    save_candidates_csv(result, trip_dir / "candidates.csv")
    save_candidates_jsonl(result, trip_dir / "candidates.jsonl")

    if generate_visualisation:
        from .visualize import build_candidate_map

        maps_dir = output_dir / "maps" / split
        maps_dir.mkdir(parents=True, exist_ok=True)
        build_candidate_map(trip_data, result, maps_dir / f"{trip_id}_candidates.html")

    return result


def _list_trips(data_dir: Path, split: str) -> list[Path]:
    split_dir = data_dir / split
    if not split_dir.is_dir():
        return []
    return sorted(
        [p for p in split_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: p.name,
    )


def run_all_trips(
    split: str,
    search_radius_m: float = 50.0,
    max_candidates: int = 5,
    output_dir: Path | str = "outputs/candidates",
    base_dir: Path | str = "Data",
    generate_visualisation: bool = False,
) -> list[CandidateGenerationResult]:
    """Generate candidates for every trip in a split and write aggregate stats.

    Returns the list of per-trip results.
    """
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    data_dir = base_dir

    results: list[CandidateGenerationResult] = []
    for trip_dir in _list_trips(data_dir, split):
        trip_id = trip_dir.name
        try:
            result = run_trip(
                split=split,
                trip_id=trip_id,
                search_radius_m=search_radius_m,
                max_candidates=max_candidates,
                output_dir=output_dir,
                generate_visualisation=generate_visualisation,
                base_dir=base_dir,
            )
            results.append(result)
        except TripDataError as exc:
            print(f"Skipping {split}/{trip_id}: {exc}", file=sys.stderr)

    # Write aggregate statistics CSV.
    stats_path = output_dir / f"candidate_statistics_{split}.csv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "trip_id",
                "num_observations",
                "num_arcs_indexed",
                "search_radius_m",
                "max_candidates",
                "num_candidates",
                "observations_with_candidates",
                "observations_without_candidates",
                "miss_rate",
                "avg_candidates_per_observation",
                "max_candidates_per_observation",
                "min_candidates_per_observation",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_summary_dict())

    print(f"\nAggregate statistics written to: {stats_path}")
    print(f"Trips processed: {len(results)}")
    if results:
        total_obs = sum(r.num_observations for r in results)
        total_cand = sum(r.num_candidates for r in results)
        total_missed = sum(
            r.num_observations - r.observations_with_candidates for r in results
        )
        print(f"Total GPS observations: {total_obs}")
        print(f"Total candidates: {total_cand}")
        print(f"Total observations without candidates: {total_missed}")
        print(f"Overall miss rate: {total_missed / total_obs:.4f}" if total_obs else "Overall miss rate: N/A")
    return results


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate map-matching candidates for a PathSnap trip."
    )
    parser.add_argument(
        "--split", required=True, help="Dataset split (e.g. train, student_test)."
    )
    parser.add_argument(
        "--trip", default=None, help="Trip identifier (e.g. 00000001)."
    )
    parser.add_argument(
        "--all-trips",
        action="store_true",
        help="Process all trips in the split instead of a single --trip.",
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
        default="outputs/candidates",
        help="Output directory (default: outputs/candidates).",
    )
    parser.add_argument(
        "--base-dir",
        default="Data",
        help="Repository root containing the Data directory (default: Data).",
    )
    parser.add_argument(
        "--visualise",
        action="store_true",
        help="Generate a Folium visualisation map.",
    )
    args = parser.parse_args(argv)

    if not args.all_trips and args.trip is None:
        parser.error("--trip is required unless --all-trips is specified.")

    try:
        if args.all_trips:
            results = run_all_trips(
                split=args.split,
                search_radius_m=args.search_radius,
                max_candidates=args.max_candidates,
                output_dir=Path(args.output_dir),
                generate_visualisation=args.visualise,
                base_dir=Path(args.base_dir),
            )
        else:
            result = run_trip(
                split=args.split,
                trip_id=args.trip,
                search_radius_m=args.search_radius,
                max_candidates=args.max_candidates,
                output_dir=Path(args.output_dir),
                generate_visualisation=args.visualise,
                base_dir=Path(args.base_dir),
            )
    except TripDataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - clear CLI error message
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not args.all_trips:
        result: CandidateGenerationResult
        print(f"Trip: {result.split}/{result.trip_id}")
        print(f"GPS observations: {result.num_observations}")
        print(f"Road arcs indexed: {result.num_arcs_indexed}")
        print(f"Search radius: {result.search_radius_m} m")
        print(f"Max candidates per observation: {result.max_candidates}")
        print(f"Total candidates: {result.num_candidates}")
        print(f"Observations with candidates: {result.observations_with_candidates}")
        print(f"Observations without candidates: {result.num_observations - result.observations_with_candidates}")
        print(f"Miss rate: {result.miss_rate:.4f}")
        print(f"Avg candidates per observation: {result.avg_candidates_per_observation:.2f}")
        print(f"Max candidates per observation: {result._max_candidates_per_obs()}")
        print(f"Min candidates per observation: {result._min_candidates_per_obs()}")
        for warning in result.load_warnings:
            print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
