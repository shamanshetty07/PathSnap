"""Command-line entry point for visualising a PathSnap reference route.

Usage::

    python -m exploration.visualize_route --split train --trip 00000001
    python -m exploration.visualize_route --split student_test --trip 00000076

The command loads the trip, reconstructs its reference route, validates route
continuity, builds an interactive Folium map, saves it under
``outputs/maps/`` and prints a trip analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import folium

from .data_loader import TripDataError, load_trip
from .route_reconstruction import ReconstructedRoute, reconstruct_reference_route

# Default repository root is the parent of the ``exploration`` package.
DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MAPS_DIR = DEFAULT_BASE_DIR / "outputs" / "maps"

# Folium expects (latitude, longitude); the dataset stores (longitude, latitude).
MAP_TILES = "OpenStreetMap"


def _build_map(
    trip_data,
    route: ReconstructedRoute,
) -> folium.Map:
    """Create the Folium map showing GPS observations and the reference route."""
    # Determine a sensible map centre: prefer the route start, fall back to the
    # GPS track centroid.
    centre = route.start_coordinate
    if centre is None and not trip_data.track.empty:
        centre = (
            float(trip_data.track["latitude"].mean()),
            float(trip_data.track["longitude"].mean()),
        )
    if centre is None:
        centre = (0.0, 0.0)

    m = folium.Map(location=centre, zoom_start=14, tiles=MAP_TILES)

    # Raw GPS observations as blue points. To keep the map readable for very
    # long tracks we render every point (tracks here are small enough); a
    # CircleMarker per point is used so the layer stays lightweight.
    if not trip_data.track.empty:
        gps_layer = folium.FeatureGroup(name="GPS observations (blue)", show=True)
        for _, row in trip_data.track.iterrows():
            folium.CircleMarker(
                location=[float(row["latitude"]), float(row["longitude"])],
                radius=2,
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.6,
                weight=1,
                popup="GPS observation",
            ).add_to(gps_layer)
        gps_layer.add_to(m)

    # Reconstructed reference route: one green PolyLine per connected segment.
    # Segments are never joined across a disconnected transition.
    route_layer = folium.FeatureGroup(name="Reference route (green)", show=True)
    for segment in route.segments:
        if len(segment) < 2:
            continue
        folium.PolyLine(
            locations=segment,
            color="green",
            weight=4,
            opacity=0.9,
            popup="Reference route segment",
        ).add_to(route_layer)
    route_layer.add_to(m)

    # Start (green) and end (red) markers.
    if route.start_coordinate is not None:
        folium.Marker(
            location=list(route.start_coordinate),
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
            popup="Start location",
            tooltip="Start",
        ).add_to(m)
    if route.end_coordinate is not None:
        folium.Marker(
            location=list(route.end_coordinate),
            icon=folium.Icon(color="red", icon="stop", prefix="fa"),
            popup="End location",
            tooltip="End",
        ).add_to(m)

    # Disconnected transitions clearly identified with orange markers.
    if route.disconnected_transitions:
        disc_layer = folium.FeatureGroup(
            name="Disconnected transitions", show=True
        )
        for transition in route.disconnected_transitions:
            if transition.prev_to_coord is not None:
                folium.Marker(
                    location=list(transition.prev_to_coord),
                    icon=folium.Icon(color="orange", icon="exclamation-triangle", prefix="fa"),
                    popup=(
                        f"Disconnected transition<br>"
                        f"Arc {transition.prev_arc_id} -> {transition.next_arc_id}<br>"
                        f"End node {transition.prev_to_node}"
                    ),
                    tooltip="Disconnected transition (end)",
                ).add_to(disc_layer)
            if transition.next_from_coord is not None:
                folium.Marker(
                    location=list(transition.next_from_coord),
                    icon=folium.Icon(color="orange", icon="exclamation-triangle", prefix="fa"),
                    popup=(
                        f"Disconnected transition<br>"
                        f"Arc {transition.prev_arc_id} -> {transition.next_arc_id}<br>"
                        f"Start node {transition.next_from_node}"
                    ),
                    tooltip="Disconnected transition (start)",
                ).add_to(disc_layer)
        disc_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def _save_map(m: folium.Map, split: str, trip_id: str, maps_dir: Path) -> Path:
    maps_dir.mkdir(parents=True, exist_ok=True)
    out_path = maps_dir / f"{split}_{trip_id}_reference_route.html"
    m.save(str(out_path))
    return out_path


def _print_analysis(trip_data, route: ReconstructedRoute) -> None:
    print("=" * 60)
    print(f"Trip analysis: split={trip_data.split}, trip_id={trip_data.trip_id}")
    print("=" * 60)
    print(f"GPS observations          : {len(trip_data.track)}")
    print(f"Road nodes                : {len(trip_data.nodes)}")
    print(f"Road arcs                 : {len(trip_data.arcs)}")
    print(f"Reference-route arcs      : {len(route.arc_ids)}")
    print(f"Reconstructed coordinates : {route.num_coordinates}")
    print(f"Connected segments        : {route.num_segments}")
    print(f"Disconnected transitions  : {len(route.disconnected_transitions)}")
    print(f"Fully connected           : {route.is_fully_connected}")
    if route.validation.invalid_arc_ids:
        print(f"Invalid arc IDs           : {route.validation.invalid_arc_ids}")
    if route.validation.missing_node_ids:
        print(f"Missing node IDs          : {sorted(set(route.validation.missing_node_ids))}")
    for msg in route.validation.messages:
        print(f"Validation                : {msg}")
    for warning in route.load_warnings:
        print(f"Warning                   : {warning}")
    if route.disconnected_transitions:
        print("\nDisconnected transitions:")
        for t in route.disconnected_transitions:
            print(
                f"  - route position {t.transition_index}: "
                f"arc {t.prev_arc_id} (node {t.prev_to_node}) -> "
                f"arc {t.next_arc_id} (node {t.next_from_node})"
            )
    print("=" * 60)


def run(split: str, trip_id: str, base_dir: Path = DEFAULT_BASE_DIR, maps_dir: Path = DEFAULT_MAPS_DIR) -> Path:
    """Load, reconstruct, validate, visualise and save a trip's reference route.

    Returns the path to the saved HTML map.
    """
    trip_data = load_trip(
        split=split,
        trip_id=trip_id,
        base_dir=base_dir / "Data",
    )
    route = reconstruct_reference_route(trip_data)

    map_obj = _build_map(trip_data, route)
    out_path = _save_map(map_obj, split, trip_id, maps_dir)

    _print_analysis(trip_data, route)
    print(f"Map saved to: {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualise a PathSnap reference route with Folium."
    )
    parser.add_argument(
        "--split",
        required=True,
        help="Dataset split (e.g. train, student_test).",
    )
    parser.add_argument(
        "--trip",
        required=True,
        help="Trip identifier (e.g. 00000001).",
    )
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help="Repository root containing the Data directory.",
    )
    parser.add_argument(
        "--maps-dir",
        default=str(DEFAULT_MAPS_DIR),
        help="Directory where the HTML map is written.",
    )
    args = parser.parse_args(argv)

    try:
        run(
            split=args.split,
            trip_id=args.trip,
            base_dir=Path(args.base_dir),
            maps_dir=Path(args.maps_dir),
        )
    except TripDataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - clear CLI error message
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
