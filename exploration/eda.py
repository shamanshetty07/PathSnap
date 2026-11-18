"""End-to-end exploratory data analysis for PathSnap.

Executes the complete EDA over the ``train`` and ``student_test`` splits:

1. Dataset inventory (trip counts, file presence, malformed records).
2. GPS trajectory analysis (``.track``).
3. Road-network analysis (``.nodes`` / ``.arcs``).
4. Reference-route analysis (``.route``).
5. GPS-to-road-network spatial analysis (KD-tree + Haversine).

Results are written to ``outputs/eda/``:

* ``dataset_inventory.csv``
* ``gps_statistics.csv``
* ``network_statistics.csv``
* ``route_statistics.csv``
* ``spatial_statistics.csv``
* ``plots/``        - Matplotlib statistical plots
* ``maps/``         - Folium interactive maps for representative trips
* ``eda_report.md`` - Human-readable report

Run with::

    python -m exploration.eda

Individual analysis modules (``gps_analysis``, ``network_analysis``,
``reference_route_analysis``, ``spatial_analysis``) can also be imported and
used independently on a single :class:`~exploration.data_loader.TripData`.

The original dataset is never modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_loader import TripDataError, load_trip
from .gps_analysis import analyze_gps, haversine_distance_m
from .network_analysis import analyze_network
from .reference_route_analysis import analyze_reference_route
from .route_reconstruction import reconstruct_reference_route
from .spatial_analysis import analyze_spatial

# ``DEFAULT_DATA_DIR`` is imported above via a guarded placeholder; define it
# explicitly here for clarity.
DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = DEFAULT_BASE_DIR / "Data"
DEFAULT_OUTPUT_DIR = DEFAULT_BASE_DIR / "outputs" / "eda"

SPLITS = ("train", "student_test")

# Representative trips for which Folium trajectory maps are generated.
REPRESENTATIVE_TRIPS = {
    "train": ["00000001", "00000013", "00000055"],
    "student_test": ["00000076"],
}

# Plotting caps to keep figures fast while remaining representative.
MAX_GEO_SCATTER_POINTS = 60000
MAX_POOLED_SPEEDS = 120000
MAX_POOLED_INTERVALS = 120000
MAX_POOLED_NEAREST = 120000
MAX_POOLED_DEGREES = 100000
MAX_DEGREES_PER_TRIP = 5000

RNG = np.random.default_rng(0)


def _subsample(values: np.ndarray, max_n: int) -> np.ndarray:
    if values.size <= max_n:
        return values
    return values[RNG.choice(values.size, max_n, replace=False)]


def _list_trips(data_dir: Path, split: str) -> list[Path]:
    split_dir = data_dir / split
    if not split_dir.is_dir():
        return []
    return sorted(
        [p for p in split_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: p.name,
    )


def _process_trip(split: str, trip_dir: Path) -> dict | None:
    """Load and analyse one trip. Returns a result dict or None on hard error."""
    trip_id = trip_dir.name
    try:
        trip_data = load_trip(split=split, trip_id=trip_id, base_dir=DEFAULT_DATA_DIR)
    except TripDataError as exc:
        return {
            "trip_id": trip_id,
            "split": split,
            "error": str(exc),
            "gps": None, "network": None, "route": None, "spatial": None,
            "arrays": None,
        }

    gps_stats, gps_arrays = analyze_gps(trip_data, trip_dir)
    network_stats, degree_array = analyze_network(trip_data)
    route_stats = analyze_reference_route(trip_data)
    spatial_result = analyze_spatial(trip_data)

    return {
        "trip_id": trip_id,
        "split": split,
        "error": None,
        "trip_data": trip_data,
        "gps": gps_stats,
        "network": network_stats,
        "route": route_stats,
        "spatial": spatial_result.stats,
        "arrays": {
            "gps": gps_arrays,
            "degrees": degree_array,
            "nearest": spatial_result.nearest_distances_m,
        },
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _safe_float(values, default=0.0):
    values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not values:
        return default
    return float(np.mean(values))


def _plot_gps_observations_per_trip(rows, out_path: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    df = df.sort_values("num_observations", ascending=False)
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(df)), df["num_observations"], color="#4c78a8", width=0.8)
    plt.xlabel("Trip (sorted by observation count)")
    plt.ylabel("GPS observations")
    plt.title("GPS observations per trip")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _plot_timestamp_intervals(intervals: np.ndarray, out_path: Path) -> None:
    intervals = _subsample(intervals, MAX_POOLED_INTERVALS)
    plt.figure(figsize=(8, 5))
    plt.hist(intervals, bins=60, color="#4c78a8", edgecolor="black", alpha=0.8)
    plt.xlabel("Timestamp interval (s)")
    plt.ylabel("Count")
    plt.title("GPS timestamp interval distribution")
    plt.axvline(1.0, color="red", linestyle="--", label="1 s")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _plot_speed_distribution(speeds: np.ndarray, out_path: Path) -> None:
    speeds = _subsample(speeds, MAX_POOLED_SPEEDS)
    # Clip extreme outliers for readability but retain the tail info in title.
    display = np.clip(speeds, 0, np.percentile(speeds, 99.5)) if speeds.size else speeds
    plt.figure(figsize=(8, 5))
    plt.hist(display, bins=60, color="#e45756", edgecolor="black", alpha=0.8)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Count")
    plt.title("GPS speed distribution between consecutive observations")
    if speeds.size:
        plt.axvline(30.0, color="black", linestyle="--", label="30 m/s outlier threshold")
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _plot_route_length_distribution(route_lengths: list[int], out_path: Path) -> None:
    arr = np.array(route_lengths, dtype=np.float64)
    plt.figure(figsize=(8, 5))
    plt.hist(arr, bins=max(10, min(40, len(arr))), color="#54a24b",
             edgecolor="black", alpha=0.8)
    plt.xlabel("Reference-route length (arcs)")
    plt.ylabel("Number of trips")
    plt.title("Reference-route length distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _plot_nodes_arcs(network_rows, out_path: Path) -> None:
    df = pd.DataFrame(network_rows)
    if df.empty:
        return
    plt.figure(figsize=(8, 6))
    plt.scatter(df["num_nodes"], df["num_arcs"], c="#4c78a8", alpha=0.6, edgecolor="black")
    plt.xlabel("Road nodes")
    plt.ylabel("Road arcs")
    plt.title("Road-network size per trip")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _plot_node_degree_distribution(degrees: np.ndarray, out_path: Path) -> None:
    degrees = _subsample(degrees, MAX_POOLED_DEGREES)
    degrees = degrees[degrees > 0]  # degree-0 nodes are isolated/unreferenced
    if degrees.size == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(degrees, bins=40, color="#72b7b2", edgecolor="black", alpha=0.8, log=True)
    plt.xlabel("Node total degree (in + out)")
    plt.ylabel("Number of nodes (log scale)")
    plt.title("Node degree distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _plot_nearest_node_distance(distances: np.ndarray, out_path: Path) -> None:
    distances = _subsample(distances, MAX_POOLED_NEAREST)
    if distances.size == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(distances, bins=60, color="#f28e2b", edgecolor="black", alpha=0.8)
    plt.xlabel("Nearest-node distance (m)")
    plt.ylabel("Number of GPS observations")
    plt.title("Distance from GPS observations to nearest road node")
    for thr in (5, 10, 20, 50):
        plt.axvline(thr, color="black", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _plot_gps_trajectories(trip_records, out_path: Path) -> None:
    """Lat/lon trajectories for a few representative trips."""
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    for i, rec in enumerate(trip_records[:6]):
        track = rec["trip_data"].track
        if track.empty:
            continue
        plt.plot(track["longitude"], track["latitude"], color=colors[i % len(colors)],
                 label=f"{rec['split']} {rec['trip_id']}", alpha=0.8, linewidth=1.5)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Representative GPS trajectories (longitude, latitude)")
    plt.legend(fontsize="small", loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _plot_gps_route_overlay(trip_records, out_path: Path) -> None:
    """GPS points overlaid with reconstructed reference route for one trip."""
    if not trip_records:
        return
    rec = trip_records[0]
    trip_data = rec["trip_data"]
    track = trip_data.track
    route = reconstruct_reference_route(trip_data)
    fig, ax = plt.subplots(figsize=(8, 6))
    if not track.empty:
        ax.scatter(track["longitude"], track["latitude"], c="blue", s=6,
                   label="GPS observations", alpha=0.6, edgecolor="none")
    added = False
    for segment in route.segments:
        if len(segment) < 2:
            continue
        lons = [c[1] for c in segment]
        lats = [c[0] for c in segment]
        ax.plot(lons, lats, color="green", linewidth=2,
                label="Reference route" if not added else None)
        added = True
    if track.empty and not added:
        return
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize="small", loc="best")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"GPS vs reference route - {rec['split']} {rec['trip_id']}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _geographic_scatter(all_lat, all_lon, out_path: Path) -> None:
    if all_lat.size == 0:
        return
    lat = _subsample(all_lat, MAX_GEO_SCATTER_POINTS)
    lon = _subsample(all_lon, MAX_GEO_SCATTER_POINTS)
    plt.figure(figsize=(8, 6))
    plt.scatter(lon, lat, s=3, c="#4c78a8", alpha=0.4, edgecolor="none")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Geographic distribution of all GPS observations")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _folium_map(trip_data, route) -> folium.Map:
    centre = route.start_coordinate
    if centre is None and not trip_data.track.empty:
        centre = (float(trip_data.track["latitude"].mean()),
                  float(trip_data.track["longitude"].mean()))
    if centre is None:
        centre = (0.0, 0.0)

    m = folium.Map(location=centre, zoom_start=14, tiles="OpenStreetMap")

    if not trip_data.track.empty:
        gps_layer = folium.FeatureGroup(name="GPS observations (blue)", show=True)
        for _, row in trip_data.track.iterrows():
            folium.CircleMarker(
                location=[float(row["latitude"]), float(row["longitude"])],
                radius=2, color="blue", fill=True, fill_color="blue",
                fill_opacity=0.5, weight=1,
            ).add_to(gps_layer)
        gps_layer.add_to(m)

    route_layer = folium.FeatureGroup(name="Reference route (green)", show=True)
    for segment in route.segments:
        if len(segment) < 2:
            continue
        folium.PolyLine(
            locations=segment, color="green", weight=4, opacity=0.9,
        ).add_to(route_layer)
    route_layer.add_to(m)

    if route.start_coordinate is not None:
        folium.Marker(
            location=list(route.start_coordinate),
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
            popup="Start", tooltip="Start",
        ).add_to(m)
    if route.end_coordinate is not None:
        folium.Marker(
            location=list(route.end_coordinate),
            icon=folium.Icon(color="red", icon="stop", prefix="fa"),
            popup="End", tooltip="End",
        ).add_to(m)

    if route.disconnected_transitions:
        disc_layer = folium.FeatureGroup(name="Disconnected transitions", show=True)
        for t in route.disconnected_transitions:
            popup = (
                f"Disconnected transition<br>"
                f"Arc {t.prev_arc_id} -> {t.next_arc_id}<br>"
                f"End node {t.prev_to_node} / start node {t.next_from_node}"
            )
            for coord in (t.prev_to_coord, t.next_from_coord):
                if coord is not None:
                    folium.Marker(
                        location=list(coord),
                        icon=folium.Icon(color="orange",
                                         icon="exclamation-triangle", prefix="fa"),
                        popup=popup, tooltip="Disconnected transition",
                    ).add_to(disc_layer)
        disc_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def _build_maps(trip_records, maps_dir: Path) -> list[str]:
    maps_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    wanted = set()
    for split, ids in REPRESENTATIVE_TRIPS.items():
        for tid in ids:
            wanted.add((split, tid))
    # Also map every trip that has a disconnected transition.
    for rec in trip_records:
        if rec["error"] is None and rec["route"].disconnected_transitions:
            wanted.add((rec["split"], rec["trip_id"]))

    for rec in trip_records:
        if rec["error"] is not None:
            continue
        key = (rec["split"], rec["trip_id"])
        if key not in wanted:
            continue
        trip_data = rec["trip_data"]
        route = reconstruct_reference_route(trip_data)
        m = _folium_map(trip_data, route)
        out = maps_dir / f"{rec['split']}_{rec['trip_id']}_eda.html"
        m.save(str(out))
        saved.append(str(out))
    return saved


def _summary_block(rows: list[dict], key: str) -> dict:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return {"count": 0}
    arr = np.array(vals, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "total": int(np.sum(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _write_report(report_path: Path, inventory_rows, gps_rows, network_rows,
                  route_rows, spatial_rows, maps) -> None:
    lines: list[str] = []
    lines.append("# PathSnap EDA Report")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("The EDA processes every trip in the ``train`` and ``student_test`` "
                 "splits. For each trip it loads ``.track``, ``.nodes``, "
                 "``.arcs`` and ``.route`` (without modifying the dataset), "
                 "then computes GPS, road-network, reference-route and spatial "
                 "statistics. Distances use the Haversine formula because the "
                 "dataset spans multiple geographical regions and no single "
                 "projected coordinate system is assumed.")
    lines.append("")
    lines.append("Nearest-node distance is a preliminary diagnostic only: it "
                 "measures proximity to a road *node*, not to a road segment, "
                 "and is not equivalent to map matching. Disconnected reference "
                 "routes are reported, never silently repaired.")
    lines.append("")

    lines.append("## Dataset inventory")
    lines.append("")
    for split in SPLITS:
        split_rows = [r for r in inventory_rows if r["split"] == split and not r.get("error")]
        err_rows = [r for r in inventory_rows if r["split"] == split and r.get("error")]
        lines.append(f"### {split}")
        lines.append("")
        lines.append(f"- Trips processed: {len(split_rows)}")
        lines.append(f"- Trips with load errors: {len(err_rows)}")
        if split_rows:
            gps_sum = _summary_block(split_rows, "num_gps_observations")
            route_sum = _summary_block(split_rows, "route_length")
            nodes_sum = _summary_block(split_rows, "num_nodes")
            arcs_sum = _summary_block(split_rows, "num_arcs")
            lines.append(f"- GPS observations: total={gps_sum.get('total', 0)}, "
                         f"mean/trip={gps_sum.get('mean', 0):.1f}, "
                         f"median={gps_sum.get('median', 0):.1f}, "
                         f"min={gps_sum.get('min', 0)}, max={gps_sum.get('max', 0)}")
            lines.append(f"- Reference-route length (arcs): mean={route_sum.get('mean', 0):.1f}, "
                         f"median={route_sum.get('median', 0):.1f}, "
                         f"min={route_sum.get('min', 0)}, max={route_sum.get('max', 0)}")
            lines.append(f"- Road nodes: total={nodes_sum.get('total', 0)}, "
                         f"mean/trip={nodes_sum.get('mean', 0):.0f}")
            lines.append(f"- Road arcs: total={arcs_sum.get('total', 0)}, "
                         f"mean/trip={arcs_sum.get('mean', 0):.0f}")
            n_disc = sum(1 for r in split_rows if r.get("disconnected_transitions", 0) > 0)
            n_invalid = sum(1 for r in split_rows if r.get("invalid_arc_references", 0) > 0)
            lines.append(f"- Trips with disconnected reference-route transitions: {n_disc}")
            lines.append(f"- Trips with invalid arc references: {n_invalid}")
        lines.append("")

    lines.append("## GPS data analysis")
    lines.append("")
    lines.append("- Timestamps are examined for 1-second regularity, gaps "
                 "(>1 s, >5 s, >10 s), non-increasing values and duplicate "
                 "coordinates.")
    lines.append("- Consecutive-observation distances and speeds use the "
                 "Haversine formula; segments above 30 m/s are flagged as "
                 "potential outliers.")
    lines.append("- Malformed / invalid coordinate rows are counted and flagged, "
                 "not removed.")
    lines.append("")

    lines.append("## Road-network analysis")
    lines.append("")
    lines.append("- Node/arc counts, referenced vs isolated nodes, connected "
                 "components (arcs treated as undirected), degree distribution, "
                 "self-loops, duplicate directed arcs and invalid node "
                 "references are reported.")
    lines.append("- Network orientation (directed/undirected/mixed) is inferred "
                 "from arc reciprocity, not assumed.")
    lines.append("")

    lines.append("## Reference-route analysis")
    lines.append("")
    lines.append("- Route length, unique vs repeated arcs, continuity, "
                 "disconnected transitions and invalid arc/node references are "
                 "reported using the existing route-validation module.")
    lines.append("")

    lines.append("## GPS-to-road-network spatial analysis")
    lines.append("")
    lines.append("- A KD-tree locates candidate nearest road nodes; exact "
                 "Haversine distances are then computed. Buckets report the "
                 "share of observations within 5 / 10 / 20 / 50 m of a node.")
    lines.append("")

    lines.append("## Anomalies and data-quality issues")
    lines.append("")
    n_missing_gps = sum(1 for r in gps_rows if r.get("invalid_coordinate_rows", 0) > 0)
    n_dup = sum(1 for r in gps_rows if r.get("duplicate_coordinates", 0) > 0)
    n_noninc = sum(1 for r in gps_rows if r.get("non_increasing_timestamps", 0) > 0)
    n_highspeed = sum(1 for r in gps_rows if r.get("num_high_speed_segments", 0) > 0)
    n_isolated = sum(1 for r in network_rows if r.get("isolated_nodes", 0) > 0)
    n_selfloop = sum(1 for r in network_rows if r.get("num_self_loops", 0) > 0)
    n_duparc = sum(1 for r in network_rows if r.get("duplicate_directed_arcs", 0) > 0)
    n_invalid_node = sum(1 for r in network_rows if r.get("invalid_node_references", 0) > 0)
    lines.append(f"- Trips with invalid GPS coordinate rows: {n_missing_gps}")
    lines.append(f"- Trips with duplicate GPS coordinates: {n_dup}")
    lines.append(f"- Trips with non-increasing timestamps: {n_noninc}")
    lines.append(f"- Trips with high-speed (>30 m/s) GPS segments: {n_highspeed}")
    lines.append(f"- Trips with isolated road nodes: {n_isolated}")
    lines.append(f"- Trips with self-loop arcs: {n_selfloop}")
    lines.append(f"- Trips with duplicate directed arcs: {n_duparc}")
    lines.append(f"- Trips with invalid node references in arcs: {n_invalid_node}")
    lines.append("")

    lines.append("## Generated outputs")
    lines.append("")
    lines.append("- ``outputs/eda/dataset_inventory.csv``")
    lines.append("- ``outputs/eda/gps_statistics.csv``")
    lines.append("- ``outputs/eda/network_statistics.csv``")
    lines.append("- ``outputs/eda/route_statistics.csv``")
    lines.append("- ``outputs/eda/spatial_statistics.csv``")
    lines.append("- ``outputs/eda/plots/`` (Matplotlib figures)")
    lines.append("- ``outputs/eda/maps/`` (Folium interactive HTML maps)")
    lines.append("")
    if maps:
        lines.append("## Folium maps")
        lines.append("")
        for mpath in maps:
            lines.append(f"- {mpath}")
        lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- Connected components and graph metrics can be costly for very "
                 "large networks; results are computed per trip and not cached "
                 "across trips.")
    lines.append("- Nearest-node distance is approximate to the true nearest "
                 "road segment and is not a substitute for map matching.")
    lines.append("- No GPS observations are removed or imputed; suspicious "
                 "records are flagged for manual investigation.")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(base_dir: Path = DEFAULT_BASE_DIR,
        output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Run the complete EDA and return the output directory."""
    data_dir = base_dir / "Data"
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    maps_dir = output_dir / "maps"
    plots_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    inventory_rows: list[dict] = []
    gps_rows: list[dict] = []
    network_rows: list[dict] = []
    route_rows: list[dict] = []
    spatial_rows: list[dict] = []
    trip_records: list[dict] = []

    # Pooled plot data.
    all_lat_parts: list[np.ndarray] = []
    all_lon_parts: list[np.ndarray] = []
    all_speeds_parts: list[np.ndarray] = []
    all_intervals_parts: list[np.ndarray] = []
    all_route_lengths: list[int] = []
    all_nearest_parts: list[np.ndarray] = []
    all_degrees_parts: list[np.ndarray] = []

    for split in SPLITS:
        for trip_dir in _list_trips(data_dir, split):
            result = _process_trip(split, trip_dir)
            trip_id = result["trip_id"]

            # Inventory row (always, even on error).
            inventory_rows.append({
                "split": split,
                "trip_id": trip_id,
                "error": result["error"] or "",
            })

            if result["error"] or result["gps"] is None:
                continue

            gps = result["gps"]
            net = result["network"]
            route = result["route"]
            spatial = result["spatial"]
            arrays = result["arrays"]

            inventory_rows[-1].update({
                "num_gps_observations": gps.num_observations,
                "num_raw_rows": gps.num_raw_rows,
                "num_nodes": net.num_nodes,
                "num_arcs": net.num_arcs,
                "route_length": route.route_length,
                "invalid_coordinate_rows": gps.invalid_coordinate_rows,
                "invalid_arc_references": route.invalid_arc_references,
                "disconnected_transitions": route.disconnected_transitions,
                "is_fully_connected": route.is_fully_connected,
            })

            gps_rows.append(gps.to_row())
            network_rows.append(net.to_row())
            route_rows.append(route.to_row())
            spatial_rows.append(spatial.to_row())

            trip_records.append(result)

            arr = arrays["gps"]
            all_lat_parts.append(arr.latitude)
            all_lon_parts.append(arr.longitude)
            all_speeds_parts.append(arr.speeds_mps)
            all_intervals_parts.append(arr.intervals)
            all_route_lengths.append(route.route_length)
            all_nearest_parts.append(arrays["nearest"])
            all_degrees_parts.append(arrays["degrees"])

    # Write CSVs.
    _write_csv(inventory_rows, output_dir / "dataset_inventory.csv")
    _write_csv(gps_rows, output_dir / "gps_statistics.csv")
    _write_csv(network_rows, output_dir / "network_statistics.csv")
    _write_csv(route_rows, output_dir / "route_statistics.csv")
    _write_csv(spatial_rows, output_dir / "spatial_statistics.csv")

    # Assemble pooled arrays.
    all_lat = np.concatenate(all_lat_parts) if all_lat_parts else np.array([])
    all_lon = np.concatenate(all_lon_parts) if all_lon_parts else np.array([])
    all_speeds = np.concatenate(all_speeds_parts) if all_speeds_parts else np.array([])
    all_intervals = np.concatenate(all_intervals_parts) if all_intervals_parts else np.array([])
    all_intervals = all_intervals[all_intervals > 0]
    all_nearest = np.concatenate(all_nearest_parts) if all_nearest_parts else np.array([])
    all_nearest = all_nearest[all_nearest > 0]
    all_degrees = np.concatenate(all_degrees_parts) if all_degrees_parts else np.array([])

    # Plots.
    _plot_gps_observations_per_trip(gps_rows, plots_dir / "gps_observations_per_trip.png")
    _plot_timestamp_intervals(all_intervals, plots_dir / "timestamp_interval_distribution.png")
    _plot_speed_distribution(all_speeds, plots_dir / "gps_speed_distribution.png")
    _plot_route_length_distribution(all_route_lengths, plots_dir / "reference_route_length_distribution.png")
    _plot_nodes_arcs(network_rows, plots_dir / "network_nodes_vs_arcs.png")
    _plot_node_degree_distribution(all_degrees, plots_dir / "node_degree_distribution.png")
    _plot_nearest_node_distance(all_nearest, plots_dir / "nearest_node_distance_distribution.png")
    _plot_gps_trajectories(trip_records, plots_dir / "representative_gps_trajectories.png")
    _plot_gps_route_overlay(trip_records, plots_dir / "gps_vs_reference_route.png")
    _geographic_scatter(all_lat, all_lon, plots_dir / "geographic_distribution_all_trips.png")

    # Folium maps.
    maps = _build_maps(trip_records, maps_dir)

    # Report.
    _write_report(output_dir / "eda_report.md", inventory_rows, gps_rows,
                  network_rows, route_rows, spatial_rows, maps)

    print(f"EDA complete. Outputs written to: {output_dir}")
    print(f"  Trips processed: train={len(_list_trips(data_dir,'train'))}, "
          f"student_test={len(_list_trips(data_dir,'student_test'))}")
    print(f"  CSV summaries: {output_dir}/*.csv")
    print(f"  Plots: {plots_dir}")
    print(f"  Maps: {maps_dir} ({len(maps)} maps)")
    print(f"  Report: {output_dir / 'eda_report.md'}")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the PathSnap dataset exploration and EDA."
    )
    parser.add_argument(
        "--base-dir", default=str(DEFAULT_BASE_DIR),
        help="Repository root containing the Data directory.",
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where EDA outputs are written.",
    )
    args = parser.parse_args(argv)
    run(base_dir=Path(args.base_dir), output_dir=Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
