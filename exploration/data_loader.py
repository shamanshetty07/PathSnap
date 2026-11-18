"""Trip data loading for the PathSnap exploration module.

This module is responsible for loading and validating the four files that
describe a single trip:

* ``.track``  - GPS observations (longitude, latitude, timestamp)
* ``.nodes``  - Road-network node coordinates (longitude, latitude)
* ``.arcs``   - Road-network edges (from_node_id, to_node_id)
* ``.route``  - Ordered sequence of reference arc IDs

The original dataset is never modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# Column names used internally. The on-disk files have no header and use a
# tab separator, so we assign these names after reading.
TRACK_COLUMNS = ["longitude", "latitude", "timestamp"]
NODES_COLUMNS = ["longitude", "latitude"]
ARCS_COLUMNS = ["from_node", "to_node"]
ROUTE_COLUMNS = ["arc_id"]

# File extensions recognised for each dataset component.
EXTENSIONS = {
    "track": ".track",
    "nodes": ".nodes",
    "arcs": ".arcs",
    "route": ".route",
}


class TripDataError(Exception):
    """Raised when a trip cannot be loaded or validated."""


@dataclass
class TripData:
    """In-memory representation of a single trip.

    Coordinates are stored exactly as found in the dataset, i.e.
    ``(longitude, latitude)``. Callers are responsible for converting to
    ``(latitude, longitude)`` before passing them to Folium.
    """

    trip_id: str
    split: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: list[int]
    base_dir: Path
    load_warnings: list[str] = field(default_factory=list)

    @property
    def data_dir(self) -> Path:
        return self.base_dir / self.split / self.trip_id


def _resolve_file(data_dir: Path, trip_id: str, kind: str) -> Optional[Path]:
    """Find the file for a dataset component.

    The repository stores files as ``<trip_id>.<ext>`` (e.g.
    ``00000001.track``) but the specification refers to them generically as
    ``.track``/``.nodes``/``.arcs``/``.route``. We accept either naming
    scheme without modifying the dataset.
    """
    extension = EXTENSIONS[kind]
    explicit = data_dir / f"{trip_id}{extension}"
    if explicit.is_file():
        return explicit
    matches = sorted(data_dir.glob(f"*{extension}"))
    if matches:
        return matches[0]
    return None


def _read_table(path: Path, names: list[str], n_expected: int) -> pd.DataFrame:
    """Read a tab-separated file with no header into a DataFrame."""
    frame = pd.read_csv(path, sep="\t", header=None, names=names, dtype=str)
    if frame.shape[1] != n_expected:
        raise TripDataError(
            f"{path.name}: expected {n_expected} columns, found {frame.shape[1]}"
        )
    return frame


def load_trip(split: str, trip_id: str, base_dir: Path | str = "Data") -> TripData:
    """Load all four files for a trip and return a :class:`TripData`.

    Missing or unreadable files are reported as warnings rather than crashing
    silently. Critical missing files (track/nodes/arcs/route) result in a
    :class:`TripDataError` so the caller can decide how to proceed.
    """
    base_dir = Path(base_dir)
    data_dir = base_dir / split / trip_id

    if not data_dir.is_dir():
        raise TripDataError(f"Trip directory not found: {data_dir}")

    warnings: list[str] = []

    # ---- .track ----
    track_path = _resolve_file(data_dir, trip_id, "track")
    if track_path is None:
        raise TripDataError(f"Missing track file in {data_dir}")
    try:
        track = _read_table(track_path, TRACK_COLUMNS, 3)
        track["longitude"] = pd.to_numeric(track["longitude"], errors="coerce")
        track["latitude"] = pd.to_numeric(track["latitude"], errors="coerce")
        track["timestamp"] = pd.to_numeric(track["timestamp"], errors="coerce")
        before = len(track)
        track = track.dropna(subset=["longitude", "latitude"])
        if len(track) < before:
            warnings.append(
                f"{track_path.name}: dropped {before - len(track)} rows with "
                "invalid coordinates"
            )
        if track.empty:
            warnings.append(f"{track_path.name}: no valid GPS observations")
    except Exception as exc:  # noqa: BLE001 - surface malformed file gracefully
        raise TripDataError(f"Failed to read track file {track_path}: {exc}") from exc

    # ---- .nodes ----
    nodes_path = _resolve_file(data_dir, trip_id, "nodes")
    if nodes_path is None:
        raise TripDataError(f"Missing nodes file in {data_dir}")
    try:
        nodes = _read_table(nodes_path, NODES_COLUMNS, 2)
        nodes["longitude"] = pd.to_numeric(nodes["longitude"], errors="coerce")
        nodes["latitude"] = pd.to_numeric(nodes["latitude"], errors="coerce")
        nodes = nodes.dropna(subset=["longitude", "latitude"])
        nodes.index.name = "node_id"
        if nodes.empty:
            warnings.append(f"{nodes_path.name}: no valid node coordinates")
    except Exception as exc:  # noqa: BLE001
        raise TripDataError(f"Failed to read nodes file {nodes_path}: {exc}") from exc

    # ---- .arcs ----
    arcs_path = _resolve_file(data_dir, trip_id, "arcs")
    if arcs_path is None:
        raise TripDataError(f"Missing arcs file in {data_dir}")
    try:
        arcs = _read_table(arcs_path, ARCS_COLUMNS, 2)
        arcs["from_node"] = pd.to_numeric(arcs["from_node"], errors="coerce", downcast="integer")
        arcs["to_node"] = pd.to_numeric(arcs["to_node"], errors="coerce", downcast="integer")
        arcs = arcs.dropna(subset=["from_node", "to_node"])
        arcs["from_node"] = arcs["from_node"].astype("int64")
        arcs["to_node"] = arcs["to_node"].astype("int64")
        arcs.index.name = "arc_id"
        if arcs.empty:
            warnings.append(f"{arcs_path.name}: no valid road arcs")
    except Exception as exc:  # noqa: BLE001
        raise TripDataError(f"Failed to read arcs file {arcs_path}: {exc}") from exc

    # ---- .route ----
    route_path = _resolve_file(data_dir, trip_id, "route")
    if route_path is None:
        raise TripDataError(f"Missing route file in {data_dir}")
    try:
        route_frame = _read_table(route_path, ROUTE_COLUMNS, 1)
        route_frame["arc_id"] = pd.to_numeric(
            route_frame["arc_id"], errors="coerce", downcast="integer"
        )
        route_frame = route_frame.dropna(subset=["arc_id"])
        route = [int(v) for v in route_frame["arc_id"].tolist()]
        if not route:
            warnings.append(f"{route_path.name}: route contains no arc IDs")
    except Exception as exc:  # noqa: BLE001
        raise TripDataError(f"Failed to read route file {route_path}: {exc}") from exc

    return TripData(
        trip_id=trip_id,
        split=split,
        track=track.reset_index(drop=True),
        nodes=nodes,
        arcs=arcs,
        route=route,
        base_dir=base_dir,
        load_warnings=warnings,
    )
