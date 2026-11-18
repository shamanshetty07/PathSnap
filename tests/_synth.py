"""Synthetic straight two-way road for feature tests (not a test module)."""
import numpy as np
import pandas as pd

from exploration.data_loader import TripData


def make_trip(n_obs=40, route=(0, 1, 2, 3), step_deg=0.0001, lat_offset=0.00002, trip_id="SYN"):
    """5 nodes 0.001 deg (~72 m) apart eastwards; arcs 0-3 eastbound, 4-7 westbound."""
    nodes = pd.DataFrame({"longitude": [8.0 + 0.001 * k for k in range(5)], "latitude": [49.0] * 5})
    nodes.index.name = "node_id"
    arcs = pd.DataFrame([(k, k + 1) for k in range(4)] + [(k + 1, k) for k in range(4)],
                        columns=["from_node", "to_node"])
    arcs.index.name = "arc_id"
    lon = 8.0 + 0.0001 + step_deg * np.arange(n_obs)
    track = pd.DataFrame({"longitude": lon, "latitude": 49.0 + lat_offset,
                          "timestamp": np.arange(1.0, n_obs + 1)})
    return TripData(trip_id=trip_id, split="synthetic", track=track, nodes=nodes, arcs=arcs,
                    route=list(route), base_dir=".")
