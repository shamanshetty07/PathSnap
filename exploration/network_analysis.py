"""Road-network analysis for PathSnap EDA.

Computes per-trip statistics for the ``.nodes`` and ``.arcs`` files:

* node / arc counts
* unique nodes referenced by arcs and isolated nodes
* connected components (arcs treated as undirected)
* node degree distribution (in / out / total)
* self-loops
* duplicate directed arcs
* invalid node references in arcs
* whether the stored representation looks directed or undirected

The original dataset is never modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class NetworkStats:
    """Result of analysing a single trip's road network."""

    trip_id: str
    split: str
    num_nodes: int
    num_arcs: int
    unique_nodes_referenced: int
    isolated_nodes: int
    connected_components: int
    mean_total_degree: float
    median_total_degree: float
    max_total_degree: int
    num_zero_degree_referenced: int
    num_self_loops: int
    duplicate_directed_arcs: int
    num_duplicate_arc_pairs: int
    invalid_node_references: int
    reciprocity: float
    network_orientation: str
    load_warnings: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "split": self.split,
            "trip_id": self.trip_id,
            "num_nodes": self.num_nodes,
            "num_arcs": self.num_arcs,
            "unique_nodes_referenced": self.unique_nodes_referenced,
            "isolated_nodes": self.isolated_nodes,
            "connected_components": self.connected_components,
            "mean_total_degree": self.mean_total_degree,
            "median_total_degree": self.median_total_degree,
            "max_total_degree": self.max_total_degree,
            "num_zero_degree_referenced": self.num_zero_degree_referenced,
            "num_self_loops": self.num_self_loops,
            "duplicate_directed_arcs": self.duplicate_directed_arcs,
            "num_duplicate_arc_pairs": self.num_duplicate_arc_pairs,
            "invalid_node_references": self.invalid_node_references,
            "reciprocity": self.reciprocity,
            "network_orientation": self.network_orientation,
        }


def _union_find_components(parent_nodes, edges):
    """Count weakly connected components via union-find over node ids.

    Used as a lightweight alternative to NetworkX for very large graphs.
    """
    parent: dict[int, int] = {}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for n in parent_nodes:
        parent.setdefault(n, n)
    for a, b in edges:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)

    return len({find(n) for n in parent})


def analyze_network(trip_data) -> NetworkStats:
    """Analyse the road network of a single :class:`TripData`."""
    nodes = trip_data.nodes
    arcs = trip_data.arcs
    warnings = list(trip_data.load_warnings)

    num_nodes = int(len(nodes))
    num_arcs = int(len(arcs))

    node_ids = set(nodes.index.astype("int64").tolist())

    if num_arcs > 0:
        from_nodes = arcs["from_node"].to_numpy(dtype=np.int64)
        to_nodes = arcs["to_node"].to_numpy(dtype=np.int64)
    else:
        from_nodes = np.array([], dtype=np.int64)
        to_nodes = np.array([], dtype=np.int64)

    # Unique nodes referenced by arcs.
    referenced = set(from_nodes.tolist()) | set(to_nodes.tolist())
    unique_nodes_referenced = len(referenced)

    # Isolated nodes: present in .nodes but never referenced by any arc.
    isolated_nodes = len(node_ids - referenced)

    # Invalid node references: arcs pointing at nodes absent from .nodes.
    invalid_node_references = int(
        (~np.isin(from_nodes, list(node_ids)) if node_ids else np.ones(len(from_nodes), dtype=bool)).sum()
    ) + int(
        (~np.isin(to_nodes, list(node_ids)) if node_ids else np.ones(len(to_nodes), dtype=bool)).sum()
    )

    # Connected components (treat arcs as undirected).
    if num_arcs > 0:
        try:
            graph = nx.Graph()
            graph.add_nodes_from(referenced)
            graph.add_edges_from(zip(from_nodes.tolist(), to_nodes.tolist()))
            connected_components = int(nx.number_connected_components(graph))
        except Exception:  # pragma: no cover - fallback
            connected_components = _union_find_components(
                list(referenced), list(zip(from_nodes.tolist(), to_nodes.tolist()))
            )
    else:
        connected_components = 0

    # Node degree distribution (only for referenced nodes).
    if num_arcs > 0:
        out_degree = pd.Series(from_nodes).value_counts()
        in_degree = pd.Series(to_nodes).value_counts()
        all_referenced = pd.Index(list(referenced))
        out_degree = out_degree.reindex(all_referenced, fill_value=0)
        in_degree = in_degree.reindex(all_referenced, fill_value=0)
        total_degree = out_degree + in_degree
        total_values = total_degree.to_numpy(dtype=np.float64)
        mean_total_degree = float(np.mean(total_values))
        median_total_degree = float(np.median(total_values))
        max_total_degree = int(np.max(total_values))
        num_zero_degree_referenced = int((total_values == 0).sum())
    else:
        mean_total_degree = median_total_degree = 0.0
        max_total_degree = 0
        num_zero_degree_referenced = 0

    # Self-loops.
    if num_arcs > 0:
        num_self_loops = int((from_nodes == to_nodes).sum())
    else:
        num_self_loops = 0

    # Duplicate directed arcs: (from, to) pairs occurring more than once.
    if num_arcs > 0:
        arcs_pairs = pd.DataFrame({"f": from_nodes, "t": to_nodes})
        pair_counts = arcs_pairs.groupby(["f", "t"]).size()
        num_duplicate_arc_pairs = int((pair_counts > 1).sum())
        duplicate_directed_arcs = int((pair_counts - 1).sum())  # extra occurrences
    else:
        num_duplicate_arc_pairs = 0
        duplicate_directed_arcs = 0

    # Reciprocity: fraction of arcs (a,b) for which (b,a) also exists.
    if num_arcs > 0:
        pair_set = set(zip(from_nodes.tolist(), to_nodes.tolist()))
        reverse_present = [
            (b, a) in pair_set for a, b in zip(from_nodes.tolist(), to_nodes.tolist())
        ]
        reciprocity = float(np.mean(reverse_present))
    else:
        reciprocity = 0.0

    if reciprocity >= 0.9:
        orientation = "undirected"
    elif reciprocity <= 0.1:
        orientation = "directed"
    else:
        orientation = "mixed"

    if invalid_node_references:
        warnings.append(
            f"{invalid_node_references} arc endpoint(s) reference node IDs absent "
            "from .nodes"
        )
    if num_self_loops:
        warnings.append(f"{num_self_loops} self-loop arc(s) (from_node == to_node)")
    if duplicate_directed_arcs:
        warnings.append(
            f"{duplicate_directed_arcs} duplicate directed arc occurrence(s)"
        )

    stats = NetworkStats(
        trip_id=trip_data.trip_id,
        split=trip_data.split,
        num_nodes=num_nodes,
        num_arcs=num_arcs,
        unique_nodes_referenced=unique_nodes_referenced,
        isolated_nodes=isolated_nodes,
        connected_components=connected_components,
        mean_total_degree=mean_total_degree,
        median_total_degree=median_total_degree,
        max_total_degree=max_total_degree,
        num_zero_degree_referenced=num_zero_degree_referenced,
        num_self_loops=num_self_loops,
        duplicate_directed_arcs=duplicate_directed_arcs,
        num_duplicate_arc_pairs=num_duplicate_arc_pairs,
        invalid_node_references=invalid_node_references,
        reciprocity=reciprocity,
        network_orientation=orientation,
        load_warnings=warnings,
    )
    return stats, total_values
