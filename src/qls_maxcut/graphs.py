"""the two instance sources, synthetic erdos-renyi and the g1 benchmark"""

import random
import hashlib
from pathlib import Path

import networkx as nx


def make_weighted_erdos_renyi_graph(
    n,
    p,
    seed=None,
    weight_low=1,
    weight_high=10,
):
    """synthetic erdos-renyi graph, each edge weighted in [weight_low, weight_high]"""
    rng = random.Random(seed)
    graph = nx.erdos_renyi_graph(n, p, seed=seed)

    # sort first, or edge order decides the weighting
    for u, v in sorted(graph.edges()):
        graph[u][v]["weight"] = rng.randint(weight_low, weight_high)

    return graph


def load_gset_graph(path, expected_sha256=None):
    """load a one-based matrix market pattern file into zero-based node labels"""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError("benchmark checksum does not match")

    graph = None
    rows = 0
    expected_edges = None
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if line.startswith("%") or not line.strip():
                continue
            fields = line.split()
            if graph is None:
                if len(fields) != 3:
                    raise ValueError("matrix market header must give rows, columns and entries")
                rows, columns, expected_edges = map(int, fields)
                if rows != columns:
                    raise ValueError("benchmark matrix must be square")
                graph = nx.Graph()
                graph.add_nodes_from(range(rows))
                continue
            if len(fields) != 2:
                raise ValueError(f"invalid entry on line {line_number}")
            one_based_u, one_based_v = map(int, fields)
            if not (1 <= one_based_u <= rows and 1 <= one_based_v <= rows):
                raise ValueError(f"node label outside 1..{rows} on line {line_number}")
            u, v = one_based_u - 1, one_based_v - 1
            if graph.has_edge(u, v):
                raise ValueError(f"duplicate edge on line {line_number}")
            graph.add_edge(u, v, weight=1)

    if graph is None:
        raise ValueError("benchmark file contains no matrix market header")
    if graph.number_of_edges() != expected_edges:
        raise ValueError(
            f"benchmark edge count mismatch: expected {expected_edges}, "
            f"found {graph.number_of_edges()}"
        )
    graph.graph["sha256"] = digest
    return graph
