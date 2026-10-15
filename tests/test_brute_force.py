import networkx as nx
import pytest

from qls_maxcut.brute_force import brute_force_maxcut, exact_local_refinement
from qls_maxcut.objective import cut_value


def test_brute_force_weighted_edge_optimum():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=3.5)
    best_value, best_assignment = brute_force_maxcut(graph)

    assert best_value == 3.5
    assert cut_value(graph, best_assignment) == best_value


@pytest.mark.parametrize(
    ("graph", "optimum"),
    (
        (nx.path_graph(3), 2),
        (nx.complete_graph(3), 2),
        (nx.cycle_graph(4), 4),
    ),
)
def test_brute_force_finds_the_known_small_graph_optimum(graph, optimum):
    best_value, best_assignment = brute_force_maxcut(graph)

    assert best_value == optimum
    assert cut_value(graph, best_assignment) == best_value


def test_exact_local_refinement_includes_internal_and_boundary_edges():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 4), (1, 2, 7), (0, 3, -2)])
    assignment = [0, 0, 0, 1]

    result = exact_local_refinement(graph, assignment, [0, 1])

    assert result["cut"] == 9
    assert result["assignment"][2:] == assignment[2:]
    assert cut_value(graph, result["assignment"]) == result["cut"]
