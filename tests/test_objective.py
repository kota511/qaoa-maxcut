import networkx as nx
import pytest

from qls_maxcut.objective import (
    apply_flip,
    build_restricted_objective,
    cut_value,
    flip_gains,
    restricted_cut_delta,
    with_restricted_values,
)


@pytest.mark.parametrize(
    ("graph", "assignment", "expected"),
    (
        (nx.Graph([(0, 1, {"weight": 3.5})]), [0, 1], 3.5),
        (nx.complete_graph(3), [0, 1, 0], 2),
        (nx.cycle_graph(4), [0, 1, 0, 1], 4),
    ),
)
def test_cut_value_totals_the_crossing_edge_weights(graph, assignment, expected):
    assert cut_value(graph, assignment) == expected


def test_signed_incremental_gains_match_full_cut_changes():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 4), (1, 2, -3), (0, 2, 2)])
    assignment = [0, 1, 0]
    gains = flip_gains(graph, assignment)
    current = cut_value(graph, assignment)

    for node in graph:
        candidate = list(assignment)
        candidate[node] = 1 - candidate[node]
        assert gains[node] == cut_value(graph, candidate) - current

    delta = apply_flip(graph, assignment, gains, 1)
    assert current + delta == cut_value(graph, assignment)
    assert gains == flip_gains(graph, assignment)


def test_incremental_gains_ignore_self_loops():
    graph = nx.Graph()
    graph.add_edge(0, 0, weight=5)
    graph.add_edge(0, 1, weight=2)
    assignment = [0, 0]
    gains = flip_gains(graph, assignment)

    delta = apply_flip(graph, assignment, gains, 0)

    assert delta == 2
    assert cut_value(graph, assignment) == 2
    assert gains == flip_gains(graph, assignment)


def test_restricted_delta_includes_internal_and_boundary_edges_once():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 5), (1, 2, 7), (0, 3, -2)])
    assignment = [0, 0, 1, 1]
    problem = build_restricted_objective(graph, assignment, [0, 1])

    assert problem["internal_edge_count"] == 1
    assert problem["boundary_edge_count"] == 2
    assert problem["internal_weight"] == 5
    assert problem["boundary_weight"] == 5
    for values in ([0, 0], [0, 1], [1, 0], [1, 1]):
        candidate = with_restricted_values(assignment, problem, values)
        assert restricted_cut_delta(problem, values) == (
            cut_value(graph, candidate) - cut_value(graph, assignment)
        )


