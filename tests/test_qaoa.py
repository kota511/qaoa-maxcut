import math

import networkx as nx
import pytest

from qls_maxcut.objective import build_restricted_objective, cut_value
from qls_maxcut.qaoa import _qaoa_probabilities, qaoa_local_refinement


def test_qaoa_refinement_value_matches_assignment():
    graph = nx.cycle_graph(4)
    assignment = [0, 0, 1, 1]
    refined = qaoa_local_refinement(
        graph,
        assignment,
        nodes_to_refine=[1, 2],
    )

    assert cut_value(graph, refined["assignment"]) == refined["cut"]


def test_qaoa_refinement_keeps_other_nodes_fixed():
    graph = nx.cycle_graph(4)
    assignment = [0, 0, 1, 1]
    refined = qaoa_local_refinement(
        graph,
        assignment,
        nodes_to_refine=[1, 2],
    )

    assert refined["cut"] == 4
    assert refined["assignment"][0] == assignment[0]
    assert refined["assignment"][3] == assignment[3]


def test_qaoa_refinement_empty_nodes_returns_initial():
    graph = nx.path_graph(3)
    assignment = [0, 1, 1]
    refined = qaoa_local_refinement(
        graph,
        assignment,
        nodes_to_refine=[],
    )

    assert refined["cut"] == cut_value(graph, assignment)
    assert refined["assignment"] == assignment


def test_qaoa_sampling_is_deterministic_and_strictly_rejects_non_improvement():
    graph = nx.path_graph(3)
    assignment = [0, 1, 0]

    first = qaoa_local_refinement(
        graph, assignment, [1], seed=7, shots=64
    )
    second = qaoa_local_refinement(
        graph, assignment, [1], seed=7, shots=64
    )

    assert first["sampled_assignment"] == second["sampled_assignment"]
    assert first["accepted"] is False
    assert first["cut"] == cut_value(graph, assignment)
    assert first["assignment"] == assignment
    assert first["shots"] == 64
    assert first["parameter_evaluations"] == 81


def test_internal_hamiltonian_sign_prefers_a_cut_edge():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1)
    problem = build_restricted_objective(graph, [0, 0], [0, 1])

    probabilities = _qaoa_probabilities(problem, math.pi / 2, math.pi / 8)

    assert probabilities[1] + probabilities[2] == pytest.approx(1)

    graph[0][1]["weight"] = -1
    problem = build_restricted_objective(graph, [0, 0], [0, 1])
    probabilities = _qaoa_probabilities(problem, math.pi / 2, math.pi / 8)

    assert probabilities[0] + probabilities[3] == pytest.approx(1)


def test_boundary_hamiltonian_and_qiskit_little_endian_order():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 2, 1), (1, 3, 1)])
    problem = build_restricted_objective(graph, [0, 1, 0, 1], [0, 1])

    probabilities = _qaoa_probabilities(problem, math.pi / 2, math.pi / 4)

    # local [1, 0] is index 1, qiskit qubit 0 is the least-significant bit
    assert probabilities[1] == pytest.approx(1)
