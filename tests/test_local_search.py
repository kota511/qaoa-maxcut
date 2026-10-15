import math

import networkx as nx
import pytest

from qls_maxcut.local_search import (
    _refine,
    _temperature_schedule,
    random_assignment,
    run_simulated_annealing,
    run_simulated_annealing_with_qaoa_refinement,
    run_tabu_search,
    run_tabu_search_with_qaoa_refinement,
    select_refinement_nodes,
)
from qls_maxcut.brute_force import brute_force_maxcut
from qls_maxcut.objective import cut_value, flip_gains


def test_random_assignment_is_binary_and_reproducible():
    first = random_assignment(5, seed=123)
    second = random_assignment(5, seed=123)

    assert first == second
    assert len(first) == 5
    assert set(first) <= {0, 1}


@pytest.mark.parametrize(
    "search",
    (
        lambda graph, initial: run_simulated_annealing(
            graph, initial, steps=30, seed=123
        ),
        lambda graph, initial: run_tabu_search(
            graph, initial, steps=30, tabu_tenure=2
        ),
    ),
)
def test_classical_search_results_match_saved_assignments(search):
    graph = nx.cycle_graph(6)
    initial = [0, 0, 0, 1, 1, 1]

    result = search(graph, initial)

    assert initial == [0, 0, 0, 1, 1, 1]
    assert cut_value(graph, result["best_assignment"]) == result["best_cut"]
    assert cut_value(graph, result["final_assignment"]) == result["final_cut"]
    assert result["best_cut"] >= result["final_cut"]


@pytest.mark.parametrize(
    "search",
    (
        lambda graph, initial: run_simulated_annealing(
            graph, initial, steps=200, seed=7
        ),
        lambda graph, initial: run_tabu_search(
            graph, initial, steps=200, tabu_tenure=3
        ),
    ),
)
def test_classical_search_is_reproducible_under_a_fixed_seed(search):
    """the paired design assumes a seeded search is bit-for-bit repeatable"""
    graph = nx.erdos_renyi_graph(14, 0.4, seed=5)
    for u, v in sorted(graph.edges()):
        graph[u][v]["weight"] = 1 + (u + v) % 9
    initial = random_assignment(len(graph), seed=11)

    first = search(graph, list(initial))
    second = search(graph, list(initial))

    for field in ("best_cut", "final_cut", "best_iteration", "num_iterations"):
        assert first[field] == second[field]
    assert first["best_assignment"] == second["best_assignment"]
    assert first["final_assignment"] == second["final_assignment"]


@pytest.mark.parametrize(
    "graph",
    (nx.path_graph(6), nx.cycle_graph(6), nx.complete_graph(5)),
)
def test_classical_searches_reach_the_known_small_graph_optimum(graph):
    optimum, _assignment = brute_force_maxcut(graph)
    initial = random_assignment(len(graph), seed=3)

    annealing = run_simulated_annealing(graph, list(initial), steps=400, seed=3)
    tabu = run_tabu_search(graph, list(initial), steps=400, tabu_tenure=2)

    assert annealing["best_cut"] == optimum
    assert tabu["best_cut"] == optimum


def test_select_refinement_nodes_uses_distance_then_node_id():
    graph = nx.path_graph(7)

    assert select_refinement_nodes(graph, center_node=3, max_nodes=4) == [3, 2, 4, 1]


def test_adaptive_annealing_schedule_uses_documented_acceptance_rates():
    temperatures = _temperature_schedule(nx.path_graph(3), gains=[-2, 0, 2], steps=5)

    assert temperatures[0] == pytest.approx(-2 / math.log(0.8))
    assert temperatures[-1] == pytest.approx(-2 / math.log(0.01))


@pytest.mark.parametrize(
    "search",
    (
        lambda graph, initial: run_simulated_annealing_with_qaoa_refinement(
            graph,
            initial,
            steps=10,
            seed=123,
            stagnation_patience=0,
        ),
        lambda graph, initial: run_tabu_search_with_qaoa_refinement(
            graph,
            initial,
            steps=10,
            stagnation_patience=0,
        ),
    ),
)
def rejected_qaoa(graph, assignment, nodes, **_kwargs):
    value = cut_value(graph, assignment)
    return {
        "cut": value,
        "assignment": list(assignment),
        "sampled_cut": value,
        "sampled_assignment": list(assignment),
        "accepted": False,
        "gamma": 0.0,
        "beta": 0.0,
        "expected_cut": float(value),
        "shots": 64,
        "parameter_evaluations": 81,
        "runtime_seconds": 0.0,
    }


@pytest.mark.parametrize("base_search", ("sa", "tabu"))
def test_rejected_diagnostics_do_not_change_classical_trajectory(
    monkeypatch, base_search
):
    graph = nx.cycle_graph(6)
    initial = [0, 0, 0, 1, 1, 1]
    monkeypatch.setattr("qls_maxcut.local_search.qaoa_local_refinement", rejected_qaoa)

    if base_search == "sa":
        baseline = run_simulated_annealing(graph, initial, steps=30, seed=99)
        hybrid = run_simulated_annealing_with_qaoa_refinement(
            graph,
            initial,
            steps=30,
            seed=99,
            stagnation_patience=1,
            max_refinement_nodes=3,
        )
    else:
        baseline = run_tabu_search(graph, initial, steps=30, tabu_tenure=2)
        hybrid = run_tabu_search_with_qaoa_refinement(
            graph,
            initial,
            steps=30,
            tabu_tenure=2,
            stagnation_patience=1,
            max_refinement_nodes=3,
            seed=99,
        )

    assert hybrid["best_cut"] == baseline["best_cut"]
    assert hybrid["best_assignment"] == baseline["best_assignment"]
    assert hybrid["final_cut"] == baseline["final_cut"]
    assert hybrid["final_assignment"] == baseline["final_assignment"]


def test_accepted_refinement_recomputes_incremental_gains(monkeypatch):
    graph = nx.cycle_graph(4)
    assignment = [0, 0, 1, 1]
    refined = [0, 1, 0, 1]

    def accept_qaoa(_graph, _assignment, _nodes, **_kwargs):
        return {
            **rejected_qaoa(graph, refined, []),
            "cut": 4,
            "assignment": refined,
            "sampled_cut": 4,
            "sampled_assignment": refined,
            "accepted": True,
        }

    monkeypatch.setattr("qls_maxcut.local_search.qaoa_local_refinement", accept_qaoa)
    updated, value, gains, _event, _diagnostic = _refine(
        graph,
        assignment,
        cut_value(graph, assignment),
        flip_gains(graph, assignment),
        cut_value(graph, assignment),
        iteration=1,
        call_index=0,
        anchor=0,
        max_refinement_nodes=4,
        seed=1,
        shots=64,
    )

    assert value == cut_value(graph, updated) == 4
    assert gains == flip_gains(graph, updated)
