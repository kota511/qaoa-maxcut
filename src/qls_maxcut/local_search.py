"""the two classical searches and the refinement step they share"""

import hashlib
import math
import random
import time
from statistics import median

import networkx as nx

from qls_maxcut.brute_force import exact_local_refinement
from qls_maxcut.objective import (
    apply_flip,
    build_restricted_objective,
    cut_value,
    flip_gains,
    restricted_cut_delta,
    with_restricted_values,
)
from qls_maxcut.qaoa import qaoa_local_refinement


def stable_seed(*parts):
    """reproducible seed from a tuple of run identifiers"""
    # python salts hash() per process
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def random_assignment(n, seed=None):
    rng = random.Random(seed)
    return [rng.randrange(2) for _ in range(n)]


def select_refinement_nodes(graph, center_node, max_nodes):
    """deterministic connected region of at most max_nodes vertices"""
    distances = nx.single_source_shortest_path_length(graph, center_node)
    return [
        node
        for node, _distance in sorted(
            distances.items(), key=lambda item: (item[1], item[0])
        )[:max_nodes]
    ]


def _temperature_schedule(graph, gains, steps):
    """geometric cooling, scaled so the schedule suits the graph it runs on"""
    nonzero_gains = [abs(gain) for gain in gains if gain]
    if nonzero_gains:
        scale = median(nonzero_gains)
    else:
        nonzero_weights = [
            abs(data.get("weight", 1))
            for _, _, data in graph.edges(data=True)
            if data.get("weight", 1)
        ]
        scale = median(nonzero_weights) if nonzero_weights else 1

    # bounds set by acceptance rate, 80% down to 1%
    initial = -scale / math.log(0.8)
    final = -scale / math.log(0.01)
    if steps <= 1:
        return [initial] * steps
    ratio = (final / initial) ** (1 / (steps - 1))
    return [initial * ratio**iteration for iteration in range(steps)]


def _uniform_local_diagnostic(graph, assignment, nodes, seed, shots):
    """sample the local subgraph uniformly, returning the best of the shots (same window as qaoa)"""
    problem = build_restricted_objective(graph, assignment, nodes)
    current_value = cut_value(graph, assignment)
    rng = random.Random(seed)
    best_value = current_value
    best_values = list(problem["current_values"])

    for _ in range(shots):
        values = [rng.randrange(2) for _node in nodes]
        value = current_value + restricted_cut_delta(problem, values)
        if value > best_value:
            best_value = value
            best_values = values

    best_assignment = with_restricted_values(assignment, problem, best_values)
    if cut_value(graph, best_assignment) != best_value:
        raise AssertionError("uniform diagnostic disagrees with full cut value")
    return best_value, best_assignment


# one refinement event, qaoa plus the two diagnostics that grade it
def _refine(
    graph,
    assignment,
    current_value,
    gains,
    best_value,
    iteration,
    call_index,
    anchor,
    max_refinement_nodes,
    seed,
    shots,
):
    nodes = select_refinement_nodes(graph, anchor, max_refinement_nodes)
    before_assignment = list(assignment)
    before_value = current_value

    qaoa = qaoa_local_refinement(
        graph,
        before_assignment,
        nodes,
        seed=stable_seed("qaoa", seed, call_index),
        shots=shots,
    )

    # diagnostics only record what was reachable, and are timed out of runtime
    diagnostic_started = time.perf_counter()
    exact = exact_local_refinement(graph, before_assignment, nodes)
    exact_seconds = time.perf_counter() - diagnostic_started

    uniform_started = time.perf_counter()
    uniform_cut, uniform_assignment = _uniform_local_diagnostic(
        graph,
        before_assignment,
        nodes,
        stable_seed("uniform", seed, call_index),
        shots,
    )
    uniform_seconds = time.perf_counter() - uniform_started
    diagnostic_seconds = exact_seconds + uniform_seconds

    if qaoa["accepted"]:
        assignment = list(qaoa["assignment"])
        current_value = qaoa["cut"]
        # several vertices moved at once, so no incremental update applies
        gains = flip_gains(graph, assignment)

    problem = exact["problem"]
    event = {
        "iteration": iteration,
        "call_index": call_index,
        "anchor": anchor,
        "nodes": nodes,
        "before_cut": before_value,
        "best_cut_before": best_value,
        "before_assignment": before_assignment,
        "qaoa_cut": qaoa["cut"],
        "qaoa_sampled_cut": qaoa["sampled_cut"],
        "qaoa_assignment": qaoa["assignment"],
        "qaoa_sampled_assignment": qaoa["sampled_assignment"],
        "qaoa_gain": qaoa["cut"] - before_value,
        "qaoa_sampled_gain": qaoa["sampled_cut"] - before_value,
        "qaoa_accepted": qaoa["accepted"],
        "exact_cut": exact["cut"],
        "exact_assignment": exact["assignment"],
        "exact_gain": exact["cut"] - before_value,
        "uniform_cut": uniform_cut,
        "uniform_assignment": uniform_assignment,
        "uniform_gain": uniform_cut - before_value,
        "gamma": qaoa["gamma"],
        "beta": qaoa["beta"],
        "expected_cut": qaoa["expected_cut"],
        "shots": qaoa["shots"],
        "parameter_evaluations": qaoa["parameter_evaluations"],
        "internal_edge_count": problem["internal_edge_count"],
        "boundary_edge_count": problem["boundary_edge_count"],
        "internal_weight": problem["internal_weight"],
        "boundary_weight": problem["boundary_weight"],
        "qaoa_runtime_seconds": qaoa["runtime_seconds"],
        "exact_runtime_seconds": exact_seconds,
        "uniform_runtime_seconds": uniform_seconds,
        "diagnostic_runtime_seconds": diagnostic_seconds,
    }
    return assignment, current_value, gains, event, diagnostic_seconds


def _result(
    assignment,
    current_value,
    best_assignment,
    best_value,
    best_iteration,
    best_wall_time,
    best_algorithm_time,
    iterations_done,
    refinement_events,
    started,
    diagnostic_seconds,
):
    wall_seconds = time.perf_counter() - started
    # algorithm time excludes the diagnostics, which are measurement not method
    return {
        "best_cut": best_value,
        "best_assignment": best_assignment,
        "final_cut": current_value,
        "final_assignment": list(assignment),
        "best_iteration": best_iteration,
        "time_to_best_seconds": best_wall_time,
        "algorithm_time_to_best_seconds": best_algorithm_time,
        "num_iterations": iterations_done,
        "num_refinement_calls": len(refinement_events),
        "subgraph_size": max(
            (len(event["nodes"]) for event in refinement_events), default=0
        ),
        "algorithm_runtime_seconds": wall_seconds - diagnostic_seconds,
        "wall_runtime_seconds": wall_seconds,
        "diagnostic_runtime_seconds": diagnostic_seconds,
        "refinement_events": refinement_events,
    }

def run_simulated_annealing(
    graph,
    initial_assignment,
    steps,
    seed=None,
    stagnation_patience=None,
    max_refinement_nodes=6,
    max_refinement_calls=3,
    qaoa_shots=64,
):
    started = time.perf_counter()
    rng = random.Random(seed)
    assignment = list(initial_assignment)
    current_value = cut_value(graph, assignment)
    gains = flip_gains(graph, assignment)
    temperatures = _temperature_schedule(graph, gains, steps)
    best_assignment = list(assignment)
    best_value = current_value
    best_iteration = 0
    best_wall_time = 0.0
    best_algorithm_time = 0.0
    iterations_done = 0
    stagnation = 0
    diagnostic_seconds = 0.0
    events = []

    for iteration, temperature in enumerate(temperatures, start=1):
        # worsening flips get likelier to be refused as it cools
        node = rng.randrange(len(graph))
        delta = gains[node]
        if delta > 0 or (
            temperature > 0 and rng.random() < math.exp(delta / temperature)
        ):
            current_value += apply_flip(graph, assignment, gains, node)

        iterations_done = iteration
        if current_value > best_value:
            best_value = current_value
            best_assignment = list(assignment)
            best_iteration = iteration
            best_wall_time = time.perf_counter() - started
            best_algorithm_time = best_wall_time - diagnostic_seconds
            stagnation = 0
        else:
            stagnation += 1

        # a baseline has no patience and never reaches here
        if (
            stagnation_patience
            and stagnation >= stagnation_patience
            and len(events) < max_refinement_calls
        ):
            # best single flip anchors it, window grows out by graph distance
            anchor = max(range(len(graph)), key=lambda item: (gains[item], -item))
            assignment, current_value, gains, event, diagnostic = _refine(
                graph,
                assignment,
                current_value,
                gains,
                best_value,
                iteration,
                len(events),
                anchor,
                max_refinement_nodes,
                seed,
                qaoa_shots,
            )
            diagnostic_seconds += diagnostic
            events.append(event)
            stagnation = 0
            if current_value > best_value:
                best_value = current_value
                best_assignment = list(assignment)
                best_iteration = iteration
                best_wall_time = time.perf_counter() - started
                best_algorithm_time = best_wall_time - diagnostic_seconds

    return _result(
        assignment,
        current_value,
        best_assignment,
        best_value,
        best_iteration,
        best_wall_time,
        best_algorithm_time,
        iterations_done,
        events,
        started,
        diagnostic_seconds,
    )


# the hybrids only add a patience
def run_simulated_annealing_with_qaoa_refinement(
    graph,
    initial_assignment,
    steps,
    seed=None,
    stagnation_patience=5,
    max_refinement_nodes=6,
    max_refinement_calls=3,
    qaoa_shots=64,
):
    return run_simulated_annealing(
        graph,
        initial_assignment,
        steps,
        seed,
        stagnation_patience,
        max_refinement_nodes,
        max_refinement_calls,
        qaoa_shots,
    )


# same shape and same patience convention as the annealing search above
def run_tabu_search(
    graph,
    initial_assignment,
    steps,
    tabu_tenure=2,
    stagnation_patience=None,
    max_refinement_nodes=6,
    max_refinement_calls=3,
    qaoa_shots=64,
    seed=None,
):
    started = time.perf_counter()
    assignment = list(initial_assignment)
    current_value = cut_value(graph, assignment)
    gains = flip_gains(graph, assignment)
    best_assignment = list(assignment)
    best_value = current_value
    best_iteration = 0
    best_wall_time = 0.0
    best_algorithm_time = 0.0
    tabu_nodes = []
    iterations_done = 0
    stagnation = 0
    diagnostic_seconds = 0.0
    events = []

    for iteration in range(1, steps + 1):
        # aspiration- a tabu vertex is allowed if it would beat the stored best
        candidates = [
            node
            for node in range(len(graph))
            if node not in tabu_nodes or current_value + gains[node] > best_value
        ]
        if not candidates:
            break
        # always moves, even downhill which is how it leaves a local optimum
        node = max(candidates, key=lambda item: (gains[item], -item))
        current_value += apply_flip(graph, assignment, gains, node)
        tabu_nodes.append(node)
        if len(tabu_nodes) > tabu_tenure:
            tabu_nodes.pop(0)

        iterations_done = iteration
        if current_value > best_value:
            best_value = current_value
            best_assignment = list(assignment)
            best_iteration = iteration
            best_wall_time = time.perf_counter() - started
            best_algorithm_time = best_wall_time - diagnostic_seconds
            stagnation = 0
        else:
            stagnation += 1

        # a baseline has no patience and never reaches here
        if (
            stagnation_patience
            and stagnation >= stagnation_patience
            and len(events) < max_refinement_calls
        ):
            # best single flip anchors it, window grows out by graph distance
            anchor = max(range(len(graph)), key=lambda item: (gains[item], -item))
            assignment, current_value, gains, event, diagnostic = _refine(
                graph,
                assignment,
                current_value,
                gains,
                best_value,
                iteration,
                len(events),
                anchor,
                max_refinement_nodes,
                seed,
                qaoa_shots,
            )
            diagnostic_seconds += diagnostic
            events.append(event)
            stagnation = 0
            if current_value > best_value:
                best_value = current_value
                best_assignment = list(assignment)
                best_iteration = iteration
                best_wall_time = time.perf_counter() - started
                best_algorithm_time = best_wall_time - diagnostic_seconds

    return _result(
        assignment,
        current_value,
        best_assignment,
        best_value,
        best_iteration,
        best_wall_time,
        best_algorithm_time,
        iterations_done,
        events,
        started,
        diagnostic_seconds,
    )


def run_tabu_search_with_qaoa_refinement(
    graph,
    initial_assignment,
    steps,
    tabu_tenure=2,
    stagnation_patience=5,
    max_refinement_nodes=6,
    max_refinement_calls=3,
    qaoa_shots=64,
    seed=None,
):
    return run_tabu_search(
        graph,
        initial_assignment,
        steps,
        tabu_tenure,
        stagnation_patience,
        max_refinement_nodes,
        max_refinement_calls,
        qaoa_shots,
        seed,
    )
