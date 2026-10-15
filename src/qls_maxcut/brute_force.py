"""exhaustive baselines, to grade the operator rather than compete with it"""

from itertools import product
from qls_maxcut.objective import (
    build_restricted_objective,
    cut_value,
    restricted_cut_delta,
    with_restricted_values,
)


def brute_force_maxcut(graph):
    """best cut value and assignment over every 0/1 assignment"""
    # 2^n, so worked example and tests only
    best_value = -float("inf")
    best_assignment = None

    for assignment in product([0, 1], repeat=len(graph)):
        value = cut_value(graph, assignment)
        if value > best_value:
            best_value = value
            best_assignment = list(assignment)

    return best_value, best_assignment


def exact_local_refinement(graph, assignment, nodes_to_refine):
    """optimise the selected vertices exactly, outside bits fixed

    the ceiling qaoa is measured against, same objective searched in full
    """
    problem = build_restricted_objective(graph, assignment, nodes_to_refine)
    current_value = cut_value(graph, assignment)
    best_value = current_value
    best_assignment = list(assignment)
    best_values = problem["current_values"]

    for values in product([0, 1], repeat=len(problem["nodes"])):
        value = current_value + restricted_cut_delta(problem, values)
        if value > best_value:
            best_value = value
            best_values = values
            best_assignment = with_restricted_values(assignment, problem, values)

    # the local score is an increment on the full cut, so the two must agree
    if cut_value(graph, best_assignment) != best_value:
        raise AssertionError("restricted objective disagrees with full cut value")

    return {
        "cut": best_value,
        "assignment": best_assignment,
        "values": list(best_values),
        "problem": problem,
    }
