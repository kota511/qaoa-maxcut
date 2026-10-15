"""depth-one qaoa on a selected neighbourhood, ideal statevector"""

import math
import random
import time

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qls_maxcut.objective import (
    build_restricted_objective,
    cut_value,
    restricted_cut_delta,
    with_restricted_values,
)


# 81-point grid over half of each period, so a better pair may lie outside it
DEFAULT_GAMMAS = [i * math.pi / 8 for i in range(9)]
DEFAULT_BETAS = [i * math.pi / 16 for i in range(9)]


def _qaoa_probabilities(problem, gamma, beta):
    """ideal p=1 probabilities for one restricted max-cut objective"""
    num_qubits = len(problem["nodes"])
    circuit = QuantumCircuit(num_qubits)
    circuit.h(range(num_qubits))

    # negated angles since max-cut maximises and the hamiltonian minimises
    for u, v, weight in problem["internal_edges"]:
        circuit.rzz(-gamma * weight, u, v)

    for qubit, (weight_to_zero, weight_to_one) in enumerate(
        problem["boundary_weights"]
    ):
        # sum w*z_fixed over the fixed neighbours, commuting rz terms combine
        boundary_bias = weight_to_zero - weight_to_one
        if boundary_bias:
            circuit.rz(-gamma * boundary_bias, qubit)

    # qiskit rx rotates by theta, the mixer by 2*beta
    for qubit in range(num_qubits):
        circuit.rx(2 * beta, qubit)

    state = Statevector.from_instruction(circuit)
    return [abs(amplitude) ** 2 for amplitude in state.data]


def qaoa_local_refinement(
    graph,
    assignment,
    nodes_to_refine,
    seed=None,
    shots=64,
):
    """refine the selected vertices with ideal p=1 parameters and finite sampling"""
    started = time.perf_counter()
    nodes_to_refine = list(nodes_to_refine)

    current_value = cut_value(graph, assignment)
    # nothing selected means nothing to refine
    if not nodes_to_refine:
        details = {
            "cut": current_value,
            "assignment": list(assignment),
            "sampled_cut": current_value,
            "sampled_assignment": list(assignment),
            "sampled_values": [],
            "accepted": False,
            "gamma": 0.0,
            "beta": 0.0,
            "expected_cut": float(current_value),
            "shots": shots,
            "parameter_evaluations": 0,
            "runtime_seconds": time.perf_counter() - started,
        }
        return details

    problem = build_restricted_objective(graph, assignment, nodes_to_refine)
    # qiskit puts qubit 0 in the lowest bit, so pull bit q out for vertex q instead of reading the index as a left-to-right bitstring
    local_values = [
        [(index >> qubit) & 1 for qubit in range(len(nodes_to_refine))]
        for index in range(2 ** len(nodes_to_refine))
    ]
    cut_values = [
        current_value + restricted_cut_delta(problem, values)
        for values in local_values
    ]

    # exact expectation per pair, affordable only because the window is small
    best_expected_value = -float("inf")
    best_probabilities = None
    best_gamma = None
    best_beta = None

    for gamma in DEFAULT_GAMMAS:
        for beta in DEFAULT_BETAS:
            probabilities = _qaoa_probabilities(problem, gamma, beta)
            expected_value = sum(
                probability * value
                for probability, value in zip(probabilities, cut_values)
            )
            if expected_value > best_expected_value:
                best_expected_value = expected_value
                best_probabilities = probabilities
                best_gamma = gamma
                best_beta = beta

    # parameters chosen on expectations and candidate drawn from finite shots
    sampled_indices = random.Random(0 if seed is None else seed).choices(
        range(len(local_values)),
        weights=best_probabilities,
        k=shots,
    )
    # best of the shots, ties to the lowest index so the seed decides everything
    best_index = max(
        sampled_indices,
        key=lambda index: (cut_values[index], -index),
    )
    sampled_values = local_values[best_index]
    sampled_cut = cut_values[best_index]
    sampled_assignment = with_restricted_values(
        assignment, problem, sampled_values
    )

    if cut_value(graph, sampled_assignment) != sampled_cut:
        raise AssertionError("QAOA local objective disagrees with full cut value")

    # refinement never moves the search backwards
    accepted = sampled_cut > current_value
    result_cut = sampled_cut if accepted else current_value
    result_assignment = sampled_assignment if accepted else list(assignment)
    details = {
        "cut": result_cut,
        "assignment": result_assignment,
        "sampled_cut": sampled_cut,
        "sampled_assignment": sampled_assignment,
        "sampled_values": sampled_values,
        "accepted": accepted,
        "gamma": best_gamma,
        "beta": best_beta,
        "expected_cut": best_expected_value,
        "shots": shots,
        "parameter_evaluations": len(DEFAULT_GAMMAS) * len(DEFAULT_BETAS),
        "runtime_seconds": time.perf_counter() - started,
    }
    return details
