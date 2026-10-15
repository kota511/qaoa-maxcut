"""the one cut definition every component in the project scores against"""


def cut_value(graph, assignment):
    """total weight of the edges whose endpoints sit on opposite sides"""
    total = 0
    for u, v in graph.edges():
        if assignment[u] != assignment[v]:
            total += graph[u][v].get("weight", 1)
    return total


def flip_gains(graph, assignment):
    """global cut change produced by flipping each vertex"""
    gains = [0] * len(graph)
    for u, v, data in graph.edges(data=True):
        if u == v:
            continue
        weight = data.get("weight", 1)
        contribution = weight if assignment[u] == assignment[v] else -weight
        gains[u] += contribution
        gains[v] += contribution
    return gains


def apply_flip(graph, assignment, gains, node):
    """flip one vertex in place and update the affected gains"""
    delta = gains[node]
    for neighbor, data in graph[node].items():
        if neighbor == node:
            continue
        weight = data.get("weight", 1)
        contribution = (
            weight if assignment[node] == assignment[neighbor] else -weight
        )
        # the edge swings cut to uncut or back, so the gain moves by 2x
        gains[neighbor] -= 2 * contribution
    assignment[node] = 1 - assignment[node]
    # flipping back is worth the negative of flipping forward
    gains[node] = -delta
    return delta


def build_restricted_objective(graph, assignment, nodes):
    """selected-vertex objective with every outside assignment fixed

    internal edges use local indices, boundary weights are aggregated by the
    fixed endpoint bit so exact and qaoa scoring share the same terms
    """
    nodes = tuple(nodes)
    node_to_local = {node: index for index, node in enumerate(nodes)}
    internal_edges = []
    boundary_weights = [[0, 0] for _ in nodes]
    boundary_edge_count = 0
    boundary_weight = 0

    for node, local_index in node_to_local.items():
        for neighbor, data in graph[node].items():
            weight = data.get("weight", 1)
            if neighbor in node_to_local:
                other_index = node_to_local[neighbor]
                # both endpoints selected, keep the edge once
                if local_index < other_index:
                    internal_edges.append((local_index, other_index, weight))
            else:
                boundary_weights[local_index][assignment[neighbor]] += weight
                boundary_edge_count += 1
                boundary_weight += weight

    problem = {
        "nodes": nodes,
        "current_values": tuple(assignment[node] for node in nodes),
        "internal_edges": tuple(internal_edges),
        "boundary_weights": tuple(tuple(weights) for weights in boundary_weights),
        "internal_edge_count": len(internal_edges),
        "boundary_edge_count": boundary_edge_count,
        "internal_weight": sum(weight for _, _, weight in internal_edges),
        "boundary_weight": boundary_weight,
    }
    problem["current_local_cut"] = restricted_cut_value(
        problem, problem["current_values"]
    )
    return problem


def restricted_cut_value(problem, values):
    total = 0
    for u, v, weight in problem["internal_edges"]:
        if values[u] != values[v]:
            total += weight
    # a boundary edge is cut when the two bits differ, hence the crossed lookup
    for value, (weight_to_zero, weight_to_one) in zip(
        values, problem["boundary_weights"]
    ):
        total += weight_to_one if value == 0 else weight_to_zero
    return total


def restricted_cut_delta(problem, values):
    return restricted_cut_value(problem, values) - problem["current_local_cut"]


def with_restricted_values(assignment, problem, values):
    """paste a local assignment back into its full-graph one"""
    candidate = list(assignment)
    for node, value in zip(problem["nodes"], values):
        candidate[node] = value
    return candidate
