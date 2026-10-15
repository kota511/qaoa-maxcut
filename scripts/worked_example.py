"""build the fixed 12-node worked example and its three-panel figure"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qls_maxcut.brute_force import brute_force_maxcut
from qls_maxcut.graphs import make_weighted_erdos_renyi_graph
from qls_maxcut.local_search import (
    random_assignment,
    run_simulated_annealing_with_qaoa_refinement,
    stable_seed,
)
from qls_maxcut.objective import cut_value

RECORD_DIR = "results/worked_example"
FIGURE_PATH = "results/figures/worked-example.pdf"

SIDE_ZERO = "#54a24b"
SIDE_ONE = "#f58518"
INTERNAL = "#4c78a8"
BOUNDARY = "#e45756"


def graph_figure(graph, event, path):
    """the three panels, before refinement, the selected region, and after"""
    selected = set(event["nodes"])
    panels = (
        ("Assignment at stagnation", event["before_assignment"]),
        ("Selected region and boundary", event["before_assignment"]),
        ("QAOA-refined assignment", event["qaoa_assignment"]),
    )
    captions = (
        f"cut = {event['before_cut']}",
        f"{len(selected)} selected, {event['internal_edge_count']} internal and "
        f"{event['boundary_edge_count']} boundary edges",
        f"sampled cut = {event['qaoa_sampled_cut']}",
    )
    positions = nx.circular_layout(graph)

    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
    for index, (axis, (title, assignment), caption) in enumerate(
        zip(axes, panels, captions)
    ):
        region = index == 1
        edge_colours, edge_widths = [], []
        for u, v in graph.edges():
            boundary = (u in selected) != (v in selected)
            internal = u in selected and v in selected
            if region and internal:
                edge_colours.append(INTERNAL)
            elif region and boundary:
                edge_colours.append(BOUNDARY)
            else:
                edge_colours.append("#aaaaaa")
            edge_widths.append(2.4 if region and (boundary or internal) else 0.9)
        node_colours = [
            "#bbbbbb" if region and node not in selected
            else SIDE_ONE if assignment[node] else SIDE_ZERO
            for node in graph
        ]
        nx.draw_networkx_edges(
            graph, positions, ax=axis, edge_color=edge_colours, width=edge_widths
        )
        nx.draw_networkx_nodes(
            graph, positions, ax=axis, node_color=node_colours,
            edgecolors="#111111", linewidths=1.2, node_size=320,
        )
        nx.draw_networkx_labels(
            graph, positions, ax=axis, font_size=8, font_color="white"
        )
        if region:
            nx.draw_networkx_edge_labels(
                graph,
                positions,
                ax=axis,
                edge_labels={
                    (u, v): data["weight"]
                    for u, v, data in graph.edges(data=True)
                    if (u in selected) != (v in selected) or (u in selected and v in selected)
                },
                font_size=7,
            )
        axis.set_title(f"{title}\n{caption}", fontsize=9)
        axis.set_axis_off()

    figure.text(
        0.5, 0.02,
        f"green is side 0 and orange is side 1; in the middle panel "
        f"blue edges are internal and red edges cross the boundary",
        ha="center", fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    # no creation date so a rebuild produces a byte-identical file
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, metadata={"CreationDate": None})
    plt.close(figure)


def main():
    output_dir = ROOT / RECORD_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = make_weighted_erdos_renyi_graph(12, 0.3, seed=1)
    initial = random_assignment(
        12, stable_seed("initial", "worked_example", 12, 0.3, 1, 0)
    )
    result = run_simulated_annealing_with_qaoa_refinement(
        graph,
        initial,
        steps=60,
        seed=stable_seed("classical", "worked_example", 12, 0.3, 1, 0, "sa"),
        stagnation_patience=12,
        max_refinement_nodes=6,
        max_refinement_calls=3,
        qaoa_shots=64,
    )
    if not result["refinement_events"]:
        raise RuntimeError("worked example did not produce a refinement event")
    event = result["refinement_events"][0]
    optimum, optimum_assignment = brute_force_maxcut(graph)
    record = {
        "graph": {
            "n": 12,
            "p": 0.3,
            "graph_seed": 1,
            "run_seed": 0,
            "edges": [
                [u, v, data["weight"]] for u, v, data in graph.edges(data=True)
            ],
        },
        "initial_cut": cut_value(graph, initial),
        "stagnated_cut": event["before_cut"],
        "qaoa_cut": event["qaoa_sampled_cut"],
        "uniform_cut": event["uniform_cut"],
        "exact_local_cut": event["exact_cut"],
        "final_cut": result["final_cut"],
        "best_cut": result["best_cut"],
        "global_optimum": optimum,
        "global_optimum_assignment": optimum_assignment,
        "first_refinement": event,
    }
    (output_dir / "worked_example.json").write_text(
        json.dumps(record, indent=2, sort_keys=True)
    )
    graph_figure(graph, event, ROOT / FIGURE_PATH)
    print(output_dir)


if __name__ == "__main__":
    main()
