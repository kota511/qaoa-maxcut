"""validate saved experiment json records against a rebuilt graph"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from qls_maxcut.brute_force import exact_local_refinement
from qls_maxcut.local_search import select_refinement_nodes
from qls_maxcut.objective import build_restricted_objective, cut_value
from scripts.run_experiments import (
    METHODS,
    canonical_json,
    config_seeds,
    core_specs,
    experiment_key,
    graph_for_spec,
    pair_records,
    read_records,
    sensitivity_records,
    sensitivity_specs,
)
import hashlib


def validate_record(record):
    """deliberately a second implementation of the runner's own check, not shared code"""
    spec_fields = (
        "study",
        "n",
        "p",
        "weights",
        "graph_seed",
        "run_seed",
        "method",
        "iterations",
        "patience",
        "k",
        "max_refinements",
        "shots",
    )
    spec = {name: record[name] for name in spec_fields}
    for name in ("graph_file", "graph_sha256"):
        if name in record:
            spec[name] = record[name]
    assert record["result_key"] == experiment_key(spec)
    graph = graph_for_spec(spec)
    assert record["node_count"] == graph.number_of_nodes()
    assert record["edge_count"] == graph.number_of_edges()
    assert record["initial_cut"] == cut_value(graph, record["initial_assignment"])
    assert record["best_cut"] == cut_value(graph, record["best_assignment"])
    assert record["final_cut"] == cut_value(graph, record["final_assignment"])
    assert record["best_cut"] >= record["initial_cut"]
    assert record["best_cut"] >= record["final_cut"]
    assert record["num_refinement_calls"] == len(record["refinement_events"])
    assert len(record["refinement_events"]) <= record["max_refinements"]

    for event in record["refinement_events"]:
        before = event["before_assignment"]
        nodes = event["nodes"]
        assert event["before_cut"] == cut_value(graph, before)
        assert nodes == select_refinement_nodes(graph, event["anchor"], record["k"])
        assert len(nodes) <= record["k"]
        outside = set(graph) - set(nodes)
        problem = build_restricted_objective(graph, before, nodes)
        assert event["internal_edge_count"] == problem["internal_edge_count"]
        assert event["boundary_edge_count"] == problem["boundary_edge_count"]
        assert event["internal_weight"] == problem["internal_weight"]
        assert event["boundary_weight"] == problem["boundary_weight"]

        for prefix in ("qaoa", "qaoa_sampled", "uniform", "exact"):
            assignment = event[f"{prefix}_assignment"]
            assert event[f"{prefix}_cut"] == cut_value(graph, assignment)
            assert all(assignment[node] == before[node] for node in outside)

        exact = exact_local_refinement(graph, before, nodes)
        assert event["exact_cut"] == exact["cut"]
        assert event["qaoa_cut"] >= event["before_cut"]
        assert event["uniform_cut"] >= event["before_cut"]
        assert event["exact_cut"] >= event["qaoa_cut"]
        assert event["exact_cut"] >= event["uniform_cut"]
        assert event["qaoa_accepted"] == (
            event["qaoa_sampled_cut"] > event["before_cut"]
        )
        if event["qaoa_accepted"]:
            assert event["qaoa_cut"] == event["qaoa_sampled_cut"]
        else:
            assert event["qaoa_cut"] == event["before_cut"]
            assert event["qaoa_assignment"] == before


# started alike, or the difference is not attributable to refinement
def validate_pairing(records):
    for methods in pair_records(records).values():
        assert set(methods) == set(METHODS)
        assignments = {tuple(row["initial_assignment"]) for row in methods.values()}
        assert len(assignments) == 1
        assert (
            methods["sa"]["classical_search_seed"]
            == methods["sa_qaoa_refinement"]["classical_search_seed"]
        )
        assert (
            methods["tabu"]["classical_search_seed"]
            == methods["tabu_qaoa_refinement"]["classical_search_seed"]
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    args = parser.parse_args(argv)

    with (Path(args.output_dir) / "config.json").open() as source:
        configuration = json.load(source)
    study = configuration["preset"]
    records = read_records(args.output_dir)
    actual = {record["result_key"] for record in records}

    if study == "core":
        expected = {
            experiment_key(spec)
            for spec in core_specs(*config_seeds(args.output_dir))
        }
        assert actual == expected, (
            f"coverage mismatch: missing={len(expected-actual)}, "
            f"unexpected={len(actual-expected)}"
        )

    if study == "sensitivity":
        expected = {
            experiment_key(spec)
            for spec in sensitivity_specs(*config_seeds(args.output_dir))
        }
        assert len(records) == len(expected) and actual == expected
        assert {record["k"] for record in records} == {4, 8}
        new_records, reference_records = sensitivity_records(
            args.output_dir, configuration["reference_results"]
        )
        for record in new_records + reference_records:
            validate_record(record)
        # the reused core rows are pinned by hash, so they cannot have drifted
        reference_digest = hashlib.sha256(
            canonical_json(
                sorted(reference_records, key=lambda row: row["result_key"])
            ).encode()
        ).hexdigest()
        assert configuration["reference_records_sha256"] == reference_digest
        result = {
            "valid_records": len(records),
            "reused_core_records": len(reference_records),
            "analysis_records": len(new_records) + len(reference_records),
        }
    else:
        for record in records:
            validate_record(record)
        validate_pairing(records)
        result = {
            "valid_records": len(records),
            "paired_instances": len(records) // len(METHODS),
        }

    result["valid_refinement_events"] = sum(
        len(record["refinement_events"]) for record in records
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
