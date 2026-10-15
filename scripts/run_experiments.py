"""resumable experiment runner and result aggregator"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx

from qls_maxcut.graphs import load_gset_graph, make_weighted_erdos_renyi_graph
from qls_maxcut.local_search import (
    random_assignment,
    run_simulated_annealing,
    run_simulated_annealing_with_qaoa_refinement,
    run_tabu_search,
    run_tabu_search_with_qaoa_refinement,
    stable_seed,
)
from qls_maxcut.objective import cut_value


METHODS = ("sa", "sa_qaoa_refinement", "tabu", "tabu_qaoa_refinement")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def experiment_key(spec):
    """identity of one run, used as its filename and to check a resume"""
    fields = (
        ("study", spec["study"]),
        ("n", spec["n"]),
        ("p", f"{spec['p']:.8g}"),
        ("weights", spec["weights"]),
        ("gs", spec["graph_seed"]),
        ("rs", spec["run_seed"]),
        ("method", spec["method"]),
        ("k", spec["k"]),
    )
    return "__".join(f"{name}={value}" for name, value in fields)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(value, output, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def graph_for_spec(spec):
    """rebuild a specs graph, generated from its seed or loaded"""
    weight_model = spec["weights"]
    if weight_model not in {"unit", "uniform_1_10", "gset"}:
        raise ValueError(f"unknown weight model: {weight_model}")
    if "graph_file" in spec:
        return load_gset_graph(
            ROOT / spec["graph_file"], spec["graph_sha256"]
        )
    if weight_model == "gset":
        raise ValueError("gset weight model requires graph_file")
    weight_low, weight_high = (1, 1) if weight_model == "unit" else (1, 10)
    return make_weighted_erdos_renyi_graph(
        spec["n"],
        spec["p"],
        seed=spec["graph_seed"],
        weight_low=weight_low,
        weight_high=weight_high,
    )


# one run of one method, seeded so a hybrid and its baseline start alike
def run_method(spec):
    method = spec["method"]
    if method in {"sa", "sa_qaoa_refinement"}:
        base = "sa"
    elif method in {"tabu", "tabu_qaoa_refinement"}:
        base = "tabu"
    else:
        raise ValueError(f"unknown method: {method}")

    graph = graph_for_spec(spec)
    # no method in the key so all four share a start and a random stream
    # matching topologies share starts and decisions across weight conditions
    pairing = (spec["n"], spec["p"], spec["graph_seed"], spec["run_seed"])
    initial_seed = stable_seed("initial", *pairing)
    initial_assignment = random_assignment(spec["n"], initial_seed)
    search_seed = stable_seed("classical", *pairing, base)
    common = {
        "graph": graph,
        "initial_assignment": initial_assignment,
        "steps": spec["iterations"],
    }

    if method == "sa":
        result = run_simulated_annealing(**common, seed=search_seed)
    elif method == "sa_qaoa_refinement":
        result = run_simulated_annealing_with_qaoa_refinement(
            **common,
            seed=search_seed,
            stagnation_patience=spec["patience"],
            max_refinement_nodes=spec["k"],
            max_refinement_calls=spec["max_refinements"],
            qaoa_shots=spec["shots"],
        )
    elif method == "tabu":
        result = run_tabu_search(
            **common, tabu_tenure=math.ceil(math.sqrt(spec["n"]))
        )
    elif method == "tabu_qaoa_refinement":
        result = run_tabu_search_with_qaoa_refinement(
            **common,
            tabu_tenure=math.ceil(math.sqrt(spec["n"])),
            stagnation_patience=spec["patience"],
            max_refinement_nodes=spec["k"],
            max_refinement_calls=spec["max_refinements"],
            qaoa_shots=spec["shots"],
            seed=search_seed,
        )
    else:
        raise ValueError(f"unknown method: {method}")

    initial_cut = cut_value(graph, initial_assignment)
    if cut_value(graph, result["best_assignment"]) != result["best_cut"]:
        raise AssertionError("saved best assignment does not match best cut")
    if cut_value(graph, result["final_assignment"]) != result["final_cut"]:
        raise AssertionError("saved final assignment does not match final cut")

    key = experiment_key(spec)
    record = {
        "schema_version": 1,
        "result_key": key,
        **spec,
        "base_search": base,
        "initial_seed": initial_seed,
        "classical_search_seed": search_seed,
        "initial_assignment": initial_assignment,
        "initial_cut": initial_cut,
        "improvement_over_initial": result["best_cut"] - initial_cut,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "actual_density": nx.density(graph),
        "average_degree": (
            2 * graph.number_of_edges() / graph.number_of_nodes()
            if graph.number_of_nodes()
            else 0
        ),
        **result,
    }
    return record


def result_path(output_dir, spec):
    return Path(output_dir) / "runs" / f"{experiment_key(spec)}.json"


def valid_result(path, spec):
    try:
        with Path(path).open() as source:
            record = json.load(source)
        return valid_record(record, spec)
    except (OSError, ValueError):
        return False


def valid_record(record, spec):
    """a saved run counts as done only if it rebuilds and still agrees"""
    if not (
        record["schema_version"] == 1
        and record["result_key"] == experiment_key(spec)
        and all(record[name] == value for name, value in spec.items())
    ):
        return False

    graph = graph_for_spec(spec)
    cut_pairs = (
        ("initial_assignment", "initial_cut"),
        ("best_assignment", "best_cut"),
        ("final_assignment", "final_cut"),
    )
    if any(
        cut_value(graph, record[assignment]) != record[value]
        for assignment, value in cut_pairs
    ):
        return False
    if (
        record["node_count"] != graph.number_of_nodes()
        or record["edge_count"] != graph.number_of_edges()
        or record["improvement_over_initial"]
        != record["best_cut"] - record["initial_cut"]
        or record["best_cut"] < max(record["initial_cut"], record["final_cut"])
        or record["num_refinement_calls"] != len(record["refinement_events"])
        or len(record["refinement_events"]) > spec["max_refinements"]
    ):
        return False

    for event in record["refinement_events"]:
        event_pairs = (
            ("before_assignment", "before_cut"),
            ("qaoa_assignment", "qaoa_cut"),
            ("qaoa_sampled_assignment", "qaoa_sampled_cut"),
            ("exact_assignment", "exact_cut"),
            ("uniform_assignment", "uniform_cut"),
        )
        if any(
            cut_value(graph, event[assignment]) != event[value]
            for assignment, value in event_pairs
        ):
            return False
        accepted = event["qaoa_sampled_cut"] > event["before_cut"]
        if (
            event["qaoa_accepted"] != accepted
            or event["qaoa_cut"]
            != (event["qaoa_sampled_cut"] if accepted else event["before_cut"])
        ):
            return False
    return True


def run_and_save(spec, output_dir):
    record = run_method(spec)
    atomic_json(result_path(output_dir, spec), record)
    return record["result_key"], record["wall_runtime_seconds"]


def core_specs(graph_seeds, run_seeds):
    """for the synthetic studies one axis varied at a time"""
    settings = []
    # degree pinned at eight, so size changes without density changing with it
    for n in (20, 50, 100, 250):
        settings.append(("size", n, 8 / (n - 1), "uniform_1_10"))
    for n in (100, 250):
        for probability in (0.1, 0.3, 0.6):
            settings.append(("density", n, probability, "uniform_1_10"))
    # reuses the density study's topologies, so weight is the only thing varying
    for n in (100, 250):
        settings.append(("weight", n, 0.3, "unit"))

    for study, n, probability, weights in settings:
        for graph_seed in graph_seeds:
            for run_seed in run_seeds:
                for method in METHODS:
                    yield {
                        "study": study,
                        "n": n,
                        "p": probability,
                        "weights": weights,
                        "graph_seed": graph_seed,
                        "run_seed": run_seed,
                        "method": method,
                        "iterations": 5 * n,
                        "patience": n,
                        "k": 6,
                        "max_refinements": 3,
                        "shots": 64,
                    }


def sensitivity_specs(graph_seeds, run_seeds):
    """hybrids at k=4 and k=8 only, the rest reused from the size study"""
    for n in (100, 250):
        for graph_seed in graph_seeds:
            for run_seed in run_seeds:
                for k in (4, 8):
                    for method in (
                        "sa_qaoa_refinement",
                        "tabu_qaoa_refinement",
                    ):
                        yield {
                            "study": "size",
                            "n": n,
                            "p": 8 / (n - 1),
                            "weights": "uniform_1_10",
                            "graph_seed": graph_seed,
                            "run_seed": run_seed,
                            "method": method,
                            "iterations": 5 * n,
                            "patience": n,
                            "k": k,
                            "max_refinements": 3,
                            "shots": 64,
                        }


def gset_specs(path, run_seeds):
    """the external benchmark, one graph with many run seeds. the file digest travels in every spec, so a swapped file fails validation"""
    path = ROOT / path
    graph = load_gset_graph(path)
    for run_seed in run_seeds:
        for method in METHODS:
            yield {
                "study": "gset_g1",
                "n": graph.number_of_nodes(),
                "p": nx.density(graph),
                "weights": "gset",
                "graph_seed": 0,
                "run_seed": run_seed,
                "method": method,
                "iterations": 5 * graph.number_of_nodes(),
                "patience": graph.number_of_nodes(),
                "k": 6,
                "max_refinements": 3,
                "shots": 64,
                "graph_file": str(path.relative_to(ROOT)),
                "graph_sha256": graph.graph["sha256"],
            }


def read_records(output_dir):
    records = []
    for path in sorted((Path(output_dir) / "runs").glob("*.json")):
        try:
            with path.open() as source:
                record = json.load(source)
            if isinstance(record, dict) and record.get("schema_version") == 1:
                records.append(record)
        except (OSError, ValueError):
            continue
    return records


def require_result_coverage(records, specs):
    """every planned run present exactly once and stops partial results directory being summarised as a complete one"""
    specs = list(specs)
    expected_by_key = {experiment_key(spec): spec for spec in specs}
    if len(expected_by_key) != len(specs):
        raise ValueError("result specifications contain duplicate keys")

    actual_keys = [record.get("result_key") for record in records]
    actual = set(actual_keys)
    expected = set(expected_by_key)
    missing = expected - actual
    unexpected = actual - expected
    duplicates = len(actual_keys) - len(actual)
    if missing or unexpected or duplicates:
        raise ValueError(
            "result coverage mismatch: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"duplicate_keys={duplicates}"
        )
    # index by key scanning the record list per spec is quadratic
    record_by_key = {record["result_key"]: record for record in records}
    invalid = [
        key
        for key, spec in expected_by_key.items()
        if not valid_record(record_by_key[key], spec)
    ]
    if invalid:
        raise ValueError(f"result validation failed for {len(invalid)} record(s)")
    return records


def configured_records(output_dir):
    output_dir = Path(output_dir)
    with (output_dir / "config.json").open() as source:
        configuration = json.load(source)
    return require_result_coverage(
        read_records(output_dir), configuration["specifications"]
    )


def config_seeds(output_dir):
    """seeds the study ran with so counts are not hardcoded"""
    with (Path(output_dir) / "config.json").open() as source:
        configuration = json.load(source)
    return tuple(configuration["graph_seeds"]), tuple(configuration["run_seeds"])


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            cells = {}
            for field in fields:
                value = row.get(field, "")
                cells[field] = (
                    canonical_json(value) if isinstance(value, (list, dict)) else value
                )
            writer.writerow(cells)


def pair_records(records):
    """group records by setting, graph seed and run seed, keyed by method"""
    grouped = {}
    for record in records:
        key = (
            record["study"],
            record["n"],
            record["p"],
            record["weights"],
            record["graph_seed"],
            record["run_seed"],
        )
        grouped.setdefault(key, {})[record["method"]] = record
    return grouped


def paired_summary(records):
    grouped = {}
    for key, methods in pair_records(records).items():
        for base, hybrid in (
            ("sa", "sa_qaoa_refinement"),
            ("tabu", "tabu_qaoa_refinement"),
        ):
            if base not in methods or hybrid not in methods:
                continue
            group = key[:4] + (base,)
            grouped.setdefault(group, []).append(
                (
                    methods[base],
                    methods[hybrid],
                    methods[hybrid]["best_cut"] - methods[base]["best_cut"],
                )
            )

    rows = []
    for (study, n, probability, weights, base), pairs in sorted(grouped.items()):
        gains = [gain for _, _, gain in pairs]
        hybrid_events = [
            event for _, hybrid, _ in pairs for event in hybrid["refinement_events"]
        ]
        rows.append(
            {
                "study": study,
                "n": n,
                "p": probability,
                "weights": weights,
                "base_search": base,
                "paired_runs": len(pairs),
                "mean_baseline_best_cut": float(
                    mean([baseline["best_cut"] for baseline, _, _ in pairs])
                ),
                "mean_hybrid_best_cut": float(
                    mean([hybrid["best_cut"] for _, hybrid, _ in pairs])
                ),
                "mean_paired_gain": float(mean(gains)),
                "hybrid_better": sum(gain > 0 for gain in gains),
                "hybrid_same": sum(gain == 0 for gain in gains),
                "hybrid_worse": sum(gain < 0 for gain in gains),
                "mean_baseline_runtime_seconds": mean(
                    [baseline["algorithm_runtime_seconds"] for baseline, _, _ in pairs]
                ),
                "mean_hybrid_runtime_seconds": mean(
                    [hybrid["algorithm_runtime_seconds"] for _, hybrid, _ in pairs]
                ),
                "mean_exact_local_gain": (
                    mean([event["exact_gain"] for event in hybrid_events])
                    if hybrid_events
                    else 0
                ),
                "mean_qaoa_gain": (
                    mean([event["qaoa_gain"] for event in hybrid_events])
                    if hybrid_events
                    else 0
                ),
                "mean_uniform_gain": (
                    mean([event["uniform_gain"] for event in hybrid_events])
                    if hybrid_events
                    else 0
                ),
                "refinement_events": len(hybrid_events),
            }
        )
    return rows


def sensitivity_records(output_dir, reference_dir):
    new_records = configured_records(output_dir)
    reference_records = [
        record
        for record in configured_records(ROOT / reference_dir)
        if record["study"] == "size"
        and record["n"] in (100, 250)
        and record["k"] == 6
    ]
    if len(reference_records) != len(new_records):
        raise ValueError("sensitivity coverage requires one referenced core record per new record")

    graph_seeds, run_seeds = config_seeds(output_dir)
    expected_new = {
        experiment_key(spec)
        for spec in sensitivity_specs(graph_seeds, run_seeds)
    }
    if {record["result_key"] for record in new_records} != expected_new:
        raise ValueError("new sensitivity result keys are incomplete or unexpected")

    grouped = {}
    for record in reference_records + new_records:
        key = (record["n"], record["graph_seed"], record["run_seed"])
        grouped.setdefault(key, []).append(record)
    if len(grouped) != 2 * len(graph_seeds) * len(run_seeds):
        raise ValueError("sensitivity reference pairing is incomplete")
    for records in grouped.values():
        baselines = {record["method"]: record for record in records if record["method"] in {"sa", "tabu"}}
        hybrids = [record for record in records if record["method"].endswith("_qaoa_refinement")]
        if set(baselines) != {"sa", "tabu"} or len(hybrids) != 6:
            raise ValueError("each sensitivity pair requires two baselines and six hybrids")
        for hybrid in hybrids:
            baseline = baselines[hybrid["base_search"]]
            if (
                hybrid["initial_assignment"] != baseline["initial_assignment"]
                or hybrid["classical_search_seed"]
                != baseline["classical_search_seed"]
                or hybrid["edge_count"] != baseline["edge_count"]
            ):
                raise ValueError("sensitivity rows are not paired to the core baseline")
    return new_records, reference_records


def paired_bootstrap_interval(rows, value_key, seed, replicates=10_000):
    """resample graph seeds, then the run seeds nested in each"""
    by_graph = {}
    for row in rows:
        by_graph.setdefault(row["graph_seed"], {}).setdefault(
            row["run_seed"], []
        ).append(row[value_key])
    graph_seeds = sorted(by_graph)
    run_seed_counts = {len(by_graph[item]) for item in graph_seeds}
    if not graph_seeds or len(run_seed_counts) != 1:
        raise ValueError("bootstrap requires the same run seeds under every graph seed")
    values_by_graph = {
        graph_seed: [
            value for values in by_graph[graph_seed].values() for value in values
        ]
        for graph_seed in graph_seeds
    }

    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        sample = []
        for graph_seed in rng.choices(graph_seeds, k=len(graph_seeds)):
            run_values = values_by_graph[graph_seed]
            sample.extend(rng.choices(run_values, k=len(run_values)))
        estimates.append(mean(sample))
    estimates.sort()
    return (
        estimates[int(0.025 * (replicates - 1))],
        estimates[int(0.975 * (replicates - 1))],
    )


def sensitivity_summary(records):
    baselines = {}
    hybrids = {}
    for record in records:
        key = (record["n"], record["graph_seed"], record["run_seed"])
        if record["method"] in {"sa", "tabu"}:
            baselines[key + (record["method"],)] = record
        else:
            hybrids[key + (record["base_search"], record["k"])] = record

    grouped = {}
    for key, hybrid in hybrids.items():
        n, graph_seed, run_seed, base, k = key
        baseline = baselines[(n, graph_seed, run_seed, base)]
        gain = hybrid["best_cut"] - baseline["best_cut"]
        grouped.setdefault((base, n, k), []).append(
            {
                "graph_seed": graph_seed,
                "run_seed": run_seed,
                "gain": gain,
                "normalised_gain": gain / baseline["best_cut"],
                "baseline": baseline,
                "hybrid": hybrid,
            }
        )

    summary = []
    for (base, n, k), pairs in sorted(grouped.items()):
        gains = [pair["gain"] for pair in pairs]
        normalised = [pair["normalised_gain"] for pair in pairs]
        events = [
            event
            for pair in pairs
            for event in pair["hybrid"]["refinement_events"]
        ]
        positive_headroom = [event for event in events if event["exact_gain"] > 0]
        gain_ci = paired_bootstrap_interval(
            pairs, "gain", stable_seed("bootstrap", base, n, k, "gain")
        )
        normalised_ci = paired_bootstrap_interval(
            pairs,
            "normalised_gain",
            stable_seed("bootstrap", base, n, k, "normalised"),
        )
        summary.append(
            {
                "base_search": base,
                "n": n,
                "k": k,
                "paired_runs": len(pairs),
                "mean_gain": mean(gains),
                "median_gain": median(gains),
                "gain_stddev": stdev(gains),
                "gain_ci_low": gain_ci[0],
                "gain_ci_high": gain_ci[1],
                "mean_normalised_gain": mean(normalised),
                "normalised_ci_low": normalised_ci[0],
                "normalised_ci_high": normalised_ci[1],
                "wins": sum(value > 0 for value in gains),
                "ties": sum(value == 0 for value in gains),
                "losses": sum(value < 0 for value in gains),
                "refinement_events": len(events),
                "accepted_refinements": sum(
                    event["qaoa_accepted"] for event in events
                ),
                "refinement_success_rate": (
                    mean([event["qaoa_accepted"] for event in events])
                    if events
                    else None
                ),
                "mean_actual_k": (
                    mean([len(event["nodes"]) for event in events])
                    if events
                    else None
                ),
                "mean_exact_gain": (
                    mean([event["exact_gain"] for event in events])
                    if events
                    else None
                ),
                "mean_qaoa_efficiency": (
                    mean(
                        [
                            event["qaoa_gain"] / event["exact_gain"]
                            for event in positive_headroom
                        ]
                    )
                    if positive_headroom
                    else None
                ),
                "mean_qaoa_gain": (
                    mean([event["qaoa_gain"] for event in events])
                    if events
                    else None
                ),
                "mean_uniform_gain": (
                    mean([event["uniform_gain"] for event in events])
                    if events
                    else None
                ),
                "qaoa_uniform_wins": sum(
                    event["qaoa_gain"] > event["uniform_gain"] for event in events
                ),
                "qaoa_uniform_ties": sum(
                    event["qaoa_gain"] == event["uniform_gain"] for event in events
                ),
                "qaoa_uniform_losses": sum(
                    event["qaoa_gain"] < event["uniform_gain"] for event in events
                ),
                "mean_algorithm_runtime_seconds": mean(
                    [pair["hybrid"]["algorithm_runtime_seconds"] for pair in pairs]
                ),
                "mean_wall_time_to_best_seconds": mean(
                    [pair["hybrid"]["time_to_best_seconds"] for pair in pairs]
                ),
            }
        )
    return summary


def print_table(headings, rows):
    widths = [max(len(str(cell)) for cell in column) for column in zip(headings, *rows)]
    for line in (headings, *rows):
        print("  ".join(str(cell).ljust(width) for cell, width in zip(line, widths)).rstrip())


def event_rows(records, identity_fields):
    """one row per refinement event, tagged with the run fields that locate it"""
    rows = []
    for record in records:
        for event in record.get("refinement_events", []):
            rows.append(
                {field: record[field] for field in identity_fields} | event
            )
    return rows


def aggregate(output_dir):
    output_dir = Path(output_dir)
    records = configured_records(output_dir)
    run_fields = [key for key in records[0] if key != "refinement_events"]
    write_csv(output_dir / "runs.csv", records, run_fields)

    events = event_rows(
        records,
        ("result_key", "study", "n", "p", "weights", "graph_seed", "run_seed", "method"),
    )
    write_csv(output_dir / "refinements.csv", events, list(events[0]))

    summaries = paired_summary(records)
    write_csv(output_dir / "summary.csv", summaries, list(summaries[0]))
    print_table(
        ("study", "n", "p", "weights", "host", "pairs", "baseline", "hybrid",
         "gain", "W/T/L"),
        [
            (
                row["study"],
                row["n"],
                f"{row['p']:.4f}",
                row["weights"],
                row["base_search"],
                row["paired_runs"],
                f"{row['mean_baseline_best_cut']:.2f}",
                f"{row['mean_hybrid_best_cut']:.2f}",
                f"{row['mean_paired_gain']:.3f}",
                f"{row['hybrid_better']}/{row['hybrid_same']}/{row['hybrid_worse']}",
            )
            for row in summaries
        ],
    )
    return records, summaries


def aggregate_sensitivity(output_dir, reference_dir):
    output_dir = Path(output_dir)
    new_records, reference_records = sensitivity_records(output_dir, reference_dir)
    combined = reference_records + new_records
    source_by_key = {
        record["result_key"]: "new" for record in new_records
    } | {
        record["result_key"]: "reused_core" for record in reference_records
    }
    combined_rows = [
        {**record, "record_source": source_by_key[record["result_key"]]}
        for record in combined
    ]
    run_fields = [
        key for key in combined_rows[0] if key != "refinement_events"
    ]
    write_csv(output_dir / "runs.csv", combined_rows, run_fields)

    events = event_rows(
        [row for row in combined_rows if row["method"].endswith("_qaoa_refinement")],
        ("result_key", "record_source", "n", "k", "graph_seed", "run_seed", "method"),
    )
    write_csv(output_dir / "refinements.csv", events, list(events[0]))

    summary = sensitivity_summary(combined)
    write_csv(output_dir / "summary.csv", summary, list(summary[0]))
    print_table(
        ("host", "n", "k", "pairs", "gain", "exact gain", "W/T/L"),
        [
            (
                row["base_search"],
                row["n"],
                row["k"],
                row["paired_runs"],
                f"{row['mean_gain']:.3f}",
                "n/a" if row["mean_exact_gain"] is None else f"{row['mean_exact_gain']:.3f}",
                f"{row['wins']}/{row['ties']}/{row['losses']}",
            )
            for row in summary
        ],
    )
    return combined, summary


def environment_manifest():
    root = Path(__file__).resolve().parents[1]
    source_files = [
        root / "pyproject.toml",
        *sorted((root / "src").rglob("*.py")),
        *sorted((root / "scripts").glob("*.py")),
    ]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("networkx", "qiskit")
        },
        "initial_command": [sys.executable, *sys.argv],
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
    }


def prepare_output(path, resume, configuration):
    path = Path(path)
    config_path = path / "config.json"
    # a resume that reaches a study the interrupted run never started begins that study instead of refusing
    if resume and config_path.is_file():
        with config_path.open() as source:
            saved = json.load(source)
        # the spec hash covers every run this directory is meant to hold, so
        # machine details and keys an older runner wrote do not block a resume
        identity = ("schema_version", "specifications_sha256", "reference_records_sha256")
        if any(saved.get(name) != configuration.get(name) for name in identity):
            raise ValueError("saved configuration does not match this invocation")
    else:
        if path.exists() and any(path.iterdir()):
            raise ValueError("refusing to reuse a non-empty directory without --resume")
        path.mkdir(parents=True, exist_ok=True)
        atomic_json(config_path, configuration)
        atomic_json(path / "environment.json", environment_manifest())
    (path / "runs").mkdir(exist_ok=True)
    return path


def run_specs(specs, output_dir, workers, aggregate_results=True):
    specs = list(specs)
    pending = [
        spec for spec in specs if not valid_result(result_path(output_dir, spec), spec)
    ]
    print(f"coverage: {len(specs)-len(pending)}/{len(specs)} complete; {len(pending)} pending")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_and_save, spec, str(output_dir)) for spec in pending]
        for index, future in enumerate(as_completed(futures), 1):
            key, seconds = future.result()
            print(f"[{index}/{len(pending)}] {key} ({seconds:.2f}s)", flush=True)
    records = read_records(output_dir)
    if aggregate_results:
        records, _summaries = aggregate(output_dir)
    else:
        require_result_coverage(records, specs)
    return len(records)


GRAPH_SEEDS = tuple(range(120))
RUN_SEEDS = (0, 1, 2)
GSET_RUN_SEEDS = tuple(range(120))
GSET_PATH = "data/G1.mtx"
STUDIES = ("core", "gset", "sensitivity")


def study_specs(study):
    if study == "core":
        return list(core_specs(GRAPH_SEEDS, RUN_SEEDS))
    if study == "gset":
        return list(gset_specs(GSET_PATH, GSET_RUN_SEEDS))
    return list(sensitivity_specs(GRAPH_SEEDS, RUN_SEEDS))


def reference_configuration(reference_results):
    """the core rows the sensitivity study reuses, pinned by hash"""
    reference_specs = [
        spec
        for spec in core_specs(*config_seeds(reference_results))
        if spec["study"] == "size" and spec["n"] in (100, 250)
    ]
    reference_records = [
        record
        for record in configured_records(reference_results)
        if record["study"] == "size" and record["n"] in (100, 250) and record["k"] == 6
    ]
    if len(reference_records) != len(reference_specs):
        raise ValueError(
            f"reference core must contain {len(reference_specs)} matching "
            "size-study rows"
        )
    return {
        "reference_results": str(Path(reference_results).relative_to(ROOT)),
        "reference_records_sha256": hashlib.sha256(
            canonical_json(
                sorted(reference_records, key=lambda row: row["result_key"])
            ).encode()
        ).hexdigest(),
    }


def run_study(study, workers, resume):
    specs = study_specs(study)
    configuration = {
        "schema_version": 1,
        "preset": study,
        "graph_seeds": [0] if study == "gset" else list(GRAPH_SEEDS),
        "run_seeds": list(GSET_RUN_SEEDS if study == "gset" else RUN_SEEDS),
        "methods": list(METHODS),
        "specifications_sha256": hashlib.sha256(
            canonical_json(specs).encode()
        ).hexdigest(),
        "specification_count": len(specs),
        "specifications": specs,
    }
    if study == "sensitivity":
        configuration.update(reference_configuration(ROOT / "results" / "core"))

    output_dir = ROOT / "results" / study
    prepare_output(output_dir, resume, configuration)
    total = run_specs(
        specs, output_dir, workers, aggregate_results=study != "sensitivity"
    )
    if study == "sensitivity":
        aggregate_sensitivity(output_dir, configuration["reference_results"])
    print(f"{study}: {total} valid records in {output_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    # leave a core free so the machine stays usable during a long run
    workers = max(1, (os.cpu_count() or 2) - 1)
    for study in STUDIES:
        run_study(study, workers, args.resume)


if __name__ == "__main__":
    main()
