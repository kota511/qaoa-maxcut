import json

import pytest

from scripts.run_experiments import (
    METHODS,
    aggregate,
    atomic_json,
    core_specs,
    experiment_key,
    graph_for_spec,
    gset_specs,
    paired_bootstrap_interval,
    prepare_output,
    require_result_coverage,
    result_path,
    run_and_save,
    run_method,
    sensitivity_specs,
    valid_result,
)


def tiny_spec():
    return {
        "study": "test",
        "n": 4,
        "p": 0.5,
        "weights": "unit",
        "graph_seed": 0,
        "run_seed": 0,
        "method": "sa",
        "iterations": 2,
        "patience": 4,
        "k": 4,
        "max_refinements": 0,
        "shots": 2,
    }


def _scientific_fields(record):
    """drop timestamps and wall-clock fields, they cannot be deterministic"""
    if isinstance(record, dict):
        return {
            key: _scientific_fields(value)
            for key, value in record.items()
            if key != "completed_utc" and not key.endswith("_seconds")
        }
    if isinstance(record, list):
        return [_scientific_fields(item) for item in record]
    return record


@pytest.mark.parametrize(
    "method", ("sa_qaoa_refinement", "tabu_qaoa_refinement")
)
def test_run_method_is_deterministic_for_a_repeated_specification(method):
    spec = {
        **tiny_spec(),
        "n": 12,
        "p": 0.4,
        "weights": "uniform_1_10",
        "method": method,
        "iterations": 60,
        "patience": 3,
        "k": 4,
        "max_refinements": 2,
        "shots": 8,
    }

    first = run_method(spec)
    second = run_method(spec)

    assert first["refinement_events"], "expected the refinement path to be exercised"
    assert _scientific_fields(first) == _scientific_fields(second)


def test_core_matrix_is_uniquely_keyed_and_scales_with_its_seeds():
    # the design product, not a literal, so raising the seed count still checks out
    for graph_seeds, run_seeds in (((0, 1, 2), (0, 1, 2)), (tuple(range(7)), (0, 1))):
        specs = list(core_specs(graph_seeds, run_seeds))
        settings = {(spec["study"], spec["n"], spec["p"], spec["weights"]) for spec in specs}
        expected = len(settings) * len(graph_seeds) * len(run_seeds) * len(METHODS)

        assert len(specs) == expected
        assert len({experiment_key(spec) for spec in specs}) == expected


def test_atomic_result_and_invalid_partial_are_resumable(tmp_path):
    spec = tiny_spec()
    output = tmp_path / "study"
    prepare_output(output, False, {"test": True})

    run_and_save(spec, output)
    path = result_path(output, spec)
    assert valid_result(path, spec)

    path.write_text("{")
    assert not valid_result(path, spec)


def test_resume_rejects_a_complete_record_with_a_corrupted_cut(tmp_path):
    spec = tiny_spec()
    output = tmp_path / "study"
    prepare_output(output, False, {"test": True})
    run_and_save(spec, output)
    path = result_path(output, spec)
    record = json.loads(path.read_text())
    record["best_cut"] += 1
    path.write_text(json.dumps(record))

    assert not valid_result(path, spec)


def test_resume_requires_matching_specifications(tmp_path):
    output = tmp_path / "study"
    saved = {"schema_version": 1, "specifications_sha256": "abc", "workers": 7}
    prepare_output(output, False, saved)

    assert prepare_output(output, True, saved) == output
    try:
        prepare_output(output, True, {**saved, "specifications_sha256": "def"})
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("a different specification set was accepted")


def test_resume_ignores_fields_outside_the_run_identity(tmp_path):
    output = tmp_path / "study"
    identity = {"schema_version": 1, "specifications_sha256": "abc"}
    prepare_output(output, False, {**identity, "workers": 7, "retired_key": "x"})

    # a config an older runner wrote, or one written on a machine with a
    # different core count, still describes the same set of runs
    assert prepare_output(output, True, {**identity, "workers": 3}) == output


def test_weight_conditions_share_initial_and_classical_seeds():
    unit = tiny_spec()
    weighted = {**unit, "study": "density", "weights": "uniform_1_10"}

    unit_result = run_method(unit)
    weighted_result = run_method(weighted)

    assert unit_result["initial_seed"] == weighted_result["initial_seed"]
    assert unit_result["classical_search_seed"] == weighted_result["classical_search_seed"]
    assert unit_result["initial_assignment"] == weighted_result["initial_assignment"]


def test_unknown_method_and_weight_model_are_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        run_method({**tiny_spec(), "method": "typo"})
    with pytest.raises(ValueError, match="unknown weight model"):
        run_method({**tiny_spec(), "weights": "typo"})


def test_aggregation_rejects_unexpected_and_missing_result_keys(tmp_path):
    spec = tiny_spec()
    output = tmp_path / "study"
    configuration = {"specifications": [spec]}
    prepare_output(output, False, configuration)
    run_and_save(spec, output)
    stale = {"schema_version": 1, "result_key": "stale"}
    atomic_json(output / "runs" / "stale.json", stale)

    with pytest.raises(ValueError, match=r"unexpected=1"):
        aggregate(output)
    with pytest.raises(ValueError, match=r"missing=1"):
        require_result_coverage([], [spec])


def test_gset_specs_store_a_portable_path_and_load_the_benchmark():
    spec = next(gset_specs("data/G1.mtx", (0,)))

    assert spec["graph_file"] == "data/G1.mtx"
    graph = graph_for_spec(spec)
    assert graph.number_of_nodes() == 800
    assert graph.graph["sha256"] == spec["graph_sha256"]


def test_sensitivity_matrix_is_uniquely_keyed_and_scales_with_its_seeds():
    for graph_seeds, run_seeds in (((0, 1, 2), (0, 1, 2)), (tuple(range(7)), (0, 1))):
        specs = list(sensitivity_specs(graph_seeds, run_seeds))
        # two sizes, two extension values of k, two hybrid methods
        expected = 2 * 2 * 2 * len(graph_seeds) * len(run_seeds)

        assert len(specs) == expected
        assert len({experiment_key(spec) for spec in specs}) == expected
    assert {spec["k"] for spec in specs} == {4, 8}
    assert {spec["method"] for spec in specs} == {
        "sa_qaoa_refinement",
        "tabu_qaoa_refinement",
    }


def test_hierarchical_bootstrap_is_deterministic():
    rows = [
        {"graph_seed": graph_seed, "run_seed": run_seed, "gain": graph_seed + run_seed}
        for graph_seed in range(3)
        for run_seed in range(3)
    ]

    first = paired_bootstrap_interval(rows, "gain", seed=123, replicates=1000)
    second = paired_bootstrap_interval(rows, "gain", seed=123, replicates=1000)

    assert first == second
    assert first[0] <= 2 <= first[1]
