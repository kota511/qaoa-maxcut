from scripts.build_figures import event_stats, headroom_buckets, setting_rows


def _row(study, n, p, gain, events=()):
    return {
        "study": study,
        "n": n,
        "p": p,
        "weights": "uniform_1_10",
        "gain": gain,
        "baseline": {"best_cut": 100, "algorithm_runtime_seconds": 0.01},
        "hybrid": {"best_cut": 100 + gain, "refinement_events": list(events)},
    }


def test_setting_rows_average_the_gains_within_a_setting():
    rows = {
        "sa": [_row("size", 20, 0.4, 4), _row("size", 20, 0.4, 6)],
        "tabu": [_row("size", 20, 0.4, 1)],
    }
    means = {(row["base"], row["n"]): row["mean_gain"] for row in setting_rows(rows)}
    assert means[("sa", 20)] == 5
    assert means[("tabu", 20)] == 1


def test_event_stats_average_each_candidate_separately():
    events = [
        {"qaoa_gain": 2, "uniform_gain": 1, "exact_gain": 4},
        {"qaoa_gain": 4, "uniform_gain": 3, "exact_gain": 6},
    ]
    stats = event_stats(events)
    assert stats["events"] == 2
    assert (stats["qaoa_gain"], stats["uniform_gain"], stats["exact_gain"]) == (3, 2, 5)


def test_headroom_skips_events_with_nothing_to_divide_by():
    usable = {
        "internal_weight": 4,
        "boundary_weight": 8,
        "before_cut": 50,
        "exact_gain": 5,
    }
    degenerate = {**usable, "internal_weight": 0}
    rows = {"sa": [_row("density", 100, 0.3, 0, [usable, degenerate])], "tabu": []}
    entry = headroom_buckets(rows)[("density", 100, 0.3)]
    # the degenerate event contributes to neither series, so both stay length one
    assert entry["ratio"] == [2.0]
    assert entry["share"] == [0.1]
