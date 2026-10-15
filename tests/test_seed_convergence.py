from scripts.seed_convergence import disjoint_groups, subset_spread


def _rows(gain_by_seed, runs=3):
    """one host's paired rows, a fixed gain repeated across run seeds"""
    return {
        base: [
            {"graph_seed": seed, "run_seed": run, "gain": gain}
            for seed, gain in gain_by_seed.items()
            for run in range(runs)
        ]
        for base in ("sa", "tabu")
    }


def test_band_narrows_and_closes_on_the_full_estimate():
    rows = _rows({seed: seed % 7 for seed in range(40)})
    spread = subset_spread(rows, [4, 40], draws=200)["sa"]
    small, whole = spread[0], spread[1]
    assert small["high"] - small["low"] > whole["high"] - whole["low"]
    # every graph drawn means one subset, so the band pinches shut
    assert whole["low"] == whole["high"] == whole["median"]
    assert abs(whole["median"] - sum(seed % 7 for seed in range(40)) / 40) < 1e-9


def test_disjoint_groups_partition_the_seeds_and_drop_the_remainder():
    rows = _rows({seed: float(seed) for seed in range(11)})
    means = disjoint_groups(rows, 4)["sa"]
    # eleven seeds give two whole groups of four, and the last three are dropped
    assert means == [1.5, 5.5]
