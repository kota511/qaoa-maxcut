"""show how the paired estimate settles as independent graphs are added"""

import random
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_experiments import (
    configured_records,
    pair_records,
    paired_bootstrap_interval,
    print_table,
)
from qls_maxcut.local_search import stable_seed

BASES = ("sa", "tabu")


CORE = "results/core"


def paired_rows(records):
    """one row per setting, graph seed, run seed and host, holding the gain"""
    rows = {base: [] for base in BASES}
    for key, methods in pair_records(records).items():
        for base in BASES:
            baseline = methods[base]
            hybrid = methods[f"{base}_qaoa_refinement"]
            rows[base].append(
                {
                    "graph_seed": key[4],
                    "run_seed": key[5],
                    "gain": hybrid["best_cut"] - baseline["best_cut"],
                }
            )
    return rows


def prefix_table(rows, counts):
    out = []
    for base in BASES:
        for m in counts:
            subset = [row for row in rows[base] if row["graph_seed"] < m]
            gains = [row["gain"] for row in subset]
            interval = paired_bootstrap_interval(
                subset, "gain", stable_seed("converge", base, str(m))
            )
            out.append(
                {
                    "base": base,
                    "graphs": m,
                    "pairs": len(gains),
                    "mean": mean(gains),
                    "low": interval[0],
                    "high": interval[1],
                    "width": interval[1] - interval[0],
                }
            )
    return out


def disjoint_groups(rows, size):
    """spread of the estimate across non-overlapping groups of graphs"""
    out = {}
    for base in BASES:
        seeds = sorted({row["graph_seed"] for row in rows[base]})
        groups = [seeds[i : i + size] for i in range(0, len(seeds), size)]
        groups = [group for group in groups if len(group) == size]
        means = []
        for group in groups:
            gains = [row["gain"] for row in rows[base] if row["graph_seed"] in group]
            means.append(mean(gains))
        out[base] = means
    return out


def subset_spread(rows, counts, draws=2000):
    """what a design of m graphs would have concluded, over many draws"""
    out = {}
    for base in BASES:
        totals = {}
        for row in rows[base]:
            carried = totals.setdefault(row["graph_seed"], [0.0, 0])
            carried[0] += row["gain"]
            carried[1] += 1
        seeds = sorted(totals)
        series = []
        for m in counts:
            rng = random.Random(stable_seed("subset", base, str(m)))
            means = []
            for _ in range(draws):
                chosen = rng.sample(seeds, m)
                gained = sum(totals[seed][0] for seed in chosen)
                paired = sum(totals[seed][1] for seed in chosen)
                means.append(gained / paired)
            means.sort()
            series.append(
                {
                    "graphs": m,
                    "low": means[int(0.10 * (draws - 1))],
                    "median": means[(draws - 1) // 2],
                    "high": means[int(0.90 * (draws - 1))],
                }
            )
        out[base] = series
    return out


def main():
    records = configured_records(ROOT / CORE)
    rows = paired_rows(records)
    available = len({row["graph_seed"] for row in rows["sa"]})
    counts = sorted(
        {3, 5, 10, available}
        | {step for step in range(15, available + 1, max(5, available // 8))}
    )
    group_sizes = sorted({3, 5, 10, max(20, available // 4)})

    print(f"seed convergence over {available} graph seeds\n")
    print_table(
        ("host", "graphs", "pairs", "gain", "low", "high", "width"),
        [
            (
                row["base"].upper(),
                row["graphs"],
                row["pairs"],
                f"{row['mean']:.4f}",
                f"{row['low']:.4f}",
                f"{row['high']:.4f}",
                f"{row['width']:.4f}",
            )
            for row in prefix_table(rows, counts)
        ],
    )

    print("\nspread across disjoint groups of graphs\n")
    print_table(
        ("host", "size", "groups", "lowest", "highest", "sd"),
        [
            (
                base.upper(),
                size,
                len(means),
                f"{min(means):.4f}",
                f"{max(means):.4f}",
                f"{stdev(means):.4f}",
            )
            for size in group_sizes
            if size <= available
            for base, means in disjoint_groups(rows, size).items()
            if len(means) > 1
        ],
    )

    print("\nspread across random subsets of graphs\n")
    print_table(
        ("host", "graphs", "low", "median", "high"),
        [
            (
                base.upper(),
                point["graphs"],
                f"{point['low']:.4f}",
                f"{point['median']:.4f}",
                f"{point['high']:.4f}",
            )
            for base, series in subset_spread(rows, counts).items()
            for point in series
        ],
    )


if __name__ == "__main__":
    main()
