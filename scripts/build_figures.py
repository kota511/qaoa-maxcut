"""render the report result figures from the results"""

import sys
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_experiments import (
    configured_records,
    pair_records,
    sensitivity_records,
    sensitivity_summary,
)

CORE = "results/core"
SENSITIVITY = "results/sensitivity"
FIGURES = "results/figures"

BASES = ("sa", "tabu")
HOST_LABEL = {"sa": "SA", "tabu": "Tabu"}
COLOURS = {
    "blue": "#6666ff",
    "orange": "#ffa040",
    "green": "#008c00",
    "purple": "#d96693",
    "grey": "#b9b9b9",
}
PALETTE = ("blue", "orange", "green", "purple", "grey")


def commas(value):
    return f"{value:,}"


def paired_rows(records):
    """one row per setting, graph seed, run seed and host, holding the gain"""
    rows = {base: [] for base in BASES}
    for key, methods in pair_records(records).items():
        for base in BASES:
            baseline = methods[base]
            hybrid = methods[f"{base}_qaoa_refinement"]
            gain = hybrid["best_cut"] - baseline["best_cut"]
            rows[base].append(
                {
                    "study": key[0],
                    "n": key[1],
                    "p": key[2],
                    "weights": key[3],
                    "graph_seed": key[4],
                    "run_seed": key[5],
                    "gain": gain,
                    "baseline": baseline,
                    "hybrid": hybrid,
                }
            )
    return rows


def host_events(rows):
    return [event for row in rows for event in row["hybrid"]["refinement_events"]]


def event_stats(events):
    """the mean local gain each candidate reached, over one set of events"""
    return {
        "events": len(events),
        "qaoa_gain": mean(event["qaoa_gain"] for event in events),
        "uniform_gain": mean(event["uniform_gain"] for event in events),
        "exact_gain": mean(event["exact_gain"] for event in events),
    }


def setting_rows(rows):
    """paired mean gain per setting and host"""
    grouped = {}
    for base in BASES:
        for row in rows[base]:
            key = (row["study"], row["n"], row["p"], row["weights"], base)
            grouped.setdefault(key, []).append(row["gain"])
    return [
        {
            "study": key[0],
            "n": key[1],
            "p": key[2],
            "weights": key[3],
            "base": key[4],
            "pairs": len(gains),
            "mean_gain": mean(gains),
        }
        for key, gains in sorted(grouped.items())
    ]


def headroom_buckets(rows):
    """boundary to internal weight and exact gain share per setting"""
    buckets = {}
    for base in BASES:
        for row in rows[base]:
            for event in row["hybrid"]["refinement_events"]:
                if event["internal_weight"] <= 0 or event["before_cut"] <= 0:
                    continue
                entry = buckets.setdefault(
                    (row["study"], row["n"], row["p"]), {"ratio": [], "share": []}
                )
                entry["ratio"].append(
                    event["boundary_weight"] / event["internal_weight"]
                )
                entry["share"].append(event["exact_gain"] / event["before_cut"])
    return buckets


def runtime_bars(rows):
    """mean algorithm runtime for each baseline and its paired hybrid"""
    bars = []
    for base in BASES:
        for label, source in (("baseline", "baseline"), ("hybrid", "hybrid")):
            bars.append(
                (
                    f"{HOST_LABEL[base]} {label}",
                    mean(
                        row[source]["algorithm_runtime_seconds"] for row in rows[base]
                    ),
                )
            )
    return bars


def bar_panel(axis, bars, y_label, title="", precision=4, suffix="", palette=PALETTE):
    """one labelled bar chart, coloured by bar position"""
    values = [value for _, value in bars]
    positions = list(range(len(bars)))
    axis.bar(
        positions,
        values,
        width=0.7,
        color=[COLOURS[palette[index % len(palette)]] for index in positions],
    )
    for position, value in zip(positions, values):
        axis.annotate(
            f"{value:.{precision}f}{suffix}",
            (position, value),
            textcoords="offset points",
            xytext=(0, 3 if value >= 0 else -10),
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=7,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels([label.replace(", ", "\n") for label, _ in bars])
    axis.set_ylabel(y_label, fontsize=8)
    if title:
        axis.set_title(title, fontsize=9, fontweight="bold")
    if suffix:
        axis.yaxis.set_major_formatter(lambda value, _: f"{value:g}{suffix}")
    axis.axhline(0, color="black", linewidth=1.1)
    axis.grid(axis="y", color="0.85", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=7)
    axis.margins(y=0.2)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def figure_refinement_diagnostics(rows):
    figure, axes = plt.subplots(1, 2, figsize=(6.6, 3.1))
    for axis, base in zip(axes, BASES):
        stats = event_stats(host_events(rows[base]))
        bar_panel(
            axis,
            [
                ("QAOA", stats["qaoa_gain"]),
                ("Uniform", stats["uniform_gain"]),
                ("Exact", stats["exact_gain"]),
            ],
            "Mean local gain (cut units)",
            f"{HOST_LABEL[base]}, {commas(stats['events'])} events",
        )
    return figure


def figure_core_size(settings):
    figure, axis = plt.subplots(figsize=(7.0, 3.2))
    bars = []
    for n in (20, 50, 100, 250):
        for base in BASES:
            match = next(
                row
                for row in settings
                if row["study"] == "size" and row["n"] == n and row["base"] == base
            )
            bars.append((f"{HOST_LABEL[base]}, ${{n={n}}}$", match["mean_gain"]))
    bar_panel(
        axis,
        bars,
        "Mean paired difference (cut units)",
        palette=("blue", "orange"),
    )
    return figure


def figure_headroom(buckets):
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 3.1))
    bar_panel(
        axes[0],
        [
            (f"$\\rho={p}$", mean(buckets[("density", 100, p)]["ratio"]))
            for p in (0.1, 0.3, 0.6)
        ],
        "Boundary/internal weight",
        "Density at $n=100$",
        precision=2,
    )
    bar_panel(
        axes[1],
        [
            (f"$n={n}$", 100 * mean(buckets[("size", n, 8 / (n - 1))]["share"]))
            for n in (20, 50, 100, 250)
        ],
        "Exact gain, % of cut",
        r"Size at degree $\approx 8$",
        precision=2,
        suffix="%",
    )
    return figure


def figure_sensitivity(summary):
    figure, axes = plt.subplots(2, 1, figsize=(6.6, 5.4))
    for axis, base in zip(axes, BASES):
        bars = []
        for n in (100, 250):
            for k in (4, 6, 8):
                row = next(
                    item
                    for item in summary
                    if item["base_search"] == base
                    and item["n"] == n
                    and item["k"] == k
                )
                bars.append(
                    (f"$k={k}$, $n={n}$", 100 * row["mean_normalised_gain"])
                )
        bar_panel(
            axis,
            bars,
            "Normalised difference",
            HOST_LABEL[base],
            suffix="%",
            palette=("blue", "orange", "green"),
        )
    return figure


def figure_runtime(rows):
    figure, axis = plt.subplots(figsize=(6.6, 3.1))
    bar_panel(
        axis,
        runtime_bars(rows),
        "Mean runtime (s)",
        palette=("grey", "purple"),
    )
    return figure


def main():
    rows = paired_rows(configured_records(ROOT / CORE))
    new_records, reference_records = sensitivity_records(
        ROOT / SENSITIVITY, ROOT / CORE
    )
    summary = sensitivity_summary(new_records + reference_records)

    figures = {
        "core-refinement-diagnostics": figure_refinement_diagnostics(rows),
        "core-size": figure_core_size(setting_rows(rows)),
        "headroom-mechanism": figure_headroom(headroom_buckets(rows)),
        "sensitivity-k": figure_sensitivity(summary),
        "runtime-overhead": figure_runtime(rows),
    }

    directory = ROOT / FIGURES
    directory.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        path = directory / f"{name}.pdf"
        figure.tight_layout()
        
        figure.savefig(path, metadata={"CreationDate": None})
        plt.close(figure)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
