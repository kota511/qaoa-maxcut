# QLS Max-Cut

A framework for testing whether depth-one QAOA refinement of a small subgraph
improves classical local search on weighted Max-Cut. Runs simulated annealing
and tabu search with and without refinement on matched graphs and seeds, and
grades each refinement against exact enumeration and a random-sampling control.

---

## Setup

Python 3.10 or later, developed and tested on 3.13.2. No pyenv required.

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest -q
```

The test suite takes a few seconds and should pass before anything else.

---

## Directory Structure

```
qls-maxcut/
│
├── src/qls_maxcut/           ← the package
│   ├── objective.py          ← cut values, flip gains, restricted objectives
│   ├── graphs.py             ← seeded synthetic graphs and the G1 loader
│   ├── brute_force.py        ← exact enumeration
│   ├── qaoa.py               ← depth-one circuits, parameter search, sampling
│   └── local_search.py       ← annealing, tabu search and the hybrids
│
├── scripts/                  ← everything runnable
├── tests/                    ← the test suite
├── data/G1.mtx               ← Gset/G1 from the SuiteSparse Matrix Collection
├── results/                  ← run records, aggregates and generated figures
└── ORIGINALITY.txt           ← authorship statement and its one exception
```

---

## Validating Results

Each command rebuilds every graph from its own seed, recomputes every saved cut,
and fails if any value disagrees with the one stored on disk.

```bash
python scripts/validate_results.py results/gset
python scripts/validate_results.py results/core
python scripts/validate_results.py results/sensitivity
```

They print nothing until they finish. Measured on 8 cores:

| command | time | expected output |
| --- | --- | --- |
| `results/gset` | 30 seconds | `{"paired_instances": 120, "valid_records": 480, "valid_refinement_events": 166}` |
| `results/core` | 5 minutes | `{"paired_instances": 4320, "valid_records": 17280, "valid_refinement_events": 14454}` |
| `results/sensitivity` | 5 minutes | `{"analysis_records": 5760, "reused_core_records": 2880, "valid_records": 2880, "valid_refinement_events": 4606}` |

Start with `results/gset`. A raised error or a non-zero exit status means a
stored value did not reproduce. The record counts are the run totals the report
states, and the 14,454 refinement events are the figure Chapter 6 reports.

`results/sensitivity` reports different fields because that study reuses the
`k=6` runs from `results/core` rather than repeating them, giving 5,760 analysis
rows from 2,880 own records and 2,880 reused.

---

## Running the Experiments

`results/` already holds every run behind the report, so this only regenerates
them from scratch and overwrites what is there.

```bash
python scripts/run_experiments.py
```

| study | output | runs |
| --- | --- | --- |
| synthetic matrix | `results/core` | 17,280 |
| G1 benchmark | `results/gset` | 480 |
| neighbourhood size | `results/sensitivity` | 2,880 |

All three run in order because the third reuses the first's baselines, and there
is no flag to run one alone. 20,640 runs total, roughly 0.8 hours of serial
compute, so **about 10 to 15 minutes of wall-clock time on 8 cores**. It uses one
process per core minus one and prints nothing until a study finishes.

Seed counts are fixed in the script, so the run reproduces the study exactly. If
it is interrupted, repeat with `--resume` to keep the valid completed runs and
execute the rest.

---

## Generating the Figures

```bash
python scripts/build_figures.py
python scripts/worked_example.py
python scripts/seed_convergence.py
```

The first two write PDFs into `results/figures`. The third prints how the
headline estimate settles as graphs are added, which justifies the graph count.
