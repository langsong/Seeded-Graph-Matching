"""Real-graph check: does skipping hubs help GOAT on the Enron email network?

Parent graph: Enron employees (utils/enron.mat, 184 people x 187 weeks), with an
edge where two people emailed in at least 6 of weeks 101-160. This is September's
E4 "sub" parent: 174 people, mean degree 11.9, highest degree 59.

Pairs: A and B each keep every parent edge independently with probability KEEP,
and B's vertices are shuffled. KEEP = 0.75 as in September (little noise) and
0.60 (more noise). 20 pairs per noise level.

Rules: top degree, hub-free top degree (top degree without its first 10 picks)
and random.

Algorithms: GOAT, with FAQ and percolation for contrast.
- GOAT: regulariser 300, as for the heavy-tailed synthetic models.
- FAQ: 3 restarts, keeping the run with the best objective.
- Percolation: one run, with round 1's settings.

The graphs are small, so every budget is run. The outcome is accuracy, the
share of non-seeds matched correctly: some vertices lose all their edges in A or
B and cannot be matched, so 90% success is often out of reach.

Run from the repository root (ds environment), about 10 minutes:

    python -m selection_rule_comparison.real_graph_check

or call run_all() from real_graph_check.ipynb. Each pair is saved when it
finishes, so rerunning resumes.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import tempfile
import time
import traceback
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache"))

import numpy as np
import pandas as pd

from experiment_common.provenance import thread_limit

from .. import run_comparison as rc
from .. import settings as S
from ..strategies import GraphA

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]  # selection_rule_comparison/ (this file is in archive/)
REPO_ROOT = EXPERIMENT_DIR.parent
ENRON_FILE = REPO_ROOT / "utils" / "enron.mat"
OUTPUT_DIR = EXPERIMENT_DIR / "data" / "real_graph"

WEEKS = (101, 160)                  # one-based, inclusive
MIN_WEEKS = 6                       # an edge needs emails in at least this many of those weeks
KEEPS = (0.60, 0.75)                # chance that each parent edge is kept in A (and, independently, in B)
N_PAIRS = 20
BUDGETS = (1, 2, 4, 6, 8, 12, 16, 24)
RULES = ("top_degree", "hub_free_top_degree", "random")
ALGORITHMS = ("goat", "faq", "percolation")
GOAT_REGULARIZER = 300
FAQ_RESTARTS = 3
SEED_BASE = 20261028

LABELS = {"top_degree": "top degree", "hub_free_top_degree": "hub-free top degree", "random": "random"}
ALGORITHM_LABELS = {"goat": "GOAT", "faq": "SGM (FAQ)", "percolation": "Percolation"}


def _now() -> str:
    return datetime.datetime.now().astimezone().strftime("%H:%M:%S")


# --------------------------------------------------------------------------
# Graphs
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def parent_graph() -> np.ndarray:
    """The Enron parent graph (boolean adjacency of the people with at least one edge)."""

    from scipy.io import loadmat

    weekly = loadmat(ENRON_FILE)["AAA"].astype(bool)
    counts = weekly[:, :, WEEKS[0] - 1:WEEKS[1]].astype(int).sum(axis=2)
    counts = counts + counts.T
    np.fill_diagonal(counts, 0)
    parent = counts >= MIN_WEEKS
    active = np.flatnonzero(parent.sum(axis=1) > 0)
    return parent[np.ix_(active, active)]


def make_pair(keep: float, pair_index: int):
    """A and B keep each parent edge independently with probability `keep`; B is shuffled."""

    parent = parent_graph()
    n = parent.shape[0]
    rng = np.random.default_rng([SEED_BASE, int(round(keep * 100)), pair_index])
    upper = np.triu_indices(n, 1)
    edges = parent[upper]

    def child():
        kept = edges & (rng.random(edges.shape) < keep)
        M = np.zeros((n, n), dtype=float)
        M[upper[0][kept], upper[1][kept]] = 1.0
        return M + M.T

    A, B_aligned = child(), child()
    shuffle = rng.permutation(n)
    B = B_aligned[np.ix_(shuffle, shuffle)]
    truth = np.argsort(shuffle)
    return A, B, truth, rng


def orderings(A: np.ndarray, rng: np.random.Generator) -> dict:
    graph = GraphA(A)
    by_degree = graph.degree_order(rng)
    return {"top_degree": by_degree, "hub_free_top_degree": by_degree[S.HUB_COUNT:],
            "random": rng.permutation(graph.n)}


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

def _evaluate(algorithm: str, A, B, truth, seeds, solver_seed: int) -> dict:
    if algorithm == "percolation":
        from experiment_common.percolation import run_percolation_once
        out = run_percolation_once(A, B, truth, seeds, algorithm_seed=S.PERCOLATION["algorithm_seed"],
                                   threshold=S.PERCOLATION["threshold"],
                                   expand_when_stuck=S.PERCOLATION["expand_when_stuck"])
        return {"accuracy": out["final_unseeded_accuracy"], "objective": np.nan, "runs": 1}
    best = None
    for restart in range(1 if algorithm == "goat" else FAQ_RESTARTS):
        out = rc._run_sgm_once(A, B, truth, seeds, rc._seed(solver_seed, restart), S.SGM["max_iter"], S.SGM["tol"],
                               transport=algorithm == "goat", transport_regularizer=GOAT_REGULARIZER,
                               transport_tol=S.SGM["transport_tol"], transport_max_iter=S.SGM["transport_max_iter"])
        if best is None or out["sgm_objective_score"] > best["objective"]:
            best = {"accuracy": out["final_unseeded_accuracy"], "objective": out["sgm_objective_score"]}
    return {**best, "runs": 1 if algorithm == "goat" else FAQ_RESTARTS}


def _task_path(keep: float, pair_index: int) -> Path:
    return OUTPUT_DIR / f"keep{keep:.2f}" / f"pair_{pair_index:02d}.csv"


def pair_job(keep: float, pair_index: int, algorithms=ALGORITHMS, blas_threads: int = 3) -> str:
    """Every algorithm, rule and budget on one pair."""

    path = _task_path(keep, pair_index)
    if path.exists():
        return f"keep {keep:.2f} pair {pair_index}: already done"
    started = time.perf_counter()
    rows = []
    with thread_limit(blas_threads):
        A, B, truth, rng = make_pair(keep, pair_index)
        aligned_b = B[np.ix_(truth, truth)]
        stats = {"n": A.shape[0], "mean_degree": float(A.sum(axis=1).mean()), "max_degree": int(A.sum(axis=1).max()),
                 "unmatchable": int(((A.sum(axis=1) == 0) | (aligned_b.sum(axis=1) == 0)).sum()),
                 "edges_kept": float((A * aligned_b).sum() / A.sum())}
        for rule, order in orderings(A, rng).items():
            for budget in BUDGETS:
                seeds = np.sort(order[:budget])
                for algorithm in algorithms:
                    solver_seed = rc._seed(SEED_BASE, int(round(keep * 100)), pair_index, budget)
                    out = _evaluate(algorithm, A, B, truth, seeds, solver_seed)
                    rows.append({"keep": keep, "pair_index": pair_index, "algorithm": algorithm, "rule": rule,
                                 "budget": budget, **out, "seed_degree": float(A[seeds].sum(axis=1).mean()), **stats})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)
    return f"keep {keep:.2f} pair {pair_index}: {len(rows)} evaluations in {time.perf_counter() - started:.0f} s"


def run_all(jobs: int = 3, keeps=KEEPS, n_pairs: int = N_PAIRS, blas_threads: int = 3) -> None:
    """Run every pair not yet done, `jobs` at a time."""

    tasks = [(k, i) for k in keeps for i in range(n_pairs) if not _task_path(k, i).exists()]
    if not tasks:
        print(f"All pairs are done. Results are in {OUTPUT_DIR}")
        return
    parent = parent_graph()
    print(f"{_now()} Enron parent graph: {parent.shape[0]} people, mean degree {parent.sum(axis=1).mean():.1f}. "
          f"{len(tasks)} pairs to run, {jobs} at a time.", flush=True)
    failures = []
    if jobs <= 1:
        for n, task in enumerate(tasks, 1):
            print(f"{_now()} [{n}/{len(tasks)}] {pair_job(*task, ALGORITHMS, blas_threads)}", flush=True)
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(pair_job, *task, ALGORITHMS, blas_threads): task for task in tasks}
            for n, future in enumerate(as_completed(futures), 1):
                keep, pair_index = futures[future]
                try:
                    print(f"{_now()} [{n}/{len(tasks)}] {future.result()}", flush=True)
                except Exception as error:  # noqa: BLE001 - report every failure
                    failures.append(f"keep {keep:.2f} pair {pair_index}")
                    print(f"{_now()} [{n}/{len(tasks)}] {failures[-1]} FAILED:", flush=True)
                    traceback.print_exception(type(error), error, error.__traceback__)
    if failures:
        raise RuntimeError(f"Failed: {', '.join(failures)}. Rerun to retry them.")
    print(f"{_now()} Done. Results are in {OUTPUT_DIR}", flush=True)


# --------------------------------------------------------------------------
# Analysis helpers for real_graph_check.ipynb
# --------------------------------------------------------------------------

def load_runs() -> pd.DataFrame:
    files = sorted(OUTPUT_DIR.glob("keep*/pair_*.csv"))
    if not files:
        raise FileNotFoundError(f"No results in {OUTPUT_DIR}. Run run_all() first.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def compare(runs: pd.DataFrame, algorithm: str, keep: float, budget: int, first: str, second: str,
            n_boot: int = 2000, seed: int = 2026) -> dict:
    """Pair-by-pair accuracy difference at one budget: first rule minus second rule."""

    from scipy.stats import binomtest

    part = runs[(runs["algorithm"] == algorithm) & np.isclose(runs["keep"], keep) & (runs["budget"] == budget)]
    table = part.pivot(index="pair_index", columns="rule", values="accuracy")
    difference = (table[first] - table[second]).to_numpy(float)
    better, worse = int((difference > 0.005).sum()), int((difference < -0.005).sum())
    boot = np.random.default_rng(seed).choice(difference, size=(n_boot, len(difference))).mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"better": better, "same": len(difference) - better - worse, "worse": worse,
            "p": binomtest(better, better + worse).pvalue if better + worse else np.nan,
            "mean change": f"{difference.mean():+.3f} [{low:+.3f}, {high:+.3f}]"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=N_PAIRS)
    parser.add_argument("--blas-threads", type=int, default=3)
    args = parser.parse_args()
    run_all(jobs=args.jobs, n_pairs=args.pairs, blas_threads=args.blas_threads)
