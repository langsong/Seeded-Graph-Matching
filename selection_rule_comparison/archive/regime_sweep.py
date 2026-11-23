"""Regime sweep: when does skipping hubs help GOAT?

Fresh Chung-Lu pairs (n = 600, mean degree 16) over a grid of
    tail weight: Pareto shape alpha = 2.0 (heavy, as in round 1), 3.0, 6.0 (light)
    noise:       rho = 0.40, 0.45 (as in round 1), 0.50, 0.55
with three rules: top degree, hub-free top degree (top degree without the 10
highest-degree vertices) and random.

GOAT's transition in rho is steep. In a calibration (2 pairs per setting), no
rule reached success with 40 seeds at rho = 0.35, while every rule needed a
single seed at rho = 0.60 and 0.75. Hence 0.40 to 0.55.

Query, run, check, stop: for each pair and rule, budgets are tried in increasing
order along a ladder, and the first success ends the search. Each search gives
the number of queries needed directly.

Run from the repository root (ds environment), about 45-75 minutes:

    python -m selection_rule_comparison.regime_sweep

or call run_all() from regime_sweep.ipynb. Each pair is saved when it
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
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache"))

import numpy as np
import pandas as pd

from experiment_common.graph_checks import first_passing_pair
from experiment_common.graphs import generate_graph_pair
from experiment_common.provenance import thread_limit

from .. import run_comparison as rc
from .. import settings as S
from ..strategies import STRATEGY_CODES, GraphA

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]  # selection_rule_comparison/ (this file is in archive/)
OUTPUT_DIR = EXPERIMENT_DIR / "data" / "regime_sweep"

ALPHAS = (2.0, 3.0, 6.0)            # Pareto shape of the Chung-Lu weights: smaller = heavier tail
RHOS = (0.40, 0.45, 0.50, 0.55)     # edge correlation between A and B
N_PAIRS = 10                        # fresh pairs per setting
ALGORITHMS = ("goat",)              # add "faq" for an FAQ comparison (about one more hour)
RULES = ("top_degree", "hub_free_top_degree", "random")
LADDER = {"goat": (1, 2, 3, 4, 6, 8, 10, 12, 14, 17, 20, 24, 28, 34, 40),
          "faq": (4, 6, 8, 10, 12, 15, 18, 21, 24, 28, 32, 36, 40, 48, 56)}
MEAN_DEGREE = 16.0
SWEEP_SEED_BASE = 20261027          # pair seeds: never used by earlier experiments
SECONDS_PER_RUN = {"goat": 3.5, "faq": 3.0}

LABELS = {"top_degree": "top degree", "hub_free_top_degree": "hub-free top degree", "random": "random"}


def _now() -> str:
    return datetime.datetime.now().astimezone().strftime("%H:%M:%S")


# --------------------------------------------------------------------------
# Graph settings
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def weight_scale(alpha: float, n: int = 600, mean_degree: float = MEAN_DEGREE) -> float:
    """Weight scale giving the requested expected mean degree (same weights, cap and clipping as the generator)."""

    from scipy.optimize import brentq
    from utils.Graphs import pareto_quantile_weights

    base = pareto_quantile_weights(n, alpha)

    def expected_mean_degree(scale):
        w = scale * base
        w = np.minimum(w, np.sqrt(w.sum()))
        P = np.clip(np.outer(w, w) / w.sum(), 0, 1)
        np.fill_diagonal(P, 0)
        return P.sum() / n

    # Rounded to 6 decimals, which reproduces round 1's Chung-Lu scale (5.452064) at alpha = 2.
    return round(float(brentq(lambda s: expected_mean_degree(s) - mean_degree, 0.5, 50.0, xtol=1e-9)), 6)


def graph_config(alpha: float, rho: float):
    return replace(S.graph_config("goat", "chung_lu"), chung_lu_alpha=float(alpha),
                   chung_lu_weight_scale=weight_scale(float(alpha)), rho=float(rho))


def setting_name(alpha: float, rho: float) -> str:
    return f"alpha{alpha:.1f}_rho{rho:.2f}"


def load_pair(alpha: float, rho: float, pair_index: int):
    """A fresh, structure-checked pair for one setting (the same pair every time)."""

    config = graph_config(alpha, rho)
    first_seed = rc._seed(SWEEP_SEED_BASE, int(round(alpha * 100)), int(round(rho * 100)), pair_index) % (2**32 - 10_000)
    offset = 0
    while True:
        seed, A, B, truth, _report, _rejected = first_passing_pair(
            lambda s: generate_graph_pair(config, s), first_seed + offset)
        if seed not in S.KNOWN_PAIR_SEEDS:
            break
        offset = seed - first_seed + 1
    return int(seed), A, B, truth


def orderings(graph: GraphA, pair_seed: int, k_max: int) -> dict:
    """Top degree, the same order without the 10 hubs, and a random order (round-1 random streams)."""

    by_degree = graph.degree_order(np.random.default_rng([S.STRATEGY_SEED_BASE, pair_seed, STRATEGY_CODES["top_degree"]]))
    shuffled = np.random.default_rng([S.STRATEGY_SEED_BASE, pair_seed, STRATEGY_CODES["random"]]).permutation(graph.n)
    return {"top_degree": by_degree[:k_max], "hub_free_top_degree": by_degree[S.HUB_COUNT:S.HUB_COUNT + k_max],
            "random": shuffled[:k_max]}


def pair_stats(A: np.ndarray, B: np.ndarray, truth: np.ndarray) -> dict:
    A = np.asarray(A) > 0
    aligned_b = np.asarray(B)[np.ix_(truth, truth)] > 0
    degree = A.sum(axis=1)
    hubs = np.argsort(-degree, kind="stable")[:S.HUB_COUNT]
    return {"mean_degree": float(degree.mean()), "max_degree": int(degree.max()),
            "hub_mean_degree": float(degree[hubs].mean()),
            "edges_kept": float((A & aligned_b).sum() / A.sum()),
            "hub_edges_kept": float((A[hubs] & aligned_b[hubs]).sum() / A[hubs].sum())}


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

def _task_path(algorithm: str, alpha: float, rho: float, pair_index: int) -> Path:
    return OUTPUT_DIR / algorithm / setting_name(alpha, rho) / f"pair_{pair_index:02d}.csv"


def pair_job(algorithm: str, alpha: float, rho: float, pair_index: int, blas_threads: int = 3) -> str:
    """Query-run-check-stop for every rule on one pair."""

    path = _task_path(algorithm, alpha, rho, pair_index)
    if path.exists():
        return f"{algorithm} {setting_name(alpha, rho)} pair {pair_index}: already done"
    started = time.perf_counter()
    ladder = LADDER[algorithm]
    restarts = S.FAQ_MAX_RESTARTS if algorithm == "faq" else 1
    rows = []
    with thread_limit(blas_threads):
        seed, A, B, truth = load_pair(alpha, rho, pair_index)
        graph = GraphA(A)
        stats = pair_stats(A, B, truth)
        pair = (A, B, truth, seed)
        for rule, order in orderings(graph, seed, ladder[-1]).items():
            for budget in ladder:
                seeds = np.sort(order[:budget])
                success = False
                for restart in range(restarts):
                    # model "chung_lu" gives the round-1 Chung-Lu solver settings (GOAT regulariser 300)
                    out = rc.evaluate(algorithm, "chung_lu", pair, seeds, budget, restart)
                    success = bool(out["success"])
                    rows.append({"algorithm": algorithm, "alpha": alpha, "rho": rho, "pair_index": pair_index,
                                 "pair_seed": seed, "rule": rule, "budget": budget, "restart": restart,
                                 "success": success, "accuracy": out["final_unseeded_accuracy"],
                                 "objective": out["objective"], "runtime_seconds": out["runtime_seconds"],
                                 "seed_degree": float(graph.degree[seeds].mean()), **stats})
                    if success:
                        break
                if success:
                    break
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)
    return (f"{algorithm} {setting_name(alpha, rho)} pair {pair_index}: {len(rows)} runs in "
            f"{time.perf_counter() - started:.0f} s")


def run_all(jobs: int = 3, algorithms=ALGORITHMS, alphas=ALPHAS, rhos=RHOS, n_pairs: int = N_PAIRS,
            blas_threads: int = 3) -> None:
    """Run every (setting, pair) not yet done, `jobs` at a time."""

    tasks = [(a, alpha, rho, i) for a in algorithms for rho in rhos for alpha in alphas for i in range(n_pairs)
             if not _task_path(a, alpha, rho, i).exists()]
    if not tasks:
        print(f"All pairs are done. Results are in {OUTPUT_DIR}")
        return
    estimate = sum(len(RULES) * 8 * SECONDS_PER_RUN[t[0]] for t in tasks) / max(jobs, 1) / 60
    print(f"{_now()} {len(tasks)} pairs to run, {jobs} at a time: roughly {estimate:.0f} minutes "
          "(harder settings take longer).", flush=True)
    keep_awake = None
    if sys.platform == "darwin":
        import shutil
        import subprocess
        if shutil.which("caffeinate"):
            keep_awake = subprocess.Popen(["caffeinate", "-i"])
    failures = []
    try:
        if jobs <= 1:
            for n, task in enumerate(tasks, 1):
                print(f"{_now()} [{n}/{len(tasks)}] {pair_job(*task, blas_threads)}", flush=True)
        else:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {pool.submit(pair_job, *task, blas_threads): task for task in tasks}
                for n, future in enumerate(as_completed(futures), 1):
                    task = futures[future]
                    try:
                        print(f"{_now()} [{n}/{len(tasks)}] {future.result()}", flush=True)
                    except Exception as error:  # noqa: BLE001 - report every failure
                        failures.append(f"{task[0]} {setting_name(task[1], task[2])} pair {task[3]}")
                        print(f"{_now()} [{n}/{len(tasks)}] {failures[-1]} FAILED:", flush=True)
                        traceback.print_exception(type(error), error, error.__traceback__)
    finally:
        if keep_awake is not None:
            keep_awake.terminate()
    if failures:
        raise RuntimeError(f"Failed: {', '.join(failures)}. Rerun to retry them.")
    print(f"{_now()} Done. Results are in {OUTPUT_DIR}", flush=True)


# --------------------------------------------------------------------------
# Analysis helpers for regime_sweep.ipynb
# --------------------------------------------------------------------------

def load_runs() -> pd.DataFrame:
    files = sorted(OUTPUT_DIR.glob("*/*/pair_*.csv"))
    if not files:
        raise FileNotFoundError(f"No results in {OUTPUT_DIR}. Run run_all() first.")
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    runs["success"] = runs["success"].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])
    return runs


def queries_needed(runs: pd.DataFrame) -> pd.DataFrame:
    """Queries needed per pair and rule. A search that never succeeds counts as one ladder step past the top."""

    keys = ["algorithm", "alpha", "rho", "pair_index", "rule"]
    ended = runs.groupby(keys).agg(last_budget=("budget", "max"), succeeded=("success", "any")).reset_index()
    rows = []
    for _, row in ended.iterrows():
        ladder = LADDER[row["algorithm"]]
        past_top = ladder[-1] + (ladder[-1] - ladder[-2])
        rows.append({**{k: row[k] for k in keys},
                     "queries": row["last_budget"] if row["succeeded"] else past_top,
                     "never": not row["succeeded"]})
    return pd.DataFrame(rows)


def compare(queries: pd.DataFrame, algorithm: str, alpha: float, rho: float, first: str, second: str,
            n_boot: int = 2000, seed: int = 2026) -> dict:
    """Pair-by-pair comparison in one setting: queries of `first` minus queries of `second`."""

    from scipy.stats import binomtest

    part = queries[(queries["algorithm"] == algorithm) & np.isclose(queries["alpha"], alpha)
                   & np.isclose(queries["rho"], rho)]
    table = part.pivot(index="pair_index", columns="rule", values="queries")
    difference = (table[first] - table[second]).to_numpy(float)
    fewer, more = int((difference < 0).sum()), int((difference > 0).sum())
    boot = np.random.default_rng(seed).choice(difference, size=(n_boot, len(difference))).mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"pairs": len(difference), "fewer": fewer, "same": len(difference) - fewer - more, "more": more,
            "p": binomtest(fewer, fewer + more).pvalue if fewer + more else np.nan,
            "mean": difference.mean(), "low": low, "high": high}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=3, help="pairs run in parallel (default 3)")
    parser.add_argument("--pairs", type=int, default=N_PAIRS, help="pairs per setting (default 10)")
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=("goat", "faq"))
    parser.add_argument("--blas-threads", type=int, default=3)
    args = parser.parse_args()
    run_all(jobs=args.jobs, algorithms=args.algorithms, n_pairs=args.pairs, blas_threads=args.blas_threads)
