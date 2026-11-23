"""Settle two borderline round-1 claims.

1. Percolation: does clustered beat top degree?
   50 fresh pairs per model (ER and PAPER; pair indices 100-149, never used
   before), with round 1's settings and budgets. One run per budget.
2. FAQ: is degree all that matters?
   Extends round 1's FAQ run from 10 to 20 pairs with run_comparison
   (python -m selection_rule_comparison.run_comparison --algorithm faq --pairs 20).
   The main notebook (rule_comparison.ipynb) then also shows FAQ at 20 pairs.

Run from the repository root (ds environment):

    python -m selection_rule_comparison.borderline_check            # both parts
    python -m selection_rule_comparison.borderline_check --part percolation

or call run_percolation() and run_faq() from borderline_check.ipynb. Results are
saved pair by pair, so rerunning resumes.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import tempfile
import time
import traceback
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
from ..strategies import STRATEGY_CODES, GraphA, ordering, seed_set_features

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]  # selection_rule_comparison/ (this file is in archive/)
OUTPUT_DIR = EXPERIMENT_DIR / "data" / "borderline" / "percolation"
ROUND1_ROOT = EXPERIMENT_DIR / "data" / "round1" / "main"

PERCOLATION_MODELS = ("sparse_er", "paper_pa")
PERCOLATION_RULES = ("top_degree", "clustered")
FIRST_PAIR = 100                    # pair indices 100-149: not used by round 1 or any earlier check
N_PAIRS = 50
FAQ_PAIRS = 20

LABELS = {"top_degree": "top degree", "clustered": "clustered", "random": "random", "spread": "spread",
          "distinguishing": "distinguishing"}
MODEL_LABELS = {"sparse_er": "ER", "paper_pa": "PAPER", "chung_lu": "Chung-Lu"}


def _now() -> str:
    return datetime.datetime.now().astimezone().strftime("%H:%M:%S")


def _flag(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


# --------------------------------------------------------------------------
# Part 1: percolation on fresh pairs
# --------------------------------------------------------------------------

def _pair_path(model: str, pair_index: int) -> Path:
    return OUTPUT_DIR / model / f"pair_{pair_index:03d}.csv"


def percolation_pair_job(model: str, pair_index: int, blas_threads: int = 3) -> str:
    """Top degree and clustered on one fresh pair, at every round-1 budget."""

    path = _pair_path(model, pair_index)
    if path.exists():
        return f"{MODEL_LABELS[model]} pair {pair_index}: already done"
    started = time.perf_counter()
    rows = []
    with thread_limit(blas_threads):
        config = S.graph_config("percolation", model)
        first_seed = rc.pair_first_seed("percolation", model, "main", pair_index)
        offset = 0
        while True:
            seed, A, B, truth, _report, _rejected = first_passing_pair(
                lambda s: generate_graph_pair(config, s), first_seed + offset)
            if seed not in S.KNOWN_PAIR_SEEDS:
                break
            offset = seed - first_seed + 1
        graph = GraphA(A)
        budgets = sorted(S.default_budgets("main", "percolation", model))
        pair = (A, B, truth, seed)
        for rule in PERCOLATION_RULES:
            rng = np.random.default_rng([S.STRATEGY_SEED_BASE, seed, STRATEGY_CODES[rule]])
            order = ordering(rule, graph, budgets[-1], rng, pool_factor=S.POOL_FACTOR, hub_count=S.HUB_COUNT)
            for budget in budgets:
                seeds = np.sort(order[:budget])
                out = rc.evaluate("percolation", model, pair, seeds, budget, 0)
                rows.append({"model": model, "pair_index": pair_index, "pair_seed": seed, "rule": rule,
                             "budget": budget, "success": bool(out["success"]),
                             "success_090": bool(out["success_090"]), "accuracy": out["final_unseeded_accuracy"],
                             "runtime_seconds": out["runtime_seconds"], "hash_a": rc._hash(A), "hash_b": rc._hash(B),
                             "seed_vertices": json.dumps([int(v) for v in seeds]),
                             **seed_set_features(graph, seeds)})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)
    return f"{MODEL_LABELS[model]} pair {pair_index}: {len(rows)} runs in {time.perf_counter() - started:.0f} s"


def run_percolation(jobs: int = 3, models=PERCOLATION_MODELS, first_pair: int = FIRST_PAIR,
                    n_pairs: int = N_PAIRS, blas_threads: int = 3) -> None:
    """Percolation on fresh pairs, `jobs` pairs at a time (about 5 minutes)."""

    tasks = [(m, i) for m in models for i in range(first_pair, first_pair + n_pairs) if not _pair_path(m, i).exists()]
    if not tasks:
        print(f"All percolation pairs are done. Results are in {OUTPUT_DIR}")
        return
    print(f"{_now()} percolation: {len(tasks)} fresh pairs to run, {jobs} at a time.", flush=True)
    failures = []
    if jobs <= 1:
        for n, task in enumerate(tasks, 1):
            print(f"{_now()} [{n}/{len(tasks)}] {percolation_pair_job(*task, blas_threads)}", flush=True)
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(percolation_pair_job, *task, blas_threads): task for task in tasks}
            for n, future in enumerate(as_completed(futures), 1):
                model, pair_index = futures[future]
                try:
                    message = future.result()
                    if n % 10 == 0 or n == len(tasks):
                        print(f"{_now()} [{n}/{len(tasks)}] {message}", flush=True)
                except Exception as error:  # noqa: BLE001 - report every failure
                    failures.append(f"{MODEL_LABELS[model]} pair {pair_index}")
                    print(f"{_now()} [{n}/{len(tasks)}] {failures[-1]} FAILED:", flush=True)
                    traceback.print_exception(type(error), error, error.__traceback__)
    if failures:
        raise RuntimeError(f"Failed: {', '.join(failures)}. Rerun to retry them.")
    print(f"{_now()} Percolation done. Results are in {OUTPUT_DIR}", flush=True)


# --------------------------------------------------------------------------
# Part 2: FAQ to 20 pairs (round 1's own runner)
# --------------------------------------------------------------------------

def run_faq(n_pairs: int = FAQ_PAIRS, jobs: int = 3) -> None:
    """Extend round 1's FAQ results to n_pairs pairs per model (about 35-40 minutes for 10 more pairs)."""

    keep_awake = None
    if sys.platform == "darwin":
        import shutil
        import subprocess
        if shutil.which("caffeinate"):
            keep_awake = subprocess.Popen(["caffeinate", "-i"])
    try:
        rc.main(["--algorithm", "faq", "--pairs", str(n_pairs), "--jobs", str(jobs), "--no-caffeinate"])
    finally:
        if keep_awake is not None:
            keep_awake.terminate()


# --------------------------------------------------------------------------
# Analysis helpers for borderline_check.ipynb
# --------------------------------------------------------------------------

def load_percolation(include_round1: bool = False) -> pd.DataFrame:
    """Success per (model, pair, rule, budget) on the fresh pairs, optionally with round 1's pairs."""

    files = sorted(OUTPUT_DIR.glob("*/pair_*.csv"))
    if not files:
        raise FileNotFoundError(f"No results in {OUTPUT_DIR}. Run run_percolation() first.")
    fresh = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    fresh["success"] = _flag(fresh["success"])
    fresh["source"] = "fresh"
    frames = [fresh[["model", "pair_index", "rule", "budget", "success", "source"]]]
    if include_round1:
        for model in fresh["model"].unique():
            old = pd.read_csv(ROUND1_ROOT / "percolation" / model / "runs.csv")
            old = old[old["strategy"].isin(PERCOLATION_RULES)].rename(columns={"strategy": "rule"})
            frames.append(old.assign(model=model, success=_flag(old["success"]), source="round 1")
                          [["model", "pair_index", "rule", "budget", "success", "source"]])
    return pd.concat(frames, ignore_index=True)


def load_faq() -> pd.DataFrame:
    """Round 1's FAQ results (success = any of up to 3 restarts), all pairs done so far."""

    frames = []
    for model in ("sparse_er", "paper_pa", "chung_lu"):
        path = ROUND1_ROOT / "faq" / model / "runs.csv"
        if not path.exists():
            continue
        runs = pd.read_csv(path)
        pairs = pd.read_csv(ROUND1_ROOT / "faq" / model / "pairs.csv")["pair_index"]
        runs = runs[runs["pair_index"].isin(pairs)]
        runs = (runs.assign(success=_flag(runs["success"])).rename(columns={"strategy": "rule"})
                .groupby(["pair_index", "rule", "budget"], as_index=False)["success"].any())
        frames.append(runs.assign(model=model))
    return pd.concat(frames, ignore_index=True)


def queries_needed(results: pd.DataFrame, by=("model",)) -> pd.DataFrame:
    """Smallest budget that succeeds, per pair and rule. A pair that never succeeds counts as one step past the grid."""

    groups = list(by) + ["rule"]
    rows = []
    for key, part in results.groupby(groups):
        table = part.pivot(index="pair_index", columns="budget", values="success").sort_index(axis=1)
        budgets = list(table.columns)
        past_grid = budgets[-1] + (budgets[-1] - budgets[-2])
        for pair_index, row in table.iterrows():
            hits = [b for b, ok in row.items() if ok]
            rows.append({**dict(zip(groups, key)), "pair_index": pair_index,
                         "queries": hits[0] if hits else past_grid, "never": not hits})
    return pd.DataFrame(rows)


def compare(queries: pd.DataFrame, first: str, second: str, n_boot: int = 2000, seed: int = 2026) -> dict:
    """Pair-by-pair comparison of queries needed: first rule minus second rule."""

    from scipy.stats import binomtest

    table = queries.pivot(index="pair_index", columns="rule", values="queries")
    difference = (table[first] - table[second]).to_numpy(float)
    fewer, more = int((difference < 0).sum()), int((difference > 0).sum())
    boot = np.random.default_rng(seed).choice(difference, size=(n_boot, len(difference))).mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"pairs": len(difference), "fewer": fewer, "same": len(difference) - fewer - more, "more": more,
            "p": binomtest(fewer, fewer + more).pvalue if fewer + more else np.nan,
            "mean change": f"{difference.mean():+.2f} [{low:+.2f}, {high:+.2f}]"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=("percolation", "faq", "both"), default="both")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if args.part in ("percolation", "both"):
        run_percolation(jobs=args.jobs)
    if args.part in ("faq", "both"):
        run_faq(jobs=args.jobs)
