"""Hub tests: why does skipping hubs help GOAT, and only GOAT?

All tests reuse round-1 pairs, so round-1 results for the other rules are
compared pair by pair.

GOAT (Chung-Lu and PAPER, 20 pairs each)
1. Hubs or uniqueness?
   New rule "hub-free top degree": top degree, skipping the 10 hubs that
   hub-free distinguishing skips. With round 1's top degree, distinguishing and
   hub-free distinguishing, this completes a 2 x 2:
   (hubs kept or skipped) x (seeds chosen by degree or by uniqueness).
2. Wrong hints or too broad?
   "Top degree, clean hub edges": round 1's top-degree seeds, with each seeded
   hub's edges in B made identical to its edges in A. Nothing else changes.
3. Are hubs harmful or just wasted?
   The first 6, 8 or 10 hub-free top-degree seeds, plus the 1, 2 or 5
   highest-degree hubs. Compared with the same ordinary seeds alone.

FAQ and percolation (Chung-Lu and PAPER, all round-1 pairs)
4. Is it specific to GOAT?
   Hub-free top degree (top degree without its first 10 picks), run with the
   round-1 settings of each algorithm: FAQ with up to 3 restarts, stopping at
   the first success; percolation with one run.

Checks: plain top degree is rerun on the first two pairs of every algorithm and
model and must reproduce round 1.

Run from the repository root (ds environment):

    python -m selection_rule_comparison.hub_test

or call run_all() from hub_test.ipynb. Each pair is saved when it finishes, so
rerunning resumes and only runs what is missing. If FAQ's round 1 is extended
to more pairs later, rerunning adds the FAQ check on the new pairs.
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
from ..strategies import STRATEGY_CODES, GraphA, seed_set_features

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]  # selection_rule_comparison/ (this file is in archive/)
ROUND1_ROOT = EXPERIMENT_DIR / "data" / "round1" / "main"
ROUND1_DIR = ROUND1_ROOT / "goat"
OUTPUT_DIR = EXPERIMENT_DIR / "data" / "hub_test"

MODELS = ("chung_lu", "paper_pa")
N_PAIRS = 20                        # GOAT pairs per model
CHECK_PAIRS = 2                     # plain top degree is rerun on pairs 0 and 1
PARTS = ("goat", "add_hubs", "other")
OTHER_ALGORITHMS = ("faq", "percolation")
ADD_HUBS_ORDINARY = (6, 8, 10)      # ordinary (hub-free top-degree) seeds
ADD_HUBS_COUNTS = (1, 2, 5)         # hubs added to them
SECONDS_PER_RUN = {"goat": 3.5, "faq": 3.0, "percolation": 0.5}

ROUND1_RULES = ("random", "top_degree", "distinguishing", "hub_free_distinguishing")
NEW_RULES = ("hub_free_top_degree", "top_degree_clean_hubs")
LABELS = {
    "random": "random", "top_degree": "top degree", "distinguishing": "distinguishing",
    "hub_free_distinguishing": "hub-free distinguishing", "hub_free_top_degree": "hub-free top degree",
    "top_degree_clean_hubs": "top degree, clean hub edges",
}
MODEL_LABELS = {"chung_lu": "Chung-Lu", "paper_pa": "PAPER", "sparse_er": "ER"}
ALGORITHM_LABELS = {"goat": "GOAT", "faq": "SGM (FAQ)", "percolation": "Percolation"}


def _now() -> str:
    return datetime.datetime.now().astimezone().strftime("%H:%M:%S")


def _flag(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


# --------------------------------------------------------------------------
# Pairs, orderings and the edge change
# --------------------------------------------------------------------------

def load_pair(model: str, pair_index: int, verify: bool = True, algorithm: str = "goat"):
    """Regenerate a round-1 pair exactly, checked against round 1's hashes."""

    config = S.graph_config(algorithm, model)
    first_seed = rc.pair_first_seed(algorithm, model, "main", pair_index)
    offset = 0
    while True:
        seed, A, B, truth, _report, _rejected = first_passing_pair(
            lambda s: generate_graph_pair(config, s), first_seed + offset)
        if seed not in S.KNOWN_PAIR_SEEDS:
            break
        offset = seed - first_seed + 1
    if verify:
        ref = pd.read_csv(ROUND1_ROOT / algorithm / model / "pairs.csv").set_index("pair_index").loc[pair_index]
        if int(ref["pair_seed"]) != int(seed) or ref["hash_a"] != rc._hash(A) or ref["hash_b"] != rc._hash(B):
            raise RuntimeError(f"{algorithm}/{model} pair {pair_index} does not match round 1.")
    return int(seed), A, B, truth


def round1_pairs(algorithm: str, model: str) -> list[int]:
    """Pair indices that round 1 finished for this algorithm and model."""

    return sorted(pd.read_csv(ROUND1_ROOT / algorithm / model / "pairs.csv")["pair_index"].astype(int))


def _rule_rng(pair_seed: int, rule: str) -> np.random.Generator:
    """The random stream round 1 used for this rule on this pair (it only breaks degree ties)."""

    return np.random.default_rng([S.STRATEGY_SEED_BASE, pair_seed, STRATEGY_CODES[rule]])


def orderings(graph: GraphA, pair_seed: int, k_max: int):
    """Round 1's hubs and top-degree order, and the new hub-free top-degree order."""

    hubs = {int(v) for v in graph.degree_order(_rule_rng(pair_seed, "hub_free_distinguishing"))[:S.HUB_COUNT]}
    by_degree = [int(v) for v in graph.degree_order(_rule_rng(pair_seed, "top_degree"))]
    hub_free = [v for v in by_degree if v not in hubs]
    return hubs, np.array(by_degree[:k_max]), np.array(hub_free[:k_max])


def _check_round1_order(algorithm: str, model: str, pair_index: int, top_degree: np.ndarray) -> None:
    saved = pd.read_csv(ROUND1_ROOT / algorithm / model / "orderings.csv")
    saved = saved[(saved["pair_index"] == pair_index) & (saved["strategy"] == "top_degree")]["ordering"]
    if json.loads(saved.iloc[0]) != [int(v) for v in top_degree[:len(json.loads(saved.iloc[0]))]]:
        raise RuntimeError(f"{algorithm}/{model} pair {pair_index}: top-degree order differs from round 1.")


def clean_hub_edges(A: np.ndarray, B: np.ndarray, truth: np.ndarray, hubs) -> np.ndarray:
    """B with each hub's edges replaced by its edges in A, mapped through the true matching."""

    B = np.array(B, copy=True)
    for hub in hubs:
        row = np.zeros_like(B[0])
        row[truth] = A[hub]                 # B[truth[hub], truth[v]] = A[hub, v]
        B[truth[hub], :] = row
        B[:, truth[hub]] = row
    return B


# --------------------------------------------------------------------------
# Jobs (one pair each; each writes one CSV file)
# --------------------------------------------------------------------------

def _task_path(part: str, algorithm: str, model: str, pair_index: int) -> Path:
    if part == "goat":
        return OUTPUT_DIR / model / f"pair_{pair_index:02d}.csv"
    if part == "add_hubs":
        return OUTPUT_DIR / "add_hubs" / model / f"pair_{pair_index:02d}.csv"
    return OUTPUT_DIR / algorithm / model / f"pair_{pair_index:02d}.csv"


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def pair_job(model: str, pair_index: int, verify: bool = True, blas_threads: int = 3) -> str:
    """GOAT tests 1 and 2 on one pair."""

    path = _task_path("goat", "goat", model, pair_index)
    if path.exists():
        return f"GOAT {MODEL_LABELS[model]} pair {pair_index}: already done"
    started = time.perf_counter()
    with thread_limit(blas_threads):
        seed, A, B, truth = load_pair(model, pair_index, verify)
        graph = GraphA(A)
        budgets = sorted(S.default_budgets("main", "goat", model))
        hubs, top_degree, hub_free = orderings(graph, seed, budgets[-1])
        if verify:
            _check_round1_order("goat", model, pair_index, top_degree)

        rows = []

        def run(rule, budget, seeds, b_matrix, hubs_cleaned):
            out = rc.evaluate("goat", model, (A, b_matrix, truth, seed), seeds, budget, 0)
            rows.append({
                "model": model, "pair_index": pair_index, "pair_seed": seed, "rule": rule, "budget": budget,
                "hubs_in_seeds": sum(int(v) in hubs for v in seeds), "hubs_cleaned": hubs_cleaned,
                "success": bool(out["success"]), "accuracy": out["final_unseeded_accuracy"],
                "objective": out["objective"], "n_iter": out["n_iter"], "runtime_seconds": out["runtime_seconds"],
                "seed_vertices": json.dumps([int(v) for v in seeds]), **seed_set_features(graph, seeds),
            })

        for budget in budgets:
            seeds = np.sort(hub_free[:budget])
            run("hub_free_top_degree", budget, seeds, B, 0)
            seeds = np.sort(top_degree[:budget])
            seeded_hubs = [int(v) for v in seeds if int(v) in hubs]
            run("top_degree_clean_hubs", budget, seeds, clean_hub_edges(A, B, truth, seeded_hubs), len(seeded_hubs))
            if pair_index < CHECK_PAIRS:
                run("top_degree_check", budget, seeds, B, 0)
    _write(path, rows)
    return f"GOAT {MODEL_LABELS[model]} pair {pair_index}: {len(rows)} runs in {time.perf_counter() - started:.0f} s"


def add_hubs_job(model: str, pair_index: int, verify: bool = True, blas_threads: int = 3) -> str:
    """GOAT test 3 on one pair: ordinary seeds with 1, 2 or 5 hubs added."""

    path = _task_path("add_hubs", "goat", model, pair_index)
    if path.exists():
        return f"GOAT add-hubs {MODEL_LABELS[model]} pair {pair_index}: already done"
    started = time.perf_counter()
    rows = []
    with thread_limit(blas_threads):
        seed, A, B, truth = load_pair(model, pair_index, verify)
        graph = GraphA(A)
        k_max = max(S.default_budgets("main", "goat", model))
        hubs, top_degree, hub_free = orderings(graph, seed, k_max)
        hubs_by_degree = [int(v) for v in top_degree if int(v) in hubs]
        for ordinary in ADD_HUBS_ORDINARY:
            for added in ADD_HUBS_COUNTS:
                seeds = np.sort(np.concatenate([hub_free[:ordinary], hubs_by_degree[:added]]))
                out = rc.evaluate("goat", model, (A, B, truth, seed), seeds, len(seeds), 0)
                rows.append({"model": model, "pair_index": pair_index, "pair_seed": seed, "ordinary": ordinary,
                             "hubs_added": added, "budget": len(seeds), "success": bool(out["success"]),
                             "accuracy": out["final_unseeded_accuracy"], "objective": out["objective"],
                             "runtime_seconds": out["runtime_seconds"],
                             "seed_vertices": json.dumps([int(v) for v in seeds])})
    _write(path, rows)
    return (f"GOAT add-hubs {MODEL_LABELS[model]} pair {pair_index}: {len(rows)} runs in "
            f"{time.perf_counter() - started:.0f} s")


def other_job(algorithm: str, model: str, pair_index: int, verify: bool = True, blas_threads: int = 3) -> str:
    """Test 4 on one pair: hub-free top degree with FAQ or percolation (round-1 settings)."""

    path = _task_path("other", algorithm, model, pair_index)
    name = f"{ALGORITHM_LABELS[algorithm]} {MODEL_LABELS[model]} pair {pair_index}"
    if path.exists():
        return f"{name}: already done"
    started = time.perf_counter()
    restarts = S.FAQ_MAX_RESTARTS if algorithm == "faq" else 1
    rows = []
    with thread_limit(blas_threads):
        seed, A, B, truth = load_pair(model, pair_index, verify, algorithm)
        graph = GraphA(A)
        budgets = sorted(S.default_budgets("main", algorithm, model))
        by_degree = np.array([int(v) for v in graph.degree_order(_rule_rng(seed, "top_degree"))])
        if verify:
            _check_round1_order(algorithm, model, pair_index, by_degree)
        orders = {"hub_free_top_degree": by_degree[S.HUB_COUNT:S.HUB_COUNT + budgets[-1]]}
        if pair_index < CHECK_PAIRS:
            orders["top_degree_check"] = by_degree[:budgets[-1]]
        for rule, order in orders.items():
            for budget in budgets:
                seeds = np.sort(order[:budget])
                for restart in range(restarts):
                    out = rc.evaluate(algorithm, model, (A, B, truth, seed), seeds, budget, restart)
                    rows.append({"algorithm": algorithm, "model": model, "pair_index": pair_index, "pair_seed": seed,
                                 "rule": rule, "budget": budget, "restart": restart, "success": bool(out["success"]),
                                 "accuracy": out["final_unseeded_accuracy"], "objective": out["objective"],
                                 "runtime_seconds": out["runtime_seconds"],
                                 "seed_vertices": json.dumps([int(v) for v in seeds])})
                    if out["success"]:
                        break
    _write(path, rows)
    return f"{name}: {len(rows)} runs in {time.perf_counter() - started:.0f} s"


def _tasks(parts, models, n_pairs):
    tasks = []
    for model in models:
        if "goat" in parts:
            tasks += [("goat", "goat", model, i) for i in range(n_pairs)]
        if "add_hubs" in parts:
            tasks += [("add_hubs", "goat", model, i) for i in range(n_pairs)]
        if "other" in parts:
            tasks += [("other", a, model, i) for a in OTHER_ALGORITHMS for i in round1_pairs(a, model)]
    return [t for t in tasks if not _task_path(*t).exists()]


def _run_task(part: str, algorithm: str, model: str, pair_index: int, verify: bool, blas_threads: int) -> str:
    if part == "goat":
        return pair_job(model, pair_index, verify, blas_threads)
    if part == "add_hubs":
        return add_hubs_job(model, pair_index, verify, blas_threads)
    return other_job(algorithm, model, pair_index, verify, blas_threads)


def _estimate_minutes(tasks, jobs: int) -> float:
    runs = {"goat": 12, "add_hubs": len(ADD_HUBS_ORDINARY) * len(ADD_HUBS_COUNTS), "other": 10}
    seconds = sum(runs[part] * SECONDS_PER_RUN[algorithm] for part, algorithm, _, _ in tasks)
    return seconds / max(jobs, 1) / 60


def run_all(jobs: int = 3, parts=PARTS, models=MODELS, n_pairs: int = N_PAIRS, blas_threads: int = 3,
            verify: bool = True) -> None:
    """Run every pair that is not done yet, `jobs` pairs at a time."""

    tasks = _tasks(parts, models, n_pairs)
    if not tasks:
        print(f"All pairs are done. Results are in {OUTPUT_DIR}")
        return
    print(f"{_now()} {len(tasks)} pairs to run, {jobs} at a time: about {_estimate_minutes(tasks, jobs):.0f} minutes.",
          flush=True)
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
                print(f"{_now()} [{n}/{len(tasks)}] {_run_task(*task, verify, blas_threads)}", flush=True)
        else:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {pool.submit(_run_task, *task, verify, blas_threads): task for task in tasks}
                for n, future in enumerate(as_completed(futures), 1):
                    part, algorithm, model, pair_index = futures[future]
                    try:
                        print(f"{_now()} [{n}/{len(tasks)}] {future.result()}", flush=True)
                    except Exception as error:  # noqa: BLE001 - report every failure, then stop
                        failures.append(f"{part} {algorithm} {MODEL_LABELS[model]} pair {pair_index}")
                        print(f"{_now()} [{n}/{len(tasks)}] {failures[-1]} FAILED:", flush=True)
                        traceback.print_exception(type(error), error, error.__traceback__)
    finally:
        if keep_awake is not None:
            keep_awake.terminate()
    if failures:
        raise RuntimeError(f"Failed: {', '.join(failures)}. Rerun to retry them.")
    print(f"{_now()} Done. Results are in {OUTPUT_DIR}", flush=True)


# --------------------------------------------------------------------------
# Analysis helpers for hub_test.ipynb
# --------------------------------------------------------------------------

def _read_folder(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("pair_*.csv"))
    if not files:
        return pd.DataFrame()
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    runs["success"] = _flag(runs["success"])
    return runs


def _new_runs(models=MODELS) -> pd.DataFrame:
    runs = pd.concat([_read_folder(OUTPUT_DIR / m) for m in models], ignore_index=True)
    if runs.empty:
        raise FileNotFoundError(f"No results in {OUTPUT_DIR}. Run run_all() first.")
    return runs


def _other_runs(models=MODELS) -> pd.DataFrame:
    return pd.concat([_read_folder(OUTPUT_DIR / a / m) for a in OTHER_ALGORITHMS for m in models],
                     ignore_index=True)


def check_against_round1(models=MODELS) -> pd.DataFrame:
    """Plain top degree rerun on the first pairs: success and accuracy must equal round 1's."""

    goat = _new_runs(models).assign(algorithm="goat", restart=0)
    other = _other_runs(models)
    runs = pd.concat([goat, other], ignore_index=True)
    runs = runs[runs["rule"] == "top_degree_check"]
    rows = []
    for (algorithm, model), mine in runs.groupby(["algorithm", "model"], sort=False):
        ref = pd.read_csv(ROUND1_ROOT / algorithm / model / "runs.csv")
        ref = ref[ref["strategy"] == "top_degree"].set_index(["pair_index", "budget", "restart"])
        mine = mine.set_index(["pair_index", "budget", "restart"])
        ref = ref.reindex(mine.index)
        rows.append({"algorithm": ALGORITHM_LABELS[algorithm], "model": MODEL_LABELS[model], "runs checked": len(mine),
                     "same success": int((mine["success"].to_numpy() == _flag(ref["success"]).to_numpy()).sum()),
                     "same accuracy": int(np.isclose(mine["accuracy"].to_numpy(float),
                                                     ref["final_unseeded_accuracy"].to_numpy(float)).sum())})
    return pd.DataFrame(rows).set_index(["algorithm", "model"])


def load_results(models=MODELS) -> pd.DataFrame:
    """GOAT: success per (model, pair, rule, budget) for the round-1 rules and the new ones, on the pairs done so far."""

    columns = ["model", "pair_index", "rule", "budget", "success", "accuracy", "f_degree", "f_signature_r3"]
    new = _new_runs(models)
    new = new[new["rule"].isin(NEW_RULES)]
    frames = [new[columns]]
    for model in models:
        done = new.loc[new["model"] == model, "pair_index"].unique()
        old = pd.read_csv(ROUND1_DIR / model / "runs.csv")
        old = old[old["strategy"].isin(ROUND1_RULES) & old["pair_index"].isin(done)]
        old = old.rename(columns={"strategy": "rule", "final_unseeded_accuracy": "accuracy"})
        frames.append(old.assign(model=model, success=_flag(old["success"]))[columns])
    return pd.concat(frames, ignore_index=True)


def load_algorithm_results(models=MODELS) -> pd.DataFrame:
    """Top degree, hub-free top degree and random for all three algorithms (success = any restart)."""

    columns = ["algorithm", "model", "pair_index", "rule", "budget", "success"]
    goat = _new_runs(models)
    goat = goat[goat["rule"] == "hub_free_top_degree"].assign(algorithm="goat")
    other = _other_runs(models)
    other = other[other["rule"] == "hub_free_top_degree"] if not other.empty else other
    new = pd.concat([goat, other], ignore_index=True)
    new = new.groupby(["algorithm", "model", "pair_index", "rule", "budget"], as_index=False)["success"].any()
    frames = [new[columns]]
    for (algorithm, model), part in new.groupby(["algorithm", "model"]):
        old = pd.read_csv(ROUND1_ROOT / algorithm / model / "runs.csv")
        old = old[old["strategy"].isin(["top_degree", "random"]) & old["pair_index"].isin(part["pair_index"].unique())]
        old = (old.assign(success=_flag(old["success"])).rename(columns={"strategy": "rule"})
               .groupby(["pair_index", "rule", "budget"], as_index=False)["success"].any())
        frames.append(old.assign(algorithm=algorithm, model=model)[columns])
    return pd.concat(frames, ignore_index=True)


def queries_needed(results: pd.DataFrame) -> pd.DataFrame:
    """Smallest budget that succeeds, per pair and rule. A pair that never succeeds counts as one step past the grid."""

    groups = [c for c in ("algorithm", "model") if c in results.columns] + ["rule"]
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


def compare(queries: pd.DataFrame, model: str, first: str, second: str, n_boot: int = 2000,
            seed: int = 2026, algorithm: str | None = None) -> dict:
    """Pair-by-pair comparison of queries needed on one model: first rule against second rule."""

    from scipy.stats import binomtest

    part = queries[queries["model"] == model]
    if algorithm is not None:
        part = part[part["algorithm"] == algorithm]
    table = part.pivot(index="pair_index", columns="rule", values="queries")
    difference = (table[first] - table[second]).to_numpy(float)
    fewer, more = int((difference < 0).sum()), int((difference > 0).sum())
    boot = np.random.default_rng(seed).choice(difference, size=(n_boot, len(difference))).mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"fewer": fewer, "same": len(difference) - fewer - more, "more": more,
            "p": binomtest(fewer, fewer + more).pvalue if fewer + more else np.nan,
            "mean change": f"{difference.mean():+.1f} [{low:+.1f}, {high:+.1f}]"}


def load_add_hubs(models=MODELS) -> pd.DataFrame:
    """Success of m ordinary seeds with h hubs added (h = 0: the hub-free top-degree run at budget m)."""

    added = pd.concat([_read_folder(OUTPUT_DIR / "add_hubs" / m) for m in models], ignore_index=True)
    if added.empty:
        raise FileNotFoundError("No add-hubs results yet. Run run_all() first.")
    base = _new_runs(models)
    base = base[(base["rule"] == "hub_free_top_degree") & base["budget"].isin(ADD_HUBS_ORDINARY)]
    base = base.assign(ordinary=base["budget"], hubs_added=0)
    base = base[base.set_index(["model", "pair_index"]).index.isin(added.set_index(["model", "pair_index"]).index)]
    columns = ["model", "pair_index", "ordinary", "hubs_added", "success", "accuracy"]
    return pd.concat([base[columns], added[columns]], ignore_index=True)


def add_hubs_table(data: pd.DataFrame) -> pd.DataFrame:
    """Per model and number of ordinary seeds: success without and with hubs, and pairs gained or lost."""

    from scipy.stats import binomtest

    rows = []
    for (model, ordinary), part in data.groupby(["model", "ordinary"]):
        table = part.pivot(index="pair_index", columns="hubs_added", values="success").fillna(False).astype(bool)
        for added in [h for h in table.columns if h > 0]:
            gained = int((~table[0] & table[added]).sum())
            lost = int((table[0] & ~table[added]).sum())
            rows.append({"model": MODEL_LABELS[model], "ordinary seeds": ordinary, "hubs added": added,
                         "success without": table[0].mean(), "success with": table[added].mean(),
                         "pairs gained": gained, "pairs lost": lost,
                         "p": binomtest(gained, gained + lost).pvalue if gained + lost else np.nan})
    return pd.DataFrame(rows).set_index(["model", "ordinary seeds", "hubs added"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=3, help="pairs run in parallel (default 3)")
    parser.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS)
    parser.add_argument("--pairs", type=int, default=N_PAIRS, help="GOAT round-1 pairs per model (default 20)")
    parser.add_argument("--blas-threads", type=int, default=3)
    args = parser.parse_args()
    run_all(jobs=args.jobs, parts=args.parts, n_pairs=args.pairs, blas_threads=args.blas_threads)
