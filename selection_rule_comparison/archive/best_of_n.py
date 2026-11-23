"""Best of N on round-1 pairs: screen random seed sets with the pooled weighted score (2 Nov 2026).

Question. Does a simple rule beat random, top degree and round 1's best hand-made rule on graph pairs the score
has never seen? The rule: draw N uniform random seed sets of the budget's size, score each from graph A alone, and
query the highest-scoring set.

Score. score = sum over f of w_f * z_f, for f = degree (seed-to-nonseed edges per seed), harmonic, pair mass and
uniqueness (capped distance fingerprints; cap 3, cap 2 for FAQ). z_f is the set's feature in standard deviations of
the random sets drawn on the same graph at the same budget, and w_f is the pooled weight from
multi_pair_models.ipynb (fixed_pair_seed_set_classification/data/multi_pair_2026-10-30/pooled_weights.json), one
set of weights per algorithm and graph model. The weights were fitted at one budget per setting; z-scores make them
usable at every budget, and whether they still work there is part of the test.

Pairs and budgets. Round 1's 20 pairs per algorithm and graph model (data/round1/main), at round 1's six budgets.
None of these pairs was used to fit the weights.

Rules run here, per pair and budget, from one draw of 1,000 distinct uniform random seed sets:
    best_100      the highest score among the first 100 sets
    best_1000     the highest score among all 1,000
    degree_1000   the most seed-to-nonseed edges among all 1,000 (screening by degree alone, as a control)
Each chosen set is run with round 1's solver settings and solver seeds, so it meets round 1's rules on equal
terms: one run for percolation and GOAT; FAQ restarts up to 3 times and stops at the first success.

Compared with round 1's stored results on the same pairs: random, top degree and round 1's best rule in each
setting (the highest mean success over the budget grid, picked after the fact, which favours it).

Checks. Every pair is regenerated and checked against round 1's hashes. On pairs 0 and 1, round 1's top-degree sets
are run again and must reproduce round 1's results exactly.

GOAT at r = 100 (robustness check, added 2 Nov). Round 1 ran GOAT with transport regulariser r = 300 on PAPER and
Chung-Lu (r = 100 on ER), tuned for those graphs; graspologic's default is r = 100. run_r100() reruns GOAT on PAPER
and Chung-Lu at r = 100, on round 1's pairs and budgets, for round 1's rules (random, top degree, clustered, spread,
distinguishing, hub-free distinguishing) and the screened rules (the same sets as at r = 300; checked against the
saved scores). Solver seeds are round 1's, so each set meets the same randomness at both regularisers. Results go to
data/best_of_n/goat_r100/<model>/; about 40 minutes with 3 jobs. n_iter is kept: runs that stop within 3 iterations
have stalled (the transport step barely moves; see temperature_audit.ipynb).

Run from the repository root (ds environment); about 1.5 hours with 3 jobs, plus 40 minutes for r = 100:

    python -m selection_rule_comparison.best_of_n
    python -m selection_rule_comparison.best_of_n --algorithms percolation goat
    python -m selection_rule_comparison.best_of_n --r100
    python -m selection_rule_comparison.best_of_n --status

or call run_all() and run_r100() from best_of_n.ipynb. Each pair is saved when it finishes, so rerunning skips
finished pairs; Ctrl-C (or the notebook's interrupt button) stops the workers too.
"""

from __future__ import annotations

import argparse
import datetime
import functools
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

from experiment_common.provenance import collect_environment, thread_limit, write_json
from fixed_pair_seed_set_classification import multi_pair_collection as MP

from . import beyond_top_degree as BT
from . import hub_test as H
from .. import run_comparison as rc
from .. import settings as S
from ..strategies import GraphA, seed_set_features

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]  # selection_rule_comparison/ (this file is in archive/)
REPO_ROOT = EXPERIMENT_DIR.parent
DATA_DIR = EXPERIMENT_DIR / "data"
ROUND1_ROOT = DATA_DIR / "round1" / "main"
OUTPUT_DIR = Path(os.environ.get("BEST_OF_N_OUTPUT_DIR") or DATA_DIR / "best_of_n")
WEIGHTS_PATH = REPO_ROOT / "fixed_pair_seed_set_classification" / "data" / "multi_pair_2026-10-30" / "pooled_weights.json"

ALGORITHMS = ("percolation", "goat", "faq")
MODELS = ("sparse_er", "paper_pa", "chung_lu")
CORE = ("degree", "harmonic", "pair_mass", "uniqueness")
N_DRAWS = 1000
SCREEN_SIZES = (100, 1000)       # best of N by the score; the first 100 draws are part of the 1,000
DEGREE_SCREEN = 1000             # best of N by degree alone
NEW_RULES = (*(f"best_{n}" for n in SCREEN_SIZES), f"degree_{DEGREE_SCREEN}")
CHECK_PAIRS = (0, 1)
ROUND1_RULES = ("top_degree", "clustered", "spread", "distinguishing", "hub_free_distinguishing", "top_harmonic")
DRAW_SEED_BASE = 20261102
ALGORITHM_CODES = {"percolation": 1, "goat": 2, "faq": 3}
SECONDS_PER_RUN = {"percolation": 0.6, "goat": 3.7, "faq": 5.0}   # wall time per run with 3 jobs (round 1)
FAQ_RUNS_PER_SET = 2.0                                              # restarts used on average (round 1)

R100_REGULARIZER = 100                                              # graspologic's default
R100_MODELS = ("paper_pa", "chung_lu")                              # where round 1 used r = 300
R100_DIR = OUTPUT_DIR / f"goat_r{R100_REGULARIZER}"
R100_ROUND1_RULES = ("random", "top_degree", "clustered", "spread", "distinguishing", "hub_free_distinguishing")
R100_RULES = (*R100_ROUND1_RULES, *NEW_RULES)
STALL_ITERATIONS = 3

ALGORITHM_LABELS = {"percolation": "Percolation", "goat": "GOAT", "faq": "SGM (FAQ)"}
MODEL_LABELS = {"sparse_er": "ER", "paper_pa": "PAPER", "chung_lu": "Chung-Lu"}
RULE_LABELS = {"random": "random", "top_degree": "top degree", "clustered": "clustered", "spread": "spread",
               "distinguishing": "distinguishing", "hub_free_distinguishing": "hub-free distinguishing",
               "top_harmonic": "top harmonic", "degree_1000": "best of 1,000 by degree",
               **{f"best_{n}": f"best of {n:,}" for n in SCREEN_SIZES}}


def _now() -> str:
    return datetime.datetime.now().astimezone().strftime("%a %H:%M")


def rule_label(rule: str) -> str:
    return RULE_LABELS.get(rule, rule.replace("_", " "))


# --------------------------------------------------------------------------
# Weights and round 1's stored sets
# --------------------------------------------------------------------------

def load_weights(path: Path | None = None) -> dict:
    """Pooled weights per SD, keyed by (algorithm, model)."""

    saved = json.loads(Path(path or WEIGHTS_PATH).read_text(encoding="utf-8"))
    weights = {}
    for key, entry in saved.items():
        algorithm, model = key.split("/")
        weights[(algorithm, model)] = {f: float(entry["weights_per_sd"][f]) for f in CORE}
    return weights


@functools.lru_cache(maxsize=None)
def _round1_runs(algorithm: str, model: str) -> pd.DataFrame:
    runs = pd.read_csv(ROUND1_ROOT / algorithm / model / "runs.csv")
    runs["success"] = BT._flag(runs["success"])
    return runs


def round1_sets(algorithm: str, model: str, pair_index: int) -> dict:
    """Round 1's seed set for every rule and budget on this pair: {(rule, budget): sorted seeds}."""

    runs = _round1_runs(algorithm, model)
    first = runs[(runs["pair_index"] == pair_index) & (runs["restart"] == 0)]
    return {(r.strategy, int(r.budget)): np.array(sorted(json.loads(r.seed_vertices)), dtype=np.int64)
            for r in first.itertuples()}


# --------------------------------------------------------------------------
# One pair
# --------------------------------------------------------------------------

def task_path(algorithm: str, model: str, pair_index: int) -> Path:
    return OUTPUT_DIR / algorithm / model / f"pair_{pair_index:03d}.csv"


def scores_path(algorithm: str, model: str, pair_index: int) -> Path:
    return OUTPUT_DIR / algorithm / model / f"pair_{pair_index:03d}_scores.csv"


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def screen(graph: GraphA, algorithm: str, budget: int, weights: dict, rng: np.random.Generator):
    """Draw N_DRAWS random sets, score them, and pick each rule's set.

    Returns the draws, a scorer for any set of this size (z-scores and score against these draws), and the index of
    each rule's chosen draw.
    """

    pool = MP.PoolFeatures(graph, MP.UNIQUENESS_CAP[algorithm])
    draws = MP.draw_pool(rng, graph.n, budget, N_DRAWS)
    values = pool.compute(draws)
    mean = {f: float(values[f].mean()) for f in CORE}
    sd = {f: float(values[f].std(ddof=1)) for f in CORE}
    w = np.array([weights[f] for f in CORE])

    def zscores(v: dict) -> np.ndarray:
        return np.column_stack([(v[f] - mean[f]) / sd[f] if sd[f] > 0 else np.zeros(len(v[f])) for f in CORE])

    draw_scores = zscores(values) @ w
    ordered = np.sort(draw_scores)

    def score(sets: np.ndarray) -> pd.DataFrame:
        v = pool.compute(np.atleast_2d(sets))
        z = zscores(v)
        s = z @ w
        table = pd.DataFrame(z, columns=[f"z_{f}" for f in CORE])
        table["score"] = s
        table["score_percentile"] = 100 * np.searchsorted(ordered, s, side="right") / len(ordered)
        return table

    chosen = {f"best_{n}": int(np.argmax(draw_scores[:n])) for n in SCREEN_SIZES}
    chosen[f"degree_{DEGREE_SCREEN}"] = int(np.argmax(values["cut_edges"][:DEGREE_SCREEN]))
    reference = {**{f"ref_mean_{f}": mean[f] for f in CORE}, **{f"ref_sd_{f}": sd[f] for f in CORE}}
    return draws, score, chosen, reference


def pair_job(algorithm: str, model: str, pair_index: int, weights: dict, verify: bool = True,
             blas_threads: int = 3) -> str:
    """The screened rules on one round-1 pair at every round-1 budget (plus the top-degree check on pairs 0-1)."""

    path = task_path(algorithm, model, pair_index)
    name = f"{ALGORITHM_LABELS[algorithm]} {MODEL_LABELS[model]} pair {pair_index}"
    if path.exists():
        return f"{name}: already done"
    started = time.perf_counter()
    with thread_limit(blas_threads):
        seed, A, B, truth = H.load_pair(model, pair_index, verify, algorithm)
        graph = GraphA(A)
        stored = round1_sets(algorithm, model, pair_index)
        budgets = sorted(S.default_budgets("main", algorithm, model))
        pair = (A, B, truth, seed)
        restarts = S.FAQ_MAX_RESTARTS if algorithm == "faq" else 1
        rows, score_rows, cache = [], [], {}
        for budget in budgets:
            rng = np.random.default_rng([DRAW_SEED_BASE, seed, budget, ALGORITHM_CODES[algorithm]])
            draws, score, chosen, reference = screen(graph, algorithm, budget, weights, rng)
            sets = {rule: np.sort(draws[i]) for rule, i in chosen.items()}
            # where every set of this budget sits on the score: the screened ones and round 1's
            named = [(rule, sets[rule], chosen[rule]) for rule in NEW_RULES]
            named += [(rule, seeds, None) for (rule, b), seeds in sorted(stored.items()) if b == budget]
            table = score(np.array([s for _, s, _ in named]))
            for (rule, seeds, index), row in zip(named, table.to_dict("records")):
                score_rows.append({"algorithm": algorithm, "model": model, "pair_index": pair_index, "pair_seed": seed,
                                   "budget": budget, "rule": rule, "draw_index": index,
                                   "seed_vertices": json.dumps([int(v) for v in seeds]), **row, **reference})
            profile = dict(zip(NEW_RULES, table.iloc[:len(NEW_RULES)].to_dict("records")))
            runs = [(rule, sets[rule], profile[rule]) for rule in NEW_RULES]
            if pair_index in CHECK_PAIRS:
                runs.append(("top_degree_check", stored[("top_degree", budget)], {}))
            for rule, seeds, info in runs:
                key = tuple(int(v) for v in seeds)
                features = seed_set_features(graph, np.array(key))
                for restart in range(restarts):
                    cached = (key, restart) in cache
                    if not cached:
                        cache[(key, restart)] = rc.evaluate(algorithm, model, pair, np.array(key), budget, restart)
                    out = cache[(key, restart)]
                    rows.append({
                        "arm": algorithm, "algorithm": algorithm, "model": model,
                        "regularizer": BT.regularizer(algorithm, model), "pair_set": "round1",
                        "pair_index": pair_index, "pair_seed": seed, "rule": rule, "budget": budget,
                        "restart": restart, "solver_seed": out["solver_seed"], "success": bool(out["success"]),
                        "success_090": bool(out["success_090"]), "accuracy": out["final_unseeded_accuracy"],
                        "objective": out["objective"], "n_iter": out["n_iter"], "converged": out["converged"],
                        "runtime_seconds": out["runtime_seconds"], "cached": cached,
                        "seed_vertices": json.dumps(list(key)), **info, **features,
                    })
                    if out["success"]:
                        break
    _write(scores_path(algorithm, model, pair_index), score_rows)
    _write(path, rows)
    runs_done = sum(not row["cached"] for row in rows)
    return f"{name}: {runs_done} runs in {(time.perf_counter() - started) / 60:.1f} min"


# --------------------------------------------------------------------------
# Running everything
# --------------------------------------------------------------------------

def planned_tasks(algorithms=ALGORITHMS, models=MODELS) -> list[tuple[str, str, int]]:
    """Round 1's pairs, by pair index first so every setting moves forward together."""

    tasks = [(a, m, i) for a in algorithms for m in models for i in H.round1_pairs(a, m)]
    return sorted(tasks, key=lambda t: (t[2], ALGORITHMS.index(t[0]), MODELS.index(t[1])))


def estimate_minutes(tasks, jobs: int = 3) -> float:
    seconds = 0.0
    for algorithm, model, pair_index in tasks:
        sets = len(S.default_budgets("main", algorithm, model)) * (len(NEW_RULES) + (pair_index in CHECK_PAIRS))
        seconds += sets * (FAQ_RUNS_PER_SET if algorithm == "faq" else 1) * SECONDS_PER_RUN[algorithm]
    return seconds / max(jobs, 1) / 60


def _ignore_interrupts() -> None:
    """Workers leave Ctrl-C to the main process, which stops them."""

    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_all(algorithms=ALGORITHMS, models=MODELS, jobs: int = 3, blas_threads: int = 3, verify: bool = True,
            weights_path: Path | None = None) -> None:
    """Run every round-1 pair that is not done yet, `jobs` pairs at a time."""

    algorithms = [a for a in ALGORITHMS if a in set(algorithms)]
    models = [m for m in MODELS if m in set(models)]
    tasks = [t for t in planned_tasks(algorithms, models) if not task_path(*t).exists()]
    if not tasks:
        print(f"All pairs are done. Results are in {OUTPUT_DIR}")
        return
    all_weights = load_weights(weights_path)
    missing = sorted({(a, m) for a, m, _ in tasks} - set(all_weights))
    if missing:
        raise KeyError(f"No pooled weights for {missing} in {weights_path or WEIGHTS_PATH}.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "settings.json", {
        "written_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "weights_path": str(weights_path or WEIGHTS_PATH),
        "weights_per_sd": {f"{a}/{m}": w for (a, m), w in all_weights.items()},
        "draws": N_DRAWS, "screen_sizes": list(SCREEN_SIZES), "degree_screen": DEGREE_SCREEN,
        "check_pairs": list(CHECK_PAIRS), "draw_seed_base": DRAW_SEED_BASE,
        "uniqueness_cap": MP.UNIQUENESS_CAP, "faq_max_restarts": S.FAQ_MAX_RESTARTS,
        "goat_regularizer": dict(S.GOAT_REGULARIZER),
        "budgets": {f"{a}/{m}": list(S.default_budgets("main", a, m)) for a in algorithms for m in models},
        "environment": collect_environment(blas_threads)})
    for algorithm in algorithms:
        part = [t for t in tasks if t[0] == algorithm]
        if part:
            print(f"{_now()}  {ALGORITHM_LABELS[algorithm]}: {len(part)} pairs (about "
                  f"{estimate_minutes(part, jobs):.0f} min of the total)", flush=True)
    print(f"{_now()}  {len(tasks)} pairs, {jobs} at a time: about {estimate_minutes(tasks, jobs):.0f} minutes. "
          "Each line below is one finished pair.", flush=True)
    keep_awake = None
    if sys.platform == "darwin":
        import shutil
        import subprocess
        if shutil.which("caffeinate"):
            keep_awake = subprocess.Popen(["caffeinate", "-i"])
    failures = []
    started = time.perf_counter()

    def finished(n, task, message, error, remaining):
        if error is None:
            print(f"{_now()}  [{n}/{len(tasks)}] {message}; about {estimate_minutes(remaining, jobs):.0f} min left",
                  flush=True)
            return
        algorithm, model, pair_index = task
        failures.append(f"{ALGORITHM_LABELS[algorithm]} {MODEL_LABELS[model]} pair {pair_index}")
        print(f"{_now()}  [{n}/{len(tasks)}] {failures[-1]} FAILED:", flush=True)
        traceback.print_exception(type(error), error, error.__traceback__)

    def arguments(task):
        algorithm, model, pair_index = task
        return (algorithm, model, pair_index, all_weights[(algorithm, model)], verify, blas_threads)

    pool = None
    try:
        if jobs <= 1:
            for n, task in enumerate(tasks, 1):
                try:
                    message = pair_job(*arguments(task))
                except Exception as error:  # noqa: BLE001 - report every failure, keep going
                    finished(n, task, None, error, [])
                else:
                    finished(n, task, message, None, tasks[n:])
        else:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor, as_completed
            pool = ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn"),
                                       initializer=_ignore_interrupts)
            futures = {pool.submit(pair_job, *arguments(task)): task for task in tasks}
            for n, future in enumerate(as_completed(futures), 1):
                try:
                    message = future.result()
                except Exception as error:  # noqa: BLE001 - report every failure, keep going
                    finished(n, futures[future], None, error, [])
                else:
                    finished(n, futures[future], message, None, [t for f, t in futures.items() if not f.done()])
            pool.shutdown(wait=True)
    except KeyboardInterrupt:
        # Stop the workers too: the notebook's interrupt button only reaches this process.
        if pool is not None:
            workers = list((getattr(pool, "_processes", None) or {}).values())
            pool.shutdown(wait=False, cancel_futures=True)
            for worker in workers:
                worker.terminate()
        print(f"\n{_now()}  Stopped. Finished pairs are kept; rerun to resume.", flush=True)
        return
    finally:
        if keep_awake is not None:
            keep_awake.terminate()
    if failures:
        raise RuntimeError(f"Failed: {', '.join(failures)}. Rerun to retry them (finished pairs are kept).")
    print(f"{_now()}  Done in {(time.perf_counter() - started) / 60:.0f} min. Results are in {OUTPUT_DIR}", flush=True)


def status(algorithms=ALGORITHMS, models=MODELS) -> pd.DataFrame:
    rows = []
    for algorithm in algorithms:
        for model in models:
            planned = H.round1_pairs(algorithm, model)
            done = sum(task_path(algorithm, model, i).exists() for i in planned)
            rows.append({"algorithm": ALGORITHM_LABELS[algorithm], "model": MODEL_LABELS[model],
                         "pairs planned": len(planned), "pairs done": done})
    return pd.DataFrame(rows).set_index(["algorithm", "model"])


# --------------------------------------------------------------------------
# GOAT at r = 100 (robustness check)
# --------------------------------------------------------------------------

def evaluate_goat(model: str, pair, seeds: np.ndarray, budget: int, regularizer: float, restart: int = 0) -> dict:
    """One GOAT run with round 1's settings and solver seed, at the given transport regulariser."""

    A, B, truth, pair_seed = pair
    seed_value = rc.solver_seed(pair_seed, budget, restart)
    out = rc._run_sgm_once(A, B, truth, seeds, seed_value, S.SGM["max_iter"], S.SGM["tol"], transport=True,
                           transport_regularizer=regularizer, transport_tol=S.SGM["transport_tol"],
                           transport_max_iter=S.SGM["transport_max_iter"])
    accuracy = out["final_unseeded_accuracy"]
    return {"solver_seed": seed_value, "final_unseeded_accuracy": accuracy,
            "success": accuracy >= S.SUCCESS_THRESHOLD["goat"], "success_090": accuracy >= 0.9,
            "objective": out["sgm_objective_score"], "n_iter": out["sgm_n_iter"], "converged": out["sgm_converged"],
            "runtime_seconds": out["sgm_runtime_seconds"]}


def r100_path(model: str, pair_index: int) -> Path:
    return R100_DIR / model / f"pair_{pair_index:03d}.csv"


def _saved_screened_sets(model: str, pair_index: int) -> dict:
    """The screened sets of the r = 300 run on this pair, {(rule, budget): seeds}, if that run is saved."""

    path = scores_path("goat", model, pair_index)
    if not path.exists():
        return {}
    saved = pd.read_csv(path)
    saved = saved[saved["rule"].isin(NEW_RULES)]
    return {(r.rule, int(r.budget)): tuple(json.loads(r.seed_vertices)) for r in saved.itertuples()}


def r100_job(model: str, pair_index: int, weights: dict, verify: bool = True, blas_threads: int = 3) -> str:
    """GOAT at r = 100 on one round-1 pair: round 1's rules and the screened rules at every round-1 budget."""

    path = r100_path(model, pair_index)
    name = f"GOAT r = {R100_REGULARIZER} {MODEL_LABELS[model]} pair {pair_index}"
    if path.exists():
        return f"{name}: already done"
    started = time.perf_counter()
    with thread_limit(blas_threads):
        seed, A, B, truth = H.load_pair(model, pair_index, verify, "goat")
        graph = GraphA(A)
        stored = round1_sets("goat", model, pair_index)
        saved = _saved_screened_sets(model, pair_index)
        pair = (A, B, truth, seed)
        rows, cache = [], {}
        for budget in sorted(S.default_budgets("main", "goat", model)):
            rng = np.random.default_rng([DRAW_SEED_BASE, seed, budget, ALGORITHM_CODES["goat"]])
            draws, _score, chosen, _reference = screen(graph, "goat", budget, weights, rng)
            sets = {rule: np.sort(draws[i]) for rule, i in chosen.items()}
            for rule in NEW_RULES:
                if (rule, budget) in saved and tuple(int(v) for v in sets[rule]) != saved[(rule, budget)]:
                    raise RuntimeError(f"{name}: {rule} at k = {budget} differs from the set saved at r = 300.")
            sets.update({rule: seeds for (rule, b), seeds in stored.items() if b == budget})
            for rule in R100_RULES:
                if rule not in sets:
                    continue
                key = tuple(int(v) for v in sets[rule])
                cached = key in cache  # the same set at the same budget meets the same solver seed
                if not cached:
                    cache[key] = evaluate_goat(model, pair, np.array(key), budget, R100_REGULARIZER)
                out = cache[key]
                rows.append({
                    "arm": "goat", "algorithm": "goat", "model": model, "regularizer": R100_REGULARIZER,
                    "pair_set": "round1", "pair_index": pair_index, "pair_seed": seed, "rule": rule,
                    "budget": budget, "restart": 0, "solver_seed": out["solver_seed"],
                    "success": bool(out["success"]), "success_090": bool(out["success_090"]),
                    "accuracy": out["final_unseeded_accuracy"], "objective": out["objective"],
                    "n_iter": out["n_iter"], "converged": out["converged"],
                    "runtime_seconds": out["runtime_seconds"], "cached": cached,
                    "seed_vertices": json.dumps(list(key)), **seed_set_features(graph, np.array(key)),
                })
    _write(path, rows)
    return f"{name}: {sum(not r['cached'] for r in rows)} runs in {(time.perf_counter() - started) / 60:.1f} min"


def _run_tasks(tasks: list, call, describe, seconds, jobs: int) -> list[str]:
    """Run call(*task) for every task, `jobs` at a time; returns the tasks that failed (described)."""

    failures, started = [], time.perf_counter()

    def finished(n, task, message, error, remaining):
        if error is None:
            left = sum(seconds(t) for t in remaining) / max(jobs, 1) / 60
            print(f"{_now()}  [{n}/{len(tasks)}] {message}; about {left:.0f} min left", flush=True)
            return
        failures.append(describe(task))
        print(f"{_now()}  [{n}/{len(tasks)}] {failures[-1]} FAILED:", flush=True)
        traceback.print_exception(type(error), error, error.__traceback__)

    keep_awake = None
    if sys.platform == "darwin":
        import shutil
        import subprocess
        if shutil.which("caffeinate"):
            keep_awake = subprocess.Popen(["caffeinate", "-i"])
    pool = None
    try:
        if jobs <= 1:
            for n, task in enumerate(tasks, 1):
                try:
                    message = call(*task)
                except Exception as error:  # noqa: BLE001 - report every failure, keep going
                    finished(n, task, None, error, [])
                else:
                    finished(n, task, message, None, tasks[n:])
        else:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor, as_completed
            pool = ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn"),
                                       initializer=_ignore_interrupts)
            futures = {pool.submit(call, *task): task for task in tasks}
            for n, future in enumerate(as_completed(futures), 1):
                try:
                    message = future.result()
                except Exception as error:  # noqa: BLE001 - report every failure, keep going
                    finished(n, futures[future], None, error, [])
                else:
                    finished(n, futures[future], message, None, [t for f, t in futures.items() if not f.done()])
            pool.shutdown(wait=True)
    except KeyboardInterrupt:
        if pool is not None:
            workers = list((getattr(pool, "_processes", None) or {}).values())
            pool.shutdown(wait=False, cancel_futures=True)
            for worker in workers:
                worker.terminate()
        print(f"\n{_now()}  Stopped. Finished pairs are kept; rerun to resume.", flush=True)
        raise
    finally:
        if keep_awake is not None:
            keep_awake.terminate()
    print(f"{_now()}  Finished in {(time.perf_counter() - started) / 60:.0f} min.", flush=True)
    return failures


def run_r100(models=R100_MODELS, jobs: int = 3, blas_threads: int = 3, verify: bool = True,
             weights_path: Path | None = None) -> None:
    """GOAT at r = 100 on round 1's pairs (PAPER and Chung-Lu), `jobs` pairs at a time."""

    models = [m for m in R100_MODELS if m in set(models)]
    tasks = sorted([(m, i) for m in models for i in H.round1_pairs("goat", m) if not r100_path(m, i).exists()],
                   key=lambda t: (t[1], R100_MODELS.index(t[0])))
    if not tasks:
        print(f"GOAT at r = {R100_REGULARIZER}: all pairs are done. Results are in {R100_DIR}")
        return
    weights = load_weights(weights_path)
    R100_DIR.mkdir(parents=True, exist_ok=True)
    write_json(R100_DIR / "settings.json", {
        "written_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "regularizer": R100_REGULARIZER, "rules": list(R100_RULES), "models": list(models),
        "solver": {**S.SGM, "transport": True, "transport_regularizer": R100_REGULARIZER},
        "solver_seeds": "round 1's (pair seed, budget, restart 0)", "weights_path": str(weights_path or WEIGHTS_PATH),
        "environment": collect_environment(blas_threads)})
    sets_per_pair = len(R100_RULES) * len(S.default_budgets("main", "goat", models[0]))
    seconds = lambda task: sets_per_pair * SECONDS_PER_RUN["goat"]  # noqa: E731
    print(f"{_now()}  GOAT at r = {R100_REGULARIZER}: {len(tasks)} pairs, {jobs} at a time: about "
          f"{sum(seconds(t) for t in tasks) / max(jobs, 1) / 60:.0f} minutes. Each line below is one finished pair.",
          flush=True)
    try:
        failures = _run_tasks([(m, i, weights[("goat", m)], verify, blas_threads) for m, i in tasks], r100_job,
                              lambda t: f"GOAT r = {R100_REGULARIZER} {MODEL_LABELS[t[0]]} pair {t[1]}",
                              seconds, jobs)
    except KeyboardInterrupt:
        return
    if failures:
        raise RuntimeError(f"Failed: {', '.join(failures)}. Rerun to retry them (finished pairs are kept).")


def r100_status(models=R100_MODELS) -> pd.DataFrame:
    rows = [{"algorithm": f"GOAT, r = {R100_REGULARIZER}", "model": MODEL_LABELS[m],
             "pairs planned": len(H.round1_pairs("goat", m)),
             "pairs done": sum(r100_path(m, i).exists() for i in H.round1_pairs("goat", m))} for m in models]
    return pd.DataFrame(rows).set_index(["algorithm", "model"])


# --------------------------------------------------------------------------
# Analysis helpers for best_of_n.ipynb
# --------------------------------------------------------------------------

def _read(pattern: str, algorithms, models) -> pd.DataFrame:
    files = [f for a in algorithms for m in models for f in sorted((OUTPUT_DIR / a / m).glob(pattern))]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def read_runs(algorithms=ALGORITHMS, models=MODELS) -> pd.DataFrame:
    """Every solver run of this experiment (one row per restart)."""

    runs = _read("pair_[0-9][0-9][0-9].csv", algorithms, models)
    if runs.empty:
        raise FileNotFoundError(f"No results in {OUTPUT_DIR} yet. Run run_all() first.")
    for column in ("success", "success_090", "cached"):
        runs[column] = BT._flag(runs[column])
    return runs


def read_scores(algorithms=ALGORITHMS, models=MODELS) -> pd.DataFrame:
    """The score of every screened set and of round 1's sets, per pair and budget."""

    return _read("pair_*_scores.csv", algorithms, models)


def check_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Round 1's top-degree sets run again on pairs 0-1: accuracy and success must equal round 1's."""

    rows = []
    check = runs[runs["rule"] == "top_degree_check"]
    for (algorithm, model), part in check.groupby(["algorithm", "model"], sort=False):
        old = _round1_runs(algorithm, model)
        old = old[old["strategy"] == "top_degree"].set_index(["pair_index", "budget", "restart"])
        merged = part.join(old[["final_unseeded_accuracy", "success"]], on=["pair_index", "budget", "restart"],
                           rsuffix="_round1")
        same = ((merged["accuracy"] - merged["final_unseeded_accuracy"]).abs() < 1e-9) & (
            merged["success"] == merged["success_round1"])
        rows.append({"algorithm": ALGORITHM_LABELS[algorithm], "model": MODEL_LABELS[model],
                     "runs compared": len(merged), "identical to round 1": int(same.sum())})
    return pd.DataFrame(rows).set_index(["algorithm", "model"]) if rows else pd.DataFrame()


def load_evaluations(algorithms=ALGORITHMS, models=MODELS) -> pd.DataFrame:
    """One row per (setting, pair, rule, budget): the screened rules and round 1's rules on the pairs done so far.

    success means any restart succeeded (FAQ's best of 3), as in round 1.
    """

    runs = read_runs(algorithms, models)
    frames = [BT._collapse(runs[runs["rule"] != "top_degree_check"])]
    for (algorithm, model), part in runs.groupby(["algorithm", "model"], sort=False):
        old = BT._reused(algorithm, model, sorted(part["pair_index"].unique()))
        old = old[old["rule"].isin(("random", *ROUND1_RULES))]
        frames.append(BT._collapse(old))
    ev = pd.concat(frames, ignore_index=True)
    ev["success"] = ev["success"].astype(bool)
    return ev


def queries_needed(ev: pd.DataFrame) -> pd.DataFrame:
    """Smallest budget that succeeds, per pair and rule (one grid step past the largest budget if none does)."""

    return BT.queries_needed(ev)


def round1_best(ev: pd.DataFrame, queries: pd.DataFrame) -> dict:
    """Round 1's best rule per setting: highest mean success over the budget grid; fewer queries break ties."""

    best = {}
    for (algorithm, model), part in ev.groupby(["algorithm", "model"], sort=False):
        part = part[part["rule"].isin(ROUND1_RULES)]
        grid = part.assign(success=part["success"].astype(float)).groupby(["rule", "pair_index"])["success"].mean()
        mean_success = grid.groupby("rule").mean()
        q = queries[(queries["arm"] == algorithm) & (queries["model"] == model)].groupby("rule")["queries"].mean()
        order = sorted(mean_success.index, key=lambda r: (-round(mean_success[r], 10), q.get(r, np.inf)))
        best[(algorithm, model)] = order[0]
    return best


def success_curves(ev: pd.DataFrame) -> pd.DataFrame:
    """Share of pairs that succeed, per setting, rule and budget."""

    data = ev.assign(success=ev["success"].astype(float))
    return data.groupby(["algorithm", "model", "rule", "budget"], as_index=False)["success"].mean()


def _grid_means(ev: pd.DataFrame, algorithm: str, model: str) -> pd.DataFrame:
    part = ev[(ev["algorithm"] == algorithm) & (ev["model"] == model)]
    return part.assign(success=part["success"].astype(float)).pivot_table(
        index="pair_index", columns="rule", values="success", aggfunc="mean")


def _settings(frame: pd.DataFrame) -> list[tuple[str, str]]:
    present = set(zip(frame["algorithm"], frame["model"]))
    return [(a, m) for a in ALGORITHMS for m in MODELS if (a, m) in present]


def _rules_shown(best_rule: str | None) -> list[str]:
    """Random, top degree, round 1's best (if it is not top degree) and the screened rules."""

    return list(dict.fromkeys(r for r in ("random", "top_degree", best_rule, *NEW_RULES) if r))


def _shown_label(rule: str, best_rule: str | None) -> str:
    return f"{rule_label(rule)} (round 1's best)" if rule == best_rule else rule_label(rule)


def summary_table(ev: pd.DataFrame, queries: pd.DataFrame, best: dict) -> pd.DataFrame:
    """Per setting: mean success over the budget grid and mean queries needed, for the rules compared."""

    rows = []
    for algorithm, model in _settings(ev):
        grid = _grid_means(ev, algorithm, model)
        best_rule = best[(algorithm, model)]
        for rule in _rules_shown(best_rule):
            if rule not in grid:
                continue
            success = BT._mean_interval(grid[rule].dropna())
            q = BT.mean_queries(queries, algorithm, model, rule)
            rows.append({"algorithm": ALGORITHM_LABELS[algorithm], "model": MODEL_LABELS[model],
                         "rule": _shown_label(rule, best_rule), "pairs": len(grid[rule].dropna()),
                         "mean success over budgets": f"{success[0]:.2f} [{success[1]:.2f}, {success[2]:.2f}]",
                         "mean queries needed": f"{q['mean']:.1f} [{q['low']:.1f}, {q['high']:.1f}]"})
    return pd.DataFrame(rows).set_index(["algorithm", "model", "rule"])


COMPARISONS = (("best_1000", "random"), ("best_1000", "top_degree"), ("best_1000", "round1_best"),
               ("best_1000", "degree_1000"), ("best_100", "random"), ("best_100", "top_degree"),
               ("best_100", "round1_best"))


def comparison_table(ev: pd.DataFrame, queries: pd.DataFrame, best: dict, comparisons=COMPARISONS) -> pd.DataFrame:
    """Pair by pair: queries needed (fewer / same / more, sign test, mean change) and mean success over the grid.

    When round 1's best rule is top degree, the comparison with it appears once, as top degree.
    """

    rows = []
    for algorithm, model in _settings(ev):
        grid = _grid_means(ev, algorithm, model)
        best_rule = best[(algorithm, model)]
        done = set()
        for first, second in comparisons:
            second_rule = best_rule if second == "round1_best" else second
            if (first, second_rule) in done:
                continue
            done.add((first, second_rule))
            result = BT.compare(queries, algorithm, model, first, second_rule)
            if not result or first not in grid or second_rule not in grid:
                continue
            difference = (grid[first] - grid[second_rule]).dropna()
            mean, low, high = BT._mean_interval(difference)
            rows.append({"algorithm": ALGORITHM_LABELS[algorithm], "model": MODEL_LABELS[model],
                         "rule": rule_label(first), "against": _shown_label(second_rule, best_rule),
                         "pairs": result["pairs"],
                         "needs fewer / same / more queries": f"{result['fewer']} / {result['same']} / {result['more']}",
                         "p": round(result["p"], 4) if result["p"] == result["p"] else np.nan,
                         "change in queries": result["mean change"],
                         "change in mean success": f"{mean:+.2f} [{low:+.2f}, {high:+.2f}]"})
    return pd.DataFrame(rows).set_index(["algorithm", "model", "rule", "against"])


def score_positions(scores: pd.DataFrame, best: dict) -> pd.DataFrame:
    """Where each rule's sets sit on the score: median percentile among the 1,000 random draws, and mean z-scores."""

    rows = []
    for algorithm, model in _settings(scores):
        part = scores[(scores["algorithm"] == algorithm) & (scores["model"] == model)]
        best_rule = best.get((algorithm, model))
        for rule in _rules_shown(best_rule):
            sub = part[part["rule"] == rule]
            if sub.empty:
                continue
            rows.append({"algorithm": ALGORITHM_LABELS[algorithm], "model": MODEL_LABELS[model],
                         "rule": _shown_label(rule, best_rule),
                         "median score percentile": round(float(sub["score_percentile"].median()), 1),
                         **{f"z {f.replace('_', ' ')}": round(float(sub[f"z_{f}"].mean()), 1) for f in CORE}})
    return pd.DataFrame(rows).set_index(["algorithm", "model", "rule"])


# --------------------------------------------------------------------------
# Individual seed sets, as in section 10 of rule_comparison.ipynb
# --------------------------------------------------------------------------

def seed_set_table(ev: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """One row per (setting, pair, budget, rule): degree and the other features, the pooled score, and the outcome.

    accuracy is the run with the best objective (FAQ: among its restarts), as in rule_comparison.ipynb.
    """

    keys = ["algorithm", "model", "pair_index", "budget", "rule"]
    columns = [*keys, "score", "score_percentile", *[f"z_{f}" for f in CORE]]
    table = ev.merge(scores[columns].drop_duplicates(keys), on=keys, how="left", validate="one_to_one")
    missing = table["score"].isna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} evaluations have no saved score.")
    return table


def scatter_budget(table: pd.DataFrame, algorithm: str, model: str) -> int:
    """The budget where the mean success over all rules' sets is closest to 50% (ties: the smaller budget)."""

    part = table[(table["algorithm"] == algorithm) & (table["model"] == model)]
    rates = part.assign(success=part["success"].astype(float)).groupby("budget")["success"].mean()
    return int(min(rates.index, key=lambda b: (abs(rates[b] - 0.5), b)))


def binned_accuracy(frame: pd.DataFrame, x_column: str, n_bins: int = 5, n_boot: int = 2000,
                    seed: int = 2026) -> pd.DataFrame:
    """Mean accuracy in equal-count bins of x (tied x values stay together); 95% intervals resample whole pairs.

    The same summary as rule_comparison.ipynb, section 10. A bin's interval is omitted with fewer than two pairs or
    when fewer than 90% of resamples contain the bin.
    """

    data = frame[["pair_index", x_column, "accuracy"]].copy()
    data = data[np.isfinite(data[x_column]) & np.isfinite(data["accuracy"])]
    columns = ["bin", "n_sets", "n_pairs", "x_min", "x_max", "x_mean", "mean_accuracy", "ci_low", "ci_high"]
    if data.empty:
        return pd.DataFrame(columns=columns)
    unique = data[x_column].nunique()
    data["_bin"] = 0 if unique == 1 or n_bins == 1 else pd.qcut(data[x_column], q=min(n_bins, unique), labels=False,
                                                                duplicates="drop")
    pairs, bins = np.sort(data["pair_index"].unique()), np.sort(data["_bin"].unique())
    totals = np.zeros((len(pairs), len(bins)))
    counts = np.zeros_like(totals)
    for i, pair in enumerate(pairs):
        part = data[data["pair_index"] == pair]
        for j, bin_id in enumerate(bins):
            values = part.loc[part["_bin"] == bin_id, "accuracy"]
            counts[i, j], totals[i, j] = len(values), values.sum()
    draws = np.random.default_rng(seed).integers(0, len(pairs), size=(n_boot, len(pairs)))
    boot_count, boot_sum = counts[draws].sum(axis=1), totals[draws].sum(axis=1)
    boot = np.divide(boot_sum, boot_count, out=np.full_like(boot_sum, np.nan), where=boot_count > 0)
    rows = []
    for j, bin_id in enumerate(bins):
        part = data[data["_bin"] == bin_id]
        finite = boot[:, j][np.isfinite(boot[:, j])]
        low = high = np.nan
        if part["pair_index"].nunique() >= 2 and len(finite) >= 0.9 * n_boot:
            low, high = np.percentile(finite, [2.5, 97.5])
        rows.append({"bin": int(bin_id) + 1, "n_sets": len(part), "n_pairs": part["pair_index"].nunique(),
                     "x_min": part[x_column].min(), "x_max": part[x_column].max(), "x_mean": part[x_column].mean(),
                     "mean_accuracy": part["accuracy"].mean(), "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows, columns=columns)


def against_top_degree(frame: pd.DataFrame, columns, include_random: bool = False) -> pd.DataFrame:
    """Each other rule's values minus top degree's on the same pair (one budget per frame)."""

    reference = frame[frame["rule"] == "top_degree"].set_index("pair_index")[list(columns)]
    excluded = ["top_degree"] + ([] if include_random else ["random"])
    others = frame[~frame["rule"].isin(excluded)].copy()
    missing = ~others["pair_index"].isin(reference.index)
    if missing.any():
        raise ValueError("Some rules have no top-degree set on the same pair.")
    for column in columns:
        others[column] = others[column].to_numpy(float) - reference.loc[others["pair_index"], column].to_numpy(float)
    return others


RULE_STYLE = {   # validated (dataviz): orange, blue and aqua for all pairs, plus neutrals; shapes tell rules apart
    "random": dict(color="#9a9a9a", marker="o"),
    "top_degree": dict(color="#2b2b2b", marker="s"),
    "top_harmonic": dict(color="#eb6834", marker="D"),
    "clustered": dict(color="#eb6834", marker="^"),
    "spread": dict(color="#eb6834", marker="v"),
    "distinguishing": dict(color="#eb6834", marker="P"),
    "hub_free_distinguishing": dict(color="#eb6834", marker="*"),
    "degree_1000": dict(color="#1baf7a", marker="X"),
    "best_100": dict(color="#5598e7", marker="o"),
    "best_1000": dict(color="#1c5cab", marker="o"),
}
BIN_STYLE = dict(marker="D", s=62, facecolor="white", edgecolor="#2b2b2b", linewidths=1.6)
X_LABELS = {"f_degree": "degree (seed-to-nonseed edges per seed)", "score": "weighted score",
            "f_harmonic": "harmonic", "f_pair_mass": "pair mass", "f_signature_r2": "uniqueness (cap 2)",
            "f_signature_r3": "uniqueness (cap 3)", "score_percentile": "score percentile"}


def _rules_in(frame: pd.DataFrame) -> list[str]:
    present = set(frame["rule"])
    return [r for r in RULE_STYLE if r in present] + sorted(present - set(RULE_STYLE))


def rule_summary(frame: pd.DataFrame, best_rule: str | None = None, n_boot: int = 2000) -> pd.DataFrame:
    """Per rule at one budget: mean degree, score and accuracy (95% bootstrap over pairs), and success rate."""

    rows = []
    for rule in _rules_in(frame):
        part = frame[frame["rule"] == rule]
        mean, low, high = BT._mean_interval(part["accuracy"].to_numpy(float), n_boot)
        rows.append({"rule": _shown_label(rule, best_rule), "sets": len(part),
                     "mean degree": part["f_degree"].mean(), "mean score": part["score"].mean(),
                     "median score percentile": part["score_percentile"].median(),
                     "mean accuracy": f"{mean:.2f} [{low:.2f}, {high:.2f}]",
                     "success rate": part["success"].astype(float).mean()})
    return pd.DataFrame(rows).set_index("rule").round(2)


def rule_means(frame: pd.DataFrame, columns, n_boot: int = 2000) -> pd.DataFrame:
    """Per rule: the mean of each column over its sets (one per pair), with 95% bootstrap intervals over pairs."""

    rows = []
    for rule in _rules_in(frame):
        part = frame[frame["rule"] == rule]
        row = {"rule": rule, "pairs": part["pair_index"].nunique()}
        for column in columns:
            mean, low, high = BT._mean_interval(part[column].to_numpy(float), n_boot)
            row.update({column: mean, f"{column}_low": low, f"{column}_high": high})
        rows.append(row)
    return pd.DataFrame(rows)


def top_degree_changes(frame: pd.DataFrame, x_columns=("f_degree", "score"), best_rule: str | None = None,
                       include_random: bool = False, n_boot: int = 2000) -> pd.DataFrame:
    """Each rule against top degree on the same pairs: mean changes (95% bootstrap over pairs), and the pairs where
    only one of the two succeeded."""

    columns = [*x_columns, "accuracy"]
    changes = against_top_degree(frame, columns, include_random)
    means = rule_means(changes, columns, n_boot)
    top = frame[frame["rule"] == "top_degree"].set_index("pair_index")["success"].astype(bool)
    rows = []
    for row in means.to_dict("records"):
        mine = frame[frame["rule"] == row["rule"]].set_index("pair_index")["success"].astype(bool)
        mine, other = mine.align(top, join="inner")
        out = {"rule": _shown_label(row["rule"], best_rule), "pairs": row["pairs"]}
        for column in columns:
            label = "accuracy" if column == "accuracy" else X_LABELS.get(column, column).split(" (")[0]
            out[f"change in {label}"] = (f"{row[column]:+.2f} [{row[column + '_low']:+.2f}, "
                                         f"{row[column + '_high']:+.2f}]")
        out["succeeds where top degree fails"] = int((mine & ~other).sum())
        out["fails where top degree succeeds"] = int((~mine & other).sum())
        rows.append(out)
    return pd.DataFrame(rows).set_index("rule")


def draw_seed_scatter(frame: pd.DataFrame, algorithm: str, model: str, budget: int, best_rule: str | None,
                      x_columns=("f_degree", "score"), paired: bool = False, summary: str = "rules",
                      n_bins: int = 5, n_boot: int = 2000, seed: int = 2026):
    """One small dot per rule's seed set on one pair (section 10 of rule_comparison.ipynb, with best of N added).

    summary='rules' (default): each rule's mean over pairs as a large marker in the rule's own colour and shape, with
    95% intervals over pairs on both axes. summary='bins': section 10's pooled means in equal-count bins of x (open
    diamonds) instead. Returns the figure and the table behind the large markers.
    """

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rules = _rules_in(frame)
    # top degree after round 1's other rules, so it stays visible where they picked the same set; best of N on top
    drawing = ([r for r in rules if r not in ("top_degree", *NEW_RULES)] + [r for r in rules if r == "top_degree"]
               + [r for r in rules if r in NEW_RULES])
    style = {r: RULE_STYLE.get(r, dict(color="#6b6b6b", marker="o")) for r in rules}
    means = rule_means(frame, [*x_columns, "accuracy"], n_boot) if summary == "rules" else None
    fig, axes = plt.subplots(1, len(x_columns), figsize=(5.9 * len(x_columns), 4.8), squeeze=False)
    tables = []
    for j, (ax, x) in enumerate(zip(axes[0], x_columns)):
        for rule in drawing:
            part = frame[frame["rule"] == rule]
            ax.scatter(part[x], part["accuracy"], s=24, marker=style[rule]["marker"], color=style[rule]["color"],
                       edgecolors="none", alpha=0.3 if summary == "rules" else 0.85, zorder=3)
        if summary == "rules":
            for rule in drawing:
                m = means[means["rule"] == rule].iloc[0]
                color, marker = style[rule]["color"], style[rule]["marker"]
                if np.isfinite(m[f"{x}_low"]):
                    ax.hlines(m["accuracy"], m[f"{x}_low"], m[f"{x}_high"], color=color, linewidth=1.5, zorder=5)
                if np.isfinite(m["accuracy_low"]):
                    ax.vlines(m[x], m["accuracy_low"], m["accuracy_high"], color=color, linewidth=1.5, zorder=5)
                ax.plot(m[x], m["accuracy"], marker=marker, markersize=16 if marker == "*" else 11, color=color,
                        markeredgecolor="white", markeredgewidth=1.3, linestyle="", zorder=6)
        else:
            bins = binned_accuracy(frame, x, n_bins, n_boot, seed + j)
            shown = bins.dropna(subset=["ci_low", "ci_high"])
            ax.vlines(shown["x_mean"], shown["ci_low"], shown["ci_high"], color="#2b2b2b", linewidth=1.8, zorder=5)
            ax.scatter(bins["x_mean"], bins["mean_accuracy"], zorder=6, **BIN_STYLE)
            tables.append(bins.assign(algorithm=algorithm, model=model, budget=budget,
                                      view="paired" if paired else "absolute", x_column=x))
        if paired:
            ax.axhline(0, color="#6b7280", linewidth=0.9, zorder=1)
            ax.axvline(0, color="#6b7280", linewidth=0.9, zorder=1)
        ax.set(xlabel=X_LABELS.get(x, x) + (" − top degree" if paired else ""),
               ylabel="accuracy − top degree" if paired else "final unseeded accuracy",
               ylim=(-1.05, 1.05) if paired else (-0.04, 1.05))
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.7, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [Line2D([0], [0], marker=style[r]["marker"], linestyle="", color=style[r]["color"],
                      markersize=11 if style[r]["marker"] == "*" else 8,
                      label=_shown_label(r, best_rule)) for r in rules]
    if summary == "rules":
        handles += [Line2D([0], [0], marker="o", linestyle="-", color="#6b6b6b", markersize=9, linewidth=1.5,
                           markeredgecolor="white", label="large: mean over pairs (95% interval)"),
                    Line2D([0], [0], marker="o", linestyle="", color="#6b6b6b", alpha=0.35, markersize=5,
                           label="small: one pair's set")]
    else:
        handles.append(Line2D([0], [0], marker="D", linestyle="", markersize=7, markerfacecolor="white",
                              markeredgecolor="#2b2b2b", markeredgewidth=1.6, label="binned mean accuracy"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, handletextpad=0.3,
               columnspacing=1.2)
    fig.suptitle(f"{ALGORITHM_LABELS[algorithm]} · {MODEL_LABELS[model]} · {budget} seeds · "
                 f"{frame['pair_index'].nunique()} pairs" + (" · compared with top degree" if paired else ""),
                 fontsize=13, fontweight="bold")
    rows_of_legend = int(np.ceil(len(handles) / 4))
    fig.tight_layout(rect=(0, 0.045 * rows_of_legend + 0.02, 1, 0.95), w_pad=2.2)
    if summary == "rules":
        table = means.assign(algorithm=algorithm, model=model, budget=budget, view="paired" if paired else "absolute")
    else:
        table = pd.concat(tables, ignore_index=True)
    return fig, table


# --------------------------------------------------------------------------
# GOAT at r = 100: analysis
# --------------------------------------------------------------------------

def read_r100(models=R100_MODELS) -> pd.DataFrame:
    files = [f for m in models for f in sorted((R100_DIR / m).glob("pair_[0-9][0-9][0-9].csv"))]
    if not files:
        raise FileNotFoundError(f"No results in {R100_DIR} yet. Run run_r100() first.")
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for column in ("success", "success_090", "cached"):
        runs[column] = BT._flag(runs[column])
    return runs


def r100_evaluations(models=R100_MODELS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same rules on the same pairs and budgets at r = 300 (round 1 and the screened runs) and at r = 100."""

    runs100 = read_r100(models)
    ev100 = BT._collapse(runs100)
    ev100["success"] = ev100["success"].astype(bool)
    pairs = sorted(runs100["pair_index"].unique())
    ev300 = load_evaluations(["goat"], models)
    ev300 = ev300[ev300["rule"].isin(R100_RULES) & ev300["pair_index"].isin(pairs)].reset_index(drop=True)
    return ev300, ev100


def _iterations_r300(models=R100_MODELS) -> pd.DataFrame:
    """n_iter of the r = 300 runs: round 1's runs for its rules, this experiment's runs for the screened ones."""

    frames = []
    for model in models:
        old = _round1_runs("goat", model)
        old = old[(old["restart"] == 0) & old["strategy"].isin(R100_ROUND1_RULES)]
        frames.append(old.rename(columns={"strategy": "rule"})[["model", "pair_index", "rule", "budget", "n_iter"]])
    new = read_runs(["goat"], models)
    frames.append(new[new["rule"].isin(NEW_RULES)][["model", "pair_index", "rule", "budget", "n_iter"]])
    return pd.concat(frames, ignore_index=True)


def regularizer_table(ev300: pd.DataFrame, ev100: pd.DataFrame, runs100: pd.DataFrame) -> pd.DataFrame:
    """Per model and rule: mean success over the budget grid at r = 300 and r = 100, the change (r = 100 minus
    r = 300, 95% bootstrap over pairs), mean queries needed, and the share of runs that stalled at each."""

    q300, q100 = queries_needed(ev300), queries_needed(ev100)
    iterations300 = _iterations_r300(sorted(set(ev100["model"])))
    rows = []
    for model in [m for m in R100_MODELS if m in set(ev100["model"])]:
        grid300, grid100 = _grid_means(ev300, "goat", model), _grid_means(ev100, "goat", model)
        for rule in R100_RULES:
            if rule not in grid300 or rule not in grid100:
                continue
            both = pd.concat([grid300[rule].rename("r300"), grid100[rule].rename("r100")], axis=1).dropna()
            change = BT._mean_interval(both["r100"] - both["r300"])
            stalled300 = iterations300[(iterations300["model"] == model) & (iterations300["rule"] == rule)]["n_iter"]
            stalled100 = runs100[(runs100["model"] == model) & (runs100["rule"] == rule)]["n_iter"]
            rows.append({
                "model": MODEL_LABELS[model], "rule": rule_label(rule), "pairs": len(both),
                "mean success, r = 300": round(float(both["r300"].mean()), 2),
                "mean success, r = 100": round(float(both["r100"].mean()), 2),
                "change": f"{change[0]:+.2f} [{change[1]:+.2f}, {change[2]:+.2f}]",
                "queries needed, r = 300": round(BT.mean_queries(q300, "goat", model, rule)["mean"], 1),
                "queries needed, r = 100": round(BT.mean_queries(q100, "goat", model, rule)["mean"], 1),
                "stalled, r = 300": round(float((pd.to_numeric(stalled300) <= STALL_ITERATIONS).mean()), 2),
                "stalled, r = 100": round(float((pd.to_numeric(stalled100) <= STALL_ITERATIONS).mean()), 2),
            })
    return pd.DataFrame(rows).set_index(["model", "rule"])


# --------------------------------------------------------------------------
# The success-versus-score curves on held-out multi-pair pairs (draft NetSci figure)
# --------------------------------------------------------------------------

def score_curves(algorithms=ALGORITHMS, models=MODELS, min_pairs: int = 5) -> pd.DataFrame:
    """Success rate by score percentile on pairs left out of the fit (multi_pair_models.ipynb, section 3)."""

    data = MP.load_sets(algorithms, models)
    rows = []
    for algorithm in algorithms:
        for model in models:
            part = data[(data["algorithm"] == algorithm) & (data["model"] == model)]
            if part["pair_index"].nunique() < min_pairs:
                continue
            scored = MP.held_out(data, algorithm, model)
            random_rate = float(scored.loc[scored["group"] == "random", "success"].mean())
            for position, (label, row) in enumerate(MP.success_by_percentile(scored).iterrows()):
                rows.append({"algorithm": algorithm, "model": model, "position": position, "bin": str(label),
                             "success": float(row["success"]), "sets": int(row["sets"]), "random": random_rate})
    return pd.DataFrame(rows)


MODEL_COLORS = {"sparse_er": "#2a78d6", "paper_pa": "#eb6834", "chung_lu": "#1baf7a"}  # validated colour order


def netsci_figure(curves: pd.DataFrame, path: Path | None = None):
    """One panel per algorithm: success rate against the score's percentile among the pair's random sets."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with plt.rc_context({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
        labels = list(MP.PERCENTILE_LABELS)
        for ax, algorithm in zip(axes, ALGORITHMS):
            # the last two bins: where best of 100 to best of 1,000 picks
            ax.axvspan(len(labels) - 2.5, len(labels) - 0.6, color="#f0f0ef", zorder=0, linewidth=0)
            if algorithm == ALGORITHMS[0]:
                ax.text(len(labels) - 1.55, 0.04, "best of\n100–1,000", ha="center", va="bottom", fontsize=6.5,
                        color="#52514e")
            for model in MODELS:
                c = curves[(curves["algorithm"] == algorithm) & (curves["model"] == model)].sort_values("position")
                if c.empty:
                    continue
                color = MODEL_COLORS[model]
                ax.axhline(c["random"].iloc[0], color=color, linewidth=1.0, linestyle=":", alpha=0.9)
                ax.plot(c["position"], c["success"], color=color, linewidth=1.8, marker="o", markersize=4,
                        markeredgecolor="white", markeredgewidth=0.6)
            ax.set_title(ALGORITHM_LABELS[algorithm])
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([label.replace("-", "–") for label in labels], rotation=45, ha="right")
            ax.set_ylim(0, 1.02)
            ax.set_xlim(-0.4, len(labels) - 0.6)
            ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
            ax.set_axisbelow(True)
        axes[0].set_ylabel("success rate")
        axes[1].set_xlabel("score percentile among the pair's random seed sets (held-out pairs)")
        handles = [Line2D([0], [0], color=MODEL_COLORS[m], linewidth=1.8, marker="o", markersize=4,
                          label=MODEL_LABELS[m]) for m in MODELS]
        handles.append(Line2D([0], [0], color="#6b6b6b", linewidth=1.0, linestyle=":", label="average random set"))
        fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02),
                   fontsize=7.5)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
            fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=ALGORITHMS)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--jobs", type=int, default=3, help="pairs run in parallel (default 3)")
    parser.add_argument("--blas-threads", type=int, default=3)
    parser.add_argument("--r100", action="store_true",
                        help=f"run GOAT at r = {R100_REGULARIZER} on PAPER and Chung-Lu (the robustness check)")
    parser.add_argument("--status", action="store_true", help="only show how far it is")
    args = parser.parse_args()
    if args.status:
        print(status(args.algorithms, args.models).to_string())
        print(r100_status().to_string())
    elif args.r100:
        run_r100(jobs=args.jobs, blas_threads=args.blas_threads)
    else:
        run_all(args.algorithms, args.models, jobs=args.jobs, blas_threads=args.blas_threads)
