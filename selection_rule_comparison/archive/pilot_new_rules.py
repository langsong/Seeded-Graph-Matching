"""Small pilot of two new seed-selection rules (21 Oct 2026).

Weighted selector (GOAT and percolation)
    Greedy forward selection that maximises the core-panel logistic score
    (seed degree, harmonic, pair mass, signature r3). The weights are fitted on
    the random sets of the training collection for the same algorithm and graph
    model, which is what core_panel_refit.ipynb reports. Seed j is the candidate
    that gives the highest score, chosen from
        "2j":  the 2j highest-degree unused vertices (as in round 1), or
        "all": every unused vertex.

Adaptive querying (GOAT only)
    Query, run, check: after each GOAT run, success is detected without the
    truth from GOAT's objective, and the loop stops at the first detection.
    All variants start from the first seeds of hub-free distinguishing:
        stop         next query = next vertex of hub-free distinguishing
        uncertainty  next query = the vertex GOAT was least sure about, among
                     the 2j highest-degree unused vertices outside the top 10
        retry        like stop, but before each new query GOAT is rerun on two
                     leave-one-out subsets of the current seeds (no new query)

Both use the round-1 main pairs, so round-1 results for the fixed rules can be
compared pair by pair. Run from the repository root (``ds`` environment):

    python -m selection_rule_comparison.pilot_new_rules           # everything
    python -m selection_rule_comparison.pilot_new_rules --only adaptive

or call ``run_all()`` from new_rules_pilot.ipynb. Results checkpoint after each
pair, so rerunning resumes.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
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
REPO_ROOT = EXPERIMENT_DIR.parent
ROUND1_DIR = EXPERIMENT_DIR / "data" / "round1" / "main"
OUTPUT_DIR = EXPERIMENT_DIR / "data" / "pilot_new_rules"
LOG_DIR = EXPERIMENT_DIR / "logs"
TRAINING_ROOT = REPO_ROOT / "fixed_pair_seed_set_classification" / "data"

# Scope (kept small: about 20-30 minutes with 3 parallel jobs).
N_PAIRS = {"goat": 10, "percolation": 20}            # round-1 main pairs 0 .. n-1
WEIGHTED_CELLS = (("goat", "chung_lu"), ("goat", "paper_pa"), ("percolation", "sparse_er"), ("percolation", "paper_pa"))
POOLS = ("2j", "all")
ADAPTIVE_MODELS = ("chung_lu", "paper_pa")
ADAPTIVE_START = {"chung_lu": 4, "paper_pa": 6}      # first budget of the round-1 grid
ADAPTIVE_VARIANTS = ("stop", "uncertainty", "retry")
BASE_RULE = "hub_free_distinguishing"
RETRIES = 2
# GOAT success is detected when objective / (|E(A)| + |E(B)|) reaches this ratio.
# In round 1 every success scored at least 0.46 and every failure at most 0.36.
DETECTION_RATIO = 0.41
RETRY_SEED_BASE = 20261023

FEATURES = ("f_degree", "f_harmonic", "f_pair_mass", "f_signature_r3")
TRAINING_COLUMNS = {
    "f_degree": "seed_nonseed_degree_mean", "f_harmonic": "seed_harmonic_mean",
    "f_pair_mass": "nonseed_witness_pair_mass_mean", "f_signature_r3": "nonseed_distance_signature_unique_fraction_r3",
}


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Weights from the training collections
# --------------------------------------------------------------------------

def training_file(algorithm: str, model: str) -> Path:
    if algorithm == "percolation":
        files = sorted((TRAINING_ROOT / "percolation_fixed_pairs_2026-10-07" / model).glob("pair_*/*_percolation_seed_sets.csv"))
    else:
        files = sorted((TRAINING_ROOT / "sgm_fixed_pairs_2026-10-13" / algorithm / model).glob("pair_*/*_seed_sets.csv"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one training file for {algorithm}/{model}, found {len(files)}.")
    return files[0]


def fit_weights(algorithm: str, model: str) -> dict:
    """Logistic regression of success on the four standardised core features (random sets only)."""

    from sklearn.linear_model import LogisticRegression

    path = training_file(algorithm, model)
    data = pd.read_csv(path)
    data = data[data["sampling_strategy"].eq("random")]
    X = data[[TRAINING_COLUMNS[f] for f in FEATURES]].to_numpy(float)
    if algorithm == "percolation":
        y = (data["final_unseeded_accuracy"] >= S.SUCCESS_THRESHOLD["percolation"]).to_numpy(int)
    else:
        y = data["sgm_success"].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"]).to_numpy(int)
    mean, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
    Z = (X - mean) / sd
    beta = LogisticRegression(C=1e6, max_iter=5000).fit(Z, y).coef_[0]
    # Degree-adjusted parametrisation, as reported in core_panel_refit (same fitted model).
    adjusted = [Z[:, 0]]
    for j in range(1, Z.shape[1]):
        residual = Z[:, j] - np.polyval(np.polyfit(Z[:, 0], Z[:, j], 1), Z[:, 0])
        adjusted.append(residual / residual.std(ddof=1))
    beta_adjusted = LogisticRegression(C=1e6, max_iter=5000).fit(np.column_stack(adjusted), y).coef_[0]
    return {
        "algorithm": algorithm, "model": model, "training_file": str(path.relative_to(REPO_ROOT)),
        "n_sets": int(len(y)), "success_rate": float(y.mean()), "n_seeds": int(data["n_seeds"].iloc[0]),
        "mean": dict(zip(FEATURES, mean.tolist())), "sd": dict(zip(FEATURES, sd.tolist())),
        "weight_per_sd": dict(zip(FEATURES, beta.tolist())),
        "degree_adjusted_per_sd": dict(zip(FEATURES, beta_adjusted.tolist())),
    }


# --------------------------------------------------------------------------
# Fast seed-set features for S + {c}, for many candidates c at once
# --------------------------------------------------------------------------

class FeatureState:
    """Same definitions as strategies.seed_set_features, updated one seed at a time."""

    def __init__(self, graph: GraphA):
        self.graph = graph
        n = graph.n
        self.A = graph.adjacency.astype(np.int64)
        self.D3 = graph.capped[3].astype(np.int64)
        self.in_set = np.zeros(n, dtype=bool)
        self.w = np.zeros(n, dtype=np.int64)          # seed neighbours of each vertex
        self.labels = np.zeros(n, dtype=np.int64)     # class of the capped distance vector to the seeds
        self.size, self.edges, self.harmonic, self.pair_mass = 0, 0, 0.0, 0

    def candidate_features(self, cands) -> dict[str, np.ndarray]:
        cands = np.asarray(cands, dtype=int)
        n, j = self.graph.n, self.size + 1
        m = n - j
        w_c = self.w[cands]
        edges = self.edges + self.graph.degree[cands] - 2 * w_c
        harmonic = self.harmonic + self.graph.harmonic[cands]
        gain = self.A[cands] @ np.where(self.in_set, 0, self.w)
        pair_mass = self.pair_mass - w_c * (w_c - 1) // 2 + gain
        keys = self.labels[:, None] * 4 + self.D3[:, cands]
        keys[self.in_set] = -1 - np.flatnonzero(self.in_set)[:, None]       # seeds never count
        keys[cands, np.arange(len(cands))] = -(n + 1) - np.arange(len(cands))  # nor does the candidate itself
        keys.sort(axis=0)
        change = keys[1:] != keys[:-1]
        edge = np.ones((1, len(cands)), dtype=bool)
        unique = (np.vstack([edge, change]) & np.vstack([change, edge]) & (keys >= 0)).sum(axis=0)
        return {"f_degree": edges / j, "f_harmonic": harmonic / j, "f_pair_mass": pair_mass / m,
                "f_signature_r3": unique / m}

    def add(self, vertex: int) -> None:
        vertex = int(vertex)
        self.in_set[vertex] = True
        self.w += self.A[vertex]
        self.size += 1
        outside = self.w[~self.in_set]
        self.edges = int(outside.sum())
        self.pair_mass = int((outside * (outside - 1) // 2).sum())
        self.harmonic += float(self.graph.harmonic[vertex])
        self.labels = np.unique(self.labels * 4 + self.D3[:, vertex], return_inverse=True)[1].astype(np.int64)


def weighted_ordering(graph: GraphA, unit_weights: dict, k_max: int, pool: str, rng) -> np.ndarray:
    """Greedy ordering: seed j maximises the score of (seeds so far + candidate)."""

    by_degree = graph.degree_order(rng)
    state, chosen = FeatureState(graph), []
    for step in range(1, k_max + 1):
        unused = by_degree[~state.in_set[by_degree]]
        cands = unused[: 2 * step] if pool == "2j" else unused
        features = state.candidate_features(cands)
        score = sum(unit_weights[f] * features[f] for f in FEATURES)
        best = int(cands[int(np.argmax(score))])  # first maximum = higher degree on exact ties
        state.add(best)
        chosen.append(best)
    return np.asarray(chosen, dtype=int)


# --------------------------------------------------------------------------
# Pairs (the round-1 main pairs, regenerated exactly) and output helpers
# --------------------------------------------------------------------------

def load_pair(algorithm: str, model: str, pair_index: int, verify: bool = True):
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
        ref = pd.read_csv(ROUND1_DIR / algorithm / model / "pairs.csv").set_index("pair_index").loc[pair_index]
        if int(ref["pair_seed"]) != int(seed) or ref["hash_a"] != rc._hash(A) or ref["hash_b"] != rc._hash(B):
            raise RuntimeError(f"{algorithm}/{model} pair {pair_index} does not match round 1.")
    return int(seed), A, B, truth


def _done_pairs(path: Path) -> set[int]:
    return set(pd.read_csv(path)["pair_index"].astype(int)) if path.exists() else set()


def _append(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# Experiment 1: weighted selector
# --------------------------------------------------------------------------

def weighted_job(algorithm: str, model: str, n_pairs: int, out_dir: Path = OUTPUT_DIR, verify: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fit = fit_weights(algorithm, model)
    (out_dir / f"weights_{algorithm}_{model}.json").write_text(json.dumps(fit, indent=2), encoding="utf-8")
    unit_weights = {f: fit["weight_per_sd"][f] / fit["sd"][f] for f in FEATURES}
    budgets = sorted(S.default_budgets("main", algorithm, model))
    path, order_path = out_dir / f"weighted_{algorithm}_{model}.csv", out_dir / f"weighted_{algorithm}_{model}_orderings.csv"
    done = _done_pairs(path)
    started = time.perf_counter()
    for pair_index in range(n_pairs):
        if pair_index in done:
            continue
        seed, A, B, truth = load_pair(algorithm, model, pair_index, verify)
        graph, pair = GraphA(A), (A, B, truth, seed)
        rows, orders, cache = [], [], {}
        for pool in POOLS:
            rng = np.random.default_rng([S.STRATEGY_SEED_BASE, seed, 100 + POOLS.index(pool)])
            order = weighted_ordering(graph, unit_weights, budgets[-1], pool, rng)
            orders.append({"pair_index": pair_index, "pair_seed": seed, "strategy": f"weighted_{pool}",
                           "ordering": json.dumps(order.tolist())})
            for budget in budgets:
                seeds = np.sort(order[:budget])
                key = tuple(int(v) for v in seeds)
                cached = key in cache
                if not cached:
                    cache[key] = rc.evaluate(algorithm, model, pair, seeds, budget, 0)
                out = cache[key]
                rows.append({"pair_index": pair_index, "pair_seed": seed, "strategy": f"weighted_{pool}",
                             "budget": budget, "seed_vertices": json.dumps(list(key)), "success": bool(out["success"]),
                             "final_unseeded_accuracy": out["final_unseeded_accuracy"], "objective": out["objective"],
                             "runtime_seconds": out["runtime_seconds"], "cached": cached,
                             **seed_set_features(graph, seeds)})
        _append(order_path, orders)
        _append(path, rows)
        print(f"{_now()} [weighted {algorithm}/{model}] pair {pair_index + 1}/{n_pairs} done; "
              f"elapsed {(time.perf_counter() - started) / 60:.1f} min", flush=True)


# --------------------------------------------------------------------------
# Experiment 2: adaptive querying with GOAT
# --------------------------------------------------------------------------

def goat_run(A, B, truth, seeds, model: str, solver_seed: int) -> dict:
    """One GOAT run (same call as experiment_common.sgm) that also returns per-vertex confidence."""

    from graspologic.match import graph_match
    import graspologic.match.wrappers as wrappers

    seeds = np.asarray(sorted(int(v) for v in seeds), dtype=int)
    captured = {}
    original = wrappers._GraphMatchSolver.compute_step_direction

    def capture(solver, gradient, rng):
        captured.setdefault("solver", solver)
        return original(solver, gradient, rng)

    wrappers._GraphMatchSolver.compute_step_direction = capture
    started = time.perf_counter()
    try:
        result = graph_match(A, B, partial_match=np.column_stack((seeds, truth[seeds])), n_init=1, n_jobs=1,
                             init=None, init_perturbation=0.0, shuffle_input=True, max_iter=S.SGM["max_iter"],
                             tol=S.SGM["tol"], rng=int(solver_seed), transport=True,
                             transport_regularizer=S.GOAT_REGULARIZER[model], transport_tol=S.SGM["transport_tol"],
                             transport_max_iter=S.SGM["transport_max_iter"])
    finally:
        wrappers._GraphMatchSolver.compute_step_direction = original
    runtime = time.perf_counter() - started
    n = A.shape[0]
    perm = np.full(n, -1, dtype=int)
    perm[result.indices_A] = result.indices_B
    unseeded = np.setdiff1d(np.arange(n), seeds)
    accuracy = float(np.mean(perm[unseeded] == truth[unseeded]))
    # Confidence 1: largest entry of each vertex's row in GOAT's final soft matching.
    solver, soft = captured.get("solver"), np.asarray(result.misc[0]["convex_solution"])
    confidence = np.full(n, np.nan)
    soft_check = 0.0
    if solver is not None and soft.shape[0] == len(unseeded):
        from scipy.optimize import linear_sum_assignment
        rows_a, cols_b = solver.perm_A[solver.n_seeds:], solver.perm_B[solver.n_seeds:]
        confidence[rows_a] = soft.max(axis=1)
        r, c = linear_sum_assignment(soft, maximize=True)
        soft_check = float(np.mean(cols_b[c] == perm[rows_a[r]]))  # ~1 if the row mapping is right
    # Confidence 2: share of the vertex's A edges kept under the final matching.
    degree = A.sum(axis=1)
    consistency = (A * B[np.ix_(perm, perm)]).sum(axis=1) / np.maximum(degree, 1)
    edges = (A.sum() + B.sum()) / 2
    ratio = float(result.score) / edges
    return {"objective": float(result.score), "ratio": ratio, "detected": ratio >= DETECTION_RATIO,
            "accuracy": accuracy, "success": accuracy >= S.SUCCESS_THRESHOLD["goat"], "runtime_seconds": runtime,
            "confidence": confidence, "consistency": consistency, "soft_check": soft_check}


def adaptive_job(model: str, n_pairs: int, out_dir: Path = OUTPUT_DIR, verify: bool = True, runner=None) -> None:
    runner = runner or goat_run
    out_dir.mkdir(parents=True, exist_ok=True)
    k_max, k0 = max(S.default_budgets("main", "goat", model)), ADAPTIVE_START[model]
    orderings = pd.read_csv(ROUND1_DIR / "goat" / model / "orderings.csv")
    base_orders = orderings[orderings["strategy"].eq(BASE_RULE)].set_index("pair_index")["ordering"]
    path, runs_path = out_dir / f"adaptive_goat_{model}.csv", out_dir / f"adaptive_goat_{model}_runs.csv"
    done = _done_pairs(path)
    started = time.perf_counter()
    for pair_index in range(n_pairs):
        if pair_index in done:
            continue
        seed, A, B, truth = load_pair("goat", model, pair_index, verify)
        graph = GraphA(A)
        base = [int(v) for v in json.loads(base_orders.loc[pair_index])]
        hub_rng = np.random.default_rng([S.STRATEGY_SEED_BASE, seed, STRATEGY_CODES[BASE_RULE]])
        hubs = set(int(v) for v in graph.degree_order(hub_rng)[:S.HUB_COUNT])  # same hubs as round 1
        by_degree = [int(v) for v in graph.degree_order(np.random.default_rng([RETRY_SEED_BASE, seed]))
                     if int(v) not in hubs]
        cache, run_rows, summary_rows = {}, [], []

        def run(seeds, restart, variant, step, kind):
            key = (tuple(sorted(seeds)), restart)
            if key not in cache:
                cache[key] = runner(A, B, truth, seeds, model, rc.solver_seed(seed, len(seeds), restart))
            out = cache[key]
            run_rows.append({"pair_index": pair_index, "pair_seed": seed, "variant": variant, "step": step,
                             "kind": kind, "n_seeds": len(seeds), "seed_vertices": json.dumps(sorted(seeds)),
                             **{k: out[k] for k in ("objective", "ratio", "detected", "accuracy", "success",
                                                    "runtime_seconds", "soft_check")}})
            return out

        for variant in ADAPTIVE_VARIANTS:
            seeds, runs_used, step, stop = list(base[:k0]), 0, 0, None
            while True:
                step += 1
                out = run(seeds, 0, variant, step, "main")
                runs_used += 1
                if out["detected"]:
                    stop = ("main", out)
                    break
                if variant == "retry":
                    rng = np.random.default_rng([RETRY_SEED_BASE, seed, len(seeds)])
                    for r, drop in enumerate(rng.choice(len(seeds), size=min(RETRIES, len(seeds)), replace=False)):
                        subset = [v for i, v in enumerate(seeds) if i != int(drop)]
                        retry_out = run(subset, 1 + r, variant, step, "retry")
                        runs_used += 1
                        if retry_out["detected"]:
                            stop = ("retry", retry_out)
                            break
                    if stop:
                        break
                if len(seeds) >= k_max:
                    break
                if variant == "uncertainty":
                    chosen = set(seeds)
                    pool = [v for v in by_degree if v not in chosen][: 2 * (len(seeds) + 1)]
                    use_soft = out["soft_check"] >= 0.95 and np.ptp(out["confidence"][pool]) > 1e-9
                    values = out["confidence"] if use_soft else out["consistency"]
                    nxt = min(pool, key=lambda v: (float(np.nan_to_num(values[v], nan=1.0)), pool.index(v)))
                    run_rows[-1]["next_query_source"] = "soft" if use_soft else "consistency"
                    seeds = seeds + [nxt]
                else:
                    seeds = list(base[: len(seeds) + 1])
            detected = stop is not None
            final = stop[1] if detected else out
            summary_rows.append({
                "pair_index": pair_index, "pair_seed": seed, "variant": variant,
                "queries": len(seeds) if detected else k_max + 1, "censored": not detected,
                "detected": detected, "stop_kind": stop[0] if detected else "", "true_success": bool(final["success"]) and detected,
                "accuracy_at_stop": final["accuracy"], "runs_used": runs_used, "seed_vertices": json.dumps(sorted(seeds)),
                "mean_seed_degree": float(graph.degree[seeds].mean()),
            })
        for row in run_rows:
            row.setdefault("next_query_source", "")
        _append(runs_path, run_rows)
        _append(path, summary_rows)
        print(f"{_now()} [adaptive goat/{model}] pair {pair_index + 1}/{n_pairs}: "
              + ", ".join(f"{r['variant']} {r['queries']}{'+' if r['censored'] else ''} queries" for r in summary_rows)
              + f"; {len(cache)} GOAT runs; elapsed {(time.perf_counter() - started) / 60:.1f} min", flush=True)


# --------------------------------------------------------------------------
# Running everything
# --------------------------------------------------------------------------

def job_list(only: str | None = None, pairs: dict | None = None) -> list[dict]:
    pairs = {**N_PAIRS, **(pairs or {})}
    jobs = []
    if only in (None, "adaptive"):
        jobs += [{"kind": "adaptive", "algorithm": "goat", "model": m, "n_pairs": pairs["goat"]} for m in ADAPTIVE_MODELS]
    if only in (None, "weighted"):
        jobs += [{"kind": "weighted", "algorithm": a, "model": m, "n_pairs": pairs[a]} for a, m in WEIGHTED_CELLS]
    return jobs


def _job(spec: dict) -> float:
    started = time.perf_counter()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"pilot_{spec['kind']}_{spec['algorithm']}_{spec['model']}.log"
    with open(log_path, "a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        print(f"{_now()} start {spec}", flush=True)
        try:
            with thread_limit(spec.get("blas_threads", 3)):
                if spec["kind"] == "weighted":
                    weighted_job(spec["algorithm"], spec["model"], spec["n_pairs"], verify=spec.get("verify", True))
                else:
                    adaptive_job(spec["model"], spec["n_pairs"], verify=spec.get("verify", True))
        except Exception:
            traceback.print_exc()
            raise
        print(f"{_now()} done", flush=True)
    return (time.perf_counter() - started) / 60


def run_all(jobs: int = 3, only: str | None = None, pairs: dict | None = None, blas_threads: int = 3,
            verify: bool = True) -> None:
    """Run the pilot; each graph model is a separate job and progress goes to logs/pilot_*.log."""

    specs = [{**spec, "blas_threads": blas_threads, "verify": verify} for spec in job_list(only, pairs)]
    keep_awake = None
    if sys.platform == "darwin":
        import shutil
        import subprocess
        if shutil.which("caffeinate"):
            keep_awake = subprocess.Popen(["caffeinate", "-i"])
    print(f"{_now()} running {len(specs)} jobs, {jobs} at a time. Progress logs:", flush=True)
    for spec in specs:
        print(f"    tail -f {LOG_DIR / ('pilot_' + spec['kind'] + '_' + spec['algorithm'] + '_' + spec['model'] + '.log')}")
    failures = []
    try:
        if jobs <= 1:
            for spec in specs:
                print(f"{_now()} {spec['kind']} {spec['algorithm']}/{spec['model']} finished in {_job(spec):.1f} min", flush=True)
        else:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {pool.submit(_job, spec): spec for spec in specs}
                for future in as_completed(futures):
                    spec = futures[future]
                    name = f"{spec['kind']} {spec['algorithm']}/{spec['model']}"
                    try:
                        print(f"{_now()} {name} finished in {future.result():.1f} min", flush=True)
                    except Exception as error:  # noqa: BLE001 - report and continue
                        failures.append(name)
                        print(f"{_now()} {name} FAILED: {error} (see its log)", flush=True)
    finally:
        if keep_awake is not None:
            keep_awake.terminate()
    if failures:
        raise RuntimeError(f"Failed: {', '.join(failures)}. Rerun to resume.")
    print(f"{_now()} all pilot jobs complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--only", choices=("weighted", "adaptive"), default=None)
    parser.add_argument("--blas-threads", type=int, default=3)
    args = parser.parse_args()
    run_all(jobs=args.jobs, only=args.only, blas_threads=args.blas_threads)
