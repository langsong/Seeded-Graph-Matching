"""Repeated A-only seed selection on native real pairs; resumable collection and analysis.

Run the companion strategy_comparison.ipynb, or:
    python -m real_data_experiments.strategy_comparison --hours 9 --jobs 3
    python -m real_data_experiments.strategy_comparison --status

The ER weights are saved synthetic weights. No real outcomes enter selection.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_common.provenance import collect_environment, thread_limit
from selection_rule_comparison.strategies import GraphA, TARGETS
from . import first_look as F

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "strategy_comparison"
ALGORITHMS = ("percolation", "faq", "goat")
METHODS = ("random", "top_degree", "degree_shortlist", "clustered", "spread",
           "distinguishing", "hub_free_distinguishing", "entropy", "best_of_n", "best_found")
RULES = ("clustered", "spread", "distinguishing", "hub_free_distinguishing", "entropy")
LABELS = {"random": "Random", "top_degree": "Exact top degree", "degree_shortlist": "Degree shortlist",
          "top_harmonic": "Harmonic shortlist", "clustered": "Clustered (randomized)",
          "spread": "Spread (randomized)", "distinguishing": "Distinguishing (randomized)",
          "hub_free_distinguishing": "Hub-free distinguishing (randomized)",
          "entropy": "Signature entropy (randomized)", "best_of_n": "Best of n: ER score",
          "best_found": "Best found: ER score"}
ALGORITHM_LABELS = {"percolation": "Percolation", "faq": "SGM (FAQ)", "goat": "GOAT (r=300)"}
COLORS = dict(zip((*METHODS, "top_harmonic"),
                  ("#888888", "#204a87", "#729fcf", "#ad494a", "#26986d", "#8856a7",
                   "#d276ae", "#969c28", "#e69f00", "#111111", "#17a2ad")))
CASES = json.loads(json.dumps(F.PAIRS))
for first in (100, 145, 160):
    for second in (first + 1, first + 2):
        CASES[f"enron_{first}_{second}"] = {
            "get": {"name": "enron", "weeks": [first, second], "vertices": "common_lcc"},
            "budgets": [5, 10, 20, 40, 60]}


def default_config():
    return json.loads(json.dumps({"version": 1, "seed_base": 2026111801, "cases": CASES, "methods": list(METHODS),
            "algorithms": list(ALGORITHMS), "test_share": F.TEST_SHARE,
            "repeats": {"small": 20, "large": 10}, "solver_runs": {"small": 3, "large": 3},
            "reference_pool": 1000, "best_of_n": 1000, "degree_pool_factor": 2,
            "shortlist_factor": 1.5, "greedy_top_fraction": 0.2, "hub_count": 10,
            "search_starts": 4, "search_steps": 12, "search_swaps": 128,
            "weights": F.RM.load_weights("sparse_er"), "solver": F.S.SGM,
            "percolation": F.S.PERCOLATION, "goat_regularizer": 300.0}))


def _rng(config, *parts):
    return np.random.default_rng([config["seed_base"], *[F._code(str(p)) for p in parts]])


def _write_csv(path, frame):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _check_time(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Collection time allowance reached; rerun to resume.")


@contextmanager
def _awake():
    process = (subprocess.Popen(["caffeinate", "-i"]) if sys.platform == "darwin"
               and shutil.which("caffeinate") else None)
    try:
        yield
    finally:
        if process is not None:
            process.terminate()
            process.wait()


def _size(case, config):
    return "large" if config["cases"][case].get("large") else "small"


def _repeats(case, config):
    return config["repeats"][_size(case, config)]


def _solver_runs(case, algorithm, config):
    return 1 if algorithm == "percolation" else config["solver_runs"][_size(case, config)]


def _folder(root, case, k):
    return Path(root) / case / f"k{k:03d}"


def prepare(root, config, blas_threads):
    config = json.loads(json.dumps(config))
    unknown = set(config["methods"]) - {*METHODS, "top_harmonic"}
    if unknown or "random" not in config["methods"]:
        raise ValueError(f"Include random and use known methods; unknown={unknown}.")
    if not set(config["algorithms"]) <= set(ALGORITHMS) or not config["algorithms"]:
        raise ValueError("Choose percolation, faq and/or goat.")
    if min(*config["repeats"].values(), *config["solver_runs"].values(),
           config["search_starts"], config["search_steps"], config["search_swaps"],
           config["best_of_n"]) < 1 or config["reference_pool"] < 2:
        raise ValueError("Counts must be positive and the reference pool needs at least two sets.")
    if not 0 < config["greedy_top_fraction"] <= 1 or config["shortlist_factor"] < 1:
        raise ValueError("Invalid randomized shortlist settings.")
    files = [Path(__file__), Path(F.__file__), ROOT / "real_data_experiments/datasets.py",
             ROOT / "selection_rule_comparison/strategies.py", Path(F.MP.__file__),
             Path(F.RM.__file__), ROOT / "experiment_common/sgm.py",
             ROOT / "experiment_common/percolation.py"]
    saved = {"config": config, "code_hashes": {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    path = Path(root) / "settings.json"
    if path.exists():
        if F._read_json(path) != saved:
            raise ValueError(f"Settings or collection code changed. Use a new output folder instead of {root}.")
    else:
        F._write_json(path, saved)
        F._write_json(Path(root) / "environment.json", collect_environment(blas_threads))
    return config


class _EntropyTarget:
    """Entropy of adjacency signatures among the remaining nonseeds, using A only."""

    def __init__(self, graph):
        self.graph = graph
        self.labels = np.zeros(graph.n, dtype=int)
        self.in_set = np.zeros(graph.n, dtype=bool)

    def score(self, vertex):
        mask = ~self.in_set
        mask[vertex] = False
        counts = np.bincount((2 * self.labels + self.graph.adjacency[:, vertex])[mask])
        p = counts[counts > 0] / counts.sum()
        return (float(-np.sum(p * np.log(p))),)

    def add(self, vertex):
        _, self.labels = np.unique(2 * self.labels + self.graph.adjacency[:, vertex], return_inverse=True)
        self.in_set[vertex] = True


def randomized_rule(method, graph, eligible, k, rng, config):
    """Draw from a rank shortlist; randomness changes choices beyond exact ties.

    The degree-first rules retain the original top-(2*j) degree candidate pool;
    entropy considers every unused eligible node.
    They choose uniformly among the best max(2, ceil(20% * pool size)) target
    choices. This is an explicit randomized variant of the original greedy rule.
    """
    eligible = np.asarray(eligible, dtype=int)
    if method == "random":
        return np.sort(rng.choice(eligible, k, replace=False))
    if method == "top_degree":
        order = eligible[np.lexsort((eligible, -graph.degree[eligible]))]
        return np.sort(order[:k])
    if method in ("degree_shortlist", "top_harmonic"):
        value = graph.degree if method == "degree_shortlist" else graph.harmonic
        order = eligible[np.lexsort((rng.random(len(eligible)), -value[eligible]))]
        pool = order[:min(len(order), math.ceil(config["shortlist_factor"] * k))]
        return np.sort(rng.choice(pool, k, replace=False))
    order = graph.degree_order(rng)
    order = order[np.isin(order, eligible)]
    if method == "hub_free_distinguishing":
        # Same definition as first_look: exclude the top ten vertices of A.
        hubs = graph.degree_order(rng)[:config["hub_count"]]
        order = order[~np.isin(order, hubs)]
    if len(order) < k:
        raise ValueError(f"{method}: only {len(order)} eligible vertices for budget {k}.")
    target = "distinguishing" if method == "hub_free_distinguishing" else method
    state = _EntropyTarget(graph) if target == "entropy" else TARGETS[target](graph)
    chosen = []
    for step in range(1, k + 1):
        remaining = order[~state.in_set[order]]
        pool = remaining if target == "entropy" else remaining[:config["degree_pool_factor"] * step]
        ranked = sorted(range(len(pool)), key=lambda i: (state.score(int(pool[i])), -i), reverse=True)
        width = min(len(pool), max(2, math.ceil(config["greedy_top_fraction"] * len(pool))))
        vertex = int(pool[rng.choice(ranked[:width])])
        chosen.append(vertex)
        state.add(vertex)
    return np.sort(chosen)


class ERScore:
    """Frozen ER weighted score with fixed-k, independent random-reference scales."""

    def __init__(self, graph, algorithm, reference, weights):
        self.graph, self.k, self.weights = graph, reference.shape[1], weights
        self.calculator = F.MP.PoolFeatures(graph, F.MP.UNIQUENESS_CAP[algorithm])
        raw = self.raw(reference)
        self.mean = {f: float(raw[f].mean()) for f in F.CORE}
        self.scale = {f: float(raw[f].std(ddof=1)) for f in F.CORE}
        for f in F.CORE:
            if self.scale[f] < 1e-9 * max(1.0, abs(self.mean[f])):
                self.scale[f] = 1.0
        n = graph.n
        self.hashes = self.calculator.hashes[np.arange(n)[None, :], self.calculator.distances]

    def raw(self, arrays):
        arrays = np.atleast_2d(arrays)
        chunk = max(1, min(64, 2_000_000 // (self.graph.n * self.k)))
        return self.calculator.compute(arrays, chunk=chunk)

    def from_raw(self, raw):
        return sum(self.weights[f] * (raw[f] - self.mean[f]) / self.scale[f] for f in F.CORE)

    def __call__(self, arrays):
        return self.from_raw(self.raw(arrays))

    def swap_scores(self, current, positions, additions):
        """Same four features as full recomputation, updating one removed/added seed."""
        gone = current[positions]
        seed_mask = np.zeros(self.graph.n, dtype=bool)
        seed_mask[current] = True
        mask = np.repeat(seed_mask[:, None], len(positions), axis=1)
        mask[gone, np.arange(len(gone))] = False
        mask[additions, np.arange(len(gone))] = True
        w = (self.graph.adjacency[:, current].sum(axis=1, dtype=np.int64)[:, None]
             - self.graph.adjacency[:, gone] + self.graph.adjacency[:, additions])
        w[mask] = 0
        fingerprint = (self.hashes[:, current].sum(axis=1)[:, None]
                       - self.hashes[:, gone] + self.hashes[:, additions])
        fingerprint[mask] = 0
        fingerprint.sort(axis=0)
        before, after = np.ones_like(mask), np.ones_like(mask)
        before[1:] = fingerprint[1:] != fingerprint[:-1]
        after[:-1] = fingerprint[:-1] != fingerprint[1:]
        raw = {"degree": w.sum(axis=0) / self.k,
               "harmonic": (self.graph.harmonic[current].sum() - self.graph.harmonic[gone]
                            + self.graph.harmonic[additions]) / self.k,
               "pair_mass": (w * (w - 1) // 2).sum(axis=0) / (self.graph.n - self.k),
               "uniqueness": (before & after & (fingerprint != 0)).sum(axis=0) / (self.graph.n - self.k)}
        return self.from_raw(raw)


def best_found(score, eligible, starts, rng, config, deadline=None):
    """Multi-start sampled improving swaps; no global or exhaustive local guarantee."""
    starts = list({tuple(sorted(map(int, s))): np.sort(s) for s in starts}.values())
    values = score(np.stack(starts))
    best = starts[int(np.argmax(values))].copy()
    best_value, evaluated, accepted = float(max(values)), len(starts), 0
    for current, value in zip(starts, values):
        current = current.copy()
        for _ in range(config["search_steps"]):
            _check_time(deadline)
            outside = np.setdiff1d(eligible, current)
            if not len(outside):
                break
            flat = rng.choice(len(current) * len(outside),
                              min(config["search_swaps"], len(current) * len(outside)), replace=False)
            positions, added = np.divmod(flat, len(outside))
            scores = score.swap_scores(current, positions, outside[added])
            evaluated += len(scores)
            j = int(np.argmax(scores))
            if scores[j] <= value + 1e-12 * max(1.0, abs(value)):
                break
            current[positions[j]] = outside[added[j]]
            current.sort()
            value = float(scores[j])
            accepted += 1
            if value > best_value:
                best, best_value = current.copy(), value
    return best, {"score_evaluations": evaluated, "accepted_swaps": accepted,
                  "initial_score": float(values[0]), "final_score": float(score(best)[0])}


def selection_pool(generator, eligible, k, requested):
    """Draw distinct sets, or enumerate the finite population if it is smaller."""
    if math.comb(len(eligible), k) < requested:
        return np.asarray(list(itertools.combinations(sorted(eligible), k)), dtype=np.int64)
    return F.G.draw_sets(generator, eligible, k, requested)


def make_plan(case, k, graph, eligible, config, deadline=None):
    """Select every membership before any algorithm outcomes are collected."""
    reference = selection_pool(_rng(config, case, k, "reference"), eligible, k, config["reference_pool"])
    scorers = {a: ERScore(graph, a, reference, config["weights"][a]) for a in config["algorithms"]}
    arrays, index, members = [], {}, []

    def add(method, draw, seeds, algorithm=None, **extra):
        key = tuple(sorted(map(int, seeds)))
        if len(key) != k or len(set(key)) != k or not set(key) <= set(eligible):
            raise ValueError("Seed set is not a valid fixed-budget subset of eligible A vertices.")
        if key not in index:
            index[key] = len(arrays)
            arrays.append(np.asarray(key, dtype=int))
        for a in ([algorithm] if algorithm else config["algorithms"]):
            members.append({"algorithm": a, "method": method, "draw": draw,
                            "candidate_id": index[key], **extra})

    exact = randomized_rule("top_degree", graph, eligible, k, _rng(config, case, k, "exact"), config)
    if "top_degree" in config["methods"]:
        add("top_degree", 0, exact)
    for draw in range(_repeats(case, config)):
        _check_time(deadline)
        built = []
        for method in config["methods"]:
            if method in ("top_degree", "best_of_n", "best_found"):
                continue
            seeds = randomized_rule(method, graph, eligible, k, _rng(config, case, k, method, draw), config)
            add(method, draw, seeds)
            if method != "random":
                built.append(seeds)
        if {"best_of_n", "best_found"} & set(config["methods"]):
            pool = selection_pool(_rng(config, case, k, "screen", draw), eligible, k, config["best_of_n"])
            for a, score in scorers.items():
                _check_time(deadline)
                values = score(pool)
                screen = pool[int(np.argmax(values))]
                if "best_of_n" in config["methods"]:
                    add("best_of_n", draw, screen, a)
                if "best_found" in config["methods"]:
                    rng = _rng(config, case, k, a, "search", draw)
                    # Always include the paired screen. Rotate additional rule starts across repetitions.
                    others = [exact, *[built[j] for j in rng.permutation(len(built))]]
                    while len(others) + 1 < config["search_starts"]:
                        others.append(np.sort(rng.choice(eligible, k, replace=False)))
                    starts = [screen, *others[:config["search_starts"] - 1]]
                    began = time.monotonic()
                    seeds, search = best_found(score, eligible, starts, rng, config, deadline)
                    add("best_found", draw, seeds, a, **search, search_seconds=time.monotonic() - began)
    sets = pd.DataFrame({"candidate_id": np.arange(len(arrays)),
                         "seed_vertices": [json.dumps(s.tolist()) for s in arrays]})
    for a, score in scorers.items():
        raw = score.raw(np.stack(arrays))
        for f in F.CORE:
            sets[f"{a}__{f}"] = raw[f]
        sets[f"{a}__score_er"] = score.from_raw(raw)
    normalization = {a: {"mean": s.mean, "scale": s.scale} for a, s in scorers.items()}
    return sets, pd.DataFrame(members), normalization


def _run_rows(path):
    if not path.exists() or not path.stat().st_size:
        return pd.DataFrame()
    F.G._completed_ids(path, "run_id")  # repair an interrupted CSV row before appending
    frame = pd.read_csv(path)
    if frame.run_id.duplicated().any():
        raise ValueError(f"Duplicate solver run IDs in {path}.")
    if not np.isfinite(frame.accuracy_test).all() or not frame.accuracy_test.between(0, 1).all():
        raise ValueError(f"Invalid saved outcomes in {path}.")
    return frame


def collect_bank(root, case, k, config, blas_threads=2, deadline=None):
    began = time.monotonic()
    folder = _folder(root, case, k)
    with thread_limit(blas_threads):
        _, A, B, truth, test, eligible, info = F.load_pair(
            case, {"pairs": config["cases"], "test_share": config["test_share"]})
        if k > len(eligible):
            raise ValueError(f"{case}: k={k} exceeds {len(eligible)} eligible seed vertices.")
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "plan.json").exists():
            saved = F._read_json(folder / "plan.json")
            if any(saved[key] != info[key] for key in ("hash_a", "hash_b", "test_vertices")):
                raise ValueError(f"Source graphs or test set changed: {folder}")
            sets, members = pd.read_csv(folder / "sets.csv"), pd.read_csv(folder / "memberships.csv")
        else:
            print(f"{case} k={k}: selecting seed sets", flush=True)
            sets, members, normalization = make_plan(case, k, GraphA(A), eligible, config, deadline)
            _write_csv(folder / "sets.csv", sets)
            _write_csv(folder / "memberships.csv", members)
            # Last file commits the plan; an interrupted plan is deterministically rebuilt.
            F._write_json(folder / "plan.json", {**info, "budget": k, "eligible": len(eligible),
                          "normalization": normalization, "written_at": F._now()})
            print(f"{case} k={k}: {len(sets)} distinct sets selected", flush=True)
        lookup = {int(row.candidate_id): np.asarray(json.loads(row.seed_vertices))
                  for row in sets.itertuples()}
        written, failures, last_report = 0, 0, time.monotonic()
        for a in config["algorithms"]:
            path = folder / f"{a}_runs.csv"
            previous = _run_rows(path)
            done = set(previous.run_id) if len(previous) else set()
            count = _solver_runs(case, a, config)
            ids = sorted(members.loc[members.algorithm == a, "candidate_id"].unique())
            expected = {int(i) * count + repeat for i in ids for repeat in range(count)}
            if not done <= expected:
                raise ValueError(f"Unexpected solver IDs in {path}.")
            if done == expected:
                continue
            rows = F.G._Rows(path)
            errors = F.G._Rows(folder / f"{a}_errors.csv")
            try:
                for repeat in range(count):
                    # Every distinct candidate receives the same RNG block for this algorithm/repetition.
                    seed = (config["percolation"]["algorithm_seed"] if a == "percolation" else
                            F.rc._seed(config["seed_base"], F._code(case), k, F._code(a), repeat))
                    for candidate_id in ids:
                        run_id = int(candidate_id) * count + repeat
                        if run_id in done:
                            continue
                        _check_time(deadline)
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            out = F.RM.run_one(a, "sparse_er", (A, B, truth), lookup[candidate_id], test,
                                               seed, config["goat_regularizer"])
                        record = {"run_id": run_id, "candidate_id": candidate_id, "repeat": repeat,
                                  "solver_seed": seed, **out, "warning_count": len(caught),
                                  "warnings": json.dumps([str(w.message) for w in caught])}
                        if out["error"] or not np.isfinite(out["accuracy_test"]):
                            errors.write({**record, "attempted_at": F._now()})
                            failures += 1
                            continue
                        rows.write(record)
                        done.add(run_id)
                        written += 1
                        if time.monotonic() - last_report >= 30:
                            print(f"{case} k={k} {a}: {len(done)}/{len(expected)} solver runs", flush=True)
                            last_report = time.monotonic()
            finally:
                rows.close()
                errors.close()
            if (folder / f"{a}_errors.csv").stat().st_size == 0:
                (folder / f"{a}_errors.csv").unlink()
    return {"message": f"{case} k={k}: {written} new runs, {failures} failures, "
                       f"{(time.monotonic() - began) / 60:.1f} min", "failures": failures}


def run(root=OUTPUT_DIR, config=None, jobs=3, blas_threads=2, hours=None, cases=None, budgets=None):
    """Notebook-safe process workers; rerunning resumes all completed plans/runs."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    deadline = None if hours is None else time.monotonic() + 3600 * hours
    with _awake(), (root / ".collection.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another strategy-comparison collector is already using {root}.") from None
        config = prepare(root, default_config() if config is None else config, blas_threads)
        todo = [(case, k) for case in (cases or config["cases"])
                for k in (budgets or config["cases"][case]["budgets"])]
        for case, k in todo:
            if k not in config["cases"][case]["budgets"]:
                raise ValueError(f"{case}: budget {k} is outside the saved grid. "
                                 "Configure a different grid in a new output folder.")
        print(f"{len(todo)} graph/budget banks; {len(config['algorithms'])} algorithms. "
              "Completed runs are reused. Ctrl-C pauses collection.", flush=True)
        failures = 0
        try:
            # Keep at most two Wikipedia workers in memory, independently of small-pair concurrency.
            for large in (False, True):
                tasks = [(case, k) for case, k in todo if (_size(case, config) == "large") == large]
                if not tasks:
                    continue
                workers = min(max(1, jobs), 2 if large else max(1, jobs), len(tasks))
                if workers == 1:
                    for case, k in tasks:
                        _check_time(deadline)
                        out = collect_bank(root, case, k, config, blas_threads, deadline)
                        print(out["message"], flush=True)
                        failures += out["failures"]
                else:
                    pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
                    try:
                        futures = [pool.submit(collect_bank, root, case, k, config, blas_threads, deadline)
                                   for case, k in tasks]
                        for future in as_completed(futures):
                            out = future.result()
                            print(out["message"], flush=True)
                            failures += out["failures"]
                        pool.shutdown(wait=True)
                    except BaseException:
                        processes = list((getattr(pool, "_processes", None) or {}).values())
                        pool.shutdown(wait=False, cancel_futures=True)
                        for process in processes:
                            process.terminate()
                        raise
        except (KeyboardInterrupt, TimeoutError) as error:
            print(f"Paused ({type(error).__name__}). Saved runs are kept; rerun to resume.", flush=True)
            return False
        if failures:
            raise RuntimeError(f"{failures} evaluations failed; inspect *_errors.csv and rerun to retry.")
    return True


def read(root=OUTPUT_DIR):
    """Return plans, memberships, completed solver blocks and their completion inventory."""
    root = Path(root)
    config = F._read_json(root / "settings.json")["config"]
    sets, members, draws, inventory = [], [], [], []
    for case, spec in config["cases"].items():
        for k in spec["budgets"]:
            folder = _folder(root, case, k)
            planned = (folder / "plan.json").exists()
            if planned:
                plan = F._read_json(folder / "plan.json")
                s = pd.read_csv(folder / "sets.csv").assign(case=case, budget=k, eligible=plan["eligible"])
                m = pd.read_csv(folder / "memberships.csv").assign(case=case, budget=k)
                if m.duplicated(["algorithm", "method", "draw"]).any():
                    raise ValueError(f"Duplicate policy memberships in {folder}.")
                sets.append(s)
                members.append(m)
            for a in config["algorithms"]:
                expected_draws = {method: (1 if method == "top_degree" else _repeats(case, config))
                                  for method in config["methods"]}
                count = _solver_runs(case, a, config)
                r = _run_rows(folder / f"{a}_runs.csv")
                expected_ids = set(m.loc[m.algorithm == a, "candidate_id"]) if planned else set()
                expected_runs = {(int(i), repeat) for i in expected_ids for repeat in range(count)}
                actual = set(zip(r.candidate_id, r.repeat)) if len(r) else set()
                if not actual <= expected_runs:
                    raise ValueError(f"Unexpected saved run memberships in {folder}/{a}_runs.csv.")
                counts = m[m.algorithm == a].groupby("method").size().to_dict() if planned else {}
                complete = planned and actual == expected_runs and counts == expected_draws
                inventory.append({"case": case, "budget": k, "algorithm": a, "planned": planned,
                                  "expected_runs": len(expected_runs) if planned else np.nan,
                                  "saved_runs": len(r), "complete": complete})
                if complete:
                    avg = r.groupby("candidate_id").agg(accuracy=("accuracy_test", "mean"),
                            solver_sd=("accuracy_test", "std"), solver_runs=("repeat", "size"),
                            convergence_share=("converged", "mean")).reset_index()
                    # Keep repeated draws even when they select the same set: their frequency defines the policy.
                    draw = m[m.algorithm == a].merge(avg, on="candidate_id", validate="many_to_one")
                    draw = draw.merge(s[["candidate_id", f"{a}__score_er"]].rename(
                        columns={f"{a}__score_er": "score_er"}), on="candidate_id", validate="many_to_one")
                    draws.append(draw)
    concat = lambda frames: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return {"config": config, "sets": concat(sets), "members": concat(members),
            "draws": concat(draws), "inventory": pd.DataFrame(inventory)}


def summarize(data):
    draw = data["draws"]
    if draw.empty:
        return pd.DataFrame()
    keys = ["case", "budget", "algorithm", "method"]
    table = draw.groupby(keys, sort=False).agg(mean=("accuracy", "mean"),
        q10=("accuracy", lambda v: v.quantile(.1)), q90=("accuracy", lambda v: v.quantile(.9)),
        selection_sd=("accuracy", "std"), mean_solver_sd=("solver_sd", "mean"),
        draws=("draw", "size"), unique_sets=("candidate_id", "nunique"),
        score_mean=("score_er", "mean")).reset_index()
    baseline = table[table.method == "random"][["case", "budget", "algorithm", "mean"]].rename(
        columns={"mean": "random_mean"})
    table = table.merge(baseline, on=["case", "budget", "algorithm"], validate="many_to_one")
    table["gain"] = table["mean"] - table.random_mean
    return table


def overlaps(data):
    """Mean shared fraction, including duplicate selections; diagonal excludes self-draw pairs."""
    output = []
    if data["members"].empty:
        return pd.DataFrame()
    for (case, k, a), members in data["members"].groupby(["case", "budget", "algorithm"], sort=False):
        s = data["sets"].query("case == @case and budget == @k").set_index("candidate_id")
        arrays = {int(i): np.asarray(json.loads(row.seed_vertices)) for i, row in s.iterrows()}
        n = max(max(seeds) for seeds in arrays.values()) + 1
        policies = list(members.method.unique())
        vectors = {}
        for method in policies:
            ids = members.loc[members.method == method, "candidate_id"].to_numpy()
            v = np.zeros((len(ids), n), dtype=np.uint16)
            for j, candidate_id in enumerate(ids):
                v[j, arrays[candidate_id]] = 1
            vectors[method] = v
        for left, right in itertools.product(policies, repeat=2):
            matrix = vectors[left] @ vectors[right].T / k
            if left == right and len(matrix) > 1:
                value = matrix[~np.eye(len(matrix), dtype=bool)].mean()
            else:
                value = matrix.mean()
            output.append({"case": case, "budget": k, "algorithm": a, "left": left, "right": right,
                           "shared_fraction": value, "random_expected": k / int(s.eligible.iloc[0])})
    return pd.DataFrame(output)


def robustness_table(summary, overlap):
    if summary.empty:
        return pd.DataFrame()
    diag = overlap[overlap.left == overlap.right].rename(columns={"left": "method"})
    table = summary.merge(diag, on=["case", "budget", "algorithm", "method"], validate="one_to_one")
    table["distinct_share"] = table.unique_sets / table.draws
    table["excess_overlap"] = table.shared_fraction - table.random_expected
    return table.groupby("method", sort=False).agg(
        median_distinct_share=("distinct_share", "median"), median_shared_fraction=("shared_fraction", "median"),
        median_overlap_above_random=("excess_overlap", "median"),
        median_selection_sd=("selection_sd", "median"), median_solver_sd=("mean_solver_sd", "median"))


def _save(fig, root, name):
    folder = Path(root) / "analysis" / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{extension}", dpi=160, bbox_inches="tight")


def plot_curves(summary, config, algorithm, root=OUTPUT_DIR, methods=None, bands=True, fit_y=True):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    methods = config["methods"] if methods is None else methods
    cases = [case for case in config["cases"] if case in set(summary.case)]
    columns = min(3, len(cases))
    rows = math.ceil(len(cases) / columns)
    height = 3.0 * rows + 1.4
    fig, axes = plt.subplots(rows, columns, figsize=(max(12, 5 * columns), height),
                             squeeze=False, sharey=not fit_y)
    for ax, case in zip(axes.flat, cases):
        part = summary[(summary.case == case) & (summary.algorithm == algorithm)]
        for method in methods:
            curve = part[part.method == method].sort_values("budget")
            if curve.empty:
                continue
            ax.plot(curve.budget, curve["mean"], color=COLORS[method], label=LABELS[method],
                    lw=2.2 if method in ("random", "best_found") else 1.4,
                    ls="--" if method in ("top_degree", "best_of_n") else "-", marker=".", ms=4)
            if bands and method != "top_degree":
                ax.fill_between(curve.budget, curve.q10, curve.q90, color=COLORS[method], alpha=.055)
        limits = (0, 1)
        shown = part[part.method.isin(methods)]
        if fit_y and not shown.empty:
            values = shown[["mean", "q10", "q90"] if bands else ["mean"]].to_numpy()
            low, high = float(np.nanmin(values)), float(np.nanmax(values))
            # Include all displayed bands, add padding, and avoid magnifying tiny differences.
            span = min(1.0, max(.10, high - low) * 1.15)
            centre = np.clip((low + high) / 2, span / 2, 1 - span / 2)
            limits = (centre - span / 2, centre + span / 2)
        ax.set(title=case.replace("_", " "), xlabel="Seeds (k)", ylim=limits)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Test accuracy")
    for ax in list(axes.flat)[len(cases):]:
        ax.set_visible(False)
    handles, labels = {}, {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles[label], labels[label] = handle, label
    fig.legend(list(handles.values()), list(labels), loc="lower center", ncol=3, fontsize=8.5,
               bbox_to_anchor=(.5, 0), frameon=False)
    title = f"{ALGORITHM_LABELS[algorithm]}: repeated seed-selection strategies"
    fig.suptitle(title + ("\nPanel-specific y-axis ranges" if fit_y else ""), y=.995)
    fig.tight_layout(rect=(0, 1.25 / height, 1, .95))
    _save(fig, root, f"{algorithm}_accuracy_curves")
    return fig


def plot_gains(summary, config, root=OUTPUT_DIR, methods=None):
    import matplotlib.pyplot as plt
    methods = [m for m in (config["methods"] if methods is None else methods) if m != "random"]
    cases = [c for c in config["cases"] if c in set(summary.case)]
    algorithms = config["algorithms"]
    values = summary.groupby(["algorithm", "method", "case"]).gain.mean()
    limit = max(.05, float(values.abs().max()))
    fig, axes = plt.subplots(1, len(algorithms), figsize=(6.0 * len(algorithms), 5.8), squeeze=False)
    for ax, algorithm in zip(axes.flat, algorithms):
        matrix = (values.xs(algorithm).unstack("case").reindex(index=methods, columns=cases)
                  if algorithm in values.index.get_level_values("algorithm") else
                  pd.DataFrame(np.nan, index=methods, columns=cases))
        im = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(cases)), [c.replace("enron_", "E ").replace("_", "–") for c in cases],
                      rotation=65, ha="right", fontsize=8)
        ax.set_yticks(range(len(methods)), [LABELS[m] for m in methods], fontsize=8)
        ax.set_title(ALGORITHM_LABELS[algorithm])
    fig.suptitle("Accuracy gain over random, averaged equally over completed budgets")
    fig.tight_layout(rect=(0, 0, .96, .94))
    fig.colorbar(im, ax=list(axes.flat), fraction=.02, pad=.02, label="Accuracy difference")
    _save(fig, root, "gains_over_random")
    return fig


def plot_overlap(overlap, config, case="wikipedia", budget=20, algorithm="faq", root=OUTPUT_DIR):
    import matplotlib.pyplot as plt
    part = overlap[(overlap.case == case) & (overlap.budget == budget) & (overlap.algorithm == algorithm)]
    if part.empty:
        row = overlap.iloc[0]
        case, budget, algorithm = row.case, row.budget, row.algorithm
        part = overlap[(overlap.case == case) & (overlap.budget == budget) & (overlap.algorithm == algorithm)]
    methods = config["methods"]
    matrix = part.pivot(index="left", columns="right", values="shared_fraction").reindex(
        index=methods, columns=methods)
    fig, ax = plt.subplots(figsize=(8.7, 7.1))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    labels = [LABELS[m].replace(" (randomized)", "") for m in methods]
    ax.set_xticks(range(len(methods)), labels, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(range(len(methods)), labels, fontsize=8)
    for (i, j), value in np.ndenumerate(matrix.to_numpy()):
        if np.isfinite(value):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if value > .6 else "#222222")
    ax.set_title(f"{case.replace('_', ' ')} · k={budget} · {ALGORITHM_LABELS[algorithm]}\n"
                 f"Mean shared seed fraction; independent random expectation {part.random_expected.iloc[0]:.2f}")
    fig.colorbar(im, ax=ax, fraction=.04, label="Shared seeds / k")
    fig.tight_layout()
    _save(fig, root, "seed_overlap")
    return fig


def export(data, summary, overlap, root=OUTPUT_DIR):
    folder = Path(root) / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    frames = {"inventory": data["inventory"], "policy_draws": data["draws"], "strategy_summary": summary,
              "overlaps": overlap, "robustness": robustness_table(summary, overlap).reset_index()}
    for name, frame in frames.items():
        _write_csv(folder / f"{name}.csv", frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--hours", type=float)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--blas-threads", type=int, default=2)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES))
    parser.add_argument("--budgets", type=int, nargs="+")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        d = read(args.output_dir)
        print(d["inventory"].to_string(index=False))
    else:
        config = (F._read_json(args.output_dir / "settings.json")["config"]
                  if (args.output_dir / "settings.json").exists() else default_config())
        run(args.output_dir, config, args.jobs, args.blas_threads, args.hours, args.cases, args.budgets)


if __name__ == "__main__":
    main()
