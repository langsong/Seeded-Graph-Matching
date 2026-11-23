"""Time-limited, resumable real-network confirmation and active-query experiments.

python -m real_data_experiments.strategy_and_active_v2 --hours 6.5 --jobs 3
python -m real_data_experiments.strategy_and_active_v2 --status
python -m real_data_experiments.strategy_and_active_v2 --smoke --hours .1

Collection only (renamed from overnight.py on 23 Nov 2026). notebooks/03_strategy_comparison_v2.ipynb and
notebooks/04_active_queries_v2.ipynb read these results with strategy_and_active_v2_analysis.
Every completed solver run / ensemble is committed atomically; old result folders are untouched.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback
import warnings

import numpy as np

from experiment_common.provenance import collect_environment, thread_limit
from selection_rule_comparison.strategies import GraphA, SignatureTarget
from . import active_test as T
from . import strategy_comparison as C

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "overnight_v2"
POLICIES = ("random", "structural", "uncertainty", "uncertainty_impact")
CORE_METHODS = ("random", "top_degree", "clustered", "spread", "distinguishing", "best_of_n", "best_found")
ACTIVE_CASES = ("enron_130_132", "enron_145_147", "enron_160_162")


def default_config(smoke=False):
    selection = C.default_config()
    selection.update(methods=list(CORE_METHODS), repeats={"small": 8, "large": 3},
                     solver_runs={"small": 2, "large": 2})
    config = {
        "version": 2, "seed_base": 2026111807, "cases": C.CASES,
        "selection": selection, "test_share": .2, "splits": {"small": 5, "large": 3},
        "pilot_draws": {"small": 8, "large": 4},
        "active_cases": list(ACTIVE_CASES), "active_algorithms": ["faq", "goat"],
        "active_bases": ["top_degree", "distinguishing"], "active_repeats": 10,
        "active_initial": 5, "active_final": 30, "ensemble_runs": {"small": 10, "large": 8},
        "init_perturbation": .2, "guard": True,
        "diagnostic_budgets": {"small": [5, 10, 20], "large": [10, 20]},
        "diagnostic_repeats": {"small": 5, "large": 2},
        "diagnostic_candidates": {"small": 0, "large": 16}, "diagnostic_runs": 3,
        "solver": T.default_config()["solver"], "goat_regularizer": 300.,
        "smoke": smoke,
    }
    if smoke:
        config.update(cases={"enron_160_162": C.CASES["enron_160_162"]},
                      active_cases=["enron_160_162"], active_repeats=1,
                      active_bases=["distinguishing"], active_initial=3, active_final=5,
                      splits={"small": 1, "large": 1}, pilot_draws={"small": 2, "large": 2},
                      ensemble_runs={"small": 3, "large": 3}, diagnostic_runs=2,
                      diagnostic_budgets={"small": [3], "large": [3]},
                      diagnostic_repeats={"small": 1, "large": 1},
                      diagnostic_candidates={"small": 4, "large": 4})
        selection.update(repeats={"small": 2, "large": 2}, reference_pool=30,
                         best_of_n=30, search_steps=2, search_swaps=12, search_starts=2)
    return json.loads(json.dumps(config))


def seed(config, *parts):
    # The pilot uses partition -1; SeedSequence only accepts nonnegative integers.
    return T._seed(config["seed_base"], *[str(p) if isinstance(p, (int, np.integer)) and p < 0 else p
                                         for p in parts])


def rng(config, *parts):
    return np.random.default_rng(seed(config, *parts))


def size(case, config):
    return "large" if config["cases"][case].get("large") else "small"


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Collection time allowance reached")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    C.F._write_json(path, value)


def save(path, metadata, **arrays):
    """One atomically committed record, including the raw assignment and correctness vector."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, metadata=np.asarray(json.dumps(metadata)), **arrays)
    temporary.replace(path)


def read(path):
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["metadata"])), {k: data[k] for k in data.files if k != "metadata"}


@lru_cache(maxsize=4)
def pair_arrays(case, spec_json):
    pair = C.F.D.get_pair(**json.loads(spec_json)["get"])
    return (np.asarray(pair.A) > 0).astype(float), (np.asarray(pair.B) > 0).astype(float)


@lru_cache(maxsize=3)
def graph_a(case, spec_json):
    return GraphA(pair_arrays(case, spec_json)[0])


def context(case, split, config):
    spec = json.dumps(config["cases"][case], sort_keys=True)
    A, aligned_B = pair_arrays(case, spec)
    n = len(A)
    shuffle = rng(config, case, split, "shuffle").permutation(n)
    test = np.sort(rng(config, case, split, "test").choice(n, round(config["test_share"] * n), replace=False))
    return {"A": A, "B": aligned_B[np.ix_(shuffle, shuffle)], "truth": np.argsort(shuffle),
            "test": test, "eligible": np.setdiff1d(np.arange(n), test), "n": n,
            "case": case, "split": split, "config": config, "size": size(case, config)}


def graph(ctx):
    return graph_a(ctx["case"], json.dumps(ctx["config"]["cases"][ctx["case"]], sort_keys=True))


def metrics(ctx, perm, seeds):
    correct = np.asarray(perm) == ctx["truth"]
    others = np.setdiff1d(np.arange(ctx["n"]), seeds)
    return {"accuracy_test": float(correct[ctx["test"]].mean()),
            "accuracy_nonseed": float(correct[others].mean()), "accuracy_all": float(correct.mean()),
            "total_correct": int(correct.sum())}


def match_once(ctx, seeds, algorithm, solver_seed, perturbation=0.):
    seeds = np.sort(np.asarray(seeds, dtype=int))
    if not set(seeds) <= set(ctx["eligible"]):
        raise ValueError("A held-out test vertex was selected as a seed")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if algorithm == "percolation":
            from experiment_common.percolation import graph_match_percolation
            before = np.random.get_state()
            start = time.monotonic()
            np.random.seed(solver_seed)
            try:
                p = ctx["config"]["selection"]["percolation"]
                perm = graph_match_percolation(ctx["A"], ctx["B"],
                    np.column_stack((seeds, ctx["truth"][seeds])), r=p["threshold"],
                    ExpandWhenStuck=p["expand_when_stuck"])
            finally:
                np.random.set_state(before)
            out = {"perm": perm, "confidence": np.full(ctx["n"], np.nan), "soft_check": np.nan,
                   "runtime_seconds": time.monotonic() - start, "converged": True}
        else:
            config = {**ctx["config"], "init_perturbation": perturbation}
            out = T.match(ctx["A"], ctx["B"], ctx["truth"], seeds, algorithm, solver_seed, config)
    if not np.array_equal(out["perm"][seeds], ctx["truth"][seeds]):
        raise RuntimeError("Matcher violated a verified seed")
    out["warnings"] = [str(w.message) for w in caught]
    return out


def evaluation(path, ctx, seeds, algorithm, solver_seed, deadline):
    if Path(path).exists():
        return read(path)[0]
    check_time(deadline)
    out = match_once(ctx, seeds, algorithm, solver_seed)
    meta = {**metrics(ctx, out["perm"], seeds), "runtime_seconds": out["runtime_seconds"],
            "solver_seed": int(solver_seed), "converged": out["converged"], "warnings": out["warnings"]}
    save(path, meta, perm=out["perm"], correct=out["perm"] == ctx["truth"], seeds=seeds,
         test=ctx["test"], truth=ctx["truth"])
    return meta


def pilot_grid(case, algorithm, config):
    if config["smoke"]:
        return [3, 6]
    if case.startswith("enron"):
        return [1, 3, 5, 10, 15, 20, 30, 40, 60]
    if case == "wikipedia":
        return {"faq": [5, 10, 15, 20, 30, 50], "goat": [5, 10, 20, 40, 60],
                "percolation": [30, 50, 75, 100, 125, 150]}[algorithm]
    return config["cases"][case]["budgets"]


def pilot_job(root, config, task, deadline):
    _, case, algorithm = task
    ctx = context(case, -1, config)  # independent of confirmation partitions and random pools
    rows = []
    for k in pilot_grid(case, algorithm, config):
        if k >= len(ctx["eligible"]):
            continue
        for draw in range(config["pilot_draws"][ctx["size"]]):
            seeds = np.sort(rng(config, case, "pilot", k, draw).choice(ctx["eligible"], k, replace=False))
            path = root / "calibration" / case / algorithm / f"k{k:03d}_d{draw:02d}.npz"
            out = evaluation(path, ctx, seeds, algorithm,
                             seed(config, case, "pilot_solver", algorithm, k, draw), deadline)
            rows.append({"budget": k, "draw": draw, **out})
    write_json(root / "calibration" / case / f"{algorithm}.json", rows)


def choose_budgets(rows, count):
    """Predeclared selection using only the independent random pilot; no method outcomes."""
    budgets = sorted({r["budget"] for r in rows})
    means = np.asarray([np.mean([r["accuracy_test"] for r in rows if r["budget"] == k]) for k in budgets])
    count = min(count, len(budgets))
    if np.ptp(means) < .1:
        selected = [budgets[i] for i in np.linspace(0, len(budgets) - 1, count).round().astype(int)]
    else:
        selected = []
        for target in np.linspace(.2, .7, count):
            candidates = [i for i, k in enumerate(budgets) if k not in selected]
            i = min(candidates, key=lambda i: (abs(means[i] - target), budgets[i]))
            selected.append(budgets[i])
    return {"budgets": sorted(selected), "pilot_means": dict(zip(map(str, budgets), means.tolist())),
            "observed_range": [float(min(means)), float(max(means))],
            "has_intermediate": bool(np.any((means >= .2) & (means <= .7)))}


def calibration_plan(root, config):
    selected = {}
    for case in config["cases"]:
        selected[case] = {}
        for algorithm in C.ALGORITHMS:
            path = root / "calibration" / case / f"{algorithm}.json"
            if not path.exists():
                return None
            selected[case][algorithm] = choose_budgets(json.loads(path.read_text()),
                                                       4 if size(case, config) == "large" else 3)
    write_json(root / "calibration" / "budgets.json", selected)
    return selected


def strategy_folder(root, case, split, algorithm, k):
    return Path(root) / "strategy" / case / f"split{split:02d}" / algorithm / f"k{k:03d}"


def strategy_job(root, config, task, deadline):
    _, case, split, algorithm, k, draw = task
    ctx = context(case, split, config)
    folder = strategy_folder(root, case, split, algorithm, k)
    plan_path = folder / "plan.json"
    if not plan_path.exists():
        selection = json.loads(json.dumps(config["selection"]))
        selection.update(seed_base=seed(config, case, split, "selection"), algorithms=[algorithm])
        if ctx["size"] == "small":
            selection["methods"] += ["hub_free_distinguishing", "entropy"]
        sets, members, normalization = C.make_plan(case, k, graph(ctx), ctx["eligible"], selection, deadline)
        plan = {"case": case, "split": split, "algorithm": algorithm, "budget": k, "n": ctx["n"],
                "test": ctx["test"].tolist(), "eligible": ctx["eligible"].tolist(),
                "hash_a": C.F.rc._hash(ctx["A"]), "hash_b": C.F.rc._hash(ctx["B"]),
                "methods": selection["methods"], "sets": sets.to_dict("records"),
                "members": members.to_dict("records"), "normalization": normalization}
        write_json(plan_path, plan)
    plan = json.loads(plan_path.read_text())
    lookup = {r["candidate_id"]: np.asarray(json.loads(r["seed_vertices"]), dtype=int) for r in plan["sets"]}
    members = [r for r in plan["members"] if r["draw"] == draw or r["method"] == "top_degree"]
    repeats = 1 if algorithm == "percolation" else config["selection"]["solver_runs"][ctx["size"]]
    for repeat in range(repeats):
        # All methods get the same random-number block and every method is visited before repetition two.
        for member in members:
            cid = member["candidate_id"]
            evaluation(folder / f"c{cid:04d}_r{repeat}.npz", ctx, lookup[cid], algorithm,
                       seed(config, case, split, algorithm, k, "strategy_solver", repeat), deadline)


def ensemble(path, ctx, seeds, algorithm, parts, runs, deadline, incumbent=None):
    """Fresh, varied feasible assignments; cache each constituent run before the aggregate."""
    path = Path(path)
    if path.exists():
        meta, arrays = read(path)
        return {**meta, **arrays}
    seeds = np.sort(seeds)
    outs = []
    for r in range(runs):
        part = path.parent / (path.stem + "_runs") / f"r{r:02d}.npz"
        if part.exists():
            meta, arrays = read(part)
            out = {**meta, **arrays}
        else:
            check_time(deadline)
            out = match_once(ctx, seeds, algorithm, seed(ctx["config"], *parts, r),
                             0. if r == 0 else ctx["config"]["init_perturbation"])
            save(part, {k: out[k] for k in ("runtime_seconds", "converged", "soft_check", "warnings")},
                 perm=out["perm"], confidence=out["confidence"])
        outs.append(out)
    permutations = np.stack([o["perm"] for o in outs])
    objectives = np.asarray([T.kept_edges(ctx["A"], ctx["B"], p) for p in permutations])
    perm = permutations[int(np.argmax(objectives))].copy()
    kept_previous = False
    if incumbent is not None and ctx["config"]["guard"]:
        fixed = T.correct(incumbent, seeds, ctx["truth"])
        if T.kept_edges(ctx["A"], ctx["B"], fixed) > objectives.max():
            perm, kept_previous = fixed, True
    # Always use the newly computed ensemble under current seed constraints, including after a guard.
    uncertainty = 1 - T.modal_agreement(permutations)
    uncertainty[seeds] = 0
    meta = {**metrics(ctx, perm, seeds), "kept_previous": kept_previous,
            "runtime_seconds": float(sum(o["runtime_seconds"] for o in outs)),
            "soft_valid_fraction": float(np.mean([o["soft_check"] == 1 for o in outs])),
            "distinct_matchings": int(len(np.unique(permutations, axis=0))), "ensemble_runs": runs,
            "objective": T.kept_edges(ctx["A"], ctx["B"], perm),
            "warnings": [w for o in outs for w in o["warnings"]]}
    arrays = {"perm": perm, "permutations": permutations, "uncertainty": uncertainty,
              "correct": perm == ctx["truth"], "seeds": seeds, "test": ctx["test"], "truth": ctx["truth"]}
    save(path, meta, **arrays)
    return {**meta, **arrays}


def structural_next(ctx, seeds, generator):
    g = graph(ctx)
    state = SignatureTarget(g)
    for v in seeds:
        state.add(int(v))
    available = np.setdiff1d(ctx["eligible"], seeds)
    order = available[np.lexsort((generator.random(len(available)), -g.degree[available]))]
    pool = order[:2 * (len(seeds) + 1)]
    ranked = sorted(pool, key=lambda v: (state.score(int(v)), int(g.degree[v])), reverse=True)
    width = min(len(pool), max(2, int(np.ceil(.2 * len(pool)))))
    return int(generator.choice(ranked[:width]))


def query_scores(ctx, state):
    u = state["uncertainty"]
    return {"uncertainty": u, "uncertainty_impact": u * (ctx["A"] @ u)}


def select_query(policy, ctx, state, generator):
    eligible = np.setdiff1d(ctx["eligible"], state["seeds"])
    if policy == "random":
        return int(generator.choice(eligible))
    if policy == "structural":
        return structural_next(ctx, state["seeds"], generator)
    scores = query_scores(ctx, state)[policy]
    # Degree breaks exact score ties, then randomized order; no outcome enters acquisition.
    order = np.lexsort((generator.random(len(eligible)), -ctx["A"].sum(axis=1)[eligible], -scores[eligible]))
    return int(eligible[order[0]])


def initial_seeds(ctx, base, k, repetition, purpose):
    return C.randomized_rule(base, graph(ctx), ctx["eligible"], k,
                             rng(ctx["config"], ctx["case"], ctx["split"], purpose, base, k, repetition),
                             ctx["config"]["selection"])


def active_job(root, config, task, deadline):
    _, case, algorithm, base, repetition = task
    split = repetition % config["splits"][size(case, config)]
    ctx = context(case, split, config)
    folder = root / "active" / case / algorithm / base / f"rep{repetition:02d}"
    parts = (case, algorithm, base, repetition, "active_solver")
    seeds = initial_seeds(ctx, base, config["active_initial"], repetition, "active")
    initial = ensemble(folder / "initial.npz", ctx, seeds, algorithm, (*parts, len(seeds)),
                       config["ensemble_runs"][ctx["size"]], deadline)
    states = {p: initial for p in POLICIES}
    # Advance all policies one budget at a time. Partial runs still have balanced lower-budget comparisons.
    for k in range(config["active_initial"], config["active_final"] + 1):
        for policy in POLICIES:
            state = states[policy]
            path = folder / policy / f"k{k:03d}.npz"
            if path.exists():
                meta, arrays = read(path)
                states[policy] = {**meta, **arrays}
                continue
            if k > config["active_initial"]:
                previous = state
                v = select_query(policy, ctx, previous,
                                 rng(config, case, algorithm, base, repetition, policy, k, "choice"))
                seeds = np.sort(np.append(previous["seeds"], v))
                state = ensemble(folder / policy / f"state{k:03d}.npz", ctx, seeds, algorithm,
                                 (*parts, k), config["ensemble_runs"][ctx["size"]], deadline,
                                 incumbent=previous["perm"])
                query_info = {"query_vertex": v, "query_was_wrong": bool(previous["perm"][v] != ctx["truth"][v]),
                              "gain_test": state["accuracy_test"] - previous["accuracy_test"],
                              "gain_all": state["accuracy_all"] - previous["accuracy_all"]}
            else:
                query_info = {}
            eligible = np.setdiff1d(ctx["eligible"], state["seeds"])
            meta = {k_: v_ for k_, v_ in state.items() if not isinstance(v_, np.ndarray)}
            meta.update(case=case, algorithm=algorithm, base=base, repetition=repetition, split=split,
                        policy=policy, budget=k, **query_info,
                        auc_uncertainty=T.auc_wrong(state["uncertainty"][eligible],
                                                   state["perm"][eligible] != ctx["truth"][eligible]))
            arrays = {k_: v_ for k_, v_ in state.items() if isinstance(v_, np.ndarray)}
            save(path, meta, **arrays)
            states[policy] = {**meta, **arrays}


def diagnostic_candidates(ctx, state, limit, generator):
    available = np.setdiff1d(ctx["eligible"], state["seeds"])
    if limit == 0 or limit >= len(available):
        return generator.permutation(available)
    scores = query_scores(ctx, state)
    orders = [generator.permutation(available),
              available[np.argsort(-scores["uncertainty"][available], kind="stable")],
              available[np.argsort(-scores["uncertainty_impact"][available], kind="stable")]]
    s = SignatureTarget(graph(ctx))
    for v in state["seeds"]:
        s.add(int(v))
    orders.append(sorted(available, key=lambda v: s.score(int(v)), reverse=True))
    chosen = []
    for i in range(len(available)):
        for order in orders:
            v = int(order[i])
            if v not in chosen:
                chosen.append(v)
                if len(chosen) >= limit:
                    return np.asarray(chosen)
    return np.asarray(chosen)


def diagnostic_job(root, config, task, deadline):
    _, case, algorithm, k, repetition = task
    split = repetition % config["splits"][size(case, config)]
    ctx = context(case, split, config)
    folder = root / "diagnostic" / case / algorithm / f"k{k:03d}" / f"rep{repetition:02d}"
    parts = (case, algorithm, k, repetition, "diagnostic")
    seeds = initial_seeds(ctx, "distinguishing", k, repetition, "diagnostic")
    initial = ensemble(folder / "initial.npz", ctx, seeds, algorithm, (*parts, "initial"),
                       config["ensemble_runs"][ctx["size"]], deadline)
    plan_path = folder / "plan.json"
    if not plan_path.exists():
        candidates = diagnostic_candidates(ctx, initial, config["diagnostic_candidates"][ctx["size"]],
                                           rng(config, *parts, "candidates"))
        write_json(plan_path, {"case": case, "algorithm": algorithm, "budget": k, "repetition": repetition,
                              "split": split, "seeds": seeds.tolist(), "candidates": candidates.tolist()})
    candidates = json.loads(plan_path.read_text())["candidates"]
    baseline = [evaluation(folder / "baseline" / f"r{r}.npz", ctx, seeds, algorithm,
                           seed(config, *parts, "paired_solver", r), deadline)
                for r in range(config["diagnostic_runs"])]
    scores = query_scores(ctx, initial)
    target = SignatureTarget(graph(ctx))
    for s in seeds:
        target.add(int(s))
    for v in candidates:
        path = folder / f"v{v:04d}.json"
        if path.exists():
            continue
        outcomes = [evaluation(folder / f"v{v:04d}" / f"r{r}.npz", ctx, np.sort(np.append(seeds, v)), algorithm,
                               seed(config, *parts, "paired_solver", r), deadline)
                    for r in range(config["diagnostic_runs"])]
        gains = np.asarray([o["accuracy_test"] - b["accuracy_test"] for o, b in zip(outcomes, baseline)])
        write_json(path, {"case": case, "algorithm": algorithm, "budget": k, "repetition": repetition,
                         "split": split, "vertex": v, "degree": int(ctx["A"][v].sum()),
                         "uncertainty": float(scores["uncertainty"][v]),
                         "uncertainty_impact": float(scores["uncertainty_impact"][v]),
                         "distinguishing_score": target.score(v)[0],
                         "was_wrong": bool(initial["perm"][v] != ctx["truth"][v]),
                         "gain_test": float(gains.mean()), "paired_gains": gains.tolist(),
                         "gain_all": float(np.mean([o["accuracy_all"] - b["accuracy_all"]
                                                   for o, b in zip(outcomes, baseline)])),
                         "baseline_test": float(np.mean([b["accuracy_test"] for b in baseline]))})


WORKERS = {"pilot": pilot_job, "strategy": strategy_job, "active": active_job, "diagnostic": diagnostic_job}


def worker(root, config, task, deadline, blas_threads):
    try:
        check_time(deadline)
        with thread_limit(blas_threads):
            WORKERS[task[0]](Path(root), config, task, deadline)
        return "done", task
    except TimeoutError:
        return "paused", task
    except Exception:
        key = hashlib.sha256(json.dumps(task).encode()).hexdigest()[:16]
        write_json(Path(root) / "errors" / f"{key}.json", {"task": task, "traceback": traceback.format_exc()})
        return "error", task


def task_key(task):
    return hashlib.sha256(json.dumps(task).encode()).hexdigest()[:20]


def phase(label, tasks, root, config, jobs, blas_threads, deadline):
    tasks = [t for t in tasks if not (root / "completed" / (task_key(t) + ".json")).exists()]
    if not tasks or time.monotonic() >= deadline:
        return
    print(f"{label}: {len(tasks)} unfinished tasks, {jobs} workers", flush=True)
    iterator = iter(tasks)
    pool = ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn"))
    pending = {}
    completed = 0
    last_report = time.monotonic()
    try:
        while True:
            while len(pending) < jobs and time.monotonic() < deadline:
                task = next(iterator, None)
                if task is None:
                    break
                pending[pool.submit(worker, str(root), config, task, deadline, blas_threads)] = task
            if not pending:
                break
            done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                try:
                    status_, task = future.result()
                except Exception:
                    write_json(root / "errors" / f"{task_key(task)}.json",
                               {"task": task, "traceback": traceback.format_exc()})
                    status_ = "error"
                if status_ == "done":
                    write_json(root / "completed" / (task_key(task) + ".json"), {"task": task})
                    # Successful retries clear their earlier error report.
                    error_key = hashlib.sha256(json.dumps(task).encode()).hexdigest()[:16]
                    (root / "errors" / f"{error_key}.json").unlink(missing_ok=True)
                    (root / "errors" / f"{task_key(task)}.json").unlink(missing_ok=True)
                    completed += 1
                elif status_ == "error":
                    print(f"ERROR: {task}; details in {root / 'errors'}", flush=True)
            if time.monotonic() - last_report >= 60:
                print(f"{label}: {completed}/{len(tasks)} tasks finished this pass", flush=True)
                last_report = time.monotonic()
    finally:
        # Workers stop between runs; an in-flight solve is allowed to finish and save.
        pool.shutdown(wait=True, cancel_futures=True)
    print(f"{label}: saved {completed} completed tasks", flush=True)


def prepare(root, config, blas_threads):
    root.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), Path(T.__file__), Path(C.__file__), Path(C.F.__file__),
             Path(__file__).with_name("datasets.py"),
             C.ROOT / "selection_rule_comparison/strategies.py", Path(C.F.MP.__file__),
             C.ROOT / "experiment_common/percolation.py"]
    expected = {"config": config, "code_hashes": {str(p.relative_to(C.ROOT)):
                hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    path = root / "settings.json"
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(f"Collection settings/code changed. Use a new --output-dir; preserving {root}.")
    else:
        write_json(path, expected)
        write_json(root / "environment.json", collect_environment(blas_threads))


def strategy_tasks(config, selected, large):
    cases = [c for c in config["cases"] if (size(c, config) == "large") == large]
    level = "large" if large else "small"
    # All splits/budgets/methods get draw zero before extra selection draws begin.
    for draw in range(config["selection"]["repeats"][level]):
        for split in range(config["splits"][level]):
            for j in range(4):
                for algorithm in C.ALGORITHMS:
                    for case in cases:
                        budgets = selected[case][algorithm]["budgets"]
                        if j < len(budgets):
                            yield ("strategy", case, split, algorithm, budgets[j], draw)


def diagnostic_tasks(config, large):
    level = "large" if large else "small"
    cases = ["wikipedia"] if large and "wikipedia" in config["cases"] else ([] if large else config["active_cases"])
    algorithms = ["faq"] if large else config["active_algorithms"]
    for repetition in range(config["diagnostic_repeats"][level]):
        for k in config["diagnostic_budgets"][level]:
            for case in cases:
                for algorithm in algorithms:
                    yield ("diagnostic", case, algorithm, k, repetition)


def status(root=OUTPUT_DIR):
    root = Path(root)
    if not (root / "settings.json").exists():
        print(f"No collection yet: {root}")
        return
    counts = {}
    for p in (root / "completed").glob("*.json"):
        kind = json.loads(p.read_text())["task"][0]
        counts[kind] = counts.get(kind, 0) + 1
    print(f"Results: {root}\nCompleted tasks: {counts}\nUnresolved errors: {len(list((root / 'errors').glob('*.json')))}")
    progress = root / "run_status.json"
    if progress.exists():
        print(progress.read_text())


def run(root=OUTPUT_DIR, hours=6.5, jobs=3, blas_threads=2, smoke=False):
    import fcntl
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = default_config(smoke)
    started = time.monotonic()
    deadline = started + hours * 3600
    with (root / ".collection.lock").open("w") as lock, C._awake():
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another collector is already using {root}") from None
        prepare(root, config, blas_threads)
        write_json(root / "run_status.json", {"state": "running", "hours_this_invocation": hours,
                                              "started_at": C.F._now()})
        pilot_small = [("pilot", c, a) for c in config["cases"] if size(c, config) == "small" for a in C.ALGORITHMS]
        pilot_large = [("pilot", c, a) for c in config["cases"] if size(c, config) == "large" for a in C.ALGORITHMS]
        phase("Small-pair random calibration", pilot_small, root, config, jobs, blas_threads, deadline)
        phase("Wikipedia random calibration", pilot_large, root, config, min(2, jobs), blas_threads, deadline)
        selected = calibration_plan(root, config)
        # These are caps, not delays; unused time immediately carries forward.
        def capped(hours_):
            return min(deadline, time.monotonic() + hours_ * 3600)
        phase("Small-pair query diagnostics", list(diagnostic_tasks(config, False)), root, config,
              jobs, blas_threads, capped(.5))
        small_tasks = list(strategy_tasks(config, selected, False)) if selected else []
        small_deadline = capped(.75)
        for draw in range(config["selection"]["repeats"]["small"]):
            phase(f"Small-pair confirmation draw {draw + 1}", [t for t in small_tasks if t[-1] == draw],
                  root, config, jobs, blas_threads, small_deadline)
        active_tasks = [("active", c, a, b, r) for r in range(config["active_repeats"])
                        for c in config["active_cases"] for a in config["active_algorithms"]
                        for b in config["active_bases"]]
        phase("Repeated Enron active queries", active_tasks, root, config, jobs, blas_threads, capped(.75))
        phase("Wikipedia one-query diagnostic", list(diagnostic_tasks(config, True)), root, config,
              min(2, jobs), blas_threads, capped(.75))
        large_tasks = list(strategy_tasks(config, selected, True)) if selected else []
        # Prevent two different draws writing the same bank's selection plan concurrently.
        for draw in range(config["selection"]["repeats"]["large"]):
            phase(f"Wikipedia confirmation draw {draw + 1}", [t for t in large_tasks if t[-1] == draw],
                  root, config, min(2, jobs), blas_threads, deadline)
        for draw in range(config["selection"]["repeats"]["small"]):
            phase(f"Remaining small confirmation draw {draw + 1}", [t for t in small_tasks if t[-1] == draw],
                  root, config, jobs, blas_threads, deadline)
        for label, tasks, workers in [
            ("Remaining active trajectories", active_tasks, jobs),
            ("Remaining small diagnostics", list(diagnostic_tasks(config, False)), jobs),
            ("Remaining Wikipedia diagnostics", list(diagnostic_tasks(config, True)), min(2, jobs))]:
            phase(label, tasks, root, config, workers, blas_threads, deadline)
        all_tasks = [*pilot_small, *pilot_large, *small_tasks, *large_tasks, *active_tasks,
                     *diagnostic_tasks(config, False), *diagnostic_tasks(config, True)]
        remaining = sum(not (root / "completed" / (task_key(t) + ".json")).exists() for t in all_tasks)
        errors = len(list((root / "errors").glob("*.json")))
        write_json(root / "run_status.json", {"state": "complete" if not remaining and selected else "paused",
                   "elapsed_hours": (time.monotonic() - started) / 3600, "remaining_tasks": remaining,
                   "calibration_complete": selected is not None, "errors": errors, "updated_at": C.F._now()})
        status(root)
        return not errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=float, default=6.5)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--blas-threads", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    root = args.output_dir or (OUTPUT_DIR.with_name("overnight_v2_smoke") if args.smoke else OUTPUT_DIR)
    if args.status:
        status(root)
    else:
        if args.hours <= 0 or args.jobs < 1 or args.blas_threads < 1:
            parser.error("hours, jobs, and blas-threads must be positive")
        if not run(root, args.hours, args.jobs, args.blas_threads, args.smoke):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
