"""Collect A-only seed banks and matching outcomes; plotting belongs in notebooks.

From the repository root, in the ds environment:
    python -u -m selection_rule_comparison.collect_score_ranges --hours 8 --jobs 3
    python -m selection_rule_comparison.collect_score_ranges --status
    python -m selection_rule_comparison.collect_score_ranges --smoke --hours 0.1

The identical command resumes successful evaluations. Original experiments and
saved selectors are read only. See SCORE_RANGE_COLLECTION.md for the design.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import traceback
import warnings

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache"))

import numpy as np
import pandas as pd

from experiment_common.graphs import generate_graph_pair
from experiment_common.provenance import collect_environment, thread_limit
from experiment_common.seed_set_sampling import select_bank, swap_paths, unique_sets
from real_data_experiments import first_look as F
from real_data_experiments import broader_candidates as FU
from . import b_replicates as BR
from . import fresh_pair_extremes as E
from . import graded_outcomes as G
from . import landscape as L
from . import learned_selector as LS
from . import regime_map as RM
from . import run_comparison as RC
from .strategies import GraphA, STRATEGY_CODES

DEFAULT_DIR = L.EXPERIMENT_DIR / "data" / "score_range_collection"
REPO_ROOT = LS.REPO_ROOT
SEED_BASE = 2026111701
REAL_BUDGETS = {"enron_130_131": [5, 10, 20], "enron_130_132": [5, 10, 20],
                "enron_145_147": [5, 10, 20], "enron_160_162": [5, 10, 20],
                "worm": [50, 100], "worm_lcc": [50, 100], "worm_male": [20, 40],
                "wikipedia": [10, 20]}
SUFFIX = {"sparse_er": "er", "chung_lu": "cl", "paper_pa": "paper"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_csv(path, frame):
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def check_time(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Collection time budget reached")


def prepare(root, args):
    fits = LS.final_models()
    if len(fits) != 27:
        raise ValueError("Expected all 27 saved final selectors before starting this collection.")
    for cell, fit in fits.items():
        if any(s < 1e-9 * max(1, abs(c)) for c, s in zip(fit["centre"], fit["scale"])):
            raise ValueError(f"Repair the near-constant normalization in {LS.cell_key(*cell)} first.")
    real = REAL_BUDGETS if not args.smoke else {"enron_130_132": [5]}
    cells = [list(c) for c in fits]
    if args.smoke:
        cells = [next(c for c in cells if c[0] == "sparse_er" and c[1] == a) for a in LS.ALGORITHMS]
    banks = []
    for case, budgets in real.items():
        for k in budgets:
            banks.append({"kind": "real", "case": case, "budget": k,
                          "id": f"real/{case}/k{k:03d}", "algorithms": list(LS.ALGORITHMS)})
    for model, algorithm, degree, rho, k in cells:
        banks.append({"kind": "synthetic", "model": model, "algorithm": algorithm,
                      "mean_degree": degree, "rho": rho, "budget": k, "pair_index": args.pair_index,
                      "case": f"{model}/{RM.cell_name(degree, rho)}",
                      "id": f"synthetic/{model}/{RM.cell_name(degree, rho)}/{algorithm}/k{k:03d}/pair_{args.pair_index:03d}",
                      "algorithms": [algorithm]})
    sources = {LS.cell_key(*c): fit for c, fit in fits.items()}
    code_files = [Path(__file__), Path(E.__file__), Path(G.__file__), Path(LS.__file__),
                  Path(F.__file__), Path(RM.__file__),
                  REPO_ROOT / "experiment_common/seed_set_sampling.py",
                  REPO_ROOT / "experiment_common/sgm.py",
                  REPO_ROOT / "experiment_common/percolation.py"]
    settings = {"version": 1, "smoke": args.smoke, "seed_base": SEED_BASE, "banks": banks,
                "sets": args.sets or (12 if args.smoke else 150),
                "wiki_sets": args.wiki_sets or (8 if args.smoke else 100),
                "random_sets": 3 if args.smoke else 30, "bins": 4 if args.smoke else 8,
                "reference_pool": 48 if args.smoke else 1000,
                "path_starts": 1 if args.smoke else 4, "path_steps": 3 if args.smoke else 24,
                "search_steps": 1 if args.smoke else 4, "search_swaps": 8 if args.smoke else 64,
                "synthetic_draws": 3, "small_solver_runs": 3, "wiki_solver_runs": 2,
                "real_goat_regularizer": 300.0, "test_share": F.TEST_SHARE,
                "blas_threads": args.blas_threads, "solver": F.S.SGM, "percolation": F.S.PERCOLATION,
                "weights": {m: RM.load_weights(m) for m in LS.MODELS},
                "selector_hashes": {key: digest(fit) for key, fit in sources.items()},
                "code_hashes": {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in code_files}}
    settings = json.loads(json.dumps(settings))
    path = root / "settings.json"
    if path.exists():
        if F._read_json(path) != settings:
            raise ValueError("Saved design, selectors or collection code changed. Use a new --output-dir.")
    else:
        root.mkdir(parents=True, exist_ok=True)
        F._write_json(path, settings)
        F._write_json(root / "selector_sources.json", sources)
        F._write_json(root / "environment.json", collect_environment(args.blas_threads))
    return settings, sources


def source_pair(task):
    if task["kind"] == "real":
        _, A, B, truth, test, eligible, info = F.load_pair(
            task["case"], {"pairs": FU.CASES, "test_share": F.TEST_SHARE})
    else:
        cell = (task["model"], task["algorithm"], task["mean_degree"], task["rho"], task["budget"])
        source = LS.cell_dir(LS.read_design(), *cell) / "fresh" / f"pair_{task['pair_index']:03d}" / "pair.json"
        info = F._read_json(source)
        A, B, truth = generate_graph_pair(RM.graph_config(task["model"], task["mean_degree"], task["rho"]),
                                          int(info["pair_seed"]))
        if RC._hash(A) != info["hash_a"] or RC._hash(B) != info["hash_b"]:
            raise ValueError(f"Fresh source graph differs: {source}")
        test = np.asarray(info["test_vertices"], dtype=int)
        eligible = np.setdiff1d(np.arange(len(A)), test)
        info = {**info, "source_pair": str(source.relative_to(REPO_ROOT))}
    return A, B, truth, test, eligible, info


class Scorer:
    """Fixed random-pool normalization and saved predictors; never accesses B or outcomes."""

    def __init__(self, task, graph, eligible, pool, settings, sources, info, deadline):
        self.task, self.graph, self.settings, self.deadline = task, graph, settings, deadline
        self.stats, self.fits = {}, {}
        self.calculators = {cap: F.MP.PoolFeatures(graph, cap) for cap in (2, 3)}
        for a in task["algorithms"]:
            check_time(deadline)
            reference = pool
            if task["kind"] == "synthetic" and not settings["smoke"]:
                # Preserve exactly the original fresh pair's random-pool normalization.
                ld = F._read_json(L.OUTPUT_DIR / task["model"] / "settings.json")
                reference = G.draw_sets(L._rng(info["pair_seed"], "pool", task["budget"], F.MP.ALGORITHM_CODES[a]),
                                        eligible, task["budget"], ld["pool"])
            raw = self.raw(a, reference)
            self.stats[a] = {f: {"mean": float(raw[f].mean()), "sd": float(raw[f].std(ddof=1))}
                             for f in F.CORE}
            if task["kind"] == "synthetic":
                key = LS.cell_key(task["model"], a, task["mean_degree"], task["rho"], task["budget"])
                self.fits[a] = [(key, sources[key])]
            else:
                self.fits[a] = [(key, fit) for key, fit in sources.items() if key.split("/")[1] == a
                                and (a != "goat" or F.S.GOAT_REGULARIZER[key.split("/")[0]] ==
                                     settings["real_goat_regularizer"])]

    def raw(self, algorithm, arrays):
        cap = F.MP.UNIQUENESS_CAP[algorithm]
        chunk = max(1, min(64, 4_000_000 // (self.graph.n * arrays.shape[1])))
        return self.calculators[cap].compute(arrays, chunk=chunk)

    def frame(self, algorithm, raw):
        frame = pd.DataFrame({f: raw[f] for f in F.CORE})
        for f in F.CORE:
            stat = self.stats[algorithm][f]
            frame[f"z_{f}"] = (frame[f] - stat["mean"]) / stat["sd"] if stat["sd"] > 0 else 0.0
        return frame

    def __call__(self, arrays):
        check_time(self.deadline)
        arrays = np.atleast_2d(arrays)
        raw_by_cap = {}
        pieces = []
        for a in self.task["algorithms"]:
            cap = F.MP.UNIQUENESS_CAP[a]
            if cap not in raw_by_cap:
                raw_by_cap[cap] = self.raw(a, arrays)
            frame = self.frame(a, raw_by_cap[cap])
            for model, suffix in SUFFIX.items():
                frame[f"score_{suffix}"] = sum(self.settings["weights"][model][a][f] * frame[f"z_{f}"] for f in F.CORE)
            predictions = [LS.predict({**fit, "bootstrap": []}, frame)[0] for _, fit in self.fits[a]]
            frame["predicted_mean"] = np.mean(predictions, axis=0)
            frame["source_range_share"] = np.mean([LS.in_range(fit, frame) for _, fit in self.fits[a]], axis=0)
            pieces.append(frame.add_prefix(f"{a}__"))
        out = pd.concat(pieces, axis=1)
        first = self.task["algorithms"][0]
        for f in F.CORE:
            out[f] = out[f"{first}__{f}"]
        out["coverage"] = self.coverage(arrays)
        return out

    def coverage(self, arrays):
        values = []
        for block in np.array_split(arrays, max(1, int(np.ceil(len(arrays) / 32)))):
            w = self.graph.adjacency[:, block].sum(axis=2, dtype=np.int64)
            w[block.T, np.arange(len(block))[None, :]] = 0
            values.extend((w > 0).sum(axis=0) / (self.graph.n - arrays.shape[1]))
        return np.asarray(values)

    def score_columns(self):
        columns = []
        for a in self.task["algorithms"]:
            suffixes = list(SUFFIX.values()) if self.task["kind"] == "real" else [SUFFIX[self.task["model"]]]
            columns.extend(f"{a}__score_{s}" for s in suffixes)
            columns.append(f"{a}__predicted_mean")
        return columns

    def predictions(self, selected):
        rows = []
        for a in self.task["algorithms"]:
            columns = {f"{a}__{f}": f for f in (*F.CORE, *LS.Z)}
            frame = selected[list(columns)].rename(columns=columns)
            for source, fit in self.fits[a]:
                rows.append(pd.DataFrame({"candidate_id": selected.candidate_id, "algorithm": a,
                                         "source": source, "predicted": LS.predict({**fit, "bootstrap": []}, frame)[0],
                                         "in_source_range": LS.in_range(fit, frame)}))
        return pd.concat(rows, ignore_index=True)


def extra_features(graph, arrays):
    rows = []
    for seeds in arrays:
        nonseeds = np.setdiff1d(np.arange(graph.n), seeds)
        signatures = graph.adjacency[np.ix_(nonseeds, seeds)]
        w = signatures.sum(axis=1, dtype=np.int64)
        _, counts = np.unique(signatures, axis=0, return_counts=True)
        fractions = counts / len(nonseeds)
        rows.append({"total_degree_mean": float(graph.degree[seeds].mean()),
                     "seed_internal_edges": int(graph.adjacency[np.ix_(seeds, seeds)].sum() // 2),
                     "coverage_two": float(np.mean(w >= 2)), "coverage_three": float(np.mean(w >= 3)),
                     "witness_q10": float(np.quantile(w, .1)), "witness_q50": float(np.quantile(w, .5)),
                     "witness_q90": float(np.quantile(w, .9)),
                     "adjacency_signature_collision_fraction": float(np.sum(counts * (counts - 1)) /
                                                                     max(1, len(nonseeds) * (len(nonseeds) - 1))),
                     "adjacency_signature_normalized_entropy": float(-np.sum(fractions * np.log(fractions)) /
                                                                      np.log(len(nonseeds))) if len(nonseeds) > 1 else 0.0})
    return pd.DataFrame(rows)


def make_bank(task, root, settings, sources, deadline):
    folder = root / task["id"]
    if (folder / "sets.csv").exists():
        return folder
    check_time(deadline)
    A, B, truth, test, eligible, info = source_pair(task)
    k = task["budget"]
    if not 0 < k < len(eligible):
        raise ValueError(f"Invalid seed budget for {task['id']}")
    graph = GraphA(A)
    seed = RC._seed(SEED_BASE, F._code(task["id"]))
    rng = np.random.default_rng(seed)
    pool = G.draw_sets(rng, eligible, k, settings["reference_pool"])
    scorer = Scorer(task, graph, eligible, pool, settings, sources, info, deadline)
    values = scorer(pool)
    columns = scorer.score_columns()
    rules = {}
    for name in F.RULES:
        check_time(deadline)
        rules[name] = np.sort(RM._ordering(name, graph, k,
                          np.random.default_rng([seed, STRATEGY_CODES[name]]), test))
    endpoints = list(rules.values())
    for weights in (graph.degree[eligible], graph.harmonic[eligible]):
        endpoints.extend(eligible[s] for s in E.top_additive_sets(weights, k, 3))
    endpoints.extend(pool[int(values[col].idxmax())] for col in columns)
    known = unique_sets([*endpoints, *pool[:settings["path_starts"]]])
    known_values = scorer(np.stack(known))
    # Sampled single-swap searches. These produce high values, without claiming global maxima.
    goals = ["degree", "pair_mass", "coverage"] + [c for c in columns if c.endswith("predicted_mean")]
    goals += [f"{a}__score_{SUFFIX[task.get('model', 'sparse_er')]}" for a in task["algorithms"]]
    for j, goal in enumerate(goals):
        check_time(deadline)
        starts = [known[int(known_values[goal].idxmax())], pool[j % settings["path_starts"]]]
        if goal in ("degree", "pair_mass", "coverage"):
            fast = E.fast_objective(graph, scorer, goal, k)
            def calculate(arrays, fast=fast):
                check_time(deadline)
                return fast(arrays)
        else:
            calculate = scorer
        endpoints.extend(E.search_sets(calculate, goal, known, known_values[goal].to_numpy(), starts,
                         graph.n, np.random.default_rng([seed, 40, j]), 3, settings["search_steps"],
                         settings["search_swaps"], candidates=eligible))
    endpoints = unique_sets(endpoints)
    paths = swap_paths(pool[:settings["path_starts"]], endpoints, np.random.default_rng([seed, 50]),
                       settings["path_steps"])
    arrays = np.stack(unique_sets([*pool, *endpoints, *paths]))
    # Compute in bounded blocks to honour the deadline during candidate generation.
    bank_values = pd.concat([scorer(block) for block in np.array_split(arrays, max(1, int(np.ceil(len(arrays)/128))))],
                            ignore_index=True)
    index = {tuple(s): i for i, s in enumerate(arrays)}
    rule_indices = {name: index[tuple(s)] for name, s in rules.items()}
    count = settings["wiki_sets"] if task["case"] == "wikipedia" else settings["sets"]
    n_random = min(settings["random_sets"], max(1, count // 5))
    selected, members, coverage = select_bank(arrays, bank_values, columns, rule_indices,
                          count, n_random, np.random.default_rng([seed, 60]), settings["bins"])
    selected_arrays = np.stack([LS._seeds(s) for s in selected.seed_vertices])
    assert all(len(np.unique(s)) == k and np.isin(s, eligible).all() for s in selected_arrays)
    selected = pd.concat([selected, extra_features(graph, selected_arrays)], axis=1)
    check_time(deadline)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "pair.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, A=A, B=B, truth=truth, test=test)
    temporary.replace(folder / "pair.npz")
    F._write_json(folder / "pair.json", {**info, **task, "sampling_seed": seed,
                  "reference_stats": scorer.stats, "candidate_bank_size": len(arrays),
                  "n_sets": len(selected), "sets_hash": hashlib.sha256(selected_arrays.tobytes()).hexdigest(),
                  "contrast_uniqueness_cap": F.MP.UNIQUENESS_CAP[task["algorithms"][0]],
                  "source_selector_hashes": settings["selector_hashes"], "written_at": F._now()})
    write_csv(folder / "memberships.csv", members)
    write_csv(folder / "coverage.csv", coverage)
    write_csv(folder / "predictions.csv", scorer.predictions(selected))
    write_csv(folder / "sets.csv", selected)  # Commit marker: written after every other bank file.
    return folder


def run_id(candidate_id, draw, repeat):
    return 100 * int(candidate_id) + 10 * int(draw) + int(repeat)


def blocks(task, stage, settings):
    if stage == "real_small" and task["kind"] == "real" and task["case"] != "wikipedia":
        return [(a, 0, r) for a in task["algorithms"]
                for r in range(1 if a == "percolation" else settings["small_solver_runs"])]
    if stage == "synthetic_first" and task["kind"] == "synthetic":
        return [(task["algorithm"], 0, 0)]
    if stage == "wikipedia" and task["case"] == "wikipedia":
        return [(a, 0, 0) for a in task["algorithms"]]
    if stage.startswith("synthetic_draw_") and task["kind"] == "synthetic":
        return [(task["algorithm"], int(stage.rsplit("_", 1)[1]), 0)]
    if stage == "wikipedia_repeats" and task["case"] == "wikipedia":
        return [(a, 0, r) for a in ("faq", "goat") for r in range(1, settings["wiki_solver_runs"])]
    return []


def stage_complete(task, root, requested):
    folder = root / task["id"]
    if not (folder / "sets.csv").exists():
        return False
    n = F._read_json(folder / "pair.json")["n_sets"]
    for a, draw, repeat in requested:
        done = G._completed_ids(folder / f"{a}_runs.csv", "run_id")
        if any(run_id(i, draw, repeat) not in done for i in range(n)):
            return False
    return True


def collect_job(task, root, settings, sources, deadline, stage, plan_only):
    root = Path(root)
    start = time.monotonic()
    written = 0
    try:
        with thread_limit(settings["blas_threads"]):
            folder = make_bank(task, root, settings, sources, deadline)
            sets = pd.read_csv(folder / "sets.csv")
            if plan_only:
                return {"task": task["id"], "sets": len(sets), "runs": 0, "state": "planned"}
            with np.load(folder / "pair.npz") as saved:
                A, B, truth, test = [saved[k] for k in ("A", "B", "truth", "test")]
            info = F._read_json(folder / "pair.json")
            draws = {0: (B, truth)}
            requested = blocks(task, stage, settings)
            for a, draw, repeat in requested:
                check_time(deadline)
                if draw not in draws:
                    probability = L.b_probability(task["model"], task["mean_degree"], task["rho"],
                                                  info["pair_seed"], A, B, truth)
                    new_b, new_truth, draw_info = BR.draw_b(A, probability, info["pair_seed"], draw)
                    draws[draw] = new_b, new_truth
                    F._write_json(folder / f"draw_{draw:02d}.json",
                                  {"draw": draw, "hash_b": RC._hash(new_b),
                                   "truth_hash": digest(new_truth.tolist()), "rejected_draws": draw_info})
                new_b, new_truth = draws[draw]
                hash_b = RC._hash(new_b)
                regularizer = (settings["real_goat_regularizer"] if task["kind"] == "real" else
                               F.S.GOAT_REGULARIZER[task["model"]]) if a == "goat" else None
                solver_seed = RC._seed(SEED_BASE, F._code(task["id"]), F.MP.ALGORITHM_CODES[a], draw, repeat, 90)
                path = folder / f"{a}_runs.csv"
                done = G._completed_ids(path, "run_id")
                rows = G._Rows(path)
                try:
                    for row in sets.itertuples():
                        rid = run_id(row.candidate_id, draw, repeat)
                        if rid in done:
                            continue
                        check_time(deadline)
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            out = RM.run_one(a, task.get("model", "sparse_er"), (A, new_b, new_truth),
                                             LS._seeds(row.seed_vertices), test, solver_seed, regularizer)
                        record = {"kind": task["kind"], "case": task["case"], "model": task.get("model", "real"),
                                  "mean_degree": task.get("mean_degree", np.nan), "rho": task.get("rho", np.nan),
                                  "budget": task["budget"], "pair_index": task.get("pair_index", 0),
                                  "algorithm": a, "candidate_id": row.candidate_id, "draw": draw,
                                  "repeat": repeat, "run_id": rid, "solver_seed": solver_seed,
                                  "regularizer": regularizer, "hash_b": hash_b,
                                  **out, "warning_count": len(caught),
                                  "warnings": json.dumps(sorted({str(w.message)[:1000] for w in caught}))}
                        if out["error"] or not np.isfinite(out["accuracy_test"]):
                            failures = G._Rows(folder / f"{a}_errors.csv")
                            try:
                                failures.write(record)
                            finally:
                                failures.close()
                            raise RuntimeError(f"Matching failed at {task['id']}/{a}/{rid}: {out['error']}")
                        rows.write(record)
                        written += 1
                finally:
                    rows.close()
    except TimeoutError:
        return {"task": task["id"], "runs": written, "state": "time limit"}
    return {"task": task["id"], "runs": written, "state": "complete",
            "minutes": round((time.monotonic() - start) / 60, 2)}


def run_stage(stage, tasks, root, settings, sources, deadline, jobs, plan_only, report):
    todo = [t for t in tasks if (not (root / t["id"] / "sets.csv").exists() if plan_only else
                                 not stage_complete(t, root, blocks(t, stage, settings)))]
    report(f"{stage}: {len(todo)} banks remaining; {jobs} workers")
    if not todo:
        return 0
    failed = 0
    pool = futures.ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn"))
    pending, iterator = {}, iter(todo)
    completed = 0

    def submit():
        if deadline is not None and time.monotonic() >= deadline:
            return False
        task = next(iterator, None)
        if task is None:
            return False
        pending[pool.submit(collect_job, task, str(root), settings, sources, deadline, stage, plan_only)] = task
        return True

    try:
        for _ in range(jobs):
            submit()
        heartbeat = time.monotonic()
        while pending:
            ready, _ = futures.wait(pending, timeout=10, return_when=futures.FIRST_COMPLETED)
            for future in ready:
                task = pending.pop(future)
                completed += 1
                try:
                    report(f"[{completed}/{len(todo)}] {future.result()}")
                except Exception:
                    failed += 1
                    text = traceback.format_exc()
                    folder = root / task["id"]
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "collection_error.txt").write_text(text)
                    report(f"FAILED {task['id']}\n{text}")
                submit()
            if time.monotonic() - heartbeat >= 60:
                left = max(0, (deadline - time.monotonic()) / 3600) if deadline is not None else None
                report(f"{stage}: {len(pending)} active; time budget remaining {left:.2f} h" if left is not None
                       else f"{stage}: {len(pending)} active")
                heartbeat = time.monotonic()
    except BaseException:
        workers = list((getattr(pool, "_processes", None) or {}).values())
        pool.shutdown(wait=False, cancel_futures=True)
        for worker in workers:
            worker.terminate()
        raise
    else:
        pool.shutdown(wait=True)
    return failed


def status(root):
    settings = F._read_json(root / "settings.json")
    rows = []
    for task in settings["banks"]:
        folder = root / task["id"]
        n = F._read_json(folder / "pair.json")["n_sets"] if (folder / "sets.csv").exists() else 0
        for a in task["algorithms"]:
            path = folder / f"{a}_runs.csv"
            # A live status check must never trim a row while a worker is writing it.
            done = set()
            if path.exists():
                with path.open(newline="") as handle:
                    reader = csv.reader(handle)
                    header = next(reader, [])
                    if "run_id" in header:
                        column = header.index("run_id")
                        done = {int(row[column]) for row in reader if len(row) == len(header)}
            expected_repeats = (1 if a == "percolation" else settings["wiki_solver_runs"] if
                                task["case"] == "wikipedia" else settings["small_solver_runs"])
            planned_sets = n or (settings["wiki_sets"] if task["case"] == "wikipedia" else settings["sets"])
            expected = planned_sets * (settings["synthetic_draws"] if task["kind"] == "synthetic" else expected_repeats)
            baseline = sum(run_id(i, 0, 0) in done for i in range(n))
            rows.append({"bank": task["id"], "algorithm": a, "sets": n, "first_B": baseline,
                         "runs": len(done), "planned_runs": expected,
                         "complete": n > 0 and len(done) == expected})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--hours", type=float, default=8, help="wall-clock budget per invocation; finish the current run")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--wiki-jobs", type=int, default=2)
    parser.add_argument("--blas-threads", type=int, default=2)
    parser.add_argument("--only", choices=("all", "real", "synthetic"), default="all")
    parser.add_argument("--cases", nargs="+", choices=REAL_BUDGETS, help="optional filter for real pairs")
    parser.add_argument("--sets", type=int, help="candidate-bank size for small real and synthetic cases")
    parser.add_argument("--wiki-sets", type=int)
    parser.add_argument("--pair-index", type=int, default=0, choices=range(10))
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="save seed banks and scores without running matching")
    parser.add_argument("--smoke", action="store_true", help="small isolated collection in tmp/score_range_smoke")
    args = parser.parse_args(argv)
    if args.hours <= 0 or min(args.jobs, args.wiki_jobs, args.blas_threads) < 1:
        parser.error("hours, workers and BLAS threads must be positive")
    if any(x is not None and x < 4 for x in (args.sets, args.wiki_sets)):
        parser.error("candidate-bank sizes must be at least 4")
    root = (args.output_dir or (REPO_ROOT / "tmp/score_range_smoke" if args.smoke else DEFAULT_DIR)).resolve()
    if args.status:
        print(status(root).to_string(index=False))
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / "collection.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"Another collection is using {root}")
        deadline = time.monotonic() + args.hours * 3600
        settings, sources = prepare(root, args)
        tasks = [t for t in settings["banks"] if (args.only == "all" or t["kind"] == args.only)
                 and (t["kind"] != "real" or not args.cases or t["case"] in args.cases)]
        with (root / "collection.log").open("a", buffering=1) as log:
            def report(message):
                line = f"{F._now()}  {message}"
                print(line, flush=True)
                log.write(line + "\n")
            report(f"Collection: {root}; {len(tasks)} banks; limit {args.hours:g} h")
            failed = 0
            stages = ["real_small", "synthetic_first", "wikipedia", "synthetic_draw_1", "synthetic_draw_2",
                      "wikipedia_repeats"]
            if args.plan_only:
                stages = ["plan"]
            try:
                for stage in stages:
                    if time.monotonic() >= deadline:
                        break
                    subset = tasks if args.plan_only else [t for t in tasks if blocks(t, stage, settings)]
                    failed += run_stage(stage, subset, root, settings, sources, deadline,
                                        min(args.jobs, args.wiki_jobs) if stage.startswith("wikipedia") else args.jobs,
                                        args.plan_only, report)
            except KeyboardInterrupt:
                report("Stopped by Ctrl-C. Successful rows are saved; rerun the same command to resume.")
                return
            report(f"Finished invocation; failed banks: {failed}. Rerun the same command to resume unfinished work.")
            report(f"Status: python -m selection_rule_comparison.collect_score_ranges --output-dir {root} --status")
            if failed:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
