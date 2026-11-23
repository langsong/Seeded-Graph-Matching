"""Compare seed-selection strategies on fresh graph pairs (round 1).

Every strategy is evaluated on the same structure-checked graph pairs at the same
budgets. A strategy is an ordering of Graph A's vertices; the seeds at budget k
are its first k vertices. Results are appended pair by pair, so an interrupted
run resumes where it stopped.

Run from the repository root (``ds`` environment). For example:

    python -m selection_rule_comparison.run_comparison                       # everything, main phase
    python -m selection_rule_comparison.run_comparison --algorithm goat      # one algorithm
    python -m selection_rule_comparison.run_comparison --pairs 10            # quicker
    python -m selection_rule_comparison.run_comparison --phase pilot         # pilot grids

The three graph models of an algorithm run in parallel (``--jobs``), each writing
progress to ``logs/``; algorithms run one after another. See README.md.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache"))

import numpy as np

from experiment_common.graph_checks import first_passing_pair
from experiment_common.graphs import generate_graph_pair
from experiment_common.provenance import collect_environment, thread_limit, write_json

from . import settings as S
from .strategies import STRATEGY_CODES, GraphA, ordering, seed_set_features

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = EXPERIMENT_DIR / "data" / "round1"

RUN_COLUMNS = (
    "algorithm", "model", "phase", "pair_index", "pair_seed", "strategy", "budget", "restart",
    "solver_seed", "seed_vertices", "final_unseeded_accuracy", "success", "success_090",
    "objective", "n_iter", "converged", "runtime_seconds", "cached",
    "h0_accuracy", "matched_fraction", "precision",
    "f_degree", "f_harmonic", "f_coverage", "f_pair_mass", "f_signature_r2", "f_signature_r3",
)
PAIR_COLUMNS = (
    "pair_index", "pair_seed", "first_seed_tried", "rejected_draws", "mean_degree_a", "max_degree_a",
    "mean_degree_b", "kept_edge_fraction", "hash_a", "hash_b", "pair_seconds",
)


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _seed(*parts: int) -> int:
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


def pair_first_seed(algorithm: str, model: str, phase: str, pair_index: int) -> int:
    family = S.FAMILY_CODES[S.GRAPH_FAMILY[algorithm]]
    seed = _seed(S.PAIR_SEED_BASE, family, S.MODEL_CODES[model], S.PHASE_CODES[phase], pair_index)
    return seed % (2**32 - 10_000)


def solver_seed(pair_seed: int, budget: int, restart: int) -> int:
    return _seed(S.SOLVER_SEED_BASE, pair_seed, budget, restart)


def _hash(adjacency: np.ndarray) -> str:
    packed = np.packbits((np.asarray(adjacency) > 0).astype(np.uint8))
    return hashlib.sha256(packed.tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Matchers
# --------------------------------------------------------------------------

_SGM_RUNNER = None


def _run_sgm_once(*args, **kwargs):
    """Import the project's graspologic wrapper only when an SGM run is needed."""

    global _SGM_RUNNER
    if _SGM_RUNNER is None:
        from experiment_common.sgm import run_sgm_once
        _SGM_RUNNER = run_sgm_once
    return _SGM_RUNNER(*args, **kwargs)


def evaluate(algorithm: str, model: str, pair, seeds: np.ndarray, budget: int, restart: int) -> dict:
    """One matcher run; returns the outcome fields of a run row."""

    adjacency_1, adjacency_2, truth, pair_seed = pair
    threshold = S.SUCCESS_THRESHOLD[algorithm]
    if algorithm == "percolation":
        from experiment_common.percolation import run_percolation_once
        out = run_percolation_once(adjacency_1, adjacency_2, truth, seeds,
                                   algorithm_seed=S.PERCOLATION["algorithm_seed"],
                                   threshold=S.PERCOLATION["threshold"],
                                   expand_when_stuck=S.PERCOLATION["expand_when_stuck"])
        accuracy = out["final_unseeded_accuracy"]
        return {
            "solver_seed": S.PERCOLATION["algorithm_seed"], "final_unseeded_accuracy": accuracy,
            "success": accuracy >= threshold, "success_090": accuracy >= 0.9, "objective": np.nan,
            "n_iter": np.nan, "converged": out["percolation_completed"],
            "runtime_seconds": out["percolation_runtime_seconds"], "h0_accuracy": np.nan,
            "matched_fraction": out["percolation_matched_unseeded_fraction"],
            "precision": out["percolation_unseeded_precision"],
        }
    seed_value = solver_seed(pair_seed, budget, restart)
    transport = algorithm == "goat"
    out = _run_sgm_once(adjacency_1, adjacency_2, truth, seeds, seed_value, S.SGM["max_iter"], S.SGM["tol"],
                        transport=transport,
                        transport_regularizer=S.GOAT_REGULARIZER[model],
                        transport_tol=S.SGM["transport_tol"],
                        transport_max_iter=S.SGM["transport_max_iter"])
    accuracy = out["final_unseeded_accuracy"]
    return {
        "solver_seed": seed_value, "final_unseeded_accuracy": accuracy, "success": accuracy >= threshold,
        "success_090": accuracy >= 0.9, "objective": out["sgm_objective_score"], "n_iter": out["sgm_n_iter"],
        "converged": out["sgm_converged"], "runtime_seconds": out["sgm_runtime_seconds"],
        "h0_accuracy": out["h0_accuracy"], "matched_fraction": np.nan, "precision": np.nan,
    }


# --------------------------------------------------------------------------
# One graph model
# --------------------------------------------------------------------------

def run_model(algorithm: str, model: str, phase: str, *, n_pairs: int, budgets: list[int],
              strategies: list[str], output_dir: Path, faq_restarts: int, blas_threads: int,
              overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / f"{name}.csv" for name in ("runs", "pairs", "orderings")}
    settings_path, metadata_path = output_dir / "settings.json", output_dir / "metadata.json"
    config = S.graph_config(algorithm, model)
    run_settings = {
        "algorithm": algorithm, "model": model, "phase": phase, "graph_family": S.GRAPH_FAMILY[algorithm],
        "graph_config": json.loads(json.dumps(asdict(config))),
        "budgets": sorted(int(b) for b in budgets), "strategies": list(strategies),
        "success_threshold": S.SUCCESS_THRESHOLD[algorithm],
        "solver": (S.PERCOLATION if algorithm == "percolation" else
                   {**S.SGM, "transport": algorithm == "goat",
                    "transport_regularizer": S.GOAT_REGULARIZER[model] if algorithm == "goat" else None,
                    "max_restarts": faq_restarts if algorithm == "faq" else 1,
                    "restart_rule": "stop at first success" if algorithm == "faq" else "single run"}),
        "pool_factor": S.POOL_FACTOR, "hub_count": S.HUB_COUNT,
        "seed_bases": {"pair": S.PAIR_SEED_BASE, "strategy": S.STRATEGY_SEED_BASE, "solver": S.SOLVER_SEED_BASE},
    }
    run_settings = json.loads(json.dumps(run_settings))
    if not budgets or min(budgets) < 1 or max(budgets) >= config.n_vertices:
        raise SystemExit("Budgets must be between 1 and n - 1.")
    if len(set(strategies)) != len(strategies):
        raise SystemExit("Each strategy may appear only once.")
    if overwrite:
        for path in [*paths.values(), settings_path, metadata_path]:
            if path.exists():
                path.unlink()
    if settings_path.exists():
        saved = json.loads(settings_path.read_text(encoding="utf-8"))
        if saved != run_settings:
            changed = sorted(k for k in run_settings if saved.get(k) != run_settings[k])
            raise SystemExit(f"{output_dir} holds results with different settings ({changed}). "
                             "Use a new --output-root, or --overwrite to restart this model.")
    else:
        write_json(settings_path, run_settings)

    done = set()
    if paths["pairs"].exists():
        with paths["pairs"].open(newline="", encoding="utf-8") as handle:
            done = {int(row["pair_index"]) for row in csv.DictReader(handle)}
    for name in ("runs", "orderings"):  # remove rows of a pair interrupted mid-write
        _keep_pairs(paths[name], done)

    preset_status = S.check_training_presets(algorithm, model)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {
        "started_at": _now(), "environment": collect_environment(blas_threads)}
    metadata.update({"status": "running", "preset_check": preset_status, "n_pairs_requested": n_pairs})
    write_json(metadata_path, metadata)
    print(f"[{algorithm}/{model}/{phase}] {preset_status}; {len(done)} of {n_pairs} pairs already done.",
          flush=True)

    budgets = sorted(int(b) for b in budgets)
    k_max = budgets[-1]
    started = time.perf_counter()
    newly_done = 0
    with thread_limit(blas_threads):
        for pair_index in range(n_pairs):
            if pair_index in done:
                continue
            pair_started = time.perf_counter()
            first_seed = pair_first_seed(algorithm, model, phase, pair_index)
            offset = 0
            while True:
                seed, adjacency_1, adjacency_2, truth, report, rejected = first_passing_pair(
                    lambda s: generate_graph_pair(config, s), first_seed + offset)
                if seed not in S.KNOWN_PAIR_SEEDS:
                    break
                offset = seed - first_seed + 1
            rejected += offset
            pair = (adjacency_1, adjacency_2, truth, seed)
            graph = GraphA(adjacency_1)
            aligned_b = adjacency_2[np.ix_(truth, truth)]

            orders = {}
            for strategy in strategies:
                rng = np.random.default_rng([S.STRATEGY_SEED_BASE, seed, STRATEGY_CODES[strategy]])
                orders[strategy] = ordering(strategy, graph, k_max, rng,
                                            pool_factor=S.POOL_FACTOR, hub_count=S.HUB_COUNT)

            rows, cache, feature_cache = [], {}, {}
            max_restarts = faq_restarts if algorithm == "faq" else 1
            for budget in budgets:
                for strategy in strategies:
                    seeds = np.sort(orders[strategy][:budget])
                    key = tuple(int(v) for v in seeds)
                    if key not in feature_cache:
                        feature_cache[key] = seed_set_features(graph, seeds)
                    for restart in range(max_restarts):
                        cached = (key, restart) in cache
                        if not cached:
                            cache[(key, restart)] = evaluate(algorithm, model, pair, seeds, budget, restart)
                        outcome = cache[(key, restart)]
                        rows.append({
                            "algorithm": algorithm, "model": model, "phase": phase, "pair_index": pair_index,
                            "pair_seed": seed, "strategy": strategy, "budget": budget, "restart": restart,
                            "seed_vertices": json.dumps(list(key)), "cached": cached,
                            **outcome, **feature_cache[key],
                        })
                        if outcome["success"]:
                            break

            pair_row = {
                "pair_index": pair_index, "pair_seed": seed, "first_seed_tried": first_seed,
                "rejected_draws": rejected, "mean_degree_a": float(graph.degree.mean()),
                "max_degree_a": int(graph.degree.max()),
                "mean_degree_b": float((adjacency_2 > 0).sum(axis=1).mean()),
                "kept_edge_fraction": float(((adjacency_1 > 0) & (aligned_b > 0)).sum() / (adjacency_1 > 0).sum()),
                "hash_a": _hash(adjacency_1), "hash_b": _hash(adjacency_2),
                "pair_seconds": round(time.perf_counter() - pair_started, 3),
            }
            _append(paths["runs"], RUN_COLUMNS, rows)
            _append(paths["orderings"], ("pair_index", "pair_seed", "strategy", "ordering"),
                    [{"pair_index": pair_index, "pair_seed": seed, "strategy": strategy,
                      "ordering": json.dumps([int(v) for v in orders[strategy]])} for strategy in strategies])
            _append(paths["pairs"], PAIR_COLUMNS, [pair_row])  # written last: marks the pair complete

            newly_done += 1
            elapsed = time.perf_counter() - started
            remaining = n_pairs - len(done) - newly_done
            eta = elapsed / newly_done * remaining
            print(f"[{algorithm}/{model}/{phase}] pair {pair_index + 1}/{n_pairs} in {pair_row['pair_seconds']:.1f}s; "
                  f"{sum(not r['cached'] for r in rows)} solver runs; elapsed {elapsed / 60:.1f} min; "
                  f"about {eta / 60:.1f} min left", flush=True)

    metadata.update({"status": "complete", "finished_at": _now(),
                     "minutes_this_session": round((time.perf_counter() - started) / 60, 2)})
    write_json(metadata_path, metadata)


def _keep_pairs(path: Path, done: set[int]) -> None:
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    kept = [row for row in rows if int(row["pair_index"]) in done]
    if len(kept) == len(rows):
        return
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    temporary.replace(path)


def _append(path: Path, columns, rows) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithm", default="all", choices=(*S.ALGORITHMS, "all"),
                        help="One algorithm, or 'all' to run percolation, GOAT and FAQ one after another.")
    parser.add_argument("--model", default="all", choices=(*S.MODELS, "all"),
                        help="One graph model, or 'all' for the three models.")
    parser.add_argument("--phase", default="main", choices=S.PHASES,
                        help="'pilot': random and top degree on a wide budget grid; 'main': all strategies.")
    parser.add_argument("--pairs", type=int, default=None,
                        help="Fresh graph pairs per model (default: 10 for the pilot, 20 for the main run).")
    parser.add_argument("--budgets", type=int, nargs="+", default=None,
                        help="Seed budgets for every selected model (default: the grid in settings.py).")
    parser.add_argument("--strategies", nargs="+", default=None, choices=tuple(STRATEGY_CODES))
    parser.add_argument("--jobs", type=int, default=3,
                        help="Graph models run in parallel (default 3; 1 runs them in this terminal).")
    parser.add_argument("--faq-restarts", type=int, default=S.FAQ_MAX_RESTARTS)
    parser.add_argument("--blas-threads", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-dir", type=Path, default=EXPERIMENT_DIR / "logs")
    parser.add_argument("--no-caffeinate", action="store_true",
                        help="Do not keep the Mac awake while running (macOS only).")
    parser.add_argument("--overwrite", action="store_true", help="Delete the selected results and restart them.")
    return parser.parse_args(argv)


def _model_job(job: dict) -> float:
    """Run one graph model in a worker process, writing its progress to a log file."""

    import contextlib
    import traceback

    started = time.perf_counter()
    kwargs = job["kwargs"]
    with open(job["log"], "a", encoding="utf-8") as log, contextlib.redirect_stdout(log), \
            contextlib.redirect_stderr(log):
        print(f"{_now()}  start {kwargs['algorithm']}/{kwargs['model']}/{kwargs['phase']}: "
              f"{kwargs['n_pairs']} pairs, budgets {' '.join(map(str, kwargs['budgets']))}", flush=True)
        try:
            run_model(**kwargs)
        except SystemExit as stop:  # report setting conflicts as ordinary errors
            print(f"{_now()}  stopped: {stop}", flush=True)
            raise RuntimeError(str(stop)) from None
        except Exception:
            traceback.print_exc()  # the full error goes to the log; the terminal gets one line
            raise
        print(f"{_now()}  done", flush=True)
    return (time.perf_counter() - started) / 60


def main(argv=None) -> None:
    args = parse_args(argv)
    algorithms = S.ALGORITHMS if args.algorithm == "all" else (args.algorithm,)
    models = S.MODELS if args.model == "all" else (args.model,)
    if sys.platform == "darwin" and not args.no_caffeinate:
        import shutil
        import subprocess
        if shutil.which("caffeinate"):  # keeps the Mac awake until this process exits
            subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])

    failures = []
    for algorithm in algorithms:
        jobs = []
        for model in models:
            kwargs = dict(
                algorithm=algorithm, model=model, phase=args.phase,
                n_pairs=args.pairs or S.N_PAIRS[args.phase][algorithm],
                budgets=args.budgets or list(S.default_budgets(args.phase, algorithm, model)),
                strategies=args.strategies or list(S.STRATEGIES[args.phase][algorithm]),
                output_dir=args.output_root / args.phase / algorithm / model,
                faq_restarts=args.faq_restarts, blas_threads=args.blas_threads, overwrite=args.overwrite,
            )
            jobs.append({"kwargs": kwargs, "log": args.log_dir / f"{args.phase}_{algorithm}_{model}.log"})

        if args.jobs <= 1 or len(jobs) == 1:
            for job in jobs:
                run_model(**job["kwargs"])
            continue

        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing

        args.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"{_now()}  {algorithm}: running {len(jobs)} graph models in parallel. Progress logs:", flush=True)
        for job in jobs:
            print(f"    tail -f {job['log']}", flush=True)
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs)), mp_context=context) as pool:
            futures = {pool.submit(_model_job, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                name = f"{algorithm}/{job['kwargs']['model']}"
                try:
                    minutes = future.result()
                    print(f"{_now()}  {name} finished in {minutes:.1f} min", flush=True)
                except Exception as error:  # noqa: BLE001 - report and continue with the other models
                    failures.append(name)
                    print(f"{_now()}  {name} FAILED: {error}  (see {job['log']})", flush=True)
    if failures:
        raise SystemExit(f"Failed: {', '.join(failures)}. Rerun the same command to resume.")
    print(f"{_now()}  all requested runs are complete.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
