"""Quick seed-budget calibration for the floor-free Chung-Lu generator.

The SGM regime uses a sparse mean-degree target of 5.7 and five solver runs;
the percolation regime uses mean degree 12 and one tie-break run. Candidate
sets are nested across seed budgets, making comparisons within a pair paired.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import replace
from pathlib import Path

# Third-party Numba caches must be writable before graspologic is imported.
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache")
)
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_mpl_cache")
)

import numpy as np

from cross_pair_seed_set_quality.percolation_data_collection import (
    run_percolation_once,
)
from experiment_common.graphs import generate_graph_pair, preset_config
from experiment_common.provenance import thread_limit
from experiment_common.sgm import run_sgm_once


PAIR_SEEDS = (3165569293, 18472911, 37051993)
SPARSE_WEIGHT_SCALE = 1.9260087733615983  # expected mean degree 5.7
DENSE_WEIGHT_SCALE = 4.076194702720809  # expected mean degree 12.0
DEFAULT_BUDGETS = {
    "sgm": (16, 19, 22, 25, 28),
    "percolation": (3, 4, 5, 6, 7, 8),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=DEFAULT_BUDGETS, required=True)
    parser.add_argument("--budgets", nargs="+", type=int)
    parser.add_argument("--pair-seeds", nargs="+", type=int, default=PAIR_SEEDS)
    parser.add_argument("--seed-sets", type=int, default=8)
    parser.add_argument("--solver-runs", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--success-threshold", type=float, default=0.90)
    parser.add_argument("--random-seed", type=int, default=20261007)
    parser.add_argument("--blas-threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    budgets = tuple(sorted(set(args.budgets or DEFAULT_BUDGETS[args.algorithm])))
    solver_runs = (
        5 if args.algorithm == "sgm" else 1
    ) if args.solver_runs is None else args.solver_runs
    if args.algorithm == "percolation" and solver_runs != 1:
        raise ValueError("Percolation calibration uses exactly one run per seed set.")
    if not budgets or budgets[0] <= 0 or args.seed_sets <= 0 or solver_runs <= 0:
        raise ValueError("Budgets and run counts must be positive.")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    weight_scale = (
        SPARSE_WEIGHT_SCALE if args.algorithm == "sgm" else DENSE_WEIGHT_SCALE
    )
    config = replace(
        preset_config("chung_lu"),
        chung_lu_alpha=2.0,
        chung_lu_weight_scale=weight_scale,
        chung_lu_probability_floor=0.0,
        rho=0.70,
        max_iter=args.max_iter,
    )
    fields = (
        "algorithm", "pair_id", "graph_pair_seed", "mean_degree",
        "weight_scale", "candidate_id", "n_seeds", "seed_vertices",
        "run_id", "algorithm_seed", "success_threshold", "success",
        "final_unseeded_accuracy", "objective_score", "runtime_seconds",
    )

    with (
        thread_limit(args.blas_threads or None),
        args.output.open("w", newline="", encoding="utf-8") as output_file,
    ):
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for pair_id, pair_seed in enumerate(args.pair_seeds):
            adjacency_1, adjacency_2, truth = generate_graph_pair(config, pair_seed)
            mean_degree = float(adjacency_1.sum(axis=1).mean())
            rng = np.random.default_rng(args.random_seed + pair_id)
            permutations = [
                rng.permutation(config.n_vertices) for _ in range(args.seed_sets)
            ]
            for candidate_id, permutation in enumerate(permutations):
                for n_seeds in budgets:
                    seeds = np.sort(permutation[:n_seeds])
                    for run_id in range(solver_runs):
                        algorithm_seed = 2026 + run_id
                        if args.algorithm == "sgm":
                            result = run_sgm_once(
                                adjacency_1, adjacency_2, truth, seeds,
                                algorithm_seed, args.max_iter, config.tol,
                            )
                            objective = result["sgm_objective_score"]
                            runtime = result["sgm_runtime_seconds"]
                        else:
                            # Each candidate gets a reproducible tie-break seed;
                            # the same seed is reused across its nested budgets.
                            algorithm_seed = 2026 + pair_id * 10000 + candidate_id
                            result = run_percolation_once(
                                adjacency_1, adjacency_2, truth, seeds,
                                algorithm_seed, threshold=2,
                                expand_when_stuck=True,
                            )
                            objective = ""
                            runtime = result["percolation_runtime_seconds"]
                        accuracy = float(result["final_unseeded_accuracy"])
                        writer.writerow({
                            "algorithm": args.algorithm,
                            "pair_id": pair_id,
                            "graph_pair_seed": pair_seed,
                            "mean_degree": mean_degree,
                            "weight_scale": weight_scale,
                            "candidate_id": candidate_id,
                            "n_seeds": n_seeds,
                            "seed_vertices": "[" + ",".join(map(str, seeds)) + "]",
                            "run_id": run_id,
                            "algorithm_seed": algorithm_seed,
                            "success_threshold": args.success_threshold,
                            "success": accuracy >= args.success_threshold,
                            "final_unseeded_accuracy": accuracy,
                            "objective_score": objective,
                            "runtime_seconds": runtime,
                        })
                    output_file.flush()
            print(
                f"{args.algorithm}: pair {pair_id + 1}/{len(args.pair_seeds)} "
                f"(seed={pair_seed}, mean degree={mean_degree:.2f})",
                flush=True,
            )
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
