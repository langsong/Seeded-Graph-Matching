"""Calibrate seed budgets for best-objective-of-R SGM rule comparisons.

This focused sweep keeps each graph model's density and correlation fixed,
varies only the number of uniformly random seeds, and evaluates multiple graph
pairs.  Within a pair and candidate ID, the seed sets are nested across budgets:
the first k vertices of one random permutation form the size-k seed set.  Thus
each budget is marginally uniform while nearby budgets receive a paired
comparison.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

# Numba tries to place compiled caches beside some third-party package files.
# Redirect those caches before importing graspologic through experiment_common.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_mpl_cache"))

import numpy as np

from experiment_common.graphs import generate_graph_pair
from experiment_common.provenance import thread_limit
from experiment_common.sgm import run_sgm_once
from fixed_pair_seed_set_classification.pairs.er_seed_selection import PROBLEM as ER
from fixed_pair_seed_set_classification.pairs.paper_pa_seed_selection import (
    PROBLEM as PAPER,
)


PROBLEMS = {"sparse_er": ER, "paper_pa": PAPER}
DEFAULT_BUDGETS = {
    "sparse_er": (24, 25, 26, 27, 28),
    "paper_pa": (11, 12, 13, 14, 15),
}
DEFAULT_PAIR_SEEDS = {
    "sparse_er": (4192889773, 18472911, 37051993),
    "paper_pa": (424242, 18472911, 37051993),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=PROBLEMS, required=True)
    parser.add_argument("--budgets", nargs="+", type=int)
    parser.add_argument("--pair-seeds", nargs="+", type=int)
    parser.add_argument("--seed-sets", type=int, default=24)
    parser.add_argument("--solver-runs", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--success-threshold", type=float, default=0.90)
    parser.add_argument("--random-seed", type=int, default=20261006)
    parser.add_argument("--blas-threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    budgets = tuple(sorted(set(args.budgets or DEFAULT_BUDGETS[args.model])))
    pair_seeds = tuple(args.pair_seeds or DEFAULT_PAIR_SEEDS[args.model])
    if not budgets or budgets[0] <= 0:
        raise ValueError("Budgets must be positive.")
    if args.seed_sets <= 0 or args.solver_runs <= 0 or args.max_iter <= 0:
        raise ValueError("Run counts and max_iter must be positive.")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    spec = PROBLEMS[args.model]
    config = spec.config
    fields = (
        "model", "pair_id", "graph_pair_seed", "mean_degree", "candidate_id",
        "n_seeds", "seed_vertices", "solver_run_id", "solver_seed",
        "max_iter", "success_threshold", "success", "h0_accuracy",
        "final_unseeded_accuracy", "sgm_n_iter", "sgm_converged",
        "sgm_objective_score", "sgm_runtime_seconds",
    )

    with (
        thread_limit(args.blas_threads or None),
        args.output.open("w", newline="", encoding="utf-8") as output_file,
    ):
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for pair_id, pair_seed in enumerate(pair_seeds):
            adjacency_1, adjacency_2, truth = generate_graph_pair(config, pair_seed)
            mean_degree = float(adjacency_1.sum(axis=1).mean())
            rng = np.random.default_rng(args.random_seed + pair_id)
            permutations = [rng.permutation(len(adjacency_1)) for _ in range(args.seed_sets)]
            for candidate_id, permutation in enumerate(permutations):
                for n_seeds in budgets:
                    seeds = np.sort(permutation[:n_seeds])
                    for solver_run_id in range(args.solver_runs):
                        solver_seed = spec.sgm_seed + solver_run_id
                        result = run_sgm_once(
                            adjacency_1, adjacency_2, truth, seeds, solver_seed,
                            args.max_iter, config.tol,
                        )
                        writer.writerow({
                            "model": args.model,
                            "pair_id": pair_id,
                            "graph_pair_seed": pair_seed,
                            "mean_degree": mean_degree,
                            "candidate_id": candidate_id,
                            "n_seeds": n_seeds,
                            "seed_vertices": "[" + ",".join(map(str, seeds)) + "]",
                            "solver_run_id": solver_run_id,
                            "solver_seed": solver_seed,
                            "max_iter": args.max_iter,
                            "success_threshold": args.success_threshold,
                            "success": result["final_unseeded_accuracy"] >= args.success_threshold,
                            **{key: result[key] for key in (
                                "h0_accuracy", "final_unseeded_accuracy", "sgm_n_iter",
                                "sgm_converged", "sgm_objective_score",
                                "sgm_runtime_seconds",
                            )},
                        })
                    output_file.flush()
            print(
                f"{args.model}: pair {pair_id + 1}/{len(pair_seeds)} "
                f"(seed={pair_seed}, mean degree={mean_degree:.2f})",
                flush=True,
            )
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
