"""Small fixed-pair calibration sweep for the longer SGM protocol.

Run as ``python -m fixed_pair_seed_set_classification.audits.calibrate_sgm_settings``.
Each row is one SGM run; no seed-set feature calculations or training occur.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiment_common.graphs import generate_graph_pair
from experiment_common.sampling import sample_unique_seed_sets
from experiment_common.sgm import run_sgm_once
from fixed_pair_seed_set_classification.pairs.chung_lu_seed_selection import (
    PROBLEM as CHUNG_LU,
)
from fixed_pair_seed_set_classification.pairs.er_seed_selection import PROBLEM as ER
from fixed_pair_seed_set_classification.pairs.paper_pa_seed_selection import (
    PROBLEM as PAPER,
)


SETTINGS = {
    "sparse_er": {
        "problem": ER,
        "probabilities": (0.008, 0.010, 0.012),
        "correlations": (0.72, 0.80, 0.86),
        "seed_counts": (10, 14, 18, 22),
    },
    "paper_pa": {
        "problem": PAPER,
        "probabilities": (0.003, 0.005, 0.007),
        "correlations": (0.50, 0.62, 0.70),
        "seed_counts": (8, 12, 16, 20),
    },
    "chung_lu": {
        "problem": CHUNG_LU,
        "probabilities": (0.003, 0.005, 0.007),
        "correlations": (0.60, 0.70, 0.78),
        "seed_counts": (12, 18, 24, 30),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SETTINGS, required=True)
    parser.add_argument("--phase", choices=("screen", "confirm"), required=True)
    parser.add_argument("--probability", type=float)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--n-seeds", type=int)
    parser.add_argument("--seed-sets", type=int, default=None)
    parser.add_argument("--solver-runs", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    setting = SETTINGS[args.model]
    spec = setting["problem"]
    if args.phase == "screen":
        combinations = [
            (p, rho, k)
            for p in setting["probabilities"]
            for rho in setting["correlations"]
            for k in setting["seed_counts"]
        ]
        n_seed_sets = 4 if args.seed_sets is None else args.seed_sets
        n_solver_runs = 1 if args.solver_runs is None else args.solver_runs
    else:
        combinations = [(args.probability, args.rho, args.n_seeds)]
        n_seed_sets = 20 if args.seed_sets is None else args.seed_sets
        n_solver_runs = 5 if args.solver_runs is None else args.solver_runs

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    fields = (
        "model", "phase", "probability", "rho", "n_seeds", "graph_pair_seed",
        "candidate_id", "solver_seed", "max_iter", "mean_degree", "success",
        "h0_accuracy", "final_unseeded_accuracy", "sgm_n_iter",
        "sgm_converged", "sgm_objective_score", "sgm_runtime_seconds",
    )
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        last_pair = None
        for setting_id, (probability, rho, n_seeds) in enumerate(combinations):
            pair_key = (probability, rho)
            if pair_key != last_pair:
                changes = {"rho": rho}
                if args.model == "sparse_er":
                    changes["er_probability"] = probability
                elif args.model == "paper_pa":
                    changes["paper_pa_probability"] = probability
                else:
                    changes["chung_lu_probability_floor"] = probability
                config = replace(spec.config, **changes)
                A, B, truth = generate_graph_pair(config, spec.graph_pair_seed)
                mean_degree = float(A.sum(axis=1).mean())
                last_pair = pair_key

            rng = np.random.default_rng(20260930 + setting_id)
            seed_sets = sample_unique_seed_sets(
                rng, len(A), n_seeds, n_seed_sets
            )
            for candidate_id, seeds in enumerate(seed_sets):
                for solver_seed in range(2026, 2026 + n_solver_runs):
                    result = run_sgm_once(
                        A, B, truth, seeds, solver_seed, args.max_iter, config.tol
                    )
                    writer.writerow({
                        "model": args.model,
                        "phase": args.phase,
                        "probability": probability,
                        "rho": rho,
                        "n_seeds": n_seeds,
                        "graph_pair_seed": spec.graph_pair_seed,
                        "candidate_id": candidate_id,
                        "solver_seed": solver_seed,
                        "max_iter": args.max_iter,
                        "mean_degree": mean_degree,
                        "success": result["final_unseeded_accuracy"] >= 0.90,
                        **{key: result[key] for key in (
                            "h0_accuracy", "final_unseeded_accuracy", "sgm_n_iter",
                            "sgm_converged", "sgm_objective_score",
                            "sgm_runtime_seconds",
                        )},
                    })
            file.flush()
            print(
                f"{args.model} {setting_id + 1}/{len(combinations)}: "
                f"p={probability}, rho={rho}, k={n_seeds}",
                flush=True,
            )
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
