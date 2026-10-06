"""Collect SGM outcomes for many seed sets on one fixed graph pair."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .pairs.er_seed_selection import PROBLEM as ER_PROBLEM
from .pairs.chung_lu_seed_selection import PROBLEM as CHUNG_LU_PROBLEM
from .pairs.enron_seed_selection import PROBLEM as ENRON_PROBLEM
from .pairs.ier_seed_selection import PROBLEM as IER_PROBLEM
from .pairs.pareto_chung_lu_seed_selection import PROBLEM as PAPER_PROBLEM
from .pairs.paper_pa_seed_selection import PROBLEM as PAPER_PA_PROBLEM
from .pairs.sbm_seed_selection import PROBLEM as SBM_PROBLEM
from .seed_selection_common import (
    DEFAULT_PROBLEM_DIR,
    FixedProblem,
    evaluate_seed_set,
    prepare_problem,
)
from .seed_set_sampling import sample_seed_sets
from experiment_common.features import (
    FIXED_PAIR_FEATURE_COLUMNS,
    FIXED_PAIR_FEATURE_SCHEMA_VERSION,
)
from experiment_common.sgm import run_sgm_once


PROBLEMS = {
    "sparse_er": ER_PROBLEM,
    "chung_lu": CHUNG_LU_PROBLEM,
    "sparse_sbm": SBM_PROBLEM,
    "paper": PAPER_PROBLEM,
    "paper_pa": PAPER_PA_PROBLEM,
    "ier": IER_PROBLEM,
    "enron": ENRON_PROBLEM,
}
FOCUS_PROBLEMS = ("sparse_er", "paper_pa", "chung_lu")

SGM_RESULT_COLUMNS = (
    "h0_accuracy",
    "final_unseeded_accuracy",
    "final_all_accuracy",
    "sgm_runtime_seconds",
    "sgm_n_iter",
    "sgm_converged",
    "sgm_objective_score",
)


def collect_problem(
    spec: FixedProblem,
    output_dir: Path,
    n_random_train: int,
    n_random_test: int,
    n_degree_weighted: int,
    n_swap_paths: int,
    n_decorrelated: int,
    pool_size: int,
    random_seed: int,
    success_threshold: float,
    overwrite: bool,
    max_iter: int = 100,
    solver_runs: int = 5,
    n_seeds: int | None = None,
) -> None:
    if max_iter <= 0 or solver_runs <= 0:
        raise ValueError("max_iter and solver_runs must be positive.")
    config = replace(
        spec.config,
        max_iter=max_iter,
        n_seeds=spec.config.n_seeds if n_seeds is None else n_seeds,
    )
    spec = replace(spec, config=config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{spec.name}_seed_sets.csv"
    runs_path = output_dir / f"{spec.name}_seed_sets_runs.csv"
    metadata_path = output_dir / f"{spec.name}_seed_sets.metadata.json"
    if not overwrite:
        for path in (output_path, runs_path, metadata_path):
            if path.exists():
                raise FileExistsError(f"{path} already exists; pass --overwrite.")

    problem = prepare_problem(spec, output_dir / "graph_pairs")
    rng = np.random.default_rng(random_seed)
    candidates = sample_seed_sets(
        problem.context_1, rng, spec.config.n_seeds,
        n_random_train=n_random_train,
        n_random_test=n_random_test,
        n_degree_weighted=n_degree_weighted,
        n_swap_paths=n_swap_paths,
        n_decorrelated=n_decorrelated,
        pool_size=pool_size,
    )
    feature_names = list(FIXED_PAIR_FEATURE_COLUMNS)
    solver_seeds = [spec.sgm_seed + index for index in range(solver_runs)]

    started = time.perf_counter()
    run_rows_written = 0
    with (
        output_path.open("w", newline="", encoding="utf-8") as output_file,
        runs_path.open("w", newline="", encoding="utf-8") as runs_file,
    ):
        summary_writer = None
        runs_writer = csv.DictWriter(
            runs_file,
            fieldnames=[
                "candidate_id", "data_split", "sampling_strategy", "problem",
                "starting_rule", "swap_count", "degree_gamma",
                "contrast_feature", "contrast_tail", "contrast_degree_bin",
                "graph_model", "sgm_run_id", "sgm_seed", "sgm_max_iter",
                "sgm_success", *SGM_RESULT_COLUMNS,
            ],
        )
        runs_writer.writeheader()
        for candidate_id, candidate in enumerate(candidates):
            seeds_1 = candidate.seeds
            sampling_fields = {
                "data_split": candidate.split,
                "sampling_strategy": candidate.strategy,
                "starting_rule": candidate.starting_rule,
                "swap_count": candidate.swap_count,
                "degree_gamma": candidate.degree_gamma,
                "contrast_feature": candidate.contrast_feature,
                "contrast_tail": candidate.contrast_tail,
                "contrast_degree_bin": candidate.contrast_degree_bin,
            }
            first = evaluate_seed_set(
                problem, seeds_1, sgm_seed=solver_seeds[0], max_iter=max_iter
            )
            run_results = []
            for run_id, sgm_seed in enumerate(solver_seeds):
                performance = (
                    {column: first[column] for column in SGM_RESULT_COLUMNS}
                    if run_id == 0
                    else run_sgm_once(
                        problem.adjacency_1,
                        problem.adjacency_2,
                        problem.true_permutation,
                        seeds_1,
                        sgm_seed,
                        max_iter,
                        config.tol,
                    )
                )
                run_results.append(performance)
                runs_writer.writerow({
                    "candidate_id": candidate_id,
                    **sampling_fields,
                    "problem": spec.name,
                    "graph_model": config.graph_model,
                    "sgm_run_id": run_id,
                    "sgm_seed": sgm_seed,
                    "sgm_max_iter": max_iter,
                    "sgm_success": bool(
                        performance["final_unseeded_accuracy"] >= success_threshold
                    ),
                    **performance,
                })
                run_rows_written += 1

            success_rate = np.mean([
                result["final_unseeded_accuracy"] >= success_threshold
                for result in run_results
            ])
            best_result = max(
                range(solver_runs),
                key=lambda index: run_results[index]["sgm_objective_score"],
            )
            row = {
                "candidate_id": candidate_id,
                **sampling_fields,
                **{key: value for key, value in first.items()
                   if key not in SGM_RESULT_COLUMNS},
                "sgm_seeds": json.dumps(solver_seeds),
                "sgm_run_count": solver_runs,
                "sgm_success_rate": float(success_rate),
                "sgm_success": bool(success_rate > 0.5),
                "sgm_converged_fraction": float(np.mean([
                    result["sgm_converged"] for result in run_results
                ])),
                "sgm_runtime_total_seconds": float(sum(
                    result["sgm_runtime_seconds"] for result in run_results
                )),
                "best_objective_final_unseeded_accuracy": run_results[
                    best_result
                ]["final_unseeded_accuracy"],
                "best_objective_sgm_seed": solver_seeds[best_result],
            }
            for column in SGM_RESULT_COLUMNS:
                row[column] = (
                    bool(all(result[column] for result in run_results))
                    if column == "sgm_converged"
                    else float(np.mean([result[column] for result in run_results]))
                )

            if summary_writer is None:
                summary_writer = csv.DictWriter(output_file, fieldnames=list(row))
                summary_writer.writeheader()
            summary_writer.writerow(row)

            completed = candidate_id + 1
            if completed % 25 == 0 or completed == len(candidates):
                elapsed = (time.perf_counter() - started) / 60
                print(
                    f"{spec.name}: {completed}/{len(candidates)} seed sets "
                    f"({elapsed:.1f} minutes)",
                    flush=True,
                )

    metadata = {
        "problem": spec.name,
        "graph_pair_seed": spec.graph_pair_seed,
        "sgm_seed": spec.sgm_seed,
        "solver_seeds": solver_seeds,
        "max_iter": max_iter,
        "solver_runs": solver_runs,
        "config": asdict(config),
        "success_threshold": success_threshold,
        "random_seed": random_seed,
        "n_random_train": n_random_train,
        "n_random_test": n_random_test,
        "n_degree_weighted": n_degree_weighted,
        "n_swap_paths": n_swap_paths,
        "n_decorrelated": n_decorrelated,
        "decorrelation_pool_size": pool_size,
        "rows": len(candidates),
        "run_rows": run_rows_written,
        "feature_columns": feature_names,
        "feature_schema_version": FIXED_PAIR_FEATURE_SCHEMA_VERSION,
        "target_columns": [
            "h0_accuracy",
            "final_unseeded_accuracy",
            "final_all_accuracy",
            "sgm_success_rate",
            "sgm_success",
        ],
        "predictor_scope": "Graph A seed-set features only",
        "output_csv": str(output_path),
        "runs_csv": str(runs_path),
        "summary_outcome": "Mean accuracy and H0; success rate over runs; majority-success label.",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_path}")
    print(f"Wrote {runs_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-model", choices=("focus", "all", *PROBLEMS), default="focus"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_PROBLEM_DIR / "sgm_pilot_400"
    )
    parser.add_argument("--random-train", type=int, default=80)
    parser.add_argument("--random-test", type=int, default=40)
    parser.add_argument("--degree-weighted", type=int, default=160)
    parser.add_argument("--swap-paths", type=int, default=80)
    parser.add_argument("--decorrelated", type=int, default=40)
    parser.add_argument("--pool-size", type=int, default=4000)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--success-threshold", type=float, default=0.90)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--solver-runs", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = (
        FOCUS_PROBLEMS if args.graph_model == "focus"
        else PROBLEMS if args.graph_model == "all"
        else (args.graph_model,)
    )
    for name in names:
        collect_problem(
            PROBLEMS[name],
            args.output_dir,
            args.random_train,
            args.random_test,
            args.degree_weighted,
            args.swap_paths,
            args.decorrelated,
            args.pool_size,
            args.random_seed,
            args.success_threshold,
            args.overwrite,
            args.max_iter,
            args.solver_runs,
            args.n_seeds,
        )


if __name__ == "__main__":
    main()
