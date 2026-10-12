"""Collect SGM outcomes for many seed sets on one or more fixed graph pairs."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import time
import warnings
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
from .seed_set_sampling import (
    CONTRASTS,
    DEFAULT_SAMPLING_COUNTS,
    SAMPLER_VERSION,
    sample_seed_sets,
)
from experiment_common.features import (
    FIXED_PAIR_FEATURE_COLUMNS,
    FIXED_PAIR_FEATURE_SCHEMA_VERSION,
    SIGNATURE_ROBUSTNESS_COLUMNS,
    signature_cap_from_formula,
)
from experiment_common.provenance import collect_environment, thread_limit, write_json
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
DEFAULT_COLLECTION_DIR = DEFAULT_PROBLEM_DIR / "sgm_balanced_2026-10-07"

SGM_RESULT_COLUMNS = (
    "h0_accuracy",
    "final_unseeded_accuracy",
    "final_all_accuracy",
    "sgm_runtime_seconds",
    "sgm_n_iter",
    "sgm_converged",
    "sgm_objective_score",
)
SAMPLING_COLUMNS = (
    "data_split", "sampling_strategy", "starting_rule", "swap_count",
    "degree_gamma", "contrast_feature", "contrast_tail", "contrast_degree_bin",
    "contrast_pair_id",
)


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def collect_problem(
    spec: FixedProblem,
    output_dir: Path,
    *,
    sampling: dict,
    random_seed: int = 1234,
    success_threshold: float = 0.90,
    overwrite: bool = False,
    max_iter: int = 100,
    solver_runs: int = 5,
    n_seeds: int | None = None,
    blas_threads: int | None = 4,
) -> None:
    """Sample seed sets for one pair and run SGM ``solver_runs`` times on each.

    ``sampling`` holds the keyword arguments of ``sample_seed_sets`` (group
    counts, pool size and contrast options).
    """

    if max_iter <= 0 or solver_runs <= 0:
        raise ValueError("max_iter and solver_runs must be positive.")
    config = replace(
        spec.config, max_iter=max_iter,
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
    sampling_info: dict = {}
    candidates = sample_seed_sets(
        problem.context_1, np.random.default_rng(random_seed), config.n_seeds,
        **sampling, sampling_info=sampling_info,
    )
    solver_seeds = [spec.sgm_seed + index for index in range(solver_runs)]
    signature_cap = signature_cap_from_formula(problem.context_1.degrees)
    if signature_cap != 3:
        warnings.warn(
            f"log(n)/log(mean degree) gives signature cap {signature_cap}, not the "
            "panel's 3; check the cap-2 and cap-4 robustness columns.",
            stacklevel=2,
        )

    metadata = {
        "status": "running",
        "started_at": _now(),
        "problem": spec.name,
        "graph_pair_seed": spec.graph_pair_seed,
        "sgm_seed": spec.sgm_seed,
        "solver_seeds": solver_seeds,
        "max_iter": max_iter,
        "solver_runs": solver_runs,
        "config": asdict(config),
        "success_threshold": success_threshold,
        "random_seed": random_seed,
        "sampler_version": SAMPLER_VERSION,
        "sampling": {**sampling, **sampling_info},
        "rows": len(candidates),
        "rows_completed": 0,
        "feature_columns": list(FIXED_PAIR_FEATURE_COLUMNS),
        "feature_schema_version": FIXED_PAIR_FEATURE_SCHEMA_VERSION,
        "robustness_feature_columns": list(SIGNATURE_ROBUSTNESS_COLUMNS),
        "signature_cap_from_formula": signature_cap,
        "predictor_scope": "Graph A seed-set features only",
        "output_csv": str(output_path),
        "runs_csv": str(runs_path),
    }

    started = time.perf_counter()
    with (
        thread_limit(blas_threads),
        output_path.open("w", newline="", encoding="utf-8") as output_file,
        runs_path.open("w", newline="", encoding="utf-8") as runs_file,
    ):
        # Recorded inside the thread limit so it shows the counts actually used,
        # and written now so an interrupted collection keeps its settings.
        metadata["environment"] = collect_environment(blas_threads)
        write_json(metadata_path, metadata)
        summary_writer = None
        runs_writer = csv.DictWriter(runs_file, fieldnames=[
            "candidate_id", *SAMPLING_COLUMNS, "problem", "graph_model",
            "sgm_run_id", "sgm_seed", "sgm_max_iter", "sgm_success",
            *SGM_RESULT_COLUMNS,
        ])
        runs_writer.writeheader()
        for candidate_id, candidate in enumerate(candidates):
            sampling_fields = {
                "data_split": candidate.split,
                "sampling_strategy": candidate.strategy,
                **{column: getattr(candidate, column) for column in SAMPLING_COLUMNS[2:]},
            }
            # The first run comes with the Graph-A features; the rest are SGM only.
            first = evaluate_seed_set(
                problem, candidate.seeds, sgm_seed=solver_seeds[0], max_iter=max_iter
            )
            results = [{column: first[column] for column in SGM_RESULT_COLUMNS}] + [
                run_sgm_once(
                    problem.adjacency_1, problem.adjacency_2, problem.true_permutation,
                    candidate.seeds, sgm_seed, max_iter, config.tol,
                )
                for sgm_seed in solver_seeds[1:]
            ]
            successes = [r["final_unseeded_accuracy"] >= success_threshold for r in results]
            for run_id, (sgm_seed, result, success) in enumerate(
                zip(solver_seeds, results, successes)
            ):
                runs_writer.writerow({
                    "candidate_id": candidate_id, **sampling_fields,
                    "problem": spec.name, "graph_model": config.graph_model,
                    "sgm_run_id": run_id, "sgm_seed": sgm_seed,
                    "sgm_max_iter": max_iter, "sgm_success": success, **result,
                })

            best = max(range(solver_runs), key=lambda i: results[i]["sgm_objective_score"])
            row = {
                "candidate_id": candidate_id,
                **sampling_fields,
                **{key: value for key, value in first.items() if key not in SGM_RESULT_COLUMNS},
                "sgm_seeds": json.dumps(solver_seeds),
                "sgm_run_count": solver_runs,
                "sgm_success_count": int(sum(successes)),
                "sgm_success_rate": float(np.mean(successes)),
                "sgm_success": bool(np.mean(successes) > 0.5),  # historical majority label
                "sgm_converged_fraction": float(np.mean([r["sgm_converged"] for r in results])),
                "sgm_runtime_total_seconds": float(sum(r["sgm_runtime_seconds"] for r in results)),
                "best_objective_final_unseeded_accuracy": results[best]["final_unseeded_accuracy"],
                "best_objective_success": bool(successes[best]),
                "best_objective_sgm_seed": solver_seeds[best],
            }
            for column in SGM_RESULT_COLUMNS:
                values = [r[column] for r in results]
                row[column] = bool(all(values)) if column == "sgm_converged" else float(np.mean(values))
            if summary_writer is None:
                summary_writer = csv.DictWriter(output_file, fieldnames=list(row))
                summary_writer.writeheader()
            summary_writer.writerow(row)
            # Flush every seed set so an interrupted run keeps its finished rows.
            output_file.flush()
            runs_file.flush()

            completed = candidate_id + 1
            if completed % 25 == 0 or completed == len(candidates):
                metadata["rows_completed"] = completed
                write_json(metadata_path, metadata)
                print(
                    f"{spec.name}: {completed}/{len(candidates)} seed sets "
                    f"({(time.perf_counter() - started) / 60:.1f} minutes)",
                    flush=True,
                )

    metadata.update({
        "status": "complete",
        "finished_at": _now(),
        "run_rows": len(candidates) * solver_runs,
        "collection_minutes": round((time.perf_counter() - started) / 60, 2),
    })
    write_json(metadata_path, metadata)
    print(f"Wrote {output_path}\nWrote {runs_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-model", choices=("focus", "all", *PROBLEMS), default="focus")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_COLLECTION_DIR)
    parser.add_argument(
        "--pair-seeds", nargs="+", type=int, default=None,
        help=(
            "Optional graph-pair generation seeds. With multiple seeds, each "
            "pair is written to its own numbered subfolder."
        ),
    )
    parser.add_argument(
        "--random-train", type=int,
        default=DEFAULT_SAMPLING_COUNTS["n_random_train"],
    )
    parser.add_argument(
        "--random-test", type=int,
        default=DEFAULT_SAMPLING_COUNTS["n_random_test"],
    )
    parser.add_argument(
        "--degree-weighted", type=int,
        default=DEFAULT_SAMPLING_COUNTS["n_degree_weighted"],
    )
    parser.add_argument(
        "--swap-paths", type=int,
        default=DEFAULT_SAMPLING_COUNTS["n_swap_paths"],
        help="Optional swap-path training sets (default: 0).",
    )
    parser.add_argument(
        "--decorrelated", type=int,
        default=DEFAULT_SAMPLING_COUNTS["n_decorrelated"],
        help="Number of degree-controlled contrast sets (even: two per pair).",
    )
    parser.add_argument("--pool-size", type=int, default=4000)
    parser.add_argument("--contrast-features", nargs="+", choices=CONTRASTS, default=list(CONTRASTS),
                        help="Features that get contrast pairs; repeat a name to give it more pairs.")
    parser.add_argument("--contrast-bins", type=int, default=6)
    parser.add_argument("--contrast-percentiles", type=float, nargs=2, default=[10.0, 90.0],
                        metavar=("LOW", "HIGH"),
                        help="Degree range for contrasts, as percentiles of the random sets.")
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--success-threshold", type=float, default=0.90)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--solver-runs", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--blas-threads", type=int, default=4,
                        help="BLAS/OpenMP thread limit during collection (0 = none). Calibration used 4.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = (
        FOCUS_PROBLEMS if args.graph_model == "focus"
        else PROBLEMS if args.graph_model == "all"
        else (args.graph_model,)
    )
    sampling = {
        "n_random_train": args.random_train,
        "n_random_test": args.random_test,
        "n_degree_weighted": args.degree_weighted,
        "n_swap_paths": args.swap_paths,
        "n_decorrelated": args.decorrelated,
        "pool_size": args.pool_size,
        "contrast_features": tuple(args.contrast_features),
        "n_contrast_bins": args.contrast_bins,
        "contrast_degree_percentiles": tuple(args.contrast_percentiles),
    }
    for name in names:
        base_spec = PROBLEMS[name]
        pair_seeds = args.pair_seeds or [base_spec.graph_pair_seed]
        for pair_index, pair_seed in enumerate(pair_seeds, start=1):
            spec = replace(base_spec, graph_pair_seed=pair_seed)
            output_dir = args.output_dir
            if args.pair_seeds is not None:
                output_dir = (
                    output_dir / name / f"pair_{pair_index}_seed_{pair_seed}"
                )
            collect_problem(
                spec, output_dir, sampling=sampling,
                random_seed=args.random_seed, success_threshold=args.success_threshold,
                overwrite=args.overwrite, max_iter=args.max_iter,
                solver_runs=args.solver_runs, n_seeds=args.n_seeds,
                blas_threads=args.blas_threads or None,
            )


if __name__ == "__main__":
    main()
