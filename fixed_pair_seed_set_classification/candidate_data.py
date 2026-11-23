"""Collect SGM outcomes for many seed sets on one or more fixed graph pairs."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .pairs.er_seed_selection import PROBLEM as ER_PROBLEM
from .pairs.er_seed_selection import GOAT_PROBLEM as ER_GOAT_PROBLEM
from .pairs.er_seed_selection import GOAT_REGULARIZER as ER_GOAT_REGULARIZER
from .pairs.chung_lu_seed_selection import PROBLEM as CHUNG_LU_PROBLEM
from .pairs.chung_lu_seed_selection import GOAT_PROBLEM as CHUNG_LU_GOAT_PROBLEM
from .pairs.chung_lu_seed_selection import GOAT_REGULARIZER as CHUNG_LU_GOAT_REGULARIZER
from .pairs.enron_seed_selection import PROBLEM as ENRON_PROBLEM
from .pairs.ier_seed_selection import PROBLEM as IER_PROBLEM
from .pairs.pareto_chung_lu_seed_selection import PROBLEM as PAPER_PROBLEM
from .pairs.paper_pa_seed_selection import PROBLEM as PAPER_PA_PROBLEM
from .pairs.paper_pa_seed_selection import GOAT_PROBLEM as PAPER_PA_GOAT_PROBLEM
from .pairs.paper_pa_seed_selection import GOAT_REGULARIZER as PAPER_PA_GOAT_REGULARIZER
from .pairs.sbm_seed_selection import PROBLEM as SBM_PROBLEM
from .seed_selection_common import (
    SGM_COLLECTION_DIR,
    FixedProblem,
    evaluate_seed_set,
    prepare_problem,
)
from .seed_set_sampling import (
    CONTRASTS,
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
from experiment_common.graph_checks import check_graph_pair
from experiment_common.sgm import SGM_DIAGNOSTIC_COLUMNS, run_sgm_once


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
GOAT_PROBLEMS = {
    "sparse_er": ER_GOAT_PROBLEM,
    "paper_pa": PAPER_PA_GOAT_PROBLEM,
    "chung_lu": CHUNG_LU_GOAT_PROBLEM,
}
GOAT_REGULARIZERS = {
    "sparse_er": ER_GOAT_REGULARIZER,
    "paper_pa": PAPER_PA_GOAT_REGULARIZER,
    "chung_lu": CHUNG_LU_GOAT_REGULARIZER,
}
DEFAULT_COLLECTION_DIR = SGM_COLLECTION_DIR
# Local SGM defaults: do not alter the sampler shared with percolation.
DEFAULT_SAMPLING_COUNTS = {
    "n_random_train": 400,
    "n_random_test": 200,
    "n_degree_weighted": 200,
    "n_swap_paths": 0,
    "n_decorrelated": 200,
}

SGM_RESULT_COLUMNS = (
    "h0_accuracy",
    "final_unseeded_accuracy",
    "final_all_accuracy",
    "sgm_runtime_seconds",
    "sgm_n_iter",
    "sgm_converged",
    "sgm_objective_score",
)
RUN_RESULT_COLUMNS = (*SGM_RESULT_COLUMNS, *SGM_DIAGNOSTIC_COLUMNS)
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
    solver_runs: int = 1,
    n_seeds: int | None = None,
    blas_threads: int | None = 4,
    solver: str = "faq",
    transport_regularizer: float | None = None,
    transport_tol: float = 0.05,
    transport_max_iter: int = 5000,
    problem_dir: Path | None = None,
) -> None:
    """Sample seed sets for one pair and run SGM ``solver_runs`` times on each.

    ``sampling`` holds the keyword arguments of ``sample_seed_sets`` (group
    counts, pool size and contrast options).
    """

    if solver not in ("faq", "goat"):
        raise ValueError("solver must be 'faq' or 'goat'.")
    if max_iter <= 0 or solver_runs <= 0:
        raise ValueError("max_iter and solver_runs must be positive.")
    if not 0 <= success_threshold <= 1:
        raise ValueError("success_threshold must be in [0, 1].")
    regularizer = (
        GOAT_REGULARIZERS.get(spec.name, 100)
        if transport_regularizer is None else transport_regularizer
    )
    if regularizer <= 0 or transport_tol <= 0 or transport_max_iter <= 0:
        raise ValueError("Transport regularizer, tolerance and iteration limit must be positive.")
    solver_options = {
        "transport": solver == "goat",
        "transport_regularizer": regularizer,
        "transport_tol": transport_tol,
        "transport_max_iter": transport_max_iter,
        "return_diagnostics": True,
    }
    config = replace(
        spec.config, max_iter=max_iter,
        n_seeds=spec.config.n_seeds if n_seeds is None else n_seeds,
    )
    spec = replace(spec, config=config)
    if not 0 < config.n_seeds < config.n_vertices:
        raise ValueError("n_seeds must be between 1 and n_vertices - 1.")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{spec.name}_seed_sets.csv"
    runs_path = output_dir / f"{spec.name}_seed_sets_runs.csv"
    metadata_path = output_dir / f"{spec.name}_seed_sets.metadata.json"
    if not overwrite:
        for path in (output_path, runs_path, metadata_path):
            if path.exists():
                raise FileExistsError(f"{path} already exists; pass --overwrite.")

    problem_dir = output_dir / "graph_pairs" if problem_dir is None else problem_dir
    problem = prepare_problem(spec, problem_dir)
    structure = check_graph_pair(problem.adjacency_1, problem.adjacency_2)
    if spec.name in FOCUS_PROBLEMS and not structure["passes"]:
        raise ValueError(
            f"Graph pair seed {spec.graph_pair_seed} fails connectivity/asymmetry certification. "
            "Choose another --pair-seeds value; the supplied seed will not be silently replaced. "
            f"Checks: {structure}"
        )
    sampling_info: dict = {}
    candidates = sample_seed_sets(
        problem.context_1, np.random.default_rng(random_seed), config.n_seeds,
        **sampling, sampling_info=sampling_info,
    )
    if not candidates:
        raise ValueError("Choose at least one seed set to collect.")
    signature_cap = signature_cap_from_formula(problem.context_1.degrees)

    metadata = {
        "status": "running",
        "started_at": _now(),
        "problem": spec.name,
        "graph_pair_seed": spec.graph_pair_seed,
        "sgm_seed": spec.sgm_seed,
        "solver": solver,
        "algorithm": "sgm_goat" if solver == "goat" else "sgm_faq",
        "n_init": 1,
        "solver_options": {key: value for key, value in solver_options.items() if key != "return_diagnostics"},
        "rng_policy": "SeedSequence([sgm_seed_base, sampling_seed, graph_pair_seed, candidate_id, run_id])",
        "label_definition": (
            "single_run_final_unseeded_accuracy >= success_threshold" if solver_runs == 1
            else "majority_of_explicit_single_initialization_runs"
        ),
        "h0_accuracy_definition": (
            "deterministic_one_to_one_projection_of_first_soft_direction" if solver == "goat"
            else "accuracy_of_first_Frank_Wolfe_assignment"
        ),
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
        "signature_panel_cap": 3,
        "graph_pair_checks": structure,
        "graph_pair_dir": str(problem_dir.resolve()),
        "graph_pair_dir_relative": os.path.relpath(problem_dir.resolve(), output_dir.resolve()),
        "problem_file": str((problem_dir / f"{spec.name}_problem.npz").resolve()),
        "predictor_scope": "Graph A seed-set features only",
        "output_csv": str(output_path),
        "runs_csv": str(runs_path),
        "warning_runs": 0,
        "nonconverged_runs": 0,
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
            "solver",
            "sgm_run_id", "sgm_seed", "sgm_max_iter", "sgm_success",
            *RUN_RESULT_COLUMNS,
        ])
        runs_writer.writeheader()
        for candidate_id, candidate in enumerate(candidates):
            sampling_fields = {
                "data_split": candidate.split,
                "sampling_strategy": candidate.strategy,
                **{column: getattr(candidate, column) for column in SAMPLING_COLUMNS[2:]},
            }
            solver_seeds = [
                int(np.random.default_rng(np.random.SeedSequence([
                    spec.sgm_seed, random_seed, spec.graph_pair_seed, candidate_id, run_id,
                ])).integers(np.iinfo(np.uint32).max))
                for run_id in range(solver_runs)
            ]
            # The first run comes with the Graph-A features; the rest are SGM only.
            first = evaluate_seed_set(
                problem, candidate.seeds, sgm_seed=solver_seeds[0], max_iter=max_iter,
                **solver_options,
            )
            results = [{column: first[column] for column in RUN_RESULT_COLUMNS}] + [
                run_sgm_once(
                    problem.adjacency_1, problem.adjacency_2, problem.true_permutation,
                    candidate.seeds, sgm_seed, max_iter, config.tol, **solver_options,
                )
                for sgm_seed in solver_seeds[1:]
            ]
            successes = [r["final_unseeded_accuracy"] >= success_threshold for r in results]
            metadata["warning_runs"] += sum(int(r["sgm_warning_count"] > 0) for r in results)
            metadata["nonconverged_runs"] += sum(int(not r["sgm_converged"]) for r in results)
            for run_id, (sgm_seed, result, success) in enumerate(
                zip(solver_seeds, results, successes)
            ):
                runs_writer.writerow({
                    "candidate_id": candidate_id, **sampling_fields,
                    "problem": spec.name, "graph_model": config.graph_model,
                    "solver": solver,
                    "sgm_run_id": run_id, "sgm_seed": sgm_seed,
                    "sgm_max_iter": max_iter, "sgm_success": success, **result,
                })

            best = max(range(solver_runs), key=lambda i: results[i]["sgm_objective_score"])
            row = {
                "candidate_id": candidate_id,
                "solver": solver,
                **sampling_fields,
                **{key: value for key, value in first.items() if key not in RUN_RESULT_COLUMNS},
                "sgm_seeds": json.dumps(solver_seeds),
                "sgm_run_count": solver_runs,
                "sgm_success_count": int(sum(successes)),
                "sgm_success_rate": float(np.mean(successes)),
                "sgm_success": bool(successes[0]) if solver_runs == 1 else bool(np.mean(successes) > 0.5),
                "sgm_converged_fraction": float(np.mean([r["sgm_converged"] for r in results])),
                "sgm_runtime_total_seconds": float(sum(r["sgm_runtime_seconds"] for r in results)),
                "best_objective_final_unseeded_accuracy": results[best]["final_unseeded_accuracy"],
                "best_objective_success": bool(successes[best]),
                "best_objective_sgm_seed": solver_seeds[best],
            }
            for column in SGM_RESULT_COLUMNS:
                values = [r[column] for r in results]
                row[column] = bool(all(values)) if column == "sgm_converged" else float(np.mean(values))
            for column in SGM_DIAGNOSTIC_COLUMNS:
                values = [r[column] for r in results]
                if column == "sgm_warnings":
                    row[column] = json.dumps([message for value in values for message in json.loads(value)])
                elif column == "sgm_warning_count":
                    row[column] = int(sum(values))
                elif column == "sgm_finite_soft_solution":
                    row[column] = bool(all(values))
                else:
                    row[column] = float(np.mean(values))
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
                    f"{spec.name}/{solver}: {completed}/{len(candidates)} seed sets "
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
    print(f"Solver checks: {metadata['warning_runs']} runs with warnings; "
          f"{metadata['nonconverged_runs']} runs reached the outer iteration limit.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-model", choices=("focus", "all", *PROBLEMS), default="focus")
    parser.add_argument("--solver", choices=("faq", "goat", "both"), default="both",
                        help="Solver preset(s) to collect. Both use the same graph pair, with their own seed budgets.")
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
    parser.add_argument("--solver-runs", type=int, default=1,
                        help="Explicit n_init=1 runs per set (default 1; increase only for a repeat audit).")
    parser.add_argument("--transport-regularizer", type=float, default=None,
                        help="Optional GOAT override; defaults are ER=100, PAPER=300, Chung-Lu=300.")
    parser.add_argument("--transport-tol", type=float, default=0.05)
    parser.add_argument("--transport-max-iter", type=int, default=5000)
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
    solvers = ("faq", "goat") if args.solver == "both" else (args.solver,)
    if "goat" in solvers and any(name not in GOAT_PROBLEMS for name in names):
        raise ValueError("GOAT presets exist only for sparse_er, paper_pa and chung_lu. Use --solver faq for historical models.")
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
        pair_seeds = args.pair_seeds or [PROBLEMS[name].graph_pair_seed]
        for pair_index, pair_seed in enumerate(pair_seeds, start=1):
            shared_pair_dir = args.output_dir / "graph_pairs" / name / f"pair_seed_{pair_seed}"
            for solver in solvers:
                base_spec = GOAT_PROBLEMS[name] if solver == "goat" else PROBLEMS[name]
                spec = replace(base_spec, graph_pair_seed=pair_seed)
                output_dir = args.output_dir / solver / name / f"pair_{pair_index}_seed_{pair_seed}"
                collect_problem(
                    spec, output_dir, sampling=sampling,
                    random_seed=args.random_seed, success_threshold=args.success_threshold,
                    overwrite=args.overwrite, max_iter=args.max_iter,
                    solver_runs=args.solver_runs, n_seeds=args.n_seeds,
                    blas_threads=args.blas_threads or None, solver=solver,
                    transport_regularizer=args.transport_regularizer,
                    transport_tol=args.transport_tol, transport_max_iter=args.transport_max_iter,
                    problem_dir=shared_pair_dir,
                )


if __name__ == "__main__":
    main()
