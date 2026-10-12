"""Collect percolation outcomes for seed sets on fixed graph pairs."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache")
)
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache")
)

from experiment_common.features import (
    FIXED_PAIR_FEATURE_COLUMNS,
    FIXED_PAIR_FEATURE_SCHEMA_VERSION,
    SIGNATURE_ROBUSTNESS_COLUMNS,
    fixed_pair_seed_set_features,
    signature_cap_from_formula,
    signature_robustness_features,
)
from experiment_common.graphs import preset_config
from experiment_common.percolation import run_percolation_once
from experiment_common.provenance import collect_environment, write_json

from .seed_selection_common import DEFAULT_PROBLEM_DIR, FixedProblem, prepare_problem
from .seed_set_sampling import SAMPLER_VERSION, sample_seed_sets


DEFAULT_COLLECTION_DIR = DEFAULT_PROBLEM_DIR / "percolation_fixed_pairs_2026-10-07"
FOCUS_PROBLEMS = ("sparse_er", "paper_pa", "chung_lu")
DEFAULT_SAMPLING_COUNTS = {
    "n_random_train": 400,
    "n_random_test": 200,
    "n_degree_weighted": 200,
    "n_swap_paths": 0,
    "n_decorrelated": 200,
}
SAMPLING_COLUMNS = (
    "data_split",
    "sampling_strategy",
    "starting_rule",
    "swap_count",
    "degree_gamma",
    "contrast_feature",
    "contrast_tail",
    "contrast_degree_bin",
    "contrast_pair_id",
)
OUTCOME_COLUMNS = (
    "final_unseeded_accuracy",
    "final_all_accuracy",
    "percolation_matched_unseeded_fraction",
    "percolation_unseeded_precision",
    "percolation_incorrect_unseeded_fraction",
    "percolation_matches_added",
    "percolation_completed",
    "percolation_runtime_seconds",
)


PROBLEMS = {
    "sparse_er": FixedProblem(
        name="sparse_er",
        config=replace(
            preset_config("sparse_er"),
            er_probability=0.020,
            rho=0.70,
            n_seeds=8,
            n_graph_pairs=1,
        ),
        graph_pair_seed=4192889773,
        pilot_success_rate=0.50,
    ),
    "paper_pa": FixedProblem(
        name="paper_pa",
        config=replace(
            preset_config("paper_pa"),
            paper_pa_alpha=0.0,
            paper_pa_beta=1.0,
            paper_pa_probability=0.0167,
            rho=0.62,
            n_seeds=7,
            n_graph_pairs=1,
        ),
        graph_pair_seed=424242,
        pilot_success_rate=0.50,
    ),
    "chung_lu": FixedProblem(
        name="chung_lu",
        config=replace(
            preset_config("chung_lu"),
            chung_lu_alpha=2.0,
            chung_lu_weight_scale=4.076194702720809,
            chung_lu_probability_floor=0.0,
            rho=0.70,
            n_seeds=5,
            n_graph_pairs=1,
        ),
        graph_pair_seed=3165569293,
        pilot_success_rate=0.45,
    ),
}


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def collect_problem(
    spec: FixedProblem,
    output_dir: Path,
    *,
    sampling: dict,
    random_seed: int = 1234,
    algorithm_seed: int = 2026,
    threshold: int = 2,
    expand_when_stuck: bool = True,
    success_threshold: float = 0.90,
    overwrite: bool = False,
) -> None:
    """Collect one percolation run for every sampled set on one fixed pair."""

    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{spec.name}_percolation_seed_sets.csv"
    metadata_path = output_dir / f"{spec.name}_percolation_seed_sets.metadata.json"
    if not overwrite:
        for path in (output_path, metadata_path):
            if path.exists():
                raise FileExistsError(f"{path} already exists; pass --overwrite.")

    problem = prepare_problem(spec, output_dir / "graph_pairs")
    sampling_info: dict = {}
    candidates = sample_seed_sets(
        problem.context_1,
        np.random.default_rng(random_seed),
        spec.config.n_seeds,
        **sampling,
        sampling_info=sampling_info,
    )
    signature_cap = signature_cap_from_formula(problem.context_1.degrees)
    metadata = {
        "status": "running",
        "started_at": _now(),
        "algorithm": "percolation_expand_when_stuck",
        "problem": spec.name,
        "graph_pair_seed": spec.graph_pair_seed,
        "config": asdict(spec.config),
        "percolation_threshold": threshold,
        "expand_when_stuck": expand_when_stuck,
        "algorithm_seed": algorithm_seed,
        "runs_per_seed_set": 1,
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
        "outcome_columns": list(OUTCOME_COLUMNS),
        "output_csv": str(output_path),
        "environment": collect_environment(None),
    }
    write_json(metadata_path, metadata)

    started = time.perf_counter()
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = None
        for candidate_id, candidate in enumerate(candidates):
            seeds_1 = candidate.seeds
            seeds_2 = problem.true_permutation[seeds_1]
            sampling_fields = {
                "data_split": candidate.split,
                "sampling_strategy": candidate.strategy,
                **{
                    column: getattr(candidate, column)
                    for column in SAMPLING_COLUMNS[2:]
                },
            }
            row = {
                "candidate_id": candidate_id,
                **sampling_fields,
                "problem": spec.name,
                "graph_model": spec.config.graph_model,
                "graph_pair_seed": spec.graph_pair_seed,
                "algorithm_seed": algorithm_seed,
                "n_vertices": spec.config.n_vertices,
                "n_seeds": spec.config.n_seeds,
                "rho": spec.config.rho,
                "percolation_threshold": threshold,
                "expand_when_stuck": expand_when_stuck,
                "seed_vertices_graph1": json.dumps(seeds_1.tolist()),
                "seed_vertices_graph2": json.dumps(seeds_2.tolist()),
                **fixed_pair_seed_set_features(problem.context_1, seeds_1),
                **signature_robustness_features(problem.context_1, seeds_1),
                **run_percolation_once(
                    problem.adjacency_1,
                    problem.adjacency_2,
                    problem.true_permutation,
                    seeds_1,
                    algorithm_seed,
                    threshold,
                    expand_when_stuck,
                ),
            }
            row["percolation_success"] = bool(
                row["final_unseeded_accuracy"] >= success_threshold
            )
            if writer is None:
                writer = csv.DictWriter(output_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            output_file.flush()

            completed = candidate_id + 1
            if completed % 25 == 0 or completed == len(candidates):
                metadata["rows_completed"] = completed
                write_json(metadata_path, metadata)
                print(
                    f"{spec.name}: {completed}/{len(candidates)} seed sets "
                    f"({(time.perf_counter() - started) / 60:.1f} minutes)",
                    flush=True,
                )

    metadata.update(
        {
            "status": "complete",
            "finished_at": _now(),
            "collection_minutes": round((time.perf_counter() - started) / 60, 2),
        }
    )
    write_json(metadata_path, metadata)
    print(f"Wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-model", choices=("focus", *FOCUS_PROBLEMS), default="focus"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_COLLECTION_DIR)
    parser.add_argument(
        "--pair-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Graph-pair seeds; each pair is written to its own subfolder.",
    )
    parser.add_argument(
        "--random-train",
        type=int,
        default=DEFAULT_SAMPLING_COUNTS["n_random_train"],
    )
    parser.add_argument(
        "--random-test",
        type=int,
        default=DEFAULT_SAMPLING_COUNTS["n_random_test"],
    )
    parser.add_argument(
        "--degree-weighted",
        type=int,
        default=DEFAULT_SAMPLING_COUNTS["n_degree_weighted"],
    )
    parser.add_argument(
        "--decorrelated",
        type=int,
        default=DEFAULT_SAMPLING_COUNTS["n_decorrelated"],
    )
    parser.add_argument("--pool-size", type=int, default=4000)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--algorithm-seed", type=int, default=2026)
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--success-threshold", type=float, default=0.90)
    parser.add_argument("--disable-expand-when-stuck", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = FOCUS_PROBLEMS if args.graph_model == "focus" else (args.graph_model,)
    sampling = {
        "n_random_train": args.random_train,
        "n_random_test": args.random_test,
        "n_degree_weighted": args.degree_weighted,
        "n_swap_paths": 0,
        "n_decorrelated": args.decorrelated,
        "pool_size": args.pool_size,
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
                spec,
                output_dir,
                sampling=sampling,
                random_seed=args.random_seed,
                algorithm_seed=args.algorithm_seed,
                threshold=args.threshold,
                expand_when_stuck=not args.disable_expand_when_stuck,
                success_threshold=args.success_threshold,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
