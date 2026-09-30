"""Collect SGM outcomes for many seed sets on one fixed graph pair."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .pairs.er_seed_selection import PROBLEM as ER_PROBLEM
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
from experiment_common.features import seed_set_features


PROBLEMS = {
    "sparse_er": ER_PROBLEM,
    "sparse_sbm": SBM_PROBLEM,
    "paper": PAPER_PROBLEM,
    "paper_pa": PAPER_PA_PROBLEM,
    "ier": IER_PROBLEM,
    "enron": ENRON_PROBLEM,
}


def draw_unique_seed_sets(
    rng: np.random.Generator,
    n_vertices: int,
    n_seeds: int,
    count: int,
    seen: set[tuple[int, ...]],
) -> list[np.ndarray]:
    samples: list[np.ndarray] = []
    while len(samples) < count:
        seeds = np.sort(rng.choice(n_vertices, size=n_seeds, replace=False))
        key = tuple(int(vertex) for vertex in seeds)
        if key not in seen:
            seen.add(key)
            samples.append(seeds)
    return samples


def select_feature_diverse_sets(
    problem,
    candidates: list[np.ndarray],
    count: int,
) -> tuple[list[np.ndarray], list[str]]:
    """Greedily select seed sets spread across standardized feature space."""

    feature_rows = [
        seed_set_features(problem.context_1, seeds) for seeds in candidates
    ]
    feature_names = list(feature_rows[0])
    values = np.asarray(
        [[row[name] for name in feature_names] for row in feature_rows], dtype=float
    )
    medians = np.nanmedian(values, axis=0)
    values = np.where(np.isfinite(values), values, medians)
    scales = values.std(axis=0)
    keep = scales > 0
    standardized = (values[:, keep] - values[:, keep].mean(axis=0)) / scales[keep]

    first = int(np.argmax(np.sum(standardized**2, axis=1)))
    selected = [first]
    minimum_distance = np.sum(
        (standardized - standardized[first]) ** 2, axis=1
    )
    minimum_distance[first] = -1.0
    while len(selected) < count:
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        distance = np.sum(
            (standardized - standardized[next_index]) ** 2, axis=1
        )
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -1.0

    return [candidates[index] for index in selected], feature_names


def collect_problem(
    spec: FixedProblem,
    output_dir: Path,
    n_random_train: int,
    n_diverse_train: int,
    n_random_test: int,
    pool_size: int,
    random_seed: int,
    success_threshold: float,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{spec.name}_seed_sets.csv"
    metadata_path = output_dir / f"{spec.name}_seed_sets.metadata.json"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite.")

    problem = prepare_problem(spec, output_dir)
    rng = np.random.default_rng(random_seed)
    seen: set[tuple[int, ...]] = set()
    n = spec.config.n_vertices
    k = spec.config.n_seeds

    random_test = draw_unique_seed_sets(rng, n, k, n_random_test, seen)
    random_train = draw_unique_seed_sets(rng, n, k, n_random_train, seen)
    pool = draw_unique_seed_sets(rng, n, k, pool_size, seen)
    diverse_train, feature_names = select_feature_diverse_sets(
        problem, pool, n_diverse_train
    )

    candidates = [
        *(('train', 'random', seeds) for seeds in random_train),
        *(('train', 'feature_diverse', seeds) for seeds in diverse_train),
        *(('test', 'random', seeds) for seeds in random_test),
    ]

    started = time.perf_counter()
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = None
        for candidate_id, (split, strategy, seeds_1) in enumerate(candidates):
            row = {
                "candidate_id": candidate_id,
                "data_split": split,
                "sampling_strategy": strategy,
            }
            row.update(evaluate_seed_set(problem, seeds_1))
            row["sgm_success"] = bool(
                row["final_unseeded_accuracy"] >= success_threshold
            )

            if writer is None:
                writer = csv.DictWriter(output_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)

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
        "success_threshold": success_threshold,
        "random_seed": random_seed,
        "n_random_train": n_random_train,
        "n_feature_diverse_train": n_diverse_train,
        "n_random_test": n_random_test,
        "feature_pool_size": pool_size,
        "rows": len(candidates),
        "feature_columns": feature_names,
        "target_columns": [
            "h0_accuracy",
            "final_unseeded_accuracy",
            "final_all_accuracy",
            "sgm_success",
        ],
        "predictor_scope": "Graph A seed-set features only",
        "output_csv": str(output_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-model", choices=("all", *PROBLEMS), default="all"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROBLEM_DIR)
    parser.add_argument("--random-train", type=int, default=500)
    parser.add_argument("--diverse-train", type=int, default=400)
    parser.add_argument("--random-test", type=int, default=100)
    parser.add_argument("--pool-size", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--success-threshold", type=float, default=0.90)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = PROBLEMS if args.graph_model == "all" else (args.graph_model,)
    for name in names:
        collect_problem(
            PROBLEMS[name],
            args.output_dir,
            args.random_train,
            args.diverse_train,
            args.random_test,
            args.pool_size,
            args.random_seed,
            args.success_threshold,
            args.overwrite,
        )


if __name__ == "__main__":
    main()
