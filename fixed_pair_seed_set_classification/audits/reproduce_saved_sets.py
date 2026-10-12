"""Re-evaluate a few saved seed sets and check that features and SGM outcomes match.

Run from the repository root, in the environment used for collection:

    python -m fixed_pair_seed_set_classification.audits.reproduce_saved_sets \
        --graph-model sparse_er --data-dir fixed_pair_seed_set_classification/data/sgm_pilot_400

``--features-only`` skips SGM (useful where graspologic is unavailable). SGM
outcomes depend on the BLAS thread count, so use the same ``--blas-threads``
as the collection (4 by default). The exit code is 1 if anything differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_common.features import (
    FIXED_PAIR_FEATURE_COLUMNS,
    fixed_pair_seed_set_features,
    signature_robustness_features,
)
from experiment_common.graphs import ExperimentConfig
from experiment_common.provenance import thread_limit

from ..candidate_data import PROBLEMS
from ..seed_selection_common import prepare_problem

SGM_COMPARE_COLUMNS = (
    "h0_accuracy",
    "final_unseeded_accuracy",
    "final_all_accuracy",
    "sgm_n_iter",
    "sgm_converged",
    "sgm_objective_score",
)
TOLERANCE = 1e-9


def choose_sets(data: pd.DataFrame, n_sets: int, seed: int) -> list[int]:
    """Spread the checked sets over the sampling strategies."""

    rng = np.random.default_rng(seed)
    groups = [
        rng.permutation(group["candidate_id"].to_numpy())
        for _, group in data.groupby("sampling_strategy", sort=True)
    ]
    chosen: list[int] = []
    depth = 0
    while len(chosen) < n_sets and any(depth < len(group) for group in groups):
        for group in groups:
            if depth < len(group) and len(chosen) < n_sets:
                chosen.append(int(group[depth]))
        depth += 1
    return chosen


def differs(saved, recomputed) -> bool:
    if isinstance(saved, (bool, np.bool_)) or isinstance(recomputed, (bool, np.bool_)):
        return bool(saved) != bool(recomputed)
    if pd.isna(saved) and pd.isna(recomputed):
        return False
    return abs(float(saved) - float(recomputed)) > TOLERANCE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graph-model", choices=tuple(PROBLEMS), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--n-sets", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--blas-threads", type=int, default=4)
    args = parser.parse_args()

    name = args.graph_model
    data = pd.read_csv(args.data_dir / f"{name}_seed_sets.csv")
    runs = pd.read_csv(args.data_dir / f"{name}_seed_sets_runs.csv")
    metadata = json.loads(
        (args.data_dir / f"{name}_seed_sets.metadata.json").read_text(encoding="utf-8")
    )
    # Reconstruct the exact saved problem rather than assuming that today's
    # calibrated preset matches a historical collection.
    saved_spec = replace(
        PROBLEMS[name],
        config=ExperimentConfig(**metadata["config"]),
        graph_pair_seed=int(metadata["graph_pair_seed"]),
        sgm_seed=int(metadata.get("sgm_seed", PROBLEMS[name].sgm_seed)),
    )
    problem = prepare_problem(saved_spec, args.data_dir / "graph_pairs")
    tol = metadata["config"]["tol"]
    feature_columns = metadata.get("feature_columns", list(FIXED_PAIR_FEATURE_COLUMNS))
    feature_columns += [c for c in metadata.get("robustness_feature_columns", []) if c in data]

    if not args.features_only:
        from experiment_common.sgm import run_sgm_once

    problems: list[str] = []
    checked_runs = 0
    chosen = choose_sets(data, args.n_sets, args.seed)
    with thread_limit(args.blas_threads or None):
        for candidate_id in chosen:
            row = data.loc[data["candidate_id"] == candidate_id].iloc[0]
            seeds = np.asarray(json.loads(row["seed_vertices_graph1"]), dtype=int)
            label = f"set {candidate_id} ({row['sampling_strategy']})"

            features = {
                **fixed_pair_seed_set_features(problem.context_1, seeds),
                **signature_robustness_features(problem.context_1, seeds),
            }
            for column in feature_columns:
                if differs(row[column], features[column]):
                    problems.append(
                        f"{label}: feature {column} saved {row[column]!r}, "
                        f"recomputed {features[column]!r}"
                    )
            if args.features_only:
                continue

            for _, run in runs.loc[runs["candidate_id"] == candidate_id].iterrows():
                result = run_sgm_once(
                    problem.adjacency_1,
                    problem.adjacency_2,
                    problem.true_permutation,
                    seeds,
                    int(run["sgm_seed"]),
                    int(run["sgm_max_iter"]),
                    tol,
                )
                checked_runs += 1
                for column in SGM_COMPARE_COLUMNS:
                    if differs(run[column], result[column]):
                        problems.append(
                            f"{label}, SGM seed {int(run['sgm_seed'])}: {column} "
                            f"saved {run[column]!r}, recomputed {result[column]!r}"
                        )
            print(f"checked {label}", flush=True)

    print(
        f"\n{name}: {len(chosen)} seed sets, {len(feature_columns)} features"
        + ("" if args.features_only else f", {checked_runs} SGM runs")
    )
    if problems:
        print(f"{len(problems)} differences:")
        for line in problems:
            print("  " + line)
        return 1
    print("Everything matches the saved data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
