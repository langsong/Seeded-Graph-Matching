"""Collect compact candidate features and one-seed SGM diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache")
)
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache")
)

import numpy as np

from experiment_common.features import build_graph_context
from .one_seed_problem import (
    PROBLEMS,
    SUCCESS_THRESHOLD,
    PreparedProblem,
    SGMRun,
    evaluate_candidate,
    load_problem,
    run_sgm,
    scalar_run_summary,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"

FEATURE_COLUMNS = (
    "candidate_degree_percentile",
    "candidate_core_percentile",
    "candidate_edges_to_existing_seeds",
    "candidate_new_one_hop_coverage_fraction",
    "candidate_new_second_witness_fraction",
    "candidate_new_two_hop_coverage_fraction",
    "candidate_mean_nearest_seed_distance_reduction",
    "candidate_q90_nearest_seed_distance_reduction",
    "candidate_signature_collision_reduction",
    "candidate_max_seed_neighborhood_jaccard",
    "candidate_existing_seed_community_fraction",
    "candidate_seed_signature_best_agreement",
    "candidate_seed_signature_agreement_gap",
    "candidate_gradient_row_top2_gap_z",
    "candidate_gradient_row_normalized_entropy",
)


def _collision_fraction(signatures: np.ndarray) -> float:
    _, counts = np.unique(signatures, axis=0, return_counts=True)
    denominator = len(signatures) * (len(signatures) - 1)
    if denominator == 0:
        return 0.0
    return float(np.sum(counts * (counts - 1)) / denominator)


def _gradient_row_features(
    baseline: SGMRun, candidate: int
) -> tuple[float, float]:
    row = int(np.flatnonzero(baseline.unseeded_1 == candidate)[0])
    scores = baseline.first_gradient[row].astype(float)
    scale = float(np.std(scores))
    standardized = (scores - np.mean(scores)) / (scale if scale > 0 else 1.0)
    ordered = np.sort(standardized)
    top2_gap = float(ordered[-1] - ordered[-2])

    weights = np.exp(standardized - np.max(standardized))
    probabilities = weights / weights.sum()
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-15))
    normalized_entropy = float(entropy / np.log(len(probabilities)))
    return top2_gap, normalized_entropy


def candidate_feature_row(
    problem: PreparedProblem,
    context,
    baseline: SGMRun,
    candidate: int,
) -> dict[str, float | int | str]:
    """Compute the compact pre-query feature panel for one candidate."""

    candidate = int(candidate)
    n = problem.adjacency_1.shape[0]
    seeds_1 = problem.seeds_1
    seeds_2 = problem.seeds_2
    augmented_seeds = np.append(seeds_1, candidate)
    remaining = np.setdiff1d(np.arange(n), augmented_seeds)

    adjacency = problem.adjacency_1
    shortest_paths = context.shortest_paths
    base_witnesses = adjacency[np.ix_(remaining, seeds_1)].sum(axis=1)
    candidate_neighbors = adjacency[remaining, candidate].astype(bool)

    base_distances = shortest_paths[np.ix_(remaining, seeds_1)]
    base_nearest = np.min(base_distances, axis=1)
    candidate_distances = shortest_paths[remaining, candidate]
    augmented_nearest = np.minimum(base_nearest, candidate_distances)
    base_capped = np.where(np.isfinite(base_nearest), base_nearest, n)
    augmented_capped = np.where(np.isfinite(augmented_nearest), augmented_nearest, n)

    two_hop_seed_counts = np.sum(base_distances <= 2, axis=1)
    candidate_within_two_hops = candidate_distances <= 2

    base_signatures = adjacency[np.ix_(remaining, seeds_1)]
    augmented_signatures = np.column_stack(
        (base_signatures, adjacency[remaining, candidate])
    )

    candidate_neighbors_set = set(context.neighbors[candidate])
    jaccards = []
    for seed in seeds_1:
        seed_neighbors = set(context.neighbors[int(seed)])
        union = candidate_neighbors_set | seed_neighbors
        jaccards.append(
            len(candidate_neighbors_set & seed_neighbors) / len(union)
            if union
            else 0.0
        )

    candidate_community = context.community_labels[candidate]
    existing_seed_community_fraction = float(
        np.mean(context.community_labels[seeds_1] == candidate_community)
    )

    signature_1 = adjacency[candidate, seeds_1]
    signatures_2 = problem.adjacency_2[np.ix_(baseline.unseeded_2, seeds_2)]
    agreements = np.mean(signatures_2 == signature_1, axis=1)
    ordered_agreements = np.sort(agreements)
    agreement_gap = float(ordered_agreements[-1] - ordered_agreements[-2])

    gradient_gap, gradient_entropy = _gradient_row_features(baseline, candidate)

    return {
        "problem": problem.spec.name,
        "candidate_vertex_graph1": candidate,
        "candidate_degree_percentile": float(
            np.mean(context.degrees <= context.degrees[candidate])
        ),
        "candidate_core_percentile": float(
            np.mean(context.core_numbers <= context.core_numbers[candidate])
        ),
        "candidate_edges_to_existing_seeds": int(
            adjacency[candidate, seeds_1].sum()
        ),
        "candidate_new_one_hop_coverage_fraction": float(
            np.mean(candidate_neighbors & (base_witnesses == 0))
        ),
        "candidate_new_second_witness_fraction": float(
            np.mean(candidate_neighbors & (base_witnesses == 1))
        ),
        "candidate_new_two_hop_coverage_fraction": float(
            np.mean(candidate_within_two_hops & (two_hop_seed_counts == 0))
        ),
        "candidate_mean_nearest_seed_distance_reduction": float(
            np.mean(base_capped - augmented_capped)
        ),
        "candidate_q90_nearest_seed_distance_reduction": float(
            np.quantile(base_capped, 0.90)
            - np.quantile(augmented_capped, 0.90)
        ),
        "candidate_signature_collision_reduction": (
            _collision_fraction(base_signatures)
            - _collision_fraction(augmented_signatures)
        ),
        "candidate_max_seed_neighborhood_jaccard": float(max(jaccards)),
        "candidate_existing_seed_community_fraction": (
            existing_seed_community_fraction
        ),
        "candidate_seed_signature_best_agreement": float(np.max(agreements)),
        "candidate_seed_signature_agreement_gap": agreement_gap,
        "candidate_gradient_row_top2_gap_z": gradient_gap,
        "candidate_gradient_row_normalized_entropy": gradient_entropy,
    }


def collect_problem(
    problem_name: str,
    output_dir: Path,
    success_threshold: float,
    overwrite: bool,
) -> None:
    problem = load_problem(problem_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / f"{problem_name}_candidate_features.csv"
    outcome_path = output_dir / f"{problem_name}_candidate_outcomes.csv"
    metadata_path = output_dir / f"{problem_name}_metadata.json"

    if not overwrite and any(
        path.exists() for path in (feature_path, outcome_path, metadata_path)
    ):
        raise FileExistsError(
            f"Output already exists for {problem_name}; pass --overwrite."
        )

    context = build_graph_context(
        problem.adjacency_1,
        compute_shortest_paths=True,
        compute_candidate_features=True,
    )
    baseline = run_sgm(problem, problem.seeds_1)
    candidates = np.setdiff1d(
        np.arange(problem.spec.config.n_vertices), problem.seeds_1
    )

    started = time.perf_counter()
    feature_writer = None
    outcome_writer = None
    successes = 0
    with feature_path.open("w", newline="", encoding="utf-8") as feature_file, (
        outcome_path.open("w", newline="", encoding="utf-8")
    ) as outcome_file:
        for index, candidate in enumerate(candidates, start=1):
            feature_row = candidate_feature_row(
                problem, context, baseline, int(candidate)
            )
            outcome_row = evaluate_candidate(
                problem,
                int(candidate),
                baseline=baseline,
                success_threshold=success_threshold,
            )

            if feature_writer is None:
                feature_writer = csv.DictWriter(
                    feature_file, fieldnames=list(feature_row)
                )
                feature_writer.writeheader()
                outcome_writer = csv.DictWriter(
                    outcome_file, fieldnames=list(outcome_row)
                )
                outcome_writer.writeheader()
            feature_writer.writerow(feature_row)
            outcome_writer.writerow(outcome_row)
            successes += int(outcome_row["outcome_success"])

            if index % 25 == 0 or index == len(candidates):
                elapsed = (time.perf_counter() - started) / 60
                print(
                    f"{problem_name}: {index}/{len(candidates)} candidates "
                    f"({elapsed:.1f} minutes)",
                    flush=True,
                )

    metadata = {
        "problem": problem_name,
        "description": problem.spec.description,
        "config": asdict(problem.spec.config),
        "graph_pair_seed": problem.spec.graph_pair_seed,
        "sgm_seed": problem.spec.sgm_seed,
        "existing_seed_vertices_graph1": problem.seeds_1.tolist(),
        "existing_seed_vertices_graph2": problem.seeds_2.tolist(),
        "baseline_sgm": scalar_run_summary(baseline),
        "success_threshold": success_threshold,
        "candidate_count": len(candidates),
        "candidate_success_rate": successes / len(candidates),
        "feature_columns": list(FEATURE_COLUMNS),
        "predictor_scope": (
            "Graph A structure plus correspondence-free Graph B and baseline "
            "SGM uncertainty; the candidate's true Graph B match is excluded"
        ),
        "gradient_diagnostics_are_postquery_outcomes": True,
        "feature_file": str(feature_path),
        "outcome_file": str(outcome_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {feature_path} and {outcome_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem", choices=("all", *PROBLEMS), default="all"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--success-threshold", type=float, default=SUCCESS_THRESHOLD
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = tuple(PROBLEMS) if args.problem == "all" else (args.problem,)
    for name in names:
        collect_problem(
            name,
            args.output_dir,
            args.success_threshold,
            args.overwrite,
        )


if __name__ == "__main__":
    main()
