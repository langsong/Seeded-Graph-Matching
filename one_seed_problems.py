"""Create three fixed problems for choosing one additional SGM seed.

Each problem exposes two correlated graphs and an existing set of known seed
correspondences.  For every unseeded vertex in graph A, the script records
features that are available before querying its correspondence.  SGM outcomes
and the revealed graph-B vertex are written to a separate table so they cannot
be included accidentally as predictors.

The fixed instances were selected by a small calibration.  In each one, the
existing seed set has final unseeded accuracy below 0.9, while adding different
vertices produces both successful and unsuccessful SGM runs.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_collection import (
    ExperimentConfig,
    FEATURE_SCHEMA_VERSION,
    GraphContext,
    build_graph_context,
    generate_graph_pair,
    graph_pair_features,
    run_sgm_once,
    seed_set_features,
)


SUCCESS_THRESHOLD = 0.90
DEFAULT_OUTPUT_DIR = Path("data/one_seed_problems")

SPARSE_SBM_PROBABILITIES = (
    (0.04, 0.01, 0.02),
    (0.01, 0.04, 0.01),
    (0.02, 0.01, 0.04),
)

# These seed-set summaries are recomputed after adding each candidate.  Their
# changes describe the candidate's marginal contribution using graph A only.
MARGINAL_FEATURES = (
    "nonseed_expected_witness_margin_mean",
    "nonseed_expected_witness_margin_at_least_one_fraction",
    "nonseed_seed_signature_unique_fraction",
    "nonseed_seed_signature_collision_pair_fraction",
    "nonseed_robust_signature_fraction",
    "nonseed_distance_signature_unique_fraction",
    "nonseed_color_refinement_round_2_unique_fraction",
)


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    description: str
    config: ExperimentConfig
    graph_pair_seed: int
    sgm_seed: int
    seed_vertices_graph1: tuple[int, ...]


PROBLEMS: dict[str, ProblemSpec] = {
    "sparse_er": ProblemSpec(
        name="sparse_er",
        description=(
            "Sparse homogeneous control: candidate quality cannot rely on an "
            "explicit community or hub structure."
        ),
        config=ExperimentConfig(
            graph_model="sparse_er",
            vertex_count=300,
            er_probability=0.02,
            rho=0.80,
            n_seeds=8,
            max_iter=30,
            tol=0.01,
        ),
        graph_pair_seed=3414274753,
        sgm_seed=680531080,
        seed_vertices_graph1=(80, 125, 139, 231, 235, 284, 285, 287),
    ),
    "sparse_sbm": ProblemSpec(
        name="sparse_sbm",
        description=(
            "Sparse three-community graph: useful candidates may add coverage "
            "where the existing seeds are underrepresented."
        ),
        config=ExperimentConfig(
            graph_model="sparse_sbm",
            n_blocks=3,
            n_per_block=100,
            sbm_block_probabilities=SPARSE_SBM_PROBABILITIES,
            rho=0.80,
            n_seeds=8,
            max_iter=30,
            tol=0.01,
        ),
        graph_pair_seed=440533379,
        sgm_seed=680291271,
        seed_vertices_graph1=(9, 37, 73, 95, 127, 168, 251, 286),
    ),
    "paper": ProblemSpec(
        name="paper",
        description=(
            "Sparse heterogeneous graph: candidate vertices vary substantially "
            "in degree, coverage, and structural role."
        ),
        config=ExperimentConfig(
            graph_model="paper",
            vertex_count=300,
            paper_alpha=2.0,
            paper_probability=0.01,
            rho=0.70,
            n_seeds=14,
            max_iter=30,
            tol=0.01,
        ),
        graph_pair_seed=1615603467,
        sgm_seed=1471549916,
        seed_vertices_graph1=(
            10,
            25,
            35,
            39,
            57,
            59,
            85,
            102,
            114,
            126,
            232,
            238,
            261,
            282,
        ),
    ),
}


def json_value(value: Any) -> Any:
    """Convert NumPy and non-finite values into JSON-safe values."""

    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_value(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def known_seed_cross_graph_features(
    context_1: GraphContext,
    context_2: GraphContext,
    seeds_1: np.ndarray,
    seeds_2: np.ndarray,
) -> dict[str, float]:
    """Summarize the known seed matches using information from both graphs."""

    degrees_1 = context_1.degrees[seeds_1]
    degrees_2 = context_2.degrees[seeds_2]
    degree_difference = np.abs(degrees_1 - degrees_2)

    seed_adjacency_1 = context_1.adjacency[np.ix_(seeds_1, seeds_1)]
    seed_adjacency_2 = context_2.adjacency[np.ix_(seeds_2, seeds_2)]
    upper = np.triu_indices(len(seeds_1), k=1)
    edge_agreement = float(
        np.mean(seed_adjacency_1[upper] == seed_adjacency_2[upper])
    )

    if np.std(degrees_1) > 0 and np.std(degrees_2) > 0:
        degree_correlation = float(np.corrcoef(degrees_1, degrees_2)[0, 1])
    else:
        degree_correlation = 0.0

    return {
        "known_seed_degree_difference_mean": float(np.mean(degree_difference)),
        "known_seed_degree_difference_std": float(np.std(degree_difference)),
        "known_seed_degree_difference_max": float(np.max(degree_difference)),
        "known_seed_degree_correlation": degree_correlation,
        "known_seed_induced_edge_agreement": edge_agreement,
    }


def candidate_features(
    context: GraphContext,
    base_seeds: np.ndarray,
    candidate: int,
    base_features: dict[str, float | int],
) -> dict[str, float | int]:
    """Compute graph-A-only features for one possible additional seed."""

    n = context.adjacency.shape[0]
    augmented_seeds = np.sort(np.append(base_seeds, candidate))
    remaining = np.setdiff1d(np.arange(n), augmented_seeds)
    existing_witness_counts = context.adjacency[:, base_seeds].sum(axis=1)
    candidate_neighbors = context.adjacency[candidate].astype(bool)
    edges_to_existing_seeds = int(
        context.adjacency[candidate, base_seeds].sum()
    )

    distances = context.shortest_paths[candidate, base_seeds]
    finite_distances = distances[np.isfinite(distances)]
    candidate_nonseed_distances = context.shortest_paths[candidate, remaining]
    finite_candidate_nonseed_distances = candidate_nonseed_distances[
        np.isfinite(candidate_nonseed_distances)
    ]

    distances_from_nonseeds_to_seeds = context.shortest_paths[
        np.ix_(remaining, base_seeds)
    ]
    base_nearest_seed_distances = np.min(
        distances_from_nonseeds_to_seeds, axis=1
    )
    augmented_nearest_seed_distances = np.minimum(
        base_nearest_seed_distances, candidate_nonseed_distances
    )
    previously_connected = np.isfinite(base_nearest_seed_distances)
    newly_connected = (~previously_connected) & np.isfinite(
        candidate_nonseed_distances
    )
    becomes_nearest_seed = (
        candidate_nonseed_distances < base_nearest_seed_distances
    )
    finite_distance_reductions = (
        base_nearest_seed_distances[previously_connected]
        - augmented_nearest_seed_distances[previously_connected]
    )

    two_hop_seed_counts = np.sum(distances_from_nonseeds_to_seeds <= 2, axis=1)
    three_hop_seed_counts = np.sum(distances_from_nonseeds_to_seeds <= 3, axis=1)
    candidate_within_two_hops = candidate_nonseed_distances <= 2
    candidate_within_three_hops = candidate_nonseed_distances <= 3

    candidate_neighbor_set = set(context.neighbors[candidate])
    jaccards = []
    for seed in base_seeds:
        seed_neighbor_set = set(context.neighbors[seed])
        union = candidate_neighbor_set | seed_neighbor_set
        jaccards.append(
            len(candidate_neighbor_set & seed_neighbor_set) / len(union)
            if union
            else 0.0
        )

    community = int(context.community_labels[candidate])
    component = int(context.component_labels[candidate])
    augmented_features = seed_set_features(context, augmented_seeds)

    row: dict[str, float | int] = {
        "candidate_vertex_graph1": int(candidate),
        "candidate_degree": float(context.degrees[candidate]),
        "candidate_degree_percentile": float(
            np.mean(context.degrees <= context.degrees[candidate])
        ),
        "candidate_pagerank": float(context.pagerank[candidate]),
        "candidate_core_number": float(context.core_numbers[candidate]),
        "candidate_clustering": float(context.clustering[candidate]),
        "candidate_component_size_fraction": float(
            context.component_sizes[component] / n
        ),
        "candidate_community_size_fraction": float(
            context.community_sizes[community] / n
        ),
        "candidate_existing_seeds_in_component": int(
            np.sum(context.component_labels[base_seeds] == component)
        ),
        "candidate_existing_seeds_in_community": int(
            np.sum(context.community_labels[base_seeds] == community)
        ),
        "candidate_edges_to_existing_seeds": edges_to_existing_seeds,
        "candidate_new_coverage_count": int(
            np.sum(candidate_neighbors[remaining] & (existing_witness_counts[remaining] == 0))
        ),
        "candidate_new_two_coverage_count": int(
            np.sum(candidate_neighbors[remaining] & (existing_witness_counts[remaining] == 1))
        ),
        "candidate_new_three_coverage_count": int(
            np.sum(candidate_neighbors[remaining] & (existing_witness_counts[remaining] == 2))
        ),
        "candidate_nearest_existing_seed_distance": (
            float(np.min(finite_distances)) if finite_distances.size else float(n)
        ),
        "candidate_mean_existing_seed_distance": (
            float(np.mean(finite_distances)) if finite_distances.size else float(n)
        ),
        "candidate_nonseed_distance_mean": (
            float(np.mean(finite_candidate_nonseed_distances))
            if finite_candidate_nonseed_distances.size
            else float(n)
        ),
        "candidate_nonseed_distance_q90": (
            float(np.quantile(finite_candidate_nonseed_distances, 0.90))
            if finite_candidate_nonseed_distances.size
            else float(n)
        ),
        "candidate_nonseed_distance_max": (
            float(np.max(finite_candidate_nonseed_distances))
            if finite_candidate_nonseed_distances.size
            else float(n)
        ),
        "candidate_nonseed_distance_disconnected_fraction": float(
            np.mean(~np.isfinite(candidate_nonseed_distances))
        ),
        "candidate_nonseed_within_two_hops_fraction": float(
            np.mean(candidate_within_two_hops)
        ),
        "candidate_nonseed_within_three_hops_fraction": float(
            np.mean(candidate_within_three_hops)
        ),
        "candidate_becomes_nearest_seed_fraction": float(
            np.mean(becomes_nearest_seed)
        ),
        "candidate_newly_connected_nonseed_count": int(
            np.sum(newly_connected)
        ),
        "candidate_distance_reduction_mean": (
            float(np.mean(finite_distance_reductions))
            if finite_distance_reductions.size
            else 0.0
        ),
        "candidate_distance_reduction_q90": (
            float(np.quantile(finite_distance_reductions, 0.90))
            if finite_distance_reductions.size
            else 0.0
        ),
        "candidate_distance_reduction_max": (
            float(np.max(finite_distance_reductions))
            if finite_distance_reductions.size
            else 0.0
        ),
        "candidate_radius_two_new_coverage_count": int(
            np.sum(candidate_within_two_hops & (two_hop_seed_counts == 0))
        ),
        "candidate_radius_two_new_two_coverage_count": int(
            np.sum(candidate_within_two_hops & (two_hop_seed_counts == 1))
        ),
        "candidate_radius_three_new_coverage_count": int(
            np.sum(candidate_within_three_hops & (three_hop_seed_counts == 0))
        ),
        "candidate_radius_three_new_two_coverage_count": int(
            np.sum(candidate_within_three_hops & (three_hop_seed_counts == 1))
        ),
        "candidate_seed_neighborhood_jaccard_mean": float(np.mean(jaccards)),
        "candidate_seed_neighborhood_jaccard_max": float(np.max(jaccards)),
    }

    for feature in MARGINAL_FEATURES:
        row[f"candidate_delta_{feature}"] = float(
            augmented_features[feature] - base_features[feature]
        )
    return row


def output_paths(output_dir: Path, problem_name: str) -> dict[str, Path]:
    prefix = output_dir / problem_name
    return {
        "problem": prefix.with_name(f"{problem_name}_problem.npz"),
        "features": prefix.with_name(f"{problem_name}_candidate_features.csv"),
        "outcomes": prefix.with_name(f"{problem_name}_candidate_outcomes.csv"),
        "metadata": prefix.with_name(f"{problem_name}_metadata.json"),
    }


def refresh_problem_features(problem_name: str, output_dir: Path) -> dict[str, Any]:
    """Rebuild a saved problem's feature table without running SGM."""

    paths = output_paths(output_dir, problem_name)
    for required_path in (paths["problem"], paths["metadata"]):
        if not required_path.exists():
            raise FileNotFoundError(f"Could not find {required_path}.")

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    with np.load(paths["problem"]) as saved_problem:
        adjacency_1 = np.asarray(saved_problem["adjacency_graph1"], dtype=np.int8)
        adjacency_2 = np.asarray(saved_problem["adjacency_graph2"], dtype=np.int8)
        seeds_1 = np.asarray(saved_problem["seed_vertices_graph1"], dtype=int)
        seeds_2 = np.asarray(saved_problem["seed_vertices_graph2"], dtype=int)
        candidate_vertices = np.asarray(
            saved_problem["candidate_vertices_graph1"], dtype=int
        )

    n = adjacency_1.shape[0]
    if adjacency_1.shape != adjacency_2.shape or adjacency_1.shape != (n, n):
        raise ValueError(f"Invalid saved adjacency matrices for {problem_name}.")
    if len(seeds_1) != len(seeds_2):
        raise ValueError(f"Mismatched saved seed sets for {problem_name}.")
    expected_candidates = np.setdiff1d(np.arange(n), seeds_1)
    if not np.array_equal(candidate_vertices, expected_candidates):
        raise ValueError(f"Invalid saved candidate list for {problem_name}.")

    context_1 = build_graph_context(adjacency_1)
    context_2 = build_graph_context(adjacency_2)
    pair_features = graph_pair_features(context_1, context_2)
    base_features_1 = seed_set_features(context_1, seeds_1)
    base_features_2 = seed_set_features(context_2, seeds_2)
    cross_seed_features = known_seed_cross_graph_features(
        context_1, context_2, seeds_1, seeds_2
    )

    repeated_context: dict[str, Any] = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "problem": problem_name,
        "n_vertices": n,
        "n_existing_seeds": len(seeds_1),
        "rho": float(metadata["config"]["rho"]),
    }
    repeated_context.update(pair_features)
    repeated_context.update(
        {f"base_graph1_{key}": value for key, value in base_features_1.items()}
    )
    repeated_context.update(
        {f"base_graph2_{key}": value for key, value in base_features_2.items()}
    )
    repeated_context.update(cross_seed_features)

    feature_rows: list[dict[str, Any]] = []
    for candidate in candidate_vertices:
        row = dict(repeated_context)
        row.update(
            candidate_features(context_1, seeds_1, int(candidate), base_features_1)
        )
        feature_rows.append(row)
    features = pd.DataFrame(feature_rows)

    if paths["outcomes"].exists():
        outcomes = pd.read_csv(
            paths["outcomes"],
            usecols=["problem", "candidate_vertex_graph1"],
        )
        feature_keys = features[["problem", "candidate_vertex_graph1"]]
        if not feature_keys.equals(outcomes):
            raise ValueError(
                f"Refreshed feature rows do not align with outcomes for {problem_name}."
            )

    temporary_features = paths["features"].with_suffix(".csv.tmp")
    features.to_csv(temporary_features, index=False)
    temporary_features.replace(paths["features"])

    metadata["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    metadata["candidate_count"] = len(candidate_vertices)
    metadata["available_information"] = {
        "graph_pair_features": pair_features,
        "base_seed_features_graph1": base_features_1,
        "base_seed_features_graph2": base_features_2,
        "known_seed_cross_graph_features": cross_seed_features,
    }
    excluded = metadata.setdefault("exclude_from_prequery_models", [])
    if "feature_schema_version" not in excluded:
        excluded.insert(0, "feature_schema_version")

    temporary_metadata = paths["metadata"].with_suffix(".json.tmp")
    with temporary_metadata.open("w", encoding="utf-8") as file:
        json.dump(json_value(metadata), file, indent=2)
        file.write("\n")
    temporary_metadata.replace(paths["metadata"])

    print(
        f"Refreshed {paths['features']} with {len(features)} rows and "
        f"{len(features.columns)} columns; SGM outcomes were not recomputed."
    )
    return metadata


def build_problem(
    spec: ProblemSpec, output_dir: Path, overwrite: bool
) -> dict[str, Any]:
    """Generate, validate, and save one fixed candidate-selection problem."""

    paths = output_paths(output_dir, spec.name)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {names}. Pass --overwrite.")

    adjacency_1, adjacency_2, true_permutation = generate_graph_pair(
        spec.config, spec.graph_pair_seed
    )
    context_1 = build_graph_context(adjacency_1)
    context_2 = build_graph_context(adjacency_2)

    seeds_1 = np.asarray(spec.seed_vertices_graph1, dtype=int)
    if len(seeds_1) != spec.config.n_seeds or len(np.unique(seeds_1)) != len(seeds_1):
        raise ValueError(f"Invalid existing seed set for {spec.name}.")
    seeds_2 = true_permutation[seeds_1]
    candidate_vertices = np.setdiff1d(np.arange(spec.config.n_vertices), seeds_1)

    pair_features = graph_pair_features(context_1, context_2)
    base_features_1 = seed_set_features(context_1, seeds_1)
    base_features_2 = seed_set_features(context_2, seeds_2)
    cross_seed_features = known_seed_cross_graph_features(
        context_1, context_2, seeds_1, seeds_2
    )

    baseline = run_sgm_once(
        adjacency_1,
        adjacency_2,
        true_permutation,
        seeds_1,
        spec.sgm_seed,
        spec.config.max_iter,
        spec.config.tol,
    )

    feature_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    repeated_context: dict[str, Any] = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "problem": spec.name,
        "n_vertices": spec.config.n_vertices,
        "n_existing_seeds": len(seeds_1),
        "rho": spec.config.rho,
    }
    repeated_context.update(pair_features)
    repeated_context.update(
        {f"base_graph1_{key}": value for key, value in base_features_1.items()}
    )
    repeated_context.update(
        {f"base_graph2_{key}": value for key, value in base_features_2.items()}
    )
    repeated_context.update(cross_seed_features)

    for index, candidate in enumerate(candidate_vertices, start=1):
        feature_row = dict(repeated_context)
        feature_row.update(
            candidate_features(context_1, seeds_1, int(candidate), base_features_1)
        )
        feature_rows.append(feature_row)

        augmented_seeds = np.sort(np.append(seeds_1, int(candidate)))
        result = run_sgm_once(
            adjacency_1,
            adjacency_2,
            true_permutation,
            augmented_seeds,
            spec.sgm_seed,
            spec.config.max_iter,
            spec.config.tol,
        )
        outcome_rows.append(
            {
                "problem": spec.name,
                "candidate_vertex_graph1": int(candidate),
                "revealed_vertex_graph2": int(true_permutation[candidate]),
                "baseline_h0_accuracy": baseline["h0_accuracy"],
                "baseline_final_unseeded_accuracy": baseline[
                    "final_unseeded_accuracy"
                ],
                "outcome_h0_accuracy": result["h0_accuracy"],
                "outcome_final_unseeded_accuracy": result[
                    "final_unseeded_accuracy"
                ],
                "outcome_final_all_accuracy": result["final_all_accuracy"],
                "outcome_accuracy_improvement": (
                    result["final_unseeded_accuracy"]
                    - baseline["final_unseeded_accuracy"]
                ),
                "outcome_success": bool(
                    result["final_unseeded_accuracy"] >= SUCCESS_THRESHOLD
                ),
                "sgm_runtime_seconds": result["sgm_runtime_seconds"],
                "sgm_n_iter": result["sgm_n_iter"],
                "sgm_converged": result["sgm_converged"],
                "sgm_objective_score": result["sgm_objective_score"],
            }
        )
        if index % 50 == 0 or index == len(candidate_vertices):
            print(f"  {spec.name}: evaluated {index}/{len(candidate_vertices)} candidates")

    features = pd.DataFrame(feature_rows)
    outcomes = pd.DataFrame(outcome_rows)
    success_rate = float(outcomes["outcome_success"].mean())
    accuracy = outcomes["outcome_final_unseeded_accuracy"]

    if float(baseline["final_unseeded_accuracy"]) >= SUCCESS_THRESHOLD:
        raise RuntimeError(f"{spec.name} base seed set is already successful.")
    if not 0.05 <= success_rate <= 0.95:
        raise RuntimeError(
            f"{spec.name} candidate success rate {success_rate:.3f} does not "
            "provide a meaningful success/failure distinction."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["problem"],
        adjacency_graph1=np.asarray(adjacency_1, dtype=np.int8),
        adjacency_graph2=np.asarray(adjacency_2, dtype=np.int8),
        seed_vertices_graph1=seeds_1,
        seed_vertices_graph2=seeds_2,
        candidate_vertices_graph1=candidate_vertices,
    )
    features.to_csv(paths["features"], index=False)
    outcomes.to_csv(paths["outcomes"], index=False)

    metadata = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "problem": spec.name,
        "description": spec.description,
        "config": asdict(spec.config),
        "graph_pair_seed": spec.graph_pair_seed,
        "sgm_seed": spec.sgm_seed,
        "success_threshold": SUCCESS_THRESHOLD,
        "known_seed_matches": [
            {"vertex_graph1": int(a), "vertex_graph2": int(b)}
            for a, b in zip(seeds_1, seeds_2)
        ],
        "baseline_sgm": baseline,
        "candidate_count": len(candidate_vertices),
        "candidate_success_count": int(outcomes["outcome_success"].sum()),
        "candidate_failure_count": int((~outcomes["outcome_success"]).sum()),
        "candidate_success_rate": success_rate,
        "candidate_accuracy_min": float(accuracy.min()),
        "candidate_accuracy_median": float(accuracy.median()),
        "candidate_accuracy_max": float(accuracy.max()),
        "available_information": {
            "graph_pair_features": pair_features,
            "base_seed_features_graph1": base_features_1,
            "base_seed_features_graph2": base_features_2,
            "known_seed_cross_graph_features": cross_seed_features,
        },
        "files": {key: str(path) for key, path in paths.items()},
        "prequery_table": paths["features"].name,
        "postquery_evaluation_table": paths["outcomes"].name,
        "join_columns": ["problem", "candidate_vertex_graph1"],
        "exclude_from_prequery_models": [
            "feature_schema_version",
            "candidate_vertex_graph1",
            "revealed_vertex_graph2",
            "baseline_h0_accuracy",
            "baseline_final_unseeded_accuracy",
            "outcome_h0_accuracy",
            "outcome_final_unseeded_accuracy",
            "outcome_final_all_accuracy",
            "outcome_accuracy_improvement",
            "outcome_success",
            "sgm_runtime_seconds",
            "sgm_n_iter",
            "sgm_converged",
            "sgm_objective_score",
        ],
    }
    with paths["metadata"].open("w", encoding="utf-8") as file:
        json.dump(json_value(metadata), file, indent=2)
        file.write("\n")

    print(
        f"Created {spec.name}: baseline={baseline['final_unseeded_accuracy']:.3f}, "
        f"candidate success rate={success_rate:.1%}."
    )
    return metadata


def print_existing_summary(output_dir: Path, names: tuple[str, ...]) -> None:
    for name in names:
        metadata_path = output_paths(output_dir, name)["metadata"]
        if not metadata_path.exists():
            raise FileNotFoundError(f"Could not find {metadata_path}.")
        metadata = json.loads(metadata_path.read_text())
        print(
            f"{name}: baseline="
            f"{metadata['baseline_sgm']['final_unseeded_accuracy']:.3f}, "
            f"successful candidates={metadata['candidate_success_count']}/"
            f"{metadata['candidate_count']} "
            f"({metadata['candidate_success_rate']:.1%})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem",
        choices=("all",) + tuple(PROBLEMS),
        default="all",
        help="Create all three problems or one selected graph model.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing problem files."
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print summaries from existing metadata without rerunning SGM.",
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help=(
            "Rebuild candidate feature CSV files from saved problem NPZ files "
            "without running SGM."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = tuple(PROBLEMS) if args.problem == "all" else (args.problem,)
    if args.summary_only and args.features_only:
        raise ValueError("Choose either --summary-only or --features-only, not both.")
    if args.summary_only:
        print_existing_summary(args.output_dir, names)
        return
    if args.features_only:
        for name in names:
            refresh_problem_features(name, args.output_dir)
        return
    for name in names:
        print(f"Creating {name} one-seed problem")
        build_problem(PROBLEMS[name], args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
