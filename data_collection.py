"""Collect seed-set features and SGM performance on correlated graph pairs.

The candidate-level features are computed from graph 1 only.  Graph-level
features summarize both graphs without using the hidden vertex correspondence.
Each SGM run records the first-direction accuracy (H0), final unseeded matching
accuracy, iteration count, convergence status, objective value, and runtime.

The first Frank-Wolfe direction is captured from the same solver run used for
the final matching, so measuring H0 does not require a second SGM run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
from graspologic.match import graph_match
import graspologic.match.wrappers as match_wrappers

from utils.Graphs import gen_ER_graphs, gen_PAPER_graphs, gen_SBM_graphs

if not hasattr(match_wrappers, "_GraphMatchSolver"):
    raise ImportError(
        "data_collection.py requires the private solver API provided by "
        "graspologic 3.4.4. Install the project requirements before running it."
    )


UINT32_MAX = np.iinfo(np.uint32).max
FEATURE_SCHEMA_VERSION = 2
GRAPH_MODELS = ("sparse_er", "sparse_sbm", "paper")
OUTPUT_FILENAMES = {
    "sparse_er": "seed_quality_sparse_er.csv",
    "sparse_sbm": "seed_quality_sparse_sbm.csv",
    "paper": "seed_quality_paper.csv",
}

ROW_CONTEXT_COLUMNS = (
    "graph_model",
    "graph_pair_key",
    "graph_pair_id",
    "candidate_id",
    "graph_pair_seed",
    "sgm_seed",
    "seed_vertices_graph1",
    "seed_vertices_graph2",
    "n_vertices",
    "n_seeds",
    "rho",
    "er_probability",
    "sbm_n_blocks",
    "sbm_n_per_block",
    "sbm_block_probabilities",
    "paper_alpha",
    "paper_probability",
)

SGM_RESULT_COLUMNS = (
    "h0_accuracy",
    "final_unseeded_accuracy",
    "final_all_accuracy",
    "sgm_runtime_seconds",
    "sgm_n_iter",
    "sgm_converged",
    "sgm_objective_score",
)

SPARSE_SBM_BLOCK_PROBABILITIES = (
    (0.02, 0.005, 0.01),
    (0.005, 0.02, 0.005),
    (0.01, 0.005, 0.02),
)


@dataclass(frozen=True)
class ExperimentConfig:
    graph_model: str = "sparse_er"
    n_graph_pairs: int = 20
    candidates_per_pair: int = 50

    vertex_count: int = 600
    n_blocks: int = 3
    n_per_block: int = 200
    sbm_block_probabilities: tuple[tuple[float, ...], ...] = (
        SPARSE_SBM_BLOCK_PROBABILITIES
    )

    er_probability: float = 0.01
    paper_alpha: float = 2.0
    paper_probability: float = 0.005

    rho: float = 0.80
    n_seeds: int = 18
    max_iter: int = 30
    tol: float = 0.01
    random_seed: int = 1234
    max_minutes: float = 30.0

    @property
    def n_vertices(self) -> int:
        if self.graph_model == "sparse_sbm":
            return self.n_blocks * self.n_per_block
        return self.vertex_count


def preset_config(graph_model: str) -> ExperimentConfig:
    """Return the agreed small-scale configuration for one graph model."""

    if graph_model == "sparse_er":
        return ExperimentConfig(
            graph_model="sparse_er",
            vertex_count=600,
            er_probability=0.01,
            rho=0.80,
            n_seeds=18,
        )
    if graph_model == "sparse_sbm":
        return ExperimentConfig(
            graph_model="sparse_sbm",
            n_blocks=3,
            n_per_block=200,
            sbm_block_probabilities=SPARSE_SBM_BLOCK_PROBABILITIES,
            rho=0.80,
            n_seeds=15,
        )
    if graph_model == "paper":
        return ExperimentConfig(
            graph_model="paper",
            vertex_count=600,
            paper_alpha=2.0,
            paper_probability=0.005,
            rho=0.70,
            n_seeds=30,
        )
    raise ValueError(f"Unknown graph model: {graph_model}")


@dataclass
class GraphContext:
    adjacency: np.ndarray
    neighbors: tuple[np.ndarray, ...]
    degrees: np.ndarray
    clustering: np.ndarray
    pagerank: np.ndarray
    core_numbers: np.ndarray
    component_labels: np.ndarray
    component_sizes: np.ndarray
    community_labels: np.ndarray
    community_sizes: np.ndarray
    n_communities: int
    shortest_paths: np.ndarray


def finite_or_nan(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else float("nan")


def quantile_or_nan(values: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.quantile(values, quantile)) if values.size else float("nan")


def normalized_partition_features(
    prefix: str, signatures: np.ndarray
) -> dict[str, float | int]:
    """Summarize how uniquely rows of a signature matrix identify items."""

    signatures = np.asarray(signatures)
    if signatures.ndim == 1:
        signatures = signatures[:, None]
    n_items = signatures.shape[0]
    if n_items == 0:
        return {
            f"{prefix}_class_count": 0,
            f"{prefix}_unique_fraction": float("nan"),
            f"{prefix}_collision_pair_fraction": float("nan"),
            f"{prefix}_largest_class_fraction": float("nan"),
            f"{prefix}_normalized_entropy": float("nan"),
        }

    _, counts = np.unique(signatures, axis=0, return_counts=True)
    proportions = counts / n_items
    collision_denominator = n_items * (n_items - 1)
    collision_fraction = (
        float(np.sum(counts * (counts - 1)) / collision_denominator)
        if collision_denominator > 0
        else 0.0
    )
    normalized_entropy = (
        float(-np.sum(proportions * np.log(proportions)) / np.log(n_items))
        if n_items > 1
        else 0.0
    )
    return {
        f"{prefix}_class_count": int(len(counts)),
        f"{prefix}_unique_fraction": float(np.sum(counts == 1) / n_items),
        f"{prefix}_collision_pair_fraction": collision_fraction,
        f"{prefix}_largest_class_fraction": float(np.max(counts) / n_items),
        f"{prefix}_normalized_entropy": normalized_entropy,
    }


def distribution_summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_q25": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_q75": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_max": float(np.max(values)),
    }


def estimated_community_labels(graph: nx.Graph, n: int) -> tuple[np.ndarray, int]:
    if graph.number_of_edges() == 0:
        return np.arange(n, dtype=int), n

    try:
        communities: Iterable[Iterable[int]] = (
            nx.community.greedy_modularity_communities(graph)
        )
    except (ValueError, ZeroDivisionError):
        communities = nx.connected_components(graph)

    labels = np.full(n, -1, dtype=int)
    community_list = list(communities)
    for label, members in enumerate(community_list):
        labels[np.fromiter(members, dtype=int)] = label

    missing = np.flatnonzero(labels < 0)
    next_label = len(community_list)
    for vertex in missing:
        labels[vertex] = next_label
        next_label += 1
    return labels, next_label


def build_graph_context(adjacency: np.ndarray) -> GraphContext:
    adjacency = np.asarray(adjacency, dtype=np.int8)
    n = adjacency.shape[0]
    graph = nx.from_numpy_array(adjacency)
    neighbors = tuple(np.flatnonzero(adjacency[vertex]) for vertex in range(n))
    degrees = adjacency.sum(axis=1).astype(float)

    clustering_dict = nx.clustering(graph)
    clustering = np.fromiter(
        (clustering_dict[vertex] for vertex in range(n)), dtype=float, count=n
    )

    try:
        pagerank_dict = nx.pagerank(graph, max_iter=500)
        pagerank = np.fromiter(
            (pagerank_dict[vertex] for vertex in range(n)), dtype=float, count=n
        )
    except nx.PowerIterationFailedConvergence:
        degree_sum = degrees.sum()
        pagerank = (
            degrees / degree_sum
            if degree_sum > 0
            else np.full(n, 1.0 / n, dtype=float)
        )

    try:
        core_dict = nx.core_number(graph)
        core_numbers = np.fromiter(
            (core_dict[vertex] for vertex in range(n)), dtype=float, count=n
        )
    except nx.NetworkXError:
        core_numbers = np.zeros(n, dtype=float)

    components = list(nx.connected_components(graph))
    component_labels = np.full(n, -1, dtype=int)
    component_sizes = np.empty(len(components), dtype=int)
    for label, members in enumerate(components):
        member_array = np.fromiter(members, dtype=int)
        component_labels[member_array] = label
        component_sizes[label] = len(member_array)

    community_labels, n_communities = estimated_community_labels(graph, n)
    community_sizes = np.bincount(community_labels, minlength=n_communities)

    shortest_paths = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(shortest_paths, 0.0)
    for source, lengths in nx.all_pairs_shortest_path_length(graph):
        for target, distance in lengths.items():
            shortest_paths[source, target] = distance

    return GraphContext(
        adjacency=adjacency,
        neighbors=neighbors,
        degrees=degrees,
        clustering=clustering,
        pagerank=pagerank,
        core_numbers=core_numbers,
        component_labels=component_labels,
        component_sizes=component_sizes,
        community_labels=community_labels,
        community_sizes=community_sizes,
        n_communities=n_communities,
        shortest_paths=shortest_paths,
    )


def graph_features(prefix: str, context: GraphContext) -> dict[str, float | int]:
    adjacency = context.adjacency
    n = adjacency.shape[0]
    graph = nx.from_numpy_array(adjacency)
    largest_component = int(np.max(context.component_sizes, initial=0))
    mean_degree = float(np.mean(context.degrees))
    degree_std = float(np.std(context.degrees))
    degree_skewness = (
        float(np.mean(((context.degrees - mean_degree) / degree_std) ** 3))
        if degree_std > 0
        else 0.0
    )
    community_proportions = context.community_sizes / n
    community_entropy = (
        float(
            -np.sum(community_proportions * np.log(community_proportions))
            / np.log(context.n_communities)
        )
        if context.n_communities > 1
        else 0.0
    )
    community_sets = [
        set(np.flatnonzero(context.community_labels == label))
        for label in range(context.n_communities)
    ]

    features: dict[str, float | int] = {
        f"{prefix}_average_clustering": float(np.mean(context.clustering)),
        f"{prefix}_transitivity": float(nx.transitivity(graph)),
        f"{prefix}_n_components": len(context.component_sizes),
        f"{prefix}_largest_component_fraction": largest_component / n,
        f"{prefix}_isolate_fraction": float(np.mean(context.degrees == 0)),
        f"{prefix}_degree_coefficient_of_variation": (
            degree_std / mean_degree if mean_degree > 0 else 0.0
        ),
        f"{prefix}_degree_skewness": degree_skewness,
        f"{prefix}_degree_assortativity": finite_or_nan(
            nx.degree_assortativity_coefficient(graph)
        ),
        f"{prefix}_estimated_communities": context.n_communities,
        f"{prefix}_community_modularity": (
            float(nx.community.modularity(graph, community_sets))
            if graph.number_of_edges() > 0
            else 0.0
        ),
        f"{prefix}_largest_community_fraction": float(
            np.max(context.community_sizes, initial=0) / n
        ),
        f"{prefix}_community_size_normalized_entropy": community_entropy,
    }
    features.update(distribution_summary(f"{prefix}_degree", context.degrees))
    return features


def graph_pair_features(
    context_1: GraphContext, context_2: GraphContext
) -> dict[str, float | int]:
    features: dict[str, float | int] = {}
    graph_1_features = graph_features("graph1", context_1)
    graph_2_features = graph_features("graph2", context_2)
    features.update(graph_1_features)
    features.update(graph_2_features)

    sorted_degree_difference = np.abs(
        np.sort(context_1.degrees) - np.sort(context_2.degrees)
    )
    features.update(
        {
            "pair_mean_degree_absolute_difference": abs(
                float(graph_1_features["graph1_degree_mean"])
                - float(graph_2_features["graph2_degree_mean"])
            ),
            "pair_sorted_degree_mean_absolute_difference": float(
                np.mean(sorted_degree_difference)
            ),
            "pair_sorted_degree_max_absolute_difference": float(
                np.max(sorted_degree_difference)
            ),
        }
    )
    return features


def color_refinement_features(
    context: GraphContext, seeds: np.ndarray, nonseeds: np.ndarray, rounds: int = 2
) -> dict[str, float | int]:
    """Measure how quickly uniquely labelled seeds distinguish Graph A vertices."""

    n = context.adjacency.shape[0]
    colors = np.zeros(n, dtype=int)
    colors[seeds] = np.arange(1, len(seeds) + 1)
    features: dict[str, float | int] = {}

    for round_number in range(1, rounds + 1):
        signatures = [
            (
                int(colors[vertex]),
                tuple(
                    sorted(
                        int(colors[neighbor])
                        for neighbor in context.neighbors[vertex]
                    )
                ),
            )
            for vertex in range(n)
        ]
        color_lookup: dict[tuple[int, tuple[int, ...]], int] = {}
        next_colors = np.empty(n, dtype=int)
        for vertex, signature in enumerate(signatures):
            if signature not in color_lookup:
                color_lookup[signature] = len(color_lookup)
            next_colors[vertex] = color_lookup[signature]
        colors = next_colors
        features.update(
            normalized_partition_features(
                f"nonseed_color_refinement_round_{round_number}",
                colors[nonseeds],
            )
        )

    return features


def seed_set_features(
    context: GraphContext, seeds: np.ndarray
) -> dict[str, float | int]:
    """Compute candidate features using only graph 1 and its seed vertices."""

    adjacency = context.adjacency
    n = adjacency.shape[0]
    seeds = np.asarray(seeds, dtype=int)
    is_seed = np.zeros(n, dtype=bool)
    is_seed[seeds] = True
    nonseeds = np.flatnonzero(~is_seed)

    seed_degrees = context.degrees[seeds]
    seed_subgraph = adjacency[np.ix_(seeds, seeds)]
    induced_edges = int(seed_subgraph.sum() // 2)
    possible_induced_edges = len(seeds) * (len(seeds) - 1) / 2
    induced_density = (
        induced_edges / possible_induced_edges if possible_induced_edges > 0 else 0.0
    )

    seed_to_nonseed = adjacency[np.ix_(seeds, nonseeds)]
    nonseed_seed_signatures = seed_to_nonseed.T.astype(np.int32, copy=False)
    nonseed_seed_neighbor_counts = nonseed_seed_signatures.sum(axis=1)
    nonseed_seed_neighbors = nonseed_seed_neighbor_counts.astype(float)
    cut_edges = int(seed_to_nonseed.sum())

    signature_features = normalized_partition_features(
        "nonseed_seed_signature", nonseed_seed_signatures
    )
    signature_overlap = nonseed_seed_signatures @ nonseed_seed_signatures.T
    signature_hamming = (
        nonseed_seed_neighbor_counts[:, None]
        + nonseed_seed_neighbor_counts[None, :]
        - 2 * signature_overlap
    )

    n_nonseeds = len(nonseeds)
    if n_nonseeds > 1:
        off_diagonal_overlap = signature_overlap.copy()
        np.fill_diagonal(off_diagonal_overlap, -1)
        expected_witness_margins = (
            nonseed_seed_neighbor_counts - np.max(off_diagonal_overlap, axis=1)
        ).astype(float)

        off_diagonal_hamming = signature_hamming.copy()
        np.fill_diagonal(off_diagonal_hamming, len(seeds) + 1)
        nearest_signature_hamming = np.min(off_diagonal_hamming, axis=1).astype(
            float
        )

        superset_ambiguity = (
            np.sum(
                signature_overlap == nonseed_seed_neighbor_counts[:, None], axis=1
            )
            - 1
        ).astype(float)

        one_difference_rows, one_difference_columns = np.nonzero(
            np.triu(signature_hamming == 1, k=1)
        )
        if one_difference_rows.size:
            marginal_distinguishability = np.sum(
                nonseed_seed_signatures[one_difference_rows]
                != nonseed_seed_signatures[one_difference_columns],
                axis=0,
            ).astype(float)
        else:
            marginal_distinguishability = np.zeros(len(seeds), dtype=float)
        possible_nonseed_pairs = n_nonseeds * (n_nonseeds - 1) / 2
        marginal_distinguishability /= possible_nonseed_pairs

    else:
        expected_witness_margins = np.zeros(n_nonseeds, dtype=float)
        nearest_signature_hamming = np.full(n_nonseeds, len(seeds), dtype=float)
        superset_ambiguity = np.zeros(n_nonseeds, dtype=float)
        marginal_distinguishability = np.zeros(len(seeds), dtype=float)

    seed_nonseed_degrees = nonseed_seed_signatures.sum(axis=0).astype(float)
    seed_neighborhood_overlap = (
        nonseed_seed_signatures.T @ nonseed_seed_signatures
    )
    seed_neighborhood_unions = (
        seed_nonseed_degrees[:, None]
        + seed_nonseed_degrees[None, :]
        - seed_neighborhood_overlap
    )
    seed_upper = np.triu_indices(len(seeds), k=1)
    seed_nonseed_jaccard = np.divide(
        seed_neighborhood_overlap[seed_upper],
        seed_neighborhood_unions[seed_upper],
        out=np.zeros(len(seed_upper[0]), dtype=float),
        where=seed_neighborhood_unions[seed_upper] > 0,
    )

    seed_neighbor_probabilities = seed_nonseed_degrees / n_nonseeds
    seed_column_entropy = np.zeros(len(seeds), dtype=float)
    interior_probabilities = (seed_neighbor_probabilities > 0) & (
        seed_neighbor_probabilities < 1
    )
    probabilities = seed_neighbor_probabilities[interior_probabilities]
    seed_column_entropy[interior_probabilities] = -(
        probabilities * np.log2(probabilities)
        + (1 - probabilities) * np.log2(1 - probabilities)
    )
    seed_unique_coverage = nonseed_seed_signatures[
        nonseed_seed_neighbor_counts == 1
    ].sum(axis=0).astype(float)

    distances_to_seeds = context.shortest_paths[np.ix_(nonseeds, seeds)]
    nearest_seed_distances = np.min(distances_to_seeds, axis=1)
    finite_nearest_seed_distances = nearest_seed_distances[
        np.isfinite(nearest_seed_distances)
    ]
    distance_signature_features = normalized_partition_features(
        "nonseed_distance_signature", distances_to_seeds
    )
    two_hop_seed_counts = np.sum(distances_to_seeds <= 2, axis=1)
    three_hop_seed_counts = np.sum(distances_to_seeds <= 3, axis=1)

    common_neighbors = adjacency[seeds].astype(np.int32) @ adjacency[seeds].T
    degree_sums = seed_degrees[:, None] + seed_degrees[None, :]
    unions = degree_sums - common_neighbors
    upper = np.triu_indices(len(seeds), k=1)
    pair_unions = unions[upper]
    pair_common = common_neighbors[upper].astype(float)
    jaccard = np.divide(
        pair_common,
        pair_unions,
        out=np.zeros_like(pair_common),
        where=pair_unions > 0,
    )

    pair_distances = context.shortest_paths[np.ix_(seeds, seeds)][upper]
    finite_distances = pair_distances[np.isfinite(pair_distances)]
    disconnected_fraction = (
        float(np.mean(~np.isfinite(pair_distances)))
        if pair_distances.size
        else 0.0
    )

    community_counts = np.bincount(
        context.community_labels[seeds], minlength=context.n_communities
    )
    represented_counts = community_counts[community_counts > 0]
    proportions = represented_counts / len(seeds)
    if len(proportions) > 1:
        normalized_entropy = float(
            -np.sum(proportions * np.log(proportions))
            / np.log(context.n_communities)
        )
    else:
        normalized_entropy = 0.0

    seeded_communities = community_counts > 0
    seedless_community_sizes = context.community_sizes[~seeded_communities]
    community_nonseed_coverage: list[float] = []
    community_nonseed_multi_coverage: list[float] = []
    nonseed_community_labels = context.community_labels[nonseeds]
    for community_label in range(context.n_communities):
        members = nonseed_community_labels == community_label
        if np.any(members):
            community_nonseed_coverage.append(
                float(np.mean(nonseed_seed_neighbor_counts[members] >= 1))
            )
            community_nonseed_multi_coverage.append(
                float(np.mean(nonseed_seed_neighbor_counts[members] >= 2))
            )

    seeded_components = np.zeros(len(context.component_sizes), dtype=bool)
    seeded_components[np.unique(context.component_labels[seeds])] = True
    seedless_component_sizes = context.component_sizes[~seeded_components]
    features: dict[str, float | int] = {
        "seed_induced_density": float(induced_density),
        "seed_cut_edges": cut_edges,
        "nonseed_covered_fraction": float(np.mean(nonseed_seed_neighbors >= 1)),
        "nonseed_covered_by_two_fraction": float(
            np.mean(nonseed_seed_neighbors >= 2)
        ),
        "nonseed_covered_by_three_fraction": float(
            np.mean(nonseed_seed_neighbors >= 3)
        ),
        "nonseed_covered_by_four_fraction": float(
            np.mean(nonseed_seed_neighbors >= 4)
        ),
        "nonseed_seed_neighbor_q10": quantile_or_nan(
            nonseed_seed_neighbors, 0.10
        ),
        "nonseed_expected_witness_margin_mean": float(
            np.mean(expected_witness_margins)
        ),
        "nonseed_expected_witness_margin_q10": quantile_or_nan(
            expected_witness_margins, 0.10
        ),
        "nonseed_expected_witness_margin_at_least_one_fraction": float(
            np.mean(expected_witness_margins >= 1)
        ),
        "nonseed_expected_witness_margin_at_least_two_fraction": float(
            np.mean(expected_witness_margins >= 2)
        ),
        "nonseed_nearest_signature_hamming_mean": float(
            np.mean(nearest_signature_hamming)
        ),
        "nonseed_nearest_signature_hamming_q10": quantile_or_nan(
            nearest_signature_hamming, 0.10
        ),
        "nonseed_nearest_signature_hamming_at_most_one_fraction": float(
            np.mean(nearest_signature_hamming <= 1)
        ),
        "nonseed_superset_ambiguity_log_mean": float(
            np.mean(np.log1p(superset_ambiguity))
        ),
        "nonseed_superset_ambiguity_max": float(np.max(superset_ambiguity)),
        "nonseed_robust_signature_fraction": float(
            np.mean(
                (nonseed_seed_neighbor_counts >= 2)
                & (nearest_signature_hamming >= 2)
            )
        ),
        "seed_nonseed_neighborhood_jaccard_mean": (
            float(np.mean(seed_nonseed_jaccard))
            if seed_nonseed_jaccard.size
            else float("nan")
        ),
        "seed_nonseed_neighborhood_jaccard_q90": quantile_or_nan(
            seed_nonseed_jaccard, 0.90
        ),
        "seed_nonseed_neighborhood_jaccard_max": (
            float(np.max(seed_nonseed_jaccard))
            if seed_nonseed_jaccard.size
            else float("nan")
        ),
        "seed_neighborhood_column_entropy_mean": float(
            np.mean(seed_column_entropy)
        ),
        "seed_neighborhood_column_entropy_min": float(
            np.min(seed_column_entropy)
        ),
        "seed_neighborhood_column_entropy_max": float(
            np.max(seed_column_entropy)
        ),
        "seed_marginal_distinguishability_fraction_mean": float(
            np.mean(marginal_distinguishability)
        ),
        "seed_marginal_distinguishability_fraction_min": float(
            np.min(marginal_distinguishability)
        ),
        "seed_marginal_distinguishability_fraction_std": float(
            np.std(marginal_distinguishability)
        ),
        "seed_zero_marginal_distinguishability_fraction": float(
            np.mean(marginal_distinguishability == 0)
        ),
        "nonseed_within_two_hops_of_seed_fraction": float(
            np.mean(nearest_seed_distances <= 2)
        ),
        "nonseed_within_three_hops_of_seed_fraction": float(
            np.mean(nearest_seed_distances <= 3)
        ),
        "nonseed_two_hop_seed_count_mean": float(np.mean(two_hop_seed_counts)),
        "nonseed_two_hop_seed_count_q10": quantile_or_nan(
            two_hop_seed_counts, 0.10
        ),
        "nonseed_within_two_hops_of_two_seeds_fraction": float(
            np.mean(two_hop_seed_counts >= 2)
        ),
        "nonseed_within_two_hops_of_three_seeds_fraction": float(
            np.mean(two_hop_seed_counts >= 3)
        ),
        "nonseed_three_hop_seed_count_mean": float(
            np.mean(three_hop_seed_counts)
        ),
        "nonseed_three_hop_seed_count_q10": quantile_or_nan(
            three_hop_seed_counts, 0.10
        ),
        "nonseed_within_three_hops_of_two_seeds_fraction": float(
            np.mean(three_hop_seed_counts >= 2)
        ),
        "nonseed_within_three_hops_of_three_seeds_fraction": float(
            np.mean(three_hop_seed_counts >= 3)
        ),
        "nonseed_nearest_seed_distance_mean": (
            float(np.mean(finite_nearest_seed_distances))
            if finite_nearest_seed_distances.size
            else float("nan")
        ),
        "nonseed_nearest_seed_distance_q90": quantile_or_nan(
            finite_nearest_seed_distances, 0.90
        ),
        "nonseed_nearest_seed_distance_max": (
            float(np.max(finite_nearest_seed_distances))
            if finite_nearest_seed_distances.size
            else float("nan")
        ),
        "nonseed_nearest_seed_distance_disconnected_fraction": float(
            np.mean(~np.isfinite(nearest_seed_distances))
        ),
        "seed_component_coverage_fraction": float(np.mean(seeded_components)),
        "seed_largest_uncovered_component_fraction": (
            float(np.max(seedless_component_sizes) / n)
            if seedless_component_sizes.size
            else 0.0
        ),
        "seed_pair_common_neighbor_mean": (
            float(np.mean(pair_common)) if pair_common.size else float("nan")
        ),
        "seed_pair_common_neighbor_std": (
            float(np.std(pair_common)) if pair_common.size else float("nan")
        ),
        "seed_pair_neighborhood_jaccard_mean": (
            float(np.mean(jaccard)) if jaccard.size else float("nan")
        ),
        "seed_pair_neighborhood_jaccard_std": (
            float(np.std(jaccard)) if jaccard.size else float("nan")
        ),
        "seed_pair_shortest_path_mean": (
            float(np.mean(finite_distances))
            if finite_distances.size
            else float("nan")
        ),
        "seed_pair_shortest_path_max": (
            float(np.max(finite_distances))
            if finite_distances.size
            else float("nan")
        ),
        "seed_pair_disconnected_fraction": disconnected_fraction,
        "seed_clustering_mean": float(np.mean(context.clustering[seeds])),
        "seed_clustering_std": float(np.std(context.clustering[seeds])),
        "seed_pagerank_mean": float(np.mean(context.pagerank[seeds])),
        "seed_pagerank_std": float(np.std(context.pagerank[seeds])),
        "seed_core_number_mean": float(np.mean(context.core_numbers[seeds])),
        "seed_core_number_std": float(np.std(context.core_numbers[seeds])),
        "seed_community_coverage_fraction": (
            len(represented_counts) / context.n_communities
        ),
        "seed_community_normalized_entropy": normalized_entropy,
        "seed_community_min_count": int(np.min(represented_counts)),
        "seed_community_max_count": int(np.max(represented_counts)),
        "seed_community_vertex_coverage_fraction": float(
            np.sum(context.community_sizes[seeded_communities]) / n
        ),
        "seed_largest_uncovered_community_fraction": (
            float(np.max(seedless_community_sizes) / n)
            if seedless_community_sizes.size
            else 0.0
        ),
        "nonseed_community_coverage_mean": float(
            np.mean(community_nonseed_coverage)
        ),
        "nonseed_community_coverage_min": float(
            np.min(community_nonseed_coverage)
        ),
        "nonseed_community_multi_coverage_mean": float(
            np.mean(community_nonseed_multi_coverage)
        ),
        "nonseed_community_multi_coverage_min": float(
            np.min(community_nonseed_multi_coverage)
        ),
    }
    features.update(
        distribution_summary("nonseed_seed_neighbor", nonseed_seed_neighbors)
    )
    features.update(signature_features)
    features.update(
        distribution_summary("seed_nonseed_degree", seed_nonseed_degrees)
    )
    features.update(distribution_summary("seed_unique_coverage", seed_unique_coverage))
    features.update(distance_signature_features)
    features.update(color_refinement_features(context, seeds, nonseeds, rounds=2))
    features.update(distribution_summary("seed_degree", seed_degrees))
    # Remove exact rescalings of retained features.  Keeping a single
    # representative prevents correlated models from splitting importance
    # among columns that contain the same information.
    for redundant_feature in (
        "nonseed_seed_neighbor_mean",  # seed_cut_edges / number of nonseeds
        "seed_nonseed_degree_mean",  # seed_cut_edges / number of seeds
        "seed_unique_coverage_mean",  # exact-one coverage / number of seeds
        # With a fixed seed budget, this is determined by cut size and the
        # induced seed density through the degree-sum identity.
        "seed_degree_mean",
    ):
        features.pop(redundant_feature, None)
    return features


def run_sgm_once(
    adjacency_1: np.ndarray,
    adjacency_2: np.ndarray,
    true_permutation: np.ndarray,
    seeds_1: np.ndarray,
    sgm_seed: int,
    max_iter: int,
    tol: float,
) -> dict[str, float | int | bool]:
    """Run SGM once and score both its first direction and final matching."""

    seeds_1 = np.asarray(seeds_1, dtype=int)
    seeds_2 = true_permutation[seeds_1]
    partial_match = np.column_stack((seeds_1, seeds_2))

    captured: dict[str, Any] = {}
    original_compute_step_direction = (
        match_wrappers._GraphMatchSolver.compute_step_direction
    )

    def capture_first_direction(
        solver: Any, gradient: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        direction = original_compute_step_direction(solver, gradient, rng)
        if "first_direction" not in captured:
            captured["solver"] = solver
            captured["first_direction"] = np.asarray(direction).copy()
        return direction

    match_wrappers._GraphMatchSolver.compute_step_direction = capture_first_direction
    started = time.perf_counter()
    try:
        result = graph_match(
            adjacency_1,
            adjacency_2,
            partial_match=partial_match,
            n_init=1,
            max_iter=max_iter,
            tol=tol,
            rng=sgm_seed,
        )
    finally:
        match_wrappers._GraphMatchSolver.compute_step_direction = (
            original_compute_step_direction
        )
    runtime_seconds = time.perf_counter() - started

    first_direction = captured.get("first_direction")
    solver = captured.get("solver")
    if first_direction is None or solver is None:
        raise RuntimeError("SGM terminated without computing a first direction.")

    unseeded_1 = solver.perm_A[solver.n_seeds :]
    unseeded_2 = solver.perm_B[solver.n_seeds :]
    first_direction_prediction = unseeded_2[np.argmax(first_direction, axis=1)]
    h0_accuracy = float(
        np.mean(first_direction_prediction == true_permutation[unseeded_1])
    )

    final_permutation = np.full(adjacency_1.shape[0], -1, dtype=int)
    final_permutation[result.indices_A] = result.indices_B
    all_vertices = np.arange(adjacency_1.shape[0])
    final_unseeded_vertices = np.setdiff1d(all_vertices, seeds_1)
    final_unseeded_accuracy = float(
        np.mean(
            final_permutation[final_unseeded_vertices]
            == true_permutation[final_unseeded_vertices]
        )
    )
    final_all_accuracy = float(np.mean(final_permutation == true_permutation))

    return {
        "h0_accuracy": h0_accuracy,
        "final_unseeded_accuracy": final_unseeded_accuracy,
        "final_all_accuracy": final_all_accuracy,
        "sgm_runtime_seconds": runtime_seconds,
        "sgm_n_iter": int(solver.n_iter_),
        "sgm_converged": bool(solver.converged_),
        "sgm_objective_score": float(result.score),
    }


def sample_unique_seed_sets(
    rng: np.random.Generator, n_vertices: int, n_seeds: int, count: int
) -> list[np.ndarray]:
    samples: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    while len(samples) < count:
        sample = np.sort(rng.choice(n_vertices, size=n_seeds, replace=False))
        key = tuple(int(vertex) for vertex in sample)
        if key not in seen:
            seen.add(key)
            samples.append(sample)
    return samples


def generate_graph_pair(
    config: ExperimentConfig, graph_pair_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(graph_pair_seed)
    if config.graph_model == "sparse_er":
        adjacency_1, adjacency_2, true_permutation = gen_ER_graphs(
            n=config.vertex_count,
            p=config.er_probability,
            rho=config.rho,
            directed=False,
            loops=False,
        )
    elif config.graph_model == "sparse_sbm":
        adjacency_1, adjacency_2, true_permutation = gen_SBM_graphs(
            n_per_block=config.n_per_block,
            n_blocks=config.n_blocks,
            block_probs=np.asarray(config.sbm_block_probabilities, dtype=float),
            rho=config.rho,
            directed=False,
            loops=False,
        )
    elif config.graph_model == "paper":
        adjacency_1, adjacency_2, true_permutation = gen_PAPER_graphs(
            n=config.vertex_count,
            alpha=config.paper_alpha,
            p=config.paper_probability,
            rho=config.rho,
            directed=False,
            loops=False,
        )
    else:
        raise ValueError(f"Unknown graph model: {config.graph_model}")
    return (
        np.asarray(adjacency_1, dtype=float),
        np.asarray(adjacency_2, dtype=float),
        np.asarray(true_permutation, dtype=int),
    )


def validate_config(config: ExperimentConfig) -> None:
    if config.graph_model not in GRAPH_MODELS:
        raise ValueError(f"graph_model must be one of {GRAPH_MODELS}.")
    if config.n_graph_pairs <= 0 or config.candidates_per_pair <= 0:
        raise ValueError("Graph-pair and candidate counts must be positive.")
    if config.vertex_count <= 0:
        raise ValueError("vertex_count must be positive.")
    if config.n_blocks <= 0 or config.n_per_block <= 0:
        raise ValueError("Block counts and sizes must be positive.")
    if not 0 < config.n_seeds < config.n_vertices:
        raise ValueError("n_seeds must be between 1 and n_vertices - 1.")
    if config.candidates_per_pair > math.comb(config.n_vertices, config.n_seeds):
        raise ValueError("Requested more unique seed sets than are possible.")
    for name, probability in (
        ("er_probability", config.er_probability),
        ("paper_probability", config.paper_probability),
        ("rho", config.rho),
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    block_probabilities = np.asarray(config.sbm_block_probabilities, dtype=float)
    if block_probabilities.shape != (config.n_blocks, config.n_blocks):
        raise ValueError(
            "sbm_block_probabilities must have shape (n_blocks, n_blocks)."
        )
    if np.any((block_probabilities < 0) | (block_probabilities > 1)):
        raise ValueError("All SBM block probabilities must be between 0 and 1.")
    if config.paper_alpha <= 0:
        raise ValueError("paper_alpha must be positive.")
    if config.max_iter <= 0 or config.tol <= 0 or config.max_minutes <= 0:
        raise ValueError("max_iter, tol, and max_minutes must be positive.")


def refresh_collected_features(output_path: Path) -> dict[str, Any]:
    """Rebuild a collected CSV's features while preserving saved SGM results."""

    output_path = output_path.expanduser().resolve()
    metadata_path = output_path.with_suffix(".metadata.json")
    for required_path in (output_path, metadata_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Could not find {required_path}.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = ExperimentConfig(**metadata["config"])
    validate_config(config)

    import pandas as pd

    existing_frame = pd.read_csv(output_path)
    required_columns = set(ROW_CONTEXT_COLUMNS) | set(SGM_RESULT_COLUMNS)
    missing_columns = required_columns - set(existing_frame.columns)
    if missing_columns:
        raise ValueError(
            f"Cannot refresh {output_path}; missing columns: "
            f"{sorted(missing_columns)}"
        )
    if existing_frame.empty:
        raise ValueError(f"Cannot refresh empty dataset {output_path}.")

    refreshed_rows: list[dict[str, Any]] = []
    grouped = existing_frame.groupby("graph_pair_key", sort=False)
    for pair_number, (pair_key, group) in enumerate(grouped, start=1):
        pair_seeds = group["graph_pair_seed"].unique()
        if len(pair_seeds) != 1:
            raise ValueError(f"Graph pair {pair_key} has multiple generation seeds.")
        graph_pair_seed = int(pair_seeds[0])
        adjacency_1, adjacency_2, true_permutation = generate_graph_pair(
            config, graph_pair_seed
        )
        context_1 = build_graph_context(adjacency_1)
        context_2 = build_graph_context(adjacency_2)
        pair_features = graph_pair_features(context_1, context_2)

        for _, saved_row in group.iterrows():
            seeds_1 = np.asarray(
                json.loads(saved_row["seed_vertices_graph1"]), dtype=int
            )
            seeds_2 = np.asarray(
                json.loads(saved_row["seed_vertices_graph2"]), dtype=int
            )
            if len(seeds_1) != config.n_seeds or len(np.unique(seeds_1)) != len(
                seeds_1
            ):
                raise ValueError(
                    f"Invalid Graph A seed set in {pair_key}, candidate "
                    f"{saved_row['candidate_id']}."
                )
            if not np.array_equal(seeds_2, true_permutation[seeds_1]):
                raise ValueError(
                    f"Regenerated correspondence does not match {pair_key}, "
                    f"candidate {saved_row['candidate_id']}."
                )

            refreshed_row: dict[str, Any] = {
                "feature_schema_version": FEATURE_SCHEMA_VERSION
            }
            refreshed_row.update(
                {column: saved_row[column] for column in ROW_CONTEXT_COLUMNS}
            )
            refreshed_row.update(pair_features)
            refreshed_row.update(seed_set_features(context_1, seeds_1))
            refreshed_row.update(
                {column: saved_row[column] for column in SGM_RESULT_COLUMNS}
            )
            refreshed_rows.append(refreshed_row)

        print(
            f"  Recomputed features for {pair_number}/{len(grouped)} graph pairs "
            f"in {output_path.name}.",
            flush=True,
        )

    refreshed_frame = pd.DataFrame(refreshed_rows)
    original_keys = existing_frame[["graph_pair_key", "candidate_id"]].reset_index(
        drop=True
    )
    refreshed_keys = refreshed_frame[
        ["graph_pair_key", "candidate_id"]
    ].reset_index(drop=True)
    if not original_keys.equals(refreshed_keys):
        raise ValueError("Refreshed rows do not align with the original observations.")

    for column in SGM_RESULT_COLUMNS:
        original = existing_frame[column].to_numpy()
        refreshed = refreshed_frame[column].to_numpy()
        if not np.array_equal(original, refreshed, equal_nan=True):
            raise ValueError(f"SGM result column changed during refresh: {column}.")

    temporary_output = output_path.with_suffix(".csv.tmp")
    refreshed_frame.to_csv(temporary_output, index=False)
    temporary_output.replace(output_path)

    metadata["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    metadata["columns"] = list(refreshed_frame.columns)
    metadata["rows_written"] = len(refreshed_frame)
    metadata["features_refreshed_without_sgm"] = True
    excluded = metadata.setdefault("exclude_from_prequery_models", [])
    if "feature_schema_version" not in excluded:
        excluded.insert(0, "feature_schema_version")

    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    with temporary_metadata.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")
    temporary_metadata.replace(metadata_path)

    print(
        f"Refreshed {output_path} with {len(refreshed_frame)} rows and "
        f"{len(refreshed_frame.columns)} columns; saved SGM results were preserved."
    )
    return metadata


def collect_data(
    config: ExperimentConfig, output_path: Path, overwrite: bool
) -> dict[str, Any]:
    validate_config(config)
    output_path = output_path.expanduser().resolve()
    metadata_path = output_path.with_suffix(".metadata.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Choose another path or pass --overwrite."
        )

    rng = np.random.default_rng(config.random_seed)
    collection_started = time.perf_counter()
    time_budget_seconds = config.max_minutes * 60.0
    safety_margin_seconds = min(60.0, 0.05 * time_budget_seconds)
    stop_new_work_at = collection_started + time_budget_seconds - safety_margin_seconds

    rows_written = 0
    completed_graph_pairs = 0
    stopped_for_time = False
    recent_sgm_runtimes: list[float] = []
    fieldnames: list[str] | None = None

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer: csv.DictWriter[str] | None = None

        for graph_pair_id in range(config.n_graph_pairs):
            if time.perf_counter() >= stop_new_work_at:
                stopped_for_time = True
                break

            graph_pair_seed = int(rng.integers(0, UINT32_MAX, dtype=np.uint32))
            adjacency_1, adjacency_2, true_permutation = generate_graph_pair(
                config, graph_pair_seed
            )
            context_1 = build_graph_context(adjacency_1)
            context_2 = build_graph_context(adjacency_2)
            pair_features = graph_pair_features(context_1, context_2)
            candidate_sets = sample_unique_seed_sets(
                rng,
                config.n_vertices,
                config.n_seeds,
                config.candidates_per_pair,
            )

            completed_candidates = 0
            for candidate_id, seeds_1 in enumerate(candidate_sets):
                remaining_seconds = stop_new_work_at - time.perf_counter()
                estimated_next_runtime = (
                    float(np.median(recent_sgm_runtimes[-20:]))
                    if len(recent_sgm_runtimes) >= 5
                    else 0.0
                )
                if remaining_seconds <= estimated_next_runtime:
                    stopped_for_time = True
                    break

                seeds_2 = true_permutation[seeds_1]
                sgm_seed = int(rng.integers(0, UINT32_MAX, dtype=np.uint32))
                candidate_features = seed_set_features(context_1, seeds_1)
                performance = run_sgm_once(
                    adjacency_1,
                    adjacency_2,
                    true_permutation,
                    seeds_1,
                    sgm_seed,
                    config.max_iter,
                    config.tol,
                )
                recent_sgm_runtimes.append(float(performance["sgm_runtime_seconds"]))

                row: dict[str, Any] = {
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "graph_model": config.graph_model,
                    "graph_pair_key": f"{config.graph_model}_{graph_pair_id}",
                    "graph_pair_id": graph_pair_id,
                    "candidate_id": candidate_id,
                    "graph_pair_seed": graph_pair_seed,
                    "sgm_seed": sgm_seed,
                    "seed_vertices_graph1": json.dumps(seeds_1.tolist()),
                    "seed_vertices_graph2": json.dumps(seeds_2.tolist()),
                    "n_vertices": config.n_vertices,
                    "n_seeds": config.n_seeds,
                    "rho": config.rho,
                    "er_probability": (
                        config.er_probability
                        if config.graph_model == "sparse_er"
                        else None
                    ),
                    "sbm_n_blocks": (
                        config.n_blocks
                        if config.graph_model == "sparse_sbm"
                        else None
                    ),
                    "sbm_n_per_block": (
                        config.n_per_block
                        if config.graph_model == "sparse_sbm"
                        else None
                    ),
                    "sbm_block_probabilities": (
                        json.dumps(config.sbm_block_probabilities)
                        if config.graph_model == "sparse_sbm"
                        else None
                    ),
                    "paper_alpha": (
                        config.paper_alpha
                        if config.graph_model == "paper"
                        else None
                    ),
                    "paper_probability": (
                        config.paper_probability
                        if config.graph_model == "paper"
                        else None
                    ),
                }
                row.update(pair_features)
                row.update(candidate_features)
                row.update(performance)

                if writer is None:
                    fieldnames = list(row)
                    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow(row)
                rows_written += 1
                completed_candidates += 1

                if rows_written % 10 == 0:
                    elapsed_minutes = (time.perf_counter() - collection_started) / 60
                    print(
                        f"Collected {rows_written} rows "
                        f"({elapsed_minutes:.1f} minutes elapsed).",
                        flush=True,
                    )

            output_file.flush()
            if completed_candidates == config.candidates_per_pair:
                completed_graph_pairs += 1
            if stopped_for_time:
                break

    elapsed_seconds = time.perf_counter() - collection_started
    metadata: dict[str, Any] = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "config": asdict(config),
        "output_csv": str(output_path),
        "columns": fieldnames or [],
        "rows_written": rows_written,
        "completed_graph_pairs": completed_graph_pairs,
        "stopped_for_time_budget": stopped_for_time,
        "elapsed_seconds": elapsed_seconds,
        "group_column_for_model_validation": "graph_pair_key",
        "target_columns": [
            "h0_accuracy",
            "final_unseeded_accuracy",
            "final_all_accuracy",
            "sgm_runtime_seconds",
        ],
        "candidate_feature_scope": "graph 1 only",
        "graph_feature_scope": (
            "permutation-invariant summaries of graph 1 and graph 2"
        ),
        "prequery_feature_prefixes": [
            "graph1_",
            "graph2_",
            "pair_",
            "seed_",
            "nonseed_",
        ],
        "exclude_from_prequery_models": [
            "seed_vertices_graph2",
            "h0_accuracy",
            "final_unseeded_accuracy",
            "final_all_accuracy",
            "sgm_runtime_seconds",
            "sgm_n_iter",
            "sgm_converged",
            "sgm_objective_score",
        ],
        "time_budget_is_soft": (
            "The collector stops before starting new SGM runs; one active run is not "
            "interrupted and may finish after the requested deadline."
        ),
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")

    print(f"Wrote {rows_written} observations to {output_path}")
    print(f"Wrote run metadata to {metadata_path}")
    if stopped_for_time:
        print("Stopped before starting another SGM run because of the time budget.")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-model",
        choices=("all",) + GRAPH_MODELS,
        default="all",
        help="Collect all recommended regimes or one selected graph model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for the preset output CSV and metadata files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Custom CSV path; available only when collecting one graph model.",
    )
    parser.add_argument("--n-graph-pairs", type=int, default=20)
    parser.add_argument("--candidates-per-pair", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=0.01)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=30.0,
        help="Soft time limit applied separately to each output dataset.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output CSV."
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help=(
            "Rebuild feature columns in existing CSV files from their saved graph "
            "seeds without rerunning SGM."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.graph_model == "all" and args.output is not None:
        raise ValueError("--output can only be used with one selected graph model.")

    graph_models = GRAPH_MODELS if args.graph_model == "all" else (args.graph_model,)
    for graph_model in graph_models:
        config = replace(
            preset_config(graph_model),
            n_graph_pairs=args.n_graph_pairs,
            candidates_per_pair=args.candidates_per_pair,
            max_iter=args.max_iter,
            tol=args.tol,
            random_seed=args.random_seed,
            max_minutes=args.max_minutes,
        )
        output_path = (
            args.output
            if args.output is not None
            else args.output_dir / OUTPUT_FILENAMES[graph_model]
        )
        if args.features_only:
            print(f"Refreshing features in {output_path} without running SGM.")
            refresh_collected_features(output_path)
            continue
        print(
            f"Collecting {graph_model}: {config.n_graph_pairs} graph pairs x "
            f"{config.candidates_per_pair} candidates, {config.n_seeds} seeds."
        )
        collect_data(config, output_path, args.overwrite)


if __name__ == "__main__":
    main()
