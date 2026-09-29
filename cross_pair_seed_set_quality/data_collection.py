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
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
from graspologic.match import graph_match
import graspologic.match.wrappers as match_wrappers

from utils.Graphs import (
    gen_ER_graphs,
    gen_IER_graphs,
    gen_PAPER_graphs,
    gen_pareto_chung_lu_graphs,
    gen_SBM_graphs,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "data"

if not hasattr(match_wrappers, "_GraphMatchSolver"):
    raise ImportError(
        "data_collection.py requires the private solver API provided by "
        "graspologic 3.4.4. Install the project requirements before running it."
    )


UINT32_MAX = np.iinfo(np.uint32).max
FEATURE_SCHEMA_VERSION = 3
GRAPH_MODELS = ("sparse_er", "sparse_sbm", "paper", "ier", "paper_pa")
OUTPUT_FILENAMES = {
    "sparse_er": "seed_quality_sparse_er.csv",
    "sparse_sbm": "seed_quality_sparse_sbm.csv",
    "paper": "seed_quality_paper.csv",
    "ier": "seed_quality_ier.csv",
    "paper_pa": "seed_quality_paper_pa.csv",
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
    paper_pa_alpha: float = 0.0
    paper_pa_beta: float = 1.0
    paper_pa_probability: float = 0.005
    ier_alpha: float = 1.0
    ier_beta: float = 99.0

    rho: float | None = 0.80
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
    if graph_model == "paper_pa":
        return ExperimentConfig(
            graph_model="paper_pa",
            vertex_count=600,
            paper_pa_alpha=0.0,
            paper_pa_beta=1.0,
            paper_pa_probability=0.005,
            rho=0.62,
            n_seeds=20,
        )
    if graph_model == "ier":
        return ExperimentConfig(
            graph_model="ier",
            vertex_count=600,
            ier_alpha=1.0,
            ier_beta=99.0,
            rho=0.77,
            n_seeds=20,
        )
    raise ValueError(f"Unknown graph model: {graph_model}")


@dataclass
class GraphContext:
    adjacency: np.ndarray
    neighbors: tuple[np.ndarray, ...]
    degrees: np.ndarray
    clustering: np.ndarray
    pagerank: np.ndarray
    betweenness: np.ndarray
    core_numbers: np.ndarray
    component_labels: np.ndarray
    component_sizes: np.ndarray
    community_labels: np.ndarray
    community_sizes: np.ndarray
    n_communities: int
    shortest_paths: np.ndarray | None
    seed_distance_cache: dict[int, np.ndarray]


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


def build_graph_context(
    adjacency: np.ndarray,
    *,
    compute_shortest_paths: bool = False,
    compute_seed_features: bool = True,
    compute_candidate_features: bool = False,
    betweenness_sources: int = 64,
) -> GraphContext:
    adjacency = np.asarray(adjacency, dtype=np.int8)
    n = adjacency.shape[0]
    graph = nx.from_numpy_array(adjacency)
    neighbors = tuple(np.flatnonzero(adjacency[vertex]) for vertex in range(n))
    degrees = adjacency.sum(axis=1).astype(float)

    components = list(nx.connected_components(graph))
    component_labels = np.full(n, -1, dtype=int)
    component_sizes = np.empty(len(components), dtype=int)
    for label, members in enumerate(components):
        member_array = np.fromiter(members, dtype=int)
        component_labels[member_array] = label
        component_sizes[label] = len(member_array)

    if compute_seed_features:
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

        betweenness_dict = nx.betweenness_centrality(
            graph,
            k=min(betweenness_sources, n) if betweenness_sources < n else None,
            normalized=True,
            seed=0,
        )
        betweenness = np.fromiter(
            (betweenness_dict[vertex] for vertex in range(n)), dtype=float, count=n
        )
        community_labels, n_communities = estimated_community_labels(graph, n)
        community_sizes = np.bincount(community_labels, minlength=n_communities)
    else:
        pagerank = np.zeros(n, dtype=float)
        betweenness = np.zeros(n, dtype=float)
        community_labels = component_labels.copy()
        community_sizes = component_sizes.copy()
        n_communities = len(components)

    if compute_candidate_features:
        clustering_dict = nx.clustering(graph)
        clustering = np.fromiter(
            (clustering_dict[vertex] for vertex in range(n)), dtype=float, count=n
        )
        try:
            core_dict = nx.core_number(graph)
            core_numbers = np.fromiter(
                (core_dict[vertex] for vertex in range(n)), dtype=float, count=n
            )
        except nx.NetworkXError:
            core_numbers = np.zeros(n, dtype=float)
    else:
        clustering = np.zeros(n, dtype=float)
        core_numbers = np.zeros(n, dtype=float)

    shortest_paths: np.ndarray | None = None
    if compute_shortest_paths:
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
        betweenness=betweenness,
        core_numbers=core_numbers,
        component_labels=component_labels,
        component_sizes=component_sizes,
        community_labels=community_labels,
        community_sizes=community_sizes,
        n_communities=n_communities,
        shortest_paths=shortest_paths,
        seed_distance_cache={},
    )


def graph_pair_features(
    context_1: GraphContext, context_2: GraphContext
) -> dict[str, float]:
    """Return a compact, correspondence-free description of graph difficulty."""

    mean_degree_1 = float(np.mean(context_1.degrees))
    mean_degree_2 = float(np.mean(context_2.degrees))
    degree_std_1 = float(np.std(context_1.degrees))
    degree_std_2 = float(np.std(context_2.degrees))
    degree_cv_1 = degree_std_1 / mean_degree_1 if mean_degree_1 > 0 else 0.0
    degree_cv_2 = degree_std_2 / mean_degree_2 if mean_degree_2 > 0 else 0.0
    largest_component_1 = float(
        np.max(context_1.component_sizes, initial=0) / len(context_1.degrees)
    )
    largest_component_2 = float(
        np.max(context_2.component_sizes, initial=0) / len(context_2.degrees)
    )
    sorted_degree_difference = np.abs(
        np.sort(context_1.degrees) - np.sort(context_2.degrees)
    )

    return {
        "pair_degree_mean_average": (mean_degree_1 + mean_degree_2) / 2,
        "pair_degree_cv_average": (degree_cv_1 + degree_cv_2) / 2,
        "pair_largest_component_fraction_min": min(
            largest_component_1, largest_component_2
        ),
        "pair_sorted_degree_mean_absolute_difference": float(
            np.mean(sorted_degree_difference)
        ),
    }


def distances_from_seeds(
    context: GraphContext, seeds: np.ndarray
) -> np.ndarray:
    """Compute vertex-to-seed distances without an all-pairs distance matrix."""

    if context.shortest_paths is not None:
        return context.shortest_paths[:, seeds]

    n = context.adjacency.shape[0]
    distances = np.full((n, len(seeds)), np.inf, dtype=float)
    for column, source in enumerate(seeds):
        source = int(source)
        cached = context.seed_distance_cache.get(source)
        if cached is not None:
            distances[:, column] = cached
            continue

        source_distances = np.full(n, np.inf, dtype=float)
        source_distances[source] = 0.0
        queue: deque[int] = deque([source])
        while queue:
            vertex = queue.popleft()
            next_distance = source_distances[vertex] + 1.0
            for neighbor in context.neighbors[vertex]:
                if not np.isfinite(source_distances[neighbor]):
                    source_distances[neighbor] = next_distance
                    queue.append(int(neighbor))
        context.seed_distance_cache[source] = source_distances
        distances[:, column] = source_distances
    return distances


def seed_set_features(
    context: GraphContext, seeds: np.ndarray
) -> dict[str, float]:
    """Compute the compact Graph-A feature panel for one seed set."""

    n = context.adjacency.shape[0]
    seeds = np.asarray(seeds, dtype=int)
    if seeds.ndim != 1 or not 0 < len(seeds) < n:
        raise ValueError("seeds must contain between 1 and n - 1 vertices.")
    if len(np.unique(seeds)) != len(seeds) or np.any((seeds < 0) | (seeds >= n)):
        raise ValueError("seeds must contain unique, valid vertex indices.")

    is_seed = np.zeros(n, dtype=bool)
    is_seed[seeds] = True
    nonseeds = np.flatnonzero(~is_seed)

    seed_degrees = context.degrees[seeds]
    seed_to_nonseed = context.adjacency[np.ix_(seeds, nonseeds)].astype(
        np.int32, copy=False
    )
    nonseed_signatures = seed_to_nonseed.T
    witness_counts = nonseed_signatures.sum(axis=1).astype(float)

    _, signature_counts = np.unique(
        nonseed_signatures, axis=0, return_counts=True
    )
    signature_proportions = signature_counts / len(nonseeds)
    signature_entropy = (
        float(
            -np.sum(signature_proportions * np.log(signature_proportions))
            / np.log(len(nonseeds))
        )
        if len(nonseeds) > 1
        else 0.0
    )
    collision_denominator = len(nonseeds) * (len(nonseeds) - 1)
    signature_collision_fraction = (
        float(
            np.sum(signature_counts * (signature_counts - 1))
            / collision_denominator
        )
        if collision_denominator > 0
        else 0.0
    )

    seed_nonseed_degrees = nonseed_signatures.sum(axis=0).astype(float)
    seed_overlap = nonseed_signatures.T @ nonseed_signatures
    seed_union = (
        seed_nonseed_degrees[:, None]
        + seed_nonseed_degrees[None, :]
        - seed_overlap
    )
    seed_pairs = np.triu_indices(len(seeds), k=1)
    seed_jaccard = np.divide(
        seed_overlap[seed_pairs],
        seed_union[seed_pairs],
        out=np.zeros(len(seed_pairs[0]), dtype=float),
        where=seed_union[seed_pairs] > 0,
    )

    distances = distances_from_seeds(context, seeds)
    nonseed_distances = distances[nonseeds]
    nearest_seed_distances = np.min(nonseed_distances, axis=1)
    finite_nearest_distances = nearest_seed_distances[
        np.isfinite(nearest_seed_distances)
    ]
    two_hop_seed_counts = np.sum(nonseed_distances <= 2, axis=1)
    three_hop_seed_counts = np.sum(nonseed_distances <= 3, axis=1)

    seed_pair_distances = distances[seeds][seed_pairs]
    finite_seed_pair_distances = seed_pair_distances[
        np.isfinite(seed_pair_distances)
    ]

    community_counts = np.bincount(
        context.community_labels[seeds], minlength=context.n_communities
    )
    represented_counts = community_counts[community_counts > 0]
    community_proportions = represented_counts / len(seeds)
    community_entropy = (
        float(
            -np.sum(community_proportions * np.log(community_proportions))
            / np.log(context.n_communities)
        )
        if context.n_communities > 1
        else 0.0
    )

    return {
        "seed_degree_mean": float(np.mean(seed_degrees)),
        "seed_pagerank_mean": float(np.mean(context.pagerank[seeds])),
        "seed_betweenness_mean": float(np.mean(context.betweenness[seeds])),
        "nonseed_covered_fraction": float(np.mean(witness_counts >= 1)),
        "nonseed_covered_by_two_fraction": float(np.mean(witness_counts >= 2)),
        "nonseed_covered_by_three_fraction": float(np.mean(witness_counts >= 3)),
        "nonseed_seed_neighbor_q10": float(np.quantile(witness_counts, 0.10)),
        "nonseed_nearest_seed_distance_mean": (
            float(np.mean(finite_nearest_distances))
            if finite_nearest_distances.size
            else float("nan")
        ),
        "nonseed_nearest_seed_distance_max": (
            float(np.max(finite_nearest_distances))
            if finite_nearest_distances.size
            else float("nan")
        ),
        "nonseed_nearest_seed_distance_disconnected_fraction": float(
            np.mean(~np.isfinite(nearest_seed_distances))
        ),
        "nonseed_within_two_hops_of_three_seeds_fraction": float(
            np.mean(two_hop_seed_counts >= 3)
        ),
        "nonseed_three_hop_seed_count_mean": float(
            np.mean(three_hop_seed_counts)
        ),
        "nonseed_seed_signature_normalized_entropy": signature_entropy,
        "nonseed_seed_signature_collision_pair_fraction": (
            signature_collision_fraction
        ),
        "seed_nonseed_neighborhood_jaccard_mean": (
            float(np.mean(seed_jaccard)) if seed_jaccard.size else 0.0
        ),
        "seed_pair_shortest_path_mean": (
            float(np.mean(finite_seed_pair_distances))
            if finite_seed_pair_distances.size
            else float("nan")
        ),
        "seed_community_normalized_entropy": community_entropy,
    }


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
        adjacency_1, adjacency_2, true_permutation = gen_pareto_chung_lu_graphs(
            n=config.vertex_count,
            alpha=config.paper_alpha,
            p=config.paper_probability,
            rho=config.rho,
            directed=False,
            loops=False,
        )
    elif config.graph_model == "paper_pa":
        adjacency_1, adjacency_2, true_permutation = gen_PAPER_graphs(
            n=config.vertex_count,
            alpha=config.paper_pa_alpha,
            beta=config.paper_pa_beta,
            p=config.paper_pa_probability,
            rho=config.rho,
        )
    elif config.graph_model == "ier":
        adjacency_1, adjacency_2, true_permutation = gen_IER_graphs(
            n=config.vertex_count,
            rho=config.rho,
            alpha=config.ier_alpha,
            beta=config.ier_beta,
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
        ("paper_pa_probability", config.paper_pa_probability),
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
    for name, value in (
        ("paper_alpha", config.paper_alpha),
        ("ier_alpha", config.ier_alpha),
        ("ier_beta", config.ier_beta),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    if not math.isfinite(config.paper_pa_alpha) or config.paper_pa_alpha < 0:
        raise ValueError("paper_pa_alpha must be nonnegative and finite.")
    if not math.isfinite(config.paper_pa_beta) or config.paper_pa_beta <= 0:
        raise ValueError("paper_pa_beta must be positive and finite.")
    if config.max_iter <= 0 or config.tol <= 0 or config.max_minutes <= 0:
        raise ValueError("max_iter, tol, and max_minutes must be positive.")


def row_context(
    config: ExperimentConfig,
    *,
    graph_pair_key: str,
    graph_pair_id: int,
    candidate_id: int,
    graph_pair_seed: int,
    sgm_seed: int,
    seeds_1: np.ndarray,
    seeds_2: np.ndarray,
) -> dict[str, Any]:
    """Build the identifiers and graph-generation settings saved on each row."""

    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "graph_model": config.graph_model,
        "graph_pair_key": graph_pair_key,
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
            config.er_probability if config.graph_model == "sparse_er" else None
        ),
        "sbm_n_blocks": (
            config.n_blocks if config.graph_model == "sparse_sbm" else None
        ),
        "sbm_n_per_block": (
            config.n_per_block if config.graph_model == "sparse_sbm" else None
        ),
        "sbm_block_probabilities": (
            json.dumps(config.sbm_block_probabilities)
            if config.graph_model == "sparse_sbm"
            else None
        ),
        "paper_alpha": (
            config.paper_alpha if config.graph_model == "paper" else None
        ),
        "paper_probability": (
            config.paper_probability if config.graph_model == "paper" else None
        ),
        "paper_pa_alpha": (
            config.paper_pa_alpha if config.graph_model == "paper_pa" else None
        ),
        "paper_pa_beta": (
            config.paper_pa_beta if config.graph_model == "paper_pa" else None
        ),
        "paper_pa_probability": (
            config.paper_pa_probability if config.graph_model == "paper_pa" else None
        ),
        "ier_alpha": config.ier_alpha if config.graph_model == "ier" else None,
        "ier_beta": config.ier_beta if config.graph_model == "ier" else None,
    }


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
    required_columns = {
        "graph_pair_key",
        "graph_pair_id",
        "candidate_id",
        "graph_pair_seed",
        "sgm_seed",
        "seed_vertices_graph1",
        "seed_vertices_graph2",
    } | set(SGM_RESULT_COLUMNS)
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
        context_2 = build_graph_context(adjacency_2, compute_seed_features=False)
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

            refreshed_row = row_context(
                config,
                graph_pair_key=str(saved_row["graph_pair_key"]),
                graph_pair_id=int(saved_row["graph_pair_id"]),
                candidate_id=int(saved_row["candidate_id"]),
                graph_pair_seed=graph_pair_seed,
                sgm_seed=int(saved_row["sgm_seed"]),
                seeds_1=seeds_1,
                seeds_2=seeds_2,
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
    metadata["config"] = asdict(config)
    metadata["columns"] = list(refreshed_frame.columns)
    metadata["rows_written"] = len(refreshed_frame)
    metadata["features_refreshed_without_sgm"] = True
    metadata["prequery_feature_prefixes"] = ["pair_", "seed_", "nonseed_"]
    metadata["exclude_from_prequery_models"] = [
        "feature_schema_version",
        "graph_pair_key",
        "graph_pair_id",
        "candidate_id",
        "graph_pair_seed",
        "sgm_seed",
        "seed_vertices_graph1",
        "seed_vertices_graph2",
        *SGM_RESULT_COLUMNS,
    ]

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
            context_2 = build_graph_context(
                adjacency_2, compute_seed_features=False
            )
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

                row = row_context(
                    config,
                    graph_pair_key=f"{config.graph_model}_{graph_pair_id}",
                    graph_pair_id=graph_pair_id,
                    candidate_id=candidate_id,
                    graph_pair_seed=graph_pair_seed,
                    sgm_seed=sgm_seed,
                    seeds_1=seeds_1,
                    seeds_2=seeds_2,
                )
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
        "prequery_feature_prefixes": ["pair_", "seed_", "nonseed_"],
        "exclude_from_prequery_models": [
            "feature_schema_version",
            "graph_pair_key",
            "graph_pair_id",
            "candidate_id",
            "graph_pair_seed",
            "sgm_seed",
            "seed_vertices_graph1",
            "seed_vertices_graph2",
            *SGM_RESULT_COLUMNS,
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
        default=DEFAULT_OUTPUT_DIR,
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
            if args.graph_model == "all" and not output_path.exists():
                print(f"Skipping {output_path}; no existing dataset was found.")
                continue
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
