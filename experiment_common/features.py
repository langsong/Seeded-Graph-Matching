"""Graph and seed-set features shared by the experiments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np

FEATURE_SCHEMA_VERSION = 3


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
