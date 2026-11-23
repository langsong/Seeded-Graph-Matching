"""Structure checks for generated graph pairs: one component and no automorphisms.

``check_graph_pair`` rejects a pair unless both graphs are connected and colour
refinement (1-dimensional Weisfeiler-Leman, started from vertex degrees) gives
every vertex of each graph its own colour. A discrete refinement proves that the
graph has no non-identity permutation P with P A P^T = A, so it is an exact,
millisecond replacement for searching for automorphisms with repeated
self-matching runs. Vertices that end up sharing a colour are the only ones an
automorphism could swap; ``n_unresolved`` counts them.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def is_connected(adjacency: np.ndarray) -> bool:
    n_components = connected_components(
        sp.csr_matrix(adjacency), directed=False, return_labels=False
    )
    return n_components == 1


def color_refinement(adjacency: np.ndarray, max_rounds: int = 100) -> np.ndarray:
    """Stable colour-refinement colour of every vertex (integers)."""

    adjacency = np.asarray(adjacency)
    neighbors = [np.flatnonzero(row) for row in adjacency]
    colors = adjacency.sum(axis=1).astype(np.int64)
    n_colors = len(np.unique(colors))
    for _ in range(max_rounds):
        signatures = [
            (int(colors[vertex]),) + tuple(sorted(colors[neighbors[vertex]].tolist()))
            for vertex in range(len(neighbors))
        ]
        table: dict[tuple[int, ...], int] = {}
        refined = np.array(
            [table.setdefault(signature, len(table)) for signature in signatures],
            dtype=np.int64,
        )
        if len(table) == n_colors:
            break
        colors, n_colors = refined, len(table)
    return colors


def graph_structure(adjacency: np.ndarray) -> dict[str, float | int | bool]:
    colors = color_refinement(adjacency)
    _, counts = np.unique(colors, return_counts=True)
    degrees = np.asarray(adjacency).sum(axis=1)
    return {
        "connected": is_connected(adjacency),
        "n_unresolved": int(counts[counts > 1].sum()),
        "mean_degree": float(degrees.mean()),
        "min_degree": int(degrees.min()),
    }


def check_graph_pair(
    adjacency_1: np.ndarray, adjacency_2: np.ndarray, max_unresolved: int = 0
) -> dict[str, float | int | bool]:
    """Pass only if both graphs are connected and have at most ``max_unresolved``
    vertices left sharing a refinement colour (0 means provably no automorphisms)."""

    first, second = graph_structure(adjacency_1), graph_structure(adjacency_2)
    passes = (
        first["connected"]
        and second["connected"]
        and first["n_unresolved"] <= max_unresolved
        and second["n_unresolved"] <= max_unresolved
    )
    report = {"passes": bool(passes)}
    report.update({f"graph1_{key}": value for key, value in first.items()})
    report.update({f"graph2_{key}": value for key, value in second.items()})
    return report


def first_passing_pair(
    generate: Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    first_seed: int,
    max_tries: int = 200,
    max_unresolved: int = 0,
):
    """Try graph_pair_seed = first_seed, first_seed + 1, ... until a pair passes.

    ``generate(seed)`` must return (adjacency_1, adjacency_2, true_permutation), e.g.
    ``lambda seed: generate_graph_pair(config, seed)``. Returns
    (seed, adjacency_1, adjacency_2, true_permutation, report, n_rejected).
    """

    for offset in range(max_tries):
        seed = first_seed + offset
        adjacency_1, adjacency_2, true_permutation = generate(seed)
        report = check_graph_pair(adjacency_1, adjacency_2, max_unresolved)
        if report["passes"]:
            return seed, adjacency_1, adjacency_2, true_permutation, report, offset
    raise RuntimeError(f"No passing pair in {max_tries} seeds from {first_seed}.")
