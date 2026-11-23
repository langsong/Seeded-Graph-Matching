"""Seed-selection strategies, each returned as an ordering of Graph A's vertices.

The seed set at budget k is the first k vertices of the ordering, so sets are
nested across budgets (buying queries one at a time). All strategies use
Graph A only.

Degree-first strategies choose seed j from the ``pool_factor * j`` highest-degree
unused vertices, taking the one that most improves their target feature, so every
prefix stays close to top degree and differences come from the feature itself.
Ties go to the higher-degree vertex; equal degrees are broken by a random key.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

SIGNATURE_CAPS = (3, 2)  # lexicographic target: unique fingerprints at cap 3, then cap 2


class GraphA:
    """Graph-A quantities shared by the strategies and the feature summary."""

    def __init__(self, adjacency: np.ndarray):
        adjacency = (np.asarray(adjacency) > 0).astype(np.int8)
        self.adjacency = adjacency
        self.n = adjacency.shape[0]
        self.degree = adjacency.sum(axis=1).astype(np.int64)
        self.neighbors = [np.flatnonzero(adjacency[v]) for v in range(self.n)]
        distances = shortest_path(csr_matrix(adjacency), method="D", unweighted=True, directed=False)
        reciprocal = np.zeros_like(distances)
        reachable = np.isfinite(distances) & (distances > 0)
        reciprocal[reachable] = 1.0 / distances[reachable]
        self.harmonic = reciprocal.sum(axis=1)
        capped = np.where(np.isfinite(distances), distances, np.inf)
        self.capped = {cap: np.minimum(capped, cap).astype(np.int16) for cap in (2, 3, 4)}

    def degree_order(self, rng: np.random.Generator) -> np.ndarray:
        """Vertices by decreasing degree, equal degrees in random order."""

        return np.lexsort((rng.random(self.n), -self.degree))


# --------------------------------------------------------------------------
# Incremental targets for the degree-first greedy rules
# --------------------------------------------------------------------------

class _WitnessState:
    def __init__(self, graph: GraphA):
        self.graph = graph
        self.witnesses = np.zeros(graph.n, dtype=np.int64)  # seed neighbours of each vertex
        self.in_set = np.zeros(graph.n, dtype=bool)

    def add(self, vertex: int) -> None:
        self.witnesses[self.graph.neighbors[vertex]] += 1
        self.in_set[vertex] = True


class PairMassTarget(_WitnessState):
    """Sum over non-seeds of C(w, 2), w = number of seed neighbours (larger = clustered)."""

    def score(self, vertex: int) -> tuple:
        neighbours = self.graph.neighbors[vertex]
        neighbours = neighbours[~self.in_set[neighbours]]
        w_v = self.witnesses[vertex]
        return (int(self.witnesses[neighbours].sum() - w_v * (w_v - 1) // 2),)


class CoverageTarget(_WitnessState):
    """Number of non-seeds with at least one seed neighbour (larger = spread)."""

    def score(self, vertex: int) -> tuple:
        neighbours = self.graph.neighbors[vertex]
        neighbours = neighbours[~self.in_set[neighbours]]
        newly_covered = int(np.sum(self.witnesses[neighbours] == 0))
        return (newly_covered - int(self.witnesses[vertex] >= 1),)


class SignatureTarget:
    """Non-seeds whose capped seed-distance fingerprint is unique: cap 3, then cap 2."""

    def __init__(self, graph: GraphA):
        self.graph = graph
        self.in_set = np.zeros(graph.n, dtype=bool)
        self.labels = {cap: np.zeros(graph.n, dtype=np.int64) for cap in SIGNATURE_CAPS}

    def _unique_after(self, cap: int, vertex: int) -> int:
        keys = self.labels[cap] * (cap + 1) + self.graph.capped[cap][:, vertex]
        mask = ~self.in_set
        mask[vertex] = False
        _, counts = np.unique(keys[mask], return_counts=True)
        return int(np.sum(counts == 1))

    def score(self, vertex: int) -> tuple:
        return tuple(self._unique_after(cap, vertex) for cap in SIGNATURE_CAPS)

    def add(self, vertex: int) -> None:
        for cap in SIGNATURE_CAPS:
            keys = self.labels[cap] * (cap + 1) + self.graph.capped[cap][:, vertex]
            _, self.labels[cap] = np.unique(keys, return_inverse=True)
        self.in_set[vertex] = True


TARGETS = {"clustered": PairMassTarget, "spread": CoverageTarget, "distinguishing": SignatureTarget}


def degree_first(graph: GraphA, k_max: int, rng: np.random.Generator, target: str,
                 *, pool_factor: int = 2, exclude: np.ndarray | None = None) -> np.ndarray:
    """Greedy ordering: seed j maximises the target among the pool_factor*j top-degree unused vertices."""

    excluded = set() if exclude is None else {int(v) for v in exclude}
    by_degree = [int(v) for v in graph.degree_order(rng) if int(v) not in excluded]
    state = TARGETS[target](graph)
    chosen: list[int] = []
    chosen_set: set[int] = set()
    for step in range(1, k_max + 1):
        pool = [v for v in by_degree if v not in chosen_set][: pool_factor * step]
        if not pool:
            raise ValueError("Ran out of candidate vertices.")
        # max over (score, earlier in the degree order); ties go to higher degree.
        best = max(range(len(pool)), key=lambda i: (state.score(pool[i]), -i))
        vertex = pool[best]
        chosen.append(vertex)
        chosen_set.add(vertex)
        state.add(vertex)
    return np.asarray(chosen, dtype=int)


# --------------------------------------------------------------------------
# Strategy registry
# --------------------------------------------------------------------------

def ordering(strategy: str, graph: GraphA, k_max: int, rng: np.random.Generator,
             *, pool_factor: int = 2, hub_count: int = 10) -> np.ndarray:
    """First k_max vertices of a strategy's ordering."""

    if strategy == "random":
        return rng.permutation(graph.n)[:k_max]
    if strategy == "top_degree":
        return graph.degree_order(rng)[:k_max]
    if strategy == "top_harmonic":
        return np.lexsort((rng.random(graph.n), -graph.harmonic))[:k_max]
    if strategy in TARGETS:
        return degree_first(graph, k_max, rng, strategy, pool_factor=pool_factor)
    if strategy == "hub_free_distinguishing":
        hubs = graph.degree_order(rng)[:hub_count]
        return degree_first(graph, k_max, rng, "distinguishing", pool_factor=pool_factor, exclude=hubs)
    raise ValueError(f"Unknown strategy: {strategy}")


STRATEGY_CODES = {name: i for i, name in enumerate(
    ("random", "top_degree", "top_harmonic", "clustered", "spread", "distinguishing", "hub_free_distinguishing"))}


# --------------------------------------------------------------------------
# Core features of a seed set (same definitions as experiment_common.features)
# --------------------------------------------------------------------------

def seed_set_features(graph: GraphA, seeds: np.ndarray) -> dict[str, float]:
    seeds = np.asarray(seeds, dtype=int)
    mask = np.ones(graph.n, dtype=bool)
    mask[seeds] = False
    nonseeds = np.flatnonzero(mask)
    witnesses = graph.adjacency[np.ix_(nonseeds, seeds)].sum(axis=1).astype(np.int64)
    features = {
        "f_degree": float(witnesses.sum() / len(seeds)),
        "f_harmonic": float(graph.harmonic[seeds].mean()),
        "f_coverage": float(np.mean(witnesses > 0)),
        "f_pair_mass": float(np.mean(witnesses * (witnesses - 1) / 2)),
    }
    for cap in (2, 3):
        signatures = graph.capped[cap][np.ix_(nonseeds, seeds)]
        _, counts = np.unique(signatures, axis=0, return_counts=True)
        features[f"f_signature_r{cap}"] = float(np.sum(counts == 1) / len(nonseeds))
    return features
