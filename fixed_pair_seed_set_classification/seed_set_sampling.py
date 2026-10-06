"""Outcome-blind seed-set sampling for fixed-pair experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiment_common.features import (
    GraphContext,
    distance_signature_unique_fraction,
    distances_from_seeds,
)


DEGREE_GAMMAS = (-1.0, -0.5, 0.5, 1.0, 2.0, 3.0)
CONTRASTS = ("coverage", "pair_mass", "unique_signatures_r3")


@dataclass
class SeedCandidate:
    seeds: np.ndarray
    split: str
    strategy: str
    degree_gamma: float | None = None
    starting_rule: str | None = None
    swap_count: int | None = None
    contrast_feature: str | None = None
    contrast_tail: str | None = None
    contrast_degree_bin: int | None = None


def _key(seeds: np.ndarray) -> tuple[int, ...]:
    return tuple(int(vertex) for vertex in seeds)


def _draw(
    rng: np.random.Generator,
    n: int,
    k: int,
    seen: set[tuple[int, ...]],
    probabilities: np.ndarray | None = None,
) -> np.ndarray:
    while True:
        seeds = np.sort(rng.choice(n, size=k, replace=False, p=probabilities))
        key = _key(seeds)
        if key not in seen:
            seen.add(key)
            return seeds


def _degree_probabilities(degrees: np.ndarray, gamma: float) -> np.ndarray:
    weights = np.maximum(degrees, 1.0) ** gamma
    return weights / weights.sum()


def _greedy_seeds(
    adjacency: np.ndarray, degrees: np.ndarray, k: int, rule: str
) -> np.ndarray:
    n = len(degrees)
    chosen = np.zeros(n, dtype=bool)
    covered = np.zeros(n, dtype=bool)
    witness = np.zeros(n, dtype=float)
    result = []
    for _ in range(k):
        remaining = np.flatnonzero(~chosen)
        if rule == "coverage":
            score = adjacency[(~covered) & (~chosen)][:, remaining].sum(axis=0)
        elif rule == "pair_mass":
            score = witness[remaining] @ adjacency[np.ix_(remaining, remaining)]
        else:
            nonseeds = remaining
            if result:
                _, groups = np.unique(
                    adjacency[np.ix_(nonseeds, result)],
                    axis=0,
                    return_inverse=True,
                )
            else:
                groups = np.zeros(len(nonseeds), dtype=int)
            score = np.zeros(len(remaining), dtype=float)
            for group in np.unique(groups):
                members = nonseeds[groups == group]
                neighbors = adjacency[np.ix_(members, remaining)].sum(axis=0)
                score += neighbors * (len(members) - neighbors)
        # Degree breaks ties, including the first step of pair-mass selection.
        vertex = int(remaining[np.lexsort((degrees[remaining], score))[-1]])
        result.append(vertex)
        chosen[vertex] = True
        covered |= adjacency[vertex].astype(bool)
        witness += adjacency[vertex]
    return np.sort(np.asarray(result, dtype=int))


def _rule_starts(context: GraphContext, k: int) -> dict[str, np.ndarray]:
    n = len(context.degrees)
    adjacency = context.adjacency.astype(float)
    distances = distances_from_seeds(context, np.arange(n))
    inverse_distances = np.divide(
        1.0,
        distances,
        out=np.zeros_like(distances),
        where=(distances > 0) & np.isfinite(distances),
    )
    harmonic = inverse_distances.sum(axis=1)
    top_degree = np.argsort(-context.degrees, kind="stable")[:k]
    main_component = context.component_labels == int(np.argmax(context.component_sizes))
    if np.sum(main_component) < k:
        raise ValueError("The largest component has fewer vertices than the seed budget.")
    spread = [int(np.flatnonzero(main_component)[
        np.argmax(context.degrees[main_component])
    ])]
    nearest = distances[:, spread[0]].copy()
    while len(spread) < k:
        nearest[~main_component] = -1
        nearest[spread] = -1
        vertex = int(np.argmax(nearest))
        spread.append(vertex)
        nearest = np.minimum(nearest, distances[:, vertex])
    return {
        "top_degree": np.sort(top_degree),
        "top_harmonic": np.sort(np.argsort(-harmonic, kind="stable")[:k]),
        "greedy_coverage": _greedy_seeds(adjacency, context.degrees, k, "coverage"),
        "greedy_pair_mass": _greedy_seeds(adjacency, context.degrees, k, "pair_mass"),
        "greedy_signatures": _greedy_seeds(adjacency, context.degrees, k, "signatures"),
        "spread_out": np.sort(np.asarray(spread, dtype=int)),
    }


def _sampler_features(
    context: GraphContext, seeds: np.ndarray
) -> tuple[float, float, float, float]:
    n = len(context.degrees)
    nonseed = np.ones(n, dtype=bool)
    nonseed[seeds] = False
    witnesses = context.adjacency[np.ix_(nonseed, seeds)].sum(axis=1).astype(float)
    unique_fraction = distance_signature_unique_fraction(
        context, seeds, np.flatnonzero(nonseed)
    )
    return (
        float(context.degrees[seeds].mean()),
        float(np.mean(witnesses > 0)),
        float(np.mean(witnesses * (witnesses - 1) / 2)),
        unique_fraction,
    )


def _matched_contrast_pair(
    measurements: np.ndarray,
    indices: np.ndarray,
    column: int,
    used: np.ndarray,
    k: int,
) -> tuple[int, int]:
    """Find the largest feature contrast at equal total seed degree in a bin."""

    available = indices[~used[indices]]
    degree_sums = np.rint(measurements[available, 0] * k).astype(int)
    best: tuple[int, int] | None = None
    best_gap = -1.0
    for degree_sum in np.unique(degree_sums):
        group = available[degree_sums == degree_sum]
        if len(group) < 2:
            continue
        ordered = group[np.argsort(measurements[group, column], kind="stable")]
        gap = measurements[ordered[-1], column] - measurements[ordered[0], column]
        if gap > best_gap:
            best = (int(ordered[0]), int(ordered[-1]))
            best_gap = gap
    if best is not None:
        return best

    # A tiny custom pool may have no repeated degree sum in a bin.
    ordered = available[np.argsort(measurements[available, 0], kind="stable")]
    differences = np.diff(measurements[ordered, 0])
    nearest = np.flatnonzero(differences == differences.min())
    position = int(nearest[np.argmax([
        abs(measurements[ordered[i + 1], column] - measurements[ordered[i], column])
        for i in nearest
    ])])
    pair = (int(ordered[position]), int(ordered[position + 1]))
    return tuple(sorted(pair, key=lambda i: measurements[i, column]))


def sample_seed_sets(
    context: GraphContext,
    rng: np.random.Generator,
    k: int,
    *,
    n_random_train: int = 80,
    n_random_test: int = 40,
    n_degree_weighted: int = 160,
    n_swap_paths: int = 80,
    n_decorrelated: int = 40,
    pool_size: int = 4000,
) -> list[SeedCandidate]:
    """Keep a random test set and cover seed sets that simple rules would pick."""

    if min(n_random_train, n_random_test, n_degree_weighted, n_swap_paths,
           n_decorrelated) < 0:
        raise ValueError("Sample counts must be nonnegative.")
    if n_decorrelated > pool_size:
        raise ValueError("The decorrelation pool must contain at least as many sets as selected.")
    n = len(context.degrees)
    seen: set[tuple[int, ...]] = set()
    candidates: list[SeedCandidate] = []
    for split, count in (("test", n_random_test), ("train", n_random_train)):
        for _ in range(count):
            candidates.append(SeedCandidate(_draw(rng, n, k, seen), split, "random"))

    probabilities = {
        gamma: _degree_probabilities(context.degrees, gamma)
        for gamma in DEGREE_GAMMAS
    }
    for index in range(n_degree_weighted):
        gamma = DEGREE_GAMMAS[index % len(DEGREE_GAMMAS)]
        candidates.append(SeedCandidate(
            _draw(rng, n, k, seen, probabilities[gamma]),
            "train", "degree_weighted", degree_gamma=gamma,
        ))

    if n_swap_paths:
        starts = _rule_starts(context, k)
        levels = tuple(dict.fromkeys(min(k, level) for level in (2, 5, 10, k // 2)))
        rule_levels = [(name, level) for name in starts for level in levels if level]
        swaps_added = 0
        for name, base in starts.items():
            key = _key(base)
            if swaps_added < n_swap_paths and key not in seen:
                seen.add(key)
                candidates.append(SeedCandidate(base, "train", "swap_path", starting_rule=name, swap_count=0))
                swaps_added += 1
        outside = {
            name: np.setdiff1d(np.arange(n), base) for name, base in starts.items()
        }
        index = 0
        while swaps_added < n_swap_paths:
            name, level = rule_levels[index % len(rule_levels)]
            base = starts[name]
            retained = rng.choice(base, size=k - level, replace=False)
            added = rng.choice(outside[name], size=level, replace=False)
            seeds = np.sort(np.concatenate((retained, added)))
            key = _key(seeds)
            if key not in seen:
                seen.add(key)
                candidates.append(SeedCandidate(
                    seeds, "train", "swap_path", starting_rule=name,
                    swap_count=level,
                ))
                swaps_added += 1
            index += 1

    if n_decorrelated:
        pool: list[np.ndarray] = []
        measurements = np.empty((pool_size, 4), dtype=float)
        pool_seen = seen.copy()
        for index in range(pool_size):
            gamma = DEGREE_GAMMAS[index % len(DEGREE_GAMMAS)]
            seeds = _draw(rng, n, k, pool_seen, probabilities[gamma])
            pool.append(seeds)
            measurements[index] = _sampler_features(context, seeds)

        degree_bins = np.empty(pool_size, dtype=int)
        for bin_id, indices in enumerate(
            np.array_split(np.argsort(measurements[:, 0], kind="stable"), 10)
        ):
            degree_bins[indices] = bin_id
        # Give each degree bin both tails of every feature. Extra pairs rotate
        # among features so the 100 selected sets cover all ten bins evenly.
        bin_feature_pairs = [
            (bin_id, (bin_id + offset) % len(CONTRASTS))
            for offset in range(len(CONTRASTS)) for bin_id in range(10)
        ]
        base, extra = divmod(n_decorrelated, 2 * len(bin_feature_pairs))
        used = np.zeros(pool_size, dtype=bool)
        for pair_index, (bin_id, feature_index) in enumerate(bin_feature_pairs):
            indices = np.flatnonzero(degree_bins == bin_id)
            column = feature_index + 1
            pair_count = base + (pair_index < extra // 2)
            for _ in range(pair_count):
                low, high = _matched_contrast_pair(
                    measurements, indices, column, used, k
                )
                for tail, pool_index in (("low", low), ("high", high)):
                    used[pool_index] = True
                    seeds = pool[pool_index]
                    seen.add(_key(seeds))
                    candidates.append(SeedCandidate(
                        seeds, "train", "degree_controlled",
                        contrast_feature=CONTRASTS[feature_index],
                        contrast_tail=tail, contrast_degree_bin=bin_id,
                    ))
            if pair_index == extra // 2 and extra % 2:
                available = indices[~used[indices]]
                pool_index = int(available[np.argmin(measurements[available, column])])
                used[pool_index] = True
                seeds = pool[pool_index]
                seen.add(_key(seeds))
                candidates.append(SeedCandidate(
                    seeds, "train", "degree_controlled",
                    contrast_feature=CONTRASTS[feature_index],
                    contrast_tail="low", contrast_degree_bin=bin_id,
                ))

    return [candidate for candidate in candidates if candidate.split == "train"] + [
        candidate for candidate in candidates if candidate.split == "test"
    ]
