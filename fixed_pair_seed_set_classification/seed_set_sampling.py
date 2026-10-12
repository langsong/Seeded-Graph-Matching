"""Outcome-blind seed-set sampling for fixed-pair experiments.

Sampler version 2 (2026-10-06):
- rule-based starting sets break ties at random (``rng``) instead of by vertex
  index;
- degree-controlled contrasts are matched on seed-to-nonseed ("clue-based")
  degree, the degree feature used in the analysis;
- contrast pairs are drawn only from the degree range of the uniform random
  sets (10th-90th percentile by default), split into equal-count bins;
- both sets of a contrast share an explicit ``contrast_pair_id``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiment_common.features import (
    GraphContext,
    distance_signature_unique_fraction,
    distances_from_seeds,
)


SAMPLER_VERSION = 2
DEGREE_GAMMAS = (-1.0, -0.5, 0.5, 1.0, 2.0, 3.0)
# The contrast pool also draws uniform sets, which populate the random-set range.
POOL_GAMMAS = (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)
CONTRASTS = ("coverage", "pair_mass", "unique_signatures_r3")
# Column of each contrast feature in the array returned by _sampler_features.
CONTRAST_COLUMNS = {"coverage": 1, "pair_mass": 2, "unique_signatures_r3": 3}
DEFAULT_SAMPLING_COUNTS = {
    "n_random_train": 200,
    "n_random_test": 100,
    "n_degree_weighted": 100,
    "n_swap_paths": 0,
    "n_decorrelated": 100,
}


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
    contrast_pair_id: int | None = None


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


def _top_k(values: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of the k largest values; ties are broken at random."""

    return np.lexsort((rng.random(len(values)), -np.asarray(values)))[:k]


def _greedy_seeds(
    adjacency: np.ndarray,
    degrees: np.ndarray,
    k: int,
    rule: str,
    rng: np.random.Generator,
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
        # Degree breaks ties (including the first step of pair-mass selection);
        # remaining ties are broken at random rather than by vertex index.
        order = np.lexsort((rng.random(len(remaining)), degrees[remaining], score))
        vertex = int(remaining[order[-1]])
        result.append(vertex)
        chosen[vertex] = True
        covered |= adjacency[vertex].astype(bool)
        witness += adjacency[vertex]
    return np.sort(np.asarray(result, dtype=int))


def _rule_starts(
    context: GraphContext, k: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
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
    top_degree = _top_k(context.degrees, k, rng)
    top_harmonic = _top_k(harmonic, k, rng)
    main_component = context.component_labels == int(np.argmax(context.component_sizes))
    if np.sum(main_component) < k:
        raise ValueError("The largest component has fewer vertices than the seed budget.")
    members = np.flatnonzero(main_component)
    member_degrees = context.degrees[members]
    spread = [int(rng.choice(members[member_degrees == member_degrees.max()]))]
    nearest = distances[:, spread[0]].copy()
    while len(spread) < k:
        nearest[~main_component] = -1
        nearest[spread] = -1
        vertex = int(rng.choice(np.flatnonzero(nearest == nearest.max())))
        spread.append(vertex)
        nearest = np.minimum(nearest, distances[:, vertex])
    return {
        "top_degree": np.sort(top_degree),
        "top_harmonic": np.sort(top_harmonic),
        "greedy_coverage": _greedy_seeds(adjacency, context.degrees, k, "coverage", rng),
        "greedy_pair_mass": _greedy_seeds(adjacency, context.degrees, k, "pair_mass", rng),
        "greedy_signatures": _greedy_seeds(adjacency, context.degrees, k, "signatures", rng),
        "spread_out": np.sort(np.asarray(spread, dtype=int)),
    }


def _sampler_features(
    context: GraphContext, seeds: np.ndarray
) -> tuple[float, float, float, float]:
    """Clue-based degree, coverage, pair mass and r3 signature uniqueness.

    The first value equals ``seed_nonseed_degree_mean`` in the feature panel:
    seed-to-nonseed edges divided by the number of seeds.
    """
    n = len(context.degrees)
    nonseed = np.ones(n, dtype=bool)
    nonseed[seeds] = False
    witnesses = context.adjacency[np.ix_(nonseed, seeds)].sum(axis=1).astype(float)
    unique_fraction = distance_signature_unique_fraction(
        context, seeds, np.flatnonzero(nonseed)
    )
    return (
        float(witnesses.sum() / len(seeds)),
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
    """Find the largest feature contrast at equal seed-to-nonseed edge count."""

    available = indices[~used[indices]]
    if len(available) < 2:
        raise ValueError(
            "A contrast degree bin ran out of pool sets; increase --pool-size "
            "or reduce --decorrelated."
        )
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
    n_random_train: int = DEFAULT_SAMPLING_COUNTS["n_random_train"],
    n_random_test: int = DEFAULT_SAMPLING_COUNTS["n_random_test"],
    n_degree_weighted: int = DEFAULT_SAMPLING_COUNTS["n_degree_weighted"],
    n_swap_paths: int = DEFAULT_SAMPLING_COUNTS["n_swap_paths"],
    n_decorrelated: int = DEFAULT_SAMPLING_COUNTS["n_decorrelated"],
    pool_size: int = 4000,
    contrast_features: tuple[str, ...] = CONTRASTS,
    n_contrast_bins: int = 6,
    contrast_degree_percentiles: tuple[float, float] = (10.0, 90.0),
    sampling_info: dict | None = None,
) -> list[SeedCandidate]:
    """Keep a random test set and cover seed sets that simple rules would pick.

    The default 500-set design contains 200 random training sets, 100 random
    test sets, 100 degree-weighted sets, and 100 degree-controlled contrast
    sets. Swap paths remain available as an optional sampling group.

    ``contrast_features`` lists the features that get degree-controlled
    contrast pairs; repeating a name gives that feature a larger share.
    Contrast pairs come from ``n_contrast_bins`` equal-count bins of
    seed-to-nonseed degree inside the ``contrast_degree_percentiles`` range of
    the uniform random sets (or of the pool, if there are no random sets).
    If ``sampling_info`` is a dict, the contrast range and bin edges are
    written into it for the metadata.
    """

    if min(n_random_train, n_random_test, n_degree_weighted, n_swap_paths,
           n_decorrelated) < 0:
        raise ValueError("Sample counts must be nonnegative.")
    if n_decorrelated > pool_size:
        raise ValueError("The decorrelation pool must contain at least as many sets as selected.")
    if n_decorrelated % 2:
        raise ValueError("n_decorrelated must be even: contrasts come in pairs.")
    unknown = set(contrast_features) - set(CONTRAST_COLUMNS)
    if unknown or not contrast_features:
        raise ValueError(f"contrast_features must be drawn from {CONTRASTS}.")
    if n_contrast_bins < 1:
        raise ValueError("n_contrast_bins must be positive.")
    low_percentile, high_percentile = contrast_degree_percentiles
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("contrast_degree_percentiles must satisfy 0 <= low < high <= 100.")
    info = sampling_info if sampling_info is not None else {}
    info["sampler_version"] = SAMPLER_VERSION
    n = len(context.degrees)
    seen: set[tuple[int, ...]] = set()
    candidates: list[SeedCandidate] = []
    for split, count in (("test", n_random_test), ("train", n_random_train)):
        for _ in range(count):
            candidates.append(SeedCandidate(_draw(rng, n, k, seen), split, "random"))

    probabilities = {
        gamma: _degree_probabilities(context.degrees, gamma)
        for gamma in sorted(set(DEGREE_GAMMAS) | set(POOL_GAMMAS))
    }
    for index in range(n_degree_weighted):
        gamma = DEGREE_GAMMAS[index % len(DEGREE_GAMMAS)]
        candidates.append(SeedCandidate(
            _draw(rng, n, k, seen, probabilities[gamma]),
            "train", "degree_weighted", degree_gamma=gamma,
        ))

    if n_swap_paths:
        starts = _rule_starts(context, k, rng)
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
            gamma = POOL_GAMMAS[index % len(POOL_GAMMAS)]
            seeds = _draw(rng, n, k, pool_seen, probabilities[gamma])
            pool.append(seeds)
            measurements[index] = _sampler_features(context, seeds)

        # Restrict contrasts to the degree range of typical (uniform random)
        # sets, where the calibrated budget puts success near 50%.
        random_degrees = np.array([
            _sampler_features(context, candidate.seeds)[0]
            for candidate in candidates if candidate.strategy == "random"
        ])
        reference = random_degrees if len(random_degrees) else measurements[:, 0]
        degree_low, degree_high = np.percentile(
            reference, [low_percentile, high_percentile]
        )
        in_range = np.flatnonzero(
            (measurements[:, 0] >= degree_low) & (measurements[:, 0] <= degree_high)
        )
        if len(in_range) < max(n_decorrelated, 2 * n_contrast_bins):
            raise ValueError(
                f"Only {len(in_range)} pool sets fall in the contrast degree range; "
                "increase --pool-size."
            )
        degree_bins = np.full(pool_size, -1, dtype=int)
        ordered_in_range = in_range[np.argsort(measurements[in_range, 0], kind="stable")]
        bin_members = np.array_split(ordered_in_range, n_contrast_bins)
        for bin_id, indices in enumerate(bin_members):
            degree_bins[indices] = bin_id
        info.update({
            "contrast_features": list(contrast_features),
            "contrast_degree_feature": "seed_nonseed_degree_mean",
            "contrast_degree_reference": (
                "uniform_random_sets" if len(random_degrees) else "contrast_pool"
            ),
            "contrast_degree_percentiles": [low_percentile, high_percentile],
            "contrast_degree_range": [float(degree_low), float(degree_high)],
            "contrast_bin_edges": [
                float(measurements[indices[0], 0]) for indices in bin_members
            ] + [float(measurements[bin_members[-1][-1], 0])],
            "contrast_pool_sets_in_range": int(len(in_range)),
            "contrast_pool_gammas": list(POOL_GAMMAS),
        })

        # Give each degree bin both tails of every feature. Extra pairs rotate
        # among features so the selected sets cover all bins evenly.
        n_features = len(contrast_features)
        bin_feature_pairs = [
            (bin_id, (bin_id + offset) % n_features)
            for offset in range(n_features) for bin_id in range(n_contrast_bins)
        ]
        base, extra = divmod(n_decorrelated // 2, len(bin_feature_pairs))
        used = np.zeros(pool_size, dtype=bool)
        pair_id = 0
        for pair_index, (bin_id, feature_index) in enumerate(bin_feature_pairs):
            indices = np.flatnonzero(degree_bins == bin_id)
            feature = contrast_features[feature_index]
            column = CONTRAST_COLUMNS[feature]
            for _ in range(base + (pair_index < extra)):
                low, high = _matched_contrast_pair(
                    measurements, indices, column, used, k
                )
                for tail, pool_index in (("low", low), ("high", high)):
                    used[pool_index] = True
                    seen.add(_key(pool[pool_index]))
                    candidates.append(SeedCandidate(
                        pool[pool_index], "train", "degree_controlled",
                        contrast_feature=feature, contrast_tail=tail,
                        contrast_degree_bin=bin_id, contrast_pair_id=pair_id,
                    ))
                pair_id += 1

    return [candidate for candidate in candidates if candidate.split == "train"] + [
        candidate for candidate in candidates if candidate.split == "test"
    ]
