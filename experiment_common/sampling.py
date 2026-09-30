"""Shared random seed-set sampling helpers."""

from __future__ import annotations

import numpy as np

UINT32_MAX = np.iinfo(np.uint32).max


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
