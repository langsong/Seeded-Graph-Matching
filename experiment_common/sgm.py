"""Single-run SGM evaluation shared by the experiments."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from graspologic.match import graph_match
import graspologic.match.wrappers as match_wrappers

if not hasattr(match_wrappers, "_GraphMatchSolver"):
    raise ImportError(
        "data_collection.py requires the private solver API provided by "
        "graspologic 3.4.4. Install the project requirements before running it."
    )


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
