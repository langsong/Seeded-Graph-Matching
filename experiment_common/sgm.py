"""Single-run SGM evaluation shared by the experiments."""

from __future__ import annotations

import time
import json
import warnings
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from graspologic.match import graph_match
import graspologic.match.wrappers as match_wrappers

if not hasattr(match_wrappers, "_GraphMatchSolver"):
    raise ImportError(
        "data_collection.py requires the private solver API provided by "
        "graspologic 3.4.4. Install the project requirements before running it."
    )


SGM_DIAGNOSTIC_COLUMNS = (
    "h0_row_argmax_accuracy", "h0_true_match_mass_mean", "h0_row_entropy_mean",
    "sgm_warning_count", "sgm_warnings", "sgm_finite_soft_solution",
)


def run_sgm_once(
    adjacency_1: np.ndarray,
    adjacency_2: np.ndarray,
    true_permutation: np.ndarray,
    seeds_1: np.ndarray,
    sgm_seed: int,
    max_iter: int,
    tol: float,
    *,
    transport: bool = False,
    transport_regularizer: float = 100,
    transport_tol: float = 0.05,
    transport_max_iter: int = 5000,
    return_diagnostics: bool = False,
) -> dict[str, float | int | bool | str]:
    """Run FAQ or GOAT once, with no restart selection.

    GOAT's H0 accuracy uses a deterministic one-to-one projection of its first
    soft direction. With ``return_diagnostics``, the original row-argmax score,
    true-match mass, entropy and numerical warnings are also returned. Historical
    FAQ callers retain the same seven result columns by default.
    """

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
    caught: list = []
    try:
        with warnings.catch_warnings(record=return_diagnostics) as caught:
            if return_diagnostics:
                warnings.simplefilter("always")
            result = graph_match(
                adjacency_1,
                adjacency_2,
                partial_match=partial_match,
                n_init=1,
                n_jobs=1,
                init=None,
                init_perturbation=0.0,
                shuffle_input=True,
                max_iter=max_iter,
                tol=tol,
                rng=int(sgm_seed),
                transport=transport,
                transport_regularizer=transport_regularizer,
                transport_tol=transport_tol,
                transport_max_iter=transport_max_iter,
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
    row_argmax = np.argmax(first_direction, axis=1)
    if transport:
        if not np.isfinite(first_direction).all():
            raise RuntimeError("GOAT returned a nonfinite first transport direction.")
        rows, columns = linear_sum_assignment(first_direction, maximize=True)
        first_assignment = np.empty(len(unseeded_1), dtype=int)
        first_assignment[rows] = columns
    else:
        first_assignment = row_argmax
    first_direction_prediction = unseeded_2[first_assignment]
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

    outcome: dict[str, float | int | bool | str] = {
        "h0_accuracy": h0_accuracy,
        "final_unseeded_accuracy": final_unseeded_accuracy,
        "final_all_accuracy": final_all_accuracy,
        "sgm_runtime_seconds": runtime_seconds,
        "sgm_n_iter": int(solver.n_iter_),
        "sgm_converged": bool(solver.converged_),
        "sgm_objective_score": float(result.score),
    }
    if return_diagnostics:
        soft_solution = np.asarray(result.misc[0]["convex_solution"])
        if not np.isfinite(soft_solution).all():
            raise RuntimeError("SGM returned a nonfinite final soft solution.")
        inverse_unseeded_2 = np.full(len(true_permutation), -1, dtype=int)
        inverse_unseeded_2[unseeded_2] = np.arange(len(unseeded_2))
        truth_columns = inverse_unseeded_2[true_permutation[unseeded_1]]
        mass = first_direction[np.arange(len(unseeded_1)), truth_columns]
        entropy = -np.sum(
            first_direction * np.log(np.clip(first_direction, 1e-300, None)), axis=1
        )
        outcome.update({
            "h0_row_argmax_accuracy": float(np.mean(unseeded_2[row_argmax] == true_permutation[unseeded_1])),
            "h0_true_match_mass_mean": float(np.mean(mass)),
            "h0_row_entropy_mean": float(np.mean(entropy)),
            "sgm_warning_count": len(caught),
            "sgm_warnings": json.dumps([str(w.message) for w in caught]),
            "sgm_finite_soft_solution": True,
        })
    return outcome
