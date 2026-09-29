"""Fixed graph-pair problems for choosing one additional SGM seed."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache")
)
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache")
)

import numpy as np
from graspologic.match import graph_match
import graspologic.match.wrappers as match_wrappers

from cross_pair_seed_set_quality.data_collection import (
    ExperimentConfig,
    generate_graph_pair,
)


SUCCESS_THRESHOLD = 0.90

SPARSE_SBM_PROBABILITIES = (
    (0.04, 0.01, 0.02),
    (0.01, 0.04, 0.01),
    (0.02, 0.01, 0.04),
)


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    description: str
    config: ExperimentConfig
    graph_pair_seed: int
    sgm_seed: int
    seed_vertices_graph1: tuple[int, ...]


@dataclass(frozen=True)
class PreparedProblem:
    spec: ProblemSpec
    adjacency_1: np.ndarray
    adjacency_2: np.ndarray
    true_permutation: np.ndarray
    seeds_1: np.ndarray
    seeds_2: np.ndarray


@dataclass
class SGMRun:
    h0_accuracy: float
    final_unseeded_accuracy: float
    final_all_accuracy: float
    runtime_seconds: float
    n_iter: int
    converged: bool
    objective_score: float
    unseeded_1: np.ndarray
    unseeded_2: np.ndarray
    first_gradient: np.ndarray
    first_direction: np.ndarray
    final_permutation: np.ndarray


PROBLEMS: dict[str, ProblemSpec] = {
    "sparse_er": ProblemSpec(
        name="sparse_er",
        description="Sparse homogeneous ER graph.",
        config=ExperimentConfig(
            graph_model="sparse_er",
            vertex_count=300,
            er_probability=0.02,
            rho=0.80,
            n_seeds=8,
            max_iter=30,
            tol=0.01,
        ),
        graph_pair_seed=3414274753,
        sgm_seed=680531080,
        seed_vertices_graph1=(80, 125, 139, 231, 235, 284, 285, 287),
    ),
    "sparse_sbm": ProblemSpec(
        name="sparse_sbm",
        description="Sparse three-community SBM graph.",
        config=ExperimentConfig(
            graph_model="sparse_sbm",
            n_blocks=3,
            n_per_block=100,
            sbm_block_probabilities=SPARSE_SBM_PROBABILITIES,
            rho=0.80,
            n_seeds=8,
            max_iter=30,
            tol=0.01,
        ),
        graph_pair_seed=440533379,
        sgm_seed=680291271,
        seed_vertices_graph1=(9, 37, 73, 95, 127, 168, 251, 286),
    ),
    "paper": ProblemSpec(
        name="paper",
        description="Sparse heterogeneous PAPER graph.",
        config=ExperimentConfig(
            graph_model="paper",
            vertex_count=300,
            paper_alpha=2.0,
            paper_probability=0.01,
            rho=0.70,
            n_seeds=14,
            max_iter=30,
            tol=0.01,
        ),
        graph_pair_seed=1615603467,
        sgm_seed=1471549916,
        seed_vertices_graph1=(
            10,
            25,
            35,
            39,
            57,
            59,
            85,
            102,
            114,
            126,
            232,
            238,
            261,
            282,
        ),
    ),
}


def load_problem(name: str) -> PreparedProblem:
    """Generate one fixed graph pair and its existing seed correspondence."""

    spec = PROBLEMS[name]
    adjacency_1, adjacency_2, true_permutation = generate_graph_pair(
        spec.config, spec.graph_pair_seed
    )
    seeds_1 = np.asarray(spec.seed_vertices_graph1, dtype=int)
    seeds_2 = true_permutation[seeds_1]
    return PreparedProblem(
        spec=spec,
        adjacency_1=np.asarray(adjacency_1, dtype=np.int8),
        adjacency_2=np.asarray(adjacency_2, dtype=np.int8),
        true_permutation=np.asarray(true_permutation, dtype=int),
        seeds_1=seeds_1,
        seeds_2=seeds_2,
    )


def run_sgm(problem: PreparedProblem, seeds_1: np.ndarray) -> SGMRun:
    """Run SGM once and retain its first gradient and assignment direction."""

    seeds_1 = np.sort(np.asarray(seeds_1, dtype=int))
    seeds_2 = problem.true_permutation[seeds_1]
    partial_match = np.column_stack((seeds_1, seeds_2))
    captured: dict[str, Any] = {}
    original = match_wrappers._GraphMatchSolver.compute_step_direction

    def capture_first_step(solver, gradient, rng):
        direction = original(solver, gradient, rng)
        if not captured:
            captured["solver"] = solver
            captured["gradient"] = np.asarray(gradient, dtype=float).copy()
            captured["direction"] = np.asarray(direction, dtype=float).copy()
            captured["unseeded_1"] = solver.perm_A[solver.n_seeds :].copy()
            captured["unseeded_2"] = solver.perm_B[solver.n_seeds :].copy()
        return direction

    match_wrappers._GraphMatchSolver.compute_step_direction = capture_first_step
    started = time.perf_counter()
    try:
        result = graph_match(
            problem.adjacency_1,
            problem.adjacency_2,
            partial_match=partial_match,
            n_init=1,
            max_iter=problem.spec.config.max_iter,
            tol=problem.spec.config.tol,
            rng=problem.spec.sgm_seed,
        )
    finally:
        match_wrappers._GraphMatchSolver.compute_step_direction = original
    runtime_seconds = time.perf_counter() - started

    solver = captured["solver"]
    unseeded_1 = captured["unseeded_1"]
    unseeded_2 = captured["unseeded_2"]
    first_direction = captured["direction"]
    first_prediction = unseeded_2[np.argmax(first_direction, axis=1)]

    final_permutation = np.full(problem.adjacency_1.shape[0], -1, dtype=int)
    final_permutation[result.indices_A] = result.indices_B

    return SGMRun(
        h0_accuracy=float(
            np.mean(first_prediction == problem.true_permutation[unseeded_1])
        ),
        final_unseeded_accuracy=float(
            np.mean(
                final_permutation[unseeded_1]
                == problem.true_permutation[unseeded_1]
            )
        ),
        final_all_accuracy=float(
            np.mean(final_permutation == problem.true_permutation)
        ),
        runtime_seconds=runtime_seconds,
        n_iter=int(solver.n_iter_),
        converged=bool(solver.converged_),
        objective_score=float(result.score),
        unseeded_1=unseeded_1,
        unseeded_2=unseeded_2,
        first_gradient=captured["gradient"],
        first_direction=first_direction,
        final_permutation=final_permutation,
    )


def scalar_run_summary(run: SGMRun) -> dict[str, float | int | bool]:
    return {
        "h0_accuracy": run.h0_accuracy,
        "final_unseeded_accuracy": run.final_unseeded_accuracy,
        "final_all_accuracy": run.final_all_accuracy,
        "sgm_runtime_seconds": run.runtime_seconds,
        "sgm_n_iter": run.n_iter,
        "sgm_converged": run.converged,
        "sgm_objective_score": run.objective_score,
    }


def _aligned_gradient(
    run: SGMRun, vertices_1: np.ndarray, vertices_2: np.ndarray
) -> np.ndarray:
    row_index = {int(vertex): index for index, vertex in enumerate(run.unseeded_1)}
    column_index = {
        int(vertex): index for index, vertex in enumerate(run.unseeded_2)
    }
    rows = [row_index[int(vertex)] for vertex in vertices_1]
    columns = [column_index[int(vertex)] for vertex in vertices_2]
    return run.first_gradient[np.ix_(rows, columns)]


def _standardize_rows(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    return centered / scale


def _gradient_state(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    standardized = _standardize_rows(matrix)
    true_scores = np.diag(standardized)
    false_scores = standardized.copy()
    np.fill_diagonal(false_scores, -np.inf)
    margins = true_scores - np.max(false_scores, axis=1)
    ranks = 1 + np.sum(standardized > true_scores[:, None], axis=1)
    return margins, ranks.astype(float)


def _first_prediction(run: SGMRun, vertices_1: np.ndarray) -> np.ndarray:
    row_index = {int(vertex): index for index, vertex in enumerate(run.unseeded_1)}
    rows = [row_index[int(vertex)] for vertex in vertices_1]
    columns = np.argmax(run.first_direction[rows], axis=1)
    return run.unseeded_2[columns]


def evaluate_candidate(
    problem: PreparedProblem,
    candidate: int,
    baseline: SGMRun | None = None,
    success_threshold: float = SUCCESS_THRESHOLD,
) -> dict[str, float | int | bool | str]:
    """Add one revealed seed and measure its SGM and first-gradient effects."""

    candidate = int(candidate)
    if candidate in problem.seeds_1:
        raise ValueError("The candidate is already an existing seed.")

    baseline = baseline or run_sgm(problem, problem.seeds_1)
    augmented_seeds = np.sort(np.append(problem.seeds_1, candidate))
    augmented = run_sgm(problem, augmented_seeds)

    n = problem.adjacency_1.shape[0]
    common_vertices_1 = np.setdiff1d(np.arange(n), augmented_seeds)
    common_vertices_2 = problem.true_permutation[common_vertices_1]
    truth = common_vertices_2

    baseline_first = _first_prediction(baseline, common_vertices_1)
    augmented_first = _first_prediction(augmented, common_vertices_1)
    baseline_first_correct = baseline_first == truth
    augmented_first_correct = augmented_first == truth

    baseline_final_correct = (
        baseline.final_permutation[common_vertices_1] == truth
    )
    augmented_final_correct = (
        augmented.final_permutation[common_vertices_1] == truth
    )

    baseline_gradient = _aligned_gradient(
        baseline, common_vertices_1, common_vertices_2
    )
    augmented_gradient = _aligned_gradient(
        augmented, common_vertices_1, common_vertices_2
    )
    baseline_margins, baseline_ranks = _gradient_state(baseline_gradient)
    augmented_margins, augmented_ranks = _gradient_state(augmented_gradient)
    margin_change = augmented_margins - baseline_margins

    revealed_vertex_2 = int(problem.true_permutation[candidate])
    seed_contribution = np.outer(
        problem.adjacency_1[common_vertices_1, candidate],
        problem.adjacency_2[common_vertices_2, revealed_vertex_2],
    )
    true_witnesses = np.diag(seed_contribution)
    false_witnesses = seed_contribution.copy().astype(float)
    np.fill_diagonal(false_witnesses, np.nan)
    true_witness_fraction = float(np.mean(true_witnesses))
    false_witness_fraction = float(np.nanmean(false_witnesses))

    baseline_common_h0 = float(np.mean(baseline_first_correct))
    augmented_common_h0 = float(np.mean(augmented_first_correct))
    baseline_common_final = float(np.mean(baseline_final_correct))
    augmented_common_final = float(np.mean(augmented_final_correct))

    return {
        "problem": problem.spec.name,
        "candidate_vertex_graph1": candidate,
        "revealed_vertex_graph2": revealed_vertex_2,
        "common_vertex_count": len(common_vertices_1),
        "baseline_h0_accuracy": baseline.h0_accuracy,
        "augmented_h0_accuracy": augmented.h0_accuracy,
        "baseline_common_h0_accuracy": baseline_common_h0,
        "augmented_common_h0_accuracy": augmented_common_h0,
        "delta_common_h0_accuracy": augmented_common_h0 - baseline_common_h0,
        "first_direction_wrong_to_right_count": int(
            np.sum(~baseline_first_correct & augmented_first_correct)
        ),
        "first_direction_right_to_wrong_count": int(
            np.sum(baseline_first_correct & ~augmented_first_correct)
        ),
        "first_direction_wrong_to_right_fraction": float(
            np.mean(~baseline_first_correct & augmented_first_correct)
        ),
        "first_direction_right_to_wrong_fraction": float(
            np.mean(baseline_first_correct & ~augmented_first_correct)
        ),
        "baseline_final_unseeded_accuracy": baseline.final_unseeded_accuracy,
        "augmented_final_unseeded_accuracy": augmented.final_unseeded_accuracy,
        "baseline_common_final_accuracy": baseline_common_final,
        "augmented_common_final_accuracy": augmented_common_final,
        "delta_common_final_accuracy": (
            augmented_common_final - baseline_common_final
        ),
        "outcome_success": bool(augmented_common_final >= success_threshold),
        "baseline_gradient_margin_mean": float(np.mean(baseline_margins)),
        "augmented_gradient_margin_mean": float(np.mean(augmented_margins)),
        "delta_gradient_margin_mean": float(np.mean(margin_change)),
        "delta_gradient_margin_median": float(np.median(margin_change)),
        "delta_gradient_margin_q10": float(np.quantile(margin_change, 0.10)),
        "gradient_margin_improved_fraction": float(np.mean(margin_change > 0)),
        "baseline_gradient_true_rank_mean": float(np.mean(baseline_ranks)),
        "augmented_gradient_true_rank_mean": float(np.mean(augmented_ranks)),
        "gradient_true_rank_improvement_mean": float(
            np.mean(baseline_ranks - augmented_ranks)
        ),
        "baseline_gradient_true_top1_fraction": float(
            np.mean(baseline_ranks <= 1)
        ),
        "augmented_gradient_true_top1_fraction": float(
            np.mean(augmented_ranks <= 1)
        ),
        "delta_gradient_true_top1_fraction": float(
            np.mean(augmented_ranks <= 1) - np.mean(baseline_ranks <= 1)
        ),
        "baseline_gradient_true_top5_fraction": float(
            np.mean(baseline_ranks <= 5)
        ),
        "augmented_gradient_true_top5_fraction": float(
            np.mean(augmented_ranks <= 5)
        ),
        "delta_gradient_true_top5_fraction": float(
            np.mean(augmented_ranks <= 5) - np.mean(baseline_ranks <= 5)
        ),
        "candidate_true_witness_fraction": true_witness_fraction,
        "candidate_false_spillover_fraction": false_witness_fraction,
        "candidate_witness_selectivity": (
            true_witness_fraction - false_witness_fraction
        ),
        "augmented_sgm_runtime_seconds": augmented.runtime_seconds,
        "augmented_sgm_n_iter": augmented.n_iter,
        "augmented_sgm_converged": augmented.converged,
        "augmented_sgm_objective_score": augmented.objective_score,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=tuple(PROBLEMS), required=True)
    parser.add_argument("--candidate", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_candidate(load_problem(args.problem), args.candidate)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
