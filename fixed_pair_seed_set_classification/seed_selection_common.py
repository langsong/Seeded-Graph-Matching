"""Shared tools for fixed-pair seed-set selection experiments."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache")
)
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache")
)

from experiment_common.features import (
    GraphContext,
    build_graph_context,
    fixed_pair_seed_set_features,
    signature_robustness_features,
)
from experiment_common.graphs import ExperimentConfig, generate_graph_pair, preset_config
from experiment_common.sgm import run_sgm_once


DEFAULT_PROBLEM_DIR = Path(__file__).resolve().parent / "data"
PairBuilder = Callable[
    [int], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]
]


@dataclass(frozen=True)
class FixedProblem:
    """Definition of one fixed correlated graph-pair benchmark."""

    name: str
    config: ExperimentConfig
    graph_pair_seed: int
    sgm_seed: int = 2026
    source_pair_id: int | None = None
    pilot_success_rate: float | None = None
    pair_builder: PairBuilder | None = None


@dataclass
class PreparedProblem:
    """A fixed graph pair and its reusable Graph-A feature context."""

    spec: FixedProblem
    adjacency_1: np.ndarray
    adjacency_2: np.ndarray
    true_permutation: np.ndarray
    context_1: GraphContext


def problem_paths(
    spec: FixedProblem, problem_dir: Path = DEFAULT_PROBLEM_DIR
) -> tuple[Path, Path]:
    return (
        problem_dir / f"{spec.name}_problem.npz",
        problem_dir / f"{spec.name}_problem.metadata.json",
    )


def _graph_generation_settings(config: ExperimentConfig) -> dict[str, Any]:
    """Return the config fields that determine a generated graph pair."""

    if config.graph_model == "chung_lu":
        return {
            "graph_model": config.graph_model,
            "rho": config.rho,
            "vertex_count": config.vertex_count,
            "chung_lu_alpha": config.effective_chung_lu_alpha,
            "chung_lu_weight_scale": config.effective_chung_lu_weight_scale,
            "chung_lu_probability_floor": (
                config.effective_chung_lu_probability_floor
            ),
        }
    fields = {
        "sparse_er": ("vertex_count", "er_probability"),
        "sparse_sbm": ("n_blocks", "n_per_block", "sbm_block_probabilities"),
        "paper": ("vertex_count", "paper_alpha", "paper_probability"),
        "paper_pa": (
            "vertex_count", "paper_pa_alpha", "paper_pa_beta",
            "paper_pa_probability",
        ),
        "ier": ("vertex_count", "ier_alpha", "ier_beta"),
        "enron": ("vertex_count",),
    }
    names = fields[config.graph_model]
    return {
        "graph_model": config.graph_model,
        "rho": config.rho,
        **{name: json.loads(json.dumps(getattr(config, name))) for name in names},
    }


def graph_pair_settings(spec: FixedProblem) -> dict[str, Any]:
    """Fields that determine a generated pair, excluding solver and seed budget."""

    return {
        "problem": spec.name,
        "graph_pair_seed": spec.graph_pair_seed,
        **_graph_generation_settings(spec.config),
    }


def prepare_problem(
    spec: FixedProblem, problem_dir: Path = DEFAULT_PROBLEM_DIR
) -> PreparedProblem:
    """Create the fixed pair once, then load and reuse it on later calls."""

    problem_path, metadata_path = problem_paths(spec, problem_dir)
    problem_dir.mkdir(parents=True, exist_ok=True)

    if problem_path.exists():
        if not metadata_path.exists():
            raise ValueError(f"Missing settings metadata for saved pair: {problem_path}")
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        saved_settings = {
            "problem": saved.get("problem"),
            "graph_pair_seed": saved.get("graph_pair_seed"),
            **_graph_generation_settings(ExperimentConfig(**saved.get("config", {}))),
        }
        expected_settings = graph_pair_settings(spec)
        if saved_settings != expected_settings:
            raise ValueError(
                f"Saved graph pair settings differ from {spec.name}: {problem_path}. "
                "Use a new --output-dir or --problem-dir."
            )
    else:
        if spec.pair_builder is None:
            adjacency_1, adjacency_2, true_permutation = generate_graph_pair(
                spec.config, spec.graph_pair_seed
            )
            source_metadata: dict[str, Any] = {}
        else:
            adjacency_1, adjacency_2, true_permutation, source_metadata = (
                spec.pair_builder(spec.graph_pair_seed)
            )
        if adjacency_1.shape[0] != spec.config.n_vertices:
            raise ValueError(
                f"Expected {spec.config.n_vertices} vertices, got "
                f"{adjacency_1.shape[0]}."
            )
        np.savez_compressed(
            problem_path,
            adjacency_graph1=np.asarray(adjacency_1, dtype=np.int8),
            adjacency_graph2=np.asarray(adjacency_2, dtype=np.int8),
            true_permutation=np.asarray(true_permutation, dtype=int),
        )
        metadata = {
            "problem": spec.name,
            "graph_pair_seed": spec.graph_pair_seed,
            "sgm_seed": spec.sgm_seed,
            "source_pair_id": spec.source_pair_id,
            "pilot_success_rate": spec.pilot_success_rate,
            "config": asdict(spec.config),
            "problem_file": str(problem_path),
            **source_metadata,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    with np.load(problem_path) as saved:
        adjacency_1 = saved["adjacency_graph1"]
        adjacency_2 = saved["adjacency_graph2"]
        true_permutation = saved["true_permutation"]

    return PreparedProblem(
        spec=spec,
        adjacency_1=adjacency_1,
        adjacency_2=adjacency_2,
        true_permutation=true_permutation,
        context_1=build_graph_context(adjacency_1, compute_seed_features=False),
    )


def evaluate_seed_set(
    problem: PreparedProblem,
    seeds_1: np.ndarray | list[int],
    *,
    sgm_seed: int | None = None,
    max_iter: int | None = None,
) -> dict[str, Any]:
    """Evaluate one Graph-A seed set on a prepared fixed graph pair."""

    seeds_1 = np.sort(np.asarray(seeds_1, dtype=int))
    config = problem.spec.config
    if len(seeds_1) != config.n_seeds or len(np.unique(seeds_1)) != len(seeds_1):
        raise ValueError(f"Expected {config.n_seeds} distinct seed vertices.")

    seeds_2 = problem.true_permutation[seeds_1]
    sgm_seed = problem.spec.sgm_seed if sgm_seed is None else sgm_seed
    max_iter = config.max_iter if max_iter is None else max_iter
    row: dict[str, Any] = {
        "problem": problem.spec.name,
        "graph_model": config.graph_model,
        "graph_pair_seed": problem.spec.graph_pair_seed,
        "sgm_seed": sgm_seed,
        "sgm_max_iter": max_iter,
        "n_vertices": config.n_vertices,
        "n_seeds": config.n_seeds,
        "rho": config.rho,
        "seed_vertices_graph1": json.dumps(seeds_1.tolist()),
        "seed_vertices_graph2": json.dumps(seeds_2.tolist()),
    }
    row.update(fixed_pair_seed_set_features(problem.context_1, seeds_1))
    row.update(signature_robustness_features(problem.context_1, seeds_1))
    row.update(
        run_sgm_once(
            problem.adjacency_1,
            problem.adjacency_2,
            problem.true_permutation,
            seeds_1,
            sgm_seed,
            max_iter,
            config.tol,
        )
    )
    return row


def run_problem_cli(
    spec: FixedProblem, problem_dir: Path = DEFAULT_PROBLEM_DIR
) -> None:
    parser = argparse.ArgumentParser(
        description=f"Evaluate one seed set on the fixed {spec.name} graph pair."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help=f"Exactly {spec.config.n_seeds} vertex indices from Graph A.",
    )
    parser.add_argument("--problem-dir", type=Path, default=problem_dir)
    args = parser.parse_args()

    result = evaluate_seed_set(prepare_problem(spec, args.problem_dir), args.seeds)
    print(json.dumps(result, indent=2))
