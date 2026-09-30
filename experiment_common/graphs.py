"""Graph settings and pair generation shared by the experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from utils.Graphs import (
    gen_ER_graphs,
    gen_IER_graphs,
    gen_PAPER_graphs,
    gen_pareto_chung_lu_graphs,
    gen_SBM_graphs,
)


GRAPH_MODELS = ("sparse_er", "sparse_sbm", "paper", "ier", "paper_pa")


SPARSE_SBM_BLOCK_PROBABILITIES = (
    (0.02, 0.005, 0.01),
    (0.005, 0.02, 0.005),
    (0.01, 0.005, 0.02),
)


@dataclass(frozen=True)
class ExperimentConfig:
    graph_model: str = "sparse_er"
    n_graph_pairs: int = 20
    candidates_per_pair: int = 50

    vertex_count: int = 600
    n_blocks: int = 3
    n_per_block: int = 200
    sbm_block_probabilities: tuple[tuple[float, ...], ...] = (
        SPARSE_SBM_BLOCK_PROBABILITIES
    )

    er_probability: float = 0.01
    paper_alpha: float = 2.0
    paper_probability: float = 0.005
    paper_pa_alpha: float = 0.0
    paper_pa_beta: float = 1.0
    paper_pa_probability: float = 0.005
    ier_alpha: float = 1.0
    ier_beta: float = 99.0

    rho: float | None = 0.80
    n_seeds: int = 18
    max_iter: int = 30
    tol: float = 0.01
    random_seed: int = 1234
    max_minutes: float = 30.0

    @property
    def n_vertices(self) -> int:
        if self.graph_model == "sparse_sbm":
            return self.n_blocks * self.n_per_block
        return self.vertex_count


def preset_config(graph_model: str) -> ExperimentConfig:
    """Return the agreed small-scale configuration for one graph model."""

    if graph_model == "sparse_er":
        return ExperimentConfig(
            graph_model="sparse_er",
            vertex_count=600,
            er_probability=0.01,
            rho=0.80,
            n_seeds=18,
        )
    if graph_model == "sparse_sbm":
        return ExperimentConfig(
            graph_model="sparse_sbm",
            n_blocks=3,
            n_per_block=200,
            sbm_block_probabilities=SPARSE_SBM_BLOCK_PROBABILITIES,
            rho=0.80,
            n_seeds=15,
        )
    if graph_model == "paper":
        return ExperimentConfig(
            graph_model="paper",
            vertex_count=600,
            paper_alpha=2.0,
            paper_probability=0.005,
            rho=0.70,
            n_seeds=30,
        )
    if graph_model == "paper_pa":
        return ExperimentConfig(
            graph_model="paper_pa",
            vertex_count=600,
            paper_pa_alpha=0.0,
            paper_pa_beta=1.0,
            paper_pa_probability=0.005,
            rho=0.62,
            n_seeds=20,
        )
    if graph_model == "ier":
        return ExperimentConfig(
            graph_model="ier",
            vertex_count=600,
            ier_alpha=1.0,
            ier_beta=99.0,
            rho=0.77,
            n_seeds=20,
        )
    raise ValueError(f"Unknown graph model: {graph_model}")


def generate_graph_pair(
    config: ExperimentConfig, graph_pair_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(graph_pair_seed)
    if config.graph_model == "sparse_er":
        adjacency_1, adjacency_2, true_permutation = gen_ER_graphs(
            n=config.vertex_count,
            p=config.er_probability,
            rho=config.rho,
            directed=False,
            loops=False,
        )
    elif config.graph_model == "sparse_sbm":
        adjacency_1, adjacency_2, true_permutation = gen_SBM_graphs(
            n_per_block=config.n_per_block,
            n_blocks=config.n_blocks,
            block_probs=np.asarray(config.sbm_block_probabilities, dtype=float),
            rho=config.rho,
            directed=False,
            loops=False,
        )
    elif config.graph_model == "paper":
        adjacency_1, adjacency_2, true_permutation = gen_pareto_chung_lu_graphs(
            n=config.vertex_count,
            alpha=config.paper_alpha,
            p=config.paper_probability,
            rho=config.rho,
            directed=False,
            loops=False,
        )
    elif config.graph_model == "paper_pa":
        adjacency_1, adjacency_2, true_permutation = gen_PAPER_graphs(
            n=config.vertex_count,
            alpha=config.paper_pa_alpha,
            beta=config.paper_pa_beta,
            p=config.paper_pa_probability,
            rho=config.rho,
        )
    elif config.graph_model == "ier":
        adjacency_1, adjacency_2, true_permutation = gen_IER_graphs(
            n=config.vertex_count,
            rho=config.rho,
            alpha=config.ier_alpha,
            beta=config.ier_beta,
            directed=False,
            loops=False,
        )
    else:
        raise ValueError(f"Unknown graph model: {config.graph_model}")
    return (
        np.asarray(adjacency_1, dtype=float),
        np.asarray(adjacency_2, dtype=float),
        np.asarray(true_permutation, dtype=int),
    )


def validate_config(config: ExperimentConfig) -> None:
    if config.graph_model not in GRAPH_MODELS:
        raise ValueError(f"graph_model must be one of {GRAPH_MODELS}.")
    if config.n_graph_pairs <= 0 or config.candidates_per_pair <= 0:
        raise ValueError("Graph-pair and candidate counts must be positive.")
    if config.vertex_count <= 0:
        raise ValueError("vertex_count must be positive.")
    if config.n_blocks <= 0 or config.n_per_block <= 0:
        raise ValueError("Block counts and sizes must be positive.")
    if not 0 < config.n_seeds < config.n_vertices:
        raise ValueError("n_seeds must be between 1 and n_vertices - 1.")
    if config.candidates_per_pair > math.comb(config.n_vertices, config.n_seeds):
        raise ValueError("Requested more unique seed sets than are possible.")
    for name, probability in (
        ("er_probability", config.er_probability),
        ("paper_probability", config.paper_probability),
        ("paper_pa_probability", config.paper_pa_probability),
        ("rho", config.rho),
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    block_probabilities = np.asarray(config.sbm_block_probabilities, dtype=float)
    if block_probabilities.shape != (config.n_blocks, config.n_blocks):
        raise ValueError(
            "sbm_block_probabilities must have shape (n_blocks, n_blocks)."
        )
    if np.any((block_probabilities < 0) | (block_probabilities > 1)):
        raise ValueError("All SBM block probabilities must be between 0 and 1.")
    for name, value in (
        ("paper_alpha", config.paper_alpha),
        ("ier_alpha", config.ier_alpha),
        ("ier_beta", config.ier_beta),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    if not math.isfinite(config.paper_pa_alpha) or config.paper_pa_alpha < 0:
        raise ValueError("paper_pa_alpha must be nonnegative and finite.")
    if not math.isfinite(config.paper_pa_beta) or config.paper_pa_beta <= 0:
        raise ValueError("paper_pa_beta must be positive and finite.")
    if config.max_iter <= 0 or config.tol <= 0 or config.max_minutes <= 0:
        raise ValueError("max_iter, tol, and max_minutes must be positive.")
