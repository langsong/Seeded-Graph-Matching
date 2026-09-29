"""Fixed Enron weeks 148–149 seed-set selection benchmark."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from cross_pair_seed_set_quality.data_collection import ExperimentConfig

from ..seed_selection_common import (
    FixedProblem,
    evaluate_seed_set,
    prepare_problem,
    run_problem_cli,
)


ENRON_FILE = Path(__file__).resolve().parents[2] / "utils" / "enron.mat"
WEEKS = (148, 149)  # One-based indices in the 187-week AAA array.


def build_enron_pair(permutation_seed: int):
    weekly = loadmat(ENRON_FILE)["AAA"]
    graph_1, graph_2 = [weekly[:, :, week - 1].astype(bool) for week in WEEKS]
    graph_1 = graph_1 | graph_1.T
    graph_2 = graph_2 | graph_2.T
    np.fill_diagonal(graph_1, False)
    np.fill_diagonal(graph_2, False)

    active = np.flatnonzero(
        (graph_1.sum(axis=1) > 0) & (graph_2.sum(axis=1) > 0)
    )
    graph_1 = graph_1[np.ix_(active, active)].astype(np.int8)
    graph_2 = graph_2[np.ix_(active, active)].astype(np.int8)
    upper = np.triu_indices(len(active), k=1)
    overlap = np.logical_and(graph_1[upper], graph_2[upper]).sum()
    union = np.logical_or(graph_1[upper], graph_2[upper]).sum()

    order = np.random.default_rng(permutation_seed).permutation(len(active))
    graph_2 = graph_2[np.ix_(order, order)]
    true_permutation = np.argsort(order)
    metadata = {
        "source_file": str(ENRON_FILE),
        "source_variable": "AAA",
        "weeks_1_based": list(WEEKS),
        "preprocessing": "Undirected edge if either email direction exists; remove loops; keep vertices active in both weeks.",
        "source_vertex_ids_1_based": (active + 1).tolist(),
        "graph_b_source_vertex_ids_1_based": (active[order] + 1).tolist(),
        "graph_b_permutation_seed": permutation_seed,
        "edge_jaccard_before_permutation": float(overlap / union),
    }
    return graph_1, graph_2, true_permutation, metadata


PROBLEM = FixedProblem(
    name="enron",
    config=ExperimentConfig(
        graph_model="enron", vertex_count=129, n_seeds=25, rho=None
    ),
    graph_pair_seed=2026,
    pilot_success_rate=0.47,
    pair_builder=build_enron_pair,
)


def check_seed_set(seeds_1):
    return evaluate_seed_set(prepare_problem(PROBLEM), seeds_1)


if __name__ == "__main__":
    run_problem_cli(PROBLEM)
