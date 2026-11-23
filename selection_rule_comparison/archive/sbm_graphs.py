"""Correlated SBM graph pairs for the community experiments (sbm_rules.py and sbm_collection.py), 2 Nov 2026.

Two kinds of graph, each matched to a round-1 model so that the only difference is community structure:

    plain             stochastic block model: within-block and between-block edge probabilities chosen so that the
                      mean degree matches round 1's ER setting and a set share of each vertex's edges stays inside its
                      block ("ER plus communities")
    degree_corrected  the same blocks on Chung-Lu probabilities w_i w_j / sum(w) (Pareto shape 2 weights, capped at
                      sqrt(sum(w)) as in round 1), with within- and between-block pairs rescaled to the same share and
                      mean degree ("Chung-Lu plus communities")

Settings (SETTINGS): four equal blocks of 150 vertices with 80% of edges inside blocks are the main ones ("sbm",
"dcsbm"). Variants with unequal blocks (300, 150, 100, 50) or weaker communities (60% inside) are defined for later.

Graph family, as in round 1: percolation uses mean degree 12 and rho = 0.70; GOAT and FAQ share pairs with mean degree
16 and rho = 0.50 (plain, as ER) or 0.45 (degree-corrected, as Chung-Lu). GOAT's regulariser is 100 on plain SBM (as
ER) and 300 on degree-corrected SBM (as Chung-Lu).

Pairs: A and B are drawn with graspologic's sample_edges_corr from the same probability matrix with edge correlation
rho; B's vertices are then shuffled (truth maps A's vertices to B's). Block labels refer to A's vertices. Every pair
passes the round-1 structure check (both graphs connected, no automorphisms). Pair seeds come from their own stream,
by graph family, setting, purpose (rules, calibration, collection) and pair index, so GOAT and FAQ share pairs.
"""

from __future__ import annotations

import json

import numpy as np

from experiment_common.graph_checks import first_passing_pair

from . import regime_sweep as RS
from .. import run_comparison as rc
from .. import settings as S

SETTINGS = {
    "sbm": {"kind": "plain", "blocks": (150, 150, 150, 150), "within": 0.80},
    "dcsbm": {"kind": "degree_corrected", "blocks": (150, 150, 150, 150), "within": 0.80},
    "sbm_unequal": {"kind": "plain", "blocks": (300, 150, 100, 50), "within": 0.80},
    "dcsbm_unequal": {"kind": "degree_corrected", "blocks": (300, 150, 100, 50), "within": 0.80},
    "sbm_weak": {"kind": "plain", "blocks": (150, 150, 150, 150), "within": 0.60},
    "dcsbm_weak": {"kind": "degree_corrected", "blocks": (150, 150, 150, 150), "within": 0.60},
}
DEFAULT_SETTINGS = ("sbm", "dcsbm")
SETTING_LABELS = {"sbm": "SBM", "dcsbm": "degree-corrected SBM", "sbm_unequal": "SBM, unequal blocks",
                  "dcsbm_unequal": "degree-corrected SBM, unequal blocks", "sbm_weak": "SBM, weak communities",
                  "dcsbm_weak": "degree-corrected SBM, weak communities"}
BASE_MODEL = {"plain": "sparse_er", "degree_corrected": "chung_lu"}   # the round-1 model without communities

FAMILY = {"percolation": "percolation", "goat": "sgm", "faq": "sgm"}
MEAN_DEGREE = {"percolation": 12.0, "sgm": 16.0}
RHO = {("percolation", "plain"): 0.70, ("percolation", "degree_corrected"): 0.70,
       ("sgm", "plain"): 0.50, ("sgm", "degree_corrected"): 0.45}
GOAT_REGULARIZER = {"plain": 100, "degree_corrected": 300}
PARETO_ALPHA = 2.0

PAIR_SEED_BASE = 20261107
FAMILY_CODES = {"percolation": 1, "sgm": 2}
SETTING_CODES = {name: i + 1 for i, name in enumerate(SETTINGS)}
PURPOSE_CODES = {"rules": 1, "calibration": 2, "collection": 3}


def kind(setting: str) -> str:
    return SETTINGS[setting]["kind"]


def rho(algorithm: str, setting: str) -> float:
    return RHO[(FAMILY[algorithm], kind(setting))]


def regularizer(algorithm: str, setting: str) -> float | None:
    return GOAT_REGULARIZER[kind(setting)] if algorithm == "goat" else None


def describe(algorithm: str, setting: str) -> dict:
    spec = SETTINGS[setting]
    family = FAMILY[algorithm]
    return {"setting": setting, "kind": spec["kind"], "blocks": list(spec["blocks"]), "within_share": spec["within"],
            "vertices": int(sum(spec["blocks"])), "mean_degree": MEAN_DEGREE[family], "rho": rho(algorithm, setting),
            "pareto_alpha": PARETO_ALPHA if spec["kind"] == "degree_corrected" else None,
            "goat_regularizer": regularizer(algorithm, setting), "base_model": BASE_MODEL[spec["kind"]]}


# --------------------------------------------------------------------------
# The probability matrix
# --------------------------------------------------------------------------

def _scale_to_targets(P0: np.ndarray, same: np.ndarray, within: float, total: float, rounds: int = 50):
    """Factors for within- and between-block pairs so that the expected degree sum is `total` with a `within` share
    inside blocks, after clipping probabilities at 1."""

    off = ~np.eye(len(P0), dtype=bool)
    inside, outside = same & off, ~same
    a = within * total / P0[inside].sum()
    b = (1 - within) * total / P0[outside].sum()
    for _ in range(rounds):  # clipping only bites inside blocks (between-block probabilities are small)
        got = np.minimum(P0[inside] * a, 1.0).sum()
        if abs(got - within * total) < 1e-9 * total:
            break
        a *= within * total / got
    return a, b


def probability_matrix(algorithm: str, setting: str) -> tuple[np.ndarray, np.ndarray]:
    """Edge probabilities and A's block labels. Uses numpy's global random state (seeded by generate_pair)."""

    spec = SETTINGS[setting]
    sizes = spec["blocks"]
    n = int(sum(sizes))
    mean_degree = MEAN_DEGREE[FAMILY[algorithm]]
    blocks = np.repeat(np.arange(len(sizes)), sizes)
    np.random.shuffle(blocks)                                   # blocks in random vertex order
    same = blocks[:, None] == blocks[None, :]
    total = mean_degree * n                                     # expected degree sum
    if spec["kind"] == "plain":
        within_pairs = sum(s * (s - 1) for s in sizes)          # ordered pairs inside blocks
        between_pairs = n * (n - 1) - within_pairs
        P = np.where(same, spec["within"] * total / within_pairs, (1 - spec["within"]) * total / between_pairs)
    else:
        from utils.Graphs import pareto_quantile_weights
        weights = RS.weight_scale(PARETO_ALPHA, n, mean_degree) * pareto_quantile_weights(n, PARETO_ALPHA)
        np.random.shuffle(weights)
        weights = np.minimum(weights, np.sqrt(weights.sum()))  # round 1's structural cap
        P0 = np.minimum(np.outer(weights, weights) / weights.sum(), 1.0)
        np.fill_diagonal(P0, 0.0)
        a, b = _scale_to_targets(P0, same, spec["within"], total)
        P = np.minimum(P0 * np.where(same, a, b), 1.0)
    P = np.array(P, dtype=float)
    np.fill_diagonal(P, 0.0)
    return P, blocks


def generate_pair(algorithm: str, setting: str, seed: int):
    """(A, B, truth, blocks) for one pair seed; same convention as experiment_common.graphs.generate_graph_pair."""

    from graspologic.simulations import sample_edges_corr

    np.random.seed(seed)
    P, blocks = probability_matrix(algorithm, setting)
    n = len(P)
    G1, G2 = sample_edges_corr(P, np.full((n, n), rho(algorithm, setting)), directed=False, loops=False)
    shuffle = np.random.permutation(n)
    G2 = G2[shuffle][:, shuffle]
    return (np.asarray(G1, dtype=float), np.asarray(G2, dtype=float), np.asarray(np.argsort(shuffle), dtype=int),
            blocks)


def first_seed(algorithm: str, setting: str, purpose: str, pair_index: int) -> int:
    seed = rc._seed(PAIR_SEED_BASE, FAMILY_CODES[FAMILY[algorithm]], SETTING_CODES[setting], PURPOSE_CODES[purpose],
                    pair_index)
    return seed % (2**32 - 10_000)


def load_pair(algorithm: str, setting: str, purpose: str, pair_index: int):
    """The structure-checked pair with this index: (seed, A, B, truth, blocks). GOAT and FAQ get the same pair."""

    start = first_seed(algorithm, setting, purpose, pair_index)
    offset = 0
    while True:
        seed, A, B, truth, _report, _rejected = first_passing_pair(
            lambda s: generate_pair(algorithm, setting, s)[:3], start + offset)
        if int(seed) not in S.KNOWN_PAIR_SEEDS:
            break
        offset = int(seed) - start + 1
    blocks = generate_pair(algorithm, setting, int(seed))[3]
    return int(seed), A, B, truth, blocks


def pair_summary(A: np.ndarray, B: np.ndarray, truth: np.ndarray, blocks: np.ndarray) -> dict:
    """Hashes and simple statistics, written with every pair."""

    degree = (A > 0).sum(axis=1)
    same = blocks[:, None] == blocks[None, :]
    edges = np.triu(A > 0, 1)
    aligned = B[np.ix_(truth, truth)]
    return {"hash_a": rc._hash(A), "hash_b": rc._hash(B), "mean_degree_a": float(degree.mean()),
            "max_degree_a": int(degree.max()), "within_share_a": float((edges & same).sum() / edges.sum()),
            "kept_edge_fraction": float(((A > 0) & (aligned > 0)).sum() / (A > 0).sum()),
            "blocks_a": json.dumps([int(b) for b in blocks])}


def blocks_covered(blocks: np.ndarray, seeds) -> int:
    return int(len(np.unique(blocks[np.asarray(seeds, dtype=int)])))


def largest_block_share(blocks: np.ndarray, seeds) -> float:
    counts = np.bincount(blocks[np.asarray(seeds, dtype=int)])
    return float(counts.max() / counts.sum())
