"""Does the pooled score carry over to new pairs, budgets and graph settings? (plans 1 and 2, 2 Nov 2026)

The pooled weights (fixed_pair_seed_set_classification/data/multi_pair_2026-10-30/pooled_weights.json) were fitted
on 24-50 fresh pairs per setting, all at one budget per setting, the training budget k0:
    percolation  ER 8, PAPER 7, Chung-Lu 5;  GOAT  ER 15, PAPER 12, Chung-Lu 9;  FAQ  ER 29, PAPER 29, Chung-Lu 27
best_of_n.py used them to pick seed sets. This module asks how well the score itself ranks seed sets outside the
data it was fitted on, in three parts:

    A  new pairs, same budget (plan 1): round 1's 20 pairs per setting, at k0.
    B  other budgets (plan 2): round 1's pairs 0-9, at k0 - d and k0 + d
       (d = 2 seeds for percolation and 3 for GOAT and FAQ; BUDGET_OFFSETS).
    C  other sizes and sparsities of the same graph model (plan 2, redesigned 2 Nov): a grid of graph size
       (300, 600, 1,200 vertices) by mean degree (x 0.75, x 1, x 1.5 of round 1's), on 10 fresh pairs per cell.
       Round 1's setting (600 vertices, x 1) is part A. Percolation runs every cell; GOAT and FAQ run 300 and 600
       vertices at all three densities, and 1,200 vertices only for Chung-Lu at x 1 (their runs are several times
       slower at 1,200 vertices). Everything else stays as in round 1: the model's shape parameters, rho and the
       solver settings. Mean degree is set through the ER edge probability, PAPER's background probability (its
       attachment tree adds about 2) or the Chung-Lu weight scale.
       The budget is calibrated first, so that a random set succeeds about half the time, as at k0 in training:
       3 random sets per pair (30 per budget) are run along a budget ladder (moving 1, 2, 4, ... steps until the
       success rate crosses 50%, then halving the gap), and the budget closest to 50% is used. If no random set
       succeeds even at the top of the ladder (or every set succeeds at the bottom), the cell is reported and
       skipped.

Seed sets. On each pair and budget, 20,000 distinct uniform random seed sets (the pool) are drawn and scored from
graph A with the pooled weights of the base setting (algorithm and graph model): score = sum of w_f * z_f, where
z_f is the feature in SDs of the pool. Each set's percentile is its score's position in the pool, as in
multi_pair_models.ipynb. Run on each pair:
    random  the first 60 (A) or 36 (B, C) sets of the pool: an unselected sample, for the average random set and
            for ranking (AUC)
    tail    40 (A) or 24 (B, C) sets from the top 5% of the rest of the pool by score, spread evenly in rank within
            three bands that match the top percentile bins: top 0.1% (10 or 6 sets, including the best one),
            99-99.9% (10 or 6) and 95-99% (20 or 12). Best of 100 picks around the 99th percentile and best of
            1,000 around the 99.9th; the set ranked r-th of the pool stands for best of 20,000 / r, so the tail
            traces best of N from N = 20 to 20,000 (best_of_n_table).
    rule    (A only) round 1's rules at k0: top degree, clustered, spread, distinguishing, and hub-free
            distinguishing for GOAT or top harmonic for percolation (regenerated from round 1's random streams and
            checked against round 1's stored orderings), to see where hand-made sets fall on the same score
Runs. One run per set with round 1's solver settings (FAQ one run, as in the training data). Each SGM run gets its
own solver seed from a new stream. Success: percolation accuracy >= 0.5; GOAT and FAQ unseeded accuracy >= 0.9.

Run from the repository root (ds environment); about 12 hours with 3 jobs (A about 3.5 h, B about 2 h,
C about 6.5 h):

    python -m selection_rule_comparison.score_transfer
    python -m selection_rule_comparison.score_transfer --parts A
    python -m selection_rule_comparison.score_transfer --status

or call run_all() from score_transfer.ipynb. Parts run in the order A, B, C. Each pair is saved when it finishes
(each calibration step too), so rerunning resumes; Ctrl-C (or the notebook's interrupt button) stops the workers
too. Results go to data/score_transfer/, or to the folder in the environment variable SCORE_TRANSFER_OUTPUT_DIR.
"""

from __future__ import annotations

import argparse
import datetime
import functools
import json
import os
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache"))

import numpy as np
import pandas as pd

from experiment_common.graph_checks import first_passing_pair
from experiment_common.graphs import generate_graph_pair
from experiment_common.provenance import collect_environment, thread_limit, write_json
from fixed_pair_seed_set_classification import multi_pair_collection as MP

from . import best_of_n as BN
from . import hub_test as H
from . import regime_sweep as RS
from .. import run_comparison as rc
from .. import settings as S
from ..strategies import STRATEGY_CODES, GraphA, ordering, seed_set_features

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]  # selection_rule_comparison/ (this file is in archive/)
DATA_DIR = EXPERIMENT_DIR / "data"
ROUND1_ROOT = DATA_DIR / "round1" / "main"
OUTPUT_DIR = Path(os.environ.get("SCORE_TRANSFER_OUTPUT_DIR") or DATA_DIR / "score_transfer")
WEIGHTS_PATH = BN.WEIGHTS_PATH

ALGORITHMS = ("percolation", "goat", "faq")
MODELS = ("sparse_er", "paper_pa", "chung_lu")
PARTS = ("A", "B", "C")
CORE = tuple(MP.CORE_FEATURES)

POOL_SIZE = 20_000
TAIL_BANDS = (("top 0.1", 0.0, 0.001), ("99-99.9", 0.001, 0.01), ("95-99", 0.01, 0.05))  # shares of the pool's rest
DESIGN = {"A": {"random": 60, "top 0.1": 10, "99-99.9": 10, "95-99": 20},
          "B": {"random": 36, "top 0.1": 6, "99-99.9": 6, "95-99": 12},
          "C": {"random": 36, "top 0.1": 6, "99-99.9": 6, "95-99": 12}}
B_PAIRS = tuple(range(10))                     # round 1's pairs used for the other budgets
C_PAIRS = tuple(range(10))                     # fresh pairs per grid cell
BUDGET_OFFSETS = {"percolation": (-2, 2), "goat": (-3, 3), "faq": (-3, 3)}

GRID_SIZES = (300, 600, 1200)                  # vertices
GRID_DENSITIES = (0.75, 1.0, 1.5)              # mean degree relative to round 1's (12 for percolation, 16 otherwise)
GRID_LARGE = {"goat": ("chung_lu",), "faq": ("chung_lu",)}   # GOAT and FAQ at 1,200 vertices: these models, x 1 only
LADDER = {"percolation": (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 24, 28, 34, 40, 48, 60),
          "goat": (2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 20, 24, 28, 34, 40, 48, 56, 64),
          "faq": (4, 6, 8, 10, 12, 15, 18, 21, 24, 28, 32, 36, 40, 48, 56, 64, 80)}
CALIBRATION_SETS = 3                           # random sets per pair at each budget tried
TARGET_RATE = 0.5
STALL_ITERATIONS = 3                           # GOAT runs that stop within 3 iterations count as stalled

SAMPLING_SEED_BASE = 20261103
SOLVER_SEED_BASE = 20261104
GRID_PAIR_SEED_BASE = 20261108
CALIBRATION_SEED_BASE = 20261106
PART_CODES = {"A": 1, "B": 2, "C": 3}
ALGORITHM_CODES = BN.ALGORITHM_CODES

SECONDS_PER_RUN = {"percolation": 0.5, "goat": 3.1, "faq": 2.4}   # per run and job with 3 jobs (best_of_n), n = 600
RUN_TIME_EXPONENT = {"percolation": 1.5, "goat": 2.5, "faq": 2.5}   # rough growth of run time with graph size
POOL_SECONDS = 8.0                                                  # drawing and scoring one pool
CALIBRATION_STEPS = 4.5                                             # budgets tried per calibration, on average

ALGORITHM_LABELS = BN.ALGORITHM_LABELS
MODEL_LABELS = BN.MODEL_LABELS
RULE_LABELS = BN.RULE_LABELS
PERCENTILE_BINS = MP.PERCENTILE_BINS
PERCENTILE_LABELS = MP.PERCENTILE_LABELS


def _now() -> str:
    return datetime.datetime.now().astimezone().strftime("%a %H:%M")


def _flag(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def training_budgets(path: Path | None = None) -> dict:
    """The pooled model's training budget k0 per (algorithm, model)."""

    saved = json.loads(Path(path or WEIGHTS_PATH).read_text(encoding="utf-8"))
    return {tuple(key.split("/")): int(entry["budget"]) for key, entry in saved.items()}


def offset_budgets(algorithm: str, model: str, budgets: dict) -> dict:
    """Part B's budgets: {condition: budget}, e.g. {'k0-3': 12, 'k0+3': 18}."""

    k0 = budgets[(algorithm, model)]
    return {f"k0{d:+d}": k0 + d for d in BUDGET_OFFSETS[algorithm] if k0 + d >= 1}


# --------------------------------------------------------------------------
# Changed graph settings (part C)
# --------------------------------------------------------------------------

def _chung_lu_mean_degree(alpha: float, scale: float, n: int) -> float:
    """Expected mean degree of the Chung-Lu generator (same weights, cap and clipping as regime_sweep)."""

    from utils.Graphs import pareto_quantile_weights

    w = scale * pareto_quantile_weights(n, alpha)
    w = np.minimum(w, np.sqrt(w.sum()))
    P = np.clip(np.outer(w, w) / w.sum(), 0, 1)
    np.fill_diagonal(P, 0)
    return float(P.sum() / n)


def cell_name(n: int, factor: float) -> str:
    return f"n{int(n)}_x{factor:g}"


def parse_cell(cell: str) -> tuple[int, float]:
    size, factor = cell.split("_x")
    return int(size[1:]), float(factor)


def cell_code(cell: str) -> int:
    n, factor = parse_cell(cell)
    return n * 1000 + int(round(factor * 100))


def grid_cells(algorithm: str, model: str) -> list[str]:
    """Part C's cells for one algorithm and model, smallest graphs first."""

    cells = []
    for n in GRID_SIZES:
        for factor in GRID_DENSITIES:
            if n == 600 and factor == 1.0:
                continue                                  # round 1's setting: part A
            if algorithm != "percolation" and n > 600 and (factor != 1.0 or model not in GRID_LARGE.get(algorithm, ())):
                continue
            cells.append(cell_name(n, factor))
    return cells


@functools.lru_cache(maxsize=None)
def base_mean_degree(algorithm: str, model: str) -> float:
    """Expected mean degree of round 1's setting (12 for percolation, 16 for GOAT and FAQ)."""

    base = S.graph_config(algorithm, model)
    n = base.vertex_count
    if model == "sparse_er":
        return float(base.er_probability * (n - 1))
    if model == "paper_pa":
        return float(2 * (n - 1) / n + base.paper_pa_probability * (n - 1))
    return round(_chung_lu_mean_degree(base.chung_lu_alpha, base.chung_lu_weight_scale, n), 4)


@functools.lru_cache(maxsize=None)
def grid_config(algorithm: str, model: str, cell: str):
    """Round 1's setting with the cell's size and mean degree; shape parameters and rho unchanged."""

    n, factor = parse_cell(cell)
    base = S.graph_config(algorithm, model)
    degree = factor * base_mean_degree(algorithm, model)
    if model == "sparse_er":
        return replace(base, vertex_count=n, er_probability=degree / (n - 1))
    if model == "paper_pa":
        return replace(base, vertex_count=n, paper_pa_probability=(degree - 2 * (n - 1) / n) / (n - 1))
    if model == "chung_lu":
        return replace(base, vertex_count=n, chung_lu_weight_scale=RS.weight_scale(base.chung_lu_alpha, n, degree))
    raise ValueError(f"Unknown model: {model}")


def cell_label(cell: str) -> str:
    n, factor = parse_cell(cell)
    return f"{n:,} vertices, degree x{factor:g}"


def condition_label(part: str, condition: str) -> str:
    """How a task's condition is shown in progress lines: nothing for part A, the budget offset for part B and the
    grid cell for part C."""

    if part == "B":
        return f" {condition}"
    if part == "C":
        return f" {cell_label(condition)}"
    return ""


def describe_cell(algorithm: str, model: str, cell: str) -> str:
    n, factor = parse_cell(cell)
    return f"{n:,} vertices, mean degree {factor * base_mean_degree(algorithm, model):.1f} (x{factor:g})"


def grid_pair(algorithm: str, model: str, cell: str, pair_index: int):
    """A fresh, structure-checked pair in this cell (GOAT and FAQ share pairs, as in round 1)."""

    config = grid_config(algorithm, model, cell)
    family = S.FAMILY_CODES[S.GRAPH_FAMILY[algorithm]]
    first = rc._seed(GRID_PAIR_SEED_BASE, family, S.MODEL_CODES[model], cell_code(cell), pair_index)
    first %= 2**32 - 10_000
    offset = 0
    while True:
        seed, A, B, truth, _report, _rejected = first_passing_pair(
            lambda s: generate_graph_pair(config, s), first + offset)
        if seed not in S.KNOWN_PAIR_SEEDS:
            break
        offset = seed - first + 1
    return int(seed), A, B, truth


# --------------------------------------------------------------------------
# The pool of random sets and the sets chosen from it
# --------------------------------------------------------------------------

class Pool:
    """POOL_SIZE distinct uniform random seed sets of one size on one graph, scored with the pooled weights."""

    def __init__(self, graph: GraphA, algorithm: str, budget: int, weights: dict, rng: np.random.Generator,
                 size: int = POOL_SIZE):
        self.calculator = MP.PoolFeatures(graph, MP.UNIQUENESS_CAP[algorithm])
        self.sets = MP.draw_pool(rng, graph.n, budget, size)
        self.values = self.calculator.compute(self.sets)
        self.mean = {f: float(self.values[f].mean()) for f in CORE}
        self.sd = {f: float(self.values[f].std(ddof=1)) for f in CORE}
        self.weights = np.array([weights[f] for f in CORE], dtype=float)
        self.scores = self._z(self.values) @ self.weights
        self._sorted_scores = np.sort(self.scores)
        self._sorted_cut = np.sort(self.values["cut_edges"])

    def _z(self, values: dict) -> np.ndarray:
        return np.column_stack([(values[f] - self.mean[f]) / self.sd[f] if self.sd[f] > 0
                                else np.zeros(len(values[f])) for f in CORE])

    def describe(self, values: dict) -> pd.DataFrame:
        """z-scores, score, the score's percentile in the pool and the degree's percentile in the pool."""

        z = self._z(values)
        score = z @ self.weights
        table = pd.DataFrame(z, columns=[f"z_{f}" for f in CORE])
        table["score"] = score
        size = len(self._sorted_scores)
        table["score_percentile"] = 100 * np.searchsorted(self._sorted_scores, score, side="right") / size
        table["degree_percentile"] = 100 * np.searchsorted(self._sorted_cut, values["cut_edges"], side="right") / size
        return table

    def describe_pool(self, indices) -> pd.DataFrame:
        indices = np.asarray(indices, dtype=int)
        return self.describe({key: value[indices] for key, value in self.values.items()})

    def describe_sets(self, sets: np.ndarray) -> pd.DataFrame:
        return self.describe(self.calculator.compute(np.atleast_2d(np.asarray(sets, dtype=np.int64))))

    def reference(self) -> dict:
        return {**{f"pool_mean_{f}": self.mean[f] for f in CORE}, **{f"pool_sd_{f}": self.sd[f] for f in CORE}}


def choose(pool: Pool, design: dict) -> list[dict]:
    """The random sets (the first ones drawn) and the tail sets (evenly in rank within each band of the rest)."""

    n_random = int(design["random"])
    chosen = [{"group": "random", "pool_index": i, "band": "", "tail_rank": np.nan} for i in range(n_random)]
    rest = np.arange(n_random, len(pool.sets))
    order = rest[np.argsort(-pool.scores[rest], kind="stable")]
    for band, low, high in TAIL_BANDS:
        count = int(design.get(band, 0))
        if count <= 0:
            continue
        start = int(round(low * len(order)))
        stop = min(len(order), max(int(round(high * len(order))), start + count))
        for position in np.unique(np.linspace(start, stop - 1, count).round().astype(int)):
            chosen.append({"group": "tail", "pool_index": int(order[position]), "band": band,
                           "tail_rank": int(position) + 1})
    return chosen


@functools.lru_cache(maxsize=None)
def _round1_orderings(algorithm: str, model: str) -> pd.DataFrame:
    return pd.read_csv(ROUND1_ROOT / algorithm / model / "orderings.csv")


def round1_rule_sets(algorithm: str, model: str, pair_index: int, pair_seed: int, graph: GraphA,
                     budget: int) -> dict:
    """Round 1's rules at this budget: regenerated from round 1's random streams, checked against its orderings."""

    stored = _round1_orderings(algorithm, model)
    stored = stored[stored["pair_index"] == pair_index].set_index("strategy")["ordering"]
    sets = {}
    for rule in S.STRATEGIES["main"][algorithm]:
        if rule == "random":
            continue
        rng = np.random.default_rng([S.STRATEGY_SEED_BASE, pair_seed, STRATEGY_CODES[rule]])
        order = [int(v) for v in ordering(rule, graph, budget, rng, pool_factor=S.POOL_FACTOR,
                                          hub_count=S.HUB_COUNT)]
        saved = json.loads(stored[rule])
        shared = min(len(saved), budget)
        if order[:shared] != saved[:shared]:
            raise RuntimeError(f"{algorithm}/{model} pair {pair_index}: {rule} does not reproduce round 1's ordering.")
        sets[rule] = np.array(sorted(order[:budget]), dtype=np.int64)
    return sets


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------

def result_path(part: str, algorithm: str, model: str, condition: str, pair_index: int) -> Path:
    folder = OUTPUT_DIR / part / algorithm / model
    if part != "A":
        folder = folder / condition
    return folder / f"pair_{pair_index:03d}.csv"


def calibration_dir(algorithm: str, model: str, cell: str) -> Path:
    return OUTPUT_DIR / "C" / algorithm / model / cell


def _graph_info(A: np.ndarray, B: np.ndarray, truth: np.ndarray, config) -> dict:
    degree = (A > 0).sum(axis=1)
    aligned = B[np.ix_(truth, truth)]
    return {"rho": float(config.rho), "hash_a": rc._hash(A), "hash_b": rc._hash(B),
            "mean_degree_a": float(degree.mean()), "max_degree_a": int(degree.max()),
            "kept_edge_fraction": float(((A > 0) & (aligned > 0)).sum() / (A > 0).sum())}


def set_job(part: str, algorithm: str, model: str, condition: str, pair_index: int, budget: int, weights: dict,
            verify: bool = True, blas_threads: int = 3) -> str:
    """Draw and score the pool on one pair at one budget, then run the chosen sets once each."""

    path = result_path(part, algorithm, model, condition, pair_index)
    name = (f"{part} · {ALGORITHM_LABELS[algorithm]} {MODEL_LABELS[model]}{condition_label(part, condition)}"
            f" · pair {pair_index} · k = {budget}")
    if path.exists():
        return f"{name}: already done"
    started = time.perf_counter()
    with thread_limit(blas_threads):
        if part == "C":
            seed, A, B, truth = grid_pair(algorithm, model, condition, pair_index)
            config = grid_config(algorithm, model, condition)
            saved = json.loads((calibration_dir(algorithm, model, condition) / "calibration.json").read_text())
            expected = {p["pair_index"]: p for p in saved["pairs"]}[pair_index]
            if (expected["pair_seed"], expected["hash_a"], expected["hash_b"]) != (seed, rc._hash(A), rc._hash(B)):
                raise RuntimeError(f"{name}: the pair differs from the one calibrated.")
        else:
            seed, A, B, truth = H.load_pair(model, pair_index, verify, algorithm)
            config = S.graph_config(algorithm, model)
        graph = GraphA(A)
        rng = np.random.default_rng([SAMPLING_SEED_BASE, seed, budget, ALGORITHM_CODES[algorithm], PART_CODES[part],
                                     cell_code(condition) if part == "C" else 0])
        pool = Pool(graph, algorithm, budget, weights, rng)
        chosen = choose(pool, DESIGN[part])
        described = pool.describe_pool([c["pool_index"] for c in chosen])
        candidates = [(c, np.sort(pool.sets[c["pool_index"]]), row)
                      for c, row in zip(chosen, described.to_dict("records"))]
        if part == "A":
            rules = round1_rule_sets(algorithm, model, pair_index, seed, graph, budget)
            described = pool.describe_sets(np.array(list(rules.values())))
            candidates += [({"group": "rule", "rule": rule, "pool_index": np.nan, "band": "", "tail_rank": np.nan},
                            seeds, row) for (rule, seeds), row in zip(rules.items(), described.to_dict("records"))]
        common = {"part": part, "condition": condition, "algorithm": algorithm, "model": model,
                  "pair_index": pair_index, "pair_seed": seed, "budget": budget, "vertices": int(A.shape[0]),
                  **_graph_info(A, B, truth, config), **pool.reference()}
        pair = (A, B, truth, seed)
        rows = []
        for candidate_id, (candidate, seeds, info) in enumerate(candidates):
            seed_value = rc._seed(SOLVER_SEED_BASE, seed, budget, candidate_id, PART_CODES[part])
            out = MP.evaluate(algorithm, model, pair, seeds, seed_value)
            rows.append({**common, "candidate_id": candidate_id, "group": candidate["group"],
                         "rule": candidate.get("rule", ""), "band": candidate["band"],
                         "tail_rank": candidate["tail_rank"], "pool_index": candidate["pool_index"],
                         "seed_vertices": json.dumps([int(v) for v in seeds]), **info,
                         "solver_seed": out["solver_seed"], "success": bool(out["success"]),
                         "success_090": bool(out["success_090"]), "accuracy": out["final_unseeded_accuracy"],
                         "objective": out["objective"], "n_iter": out["n_iter"], "converged": out["converged"],
                         "runtime_seconds": out["runtime_seconds"], **seed_set_features(graph, seeds)})
    _write(path, rows)
    return f"{name}: {len(rows)} runs in {(time.perf_counter() - started) / 60:.1f} min"


def calibration_job(algorithm: str, model: str, cell: str, base_budget: int, blas_threads: int = 3) -> str:
    """Find the budget where random sets succeed about half the time in a grid cell (part C)."""

    folder = calibration_dir(algorithm, model, cell)
    result = folder / "calibration.json"
    name = f"C · {ALGORITHM_LABELS[algorithm]} {MODEL_LABELS[model]} {cell_label(cell)} · budget calibration"
    if result.exists():
        return f"{name}: already done"
    started = time.perf_counter()
    with thread_limit(blas_threads):
        pairs = [grid_pair(algorithm, model, cell, i) for i in C_PAIRS]
        config = grid_config(algorithm, model, cell)
        rates, runs = {}, [0]

        def rate(budget: int) -> float:
            if budget in rates:
                return rates[budget]
            path = folder / "calibration" / f"k{budget:02d}.csv"
            if path.exists():
                table = pd.read_csv(path)
            else:
                rows = []
                for pair_index, (seed, A, B, truth) in zip(C_PAIRS, pairs):
                    rng = np.random.default_rng([CALIBRATION_SEED_BASE, seed, budget, ALGORITHM_CODES[algorithm],
                                                 cell_code(cell)])
                    for j, seeds in enumerate(MP.draw_pool(rng, A.shape[0], budget, CALIBRATION_SETS)):
                        out = MP.evaluate(algorithm, model, (A, B, truth, seed), seeds,
                                          rc._seed(CALIBRATION_SEED_BASE, seed, budget, j))
                        runs[0] += 1
                        rows.append({"algorithm": algorithm, "model": model, "cell": cell,
                                     "pair_index": pair_index, "pair_seed": seed, "budget": budget, "set": j,
                                     "seed_vertices": json.dumps([int(v) for v in seeds]),
                                     "solver_seed": out["solver_seed"], "success": bool(out["success"]),
                                     "accuracy": out["final_unseeded_accuracy"], "n_iter": out["n_iter"],
                                     "runtime_seconds": out["runtime_seconds"]})
                _write(path, rows)
                table = pd.DataFrame(rows)
            rates[budget] = float(_flag(table["success"]).mean())
            return rates[budget]

        chosen, status = calibrate(LADDER[algorithm], base_budget, rate)
        write_json(result, {
            "algorithm": algorithm, "model": model, "cell": cell,
            "change": describe_cell(algorithm, model, cell),
            "graph_config": json.loads(json.dumps(asdict(config))), "base_budget": base_budget,
            "ladder": list(LADDER[algorithm]), "sets_per_budget": CALIBRATION_SETS * len(C_PAIRS),
            "rates": {str(b): rates[b] for b in sorted(rates)}, "budget": chosen, "status": status,
            "pairs": [{"pair_index": i, "pair_seed": seed, "hash_a": rc._hash(A), "hash_b": rc._hash(B)}
                      for i, (seed, A, B, _truth) in zip(C_PAIRS, pairs)],
            "finished_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")})
    shown = ", ".join(f"{b}: {rates[b]:.2f}" for b in sorted(rates))
    outcome = f"budget {chosen}" if chosen else "skipped"
    return (f"{name}: {outcome} ({status}; random success by budget {shown}); {runs[0]} runs in "
            f"{(time.perf_counter() - started) / 60:.1f} min")


def calibrate(ladder, start: int, rate_of, target: float = TARGET_RATE) -> tuple[int | None, str]:
    """Galloping search along the ladder for the success rate's crossing of the target, then halving the gap.

    rate_of(budget) gives the random sets' success rate at a budget. Returns the budget, among those tried, whose
    rate is closest to the target (ties: the smaller budget), and a status.
    """

    seen = {}

    def tried(budget: int) -> float:
        seen[budget] = rate_of(budget)
        return seen[budget]

    ladder = sorted(set(int(b) for b in ladder))
    i = int(np.argmin([abs(b - start) for b in ladder]))
    below = above = None
    step = 1
    while True:
        if tried(ladder[i]) < target:
            below = i
            if above is not None or i == len(ladder) - 1:
                break
            i = min(i + step, len(ladder) - 1)
        else:
            above = i
            if below is not None or i == 0:
                break
            i = max(i - step, 0)
        step *= 2
    if above is None:      # below the target all the way to the top of the ladder
        if seen[ladder[-1]] == 0:
            return None, "no random set succeeded at the top of the ladder"
        return ladder[-1], "below 50% at the top of the ladder"
    if below is None:      # above the target all the way down
        if seen[ladder[0]] == 1:
            return None, "every random set succeeded at the bottom of the ladder"
        return ladder[0], "above 50% at the bottom of the ladder"
    if below > above:      # the rates are noisy; keep the bracket in order
        below, above = above, below
    while above - below > 1:
        middle = (below + above) // 2
        if tried(ladder[middle]) < target:
            below = middle
        else:
            above = middle
    low, high = ladder[below], ladder[above]
    if high - low >= 3:    # one more step between two ladder budgets that are far apart
        tried((low + high) // 2)
    return min(seen, key=lambda b: (abs(seen[b] - target), b)), "ok"


# --------------------------------------------------------------------------
# Running everything
# --------------------------------------------------------------------------

def _settings_snapshot(weights_path: Path, budgets: dict, weights: dict) -> dict:
    return {
        "weights_per_sd": {f"{a}/{m}": w for (a, m), w in weights.items()},
        "training_budgets": {f"{a}/{m}": k for (a, m), k in budgets.items()},
        "pool_size": POOL_SIZE, "tail_bands": [list(b) for b in TAIL_BANDS], "design": DESIGN,
        "b_pairs": list(B_PAIRS), "c_pairs": list(C_PAIRS), "budget_offsets": BUDGET_OFFSETS,
        "grid": {"sizes": list(GRID_SIZES), "mean_degree_factors": list(GRID_DENSITIES),
                 "large_graphs_for_sgm": {a: list(m) for a, m in GRID_LARGE.items()}},
        "ladder": LADDER, "calibration_sets_per_pair": CALIBRATION_SETS, "target_rate": TARGET_RATE,
        "seed_bases": {"sampling": SAMPLING_SEED_BASE, "solver": SOLVER_SEED_BASE, "grid_pairs": GRID_PAIR_SEED_BASE,
                       "calibration": CALIBRATION_SEED_BASE},
        "uniqueness_cap": MP.UNIQUENESS_CAP, "goat_regularizer": dict(S.GOAT_REGULARIZER),
        "success_threshold": dict(S.SUCCESS_THRESHOLD), "runs_per_set": 1,
    }


ADDITIVE_SETTINGS = ("b_pairs", "c_pairs", "budget_offsets", "grid")   # more pairs, budgets or cells add results
PART_C_SETTINGS = ("ladder", "calibration_sets_per_pair", "target_rate")
PART_C_SEEDS = ("grid_pairs", "calibration")


def _has_results(part: str) -> bool:
    return any((OUTPUT_DIR / part).glob("*/*/**/*.csv")) or any((OUTPUT_DIR / part).glob("*/*/*.csv"))


def _check_settings(snapshot: dict) -> None:
    """Refuse to mix results made with different settings; part C's settings only matter once part C has results."""

    path = OUTPUT_DIR / "settings.json"
    if not path.exists():
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    part_c = _has_results("C")
    plain = lambda value: json.loads(json.dumps(value))  # noqa: E731
    changed = []
    for key, value in snapshot.items():
        if key in ADDITIVE_SETTINGS or (key in PART_C_SETTINGS and not part_c):
            continue
        if key == "seed_bases":
            keep = [k for k in value if part_c or k not in PART_C_SEEDS]
            if any(plain(value[k]) != saved.get(key, {}).get(k) for k in keep):
                changed.append(key)
        elif plain(value) != saved.get(key):
            changed.append(key)
    if changed:
        raise RuntimeError(f"{OUTPUT_DIR} holds results with different settings ({', '.join(sorted(changed))}). "
                           "Set SCORE_TRANSFER_OUTPUT_DIR to a new folder, or move the old one away.")


def _calibration(algorithm: str, model: str, cell: str) -> dict | None:
    path = calibration_dir(algorithm, model, cell) / "calibration.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _c_pair_tasks(algorithm: str, model: str, cell: str) -> list[tuple]:
    saved = _calibration(algorithm, model, cell)
    if not saved or not saved["budget"]:
        return []
    return [("C", algorithm, model, cell, i, int(saved["budget"])) for i in C_PAIRS
            if not result_path("C", algorithm, model, cell, i).exists()]


def planned_tasks(parts=PARTS, algorithms=ALGORITHMS, models=MODELS, budgets: dict | None = None) -> list[tuple]:
    """Tasks not done yet: (part, algorithm, model, condition, pair_index, budget); calibrations have pair None."""

    budgets = budgets or training_budgets()
    tasks = []
    if "A" in parts:
        part = [("A", a, m, "training", i, budgets[(a, m)]) for a in algorithms for m in models
                for i in H.round1_pairs(a, m)]
        tasks += sorted(part, key=lambda t: (t[4], ALGORITHMS.index(t[1]), MODELS.index(t[2])))
    if "B" in parts:
        part = [("B", a, m, c, i, k) for a in algorithms for m in models
                for c, k in offset_budgets(a, m, budgets).items() for i in B_PAIRS]
        tasks += sorted(part, key=lambda t: (t[4], ALGORITHMS.index(t[1]), MODELS.index(t[2]), t[3]))
    if "C" in parts:
        cells = [(a, m, cell) for a in algorithms for m in models for cell in grid_cells(a, m)]
        cells.sort(key=lambda c: (parse_cell(c[2])[0], ALGORITHMS.index(c[0]), MODELS.index(c[1])))  # small first
        for a, m, cell in cells:
            if _calibration(a, m, cell) is None:
                tasks.append(("calibration", a, m, cell, None, budgets[(a, m)]))
            else:
                tasks += _c_pair_tasks(a, m, cell)
    return [t for t in tasks if t[0] == "calibration" or not result_path(t[0], t[1], t[2], t[3], t[4]).exists()]


def _task_seconds(task: tuple) -> float:
    part, algorithm, model, condition, *_ = task
    per_run = SECONDS_PER_RUN[algorithm]
    if part in ("calibration", "C"):
        per_run *= (parse_cell(condition)[0] / 600) ** RUN_TIME_EXPONENT[algorithm]
    if part == "calibration":
        later = len(C_PAIRS) * (POOL_SECONDS + sum(DESIGN["C"].values()) * per_run)  # its pairs, still to come
        return CALIBRATION_STEPS * CALIBRATION_SETS * len(C_PAIRS) * per_run + later
    rules = len(S.STRATEGIES["main"][algorithm]) - 1 if part == "A" else 0
    return POOL_SECONDS + (sum(DESIGN[part].values()) + rules) * per_run


def estimate_minutes(tasks, jobs: int = 3) -> float:
    return sum(_task_seconds(t) for t in tasks) / max(jobs, 1) / 60


def _execute(task: tuple, weights: dict, verify: bool, blas_threads: int) -> str:
    part, algorithm, model, condition, pair_index, budget = task
    if part == "calibration":
        return calibration_job(algorithm, model, condition, budget, blas_threads)
    return set_job(part, algorithm, model, condition, pair_index, budget, weights[(algorithm, model)], verify,
                   blas_threads)


def _describe(task: tuple) -> str:
    part, algorithm, model, condition, pair_index, budget = task
    where = f"{ALGORITHM_LABELS[algorithm]} {MODEL_LABELS[model]}"
    if part == "calibration":
        return f"C · {where} {cell_label(condition)} · budget calibration"
    return f"{part} · {where}{condition_label(part, condition)} · pair {pair_index}"


def run_all(parts=PARTS, algorithms=ALGORITHMS, models=MODELS, jobs: int = 3, blas_threads: int = 3,
            verify: bool = True, weights_path: Path | None = None) -> None:
    """Run every task that is not done yet, `jobs` at a time, in the order A, B, C."""

    parts = [p for p in PARTS if p in set(parts)]
    algorithms = [a for a in ALGORITHMS if a in set(algorithms)]
    models = [m for m in MODELS if m in set(models)]
    weights_path = Path(weights_path or WEIGHTS_PATH)
    budgets = training_budgets(weights_path)
    all_weights = BN.load_weights(weights_path)
    snapshot = _settings_snapshot(weights_path, budgets, all_weights)
    _check_settings(snapshot)
    tasks = planned_tasks(parts, algorithms, models, budgets)
    if not tasks:
        print(f"Everything is done. Results are in {OUTPUT_DIR}")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "settings.json", {**snapshot, "weights_path": str(weights_path),
                                              "environment": collect_environment(blas_threads),
                                              "written_at": datetime.datetime.now().astimezone().isoformat(
                                                  timespec="seconds")})
    for part in parts:
        mine = [t for t in tasks if t[0] == part or (part == "C" and t[0] == "calibration")]
        if mine:
            print(f"{_now()}  Part {part}: {len(mine)} tasks, about {estimate_minutes(mine, jobs):.0f} min", flush=True)
    print(f"{_now()}  About {estimate_minutes(tasks, jobs) / 60:.1f} hours in all with {jobs} jobs "
          "(part C's budgets are calibrated first, so its time is a guess). Each line below is one finished task.",
          flush=True)
    keep_awake = None
    if sys.platform == "darwin":
        import shutil
        import subprocess
        if shutil.which("caffeinate"):
            keep_awake = subprocess.Popen(["caffeinate", "-i"])
    failures, started, finished_count = [], time.perf_counter(), [0]
    total = [len(tasks)]

    def report(task, message, error, remaining):
        finished_count[0] += 1
        n = f"[{finished_count[0]}/{total[0]}]"
        if error is None:
            print(f"{_now()}  {n} {message}; about {estimate_minutes(remaining, jobs):.0f} min left", flush=True)
            return
        try:
            failures.append(_describe(task))
        except Exception:  # noqa: BLE001 - a label must never hide the real error
            failures.append(str(task))
        print(f"{_now()}  {n} {failures[-1]} FAILED:", flush=True)
        traceback.print_exception(type(error), error, error.__traceback__)

    def follow_ups(task):
        if task[0] != "calibration":
            return []
        new = _c_pair_tasks(task[1], task[2], task[3])
        total[0] += len(new)
        return new

    pool = None

    def stop_workers():
        if pool is not None:
            workers = list((getattr(pool, "_processes", None) or {}).values())
            pool.shutdown(wait=False, cancel_futures=True)
            for worker in workers:
                worker.terminate()

    try:
        if jobs <= 1:
            queue = list(tasks)
            while queue:
                task = queue.pop(0)
                try:
                    message = _execute(task, all_weights, verify, blas_threads)
                except Exception as error:  # noqa: BLE001 - report every failure, keep going
                    report(task, None, error, [])
                else:
                    queue += follow_ups(task)
                    report(task, message, None, queue)
        else:
            import multiprocessing
            from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
            pool = ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn"),
                                       initializer=BN._ignore_interrupts)
            futures = {pool.submit(_execute, task, all_weights, verify, blas_threads): task for task in tasks}
            while futures:
                done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
                for future in done:
                    task = futures.pop(future)
                    try:
                        message = future.result()
                    except Exception as error:  # noqa: BLE001 - report every failure, keep going
                        report(task, None, error, [])
                        continue
                    for new in follow_ups(task):
                        futures[pool.submit(_execute, new, all_weights, verify, blas_threads)] = new
                    report(task, message, None, list(futures.values()))
            pool.shutdown(wait=True)
    except KeyboardInterrupt:
        # Stop the workers too: the notebook's interrupt button only reaches this process.
        stop_workers()
        print(f"\n{_now()}  Stopped. Finished tasks are kept; rerun to resume.", flush=True)
        return
    except BaseException:
        stop_workers()     # no workers left running in the background
        raise
    finally:
        if keep_awake is not None:
            keep_awake.terminate()
    if failures:
        raise RuntimeError(f"Failed: {'; '.join(failures)}. Rerun to retry them (finished tasks are kept).")
    print(f"{_now()}  Done in {(time.perf_counter() - started) / 60:.0f} min. Results are in {OUTPUT_DIR}", flush=True)


def status(parts=PARTS, algorithms=ALGORITHMS, models=MODELS) -> pd.DataFrame:
    budgets = training_budgets()
    rows = []
    for algorithm in algorithms:
        for model in models:
            if "A" in parts:
                planned = H.round1_pairs(algorithm, model)
                rows.append({"part": "A", "algorithm": ALGORITHM_LABELS[algorithm], "model": MODEL_LABELS[model],
                             "condition": f"training budget, k = {budgets[(algorithm, model)]}",
                             "pairs planned": len(planned),
                             "pairs done": sum(result_path("A", algorithm, model, "training", i).exists()
                                               for i in planned)})
            if "B" in parts:
                for condition, k in offset_budgets(algorithm, model, budgets).items():
                    rows.append({"part": "B", "algorithm": ALGORITHM_LABELS[algorithm],
                                 "model": MODEL_LABELS[model], "condition": f"{condition}, k = {k}",
                                 "pairs planned": len(B_PAIRS),
                                 "pairs done": sum(result_path("B", algorithm, model, condition, i).exists()
                                                   for i in B_PAIRS)})
            if "C" in parts:
                for cell in grid_cells(algorithm, model):
                    saved = _calibration(algorithm, model, cell)
                    if saved is None:
                        condition, planned = f"{cell_label(cell)}: budget not calibrated yet", len(C_PAIRS)
                    elif not saved["budget"]:
                        condition, planned = f"{cell_label(cell)}: skipped ({saved['status']})", 0
                    else:
                        condition, planned = f"{cell_label(cell)}, k = {saved['budget']}", len(C_PAIRS)
                    rows.append({"part": "C", "algorithm": ALGORITHM_LABELS[algorithm],
                                 "model": MODEL_LABELS[model], "condition": condition, "pairs planned": planned,
                                 "pairs done": sum(result_path("C", algorithm, model, cell, i).exists()
                                                   for i in C_PAIRS)})
    return pd.DataFrame(rows).set_index(["part", "algorithm", "model", "condition"])


# --------------------------------------------------------------------------
# Analysis helpers for score_transfer.ipynb
# --------------------------------------------------------------------------

def read_sets(parts=PARTS, algorithms=ALGORITHMS, models=MODELS) -> pd.DataFrame:
    """Every run so far, one row per seed set."""

    patterns = {"A": "A/*/*/pair_*.csv", "B": "B/*/*/*/pair_*.csv", "C": "C/*/*/*/pair_*.csv"}
    files = [f for part in parts for f in sorted(OUTPUT_DIR.glob(patterns[part]))]
    if not files:
        raise FileNotFoundError(f"No results in {OUTPUT_DIR} yet. Run run_all() first.")
    data = pd.concat([pd.read_csv(f, keep_default_na=False, na_values=[""]) for f in files], ignore_index=True)
    data = data[data["algorithm"].isin(algorithms) & data["model"].isin(models)].copy()
    for column in ("success", "success_090"):
        data[column] = _flag(data[column])
    for column in ("rule", "band"):
        data[column] = data[column].fillna("").astype(str)
    order = {"_part": data["part"].map({p: i for i, p in enumerate(PARTS)}),
             "_algorithm": data["algorithm"].map({a: i for i, a in enumerate(ALGORITHMS)}),
             "_model": data["model"].map({m: i for i, m in enumerate(MODELS)}),
             "_condition": data["condition"].map(lambda c: cell_code(c) if str(c).startswith("n") else 0)}
    data = data.assign(**order).sort_values([*order, "budget", "pair_index", "candidate_id"], kind="stable")
    return data.drop(columns=list(order)).reset_index(drop=True)


def calibration_table() -> pd.DataFrame:
    rows = []
    for path in sorted(OUTPUT_DIR.glob("C/*/*/*/calibration.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        rates = {int(k): v for k, v in saved["rates"].items()}
        chosen = saved["budget"]
        rows.append({"algorithm": ALGORITHM_LABELS[saved["algorithm"]], "model": MODEL_LABELS[saved["model"]],
                     "cell": saved["cell"], "setting": saved["change"], "training budget": saved["base_budget"],
                     "budget used": chosen if chosen else "skipped",
                     "random success there": round(rates[chosen], 2) if chosen else np.nan,
                     "budgets tried (success rate)": ", ".join(f"{b} ({r:.2f})" for b, r in sorted(rates.items())),
                     "status": saved["status"]})
    if not rows:
        return pd.DataFrame()
    order = {label: i for i, label in enumerate(ALGORITHM_LABELS.values())}
    table = pd.DataFrame(rows)
    table["_a"] = table["algorithm"].map(order)
    table["_m"] = table["model"].map({label: i for i, label in enumerate(MODEL_LABELS.values())})
    table["_c"] = table["cell"].map(cell_code)
    return table.sort_values(["_a", "_m", "_c"]).drop(columns=["_a", "_m", "_c"]).set_index(
        ["algorithm", "model", "cell"])


def calibration_runs() -> pd.DataFrame:
    files = sorted(OUTPUT_DIR.glob("C/*/*/*/calibration/k*.csv"))
    if not files:
        return pd.DataFrame()
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    runs["success"] = _flag(runs["success"])
    return runs


def check_table(data: pd.DataFrame) -> pd.DataFrame:
    """Sets run per setting and condition, and the score's range on the random sets (a sanity check)."""

    rows = []
    for keys, part in data.groupby(["part", "algorithm", "model", "condition", "budget"], sort=False):
        random = part[part["group"] == "random"]
        rows.append({"part": keys[0], "algorithm": ALGORITHM_LABELS[keys[1]], "model": MODEL_LABELS[keys[2]],
                     "condition": keys[3], "budget": keys[4], "pairs": part["pair_index"].nunique(),
                     "random sets": len(random), "tail sets": int((part["group"] == "tail").sum()),
                     "rule sets": int((part["group"] == "rule").sum()),
                     "random sets' mean percentile": round(float(random["score_percentile"].mean()), 1),
                     "lowest tail percentile": round(float(part.loc[part["group"] == "tail",
                                                                    "score_percentile"].min()), 2)})
    return pd.DataFrame(rows).set_index(["part", "algorithm", "model", "condition"])


def _bins(percentiles: pd.Series) -> pd.Series:
    return pd.cut(percentiles, PERCENTILE_BINS, labels=PERCENTILE_LABELS, right=False)


def percentile_curves(data: pd.DataFrame) -> pd.DataFrame:
    """Success rate of random and tail sets by their score's percentile in the pool, per condition."""

    part = data[data["group"].isin(["random", "tail"])].copy()
    part["bin"] = _bins(part["score_percentile"])
    keys = ["part", "algorithm", "model", "condition", "budget"]
    table = part.groupby([*keys, "bin"], observed=False)["success"].agg(["mean", "size"]).reset_index()
    table = table.rename(columns={"mean": "success", "size": "sets"})
    table["bin"] = table["bin"].astype(str)
    table["position"] = table["bin"].map({label: i for i, label in enumerate(PERCENTILE_LABELS)}).astype(int)
    random = part[part["group"] == "random"].groupby(keys)["success"].mean().rename("random")
    return table.join(random, on=keys)


def _auc(y: np.ndarray, x: np.ndarray) -> float:
    """Area under the ROC curve (ties count half), as sklearn's roc_auc_score."""

    from scipy.stats import rankdata

    y = np.asarray(y, dtype=bool)
    positives, negatives = int(y.sum()), int((~y).sum())
    if positives == 0 or negatives == 0:
        return np.nan
    ranks = rankdata(np.asarray(x, dtype=float))
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _interval(values, n_boot: int = 2000, seed: int = 2026):
    return BN.BT._mean_interval(np.asarray(values, float)[np.isfinite(np.asarray(values, float))], n_boot, seed)


def _fmt(mean, low, high, signed: bool = False) -> str:
    if not np.isfinite(mean):
        return ""
    form = "{:+.2f}" if signed else "{:.2f}"
    return f"{form.format(mean)} [{form.format(low)}, {form.format(high)}]"


def ranking_table(data: pd.DataFrame, top_share: float = 0.2) -> pd.DataFrame:
    """How well the score ranks the random sets, against degree alone, per setting and condition.

    AUC: within each pair (pairs where random sets both succeed and fail), the chance that a successful random set
    scores higher than a failed one; 0.5 is no better than chance. 'Top fifth': success of the random sets in the top
    20% of their pair by the score, and by degree alone (seed-to-nonseed edges). Intervals: 95% bootstrap over pairs.
    """

    rows = []
    for keys, part in data.groupby(["part", "algorithm", "model", "condition", "budget"], sort=False):
        random = part[part["group"] == "random"]
        auc_score, auc_degree, top_score, top_degree, both = [], [], [], [], 0
        for _, sub in random.groupby("pair_index"):
            y = sub["success"].to_numpy(bool)
            if 0 < y.mean() < 1:
                both += 1
                auc_score.append(_auc(y, sub["score"]))
                auc_degree.append(_auc(y, sub["z_degree"]))
            count = max(1, int(round(top_share * len(sub))))
            top_score.append(sub.nlargest(count, "score")["success"].mean())
            top_degree.append(sub.nlargest(count, "z_degree")["success"].mean())
        difference = np.array(auc_score) - np.array(auc_degree)
        score_interval, degree_interval = _interval(auc_score), _interval(auc_degree)
        rows.append({
            "part": keys[0], "algorithm": keys[1], "model": keys[2], "condition": keys[3], "budget": keys[4],
            "pairs": random["pair_index"].nunique(), "random success": round(float(random["success"].mean()), 2),
            "pairs with both outcomes": both,
            "AUC score": _fmt(*score_interval), "AUC degree": _fmt(*degree_interval),
            "AUC score - degree": _fmt(*_interval(difference), signed=True),
            "top fifth by score": _fmt(*_interval(top_score)), "top fifth by degree": _fmt(*_interval(top_degree)),
            **_auc_numbers(score_interval, degree_interval),
        })
    return pd.DataFrame(rows)


def _auc_numbers(score_interval, degree_interval) -> dict:
    """The AUCs as numbers, for plot_auc."""

    return dict(zip(("auc_score", "auc_score_low", "auc_score_high", "auc_degree", "auc_degree_low",
                     "auc_degree_high"), (*score_interval, *degree_interval)))


RANKING_COLUMNS = ("random success", "pairs with both outcomes", "AUC score", "AUC degree", "AUC score - degree",
                   "top fifth by score", "top fifth by degree")


def show_ranking(ranking: pd.DataFrame) -> pd.DataFrame:
    """ranking_table() for display: readable labels, numbers as text with 95% intervals."""

    table = ranking.copy()
    table["algorithm"] = table["algorithm"].map(ALGORITHM_LABELS)
    table["model"] = table["model"].map(MODEL_LABELS)
    return table.set_index(["part", "algorithm", "model", "condition", "budget"])[list(RANKING_COLUMNS)]


def lift_table(data: pd.DataFrame) -> pd.DataFrame:
    """Success of the random sets and of the tail sets in each band (pairs' averages, 95% bootstrap over pairs)."""

    rows = []
    for keys, part in data.groupby(["part", "algorithm", "model", "condition", "budget"], sort=False):
        row = {"part": keys[0], "algorithm": ALGORITHM_LABELS[keys[1]], "model": MODEL_LABELS[keys[2]],
               "condition": keys[3], "budget": keys[4], "pairs": part["pair_index"].nunique()}
        random = part[part["group"] == "random"].groupby("pair_index")["success"].mean()
        row["random"] = _fmt(*_interval(random))
        for band, _low, _high in reversed(TAIL_BANDS):
            tail = part[(part["group"] == "tail") & (part["band"] == band)].groupby("pair_index")["success"].mean()
            row[f"tail {band}"] = _fmt(*_interval(tail))
            both = pd.concat([tail.rename("tail"), random.rename("random")], axis=1).dropna()
            row[f"tail {band} - random"] = _fmt(*_interval(both["tail"] - both["random"]), signed=True)
        rows.append(row)
    return pd.DataFrame(rows).set_index(["part", "algorithm", "model", "condition", "budget"])


def rule_points(data: pd.DataFrame) -> pd.DataFrame:
    """Part A: round 1's rules at the training budget, on the same score (one set per pair)."""

    rules = data[(data["part"] == "A") & (data["group"] == "rule")]
    rows = []
    for (algorithm, model, rule), part in rules.groupby(["algorithm", "model", "rule"], sort=False):
        mean, low, high = _interval(part["success"].astype(float))
        rows.append({"algorithm": algorithm, "model": model, "rule": rule, "pairs": len(part),
                     "score": float(part["score"].mean()), "score_low": float(part["score"].min()),
                     "score_high": float(part["score"].max()),
                     "median percentile": float(part["score_percentile"].median()),
                     "z degree": float(part["z_degree"].mean()), "success": mean, "low": low, "high": high})
    return pd.DataFrame(rows)


def binned_by_score(data: pd.DataFrame, n_bins: int = 8, n_boot: int = 2000, seed: int = 2026) -> pd.DataFrame:
    """Part A: random and tail sets in score bins (equal counts, pooled over pairs); 95% bootstrap over pairs."""

    sets = data[(data["part"] == "A") & data["group"].isin(["random", "tail"])]
    rows = []
    for (algorithm, model), part in sets.groupby(["algorithm", "model"], sort=False):
        part = part.assign(bin=pd.qcut(part["score"], n_bins, labels=False, duplicates="drop"))
        pairs = np.sort(part["pair_index"].unique())
        totals = part.pivot_table(index="pair_index", columns="bin", values="success", aggfunc="sum").reindex(pairs)
        counts = part.pivot_table(index="pair_index", columns="bin", values="success", aggfunc="size").reindex(pairs)
        totals, counts = totals.fillna(0).to_numpy(float), counts.fillna(0).to_numpy(float)
        draws = np.random.default_rng(seed).integers(0, len(pairs), size=(n_boot, len(pairs)))
        boot = totals[draws].sum(axis=1) / np.where(counts[draws].sum(axis=1) > 0, counts[draws].sum(axis=1), np.nan)
        for j, (bin_id, sub) in enumerate(part.groupby("bin")):
            low, high = np.nanpercentile(boot[:, j], [2.5, 97.5])
            rows.append({"algorithm": algorithm, "model": model, "budget": int(part["budget"].iloc[0]),
                         "bin": int(bin_id), "sets": len(sub),
                         "score": float(sub["score"].mean()), "score_min": float(sub["score"].min()),
                         "score_max": float(sub["score"].max()), "success": float(sub["success"].mean()),
                         "low": float(low), "high": float(high)})
    return pd.DataFrame(rows)


def stall_table(data: pd.DataFrame) -> pd.DataFrame:
    """GOAT: share of runs that stopped within STALL_ITERATIONS iterations, and their success."""

    goat = data[data["algorithm"] == "goat"].copy()
    if goat.empty:
        return pd.DataFrame()
    goat["stalled"] = pd.to_numeric(goat["n_iter"], errors="coerce") <= STALL_ITERATIONS
    rows = []
    for keys, part in goat.groupby(["part", "model", "condition", "budget"], sort=False):
        stalled = part[part["stalled"]]
        rows.append({"part": keys[0], "model": MODEL_LABELS[keys[1]], "condition": keys[2], "budget": keys[3],
                     "runs": len(part), "stalled share": round(float(part["stalled"].mean()), 3),
                     "stalled share, random sets": round(float(part.loc[part["group"] == "random", "stalled"].mean()), 3),
                     "stalled share, tail sets": round(float(part.loc[part["group"] == "tail", "stalled"].mean()), 3),
                     "success when stalled": round(float(stalled["success"].mean()), 2) if len(stalled) else np.nan,
                     "median iterations": float(pd.to_numeric(part["n_iter"], errors="coerce").median())})
    return pd.DataFrame(rows).set_index(["part", "model", "condition", "budget"])


N_EDGES = (20, 50, 100, 300, 1_000, 5_000, POOL_SIZE + 1)
N_LABELS = ("20-50", "50-100", "100-300", "300-1,000", "1,000-5,000", "5,000-20,000")


def best_of_n_table(data: pd.DataFrame, part: str = "A", n_boot: int = 2000, seed: int = 2026) -> pd.DataFrame:
    """Success of the sets that best of N would pick, for N from 20 to the pool size (20,000).

    A set ranked r-th of the pool by score is about what best of N = pool size / r picks (the best of N random
    sets sits near the top 1/N of the score distribution). Tail sets are grouped by that N; the random sets are
    N = 1. Success is pooled over pairs, with 95% intervals from resampling whole pairs.
    """

    sets = data[(data["part"] == part) & data["group"].isin(["random", "tail"])].copy()
    rank = POOL_SIZE * (1 - sets["score_percentile"] / 100) + 1          # 1 = the pool's best set
    sets["n"] = np.where(sets["group"] == "random", 1.0, POOL_SIZE / rank)
    labels = pd.cut(sets["n"], N_EDGES, labels=N_LABELS, right=False).astype(object)
    sets["n_bin"] = np.where(sets["group"] == "random", "1 (random)", labels)
    sets = sets.dropna(subset=["n_bin"])
    rows = []
    rng = np.random.default_rng(seed)
    for (algorithm, model, condition, budget), setting in sets.groupby(["algorithm", "model", "condition", "budget"],
                                                                      sort=False):
        pairs = np.sort(setting["pair_index"].unique())
        draws = rng.integers(0, len(pairs), size=(n_boot, len(pairs)))
        for label in ("1 (random)", *N_LABELS):
            sub = setting[setting["n_bin"] == label]
            if sub.empty:
                continue
            by_pair = sub.groupby("pair_index")["success"].agg(["sum", "size"]).reindex(pairs, fill_value=0)
            boot = by_pair["sum"].to_numpy()[draws].sum(axis=1) / np.maximum(by_pair["size"].to_numpy()[draws].sum(
                axis=1), 1)
            rows.append({"algorithm": algorithm, "model": model, "condition": condition, "budget": budget,
                         "N": label, "n_mid": float(np.exp(np.log(sub["n"]).mean())), "sets": len(sub),
                         "pairs": sub["pair_index"].nunique(), "success": float(sub["success"].mean()),
                         "low": float(np.percentile(boot, 2.5)), "high": float(np.percentile(boot, 97.5))})
    return pd.DataFrame(rows)


def reference_tables(algorithms=ALGORITHMS, models=MODELS, min_pairs: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The multi-pair data's held-out results (multi_pair_models.ipynb, section 3), for comparison:
    success by score percentile, and AUC of the held-out score and of degree alone on the random sets."""

    data = MP.load_sets(algorithms, models)
    curves, aucs = [], []
    for algorithm in algorithms:
        for model in models:
            part = data[(data["algorithm"] == algorithm) & (data["model"] == model)]
            if part["pair_index"].nunique() < min_pairs:
                continue
            scored = MP.held_out(data, algorithm, model)
            random_rate = float(scored.loc[scored["group"] == "random", "success"].mean())
            for position, (label, row) in enumerate(MP.success_by_percentile(scored).iterrows()):
                curves.append({"algorithm": algorithm, "model": model, "position": position, "bin": str(label),
                               "success": float(row["success"]), "sets": int(row["sets"]), "random": random_rate})
            score_auc, degree_auc = [], []
            for _, sub in scored[scored["group"] == "random"].groupby("pair_index"):
                y = sub["success"].to_numpy(bool)
                if 0 < y.mean() < 1:
                    score_auc.append(_auc(y, sub["score"]))
                    degree_auc.append(_auc(y, sub["z_degree"]))
            aucs.append({"algorithm": algorithm, "model": model, "pairs": len(score_auc),
                         **_auc_numbers(_interval(score_auc), _interval(degree_auc))})
    return pd.DataFrame(curves), pd.DataFrame(aucs)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

INK = {"primary": "#0b0b0b", "secondary": "#52514e", "grid": "#e6e6e6"}
REFERENCE_STYLE = dict(color="#9a9a9a", linewidth=1.6, linestyle="--", marker="o", markersize=3.5)
TEST_COLOR = "#1c5cab"
BUDGET_COLORS = ("#86b6ef", "#2a78d6", "#104281")          # fewer seeds, training budget, more seeds
DENSITY_COLORS = {0.75: "#86b6ef", 1.0: "#2a78d6", 1.5: "#104281"}   # sparser to denser: light to dark blue
SIZE_COLORS = {300: "#86b6ef", 600: "#2a78d6", 1200: "#104281"}        # smaller to larger: light to dark blue


def _grid(n_rows: int, height: float = 3.4):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, height * n_rows), squeeze=False)
    return fig, axes


def _percentile_axes(ax, title: str) -> None:
    labels = [label.replace("-", "–") for label in PERCENTILE_LABELS]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8.5)
    ax.set_xlim(-0.4, len(labels) - 0.6)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="y", color=INK["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_transfer(curves: pd.DataFrame, reference: pd.DataFrame | None = None):
    """Part A: success by score percentile on round 1's pairs, with the multi-pair held-out curve."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    part = curves[curves["part"] == "A"]
    algorithms = [a for a in ALGORITHMS if a in set(part["algorithm"])]
    fig, axes = _grid(len(algorithms))
    for row, algorithm in enumerate(algorithms):
        for column, model in enumerate(MODELS):
            ax = axes[row, column]
            c = part[(part["algorithm"] == algorithm) & (part["model"] == model)].sort_values("position")
            if c.empty:
                ax.set_visible(False)
                continue
            if reference is not None and not reference.empty:
                r = reference[(reference["algorithm"] == algorithm) & (reference["model"] == model)]
                if not r.empty:
                    r = r.sort_values("position")
                    ax.plot(r["position"], r["success"], **REFERENCE_STYLE)
                    ax.axhline(r["random"].iloc[0], color=REFERENCE_STYLE["color"], linewidth=1.0, linestyle=":")
            shown = c[c["sets"] > 0]
            ax.plot(shown["position"], shown["success"], color=TEST_COLOR, linewidth=2.0, marker="o", markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.6)
            ax.axhline(c["random"].iloc[0], color=TEST_COLOR, linewidth=1.0, linestyle=":")
            pairs = c["sets"].sum()
            _percentile_axes(ax, f"{ALGORITHM_LABELS[algorithm]} · {MODEL_LABELS[model]} · k = {c['budget'].iloc[0]}"
                                 f"\n{int(pairs):,} sets")
        axes[row, 0].set_ylabel("success rate")
    for ax in axes[-1]:
        ax.set_xlabel("score percentile among the pair's random seed sets")
    handles = [Line2D([0], [0], color=TEST_COLOR, linewidth=2.0, marker="o", markersize=4.5,
                      label="round 1's pairs (new to the score)"),
               Line2D([0], [0], **REFERENCE_STYLE, label="multi-pair pairs, held out (multi_pair_models.ipynb)"),
               Line2D([0], [0], color="#6b6b6b", linewidth=1.0, linestyle=":", label="average random set (same colour)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    return fig


RULE_MARKERS = {"top_degree": ("s", "#2b2b2b", 7), "top_harmonic": ("D", "#eb6834", 6),
                "clustered": ("^", "#eb6834", 7.5), "spread": ("v", "#eb6834", 7.5),
                "distinguishing": ("P", "#eb6834", 8), "hub_free_distinguishing": ("*", "#eb6834", 11)}


def plot_raw_score(binned: pd.DataFrame, rules: pd.DataFrame):
    """Part A: success against the raw score; round 1's rules placed on the same score (one marker shape each)."""

    from matplotlib.lines import Line2D

    algorithms = [a for a in ALGORITHMS if a in set(binned["algorithm"])]
    fig, axes = _grid(len(algorithms), height=3.5)
    for row, algorithm in enumerate(algorithms):
        for column, model in enumerate(MODELS):
            ax = axes[row, column]
            b = binned[(binned["algorithm"] == algorithm) & (binned["model"] == model)].sort_values("score")
            if b.empty:
                ax.set_visible(False)
                continue
            ax.axvspan(b["score_min"].min(), b["score_max"].max(), color="#f0f0ef", zorder=0, linewidth=0)
            ax.vlines(b["score"], b["low"], b["high"], color=TEST_COLOR, linewidth=1.6, zorder=3)
            ax.plot(b["score"], b["success"], color=TEST_COLOR, linewidth=1.4, marker="o", markersize=5,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=4)
            r = rules[(rules["algorithm"] == algorithm) & (rules["model"] == model)]
            for point in r.itertuples():
                marker, color, size = RULE_MARKERS.get(point.rule, ("P", "#eb6834", 7))
                ax.vlines(point.score, point.low, point.high, color=color, linewidth=0.9, alpha=0.55, zorder=5)
                ax.plot(point.score, point.success, marker=marker, markersize=size, color=color,
                        markeredgecolor="white", markeredgewidth=0.6, linestyle="", zorder=6)
            ax.set_ylim(-0.03, 1.05)
            ax.axvline(0, color="#c8c8c8", linewidth=0.8, zorder=1)
            ax.grid(axis="y", color=INK["grid"], linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{ALGORITHM_LABELS[algorithm]} · {MODEL_LABELS[model]} · k = {int(b['budget'].iloc[0])}",
                         fontsize=10.5)
        axes[row, 0].set_ylabel("success rate")
    axes[-1, 1].set_xlabel("weighted score (SDs of random sets; 0 = the average random set)")
    shown = [rule for rule in RULE_MARKERS if rule in set(rules["rule"])]
    handles = [Line2D([0], [0], color=TEST_COLOR, linewidth=1.4, marker="o", markersize=5,
                      label="random and tail sets, binned by score")]
    handles += [Line2D([0], [0], marker=RULE_MARKERS[rule][0], color=RULE_MARKERS[rule][1], linestyle="",
                       markersize=RULE_MARKERS[rule][2], label=RULE_LABELS.get(rule, rule)) for rule in shown]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def plot_best_of_n(table: pd.DataFrame, rules: pd.DataFrame | None = None):
    """Part A: success of best of N against N (log scale), with top degree and round 1's best rule at the same
    budget as dashed lines."""

    from matplotlib.lines import Line2D

    algorithms = [a for a in ALGORITHMS if a in set(table["algorithm"])]
    fig, axes = _grid(len(algorithms), height=3.3)
    for row, algorithm in enumerate(algorithms):
        for column, model in enumerate(MODELS):
            ax = axes[row, column]
            c = table[(table["algorithm"] == algorithm) & (table["model"] == model)].sort_values("n_mid")
            if c.empty:
                ax.set_visible(False)
                continue
            ax.axvspan(100, 1_000, color="#f0f0ef", zorder=0, linewidth=0)
            ax.vlines(c["n_mid"], c["low"], c["high"], color=TEST_COLOR, linewidth=1.5, zorder=3)
            ax.plot(c["n_mid"], c["success"], color=TEST_COLOR, linewidth=1.9, marker="o", markersize=5,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=4)
            if rules is not None and not rules.empty:
                r = rules[(rules["algorithm"] == algorithm) & (rules["model"] == model)]
                if not r.empty:
                    top = r[r["rule"] == "top_degree"]
                    best = r.sort_values(["success", "rule"], ascending=[False, True]).iloc[0]
                    lines = [(top["success"].iloc[0], "#2b2b2b", "top degree")] if not top.empty else []
                    if best["rule"] != "top_degree":
                        lines.append((best["success"], "#eb6834", RULE_LABELS.get(best["rule"], best["rule"])))
                    close = len(lines) == 2 and abs(lines[0][0] - lines[1][0]) < 0.05
                    for k, (value, color, label) in enumerate(lines):
                        below = close and k == 1          # two labels at the same height: put one under its line
                        ax.axhline(value, color=color, linewidth=1.2, linestyle="--", zorder=2)
                        ax.annotate(label, (1.0, value), xycoords=("axes fraction", "data"),
                                    xytext=(-3, -3 if below else 3), textcoords="offset points", ha="right",
                                    va="top" if below else "bottom", fontsize=7.5, color=color)
            ax.set_xscale("log")
            ax.set_xlim(0.7, POOL_SIZE * 1.6)
            ax.set_xticks([1, 10, 100, 1_000, 10_000])
            ax.set_xticklabels(["1\n(random)", "10", "100", "1,000", "10,000"], fontsize=8.5)
            ax.set_ylim(-0.03, 1.05)
            ax.grid(axis="y", color=INK["grid"], linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{ALGORITHM_LABELS[algorithm]} · {MODEL_LABELS[model]} · k = {int(c['budget'].iloc[0])}",
                         fontsize=10.5)
        axes[row, 0].set_ylabel("success rate")
    axes[-1, 1].set_xlabel("N: best of N random seed sets by the score (log scale)")
    handles = [Line2D([0], [0], color=TEST_COLOR, linewidth=1.9, marker="o", markersize=5,
                      label="best of N (95% interval over pairs)"),
               Line2D([0], [0], color="#2b2b2b", linewidth=1.2, linestyle="--", label="top degree"),
               Line2D([0], [0], color="#eb6834", linewidth=1.2, linestyle="--",
                      label="round 1's best rule at this budget")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def plot_conditions(curves: pd.DataFrame, which: str = "B"):
    """Part B: success by score percentile at the other budgets, with the training-budget curve (part A) for
    comparison. Dotted lines: each budget's average random set. (Part C has plot_grid.)"""

    from matplotlib.lines import Line2D

    base = curves[curves["part"] == "A"]
    other = curves[curves["part"] == which]
    algorithms = [a for a in ALGORITHMS if a in set(other["algorithm"])]
    fig, axes = _grid(len(algorithms))
    for row, algorithm in enumerate(algorithms):
        for column, model in enumerate(MODELS):
            ax = axes[row, column]
            mine = other[(other["algorithm"] == algorithm) & (other["model"] == model)]
            if mine.empty:
                ax.set_visible(False)
                continue
            lines = []
            training = base[(base["algorithm"] == algorithm) & (base["model"] == model)]
            order = sorted(set(mine["budget"]) | set(training["budget"]))
            frames = pd.concat([training, mine])
            k0 = int(training["budget"].iloc[0]) if not training.empty else None
            if len(order) == len(BUDGET_COLORS):
                colors = BUDGET_COLORS
            else:  # more budgets than the three-step ramp: sample the same blue ramp
                import matplotlib
                colors = [matplotlib.colormaps["Blues"](x) for x in np.linspace(0.4, 0.95, len(order))]
            for budget, color in zip(order, colors):
                c = frames[frames["budget"] == budget]
                lines.append((c, color, f"k = {budget}" + (" (training)" if budget == k0 else "")))
            for c, color, label in lines:
                c = c.sort_values("position")
                shown = c[c["sets"] > 0]
                ax.plot(shown["position"], shown["success"], color=color, linewidth=1.9, marker="o", markersize=4,
                        markeredgecolor="white", markeredgewidth=0.6, label=label)
                ax.axhline(c["random"].iloc[0], color=color, linewidth=1.0, linestyle=":")
            _percentile_axes(ax, f"{ALGORITHM_LABELS[algorithm]} · {MODEL_LABELS[model]}")
            ax.legend(loc="best", fontsize=7.5, frameon=True, framealpha=0.85, edgecolor="none",
                      handlelength=1.6)
        axes[row, 0].set_ylabel("success rate")
    for ax in axes[-1]:
        ax.set_xlabel("score percentile among the pair's random seed sets")
    handles = [Line2D([0], [0], color="#6b6b6b", linewidth=1.0, linestyle=":",
                      label="average random set (dotted, same colour as its curve)")]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def _with_cells(frame: pd.DataFrame) -> pd.DataFrame:
    """Size and mean-degree factor of every row: part A is round 1's setting (600 vertices, x 1)."""

    frame = frame[frame["part"].isin(["A", "C"])].copy()
    cells = [parse_cell(c) if p == "C" else (600, 1.0) for p, c in zip(frame["part"], frame["condition"])]
    frame["n"] = [n for n, _ in cells]
    frame["factor"] = [f for _, f in cells]
    return frame


def plot_grid(curves: pd.DataFrame, algorithm: str):
    """Part C: success by score percentile for one algorithm. Rows: graph models; columns: graph sizes; one line per
    mean degree (light: sparser, dark: denser). Round 1's setting (600 vertices, x 1) is part A."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data = _with_cells(curves[curves["algorithm"] == algorithm])
    fig, axes = plt.subplots(len(MODELS), len(GRID_SIZES), figsize=(13, 3.3 * len(MODELS)), squeeze=False)
    for row, model in enumerate(MODELS):
        for column, n in enumerate(GRID_SIZES):
            ax = axes[row, column]
            mine = data[(data["model"] == model) & (data["n"] == n)]
            _percentile_axes(ax, f"{MODEL_LABELS[model]} · {n:,} vertices")
            if mine.empty:
                ax.text(0.5, 0.5, "not in the grid" if algorithm != "percolation" and n > 600 else "no results yet",
                        transform=ax.transAxes, ha="center", va="center", fontsize=9, color=INK["secondary"])
                continue
            for factor in GRID_DENSITIES:
                c = mine[mine["factor"] == factor].sort_values("position")
                if c.empty:
                    continue
                shown = c[c["sets"] > 0]
                color = DENSITY_COLORS[factor]
                label = (f"degree x{factor:g}, k = {int(c['budget'].iloc[0])}"
                         + (" (round 1)" if c["part"].iloc[0] == "A" else ""))
                ax.plot(shown["position"], shown["success"], color=color, linewidth=1.9, marker="o", markersize=4,
                        markeredgecolor="white", markeredgewidth=0.6, label=label)
                ax.axhline(c["random"].iloc[0], color=color, linewidth=1.0, linestyle=":")
            ax.legend(loc="best", fontsize=7.5, frameon=True, framealpha=0.85, edgecolor="none", handlelength=1.6)
        axes[row, 0].set_ylabel("success rate")
    for ax in axes[-1]:
        ax.set_xlabel("score percentile among the pair's random seed sets")
    fig.suptitle(f"{ALGORITHM_LABELS[algorithm]}: success by score percentile across graph sizes and mean degrees",
                 fontsize=12.5)
    handles = [Line2D([0], [0], color=DENSITY_COLORS[f], linewidth=1.9, marker="o", markersize=4,
                      label=f"mean degree x{f:g}") for f in GRID_DENSITIES]
    handles.append(Line2D([0], [0], color="#6b6b6b", linewidth=1.0, linestyle=":",
                          label="average random set (dotted, same colour)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    return fig


def plot_grid_auc(ranking: pd.DataFrame):
    """Part C (with part A at 600 vertices, x 1): the score's AUC on random sets across the grid (solid), and degree
    alone (dashed), against mean degree, one colour per graph size."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data = _with_cells(ranking)
    algorithms = [a for a in ALGORITHMS if a in set(data["algorithm"])]
    fig, axes = plt.subplots(len(algorithms), len(MODELS), figsize=(13, 3.2 * len(algorithms)), squeeze=False,
                             sharey=True)
    positions = {f: i for i, f in enumerate(GRID_DENSITIES)}
    for row, algorithm in enumerate(algorithms):
        for column, model in enumerate(MODELS):
            ax = axes[row, column]
            mine = data[(data["algorithm"] == algorithm) & (data["model"] == model)]
            for n in GRID_SIZES:
                line = mine[mine["n"] == n].sort_values("factor")
                if line.empty:
                    continue
                x = line["factor"].map(positions)
                ax.plot(x, line["auc_score"], color=SIZE_COLORS[n], linewidth=1.9, marker="o", markersize=5,
                        markeredgecolor="white", markeredgewidth=0.6)
                ax.vlines(x, line["auc_score_low"], line["auc_score_high"], color=SIZE_COLORS[n], linewidth=1.0)
                ax.plot(x, line["auc_degree"], color=SIZE_COLORS[n], linewidth=1.2, linestyle="--", marker="o",
                        markersize=4, markerfacecolor="white")
            ax.axhline(0.5, color="#9a9a9a", linewidth=0.9, linestyle=":")
            ax.set_xticks(range(len(GRID_DENSITIES)))
            ax.set_xticklabels([f"x{f:g}" for f in GRID_DENSITIES])
            ax.set_xlim(-0.4, len(GRID_DENSITIES) - 0.6)
            ax.grid(axis="y", color=INK["grid"], linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{ALGORITHM_LABELS[algorithm]} · {MODEL_LABELS[model]}", fontsize=10.5)
        axes[row, 0].set_ylabel("AUC within pairs")
    for ax in axes[-1]:
        ax.set_xlabel("mean degree relative to round 1's")
    handles = [Line2D([0], [0], color=SIZE_COLORS[n], linewidth=1.9, marker="o", markersize=5, label=f"{n:,} vertices")
               for n in GRID_SIZES]
    handles += [Line2D([0], [0], color="#6b6b6b", linewidth=1.9, label="weighted score (95% interval)"),
                Line2D([0], [0], color="#6b6b6b", linewidth=1.2, linestyle="--", marker="o", markerfacecolor="white",
                       label="degree alone"),
                Line2D([0], [0], color="#9a9a9a", linewidth=0.9, linestyle=":", label="chance (0.5)")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    return fig


def plot_auc(ranking: pd.DataFrame, reference: pd.DataFrame | None = None, parts=("A", "B")):
    """AUC of the score and of degree alone on random sets, per setting and condition (dumbbells)."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ranking = ranking[ranking["part"].isin(parts)]
    algorithms = [a for a in ALGORITHMS if a in set(ranking["algorithm"])]
    n_rows = max(len(ranking[ranking["algorithm"] == a]) for a in algorithms) + 3
    fig, axes = plt.subplots(1, len(algorithms), figsize=(4.7 * len(algorithms), 1.2 + 0.36 * n_rows),
                             sharex=True, squeeze=False)
    axes = axes[0]
    for ax, algorithm in zip(axes, algorithms):
        mine = ranking[ranking["algorithm"] == algorithm].copy()
        mine["_m"] = mine["model"].map({m: i for i, m in enumerate(MODELS)})
        mine["_p"] = (mine["part"] == "C").astype(int)
        mine["_c"] = [cell_code(c) if p == "C" else b for p, c, b in zip(mine["part"], mine["condition"],
                                                                          mine["budget"])]
        mine = mine.sort_values(["_m", "_p", "_c"])
        rows = []
        for model, sub in mine.groupby("model", sort=False):
            if reference is not None and not reference.empty:
                for r in reference[(reference["algorithm"] == algorithm) &
                                   (reference["model"] == model)].to_dict("records"):
                    rows.append({**r, "label": f"{MODEL_LABELS[model]} · multi-pair data, held out",
                                 "reference": True})
            for r in sub.to_dict("records"):
                label = {"A": f"k = {r['budget']} (training budget)", "B": f"k = {r['budget']}",
                         "C": f"{cell_label(r['condition']) if r['part'] == 'C' else ''}, k = {r['budget']}"}[r["part"]]
                rows.append({**r, "label": f"{MODEL_LABELS[model]} · {label}", "reference": False})
        for y, r in enumerate(rows):
            if not (np.isfinite(r["auc_score"]) and np.isfinite(r["auc_degree"])):
                continue
            score_color = "#9a9a9a" if r["reference"] else "#2a78d6"
            ax.plot([r["auc_degree"], r["auc_score"]], [y, y], color="#d6d6d6", linewidth=1.4, zorder=1)
            ax.hlines(y - 0.2, r["auc_score_low"], r["auc_score_high"], color=score_color, linewidth=1.0, zorder=2)
            ax.hlines(y + 0.2, r["auc_degree_low"], r["auc_degree_high"], color="#6b6b6b", linewidth=1.0, zorder=2)
            ax.plot(r["auc_degree"], y, marker="o", markersize=6, markerfacecolor="white", markeredgecolor="#6b6b6b",
                    markeredgewidth=1.4, linestyle="", zorder=3)
            ax.plot(r["auc_score"], y, marker="o", markersize=6.5, color=score_color, markeredgecolor="white",
                    markeredgewidth=0.6, linestyle="", zorder=4)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["label"] for r in rows], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0.5, color="#9a9a9a", linewidth=0.9, linestyle="--", zorder=0)
        ax.grid(axis="x", color=INK["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(ALGORITHM_LABELS[algorithm], fontsize=11)
        ax.set_xlabel("AUC within pairs (random sets)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [Line2D([0], [0], marker="o", color="#2a78d6", linestyle="", markersize=6.5, label="weighted score"),
               Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="#6b6b6b", markeredgewidth=1.4,
                      linestyle="", markersize=6, label="degree alone"),
               Line2D([0], [0], marker="o", color="#9a9a9a", linestyle="", markersize=6.5,
                      label="weighted score on the multi-pair data, held out"),
               Line2D([0], [0], color="#9a9a9a", linestyle="--", linewidth=0.9, label="chance (0.5)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS)
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=ALGORITHMS)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--jobs", type=int, default=3, help="tasks run in parallel (default 3)")
    parser.add_argument("--blas-threads", type=int, default=3)
    parser.add_argument("--status", action="store_true", help="only show how far it is")
    args = parser.parse_args()
    if args.status:
        print(status(args.parts, args.algorithms, args.models).to_string())
    else:
        run_all(args.parts, args.algorithms, args.models, jobs=args.jobs, blas_threads=args.blas_threads)
