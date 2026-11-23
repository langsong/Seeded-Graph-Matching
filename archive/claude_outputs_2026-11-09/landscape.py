"""Match-ratio landscape: how single-run match ratios spread, and how they track seed-set features, across graph
models, cells and seed-set designs (7 Nov 2026).

Why. The regime map (regime_map.py) found where single runs end all-or-nothing and where the match ratio is spread
out. The group asked to see match ratio against seed-set features where outcomes are graded, including settings where
seed choice does not pay. This collects that data in fixed cells, on new pairs, for ER, Chung-Lu and PAPER.

Cells. Per algorithm, three (mean degree, rho) cells from the ER map: one where round 1's best rule gained a lot over
random sets, one moderate, one where rules barely helped (CELLS). The same cells run in every graph model (mean degree
10 instead of 8 for Chung-Lu, as in its map). A cell runs at the budget where random sets average LEVEL (default 50%),
read from that model's regime-map stage 1; when the curve never crosses 50%, at the budget closest to 50%. Budgets
are frozen in <model>/budgets.json the first time they are read.

Seed sets per pair (PAIRS new pairs per cell; algorithms in the same cell share pairs). No set uses a test vertex.
    random     N_RANDOM uniform random sets: the first sets of a POOL-set random pool
    tilted     TILT_SETS sets per gamma in GAMMAS: vertices drawn (without replacement) with probability
               proportional to degree ** gamma
    quantile   for degree, harmonic, pair mass, uniqueness and the frozen score, one set at each of QUANTILES
               (lowest to highest) of the pool's remaining sets
    rule       round 1's rules for the algorithm
    screen     best of 100 and of 1,000 by the frozen score, best of 1,000 by degree (pool sets after the random ones)
Every set gets the four core features (with z-scores against the pool), the frozen score and its percentile, the 11
fixed-pair features, and B-side "oracle" features that no selection rule can see: how many of the set's edges and
seed-to-non-seed clues survive in B.

Runs. FAQ runs every set RUNS["faq"] times with different solver seeds (solver luck dominates some graded FAQ cells);
GOAT and percolation once (same seeds, same result). Runs go repeat by repeat, so a stopped pair already has one run
of every set. On the first pair of each cell, two extra tasks:
    redraws      REDRAW_SETS random sets (stratified by frozen score) on REDRAWS fresh draws of B given A, from the
                 pair's own model (as in b_replicates.py): the seed-set share of the outcome and its noise ceiling
    regularizer  GOAT only: the same sets with the regulariser scaled by each of REG_FACTORS

Outcome. Accuracy on N_TEST fixed test vertices per pair (never seeds), as in the regime map. Non-seed accuracy,
success, first-step accuracy, objective, iterations and run time are recorded too.

Run from the repository root (ds environment):

    python -m selection_rule_comparison.landscape --model sparse_er
    python -m selection_rule_comparison.landscape --model chung_lu --status
    python -m selection_rule_comparison.landscape --model paper_pa --check-pairs     # can each cell draw pairs?
    python -m selection_rule_comparison.landscape --model sparse_er --levels 0.2 0.8 --pairs 1 --no-extras
    python -m selection_rule_comparison.landscape --model sparse_er --cells faq:16:0.2 goat:16:0.2
    python -m selection_rule_comparison.landscape --model sparse_er --summary        # a first look at the results

Results: data/landscape/<model>/<cell>/<algorithm>/k<budget>/pair_XXX/ with sets.csv (one row per set), runs.csv,
redraws.csv, regularizer.csv and pair.json, or under LANDSCAPE_OUTPUT_DIR. Rows are written run by run, so a stopped
run resumes where it stopped (Ctrl-C stops the workers too). `read(model)` joins sets and runs for analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache"))

import numpy as np
import pandas as pd

from experiment_common.features import (
    build_graph_context,
    fixed_pair_seed_set_features,
    signature_robustness_features,
)
from experiment_common.provenance import collect_environment, thread_limit, write_json
from fixed_pair_seed_set_classification import multi_pair_collection as MP

from . import b_replicates as BR
from . import graded_outcomes as G
from . import regime_map as RM
from . import run_comparison as rc
from . import settings as S
from .strategies import STRATEGY_CODES, GraphA

EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("LANDSCAPE_OUTPUT_DIR") or EXPERIMENT_DIR / "data" / "landscape")

ALGORITHMS = RM.ALGORITHMS
MODELS = RM.MODELS
CELLS = {   # (mean degree, rho) in ER terms; each algorithm: large, moderate and small rule gain in the ER map
    "percolation": ((8, 0.7), (16, 0.4), (32, 0.3)),
    "faq": ((32, 0.3), (8, 0.5), (16, 0.3)),
    "goat": ((8, 0.5), (8, 0.4), (16, 0.3)),
}
LEVEL = 0.5
PAIRS = 3
PAIR_INDEX_OFFSET = 100   # landscape pair i is regime-map pair index 100 + i, never one of the map's own pairs
N_TEST = RM.N_TEST
POOL = 20_000
N_RANDOM = 350
SCREEN_POOL = 1_000       # the pool sets after the random ones, screened by best of 100 / 1,000
GAMMAS = (-1.0, -0.5, 0.5, 1.0, 2.0, 3.0)
TILT_SETS = 10
QUANTILE_FEATURES = ("degree", "harmonic", "pair_mass", "uniqueness", "score")
QUANTILES = (0.0, 0.002, 0.01, 0.03, 0.1, 0.3, 0.7, 0.9, 0.97, 0.99, 0.998, 1.0)
SCREENS = ("best_100", "best_1000", "degree_1000")
RUNS = {"percolation": 1, "faq": 3, "goat": 1}
REDRAW_SETS, REDRAWS = 100, 10
REG_FACTORS = (0.5, 2.0)
KINDS = ("sets", "redraws", "regularizer")

SMOKE = {   # --smoke: a tiny design that exercises every code path in minutes (use its own output folder)
    "pairs": 1, "pool": 2_000, "n_random": 6, "screen_pool": 200, "tilt_sets": 1, "quantiles": [0.0, 1.0],
    "runs": {"percolation": 1, "faq": 2, "goat": 1}, "redraw_sets": 4, "redraws": 2, "cells_per_algorithm": 1,
}

SAMPLING_SEED_BASE, SOLVER_SEED_BASE = 20261180, 20261181
PURPOSE = {"pool": 1, "tilt": 2, "rules": 3, "redraw_sets": 4}
KIND_CODES = {"sets": 1, "redraws": 2, "regularizer": 3}
CORE = RM.CORE
LABELS, MODEL_LABELS = RM.LABELS, RM.MODEL_LABELS
ORACLE_COLUMNS = ("oracle_cut_edges_kept", "oracle_cut_kept_share", "oracle_b_degree", "oracle_seed_edges_kept_share",
                  "oracle_nonseeds_with_common_clue", "oracle_common_clues_mean")
OUTCOME_COLUMNS = ("accuracy_test", "accuracy", "success", "h0_accuracy", "objective", "n_iter", "converged",
                   "runtime_seconds", "matched_fraction", "error")


def _now() -> str:
    return RM._now()


def _clock() -> str:
    return RM._clock()


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, content: dict) -> None:
    """Atomic like provenance.write_json, with a temporary file of its own per process (tasks of one pair may write
    at the same time)."""

    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _model_degree(model: str, degree: float) -> float:
    """Chung-Lu uses mean degree 10 where the ER cells use 8 (its map has no 8)."""
    return 10 if model == "chung_lu" and degree == 8 else degree


def cells_for(model: str, algorithms=ALGORITHMS, extra=None, smoke: bool = False) -> list[tuple[str, float, float]]:
    """(algorithm, mean degree, rho) cells for this model, the standard ones first, then any extra ones."""
    out = []
    for a in algorithms:
        cells = CELLS[a][:SMOKE["cells_per_algorithm"]] if smoke else CELLS[a]
        out += [(a, _model_degree(model, d), float(rho)) for d, rho in cells]
    for a, d, rho in extra or []:
        if a in algorithms and (a, d, rho) not in out:
            out.append((a, d, float(rho)))
    return out


# --------------------------------------------------------------------------
# Design and budgets
# --------------------------------------------------------------------------

def _fixed(model: str, smoke: bool) -> dict:
    design = {
        "version": 1, "model": model, "smoke": smoke, "n_vertices": RM.N_VERTICES, "n_test": N_TEST,
        "pool": POOL, "n_random": N_RANDOM, "screen_pool": SCREEN_POOL, "gammas": list(GAMMAS),
        "tilt_sets": TILT_SETS, "quantile_features": list(QUANTILE_FEATURES), "quantiles": list(QUANTILES),
        "screens": list(SCREENS), "runs": RUNS, "redraw_sets": REDRAW_SETS, "redraws": REDRAWS,
        "reg_factors": list(REG_FACTORS), "rules": {a: list(RM.RULES[a]) for a in ALGORITHMS},
        "pair_index_offset": PAIR_INDEX_OFFSET, "weights": RM.load_weights(model),
        "seed_bases": {"sampling": SAMPLING_SEED_BASE, "solver": SOLVER_SEED_BASE, "draws": BR.DRAW_SEED_BASE},
        "solver": {"sgm": S.SGM, "percolation": S.PERCOLATION, "goat_regularizer": S.GOAT_REGULARIZER[model]},
        "uniqueness_cap": {a: MP.UNIQUENESS_CAP[a] for a in ALGORITHMS},
    }
    if smoke:
        design.update({k: v for k, v in SMOKE.items() if k in design})
    return json.loads(json.dumps(design))


def _map_pair_seeds(model: str) -> set[int]:
    seeds = set()
    for path in (RM.OUTPUT_DIR / model).glob("d*_rho*/*/stage*/pair_*/pair.json"):
        try:
            seeds.add(int(_read_json(path)["pair_seed"]))
        except (KeyError, ValueError, OSError):
            pass
    return seeds


def prepare(model: str, blas_threads: int = 3, smoke: bool = False) -> dict:
    """The design every job of this model uses, frozen in <model>/settings.json once results exist."""

    path = OUTPUT_DIR / model / "settings.json"
    fixed = _fixed(model, smoke)
    if path.exists():
        saved = _read_json(path)
        differs = [key for key in fixed if saved.get(key) != fixed[key]]
        if differs:
            raise ValueError(f"The saved design in {path} differs ({', '.join(differs)}). Set LANDSCAPE_OUTPUT_DIR "
                             "to a new folder to keep the results apart.")
        excluded = saved["excluded_pair_seeds"]
    else:
        excluded = sorted(G.known_pair_seeds() | G._graded_pair_seeds() | _map_pair_seeds(model))
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {**fixed, "excluded_pair_seeds": excluded, "written_at": _now(),
                          "environment": collect_environment(blas_threads)})
    return {**fixed, "excluded_pair_seeds": excluded, "blas_threads": blas_threads}


def _budget_key(algorithm: str, degree: float, rho: float, level: float) -> str:
    return f"{algorithm}/{RM.cell_name(degree, rho)}/level{level:.2f}"


def cell_budget(model: str, algorithm: str, degree: float, rho: float, level: float = LEVEL) -> dict:
    """{"budget": k or None, "how": ..., "curve": ...} from the regime map's stage 1, frozen on first use."""

    path = OUTPUT_DIR / model / "budgets.json"
    saved = _read_json(path) if path.exists() else {}
    key = _budget_key(algorithm, degree, rho, level)
    if key in saved:
        return saved[key]
    pairs = range(RM.STAGE1_PAIRS)
    if not all(RM._stage1_complete(model, degree, rho, algorithm, i) for i in pairs):
        folder = RM.OUTPUT_DIR / model / RM.cell_name(degree, rho) / algorithm / "stage1"
        return {"budget": None, "how": f"regime map stage 1 not finished in {folder}", "curve": {}}
    runs = pd.concat([pd.read_csv(RM.stage_dir(model, degree, rho, algorithm, 1, i) / "runs.csv") for i in pairs],
                     ignore_index=True)
    curve = RM.cell_curve(runs)
    k = RM.crossing(curve.index, curve.to_numpy(), level)
    candidates = RM.N_VERTICES - N_TEST
    if k is not None:
        budget, how = int(min(max(round(k), 3), candidates)), f"random sets reach {level:.0%} at about {k:.1f} seeds"
    elif level == 0.5:
        budget = RM.middle_budget({int(b): float(m) for b, m in curve.items()})
        how = (f"no 50% crossing between {int(curve.index.min())} and {int(curve.index.max())} seeds; budget with "
               f"mean closest to 50% ({curve[budget]:.2f})") if budget is not None else "no finite curve"
    else:
        budget, how = None, f"random sets never cross {level:.0%} in the map's range"
    entry = {"budget": None if budget is None else int(budget), "how": how,
             "curve": {str(int(b)): float(m) for b, m in curve.items()}, "map_pairs": len(pairs), "written_at": _now()}
    saved[key] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, saved)
    return entry


# --------------------------------------------------------------------------
# Seed sets and their features
# --------------------------------------------------------------------------

def _rng(pair_seed: int, purpose: str, k: int, *values: int) -> np.random.Generator:
    return np.random.default_rng([SAMPLING_SEED_BASE, int(pair_seed), PURPOSE[purpose], int(k), *map(int, values)])


def _solver_seed(pair_seed: int, k: int, algorithm: str, kind: str, set_id: int, repeat: int) -> int:
    return rc._seed(SOLVER_SEED_BASE, pair_seed, k, MP.ALGORITHM_CODES[algorithm], KIND_CODES[kind], set_id, repeat)


def plan_sets(algorithm: str, pair_seed: int, A: np.ndarray, graph: GraphA, test: np.ndarray,
              candidates: np.ndarray, k: int, design: dict):
    """Every seed set of a pair, with its design label, core features, z-scores and frozen score.

    Returns (sets table, set arrays, pool stats). Deterministic: every task of the pair rebuilds the same plan.
    """

    weights = design["weights"][algorithm]
    pool = G.draw_sets(_rng(pair_seed, "pool", k, MP.ALGORITHM_CODES[algorithm]), candidates, k, design["pool"])
    features = G._core_features(graph, algorithm, pool)
    stats = {f: {"mean": float(features[f].mean()), "sd": float(features[f].std(ddof=1))} for f in CORE}

    def z(f, values):
        return (values - stats[f]["mean"]) / stats[f]["sd"] if stats[f]["sd"] > 0 else np.zeros(len(values))

    score = sum(weights[f] * z(f, features[f]) for f in CORE)
    sorted_score = np.sort(score)
    rows, arrays = [], []

    def add(kind, detail, seeds, raw, pool_index=-1):
        values = {f: float(raw[f]) for f in CORE}
        zs = {f: float(z(f, np.array([values[f]]))[0]) for f in CORE}
        value = float(sum(weights[f] * zs[f] for f in CORE))
        rows.append({"set_id": len(rows), "design": kind, "detail": detail, "pool_index": int(pool_index),
                     **values, **{f"z_{f}": zs[f] for f in CORE}, "score": value,
                     "score_percentile": float(100 * np.searchsorted(sorted_score, value + 1e-12) / len(score))})
        arrays.append(np.sort(np.asarray(seeds, dtype=np.int64)))

    n_random, n_screen = design["n_random"], design["screen_pool"]
    for j in range(n_random):
        add("random", "", pool[j], {f: features[f][j] for f in CORE}, j)

    # degree-tilted sets: vertices drawn with probability proportional to degree ** gamma
    weights_by_degree = graph.degree[candidates].astype(float)
    tilted = []
    for g, gamma in enumerate(design["gammas"]):
        p = weights_by_degree ** gamma
        p = p / p.sum()
        rng = _rng(pair_seed, "tilt", k, MP.ALGORITHM_CODES[algorithm], g)
        for _ in range(design["tilt_sets"]):
            tilted.append((f"gamma={gamma:g}", np.sort(rng.choice(candidates, size=k, replace=False, p=p))))
    if tilted:
        tilt_features = G._core_features(graph, algorithm, np.stack([s for _, s in tilted]))
        for t, (detail, seeds) in enumerate(tilted):
            add("tilted", detail, seeds, {f: tilt_features[f][t] for f in CORE})

    # screens from the pool sets after the random ones
    screen = np.arange(n_random, n_random + n_screen)
    used = np.zeros(len(pool), dtype=bool)
    used[:n_random] = True
    picks = {"best_100": screen[int(np.argmax(score[screen[:100]]))],
             "best_1000": screen[int(np.argmax(score[screen]))],
             "degree_1000": screen[int(np.argmax(features["degree"][screen]))]}
    for name in design["screens"]:
        index = int(picks[name])
        used[index] = True
        add("screen", name, pool[index], {f: features[f][index] for f in CORE}, index)

    # quantile picks: lowest to highest of each feature among the rest of the pool
    rest = np.arange(n_random + n_screen, len(pool))
    for feature in design["quantile_features"]:
        values = score if feature == "score" else features[feature]
        order = rest[np.argsort(values[rest], kind="stable")]
        for q in design["quantiles"]:
            position = int(round(q * (len(order) - 1)))
            step = -1 if q > 0.5 else 1
            while used[order[position]] and 0 <= position + step < len(order):
                position += step
            index = int(order[position])
            used[index] = True
            add("quantile", f"{feature}@{q:g}", pool[index], {f: features[f][index] for f in CORE}, index)

    # round 1's rules, never seeding a test vertex
    rules = design["rules"][algorithm]
    rule_sets = np.stack([np.sort(RM._ordering(rule, graph, k, _rng(pair_seed, "rules", k, STRATEGY_CODES[rule]),
                                               test)[:k]) for rule in rules])
    rule_features = G._core_features(graph, algorithm, rule_sets)
    for r, rule in enumerate(rules):
        add("rule", rule, rule_sets[r], {f: rule_features[f][r] for f in CORE})

    table = pd.DataFrame(rows)
    return table, arrays, stats


def oracle_features(A: np.ndarray, aligned_b: np.ndarray, seeds: np.ndarray) -> dict:
    """What survives in B (aligned to A's labels): clues no graph-A rule can see."""

    n, k = len(A), len(seeds)
    is_seed = np.zeros(n, dtype=bool)
    is_seed[seeds] = True
    a_rows = A[seeds] > 0
    common = a_rows & (aligned_b[seeds] > 0)
    cut_a = int(a_rows[:, ~is_seed].sum())
    kept = int(common[:, ~is_seed].sum())
    inside = a_rows[:, is_seed]
    inside_kept = common[:, is_seed]
    clues = common[:, ~is_seed].sum(axis=0)
    return {"oracle_cut_edges_kept": kept,
            "oracle_cut_kept_share": kept / cut_a if cut_a else np.nan,
            "oracle_b_degree": float((aligned_b[seeds][:, ~is_seed] > 0).sum() / k),
            "oracle_seed_edges_kept_share": float(inside_kept.sum() / inside.sum()) if inside.sum() else np.nan,
            "oracle_nonseeds_with_common_clue": float(np.mean(clues >= 1)),
            "oracle_common_clues_mean": float(clues.mean())}


def _with_all_features(table: pd.DataFrame, arrays: list, A: np.ndarray, aligned_b: np.ndarray) -> pd.DataFrame:
    context = build_graph_context(A, compute_seed_features=False)
    extra = []
    for seeds in arrays:
        extra.append({**fixed_pair_seed_set_features(context, seeds), **signature_robustness_features(context, seeds),
                      **oracle_features(A, aligned_b, seeds)})
    out = pd.concat([table.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    out.insert(4, "seed_vertices", [json.dumps([int(v) for v in seeds]) for seeds in arrays])
    return out


def redraw_sets(table: pd.DataFrame, pair_seed: int, k: int, n_sets: int) -> np.ndarray:
    """Set ids of random sets spread over the tenths of the frozen score (n_sets // 10 per tenth)."""

    random = table[table["design"] == "random"].sort_values("score", kind="stable")
    groups = np.array_split(random["set_id"].to_numpy(), 10)
    rng = _rng(pair_seed, "redraw_sets", k)
    per = max(1, n_sets // 10)
    chosen = [rng.choice(group, size=min(per, len(group)), replace=False) for group in groups if len(group)]
    return np.sort(np.concatenate(chosen))[:n_sets]


# --------------------------------------------------------------------------
# B given A (fresh draws of B from the pair's own model)
# --------------------------------------------------------------------------

def b_probability(model: str, degree: float, rho: float, pair_seed: int, A: np.ndarray, B: np.ndarray,
                  truth: np.ndarray) -> np.ndarray:
    """Probability that each vertex pair has an edge in a new B, given A (same model as the pair)."""

    config = RM.graph_config(model, degree, rho)
    tree = None
    if model == "sparse_er":
        P = np.full(A.shape, config.er_probability)
    elif model == "chung_lu":
        A2, _, _, P = BR._chung_lu_pair(config, pair_seed)
        if rc._hash(A2) != rc._hash(A):
            raise RuntimeError(f"Chung-Lu pair {pair_seed} did not regenerate identically.")
    elif model == "paper_pa":
        P = np.full(A.shape, config.paper_pa_probability)
        tree = BR.paper_tree(config, pair_seed)
        aligned = B[np.ix_(truth, truth)]
        if np.any((tree > 0) & ((A == 0) | (aligned == 0))):
            raise RuntimeError(f"PAPER pair {pair_seed}: the rebuilt tree is not in both graphs.")
    else:
        raise ValueError(f"No B model for {model}")
    probability = np.where(A > 0, P + rho * (1 - P), P * (1 - rho))
    if tree is not None:
        probability[tree > 0] = 1.0
    np.fill_diagonal(probability, 0.0)
    return probability


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------

def pair_dir(model: str, algorithm: str, degree: float, rho: float, budget: int, pair_index: int) -> Path:
    return OUTPUT_DIR / model / RM.cell_name(degree, rho) / algorithm / f"k{budget:03d}" / f"pair_{pair_index:03d}"


def describe(task) -> str:
    model, algorithm, degree, rho, level, budget, kind, i = task
    return (f"{kind} {LABELS[algorithm]} {MODEL_LABELS[model]}, mean degree {degree:g}, rho {rho:.2f}, "
            f"{budget} seeds ({level:.0%} level), pair {i}")


def _planned(task, design: dict) -> int:
    model, algorithm, degree, rho, level, budget, kind, i = task
    if kind == "sets":
        n_sets = (design["n_random"] + len(design["gammas"]) * design["tilt_sets"] + len(design["screens"])
                  + len(design["quantile_features"]) * len(design["quantiles"]) + len(design["rules"][algorithm]))
        return n_sets * design["runs"][algorithm]
    if kind == "redraws":
        return design["redraw_sets"] * design["redraws"]
    return design["redraw_sets"] * len(design["reg_factors"])


def _left(task, design: dict) -> int:
    model, algorithm, degree, rho, level, budget, kind, i = task
    file = {"sets": "runs.csv", "redraws": "redraws.csv", "regularizer": "regularizer.csv"}[kind]
    path = pair_dir(model, algorithm, degree, rho, budget, i) / file
    return max(_planned(task, design) - len(G._completed_ids(path, "run_id")), 0)


def _load(task, design: dict):
    model, algorithm, degree, rho, level, budget, kind, i = task
    seed, A, B, truth, test, candidates, info = RM.load_pair(model, degree, rho, PAIR_INDEX_OFFSET + i, design)
    graph = GraphA(A)
    table, arrays, stats = plan_sets(algorithm, seed, A, graph, test, candidates, budget, design)
    digest = hashlib.sha256(b"".join(s.tobytes() for s in arrays) + test.tobytes()).hexdigest()[:16]
    return seed, A, B, truth, test, info, table, arrays, stats, digest


def _write_sets(folder: Path, table: pd.DataFrame, arrays: list, A: np.ndarray, B: np.ndarray, truth: np.ndarray,
                info: dict, digest: str) -> None:
    sets_path = folder / "sets.csv"
    if sets_path.exists() and (folder / "sets.hash").exists() and (folder / "sets.hash").read_text() == digest:
        return
    full = _with_all_features(table, arrays, A, B[np.ix_(truth, truth)])
    for column, value in (("pair_seed", info["pair_seed"]), ("rho", info["rho"]), ("mean_degree", info["mean_degree"]),
                          ("model", info["model"])):
        full.insert(0, column, value)
    temporary = sets_path.with_suffix(f".{os.getpid()}.tmp")
    full.to_csv(temporary, index=False)
    temporary.replace(sets_path)
    (folder / "sets.hash").write_text(digest)


def job(task, design: dict) -> str:
    model, algorithm, degree, rho, level, budget, kind, i = task
    folder = pair_dir(model, algorithm, degree, rho, budget, i)
    if _left(task, design) == 0:
        return f"{describe(task)}: already done"
    started = time.perf_counter()
    with thread_limit(design["blas_threads"]):
        seed, A, B, truth, test, info, table, arrays, stats, digest = _load(task, design)
        folder.mkdir(parents=True, exist_ok=True)
        info_path = folder / "pair.json"
        if info_path.exists():
            saved = _read_json(info_path)
            for key in ("hash_a", "hash_b", "design_hash"):
                if key in saved and saved[key] != {**info, "design_hash": digest}[key]:
                    raise RuntimeError(f"{describe(task)}: the saved runs come from a different pair or plan ({key}). "
                                       "Set LANDSCAPE_OUTPUT_DIR to a new folder to keep them apart.")
        else:
            _write_json(info_path, {**info, "algorithm": algorithm, "level": level, "budget": budget,
                                    "design_hash": digest, "pool_stats": stats, "n_sets": len(table),
                                    "written_at": _now()})
        _write_sets(folder, table, arrays, A, B, truth, info, digest)
        base = {"model": model, "mean_degree": degree, "rho": rho, "algorithm": algorithm, "level": level,
                "budget": budget, "pair_index": i, "pair_seed": seed}
        written = 0
        if kind == "sets":
            path = folder / "runs.csv"
            done = G._completed_ids(path, "run_id")
            rows = G._Rows(path)
            try:
                for repeat in range(design["runs"][algorithm]):          # every set once, then the repeats
                    for set_id, seeds in enumerate(arrays):
                        run_id = 10 * set_id + repeat
                        if run_id in done:
                            continue
                        solver_seed = _solver_seed(seed, budget, algorithm, "sets", set_id, repeat)
                        out = RM.run_one(algorithm, model, (A, B, truth), seeds, test, solver_seed)
                        rows.write({**base, "set_id": set_id, "design": table.at[set_id, "design"],
                                    "detail": table.at[set_id, "detail"], "repeat": repeat, "run_id": run_id,
                                    "solver_seed": solver_seed, **{c: out[c] for c in OUTCOME_COLUMNS}})
                        written += 1
            finally:
                rows.close()
        else:
            chosen = redraw_sets(table, seed, budget, design["redraw_sets"])
            if kind == "redraws":
                probability = b_probability(model, degree, rho, seed, A, B, truth)
                path = folder / "redraws.csv"
                done = G._completed_ids(path, "run_id")
                draws_path = folder / "draws.json"
                draws = _read_json(draws_path) if draws_path.exists() else {}
                rows = G._Rows(path)
                try:
                    for draw in range(1, design["redraws"] + 1):
                        ids = [100_000 * draw + int(s) for s in chosen]
                        if all(r in done for r in ids):
                            continue
                        new_b, new_truth, rejected = BR.draw_b(A, probability, seed, draw)
                        draws[str(draw)] = {"hash_b": rc._hash(new_b), "rejected": int(rejected)}
                        _write_json(draws_path, draws)
                        for set_id, run_id in zip(chosen, ids):
                            if run_id in done:
                                continue
                            solver_seed = _solver_seed(seed, budget, algorithm, "redraws", int(set_id), draw)
                            out = RM.run_one(algorithm, model, (A, new_b, new_truth), arrays[int(set_id)], test,
                                             solver_seed)
                            rows.write({**base, "set_id": int(set_id), "draw": draw, "run_id": run_id,
                                        "solver_seed": solver_seed, **{c: out[c] for c in OUTCOME_COLUMNS}})
                            written += 1
                finally:
                    rows.close()
            else:   # regularizer (GOAT)
                path = folder / "regularizer.csv"
                done = G._completed_ids(path, "run_id")
                rows = G._Rows(path)
                try:
                    for f, factor in enumerate(design["reg_factors"], 1):
                        regularizer = design["solver"]["goat_regularizer"] * factor
                        for set_id in chosen:
                            run_id = 100_000 * f + int(set_id)
                            if run_id in done:
                                continue
                            solver_seed = _solver_seed(seed, budget, algorithm, "regularizer", int(set_id), f)
                            out = RM.run_one(algorithm, model, (A, B, truth), arrays[int(set_id)], test, solver_seed,
                                             regularizer)
                            rows.write({**base, "set_id": int(set_id), "factor": factor, "regularizer": regularizer,
                                        "run_id": run_id, "solver_seed": solver_seed,
                                        **{c: out[c] for c in OUTCOME_COLUMNS}})
                            written += 1
                finally:
                    rows.close()
    return f"{describe(task)}: {written} runs in {(time.perf_counter() - started) / 60:.1f} min"


# --------------------------------------------------------------------------
# Planning and running
# --------------------------------------------------------------------------

def _budget(model: str, algorithm: str, degree: float, rho: float, level: float, budget: int | None) -> dict:
    if budget:
        return {"budget": int(budget), "how": "set with --budget"}
    return cell_budget(model, algorithm, degree, rho, level)


def tasks_for(model: str, algorithms=ALGORITHMS, levels=(LEVEL,), extra_cells=None, n_pairs: int | None = None,
              extras: bool = True, smoke: bool = False, design: dict | None = None, report=print,
              budget: int | None = None):
    """All tasks for a model, and notes on cells skipped (no budget yet)."""

    design = design or prepare(model, smoke=smoke)
    n_pairs = n_pairs or (SMOKE["pairs"] if smoke else PAIRS)
    tasks, notes = [], []
    for algorithm, degree, rho in cells_for(model, algorithms, extra_cells, smoke):
        for level in levels:
            entry = _budget(model, algorithm, degree, rho, level, budget)
            if entry["budget"] is None:
                notes.append(f"{LABELS[algorithm]} d{degree:g} rho {rho:.2f} at {level:.0%}: skipped ({entry['how']})")
                continue
            for i in range(n_pairs):
                kinds = ["sets"]
                if extras and i == 0:
                    kinds += ["redraws"] + (["regularizer"] if algorithm == "goat" else [])
                tasks += [(model, algorithm, degree, rho, float(level), int(entry["budget"]), kind, i)
                          for kind in kinds]
    for note in notes:
        report(note)
    return tasks, design


def run(model: str, algorithms=ALGORITHMS, levels=(LEVEL,), extra_cells=None, n_pairs: int | None = None,
        extras: bool = True, jobs: int = 3, blas_threads: int = 3, smoke: bool = False,
        budget: int | None = None) -> bool:
    """Run every task not yet finished. Returns False if stopped with Ctrl-C."""

    design = prepare(model, blas_threads, smoke)
    tasks, _ = tasks_for(model, algorithms, levels, extra_cells, n_pairs, extras, smoke, design, budget=budget)
    tasks = [t for t in tasks if _left(t, design) > 0]
    if not tasks:
        print(f"Landscape: nothing left to run for {MODEL_LABELS[model]}. Results are in {OUTPUT_DIR / model}")
        return True

    def seconds(task):
        return _left(task, design) * RM.SECONDS_PER_RUN[task[1]]

    tasks.sort(key=seconds, reverse=True)
    print(f"{_clock()}  Landscape, {MODEL_LABELS[model]}: {len(tasks)} tasks, {jobs} at a time, about "
          f"{sum(seconds(t) for t in tasks) / jobs / 3600:.1f} h (a rough guess). One line per finished task.",
          flush=True)
    return _run(tasks, job, design, seconds, jobs)


def _run(tasks: list, job_function, design: dict, seconds, jobs: int) -> bool:
    """Run tasks in parallel (spawned workers), report each one, keep going after failures."""

    failures, total = [], len(tasks)

    def report(n, task, message, error, remaining):
        if error is None:
            left = sum(seconds(t) for t in remaining) / max(jobs, 1) / 3600
            print(f"{_clock()}  [{n}/{total}] {message}; about {left:.1f} h left", flush=True)
            return
        failures.append(describe(task))
        print(f"{_clock()}  [{n}/{total}] {failures[-1]} FAILED:", flush=True)
        traceback.print_exception(type(error), error, error.__traceback__)

    keep_awake = None
    if sys.platform == "darwin":
        import shutil
        import subprocess
        if shutil.which("caffeinate"):
            keep_awake = subprocess.Popen(["caffeinate", "-i"])
    pool = None
    try:
        if jobs <= 1:
            for n, task in enumerate(tasks, 1):
                try:
                    message = job_function(task, design)
                except Exception as error:  # noqa: BLE001 - report every failure, keep going
                    report(n, task, None, error, [])
                    continue
                report(n, task, message, None, tasks[n:])
        else:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor, as_completed
            pool = ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn"),
                                       initializer=G._ignore_interrupts)
            futures = {pool.submit(job_function, task, design): task for task in tasks}
            for n, future in enumerate(as_completed(futures), 1):
                try:
                    message = future.result()
                except Exception as error:  # noqa: BLE001
                    report(n, futures[future], None, error, [])
                    continue
                report(n, futures[future], message, None, [t for f, t in futures.items() if not f.done()])
            pool.shutdown(wait=True)
    except KeyboardInterrupt:
        if pool is not None:
            workers = list((getattr(pool, "_processes", None) or {}).values())
            pool.shutdown(wait=False, cancel_futures=True)
            for worker in workers:
                worker.terminate()
        print(f"\n{_clock()}  Stopped. Finished runs are kept; rerun to resume.", flush=True)
        return False
    finally:
        if keep_awake is not None:
            keep_awake.terminate()
    if failures:
        raise RuntimeError(f"Failed: {'; '.join(failures)}. Rerun to retry them (finished runs are kept).")
    print(f"{_clock()}  Done. Results are in {OUTPUT_DIR}", flush=True)
    return True


# --------------------------------------------------------------------------
# Checks, status and reading
# --------------------------------------------------------------------------

def check_pairs(model: str, algorithms=ALGORITHMS, extra_cells=None) -> pd.DataFrame:
    """Draw the first landscape pair of every cell (and one redraw of B): catches settings too sparse for the
    structure check before a long run. No matching runs."""

    design = {"excluded_pair_seeds": [], "n_test": N_TEST}
    rows, seen = [], set()
    for algorithm, degree, rho in cells_for(model, algorithms, extra_cells):
        if (degree, rho) in seen:
            continue
        seen.add((degree, rho))
        started = time.perf_counter()
        try:
            seed, A, B, truth, test, candidates, info = RM.load_pair(model, degree, rho, PAIR_INDEX_OFFSET, design)
            probability = b_probability(model, degree, rho, seed, A, B, truth)
            _, _, rejected = BR.draw_b(A, probability, seed, 1)
            rows.append({"mean degree": degree, "rho": rho, "pair": "ok", "rejected draws": info["rejected_draws"],
                         "mean degree of A": round(info["mean_degree_a"], 2),
                         "redraw of B": f"ok ({rejected} rejected)",
                         "seconds": round(time.perf_counter() - started, 1)})
        except Exception as error:  # noqa: BLE001
            rows.append({"mean degree": degree, "rho": rho, "pair": f"FAILED: {error}"[:120],
                         "seconds": round(time.perf_counter() - started, 1)})
    return pd.DataFrame(rows)


def status(model: str, algorithms=ALGORITHMS, levels=(LEVEL,), extra_cells=None, n_pairs: int | None = None,
           extras: bool = True, smoke: bool = False, budget: int | None = None) -> pd.DataFrame:
    """Runs finished per cell and kind of task. Budget "-" means not chosen yet (the regime map's stage 1 for that
    cell isn't finished, or the landscape hasn't been started)."""

    path = OUTPUT_DIR / model / "settings.json"
    design = _read_json(path) if path.exists() else _fixed(model, smoke)
    design = {**design, "excluded_pair_seeds": []}
    saved = OUTPUT_DIR / model / "budgets.json"
    budgets = _read_json(saved) if saved.exists() else {}
    rows = []
    for algorithm, degree, rho in cells_for(model, algorithms, extra_cells, smoke):
        for level in levels:
            entry = ({"budget": int(budget)} if budget
                     else budgets.get(_budget_key(algorithm, degree, rho, level)))
            row = {"algorithm": LABELS[algorithm], "mean degree": degree, "rho": rho, "level": level,
                   "budget": "-" if entry is None else (entry["budget"] or "skipped")}
            if entry and entry["budget"]:
                for kind in KINDS:
                    if kind == "regularizer" and algorithm != "goat":
                        continue
                    pairs = range(n_pairs or (SMOKE["pairs"] if smoke else PAIRS)) if kind == "sets" else (
                        [0] if extras else [])
                    tasks = [(model, algorithm, degree, rho, level, entry["budget"], kind, i) for i in pairs]
                    planned = sum(_planned(t, design) for t in tasks)
                    done = planned - sum(_left(t, design) for t in tasks)
                    row[kind] = f"{done} of {planned} runs" if planned else "-"
            rows.append(row)
    return pd.DataFrame(rows).set_index(["algorithm", "mean degree", "rho", "level"])


def read(model: str, kind: str = "sets") -> pd.DataFrame:
    """Runs joined with their sets' features. kind: "sets" (runs.csv), "redraws" or "regularizer"."""

    file = {"sets": "runs.csv", "redraws": "redraws.csv", "regularizer": "regularizer.csv"}[kind]
    frames = []
    for path in sorted((OUTPUT_DIR / model).glob(f"d*_rho*/*/k*/pair_*/{file}")):
        if path.stat().st_size == 0 or not (path.parent / "sets.csv").exists():
            continue
        runs = pd.read_csv(path)
        sets = pd.read_csv(path.parent / "sets.csv")
        sets = sets.drop(columns=[c for c in ("model", "mean_degree", "rho", "pair_seed", "design", "detail")
                                  if c in sets.columns and c in runs.columns])
        frames.append(runs.merge(sets, on="set_id", how="left"))
    if not frames:
        raise FileNotFoundError(f"No landscape {kind} results yet in {OUTPUT_DIR / model}")
    return pd.concat(frames, ignore_index=True)


def summary(model: str, data: pd.DataFrame | None = None) -> pd.DataFrame:
    """A first look per cell and budget: random sets' mean and SD of test accuracy (FAQ's repeats averaged per set),
    the share of random sets ending part-way (20-80%), the Spearman correlation of the frozen score and of each core
    feature with accuracy over random sets (within pairs, averaged over pairs), and the mean accuracy of every rule,
    screen and degree tilt."""

    data = read(model) if data is None else data
    data = data[data["error"].fillna("").astype(str) == ""]
    rows = []
    for (algorithm, degree, rho, budget), part in data.groupby(["algorithm", "mean_degree", "rho", "budget"]):
        per_set = part.groupby(["pair_index", "set_id"]).agg(
            accuracy=("accuracy_test", "mean"), design=("design", "first"), detail=("detail", "first"),
            score=("score", "first"), **{f: (f, "first") for f in CORE}).reset_index()
        random = per_set[per_set["design"] == "random"]
        row = {"algorithm": algorithm, "mean degree": degree, "rho": rho, "budget": int(budget),
               "pairs": part["pair_index"].nunique(), "random sets": len(random),
               "random mean": random["accuracy"].mean(), "random SD": random["accuracy"].std(),
               "part-way share": random["accuracy"].between(0.2, 0.8).mean()}
        for column in ("score", *CORE):
            values = [g[column].rank().corr(g["accuracy"].rank()) for _, g in random.groupby("pair_index")
                      if g["accuracy"].nunique() > 1 and g[column].nunique() > 1]
            row[f"rho({column})"] = float(np.mean(values)) if values else np.nan
        for kind in ("rule", "screen", "tilted"):
            for detail, group in per_set[per_set["design"] == kind].groupby("detail"):
                row[detail] = group["accuracy"].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index(["algorithm", "mean degree", "rho", "budget"])


# --------------------------------------------------------------------------
# Analysis (landscape.ipynb)
# --------------------------------------------------------------------------

BLUE, ORANGE, RED, INK, INK2, GRID, BAND, NEUTRAL = (RM.BLUE, RM.ORANGE, RM.RED, RM.INK, RM.INK2, RM.GRID, RM.BAND,
                                                     RM.NEUTRAL)
CONTEXT = "#b9b8b2"                    # random sets on one B: the grey layer behind the blue means over draws of B
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")   # validated categorical slots 1-5, in this order
TILT_RAMP = ("#86b6ef", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b")   # gamma -1 ... 3 (light to dark)
MIDDLE = RM.MIDDLE
KEYS = ["model", "algorithm", "mean_degree", "rho", "level", "budget", "pair_index", "set_id"]
CELL_KEYS = ["model", "algorithm", "mean_degree", "rho", "level", "budget"]
RUN_COLUMNS = {"repeat", "run_id", "solver_seed", "accuracy_test", "accuracy", "success", "h0_accuracy", "objective",
               "n_iter", "converged", "runtime_seconds", "matched_fraction", "error", "draw", "factor", "regularizer",
               "seed_vertices", "pair_seed"}
FEATURE_LABELS = {
    "z_degree": "degree (z)", "z_harmonic": "harmonic (z)", "z_pair_mass": "pair mass (z)",
    "z_uniqueness": "uniqueness (z)", "score": "frozen score",
    **{f: f"{label}" for f, label in G.FEATURE_LABELS.items()},
    "oracle_cut_edges_kept": "B: seed-non-seed edges kept", "oracle_cut_kept_share": "B: share of those edges kept",
    "oracle_b_degree": "B: degree of the seeds' matches", "oracle_seed_edges_kept_share": "B: seed-seed edges kept",
    "oracle_nonseeds_with_common_clue": "B: non-seeds with a kept clue",
    "oracle_common_clues_mean": "B: kept clues per non-seed",
}
FEATURE_GROUPS = {   # rows of the feature heatmap, in this order. The core z-scores are left out: within a pair they
    # rank sets exactly as the fixed-pair degree, harmonic, pair mass and uniqueness do; so is the kept-clues mean, which
    # ranks them as the seed-non-seed edges kept do
    "graph A: the frozen score": ["score"],
    "graph A: fixed-pair features": list(MP.FEATURE_COLUMNS),
    "graph B (not visible to a rule)": [c for c in ORACLE_COLUMNS if c != "oracle_common_clues_mean"],
}
QUANTILE_LABELS = {"degree": "degree", "harmonic": "harmonic", "pair_mass": "pair mass", "uniqueness": "uniqueness",
                   "score": "frozen score"}


def _ok(frame: pd.DataFrame) -> pd.Series:
    """Runs without an error."""
    return frame["error"].fillna("").astype(str).str.len() == 0


def _spearman(x, y) -> float:
    x, y = pd.Series(np.asarray(x, dtype=float)), pd.Series(np.asarray(y, dtype=float))
    keep = x.notna() & y.notna()
    if keep.sum() < 3 or x[keep].nunique() < 2 or y[keep].nunique() < 2:
        return np.nan
    return float(x[keep].rank().corr(y[keep].rank()))


def _within_pair(part: pd.DataFrame, x: str, y: str) -> float:
    """Spearman correlation inside each pair, averaged over pairs (pairs differ in level, so pooling them would mix
    pair differences into the ranking)."""
    values = [_spearman(g[x], g[y]) for _, g in part.groupby("pair_index")]
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else np.nan


def _order(table: pd.DataFrame) -> pd.DataFrame:
    """Rows in a fixed order: model (ER, Chung-Lu, PAPER), algorithm (percolation, FAQ, GOAT), mean degree, rho,
    level."""
    if table.empty:
        return table
    keys = {"model": {m: i for i, m in enumerate(MODELS)}, "algorithm": {a: i for i, a in enumerate(ALGORITHMS)}}
    order = table.assign(_m=table["model"].map(keys["model"]), _a=table["algorithm"].map(keys["algorithm"]))
    columns = ["_m", "_a"] + [c for c in ("mean_degree", "rho", "level", "budget") if c in table.columns]
    return order.sort_values(columns, kind="stable").drop(columns=["_m", "_a"]).reset_index(drop=True)


def _standard(model: str) -> list[tuple[str, float, float]]:
    return cells_for(model)


def cell_kind(model: str, algorithm: str, degree: float, rho: float, level: float) -> str:
    """'standard' (the 9 cells at the 50% budget), 'level 20%' / 'level 80%' (the same cells at other budgets), or
    'extra' (cells added with --cells)."""
    if (algorithm, degree, float(rho)) in {(a, d, float(r)) for a, d, r in _standard(model)}:
        return "standard" if np.isclose(level, LEVEL) else f"level {level:.0%}"
    return "extra"


def cell_title(algorithm: str, degree: float, rho: float, budget: int | None = None) -> str:
    text = f"{LABELS[algorithm]} · d{degree:g}, ρ {rho:g}"
    return text if budget is None else f"{text} · {int(budget)} seeds"


def per_set(models=MODELS) -> pd.DataFrame:
    """One row per pair and seed set: its features, and its test accuracy on the pair's saved B ("accuracy": FAQ's
    3 runs averaged; "accuracy_first": the first run only, a true single run), first-step accuracy, and the cell's
    kind (standard, other level, extra)."""

    frames = []
    for model in models:
        try:
            data = read(model)
        except FileNotFoundError:
            continue
        data = data[_ok(data)].sort_values("repeat", kind="stable")
        outcome = data.groupby(KEYS, sort=False).agg(accuracy=("accuracy_test", "mean"),
                                                     accuracy_first=("accuracy_test", "first"),
                                                     h0_accuracy=("h0_accuracy", "mean"),
                                                     runs=("accuracy_test", "size"))
        features = data.drop_duplicates(KEYS).set_index(KEYS)
        features = features[[c for c in features.columns if c not in RUN_COLUMNS]]
        frames.append(outcome.join(features).reset_index())
    if not frames:
        raise FileNotFoundError(f"No landscape results yet in {OUTPUT_DIR}")
    out = pd.concat(frames, ignore_index=True)
    out["kind"] = [cell_kind(*row) for row in
                   out[["model", "algorithm", "mean_degree", "rho", "level"]].itertuples(index=False)]
    return out


def overview(sets: pd.DataFrame) -> pd.DataFrame:
    """What was collected: per model and kind of cell, the cells, pairs, seed sets and runs."""
    rows = []
    for (model, kind), part in sets.groupby(["model", "kind"], sort=False):
        rows.append({"model": MODEL_LABELS[model], "kind": kind, "cells": len(part[CELL_KEYS].drop_duplicates()),
                     "pairs": len(part[CELL_KEYS + ["pair_index"]].drop_duplicates()), "seed sets": len(part),
                     "runs": int(part["runs"].sum())})
    return pd.DataFrame(rows).set_index(["model", "kind"])


def draws(sets: pd.DataFrame, models=MODELS) -> pd.DataFrame:
    """Test accuracy of each redrawn set on each draw of B (pair 0, 100 random sets per cell): draw 0 is the pair's
    saved B (the set's first run there), draws 1-10 are fresh draws of B given A."""

    frames = []
    for model in models:
        try:
            redraws = read(model, "redraws")
        except FileNotFoundError:
            continue
        fresh = redraws[_ok(redraws)][KEYS + ["draw", "accuracy_test"]]
        saved = sets.merge(fresh[KEYS].drop_duplicates(), on=KEYS)[KEYS + ["accuracy_first"]]
        frames.append(pd.concat([fresh, saved.rename(columns={"accuracy_first": "accuracy_test"}).assign(draw=0)],
                                ignore_index=True))
    if not frames:
        return pd.DataFrame(columns=KEYS + ["draw", "accuracy_test"])
    return pd.concat(frames, ignore_index=True)


def set_quality(draw_table: pd.DataFrame, sets: pd.DataFrame) -> pd.DataFrame:
    """Per redrawn set: its mean test accuracy over the draws of B (its quality as far as graph A can tell), the share
    of draws where it reaches 0.5, its 10th percentile and spread over draws, with its features."""

    groups = draw_table.groupby(KEYS)["accuracy_test"]
    quality = pd.DataFrame({"mean_over_B": groups.mean(),
                            "success_over_B": (draw_table["accuracy_test"] >= 0.5).groupby(
                                [draw_table[k] for k in KEYS]).mean(),
                            "p10_over_B": groups.quantile(0.1), "sd_over_B": groups.std(), "draws": groups.size()})
    features = sets.set_index(KEYS)[["score", "z_degree", "design", "kind"]]
    return quality.join(features, how="left").reset_index()


def variance_table(draw_table: pd.DataFrame) -> pd.DataFrame:
    """How much of a single run is the seed set? Per cell, from the 100 sets x 11 draws of B (two-way, without
    replication): the shares of single-run variance due to the draw of B (moves every set together), the seed set,
    and the rest (which sets this B favours, and solver noise); the set's share on a fixed B; its square root, the
    ceiling on any graph-A predictor's correlation with single runs; and how well rankings on one draw carry over
    to another (mean Spearman between draws)."""

    rows = []
    for key, part in draw_table.groupby(CELL_KEYS, sort=False):
        matrix = part.pivot_table(index="set_id", columns="draw", values="accuracy_test").dropna()
        y = matrix.to_numpy()
        n, k = y.shape
        if n < 10 or k < 3:
            continue
        grand, rows_mean, cols_mean = y.mean(), y.mean(axis=1), y.mean(axis=0)
        msr = k * ((rows_mean - grand) ** 2).sum() / (n - 1)
        msc = n * ((cols_mean - grand) ** 2).sum() / (k - 1)
        mse = ((y - rows_mean[:, None] - cols_mean[None, :] + grand) ** 2).sum() / ((n - 1) * (k - 1))
        var_set, var_draw = max((msr - mse) / k, 0.0), max((msc - mse) / n, 0.0)
        total = var_set + var_draw + mse
        fixed_b = max((msr - mse) / (msr + (k - 1) * mse), 0.0) if msr + (k - 1) * mse > 0 else np.nan
        agreement = np.nanmean([_spearman(y[:, i], y[:, j]) for i in range(k) for j in range(i + 1, k)])
        rows.append({**dict(zip(CELL_KEYS, key)), "kind": cell_kind(*key[:5]), "sets": n, "draws of B": k,
                     "share: draw of B": var_draw / total if total else np.nan,
                     "share: seed set": var_set / total if total else np.nan,
                     "share: rest": mse / total if total else np.nan,
                     "seed-set share on a fixed B": fixed_b, "ceiling on r": np.sqrt(fixed_b),
                     "ranking agreement between draws": agreement,
                     "draw means": f"{cols_mean.min():.2f}-{cols_mean.max():.2f}"})
    return _order(pd.DataFrame(rows))


def ranking_table(sets: pd.DataFrame, quality: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per cell, over random sets: the mean and SD of test accuracy, the share ending part-way (0.2-0.8, first runs),
    and how well the frozen score and degree alone rank the sets: on the saved B (Spearman within pair, averaged
    over pairs) and, for the 100 redrawn sets, against their mean and success rate over the draws of B."""

    rows = []
    for key, part in sets[sets["design"] == "random"].groupby(CELL_KEYS, sort=False):
        row = {**dict(zip(CELL_KEYS, key)), "kind": part["kind"].iloc[0], "pairs": part["pair_index"].nunique(),
               "random sets": len(part), "mean": part["accuracy"].mean(), "SD": part["accuracy"].std(),
               "part-way share": part["accuracy_first"].between(*MIDDLE).mean(),
               "score, one B": _within_pair(part, "score", "accuracy"),
               "degree, one B": _within_pair(part, "z_degree", "accuracy")}
        if quality is not None:
            q = quality.merge(pd.DataFrame([dict(zip(CELL_KEYS, key))]), on=CELL_KEYS)
            if len(q):
                row.update({"score, mean over B": _spearman(q["score"], q["mean_over_B"]),
                            "degree, mean over B": _spearman(q["z_degree"], q["mean_over_B"]),
                            "score, success over B": _spearman(q["score"], q["success_over_B"])})
        rows.append(row)
    return _order(pd.DataFrame(rows))


def show(table: pd.DataFrame, kind: str | None = "standard", digits: int = 2) -> pd.DataFrame:
    """A table for display: readable cell names as the index, numbers rounded."""

    out = table if kind is None or "kind" not in table else table[table["kind"] == kind]
    out = out.assign(model=out["model"].map(MODEL_LABELS), algorithm=out["algorithm"].map(LABELS))
    out = out.rename(columns={"mean_degree": "mean degree", "rho": "ρ"})
    index = [c for c in ("model", "algorithm", "mean degree", "ρ", "level", "budget") if c in out.columns]
    out = out.drop(columns=[c for c in ("kind",) if c in out.columns]).set_index(index)
    return out.round(digits)


def _cells(sets: pd.DataFrame, model: str, kind: str = "standard") -> list[tuple]:
    """(algorithm, mean degree, rho, level, budget) of the model's cells of this kind: the standard cells in the
    CELLS order (large, moderate, small rule gain in the ER map), others sorted."""

    part = sets[(sets["model"] == model) & (sets["kind"] == kind)]
    found = part[["algorithm", "mean_degree", "rho", "level", "budget"]].drop_duplicates()
    order = {(a, _model_degree(model, d), float(r)): i for a in ALGORITHMS for i, (d, r) in enumerate(CELLS[a])}
    found = found.assign(algorithm_order=found["algorithm"].map({a: i for i, a in enumerate(ALGORITHMS)}),
                         cell_order=[order.get((a, d, float(r)), 99) for a, d, r in
                                     found[["algorithm", "mean_degree", "rho"]].itertuples(index=False)])
    found = found.sort_values(["algorithm_order", "cell_order", "mean_degree", "rho", "level"])
    return [tuple(row) for row in found[["algorithm", "mean_degree", "rho", "level", "budget"]].itertuples(index=False)]


def _grid(cells: list[tuple]):
    """Rows: algorithms present; columns: that algorithm's cells. When every algorithm has a single cell, one row with
    a column per algorithm (returned as a single row keyed by the first algorithm)."""
    algorithms = [a for a in ALGORITHMS if any(c[0] == a for c in cells)]
    per_row = {a: [c for c in cells if c[0] == a] for a in algorithms}
    if len(algorithms) > 1 and all(len(v) == 1 for v in per_row.values()):
        return [algorithms[0]], {algorithms[0]: [v[0] for v in per_row.values()]}, len(algorithms)
    return algorithms, per_row, max(len(v) for v in per_row.values())


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_score_vs_accuracy(sets: pd.DataFrame, quality: pd.DataFrame | None, model: str, kind: str = "standard",
                           x: str = "score"):
    """Frozen score (or another feature) against test accuracy, one panel per cell.
    Grey: every random set on its pair's saved B (FAQ: mean of its 3 runs), all pairs.
    Blue: the 100 redrawn random sets of pair 0, each averaged over 11 draws of B (its quality as far as graph A can
    tell). Titles: Spearman within pair on the saved B, and Spearman with the mean over draws of B."""

    import matplotlib.pyplot as plt

    cells = _cells(sets, model, kind)
    if not cells:
        raise ValueError(f"No {kind} cells for {model}")
    algorithms, per_row, columns = _grid(cells)
    xlabel = FEATURE_LABELS.get(x, x) + (" (z units)" if x == "score" else "")
    with plt.rc_context({"font.size": 9}):
        fig, axes = plt.subplots(len(algorithms), columns, figsize=(3.7 * columns, 3.3 * len(algorithms) + 0.7),
                                 squeeze=False, sharey=True)
        for r, row_key in enumerate(algorithms):
            for c in range(columns):
                ax = axes[r, c]
                if c >= len(per_row[row_key]):
                    ax.axis("off")
                    continue
                a, d, rho, level, budget = per_row[row_key][c]
                cell = {"model": model, "algorithm": a, "mean_degree": d, "rho": rho, "level": level, "budget": budget}
                part = sets.merge(pd.DataFrame([cell]), on=CELL_KEYS)
                random = part[part["design"] == "random"]
                _style(ax)
                ax.scatter(random[x], random["accuracy"], s=7, color=CONTEXT, alpha=0.5, linewidths=0,
                           label="random sets on the saved B (all pairs)")
                text = f"Spearman: one B {_within_pair(random, x, 'accuracy'):.2f}"
                if quality is not None:
                    q = quality.merge(pd.DataFrame([cell]), on=CELL_KEYS)
                    if len(q):
                        ax.scatter(q[x], q["mean_over_B"], s=24, color=BLUE, edgecolors="white", linewidths=0.7,
                                   zorder=3, label="100 sets, mean over 11 draws of B")
                        text += f" · mean over B {_spearman(q[x], q['mean_over_B']):.2f}"
                level_note = "" if kind == "standard" else f" · {level:.0%} level"
                ax.set_title(f"{cell_title(a, d, rho, budget)}{level_note}\n", loc="left", fontsize=9.5)
                ax.text(0, 1.02, text, transform=ax.transAxes, va="bottom", fontsize=8.3, color=INK2)
                ax.set_ylim(-0.03, 1.03)
                if c == 0:
                    ax.set_ylabel("test accuracy")
                if r == len(algorithms) - 1:
                    ax.set_xlabel(xlabel)
        handles, labels = [], []
        for ax in axes.ravel():
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        height = fig.get_figheight()
        fig.suptitle(f"{MODEL_LABELS[model]}: {FEATURE_LABELS.get(x, x)} vs test accuracy", x=0.01, y=0.995,
                     ha="left", va="top", fontsize=11.5)
        fig.legend(handles, labels, loc="upper left", frameon=False, ncol=3, fontsize=9, markerscale=1.6,
                   bbox_to_anchor=(0.005, 1 - 0.4 / height))
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.68 / height))
    return fig


def plot_designed_sets(sets: pd.DataFrame, model: str, kind: str = "standard"):
    """Every seed set on its pair's saved B, by design: random sets (grey), degree-tilted sets (blue, light to dark
    for gamma -1 to 3), the pool's sets picked at quantiles of the frozen score (pink), round 1's rules (orange
    shapes) and the best-of screens (aqua). Shows where constructed sets sit on the score axis and whether accuracy
    keeps rising past the random sets' range."""

    import matplotlib.pyplot as plt

    cells = _cells(sets, model, kind)
    algorithms, per_row, columns = _grid(cells)
    gammas = sorted({float(g.split("=")[1]) for g in sets.loc[sets["design"] == "tilted", "detail"].unique()})
    with plt.rc_context({"font.size": 9}):
        fig, axes = plt.subplots(len(algorithms), columns, figsize=(3.7 * columns, 3.3 * len(algorithms) + 1.1),
                                 squeeze=False, sharey=True)
        for r, row_key in enumerate(algorithms):
            for c in range(columns):
                ax = axes[r, c]
                if c >= len(per_row[row_key]):
                    ax.axis("off")
                    continue
                a, d, rho, level, budget = per_row[row_key][c]
                cell = {"model": model, "algorithm": a, "mean_degree": d, "rho": rho, "level": level, "budget": budget}
                part = sets.merge(pd.DataFrame([cell]), on=CELL_KEYS)
                _style(ax)
                random = part[part["design"] == "random"]
                ax.scatter(random["score"], random["accuracy"], s=6, color=CONTEXT, alpha=0.45, linewidths=0,
                           label="random sets")
                for g, color in zip(gammas, TILT_RAMP):
                    tilt = part[part["detail"] == f"gamma={g:g}"]
                    ax.scatter(tilt["score"], tilt["accuracy"], s=14, color=color, linewidths=0, alpha=0.9,
                               label=f"degree tilt γ = {g:g}")
                quant = part[part["detail"].astype(str).str.startswith("score@")]
                ax.scatter(quant["score"], quant["accuracy"], s=18, color="#e87ba4", marker="D", linewidths=0,
                           label="pool sets at score quantiles")
                for method in RM.RULES[a]:
                    rule = part[part["detail"] == method]
                    style = RM.METHOD_STYLE[method]
                    ax.scatter(rule["score"], rule["accuracy"], s=46, color=style["color"], marker=style["marker"],
                               edgecolors="white", linewidths=0.6, zorder=4, label=RM.METHOD_LABELS[method])
                screens = part[part["design"] == "screen"]
                ax.scatter(screens["score"], screens["accuracy"], s=30, color="#1baf7a", marker="X", linewidths=0,
                           zorder=4, label="best-of screens")
                ax.set_title(cell_title(a, d, rho, budget), loc="left", fontsize=9.5)
                ax.set_ylim(-0.03, 1.03)
                if c == 0:
                    ax.set_ylabel("test accuracy (saved B)")
                if r == len(algorithms) - 1:
                    ax.set_xlabel("frozen score (z units)")
        handles, labels = [], []
        for ax in axes.ravel():
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        height = fig.get_figheight()
        fig.suptitle(f"{MODEL_LABELS[model]}: every seed set by design (one point per set and pair)", x=0.01,
                     y=0.995, ha="left", va="top", fontsize=11.5)
        fig.legend(handles, labels, loc="upper left", frameon=False, ncol=6, fontsize=8.5, markerscale=1.2,
                   bbox_to_anchor=(0.005, 1 - 0.38 / height))
        fig.tight_layout(rect=(0, 0, 1, 1 - 1.05 / height))
    return fig


def plot_distributions(sets: pd.DataFrame, model: str, kind: str = "standard"):
    """Histograms of single runs (first runs) of random sets per cell: graded cells fill the 0.2-0.8 band (grey),
    all-or-nothing cells pile up at the ends. Ticks: each pair's mean."""

    import matplotlib.pyplot as plt

    cells = _cells(sets, model, kind)
    algorithms, per_row, columns = _grid(cells)
    bins = np.linspace(0, 1, 21)
    with plt.rc_context({"font.size": 9}):
        fig, axes = plt.subplots(len(algorithms), columns, figsize=(3.7 * columns, 2.5 * len(algorithms) + 0.5),
                                 squeeze=False)
        for r, row_key in enumerate(algorithms):
            for c in range(columns):
                ax = axes[r, c]
                if c >= len(per_row[row_key]):
                    ax.axis("off")
                    continue
                a, d, rho, level, budget = per_row[row_key][c]
                cell = {"model": model, "algorithm": a, "mean_degree": d, "rho": rho, "level": level, "budget": budget}
                random = sets.merge(pd.DataFrame([cell]), on=CELL_KEYS)
                random = random[random["design"] == "random"]
                ax.axvspan(*MIDDLE, color=BAND, zorder=0)
                ax.hist(random["accuracy_first"], bins=bins, color=BLUE, edgecolor="white", linewidth=0.6, zorder=2)
                for mean in random.groupby("pair_index")["accuracy_first"].mean():
                    ax.axvline(mean, ymin=0, ymax=0.06, color=INK, linewidth=1.4, zorder=3)
                share = random["accuracy_first"].between(*MIDDLE).mean()
                ax.set_title(f"{cell_title(a, d, rho, budget)}\n{share:.0%} of runs part-way", loc="left",
                             fontsize=9.2)
                ax.set_xlim(0, 1)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                if r == len(algorithms) - 1:
                    ax.set_xlabel("test accuracy, single run")
                if c == 0:
                    ax.set_ylabel("random sets")
        fig.suptitle(f"{MODEL_LABELS[model]}: how single runs of random sets end (grey band: part-way, 0.2-0.8; "
                     "ticks: pair means)", x=0.01, ha="left", fontsize=11.5)
        fig.tight_layout()
    return fig


CONSTRUCT_COLUMNS = ("random sets", "best of 100 by score", "best of 1,000 by score", "best of 1,000 by degree",
                     "best of ~18,650 by score", "degree tilt γ = 3", "round 1's rules (mean)", "best rule")


def construct_table(sets: pd.DataFrame, kind: str = "standard") -> pd.DataFrame:
    """Screening random sets against building sets, per cell (test accuracy on the saved B, averaged over pairs):
    random sets; the best of 100 / 1,000 random sets by the frozen score and of 1,000 by degree; the pool's top set by
    score (about 18,650 candidates: quantile pick score@1); sets drawn with probability proportional to degree^3; round
    1's rules on average, and the best rule (picked per cell, so optimistic)."""

    picks = {"best of 100 by score": "best_100", "best of 1,000 by score": "best_1000",
             "best of 1,000 by degree": "degree_1000", "best of ~18,650 by score": "score@1",
             "degree tilt γ = 3": "gamma=3"}
    rows = []
    for key, part in sets[sets["kind"] == kind].groupby(CELL_KEYS, sort=False):
        row = dict(zip(CELL_KEYS, key))
        row["kind"] = kind
        row["random sets"] = part.loc[part["design"] == "random", "accuracy"].mean()
        for name, detail in picks.items():
            row[name] = part.loc[part["detail"] == detail, "accuracy"].mean()
        rules = part[part["design"] == "rule"]
        row["round 1's rules (mean)"] = rules["accuracy"].mean()
        by_rule = rules.groupby("detail")["accuracy"].mean()
        row["best rule"] = by_rule.max() if len(by_rule) else np.nan
        row["which rule"] = by_rule.idxmax() if len(by_rule) else ""
        rows.append(row)
    return _order(pd.DataFrame(rows))


def plot_construct(table: pd.DataFrame):
    """One panel per model: test accuracy of each way of choosing seeds (x), one thin line per cell and the median
    over cells (thick). Screening random sets moves along the left half; building sets reaches the right half."""

    import matplotlib.pyplot as plt

    models = [m for m in MODELS if m in set(table["model"])]
    columns = list(CONSTRUCT_COLUMNS)
    x = np.arange(len(columns))
    with plt.rc_context({"font.size": 9}):
        fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.6), squeeze=False, sharey=True)
        for ax, model in zip(axes[0], models):
            part = table[table["model"] == model]
            for _, row in part.iterrows():
                ax.plot(x, row[columns].to_numpy(dtype=float), color=CONTEXT, linewidth=1, marker="o", markersize=3)
            median = part[columns].median()
            ax.plot(x, median.to_numpy(dtype=float), color=BLUE, linewidth=2.4, marker="o", markersize=6,
                    markeredgecolor="white", label="median over cells")
            _style(ax)
            ax.set_xticks(x, ["random\nsets", "best of\n100 by\nscore", "best of\n1,000 by\nscore",
                              "best of\n1,000 by\ndegree", "best of\n~18,650\nby score", "degree\ntilt\nγ = 3",
                              "rules\n(mean)", "best\nrule"], fontsize=7.8)
            ax.set_title(f"{MODEL_LABELS[model]} ({len(part)} cells)", loc="left", fontsize=10)
            ax.set_ylim(0, 1.02)
        axes[0, 0].set_ylabel("test accuracy (saved B, mean over pairs)")
        axes[0, 0].legend(frameon=False, loc="lower right")
        fig.suptitle("Screening random sets vs building sets: grey lines are cells, blue the median", x=0.01,
                     ha="left", fontsize=11.5)
        fig.tight_layout()
    return fig


def tilt_table(sets: pd.DataFrame, kind: str = "standard") -> pd.DataFrame:
    """Gain in test accuracy of degree-tilted sets over random sets of the same pair, per cell and gamma (averaged
    over the tilted sets and pairs)."""

    part = sets[sets["kind"] == kind]
    random = part[part["design"] == "random"].groupby(CELL_KEYS + ["pair_index"])["accuracy"].mean()
    tilted = part[part["design"] == "tilted"].copy()
    tilted["gamma"] = tilted["detail"].str.split("=").str[1].astype(float)
    tilted = tilted.join(random.rename("random_mean"), on=CELL_KEYS + ["pair_index"])
    tilted["gain"] = tilted["accuracy"] - tilted["random_mean"]
    return tilted.groupby(CELL_KEYS + ["gamma"])["gain"].mean().unstack("gamma").reset_index()


def plot_tilt(sets: pd.DataFrame, kind: str = "standard"):
    """Degree tilt: sets drawn with probability proportional to degree^gamma. One panel per model and algorithm;
    one line per cell: gain in test accuracy over random sets of the same pair (0 = random sets)."""

    import matplotlib.pyplot as plt

    table = tilt_table(sets, kind)
    models = [m for m in MODELS if m in set(table["model"])]
    gammas = [c for c in table.columns if isinstance(c, float)]
    with plt.rc_context({"font.size": 9}):
        fig, axes = plt.subplots(len(models), len(ALGORITHMS), figsize=(4.3 * len(ALGORITHMS), 2.9 * len(models)),
                                 squeeze=False, sharex=True, sharey=True)
        for r, model in enumerate(models):
            for c, a in enumerate(ALGORITHMS):
                ax = axes[r, c]
                _style(ax)
                ax.axhline(0, color=INK2, linewidth=0.8)
                part = table[(table["model"] == model) & (table["algorithm"] == a)]
                for (_, row), color in zip(part.iterrows(), SERIES):
                    ax.plot(gammas, row[gammas].to_numpy(dtype=float), color=color, linewidth=2, marker="o",
                            markersize=5, markeredgecolor="white",
                            label=f"d{row['mean_degree']:g}, ρ {row['rho']:g} · {int(row['budget'])} seeds")
                ax.set_title(f"{MODEL_LABELS[model]} · {LABELS[a]}", loc="left", fontsize=9.5)
                ax.legend(frameon=False, fontsize=7.5, loc="upper left")
                if c == 0:
                    ax.set_ylabel("gain over random sets")
                if r == len(models) - 1:
                    ax.set_xlabel("γ (probability ∝ degree^γ)")
        fig.suptitle("Degree-tilted seed sets: below 0 low-degree sets, above 0 high-degree sets", x=0.01, ha="left",
                     fontsize=11.5)
        fig.tight_layout()
    return fig


def plot_quantiles(sets: pd.DataFrame, model: str, kind: str = "standard"):
    """Sets from the pool picked at quantiles of one feature (lowest to highest), one line per feature: gain in test
    accuracy over random sets of the same pair, averaged over pairs. A feature that matters climbs from left to right;
    its last point is the top set of about 18,650."""

    import matplotlib.pyplot as plt

    cells = _cells(sets, model, kind)
    algorithms, per_row, columns = _grid(cells)
    part = sets[(sets["model"] == model) & (sets["kind"] == kind)]
    random = part[part["design"] == "random"].groupby(CELL_KEYS + ["pair_index"])["accuracy"].mean()
    picks = part[part["design"] == "quantile"].copy()
    picks[["feature", "q"]] = picks["detail"].str.split("@", expand=True)
    picks["q"] = picks["q"].astype(float)
    picks = picks.join(random.rename("random_mean"), on=CELL_KEYS + ["pair_index"])
    picks["gain"] = picks["accuracy"] - picks["random_mean"]
    quantiles = sorted(picks["q"].unique())
    position = {q: i for i, q in enumerate(quantiles)}
    with plt.rc_context({"font.size": 9}):
        fig, axes = plt.subplots(len(algorithms), columns, figsize=(3.8 * columns, 3.1 * len(algorithms) + 0.6),
                                 squeeze=False, sharey=True)
        for r, row_key in enumerate(algorithms):
            for c in range(columns):
                ax = axes[r, c]
                if c >= len(per_row[row_key]):
                    ax.axis("off")
                    continue
                a, d, rho, level, budget = per_row[row_key][c]
                cell = picks[(picks["algorithm"] == a) & (picks["mean_degree"] == d) & (picks["rho"] == rho)
                             & (picks["budget"] == budget)]
                _style(ax)
                ax.axhline(0, color=INK2, linewidth=0.8)
                for feature, color in zip(QUANTILE_LABELS, SERIES):
                    line = cell[cell["feature"] == feature].groupby("q")["gain"].mean()
                    ax.plot([position[q] for q in line.index], line.to_numpy(), color=color, linewidth=1.8,
                            marker="o", markersize=3.5, label=QUANTILE_LABELS[feature])
                ax.set_xticks(range(len(quantiles)), [f"{q:g}" for q in quantiles], rotation=90, fontsize=7)
                ax.set_title(cell_title(a, d, rho, budget), loc="left", fontsize=9.5)
                if c == 0:
                    ax.set_ylabel("gain over random sets")
                if r == len(algorithms) - 1:
                    ax.set_xlabel("quantile of the feature in the pool")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        height = fig.get_figheight()
        fig.suptitle(f"{MODEL_LABELS[model]}: pushing one feature to its extremes (sets from the pool at quantiles)",
                     x=0.01, y=0.995, ha="left", va="top", fontsize=11.5)
        fig.legend(handles, labels, loc="upper left", frameon=False, ncol=5, fontsize=9,
                   bbox_to_anchor=(0.005, 1 - 0.38 / height))
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.62 / height))
    return fig


def level_table(sets: pd.DataFrame, model: str = "sparse_er") -> pd.DataFrame:
    """The same cells at the budgets where random sets average 20%, 50% and 80% (ER: one pair at 20% and 80%, five
    at 50%): random sets' mean and part-way share, the score's Spearman on the saved B, and the gain of the best rule
    and of the best of 1,000 by score over random sets."""

    standard = {(a, d, float(r)) for a, d, r in _standard(model)}
    part = sets[(sets["model"] == model)]
    part = part[[(a, d, float(r)) in standard for a, d, r in
                 part[["algorithm", "mean_degree", "rho"]].itertuples(index=False)]]
    rows = []
    for key, cell in part.groupby(CELL_KEYS, sort=False):
        random = cell[cell["design"] == "random"]
        by_rule = cell[cell["design"] == "rule"].groupby("detail")["accuracy"].mean()
        rows.append({**dict(zip(CELL_KEYS, key)), "pairs": cell["pair_index"].nunique(),
                     "random mean": random["accuracy"].mean(),
                     "part-way share": random["accuracy_first"].between(*MIDDLE).mean(),
                     "score, one B": _within_pair(random, "score", "accuracy"),
                     "best rule gain": by_rule.max() - random["accuracy"].mean() if len(by_rule) else np.nan,
                     "best of 1,000 gain": cell.loc[cell["detail"] == "best_1000", "accuracy"].mean()
                     - random["accuracy"].mean()})
    return _order(pd.DataFrame(rows))


def feature_table(sets: pd.DataFrame, kind: str = "standard", outcome: str = "accuracy") -> pd.DataFrame:
    """Spearman correlation of each feature with test accuracy over random sets on the saved B (within pair,
    averaged over pairs): rows features (graph A first, then graph B), columns cells."""

    columns = {}
    random = sets[(sets["design"] == "random") & (sets["kind"] == kind)]
    for key in _order(random[CELL_KEYS].drop_duplicates()).itertuples(index=False):
        m, a, d, rho, level, budget = key
        part = random.merge(pd.DataFrame([dict(zip(CELL_KEYS, key))]), on=CELL_KEYS)
        name = f"{MODEL_LABELS[m]} · {LABELS[a]} · d{d:g}, ρ {rho:g}"
        columns[name] = {f: _within_pair(part, f, outcome) for group in FEATURE_GROUPS.values() for f in group
                         if f in part.columns}
    table = pd.DataFrame(columns)
    table.index = [FEATURE_LABELS.get(f, f) for f in table.index]
    return table


def plot_feature_heatmap(table: pd.DataFrame):
    """The feature table as a heatmap: blue, accuracy rises with the feature; red, it falls."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("red_grey_blue", (RED, NEUTRAL, BLUE))
    limit = max(0.3, float(np.nanmax(np.abs(table.to_numpy()))))
    with plt.rc_context({"font.size": 8.5}):
        fig, ax = plt.subplots(figsize=(0.42 * table.shape[1] + 4.5, 0.33 * table.shape[0] + 2.2))
        image = ax.imshow(table.to_numpy(dtype=float), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                value = table.iat[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}".replace("0.", ".").replace("-.", "-."), ha="center", va="center",
                            fontsize=6.5, color="white" if abs(value) > 0.6 * limit else INK)
        ax.set_xticks(range(table.shape[1]), table.columns, rotation=90)
        ax.set_yticks(range(table.shape[0]), table.index)
        start = 0
        for group, features in FEATURE_GROUPS.items():
            start += len(features)
            if start < table.shape[0]:
                ax.axhline(start - 0.5, color="white", linewidth=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01)
        bar.set_label("Spearman with test accuracy (random sets, one B)")
        ax.set_title("Which features track the match ratio? Rows: graph-A features, then graph-B features no rule "
                     "can see", loc="left", fontsize=10.5)
        fig.tight_layout()
    return fig


def first_step_table(sets: pd.DataFrame, kind: str = "standard") -> pd.DataFrame:
    """Direction 1b: first Hungarian step accuracy as a cheap stand-in for final accuracy (FAQ and GOAT). Spearman
    within pair over random sets: first step vs final, frozen score vs final, frozen score vs first step."""

    rows = []
    part = sets[(sets["design"] == "random") & (sets["kind"] == kind) & sets["algorithm"].isin(["faq", "goat"])]
    for key, cell in part.groupby(CELL_KEYS, sort=False):
        rows.append({**dict(zip(CELL_KEYS, key)), "kind": kind,
                     "first-step mean": cell["h0_accuracy"].mean(), "final mean": cell["accuracy"].mean(),
                     "first step vs final": _within_pair(cell, "h0_accuracy", "accuracy"),
                     "score vs final": _within_pair(cell, "score", "accuracy"),
                     "score vs first step": _within_pair(cell, "score", "h0_accuracy")})
    return _order(pd.DataFrame(rows))


def regularizer_table(sets: pd.DataFrame, models=MODELS) -> pd.DataFrame:
    """GOAT with the regulariser at 0.5x, 1x (the saved runs) and 2x, for the 100 redrawn sets of pair 0: mean test
    accuracy, and how well the ranking of the sets at 1x carries over (Spearman)."""

    rows = []
    for model in models:
        try:
            data = read(model, "regularizer")
        except FileNotFoundError:
            continue
        data = data[_ok(data)]
        for key, part in data.groupby(CELL_KEYS, sort=False):
            wide = part.pivot_table(index="set_id", columns="factor", values="accuracy_test")
            saved = sets.merge(pd.DataFrame([dict(zip(CELL_KEYS, key))]), on=CELL_KEYS)
            saved = saved[saved["pair_index"] == 0].set_index("set_id")["accuracy_first"]
            wide[1.0] = saved.reindex(wide.index)
            wide = wide.dropna()
            row = {**dict(zip(CELL_KEYS, key)), "sets": len(wide),
                   "regulariser": float(part["regularizer"].iloc[0] / part["factor"].iloc[0])}
            for factor in sorted(wide.columns):
                row[f"mean at {factor:g}x"] = wide[factor].mean()
            for factor in sorted(wide.columns):
                if factor != 1.0:
                    row[f"ranking {factor:g}x vs 1x"] = _spearman(wide[factor], wide[1.0])
            rows.append(row)
    return _order(pd.DataFrame(rows))


def stage2_levels(models=MODELS) -> pd.DataFrame:
    """Regime map stage 2 (every cell with a 20-80% rise): per cell, the gain of the best of round 1's rules (picked
    by its average gain) and of the best of 1,000 random sets by score over random sets, at the budgets where random
    sets average 20%, 50% and 80% (paired within pair, averaged over pairs; regime_map.gain_table)."""

    rows = []
    for model in models:
        try:
            data = RM.read_stage(model, 2)
        except FileNotFoundError:
            continue
        stage1 = RM.read_stage(model, 1)
        gains = RM.gain_table(data, stage1)
        for (m, d, rho, a), part in gains.groupby(["model", "mean degree", "rho", "algorithm"], sort=False):
            rules = part[part["method"].isin(RM.RULES[a])].dropna(subset=["gain"])
            screen = part[part["method"] == "best_1000"]
            if rules.empty:
                continue
            best = rules.loc[rules["gain"].idxmax()]
            for level in (0.2, 0.5, 0.8):
                column = f"gain at {level:.0%}"
                rows.append({"model": m, "algorithm": a, "mean degree": d, "rho": rho, "level": level,
                             "best rule": best["method"], "best rule gain": best.get(column, np.nan),
                             "best of 1,000 gain": screen[column].iloc[0] if len(screen) and column in screen
                             else np.nan})
    return pd.DataFrame(rows)


def stage2_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Median over cells of the best rule's and the best of 1,000's gain over random sets, per model, algorithm and
    level (from stage2_levels), with the number of cells."""

    medians = table.groupby(["model", "algorithm", "level"])[["best rule gain", "best of 1,000 gain"]].median()
    medians = medians.unstack("level")
    medians.columns = [f"{name}, {level:.0%}" for name, level in medians.columns]
    cells = table[["model", "algorithm", "mean degree", "rho"]].drop_duplicates().groupby(["model", "algorithm"]).size()
    medians.insert(0, "cells", cells)
    out = _order(medians.reset_index())
    out["model"] = out["model"].map(MODEL_LABELS)
    out["algorithm"] = out["algorithm"].map(LABELS)
    return out.set_index(["model", "algorithm"]).round(2)


def plot_stage2_levels(table: pd.DataFrame):
    """Where seed choice pays: median over cells (dots) and middle half (bars) of the best rule's gain over random sets
    at the 20%, 50% and 80% budgets, per algorithm and model."""

    import matplotlib.pyplot as plt

    colors = dict(zip(MODELS, SERIES))
    with plt.rc_context({"font.size": 9}):
        fig, axes = plt.subplots(1, len(ALGORITHMS), figsize=(4.4 * len(ALGORITHMS), 4.0), sharey=True)
        for ax, a in zip(axes, ALGORITHMS):
            _style(ax)
            ax.axhline(0, color=INK2, linewidth=0.8)
            for i, model in enumerate(MODELS):
                part = table[(table["model"] == model) & (table["algorithm"] == a)]
                if part.empty:
                    continue
                stats = part.groupby("level")["best rule gain"].quantile([0.25, 0.5, 0.75]).unstack()
                x = np.array([20, 50, 80])[: len(stats)] + (i - 1) * 2.5
                ax.vlines(x, stats[0.25], stats[0.75], color=colors[model], linewidth=2, alpha=0.4)
                ax.plot(x, stats[0.5], color=colors[model], linewidth=2, marker="o", markersize=6,
                        markeredgecolor="white", label=f"{MODEL_LABELS[model]} ({part['mean degree'].size // 3} cells)")
            ax.set_xticks([20, 50, 80], ["20%", "50%", "80%"])
            ax.set_xlabel("budget where random sets average ...")
            ax.set_title(LABELS[a], loc="left", fontsize=10)
            ax.legend(frameon=False, fontsize=8)
        axes[0].set_ylabel("best rule minus random sets\n(test accuracy; median and middle half of cells)")
        fig.suptitle("Where seed choice pays: regime map stage 2 (best rule picked per cell, so slightly optimistic)",
                     x=0.01, ha="left", fontsize=11.5)
        fig.tight_layout()
    return fig


def _parse_cells(items) -> list[tuple[str, float, float]]:
    cells = []
    for item in items or []:
        algorithm, degree, rho = item.split(":")
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm in --cells {item!r}")
        degree = float(degree)
        cells.append((algorithm, int(degree) if degree.is_integer() else degree, float(rho)))
    return cells


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="sparse_er", choices=MODELS)
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=ALGORITHMS)
    parser.add_argument("--levels", nargs="+", type=float, default=[LEVEL],
                        help="random-set accuracy level(s) that set each cell's budget (default 0.5)")
    parser.add_argument("--cells", nargs="+", help="extra cells as algorithm:mean_degree:rho, e.g. faq:16:0.2")
    parser.add_argument("--pairs", type=int, help=f"pairs per cell (default {PAIRS})")
    parser.add_argument("--no-extras", action="store_true", help="skip the B redraws and GOAT regulariser subsets")
    parser.add_argument("--budget", type=int, help="use this many seeds in every cell instead of the regime map's "
                        "budget (for tests; use a separate LANDSCAPE_OUTPUT_DIR)")
    parser.add_argument("--smoke", action="store_true", help="tiny design for a quick test (use a separate "
                        "LANDSCAPE_OUTPUT_DIR)")
    parser.add_argument("--status", action="store_true", help="only show how far it is")
    parser.add_argument("--check-pairs", action="store_true", help="only check that every cell can draw pairs")
    parser.add_argument("--summary", action="store_true", help="only print a first summary of the results so far")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--blas-threads", type=int, default=3)
    args = parser.parse_args(argv)
    extra = _parse_cells(args.cells)
    pd.set_option("display.width", 200)
    if args.check_pairs:
        print(check_pairs(args.model, args.algorithms, extra).to_string(index=False))
        return
    if args.summary:
        pd.set_option("display.max_columns", 60)
        print(summary(args.model).round(3).to_string())
        return
    if args.status:
        print(status(args.model, args.algorithms, args.levels, extra, args.pairs, not args.no_extras,
                     args.smoke, args.budget).to_string())
        return
    finished = run(args.model, args.algorithms, args.levels, extra, args.pairs, not args.no_extras, args.jobs,
                   args.blas_threads, args.smoke, args.budget)
    if not finished:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
