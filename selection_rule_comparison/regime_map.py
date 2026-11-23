"""Regime map: where do single runs end all-or-nothing, where are they graded, and where does seed choice matter?
(6 Nov 2026)

Why. Everything so far used calibrated settings where random seed sets succeed about half the time and single runs
end all-or-nothing. Seed choice may matter elsewhere too, including settings where the match ratio is spread out
(graded_outcomes.ipynb, Hernan's sparse percolation runs). This maps percolation, FAQ and GOAT over edge correlation,
density and budget before any large data collection.

Grid. 600 vertices; mean degree in DEGREES (8, 16, 32 for ER; 10, 16, 32 for Chung-Lu, where mean degree 8 rarely
gives a connected pair); edge correlation rho in RHOS. ER first, then Chung-Lu, PAPER later (one model at a time).
All three algorithms run on the same structure-checked pairs (both graphs connected, no automorphisms) and, in
stage 1, on the same random seed sets.

Stage 1: what single runs look like (random sets only). Per setting and algorithm, STAGE1_PAIRS pairs, budgets
3 to 300 seeds (each about 1.5x the last), STAGE1_SETS random sets per budget, one run each. Early cuts: when the first
EARLY_SETS runs at a budget all end below FLOOR or all above CEILING, the rest are skipped; climbing stops after two
budgets in a row average above CEILING. At the budget whose mean is closest to 0.5, FAQ and GOAT run REPEAT_SETS sets
REPEATS more times with other solver seeds (solver luck), GOAT runs REG_SETS sets with its regulariser scaled by each
of REG_FACTORS, and the core features of SPREAD_POOL random sets give their spread (what a best-of-N screen could use).

Stage 2: does seed choice matter? For each setting and algorithm whose random-set curve rises inside the budget range:
the budgets where random sets average 20%, 50% and 80% (from stage 1), on STAGE2_PAIRS pairs (the stage 1 pairs plus
new ones). Per pair and budget: RANDOM_SETS random sets (one run each); round 1's rules for the algorithm, never
seeding a test vertex; best of 100 / 1,000 by the frozen score and best of 1,000 by degree, from POOL random sets.
FAQ runs each rule or best-of set RUNS["faq"] times; GOAT and percolation give the same result for the same seeds and
run once.

Outcome. Accuracy on N_TEST fixed test vertices per pair that are never seeds, so it means the same thing at every
budget and for every method. Non-seed accuracy and round 1's success flag (non-seed accuracy >= 0.9 for SGM, >= 0.5
for percolation) are recorded too.

Run from the repository root (ds environment), or from regime_map.ipynb:

    python -m selection_rule_comparison.regime_map --stage 1 --model sparse_er
    python -m selection_rule_comparison.regime_map --stage 2 --model sparse_er
    python -m selection_rule_comparison.regime_map --status --model sparse_er

Results go to data/regime_map/<model>/ (or REGIME_MAP_OUTPUT_DIR). Rows are written run by run, so a stopped run
resumes where it stopped (Ctrl-C stops the workers too).
"""

from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
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

from . import graded_outcomes as G
from . import run_comparison as rc
from . import settings as S
from .strategies import STRATEGY_CODES, GraphA

EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("REGIME_MAP_OUTPUT_DIR") or EXPERIMENT_DIR / "data" / "regime_map")

ALGORITHMS = ("percolation", "faq", "goat")
MODELS = ("sparse_er", "chung_lu", "paper_pa")
DEGREES = {"sparse_er": (8, 16, 32), "chung_lu": (10, 16, 32), "paper_pa": (8, 16, 32)}
RHOS = (0.2, 0.3, 0.4, 0.5, 0.7, 0.9)
BUDGETS = (3, 5, 8, 12, 18, 27, 40, 60, 90, 135, 200, 300)
N_VERTICES = 600
N_TEST = 120              # fixed test vertices per pair (20%), never seeds

STAGE1_PAIRS = 2
STAGE1_SETS = 8           # random sets per budget ...
EARLY_SETS = 4            # ... cut to this many when they all end below FLOOR or all above CEILING
FLOOR, CEILING = 0.05, 0.95
REPEAT_SETS, REPEATS = 5, 2            # FAQ and GOAT, middle budget: solver luck
REG_SETS, REG_FACTORS = 8, (0.5, 2.0)  # GOAT, middle budget: regulariser halved and doubled
SPREAD_POOL = 1_000                    # random sets whose features give the spread at the middle budget

STAGE2_PAIRS = 4
STAGE2_LEVELS = (0.2, 0.5, 0.8)        # random-set mean accuracy at the stage 2 budgets
RANDOM_SETS = 10
POOL = 1_000
RUNS = {"percolation": 1, "faq": 3, "goat": 1}
RULES = {a: tuple(s for s in S.STRATEGIES["main"][a] if s != "random") for a in ALGORITHMS}
SCREENS = ("best_100", "best_1000", "degree_1000")
METHODS = ("random", "top_degree", "top_harmonic", "clustered", "spread", "distinguishing",
           "hub_free_distinguishing", *SCREENS)

MIDDLE = (0.2, 0.8)       # a run "ends part-way" if its test accuracy is in this range
ALL_OR_NOTHING = (0.1, 0.9)   # ... and "all-or-nothing" if at or below the first or at or above the second
SECONDS_PER_RUN = {"percolation": 0.5, "faq": 2.5, "goat": 3.5}   # rough wall time per run with 3 jobs
STAGE1_RUNS_GUESS = 60    # typical runs per stage 1 pair, for time estimates only

PAIR_SEED_BASE, SAMPLING_SEED_BASE, SOLVER_SEED_BASE, TEST_SEED_BASE = 20261170, 20261171, 20261172, 20261173
PURPOSE = {"stage1_sets": 1, "spread": 2, "stage2_pool": 3, "rules": 4}

CORE = MP.CORE_FEATURES
LABELS, MODEL_LABELS = MP.ALGORITHM_LABELS, MP.MODEL_LABELS
METHOD_LABELS = {**G.METHOD_LABELS, "top_harmonic": "top harmonic"}
METHOD_SHORT = {"top_degree": "top deg.", "top_harmonic": "top harm.", "clustered": "clustered", "spread": "spread",
                "distinguishing": "disting.", "hub_free_distinguishing": "hub-free", "best_100": "best 100",
                "best_1000": "best 1,000", "degree_1000": "deg. 1,000"}
METHOD_STYLE = {**G.METHOD_STYLE, "top_harmonic": dict(color="#52514e", marker="D")}

# validated palette (dataviz): slot 1 blue, slot 2 orange, red pole, neutrals, blue sequential ramp
BLUE, ORANGE, RED, INK, INK2, GRID, BAND, NEUTRAL = (
    "#2a78d6", "#eb6834", "#e34948", "#0b0b0b", "#52514e", "#e6e6e6", "#f0f0ef", "#f0efec")
BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _clock() -> str:
    return datetime.datetime.now().strftime("%a %H:%M")


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Settings, weights and pairs
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def chung_lu_scale(degree: float, alpha: float = 2.0, floor: float = 0.0, n: int = N_VERTICES) -> float:
    """Weight scale giving this expected mean degree in utils.Graphs' Chung-Lu model (same weights, cap and floor)."""

    from utils.Graphs import pareto_quantile_weights

    base = pareto_quantile_weights(n, alpha)

    def expected(scale):
        weights = scale * base
        weights = np.minimum(weights, np.sqrt(weights.sum()))
        probability = np.minimum(np.outer(weights, weights) / weights.sum() + floor, 1.0)
        np.fill_diagonal(probability, 0.0)
        return probability.sum() / n

    low, high = 0.01, 1_000.0
    for _ in range(80):
        middle = np.sqrt(low * high)
        low, high = (middle, high) if expected(middle) < degree else (low, middle)
    return float(np.sqrt(low * high))


def graph_config(model: str, degree: float, rho: float):
    """Graph settings for one grid cell (graph fields only; the solver settings are round 1's)."""

    base = replace(S.graph_config("faq", model), n_graph_pairs=1, rho=float(rho))
    if model == "sparse_er":
        config = replace(base, er_probability=degree / (N_VERTICES - 1))
    elif model == "chung_lu":
        config = replace(base, chung_lu_weight_scale=chung_lu_scale(
            float(degree), float(base.chung_lu_alpha), float(base.chung_lu_probability_floor or 0.0)))
    elif model == "paper_pa":   # a shared preferential-attachment tree (mean degree about 2) plus correlated ER edges
        config = replace(base, paper_pa_probability=(degree - 2) / (N_VERTICES - 1))
    else:
        raise ValueError(f"Unknown model: {model}")
    if config.n_vertices != N_VERTICES:
        raise ValueError(f"{model}: expected {N_VERTICES} vertices, the preset has {config.n_vertices}")
    return config


def load_weights(model: str) -> dict:
    """The frozen pooled weights (per SD) for this model, fitted for success at round 1's settings."""

    saved = _read_json(G.WEIGHTS_PATH)
    return {a: {f: float(saved[f"{a}/{model}"]["weights_per_sd"][f]) for f in CORE} for a in ALGORITHMS}


def cell_name(degree: float, rho: float) -> str:
    return f"d{int(degree):02d}_rho{rho:.2f}"


def first_seed(model: str, degree: float, rho: float, pair_index: int) -> int:
    seed = rc._seed(PAIR_SEED_BASE, S.MODEL_CODES[model], int(degree), int(round(100 * rho)), pair_index)
    return seed % (2**32 - 100_000)


def test_vertices(pair_seed: int, n: int, n_test: int) -> np.ndarray:
    rng = np.random.default_rng([TEST_SEED_BASE, int(pair_seed)])
    return np.sort(rng.choice(n, n_test, replace=False))


def load_pair(model: str, degree: float, rho: float, pair_index: int, design: dict):
    """The structure-checked pair with this index (shared by all algorithms), its test vertices and details."""

    config = graph_config(model, degree, rho)
    excluded = set(design["excluded_pair_seeds"])
    start = cursor = first_seed(model, degree, rho, pair_index)
    rejected = 0
    while True:
        try:
            seed, A, B, truth, _report, n_rejected = first_passing_pair(
                lambda s: generate_graph_pair(config, s), cursor, max_tries=500)
        except RuntimeError as error:
            raise RuntimeError(f"{MODEL_LABELS[model]}, mean degree {degree}, rho {rho:.2f}: no pair passed the "
                               "structure check (connected, no automorphisms) in 500 draws; this setting is too "
                               "sparse for it.") from error
        rejected += int(n_rejected)
        if int(seed) not in excluded:
            break
        cursor = int(seed) + 1
    n = len(A)
    test = test_vertices(seed, n, design["n_test"])
    candidates = np.setdiff1d(np.arange(n), test)
    aligned = B[np.ix_(truth, truth)]
    info = {"model": model, "mean_degree": degree, "rho": rho, "pair_index": pair_index, "pair_seed": int(seed),
            "first_seed_tried": int(start), "rejected_draws": rejected, "hash_a": rc._hash(A), "hash_b": rc._hash(B),
            "mean_degree_a": float(A.sum(axis=1).mean()), "max_degree_a": int(A.sum(axis=1).max()),
            "kept_edge_fraction": float(((A > 0) & (aligned > 0)).sum() / max((A > 0).sum(), 1)),
            "graph_config": asdict(config), "test_vertices": [int(v) for v in test]}
    return int(seed), A, B, truth, test, candidates, info


def _fixed(model: str) -> dict:
    return json.loads(json.dumps({
        "version": 1, "model": model, "n_vertices": N_VERTICES, "n_test": N_TEST, "budgets": list(BUDGETS),
        "stage1": {"sets": STAGE1_SETS, "early_sets": EARLY_SETS, "floor": FLOOR, "ceiling": CEILING,
                   "repeat_sets": REPEAT_SETS, "repeats": REPEATS, "reg_sets": REG_SETS,
                   "reg_factors": list(REG_FACTORS), "spread_pool": SPREAD_POOL},
        "stage2": {"levels": list(STAGE2_LEVELS), "random_sets": RANDOM_SETS, "pool": POOL, "runs": RUNS,
                   "rules": {a: list(RULES[a]) for a in ALGORITHMS}, "screens": list(SCREENS)},
        "weights": load_weights(model), "weights_path": str(G.WEIGHTS_PATH.relative_to(G.REPO_ROOT)),
        "seed_bases": {"pair": PAIR_SEED_BASE, "sampling": SAMPLING_SEED_BASE, "solver": SOLVER_SEED_BASE,
                       "test_vertices": TEST_SEED_BASE},
        "solver": {"sgm": S.SGM, "percolation": S.PERCOLATION, "goat_regularizer": S.GOAT_REGULARIZER[model]},
        "success_threshold": S.SUCCESS_THRESHOLD, "uniqueness_cap": {a: MP.UNIQUENESS_CAP[a] for a in ALGORITHMS},
        "pool_factor": S.POOL_FACTOR, "hub_count": S.HUB_COUNT,
    }))


def prepare(model: str = "sparse_er", blas_threads: int = 3) -> dict:
    """The design every job of this model uses, frozen in <model>/settings.json once results exist. More pairs,
    degrees or rhos later are fine; other changes need a new folder."""

    path = OUTPUT_DIR / model / "settings.json"
    fixed = _fixed(model)
    if path.exists():
        saved = _read_json(path)
        differs = [key for key in fixed if saved.get(key) != fixed[key]]
        if differs:
            raise ValueError(f"The saved design in {path} differs ({', '.join(differs)}). Set "
                             "REGIME_MAP_OUTPUT_DIR to a new folder to keep the results apart.")
        excluded = saved["excluded_pair_seeds"]
    else:
        excluded = sorted(G.known_pair_seeds() | G._graded_pair_seeds())
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {**fixed, "excluded_pair_seeds": excluded, "written_at": _now(),
                          "environment": collect_environment(blas_threads)})
    return {**fixed, "excluded_pair_seeds": excluded, "blas_threads": blas_threads}


# --------------------------------------------------------------------------
# One run (percolation, FAQ or GOAT), scored on the test vertices
# --------------------------------------------------------------------------

_PERCOLATION: dict = {}


def _percolation():
    """experiment_common.percolation with its matcher wrapped so the final matching is kept."""

    from experiment_common import percolation

    if not getattr(percolation.graph_match_percolation, "_keeps_matching", False):
        original = percolation.graph_match_percolation

        def graph_match_percolation(*args, **kwargs):
            result = original(*args, **kwargs)
            _PERCOLATION["matching"] = np.asarray(result)
            return result

        graph_match_percolation._keeps_matching = True
        percolation.graph_match_percolation = graph_match_percolation
    return percolation


def _failed(error) -> dict:
    return {"accuracy_test": np.nan, "accuracy": np.nan, "success": np.nan, "h0_accuracy": np.nan,
            "objective": np.nan, "n_iter": np.nan, "converged": False, "runtime_seconds": np.nan,
            "matched_fraction": np.nan, "error": f"{type(error).__name__}: {error}"[:300]}


def run_one(algorithm: str, model: str, pair, seeds: np.ndarray, test: np.ndarray, solver_seed: int,
            regularizer: float | None = None) -> dict:
    A, B, truth = pair
    seeds = np.asarray(seeds, dtype=int)
    threshold = S.SUCCESS_THRESHOLD[algorithm]
    if algorithm == "percolation":
        module = _percolation()
        _PERCOLATION.clear()
        try:
            out = module.run_percolation_once(A, B, truth, seeds, S.PERCOLATION["algorithm_seed"],
                                              S.PERCOLATION["threshold"], S.PERCOLATION["expand_when_stuck"])
        except Exception as error:  # noqa: BLE001 - a failed run is recorded, not fatal
            return _failed(error)
        matching = _PERCOLATION.get("matching")
        accuracy = float(out["final_unseeded_accuracy"])
        return {"accuracy_test": float(np.mean(matching[test] == truth[test])) if matching is not None else np.nan,
                "accuracy": accuracy, "success": float(accuracy >= threshold), "h0_accuracy": np.nan,
                "objective": np.nan, "n_iter": np.nan, "converged": bool(out["percolation_completed"]),
                "runtime_seconds": float(out["percolation_runtime_seconds"]),
                "matched_fraction": float(out["percolation_matched_unseeded_fraction"]),
                "error": "" if matching is not None else "final matching not captured"}
    sgm = G._sgm()
    G._CAPTURE.clear()
    regularizer = S.GOAT_REGULARIZER[model] if regularizer is None else regularizer
    try:
        out = sgm.run_sgm_once(A, B, truth, seeds, solver_seed, S.SGM["max_iter"], S.SGM["tol"],
                               transport=algorithm == "goat", transport_regularizer=regularizer,
                               transport_tol=S.SGM["transport_tol"], transport_max_iter=S.SGM["transport_max_iter"])
    except Exception as error:  # noqa: BLE001
        return _failed(error)
    accuracy = float(out["final_unseeded_accuracy"])
    matched = "indices_A" in G._CAPTURE
    test_accuracy = np.nan
    if matched:
        matching = np.full(len(truth), -1, dtype=np.int64)
        matching[G._CAPTURE["indices_A"]] = G._CAPTURE["indices_B"]
        test_accuracy = float(np.mean(matching[test] == truth[test]))
    return {"accuracy_test": test_accuracy, "accuracy": accuracy, "success": float(accuracy >= threshold),
            "h0_accuracy": float(out["h0_accuracy"]), "objective": float(out["sgm_objective_score"]),
            "n_iter": int(out["sgm_n_iter"]), "converged": bool(out["sgm_converged"]),
            "runtime_seconds": float(out["sgm_runtime_seconds"]), "matched_fraction": np.nan,
            "error": "" if matched else "final matching not captured"}


def _solver_seed(pair_seed: int, k: int, algorithm: str, stage: int, set_code: int, repeat: int) -> int:
    return rc._seed(SOLVER_SEED_BASE, pair_seed, k, MP.ALGORITHM_CODES[algorithm], stage, set_code, repeat)


def _saved_outcomes(path: Path) -> dict[int, float]:
    """run_id -> test accuracy for the rows already written (a row cut off by an interruption is removed first)."""

    if not path.exists() or path.stat().st_size == 0:
        return {}
    G._completed_ids(path, "run_id")   # drops a cut-off last row
    saved = pd.read_csv(path, usecols=["run_id", "accuracy_test"])
    return dict(zip(saved["run_id"].astype(int), saved["accuracy_test"].astype(float)))


def _check_saved(path: Path, info: dict, name: str, keys=("hash_a", "hash_b")) -> None:
    if path.exists():
        saved = _read_json(path)
        for key in keys:
            if key in saved and saved[key] != info[key]:
                raise RuntimeError(f"{name}: the saved runs come from a different pair or plan ({key} differs). "
                                   "Set REGIME_MAP_OUTPUT_DIR to a new folder to keep them apart.")


def draw_sets(rng: np.random.Generator, candidates: np.ndarray, k: int, size: int) -> np.ndarray:
    return G.draw_sets(rng, candidates, k, size)


def _rng(pair_seed: int, purpose: str, value: int) -> np.random.Generator:
    return np.random.default_rng([SAMPLING_SEED_BASE, int(pair_seed), PURPOSE[purpose], int(value)])


# --------------------------------------------------------------------------
# Stage 1: budget ladder with random sets
# --------------------------------------------------------------------------

def stage_dir(model: str, degree: float, rho: float, algorithm: str, stage: int, pair_index: int) -> Path:
    return OUTPUT_DIR / model / cell_name(degree, rho) / algorithm / f"stage{stage}" / f"pair_{pair_index:03d}"


def describe(task) -> str:
    model, degree, rho, algorithm, pair_index = task[:5]
    stage = 1 if len(task) == 5 else 2
    return (f"stage {stage} {LABELS[algorithm]} {MODEL_LABELS[model]}, mean degree {degree}, rho {rho:.2f}, "
            f"pair {pair_index}")


def _stage1_id(budget_index: int, set_index: int, repeat: int = 0, variant: int = 0) -> int:
    return 100_000 * variant + 1_000 * budget_index + 10 * set_index + repeat


def _finite(values) -> list[float]:
    return [float(v) for v in values if v is not None and np.isfinite(v)]


def _number(value) -> bool:
    """True for a finite number (not None or NaN)."""

    try:
        return value is not None and bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def middle_budget(means: dict) -> int | None:
    """The budget whose mean test accuracy is closest to 0.5 (the smaller one on a tie)."""

    usable = {k: m for k, m in means.items() if np.isfinite(m)}
    if not usable:
        return None
    return min(usable, key=lambda k: (abs(usable[k] - 0.5), k))


def feature_spread(A: np.ndarray, graph: GraphA, algorithm: str, candidates: np.ndarray, pair_seed: int, k: int,
                   design: dict) -> dict:
    """How much random sets of k seeds differ: core features' mean, SD and coefficient of variation, and the SD of
    the frozen score (in units of its weights)."""

    pool = draw_sets(_rng(pair_seed, "spread", k), candidates, k, design["stage1"]["spread_pool"])
    features, _, stats, score, _ = G._scored(A, algorithm, pool, design["weights"][algorithm], graph=graph)
    out = {"budget": int(k), "sets": len(pool), "score_sd": float(np.std(score, ddof=1))}
    for f in CORE:
        mean, sd = stats[f]["mean"], stats[f]["sd"]
        out[f] = {"mean": mean, "sd": sd, "cv": float(sd / abs(mean)) if mean else np.nan}
    return out


def stage1_job(task, design: dict) -> str:
    model, degree, rho, algorithm, i = task
    folder = stage_dir(model, degree, rho, algorithm, 1, i)
    runs_path, info_path = folder / "runs.csv", folder / "pair.json"
    if info_path.exists() and _read_json(info_path).get("status") == "complete":
        return f"{describe(task)}: already done"
    started = time.perf_counter()
    s1 = design["stage1"]
    with thread_limit(design["blas_threads"]):
        seed, A, B, truth, test, candidates, info = load_pair(model, degree, rho, i, design)
        folder.mkdir(parents=True, exist_ok=True)
        _check_saved(info_path, info, describe(task))
        write_json(info_path, {**info, "algorithm": algorithm, "status": "running"})
        saved = _saved_outcomes(runs_path)
        pair, written = (A, B, truth), [0]
        rows = G._Rows(runs_path)

        def run(b, k, j, seeds, repeat=0, variant=0, factor=None):
            run_id = _stage1_id(b, j, repeat, variant)
            if run_id in saved:
                return saved[run_id]
            regularizer = None if factor is None else design["solver"]["goat_regularizer"] * factor
            solver_seed = _solver_seed(seed, k, algorithm, 1, j, repeat + 10 * variant)
            out = run_one(algorithm, model, pair, seeds, test, solver_seed, regularizer)
            rows.write({"model": model, "mean_degree": degree, "rho": rho, "algorithm": algorithm, "stage": 1,
                        "pair_index": i, "pair_seed": seed, "budget": k, "set_index": j, "repeat": repeat,
                        "variant": variant, "regularizer": regularizer if regularizer is not None else np.nan,
                        "run_id": run_id, "seed_vertices": json.dumps([int(v) for v in seeds]),
                        "solver_seed": solver_seed, **out})
            saved[run_id] = out["accuracy_test"]
            written[0] += 1
            return out["accuracy_test"]

        means, streak, spread, middle = {}, 0, None, None
        try:
            for b, k in enumerate(design["budgets"]):
                if k > len(candidates):
                    break
                sets = draw_sets(_rng(seed, "stage1_sets", k), candidates, k, s1["sets"])
                values = [run(b, k, j, sets[j]) for j in range(s1["early_sets"])]
                finite = _finite(values)
                decided = len(finite) == len(values) and (all(v <= s1["floor"] for v in finite)
                                                          or all(v >= s1["ceiling"] for v in finite))
                if not decided:
                    values += [run(b, k, j, sets[j]) for j in range(s1["early_sets"], s1["sets"])]
                finite = _finite(values)
                means[k] = float(np.mean(finite)) if finite else np.nan
                streak = streak + 1 if finite and means[k] >= s1["ceiling"] else 0
                if streak >= 2:
                    break
            middle = middle_budget(means)
            if middle is not None:
                b = design["budgets"].index(middle)
                sets = draw_sets(_rng(seed, "stage1_sets", middle), candidates, middle, s1["sets"])
                n_run = sum(1 for j in range(s1["sets"]) if _stage1_id(b, j) in saved)
                if algorithm in ("faq", "goat"):
                    for j in range(min(s1["repeat_sets"], n_run)):
                        for r in range(1, s1["repeats"] + 1):
                            run(b, middle, j, sets[j], repeat=r)
                if algorithm == "goat":
                    for v, factor in enumerate(s1["reg_factors"], 1):
                        for j in range(min(s1["reg_sets"], n_run)):
                            run(b, middle, j, sets[j], variant=v, factor=factor)
                spread = feature_spread(A, GraphA(A), algorithm, candidates, seed, middle, design)
        finally:
            rows.close()
        write_json(info_path, {**info, "algorithm": algorithm, "status": "complete", "middle_budget": middle,
                               "budget_means": {str(k): m for k, m in means.items()}, "spread": spread,
                               "finished_at": _now()})
    return (f"{describe(task)}: {written[0]} runs in {(time.perf_counter() - started) / 60:.1f} min, "
            f"middle budget {middle}")


def _stage1_complete(model, degree, rho, algorithm, i) -> bool:
    path = stage_dir(model, degree, rho, algorithm, 1, i) / "pair.json"
    return path.exists() and _read_json(path).get("status") == "complete"


def _grid(model: str, algorithms, degrees, rhos):
    degrees = DEGREES[model] if degrees is None else degrees
    return [(d, float(rho), a) for d in degrees for rho in rhos for a in algorithms]


def run_stage1(model: str = "sparse_er", algorithms=ALGORITHMS, degrees=None, rhos=RHOS,
               n_pairs: int = STAGE1_PAIRS, jobs: int = 3, blas_threads: int = 3) -> bool:
    """Run the budget ladder with random sets for every grid cell and algorithm. Returns False if stopped."""

    design = prepare(model, blas_threads)
    tasks = [(model, d, rho, a, i) for i in range(n_pairs) for d, rho, a in _grid(model, algorithms, degrees, rhos)
             if not _stage1_complete(model, d, rho, a, i)]
    if not tasks:
        print(f"Stage 1 is complete for {MODEL_LABELS[model]}. Results are in {OUTPUT_DIR / model}")
        return True

    def seconds(task):
        return STAGE1_RUNS_GUESS * SECONDS_PER_RUN[task[3]]

    tasks.sort(key=seconds, reverse=True)
    print(f"{_clock()}  Stage 1, {MODEL_LABELS[model]}: {len(tasks)} pair runs, {jobs} at a time, about "
          f"{sum(seconds(t) for t in tasks) / jobs / 60:.0f} min (a rough guess). One line per finished pair.",
          flush=True)
    return _run(tasks, stage1_job, design, seconds, jobs)


# --------------------------------------------------------------------------
# Stage 2: rules and best of N at the budgets where random sets reach 20%, 50% and 80%
# --------------------------------------------------------------------------

def _ordering(rule: str, graph: GraphA, k_max: int, rng: np.random.Generator, exclude) -> np.ndarray:
    if rule == "top_harmonic":
        order = np.lexsort((rng.random(graph.n), -graph.harmonic))
        return order[~np.isin(order, np.asarray(exclude, dtype=int))][:k_max]
    return G.rule_ordering(rule, graph, k_max, rng, exclude)


def crossing(budgets, means, level: float):
    """Budget where a curve first reaches `level` (log-linear interpolation on its running maximum), or None when it
    starts above the level or never reaches it."""

    budgets, means = np.asarray(budgets, dtype=float), np.asarray(means, dtype=float)
    keep = np.isfinite(means)
    budgets, means = budgets[keep], np.maximum.accumulate(means[keep]) if keep.any() else means[keep]
    if len(budgets) == 0 or means[0] >= level or means[-1] < level:
        return None
    j = int(np.argmax(means >= level))
    low, high = means[j - 1], means[j]
    t = 1.0 if high == low else (level - low) / (high - low)
    return float(np.exp(np.log(budgets[j - 1]) + t * (np.log(budgets[j]) - np.log(budgets[j - 1]))))


def cell_curve(runs: pd.DataFrame) -> pd.Series:
    """Mean test accuracy of random sets per budget, averaged over pairs. A pair that stopped climbing because it
    saturated keeps its last mean at the larger budgets."""

    main = runs[(runs["repeat"] == 0) & (runs["variant"] == 0)]
    table = main.groupby(["pair_index", "budget"])["accuracy_test"].mean().unstack("budget").sort_index(axis=1)
    return table.ffill(axis=1).mean(axis=0)


def stage2_budgets(model: str, degree: float, rho: float, algorithm: str, design: dict,
                   n_pairs: int = STAGE1_PAIRS):
    """[(level, budget), ...] for stage 2, frozen in stage2/budgets.json; [] when the curve doesn't rise in range,
    None while stage 1 is unfinished."""

    path = OUTPUT_DIR / model / cell_name(degree, rho) / algorithm / "stage2" / "budgets.json"
    if path.exists():
        return [tuple(item) for item in _read_json(path)["budgets"]]
    if not all(_stage1_complete(model, degree, rho, algorithm, i) for i in range(n_pairs)):
        return None
    frames = [pd.read_csv(stage_dir(model, degree, rho, algorithm, 1, i) / "runs.csv") for i in range(n_pairs)]
    curve = cell_curve(pd.concat(frames, ignore_index=True))
    candidates = design["n_vertices"] - design["n_test"]
    chosen, seen = [], set()
    for level in design["stage2"]["levels"]:
        k = crossing(curve.index, curve.to_numpy(), level)
        if k is None:
            continue
        k = int(min(max(round(k), 3), min(max(design["budgets"]), candidates)))
        if k not in seen:
            chosen.append((float(level), k))
            seen.add(k)
    reason = "" if len(chosen) >= 2 else "random sets' accuracy doesn't rise across 20-80% inside the budget range"
    chosen = chosen if len(chosen) >= 2 else []
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"budgets": chosen, "curve": {str(int(k)): float(v) for k, v in curve.items()},
                      "reason": reason, "written_at": _now()})
    return chosen


def stage2_plan(algorithm: str, pair_seed: int, A: np.ndarray, graph: GraphA, test: np.ndarray,
                candidates: np.ndarray, budgets, design: dict):
    """Every run on one pair, in order: per budget, the random sets, round 1's rules and the best-of sets, each with
    its core features, z-scores and frozen score against the budget's pool of random sets."""

    s2 = design["stage2"]
    weights, rules, repeats = design["weights"][algorithm], s2["rules"][algorithm], s2["runs"][algorithm]
    k_max = max(k for _, k in budgets)
    orderings = {rule: _ordering(rule, graph, k_max, _rng(pair_seed, "rules", STRATEGY_CODES[rule]), test)
                 for rule in rules}
    plan, pool_stats = [], {}
    for level, k in budgets:
        pool = draw_sets(_rng(pair_seed, "stage2_pool", k), candidates, k, s2["pool"])
        features, _, stats, score, _ = G._scored(A, algorithm, pool, weights, graph=graph)
        pool_stats[str(k)] = stats

        def add(method, code, seeds, raw, pool_index, runs):
            z = {f: (raw[f] - stats[f]["mean"]) / stats[f]["sd"] if stats[f]["sd"] > 0 else 0.0 for f in CORE}
            value = float(sum(weights[f] * z[f] for f in CORE))
            for repeat in range(runs):
                plan.append({"level": level, "budget": k, "method": method, "set_code": code,
                             "pool_index": pool_index, "repeat": repeat, "seeds": np.asarray(seeds, dtype=np.int64),
                             **{f: float(raw[f]) for f in CORE}, **{f"z_{f}": float(z[f]) for f in CORE},
                             "score": value, "score_percentile": float(100 * np.mean(score <= value + 1e-12))})

        for j in range(s2["random_sets"]):
            add("random", j, pool[j], {f: features[f][j] for f in CORE}, j, 1)
        rule_sets = np.sort(np.stack([orderings[rule][:k] for rule in rules]), axis=1)
        rule_features = G._core_features(graph, algorithm, rule_sets)
        for r, rule in enumerate(rules):
            add(rule, 1000 + STRATEGY_CODES[rule], rule_sets[r], {f: rule_features[f][r] for f in CORE}, -1,
                repeats)
        picks = {"best_100": int(np.argmax(score[:100])), "best_1000": int(np.argmax(score)),
                 "degree_1000": int(np.argmax(features["degree"]))}
        for s, name in enumerate(s2["screens"]):
            add(name, 2000 + s, pool[picks[name]], {f: features[f][picks[name]] for f in CORE}, picks[name],
                repeats)
    return plan, pool_stats


def stage2_job(task, design: dict) -> str:
    model, degree, rho, algorithm, i, budgets = task
    folder = stage_dir(model, degree, rho, algorithm, 2, i)
    runs_path, info_path = folder / "runs.csv", folder / "pair.json"
    started = time.perf_counter()
    with thread_limit(design["blas_threads"]):
        seed, A, B, truth, test, candidates, info = load_pair(model, degree, rho, i, design)
        graph = GraphA(A)
        plan, stats = stage2_plan(algorithm, seed, A, graph, test, candidates, budgets, design)
        done = G._completed_ids(runs_path, "run_id")
        if len(done) >= len(plan):
            return f"{describe(task)}: already done"
        digest = hashlib.sha256(b"".join(item["seeds"].tobytes() for item in plan) + test.tobytes()).hexdigest()[:16]
        info = {**info, "algorithm": algorithm, "budgets": [list(b) for b in budgets], "design_hash": digest,
                "feature_stats": stats, "status": "running"}
        folder.mkdir(parents=True, exist_ok=True)
        _check_saved(info_path, info, describe(task), keys=("hash_a", "hash_b", "design_hash"))
        write_json(info_path, info)
        pair, written = (A, B, truth), 0
        rows = G._Rows(runs_path)
        try:
            for run_id, item in enumerate(plan):
                if run_id in done:
                    continue
                solver_seed = _solver_seed(seed, item["budget"], algorithm, 2, item["set_code"], item["repeat"])
                out = run_one(algorithm, model, pair, item["seeds"], test, solver_seed)
                rows.write({"model": model, "mean_degree": degree, "rho": rho, "algorithm": algorithm, "stage": 2,
                            "pair_index": i, "pair_seed": seed, "level": item["level"], "budget": item["budget"],
                            "method": item["method"], "set_code": item["set_code"], "pool_index": item["pool_index"],
                            "repeat": item["repeat"], "run_id": run_id,
                            "seed_vertices": json.dumps([int(v) for v in item["seeds"]]),
                            **{f: item[f] for f in CORE}, **{f"z_{f}": item[f"z_{f}"] for f in CORE},
                            "score": item["score"], "score_percentile": item["score_percentile"],
                            "solver_seed": solver_seed, **out})
                written += 1
        finally:
            rows.close()
        write_json(info_path, {**info, "status": "complete", "finished_at": _now()})
    return f"{describe(task)}: {written} runs in {(time.perf_counter() - started) / 60:.1f} min"


def _stage2_left(task, design) -> int:
    model, degree, rho, algorithm, i, budgets = task
    s2 = design["stage2"]
    per_budget = s2["random_sets"] + (len(s2["rules"][algorithm]) + len(s2["screens"])) * s2["runs"][algorithm]
    path = stage_dir(model, degree, rho, algorithm, 2, i) / "runs.csv"
    return max(per_budget * len(budgets) - len(G._completed_ids(path, "run_id")), 0)


def run_stage2(model: str = "sparse_er", algorithms=ALGORITHMS, degrees=None, rhos=RHOS,
               n_pairs: int = STAGE2_PAIRS, jobs: int = 3, blas_threads: int = 3) -> bool:
    """Run rules and best of N against random sets in every cell whose stage 1 curve rises in range."""

    design = prepare(model, blas_threads)
    tasks, waiting, flat = [], [], []
    for d, rho, a in _grid(model, algorithms, degrees, rhos):
        budgets = stage2_budgets(model, d, rho, a, design)
        if budgets is None:
            waiting.append(f"{LABELS[a]} d{d} rho {rho:.2f}")
            continue
        if not budgets:
            flat.append(f"{LABELS[a]} d{d} rho {rho:.2f}")
            continue
        tasks += [(model, d, rho, a, i, tuple(budgets)) for i in range(n_pairs)]
    if waiting:
        print(f"Stage 1 isn't finished for {len(waiting)} cells; they are skipped: {', '.join(waiting)}")
    if flat:
        print(f"No 20-80% rise inside the budget range, so no stage 2, for {len(flat)} cells: {', '.join(flat)}")
    tasks = [t for t in tasks if _stage2_left(t, design) > 0]
    if not tasks:
        print(f"Stage 2 has nothing left to run for {MODEL_LABELS[model]}.")
        return True

    def seconds(task):
        return _stage2_left(task, design) * SECONDS_PER_RUN[task[3]]

    tasks.sort(key=seconds, reverse=True)
    print(f"{_clock()}  Stage 2, {MODEL_LABELS[model]}: {len(tasks)} pair runs, {jobs} at a time, about "
          f"{sum(seconds(t) for t in tasks) / jobs / 60:.0f} min (a rough guess). One line per finished pair.",
          flush=True)
    return _run(tasks, stage2_job, design, seconds, jobs)


# --------------------------------------------------------------------------
# Running jobs in parallel
# --------------------------------------------------------------------------

def _run(tasks: list, job, design: dict, seconds, jobs: int) -> bool:
    failures, total = [], len(tasks)

    def report(n, task, message, error, remaining):
        if error is None:
            left = sum(seconds(t) for t in remaining) / max(jobs, 1) / 60
            print(f"{_clock()}  [{n}/{total}] {message}; about {left:.0f} min left", flush=True)
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
                    message = job(task, design)
                except Exception as error:  # noqa: BLE001 - report every failure, keep going
                    report(n, task, None, error, [])
                    continue
                report(n, task, message, None, tasks[n:])
        else:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor, as_completed
            pool = ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn"),
                                       initializer=G._ignore_interrupts)
            futures = {pool.submit(job, task, design): task for task in tasks}
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


def status(model: str = "sparse_er", algorithms=ALGORITHMS, degrees=None, rhos=RHOS,
           stage1_pairs: int = STAGE1_PAIRS, stage2_pairs: int = STAGE2_PAIRS) -> pd.DataFrame:
    """Pairs finished per cell and algorithm, for both stages."""

    rows = []
    for d, rho, a in _grid(model, algorithms, degrees, rhos):
        path = OUTPUT_DIR / model / cell_name(d, rho) / a / "stage2" / "budgets.json"
        budgets = _read_json(path)["budgets"] if path.exists() else None
        done2 = sum(1 for i in range(stage2_pairs)
                    if (stage_dir(model, d, rho, a, 2, i) / "pair.json").exists()
                    and _read_json(stage_dir(model, d, rho, a, 2, i) / "pair.json").get("status") == "complete")
        rows.append({"algorithm": LABELS[a], "mean degree": d, "rho": rho,
                     "stage 1 pairs": f"{sum(_stage1_complete(model, d, rho, a, i) for i in range(stage1_pairs))}"
                                      f" of {stage1_pairs}",
                     "stage 2 budgets": "-" if budgets is None else (", ".join(str(k) for _, k in budgets) or "none"),
                     "stage 2 pairs": f"{done2} of {stage2_pairs}" if budgets else "-"})
    return pd.DataFrame(rows).set_index(["algorithm", "mean degree", "rho"])


# --------------------------------------------------------------------------
# Reading stage 1: the shape of single runs
# --------------------------------------------------------------------------

def read_stage(model: str = "sparse_er", stage: int = 1) -> pd.DataFrame:
    files = sorted((OUTPUT_DIR / model).glob(f"d*_rho*/*/stage{stage}/pair_*/runs.csv"))
    frames = [pd.read_csv(path) for path in files if path.stat().st_size > 0]
    if not frames:
        raise FileNotFoundError(f"No stage {stage} results yet in {OUTPUT_DIR / model}")
    return pd.concat(frames, ignore_index=True)


def _shares(values: pd.Series) -> tuple[float, float]:
    values = values.dropna()
    if values.empty:
        return np.nan, np.nan
    middle = float(((values >= MIDDLE[0]) & (values <= MIDDLE[1])).mean())
    ends = float(((values <= ALL_OR_NOTHING[0]) | (values >= ALL_OR_NOTHING[1])).mean())
    return middle, ends


def solver_share(groups: pd.Series, values: pd.Series) -> float:
    """Share of the variance between single runs that comes from the solver seed alone: one-way analysis of
    variance over sets run several times (set variance from the mean squares, set to 0 when negative)."""

    frame = pd.DataFrame({"g": np.asarray(groups), "y": np.asarray(values, dtype=float)}).dropna()
    sizes = frame.groupby("g")["y"].size()
    if len(sizes) < 2 or (sizes < 2).any():
        return np.nan
    means = frame.groupby("g")["y"].transform("mean")
    ms_within = float(((frame["y"] - means) ** 2).sum() / (sizes - 1).sum())
    group_means = frame.groupby("g")["y"].mean()
    ms_between = float((sizes * (group_means - frame["y"].mean()) ** 2).sum() / (len(sizes) - 1))
    between = max((ms_between - ms_within) / sizes.mean(), 0.0)
    total = ms_within + between
    return float(ms_within / total) if total > 0 else 0.0


def _label(curve: pd.Series, k50, middle: float, ends: float) -> str:
    if curve.empty or not np.isfinite(curve.max()):
        return "no data"
    if k50 is None:
        return "easy (below 3 seeds)" if curve.iloc[0] >= 0.5 else (
            "fails throughout" if curve.max() < MIDDLE[0] else "partial at 300 seeds")
    if np.isfinite(ends) and ends >= 0.7:
        return "all-or-nothing"
    if np.isfinite(middle) and middle >= 0.6:
        return "graded"
    return "mixed"


def regime_table(data: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
    """One row per cell and algorithm: where random sets' accuracy rises, how single runs are spread at the budget
    closest to 0.5 (per pair), the ceiling, solver luck, GOAT's regulariser check and the spread of random sets'
    features."""

    rows = []
    for (m, d, rho, a), part in data.groupby(["model", "mean_degree", "rho", "algorithm"], sort=False):
        curve = cell_curve(part)
        main = part[(part["repeat"] == 0) & (part["variant"] == 0)]
        k = {level: crossing(curve.index, curve.to_numpy(), level) for level in (0.2, 0.5, 0.8)}
        per_pair, middles, ends, solver, reg_low, reg_high, spread_cv, score_sd = [], [], [], [], [], [], [], []
        for i, pair_runs in main.groupby("pair_index"):
            means = pair_runs.groupby("budget")["accuracy_test"].mean()
            per_pair.append(crossing(means.index, means.to_numpy(), 0.5))
            middle = middle_budget(means.to_dict())
            if middle is None:
                continue
            share_middle, share_ends = _shares(pair_runs.loc[pair_runs["budget"] == middle, "accuracy_test"])
            middles.append(share_middle)
            ends.append(share_ends)
            at_middle = part[(part["pair_index"] == i) & (part["budget"] == middle) & (part["variant"] == 0)]
            repeated = at_middle.dropna(subset=["accuracy_test"]).groupby("set_index").filter(lambda g: len(g) > 1)
            share = solver_share(repeated["set_index"], repeated["accuracy_test"])
            if np.isfinite(share):
                solver.append(share)
            variants = part[(part["pair_index"] == i) & (part["budget"] == middle)]
            base = variants[(variants["variant"] == 0) & (variants["repeat"] == 0)].set_index("set_index")["accuracy_test"]
            for v, store in ((1, reg_low), (2, reg_high)):
                scaled = variants[variants["variant"] == v].set_index("set_index")["accuracy_test"]
                if len(scaled):
                    store.append(float((scaled - base.reindex(scaled.index)).mean()))
            info_path = stage_dir(m, d, rho, a, 1, int(i)) / "pair.json"
            if info_path.exists():
                spread = _read_json(info_path).get("spread") or {}
                if spread:
                    spread_cv.append(np.nanmean([spread[f]["cv"] for f in ("degree", "pair_mass")]))
                    score_sd.append(spread["score_sd"])
        middle_share = float(np.mean(middles)) if middles else np.nan
        ends_share = float(np.mean(ends)) if ends else np.nan
        rows.append({
            "model": m, "mean degree": d, "rho": rho, "algorithm": a, "pairs": main["pair_index"].nunique(),
            "regime": _label(curve, k[0.5], middle_share, ends_share),
            "k20": k[0.2], "k50": k[0.5], "k80": k[0.8],
            "k50 per pair": ", ".join("-" if v is None else f"{v:.0f}" for v in per_pair),
            "rise (k80/k20)": k[0.8] / k[0.2] if k[0.2] and k[0.8] else np.nan,
            "part-way share": middle_share, "all-or-nothing share": ends_share,
            "ceiling": float(curve.max()) if len(curve) else np.nan,
            "solver share": float(np.mean(solver)) if solver else np.nan,
            "GOAT reg x0.5": float(np.mean(reg_low)) if reg_low else np.nan,
            "GOAT reg x2": float(np.mean(reg_high)) if reg_high else np.nan,
            "feature CV": float(np.mean(spread_cv)) if spread_cv else np.nan,
            "score SD": float(np.mean(score_sd)) if score_sd else np.nan,
            "runs": len(part),
        })
    table = pd.DataFrame(rows)
    if model is not None:
        table = table[table["model"] == model]
    return table


def show_regime_table(table: pd.DataFrame, algorithm: str | None = None) -> pd.DataFrame:
    out = table if algorithm is None else table[table["algorithm"] == algorithm]
    out = out.assign(algorithm=out["algorithm"].map(LABELS)).drop(columns=["model"])
    if out["GOAT reg x0.5"].isna().all():
        out = out.drop(columns=["GOAT reg x0.5", "GOAT reg x2"])
    out = out.set_index(["algorithm", "mean degree", "rho"])
    for column in ("k20", "k50", "k80", "rise (k80/k20)"):
        out[column] = out[column].round(1)
    return out.round(3).astype(object).where(out.notna(), "")


def plot_regimes(table: pd.DataFrame, model: str):
    """One heatmap per algorithm: rows rho, columns mean degree. Colour: share of single runs ending part-way at the
    budget closest to 0.5 (light = all-or-nothing, dark = graded). Text: the budget where random sets average 50%."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    table = table[table["model"] == model]
    algorithms = [a for a in ALGORITHMS if a in set(table["algorithm"])]
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    with plt.rc_context({"font.size": 9.5}):
        fig, axes = plt.subplots(1, len(algorithms), figsize=(4.2 * len(algorithms) + 1.2, 5.0), squeeze=False,
                                 constrained_layout=True)
        for ax, a in zip(axes[0], algorithms):
            part = table[table["algorithm"] == a]
            degrees = sorted(part["mean degree"].unique())
            rhos = sorted(part["rho"].unique(), reverse=True)
            grid = np.full((len(rhos), len(degrees)), np.nan)
            for r, rho in enumerate(rhos):
                for c, d in enumerate(degrees):
                    row = part[(part["rho"] == rho) & (part["mean degree"] == d)]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    if _number(row["k50"]):
                        grid[r, c] = row["part-way share"]
            image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1, aspect="auto")
            ax.set_facecolor(GRID)
            for r, rho in enumerate(rhos):
                for c, d in enumerate(degrees):
                    row = part[(part["rho"] == rho) & (part["mean degree"] == d)]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    if _number(row["k50"]):
                        text = f"{row['k50']:.0f} seeds\n{row['regime']}"
                    else:
                        text = row["regime"].replace(" (", "\n(").replace(" at ", "\nat ")
                    value = grid[r, c]
                    color = "white" if np.isfinite(value) and value > 0.55 else INK
                    ax.text(c, r, text, ha="center", va="center", fontsize=8, color=color)
            ax.set_xticks(range(len(degrees)), [f"{d:g}" for d in degrees])
            ax.set_yticks(range(len(rhos)), [f"{rho:.2f}" for rho in rhos])
            ax.set_xlabel("mean degree")
            if ax is axes[0, 0]:
                ax.set_ylabel("edge correlation rho")
            ax.set_title(LABELS[a], fontsize=11)
            for spine in ax.spines.values():
                spine.set_visible(False)
        bar = fig.colorbar(image, ax=axes[0].tolist(), fraction=0.03, pad=0.02)
        bar.set_label("share of single runs ending part-way (0.2-0.8)\nat the budget closest to 0.5")
        fig.suptitle(f"{MODEL_LABELS[model]}, 600 vertices: where single runs end all-or-nothing and where they are "
                     "graded\ntext: budget where random sets average 50% test accuracy; grey: no 50% crossing "
                     "between 3 and 300 seeds", fontsize=11.5)
    return fig


def plot_curves(data: pd.DataFrame, algorithm: str, model: str):
    """Test accuracy of single random sets against budget, one panel per cell (rows rho, columns mean degree).
    Lines: mean per pair. Grey band: 0.2-0.8."""

    import matplotlib.pyplot as plt

    part = data[(data["algorithm"] == algorithm) & (data["model"] == model) & (data["repeat"] == 0)
                & (data["variant"] == 0)]
    degrees = sorted(part["mean_degree"].unique())
    rhos = sorted(part["rho"].unique(), reverse=True)
    rng = np.random.default_rng(0)
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9}):
        fig, axes = plt.subplots(len(rhos), len(degrees), figsize=(4.1 * len(degrees), 2.35 * len(rhos)),
                                 squeeze=False, sharex=True, sharey=True)
        for r, rho in enumerate(rhos):
            for c, d in enumerate(degrees):
                ax = axes[r, c]
                cell = part[(part["rho"] == rho) & (part["mean_degree"] == d)]
                ax.axhspan(*MIDDLE, color=BAND, zorder=0)
                if not cell.empty:
                    jitter = np.exp(rng.uniform(-0.06, 0.06, len(cell)))
                    ax.scatter(cell["budget"] * jitter, cell["accuracy_test"], s=11, color=BLUE, alpha=0.5,
                               linewidths=0, zorder=2)
                    for i, style in zip(sorted(cell["pair_index"].unique()), ("-", "--", ":", "-.")):
                        means = cell[cell["pair_index"] == i].groupby("budget")["accuracy_test"].mean()
                        ax.plot(means.index, means.to_numpy(), style, color=INK, linewidth=1.2, zorder=3)
                ax.set_xscale("log")
                ax.set_xticks([3, 10, 30, 100, 300], ["3", "10", "30", "100", "300"])
                ax.set_ylim(-0.03, 1.03)
                ax.grid(axis="y", color=GRID, linewidth=0.6)
                ax.set_title(f"rho {rho:.2f}, mean degree {d:g}", fontsize=9.5)
                if c == 0:
                    ax.set_ylabel("test accuracy")
                if r == len(rhos) - 1:
                    ax.set_xlabel("seeds")
        fig.suptitle(f"{LABELS[algorithm]} · {MODEL_LABELS[model]}, 600 vertices: single random seed sets (dots) and "
                     "each pair's mean (lines); grey band 0.2-0.8", fontsize=11.5)
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Reading stage 2: gains over random sets
# --------------------------------------------------------------------------

def stage2_means(data: pd.DataFrame, outcome: str = "accuracy_test") -> pd.DataFrame:
    """Mean outcome per cell, algorithm, pair, level and budget (rows) and method (columns)."""

    keys = ["model", "mean_degree", "rho", "algorithm", "pair_index", "level", "budget"]
    table = data.dropna(subset=[outcome]).groupby(keys + ["method"])[outcome].mean().unstack("method")
    return table[[m for m in METHODS if m in table.columns]]


def _seeds_saved(random_curve: pd.Series, accuracy: float, budget: int) -> float:
    """How many more seeds random sets would need for this accuracy, as a share of the budget: log-linear
    interpolation on the running maximum of the pair's random-set curve. Beyond the curve's range the nearest end is
    used, so the value is then a bound that errs toward zero (a method that beats random sets at every budget on the
    curve gets the largest budget)."""

    curve = random_curve.dropna().sort_index()
    budgets, means = curve.index.to_numpy(dtype=float), np.maximum.accumulate(curve.to_numpy(dtype=float))
    if len(budgets) < 2 or not np.isfinite(accuracy):
        return np.nan
    if accuracy <= means[0]:
        equivalent = budgets[0]
    elif accuracy > means[-1]:
        equivalent = budgets[-1]
    else:
        j = int(np.argmax(means >= accuracy))
        low, high = means[j - 1], means[j]
        t = 1.0 if high == low else (accuracy - low) / (high - low)
        equivalent = np.exp(np.log(budgets[j - 1]) + t * (np.log(budgets[j]) - np.log(budgets[j - 1])))
    return float(equivalent / budget - 1)


def random_curves(data: pd.DataFrame, stage1: pd.DataFrame | None = None,
                  outcome: str = "accuracy_test") -> pd.Series:
    """Random sets' mean outcome per cell, algorithm, pair and budget: stage 2's random sets, pooled with stage 1's
    random sets on the pairs that were in stage 1 (budgets 3 to 300)."""

    keys = ["model", "mean_degree", "rho", "algorithm", "pair_index", "budget"]
    frames = [data.loc[data["method"] == "random", keys + [outcome]]]
    if stage1 is not None and len(stage1):
        main = stage1[(stage1["repeat"] == 0) & (stage1["variant"] == 0)]
        frames.append(main[keys + [outcome]])
    runs = pd.concat(frames, ignore_index=True).dropna(subset=[outcome])
    return runs.groupby(keys)[outcome].mean()


def gain_table(data: pd.DataFrame, stage1: pd.DataFrame | None = None,
               outcome: str = "accuracy_test") -> pd.DataFrame:
    """Per cell, algorithm and method: gain over random sets at each stage 2 budget (paired within pair) and on
    average, with a 95% t-interval over pairs, and seeds saved (median over pairs and budgets). Pass stage 1's runs
    so that seeds saved is read off each stage 1 pair's full curve; without them only the stage 2 budgets are used
    and more values hit the ends of the range."""

    means = stage2_means(data, outcome)
    curves = random_curves(data, stage1, outcome)
    rows = []
    for (m, d, rho, a), part in means.groupby(level=["model", "mean_degree", "rho", "algorithm"], sort=False):
        for method in part.columns:
            if method == "random" or part[method].isna().all():
                continue
            gain = (part[method] - part["random"]).dropna()
            per_pair = gain.groupby(level="pair_index").mean()
            mean, low, high = G._interval(per_pair)
            row = {"model": m, "mean degree": d, "rho": rho, "algorithm": a, "method": method,
                   "pairs": len(per_pair), "gain": mean, "low": low, "high": high}
            for level, at_level in gain.groupby(level="level"):
                row[f"gain at {level:.0%}"] = float(at_level.mean())
            saved = []
            for i, pair_part in part.groupby(level="pair_index"):
                random_curve = curves.loc[(m, d, rho, a, i)]
                for idx, value in pair_part[method].items():
                    saved.append(_seeds_saved(random_curve, value, int(idx[-1])))
            row["seeds saved"] = float(np.nanmedian(saved)) if np.isfinite(saved).any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def random_shape(data: pd.DataFrame) -> pd.DataFrame:
    """Part-way and all-or-nothing shares of random sets' single runs at the stage 2 budget where they average 50%
    (10 sets x 4 pairs, so firmer than stage 1's 16 runs)."""

    part = data[(data["method"] == "random") & np.isclose(data["level"], 0.5)]
    rows = []
    for (m, d, rho, a), cell in part.groupby(["model", "mean_degree", "rho", "algorithm"], sort=False):
        middle, ends = _shares(cell["accuracy_test"])
        rows.append({"model": m, "mean degree": d, "rho": rho, "algorithm": a, "random runs at 50%": len(cell),
                     "part-way share": middle, "all-or-nothing share": ends})
    return pd.DataFrame(rows)


def best_table(gains: pd.DataFrame, data: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per cell and algorithm: the best of round 1's rules, and the best-of-N screens, by average gain. With the
    stage 2 runs, also the shape of random sets' single runs at the 50% budget."""

    rows = []
    for (m, d, rho, a), part in gains.groupby(["model", "mean degree", "rho", "algorithm"], sort=False):
        rules = part[part["method"].isin(RULES[a])].dropna(subset=["gain"])
        best = rules.loc[rules["gain"].idxmax()] if len(rules) else None
        screens = part.set_index("method")
        rows.append({"model": m, "mean degree": d, "rho": rho, "algorithm": a,
                     "best rule": best["method"] if best is not None else "",
                     "best rule gain": best["gain"] if best is not None else np.nan,
                     "best rule interval": (f"{best['low']:+.3f} to {best['high']:+.3f}"
                                            if best is not None and np.isfinite(best["low"]) else ""),
                     "best rule seeds saved": best["seeds saved"] if best is not None else np.nan,
                     "top degree gain": screens["gain"].get("top_degree", np.nan),
                     "best of 1,000 by score gain": screens["gain"].get("best_1000", np.nan),
                     "best of 1,000 by degree gain": screens["gain"].get("degree_1000", np.nan)})
    table = pd.DataFrame(rows)
    if data is not None and len(table):
        table = table.merge(random_shape(data), on=["model", "mean degree", "rho", "algorithm"], how="left")
    return table


def show_gain_table(gains: pd.DataFrame, algorithm: str | None = None) -> pd.DataFrame:
    out = gains if algorithm is None else gains[gains["algorithm"] == algorithm]
    out = out.assign(algorithm=out["algorithm"].map(LABELS), method=out["method"].map(METHOD_LABELS))
    out = out.assign(**{"95% interval": [("" if pd.isna(lo) else f"{lo:+.3f} to {hi:+.3f}")
                                         for lo, hi in zip(out["low"], out["high"])]})
    out = out.drop(columns=["model", "low", "high"]).set_index(["algorithm", "mean degree", "rho", "method"])
    return out.round(3).astype(object).where(out.notna(), "")


def plot_gains(gains: pd.DataFrame, model: str, regimes: pd.DataFrame | None = None):
    """Top row: the best of round 1's rules per cell (its gain in test accuracy over random sets, averaged over the
    stage 2 budgets). Bottom row: best of 1,000 random sets by the frozen score. Blue: better than random; red:
    worse; grey: no stage 2 for that cell. Pass the stage 1 regime table to show every cell of the grid."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    best = best_table(gains[gains["model"] == model])
    present = set(best["algorithm"]) | (set() if regimes is None else set(regimes["algorithm"]))
    algorithms = [a for a in ALGORITHMS if a in present]
    limit = max(0.05, float(np.nanmax(np.abs(best[["best rule gain", "best of 1,000 by score gain"]].to_numpy()))))
    cmap = LinearSegmentedColormap.from_list("red_grey_blue", (RED, NEUTRAL, BLUE))
    rows = (("best rule gain", "best of round 1's rules"), ("best of 1,000 by score gain", "best of 1,000 by score"))
    with plt.rc_context({"font.size": 9.5}):
        fig, axes = plt.subplots(2, len(algorithms), figsize=(4.2 * len(algorithms) + 1.2, 8.8), squeeze=False,
                                 constrained_layout=True)
        for c, a in enumerate(algorithms):
            part = best[best["algorithm"] == a]
            grid_source = part if regimes is None else regimes[(regimes["model"] == model)
                                                               & (regimes["algorithm"] == a)]
            degrees = sorted(grid_source["mean degree"].unique())
            rhos = sorted(grid_source["rho"].unique(), reverse=True)
            for r, (column, title) in enumerate(rows):
                ax = axes[r, c]
                grid = np.full((len(rhos), len(degrees)), np.nan)
                for i, rho in enumerate(rhos):
                    for j, d in enumerate(degrees):
                        row = part[(part["rho"] == rho) & (part["mean degree"] == d)]
                        if len(row):
                            grid[i, j] = row.iloc[0][column]
                image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
                ax.set_facecolor(GRID)
                for i, rho in enumerate(rhos):
                    for j, d in enumerate(degrees):
                        row = part[(part["rho"] == rho) & (part["mean degree"] == d)]
                        if len(row) and np.isfinite(grid[i, j]):
                            text = f"{grid[i, j]:+.2f}"
                            if r == 0:
                                text += f"\n{METHOD_SHORT.get(row.iloc[0]['best rule'], '')}"
                            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                                    color="white" if abs(grid[i, j]) > 0.6 * limit else INK)
                ax.set_xticks(range(len(degrees)), [f"{d:g}" for d in degrees])
                ax.set_yticks(range(len(rhos)), [f"{rho:.2f}" for rho in rhos])
                ax.set_xlabel("mean degree")
                if c == 0:
                    ax.set_ylabel("edge correlation rho")
                ax.set_title(f"{LABELS[a]}: {title}", fontsize=10.5)
                for spine in ax.spines.values():
                    spine.set_visible(False)
        bar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
        bar.set_label("gain in test accuracy over random sets\n(mean over the stage 2 budgets and pairs)")
        fig.suptitle(f"{MODEL_LABELS[model]}: does seed choice matter? Grey: no stage 2 (no 20-80% rise in range)",
                     fontsize=11.5)
    return fig


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, choices=(1, 2), help="run stage 1 or stage 2")
    parser.add_argument("--model", default="sparse_er", choices=MODELS)
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=ALGORITHMS)
    parser.add_argument("--degrees", nargs="+", type=float, help="mean degrees (default: the model's three)")
    parser.add_argument("--rhos", nargs="+", type=float, default=list(RHOS))
    parser.add_argument("--pairs", type=int, help="pairs per cell (default: 2 for stage 1, 4 for stage 2)")
    parser.add_argument("--status", action="store_true", help="only show how far it is")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--blas-threads", type=int, default=3)
    args = parser.parse_args(argv)
    degrees = None if args.degrees is None else [int(d) if float(d).is_integer() else d for d in args.degrees]
    if args.status or args.stage is None:
        print(status(args.model, args.algorithms, degrees, args.rhos).to_string())
        return
    if args.stage == 1:
        run_stage1(args.model, args.algorithms, degrees, args.rhos, args.pairs or STAGE1_PAIRS, args.jobs,
                   args.blas_threads)
    else:
        run_stage2(args.model, args.algorithms, degrees, args.rhos, args.pairs or STAGE2_PAIRS, args.jobs,
                   args.blas_threads)


if __name__ == "__main__":
    main()
