"""Graded outcomes: does the weighted score track the match ratio when runs end part-way? (4 Nov 2026)

Why. At round 1's settings almost every run either recovers the whole matching or fails: only 0-6% of runs end
between 0.35 and 0.85 accuracy (fresh_pair_scores.ipynb), so each run is close to a coin flip. The group asked for
settings where single runs end with partial, well-spread match ratios, to see whether the weighted score tracks the
match ratio more clearly there.

How. Lower the edge correlation rho of the ER pairs used for GOAT and FAQ (600 vertices, mean degree 16; round 1
uses rho = 0.50). A rough check on 4 Nov with scipy's FAQ (the same algorithm as graspologic's) found graded runs at
rho = 0.3: with 150 random seeds, six runs ended between 0.44 and 0.68 accuracy (0.20-0.32 with 100 seeds,
0.75-0.83 with 200). At rho = 0.4 and 0.5, runs were still all or nothing.

1. Pilot. For each algorithm, rho in PILOT_RHOS and a ladder of budgets, PILOT_SETS random seed sets on one pair per
   rho, one run each. recommend() picks, for each algorithm, the setting where most runs end between 0.2 and 0.8
   and the mean is closest to 0.5 (or reports that none qualifies).
2. Collection. In the chosen setting, COLLECTION_PAIRS fresh pairs x COLLECTION_SETS random seed sets, one run each,
   with the 11 graph-A features of every set. The first REPEAT_SETS sets of each pair are run REPEATS more times with
   other solver seeds, to measure how much of the spread is solver luck.
3. Analysis. The frozen pooled score (fitted for success at rho = 0.5 and never refitted), degree alone and single
   features against the match ratio; then weights refitted on the match ratio, leaving one pair out, with the 4 core
   features and with all 11.
4. Rules at low rho (added 5 Nov). Do round 1's rules and best of N still beat random seed sets when runs end
   part-way? RULE_PAIRS new pairs per rho in RULE_RHOS, five budgets each (RULE_BUDGETS). Per pair and budget:
   RULE_RANDOM_SETS random sets (one run each); round 1's rules (top degree, clustered, spread, distinguishing, and
   hub-free distinguishing for GOAT), never picking a test vertex; and best of 100 / best of 1,000 by the frozen
   score and best of 1,000 by degree, screened from RULE_POOL random sets (as in best_of_n.py). Each rule or best-of
   set runs RULE_RUNS times (3 for FAQ, averaged; GOAT gives the same result every time for the same seeds).

Fixed test vertices. With 100+ seeds, which vertices are seeds changes which vertices accuracy is measured on: a set
full of low-degree, hard-to-match vertices removes them from the non-seed pool and raises non-seed accuracy without
being more informative. So each pair sets aside N_TEST fixed test vertices (20%) that are never seeds, and the main
outcome, accuracy_test, is the share of them matched correctly: every seed set is judged on the same vertices. The
usual non-seed accuracy is recorded as well (accuracy).

Runs. graspologic FAQ or GOAT with round 1's solver settings (max_iter 100, tol 0.01; GOAT regulariser 100, as for
ER), one run per set with its own solver seed. Pairs come from a new seed stream that skips every pair seed used so
far; FAQ and GOAT share pairs.

Run from the repository root (ds environment):

    python -m selection_rule_comparison.graded_outcomes --pilot
    python -m selection_rule_comparison.graded_outcomes --collect
    python -m selection_rule_comparison.graded_outcomes --collect --choice faq=0.30:150 goat=0.30:70
    python -m selection_rule_comparison.graded_outcomes --rules
    python -m selection_rule_comparison.graded_outcomes --status

--collect without --choice uses the pilot's recommendation. Or run graded_outcomes.ipynb from the top. Rows are
written set by set, so a stopped run resumes where it stopped (Ctrl-C, or the notebook's interrupt button, stops the
workers too). Results go to data/graded_outcomes/, or to GRADED_OUTCOMES_OUTPUT_DIR if it is set.
"""

from __future__ import annotations

import argparse
import csv
import datetime
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

from experiment_common.features import (
    build_graph_context,
    fixed_pair_seed_set_features,
    signature_robustness_features,
)
from experiment_common.graph_checks import first_passing_pair
from experiment_common.graphs import generate_graph_pair
from experiment_common.provenance import collect_environment, thread_limit, write_json
from fixed_pair_seed_set_classification import multi_pair_collection as MP

from . import run_comparison as rc
from . import settings as S
from .strategies import STRATEGY_CODES, TARGETS, GraphA, degree_first

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent
OUTPUT_DIR = Path(os.environ.get("GRADED_OUTCOMES_OUTPUT_DIR") or EXPERIMENT_DIR / "data" / "graded_outcomes")
WEIGHTS_PATH = REPO_ROOT / "fixed_pair_seed_set_classification" / "data" / "multi_pair_2026-10-30" / "pooled_weights.json"

ALGORITHMS = ("faq", "goat")
MODEL = "sparse_er"
CORE = MP.CORE_FEATURES                      # degree, harmonic, pair_mass, uniqueness
ALL_FEATURES = tuple(MP.FEATURE_COLUMNS)     # the 11 graph-A features (the 9 fixed-pair ones, uniqueness at cap 2 and 4)

PILOT_RHOS = (0.25, 0.30, 0.35)
PILOT_BUDGETS = {"faq": (60, 90, 120, 150, 180, 220, 260), "goat": (20, 35, 50, 70, 100, 140, 200)}
PILOT_SETS = 24          # random sets run per pilot setting
PILOT_POOL = 2_000       # random sets drawn per pilot setting to standardise the features (the first PILOT_SETS run)
COLLECTION_PAIRS = 3
COLLECTION_SETS = 1_000
REPEAT_SETS = 50         # sets per collection pair run again ...
REPEATS = 2              # ... this many more times, with other solver seeds
N_TEST = 120             # fixed test vertices per pair (20% of 600), never seeds
MIDDLE = (0.2, 0.8)      # a run "ends part-way" if its test accuracy is in this range
MIN_MIDDLE_SHARE = 0.75  # recommend(): at least this share of a setting's runs must end part-way

PAIR_SEED_BASE = 20261104
SAMPLING_SEED_BASE = 20261114
SOLVER_SEED_BASE = 20261124
TEST_SEED_BASE = 20261134
PURPOSE_CODES = {"pilot": 1, "collection": 2, "rules": 3}
SECONDS_PER_RUN = {"faq": 2.5, "goat": 5.0}  # rough wall time per run with 3 jobs at rho 0.25-0.35 (estimates only)

# Rules at low rho (section 8 of the notebook). Budgets come from the pilot: random sets' mean test accuracy runs from
# about 0.1 to 0.6 (rho 0.25), 0.15 to 0.85 (0.30) and across the jump (0.35).
RULE_RHOS = (0.25, 0.30, 0.35)
RULE_BUDGETS = {
    "faq": {0.25: (120, 150, 180, 220, 260), 0.30: (90, 120, 150, 180, 220), 0.35: (40, 50, 60, 70, 80)},
    "goat": {0.25: (100, 140, 170, 200, 240), 0.30: (70, 100, 120, 140, 170), 0.35: (30, 40, 50, 60, 70)},
}
RULE_PAIRS = 5           # new pairs per rho, shared by FAQ and GOAT
RULE_RANDOM_SETS = 10    # random sets run per pair and budget, one run each
RULE_POOL = 1_000        # random sets drawn per pair and budget: the z-scores, and the pool for best of 100 / 1,000
RULE_RUNS = {"faq": 3, "goat": 1}   # runs per rule or best-of set (GOAT gives the same result for the same seeds)
RULES = {a: tuple(s for s in S.STRATEGIES["main"][a] if s != "random") for a in ALGORITHMS}
SCREENS = ("best_100", "best_1000", "degree_1000")
METHODS = ("random", "top_degree", "clustered", "spread", "distinguishing", "hub_free_distinguishing", *SCREENS)
RULE_SECONDS_PER_RUN = {"faq": 1.6, "goat": 2.8}   # wall time per run with 3 jobs in the pilot, over these budgets

LABELS = MP.ALGORITHM_LABELS
FEATURE_LABELS = {
    "seed_nonseed_degree_mean": "degree",
    "seed_harmonic_mean": "harmonic",
    "nonseed_covered_fraction": "coverage",
    "nonseed_witness_overlap_fraction": "witness overlap",
    "nonseed_witness_pair_mass_mean": "pair mass",
    "nonseed_nearest_seed_distance_mean": "distance to nearest seed",
    "nonseed_three_hop_seed_count_mean": "seeds within 3 hops",
    "nonseed_distance_signature_unique_fraction_r3": "uniqueness (cap 3)",
    "seed_nonseed_neighborhood_jaccard_mean": "Jaccard overlap",
    "nonseed_distance_signature_unique_fraction_r2": "uniqueness (cap 2)",
    "nonseed_distance_signature_unique_fraction_r4": "uniqueness (cap 4)",
}

# validated palette (dataviz): slot 1 blue, slot 2 orange, plus neutrals
BLUE, ORANGE, INK, INK2, GRID, BAND = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e6e6", "#f0f0ef"

METHOD_LABELS = {"random": "random", "top_degree": "top degree", "clustered": "clustered (pair mass)",
                 "spread": "spread (coverage)", "distinguishing": "distinguishing",
                 "hub_free_distinguishing": "hub-free distinguishing", "best_100": "best of 100 by score",
                 "best_1000": "best of 1,000 by score", "degree_1000": "best of 1,000 by degree"}
METHOD_STYLE = {   # the same as best_of_n.RULE_STYLE: rules in orange told apart by shape, screens in blues and aqua
    "random": dict(color="#9a9a9a", marker="o"),
    "top_degree": dict(color="#2b2b2b", marker="s"),
    "clustered": dict(color="#eb6834", marker="^"),
    "spread": dict(color="#eb6834", marker="v"),
    "distinguishing": dict(color="#eb6834", marker="P"),
    "hub_free_distinguishing": dict(color="#eb6834", marker="*"),
    "degree_1000": dict(color="#1baf7a", marker="X"),
    "best_100": dict(color="#5598e7", marker="o"),
    "best_1000": dict(color="#1c5cab", marker="o"),
}


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _clock() -> str:
    return datetime.datetime.now().astimezone().strftime("%a %H:%M")


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _spearman(x, y) -> float:
    frame = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(frame["x"].rank().corr(frame["y"].rank()))


# --------------------------------------------------------------------------
# Settings, weights and pairs
# --------------------------------------------------------------------------

def graph_config(algorithm: str, rho: float):
    """Round 1's ER settings for GOAT and FAQ (600 vertices, mean degree 16), with another edge correlation."""

    return replace(S.graph_config(algorithm, MODEL), rho=float(rho), n_graph_pairs=1)


def load_weights(path: Path = WEIGHTS_PATH) -> dict:
    """The frozen pooled weights (per SD) for ER, fitted for success at round 1's settings."""

    saved = _read_json(path)
    return {a: {f: float(saved[f"{a}/{MODEL}"]["weights_per_sd"][f]) for f in CORE} for a in ALGORITHMS}


def known_pair_seeds() -> set[int]:
    """Pair seeds used anywhere so far (training, calibration, round 1, score transfer part C, fresh pairs)."""

    seeds = {int(s) for s in MP.excluded_pair_seeds()}
    for folder in (MP.OUTPUT_DIR, EXPERIMENT_DIR / "data" / "fresh_pair_scores"):
        for path in folder.glob("*/*/pair_*/pair.json"):
            try:
                seeds.add(int(_read_json(path)["pair_seed"]))
            except (KeyError, ValueError, OSError):
                pass
    for path in (EXPERIMENT_DIR / "data" / "score_transfer" / "C").glob("*/*/*/pair_*.csv"):
        try:
            seeds.add(int(pd.read_csv(path, usecols=["pair_seed"], nrows=1).iloc[0, 0]))
        except (KeyError, ValueError, OSError, IndexError):
            pass
    return seeds


def first_seed(algorithm: str, rho: float, purpose: str, pair_index: int) -> int:
    family = S.FAMILY_CODES[S.GRAPH_FAMILY[algorithm]]   # the same for GOAT and FAQ, so they share pairs
    seed = rc._seed(PAIR_SEED_BASE, family, int(round(100 * rho)), PURPOSE_CODES[purpose], pair_index)
    return seed % (2**32 - 10_000)


def load_pair(algorithm: str, rho: float, purpose: str, pair_index: int, excluded: set[int]):
    """The structure-checked pair with this index (both graphs connected, no automorphisms)."""

    config = graph_config(algorithm, rho)
    start = cursor = first_seed(algorithm, rho, purpose, pair_index)
    rejected = 0
    while True:
        seed, A, B, truth, _report, n_rejected = first_passing_pair(lambda s: generate_graph_pair(config, s), cursor)
        rejected += int(n_rejected)
        if int(seed) not in excluded:
            break
        cursor = int(seed) + 1
    return int(seed), A, B, truth, {"first_seed_tried": int(start), "rejected_draws": rejected}


def test_vertices(pair_seed: int, n: int, n_test: int = N_TEST) -> np.ndarray:
    rng = np.random.default_rng([TEST_SEED_BASE, int(pair_seed)])
    return np.sort(rng.choice(n, n_test, replace=False))


def _pair(algorithm: str, rho: float, purpose: str, pair_index: int, design: dict):
    seed, A, B, truth, draw = load_pair(algorithm, rho, purpose, pair_index, set(design["excluded_pair_seeds"]))
    n = len(A)
    test = test_vertices(seed, n, design["n_test"])
    candidates = np.setdiff1d(np.arange(n), test)
    aligned = B[np.ix_(truth, truth)]
    info = {"pair_seed": seed, **draw, "hash_a": rc._hash(A), "hash_b": rc._hash(B),
            "mean_degree_a": float(A.sum(axis=1).mean()), "max_degree_a": int(A.sum(axis=1).max()),
            "kept_edge_fraction": float(((A > 0) & (aligned > 0)).sum() / (A > 0).sum()),
            "test_vertices": [int(v) for v in test]}
    return seed, A, B, truth, test, candidates, info


# --------------------------------------------------------------------------
# Seed sets, features and the frozen score
# --------------------------------------------------------------------------

def draw_sets(rng: np.random.Generator, candidates: np.ndarray, k: int, size: int) -> np.ndarray:
    """`size` distinct uniform random k-subsets of the candidate vertices, each sorted."""

    index = MP.draw_pool(rng, len(candidates), k, size)
    return np.sort(candidates[index], axis=1)


def _core_features(graph: GraphA, algorithm: str, sets: np.ndarray) -> dict:
    k = sets.shape[1]
    chunk = max(8, int(8_000_000 // (graph.n * k)))     # keeps the fingerprint arrays small for large k
    return MP.PoolFeatures(graph, MP.UNIQUENESS_CAP[algorithm]).compute(sets, chunk=chunk)


def _scored(A: np.ndarray, algorithm: str, sets: np.ndarray, weights: dict, graph: GraphA | None = None):
    """Core features of every set, their z-scores against these sets, and the frozen score."""

    graph = GraphA(A) if graph is None else graph
    features = _core_features(graph, algorithm, sets)
    z, stats = {}, {}
    for f in CORE:
        mean, sd = float(features[f].mean()), float(features[f].std(ddof=1))
        stats[f] = {"mean": mean, "sd": sd}
        z[f] = (features[f] - mean) / sd if sd > 0 else np.zeros(len(sets))
    score = sum(weights[f] * z[f] for f in CORE)
    percentile = 100 * pd.Series(score).rank(method="max").to_numpy() / len(score)
    return features, z, stats, score, percentile


# --------------------------------------------------------------------------
# One run: graspologic FAQ or GOAT, keeping the final matching to score the test vertices
# --------------------------------------------------------------------------

_CAPTURE: dict = {}


def _sgm():
    """experiment_common.sgm with its graph_match call wrapped so the final matching is kept."""

    from experiment_common import sgm

    if not getattr(sgm.graph_match, "_keeps_matching", False):
        original = sgm.graph_match

        def graph_match(*args, **kwargs):
            result = original(*args, **kwargs)
            _CAPTURE["indices_A"] = np.asarray(result.indices_A)
            _CAPTURE["indices_B"] = np.asarray(result.indices_B)
            return result

        graph_match._keeps_matching = True
        sgm.graph_match = graph_match
    return sgm


def run_once(algorithm: str, pair, seeds: np.ndarray, test: np.ndarray, solver_seed: int) -> dict:
    A, B, truth = pair
    sgm = _sgm()
    _CAPTURE.clear()
    try:
        out = sgm.run_sgm_once(A, B, truth, seeds, solver_seed, S.SGM["max_iter"], S.SGM["tol"],
                               transport=algorithm == "goat", transport_regularizer=S.GOAT_REGULARIZER[MODEL],
                               transport_tol=S.SGM["transport_tol"], transport_max_iter=S.SGM["transport_max_iter"])
    except Exception as error:  # noqa: BLE001 - a failed run is recorded, not fatal
        return {"accuracy_test": np.nan, "accuracy": np.nan, "h0_accuracy": np.nan, "objective": np.nan,
                "n_iter": np.nan, "converged": False, "runtime_seconds": np.nan,
                "error": f"{type(error).__name__}: {error}"[:300]}
    if "indices_A" not in _CAPTURE:
        return {"accuracy_test": np.nan, "accuracy": float(out["final_unseeded_accuracy"]),
                "h0_accuracy": float(out["h0_accuracy"]), "objective": float(out["sgm_objective_score"]),
                "n_iter": int(out["sgm_n_iter"]), "converged": bool(out["sgm_converged"]),
                "runtime_seconds": float(out["sgm_runtime_seconds"]), "error": "final matching not captured"}
    matching = np.full(len(truth), -1, dtype=np.int64)
    matching[_CAPTURE["indices_A"]] = _CAPTURE["indices_B"]
    return {"accuracy_test": float(np.mean(matching[test] == truth[test])),
            "accuracy": float(out["final_unseeded_accuracy"]), "h0_accuracy": float(out["h0_accuracy"]),
            "objective": float(out["sgm_objective_score"]), "n_iter": int(out["sgm_n_iter"]),
            "converged": bool(out["sgm_converged"]), "runtime_seconds": float(out["sgm_runtime_seconds"]),
            "error": ""}


def _solver_seed(pair_seed: int, k: int, algorithm: str, purpose: str, candidate_id: int, repeat: int) -> int:
    return rc._seed(SOLVER_SEED_BASE, pair_seed, k, MP.ALGORITHM_CODES[algorithm], PURPOSE_CODES[purpose],
                    candidate_id, repeat)


def _completed_ids(path: Path, column: str) -> set[int]:
    """Ids already written to a CSV; a row cut off by an interruption is removed first."""

    if not path.exists():
        return set()
    if column == "candidate_id":
        return MP._completed_rows(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return set()
    header, body = rows[0], rows[1:]
    complete = [row for row in body if len(row) == len(header)]
    if len(complete) != len(body):
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(complete)
        temporary.replace(path)
    position = header.index(column)
    return {int(row[position]) for row in complete}


class _Rows:
    """Append rows to a CSV, writing the header only for a new file, flushing after every row."""

    def __init__(self, path: Path):
        self.path = path
        self.new = not path.exists() or path.stat().st_size == 0
        self.handle = path.open("a", newline="", encoding="utf-8")
        self.writer = None

    def write(self, row: dict) -> None:
        if self.writer is None:
            self.writer = csv.DictWriter(self.handle, fieldnames=list(row))
            if self.new:
                self.writer.writeheader()
        self.writer.writerow(row)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def _check_saved(path: Path, info: dict, name: str) -> None:
    if path.exists():
        saved = _read_json(path)
        for key in ("hash_a", "hash_b", "design_hash"):
            if saved.get(key) != info[key]:
                raise RuntimeError(f"{name}: the saved runs come from a different pair or seed sets ({key} differs). "
                                   "Set GRADED_OUTCOMES_OUTPUT_DIR to a new folder to keep them apart.")


# --------------------------------------------------------------------------
# Design (frozen once results exist)
# --------------------------------------------------------------------------

def prepare(blas_threads: int = 3) -> dict:
    """The design every job uses. Refuses to mix results made with different weights or set counts."""

    path = OUTPUT_DIR / "settings.json"
    fixed = {
        "version": 1, "model": MODEL,
        "base_graph_config": {a: asdict(S.graph_config(a, MODEL)) for a in ALGORITHMS},
        "weights": load_weights(), "weights_path": str(WEIGHTS_PATH.relative_to(REPO_ROOT)),
        "n_test": N_TEST, "pilot_sets": PILOT_SETS, "pilot_pool": PILOT_POOL, "collection_sets": COLLECTION_SETS,
        "repeat_sets": REPEAT_SETS, "repeats": REPEATS,
        "seed_bases": {"pair": PAIR_SEED_BASE, "sampling": SAMPLING_SEED_BASE, "solver": SOLVER_SEED_BASE,
                       "test_vertices": TEST_SEED_BASE},
        "solver": {**S.SGM, "goat_regularizer": S.GOAT_REGULARIZER[MODEL], "runs_per_set": 1},
        "uniqueness_cap": {a: MP.UNIQUENESS_CAP[a] for a in ALGORITHMS}, "blas_threads": blas_threads,
    }
    fixed = json.loads(json.dumps(fixed))
    if path.exists():
        saved = _read_json(path)
        differs = [key for key in fixed if saved.get(key) != fixed[key]]
        if differs:
            raise ValueError(f"The saved design in {path} differs ({', '.join(differs)}). Set "
                             "GRADED_OUTCOMES_OUTPUT_DIR to a new folder to keep the results apart.")
        return {**fixed, "excluded_pair_seeds": saved["excluded_pair_seeds"]}
    design = {**fixed, "excluded_pair_seeds": sorted(known_pair_seeds())}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(path, {**design, "written_at": _now(), "environment": collect_environment(blas_threads)})
    return design


# --------------------------------------------------------------------------
# Pilot: one pair per rho, a ladder of budgets, PILOT_SETS random sets each
# --------------------------------------------------------------------------

def pilot_path(algorithm: str, rho: float, budget: int) -> Path:
    return OUTPUT_DIR / "pilot" / algorithm / f"rho{rho:.2f}_k{budget:03d}.csv"


def describe_pilot(task) -> str:
    algorithm, rho, budget = task
    return f"pilot {LABELS[algorithm]} rho = {rho:.2f}, k = {budget}"


def pilot_job(task, design: dict) -> str:
    algorithm, rho, k = task
    path = pilot_path(algorithm, rho, k)
    target = design["pilot_sets"]
    done = _completed_ids(path, "candidate_id")
    if len(done) >= target:
        return f"{describe_pilot(task)}: already done"
    started = time.perf_counter()
    with thread_limit(design["blas_threads"]):
        seed, A, B, truth, test, candidates, info = _pair(algorithm, rho, "pilot", 0, design)
        rng = np.random.default_rng([SAMPLING_SEED_BASE, seed, k, MP.ALGORITHM_CODES[algorithm],
                                     PURPOSE_CODES["pilot"]])
        pool = draw_sets(rng, candidates, k, design["pilot_pool"])
        features, z, stats, score, percentile = _scored(A, algorithm, pool, design["weights"][algorithm])
        info = {**info, "algorithm": algorithm, "rho": rho, "budget": k, "status": "running",
                "design_hash": hashlib.sha256(pool[:target].tobytes() + test.tobytes()).hexdigest()[:16],
                "feature_stats": stats, "weights": design["weights"][algorithm]}
        path.parent.mkdir(parents=True, exist_ok=True)
        info_path = path.with_suffix(".json")
        _check_saved(info_path, info, describe_pilot(task))
        write_json(info_path, info)
        rows, written = _Rows(path), 0
        try:
            for j in range(target):
                if j in done:
                    continue
                seeds = pool[j]
                out = run_once(algorithm, (A, B, truth), seeds, test,
                               _solver_seed(seed, k, algorithm, "pilot", j, 0))
                rows.write({"algorithm": algorithm, "model": MODEL, "rho": rho, "budget": k, "pair_index": 0,
                            "pair_seed": seed, "candidate_id": j, "seed_vertices": json.dumps([int(v) for v in seeds]),
                            **{f: float(features[f][j]) for f in CORE}, **{f"z_{f}": float(z[f][j]) for f in CORE},
                            "score": float(score[j]), "score_percentile": float(percentile[j]),
                            "solver_seed": _solver_seed(seed, k, algorithm, "pilot", j, 0), **out})
                written += 1
        finally:
            rows.close()
        write_json(info_path, {**info, "status": "complete", "finished_at": _now()})
    return f"{describe_pilot(task)}: {written} runs in {(time.perf_counter() - started) / 60:.1f} min"


def pilot_tasks(algorithms=ALGORITHMS, rhos=PILOT_RHOS, budgets: dict | None = None) -> list:
    budgets = {**PILOT_BUDGETS, **(budgets or {})}
    return [(a, float(rho), int(k)) for a in algorithms for rho in rhos for k in budgets[a]]


def run_pilot(algorithms=ALGORITHMS, rhos=PILOT_RHOS, budgets: dict | None = None, jobs: int = 3,
              blas_threads: int = 3) -> bool:
    """Run every pilot setting that is not complete yet. Returns False if stopped."""

    design = prepare(blas_threads)
    tasks = [t for t in pilot_tasks(algorithms, rhos, budgets)
             if len(_completed_ids(pilot_path(*t), "candidate_id")) < design["pilot_sets"]]
    if not tasks:
        print(f"The pilot is complete. Results are in {OUTPUT_DIR / 'pilot'}")
        return True

    def seconds(task):
        return (design["pilot_sets"] - len(_completed_ids(pilot_path(*task), "candidate_id"))) * SECONDS_PER_RUN[task[0]]

    tasks.sort(key=seconds, reverse=True)
    print(f"{_clock()}  Pilot: {len(tasks)} settings, {jobs} at a time, about "
          f"{sum(seconds(t) for t in tasks) / jobs / 60:.0f} min (a rough guess). One line per finished setting.",
          flush=True)
    return _run(tasks, pilot_job, design, describe_pilot, seconds, jobs)


def pilot_status(algorithms=ALGORITHMS, rhos=PILOT_RHOS, budgets: dict | None = None) -> pd.DataFrame:
    rows = []
    for a, rho, k in pilot_tasks(algorithms, rhos, budgets):
        rows.append({"algorithm": LABELS[a], "rho": rho, "budget": k,
                     "runs done": len(_completed_ids(pilot_path(a, rho, k), "candidate_id"))})
    table = pd.DataFrame(rows)
    return table.pivot_table(index=["algorithm", "rho"], columns="budget", values="runs done", aggfunc="first")


# --------------------------------------------------------------------------
# Collection: COLLECTION_PAIRS pairs x COLLECTION_SETS random sets in the chosen setting
# --------------------------------------------------------------------------

def collection_dir(algorithm: str, rho: float, budget: int, pair_index: int) -> Path:
    return OUTPUT_DIR / "collection" / algorithm / f"rho{rho:.2f}_k{budget:03d}" / f"pair_{pair_index:03d}"


def describe_collection(task) -> str:
    algorithm, rho, budget, pair_index = task
    return f"{LABELS[algorithm]} rho = {rho:.2f}, k = {budget}, pair {pair_index}"


def collection_job(task, design: dict) -> str:
    algorithm, rho, k, i = task
    folder = collection_dir(algorithm, rho, k, i)
    sets_path, repeats_path, info_path = folder / "sets.csv", folder / "repeats.csv", folder / "pair.json"
    count, repeat_sets, repeats = design["collection_sets"], design["repeat_sets"], design["repeats"]
    done = _completed_ids(sets_path, "candidate_id")
    done_repeats = _completed_ids(repeats_path, "run_id")
    if len(done) >= count and len(done_repeats) >= repeat_sets * repeats:
        return f"{describe_collection(task)}: already done"
    started = time.perf_counter()
    with thread_limit(design["blas_threads"]):
        seed, A, B, truth, test, candidates, info = _pair(algorithm, rho, "collection", i, design)
        rng = np.random.default_rng([SAMPLING_SEED_BASE, seed, k, MP.ALGORITHM_CODES[algorithm],
                                     PURPOSE_CODES["collection"]])
        sets = draw_sets(rng, candidates, k, count)
        features, z, stats, score, percentile = _scored(A, algorithm, sets, design["weights"][algorithm])
        info = {**info, "algorithm": algorithm, "rho": rho, "budget": k, "pair_index": i, "status": "running",
                "design_hash": hashlib.sha256(sets.tobytes() + test.tobytes()).hexdigest()[:16],
                "feature_stats": stats, "weights": design["weights"][algorithm]}
        folder.mkdir(parents=True, exist_ok=True)
        _check_saved(info_path, info, describe_collection(task))
        write_json(info_path, {**info, "started_at": _now()})
        context = build_graph_context(A, compute_seed_features=False)
        pair = (A, B, truth)
        written = 0
        rows = _Rows(sets_path)
        try:
            for j, seeds in enumerate(sets):
                if j in done:
                    continue
                solver_seed = _solver_seed(seed, k, algorithm, "collection", j, 0)
                out = run_once(algorithm, pair, seeds, test, solver_seed)
                rows.write({"algorithm": algorithm, "model": MODEL, "rho": rho, "budget": k, "pair_index": i,
                            "pair_seed": seed, "candidate_id": j, "seed_vertices": json.dumps([int(v) for v in seeds]),
                            **{f: float(features[f][j]) for f in CORE}, **{f"z_{f}": float(z[f][j]) for f in CORE},
                            "score": float(score[j]), "score_percentile": float(percentile[j]),
                            "solver_seed": solver_seed, **out,
                            **fixed_pair_seed_set_features(context, seeds),
                            **signature_robustness_features(context, seeds)})
                written += 1
        finally:
            rows.close()
        rows = _Rows(repeats_path)
        try:
            for j in range(repeat_sets):
                for r in range(1, repeats + 1):
                    run_id = 100 * j + r
                    if run_id in done_repeats:
                        continue
                    solver_seed = _solver_seed(seed, k, algorithm, "collection", j, r)
                    out = run_once(algorithm, pair, sets[j], test, solver_seed)
                    rows.write({"algorithm": algorithm, "rho": rho, "budget": k, "pair_index": i, "candidate_id": j,
                                "repeat": r, "run_id": run_id, "solver_seed": solver_seed, **out})
                    written += 1
        finally:
            rows.close()
        write_json(info_path, {**info, "status": "complete", "finished_at": _now()})
    return f"{describe_collection(task)}: {written} runs in {(time.perf_counter() - started) / 60:.0f} min"


def _valid_choice(choice: dict | None) -> dict:
    return {a: (float(c[0]), int(c[1])) for a, c in (choice or {}).items() if c is not None and a in ALGORITHMS}


def collection_tasks(choice: dict, n_pairs: int = COLLECTION_PAIRS) -> list:
    choice = _valid_choice(choice)
    return [(a, rho, k, i) for a, (rho, k) in choice.items() for i in range(n_pairs)]


def _collection_left(task, design: dict) -> int:
    folder = collection_dir(*task)
    return (max(design["collection_sets"] - len(_completed_ids(folder / "sets.csv", "candidate_id")), 0) +
            max(design["repeat_sets"] * design["repeats"] - len(_completed_ids(folder / "repeats.csv", "run_id")), 0))


def run_collection(choice: dict, n_pairs: int = COLLECTION_PAIRS, jobs: int = 3, blas_threads: int = 3) -> bool:
    """Collect COLLECTION_SETS random sets on each of n_pairs pairs, for every algorithm with a chosen setting
    {algorithm: (rho, budget)}. An algorithm whose choice is None is skipped. Returns False if stopped."""

    for algorithm in ALGORITHMS:
        if algorithm in (choice or {}) and choice[algorithm] is None:
            print(f"{LABELS[algorithm]}: no setting chosen (the pilot found none where runs end part-way), skipped.")
    design = prepare(blas_threads)
    tasks = [t for t in collection_tasks(choice, n_pairs) if _collection_left(t, design) > 0]
    if not tasks:
        print(f"The collection is complete. Results are in {OUTPUT_DIR / 'collection'}")
        return True

    def seconds(task):
        return _collection_left(task, design) * SECONDS_PER_RUN[task[0]]

    tasks.sort(key=seconds, reverse=True)
    print(f"{_clock()}  Collection: {len(tasks)} pairs, {jobs} at a time, about "
          f"{sum(seconds(t) for t in tasks) / jobs / 3600:.1f} h (a rough guess). One line per finished pair.",
          flush=True)
    return _run(tasks, collection_job, design, describe_collection, seconds, jobs)


def collection_status(choice: dict, n_pairs: int = COLLECTION_PAIRS) -> pd.DataFrame:
    rows = []
    for a, rho, k, i in collection_tasks(choice, n_pairs):
        folder = collection_dir(a, rho, k, i)
        rows.append({"algorithm": LABELS[a], "rho": rho, "budget": k, "pair": i,
                     "sets done": len(_completed_ids(folder / "sets.csv", "candidate_id")),
                     "repeat runs done": len(_completed_ids(folder / "repeats.csv", "run_id"))})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(["algorithm", "rho", "budget", "pair"])


# --------------------------------------------------------------------------
# Running jobs in parallel
# --------------------------------------------------------------------------

def _ignore_interrupts() -> None:
    """Workers leave Ctrl-C to the main process, which stops them."""

    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _run(tasks: list, job, design: dict, describe, seconds, jobs: int) -> bool:
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
                                       initializer=_ignore_interrupts)
            futures = {pool.submit(job, task, design): task for task in tasks}
            for n, future in enumerate(as_completed(futures), 1):
                try:
                    message = future.result()
                except Exception as error:  # noqa: BLE001 - report every failure, keep going
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
        raise RuntimeError(f"Failed: {', '.join(failures)}. Rerun to retry them (finished runs are kept).")
    print(f"{_clock()}  Done. Results are in {OUTPUT_DIR}", flush=True)
    return True


# --------------------------------------------------------------------------
# Pilot analysis
# --------------------------------------------------------------------------

def read_pilot() -> pd.DataFrame:
    files = sorted((OUTPUT_DIR / "pilot").glob("*/rho*_k*.csv"))
    frames = [pd.read_csv(path) for path in files if path.stat().st_size > 0]
    if not frames:
        raise FileNotFoundError(f"No pilot results yet in {OUTPUT_DIR / 'pilot'}")
    return pd.concat(frames, ignore_index=True)


def pilot_table(data: pd.DataFrame) -> pd.DataFrame:
    """Per algorithm, rho and budget: how the single-run test accuracies spread."""

    rows = []
    for (a, rho, k), part in data.groupby(["algorithm", "rho", "budget"]):
        ok = part.dropna(subset=["accuracy_test"])
        acc = ok["accuracy_test"]
        rows.append({"algorithm": a, "rho": float(rho), "budget": int(k), "runs": len(ok),
                     "failed runs": len(part) - len(ok), "mean": acc.mean(), "sd": acc.std(ddof=1),
                     "q10": acc.quantile(0.1), "median": acc.median(), "q90": acc.quantile(0.9),
                     "share part-way": acc.between(*MIDDLE).mean(), "non-seed accuracy": ok["accuracy"].mean(),
                     "Spearman with score": _spearman(ok["score"], acc),
                     "seconds per run": ok["runtime_seconds"].mean()})
    return pd.DataFrame(rows)


def recommend(table: pd.DataFrame, min_share: float = MIN_MIDDLE_SHARE) -> dict:
    """For each algorithm, the setting where at least min_share of runs end part-way (test accuracy 0.2-0.8) and
    the mean is closest to 0.5; None if no setting qualifies."""

    choice = {}
    for a in ALGORITHMS:
        part = table[(table["algorithm"] == a) & (table["runs"] >= min(10, PILOT_SETS))]
        good = part[part["share part-way"] >= min_share]
        if good.empty:
            choice[a] = None
            continue
        best = good.assign(distance=(good["mean"] - 0.5).abs()).sort_values(["distance", "sd"],
                                                                         ascending=[True, False]).iloc[0]
        choice[a] = (float(best["rho"]), int(best["budget"]))
    return choice


def describe_choice(choice: dict) -> str:
    parts = []
    for a in ALGORITHMS:
        if a in (choice or {}):
            c = choice[a]
            parts.append(f"{LABELS[a]}: " + ("none qualifies" if c is None else f"rho = {c[0]:.2f}, k = {c[1]}"))
    return "; ".join(parts)


def show_pilot(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["algorithm"] = out["algorithm"].map(LABELS)
    out = out.set_index(["algorithm", "rho", "budget"])
    for column in ("mean", "sd", "q10", "median", "q90", "share part-way", "non-seed accuracy", "Spearman with score"):
        out[column] = out[column].round(2)
    out["seconds per run"] = out["seconds per run"].round(1)
    return out


def plot_pilot(data: pd.DataFrame, choice: dict | None = None):
    """Single-run test accuracy by budget: one row per algorithm, one column per rho."""

    import matplotlib.pyplot as plt

    algorithms = [a for a in ALGORITHMS if a in set(data["algorithm"])]
    rhos = sorted(data["rho"].unique())
    rng = np.random.default_rng(0)
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5}):
        fig, axes = plt.subplots(len(algorithms), len(rhos), figsize=(4.6 * len(rhos), 3.6 * len(algorithms)),
                                 squeeze=False, sharey=True)
        for r, a in enumerate(algorithms):
            budgets = sorted(data.loc[data["algorithm"] == a, "budget"].unique())
            for c, rho in enumerate(rhos):
                ax = axes[r, c]
                ax.axhspan(*MIDDLE, color=BAND, zorder=0, linewidth=0)
                part = data[(data["algorithm"] == a) & (data["rho"] == rho)].dropna(subset=["accuracy_test"])
                for x, k in enumerate(budgets):
                    values = part.loc[part["budget"] == k, "accuracy_test"].to_numpy()
                    if not len(values):
                        continue
                    ax.scatter(x + rng.uniform(-0.18, 0.18, len(values)), values, s=14, color=BLUE, alpha=0.55,
                               linewidths=0, zorder=2)
                    ax.plot([x - 0.3, x + 0.3], [np.median(values)] * 2, color=INK, linewidth=2, zorder=3)
                    if choice and choice.get(a) is not None and np.isclose(choice[a][0], rho) and choice[a][1] == k:
                        ax.add_patch(plt.Rectangle((x - 0.42, -0.01), 0.84, 1.02, fill=False, edgecolor=ORANGE,
                                                   linewidth=2, zorder=4))
                        ax.text(x, 1.04, "chosen", ha="center", va="bottom", color=INK, fontsize=8.5)
                ax.set_xticks(range(len(budgets)), [str(k) for k in budgets])
                ax.set_ylim(-0.02, 1.08)
                ax.grid(axis="y", color=GRID, linewidth=0.7)
                ax.set_axisbelow(True)
                ax.set_title(f"{LABELS[a]} · rho = {rho:.2f}", fontsize=10.5)
                if r == len(algorithms) - 1:
                    ax.set_xlabel("seeds (budget)")
                if c == 0:
                    ax.set_ylabel(f"accuracy on the {N_TEST} test vertices")
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markersize=6,
                          label="one run from a random seed set"),
                   Line2D([0], [0], color=INK, linewidth=2, label="median"),
                   Patch(color=BAND, label=f"ends part-way ({MIDDLE[0]}-{MIDDLE[1]})")]
        if choice and any(c is not None for c in choice.values()):
            handles.append(Patch(facecolor="none", edgecolor=ORANGE, linewidth=2, label="chosen for the collection"))
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
                   bbox_to_anchor=(0.5, 1.0))
        fig.suptitle("Pilot: how single runs end at lower edge correlation (ER, 600 vertices, mean degree 16)",
                     y=1.05, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# --------------------------------------------------------------------------
# Collection analysis
# --------------------------------------------------------------------------

def _choice_folders(choice: dict):
    for a, (rho, k) in _valid_choice(choice).items():
        yield a, OUTPUT_DIR / "collection" / a / f"rho{rho:.2f}_k{k:03d}"


def read_collection(choice: dict) -> pd.DataFrame:
    """All collected sets in the chosen settings, with every feature also z-scored within its pair."""

    frames = []
    for _, folder in _choice_folders(choice):
        frames += [pd.read_csv(path) for path in sorted(folder.glob("pair_*/sets.csv")) if path.stat().st_size > 0]
    if not frames:
        raise FileNotFoundError(f"No collected sets yet for {describe_choice(choice)}")
    data = pd.concat(frames, ignore_index=True)
    groups = data.groupby(["algorithm", "pair_index"])
    for column in ALL_FEATURES:
        mean, sd = groups[column].transform("mean"), groups[column].transform("std")
        data[f"z_{column}"] = np.where(sd > 0, (data[column] - mean) / sd.where(sd > 0, 1.0), 0.0)
    return data


def read_repeats(choice: dict) -> pd.DataFrame:
    frames = []
    for _, folder in _choice_folders(choice):
        frames += [pd.read_csv(path) for path in sorted(folder.glob("pair_*/repeats.csv")) if path.stat().st_size > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def algorithms_in(data: pd.DataFrame) -> list[str]:
    return [a for a in ALGORITHMS if a in set(data["algorithm"])]


def outcome_table(data: pd.DataFrame) -> pd.DataFrame:
    """How the outcomes spread on each pair."""

    rows = []
    for (a, i), part in data.groupby(["algorithm", "pair_index"]):
        ok = part.dropna(subset=["accuracy_test"])
        acc = ok["accuracy_test"]
        rows.append({"algorithm": LABELS[a], "pair": int(i), "rho": float(part["rho"].iloc[0]),
                     "budget": int(part["budget"].iloc[0]), "sets": len(ok), "failed runs": len(part) - len(ok),
                     "test accuracy: mean": acc.mean(), "sd": acc.std(ddof=1), "q10": acc.quantile(0.1),
                     "median": acc.median(), "q90": acc.quantile(0.9),
                     "share part-way": acc.between(*MIDDLE).mean(), "non-seed accuracy: mean": ok["accuracy"].mean()})
    return pd.DataFrame(rows).set_index(["algorithm", "pair"]).round(3)


def plot_outcomes(data: pd.DataFrame, outcome: str = "accuracy_test"):
    """Histogram of the single-run outcomes on each pair."""

    import matplotlib.pyplot as plt

    algorithms = algorithms_in(data)
    pairs = sorted(data["pair_index"].unique())
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5}):
        fig, axes = plt.subplots(len(algorithms), len(pairs), figsize=(4.3 * len(pairs), 2.9 * len(algorithms)),
                                 squeeze=False, sharex=True)
        bins = np.linspace(0, 1, 41)
        for r, a in enumerate(algorithms):
            for c, i in enumerate(pairs):
                ax = axes[r, c]
                values = data.loc[(data["algorithm"] == a) & (data["pair_index"] == i), outcome].dropna()
                ax.axvspan(*MIDDLE, color=BAND, zorder=0, linewidth=0)
                ax.hist(values, bins=bins, color=BLUE, edgecolor="white", linewidth=0.6, zorder=2)
                ax.grid(axis="y", color=GRID, linewidth=0.7)
                ax.set_axisbelow(True)
                share = values.between(*MIDDLE).mean() if len(values) else float("nan")
                ax.set_title(f"{LABELS[a]} · pair {i}: {len(values):,} runs, {share:.0%} part-way", fontsize=10)
                if r == len(algorithms) - 1:
                    ax.set_xlabel("accuracy on the test vertices" if outcome == "accuracy_test" else outcome)
                if c == 0:
                    ax.set_ylabel("runs")
        fig.tight_layout()
    return fig


def _feature_columns(kind: str) -> list[str]:
    return [f"z_{f}" for f in CORE] if kind == "core" else [f"z_{c}" for c in ALL_FEATURES]


def refit(data: pd.DataFrame, outcome: str = "accuracy_test", kind: str = "all"):
    """Weights refitted on the outcome (ridge regression on within-pair z-scores, alpha = 1).

    Returns held-out predictions (each pair predicted by a fit on the other pairs; NaN with a single pair) and each
    algorithm's coefficients from a fit on all its pairs (change in the outcome per SD of a feature, others fixed)."""

    from sklearn.linear_model import Ridge

    columns = _feature_columns(kind)
    predictions = pd.Series(np.nan, index=data.index, dtype=float)
    coefficients = {}
    for a, part in data.groupby("algorithm"):
        part = part.dropna(subset=[outcome, *columns])
        if part.empty:
            continue
        pairs = sorted(part["pair_index"].unique())
        if len(pairs) > 1:
            for held in pairs:
                train, test = part[part["pair_index"] != held], part[part["pair_index"] == held]
                model = Ridge(alpha=1.0).fit(train[columns].to_numpy(), train[outcome].to_numpy())
                predictions.loc[test.index] = model.predict(test[columns].to_numpy())
        model = Ridge(alpha=1.0).fit(part[columns].to_numpy(), part[outcome].to_numpy())
        coefficients[a] = pd.Series(model.coef_, index=[c[2:] for c in columns])
    return predictions, coefficients


RANKERS = {"frozen score": "score", "degree": "z_degree", "harmonic": "z_harmonic", "pair mass": "z_pair_mass",
           "uniqueness": "z_uniqueness", "coverage": "z_nonseed_covered_fraction",
           "refit, 4 features": "refit_core", "refit, 11 features": "refit_all"}


def add_refits(data: pd.DataFrame, outcome: str = "accuracy_test") -> pd.DataFrame:
    data = data.copy()
    data["refit_core"], _ = refit(data, outcome, "core")
    data["refit_all"], _ = refit(data, outcome, "all")
    return data


def ranking_table(data: pd.DataFrame, outcome: str = "accuracy_test") -> pd.DataFrame:
    """Per pair: Spearman of each ranking with the outcome, and the mean outcome of the top and bottom tenth."""

    data = add_refits(data.dropna(subset=[outcome]), outcome)
    rows = []
    for (a, i), part in data.groupby(["algorithm", "pair_index"]):
        row = {"algorithm": LABELS[a], "pair": f"pair {int(i)}", "sets": len(part)}
        for label, column in RANKERS.items():
            row[f"Spearman: {label}"] = _spearman(part[column], part[outcome])
        tenth = max(1, round(0.1 * len(part)))
        row["mean: all sets"] = part[outcome].mean()
        row["mean: bottom 10% by frozen score"] = part.nsmallest(tenth, "score")[outcome].mean()
        row["mean: top 10% by frozen score"] = part.nlargest(tenth, "score")[outcome].mean()
        row["mean: top 10% by degree"] = part.nlargest(tenth, "z_degree")[outcome].mean()
        refitted = part.dropna(subset=["refit_all"])
        row["mean: top 10% by refit (11)"] = (refitted.nlargest(tenth, "refit_all")[outcome].mean()
                                              if len(refitted) else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows).set_index(["algorithm", "pair"])


def show_ranking(table: pd.DataFrame) -> pd.DataFrame:
    """The ranking table with each algorithm's mean over pairs added."""

    means = table.groupby(level="algorithm", sort=False).mean(numeric_only=True)
    means.index = pd.MultiIndex.from_tuples([(a, "mean over pairs") for a in means.index], names=table.index.names)
    parts = []
    for a in table.index.get_level_values("algorithm").unique():
        parts += [table.xs(a, level="algorithm", drop_level=False), means.xs(a, level="algorithm", drop_level=False)]
    return pd.concat(parts).round(2)


def weights_table(data: pd.DataFrame, outcome: str = "accuracy_test") -> pd.DataFrame:
    """The frozen weights (log-odds of success per SD) next to the weights refitted on the outcome (outcome per SD)."""

    frozen = load_weights()
    _, core = refit(data.dropna(subset=[outcome]), outcome, "core")
    _, full = refit(data.dropna(subset=[outcome]), outcome, "all")
    rows = []
    for a in algorithms_in(data):
        names = MP.core_columns(a)                       # core feature -> its column among the 11
        core_of = {column: f for f, column in names.items()}
        for column in ALL_FEATURES:
            f = core_of.get(column)
            rows.append({"algorithm": LABELS[a], "feature": FEATURE_LABELS[column],
                         "frozen weight (success, rho 0.5)": frozen[a][f] if f else np.nan,
                         "refit, 4 features": core[a][f] if (f and a in core) else np.nan,
                         "refit, 11 features": full[a][column] if a in full else np.nan})
    return pd.DataFrame(rows).set_index(["algorithm", "feature"]).round(3)


def repeat_table(data: pd.DataFrame, repeats: pd.DataFrame, outcome: str = "accuracy_test") -> pd.DataFrame:
    """How much the same seed set's outcome changes with the solver seed alone (same A, B and seeds)."""

    if repeats.empty:
        return pd.DataFrame()
    rows = []
    for (a, i), part in repeats.groupby(["algorithm", "pair_index"]):
        first = data[(data["algorithm"] == a) & (data["pair_index"] == i)][["candidate_id", outcome]]
        runs = pd.concat([first[first["candidate_id"].isin(part["candidate_id"])].assign(repeat=0),
                          part[["candidate_id", "repeat", outcome]]], ignore_index=True).dropna(subset=[outcome])
        wide = runs.pivot_table(index="candidate_id", columns="repeat", values=outcome, aggfunc="first").dropna()
        all_sets = data.loc[(data["algorithm"] == a) & (data["pair_index"] == i), outcome].dropna()
        within = float(wide.var(axis=1, ddof=1).mean()) if len(wide) else float("nan")
        rows.append({"algorithm": LABELS[a], "pair": int(i), "sets run 3 times": len(wide),
                     "SD across all sets": float(all_sets.std(ddof=1)),
                     "SD across runs of the same set": float(np.sqrt(within)),
                     "share of variance from solver seed": within / float(all_sets.var(ddof=1)),
                     "run-to-run Spearman": _spearman(wide[0], wide[1]) if {0, 1} <= set(wide.columns) else np.nan})
    return pd.DataFrame(rows).set_index(["algorithm", "pair"]).round(3)


def _decile_means(x: np.ndarray, y: np.ndarray):
    order = np.argsort(x, kind="stable")
    chunks = np.array_split(order, 10)
    centers = np.array([x[c].mean() for c in chunks])
    means = np.array([y[c].mean() for c in chunks])
    halfwidth = np.array([1.96 * y[c].std(ddof=1) / np.sqrt(len(c)) if len(c) > 1 else 0.0 for c in chunks])
    return centers, means, halfwidth


def plot_scores(data: pd.DataFrame, algorithm: str, outcome: str = "accuracy_test"):
    """Outcome against three rankings (frozen score, degree, refit on 11 features from the other pairs), per pair."""

    import matplotlib.pyplot as plt

    part = add_refits(data[data["algorithm"] == algorithm].dropna(subset=[outcome]), outcome)
    pairs = sorted(part["pair_index"].unique())
    rows = [("score", "frozen score (fitted for success at rho = 0.5)"), ("z_degree", "degree alone (z-score)")]
    if part["refit_all"].notna().any():
        rows.append(("refit_all", "refitted on this outcome (11 features, other pairs)"))
    label = (f"accuracy on the {N_TEST} test vertices" if outcome == "accuracy_test" else "non-seed accuracy")
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5}):
        fig, axes = plt.subplots(len(rows), len(pairs), figsize=(4.6 * len(pairs), 3.3 * len(rows)),
                                 squeeze=False, sharey=True)
        for r, (column, name) in enumerate(rows):
            for c, i in enumerate(pairs):
                ax = axes[r, c]
                sub = part[part["pair_index"] == i].dropna(subset=[column])
                x, y = sub[column].to_numpy(float), sub[outcome].to_numpy(float)
                ax.scatter(x, y, s=8, color=BLUE, alpha=0.25, linewidths=0, rasterized=True, zorder=2)
                if len(sub) >= 20:
                    centers, means, halfwidth = _decile_means(x, y)
                    ax.errorbar(centers, means, yerr=halfwidth, color=ORANGE, linewidth=2, marker="o", markersize=4,
                                capsize=2, zorder=3)
                ax.grid(color=GRID, linewidth=0.6)
                ax.set_axisbelow(True)
                ax.set_title(f"pair {i} · Spearman {_spearman(x, y):.2f}", fontsize=10)
                ax.set_xlabel(name)
                if c == 0:
                    ax.set_ylabel(label)
        rho, k = float(part["rho"].iloc[0]), int(part["budget"].iloc[0])
        fig.suptitle(f"{LABELS[algorithm]} · ER, 600 vertices, mean degree 16, rho = {rho:.2f}, k = {k}: "
                     "one dot per random seed set; orange = mean in tenths of the x-axis, with 95% intervals",
                     fontsize=11.5)
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Rules at low rho: round 1's rules and best of N against random seed sets
# --------------------------------------------------------------------------

def _rules_fixed() -> dict:
    return json.loads(json.dumps({
        "version": 1,
        "budgets": {a: {f"{rho:.2f}": list(ks) for rho, ks in RULE_BUDGETS[a].items()} for a in ALGORITHMS},
        "random_sets": RULE_RANDOM_SETS, "pool": RULE_POOL, "runs": RULE_RUNS,
        "rules": {a: list(RULES[a]) for a in ALGORITHMS}, "screens": list(SCREENS),
        "pool_factor": S.POOL_FACTOR, "hub_count": S.HUB_COUNT,
    }))


def _graded_pair_seeds() -> set[int]:
    """Pair seeds the pilot and the collection used; the rules part never reuses them."""

    seeds = set()
    paths = [*(OUTPUT_DIR / "pilot").glob("*/rho*_k*.json"), *(OUTPUT_DIR / "collection").glob("*/*/pair_*/pair.json")]
    for path in paths:
        try:
            seeds.add(int(_read_json(path)["pair_seed"]))
        except (KeyError, ValueError, OSError):
            pass
    return seeds


def rules_prepare(blas_threads: int = 3) -> dict:
    """The rules design on top of the main one (the same frozen weights, solver settings and test vertices per pair),
    frozen in rules/settings.json. More pairs or fewer rhos later are fine; other changes need a new folder."""

    base = prepare(blas_threads)
    path = OUTPUT_DIR / "rules" / "settings.json"
    fixed = _rules_fixed()
    if path.exists():
        saved = _read_json(path)
        differs = [key for key in fixed if saved.get(key) != fixed[key]]
        if differs:
            raise ValueError(f"The saved rules design in {path} differs ({', '.join(differs)}). Set "
                             "GRADED_OUTCOMES_OUTPUT_DIR to a new folder to keep the results apart.")
        excluded = saved["excluded_pair_seeds"]
    else:
        excluded = sorted(set(base["excluded_pair_seeds"]) | _graded_pair_seeds())
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {**fixed, "excluded_pair_seeds": excluded, "written_at": _now()})
    return {**base, **fixed, "excluded_pair_seeds": excluded}


def rule_ordering(rule: str, graph: GraphA, k_max: int, rng: np.random.Generator, exclude) -> np.ndarray:
    """Round 1's ordering for this rule (strategies.ordering), never using the excluded vertices."""

    exclude = np.asarray(exclude, dtype=int)
    if rule == "top_degree":
        order = graph.degree_order(rng)
        return order[~np.isin(order, exclude)][:k_max]
    if rule in TARGETS:
        return degree_first(graph, k_max, rng, rule, pool_factor=S.POOL_FACTOR, exclude=exclude)
    if rule == "hub_free_distinguishing":
        hubs = graph.degree_order(rng)[:S.HUB_COUNT]
        return degree_first(graph, k_max, rng, "distinguishing", pool_factor=S.POOL_FACTOR,
                            exclude=np.union1d(exclude, hubs))
    raise ValueError(f"Unknown rule: {rule}")


def rules_dir(algorithm: str, rho: float, pair_index: int) -> Path:
    return OUTPUT_DIR / "rules" / algorithm / f"rho{rho:.2f}" / f"pair_{pair_index:03d}"


def describe_rules(task) -> str:
    algorithm, rho, pair_index = task
    return f"rules {LABELS[algorithm]} rho = {rho:.2f}, pair {pair_index}"


def _rule_budgets(design: dict, algorithm: str, rho: float) -> list[int]:
    return [int(k) for k in design["budgets"][algorithm][f"{rho:.2f}"]]


def _rule_runs_planned(design: dict, algorithm: str, rho: float) -> int:
    per_budget = (design["random_sets"] +
                  (len(design["rules"][algorithm]) + len(design["screens"])) * design["runs"][algorithm])
    return per_budget * len(_rule_budgets(design, algorithm, rho))


def rule_plan(algorithm: str, rho: float, pair_seed: int, A: np.ndarray, test: np.ndarray, candidates: np.ndarray,
              design: dict):
    """Every run on one pair, in order: per budget, the random sets, then round 1's rules, then the best-of sets.

    Each run carries its set and the set's core features, z-scores and frozen score against the budget's pool of
    random sets. Also returns the pool's feature means and SDs per budget."""

    graph = GraphA(A)
    weights = design["weights"][algorithm]
    budgets = _rule_budgets(design, algorithm, rho)
    rules, repeats = design["rules"][algorithm], design["runs"][algorithm]
    orderings = {}
    for rule in rules:   # nested across budgets; the same for FAQ and GOAT (the rng depends on the pair and rule)
        rng = np.random.default_rng([SAMPLING_SEED_BASE, pair_seed, PURPOSE_CODES["rules"], STRATEGY_CODES[rule]])
        orderings[rule] = rule_ordering(rule, graph, max(budgets), rng, test)
    plan, pool_stats = [], {}
    for k in budgets:
        rng = np.random.default_rng([SAMPLING_SEED_BASE, pair_seed, k, PURPOSE_CODES["rules"]])
        pool = draw_sets(rng, candidates, k, design["pool"])          # the same random sets for FAQ and GOAT
        features, _, stats, score, _ = _scored(A, algorithm, pool, weights, graph=graph)
        pool_stats[str(k)] = stats

        def add(method, code, seeds, raw, pool_index, runs):
            z = {f: (raw[f] - stats[f]["mean"]) / stats[f]["sd"] if stats[f]["sd"] > 0 else 0.0 for f in CORE}
            value = float(sum(weights[f] * z[f] for f in CORE))
            share = float(100 * np.mean(score <= value + 1e-12))
            for repeat in range(runs):
                plan.append({"budget": k, "method": method, "set_code": code, "pool_index": pool_index,
                             "repeat": repeat, "seeds": np.asarray(seeds, dtype=np.int64),
                             **{f: float(raw[f]) for f in CORE}, **{f"z_{f}": float(z[f]) for f in CORE},
                             "score": value, "score_percentile": share})

        for j in range(design["random_sets"]):
            add("random", j, pool[j], {f: features[f][j] for f in CORE}, j, 1)
        rule_sets = np.sort(np.stack([orderings[rule][:k] for rule in rules]), axis=1)
        rule_features = _core_features(graph, algorithm, rule_sets)
        for r, rule in enumerate(rules):
            add(rule, 1000 + STRATEGY_CODES[rule], rule_sets[r], {f: rule_features[f][r] for f in CORE}, -1, repeats)
        picks = {"best_100": int(np.argmax(score[:100])), "best_1000": int(np.argmax(score)),
                 "degree_1000": int(np.argmax(features["degree"]))}
        for s, name in enumerate(design["screens"]):
            j = picks[name]
            add(name, 2000 + s, pool[j], {f: features[f][j] for f in CORE}, j, repeats)
    return plan, pool_stats


def rules_job(task, design: dict) -> str:
    algorithm, rho, i = task
    folder = rules_dir(algorithm, rho, i)
    runs_path, info_path = folder / "runs.csv", folder / "pair.json"
    planned = _rule_runs_planned(design, algorithm, rho)
    done = _completed_ids(runs_path, "run_id")
    if len(done) >= planned:
        return f"{describe_rules(task)}: already done"
    started = time.perf_counter()
    with thread_limit(design["blas_threads"]):
        seed, A, B, truth, test, candidates, info = _pair(algorithm, rho, "rules", i, design)
        plan, stats = rule_plan(algorithm, rho, seed, A, test, candidates, design)
        if len(plan) != planned:
            raise RuntimeError(f"{describe_rules(task)}: expected {planned} runs, planned {len(plan)}")
        digest = hashlib.sha256(b"".join(item["seeds"].tobytes() for item in plan) + test.tobytes()).hexdigest()[:16]
        info = {**info, "algorithm": algorithm, "rho": rho, "pair_index": i,
                "budgets": _rule_budgets(design, algorithm, rho), "status": "running", "design_hash": digest,
                "feature_stats": stats, "weights": design["weights"][algorithm]}
        folder.mkdir(parents=True, exist_ok=True)
        _check_saved(info_path, info, describe_rules(task))
        write_json(info_path, info)
        pair = (A, B, truth)
        rows, written = _Rows(runs_path), 0
        try:
            for run_id, item in enumerate(plan):
                if run_id in done:
                    continue
                solver_seed = _solver_seed(seed, item["budget"], algorithm, "rules", item["set_code"], item["repeat"])
                out = run_once(algorithm, pair, item["seeds"], test, solver_seed)
                rows.write({"algorithm": algorithm, "model": MODEL, "rho": rho, "budget": item["budget"],
                            "pair_index": i, "pair_seed": seed, "method": item["method"],
                            "set_code": item["set_code"], "pool_index": item["pool_index"], "repeat": item["repeat"],
                            "run_id": run_id, "seed_vertices": json.dumps([int(v) for v in item["seeds"]]),
                            **{f: item[f] for f in CORE}, **{f"z_{f}": item[f"z_{f}"] for f in CORE},
                            "score": item["score"], "score_percentile": item["score_percentile"],
                            "solver_seed": solver_seed, **out})
                written += 1
        finally:
            rows.close()
        write_json(info_path, {**info, "status": "complete", "finished_at": _now()})
    return f"{describe_rules(task)}: {written} runs in {(time.perf_counter() - started) / 60:.1f} min"


def rules_tasks(algorithms=ALGORITHMS, rhos=RULE_RHOS, n_pairs: int = RULE_PAIRS) -> list:
    unknown = [rho for rho in rhos if not any(np.isclose(rho, r) for r in RULE_RHOS)]
    if unknown:
        raise ValueError(f"No budgets set for rho = {unknown}; add them to RULE_BUDGETS first.")
    return [(a, float(rho), i) for a in algorithms for rho in rhos for i in range(n_pairs)]


def _rules_left(task, design: dict) -> int:
    planned = _rule_runs_planned(design, task[0], task[1])
    return max(planned - len(_completed_ids(rules_dir(*task) / "runs.csv", "run_id")), 0)


def run_rules(algorithms=ALGORITHMS, rhos=RULE_RHOS, n_pairs: int = RULE_PAIRS, jobs: int = 3,
              blas_threads: int = 3) -> bool:
    """Run random sets, round 1's rules and best of N at low rho on n_pairs new pairs per rho. Returns False if
    stopped."""

    design = rules_prepare(blas_threads)
    tasks = [t for t in rules_tasks(algorithms, rhos, n_pairs) if _rules_left(t, design) > 0]
    if not tasks:
        print(f"The rules part is complete. Results are in {OUTPUT_DIR / 'rules'}")
        return True

    def seconds(task):
        return _rules_left(task, design) * RULE_SECONDS_PER_RUN[task[0]]

    tasks.sort(key=seconds, reverse=True)
    print(f"{_clock()}  Rules: {len(tasks)} pairs, {jobs} at a time, about "
          f"{sum(seconds(t) for t in tasks) / jobs / 60:.0f} min (a rough guess). One line per finished pair.",
          flush=True)
    return _run(tasks, rules_job, design, describe_rules, seconds, jobs)


def rules_status(algorithms=ALGORITHMS, rhos=RULE_RHOS, n_pairs: int = RULE_PAIRS) -> pd.DataFrame:
    fixed = _rules_fixed()
    rows = []
    for a, rho, i in rules_tasks(algorithms, rhos, n_pairs):
        planned = _rule_runs_planned(fixed, a, rho)
        done = len(_completed_ids(rules_dir(a, rho, i) / "runs.csv", "run_id"))
        rows.append({"algorithm": LABELS[a], "rho": rho, "pairs complete": int(done >= planned),
                     "runs done": done, "runs planned": planned})
    return pd.DataFrame(rows).groupby(["algorithm", "rho"], sort=False).sum()


def read_rules() -> pd.DataFrame:
    files = sorted((OUTPUT_DIR / "rules").glob("*/rho*/pair_*/runs.csv"))
    frames = [pd.read_csv(path) for path in files if path.stat().st_size > 0]
    if not frames:
        raise FileNotFoundError(f"No rules results yet in {OUTPUT_DIR / 'rules'}")
    return pd.concat(frames, ignore_index=True)


def rule_means(data: pd.DataFrame, outcome: str = "accuracy_test") -> pd.DataFrame:
    """Mean outcome per algorithm, rho, pair and budget (rows) and method (columns). Random averages its sets; a rule
    or best-of set averages its runs."""

    ok = data.dropna(subset=[outcome])
    table = ok.groupby(["algorithm", "rho", "pair_index", "budget", "method"])[outcome].mean().unstack("method")
    return table[[m for m in METHODS if m in table.columns]]


def _interval(values: pd.Series) -> tuple[float, float, float]:
    """Mean and 95% t-interval of the mean (no interval with fewer than 2 values)."""

    from scipy import stats

    values = values.dropna()
    if len(values) < 2:
        return (float(values.mean()) if len(values) else np.nan), np.nan, np.nan
    mean = float(values.mean())
    half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def rule_table(data: pd.DataFrame, outcome: str = "accuracy_test") -> pd.DataFrame:
    """Per algorithm, rho and method: the mean outcome, and the gain over random sets on the same pairs and budgets.

    Each pair's gain is averaged over the budgets first; the interval is a 95% t-interval over pairs."""

    means = rule_means(data, outcome)
    rows = []
    for (a, rho), part in means.groupby(level=["algorithm", "rho"], sort=False):
        for method in part.columns:
            values = part[method].dropna()
            if values.empty:
                continue
            row = {"algorithm": a, "rho": float(rho), "method": method,
                   "pairs": values.index.get_level_values("pair_index").nunique(), "mean accuracy": values.mean()}
            if method != "random":
                gain = (part[method] - part["random"]).dropna()
                mean, low, high = _interval(gain.groupby(level="pair_index").mean())
                row.update({"gain over random": mean, "low": low, "high": high,
                            "cells above random": f"{int((gain > 0).sum())} of {len(gain)}"})
                if method != "top_degree" and "top_degree" in part.columns:
                    versus = (part[method] - part["top_degree"]).dropna()
                    row["gain over top degree"] = _interval(versus.groupby(level="pair_index").mean())[0]
            rows.append(row)
    return pd.DataFrame(rows)


def show_rule_table(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["algorithm"] = out["algorithm"].map(LABELS)
    out["method"] = out["method"].map(METHOD_LABELS)
    if "low" in out:
        out["95% interval"] = ["" if pd.isna(low) else f"{low:+.3f} to {high:+.3f}"
                               for low, high in zip(out["low"], out["high"])]
    out = out.set_index(["algorithm", "rho", "method"])
    columns = ["pairs", "mean accuracy", "gain over random", "95% interval", "cells above random",
               "gain over top degree"]
    out = out[[c for c in columns if c in out.columns]]
    for column in ("mean accuracy", "gain over random", "gain over top degree"):
        if column in out:
            out[column] = out[column].round(3)
    return out.astype(object).where(out.notna(), "")


def rule_feature_table(data: pd.DataFrame) -> pd.DataFrame:
    """Where each method's sets sit among random sets: the core features' z-scores (SDs of the random pool on the
    same pair and budget) and the frozen score's percentile in the pool, averaged over rhos, pairs and budgets."""

    columns = [f"z_{f}" for f in CORE] + ["score", "score_percentile"]
    sets = data.drop_duplicates(["algorithm", "rho", "pair_index", "budget", "method", "set_code"])
    table = sets.groupby(["algorithm", "method"])[columns].mean().reset_index()
    table["order"] = table["method"].map({m: i for i, m in enumerate(METHODS)})
    table = table.sort_values(["algorithm", "order"])
    table["algorithm"] = table["algorithm"].map(LABELS)
    table["method"] = table["method"].map(METHOD_LABELS)
    names = {f"z_{f}": f"{MP.FEATURE_LABELS[f]} (z)" for f in CORE}
    names.update(score="frozen score", score_percentile="score percentile in the pool")
    return table.drop(columns="order").rename(columns=names).set_index(["algorithm", "method"]).round(2)


def plot_rules(data: pd.DataFrame, algorithm: str, outcome: str = "accuracy_test"):
    """Gain over random seed sets by budget: round 1's rules (top row) and best of N (bottom row), one column per rho.

    Each point is the mean over pairs of (method - random sets' mean) on the same pair and budget, with a 95%
    t-interval over pairs. Under each budget: random sets' mean outcome."""

    import matplotlib.pyplot as plt
    from scipy import stats

    means = rule_means(data[data["algorithm"] == algorithm], outcome)
    rhos = sorted(means.index.get_level_values("rho").unique())
    groups = [("round 1's rules", [m for m in RULES[algorithm] if m in means.columns]),
              ("best of N", [m for m in SCREENS if m in means.columns])]
    label = "test accuracy" if outcome == "accuracy_test" else "non-seed accuracy"
    n_pairs = means.index.get_level_values("pair_index").nunique()
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5}):
        fig, axes = plt.subplots(len(groups), len(rhos), figsize=(4.9 * len(rhos) + 1.6, 3.9 * len(groups)),
                                 squeeze=False, sharey=True)
        for c, rho in enumerate(rhos):
            part = means.xs(rho, level="rho")
            budgets = sorted(part.index.get_level_values("budget").unique())
            base = part["random"].groupby(level="budget").mean()
            for r, (title, methods) in enumerate(groups):
                ax = axes[r, c]
                ax.axhline(0, color=INK2, linewidth=1, zorder=1)
                offsets = np.linspace(-0.22, 0.22, len(methods)) if len(methods) > 1 else np.zeros(1)
                for method, offset in zip(methods, offsets):
                    gain = (part[method] - part["random"]).dropna().groupby(level="budget")
                    mean = gain.mean().reindex(budgets)
                    count = gain.count().reindex(budgets).fillna(0).to_numpy()
                    spread = gain.std(ddof=1).reindex(budgets).to_numpy()
                    with np.errstate(invalid="ignore", divide="ignore"):
                        half = np.where(count > 1, stats.t.ppf(0.975, np.maximum(count - 1, 1)) * spread /
                                        np.sqrt(np.maximum(count, 1)), np.nan)
                    style = METHOD_STYLE[method]
                    ax.errorbar(np.arange(len(budgets)) + offset, mean.to_numpy(), yerr=half, color=style["color"],
                                marker=style["marker"], markersize=9 if style["marker"] == "*" else 6.5,
                                markeredgecolor="white", markeredgewidth=0.6, linewidth=1.3, capsize=2,
                                label=METHOD_LABELS[method], zorder=3)
                ax.set_xticks(range(len(budgets)), [f"{k}\n{base.get(k, np.nan):.2f}" for k in budgets])
                ax.grid(axis="y", color=GRID, linewidth=0.7)
                ax.set_axisbelow(True)
                ax.set_title(f"{title} · rho = {rho:.2f}", fontsize=10.5)
                if c == 0:
                    ax.set_ylabel(f"gain in {label}\nover random sets")
                if r == len(groups) - 1:
                    ax.set_xlabel(f"seeds (top) · random sets' mean {label} (below)")
        for r in range(len(groups)):
            axes[r, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
        fig.suptitle(f"{LABELS[algorithm]} · ER, 600 vertices, mean degree 16: gain over random seed sets "
                     f"(mean over {n_pairs} pairs, 95% intervals)", fontsize=12)
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def _parse_choice(items) -> dict:
    """faq=0.30:150 goat=0.30:70 -> {"faq": (0.30, 150), "goat": (0.30, 70)}"""

    choice = {}
    for item in items or []:
        algorithm, cell = item.split("=")
        rho, k = cell.split(":")
        choice[algorithm.strip()] = (float(rho), int(k))
    return choice


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", action="store_true", help="run the pilot")
    parser.add_argument("--collect", action="store_true", help="run the collection")
    parser.add_argument("--choice", nargs="+", help="settings for the collection, e.g. faq=0.30:150 goat=0.30:70")
    parser.add_argument("--rules", action="store_true", help="run round 1's rules and best of N at low rho")
    parser.add_argument("--rule-pairs", type=int, default=RULE_PAIRS, help="pairs per rho for --rules")
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=ALGORITHMS)
    parser.add_argument("--status", action="store_true", help="only show how far it is")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--blas-threads", type=int, default=3)
    args = parser.parse_args(argv)
    if args.status:
        print(pilot_status(args.algorithms).to_string())
        try:
            choice = _parse_choice(args.choice) or recommend(pilot_table(read_pilot()))
            print(collection_status(choice).to_string())
        except FileNotFoundError:
            pass
        print(rules_status(args.algorithms, n_pairs=args.rule_pairs).to_string())
        return
    if args.pilot:
        run_pilot(args.algorithms, jobs=args.jobs, blas_threads=args.blas_threads)
    if args.collect:
        choice = _parse_choice(args.choice)
        if not choice:
            choice = recommend(pilot_table(read_pilot()))
            print(f"Using the pilot's recommendation: {describe_choice(choice)}")
        choice = {a: c for a, c in choice.items() if a in args.algorithms}
        run_collection(choice, jobs=args.jobs, blas_threads=args.blas_threads)
    if args.rules:
        run_rules(args.algorithms, n_pairs=args.rule_pairs, jobs=args.jobs, blas_threads=args.blas_threads)


if __name__ == "__main__":
    main()
