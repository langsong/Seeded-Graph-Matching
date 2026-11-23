"""Learned selector at a fixed budget (10 Nov 2026).

Question. At each landscape cell's budget k, using all k queries, can a seed-set score fitted where good sets live
pick sets that beat the best hand-made rule? The frozen pooled score was fitted at round 1's settings on sets from a
random pool (plain random sets, the pool's extreme tail, degree-matched contrasts) with single-run pass/fail labels.
Rule-built sets sit far outside that range (top degree is about 11-13 SD above the average random set in seed degree;
the farthest training sets reach about 5 on ER), and the best rule changes with the regime.

Steps. Each resumes where it stopped; run them in order, or all at once with --all.

    weights      How much do fitted weights move across the 27 standard landscape cells, and does a score fitted in
                 one cell rank sets in another? Landscape data only, no runs (weight_table, transfer_matrix;
                 --step weights prints a summary).
    constructed  Labels averaged over B for sets that span random to rule-built. On the first landscape pair of each
                 standard cell: round 1's rules, the best-of screens, degree tilts (the landscape's first
                 EXISTING_PER_GAMMA sets per gamma, plus NEW_PER_GAMMA new sets at each of NEW_GAMMAS), quantile picks
                 (QUANTILES) and swap paths that turn PATH_STARTS random sets into each rule's set one swap at a time
                 (sets at PATH_FRACTIONS of the way). Each set runs once on the pair's saved B and on the same DRAWS
                 redraws of B the landscape used, so the landscape's 100 redrawn random sets join the data with labels
                 on the same 11 Bs.
    round        Per cell: fit the selector to everything labelled so far, search for better sets, run the proposals
                 on the same 11 Bs. Rounds 1 to ROUNDS, each after the previous one has finished.
    finalize     Fit each cell's final selector to all its data (final_model.json, with leave-one-family-out checks).
    fresh        FRESH_PAIRS new pairs per cell at the same k. On each: the final selector's set, the frozen score's
                 set built by the same search, every rule (the best rule is fixed in advance from the landscape, in
                 best_rules.json), and the best of 1,000 random sets by the frozen score, each on the saved B and
                 FRESH_DRAWS redraws of B; and FRESH_RANDOM random sets on the saved B.

Selector. Logistic in the four core z-scores (against the pair's random pool, as in the landscape) plus degree
squared, fitted to mean test accuracy over the Bs (each set weighted by its number of Bs; ridge RIDGE on the
standardised terms). BOOTSTRAP refits on resampled sets give a prediction SD, used for exploration.

Search. Single swaps from each rule's set and the best labelled sets (SWAPS sampled swaps per step, at most
ITERATIONS steps), within a pool of the POOL_FACTOR * k highest-degree candidates plus every rule's vertices, and only
through sets whose four z-scores stay inside the labelled sets' range (+- GUARD_MARGIN): the fitted score is not
trusted where it has no data. Each round proposes EXPLOIT sets by predicted accuracy and EXPLORE sets by predicted
accuracy + KAPPA * its SD, never a set already labelled.

Run from the repository root (ds environment). Ctrl-C stops the workers too; rerun to resume.

    python -m selection_rule_comparison.learned_selector --smoke            # every step, tiny, in tmp/
    python -m selection_rule_comparison.learned_selector --all              # every step, in order
    python -m selection_rule_comparison.learned_selector --step constructed
    python -m selection_rule_comparison.learned_selector --step round --round 1
    python -m selection_rule_comparison.learned_selector --step round --round 2
    python -m selection_rule_comparison.learned_selector --step finalize
    python -m selection_rule_comparison.learned_selector --step fresh
    python -m selection_rule_comparison.learned_selector --status
    python -m selection_rule_comparison.learned_selector --step weights      # step 1's summary (no runs)

Results: data/learned_selector/ (or LEARNED_SELECTOR_OUTPUT_DIR), per cell <model>/<cell>/<algorithm>/k<budget>/:
pair_000/constructed_sets.csv and constructed_runs.csv, pair_000/proposals.csv and proposal_runs.csv,
model_round<r>.json, final_model.json, fresh/pair_XXX/sets.csv and runs.csv; settings.json and best_rules.json at the
top. learned_selector.ipynb reads them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache"))

import numpy as np
import pandas as pd

from experiment_common.provenance import collect_environment, thread_limit, write_json
from fixed_pair_seed_set_classification import multi_pair_collection as MP

from . import b_replicates as BR
from . import graded_outcomes as G
from . import landscape as L
from . import regime_map as RM
from . import run_comparison as rc
from .strategies import GraphA

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent
DEFAULT_DIR = EXPERIMENT_DIR / "data" / "learned_selector"
SMOKE_DIR = REPO_ROOT / "tmp" / "learned_selector_smoke"

MODELS, ALGORITHMS = L.MODELS, L.ALGORITHMS
CORE = MP.CORE_FEATURES                                   # degree, harmonic, pair_mass, uniqueness
Z = tuple(f"z_{f}" for f in CORE)
TERMS = (*Z, "z_degree_sq")
LABELS, MODEL_LABELS = RM.LABELS, RM.MODEL_LABELS

CONSTRUCTED_PAIRS = (0,)      # landscape pairs that get constructed sets (pair 0 has the landscape's redraws of B)
DRAWS = 10                    # redraws of B per set, the landscape's; draw 0 is the pair's saved B
EXISTING_GAMMAS, EXISTING_PER_GAMMA = (-1.0, -0.5, 0.5, 1.0, 2.0, 3.0), 3
NEW_GAMMAS, NEW_PER_GAMMA = (5.0, 8.0), 5
QUANTILES = {"degree": (0.0, 0.01, 0.9, 0.97, 0.99, 0.998, 1.0), "score": (0.0, 0.01, 0.9, 0.97, 0.99, 0.998, 1.0),
             "harmonic": (0.9, 0.97, 0.99, 0.998, 1.0), "pair_mass": (0.9, 0.97, 0.99, 0.998, 1.0),
             "uniqueness": (0.9, 0.97, 0.99, 0.998, 1.0)}
PATH_STARTS, PATH_FRACTIONS = 2, (0.25, 0.5, 0.75)
RIDGE, BOOTSTRAP = 1.0, 30
POOL_FACTOR, GUARD_MARGIN = 2, 0.5
ITERATIONS, SWAPS = 25, 500
ROUNDS, EXPLOIT, EXPLORE, KAPPA = 2, 2, 2, 1.0
FRESH_PAIRS, FRESH_OFFSET, FRESH_DRAWS, FRESH_RANDOM = 10, 200, 4, 20
TAKEOFF = 0.8                 # a run "takes off" at this test accuracy (for the resilience summary)

SMOKE = {   # --smoke: every step on a tiny design (one cell per algorithm, ER only), in its own folder
    "models": ["sparse_er"], "cells_per_algorithm": 1, "draws": 1, "existing_gammas": [1.0, 3.0],
    "existing_per_gamma": 1, "new_gammas": [5.0], "new_per_gamma": 1, "quantiles": {"degree": [1.0], "score": [1.0]},
    "path_starts": 1, "path_fractions": [0.5], "bootstrap": 4, "iterations": 2, "swaps": 30, "exploit": 1,
    "explore": 1, "fresh_pairs": 1, "fresh_draws": 1, "fresh_random": 2,
}

SAMPLING_SEED_BASE, SOLVER_SEED_BASE, FIT_SEED_BASE = 20261190, 20261191, 20261192
PURPOSE = {"tilt": 1, "path": 2, "search": 3}
KIND_CODES = {"constructed": 1, "proposal": 2, "fresh": 3}
STEPS = ("constructed", "round", "finalize", "fresh")

FAMILY_ORDER = ("random", "tilt", "quantile", "screen", "path", "rule", "proposal")
FAMILY_LABELS = {"random": "random sets", "tilt": "degree tilt", "quantile": "pool quantile picks",
                 "screen": "best-of screens", "path": "swap paths toward a rule", "rule": "round 1's rules",
                 "proposal": "selector proposals"}
METHOD_ORDER = ("random", "best_1000", "frozen_built", "best_rule", "learned")
METHOD_LABELS = {**RM.METHOD_LABELS, "random": "random sets", "best_1000": "best of 1,000 by frozen score",
                 "frozen_built": "frozen score, built", "best_rule": "best rule (fixed in advance)",
                 "learned": "learned selector"}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _now() -> str:
    return RM._now()


def _clock() -> str:
    return RM._clock()


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, content: dict) -> None:
    L._write_json(Path(path), content)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _key(seeds) -> str:
    """A set's identity: its sorted vertices as JSON (the landscape's seed_vertices format)."""
    return json.dumps([int(v) for v in np.sort(np.asarray(seeds, dtype=np.int64))])


def _seeds(key: str) -> np.ndarray:
    return np.asarray(json.loads(key), dtype=np.int64)


def _spearman(x, y) -> float:
    return L._spearman(x, y)


def _rng(pair_seed: int, purpose: str, k: int, *values: int) -> np.random.Generator:
    return np.random.default_rng([SAMPLING_SEED_BASE, int(pair_seed), PURPOSE[purpose], int(k), *map(int, values)])


def _count_ids(path: Path) -> int:
    """Complete rows in a run file (read only: never repairs a file another process may be writing)."""
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return 0
    return sum(1 for row in rows[1:] if len(row) == len(rows[0]))


def output_dir(smoke: bool = False) -> Path:
    configured = os.environ.get("LEARNED_SELECTOR_OUTPUT_DIR")
    if configured:
        return Path(configured)
    return SMOKE_DIR if smoke else DEFAULT_DIR


# --------------------------------------------------------------------------
# Design, cells and folders
# --------------------------------------------------------------------------

def _fixed(smoke: bool) -> dict:
    design = {
        "version": 1, "smoke": smoke, "models": list(MODELS), "cells_per_algorithm": None,
        "constructed_pairs": list(CONSTRUCTED_PAIRS), "draws": DRAWS, "existing_gammas": list(EXISTING_GAMMAS),
        "existing_per_gamma": EXISTING_PER_GAMMA, "new_gammas": list(NEW_GAMMAS), "new_per_gamma": NEW_PER_GAMMA,
        "quantiles": {f: list(q) for f, q in QUANTILES.items()}, "path_starts": PATH_STARTS,
        "path_fractions": list(PATH_FRACTIONS), "terms": list(TERMS), "ridge": RIDGE, "bootstrap": BOOTSTRAP,
        "pool_factor": POOL_FACTOR, "guard_margin": GUARD_MARGIN, "iterations": ITERATIONS, "swaps": SWAPS,
        "rounds": ROUNDS, "exploit": EXPLOIT, "explore": EXPLORE, "kappa": KAPPA, "fresh_pairs": FRESH_PAIRS,
        "fresh_offset": FRESH_OFFSET, "fresh_draws": FRESH_DRAWS, "fresh_random": FRESH_RANDOM,
        "seed_bases": {"sampling": SAMPLING_SEED_BASE, "solver": SOLVER_SEED_BASE, "fit": FIT_SEED_BASE},
    }
    if smoke:
        design.update(SMOKE)
    return json.loads(json.dumps(design))


def prepare(smoke: bool = False, blas_threads: int = 3) -> dict:
    """The design every step uses, frozen in settings.json the first time."""

    root = output_dir(smoke)
    path = root / "settings.json"
    fixed = _fixed(smoke)
    if path.exists():
        saved = _read_json(path)
        differs = [key for key in fixed if saved.get(key) != fixed[key]]
        if differs:
            raise ValueError(f"The saved design in {path} differs ({', '.join(differs)}). Set "
                             "LEARNED_SELECTOR_OUTPUT_DIR to a new folder to keep the results apart.")
    else:
        root.mkdir(parents=True, exist_ok=True)
        write_json(path, {**fixed, "written_at": _now(), "environment": collect_environment(blas_threads)})
    return {**fixed, "output_dir": str(root), "blas_threads": blas_threads}


def read_design(smoke: bool = False) -> dict:
    """The saved design (for analysis; no environment check)."""
    root = output_dir(smoke)
    path = root / "settings.json"
    saved = _read_json(path) if path.exists() else _fixed(smoke)
    return {**saved, "output_dir": str(root), "blas_threads": 3}


def cells(design: dict, models=None, report=print) -> list[tuple]:
    """(model, algorithm, mean degree, rho, budget) of the standard landscape cells, in the landscape's order."""

    out = []
    for model in models or design["models"]:
        for algorithm in ALGORITHMS:
            standard = L.cells_for(model, (algorithm,))
            if design.get("cells_per_algorithm"):
                standard = standard[: design["cells_per_algorithm"]]
            for a, degree, rho in standard:
                entry = L.cell_budget(model, a, degree, rho, L.LEVEL)
                if entry["budget"] is None:
                    report(f"{MODEL_LABELS[model]} {LABELS[a]} d{degree:g} rho {rho:.2f}: skipped ({entry['how']})")
                    continue
                out.append((model, a, degree, float(rho), int(entry["budget"])))
    return out


def cell_dir(design: dict, model: str, algorithm: str, degree: float, rho: float, budget: int) -> Path:
    return Path(design["output_dir"]) / model / RM.cell_name(degree, rho) / algorithm / f"k{int(budget):03d}"


def cell_key(model: str, algorithm: str, degree: float, rho: float, budget: int) -> str:
    return f"{model}/{algorithm}/{RM.cell_name(degree, rho)}/k{int(budget):03d}"


def cell_label(model: str, algorithm: str, degree: float, rho: float, budget: int | None = None) -> str:
    return f"{MODEL_LABELS[model]} · {L.cell_title(algorithm, degree, rho, budget)}"


def describe(task) -> str:
    step, model, algorithm, degree, rho, budget, index = task[:7]
    text = f"{step} {cell_label(model, algorithm, degree, rho, budget)}"
    if step == "round":
        return f"{text}, round {task[7]}"
    return f"{text}, {'fresh ' if step == 'fresh' else ''}pair {index}"


# --------------------------------------------------------------------------
# Pairs, draws of B and features
# --------------------------------------------------------------------------

def _landscape_seeds(model: str, degree: float, rho: float) -> set[int]:
    seeds = set()
    for path in (L.OUTPUT_DIR / model / RM.cell_name(degree, rho)).glob("*/k*/pair_*/pair.json"):
        try:
            seeds.add(int(_read_json(path)["pair_seed"]))
        except (KeyError, ValueError, OSError):
            pass
    return seeds


def _digest(arrays, test) -> str:
    return hashlib.sha256(b"".join(np.asarray(s, dtype=np.int64).tobytes() for s in arrays)
                          + np.asarray(test, dtype=np.int64).tobytes()).hexdigest()[:16]


def context(model: str, algorithm: str, degree: float, rho: float, budget: int, pair_index: int,
            fresh_offset: int | None = None) -> SimpleNamespace:
    """A landscape pair (fresh_offset None; checked against the landscape's saved pair and plan) or a fresh pair:
    graphs, test vertices, the landscape's plan of sets with the pool's feature statistics, and the frozen weights."""

    ldesign = L.prepare(model)
    excluded = set(ldesign["excluded_pair_seeds"])
    if fresh_offset is None:
        index = L.PAIR_INDEX_OFFSET + pair_index
    else:
        index = int(fresh_offset) + pair_index
        excluded |= _landscape_seeds(model, degree, rho)
    seed, A, B, truth, test, candidates, info = RM.load_pair(model, degree, rho, index,
                                                             {**ldesign, "excluded_pair_seeds": sorted(excluded)})
    graph = GraphA(A)
    table, arrays, stats = L.plan_sets(algorithm, seed, A, graph, test, candidates, int(budget), ldesign)
    landscape = None if fresh_offset is not None else L.pair_dir(model, algorithm, degree, rho, budget, pair_index)
    ctx = SimpleNamespace(model=model, algorithm=algorithm, degree=degree, rho=float(rho), budget=int(budget),
                          pair_index=pair_index, fresh=fresh_offset is not None, seed=int(seed), A=A, B=B,
                          truth=truth, test=test, candidates=candidates, info=info, graph=graph, table=table,
                          arrays=arrays, stats=stats, weights=ldesign["weights"][algorithm], ldesign=ldesign,
                          probability=None, landscape=landscape, digest=_digest(arrays, test))
    if landscape is not None:
        _check_landscape(ctx)
    return ctx


def _check_landscape(ctx: SimpleNamespace) -> None:
    """The regenerated pair and plan must be the landscape's (its runs and redraws are reused)."""

    label = cell_label(ctx.model, ctx.algorithm, ctx.degree, ctx.rho, ctx.budget)
    path = ctx.landscape / "pair.json"
    if not path.exists():
        raise RuntimeError(f"{label}: no landscape pair {ctx.pair_index} in {ctx.landscape}; this step builds on the "
                           "landscape's results.")
    saved = _read_json(path)
    for key, value in (("hash_a", ctx.info["hash_a"]), ("hash_b", ctx.info["hash_b"]), ("design_hash", ctx.digest)):
        if key in saved and saved[key] != value:
            raise RuntimeError(f"{label}: the regenerated pair or plan differs from the landscape's ({key}).")


def _check_draw(ctx: SimpleNamespace, draw: int, new_b: np.ndarray) -> None:
    if ctx.landscape is not None and (ctx.landscape / "draws.json").exists():
        saved = _read_json(ctx.landscape / "draws.json").get(str(draw))
        if saved and saved["hash_b"] != rc._hash(new_b):
            raise RuntimeError(f"Draw {draw} of B differs from the landscape's for pair seed {ctx.seed}.")


def draw_b(ctx: SimpleNamespace, draw: int):
    """B and its true matching for one draw: 0 is the pair's saved B, 1.. are the landscape's redraws of B given A."""

    if draw == 0:
        return ctx.B, ctx.truth
    if ctx.probability is None:
        ctx.probability = L.b_probability(ctx.model, ctx.degree, ctx.rho, ctx.seed, ctx.A, ctx.B, ctx.truth)
    new_b, new_truth, _ = BR.draw_b(ctx.A, ctx.probability, ctx.seed, draw)
    _check_draw(ctx, draw, new_b)
    return new_b, new_truth


def features_of(ctx: SimpleNamespace, sets) -> pd.DataFrame:
    """The four core features of each set, their z-scores against the pair's random pool, and the frozen score."""

    sets = np.atleast_2d(np.asarray(sets, dtype=np.int64))
    raw = G._core_features(ctx.graph, ctx.algorithm, sets)
    frame = pd.DataFrame({f: np.asarray(raw[f], dtype=float) for f in CORE})
    for f in CORE:
        mean, sd = ctx.stats[f]["mean"], ctx.stats[f]["sd"]
        frame[f"z_{f}"] = (frame[f] - mean) / sd if sd > 0 else 0.0
    frame["score"] = sum(ctx.weights[f] * frame[f"z_{f}"] for f in CORE)
    return frame


def _rule_sets(ctx: SimpleNamespace) -> list[tuple[str, np.ndarray]]:
    rules = ctx.table.loc[ctx.table["design"] == "rule", ["set_id", "detail"]]
    return [(str(detail), ctx.arrays[int(set_id)]) for set_id, detail in rules.itertuples(index=False)]


def _run_sets(ctx: SimpleNamespace, items, path: Path, kind: str) -> int:
    """Run sets on draws of B, draw by draw (each B is drawn once), resuming from what `path` already has.
    items: (set_id, seeds, draws). run_id = 100 * set_id + draw."""

    done = G._completed_ids(path, "run_id")
    every = sorted({d for _, _, draws in items for d in draws})
    base = {"model": ctx.model, "mean_degree": ctx.degree, "rho": ctx.rho, "algorithm": ctx.algorithm,
            "budget": ctx.budget, "pair_index": ctx.pair_index, "pair_seed": ctx.seed}
    written = 0
    rows = G._Rows(path)
    try:
        for draw in every:
            todo = [(s, seeds) for s, seeds, draws in items if draw in draws and 100 * s + draw not in done]
            if not todo:
                continue
            new_b, new_truth = draw_b(ctx, draw)
            for set_id, seeds in todo:
                solver_seed = rc._seed(SOLVER_SEED_BASE, ctx.seed, ctx.budget, MP.ALGORITHM_CODES[ctx.algorithm],
                                       KIND_CODES[kind], int(set_id), int(draw))
                out = RM.run_one(ctx.algorithm, ctx.model, (ctx.A, new_b, new_truth), seeds, ctx.test, solver_seed)
                rows.write({**base, "set_id": int(set_id), "draw": int(draw), "run_id": 100 * int(set_id) + int(draw),
                            "solver_seed": solver_seed, **{c: out[c] for c in L.OUTCOME_COLUMNS}})
                written += 1
    finally:
        rows.close()
    return written


# --------------------------------------------------------------------------
# Step 2: constructed sets
# --------------------------------------------------------------------------

def _has_redraws(ctx: SimpleNamespace, design: dict) -> bool:
    path = ctx.landscape / "redraws.csv"
    if not path.exists() or path.stat().st_size == 0:
        return False
    return int(pd.read_csv(path, usecols=["draw"])["draw"].max()) >= design["draws"]


def constructed_plan(ctx: SimpleNamespace, design: dict) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """The pair's constructed sets (deterministic): rules, screens, tilts, quantile picks, swap paths, and random sets
    when the landscape has no redraws for this pair. A set reached twice is kept once ("also" names the other
    designs that gave it). landscape_set_id is the landscape's id of the same set (-1 for new sets); a swap path's
    random start is path_start_set_id."""

    table, arrays, k = ctx.table, ctx.arrays, ctx.budget
    code = MP.ALGORITHM_CODES[ctx.algorithm]
    rows, sets = [], []

    def add(family, detail, seeds, source=-1, start=-1):
        rows.append({"family": family, "detail": detail, "landscape_set_id": int(source),
                     "path_start_set_id": int(start)})
        sets.append(np.sort(np.asarray(seeds, dtype=np.int64)))

    for name in ("rule", "screen"):
        for set_id, detail in table.loc[table["design"] == name, ["set_id", "detail"]].itertuples(index=False):
            add(name, str(detail), arrays[int(set_id)], set_id)
    for gamma in design["existing_gammas"]:
        part = table[table["detail"] == f"gamma={gamma:g}"].head(design["existing_per_gamma"])
        for j, set_id in enumerate(part["set_id"]):
            add("tilt", f"gamma={gamma:g}#{j}", arrays[int(set_id)], set_id)
    degree = ctx.graph.degree[ctx.candidates].astype(float)
    for g, gamma in enumerate(design["new_gammas"]):
        p = degree ** gamma
        p = p / p.sum()
        rng = _rng(ctx.seed, "tilt", k, code, g)
        for j in range(design["new_per_gamma"]):
            add("tilt", f"gamma={gamma:g}#{j}", rng.choice(ctx.candidates, size=k, replace=False, p=p))
    for feature, quantiles in design["quantiles"].items():
        for q in quantiles:
            for set_id in table.loc[table["detail"] == f"{feature}@{q:g}", "set_id"].head(1):
                add("quantile", f"{feature}@{q:g}", arrays[int(set_id)], set_id)
    redrawn = L.redraw_sets(table, ctx.seed, k, ctx.ldesign["redraw_sets"])
    rng = _rng(ctx.seed, "path", k, code)
    starts = rng.choice(redrawn, size=min(design["path_starts"], len(redrawn)), replace=False)
    for s, start in enumerate(starts):
        first = arrays[int(start)]
        for rule, last in _rule_sets(ctx):
            leave = rng.permutation(np.setdiff1d(first, last))
            enter = rng.permutation(np.setdiff1d(last, first))
            for fraction in design["path_fractions"]:
                j = int(round(fraction * len(leave)))
                if 0 < j < len(leave):
                    add("path", f"{rule}|start{s}|{fraction:g}",
                        np.union1d(np.setdiff1d(first, leave[:j]), enter[:j]), start=start)
    if not _has_redraws(ctx, design):
        for set_id in redrawn:
            add("random", "", arrays[int(set_id)], set_id)
    keep, seen = [], {}
    for i, seeds in enumerate(sets):
        key = _key(seeds)
        if key in seen:          # e.g. several rules pick the same few hubs at tiny budgets
            first = rows[seen[key]]
            first["also"] = "; ".join(x for x in (first.get("also", ""), f"{rows[i]['family']}:{rows[i]['detail']}") if x)
        else:
            seen[key] = i
            keep.append(i)
    frame = pd.DataFrame([{**rows[i], "also": rows[i].get("also", "")} for i in keep])
    chosen = [sets[i] for i in keep]
    frame.insert(0, "set_id", np.arange(len(frame)))
    frame["seed_vertices"] = [_key(s) for s in chosen]
    frame = pd.concat([frame, features_of(ctx, np.stack(chosen))], axis=1)
    return frame, chosen


def _constructed(ctx: SimpleNamespace, design: dict, folder: Path):
    frame, sets = constructed_plan(ctx, design)
    digest = hashlib.sha256("|".join(frame["seed_vertices"]).encode()).hexdigest()[:16]
    hash_path, sets_path = folder / "constructed.hash", folder / "constructed_sets.csv"
    saved = hash_path.read_text() if hash_path.exists() else None
    if saved is not None and saved != digest and _count_ids(folder / "constructed_runs.csv"):
        raise RuntimeError(f"{folder}: the saved runs come from a different plan of sets. Set "
                           "LEARNED_SELECTOR_OUTPUT_DIR to a new folder to keep them apart.")
    if saved != digest or not sets_path.exists():
        out = frame.copy()
        for column, value in (("pair_seed", ctx.seed), ("budget", ctx.budget), ("rho", ctx.rho),
                              ("mean_degree", ctx.degree), ("algorithm", ctx.algorithm), ("model", ctx.model)):
            out.insert(0, column, value)
        _write_csv(sets_path, out)
        hash_path.write_text(digest)
        _write_json(folder / "pair.json", {**ctx.info, "algorithm": ctx.algorithm, "budget": ctx.budget,
                                           "landscape_design_hash": ctx.digest, "constructed_hash": digest,
                                           "n_sets": len(frame), "written_at": _now()})
    return frame, sets


def constructed_job(task, design: dict) -> str:
    _, model, algorithm, degree, rho, budget, pair_index = task
    folder = cell_dir(design, model, algorithm, degree, rho, budget) / f"pair_{pair_index:03d}"
    started = time.perf_counter()
    with thread_limit(design["blas_threads"]):
        ctx = context(model, algorithm, degree, rho, budget, pair_index)
        folder.mkdir(parents=True, exist_ok=True)
        frame, sets = _constructed(ctx, design, folder)
        draws = tuple(range(design["draws"] + 1))
        items = [(int(s), seeds, draws) for s, seeds in zip(frame["set_id"], sets)]
        written = _run_sets(ctx, items, folder / "constructed_runs.csv", "constructed")
    return f"{describe(task)}: {len(frame)} sets, {written} runs in {(time.perf_counter() - started) / 60:.1f} min"


# --------------------------------------------------------------------------
# Labels: every labelled set of a cell
# --------------------------------------------------------------------------

def _labels(runs: pd.DataFrame) -> pd.DataFrame:
    """Per set: mean, SD, worst and take-off share of test accuracy over its Bs; Bs with a result; rows written."""

    if runs.empty:
        return pd.DataFrame(columns=["mean_over_B", "sd_over_B", "min_over_B", "takeoff_share", "draws", "written"])
    ok = runs[L._ok(runs)]
    accuracy = ok.groupby("set_id")["accuracy_test"]
    out = pd.DataFrame({"mean_over_B": accuracy.mean(), "sd_over_B": accuracy.std(), "min_over_B": accuracy.min(),
                        "takeoff_share": (ok["accuracy_test"] >= TAKEOFF).groupby(ok["set_id"]).mean(),
                        "draws": accuracy.size()})
    written = runs.groupby("set_id").size().rename("written")
    return out.join(written, how="outer")


def landscape_random(model: str, algorithm: str, degree: float, rho: float, budget: int, pair_index: int,
                     draws: int) -> pd.DataFrame:
    """The landscape's redrawn random sets of this pair with labels over its saved B (first run) and redraws."""

    folder = L.pair_dir(model, algorithm, degree, rho, budget, pair_index)
    if not (folder / "redraws.csv").exists():
        return pd.DataFrame()
    sets = pd.read_csv(folder / "sets.csv")
    first = pd.read_csv(folder / "runs.csv")
    first = first[first["repeat"] == 0]
    redraws = pd.read_csv(folder / "redraws.csv")
    redraws = redraws[redraws["draw"] <= draws]
    ids = np.sort(redraws["set_id"].unique())
    columns = ["set_id", "draw", "accuracy_test", "error"]
    runs = pd.concat([first[first["set_id"].isin(ids)].assign(draw=0)[columns], redraws[columns]], ignore_index=True)
    out = sets.set_index("set_id").loc[ids, ["seed_vertices", *CORE, *Z, "score"]].join(_labels(runs), how="left")
    out = out.reset_index()
    out.insert(1, "family", "random")
    out.insert(2, "detail", "")
    out["source"] = "landscape"
    return out


def training(design: dict, model: str, algorithm: str, degree: float, rho: float, budget: int, pair_index: int = 0,
             upto_round: int | None = None, complete: bool = True) -> pd.DataFrame:
    """Every labelled set of a cell's pair: the landscape's redrawn random sets, the constructed sets and the
    proposals of rounds 1..upto_round (default: all). complete=True requires every constructed set and proposal to
    have run on every draw; complete=False keeps what has results so far."""

    folder = cell_dir(design, model, algorithm, degree, rho, budget) / f"pair_{pair_index:03d}"
    label = cell_label(model, algorithm, degree, rho, budget)
    frames = [landscape_random(model, algorithm, degree, rho, budget, pair_index, design["draws"])]
    keep = ["set_id", "family", "detail", "seed_vertices", *CORE, *Z, "score"]
    planned = design["draws"] + 1
    if (folder / "constructed_sets.csv").exists():
        sets = pd.read_csv(folder / "constructed_sets.csv")
        path = folder / "constructed_runs.csv"
        runs = pd.read_csv(path) if _count_ids(path) else pd.DataFrame()
        frames.append(sets[keep].join(_labels(runs), on="set_id").assign(source="constructed"))
    elif complete:
        raise RuntimeError(f"{label}: no constructed sets yet (run --step constructed first).")
    upto_round = design["rounds"] if upto_round is None else upto_round
    if upto_round > 0 and (folder / "proposals.csv").exists():
        proposals = pd.read_csv(folder / "proposals.csv")
        proposals = proposals[proposals["round"] <= upto_round]
        path = folder / "proposal_runs.csv"
        runs = pd.read_csv(path) if _count_ids(path) else pd.DataFrame()
        part = proposals.assign(family="proposal", detail=proposals["kind"] + " r" + proposals["round"].astype(str))
        frames.append(part[keep].join(_labels(runs), on="set_id").assign(source="proposal"))
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    for column in ("draws", "written"):
        frame[column] = frame[column].fillna(0).astype(int)
    if complete:
        ours = frame[frame["source"] != "landscape"]
        short = ours[ours["written"] < planned]
        if len(short):
            raise RuntimeError(f"{label}: {len(short)} constructed sets or proposals have not run on all {planned} "
                               "Bs yet; finish the earlier step first.")
    frame = frame[frame["draws"] > 0].reset_index(drop=True)
    for column, value in (("pair_index", pair_index), ("budget", int(budget)), ("rho", float(rho)),
                          ("mean_degree", degree), ("algorithm", algorithm), ("model", model)):
        frame.insert(0, column, value)
    return frame


# --------------------------------------------------------------------------
# Step 3: the selector, its checks and the search
# --------------------------------------------------------------------------

def _terms(frame: pd.DataFrame) -> np.ndarray:
    z = frame[list(Z)].to_numpy(dtype=float)
    return np.column_stack([z, z[:, 0] ** 2])


def _logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float) -> np.ndarray:
    """Weighted logistic fit of a share y in [0, 1] (binomial deviance), ridge on everything but the intercept."""

    from scipy.optimize import minimize
    from scipy.special import expit

    design_matrix = np.column_stack([np.ones(len(X)), X])

    def loss(beta):
        eta = design_matrix @ beta
        value = float(np.sum(w * (np.logaddexp(0.0, eta) - y * eta)) + 0.5 * ridge * np.sum(beta[1:] ** 2))
        gradient = design_matrix.T @ (w * (expit(eta) - y))
        gradient[1:] += ridge * beta[1:]
        return value, gradient

    start = np.zeros(design_matrix.shape[1])
    mean = float(np.clip(np.average(y, weights=w), 1e-3, 1 - 1e-3))
    start[0] = np.log(mean / (1 - mean))
    return minimize(loss, start, jac=True, method="L-BFGS-B").x


def fit_selector(frame: pd.DataFrame, design: dict, seed: int, bootstrap: int | None = None) -> dict:
    """The selector fitted to labelled sets (mean test accuracy over the Bs), with bootstrap refits and the labelled
    sets' range of each z-score (the search guard)."""

    frame = frame[frame["draws"] > 0]
    X = _terms(frame)
    centre, scale = X.mean(axis=0), X.std(axis=0)
    # A term that is (numerically) constant over the labelled sets carries no information; dividing by its ~0 SD
    # would turn any other value on a new pair into an enormous standardised value. Its column stays 0 in training,
    # so its coefficient stays 0 and it drops out of every prediction.
    scale[scale < 1e-9 * np.maximum(1.0, np.abs(centre))] = 1.0
    Xs = (X - centre) / scale
    y = frame["mean_over_B"].to_numpy(dtype=float).clip(0.0, 1.0)
    w = frame["draws"].to_numpy(dtype=float)
    w = w / w.mean()
    beta = _logistic(Xs, y, w, design["ridge"])
    rng = np.random.default_rng([FIT_SEED_BASE, int(seed)])
    boot = []
    for _ in range(design["bootstrap"] if bootstrap is None else bootstrap):
        index = rng.integers(0, len(y), len(y))
        boot.append(_logistic(Xs[index], y[index], w[index], design["ridge"]).tolist())
    model = {"terms": list(TERMS), "centre": centre.tolist(), "scale": scale.tolist(), "beta": beta.tolist(),
             "bootstrap": boot, "guard": {f: [float(frame[f].min()), float(frame[f].max())] for f in Z},
             "margin": design["guard_margin"], "n_sets": int(len(frame)),
             "families": {str(k): int(v) for k, v in frame["family"].value_counts().items()}}
    predicted, _ = predict(model, frame)
    model["in_sample_spearman"] = _spearman(predicted, y)
    model["in_sample_mae"] = float(np.mean(np.abs(predicted - y)))
    return model


def predict(model: dict, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """Predicted mean accuracy over B, and its SD over the bootstrap refits (None without refits)."""

    from scipy.special import expit

    Xs = (_terms(frame) - np.asarray(model["centre"])) / np.asarray(model["scale"])
    design_matrix = np.column_stack([np.ones(len(Xs)), Xs])
    mean = expit(design_matrix @ np.asarray(model["beta"]))
    if not model.get("bootstrap"):
        return mean, None
    return mean, expit(design_matrix @ np.asarray(model["bootstrap"]).T).std(axis=1)


def in_range(model: dict, frame: pd.DataFrame) -> np.ndarray:
    """Sets whose four z-scores lie within the labelled sets' range, widened by the margin."""

    inside = np.ones(len(frame), dtype=bool)
    for f, (low, high) in model["guard"].items():
        values = frame[f].to_numpy(dtype=float)
        inside &= (values >= low - model["margin"]) & (values <= high + model["margin"])
    return inside


def lofo(frame: pd.DataFrame, design: dict, seed: int) -> pd.DataFrame:
    """Leave one family out: fit without it, predict it. Can the selector rank and size sets of a kind it has not
    seen (rules, paths, proposals ...)? The frozen score's ranking of the same sets for comparison."""

    rows = []
    frame = frame[frame["draws"] > 0]
    for family, held in frame.groupby("family"):
        rest = frame[frame["family"] != family]
        if len(held) < 3 or len(rest) < 20:
            continue
        model = fit_selector(rest, design, seed, bootstrap=0)
        predicted, _ = predict(model, held)
        y = held["mean_over_B"].to_numpy(dtype=float)
        rows.append({"held out": family, "sets": len(held), "observed mean": float(y.mean()),
                     "predicted mean": float(predicted.mean()), "MAE": float(np.mean(np.abs(predicted - y))),
                     "Spearman": _spearman(predicted, y), "frozen score Spearman": _spearman(held["score"], y),
                     "share in range": float(np.mean(in_range(model, held)))})
    return pd.DataFrame(rows)


def pair_guard(model: dict, ctx: SimpleNamespace) -> dict:
    """The selector with its range widened to cover this pair's own rule sets. On the labelled pair the rules are
    already inside it; on a fresh pair their z-scores can sit a little further out, and the search should be able to
    reach (and start from) them."""

    rules = features_of(ctx, np.stack([seeds for _, seeds in _rule_sets(ctx)]))
    guard = {f: [min(low, float(rules[f].min())), max(high, float(rules[f].max()))]
             for f, (low, high) in model["guard"].items()}
    return {**model, "guard": guard}


def high_degree_pool(ctx: SimpleNamespace, design: dict) -> np.ndarray:
    """Search pool: the POOL_FACTOR * k highest-degree candidates (no test vertices) and every rule's vertices."""

    candidates = ctx.candidates
    order = candidates[np.lexsort((candidates, -ctx.graph.degree[candidates]))]
    pool = order[: min(len(order), design["pool_factor"] * ctx.budget)]
    rules = [seeds for _, seeds in _rule_sets(ctx)]
    return np.union1d(pool, np.concatenate(rules) if rules else pool).astype(np.int64)


def learned_objective(ctx: SimpleNamespace, model: dict, kappa: float = 0.0):
    if not kappa:
        model = {**model, "bootstrap": []}       # the refits only matter for the SD

    def objective(sets) -> np.ndarray:
        frame = features_of(ctx, sets)
        mean, sd = predict(model, frame)
        value = mean + kappa * sd if (kappa and sd is not None) else mean
        return np.where(in_range(model, frame), value, -np.inf)
    return objective


def frozen_objective(ctx: SimpleNamespace, model: dict):
    """The frozen score, searched the same way and inside the same guard as the selector."""
    def objective(sets) -> np.ndarray:
        frame = features_of(ctx, sets)
        return np.where(in_range(model, frame), frame["score"].to_numpy(dtype=float), -np.inf)
    return objective


def search(ctx: SimpleNamespace, starts, pool: np.ndarray, objective, design: dict, rng: np.random.Generator,
           keep: int = 20):
    """Single-swap hill climbing from each start (best of `swaps` sampled swaps per step). Returns the final set of
    each start and every good set seen, best first, as (value, set)."""

    k = ctx.budget
    found = {}

    def remember(values, sets):
        for i in np.argsort(-values)[:keep]:
            if not np.isfinite(values[i]):
                break
            key = _key(sets[i])
            if key not in found or found[key][0] < values[i]:
                found[key] = (float(values[i]), np.array(sets[i], dtype=np.int64))

    finals = []
    for start in starts:
        current = np.sort(np.asarray(start, dtype=np.int64))
        value = float(objective(current[None, :])[0])
        remember(np.array([value]), current[None, :])
        for _ in range(design["iterations"]):
            outside = pool[~np.isin(pool, current)]
            if len(outside) == 0:
                break
            total = k * len(outside)
            flat = rng.choice(total, size=min(design["swaps"], total), replace=False)
            position, index = np.divmod(flat, len(outside))
            candidates = np.repeat(current[None, :], len(flat), axis=0)
            candidates[np.arange(len(flat)), position] = outside[index]
            candidates.sort(axis=1)
            values = objective(candidates)
            remember(values, candidates)
            best = int(np.argmax(values))
            if not values[best] > value + 1e-9:
                break
            current, value = candidates[best], float(values[best])
        finals.append((value, current))
    ranked = sorted(found.values(), key=lambda item: -item[0])
    return finals, ranked


def _unique(sets) -> list[np.ndarray]:
    out, seen = [], set()
    for seeds in sets:
        key = _key(seeds)
        if key not in seen:
            seen.add(key)
            out.append(np.sort(np.asarray(seeds, dtype=np.int64)))
    return out


def _fit_seed(key: str, round_number: int) -> int:
    return rc._seed(FIT_SEED_BASE, int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), round_number)


def propose(ctx: SimpleNamespace, frame: pd.DataFrame, model: dict, design: dict, round_number: int) -> pd.DataFrame:
    """New sets to label: EXPLOIT by predicted accuracy, then EXPLORE by predicted accuracy + KAPPA * SD, searched
    from every rule's set and the best labelled sets (observed and predicted)."""

    rng = _rng(ctx.seed, "search", ctx.budget, MP.ALGORITHM_CODES[ctx.algorithm], round_number)
    pool = high_degree_pool(ctx, design)
    labelled = frame.assign(predicted=predict(model, frame)[0])
    picks = pd.concat([labelled.nlargest(3, "mean_over_B"), labelled.nlargest(3, "predicted")])
    starts = _unique([seeds for _, seeds in _rule_sets(ctx)] + [_seeds(s) for s in picks["seed_vertices"]])
    known = set(frame["seed_vertices"])
    chosen, rows = [], []
    for kind, kappa, count in (("exploit", 0.0, design["exploit"]), ("explore", design["kappa"], design["explore"])):
        _, ranked = search(ctx, starts, pool, learned_objective(ctx, model, kappa), design, rng)
        added = 0
        for value, seeds in ranked:
            key = _key(seeds)
            if added >= count:
                break
            if key in known or not np.isfinite(value):
                continue
            known.add(key)
            chosen.append(seeds)
            rows.append({"kind": kind, "objective": value})
            added += 1
    if not chosen:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    table.insert(0, "set_id", [100 * round_number + j for j in range(len(table))])
    table.insert(0, "round", round_number)
    table["seed_vertices"] = [_key(s) for s in chosen]
    features = features_of(ctx, np.stack(chosen))
    mean, sd = predict(model, features)
    table = pd.concat([table, features], axis=1)
    table["predicted"], table["predicted_sd"] = mean, sd
    return table


def round_job(task, design: dict) -> str:
    _, model, algorithm, degree, rho, budget, pair_index, round_number = task
    base = cell_dir(design, model, algorithm, degree, rho, budget)
    folder = base / f"pair_{pair_index:03d}"
    started = time.perf_counter()
    with thread_limit(design["blas_threads"]):
        ctx = context(model, algorithm, degree, rho, budget, pair_index)
        path = folder / "proposals.csv"
        saved = pd.read_csv(path) if path.exists() else pd.DataFrame()
        mine = saved[saved["round"] == round_number] if len(saved) else saved
        if mine.empty:
            frame = training(design, model, algorithm, degree, rho, budget, pair_index, round_number - 1, True)
            key = cell_key(model, algorithm, degree, rho, budget)
            fit = fit_selector(frame, design, _fit_seed(key, round_number))
            _write_json(base / f"model_round{round_number}.json",
                        {**fit, "round": round_number, "labelled_rounds": round_number - 1, "written_at": _now()})
            mine = propose(ctx, frame, fit, design, round_number)
            if mine.empty:
                return f"{describe(task)}: no new set within the pool and range"
            _write_csv(path, pd.concat([saved, mine], ignore_index=True) if len(saved) else mine)
        draws = tuple(range(design["draws"] + 1))
        items = [(int(s), _seeds(v), draws) for s, v in zip(mine["set_id"], mine["seed_vertices"])]
        written = _run_sets(ctx, items, folder / "proposal_runs.csv", "proposal")
    return f"{describe(task)}: {len(mine)} proposals, {written} runs in {(time.perf_counter() - started) / 60:.1f} min"


def finalize(design: dict, cell_list, report=print) -> pd.DataFrame:
    """Fit every cell's final selector to all its labelled sets (final_model.json, with leave-one-family-out
    checks). Cells whose data are incomplete are reported and skipped."""

    rows = []
    for cell in cell_list:
        model, algorithm, degree, rho, budget = cell
        try:
            frame = training(design, *cell, pair_index=0, upto_round=design["rounds"], complete=True)
        except RuntimeError as error:
            report(f"finalize {cell_label(*cell)}: skipped ({error})")
            continue
        key = cell_key(*cell)
        seed = _fit_seed(key, 99)
        fit = fit_selector(frame, design, seed)
        checks = lofo(frame, design, seed)
        best = frame.loc[frame["mean_over_B"].idxmax()]
        detail = str(best["detail"]) if pd.notna(best["detail"]) else ""
        _write_json(cell_dir(design, *cell) / "final_model.json",
                    {**fit, "lofo": json.loads(checks.to_json(orient="records")) if len(checks) else [],
                     "cell": key, "best_labelled": {"family": str(best["family"]), "detail": detail,
                                                    "mean_over_B": float(best["mean_over_B"])},
                     "written_at": _now()})
        rows.append({"cell": key, "sets": len(frame), "in-sample Spearman": fit["in_sample_spearman"]})
    report(f"{_clock()}  Final selectors: {len(rows)} of {len(cell_list)} cells.")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Step 4: fresh pairs
# --------------------------------------------------------------------------

def best_rules(design: dict, cell_list) -> dict:
    """The best rule per cell, fixed before any fresh pair runs: highest mean test accuracy over the landscape's
    pairs (saved B; FAQ's 3 runs averaged). Saved in best_rules.json."""

    path = Path(design["output_dir"]) / "best_rules.json"
    saved = _read_json(path) if path.exists() else {}
    missing = [c for c in cell_list if cell_key(*c) not in saved]
    for model in sorted({c[0] for c in missing}):
        sets = L.per_set([model])
        rules = sets[(sets["design"] == "rule") & (sets["kind"] == "standard")]
        for cell in [c for c in missing if c[0] == model]:
            _, algorithm, degree, rho, budget = cell
            part = rules[(rules["algorithm"] == algorithm) & (rules["mean_degree"] == degree)
                         & np.isclose(rules["rho"], rho) & (rules["budget"] == budget)]
            means = part.groupby("detail")["accuracy"].mean().sort_values(ascending=False)
            if means.empty:
                continue
            saved[cell_key(*cell)] = {"best_rule": str(means.index[0]),
                                      "landscape_means": {k: round(float(v), 4) for k, v in means.items()},
                                      "pairs": int(part["pair_index"].nunique()), "written_at": _now()}
    if missing:
        _write_json(path, saved)
    return saved


def _fresh_sets(ctx: SimpleNamespace, final: dict, design: dict) -> pd.DataFrame:
    table, arrays = ctx.table, ctx.arrays
    every = list(range(design["fresh_draws"] + 1))
    rows = []

    def add(method, detail, seeds, draws, note=""):
        rows.append({"method": method, "detail": detail, "seeds": np.sort(np.asarray(seeds, dtype=np.int64)),
                     "planned_draws": json.dumps(draws), "note": note})

    for set_id in table.loc[table["design"] == "random", "set_id"].head(design["fresh_random"]):
        add("random", "", arrays[int(set_id)], [0])
    rules = _rule_sets(ctx)
    for rule, seeds in rules:
        add("rule", rule, seeds, every)
    screen = table.loc[table["detail"] == "best_1000", "set_id"]
    starts = [seeds for _, seeds in rules]
    for set_id in screen:
        add("best_1000", "best_1000", arrays[int(set_id)], every)
        starts.append(arrays[int(set_id)])
    starts = _unique(starts)
    pool = high_degree_pool(ctx, design)
    final = pair_guard(final, ctx)
    for method, objective in (("learned", learned_objective(ctx, final, 0.0)),
                              ("frozen_built", frozen_objective(ctx, final))):
        rng = _rng(ctx.seed, "search", ctx.budget, MP.ALGORITHM_CODES[ctx.algorithm], 900 + len(rows))
        _, ranked = search(ctx, starts, pool, objective, design, rng)
        if ranked:
            add(method, "", ranked[0][1], every)
        else:   # no set in range anywhere: the start the selector likes best, flagged
            frame = features_of(ctx, np.stack(starts))
            values = predict(final, frame)[0] if method == "learned" else frame["score"].to_numpy()
            add(method, "", starts[int(np.argmax(values))], every, "no set in the labelled range")
    frame = pd.DataFrame(rows)
    keys = [_key(s) for s in frame["seeds"]]
    first = {}
    for i, key in enumerate(keys):
        first.setdefault(key, i)
    frame["same_as"] = [("" if first[key] == i else f"{frame.at[first[key], 'method']}:{frame.at[first[key], 'detail']}")
                        for i, key in enumerate(keys)]
    frame.insert(0, "set_id", np.arange(len(frame)))
    frame["seed_vertices"] = keys
    features = features_of(ctx, np.stack(frame["seeds"].to_list()))
    frame = pd.concat([frame.drop(columns=["seeds"]), features], axis=1)
    frame["predicted"] = predict(final, features)[0]
    frame["in_range"] = in_range(final, features)
    return frame


def fresh_job(task, design: dict) -> str:
    _, model, algorithm, degree, rho, budget, index = task
    base = cell_dir(design, model, algorithm, degree, rho, budget)
    folder = base / "fresh" / f"pair_{index:03d}"
    started = time.perf_counter()
    final_path = base / "final_model.json"
    if not final_path.exists():
        raise RuntimeError(f"{describe(task)}: no final selector yet (run --step finalize first).")
    final = _read_json(final_path)
    with thread_limit(design["blas_threads"]):
        ctx = context(model, algorithm, degree, rho, budget, index, fresh_offset=design["fresh_offset"])
        folder.mkdir(parents=True, exist_ok=True)
        sets_path = folder / "sets.csv"
        if sets_path.exists():
            frame = pd.read_csv(sets_path)
            if int(frame["pair_seed"].iloc[0]) != ctx.seed:
                raise RuntimeError(f"{describe(task)}: the saved sets belong to another pair.")
        else:
            frame = _fresh_sets(ctx, final, design)
            for column, value in (("pair_seed", ctx.seed), ("pair_index", index), ("budget", ctx.budget),
                                  ("rho", ctx.rho), ("mean_degree", degree), ("algorithm", algorithm),
                                  ("model", model)):
                frame.insert(0, column, value)
            _write_csv(sets_path, frame)
            _write_json(folder / "pair.json", {**ctx.info, "algorithm": algorithm, "budget": ctx.budget,
                                               "final_model_written_at": final.get("written_at"),
                                               "written_at": _now()})
        items = [(int(s), _seeds(v), tuple(json.loads(d)))
                 for s, v, d in zip(frame["set_id"], frame["seed_vertices"], frame["planned_draws"])]
        written = _run_sets(ctx, items, folder / "runs.csv", "fresh")
    return f"{describe(task)}: {written} runs in {(time.perf_counter() - started) / 60:.1f} min"


# --------------------------------------------------------------------------
# Planning, running and status
# --------------------------------------------------------------------------

JOBS = {"constructed": constructed_job, "round": round_job, "fresh": fresh_job}


def _planned(task, design: dict) -> tuple[int, int]:
    """(runs planned, runs written) for a task; planned counts are exact once the task has written its sets."""

    step, model, algorithm, degree, rho, budget, index = task[:7]
    base = cell_dir(design, model, algorithm, degree, rho, budget)
    n_rules = len(RM.RULES[algorithm])
    if step == "constructed":
        folder = base / f"pair_{index:03d}"
        if (folder / "constructed_sets.csv").exists():
            n_sets = len(pd.read_csv(folder / "constructed_sets.csv", usecols=["set_id"]))
        else:
            n_sets = (n_rules + len(L.SCREENS) + len(design["existing_gammas"]) * design["existing_per_gamma"]
                      + len(design["new_gammas"]) * design["new_per_gamma"]
                      + sum(len(q) for q in design["quantiles"].values())
                      + design["path_starts"] * n_rules * len(design["path_fractions"]))
        return n_sets * (design["draws"] + 1), _count_ids(folder / "constructed_runs.csv")
    if step == "round":
        folder = base / f"pair_{index:03d}"
        round_number = task[7]
        n_sets = design["exploit"] + design["explore"]
        written = 0
        if (folder / "proposals.csv").exists():
            proposals = pd.read_csv(folder / "proposals.csv")
            mine = proposals[proposals["round"] == round_number]
            if len(mine):
                n_sets = len(mine)
            path = folder / "proposal_runs.csv"
            if _count_ids(path):
                runs = pd.read_csv(path, usecols=["set_id"])
                written = int((runs["set_id"] // 100 == round_number).sum())
        return n_sets * (design["draws"] + 1), written
    folder = base / "fresh" / f"pair_{index:03d}"
    if (folder / "sets.csv").exists():
        planned = int(sum(len(json.loads(d)) for d in pd.read_csv(folder / "sets.csv")["planned_draws"]))
    else:
        planned = design["fresh_random"] + (n_rules + 3) * (design["fresh_draws"] + 1)
    return planned, _count_ids(folder / "runs.csv")


def _left(task, design: dict) -> int:
    planned, written = _planned(task, design)
    return max(planned - written, 0)


def tasks_for(step: str, design: dict, cell_list, round_number: int | None = None) -> list[tuple]:
    if step == "constructed":
        return [("constructed", *c, p) for c in cell_list for p in design["constructed_pairs"]]
    if step == "round":
        return [("round", *c, 0, int(round_number)) for c in cell_list]
    if step == "fresh":
        return [("fresh", *c, i) for c in cell_list for i in range(design["fresh_pairs"])]
    raise ValueError(f"Unknown step: {step}")


def _run_tasks(tasks: list, design: dict, jobs: int, title: str) -> bool:
    """Run tasks in parallel (spawned workers), report each one, keep going after failures. False if stopped."""

    tasks = [t for t in tasks if _left(t, design) > 0]
    if not tasks:
        print(f"{_clock()}  {title}: nothing left to run.", flush=True)
        return True

    def seconds(task):
        return _left(task, design) * RM.SECONDS_PER_RUN[task[2]]

    tasks.sort(key=seconds, reverse=True)
    total = len(tasks)
    print(f"{_clock()}  {title}: {total} tasks, {jobs} at a time, about "
          f"{sum(seconds(t) for t in tasks) / max(jobs, 1) / 3600:.1f} h (a rough guess). One line per finished task.",
          flush=True)
    failures = []

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
                    message = JOBS[task[0]](task, design)
                except Exception as error:  # noqa: BLE001 - report every failure, keep going
                    report(n, task, None, error, [])
                    continue
                report(n, task, message, None, tasks[n:])
        else:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor, as_completed
            pool = ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn"),
                                       initializer=G._ignore_interrupts)
            futures = {pool.submit(JOBS[task[0]], task, design): task for task in tasks}
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
    print(f"{_clock()}  {title}: done. Results are in {design['output_dir']}", flush=True)
    return True


def run_step(step: str, design: dict, models=None, round_number: int | None = None, jobs: int = 3) -> bool:
    cell_list = cells(design, models)
    if step == "finalize":
        finalize(design, cell_list)
        return True
    if step == "round" and not round_number:
        raise ValueError("--step round needs --round 1, 2, ...")
    if step == "fresh":
        best_rules(design, cell_list)
        missing = [c for c in cell_list if not (cell_dir(design, *c) / "final_model.json").exists()]
        if missing:
            finalize(design, missing)
    title = f"Learned selector, {step}" + (f" {round_number}" if step == "round" else "")
    return _run_tasks(tasks_for(step, design, cell_list, round_number), design, jobs, title)


def run_all(design: dict, models=None, jobs: int = 3) -> bool:
    """Every step in order: constructed sets, the rounds, final selectors, fresh pairs."""

    if not run_step("constructed", design, models, jobs=jobs):
        return False
    for r in range(1, design["rounds"] + 1):
        if not run_step("round", design, models, r, jobs):
            return False
    run_step("finalize", design, models)
    return run_step("fresh", design, models, jobs=jobs)


def status(design: dict, models=None) -> pd.DataFrame:
    """Runs written per cell and step."""

    rows = []
    for cell in cells(design, models, report=lambda _: None):
        model, algorithm, degree, rho, budget = cell
        row = {"model": MODEL_LABELS[model], "algorithm": LABELS[algorithm], "mean degree": degree, "ρ": rho,
               "budget": budget}
        for step, extra in (("constructed", None), *(("round", r) for r in range(1, design["rounds"] + 1)),
                            ("fresh", None)):
            tasks = tasks_for(step, design, [cell], extra)
            planned = written = 0
            for task in tasks:
                p, w = _planned(task, design)
                planned, written = planned + p, written + w
            name = f"round {extra}" if step == "round" else step
            row[name] = f"{written} of {planned}"
        row["final"] = "yes" if (cell_dir(design, *cell) / "final_model.json").exists() else "-"
        rows.append(row)
    return pd.DataFrame(rows).set_index(["model", "algorithm", "mean degree", "ρ", "budget"])


# --------------------------------------------------------------------------
# Step 1: how much do fitted weights move across cells? (landscape data, no runs)
# --------------------------------------------------------------------------

def _ols(X: np.ndarray, y: np.ndarray):
    """Least squares with an intercept: slopes, their HC1 standard errors, and R²."""

    design_matrix = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(design_matrix, y, rcond=None)
    residual = y - design_matrix @ beta
    n, p = design_matrix.shape
    bread = np.linalg.pinv(design_matrix.T @ design_matrix)
    meat = design_matrix.T @ (design_matrix * residual[:, None] ** 2)
    covariance = bread @ meat @ bread * n / max(n - p, 1)
    r2 = 1 - float(residual @ residual) / float(((y - y.mean()) ** 2).sum()) if np.ptp(y) > 0 else np.nan
    return beta[1:], np.sqrt(np.diag(covariance))[1:], r2


def _centred(part: pd.DataFrame, columns) -> pd.DataFrame:
    """Columns with each pair's mean removed (pairs differ in level; z-scores are already per pair)."""
    return part[list(columns)] - part.groupby("pair_index")[list(columns)].transform("mean")


def _short(model: str, algorithm: str, degree: float, rho: float) -> str:
    models = {"sparse_er": "ER", "chung_lu": "CL", "paper_pa": "PA"}
    algorithms = {"percolation": "perc", "faq": "FAQ", "goat": "GOAT"}
    return f"{models[model]} {algorithms[algorithm]} d{degree:g} ρ{rho:g}"


def weight_table(sets: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per standard landscape cell: test accuracy on the four core z-scores over random sets (each pair's mean
    removed; FAQ's 3 runs averaged): slopes per SD with 95% intervals, R², how the fitted direction compares with
    the frozen weights (cosine, and rank agreement of the two scores over the cell's random sets), and the
    degree-harmonic correlation that makes their split of credit unstable."""

    sets = L.per_set() if sets is None else sets
    random = sets[(sets["kind"] == "standard") & (sets["design"] == "random")]
    rows = []
    for key, part in random.groupby(L.CELL_KEYS, sort=False):
        cell = dict(zip(L.CELL_KEYS, key))
        X = _centred(part, Z).to_numpy(dtype=float)
        y = _centred(part, ["accuracy"])["accuracy"].to_numpy(dtype=float)
        slopes, se, r2 = _ols(X, y)
        frozen = np.array([RM.load_weights(cell["model"])[cell["algorithm"]][f] for f in CORE])
        row = {**cell, "sets": len(part), "pairs": part["pair_index"].nunique(), "R2": r2}
        for f, slope, error in zip(CORE, slopes, se):
            row[f] = slope
            row[f"{f} low"], row[f"{f} high"] = slope - 1.96 * error, slope + 1.96 * error
        norm = np.linalg.norm(slopes) * np.linalg.norm(frozen)
        row["cosine with frozen"] = float(slopes @ frozen / norm) if norm > 0 else np.nan
        row["rank agreement with frozen"] = _spearman(X @ slopes, X @ frozen)
        row["degree-harmonic r"] = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
        rows.append(row)
    return L._order(pd.DataFrame(rows))


def transfer_matrix(sets: pd.DataFrame | None = None, quality: pd.DataFrame | None = None) -> pd.DataFrame:
    """Does a score fitted in one cell rank sets in another? Rows: the linear score fitted in a source cell (random
    sets of pairs 1-4, single runs, each pair's mean removed), plus the frozen score and degree alone. Columns:
    target cells. Values: Spearman of the score with mean accuracy over the 11 Bs of the target cell's 100 redrawn
    sets (pair 0, never used to fit). The diagonal is each cell's own fit tested on its own held-out pair."""

    sets = L.per_set() if sets is None else sets
    quality = L.set_quality(L.draws(sets), sets) if quality is None else quality
    standard = sets[sets["kind"] == "standard"]
    random = standard[standard["design"] == "random"]
    fits = {}
    for key, part in random.groupby(L.CELL_KEYS, sort=False):
        part = part[part["pair_index"] != 0]
        if part["pair_index"].nunique() < 2:
            continue
        slopes, _, _ = _ols(_centred(part, Z).to_numpy(dtype=float),
                            _centred(part, ["accuracy"])["accuracy"].to_numpy(dtype=float))
        fits[key] = slopes
    targets = quality[quality["kind"] == "standard"].drop(columns=["score", "z_degree", "design", "kind"])
    targets = targets.merge(standard[L.KEYS + list(Z) + ["score"]], on=L.KEYS, how="left")
    found = list(fits)     # ordered by algorithm, then model: each algorithm's cells form a block
    order = L._order(pd.DataFrame([{**dict(zip(L.CELL_KEYS, k)), "_i": i} for i, k in enumerate(found)]))
    order = order.assign(_a=order["algorithm"].map({a: i for i, a in enumerate(ALGORITHMS)}))
    keys = [found[i] for i in order.sort_values("_a", kind="stable")["_i"]]
    labels = {k: _short(k[0], k[1], k[2], k[3]) for k in keys}
    matrix = {}
    for target in keys:
        part = targets.merge(pd.DataFrame([dict(zip(L.CELL_KEYS, target))]), on=L.CELL_KEYS)
        if len(part) < 10:
            continue
        column = {labels[source]: _spearman(part[list(Z)].to_numpy(dtype=float) @ fits[source], part["mean_over_B"])
                  for source in keys}
        column["frozen score"] = _spearman(part["score"], part["mean_over_B"])
        column["degree alone"] = _spearman(part["z_degree"], part["mean_over_B"])
        matrix[labels[target]] = column
    out = pd.DataFrame(matrix)
    return out.reindex([labels[k] for k in keys] + ["frozen score", "degree alone"])


def transfer_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    """Per target cell: its own fit (held-out pair), the frozen score, degree alone, and fits from the other cells of
    the same algorithm (median and best)."""

    rows = []
    for target in matrix.columns:
        algorithm = target.split()[1]
        same = [s for s in matrix.index if s not in ("frozen score", "degree alone") and s.split()[1] == algorithm
                and s != target]
        values = matrix.loc[same, target].dropna()
        rows.append({"target cell": target, "own fit": matrix.at[target, target] if target in matrix.index else np.nan,
                     "frozen score": matrix.at["frozen score", target], "degree alone": matrix.at["degree alone", target],
                     "other cells, median": float(values.median()) if len(values) else np.nan,
                     "other cells, best": float(values.max()) if len(values) else np.nan,
                     "best source": values.idxmax() if len(values) else ""})
    return pd.DataFrame(rows).set_index("target cell")


# --------------------------------------------------------------------------
# Analysis of steps 2-4 (learned_selector.ipynb)
# --------------------------------------------------------------------------

CONTEXT, SERIES, TILT_RAMP = L.CONTEXT, L.SERIES, L.TILT_RAMP
INK, INK2, GRID = RM.INK, RM.INK2, RM.GRID
FAMILY_STYLE = {"tilt": dict(color="#3987e5", marker="o", s=16), "quantile": dict(color="#e87ba4", marker="D", s=18),
                "screen": dict(color="#1baf7a", marker="X", s=30), "path": dict(color="#eda100", marker="o", s=16),
                "proposal": dict(color=INK, marker="*", s=90)}


def read_training(design: dict | None = None, models=None) -> pd.DataFrame:
    """Every cell's labelled sets so far (pair 0, all rounds); empty if step 2 has not started."""

    design = design or read_design()
    frames = []
    for cell in cells(design, models, report=lambda _: None):
        frame = training(design, *cell, pair_index=0, upto_round=design["rounds"], complete=False)
        if len(frame) and (frame["source"] != "landscape").any():
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _cell_list(frame: pd.DataFrame, model: str) -> list[tuple]:
    part = frame[frame["model"] == model][["algorithm", "mean_degree", "rho", "budget"]].drop_duplicates()
    order = {(a, L._model_degree(model, d), float(r)): i for a in ALGORITHMS for i, (d, r) in enumerate(L.CELLS[a])}
    part = part.assign(a=part["algorithm"].map({a: i for i, a in enumerate(ALGORITHMS)}),
                       c=[order.get((a, d, float(r)), 99) for a, d, r in
                          part[["algorithm", "mean_degree", "rho"]].itertuples(index=False)])
    part = part.sort_values(["a", "c"])
    return [tuple(r) for r in part[["algorithm", "mean_degree", "rho", "budget"]].itertuples(index=False)]


def plot_constructed(frame: pd.DataFrame, model: str, x: str = "z_degree"):
    """Mean test accuracy over the 11 Bs against seed degree (z against random sets) for every labelled set:
    random sets (grey), degree tilts, quantile picks, screens, swap paths, round 1's rules (orange shapes) and the
    selector's proposals (black stars). Use x='predicted_accuracy' after add_final_predictions for the final
    selector's in-sample diagnostic, with a y=x calibration reference."""

    import matplotlib.pyplot as plt

    found = _cell_list(frame, model)
    if not found:
        print(f"No labelled sets yet for {MODEL_LABELS[model]}.")
        return None
    algorithms = [a for a in ALGORITHMS if any(c[0] == a for c in found)]
    rows_of = [[c for c in found if c[0] == a] for a in algorithms]
    if len(rows_of) > 1 and all(len(r) == 1 for r in rows_of):     # one cell per algorithm: a single row
        rows_of = [[r[0] for r in rows_of]]
    columns = max(len(r) for r in rows_of)
    with plt.rc_context({"font.size": 9}):
        fig, axes = plt.subplots(len(rows_of), columns, figsize=(3.8 * columns, 3.2 * len(rows_of) + 1.0),
                                 squeeze=False, sharey=True)
        for r, row in enumerate(rows_of):
            for c in range(columns):
                ax = axes[r, c]
                if c >= len(row):
                    ax.axis("off")
                    continue
                a, degree, rho, budget = row[c]
                part = frame[(frame["model"] == model) & (frame["algorithm"] == a) & (frame["mean_degree"] == degree)
                             & np.isclose(frame["rho"], rho) & (frame["budget"] == budget)]
                L._style(ax)
                random = part[part["family"] == "random"]
                ax.scatter(random[x], random["mean_over_B"], s=10, color=CONTEXT, linewidths=0, label="random sets")
                for family in ("tilt", "quantile", "screen", "path", "proposal"):
                    sub = part[part["family"] == family]
                    if len(sub):
                        ax.scatter(sub[x], sub["mean_over_B"], linewidths=0, alpha=0.9, zorder=3,
                                   label=FAMILY_LABELS[family], **FAMILY_STYLE[family])
                for rule in RM.RULES[a]:
                    sub = part[(part["family"] == "rule") & (part["detail"] == rule)]
                    style = RM.METHOD_STYLE[rule]
                    ax.scatter(sub[x], sub["mean_over_B"], s=48, color=style["color"], marker=style["marker"],
                               edgecolors="white", linewidths=0.6, zorder=4, label=RM.METHOD_LABELS[rule])
                if x == "predicted_accuracy":
                    ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.8, linestyle="--", zorder=1)
                    ax.set_xlim(-0.03, 1.03)
                ax.set_title(L.cell_title(a, degree, rho, budget), loc="left", fontsize=9.5)
                ax.set_ylim(-0.03, 1.03)
                if c == 0:
                    ax.set_ylabel("mean test accuracy over 11 Bs")
                if r == len(rows_of) - 1:
                    ax.set_xlabel("learned predicted accuracy" if x == "predicted_accuracy"
                                  else L.FEATURE_LABELS.get(x, x))
        handles, labels = [], []
        for ax in axes.ravel():
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        height = fig.get_figheight()
        title = (f"{MODEL_LABELS[model]}: final selector on its training sets (pair 0, mean over 11 Bs)"
                 if x == "predicted_accuracy" else
                 f"{MODEL_LABELS[model]}: labelled sets, random to rule-built (pair 0, mean over the saved B and "
                 "its redraws)")
        fig.suptitle(title, x=0.01, y=0.995, ha="left", va="top", fontsize=11.5)
        fig.legend(handles, labels, loc="upper left", frameon=False, ncol=6, fontsize=8.5,
                   bbox_to_anchor=(0.005, 1 - 0.38 / height))
        fig.tight_layout(rect=(0, 0, 1, 1 - 1.0 / height))
    return fig


def resilience_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Round 1's rules over the 11 Bs of pair 0: mean accuracy, share of Bs where the run took off (accuracy >= 0.8),
    the worst B, and the random sets' mean for comparison. How often does a good rule fail on an unlucky B?"""

    rows = []
    for key, part in frame.groupby(["model", "algorithm", "mean_degree", "rho", "budget"], sort=False):
        random = part[part["family"] == "random"]
        rules = part[part["family"] == "rule"]
        for _, rule in rules.iterrows():
            rows.append({"model": key[0], "algorithm": key[1], "mean_degree": key[2], "rho": key[3],
                         "budget": key[4], "rule": RM.METHOD_LABELS.get(rule["detail"], rule["detail"]),
                         "mean over Bs": rule["mean_over_B"], "took off (share of Bs)": rule["takeoff_share"],
                         "worst B": rule["min_over_B"], "SD over Bs": rule["sd_over_B"],
                         "random sets: mean": random["mean_over_B"].mean(),
                         "random sets: took off": random["takeoff_share"].mean()})
    return L._order(pd.DataFrame(rows))


def final_models(design: dict | None = None, models=None) -> dict:
    design = design or read_design()
    out = {}
    for cell in cells(design, models, report=lambda _: None):
        path = cell_dir(design, *cell) / "final_model.json"
        if path.exists():
            out[cell] = _read_json(path)
    return out


def add_final_predictions(frame: pd.DataFrame, design: dict | None = None) -> pd.DataFrame:
    """Score saved feature rows with each cell's saved final selector; never refit or run matching."""

    out = frame.copy()
    out["predicted_accuracy"] = np.nan
    fits = final_models(design)
    for cell, part in out.groupby(["model", "algorithm", "mean_degree", "rho", "budget"], sort=False):
        if cell in fits:
            out.loc[part.index, "predicted_accuracy"] = predict(fits[cell], part)[0]
    return out


def plot_fresh_predictions(frame: pd.DataFrame, model: str, algorithm: str):
    """Paired views of the same fresh strategy outputs: learned prediction and frozen score vs observed accuracy.
    Every point is one evaluated strategy's set on one fresh pair. Random sets have one B; all other sets have
    five. A set selected by multiple strategies appears under each strategy's own saved evaluation."""

    import matplotlib.pyplot as plt

    frame = frame[(frame["model"] == model) & (frame["algorithm"] == algorithm)]
    found = _cell_list(frame, model)
    if not found:
        return None
    styles = {"random": dict(color=CONTEXT, marker="o", s=10),
              "best_1000": dict(color="#1baf7a", marker="X", s=30),
              "frozen_built": dict(color="#e87ba4", marker="D", s=28),
              "learned": dict(color="#2a78d6", marker="*", s=65)}
    with plt.rc_context({"font.size": 9}):
        fig, axes = plt.subplots(len(found), 2, figsize=(9.5, 3.1 * len(found) + 1.3),
                                 squeeze=False, sharey=True)
        for r, (a, degree, rho, budget) in enumerate(found):
            part = frame[(frame["mean_degree"] == degree) & np.isclose(frame["rho"], rho)
                         & (frame["budget"] == budget)]
            # One common mask keeps the evaluated points identical in the two columns.
            part = part.dropna(subset=["predicted_accuracy", "score", "mean_over_B"])
            for c, x in enumerate(("predicted_accuracy", "score")):
                ax = axes[r, c]
                L._style(ax)
                for method in ("random", "best_1000", "frozen_built", "learned"):
                    sub = part[part["method"] == method]
                    if len(sub):
                        ax.scatter(sub[x], sub["mean_over_B"], linewidths=0,
                                   alpha=0.5 if method == "random" else 0.85, zorder=2 if method == "random" else 4,
                                   label="random sets (one B)" if method == "random" else METHOD_LABELS[method],
                                   **styles[method])
                for rule in RM.RULES[a]:
                    sub = part[(part["method"] == "rule") & (part["detail"] == rule)]
                    if len(sub):
                        style = RM.METHOD_STYLE[rule]
                        ax.scatter(sub[x], sub["mean_over_B"], s=32, color=style["color"], marker=style["marker"],
                                   edgecolors="white", linewidths=0.4, alpha=0.85, zorder=3,
                                   label=RM.METHOD_LABELS[rule])
                if c == 0:
                    ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.8, linestyle="--", zorder=1)
                    ax.set_xlim(-0.03, 1.03)
                    ax.set_ylabel("observed test accuracy\n5 Bs; random: 1 B")
                ax.set_ylim(-0.03, 1.03)
                ax.set_title(L.cell_title(a, degree, rho, budget), loc="left", fontsize=9.5)
                ax.set_xlabel("learned predicted accuracy" if c == 0 else "frozen weighted score")
        handles, labels = [], []
        for ax in axes.ravel():
            for h, label in zip(*ax.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(h)
                    labels.append(label)
        height = fig.get_figheight()
        fig.suptitle(f"{MODEL_LABELS[model]} · {LABELS[algorithm]}: fresh pairs, same evaluated sets in both columns",
                     x=0.01, y=0.995, ha="left", va="top", fontsize=11)
        fig.legend(handles, labels, loc="upper left", frameon=False, ncol=3, fontsize=8,
                   bbox_to_anchor=(0.005, 1 - 0.35 / height))
        fig.tight_layout(rect=(0, 0, 1, 1 - 1.3 / height))
    return fig


def coefficient_table(design: dict | None = None, models=None) -> pd.DataFrame:
    """Each cell's final selector: standardised coefficients (per SD of each term over the cell's labelled sets),
    in-sample fit, and how many sets of each family it saw."""

    rows = []
    for (model, algorithm, degree, rho, budget), fit in final_models(design, models).items():
        row = {"model": model, "algorithm": algorithm, "mean_degree": degree, "rho": rho, "budget": budget,
               "sets": fit["n_sets"], "in-sample Spearman": fit["in_sample_spearman"],
               "in-sample MAE": fit["in_sample_mae"]}
        row.update({term: beta for term, beta in zip(fit["terms"], fit["beta"][1:])})
        rows.append(row)
    return L._order(pd.DataFrame(rows))


def lofo_table(design: dict | None = None, models=None) -> pd.DataFrame:
    """Leave-one-family-out checks of the final selectors, one row per cell and held-out family."""

    rows = []
    for (model, algorithm, degree, rho, budget), fit in final_models(design, models).items():
        for check in fit.get("lofo", []):
            rows.append({"model": model, "algorithm": algorithm, "mean_degree": degree, "rho": rho,
                         "budget": budget, **check})
    return L._order(pd.DataFrame(rows))


def proposal_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per cell and round: the proposals' predicted and observed mean accuracy over the 11 Bs, against the best rule
    and the best set labelled before the round (pair 0; the selector saw these Bs' other sets, so this is a check of
    the loop, not the test, which is step 4)."""

    rows = []
    for key, part in frame.groupby(["model", "algorithm", "mean_degree", "rho", "budget"], sort=False):
        rules = part[part["family"] == "rule"]
        best_rule = rules.loc[rules["mean_over_B"].idxmax()] if len(rules) else None
        proposals = part[part["family"] == "proposal"]
        for detail, group in proposals.groupby("detail"):
            round_number = int(detail.rsplit("r", 1)[1])
            earlier = part[(part["family"] != "proposal")
                           | (part["detail"].str.extract(r"r(\d+)$")[0].astype(float) < round_number)]
            rows.append({"model": key[0], "algorithm": key[1], "mean_degree": key[2], "rho": key[3],
                         "budget": key[4], "proposals": detail, "sets": len(group),
                         "best observed": group["mean_over_B"].max(), "mean observed": group["mean_over_B"].mean(),
                         "best rule": RM.METHOD_LABELS.get(best_rule["detail"], best_rule["detail"])
                         if best_rule is not None else "",
                         "best rule observed": best_rule["mean_over_B"] if best_rule is not None else np.nan,
                         "best labelled before": earlier["mean_over_B"].max()})
    return L._order(pd.DataFrame(rows))


def read_fresh(design: dict | None = None, models=None) -> pd.DataFrame:
    """Fresh pairs: per pair and set, mean test accuracy over its Bs, with the set's method and features."""

    design = design or read_design()
    frames = []
    for cell in cells(design, models, report=lambda _: None):
        for folder in sorted((cell_dir(design, *cell) / "fresh").glob("pair_*")):
            if not (folder / "sets.csv").exists() or not _count_ids(folder / "runs.csv"):
                continue
            sets = pd.read_csv(folder / "sets.csv")
            labels = _labels(pd.read_csv(folder / "runs.csv"))
            frames.append(sets.join(labels, on="set_id"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fresh_table(fresh: pd.DataFrame, design: dict | None = None) -> pd.DataFrame:
    """Per cell: mean over fresh pairs of each method's accuracy (mean over the pair's Bs; random sets on the saved B
    only), and the learned selector and the built frozen score against the best rule fixed in advance: mean paired
    difference with a 95% interval over pairs, and pairs won / tied / lost (ties within 0.01)."""

    design = design or read_design()
    path = Path(design["output_dir"]) / "best_rules.json"
    best = _read_json(path) if path.exists() else {}
    rows = []
    for key, part in fresh.groupby(["model", "algorithm", "mean_degree", "rho", "budget"], sort=False):
        rule = best.get(cell_key(*key), {}).get("best_rule")
        per_pair = {}
        for pair, group in part.groupby("pair_index"):
            values = {"random": group.loc[group["method"] == "random", "mean_over_B"].mean()}
            for method in ("best_1000", "frozen_built", "learned"):
                values[method] = group.loc[group["method"] == method, "mean_over_B"].mean()
            for name in RM.RULES[key[1]]:
                values[name] = group.loc[(group["method"] == "rule") & (group["detail"] == name), "mean_over_B"].mean()
            values["best_rule"] = values.get(rule, np.nan)
            per_pair[pair] = values
        table = pd.DataFrame(per_pair).T
        row = {"model": key[0], "algorithm": key[1], "mean_degree": key[2], "rho": key[3], "budget": key[4],
               "pairs": len(table), "best rule": RM.METHOD_LABELS.get(rule, rule) if rule else "-"}
        for method in METHOD_ORDER:
            row[method] = table[method].mean() if method in table else np.nan
        for method in ("learned", "frozen_built"):
            diff = (table[method] - table["best_rule"]).dropna()
            if len(diff):
                half = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
                row[f"{method} − best rule"] = diff.mean()
                row[f"{method} − best rule, 95% low"] = diff.mean() - half
                row[f"{method} − best rule, 95% high"] = diff.mean() + half
                row[f"{method}: won/tied/lost"] = (f"{int((diff > 0.01).sum())}/{int((diff.abs() <= 0.01).sum())}/"
                                                   f"{int((diff < -0.01).sum())}")
        rows.append(row)
    return L._order(pd.DataFrame(rows))


def plot_fresh(table: pd.DataFrame):
    """Fresh pairs, one row per cell: mean accuracy of random sets, the best of 1,000 by the frozen score, the frozen
    score built, the best rule (fixed in advance) and the learned selector."""

    import matplotlib.pyplot as plt

    if table.empty:
        print("No fresh-pair results yet.")
        return None
    table = table.reset_index(drop=True)
    labels = [_short(m, a, d, r) + f" · {int(k)}" for m, a, d, r, k in
              table[["model", "algorithm", "mean_degree", "rho", "budget"]].itertuples(index=False)]
    styles = {"random": dict(color=CONTEXT, marker="o"), "best_1000": dict(color="#1baf7a", marker="X"),
              "frozen_built": dict(color="#e87ba4", marker="D"), "best_rule": dict(color="#eb6834", marker="s"),
              "learned": dict(color="#2a78d6", marker="o")}
    with plt.rc_context({"font.size": 9}):
        height = 0.32 * len(table) + 2.0
        fig, ax = plt.subplots(figsize=(8.0, height))
        L._style(ax)
        y = np.arange(len(table))[::-1]
        for method in METHOD_ORDER:
            if method in table:
                ax.scatter(table[method], y, s=46 if method == "learned" else 34, linewidths=0, zorder=3,
                           label=METHOD_LABELS[method], **styles[method])
        ax.set_yticks(y, labels)
        ax.set_ylim(-0.6, len(table) - 0.4)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("test accuracy on fresh pairs (mean over pairs; each pair's Bs averaged)")
        fig.suptitle("Fresh pairs at each cell's budget", x=0.01, y=0.995, ha="left", va="top", fontsize=11)
        fig.legend(*ax.get_legend_handles_labels(), loc="upper left", frameon=False, ncol=3, fontsize=8.5,
                   bbox_to_anchor=(0.005, 1 - 0.32 / height))
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.85 / height))
    return fig


def plot_weights(table: pd.DataFrame):
    """Step 1: per cell, the fitted slope of test accuracy on each core z-score (random sets, each pair's mean
    removed), with 95% intervals. One panel per algorithm; cells grouped by model."""

    import matplotlib.pyplot as plt

    algorithms = [a for a in ALGORITHMS if a in set(table["algorithm"])]
    names = {"degree": "degree", "harmonic": "harmonic", "pair_mass": "pair mass", "uniqueness": "uniqueness"}
    with plt.rc_context({"font.size": 9}):
        fig, axes = plt.subplots(1, len(algorithms), figsize=(6.0 * len(algorithms), 4.8), squeeze=False)
        for ax, algorithm in zip(axes[0], algorithms):
            part = table[table["algorithm"] == algorithm].reset_index(drop=True)
            L._style(ax)
            ax.axhline(0, color=INK2, linewidth=0.8)
            x = np.arange(len(part))
            for j, f in enumerate(CORE):
                offset = (j - 1.5) * 0.17
                ax.errorbar(x + offset, part[f], yerr=[part[f] - part[f"{f} low"], part[f"{f} high"] - part[f]],
                            fmt="o", ms=4, color=SERIES[j], elinewidth=1, capsize=0, label=names[f])
            ax.set_xticks(x, [_short(m, a, d, r).split(" ", 2)[0] + "\n" + f"d{d:g}\nρ{r:g}\nk{int(k)}"
                              for m, a, d, r, k in part[["model", "algorithm", "mean_degree", "rho", "budget"]]
                              .itertuples(index=False)], fontsize=7.5)
            ax.set_title(LABELS[algorithm], loc="left", fontsize=10.5)
            ax.set_ylabel("change in test accuracy per SD")
        axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left")
        fig.suptitle("Step 1: fitted weights per cell (random sets, single runs, within pair)", x=0.01, ha="left",
                     fontsize=11.5)
        fig.tight_layout()
    return fig


def plot_transfer(matrix: pd.DataFrame):
    """Step 1: the transfer matrix as a heatmap (rows: where the score was fitted; columns: where it is tested)."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    colormap = LinearSegmentedColormap.from_list("diverging", ["#eb6834", "#f7f7f5", "#2a78d6"])
    values = matrix.to_numpy(dtype=float)
    with plt.rc_context({"font.size": 8}):
        fig, ax = plt.subplots(figsize=(0.36 * values.shape[1] + 3.2, 0.32 * values.shape[0] + 2.2))
        image = ax.imshow(values, cmap=colormap, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(values.shape[1]), matrix.columns, rotation=90)
        ax.set_yticks(range(values.shape[0]), matrix.index)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, f"{values[i, j]:.1f}".replace("0.", "."), ha="center", va="center", fontsize=6,
                            color=INK if abs(values[i, j]) < 0.6 else "white")
        for labels, line in ((list(matrix.columns), ax.axvline), (list(matrix.index), ax.axhline)):
            for i in range(1, len(labels)):
                before, here = labels[i - 1].split(), labels[i].split()
                if here[1:2] != before[1:2]:          # another algorithm (or the reference rows)
                    line(i - 0.5, color=INK, linewidth=1.2)
                elif here[0] != before[0]:            # another graph model
                    line(i - 0.5, color=INK2, linewidth=0.5, linestyle=":")
        ax.set_xlabel("tested on (pair 0's 100 sets, mean over 11 Bs)")
        ax.set_ylabel("score fitted on (pairs 1-4)")
        fig.colorbar(image, ax=ax, shrink=0.6, label="Spearman")
        ax.set_title("Step 1: does a score fitted in one cell rank sets in another?", loc="left", fontsize=10.5)
        fig.tight_layout()
    return fig


def show(table: pd.DataFrame, digits: int = 2) -> pd.DataFrame:
    """A table for display: readable cell names as the index, numbers rounded."""

    out = table.assign(model=table["model"].map(MODEL_LABELS), algorithm=table["algorithm"].map(LABELS))
    out = out.rename(columns={"mean_degree": "mean degree", "rho": "ρ"})
    index = [c for c in ("model", "algorithm", "mean degree", "ρ", "level", "budget") if c in out.columns]
    out = out.drop(columns=[c for c in ("kind",) if c in out.columns]).set_index(index)
    return out.round(digits)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", choices=("weights", *STEPS))
    parser.add_argument("--round", type=int, help="round number for --step round")
    parser.add_argument("--all", action="store_true", help="every step in order")
    parser.add_argument("--smoke", action="store_true", help="every step on a tiny design, in tmp/learned_selector_smoke")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--models", nargs="+", choices=MODELS, help="only these graph models")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--blas-threads", type=int, default=3)
    args = parser.parse_args(argv)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 200)
    if args.step == "weights":
        table = weight_table()
        print(show(table[["model", "algorithm", "mean_degree", "rho", "budget", *CORE, "R2", "cosine with frozen",
                          "rank agreement with frozen", "degree-harmonic r"]]).to_string())
        print()
        print(transfer_summary(transfer_matrix()).round(2).to_string())
        return
    design = prepare(args.smoke, args.blas_threads)
    if args.status:
        print(status(design, args.models).to_string())
        return
    if args.smoke or args.all:
        ok = run_all(design, args.models, args.jobs)
        if ok and args.smoke:
            print(status(design, args.models).to_string())
            print(f"\nSmoke test finished. Its results are in {design['output_dir']} (delete the folder when done).")
        return
    if not args.step:
        parser.print_help()
        return
    run_step(args.step, design, args.models, args.round, args.jobs)


if __name__ == "__main__":
    main()
