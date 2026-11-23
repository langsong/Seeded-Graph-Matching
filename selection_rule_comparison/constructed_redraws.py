"""Constructed seed sets on the 10 redraws of B, with more intermediate sets (11 Nov 2026).

Why. fresh_pair_scores ran each constructed set (metric extremes, pool tails, rules, intermediate sets) once, on its
pair's saved B. Sets scored above every random set succeeded there 88-98% of the time, but one run cannot say
whether a particular set is reliable or was lucky on that B. b_replicates drew 10 new Bs for every pair and ran 100
random sets on them. Here the constructed sets meet the same 10 Bs. Each set gets a success rate over 11 Bs (the
saved B is draw 0) with an interval, and it can be compared with the random sets on exactly the same graphs.

Sets, per pair. All are chosen on A only, with the frozen score and the pair's original normalisation.
1. Every distinct set in selected_sets.csv and metric_sets.csv. Their saved-B runs are reused.
2. Lowest score: the 3 lowest-scoring sets found by swap search, below the random sets.
3. More intermediate sets. Swap paths run from 5 starts (the random sets at the 5th, 35th, 65th and 95th
   percentiles of the pair's random scores, and the top-degree rule set) to 8 targets (the best set of each metric
   group in metric_sets.csv, and the lowest-score set), one vertex at a time. Every set along a path is a candidate.
   40 are kept, 2 per equal-width score bin: 13 bins from the highest random score up to the highest candidate,
   4 across the random sets' range and 3 below it. Within a bin they are spread over targets, then starts, then
   taken nearest the bin centre; places left by empty bins go to bins with candidates to spare. These sets also run
   on the saved B.
The list is written once per pair (catalog.csv) and reused when resuming.

Draws of B. b_replicates' 10 draws, regenerated from its seeds (b_replicates.draw_b). Each draw's hash is checked
against b_replicates' draws.json, so the constructed and the random sets meet exactly the same Bs.

Runs. Every set once on each draw, with round 1's solver settings and a new solver seed per run. The work goes draw
by draw: every set on every pair gets draw 1 before any set gets draw 2. A run stopped early therefore still gives
every set the same number of draws.

Run from the repository root (ds environment), best in a terminal so that it survives a closed notebook:

    python -m selection_rule_comparison.constructed_redraws --jobs 3
    python -m selection_rule_comparison.constructed_redraws --status      # from another window
    python -m selection_rule_comparison.constructed_redraws --sets-only   # build the set lists only
    python -m selection_rule_comparison.constructed_redraws --summary     # tables and figures, no runs

or from constructed_redraws.ipynb. Results go to data/fresh_pair_scores/constructed_redraws/, or to
CONSTRUCTED_REDRAWS_OUTPUT_DIR if it is set. Rows are written run by run, so a stopped run resumes where it stopped
(Ctrl-C stops the workers too).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache"))

import numpy as np
import pandas as pd

from experiment_common.graphs import generate_graph_pair
from experiment_common.provenance import collect_environment, thread_limit, write_json
from fixed_pair_seed_set_classification import multi_pair_collection as MP

from . import b_replicates as BR
from . import fresh_pair_extremes as FE
from . import fresh_pair_scores as F
from . import run_comparison as rc
from . import settings as S
from .strategies import GraphA

OUTPUT_DIR = Path(os.environ.get("CONSTRUCTED_REDRAWS_OUTPUT_DIR") or F.OUTPUT_DIR / "constructed_redraws")
ALGORITHMS, MODELS, CORE = MP.ALGORITHMS, MP.MODELS, MP.CORE_FEATURES
LABELS, MODEL_LABELS = MP.ALGORITHM_LABELS, MP.MODEL_LABELS
N_PAIRS, DRAWS = BR.N_PAIRS, BR.DRAWS
SOLVER_SEED_BASE, BRIDGE_SEED_BASE = 20261200, 20261201
# Rough seconds per run. Constructed sets converge faster than random ones on the saved Bs.
SECONDS_PER_RUN = {"percolation": 0.5, "goat": 2.6, "faq": 1.5}

BRIDGES = {
    "start_quantiles": [0.05, 0.35, 0.65, 0.95],   # random sets nearest these quantiles of the pair's random scores
    "start_rules": ["top_degree"],                 # rule sets from selected_sets.csv
    "targets": ["global_degree", "global_harmonic", "high_cut_degree", "high_pair_mass", "high_uniqueness",
                "high_coverage", "high_score", "low_score"],   # rank-1 set of each metric group; the lowest-score set
    "bins": {"below_random": 3, "random_range": 4, "above_random": 13},   # equal-width score bins per range
    "per_bin": 2,
    "low_sets": 3,           # lowest-score sets kept
    "low_steps_min": 8,      # swap-search steps: max(this, budget)
    "low_swaps": 256,        # swaps tried per step
}
KEYS = ["algorithm", "model", "pair_index"]
SAVED_COLUMNS = ("accuracy", "success", "success_090", "solver_seed", "objective", "n_iter", "converged",
                 "runtime_seconds", "h0_accuracy")
RULES = ("top_degree", "top_harmonic", "clustered", "spread", "distinguishing", "hub_free_distinguishing")
METRIC_ORDER = ("high_score", "global_degree", "global_harmonic", "high_cut_degree", "high_pair_mass",
                "high_uniqueness", "high_coverage")
GROUP_ORDER = (*RULES, *METRIC_ORDER, "score_tail", "degree_tail", "score_bridge", "bridge", "low_score")
GROUP_LABELS = {**F.SELECTION_LABELS, "score_bridge": "intermediate sets (earlier)",
                "bridge": "intermediate sets (new)", "low_score": "lowest score found"}

# Figure colours: validated categorical slots 1-3 (dataviz) with distinct markers, a recessive neutral for random sets.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, NEUTRAL, GRID = "#0b0b0b", "#52514e", "#a3a29c", "#e6e6e6"
FAMILY_STYLES = {
    "intermediate": dict(color=BLUE, marker="o", s=20, label="intermediate sets"),
    "extremes, tails and rules": dict(color=ORANGE, marker="D", s=24, label="metric extremes, pool tails, rules"),
    "lowest score": dict(color=AQUA, marker="v", s=34, label="lowest score found"),
}


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _key(seeds) -> str:
    return json.dumps(sorted(int(v) for v in seeds))


def _sets(strings) -> list[np.ndarray]:
    return [np.asarray(json.loads(s), dtype=np.int64) for s in strings]


def _flag(values) -> pd.Series:
    """True/False written by csv (bool or text) as 1.0/0.0."""
    mapping = {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, "true": 1.0, "false": 0.0, 1: 1.0, 0: 0.0}
    series = pd.Series(values)
    return series.map(lambda v: mapping.get(v, np.nan) if not isinstance(v, float) else v).astype(float)


def task_dir(algorithm: str, model: str, pair_index: int) -> Path:
    return OUTPUT_DIR / algorithm / model / f"pair_{pair_index:03d}"


def describe(task) -> str:
    text = f"{LABELS[task[0]]} {MODEL_LABELS[task[1]]} pair {task[2]}"
    return text if len(task) == 3 else f"{text} draw {task[3]}"


def pair_tasks(algorithms=ALGORITHMS, models=MODELS, n_pairs: int = N_PAIRS) -> list:
    return [(a, m, i) for i in range(n_pairs) for a in algorithms for m in models]


def prepare(blas_threads: int = 3) -> dict:
    """The design every job uses. Other draws or pairs later are fine; other changes need a new folder."""

    path = OUTPUT_DIR / "settings.json"
    fixed = json.loads(json.dumps({
        "version": 1, "blas_threads": int(blas_threads),
        "seed_bases": {"solver": SOLVER_SEED_BASE, "bridges": BRIDGE_SEED_BASE, "draws": BR.DRAW_SEED_BASE},
        "bridges": BRIDGES, "percolation": S.PERCOLATION, "sgm": S.SGM, "goat_regularizers": S.GOAT_REGULARIZER,
        "success_thresholds": S.SUCCESS_THRESHOLD,
    }))
    if path.exists():
        saved = _read_json(path)
        differs = [key for key in fixed if saved.get(key) != fixed[key]]
        if differs:
            raise ValueError(f"The saved design in {path} differs ({', '.join(differs)}). Use the saved setting "
                             "(for example --blas-threads 3), or set CONSTRUCTED_REDRAWS_OUTPUT_DIR to a new folder.")
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(path, {**fixed, "written_at": _now(), "environment": collect_environment(blas_threads)})
    return fixed


# --------------------------------------------------------------------------
# The sets of one pair (catalog.csv)
# --------------------------------------------------------------------------

def existing_sets(task) -> pd.DataFrame:
    """Distinct sets of selected_sets.csv and metric_sets.csv, each with every group it is in and its saved-B run."""

    folder = F.pair_dir(*task)
    frames = []
    for source, name in (("selected", "selected_sets.csv"), ("metric", "metric_sets.csv")):
        path = folder / name
        if not path.exists():
            raise FileNotFoundError(f"{describe(task)}: {name} is missing. Run fresh_pair_scores' selected sets and "
                                    "fresh_pair_extremes first.")
        frames.append(pd.read_csv(path).assign(source=source))
    rows = pd.concat(frames, ignore_index=True)
    rows["key"] = [_key(json.loads(s)) for s in rows["seed_vertices"]]
    rows["membership"] = (rows["source"] + ":" + rows["detail"].astype(str) + ":" +
                          rows["selection_rank"].astype(int).astype(str))
    sources = rows.groupby("key", sort=False)["membership"].agg(";".join)
    first = rows.drop_duplicates("key").set_index("key").loc[sources.index]
    out = pd.DataFrame({"key": sources.index, "seed_vertices": first["seed_vertices"].to_numpy(),
                        "origin": "existing", "sources": sources.to_numpy(),
                        "saved_score": first["score"].to_numpy(float)})
    for column in SAVED_COLUMNS:
        out[f"saved_{column}"] = first[column].to_numpy() if column in first else np.nan
    return out


def lowest_score_sets(features, random_sets: list, random_scores: np.ndarray, n: int, k: int, rng) -> list:
    """The lowest-scoring sets found by sampled single-swap descent from the three lowest random sets."""

    def objective(sets):
        return {"low_score": -np.asarray(features(sets)["score"], dtype=float)}

    order = np.argsort(random_scores, kind="stable")
    starts = [random_sets[j] for j in order[:3]]
    steps = max(BRIDGES["low_steps_min"], k)
    found = FE.search_sets(objective, "low_score", random_sets, -np.asarray(random_scores, dtype=float), starts, n,
                           rng, BRIDGES["low_sets"], steps, BRIDGES["low_swaps"])
    return found[:BRIDGES["low_sets"]]


def swap_path(start: np.ndarray, target: np.ndarray, rng) -> list:
    """Sets met when swapping start into target one vertex at a time, in random order (neither end included)."""

    remove = rng.permutation(np.setdiff1d(start, target))
    add = rng.permutation(np.setdiff1d(target, start))
    current, out = set(int(v) for v in start), []
    for step, (gone, entered) in enumerate(zip(remove, add), 1):
        if step == len(remove):
            break
        current.discard(int(gone))
        current.add(int(entered))
        out.append((np.asarray(sorted(current), dtype=np.int64), step, len(remove)))
    return out


def bridge_edges(scores: np.ndarray, random_scores: np.ndarray, bins: dict) -> np.ndarray:
    """Score-bin edges: equal-width bins below the random sets' range, across it and above it, as far as the
    candidates reach."""

    lo, hi = float(np.min(scores)), float(np.max(scores))
    low, high = float(np.min(random_scores)), float(np.max(random_scores))
    pieces = []
    for (start, end), count in zip(((lo, low), (low, high), (high, hi)),
                                   (bins["below_random"], bins["random_range"], bins["above_random"])):
        start, end = max(start, lo), min(end, hi)
        if end > start and count > 0:
            pieces.append(np.linspace(start, end, count + 1))
    edges = np.unique(np.concatenate(pieces)) if pieces else np.array([lo, hi])
    edges[-1] += 1e-12
    return edges


def pick_bridges(scores: np.ndarray, directions: list, starts: list, edges: np.ndarray, per_bin: int):
    """per_bin candidates per score bin, spread over targets and then starts, nearest the bin centre. Places left
    by empty bins go to bins with candidates to spare. Returns (chosen indices, their bins), by score."""

    scores = np.asarray(scores, dtype=float)
    bins = len(edges) - 1
    which = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, bins - 1)
    pools = {b: [int(j) for j in np.flatnonzero(which == b)] for b in range(bins)}
    chosen, by_bin = {}, {b: Counter() for b in range(bins)}
    used_direction, used_start = Counter(), Counter()

    def take(b):
        centre = (edges[b] + edges[b + 1]) / 2
        j = min(pools[b], key=lambda c: (by_bin[b][directions[c]], used_direction[directions[c]],
                                         used_start[starts[c]], abs(scores[c] - centre), c))
        pools[b].remove(j)
        chosen[j] = b
        by_bin[b][directions[j]] += 1
        used_direction[directions[j]] += 1
        used_start[starts[j]] += 1

    for b in range(bins):
        for _ in range(per_bin):
            if pools[b]:
                take(b)
    while len(chosen) < bins * per_bin and any(pools.values()):
        filled = Counter(chosen.values())
        take(min((b for b in range(bins) if pools[b]), key=lambda b: (filled[b], b)))
    order = sorted(chosen, key=lambda j: (scores[j], j))
    return order, [chosen[j] for j in order]


def catalog_job(task, design: dict) -> str:
    """Build and save the pair's list of constructed sets (search on A only; no matching runs)."""
    import fcntl

    a, m, i = task
    folder = task_dir(*task)
    folder.mkdir(parents=True, exist_ok=True)
    path, meta_path = folder / "catalog.csv", folder / "catalog.json"
    started = time.perf_counter()
    with (folder / ".catalog.lock").open("a") as lock, thread_limit(design["blas_threads"]):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists() and meta_path.exists():
            return f"{describe(task)}: set list already built"
        info, random = BR._fresh(task)
        seed, k, code = int(info["pair_seed"]), int(info["budget"]), MP.ALGORITHM_CODES[a]
        A, B, _ = generate_graph_pair(S.graph_config(a, m), seed)
        if rc._hash(A) != info["hash_a"] or rc._hash(B) != info["hash_b"]:
            raise RuntimeError(f"{describe(task)}: the regenerated pair differs from fresh_pair_scores' saved pair.")
        graph = GraphA(A)
        features = FE.feature_calculator(graph, a, info)
        existing = existing_sets(task)
        known = set(existing["key"])
        random = random.dropna(subset=["score"]).drop_duplicates("candidate_id").reset_index(drop=True)
        random_sets, random_scores = _sets(random["seed_vertices"]), random["score"].to_numpy(float)
        random_keys = {_key(s) for s in random_sets}

        # Lowest-score sets.
        low_rng = np.random.default_rng([BRIDGE_SEED_BASE, seed, code, 99])
        lows = []
        for seeds in lowest_score_sets(features, random_sets, random_scores, graph.n, k, low_rng):
            if _key(seeds) not in known and _key(seeds) not in random_keys:
                lows.append(seeds)
                known.add(_key(seeds))

        # Starts and targets of the swap paths.
        selected = pd.read_csv(F.pair_dir(*task) / "selected_sets.csv")
        metric = pd.read_csv(F.pair_dir(*task) / "metric_sets.csv")
        starts = []
        for q in BRIDGES["start_quantiles"]:
            j = int(np.argmin(np.abs(random_scores - np.quantile(random_scores, q))))
            starts.append((f"random_q{round(q * 100):02d}", random_sets[j]))
        for rule in BRIDGES["start_rules"]:
            rows = selected[selected["detail"] == rule]
            if len(rows):
                starts.append((rule, np.asarray(json.loads(rows.iloc[0]["seed_vertices"]), dtype=np.int64)))
        targets = []
        for name in BRIDGES["targets"]:
            if name == "low_score":
                if lows:
                    targets.append((name, lows[0]))
                continue
            rows = metric[(metric["detail"] == name) & (metric["selection_rank"] == 1)]
            if len(rows):
                targets.append((name, np.asarray(json.loads(rows.iloc[0]["seed_vertices"]), dtype=np.int64)))

        candidates, seen = [], set(known) | random_keys
        for s, (start_name, start) in enumerate(starts):
            for t, (target_name, target) in enumerate(targets):
                rng = np.random.default_rng([BRIDGE_SEED_BASE, seed, code, s, t])
                for seeds, step, length in swap_path(start, target, rng):
                    key = _key(seeds)
                    if key not in seen:
                        seen.add(key)
                        candidates.append({"seeds": seeds, "start": start_name, "direction": target_name,
                                           "step": step, "path_length": length})
        bridges, bins = [], []
        if candidates:
            scores = features(np.stack([c["seeds"] for c in candidates]))["score"]
            edges = bridge_edges(scores, random_scores, BRIDGES["bins"])
            picked, bins = pick_bridges(scores, [c["direction"] for c in candidates],
                                        [c["start"] for c in candidates], edges, BRIDGES["per_bin"])
            bridges = [candidates[j] for j in picked]

        # One table: existing sets first, then the lowest-score sets, then the new intermediate sets.
        new = pd.DataFrame(
            [{"seed_vertices": json.dumps(s.tolist()), "origin": "low_score", "sources": f"new:low_score:{r}"}
             for r, s in enumerate(lows, 1)] +
            [{"seed_vertices": json.dumps(c["seeds"].tolist()), "origin": "bridge", "sources": f"new:bridge:{b + 1}",
              "direction": c["direction"], "start": c["start"], "step": c["step"], "path_length": c["path_length"],
              "bridge_bin": b + 1} for c, b in zip(bridges, bins)])
        catalog = pd.concat([existing.drop(columns="key"), new], ignore_index=True)
        catalog.insert(0, "set_id", np.arange(len(catalog)))
        for column in ("direction", "start", "step", "path_length", "bridge_bin"):
            if column not in catalog:
                catalog[column] = np.nan
        catalog = catalog.assign(algorithm=a, model=m, pair_index=i, pair_seed=seed, budget=k)
        raw = features(np.stack(_sets(catalog["seed_vertices"])))
        for f in (*CORE, *(f"z_{f}" for f in CORE), "total_degree", "coverage", "score"):
            catalog[f] = np.asarray(raw[f], dtype=float)
        catalog["random_score_max"] = float(random_scores.max())
        catalog["random_score_p90"] = float(np.quantile(random_scores, 0.9))
        stored = catalog["saved_score"].to_numpy(float)
        difference = np.nanmax(np.abs(stored - catalog["score"].to_numpy(float))) if np.isfinite(stored).any() else 0
        front = ["algorithm", "model", "pair_index", "pair_seed", "budget", "set_id", "origin", "sources",
                 "direction", "start", "step", "path_length", "bridge_bin", "score", "random_score_max"]
        catalog = catalog[front + [c for c in catalog.columns if c not in front]]
        temporary = path.with_suffix(".tmp")
        catalog.to_csv(temporary, index=False)
        temporary.replace(path)
        counts = catalog["origin"].value_counts().to_dict()
        write_json(meta_path, {
            "algorithm": a, "model": m, "pair_index": i, "pair_seed": seed, "budget": k,
            "hash_a": info["hash_a"], "hash_b": info["hash_b"], "sets": len(catalog), "counts": counts,
            "bridge_candidates": len(candidates), "starts": [name for name, _ in starts],
            "targets": [name for name, _ in targets],
            "max_score_difference_from_saved": float(difference), "bridges": BRIDGES, "written_at": _now(),
            "seconds": round(time.perf_counter() - started, 1)})
    return (f"{describe(task)}: {len(catalog)} sets ({counts.get('existing', 0)} already run on the saved B, "
            f"{counts.get('low_score', 0)} lowest-score, {counts.get('bridge', 0)} new intermediate) in "
            f"{time.perf_counter() - started:.0f} s")


def read_catalog_of(task) -> pd.DataFrame:
    path = task_dir(*task[:3]) / "catalog.csv"
    if not path.exists():
        raise FileNotFoundError(f"{describe(task[:3])}: no set list yet ({path}).")
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# Runs: one pair and one draw of B (one job)
# --------------------------------------------------------------------------

def planned_ids(catalog: pd.DataFrame, draw: int) -> list[int]:
    """Draw 0 is the saved B, already run for the existing sets."""

    if draw == 0:
        return [int(c) for c in catalog.loc[catalog["saved_accuracy"].isna(), "set_id"]]
    return [int(c) for c in catalog["set_id"]]


def _completed(path: Path) -> set[int]:
    """set_ids already written; a row cut off by an interruption is removed first."""

    if not path.exists():
        return set()
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
    position = header.index("set_id")
    return {int(row[position]) for row in complete}


def _b_replicates_hash(a: str, m: str, i: int, draw: int):
    path = BR.task_dir(a, m, i) / "draws.json"
    if not path.exists():
        return None
    return _read_json(path).get("draws", {}).get(str(draw), {}).get("hash_b")


def draw_job(task, design: dict) -> str:
    import fcntl

    a, m, i, r = task
    folder = task_dir(a, m, i)
    catalog = read_catalog_of(task)
    digest = hashlib.sha256((folder / "catalog.csv").read_bytes()).hexdigest()[:16]
    ids = planned_ids(catalog, r)
    path, info_path = folder / f"draw_{r:02d}.csv", folder / f"draw_{r:02d}.json"
    started = time.perf_counter()
    with (folder / f".draw_{r:02d}.lock").open("a") as lock, thread_limit(design["blas_threads"]):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if info_path.exists() and _read_json(info_path).get("catalog_hash", digest) != digest:
            raise RuntimeError(f"{describe(task)}: catalog.csv changed after this draw's runs started, so set_ids "
                               "may no longer match. Set CONSTRUCTED_REDRAWS_OUTPUT_DIR to a new folder.")
        done = _completed(path)
        todo = [c for c in ids if c not in done]
        if not todo:
            return f"{describe(task)}: already done"
        info, _ = BR._fresh(task[:3])
        seed, k = int(info["pair_seed"]), int(info["budget"])
        A, B, truth, probability = BR.regenerate(a, m, seed)
        if rc._hash(A) != info["hash_a"] or rc._hash(B) != info["hash_b"]:
            raise RuntimeError(f"{describe(task)}: the regenerated pair differs from fresh_pair_scores' saved pair.")
        if r == 0:
            B_r, truth_r, rejected = B, truth, 0
        else:
            B_r, truth_r, rejected = BR.draw_b(A, probability, seed, r)
        hash_b, reference = rc._hash(B_r), (_b_replicates_hash(a, m, i, r) if r else info["hash_b"])
        if reference is not None and reference != hash_b:
            raise RuntimeError(f"{describe(task)}: this draw of B differs from b_replicates' draw {r} "
                               f"({hash_b} vs {reference}).")
        aligned = B_r[np.ix_(truth_r, truth_r)]
        write_json(info_path, {
            "draw": r, "catalog_hash": digest, "hash_b": hash_b,
            "same_as_b_replicates": None if reference is None else True,
            "rejected_draws": int(rejected), "edges": int(B_r.sum() // 2),
            "kept_edge_fraction": float(((A > 0) & (aligned > 0)).sum() / (A > 0).sum()), "checked_at": _now()})
        rows = catalog.set_index("set_id")
        new_file = not path.exists() or path.stat().st_size == 0
        written = 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = None
            for c in todo:
                row = rows.loc[c]
                seeds = np.asarray(json.loads(row["seed_vertices"]), dtype=int)
                solver_seed = rc._seed(SOLVER_SEED_BASE, seed, k, MP.ALGORITHM_CODES[a], int(c), r)
                out = MP.evaluate(a, m, (A, B_r, truth_r, seed), seeds, solver_seed)
                record = {"algorithm": a, "model": m, "pair_index": i, "pair_seed": seed, "budget": k,
                          "set_id": int(c), "draw": r, "origin": row["origin"], "score": float(row["score"]),
                          "accuracy": float(out.pop("final_unseeded_accuracy")), **out}
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(record))
                    if new_file:
                        writer.writeheader()
                writer.writerow(record)
                handle.flush()
                written += 1
    return f"{describe(task)}: {written} runs in {(time.perf_counter() - started) / 60:.1f} min"


def _left(task, catalogs: dict) -> int:
    catalog = catalogs.get(task[:3])
    if catalog is None:
        return 0
    done = _completed(task_dir(*task[:3]) / f"draw_{task[3]:02d}.csv")
    return len([c for c in planned_ids(catalog, task[3]) if c not in done])


def _catalogs(tasks) -> dict:
    out = {}
    for t in tasks:
        path = task_dir(*t) / "catalog.csv"
        if path.exists() and (task_dir(*t) / "catalog.json").exists():
            out[t] = pd.read_csv(path)
    return out


def run_all(algorithms=ALGORITHMS, models=MODELS, n_pairs: int = N_PAIRS, n_draws: int = DRAWS, jobs: int = 3,
            blas_threads: int = 3, sets_only: bool = False) -> bool:
    """Build the set lists, then run every set on the saved B (new sets only) and on draws 1..n_draws of B,
    draw by draw. Returns False if stopped."""

    from . import jobs

    design = prepare(blas_threads)
    pairs = pair_tasks(algorithms, models, n_pairs)
    failures = []
    try:
        missing = [t for t in pairs if not (task_dir(*t) / "catalog.json").exists()]
        if missing:
            print(f"{_now()}  Building the set lists for {len(missing)} pairs (searches on A, no matching runs).",
                  flush=True)
            failures += jobs.run_tasks(missing, catalog_job, (design,), describe, lambda t: 60, lambda _: [], jobs)
        catalogs = _catalogs(pairs)
        if catalogs:
            sizes = pd.concat(catalogs.values()).groupby(["algorithm", "model", "origin"]).size()
            table = sizes.unstack("origin", fill_value=0).to_string()
            print(f"Sets per setting (all its pairs):\n{table}", flush=True)
        if sets_only:
            return not failures
        pending, seconds = [], {}
        for r in range(n_draws + 1):
            for t in pairs:
                left = _left((*t, r), catalogs)
                if left:
                    pending.append((*t, r))
                    seconds[(*t, r)] = left * SECONDS_PER_RUN[t[0]]
        if pending:
            runs = sum(seconds[t] / SECONDS_PER_RUN[t[0]] for t in pending)
            hours = sum(seconds.values()) / max(jobs, 1) / 3600
            span = f"{hours:.1f} hours" if hours >= 1 else f"{hours * 60:.0f} min"
            print(f"{_now()}  {len(pending)} pair-draws to run ({runs:,.0f} runs), about {span} with {jobs} jobs "
                  f"(a rough guess). Draw by draw; saving to {OUTPUT_DIR}", flush=True)
            failures += jobs.run_tasks(pending, draw_job, (design,), describe, lambda t: seconds.get(t, 0),
                                     lambda _: [], jobs)
    except KeyboardInterrupt:
        print("Stopped. Finished runs are kept; rerun to resume.", flush=True)
        return False
    if failures:
        raise RuntimeError(f"Failed: {', '.join(failures)}. Rerun to retry (finished runs are kept).")
    print(f"{_now()}  Done. Results are in {OUTPUT_DIR}", flush=True)
    return True


def status(algorithms=ALGORITHMS, models=MODELS, n_pairs: int = N_PAIRS, n_draws: int = DRAWS) -> pd.DataFrame:
    """Runs done per setting and draw of B (all pairs), as done/planned; draw 0 is the saved B (new sets only)."""

    pairs = pair_tasks(algorithms, models, n_pairs)
    catalogs = _catalogs(pairs)
    rows = []
    for a in algorithms:
        for m in models:
            mine = [t for t in pairs if t[:2] == (a, m)]
            row = {"algorithm": a, "model": m, "set lists": f"{sum(t in catalogs for t in mine)}/{len(mine)}",
                   "sets": sum(len(catalogs[t]) for t in mine if t in catalogs)}
            for r in range(n_draws + 1):
                planned = sum(len(planned_ids(catalogs[t], r)) for t in mine if t in catalogs)
                left = sum(_left((*t, r), catalogs) for t in mine)
                row[f"draw {r}"] = f"{planned - left}/{planned}"
            rows.append(row)
    return pd.DataFrame(rows).set_index(["algorithm", "model"])


# --------------------------------------------------------------------------
# Reading the results
# --------------------------------------------------------------------------

def read_catalog() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(OUTPUT_DIR.glob("*/*/pair_*/catalog.csv"))]
    if not frames:
        raise FileNotFoundError(f"No set lists yet in {OUTPUT_DIR}")
    return pd.concat(frames, ignore_index=True)


def read_runs(complete_draws_only: bool = True, catalog: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per set and draw of B: draw 0 is the saved B (the saved runs for the existing sets). With
    complete_draws_only, a pair keeps only the draws that all its sets have finished, so its sets share their Bs."""

    catalog = read_catalog() if catalog is None else catalog
    columns = ["set_id", "draw", *SAVED_COLUMNS]
    frames = []
    for (a, m, i), sets in catalog.groupby(KEYS, sort=False):
        saved = sets[sets["saved_accuracy"].notna()]
        parts = [pd.DataFrame({"set_id": saved["set_id"].to_numpy(), "draw": 0,
                               **{c: saved[f"saved_{c}"].to_numpy() for c in SAVED_COLUMNS}})]
        for path in sorted(task_dir(a, m, i).glob("draw_*.csv")):
            if path.stat().st_size:
                frame = pd.read_csv(path)
                parts.append(frame[[c for c in columns if c in frame.columns]])
        runs = pd.concat(parts, ignore_index=True).dropna(subset=["accuracy"])
        runs["success"] = _flag(runs["success"]).to_numpy()
        runs = runs.dropna(subset=["success"]).drop_duplicates(["set_id", "draw"])
        if complete_draws_only:
            counts = runs.groupby("draw")["set_id"].nunique()
            runs = runs[runs["draw"].isin(counts.index[counts == len(sets)])]
        frames.append(runs.assign(algorithm=a, model=m, pair_index=i))
    data = pd.concat(frames, ignore_index=True)
    return data[KEYS + [c for c in data.columns if c not in KEYS]]


def wilson(successes, trials, z: float = 1.959964):
    """Wilson 95% interval for a binomial proportion."""

    s, n = np.asarray(successes, dtype=float), np.asarray(trials, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = s / n
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def plot_group(sources: str) -> str:
    """One label per set for the figures: rules first, then the score's own extremes, other metrics, tails,
    intermediate sets."""

    details = {part.split(":")[1] for part in str(sources).split(";")}
    return next((name for name in GROUP_ORDER if name in details), "other")


def family(group: str) -> str:
    if group in ("score_bridge", "bridge"):
        return "intermediate"
    if group == "low_score":
        return "lowest score"
    return "extremes, tails and rules"


def random_rates(runs: pd.DataFrame | None = None) -> pd.DataFrame:
    """b_replicates' 100 random sets per pair: success rate over the saved B and its draws. Given the constructed
    sets' runs (read_runs()), only the draws those runs have on each pair are used, so both meet the same Bs."""

    data = BR.read_runs(include_original=True)
    if runs is not None:
        used = runs[KEYS + ["draw"]].drop_duplicates()
        data = data.merge(used, on=KEYS + ["draw"], how="inner")
    rates = BR.set_rates(data)
    rates["wilson_low"], rates["wilson_high"] = wilson(rates["success_rate"] * rates["draws"], rates["draws"])
    return rates


def set_rates(runs: pd.DataFrame | None = None, catalog: pd.DataFrame | None = None,
              random: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per constructed set: success rate over its Bs with a Wilson interval, the set's score and groups,
    and the random sets' mean rate on the same pair and Bs."""

    catalog = read_catalog() if catalog is None else catalog
    runs = read_runs(catalog=catalog) if runs is None else runs
    random = _random_or_none(runs) if random is None else random
    keys = KEYS + ["set_id"]
    rates = runs.groupby(keys, sort=False).agg(draws=("success", "size"), successes=("success", "sum"),
                                               mean_accuracy=("accuracy", "mean")).reset_index()
    rates["success_rate"] = rates["successes"] / rates["draws"]
    rates["wilson_low"], rates["wilson_high"] = wilson(rates["successes"], rates["draws"])
    saved = runs[runs["draw"] == 0].drop_duplicates(keys).set_index(keys)["success"].rename("success_on_saved_b")
    rates = rates.join(saved, on=keys)
    rates = rates.merge(catalog.drop(columns=[c for c in catalog.columns if c.startswith("saved_")]), on=keys,
                        how="left")
    rates["group"] = rates["sources"].map(plot_group)
    rates["family"] = rates["group"].map(family)
    rates["above_random"] = rates["score"] > rates["random_score_max"]
    if random is not None and len(random):
        mean = random.groupby(KEYS)["success_rate"].mean().rename("random_mean_rate")
        rates = rates.join(mean, on=KEYS)
        rates["beats_random"] = rates["wilson_low"] > rates["random_mean_rate"]
    return rates


def _random_or_none(runs: pd.DataFrame | None = None):
    try:
        return random_rates(read_runs() if runs is None else runs)
    except FileNotFoundError:
        return None


def memberships(rates: pd.DataFrame) -> pd.DataFrame:
    """One row per set and group it belongs to (a set can be in several groups)."""

    rows = []
    for row in rates[KEYS + ["set_id", "sources"]].itertuples(index=False):
        for part in str(row.sources).split(";"):
            source, detail, rank = part.split(":")
            rows.append({**{k: getattr(row, k) for k in KEYS}, "set_id": row.set_id, "source": source,
                         "membership": detail, "rank": int(rank)})
    return pd.DataFrame(rows).merge(rates.drop(columns="sources"), on=KEYS + ["set_id"], how="left")


def _pooled_interval(part: pd.DataFrame, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval for the pooled success rate, resampling pairs and then sets within each pair."""

    groups = [g[["successes", "draws"]].to_numpy(float) for _, g in part.groupby(KEYS)]
    if not groups:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = np.empty(reps)
    for b in range(reps):
        s = d = 0.0
        for p in rng.integers(0, len(groups), len(groups)):
            sample = groups[p][rng.integers(0, len(groups[p]), len(groups[p]))]
            s, d = s + sample[:, 0].sum(), d + sample[:, 1].sum()
        values[b] = s / d
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def _summary_row(part: pd.DataFrame, random: pd.DataFrame | None) -> dict:
    low, high = _pooled_interval(part)
    row = {"sets": len(part), "Bs per set": int(part["draws"].median()) if len(part) else 0,
           "saved B (one run)": part["success_on_saved_b"].mean(),
           "success over all Bs": part["successes"].sum() / part["draws"].sum() if len(part) else np.nan,
           "interval low": low, "interval high": high,
           "median set": part["success_rate"].median(), "worst set": part["success_rate"].min(),
           "sets ≥ 0.9": (part["success_rate"] >= 0.9).mean(),
           "sets on every B": (part["successes"] == part["draws"]).mean()}
    if "beats_random" in part:
        row["sets above random (Wilson)"] = part["beats_random"].mean()
    if random is not None and len(part):
        pairs = part[KEYS].drop_duplicates()
        same = random.merge(pairs, on=KEYS)
        row["random sets, same pairs"] = same["success_rate"].mean()
        top = same[same.groupby(KEYS)["score"].rank(ascending=False, method="first") <= 10]
        row["top 10 random by score"] = top["success_rate"].mean()
    return row


def above_random_table(rates: pd.DataFrame, random: pd.DataFrame | None = None, by_model: bool = False) -> pd.DataFrame:
    """Constructed sets scoring above every random set of their pair (lowest-score sets excluded), per algorithm.
    Success over all Bs pools runs; its interval resamples pairs and sets. 'sets above random (Wilson)' is the share
    of sets whose 95% lower bound beats the random sets' mean rate on the same pair."""

    random = _random_or_none() if random is None else random
    return _by_setting(rates[rates["above_random"] & (rates["origin"] != "low_score")], random, by_model)


def membership_table(rates: pd.DataFrame, membership: str = "high_score", by_model: bool = False,
                     random: pd.DataFrame | None = None) -> pd.DataFrame:
    """One group across settings, per algorithm (or per setting) and all together. A set in several groups counts
    here if it is in this one. membership is a detail name, e.g. 'high_score' (the frozen score: 10 best found),
    'high_uniqueness', 'global_degree', 'score_tail', 'top_degree', 'score_bridge', 'bridge' or 'low_score'."""

    random = _random_or_none() if random is None else random
    data = memberships(rates)
    data = data[data["membership"] == membership].drop_duplicates(KEYS + ["set_id"])
    return _by_setting(data, random, by_model)


def _by_setting(data: pd.DataFrame, random: pd.DataFrame | None, by_model: bool) -> pd.DataFrame:
    group = ["algorithm", "model"] if by_model else ["algorithm"]
    order = {a: n for n, a in enumerate(ALGORITHMS)}
    rows = []
    for key, part in data.groupby(group, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(group, key)), **_summary_row(part, random)})
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    table = table.sort_values(group, key=lambda s: s.map(order) if s.name == "algorithm" else s, kind="stable")
    table["algorithm"] = table["algorithm"].map(LABELS)
    if by_model:
        table["model"] = table["model"].map(MODEL_LABELS)
    else:
        table = pd.concat([table, pd.DataFrame([{"algorithm": "all", **_summary_row(data, random)}])],
                          ignore_index=True)
    return table.set_index(group).round(3)


def group_table(rates: pd.DataFrame, algorithm: str | None = None, model: str | None = None) -> pd.DataFrame:
    """Per group (sets in several groups count in each): saved-B success and success over all Bs. 'sets above random
    (Wilson)' compares each set with the random sets' mean rate on its pair (set_rates' random_mean_rate)."""

    data = memberships(rates)
    if algorithm is not None:
        data = data[data["algorithm"] == algorithm]
    if model is not None:
        data = data[data["model"] == model]
    rows = []
    for (a, m, g), part in data.groupby(["algorithm", "model", "membership"], sort=False):
        part = part.drop_duplicates(KEYS + ["set_id"])
        rows.append({"algorithm": LABELS[a], "model": MODEL_LABELS[m], "group": GROUP_LABELS.get(g, g),
                     "mean score": part["score"].mean(), **_summary_row(part, None)})
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    rank = {name: n for n, name in enumerate(GROUP_LABELS.get(g, g) for g in GROUP_ORDER)}
    table = table.sort_values(["algorithm", "model", "group"], key=lambda s: s.map(rank) if s.name == "group" else s,
                              kind="stable")
    return table.set_index(["algorithm", "model", "group"]).round(3)


def pair_table(rates: pd.DataFrame) -> pd.DataFrame:
    """Per pair: how the sets above the random range do, and whether the score still ranks them by success rate."""

    rows = []
    for (a, m, i), part in rates.groupby(KEYS, sort=False):
        top = part[part["above_random"] & (part["origin"] != "low_score")]
        rows.append({"algorithm": LABELS[a], "model": MODEL_LABELS[m], "pair": int(i),
                     "sets": len(part), "Bs per set": int(part["draws"].median()),
                     "above random range": len(top),
                     "their success over all Bs": top["successes"].sum() / max(top["draws"].sum(), 1),
                     "their saved-B success": top["success_on_saved_b"].mean(),
                     "random sets' rate": part["random_mean_rate"].iloc[0] if "random_mean_rate" in part else np.nan,
                     "Spearman, all constructed": BR._spearman(part["score"], part["success_rate"]),
                     "Spearman, above random": BR._spearman(top["score"], top["success_rate"]),
                     "lowest-score sets' rate": part.loc[part["origin"] == "low_score", "success_rate"].mean()})
    return pd.DataFrame(rows).set_index(["algorithm", "model", "pair"]).round(3)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _style(ax):
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.04, 1.04)


def _jitter(values, rng, width=0.012):
    return np.asarray(values, dtype=float) + rng.uniform(-width, width, len(values))


def plot_overview(rates: pd.DataFrame, random: pd.DataFrame | None = None, bins: int = 12):
    """Success rate over the Bs against the frozen score, all pairs of a setting together. Grey: b_replicates'
    random sets. The dark line is the mean rate of all sets in equal-width score bins (bins with 5+ sets). The
    dotted line marks the highest random score (median over pairs). Rates are multiples of 1/11, so dots are
    jittered up or down by about 0.01."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    random = _random_or_none() if random is None else random
    present = [(a, m) for a in ALGORITHMS for m in MODELS if ((rates["algorithm"] == a) & (rates["model"] == m)).any()]
    algorithms = [a for a in ALGORITHMS if any(p[0] == a for p in present)]
    models = [m for m in MODELS if any(p[1] == m for p in present)]
    rng = np.random.default_rng(0)
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5}):
        fig, axes = plt.subplots(len(algorithms), len(models), figsize=(4.4 * len(models), 3.5 * len(algorithms)),
                                 squeeze=False, sharey=True)
        for r, a in enumerate(algorithms):
            for c, m in enumerate(models):
                ax = axes[r, c]
                part = rates[(rates["algorithm"] == a) & (rates["model"] == m)]
                if part.empty:
                    ax.set_visible(False)
                    continue
                xs, ys = [part["score"].to_numpy(float)], [part["success_rate"].to_numpy(float)]
                if random is not None:
                    rnd = random[(random["algorithm"] == a) & (random["model"] == m)]
                    ax.scatter(rnd["score"], _jitter(rnd["success_rate"], rng), s=9, color=NEUTRAL, alpha=0.6,
                               linewidths=0, zorder=2)
                    xs.append(rnd["score"].to_numpy(float))
                    ys.append(rnd["success_rate"].to_numpy(float))
                for name, style in FAMILY_STYLES.items():
                    sub = part[part["family"] == name]
                    style = {k: v for k, v in style.items() if k != "label"}
                    ax.scatter(sub["score"], _jitter(sub["success_rate"], rng), alpha=0.8, edgecolors="white",
                               linewidths=0.5, zorder=3, **style)
                x, y = np.concatenate(xs), np.concatenate(ys)
                edges = np.linspace(x.min(), x.max() + 1e-12, bins + 1)
                which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
                centres = [(x[which == b].mean(), y[which == b].mean()) for b in range(bins) if (which == b).sum() >= 5]
                if centres:
                    cx, cy = zip(*centres)
                    ax.plot(cx, cy, color=INK, linewidth=1.6, marker="o", markersize=3.5, zorder=4)
                ax.axvline(part.groupby("pair_index")["random_score_max"].first().median(), color=INK2,
                           linewidth=0.9, linestyle=":", zorder=1)
                _style(ax)
                draws = int(part["draws"].median())
                ax.set_title(f"{LABELS[a]} · {MODEL_LABELS[m]} · k = {int(part['budget'].iloc[0])} · {draws} Bs",
                             fontsize=10)
                if r == len(algorithms) - 1:
                    ax.set_xlabel("frozen score")
                if c == 0:
                    ax.set_ylabel("success rate over the Bs")
        handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=NEUTRAL, markeredgecolor=NEUTRAL,
                          markersize=5, label="random sets (b_replicates)")]
        handles += [Line2D([0], [0], marker=s["marker"], color="none", markerfacecolor=s["color"],
                           markeredgecolor="white", markersize=7, label=s["label"]) for s in FAMILY_STYLES.values()]
        handles += [Line2D([0], [0], color=INK, linewidth=1.6, marker="o", markersize=3.5,
                           label="mean rate, all sets, by score bin"),
                    Line2D([0], [0], color=INK2, linewidth=0.9, linestyle=":", label="highest random score")]
        fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.suptitle("Constructed and random seed sets: success rate over the saved B and its redraws",
                     y=1.045, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def _group_style(group: str, direction=None) -> dict:
    if group == "bridge":
        color = F.METRIC_STYLES.get(direction, {}).get("color", ORANGE if direction == "low_score" else INK2)
        return dict(marker="o", s=26, facecolors="none", edgecolors=color, linewidths=1.1)
    if group == "low_score":
        return dict(color=ORANGE, marker="v", s=48, edgecolors="white", linewidths=0.6)
    if group in RULES:
        return dict(**F.SELECTION_STYLES[group], edgecolors=INK, linewidths=0.8)
    if group in ("score_tail", "degree_tail"):
        style = F.SELECTION_STYLES[group]
        return dict(marker=style["marker"], s=style["s"], facecolors="none", edgecolors=style["color"],
                    linewidths=1.2)
    return dict(**F.METRIC_STYLES.get(group, dict(color=INK2, marker="o", s=20)), edgecolors="white", linewidths=0.6)


def plot_setting(rates: pd.DataFrame, algorithm: str, model: str, random: pd.DataFrame | None = None,
                 intervals: bool = True):
    """One panel per pair: each constructed set once, by its main group (rules first, then the score's extremes,
    other metric extremes, pool tails, intermediate sets), with its Wilson 95% interval as a thin grey line.
    New intermediate sets are hollow circles in the colour of the metric they head towards (orange: towards the
    lowest score). Grey dots: the random sets."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    random = _random_or_none() if random is None else random
    part = rates[(rates["algorithm"] == algorithm) & (rates["model"] == model)]
    pairs = sorted(part["pair_index"].unique())
    rng = np.random.default_rng(1)
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5}):
        fig, axes = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 4.6), squeeze=False, sharey=True)
        for ax, i in zip(axes[0], pairs):
            sub = part[part["pair_index"] == i]
            if random is not None:
                rnd = random[(random["algorithm"] == algorithm) & (random["model"] == model) &
                             (random["pair_index"] == i)]
                ax.scatter(rnd["score"], _jitter(rnd["success_rate"], rng), s=10, color=NEUTRAL, alpha=0.6,
                           linewidths=0, zorder=2, label="random sets (b_replicates)")
            y = _jitter(sub["success_rate"], rng)
            if intervals:
                ax.vlines(sub["score"], sub["wilson_low"], sub["wilson_high"], color="#dcdbd5", linewidth=0.6,
                          zorder=1)
            for g in GROUP_ORDER:
                chosen = (sub["group"] == g).to_numpy()
                if not chosen.any():
                    continue
                if g == "bridge":
                    for direction in BRIDGES["targets"]:
                        mask = chosen & (sub["direction"] == direction).to_numpy()
                        if mask.any():
                            ax.scatter(sub["score"][mask], y[mask], zorder=3, **_group_style(g, direction))
                    continue
                ax.scatter(sub["score"][chosen], y[chosen], zorder=4 if g in RULES else 3,
                           label=GROUP_LABELS.get(g, g), **_group_style(g))
            ax.axvline(sub["random_score_max"].iloc[0], color=INK2, linewidth=0.9, linestyle=":", zorder=1)
            _style(ax)
            top = sub[sub["above_random"] & (sub["origin"] != "low_score")]
            rate = top["successes"].sum() / max(top["draws"].sum(), 1)
            ax.set_title(f"pair {i} · {len(sub)} sets · {int(sub['draws'].median())} Bs\n"
                         f"above the random range: {len(top)} sets, success {rate:.2f}", fontsize=10)
            ax.set_xlabel("frozen score (dotted: highest random score)")
        axes[0, 0].set_ylabel("success rate over the Bs")
        handles = {}
        for ax in axes[0]:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(label, handle)
        if (part["group"] == "bridge").any():
            handles["new intermediate sets: hollow, in the colour of the group they head to"] = Line2D(
                [0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=INK2, markersize=6)
        fig.legend(list(handles.values()), list(handles), loc="lower center", ncol=4, frameon=False, fontsize=8.5)
        k = int(part["budget"].iloc[0])
        fig.suptitle(f"{LABELS[algorithm]} · {MODEL_LABELS[model]} · n = 600, k = {k}: each constructed set's "
                     "success rate over the saved B and its redraws", fontsize=11.5)
        rows_of_legend = int(np.ceil(len(handles) / 4))
        fig.tight_layout(rect=(0, 0.05 + 0.035 * rows_of_legend, 1, 1))
    return fig


def save_figures(rates: pd.DataFrame | None = None, random: pd.DataFrame | None = None) -> list[Path]:
    import matplotlib.pyplot as plt

    rates = set_rates() if rates is None else rates
    random = _random_or_none() if random is None else random
    directory = OUTPUT_DIR / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    figures = [("overview", plot_overview(rates, random))]
    for a in ALGORITHMS:
        for m in MODELS:
            if ((rates["algorithm"] == a) & (rates["model"] == m)).any():
                figures.append((f"{a}_{m}", plot_setting(rates, a, m, random)))
    for name, fig in figures:
        for suffix in ("png", "pdf"):
            path = directory / f"{name}.{suffix}"
            fig.savefig(path, dpi=170, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)
    return paths


def summarize(save: bool = True) -> dict:
    """The tables, printed; with save, also CSVs and figures in OUTPUT_DIR."""

    runs = read_runs()
    random = _random_or_none(runs)
    rates = set_rates(runs, random=random)
    tables = {"above_random": above_random_table(rates, random),
              "above_random_by_setting": above_random_table(rates, random, by_model=True),
              "score_extremes": membership_table(rates, "high_score", random=random),
              "groups": group_table(rates), "pairs": pair_table(rates)}
    for name, table in tables.items():
        print(f"\n{name}\n{table.to_string()}")
    if save:
        rates.to_csv(OUTPUT_DIR / "set_rates.csv", index=False)
        for name, table in tables.items():
            table.to_csv(OUTPUT_DIR / f"{name}.csv")
        print("\nFigures:", *save_figures(rates, random)[::2], sep="\n  ")
    return {"rates": rates, **tables}


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--pairs", type=int, default=N_PAIRS)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--blas-threads", type=int, default=3)
    parser.add_argument("--status", action="store_true", help="only show how far it is")
    parser.add_argument("--sets-only", action="store_true", help="build the set lists, run nothing")
    parser.add_argument("--summary", action="store_true", help="tables and figures from the finished runs")
    args = parser.parse_args(argv)
    if args.status:
        print(status(args.algorithms, args.models, args.pairs, args.draws).to_string())
    elif args.summary:
        summarize()
    elif run_all(args.algorithms, args.models, args.pairs, args.draws, args.jobs, args.blas_threads,
                 args.sets_only) and not args.sets_only:
        summarize()


if __name__ == "__main__":
    main()
