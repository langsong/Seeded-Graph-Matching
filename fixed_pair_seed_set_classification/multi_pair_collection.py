"""Many-pair seed-set collection for pooled prediction models (30 Oct 2026).

Why. The seed-set models so far were fitted on one graph pair per setting (one A and one B), so their weights
describe that pair. Following Prof. Lyzinski's suggestion, this collects the same kind of data on many pairs, so a
pooled model's weights are consistent and generalise, and the score can be used to screen random seed sets
(best of N).

Pairs. Fresh, structure-checked pairs with round 1's graph settings (selection_rule_comparison.settings), from a new
seed stream (phase code 3). They never coincide with round-1, borderline_check or calibration pairs, so round-1
pairs stay free for testing. FAQ and GOAT share their pairs, as in round 1.

Budget. One per setting, where a single run from a uniform random set succeeded about half the time on round 1's
20 pairs (a rough guide from 20 sets; FAQ ER is probably lower, about a third):
    percolation  ER 8, PAPER 7, Chung-Lu 5
    GOAT         ER 15, PAPER 12, Chung-Lu 9
    FAQ          ER 29, PAPER 29, Chung-Lu 27

Seed sets (500 per pair). Every set is a uniform random set. The tail and contrast sets are picked from a
20,000-set random pool by their graph-A features only, never by outcomes, so fitting P(success | features) on them
is not biased; they put more data where best of N picks and where degree and the other features separate.
    random    300  the first 300 sets of the pool: the population that best of N screens
    tail      100  25 for each direction (degree, pair mass, uniqueness, and the sum of their z-scores), spread
                   evenly over the top 5% of the rest of the pool, from the 95th percentile to the very best set
    contrast  100  50 degree-matched pairs: the same number of seed-to-nonseed edges but low against high pair mass
                   (25 pairs) or uniqueness (25 pairs), within the 10th-90th percentile of the random sets' degree
Uniqueness is the share of non-seeds with a unique fingerprint (distances to the seeds, capped at 3; at 2 for FAQ,
where cap 3 is saturated). The pool's core features (degree, harmonic, pair mass, uniqueness) are saved, so any
score can later be turned into a percentile among the pair's random sets.

Runs. One run per set, with round 1's solver settings: percolation with r = 2 and ExpandWhenStuck (success =
accuracy >= 0.5); GOAT with regulariser ER 100, PAPER 300, Chung-Lu 300, and FAQ (success = unseeded accuracy
>= 0.9). FAQ gets one run, as in the fixed-pair data: if restarts are independent, best-of-3 success increases with
single-run success, so sets rank the same, at half the cost. Each set has its own solver seed.

Run from the repository root (ds environment):

    python -m fixed_pair_seed_set_classification.multi_pair_collection
    python -m fixed_pair_seed_set_classification.multi_pair_collection --algorithms percolation goat
    python -m fixed_pair_seed_set_classification.multi_pair_collection --sets-per-pair 1000 --pairs percolation=25 goat=16 faq=12
    python -m fixed_pair_seed_set_classification.multi_pair_collection --status

or call run_all() from multi_pair_models.ipynb. Rows are written set by set, so a stopped run resumes where it
stopped (Ctrl-C, or the notebook's interrupt button, stops the workers too). Pairs run in order of pair index across
all settings, so stopping early leaves every setting with a similar number of pairs. Results go to OUTPUT_DIR, or
to the folder in the environment variable MULTI_PAIR_OUTPUT_DIR if it is set.
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
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache"))

import numpy as np
import pandas as pd

from experiment_common.features import (
    FIXED_PAIR_FEATURE_COLUMNS,
    SIGNATURE_ROBUSTNESS_COLUMNS,
    build_graph_context,
    fixed_pair_seed_set_features,
    signature_robustness_features,
)
from experiment_common.graph_checks import first_passing_pair
from experiment_common.graphs import generate_graph_pair
from experiment_common.provenance import collect_environment, thread_limit, write_json
from selection_rule_comparison import run_comparison as rc
from selection_rule_comparison import settings as S
from selection_rule_comparison.strategies import GraphA

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
OUTPUT_DIR = Path(os.environ.get("MULTI_PAIR_OUTPUT_DIR") or PACKAGE_DIR / "data" / "multi_pair_2026-10-30")
FIXED_PAIR_DATA = PACKAGE_DIR / "data"
ROUND1_DIR = REPO_ROOT / "selection_rule_comparison" / "data" / "round1"
BORDERLINE_DIR = REPO_ROOT / "selection_rule_comparison" / "data" / "borderline" / "percolation"

ALGORITHMS = ("percolation", "goat", "faq")
MODELS = ("sparse_er", "paper_pa", "chung_lu")
BUDGET = {
    ("percolation", "sparse_er"): 8, ("percolation", "paper_pa"): 7, ("percolation", "chung_lu"): 5,
    ("goat", "sparse_er"): 15, ("goat", "paper_pa"): 12, ("goat", "chung_lu"): 9,
    ("faq", "sparse_er"): 29, ("faq", "paper_pa"): 29, ("faq", "chung_lu"): 27,
}
N_PAIRS = {"percolation": 50, "goat": 32, "faq": 24}     # per graph model; about 40 hours with 3 jobs
GROUPS = {"random": 300, "tail": 100, "contrast": 100}   # 500 sets per pair; see groups_for()
POOL_SIZE = 20_000
TAIL_SHARE = 0.05
TAIL_DIRECTIONS = ("degree", "pair_mass", "uniqueness", "combined")
CONTRAST_FEATURES = ("pair_mass", "uniqueness")
CONTRAST_BINS = 5
CONTRAST_PERCENTILES = (10.0, 90.0)
UNIQUENESS_CAP = {"percolation": 3, "goat": 3, "faq": 2}
CORE_FEATURES = ("degree", "harmonic", "pair_mass", "uniqueness")

TRAINING_PHASE_CODE = 3            # round 1 used 1 (pilot) and 2 (main)
SAMPLING_SEED_BASE = 20261030
SOLVER_SEED_BASE = 20261031
HASH_SEED = 20261029               # random fingerprint hashes for the pool's uniqueness
ALGORITHM_CODES = {"percolation": 1, "goat": 2, "faq": 3}
SECONDS_PER_SET = {"percolation": 0.6, "goat": 3.7, "faq": 5.5}   # wall time per run with 3 jobs (round 1)

ALGORITHM_LABELS = {"percolation": "Percolation", "goat": "GOAT", "faq": "SGM (FAQ)"}
MODEL_LABELS = {"sparse_er": "ER", "paper_pa": "PAPER", "chung_lu": "Chung-Lu"}
FEATURE_LABELS = {"degree": "degree", "harmonic": "harmonic", "pair_mass": "pair mass", "uniqueness": "uniqueness"}
FEATURE_COLUMNS = (*FIXED_PAIR_FEATURE_COLUMNS, *SIGNATURE_ROBUSTNESS_COLUMNS)
SAMPLING_FIELDS = ("group", "direction", "tail_rank", "tail_share", "contrast_feature", "contrast_tail",
                   "contrast_pair_id", "degree_bin", "pool_index")


def groups_for(sets_per_pair: int) -> dict[str, int]:
    """Split a number of sets per pair as 60% random, 20% tail, 20% contrast (an even number)."""

    contrast = 2 * round(0.1 * sets_per_pair)
    tail = round(0.2 * sets_per_pair)
    return {"random": sets_per_pair - tail - contrast, "tail": tail, "contrast": contrast}


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _clock() -> str:
    return datetime.datetime.now().astimezone().strftime("%a %H:%M")


def _flag(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def core_columns(algorithm: str) -> dict[str, str]:
    """Names of the core features in sets.csv (the fixed-pair collections' names)."""

    cap = UNIQUENESS_CAP[algorithm]
    return {"degree": "seed_nonseed_degree_mean", "harmonic": "seed_harmonic_mean",
            "pair_mass": "nonseed_witness_pair_mass_mean",
            "uniqueness": f"nonseed_distance_signature_unique_fraction_r{cap}"}


# --------------------------------------------------------------------------
# Pairs
# --------------------------------------------------------------------------

_EXCLUDED: set[int] | None = None


def excluded_pair_seeds() -> set[int]:
    """Pair seeds already used for training, calibration or testing (never reused here)."""

    global _EXCLUDED
    if _EXCLUDED is None:
        seeds = {int(s) for s in S.KNOWN_PAIR_SEEDS}
        for path in ROUND1_DIR.glob("*/*/*/pairs.csv"):
            seeds |= set(pd.read_csv(path)["pair_seed"].astype(int))
        for path in BORDERLINE_DIR.glob("*/pair_*.csv"):
            seeds |= set(pd.read_csv(path, usecols=["pair_seed"])["pair_seed"].astype(int))
        _EXCLUDED = seeds
    return _EXCLUDED


def first_pair_seed(algorithm: str, model: str, pair_index: int) -> int:
    family = S.FAMILY_CODES[S.GRAPH_FAMILY[algorithm]]
    seed = rc._seed(S.PAIR_SEED_BASE, family, S.MODEL_CODES[model], TRAINING_PHASE_CODE, pair_index)
    return seed % (2**32 - 10_000)


def load_pair(algorithm: str, model: str, pair_index: int):
    """The structure-checked training pair with this index (the same for GOAT and FAQ)."""

    config = S.graph_config(algorithm, model)
    start = first_pair_seed(algorithm, model, pair_index)
    excluded = excluded_pair_seeds()
    offset, skipped = 0, 0
    while True:
        seed, A, B, truth, _report, rejected = first_passing_pair(
            lambda s: generate_graph_pair(config, s), start + offset)
        if int(seed) not in excluded:
            break
        skipped += 1
        offset = int(seed) - start + 1
    return int(seed), A, B, truth, {"first_seed_tried": start, "rejected_draws": int(rejected) + offset,
                                    "skipped_used_seeds": skipped}


# --------------------------------------------------------------------------
# The random pool and its core features (vectorised)
# --------------------------------------------------------------------------

def draw_pool(rng: np.random.Generator, n: int, k: int, size: int) -> np.ndarray:
    """`size` distinct uniform random k-subsets of range(n), each sorted."""

    from math import comb

    if size > comb(n, k):  # otherwise the loop below would never finish
        raise ValueError(f"Cannot draw {size:,} different seed sets of size {k} from {n} vertices: only "
                         f"{comb(n, k):,} exist.")
    sets = np.empty((size, k), dtype=np.int64)
    seen: set[bytes] = set()
    filled = 0
    while filled < size:
        block = np.argpartition(rng.random((min(size - filled, 2048), n)), k, axis=1)[:, :k]
        block.sort(axis=1)
        for row in block:
            key = row.tobytes()
            if key in seen:
                continue
            seen.add(key)
            sets[filled] = row
            filled += 1
            if filled == size:
                break
    return sets


class PoolFeatures:
    """Degree, harmonic, pair mass and uniqueness of many seed sets at once.

    Same definitions as experiment_common.features (checked in tests): degree = seed-to-nonseed edges / k;
    harmonic = mean harmonic centrality of the seeds; pair mass = mean over non-seeds of C(w, 2), w = seed
    neighbours; uniqueness = share of non-seeds whose capped distance vector to the seeds is unique (fingerprints
    are compared through random 64-bit hashes).
    """

    def __init__(self, graph: GraphA, cap: int):
        self.n = graph.n
        self.adjacency = graph.adjacency.astype(np.int16)
        self.harmonic = graph.harmonic
        self.distances = graph.capped[cap].astype(np.intp)
        rng = np.random.default_rng(HASH_SEED)
        self.hashes = rng.integers(1, np.iinfo(np.int64).max, size=(self.n, cap + 1), dtype=np.int64).astype(np.uint64)

    def compute(self, sets: np.ndarray, chunk: int = 256) -> dict[str, np.ndarray]:
        m, k = sets.shape
        out = {name: np.empty(m) for name in CORE_FEATURES}
        out["cut_edges"] = np.empty(m, dtype=np.int64)
        for start in range(0, m, chunk):
            block = sets[start:start + chunk]
            b = len(block)
            columns = np.arange(b)
            is_seed = np.zeros((self.n, b), dtype=bool)
            is_seed[block.T, columns[None, :]] = True
            w = self.adjacency[:, block].sum(axis=2, dtype=np.int64)          # (n, b) seed neighbours
            w = np.where(is_seed, 0, w)
            cut = w.sum(axis=0)
            out["cut_edges"][start:start + b] = cut
            out["degree"][start:start + b] = cut / k
            out["pair_mass"][start:start + b] = (w * (w - 1) // 2).sum(axis=0) / (self.n - k)
            out["harmonic"][start:start + b] = self.harmonic[block].mean(axis=1)
            fingerprint = self.hashes[block[None, :, :], self.distances[:, block]].sum(axis=2)  # wraps mod 2**64
            fingerprint[is_seed] = 0
            fingerprint.sort(axis=0)
            new_before = np.ones(fingerprint.shape, dtype=bool)
            new_before[1:] = fingerprint[1:] != fingerprint[:-1]
            new_after = np.ones(fingerprint.shape, dtype=bool)
            new_after[:-1] = fingerprint[:-1] != fingerprint[1:]
            unique = (new_before & new_after & (fingerprint != 0)).sum(axis=0)
            out["uniqueness"][start:start + b] = unique / (self.n - k)
        return out


def sample_candidates(graph: GraphA, algorithm: str, k: int, rng: np.random.Generator,
                      groups: dict | None = None, pool_size: int = POOL_SIZE):
    """The pair's seed sets: indices into a random pool, with their group labels."""

    groups = {**GROUPS, **(groups or {})}
    pool = draw_pool(rng, graph.n, k, pool_size)
    features = PoolFeatures(graph, UNIQUENESS_CAP[algorithm]).compute(pool)
    n_random = groups["random"]
    random_index = np.arange(n_random)
    candidates = [{"group": "random", "pool_index": int(i)} for i in random_index]
    used = np.zeros(pool_size, dtype=bool)
    used[random_index] = True
    rest = np.arange(n_random, pool_size)

    # tail: spread evenly over the top TAIL_SHARE of the rest of the pool, in each direction
    z = {}
    for name in ("degree", "pair_mass", "uniqueness"):
        values = features[name]
        sd = values[random_index].std(ddof=1)
        z[name] = (values - values[random_index].mean()) / sd if sd > 0 else np.zeros(pool_size)
    z["combined"] = z["degree"] + z["pair_mass"] + z["uniqueness"]
    top = max(1, int(round(TAIL_SHARE * len(rest))))
    per_direction, extra = divmod(groups["tail"], len(TAIL_DIRECTIONS))
    for d, direction in enumerate(TAIL_DIRECTIONS):
        count = per_direction + (d < extra)
        if count == 0:
            continue
        order = rest[np.argsort(-z[direction][rest], kind="stable")]
        for target in np.linspace(0, top - 1, count).round().astype(int):
            position = int(target)
            while used[order[position]]:
                position += 1
            used[order[position]] = True
            candidates.append({"group": "tail", "pool_index": int(order[position]), "direction": direction,
                               "tail_rank": position + 1, "tail_share": (position + 1) / len(rest)})

    # contrasts: equal seed-to-nonseed edges, largest gap in the feature, inside the random sets' degree range
    low, high = np.percentile(features["degree"][random_index], CONTRAST_PERCENTILES)
    eligible = rest[(~used[rest]) & (features["degree"][rest] >= low) & (features["degree"][rest] <= high)]
    eligible = eligible[np.argsort(features["degree"][eligible], kind="stable")]
    bins = np.array_split(eligible, CONTRAST_BINS)
    slots = [(b, f) for f in range(len(CONTRAST_FEATURES)) for b in range(CONTRAST_BINS)]
    per_slot, extra = divmod(groups["contrast"] // 2, len(slots))
    pair_id = 0
    for slot, (bin_id, feature_index) in enumerate(slots):
        feature = CONTRAST_FEATURES[feature_index]
        members = bins[bin_id]
        for _ in range(per_slot + (slot < extra)):
            free = members[~used[members]]
            cut = features["cut_edges"][free]
            best, best_gap = None, 0.0
            for value in np.unique(cut):
                group = free[cut == value]
                if len(group) < 2:
                    continue
                values = features[feature][group]
                gap = float(values.max() - values.min())
                if gap > best_gap:
                    best_gap = gap
                    best = (int(group[np.argmin(values)]), int(group[np.argmax(values)]))
            if best is None:
                raise ValueError("No degree-matched pair with a feature gap is left; increase the pool size.")
            for tail, index in zip(("low", "high"), best):
                used[index] = True
                candidates.append({"group": "contrast", "pool_index": index, "contrast_feature": feature,
                                   "contrast_tail": tail, "contrast_pair_id": pair_id, "degree_bin": bin_id})
            pair_id += 1

    info = {"pool_size": pool_size, "groups": groups, "tail_share": TAIL_SHARE,
            "tail_directions": list(TAIL_DIRECTIONS), "contrast_features": list(CONTRAST_FEATURES),
            "contrast_bins": CONTRAST_BINS, "contrast_percentiles": list(CONTRAST_PERCENTILES),
            "contrast_degree_range": [float(low), float(high)], "uniqueness_cap": UNIQUENESS_CAP[algorithm]}
    return pool, features, candidates, info


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------

def solver_seed(pair_seed: int, budget: int, candidate_id: int) -> int:
    return rc._seed(SOLVER_SEED_BASE, pair_seed, budget, candidate_id)


def evaluate(algorithm: str, model: str, pair, seeds: np.ndarray, seed_value: int) -> dict:
    """One run with round 1's solver settings (each SGM run gets its own solver seed)."""

    if algorithm == "percolation":
        return rc.evaluate("percolation", model, pair, seeds, len(seeds), 0)
    A, B, truth, _ = pair
    out = rc._run_sgm_once(A, B, truth, seeds, seed_value, S.SGM["max_iter"], S.SGM["tol"],
                           transport=algorithm == "goat", transport_regularizer=S.GOAT_REGULARIZER[model],
                           transport_tol=S.SGM["transport_tol"], transport_max_iter=S.SGM["transport_max_iter"])
    accuracy = out["final_unseeded_accuracy"]
    return {"solver_seed": seed_value, "final_unseeded_accuracy": accuracy,
            "success": accuracy >= S.SUCCESS_THRESHOLD[algorithm], "success_090": accuracy >= 0.9,
            "objective": out["sgm_objective_score"], "n_iter": out["sgm_n_iter"], "converged": out["sgm_converged"],
            "runtime_seconds": out["sgm_runtime_seconds"], "h0_accuracy": out["h0_accuracy"],
            "matched_fraction": np.nan, "precision": np.nan}


# --------------------------------------------------------------------------
# One pair (one job), written set by set
# --------------------------------------------------------------------------

def pair_dir(algorithm: str, model: str, pair_index: int) -> Path:
    return OUTPUT_DIR / algorithm / model / f"pair_{pair_index:03d}"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def is_complete(algorithm: str, model: str, pair_index: int) -> bool:
    info = pair_dir(algorithm, model, pair_index) / "pair.json"
    return info.exists() and _read_json(info).get("status") == "complete"


def _completed_rows(path: Path) -> set[int]:
    """Candidate ids already written; a row cut off by an interruption is removed first."""

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
    position = header.index("candidate_id")
    return {int(row[position]) for row in complete}


def pair_job(algorithm: str, model: str, pair_index: int, blas_threads: int = 3, groups: dict | None = None) -> str:
    """Sample the pair's seed sets and run each once (resumes from sets.csv)."""

    groups = {**GROUPS, **(groups or {})}
    name = f"{ALGORITHM_LABELS[algorithm]} {MODEL_LABELS[model]} pair {pair_index}"
    folder = pair_dir(algorithm, model, pair_index)
    info_path, sets_path, pool_path = folder / "pair.json", folder / "sets.csv", folder / "pool_features.npz"
    if is_complete(algorithm, model, pair_index):
        return f"{name}: already done"
    started = time.perf_counter()
    with thread_limit(blas_threads):
        seed, A, B, truth, draw = load_pair(algorithm, model, pair_index)
        k = BUDGET[(algorithm, model)]
        graph = GraphA(A)
        rng = np.random.default_rng([SAMPLING_SEED_BASE, seed, k, ALGORITHM_CODES[algorithm]])
        pool, features, candidates, sampling = sample_candidates(graph, algorithm, k, rng, groups)
        chosen = pool[[c["pool_index"] for c in candidates]]
        design = hashlib.sha256(chosen.tobytes()).hexdigest()[:16]
        aligned_b = B[np.ix_(truth, truth)]
        info = {
            "status": "running", "algorithm": algorithm, "model": model, "pair_index": pair_index,
            "pair_seed": seed, **draw, "budget": k, "n_sets": len(candidates), "design_hash": design,
            "hash_a": rc._hash(A), "hash_b": rc._hash(B), "mean_degree_a": float(graph.degree.mean()),
            "max_degree_a": int(graph.degree.max()),
            "kept_edge_fraction": float(((A > 0) & (aligned_b > 0)).sum() / (A > 0).sum()),
            "solver": (S.PERCOLATION if algorithm == "percolation" else
                       {**S.SGM, "transport": algorithm == "goat",
                        "transport_regularizer": S.GOAT_REGULARIZER[model] if algorithm == "goat" else None,
                        "runs_per_set": 1}),
            "success_threshold": S.SUCCESS_THRESHOLD[algorithm], "sampling": sampling, "started_at": _now(),
        }
        folder.mkdir(parents=True, exist_ok=True)
        if info_path.exists():
            saved = _read_json(info_path)
            if (saved.get("design_hash"), saved.get("hash_a"), saved.get("hash_b")) != (design, info["hash_a"],
                                                                                    info["hash_b"]):
                raise RuntimeError(f"{name}: the saved sets come from a different pair or design. "
                                   "Move that folder away, or set a new OUTPUT_DIR.")
            info["started_at"] = saved.get("started_at", info["started_at"])
        write_json(info_path, info)
        if not pool_path.exists():
            np.savez_compressed(pool_path, **{f: features[f].astype(np.float32) for f in CORE_FEATURES},
                                cut_edges=features["cut_edges"].astype(np.int32), random_count=groups["random"])
        done = _completed_rows(sets_path)
        context = build_graph_context(A, compute_seed_features=False)
        pair = (A, B, truth, seed)
        new = 0
        new_file = not sets_path.exists() or sets_path.stat().st_size == 0
        with sets_path.open("a", newline="", encoding="utf-8") as handle:
            writer = None
            for candidate_id, candidate in enumerate(candidates):
                if candidate_id in done:
                    continue
                seeds = np.sort(pool[candidate["pool_index"]])
                out = evaluate(algorithm, model, pair, seeds, solver_seed(seed, k, candidate_id))
                row = {"algorithm": algorithm, "model": model, "pair_index": pair_index, "pair_seed": seed,
                       "budget": k, "candidate_id": candidate_id,
                       **{field: candidate.get(field) for field in SAMPLING_FIELDS},
                       "seed_vertices": json.dumps([int(v) for v in seeds]),
                       "solver_seed": out["solver_seed"], "success": bool(out["success"]),
                       "success_090": bool(out["success_090"]), "accuracy": out["final_unseeded_accuracy"],
                       "objective": out["objective"], "n_iter": out["n_iter"], "converged": out["converged"],
                       "runtime_seconds": out["runtime_seconds"], "h0_accuracy": out["h0_accuracy"],
                       "matched_fraction": out["matched_fraction"], "precision": out["precision"],
                       **fixed_pair_seed_set_features(context, seeds),
                       **signature_robustness_features(context, seeds)}
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    if new_file:
                        writer.writeheader()
                writer.writerow(row)
                handle.flush()
                new += 1
        info.update({"status": "complete", "finished_at": _now(),
                     "minutes_last_session": round((time.perf_counter() - started) / 60, 1)})
        write_json(info_path, info)
    return f"{name}: {new} sets in {(time.perf_counter() - started) / 60:.0f} min"


# --------------------------------------------------------------------------
# Running everything
# --------------------------------------------------------------------------

def planned_tasks(algorithms=ALGORITHMS, models=MODELS, n_pairs: dict | None = None) -> list[tuple[str, str, int]]:
    """All (algorithm, model, pair) jobs, by pair index first so every setting moves forward together."""

    n_pairs = {**N_PAIRS, **(n_pairs or {})}
    tasks = [(a, m, i) for a in algorithms for m in models for i in range(n_pairs[a])]
    return sorted(tasks, key=lambda t: (t[2], ALGORITHMS.index(t[0]), MODELS.index(t[1])))


def _remaining_sets(task, groups: dict | None = None) -> int:
    sets_path = pair_dir(*task) / "sets.csv"
    done = 0
    if sets_path.exists():
        with sets_path.open(encoding="utf-8") as handle:
            done = max(sum(1 for _ in handle) - 1, 0)
    return max(sum({**GROUPS, **(groups or {})}.values()) - done, 0)


def estimate_hours(tasks, jobs: int = 3, groups: dict | None = None) -> float:
    return sum(_remaining_sets(t, groups) * SECONDS_PER_SET[t[0]] for t in tasks) / max(jobs, 1) / 3600


def run_all(algorithms=ALGORITHMS, models=MODELS, n_pairs: dict | None = None, jobs: int = 3,
            blas_threads: int = 3, groups: dict | None = None) -> None:
    """Run every pair that is not complete yet, `jobs` pairs at a time.

    n_pairs: pairs per graph model, e.g. {"goat": 40} (default N_PAIRS). groups: sets per pair by group
    (default GROUPS; groups_for(1000) gives 1,000 sets per pair). Changing groups for pairs already started
    is refused, because their sets would no longer match.
    """

    groups = {**GROUPS, **(groups or {})}
    algorithms = [a for a in ALGORITHMS if a in set(algorithms)]
    models = [m for m in MODELS if m in set(models)]
    tasks = [t for t in planned_tasks(algorithms, models, n_pairs) if not is_complete(*t)]
    if not tasks:
        print(f"All planned pairs are complete. Results are in {OUTPUT_DIR}")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "settings.json", {
        "written_at": _now(), "budgets": {f"{a}/{m}": k for (a, m), k in BUDGET.items()},
        "pairs_per_model": {**N_PAIRS, **(n_pairs or {})}, "groups": groups, "pool_size": POOL_SIZE,
        "tail_share": TAIL_SHARE, "tail_directions": list(TAIL_DIRECTIONS),
        "contrast_features": list(CONTRAST_FEATURES), "contrast_bins": CONTRAST_BINS,
        "contrast_percentiles": list(CONTRAST_PERCENTILES), "uniqueness_cap": UNIQUENESS_CAP,
        "seed_streams": {"pair_phase_code": TRAINING_PHASE_CODE, "sampling": SAMPLING_SEED_BASE,
                         "solver": SOLVER_SEED_BASE, "fingerprint_hash": HASH_SEED},
        "environment": collect_environment(blas_threads)})
    for algorithm in algorithms:
        part = [t for t in tasks if t[0] == algorithm]
        if part:
            print(f"{_clock()}  {ALGORITHM_LABELS[algorithm]}: {len(part)} pairs to run "
                  f"(about {estimate_hours(part, jobs, groups):.1f} h of the total)", flush=True)
    print(f"{_clock()}  {len(tasks)} pairs, {jobs} at a time: about {estimate_hours(tasks, jobs, groups):.0f} hours. "
          "Each line below is one finished pair.", flush=True)
    keep_awake = None
    if sys.platform == "darwin":
        import shutil
        import subprocess
        if shutil.which("caffeinate"):
            keep_awake = subprocess.Popen(["caffeinate", "-i"])
    failures = []
    started = time.perf_counter()

    def finished(n: int, task, message: str | None, error: BaseException | None, remaining) -> None:
        if error is None:
            left = estimate_hours(remaining, jobs, groups)
            print(f"{_clock()}  [{n}/{len(tasks)}] {message}; about {left:.1f} h left", flush=True)
            return
        algorithm, model, pair_index = task
        failures.append(f"{ALGORITHM_LABELS[algorithm]} {MODEL_LABELS[model]} pair {pair_index}")
        print(f"{_clock()}  [{n}/{len(tasks)}] {failures[-1]} FAILED:", flush=True)
        traceback.print_exception(type(error), error, error.__traceback__)

    pool = None
    try:
        if jobs <= 1:
            for n, task in enumerate(tasks, 1):
                try:
                    message = pair_job(*task, blas_threads, groups)
                except Exception as error:  # noqa: BLE001 - report every failure, keep going
                    finished(n, task, None, error, [])
                else:
                    finished(n, task, message, None, tasks[n:])
        else:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor, as_completed
            pool = ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn"),
                                       initializer=_ignore_interrupts)
            futures = {pool.submit(pair_job, *task, blas_threads, groups): task for task in tasks}
            for n, future in enumerate(as_completed(futures), 1):
                try:
                    message = future.result()
                except Exception as error:  # noqa: BLE001 - report every failure, keep going
                    finished(n, futures[future], None, error, [])
                else:
                    finished(n, futures[future], message, None, [t for f, t in futures.items() if not f.done()])
            pool.shutdown(wait=True)
    except KeyboardInterrupt:
        # Stop the workers too: the notebook's interrupt button only reaches this process.
        if pool is not None:
            workers = list((getattr(pool, "_processes", None) or {}).values())
            pool.shutdown(wait=False, cancel_futures=True)
            for worker in workers:
                worker.terminate()
        print(f"\n{_clock()}  Stopped. Finished sets are kept; rerun to resume.", flush=True)
        return
    finally:
        if keep_awake is not None:
            keep_awake.terminate()
    hours = (time.perf_counter() - started) / 3600
    if failures:
        raise RuntimeError(f"Failed: {', '.join(failures)}. Rerun to retry them (finished sets are kept).")
    print(f"{_clock()}  Done in {hours:.1f} h. Results are in {OUTPUT_DIR}", flush=True)


def _ignore_interrupts() -> None:
    """Workers leave Ctrl-C to the main process, which stops them."""

    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def status(algorithms=ALGORITHMS, models=MODELS, n_pairs: dict | None = None,
           groups: dict | None = None) -> pd.DataFrame:
    """Pairs planned, complete and started, and sets written, per setting."""

    total = sum({**GROUPS, **(groups or {})}.values())
    rows = []
    for algorithm in algorithms:
        for model in models:
            planned = [t for t in planned_tasks([algorithm], [model], n_pairs)]
            complete = sum(is_complete(*t) for t in planned)
            started = sum((pair_dir(*t) / "pair.json").exists() for t in planned)
            sets = sum(total - _remaining_sets(t, groups) for t in planned if (pair_dir(*t) / "sets.csv").exists())
            rows.append({"algorithm": ALGORITHM_LABELS[algorithm], "model": MODEL_LABELS[model],
                         "budget": BUDGET[(algorithm, model)], "pairs planned": len(planned),
                         "pairs complete": complete, "pairs started": started, "sets written": sets})
    return pd.DataFrame(rows).set_index(["algorithm", "model"])


# --------------------------------------------------------------------------
# Analysis helpers for multi_pair_models.ipynb
# --------------------------------------------------------------------------

Z = [f"z_{f}" for f in CORE_FEATURES]


def load_sets(algorithms=ALGORITHMS, models=MODELS, complete_only: bool = True) -> pd.DataFrame:
    """All written sets, with z-scored core features (relative to each pair's random sets)."""

    frames = []
    for algorithm in algorithms:
        for model in models:
            for folder in sorted((OUTPUT_DIR / algorithm / model).glob("pair_*")):
                if complete_only and not is_complete(algorithm, model, int(folder.name.split("_")[1])):
                    continue
                if (folder / "sets.csv").exists():
                    frames.append(pd.read_csv(folder / "sets.csv"))
    if not frames:
        raise FileNotFoundError(f"No finished pairs in {OUTPUT_DIR} yet.")
    data = pd.concat(frames, ignore_index=True)
    # A set written twice (two runs of the same pair at once) counts once.
    data = data.drop_duplicates(subset=["algorithm", "model", "pair_index", "candidate_id"], ignore_index=True)
    for column in ("success", "success_090"):
        data[column] = _flag(data[column])
    return standardize(data)


def standardize(data: pd.DataFrame) -> pd.DataFrame:
    """Core features (degree, harmonic, pair mass, uniqueness) as z-scores against the pair's random sets."""

    data = data.copy()
    for feature in CORE_FEATURES:
        data[feature] = np.nan
    for algorithm in data["algorithm"].unique():
        rows = data["algorithm"] == algorithm
        for feature, column in core_columns(algorithm).items():
            data.loc[rows, feature] = data.loc[rows, column]
    keys = ["algorithm", "model", "pair_index"]
    random = data[data["group"] == "random"].groupby(keys)[list(CORE_FEATURES)]
    mean, sd = random.mean(), random.std(ddof=1)
    joined = data[keys].join(mean, on=keys).join(sd, on=keys, rsuffix="_sd")
    for feature in CORE_FEATURES:
        scale = joined[f"{feature}_sd"].replace(0, np.nan)
        data[f"z_{feature}"] = ((data[feature] - joined[feature]) / scale).fillna(0.0)
    return data


def _informative(part: pd.DataFrame) -> pd.DataFrame:
    """Pairs where both outcomes occur (the only ones that say anything once each pair has its own baseline)."""

    rate = part.groupby("pair_index")["success"].transform("mean")
    return part[(rate > 0) & (rate < 1)]


def fit_pooled(part: pd.DataFrame, features=CORE_FEATURES, groups=("random", "tail")) -> pd.Series:
    """Logistic regression on z-scored features with its own intercept for every pair (weights per SD)."""

    from sklearn.linear_model import LogisticRegression

    part = _informative(part[part["group"].isin(groups)])
    columns = [f"z_{f}" for f in features]
    dummies = pd.get_dummies(part["pair_index"]).to_numpy(float)
    X = np.hstack([part[columns].to_numpy(float), dummies])
    model = LogisticRegression(C=1e4, fit_intercept=False, max_iter=10_000).fit(X, part["success"].astype(int))
    return pd.Series(model.coef_[0][:len(columns)], index=list(features))


def pooled_weights(data: pd.DataFrame, algorithm: str, model: str, features=CORE_FEATURES, n_boot: int = 100,
                   seed: int = 2026) -> pd.DataFrame:
    """Pooled weights with 95% intervals from resampling whole pairs."""

    part = data[(data["algorithm"] == algorithm) & (data["model"] == model)]
    estimate = fit_pooled(part, features)
    by_pair = {p: g for p, g in part.groupby("pair_index")}
    pairs = list(by_pair)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(pairs, size=len(pairs), replace=True)
        sample = pd.concat([by_pair[p].assign(pair_index=j) for j, p in enumerate(chosen)], ignore_index=True)
        try:
            boots.append(fit_pooled(sample, features))
        except ValueError:  # a resample without both outcomes
            continue
    boots = pd.DataFrame(boots)
    return pd.DataFrame({"weight": estimate, "low": boots.quantile(0.025), "high": boots.quantile(0.975),
                         "pairs": part["pair_index"].nunique()})


def per_pair_weights(data: pd.DataFrame, algorithm: str, model: str, features=CORE_FEATURES,
                     groups=("random", "tail"), minority: float = 0.1) -> pd.DataFrame:
    """The same model fitted on each pair alone (pairs where the rarer outcome is at least 10% of sets)."""

    from sklearn.linear_model import LogisticRegression

    part = data[(data["algorithm"] == algorithm) & (data["model"] == model) & data["group"].isin(groups)]
    rows = []
    for pair_index, sub in part.groupby("pair_index"):
        y = sub["success"].astype(int).to_numpy()
        if min(y.mean(), 1 - y.mean()) < minority:
            continue
        model_fit = LogisticRegression(C=1e4, max_iter=10_000).fit(sub[[f"z_{f}" for f in features]].to_numpy(float), y)
        rows.append({"pair_index": pair_index, **dict(zip(features, model_fit.coef_[0]))})
    return pd.DataFrame(rows)


def held_out(data: pd.DataFrame, algorithm: str, model: str, features=CORE_FEATURES, n_splits: int = 5) -> pd.DataFrame:
    """Scores for each pair from a model fitted without that pair (grouped cross-validation), plus the
    score's percentile among the pair's 20,000 random pool sets."""

    from sklearn.model_selection import GroupKFold

    part = data[(data["algorithm"] == algorithm) & (data["model"] == model)].copy()
    pairs = part["pair_index"].to_numpy()
    part["score"] = np.nan
    part["percentile"] = np.nan
    columns = [f"z_{f}" for f in features]
    for train, test in GroupKFold(n_splits=min(n_splits, len(np.unique(pairs)))).split(part, groups=pairs):
        weights = fit_pooled(part.iloc[train], features)
        test_part = part.iloc[test]
        part.loc[test_part.index, "score"] = test_part[columns].to_numpy(float) @ weights.to_numpy()
        for pair_index, sub in test_part.groupby("pair_index"):
            pool = pool_scores(algorithm, model, pair_index, weights)
            part.loc[sub.index, "percentile"] = 100 * np.searchsorted(
                pool, part.loc[sub.index, "score"].to_numpy(), side="right") / len(pool)
    return part


def pool_scores(algorithm: str, model: str, pair_index: int, weights: pd.Series) -> np.ndarray:
    """Sorted scores of the pair's 20,000 random pool sets, z-scored like the evaluated sets."""

    with np.load(pair_dir(algorithm, model, pair_index) / "pool_features.npz") as saved:
        n_random = int(saved["random_count"])
        score = np.zeros(len(saved["degree"]))
        for feature, weight in weights.items():
            values = saved[feature]
            sd = values[:n_random].std(ddof=1)
            z = (values - values[:n_random].mean()) / sd if sd > 0 else np.zeros(len(values))
            score += weight * z
    return np.sort(score)


def within_pair_auc(scored: pd.DataFrame, column: str = "score") -> tuple[float, int]:
    """Mean AUC within held-out pairs, on their random sets only."""

    from sklearn.metrics import roc_auc_score

    values = []
    for _, sub in scored[scored["group"] == "random"].groupby("pair_index"):
        if 0 < sub["success"].mean() < 1:
            values.append(roc_auc_score(sub["success"].astype(int), sub[column]))
    return (float(np.mean(values)) if values else np.nan), len(values)


PERCENTILE_BINS = (0, 20, 40, 60, 80, 95, 99, 99.9, 100.0001)
PERCENTILE_LABELS = ("0-20", "20-40", "40-60", "60-80", "80-95", "95-99", "99-99.9", "top 0.1")


def success_by_percentile(scored: pd.DataFrame) -> pd.DataFrame:
    """Success rate of held-out sets by their score's percentile among random sets (random and tail sets)."""

    part = scored[scored["group"].isin(["random", "tail"])].dropna(subset=["percentile"])
    bins = pd.cut(part["percentile"], PERCENTILE_BINS, labels=PERCENTILE_LABELS, right=False)
    table = part.groupby(bins, observed=False)["success"].agg(["mean", "size"])
    return table.rename(columns={"mean": "success", "size": "sets"})


def contrast_table(data: pd.DataFrame) -> pd.DataFrame:
    """Degree-matched contrasts: pairs where only the high-feature set succeeded against only the low one."""

    from scipy.stats import binomtest

    part = data[data["group"] == "contrast"]
    rows = []
    for (algorithm, model, feature), sub in part.groupby(["algorithm", "model", "contrast_feature"], sort=False):
        wide = sub.pivot_table(index=["pair_index", "contrast_pair_id"], columns="contrast_tail", values="success",
                               aggfunc="first").dropna()
        high = int((wide["high"].astype(bool) & ~wide["low"].astype(bool)).sum())
        low = int((wide["low"].astype(bool) & ~wide["high"].astype(bool)).sum())
        rows.append({"algorithm": ALGORITHM_LABELS[algorithm], "model": MODEL_LABELS[model],
                     "feature": FEATURE_LABELS[feature], "contrasts": len(wide), "high wins": high, "low wins": low,
                     "p": binomtest(high, high + low).pvalue if high + low else np.nan})
    return pd.DataFrame(rows).set_index(["algorithm", "model", "feature"]) if rows else pd.DataFrame()


def single_pair_weights(algorithm: str, model: str, features=CORE_FEATURES) -> pd.Series | None:
    """The same model fitted on the old fixed pair's 600 random sets (one pair, one B), for comparison."""

    from sklearn.linear_model import LogisticRegression

    if algorithm == "percolation":
        files = sorted((FIXED_PAIR_DATA / "percolation_fixed_pairs_2026-10-07" / model).glob(
            "pair_*/*_percolation_seed_sets.csv"))
    else:
        files = sorted((FIXED_PAIR_DATA / "sgm_fixed_pairs_2026-10-13" / algorithm / model).glob("pair_*/*_seed_sets.csv"))
    if len(files) != 1:
        return None
    old = pd.read_csv(files[0])
    old = old[old["sampling_strategy"] == "random"]
    if algorithm == "percolation":
        y = (old["final_unseeded_accuracy"] >= S.SUCCESS_THRESHOLD["percolation"]).astype(int)
    else:
        y = _flag(old["sgm_success"]).astype(int)
    columns = [core_columns(algorithm)[f] for f in features]
    X = old[columns].to_numpy(float)
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    fit = LogisticRegression(C=1e4, max_iter=10_000).fit(X, y)
    return pd.Series(fit.coef_[0], index=list(features))


def save_weights(tables: dict, path: Path | None = None) -> Path:
    """Pooled weights for best of N: per SD of the pair's random sets, for the core features."""

    path = OUTPUT_DIR / "pooled_weights.json" if path is None else path
    payload = {}
    for (algorithm, model), table in tables.items():
        payload[f"{algorithm}/{model}"] = {
            "weights_per_sd": {f: float(table.loc[f, "weight"]) for f in table.index},
            "interval_95": {f: [float(table.loc[f, "low"]), float(table.loc[f, "high"])] for f in table.index},
            "feature_columns": {f: core_columns(algorithm)[f] for f in table.index},
            "standardisation": "z-score against the pair's uniform random sets of the same size",
            "budget": BUDGET[(algorithm, model)], "pairs": int(table["pairs"].iloc[0]),
        }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=ALGORITHMS)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--jobs", type=int, default=3, help="pairs run in parallel (default 3)")
    parser.add_argument("--blas-threads", type=int, default=3)
    parser.add_argument("--sets-per-pair", type=int, default=sum(GROUPS.values()),
                        help="seed sets per pair (default 500: 60%% random, 20%% tail, 20%% contrast)")
    parser.add_argument("--pairs", nargs="+", default=[], metavar="ALGORITHM=N",
                        help="pairs per graph model, e.g. --pairs goat=16 faq=12 "
                             "(default percolation=50 goat=32 faq=24)")
    parser.add_argument("--status", action="store_true", help="only show how far the collection is")
    args = parser.parse_args()
    pairs = {}
    for item in args.pairs:
        name, _, value = item.partition("=")
        if name not in ALGORITHMS or not value.isdigit():
            parser.error(f"--pairs takes ALGORITHM=N with ALGORITHM one of {', '.join(ALGORITHMS)}; got {item!r}")
        pairs[name] = int(value)
    if args.status:
        print(status(args.algorithms, args.models, pairs, groups_for(args.sets_per_pair)).to_string())
    else:
        run_all(args.algorithms, args.models, n_pairs=pairs, jobs=args.jobs, blas_threads=args.blas_threads,
                groups=groups_for(args.sets_per_pair))
