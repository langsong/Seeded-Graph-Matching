"""Many-pair seed-set collection on SBM graphs, to fit SBM's own pooled weights (plan a, 2 Nov 2026).

Why. The pooled weights (multi_pair_collection.py, multi_pair_models.ipynb) were fitted on ER, PAPER and Chung-Lu
pairs, which have no communities. Rather than ask whether those weights transfer to SBM, this collects the same kind
of data on SBM pairs, so SBM gets its own weights. Comparing them with the other models' weights shows whether the
same features matter once vertices form communities, and whether best of N works on SBM with its own score.

Graphs. The two main settings of sbm_rules.py (selection_rule_comparison/sbm_graphs.py): plain SBM ("ER plus
communities") and degree-corrected SBM ("Chung-Lu plus communities"), 4 equal blocks of 150 vertices with 80% of
edges inside blocks; mean degree, rho and GOAT's regulariser as the matched round-1 model. The pairs come from their
own seed stream (purpose "collection"), so they never coincide with sbm_rules' test or calibration pairs.

Budget. One per algorithm and setting: k50 from sbm_rules' budget calibration, the budget where a single run from a
uniform random set succeeds about half the time (as the training budgets of the multi-pair collection were chosen).
If a calibration is missing, it runs first here and is saved where sbm_rules keeps it, so both experiments share it.

Seed sets (500 per pair) and runs: exactly as in multi_pair_collection.py.
    random    300  the first 300 sets of a 20,000-set random pool
    tail      100  25 for each direction (degree, pair mass, uniqueness, and the sum of their z-scores), spread
                   evenly over the top 5% of the rest of the pool
    contrast  100  50 degree-matched pairs, low against high pair mass (25) or uniqueness (25)
One run per set with round 1's solver settings (FAQ one run, as in the training data), each with its own solver
seed. Besides the usual features, every set records how many blocks its seeds cover and the largest block's share of
its seeds (from the true block labels); the pool's 20,000 sets get the same, so block share can be z-scored and
tested as a fifth feature.

Pairs per setting: percolation 30, GOAT 20, FAQ 16. Run from the repository root (ds environment); about 11 hours
with 3 jobs:

    python -m fixed_pair_seed_set_classification.sbm_collection
    python -m fixed_pair_seed_set_classification.sbm_collection --algorithms goat --settings dcsbm
    python -m fixed_pair_seed_set_classification.sbm_collection --pairs goat=10 faq=8
    python -m fixed_pair_seed_set_classification.sbm_collection --status

or call run_all() from sbm_models.ipynb. Rows are written set by set, so a stopped run resumes where it stopped;
pairs run in order of pair index across settings, so stopping early leaves every setting with a similar number of
pairs. Results go to data/sbm_multi_pair/<algorithm>/<setting>/pair_NNN/ (pair.json, sets.csv, pool_features.npz,
the multi-pair collection's format), or to the folder in the environment variable SBM_MULTI_PAIR_OUTPUT_DIR.
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
from selection_rule_comparison.archive import best_of_n as BN
from selection_rule_comparison.archive import beyond_top_degree as BT
from selection_rule_comparison import run_comparison as rc
from selection_rule_comparison.archive import sbm_graphs as G
from selection_rule_comparison.archive import sbm_rules as SR
from selection_rule_comparison.archive import score_transfer as ST
from selection_rule_comparison import settings as S
from selection_rule_comparison.strategies import GraphA

from . import multi_pair_collection as MP

PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("SBM_MULTI_PAIR_OUTPUT_DIR") or PACKAGE_DIR / "data" / "sbm_multi_pair")

ALGORITHMS = ("percolation", "goat", "faq")
SETTINGS = tuple(G.DEFAULT_SETTINGS)
N_PAIRS = {"percolation": 30, "goat": 20, "faq": 16}     # per setting; about 11 hours with 3 jobs
GROUPS = dict(MP.GROUPS)                                 # 300 random, 100 tail, 100 contrast
CORE = tuple(MP.CORE_FEATURES)
BLOCK_FEATURES = ("largest_block_share", "blocks_covered")

SAMPLING_SEED_BASE = 20261110
SOLVER_SEED_BASE = 20261111
ALGORITHM_CODES = MP.ALGORITHM_CODES
SECONDS_PER_SET = {"percolation": 0.5, "goat": 3.1, "faq": 2.3}   # per set and job with 3 jobs (multi-pair data)

ALGORITHM_LABELS = MP.ALGORITHM_LABELS
SETTING_LABELS = G.SETTING_LABELS
MODEL_LABELS = MP.MODEL_LABELS
FEATURE_LABELS = {**MP.FEATURE_LABELS, "largest_block_share": "largest block's share"}


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def block_features(blocks: np.ndarray, sets: np.ndarray) -> dict[str, np.ndarray]:
    """Largest block's share of the seeds and the number of blocks covered, for many sets at once."""

    sets = np.atleast_2d(np.asarray(sets, dtype=np.int64))
    labels = blocks[sets]
    counts = np.stack([(labels == b).sum(axis=1) for b in range(int(blocks.max()) + 1)], axis=1)
    return {"largest_block_share": counts.max(axis=1) / sets.shape[1], "blocks_covered": (counts > 0).sum(axis=1)}


def budget(algorithm: str, setting: str) -> int | None:
    """k50 from sbm_rules' calibration (None if not calibrated yet, or skipped)."""

    saved = SR.load_calibration(algorithm, setting)
    return int(saved["k50"]) if saved and saved["k50"] else None


# --------------------------------------------------------------------------
# One pair (one job), written set by set
# --------------------------------------------------------------------------

def pair_dir(algorithm: str, setting: str, pair_index: int) -> Path:
    return OUTPUT_DIR / algorithm / setting / f"pair_{pair_index:03d}"


def is_complete(algorithm: str, setting: str, pair_index: int) -> bool:
    info = pair_dir(algorithm, setting, pair_index) / "pair.json"
    return info.exists() and MP._read_json(info).get("status") == "complete"


def pair_job(algorithm: str, setting: str, pair_index: int, blas_threads: int = 3, groups: dict | None = None) -> str:
    """Sample the pair's seed sets and run each once (resumes from sets.csv)."""

    groups = {**GROUPS, **(groups or {})}
    name = f"{ALGORITHM_LABELS[algorithm]} {SETTING_LABELS[setting]} pair {pair_index}"
    folder = pair_dir(algorithm, setting, pair_index)
    info_path, sets_path, pool_path = folder / "pair.json", folder / "sets.csv", folder / "pool_features.npz"
    if is_complete(algorithm, setting, pair_index):
        return f"{name}: already done"
    k = budget(algorithm, setting)
    if k is None:
        raise RuntimeError(f"{name}: no calibrated budget. Run the calibration first (run_all does).")
    base = SR.base_model(setting)
    started = time.perf_counter()
    with thread_limit(blas_threads):
        seed, A, B, truth, blocks = G.load_pair(algorithm, setting, "collection", pair_index)
        graph = GraphA(A)
        rng = np.random.default_rng([SAMPLING_SEED_BASE, seed, k, ALGORITHM_CODES[algorithm]])
        pool, features, candidates, sampling = MP.sample_candidates(graph, algorithm, k, rng, groups)
        chosen = pool[[c["pool_index"] for c in candidates]]
        design = hashlib.sha256(chosen.tobytes()).hexdigest()[:16]
        summary = G.pair_summary(A, B, truth, blocks)
        info = {
            "status": "running", "algorithm": algorithm, "model": setting, "setting": setting,
            "kind": G.kind(setting), "base_model": base, "graphs": G.describe(algorithm, setting),
            "pair_index": pair_index, "pair_seed": seed, "budget": k, "budget_source": "k50 from sbm_rules",
            "n_sets": len(candidates), "design_hash": design,
            **{key: summary[key] for key in ("hash_a", "hash_b", "mean_degree_a", "max_degree_a", "within_share_a",
                                             "kept_edge_fraction")},
            "blocks_a": json.loads(summary["blocks_a"]),
            "solver": (S.PERCOLATION if algorithm == "percolation" else
                       {**S.SGM, "transport": algorithm == "goat",
                        "transport_regularizer": G.regularizer(algorithm, setting), "runs_per_set": 1}),
            "success_threshold": S.SUCCESS_THRESHOLD[algorithm], "sampling": sampling,
            "seed_streams": {"pair": G.PAIR_SEED_BASE, "sampling": SAMPLING_SEED_BASE, "solver": SOLVER_SEED_BASE,
                             "fingerprint_hash": MP.HASH_SEED},
            "started_at": _now(),
        }
        folder.mkdir(parents=True, exist_ok=True)
        if info_path.exists():
            saved = MP._read_json(info_path)
            if (saved.get("design_hash"), saved.get("hash_a"), saved.get("hash_b")) != (design, info["hash_a"],
                                                                                    info["hash_b"]):
                raise RuntimeError(f"{name}: the saved sets come from a different pair or design. "
                                   "Move that folder away, or set a new SBM_MULTI_PAIR_OUTPUT_DIR.")
            info["started_at"] = saved.get("started_at", info["started_at"])
        write_json(info_path, info)
        if not pool_path.exists():
            extra = block_features(blocks, pool)
            np.savez_compressed(pool_path, **{f: features[f].astype(np.float32) for f in CORE},
                                cut_edges=features["cut_edges"].astype(np.int32), random_count=groups["random"],
                                largest_block_share=extra["largest_block_share"].astype(np.float32),
                                blocks_covered=extra["blocks_covered"].astype(np.int8))
        done = MP._completed_rows(sets_path)
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
                out = MP.evaluate(algorithm, base, pair, seeds, rc._seed(SOLVER_SEED_BASE, seed, k, candidate_id))
                extra = block_features(blocks, seeds)
                row = {"algorithm": algorithm, "model": setting, "kind": G.kind(setting), "base_model": base,
                       "pair_index": pair_index, "pair_seed": seed, "budget": k, "candidate_id": candidate_id,
                       **{field: candidate.get(field) for field in MP.SAMPLING_FIELDS},
                       "seed_vertices": json.dumps([int(v) for v in seeds]),
                       "solver_seed": out["solver_seed"], "success": bool(out["success"]),
                       "success_090": bool(out["success_090"]), "accuracy": out["final_unseeded_accuracy"],
                       "objective": out["objective"], "n_iter": out["n_iter"], "converged": out["converged"],
                       "runtime_seconds": out["runtime_seconds"], "h0_accuracy": out["h0_accuracy"],
                       "matched_fraction": out["matched_fraction"], "precision": out["precision"],
                       "largest_block_share": float(extra["largest_block_share"][0]),
                       "blocks_covered": int(extra["blocks_covered"][0]),
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

def planned_tasks(algorithms=ALGORITHMS, settings=SETTINGS, n_pairs: dict | None = None) -> list[tuple]:
    """Calibrations still missing, then all (algorithm, setting, pair) jobs by pair index across settings."""

    n_pairs = {**N_PAIRS, **(n_pairs or {})}
    calibrations = [("calibration", a, s, None) for a in algorithms for s in settings
                    if SR.load_calibration(a, s) is None]
    pairs = [("pair", a, s, i) for a in algorithms for s in settings for i in range(n_pairs[a])
             if SR.load_calibration(a, s) is not None and budget(a, s) and not is_complete(a, s, i)]
    pairs.sort(key=lambda t: (t[3], ALGORITHMS.index(t[1]), list(G.SETTINGS).index(t[2])))
    return calibrations + pairs


def _remaining_sets(task: tuple, groups: dict) -> int:
    sets_path = pair_dir(task[1], task[2], task[3]) / "sets.csv"
    done = 0
    if sets_path.exists():
        with sets_path.open(encoding="utf-8") as handle:
            done = max(sum(1 for _ in handle) - 1, 0)
    return max(sum(groups.values()) - done, 0)


def _task_seconds(task: tuple, groups: dict, n_pairs: dict) -> float:
    kind, algorithm, _setting, _ = task
    if kind == "calibration":
        runs = SR.CALIBRATION_SETS * len(SR.CALIBRATION_PAIRS) * ST.CALIBRATION_STEPS
        return (runs + n_pairs[algorithm] * sum(groups.values())) * SECONDS_PER_SET[algorithm]
    return _remaining_sets(task, groups) * SECONDS_PER_SET[algorithm]


def estimate_hours(tasks, jobs: int = 3, groups: dict | None = None, n_pairs: dict | None = None) -> float:
    groups, n_pairs = {**GROUPS, **(groups or {})}, {**N_PAIRS, **(n_pairs or {})}
    return sum(_task_seconds(t, groups, n_pairs) for t in tasks) / max(jobs, 1) / 3600


def _execute(task: tuple, blas_threads: int, groups: dict) -> str:
    kind, algorithm, setting, pair_index = task
    if kind == "calibration":
        return SR.calibration_job(algorithm, setting, blas_threads)
    return pair_job(algorithm, setting, pair_index, blas_threads, groups)


def _describe(task: tuple) -> str:
    kind, algorithm, setting, pair_index = task
    where = f"{ALGORITHM_LABELS[algorithm]} {SETTING_LABELS[setting]}"
    return f"{where} · budget calibration" if kind == "calibration" else f"{where} pair {pair_index}"


def run_all(algorithms=ALGORITHMS, settings=SETTINGS, n_pairs: dict | None = None, jobs: int = 3,
            blas_threads: int = 3, groups: dict | None = None) -> None:
    """Run every pair that is not complete yet, `jobs` at a time (calibrating budgets first where needed).

    n_pairs: pairs per setting by algorithm, e.g. {"goat": 10} (default N_PAIRS). groups: sets per pair by group
    (default 300 random, 100 tail, 100 contrast; MP.groups_for(1000) gives 1,000 sets per pair). Changing groups for
    pairs already started is refused, because their sets would no longer match.
    """

    groups = {**GROUPS, **(groups or {})}
    n_pairs = {**N_PAIRS, **(n_pairs or {})}
    algorithms = [a for a in ALGORITHMS if a in set(algorithms)]
    unknown = sorted(set(settings) - set(G.SETTINGS))
    if unknown:
        raise ValueError(f"Unknown settings {unknown}; choose from {list(G.SETTINGS)}.")
    settings = [s for s in G.SETTINGS if s in set(settings)]
    weights = SR.averaged_weights()
    for algorithm in algorithms:            # the calibrations live with sbm_rules' results and its settings
        for setting in settings:
            SR._check_and_save_settings(algorithm, setting, weights, blas_threads)
    tasks = planned_tasks(algorithms, settings, n_pairs)
    if not tasks:
        print(f"All planned pairs are complete. Results are in {OUTPUT_DIR}")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "settings.json", {
        "written_at": _now(), "settings": {s: G.SETTINGS[s] for s in settings},
        "budgets": {f"{a}/{s}": budget(a, s) for a in algorithms for s in settings},
        "pairs_per_setting": n_pairs, "groups": groups, "pool_size": MP.POOL_SIZE, "tail_share": MP.TAIL_SHARE,
        "tail_directions": list(MP.TAIL_DIRECTIONS), "contrast_features": list(MP.CONTRAST_FEATURES),
        "contrast_bins": MP.CONTRAST_BINS, "contrast_percentiles": list(MP.CONTRAST_PERCENTILES),
        "uniqueness_cap": MP.UNIQUENESS_CAP,
        "seed_streams": {"pair": G.PAIR_SEED_BASE, "pair_purpose": "collection", "sampling": SAMPLING_SEED_BASE,
                         "solver": SOLVER_SEED_BASE, "fingerprint_hash": MP.HASH_SEED},
        "environment": collect_environment(blas_threads)})
    for algorithm in algorithms:
        part = [t for t in tasks if t[1] == algorithm]
        if part:
            print(f"{MP._clock()}  {ALGORITHM_LABELS[algorithm]}: {len(part)} tasks "
                  f"(about {estimate_hours(part, jobs, groups, n_pairs):.1f} h of the total)", flush=True)
    print(f"{MP._clock()}  {len(tasks)} tasks, {jobs} at a time: about "
          f"{estimate_hours(tasks, jobs, groups, n_pairs):.1f} hours. Each line below is one finished task.",
          flush=True)

    def follow_ups(task):
        if task[0] != "calibration" or not budget(task[1], task[2]):
            return []
        return [("pair", task[1], task[2], i) for i in range(n_pairs[task[1]]) if not is_complete(task[1], task[2], i)]

    started = time.perf_counter()
    try:
        failures = SR.run_tasks(tasks, _execute, (blas_threads, groups), _describe,
                                lambda task: _task_seconds(task, groups, n_pairs), follow_ups, jobs)
    except KeyboardInterrupt:
        return
    if failures:
        raise RuntimeError(f"Failed: {', '.join(failures)}. Rerun to retry them (finished sets are kept).")
    print(f"{MP._clock()}  Done in {(time.perf_counter() - started) / 3600:.1f} h. Results are in {OUTPUT_DIR}",
          flush=True)


def status(algorithms=ALGORITHMS, settings=SETTINGS, n_pairs: dict | None = None,
           groups: dict | None = None) -> pd.DataFrame:
    """Budget, pairs planned, complete and started, and sets written, per algorithm and setting."""

    n_pairs = {**N_PAIRS, **(n_pairs or {})}
    groups = {**GROUPS, **(groups or {})}
    rows = []
    for algorithm in algorithms:
        for setting in settings:
            planned = [("pair", algorithm, setting, i) for i in range(n_pairs[algorithm])]
            saved = SR.load_calibration(algorithm, setting)
            shown = ("not calibrated yet" if saved is None else
                     saved["k50"] if saved["k50"] else f"skipped ({saved['status']})")
            rows.append({"algorithm": ALGORITHM_LABELS[algorithm], "setting": SETTING_LABELS[setting],
                         "budget (k50)": shown, "pairs planned": len(planned),
                         "pairs complete": sum(is_complete(algorithm, setting, t[3]) for t in planned),
                         "pairs started": sum((pair_dir(algorithm, setting, t[3]) / "pair.json").exists()
                                              for t in planned),
                         "sets written": sum(sum(groups.values()) - _remaining_sets(t, groups) for t in planned
                                             if (pair_dir(algorithm, setting, t[3]) / "sets.csv").exists())})
    return pd.DataFrame(rows).set_index(["algorithm", "setting"])


# --------------------------------------------------------------------------
# Analysis helpers for sbm_models.ipynb
# --------------------------------------------------------------------------

def _zscore(data: pd.DataFrame, column: str) -> pd.Series:
    """A feature in SDs of the pair's random sets (0 where it does not vary)."""

    keys = ["algorithm", "model", "pair_index"]
    random = data[data["group"] == "random"].groupby(keys)[column]
    joined = data[keys].join(random.mean().rename("_mean"), on=keys).join(random.std(ddof=1).rename("_sd"), on=keys)
    return ((data[column] - joined["_mean"]) / joined["_sd"].replace(0, np.nan)).fillna(0.0)


def load_sets(algorithms=ALGORITHMS, settings=SETTINGS, complete_only: bool = True) -> pd.DataFrame:
    """All written sets, with the core features and block share z-scored against each pair's random sets."""

    frames = []
    for algorithm in algorithms:
        for setting in settings:
            for folder in sorted((OUTPUT_DIR / algorithm / setting).glob("pair_*")):
                if complete_only and not is_complete(algorithm, setting, int(folder.name.split("_")[1])):
                    continue
                if (folder / "sets.csv").exists():
                    frames.append(pd.read_csv(folder / "sets.csv"))
    if not frames:
        raise FileNotFoundError(f"No finished pairs in {OUTPUT_DIR} yet.")
    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=["algorithm", "model", "pair_index", "candidate_id"], ignore_index=True)
    for column in ("success", "success_090"):
        data[column] = MP._flag(data[column])
    data = MP.standardize(data)
    for feature in BLOCK_FEATURES:
        data[f"z_{feature}"] = _zscore(data, feature)
    return data


def _settings_in(data: pd.DataFrame) -> list[tuple[str, str]]:
    present = set(zip(data["algorithm"], data["model"]))
    return [(a, s) for a in ALGORITHMS for s in G.SETTINGS if (a, s) in present]


def overview(data: pd.DataFrame) -> pd.DataFrame:
    """Pairs, budget, and how often random sets succeed (mean over pairs and its spread)."""

    rows = []
    for algorithm, setting in _settings_in(data):
        part = data[(data["algorithm"] == algorithm) & (data["model"] == setting) & (data["group"] == "random")]
        rates = part.groupby("pair_index")["success"].mean()
        rows.append({"algorithm": ALGORITHM_LABELS[algorithm], "setting": SETTING_LABELS[setting],
                     "budget": int(part["budget"].iloc[0]), "pairs": len(rates),
                     "random success, mean": round(float(rates.mean()), 2),
                     "random success, lowest to highest pair": f"{rates.min():.2f} to {rates.max():.2f}",
                     "pairs with both outcomes": int(((rates > 0) & (rates < 1)).sum()),
                     "mean largest block share (random sets)": round(float(part["largest_block_share"].mean()), 2)})
    return pd.DataFrame(rows).set_index(["algorithm", "setting"])


def pooled_weights(data: pd.DataFrame, algorithm: str, setting: str, features=CORE, n_boot: int = 100) -> pd.DataFrame:
    """SBM's own pooled weights (per SD) with 95% intervals from resampling whole pairs (MP.pooled_weights)."""

    return MP.pooled_weights(data, algorithm, setting, features, n_boot=n_boot)


def weights_table(data: pd.DataFrame, n_boot: int = 100, weights_path: Path | None = None) -> pd.DataFrame:
    """The pooled weights of every model side by side: ER, PAPER and Chung-Lu (saved) and the SBM settings (fitted
    here). One row per algorithm, model and feature, with 95% intervals."""

    saved = json.loads(Path(weights_path or BN.WEIGHTS_PATH).read_text(encoding="utf-8"))
    rows = []
    for algorithm in ALGORITHMS:
        for model in BN.MODELS:
            entry = saved.get(f"{algorithm}/{model}")
            if entry is None or algorithm not in set(data["algorithm"]):
                continue
            for feature in CORE:
                low, high = entry["interval_95"][feature]
                rows.append({"algorithm": algorithm, "model": model, "feature": feature,
                             "weight": entry["weights_per_sd"][feature], "low": low, "high": high,
                             "pairs": entry["pairs"], "budget": entry["budget"]})
    for algorithm, setting in _settings_in(data):
        table = pooled_weights(data, algorithm, setting, n_boot=n_boot)
        k = int(data.loc[(data["algorithm"] == algorithm) & (data["model"] == setting), "budget"].iloc[0])
        for feature, row in table.iterrows():
            rows.append({"algorithm": algorithm, "model": setting, "feature": feature, "weight": row["weight"],
                         "low": row["low"], "high": row["high"], "pairs": int(row["pairs"]), "budget": k})
    return pd.DataFrame(rows)


def show_weights(table: pd.DataFrame) -> pd.DataFrame:
    """weights_table as one column per model: weight [95% interval]."""

    labels = {**MODEL_LABELS, **SETTING_LABELS}
    shown = table.assign(algorithm=table["algorithm"].map(ALGORITHM_LABELS), model=table["model"].map(labels),
                         feature=table["feature"].map(FEATURE_LABELS),
                         value=[f"{w:+.2f} [{lo:+.2f}, {hi:+.2f}]" for w, lo, hi in
                                zip(table["weight"], table["low"], table["high"])])
    order = [labels[m] for m in (*BN.MODELS, *G.SETTINGS) if m in set(table["model"])]
    wide = shown.pivot_table(index=["algorithm", "feature"], columns="model", values="value", aggfunc="first")
    algorithms = [ALGORITHM_LABELS[a] for a in ALGORITHMS if ALGORITHM_LABELS[a] in set(shown["algorithm"])]
    features = [FEATURE_LABELS[f] for f in CORE]
    wide = wide.reindex(pd.MultiIndex.from_product([algorithms, features], names=["algorithm", "feature"]))
    return wide[[c for c in order if c in wide.columns]]


def pool_scores(algorithm: str, setting: str, pair_index: int, weights: pd.Series) -> np.ndarray:
    """Sorted scores of the pair's 20,000 random pool sets, z-scored like the evaluated sets."""

    with np.load(pair_dir(algorithm, setting, pair_index) / "pool_features.npz") as saved:
        n_random = int(saved["random_count"])
        score = np.zeros(len(saved["degree"]))
        for feature, weight in weights.items():
            values = saved[feature].astype(float)
            sd = values[:n_random].std(ddof=1)
            z = (values - values[:n_random].mean()) / sd if sd > 0 else np.zeros(len(values))
            score += weight * z
    return np.sort(score)


def held_out(data: pd.DataFrame, algorithm: str, setting: str, features=CORE, n_splits: int = 5) -> pd.DataFrame:
    """Scores for each pair from a model fitted without that pair (grouped cross-validation), plus each score's
    percentile among the pair's 20,000 random pool sets (as MP.held_out)."""

    from sklearn.model_selection import GroupKFold

    part = data[(data["algorithm"] == algorithm) & (data["model"] == setting)].copy()
    pairs = part["pair_index"].to_numpy()
    part["score"] = np.nan
    part["percentile"] = np.nan
    columns = [f"z_{f}" for f in features]
    for train, test in GroupKFold(n_splits=min(n_splits, len(np.unique(pairs)))).split(part, groups=pairs):
        weights = MP.fit_pooled(part.iloc[train], features)
        test_part = part.iloc[test]
        part.loc[test_part.index, "score"] = test_part[columns].to_numpy(float) @ weights.to_numpy()
        for pair_index, sub in test_part.groupby("pair_index"):
            pool = pool_scores(algorithm, setting, pair_index, weights)
            part.loc[sub.index, "percentile"] = 100 * np.searchsorted(
                pool, part.loc[sub.index, "score"].to_numpy(), side="right") / len(pool)
    return part


def _pair_aucs(part: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """Within-pair AUC on the random sets, for pairs where random sets both succeed and fail."""

    from sklearn.metrics import roc_auc_score

    frame = part.assign(_score=score)
    values = []
    for _, sub in frame[frame["group"] == "random"].groupby("pair_index"):
        if 0 < sub["success"].mean() < 1:
            values.append(roc_auc_score(sub["success"].astype(int), sub["_score"]))
    return np.asarray(values, dtype=float)


def _fmt(values: np.ndarray) -> str:
    if len(values) == 0:
        return ""
    mean, low, high = BT._mean_interval(values)
    return f"{mean:.3f} [{low:.3f}, {high:.3f}]"


def ranking_table(data: pd.DataFrame, weights_path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Within-pair AUC on SBM's random sets for several scores, with 95% intervals over pairs:
        own score            SBM's pooled weights, each pair scored by a model fitted without it
        own score + blocks   the same with the largest block's share as a fifth feature
        degree alone         seed-to-nonseed edges
        matched model        the weights of the round-1 model without communities (ER or Chung-Lu)
        averaged weights     the ER, PAPER and Chung-Lu weights averaged
    Also returns the held-out frames (for success by percentile)."""

    saved = BN.load_weights(weights_path)
    rows, scored = [], {}
    for algorithm, setting in _settings_in(data):
        part = data[(data["algorithm"] == algorithm) & (data["model"] == setting)]
        own = held_out(data, algorithm, setting)
        with_blocks = held_out(data, algorithm, setting, features=(*CORE, "largest_block_share"))
        scored[(algorithm, setting)] = own
        z = part[[f"z_{f}" for f in CORE]].to_numpy(float)
        matched = np.array([saved[(algorithm, SR.base_model(setting))][f] for f in CORE])
        averaged = np.array([np.mean([saved[(algorithm, m)][f] for m in BN.MODELS]) for f in CORE])
        scores = {"own score (held out)": own.loc[part.index, "score"].to_numpy(float),
                  "own score + block share (held out)": with_blocks.loc[part.index, "score"].to_numpy(float),
                  "degree alone": part["z_degree"].to_numpy(float),
                  f"matched model's weights ({MODEL_LABELS[SR.base_model(setting)]})": z @ matched,
                  "ER, PAPER, Chung-Lu weights averaged": z @ averaged}
        for name, values in scores.items():
            aucs = _pair_aucs(part, values)
            rows.append({"algorithm": ALGORITHM_LABELS[algorithm], "setting": SETTING_LABELS[setting],
                         "score": name, "pairs with both outcomes": len(aucs), "AUC": _fmt(aucs),
                         "auc_mean": float(aucs.mean()) if len(aucs) else np.nan})
    return pd.DataFrame(rows).set_index(["algorithm", "setting", "score"]), scored


def reference_auc(algorithms=ALGORITHMS, models=BN.MODELS) -> pd.DataFrame:
    """For comparison: the held-out AUC of each round-1 model's own score on the multi-pair data (ER, PAPER,
    Chung-Lu), computed as in multi_pair_models.ipynb."""

    rows = []
    for algorithm in algorithms:
        for model in models:
            try:
                data = MP.load_sets([algorithm], [model])
            except FileNotFoundError:
                continue
            scored = MP.held_out(data, algorithm, model)
            own = _pair_aucs(scored, scored["score"].to_numpy(float))
            degree = _pair_aucs(scored, scored["z_degree"].to_numpy(float))
            rows.append({"algorithm": ALGORITHM_LABELS[algorithm], "model": MODEL_LABELS[model],
                         "pairs with both outcomes": len(own), "own score (held out)": _fmt(own),
                         "degree alone": _fmt(degree)})
    return pd.DataFrame(rows).set_index(["algorithm", "model"]) if rows else pd.DataFrame()


def success_by_percentile(scored: dict) -> pd.DataFrame:
    """Success of held-out random and tail sets by their score's percentile among the pair's random sets."""

    rows = []
    for (algorithm, setting), frame in scored.items():
        table = MP.success_by_percentile(frame)
        random_rate = float(frame.loc[frame["group"] == "random", "success"].mean())
        for label, row in table.iterrows():
            rows.append({"algorithm": algorithm, "setting": setting, "percentile": str(label),
                         "success": row["success"], "sets": int(row["sets"]), "random": random_rate})
    return pd.DataFrame(rows)


def contrast_table(data: pd.DataFrame) -> pd.DataFrame:
    """Degree-matched contrasts: pairs where only the high-feature set succeeded against only the low one."""

    from scipy.stats import binomtest

    part = data[data["group"] == "contrast"]
    rows = []
    for (algorithm, setting, feature), sub in part.groupby(["algorithm", "model", "contrast_feature"], sort=False):
        wide = sub.pivot_table(index=["pair_index", "contrast_pair_id"], columns="contrast_tail", values="success",
                               aggfunc="first").dropna()
        high = int((wide["high"].astype(bool) & ~wide["low"].astype(bool)).sum())
        low = int((wide["low"].astype(bool) & ~wide["high"].astype(bool)).sum())
        rows.append({"algorithm": ALGORITHM_LABELS[algorithm], "setting": SETTING_LABELS[setting],
                     "feature": FEATURE_LABELS[feature], "contrasts": len(wide), "high wins": high, "low wins": low,
                     "p": binomtest(high, high + low).pvalue if high + low else np.nan})
    return pd.DataFrame(rows).set_index(["algorithm", "setting", "feature"]) if rows else pd.DataFrame()


def block_share_table(data: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
    """Success of the random sets by how concentrated their seeds are (the largest block's share of the seeds), with
    a 95% interval from resampling pairs. Few distinct shares (small budgets) are shown one by one; otherwise they
    are grouped into about n_bins bins of similar size."""

    rows = []
    for algorithm, setting in _settings_in(data):
        part = data[(data["algorithm"] == algorithm) & (data["model"] == setting)]
        random = part[part["group"] == "random"]
        values = random["largest_block_share"].round(6)
        if values.nunique() <= 6:
            bins = values.rank(method="dense").astype(int) - 1
        else:
            bins = pd.Series(pd.qcut(values, n_bins, labels=False, duplicates="drop"), index=random.index)
        for b in sorted(bins.unique()):
            sub = random[bins == b]
            low, high = sub["largest_block_share"].min(), sub["largest_block_share"].max()
            by_pair = sub.groupby("pair_index")["success"].mean()
            mean, ci_low, ci_high = BT._mean_interval(by_pair.to_numpy(float))
            rows.append({"algorithm": ALGORITHM_LABELS[algorithm], "setting": SETTING_LABELS[setting],
                         "largest block's share": f"{low:.2f}" if np.isclose(low, high) else f"{low:.2f} to {high:.2f}",
                         "sets": len(sub), "pairs": len(by_pair),
                         "success (mean over pairs)": f"{mean:.2f} [{ci_low:.2f}, {ci_high:.2f}]"})
    return pd.DataFrame(rows).set_index(["algorithm", "setting", "largest block's share"])


def five_feature_weights(data: pd.DataFrame, n_boot: int = 100) -> pd.DataFrame:
    """Pooled weights per SD with the largest block's share added to the four core features (95% intervals from
    resampling pairs): does concentrating the seeds in one block matter at equal degree, harmonic, pair mass and
    uniqueness?"""

    rows = []
    features = (*CORE, "largest_block_share")
    for algorithm, setting in _settings_in(data):
        table = pooled_weights(data, algorithm, setting, features=features, n_boot=n_boot)
        rows.append({"algorithm": ALGORITHM_LABELS[algorithm], "setting": SETTING_LABELS[setting],
                     **{FEATURE_LABELS[f]: f"{table.loc[f, 'weight']:+.2f} [{table.loc[f, 'low']:+.2f}, "
                                           f"{table.loc[f, 'high']:+.2f}]" for f in features}})
    return pd.DataFrame(rows).set_index(["algorithm", "setting"])


def save_weights(table: pd.DataFrame, path: Path | None = None) -> Path:
    """SBM's pooled weights for best of N, in the format of the multi-pair pooled_weights.json."""

    path = OUTPUT_DIR / "pooled_weights.json" if path is None else path
    payload = {}
    for (algorithm, model), part in table[table["model"].isin(G.SETTINGS)].groupby(["algorithm", "model"], sort=False):
        part = part.set_index("feature")
        payload[f"{algorithm}/{model}"] = {
            "weights_per_sd": {f: float(part.loc[f, "weight"]) for f in part.index},
            "interval_95": {f: [float(part.loc[f, "low"]), float(part.loc[f, "high"])] for f in part.index},
            "feature_columns": {f: MP.core_columns(algorithm)[f] for f in part.index},
            "standardisation": "z-score against the pair's uniform random sets of the same size",
            "budget": int(part["budget"].iloc[0]), "pairs": int(part["pairs"].iloc[0]),
            "graphs": G.describe(algorithm, model)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

# ER and its SBM in blue, Chung-Lu and its DC-SBM in aqua, PAPER in orange (validated colour order); open markers
# for the round-1 models, filled for the SBM settings
MODEL_STYLE = {"sparse_er": ("#2a78d6", "o", "white"), "paper_pa": ("#eb6834", "^", "white"),
               "chung_lu": ("#1baf7a", "s", "white"), "sbm": ("#2a78d6", "o", "#2a78d6"),
               "dcsbm": ("#1baf7a", "s", "#1baf7a"), "sbm_unequal": ("#2a78d6", "D", "#2a78d6"),
               "dcsbm_unequal": ("#1baf7a", "D", "#1baf7a"), "sbm_weak": ("#2a78d6", "v", "#2a78d6"),
               "dcsbm_weak": ("#1baf7a", "v", "#1baf7a")}
# lines per setting: blue for plain SBM, aqua for degree-corrected; variants differ by line style and marker
SETTING_STYLE = {"sbm": ("#2a78d6", "-", "o"), "dcsbm": ("#1baf7a", "-", "s"),
                 "sbm_unequal": ("#2a78d6", "--", "D"), "dcsbm_unequal": ("#1baf7a", "--", "D"),
                 "sbm_weak": ("#2a78d6", "-.", "v"), "dcsbm_weak": ("#1baf7a", "-.", "v")}


def plot_weights(table: pd.DataFrame):
    """Pooled weights per feature, one panel per algorithm: round-1 models open, SBM settings filled."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    algorithms = [a for a in ALGORITHMS if a in set(table["algorithm"])]
    models = [m for m in (*BN.MODELS, *G.SETTINGS) if m in set(table["model"])]
    fig, axes = plt.subplots(1, len(algorithms), figsize=(4.6 * len(algorithms), 4.2), squeeze=False,
                             constrained_layout=True, sharey=True)
    offsets = np.linspace(-0.3, 0.3, len(models))
    labels = {**MODEL_LABELS, **SETTING_LABELS}
    for ax, algorithm in zip(axes[0], algorithms):
        SR._axes(ax)
        ax.axvline(0, color=SR.INK["secondary"], linewidth=1.0, linestyle=":")
        for offset, model in zip(offsets, models):
            sub = table[(table["algorithm"] == algorithm) & (table["model"] == model)].set_index("feature")
            color, marker, face = MODEL_STYLE[model]
            for y, feature in enumerate(CORE):
                if feature not in sub.index:
                    continue
                r = sub.loc[feature]
                ax.errorbar(r["weight"], y + offset, xerr=[[r["weight"] - r["low"]], [r["high"] - r["weight"]]],
                            fmt=marker, color=color, markerfacecolor=face, markersize=6.5, elinewidth=1.3, capsize=0)
        ax.set_yticks(range(len(CORE)))
        ax.set_yticklabels([FEATURE_LABELS[f] for f in CORE], fontsize=9.5)
        ax.set_ylim(len(CORE) - 0.5, -0.5)
        ax.set_title(ALGORITHM_LABELS[algorithm], fontsize=11, color=SR.INK["primary"])
        ax.set_xlabel("weight per SD (log-odds of success)", fontsize=9.5, color=SR.INK["secondary"])
    handles = [Line2D([], [], marker=MODEL_STYLE[m][1], color=MODEL_STYLE[m][0], markerfacecolor=MODEL_STYLE[m][2],
                      linestyle="none", markersize=7, label=labels[m]) for m in models]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(models), frameon=False, fontsize=9.5)
    return fig


def plot_percentile_success(table: pd.DataFrame):
    """Success of held-out sets by score percentile (SBM's own score), one panel per algorithm, one line per setting."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    algorithms = [a for a in ALGORITHMS if a in set(table["algorithm"])]
    settings = [s for s in G.SETTINGS if s in set(table["setting"])]
    fig, axes = plt.subplots(1, len(algorithms), figsize=(4.8 * len(algorithms), 3.9), squeeze=False,
                             constrained_layout=True)
    labels = list(MP.PERCENTILE_LABELS)
    for ax, algorithm in zip(axes[0], algorithms):
        SR._axes(ax)
        for setting in settings:
            sub = table[(table["algorithm"] == algorithm) & (table["setting"] == setting)]
            if sub.empty:
                continue
            sub = sub.set_index("percentile").reindex(labels)
            color, style, marker = SETTING_STYLE[setting]
            shown = (sub["sets"].fillna(0) > 0).to_numpy()
            ax.plot(np.arange(len(labels))[shown], sub.loc[shown, "success"], color=color, linestyle=style,
                    marker=marker, linewidth=2, markersize=5.5)
            ax.axhline(float(sub["random"].dropna().iloc[0]), color=color, linewidth=1.2, linestyle=":")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(ALGORITHM_LABELS[algorithm], fontsize=11, color=SR.INK["primary"])
        ax.set_xlabel("held-out score percentile among the pair's random sets", fontsize=9, color=SR.INK["secondary"])
        ax.set_ylabel("success rate", fontsize=9.5, color=SR.INK["secondary"])
    handles = [Line2D([], [], color=SETTING_STYLE[s][0], linestyle=SETTING_STYLE[s][1], marker=SETTING_STYLE[s][2],
                      linewidth=2, label=SETTING_LABELS[s]) for s in settings]
    handles.append(Line2D([], [], color=SR.INK["secondary"], linestyle=":", label="average random set"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=9.5)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=ALGORITHMS)
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(G.SETTINGS))
    parser.add_argument("--jobs", type=int, default=3, help="tasks run in parallel (default 3)")
    parser.add_argument("--blas-threads", type=int, default=3)
    parser.add_argument("--sets-per-pair", type=int, default=sum(GROUPS.values()),
                        help="seed sets per pair (default 500: 60%% random, 20%% tail, 20%% contrast)")
    parser.add_argument("--pairs", nargs="+", default=[], metavar="ALGORITHM=N",
                        help="pairs per setting, e.g. --pairs goat=10 faq=8 (default percolation=30 goat=20 faq=16)")
    parser.add_argument("--status", action="store_true", help="only show how far the collection is")
    args = parser.parse_args()
    pairs = {}
    for item in args.pairs:
        name, _, value = item.partition("=")
        if name not in ALGORITHMS or not value.isdigit():
            parser.error(f"--pairs takes ALGORITHM=N with ALGORITHM one of {', '.join(ALGORITHMS)}; got {item!r}")
        pairs[name] = int(value)
    groups = MP.groups_for(args.sets_per_pair)
    if args.status:
        print(status(args.algorithms, args.settings, pairs, groups).to_string())
    else:
        run_all(args.algorithms, args.settings, n_pairs=pairs, jobs=args.jobs, blas_threads=args.blas_threads,
                groups=groups)
