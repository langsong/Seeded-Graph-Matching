"""Do round 1's seed rules still beat random on graphs with communities? (plan b, 2 Nov 2026)

Question. Round 1 found rules that pick better seeds than random on ER, PAPER and Chung-Lu graphs (for example,
seeds whose neighbourhoods overlap for percolation, and seeds that tell vertices apart for GOAT). Do the same rules
still help on stochastic block models, where the vertices form communities? Only the general rules are tested here:
the pooled score's weights were fitted on graphs without communities, and SBM gets its own weights from
fixed_pair_seed_set_classification/sbm_collection.py (plan a).

Graphs (sbm_graphs.py). Two kinds of SBM, each matched to a round-1 model so that communities are the only change:
    sbm    plain SBM, "ER plus communities": 4 equal blocks of 150 vertices, 80% of the edges inside blocks
    dcsbm  degree-corrected SBM, "Chung-Lu plus communities": the same blocks on Chung-Lu weights (Pareto shape 2)
Mean degree, rho and GOAT's regulariser are the matched model's: percolation mean degree 12 and rho 0.70; GOAT and
FAQ mean degree 16 and rho 0.50 (SBM, as ER) or 0.45 (DC-SBM, as Chung-Lu); GOAT regulariser 100 (SBM) or 300
(DC-SBM). Variants with unequal blocks (300/150/100/50) or weak communities (60% inside) are defined in
sbm_graphs.SETTINGS and can be run later by name.

Budgets. Communities may change how many seeds a random set needs, so every algorithm and setting is calibrated
first, as in score_transfer's part C: 3 random sets on each of 10 separate calibration pairs (30 sets per budget),
one run each, along a budget ladder that starts at the matched model's training budget. k50 is the budget where
random sets succeed closest to half the time. The budget grid is k50 times
    percolation  0.4, 0.55, 0.7, 0.85, 1.0, 1.25
    GOAT         0.4, 0.55, 0.65, 0.75, 0.9, 1.1
    FAQ          0.55, 0.65, 0.75, 0.85, 0.95, 1.1
(rounded, repeats dropped, never below 2 seeds; extended upward to at least 5 budgets when k50 is small): about
where round 1's grids sat relative to the training budgets. Best of 1,000 needs 1,000 different seed sets, and with
one seed there are only as many as vertices, hence the floor of 2 (added 3 Nov, after GOAT on degree-corrected SBM
calibrated to k50 = 2).

Rules, on 20 fresh pairs per setting (none used for calibration):
    round 1's rules       random, top degree, clustered (pair mass), spread (coverage), distinguishing, plus top
                          harmonic (percolation) or hub-free distinguishing (GOAT): round 1's code and settings
    community_top_degree  top degree spread over the blocks in proportion to their sizes (round robin for equal
                          blocks), from the true block labels: does spreading the seeds over communities help?
    best_1000             the highest pooled score among 1,000 random sets, with the ER, PAPER and Chung-Lu weights
                          averaged. Not a fair test of the weights, which never saw communities; shown for reference.
    degree_1000           the most seed-to-nonseed edges among the same 1,000 sets (the control for best_1000)
The ordering rules give nested sets across budgets, as in round 1.

Direct test of SBM's own weights (run_sbm_weights, added 3 Nov). Once sbm_models.ipynb has fitted and saved SBM's own
pooled weights (from separate collection pairs), best of 100 and best of 1,000 are run with those weights on this
experiment's 20 test pairs per setting and budget grid. They choose from the same 1,000 random draws as best_1000
above (checked) and meet the same solver seeds, so the two differ only in the weights. A set already run above at
the same budget is not run again: its stored outcome is the same run. Every rule's set is also scored with SBM's
weights, so the score can pick among round 1's rules afterwards. About 25 minutes with 3 jobs; results go to
<algorithm>/<setting>/sbm_weights/.

Runs. Round 1's protocol: percolation and GOAT one run, FAQ up to 3 restarts (stopping at the first success).
Solver seeds depend on the pair, budget and restart only, so every rule meets the same solver randomness. Success:
percolation accuracy >= 0.5; GOAT and FAQ unseeded accuracy >= 0.9. Each row also records how many blocks the seeds
cover and the largest block's share of the seeds, and where the set falls on the averaged score.

Run from the repository root (ds environment); about 1.5 hours with 3 jobs for the two main settings:

    python -m selection_rule_comparison.sbm_rules
    python -m selection_rule_comparison.sbm_rules --algorithms goat --settings sbm
    python -m selection_rule_comparison.sbm_rules --status
    python -m selection_rule_comparison.sbm_rules --sbm-weights      # the direct test, after sbm_models.ipynb

or call run_all() from sbm_rules.ipynb. Calibrations run first and each one queues its 20 pairs as soon as it
finishes. Every calibration step and every pair is saved when it finishes, so rerunning resumes; Ctrl-C (or the
notebook's interrupt button) stops the workers too. Results go to data/sbm_rules/<algorithm>/<setting>/, or to the
folder in the environment variable SBM_RULES_OUTPUT_DIR.
"""

from __future__ import annotations

import argparse
import datetime
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

from experiment_common.provenance import collect_environment, thread_limit, write_json
from fixed_pair_seed_set_classification import multi_pair_collection as MP

from . import best_of_n as BN
from . import beyond_top_degree as BT
from .. import run_comparison as rc
from . import sbm_graphs as G
from . import score_transfer as ST
from .. import settings as S
from ..strategies import STRATEGY_CODES, GraphA, ordering, seed_set_features

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]  # selection_rule_comparison/ (this file is in archive/)
DATA_DIR = EXPERIMENT_DIR / "data"
ROUND1_ROOT = DATA_DIR / "round1" / "main"
OUTPUT_DIR = Path(os.environ.get("SBM_RULES_OUTPUT_DIR") or DATA_DIR / "sbm_rules")

ALGORITHMS = ("percolation", "goat", "faq")
SETTINGS = tuple(G.DEFAULT_SETTINGS)
N_PAIRS = 20
CALIBRATION_PAIRS = tuple(range(10))
CALIBRATION_SETS = 3                     # random sets per pair at each budget tried
TARGET_RATE = 0.5
LADDER = ST.LADDER
BUDGET_FACTORS = {"percolation": (0.4, 0.55, 0.7, 0.85, 1.0, 1.25),
                  "goat": (0.4, 0.55, 0.65, 0.75, 0.9, 1.1),
                  "faq": (0.55, 0.65, 0.75, 0.85, 0.95, 1.1)}
MIN_BUDGETS = 5
MIN_BUDGET = 2                           # best of 1,000 needs 1,000 different sets: one seed gives only n of them
NEW_RULES = ("community_top_degree", "best_1000", "degree_1000")
SCREENED = ("best_1000", "degree_1000")
COMMUNITY_CODE = 50                      # community_top_degree's random stream (round 1's rules use codes 0-6)

SBM_WEIGHTS_PATH = (Path(os.environ.get("SBM_MULTI_PAIR_OUTPUT_DIR") or EXPERIMENT_DIR.parent
                         / "fixed_pair_seed_set_classification" / "data" / "sbm_multi_pair") / "pooled_weights.json")
SBM_ARM = "sbm_weights"                  # subfolder for the direct test of SBM's own weights
SBM_RULES = tuple(f"sbm_best_{n}" for n in BN.SCREEN_SIZES)          # best of 100 and best of 1,000
SELECTION_RULES = ("random", "top_degree", "degree_1000", "best_1000", *SBM_RULES)

CALIBRATION_SEED_BASE = 20261109
DRAW_SEED_BASE = BN.DRAW_SEED_BASE
ALGORITHM_CODES = BN.ALGORITHM_CODES

SECONDS_PER_RUN = ST.SECONDS_PER_RUN     # per run and job with 3 jobs, n = 600
FAQ_RUNS_PER_SET = 2.0                   # restarts used on average (round 1)
CACHED_SHARE = 0.8                       # sets that are new (the rest repeat another rule's set at that budget)

ALGORITHM_LABELS = BN.ALGORITHM_LABELS
SETTING_LABELS = G.SETTING_LABELS
SHORT_LABELS = {"sbm": "SBM", "dcsbm": "DC-SBM", "sbm_unequal": "SBM unequal", "dcsbm_unequal": "DC-SBM unequal",
                "sbm_weak": "SBM weak", "dcsbm_weak": "DC-SBM weak"}
MODEL_LABELS = BN.MODEL_LABELS
RULE_LABELS = {**BN.RULE_LABELS, "community_top_degree": "top degree by block",
               "best_1000": "best of 1,000 (averaged weights)", "degree_1000": "best of 1,000 by degree",
               **{f"sbm_best_{n}": f"best of {n:,} (SBM weights)" for n in BN.SCREEN_SIZES}}


def _now() -> str:
    return datetime.datetime.now().astimezone().strftime("%a %H:%M")


def _flag(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def rule_label(rule: str) -> str:
    return RULE_LABELS.get(rule, rule.replace("_", " "))


def rules_for(algorithm: str) -> tuple[str, ...]:
    return (*S.STRATEGIES["main"][algorithm], *NEW_RULES)


def base_model(setting: str) -> str:
    """The round-1 model this setting adds communities to (ER for plain SBM, Chung-Lu for degree-corrected)."""

    return G.BASE_MODEL[G.kind(setting)]


def averaged_weights(path: Path | None = None) -> dict:
    """The pooled weights per SD of ER, PAPER and Chung-Lu, averaged per algorithm."""

    weights = BN.load_weights(path)
    return {a: {f: float(np.mean([weights[(a, m)][f] for m in BN.MODELS])) for f in BN.CORE} for a in ALGORITHMS}


def budget_grid(algorithm: str, k50: int) -> list[int]:
    """k50 times the algorithm's factors, rounded half up, repeats dropped, at least MIN_BUDGET seeds. When k50 is so
    small that fewer than MIN_BUDGETS remain, the grid is extended upward one seed at a time."""

    grid = sorted({max(MIN_BUDGET, int(np.floor(f * k50 + 0.5))) for f in BUDGET_FACTORS[algorithm]})
    while len(grid) < MIN_BUDGETS:
        grid.append(grid[-1] + 1)
    return grid


# --------------------------------------------------------------------------
# The community rule
# --------------------------------------------------------------------------

def community_ordering(graph: GraphA, blocks: np.ndarray, k_max: int, rng: np.random.Generator) -> np.ndarray:
    """Top degree spread over the blocks in proportion to their sizes (round robin for equal blocks).

    Seed j goes to the block with the largest size / (seeds it already has + 1) (D'Hondt's rule), and within that
    block to its highest-degree unused vertex. Ties between blocks go to the block whose next vertex comes first in
    the degree order (higher degree; equal degrees in random order).
    """

    order = graph.degree_order(rng)
    position = np.empty(graph.n, dtype=np.int64)
    position[order] = np.arange(graph.n)
    labels = [int(b) for b in np.unique(blocks)]
    queues = {b: order[blocks[order] == b] for b in labels}
    taken = dict.fromkeys(labels, 0)
    chosen = []
    for _ in range(k_max):
        open_blocks = [b for b in labels if taken[b] < len(queues[b])]
        block = max(open_blocks, key=lambda b: (len(queues[b]) / (taken[b] + 1), -position[queues[b][taken[b]]]))
        chosen.append(int(queues[block][taken[block]]))
        taken[block] += 1
    return np.asarray(chosen, dtype=int)


# --------------------------------------------------------------------------
# Budget calibration
# --------------------------------------------------------------------------

def setting_dir(algorithm: str, setting: str) -> Path:
    return OUTPUT_DIR / algorithm / setting


def calibration_path(algorithm: str, setting: str) -> Path:
    return setting_dir(algorithm, setting) / "calibration.json"


def load_calibration(algorithm: str, setting: str) -> dict | None:
    path = calibration_path(algorithm, setting)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _refresh_budgets(algorithm: str, setting: str) -> None:
    """Bring a saved budget grid up to date with budget_grid() (for example after the 2-seed floor was added), as
    long as no pair of that setting has been run with the old grid."""

    saved = load_calibration(algorithm, setting)
    if not saved or not saved["k50"]:
        return
    grid = budget_grid(algorithm, int(saved["k50"]))
    if saved["budgets"] == grid or any(setting_dir(algorithm, setting).glob("pair_[0-9][0-9][0-9].csv")):
        return
    saved.update({"previous_budgets": saved["budgets"], "budgets": grid,
                  "budgets_revised_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")})
    write_json(calibration_path(algorithm, setting), saved)
    print(f"{_now()}  {ALGORITHM_LABELS[algorithm]} {SETTING_LABELS[setting]}: budget grid updated to "
          f"{', '.join(map(str, grid))} (was {', '.join(map(str, saved['previous_budgets']))}).", flush=True)


def calibration_job(algorithm: str, setting: str, blas_threads: int = 3) -> str:
    """Find k50, the budget where random sets succeed about half the time, on the calibration pairs."""

    path = calibration_path(algorithm, setting)
    name = f"{ALGORITHM_LABELS[algorithm]} {SETTING_LABELS[setting]} · budget calibration"
    if path.exists():
        return f"{name}: already done"
    started = time.perf_counter()
    base = base_model(setting)
    start = ST.training_budgets()[(algorithm, base)]
    folder = setting_dir(algorithm, setting) / "calibration"
    rates, runs = {}, [0]
    with thread_limit(blas_threads):
        pairs = [G.load_pair(algorithm, setting, "calibration", i) for i in CALIBRATION_PAIRS]

        def rate(budget: int) -> float:
            if budget in rates:
                return rates[budget]
            file = folder / f"k{budget:02d}.csv"
            if file.exists():
                table = pd.read_csv(file)
            else:
                rows = []
                for pair_index, (seed, A, B, truth, blocks) in zip(CALIBRATION_PAIRS, pairs):
                    rng = np.random.default_rng([CALIBRATION_SEED_BASE, seed, budget, ALGORITHM_CODES[algorithm]])
                    for j, seeds in enumerate(MP.draw_pool(rng, A.shape[0], budget, CALIBRATION_SETS)):
                        out = MP.evaluate(algorithm, base, (A, B, truth, seed), seeds,
                                          rc._seed(CALIBRATION_SEED_BASE, seed, budget, j))
                        runs[0] += 1
                        rows.append({"algorithm": algorithm, "setting": setting, "pair_index": pair_index,
                                     "pair_seed": seed, "budget": budget, "set": j,
                                     "seed_vertices": json.dumps([int(v) for v in seeds]),
                                     "blocks_covered": G.blocks_covered(blocks, seeds),
                                     "largest_block_share": G.largest_block_share(blocks, seeds),
                                     "solver_seed": out["solver_seed"], "success": bool(out["success"]),
                                     "accuracy": out["final_unseeded_accuracy"], "n_iter": out["n_iter"],
                                     "runtime_seconds": out["runtime_seconds"]})
                _write(file, rows)
                table = pd.DataFrame(rows)
            rates[budget] = float(_flag(table["success"]).mean())
            return rates[budget]

        k50, status = ST.calibrate(LADDER[algorithm], start, rate, TARGET_RATE)
    write_json(path, {
        "algorithm": algorithm, "setting": setting, "graphs": G.describe(algorithm, setting),
        "base_model": base, "start_budget": start, "ladder": list(LADDER[algorithm]),
        "sets_per_budget": CALIBRATION_SETS * len(CALIBRATION_PAIRS), "rates": {str(b): rates[b] for b in sorted(rates)},
        "k50": k50, "status": status, "budgets": budget_grid(algorithm, k50) if k50 else [],
        "pairs": [{"pair_index": i, "pair_seed": seed, "hash_a": rc._hash(A), "hash_b": rc._hash(B)}
                  for i, (seed, A, B, _truth, _blocks) in zip(CALIBRATION_PAIRS, pairs)],
        "finished_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")})
    shown = ", ".join(f"{b}: {rates[b]:.2f}" for b in sorted(rates))
    outcome = f"k50 = {k50}, budgets {', '.join(map(str, budget_grid(algorithm, k50)))}" if k50 else "skipped"
    return (f"{name}: {outcome} ({status}; random success by budget {shown}); {runs[0]} runs in "
            f"{(time.perf_counter() - started) / 60:.1f} min")


# --------------------------------------------------------------------------
# One pair
# --------------------------------------------------------------------------

def pair_path(algorithm: str, setting: str, pair_index: int) -> Path:
    return setting_dir(algorithm, setting) / f"pair_{pair_index:03d}.csv"


def pair_job(algorithm: str, setting: str, pair_index: int, budgets, weights: dict, blas_threads: int = 3) -> str:
    """Every rule on one pair at every budget of the grid."""

    path = pair_path(algorithm, setting, pair_index)
    name = f"{ALGORITHM_LABELS[algorithm]} {SETTING_LABELS[setting]} · pair {pair_index}"
    if path.exists():
        return f"{name}: already done"
    started = time.perf_counter()
    base = base_model(setting)
    budgets = sorted(int(b) for b in budgets)
    with thread_limit(blas_threads):
        seed, A, B, truth, blocks = G.load_pair(algorithm, setting, "rules", pair_index)
        graph = GraphA(A)
        summary = G.pair_summary(A, B, truth, blocks)
        summary.pop("blocks_a")
        orders = {}
        for rule in S.STRATEGIES["main"][algorithm]:
            rng = np.random.default_rng([S.STRATEGY_SEED_BASE, seed, STRATEGY_CODES[rule]])
            orders[rule] = ordering(rule, graph, budgets[-1], rng, pool_factor=S.POOL_FACTOR, hub_count=S.HUB_COUNT)
        orders["community_top_degree"] = community_ordering(
            graph, blocks, budgets[-1], np.random.default_rng([S.STRATEGY_SEED_BASE, seed, COMMUNITY_CODE]))
        pair = (A, B, truth, seed)
        restarts = S.FAQ_MAX_RESTARTS if algorithm == "faq" else 1
        rules = rules_for(algorithm)
        rows, cache = [], {}
        for budget in budgets:
            rng = np.random.default_rng([DRAW_SEED_BASE, seed, budget, ALGORITHM_CODES[algorithm]])
            draws, score, chosen, _reference = BN.screen(graph, algorithm, budget, weights, rng)
            sets = {rule: np.sort(order[:budget]) for rule, order in orders.items()}
            sets.update({rule: np.sort(draws[chosen[rule]]) for rule in SCREENED})
            placed = score(np.array([sets[rule] for rule in rules]))   # where each set falls on the averaged score
            for rule, info in zip(rules, placed.to_dict("records")):
                key = tuple(int(v) for v in sets[rule])
                described = {"blocks_covered": G.blocks_covered(blocks, key),
                             "largest_block_share": G.largest_block_share(blocks, key),
                             **info, **seed_set_features(graph, np.array(key))}
                for restart in range(restarts):
                    cached = (key, restart) in cache
                    if not cached:
                        cache[(key, restart)] = rc.evaluate(algorithm, base, pair, np.array(key), budget, restart)
                    out = cache[(key, restart)]
                    rows.append({
                        "algorithm": algorithm, "setting": setting, "kind": G.kind(setting), "base_model": base,
                        "regularizer": G.regularizer(algorithm, setting) or 0, "pair_index": pair_index,
                        "pair_seed": seed, "rule": rule, "budget": budget, "restart": restart,
                        "solver_seed": out["solver_seed"], "success": bool(out["success"]),
                        "success_090": bool(out["success_090"]), "accuracy": out["final_unseeded_accuracy"],
                        "objective": out["objective"], "n_iter": out["n_iter"], "converged": out["converged"],
                        "runtime_seconds": out["runtime_seconds"], "cached": cached,
                        "seed_vertices": json.dumps(list(key)), **described, **summary})
                    if out["success"]:
                        break
    _write(path, rows)
    return f"{name}: {sum(not r['cached'] for r in rows)} runs in {(time.perf_counter() - started) / 60:.1f} min"


# --------------------------------------------------------------------------
# Running everything
# --------------------------------------------------------------------------

def _snapshot(algorithm: str, setting: str, weights: dict) -> dict:
    """The settings that results in one folder depend on (more pairs may be added later)."""

    return json.loads(json.dumps({
        "graphs": G.describe(algorithm, setting), "setting_spec": G.SETTINGS[setting], "rules": list(rules_for(algorithm)),
        "budget_factors": list(BUDGET_FACTORS[algorithm]), "ladder": list(LADDER[algorithm]),
        "calibration": {"pairs": list(CALIBRATION_PAIRS), "sets_per_pair": CALIBRATION_SETS,
                        "target_rate": TARGET_RATE, "start": "the matched round-1 model's training budget"},
        "averaged_weights_per_sd": weights[algorithm], "draws": BN.N_DRAWS,
        "solver": (S.PERCOLATION if algorithm == "percolation" else
                   {**S.SGM, "transport": algorithm == "goat",
                    "transport_regularizer": G.regularizer(algorithm, setting),
                    "max_restarts": S.FAQ_MAX_RESTARTS if algorithm == "faq" else 1}),
        "success_threshold": S.SUCCESS_THRESHOLD[algorithm], "pool_factor": S.POOL_FACTOR, "hub_count": S.HUB_COUNT,
        "seed_bases": {"pairs": G.PAIR_SEED_BASE, "strategies": S.STRATEGY_SEED_BASE, "solver": S.SOLVER_SEED_BASE,
                       "calibration": CALIBRATION_SEED_BASE, "draws": DRAW_SEED_BASE},
    }))


def _check_and_save_settings(algorithm: str, setting: str, weights: dict, blas_threads: int) -> None:
    path = setting_dir(algorithm, setting) / "settings.json"
    snapshot = _snapshot(algorithm, setting, weights)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        changed = sorted(k for k in snapshot if saved.get(k) != snapshot[k])
        if changed:
            raise RuntimeError(f"{path.parent} holds results made with different settings ({', '.join(changed)}). "
                               "Set SBM_RULES_OUTPUT_DIR to a new folder, or move the old one away.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {**snapshot, "environment": collect_environment(blas_threads),
                      "written_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")})


def _pair_tasks(algorithm: str, setting: str, n_pairs: int) -> list[tuple]:
    saved = load_calibration(algorithm, setting)
    if not saved or not saved["k50"]:
        return []
    return [("pair", algorithm, setting, i) for i in range(n_pairs) if not pair_path(algorithm, setting, i).exists()]


def planned_tasks(algorithms=ALGORITHMS, settings=SETTINGS, n_pairs: int = N_PAIRS) -> list[tuple]:
    """Tasks not done yet: calibrations first, then pairs by pair index across settings."""

    calibrations, pairs = [], []
    for algorithm in algorithms:
        for setting in settings:
            if load_calibration(algorithm, setting) is None:
                calibrations.append(("calibration", algorithm, setting, None))
            else:
                pairs += _pair_tasks(algorithm, setting, n_pairs)
    pairs.sort(key=lambda t: (t[3], ALGORITHMS.index(t[1]), list(G.SETTINGS).index(t[2])))
    return calibrations + pairs


def _task_seconds(task: tuple, n_pairs: int = N_PAIRS) -> float:
    kind, algorithm, setting, _ = task
    per_run = SECONDS_PER_RUN[algorithm] * (FAQ_RUNS_PER_SET if algorithm == "faq" else 1)
    if kind == "calibration":
        later = n_pairs * 6 * len(rules_for(algorithm)) * CACHED_SHARE * per_run   # its pairs, still to come
        return ST.CALIBRATION_STEPS * CALIBRATION_SETS * len(CALIBRATION_PAIRS) * SECONDS_PER_RUN[algorithm] + later
    saved = load_calibration(algorithm, setting) or {}
    budgets = len(saved.get("budgets") or [0] * 6)
    return budgets * len(rules_for(algorithm)) * CACHED_SHARE * per_run


def estimate_minutes(tasks, jobs: int = 3, n_pairs: int = N_PAIRS) -> float:
    return sum(_task_seconds(t, n_pairs) for t in tasks) / max(jobs, 1) / 60


def _execute(task: tuple, weights: dict, blas_threads: int) -> str:
    kind, algorithm, setting, pair_index = task
    if kind == "calibration":
        return calibration_job(algorithm, setting, blas_threads)
    budgets = load_calibration(algorithm, setting)["budgets"]
    return pair_job(algorithm, setting, pair_index, budgets, weights[algorithm], blas_threads)


def _describe(task: tuple) -> str:
    kind, algorithm, setting, pair_index = task
    where = f"{ALGORITHM_LABELS[algorithm]} {SETTING_LABELS[setting]}"
    return f"{where} · budget calibration" if kind == "calibration" else f"{where} · pair {pair_index}"


def _ignore_interrupts() -> None:
    """Workers leave Ctrl-C to the main process, which stops them."""

    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_tasks(tasks: list, execute, extra: tuple, describe, seconds, follow_ups, jobs: int) -> list[str]:
    """Run execute(task, *extra) for every task, `jobs` at a time. Tasks returned by follow_ups(task) once a task has
    finished are queued as well. Returns the tasks that failed (described). Ctrl-C stops the workers too, then
    re-raises KeyboardInterrupt."""

    failures, count, total = [], [0], [len(tasks)]

    def report(task, message, error, remaining):
        count[0] += 1
        n = f"[{count[0]}/{total[0]}]"
        if error is None:
            left = sum(seconds(t) for t in remaining) / max(jobs, 1) / 60
            print(f"{_now()}  {n} {message}; about {left:.0f} min left", flush=True)
            return
        try:
            failures.append(describe(task))
        except Exception:  # noqa: BLE001 - a label must never hide the real error
            failures.append(str(task))
        print(f"{_now()}  {n} {failures[-1]} FAILED:", flush=True)
        traceback.print_exception(type(error), error, error.__traceback__)

    def more(task):
        new = follow_ups(task)
        total[0] += len(new)
        return new

    keep_awake = None
    if sys.platform == "darwin":
        import shutil
        import subprocess
        if shutil.which("caffeinate"):
            keep_awake = subprocess.Popen(["caffeinate", "-i"])
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
                    message = execute(task, *extra)
                except Exception as error:  # noqa: BLE001 - report every failure, keep going
                    report(task, None, error, [])
                    continue
                queue += more(task)
                report(task, message, None, queue)
        else:
            import multiprocessing
            from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
            pool = ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn"),
                                       initializer=_ignore_interrupts)
            futures = {pool.submit(execute, task, *extra): task for task in tasks}
            while futures:
                done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
                for future in done:
                    task = futures.pop(future)
                    try:
                        message = future.result()
                    except Exception as error:  # noqa: BLE001 - report every failure, keep going
                        report(task, None, error, [])
                        continue
                    for new in more(task):
                        futures[pool.submit(execute, new, *extra)] = new
                    report(task, message, None, list(futures.values()))
            pool.shutdown(wait=True)
    except KeyboardInterrupt:
        # Stop the workers too: the notebook's interrupt button only reaches this process.
        stop_workers()
        print(f"\n{_now()}  Stopped. Finished tasks are kept; rerun to resume.", flush=True)
        raise
    except BaseException:
        stop_workers()     # no workers left running in the background
        raise
    finally:
        if keep_awake is not None:
            keep_awake.terminate()
    return failures


def run_all(algorithms=ALGORITHMS, settings=SETTINGS, jobs: int = 3, blas_threads: int = 3, n_pairs: int = N_PAIRS,
            weights_path: Path | None = None) -> None:
    """Calibrate the budgets, then run every rule on n_pairs pairs per setting, `jobs` tasks at a time."""

    algorithms = [a for a in ALGORITHMS if a in set(algorithms)]
    unknown = sorted(set(settings) - set(G.SETTINGS))
    if unknown:
        raise ValueError(f"Unknown settings {unknown}; choose from {list(G.SETTINGS)}.")
    settings = [s for s in G.SETTINGS if s in set(settings)]
    weights = averaged_weights(weights_path)
    for algorithm in algorithms:
        for setting in settings:
            _check_and_save_settings(algorithm, setting, weights, blas_threads)
            _refresh_budgets(algorithm, setting)
    tasks = planned_tasks(algorithms, settings, n_pairs)
    if not tasks:
        print(f"Everything is done. Results are in {OUTPUT_DIR}")
        return
    started = time.perf_counter()
    print(f"{_now()}  {len(tasks)} tasks, {jobs} at a time: about {estimate_minutes(tasks, jobs, n_pairs):.0f} min "
          "(budgets are calibrated first, so this is a guess). Each line below is one finished task.", flush=True)

    def follow_ups(task):
        return _pair_tasks(task[1], task[2], n_pairs) if task[0] == "calibration" else []

    try:
        failures = run_tasks(tasks, _execute, (weights, blas_threads), _describe,
                             lambda task: _task_seconds(task, n_pairs), follow_ups, jobs)
    except KeyboardInterrupt:
        return
    if failures:
        raise RuntimeError(f"Failed: {'; '.join(failures)}. Rerun to retry them (finished tasks are kept).")
    print(f"{_now()}  Done in {(time.perf_counter() - started) / 60:.0f} min. Results are in {OUTPUT_DIR}", flush=True)


def status(algorithms=ALGORITHMS, settings=SETTINGS, n_pairs: int = N_PAIRS) -> pd.DataFrame:
    rows = []
    for algorithm in algorithms:
        for setting in settings:
            saved = load_calibration(algorithm, setting)
            if saved is None:
                budgets, planned = "not calibrated yet", n_pairs
            elif not saved["k50"]:
                budgets, planned = f"skipped ({saved['status']})", 0
            else:
                budgets, planned = f"k50 = {saved['k50']}: {', '.join(map(str, saved['budgets']))}", n_pairs
            rows.append({"algorithm": ALGORITHM_LABELS[algorithm], "setting": SETTING_LABELS[setting],
                         "budgets": budgets, "pairs planned": planned,
                         "pairs done": sum(pair_path(algorithm, setting, i).exists() for i in range(n_pairs))})
    return pd.DataFrame(rows).set_index(["algorithm", "setting"])


# --------------------------------------------------------------------------
# Direct test of SBM's own weights (after sbm_collection.py and sbm_models.ipynb)
# --------------------------------------------------------------------------

def load_sbm_weights(path: Path | None = None) -> dict:
    """SBM's own pooled weights per SD, {(algorithm, setting): weights}, as saved by sbm_models.ipynb (section 6)."""

    path = Path(path or SBM_WEIGHTS_PATH)
    if not path.exists():
        raise FileNotFoundError(f"No SBM weights at {path}. Fit and save them first (sbm_models.ipynb, section 6).")
    return BN.load_weights(path)


def _sbm_fit_info(path: Path | None = None) -> dict:
    saved = json.loads(Path(path or SBM_WEIGHTS_PATH).read_text(encoding="utf-8"))
    return {tuple(key.split("/")): {"budget": entry.get("budget"), "pairs": entry.get("pairs")}
            for key, entry in saved.items()}


def sbm_arm_dir(algorithm: str, setting: str) -> Path:
    return setting_dir(algorithm, setting) / SBM_ARM


def sbm_arm_path(algorithm: str, setting: str, pair_index: int) -> Path:
    return sbm_arm_dir(algorithm, setting) / f"pair_{pair_index:03d}.csv"


def sbm_scores_path(algorithm: str, setting: str, pair_index: int) -> Path:
    return sbm_arm_dir(algorithm, setting) / f"pair_{pair_index:03d}_scores.csv"


OUTCOME_FIELDS = ("solver_seed", "success", "success_090", "accuracy", "objective", "n_iter", "converged",
                  "runtime_seconds")


def sbm_weights_job(algorithm: str, setting: str, pair_index: int, sbm_weights: dict, averaged: dict,
                    blas_threads: int = 3) -> str:
    """Best of 100 and best of 1,000 with SBM's own weights on one test pair, at every budget of the grid.

    The 1,000 random draws are the ones the averaged-weight best of 1,000 chose from (checked against the stored
    set), and the solver seeds are the same, so the two differ only in the weights. A set already run at the same
    budget (by any rule) is not run again: its stored outcome is the same run. Every rule's set is also scored with
    SBM's weights and saved, so the score can pick among round 1's rules afterwards.
    """

    path = sbm_arm_path(algorithm, setting, pair_index)
    name = f"{ALGORITHM_LABELS[algorithm]} {SETTING_LABELS[setting]} · pair {pair_index} · SBM weights"
    if path.exists():
        return f"{name}: already done"
    stored_path = pair_path(algorithm, setting, pair_index)
    if not stored_path.exists():
        raise RuntimeError(f"{name}: run the rules experiment on this pair first (run_all).")
    started = time.perf_counter()
    stored = pd.read_csv(stored_path)
    stored["success"] = _flag(stored["success"])
    stored["success_090"] = _flag(stored["success_090"])
    base = base_model(setting)
    budgets = sorted(int(b) for b in stored["budget"].unique())
    restarts = S.FAQ_MAX_RESTARTS if algorithm == "faq" else 1
    with thread_limit(blas_threads):
        seed, A, B, truth, blocks = G.load_pair(algorithm, setting, "rules", pair_index)
        first_row = stored.iloc[0]
        if (seed, rc._hash(A), rc._hash(B)) != (int(first_row["pair_seed"]), first_row["hash_a"], first_row["hash_b"]):
            raise RuntimeError(f"{name}: the pair differs from the one the rules experiment ran.")
        graph = GraphA(A)
        summary = G.pair_summary(A, B, truth, blocks)
        summary.pop("blocks_a")
        pair = (A, B, truth, seed)
        done = {}       # (set, budget) -> {restart: outcome}, from the rules experiment and from this job
        for row in stored.to_dict("records"):
            done.setdefault((row["seed_vertices"], int(row["budget"])), {})[int(row["restart"])] = (
                {f: row[f] for f in OUTCOME_FIELDS}, row["rule"])
        rows, score_rows = [], []
        for budget in budgets:
            first = stored[(stored["budget"] == budget) & (stored["restart"] == 0)]
            rule_sets = dict(zip(first["rule"], first["seed_vertices"]))
            draw_seed = [DRAW_SEED_BASE, seed, budget, ALGORITHM_CODES[algorithm]]
            draws, average_score, average_chosen, _ = BN.screen(graph, algorithm, budget, averaged,
                                                                np.random.default_rng(draw_seed))
            if json.dumps([int(v) for v in np.sort(draws[average_chosen["best_1000"]])]) != rule_sets.get("best_1000"):
                raise RuntimeError(f"{name}: k = {budget}: the 1,000 draws differ from the rules experiment's.")
            _, sbm_score, sbm_chosen, _ = BN.screen(graph, algorithm, budget, sbm_weights,
                                                    np.random.default_rng(draw_seed))
            new = {f"sbm_best_{n}": np.sort(draws[sbm_chosen[f"best_{n}"]]) for n in BN.SCREEN_SIZES}
            # every set at this budget on SBM's score: round 1's rules, the other rules and the new ones
            named = [(rule, np.array(json.loads(seeds), dtype=np.int64)) for rule, seeds in rule_sets.items()]
            named += list(new.items())
            on_sbm = sbm_score(np.array([seeds for _, seeds in named])).to_dict("records")
            for (rule, seeds), info in zip(named, on_sbm):
                score_rows.append({"algorithm": algorithm, "setting": setting, "pair_index": pair_index,
                                   "pair_seed": seed, "budget": budget, "rule": rule,
                                   "seed_vertices": json.dumps([int(v) for v in seeds]),
                                   **{f"sbm_{key}": value for key, value in info.items()}})
            sbm_info = dict(zip([rule for rule, _ in named], on_sbm))
            on_average = average_score(np.array(list(new.values()))).to_dict("records")
            for (rule, seeds), info in zip(new.items(), on_average):
                key = json.dumps([int(v) for v in seeds])
                described = {"blocks_covered": G.blocks_covered(blocks, seeds),
                             "largest_block_share": G.largest_block_share(blocks, seeds), **info,
                             "sbm_score": sbm_info[rule]["score"],
                             "sbm_score_percentile": sbm_info[rule]["score_percentile"],
                             **seed_set_features(graph, seeds)}
                for restart in range(restarts):
                    previous = done.get((key, budget), {}).get(restart)
                    if previous is None:
                        out = rc.evaluate(algorithm, base, pair, seeds, budget, restart)
                        outcome = {"solver_seed": out["solver_seed"], "success": bool(out["success"]),
                                   "success_090": bool(out["success_090"]), "accuracy": out["final_unseeded_accuracy"],
                                   "objective": out["objective"], "n_iter": out["n_iter"],
                                   "converged": out["converged"], "runtime_seconds": out["runtime_seconds"]}
                        done.setdefault((key, budget), {})[restart] = (outcome, rule)
                        cached, same_as = False, ""
                    else:
                        (outcome, same_as), cached = previous, True
                    rows.append({
                        "algorithm": algorithm, "setting": setting, "kind": G.kind(setting), "base_model": base,
                        "regularizer": G.regularizer(algorithm, setting) or 0, "pair_index": pair_index,
                        "pair_seed": seed, "rule": rule, "budget": budget, "restart": restart, **outcome,
                        "cached": cached, "same_set_as": same_as, "seed_vertices": key, **described, **summary})
                    if bool(outcome["success"]):
                        break
    _write(sbm_scores_path(algorithm, setting, pair_index), score_rows)
    _write(path, rows)
    return f"{name}: {sum(not r['cached'] for r in rows)} runs in {(time.perf_counter() - started) / 60:.1f} min"


def _sbm_snapshot(algorithm: str, setting: str, sbm_weights: dict, averaged: dict, fit: dict,
                  weights_path: Path) -> dict:
    return json.loads(json.dumps({
        "sbm_weights_path": str(weights_path), "sbm_weights_per_sd": sbm_weights[(algorithm, setting)],
        "sbm_weights_fitted_on": fit.get((algorithm, setting), {}), "averaged_weights_per_sd": averaged[algorithm],
        "screen_sizes": list(BN.SCREEN_SIZES), "draws": BN.N_DRAWS, "draw_seed_base": DRAW_SEED_BASE,
        "pairs": "the rules experiment's test pairs and budget grid; its solver seeds"}))


def _check_and_save_sbm_settings(algorithm: str, setting: str, sbm_weights: dict, averaged: dict, fit: dict,
                                 weights_path: Path) -> None:
    path = sbm_arm_dir(algorithm, setting) / "settings.json"
    snapshot = _sbm_snapshot(algorithm, setting, sbm_weights, averaged, fit, weights_path)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        keys = ("sbm_weights_per_sd", "averaged_weights_per_sd", "screen_sizes", "draws", "draw_seed_base")
        changed = [k for k in keys if saved.get(k) != snapshot[k]]
        if changed and any(sbm_arm_dir(algorithm, setting).glob("pair_*.csv")):
            raise RuntimeError(f"{path.parent} holds results made with different weights or settings "
                               f"({', '.join(changed)}), for example SBM weights refitted on more pairs. Move that "
                               "folder away (or rename it) to rerun with the current weights.")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {**snapshot, "written_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")})


def _sbm_tasks(algorithms, settings, n_pairs: int, sbm_weights: dict) -> list[tuple]:
    tasks = [("sbm_weights", a, s, i) for a in algorithms for s in settings if (a, s) in sbm_weights
             for i in range(n_pairs) if pair_path(a, s, i).exists() and not sbm_arm_path(a, s, i).exists()]
    return sorted(tasks, key=lambda t: (t[3], ALGORITHMS.index(t[1]), list(G.SETTINGS).index(t[2])))


def _sbm_task_seconds(task: tuple) -> float:
    _, algorithm, setting, _ = task
    saved = load_calibration(algorithm, setting) or {}
    budgets = len(saved.get("budgets") or [0] * 6)
    per_run = SECONDS_PER_RUN[algorithm] * (FAQ_RUNS_PER_SET if algorithm == "faq" else 1)
    return budgets * len(SBM_RULES) * 0.9 * per_run + budgets * 0.5      # 0.5 s per budget for the two screens


def _execute_sbm(task: tuple, sbm_weights: dict, averaged: dict, blas_threads: int) -> str:
    _, algorithm, setting, pair_index = task
    return sbm_weights_job(algorithm, setting, pair_index, sbm_weights[(algorithm, setting)], averaged[algorithm],
                           blas_threads)


def _describe_sbm(task: tuple) -> str:
    _, algorithm, setting, pair_index = task
    return f"{ALGORITHM_LABELS[algorithm]} {SETTING_LABELS[setting]} · pair {pair_index} · SBM weights"


def run_sbm_weights(algorithms=ALGORITHMS, settings=SETTINGS, jobs: int = 3, blas_threads: int = 3,
                    n_pairs: int = N_PAIRS, weights_path: Path | None = None) -> None:
    """The direct test of SBM's own weights on the rules experiment's test pairs, `jobs` pairs at a time."""

    weights_path = Path(weights_path or SBM_WEIGHTS_PATH)
    sbm_weights = load_sbm_weights(weights_path)
    fit = _sbm_fit_info(weights_path)
    averaged = averaged_weights()
    algorithms = [a for a in ALGORITHMS if a in set(algorithms)]
    settings = [s for s in G.SETTINGS if s in set(settings)]
    missing = [f"{ALGORITHM_LABELS[a]} {SETTING_LABELS[s]}" for a in algorithms for s in settings
               if (a, s) not in sbm_weights]
    if missing:
        print(f"No SBM weights saved for {', '.join(missing)}: skipped.", flush=True)
    for algorithm in algorithms:
        for setting in settings:
            if (algorithm, setting) in sbm_weights:
                _check_and_save_sbm_settings(algorithm, setting, sbm_weights, averaged, fit, weights_path)
    waiting = sum(1 for a in algorithms for s in settings if (a, s) in sbm_weights for i in range(n_pairs)
                  if not pair_path(a, s, i).exists())
    if waiting:
        print(f"{waiting} test pairs have no rules results yet and are skipped; run run_all() first.", flush=True)
    tasks = _sbm_tasks(algorithms, settings, n_pairs, sbm_weights)
    if not tasks:
        print(f"Nothing to run for SBM's weights. Results are in {OUTPUT_DIR}/<algorithm>/<setting>/{SBM_ARM}/")
        return
    started = time.perf_counter()
    print(f"{_now()}  SBM's own weights: {len(tasks)} pairs, {jobs} at a time: about "
          f"{sum(_sbm_task_seconds(t) for t in tasks) / max(jobs, 1) / 60:.0f} min. Each line below is one finished "
          "pair.", flush=True)
    try:
        failures = run_tasks(tasks, _execute_sbm, (sbm_weights, averaged, blas_threads), _describe_sbm,
                             _sbm_task_seconds, lambda task: [], jobs)
    except KeyboardInterrupt:
        return
    if failures:
        raise RuntimeError(f"Failed: {'; '.join(failures)}. Rerun to retry them (finished pairs are kept).")
    print(f"{_now()}  Done in {(time.perf_counter() - started) / 60:.0f} min.", flush=True)


def sbm_weights_status(algorithms=ALGORITHMS, settings=SETTINGS, n_pairs: int = N_PAIRS,
                       weights_path: Path | None = None) -> pd.DataFrame:
    path = Path(weights_path or SBM_WEIGHTS_PATH)
    fit = _sbm_fit_info(path) if path.exists() else {}
    rows = []
    for algorithm in algorithms:
        for setting in settings:
            info = fit.get((algorithm, setting))
            rows.append({"algorithm": ALGORITHM_LABELS[algorithm], "setting": SETTING_LABELS[setting],
                         "SBM weights": (f"fitted at k = {info['budget']} on {info['pairs']} pairs" if info
                                         else "not saved yet"),
                         "test pairs with rules results": sum(pair_path(algorithm, setting, i).exists()
                                                              for i in range(n_pairs)),
                         "test pairs with SBM-weight results": sum(sbm_arm_path(algorithm, setting, i).exists()
                                                                   for i in range(n_pairs))})
    return pd.DataFrame(rows).set_index(["algorithm", "setting"])


# --------------------------------------------------------------------------
# Analysis helpers for sbm_rules.ipynb
# --------------------------------------------------------------------------

KEYS = ["arm", "algorithm", "model", "regularizer", "pair_set", "pair_index", "rule", "budget"]
FEATURES = ("f_degree", "f_harmonic", "f_coverage", "f_pair_mass", "f_signature_r2", "f_signature_r3")
CARRIED = ("seed_vertices", "blocks_covered", "largest_block_share", "score", "score_percentile",
           *[f"z_{f}" for f in BN.CORE], *FEATURES)


def _present(algorithms, settings) -> list[tuple[str, str]]:
    return [(a, s) for a in algorithms for s in settings if any(setting_dir(a, s).glob("pair_*.csv"))]


def read_runs(algorithms=ALGORITHMS, settings=SETTINGS) -> pd.DataFrame:
    """Every solver run (one row per restart)."""

    pattern = "pair_[0-9][0-9][0-9].csv"
    files = [f for a, s in _present(algorithms, settings) for folder in (setting_dir(a, s), sbm_arm_dir(a, s))
             for f in sorted(folder.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No results in {OUTPUT_DIR} yet. Run run_all() first.")
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for column in ("success", "success_090", "cached"):
        runs[column] = _flag(runs[column])
    return runs


def read_sbm_scores(algorithms=ALGORITHMS, settings=SETTINGS) -> pd.DataFrame:
    """Every rule's set on SBM's own score, per pair and budget (written by the direct test)."""

    files = [f for a in algorithms for s in settings for f in sorted(sbm_arm_dir(a, s).glob("pair_*_scores.csv"))]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def _collapse(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per evaluation: success if any restart succeeded (FAQ's best of 3); accuracy of the best-objective run."""

    runs = runs.copy()
    runs["regularizer"] = pd.to_numeric(runs["regularizer"], errors="coerce").fillna(0).astype(int)  # 0: no GOAT
    runs["order_key"] = pd.to_numeric(runs["objective"], errors="coerce").fillna(-np.inf)
    selected = runs.sort_values("order_key", kind="stable").groupby(KEYS, sort=False).tail(1).set_index(KEYS)
    grouped = runs.groupby(KEYS, sort=False)
    out = pd.DataFrame({"success": grouped["success"].any(), "success_090": grouped["success_090"].any(),
                        "accuracy": selected["accuracy"], "runs": grouped.size()})
    for column in CARRIED:
        if column in runs:
            out[column] = grouped[column].first()
    return out.reset_index()


def load_evaluations(algorithms=ALGORITHMS, settings=SETTINGS) -> pd.DataFrame:
    """One row per (algorithm, setting, pair, rule, budget). model holds the setting, arm the algorithm."""

    runs = read_runs(algorithms, settings)
    runs = runs.assign(arm=runs["algorithm"], model=runs["setting"], pair_set="sbm")
    ev = _collapse(runs)
    ev["success"] = ev["success"].astype(bool)
    return ev


def round1_evaluations(algorithms=ALGORITHMS, models=("sparse_er", "chung_lu")) -> pd.DataFrame:
    """Round 1's stored results on the matched models (20 pairs each), in the same form."""

    frames = []
    for algorithm in algorithms:
        for model in models:
            path = ROUND1_ROOT / algorithm / model / "runs.csv"
            if not path.exists():
                continue
            old = pd.read_csv(path).rename(columns={"strategy": "rule", "final_unseeded_accuracy": "accuracy"})
            old = old.assign(arm=algorithm, regularizer=BT.regularizer(algorithm, model), pair_set="round1",
                             success=_flag(old["success"]), success_090=_flag(old["success_090"]))
            frames.append(_collapse(old))
    if not frames:
        raise FileNotFoundError(f"No round-1 results in {ROUND1_ROOT}.")
    ev = pd.concat(frames, ignore_index=True)
    ev["success"] = ev["success"].astype(bool)
    return ev


def queries_needed(ev: pd.DataFrame) -> pd.DataFrame:
    """Smallest budget that succeeds, per pair and rule (one grid step past the largest budget if none does)."""

    return BT.queries_needed(ev)


def success_curves(ev: pd.DataFrame) -> pd.DataFrame:
    """Share of pairs that succeed, per algorithm, setting, rule and budget."""

    data = ev.assign(success=ev["success"].astype(float))
    return data.groupby(["algorithm", "model", "rule", "budget"], as_index=False).agg(
        success=("success", "mean"), pairs=("success", "size"))


def _grid_means(ev: pd.DataFrame, algorithm: str, model: str) -> pd.DataFrame:
    """Each pair's mean success over the budget grid, per rule (pairs as rows)."""

    part = ev[(ev["algorithm"] == algorithm) & (ev["model"] == model)]
    return part.assign(success=part["success"].astype(float)).pivot_table(
        index="pair_index", columns="rule", values="success", aggfunc="mean")


def _rules_in(frame: pd.DataFrame) -> list[str]:
    present = set(frame["rule"])
    order = [r for a in ALGORITHMS for r in rules_for(a)]
    return list(dict.fromkeys(r for r in order if r in present)) + sorted(present - set(order))


def _label(algorithm: str, setting: str) -> tuple[str, str]:
    return ALGORITHM_LABELS[algorithm], SETTING_LABELS.get(setting, MODEL_LABELS.get(setting, setting))


def summary_table(ev: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """Per algorithm and setting: each rule's mean success over the budget grid and mean queries needed (95%
    intervals from resampling pairs)."""

    rows = []
    for algorithm, setting in _settings_in(ev):
        grid = _grid_means(ev, algorithm, setting)
        for rule in _rules_in(ev[(ev["algorithm"] == algorithm) & (ev["model"] == setting)]):
            success = BT._mean_interval(grid[rule].dropna())
            q = BT.mean_queries(queries, algorithm, setting, rule)
            rows.append({"algorithm": _label(algorithm, setting)[0], "setting": _label(algorithm, setting)[1],
                         "rule": rule_label(rule), "pairs": len(grid[rule].dropna()),
                         "mean success over budgets": f"{success[0]:.2f} [{success[1]:.2f}, {success[2]:.2f}]",
                         "mean queries needed": f"{q['mean']:.1f} [{q['low']:.1f}, {q['high']:.1f}]"})
    return pd.DataFrame(rows).set_index(["algorithm", "setting", "rule"])


def _settings_in(frame: pd.DataFrame) -> list[tuple[str, str]]:
    present = set(zip(frame["algorithm"], frame["model"]))
    order = [*G.SETTINGS, *BN.MODELS]
    return [(a, s) for a in ALGORITHMS for s in order if (a, s) in present]


def comparison_table(ev: pd.DataFrame, queries: pd.DataFrame, against=("random", "top_degree")) -> pd.DataFrame:
    """Every rule against random and against top degree, pair by pair: queries needed (fewer / same / more, sign
    test) and the change in mean success over the budget grid."""

    rows = []
    for algorithm, setting in _settings_in(ev):
        grid = _grid_means(ev, algorithm, setting)
        rules = _rules_in(ev[(ev["algorithm"] == algorithm) & (ev["model"] == setting)])
        for reference in against:
            for rule in rules:
                if rule in against[:against.index(reference) + 1] or rule not in grid or reference not in grid:
                    continue
                result = BT.compare(queries, algorithm, setting, rule, reference)
                if not result:
                    continue
                mean, low, high = BT._mean_interval((grid[rule] - grid[reference]).dropna())
                rows.append({"algorithm": _label(algorithm, setting)[0], "setting": _label(algorithm, setting)[1],
                             "against": rule_label(reference), "rule": rule_label(rule), "pairs": result["pairs"],
                             "needs fewer / same / more queries": f"{result['fewer']} / {result['same']} / {result['more']}",
                             "p": round(result["p"], 4) if result["p"] == result["p"] else np.nan,
                             "change in queries": result["mean change"],
                             "change in mean success": f"{mean:+.2f} [{low:+.2f}, {high:+.2f}]"})
    return pd.DataFrame(rows).set_index(["algorithm", "setting", "against", "rule"])


def community_effect(ev: pd.DataFrame, round1: pd.DataFrame, against: str = "random") -> pd.DataFrame:
    """Each round-1 rule's gain over `against` (mean success over the budget grid, paired by pair) on the SBM setting
    and on its matched round-1 model, with 95% intervals. The grids differ (k50-based here, pilot-based in round 1),
    so compare signs and rough sizes rather than exact values."""

    rows = []
    for algorithm, setting in _settings_in(ev):
        if setting not in G.SETTINGS:
            continue
        base = base_model(setting)
        sbm, old = _grid_means(ev, algorithm, setting), _grid_means(round1, algorithm, base)
        for rule in S.STRATEGIES["main"][algorithm]:
            if rule in ("random", against) or rule not in sbm or against not in sbm:
                continue
            here = BT._mean_interval((sbm[rule] - sbm[against]).dropna())
            there = (BT._mean_interval((old[rule] - old[against]).dropna()) if rule in old and against in old
                     else (np.nan, np.nan, np.nan))
            rows.append({"algorithm": algorithm, "setting": setting, "base_model": base, "rule": rule,
                         "sbm": here[0], "sbm_low": here[1], "sbm_high": here[2],
                         "base": there[0], "base_low": there[1], "base_high": there[2]})
    return pd.DataFrame(rows)


def show_community_effect(table: pd.DataFrame, against: str = "random") -> pd.DataFrame:
    rows = []
    for r in table.to_dict("records"):
        rows.append({"algorithm": ALGORITHM_LABELS[r["algorithm"]], "setting": SETTING_LABELS[r["setting"]],
                     "rule": rule_label(r["rule"]),
                     f"with communities: change vs {rule_label(against)}":
                         f"{r['sbm']:+.2f} [{r['sbm_low']:+.2f}, {r['sbm_high']:+.2f}]",
                     f"round 1 ({MODEL_LABELS[r['base_model']]}): change vs {rule_label(against)}":
                         (f"{r['base']:+.2f} [{r['base_low']:+.2f}, {r['base_high']:+.2f}]"
                          if r["base"] == r["base"] else "")})
    return pd.DataFrame(rows).set_index(["algorithm", "setting", "rule"])


def coverage_table(ev: pd.DataFrame) -> pd.DataFrame:
    """Where each rule's seeds land: blocks covered and the largest block's share of the seeds, averaged over pairs
    and budgets, with the rule's mean success over the grid."""

    rows = []
    for algorithm, setting in _settings_in(ev):
        part = ev[(ev["algorithm"] == algorithm) & (ev["model"] == setting)]
        blocks = len(G.SETTINGS[setting]["blocks"])
        for rule in _rules_in(part):
            sub = part[part["rule"] == rule]
            rows.append({"algorithm": ALGORITHM_LABELS[algorithm], "setting": SETTING_LABELS[setting],
                         "rule": rule_label(rule),
                         f"blocks covered (of {blocks})": round(float(sub["blocks_covered"].mean()), 2),
                         "sets covering every block": round(float((sub["blocks_covered"] == blocks).mean()), 2),
                         "largest block's share of seeds": round(float(sub["largest_block_share"].mean()), 2),
                         "mean success": round(float(sub["success"].astype(float).mean()), 2)})
    return pd.DataFrame(rows).set_index(["algorithm", "setting", "rule"])


SELECTION_COMPARISONS = (("sbm_best_1000", "random"), ("sbm_best_100", "random"), ("sbm_best_1000", "best_1000"),
                         ("sbm_best_1000", "degree_1000"), ("sbm_best_1000", "top_degree"))


def selection_summary(ev: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """summary_table for the rules of the direct test: random, top degree, best of 1,000 by degree and by the
    averaged score, and best of 100 and 1,000 by SBM's own score."""

    return summary_table(ev[ev["rule"].isin(SELECTION_RULES)], queries)


def selection_comparison(ev: pd.DataFrame, queries: pd.DataFrame, comparisons=SELECTION_COMPARISONS) -> pd.DataFrame:
    """Best of N by SBM's own score against random, the averaged score, degree alone and top degree, pair by pair:
    queries needed (fewer / same / more, sign test) and the change in mean success over the budget grid."""

    rows = []
    for algorithm, setting in _settings_in(ev):
        grid = _grid_means(ev, algorithm, setting)
        for first, second in comparisons:
            if first not in grid or second not in grid:
                continue
            result = BT.compare(queries, algorithm, setting, first, second)
            if not result:
                continue
            mean, low, high = BT._mean_interval((grid[first] - grid[second]).dropna())
            rows.append({"algorithm": _label(algorithm, setting)[0], "setting": _label(algorithm, setting)[1],
                         "rule": rule_label(first), "against": rule_label(second), "pairs": result["pairs"],
                         "needs fewer / same / more queries": f"{result['fewer']} / {result['same']} / {result['more']}",
                         "p": round(result["p"], 4) if result["p"] == result["p"] else np.nan,
                         "change in queries": result["mean change"],
                         "change in mean success": f"{mean:+.2f} [{low:+.2f}, {high:+.2f}]"})
    return pd.DataFrame(rows).set_index(["algorithm", "setting", "rule", "against"]) if rows else pd.DataFrame()


def score_picked_table(ev: pd.DataFrame, sbm_scores: pd.DataFrame | None = None, n_boot: int = 2000) -> pd.DataFrame:
    """On each pair and budget, take the round-1 rule whose set scores highest, with the averaged weights and (once
    the direct test has run) with SBM's own weights. Mean success over the budget grid, next to top degree and the
    best fixed rule chosen in hindsight; differences pair by pair with 95% intervals. Top degree by block is left
    out of the candidates: it uses the true block labels."""

    rows = []
    for algorithm, setting in _settings_in(ev):
        if setting not in G.SETTINGS:
            continue
        candidates = [r for r in S.STRATEGIES["main"][algorithm] if r != "random"]
        part = ev[(ev["algorithm"] == algorithm) & (ev["model"] == setting) & ev["rule"].isin(candidates)]
        success = part.pivot_table(index=["pair_index", "budget"], columns="rule", values="success",
                                   aggfunc="first").astype(float).dropna()
        if success.empty:
            continue
        scores = {"averaged weights": part.pivot_table(index=["pair_index", "budget"], columns="rule",
                                                       values="score", aggfunc="first")}
        if sbm_scores is not None and not sbm_scores.empty:
            sub = sbm_scores[(sbm_scores["algorithm"] == algorithm) & (sbm_scores["setting"] == setting)
                             & sbm_scores["rule"].isin(candidates)]
            if not sub.empty:
                scores["SBM weights"] = sub.pivot_table(index=["pair_index", "budget"], columns="rule",
                                                        values="sbm_score", aggfunc="first")
        best = success.mean().idxmax()
        per_pair = {"top degree": success["top_degree"], "best fixed rule (hindsight)": success[best]}
        most = {}
        for name, table in scores.items():
            table = table.reindex(index=success.index, columns=success.columns).dropna()
            picked = table.idxmax(axis=1)
            per_pair[f"score-picked, {name}"] = pd.Series(
                [success.at[i, r] for i, r in picked.items()], index=picked.index)
            counts = picked.value_counts(normalize=True)
            most[name] = f"{rule_label(counts.index[0])} ({counts.iloc[0]:.0%})"
        means = pd.DataFrame(per_pair).groupby(level="pair_index").mean()
        row = {"algorithm": ALGORITHM_LABELS[algorithm], "setting": SETTING_LABELS[setting], "pairs": len(means),
               **{column: round(float(means[column].mean()), 2) for column in means.columns},
               "best fixed rule": rule_label(best)}
        for name in scores:
            column = f"score-picked, {name}"
            for reference in ("top degree", "best fixed rule (hindsight)"):
                m, low, high = BT._mean_interval((means[column] - means[reference]).dropna().to_numpy(float), n_boot)
                row[f"{name}: picked - {reference}"] = f"{m:+.2f} [{low:+.2f}, {high:+.2f}]"
            row[f"{name}: rule picked most often"] = most[name]
        rows.append(row)
    return pd.DataFrame(rows).set_index(["algorithm", "setting"]) if rows else pd.DataFrame()


def calibration_table(algorithms=ALGORITHMS, settings=SETTINGS) -> pd.DataFrame:
    """The calibrated budgets, with the matched round-1 model's training budget for comparison."""

    rows = []
    for algorithm in algorithms:
        for setting in settings:
            saved = load_calibration(algorithm, setting)
            if saved is None:
                continue
            rates = ", ".join(f"{b}: {r:.2f}" for b, r in sorted(saved["rates"].items(), key=lambda x: int(x[0])))
            rows.append({"algorithm": ALGORITHM_LABELS[algorithm], "setting": SETTING_LABELS[setting],
                         f"matched model's training budget": f"{saved['start_budget']} "
                                                            f"({MODEL_LABELS[saved['base_model']]})",
                         "k50": saved["k50"], "status": saved["status"],
                         "random success by budget (30 sets each)": rates,
                         "budget grid": ", ".join(map(str, saved["budgets"]))})
    return pd.DataFrame(rows).set_index(["algorithm", "setting"]) if rows else pd.DataFrame()


def pair_table(runs: pd.DataFrame) -> pd.DataFrame:
    """The test pairs: mean and largest degree, share of edges inside blocks, and the share of A's edges kept in B."""

    first = runs.drop_duplicates(["algorithm", "setting", "pair_index"])
    table = first.groupby(["algorithm", "setting"], sort=False).agg(
        pairs=("pair_index", "size"), mean_degree=("mean_degree_a", "mean"), largest_degree=("max_degree_a", "mean"),
        share_inside_blocks=("within_share_a", "mean"), edges_kept_in_B=("kept_edge_fraction", "mean"))
    table.index = pd.MultiIndex.from_tuples([(ALGORITHM_LABELS[a], SETTING_LABELS[s]) for a, s in table.index],
                                            names=["algorithm", "setting"])
    return table.round(2)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

INK = {"primary": "#0b0b0b", "secondary": "#52514e", "grid": "#e6e6e6"}
RULE_STYLE = {**{rule: {**style, "linestyle": "-", "markerfacecolor": style["color"]}
                 for rule, style in BN.RULE_STYLE.items()},
              # top degree's neutral ink with an open marker and a dashed line: a variant of top degree
              "community_top_degree": {"color": "#2b2b2b", "marker": "s", "linestyle": "--", "markerfacecolor": "white"},
              # best of N by SBM's own score: the screened rules' blues, dashed with open diamonds
              "sbm_best_100": {"color": "#5598e7", "marker": "D", "linestyle": "--", "markerfacecolor": "white"},
              "sbm_best_1000": {"color": "#1c5cab", "marker": "D", "linestyle": "--", "markerfacecolor": "white"}}


def _axes(ax) -> None:
    ax.grid(True, color=INK["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK["secondary"], labelsize=9)


def plot_success_curves(curves: pd.DataFrame):
    """Share of pairs that succeed against the budget: one panel per algorithm (rows) and setting (columns)."""

    import matplotlib.pyplot as plt

    algorithms = [a for a in ALGORITHMS if a in set(curves["algorithm"])]
    settings = [s for s in G.SETTINGS if s in set(curves["model"])]
    fig, axes = plt.subplots(len(algorithms), len(settings), figsize=(5.2 * len(settings), 3.7 * len(algorithms)),
                             squeeze=False, constrained_layout=True)
    handles = {}
    for i, algorithm in enumerate(algorithms):
        for j, setting in enumerate(settings):
            ax = axes[i, j]
            part = curves[(curves["algorithm"] == algorithm) & (curves["model"] == setting)]
            _axes(ax)
            if part.empty:
                ax.text(0.5, 0.5, "no results yet", ha="center", va="center", color=INK["secondary"],
                        transform=ax.transAxes)
                continue
            for rule in _rules_in(part):
                sub = part[part["rule"] == rule].sort_values("budget")
                style = RULE_STYLE.get(rule, {"color": "#52514e", "marker": "o", "linestyle": "-",
                                              "markerfacecolor": "#52514e"})
                width = 2.4 if rule in ("random", "top_degree") else 1.6
                (line,) = ax.plot(sub["budget"], sub["success"], color=style["color"], marker=style["marker"],
                                  linestyle=style["linestyle"], markerfacecolor=style["markerfacecolor"],
                                  markeredgecolor=style["color"], linewidth=width, markersize=6,
                                  label=rule_label(rule))
                handles.setdefault(rule, line)
            saved = load_calibration(algorithm, setting) or {}
            ax.set_xticks(sorted(part["budget"].unique()))
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"{ALGORITHM_LABELS[algorithm]} · {SETTING_LABELS[setting]}"
                         + (f" (k50 = {saved['k50']})" if saved.get("k50") else ""), fontsize=11,
                         color=INK["primary"])
            ax.set_xlabel("seeds (budget)", fontsize=9.5, color=INK["secondary"])
            if j == 0:
                ax.set_ylabel("share of pairs that succeed", fontsize=9.5, color=INK["secondary"])
    order = [r for r in _rules_in(curves) if r in handles]
    fig.legend([handles[r] for r in order], [rule_label(r) for r in order], loc="outside lower center",
               ncol=min(5, len(order)), frameon=False, fontsize=9.5)
    return fig


def plot_community_effect(table: pd.DataFrame, against: str = "random"):
    """Each round-1 rule's gain over `against`, with communities (filled) and in round 1 on the matched model (open)."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    algorithms = [a for a in ALGORITHMS if a in set(table["algorithm"])]
    settings = [s for s in G.SETTINGS if s in set(table["setting"])]
    fig, axes = plt.subplots(len(algorithms), len(settings), figsize=(5.0 * len(settings), 0.55 * 7 * len(algorithms) + 0.8),
                             squeeze=False, constrained_layout=True)
    for i, algorithm in enumerate(algorithms):
        for j, setting in enumerate(settings):
            ax = axes[i, j]
            _axes(ax)
            part = table[(table["algorithm"] == algorithm) & (table["setting"] == setting)]
            ax.axvline(0, color=INK["secondary"], linewidth=1.0, linestyle=":")
            if part.empty:
                ax.text(0.5, 0.5, "no results yet", ha="center", va="center", color=INK["secondary"],
                        transform=ax.transAxes)
                continue
            rules = list(part["rule"])
            for y, r in enumerate(part.to_dict("records")):
                style = RULE_STYLE.get(r["rule"], {"color": "#52514e", "marker": "o"})
                if r["base"] == r["base"]:
                    ax.errorbar(r["base"], y + 0.17, xerr=[[r["base"] - r["base_low"]], [r["base_high"] - r["base"]]],
                                fmt=style["marker"], color=style["color"], markerfacecolor="white", markersize=7,
                                elinewidth=1.2, capsize=0)
                ax.errorbar(r["sbm"], y - 0.17, xerr=[[r["sbm"] - r["sbm_low"]], [r["sbm_high"] - r["sbm"]]],
                            fmt=style["marker"], color=style["color"], markersize=7, elinewidth=1.6, capsize=0)
            ax.set_yticks(range(len(rules)))
            ax.set_yticklabels([rule_label(r) for r in rules], fontsize=9.5)
            ax.set_ylim(len(rules) - 0.5, -0.5)
            ax.set_xlim(-1.0, 1.0)
            base = MODEL_LABELS[part["base_model"].iloc[0]]
            ax.set_title(f"{ALGORITHM_LABELS[algorithm]} · {SETTING_LABELS[setting]} (vs {base})", fontsize=11,
                         color=INK["primary"])
            ax.set_xlabel(f"change in mean success vs {rule_label(against)}", fontsize=9.5, color=INK["secondary"])
    legend = [Line2D([], [], marker="o", color="#52514e", markerfacecolor="#52514e", linestyle="none",
                     label="with communities (this experiment)"),
              Line2D([], [], marker="o", color="#52514e", markerfacecolor="white", linestyle="none",
                     label="round 1, matched model without communities")]
    fig.legend(handles=legend, loc="outside lower center", ncol=2, frameon=False, fontsize=9.5)
    return fig


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=ALGORITHMS)
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(G.SETTINGS))
    parser.add_argument("--pairs", type=int, default=N_PAIRS, help=f"pairs per setting (default {N_PAIRS})")
    parser.add_argument("--jobs", type=int, default=3, help="tasks run in parallel (default 3)")
    parser.add_argument("--blas-threads", type=int, default=3)
    parser.add_argument("--status", action="store_true", help="only show how far the run is")
    parser.add_argument("--sbm-weights", action="store_true",
                        help="the direct test of SBM's own weights (after sbm_models.ipynb has saved them)")
    args = parser.parse_args(argv)
    if args.status:
        table = sbm_weights_status if args.sbm_weights else status
        print(table(args.algorithms, args.settings, args.pairs).to_string())
    elif args.sbm_weights:
        run_sbm_weights(args.algorithms, args.settings, jobs=args.jobs, blas_threads=args.blas_threads,
                        n_pairs=args.pairs)
    else:
        run_all(args.algorithms, args.settings, jobs=args.jobs, blas_threads=args.blas_threads, n_pairs=args.pairs)


if __name__ == "__main__":
    main(sys.argv[1:])
