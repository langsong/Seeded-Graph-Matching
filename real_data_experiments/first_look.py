"""First look at real graph pairs: do round 1's rules and the frozen scores pick better seeds than random sets?
(7 Nov 2026)

Pairs (real_data_experiments.datasets; unweighted, undirected, no self-loops; A is the graph seeds are chosen from):
    worm            C. elegans hermaphrodite, chemical (A) vs gap-junction (B) synapses, all 279 neurons
    worm_lcc        the same, the 247 neurons in the largest connected component of both graphs
    worm_male       C. elegans male, chemical vs gap junction, the 167 neurons named in both
    enron_130_131   Enron e-mail, week 130 (A) vs week 131 (B), largest component of both (145 addresses)
    enron_130_132   week 130 vs week 132 (141 addresses)
    wikipedia       English (A) vs French (B) articles, all 1,382; the large pair, run on its own

Per pair, TEST_SHARE of the vertices are fixed test vertices that are never seeds, and B's labels are shuffled once.
Per pair and budget (BUDGETS):
    random     the first RANDOM_SETS of a pool of POOL uniform random seed sets
    rule       round 1's rules (RULES; the union over algorithms), never seeding a test vertex
    screen     the best set of the pool by each graph model's frozen score for the algorithm (best_er, best_cl,
               best_paper) and by degree (best_degree)
Every set gets the four core features with z-scores against the pool, the frozen score of all three models with its
percentile in the pool, and its seed vertices.
Variants: percolation (r = 2), FAQ (RUNS["faq"] runs per set with different solver seeds), GOAT with regulariser 100
(the ER setting) and 300 (Chung-Lu and PAPER). Round 1's solver settings otherwise.
Outcome: accuracy on the test vertices ("accuracy_test"), plus non-seed accuracy, first-step accuracy, objective,
iterations and run time. For the hermaphrodite worm also accuracy by neuron class (motor, interneuron, sensory) and
"same_class_test", the share of test neurons matched to a neuron of the same class.

Wikipedia is much slower (1,382 vertices: tens of seconds per FAQ or GOAT run), so it has fewer random sets and one
FAQ run per set, and its tasks run in order of priority (percolation, FAQ, GOAT 100, GOAT 300) instead of longest
first.

Run from the repository root (ds environment):

    python -m real_data_experiments.first_look                       # the five small pairs
    python -m real_data_experiments.first_look --pairs wikipedia
    python -m real_data_experiments.first_look --pairs worm worm_lcc --variants faq goat_r300
    python -m real_data_experiments.first_look --status
    python -m real_data_experiments.first_look --summary              # random sets vs rules and screens

Results: real_data_experiments/results/first_look/<pair>/<variant>/k<budget>.csv (one row per run, with the set's
features) and <pair>/pair.json, or under FIRST_LOOK_OUTPUT_DIR. Rows are written run by run, so a stopped run resumes
where it stopped (Ctrl-C stops the workers too). `read()` loads everything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
import zlib
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache"))

import numpy as np
import pandas as pd

from experiment_common.provenance import collect_environment, thread_limit
from fixed_pair_seed_set_classification import multi_pair_collection as MP
from selection_rule_comparison import graded_outcomes as G
from selection_rule_comparison import regime_map as RM
from selection_rule_comparison import run_comparison as rc
from selection_rule_comparison import settings as S
from selection_rule_comparison.strategies import STRATEGY_CODES, GraphA

from . import datasets as D

PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("FIRST_LOOK_OUTPUT_DIR") or PACKAGE_DIR / "results" / "first_look")

PAIRS = {
    "worm": {"get": {"name": "worm"}, "budgets": (25, 50, 100, 150, 200)},
    "worm_lcc": {"get": {"name": "worm", "vertices": "common_lcc"}, "budgets": (25, 50, 100, 150)},
    "worm_male": {"get": {"name": "worm_male"}, "budgets": (10, 20, 40, 60, 80)},
    "enron_130_131": {"get": {"name": "enron", "weeks": [130, 131], "vertices": "common_lcc"},
                      "budgets": (5, 10, 20, 40, 60)},
    "enron_130_132": {"get": {"name": "enron", "weeks": [130, 132], "vertices": "common_lcc"},
                      "budgets": (5, 10, 20, 40, 60)},
    "wikipedia": {"get": {"name": "wikipedia"}, "budgets": (10, 20, 30, 50, 75), "large": True},
}
SMALL = tuple(p for p in PAIRS if not PAIRS[p].get("large"))
LARGE = tuple(p for p in PAIRS if PAIRS[p].get("large"))
VARIANTS = {
    "percolation": {"algorithm": "percolation", "regularizer": None},
    "faq": {"algorithm": "faq", "regularizer": None},
    "goat_r100": {"algorithm": "goat", "regularizer": 100.0},
    "goat_r300": {"algorithm": "goat", "regularizer": 300.0},
}
TEST_SHARE = 0.2
POOL = 1_000
RULES = ("top_degree", "top_harmonic", "clustered", "spread", "distinguishing", "hub_free_distinguishing")
MODELS = RM.MODELS
SCREEN_NAMES = {"sparse_er": "best_er", "chung_lu": "best_cl", "paper_pa": "best_paper"}
SIZES = {   # small pairs and the large one (Wikipedia)
    "small": {"random_sets": 30, "rules": list(RULES), "screens": True,
              "runs": {"percolation": 1, "faq": 3, "goat": 1}},
    "large": {"random_sets": 15, "rules": list(RULES), "screens": True,
              "runs": {"percolation": 1, "faq": 1, "goat": 1}},
}
SMOKE = {   # --smoke: one budget on two small pairs and a single Wikipedia set per variant (its run time)
    "pairs": ("worm", "enron_130_131", "wikipedia"),
    "budgets": {"worm": (100,), "enron_130_131": (20,), "wikipedia": (50,)},
    "pool": 100,
    "sizes": {"small": {"random_sets": 2, "rules": ["top_degree", "spread"], "screens": True,
                        "runs": {"percolation": 1, "faq": 2, "goat": 1}},
              "large": {"random_sets": 1, "rules": [], "screens": False,
                        "runs": {"percolation": 1, "faq": 1, "goat": 1}}},
}
SEED_BASE = 20261190
PURPOSE = {"shuffle": 1, "test": 2, "pool": 3, "rules": 4, "solver": 5}
CORE = RM.CORE
WORM_CLASSES = ("motor", "interneuron", "sensory")
LABELS = {"percolation": "Percolation", "faq": "SGM (FAQ)", "goat_r100": "GOAT (r 100)", "goat_r300": "GOAT (r 300)"}


def _now() -> str:
    return RM._now()


def _clock() -> str:
    return RM._clock()


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, content: dict) -> None:
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _code(pair_name: str) -> int:
    return zlib.crc32(pair_name.encode("utf-8"))


def _size(pair_name: str) -> str:
    return "large" if PAIRS[pair_name].get("large") else "small"


# --------------------------------------------------------------------------
# Design, pairs and seed sets
# --------------------------------------------------------------------------

def _fixed(smoke: bool) -> dict:
    design = {
        "version": 1, "smoke": smoke, "pairs": {name: {**spec, "budgets": list(spec["budgets"])}
                                                for name, spec in PAIRS.items()},
        "variants": VARIANTS, "test_share": TEST_SHARE, "pool": POOL, "sizes": SIZES, "seed_base": SEED_BASE,
        "weights": {m: RM.load_weights(m) for m in MODELS},
        "solver": {"sgm": S.SGM, "percolation": S.PERCOLATION},
        "uniqueness_cap": {a: MP.UNIQUENESS_CAP[a] for a in RM.ALGORITHMS},
    }
    if smoke:
        for name, budgets in SMOKE["budgets"].items():
            design["pairs"][name]["budgets"] = list(budgets)
        design["pool"], design["sizes"] = SMOKE["pool"], SMOKE["sizes"]
    return json.loads(json.dumps(design))


def prepare(blas_threads: int = 3, smoke: bool = False) -> dict:
    """The design every job uses, frozen in settings.json once results exist."""

    path = OUTPUT_DIR / "settings.json"
    fixed = _fixed(smoke)
    if path.exists():
        saved = _read_json(path)
        differs = [key for key in fixed if saved.get(key) != fixed[key]]
        if differs:
            raise ValueError(f"The saved design in {path} differs ({', '.join(differs)}). Set FIRST_LOOK_OUTPUT_DIR "
                             "to a new folder to keep the results apart.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, {**fixed, "written_at": _now(), "environment": collect_environment(blas_threads)})
    return {**fixed, "blas_threads": blas_threads}


def _get_kwargs(spec: dict) -> dict:
    kwargs = dict(spec["get"])
    if "weeks" in kwargs:
        kwargs["weeks"] = tuple(kwargs["weeks"])
    return kwargs


def load_pair(pair_name: str, design: dict):
    """(pair, A, shuffled B, truth, test vertices, candidate seeds, details)."""

    pair = D.get_pair(**_get_kwargs(design["pairs"][pair_name]))
    code = _code(pair_name)
    A, B, truth = pair.for_matching(rc._seed(SEED_BASE, code, PURPOSE["shuffle"]))
    n = pair.n
    n_test = int(round(design["test_share"] * n))
    test = np.sort(np.random.default_rng([SEED_BASE, code, PURPOSE["test"]]).choice(n, n_test, replace=False))
    candidates = np.setdiff1d(np.arange(n), test)
    info = {"pair": pair_name, "settings": pair.settings, "n": int(n), "n_test": int(n_test),
            "hash_a": rc._hash(A), "hash_b": rc._hash(B), "test_vertices": [int(v) for v in test],
            "description": {k: (v.item() if hasattr(v, "item") else v) for k, v in D.describe(pair.A, pair.B).items()}}
    return pair, A, B, truth, test, candidates, info


def plan(pair_name: str, algorithm: str, graph: GraphA, test: np.ndarray, candidates: np.ndarray, k: int,
         design: dict):
    """The seed sets run at one budget, with their features: (table, list of sorted seed arrays)."""

    size = design["sizes"][_size(pair_name)]
    code = _code(pair_name)
    pool = G.draw_sets(np.random.default_rng([SEED_BASE, code, PURPOSE["pool"], k]), candidates, k, design["pool"])
    features = G._core_features(graph, algorithm, pool)
    stats = {f: (float(features[f].mean()), float(features[f].std(ddof=1))) for f in CORE}

    def z(f, values):
        mean, sd = stats[f]
        return (np.asarray(values, dtype=float) - mean) / sd if sd > 0 else np.zeros(np.size(values))

    weights = {m: design["weights"][m][algorithm] for m in MODELS}
    pool_scores = {m: np.sort(sum(weights[m][f] * z(f, features[f]) for f in CORE)) for m in MODELS}
    rows, arrays = [], []

    def add(kind, detail, seeds, raw, pool_index=-1):
        zs = {f: float(z(f, [raw[f]])[0]) for f in CORE}
        row = {"set_id": len(rows), "design": kind, "detail": detail, "pool_index": int(pool_index),
               **{f: float(raw[f]) for f in CORE}, **{f"z_{f}": zs[f] for f in CORE}}
        for m in MODELS:
            value = float(sum(weights[m][f] * zs[f] for f in CORE))
            row[f"score_{SCREEN_NAMES[m][5:]}"] = value
            row[f"percentile_{SCREEN_NAMES[m][5:]}"] = float(
                100 * np.searchsorted(pool_scores[m], value + 1e-12) / len(pool))
        rows.append(row)
        arrays.append(np.sort(np.asarray(seeds, dtype=np.int64)))

    for j in range(min(size["random_sets"], len(pool))):
        add("random", "", pool[j], {f: features[f][j] for f in CORE}, j)
    if size["rules"]:
        k_max = max(design["pairs"][pair_name]["budgets"])
        rule_sets = np.stack([
            np.sort(RM._ordering(rule, graph, k_max,
                                 np.random.default_rng([SEED_BASE, code, PURPOSE["rules"], STRATEGY_CODES[rule]]),
                                 test)[:k]) for rule in size["rules"]])
        rule_features = G._core_features(graph, algorithm, rule_sets)
        for r, rule in enumerate(size["rules"]):
            add("rule", rule, rule_sets[r], {f: rule_features[f][r] for f in CORE})
    if size["screens"]:
        for m in MODELS:
            score = sum(weights[m][f] * z(f, features[f]) for f in CORE)
            best = int(np.argmax(score))
            add("screen", SCREEN_NAMES[m], pool[best], {f: features[f][best] for f in CORE}, best)
        best = int(np.argmax(features["degree"]))
        add("screen", "best_degree", pool[best], {f: features[f][best] for f in CORE}, best)
    return pd.DataFrame(rows), arrays


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------

def result_path(pair_name: str, variant: str, k: int) -> Path:
    return OUTPUT_DIR / pair_name / variant / f"k{k:03d}.csv"


def describe(task) -> str:
    pair_name, variant, k = task
    return f"{pair_name}, {LABELS[variant]}, {k} seeds"


def _runs(pair_name: str, variant: str, design: dict) -> int:
    return design["sizes"][_size(pair_name)]["runs"][VARIANTS[variant]["algorithm"]]


def _planned(task, design: dict) -> int:
    pair_name, variant, k = task
    size = design["sizes"][_size(pair_name)]
    n_sets = min(size["random_sets"], design["pool"]) + len(size["rules"]) + (4 if size["screens"] else 0)
    return n_sets * _runs(pair_name, variant, design)


def _left(task, design: dict) -> int:
    return max(_planned(task, design) - len(G._completed_ids(result_path(*task), "run_id")), 0)


def _matching(algorithm: str, n: int) -> np.ndarray | None:
    """The final matching of the last run (A vertex -> B vertex, -1 if unmatched), as captured by regime_map."""

    if algorithm == "percolation":
        matching = RM._PERCOLATION.get("matching")
        return None if matching is None else np.asarray(matching, dtype=np.int64)
    if "indices_A" not in G._CAPTURE:
        return None
    matching = np.full(n, -1, dtype=np.int64)
    matching[G._CAPTURE["indices_A"]] = G._CAPTURE["indices_B"]
    return matching


def class_measures(matching: np.ndarray | None, truth: np.ndarray, test: np.ndarray, classes: np.ndarray) -> dict:
    """Worm: test accuracy per neuron class, and the share of test neurons matched to a neuron of the same class."""

    out = {f"accuracy_test_{c}": np.nan for c in WORM_CLASSES}
    out["same_class_test"] = np.nan
    if matching is None:
        return out
    class_of_b = classes[np.argsort(truth)]          # class of each vertex of the shuffled B
    for c in WORM_CLASSES:
        vertices = test[classes[test] == c]
        if len(vertices):
            out[f"accuracy_test_{c}"] = float(np.mean(matching[vertices] == truth[vertices]))
    matched = matching[test]
    same = np.zeros(len(test), dtype=bool)
    ok = matched >= 0
    same[ok] = class_of_b[matched[ok]] == classes[test][ok]
    out["same_class_test"] = float(same.mean())
    return out


OUTCOMES = ("accuracy_test", "accuracy", "success", "h0_accuracy", "objective", "n_iter", "converged",
            "runtime_seconds", "matched_fraction", "error")


def job(task, design: dict) -> str:
    pair_name, variant, k = task
    path = result_path(pair_name, variant, k)
    if _left(task, design) == 0:
        return f"{describe(task)}: already done"
    started = time.perf_counter()
    algorithm, regularizer = VARIANTS[variant]["algorithm"], VARIANTS[variant]["regularizer"]
    with thread_limit(design["blas_threads"]):
        pair, A, B, truth, test, candidates, info = load_pair(pair_name, design)
        if k > len(candidates):
            raise ValueError(f"{describe(task)}: only {len(candidates)} vertices can be seeds")
        info_path = OUTPUT_DIR / pair_name / "pair.json"
        info_path.parent.mkdir(parents=True, exist_ok=True)
        if info_path.exists():
            saved = _read_json(info_path)
            for key in ("hash_a", "hash_b", "test_vertices"):
                if saved.get(key) != info[key]:
                    raise RuntimeError(f"{describe(task)}: the saved runs come from a different pair ({key}). Set "
                                       "FIRST_LOOK_OUTPUT_DIR to a new folder to keep them apart.")
        else:
            _write_json(info_path, {**info, "written_at": _now()})
        graph = GraphA(A)
        table, arrays = plan(pair_name, algorithm, graph, test, candidates, k, design)
        path.parent.mkdir(parents=True, exist_ok=True)
        done = G._completed_ids(path, "run_id")
        classes = pair.vertex_class
        rows, written = G._Rows(path), 0
        try:
            for repeat in range(_runs(pair_name, variant, design)):     # every set once, then the repeats
                for set_id, seeds in enumerate(arrays):
                    run_id = 10 * set_id + repeat
                    if run_id in done:
                        continue
                    solver_seed = rc._seed(SEED_BASE, _code(pair_name), PURPOSE["solver"], k,
                                           MP.ALGORITHM_CODES[algorithm], set_id, repeat)
                    out = RM.run_one(algorithm, "sparse_er", (A, B, truth), seeds, test, solver_seed, regularizer)
                    extra = class_measures(_matching(algorithm, len(A)), truth, test, classes) \
                        if classes is not None else {}
                    rows.write({"pair": pair_name, "n": len(A), "variant": variant, "algorithm": algorithm,
                                "regularizer": regularizer if regularizer is not None else np.nan, "budget": k,
                                **table.iloc[set_id].to_dict(), "repeat": repeat, "run_id": run_id,
                                "solver_seed": solver_seed, **{c: out[c] for c in OUTCOMES}, **extra,
                                "seed_vertices": json.dumps([int(v) for v in seeds])})
                    written += 1
        finally:
            rows.close()
    return f"{describe(task)}: {written} runs in {(time.perf_counter() - started) / 60:.1f} min"


_SIZES: dict = {}


def _seconds_per_run(pair_name: str, variant: str, design: dict) -> float:
    """Rough wall time per run (time estimates only): the regime map's figure for 600 vertices, scaled by n^2.5."""

    if pair_name not in _SIZES:
        _SIZES[pair_name] = D.get_pair(**_get_kwargs(design["pairs"][pair_name])).n
    return RM.SECONDS_PER_RUN[VARIANTS[variant]["algorithm"]] * max(_SIZES[pair_name] / 600, 0.1) ** 2.5


def tasks_for(pairs, variants, design: dict) -> list:
    tasks = []
    for pair_name in pairs:
        for variant in variants:
            tasks += [(pair_name, variant, int(k)) for k in design["pairs"][pair_name]["budgets"]]
    return tasks


def run(pairs=SMALL, variants=tuple(VARIANTS), jobs: int = 3, blas_threads: int = 3, smoke: bool = False) -> bool:
    """Run every unfinished task. Small pairs go longest first; the large pair in order of priority (percolation,
    FAQ, GOAT 100, GOAT 300; small budgets first). Returns False if stopped with Ctrl-C."""

    design = prepare(blas_threads, smoke)
    tasks = [t for t in tasks_for(pairs, variants, design) if _left(t, design) > 0]
    if not tasks:
        print(f"First look: nothing left to run for {', '.join(pairs)}. Results are in {OUTPUT_DIR}")
        return True

    def seconds(task):
        return _left(task, design) * _seconds_per_run(task[0], task[1], design)

    small = sorted((t for t in tasks if _size(t[0]) == "small"), key=seconds, reverse=True)
    large = [t for t in tasks if _size(t[0]) == "large"]          # already in priority order
    tasks = small + large
    print(f"{_clock()}  First look ({', '.join(pairs)}): {len(tasks)} tasks, {jobs} at a time, about "
          f"{sum(seconds(t) for t in tasks) / jobs / 3600:.1f} h (a rough guess). One line per finished task.",
          flush=True)
    return _run(tasks, design, seconds, jobs)


def _run(tasks: list, design: dict, seconds, jobs: int) -> bool:
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


# --------------------------------------------------------------------------
# Status, reading and a first summary
# --------------------------------------------------------------------------

def status(pairs=tuple(PAIRS), variants=tuple(VARIANTS), smoke: bool = False) -> pd.DataFrame:
    path = OUTPUT_DIR / "settings.json"
    design = _read_json(path) if path.exists() else _fixed(smoke)
    rows = []
    for pair_name in pairs:
        row = {"pair": pair_name}
        for variant in variants:
            tasks = tasks_for([pair_name], [variant], design)
            planned = sum(_planned(t, design) for t in tasks)
            done = planned - sum(_left(t, design) for t in tasks)
            row[variant] = f"{done} of {planned}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("pair")


def read() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(OUTPUT_DIR.glob("*/*/k*.csv")) if path.stat().st_size > 0]
    if not frames:
        raise FileNotFoundError(f"No first-look results yet in {OUTPUT_DIR}")
    return pd.concat(frames, ignore_index=True)


def summary(data: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per pair, variant and budget: random sets' mean test accuracy (and SD over sets), every rule and screen's mean
    over its runs, and the Spearman correlation between each model's frozen score and accuracy over random sets."""

    data = read() if data is None else data
    data = data[data["error"].fillna("").astype(str) == ""]
    rows = []
    for (pair_name, variant, k), part in data.groupby(["pair", "variant", "budget"], sort=False):
        random = part[part["design"] == "random"]
        per_set = random.groupby("set_id")["accuracy_test"].mean()
        row = {"pair": pair_name, "variant": variant, "budget": int(k), "random sets": len(per_set),
               "random mean": per_set.mean(), "random SD": per_set.std(ddof=1)}
        for detail, values in part[part["design"] != "random"].groupby("detail")["accuracy_test"]:
            row[detail] = values.mean()
        scores = random.groupby("set_id")[[f"score_{m[5:]}" for m in SCREEN_NAMES.values()]].first()
        for column in scores:
            row[f"rho({column})"] = (scores[column].rank().corr(per_set.rank())
                                     if per_set.nunique() > 1 and len(per_set) > 2 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index(["pair", "variant", "budget"])


# --------------------------------------------------------------------------
# Analysis (first_look.ipynb)
# --------------------------------------------------------------------------

BLUE, ORANGE, RED, INK, INK2, GRID, BAND, NEUTRAL = (RM.BLUE, RM.ORANGE, RM.RED, RM.INK, RM.INK2, RM.GRID, RM.BAND,
                                                     RM.NEUTRAL)
CONTEXT = "#b9b8b2"   # random seed sets
VARIANT_COLORS = {"faq": "#2a78d6", "goat_r100": "#eb6834", "goat_r300": "#1baf7a", "percolation": "#eda100"}
PAIR_LABELS = {"worm": "Worm (hermaphrodite), all 279", "worm_lcc": "Worm, common largest component (247)",
               "worm_male": "Worm (male), 167", "enron_130_131": "Enron, weeks 130 vs 131 (145)",
               "enron_130_132": "Enron, weeks 130 vs 132 (141)", "wikipedia": "Wikipedia, English vs French (1,382)"}
PAIR_SHORT = {"worm": "Worm (all)", "worm_lcc": "Worm (LCC)", "worm_male": "Male worm", "enron_130_131": "Enron 130-131",
              "enron_130_132": "Enron 130-132", "wikipedia": "Wikipedia"}
SCREENS = ("best_er", "best_cl", "best_paper", "best_degree")
SCREEN_LABELS = {"best_er": "best of 1,000 by ER score", "best_cl": "best of 1,000 by Chung-Lu score",
                 "best_paper": "best of 1,000 by PAPER score", "best_degree": "best of 1,000 by degree"}
METHODS = (*RULES, *SCREENS)
METHOD_LABELS = {**{m: RM.METHOD_LABELS[m] for m in RULES}, **SCREEN_LABELS}
METHOD_STYLE = {**{m: RM.METHOD_STYLE[m] for m in RULES},
                "best_er": dict(color="#1c5cab", marker="o"), "best_cl": dict(color="#5598e7", marker="s"),
                "best_paper": dict(color="#0d366b", marker="^"), "best_degree": dict(color="#1baf7a", marker="X")}
SCORE_COLUMNS = {"score_er": "ER score", "score_cl": "Chung-Lu score", "score_paper": "PAPER score",
                 "z_degree": "degree (z)"}
SET_KEYS = ["pair", "variant", "budget", "set_id"]
CLASS_COLUMNS = ("same_class_test", "accuracy_test_motor", "accuracy_test_interneuron", "accuracy_test_sensory")


def _ok(frame: pd.DataFrame) -> pd.Series:
    return frame["error"].fillna("").astype(str).str.len() == 0


def _spearman(x, y) -> float:
    x, y = pd.Series(np.asarray(x, dtype=float)), pd.Series(np.asarray(y, dtype=float))
    keep = x.notna() & y.notna()
    if keep.sum() < 3 or x[keep].nunique() < 2 or y[keep].nunique() < 2:
        return np.nan
    return float(x[keep].rank().corr(y[keep].rank()))


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def per_set(data: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per pair, variant, budget and seed set: test accuracy ("accuracy": FAQ's runs averaged;
    "accuracy_first": the first run), first-step accuracy, the worm's class measures, and the set's features and
    frozen scores."""

    data = read() if data is None else data
    data = data[_ok(data)].sort_values("repeat", kind="stable")
    aggregations = {"accuracy": ("accuracy_test", "mean"), "accuracy_first": ("accuracy_test", "first"),
                    "h0_accuracy": ("h0_accuracy", "mean"), "runs": ("accuracy_test", "size")}
    aggregations.update({c: (c, "mean") for c in CLASS_COLUMNS if c in data.columns})
    outcome = data.groupby(SET_KEYS, sort=False).agg(**aggregations)
    keep = ["n", "algorithm", "regularizer", "design", "detail", "pool_index", *CORE, *(f"z_{f}" for f in CORE),
            *SCORE_COLUMNS, "percentile_er", "percentile_cl", "percentile_paper"]
    features = data.drop_duplicates(SET_KEYS).set_index(SET_KEYS)
    out = outcome.join(features[[c for c in dict.fromkeys(keep) if c in features.columns]]).reset_index()
    out["method"] = np.where(out["design"] == "random", "random", out["detail"].fillna(""))
    return out


def pair_table() -> pd.DataFrame:
    """The real pairs as run: size, test vertices, edges and degrees of A and B, and how much they overlap (edge
    correlation; share of A's edges also in B)."""

    rows = []
    for name in PAIRS:
        path = OUTPUT_DIR / name / "pair.json"
        if not path.exists():
            continue
        info = _read_json(path)
        d = info["description"]
        rows.append({"pair": PAIR_LABELS[name], "vertices": info["n"], "test vertices": info["n_test"],
                     "edges in A": d["A_edges"], "edges in B": d["B_edges"],
                     "mean degree A": round(d["A_mean_degree"], 1), "mean degree B": round(d["B_mean_degree"], 1),
                     "degree CV of A": round(d["A_degree_cv"], 2), "isolated in B": d["B_isolated"],
                     "edge correlation": round(d["edge_correlation"], 3),
                     "share of A's edges in B": round(d["share_of_A_edges_in_B"], 3),
                     "budgets": ", ".join(str(k) for k in PAIRS[name]["budgets"])})
    return pd.DataFrame(rows).set_index("pair")


def _random_stats(per: pd.DataFrame) -> pd.DataFrame:
    random = per[per["design"] == "random"].groupby(["pair", "variant", "budget"])["accuracy"]
    return pd.DataFrame({"random mean": random.mean(), "random SD": random.std(),
                         "random 10th pct": random.quantile(0.1), "random 90th pct": random.quantile(0.9),
                         "best random set": random.max(), "random sets": random.size()})


def budget_table(per: pd.DataFrame, variant: str = "faq") -> pd.DataFrame:
    """Per pair and budget: random sets' mean, SD, 10th and 90th percentiles and best set, then each rule's and screen's
    test accuracy (FAQ: mean of its runs)."""

    part = per[per["variant"] == variant]
    stats = _random_stats(part).xs(variant, level="variant")
    methods = part[part["design"] != "random"].pivot_table(index=["pair", "budget"], columns="method",
                                                           values="accuracy")
    methods = methods[[m for m in METHODS if m in methods.columns]].rename(columns=METHOD_LABELS)
    out = stats.join(methods)
    order = {p: i for i, p in enumerate(PAIRS)}
    out = out.reset_index().assign(_o=lambda t: t["pair"].map(order)).sort_values(["_o", "budget"]).drop(columns="_o")
    return out.assign(pair=out["pair"].map(PAIR_LABELS)).set_index(["pair", "budget"]).round(3)


def plot_budgets(per: pd.DataFrame, variant: str = "faq",
                 methods=("distinguishing", "hub_free_distinguishing", "top_degree", "best_er", "clustered")):
    """Test accuracy against the number of seeds, one panel per pair: random sets' mean (grey line) and 10th-90th
    percentile (grey band), and the chosen rules and screens (lines)."""

    import matplotlib.pyplot as plt

    part = per[per["variant"] == variant]
    pairs = [p for p in PAIRS if p in set(part["pair"])]
    columns = 3
    rows = int(np.ceil(len(pairs) / columns))
    colors = dict(zip(methods, (BLUE, ORANGE, "#1baf7a", "#eda100", "#e87ba4")))
    with plt.rc_context({"font.size": 9}):
        fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.4 * rows + 0.9), squeeze=False)
        for ax, pair in zip(axes.ravel(), pairs):
            cell = part[part["pair"] == pair]
            random = cell[cell["design"] == "random"].groupby("budget")["accuracy"]
            budgets = np.array(sorted(cell["budget"].unique()))
            ax.fill_between(budgets, random.quantile(0.1).reindex(budgets), random.quantile(0.9).reindex(budgets),
                            color=CONTEXT, alpha=0.3, linewidth=0, label="random sets, 10th-90th percentile")
            ax.plot(budgets, random.mean().reindex(budgets), color="#8a8984", linewidth=2, label="random sets, mean")
            for method in methods:
                line = cell[cell["method"] == method].groupby("budget")["accuracy"].mean().reindex(budgets)
                ax.plot(budgets, line, color=colors[method], linewidth=2, marker=METHOD_STYLE[method]["marker"],
                        markersize=6, markeredgecolor="white", label=METHOD_LABELS[method])
            _style(ax)
            ax.set_title(PAIR_LABELS[pair], loc="left", fontsize=9.5)
            ax.set_xlabel("seeds")
            ax.set_ylim(bottom=0)
        for ax in axes.ravel()[len(pairs):]:
            ax.axis("off")
        for r in range(rows):
            axes[r, 0].set_ylabel("test accuracy")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        height = fig.get_figheight()
        fig.suptitle(f"Real pairs, {LABELS[variant]}: random seed sets vs rules and screens", x=0.01, y=0.995,
                     ha="left", va="top", fontsize=11.5)
        fig.legend(handles, labels, loc="upper left", ncol=3, frameon=False, fontsize=8.8,
                   bbox_to_anchor=(0.005, 1 - 0.36 / height))
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.85 / height))
    return fig


def gain_table(per: pd.DataFrame) -> pd.DataFrame:
    """Each rule's and screen's gain in test accuracy over the random sets' mean at the same pair, variant and
    budget, averaged over budgets: one column per pair, then all pairs, and the share of (pair, budget) cases above
    the random mean."""

    stats = _random_stats(per)["random mean"]
    others = per[per["design"] != "random"].join(stats, on=["pair", "variant", "budget"])
    others = others.assign(gain=others["accuracy"] - others["random mean"])
    table = others.pivot_table(index=["variant", "method"], columns="pair", values="gain", aggfunc="mean")
    table = table[[p for p in PAIRS if p in table.columns]]
    table["all pairs"] = others.groupby(["variant", "method"])["gain"].mean()
    table["share above random"] = others.assign(up=others["gain"] > 0).groupby(["variant", "method"])["up"].mean()
    order = {(v, m): (i, j) for i, v in enumerate(VARIANTS) for j, m in enumerate(METHODS)}
    table = table.iloc[sorted(range(len(table)), key=lambda i: order.get(table.index[i], (99, 99)))]
    table.index = pd.MultiIndex.from_tuples([(LABELS[v], METHOD_LABELS.get(m, m)) for v, m in table.index],
                                            names=["variant", "method"])
    return table.rename(columns=PAIR_LABELS).round(3)


def plot_gain_heatmap(per: pd.DataFrame, variant: str = "faq"):
    """Gain over random sets (averaged over budgets) per method (rows) and pair (columns): blue above random, red
    below."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    table = gain_table(per)
    table = table.xs(LABELS[variant], level="variant").drop(columns=["share above random"])
    cmap = LinearSegmentedColormap.from_list("red_grey_blue", (RED, NEUTRAL, BLUE))
    limit = max(0.05, float(np.nanmax(np.abs(table.to_numpy(dtype=float)))))
    with plt.rc_context({"font.size": 9}):
        fig, ax = plt.subplots(figsize=(1.3 * table.shape[1] + 3.5, 0.42 * table.shape[0] + 1.8))
        image = ax.imshow(table.to_numpy(dtype=float), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                value = table.iat[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(value) > 0.6 * limit else INK)
        ax.set_xticks(range(table.shape[1]), [c.replace(", ", ",\n", 1) for c in table.columns], fontsize=8)
        ax.set_yticks(range(table.shape[0]), table.index)
        ax.axvline(table.shape[1] - 1.5, color="white", linewidth=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
        bar.set_label("gain in test accuracy over random sets")
        ax.set_title(f"{LABELS[variant]}: gain over random sets, averaged over budgets", loc="left", fontsize=10.5)
        fig.tight_layout()
    return fig


def score_table(per: pd.DataFrame) -> pd.DataFrame:
    """Do the frozen scores rank random sets on real pairs? Spearman between each score (and degree alone) and test
    accuracy over the random sets of one pair, variant and budget, averaged over budgets."""

    rows = []
    for (pair, variant, budget), part in per[per["design"] == "random"].groupby(["pair", "variant", "budget"]):
        rows.append({"pair": pair, "variant": variant, "budget": budget,
                     **{label: _spearman(part[column], part["accuracy"]) for column, label in SCORE_COLUMNS.items()}})
    table = pd.DataFrame(rows).groupby(["pair", "variant"])[list(SCORE_COLUMNS.values())].mean()
    order = {(p, v): (i, j) for i, p in enumerate(PAIRS) for j, v in enumerate(VARIANTS)}
    table = table.iloc[sorted(range(len(table)), key=lambda i: order.get(table.index[i], (99, 99)))]
    table.index = pd.MultiIndex.from_tuples([(PAIR_LABELS[p], LABELS[v]) for p, v in table.index],
                                            names=["pair", "variant"])
    return table.round(2)


def plot_scores(per: pd.DataFrame, variant: str = "faq", score: str = "score_er"):
    """Frozen score (or degree) against test accuracy on real pairs: rows are pairs, columns budgets. Grey: random
    sets; orange shapes: round 1's rules; blues and aqua: the screens. Titles: Spearman over the random sets."""

    import matplotlib.pyplot as plt

    part = per[per["variant"] == variant]
    pairs = [p for p in PAIRS if p in set(part["pair"])]
    columns = max(len(PAIRS[p]["budgets"]) for p in pairs)
    with plt.rc_context({"font.size": 8.5}):
        fig, axes = plt.subplots(len(pairs), columns, figsize=(3.0 * columns, 2.55 * len(pairs) + 1.0),
                                 squeeze=False)
        for r, pair in enumerate(pairs):
            budgets = sorted(part.loc[part["pair"] == pair, "budget"].unique())
            for c in range(columns):
                ax = axes[r, c]
                if c >= len(budgets):
                    ax.axis("off")
                    continue
                cell = part[(part["pair"] == pair) & (part["budget"] == budgets[c])]
                random = cell[cell["design"] == "random"]
                _style(ax)
                ax.scatter(random[score], random["accuracy"], s=14, color=CONTEXT, linewidths=0, label="random sets")
                for method in METHODS:
                    chosen = cell[cell["method"] == method]
                    if chosen.empty:
                        continue
                    style = METHOD_STYLE[method]
                    ax.scatter(chosen[score], chosen["accuracy"], s=34, color=style["color"], marker=style["marker"],
                               edgecolors="white", linewidths=0.6, zorder=3, label=METHOD_LABELS[method])
                rho = _spearman(random[score], random["accuracy"])
                ax.set_title(f"{budgets[c]} seeds · ρ_s {rho:.2f}" if np.isfinite(rho) else f"{budgets[c]} seeds",
                             loc="left", fontsize=8.8)
                if c == 0:
                    ax.set_ylabel(f"{PAIR_SHORT[pair]}\ntest accuracy", fontsize=8.5)
                if r == len(pairs) - 1:
                    ax.set_xlabel(SCORE_COLUMNS.get(score, score))
        handles, labels = [], []
        for ax in axes.ravel():
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        height = fig.get_figheight()
        fig.suptitle(f"Real pairs, {LABELS[variant]}: {SCORE_COLUMNS.get(score, score)} vs test accuracy "
                     "(ρ_s: Spearman over random sets)", x=0.01, y=0.998, ha="left", va="top", fontsize=11)
        fig.legend(handles, labels, loc="upper left", ncol=6, frameon=False, fontsize=8,
                   bbox_to_anchor=(0.005, 1 - 0.32 / height))
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.95 / height))
    return fig


def variant_table(per: pd.DataFrame) -> pd.DataFrame:
    """Random sets' mean test accuracy per pair, budget and solver variant."""

    table = per[per["design"] == "random"].pivot_table(index=["pair", "budget"], columns="variant", values="accuracy")
    table = table[[v for v in VARIANTS if v in table.columns]].rename(columns=LABELS)
    order = {p: i for i, p in enumerate(PAIRS)}
    table = table.reset_index().assign(_o=lambda t: t["pair"].map(order)).sort_values(["_o", "budget"])
    return table.drop(columns="_o").assign(pair=lambda t: t["pair"].map(PAIR_LABELS)).set_index(
        ["pair", "budget"]).round(3)


def plot_variants(per: pd.DataFrame):
    """Random sets' mean test accuracy against seeds, one line per solver variant: how much the matcher (and GOAT's
    regulariser) matters on each real pair."""

    import matplotlib.pyplot as plt

    random = per[per["design"] == "random"]
    pairs = [p for p in PAIRS if p in set(random["pair"])]
    columns = 3
    rows = int(np.ceil(len(pairs) / columns))
    with plt.rc_context({"font.size": 9}):
        fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.3 * rows + 0.7), squeeze=False)
        for ax, pair in zip(axes.ravel(), pairs):
            for variant in VARIANTS:
                line = random[(random["pair"] == pair) & (random["variant"] == variant)].groupby("budget")[
                    "accuracy"].mean()
                if len(line):
                    ax.plot(line.index, line.to_numpy(), color=VARIANT_COLORS[variant], linewidth=2, marker="o",
                            markersize=5, markeredgecolor="white", label=LABELS[variant])
            _style(ax)
            ax.set_title(PAIR_LABELS[pair], loc="left", fontsize=9.5)
            ax.set_xlabel("seeds")
            ax.set_ylim(bottom=0)
        for ax in axes.ravel()[len(pairs):]:
            ax.axis("off")
        for r in range(rows):
            axes[r, 0].set_ylabel("test accuracy, random sets (mean)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        height = fig.get_figheight()
        fig.suptitle("Real pairs: the four solver variants on random seed sets", x=0.01, y=0.995, ha="left",
                     va="top", fontsize=11.5)
        fig.legend(handles, labels, loc="upper left", ncol=4, frameon=False, fontsize=9,
                   bbox_to_anchor=(0.005, 1 - 0.36 / height))
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.62 / height))
    return fig


def same_class_chance(pair: str = "worm") -> float:
    """Share of test neurons a random matching puts in the right class: the sum of squared class shares."""

    path = OUTPUT_DIR / "settings.json"
    design = _read_json(path) if path.exists() else _fixed(False)
    real, _, _, _, test, _, _ = load_pair(pair, design)
    if real.vertex_class is None:
        return np.nan
    shares = pd.Series(real.vertex_class[test]).value_counts(normalize=True)
    return float((shares ** 2).sum())


def _groups(part: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(part["design"] == "random", "random", part["method"]), index=part.index)


def _ordered(table: pd.DataFrame) -> pd.DataFrame:
    order = ["random", *METHODS]
    table = table.reindex([g for g in order if g in table.index])
    table.index = ["random sets (mean)" if g == "random" else METHOD_LABELS[g] for g in table.index]
    return table


def class_table(per: pd.DataFrame, pair: str = "worm", variant: str = "faq") -> pd.DataFrame:
    """The worm: test accuracy and the share of test neurons matched to a neuron of the same class, per budget (columns),
    for random sets (mean) and each rule and screen (rows). Compare the same-class share with same_class_chance()."""

    part = per[(per["pair"] == pair) & (per["variant"] == variant)]
    if "same_class_test" not in part.columns or part["same_class_test"].isna().all():
        raise ValueError(f"No class measures for {pair}")
    part = part.assign(group=_groups(part))
    tables = {name: _ordered(part.pivot_table(index="group", columns="budget", values=column, aggfunc="mean"))
              for name, column in (("test accuracy", "accuracy"), ("same class", "same_class_test"))}
    return pd.concat(tables, axis=1).round(3)


def class_breakdown(per: pd.DataFrame, pair: str = "worm", variant: str = "faq") -> pd.DataFrame:
    """The worm: test accuracy within each neuron class (motor, interneuron, sensory), averaged over budgets, for random
    sets (mean) and each rule and screen."""

    part = per[(per["pair"] == pair) & (per["variant"] == variant)]
    part = part.assign(group=_groups(part))
    columns = {"accuracy_test_motor": "motor", "accuracy_test_interneuron": "interneuron",
               "accuracy_test_sensory": "sensory", "accuracy": "all test neurons"}
    table = part.groupby("group")[list(columns)].mean().rename(columns=columns)
    return _ordered(table).round(3)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", nargs="+", choices=(*PAIRS, "small", "all"), default=["small"],
                        help="pairs to run: names, 'small' (default: all but Wikipedia) or 'all'")
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--smoke", action="store_true", help="tiny design for a quick test, including one Wikipedia "
                        "set per variant (use a separate FIRST_LOOK_OUTPUT_DIR)")
    parser.add_argument("--status", action="store_true", help="only show how far it is")
    parser.add_argument("--summary", action="store_true", help="only print the summary table")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--blas-threads", type=int, default=3)
    args = parser.parse_args(argv)
    names = []
    for name in args.pairs:
        names += list(SMALL if name == "small" else PAIRS if name == "all" else [name])
    if args.pairs == ["small"] and (args.smoke or args.status):
        names = list(SMOKE["pairs"]) if args.smoke else list(PAIRS)
    names = list(dict.fromkeys(names))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    if args.status:
        print(status(names, args.variants, args.smoke).to_string())
        return
    if args.summary:
        print(summary().round(3).to_string())
        return
    if not run(names, args.variants, args.jobs, args.blas_threads, args.smoke):
        raise SystemExit(130)


if __name__ == "__main__":
    main()
