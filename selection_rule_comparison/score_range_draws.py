"""Draws 3-10 of B for the 27 synthetic score-range banks, so every seed set gets 11 draws.

The score-range collection (collect_score_ranges.py) evaluated each synthetic bank on
draws 0-2 of B (draw 0 is the saved B). This add-on evaluates the same seed sets on
draws 3-10 from the same conditional-B generator (b_replicates.draw_b), with the same
matcher settings and solver-seed rule. A set's accuracy can then be averaged over 11
draws of B, as in the landscape notebook, across the bank's full score range.

From the repository root, in the ds environment:
    caffeinate -i python -u -m selection_rule_comparison.score_range_draws --hours 10 --jobs 3
    python -m selection_rule_comparison.score_range_draws --status
    python -m selection_rule_comparison.score_range_draws --smoke --hours 0.2

Rerun the same command to resume: completed runs are skipped and a row cut off by an
interruption is removed. Work is queued draw by draw (draw 3 for every bank, then draw 4,
...), so stopping early leaves banks with equal numbers of draws, give or take one.

The original collection is read only. Outputs go to
selection_rule_comparison/data/score_range_draws/<bank>/<algorithm>_draw_XX_runs.csv,
one file per bank and draw, with the same columns as the original run files.
Analysis: score_range_11_draws.ipynb (score_range_draws_analysis.py).
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import traceback
import warnings

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache"))

import numpy as np
import pandas as pd

from experiment_common.provenance import collect_environment, thread_limit
from real_data_experiments import first_look as F
from . import b_replicates as BR
from . import collect_score_ranges as C
from . import graded_outcomes as G
from . import landscape as L
from . import learned_selector as LS
from . import regime_map as RM
from . import run_comparison as RC

SOURCE_DIR = C.DEFAULT_DIR
DEFAULT_DIR = L.EXPERIMENT_DIR / "data" / "score_range_draws"
SMOKE_DIR = C.REPO_ROOT / "tmp" / "score_range_draws_smoke"
FIRST_DRAW, LAST_DRAW = 3, 10   # the source collection holds draws 0-2
CHECK_DRAW = 1                  # every job regenerates this draw and compares it with the source's saved hash
SMOKE_SETS = 4
ALGORITHMS = ("percolation", "faq", "goat")
CODE_FILES = ("selection_rule_comparison/score_range_draws.py", "selection_rule_comparison/b_replicates.py",
              "selection_rule_comparison/landscape.py", "selection_rule_comparison/regime_map.py",
              "selection_rule_comparison/run_comparison.py", "selection_rule_comparison/settings.py",
              "experiment_common/sgm.py", "experiment_common/percolation.py",
              "experiment_common/graph_checks.py", "experiment_common/graphs.py", "utils/Graphs.py")


def run_id(candidate_id, draw, repeat=0):
    """Unique within a bank. The source's 100*id + 10*draw + repeat would collide at draw 10."""

    return 1000 * int(candidate_id) + 10 * int(draw) + int(repeat)


def runs_path(root, task, draw):
    return Path(root) / task["id"] / f"{task['algorithm']}_draw_{draw:02d}_runs.csv"


def check_time(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Collection time budget reached")


def complete_ids(path):
    """Run ids in complete rows. Read only: safe while a worker is appending to the file."""

    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or "run_id" not in header:
            return set()
        column = header.index("run_id")
        return {int(row[column]) for row in reader if len(row) == len(header)}


def median_seconds(source, task):
    """Median solver time per run on the bank's first three draws (for job order and estimates)."""

    path = Path(source) / task["id"] / f"{task['algorithm']}_runs.csv"
    try:
        values = pd.read_csv(path, usecols=["runtime_seconds"]).runtime_seconds.dropna()
    except (FileNotFoundError, ValueError):
        return np.nan
    return float(values.median()) if len(values) else np.nan


def design(root, source, smoke, last_draw, blas_threads):
    """The frozen design. Read only: it never writes."""

    source_settings = F._read_json(source / "settings.json")
    if source_settings["synthetic_draws"] != FIRST_DRAW:
        raise ValueError(f"Expected a source collection with draws 0-{FIRST_DRAW - 1}; "
                         f"it has {source_settings['synthetic_draws']} synthetic draws.")
    tasks = [t for t in source_settings["banks"] if t["kind"] == "synthetic"]
    if smoke:  # one small bank per algorithm
        tasks = [min((t for t in tasks if t["algorithm"] == a), key=lambda t: (t["budget"], t["id"]))
                 for a in ALGORITHMS if any(t["algorithm"] == a for t in tasks)]
    banks = []
    for task in tasks:
        folder = source / task["id"]
        if not (folder / "sets.csv").exists():
            raise ValueError(f"Source bank {task['id']} has no committed seed bank (sets.csv).")
        meta = F._read_json(folder / "pair.json")
        banks.append({**task, "pair_seed": int(meta["pair_seed"]), "n_sets": int(meta["n_sets"]),
                      "sets_hash": meta["sets_hash"],
                      "check_draw": F._read_json(folder / f"draw_{CHECK_DRAW:02d}.json")})
    if not banks:
        raise ValueError(f"No synthetic banks in {source}")
    last = FIRST_DRAW if smoke else last_draw
    if last < FIRST_DRAW:
        raise ValueError(f"--last-draw must be at least {FIRST_DRAW}")
    repo = C.REPO_ROOT
    settings = {"version": 1, "smoke": bool(smoke), "source": str(source),
                "source_settings_hash": C.digest(source_settings), "seed_base": C.SEED_BASE,
                "draws": list(range(FIRST_DRAW, last + 1)), "sets_per_bank": SMOKE_SETS if smoke else None,
                "banks": banks, "blas_threads": blas_threads, "solver": F.S.SGM, "percolation": F.S.PERCOLATION,
                "goat_regularizer": F.S.GOAT_REGULARIZER,
                "code_hashes": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in CODE_FILES}}
    return json.loads(json.dumps(settings))


def prepare(root, source, args):
    settings = design(root, source, args.smoke, args.last_draw, args.blas_threads)
    path = root / "settings.json"
    if path.exists():
        saved = F._read_json(path)
        if saved != settings:
            changed = sorted(k for k in set(saved) | set(settings) if saved.get(k) != settings.get(k))
            raise ValueError(f"Saved design differs from the current one ({', '.join(changed)}). "
                             "Restore the original command and files, or use a new --output-dir.")
    else:
        root.mkdir(parents=True, exist_ok=True)
        F._write_json(path, settings)
        F._write_json(root / "environment.json", collect_environment(args.blas_threads))
    return settings


def collect_draw(task, draw, root, source, settings, deadline):
    """Evaluate every set of one bank on one draw of B. Returns a summary; raises on failed matching."""

    root, source = Path(root), Path(source)
    label = f"{task['id']} · draw {draw}"
    start, written = time.monotonic(), 0
    try:
        with thread_limit(settings["blas_threads"]):
            src = source / task["id"]
            meta = F._read_json(src / "pair.json")
            sets = pd.read_csv(src / "sets.csv", usecols=["candidate_id", "seed_vertices"])
            if meta["sets_hash"] != task["sets_hash"] or len(sets) != task["n_sets"]:
                raise RuntimeError(f"Source bank changed after this design was saved: {task['id']}")
            if settings["sets_per_bank"]:
                sets = sets.head(settings["sets_per_bank"])
            path = runs_path(root, task, draw)
            path.parent.mkdir(parents=True, exist_ok=True)
            done = G._completed_ids(path, "run_id")  # also trims a row cut off by an interruption
            todo = [row for row in sets.itertuples(index=False) if run_id(row.candidate_id, draw) not in done]
            if not todo:
                return {"job": label, "runs": 0, "state": "complete"}
            check_time(deadline)
            with np.load(src / "pair.npz") as saved:
                A, B, truth, test = (saved[k] for k in ("A", "B", "truth", "test"))
            probability = L.b_probability(task["model"], task["mean_degree"], task["rho"], task["pair_seed"], A, B, truth)
            check_b, check_truth, check_rejected = BR.draw_b(A, probability, task["pair_seed"], CHECK_DRAW)
            regenerated = {"draw": CHECK_DRAW, "hash_b": RC._hash(check_b),
                           "truth_hash": C.digest(check_truth.tolist()), "rejected_draws": check_rejected}
            if regenerated != task["check_draw"]:
                raise RuntimeError(f"The B generator no longer reproduces the source's draw {CHECK_DRAW} "
                                   f"for {task['id']}; stop and check the graph code.")
            new_b, new_truth, rejected = BR.draw_b(A, probability, task["pair_seed"], draw)
            draw_info = {"draw": draw, "hash_b": RC._hash(new_b), "truth_hash": C.digest(new_truth.tolist()),
                         "rejected_draws": rejected}
            draw_path = path.parent / f"draw_{draw:02d}.json"
            if draw_path.exists():
                if F._read_json(draw_path) != draw_info:
                    raise RuntimeError(f"Draw {draw} of {task['id']} regenerated differently from its saved record.")
            else:
                F._write_json(draw_path, draw_info)
            a = task["algorithm"]
            regularizer = F.S.GOAT_REGULARIZER[task["model"]] if a == "goat" else None
            solver_seed = RC._seed(C.SEED_BASE, F._code(task["id"]), F.MP.ALGORITHM_CODES[a], draw, 0, 90)
            rows = G._Rows(path)
            try:
                for row in todo:
                    check_time(deadline)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        out = RM.run_one(a, task["model"], (A, new_b, new_truth), LS._seeds(row.seed_vertices),
                                         test, solver_seed, regularizer)
                    record = {"kind": "synthetic", "case": task["case"], "model": task["model"],
                              "mean_degree": task["mean_degree"], "rho": task["rho"], "budget": task["budget"],
                              "pair_index": task["pair_index"], "algorithm": a, "candidate_id": row.candidate_id,
                              "draw": draw, "repeat": 0, "run_id": run_id(row.candidate_id, draw),
                              "solver_seed": solver_seed, "regularizer": regularizer, "hash_b": draw_info["hash_b"],
                              **out, "warning_count": len(caught),
                              "warnings": json.dumps(sorted({str(w.message)[:1000] for w in caught}))}
                    if out["error"] or not np.isfinite(out["accuracy_test"]):
                        failures = G._Rows(path.parent / f"{a}_draw_{draw:02d}_errors.csv")
                        try:
                            failures.write(record)
                        finally:
                            failures.close()
                        raise RuntimeError(f"Matching failed at {label}, set {row.candidate_id}: {out['error']}")
                    rows.write(record)
                    written += 1
            finally:
                rows.close()
    except TimeoutError:
        return {"job": label, "runs": written, "state": "time limit"}
    return {"job": label, "runs": written, "state": "complete",
            "minutes": round((time.monotonic() - start) / 60, 2)}


def queue(settings, root, source):
    """Unfinished (bank, draw) jobs: draw by draw, slowest banks first within a draw."""

    seconds = {t["id"]: median_seconds(source, t) for t in settings["banks"]}
    order = sorted(settings["banks"], key=lambda t: (-np.nan_to_num(seconds[t["id"]] * t["n_sets"], nan=0.0), t["id"]))
    jobs = []
    for draw in settings["draws"]:
        for task in order:
            n = settings["sets_per_bank"] or task["n_sets"]
            if len(complete_ids(runs_path(root, task, draw))) < n:
                jobs.append((task, draw))
    return jobs, seconds


def remaining_seconds(jobs, root, settings, seconds):
    total = 0.0
    for task, draw in jobs:
        n = settings["sets_per_bank"] or task["n_sets"]
        left = max(0, n - len(complete_ids(runs_path(root, task, draw))))
        total += left * np.nan_to_num(seconds[task["id"]], nan=5.0)
    return total


def run_jobs(jobs, root, source, settings, deadline, workers, report):
    """One process pool over all jobs; a failed job is logged and the others continue."""

    failed, completed = 0, 0
    if not jobs:
        return 0

    def record_failure(task, draw, text):
        folder = Path(root) / task["id"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"collection_error_draw_{draw:02d}.txt").write_text(text)
        report(f"FAILED {task['id']} · draw {draw}\n{text}")

    if workers == 1:  # in-process: no pool start-up, and tests can replace the matcher
        for task, draw in jobs:
            if time.monotonic() >= deadline:
                break
            completed += 1
            try:
                report(f"[{completed}/{len(jobs)}] {collect_draw(task, draw, root, source, settings, deadline)}")
            except Exception:
                failed += 1
                record_failure(task, draw, traceback.format_exc())
        return failed

    pool = futures.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
    pending, iterator = {}, iter(jobs)

    def submit():
        if time.monotonic() >= deadline:
            return False
        job = next(iterator, None)
        if job is None:
            return False
        task, draw = job
        pending[pool.submit(collect_draw, task, draw, str(root), str(source), settings, deadline)] = job
        return True

    try:
        for _ in range(workers):
            submit()
        heartbeat = time.monotonic()
        while pending:
            ready, _ = futures.wait(pending, timeout=10, return_when=futures.FIRST_COMPLETED)
            for future in ready:
                task, draw = pending.pop(future)
                completed += 1
                try:
                    report(f"[{completed}/{len(jobs)}] {future.result()}")
                except Exception:
                    failed += 1
                    record_failure(task, draw, traceback.format_exc())
                submit()
            if time.monotonic() - heartbeat >= 300:
                left = max(0.0, (deadline - time.monotonic()) / 3600)
                report(f"{len(pending)} jobs active, {completed}/{len(jobs)} finished; "
                       f"time budget remaining {left:.2f} h")
                heartbeat = time.monotonic()
    except BaseException:
        processes = list((getattr(pool, "_processes", None) or {}).values())
        pool.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            process.terminate()
        raise
    else:
        pool.shutdown(wait=True)
    return failed


def status(root, source, smoke=False, last_draw=LAST_DRAW, blas_threads=2, workers=3):
    """Progress per bank; works before the first run (design read from the source, nothing written)."""

    root, source = Path(root), Path(source)
    settings = (F._read_json(root / "settings.json") if (root / "settings.json").exists()
                else design(root, source, smoke, last_draw, blas_threads))
    rows = []
    for task in settings["banks"]:
        n = settings["sets_per_bank"] or task["n_sets"]
        done = {d: len(complete_ids(runs_path(root, task, d))) for d in settings["draws"]}
        seconds = median_seconds(source, task)
        left = sum(max(0, n - v) for v in done.values())
        rows.append({"bank": task["id"].removeprefix("synthetic/").removesuffix("/pair_000"),
                     "sets": n, "draws_complete": FIRST_DRAW + sum(v >= n for v in done.values()),
                     "draws_planned": FIRST_DRAW + len(settings["draws"]),
                     "runs_done": sum(min(v, n) for v in done.values()), "runs_planned": n * len(settings["draws"]),
                     "median_s_per_run": round(seconds, 2) if np.isfinite(seconds) else np.nan,
                     "hours_left_1_worker": round(left * np.nan_to_num(seconds, nan=5.0) / 3600, 2)})
    table = pd.DataFrame(rows)
    hours = table.hours_left_1_worker.sum()
    summary = (f"{int(table.runs_done.sum())} of {int(table.runs_planned.sum())} runs done; "
               f"about {hours / max(1, workers):.1f} h of solver time left on {workers} worker"
               f"{'s' if workers != 1 else ''} ({hours:.1f} h on one; wall time runs somewhat longer).")
    return table, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, help=f"default: {DEFAULT_DIR}")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR, help="finished score-range collection (read only)")
    parser.add_argument("--hours", type=float, default=18, help="wall-clock budget for this invocation; "
                        "runs in progress finish, then the workers stop")
    parser.add_argument("--jobs", type=int, default=3, help="parallel workers")
    parser.add_argument("--blas-threads", type=int, default=2, help="BLAS threads per worker (part of the design)")
    parser.add_argument("--last-draw", type=int, default=LAST_DRAW, help="last draw of B to collect (part of the design)")
    parser.add_argument("--status", action="store_true", help="print progress and the estimated time left")
    parser.add_argument("--smoke", action="store_true", help=f"one small bank per algorithm, draw {FIRST_DRAW}, "
                        f"{SMOKE_SETS} sets each, in {SMOKE_DIR.relative_to(C.REPO_ROOT)}")
    args = parser.parse_args(argv)
    if args.hours <= 0 or args.jobs < 1 or args.blas_threads < 1:
        parser.error("hours, workers and BLAS threads must be positive")
    root = (args.output_dir or (SMOKE_DIR if args.smoke else DEFAULT_DIR)).resolve()
    source = args.source_dir.resolve()
    if root == source or source in root.parents:
        parser.error("The output folder must be outside the source collection.")
    if args.status:
        table, summary = status(root, source, args.smoke, args.last_draw, args.blas_threads, args.jobs)
        print(table.to_string(index=False))
        print(summary)
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / "collection.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"Another collection is using {root}")
        deadline = time.monotonic() + args.hours * 3600
        settings = prepare(root, source, args)
        jobs, seconds = queue(settings, root, source)
        with (root / "collection.log").open("a", buffering=1, encoding="utf-8") as log:
            def report(message):
                line = f"{F._now()}  {message}"
                print(line, flush=True)
                log.write(line + "\n")

            left = remaining_seconds(jobs, root, settings, seconds) / 3600
            report(f"Extra draws {settings['draws'][0]}-{settings['draws'][-1]} for {len(settings['banks'])} banks; "
                   f"{len(jobs)} bank-draw jobs left; about {left / args.jobs:.1f} h of solver time on "
                   f"{args.jobs} worker{'s' if args.jobs != 1 else ''} ({left:.1f} h on one; wall time runs "
                   f"somewhat longer); limit {args.hours:g} h. Output: {root}")
            try:
                failed = run_jobs(jobs, root, source, settings, deadline, args.jobs, report)
            except KeyboardInterrupt:
                report("Stopped by Ctrl-C. Finished runs are saved; rerun the same command to resume.")
                return
            table, summary = status(root, source, args.smoke, args.last_draw, args.blas_threads, args.jobs)
            report(f"Finished this invocation; failed jobs: {failed}. {summary}")
            report(f"Status: python -m selection_rule_comparison.score_range_draws"
                   f"{' --smoke' if args.smoke else ''} --status"
                   + (f" --output-dir {root}" if args.output_dir else ""))
            if failed:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
