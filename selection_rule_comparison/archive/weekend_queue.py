"""Weekend queue (7-9 Nov 2026): the long runs for the Mac mini, one step after another, for up to 35 hours.

Steps, in order. Each is an ordinary command (the queue only runs them in turn), so any step can also be run or rerun
by hand. Every step resumes where it stopped and skips what is already finished.

     1  first_look        real pairs: both worms and two Enron week pairs                         about 0.5 h
     2  landscape_er      match-ratio landscape, ER                                                about 5.5 h
     3  map_cl_stage1     regime map, Chung-Lu, stage 1 (random-set ladders; the landscape budgets) about 1 h
     4  map_paper_stage1  regime map, PAPER, stage 1                                               about 1 h
     5  landscape_cl      landscape, Chung-Lu                                                      about 5.5 h
     6  landscape_paper   landscape, PAPER                                                         about 5.5 h
     7  map_cl_stage2     regime map, Chung-Lu, stage 2 (rules and best-of-N at the 20/50/80% budgets) about 1.5 h
     8  map_paper_stage2  regime map, PAPER, stage 2                                               about 1.5 h
     9  fillers_levels    landscape, ER, the same cells at the 20% and 80% levels (1 pair, no extras) about 2.5 h
    10  fillers_cells     landscape, ER, mean degree 16 and rho 0.2, all three algorithms          about 1.5 h
    11  first_look_wiki   real pairs: English vs French Wikipedia (1,382 vertices, slow)           about 3 h
    12  more_pairs        landscape, a 4th and then a 5th pair per cell on all three models        until the deadline

The hours are guesses from the run times of the ER regime map (6-7 Nov) and assume 3 jobs. At the deadline (--hours,
default 35) the running step is stopped cleanly (finished runs are kept) and the rest are skipped. A failed step is
logged and the queue goes on with the next one.

Before leaving, run the smoke test (about 10 minutes; it writes only to tmp/weekend_smoke/ and the logs):

    python -m selection_rule_comparison.weekend_queue --smoke

Then start the queue in a Terminal window (closing the window does not stop it; Ctrl-C or --stop does):

    python -m selection_rule_comparison.weekend_queue
    python -m selection_rule_comparison.weekend_queue --status       # from another window: how far it is
    python -m selection_rule_comparison.weekend_queue --stop         # stop it cleanly
    python -m selection_rule_comparison.weekend_queue --steps landscape_cl landscape_paper --hours 12
    python -m selection_rule_comparison.weekend_queue --dry-run      # only print the plan and the commands

Logs: selection_rule_comparison/logs/weekend/<start time>/ with queue.log (the queue's own lines), state.json and
one log per step with everything the step printed.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]  # selection_rule_comparison/ (this file is in archive/)
REPO_ROOT = EXPERIMENT_DIR.parent
LOG_ROOT = EXPERIMENT_DIR / "logs" / "weekend"
SMOKE_DIR = REPO_ROOT / "tmp" / "weekend_smoke"
DEFAULT_HOURS = 35.0
GRACE_SECONDS = 180       # after a stop request, time a step gets to stop cleanly before it is terminated
OUTPUT_VARIABLES = ("REGIME_MAP_OUTPUT_DIR", "LANDSCAPE_OUTPUT_DIR", "FIRST_LOOK_OUTPUT_DIR")

LANDSCAPE = "selection_rule_comparison.landscape"
MAP = "selection_rule_comparison.regime_map"
FIRST_LOOK = "real_data_experiments.first_look"
QUEUE = "selection_rule_comparison.archive.weekend_queue"


@dataclass(frozen=True)
class Step:
    name: str
    description: str
    commands: tuple           # ((module, (arguments, ...)), ...), run one after another
    hours: float              # a rough guess, for the plan only
    env: tuple = ()           # ((variable, value), ...) set for this step only
    parallel: bool = True     # pass --jobs and --blas-threads on


STEPS = (
    Step("first_look", "Real pairs: both worms and two Enron week pairs, all four solver variants",
         ((FIRST_LOOK, ("--pairs", "small")),), 0.5),
    Step("landscape_er", "Match-ratio landscape, ER: 9 cells x 3 pairs, about 480 seed sets per pair",
         ((LANDSCAPE, ("--model", "sparse_er")),), 5.5),
    Step("map_cl_stage1", "Regime map, Chung-Lu, stage 1: random-set ladders (they set the landscape budgets)",
         ((MAP, ("--stage", "1", "--model", "chung_lu")),), 1.0),
    Step("map_paper_stage1", "Regime map, PAPER, stage 1",
         ((MAP, ("--stage", "1", "--model", "paper_pa")),), 1.0),
    Step("landscape_cl", "Match-ratio landscape, Chung-Lu",
         ((LANDSCAPE, ("--model", "chung_lu")),), 5.5),
    Step("landscape_paper", "Match-ratio landscape, PAPER",
         ((LANDSCAPE, ("--model", "paper_pa")),), 5.5),
    Step("map_cl_stage2", "Regime map, Chung-Lu, stage 2: rules and best-of-N vs random sets at 20/50/80%",
         ((MAP, ("--stage", "2", "--model", "chung_lu")),), 1.5),
    Step("map_paper_stage2", "Regime map, PAPER, stage 2",
         ((MAP, ("--stage", "2", "--model", "paper_pa")),), 1.5),
    Step("fillers_levels", "Landscape, ER: the same cells at the 20% and 80% levels (1 pair per cell, no extras)",
         ((LANDSCAPE, ("--model", "sparse_er", "--levels", "0.2", "0.8", "--pairs", "1", "--no-extras")),), 2.5),
    Step("fillers_cells", "Landscape, ER: mean degree 16, rho 0.2 for all three algorithms (300 seeds: random sets "
         "stay below 50% in the map's range)",
         ((LANDSCAPE, ("--model", "sparse_er", "--cells", "percolation:16:0.2", "faq:16:0.2", "goat:16:0.2")),), 1.5),
    Step("first_look_wiki", "Real pairs: English vs French Wikipedia (1,382 vertices, slow)",
         ((FIRST_LOOK, ("--pairs", "wikipedia")),), 3.0),
    Step("more_pairs", "Landscape: a 4th pair per cell on ER, Chung-Lu and PAPER, then a 5th, until the deadline",
         tuple((LANDSCAPE, ("--model", model, "--pairs", str(n))) for n in (4, 5)
               for model in ("sparse_er", "chung_lu", "paper_pa")), 6.0),
)


def smoke_steps(folder: Path) -> tuple:
    maps, landscape, first_look = str(folder / "regime_map"), str(folder / "landscape"), str(folder / "first_look")
    cell = ("--algorithms", "percolation", "--degrees", "16", "--rhos", "0.5")
    return (
        Step("check_pairs", "Every cell of the Chung-Lu and PAPER maps and landscapes can draw a pair (no runs)",
             ((QUEUE, ("--check-pairs",)),), 0.02, parallel=False),
        Step("map_cl", "Regime map stages 1 and 2, Chung-Lu, one cell, percolation only",
             ((MAP, ("--stage", "1", "--model", "chung_lu", *cell)),
              (MAP, ("--stage", "2", "--model", "chung_lu", *cell, "--pairs", "1"))), 0.03,
             env=(("REGIME_MAP_OUTPUT_DIR", maps),)),
        Step("map_paper", "Regime map stages 1 and 2, PAPER, one cell, percolation only",
             ((MAP, ("--stage", "1", "--model", "paper_pa", *cell)),
              (MAP, ("--stage", "2", "--model", "paper_pa", *cell, "--pairs", "1"))), 0.03,
             env=(("REGIME_MAP_OUTPUT_DIR", maps),)),
        Step("landscape_er", "Landscape, ER: a tiny design in each algorithm's first cell (budgets from your ER "
             "map), then the same command again, which must find nothing left to run",
             ((LANDSCAPE, ("--model", "sparse_er", "--smoke")), (LANDSCAPE, ("--model", "sparse_er", "--smoke"))),
             0.07, env=(("LANDSCAPE_OUTPUT_DIR", landscape),)),
        Step("first_look", "Real pairs: worm and Enron (one budget, a few sets) and one Wikipedia set per variant",
             ((FIRST_LOOK, ("--smoke",)),), 0.05, env=(("FIRST_LOOK_OUTPUT_DIR", first_look),)),
    )


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _clock(when: float | None = None) -> str:
    return datetime.datetime.fromtimestamp(time.time() if when is None else when).strftime("%a %H:%M")


def _iso(when: float) -> str:
    return datetime.datetime.fromtimestamp(when).astimezone().isoformat(timespec="seconds")


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


def _write_json(path: Path, content: dict) -> None:
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _command_text(module: str, arguments) -> str:
    return " ".join(["python -m", module, *arguments])


def _alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def select_steps(names) -> tuple:
    if not names:
        return STEPS
    chosen = []
    for name in names:
        if name.isdigit() and 1 <= int(name) <= len(STEPS):
            chosen.append(STEPS[int(name) - 1])
        else:
            matches = [s for s in STEPS if s.name == name]
            if not matches:
                raise SystemExit(f"Unknown step {name!r}. Steps: {', '.join(s.name for s in STEPS)}")
            chosen.append(matches[0])
    return tuple(dict.fromkeys(chosen))


def plan_text(steps, start: float, deadline: float | None) -> str:
    lines, clock = [], start
    for i, step in enumerate(steps, 1):
        clock += step.hours * 3600
        lines.append(f"  {i:2d}  {step.name:17s} {step.hours:4.1f} h   ends about {_clock(clock)}   {step.description}")
    if deadline is not None:
        lines.append(f"  deadline {_clock(deadline)} ({_iso(deadline)}); the hours are rough guesses")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Running one command, with the deadline and Ctrl-C
# --------------------------------------------------------------------------

def _pump(stream, log) -> None:
    for line in iter(stream.readline, ""):
        try:
            log.write(line)
            log.flush()
        except (OSError, ValueError):   # the log was closed: the step is over
            return
        try:
            sys.stdout.write(line)
            sys.stdout.flush()
        except (OSError, ValueError):   # the terminal window was closed: keep logging
            pass


def _kill_group(process: subprocess.Popen) -> None:
    """Terminate a step with its workers (each step runs in a process group of its own)."""

    for sig, wait in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            process.wait(wait)
            return
        except subprocess.TimeoutExpired:
            continue


def _stop(process: subprocess.Popen) -> None:
    """Ask a step to stop like Ctrl-C would (it keeps finished runs), then terminate it if it doesn't."""

    try:
        process.send_signal(signal.SIGINT)
        process.wait(GRACE_SECONDS)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        _kill_group(process)
    except ProcessLookupError:
        pass


def run_command(argv: list, env: dict, log, deadline: float | None) -> tuple[int, str | None]:
    """Run one command, copying its output to the log and the terminal. Returns (exit code, why it was stopped)."""

    process = subprocess.Popen(argv, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, start_new_session=True)
    reader = threading.Thread(target=_pump, args=(process.stdout, log), daemon=True)
    reader.start()
    reason = None
    try:
        while process.poll() is None:
            if deadline is not None and time.time() >= deadline:
                reason = "deadline"
                _stop(process)
                break
            time.sleep(1)
        process.wait()
    except KeyboardInterrupt:          # Ctrl-C, --stop or kill: pass it on to the step
        reason = "stopped"
        _stop(process)
        try:
            process.wait(30)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            _kill_group(process)
    reader.join(timeout=10)
    return process.returncode, reason


# --------------------------------------------------------------------------
# The queue
# --------------------------------------------------------------------------

def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


def run_queue(steps, hours: float | None, jobs: int, blas_threads: int, label: str = "",
              until: float | None = None, smoke_dir: Path | None = None) -> int:
    start = time.time()
    deadline = until if until is not None else (start + hours * 3600 if hours else None)
    stamp = datetime.datetime.fromtimestamp(start).strftime("%Y-%m-%d_%H%M%S")
    folder = LOG_ROOT / f"{label}{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    queue_log = (folder / "queue.log").open("a", encoding="utf-8")

    def note(message: str) -> None:
        for line in message.splitlines() or [""]:
            text = f"{_clock()}  {line}"
            queue_log.write(text + "\n")
            queue_log.flush()
            try:
                print(text, flush=True)
            except (OSError, ValueError):
                pass

    state = {"pid": os.getpid(), "label": label.rstrip("_") or "queue", "started_at": _iso(start),
             "deadline": _iso(deadline) if deadline else None, "log_dir": str(folder),
             "steps": [{"name": s.name, "description": s.description, "hours_guess": s.hours, "status": "waiting",
                        "commands": [_command_text(m, a) for m, a in s.commands]} for s in steps]}
    state_path = folder / "state.json"
    _write_json(state_path, state)

    signal.signal(signal.SIGTERM, _raise_interrupt)     # --stop and kill: stop cleanly, like Ctrl-C
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)    # closing the Terminal window doesn't stop the queue
    keep_awake = None
    if sys.platform == "darwin":
        import shutil
        if shutil.which("caffeinate"):
            keep_awake = subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])

    note(f"Weekend queue{' (smoke test)' if label else ''}: {len(steps)} steps, logs in {folder}")
    note(plan_text(steps, start, deadline))
    for variable in OUTPUT_VARIABLES:
        if variable in os.environ and not label:
            note(f"Note: {variable} is set to {os.environ[variable]}; results of the matching steps go there.")
    interrupted, exit_code = False, 0
    for i, (step, entry) in enumerate(zip(steps, state["steps"]), 1):
        if interrupted or (deadline is not None and time.time() >= deadline):
            entry["status"] = "skipped (stopped)" if interrupted else "skipped (deadline)"
            continue
        env = {**os.environ, **dict(step.env), "PYTHONUNBUFFERED": "1"}
        log_path = folder / f"{i:02d}_{step.name}.log"
        entry.update({"status": "running", "started_at": _iso(time.time()), "log": str(log_path)})
        _write_json(state_path, state)
        note(f"Step {i}/{len(steps)} {step.name}: {step.description}")
        codes, reason, step_start = [], None, time.time()
        with log_path.open("a", encoding="utf-8") as log:
            for module, arguments in step.commands:
                arguments = list(arguments)
                if step.parallel:
                    arguments += ["--jobs", str(jobs), "--blas-threads", str(blas_threads)]
                argv = [sys.executable, "-u", "-m", module, *arguments]
                log.write(f"\n$ {_command_text(module, arguments)}\n")
                log.flush()
                note(f"  $ {_command_text(module, arguments)}")
                try:
                    code, reason = run_command(argv, env, log, deadline)
                except KeyboardInterrupt:
                    code, reason = 130, "stopped"
                codes.append(code)
                if reason is not None:
                    break
        took = time.time() - step_start
        if reason == "stopped":
            interrupted = True
            entry["status"] = "stopped (Ctrl-C or --stop)"
        elif reason == "deadline":
            entry["status"] = "stopped at the deadline"
        elif all(c == 0 for c in codes):
            entry["status"] = "done"
        else:
            entry["status"] = f"failed (exit {', '.join(str(c) for c in codes if c != 0)}); see the log"
            exit_code = 1
        entry.update({"finished_at": _iso(time.time()), "minutes": round(took / 60, 1), "exit_codes": codes})
        _write_json(state_path, state)
        note(f"Step {i}/{len(steps)} {step.name}: {entry['status']} after {_hours(took)}")
    state["finished_at"] = _iso(time.time())
    _write_json(state_path, state)
    note("Summary:")
    for entry in state["steps"]:
        took = f"{entry['minutes']:.0f} min" if "minutes" in entry else ""
        note(f"  {entry['name']:17s} {entry['status']:32s} {took}")
    if smoke_dir is not None and not interrupted:
        try:
            note(smoke_report(smoke_dir, jobs))
        except Exception as error:  # noqa: BLE001 - the report is a convenience
            note(f"(No timing report: {type(error).__name__}: {error})")
    if keep_awake is not None:
        keep_awake.terminate()
    queue_log.close()
    return 130 if interrupted else exit_code


# --------------------------------------------------------------------------
# Smoke test: pair checks and the timing report
# --------------------------------------------------------------------------

def check_pairs(models=("chung_lu", "paper_pa")) -> int:
    """Draw pair 0 of every cell of the regime-map grid and the first landscape pair of every landscape cell (with one
    redraw of B), for each model. No matching runs. Returns the number of cells that failed."""

    import pandas as pd

    from .. import landscape as L
    from .. import regime_map as RM

    pd.set_option("display.width", 200)
    failures = 0
    for model in models:
        rows = []
        for degree in RM.DEGREES[model]:
            for rho in RM.RHOS:
                started = time.perf_counter()
                try:
                    _, _, _, _, _, _, info = RM.load_pair(model, degree, rho, 0, {"excluded_pair_seeds": [],
                                                                                 "n_test": RM.N_TEST})
                    result = f"ok ({info['rejected_draws']} rejected)"
                except Exception as error:  # noqa: BLE001
                    result, failures = f"FAILED: {error}"[:110], failures + 1
                rows.append({"mean degree": degree, "rho": rho, "pair 0": result,
                             "seconds": round(time.perf_counter() - started, 1)})
        print(f"\n{RM.MODEL_LABELS[model]}: regime map grid", flush=True)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
        table = L.check_pairs(model)
        failures += int(table["pair"].astype(str).str.startswith("FAILED").sum())
        print(f"\n{RM.MODEL_LABELS[model]}: landscape cells", flush=True)
        print(table.to_string(index=False), flush=True)
    print(f"\n{'All cells can draw pairs.' if not failures else f'{failures} cells FAILED (see above).'}", flush=True)
    return failures


def smoke_report(folder: Path, jobs: int) -> str:
    """Run times measured in the smoke test, and what they mean for the long steps."""

    import pandas as pd

    from real_data_experiments import first_look as F

    from .. import landscape as L
    from .. import regime_map as RM

    def gather(paths, keys):
        frames = []
        for path in paths:
            if path.stat().st_size == 0:
                continue
            frame = pd.read_csv(path)
            if "runtime_seconds" in frame:
                frames.append(frame[[c for c in (*keys, "runtime_seconds", "accuracy_test", "error") if c in frame]])
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    lines = ["Smoke-test run times (median seconds per run, from the runs above):"]
    errors = []
    landscape = gather([p for p in (folder / "landscape").glob("**/*.csv")
                        if p.name in ("runs.csv", "redraws.csv", "regularizer.csv")], ("model", "algorithm"))
    maps = gather(list((folder / "regime_map").glob("**/runs.csv")), ("model", "algorithm"))
    looks = gather(list((folder / "first_look").glob("*/*/k*.csv")), ("pair", "variant"))
    for name, frame, keys in (("landscape", landscape, ["model", "algorithm"]), ("regime map", maps,
                                                                                   ["model", "algorithm"]),
                              ("first look", looks, ["pair", "variant"])):
        if frame.empty:
            lines.append(f"  {name}: no runs found")
            continue
        table = frame.groupby(keys)["runtime_seconds"].agg(["size", "median"])
        lines.append(f"  {name}:")
        lines += [f"    {' / '.join(map(str, key))}: {row['median']:.2f} s ({int(row['size'])} runs)"
                  for key, row in table.iterrows()]
        bad = frame[frame["error"].fillna("").astype(str) != ""]
        errors += [f"    {name}: {' / '.join(str(r[k]) for k in keys)}: {r['error']}"
                   for _, r in bad.head(5).iterrows()]
        missing = int(frame["accuracy_test"].isna().sum()) if "accuracy_test" in frame else 0
        if missing:
            errors.append(f"    {name}: {missing} runs without a test accuracy")

    if not landscape.empty:
        medians = landscape.groupby("algorithm")["runtime_seconds"].median()
        design = L._fixed("sparse_er", False)
        total = 0.0
        for algorithm in L.ALGORITHMS:
            if algorithm not in medians:
                continue
            kinds = ["sets"] * L.PAIRS + ["redraws"] + (["regularizer"] if algorithm == "goat" else [])
            runs = len(L.CELLS[algorithm]) * sum(
                L._planned(("sparse_er", algorithm, 0, 0, 0.5, 0, kind, 0), design) for kind in kinds)
            total += runs * float(medians[algorithm])
        lines.append(f"  => one landscape model (the 9 standard cells): about {_hours(total / jobs)} with {jobs} "
                     f"jobs, if the other cells run as fast as these (the queue assumes 5.5 h).")
    if not looks.empty:
        design = F._fixed(False)
        wiki = looks[looks["pair"].isin(F.LARGE)].groupby("variant")["runtime_seconds"].median()
        total = sum(F._planned((pair, variant, k), design) * float(wiki[variant])
                    for pair in F.LARGE for variant in wiki.index for k in design["pairs"][pair]["budgets"])
        if len(wiki):
            lines.append(f"  => Wikipedia (step first_look_wiki): about {_hours(total / jobs)} with {jobs} jobs "
                         "(the queue assumes 3 h; it is the second-to-last step).")
    lines.append(f"  The regime map assumed {RM.SECONDS_PER_RUN} seconds per run.")
    if errors:
        lines.append("Runs that recorded an error (look at these before leaving):")
        lines += errors
    else:
        lines.append("No run recorded an error.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Status and stop
# --------------------------------------------------------------------------

def _latest(smoke: bool = False) -> Path | None:
    if not LOG_ROOT.exists():
        return None
    folders = sorted(p for p in LOG_ROOT.iterdir() if (p / "state.json").exists()
                     and p.name.startswith("smoke_") == smoke)
    return folders[-1] if folders else None


def _running(folder: Path) -> bool:
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    return "finished_at" not in state and _alive(state["pid"]) and int(state["pid"]) != os.getpid()


def status(smoke: bool = False, tail: int = 4) -> str:
    folder = _latest(smoke)
    if folder is None:
        return f"No {'smoke test' if smoke else 'queue'} has run yet (logs go to {LOG_ROOT})."
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    running = _alive(state["pid"]) and "finished_at" not in state
    lines = [f"{'Smoke test' if smoke else 'Queue'} started {state['started_at']}, "
             f"{'running (pid ' + str(state['pid']) + ')' if running else 'not running'}; deadline "
             f"{state.get('deadline') or 'none'}; logs in {folder}"]
    for i, entry in enumerate(state["steps"], 1):
        took = f"{entry['minutes']:.0f} min" if "minutes" in entry else (
            f"since {entry['started_at'][11:16]}" if entry.get("started_at") else "")
        lines.append(f"  {i:2d}  {entry['name']:17s} {entry['status']:32s} {took}")
        if entry["status"] == "running" and entry.get("log") and Path(entry["log"]).exists():
            text = Path(entry["log"]).read_text(encoding="utf-8", errors="replace").splitlines()
            lines += [f"        | {line[:160]}" for line in text[-tail:]]
    if not running and "finished_at" not in state:
        lines.append("  The queue stopped without finishing (machine restarted or process killed?). Start it again: "
                     "finished work is skipped.")
    return "\n".join(lines)


def stop() -> str:
    folder = _latest(False)
    if folder is None:
        return "No queue has run yet."
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    if not _alive(state["pid"]) or "finished_at" in state:
        return "The queue is not running."
    os.kill(int(state["pid"]), signal.SIGTERM)
    return (f"Asked the queue (pid {state['pid']}) to stop. The running step stops within a few minutes and keeps its "
            "finished runs; rerun the queue later to resume.")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", help="run only these steps (names or numbers), in this order")
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS, help="stop after this many hours "
                        f"(default {DEFAULT_HOURS:g}; 0 for no deadline)")
    parser.add_argument("--until", help="stop at this local time instead, e.g. '2026-11-09 18:00'")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--blas-threads", type=int, default=3)
    parser.add_argument("--smoke", action="store_true", help="run the smoke test (tiny versions of the steps, "
                        f"results in {SMOKE_DIR.relative_to(REPO_ROOT)}/) and report run times")
    parser.add_argument("--status", action="store_true", help="show how far the latest queue (or smoke test) is")
    parser.add_argument("--stop", action="store_true", help="stop the running queue cleanly")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and commands, run nothing")
    parser.add_argument("--check-pairs", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.check_pairs:
        raise SystemExit(1 if check_pairs() else 0)
    if args.status:
        print(status(args.smoke))
        return
    if args.stop:
        print(stop())
        return
    until = None
    if args.until:
        until = datetime.datetime.strptime(args.until, "%Y-%m-%d %H:%M").timestamp()
    running = [f for f in (_latest(False), _latest(True)) if f is not None and _running(f)]
    if running and not args.dry_run:
        raise SystemExit(f"A queue or smoke test is already running (logs in {running[0]}). Use --status, or --stop "
                         "to stop it first.")
    if args.smoke:
        steps = smoke_steps(SMOKE_DIR)
        if args.dry_run:
            print(plan_text(steps, time.time(), None))
            return
        import shutil
        if SMOKE_DIR.exists():
            if not (SMOKE_DIR / ".weekend_smoke").exists():
                raise SystemExit(f"{SMOKE_DIR} exists but was not made by this smoke test; move it away first.")
            shutil.rmtree(SMOKE_DIR)
        SMOKE_DIR.mkdir(parents=True)
        (SMOKE_DIR / ".weekend_smoke").write_text("Made by selection_rule_comparison.weekend_queue --smoke; "
                                                  "safe to delete.\n")
        raise SystemExit(run_queue(steps, None, args.jobs, args.blas_threads, label="smoke_", smoke_dir=SMOKE_DIR))
    steps = select_steps(args.steps)
    if args.dry_run:
        start = time.time()
        deadline = until or (start + args.hours * 3600 if args.hours else None)
        print(plan_text(steps, start, deadline))
        for step in steps:
            for module, arguments in step.commands:
                extra = ["--jobs", str(args.jobs), "--blas-threads", str(args.blas_threads)] if step.parallel else []
                print(f"  {step.name}: {_command_text(module, [*arguments, *extra])}")
        return
    raise SystemExit(run_queue(steps, args.hours, args.jobs, args.blas_threads, until=until))


if __name__ == "__main__":
    main()
