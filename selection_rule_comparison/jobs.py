"""Run collection tasks in parallel, with progress lines, failure reports and Ctrl-C handling.

Moved unchanged from sbm_rules.py (now archive/sbm_rules.py) on 23 Nov 2026. Used by b_replicates,
constructed_redraws, fresh_pair_scores, fresh_pair_extremes and landscape_extremes.
"""

from __future__ import annotations

import datetime
import sys
import traceback


def _now() -> str:
    return datetime.datetime.now().astimezone().strftime("%a %H:%M")


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
