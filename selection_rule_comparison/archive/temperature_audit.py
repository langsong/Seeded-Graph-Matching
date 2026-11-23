"""Paired GOAT regularization audit; used by temperature_audit.ipynb.

The installed transport routine is called unchanged. Instrumentation records
its gradient scale, transport entropy and numerical diagnostics. Runs are
sequential because graspologic's private hooks are temporarily instrumented.
Each completed call is saved atomically, including its full final permutation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import tempfile
import time
import warnings
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache"))

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from graspologic.match import graph_match
import graspologic.match.solver as solver_module
import graspologic.match.wrappers as wrappers

from experiment_common.provenance import collect_environment, thread_limit, write_json
from . import hub_test as H
from .. import run_comparison as rc
from .. import settings as S
from ..strategies import GraphA

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "temperature_audit"
RULES = ("top_degree", "hub_free_top_degree")
LABELS = {"top_degree": "Top degree", "hub_free_top_degree": "Hub-free top degree"}
MODEL_LABELS = {"chung_lu": "Chung–Lu", "paper_pa": "PAPER", "sparse_er": "ER"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _matrix_sha(matrix):
    matrix = np.ascontiguousarray(matrix)
    return hashlib.sha256(str((matrix.shape, matrix.dtype.str)).encode() + matrix.tobytes()).hexdigest()


def _entropy(matrix):
    """Mean row entropy / log(number of columns); 0=sharp, 1=uniform."""
    matrix = np.asarray(matrix, dtype=float)
    if not np.isfinite(matrix).all() or np.any(matrix < -1e-12):
        return float("nan")
    positive = np.maximum(matrix, 0)
    sums = positive.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        return float("nan")
    p = positive / sums
    logs = np.zeros_like(p)
    np.log(p, out=logs, where=p > 0)
    return float(-(p * logs).sum(axis=1).mean() / np.log(max(2, p.shape[1])))


def _marginal_error(matrix):
    if not np.isfinite(matrix).all():
        return float("inf")
    return float(max(np.max(np.abs(matrix.sum(axis=0) - 1)),
                     np.max(np.abs(matrix.sum(axis=1) - 1))))


def run_once(A, B, truth, seeds, solver_seed, regularizer, *, max_iter=100,
             tol=0.01, transport_tol=0.05, transport_max_iter=5000):
    """Unmodified GOAT with diagnostic hooks, always restored on exit.

    Does not consume random numbers or replace the Sinkhorn calculation.
    Truth is used only for supplied seeds and retrospective accuracy diagnostics.
    """
    seeds = np.sort(np.asarray(seeds, dtype=int))
    truth = np.asarray(truth, dtype=int)
    n = len(A)
    if A.shape != B.shape or A.shape != (n, n) or not 0 < len(seeds) < n:
        raise ValueError("Expected equal square graphs and a nonempty, partial seed set.")
    if len(np.unique(seeds)) != len(seeds) or np.any((seeds < 0) | (seeds >= n)):
        raise ValueError("Seeds must be distinct valid vertices.")
    if not np.array_equal(np.sort(truth), np.arange(n)):
        raise ValueError("Truth must be a permutation.")
    if regularizer <= 0 or not np.isfinite(regularizer):
        raise ValueError("Regularizer must be positive and finite.")

    cls = wrappers._GraphMatchSolver
    original_direction = cls.compute_step_direction
    original_sinkhorn = solver_module.sinkhorn
    trace, captured, inner = [], {}, {}

    def measured_sinkhorn(*args, **kwargs):
        result = original_sinkhorn(*args, **kwargs)
        if kwargs.get("log"):
            log = result[1]
            errors = log.get("err", [])
            inner.update(sinkhorn_n_iter=int(log.get("niter", -1)),
                         sinkhorn_reported_error=float(errors[-1]) if len(errors) else float("nan"))
        return result

    def measured_direction(solver, gradient, rng):
        inner.clear()
        Q = original_direction(solver, gradient, rng)
        abs_gradient = np.abs(gradient)
        maximum = float(np.max(abs_gradient))
        # Diagnostic reductions do not affect the solver's matrices or RNG.
        row = {"iteration": int(solver.n_iter_), "gradient_max_abs": maximum,
               "epsilon_magnitude": maximum / regularizer,
               "transport_entropy": _entropy(Q),
               "transport_marginal_error": _marginal_error(Q),
               "transport_finite": bool(np.isfinite(Q).all()), **inner}
        if not trace:
            p95 = float(np.percentile(abs_gradient, 95))
            row.update(gradient_p95_abs=p95,
                       gradient_p95_over_max=p95 / maximum if maximum else float("nan"),
                       seed_term_norm_over_gradient_norm=float(np.linalg.norm(solver.constant_sum) /
                                                               max(np.linalg.norm(gradient), 1e-300)))
            captured.update(first_Q=Q.copy(), perm_A=solver.perm_A.copy(), perm_B=solver.perm_B.copy())
        trace.append(row)
        return Q

    cls.compute_step_direction = measured_direction
    solver_module.sinkhorn = measured_sinkhorn
    started = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = graph_match(
                A, B, partial_match=np.column_stack((seeds, truth[seeds])),
                n_init=1, n_jobs=1, init=None, init_perturbation=0.0,
                shuffle_input=True, max_iter=max_iter, tol=tol, rng=int(solver_seed),
                transport=True, transport_regularizer=float(regularizer),
                transport_tol=transport_tol, transport_max_iter=transport_max_iter)
    finally:
        cls.compute_step_direction = original_direction
        solver_module.sinkhorn = original_sinkhorn
    runtime = time.perf_counter() - started
    if not trace:
        raise RuntimeError("No transport directions were observed.")
    permutation = np.full(n, -1, dtype=int)
    permutation[result.indices_A] = result.indices_B
    if not np.array_equal(np.sort(permutation), np.arange(n)):
        raise RuntimeError("Final matching is not bijective.")
    if not np.array_equal(permutation[seeds], truth[seeds]):
        raise RuntimeError("Final matching does not preserve the seed correspondences.")
    reconstructed_score = float(np.sum(A * B[np.ix_(permutation, permutation)]))
    if not np.isclose(reconstructed_score, result.score):
        raise RuntimeError("Reconstructed objective disagrees with the solver.")
    soft = np.asarray(result.misc[0]["convex_solution"])
    finite = bool(np.isfinite(soft).all() and all(t["transport_finite"] for t in trace))
    if not finite:
        raise RuntimeError("Nonfinite soft solution or transport direction.")
    nonseeds = np.setdiff1d(np.arange(n), seeds)
    accuracy = float(np.mean(permutation[nonseeds] == truth[nonseeds]))
    rows, columns = linear_sum_assignment(captured["first_Q"], maximize=True)
    unseed_A = captured["perm_A"][len(seeds):]
    unseed_B = captured["perm_B"][len(seeds):]
    first_accuracy = float(np.mean(unseed_B[columns] == truth[unseed_A[rows]]))
    first, last = trace[0], trace[-1]
    max_error = max(t["transport_marginal_error"] for t in trace)
    inner_limit = any(t.get("sinkhorn_n_iter", -1) >= transport_max_iter - 1 for t in trace)
    converged = bool(result.misc[0]["converged"])
    summary = {
        "status": "ok", "accuracy": accuracy, "success": bool(accuracy >= 0.90),
        "objective": float(result.score), "first_direction_accuracy": first_accuracy,
        "runtime_seconds": runtime, "n_iter": int(result.misc[0]["n_iter"]),
        "converged": converged, "warning_count": len(caught),
        "warning_messages": [str(w.message) for w in caught], "finite_soft": finite,
        "first_gradient_max_abs": first["gradient_max_abs"],
        "first_gradient_p95_abs": first["gradient_p95_abs"],
        "first_gradient_p95_over_max": first["gradient_p95_over_max"],
        "first_seed_term_norm_fraction": first["seed_term_norm_over_gradient_norm"],
        "first_epsilon_magnitude": first["epsilon_magnitude"],
        "last_gradient_max_abs": last["gradient_max_abs"],
        "first_transport_entropy": first["transport_entropy"],
        "last_transport_entropy": last["transport_entropy"],
        "final_soft_entropy": _entropy(soft),
        "max_transport_marginal_error": max_error, "final_soft_marginal_error": _marginal_error(soft),
        "inner_limit_reached": inner_limit,
        "numerically_clean": bool(converged and not inner_limit and len(caught) == 0 and
                                  max_error <= transport_tol and _marginal_error(soft) <= transport_tol),
    }
    return summary, trace, permutation


def _reference(model, pair_index, rule, budget, seeds):
    if rule == "top_degree":
        path = H.ROUND1_ROOT / "goat" / model / "runs.csv"
        if not path.exists():
            return None
        d = pd.read_csv(path)
        d = d[(d.pair_index == pair_index) & (d.strategy == rule) & (d.budget == budget)]
        column = "final_unseeded_accuracy"
    else:
        path = H.OUTPUT_DIR / model / f"pair_{pair_index:02d}.csv"
        if not path.exists():
            return None
        d = pd.read_csv(path)
        d = d[(d.rule == rule) & (d.budget == budget)]
        column = "accuracy"
    if d.empty:
        return None
    row = d.iloc[0]
    if sorted(json.loads(row.seed_vertices)) != sorted(seeds):
        raise ValueError("Reference run used a different seed set.")
    return {"accuracy": float(row[column]), "objective": float(row.objective), "path": str(path)}


def run_audit(*, models=("chung_lu", "paper_pa"), pair_indices=tuple(range(5)),
              budgets=(8, 10), regularizers=(100, 300, 600), output_dir=DEFAULT_OUTPUT,
              blas_threads=3, max_iter=100, tol=0.01, transport_tol=0.05,
              transport_max_iter=5000, retry_errors=False):
    """Run/resume the requested cells. Use a new folder for changed scientific settings.

    Pair indices may be extended without changing the manifest. Seed sets and
    pair archives are frozen before varying regularization. No stopping on success.
    """
    models, pair_indices = tuple(models), tuple(int(i) for i in pair_indices)
    budgets = tuple(sorted(set(int(k) for k in budgets)))
    regularizers = tuple(sorted(set(float(r) for r in regularizers)))
    if not models or not pair_indices or not budgets or not regularizers:
        raise ValueError("Select at least one model, pair, budget, and regularizer.")
    if len(set(pair_indices)) != len(pair_indices) or min(pair_indices) < 0:
        raise ValueError("Pair indices must be distinct and nonnegative.")
    if any(m not in S.MODELS for m in models) or len(set(models)) != len(models):
        raise ValueError("Models must be distinct supported graph models.")
    if any(not np.isfinite(r) or r <= 0 for r in regularizers):
        raise ValueError("Regularizers must be positive and finite.")
    for model in models:
        if min(budgets) < 1 or max(budgets) > S.graph_config("goat", model).n_vertices - S.HUB_COUNT:
            raise ValueError("Invalid seed budget.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), Path(H.__file__), Path(rc.__file__), Path(S.__file__),
                    Path(solver_module.__file__), ROOT / "selection_rule_comparison" / "strategies.py"]
    manifest = {
        "version": 1, "models": list(models), "budgets": list(budgets), "regularizers": list(regularizers),
        "rules": list(RULES), "blas_threads": blas_threads,
        "solver": {"max_iter": max_iter, "tol": tol, "transport_tol": transport_tol,
                   "transport_max_iter": transport_max_iter},
        "graph_configs": {m: asdict(S.graph_config("goat", m)) for m in models},
        "seed_streams": {"strategy": S.STRATEGY_SEED_BASE, "solver": S.SOLVER_SEED_BASE},
        "source_sha256": {str(p): _sha(p) for p in source_paths},
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "graspologic", "POT")},
        "installed_transport_source": inspect.getsource(wrappers._GraphMatchSolver.linear_sum_transport),
    }
    manifest = json.loads(json.dumps(manifest))
    settings_path = output_dir / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != manifest:
            raise ValueError("Audit settings/source versions changed. Choose a new OUTPUT_DIR; existing results are preserved.")
    else:
        write_json(settings_path, manifest)
        write_json(output_dir / "environment.json", collect_environment(blas_threads))
    total = len(models) * len(pair_indices) * len(budgets) * len(RULES) * len(regularizers)
    print(f"Requested {total} GOAT runs; completed runs will be reused.", flush=True)
    completed, new, errors, started = 0, 0, 0, time.perf_counter()
    with thread_limit(blas_threads):
        for model in models:
            for pair_index in pair_indices:
                pair_dir = output_dir / model / f"pair_{pair_index:02d}"
                pair_dir.mkdir(parents=True, exist_ok=True)
                seed, A, B, truth = H.load_pair(model, pair_index, verify=True)
                graph = GraphA(A)
                hubs, td, hf = H.orderings(graph, seed, max(budgets))
                orders = dict(top_degree=td, hub_free_top_degree=hf)
                metadata = {"model": model, "pair_index": pair_index, "pair_seed": seed,
                            "hash_A": _matrix_sha(A), "hash_B": _matrix_sha(B), "hash_truth": _matrix_sha(truth),
                            "hubs": sorted(hubs), "seed_sets": {rule: {str(k): sorted(map(int, order[:k]))
                                                                      for k in budgets} for rule, order in orders.items()}}
                metadata_path = pair_dir / "pair.json"
                if metadata_path.exists():
                    if json.loads(metadata_path.read_text()) != metadata:
                        raise ValueError("Frozen pair or seed sets changed.")
                    with np.load(pair_dir / "pair.npz", allow_pickle=False) as frozen:
                        if not all(np.array_equal(frozen[k], v) for k, v in {"A": A, "B": B, "truth": truth}.items()):
                            raise ValueError("Frozen pair archive does not match regenerated graphs.")
                else:
                    np.savez_compressed(pair_dir / "pair.npz", A=A, B=B, truth=truth)
                    write_json(metadata_path, metadata)
                for k in budgets:
                    for rule in RULES:
                        seeds = metadata["seed_sets"][rule][str(k)]
                        for regularizer in regularizers:
                            path = pair_dir / f"k{k:02d}_{rule}_reg{regularizer:g}.json"
                            if path.exists():
                                saved = json.loads(path.read_text())
                                if saved["summary"]["status"] == "ok" or not retry_errors:
                                    completed += 1
                                    errors += int(saved["summary"]["status"] != "ok")
                                    continue
                            identifiers = {"model": model, "pair_index": pair_index, "pair_seed": seed,
                                           "budget": k, "rule": rule, "regularizer": regularizer,
                                           "solver_seed": rc.solver_seed(seed, k, 0), "seed_vertices": seeds}
                            try:
                                summary, trace, permutation = run_once(
                                    A, B, truth, seeds, identifiers["solver_seed"], regularizer,
                                    max_iter=max_iter, tol=tol, transport_tol=transport_tol,
                                    transport_max_iter=transport_max_iter)
                                if regularizer == S.GOAT_REGULARIZER[model]:
                                    reference = _reference(model, pair_index, rule, k, seeds)
                                    if reference is not None:
                                        summary.update(reference_accuracy=reference["accuracy"],
                                                       reference_accuracy_match=bool(np.isclose(summary["accuracy"], reference["accuracy"])),
                                                       reference_objective_match=bool(np.isclose(summary["objective"], reference["objective"])))
                                payload = {"summary": {**identifiers, **summary}, "trace": trace,
                                           "final_permutation": permutation.tolist()}
                            except Exception as error:
                                errors += 1
                                payload = {"summary": {**identifiers, "status": "error", "error": repr(error)},
                                           "trace": [], "final_permutation": None}
                                print(f"Run error ({model}, pair {pair_index}, k={k}, {rule}, r={regularizer:g}): {error}", flush=True)
                            write_json(path, payload)
                            completed += 1
                            new += 1
                elapsed = time.perf_counter() - started
                remaining = total - completed
                eta = elapsed / new * remaining if new else 0
                print(f"{model}, pair {pair_index}: {completed}/{total} cells saved; "
                      f"elapsed {elapsed/60:.1f} min; about {eta/60:.1f} min remaining", flush=True)
    write_json(output_dir / "status.json", {"requested_runs": total, "cells_saved": completed,
                                            "errors": errors, "pair_indices": list(pair_indices)})
    results, traces = load_results(output_dir, models=models, pair_indices=pair_indices)
    results.to_csv(output_dir / "runs.csv", index=False)
    traces.to_csv(output_dir / "iteration_trace.csv", index=False)
    print(f"Audit finished: {total-errors}/{total} successful calls, {errors} errors. Saved to {output_dir}", flush=True)
    return results, traces


def load_results(output_dir=DEFAULT_OUTPUT, *, models=None, pair_indices=None):
    summaries, traces = [], []
    for path in sorted(Path(output_dir).glob("*/pair_*/k*_reg*.json")):
        payload = json.loads(path.read_text())
        row = payload["summary"]
        if models is not None and row["model"] not in models:
            continue
        if pair_indices is not None and row["pair_index"] not in pair_indices:
            continue
        summaries.append(row)
        ids = {k: row[k] for k in ("model", "pair_index", "budget", "rule", "regularizer")}
        traces.extend({**ids, **t} for t in payload["trace"])
    if not summaries:
        raise FileNotFoundError("No audit results found. Run the experiment cell first.")
    return pd.DataFrame(summaries), pd.DataFrame(traces)
