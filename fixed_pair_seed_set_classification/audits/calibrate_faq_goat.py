"""Small, resumable sweeps using the public graspologic FAQ/GOAT API.

Pairs are filtered only for connectivity and a discrete color refinement.
Seed sets are nested across budgets and shared across solver comparisons.
The primary run of a candidate has an independent reproducible RNG seed.
This script writes a new calibration directory and does not update presets.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
import warnings
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_mpl_cache"))

import numpy as np
import pandas as pd
from graspologic.match import graph_match

from experiment_common.graph_checks import check_graph_pair
from experiment_common.graphs import generate_graph_pair, preset_config
from experiment_common.provenance import collect_environment, thread_limit, write_json


SETTINGS = {
    "sparse_er": {
        "rho": 0.50, "er_probability": 16 / 599,
        "pair_seeds": [4192889773, 12672911, 88424077],
        "faq": [29, 31, 33], "goat": [13, 15, 17], "reg": 100,
    },
    "paper_pa": {
        "rho": 0.40, "paper_pa_probability": 0.023378,
        "paper_pa_alpha": 0.0, "paper_pa_beta": 1.0,
        "pair_seeds": [424242, 7301001, 7301002],
        "faq": [27, 29, 31], "goat": [8, 10, 12], "reg": 300,
    },
    "chung_lu": {
        "rho": 0.45, "chung_lu_alpha": 2.0,
        "chung_lu_weight_scale": 5.452064, "chung_lu_probability_floor": 0.0,
        "pair_seeds": [3165569293, 18472911, 37051993],
        "faq": [23, 25, 27], "goat": [7, 9, 11], "reg": 300,
    },
}
MODEL_IDS = {name: i for i, name in enumerate(SETTINGS)}
FIELDS = [
    "model", "solver", "pair_id", "pair_seed", "candidate_id", "k", "run_id",
    "rng_seed", "regularizer", "accuracy", "success", "objective", "n_iter",
    "converged", "seconds", "warnings", "finite_soft_solution",
    "max_row_sum_error", "max_column_sum_error", "soft_distance_from_uniform",
    "mean_soft_row_entropy", "no_seed_neighbor_fraction_A", "seed_vertices",
    "final_permutation",
]


def run_key(row):
    return (row["model"], row["solver"], *(int(row[key]) for key in ["pair_id", "candidate_id", "k", "run_id"]))


def load_or_generate_pairs(model, output):
    settings = SETTINGS[model]
    config = replace(
        preset_config(model), vertex_count=600, max_iter=100, tol=0.01,
        **{key: value for key, value in settings.items() if key not in {"pair_seeds", "faq", "goat", "reg"}},
    )
    pairs = []
    pair_dir = output / "pairs"
    pair_dir.mkdir(exist_ok=True)
    for pair_id, first_seed in enumerate(settings["pair_seeds"]):
        path = pair_dir / f"{model}_pair_{pair_id}.npz"
        metadata_path = path.with_suffix(".json")
        if path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata["config"] != json.loads(json.dumps(asdict(config))):
                raise ValueError(f"Configuration mismatch: {path}")
            with np.load(path) as saved:
                A, B, truth = (saved[key] for key in ["A", "B", "truth"])
        else:
            for rejected in range(200):
                pair_seed = first_seed + rejected
                A, B, truth = generate_graph_pair(config, pair_seed)
                structure = check_graph_pair(A, B)
                if structure["passes"]:
                    break
            else:
                raise RuntimeError("No structurally valid pair in 200 tries")
            aligned_B = B[np.ix_(truth, truth)]
            upper = np.triu_indices(len(A), k=1)
            metadata = {
                "model": model, "pair_id": pair_id, "pair_seed": pair_seed,
                "config": asdict(config), "rejected_draws": rejected,
                "structure": structure,
                "max_degree_A": int(A.sum(axis=1).max()),
                "max_degree_B": int(B.sum(axis=1).max()),
                "retained_edge_fraction": float((A * aligned_B).sum() / A.sum()),
                "observed_edge_correlation": float(np.corrcoef(A[upper], aligned_B[upper])[0, 1]),
            }
            np.savez_compressed(path, A=A, B=B, truth=truth)
            write_json(metadata_path, metadata)
        pairs.append((A, B, truth, metadata))
    return pairs


def evaluate(A, B, truth, seeds, rng_seed, solver, regularizer, *, max_iter=100, transport_max_iter=1000):
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = graph_match(
            A, B, partial_match=np.column_stack((seeds, truth[seeds])),
            init=None, init_perturbation=0.0, n_init=1, n_jobs=1,
            shuffle_input=True, max_iter=max_iter, tol=0.01, rng=int(rng_seed),
            transport=solver == "goat", transport_regularizer=regularizer,
            transport_tol=0.05, transport_max_iter=transport_max_iter,
        )
    prediction = np.empty(len(A), dtype=int)
    prediction[result.indices_A] = result.indices_B
    unseeded = np.setdiff1d(np.arange(len(A)), seeds)
    accuracy = float(np.mean(prediction[unseeded] == truth[unseeded]))
    misc = result.misc[0]
    P = misc["convex_solution"]
    entropy = -(np.where(P > 0, P, 0) * np.log(np.clip(P, 1e-300, None))).sum(axis=1).mean()
    return {
        "accuracy": accuracy, "success": accuracy >= 0.9,
        "objective": float(result.score), "n_iter": int(misc["n_iter"]),
        "converged": bool(misc["converged"]),
        "seconds": time.perf_counter() - started,
        "warnings": json.dumps([str(w.message) for w in caught]),
        "finite_soft_solution": bool(np.all(np.isfinite(P))),
        "max_row_sum_error": float(np.max(np.abs(P.sum(axis=1) - 1))),
        "max_column_sum_error": float(np.max(np.abs(P.sum(axis=0) - 1))),
        "soft_distance_from_uniform": float(np.linalg.norm(P - 1 / len(P)) / np.sqrt(len(P))),
        "mean_soft_row_entropy": float(entropy),
        "no_seed_neighbor_fraction_A": float(np.mean(A[np.ix_(unseeded, seeds)].sum(axis=1) == 0)),
        "seed_vertices": json.dumps(seeds.tolist()),
        "final_permutation": json.dumps(prediction.tolist()),
    }


def summarize(output):
    data = pd.read_csv(output / "runs.csv")
    primary = data[data.run_id == 0]
    keys = ["model", "solver", "k"]
    pooled = primary.groupby(keys).agg(
        sets=("success", "size"), successes=("success", "sum"),
        success_rate=("success", "mean"), mean_accuracy=("accuracy", "mean"),
        converged_fraction=("converged", "mean"), mean_seconds=("seconds", "mean"),
    )
    pair = primary.groupby(keys + ["pair_id"]).agg(
        sets=("success", "size"), successes=("success", "sum"),
        success_rate=("success", "mean"), mean_accuracy=("accuracy", "mean"),
    )
    pooled.to_csv(output / "pooled_summary.csv")
    pair.to_csv(output / "pair_summary.csv")
    print(pooled.round(3).to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["screen", "confirm", "audit"], default="screen")
    parser.add_argument("--plan", type=Path, help="JSON {model: {faq: [k], goat: [k]}}")
    parser.add_argument("--sets", type=int, default=4)
    parser.add_argument("--run-ids", nargs="+", type=int, default=[0])
    parser.add_argument("--pair-ids", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--models", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--transport-max-iter", type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text()) if args.plan else SETTINGS
    csv_path = args.output / "runs.csv"
    completed = set()
    if csv_path.exists():
        with csv_path.open() as f:
            completed = {run_key(row) for row in csv.DictReader(f)}

    started = time.perf_counter()
    n_new = 0
    with thread_limit(args.threads):
        metadata_path = args.output / "environment.json"
        if metadata_path.exists():
            saved = json.loads(metadata_path.read_text())
            if saved["blas_threads_limit"] != args.threads:
                raise ValueError("Do not mix thread counts within one calibration")
            if saved.get("transport_max_iter", 1000) != args.transport_max_iter:
                raise ValueError("Use a separate output directory for a different transport iteration limit")
        else:
            environment = collect_environment(args.threads)
            environment["transport_max_iter"] = args.transport_max_iter
            write_json(metadata_path, environment)
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if f.tell() == 0:
                writer.writeheader()
            for model in args.models:
                model_id = MODEL_IDS[model]
                pairs = load_or_generate_pairs(model, args.output)
                for pair_id in args.pair_ids:
                    A, B, truth, pair_metadata = pairs[pair_id]
                    permutations = [np.random.default_rng(np.random.SeedSequence([20261013, model_id, pair_id, i])).permutation(600) for i in range(args.sets)]
                    for solver in ["faq", "goat"]:
                        for k in plan[model][solver]:
                            cell_started = time.perf_counter()
                            cell_new = 0
                            cell_successes = 0
                            for candidate_id, permutation in enumerate(permutations):
                                seeds = np.sort(permutation[:k])
                                for run_id in args.run_ids:
                                    base = {
                                        "model": model, "solver": solver, "pair_id": pair_id,
                                        "pair_seed": pair_metadata["pair_seed"], "candidate_id": candidate_id,
                                        "k": k, "run_id": run_id,
                                    }
                                    if run_key(base) in completed:
                                        continue
                                    rng_seed = int(np.random.default_rng(np.random.SeedSequence([20261013, model_id, pair_id, candidate_id, run_id, 99])).integers(2**32 - 1))
                                    regularizer = SETTINGS[model]["reg"]
                                    result = evaluate(
                                        A, B, truth, seeds, rng_seed, solver, regularizer,
                                        transport_max_iter=args.transport_max_iter,
                                    )
                                    writer.writerow({**base, "rng_seed": rng_seed, "regularizer": regularizer if solver == "goat" else "", **result})
                                    f.flush()
                                    completed.add(run_key(base))
                                    cell_new += 1
                                    n_new += 1
                                    cell_successes += int(result["success"])
                            if cell_new:
                                print(f"{args.phase}: {model} pair {pair_id} {solver} k={k}: {cell_successes}/{cell_new} new successes; {time.perf_counter()-cell_started:.1f}s", flush=True)
    print(f"Saved {n_new} new runs in {time.perf_counter()-started:.1f}s", flush=True)
    summarize(args.output)


if __name__ == "__main__":
    main()
