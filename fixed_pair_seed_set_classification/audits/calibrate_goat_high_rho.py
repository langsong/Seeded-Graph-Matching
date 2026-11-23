"""GOAT budget calibration on the existing Graph A with increased correlation.

Reuses the project generator and verifies Graph A and the truth permutation
against the October 13 collection. Each phase uses fresh seed sets; saved
permutations, environment, source hashes, and graph pairs support reproduction.
Existing outcome files are appended, never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache"))

import numpy as np
import pandas as pd

from experiment_common.graph_checks import check_graph_pair
from experiment_common.graphs import ExperimentConfig, generate_graph_pair
from experiment_common.provenance import collect_environment, thread_limit, write_json
from fixed_pair_seed_set_classification.audits.calibrate_faq_goat import evaluate

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/sgm_fixed_pairs_2026-10-13"
SETTINGS = {
    "sparse_er": {"seed": 88424077, "rho": 0.70, "reg": 100, "id": 0},
    "paper_pa": {"seed": 7301002, "rho": 0.62, "reg": 300, "id": 1},
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_pair(model, output):
    settings = SETTINGS[model]
    source_dir = SOURCE / "graph_pairs" / model / f"pair_seed_{settings['seed']}"
    source_npz = source_dir / f"{model}_problem.npz"
    source_json = source_dir / f"{model}_problem.metadata.json"
    original = json.loads(source_json.read_text())
    config = replace(ExperimentConfig(**original["config"]), rho=settings["rho"])
    A, B, truth = generate_graph_pair(config, settings["seed"])
    with np.load(source_npz) as saved:
        assert np.array_equal(A, saved["adjacency_graph1"]), "Graph A changed"
        assert np.array_equal(truth, saved["true_permutation"]), "Truth changed"
    checks = check_graph_pair(A, B)
    assert checks["passes"], checks
    upper = np.triu_indices(len(A), 1)
    aligned_B = B[np.ix_(truth, truth)]
    metadata = {
        "model": model, "graph_pair_seed": settings["seed"],
        "config": asdict(config), "structure": checks,
        "source_pair": str(source_npz), "source_pair_sha256": digest(source_npz),
        "same_graph_A": True, "same_truth": True,
        "generation": "Same generator and pair RNG seed; change only rho. No rejected draws.",
        "retained_edge_fraction": float((A * aligned_B).sum() / A.sum()),
        "observed_edge_correlation": float(np.corrcoef(A[upper], aligned_B[upper])[0, 1]),
        "solver": "goat", "regularizer": settings["reg"],
    }
    directory = output / "pairs" / model
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{model}_problem.npz"
    if target.exists():
        with np.load(target) as saved:
            for key, array in zip(
                ["adjacency_graph1", "adjacency_graph2", "true_permutation"],
                [A, B, truth],
            ):
                assert np.array_equal(saved[key], array), f"Saved pair differs: {key}"
    else:
        np.savez_compressed(target, adjacency_graph1=A, adjacency_graph2=B,
                            true_permutation=truth)
        write_json(target.with_suffix(".metadata.json"), metadata)
    return A, B, truth


def summarize(output):
    data = pd.read_csv(output / "runs.csv")
    keys = ["model", "phase", "rho", "k"]
    table = data.groupby(keys).agg(
        sets=("success", "size"), successes=("success", "sum"),
        success_rate=("success", "mean"), mean_accuracy=("accuracy", "mean"),
        converged_fraction=("converged", "mean"),
        warning_runs=("warnings", lambda values: sum(v != "[]" for v in values)),
        mean_seconds=("seconds", "mean"),
    ).reset_index()
    z = 1.96
    n, p = table["sets"], table["success_rate"]
    center = (p + z*z/(2*n))/(1 + z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1 + z*z/n)
    table["wilson_low"] = center - half
    table["wilson_high"] = center + half
    table.to_csv(output / "summary.csv", index=False)
    print(table.round(4).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    parser.add_argument("--budgets", nargs="+", type=int, required=True)
    parser.add_argument("--sets", type=int, default=20)
    parser.add_argument("--phase", choices=["screen", "refine", "confirm"], default="screen")
    args = parser.parse_args()
    assert args.sets > 0 and all(0 <= k < 600 for k in args.budgets)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "runs.csv"
    existing = []
    if path.exists():
        with path.open() as stream:
            existing = list(csv.DictReader(stream))
    done = {(r["model"], r["phase"], int(r["k"]), int(r["candidate_id"])) for r in existing}
    phase_id = ["screen", "refine", "confirm"].index(args.phase)
    with thread_limit(4):
        env_path = args.output / "environment.json"
        environment = collect_environment(4)
        environment["solver_settings"] = {
            "n_init": 1, "max_iter": 100, "tol": 0.01,
            "init": "barycenter", "init_perturbation": 0, "shuffle_input": True,
            "transport": True, "transport_tol": 0.05, "transport_max_iter": 5000,
        }
        environment["scripts_sha256"] = {
            str(p): digest(p) for p in [
                Path(__file__), Path(__file__).with_name("calibrate_faq_goat.py"),
                ROOT.parent / "utils/Graphs.py",
            ]
        }
        if env_path.exists():
            previous = json.loads(env_path.read_text())
            for key in ["package_versions", "blas_threads_limit", "solver_settings", "scripts_sha256"]:
                assert environment[key] == previous[key], f"Environment changed: {key}"
        else:
            write_json(env_path, environment)
        for model in args.models:
            A, B, truth = prepare_pair(model, args.output)
            settings = SETTINGS[model]
            # Exclude every set from earlier phases and the original collection.
            with (SOURCE / "goat" / model / f"pair_1_seed_{settings['seed']}" /
                  f"{model}_seed_sets.csv").open() as stream:
                original_rows = list(csv.DictReader(stream))
            original_seed_key = next(
                (key for key in ["seed_vertices_graph1", "seed_vertices", "seeds", "seed_set", "seed_indices"]
                 if key in original_rows[0]), None
            )
            assert original_seed_key is not None, "Cannot locate original seed sets"
            original_sets = {tuple(sorted(json.loads(r[original_seed_key]))) for r in original_rows}
            for k in args.budgets:
                occupied = original_sets | {
                    tuple(json.loads(r["seed_vertices"])) for r in existing
                    if r["model"] == model and r["phase"] != args.phase
                }
                rng = np.random.default_rng(np.random.SeedSequence(
                    [20261020, settings["id"], phase_id, k]))
                target_count = 1 if k == 0 else args.sets
                if k == 0:
                    occupied.discard(())
                successes = 0
                for candidate_id in range(target_count):
                    for _ in range(100000):
                        seeds = np.sort(rng.choice(len(A), size=k, replace=False))
                        seed_tuple = tuple(seeds.tolist())
                        if seed_tuple not in occupied:
                            occupied.add(seed_tuple)
                            break
                    else:
                        raise RuntimeError("Could not draw a fresh seed set")
                    if (model, args.phase, k, candidate_id) in done:
                        continue
                    solver_rng = int(np.random.default_rng(np.random.SeedSequence(
                        [20261020, settings["id"], phase_id, k, candidate_id, 99]
                    )).integers(2**32 - 1))
                    result = evaluate(A, B, truth, seeds, solver_rng, "goat",
                                      settings["reg"], transport_max_iter=5000)
                    row = {
                        "model": model, "phase": args.phase, "rho": settings["rho"],
                        "k": k, "candidate_id": candidate_id, "rng_seed": solver_rng,
                        "regularizer": settings["reg"], **result,
                    }
                    with path.open("a", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        if stream.tell() == 0:
                            writer.writeheader()
                        writer.writerow(row)
                    successes += result["success"]
                    if (candidate_id + 1) % 10 == 0 or candidate_id + 1 == target_count:
                        print(f"{model} {args.phase} k={k}: {candidate_id+1}/{target_count}, "
                              f"{successes} successes in new runs", flush=True)
                summarize(args.output)


if __name__ == "__main__":
    main()
