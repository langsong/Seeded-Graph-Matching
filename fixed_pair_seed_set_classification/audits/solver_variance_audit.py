"""Repeat SGM on 200 historical ``paper`` (Pareto–Chung–Lu) seed sets.

Run from the repository root with:
    python -m fixed_pair_seed_set_classification.audits.solver_variance_audit
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_common.sgm import run_sgm_once


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
AUDIT_DIR = DATA_DIR / "solver_variance"
SOURCE_CSV = DATA_DIR / "paper_seed_sets.csv"
PROBLEM_NPZ = DATA_DIR / "paper_problem.npz"
OUTPUT_CSV = AUDIT_DIR / "paper_solver_variance_runs.csv"
METADATA_JSON = AUDIT_DIR / "paper_solver_variance.metadata.json"
SELECTION_SEED = 314159
SOLVER_SEEDS = [2026, 74993, 419567, 739391, 982451]
SETS_PER_STRATEGY = 100
MAX_ITER = 30
TOL = 0.01


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE_CSV)
    selected = pd.concat(
        [
            source.loc[
                (source["data_split"] == "train")
                & (source["sampling_strategy"] == strategy)
            ].sample(n=SETS_PER_STRATEGY, random_state=SELECTION_SEED)
            for strategy in ("random", "feature_diverse")
        ],
        ignore_index=True,
    ).sort_values("candidate_id")

    with np.load(PROBLEM_NPZ) as saved:
        adjacency_1 = saved["adjacency_graph1"]
        adjacency_2 = saved["adjacency_graph2"]
        truth = saved["true_permutation"]

    metadata = {
        "source_csv": str(SOURCE_CSV),
        "problem_npz": str(PROBLEM_NPZ),
        "output_csv": str(OUTPUT_CSV),
        "selection_seed": SELECTION_SEED,
        "candidate_ids": selected["candidate_id"].astype(int).tolist(),
        "solver_seeds": SOLVER_SEEDS,
        "sets_per_strategy": SETS_PER_STRATEGY,
        "max_iter": MAX_ITER,
        "tol": TOL,
        "success_threshold": 0.90,
        "note": "The first solver seed reproduces the original saved outcome.",
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2) + "\n")

    completed = set()
    if OUTPUT_CSV.exists():
        previous = pd.read_csv(OUTPUT_CSV)
        completed = set(
            zip(previous["candidate_id"].astype(int), previous["solver_seed"].astype(int))
        )

    fieldnames = [
        "candidate_id",
        "sampling_strategy",
        "solver_seed",
        "h0_accuracy",
        "final_unseeded_accuracy",
        "final_all_accuracy",
        "sgm_runtime_seconds",
        "sgm_n_iter",
        "sgm_converged",
        "sgm_objective_score",
        "sgm_success",
    ]
    write_header = not OUTPUT_CSV.exists()
    with OUTPUT_CSV.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for index, original in enumerate(selected.itertuples(index=False), 1):
            seeds_1 = np.asarray(json.loads(original.seed_vertices_graph1), dtype=int)
            for solver_seed in SOLVER_SEEDS:
                if (original.candidate_id, solver_seed) in completed:
                    continue
                outcome = run_sgm_once(
                    adjacency_1,
                    adjacency_2,
                    truth,
                    seeds_1,
                    solver_seed,
                    MAX_ITER,
                    TOL,
                )
                writer.writerow(
                    {
                        "candidate_id": original.candidate_id,
                        "sampling_strategy": original.sampling_strategy,
                        "solver_seed": solver_seed,
                        **outcome,
                        "sgm_success": outcome["final_unseeded_accuracy"] >= 0.90,
                    }
                )
            file.flush()
            if index % 10 == 0:
                print(f"Evaluated {index}/{len(selected)} seed sets", flush=True)
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
