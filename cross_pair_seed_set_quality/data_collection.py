"""Collect seed-set features and SGM performance on correlated graph pairs.

The candidate-level features are computed from graph 1 only.  Graph-level
features summarize both graphs without using the hidden vertex correspondence.
Each SGM run records the first-direction accuracy (H0), final unseeded matching
accuracy, iteration count, convergence status, objective value, and runtime.

The first Frank-Wolfe direction is captured from the same solver run used for
the final matching, so measuring H0 does not require a second SGM run.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from experiment_common.features import (
    FEATURE_SCHEMA_VERSION,
    GraphContext,
    build_graph_context,
    graph_pair_features,
    seed_set_features,
)
from experiment_common.graphs import (
    GRAPH_MODELS,
    ExperimentConfig,
    generate_graph_pair,
    preset_config,
    validate_config,
)
from experiment_common.sampling import UINT32_MAX, sample_unique_seed_sets
from experiment_common.sgm import run_sgm_once

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "data"

OUTPUT_FILENAMES = {
    "sparse_er": "seed_quality_sparse_er.csv",
    "sparse_sbm": "seed_quality_sparse_sbm.csv",
    "paper": "seed_quality_paper.csv",
    "ier": "seed_quality_ier.csv",
    "paper_pa": "seed_quality_paper_pa.csv",
}

SGM_RESULT_COLUMNS = (
    "h0_accuracy",
    "final_unseeded_accuracy",
    "final_all_accuracy",
    "sgm_runtime_seconds",
    "sgm_n_iter",
    "sgm_converged",
    "sgm_objective_score",
)


def row_context(
    config: ExperimentConfig,
    *,
    graph_pair_key: str,
    graph_pair_id: int,
    candidate_id: int,
    graph_pair_seed: int,
    sgm_seed: int,
    seeds_1: np.ndarray,
    seeds_2: np.ndarray,
) -> dict[str, Any]:
    """Build the identifiers and graph-generation settings saved on each row."""

    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "graph_model": config.graph_model,
        "graph_pair_key": graph_pair_key,
        "graph_pair_id": graph_pair_id,
        "candidate_id": candidate_id,
        "graph_pair_seed": graph_pair_seed,
        "sgm_seed": sgm_seed,
        "seed_vertices_graph1": json.dumps(seeds_1.tolist()),
        "seed_vertices_graph2": json.dumps(seeds_2.tolist()),
        "n_vertices": config.n_vertices,
        "n_seeds": config.n_seeds,
        "rho": config.rho,
        "er_probability": (
            config.er_probability if config.graph_model == "sparse_er" else None
        ),
        "sbm_n_blocks": (
            config.n_blocks if config.graph_model == "sparse_sbm" else None
        ),
        "sbm_n_per_block": (
            config.n_per_block if config.graph_model == "sparse_sbm" else None
        ),
        "sbm_block_probabilities": (
            json.dumps(config.sbm_block_probabilities)
            if config.graph_model == "sparse_sbm"
            else None
        ),
        "paper_alpha": (
            config.paper_alpha if config.graph_model == "paper" else None
        ),
        "paper_probability": (
            config.paper_probability if config.graph_model == "paper" else None
        ),
        "paper_pa_alpha": (
            config.paper_pa_alpha if config.graph_model == "paper_pa" else None
        ),
        "paper_pa_beta": (
            config.paper_pa_beta if config.graph_model == "paper_pa" else None
        ),
        "paper_pa_probability": (
            config.paper_pa_probability if config.graph_model == "paper_pa" else None
        ),
        "ier_alpha": config.ier_alpha if config.graph_model == "ier" else None,
        "ier_beta": config.ier_beta if config.graph_model == "ier" else None,
    }


def refresh_collected_features(output_path: Path) -> dict[str, Any]:
    """Rebuild a collected CSV's features while preserving saved SGM results."""

    output_path = output_path.expanduser().resolve()
    metadata_path = output_path.with_suffix(".metadata.json")
    for required_path in (output_path, metadata_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Could not find {required_path}.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = ExperimentConfig(**metadata["config"])
    validate_config(config)

    import pandas as pd

    existing_frame = pd.read_csv(output_path)
    required_columns = {
        "graph_pair_key",
        "graph_pair_id",
        "candidate_id",
        "graph_pair_seed",
        "sgm_seed",
        "seed_vertices_graph1",
        "seed_vertices_graph2",
    } | set(SGM_RESULT_COLUMNS)
    missing_columns = required_columns - set(existing_frame.columns)
    if missing_columns:
        raise ValueError(
            f"Cannot refresh {output_path}; missing columns: "
            f"{sorted(missing_columns)}"
        )
    if existing_frame.empty:
        raise ValueError(f"Cannot refresh empty dataset {output_path}.")

    refreshed_rows: list[dict[str, Any]] = []
    grouped = existing_frame.groupby("graph_pair_key", sort=False)
    for pair_number, (pair_key, group) in enumerate(grouped, start=1):
        pair_seeds = group["graph_pair_seed"].unique()
        if len(pair_seeds) != 1:
            raise ValueError(f"Graph pair {pair_key} has multiple generation seeds.")
        graph_pair_seed = int(pair_seeds[0])
        adjacency_1, adjacency_2, true_permutation = generate_graph_pair(
            config, graph_pair_seed
        )
        context_1 = build_graph_context(adjacency_1)
        context_2 = build_graph_context(adjacency_2, compute_seed_features=False)
        pair_features = graph_pair_features(context_1, context_2)

        for _, saved_row in group.iterrows():
            seeds_1 = np.asarray(
                json.loads(saved_row["seed_vertices_graph1"]), dtype=int
            )
            seeds_2 = np.asarray(
                json.loads(saved_row["seed_vertices_graph2"]), dtype=int
            )
            if len(seeds_1) != config.n_seeds or len(np.unique(seeds_1)) != len(
                seeds_1
            ):
                raise ValueError(
                    f"Invalid Graph A seed set in {pair_key}, candidate "
                    f"{saved_row['candidate_id']}."
                )
            if not np.array_equal(seeds_2, true_permutation[seeds_1]):
                raise ValueError(
                    f"Regenerated correspondence does not match {pair_key}, "
                    f"candidate {saved_row['candidate_id']}."
                )

            refreshed_row = row_context(
                config,
                graph_pair_key=str(saved_row["graph_pair_key"]),
                graph_pair_id=int(saved_row["graph_pair_id"]),
                candidate_id=int(saved_row["candidate_id"]),
                graph_pair_seed=graph_pair_seed,
                sgm_seed=int(saved_row["sgm_seed"]),
                seeds_1=seeds_1,
                seeds_2=seeds_2,
            )
            refreshed_row.update(pair_features)
            refreshed_row.update(seed_set_features(context_1, seeds_1))
            refreshed_row.update(
                {column: saved_row[column] for column in SGM_RESULT_COLUMNS}
            )
            refreshed_rows.append(refreshed_row)

        print(
            f"  Recomputed features for {pair_number}/{len(grouped)} graph pairs "
            f"in {output_path.name}.",
            flush=True,
        )

    refreshed_frame = pd.DataFrame(refreshed_rows)
    original_keys = existing_frame[["graph_pair_key", "candidate_id"]].reset_index(
        drop=True
    )
    refreshed_keys = refreshed_frame[
        ["graph_pair_key", "candidate_id"]
    ].reset_index(drop=True)
    if not original_keys.equals(refreshed_keys):
        raise ValueError("Refreshed rows do not align with the original observations.")

    for column in SGM_RESULT_COLUMNS:
        original = existing_frame[column].to_numpy()
        refreshed = refreshed_frame[column].to_numpy()
        if not np.array_equal(original, refreshed, equal_nan=True):
            raise ValueError(f"SGM result column changed during refresh: {column}.")

    temporary_output = output_path.with_suffix(".csv.tmp")
    refreshed_frame.to_csv(temporary_output, index=False)
    temporary_output.replace(output_path)

    metadata["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    metadata["config"] = asdict(config)
    metadata["columns"] = list(refreshed_frame.columns)
    metadata["rows_written"] = len(refreshed_frame)
    metadata["features_refreshed_without_sgm"] = True
    metadata["prequery_feature_prefixes"] = ["pair_", "seed_", "nonseed_"]
    metadata["exclude_from_prequery_models"] = [
        "feature_schema_version",
        "graph_pair_key",
        "graph_pair_id",
        "candidate_id",
        "graph_pair_seed",
        "sgm_seed",
        "seed_vertices_graph1",
        "seed_vertices_graph2",
        *SGM_RESULT_COLUMNS,
    ]

    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    with temporary_metadata.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")
    temporary_metadata.replace(metadata_path)

    print(
        f"Refreshed {output_path} with {len(refreshed_frame)} rows and "
        f"{len(refreshed_frame.columns)} columns; saved SGM results were preserved."
    )
    return metadata


def collect_data(
    config: ExperimentConfig, output_path: Path, overwrite: bool
) -> dict[str, Any]:
    validate_config(config)
    output_path = output_path.expanduser().resolve()
    metadata_path = output_path.with_suffix(".metadata.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Choose another path or pass --overwrite."
        )

    rng = np.random.default_rng(config.random_seed)
    collection_started = time.perf_counter()
    time_budget_seconds = config.max_minutes * 60.0
    safety_margin_seconds = min(60.0, 0.05 * time_budget_seconds)
    stop_new_work_at = collection_started + time_budget_seconds - safety_margin_seconds

    rows_written = 0
    completed_graph_pairs = 0
    stopped_for_time = False
    recent_sgm_runtimes: list[float] = []
    fieldnames: list[str] | None = None

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer: csv.DictWriter[str] | None = None

        for graph_pair_id in range(config.n_graph_pairs):
            if time.perf_counter() >= stop_new_work_at:
                stopped_for_time = True
                break

            graph_pair_seed = int(rng.integers(0, UINT32_MAX, dtype=np.uint32))
            adjacency_1, adjacency_2, true_permutation = generate_graph_pair(
                config, graph_pair_seed
            )
            context_1 = build_graph_context(adjacency_1)
            context_2 = build_graph_context(
                adjacency_2, compute_seed_features=False
            )
            pair_features = graph_pair_features(context_1, context_2)
            candidate_sets = sample_unique_seed_sets(
                rng,
                config.n_vertices,
                config.n_seeds,
                config.candidates_per_pair,
            )

            completed_candidates = 0
            for candidate_id, seeds_1 in enumerate(candidate_sets):
                remaining_seconds = stop_new_work_at - time.perf_counter()
                estimated_next_runtime = (
                    float(np.median(recent_sgm_runtimes[-20:]))
                    if len(recent_sgm_runtimes) >= 5
                    else 0.0
                )
                if remaining_seconds <= estimated_next_runtime:
                    stopped_for_time = True
                    break

                seeds_2 = true_permutation[seeds_1]
                sgm_seed = int(rng.integers(0, UINT32_MAX, dtype=np.uint32))
                candidate_features = seed_set_features(context_1, seeds_1)
                performance = run_sgm_once(
                    adjacency_1,
                    adjacency_2,
                    true_permutation,
                    seeds_1,
                    sgm_seed,
                    config.max_iter,
                    config.tol,
                )
                recent_sgm_runtimes.append(float(performance["sgm_runtime_seconds"]))

                row = row_context(
                    config,
                    graph_pair_key=f"{config.graph_model}_{graph_pair_id}",
                    graph_pair_id=graph_pair_id,
                    candidate_id=candidate_id,
                    graph_pair_seed=graph_pair_seed,
                    sgm_seed=sgm_seed,
                    seeds_1=seeds_1,
                    seeds_2=seeds_2,
                )
                row.update(pair_features)
                row.update(candidate_features)
                row.update(performance)

                if writer is None:
                    fieldnames = list(row)
                    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow(row)
                rows_written += 1
                completed_candidates += 1

                if rows_written % 10 == 0:
                    elapsed_minutes = (time.perf_counter() - collection_started) / 60
                    print(
                        f"Collected {rows_written} rows "
                        f"({elapsed_minutes:.1f} minutes elapsed).",
                        flush=True,
                    )

            output_file.flush()
            if completed_candidates == config.candidates_per_pair:
                completed_graph_pairs += 1
            if stopped_for_time:
                break

    elapsed_seconds = time.perf_counter() - collection_started
    metadata: dict[str, Any] = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "config": asdict(config),
        "output_csv": str(output_path),
        "columns": fieldnames or [],
        "rows_written": rows_written,
        "completed_graph_pairs": completed_graph_pairs,
        "stopped_for_time_budget": stopped_for_time,
        "elapsed_seconds": elapsed_seconds,
        "group_column_for_model_validation": "graph_pair_key",
        "target_columns": [
            "h0_accuracy",
            "final_unseeded_accuracy",
            "final_all_accuracy",
            "sgm_runtime_seconds",
        ],
        "candidate_feature_scope": "graph 1 only",
        "graph_feature_scope": (
            "permutation-invariant summaries of graph 1 and graph 2"
        ),
        "prequery_feature_prefixes": ["pair_", "seed_", "nonseed_"],
        "exclude_from_prequery_models": [
            "feature_schema_version",
            "graph_pair_key",
            "graph_pair_id",
            "candidate_id",
            "graph_pair_seed",
            "sgm_seed",
            "seed_vertices_graph1",
            "seed_vertices_graph2",
            *SGM_RESULT_COLUMNS,
        ],
        "time_budget_is_soft": (
            "The collector stops before starting new SGM runs; one active run is not "
            "interrupted and may finish after the requested deadline."
        ),
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")

    print(f"Wrote {rows_written} observations to {output_path}")
    print(f"Wrote run metadata to {metadata_path}")
    if stopped_for_time:
        print("Stopped before starting another SGM run because of the time budget.")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-model",
        choices=("all",) + GRAPH_MODELS,
        default="all",
        help="Collect all recommended regimes or one selected graph model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the preset output CSV and metadata files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Custom CSV path; available only when collecting one graph model.",
    )
    parser.add_argument("--n-graph-pairs", type=int, default=20)
    parser.add_argument("--candidates-per-pair", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=0.01)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=30.0,
        help="Soft time limit applied separately to each output dataset.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output CSV."
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help=(
            "Rebuild feature columns in existing CSV files from their saved graph "
            "seeds without rerunning SGM."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.graph_model == "all" and args.output is not None:
        raise ValueError("--output can only be used with one selected graph model.")

    graph_models = GRAPH_MODELS if args.graph_model == "all" else (args.graph_model,)
    for graph_model in graph_models:
        config = replace(
            preset_config(graph_model),
            n_graph_pairs=args.n_graph_pairs,
            candidates_per_pair=args.candidates_per_pair,
            max_iter=args.max_iter,
            tol=args.tol,
            random_seed=args.random_seed,
            max_minutes=args.max_minutes,
        )
        output_path = (
            args.output
            if args.output is not None
            else args.output_dir / OUTPUT_FILENAMES[graph_model]
        )
        if args.features_only:
            if args.graph_model == "all" and not output_path.exists():
                print(f"Skipping {output_path}; no existing dataset was found.")
                continue
            print(f"Refreshing features in {output_path} without running SGM.")
            refresh_collected_features(output_path)
            continue
        print(
            f"Collecting {graph_model}: {config.n_graph_pairs} graph pairs x "
            f"{config.candidates_per_pair} candidates, {config.n_seeds} seeds."
        )
        collect_data(config, output_path, args.overwrite)


if __name__ == "__main__":
    main()
