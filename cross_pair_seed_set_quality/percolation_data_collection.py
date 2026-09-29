"""Collect seed-set features and ExpandWhenStuck percolation outcomes.

Predictors use the same compact Graph-A seed-set features and
correspondence-free graph-pair summaries as ``data_collection.py``.  The
percolation results include final accuracy, matched coverage, precision, and
runtime.  Correct seed correspondences and all algorithm outputs are excluded
from the documented pre-query predictor set.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ExpandWhenStuck import graph_match_percolation
from .data_collection import (
    ExperimentConfig,
    FEATURE_SCHEMA_VERSION,
    UINT32_MAX,
    build_graph_context,
    generate_graph_pair,
    graph_pair_features,
    preset_config as graph_preset_config,
    sample_unique_seed_sets,
    seed_set_features,
    validate_config as validate_graph_config,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "data"


COLLECTION_SCHEMA_VERSION = 1
CALIBRATION_SUCCESS_THRESHOLD = 0.90
OUTPUT_FILENAMES = {
    "sparse_er": "percolation_seed_quality_sparse_er.csv",
    "sparse_sbm": "percolation_seed_quality_sparse_sbm.csv",
    "paper": "percolation_seed_quality_paper.csv",
    "ier": "percolation_seed_quality_ier.csv",
}
GRAPH_MODELS = tuple(OUTPUT_FILENAMES)

PERCOLATION_RESULT_COLUMNS = (
    "final_unseeded_accuracy",
    "final_all_accuracy",
    "percolation_matched_unseeded_fraction",
    "percolation_unseeded_precision",
    "percolation_incorrect_unseeded_fraction",
    "percolation_matches_added",
    "percolation_completed",
    "percolation_runtime_seconds",
)

CALIBRATION_NOTES = {
    "sparse_er": (
        "At rho=0.80 and 18 random seeds, 22/54 calibration runs reached "
        "90% final unseeded accuracy."
    ),
    "sparse_sbm": (
        "At rho=0.80 and 14 random seeds, 20/54 calibration runs reached "
        "90% final unseeded accuracy."
    ),
    "paper": (
        "For PAPER p=0.01, rho=0.75, and 10 random seeds, 15/40 "
        "calibration runs reached 90% final unseeded accuracy."
    ),
    "ier": (
        "For Beta(1, 99), rho=0.77, and 20 random seeds, 23/54 calibration "
        "runs reached 90% final unseeded accuracy."
    ),
}


@dataclass(frozen=True)
class PercolationConfig:
    graph: ExperimentConfig
    threshold: int = 2
    expand_when_stuck: bool = True

    @property
    def graph_model(self) -> str:
        return self.graph.graph_model

    @property
    def n_vertices(self) -> int:
        return self.graph.n_vertices


def preset_config(graph_model: str) -> PercolationConfig:
    """Return a calibrated small-scale percolation configuration."""

    graph = graph_preset_config(graph_model)
    if graph_model == "sparse_er":
        graph = replace(graph, n_seeds=18)
    elif graph_model == "sparse_sbm":
        graph = replace(graph, n_seeds=14)
    elif graph_model == "paper":
        graph = replace(graph, paper_probability=0.01, rho=0.75, n_seeds=10)
    elif graph_model == "ier":
        graph = replace(graph, rho=0.77, n_seeds=20)
    else:
        raise ValueError(f"Unknown graph model: {graph_model}")
    return PercolationConfig(graph=graph)


def validate_config(config: PercolationConfig) -> None:
    validate_graph_config(config.graph)
    if not isinstance(config.threshold, int) or config.threshold <= 0:
        raise ValueError("threshold must be a positive integer.")
    if not isinstance(config.expand_when_stuck, bool):
        raise ValueError("expand_when_stuck must be boolean.")


def config_from_dict(values: dict[str, Any]) -> PercolationConfig:
    return PercolationConfig(
        graph=ExperimentConfig(**values["graph"]),
        threshold=int(values["threshold"]),
        expand_when_stuck=bool(values["expand_when_stuck"]),
    )


def run_percolation_once(
    adjacency_1: np.ndarray,
    adjacency_2: np.ndarray,
    true_permutation: np.ndarray,
    seeds_1: np.ndarray,
    algorithm_seed: int,
    threshold: int,
    expand_when_stuck: bool,
) -> dict[str, float | int | bool]:
    """Run percolation once and score unmatched vertices as incorrect."""

    seeds_1 = np.asarray(seeds_1, dtype=int)
    seeds_2 = true_permutation[seeds_1]
    partial_match = np.column_stack((seeds_1, seeds_2))

    random_state = np.random.get_state()
    np.random.seed(algorithm_seed)
    started = time.perf_counter()
    try:
        predicted = graph_match_percolation(
            adjacency_1,
            adjacency_2,
            partial_match,
            r=threshold,
            ExpandWhenStuck=expand_when_stuck,
        )
    finally:
        runtime_seconds = time.perf_counter() - started
        np.random.set_state(random_state)

    predicted = np.asarray(predicted, dtype=int)
    n = adjacency_1.shape[0]
    if predicted.shape != (n,):
        raise RuntimeError("Percolation returned an invalid permutation shape.")
    matched_targets = predicted[predicted >= 0]
    if np.any(matched_targets >= n) or len(np.unique(matched_targets)) != len(
        matched_targets
    ):
        raise RuntimeError("Percolation returned invalid or repeated matches.")
    if not np.array_equal(predicted[seeds_1], seeds_2):
        raise RuntimeError("Percolation did not preserve the supplied seed matches.")

    all_vertices = np.arange(n)
    unseeded = np.setdiff1d(all_vertices, seeds_1)
    unseeded_prediction = predicted[unseeded]
    matched = unseeded_prediction >= 0
    correct = unseeded_prediction == true_permutation[unseeded]
    incorrect = matched & ~correct
    matched_count = int(np.sum(matched))

    return {
        "final_unseeded_accuracy": float(np.mean(correct)),
        "final_all_accuracy": float(np.mean(predicted == true_permutation)),
        "percolation_matched_unseeded_fraction": float(np.mean(matched)),
        "percolation_unseeded_precision": (
            float(np.sum(correct) / matched_count) if matched_count else 0.0
        ),
        "percolation_incorrect_unseeded_fraction": float(np.mean(incorrect)),
        "percolation_matches_added": matched_count,
        "percolation_completed": bool(matched_count == len(unseeded)),
        "percolation_runtime_seconds": runtime_seconds,
    }


def row_context(
    config: PercolationConfig,
    *,
    graph_pair_key: str,
    graph_pair_id: int,
    candidate_id: int,
    graph_pair_seed: int,
    algorithm_seed: int,
    seeds_1: np.ndarray,
    seeds_2: np.ndarray,
) -> dict[str, Any]:
    graph = config.graph
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "collection_schema_version": COLLECTION_SCHEMA_VERSION,
        "algorithm": "percolation_expand_when_stuck",
        "graph_model": graph.graph_model,
        "graph_pair_key": graph_pair_key,
        "graph_pair_id": graph_pair_id,
        "candidate_id": candidate_id,
        "graph_pair_seed": graph_pair_seed,
        "algorithm_seed": algorithm_seed,
        "seed_vertices_graph1": json.dumps(seeds_1.tolist()),
        "seed_vertices_graph2": json.dumps(seeds_2.tolist()),
        "n_vertices": graph.n_vertices,
        "n_seeds": graph.n_seeds,
        "rho": graph.rho,
        "percolation_threshold": config.threshold,
        "expand_when_stuck": config.expand_when_stuck,
        "er_probability": (
            graph.er_probability if graph.graph_model == "sparse_er" else None
        ),
        "sbm_n_blocks": (
            graph.n_blocks if graph.graph_model == "sparse_sbm" else None
        ),
        "sbm_n_per_block": (
            graph.n_per_block if graph.graph_model == "sparse_sbm" else None
        ),
        "sbm_block_probabilities": (
            json.dumps(graph.sbm_block_probabilities)
            if graph.graph_model == "sparse_sbm"
            else None
        ),
        "paper_alpha": (
            graph.paper_alpha if graph.graph_model == "paper" else None
        ),
        "paper_probability": (
            graph.paper_probability if graph.graph_model == "paper" else None
        ),
        "ier_alpha": graph.ier_alpha if graph.graph_model == "ier" else None,
        "ier_beta": graph.ier_beta if graph.graph_model == "ier" else None,
    }


def refresh_collected_features(output_path: Path) -> dict[str, Any]:
    """Refresh predictors while preserving previously collected outcomes."""

    import pandas as pd

    output_path = output_path.expanduser().resolve()
    metadata_path = output_path.with_suffix(".metadata.json")
    for required_path in (output_path, metadata_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Could not find {required_path}.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = config_from_dict(metadata["config"])
    validate_config(config)
    existing = pd.read_csv(output_path)
    required_columns = {
        "graph_pair_key",
        "graph_pair_id",
        "candidate_id",
        "graph_pair_seed",
        "algorithm_seed",
        "seed_vertices_graph1",
        "seed_vertices_graph2",
    } | set(PERCOLATION_RESULT_COLUMNS)
    missing = required_columns - set(existing.columns)
    if missing:
        raise ValueError(f"Cannot refresh {output_path}; missing {sorted(missing)}.")
    if existing.empty:
        raise ValueError(f"Cannot refresh empty dataset {output_path}.")

    rows: list[dict[str, Any]] = []
    grouped = existing.groupby("graph_pair_key", sort=False)
    for pair_number, (pair_key, group) in enumerate(grouped, start=1):
        pair_seeds = group["graph_pair_seed"].unique()
        if len(pair_seeds) != 1:
            raise ValueError(f"Graph pair {pair_key} has multiple generation seeds.")
        graph_pair_seed = int(pair_seeds[0])
        adjacency_1, adjacency_2, true_permutation = generate_graph_pair(
            config.graph, graph_pair_seed
        )
        context_1 = build_graph_context(adjacency_1)
        context_2 = build_graph_context(adjacency_2, compute_seed_features=False)
        pair_features = graph_pair_features(context_1, context_2)

        for _, saved in group.iterrows():
            seeds_1 = np.asarray(json.loads(saved["seed_vertices_graph1"]), dtype=int)
            seeds_2 = np.asarray(json.loads(saved["seed_vertices_graph2"]), dtype=int)
            if not np.array_equal(seeds_2, true_permutation[seeds_1]):
                raise ValueError(
                    f"Regenerated correspondence does not match {pair_key}, "
                    f"candidate {saved['candidate_id']}."
                )
            row = row_context(
                config,
                graph_pair_key=str(saved["graph_pair_key"]),
                graph_pair_id=int(saved["graph_pair_id"]),
                candidate_id=int(saved["candidate_id"]),
                graph_pair_seed=graph_pair_seed,
                algorithm_seed=int(saved["algorithm_seed"]),
                seeds_1=seeds_1,
                seeds_2=seeds_2,
            )
            row.update(pair_features)
            row.update(seed_set_features(context_1, seeds_1))
            row.update({column: saved[column] for column in PERCOLATION_RESULT_COLUMNS})
            rows.append(row)

        print(
            f"  Recomputed features for {pair_number}/{len(grouped)} graph pairs "
            f"in {output_path.name}.",
            flush=True,
        )

    refreshed = pd.DataFrame(rows)
    original_keys = existing[["graph_pair_key", "candidate_id"]].reset_index(
        drop=True
    )
    refreshed_keys = refreshed[["graph_pair_key", "candidate_id"]].reset_index(
        drop=True
    )
    if not original_keys.equals(refreshed_keys):
        raise ValueError("Refreshed rows do not align with the original observations.")
    for column in PERCOLATION_RESULT_COLUMNS:
        if not np.array_equal(
            existing[column].to_numpy(),
            refreshed[column].to_numpy(),
            equal_nan=True,
        ):
            raise ValueError(f"Outcome changed during feature refresh: {column}.")

    temporary_output = output_path.with_suffix(".csv.tmp")
    refreshed.to_csv(temporary_output, index=False)
    temporary_output.replace(output_path)

    metadata.update(
        {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "collection_schema_version": COLLECTION_SCHEMA_VERSION,
            "config": asdict(config),
            "columns": list(refreshed.columns),
            "rows_written": len(refreshed),
            "features_refreshed_without_percolation": True,
        }
    )
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    temporary_metadata.replace(metadata_path)
    print(
        f"Refreshed {output_path} with {len(refreshed)} rows; stored "
        "percolation results were preserved."
    )
    return metadata


def collect_data(
    config: PercolationConfig, output_path: Path, overwrite: bool
) -> dict[str, Any]:
    validate_config(config)
    graph = config.graph
    output_path = output_path.expanduser().resolve()
    metadata_path = output_path.with_suffix(".metadata.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Choose another path or pass --overwrite."
        )

    rng = np.random.default_rng(graph.random_seed)
    started = time.perf_counter()
    time_budget_seconds = graph.max_minutes * 60.0
    safety_margin_seconds = min(60.0, 0.05 * time_budget_seconds)
    stop_new_work_at = started + time_budget_seconds - safety_margin_seconds
    recent_runtimes: list[float] = []
    rows_written = 0
    completed_graph_pairs = 0
    stopped_for_time = False
    fieldnames: list[str] | None = None

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer: csv.DictWriter[str] | None = None
        for graph_pair_id in range(graph.n_graph_pairs):
            if time.perf_counter() >= stop_new_work_at:
                stopped_for_time = True
                break

            graph_pair_seed = int(rng.integers(0, UINT32_MAX, dtype=np.uint32))
            adjacency_1, adjacency_2, true_permutation = generate_graph_pair(
                graph, graph_pair_seed
            )
            context_1 = build_graph_context(adjacency_1)
            context_2 = build_graph_context(
                adjacency_2, compute_seed_features=False
            )
            pair_features = graph_pair_features(context_1, context_2)
            candidates = sample_unique_seed_sets(
                rng, graph.n_vertices, graph.n_seeds, graph.candidates_per_pair
            )

            completed_candidates = 0
            for candidate_id, seeds_1 in enumerate(candidates):
                estimated_runtime = (
                    float(np.median(recent_runtimes[-20:]))
                    if len(recent_runtimes) >= 5
                    else 0.0
                )
                if stop_new_work_at - time.perf_counter() <= estimated_runtime:
                    stopped_for_time = True
                    break

                seeds_2 = true_permutation[seeds_1]
                algorithm_seed = int(
                    rng.integers(0, UINT32_MAX, dtype=np.uint32)
                )
                features = seed_set_features(context_1, seeds_1)
                outcome = run_percolation_once(
                    adjacency_1,
                    adjacency_2,
                    true_permutation,
                    seeds_1,
                    algorithm_seed,
                    config.threshold,
                    config.expand_when_stuck,
                )
                recent_runtimes.append(
                    float(outcome["percolation_runtime_seconds"])
                )

                row = row_context(
                    config,
                    graph_pair_key=(
                        f"percolation_{graph.graph_model}_{graph_pair_id}"
                    ),
                    graph_pair_id=graph_pair_id,
                    candidate_id=candidate_id,
                    graph_pair_seed=graph_pair_seed,
                    algorithm_seed=algorithm_seed,
                    seeds_1=seeds_1,
                    seeds_2=seeds_2,
                )
                row.update(pair_features)
                row.update(features)
                row.update(outcome)
                if writer is None:
                    fieldnames = list(row)
                    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow(row)
                rows_written += 1
                completed_candidates += 1

                if rows_written % 10 == 0:
                    elapsed_minutes = (time.perf_counter() - started) / 60
                    print(
                        f"Collected {rows_written} rows "
                        f"({elapsed_minutes:.1f} minutes elapsed).",
                        flush=True,
                    )

            output_file.flush()
            if completed_candidates == graph.candidates_per_pair:
                completed_graph_pairs += 1
            if stopped_for_time:
                break

    elapsed_seconds = time.perf_counter() - started
    metadata: dict[str, Any] = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "collection_schema_version": COLLECTION_SCHEMA_VERSION,
        "config": asdict(config),
        "output_csv": str(output_path),
        "columns": fieldnames or [],
        "rows_written": rows_written,
        "completed_graph_pairs": completed_graph_pairs,
        "stopped_for_time_budget": stopped_for_time,
        "elapsed_seconds": elapsed_seconds,
        "calibration_success_threshold": CALIBRATION_SUCCESS_THRESHOLD,
        "preset_calibration": CALIBRATION_NOTES[graph.graph_model],
        "group_column_for_model_validation": "graph_pair_key",
        "target_columns": list(PERCOLATION_RESULT_COLUMNS),
        "candidate_feature_scope": "graph 1 only",
        "graph_feature_scope": (
            "permutation-invariant summaries of graph 1 and graph 2"
        ),
        "prequery_feature_prefixes": ["pair_", "seed_", "nonseed_"],
        "exclude_from_prequery_models": [
            "feature_schema_version",
            "collection_schema_version",
            "graph_pair_key",
            "graph_pair_id",
            "candidate_id",
            "graph_pair_seed",
            "algorithm_seed",
            "seed_vertices_graph1",
            "seed_vertices_graph2",
            *PERCOLATION_RESULT_COLUMNS,
        ],
        "time_budget_is_soft": (
            "The collector stops before starting a new percolation run; one "
            "active run is not interrupted."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {rows_written} observations to {output_path}")
    print(f"Wrote run metadata to {metadata_path}")
    if stopped_for_time:
        print("Stopped before another run because of the time budget.")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-model",
        choices=("all",) + GRAPH_MODELS,
        default="all",
        help="Collect every calibrated regime or one selected graph model.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Custom CSV path; available only for one selected graph model.",
    )
    parser.add_argument("--n-graph-pairs", type=int, default=20)
    parser.add_argument("--candidates-per-pair", type=int, default=50)
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Override the calibrated witness threshold (default: 2).",
    )
    parser.add_argument(
        "--disable-expand-when-stuck",
        action="store_true",
        help="Run basic percolation without the ExpandWhenStuck phase.",
    )
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--max-minutes", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Refresh features from saved graph seeds without rerunning percolation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.graph_model == "all" and args.output is not None:
        raise ValueError("--output can only be used with one graph model.")

    graph_models = GRAPH_MODELS if args.graph_model == "all" else (args.graph_model,)
    for graph_model in graph_models:
        config = preset_config(graph_model)
        graph = replace(
            config.graph,
            n_graph_pairs=args.n_graph_pairs,
            candidates_per_pair=args.candidates_per_pair,
            random_seed=args.random_seed,
            max_minutes=args.max_minutes,
        )
        config = replace(
            config,
            graph=graph,
            threshold=(
                args.threshold if args.threshold is not None else config.threshold
            ),
            expand_when_stuck=not args.disable_expand_when_stuck,
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
            print(f"Refreshing features in {output_path} without running percolation.")
            refresh_collected_features(output_path)
            continue

        print(
            f"Collecting {graph_model}: {graph.n_graph_pairs} graph pairs x "
            f"{graph.candidates_per_pair} candidates, {graph.n_seeds} seeds, "
            f"threshold {config.threshold}."
        )
        collect_data(config, output_path, args.overwrite)


if __name__ == "__main__":
    main()
