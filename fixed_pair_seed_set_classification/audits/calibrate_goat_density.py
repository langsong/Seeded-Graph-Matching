"""Calibrate GOAT budgets at degree 10 or 12 while holding high rho fixed.

Uses the existing calibration driver's uniform sampling, fresh confirmation
phases, provenance checks, and public graspologic evaluator. Changing density
changes Graph A; each density's saved pair is then fixed across seed budgets.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from . import calibrate_goat_high_rho as calibration

import numpy as np

from experiment_common.graph_checks import check_graph_pair
from experiment_common.graphs import ExperimentConfig, generate_graph_pair
from experiment_common.provenance import write_json


def prepare_density_pair(model, output, target_degree):
    settings = calibration.SETTINGS[model]
    source_dir = calibration.SOURCE / "graph_pairs" / model / f"pair_seed_{settings['seed']}"
    source_json = source_dir / f"{model}_problem.metadata.json"
    original = json.loads(source_json.read_text())
    source_config = ExperimentConfig(**original["config"])
    n = source_config.vertex_count
    # A PA tree has n-1 edges. ER edges already in the tree add no degree.
    tree_mean_degree = 2 * (n - 1) / n
    density = (
        target_degree / (n - 1) if model == "sparse_er"
        else (target_degree - tree_mean_degree) / (n - 1 - tree_mean_degree)
    )
    density_field = "er_probability" if model == "sparse_er" else "paper_pa_probability"
    config = replace(source_config, rho=settings["rho"], **{density_field: density})
    A, B, truth = generate_graph_pair(config, settings["seed"])
    structure = check_graph_pair(A, B)
    if not structure["passes"]:
        raise ValueError(f"{model}, degree {target_degree}: structural checks failed: {structure}")
    # Verify a later comparison at lower rho can preserve this new Graph A.
    control_A, _, control_truth = generate_graph_pair(
        replace(config, rho=source_config.rho), settings["seed"]
    )
    assert np.array_equal(A, control_A)
    assert np.array_equal(truth, control_truth)
    aligned_B = B[np.ix_(truth, truth)]
    upper = np.triu_indices(n, 1)
    metadata = {
        "model": model, "graph_pair_seed": settings["seed"],
        "target_mean_degree": target_degree, "config": asdict(config),
        "budget_note": "config.n_seeds is inherited and unused for pair generation; actual k is in runs.csv",
        "structure": structure, "source_config": str(source_json),
        "source_config_sha256": calibration.digest(source_json),
        "same_graph_A_as_degree16": False,
        "same_graph_A_for_lower_rho_at_this_density_verified": True,
        "generation": "Original project generator and pair RNG seed; change density and rho. No rejected draws.",
        "retained_edge_fraction": float((A * aligned_B).sum() / A.sum()),
        "observed_edge_correlation": float(np.corrcoef(A[upper], aligned_B[upper])[0, 1]),
        "solver": "goat", "regularizer": settings["reg"],
    }
    directory = output / "pairs" / model
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model}_problem.npz"
    arrays = dict(adjacency_graph1=A, adjacency_graph2=B, true_permutation=truth)
    if path.exists():
        assert json.loads(path.with_suffix(".metadata.json").read_text()) == metadata
        with np.load(path) as saved:
            assert all(np.array_equal(saved[key], value) for key, value in arrays.items())
    else:
        np.savez_compressed(path, **arrays)
        write_json(path.with_suffix(".metadata.json"), metadata)
    return A, B, truth


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mean-degree", type=int, choices=[10, 12], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "target_mean_degree": args.mean_degree,
        "rho": {"sparse_er": 0.70, "paper_pa": 0.62},
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    path = args.output / "density_calibration.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "Different density configuration"
    else:
        write_json(path, manifest)
    calibration.prepare_pair = lambda model, output: prepare_density_pair(
        model, output, args.mean_degree
    )
    sys.argv = [sys.argv[0], "--output", str(args.output), *remaining]
    calibration.main()


if __name__ == "__main__":
    main()
