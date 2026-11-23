"""Bounded integration checks for one-run FAQ/GOAT collection."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_mpl_cache"))

import numpy as np
import pandas as pd

from experiment_common.graphs import generate_graph_pair, preset_config
from experiment_common.provenance import thread_limit
from experiment_common.sgm import SGM_DIAGNOSTIC_COLUMNS, run_sgm_once
from fixed_pair_seed_set_classification import candidate_data as collection
from fixed_pair_seed_set_classification.count_analysis import load_collection
from fixed_pair_seed_set_classification.seed_selection_common import FixedProblem
from fixed_pair_seed_set_classification.audits import reproduce_saved_sets


class FAQGOATCollectionTests(unittest.TestCase):
    def setUp(self):
        self.spec = FixedProblem(
            name="sparse_er",
            config=replace(preset_config("sparse_er"), vertex_count=48,
                           er_probability=0.35, rho=0.8, n_seeds=6, max_iter=100),
            graph_pair_seed=5555,
        )

    def test_cli_routes_both_solvers_to_one_primary_pair_per_model(self):
        with patch.object(sys, "argv", ["candidate_data"]), patch.object(collection, "collect_problem") as collect:
            collection.main()
        self.assertEqual(collect.call_count, 6)
        for name in collection.FOCUS_PROBLEMS:
            calls = [call for call in collect.call_args_list if call.args[0].name == name]
            self.assertEqual(len(calls), 2)
            faq, goat = calls
            self.assertEqual(faq.kwargs["solver"], "faq")
            self.assertEqual(goat.kwargs["solver"], "goat")
            self.assertEqual(faq.args[0], collection.PROBLEMS[name])
            self.assertEqual(goat.args[0], collection.GOAT_PROBLEMS[name])
            self.assertEqual(faq.kwargs["problem_dir"], goat.kwargs["problem_dir"])
            self.assertNotEqual(faq.args[1], goat.args[1])
            self.assertEqual(faq.args[0].graph_pair_seed, goat.args[0].graph_pair_seed)
            for call in calls:
                self.assertEqual(call.kwargs["solver_runs"], 1)
                counts = call.kwargs["sampling"]
                self.assertEqual(sum(counts[key] for key in collection.DEFAULT_SAMPLING_COUNTS), 1000)
                self.assertEqual(counts["n_random_test"], 200)
                self.assertEqual(call.kwargs["transport_max_iter"], 5000)

    def test_real_api_preserves_legacy_columns_and_reports_soft_direction(self):
        A, B, truth = generate_graph_pair(self.spec.config, self.spec.graph_pair_seed)
        seeds = np.array([0, 1, 2, 3, 4, 5])
        with thread_limit(1):
            faq = run_sgm_once(A, B, truth, seeds, 2026, 100, 0.01)
            goat = run_sgm_once(A, B, truth, seeds, 2026, 100, 0.01,
                                transport=True, return_diagnostics=True)
        self.assertEqual(set(faq), set(collection.SGM_RESULT_COLUMNS))
        self.assertEqual(set(goat), set(collection.RUN_RESULT_COLUMNS))
        for result in [faq, goat]:
            self.assertTrue(0 <= result["final_unseeded_accuracy"] <= 1)
            self.assertTrue(0 <= result["h0_accuracy"] <= 1)
        self.assertTrue(goat["sgm_finite_soft_solution"])
        self.assertGreater(goat["h0_row_entropy_mean"], 0)
        self.assertEqual(goat["sgm_warning_count"], len(json.loads(goat["sgm_warnings"])))

    def test_real_collection_labels_shared_pair_and_reproduction(self):
        sampling = dict(n_random_train=2, n_random_test=2, n_degree_weighted=2,
                        n_swap_paths=0, n_decorrelated=2, pool_size=120,
                        n_contrast_bins=2, contrast_features=("coverage",))
        with tempfile.TemporaryDirectory(prefix="sgm_collection_check_") as directory:
            root = Path(directory)
            shared_pair = root / "shared_pair"
            frames = []
            for solver in ["faq", "goat"]:
                output = root / solver
                with contextlib.redirect_stdout(io.StringIO()):
                    collection.collect_problem(self.spec, output, sampling=sampling,
                                               solver=solver, problem_dir=shared_pair, blas_threads=1)
                data, metadata = load_collection(output, "sparse_er", restarts=1)
                runs = pd.read_csv(output / "sparse_er_seed_sets_runs.csv")
                self.assertEqual(len(data), 8)
                self.assertEqual(len(runs), 8)
                self.assertEqual(metadata["status"], "complete")
                self.assertEqual(metadata["solver"], solver)
                self.assertEqual(metadata["solver_runs"], 1)
                self.assertEqual(metadata["n_init"], 1)
                self.assertTrue(metadata["graph_pair_checks"]["passes"])
                self.assertTrue(data["run_count"].eq(1).all())
                self.assertTrue((data["sgm_success"] == (data["mean_final_accuracy"] >= 0.9)).all())
                self.assertTrue(data["success_count"].isin([0, 1]).all())
                self.assertTrue((data["success_fraction"] == data["sgm_success"].astype(int)).all())
                self.assertTrue(data["sgm_seeds"].map(lambda x: len(json.loads(x)) == 1).all())
                self.assertEqual(runs["sgm_seed"].nunique(), 8)
                self.assertTrue(set(SGM_DIAGNOSTIC_COLUMNS) <= set(runs))
                self.assertTrue(metadata["solver_options"]["transport"] == (solver == "goat"))
                self.assertTrue((output / metadata["graph_pair_dir_relative"]).resolve() == shared_pair.resolve())
                self.assertFalse((output / "graph_pairs").exists())
                frames.append(data)
                with patch.object(sys, "argv", ["reproduce_saved_sets", "--graph-model", "sparse_er",
                                                "--data-dir", str(output), "--n-sets", "2", "--blas-threads", "1"]), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(reproduce_saved_sets.main(), 0)
                with self.assertRaises(FileExistsError):
                    collection.collect_problem(self.spec, output, sampling=sampling, solver=solver,
                                               problem_dir=shared_pair, blas_threads=1)
            self.assertEqual(frames[0]["seed_vertices_graph1"].tolist(), frames[1]["seed_vertices_graph1"].tolist())
            np.testing.assert_array_equal(frames[0]["sgm_seed"], frames[1]["sgm_seed"])
            self.assertEqual(len(list(shared_pair.glob("*.npz"))), 1)


if __name__ == "__main__":
    unittest.main()
