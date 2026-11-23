"""Checks of the two-phase active test's bookkeeping, using a stand-in matcher (no graspologic needed).

    python -m unittest real_data_experiments.tests.test_active_test
"""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .. import active_test as T


def neighbour_matcher(A, B, truth, seeds, algorithm, solver_seed, config):
    """Stand-in: seeds and their A-neighbours are matched correctly, the rest are shuffled.

    Its confidence is high exactly on the correctly matched vertices, so least-confident
    queries should hit errors."""

    n, rng = len(truth), np.random.default_rng(solver_seed)
    seeds = {int(v) for v in seeds}
    known = sorted(seeds | {int(w) for s in seeds for w in np.flatnonzero(A[s])})
    rest = np.setdiff1d(np.arange(n), known)
    perm = np.empty(n, dtype=int)
    perm[known] = truth[known]
    perm[rest] = rng.permutation(truth[rest])
    confidence = np.where(np.isin(np.arange(n), known), 0.9, 0.1) + 0.01 * rng.random(n)
    confidence[sorted(seeds)] = np.nan
    return {"perm": perm, "confidence": confidence, "soft_check": 1.0, "runtime_seconds": 0.0,
            "converged": True}


def collapsing_matcher(A, B, truth, seeds, algorithm, solver_seed, config):
    """Stand-in that is right with the phase-one seeds and collapses once more seeds are added."""

    n = len(truth)
    if len(seeds) <= collapsing_matcher.k1:
        perm = truth.copy()
    else:   # every vertex shifted to its neighbour's partner, except the seeds
        perm = T.correct(np.roll(truth, 1), sorted(seeds), truth)
    return {"perm": perm, "confidence": np.full(n, np.nan), "soft_check": float("nan"),
            "runtime_seconds": 0.0, "converged": True}


class ActiveTestChecks(unittest.TestCase):
    def setUp(self):
        self.config = T.default_config()
        self.config["cases"] = {"toy": {"get": {"name": "synthetic", "n": 60, "p": 0.08, "rho": 0.8, "seed": 3},
                                        "budgets": [6, 11], "batch": 2}}
        self.config.update(random_draws={"small": 2, "large": 1}, control_draws={"small": 2, "large": 1},
                           restarts={"faq": {"small": 2, "large": 1}, "goat": {"small": 1, "large": 1}})

    def test_correct_fills_answers_and_keeps_a_matching(self):
        rng = np.random.default_rng(0)
        truth, perm = rng.permutation(30), rng.permutation(30)
        answers = [3, 7, 11, 12]
        fixed = T.correct(perm, answers, truth)
        self.assertEqual(sorted(fixed), list(range(30)))
        np.testing.assert_array_equal(fixed[answers], truth[answers])
        self.assertLessEqual(int(np.sum(fixed != perm)), 2 * len(answers))

    def test_auc(self):
        wrong = np.array([1, 1, 0, 0, 0], dtype=bool)
        self.assertEqual(T.auc_wrong(np.array([5, 4, 3, 2, 1.0]), wrong), 1.0)
        self.assertEqual(T.auc_wrong(np.array([1, 2, 3, 4, 5.0]), wrong), 0.0)
        self.assertEqual(T.auc_wrong(np.zeros(5), wrong), 0.5)
        self.assertTrue(math.isnan(T.auc_wrong(np.zeros(3), np.zeros(3, dtype=bool))))

    def test_every_policy_spends_exactly_k_and_rounds_add_up(self):
        for K in (6, 11):
            finals, rounds = T.run_block("toy", "faq", K, self.config, matcher=neighbour_matcher, blas_threads=None)
            self.assertEqual(len(finals), 2 + 4 + 1 + 1 + 2)
            k1 = round(0.5 * K)
            for row in finals:
                seeds = json.loads(row["seeds"])
                self.assertEqual(len(seeds), K)
                self.assertEqual(len(set(seeds)), K)
                self.assertAlmostEqual(row["acc_all"], row["total_correct"] / 60)
                if row["policy"].startswith("two_phase"):
                    self.assertEqual(row["phase1_seeds"], k1)
                    self.assertEqual(row["rounds"], math.ceil((K - k1) / 2))
            for policy in T.TWO_PHASE:
                path = [r for r in rounds if r["policy"] == policy and r["draw"] == 0]
                self.assertEqual([r["n_seeds"] for r in path][0], k1)
                self.assertEqual(path[-1]["n_seeds"], K)

    def test_confidence_finds_errors_when_it_carries_information(self):
        _, rounds = T.run_block("toy", "goat", 11, self.config, matcher=neighbour_matcher, blas_threads=None)
        active = [r for r in rounds if r["policy"] == "two_phase_confidence" and "hit_rate" in r]
        control = [r for r in rounds if r["policy"] == "two_phase_random" and "hit_rate" in r]
        self.assertTrue(all(r["certainty_source"] == "soft matching" for r in active))
        self.assertTrue(all(r["auc_confidence"] > 0.9 for r in active))
        self.assertGreater(np.mean([r["hit_rate"] for r in active]), 0.8)
        self.assertGreater(np.mean([r["hit_rate"] for r in active]), np.mean([r["hit_rate"] for r in control]))

    def test_guard_keeps_the_corrected_previous_matching(self):
        collapsing_matcher.k1 = 3
        finals, rounds = T.run_block("toy", "goat", 6, self.config, matcher=collapsing_matcher, blas_threads=None)
        for row in finals:
            if row["policy"].startswith("two_phase"):
                self.assertGreaterEqual(row["kept_previous_rounds"], 1)
                self.assertEqual(row["acc_all"], 1.0)
        unguarded = [r["acc_all_unguarded"] for r in rounds if r["kept_previous"]]
        self.assertTrue(unguarded and all(u < 0.2 for u in unguarded))

    def test_resume_and_design_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            T.run(root, self.config, cases=["toy"], matcher=neighbour_matcher)
            with (root / "finals.csv").open() as handle:
                first = len(list(csv.DictReader(handle)))
            with (root / "rounds.csv").open() as handle:
                first_rounds = len(list(csv.DictReader(handle)))
            T.run(root, self.config, cases=["toy"], matcher=neighbour_matcher)
            with (root / "finals.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), first)
            with (root / "rounds.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), first_rounds)
            self.assertEqual(first, 2 * 2 * 10)   # 2 algorithms x 2 budgets x 10 policy draws
            table = T.summary(root)
            self.assertIn("confidence vs top degree", table.columns)
            changed = dict(self.config, phase1_share=0.25)
            with self.assertRaises(ValueError):
                T.run(root, changed, cases=["toy"], matcher=neighbour_matcher)


if __name__ == "__main__":
    unittest.main()
