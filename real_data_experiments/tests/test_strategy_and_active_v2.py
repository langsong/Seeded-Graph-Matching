"""Checks of experimental validity; run with python -m unittest real_data_experiments.tests.test_strategy_and_active_v2."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from .. import active_test as T
from .. import strategy_and_active_v2 as O
from .. import strategy_and_active_v2_analysis as N


class OvernightChecks(unittest.TestCase):
    def test_projection_accepts_ties_but_rejects_suboptimal_or_invalid(self):
        soft = np.array([[.4, .4, .2], [.4, .4, .2], [.2, .2, .6]])
        self.assertEqual(T.projection_check(soft, [0, 1, 2]), 1)
        self.assertEqual(T.projection_check(soft, [1, 0, 2]), 1)
        self.assertEqual(T.projection_check(soft, [2, 1, 0]), 0)
        self.assertEqual(T.projection_check(soft, [0, 0, 2]), 0)

    def test_modal_agreement_does_not_depend_on_best_run(self):
        permutations = np.array([[0, 1, 2], [1, 0, 2], [1, 0, 2]])
        np.testing.assert_allclose(T.modal_agreement(permutations), [2 / 3, 2 / 3, 1])

    def test_query_policies_cannot_see_unrevealed_truth_or_query_test(self):
        config = O.default_config(True)
        ctx = O.context("enron_160_162", 0, config)
        seeds = O.initial_seeds(ctx, "distinguishing", 3, 0, "test")
        state = {"seeds": seeds, "uncertainty": np.linspace(0, 1, ctx["n"]), "perm": np.arange(ctx["n"])}
        alternative = {**ctx, "truth": np.random.default_rng(2).permutation(ctx["n"])}
        for policy in O.POLICIES:
            a = O.select_query(policy, ctx, state, np.random.default_rng(1))
            b = O.select_query(policy, alternative, state, np.random.default_rng(1))
            self.assertEqual(a, b)
            self.assertIn(a, ctx["eligible"])
            self.assertNotIn(a, seeds)
        self.assertFalse(set(ctx["test"]) & set(seeds))
        pilot = O.context("enron_160_162", -1, config)
        self.assertFalse(np.array_equal(ctx["test"], pilot["test"]))

    def test_ensemble_resumes_constituent_runs_and_guard_uses_fresh_uncertainty(self):
        config = O.default_config(True)
        ctx = O.context("enron_160_162", 0, config)
        seeds = np.asarray(ctx["eligible"][:3])
        calls = []

        def fake(ctx, seeds, algorithm, solver_seed, perturbation=0):
            calls.append(solver_seed)
            perm = T.correct(np.random.default_rng(solver_seed).permutation(ctx["n"]), seeds, ctx["truth"])
            return {"perm": perm, "confidence": np.ones(ctx["n"]), "soft_check": 1.,
                    "runtime_seconds": 0., "converged": True, "warnings": []}

        with tempfile.TemporaryDirectory() as folder, patch.object(O, "match_once", fake):
            path = Path(folder) / "state.npz"
            s = O.ensemble(path, ctx, seeds, "faq", ("test",), 3, time.monotonic() + 10,
                           incumbent=ctx["truth"])
            np.testing.assert_allclose(s["uncertainty"], 1 - T.modal_agreement(s["permutations"]))
            self.assertEqual(len(calls), 3)
            path.unlink()  # emulate interruption after constituents, before aggregate commit
            again = O.ensemble(path, ctx, seeds, "faq", ("test",), 3, time.monotonic() + 10,
                               incumbent=ctx["truth"])
            self.assertEqual(len(calls), 3)
            np.testing.assert_array_equal(s["perm"], again["perm"])
            self.assertAlmostEqual(s["accuracy_all"],
                                   (len(seeds) + (ctx["n"] - len(seeds)) * s["accuracy_nonseed"]) / ctx["n"])

    def test_equal_partition_weight_and_paired_random_baseline(self):
        rows = []
        for split, draws, value in [(0, 3, .2), (1, 1, .8)]:
            for draw in range(draws):
                for method, acc in [("random", .1), ("distinguishing", value)]:
                    rows.append(dict(case="x", algorithm="faq", budget=10, split=split, draw=draw,
                                     method=method, accuracy_test=acc, score_er=0))
        data = {"draws": pd.DataFrame(rows), "calibration": {"x": {"faq": {"pilot_means": {"10": .3}}}}}
        s = N.summarize_strategy(data).set_index("method")
        self.assertAlmostEqual(s.loc["distinguishing", "mean"], .5)
        self.assertAlmostEqual(s.loc["distinguishing", "gain"], .4)

    def test_calibration_keeps_range_when_no_intermediate_regime(self):
        rows = [{"budget": k, "accuracy_test": .01} for k in [5, 10, 20, 40, 60]]
        plan = O.choose_budgets(rows, 3)
        self.assertEqual(plan["budgets"], [5, 20, 60])
        self.assertFalse(plan["has_intermediate"])


if __name__ == "__main__":
    unittest.main()
