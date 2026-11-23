"""Checks of score-search fidelity, randomized selection and repeated-policy accounting."""

import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .. import strategy_comparison as C


class StrategyComparisonTests(unittest.TestCase):
    def setUp(self):
        self.config = C.default_config()
        self.config.update(reference_pool=24, best_of_n=24, search_steps=3, search_swaps=32,
                           search_starts=3, repeats={"small": 3, "large": 3},
                           solver_runs={"small": 2, "large": 2})
        self.config["cases"] = {"toy": {"get": {}, "budgets": [5]}}
        rng = np.random.default_rng(14)
        a = np.triu(rng.random((28, 28)) < .16, 1)
        self.graph = C.GraphA(a | a.T)
        self.eligible = np.arange(23)
        self.reference = C.F.G.draw_sets(rng, self.eligible, 5, 24)

    def test_incremental_score_matches_full_features_for_both_caps(self):
        current = self.reference[0]
        outside = np.setdiff1d(self.eligible, current)
        positions, choices = np.divmod(np.arange(len(current) * len(outside)), len(outside))
        swapped = np.repeat(current[None, :], len(positions), axis=0)
        swapped[np.arange(len(swapped)), positions] = outside[choices]
        swapped.sort(axis=1)
        for algorithm in C.ALGORITHMS:
            score = C.ERScore(self.graph, algorithm, self.reference, self.config["weights"][algorithm])
            np.testing.assert_allclose(score.swap_scores(current, positions, outside[choices]),
                                       score(swapped), atol=1e-12)

    def test_randomized_rules_are_valid_reproducible_and_diverse(self):
        for method in ("random", "degree_shortlist", *C.RULES, "top_harmonic"):
            results = []
            for seed in range(8):
                s = C.randomized_rule(method, self.graph, self.eligible, 5,
                                      np.random.default_rng(seed), self.config)
                self.assertEqual(len(set(s)), 5)
                self.assertTrue(set(s) <= set(self.eligible))
                again = C.randomized_rule(method, self.graph, self.eligible, 5,
                                          np.random.default_rng(seed), self.config)
                np.testing.assert_array_equal(s, again)
                results.append(tuple(s))
            self.assertGreater(len(set(results)), 1, method)

    def test_plan_preserves_policy_draws_and_paired_score_improvement(self):
        sets, members, norm = C.make_plan("toy", 5, self.graph, self.eligible, self.config)
        self.assertFalse(sets.seed_vertices.duplicated().any())
        for algorithm in C.ALGORITHMS:
            m = members[members.algorithm == algorithm]
            self.assertEqual(len(m), 1 + 3 * (len(C.METHODS) - 1))
            values = m.merge(sets[["candidate_id", f"{algorithm}__score_er"]], on="candidate_id")
            score = values.pivot(index="draw", columns="method", values=f"{algorithm}__score_er")
            self.assertTrue((score.best_found >= score.best_of_n - 1e-12).all())
            self.assertTrue(all(value > 0 for value in norm[algorithm]["scale"].values()))

    def test_singleton_pool_enumerates_all_eligible_vertices_and_plan_finishes(self):
        config = {**self.config, "reference_pool": 1000, "best_of_n": 1000}
        pool = C.selection_pool(np.random.default_rng(1), self.eligible, 1, 1000)
        np.testing.assert_array_equal(pool, self.eligible[:, None])
        sets, members, _ = C.make_plan("toy", 1, self.graph, self.eligible, config)
        self.assertTrue(all(len(json.loads(s)) == 1 for s in sets.seed_vertices))
        for algorithm in C.ALGORITHMS:
            score = C.ERScore(self.graph, algorithm, pool, config["weights"][algorithm])
            values = members[members.algorithm == algorithm].merge(sets, on="candidate_id")
            screened = values.loc[values.method == "best_of_n", f"{algorithm}__score_er"]
            self.assertTrue(np.isfinite(screened).all())
            np.testing.assert_allclose(screened, score(pool).max())

    def test_feasible_selection_pool_preserves_original_rng_stream(self):
        expected = C.F.G.draw_sets(np.random.default_rng(44), self.eligible, 5, 1000)
        actual = C.selection_pool(np.random.default_rng(44), self.eligible, 5, 1000)
        np.testing.assert_array_equal(actual, expected)

    def test_search_with_all_swaps_stops_at_a_single_swap_optimum(self):
        score = C.ERScore(self.graph, "faq", self.reference, self.config["weights"]["faq"])
        config = {**self.config, "search_swaps": 10000, "search_steps": 100}
        s, info = C.best_found(score, self.eligible, [self.reference[0]], np.random.default_rng(9), config)
        self.assertGreaterEqual(score(s)[0] + 1e-12, info["initial_score"])
        neighbours = []
        for i, v in itertools.product(range(5), np.setdiff1d(self.eligible, s)):
            t = s.copy(); t[i] = v
            neighbours.append(np.sort(t))
        self.assertLessEqual(float(score(np.stack(neighbours)).max()), score(s)[0] + 1e-12)

    def test_resume_deduplicates_runs_and_averaging_preserves_draw_frequency(self):
        config = self.config
        config["cases"] = {"toy": {"get": {}, "budgets": [5]}}
        a = self.graph.adjacency
        test = np.arange(23, 28)
        info = {"hash_a": "a", "hash_b": "b", "test_vertices": test.tolist()}
        context = (None, a, a, np.arange(len(a)), test, self.eligible, info)

        def outcome(algorithm, model, pair, seeds, test, seed, regularizer):
            return {"accuracy_test": float(seeds.sum() % 11) / 10, "accuracy": .5,
                    "converged": True, "error": ""}

        with tempfile.TemporaryDirectory() as tmp, patch.object(C.F, "load_pair", return_value=context), \
             patch.object(C.F.RM, "run_one", side_effect=outcome) as evaluate:
            root = Path(tmp)
            C.prepare(root, config, 1)
            C.collect_bank(root, "toy", 5, config, 1)
            called = evaluate.call_count
            C.collect_bank(root, "toy", 5, config, 1)
            self.assertEqual(evaluate.call_count, called)
            data = C.read(root)
            self.assertTrue(data["inventory"].complete.all())
            summary = C.summarize(data)
            for algorithm in C.ALGORITHMS:
                expected = data["draws"].query("algorithm == @algorithm and method == 'random'").accuracy.mean()
                actual = summary.query("algorithm == @algorithm and method == 'random'")["mean"].iloc[0]
                self.assertEqual(expected, actual)
            overlap = C.overlaps(data)
            same = overlap.query("left == 'top_degree' and right == 'top_degree'")
            self.assertTrue((same.shared_fraction == 1).all())
            # Remove one solver row: the whole algorithm/bank must be excluded, not averaged unevenly.
            path = root / "toy/k005/faq_runs.csv"
            frame = C.pd.read_csv(path)
            frame.iloc[:-1].to_csv(path, index=False)
            partial = C.read(root)
            self.assertNotIn("faq", set(partial["draws"].algorithm))
            C.collect_bank(root, "toy", 5, config, 1)
            self.assertEqual(evaluate.call_count, called + 1)
            self.assertTrue(C.read(root)["inventory"].complete.all())


if __name__ == "__main__":
    unittest.main()
