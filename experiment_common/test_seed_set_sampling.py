"""Checks for score-range coverage, controls and fixed-size candidate paths."""

import itertools
import unittest

import numpy as np
import pandas as pd

from experiment_common.seed_set_sampling import select_bank, swap_paths


class SeedBankSamplingTests(unittest.TestCase):
    def test_paths_keep_budget_and_reach_endpoints(self):
        start, target = np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])
        paths = swap_paths([start], [target], np.random.default_rng(1), steps=4)
        self.assertEqual(len(paths), 4)
        self.assertTrue(all(len(s) == len(np.unique(s)) == 4 for s in paths))
        self.assertTrue(np.array_equal(paths[-1], target))
        self.assertEqual([len(np.intersect1d(s, target)) for s in paths], [1, 2, 3, 4])

    def test_controls_are_preserved_and_constructed_range_is_covered(self):
        arrays = np.asarray(list(itertools.combinations(range(12), 3)))
        rng = np.random.default_rng(7)
        rng.shuffle(arrays)
        degree = arrays.sum(axis=1).astype(float)
        values = pd.DataFrame({"degree": degree, "pair_mass": arrays[:, 0],
                               "coverage": arrays[:, 2] / 12, "uniqueness": np.ones(len(arrays)),
                               "score": degree, "prediction": degree ** 2})
        args = (arrays, values, ["score", "prediction"], {"degree_rule": int(degree.argmax())}, 60, 10)
        selected, members, coverage = select_bank(*args, np.random.default_rng(15))
        self.assertEqual(len(selected), 60)
        self.assertEqual(selected.seed_vertices.nunique(), 60)
        self.assertEqual((selected.family == "random").sum(), 10)
        self.assertEqual(members[members.family == "rule"].detail.tolist(), ["degree_rule"])
        occupied = coverage[coverage.available > 0]
        self.assertTrue((occupied.selected > 0).all())
        again, _, _ = select_bank(*args, np.random.default_rng(15))
        pd.testing.assert_frame_equal(selected, again)

    def test_constant_scores_do_not_create_false_range(self):
        arrays = np.asarray(list(itertools.combinations(range(8), 2)))
        values = pd.DataFrame({c: np.ones(len(arrays)) for c in
                               ["degree", "pair_mass", "coverage", "uniqueness", "score"]})
        selected, _, coverage = select_bank(arrays, values, ["score"], {}, 12, 3, np.random.default_rng(2))
        self.assertEqual(len(selected), 12)
        self.assertTrue(coverage.constant_score.all())
        self.assertEqual(coverage.available.tolist(), [28, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(coverage.selected.sum(), 12)

    def test_identical_rule_sets_keep_both_memberships(self):
        arrays = np.asarray(list(itertools.combinations(range(8), 2)))
        values = pd.DataFrame({c: np.arange(len(arrays), dtype=float) for c in
                               ["degree", "pair_mass", "coverage", "uniqueness", "score"]})
        selected, members, _ = select_bank(arrays, values, ["score"], {"degree": 0, "harmonic": 0},
                                           12, 3, np.random.default_rng(2))
        rules = members[members.family == "rule"]
        self.assertEqual(rules.detail.tolist(), ["degree", "harmonic"])
        self.assertEqual(rules.candidate_id.nunique(), 1)
        self.assertEqual(selected.family.eq("random").sum(), 3)


if __name__ == "__main__":
    unittest.main()
