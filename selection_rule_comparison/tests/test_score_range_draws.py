"""Checks for score_range_draws with a stand-in matcher. Reads two banks of the saved score-range
collection and works on temporary copies; nothing in the repository's data folders is written.

    python -m unittest selection_rule_comparison.test_score_range_draws
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from selection_rule_comparison import b_replicates as BR
from selection_rule_comparison import landscape as L
from selection_rule_comparison import run_comparison as RC
from selection_rule_comparison import score_range_draws as D
from selection_rule_comparison import score_range_draws_analysis as DA

BANKS = ("synthetic/sparse_er/d08_rho0.70/percolation/k013/pair_000",
         "synthetic/paper_pa/d08_rho0.40/goat/k007/pair_000")


def fake_run_one(algorithm, model, pair, seeds, test, solver_seed, regularizer=None):
    """Deterministic stand-in for regime_map.run_one: fast, and depends on the set, the draw and the seed."""

    A, B, truth = pair
    rng = np.random.default_rng([solver_seed, int(np.sum(seeds)), int(B.sum())])
    accuracy = float(rng.random())
    return {"accuracy_test": accuracy, "accuracy": accuracy, "success": float(accuracy >= .5),
            "h0_accuracy": np.nan, "objective": np.nan, "n_iter": 1, "converged": True,
            "runtime_seconds": 0.0, "matched_fraction": np.nan, "error": ""}


class ScoreRangeDraws(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="score_range_draws_test_"))
        cls.source = cls.tmp / "source"
        settings = json.loads((D.SOURCE_DIR / "settings.json").read_text())
        settings["banks"] = [b for b in settings["banks"] if b["id"] in BANKS]
        cls.source.mkdir()
        (cls.source / "settings.json").write_text(json.dumps(settings))
        for bank in BANKS:
            shutil.copytree(D.SOURCE_DIR / bank, cls.source / bank)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def collect(self, output, *extra, hours="1"):
        with mock.patch.object(D.RM, "run_one", fake_run_one):
            D.main(["--source-dir", str(self.source), "--output-dir", str(output), "--jobs", "1",
                    "--last-draw", "4", "--hours", hours, *extra])

    def test_collect_resume_and_status(self):
        out = self.tmp / "out_main"
        self.collect(out)
        settings = json.loads((out / "settings.json").read_text())
        for task in settings["banks"]:
            for draw in (3, 4):
                path = D.runs_path(out, task, draw)
                self.assertEqual(len(D.complete_ids(path)), task["n_sets"])
                self.assertTrue((path.parent / f"draw_{draw:02d}.json").exists())
        sizes = {p: p.stat().st_size for p in out.rglob("*_runs.csv")}
        self.collect(out)  # resume: nothing left to do
        self.assertEqual(sizes, {p: p.stat().st_size for p in out.rglob("*_runs.csv")})
        table, _ = D.status(out, self.source)
        self.assertTrue((table.draws_complete == 5).all())

    def test_draw_matches_generator_and_seed_rule(self):
        out = self.tmp / "out_seed"
        self.collect(out)
        task = json.loads((out / "settings.json").read_text())["banks"][0]
        with np.load(self.source / task["id"] / "pair.npz") as saved:
            A, B, truth = saved["A"], saved["B"], saved["truth"]
        probability = L.b_probability(task["model"], task["mean_degree"], task["rho"], task["pair_seed"], A, B, truth)
        new_b, _, _ = BR.draw_b(A, probability, task["pair_seed"], 3)
        record = json.loads((out / task["id"] / "draw_03.json").read_text())
        self.assertEqual(record["hash_b"], RC._hash(new_b))
        rows = DA._complete_rows(D.runs_path(out, task, 3))
        self.assertEqual(set(rows.hash_b), {record["hash_b"]})
        self.assertEqual(len(set(rows.solver_seed)), 1)

    def test_time_limit_then_resume(self):
        out = self.tmp / "out_time"
        self.collect(out, hours="0.0000001")
        self.assertEqual(list(out.rglob("*_runs.csv")), [])
        self.collect(out)
        table, _ = D.status(out, self.source)
        self.assertTrue((table.runs_done == table.runs_planned).all())

    def test_cut_off_row_is_removed_on_resume(self):
        out = self.tmp / "out_cut"
        self.collect(out)
        task = json.loads((out / "settings.json").read_text())["banks"][0]
        path = D.runs_path(out, task, 4)
        lines = path.read_text().splitlines(keepends=True)
        path.write_text("".join(lines[:-1]) + lines[-1][: len(lines[-1]) // 2])  # last row cut in half
        self.assertEqual(len(D.complete_ids(path)), task["n_sets"] - 1)
        self.collect(out)
        self.assertEqual(len(D.complete_ids(path)), task["n_sets"])
        self.assertEqual(len(path.read_text().splitlines()), task["n_sets"] + 1)

    def test_changed_design_is_refused(self):
        out = self.tmp / "out_design"
        self.collect(out)
        with self.assertRaises(ValueError):
            with mock.patch.object(D.RM, "run_one", fake_run_one):
                D.main(["--source-dir", str(self.source), "--output-dir", str(out), "--jobs", "1",
                        "--last-draw", "5"])

    def test_analysis_uses_complete_draws_only(self):
        out = self.tmp / "out_analysis"
        self.collect(out)
        data, runs = DA.load(self.source, out)
        frame = DA.per_set(data, runs)
        self.assertTrue((frame.n_draws == 5).all())
        # Drop one set's draw-4 row: draw 4 is then incomplete for that bank, so it is left out there.
        task = json.loads((out / "settings.json").read_text())["banks"][1]
        partial = runs[~((runs.bank_id == task["id"]) & (runs.draw == 4) & (runs.candidate_id == 0))]
        frame = DA.per_set(data, partial)
        self.assertEqual(set(frame[frame.bank_id == task["id"]].n_draws), {4})
        self.assertEqual(set(frame[frame.bank_id != task["id"]].n_draws), {5})
        by_hand = runs[(runs.bank_id == task["id"]) & (runs.candidate_id == 0)].accuracy_test.mean()
        got = DA.per_set(data, runs)
        self.assertAlmostEqual(got[(got.bank_id == task["id"]) & (got.candidate_id == 0)]["mean"].iloc[0], by_hand)


if __name__ == "__main__":
    unittest.main()
