"""Calibrated floor-free Pareto-Chung-Lu seed-set selection benchmark."""

from dataclasses import replace

from ..seed_selection_common import (
    DEFAULT_PROBLEM_DIR,
    FixedProblem,
    evaluate_seed_set,
    prepare_problem,
    preset_config,
    run_problem_cli,
)


PROBLEM = FixedProblem(
    name="chung_lu",
    config=replace(
        preset_config("chung_lu"),
        chung_lu_alpha=2.0,
        # Expected mean degree 5.7 after the structural cutoff.
        chung_lu_weight_scale=1.9260087733615983,
        chung_lu_probability_floor=0.0,
        n_graph_pairs=1,
        n_seeds=17,
        max_iter=100,
    ),
    graph_pair_seed=3165569293,
    pilot_success_rate=0.5625,
)
PROBLEM_DIR = DEFAULT_PROBLEM_DIR / "sgm_multi_rng" / "graph_pairs"


def check_seed_set(seeds_1):
    return evaluate_seed_set(prepare_problem(PROBLEM, PROBLEM_DIR), seeds_1)


if __name__ == "__main__":
    run_problem_cli(PROBLEM, PROBLEM_DIR)
