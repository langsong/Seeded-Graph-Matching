"""Fixed sparse ER seed-set selection benchmark."""

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
    name="sparse_er",
    config=replace(
        preset_config("sparse_er"), n_graph_pairs=1, rho=0.70,
        n_seeds=26, max_iter=100,
    ),
    graph_pair_seed=4192889773,
    pilot_success_rate=0.556,
)
PROBLEM_DIR = DEFAULT_PROBLEM_DIR / "sgm_multi_rng" / "graph_pairs"


def check_seed_set(seeds_1):
    return evaluate_seed_set(prepare_problem(PROBLEM, PROBLEM_DIR), seeds_1)


if __name__ == "__main__":
    run_problem_cli(PROBLEM, PROBLEM_DIR)
