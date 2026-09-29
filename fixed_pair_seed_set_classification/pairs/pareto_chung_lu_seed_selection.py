"""Fixed Pareto–Chung–Lu benchmark, historically labeled ``paper``."""

from dataclasses import replace

from ..seed_selection_common import (
    FixedProblem,
    evaluate_seed_set,
    prepare_problem,
    preset_config,
    run_problem_cli,
)


PROBLEM = FixedProblem(
    name="paper",
    config=replace(preset_config("paper"), n_graph_pairs=1),
    graph_pair_seed=3165569293,
    source_pair_id=18,
    pilot_success_rate=0.48,
)


def check_seed_set(seeds_1):
    return evaluate_seed_set(prepare_problem(PROBLEM), seeds_1)


if __name__ == "__main__":
    run_problem_cli(PROBLEM)
