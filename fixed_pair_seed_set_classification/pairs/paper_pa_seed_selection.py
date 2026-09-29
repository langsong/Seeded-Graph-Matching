"""Fixed preferential-attachment-plus-ER PAPER seed-set benchmark."""

from dataclasses import replace

from ..seed_selection_common import (
    FixedProblem,
    evaluate_seed_set,
    prepare_problem,
    preset_config,
    run_problem_cli,
)


PROBLEM = FixedProblem(
    name="paper_pa",
    config=replace(preset_config("paper_pa"), n_graph_pairs=1),
    graph_pair_seed=424242,
    pilot_success_rate=7 / 12,
)


def check_seed_set(seeds_1):
    return evaluate_seed_set(prepare_problem(PROBLEM), seeds_1)


if __name__ == "__main__":
    run_problem_cli(PROBLEM)
