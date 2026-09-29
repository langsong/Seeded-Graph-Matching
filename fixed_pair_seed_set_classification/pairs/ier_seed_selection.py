"""Fixed IER seed-set selection benchmark."""

from dataclasses import replace

from ..seed_selection_common import (
    FixedProblem,
    evaluate_seed_set,
    prepare_problem,
    preset_config,
    run_problem_cli,
)


PROBLEM = FixedProblem(
    name="ier",
    config=replace(preset_config("ier"), n_graph_pairs=1),
    graph_pair_seed=808907875,
    source_pair_id=15,
    pilot_success_rate=0.52,
)


def check_seed_set(seeds_1):
    return evaluate_seed_set(prepare_problem(PROBLEM), seeds_1)


if __name__ == "__main__":
    run_problem_cli(PROBLEM)
