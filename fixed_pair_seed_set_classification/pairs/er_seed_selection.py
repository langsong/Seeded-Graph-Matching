"""Fixed mean-degree-16 ER benchmarks for FAQ and GOAT."""

from dataclasses import replace

from ..seed_selection_common import (
    SGM_COLLECTION_DIR,
    FixedProblem,
    evaluate_seed_set,
    prepare_problem,
    preset_config,
    run_problem_cli,
)


PROBLEM = FixedProblem(
    name="sparse_er",
    config=replace(
        preset_config("sparse_er"), n_graph_pairs=1, er_probability=16 / 599,
        rho=0.50, n_seeds=29, max_iter=100,
    ),
    graph_pair_seed=88424077,
    pilot_success_rate=8 / 12,
)
GOAT_PROBLEM = replace(PROBLEM, config=replace(PROBLEM.config, n_seeds=12), pilot_success_rate=7 / 12)
GOAT_REGULARIZER = 100
PROBLEM_DIR = SGM_COLLECTION_DIR / "graph_pairs" / PROBLEM.name / f"pair_seed_{PROBLEM.graph_pair_seed}"


def check_seed_set(seeds_1, *, solver="faq"):
    if solver not in ("faq", "goat"):
        raise ValueError("solver must be 'faq' or 'goat'.")
    spec = GOAT_PROBLEM if solver == "goat" else PROBLEM
    return evaluate_seed_set(prepare_problem(spec, PROBLEM_DIR), seeds_1,
                             transport=solver == "goat", transport_regularizer=GOAT_REGULARIZER)


if __name__ == "__main__":
    run_problem_cli(PROBLEM, PROBLEM_DIR, goat_spec=GOAT_PROBLEM, transport_regularizer=GOAT_REGULARIZER)
