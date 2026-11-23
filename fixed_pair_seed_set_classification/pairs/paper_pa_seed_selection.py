"""Fixed mean-degree-16 PAPER benchmarks for FAQ and GOAT."""

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
    name="paper_pa",
    config=replace(
        preset_config("paper_pa"), n_graph_pairs=1,
        paper_pa_alpha=0.0, paper_pa_beta=1.0, paper_pa_probability=0.023378,
        rho=0.40, n_seeds=31, max_iter=100,
    ),
    graph_pair_seed=7301002,
    pilot_success_rate=7 / 12,
)
GOAT_PROBLEM = replace(PROBLEM, config=replace(PROBLEM.config, n_seeds=12), pilot_success_rate=5 / 12)
GOAT_REGULARIZER = 300
PROBLEM_DIR = SGM_COLLECTION_DIR / "graph_pairs" / PROBLEM.name / f"pair_seed_{PROBLEM.graph_pair_seed}"


def check_seed_set(seeds_1, *, solver="faq"):
    if solver not in ("faq", "goat"):
        raise ValueError("solver must be 'faq' or 'goat'.")
    spec = GOAT_PROBLEM if solver == "goat" else PROBLEM
    return evaluate_seed_set(prepare_problem(spec, PROBLEM_DIR), seeds_1,
                             transport=solver == "goat", transport_regularizer=GOAT_REGULARIZER)


if __name__ == "__main__":
    run_problem_cli(PROBLEM, PROBLEM_DIR, goat_spec=GOAT_PROBLEM, transport_regularizer=GOAT_REGULARIZER)
