"""Round-1 settings for the seed-selection rule comparison.

Graph settings copy the collections the 20 Oct predictions came from:
percolation uses the 7 Oct fixed-pair settings, and FAQ and GOAT use the
13 Oct settings (the two SGM solvers share the same graph pairs).
``check_training_presets`` verifies the copies against the training modules.
"""

from __future__ import annotations

from dataclasses import replace

from experiment_common.graphs import ExperimentConfig, preset_config

ALGORITHMS = ("percolation", "goat", "faq")
MODELS = ("sparse_er", "paper_pa", "chung_lu")
PHASES = ("pilot", "main")

# Graph families. FAQ and GOAT share a family, so pair i is the same graph pair
# for both solvers.
GRAPH_FAMILY = {"percolation": "percolation_2026-10-07", "goat": "sgm_2026-10-13", "faq": "sgm_2026-10-13"}


def graph_config(algorithm: str, model: str) -> ExperimentConfig:
    """Graph-generation settings for one algorithm and graph model."""

    if GRAPH_FAMILY[algorithm] == "percolation_2026-10-07":
        if model == "sparse_er":
            return replace(preset_config("sparse_er"), er_probability=0.020, rho=0.70, n_graph_pairs=1)
        if model == "paper_pa":
            return replace(preset_config("paper_pa"), paper_pa_alpha=0.0, paper_pa_beta=1.0,
                           paper_pa_probability=0.0167, rho=0.62, n_graph_pairs=1)
        if model == "chung_lu":
            return replace(preset_config("chung_lu"), chung_lu_alpha=2.0,
                           chung_lu_weight_scale=4.076194702720809, chung_lu_probability_floor=0.0,
                           rho=0.70, n_graph_pairs=1)
    else:
        if model == "sparse_er":
            return replace(preset_config("sparse_er"), er_probability=16 / 599, rho=0.50,
                           max_iter=100, n_graph_pairs=1)
        if model == "paper_pa":
            return replace(preset_config("paper_pa"), paper_pa_alpha=0.0, paper_pa_beta=1.0,
                           paper_pa_probability=0.023378, rho=0.40, max_iter=100, n_graph_pairs=1)
        if model == "chung_lu":
            return replace(preset_config("chung_lu"), chung_lu_alpha=2.0, chung_lu_weight_scale=5.452064,
                           chung_lu_probability_floor=0.0, rho=0.45, max_iter=100, n_graph_pairs=1)
    raise ValueError(f"Unknown algorithm/model: {algorithm}/{model}")


# Solver settings (same as the training collections).
PERCOLATION = {"threshold": 2, "expand_when_stuck": True, "algorithm_seed": 2026}
SGM = {"max_iter": 100, "tol": 0.01, "transport_tol": 0.05, "transport_max_iter": 5000}
GOAT_REGULARIZER = {"sparse_er": 100, "paper_pa": 300, "chung_lu": 300}
FAQ_MAX_RESTARTS = 3  # stop at the first success; equals best-of-3 because the objective flags success

# Success labels. Percolation uses the upper mode (accuracy >= 0.5, recommended
# on 7 Oct); accuracy >= 0.9 is recorded for every algorithm as well.
SUCCESS_THRESHOLD = {"percolation": 0.5, "goat": 0.9, "faq": 0.9}

# Strategies per algorithm and phase (see strategies.py).
STRATEGIES = {
    "pilot": {algorithm: ("random", "top_degree") for algorithm in ALGORITHMS},
    "main": {
        "percolation": ("random", "top_degree", "top_harmonic", "clustered", "spread", "distinguishing"),
        "goat": ("random", "top_degree", "clustered", "spread", "distinguishing", "hub_free_distinguishing"),
        "faq": ("random", "top_degree", "clustered", "spread", "distinguishing"),
    },
}

# Budgets (number of seeds). Pilot grids are wide and shared by the three
# models. Main grids were chosen per model from the 20 Oct pilot (10 pairs,
# random vs top degree): they start where both baselines still fail, so a
# strategy that beats top degree can show it, and end where random mostly
# succeeds. Override with --budgets.
BUDGETS = {
    "pilot": {
        "percolation": (1, 2, 3, 4, 5, 6, 8, 10, 12),
        "goat": (2, 4, 6, 8, 10, 12, 14, 16, 20),
        "faq": (4, 8, 12, 16, 20, 24, 28, 32, 40),
    },
    "main": {
        "percolation": {"sparse_er": (3, 4, 5, 6, 8, 10), "paper_pa": (2, 3, 4, 6, 8, 10),
                        "chung_lu": (2, 3, 4, 5, 6, 8)},
        "goat": {"sparse_er": (6, 8, 9, 10, 12, 16), "paper_pa": (6, 8, 9, 10, 12, 16),
                 "chung_lu": (4, 6, 8, 10, 14, 20)},
        "faq": {"sparse_er": (16, 18, 20, 22, 24, 28), "paper_pa": (14, 16, 18, 20, 24, 32),
                "chung_lu": (12, 15, 18, 21, 24, 28)},
    },
}


def default_budgets(phase: str, algorithm: str, model: str) -> tuple[int, ...]:
    grid = BUDGETS[phase][algorithm]
    return tuple(grid[model] if isinstance(grid, dict) else grid)


# Pairs per model. The main run is a proof of concept (20 pairs); rerun the same
# command with a larger --pairs later to add pairs to the same results.
N_PAIRS = {
    "pilot": {"percolation": 10, "goat": 10, "faq": 10},
    "main": {"percolation": 20, "goat": 20, "faq": 20},
}

# Random-number streams. Pair seeds depend on the graph family, model, phase
# and pair index (not on the solver), so pilot and main pairs differ and FAQ and
# GOAT share pairs. Solver seeds depend on the pair, budget and restart only, so
# every strategy sees the same solver randomness (common random numbers).
PAIR_SEED_BASE = 20261020
STRATEGY_SEED_BASE = 20261021
SOLVER_SEED_BASE = 20261022
FAMILY_CODES = {"percolation_2026-10-07": 1, "sgm_2026-10-13": 2}
MODEL_CODES = {"sparse_er": 1, "paper_pa": 2, "chung_lu": 3}
PHASE_CODES = {"pilot": 1, "main": 2}

# Pair seeds used for training collections and calibrations; never reused.
KNOWN_PAIR_SEEDS = frozenset({
    4192889773, 424242, 3165569293, 18472911, 37051993,   # percolation 7 Oct, plus replication seeds
    88424077, 7301002, 12672911, 7301001,                  # 13 Oct SGM calibration and collection
    4192889786, 8675309,                                   # RNG diagnostics
})

HUB_COUNT = 10        # hubs removed from the pool by hub_free_distinguishing
POOL_FACTOR = 2       # degree-first rules choose seed j from the 2j highest-degree unused vertices


def check_training_presets(algorithm: str, model: str) -> str:
    """Compare graph settings with the training collection's preset.

    Returns a short status string. Raises if the settings differ. Importing
    the training modules needs the full project environment (graspologic,
    networkx); if that fails, the check is skipped with a note.
    """

    fields = ("graph_model", "vertex_count", "er_probability", "paper_pa_alpha", "paper_pa_beta",
              "paper_pa_probability", "effective_chung_lu_alpha", "effective_chung_lu_weight_scale",
              "effective_chung_lu_probability_floor", "rho")
    try:
        if GRAPH_FAMILY[algorithm] == "percolation_2026-10-07":
            from fixed_pair_seed_set_classification.percolation_data import PROBLEMS
            reference = PROBLEMS[model].config
        else:
            from fixed_pair_seed_set_classification.candidate_data import GOAT_PROBLEMS
            reference = GOAT_PROBLEMS[model].config
    except Exception as error:  # noqa: BLE001 - any import problem only skips the check
        return f"training preset check skipped ({type(error).__name__}: {error})"
    ours = graph_config(algorithm, model)
    differences = [field for field in fields if getattr(ours, field) != getattr(reference, field)]
    if differences:
        raise ValueError(f"{algorithm}/{model} settings differ from the training preset: {differences}")
    return "graph settings match the training preset"
