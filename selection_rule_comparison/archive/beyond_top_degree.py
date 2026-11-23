"""Beyond top degree: does a seed feature help on its own, away from top degree?

Round 1's feature rules pick seed j among the 2j highest-degree unused nodes, so
their seeds stay close to top degree. Here each feature is followed along a path
away from top degree: seed j is picked among the m*j highest-degree unused nodes,
for m = 2 (round 1), 8 and "all nodes" (percolation adds 4 and 16). Top degree is
where every path starts.

Objectives (greedy; ties go to the higher-degree node, as in round 1)
    pair_mass   pairs of seeds that share a non-seed neighbour (round 1's
                "clustered" target; pair_mass_2j is round 1's clustered rule).
    uniqueness  how well the seeds tell non-seeds apart by their fingerprint
                (capped distances to the seeds).
                  GOAT and percolation (cap 3): entropy of the fingerprints, i.e.
                    how evenly the seeds split nodes into fingerprint groups.
                  FAQ (cap 2; cap 3 is saturated at FAQ budgets): number of unique
                    fingerprints, with the entropy breaking ties.
                Chosen by a graph-A check (28 Oct): when every node is allowed,
                round 1's unique-count target ends with less uniqueness than at 2j.
    model       the frozen logistic score (degree, harmonic, pair mass,
                uniqueness r3), fitted on the fixed-pair random sets with
                pilot_new_rules.fit_weights. GOAT and percolation only.
    witness     percolation only, no fitted weights: the expected number of
                non-seeds with at least two seed edges that survive in B, the sum
                over v of P(Binomial(w_v, q) >= 2). w_v is v's number of seed
                neighbours and q = rho + (1 - rho) * mean degree / (n - 1) is the
                chance that an edge of A is kept in B under the graph settings.

Look-alikes
    For each all-node rule: at every position, a random unused node whose degree
    is as close as possible to that rule's seed (the same degree when one is free).
    Rule minus look-alike is what the feature adds at that degree level, and
    look-alike minus random is what the degree level itself buys.

Arms (existing pairs only, so earlier results are reused pair by pair)
    percolation  round-1 pairs 0-19 (ER, PAPER, Chung-Lu), plus the 50 fresh
                 borderline pairs 100-149 (ER, PAPER), where random is also run
    goat         round-1 pairs at round 1's regulariser (ER 100, PAPER 300, Chung-Lu 300)
    goat_other   the same pairs at the other regulariser (ER 300, PAPER 100,
                 Chung-Lu 100); top degree, random and hub-free top degree are rerun there
    faq          round-1 pairs, ER and PAPER. On Chung-Lu neither feature moves away
                 from top degree while improving, so FAQ skips it.

Checks: top degree is rerun on pairs 0, 1, 100 and 101 of the round-1 settings and
must reproduce the stored results; every pair is checked against its stored hashes.

Run from the repository root (ds environment):

    python -m selection_rule_comparison.beyond_top_degree                     # every arm
    python -m selection_rule_comparison.beyond_top_degree --arms percolation goat

or call run_all() from beyond_top_degree.ipynb. Each pair is saved when it finishes,
so rerunning resumes and only runs what is missing.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache"))

import numpy as np
import pandas as pd

from experiment_common.graph_checks import first_passing_pair
from experiment_common.graphs import generate_graph_pair
from experiment_common.provenance import thread_limit

from . import hub_test as H
from . import pilot_new_rules as P
from .. import run_comparison as rc
from .. import settings as S
from ..strategies import STRATEGY_CODES, GraphA, degree_first, seed_set_features

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]  # selection_rule_comparison/ (this file is in archive/)
DATA_DIR = EXPERIMENT_DIR / "data"
ROUND1_ROOT = DATA_DIR / "round1" / "main"
BORDERLINE_DIR = DATA_DIR / "borderline" / "percolation"
HUB_TEST_DIR = DATA_DIR / "hub_test"
PILOT_DIR = DATA_DIR / "pilot_new_rules"
OUTPUT_DIR = DATA_DIR / "beyond_top_degree"

ARMS = ("percolation", "goat", "goat_other", "faq")
ARM_ALGORITHM = {"percolation": "percolation", "goat": "goat", "goat_other": "goat", "faq": "faq"}
ALL_MODELS = ("sparse_er", "paper_pa", "chung_lu")
ARM_MODELS = {"percolation": ALL_MODELS, "goat": ALL_MODELS, "goat_other": ALL_MODELS,
              "faq": ("sparse_er", "paper_pa")}
OTHER_REGULARIZER = {"sparse_er": 300, "paper_pa": 100, "chung_lu": 100}
N_ROUND1_PAIRS = 20
FRESH_FIRST = 100                                  # borderline_check's fresh percolation pairs 100-149
FRESH_PAIRS = range(FRESH_FIRST, FRESH_FIRST + 50)
FRESH_MODELS = ("sparse_er", "paper_pa")
CHECK_PAIRS = (0, 1, FRESH_FIRST, FRESH_FIRST + 1)  # top degree is rerun here and must match

POOL_FACTORS = {"2j": 2, "4j": 4, "8j": 8, "16j": 16, "all": None}
POOL_ORDER = ("1j", "2j", "4j", "8j", "16j", "all")   # "1j" stands for top degree
PATHS = {
    "pair_mass": ("pair_mass_2j", "pair_mass_4j", "pair_mass_8j", "pair_mass_16j", "pair_mass_all"),
    "uniqueness": ("uniqueness_2j", "uniqueness_8j", "uniqueness_all"),
    "model": ("model_2j", "model_8j"),
}

# Random streams. Except for random and the look-alikes they only break degree ties.
PAIR_MASS_CODE = STRATEGY_CODES["clustered"]   # round 1's stream, so pair_mass_2j is round 1's clustered
MODEL_CODE = 100                               # the pilot's stream, so model_2j is the pilot's weighted_2j
UNIQUENESS_CODE = 110
WITNESS_CODE = 120
LOOKALIKE_SOURCES = ("pair_mass_all", "uniqueness_all")
LOOKALIKE_CODE = 300                           # + 10 * source index + draw


def _lookalikes(sources, draws: int) -> list[str]:
    return [f"lookalike_{source}_{draw}" for source in sources for draw in range(1, draws + 1)]


_PERCOLATION_RULES = ["pair_mass_4j", "pair_mass_8j", "pair_mass_16j", "pair_mass_all",
                      "uniqueness_2j", "uniqueness_8j", "uniqueness_all", "model_2j", "model_8j", "witness_all",
                      *_lookalikes(LOOKALIKE_SOURCES, 2)]
_GOAT_RULES = ["pair_mass_8j", "pair_mass_all", "uniqueness_2j", "uniqueness_8j", "uniqueness_all",
               "model_2j", "model_8j", "lookalike_pair_mass_all_1", *_lookalikes(["uniqueness_all"], 2)]
RULES = {
    "percolation": {model: list(_PERCOLATION_RULES) for model in ALL_MODELS},
    "goat": {
        "sparse_er": list(_GOAT_RULES),
        "paper_pa": list(_GOAT_RULES),
        # Chung-Lu: maximising pair mass stays on the hubs, so only uniqueness leaves top degree.
        "chung_lu": ["uniqueness_2j", "uniqueness_8j", "uniqueness_all", "model_2j", "model_8j",
                     *_lookalikes(["uniqueness_all"], 2)],
    },
    "goat_other": {
        "sparse_er": ["top_degree", "random", "pair_mass_all", "uniqueness_all",
                      *_lookalikes(LOOKALIKE_SOURCES, 1)],
        "paper_pa": ["top_degree", "random", "hub_free_top_degree", "pair_mass_all", "uniqueness_all",
                     *_lookalikes(LOOKALIKE_SOURCES, 1)],
        "chung_lu": ["top_degree", "random", "hub_free_top_degree", "uniqueness_all",
                     *_lookalikes(["uniqueness_all"], 1)],
    },
    "faq": {model: ["pair_mass_all", "uniqueness_2j", "uniqueness_8j", "uniqueness_all",
                    *_lookalikes(LOOKALIKE_SOURCES, 1)] for model in ("sparse_er", "paper_pa")},
}

# Earlier results reused for the same pairs (round 1, hub_test and borderline_check).
REUSED_ROUND1 = ("top_degree", "random", "clustered", "spread", "distinguishing", "hub_free_distinguishing",
                 "top_harmonic")
BEST_ROUND1 = {("percolation", m): "clustered" for m in ALL_MODELS}
BEST_ROUND1.update({("goat", "sparse_er"): "distinguishing", ("goat", "paper_pa"): "hub_free_distinguishing",
                    ("goat", "chung_lu"): "hub_free_distinguishing", ("faq", "sparse_er"): "distinguishing",
                    ("faq", "paper_pa"): "distinguishing"})

UNIQUENESS_FEATURE = {"percolation": "f_signature_r3", "goat": "f_signature_r3", "faq": "f_signature_r2"}
FEATURE_COLUMNS = ("f_degree", "f_harmonic", "f_coverage", "f_pair_mass", "f_signature_r2", "f_signature_r3")
SECONDS_PER_RUN = {"percolation": 0.6, "goat": 3.7, "faq": 7.5}  # wall time per run with 3 jobs (round 1, borderline)
FAQ_RUNS_PER_BUDGET = 2.0                      # restarts used on average (round 1)

MODEL_LABELS = {"sparse_er": "ER", "paper_pa": "PAPER", "chung_lu": "Chung-Lu"}
ALGORITHM_LABELS = {"percolation": "Percolation", "goat": "GOAT", "faq": "SGM (FAQ)"}
ARM_LABELS = {"percolation": "Percolation", "goat": "GOAT", "goat_other": "GOAT, other regulariser",
              "faq": "SGM (FAQ)"}
OBJECTIVE_LABELS = {"pair_mass": "pair mass", "uniqueness": "uniqueness", "model": "frozen model",
                    "witness": "witness score"}
POOL_LABELS = {"1j": "top degree", "2j": "2j", "4j": "4j", "8j": "8j", "16j": "16j", "all": "all nodes"}
FIXED_LABELS = {"top_degree": "top degree", "random": "random", "clustered": "clustered (round 1)",
                "spread": "spread (round 1)", "distinguishing": "distinguishing (round 1)",
                "hub_free_distinguishing": "hub-free distinguishing", "hub_free_top_degree": "hub-free top degree",
                "top_harmonic": "top harmonic", "top_degree_check": "top degree (check)"}


def _now() -> str:
    return datetime.datetime.now().astimezone().strftime("%H:%M:%S")


def _flag(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def regularizer(arm: str, model: str) -> int:
    """GOAT's transport regulariser in this arm (0 for percolation and FAQ)."""

    if arm == "goat":
        return int(S.GOAT_REGULARIZER[model])
    if arm == "goat_other":
        return int(OTHER_REGULARIZER[model])
    return 0


def describe_rule(rule: str) -> dict:
    """Family (objective), pool and look-alike source of a rule name."""

    if rule.startswith("lookalike_"):
        rest = rule[len("lookalike_"):]
        last = rest.rsplit("_", 1)
        source, draw = (last[0], int(last[1])) if last[-1].isdigit() else (rest, None)
        return {"family": "lookalike", "pool": "all", "source": source, "draw": draw}
    head, _, tail = rule.rpartition("_")
    if head in ("pair_mass", "uniqueness", "model", "witness") and tail in POOL_FACTORS:
        return {"family": head, "pool": tail, "source": None, "draw": None}
    if rule == "top_degree":
        return {"family": "top_degree", "pool": "1j", "source": None, "draw": None}
    return {"family": rule, "pool": None, "source": None, "draw": None}


def rule_label(rule: str) -> str:
    """Plain label for plots and tables."""

    info = describe_rule(rule)
    if info["family"] == "lookalike":
        draw = f" (draw {info['draw']})" if info["draw"] else ""
        return f"look-alike of {rule_label(info['source'])}{draw}"
    if info["family"] in OBJECTIVE_LABELS:
        return f"{OBJECTIVE_LABELS[info['family']]}, {POOL_LABELS[info['pool']]}"
    return FIXED_LABELS.get(rule, rule.replace("_", " "))


# --------------------------------------------------------------------------
# Greedy targets on graph A
# --------------------------------------------------------------------------

class FingerprintTarget:
    """Non-seeds grouped by their capped distances to the seeds.

    score = "entropy": entropy of the group sizes (how evenly the seeds split the nodes).
    score = "count_entropy": number of unique fingerprints, then the entropy.
    """

    def __init__(self, graph: GraphA, cap: int, score: str = "entropy"):
        self.graph, self.cap, self.kind = graph, cap, score
        self.distances = graph.capped[cap].astype(np.int64)
        self.in_set = np.zeros(graph.n, dtype=bool)
        self.labels = np.zeros(graph.n, dtype=np.int64)

    def score(self, vertex: int) -> tuple:
        keys = self.labels * (self.cap + 1) + self.distances[:, vertex]
        mask = ~self.in_set
        mask[vertex] = False
        _, counts = np.unique(keys[mask], return_counts=True)
        share = counts / counts.sum()
        entropy = round(float(-(share * np.log(share)).sum()), 10)  # rounded so exact ties stay ties
        if self.kind == "entropy":
            return (entropy,)
        return (int((counts == 1).sum()), entropy)

    def add(self, vertex: int) -> None:
        keys = self.labels * (self.cap + 1) + self.distances[:, vertex]
        _, self.labels = np.unique(keys, return_inverse=True)
        self.labels = self.labels.astype(np.int64)
        self.in_set[vertex] = True


class WitnessTarget:
    """Expected number of non-seeds with at least two seed edges that survive in B."""

    def __init__(self, graph: GraphA, keep: float):
        self.graph = graph
        self.in_set = np.zeros(graph.n, dtype=bool)
        self.w = np.zeros(graph.n, dtype=np.int64)
        k = np.arange(int(graph.degree.max()) + 3)
        miss = 1.0 - keep
        self.ready = 1.0 - miss ** k - k * keep * miss ** np.maximum(k - 1, 0)  # P(Binomial(k, keep) >= 2)
        self.ready[:2] = 0.0
        self.total = 0.0

    def score(self, vertex: int) -> tuple:
        neighbours = self.graph.neighbors[vertex]
        neighbours = neighbours[~self.in_set[neighbours]]
        gain = float((self.ready[self.w[neighbours] + 1] - self.ready[self.w[neighbours]]).sum())
        return (round(self.total - float(self.ready[self.w[vertex]]) + gain, 10),)

    def add(self, vertex: int) -> None:
        self.in_set[vertex] = True
        self.w[self.graph.neighbors[vertex]] += 1
        self.total = float(self.ready[self.w[~self.in_set]].sum())


def uniqueness_target(algorithm: str, graph: GraphA) -> FingerprintTarget:
    if algorithm == "faq":
        return FingerprintTarget(graph, cap=2, score="count_entropy")
    return FingerprintTarget(graph, cap=3, score="entropy")


def witness_keep(model: str, graph: GraphA) -> float:
    """Chance that an edge of A is kept in B: P(B | A) = P + rho (1 - P), P the edge density."""

    rho = float(S.graph_config("percolation", model).rho)
    density = float(graph.degree.mean()) / (graph.n - 1)
    return rho + (1.0 - rho) * density


def pool_greedy(graph: GraphA, k_max: int, rng: np.random.Generator, target, pool_factor: int | None) -> np.ndarray:
    """Seed j maximises the target among the pool_factor*j highest-degree unused nodes (all if None).

    The same loop as strategies.degree_first: ties go to the node earlier in the degree order.
    """

    by_degree = [int(v) for v in graph.degree_order(rng)]
    chosen: list[int] = []
    chosen_set: set[int] = set()
    for step in range(1, k_max + 1):
        unused = [v for v in by_degree if v not in chosen_set]
        pool = unused if pool_factor is None else unused[: pool_factor * step]
        best = max(range(len(pool)), key=lambda i: (target.score(pool[i]), -i))
        vertex = pool[best]
        chosen.append(vertex)
        chosen_set.add(vertex)
        target.add(vertex)
    return np.asarray(chosen, dtype=int)


def model_ordering(graph: GraphA, unit_weights: dict, k_max: int, pool_factor: int | None,
                   rng: np.random.Generator) -> np.ndarray:
    """Greedy on the frozen logistic score (pilot_new_rules.weighted_ordering with any pool width)."""

    by_degree = graph.degree_order(rng)
    state, chosen = P.FeatureState(graph), []
    for step in range(1, k_max + 1):
        unused = by_degree[~state.in_set[by_degree]]
        candidates = unused if pool_factor is None else unused[: pool_factor * step]
        features = state.candidate_features(candidates)
        score = sum(unit_weights[f] * features[f] for f in P.FEATURES)
        best = int(candidates[int(np.argmax(score))])  # first maximum = higher degree on exact ties
        state.add(best)
        chosen.append(best)
    return np.asarray(chosen, dtype=int)


def lookalike_ordering(graph: GraphA, source: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """At each position, a random unused node with the degree closest to the source's node."""

    used = np.zeros(graph.n, dtype=bool)
    order = []
    for vertex in source:
        free = np.flatnonzero(~used)
        gap = np.abs(graph.degree[free] - graph.degree[int(vertex)])
        pick = int(rng.choice(free[gap == gap.min()]))
        used[pick] = True
        order.append(pick)
    return np.asarray(order, dtype=int)


def build_orderings(arm: str, model: str, graph: GraphA, pair_seed: int, k_max: int, rules,
                    unit_weights: dict | None = None) -> dict[str, np.ndarray]:
    """Each rule's ordering of graph A's nodes, up to the largest budget."""

    algorithm = ARM_ALGORITHM[arm]
    orders: dict[str, np.ndarray] = {}

    def stream(code: int) -> np.random.Generator:
        return np.random.default_rng([S.STRATEGY_SEED_BASE, pair_seed, code])

    def get(rule: str) -> np.ndarray:
        if rule in orders:
            return orders[rule]
        if rule in ("top_degree", "top_degree_check"):
            order = graph.degree_order(stream(STRATEGY_CODES["top_degree"]))[:k_max]
        elif rule == "random":
            order = stream(STRATEGY_CODES["random"]).permutation(graph.n)[:k_max]
        elif rule == "hub_free_top_degree":
            order = H.orderings(graph, pair_seed, k_max)[2]
        elif rule.startswith("lookalike_"):
            info = describe_rule(rule)
            code = LOOKALIKE_CODE + 10 * LOOKALIKE_SOURCES.index(info["source"]) + info["draw"]
            order = lookalike_ordering(graph, get(info["source"]), stream(code))
        else:
            info = describe_rule(rule)
            factor = POOL_FACTORS[info["pool"]]
            if info["family"] == "pair_mass":
                order = degree_first(graph, k_max, stream(PAIR_MASS_CODE), "clustered",
                                     pool_factor=factor if factor else graph.n)
            elif info["family"] == "uniqueness":
                order = pool_greedy(graph, k_max, stream(UNIQUENESS_CODE), uniqueness_target(algorithm, graph), factor)
            elif info["family"] == "witness":
                order = pool_greedy(graph, k_max, stream(WITNESS_CODE),
                                    WitnessTarget(graph, witness_keep(model, graph)), factor)
            elif info["family"] == "model":
                if unit_weights is None:
                    raise ValueError(f"{rule} needs the frozen model's weights.")
                order = model_ordering(graph, unit_weights, k_max, factor, stream(MODEL_CODE))
            else:
                raise ValueError(f"Unknown rule: {rule}")
        orders[rule] = np.asarray(order, dtype=int)
        return orders[rule]

    return {rule: get(rule) for rule in rules}


# --------------------------------------------------------------------------
# Pairs, weights and one solver run
# --------------------------------------------------------------------------

def pair_indices(arm: str, model: str) -> list[int]:
    """Round-1 pairs 0-19, plus the fresh borderline pairs for percolation on ER and PAPER."""

    pairs = [i for i in H.round1_pairs(ARM_ALGORITHM[arm], model) if i < N_ROUND1_PAIRS]
    if arm == "percolation" and model in FRESH_MODELS:
        pairs += [i for i in FRESH_PAIRS if (BORDERLINE_DIR / model / f"pair_{i:03d}.csv").exists()]
    return pairs


def rules_for(arm: str, model: str, pair_index: int) -> list[str]:
    rules = list(RULES[arm][model])
    if pair_index >= FRESH_FIRST:
        rules = ["random"] + rules
    if arm != "goat_other" and pair_index in CHECK_PAIRS:
        rules = ["top_degree_check"] + rules
    return rules


def load_pair(arm: str, model: str, pair_index: int, verify: bool = True):
    """Regenerate a stored pair exactly (round 1's pairs, or borderline_check's fresh pairs)."""

    algorithm = ARM_ALGORITHM[arm]
    if pair_index < FRESH_FIRST:
        seed, A, B, truth = H.load_pair(model, pair_index, verify, algorithm)
        return seed, A, B, truth, "round1"
    config = S.graph_config("percolation", model)
    first_seed = rc.pair_first_seed("percolation", model, "main", pair_index)
    offset = 0
    while True:
        seed, A, B, truth, _report, _rejected = first_passing_pair(
            lambda s: generate_graph_pair(config, s), first_seed + offset)
        if seed not in S.KNOWN_PAIR_SEEDS:
            break
        offset = seed - first_seed + 1
    if verify:
        ref = pd.read_csv(BORDERLINE_DIR / model / f"pair_{pair_index:03d}.csv").iloc[0]
        if int(ref["pair_seed"]) != int(seed) or ref["hash_a"] != rc._hash(A) or ref["hash_b"] != rc._hash(B):
            raise RuntimeError(f"percolation/{model} fresh pair {pair_index} does not match borderline_check.")
    return int(seed), A, B, truth, "fresh"


def fit_or_load_weights(algorithm: str, model: str) -> dict:
    """The frozen model's fit (pilot_new_rules.fit_weights), saved once under weights/."""

    path = OUTPUT_DIR / "weights" / f"{algorithm}_{model}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    fit = P.fit_weights(algorithm, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fit, indent=2), encoding="utf-8")
    return fit


def unit_weights_for(fit: dict) -> dict:
    return {f: fit["weight_per_sd"][f] / fit["sd"][f] for f in P.FEATURES}


def evaluate(arm: str, model: str, pair, seeds: np.ndarray, budget: int, restart: int) -> dict:
    """One solver run with round 1's settings; GOAT uses this arm's regulariser."""

    algorithm = ARM_ALGORITHM[arm]
    if algorithm != "goat":
        return rc.evaluate(algorithm, model, pair, seeds, budget, restart)
    A, B, truth, pair_seed = pair
    seed_value = rc.solver_seed(pair_seed, budget, restart)
    out = rc._run_sgm_once(A, B, truth, seeds, seed_value, S.SGM["max_iter"], S.SGM["tol"], transport=True,
                           transport_regularizer=regularizer(arm, model), transport_tol=S.SGM["transport_tol"],
                           transport_max_iter=S.SGM["transport_max_iter"])
    accuracy = out["final_unseeded_accuracy"]
    return {
        "solver_seed": seed_value, "final_unseeded_accuracy": accuracy,
        "success": accuracy >= S.SUCCESS_THRESHOLD["goat"], "success_090": accuracy >= 0.9,
        "objective": out["sgm_objective_score"], "n_iter": out["sgm_n_iter"], "converged": out["sgm_converged"],
        "runtime_seconds": out["sgm_runtime_seconds"], "h0_accuracy": out["h0_accuracy"],
        "matched_fraction": np.nan, "precision": np.nan,
    }


# --------------------------------------------------------------------------
# Jobs (one pair each; each writes one CSV file)
# --------------------------------------------------------------------------

def _task_path(arm: str, model: str, pair_index: int) -> Path:
    return OUTPUT_DIR / arm / model / f"pair_{pair_index:03d}.csv"


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def pair_job(arm: str, model: str, pair_index: int, unit_weights: dict | None = None, verify: bool = True,
             blas_threads: int = 3) -> str:
    """Every rule of this arm on one pair, at every round-1 budget."""

    path = _task_path(arm, model, pair_index)
    name = f"{ARM_LABELS[arm]} {MODEL_LABELS[model]} pair {pair_index}"
    if path.exists():
        return f"{name}: already done"
    started = time.perf_counter()
    algorithm = ARM_ALGORITHM[arm]
    with thread_limit(blas_threads):
        seed, A, B, truth, pair_set = load_pair(arm, model, pair_index, verify)
        graph = GraphA(A)
        budgets = sorted(S.default_budgets("main", algorithm, model))
        rules = rules_for(arm, model, pair_index)
        orders = build_orderings(arm, model, graph, seed, budgets[-1], ["top_degree", *rules], unit_weights)
        top = orders["top_degree"]
        rank = 1 + np.searchsorted(np.sort(-graph.degree), -graph.degree, side="left")  # 1 = highest degree
        pair = (A, B, truth, seed)
        restarts = S.FAQ_MAX_RESTARTS if algorithm == "faq" else 1
        rows, cache, profile = [], {}, {}
        for rule in rules:
            for budget in budgets:
                seeds = np.sort(orders[rule][:budget])
                key = tuple(int(v) for v in seeds)
                if key not in profile:
                    profile[key] = {
                        "rank_median": float(np.median(rank[seeds])), "rank_max": int(rank[seeds].max()),
                        "top_degree_overlap": len(set(key) & {int(v) for v in top[:budget]}) / budget,
                        **seed_set_features(graph, seeds),
                    }
                for restart in range(restarts):
                    cached = (key, restart) in cache
                    if not cached:
                        cache[(key, restart)] = evaluate(arm, model, pair, seeds, budget, restart)
                    out = cache[(key, restart)]
                    rows.append({
                        "arm": arm, "algorithm": algorithm, "model": model, "regularizer": regularizer(arm, model),
                        "pair_set": pair_set, "pair_index": pair_index, "pair_seed": seed, "rule": rule,
                        "budget": budget, "restart": restart, "success": bool(out["success"]),
                        "success_090": bool(out["success_090"]), "accuracy": out["final_unseeded_accuracy"],
                        "objective": out["objective"], "n_iter": out["n_iter"], "converged": out["converged"],
                        "runtime_seconds": out["runtime_seconds"], "cached": cached,
                        "seed_vertices": json.dumps(list(key)), **profile[key],
                    })
                    if out["success"]:
                        break
    _write(path, rows)
    runs = sum(not row["cached"] for row in rows)
    return f"{name}: {runs} runs in {time.perf_counter() - started:.0f} s"


def _tasks(arms) -> list[tuple[str, str, int]]:
    tasks = [(arm, model, i) for arm in arms for model in ARM_MODELS[arm] for i in pair_indices(arm, model)
             if not _task_path(arm, model, i).exists()]
    # arm by arm; within an arm the models move forward together, round-1 pairs before fresh ones
    return sorted(tasks, key=lambda t: (ARMS.index(t[0]), t[2] >= FRESH_FIRST, t[2], ARM_MODELS[t[0]].index(t[1])))


def _estimate_minutes(tasks, jobs: int) -> float:
    seconds = 0.0
    for arm, model, pair_index in tasks:
        algorithm = ARM_ALGORITHM[arm]
        runs = len(rules_for(arm, model, pair_index)) * len(S.default_budgets("main", algorithm, model))
        runs *= FAQ_RUNS_PER_BUDGET if algorithm == "faq" else 1
        seconds += runs * SECONDS_PER_RUN[algorithm]
    return seconds / max(jobs, 1) / 60


def prepare_weights(arms) -> dict:
    """Unit weights of the frozen model for every (algorithm, model) whose rules need them."""

    needed = sorted({(ARM_ALGORITHM[a], m) for a in arms for m in ARM_MODELS[a]
                     if any(r.startswith("model_") for r in RULES[a][m])})
    return {key: unit_weights_for(fit_or_load_weights(*key)) for key in needed}


def _settings_snapshot() -> dict:
    return {
        "rules": RULES, "check_pairs": list(CHECK_PAIRS), "fresh_pairs": [FRESH_PAIRS.start, FRESH_PAIRS.stop - 1],
        "pool_factors": {k: v for k, v in POOL_FACTORS.items()},
        "regularizers": {"goat": dict(S.GOAT_REGULARIZER), "goat_other": OTHER_REGULARIZER},
        "uniqueness": {"goat": "entropy of cap-3 fingerprints", "percolation": "entropy of cap-3 fingerprints",
                       "faq": "unique cap-2 fingerprints, entropy breaks ties"},
        "witness_keep": "rho + (1 - rho) * mean degree / (n - 1)",
        "streams": {"pair_mass": PAIR_MASS_CODE, "model": MODEL_CODE, "uniqueness": UNIQUENESS_CODE,
                    "witness": WITNESS_CODE, "lookalike": LOOKALIKE_CODE},
        "budgets": {a: {m: list(S.default_budgets("main", ARM_ALGORITHM[a], m)) for m in ARM_MODELS[a]} for a in ARMS},
    }


def run_all(arms=ARMS, jobs: int = 3, blas_threads: int = 3, verify: bool = True) -> None:
    """Run every pair that is not done yet, `jobs` pairs at a time."""

    arms = [a for a in ARMS if a in set(arms)]
    tasks = _tasks(arms)
    if not tasks:
        print(f"All pairs are done. Results are in {OUTPUT_DIR}")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "settings.json").write_text(json.dumps(_settings_snapshot(), indent=2), encoding="utf-8")
    weights = prepare_weights(arms)
    for arm in arms:
        part = [t for t in tasks if t[0] == arm]
        if part:
            print(f"{_now()} {ARM_LABELS[arm]}: {len(part)} pairs, about {_estimate_minutes(part, jobs):.0f} minutes",
                  flush=True)
    print(f"{_now()} {len(tasks)} pairs to run, {jobs} at a time: about {_estimate_minutes(tasks, jobs):.0f} "
          f"minutes in total.", flush=True)
    keep_awake = None
    if sys.platform == "darwin":
        import shutil
        import subprocess
        if shutil.which("caffeinate"):
            keep_awake = subprocess.Popen(["caffeinate", "-i"])
    failures = []

    def arguments(task):
        arm, model, pair_index = task
        return (arm, model, pair_index, weights.get((ARM_ALGORITHM[arm], model)), verify, blas_threads)

    try:
        if jobs <= 1:
            for n, task in enumerate(tasks, 1):
                print(f"{_now()} [{n}/{len(tasks)}] {pair_job(*arguments(task))}", flush=True)
        else:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {pool.submit(pair_job, *arguments(task)): task for task in tasks}
                for n, future in enumerate(as_completed(futures), 1):
                    arm, model, pair_index = futures[future]
                    try:
                        print(f"{_now()} [{n}/{len(tasks)}] {future.result()}", flush=True)
                    except Exception as error:  # noqa: BLE001 - report every failure, then stop
                        failures.append(f"{ARM_LABELS[arm]} {MODEL_LABELS[model]} pair {pair_index}")
                        print(f"{_now()} [{n}/{len(tasks)}] {failures[-1]} FAILED:", flush=True)
                        traceback.print_exception(type(error), error, error.__traceback__)
    finally:
        if keep_awake is not None:
            keep_awake.terminate()
    if failures:
        raise RuntimeError(f"Failed: {', '.join(failures)}. Rerun to retry them.")
    print(f"{_now()} Done. Results are in {OUTPUT_DIR}", flush=True)


def status(arms=ARMS) -> pd.DataFrame:
    """Pairs planned and done per arm and model."""

    rows = []
    for arm in arms:
        for model in ARM_MODELS[arm]:
            planned = pair_indices(arm, model)
            done = [i for i in planned if _task_path(arm, model, i).exists()]
            rows.append({"arm": ARM_LABELS[arm], "model": MODEL_LABELS[model], "pairs planned": len(planned),
                         "pairs done": len(done), "new rules": len(RULES[arm][model])})
    return pd.DataFrame(rows).set_index(["arm", "model"])


# --------------------------------------------------------------------------
# Loading results (new runs plus the earlier results for the same pairs)
# --------------------------------------------------------------------------

KEYS = ["arm", "algorithm", "model", "regularizer", "pair_set", "pair_index", "rule", "budget"]


def read_runs(arms=ARMS) -> pd.DataFrame:
    """Every new solver run (one row per restart)."""

    files = [f for arm in arms for f in sorted((OUTPUT_DIR / arm).glob("*/pair_*.csv"))]
    if not files:
        return pd.DataFrame()
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for column in ("success", "success_090", "cached"):
        runs[column] = _flag(runs[column])
    return runs


def _collapse(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per evaluation: success if any restart succeeded; accuracy of the best-objective run."""

    runs = runs.copy()
    if "objective" not in runs:
        runs["objective"] = np.nan
    runs["order_key"] = pd.to_numeric(runs["objective"], errors="coerce").fillna(-np.inf)
    selected = runs.sort_values("order_key", kind="stable").groupby(KEYS, sort=False).tail(1).set_index(KEYS)
    grouped = runs.groupby(KEYS, sort=False)
    out = pd.DataFrame({"success": grouped["success"].any(), "success_090": grouped["success_090"].any(),
                        "accuracy": selected["accuracy"], "runs": grouped.size()})
    for column in ("seed_vertices", "rank_median", "rank_max", *FEATURE_COLUMNS):
        if column in runs:
            out[column] = grouped[column].first()
    return out.reset_index()


def _reused(arm: str, model: str, pairs) -> pd.DataFrame:
    """Round 1, hub_test and borderline_check results for these pairs (same seeds, same settings)."""

    algorithm, reg = ARM_ALGORITHM[arm], regularizer(arm, model)
    if arm == "goat_other":
        return pd.DataFrame()
    frames = []
    round1 = [i for i in pairs if i < FRESH_FIRST]
    if round1:
        old = pd.read_csv(ROUND1_ROOT / algorithm / model / "runs.csv")
        old = old[old["pair_index"].isin(round1) & old["strategy"].isin(REUSED_ROUND1)]
        old = old.rename(columns={"strategy": "rule", "final_unseeded_accuracy": "accuracy"})
        frames.append(old.assign(success=_flag(old["success"]), success_090=_flag(old["success_090"])))
        if algorithm == "goat":  # hub-free top degree, for the comparison of GOAT's two regularisers
            hub = H._read_folder(HUB_TEST_DIR / model)
            if not hub.empty:
                hub = hub[(hub["rule"] == "hub_free_top_degree") & hub["pair_index"].isin(round1)]
                frames.append(hub.assign(success_090=hub["accuracy"] >= 0.9))
    fresh = [i for i in pairs if i >= FRESH_FIRST]
    if fresh and arm == "percolation":
        files = [BORDERLINE_DIR / model / f"pair_{i:03d}.csv" for i in fresh]
        old = pd.concat([pd.read_csv(f) for f in files if f.exists()], ignore_index=True)
        frames.append(old.assign(success=_flag(old["success"]), success_090=_flag(old["success_090"])))
    if not frames:
        return pd.DataFrame()
    old = pd.concat(frames, ignore_index=True)
    old = old.assign(arm=arm, algorithm=algorithm, model=model, regularizer=reg,
                     pair_set=np.where(old["pair_index"] >= FRESH_FIRST, "fresh", "round1"))
    return old


def _overlap_with_top_degree(ev: pd.DataFrame) -> pd.Series:
    top = ev[ev["rule"] == "top_degree"]
    lookup = {(r.arm, r.model, r.pair_index, r.budget): set(json.loads(r.seed_vertices))
              for r in top.itertuples() if isinstance(r.seed_vertices, str)}

    def overlap(r):
        reference = lookup.get((r.arm, r.model, r.pair_index, r.budget))
        if reference is None or not isinstance(r.seed_vertices, str):
            return np.nan
        return len(set(json.loads(r.seed_vertices)) & reference) / r.budget

    return pd.Series([overlap(r) for r in ev.itertuples()], index=ev.index, dtype=float)


def load_evaluations(arms=ARMS) -> pd.DataFrame:
    """One row per (arm, model, pair, rule, budget), new rules and reused ones, on the pairs done so far."""

    runs = read_runs(arms)
    if runs.empty:
        raise FileNotFoundError(f"No results in {OUTPUT_DIR}. Run run_all() first.")
    frames = [_collapse(runs[runs["rule"] != "top_degree_check"])]
    for (arm, model), part in runs.groupby(["arm", "model"], sort=False):
        old = _reused(arm, model, sorted(part["pair_index"].unique()))
        if not old.empty:
            frames.append(_collapse(old))
    ev = pd.concat(frames, ignore_index=True)
    clustered = ev[ev["rule"] == "clustered"].assign(rule="pair_mass_2j")  # round 1's clustered = pair mass at 2j
    ev = pd.concat([ev, clustered], ignore_index=True)
    ev["top_degree_overlap"] = _overlap_with_top_degree(ev)
    info = pd.DataFrame([describe_rule(r) for r in ev["rule"]], index=ev.index)
    return pd.concat([ev, info], axis=1)


def middle_budget(algorithm: str, model: str) -> int:
    budgets = sorted(S.default_budgets("main", algorithm, model))
    return budgets[len(budgets) // 2]


# --------------------------------------------------------------------------
# Analysis helpers for beyond_top_degree.ipynb
# --------------------------------------------------------------------------

def queries_needed(ev: pd.DataFrame) -> pd.DataFrame:
    """Smallest budget that succeeds, per pair and rule. A pair that never succeeds counts as one step past the grid.

    Look-alike draws are also averaged per pair into one rule, lookalike_<source>.
    """

    rows = []
    for (arm, model), part in ev.groupby(["arm", "model"], sort=False):
        budgets = sorted(part["budget"].unique())
        past_grid = budgets[-1] + (budgets[-1] - budgets[-2])
        table = part.assign(success=part["success"].astype(int)).pivot_table(
            index=["pair_set", "pair_index", "rule"], columns="budget", values="success", aggfunc="max")
        table = table.reindex(columns=budgets).dropna()
        for (pair_set, pair_index, rule), row in table.iterrows():
            hits = [b for b, ok in row.items() if ok >= 1]
            rows.append({"arm": arm, "model": model, "pair_set": pair_set, "pair_index": pair_index, "rule": rule,
                         "queries": float(hits[0] if hits else past_grid), "never": not hits})
    queries = pd.DataFrame(rows)
    draws = queries[queries["rule"].str.match(r"lookalike_.*_\d+$")]
    if not draws.empty:
        mean = (draws.assign(rule=draws["rule"].str.replace(r"_\d+$", "", regex=True))
                .groupby(["arm", "model", "pair_set", "pair_index", "rule"], as_index=False)
                .agg(queries=("queries", "mean"), never=("never", "all")))
        queries = pd.concat([queries, mean], ignore_index=True)
    return queries


def success_curves(ev: pd.DataFrame) -> pd.DataFrame:
    """Share of pairs that succeed, per budget; look-alike draws are averaged per pair first."""

    data = ev.assign(success=ev["success"].astype(float))
    draws = data[data["rule"].str.match(r"lookalike_.*_\d+$")]
    if not draws.empty:
        mean = (draws.assign(rule=draws["rule"].str.replace(r"_\d+$", "", regex=True))
                .groupby(["arm", "model", "pair_index", "rule", "budget"], as_index=False)["success"].mean())
        data = pd.concat([data, mean], ignore_index=True)
    return data.groupby(["arm", "model", "rule", "budget"], as_index=False)["success"].mean()


def _mean_interval(values, n_boot: int = 2000, seed: int = 2026):
    values = np.asarray(values, float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boot = np.random.default_rng(seed).choice(values, size=(n_boot, len(values))).mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def mean_queries(queries: pd.DataFrame, arm: str, model: str, rule: str) -> dict:
    values = queries[(queries["arm"] == arm) & (queries["model"] == model) & (queries["rule"] == rule)]["queries"]
    mean, low, high = _mean_interval(values)
    return {"mean": mean, "low": low, "high": high, "pairs": len(values)}


def path_summary(queries: pd.DataFrame, arm: str, model: str) -> pd.DataFrame:
    """Mean queries needed along each objective's path, starting from top degree."""

    present = set(queries.loc[(queries["arm"] == arm) & (queries["model"] == model), "rule"])
    rows = []
    for objective, rules in PATHS.items():
        rules = [r for r in rules if r in present]
        if not [r for r in rules if r != "pair_mass_2j"]:  # round 1's clustered alone is not a path
            continue
        for rule in ["top_degree", *rules]:
            if rule not in present:
                continue
            rows.append({"objective": objective, "rule": rule, "pool": describe_rule(rule)["pool"],
                         **mean_queries(queries, arm, model, rule)})
    return pd.DataFrame(rows)


def compare(queries: pd.DataFrame, arm: str, model: str, first: str, second: str, n_boot: int = 2000,
            seed: int = 2026) -> dict | None:
    """Pair-by-pair comparison of queries needed: first rule against second rule (negative = first needs fewer)."""

    from scipy.stats import binomtest

    part = queries[(queries["arm"] == arm) & (queries["model"] == model) & queries["rule"].isin([first, second])]
    table = part.pivot_table(index="pair_index", columns="rule", values="queries", aggfunc="first")
    if first not in table or second not in table:
        return None
    table = table[[first, second]].dropna()
    difference = (table[first] - table[second]).to_numpy(float)
    if len(difference) == 0:
        return None
    fewer, more = int((difference < 0).sum()), int((difference > 0).sum())
    mean, low, high = _mean_interval(difference, n_boot, seed)
    return {"pairs": len(difference), "fewer": fewer, "same": len(difference) - fewer - more, "more": more,
            "p": binomtest(fewer, fewer + more).pvalue if fewer + more else np.nan,
            "mean change": f"{mean:+.2f} [{low:+.2f}, {high:+.2f}]"}


def comparison_table(queries: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Each new rule against top degree and random; all-node rules also against their look-alike and round 1's best."""

    rows = []
    for model in ARM_MODELS[arm]:
        part = queries[(queries["arm"] == arm) & (queries["model"] == model)]
        if part.empty:
            continue
        present = set(part["rule"])
        new = [r for r in RULES[arm][model] if r in present and not r.startswith("lookalike_")
               and r not in ("top_degree", "random")]
        for rule in new:
            references = ["top_degree", "random"]
            if f"lookalike_{rule}" in present:
                references.append(f"lookalike_{rule}")
            best = BEST_ROUND1.get((ARM_ALGORITHM[arm], model))
            if rule.endswith("_all") and best in present:
                references.append(best)
            for reference in references:
                result = compare(queries, arm, model, rule, reference)
                if result:
                    rows.append({"model": MODEL_LABELS[model], "rule": rule_label(rule),
                                 "against": rule_label(reference), **result})
    return pd.DataFrame(rows).set_index(["model", "rule", "against"]) if rows else pd.DataFrame()


def gain_split(queries: pd.DataFrame, arm: str, model: str, rule: str) -> dict | None:
    """Queries saved against random, split into the degree level (look-alike) and the feature itself."""

    lookalike = f"lookalike_{rule}"
    part = queries[(queries["arm"] == arm) & (queries["model"] == model)
                   & queries["rule"].isin(["random", lookalike, rule])]
    table = part.pivot_table(index="pair_index", columns="rule", values="queries", aggfunc="first")
    if not {"random", lookalike, rule} <= set(table.columns):
        return None
    table = table[["random", lookalike, rule]].dropna()
    pieces = {"degree level (random - look-alike)": table["random"] - table[lookalike],
              "feature (look-alike - rule)": table[lookalike] - table[rule],
              "total (random - rule)": table["random"] - table[rule]}
    out = {"pairs": len(table)}
    for name, values in pieces.items():
        mean, low, high = _mean_interval(values)
        out[name] = (mean, low, high)
    return out


def seed_profile(ev: pd.DataFrame, arm: str, model: str) -> pd.DataFrame:
    """Means over pairs at the middle budget: share of seeds shared with top degree, pair mass and uniqueness."""

    algorithm = ARM_ALGORITHM[arm]
    k = middle_budget(algorithm, model)
    part = ev[(ev["arm"] == arm) & (ev["model"] == model) & (ev["budget"] == k)].copy()
    part["rule"] = part["rule"].str.replace(r"^(lookalike_.*)_\d+$", r"\1", regex=True)
    columns = ["top_degree_overlap", "f_degree", "f_pair_mass", UNIQUENESS_FEATURE[algorithm]]
    profile = part.groupby("rule")[columns].mean()
    if "rank_median" in part:
        profile["rank_median"] = part.groupby("rule")["rank_median"].mean()
    return profile.assign(budget=k)


def regularizer_table(queries: pd.DataFrame, model: str) -> pd.DataFrame:
    """GOAT: mean queries needed at round 1's regulariser and at the other one, for the rules run at both."""

    first = queries[(queries["arm"] == "goat") & (queries["model"] == model)]
    second = queries[(queries["arm"] == "goat_other") & (queries["model"] == model)]
    rules = [r for r in RULES["goat_other"][model] if r in set(first["rule"]) and r in set(second["rule"])]
    rows = []
    for rule in rules:
        a = first[first["rule"] == rule].set_index("pair_index")["queries"]
        b = second[second["rule"] == rule].set_index("pair_index")["queries"]
        common = a.index.intersection(b.index)
        rows.append({"rule": rule, "pairs": len(common),
                     f"regulariser {S.GOAT_REGULARIZER[model]}": a[common].mean(),
                     f"regulariser {OTHER_REGULARIZER[model]}": b[common].mean()})
    return pd.DataFrame(rows).set_index("rule") if rows else pd.DataFrame()


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_against_stored(arms=ARMS) -> pd.DataFrame:
    """Top degree rerun on pairs 0, 1, 100 and 101: success and accuracy must equal the stored runs."""

    runs = read_runs(arms)
    if runs.empty:
        return pd.DataFrame()
    runs = runs[runs["rule"] == "top_degree_check"]
    rows = []
    for (arm, model), mine in runs.groupby(["arm", "model"], sort=False):
        algorithm = ARM_ALGORITHM[arm]
        for pair_set, part in mine.groupby("pair_set"):
            if pair_set == "round1":
                ref = pd.read_csv(ROUND1_ROOT / algorithm / model / "runs.csv")
                ref = ref[ref["strategy"] == "top_degree"].rename(columns={"final_unseeded_accuracy": "accuracy"})
            else:
                ref = pd.concat([pd.read_csv(BORDERLINE_DIR / model / f"pair_{i:03d}.csv")
                                 for i in sorted(part["pair_index"].unique())], ignore_index=True)
                ref = ref[ref["rule"] == "top_degree"].assign(restart=0)
            ref = ref.set_index(["pair_index", "budget", "restart"])
            part = part.set_index(["pair_index", "budget", "restart"])
            ref = ref.reindex(part.index)
            rows.append({"arm": ARM_LABELS[arm], "model": MODEL_LABELS[model], "pairs": pair_set,
                         "runs checked": len(part),
                         "same success": int((part["success"].to_numpy() == _flag(ref["success"]).to_numpy()).sum()),
                         "same accuracy": int(np.isclose(part["accuracy"].to_numpy(float),
                                                         ref["accuracy"].to_numpy(float)).sum())})
    return pd.DataFrame(rows).set_index(["arm", "model", "pairs"]) if rows else pd.DataFrame()


def check_model_against_pilot(arms=ARMS) -> pd.DataFrame:
    """model_2j should pick exactly the pilot's weighted_2j seeds (same weights and tie-break stream)."""

    runs = read_runs(arms)
    if runs.empty:
        return pd.DataFrame()
    rows = []
    for path in sorted(PILOT_DIR.glob("weighted_*.csv")):
        if path.name.endswith("_orderings.csv"):
            continue
        algorithm, model = path.stem[len("weighted_"):].split("_", 1)
        arm = algorithm
        mine = runs[(runs["arm"] == arm) & (runs["model"] == model) & (runs["rule"] == "model_2j")
                    & (runs["restart"] == 0)]
        if mine.empty:
            continue
        pilot = pd.read_csv(path)
        pilot = pilot[pilot["strategy"] == "weighted_2j"].set_index(["pair_index", "budget"])["seed_vertices"]
        mine = mine.set_index(["pair_index", "budget"])["seed_vertices"]
        common = mine.index.intersection(pilot.index)
        same = sum(json.loads(mine[i]) == json.loads(pilot[i]) for i in common)
        rows.append({"arm": ARM_LABELS[arm], "model": MODEL_LABELS[model], "seed sets compared": len(common),
                     "identical": same})
    return pd.DataFrame(rows).set_index(["arm", "model"]) if rows else pd.DataFrame()


def weights_table() -> pd.DataFrame:
    """The frozen model's weights (log-odds per SD of the training random sets), and the pilot's where saved."""

    rows = []
    for path in sorted((OUTPUT_DIR / "weights").glob("*.json")):
        fit = json.loads(path.read_text(encoding="utf-8"))
        row = {"algorithm": ALGORITHM_LABELS[fit["algorithm"]], "model": MODEL_LABELS[fit["model"]],
               "training sets": fit["n_sets"], "training budget": fit["n_seeds"],
               **{f"{f[2:]}": fit["weight_per_sd"][f] for f in P.FEATURES}}
        pilot = PILOT_DIR / f"weights_{fit['algorithm']}_{fit['model']}.json"
        if pilot.exists():
            saved = json.loads(pilot.read_text(encoding="utf-8"))
            row["same as pilot"] = all(np.isclose(saved["weight_per_sd"][f], fit["weight_per_sd"][f])
                                       for f in P.FEATURES)
        rows.append(row)
    return pd.DataFrame(rows).set_index(["algorithm", "model"]) if rows else pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS,
                        help="which arms to run (default: all four, in this order)")
    parser.add_argument("--jobs", type=int, default=3, help="pairs run in parallel (default 3)")
    parser.add_argument("--blas-threads", type=int, default=3)
    args = parser.parse_args()
    run_all(arms=args.arms, jobs=args.jobs, blas_threads=args.blas_threads)
