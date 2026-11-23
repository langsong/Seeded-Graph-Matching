"""Two-phase active querying on native real pairs: a small initial test (17 Nov 2026).

Question. At the same total budget K, does spending the second part of the budget on the
vertices the matcher is least sure about (as in Malmi, Gionis & Terzi 2017) beat spending it
on more structural seeds?

Every policy ends with exactly K verified seeds (a query reveals one vertex's true partner):

    random, top_degree, clustered,          static: all K seeds chosen from A before matching
    distinguishing, hub_free_distinguishing
    two_phase_confidence                    the first K1 = round(phase1_share * K) seeds from the
    two_phase_consistency                   base rule (top degree); then rounds: run the matcher,
    two_phase_random                        query the `batch` least certain unqueried vertices,
                                            re-run, ... until K seeds

Certainty (least certain first; ties go to the higher-degree vertex):
    confidence   GOAT: the largest entry of the vertex's row in the final soft matching.
                 FAQ: the share of solver restarts that agree with the best-objective run's
                 partner, with the soft row maximum as tie-break (one restart: the soft row maximum).
    consistency  the share of the vertex's A edges that its current match keeps in B.
    random       control: each batch is drawn uniformly from the unqueried vertices, with the same
                 rounds and re-runs. It separates *which* vertices from *more seeds plus re-runs*.
After each re-run the new matching is kept only if it keeps at least as many edges as the
previous matching with the new answers filled in. Otherwise the corrected previous matching is
kept (a label-free guard against GOAT's collapses near take-off).

Outcome. Every vertex can be queried. The main outcome is `acc_all`: total correct matches as a
share of n (the K verified seeds plus the correctly matched unqueried vertices). Accuracy on the
unqueried vertices is saved too. Per round, rounds.csv records how well each certainty separates
wrongly from correctly matched vertices (`auc_*`; 0.5 = no information) and the share of the
queried batch that the current matching had wrong (`hit_rate`).

Run from the repository root (ds environment):

    python -m real_data_experiments.active_test --smoke             # about a minute, own folder
    python -m real_data_experiments.active_test                     # three Enron pairs, minutes
    python -m real_data_experiments.active_test --cases wikipedia   # roughly 30-60 minutes
    python -m real_data_experiments.active_test --summary

Results: results/active_test/{settings.json, finals.csv, rounds.csv}. A (pair, algorithm,
budget) block is written when it finishes; finished blocks are skipped on rerun.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from experiment_common.provenance import collect_environment, thread_limit
from selection_rule_comparison.strategies import GraphA, ordering

from . import datasets as D

PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("ACTIVE_TEST_OUTPUT_DIR") or PACKAGE_DIR / "results" / "active_test")
SMOKE_DIR = PACKAGE_DIR / "results" / "active_test_smoke"

ALGORITHMS = ("faq", "goat")
STATIC = ("random", "top_degree", "clustered", "distinguishing", "hub_free_distinguishing")
TWO_PHASE = ("two_phase_confidence", "two_phase_consistency", "two_phase_random")
POLICIES = (*STATIC, *TWO_PHASE)
LABELS = {"random": "Random", "top_degree": "Top degree", "clustered": "Clustered",
          "distinguishing": "Distinguishing", "hub_free_distinguishing": "Hub-free distinguishing",
          "two_phase_confidence": "Two-phase: matcher confidence",
          "two_phase_consistency": "Two-phase: edge consistency",
          "two_phase_random": "Two-phase: random batches (control)",
          "faq": "SGM (FAQ)", "goat": "GOAT (r=300)"}


def _enron(first: int, second: int) -> dict:
    return {"get": {"name": "enron", "weeks": [first, second], "vertices": "common_lcc"},
            "budgets": [10, 20, 30], "batch": 5}


CASES = {
    "enron_130_132": _enron(130, 132),
    "enron_145_147": _enron(145, 147),
    "enron_160_162": _enron(160, 162),
    "wikipedia": {"get": {"name": "wikipedia"}, "budgets": [20, 40], "batch": 10, "large": True},
}
DEFAULT_CASES = ("enron_130_132", "enron_145_147", "enron_160_162")

FINAL_FIELDS = ("case", "n", "algorithm", "budget", "policy", "draw", "phase1_seeds", "batch", "rounds",
                "acc_all", "acc_unqueried", "total_correct", "acc_all_restart_mean", "edge_ratio",
                "kept_previous_rounds", "soft_check", "runtime_seconds", "seeds")
ROUND_FIELDS = ("case", "n", "algorithm", "budget", "policy", "draw", "round", "n_seeds", "acc_all",
                "acc_unqueried", "acc_all_unguarded", "edge_ratio", "kept_previous", "soft_check",
                "certainty_source", "auc_confidence", "auc_consistency", "error_rate", "batch",
                "hit_rate", "chosen_mean_degree", "chosen", "runtime_seconds")


def default_config() -> dict:
    return json.loads(json.dumps({
        "version": 1, "seed_base": 2026111701, "cases": CASES, "algorithms": list(ALGORITHMS),
        "phase1_share": 0.5, "base_rule": "top_degree", "guard": True,
        "random_draws": {"small": 10, "large": 3}, "control_draws": {"small": 5, "large": 2},
        "restarts": {"faq": {"small": 3, "large": 1}, "goat": {"small": 1, "large": 1}},
        # Same as selection_rule_comparison.settings.SGM and the earlier real-pair runs.
        "solver": {"max_iter": 100, "tol": 0.01, "transport_tol": 0.05, "transport_max_iter": 5000},
        "goat_regularizer": 300.0}))


def smoke_config() -> dict:
    config = default_config()
    config["cases"] = {"enron_130_132": {**CASES["enron_130_132"], "budgets": [10]}}
    config.update(random_draws={"small": 2, "large": 1}, control_draws={"small": 1, "large": 1},
                  restarts={"faq": {"small": 2, "large": 1}, "goat": {"small": 1, "large": 1}})
    return config


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _seed(*parts) -> int:
    codes = [int(p) if isinstance(p, (int, np.integer)) else zlib.crc32(str(p).encode("utf-8"))
             for p in parts]
    return int(np.random.SeedSequence(codes).generate_state(1)[0])


def _rng(*parts) -> np.random.Generator:
    return np.random.default_rng(_seed(*parts))


def _size(case: str, config: dict) -> str:
    return "large" if config["cases"][case].get("large") else "small"


def load_case(case: str, config: dict):
    """(A, shuffled B, truth) as 0/1 float matrices; truth[v] is v's partner in the shuffled B."""

    spec = dict(config["cases"][case]["get"])
    if spec.get("name") == "synthetic":   # small correlated ER pair, for tests and quick checks
        rng = np.random.default_rng(spec.get("seed", 0))
        n, p, rho = spec.get("n", 60), spec.get("p", 0.1), spec.get("rho", 0.8)
        upper = np.triu(np.ones((n, n), dtype=bool), 1)
        a = upper & (rng.random((n, n)) < p)
        keep = np.where(a, p + rho * (1 - p), p * (1 - rho))
        b = upper & (rng.random((n, n)) < keep)
        A, B0 = (a | a.T).astype(float), (b | b.T).astype(float)
        shuffle = rng.permutation(n)
        return A, B0[np.ix_(shuffle, shuffle)], np.argsort(shuffle)
    pair = D.get_pair(**spec)
    A, B, truth = pair.for_matching(_seed(config["seed_base"], case, "shuffle"))
    return (np.asarray(A) > 0).astype(float), (np.asarray(B) > 0).astype(float), np.asarray(truth, dtype=int)


def kept_edges(A: np.ndarray, B: np.ndarray, perm: np.ndarray) -> float:
    """A edges whose matched endpoints are adjacent in B: the label-free objective."""

    return float((A * B[np.ix_(perm, perm)]).sum() / 2)


def consistency(A: np.ndarray, B: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Share of each vertex's A edges that its current match keeps in B (0 for isolated vertices)."""

    degree = A.sum(axis=1)
    kept = (A * B[np.ix_(perm, perm)]).sum(axis=1)
    return np.where(degree > 0, kept / np.maximum(degree, 1), 0.0)


def evaluate(perm: np.ndarray, truth: np.ndarray, seeds) -> dict:
    """Total correct matches: verified seeds plus correctly matched unqueried vertices."""

    n = len(truth)
    seeds = np.asarray(seeds, dtype=int)
    unqueried = np.setdiff1d(np.arange(n), seeds)
    correct = int(np.sum(perm[unqueried] == truth[unqueried]))
    return {"acc_all": (len(seeds) + correct) / n, "total_correct": len(seeds) + correct,
            "acc_unqueried": correct / len(unqueried) if len(unqueried) else float("nan")}


def correct(perm: np.ndarray, answers, truth: np.ndarray) -> np.ndarray:
    """The matching with the queried vertices set to their true partners; displaced vertices swap."""

    perm = perm.copy()
    owner = np.empty_like(perm)
    owner[perm] = np.arange(len(perm))
    for v in answers:
        u, w = truth[v], owner[truth[v]]
        if w != v:
            old = perm[v]
            perm[v], perm[w] = u, old
            owner[u], owner[old] = v, w
    return perm


def auc_wrong(uncertainty: np.ndarray, wrong: np.ndarray) -> float:
    """P(a wrongly matched vertex looks less certain than a correctly matched one); ties count half."""

    from scipy.stats import rankdata

    wrong = np.asarray(wrong, dtype=bool)
    positives, negatives = int(wrong.sum()), int((~wrong).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(np.asarray(uncertainty, dtype=float))
    return float((ranks[wrong].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def projection_check(soft, assigned_columns):
    """Validate a projection by its score: equally optimal tied assignments are valid."""
    from scipy.optimize import linear_sum_assignment

    soft = np.asarray(soft)
    assigned_columns = np.asarray(assigned_columns, dtype=int)
    if (soft.ndim != 2 or soft.shape[0] != soft.shape[1]
            or not np.isfinite(soft).all()
            or sorted(assigned_columns.tolist()) != list(range(len(soft)))):
        return 0.0
    rows, columns = linear_sum_assignment(soft, maximize=True)
    optimum = soft[rows, columns].sum()
    actual = soft[np.arange(len(soft)), assigned_columns].sum()
    return float(np.isclose(actual, optimum, rtol=1e-9, atol=1e-9))


def modal_agreement(permutations):
    """Frequency of each vertex's most common partner, not agreement with one chosen run."""
    permutations = np.asarray(permutations)
    return np.asarray([np.unique(column, return_counts=True)[1].max() / len(permutations)
                       for column in permutations.T])


# --------------------------------------------------------------------------
# Matcher
# --------------------------------------------------------------------------

def match(A, B, truth, seeds, algorithm: str, solver_seed: int, config: dict) -> dict:
    """One FAQ or GOAT run given the verified seed pairs (the same call as experiment_common.sgm).

    Returns the full matching and each unseeded vertex's largest entry in the final soft
    matching, mapped to vertices as in pilot_new_rules.goat_run. `soft_check` is about 1 when
    that mapping agrees with the final matching. Only the seed pairs use the truth.
    """

    from graspologic.match import graph_match
    import graspologic.match.wrappers as wrappers
    from scipy.optimize import linear_sum_assignment

    seeds = np.asarray(sorted(int(v) for v in seeds), dtype=int)
    solver_settings = config["solver"]
    captured = {}
    original = wrappers._GraphMatchSolver.compute_step_direction

    def capture(solver, gradient, rng):
        captured.setdefault("solver", solver)
        return original(solver, gradient, rng)

    wrappers._GraphMatchSolver.compute_step_direction = capture
    started = time.perf_counter()
    try:
        result = graph_match(A, B, partial_match=np.column_stack((seeds, truth[seeds])), n_init=1, n_jobs=1,
                             init=None, init_perturbation=config.get("init_perturbation", 0.0), shuffle_input=True,
                             max_iter=solver_settings["max_iter"], tol=solver_settings["tol"],
                             rng=int(solver_seed), transport=algorithm == "goat",
                             transport_regularizer=config["goat_regularizer"],
                             transport_tol=solver_settings["transport_tol"],
                             transport_max_iter=solver_settings["transport_max_iter"])
    finally:
        wrappers._GraphMatchSolver.compute_step_direction = original
    runtime = time.perf_counter() - started
    n = A.shape[0]
    perm = np.full(n, -1, dtype=int)
    perm[np.asarray(result.indices_A)] = np.asarray(result.indices_B)
    if (perm < 0).any() or len(np.unique(perm)) != n:
        raise RuntimeError("The matcher did not return a one-to-one matching of all vertices.")
    confidence, soft_check = np.full(n, np.nan), float("nan")
    solver = captured.get("solver")
    misc = result.misc[0] if getattr(result, "misc", None) else {}
    soft = misc.get("convex_solution") if isinstance(misc, dict) else None
    if solver is not None and soft is not None:
        soft = np.asarray(soft, dtype=float)
        rows_a, cols_b = solver.perm_A[solver.n_seeds:], solver.perm_B[solver.n_seeds:]
        if soft.shape == (len(rows_a), len(cols_b)) and np.isfinite(soft).all():
            confidence[rows_a] = soft.max(axis=1)
            inverse_columns = np.full(n, -1, dtype=int)
            inverse_columns[cols_b] = np.arange(len(cols_b))
            soft_check = projection_check(soft, inverse_columns[perm[rows_a]])
    converged = bool(getattr(solver, "converged_", False)) if solver is not None else False
    return {"perm": perm, "confidence": confidence, "soft_check": soft_check,
            "runtime_seconds": runtime, "converged": converged}


def run_round(ctx: dict, seeds, algorithm: str, parts: tuple) -> dict:
    """The matcher's restarts with these seeds; the best-objective run is the current matching."""

    A, B, truth = ctx["A"], ctx["B"], ctx["truth"]
    restarts = ctx["config"]["restarts"][algorithm][ctx["size"]]
    outs = [ctx["matcher"](A, B, truth, seeds, algorithm, _seed(*parts, r), ctx["config"])
            for r in range(restarts)]
    objectives = [kept_edges(A, B, o["perm"]) for o in outs]
    best = int(np.argmax(objectives))
    perm = outs[best]["perm"]
    agreement = modal_agreement([o["perm"] for o in outs]) if restarts > 1 else None
    return {"perm": perm, "objective": objectives[best], "agreement": agreement,
            "confidence": outs[best]["confidence"], "soft_check": outs[best]["soft_check"],
            "acc_all_restart_mean": float(np.mean([evaluate(o["perm"], truth, seeds)["acc_all"] for o in outs])),
            "runtime_seconds": float(sum(o["runtime_seconds"] for o in outs)), "kept_previous": False}


def certainty(kind: str, algorithm: str, state: dict, ctx: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """(primary, tie-break, source) per vertex; lower values are less certain."""

    cons = consistency(ctx["A"], ctx["B"], state["perm"])
    soft_ok = np.isfinite(state["soft_check"]) and state["soft_check"] >= 0.95
    soft = np.nan_to_num(state["confidence"], nan=1.0) if soft_ok else None
    if kind == "consistency":
        return cons, (soft if soft is not None else np.zeros_like(cons)), "consistency"
    if algorithm == "faq" and state["agreement"] is not None:
        return state["agreement"], (soft if soft is not None else cons), "restart agreement"
    if soft is not None:
        return soft, cons, "soft matching"
    return cons, np.zeros_like(cons), "consistency (soft matching unavailable)"


def least_certain(primary, tiebreak, candidates, degree, b: int, rng) -> np.ndarray:
    order = np.lexsort((rng.random(len(candidates)), -degree[candidates],
                        tiebreak[candidates], primary[candidates]))
    return candidates[order[:b]]


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------

def _base_row(ctx, algorithm, K, policy, draw) -> dict:
    return {"case": ctx["case"], "n": ctx["n"], "algorithm": algorithm, "budget": K,
            "policy": policy, "draw": draw}


def static_policy(ctx: dict, algorithm: str, K: int, policy: str, draw: int) -> dict:
    config = ctx["config"]
    if policy == "random":   # the same random sets for both algorithms
        seeds = _rng(config["seed_base"], ctx["case"], K, "random", draw).choice(ctx["n"], K, replace=False)
    else:
        seeds = ctx["orderings"][policy][:K]
    seeds = [int(v) for v in seeds]
    state = run_round(ctx, seeds, algorithm, (config["seed_base"], ctx["case"], algorithm, K, policy, draw, 0))
    return {**_base_row(ctx, algorithm, K, policy, draw), "phase1_seeds": K, "batch": 0, "rounds": 0,
            **evaluate(state["perm"], ctx["truth"], seeds), "acc_all_restart_mean": state["acc_all_restart_mean"],
            "edge_ratio": state["objective"] / ctx["edge_scale"], "kept_previous_rounds": 0,
            "soft_check": state["soft_check"], "runtime_seconds": state["runtime_seconds"],
            "seeds": json.dumps(sorted(seeds))}


def two_phase_policy(ctx: dict, algorithm: str, K: int, policy: str, draw: int) -> tuple[dict, list[dict]]:
    config, A, B, truth, degree = ctx["config"], ctx["A"], ctx["B"], ctx["truth"], ctx["degree"]
    k1 = max(1, min(K, int(round(config["phase1_share"] * K))))
    seeds = [int(v) for v in ctx["orderings"][config["base_rule"]][:k1]]
    parts = (config["seed_base"], ctx["case"], algorithm, K, policy, draw)
    rng = _rng(*parts, "choices")
    state = run_round(ctx, seeds, algorithm, (*parts, 0))
    rounds, runtime, kept_rounds, unguarded = [], state["runtime_seconds"], 0, None
    for r in range(K + 1):
        row = {**_base_row(ctx, algorithm, K, policy, draw), "round": r, "n_seeds": len(seeds),
               **{k: v for k, v in evaluate(state["perm"], truth, seeds).items() if k != "total_correct"},
               "acc_all_unguarded": unguarded, "edge_ratio": state["objective"] / ctx["edge_scale"],
               "kept_previous": state["kept_previous"], "soft_check": state["soft_check"],
               "runtime_seconds": state["runtime_seconds"]}
        if len(seeds) >= K:
            rounds.append(row)
            break
        b = min(ctx["batch"], K - len(seeds))
        candidates = np.setdiff1d(np.arange(ctx["n"]), seeds)
        wrong = state["perm"][candidates] != truth[candidates]
        for kind in ("confidence", "consistency"):   # does each certainty point at the current errors?
            p, t, _ = certainty(kind, algorithm, state, ctx)
            row[f"auc_{kind}"] = auc_wrong(-(p + 1e-6 * t)[candidates], wrong)
        if policy == "two_phase_random":
            chosen, source = rng.choice(candidates, size=b, replace=False), "random"
        else:
            p, t, source = certainty(policy.removeprefix("two_phase_"), algorithm, state, ctx)
            chosen = least_certain(p, t, candidates, degree, b, rng)
        row.update(certainty_source=source, error_rate=float(np.mean(wrong)), batch=b,
                   hit_rate=float(np.mean(state["perm"][chosen] != truth[chosen])),
                   chosen_mean_degree=float(np.mean(degree[chosen])), chosen=json.dumps(sorted(map(int, chosen))))
        rounds.append(row)
        previous = state
        seeds = seeds + [int(v) for v in chosen]
        state = run_round(ctx, seeds, algorithm, (*parts, r + 1))
        runtime += state["runtime_seconds"]
        unguarded = evaluate(state["perm"], truth, seeds)["acc_all"]
        if config["guard"]:
            fixed = correct(previous["perm"], chosen, truth)
            fixed_objective = kept_edges(A, B, fixed)
            if fixed_objective > state["objective"]:
                # Retain fresh uncertainty computed with all current answers, not the old state.
                state = {**state, "perm": fixed, "objective": fixed_objective, "kept_previous": True,
                         "runtime_seconds": state["runtime_seconds"], "acc_all_restart_mean": float("nan")}
                kept_rounds += 1
    last = rounds[-1]
    final = {**_base_row(ctx, algorithm, K, policy, draw), "phase1_seeds": k1, "batch": ctx["batch"],
             "rounds": len(rounds) - 1, **evaluate(state["perm"], truth, seeds),
             "acc_all_restart_mean": state["acc_all_restart_mean"], "edge_ratio": last["edge_ratio"],
             "kept_previous_rounds": kept_rounds, "soft_check": state["soft_check"], "runtime_seconds": runtime,
             "seeds": json.dumps(sorted(seeds))}
    return final, rounds


def run_block(case: str, algorithm: str, K: int, config: dict, matcher=None,
              blas_threads: int | None = 2) -> tuple[list[dict], list[dict]]:
    """Every policy for one pair, algorithm and budget."""

    started = time.perf_counter()
    with thread_limit(blas_threads):
        A, B, truth = load_case(case, config)
        graph = GraphA(A)
        k_max = max(config["cases"][case]["budgets"])
        orderings = {rule: np.asarray(ordering(rule, graph, k_max, _rng(config["seed_base"], case, "order", rule)))
                     for rule in STATIC if rule != "random"}
        size = _size(case, config)
        ctx = {"case": case, "config": config, "A": A, "B": B, "truth": truth, "n": A.shape[0],
               "degree": A.sum(axis=1), "edge_scale": (A.sum() + B.sum()) / 4, "orderings": orderings,
               "size": size, "batch": int(config["cases"][case]["batch"]), "matcher": matcher or match}
        finals, rounds = [], []
        for draw in range(config["random_draws"][size]):
            finals.append(static_policy(ctx, algorithm, K, "random", draw))
        for rule in STATIC[1:]:
            finals.append(static_policy(ctx, algorithm, K, rule, 0))
        for policy in TWO_PHASE:
            for draw in range(config["control_draws"][size] if policy == "two_phase_random" else 1):
                final, path = two_phase_policy(ctx, algorithm, K, policy, draw)
                finals.append(final)
                rounds.extend(path)
    print(f"{case} {algorithm} K={K}: {len(finals)} policies in {(time.perf_counter() - started) / 60:.1f} min",
          flush=True)
    return finals, rounds


# --------------------------------------------------------------------------
# Collection and saved results
# --------------------------------------------------------------------------

def _append(path: Path, fields: tuple, rows: list[dict]) -> None:
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", restval="")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def _completed(root: Path) -> set:
    path = Path(root) / "finals.csv"
    if not path.exists():
        return set()
    with path.open() as handle:
        return {(r["case"], r["algorithm"], int(r["budget"])) for r in csv.DictReader(handle)}


def _drop_unfinished_rounds(root: Path) -> None:
    """Remove round rows of a block that was interrupted before its finals were written."""

    path = root / "rounds.csv"
    if not path.exists():
        return
    done = _completed(root)
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    keep = [r for r in rows if (r["case"], r["algorithm"], int(r["budget"])) in done]
    if len(keep) < len(rows):
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ROUND_FIELDS, extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(keep)
        temporary.replace(path)


def prepare(root: Path, config: dict, blas_threads) -> dict:
    """Save the design once; a different design needs a different output folder."""

    root.mkdir(parents=True, exist_ok=True)
    path = root / "settings.json"
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["config"] != config:
            raise ValueError(f"The design differs from {path}. Use a new output folder for a new design.")
        return config
    try:
        environment = collect_environment(blas_threads)
    except Exception as error:  # noqa: BLE001 - provenance is best effort
        environment = {"error": str(error)}
    path.write_text(json.dumps({"config": config, "code_sha256": code_hash, "environment": environment,
                                "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2))
    return config


def run(root=OUTPUT_DIR, config=None, cases=None, jobs: int = 2, blas_threads: int | None = 2,
        matcher=None) -> None:
    """Collect every missing (pair, algorithm, budget) block. Small pairs run before Wikipedia."""

    root = Path(root)
    config = prepare(root, config or default_config(), blas_threads)
    cases = [c for c in (cases or DEFAULT_CASES) if c in config["cases"]] or list(config["cases"])
    _drop_unfinished_rounds(root)
    done = _completed(root)
    todo = [(c, a, k) for c in sorted(cases, key=lambda c: _size(c, config) == "large")
            for a in config["algorithms"] for k in config["cases"][c]["budgets"] if (c, a, k) not in done]
    print(f"{len(todo)} blocks to run ({len(done)} already saved) in {root}", flush=True)

    def save(result):
        finals, rounds = result
        _append(root / "rounds.csv", ROUND_FIELDS, rounds)
        _append(root / "finals.csv", FINAL_FIELDS, finals)   # finals last: they mark the block done

    if jobs <= 1 or matcher is not None:
        for task in todo:
            save(run_block(*task, config, matcher, blas_threads))
        return
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(run_block, *task, config, None, blas_threads) for task in todo]
        for future in as_completed(futures):
            save(future.result())


def read(root=OUTPUT_DIR):
    import pandas as pd

    root = Path(root)
    finals = pd.read_csv(root / "finals.csv") if (root / "finals.csv").exists() else pd.DataFrame()
    rounds = pd.read_csv(root / "rounds.csv") if (root / "rounds.csv").exists() else pd.DataFrame()
    return finals, rounds


def summary(root=OUTPUT_DIR):
    """Mean total-correct share per policy (random draws averaged), with the two key differences."""

    finals, _ = read(root)
    if finals.empty:
        raise RuntimeError(f"No saved results in {root}.")
    table = finals.groupby(["case", "algorithm", "budget", "policy"]).acc_all.mean().unstack("policy")
    table = table[[p for p in POLICIES if p in table.columns]]
    for kind in ("confidence", "consistency"):
        active = f"two_phase_{kind}"
        if {active, "top_degree", "two_phase_random"} <= set(table.columns):
            table[f"{kind} vs top degree"] = table[active] - table["top_degree"]
            table[f"{kind} vs random batches"] = table[active] - table["two_phase_random"]
    return table


def diagnostics(root=OUTPUT_DIR):
    """Per active round: does certainty find the errors, and did the queried batch hit them?"""

    _, rounds = read(root)
    active = rounds[rounds.batch.notna() & (rounds.batch.astype(str) != "")]
    keys = ["case", "algorithm", "budget", "policy"]
    return active.groupby(keys).agg(rounds=("round", "size"), error_rate=("error_rate", "mean"),
                                    hit_rate=("hit_rate", "mean"), auc_confidence=("auc_confidence", "mean"),
                                    auc_consistency=("auc_consistency", "mean"),
                                    chosen_degree=("chosen_mean_degree", "mean"))


def plot(root=OUTPUT_DIR, algorithm: str = "faq"):
    """Total-correct share against budget, one panel per pair."""

    import matplotlib.pyplot as plt

    finals, _ = read(root)
    data = finals[finals.algorithm == algorithm]
    cases = list(dict.fromkeys(data.case))
    fig, axes = plt.subplots(1, len(cases), figsize=(4.2 * len(cases), 3.6), squeeze=False)
    styles = {"random": ("#888888", "-"), "top_degree": ("#204a87", "-"), "clustered": ("#ad494a", "-"),
              "distinguishing": ("#8856a7", "-"), "hub_free_distinguishing": ("#d276ae", "-"),
              "two_phase_confidence": ("#e69f00", "--"), "two_phase_consistency": ("#26986d", "--"),
              "two_phase_random": ("#111111", ":")}
    for ax, case in zip(axes[0], cases):
        means = data[data.case == case].groupby(["policy", "budget"]).acc_all.mean()
        for policy in POLICIES:
            if policy in means.index.get_level_values(0):
                color, line = styles[policy]
                series = means.loc[policy]
                ax.plot(series.index, series.values, line, color=color, marker="o", ms=3, label=LABELS[policy])
        ax.set(title=case.replace("_", " "), xlabel="total queries K", xticks=sorted(data.budget.unique()))
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("correct matches / n (seeds included)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(LABELS[algorithm])
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--blas-threads", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true", help="tiny design in results/active_test_smoke")
    parser.add_argument("--summary", action="store_true", help="print the saved results and exit")
    args = parser.parse_args()
    root = args.output_dir or (SMOKE_DIR if args.smoke else OUTPUT_DIR)
    if not args.summary:
        run(root, smoke_config() if args.smoke else None, args.cases, args.jobs, args.blas_threads)
    import pandas as pd

    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 3):
        print(summary(root).round(3))
        print(diagnostics(root).round(3))


if __name__ == "__main__":
    main()
