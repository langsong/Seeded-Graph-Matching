"""Real-pair follow-up: broader A-only candidates and transfer of saved synthetic selectors.

No model is fitted here. Native-pair repetitions change the test split, B's labels and
solver randomness. Separate real-topology controls redraw a perturbed B while keeping
A, the test vertices and seed sets fixed. Original first-look results are read only.

python -m real_data_experiments.broader_candidates --phase wide --cases enron_130_132 --budgets 10 --replicates 1
python -m real_data_experiments.broader_candidates --phase stability
python -m real_data_experiments.broader_candidates --phase controls
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_common.provenance import collect_environment, thread_limit
from selection_rule_comparison import fresh_pair_extremes as E
from selection_rule_comparison import learned_selector as LS
from selection_rule_comparison import graded_outcomes as G
from selection_rule_comparison.strategies import GraphA, STRATEGY_CODES
from . import first_look as F

OUTPUT_DIR = Path(os.environ.get("REAL_FOLLOW_UP_OUTPUT_DIR") or F.PACKAGE_DIR / "results" / "follow_up")
SEED_BASE = 20261112
CASES = {name: {**spec, "role": "exploratory", "window": "original"} for name, spec in F.PAIRS.items()}
for first, seconds, role in ((100, (101, 102), "development"), (145, (146, 147), "test"),
                             (160, (161, 162), "test")):
    for second in seconds:
        CASES[f"enron_{first}_{second}"] = {
            "get": {"name": "enron", "weeks": [first, second], "vertices": "common_lcc"},
            "budgets": (5, 10, 20), "role": role, "window": str(first),
        }
PRIMARY = ("wikipedia", "enron_130_132", "enron_145_146", "enron_145_147", "enron_160_161", "enron_160_162")
CONFIGS = {
    "percolation": {"algorithm": "percolation", "regularizer": None, "max_iter": 100},
    "faq": {"algorithm": "faq", "regularizer": None, "max_iter": 100},
    "faq_500": {"algorithm": "faq", "regularizer": None, "max_iter": 500},
    "goat_r100": {"algorithm": "goat", "regularizer": 100.0, "max_iter": 100},
    "goat_r300": {"algorithm": "goat", "regularizer": 300.0, "max_iter": 100},
}
PHASES = {
    "wide": {"cases": PRIMARY, "variants": ("percolation", "faq", "goat_r300"), "replicates": 3},
    "stability": {"cases": ("enron_100_101", "enron_100_102", "enron_130_131"),
                  "variants": ("faq", "faq_500", "goat_r100", "goat_r300"), "replicates": 3},
    "controls": {"cases": ("enron_130_132", "wikipedia"), "variants": ("percolation", "faq", "goat_r300"),
                 "replicates": 5},
}
POLICIES = ("random", "top_degree", "top_harmonic", "clustered", "spread", "distinguishing", "hub_free_distinguishing",
            *F.SCREENS,
            "frozen_pick_er", "frozen_pick_cl", "frozen_pick_paper", "synthetic_mean_pick")
LABELS = {**F.METHOD_LABELS, "random": "random mean", "frozen_pick_er": "ER score: expanded bank",
          "frozen_pick_cl": "CL score: expanded bank", "frozen_pick_paper": "PAPER score: expanded bank",
          "synthetic_mean_pick": "synthetic selectors: mean prediction"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _write_csv(path, frame):
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def saved_selectors():
    """All nine saved source settings per algorithm, in a fixed order; never choose using real outcomes."""
    fits = LS.final_models()
    if len(fits) != 27:
        raise RuntimeError("Expected the 27 saved final synthetic selectors.")
    return {algorithm: [(LS.cell_key(*cell), fit) for cell, fit in sorted(fits.items()) if cell[1] == algorithm]
            for algorithm in LS.ALGORITHMS}


def design():
    return {"version": 1, "seed_base": SEED_BASE, "pool": 1000, "random_sets": 30,
            "solver_runs": 3, "top_n": 10, "steps": 8, "swaps": 256, "test_share": F.TEST_SHARE,
            "control_copy_probability": 0.8, "cases": CASES, "configs": CONFIGS,
            "weights": F._read_json(F.OUTPUT_DIR / "settings.json")["weights"],
            "solver": F.S.SGM, "percolation": F.S.PERCOLATION,
            "selector_policy": "equal mean; matching algorithm and, for GOAT, matching source regularizer",
            "selector_hashes": {key: _digest(fit) for pairs in saved_selectors().values() for key, fit in pairs}}


def prepare():
    current = json.loads(json.dumps(design()))
    path = OUTPUT_DIR / "settings.json"
    if path.exists():
        if F._read_json(path) != current:
            raise ValueError("Follow-up settings or saved selectors changed; use a new REAL_FOLLOW_UP_OUTPUT_DIR.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        F._write_json(path, current)
        F._write_json(OUTPUT_DIR / "selector_sources.json",
                      {a: [{"source": key, "model": fit} for key, fit in pairs]
                       for a, pairs in saved_selectors().items()})
        F._write_json(OUTPUT_DIR / "environment.json", collect_environment(3))
    return current


def context(case, replicate, phase, settings):
    """A fixed natural pair, or a separate controlled perturbation; return the matching convention unchanged."""
    base = {"pairs": CASES, "test_share": settings["test_share"]}
    pair, A, B, truth, test, eligible, info = F.load_pair(case, base)
    code = F._code(case)
    planning_rep = 0 if phase == "controls" else replicate
    if phase == "controls":
        rng = np.random.default_rng([SEED_BASE, code, 30, replicate])
        upper = np.triu_indices(len(A), 1)
        probability = settings["control_copy_probability"] * A[upper] + (
            1 - settings["control_copy_probability"]) * A[upper].mean()
        aligned = np.zeros_like(A)
        aligned[upper] = rng.random(len(probability)) < probability
        aligned += aligned.T
        order = rng.permutation(len(A))
        B, truth = aligned[np.ix_(order, order)], np.argsort(order)
    elif replicate:
        rng = np.random.default_rng([SEED_BASE, code, 31, replicate])
        order = rng.permutation(len(A))
        B, truth = pair.B[np.ix_(order, order)], np.argsort(order)
        test = np.sort(rng.choice(len(A), len(test), replace=False))
        eligible = np.setdiff1d(np.arange(len(A)), test)
    info = {**info, "hash_b": F.rc._hash(B), "test_vertices": test.tolist(),
            "phase": phase, "replicate": replicate, "role": CASES[case]["role"],
            "window": CASES[case]["window"], "truth_hash": _digest(truth.tolist()),
            "planning_seed": F.rc._seed(SEED_BASE, code, planning_rep),
            "interpretation": "real-topology simulation" if phase == "controls" else "native real pair"}
    return A, B, truth, test, eligible, info


def candidate_plan(case, algorithm, k, graph, test, eligible, info, settings, phase, variant=None):
    """One A-only bank, then rank that same bank with all frozen scores and saved synthetic models."""
    seed = info["planning_seed"]
    pool = G.draw_sets(np.random.default_rng([seed, k, 1]), eligible, k, settings["pool"])
    raw = F.MP.PoolFeatures(graph, F.MP.UNIQUENESS_CAP[algorithm]).compute(pool)
    reference = {"budget": k, "pair_seed": seed,
                 "feature_mean": {f: float(raw[f].mean()) for f in F.CORE},
                 "feature_sd": {f: float(raw[f].std(ddof=1)) for f in F.CORE},
                 "weights_per_sd": settings["weights"]["sparse_er"][algorithm]}
    calculate = E.feature_calculator(graph, algorithm, reference)
    arrays, rows = [], []

    def add(seeds, detail, family, rank=1):
        arrays.append(np.sort(seeds))
        rows.append({"detail": detail, "family": family, "rank": rank,
                     "seed_vertices": LS._key(seeds)})

    n_random = 3 if phase == "stability" else (15 if case == "wikipedia" else settings["random_sets"])
    for j, seeds in enumerate(pool[:n_random]):
        add(seeds, "random", "random", j + 1)
    for rule in F.RULES:
        rng = np.random.default_rng([seed, k, STRATEGY_CODES[rule], 2])
        add(F.RM._ordering(rule, graph, k, rng, test), rule, "rule")
    z = {f: (raw[f] - reference["feature_mean"][f]) / reference["feature_sd"][f]
         if reference["feature_sd"][f] > 0 else np.zeros(len(pool)) for f in F.CORE}
    for model, suffix in (("sparse_er", "er"), ("chung_lu", "cl"), ("paper_pa", "paper")):
        scores = sum(settings["weights"][model][algorithm][f] * z[f] for f in F.CORE)
        add(pool[np.argmax(scores)], f"best_{suffix}", "screen")
    add(pool[np.argmax(raw["degree"])], "best_degree", "screen")
    if phase != "stability":
        prior = pd.DataFrame(rows)
        known = calculate(np.stack(arrays))
        anchor = prior.iloc[[int(np.argmax(known["score"]))]].assign(detail="score_tail")
        plan, extra = E.metric_plan(graph, algorithm, reference, pd.concat([prior, anchor]),
                                    prior[prior.family == "random"], settings,
                                    candidates=eligible, fast_objectives=True)
        for row, seeds in zip(plan.itertuples(), extra):
            detail = "high_score_er" if row.detail == "high_score" else row.detail
            add(seeds, detail, row.group, row.selection_rank)
        # The other two frozen objectives use the same normalization and the same search budget.
        for model, suffix in (("chung_lu", "cl"), ("paper_pa", "paper")):
            calculate_other = E.feature_calculator(graph, algorithm,
                              {**reference, "weights_per_sd": settings["weights"][model][algorithm]})
            known_sets = E.unique_sets(arrays)
            values = calculate_other(np.stack(known_sets))["score"]
            starts = [known_sets[np.argmax(values)], arrays[n_random], arrays[0]]
            selected = E.search_sets(calculate_other, "score", known_sets, values, starts, graph.n,
                       np.random.default_rng([seed, k, 40, F._code(model)]), settings["top_n"],
                       settings["steps"], settings["swaps"], candidates=eligible)
            for rank, seeds in enumerate(selected, 1):
                add(seeds, f"high_score_{suffix}", "search", rank)
    memberships = pd.DataFrame(rows)
    unique = memberships.drop_duplicates("seed_vertices").reset_index(drop=True)
    unique.insert(0, "candidate_id", np.arange(len(unique)))
    values = calculate(np.stack([LS._seeds(s) for s in unique.seed_vertices]))
    sets = pd.concat([unique[["candidate_id", "seed_vertices", "family", "detail"]], pd.DataFrame(values)], axis=1)
    memberships = memberships.merge(sets[["candidate_id", "seed_vertices"]], on="seed_vertices").drop(columns="seed_vertices")
    for model, suffix in (("sparse_er", "er"), ("chung_lu", "cl"), ("paper_pa", "paper")):
        sets[f"score_{suffix}"] = sum(settings["weights"][model][algorithm][f] * sets[f"z_{f}"] for f in F.CORE)
    if phase != "stability":
        # Report each metric's ten best encountered sets across the entire common bank.
        metric_columns = {"high_cut_degree": "degree", "high_pair_mass": "pair_mass",
                          "high_uniqueness": "uniqueness", "high_coverage": "coverage",
                          **{f"high_score_{m}": f"score_{m}" for m in ("er", "cl", "paper")}}
        memberships = memberships[~memberships.detail.isin(metric_columns)].copy()
        for detail, column in metric_columns.items():
            picked = sets.sort_values([column, "candidate_id"], ascending=[False, True]).head(settings["top_n"])
            memberships = pd.concat([memberships, pd.DataFrame({"candidate_id": picked.candidate_id,
                          "detail": detail, "family": "search", "rank": np.arange(1, len(picked) + 1)})], ignore_index=True)
    predictions = []
    fits = saved_selectors()[algorithm]
    if algorithm == "goat":
        regularizer = CONFIGS[variant or "goat_r300"]["regularizer"]
        fits = [(source, fit) for source, fit in fits
                if F.S.GOAT_REGULARIZER[source.split("/")[0]] == regularizer]
    for source, fit in fits:
        predicted, _ = LS.predict({**fit, "bootstrap": []}, sets)
        predictions.append(pd.DataFrame({"candidate_id": sets.candidate_id, "source": source,
                                        "predicted": predicted, "in_source_range": LS.in_range(fit, sets)}))
    predictions = pd.concat(predictions, ignore_index=True)
    summary = predictions.groupby("candidate_id").agg(predicted_mean=("predicted", "mean"),
                                                       source_range_share=("in_source_range", "mean"))
    sets = sets.join(summary, on="candidate_id")
    sets["source_count"] = len(fits)
    picks = []
    for suffix in ("er", "cl", "paper"):
        picks.append((f"frozen_pick_{suffix}", int(sets.loc[sets[f"score_{suffix}"].idxmax(), "candidate_id"])))
    picks.append(("synthetic_mean_pick", int(sets.loc[sets.predicted_mean.idxmax(), "candidate_id"])))
    for source, part in predictions.groupby("source", sort=False):
        in_range = part[part.in_source_range]
        ranked = in_range if len(in_range) else part
        picks.append((f"synthetic:{source}", int(ranked.loc[ranked.predicted.idxmax(), "candidate_id"])))
    memberships = pd.concat([memberships, pd.DataFrame([
        {"candidate_id": c, "detail": name, "family": "policy", "rank": 1} for name, c in picks])], ignore_index=True)
    assert all(len(LS._seeds(s)) == k and not np.intersect1d(LS._seeds(s), test).size for s in sets.seed_vertices)
    return sets, memberships, predictions, reference


def folder_for(task):
    phase, case, variant, k, replicate = task
    return OUTPUT_DIR / phase / case / variant / f"k{k:03d}" / f"rep_{replicate:03d}"


def prepare_job(task, settings):
    phase, case, variant, k, replicate = task
    folder = folder_for(task)
    config = CONFIGS[variant]
    A, B, truth, test, eligible, info = context(case, replicate, phase, settings)
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "sets.csv").exists():
        sets, members, predictions, reference = candidate_plan(
            case, config["algorithm"], k, GraphA(A), test, eligible, info, settings, phase, variant)
        _write_csv(folder / "memberships.csv", members)
        _write_csv(folder / "predictions.csv", predictions)
        F._write_json(folder / "pair.json", {**info, "score_reference": reference})
        _write_csv(folder / "sets.csv", sets)
    else:
        saved = F._read_json(folder / "pair.json")
        for key in ("hash_a", "hash_b", "truth_hash", "test_vertices", "planning_seed"):
            if saved[key] != info[key]:
                raise ValueError(f"Saved follow-up context changed: {folder}, {key}")
        sets = pd.read_csv(folder / "sets.csv")
    return folder, A, B, truth, test, sets


def plan_job(task, settings):
    with thread_limit(3):
        _, _, _, _, _, sets = prepare_job(task, settings)
    return f"{'/'.join(map(str, task))}: {len(sets)} candidates ready"


def collect_job(task, settings):
    phase, case, variant, k, replicate = task
    config = CONFIGS[variant]
    with thread_limit(3):
        folder, A, B, truth, test, sets = prepare_job(task, settings)
        done = G._completed_ids(folder / "runs.csv", "run_id")
        rows, written = G._Rows(folder / "runs.csv"), 0
        old_limit = F.S.SGM["max_iter"]
        F.S.SGM["max_iter"] = config["max_iter"]
        try:
            count = 1 if config["algorithm"] == "percolation" else settings["solver_runs"]
            for repeat in range(count):
                # Common solver randomness across every candidate; identical sets are evaluated once.
                solver_seed = F.rc._seed(SEED_BASE, F._code(case), replicate, repeat, 90)
                for row in sets.itertuples():
                    run_id = 10 * int(row.candidate_id) + repeat
                    if run_id in done:
                        continue
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        out = F.RM.run_one(config["algorithm"], "sparse_er", (A, B, truth),
                              LS._seeds(row.seed_vertices), test, solver_seed, config["regularizer"])
                    if out["error"]:
                        raise RuntimeError(out["error"])
                    rows.write({"candidate_id": row.candidate_id, "run_id": run_id, "repeat": repeat,
                                "solver_seed": solver_seed, **out, "warning_count": len(caught),
                                "warnings": json.dumps([str(w.message) for w in caught])})
                    written += 1
        finally:
            rows.close()
            F.S.SGM["max_iter"] = old_limit
    return f"{phase}/{case}/{variant}/k{k}/rep{replicate}: {written} new runs"


def tasks(phase, cases=None, variants=None, budgets=None, replicates=None):
    spec = PHASES[phase]
    return [(phase, case, variant, k, rep)
            for case in (cases or spec["cases"]) for variant in (variants or spec["variants"])
            for k in (budgets or ((10, 20, 30) if case == "wikipedia" else
                                  (5, 10, 20) if case.startswith("enron") else CASES[case]["budgets"]))
            for rep in range(spec["replicates"] if replicates is None else replicates)]


def run(phase="wide", cases=None, variants=None, budgets=None, replicates=None, jobs=3, plan_only=False):
    settings = prepare()
    todo = tasks(phase, cases, variants, budgets, replicates)
    worker = plan_job if plan_only else collect_job
    print(f"{'Preparing' if plan_only else 'Collecting'} {len(todo)} {phase} tasks; resumes saved work.", flush=True)
    if jobs == 1:
        for task in todo:
            print(worker(task, settings), flush=True)
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        pool = ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn"))
        try:
            futures = {pool.submit(worker, task, settings): task for task in todo}
            for future in as_completed(futures):
                print(future.result(), flush=True)
            pool.shutdown(wait=True)
        except BaseException:
            workers = list((getattr(pool, "_processes", None) or {}).values())
            pool.shutdown(wait=False, cancel_futures=True)
            for worker in workers:
                worker.terminate()
            raise


def inventory():
    rows = []
    for name, spec in CASES.items():
        if name not in PRIMARY and spec["role"] != "development":
            continue
        pair = F.D.get_pair(**F._get_kwargs(spec))
        rows.append({"case": name, "role": spec["role"], "time window": spec["window"], "n": pair.n,
                     "test vertices": round(F.TEST_SHARE * pair.n),
                     "edge correlation": F.D.describe(pair.A, pair.B)["edge_correlation"]})
    return pd.DataFrame(rows).set_index("case").round(3)


def score_existing():
    """Transfer predictions on the original saved candidate sets, with no new matching or model fitting."""
    frame = F.per_set().rename(columns={"pair": "case", "set_id": "candidate_id"})
    frame["phase"], frame["replicate"], frame["family"] = "first_look", 0, frame["design"]
    fits = saved_selectors()
    frames = []
    for (algorithm, variant), part in frame.groupby(["algorithm", "variant"], sort=False):
        sources = fits[algorithm]
        if algorithm == "goat":
            sources = [(source, fit) for source, fit in sources
                       if F.S.GOAT_REGULARIZER[source.split("/")[0]] == CONFIGS[variant]["regularizer"]]
        predicted = np.stack([LS.predict({**fit, "bootstrap": []}, part)[0] for _, fit in sources])
        inside = np.stack([LS.in_range(fit, part) for _, fit in sources])
        frames.append(part.assign(predicted_mean=predicted.mean(axis=0), source_range_share=inside.mean(axis=0),
                                  source_count=len(sources)))
    scored = pd.concat(frames, ignore_index=True)
    members = scored[KEYS + ["method", "family"]].rename(columns={"method": "detail"})
    picks = scored.loc[scored.groupby(KEYS[:5]).predicted_mean.idxmax(), KEYS]
    members = pd.concat([members, picks.assign(detail="synthetic_mean_pick", family="policy")], ignore_index=True)
    return scored, members


def collection_status(phase="wide"):
    rows = []
    for path in sorted((OUTPUT_DIR / phase).glob("*/*/k*/rep_*/sets.csv")):
        case, variant, budget, rep = path.parent.parts[-4:]
        expected = len(pd.read_csv(path)) * (1 if CONFIGS[variant]["algorithm"] == "percolation" else 3)
        runs = path.with_name("runs.csv")
        done = len(pd.read_csv(runs)) if runs.exists() and runs.stat().st_size else 0
        rows.append({"case": case, "variant": variant, "budget": int(budget[1:]),
                     "replicate": int(rep[4:]), "runs": done, "planned": expected})
    return pd.DataFrame(rows)


def read(phase="wide"):
    sets, memberships, predictions = [], [], []
    for path in sorted((OUTPUT_DIR / phase).glob("*/*/k*/rep_*/sets.csv")):
        folder = path.parent
        case, variant, budget, rep = folder.parts[-4:]
        meta = {"phase": phase, "case": case, "variant": variant,
                "budget": int(budget[1:]), "replicate": int(rep[4:])}
        s = pd.read_csv(path)
        run_path = folder / "runs.csv"
        if not run_path.exists():
            continue
        runs = pd.read_csv(run_path)
        ok = runs[F._ok(runs)]
        outcome = ok.groupby("candidate_id").agg(accuracy=("accuracy_test", "mean"),
                  runs=("accuracy_test", "size"), converged=("converged", "mean"),
                  iterations=("n_iter", "mean"), warnings=("warning_count", "sum"))
        expected = 1 if CONFIGS[variant]["algorithm"] == "percolation" else 3
        outcome.loc[outcome.runs < expected, "accuracy"] = np.nan
        if not s.candidate_id.isin(outcome.index[outcome.runs >= expected]).all():
            outcome["accuracy"] = np.nan  # Compare only completed, common candidate banks.
        sets.append(s.join(outcome, on="candidate_id").assign(**meta))
        memberships.append(pd.read_csv(folder / "memberships.csv").assign(**meta))
        predictions.append(pd.read_csv(folder / "predictions.csv").assign(**meta))
    return tuple(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
                 for parts in (sets, memberships, predictions))


KEYS = ["phase", "case", "variant", "budget", "replicate", "candidate_id"]


def policy_table(sets, memberships):
    if sets.empty:
        return pd.DataFrame()
    joined = memberships.merge(sets[KEYS + ["accuracy"]], on=KEYS).dropna(subset=["accuracy"])
    selected = joined[joined.detail.isin(POLICIES)]
    # Average solver runs first, seed sets within a policy second, and test-split repetitions last.
    per_rep = selected.groupby(KEYS[:-1] + ["detail"]).accuracy.mean().unstack("detail")
    return per_rep.groupby(KEYS[:4]).mean().rename(columns=LABELS).round(3)


def policy_repetitions(sets, memberships):
    if sets.empty:
        return pd.DataFrame()
    joined = memberships.merge(sets[KEYS + ["accuracy"]], on=KEYS).dropna(subset=["accuracy"])
    joined = joined[joined.detail.isin(POLICIES)]
    return joined.groupby(KEYS[:-1] + ["detail"]).accuracy.mean().unstack("detail").rename(columns=LABELS).round(3)


def control_means(sets, memberships):
    """Average completed controlled-B draws for the identical A-only candidate bank."""
    if sets.empty:
        return sets, memberships
    keys = ["case", "variant", "budget", "candidate_id"]
    for _, part in sets.groupby(keys):
        if part.seed_vertices.nunique() != 1:
            raise ValueError("Controlled-B draws must reuse identical candidate sets.")
    outcome = sets.dropna(subset=["accuracy"]).groupby(keys).agg(
        accuracy=("accuracy", "mean"), B_draws=("replicate", "nunique"))
    first = sets.drop_duplicates(keys).drop(columns="accuracy").set_index(keys).join(outcome).reset_index()
    first["replicate"] = 0
    members = memberships.drop(columns="replicate").drop_duplicates().assign(replicate=0)
    return first, members


def plot_scores(sets, memberships, case, variant="faq", replicate=0):
    import matplotlib.pyplot as plt
    part = sets[(sets.case == case) & (sets.variant == variant) & (sets.replicate == replicate)].dropna(subset=["accuracy"])
    if part.empty:
        return None
    budgets = sorted(part.budget.unique())
    fig, axes = plt.subplots(len(budgets), 2, figsize=(10.5, 3.1 * len(budgets) + 1.4), squeeze=False, sharey=True)
    styles = {"random": ("#b9b8b2", "o"), "rule": ("#eb6834", "s"), "screen": ("#5598e7", "X")}
    for r, k in enumerate(budgets):
        sub = part[part.budget == k]
        for c, x in enumerate(("score_er", "predicted_mean")):
            ax = axes[r, c]
            F._style(ax)
            for family, (color, marker) in styles.items():
                chosen = sub[sub.family == family]
                ax.scatter(chosen[x], chosen.accuracy, s=12 if family == "random" else 25, color=color,
                           marker=marker, alpha=0.65, linewidths=0, label=family)
            members = memberships[(memberships.case == case) & (memberships.variant == variant)
                      & (memberships.budget == k) & (memberships.replicate == replicate)]
            for detail, style in E.F.METRIC_STYLES.items():
                name = "high_score_er" if detail == "high_score" else detail
                chosen = sub[sub.candidate_id.isin(members.loc[members.detail == name, "candidate_id"])]
                if len(chosen):
                    label = E.F.SELECTION_LABELS[detail].replace("exact top", "exact eligible top")
                    if detail == "high_score":
                        label = "ER " + label
                    ax.scatter(chosen[x], chosen.accuracy, **style, label=label, linewidths=0.4,
                               edgecolors="white", alpha=0.8)
            for suffix, color in (("cl", "#4d92a8"), ("paper", "#a86c46")):
                chosen = sub[sub.candidate_id.isin(members.loc[members.detail == f"high_score_{suffix}", "candidate_id"])]
                if len(chosen):
                    ax.scatter(chosen[x], chosen.accuracy, s=35, marker="*", color=color,
                               label=f"{suffix.upper()} score: 10 best found", alpha=0.8)
            picks = memberships[(memberships.case == case) & (memberships.variant == variant)
                    & (memberships.budget == k) & (memberships.replicate == replicate)
                    & (memberships.detail == "synthetic_mean_pick")]
            chosen = sub[sub.candidate_id.isin(picks.candidate_id)]
            ax.scatter(chosen[x], chosen.accuracy, color="#202020", marker="*", s=90, label="synthetic mean pick")
            ax.set_title(f"{k} seeds", loc="left")
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel("frozen ER weighted score" if c == 0 else
                          f"mean prediction of {int(sub.source_count.iloc[0])} synthetic selectors")
            if c == 0:
                ax.set_ylabel("mean test accuracy")
            else:
                ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", lw=0.8, zorder=0)
                ax.set_xlim(-0.03, 1.03)
    context_label = (f"mean over {int(part.B_draws.min())} controlled Bs" if "B_draws" in part
                     else f"repetition {replicate}")
    fig.suptitle(f"{case} · {variant} · {context_label}: same sets in both columns", x=0.01, ha="left")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.97),
               frameon=False, fontsize=8)
    legend_rows = int(np.ceil(len(axes[0, 0].get_legend_handles_labels()[1]) / 3))
    height = fig.get_figheight()
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.6 / height,
                        top=1 - max(1.15, 0.5 + 0.2 * legend_rows) / height, hspace=0.42, wspace=0.10)
    return fig


def transfer_table(sets, predictions):
    if sets.empty:
        return pd.DataFrame()
    joined = predictions.merge(sets[KEYS + ["accuracy"]], on=KEYS).dropna(subset=["accuracy"])
    rows = []
    for key, part in joined.groupby(KEYS[:5] + ["source"]):
        rows.append(dict(zip(KEYS[:5] + ["source"], key),
                    MAE=float(np.mean(np.abs(part.predicted - part.accuracy))),
                    Spearman=F._spearman(part.predicted, part.accuracy),
                    in_source_range=float(part.in_source_range.mean())))
    return pd.DataFrame(rows).groupby(KEYS[:4] + ["source"])[["MAE", "Spearman", "in_source_range"]].mean().round(3)


def source_policy_table(sets, memberships, predictions):
    if sets.empty:
        return pd.DataFrame()
    picks = memberships[memberships.detail.str.startswith("synthetic:")].copy()
    picks["source"] = picks.detail.str.removeprefix("synthetic:")
    picks = picks.merge(predictions, on=KEYS + ["source"]).merge(sets[KEYS + ["accuracy"]], on=KEYS)
    return picks.dropna(subset=["accuracy"]).groupby(KEYS[:4] + ["source"]).agg(
        selected_accuracy=("accuracy", "mean"), predicted_accuracy=("predicted", "mean"),
        selected_in_source_range=("in_source_range", "mean")).round(3)


def stability_table(sets):
    if sets.empty:
        return pd.DataFrame()
    return sets.groupby(["case", "budget", "variant"]).agg(
        accuracy=("accuracy", "mean"), convergence_share=("converged", "mean"),
        mean_iterations=("iterations", "mean"), warning_count=("warnings", "sum")).round(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, default="wide")
    parser.add_argument("--cases", nargs="+", choices=CASES)
    parser.add_argument("--variants", nargs="+", choices=CONFIGS)
    parser.add_argument("--budgets", nargs="+", type=int)
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="save A-only candidates and predictions; no matching")
    args = parser.parse_args()
    if args.status:
        print(collection_status(args.phase).to_string(index=False))
    else:
        run(args.phase, args.cases, args.variants, args.budgets, args.replicates, args.jobs, args.plan_only)


if __name__ == "__main__":
    main()
