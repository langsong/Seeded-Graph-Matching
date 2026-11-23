"""Read-only analysis for strategy_and_active_v2.py (formerly overnight.py); no collection, fitting, or matching calls."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import strategy_and_active_v2 as O
from . import strategy_comparison as C

METRICS = {"accuracy_test": "Fixed-test accuracy", "accuracy_nonseed": "Nonseed accuracy",
           "accuracy_all": "Accuracy including seeds"}
ACTIVE_LABELS = {"random": "Random continuation", "structural": "Distinguishing continuation",
                 "uncertainty": "Ensemble uncertainty", "uncertainty_impact": "Uncertainty × unresolved neighbors"}


def read_strategy(root=O.OUTPUT_DIR):
    root = Path(root)
    config = json.loads((root / "settings.json").read_text())["config"]
    rows, inventory = [], []
    for path in sorted((root / "strategy").glob("*/*/*/*/plan.json")):
        plan = json.loads(path.read_text())
        case, split, algorithm, k = (plan[x] for x in ("case", "split", "algorithm", "budget"))
        lookup = {s["candidate_id"]: s for s in plan["sets"]}
        repeats = 1 if algorithm == "percolation" else config["selection"]["solver_runs"][O.size(case, config)]
        cache = {}
        for cid in lookup:
            files = [path.parent / f"c{cid:04d}_r{r}.npz" for r in range(repeats)]
            if all(p.exists() for p in files):
                outcomes = [O.read(p)[0] for p in files]
                cache[cid] = {m: np.mean([o[m] for o in outcomes]) for m in METRICS}
                cache[cid]["solver_sd"] = float(np.std([o["accuracy_test"] for o in outcomes], ddof=1)) if repeats > 1 else 0.
        complete_draws = 0
        for draw in range(config["selection"]["repeats"][O.size(case, config)]):
            members = [m for m in plan["members"] if m["draw"] == draw or m["method"] == "top_degree"]
            if set(m["method"] for m in members) != set(plan["methods"]):
                raise ValueError(f"Incomplete selection plan: {path}")
            # Include a draw only when every method has all solver repetitions.
            if not all(m["candidate_id"] in cache for m in members):
                continue
            complete_draws += 1
            for m in members:
                s = lookup[m["candidate_id"]]
                rows.append({**m, **cache[m["candidate_id"]], "case": case, "split": split,
                             "budget": k, "algorithm": algorithm, "draw": draw,
                             "seed_vertices": s["seed_vertices"], "eligible": len(plan["eligible"]),
                             "score_er": s[f"{algorithm}__score_er"]})
        inventory.append({"case": case, "split": split, "algorithm": algorithm, "budget": k,
                          "complete_draws": complete_draws,
                          "planned_draws": config["selection"]["repeats"][O.size(case, config)],
                          "saved_evaluations": len(list(path.parent.glob("c*_r*.npz")))})
    calibration = root / "calibration" / "budgets.json"
    selected = json.loads(calibration.read_text()) if calibration.exists() else {}
    return {"config": config, "draws": pd.DataFrame(rows), "inventory": pd.DataFrame(inventory),
            "calibration": selected}


def summarize_strategy(data, metric="accuracy_test"):
    d = data["draws"]
    if d.empty:
        return pd.DataFrame()
    if metric not in METRICS:
        raise ValueError(metric)
    keys = ["case", "algorithm", "budget"]
    paired = d.merge(d[d.method == "random"][[*keys, "split", "draw", metric]].rename(columns={metric: "baseline"}),
                     on=[*keys, "split", "draw"], validate="many_to_one")
    paired["paired_gain"] = paired[metric] - paired.baseline
    # Equal partition weight; avoid giving a partially finished partition more weight just because it has more draws.
    by_split = paired.groupby([*keys, "method", "split"]).agg(
        mean=(metric, "mean"), gain=("paired_gain", "mean"), random_mean=("baseline", "mean"))
    result = by_split.groupby([*keys, "method"]).mean().reset_index()
    spread = paired.groupby([*keys, "method"]).agg(
        q10=(metric, lambda x: x.quantile(.1)), q90=(metric, lambda x: x.quantile(.9)),
        splits=("split", "nunique"), draws=("draw", "size"), score_mean=("score_er", "mean"))
    result = result.merge(spread.reset_index(), on=[*keys, "method"])
    result["pilot_random_accuracy"] = [data["calibration"][r.case][r.algorithm]["pilot_means"][str(r.budget)]
                                       for r in result.itertuples()]
    return result


def plot_config(data):
    config = data["config"]
    return {"methods": [*O.CORE_METHODS, "hub_free_distinguishing", "entropy"],
            "algorithms": list(C.ALGORITHMS), "cases": config["cases"]}


def plot_strategy(summary, data, algorithm, root, metric="accuracy_test", bands=True):
    fig = C.plot_curves(summary, plot_config(data), algorithm, root, bands=bands, fit_y=True)
    for ax in fig.axes:
        if ax.get_ylabel():
            ax.set_ylabel(METRICS[metric])
    fig.suptitle(f"{C.ALGORITHM_LABELS[algorithm]} · independent random-budget calibration\n"
                 "Means weight test partitions equally; panel-specific y-axis ranges", y=.995)
    C._save(fig, root, f"{algorithm}_accuracy_curves")
    return fig


def overlaps(data, case="wikipedia", budget=20, algorithm="faq"):
    d = data["draws"]
    if d.empty:
        return pd.DataFrame()
    selected = d[(d.case == case) & (d.budget == budget) & (d.algorithm == algorithm)]
    if selected.empty:
        first = d.iloc[0]
        case, budget, algorithm = first.case, first.budget, first.algorithm
        selected = d[(d.case == case) & (d.budget == budget) & (d.algorithm == algorithm)]
    rows = []
    for split, frame in selected.groupby("split"):
        policies = {}
        for method, part in frame.groupby("method", sort=False):
            # Exact degree is reused as a paired baseline, but is one set per partition.
            if method == "top_degree":
                part = part.iloc[:1]
            policies[method] = [set(json.loads(v)) for v in part.seed_vertices]
        for left, right in itertools.product(policies, repeat=2):
            values = [len(a & b) / budget for i, a in enumerate(policies[left])
                      for j, b in enumerate(policies[right])
                      if left != right or i != j or len(policies[left]) == 1]
            rows.append({"case": case, "budget": budget, "algorithm": algorithm,
                         "left": left, "right": right, "shared_fraction": np.mean(values),
                         "random_expected": budget / frame.eligible.iloc[0]})
    return pd.DataFrame(rows).groupby(["case", "budget", "algorithm", "left", "right"], sort=False).mean().reset_index()


def read_active(root=O.OUTPUT_DIR):
    rows = [O.read(p)[0] for p in Path(root).glob("active/*/*/*/*/*/k*.npz")]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    keys = ["case", "algorithm", "base", "repetition", "budget"]
    balanced = frame.groupby(keys).policy.nunique().eq(len(O.POLICIES)).rename("balanced")
    return frame.merge(balanced.reset_index(), on=keys)


def active_summary(frame, metric="accuracy_test"):
    if frame.empty:
        return pd.DataFrame()
    frame = frame[frame.balanced]
    keys = ["case", "algorithm", "base", "budget", "policy"]
    by_split = frame.groupby([*keys, "split"])[metric].mean()
    mean = by_split.groupby(keys).mean().rename("mean")
    counts = frame.groupby(keys).agg(repetitions=("repetition", "nunique"),
                                    q10=(metric, lambda s: s.quantile(.1)), q90=(metric, lambda s: s.quantile(.9)))
    return pd.concat([mean, counts], axis=1).reset_index()


def plot_active(summary, algorithm, root, metric="accuracy_test"):
    import matplotlib.pyplot as plt
    part = summary[summary.algorithm == algorithm]
    cases, bases = list(part.case.unique()), list(part.base.unique())
    fig, axes = plt.subplots(len(bases), len(cases), figsize=(5 * len(cases), 3.6 * len(bases)), squeeze=False)
    for i, base in enumerate(bases):
        for j, case in enumerate(cases):
            ax = axes[i, j]
            for policy in O.POLICIES:
                line = part[(part.case == case) & (part.base == base) & (part.policy == policy)].sort_values("budget")
                ax.plot(line.budget, line["mean"], label=ACTIVE_LABELS[policy], marker=".")
                ax.fill_between(line.budget, line.q10, line.q90, alpha=.06)
            ax.set(title=f"{case.replace('_', ' ')}\nInitial rule: {base}", xlabel="Verified seeds",
                   ylabel=METRICS[metric], ylim=(0, 1))
            ax.grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"{C.ALGORITHM_LABELS[algorithm]}: repeated active queries", y=.99)
    fig.tight_layout(rect=(0, .09, 1, .95))
    C._save(fig, root, f"active_{algorithm}_{metric}")
    return fig


def read_diagnostics(root=O.OUTPUT_DIR):
    rows = []
    for p in Path(root).glob("diagnostic/*/*/*/*/plan.json"):
        plan = json.loads(p.read_text())
        complete = all((p.parent / f"v{v:04d}.json").exists() for v in plan["candidates"])
        for q in p.parent.glob("v*.json"):
            rows.append({**json.loads(q.read_text()), "state_complete": complete,
                         "planned_candidates": len(plan["candidates"])})
    return pd.DataFrame(rows)


def diagnostic_summary(frame):
    from scipy.stats import spearmanr
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in frame[frame.state_complete].groupby(["case", "algorithm", "budget", "repetition"]):
        row = dict(zip(["case", "algorithm", "budget", "repetition"], keys))
        row.update(candidates=len(g), mean_gain=g.gain_test.mean(), best_observed_gain=g.gain_test.max())
        for feature in ("uncertainty", "uncertainty_impact", "degree", "distinguishing_score"):
            row[f"rho_{feature}"] = (spearmanr(g[feature], g.gain_test).statistic
                                       if g[feature].nunique() > 1 and g.gain_test.nunique() > 1 else np.nan)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby(["case", "algorithm", "budget"]).agg(
        states=("repetition", "nunique"), candidates_per_state=("candidates", "mean"),
        mean_gain=("mean_gain", "mean"), best_observed_gain=("best_observed_gain", "mean"),
        rho_uncertainty=("rho_uncertainty", "mean"), rho_impact=("rho_uncertainty_impact", "mean"),
        rho_degree=("rho_degree", "mean"), rho_distinguishing=("rho_distinguishing_score", "mean"))


def export(root, **frames):
    folder = Path(root) / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(folder / f"{name}.csv", index=False)
