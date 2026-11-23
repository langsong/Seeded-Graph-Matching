"""Read-only analysis of collect_score_ranges outputs. No collection or fitting imports.

The real and synthetic score_range_analysis notebooks call these functions.
Every plotted observation is a unique seed set within one bank and algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "selection_rule_comparison/data/score_range_collection"
DEFAULT_EXPORT = REPO_ROOT / "selection_rule_comparison/data/score_range_analysis"
ALGORITHMS = ("percolation", "faq", "goat")
ALG_LABELS = {"percolation": "Percolation", "faq": "FAQ", "goat": "GOAT"}
MODEL_LABELS = {"sparse_er": "ER", "chung_lu": "Chung–Lu", "paper_pa": "PAPER"}
SUFFIX = {"sparse_er": "er", "chung_lu": "cl", "paper_pa": "paper"}
CASE_LABELS = {"worm": "Worm full", "worm_lcc": "Worm common LCC", "worm_male": "Worm male",
               "wikipedia": "Wikipedia EN/FR", **{f"enron_{a}_{b}": f"Enron {a}–{b}" for a, b in
               [(130, 131), (130, 132), (145, 147), (160, 162)]}}
RULES = ("top_degree", "top_harmonic", "clustered", "spread", "distinguishing", "hub_free_distinguishing")
RULE_LABELS = {"top_degree": "Highest degree", "top_harmonic": "Highest harmonic", "clustered": "Clustered",
               "spread": "Spread", "distinguishing": "Distinguishing", "hub_free_distinguishing": "Hub-free distinguishing"}
COLORS = {"random": "#969696", "range": "#2878b5", "contrast": "#9254a1", "endpoint": "#df8b25",
          "rule": "#22835e"}
MARKERS = {"random": "o", "range": "o", "contrast": "^", "endpoint": "D", "rule": "*"}
FAMILY_LABELS = {"random": "Uniform random", "range": "Score-range picks", "contrast": "Feature contrasts",
                 "endpoint": "Score endpoints", "rule": "Hand rules"}
FEATURES = {"degree": "Cut degree", "total_degree_mean": "Total degree", "harmonic": "Harmonic",
            "pair_mass": "Pair mass", "uniqueness": "Distance uniqueness", "coverage": "Coverage ≥1",
            "coverage_two": "Coverage ≥2", "coverage_three": "Coverage ≥3",
            "adjacency_signature_collision_fraction": "Signature collisions",
            "adjacency_signature_normalized_entropy": "Signature entropy"}
KEY = ["bank_id", "algorithm", "candidate_id"]
POLICY_LABELS = {"random_mean": "Random mean", **RULE_LABELS, "frozen_er": "Frozen ER pick",
                 "frozen_cl": "Frozen CL pick", "frozen_paper": "Frozen PAPER pick",
                 "frozen_own": "Own frozen pick", "learned_mean": "Learned pick (unrestricted)",
                 "learned_guarded": "Learned pick (in range)", "oracle": "Best observed (hindsight)"}


def rho(x, y):
    frame = pd.DataFrame({"x": np.asarray(x), "y": np.asarray(y)}).dropna()
    if len(frame) < 3 or frame.x.nunique() < 2 or frame.y.nunique() < 2:
        return np.nan
    return float(frame.x.rank().corr(frame.y.rank()))


def label(row):
    if row["kind"] == "real":
        return f"{CASE_LABELS[row['case']]} · k={int(row['budget'])}"
    return f"{MODEL_LABELS[row['model']]} · d={row['mean_degree']:g}, ρ={row['rho']:g}, k={int(row['budget'])}"


@dataclass
class Collection:
    root: Path
    settings: dict
    banks: pd.DataFrame
    sets: pd.DataFrame
    runs: pd.DataFrame
    memberships: pd.DataFrame
    predictions: pd.DataFrame
    coverage: pd.DataFrame

    def per_set(self, view="mean"):
        """Common complete blocks only: mean=all planned blocks, single=draw 0/repeat 0."""
        if view not in ("mean", "single"):
            raise ValueError("view must be mean or single")
        r = self.runs
        if view == "single":
            r = r[(r.draw == 0) & (r.repeat == 0)]
        groups = r.groupby(KEY, sort=False)
        outcomes = groups.agg(accuracy=("accuracy_test", "mean"), accuracy_sd=("accuracy_test", "std"),
                              accuracy_min=("accuracy_test", "min"), accuracy_max=("accuracy_test", "max"),
                              n_runs=("accuracy_test", "size"), n_draws=("draw", "nunique"),
                              converged_share=("converged", "mean"), iterations=("n_iter", "mean"),
                              runtime_seconds=("runtime_seconds", "sum"))
        out = self.sets.merge(outcomes, on=KEY, how="left", validate="one_to_one")
        expected = np.ones(len(out), dtype=int) if view == "single" else out.expected_runs.to_numpy()
        if not np.array_equal(out.n_runs.fillna(0).to_numpy(), expected):
            raise ValueError(f"Incomplete {view} evaluation blocks. Inspect inventory or resume collection first.")
        out["accuracy_sd"] = out.accuracy_sd.fillna(0)
        out["view"] = view
        return out


def load_collection(root=DEFAULT_ROOT, kind="real", frozen_reference="er"):
    """Use saved features/predictions, never live model files or matching code."""
    root = Path(root)
    settings = json.loads((root / "settings.json").read_text())
    banks, sets, runs, members, predictions, coverage = [], [], [], [], [], []
    for task in settings["banks"]:
        if task["kind"] != kind:
            continue
        folder = root / task["id"]
        meta = json.loads((folder / "pair.json").read_text())
        s = pd.read_csv(folder / "sets.csv")
        m = pd.read_csv(folder / "memberships.csv").fillna({"contrast_id": ""})
        p = pd.read_csv(folder / "predictions.csv")
        c = pd.read_csv(folder / "coverage.csv")
        common = {"bank_id": task["id"], "kind": kind, "case": task["case"], "budget": task["budget"],
                  "model": task.get("model", "real"), "mean_degree": task.get("mean_degree", np.nan),
                  "rho": task.get("rho", np.nan), "pair_index": task.get("pair_index", 0),
                  "n": int(meta.get("n", meta.get("graph_config", {}).get("vertex_count", 600))),
                  "n_test": len(meta["test_vertices"])}
        if s.candidate_id.duplicated().any() or s.seed_vertices.duplicated().any():
            raise ValueError(f"Duplicate candidate in {folder}")
        for a in task["algorithms"]:
            part = s[[col for col in s if "__" not in col]].copy()
            for col in s:
                if col.startswith(f"{a}__"):
                    part[col.split("__", 1)[1]] = s[col]
            part = part.rename(columns={"predicted_mean": "predicted_accuracy"})
            part["contrast_uniqueness"] = s.uniqueness
            part["contrast_uniqueness_cap"] = meta["contrast_uniqueness_cap"]
            own = SUFFIX[task["model"]] if kind == "synthetic" else frozen_reference
            part["frozen_score"] = part[f"score_{own}"]
            part["frozen_reference"] = own
            expected = (settings["synthetic_draws"] if kind == "synthetic" else
                        1 if a == "percolation" else settings["wiki_solver_runs"] if task["case"] == "wikipedia"
                        else settings["small_solver_runs"])
            part = part.assign(**common, algorithm=a, expected_runs=expected)
            r = pd.read_csv(folder / f"{a}_runs.csv").assign(bank_id=task["id"])
            if r.duplicated(["candidate_id", "draw", "repeat"]).any():
                raise ValueError(f"Duplicate outcome in {folder}/{a}")
            if r.error.fillna("").ne("").any() or not r.accuracy_test.between(0, 1).all():
                raise ValueError(f"Invalid outcome in {folder}/{a}")
            expected_keys = {(int(i), d, 0) for i in s.candidate_id for d in range(expected)} if kind == "synthetic" else {
                (int(i), 0, repeat) for i in s.candidate_id for repeat in range(expected)}
            actual = set(zip(r.candidate_id, r.draw, r.repeat))
            banks.append({**common, "algorithm": a, "sets": len(s), "random_sets": int(s.family.eq("random").sum()),
                          "runs": len(r), "expected_runs": len(s) * expected,
                          "complete": actual == expected_keys, "candidate_pool": meta["candidate_bank_size"]})
            sets.append(part)
            runs.append(r)
            members.append(m.assign(bank_id=task["id"], algorithm=a).drop_duplicates())
        predictions.append(p.assign(bank_id=task["id"]))
        coverage.append(c.assign(bank_id=task["id"]))
    return Collection(root, settings, pd.DataFrame(banks), pd.concat(sets, ignore_index=True),
                      pd.concat(runs, ignore_index=True), pd.concat(members, ignore_index=True),
                      pd.concat(predictions, ignore_index=True), pd.concat(coverage, ignore_index=True))


def inventory(data):
    return data.banks[["case", "model", "algorithm", "mean_degree", "rho", "budget", "n", "sets",
                       "random_sets", "runs", "expected_runs", "complete"]].reset_index(drop=True)


def coverage_table(frame):
    rows = []
    for (bank, a), g in frame.groupby(["bank_id", "algorithm"], sort=False):
        random = g[g.family == "random"]
        row = {"bank_id": bank, "algorithm": a, "panel": label(g.iloc[0]), "sets": len(g), "random": len(random)}
        for col, name in (("frozen_score", "frozen"), ("predicted_accuracy", "predicted")):
            span = float(g[col].max() - g[col].min())
            reference = float(random[col].max() - random[col].min())
            row.update({f"{name}_min": g[col].min(), f"{name}_max": g[col].max(),
                        f"{name}_random_min": random[col].min(), f"{name}_random_max": random[col].max(),
                        f"{name}_span_ratio": span / reference if reference > 1e-12 else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def score_diagnostics(frame):
    rows = []
    for (bank, a), g in frame.groupby(["bank_id", "algorithm"], sort=False):
        random = g[g.family.eq("random")]
        row = {"bank_id": bank, "algorithm": a, "panel": label(g.iloc[0]), "sets": len(g),
               "observed_mean": g.accuracy.mean(), "predicted_mean": g.predicted_accuracy.mean(),
               "prediction_mae": (g.predicted_accuracy - g.accuracy).abs().mean(),
               "prediction_bias": (g.predicted_accuracy - g.accuracy).mean(),
               "source_range_share": g.source_range_share.mean()}
        for col in ("frozen_score", "score_er", "score_cl", "score_paper", "predicted_accuracy"):
            row[f"rho_{col}"] = rho(g[col], g.accuracy)
            row[f"random_rho_{col}"] = rho(random[col], random.accuracy)
        rows.append(row)
    return pd.DataFrame(rows)


def feature_diagnostics(frame):
    rows = []
    for (bank, a), g in frame.groupby(["bank_id", "algorithm"], sort=False):
        for col in FEATURES:
            rows.append({"bank_id": bank, "algorithm": a, "panel": label(g.iloc[0]), "feature": col,
                         "rho": rho(g[col], g.accuracy), "sets": len(g),
                         "random_rho": rho(g.loc[g.family.eq("random"), col], g.loc[g.family.eq("random"), "accuracy"])})
    return pd.DataFrame(rows)


def policy_table(data, frame):
    """Feature/prediction-based picks within the common evaluated bank; no real-data fitting."""
    rows = []
    for (bank, a), g in frame.groupby(["bank_id", "algorithm"], sort=False):
        g = g.set_index("candidate_id", drop=False)
        baseline = float(g.loc[g.family.eq("random"), "accuracy"].mean())

        def add(policy, picked=None, fallback=False):
            if picked is None:
                accuracy, score, candidate, share = baseline, np.nan, np.nan, np.nan
            else:
                accuracy, score, candidate, share = picked.accuracy, picked.predicted_accuracy, picked.candidate_id, picked.source_range_share
            rows.append({"bank_id": bank, "algorithm": a, "panel": label(g.iloc[0]), "policy": policy,
                         "candidate_id": candidate, "accuracy": accuracy, "random_mean": baseline,
                         "gain": accuracy - baseline, "predicted_accuracy": score,
                         "source_range_share": share, "range_fallback": fallback})

        add("random_mean")
        m = data.memberships[(data.memberships.bank_id == bank) & (data.memberships.algorithm == a)]
        for rule in RULES:
            ids = m.loc[(m.family == "rule") & (m.detail == rule), "candidate_id"].unique()
            if len(ids) != 1:
                raise ValueError(f"Expected one membership for {bank}/{a}/{rule}")
            add(rule, g.loc[ids[0]])
        frozen = ["er", "cl", "paper"] if g.iloc[0]["kind"] == "real" else [SUFFIX[g.iloc[0].model]]
        for suffix in frozen:
            add(f"frozen_{suffix}" if g.iloc[0]["kind"] == "real" else "frozen_own",
                g.loc[g[f"score_{suffix}"].idxmax()])
        add("learned_mean", g.loc[g.predicted_accuracy.idxmax()])
        source = data.predictions[(data.predictions.bank_id == bank) & (data.predictions.algorithm == a)]
        if g.iloc[0]["kind"] == "synthetic":
            inside = source.loc[source.in_source_range, "candidate_id"]
            eligible = g.loc[inside] if len(inside) else g
            add("learned_guarded", eligible.loc[eligible.predicted_accuracy.idxmax()], not len(inside))
        add("oracle", g.loc[g.accuracy.idxmax()])
    return pd.DataFrame(rows)


def source_diagnostics(data, frame):
    combined = data.predictions.merge(frame[KEY + ["accuracy", "family"]], on=KEY, validate="many_to_one")
    baselines = frame[frame.family.eq("random")].groupby(["bank_id", "algorithm"]).accuracy.mean()
    rows = []
    panels = frame.drop_duplicates(["bank_id", "algorithm"]).set_index(["bank_id", "algorithm"])
    for (bank, a, source), g in combined.groupby(["bank_id", "algorithm", "source"], sort=False):
        inside = g[g.in_source_range]
        eligible = inside if len(inside) else g
        pick = eligible.loc[eligible.predicted.idxmax()]
        rows.append({"bank_id": bank, "algorithm": a, "panel": label(panels.loc[(bank, a), :]), "source": source,
                     "rho": rho(g.predicted, g.accuracy), "mae": (g.predicted - g.accuracy).abs().mean(),
                     "bias": (g.predicted - g.accuracy).mean(), "in_range_share": g.in_source_range.mean(),
                     "candidate_id": pick.candidate_id, "pick_accuracy": pick.accuracy,
                     "pick_gain": pick.accuracy - baselines.loc[bank, a], "range_fallback": not len(inside)})
    return pd.DataFrame(rows)


def contrast_table(data, frame):
    members = data.memberships[data.memberships.family.eq("contrast_member")]
    outcomes = frame.set_index(KEY).sort_index()
    rows = []
    for (bank, a, contrast), g in members.groupby(["bank_id", "algorithm", "contrast_id"], sort=False):
        low = g[g.detail.str.startswith("low_")].candidate_id.unique()
        high = g[g.detail.str.startswith("high_")].candidate_id.unique()
        if len(low) != 1 or len(high) != 1:
            raise ValueError(f"Ambiguous saved contrast: {bank}/{a}/{contrast}")
        lo, hi = outcomes.loc[bank, a, int(low[0])], outcomes.loc[bank, a, int(high[0])]
        feature = contrast.split(":")[-1]
        basis = "contrast_uniqueness" if feature == "uniqueness" else feature
        rows.append({"bank_id": bank, "algorithm": a, "panel": label(lo), "contrast_id": contrast,
                     "feature": feature, "uniqueness_cap": lo.contrast_uniqueness_cap,
                     "low_id": low[0], "high_id": high[0], "low_accuracy": lo.accuracy, "high_accuracy": hi.accuracy,
                     "feature_delta": hi[basis] - lo[basis], "algorithm_feature_delta": hi[feature] - lo[feature],
                     "degree_delta": hi.degree - lo.degree, "degree_z_gap": abs(hi.z_degree - lo.z_degree),
                     "accuracy_delta": hi.accuracy - lo.accuracy})
    return pd.DataFrame(rows)


def convergence_table(data, frame):
    rows = []
    for (bank, a), g in frame.groupby(["bank_id", "algorithm"], sort=False):
        if a == "percolation":  # Its flag means completed matching, not optimizer convergence.
            continue
        r = data.runs[(data.runs.bank_id == bank) & (data.runs.algorithm == a)]
        complete = g[g.converged_share == 1]
        rows.append({"bank_id": bank, "algorithm": a, "panel": label(g.iloc[0]), "runs": len(r),
                     "not_converged_runs": int((~r.converged).sum()), "not_converged_share": float((~r.converged).mean()),
                     "sets_all_runs_converged": len(complete), "sets_total": len(g),
                     "rho_frozen_all": rho(g.frozen_score, g.accuracy),
                     "rho_frozen_converged_subset": rho(complete.frozen_score, complete.accuracy),
                     "rho_learned_all": rho(g.predicted_accuracy, g.accuracy),
                     "rho_learned_converged_subset": rho(complete.predicted_accuracy, complete.accuracy)})
    return pd.DataFrame(rows)


def averaging_table(single, mean):
    before, after = score_diagnostics(single), score_diagnostics(mean)
    columns = ["bank_id", "algorithm", "rho_frozen_score", "random_rho_frozen_score",
               "rho_predicted_accuracy", "random_rho_predicted_accuracy"]
    return before[columns].merge(after[columns], on=["bank_id", "algorithm"], suffixes=("_single", "_mean"))


def overview(diagnostics, policies):
    rows = []
    for a, g in diagnostics.groupby("algorithm", sort=False):
        row = {"algorithm": a, "banks": len(g), "median_frozen_rho": g.rho_frozen_score.median(),
               "median_learned_rho": g.rho_predicted_accuracy.median(),
               "positive_frozen_rho": int(g.rho_frozen_score.gt(0).sum()),
               "positive_learned_rho": int(g.rho_predicted_accuracy.gt(0).sum()),
               "median_prediction_mae": g.prediction_mae.median(), "median_prediction_bias": g.prediction_bias.median()}
        for policy in ("frozen_er", "frozen_own", "learned_mean", "learned_guarded"):
            p = policies[(policies.algorithm == a) & (policies.policy == policy)]
            if len(p):
                row[f"{policy}_median_gain"] = p.gain.median()
                row[f"{policy}_wins_vs_random"] = int(p.gain.gt(.01).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("algorithm")


def export_tables(directory, **tables):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(directory / f"{name}.csv", index=False)


def save_figure(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    return fig


def scatter(ax, frame, x, calibration=False):
    for family in COLORS:
        g = frame[frame.family.eq(family)]
        ax.scatter(g[x], g.accuracy, c=COLORS[family], marker=MARKERS[family],
                   s=50 if family == "rule" else 24 if family in ("contrast", "endpoint") else 15,
                   alpha=.9 if family == "rule" else .65, linewidths=.3, edgecolors="white")
    if calibration:
        ax.plot([0, 1], [0, 1], color="#777777", ls="--", lw=.8, zorder=0)
        ax.set_xlim(-.025, 1.025)
    ax.set_ylim(-.035, 1.035)
    random = frame[frame.family.eq("random")]
    def fmt(v):
        return "NA" if not np.isfinite(v) else f"{v:.2f}"
    ax.text(.03, .97, f"ρ all={fmt(rho(frame[x], frame.accuracy))}\nρ random={fmt(rho(random[x], random.accuracy))}",
            va="top", transform=ax.transAxes, fontsize=8,
            bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": 1})
    ax.grid(alpha=.15)
    ax.spines[["top", "right"]].set_visible(False)


def family_legend(fig):
    handles = [Line2D([], [], ls="", marker=MARKERS[f], color=COLORS[f], label=FAMILY_LABELS[f],
                      markersize=7 if f == "rule" else 5) for f in COLORS]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9)


def plot_real_scores(frame, case):
    part = frame[frame.case.eq(case)]
    budgets = sorted(part.budget.unique())
    fig, axes = plt.subplots(len(budgets), 6, figsize=(19, 2.9 * len(budgets) + .8), squeeze=False, sharey=True)
    reference = part.frozen_reference.iloc[0].upper()
    for r, k in enumerate(budgets):
        for j, a in enumerate(ALGORITHMS):
            g = part[(part.algorithm == a) & (part.budget == k)]
            for c, x in enumerate(("frozen_score", "predicted_accuracy")):
                ax = axes[r, 2*j+c]
                scatter(ax, g, x, calibration=c == 1)
                ax.set_title(f"{ALG_LABELS[a]} · k={k}", fontsize=10)
                ax.set_xlabel(f"Frozen {reference} score" if c == 0 else "Mean synthetic prediction", fontsize=9)
        axes[r, 0].set_ylabel("Mean test accuracy")
    fig.suptitle(CASE_LABELS[case], fontsize=14)
    family_legend(fig)
    fig.tight_layout(rect=(0, .055, 1, .97))
    return fig


def plot_synthetic_scores(frame, model, algorithm):
    part = frame[(frame.model == model) & (frame.algorithm == algorithm)]
    groups = list(part.groupby("bank_id", sort=False))
    fig, axes = plt.subplots(len(groups), 2, figsize=(11, 2.9*len(groups)+.8), squeeze=False, sharey=True)
    for r, (_, g) in enumerate(groups):
        for c, x in enumerate(("frozen_score", "predicted_accuracy")):
            scatter(axes[r, c], g, x, calibration=c == 1)
            axes[r, c].set_title(label(g.iloc[0]), fontsize=10)
            axes[r, c].set_xlabel("Own frozen weighted score" if c == 0 else "Own saved selector prediction")
        axes[r, 0].set_ylabel("Mean accuracy over 3 Bs")
    fig.suptitle(f"{MODEL_LABELS[model]} · {ALG_LABELS[algorithm]}", fontsize=14)
    family_legend(fig)
    fig.tight_layout(rect=(0, .055, 1, .97))
    return fig


def heatmap(matrix, title, vmin=-1, vmax=1, annotate=True, cmap="RdBu", figsize=None):
    matrix = matrix.astype(float)
    fig, ax = plt.subplots(figsize=figsize or (max(7, .85*len(matrix.columns)+3), max(3.5, .34*len(matrix)+2)))
    palette = plt.get_cmap(cmap).copy()
    palette.set_bad("#ededed")
    image = ax.imshow(np.ma.masked_invalid(matrix.to_numpy()), aspect="auto", cmap=palette, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=8)
    if annotate:
        for r, row in enumerate(matrix.to_numpy()):
            for c, value in enumerate(row):
                text = "—" if not np.isfinite(value) else f"{value:.2f}" if vmax <= 1 else f"{value:.0f}"
                ax.text(c, r, text, ha="center", va="center", fontsize=7,
                        color="white" if np.isfinite(value) and abs(value) > .65*max(abs(vmin), abs(vmax)) else "#222222")
    ax.set_title(title, loc="left", pad=12)
    fig.colorbar(image, ax=ax, fraction=.035, pad=.03)
    fig.tight_layout()
    return fig


def plot_coverage(data, frame, algorithm):
    columns = {row.bank_id: f"{algorithm}__score_{row.frozen_reference}"
               for row in frame[frame.algorithm.eq(algorithm)].drop_duplicates("bank_id").itertuples()}
    c = data.coverage
    c = c[c.apply(lambda row: row.score == columns.get(row.bank_id), axis=1)].copy()
    panels = {row.bank_id: label(row._asdict()) for row in frame[frame.algorithm.eq(algorithm)].drop_duplicates("bank_id").itertuples()}
    c["panel"] = c.bank_id.map(panels)
    matrix = c.pivot(index="panel", columns="bin", values="selected").reindex(index=c.panel.drop_duplicates())
    matrix.columns = [f"Interval {int(v)+1}" for v in matrix.columns]
    return heatmap(matrix, f"{ALG_LABELS[algorithm]}: evaluated sets across frozen-score intervals",
                   vmin=0, vmax=max(1, float(matrix.max().max())), cmap="Blues")


def plot_rank_summary(diagnostics, algorithm):
    g = diagnostics[diagnostics.algorithm.eq(algorithm)].set_index("panel")
    cols = {"rho_score_er": "Frozen ER", "rho_score_cl": "Frozen CL", "rho_score_paper": "Frozen PAPER",
            "rho_predicted_accuracy": "Synthetic prediction"}
    return heatmap(g[list(cols)].rename(columns=cols), f"{ALG_LABELS[algorithm]}: Spearman correlation with accuracy")


def plot_feature_summary(features, algorithm):
    g = features[features.algorithm.eq(algorithm)]
    matrix = g.pivot(index="panel", columns="feature", values="rho").reindex(
        index=g.panel.drop_duplicates(), columns=FEATURES)
    return heatmap(matrix.rename(columns=FEATURES), f"{ALG_LABELS[algorithm]}: feature–accuracy Spearman correlations")


def plot_policy_summary(policies, algorithm):
    g = policies[(policies.algorithm == algorithm) & ~policies.policy.isin(["random_mean", "oracle"])]
    order = [k for k in POLICY_LABELS if k in set(g.policy)]
    matrix = g.pivot(index="panel", columns="policy", values="gain").reindex(
        index=g.panel.drop_duplicates(), columns=order)
    return heatmap(matrix.rename(columns=POLICY_LABELS), f"{ALG_LABELS[algorithm]}: accuracy gain over random mean")


def plot_sources(sources, algorithm, metric="rho"):
    g = sources[sources.algorithm.eq(algorithm)]
    g = g.assign(source_label=g.source.str.replace(f"/{algorithm}/", "/", regex=False))
    matrix = g.pivot(index="panel", columns="source_label", values=metric).reindex(index=g.panel.drop_duplicates())
    titles = {"rho": "source prediction–accuracy Spearman correlation", "pick_gain": "source pick: gain over random mean"}
    return heatmap(matrix, f"{ALG_LABELS[algorithm]}: {titles[metric]}")


def plot_feature_scatter(frame, case, budget, algorithm="faq"):
    g = frame[(frame.case == case) & (frame.budget == budget) & (frame.algorithm == algorithm)]
    columns = ["degree", "pair_mass", "uniqueness", "coverage_two", "coverage_three",
               "adjacency_signature_collision_fraction"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    for ax, col in zip(axes.flat, columns):
        scatter(ax, g, col)
        ax.set_xlabel(FEATURES[col])
        ax.set_ylabel("Mean test accuracy")
    fig.suptitle(f"{CASE_LABELS[case]} · k={budget} · {ALG_LABELS[algorithm]}")
    family_legend(fig)
    fig.tight_layout(rect=(0, .06, 1, .97))
    return fig


def plot_contrasts(contrasts, max_degree_gap=.25):
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), squeeze=False)
    for r, a in enumerate(ALGORITHMS):
        for c, feature in enumerate(("pair_mass", "coverage", "uniqueness")):
            ax = axes[r, c]
            g = contrasts[(contrasts.algorithm == a) & (contrasts.feature == feature) &
                          (contrasts.degree_z_gap <= max_degree_gap)]
            ax.scatter(g.feature_delta, g.accuracy_delta, s=22, alpha=.6, c="#2878b5")
            ax.axhline(0, color="#888888", lw=.8)
            ax.set_title(f"{ALG_LABELS[a]} · n={len(g)} contrasts", fontsize=10)
            ax.set_xlabel(f"High − low {FEATURES[feature].lower()}")
            ax.set_ylabel("Accuracy difference")
            ax.grid(alpha=.15)
    fig.suptitle(f"Saved contrasts with |cut-degree difference| ≤ {max_degree_gap:g} random-pool SD")
    fig.tight_layout(rect=(0, 0, 1, .97))
    return fig


def plot_averaging(single, mean, algorithm):
    one = single[(single.algorithm == algorithm) & single.family.eq("random")]
    avg = mean[(mean.algorithm == algorithm) & mean.family.eq("random")]
    models = list(MODEL_LABELS)
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), squeeze=False, sharey=True)
    for c, model in enumerate(models):
        banks = avg[avg.model == model].bank_id.unique()
        for r, bank in enumerate(banks):
            a, b = one[one.bank_id == bank], avg[avg.bank_id == bank]
            ax = axes[r, c]
            ax.scatter(a.frozen_score, a.accuracy, s=24, facecolors="none", edgecolors="#999999", label="One B")
            ax.scatter(b.frozen_score, b.accuracy, s=19, c="#2878b5", label="Mean over 3 Bs")
            ax.set_title(label(b.iloc[0]), fontsize=9)
            before, after = rho(a.frozen_score, a.accuracy), rho(b.frozen_score, b.accuracy)
            ax.text(.03, .97, f"ρ one B={before:.2f}\nρ three Bs={after:.2f}", va="top", transform=ax.transAxes, fontsize=8)
            ax.set_ylim(-.035, 1.035)
            ax.set_xlabel("Own frozen weighted score")
            if c == 0:
                ax.set_ylabel("Random-set accuracy")
            ax.grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"{ALG_LABELS[algorithm]}: the same random sets, one B versus three Bs")
    fig.tight_layout(rect=(0, .04, 1, .97))
    return fig
