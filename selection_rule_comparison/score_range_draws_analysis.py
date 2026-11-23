"""Read-only analysis of the synthetic score-range banks over every collected draw of B (up to 11).

score_range_11_draws.ipynb calls these functions. Draws 0-2 come from the score-range
collection (collect_score_ranges.py) and draws 3-10 from score_range_draws.py. Within a
bank, sets are averaged only over draws that every set in the bank has finished, so the
means in one panel are comparable. No collection, matching or fitting imports.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from . import score_range_analysis as A

SOURCE = A.DEFAULT_ROOT
EXTRA = A.REPO_ROOT / "selection_rule_comparison/data/score_range_draws"
EXPORT = EXTRA / "analysis"
OUTCOMES = {"accuracy_test": "test accuracy", "success": "success rate"}
X_LABELS = {"frozen_score": "Own frozen weighted score (z units)",
            "z_degree": "Cut degree (z-score against the random pool)",
            "predicted_accuracy": "Own saved selector prediction"}
X_NAMES = {"frozen_score": "Frozen weighted score", "z_degree": "Degree alone",
           "predicted_accuracy": "Saved selector prediction"}
MODEL_COLORS = {"sparse_er": "#2a78d6", "chung_lu": "#eb6834", "paper_pa": "#1baf7a"}  # landscape slots 1-3
ONE_DRAW = "#c4c4c4"    # a single draw of B, behind the means
INK, INK2 = "#222222", "#555555"
KEY = ["bank_id", "algorithm", "candidate_id"]
RUN_COLUMNS = ["bank_id", "algorithm", "candidate_id", "draw", "repeat", "accuracy_test", "success", "error"]
BINS = 8


def _complete_rows(path):
    """Rows with every field present; never rewrites a file a collector may still be appending to."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            return pd.DataFrame(columns=RUN_COLUMNS)
        rows = [row for row in reader if len(row) == len(header)]
    return pd.DataFrame(rows, columns=header)


def _typed(frame):
    frame = frame.copy()
    for column in ("candidate_id", "draw", "repeat"):
        frame[column] = pd.to_numeric(frame[column]).astype(int)
    for column in ("accuracy_test", "success"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["error"] = frame["error"].fillna("").astype(str).replace("nan", "")
    return frame


def load(source=SOURCE, extra=EXTRA):
    """The score-range collection (draws 0-2, must be complete) plus any extra draws collected so far."""

    data = A.load_collection(source, kind="synthetic")
    pieces = [_typed(data.runs[RUN_COLUMNS])]
    extra = Path(extra)
    files = 0
    for task in data.settings["banks"]:
        if task["kind"] != "synthetic":
            continue
        for path in sorted((extra / task["id"]).glob(f"{task['algorithm']}_draw_*_runs.csv")):
            frame = _complete_rows(path)
            if len(frame):
                files += 1
                pieces.append(_typed(frame.assign(bank_id=task["id"])[RUN_COLUMNS]))
    runs = pd.concat(pieces, ignore_index=True)
    if runs.duplicated(KEY + ["draw", "repeat"]).any():
        raise ValueError("Duplicate outcomes for one set and draw. Do the extra draws overlap draws 0-2?")
    if runs.error.ne("").any() or not runs.accuracy_test.between(0, 1).all():
        raise ValueError("Invalid or failed outcomes in the run files.")
    data.extra_files = files
    return data, runs


def complete_draws(data, runs):
    """Per bank and algorithm: draws that every set in the bank has finished, in draw order."""

    sets = data.sets.groupby(["bank_id", "algorithm"]).candidate_id.nunique()
    counts = runs[runs.repeat.eq(0)].groupby(["bank_id", "algorithm", "draw"]).candidate_id.nunique()
    out = {}
    for (bank, a), n in sets.items():
        part = counts.loc[(bank, a)] if (bank, a) in counts.index.droplevel("draw") else pd.Series(dtype=int)
        out[(bank, a)] = sorted(int(d) for d, c in part.items() if c == n)
    return out


def inventory(data, runs):
    draws = complete_draws(data, runs)
    rows = []
    for (bank, a), used in draws.items():
        info = data.sets[(data.sets.bank_id == bank) & (data.sets.algorithm == a)].iloc[0]
        rows.append({"panel": A.label(info), "algorithm": a, "sets": int(((data.sets.bank_id == bank) &
                     (data.sets.algorithm == a)).sum()), "complete_draws": len(used),
                     "draws": ",".join(map(str, used)),
                     "extra_runs": int(((runs.bank_id == bank) & (runs.algorithm == a) & (runs.draw >= 3)).sum())})
    return pd.DataFrame(rows)


def per_set(data, runs, outcome="accuracy_test", draws=None):
    """One row per set: its outcome on draw 0, the mean over draws 0-2, and the mean over the bank's
    complete draws (or over the first `draws` of them)."""

    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}")
    used = complete_draws(data, runs)
    frames = []
    for (bank, a), available in used.items():
        if len(available) < 3 or available[:3] != [0, 1, 2]:
            raise ValueError(f"{bank}/{a}: draws 0-2 are not complete in the score-range collection.")
        chosen = available if draws is None else available[:max(1, draws)]
        r = runs[(runs.bank_id == bank) & (runs.algorithm == a) & runs.repeat.eq(0) & runs.draw.isin(available)]
        wide = r.pivot(index="candidate_id", columns="draw", values=outcome)[available]
        frames.append(pd.DataFrame({"bank_id": bank, "algorithm": a, "candidate_id": wide.index,
                                    "one": wide[0].to_numpy(), "three": wide[[0, 1, 2]].mean(axis=1).to_numpy(),
                                    "mean": wide[chosen].mean(axis=1).to_numpy(),
                                    "sd": wide[chosen].std(axis=1, ddof=1).to_numpy(), "n_draws": len(chosen)}))
    outcomes = pd.concat(frames, ignore_index=True)
    return data.sets.merge(outcomes, on=KEY, how="inner", validate="one_to_one").assign(outcome=outcome)


def _style(ax):
    ax.grid(alpha=.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)


def _fmt(value):
    return "NA" if not np.isfinite(value) else f"{value:.2f}"


def _binned(x, y, bins=BINS):
    """Mean of y within equal-count bins of x: the shape of the trend, not a fitted model."""

    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 2 * bins:
        return frame.iloc[:0]
    frame["bin"] = pd.qcut(frame.x.rank(method="first"), bins, labels=False)
    return frame.groupby("bin")[["x", "y"]].mean()


def _cells(part, algorithm):
    return part[part.algorithm.eq(algorithm)].drop_duplicates("bank_id").sort_values(["mean_degree", "rho"])


def plot_model(frame, model, x="frozen_score"):
    """One figure per graph model: rows are matchers, columns their three cells.
    Coloured marks: each set's mean over draws of B. Open grey circles: the same set on draw 0 alone.
    Black line: mean within eight equal-count bins of x."""

    part = frame[frame.model.eq(model)]
    outcome = OUTCOMES[part.outcome.iloc[0]]
    n_max = int(part.n_draws.max())
    with plt.rc_context({"font.size": 9}):
        fig, axes = plt.subplots(3, 3, figsize=(13.6, 12.8), squeeze=False, sharey=True)
        for r, a in enumerate(A.ALGORITHMS):
            cells = _cells(part, a)
            for c in range(3):
                ax = axes[r, c]
                if c >= len(cells):
                    ax.axis("off")
                    continue
                g = part[part.bank_id.eq(cells.iloc[c].bank_id)]
                n = int(g.n_draws.iloc[0])
                _style(ax)
                ax.scatter(g[x], g.one, s=16, facecolors="none", edgecolors=ONE_DRAW, linewidths=.7, zorder=1)
                for family in A.COLORS:
                    f = g[g.family.eq(family)]
                    ax.scatter(f[x], f["mean"], c=A.COLORS[family], marker=A.MARKERS[family],
                               s=60 if family == "rule" else 26 if family in ("contrast", "endpoint") else 18,
                               alpha=.95 if family == "rule" else .75, linewidths=.4, edgecolors="white", zorder=3)
                line = _binned(g[x], g["mean"])
                ax.plot(line.x, line.y, color=INK, lw=1.6, marker="o", ms=3.5, zorder=4)
                random = g[g.family.eq("random")]
                counts = [(1, g.one)] + ([(3, g.three)] if n > 3 else []) + [(n, g["mean"])]
                text = " · ".join(f"{m} draw{'s' if m > 1 else ''} {_fmt(A.rho(g[x], y))}" for m, y in counts)
                ax.text(0, 1.015, f"Spearman, all {len(g)} sets: {text}\n{len(random)} random sets, {n} draws: "
                        f"{_fmt(A.rho(random[x], random['mean']))}", va="bottom", transform=ax.transAxes,
                        fontsize=7.8, color=INK2)
                info = g.iloc[0]
                ax.set_title(f"{A.ALG_LABELS[a]} · mean degree {info.mean_degree:g}, ρ {info.rho:g}, "
                             f"k = {int(info.budget)}\n\n", loc="left", fontsize=9.5, color=INK)
                ax.set_ylim(-.035, 1.035)
                if c == 0:
                    ax.set_ylabel(f"{outcome.capitalize()}, mean over draws of B")
                if r == 2:
                    ax.set_xlabel(X_LABELS[x])
        handles = [Line2D([], [], ls="", marker=A.MARKERS[f], color=A.COLORS[f], label=A.FAMILY_LABELS[f],
                          markersize=8 if f == "rule" else 5.5) for f in A.COLORS]
        handles += [Line2D([], [], ls="", marker="o", markerfacecolor="none", markeredgecolor=ONE_DRAW,
                           label="Same set, draw 0 only"),
                    Line2D([], [], color=INK, lw=1.6, marker="o", ms=3.5, label=f"Mean in {BINS} equal-count bins")]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
        fig.suptitle(f"{A.MODEL_LABELS[model]}: {X_NAMES[x].lower()} vs {outcome}, "
                     f"each set averaged over up to {n_max} draws of B", x=.01, ha="left", fontsize=12.5, color=INK)
        fig.tight_layout(rect=(0, .065, 1, .97))
    return fig


def rho_by_draws(data, runs, outcome="accuracy_test", x="frozen_score"):
    """Spearman of x with the mean over the first m complete draws, m = 1, 2, ..., per bank."""

    used = complete_draws(data, runs)
    rows = []
    for (bank, a), available in used.items():
        sets = data.sets[(data.sets.bank_id == bank) & (data.sets.algorithm == a)].set_index("candidate_id")
        r = runs[(runs.bank_id == bank) & (runs.algorithm == a) & runs.repeat.eq(0) & runs.draw.isin(available)]
        wide = r.pivot(index="candidate_id", columns="draw", values=outcome).reindex(sets.index)[available]
        random = sets.family.eq("random")
        info = sets.iloc[0]
        for m in range(1, len(available) + 1):
            mean = wide.iloc[:, :m].mean(axis=1)
            rows.append({"bank_id": bank, "algorithm": a, "model": info.model, "panel": A.label(info),
                         "mean_degree": info.mean_degree, "rho": info.rho,
                         "draws": m, "rho_all": A.rho(sets[x], mean),
                         "rho_random": A.rho(sets.loc[random, x], mean[random])})
    return pd.DataFrame(rows)


def plot_rho_by_draws(table, x="frozen_score"):
    """One panel per matcher; one line per cell, coloured by graph model (legend plus a table view)."""

    styles = ("-", "--", ":")
    with plt.rc_context({"font.size": 9}):
        fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4), sharey=True)
        for ax, a in zip(axes, A.ALGORITHMS):
            _style(ax)
            part = table[table.algorithm.eq(a)]
            for model, cells in part.groupby("model", sort=False):
                order = cells.drop_duplicates("bank_id").sort_values(["mean_degree", "rho"]).bank_id
                for style, bank in zip(styles, order):
                    g = cells[cells.bank_id.eq(bank)]
                    ax.plot(g.draws, g.rho_all, color=MODEL_COLORS[model], ls=style, lw=2, marker="o", ms=3)
            ax.axhline(0, color="#999999", lw=.8, zorder=0)
            ax.set_title(A.ALG_LABELS[a], loc="left", fontsize=10, color=INK)
            ax.set_xlabel("Draws of B averaged")
            ax.set_xticks(range(1, int(table.draws.max()) + 1))
        axes[0].set_ylabel(f"Spearman: {X_NAMES[x].lower()} vs mean outcome")
        axes[0].set_ylim(min(-.2, table.rho_all.min() - .05), 1)
        handles = [Line2D([], [], color=c, lw=2, label=A.MODEL_LABELS[m]) for m, c in MODEL_COLORS.items()]
        handles += [Line2D([], [], color=INK2, lw=1.6, ls=s, label=f"{t} column of the scatter figures")
                    for s, t in zip(styles, ("Left", "Middle", "Right"))]
        fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9)
        fig.suptitle(f"All sets in each bank: does averaging over more draws of B sharpen the ranking?",
                     x=.01, ha="left", fontsize=12, color=INK)
        fig.tight_layout(rect=(0, .08, 1, .94))
    return fig


def summary(data, frame):
    """Per cell: Spearman of each A-only score with the mean outcome over draws, and what the top of
    the score range buys relative to the uniform random controls (gains in outcome units)."""

    rows = []
    for (bank, a), g in frame.groupby(["bank_id", "algorithm"], sort=False):
        g = g.set_index("candidate_id", drop=False)
        random = g[g.family.eq("random")]
        baseline = random["mean"].mean()
        top = g.nlargest(max(1, int(round(.1 * len(g)))), "frozen_score")
        m = data.memberships[(data.memberships.bank_id == bank) & (data.memberships.algorithm == a)]
        rules = {}
        for rule in A.RULES:
            ids = m.loc[m.family.eq("rule") & m.detail.eq(rule), "candidate_id"].unique()
            if len(ids) == 1 and ids[0] in g.index:
                rules[rule] = g.loc[ids[0], "mean"]
        best = max(rules, key=rules.get) if rules else None
        info = g.iloc[0]
        rows.append({"panel": A.label(info), "algorithm": a, "model": info.model, "sets": len(g),
                     "draws": int(info.n_draws),
                     "rho_frozen_1": A.rho(g.frozen_score, g.one), "rho_frozen_3": A.rho(g.frozen_score, g.three),
                     "rho_frozen_all": A.rho(g.frozen_score, g["mean"]),
                     "rho_frozen_random_all": A.rho(random.frozen_score, random["mean"]),
                     "rho_degree_all": A.rho(g.z_degree, g["mean"]),
                     "rho_selector_all": A.rho(g.predicted_accuracy, g["mean"]),
                     "random_mean": baseline, "top_decile_gain": top["mean"].mean() - baseline,
                     "highest_score_gain": g.loc[g.frozen_score.idxmax(), "mean"] - baseline,
                     "best_rule": A.RULE_LABELS.get(best, best),
                     "best_rule_gain": rules[best] - baseline if best else np.nan})
    return pd.DataFrame(rows)


def save(fig, name, directory=None):
    """PNG and PDF under <directory>/figures (default: data/score_range_draws/analysis)."""

    return A.save_figure(fig, Path(directory or EXPORT) / "figures", name)
