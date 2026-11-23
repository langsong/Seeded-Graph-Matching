"""Figures for the report (static PNG, light surface)."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))   # this folder
REPO = _os.path.dirname(HERE)                          # repository root
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

OUT = HERE + "/results"
FIG = HERE + "/figures"
os.makedirs(FIG, exist_ok=True)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BASE = "#8f8e89"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
LBL = {"sparse_er": "Sparse ER", "sparse_sbm": "Sparse SBM", "paper": "PAPER (heavy-tailed)",
       "ier": "Inhomogeneous ER", "dense_sbm": "Dense SBM (rho=.36)"}


def line(ax, x, y, color, label, dashed=False):
    ax.plot(x, y, color=color, lw=1.8, ls="--" if dashed else "-", solid_capstyle="round", label=label,
            marker="o", ms=5, mec=SURFACE, mew=1.2)


def fig_e1():
    df = pd.read_csv(f"{OUT}/e1_runs_iter100.csv")
    df["succ"] = df.acc >= 0.9
    cell = df.groupby(["model", "pair", "k", "rule"]).succ.mean().reset_index()
    t = cell.groupby(["model", "k", "rule"]).succ.mean().unstack("rule")
    rules = [("random", BASE, True), ("degree", SLOTS[0], False), ("betweenness", SLOTS[1], False),
             ("cov1c2", SLOTS[2], False), ("dispersion", SLOTS[3], False)]
    names = {"random": "random", "degree": "top degree", "betweenness": "top betweenness",
             "cov1c2": "greedy 2-witness coverage", "dispersion": "spread out (k-center)"}
    models = ["sparse_er", "sparse_sbm", "ier", "paper", "dense_sbm"]
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.1), sharey=True)
    for ax, m in zip(axes, models):
        tm = t.loc[m]
        for r, c, d in rules:
            line(ax, tm.index, tm[r], c, names[r], dashed=d)
        ax.set_title(LBL[m], loc="left")
        ax.set_xlabel("queries k (seeds)")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(tm.index)
    axes[0].set_ylabel("P(accuracy >= 0.9)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig1_static_rules.png", dpi=160, bbox_inches="tight")


def fig_e0():
    runs = pd.read_csv(f"{OUT}/e0_runs.csv")
    models = ["sparse_er", "sparse_sbm", "paper", "ier"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
    ax = axes[0]
    for i, m in enumerate(models):
        d = runs[runs.model == m]
        ys = [np.mean(d[c] >= 0.9) for c in ["acc30", "acc60", "acc100", "acc_final"]]
        line(ax, [30, 60, 100, 150], ys, SLOTS[i], LBL[m])
    ax.set_xticks([30, 60, 100, 150])
    ax.set_xlabel("iteration cap (max_iter)")
    ax.set_ylabel("P(accuracy >= 0.9)")
    ax.set_title("Same seed sets, longer runs", loc="left")
    ax.set_ylim(0.3, 0.85)
    ax = axes[1]
    T = list(range(1, 31))
    for i, m in enumerate(models):
        d = runs[runs.model == m]
        trajs = [json.loads(x) for x in d.traj]
        y = (d.acc_final >= 0.9).to_numpy()   # success when run to convergence (<=150 iterations)
        aucs = [roc_auc_score(y, [tr[t - 1] if len(tr) >= t else tr[-1] for tr in trajs]) for t in T]
        ax.plot(T, aucs, color=SLOTS[i], lw=1.8, label=LBL[m])
    ax.axhline(0.5, color=GRID, lw=1)
    ax.set_xlabel("Frank-Wolfe iteration t")
    ax.set_ylabel("AUC for eventual success")
    ax.set_title("Accuracy of the assignment direction at iteration t\npredicts eventual success early", loc="left")
    ax.set_ylim(0.78, 1.005)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig0_protocol.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    which = sys.argv[1:] or ["e0", "e1"]
    if "e0" in which:
        fig_e0()
    if "e1" in which:
        fig_e1()


def fig_mechanism():
    import sgmlib as L
    V = pd.read_csv(f"{OUT}/e0_vertex.csv.gz")
    models = ["sparse_er", "sparse_sbm", "paper", "ier"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
    ax = axes[0]
    for i, m in enumerate(models):
        d = V[(V.model == m) & (~V.success)]
        t = d.groupby(d.w1.clip(upper=3)).correct.mean()
        line(ax, t.index, t.values, SLOTS[i], LBL[m])
    ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["0", "1", "2", "3+"])
    ax.set_xlabel("seed neighbours of the vertex")
    ax.set_ylabel("P(vertex matched correctly)")
    ax.set_title("Inside failed runs, only multiply-witnessed\nvertices get matched", loc="left")
    ax.set_ylim(0, 1.02)
    ax = axes[1]
    df = pd.read_csv(f"{OUT}/e1_runs_iter100.csv")
    df = df[df.model.isin(models)]
    nnz = {}
    for key in df.pair.unique():
        nnz[key] = L.load_npz_pair(f"{OUT}/pairs/{key}.npz", key).A.nnz
    df["pres"] = df.score / df.pair.map(nnz)
    for i, m in enumerate(models):
        d = df[df.model == m]
        ax.scatter(d.pres, d.acc, s=7, color=SLOTS[i], alpha=0.35, linewidths=0, label=LBL[m])
    ax.set_xlabel("share of Graph-A edges preserved by the SGM output (label-free)")
    ax.set_ylabel("true accuracy on non-seeds")
    ax.set_title("Success is visible without ground truth", loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig2_mechanism.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__" and "mech" in sys.argv[1:]:
    fig_mechanism()
