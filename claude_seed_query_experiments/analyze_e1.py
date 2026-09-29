"""Analysis of E1 (static rules at equal budget)."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))   # this folder
REPO = _os.path.dirname(HERE)                          # repository root
import json, sys
import numpy as np, pandas as pd

OUT = HERE + "/results"
TAG = sys.argv[1] if len(sys.argv) > 1 else "iter100"
df = pd.read_csv(f"{OUT}/e1_runs_{TAG}.csv")
df["succ"] = df.acc >= 0.9
df["succ30"] = df.acc30 >= 0.9
MODELS = ["sparse_er", "sparse_sbm", "paper", "ier", "dense_sbm"]
ORDER = ["random", "degree", "betweenness", "reach3", "cov1c2", "cov2c3", "genrank", "strat_degree", "dispersion"]

# per (pair, k, rule): average over restarts and random draws
cell = df.groupby(["model", "pair", "k", "rule"]).agg(acc=("acc", "mean"), succ=("succ", "mean"),
                                                        acc30=("acc30", "mean"), succ30=("succ30", "mean"),
                                                        n_runs=("acc", "size")).reset_index()


def table(metric):
    t = cell.groupby(["model", "k", "rule"])[metric].mean().unstack("rule")[ORDER]
    return t


def paired_diff(metric, base):
    """Mean over (pair, k) of rule - base, with a pair-level bootstrap CI."""
    w = cell.pivot_table(index=["model", "pair", "k"], columns="rule", values=metric)
    rows = []
    rng = np.random.default_rng(0)
    for m in MODELS:
        if m not in w.index.get_level_values(0):
            continue
        wm = w.loc[m]
        pairs = wm.index.get_level_values(0).unique()
        for rule in ORDER:
            if rule == base:
                continue
            d = (wm[rule] - wm[base])
            per_pair = d.groupby(level=0).mean()
            boots = [per_pair.loc[rng.choice(pairs, len(pairs))].mean() for _ in range(2000)]
            rows.append(dict(model=m, rule=rule, diff=d.mean(), lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    for metric in ["succ", "acc", "succ30"]:
        print(f"\n===== {metric} by model x budget x rule =====")
        print(table(metric).round(2).to_string())
    for base in ["random", "degree"]:
        print(f"\n===== success-probability difference vs {base} (mean over pairs x budgets; 95% pair-bootstrap CI) =====")
        d = paired_diff("succ", base)
        print(d.pivot(index="rule", columns="model", values="diff").round(2).to_string())
        d["ci"] = d.apply(lambda r: f"[{r.lo:+.2f},{r.hi:+.2f}]", axis=1)
        print(d.pivot(index="rule", columns="model", values="ci").to_string())
    print("\n===== accuracy difference vs degree =====")
    d = paired_diff("acc", "degree")
    print(d.pivot(index="rule", columns="model", values="diff").round(3).to_string())
    # queries needed: smallest k (per pair) with success >= 0.5 (interpolated), averaged over pairs
    print("\n===== budget needed for success probability >= 0.5 (linear interp. over k, mean over pairs; inf -> not reached) =====")
    recs = []
    for (m, p, rule), g in cell.groupby(["model", "pair", "rule"]):
        g = g.sort_values("k")
        ks, ys = g.k.to_numpy(float), g.succ.to_numpy(float)
        kneed = np.inf
        for i in range(len(ks)):
            if ys[i] >= 0.5:
                if i == 0:
                    kneed = ks[0]
                else:
                    kneed = ks[i - 1] + (0.5 - ys[i - 1]) * (ks[i] - ks[i - 1]) / max(ys[i] - ys[i - 1], 1e-9)
                break
        recs.append(dict(model=m, pair=p, rule=rule, kneed=kneed))
    kn = pd.DataFrame(recs)
    kn["reached"] = np.isfinite(kn.kneed)
    summ = kn.groupby(["model", "rule"]).agg(median_k=("kneed", "median"), reached=("reached", "mean")).unstack("model")
    print(summ.round(1).to_string())
    # set features of the chosen sets
    print("\n===== mean Graph-A features of the chosen seed sets (all budgets) =====")
    print(df.groupby(["model", "rule"])[["f_deg", "f_three_hop", "f_cov2", "f_2hop3", "f_pairdist"]].mean().round(2).unstack("model").to_string())
    print("\n===== iteration effect: success at 30 vs 100 iterations, by rule (all models) =====")
    print(cell.groupby(["model", "rule"])[["succ30", "succ"]].mean().round(2).unstack("model").to_string())
