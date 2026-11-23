import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))   # this folder
REPO = _os.path.dirname(HERE)                          # repository root
import json, sys
import numpy as np, pandas as pd

OUT = HERE + "/results"
pd.set_option("display.width", 220)


def e2():
    df = pd.read_csv(f"{OUT}/e2_runs.csv")
    df["cond"] = np.where(df.k0 > 0, "with S0 (0.4k verified + 0.4k queries)", "no S0 (0.8k queries)")
    strat_order = ["random", "degree", "reach3", "cov2c3", "ad_consistency", "ad_disagreement", "ad_phase", "ad_cov_uncertain"]
    # average the two random draws
    agg = df.groupby(["cond", "model", "pair", "s0_draw", "strategy"])[["p_success", "mean_acc", "best_obj_acc"]].mean().reset_index()
    for metric in ["p_success", "mean_acc"]:
        print(f"\n===== E2 {metric}: mean over pairs and S0 draws =====")
        t = agg.groupby(["cond", "model", "strategy"])[metric].mean().unstack("strategy").reindex(columns=strat_order)
        print(t.round(2).to_string())
        print(f"\n----- overall (all models) -----")
        print(agg.groupby(["cond", "strategy"])[metric].mean().unstack("strategy").reindex(columns=strat_order).round(3).to_string())
    # paired differences vs degree and vs cov2c3 with bootstrap over configurations
    w = agg.pivot_table(index=["cond", "model", "pair", "s0_draw"], columns="strategy", values="p_success")
    rng = np.random.default_rng(0)
    print("\n===== paired difference in success probability (mean [95% bootstrap CI over configurations]) =====")
    for cond in w.index.get_level_values(0).unique():
        wc = w.loc[cond]
        for base in ["degree", "cov2c3"]:
            line = f"{cond[:12]:12s} vs {base:7s}: "
            for s in strat_order:
                if s == base:
                    continue
                d = (wc[s] - wc[base]).to_numpy()
                b = [rng.choice(d, len(d)).mean() for _ in range(2000)]
                line += f"{s}={d.mean():+.2f}[{np.percentile(b,2.5):+.2f},{np.percentile(b,97.5):+.2f}] "
            print(line)
    # state trace: how often phase strategy saw an ignited state
    tr = df[df.strategy.str.startswith("ad_")].copy()
    ign = []
    for t in tr.trace:
        for rd in json.loads(t):
            if rd.get("state_full_agree") is not None:
                ign.append(rd["state_full_agree"])
    if ign:
        print(f"\nshare of adaptive rounds that saw >=50% of vertices with full restart agreement: {np.mean(np.array(ign) >= 0.5):.2f}")


def e4():
    df = pd.read_csv(f"{OUT}/e4_runs.csv")
    order = ["random", "degree", "betweenness", "reach3", "cov1c2", "cov2c3", "genrank", "dispersion", "twins", "hybrid_twins"]
    cell = df.groupby(["pair", "k", "rule"]).agg(acc=("acc", "mean"), acc30=("acc30", "mean"),
                                                  succ=("acc", lambda x: np.mean(x >= 0.9))).reset_index()
    for metric in ["acc", "acc30"]:
        print(f"\n===== E4 Enron: mean non-seed {metric} by pair x budget x rule =====")
        print(cell.groupby(["pair", "k", "rule"])[metric].mean().unstack("rule").reindex(columns=order).round(2).to_string())
    # random spread across draws: how much do random sets differ at each k
    rd = df[df.rule == "random"].groupby(["pair", "k", "draw"]).acc.mean().groupby(["pair", "k"]).agg(["min", "max", "std"])
    print("\nrandom seed sets: spread of mean accuracy across 8 draws")
    print(rd.round(2).to_string())


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("e2", "both"):
        e2()
    if which in ("e4", "both"):
        e4()
