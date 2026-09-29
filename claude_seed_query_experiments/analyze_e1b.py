import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))   # this folder
REPO = _os.path.dirname(HERE)                          # repository root
import numpy as np, pandas as pd
df = pd.read_csv(HERE + "/results/e1b_runs.csv")
df["succ"] = df.acc >= 0.9
cell = df.groupby(["model", "pair", "k", "rule"]).agg(succ=("succ", "mean"), acc=("acc", "mean")).reset_index()
ORDER = ["random", "degree", "betweenness", "reach3", "cov1c2", "cov2c3", "strat_degree"]
print("E1b: success probability at the focused budget (12 new pairs x 2 restarts; random = 2 draws)")
print(cell.groupby(["model", "k", "rule"]).succ.mean().unstack("rule").reindex(columns=ORDER).round(2).to_string())
print("\nmean accuracy:")
print(cell.groupby(["model", "k", "rule"]).acc.mean().unstack("rule").reindex(columns=ORDER).round(2).to_string())
w = cell.pivot_table(index=["model", "pair"], columns="rule", values="succ")
rng = np.random.default_rng(0)
print("\npaired difference vs top degree in success probability, mean [95% bootstrap CI over pairs]")
for m in w.index.get_level_values(0).unique():
    wm = w.loc[m]
    line = f"  {m:10s}"
    for r in ORDER:
        if r == "degree": continue
        d = (wm[r] - wm["degree"]).to_numpy()
        b = [rng.choice(d, len(d)).mean() for _ in range(4000)]
        line += f"  {r}={d.mean():+.2f}[{np.percentile(b,2.5):+.2f},{np.percentile(b,97.5):+.2f}]"
    print(line)
# pooled over the four regimes
print("\npooled over regimes (48 pairs):")
line = ""
for r in ORDER:
    if r == "degree": continue
    d = (w[r] - w["degree"]).to_numpy()
    b = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    line += f"  {r}={d.mean():+.2f}[{np.percentile(b,2.5):+.2f},{np.percentile(b,97.5):+.2f}]"
print(line)
