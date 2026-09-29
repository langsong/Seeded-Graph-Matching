"""Analysis of E3: one-seed marginal value with restarts."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))   # this folder
REPO = _os.path.dirname(HERE)                          # repository root
import json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import sgmlib as L
import e3_one_seed as E

OUT = HERE + "/results"
SRC = E.SRC
runs = pd.read_csv(f"{OUT}/e3_runs.csv")
runs["succ"] = runs.acc >= 0.9
runs["succ30"] = runs.acc30 >= 0.9


def spear(a, b):
    a, b = pd.Series(a), pd.Series(b)
    m = a.notna() & b.notna()
    return float(np.corrcoef(a[m].rank(), b[m].rank())[0, 1])


rows_all = []
for name in E.PROBLEMS:
    pair, S0, out = E.load_problem(name)
    feats = pd.read_csv(f"{SRC}/{name}_candidate_features.csv").set_index("candidate_vertex_graph1")
    G = L.gstats(pair.A)
    r = runs[runs.problem == name]
    base = r[r.candidate == -1]
    p0, p0_30, a0 = base.succ.mean(), base.succ30.mean(), base.acc.mean()
    # label-free uncertainty from baseline restarts
    perms = [np.array(json.loads(p)) for p in base.perm]
    best = perms[int(np.argmax(base.score.to_numpy()))]
    Bp = pair.B[best][:, best]
    preserved = np.asarray(pair.A.multiply(Bp).sum(axis=1)).ravel()
    cons = np.where(G.deg > 0, preserved / np.maximum(G.deg, 1), 0.0)
    agree = np.mean([p == best for p in perms], axis=0)
    base_correct = (best == pair.truth).astype(float)
    c = r[r.candidate >= 0].groupby("candidate").agg(p=("succ", "mean"), p30=("succ30", "mean"), acc=("acc", "mean")).reset_index()
    c["dp"] = c.p - p0
    c["dacc"] = c.acc - a0
    v = c.candidate.to_numpy()
    c["degree"] = G.deg[v]
    c["reach3"] = G.reach(3)[v]
    M1 = G.ball(1)
    cnt = M1[:, S0].sum(axis=1)
    c["new_2witness"] = [int(((cnt == 1) & M1[:, u]).sum()) for u in v]        # vertices that gain a 2nd seed neighbour
    c["consistency"] = cons[v]
    c["agreement"] = agree[v]
    c["baseline_correct"] = base_correct[v]
    c["dist_to_seeds"] = G.dist[np.ix_(v, S0)].min(axis=1)
    for col in ["candidate_gradient_row_top2_gap_z", "candidate_gradient_row_normalized_entropy",
                "candidate_new_second_witness_fraction", "candidate_signature_collision_reduction"]:
        c[col] = feats.loc[v, col].to_numpy()
    old = out.set_index("candidate_vertex_graph1").loc[v]
    c["old_single_run_success"] = old.outcome_success.astype(float).to_numpy()
    c["problem"] = name
    rows_all.append(c)
    print(f"\n===== {name}: n={pair.n}, |S0|={len(S0)} =====")
    print(f"baseline success prob: {p0_30:.2f} @30 iters, {p0:.2f} @100 iters (16 restarts); baseline mean acc @100 = {a0:.2f}")
    print(f"candidates evaluated: {len(c)}; mean p_v = {c.p.mean():.2f} (@30: {c.p30.mean():.2f})")
    print(f"share of candidates with p_v >= 0.8: {np.mean(c.p >= 0.8):.2f}; with p_v <= 0.2: {np.mean(c.p <= 0.2):.2f}")
    # reliability: between-candidate variance of p vs binomial noise (5 restarts)
    Rn = 5
    noise = np.mean(c.p * (1 - c.p)) / (Rn - 1)
    print(f"between-candidate share of variance in p_v: {max(c.p.var() - noise, 0) / c.p.var():.2f}")
    # old labels vs new p30
    if c.old_single_run_success.nunique() > 1:
        print(f"old single-run labels (iter 30) vs new 5-restart p@30: AUC = {roc_auc_score(c.old_single_run_success, c.p30):.2f}")
    print("Spearman with delta success prob (dp) and delta accuracy (dacc):")
    for col in ["degree", "reach3", "new_2witness", "consistency", "agreement", "dist_to_seeds",
                "candidate_gradient_row_top2_gap_z", "candidate_gradient_row_normalized_entropy",
                "candidate_signature_collision_reduction", "baseline_correct"]:
        print(f"   {col:45s} dp {spear(c[col], c.dp):+.2f}   dacc {spear(c[col], c.dacc):+.2f}")
    print("pick-the-best simulation (mean p_v of the top-1 / top-10 candidates by score):")
    for col, sgn in [("degree", 1), ("reach3", 1), ("new_2witness", 1), ("consistency", -1), ("agreement", -1),
                     ("candidate_gradient_row_top2_gap_z", 1), ("candidate_gradient_row_top2_gap_z", -1)]:
        s = sgn * c[col].to_numpy() + np.random.default_rng(0).random(len(c)) * 1e-9
        o = np.argsort(-s)
        print(f"   {('-' if sgn < 0 else '+') + col:47s} top1 p={c.p.iloc[o[0]]:.2f}  top10 p={c.p.iloc[o[:10]].mean():.2f}")
    print(f"   {'random candidate':47s} p={c.p.mean():.2f};  oracle best p={c.p.max():.2f}")
pd.concat(rows_all).to_csv(f"{OUT}/e3_candidates.csv", index=False)
