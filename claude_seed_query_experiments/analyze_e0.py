import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))   # this folder
REPO = _os.path.dirname(HERE)                          # repository root
import json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import sgmlib as L

OUT = HERE + "/results"
DATA = REPO + "/fixed_pair_seed_set_classification/data"
runs = pd.read_csv(f"{OUT}/e0_runs.csv")
sets = pd.read_csv(f"{OUT}/e0_sets.csv")
runs = runs.merge(sets, on=["model", "set_id"])
for c in ["acc30", "acc60", "acc100", "acc_final"]:
    runs["s" + c[3:]] = runs[c] >= 0.9

print("=== 1. Validation against saved graspologic outcomes (iteration cap 30) ===")
g = runs.groupby(["model", "set_id"]).agg(p30=("s30", "mean"), m30=("acc30", "mean"), saved=("saved_success", "first"),
                                         saved_acc=("saved_acc", "first"), gl=("gl_success_count", "first"),
                                         pfin=("sinal" if False else "s_final", "mean"), mfin=("acc_final", "mean"))
for m, d in g.groupby(level=0):
    auc = roc_auc_score(d.saved, d.p30) if d.saved.nunique() > 1 else np.nan
    line = f"{m:10s} sets={len(d)} mine p(success@30)={d.p30.mean():.2f} saved label rate={d.saved.mean():.2f} AUC(mine p30 -> saved single-run label)={auc:.2f}"
    if m == "paper":
        r = np.corrcoef(d.gl / 5, d.p30)[0, 1]
        line += f" | corr(graspologic 5-run rate, mine 3-run rate)={r:.2f}; graspologic mean={d.gl.mean()/5:.2f}"
    print(line)

print("\n=== 2. Iteration cap: success probability by checkpoint ===")
print(runs.groupby("model")[["s30", "s60", "s100", "s_final"]].mean().round(3).to_string())
print("\nrun-level transitions from iteration 30 to final (150):")
for m, d in runs.groupby("model"):
    f2s = ((~d.s30) & d.s_final).mean(); s2f = (d.s30 & ~d.s_final).mean()
    print(f"  {m:10s} fail@30->success@final={f2s:.3f}  success@30->fail@final={s2f:.3f}  converged by 150={d.converged.mean():.2f}  median iters={d.n_iter.median():.0f}")

print("\n=== 3. Solver noise: sets with mixed outcomes across 3 restarts ===")
for col in ["s30", "s_final"]:
    mixed = runs.groupby(["model", "set_id"])[col].agg(lambda x: 0 < x.mean() < 1).groupby(level=0).mean()
    print(col, mixed.round(2).to_dict())

print("\n=== 4. When is the outcome decided? AUC of direction accuracy at iteration t for final success@30 ===")
T = [1, 2, 3, 5, 8, 12, 20, 30]
rows = []
for m, d in runs.groupby("model"):
    trajs = [json.loads(t) for t in d.traj]
    y = d.s30.to_numpy()
    rec = {"model": m}
    for t in T:
        x = np.array([tr[t - 1] if len(tr) >= t else tr[-1] for tr in trajs])
        rec[f"t{t}"] = roc_auc_score(y, x) if 0 < y.mean() < 1 else np.nan
    rows.append(rec)
print(pd.DataFrame(rows).set_index("model").round(2).to_string())
print("\nmean direction accuracy at iteration t (success vs failure @30):")
for m, d in runs.groupby("model"):
    trajs = [json.loads(t) for t in d.traj]
    for lab, mask in [("succ", d.s30.to_numpy()), ("fail", ~d.s30.to_numpy())]:
        vals = [np.mean([tr[t - 1] if len(tr) >= t else tr[-1] for tr, mk in zip(trajs, mask) if mk]) for t in T]
        print(f"  {m:10s} {lab}: " + " ".join(f"t{t}={v:.2f}" for t, v in zip(T, vals)))

print("\n=== 5. Per-vertex correctness vs witnesses (final, all runs) ===")
recs = []
for m, d in runs.groupby("model"):
    pair = L.load_npz_pair(f"{DATA}/{m}_problem.npz", m)
    G = L.gstats(pair.A)
    seedsets = {int(r.set_id): None for r in d.itertuples()}
    df_sets = pd.read_csv(f"{DATA}/{m}_seed_sets.csv").set_index("candidate_id")
    for r in d.itertuples():
        S = np.array(json.loads(df_sets.loc[r.set_id].seed_vertices_graph1))
        non = np.array(json.loads(r.nonseeds))
        bits = np.unpackbits(np.frombuffer(bytes.fromhex(r.correct), dtype=np.uint8))[: len(non)].astype(bool)
        dd = G.dist[np.ix_(non, S)]
        recs.append(pd.DataFrame(dict(model=m, success=r.s_final, correct=bits, deg=G.deg[non],
                                      w1=(dd <= 1).sum(1), w2=(dd <= 2).sum(1), w3=(dd <= 3).sum(1))))
V = pd.concat(recs)
V.to_csv(f"{OUT}/e0_vertex.csv.gz", index=False)
for m, d in V.groupby("model"):
    print(f"--- {m}: P(correct) by #seed neighbours w1 (all runs | failed runs | successful runs)")
    for lab, dd in [("all", d), ("fail", d[~d.success]), ("succ", d[d.success])]:
        t = dd.groupby(dd.w1.clip(upper=3)).correct.mean().round(2).to_dict()
        print(f"    {lab}: {t}")
    s = d[d.success]
    print("    in successful runs, P(correct) by degree:", s.groupby(s.deg.clip(upper=4)).correct.mean().round(2).to_dict(),
          " share of errors with degree<=1:", round(float((s[~s.correct].deg <= 1).mean()), 2))
