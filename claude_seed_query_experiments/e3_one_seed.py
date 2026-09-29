"""E3: the one-seed marginal problem (existing repo problems) re-evaluated with solver restarts.

For each saved problem (n=300 pair, existing verified seeds S0 sitting near the tipping point):
  * baseline: 16 restarts with S0 (max_iter 100, checkpoint 30) -> p0, plus label-free per-vertex
    uncertainty from the baseline solutions (local consistency on the best-objective restart,
    agreement across restarts)
  * every candidate v: 6 restarts with S0 + {v} -> p_v, mean accuracy
"""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))   # this folder
REPO = _os.path.dirname(HERE)                          # repository root
import json, os, time
import numpy as np, pandas as pd
from multiprocessing import Pool
import sgmlib as L

SRC = REPO + "/one_seed_marginal_problem/data"
OUT = HERE + "/results"
PROBLEMS = ["paper", "sparse_er", "sparse_sbm"]
R_BASE, R_CAND, MAX_ITER, N_CAND = 16, 5, 100, 150
PAIRS = {}


def load_problem(name):
    d = np.load(f"{SRC}/{name}_problem.npz")
    out = pd.read_csv(f"{SRC}/{name}_candidate_outcomes.csv")
    n = d["adjacency_graph1"].shape[0]
    truth = -np.ones(n, dtype=int)
    truth[d["seed_vertices_graph1"]] = d["seed_vertices_graph2"]
    truth[out.candidate_vertex_graph1.to_numpy()] = out.revealed_vertex_graph2.to_numpy()
    assert (truth >= 0).all() and len(np.unique(truth)) == n
    pair = L.Pair(name, L._to_csr(d["adjacency_graph1"]), L._to_csr(d["adjacency_graph2"]), truth)
    return pair, [int(v) for v in d["seed_vertices_graph1"]], out


def init():
    for p in PROBLEMS:
        PAIRS[p] = load_problem(p)


def cand_job(args):
    name, v, r = args
    pair, S0, _ = PAIRS[name]
    S = S0 + [v] if v >= 0 else S0
    res = L.sgm(pair, S, rng=np.random.default_rng([31, v + 1, r, len(name)]), max_iter=MAX_ITER, checkpoints=(30,))
    return dict(problem=name, candidate=v, restart=r, acc=res.acc, acc30=res.checkpoints[30],
                h0=res.h0_acc, score=res.score, n_iter=res.n_iter, perm=json.dumps(res.perm.tolist()) if v < 0 else "")


if __name__ == "__main__":
    tasks = []
    for name in PROBLEMS:
        pair, S0, out = load_problem(name)
        for r in range(R_BASE):
            tasks.append((name, -1, r))
        cands = out.candidate_vertex_graph1.to_numpy()
        cands = np.random.default_rng(len(name)).choice(cands, size=min(N_CAND, len(cands)), replace=False)
        for v in cands:
            for r in range(R_CAND):
                tasks.append((name, int(v), r))
    rows, t0 = [], time.time()
    with Pool(2, initializer=init) as pool:
        for i, row in enumerate(pool.imap_unordered(cand_job, tasks, chunksize=4)):
            rows.append(row)
            if (i + 1) % 400 == 0 or i + 1 == len(tasks):
                pd.DataFrame(rows).to_csv(f"{OUT}/e3_runs.csv", index=False)
                print(f"{i+1}/{len(tasks)} runs, {time.time()-t0:.0f}s", flush=True)
