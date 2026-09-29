"""E4: real-graph check on Enron email graphs (utils/enron.mat, 184 employees x 187 weeks).

Three real pairs with known identities:
  * week:   weeks 148 vs 149 (the repo's fixed Enron pair, 129 shared active vertices)
  * window: 4-week unions, weeks 133-136 vs 137-140 (1-based), vertices active in both
  * sub:    parent = edges active in >= 6 of weeks 101-160; two children keep each edge w.p. 0.75
Rules are compared across budgets with 6 solver restarts (max_iter 100).
Extra rule 'hybrid_twins': half the budget on top-degree vertices (ignition), the rest on
structural twins (vertices whose Graph-A neighbourhoods are identical and therefore cannot be
told apart by any matching objective).
"""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))   # this folder
REPO = _os.path.dirname(HERE)                          # repository root
import json, os, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.io import loadmat
import sgmlib as L

OUT = HERE + "/results"
MAT = REPO + "/utils/enron.mat"
NPZ = REPO + "/fixed_pair_seed_set_classification/data/enron_problem.npz"
R, MAX_ITER, N_RANDOM = 6, 100, 8
BUDGETS = [2, 4, 6, 8, 12, 16, 24]
RULES = ["degree", "betweenness", "reach3", "cov1c2", "cov2c3", "genrank", "dispersion", "twins", "hybrid_twins"]
PAIRS, STATS = {}, {}


def _undirected(M):
    M = M | M.T
    np.fill_diagonal(M, False)
    return M


def build_pairs():
    AAA = loadmat(MAT)["AAA"].astype(bool)
    out = {}
    out["week"] = L.load_npz_pair(NPZ, "week")
    rng = np.random.default_rng(2026)
    # window pair
    G1 = _undirected(AAA[:, :, 132:136].any(axis=2)); G2 = _undirected(AAA[:, :, 136:140].any(axis=2))
    act = np.flatnonzero((G1.sum(1) > 0) & (G2.sum(1) > 0))
    G1, G2 = G1[np.ix_(act, act)], G2[np.ix_(act, act)]
    sh = rng.permutation(len(act))
    out["window"] = L.Pair("window", L._to_csr(G1.astype(float)), L._to_csr(G2[np.ix_(sh, sh)].astype(float)), np.argsort(sh))
    # edge-subsampled parent
    C = AAA[:, :, 100:160].astype(int).sum(axis=2); C = C + C.T; np.fill_diagonal(C, 0)
    Pm = C >= 6
    act = np.flatnonzero(Pm.sum(1) > 0)
    Pm = Pm[np.ix_(act, act)]
    iu = np.triu_indices(len(act), 1)
    e = Pm[iu]
    k1 = e & (rng.random(e.shape) < 0.75); k2 = e & (rng.random(e.shape) < 0.75)
    def mk(keep):
        M = np.zeros(Pm.shape, dtype=bool); M[iu[0][keep], iu[1][keep]] = True; return M | M.T
    H1, H2 = mk(k1), mk(k2)
    sh = rng.permutation(len(act))
    out["sub"] = L.Pair("sub", L._to_csr(H1.astype(float)), L._to_csr(H2[np.ix_(sh, sh)].astype(float)), np.argsort(sh))
    return out


def twin_order(G):
    """Vertices ordered for twin resolution: members of identical-neighbourhood classes, one
    representative left out per class, larger classes / higher degree first."""
    A = G.A.tolil()
    keys = {}
    for v in range(A.shape[0]):
        keys.setdefault(tuple(A.rows[v]), []).append(v)
    cands = []
    for members in keys.values():
        if len(members) >= 2 and len(members[0:1]) and len(A.rows[members[0]]) > 0:
            ms = sorted(members, key=lambda v: -G.deg[v])
            for v in ms[:-1]:
                cands.append((len(members), G.deg[v], v))
    cands.sort(key=lambda t: (-t[0], -t[1]))
    return [v for _, _, v in cands]


def select(rule, G, k, rng):
    if rule == "twins":
        order = twin_order(G)
        picks = order[:k]
        if len(picks) < k:
            picks += L.select("degree", G, k - len(picks), rng, S0=picks)
        return picks
    if rule == "hybrid_twins":
        h = int(np.ceil(k / 2))
        picks = L.select("degree", G, h, rng)
        rest = [v for v in twin_order(G) if v not in picks][: k - h]
        picks += rest
        if len(picks) < k:
            picks += L.select("degree", G, k - len(picks), rng, S0=picks)
        return picks
    return L.select(rule, G, k, rng)


def init():
    global PAIRS, STATS
    PAIRS = build_pairs()
    STATS = {k: L.gstats(p.A) for k, p in PAIRS.items()}


def job(args):
    key, k, rule, draw = args
    pair, G = PAIRS[key], STATS[key]
    rng = np.random.default_rng([77, k, draw, sum(map(ord, rule)), len(key)])
    S = select(rule, G, k, rng)
    assert len(set(S)) == k
    out = []
    for r in range(R):
        res = L.sgm(pair, S, rng=np.random.default_rng([13, r, k, draw, len(key)]), max_iter=MAX_ITER, checkpoints=(30,))
        out.append(dict(pair=key, n=pair.n, k=k, rule=rule, draw=draw, restart=r, acc=res.acc,
                        acc30=res.checkpoints[30], h0=res.h0_acc, score=res.score, n_iter=res.n_iter,
                        seeds=json.dumps([int(v) for v in S])))
    return out


if __name__ == "__main__":
    init()
    for key, p in PAIRS.items():
        G = STATS[key]
        B = p.B[p.truth][:, p.truth]
        shared = p.A.multiply(B).sum() / 2
        print(f"{key}: n={p.n} edgesA={p.A.nnz//2} edgesB={p.B.nnz//2} shared={int(shared)} "
              f"avgdeg={G.deg.mean():.1f} maxdeg={G.deg.max():.0f} deg1={int((G.deg==1).sum())} "
              f"twin-resolvable vertices={len(twin_order(G))}")
    tasks = []
    for key in PAIRS:
        for k in BUDGETS:
            for d in range(N_RANDOM):
                tasks.append((key, k, "random", d))
            for rule in RULES:
                tasks.append((key, k, rule, 0))
    rows, t0 = [], time.time()
    with Pool(2, initializer=init) as pool:
        for i, res in enumerate(pool.imap_unordered(job, tasks, chunksize=2)):
            rows.extend(res)
            if (i + 1) % 50 == 0 or i + 1 == len(tasks):
                pd.DataFrame(rows).to_csv(f"{OUT}/e4_runs.csv", index=False)
                print(f"{i+1}/{len(tasks)} sets, {time.time()-t0:.0f}s", flush=True)
