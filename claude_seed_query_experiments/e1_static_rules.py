"""E1: static (non-adaptive) query rules at equal budget on fresh graph pairs.

For each regime, generate fresh correlated pairs, choose k Graph-A vertices with each rule,
reveal their true partners (the paid queries), and run SGM with R tie-breaking restarts.
Outcomes: mean non-seed accuracy, success probability (acc >= 0.9), best-objective-of-R accuracy.
"""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))   # this folder
REPO = _os.path.dirname(HERE)                          # repository root
import json, os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
import sgmlib as L
import zlib


def stable(x):
    return zlib.crc32(str(x).encode()) % 10**6

OUT = HERE + "/results"
PAIR_DIR = f"{OUT}/pairs"
MAX_ITER = int(os.environ.get("E1_MAX_ITER", 100))
R = 2
N_PAIRS = int(os.environ.get("E1_PAIRS", 6))
N_RANDOM = 3
RULES = ["degree", "betweenness", "reach3", "cov1c2", "cov2c3", "genrank", "strat_degree", "dispersion"]
BUDGETS = {
    "sparse_er": [7, 11, 14, 18, 23],
    "sparse_sbm": [6, 9, 12, 15, 20],
    "paper": [12, 18, 24, 30, 39],
    "ier": [8, 12, 16, 20, 26],
    "dense_sbm": [8, 12, 16, 20, 26],
}
MODELS = list(BUDGETS)
PAIRS, STATS = {}, {}


def pair_key(model, i):
    return f"{model}_{i}"


def make_pairs():
    os.makedirs(PAIR_DIR, exist_ok=True)
    for mi, model in enumerate(MODELS):
        for i in range(N_PAIRS):
            path = f"{PAIR_DIR}/{pair_key(model, i)}.npz"
            if os.path.exists(path):
                continue
            pair = L.gen_pair(model, np.random.default_rng([2026, mi, i]))
            np.savez_compressed(path, adjacency_graph1=pair.A.toarray().astype(np.int8),
                                adjacency_graph2=pair.B.toarray().astype(np.int8),
                                true_permutation=pair.truth)


def load(key):
    if key not in PAIRS:
        PAIRS[key] = L.load_npz_pair(f"{PAIR_DIR}/{key}.npz", key)
        STATS[key] = L.gstats(PAIRS[key].A)
    return PAIRS[key], STATS[key]


def set_features(G, S):
    S = np.asarray(S)
    n = G.dist.shape[0]
    non = np.setdiff1d(np.arange(n), S)
    d = G.dist[np.ix_(non, S)]
    w1 = (d <= 1).sum(1)
    pd_ = G.dist[np.ix_(S, S)][np.triu_indices(len(S), 1)]
    pd_ = pd_[np.isfinite(pd_)]
    return dict(f_deg=float(G.deg[S].mean()), f_three_hop=float((d <= 3).sum(1).mean()),
                f_cov1=float((w1 >= 1).mean()), f_cov2=float((w1 >= 2).mean()),
                f_2hop3=float(((d <= 2).sum(1) >= 3).mean()),
                f_pairdist=float(pd_.mean()) if pd_.size else np.nan)


def job(args):
    key, model, k, rule, draw = args
    pair, G = load(key)
    rng = np.random.default_rng([11, stable(key), k, RULES.index(rule) if rule in RULES else 99, draw])
    t0 = time.time()
    S = L.select(rule, G, k, rng)
    sel_secs = time.time() - t0
    out = []
    for r in range(R):
        res = L.sgm(pair, S, rng=np.random.default_rng([3, r, k, draw, stable(key)]), max_iter=MAX_ITER, checkpoints=(30,))
        non = np.setdiff1d(np.arange(pair.n), S)
        correct = res.perm[non] == pair.truth[non]
        out.append(dict(pair=key, model=model, k=k, rule=rule, draw=draw, restart=r,
                        acc=res.acc, acc30=res.checkpoints[30], h0=res.h0_acc, score=res.score, n_iter=res.n_iter,
                        converged=res.converged, sel_secs=sel_secs,
                        seeds=json.dumps([int(v) for v in S]),
                        correct=np.packbits(correct).tobytes().hex(), **set_features(G, S)))
    return out


if __name__ == "__main__":
    make_pairs()
    tag = os.environ.get("E1_TAG", f"iter{MAX_ITER}")
    models = sys.argv[1:] or MODELS
    tasks = []
    for model in models:
        for i in range(N_PAIRS):
            key = pair_key(model, i)
            for k in BUDGETS[model]:
                for draw in range(N_RANDOM):
                    tasks.append((key, model, k, "random", draw))
                for rule in RULES:
                    tasks.append((key, model, k, rule, 0))
    # interleave so partial results cover all rules
    rows, t0 = [], time.time()
    path = f"{OUT}/e1_runs_{tag}.csv"
    with Pool(2) as pool:
        for i, res in enumerate(pool.imap_unordered(job, tasks, chunksize=1)):
            rows.extend(res)
            if (i + 1) % 40 == 0 or i + 1 == len(tasks):
                pd.DataFrame(rows).to_csv(path, index=False)
                print(f"{i+1}/{len(tasks)} sets, {time.time()-t0:.0f}s", flush=True)
