"""E1b: focused head-to-head at the most discriminating budget per regime, on 12 new pairs.

Budgets are chosen from E1 as the k where top-degree seeds succeed about half the time,
so differences between strong rules are easiest to see.
"""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))   # this folder
REPO = _os.path.dirname(HERE)                          # repository root
import json, os, sys, time, zlib
import numpy as np, pandas as pd
from multiprocessing import Pool
import sgmlib as L

OUT = HERE + "/results"
PAIR_DIR = f"{OUT}/pairs_e1b"
MAX_ITER, R, N_PAIRS = 100, 2, 12
BUDGETS = json.loads(os.environ.get("E1B_BUDGETS", '{"sparse_er": 11, "sparse_sbm": 9, "paper": 18, "ier": 12}'))
RULES = ["degree", "betweenness", "reach3", "cov1c2", "cov2c3", "strat_degree"]
N_RANDOM = 2
PAIRS, STATS = {}, {}


def stable(x):
    return zlib.crc32(str(x).encode()) % 10**6


def load(key):
    if key not in PAIRS:
        PAIRS[key] = L.load_npz_pair(f"{PAIR_DIR}/{key}.npz", key)
        STATS[key] = L.gstats(PAIRS[key].A)
    return PAIRS[key], STATS[key]


def job(args):
    key, model, k, rule, draw = args
    pair, G = load(key)
    rng = np.random.default_rng([21, stable(key), k, stable(rule), draw])
    S = L.select(rule, G, k, rng)
    out = []
    for r in range(R):
        res = L.sgm(pair, S, rng=np.random.default_rng([8, r, k, draw, stable(key)]), max_iter=MAX_ITER, checkpoints=(30,))
        out.append(dict(pair=key, model=model, k=k, rule=rule, draw=draw, restart=r, acc=res.acc,
                        acc30=res.checkpoints[30], h0=res.h0_acc, score=res.score, n_iter=res.n_iter,
                        seeds=json.dumps([int(v) for v in S])))
    return out


if __name__ == "__main__":
    os.makedirs(PAIR_DIR, exist_ok=True)
    tasks = []
    for mi, (model, k) in enumerate(BUDGETS.items()):
        for i in range(N_PAIRS):
            key = f"{model}_{i}"
            path = f"{PAIR_DIR}/{key}.npz"
            if not os.path.exists(path):
                pair = L.gen_pair(model, np.random.default_rng([777, mi, i]))
                np.savez_compressed(path, adjacency_graph1=pair.A.toarray().astype(np.int8),
                                    adjacency_graph2=pair.B.toarray().astype(np.int8), true_permutation=pair.truth)
            for d in range(N_RANDOM):
                tasks.append((key, model, k, "random", d))
            for rule in RULES:
                tasks.append((key, model, k, rule, 0))
    rows, t0 = [], time.time()
    with Pool(2) as pool:
        for i, res in enumerate(pool.imap_unordered(job, tasks, chunksize=1)):
            rows.extend(res)
            if (i + 1) % 40 == 0 or i + 1 == len(tasks):
                pd.DataFrame(rows).to_csv(f"{OUT}/e1b_runs.csv", index=False)
                print(f"{i+1}/{len(tasks)} sets, {time.time()-t0:.0f}s", flush=True)
