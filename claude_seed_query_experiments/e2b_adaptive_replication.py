"""E2b: replication of the one adaptive gain seen in E2, on new pairs, plus a static control.

Setting as in E2 (with verified matches): S0 = 0.4*kcal random verified vertices, q = 0.4*kcal queries.
Strategies:
  degree, reach3                    static baselines
  reach3_wd  (static control)       reach3 x (1 - P(correct | #S0-neighbours)), using the witness curve
                                    measured in E0: P = .12, .38, .68, .90 for 0, 1, 2, 3+ seed neighbours.
                                    Tests whether the adaptive gain is just "skip vertices the verified
                                    seeds already pin down".
  ad_disagreement (adaptive)        as in E2: reach3 x (1 - agreement across restarts), 3 batches.
"""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))   # this folder
REPO = _os.path.dirname(HERE)                          # repository root
import json, os, time
import numpy as np, pandas as pd
from multiprocessing import Pool
import sgmlib as L
import e2_adaptive as E

OUT = HERE + "/results"
PAIR_DIR = f"{OUT}/pairs_e2b"
N_PAIRS = 4
STRATS = ["degree", "reach3", "reach3_wd", "ad_disagreement"]
WCURVE = np.array([0.12, 0.38, 0.68, 0.90])
PAIRS, STATS = {}, {}


def load(key):
    if key not in PAIRS:
        PAIRS[key] = L.load_npz_pair(f"{PAIR_DIR}/{key}.npz", key)
        STATS[key] = L.gstats(PAIRS[key].A)
    return PAIRS[key], STATS[key]


def job(args):
    key, model, s0_draw, k0, q, strategy = args
    pair, G = load(key)
    rng = np.random.default_rng([99, s0_draw, E.stable(key)])
    S0 = [int(v) for v in rng.choice(pair.n, size=k0, replace=False)]
    srng = np.random.default_rng([6, s0_draw, sum(map(ord, strategy)), E.stable(key)])
    if strategy in ("degree", "reach3"):
        Q = L.select(strategy, G, q, srng, S0=S0)
    elif strategy == "reach3_wd":
        w1 = G.ball(1)[:, S0].sum(axis=1)
        score = G.reach(3) * (1.0 - WCURVE[np.minimum(w1, 3)]) + srng.random(pair.n) * 1e-9
        score[S0] = -np.inf
        Q = [int(v) for v in np.argsort(-score)[:q]]
    else:
        S = list(S0)
        for rd, b in enumerate([len(a) for a in np.array_split(np.arange(q), E.ROUNDS)]):
            pi, cons, agree, st = E.uncertainty(pair, G, S, rng_seed=1000 + rd)
            S.extend(E.adaptive_pick("ad_disagreement", G, b, S, cons, agree, st["state_full_agree"], srng))
        Q = S[len(S0):]
    S = S0 + list(Q)
    ev = E.evaluate(pair, S, seed=777)
    return dict(pair=key, model=model, s0_draw=s0_draw, k0=k0, q=q, strategy=strategy, seeds=json.dumps(S), **ev)


if __name__ == "__main__":
    os.makedirs(PAIR_DIR, exist_ok=True)
    tasks = []
    for mi, (model, kc) in enumerate(E.KCAL.items()):
        k0 = q = int(round(0.4 * kc))
        for i in range(N_PAIRS):
            key = f"{model}_{i}"
            path = f"{PAIR_DIR}/{key}.npz"
            if not os.path.exists(path):
                pair = L.gen_pair(model, np.random.default_rng([5151, mi, i]))
                np.savez_compressed(path, adjacency_graph1=pair.A.toarray().astype(np.int8),
                                    adjacency_graph2=pair.B.toarray().astype(np.int8), true_permutation=pair.truth)
            for s0_draw in range(2):
                for s in STRATS:
                    tasks.append((key, model, s0_draw, k0, q, s))
    rows, t0 = [], time.time()
    with Pool(2) as pool:
        for i, row in enumerate(pool.imap_unordered(job, tasks, chunksize=1)):
            rows.append(row)
            if (i + 1) % 16 == 0 or i + 1 == len(tasks):
                pd.DataFrame(rows).to_csv(f"{OUT}/e2b_runs.csv", index=False)
                print(f"{i+1}/{len(tasks)} configs, {time.time()-t0:.0f}s", flush=True)
