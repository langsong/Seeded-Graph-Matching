"""E2: adaptive (batch-sequential) querying starting from already verified matches.

Setting per configuration: a fresh pair, an initial set S0 of verified matches, a query budget q
spent in 3 batches. Static rules choose all q queries from Graph A (conditioning on S0 where the
rule allows it). Adaptive rules run SGM with R_A restarts after each batch and use label-free
uncertainty of the current solution:
  * local consistency c(v): share of v's Graph-A edges preserved at its current match
    (computed on the best-objective restart)
  * agreement a(v): share of restarts that agree with the best-objective match of v
Final evaluation: R_EVAL fresh restarts with seeds S0 + queries.
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
PAIR_DIR = f"{OUT}/pairs_e2"
MAX_ITER = 100          # evaluation runs
MAX_ITER_STATE = 60     # cheaper runs used only to estimate the current uncertainty
R_A, R_EVAL, ROUNDS = 3, 4, 3
KCAL = {"sparse_er": 18, "sparse_sbm": 15, "paper": 30, "ier": 20}
N_PAIRS = 3
STATIC = ["random", "degree", "reach3", "cov2c3"]
ADAPTIVE = ["ad_consistency", "ad_disagreement", "ad_phase", "ad_cov_uncertain"]
PAIRS, STATS = {}, {}


def load(key):
    if key not in PAIRS:
        PAIRS[key] = L.load_npz_pair(f"{PAIR_DIR}/{key}.npz", key)
        STATS[key] = L.gstats(PAIRS[key].A)
    return PAIRS[key], STATS[key]


def make_pairs():
    os.makedirs(PAIR_DIR, exist_ok=True)
    for mi, model in enumerate(KCAL):
        for i in range(N_PAIRS):
            path = f"{PAIR_DIR}/{model}_{i}.npz"
            if not os.path.exists(path):
                pair = L.gen_pair(model, np.random.default_rng([4242, mi, i]))
                np.savez_compressed(path, adjacency_graph1=pair.A.toarray().astype(np.int8),
                                    adjacency_graph2=pair.B.toarray().astype(np.int8),
                                    true_permutation=pair.truth)


def uncertainty(pair, G, S, rng_seed):
    """Run R_A restarts; return best-objective perm, local consistency, agreement, summary."""
    runs = [L.sgm(pair, S, rng=np.random.default_rng([rng_seed, r]), max_iter=MAX_ITER_STATE) for r in range(R_A)]
    best = max(runs, key=lambda r: r.score)
    pi = best.perm
    A, B = pair.A, pair.B
    # local consistency: for each v, # neighbours u with pi(u) adjacent to pi(v) in B
    Bperm = B[pi][:, pi]                       # Bperm[v, u] = B[pi(v), pi(u)]
    preserved = np.asarray(A.multiply(Bperm).sum(axis=1)).ravel()
    deg = G.deg
    cons = np.where(deg > 0, preserved / np.maximum(deg, 1), 0.0)
    agree = np.mean([r.perm == pi for r in runs], axis=0)
    ignited = float(np.mean(agree == 1.0))
    return pi, cons, agree, dict(state_best_acc=best.acc, state_mean_acc=float(np.mean([r.acc for r in runs])),
                                 state_full_agree=ignited, state_mean_cons=float(cons[deg > 0].mean()))


def adaptive_pick(strategy, G, b, S, cons, agree, full_agree, rng):
    reach = G.reach(3)
    w = reach / reach.max()
    if strategy == "ad_consistency":
        score = (1.0 - cons) * w
    elif strategy == "ad_disagreement":
        score = (1.0 - agree) * w + 1e-3 * w
    elif strategy == "ad_phase":
        if full_agree < 0.5:          # restarts do not agree on most vertices: not ignited yet
            return L.select("cov2c3", G, b, rng, S0=S)
        score = (1.0 - cons) * w
    elif strategy == "ad_cov_uncertain":
        # greedy two-witness coverage, counting only vertices whose current match looks unreliable
        U = ((cons < 0.5) | (agree < 1.0)).astype(float)
        M = G.ball(1).astype(float)
        cnt = M[:, list(S)].sum(axis=1)
        picks = []
        for _ in range(b):
            gain = M.T @ (U * (cnt < 2)) + 1e-6 * w + rng.random(len(w)) * 1e-9
            gain[list(S) + picks] = -np.inf
            u = int(np.argmax(gain))
            picks.append(u)
            cnt += M[:, u]
        return picks
    score = score + rng.random(len(score)) * 1e-9
    score[list(S)] = -np.inf
    return [int(v) for v in np.argsort(-score)[:b]]


def evaluate(pair, S, seed):
    res = L.run_restarts(pair, S, R=R_EVAL, base_seed=seed, max_iter=MAX_ITER)
    return {k: v for k, v in res.items() if k != "runs"}


def job(args):
    key, model, s0_draw, k0, q, strategy, draw = args
    pair, G = load(key)
    rng = np.random.default_rng([99, s0_draw, stable(key)])
    S0 = [int(v) for v in rng.choice(pair.n, size=k0, replace=False)] if k0 > 0 else []
    srng = np.random.default_rng([5, s0_draw, draw, sum(map(ord, strategy)), stable(key)])
    t0 = time.time()
    trace = []
    if strategy in STATIC:
        Q = L.select(strategy, G, q, srng, S0=S0)
    else:
        S = list(S0)
        batches = [len(a) for a in np.array_split(np.arange(q), ROUNDS)]
        for rd, b in enumerate(batches):
            if len(S) == 0:
                picks = L.select("cov2c3", G, b, srng, S0=S)      # nothing to be uncertain about yet
                trace.append(dict(round=rd, state=None))
            else:
                pi, cons, agree, st = uncertainty(pair, G, S, rng_seed=1000 + rd)
                picks = adaptive_pick(strategy, G, b, S, cons, agree, st["state_full_agree"], srng)
                trace.append(dict(round=rd, **st))
            S.extend(picks)
        Q = S[len(S0):]
    S = S0 + list(Q)
    ev = evaluate(pair, S, seed=777)
    feats = dict(f_deg_q=float(G.deg[Q].mean()), f_reach3_q=float(G.reach(3)[Q].mean()))
    return dict(pair=key, model=model, s0_draw=s0_draw, draw=draw, k0=k0, q=q, k_total=len(S), strategy=strategy,
                secs=time.time() - t0, seeds=json.dumps(S), trace=json.dumps(trace), **feats, **ev)


if __name__ == "__main__":
    make_pairs()
    tasks = []
    for model, kc in KCAL.items():
        k0 = int(round(0.4 * kc))
        q = int(round(0.4 * kc))
        for i in range(N_PAIRS):
            key = f"{model}_{i}"
            for s0_draw in range(2):
                for strat in STATIC + ADAPTIVE:
                    tasks.append((key, model, s0_draw, k0, q, strat, 0))
                tasks.append((key, model, s0_draw, k0, q, "random", 1))
            # no verified matches at all: the whole 0.8*kcal budget is spent on queries
            for strat in STATIC + ADAPTIVE:
                tasks.append((key, model, 9, 0, k0 + q, strat, 0))
            tasks.append((key, model, 9, 0, k0 + q, "random", 1))
    rows, t0 = [], time.time()
    with Pool(2) as pool:
        for i, row in enumerate(pool.imap_unordered(job, tasks, chunksize=1)):
            rows.append(row)
            if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                pd.DataFrame(rows).to_csv(f"{OUT}/e2_runs.csv", index=False)
                print(f"{i+1}/{len(tasks)} configs, {time.time()-t0:.0f}s", flush=True)
