"""E0: protocol check on the saved fixed pairs.

For saved seed sets on the four synthetic fixed pairs, run the re-implemented SGM
with 3 tie-breaking seeds up to 150 iterations, recording
  * accuracy of the projected iterate at iterations 30 (graspologic default cap), 60, 100, final
  * H0 and the accuracy of every Frank-Wolfe assignment direction (trajectory)
  * per-vertex final correctness (for witness-count analysis)
PAPER uses sets from the saved graspologic solver-variance audit (5 graspologic runs per set)
so both implementations can be compared set by set.
"""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))   # this folder
REPO = _os.path.dirname(HERE)                          # repository root
import json, os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
import sgmlib as L

DATA = REPO + "/fixed_pair_seed_set_classification/data"
OUT = HERE + "/results"
MODELS = ["sparse_er", "sparse_sbm", "paper", "ier"]
R = 3
PAIRS = {}


def init():
    for m in MODELS:
        PAIRS[m] = L.load_npz_pair(f"{DATA}/{m}_problem.npz", m)


def task(args):
    model, set_id, seeds, r = args
    pair = PAIRS[model]
    t0 = time.time()
    res = L.sgm(pair, seeds, rng=np.random.default_rng([7, set_id, r]), max_iter=150, tol=0.01,
                record_traj=True, checkpoints=(30, 60, 100))
    non = np.setdiff1d(np.arange(pair.n), seeds)
    correct = (res.perm[non] == pair.truth[non])
    return dict(model=model, set_id=set_id, restart=r, h0=res.h0_acc,
                acc30=res.checkpoints[30], acc60=res.checkpoints[60], acc100=res.checkpoints[100],
                acc_final=res.acc, n_iter=res.n_iter, converged=res.converged, score=res.score,
                traj=json.dumps([round(x, 4) for x in res.traj]), secs=time.time() - t0,
                correct=np.packbits(correct).tobytes().hex(), nonseeds=json.dumps(non.tolist()))


def build_tasks(n_per_model=50, seed=0):
    rng = np.random.default_rng(seed)
    tasks, meta = [], []
    for m in MODELS:
        df = pd.read_csv(f"{DATA}/{m}_seed_sets.csv")
        if m == "paper":
            aud = pd.read_csv(f"{DATA}/solver_variance/paper_solver_variance_runs.csv")
            cnt = aud.groupby("candidate_id").sgm_success.sum()
            ids = []
            for k in range(6):   # stratify by graspologic 5-run success count
                pool = cnt.index[cnt == k].to_numpy()
                ids += list(rng.choice(pool, size=min(10, len(pool)), replace=False))
            sel = df.set_index("candidate_id").loc[ids].reset_index()
            sel["gl_success_count"] = cnt.loc[ids].to_numpy()
        else:
            d = df[df.sampling_strategy == "random"]
            s1 = d[d.sgm_success == True].sample(n_per_model // 2, random_state=seed)
            s0 = d[d.sgm_success == False].sample(n_per_model // 2, random_state=seed)
            sel = pd.concat([s1, s0])
            sel["gl_success_count"] = np.nan
        for _, row in sel.iterrows():
            seeds = np.array(json.loads(row.seed_vertices_graph1))
            meta.append(dict(model=m, set_id=int(row.candidate_id), saved_success=bool(row.sgm_success),
                             saved_acc=row.final_unseeded_accuracy, saved_h0=row.h0_accuracy,
                             gl_success_count=row.gl_success_count))
            for r in range(R):
                tasks.append((m, int(row.candidate_id), seeds, r))
    return tasks, pd.DataFrame(meta)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    tasks, meta = build_tasks()
    meta.to_csv(f"{OUT}/e0_sets.csv", index=False)
    rows = []
    t0 = time.time()
    with Pool(2, initializer=init) as pool:
        for i, row in enumerate(pool.imap_unordered(task, tasks, chunksize=2)):
            rows.append(row)
            if (i + 1) % 50 == 0 or i + 1 == len(tasks):
                pd.DataFrame(rows).to_csv(f"{OUT}/e0_runs.csv", index=False)
                print(f"{i+1}/{len(tasks)} runs, {time.time()-t0:.0f}s", flush=True)
