import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))   # this folder
REPO = _os.path.dirname(HERE)                          # repository root
import numpy as np, time, pandas as pd, json
from scipy.optimize import quadratic_assignment
import sgmlib as L
D = REPO + "/fixed_pair_seed_set_classification/data"
for name in ["paper", "sparse_er"]:
    pair = L.load_npz_pair(f"{D}/{name}_problem.npz", name)
    df = pd.read_csv(f"{D}/{name}_seed_sets.csv")
    A = pair.A.toarray(); B = pair.B.toarray()
    for i in range(3):
        seeds = np.array(json.loads(df.seed_vertices_graph1.iloc[i]))
        pm = np.column_stack([seeds, pair.truth[seeds]])
        t0 = time.time()
        res = quadratic_assignment(A, B, method="faq", options=dict(maximize=True, partial_match=pm, P0="barycenter", shuffle_input=False, maxiter=30, tol=0.01))
        t1 = time.time()
        mine = L.sgm(pair, seeds, rng=None, max_iter=30, tol=0.01)
        t2 = time.time()
        non = np.setdiff1d(np.arange(pair.n), seeds)
        acc_scipy = np.mean(res.col_ind[non] == pair.truth[non])
        print(f"{name} set{i}: identical perm={np.array_equal(res.col_ind, mine.perm)} nit scipy={res.nit} mine={mine.n_iter} acc scipy={acc_scipy:.3f} mine={mine.acc:.3f} score {res.fun:.0f}/{mine.score:.0f}  time scipy {t1-t0:.2f}s mine {t2-t1:.2f}s")
