import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))   # this folder
REPO = _os.path.dirname(HERE)                          # repository root
import numpy as np, pandas as pd, json
import scipy.sparse as sp
import sgmlib as L
D = REPO + "/fixed_pair_seed_set_classification/data"
pair = L.load_npz_pair(f"{D}/paper_problem.npz", "paper")
df = pd.read_csv(f"{D}/paper_seed_sets.csv")
seeds = np.array(json.loads(df.seed_vertices_graph1.iloc[0]))
n = pair.n; s = len(seeds); u = n - s
A = pair.A.toarray(); B = pair.B.toarray()
non_A = np.setdiff1d(np.arange(n), seeds); non_B = np.setdiff1d(np.arange(n), pair.truth[seeds])
pA = np.concatenate([seeds, non_A]); pB = np.concatenate([pair.truth[seeds], non_B])
Ap = A[pA][:, pA]; Bp = B[pB][:, pB]
A21, A12, A22 = Ap[s:, :s], Ap[:s, s:], Ap[s:, s:]
B21, B12, B22 = Bp[s:, :s], Bp[:s, s:], Bp[s:, s:]
const = A21 @ B21.T + A12.T @ B12
P = np.ones((u, u)) / u
g_dense = const + A22 @ P @ B22.T + A22.T @ P @ B22
sA22 = sp.csr_matrix(A22); sB22 = sp.csr_matrix(B22)
X = sA22 @ P
g_sparse = const + (sB22 @ X.T).T + (sB22.T.tocsr() @ (sA22.T.tocsr() @ P).T).T
print("max abs diff grad:", np.abs(g_dense - g_sparse).max(), " n distinct values:", len(np.unique(np.round(g_dense, 9))))
from scipy.optimize import linear_sum_assignment
c1 = linear_sum_assignment(g_dense, maximize=True)[1]; c2 = linear_sum_assignment(g_sparse, maximize=True)[1]
print("LAP identical:", np.array_equal(c1, c2), " objective equal:", np.isclose(g_dense[np.arange(u), c1].sum(), g_dense[np.arange(u), c2].sum()))
