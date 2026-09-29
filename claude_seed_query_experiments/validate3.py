import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))   # this folder
REPO = _os.path.dirname(HERE)                          # repository root
import numpy as np, pandas as pd, json
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
D = REPO + "/fixed_pair_seed_set_classification/data"
import sgmlib as L
pair = L.load_npz_pair(f"{D}/sparse_er_problem.npz", "x")
df = pd.read_csv(f"{D}/sparse_er_seed_sets.csv")
seeds = np.array(json.loads(df.seed_vertices_graph1.iloc[2]))
n = pair.n; s = len(seeds); u = n - s
A = pair.A.toarray(); B = pair.B.toarray()
non_A = np.setdiff1d(np.arange(n), seeds); non_B = np.setdiff1d(np.arange(n), pair.truth[seeds])
pA = np.concatenate([seeds, non_A]); pB = np.concatenate([pair.truth[seeds], non_B])
Ap = A[pA][:, pA]; Bp = B[pB][:, pB]
A21, A12, A22 = Ap[s:, :s], Ap[:s, s:], Ap[s:, s:]
B21, B12, B22 = Bp[s:, :s], Bp[:s, s:], Bp[s:, s:]
sA21, sA12, sA22, sB21, sB12, sB22 = map(sp.csr_matrix, (A21, A12, A22, B21, B12, B22))
rng = np.random.default_rng(1)
P = rng.random((u, u)); P = P / P.sum(1, keepdims=True)   # arbitrary P to test the formulas
cols = rng.permutation(u); rows = np.arange(u)
Q = np.eye(u)[cols]; R = P - Q
# scipy dense formulas
b21 = ((R.T @ A21) * B21).sum(); b12 = ((R.T @ A12.T) * B12.T).sum()
AR22 = A22.T @ R; BR22 = B22 @ R.T
b22a = (AR22 * B22.T[cols]).sum(); b22b = (A22 * BR22[cols]).sum(); a = (AR22.T * BR22).sum()
# sparse formulas from sgmlib
Rs = P.copy(); Rs[rows, cols] -= 1
sb21 = sB21.multiply((sA21.T @ Rs).T).sum(); sb12 = sB12.T.multiply((sA12 @ Rs).T).sum()
sAR22 = sA22.T.tocsr() @ Rs; sBR22 = sB22 @ Rs.T
sb22a = sB22.T.tocsr()[cols].multiply(sAR22).sum(); sb22b = sA22.multiply(sBR22[cols]).sum(); sa = (sAR22.T * sBR22).sum()
for nm, x, y in [("b21", b21, sb21), ("b12", b12, sb12), ("b22a", b22a, sb22a), ("b22b", b22b, sb22b), ("a", a, sa)]:
    print(nm, x, y, np.isclose(x, y))
