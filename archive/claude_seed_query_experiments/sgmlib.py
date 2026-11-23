"""Self-contained seeded graph matching (SGM) toolkit for seed-query experiments.

graspologic could not be installed in this environment (package index blocked),
so this module re-implements the FAQ-based SGM solver used by
``graspologic.match.graph_match`` / ``scipy.optimize.quadratic_assignment``
(method="faq", barycenter start, Frank-Wolfe steps with exact line search,
final projection by linear assignment).  Solver randomness mimics graspologic's
``shuffle_input``: the unseeded B vertices are shuffled and every linear
assignment call sees randomly permuted rows, which only changes tie breaking.

The implementation uses sparse matrix products, so one n=600 run costs about as
much as the linear assignment problems it solves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import shortest_path

# ---------------------------------------------------------------------------
# Graph pairs
# ---------------------------------------------------------------------------


@dataclass
class Pair:
    name: str
    A: sp.csr_matrix          # graph A (symmetric 0/1)
    B: sp.csr_matrix          # graph B (symmetric 0/1, relabelled)
    truth: np.ndarray         # truth[a] = matching vertex of a in B
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.A.shape[0]


def _to_csr(M) -> sp.csr_matrix:
    M = sp.csr_matrix(M, dtype=np.float64)
    M.eliminate_zeros()
    return M


def load_npz_pair(path: str, name: str) -> Pair:
    d = np.load(path)
    return Pair(name, _to_csr(d["adjacency_graph1"]), _to_csr(d["adjacency_graph2"]),
                np.asarray(d["true_permutation"], dtype=int))


def sample_corr_pair(P: np.ndarray, rho: float, rng: np.random.Generator, name: str,
                     meta: dict | None = None) -> Pair:
    """Correlated Bernoulli pair with marginal edge probabilities P and edge correlation rho.

    Same construction as graspologic.simulations.sample_edges_corr:
    G1 ~ Bern(P); G2 ~ Bern(P + rho(1-P)) where G1 = 1 and Bern(P(1-rho)) elsewhere.
    G2 is then randomly relabelled.
    """
    n = P.shape[0]
    iu = np.triu_indices(n, k=1)
    p = np.clip(P[iu], 0.0, 1.0)
    g1 = rng.random(p.shape[0]) < p
    p2 = np.where(g1, p + rho * (1.0 - p), p * (1.0 - rho))
    g2 = rng.random(p.shape[0]) < p2
    A = sp.coo_matrix((np.ones(g1.sum()), (iu[0][g1], iu[1][g1])), shape=(n, n))
    A = A + A.T
    B0 = sp.coo_matrix((np.ones(g2.sum()), (iu[0][g2], iu[1][g2])), shape=(n, n))
    B0 = (B0 + B0.T).tocsr()
    shuffle = rng.permutation(n)
    B = B0[shuffle][:, shuffle]
    truth = np.argsort(shuffle)
    return Pair(name, _to_csr(A), _to_csr(B), truth, meta or {})


def gen_pair(model: str, rng: np.random.Generator, **kw) -> Pair:
    """Generators matching the presets in cross_pair_seed_set_quality.preset_config."""
    if model == "sparse_er":
        n, p, rho = kw.get("n", 600), kw.get("p", 0.01), kw.get("rho", 0.80)
        P = np.full((n, n), p)
    elif model == "sparse_sbm":
        n_per, rho = kw.get("n_per_block", 200), kw.get("rho", 0.80)
        Bp = np.asarray(kw.get("block_probs", [[0.02, 0.005, 0.01], [0.005, 0.02, 0.005], [0.01, 0.005, 0.02]]))
        labels = np.repeat(np.arange(Bp.shape[0]), n_per)
        P = Bp[labels][:, labels]
    elif model == "dense_sbm":   # the cross-pair "sbm" regime: n=300, p_in=.3, p_out=.1, rho=.36
        n_per, rho = kw.get("n_per_block", 100), kw.get("rho", 0.36)
        pin, pout = kw.get("p_in", 0.3), kw.get("p_out", 0.1)
        labels = np.repeat(np.arange(3), n_per)
        P = np.where(labels[:, None] == labels[None, :], pin, pout).astype(float)
    elif model == "paper":
        n, alpha, p, rho = kw.get("n", 600), kw.get("alpha", 2.0), kw.get("p", 0.005), kw.get("rho", 0.70)
        d = rng.pareto(alpha, size=n) + 2
        s, m = d.sum(), d.max()
        if m * m / s > 1.0:
            d = d * (s / m ** 2) * 0.95
        P = np.outer(d, d) / d.sum() + p
    elif model == "ier":
        n, a, b, rho = kw.get("n", 600), kw.get("a", 1.0), kw.get("b", 99.0), kw.get("rho", 0.77)
        P = np.zeros((n, n))
        iu = np.triu_indices(n, k=1)
        P[iu] = rng.beta(a, b, size=iu[0].shape[0])
        P = P + P.T
    else:
        raise ValueError(model)
    np.fill_diagonal(P, 0.0)
    return sample_corr_pair(P, rho, rng, model, {"model": model, **kw})


# ---------------------------------------------------------------------------
# SGM (FAQ) solver
# ---------------------------------------------------------------------------


def _lsa(M: np.ndarray, rng: np.random.Generator | None, maximize: bool = True) -> np.ndarray:
    """Linear assignment; with rng, rows are randomly permuted first (tie breaking only)."""
    if rng is None:
        return linear_sum_assignment(M, maximize=maximize)[1]
    perm = rng.permutation(M.shape[0])
    cols = linear_sum_assignment(M[perm], maximize=maximize)[1]
    out = np.empty_like(cols)
    out[perm] = cols
    return out


@dataclass
class SGMResult:
    perm: np.ndarray                 # perm[a] = matched B vertex
    n_iter: int
    converged: bool
    score: float                     # number of edge agreements * 2 (trace A P B P^T)
    h0_acc: float | None = None      # accuracy of the first assignment direction (unseeded)
    acc: float | None = None         # final accuracy on unseeded A vertices
    traj: list | None = None         # accuracy of the assignment direction at each iteration
    checkpoints: dict | None = None  # {iteration: projected accuracy} for requested iterations
    P: np.ndarray | None = None      # final doubly-stochastic iterate (unseeded block)
    grad0: np.ndarray | None = None  # first gradient (unseeded block, rows=A order, cols=B order)
    order_A: np.ndarray | None = None
    order_B: np.ndarray | None = None


def sgm(pair: Pair, seeds_A, rng=None, max_iter: int = 30, tol: float = 0.01,
        record_traj: bool = False, checkpoints=(), keep_P: bool = False,
        keep_grad0: bool = False, seeds_B=None) -> SGMResult:
    """Seeded graph matching of pair.A to pair.B with seeds_A matched to their true partners.

    rng: None -> deterministic (identical to scipy's FAQ without shuffling);
         int/Generator -> graspologic-like random tie breaking.
    """
    A, B, truth = pair.A, pair.B, pair.truth
    n = A.shape[0]
    seeds_A = np.asarray(seeds_A, dtype=int)
    seeds_B = truth[seeds_A] if seeds_B is None else np.asarray(seeds_B, dtype=int)
    s = len(seeds_A)
    if rng is not None and not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    non_A = np.setdiff1d(np.arange(n), seeds_A)
    non_B = np.setdiff1d(np.arange(n), seeds_B)
    if rng is not None:
        non_B = rng.permutation(non_B)
    perm_A = np.concatenate([seeds_A, non_A])
    perm_B = np.concatenate([seeds_B, non_B])
    u = n - s

    Ap = A[perm_A][:, perm_A].tocsr()
    Bp = B[perm_B][:, perm_B].tocsr()
    A12, A21, A22 = Ap[:s, s:].tocsr(), Ap[s:, :s].tocsr(), Ap[s:, s:].tocsr()
    B12, B21, B22 = Bp[:s, s:].tocsr(), Bp[s:, :s].tocsr(), Bp[s:, s:].tocsr()
    A22T, B22T = A22.T.tocsr(), B22.T.tocsr()
    B21T_csr = B21.T.tocsr()
    const = (A21 @ B21T_csr + A12.T @ B12)
    const = const.toarray() if sp.issparse(const) else np.asarray(const)

    truth_rows = None
    if truth is not None:
        # target column (in the unseeded B block) of every unseeded A row
        pos_B = np.empty(n, dtype=int)
        pos_B[perm_B] = np.arange(n)
        truth_rows = pos_B[truth[perm_A[s:]]] - s   # may be negative if truth maps to a seed (wrong seed)

    def acc_of(cols):
        if truth_rows is None:
            return None
        return float(np.mean(cols == truth_rows))

    P = np.full((u, u), 1.0 / u)
    traj = [] if record_traj else None
    chk = {}
    h0 = None
    grad0 = None
    converged = False
    rows = np.arange(u)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        X = A22 @ P                                   # u x u
        grad = const + (B22 @ X.T).T + (B22T @ (A22T @ P).T).T
        cols = _lsa(grad, rng)
        if n_iter == 1:
            h0 = acc_of(cols)
            if keep_grad0:
                grad0 = grad.copy()
        if record_traj:
            traj.append(acc_of(cols))
        # exact line search on f(alpha * P + (1 - alpha) * Q)
        R = P.copy()
        R[rows, cols] -= 1.0
        b21 = B21.multiply((A21.T @ R).T).sum()
        b12 = B12.T.multiply((A12 @ R).T).sum()
        AR22 = A22T @ R
        BR22 = (B22 @ R.T)
        b22a = B22T[cols].multiply(AR22).sum()
        b22b = A22.multiply(BR22[cols]).sum()
        a = float((AR22.T * BR22).sum())
        b = float(b21 + b12 + b22a + b22b)
        if a < 0 and 0 <= -b / (2 * a) <= 1:
            alpha = -b / (2 * a)
        else:
            alpha = float(np.argmin([0, -(b + a)]))
        P_new = alpha * P
        P_new[rows, cols] += (1.0 - alpha)
        step = np.linalg.norm(P - P_new) / math.sqrt(u)
        P = P_new
        if n_iter in checkpoints:
            chk[n_iter] = acc_of(_lsa(P, rng))
        if step < tol:
            converged = True
            break
    final_cols = _lsa(P, rng)
    perm = np.empty(n, dtype=int)
    perm[perm_A[:s]] = perm_B[:s]
    perm[perm_A[s:]] = perm_B[s + final_cols]
    score = float(A.multiply(B[perm][:, perm]).sum())
    acc = acc_of(final_cols)
    # checkpoints after convergence equal the final answer
    for c in checkpoints:
        if c not in chk and c >= n_iter:
            chk[c] = acc
    return SGMResult(perm=perm, n_iter=n_iter, converged=converged, score=score, h0_acc=h0,
                     acc=acc, traj=traj, checkpoints=chk or None,
                     P=P if keep_P else None, grad0=grad0,
                     order_A=perm_A if (keep_P or keep_grad0) else None,
                     order_B=perm_B if (keep_P or keep_grad0) else None)


def run_restarts(pair: Pair, seeds_A, R: int = 4, base_seed: int = 0, success: float = 0.9,
                 **kw) -> dict:
    """R independent solver runs (graspologic-like tie-break noise). Returns summary + runs."""
    runs = [sgm(pair, seeds_A, rng=np.random.default_rng([base_seed, r]), **kw) for r in range(R)]
    accs = np.array([r.acc for r in runs])
    scores = np.array([r.score for r in runs])
    best = runs[int(np.argmax(scores))]
    return {
        "mean_acc": float(accs.mean()),
        "p_success": float(np.mean(accs >= success)),
        "best_obj_acc": float(best.acc),
        "best_obj_success": float(best.acc >= success),
        "mean_h0": float(np.mean([r.h0_acc for r in runs])),
        "mean_iter": float(np.mean([r.n_iter for r in runs])),
        "frac_converged": float(np.mean([r.converged for r in runs])),
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# Graph-A structure and selection rules
# ---------------------------------------------------------------------------


@dataclass
class GStats:
    A: sp.csr_matrix
    deg: np.ndarray
    dist: np.ndarray                 # all-pairs hop distances (inf if disconnected)
    _cache: dict = field(default_factory=dict)

    def reach(self, r: int) -> np.ndarray:
        key = ("reach", r)
        if key not in self._cache:
            self._cache[key] = ((self.dist <= r).sum(axis=1) - 1).astype(float)
        return self._cache[key]

    def ball(self, r: int) -> np.ndarray:
        key = ("ball", r)
        if key not in self._cache:
            M = (self.dist <= r)
            np.fill_diagonal(M, False)
            self._cache[key] = M
        return self._cache[key]

    def pagerank(self) -> np.ndarray:
        if "pr" not in self._cache:
            import networkx as nx
            G = nx.from_scipy_sparse_array(self.A)
            pr = nx.pagerank(G, max_iter=500)
            self._cache["pr"] = np.array([pr[i] for i in range(self.A.shape[0])])
        return self._cache["pr"]

    def betweenness(self) -> np.ndarray:
        if "btw" not in self._cache:
            import networkx as nx
            G = nx.from_scipy_sparse_array(self.A)
            bt = nx.betweenness_centrality(G)
            self._cache["btw"] = np.array([bt[i] for i in range(self.A.shape[0])])
        return self._cache["btw"]

    def communities(self, seed: int = 0) -> np.ndarray:
        if "comm" not in self._cache:
            import networkx as nx
            G = nx.from_scipy_sparse_array(self.A)
            comms = nx.community.louvain_communities(G, seed=seed)
            lab = np.empty(self.A.shape[0], dtype=int)
            for i, c in enumerate(comms):
                lab[list(c)] = i
            self._cache["comm"] = lab
        return self._cache["comm"]


def gstats(A: sp.csr_matrix) -> GStats:
    dist = shortest_path(A, method="D", unweighted=True, directed=False)
    return GStats(A=A, deg=np.asarray(A.sum(axis=1)).ravel(), dist=dist)


def _top(score: np.ndarray, q: int, exclude, rng) -> list[int]:
    score = score.astype(float).copy()
    finite = np.isfinite(score)
    scale = (np.abs(score[finite]).max() if finite.any() else 0.0) + 1.0
    score[finite] = score[finite] + rng.random(int(finite.sum())) * 1e-9 * scale
    score[list(exclude)] = -np.inf
    return [int(v) for v in np.argsort(-score)[:q]]


def greedy_capped_coverage(G: GStats, q: int, D: int, c: int, S0, rng) -> list[int]:
    """Greedy maximisation of sum_v min(c, #seeds within D hops of v) (monotone submodular)."""
    M = G.ball(D)                      # M[v, u] = u within D hops of v (u != v)
    n = M.shape[0]
    cnt = M[:, list(S0)].sum(axis=1).astype(float) if len(S0) else np.zeros(n)
    chosen = list(S0)
    tiebreak = G.deg + rng.random(n) * 1e-3
    new = []
    for _ in range(q):
        deficit = (cnt < c).astype(float)          # vertices that still gain from one more witness
        gain = M.T.astype(float) @ deficit          # gain[u] = # vertices v with u in N_D(v) and cnt[v] < c
        gain = gain + 1e-6 * tiebreak
        gain[chosen] = -np.inf
        u = int(np.argmax(gain))
        chosen.append(u)
        new.append(u)
        cnt += M[:, u]
    return new


def dispersion(G: GStats, q: int, S0, rng) -> list[int]:
    """Farthest-first traversal (k-center greedy) over well-connected vertices.

    Candidates are restricted to the largest connected component and degree >= 3, so the rule
    spreads seeds over the core instead of picking isolated vertices or leaves.
    """
    n = G.dist.shape[0]
    reach_all = np.isfinite(G.dist).sum(axis=1)
    lcc = reach_all == reach_all.max()
    eligible = lcc & (G.deg >= 3)
    d = np.where(np.isfinite(G.dist), G.dist, n)
    chosen = list(S0)
    new = []
    if not chosen:
        first = int(rng.choice(np.flatnonzero(eligible)))
        chosen.append(first)
        new.append(first)
    mind = d[:, chosen].min(axis=1)
    while len(new) < q:
        score = mind + rng.random(n) * 1e-3
        score[~eligible] = -np.inf
        score[chosen] = -np.inf
        u = int(np.argmax(score))
        chosen.append(u)
        new.append(u)
        mind = np.minimum(mind, d[:, u])
    return new


def three_hop_mean(G: GStats, S) -> float:
    S = np.asarray(S)
    non = np.setdiff1d(np.arange(G.dist.shape[0]), S)
    return float((G.dist[np.ix_(non, S)] <= 3).sum(axis=1).mean())


def gen_and_rank(G: GStats, q: int, S0, rng, M: int = 50, score="three_hop") -> list[int]:
    n = G.dist.shape[0]
    pool = np.setdiff1d(np.arange(n), list(S0))
    best, best_val = None, -np.inf
    for _ in range(M):
        cand = rng.choice(pool, size=q, replace=False)
        S = np.concatenate([np.asarray(list(S0), dtype=int), cand])
        val = three_hop_mean(G, S)
        if val > best_val:
            best, best_val = cand, val
    return [int(v) for v in best]


def stratified_degree(G: GStats, q: int, S0, rng) -> list[int]:
    lab = G.communities()
    comms, sizes = np.unique(lab, return_counts=True)
    alloc = np.floor(q * sizes / sizes.sum()).astype(int)
    rem = q - alloc.sum()
    order = np.argsort(-(q * sizes / sizes.sum() - alloc))
    alloc[order[:rem]] += 1
    new = []
    excl = set(S0)
    for cmt, k in zip(comms, alloc):
        if k == 0:
            continue
        score = np.where(lab == cmt, G.deg, -np.inf)
        picks = _top(score, k, excl | set(new), rng)
        new.extend(picks)
    return new


def select(rule: str, G: GStats, q: int, rng, S0=()) -> list[int]:
    S0 = list(S0)
    excl = set(S0)
    if q <= 0:
        return []
    if rule == "random":
        pool = np.setdiff1d(np.arange(G.dist.shape[0]), S0)
        return [int(v) for v in rng.choice(pool, size=q, replace=False)]
    if rule == "degree":
        return _top(G.deg, q, excl, rng)
    if rule == "pagerank":
        return _top(G.pagerank(), q, excl, rng)
    if rule == "betweenness":
        return _top(G.betweenness(), q, excl, rng)
    if rule == "reach2":
        return _top(G.reach(2), q, excl, rng)
    if rule == "reach3":
        return _top(G.reach(3), q, excl, rng)
    if rule == "cov1c2":
        return greedy_capped_coverage(G, q, D=1, c=2, S0=S0, rng=rng)
    if rule == "cov2c3":
        return greedy_capped_coverage(G, q, D=2, c=3, S0=S0, rng=rng)
    if rule == "cov2c5":
        return greedy_capped_coverage(G, q, D=2, c=5, S0=S0, rng=rng)
    if rule == "dispersion":
        return dispersion(G, q, S0, rng)
    if rule == "genrank":
        return gen_and_rank(G, q, S0, rng)
    if rule == "strat_degree":
        return stratified_degree(G, q, S0, rng)
    raise ValueError(rule)
