"""Wider score coverage on fresh_pair_scores' saved graph pairs.

Exact top-ten fixed-size sets for mean total vertex degree and mean harmonic
centrality; multi-start swap search for cut degree, pair mass, uniqueness,
coverage and the frozen score; intermediate sets sampled across the score range.
Only the two additive objectives have guaranteed global top-ten rankings.

Run: python -m selection_rule_comparison.fresh_pair_extremes --jobs 3
Results: metric_sets.csv and metric_selection.json beside each original pair.
"""

from __future__ import annotations

import argparse
import heapq
import json

import numpy as np
import pandas as pd

from . import fresh_pair_scores as F
from . import jobs

GOALS = ("degree", "pair_mass", "uniqueness", "coverage", "score")
RULE_FOR = {"degree": "top_degree", "pair_mass": "clustered", "uniqueness": "distinguishing",
            "coverage": "spread", "score": "score_tail"}
DETAIL_FOR = {"degree": "high_cut_degree", **{f: f"high_{f}" for f in GOALS if f != "degree"}}
SEARCH_SEED_BASE = 20261125


def top_additive_sets(values: np.ndarray, k: int, count: int = 10) -> list[np.ndarray]:
    """Exact k-best combinations of fixed vertex weights, with ties allowed.

    Sort once. A best-first heap explores combinations obtained by advancing one
    sorted index. Every edge in this search graph weakly decreases the objective,
    so each popped combination is globally next-best; no enumeration of C(n,k).
    """
    values = np.asarray(values, dtype=float)
    order = np.lexsort((np.arange(len(values)), -values))
    sorted_values = values[order]
    first = tuple(range(k))
    heap, seen = [(-float(sorted_values[:k].sum()), first)], {first}
    result = []
    while heap and len(result) < count:
        _, indices = heapq.heappop(heap)
        result.append(np.sort(order[list(indices)]))
        for position in range(k):
            limit = indices[position + 1] if position + 1 < k else len(values)
            if indices[position] + 1 >= limit:
                continue
            child = (*indices[:position], indices[position] + 1, *indices[position + 1:])
            if child not in seen:
                seen.add(child)
                heapq.heappush(heap, (-float(sorted_values[list(child)].sum()), child))
    return result


def unique_sets(sets) -> list[np.ndarray]:
    found = {}
    for seeds in sets:
        key = tuple(sorted(int(v) for v in seeds))
        found.setdefault(key, np.asarray(key, dtype=np.int64))
    return list(found.values())


def feature_calculator(graph, algorithm: str, info: dict):
    calculator = F.MP.PoolFeatures(graph, F.MP.UNIQUENESS_CAP[algorithm])
    k = int(info["budget"])

    def features(sets):
        sets = np.atleast_2d(np.asarray(sets, dtype=np.int64))
        raw = calculator.compute(sets)
        raw["total_degree"] = graph.degree[sets].mean(axis=1)
        coverage = []
        for start in range(0, len(sets), 128):
            block = sets[start:start + 128]
            covered = graph.adjacency[:, block].sum(axis=2) > 0
            covered[block.T, np.arange(len(block))[None, :]] = False
            coverage.extend(covered.sum(axis=0) / (graph.n - k))
        raw["coverage"] = np.asarray(coverage)
        for f in F.CORE:
            raw[f"z_{f}"] = ((raw[f] - info["feature_mean"][f]) / info["feature_sd"][f]
                             if info["feature_sd"][f] > 0 else np.zeros(len(sets)))
        raw["score"] = sum(info["weights_per_sd"][f] * raw[f"z_{f}"] for f in F.CORE)
        return raw
    return features


def search_sets(features, goal: str, known, known_values, starts, n: int, rng,
                top_n: int, steps: int, swaps: int, candidates=None):
    """Ten best distinct sets encountered by sampled single-swap hill climbing.

    All vertices can enter the set. These are high-value candidates, without a
    global-optimality guarantee or a guarantee of exhaustive local optimality.
    """
    best = {}
    eligible = np.arange(n) if candidates is None else np.asarray(candidates, dtype=np.int64)

    def remember(sets, values):
        for j in np.argsort(-values, kind="stable")[:top_n]:
            best[tuple(int(v) for v in sets[j])] = float(values[j])
        ranked = sorted(best, key=lambda key: (-best[key], key))
        for key in ranked[top_n:]:
            del best[key]

    known = np.stack(known)
    remember(known, known_values)
    if goal not in ("uniqueness", "coverage") or min(best.values()) < 1.0:
        for start in unique_sets(starts):
            current = np.sort(start)
            value = float(features(current[None, :])[goal][0])
            for _ in range(steps):
                outside = np.setdiff1d(eligible, current)
                flat = rng.choice(len(current) * len(outside),
                                  size=min(swaps, len(current) * len(outside)), replace=False)
                position, addition = np.divmod(flat, len(outside))
                candidates = np.repeat(current[None, :], len(flat), axis=0)
                candidates[np.arange(len(flat)), position] = outside[addition]
                candidates.sort(axis=1)
                values = features(candidates)[goal]
                remember(candidates, values)
                j = int(np.argmax(values))
                if values[j] <= value + 1e-12:
                    break
                current, value = candidates[j], float(values[j])
    return [np.asarray(key, dtype=np.int64) for key in sorted(best, key=lambda key: (-best[key], key))]


def fast_objective(graph, features, goal: str, k: int):
    """Avoid computing distance fingerprints when searching a neighbourhood-only objective."""
    if goal not in ("degree", "pair_mass", "coverage"):
        return features

    def calculate(sets):
        sets = np.atleast_2d(np.asarray(sets, dtype=np.int64))
        if goal == "degree":
            internal = graph.adjacency[sets[:, :, None], sets[:, None, :]].sum(axis=(1, 2))
            return {goal: (graph.degree[sets].sum(axis=1) - internal) / k}
        values = []
        for start in range(0, len(sets), 128):
            block = sets[start:start + 128]
            w = graph.adjacency[:, block].sum(axis=2, dtype=np.int64)
            w[block.T, np.arange(len(block))[None, :]] = 0
            value = ((w > 0).sum(axis=0) if goal == "coverage" else (w * (w - 1) // 2).sum(axis=0))
            values.extend(value / (graph.n - k))
        return {goal: np.asarray(values)}
    return calculate


def metric_plan(graph, algorithm: str, info: dict, old: pd.DataFrame, random: pd.DataFrame, design: dict,
                candidates=None, fast_objectives=False):
    """Fixed-k metric extremes and score-stratified intermediates, selected on A only."""
    k, seed, top_n = int(info["budget"]), int(info["pair_seed"]), design["top_n"]
    features = feature_calculator(graph, algorithm, info)
    eligible = np.arange(graph.n) if candidates is None else np.asarray(candidates, dtype=np.int64)
    degree = [np.sort(eligible[s]) for s in top_additive_sets(graph.degree[eligible], k, top_n)]
    harmonic = [np.sort(eligible[s]) for s in top_additive_sets(graph.harmonic[eligible], k, top_n)]
    reference = [np.asarray(json.loads(s), dtype=np.int64) for s in random.seed_vertices.head(3)]
    old_sets = [np.asarray(json.loads(s), dtype=np.int64) for s in old.seed_vertices]
    baseline = [np.asarray(json.loads(s), dtype=np.int64) for s in random.seed_vertices]
    known = unique_sets([*degree, *harmonic, *old_sets, *baseline])
    known_values = features(np.stack(known))
    groups = [("global_degree", degree), ("global_harmonic", harmonic)]
    for index, goal in enumerate(GOALS):
        target = old[old.detail == RULE_FOR[goal]]
        anchor = (np.asarray(json.loads(target.iloc[0].seed_vertices), dtype=np.int64)
                  if len(target) else degree[0])
        starts = [known[int(np.argmax(known_values[goal]))], anchor, reference[0]]
        rng = np.random.default_rng([SEARCH_SEED_BASE, seed, F.MP.ALGORITHM_CODES[algorithm], index])
        scorer = fast_objective(graph, features, goal, k) if fast_objectives else features
        selected = search_sets(scorer, goal, known, known_values[goal], starts, graph.n, rng, top_n,
                               design["steps"], design["swaps"], candidates=eligible)
        groups.append((DETAIL_FOR[goal], selected))

    arrays, descriptions = [], []
    for detail, sets in groups:
        for rank, seeds in enumerate(sets, 1):
            arrays.append(seeds)
            descriptions.append({"group": "global" if detail.startswith("global_") else "search",
                                 "detail": detail, "selection_rank": rank, "pool_index": -1})

    # All intermediate swaps from one random set towards each metric's best set.
    # Sample across the resulting score range, rather than collecting only extrema.
    paths = []
    rng = np.random.default_rng([SEARCH_SEED_BASE, seed, F.MP.ALGORITHM_CODES[algorithm], 100])
    start = reference[0]
    for _, target_sets in groups:
        target = target_sets[0]
        remove = rng.permutation(np.setdiff1d(start, target))
        add = rng.permutation(np.setdiff1d(target, start))
        current = set(int(v) for v in start)
        for gone, entered in list(zip(remove, add))[:-1]:
            current.remove(int(gone)); current.add(int(entered))
            paths.append(np.asarray(sorted(current), dtype=np.int64))
    already = {tuple(s) for s in arrays}
    paths = [s for s in unique_sets(paths) if tuple(s) not in already]
    if paths:
        scores = features(np.stack(paths))["score"]
        edges = np.linspace(scores.min(), scores.max() + 1e-12, 11)
        bins = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, 9)
        for b in range(10):
            members = np.flatnonzero(bins == b)
            centre = (edges[b] + edges[b + 1]) / 2
            chosen = members[np.argsort(np.abs(scores[members] - centre), kind="stable")[:2]]
            for j in chosen:
                arrays.append(paths[j])
                descriptions.append({"group": "bridge", "detail": "score_bridge",
                                     "selection_rank": b + 1, "pool_index": -1})
    raw = features(np.stack(arrays))
    table = pd.DataFrame(descriptions)
    table.insert(0, "candidate_id", np.arange(len(table)))
    table["seed_vertices"] = [json.dumps(s.tolist()) for s in arrays]
    for f in (*F.CORE, *(f"z_{f}" for f in F.CORE), "total_degree", "coverage", "score"):
        table[f] = raw[f]
    return table, arrays


def plot_sets(collected: pd.DataFrame, top_n=10) -> pd.DataFrame:
    """Rank all collected candidates for each non-additive metric before plotting.

    One search can find a better set for another search's objective. Pool their
    evaluated candidates, breaking objective ties by frozen score. Retain the
    generating family in source_detail and preserve the exact additive groups.
    """
    unique = collected.drop_duplicates("seed_vertices").copy()
    groups = [collected[collected.group == "global"].copy()]
    for goal in GOALS:
        columns = [goal] if goal == "score" else [goal, "score"]
        chosen = unique.sort_values(columns, ascending=False, kind="stable").head(top_n).copy()
        chosen["source_detail"] = chosen["detail"]
        chosen["detail"], chosen["group"] = DETAIL_FOR[goal], "search"
        chosen["selection_rank"] = np.arange(1, len(chosen) + 1)
        groups.append(chosen)
    groups.append(collected[collected.group == "bridge"].copy())
    out = pd.concat(groups, ignore_index=True)
    out["source_candidate_id"] = out["candidate_id"]
    out["candidate_id"] = np.arange(len(out))
    return out


def run_all(algorithms=F.ALGORITHMS, models=F.MODELS, n_pairs=3, jobs=3, blas_threads=3):
    design = {"kind": "metrics", "top_n": 10, "steps": 8, "swaps": 256,
              "blas_threads": int(blas_threads), "search_seed_base": SEARCH_SEED_BASE}
    failures = jobs.run_tasks(F.tasks(algorithms, models, n_pairs), F.selected_job, (design,), F.describe,
                              lambda t: 90 * F.SECONDS_PER_SET[t[0]], lambda _: [], jobs)
    if failures:
        raise RuntimeError(f"Failed metric-set pairs: {', '.join(failures)}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="+", choices=F.ALGORITHMS, default=F.ALGORITHMS)
    parser.add_argument("--models", nargs="+", choices=F.MODELS, default=F.MODELS)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--blas-threads", type=int, default=3)
    args = parser.parse_args()
    run_all(args.algorithms, args.models, args.pairs, args.jobs, args.blas_threads)
    F.plot_scores()


if __name__ == "__main__":
    main()
