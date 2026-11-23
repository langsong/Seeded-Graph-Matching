"""Separate single-B, wider-score plots for landscape.ipynb section 3.

Use pair 0's original A/B, pool normalization, budget and 120 held-out test
vertices in each standard cell. Reuse first-run outcomes and label-free
constructed sets already evaluated on that B; collect missing metric extremes
and score-stratified intermediate sets. The original landscape data stay intact.

Run: python -m selection_rule_comparison.landscape_extremes --jobs 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_common.provenance import thread_limit
from . import fresh_pair_extremes as E
from . import fresh_pair_scores as F
from . import graded_outcomes as G
from . import jobs
from . import landscape as L
from . import regime_map as RM
from . import run_comparison as rc

OUTPUT_DIR = L.EXPERIMENT_DIR / "data" / "landscape_extremes"
SOLVER_SEED_BASE = 20261127
PAIR_INDEX = 0


def tasks(models=L.MODELS):
    out = []
    for model in models:
        for algorithm, degree, rho in L.cells_for(model):
            budget = L.cell_budget(model, algorithm, degree, rho)["budget"]
            if budget is None:
                raise RuntimeError(f"No saved landscape budget for {model}/{algorithm}/{degree}/{rho}.")
            out.append((model, algorithm, degree, rho, int(budget)))
    return out


def folder_for(task):
    model, algorithm, degree, rho, budget = task
    return OUTPUT_DIR / model / RM.cell_name(degree, rho) / algorithm / f"k{budget:03d}" / "pair_000"


def describe(task):
    model, algorithm, degree, rho, budget = task
    return f"{L.MODEL_LABELS[model]} {L.LABELS[algorithm]} d{degree:g} rho{rho:g}, k={budget}"


def saved_candidates(task):
    """Original first-run results and A-only constructed sets on the same B.

    Adaptive selector proposals are excluded because their search used labels
    from this pair. Constructed tilts, rules and swap paths were chosen on A only.
    """
    model, algorithm, degree, rho, budget = task
    source = L.pair_dir(model, algorithm, degree, rho, budget, PAIR_INDEX)
    info = L._read_json(source / "pair.json")
    sets = pd.read_csv(source / "sets.csv")
    runs = pd.read_csv(source / "runs.csv")
    runs = runs[(runs["repeat"] == 0) & L._ok(runs)]
    columns = ["set_id", "solver_seed", *L.OUTCOME_COLUMNS]
    base = sets.merge(runs[columns], on="set_id", how="inner").assign(source="landscape first run")
    prior = base.copy()
    constructed = (L.EXPERIMENT_DIR / "data" / "learned_selector" / model / RM.cell_name(degree, rho)
                   / algorithm / f"k{budget:03d}" / "pair_000")
    if (constructed / "constructed_sets.csv").exists():
        saved = L._read_json(constructed / "pair.json")
        for key in ("hash_a", "hash_b", "test_vertices"):
            if saved[key] != info[key]:
                raise RuntimeError(f"{describe(task)}: constructed sets use a different graph pair.")
        extra = pd.read_csv(constructed / "constructed_sets.csv")
        extra_runs = pd.read_csv(constructed / "constructed_runs.csv")
        extra_runs = extra_runs[(extra_runs["draw"] == 0) & L._ok(extra_runs)]
        extra = extra.merge(extra_runs[columns], on="set_id", how="inner")
        prior = pd.concat([base, extra.assign(source="constructed saved B")], ignore_index=True)
    # The score search can start from the strongest frozen-score candidate already available.
    anchor = prior.loc[[prior["score"].idxmax()]].copy()
    anchor["detail"] = "score_tail"
    return info, base, pd.concat([prior, anchor], ignore_index=True)


def job(task, design):
    model, algorithm, degree, rho, budget = task
    folder = folder_for(task)
    path, meta_path = folder / "metric_sets.csv", folder / "selection.json"
    if meta_path.exists():
        saved = L._read_json(meta_path)
        if saved["design"] != design:
            raise ValueError(f"{describe(task)}: saved extension design differs.")
        if len(G._completed_ids(path, "run_id")) == saved["n_sets"]:
            return f"{describe(task)}: already done"
    with thread_limit(design["blas_threads"]):
        info, base, prior = saved_candidates(task)
        seed = int(info["pair_seed"])
        A, B, truth = F.generate_graph_pair(RM.graph_config(model, degree, rho), seed)
        if rc._hash(A) != info["hash_a"] or rc._hash(B) != info["hash_b"]:
            raise RuntimeError(f"{describe(task)}: regenerated graph pair differs.")
        test = np.asarray(info["test_vertices"], dtype=np.int64)
        eligible = np.setdiff1d(np.arange(len(A)), test)
        weights = L._read_json(L.OUTPUT_DIR / model / "settings.json")["weights"][algorithm]
        score_info = {"pair_seed": seed, "budget": budget, "weights_per_sd": weights,
                      "feature_mean": {f: info["pool_stats"][f]["mean"] for f in L.CORE},
                      "feature_sd": {f: info["pool_stats"][f]["sd"] for f in L.CORE}}
        random = base[base["design"] == "random"]
        graph = F.GraphA(A)
        plan, arrays = E.metric_plan(graph, algorithm, score_info, prior, random, design,
                                     candidates=eligible, fast_objectives=True)
        if any(np.intersect1d(s, test).size for s in arrays):
            raise RuntimeError(f"{describe(task)}: a constructed set contains a test vertex.")
        metadata = {"design": design, "n_sets": len(plan), "pair_seed": seed, "budget": budget,
                    "hash_a": info["hash_a"], "hash_b": info["hash_b"], "test_vertices": test.tolist(),
                    "score_reference": score_info,
                    "plan_hash": hashlib.sha256(plan.to_csv(index=False).encode()).hexdigest()}
        if meta_path.exists() and L._read_json(meta_path) != metadata:
            raise ValueError(f"{describe(task)}: regenerated extension plan differs.")
        folder.mkdir(parents=True, exist_ok=True)
        L._write_json(meta_path, metadata)
        done = G._completed_ids(path, "run_id")
        saved_rows = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
        cache = {}
        for row in pd.concat([base, prior, saved_rows], ignore_index=True).to_dict("records"):
            key = tuple(json.loads(row["seed_vertices"]))
            cache.setdefault(key, {"out": {c: row[c] for c in ("solver_seed", *L.OUTCOME_COLUMNS)},
                                   "source": row.get("source", "saved extension")})
        rows, written, new = G._Rows(path), 0, 0
        try:
            for j, seeds in enumerate(arrays):
                if j in done:
                    continue
                key = tuple(int(v) for v in seeds)
                cached = key in cache
                if cached:
                    out, source = dict(cache[key]["out"]), cache[key]["source"]
                    out["error"] = ""
                else:
                    solver_seed = rc._seed(SOLVER_SEED_BASE, seed, budget, F.MP.ALGORITHM_CODES[algorithm], j)
                    out = RM.run_one(algorithm, model, (A, B, truth), seeds, test, solver_seed)
                    if out["error"]:
                        raise RuntimeError(f"{describe(task)}: {out['error']}")
                    out["solver_seed"] = solver_seed
                    source = "new single-B run"
                    cache[key] = {"out": out, "source": source}
                    new += 1
                rows.write({"model": model, "algorithm": algorithm, "mean_degree": degree, "rho": rho,
                            "level": L.LEVEL, "budget": budget, "pair_index": PAIR_INDEX, "pair_seed": seed,
                            **plan.iloc[j].to_dict(), "run_id": j, **out, "cached": cached, "source": source})
                written += 1
        finally:
            rows.close()
    return f"{describe(task)}: {written} rows, {new} new matching calls"


def run_all(models=L.MODELS, jobs=3, blas_threads=3):
    for model in models:
        L.prepare(model, blas_threads=blas_threads)  # validate the original graph/solver design
    design = {"version": 1, "top_n": 10, "steps": 8, "swaps": 256, "pair_index": PAIR_INDEX,
              "blas_threads": int(blas_threads), "solver_seed_base": SOLVER_SEED_BASE,
              "search_seed_base": E.SEARCH_SEED_BASE, "draw": "original saved B only"}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "settings.json"
    if path.exists() and L._read_json(path) != design:
        raise ValueError(f"The design in {path} differs from this run.")
    L._write_json(path, design)
    failures = jobs.run_tasks(tasks(models), job, (design,), describe,
                              lambda t: 90 * F.SECONDS_PER_SET[t[1]], lambda _: [], jobs)
    if failures:
        raise RuntimeError(f"Failed landscape extension cells: {', '.join(failures)}")
    return True


def read():
    frames = [E.plot_sets(pd.read_csv(p)) for p in sorted(OUTPUT_DIR.glob("*/*/*/k*/pair_000/metric_sets.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_single_b(sets, selected, model, save=True):
    """One original B and first solver run per set, alongside A-only metric constructions."""
    import matplotlib.pyplot as plt

    cells = L._cells(sets, model)
    algorithms, per_row, columns = L._grid(cells)
    with plt.rc_context({"font.size": 9}):
        fig, axes = plt.subplots(len(algorithms), columns,
                                 figsize=(4.1 * columns, 3.25 * len(algorithms) + 1.2), sharey=True, squeeze=False)
        for r, row_key in enumerate(algorithms):
            for c in range(columns):
                ax = axes[r, c]
                if c >= len(per_row[row_key]):
                    ax.axis("off")
                    continue
                a, d, rho, level, budget = per_row[row_key][c]
                cell = {"model": model, "algorithm": a, "mean_degree": d, "rho": rho,
                        "level": level, "budget": budget}
                part = sets.merge(pd.DataFrame([cell]), on=L.CELL_KEYS)
                random = part[(part["pair_index"] == PAIR_INDEX) & (part["design"] == "random")]
                L._style(ax)
                ax.scatter(random["score"], random["accuracy_first"], s=8, color=L.CONTEXT,
                           alpha=0.5, linewidths=0, label="350 random sets, original B")
                if not selected.empty:
                    extra = selected.merge(pd.DataFrame([cell]), on=L.CELL_KEYS)
                    F.scatter_selected(ax, extra.assign(accuracy=extra["accuracy_test"]))
                ax.set_title(L.cell_title(a, d, rho, budget), loc="left", fontsize=9.5)
                ax.set_ylim(-0.03, 1.03)
                if c == 0:
                    ax.set_ylabel("accuracy on 120 fixed test vertices")
                if r == len(algorithms) - 1:
                    ax.set_xlabel("Frozen weighted score")
        handles, labels = [], []
        for ax in axes.ravel():
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle); labels.append(label)
        labels = [label.replace("exact top 10", "exact eligible top 10") for label in labels]
        fig.suptitle(f"{L.MODEL_LABELS[model]}: wider seed-set score range on one saved B per cell\n"
                     "Pair 0; one solver run per set; original normalization; test vertices excluded from seeds",
                     x=0.01, y=0.995, ha="left", va="top", fontsize=11)
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
        fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    if save:
        directory = OUTPUT_DIR / "figures"
        directory.mkdir(exist_ok=True, parents=True)
        for suffix in ("png", "pdf"):
            fig.savefig(directory / f"{model}_single_b.{suffix}", dpi=180, bbox_inches="tight")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=L.MODELS, default=L.MODELS)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    run_all(args.models, args.jobs)
    selected, sets = read(), L.per_set()
    import matplotlib.pyplot as plt
    for model in args.models:
        plot_single_b(sets, selected, model)
        plt.close()


if __name__ == "__main__":
    main()
