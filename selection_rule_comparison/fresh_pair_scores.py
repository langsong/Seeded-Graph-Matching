"""Dense score-versus-accuracy plots on fresh pairs at the training settings.

Default: 3 pairs x 1,000 uniform random sets x 9 settings = 27,000 single
matching runs. The four-feature pooled logistic weights are frozen. No fitting,
budget calibration or matching restarts take place in the random collection.
An optional selected-set extension adds degree/score tails and round-1 rules.

    python -m selection_rule_comparison.fresh_pair_scores
    python -m selection_rule_comparison.fresh_pair_scores --status
    python -m selection_rule_comparison.fresh_pair_scores --plot-only
    python -m selection_rule_comparison.fresh_pair_scores --compare-models
    python -m selection_rule_comparison.fresh_pair_scores --selected

Rows are flushed after every set; rerunning resumes. Ctrl-C stops the workers.
FRESH_PAIR_SCORES_OUTPUT_DIR can select a separate folder for a small trial.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache"))

import numpy as np
import pandas as pd

from experiment_common.graph_checks import first_passing_pair
from experiment_common.graphs import generate_graph_pair
from experiment_common.provenance import collect_environment, thread_limit, write_json
from fixed_pair_seed_set_classification import multi_pair_collection as MP
from . import pooled_weights as PW
from . import run_comparison as rc
from . import jobs
from . import settings as S
from .strategies import GraphA, STRATEGY_CODES, ordering

OUTPUT_DIR = Path(os.environ.get("FRESH_PAIR_SCORES_OUTPUT_DIR") or
                  Path(__file__).resolve().parent / "data" / "fresh_pair_scores")
ALGORITHMS, MODELS, CORE = MP.ALGORITHMS, MP.MODELS, MP.CORE_FEATURES
N_PAIRS, SETS_PER_PAIR = 3, 1000
PAIR_SEED_BASE, SAMPLING_SEED_BASE, SOLVER_SEED_BASE = 20261110, 20261111, 20261112
SECONDS_PER_SET = {"percolation": 0.5, "goat": 3.4, "faq": 2.5}
SELECTED_POOL, SELECTED_TOP = 20_000, 10
SELECTION_LABELS = {
    "degree_tail": "10 highest-degree pool sets", "score_tail": "10 highest-score pool sets",
    "top_degree": "top degree", "top_harmonic": "top harmonic", "clustered": "clustered",
    "spread": "spread", "distinguishing": "distinguishing",
    "hub_free_distinguishing": "hub-free distinguishing",
}
SELECTION_STYLES = {
    "degree_tail": dict(color="#8e57b5", marker="^", s=48),
    "score_tail": dict(color="#159b76", marker="D", s=42),
    "top_degree": dict(color="#202020", marker="*", s=150),
    "top_harmonic": dict(color="#2878c8", marker="P", s=100),
    "clustered": dict(color="#bf405c", marker="s", s=85),
    "spread": dict(color="#bd8a18", marker="v", s=90),
    "distinguishing": dict(color="#16a0af", marker="X", s=95),
    "hub_free_distinguishing": dict(color="#6f5c42", marker="h", s=100),
}
METRIC_STYLES = {
    "global_degree": dict(color="#8e57b5", marker="^", s=45),
    "global_harmonic": dict(color="#2878c8", marker="P", s=48),
    "high_cut_degree": dict(color="#ad5d93", marker="d", s=39),
    "high_pair_mass": dict(color="#bf405c", marker="s", s=39),
    "high_uniqueness": dict(color="#16a0af", marker="X", s=43),
    "high_coverage": dict(color="#bd8a18", marker="v", s=45),
    "high_score": dict(color="#159b76", marker="*", s=85),
    "score_bridge": dict(color="#6f5c42", marker="o", s=23),
}
SELECTION_LABELS.update({
    "global_degree": "total vertex degree: exact top 10",
    "global_harmonic": "harmonic: exact top 10",
    "high_cut_degree": "cut degree: 10 best found",
    "high_pair_mass": "pair mass: 10 best found",
    "high_uniqueness": "uniqueness: 10 best found",
    "high_coverage": "coverage: 10 best found",
    "high_score": "frozen score: 10 best found",
    "score_bridge": "intermediate constructed sets",
})


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pair_dir(algorithm: str, model: str, pair_index: int) -> Path:
    return OUTPUT_DIR / algorithm / model / f"pair_{pair_index:03d}"


def _excluded_seeds() -> list[int]:
    """Exclude old fixed pairs, round-1 pairs, pooled training and grid-test pairs."""
    seeds = set(MP.excluded_pair_seeds())
    for path in MP.OUTPUT_DIR.glob("*/*/pair_*/pair.json"):
        seeds.add(int(_read(path)["pair_seed"]))
    grid = Path(__file__).resolve().parent / "data" / "score_transfer" / "C"
    for path in grid.glob("*/*/*/pair_*.csv"):
        seeds.add(int(pd.read_csv(path, usecols=["pair_seed"], nrows=1).iloc[0, 0]))
    return sorted(seeds)


def prepare(sets_per_pair: int, blas_threads: int) -> dict:
    """Freeze the full design; adding pairs or running a subset is safe on resume."""
    path = OUTPUT_DIR / "settings.json"
    saved = _read(path) if path.exists() else None
    design = {
        "version": 1, "sets_per_pair": sets_per_pair, "blas_threads": blas_threads,
        "weights": _read(PW.WEIGHTS_PATH), "features": list(CORE),
        "graph_configs": {f"{a}/{m}": asdict(S.graph_config(a, m)) for a in ALGORITHMS for m in MODELS},
        "percolation": S.PERCOLATION, "sgm": S.SGM, "goat_regularizers": S.GOAT_REGULARIZER,
        "success_thresholds": S.SUCCESS_THRESHOLD, "runs_per_set": 1,
        "pair_seed_base": PAIR_SEED_BASE, "sampling_seed_base": SAMPLING_SEED_BASE,
        "solver_seed_base": SOLVER_SEED_BASE, "uniqueness_cap": MP.UNIQUENESS_CAP,
        "normalization": "mean and sample SD of all uniform sets on the same pair and budget",
        "excluded_pair_seeds": saved["excluded_pair_seeds"] if saved else _excluded_seeds(),
    }
    # Match JSON's representation (e.g. tuples become lists) before comparing.
    design = json.loads(json.dumps(design))
    if saved is not None and saved != design:
        raise ValueError("The saved design differs (weights, set count, parameters or threads). "
                         "Use a new FRESH_PAIR_SCORES_OUTPUT_DIR to keep the results separate.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if saved is None:
        write_json(path, design)
        write_json(OUTPUT_DIR / "environment.json", collect_environment(blas_threads))
    return design


def load_pair(algorithm: str, model: str, pair_index: int, excluded: set[int]):
    family = S.FAMILY_CODES[S.GRAPH_FAMILY[algorithm]]
    start = rc._seed(PAIR_SEED_BASE, family, S.MODEL_CODES[model], pair_index) % (2**32 - 10_000)
    cursor, rejected_total = start, 0
    while True:
        seed, A, B, truth, _, rejected = first_passing_pair(
            lambda value: generate_graph_pair(S.graph_config(algorithm, model), value), cursor)
        rejected_total += int(rejected)
        if int(seed) not in excluded:
            return int(seed), A, B, truth, {"first_seed_tried": start, "rejected_draws": rejected_total}
        cursor = int(seed) + 1


def _complete(task, count: int) -> bool:
    folder = pair_dir(*task)
    info = folder / "pair.json"
    return (info.exists() and _read(info).get("status") == "complete" and
            len(MP._completed_rows(folder / "sets.csv")) == count)


def describe(task) -> str:
    a, m, i = task
    return f"{MP.ALGORITHM_LABELS[a]} {MP.MODEL_LABELS[m]} pair {i}"


def pair_job(task, design: dict) -> str:
    """Compute features before outcomes, then save every single matching run."""
    import fcntl

    a, m, i = task
    folder = pair_dir(*task)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (folder / ".lock").open("a") as lock, thread_limit(design["blas_threads"]):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        count = design["sets_per_pair"]
        if _complete(task, count):
            return f"{describe(task)}: already done"
        seed, A, B, truth, draw = load_pair(a, m, i, set(design["excluded_pair_seeds"]))
        entry = design["weights"][f"{a}/{m}"]
        k = int(entry["budget"])
        rng = np.random.default_rng([SAMPLING_SEED_BASE, seed, k, MP.ALGORITHM_CODES[a]])
        sets = MP.draw_pool(rng, len(A), k, count)
        features = MP.PoolFeatures(GraphA(A), MP.UNIQUENESS_CAP[a]).compute(sets)
        mean = {f: float(features[f].mean()) for f in CORE}
        sd = {f: float(features[f].std(ddof=1)) for f in CORE}
        z = {f: (features[f] - mean[f]) / sd[f] if sd[f] > 0 else np.zeros(count) for f in CORE}
        score = sum(float(entry["weights_per_sd"][f]) * z[f] for f in CORE)
        design_hash = hashlib.sha256(sets.tobytes() + score.tobytes()).hexdigest()
        info = {"status": "running", "algorithm": a, "model": m, "pair_index": i,
                "pair_seed": seed, **draw, "budget": k, "sets_per_pair": count,
                "hash_a": rc._hash(A), "hash_b": rc._hash(B), "design_hash": design_hash,
                "feature_mean": mean, "feature_sd": sd, "weights_per_sd": entry["weights_per_sd"]}
        info_path, rows_path = folder / "pair.json", folder / "sets.csv"
        if info_path.exists():
            saved = _read(info_path)
            if any(saved[key] != info[key] for key in ("hash_a", "hash_b", "design_hash")):
                raise ValueError(f"{describe(task)}: the regenerated graph, seeds or scores differ from saved data.")
        write_json(info_path, info)
        done = MP._completed_rows(rows_path)
        new_file = not rows_path.exists() or rows_path.stat().st_size == 0
        written = 0
        with rows_path.open("a", newline="", encoding="utf-8") as handle:
            writer = None
            for j, seeds in enumerate(sets):
                if j in done:
                    continue
                solver_seed = rc._seed(SOLVER_SEED_BASE, seed, k, MP.ALGORITHM_CODES[a], j)
                out = MP.evaluate(a, m, (A, B, truth, seed), seeds, solver_seed)
                row = {"algorithm": a, "model": m, "pair_index": i, "pair_seed": seed,
                       "budget": k, "candidate_id": j, "group": "random",
                       "seed_vertices": json.dumps(seeds.tolist()),
                       **{f: float(features[f][j]) for f in CORE},
                       **{f"z_{f}": float(z[f][j]) for f in CORE}, "score": float(score[j]),
                       "accuracy": float(out.pop("final_unseeded_accuracy")), **out}
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    if new_file:
                        writer.writeheader()
                writer.writerow(row)
                handle.flush()
                written += 1
        info.update(status="complete", seconds_last_session=time.perf_counter() - started)
        write_json(info_path, info)
    return f"{describe(task)}: {written:,} sets in {(time.perf_counter() - started) / 60:.0f} min"


def tasks(algorithms=ALGORITHMS, models=MODELS, n_pairs=N_PAIRS) -> list:
    return [(a, m, i) for i in range(n_pairs) for a in algorithms for m in models]


def status(algorithms=ALGORITHMS, models=MODELS, n_pairs=N_PAIRS) -> pd.DataFrame:
    rows = []
    for a in algorithms:
        for m in models:
            count = pairs = 0
            for i in range(n_pairs):
                path = pair_dir(a, m, i) / "sets.csv"
                if path.exists():
                    frame = pd.read_csv(path, usecols=["candidate_id", "accuracy"]).dropna()
                    count += frame["candidate_id"].nunique()
                info = pair_dir(a, m, i) / "pair.json"
                pairs += int(info.exists() and _read(info).get("status") == "complete")
            rows.append({"algorithm": a, "model": m, "complete_pairs": pairs, "sets": count})
    return pd.DataFrame(rows)


def run_all(algorithms=ALGORITHMS, models=MODELS, n_pairs=N_PAIRS, sets_per_pair=SETS_PER_PAIR,
            jobs=3, blas_threads=3) -> bool:
    if min(n_pairs, jobs, blas_threads) < 1 or sets_per_pair < 2:
        raise ValueError("Pairs, jobs and threads must be positive; use at least two sets per pair.")
    for a in algorithms:
        for m in models:
            S.check_training_presets(a, m)
    design = prepare(sets_per_pair, blas_threads)
    pending = [t for t in tasks(algorithms, models, n_pairs) if not _complete(t, sets_per_pair)]

    def seconds(task):
        path = pair_dir(*task) / "sets.csv"
        done = len(MP._completed_rows(path)) if path.exists() else 0
        return max(sets_per_pair - done, 0) * SECONDS_PER_SET[task[0]]

    hours = sum(seconds(t) for t in pending) / jobs / 3600
    print(f"{len(pending)} pairs remaining; about {hours:.1f} hours with {jobs} jobs. "
          f"Saving to {OUTPUT_DIR}", flush=True)
    try:
        failures = jobs.run_tasks(pending, pair_job, (design,), describe, seconds, lambda _: [], jobs)
    except KeyboardInterrupt:
        return False
    if failures:
        raise RuntimeError(f"Failed pairs: {', '.join(failures)}. Rerun to retry.")
    return True


def read_sets(complete_only=False) -> pd.DataFrame:
    frames = []
    for path in sorted(OUTPUT_DIR.glob("*/*/pair_*/sets.csv")):
        if complete_only and _read(path.parent / "pair.json").get("status") != "complete":
            continue
        frame = pd.read_csv(path).dropna(subset=["candidate_id", "score", "accuracy"])
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No results yet in {OUTPUT_DIR}")
    return pd.concat(frames, ignore_index=True).drop_duplicates(["algorithm", "model", "pair_index", "candidate_id"])


def selection_plan(graph: GraphA, algorithm: str, info: dict, pool_size=SELECTED_POOL,
                   top_n=SELECTED_TOP) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Select on A alone, using the original 1,000 random sets' means/SDs and frozen weights.

    The degree tail maximises seed-to-nonseed degree, the core feature used by the score.
    These are the top sets in a finite random pool, not global optima. Rules build sets directly.
    """
    seed, k = int(info["pair_seed"]), int(info["budget"])
    rng = np.random.default_rng([SAMPLING_SEED_BASE, seed, k, MP.ALGORITHM_CODES[algorithm], 1])
    pool = MP.draw_pool(rng, graph.n, k, pool_size)
    calculator = MP.PoolFeatures(graph, MP.UNIQUENESS_CAP[algorithm])
    values = calculator.compute(pool)

    def scored(raw):
        z = {f: (raw[f] - info["feature_mean"][f]) / info["feature_sd"][f]
             if info["feature_sd"][f] > 0 else np.zeros(len(raw[f])) for f in CORE}
        return z, sum(info["weights_per_sd"][f] * z[f] for f in CORE)

    _, pool_scores = scored(values)
    arrays, descriptions = [], []
    for group, target in (("degree_tail", values["degree"]), ("score_tail", pool_scores)):
        for rank, j in enumerate(np.argsort(-target, kind="stable")[:top_n], 1):
            arrays.append(pool[j])
            descriptions.append({"group": group, "detail": group, "selection_rank": rank, "pool_index": int(j)})
    for rule in S.STRATEGIES["main"][algorithm]:
        if rule == "random":
            continue
        rule_rng = np.random.default_rng([SAMPLING_SEED_BASE, seed, k, STRATEGY_CODES[rule], 2])
        arrays.append(np.sort(ordering(rule, graph, k, rule_rng,
                                      pool_factor=S.POOL_FACTOR, hub_count=S.HUB_COUNT)))
        descriptions.append({"group": "rule", "detail": rule, "selection_rank": 1, "pool_index": -1})
    raw = calculator.compute(np.stack(arrays))
    z, score = scored(raw)
    table = pd.DataFrame(descriptions)
    table.insert(0, "candidate_id", np.arange(len(table)))
    table["seed_vertices"] = [json.dumps(s.tolist()) for s in arrays]
    for f in CORE:
        table[f], table[f"z_{f}"] = raw[f], z[f]
    table["score"] = score
    return table, arrays


def selected_job(task, design: dict) -> str:
    """Add selected sets on the same saved A/B; preserve all original random runs."""
    a, m, i = task
    folder = pair_dir(*task)
    info = _read(folder / "pair.json")
    if not _complete(task, info["sets_per_pair"]):
        raise RuntimeError(f"{describe(task)}: finish the random collection first.")
    metrics = design.get("kind") == "metrics"
    path = folder / ("metric_sets.csv" if metrics else "selected_sets.csv")
    meta_path = folder / ("metric_selection.json" if metrics else "selection.json")
    if meta_path.exists():
        saved_meta = _read(meta_path)
        if any(saved_meta[key] != value for key, value in design.items()):
            raise ValueError(f"{describe(task)}: selected-set design differs from saved results.")
        expected = (saved_meta["n_sets"] if metrics else
                    2 * design["top_n"] + len(S.STRATEGIES["main"][a]) - 1)
        if len(MP._completed_rows(path)) == expected:
            return f"{describe(task)}: selected sets already done"
    with thread_limit(design["blas_threads"]):
        seed, k = int(info["pair_seed"]), int(info["budget"])
        A, B, truth = generate_graph_pair(S.graph_config(a, m), seed)
        if rc._hash(A) != info["hash_a"] or rc._hash(B) != info["hash_b"]:
            raise RuntimeError(f"{describe(task)}: regenerated graphs differ from the saved pair.")
        original = pd.read_csv(folder / "sets.csv")
        previous = pd.read_csv(folder / "selected_sets.csv") if metrics else pd.DataFrame()
        if metrics:
            from .fresh_pair_extremes import metric_plan
            plan, arrays = metric_plan(GraphA(A), a, info, previous, original, design)
        else:
            plan, arrays = selection_plan(GraphA(A), a, info, design["pool_size"], design["top_n"])
        metadata = {"version": 1, **design, "hash_a": info["hash_a"], "hash_b": info["hash_b"],
                    "normalization": "original random collection; no refit or recentering",
                    "plan_hash": hashlib.sha256(plan.to_csv(index=False).encode()).hexdigest()}
        if metrics:
            metadata["n_sets"] = len(plan)
        if meta_path.exists() and _read(meta_path) != metadata:
            raise ValueError(f"{describe(task)}: selected-set design differs from saved results.")
        write_json(meta_path, metadata)
        MP._completed_rows(path)  # discard an interrupted last row before reading
        saved = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
        done = set(saved["candidate_id"]) if len(saved) else set()
        cache = {tuple(json.loads(row["seed_vertices"])): row for row in
                 pd.concat([original, previous, saved], ignore_index=True).to_dict("records")}
        fields = ("solver_seed", "success", "success_090", "objective", "n_iter", "converged", "runtime_seconds",
                  "h0_accuracy", "matched_fraction", "precision")
        written = 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = None
            for j, seeds in enumerate(arrays):
                if j in done:
                    continue
                key = tuple(int(v) for v in seeds)
                if key in cache:
                    out = {"accuracy": cache[key]["accuracy"], **{f: cache[key][f] for f in fields}}
                    cached = True
                else:
                    offset = 2_000_000 if metrics else 1_000_000
                    solver_seed = rc._seed(SOLVER_SEED_BASE, seed, k, MP.ALGORITHM_CODES[a], offset + j)
                    out = MP.evaluate(a, m, (A, B, truth, seed), seeds, solver_seed)
                    out["accuracy"] = float(out.pop("final_unseeded_accuracy"))
                    cache[key] = out
                    cached = False
                row = {"algorithm": a, "model": m, "pair_index": i, "pair_seed": seed, "budget": k,
                       **plan.iloc[j].to_dict(), **out, "cached": cached}
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    if not len(saved):
                        writer.writeheader()
                writer.writerow(row)
                handle.flush()
                written += 1
    return f"{describe(task)}: {written} selected sets added"


def run_selected(algorithms=ALGORITHMS, models=MODELS, n_pairs=N_PAIRS, jobs=3, blas_threads=3,
                 pool_size=SELECTED_POOL, top_n=SELECTED_TOP) -> bool:
    design = {"pool_size": int(pool_size), "top_n": int(top_n), "blas_threads": int(blas_threads)}
    pending = tasks(algorithms, models, n_pairs)
    failures = jobs.run_tasks(pending, selected_job, (design,), describe,
                            lambda t: (2 * top_n + len(S.STRATEGIES["main"][t[0]]) - 1) * SECONDS_PER_SET[t[0]],
                            lambda _: [], jobs)
    if failures:
        raise RuntimeError(f"Failed selected-set pairs: {', '.join(failures)}")
    return True


def read_selected(prefer_metrics=True) -> pd.DataFrame:
    """Metric extremes when available, otherwise the earlier pool-tail extension."""
    paths = sorted(OUTPUT_DIR.glob("*/*/pair_*/selected_sets.csv"))
    frames = []
    for path in paths:
        metric_path = path.with_name("metric_sets.csv")
        if prefer_metrics and metric_path.exists():
            from .fresh_pair_extremes import plot_sets
            frames.append(plot_sets(pd.read_csv(metric_path)))
        else:
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def scatter_selected(ax, part: pd.DataFrame) -> None:
    """Distinct markers for the two pool tails and each constructed rule."""
    styles = METRIC_STYLES if part["detail"].isin(METRIC_STYLES).any() else SELECTION_STYLES
    for name, style in styles.items():
        selected = part[part["detail"] == name]
        if len(selected):
            ax.scatter(selected["score"], selected["accuracy"], **style, label=SELECTION_LABELS[name],
                       edgecolors="white", linewidths=0.6, alpha=0.95, zorder=4)


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr

    rows = []
    for (a, m, i), part in data.groupby(["algorithm", "model", "pair_index"], sort=False):
        varied = part["score"].nunique() > 1 and part["accuracy"].nunique() > 1
        rho = float(spearmanr(part["score"], part["accuracy"]).statistic) if varied else np.nan
        fifth = max(1, int(np.ceil(len(part) / 5)))
        rows.append({"algorithm": a, "model": m, "pair_index": i, "budget": int(part["budget"].iloc[0]),
                     "sets": len(part), "spearman": rho, "mean_accuracy": part["accuracy"].mean(),
                     "top_fifth_accuracy": part.nlargest(fifth, "score")["accuracy"].mean(),
                     "bottom_fifth_accuracy": part.nsmallest(fifth, "score")["accuracy"].mean()})
    return pd.DataFrame(rows)


def plot_scores(data=None, save=True, show=False, selected=None) -> list[Path]:
    """One figure per setting, one panel per fresh pair; y is actual accuracy."""
    import matplotlib.pyplot as plt

    data = read_sets() if data is None else data
    selected = read_selected() if selected is None else selected
    directory = OUTPUT_DIR / "figures"
    if save:
        directory.mkdir(parents=True, exist_ok=True)
    paths = []
    summary = summarize(data)
    for (a, m), setting in data.groupby(["algorithm", "model"], sort=False):
        groups = list(setting.groupby("pair_index"))
        fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4.8), sharey=True, squeeze=False)
        for ax, (i, part) in zip(axes[0], groups):
            ax.scatter(part["score"], part["accuracy"], s=9, alpha=0.22, rasterized=True, color="#858d96",
                       label="1,000 uniform random sets")
            if not selected.empty:
                extra = selected[(selected.algorithm == a) & (selected.model == m) & (selected.pair_index == i)]
                scatter_selected(ax, extra)
            rho = summary[(summary.algorithm == a) & (summary.model == m) & (summary.pair_index == i)].iloc[0].spearman
            ax.set(title=f"Pair {i} · {len(part):,} random sets · random Spearman {rho:.2f}",
                   xlabel="Frozen weighted score", ylim=(-0.02, 1.02))
            ax.grid(alpha=0.15)
        axes[0, 0].set_ylabel("Unseeded matching accuracy")
        k = int(setting["budget"].iloc[0])
        fig.suptitle(f"{MP.ALGORITHM_LABELS[a]} · {MP.MODEL_LABELS[m]} · n = 600, k = {k}\n"
                     "Previously fitted weights; normalization fixed to the original random sets")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0.16, 1, 1))
        if save:
            for suffix in ("png", "pdf"):
                path = directory / f"{a}_{m}.{suffix}"
                fig.savefig(path, dpi=180, bbox_inches="tight")
                paths.append(path)
        if show:
            plt.show()
        plt.close(fig)
    if save:
        summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    return paths


def compare_models(save=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Old four-feature refit vs frozen pooled model on the SAME external pairs.

    Read existing score-transfer A only. Fit old weights using the old fixed
    pair's 600 random sets, in the same raw-feature coordinates as the pooled
    model. Neither model sees any of A's test outcomes during fitting.
    """
    from sklearn.metrics import roc_auc_score
    from .archive import score_transfer as ST

    data = ST.read_sets(parts=("A",))
    weights, performance, coefficients, pair_rows = PW.load_weights(), [], [], []
    rng = np.random.default_rng(20261103)
    for a in ALGORITHMS:
        for m in MODELS:
            old = MP.single_pair_weights(a, m)
            if old is None:
                raise FileNotFoundError(f"Expected one old fixed-pair collection for {a}/{m}")
            for f in CORE:
                coefficients.append({"algorithm": a, "model": m, "feature": f,
                                     "fixed_pair_weight": old[f], "pooled_weight": weights[(a, m)][f]})
            part = data[(data.algorithm == a) & (data.model == m) & (data.group == "random")].copy()
            part["old_score"] = part[[f"z_{f}" for f in CORE]].to_numpy() @ old.loc[list(CORE)].to_numpy()
            values = []
            for i, pair in part.groupby("pair_index"):
                if pair.success.nunique() < 2:
                    continue
                old_auc = roc_auc_score(pair.success, pair.old_score)
                pooled_auc = roc_auc_score(pair.success, pair.score)
                degree_auc = roc_auc_score(pair.success, pair.z_degree)
                values.append([old_auc, pooled_auc, degree_auc])
                pair_rows.append({"algorithm": a, "model": m, "pair_index": i,
                                  "fixed_pair_auc": old_auc, "pooled_auc": pooled_auc, "degree_auc": degree_auc})
            values = np.asarray(values)
            delta = values[:, 1] - values[:, 0]
            draws = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
            performance.append({"algorithm": a, "model": m, "test_pairs": len(values), "test_sets": len(part),
                                "fixed_pair_auc": values[:, 0].mean(), "pooled_auc": values[:, 1].mean(),
                                "degree_auc": values[:, 2].mean(), "auc_difference": delta.mean(),
                                "difference_low": np.quantile(draws, .025),
                                "difference_high": np.quantile(draws, .975)})
    performance, coefficients = pd.DataFrame(performance), pd.DataFrame(coefficients)
    if save:
        directory = OUTPUT_DIR / "model_comparison"
        directory.mkdir(parents=True, exist_ok=True)
        performance.to_csv(directory / "performance.csv", index=False)
        coefficients.to_csv(directory / "weights.csv", index=False)
        pd.DataFrame(pair_rows).to_csv(directory / "per_pair_auc.csv", index=False)
    return performance, coefficients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=ALGORITHMS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--pairs", type=int, default=N_PAIRS)
    parser.add_argument("--sets-per-pair", type=int, default=SETS_PER_PAIR)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--blas-threads", type=int, default=3)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--compare-models", action="store_true")
    parser.add_argument("--selected", action="store_true", help="add pool tails and round-1 rules on the saved pairs")
    args = parser.parse_args()
    if args.status:
        print(status(args.algorithms, args.models, args.pairs).to_string(index=False))
    elif args.compare_models:
        performance, _ = compare_models()
        print(performance.round(3).to_string(index=False))
    elif args.selected:
        run_selected(args.algorithms, args.models, args.pairs, args.jobs, args.blas_threads)
        plot_scores()
    elif args.plot_only:
        plot_scores()
        print(summarize(read_sets()).round(3).to_string(index=False))
    elif run_all(args.algorithms, args.models, args.pairs, args.sets_per_pair, args.jobs, args.blas_threads):
        plot_scores()
        print(summarize(read_sets()).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
