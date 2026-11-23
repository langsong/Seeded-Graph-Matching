"""B replicates: does the frozen score predict a seed set's success rate over fresh draws of B? (5 Nov 2026)

Why. In fresh_pair_scores.ipynb every seed set was run once on one graph pair (A, B), and almost every run either
recovered the matching or failed, so each dot is a 0 or a 1. A single run mixes three kinds of luck: the seed set
(what a score computed from graph A can see), the particular draw of B (B is a noisy copy of A), and the solver's
random start. Here A and the seed sets stay fixed and B is drawn again from the same model, so each set gets a success
rate instead of a single outcome, and the spread of single runs can be split into the part due to the seed set, the
part due to the draw of B, and the rest.

How. For each of fresh_pair_scores' settings and pairs:
1. Sets. SETS_PER_TENTH sets from each tenth of the frozen score among the pair's 1,000 random sets (100 in all), with
   their saved features and scores. Nothing is refitted.
2. Draws of B. New graphs B given A, from the model that made the pair: a vertex pair of B has an edge with
   probability P + rho (1 - P) where A has an edge and P (1 - rho) where A has none, independently, with P the pair's
   edge probability (p for ER; w_i w_j / sum(w) for Chung-Lu, with the pair's own weights; p for PAPER, whose B also
   always keeps A's preferential-attachment tree, as in the generator). As for the original pairs, B must be connected
   and have no automorphisms, otherwise it is drawn again. Then B's labels are shuffled.
3. Runs. Every chosen set once on each new B, with round 1's solver settings and a new solver seed per run
   (multi_pair_collection.evaluate, as in fresh_pair_scores). The saved run on the original B counts as draw 0.

Every set on a pair meets the same draws of B, enabling paired comparisons; a particular draw can still favour
some seed sets over others. GOAT and FAQ share pairs, so they also share the draws.

Run from the repository root (ds environment):

    python -m selection_rule_comparison.b_replicates
    python -m selection_rule_comparison.b_replicates --algorithms faq
    python -m selection_rule_comparison.b_replicates --status

or from the last section of fresh_pair_scores.ipynb. Rows are written run by run and draw by draw, so a stopped run
resumes where it stopped (Ctrl-C stops the workers too). Results go to data/fresh_pair_scores/b_replicates/, or to
B_REPLICATES_OUTPUT_DIR if it is set.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sgm_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sgm_matplotlib_cache"))

import numpy as np
import pandas as pd

from experiment_common.graph_checks import check_graph_pair
from experiment_common.graphs import generate_graph_pair
from experiment_common.provenance import collect_environment, thread_limit, write_json
from fixed_pair_seed_set_classification import multi_pair_collection as MP

from . import fresh_pair_scores as F
from . import run_comparison as rc
from . import settings as S

OUTPUT_DIR = Path(os.environ.get("B_REPLICATES_OUTPUT_DIR") or F.OUTPUT_DIR / "b_replicates")
ALGORITHMS, MODELS = MP.ALGORITHMS, MP.MODELS
N_PAIRS = 3              # fresh_pair_scores' pairs per setting
SETS_PER_TENTH = 10      # sets taken from each tenth of the frozen score (100 per pair)
DRAWS = 10               # new draws of B per pair; the original B is one more
DRAW_SEED_BASE, SELECT_SEED_BASE, SOLVER_SEED_BASE = 20261160, 20261161, 20261162

LABELS, MODEL_LABELS = MP.ALGORITHM_LABELS, MP.MODEL_LABELS
# validated palette (dataviz): slot 1 blue, slot 2 orange, plus neutrals
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e6e6"


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _spearman(x, y) -> float:
    frame = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(frame["x"].rank().corr(frame["y"].rank()))


# --------------------------------------------------------------------------
# B given A
# --------------------------------------------------------------------------

def paper_tree(config, pair_seed: int) -> np.ndarray:
    """The preferential-attachment tree that A and B share, rebuilt with the generator's own random draws
    (utils.Graphs.gen_PAPER_graphs draws the tree first, right after np.random.seed(pair_seed))."""

    n, alpha, beta = config.vertex_count, config.paper_pa_alpha, config.paper_pa_beta
    state = np.random.RandomState(int(pair_seed))
    tree = np.zeros((n, n), dtype=np.int8)
    degrees = np.zeros(n, dtype=int)
    for vertex in range(1, n):
        if vertex == 1:
            parent = 0
        else:
            weights = alpha + beta * degrees[:vertex]
            parent = state.choice(vertex, p=weights / weights.sum())
        tree[vertex, parent] = tree[parent, vertex] = 1
        degrees[vertex] = 1
        degrees[parent] += 1
    return tree


def _chung_lu_pair(config, pair_seed: int):
    """Regenerate a Chung-Lu pair and keep the edge probabilities the generator sampled it from."""

    import utils.Graphs as generators

    captured = {}
    original = generators.sample_edges_corr

    def keep(P, R, *args, **kwargs):
        captured["P"] = np.array(P, dtype=float)
        return original(P, R, *args, **kwargs)

    generators.sample_edges_corr = keep
    try:
        A, B, truth = generate_graph_pair(config, pair_seed)
    finally:
        generators.sample_edges_corr = original
    return A, B, truth, captured["P"]


def regenerate(algorithm: str, model: str, pair_seed: int):
    """The saved pair, and the probability that each vertex pair of a new B has an edge, given A."""

    config = S.graph_config(algorithm, model)
    tree = None
    if model == "chung_lu":
        A, B, truth, P = _chung_lu_pair(config, pair_seed)
    else:
        A, B, truth = generate_graph_pair(config, pair_seed)
        if model == "sparse_er":
            P = np.full(A.shape, config.er_probability)
        elif model == "paper_pa":
            P = np.full(A.shape, config.paper_pa_probability)
            tree = paper_tree(config, pair_seed)
            aligned = B[np.ix_(truth, truth)]
            if np.any((tree > 0) & ((A == 0) | (aligned == 0))):
                raise RuntimeError(f"PAPER pair {pair_seed}: the rebuilt tree is not in both graphs.")
        else:
            raise ValueError(f"No B model for {model}")
    rho = config.rho
    probability = np.where(A > 0, P + rho * (1 - P), P * (1 - rho))
    if tree is not None:
        probability[tree > 0] = 1.0
    np.fill_diagonal(probability, 0.0)
    return A, B, truth, probability


def draw_b(A: np.ndarray, probability: np.ndarray, pair_seed: int, draw: int, max_tries: int = 200):
    """Draw `draw` of B given A: (B with shuffled labels, the true matching, rejected draws before it)."""

    rng = np.random.default_rng([DRAW_SEED_BASE, int(pair_seed), int(draw)])
    n = len(A)
    rows, columns = np.triu_indices(n, 1)
    upper = probability[rows, columns]
    for rejected in range(max_tries):
        edges = rng.random(len(upper)) < upper
        aligned = np.zeros((n, n))
        aligned[rows[edges], columns[edges]] = 1.0
        aligned += aligned.T
        if check_graph_pair(A, aligned)["passes"]:
            permutation = rng.permutation(n)
            return aligned[np.ix_(permutation, permutation)], np.argsort(permutation), rejected
    raise RuntimeError(f"No connected B without automorphisms in {max_tries} draws (pair {pair_seed}, draw {draw}).")


# --------------------------------------------------------------------------
# Design, sets and one pair (one job)
# --------------------------------------------------------------------------

def task_dir(algorithm: str, model: str, pair_index: int) -> Path:
    return OUTPUT_DIR / algorithm / model / f"pair_{pair_index:03d}"


def describe(task) -> str:
    a, m, i = task
    return f"{LABELS[a]} {MODEL_LABELS[m]} pair {i}"


def prepare(n_draws: int = DRAWS, blas_threads: int = 3) -> dict:
    """The design every job uses. More draws or pairs later are fine; other changes need a new folder."""

    path = OUTPUT_DIR / "settings.json"
    fixed = json.loads(json.dumps({
        "version": 1, "sets_per_tenth": SETS_PER_TENTH, "blas_threads": blas_threads,
        "seed_bases": {"draws": DRAW_SEED_BASE, "selection": SELECT_SEED_BASE, "solver": SOLVER_SEED_BASE},
        "percolation": S.PERCOLATION, "sgm": S.SGM, "goat_regularizers": S.GOAT_REGULARIZER,
        "success_thresholds": S.SUCCESS_THRESHOLD,
    }))
    if path.exists():
        saved = _read_json(path)
        differs = [key for key in fixed if saved.get(key) != fixed[key]]
        if differs:
            raise ValueError(f"The saved design in {path} differs ({', '.join(differs)}). Set "
                             "B_REPLICATES_OUTPUT_DIR to a new folder to keep the results apart.")
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(path, {**fixed, "written_at": _now(), "environment": collect_environment(blas_threads)})
    return {**fixed, "draws": int(n_draws)}


def choose_sets(sets: pd.DataFrame, pair_seed: int, algorithm: str, per_tenth: int = SETS_PER_TENTH) -> pd.DataFrame:
    """per_tenth sets drawn at random from each tenth of the frozen score, sorted by score."""

    sets = sets.dropna(subset=["score"]).drop_duplicates("candidate_id")
    rng = np.random.default_rng([SELECT_SEED_BASE, int(pair_seed), MP.ALGORITHM_CODES[algorithm]])
    tenth = pd.qcut(sets["score"].rank(method="first"), 10, labels=False)
    chosen = []
    for t in range(10):
        ids = np.sort(sets.loc[tenth == t, "candidate_id"].to_numpy())
        chosen += [int(c) for c in rng.choice(ids, size=min(per_tenth, len(ids)), replace=False)]
    return sets[sets["candidate_id"].isin(chosen)].sort_values("score").reset_index(drop=True)


def _completed_ids(path: Path) -> set[int]:
    """run_ids already written; a row cut off by an interruption is removed first."""

    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return set()
    header, body = rows[0], rows[1:]
    complete = [row for row in body if len(row) == len(header)]
    if len(complete) != len(body):
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(complete)
        temporary.replace(path)
    position = header.index("run_id")
    return {int(row[position]) for row in complete}


def _run_id(draw: int, candidate_id: int) -> int:
    return 10_000 * int(draw) + int(candidate_id)


def _fresh(task) -> tuple[dict, pd.DataFrame]:
    folder = F.pair_dir(*task)
    info_path = folder / "pair.json"
    if not info_path.exists() or _read_json(info_path).get("status") != "complete":
        raise FileNotFoundError(f"{describe(task)}: fresh_pair_scores has not finished this pair yet.")
    return _read_json(info_path), pd.read_csv(folder / "sets.csv")


def pair_job(task, design: dict) -> str:
    import fcntl

    a, m, i = task
    folder = task_dir(*task)
    folder.mkdir(parents=True, exist_ok=True)
    runs_path, info_path = folder / "runs.csv", folder / "draws.json"
    started = time.perf_counter()
    with (folder / ".lock").open("a") as lock, thread_limit(design["blas_threads"]):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fresh, sets = _fresh(task)
        seed, k = int(fresh["pair_seed"]), int(fresh["budget"])
        chosen = choose_sets(sets, seed, a, design["sets_per_tenth"])
        ids = [int(c) for c in chosen["candidate_id"]]
        done = _completed_ids(runs_path)
        todo = [r for r in range(1, design["draws"] + 1) if any(_run_id(r, c) not in done for c in ids)]
        if not todo:
            return f"{describe(task)}: already done"
        A, B, truth, probability = regenerate(a, m, seed)
        if rc._hash(A) != fresh["hash_a"] or rc._hash(B) != fresh["hash_b"]:
            raise RuntimeError(f"{describe(task)}: the regenerated pair differs from fresh_pair_scores' saved pair.")
        info = {"algorithm": a, "model": m, "pair_index": i, "pair_seed": seed, "budget": k,
                "hash_a": fresh["hash_a"], "hash_b": fresh["hash_b"], "candidate_ids": ids,
                "mean_degree_a": float(A.sum(axis=1).mean()), "rho": S.graph_config(a, m).rho}
        saved = _read_json(info_path) if info_path.exists() else {}
        for key in ("hash_a", "hash_b", "candidate_ids"):
            if key in saved and saved[key] != info[key]:
                raise RuntimeError(f"{describe(task)}: the saved runs use another pair or other sets ({key} differs). "
                                   "Set B_REPLICATES_OUTPUT_DIR to a new folder to keep them apart.")
        draws = saved.get("draws", {})
        new_file = not runs_path.exists() or runs_path.stat().st_size == 0
        written = 0
        with runs_path.open("a", newline="", encoding="utf-8") as handle:
            writer = None
            for r in todo:
                B_r, truth_r, rejected = draw_b(A, probability, seed, r)
                aligned = B_r[np.ix_(truth_r, truth_r)]
                draws[str(r)] = {"hash_b": rc._hash(B_r), "rejected_draws": int(rejected),
                                 "edges": int(B_r.sum() // 2),
                                 "kept_edge_fraction": float(((A > 0) & (aligned > 0)).sum() / (A > 0).sum())}
                write_json(info_path, {**info, "status": "running", "draws": draws})
                for row in chosen.itertuples(index=False):
                    run_id = _run_id(r, row.candidate_id)
                    if run_id in done:
                        continue
                    seeds = np.asarray(json.loads(row.seed_vertices), dtype=int)
                    solver_seed = rc._seed(SOLVER_SEED_BASE, seed, k, MP.ALGORITHM_CODES[a], int(row.candidate_id), r)
                    out = MP.evaluate(a, m, (A, B_r, truth_r, seed), seeds, solver_seed)
                    record = {"algorithm": a, "model": m, "pair_index": i, "pair_seed": seed, "budget": k,
                              "candidate_id": int(row.candidate_id), "draw": r, "run_id": run_id,
                              "score": float(row.score), "z_degree": float(row.z_degree),
                              "accuracy": float(out.pop("final_unseeded_accuracy")), **out}
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(record))
                        if new_file:
                            writer.writeheader()
                    writer.writerow(record)
                    handle.flush()
                    written += 1
        finished = _completed_ids(runs_path)
        complete = all(_run_id(r, c) in finished for r in range(1, design["draws"] + 1) for c in ids)
        write_json(info_path, {**info, "status": "complete" if complete else "running", "draws": draws,
                               "finished_at": _now()})
    return f"{describe(task)}: {written:,} runs in {(time.perf_counter() - started) / 60:.0f} min"


def tasks(algorithms=ALGORITHMS, models=MODELS, n_pairs: int = N_PAIRS) -> list:
    """Pair 0 of every setting first, then pair 1, and so on."""

    return [(a, m, i) for i in range(n_pairs) for a in algorithms for m in models]


def _left(task, n_draws: int) -> int:
    planned = n_draws * 10 * SETS_PER_TENTH
    path = task_dir(*task) / "runs.csv"
    done = len([r for r in _completed_ids(path) if r // 10_000 <= n_draws]) if path.exists() else 0
    return max(planned - done, 0)


def run_all(algorithms=ALGORITHMS, models=MODELS, n_pairs: int = N_PAIRS, n_draws: int = DRAWS, jobs: int = 3,
            blas_threads: int = 3) -> bool:
    """Run every chosen set on n_draws new draws of B, for each setting and pair. Returns False if stopped."""

    from . import jobs

    design = prepare(n_draws, blas_threads)
    pending = [t for t in tasks(algorithms, models, n_pairs) if _left(t, n_draws) > 0]
    if not pending:
        print(f"All B replicates are done. Results are in {OUTPUT_DIR}")
        return True

    def seconds(task):
        return _left(task, n_draws) * F.SECONDS_PER_SET[task[0]]

    hours = sum(seconds(t) for t in pending) / jobs / 3600
    print(f"{len(pending)} pairs to run ({sum(_left(t, n_draws) for t in pending):,} runs), about {hours:.1f} hours "
          f"with {jobs} jobs (a rough guess). Saving to {OUTPUT_DIR}", flush=True)
    try:
        failures = jobs.run_tasks(pending, pair_job, (design,), describe, seconds, lambda _: [], jobs)
    except KeyboardInterrupt:
        print("Stopped. Finished runs are kept; rerun to resume.", flush=True)
        return False
    if failures:
        raise RuntimeError(f"Failed pairs: {', '.join(failures)}. Rerun to retry (finished runs are kept).")
    print(f"Done. Results are in {OUTPUT_DIR}", flush=True)
    return True


def status(algorithms=ALGORITHMS, models=MODELS, n_pairs: int = N_PAIRS, n_draws: int = DRAWS) -> pd.DataFrame:
    rows = []
    for a, m, i in tasks(algorithms, models, n_pairs):
        planned = n_draws * 10 * SETS_PER_TENTH
        rows.append({"algorithm": a, "model": m, "pair": i, "runs done": planned - _left((a, m, i), n_draws),
                     "runs planned": planned})
    return pd.DataFrame(rows).pivot_table(index=["algorithm", "model"], columns="pair",
                                          values="runs done", aggfunc="first")


# --------------------------------------------------------------------------
# Reading the results
# --------------------------------------------------------------------------

OUTCOME_COLUMNS = ("accuracy", "success", "h0_accuracy")


def read_runs(include_original: bool = True) -> pd.DataFrame:
    """Every run on a new B, plus (draw 0) the chosen sets' saved runs on the original B. Only draws that every chosen
    set of a pair has finished are kept, so each set on a pair has the same draws."""

    frames = []
    for info_path in sorted(OUTPUT_DIR.glob("*/*/pair_*/draws.json")):
        info = _read_json(info_path)
        runs_path = info_path.parent / "runs.csv"
        if not runs_path.exists() or runs_path.stat().st_size == 0:
            continue
        runs = pd.read_csv(runs_path)
        ids = set(info["candidate_ids"])
        counts = runs.groupby("draw")["candidate_id"].nunique()
        runs = runs[runs["draw"].isin(counts.index[counts == len(ids)])]
        if include_original:
            sets = pd.read_csv(F.pair_dir(info["algorithm"], info["model"], info["pair_index"]) / "sets.csv")
            original = sets[sets["candidate_id"].isin(ids)].drop_duplicates("candidate_id")
            original = original[["algorithm", "model", "pair_index", "pair_seed", "budget", "candidate_id", "score",
                                 "z_degree", *OUTCOME_COLUMNS]].assign(draw=0)
            runs = pd.concat([original, runs], ignore_index=True)
        frames.append(runs)
    if not frames:
        raise FileNotFoundError(f"No B replicates yet in {OUTPUT_DIR}")
    data = pd.concat(frames, ignore_index=True)
    data["success"] = data["success"].astype(float)
    return data


def settings_in(data: pd.DataFrame) -> list[tuple[str, str]]:
    present = set(zip(data["algorithm"], data["model"]))
    return [(a, m) for a in ALGORITHMS for m in MODELS if (a, m) in present]


def set_rates(data: pd.DataFrame) -> pd.DataFrame:
    """One row per seed set: its success rate and mean accuracy over the draws of B, and its score."""

    keys = ["algorithm", "model", "pair_index", "candidate_id"]
    rates = data.groupby(keys, sort=False).agg(
        score=("score", "first"), z_degree=("z_degree", "first"), budget=("budget", "first"),
        draws=("success", "count"), success_rate=("success", "mean"), mean_accuracy=("accuracy", "mean")).reset_index()
    original = data[data["draw"] == 0].drop_duplicates(keys).set_index(keys)["success"].rename("success_on_original_b")
    return rates.join(original, on=keys)


def variance_shares(table: np.ndarray) -> dict:
    """Two-way split of single-run outcomes (rows = seed sets, columns = draws of B, one run per cell):
    share of the variance from the seed set, from the draw of B, and the rest (solver luck and set-by-draw effects).
    Moment estimates from the mean squares, set to 0 when negative."""

    y = np.asarray(table, dtype=float)
    n_sets, n_draws = y.shape
    if n_sets < 2 or n_draws < 2:
        return {"set": np.nan, "draw": np.nan, "rest": np.nan, "set_sd": np.nan}
    grand, rows, columns = y.mean(), y.mean(axis=1), y.mean(axis=0)
    ms_set = n_draws * ((rows - grand) ** 2).sum() / (n_sets - 1)
    ms_draw = n_sets * ((columns - grand) ** 2).sum() / (n_draws - 1)
    residual = y - rows[:, None] - columns[None, :] + grand
    ms_rest = (residual ** 2).sum() / ((n_sets - 1) * (n_draws - 1))
    parts = {"set": max((ms_set - ms_rest) / n_draws, 0.0), "draw": max((ms_draw - ms_rest) / n_sets, 0.0),
             "rest": ms_rest}
    total = sum(parts.values())
    if total <= 0:
        return {"set": np.nan, "draw": np.nan, "rest": np.nan, "set_sd": 0.0}
    return {**{key: value / total for key, value in parts.items()}, "set_sd": float(np.sqrt(parts["set"]))}


def summary_table(data: pd.DataFrame, outcome: str = "success") -> pd.DataFrame:
    """Per setting and pair: how much the draw of B matters, and how well the score ranks single runs and rates."""

    fresh = F.read_sets()
    rows = []
    for (a, m, i), part in data.groupby(["algorithm", "model", "pair_index"], sort=False):
        wide = part.pivot_table(index="candidate_id", columns="draw", values=outcome, aggfunc="first").dropna(axis=1)
        rates = part.groupby("candidate_id").agg(score=("score", "first"), z_degree=("z_degree", "first"),
                                                 rate=(outcome, "mean"))
        by_draw = wide.mean(axis=0)
        shares = variance_shares(wide.to_numpy())
        single = fresh[(fresh["algorithm"] == a) & (fresh["model"] == m) & (fresh["pair_index"] == i)]
        original = part[part["draw"] == 0]
        rows.append({
            "algorithm": LABELS[a], "model": MODEL_LABELS[m], "pair": int(i), "budget": int(part["budget"].iloc[0]),
            "sets": len(wide), "draws of B": wide.shape[1],
            "success rate": float(wide.to_numpy().mean()),
            "original B": float(by_draw.get(0, np.nan)),
            "hardest new B": float(by_draw.drop(0, errors="ignore").min()),
            "easiest new B": float(by_draw.drop(0, errors="ignore").max()),
            "share from seed set": shares["set"], "share from draw of B": shares["draw"],
            "share from the rest": shares["rest"], "SD of sets' true rates": shares["set_sd"],
            "Spearman, 1,000 single runs": _spearman(single["score"], single[outcome]),
            "Spearman, these sets' single runs": _spearman(original["score"], original[outcome]),
            "Spearman with success rate": _spearman(rates["score"], rates["rate"]),
            "degree: Spearman with success rate": _spearman(rates["z_degree"], rates["rate"]),
        })
    return pd.DataFrame(rows).set_index(["algorithm", "model", "pair"]).round(3)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def plot_rates(data: pd.DataFrame, algorithm: str, model: str, outcome: str = "success"):
    """One dot per seed set: its success rate over the draws of B (y) against the frozen score (top row) and degree
    alone (bottom row), one column per pair."""

    import matplotlib.pyplot as plt

    part = data[(data["algorithm"] == algorithm) & (data["model"] == model)]
    pairs = sorted(part["pair_index"].unique())
    rates = part.groupby(["pair_index", "candidate_id"]).agg(
        score=("score", "first"), z_degree=("z_degree", "first"), rate=(outcome, "mean"),
        draws=(outcome, "count")).reset_index()
    threshold = S.SUCCESS_THRESHOLD[algorithm]
    draws = int(rates["draws"].median()) if len(rates) else 0
    if outcome == "success":
        label = f"success rate over {draws} draws of B\n(accuracy ≥ {threshold})"
    else:
        label = f"mean accuracy over {draws} draws of B"
    rows = [("score", "frozen weighted score"), ("z_degree", "degree alone (z-score)")]
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5}):
        fig, axes = plt.subplots(len(rows), len(pairs), figsize=(4.8 * len(pairs), 3.5 * len(rows)),
                                 squeeze=False, sharey=True)
        for r, (column, name) in enumerate(rows):
            for c, i in enumerate(pairs):
                ax = axes[r, c]
                sub = rates[rates["pair_index"] == i]
                x, y = sub[column].to_numpy(float), sub["rate"].to_numpy(float)
                ax.scatter(x, y, s=18, color=BLUE, alpha=0.55, linewidths=0, zorder=2)
                ax.set_ylim(-0.03, 1.03)
                ax.grid(color=GRID, linewidth=0.6)
                ax.set_axisbelow(True)
                ax.set_title(f"pair {i} · {len(sub)} sets · Spearman {_spearman(x, y):.2f}", fontsize=10)
                ax.set_xlabel(name)
                if c == 0:
                    ax.set_ylabel(label)
        k = int(part["budget"].iloc[0])
        fig.suptitle(f"{LABELS[algorithm]} · {MODEL_LABELS[model]} · n = 600, k = {k}: one dot per seed set, run on "
                     f"{draws} draws of B given A",
                     fontsize=11.5)
        fig.tight_layout()
    return fig


def plot_draws(data: pd.DataFrame, outcome: str = "success"):
    """Is B the coin? Each dot is one draw of B on one pair: the share of the chosen seed sets that succeed on it.
    The original B is orange."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    present = settings_in(data)
    algorithms = [a for a in ALGORITHMS if any(p[0] == a for p in present)]
    models = [m for m in MODELS if any(p[1] == m for p in present)]
    rng = np.random.default_rng(0)
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5}):
        fig, axes = plt.subplots(len(algorithms), len(models), figsize=(3.9 * len(models), 3.1 * len(algorithms)),
                                 squeeze=False, sharey=True)
        for r, a in enumerate(algorithms):
            for c, m in enumerate(models):
                ax = axes[r, c]
                part = data[(data["algorithm"] == a) & (data["model"] == m)]
                if part.empty:
                    ax.set_visible(False)
                    continue
                pairs = sorted(part["pair_index"].unique())
                shares = []
                for x, i in enumerate(pairs):
                    wide = part[part["pair_index"] == i].pivot_table(
                        index="candidate_id", columns="draw", values=outcome, aggfunc="first").dropna(axis=1)
                    by_draw = wide.mean(axis=0)
                    new = by_draw.drop(0, errors="ignore")
                    ax.scatter(x + rng.uniform(-0.12, 0.12, len(new)), new.to_numpy(), s=22, color=BLUE, alpha=0.7,
                               linewidths=0, zorder=2)
                    if 0 in by_draw.index:
                        ax.scatter([x], [by_draw[0]], s=60, marker="D", color=ORANGE, edgecolors="white",
                                   linewidths=0.8, zorder=3)
                    shares.append(variance_shares(wide.to_numpy())["draw"])
                ax.set_xticks(range(len(pairs)), [f"pair {i}" for i in pairs])
                ax.set_xlim(-0.6, len(pairs) - 0.4)
                ax.set_ylim(-0.03, 1.03)
                ax.grid(axis="y", color=GRID, linewidth=0.6)
                ax.set_axisbelow(True)
                share = np.nanmean(shares) if shares else np.nan
                ax.set_title(f"{LABELS[a]} · {MODEL_LABELS[m]} · draw of B: {share:.0%} of variance", fontsize=10)
                if c == 0:
                    ax.set_ylabel("share of seed sets that succeed")
        handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=BLUE,
                          markersize=6, label="one new draw of B"),
                   Line2D([0], [0], marker="D", color="none", markerfacecolor=ORANGE, markeredgecolor="white",
                          markersize=7, label="the original B")]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.suptitle("How much does the draw of B matter? Success across the same seed sets, one dot per draw of B",
                     y=1.04, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--pairs", type=int, default=N_PAIRS)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--blas-threads", type=int, default=3)
    parser.add_argument("--status", action="store_true", help="only show how far it is")
    args = parser.parse_args(argv)
    if args.status:
        print(status(args.algorithms, args.models, args.pairs, args.draws).to_string())
        return
    if run_all(args.algorithms, args.models, args.pairs, args.draws, args.jobs, args.blas_threads):
        print(summary_table(read_runs()).to_string())


if __name__ == "__main__":
    main()
